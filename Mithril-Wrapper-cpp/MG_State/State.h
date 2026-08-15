// Mithril-Wrapper - MG_State/State.h
// OpenGL 3.3 Core Profile state machine — rewritten for correctness &
// robustness. Deep reference: MobileGL architecture (free_list name allocator,
// per-target binding slots, error queue, thread-local current context), but
// no MobileGL code is copied.
//
// Design principles (see specs/rewrite-gl-state-machine/spec.md):
//  1. NameAllocator  — free_list + valid_bits; O(1) validity + name reuse.
//  2. BindingSlot    — per-target binding with uint16 version for dirty tracking.
//  3. ErrorState     — std::deque<GLenum> queue; glGetError returns earliest.
//  4. thread_local   — per-thread current context; no phantom state.
//  5. Default objects — VAO 0 / FBO 0 / per-target Texture 0 are real & immortal.
//  6. Lifecycle      — Gen* allocates name (lazy create on first Bind);
//                      Delete unbinds + erases + releases name for reuse.
//  7. No dual sources — capabilities are bool fields only (no enabledCaps set);
//                      ELEMENT_ARRAY lives only in the current VAO.
#ifndef MITHRIL_STATE_H
#define MITHRIL_STATE_H

#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <vulkan/vulkan.h>

#include <GL/gl.h>
#include "../MG_Impl/Log.h"

// ---- Standard GL enums absent from the project's minimal glcorearb.h ----
// Guarded so a fuller GL header is harmless.
#ifndef GL_CLIP_DISTANCE0
#define GL_CLIP_DISTANCE0   0x3000
#endif
#ifndef GL_CLIP_DISTANCE1
#define GL_CLIP_DISTANCE1   0x3001
#endif
#ifndef GL_CLIP_DISTANCE2
#define GL_CLIP_DISTANCE2   0x3002
#endif
#ifndef GL_CLIP_DISTANCE3
#define GL_CLIP_DISTANCE3   0x3003
#endif
#ifndef GL_CLIP_DISTANCE4
#define GL_CLIP_DISTANCE4   0x3004
#endif
#ifndef GL_CLIP_DISTANCE5
#define GL_CLIP_DISTANCE5   0x3005
#endif
#ifndef GL_CLIP_DISTANCE6
#define GL_CLIP_DISTANCE6   0x3006
#endif
#ifndef GL_CLIP_DISTANCE7
#define GL_CLIP_DISTANCE7   0x3007
#endif
#ifndef GL_DEPTH_CLAMP
#define GL_DEPTH_CLAMP                  0x864F
#endif
#ifndef GL_DISPATCH_INDIRECT_BUFFER
#define GL_DISPATCH_INDIRECT_BUFFER     0x90EE
#endif
#ifndef GL_INTERLEAVED_ATTRIBS
#define GL_INTERLEAVED_ATTRIBS          0x8C8F
#endif
#ifndef GL_PARAMETER_BUFFER
#define GL_PARAMETER_BUFFER             0x80EE
#endif
#ifndef GL_QUERY_BUFFER
#define GL_QUERY_BUFFER                 0x9192
#endif
#ifndef GL_SAMPLE_MASK
#define GL_SAMPLE_MASK                  0x8E51
#endif
#ifndef GL_SYNC_GPU_COMMANDS_COMPLETE
#define GL_SYNC_GPU_COMMANDS_COMPLETE   0x9117
#endif
#ifndef GL_TEXTURE_CUBE_MAP_ARRAY
#define GL_TEXTURE_CUBE_MAP_ARRAY       0x9009
#endif
// GL uniform type enums absent from the project's minimal glcorearb.h.
#ifndef GL_FLOAT_VEC2
#define GL_FLOAT_VEC2                   0x8B50
#endif
#ifndef GL_FLOAT_VEC3
#define GL_FLOAT_VEC3                   0x8B51
#endif
#ifndef GL_FLOAT_VEC4
#define GL_FLOAT_VEC4                   0x8B52
#endif
#ifndef GL_FLOAT_MAT2
#define GL_FLOAT_MAT2                   0x8B5A
#endif
#ifndef GL_FLOAT_MAT3
#define GL_FLOAT_MAT3                   0x8B5B
#endif
#ifndef GL_FLOAT_MAT4
#define GL_FLOAT_MAT4                   0x8B5C
#endif
#ifndef GL_FLOAT_MAT2x3
#define GL_FLOAT_MAT2x3                 0x8B65
#endif
#ifndef GL_FLOAT_MAT2x4
#define GL_FLOAT_MAT2x4                 0x8B66
#endif
#ifndef GL_FLOAT_MAT3x2
#define GL_FLOAT_MAT3x2                 0x8B67
#endif
#ifndef GL_FLOAT_MAT3x4
#define GL_FLOAT_MAT3x4                 0x8B68
#endif
#ifndef GL_FLOAT_MAT4x2
#define GL_FLOAT_MAT4x2                 0x8B69
#endif
#ifndef GL_FLOAT_MAT4x3
#define GL_FLOAT_MAT4x3                 0x8B6A
#endif
#ifndef GL_SAMPLER_2D
#define GL_SAMPLER_2D                   0x8B5E
#endif

namespace mithril {

// ---- Constants ----
constexpr int kMaxVertexAttribs     = 16;
constexpr int kMaxVertexBindings     = 16;
constexpr int kMaxSampleMaskWords    = 1;   // GL_MAX_SAMPLE_MASK_WORDS
constexpr int kMaxColorAttachments  = 8;
constexpr int kMaxTextureUnits      = 32;
constexpr int kMaxIndexedBindings   = 36;  // >= GL_MAX_UNIFORM_BUFFER_BINDINGS
constexpr int kMaxClipDistances     = 8;
constexpr int kErrorQueueLimit      = 64;

// ---- Strongly-typed target enums ----
enum class BufferTarget : int {
    Array, Uniform, CopyRead, CopyWrite, PixelPack, PixelUnpack,
    TransformFeedback, AtomicCounter, ShaderStorage, DrawIndirect,
    DispatchIndirect, Query, Parameter, Texture,
    Count
    // NOTE: ElementArray is intentionally absent — it lives in the current VAO.
};
constexpr int kBufferTargetCount = static_cast<int>(BufferTarget::Count);

enum class TextureTarget : int {
    _1D, _2D, _3D, CubeMap, Rectangle, _2DMultisample, Buffer,
    _1DArray, _2DArray, CubeMapArray, _2DMultisampleArray,
    Count
};
constexpr int kTextureTargetCount = static_cast<int>(TextureTarget::Count);

// Convert GL enums to strongly-typed enums (returns Count on unknown).
BufferTarget  bufferTargetFromGL(GLenum target) noexcept;
TextureTarget textureTargetFromGL(GLenum target) noexcept;
// Convert back to GL enum.
GLenum bufferTargetToGL(BufferTarget t) noexcept;
GLenum textureTargetToGL(TextureTarget t) noexcept;

// Indexed buffer binding categories.
enum class IndexedBufferTarget : int {
    Uniform = 0, TransformFeedback = 1, AtomicCounter = 2, ShaderStorage = 3,
    Count
};
constexpr int kIndexedBufferCategoryCount = static_cast<int>(IndexedBufferTarget::Count);
IndexedBufferTarget indexedBufferTargetFromGL(GLenum target) noexcept;

// ---- NameAllocator: free_list + valid_bits ----
class NameAllocator {
    GLuint              m_next = 1;       // 0 reserved for default objects
    std::vector<GLuint> m_freeList;       // released names available for reuse
    std::vector<bool>   m_valid;          // name -> valid?  (index = name)
public:
    void generate(GLsizei n, GLuint* out);  // consume freeList first, then m_next++
    bool insert(GLuint id);                 // explicit insert (e.g. default name 0)
    void release(GLuint id);                // mark invalid + push to freeList
    bool valid(GLuint id) const;            // O(1) validity check
    void clear();
};

// ---- BindingSlot: name + version for dirty tracking ----
struct BindingSlot {
    GLuint   name    = 0;
    uint16_t version = 0;
    void bind(GLuint n) {
        if (name == n) return;  // dedup: same binding, no version bump
        name = n;
        ++version;
    }
    bool bound() const { return name != 0; }
};

// Indexed variant (UBO/SSBO/TF/AtomicCounter with offset+size).
struct IndexedBindingSlot {
    GLuint     name              = 0;
    GLintptr   offset            = 0;
    GLsizeiptr size              = 0;
    bool       hasExplicitRange  = false;
    uint16_t   version           = 0;
    void bind(GLuint n) {
        if (name == n && !hasExplicitRange) return;
        name = n; offset = 0; size = 0; hasExplicitRange = false; ++version;
    }
    void bindRange(GLuint n, GLintptr off, GLsizeiptr sz) {
        name = n; offset = off; size = sz; hasExplicitRange = true; ++version;
    }
};

// ---- ErrorState: GL error queue (FIFO) + internal error channel ----
struct ErrorState {
    std::deque<GLenum>     glErrors;
    std::deque<std::string> internalErrors;
    void recordGL(GLenum err);
    GLenum popGL();                       // returns GL_NO_ERROR if empty
    void recordInternal(std::string msg);
    bool hasInternal() const;
    std::string popInternal();
    void clear();
};

// ---- Vertex attribute array state (per VAO) ----
struct VertexAttrib {
    bool      enabled    = false;
    GLint     size       = 4;
    GLenum    type       = GL_FLOAT;
    bool      normalized = false;
    bool      integer    = false;
    GLsizei   stride     = 0;
    const void* pointer  = nullptr;      // offset when a VBO is bound
    GLuint    boundBuffer = 0;           // GL_ARRAY_BUFFER at bind time
    GLuint    divisor    = 0;
    GLuint    bindingIndex = 0;          // vertex buffer binding this attrib reads from
};

// ---- Vertex buffer binding point (per VAO) ----
// GL 4.3 separate-attribute-format model: an attribute names a *binding index*,
// and the instance-step divisor lives on the binding, not on the attribute
// (glVertexAttribDivisor is spec'd as shorthand for setting the divisor of the
// binding the attribute currently points at). Pipeline.cpp reads
// bindings[attribs[loc].bindingIndex].divisor to pick
// VK_VERTEX_INPUT_RATE_INSTANCE — keep that the single source of truth.
struct VertexBinding {
    GLuint   buffer  = 0;   // buffer object bound to this binding point
    GLintptr offset  = 0;   // byte offset of the first element
    GLsizei  stride  = 0;   // byte stride between elements
    GLuint   divisor = 0;   // 0 = per-vertex, >0 = per-instance
};

struct VertexArray {
    GLuint    id = 0;
    VertexAttrib attribs[kMaxVertexAttribs];
    VertexBinding bindings[kMaxVertexBindings];
    GLuint    elementArrayBuffer = 0;    // GL_ELEMENT_ARRAY_BUFFER — sole source
    uint16_t  attribVersions[kMaxVertexAttribs] = {};
    uint32_t  configVersion = 0;
    bool      markedForDeletion = false;

    // GL default state: attribute i reads from vertex buffer binding i. A plain
    // in-class initializer cannot express "index of the array slot", so the
    // identity mapping is established here. Value-init (`VertexArray vao{};`)
    // still routes through this constructor.
    VertexArray() {
        for (int i = 0; i < kMaxVertexAttribs; ++i) {
            attribs[i].bindingIndex =
                (GLuint)(i < kMaxVertexBindings ? i : kMaxVertexBindings - 1);
        }
    }
};

// ---- Buffer ----
struct Buffer {
    GLuint       id = 0;
    GLenum       lastTarget = GL_ARRAY_BUFFER;
    GLsizeiptr   size = 0;
    GLenum       usage = GL_STATIC_DRAW;
    std::vector<uint8_t> data;
    void*        mapped = nullptr;
    GLbitfield   mapAccess = 0;
    GLintptr     mapOffset = 0;
    GLsizeiptr   mapLength = 0;
    uint64_t     contentVersion = 0;
    // GL 4.4 ARB_buffer_storage: immutable storage flags
    bool         immutable = false;          // set by glBufferStorage
    GLbitfield   storageFlags = 0;           // GL_DYNAMIC_STORAGE_BIT etc
};

// ---- Texture (fields completed per GL 3.3 Core) ----
struct Texture {
    GLuint    id = 0;
    GLenum    target = GL_TEXTURE_2D;
    GLint     internalFormat = GL_RGBA8;
    GLsizei   width = 0;
    GLsizei   height = 0;
    GLsizei   depth = 1;
    GLint     levels = 1;
    bool      isCompressed = false;

    // Sampler params (embedded — used when no sampler object is bound)
    GLint     minFilter = GL_NEAREST_MIPMAP_LINEAR;
    GLint     magFilter = GL_LINEAR;
    GLint     wrapS = GL_REPEAT;
    GLint     wrapT = GL_REPEAT;
    GLint     wrapR = GL_REPEAT;
    GLfloat   borderColor[4] = {0, 0, 0, 0};
    bool      generateMipmaps = false;

    // Completed params (P1-4 / spec 4.2)
    GLint     baseLevel = 0;
    GLint     maxLevel = 1000;
    GLfloat   minLod = -1000;
    GLfloat   maxLod = 1000;
    GLfloat   lodBias = 0;
    GLfloat   maxAnisotropy = 1.0f;
    GLenum    compareMode = GL_NONE;
    GLenum    compareFunc = GL_LEQUAL;
    GLenum    swizzleR = GL_RED;
    GLenum    swizzleG = GL_GREEN;
    GLenum    swizzleB = GL_BLUE;
    GLenum    swizzleA = GL_ALPHA;
    GLint     borderColorI[4] = {0, 0, 0, 0};
    GLint     borderColorUI[4] = {0, 0, 0, 0};
    bool      immutable = false;
    GLint     immutableLevels = 0;
    GLsizei   samples = 0;
    bool      fixedSampleLocations = false;

    uint16_t  paramsVersion = 0;
    uint64_t  contentVersion = 0;
};

// ---- Sampler object ----
struct Sampler {
    GLuint    id = 0;
    uint64_t  lifetimeId = 0;          // global monotonic, survives name reuse
    uint16_t  version = 0;
    GLint     minFilter = GL_NEAREST_MIPMAP_LINEAR;
    GLint     magFilter = GL_LINEAR;
    GLint     wrapS = GL_REPEAT;
    GLint     wrapT = GL_REPEAT;
    GLint     wrapR = GL_REPEAT;
    GLfloat   minLod = -1000;
    GLfloat   maxLod = 1000;
    GLfloat   lodBias = 0;
    GLfloat   maxAnisotropy = 1.0f;
    GLenum    compareMode = GL_NONE;
    GLenum    compareFunc = GL_LEQUAL;
    GLfloat   borderColor[4] = {0, 0, 0, 0};
    GLint     borderColorI[4] = {0, 0, 0, 0};
    GLint     borderColorUI[4] = {0, 0, 0, 0};
    bool      markedForDeletion = false;
};

// ---- Renderbuffer ----
struct Renderbuffer {
    GLuint    id = 0;
    GLenum    internalFormat = GL_RGBA8;
    GLsizei   width = 0;
    GLsizei   height = 0;
    GLsizei   samples = 0;
    bool      markedForDeletion = false;
};

// ---- Shader ----
struct Shader {
    GLuint    id = 0;
    GLenum    type = GL_VERTEX_SHADER;
    std::string source;
    bool      compiled = false;
    std::string infoLog;
    std::vector<uint32_t> spirv;
    bool      markedForDeletion = false;
    int       attachCount = 0;
};

// ---- Uniform / Attrib metadata ----
struct Uniform {
    std::string name;
    GLenum      type = 0;
    GLint       size = 0;
    GLint       location = -1;
    GLint       blockIndex = -1;
    GLint       blockBinding = -1;
    GLint       offset = -1;
    GLint       arrayStride = 0;
    GLint       matrixStride = 0;
    bool        rowMajor = false;
    std::vector<float> value;
};

struct Attrib {
    std::string name;
    GLenum      type = 0;
    GLint       size = 0;
    GLint       location = -1;
};

// ---- Uniform / storage block reflection caches ----
// Populated at link time by the descriptor-set reflection path and read back by
// DescriptorSet.cpp when binding UBOs / SSBOs. (These were referenced by
// DescriptorSet.cpp but never committed to the header — added to unblock build.)
struct UniformBlockInfo {
    std::string name;            // reflected block name
    GLuint      bindingPoint = 0;  // GL uniform-block binding point
    uint32_t    dataSize = 0;      // block byte size (0 = unknown)
};

struct StorageBlockInfo {
    GLuint descriptorBinding = 0;  // SSBO descriptor binding (== reflected binding)
};

// ---- Program ----
struct Program {
    GLuint    id = 0;
    std::vector<GLuint> attachedShaders;
    bool      linked = false;
    std::string infoLog;
    std::unordered_map<std::string, Uniform> uniforms;
    std::unordered_map<GLint, std::string> uniformByLocation;
    std::unordered_map<std::string, Attrib> attribs;
    std::unordered_map<std::string, GLuint> uniformBlocks;
    std::unordered_map<std::string, GLuint> attribBindings;
    // SPIR-V for each linked stage (consumed by backend_get_or_create_pipeline).
    // vertexSpirv:         Z remap, NO Y flip — for user-created FBOs.
    // vertexSpirvYFlipped: Z remap + Y flip — for default framebuffer (FBO 0).
    // (Red/black screen fix — must be preserved.)
    std::vector<uint32_t> vertexSpirv;
    std::vector<uint32_t> vertexSpirvYFlipped;
    std::vector<uint32_t> fragmentSpirv;
    // Compute stage (GL_COMPUTE_SHADER). Empty for graphics-only programs;
    // backend_get_or_create_compute_pipeline() bails out on empty.
    std::vector<uint32_t> computeSpirv;
    // Uniform block binding table (glUniformBlockBinding).
    std::unordered_map<GLuint, GLuint> uniformBlockBindings;
    // UBO backing stores keyed by descriptor binding. store_uniform*() writes
    // raw bytes at each uniform's offset; DescriptorSet.cpp memcpys the whole
    // store into the UBO payload at draw time. Sized to the reflected UBO size.
    // 对照 MobileGL DirectVulkan.cpp:171-244 AddBufferVariablesRecursive.
    std::unordered_map<GLuint, std::vector<uint8_t>> uboBackingStore;
    std::unordered_map<GLuint, uint32_t> uboSizes;
    // Sampler descriptor binding -> GL texture unit (set by glUniform1i).
    // Keyed by SPIR-V descriptor binding; value is the GL texture unit the
    // app bound via glUniform1i(samplerLoc, unit). -1 = unset (fallback to
    // binding-as-unit for simple single-texture shaders).
    std::unordered_map<GLuint, GLint> samplerUnitMap;
    // Descriptor-set reflection caches (read by DescriptorSet.cpp; populated at
    // link time). Added to unblock build — referenced but never committed.
    uint32_t    linkVersion = 0;   // bumped on each (re)link; invalidates UBO plans
    std::unordered_map<GLuint, GLuint> blockIndexForDescriptor;  // descriptor binding -> GL block index
    std::vector<UniformBlockInfo>      blockInfos;               // per-GL-block reflection info
    std::unordered_map<GLuint, GLint>  samplerUnitForBinding;   // descriptor binding -> GL texture unit
    std::vector<StorageBlockInfo>      storageBlockInfos;        // per-SSBO reflection info
    std::unordered_map<GLuint, GLuint> storageBlockBindings;     // storage block index -> GL binding point
    // Transform feedback varyings (recorded, backend wiring deferred).
    std::vector<std::string> tfVaryings;
    GLenum tfBufferMode = GL_INTERLEAVED_ATTRIBS;
    bool    markedForDeletion = false;
};

// ---- Framebuffer ----
struct FBOAttachment {
    GLuint texture = 0;
    GLenum textarget = 0;
    GLint  level = 0;
    GLint  layer = 0;
    bool   layered = false;
    GLuint renderbuffer = 0;
};

struct Framebuffer {
    GLuint    id = 0;
    FBOAttachment colors[kMaxColorAttachments];
    FBOAttachment depth;
    FBOAttachment stencil;
    GLenum    drawBuffers[kMaxColorAttachments] = {GL_NONE, GL_NONE, GL_NONE, GL_NONE,
                                                     GL_NONE, GL_NONE, GL_NONE, GL_NONE};
    GLsizei   drawBufferCount = 0;
    GLenum    readBuffer = GL_NONE;
    uint16_t  attachmentVersions[3] = {0, 0, 0};  // color / depth / stencil
    bool      markedForDeletion = false;
};

// ---- Query ----
enum class QueryTarget : int {
    SamplesPassed, AnySamplesPassed, PrimitivesGenerated,
    TimeElapsed, Timestamp, Count
};

struct Query {
    GLuint       id = 0;
    QueryTarget  target = QueryTarget::Count;
    bool         active = false;
    bool         ended = false;
    bool         resultCached = false;
    uint64_t     cachedResult = 0;
    bool         markedForDeletion = false;
};

// ---- Sync ----
struct Sync {
    void*       handle = nullptr;
    GLenum      condition = GL_SYNC_GPU_COMMANDS_COMPLETE;
    GLbitfield  flags = 0;
    bool        signaled = false;
    bool        markedForDeletion = false;
};

// ---- Transform Feedback ----
struct TransformFeedback {
    GLuint    id = 0;
    bool      active = false;
    bool      paused = false;
    GLenum    primitiveMode = GL_POINTS;
    bool      markedForDeletion = false;
};

// ---- Pixel store state (pack + unpack) ----
struct PixelStoreState {
    // Unpack
    GLint   unpackAlignment     = 4;
    GLint   unpackRowLength     = 0;
    GLint   unpackImageHeight   = 0;
    GLint   unpackSkipRows      = 0;
    GLint   unpackSkipPixels    = 0;
    GLint   unpackSkipImages    = 0;
    bool    unpackSwapBytes     = false;
    bool    unpackLSBFirst      = false;
    // Pack
    GLint   packAlignment       = 4;
    GLint   packRowLength       = 0;
    GLint   packImageHeight     = 0;
    GLint   packSkipRows        = 0;
    GLint   packSkipPixels      = 0;
    GLint   packSkipImages      = 0;
    bool    packSwapBytes       = false;
    bool    packLSBFirst        = false;
};

// ---- Per-draw-buffer blend state ----
struct BlendState {
    bool    enabled = false;
    GLenum  srcRGB = GL_ONE, dstRGB = GL_ZERO;
    GLenum  srcA   = GL_ONE, dstA   = GL_ZERO;
    GLenum  eqRGB  = GL_FUNC_ADD, eqA = GL_FUNC_ADD;
};

// ---- Proxy texture state ----
struct ProxyTextureState {
    GLint width = 0;
    GLint height = 0;
    GLint depth = 0;
    GLint internalFormat = 0;
    bool  valid = false;
};

// =========================================================================
// GLState — top-level state machine
//
// Object tables remain as unordered_map (accessible via state_get_* helpers).
// Binding tracking uses per-target BindingSlot arrays.
// Render state (capabilities/blend/depth/stencil/rasterizer/pixelstore/clear/
// viewport/scissor) is kept as flat fields — these work correctly and are
// accessed by many files including the Vulkan backend.
// =========================================================================
struct GLState {
    bool initialized = false;

    // ---- Name allocators (one per object type) ----
    NameAllocator bufferNames, textureNames, samplerNames, vaoNames,
                  fboNames, programNames, shaderNames, renderbufferNames,
                  queryNames, tfNames;

    // ---- Object tables ----
    std::unordered_map<GLuint, VertexArray>      vaos;
    std::unordered_map<GLuint, Buffer>           buffers;
    std::unordered_map<GLuint, Texture>          textures;
    std::unordered_map<GLuint, Shader>           shaders;
    std::unordered_map<GLuint, Program>          programs;
    std::unordered_map<GLuint, Framebuffer>      framebuffers;
    std::unordered_map<GLuint, Sampler>          samplers;
    std::unordered_map<GLuint, Renderbuffer>     renderbuffers;
    std::unordered_map<GLuint, Query>            queries;
    std::unordered_map<void*, Sync>              syncObjects;   // key = handle
    std::unordered_map<GLuint, TransformFeedback> transformFeedbacks;

    // ---- Buffer bindings (per-target, non-indexed) ----
    // Indexed by BufferTarget enum. ElementArray is NOT here (lives in VAO).
    BindingSlot bufferBindings[kBufferTargetCount];
    // ---- Indexed buffer bindings (UBO/TF/AtomicCounter/SSBO) ----
    IndexedBindingSlot indexedBufferBindings[kIndexedBufferCategoryCount][kMaxIndexedBindings];
    int touchedIndexed[kIndexedBufferCategoryCount] = {};

    // ---- Texture bindings (per-unit per-target) ----
    BindingSlot textureBindings[kMaxTextureUnits][kTextureTargetCount];
    GLuint      samplerBindings[kMaxTextureUnits] = {};  // sampler object name per unit
    GLuint      imageTextureUnits[kMaxTextureUnits] = {}; // texture object bound to each unit (for sampler descriptors)
    Texture     defaultTextures[kTextureTargetCount];    // name=0, immortal, one per target
    int         activeTextureUnit = 0;                   // GL_TEXTURE0 relative
    uint64_t    textureBindGeneration = 0;
    int         maxTouchedTextureUnit = 0;

    // ---- Current object bindings (flat GLuint for convenience) ----
    GLuint      currentVAO = 0;        // VAO 0 = default
    GLuint      currentDrawFBO = 0;    // FBO 0 = default (EGL surface)
    GLuint      currentReadFBO = 0;
    GLuint      currentProgram = 0;
    GLuint      currentRenderbuffer = 0;
    GLuint      currentTransformFeedback = 0;  // TF 0 = default

    // ---- EGL-backed default framebuffer ----
    VkImageView eglDefaultColor = VK_NULL_HANDLE;
    VkImageView eglDefaultDepth = VK_NULL_HANDLE;
    VkImage     eglDefaultColorImage = VK_NULL_HANDLE;
    VkImage     eglDefaultDepthImage = VK_NULL_HANDLE;
    VkFormat    eglDefaultColorFormat = VK_FORMAT_UNDEFINED;
    VkFormat    eglDefaultDepthFormat = VK_FORMAT_UNDEFINED;
    int         eglDefaultWidth  = 0;
    int         eglDefaultHeight = 0;

    // ---- Clear values ----
    float   clearColor[4] = {0, 0, 0, 0};
    double  clearDepth = 1.0;
    GLint   clearStencil = 0;

    // ---- Depth state ----
    bool    depthTest = false;
    bool    depthMask = true;
    GLenum  depthFunc = GL_LESS;
    double  depthNear = 0.0;
    double  depthFar = 1.0;

    // ---- Blend state (per-draw-buffer) ----
    BlendState blends[kMaxColorAttachments];
    float   blendColor[4] = {0, 0, 0, 0};

    // ---- Color mask (per-draw-buffer) ----
    bool    colorMask[kMaxColorAttachments][4] = {
        {true, true, true, true}, {true, true, true, true},
        {true, true, true, true}, {true, true, true, true},
        {true, true, true, true}, {true, true, true, true},
        {true, true, true, true}, {true, true, true, true}
    };

    // ---- Stencil state (front + back) ----
    bool    stencilTest = false;
    GLuint  stencilMask = ~0u;
    GLuint  stencilBackMask = ~0u;
    GLenum  stencilFunc = GL_ALWAYS, stencilBackFunc = GL_ALWAYS;
    GLint   stencilRef = 0, stencilBackRef = 0;
    GLuint  stencilValueMask = ~0u, stencilBackValueMask = ~0u;
    GLenum  stencilSfail = GL_KEEP, stencilDpfail = GL_KEEP, stencilDppass = GL_KEEP;
    GLenum  stencilBackSfail = GL_KEEP, stencilBackDpfail = GL_KEEP, stencilBackDppass = GL_KEEP;

    // ---- Rasterizer ----
    bool    cullFace = false;
    GLenum  cullMode = GL_BACK;
    GLenum  frontFace = GL_CCW;
    GLenum  polygonModeFront = GL_FILL;
    GLenum  polygonModeBack = GL_FILL;
    float   polygonOffsetFactor = 0.0f;
    float   polygonOffsetUnits = 0.0f;
    float   polygonOffsetClamp = 0.0f;  // GL 4.6 ARB_polygon_offset_clamp
    bool    polygonOffsetFill = false;
    float   lineWidth = 1.0f;
    float   pointSize = 1.0f;

    // ---- Capabilities (single source of truth — no enabledCaps set) ----
    bool    scissorTest = false;
    bool    dither = true;              // P1-8: default enabled
    bool    multisample = true;         // P1-8: default enabled
    bool    sampleAlphaToCoverage = false;
    bool    sampleCoverage = false;
    bool    sampleMask = false;
    GLfloat sampleCoverageValue = 1.0f;
    bool    sampleCoverageInvert = false;
    // GL 4.0 ARB_sample_shading (GL_SAMPLE_SHADING / glMinSampleShading).
    bool    sampleShadingEnabled = false;
    GLfloat minSampleShading = 0.0f;
    // GL 4.0 glSampleMaski. GL_MAX_SAMPLE_MASK_WORDS is 1 here, which also
    // matches Vulkan's ceil(rasterizationSamples/32) == 1 for <= 32 samples.
    // Pipeline.cpp hands &sampleMaskValue[0] to VkPipelineMultisampleStateCreateInfo,
    // so this must live as long as the GLState — it does. Default = all bits set
    // (no samples masked off), which is the GL default.
    GLuint  sampleMaskValue[kMaxSampleMaskWords] = { 0xFFFFFFFFu };
    bool    programPointSize = false;
    bool    primitiveRestart = false;
    GLuint  primitiveRestartIndex = 0;
    // FIX (根因 AF - Primitive Restart): GL_PRIMITIVE_RESTART_FIXED_INDEX 状态。
    // 与 primitiveRestart（GL_PRIMITIVE_RESTART）独立维护；Pipeline.cpp 的
    // ia.primitiveRestartEnable 在两者任一启用时为 VK_TRUE。
    // 深度对照 MobileGL VulkanRenderer.cpp:3861-3877。
    bool    primitiveRestartFixedIndex = false;
    bool    framebufferSRGB = false;
    bool    depthClamp = false;
    bool    rasterizerDiscard = false;
    bool    textureCubeMapSeamless = false;
    bool    clipDistance[kMaxClipDistances] = {};

    // GL 4.5 ARB_clip_control: clip volume origin and depth mode.
    // GL_LOWER_LEFT + GL_NEGATIVE_ONE_TO_ONE = traditional GL (default).
    // GL_UPPER_LEFT + GL_ZERO_TO_ONE = Vulkan/Metal native convention.
    GLenum  clipOrigin = 0x8CA1;      // GL_LOWER_LEFT
    GLenum  clipDepthMode = 0x935E;   // GL_NEGATIVE_ONE_TO_ONE

    // ---- Viewport / scissor ----
    GLint   viewportX = 0, viewportY = 0;
    GLsizei viewportW = 0, viewportH = 0;
    GLint   scissorX = 0, scissorY = 0;
    GLsizei scissorW = 0, scissorH = 0;

    // FIX (根因 AG - BaseVertex / BaseInstance): draw-time base offsets.
    // 由 glDrawElementsBaseVertex / glDrawElementsInstancedBaseVertex 设置
    // currentBaseVertex；由 glDrawElementsInstancedBaseInstance /
    // glDrawArraysInstancedBaseInstance 设置 currentBaseInstance。
    // backend_draw_indexed / backend_draw_indexed_instanced /
    // backend_draw / backend_draw_instanced 从 g_state 读取后传给
    // vkCmdDrawIndexed 的 vertexOffset / firstInstance 与 vkCmdDraw 的
    // firstInstance。每次 draw 完成后由 Drawing.cpp 重置为 0，避免泄漏到
    // 下一个 draw（无 BaseVertex 的 draw 应保持 vertexOffset=0）。
    // 深度对照 MobileGL drawParams.baseVertex / drawParams.baseInstance。
    int32_t  currentBaseVertex = 0;
    uint32_t currentBaseInstance = 0;

    // ---- Pixel store ----
    PixelStoreState pixelStore;

    // ---- Error state (queue, not single slot) ----
    ErrorState errors;

    // ---- Proxy texture ----
    ProxyTextureState proxyTexture2D;

    // ---- Sync handle allocator ----
    void* nextSyncHandle = (void*)0x10;  // monotonic, avoids sentinel 0x1

    // ---- Sampler lifetime ID allocator ----
    uint64_t nextSamplerLifetimeId = 1;

    // ---- Capability dispatch helpers ----
    void setCapability(GLenum cap, bool enabled);
    bool isCapabilityEnabled(GLenum cap) const;
    // True iff `cap` is one of the capabilities this GLState actually tracks.
    // glEnable/glDisable use this to raise GL_INVALID_ENUM for unrecognised
    // caps per the GL spec, instead of silently ignoring them.
    bool isKnownCapability(GLenum cap) const;

    // ---- Texture query helper (for backend DescriptorSet.cpp) ----
    // Returns the first non-zero texture name bound to `unit` across all targets.
    GLuint boundTextureForUnit(GLuint unit) const;
    // Returns the texture name bound to `unit` for a specific target.
    GLuint boundTextureForUnit(GLuint unit, TextureTarget target) const {
        if (unit >= kMaxTextureUnits) return 0;
        return textureBindings[unit][static_cast<int>(target)].name;
    }

    // ---- Render state version (for future backend dirty tracking) ----
    uint16_t renderVersion = 0;
    void bumpRenderVersion() { ++renderVersion; }

    // ---- First-frame diagnostic (B1) ----
    // 已呈现帧计数，供首帧诊断聚合使用（仅前 N 帧输出，避免刷屏）。
    // 由 CommandStream.cpp 的 present 路径自增；Drawing.cpp 读取。
    uint32_t presentedFrames = 0;
};

// ---- Thread-local current context pointer ----
extern thread_local GLState* g_state;

// ---- EGL initialized flag (set by eglInitialize) ----
extern bool g_eglInitialized;

// ---- State lifecycle ----
bool     state_init();
GLState* state_create();
void     state_destroy(GLState* s);

// ---- Object accessors (signatures unchanged for backward compat) ----
VertexArray*  state_get_vao(GLuint id);
Buffer*       state_get_buffer(GLuint id);
Texture*      state_get_texture(GLuint id);
Shader*       state_get_shader(GLuint id);
Program*      state_get_program(GLuint id);
Framebuffer*  state_get_framebuffer(GLuint id);
Sampler*      state_get_sampler(GLuint id);
Renderbuffer* state_get_renderbuffer(GLuint id);
TransformFeedback* state_get_transform_feedback(GLuint id);
Query*        state_get_query(GLuint id);

// ---- Error helpers ----
void   state_set_error(GLenum err);
GLenum state_take_error();

// ---- Name allocation (unified via NameAllocator) ----
void state_gen_names(const char* kind, GLsizei n, GLuint* out);

// ---- Capability check helper (for glIsEnabled) ----
bool state_is_capability_enabled(GLenum cap);

} // namespace mithril

#endif // MITHRIL_STATE_H
