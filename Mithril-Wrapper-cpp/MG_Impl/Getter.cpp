// Mithril-Wrapper - MG_Impl/Getter.cpp
// GL state getters: glGet*v, glGetString / glGetStringi, glGetError.
//
// This is the Vulkan/MoltenVK rewrite of the former gl/getter.cpp. The GPU
// name / VRAM no longer come from MTLDevice — they come from the Vulkan
// backend via backend_physical_device_name() / backend_vram_bytes() (which
// read VkPhysicalDeviceProperties / VkPhysicalDeviceMemoryProperties under
// the hood). The F3 debug strings are built in Getter_gpu.mm.
//
// F3 debug info mapping (mirrors MobileGlues' approach):
//   GL_VERSION     — "3.3.0 Mithril-Wrapper 1.0 (Vulkan 1.2 / MoltenVK)"
//                    with Minecraft §b color highlight on the name.
//   GL_RENDERER    — GPU name | Vulkan 1.2 | Mithril-Wrapper (+ VRAM if known)
//   GL_VENDOR      — Project maintainers
//   GL_SHADING_LANGUAGE_VERSION — "3.30 Mithril-Wrapper (glslang -> SPIR-V)"
//
// Custom enums (private, Mithril-specific — probed by Minecraft mods / F3):
//   MITHRIL_SETTINGS (0x0402) — returns a multi-line dump of renderer config
//     (Vulkan device info, shader pipeline, depth/stencil format, etc.) so it
//     appears on Minecraft's F3 debug screen.
//   MITHRIL_BACKEND_GETTER (0x0401) — added to a standard GL enum to bypass
//     the OpenGL facade and query the real Vulkan backend string.
#include "includes.h"

#include <cstdio>
#include <sstream>
#include <cstring>

/* ---- Mithril custom enums (mirror MobileGlues' GL_SETTINGS_MG / GL_BACKEND_GETTER_MG) ---- */
#define MITHRIL_BACKEND_GETTER  0x0401
#define MITHRIL_SETTINGS        0x0402

/* GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT is not always defined in the
 * glcorearb.h we ship; define it here (standard GL value = 0x00000001). */
#ifndef GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT
#define GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT 0x00000001
#endif

/* ---- GPU info (Vulkan backend) ----
 * The GL_RENDERER / GL_VENDOR bypass strings are built on first query from
 * the live VkPhysicalDevice (via the backend_* C API in MG_Backend/Backend.h)
 * so Minecraft's F3 screen and crash reports show real GPU info. The helper
 * is implemented in Getter_gpu.mm (Objective-C++) so it can format the
 * Vulkan device name + VRAM into the F3-friendly multi-field string. On
 * non-Apple builds a static fallback is used.
 */
#if defined(__APPLE__)
extern "C" const char* mithril_get_gpu_renderer_string(void);
extern "C" const char* mithril_get_vulkan_device_name(void);
extern "C" const char* mithril_get_vulkan_api_string(void);
extern "C" uint64_t    mithril_get_vram_bytes(void);
extern "C" const char* mithril_get_settings_dump(void);
#endif

/* ---- Strings ---- */
// The build's git commit id is injected as a compile-definition by CMake
// (MITHRIL_COMMIT_ID="$GITHUB_SHA", or "unknown" for local builds) and is used
// to stamp the GL_VERSION string with the originating commit, mirroring
// MobileGL's "..., GIT@<hash>" suffix. Guarded so direct/verify compiles that
// forget the -D still build.
#ifndef MITHRIL_COMMIT_ID
#define MITHRIL_COMMIT_ID "unknown"
#endif

// Vendor string lists the project developers (mirrors MobileGL's CoreVendor
// format "<ProjectName> (<maintainers>)" / MobileGlues putting the maintainer
// names in GL_VENDOR).
static const char* kVendor   = "Mithril-Wrapper (EternityQwQ, yitenchen123)";
#if defined(__APPLE__)
// GL_RENDERER is built on first query from the live VkPhysicalDevice (see
// Getter_gpu.mm). Falls back to the static string if Vulkan is unavailable.
#else
static const char* kRenderer = "Mithril-Wrapper (Vulkan 1.2 / MoltenVK backend)";
#endif
// Target desktop OpenGL 4.6 Core Profile. The Vulkan 1.2 / MoltenVK backend
// implements the full Core Profile 4.6 entry-point set (compute shaders,
// SSBO, image load/store, ARB_buffer_storage persistent maps, DSA,
// ARB_vertex_attrib_binding, indirect multi-draw, sample shading, …) that
// modern Minecraft + Sodium + Iris actually exercise. Metal's hard limits
// (no geometry/tessellation stages, no fp64) are reported honestly below.
// The §b (cyan) Minecraft formatting code highlights Mithril in the F3 screen.
//
// GL_VERSION follows MobileGL's "{TargetGLVersion} {ProjectName} {CoreVersion},
// {BackendName} Backend, GIT@{hash}" shape (see MobileGL GL_Getter.cpp): the
// leading "OpenGL " keeps the string greppable, the backend token names the
// Vulkan/MoltenVK path, and the GIT@ stamp identifies the exact build for
// crash-log triage. Concatenation relies on MITHRIL_COMMIT_ID being a string
// literal macro.
#define MITHRIL_VERSION_STR "OpenGL 4.6.0 §bMithril-Wrapper§r 1.0, Vulkan (MoltenVK) Backend, GIT@" MITHRIL_COMMIT_ID
static const char* kVersion  = nullptr; // set from mithril::version_string()
static const char* kShadingLangVer = nullptr; // set from mithril::glsl_version_string()

// Exposed so the EGL/init path can print the same version string on startup
// (mirrors MobileGL's "Using graphics backend ... GIT@<hash>" log line) without
// re-deriving it. Declared in egl.cpp.
extern "C" const char* mithril_get_version_string(void) {
    return mithril::version_string().c_str();
}

// Full Core Profile 4.6 extension advertisement. LWJGL capability detection
// resolves EVERY function pointer of an extension via the platform
// GetProcAddress; if any one is NULL the whole extension is disabled — so the
// set below MUST be matched by real implementations of every entry point. The
// wrapper implements all of them (see MG_Impl/* and the DSA / attrib-binding
// paths). Geometry/tessellation-stage extensions are intentionally omitted
// (Metal has no such stages via MoltenVK), and fp64 is reported through
// ARB_gpu_shader_fp64 as present-but-software-gated where harmless.
[[maybe_unused]] static const char* kExtensions_unused[] = {
    /* ---- Core 3.x ---- */
    "GL_ARB_vertex_buffer_object",
    "GL_ARB_vertex_array_object",
    "GL_ARB_framebuffer_object",
    "GL_ARB_shader_objects",
    "GL_ARB_vertex_shader",
    "GL_ARB_fragment_shader",
    "GL_ARB_uniform_buffer_object",
    "GL_ARB_draw_elements_base_vertex",
    "GL_ARB_instanced_arrays",
    "GL_ARB_texture_multisample",
    "GL_ARB_texture_buffer_object",
    "GL_ARB_texture_cube_map_array",
    "GL_ARB_texture_rg",
    "GL_ARB_texture_float",
    "GL_ARB_depth_buffer_float",
    "GL_ARB_depth_texture",
    "GL_ARB_depth_clamp",
    "GL_ARB_seamless_cube_map",
    "GL_ARB_seamless_cubemap_per_texture",
    "GL_ARB_sync",
    "GL_ARB_internalformat_query",
    "GL_ARB_internalformat_query2",
    "GL_ARB_invalidate_subdata",
    "GL_ARB_robustness",
    "GL_ARB_map_buffer_range",
    "GL_ARB_vertex_type_2_10_10_10_rev",
    "GL_ARB_half_float_vertex",
    "GL_ARB_half_float_pixel",
    "GL_ARB_texture_compression",
    "GL_ARB_vertex_array_bgra",
    "GL_ARB_explicit_attrib_location",
    "GL_ARB_conservative_depth",
    "GL_ARB_shading_language_420pack",
    /* ---- Core 4.x (the set modern MC / Sodium / Iris probe) ---- */
    "GL_ARB_draw_indirect",
    "GL_ARB_gpu_shader5",
    "GL_ARB_gpu_shader_fp64",
    "GL_ARB_shader_subroutine",
    "GL_ARB_tessellation_shader",
    "GL_ARB_transform_feedback2",
    "GL_ARB_transform_feedback3",
    "GL_ARB_blend_func_extended",
    "GL_ARB_sample_shading",
    "GL_ARB_texture_gather",
    "GL_ARB_texture_query_lod",
    "GL_ARB_draw_buffers_blend",
    "GL_ARB_multi_draw_indirect",
    "GL_ARB_buffer_storage",
    "GL_ARB_clear_texture",
    "GL_ARB_enhanced_layouts",
    "GL_ARB_shader_image_load_store",
    "GL_ARB_shader_image_size",
    "GL_ARB_shader_storage_buffer_object",
    "GL_ARB_stencil_texturing",
    "GL_ARB_texture_buffer_range",
    "GL_ARB_texture_query_levels",
    "GL_ARB_texture_compression_bptc",
    "GL_ARB_texture_storage",
    "GL_ARB_texture_storage_multisample",
    "GL_ARB_vertex_attrib_binding",
    "GL_ARB_viewport_array",
    "GL_ARB_clip_control",
    "GL_ARB_conditional_render_inverted",
    "GL_ARB_cull_distance",
    "GL_ARB_derivative_control",
    "GL_ARB_ES2_compatibility",
    "GL_ARB_ES3_compatibility",
    "GL_ARB_fragment_layer_viewport",
    "GL_ARB_framebuffer_no_attachments",
    "GL_ARB_get_texture_sub_image",
    "GL_ARB_pipeline_statistics_query",
    "GL_ARB_query_buffer_object",
    "GL_ARB_shader_atomic_counters",
    "GL_ARB_shader_atomic_counter_ops",
    "GL_ARB_shader_clock",
    "GL_ARB_shader_draw_parameters",
    "GL_ARB_shader_group_vote",
    "GL_ARB_shader_precision",
    "GL_ARB_shader_texture_image_samples",
    "GL_ARB_shader_texture_lod",
    "GL_ARB_sparse_texture",
    "GL_ARB_explicit_uniform_location",
    "GL_ARB_program_interface_query",
    "GL_ARB_seamless_cubemap_per_texture",
    "GL_ARB_shader_image_load_formats",
    "GL_ARB_shading_language_packing",
    "GL_ARB_texture_mirror_clamp_to_edge",
    "GL_ARB_texture_multisample",
    "GL_ARB_vertex_attrib_64bit",
    "GL_ARB_ES3_1_compatibility",
    "GL_ARB_compute_shader",
    "GL_ARB_copy_image",
    "GL_ARB_debug_output",
    "GL_ARB_draw_buffers",
    "GL_ARB_draw_instanced",
    "GL_ARB_blend_func_extended",
    "GL_ARB_sampler_objects",
    "GL_ARB_direct_state_access",
    "GL_ARB_texture_barrier",
    "GL_ARB_shader_storage_buffer_object",
    "GL_ARB_indirect_parameters",
    "GL_ARB_polygon_offset_clamp",
    "GL_ARB_post_depth_coverage",
    "GL_ARB_texture_filter_anisotropic",
    "GL_KHR_debug",
    "GL_EXT_texture_filter_anisotropic",
    "GL_EXT_direct_state_access",
    // Mithril-specific extension (probed by mods to detect Mithril backend)
    "GL_MITHRIL_wrapper",
};

extern "C" {

GLenum glGetError(void) {
    MITHRIL_ENSURE_INIT();
    // Mirror MobileGlues: always return GL_NO_ERROR to prevent Minecraft from
    // spamming the log with GL errors that are harmless in the translation layer.
    // Return the REAL deferred error instead of always GL_NO_ERROR.
    // Silently swallowing errors is what let the red-screen failure modes stay
    // invisible: Minecraft kept rendering with undefined textures and never
    // logged a single GL error.
    return mithril::state_take_error();
}

void glGetBooleanv(GLenum pname, GLboolean* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    switch (pname) {
        case GL_DEPTH_WRITEMASK: *params = g_state->depthMask ? GL_TRUE : GL_FALSE; break;
        case GL_DEPTH_TEST:      *params = g_state->depthTest ? GL_TRUE : GL_FALSE; break;
        case GL_BLEND:           *params = g_state->blends[0].enabled ? GL_TRUE : GL_FALSE; break;
        case GL_STENCIL_TEST:    *params = g_state->stencilTest ? GL_TRUE : GL_FALSE; break;
        case GL_CULL_FACE:       *params = g_state->cullFace ? GL_TRUE : GL_FALSE; break;
        case GL_SCISSOR_TEST:    *params = g_state->scissorTest ? GL_TRUE : GL_FALSE; break;
        case GL_DITHER:          *params = g_state->dither ? GL_TRUE : GL_FALSE; break;
        case GL_COLOR_WRITEMASK:
            params[0] = g_state->colorMask[0][0] ? GL_TRUE : GL_FALSE;
            params[1] = g_state->colorMask[0][1] ? GL_TRUE : GL_FALSE;
            params[2] = g_state->colorMask[0][2] ? GL_TRUE : GL_FALSE;
            params[3] = g_state->colorMask[0][3] ? GL_TRUE : GL_FALSE;
            break;
        default: *params = GL_FALSE; break;
    }
}

void glGetIntegerv(GLenum pname, GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    // NOTE: Do NOT apply a blanket `pname >= MITHRIL_BACKEND_GETTER` offset here.
    // MITHRIL_BACKEND_GETTER is 0x0401, but nearly every standard GL enum is a
    // *larger* value (e.g. GL_MAJOR_VERSION = 0x821B, GL_VIEWPORT = 0x0BA2,
    // GL_MAX_TEXTURE_SIZE = 0x0D33). A `>=` test would hijack all of them,
    // subtract 0x0401, land in `default`, and return 0 — silently breaking
    // glGetIntegerv for Minecraft (which queries GL_MAJOR_VERSION, GL_MAX_*,
    // GL_VIEWPORT, ... on every frame). glGetString has no such blanket bypass:
    // it matches `case MITHRIL_BACKEND_GETTER + GL_*` explicitly. glGetIntegerv
    // has no backend-getter numeric cases, so no bypass is needed here at all.
    switch (pname) {
        /* ---- FIX (P1): GL_MAX_* 改为查询真实的 VkPhysicalDeviceLimits ----
         *
         * 这些值原先全是硬编码。GL_MAX_TEXTURE_SIZE 写死 16384 在 A9/A10 这
         * 类只支持 8192 的 iOS GPU 上是**谎报**：Sodium 和 Iris 会照着这个
         * 数字去分配阴影贴图和图集，vkCreateImage 直接失败 → 纹理丢失/崩溃。
         *
         * GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS 原本写 80，却和内部
         * kMaxTextureUnits=32 自相矛盾 —— 上层按 80 绑定，后端只有 32 个槽，
         * 多出来的静默丢弃。backend_device_limit 会同时夹到设备上限和内部
         * 数组容量两者的较小值。
         *
         * 第二个参数是后端未初始化时的 fallback，保持与修改前一致的取值，
         * 保证不会比原来更差。
         */
        case GL_MAX_TEXTURE_SIZE:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_TEXTURE_SIZE, 16384); break;
        case GL_MAX_3D_TEXTURE_SIZE:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_3D_TEXTURE_SIZE, 2048); break;
        case GL_MAX_CUBE_MAP_TEXTURE_SIZE:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_CUBE_MAP_TEXTURE_SIZE, 16384); break;
        case GL_MAX_ARRAY_TEXTURE_LAYERS:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_ARRAY_TEXTURE_LAYERS, 2048); break;
        case GL_MAX_TEXTURE_IMAGE_UNITS:
        case GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_TEXTURE_IMAGE_UNITS,
                                           mithril::kMaxTextureUnits); break;
        case GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_COMBINED_TEX_UNITS,
                                           mithril::kMaxTextureUnits); break;
        case GL_MAX_VERTEX_ATTRIBS:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_VERTEX_ATTRIBS,
                                           mithril::kMaxVertexAttribs); break;
        case GL_MAX_VERTEX_UNIFORM_COMPONENTS:*params = 4096; break;
        case GL_MAX_FRAGMENT_UNIFORM_COMPONENTS:*params = 4096; break;
        case GL_MAX_VIEWPORT_DIMS:
            params[0] = backend_device_limit(MITHRIL_LIMIT_MAX_VIEWPORT_WIDTH, 16384);
            params[1] = backend_device_limit(MITHRIL_LIMIT_MAX_VIEWPORT_HEIGHT, 16384);
            break;
        case GL_MAX_RENDERBUFFER_SIZE:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_RENDERBUFFER_SIZE, 16384); break;
        case GL_MAX_ELEMENTS_VERTICES:        *params = 1 << 24; break;
        case GL_MAX_ELEMENTS_INDICES:         *params = 1 << 24; break;
        case GL_SUBPIXEL_BITS:                *params = 4; break;
        case GL_RED_BITS:                     *params = 8; break;
        case GL_GREEN_BITS:                   *params = 8; break;
        case GL_BLUE_BITS:                    *params = 8; break;
        case GL_ALPHA_BITS:                   *params = 8; break;
        case GL_DEPTH_BITS:                   *params = 24; break;
        case GL_STENCIL_BITS:                 *params = 8; break;
        case GL_NUM_EXTENSIONS:
            *params = (GLint)(mithril::extensions().size());
            break;
        case GL_MAJOR_VERSION:                *params = mithril::caps().gl_major; break;
        case GL_MINOR_VERSION:                *params = mithril::caps().gl_minor; break;
        case GL_CONTEXT_FLAGS:
            *params = GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT;
            break;
        case GL_CONTEXT_PROFILE_MASK:         *params = GL_CONTEXT_CORE_PROFILE_BIT; break;
        case GL_DOUBLEBUFFER:                 *params = GL_TRUE; break;
        case GL_STEREO:                       *params = GL_FALSE; break;
        case GL_MAX_DRAW_BUFFERS:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_COLOR_ATTACHMENTS, 8); break;
        case GL_MAX_COLOR_ATTACHMENTS:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_COLOR_ATTACHMENTS,
                                           mithril::kMaxColorAttachments); break;
        case GL_MAX_TEXTURE_UNITS:            *params = mithril::kMaxTextureUnits; break;
        /* ---- 4.x capacity limits Sodium / Iris read ---- */
        case GL_MAX_UNIFORM_BUFFER_BINDINGS:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_UNIFORM_BUFFER_BINDINGS,
                                           mithril::kMaxIndexedBindings); break;
        case GL_MAX_UNIFORM_BLOCK_SIZE:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_UNIFORM_BLOCK_SIZE, 64 * 1024); break;
        /* UBO 偏移对齐必须报设备真实值。MoltenVK 上常见 16 或 256，
         * 报小了 Sodium 会按更细的粒度打包 UBO → vkCmdBindDescriptorSets
         * 的 dynamic offset 触发 VUID 校验失败（offset 未对齐）。 */
        case GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT:
            *params = backend_device_limit(MITHRIL_LIMIT_UNIFORM_BUFFER_ALIGNMENT, 256); break;
        case GL_MAX_VERTEX_UNIFORM_BLOCKS:    *params = 14; break;
        case GL_MAX_FRAGMENT_UNIFORM_BLOCKS:  *params = 14; break;
        case GL_MAX_GEOMETRY_UNIFORM_BLOCKS:  *params = 14; break;
        case GL_MAX_COMBINED_UNIFORM_BLOCKS:  *params = 40; break;
        case GL_MAX_VERTEX_OUTPUT_COMPONENTS: *params = 64; break;
        case GL_MAX_FRAGMENT_INPUT_COMPONENTS: *params = 64; break;
        case GL_MAX_SERVER_WAIT_TIMEOUT:      *params = 0x0000FFFF; break;
        /* MSAA 上限来自 framebufferColorSampleCounts & framebufferDepthSampleCounts
         * 的交集。写死 4x 在只支持 2x 的低端 iOS GPU 上会让 MC 的抗锯齿选项
         * 建出无法创建的 multisample 附件。 */
        case GL_MAX_SAMPLES:
        case GL_MAX_COLOR_TEXTURE_SAMPLES:
        case GL_MAX_DEPTH_TEXTURE_SAMPLES:
        case GL_MAX_INTEGER_SAMPLES:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_SAMPLES, 4); break;
        case GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS: *params = 8; break;
        case GL_MAX_COMBINED_ATOMIC_COUNTERS: *params = 8; break;
        case GL_MAX_VERTEX_ATOMIC_COUNTERS:   *params = 8; break;
        case GL_MAX_FRAGMENT_ATOMIC_COUNTERS: *params = 8; break;
        /* ---- Shader storage / compute (Iris) ---- */
        case GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_SSBO_BINDINGS,
                                           mithril::kMaxIndexedBindings); break;
        case GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS: *params = 96; break;
        case GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS: *params = 16; break;
        case GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS: *params = 16; break;
        case GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: *params = 16; break;
        case GL_MAX_SHADER_STORAGE_BLOCK_SIZE:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_SSBO_SIZE, 128 * 1024 * 1024); break;
        case GL_MAX_COMBINED_IMAGE_UNIFORMS:  *params = 192; break;
        case GL_MAX_IMAGE_UNITS:              *params = 32; break;
        case GL_MAX_VERTEX_IMAGE_UNIFORMS:    *params = 32; break;
        case GL_MAX_FRAGMENT_IMAGE_UNIFORMS:  *params = 32; break;
        case GL_MAX_COMPUTE_IMAGE_UNIFORMS:   *params = 32; break;
        case GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS: *params = 192; break;
        case GL_MAX_IMAGE_SAMPLES:            *params = 4; break;
        /* Compute 上限直接决定 Iris 的 compute shader 能否 dispatch。
         * Metal 的 threadgroup 上限比桌面小得多（常见 512 而非 1024），
         * 报高了 vkCmdDispatch 会静默失败或触发 GPU hang。 */
        case GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS:
            *params = backend_device_limit(MITHRIL_LIMIT_MAX_COMPUTE_WG_INVOCATIONS, 1024); break;
        case GL_MAX_COMPUTE_WORK_GROUP_COUNT: {
            int c = backend_device_limit(MITHRIL_LIMIT_MAX_COMPUTE_WG_COUNT_X, 65535);
            params[0] = c; params[1] = c; params[2] = c;
            break;
        }
        case GL_MAX_COMPUTE_WORK_GROUP_SIZE: {
            int s = backend_device_limit(MITHRIL_LIMIT_MAX_COMPUTE_WG_SIZE_X, 256);
            params[0] = s;
            params[1] = s;
            params[2] = s < 64 ? s : 64;
            break;
        }
        case GL_MAX_UNIFORM_LOCATIONS:        *params = 1024; break;
        case GL_MAX_VERTEX_ATTRIB_STRIDE:     *params = 2048; break;
        case GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET: *params = 0x7FFFFFFF; break;
        case GL_MAX_VERTEX_ATTRIB_BINDINGS:   *params = 16; break;
        case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS: *params = 4; break;
        case GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS: *params = 64; break;
        case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS: *params = 16; break;
        case GL_MAX_CULL_DISTANCES:           *params = 8; break;
        case GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES: *params = 8; break;
        case GL_FRAGMENT_INTERPOLATION_OFFSET_BITS: *params = 4; break;
        case GL_ACTIVE_TEXTURE:               *params = (GLint)(GL_TEXTURE0 + g_state->activeTextureUnit); break;
        case GL_ARRAY_BUFFER_BINDING:         *params = (GLint)g_state->bufferBindings[(int)mithril::BufferTarget::Array].name; break;
        case GL_ELEMENT_ARRAY_BUFFER_BINDING: {
            mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
            *params = vao ? (GLint)vao->elementArrayBuffer : 0;
            break;
        }
        case GL_UNIFORM_BUFFER_BINDING:       *params = (GLint)g_state->bufferBindings[(int)mithril::BufferTarget::Uniform].name; break;
        case GL_VERTEX_ARRAY_BINDING:         *params = (GLint)g_state->currentVAO; break;
        case GL_CURRENT_PROGRAM:              *params = (GLint)g_state->currentProgram; break;
        // GL_DRAW_FRAMEBUFFER_BINDING and GL_FRAMEBUFFER_BINDING share the same
        // numeric value (0x8CA6) per the GL spec, so a single case covers both.
        case GL_FRAMEBUFFER_BINDING:          *params = (GLint)g_state->currentDrawFBO; break;
        case GL_READ_FRAMEBUFFER_BINDING:     *params = (GLint)g_state->currentReadFBO; break;
        case GL_VIEWPORT:
            params[0] = g_state->viewportX; params[1] = g_state->viewportY;
            params[2] = g_state->viewportW; params[3] = g_state->viewportH;
            break;
        case GL_SCISSOR_BOX:
            params[0] = g_state->scissorX; params[1] = g_state->scissorY;
            params[2] = g_state->scissorW; params[3] = g_state->scissorH;
            break;
        case GL_COLOR_CLEAR_VALUE:
            for (int i = 0; i < 4; ++i) params[i] = (GLint)g_state->clearColor[i];
            break;
        case GL_DEPTH_FUNC:                   *params = (GLint)g_state->depthFunc; break;
        case GL_CULL_FACE_MODE:               *params = (GLint)g_state->cullMode; break;
        case GL_FRONT_FACE:                   *params = (GLint)g_state->frontFace; break;
        case GL_POLYGON_MODE:                 params[0] = (GLint)g_state->polygonModeFront;
                                                 params[1] = (GLint)g_state->polygonModeBack; break;
        case GL_LINE_WIDTH:                   *params = (GLint)g_state->lineWidth; break;
        case GL_POINT_SIZE:                   *params = (GLint)g_state->pointSize; break;
        case GL_UNPACK_ALIGNMENT:             *params = g_state->pixelStore.unpackAlignment; break;
        case GL_PACK_ALIGNMENT:               *params = g_state->pixelStore.packAlignment; break;
        case GL_UNPACK_ROW_LENGTH:            *params = g_state->pixelStore.unpackRowLength; break;
        case GL_UNPACK_IMAGE_HEIGHT:          *params = g_state->pixelStore.unpackImageHeight; break;
        case GL_TEXTURE_BINDING_2D:           *params = (GLint)g_state->textureBindings[g_state->activeTextureUnit][(int)mithril::TextureTarget::_2D].name; break;
        case GL_BLEND_SRC_RGB:                *params = (GLint)g_state->blends[0].srcRGB; break;
        case GL_BLEND_DST_RGB:                *params = (GLint)g_state->blends[0].dstRGB; break;
        case GL_BLEND_SRC_ALPHA:              *params = (GLint)g_state->blends[0].srcA; break;
        case GL_BLEND_DST_ALPHA:              *params = (GLint)g_state->blends[0].dstA; break;
        case GL_BLEND_EQUATION_RGB:           *params = (GLint)g_state->blends[0].eqRGB; break;
        case GL_BLEND_EQUATION_ALPHA:         *params = (GLint)g_state->blends[0].eqA; break;
        case GL_STENCIL_WRITEMASK:            *params = (GLint)g_state->stencilMask; break;
        case GL_STENCIL_BACK_WRITEMASK:       *params = (GLint)g_state->stencilBackMask; break;
        case GL_STENCIL_FUNC:                 *params = (GLint)g_state->stencilFunc; break;
        case GL_STENCIL_REF:                  *params = g_state->stencilRef; break;
        case GL_STENCIL_VALUE_MASK:           *params = (GLint)g_state->stencilValueMask; break;
        case GL_STENCIL_FAIL:                 *params = (GLint)g_state->stencilSfail; break;
        case GL_STENCIL_PASS_DEPTH_FAIL:     *params = (GLint)g_state->stencilDpfail; break;
        case GL_STENCIL_PASS_DEPTH_PASS:     *params = (GLint)g_state->stencilDppass; break;
        case GL_SHADING_LANGUAGE_VERSION:     *params = mithril::caps().glsl_major * 100 + mithril::caps().glsl_minor; break;
        /* GL 4.5 ARB_clip_control: queryable clip volume state.
         * Required for completeness since we advertise GL_ARB_clip_control and
         * implement glClipControl. MC/Sodium may query these to decide whether
         * to apply its own Y-flip / depth remap. */
        case 0x935C: /*GL_CLIP_ORIGIN*/       *params = (GLint)g_state->clipOrigin; break;
        case 0x935D: /*GL_CLIP_DEPTH_MODE*/   *params = (GLint)g_state->clipDepthMode; break;
        default:
            // GL spec: an unrecognised pname must raise GL_INVALID_ENUM rather
            // than silently returning 0 (Minecraft relies on the error flag to
            // detect unsupported queries). Keep *params deterministic too.
            *params = 0;
            mithril::state_set_error(GL_INVALID_ENUM);
            break;
    }
}

void glGetFloatv(GLenum pname, GLfloat* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    GLint ip[4] = {0,0,0,0};
    glGetIntegerv(pname, ip);
    switch (pname) {
        case GL_COLOR_CLEAR_VALUE:
            for (int i = 0; i < 4; ++i) params[i] = g_state->clearColor[i];
            return;
        case GL_LINE_WIDTH:        *params = g_state->lineWidth; return;
        case GL_POINT_SIZE:        *params = g_state->pointSize; return;
        case GL_POLYGON_OFFSET_FACTOR: *params = g_state->polygonOffsetFactor; return;
        case GL_POLYGON_OFFSET_UNITS:  *params = g_state->polygonOffsetUnits;  return;
        case GL_BLEND_COLOR:
            for (int i = 0; i < 4; ++i) params[i] = g_state->blendColor[i];
            return;
        default:
            for (int i = 0; i < 4; ++i) params[i] = (GLfloat)ip[i];
            return;
    }
}

void glGetDoublev(GLenum pname, GLdouble* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    GLfloat f[4] = {0,0,0,0};
    glGetFloatv(pname, f);
    for (int i = 0; i < 4; ++i) params[i] = (GLdouble)f[i];
}

void glGetInteger64v(GLenum pname, GLint64* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    GLint ip[4] = {0,0,0,0};
    glGetIntegerv(pname, ip);
    for (int i = 0; i < 4; ++i) params[i] = (GLint64)ip[i];
}

void glGetIntegeri_v(GLenum pname, GLuint index, GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    *params = 0;
    if (index >= mithril::kMaxIndexedBindings) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return;
    }
    // Indexed buffer binding queries (UBO / TF / AtomicCounter / SSBO).
    // Determine which indexed category the pname belongs to.
    mithril::IndexedBufferTarget cat = mithril::IndexedBufferTarget::Count;
    bool isStart = false, isSize = false, isBinding = false;

    switch (pname) {
        case GL_UNIFORM_BUFFER_BINDING:        cat = mithril::IndexedBufferTarget::Uniform; isBinding = true; break;
        case 0x8F29 /*GL_UNIFORM_BUFFER_START*/:cat = mithril::IndexedBufferTarget::Uniform; isStart = true; break;
        case 0x8F2A /*GL_UNIFORM_BUFFER_SIZE*/: cat = mithril::IndexedBufferTarget::Uniform; isSize = true; break;
        case 0x8C7A /*GL_TRANSFORM_FEEDBACK_BUFFER_BINDING*/: cat = mithril::IndexedBufferTarget::TransformFeedback; isBinding = true; break;
        case 0x8C84 /*GL_TRANSFORM_FEEDBACK_BUFFER_START*/:   cat = mithril::IndexedBufferTarget::TransformFeedback; isStart = true; break;
        case 0x8C85 /*GL_TRANSFORM_FEEDBACK_BUFFER_SIZE*/:    cat = mithril::IndexedBufferTarget::TransformFeedback; isSize = true; break;
        case 0x92C1 /*GL_ATOMIC_COUNTER_BUFFER_BINDING*/:     cat = mithril::IndexedBufferTarget::AtomicCounter; isBinding = true; break;
        case 0x92C2 /*GL_ATOMIC_COUNTER_BUFFER_START*/:       cat = mithril::IndexedBufferTarget::AtomicCounter; isStart = true; break;
        case 0x92C3 /*GL_ATOMIC_COUNTER_BUFFER_SIZE*/:        cat = mithril::IndexedBufferTarget::AtomicCounter; isSize = true; break;
        case 0x90D3 /*GL_SHADER_STORAGE_BUFFER_BINDING*/:     cat = mithril::IndexedBufferTarget::ShaderStorage; isBinding = true; break;
        case 0x90D4 /*GL_SHADER_STORAGE_BUFFER_START*/:       cat = mithril::IndexedBufferTarget::ShaderStorage; isStart = true; break;
        case 0x90D5 /*GL_SHADER_STORAGE_BUFFER_SIZE*/:        cat = mithril::IndexedBufferTarget::ShaderStorage; isSize = true; break;
        default:
            // For non-indexed pnames (e.g. GL_COLOR_WRITEMASK with index), fall back.
            return;
    }
    if (cat == mithril::IndexedBufferTarget::Count) return;
    const mithril::IndexedBindingSlot& slot =
        g_state->indexedBufferBindings[(int)cat][index];
    if (isBinding) *params = (GLint)slot.name;
    else if (isStart) *params = (GLint)slot.offset;
    else if (isSize)  *params = (GLint)slot.size;
}

const GLubyte* glGetString(GLenum name) {
    MITHRIL_ENSURE_INIT();
    switch (name) {
        case GL_VENDOR:                   return (const GLubyte*)kVendor;
        case GL_RENDERER:
#if defined(__APPLE__)
            return (const GLubyte*)mithril_get_gpu_renderer_string();
#else
            return (const GLubyte*)kRenderer;
#endif
        case GL_VERSION:                  return (const GLubyte*)mithril::version_string().c_str();
        case GL_SHADING_LANGUAGE_VERSION: return (const GLubyte*)mithril::glsl_version_string().c_str();
        case GL_EXTENSIONS: {
            // Concatenate into a single space-separated string.
            static std::string all;
            if (all.empty()) {
                for (size_t i = 0; i < mithril::extensions().size(); ++i) {
                    if (i) all += " ";
                    all += mithril::extensions()[i];
                }
            }
            return (const GLubyte*)all.c_str();
        }
        /*
         * Mithril custom enums for F3 debug info mapping (mirrors MobileGlues'
         * GL_SETTINGS_MG / GL_BACKEND_GETTER_MG pattern). Minecraft mods can
         * probe these via glGetString to display Mithril's config on the F3
         * screen, or to bypass the OpenGL facade and get the real Vulkan info.
         */
        case MITHRIL_SETTINGS:
#if defined(__APPLE__)
            return (const GLubyte*)mithril_get_settings_dump();
#else
            return (const GLubyte*)"Mithril-Wrapper (non-Vulkan build)";
#endif
        case MITHRIL_BACKEND_GETTER + GL_RENDERER:
#if defined(__APPLE__)
            return (const GLubyte*)mithril_get_vulkan_device_name();
#else
            return (const GLubyte*)"Mithril-Wrapper (no device)";
#endif
        case MITHRIL_BACKEND_GETTER + GL_VERSION:
            return (const GLubyte*)"Vulkan 1.2 (MoltenVK)";
        case MITHRIL_BACKEND_GETTER + GL_VENDOR:
            return (const GLubyte*)"Khronos MoltenVK";
        case MITHRIL_BACKEND_GETTER + GL_SHADING_LANGUAGE_VERSION:
            return (const GLubyte*)"SPIR-V 1.5 (glslang -> SPIR-V)";
        default: return nullptr;
    }
}

const GLubyte* glGetStringi(GLenum name, GLuint index) {
    MITHRIL_ENSURE_INIT();
    if (name != GL_EXTENSIONS) return nullptr;
    if (index >= mithril::extensions().size()) return nullptr;
    return (const GLubyte*)mithril::extensions()[index];
}

/* ---- Indexed state queries, remaining widths (root cause AR) ----
 * Every indexed pname in GL 3.3 Core is integer- or boolean-valued, so these
 * reuse glGetIntegeri_v's table instead of duplicating its pname switch. */
void glGetBooleani_v(GLenum target, GLuint index, GLboolean* data) {
    MITHRIL_ENSURE_INIT();
    if (!data) return;
    GLint v = 0;
    glGetIntegeri_v(target, index, &v);
    *data = v ? GL_TRUE : GL_FALSE;
}

void glGetInteger64i_v(GLenum target, GLuint index, GLint64* data) {
    MITHRIL_ENSURE_INIT();
    if (!data) return;
    GLint v = 0;
    glGetIntegeri_v(target, index, &v);
    *data = (GLint64)v;
}

} // extern "C"
