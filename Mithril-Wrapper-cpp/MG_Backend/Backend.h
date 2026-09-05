// Mithril-Wrapper - MG_Backend/Backend.h
// Abstract backend interface (C API) for the Vulkan 1.2 / MoltenVK backend.
//
// This is the Vulkan equivalent of the former metal/metal_context.h +
// metal_objects.h + metal_pipeline.h trio. The implementation lives in
// MG_Backend/DirectVulkan/ and talks to Vulkan directly; MoltenVK then
// cross-translates the SPIR-V shaders and Vulkan commands to Metal 2
// internally, so no Metal code remains in this project.
//
// Handles passed across this boundary (VkBuffer / VkImage / VkImageView /
// VkSampler / VkPipeline) are real Vulkan handles. The GL frontend never
// creates or destroys them directly — it goes through backend_get_or_create_*.
#ifndef MITHRIL_BACKEND_H
#define MITHRIL_BACKEND_H

#include <cstdint>
#include <cstddef>

#include <vulkan/vulkan.h>
#include <GL/gl.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Lifecycle ----
 * backend_init() creates the VkInstance / VkPhysicalDevice / VkDevice /
 * VkQueue / VkCommandPool once. It is idempotent. backend_available() reports
 * whether the Vulkan backend came up (eglInitialize gates on this).
 */
void backend_init(void);
void backend_shutdown(void);
int  backend_available(void);

/*
 * Physical-device introspection (used by Getter_gpu.cpp to build the
 * GL_RENDERER string from VkPhysicalDeviceProperties instead of MTLDevice).
 */
const char* backend_physical_device_name(void);   // e.g. "Apple A14 GPU"
uint64_t    backend_vram_bytes(void);             // 0 if unknown

/* ---- Swapchain (created/owned by EGL) ----
 * EGL installs the per-frame swapchain color/depth VkImageViews onto the
 * active GLState (eglDefaultColor/Depth). The backend never creates the
 * swapchain itself — that is the EGL layer's job (it owns the CAMetalLayer).
 */

/* ---- Clear values applied to the load op of the next render pass ---- */
void backend_set_clear_color(float r, float g, float b, float a);
void backend_set_clear_depth(double d);
void backend_set_clear_stencil(int s);

/* Load op for the next pass: CLEAR (glClear) or LOAD (draw pass). */
void backend_set_load_clear(void);
void backend_set_load_load(void);

/*
 * Clear specific aspects of the current framebuffer's attachments using
 * vkCmdClearAttachments. MUST be called inside a render pass (after
 * backend_begin_render_pass and before backend_end_render_pass). The clear
 * respects GL scissor: if scissor test is enabled, the clear rect is the
 * scissor rect; otherwise it is the full framebuffer (0,0,w,h).
 *
 *   mask : GLbitfield of GL_COLOR_BUFFER_BIT / GL_DEPTH_BUFFER_BIT /
 *          GL_STENCIL_BUFFER_BIT (OR'd together, exactly as glClear takes).
 *   x,y,w,h : framebuffer dimensions (used when scissor test is disabled).
 *
 * This replaces the old approach of using loadOp=CLEAR for glClear, which
 * cleared ALL attachments regardless of mask — so glClear(GL_DEPTH_BUFFER_BIT)
 * would wipe the color buffer too, causing black screen.
 */
void backend_clear_attachments(GLbitfield mask, int x, int y, int w, int h);

/*
 * Clear ONE attachment with an explicit value — the backing call for
 * glClearBuffer{fv,iv,uiv,fi} (root cause AP).
 *
 * backend_clear_attachments always clears every colour attachment with the
 * context-wide glClearColor. That is right for glClear(), but glClearBuffer*
 * names a single draw buffer and carries its own value, which is how a
 * deferred renderer wipes just its normal or velocity target between passes.
 *
 *   buffer     : GL_COLOR | GL_DEPTH | GL_STENCIL | GL_DEPTH_STENCIL
 *   drawbuffer : colour attachment index for GL_COLOR, otherwise ignored
 *   color      : RGBA, used only for GL_COLOR (NULL -> zeros)
 *   depth      : caller must already have clamped this to [0,1]
 *                (VUID-VkClearDepthStencilValue-depth-00022)
 *
 * No-op unless a render pass is active, matching backend_clear_attachments.
 */
void backend_clear_buffer_indexed(GLenum buffer, GLint drawbuffer,
                                  const float color[4], float depth,
                                  GLuint stencil);

/*
 * Begin a dynamic-rendering pass against the given attachments.
 *   color_views : array of VkImageView (VK_NULL_HANDLE entries allowed)
 *   color_count : number of color attachments
 *   depth_view  : VkImageView for depth/stencil (may be VK_NULL_HANDLE)
 *   width/height: render area
 *   samples     : 1 for now
 * If a pass is already active, this is a no-op (coalesce draws into one pass).
 */
void backend_begin_render_pass(VkImageView* color_views, int color_count,
                               VkImageView depth_view, int width, int height,
                               int samples);

/*
 * Register the GL texture names backing the user-FBO color/depth attachments
 * for the NEXT backend_begin_render_pass. VK_KHR_dynamic_rendering does NOT
 * auto-transition attachment image layouts — it only validates each image is
 * in the layout declared by VkRenderingAttachmentInfo.imageLayout. User-FBO
 * color/depth textures live in the backend texture table (TextureEntry) with
 * a tracked currentLayout; without an explicit barrier to
 * COLOR_ATTACHMENT_OPTIMAL / DEPTH_STENCIL_ATTACHMENT_OPTIMAL, the actual
 * layout (SHADER_READ_ONLY_OPTIMAL from a prior upload) mismatches the
 * declared layout → spec violation → MoltenVK drops the draw → black screen
 * (root cause Y). begin_render_pass reads these tex_ids, looks up each
 * TextureEntry, and barriers its image to attachment-optimal; end_render_pass
 * barriers them back to a read-only layout and updates currentLayout, then
 * auto-clears the registration.
 *
 *   color_tex_ids : array of GL texture names (may be NULL when color_count==0)
 *   color_count   : number of color attachments (clamped to 8)
 *   depth_tex_id  : GL texture name for the depth attachment (0 = none)
 *
 * For swapchain rendering (FBO 0) pass NULL/0/0 to clear any stale
 * registration — the swapchain path's barriers are handled by the
 * activeSwapchain block in begin_render_pass / commit_frame.
 */
void backend_set_fbo_attachment_tex_ids(GLuint* color_tex_ids, int color_count,
                                        GLuint depth_tex_id);

/*
 * GL 4.3 ARB_invalidate_subdata: mark framebuffer attachments for discard.
 * The NEXT backend_begin_render_pass will use VK_ATTACHMENT_STORE_OP_DONT_CARE
 * for the specified attachments (instead of STORE), then auto-clear the flags
 * (invalidation is one-shot). On TBDR GPUs (Apple Silicon), this skips the
 * tile-memory → system-memory writeback, saving bandwidth and VRAM pressure.
 *
 *   color_mask : bit i set = discard color attachment i
 *   depth      : true = discard depth
 *   stencil    : true = discard stencil
 */
void backend_set_invalidate_attachments(uint32_t color_mask, bool depth, bool stencil);

/* End + commit the active render pass / command buffer. */
void backend_end_render_pass(void);
void backend_commit(void);

/*
 * Register the swapchain whose currently-acquired image backs framebuffer 0
 * for the current frame. Called by EGL (install_surface_on_state) right after
 * backend_swapchain_acquire_color(). begin_render_pass() / commit_frame() use
 * it to record the PRESENT_SRC/UNDEFINED <-> COLOR_ATTACHMENT_OPTIMAL layout
 * barriers on the swapchain color image, the one-shot UNDEFINED ->
 * DEPTH_STENCIL_ATTACHMENT_OPTIMAL barrier on the depth image, and to signal
 * the swapchain's per-image renderFinished semaphore on submit. Pass nullptr
 * when no surface is current (headless / surface destroyed).
 */
void backend_set_active_swapchain(void* swapchain_state);

/*
 * Update the swapchain's tracked actual drawable size (from the native
 * window's current size, e.g. CAMetalLayer.drawableSize). Called by EGL's
 * install_surface_on_state() after each acquire so begin_render_pass() can
 * clamp the render area to the ACTUAL IOSurface dimensions, not just the
 * swapchain's creation-time extent. This prevents IOSurfaceBindAccel SIGSEGV
 * when the drawableSize changed after swapchain creation (e.g. GLFW resized
 * the window between swapchain creation and the first frame).
 */
void backend_swapchain_set_drawable_size(void* swapchain_state, int w, int h);

/*
 * Mark the swapchain for rebuild on the next eglSwapBuffers. Called by EGL
 * when drawableSize no longer matches the swapchain's creation-time extent —
 * the swapchain's VkImages are the wrong size and must be recreated to match
 * the current CAMetalLayer drawableSize, otherwise MoltenVK presents a
 * mismatched drawable (red screen / garbage).
 */
void backend_swapchain_mark_rebuild(void* swapchain_state);

/*
 * Drain any in-flight GPU work that references the active swapchain, then
 * detach the swapchain from the encoder so subsequent begin_render_pass() /
 * commit_frame() calls cannot record barriers against its (soon-to-be-destroyed)
 * images. Called by EGL BEFORE backend_destroy_swapchain() (in
 * eglDestroySurface / ensure_swapchain resize path).
 *
 * Without this call, destroying a swapchain whose currentImage is still
 * encoded in the pending command buffer leaves a dangling reference: the
 * next vkQueueSubmit hands MoltenVK a command that references an already-
 * freed IOSurface, and IOSurfaceBindAccel crashes in the GPU driver with
 * SIGSEGV (UAF). The vkDeviceWaitIdle() inside ensures the GPU has finished
 * reading the swapchain images before they are torn down; the
 * set_active_swapchain(nullptr) ensures the encoder never records against
 * the dead swapchain again.
 */
void backend_drain_and_detach_swapchain(void);

/*
 * Query whether the swapchain has been marked dead by a fatal Vulkan error
 * (VK_ERROR_OUT_OF_DEVICE_MEMORY / VK_ERROR_SURFACE_LOST_KHR /
 * VK_ERROR_DEVICE_LOST from vkAcquireNextImageKHR / vkQueuePresentKHR /
 * vkQueueSubmit). EGL calls this in eglSwapBuffers; if it returns true, EGL
 * drains + destroys the dead swapchain and creates a fresh one before
 * continuing. Without this query, a dead swapchain would keep returning null
 * from acquire and the render thread would spin in a no-op loop (the
 * "VK_NOT_READY death spiral" reported under GPU OOM).
 */
int backend_swapchain_needs_rebuild(void* swapchain_state);

/*
 * Encoder-side dynamic state setters (vkCmdSet* under dynamic rendering).
 * Each is a no-op when no render pass is active. Stage: 0 = vertex, 1 = fragment
 * (used for buffer/texture/sampler binding).
 */
void backend_bind_pipeline(VkPipeline pipeline);
void backend_set_viewport(int x, int y, int w, int h, double znear, double zfar);
void backend_set_scissor(int x, int y, int w, int h);
void backend_set_vertex_buffer(int slot, VkBuffer buffer, VkDeviceSize offset);
void backend_set_fragment_buffer(int slot, VkBuffer buffer, VkDeviceSize offset);
void backend_set_vertex_texture(int slot, VkImageView view, VkSampler sampler);
void backend_set_fragment_texture(int slot, VkImageView view, VkSampler sampler);
void backend_set_blend_color(float r, float g, float b, float a);
void backend_set_depth_bias(float slope, float clamp);
void backend_set_cull_mode(int mode);        /* 0=None,1=Front,2=Back */
void backend_set_front_face(int ccw);        /* 1=CCW, 0=CW */
void backend_set_depth_test(int enabled, int write_mask, int compare_func);
void backend_set_color_write_mask(int r, int g, int b, int a);
void backend_set_stencil_state(int enabled, int func, int ref, int mask,
                               int sfail, int dpfail, int dppass);

/* Draw primitives. `index_type` 0=U16 (VK_INDEX_TYPE_UINT16), 1=U32,
 * 2=U8 (VK_INDEX_TYPE_UINT8_EXT, requires VK_EXT_index_type_uint8). */
void backend_draw_arrays(int primitive, int first, int count);
void backend_draw_indexed(int primitive, int count, int index_type,
                          VkBuffer index_buffer, VkDeviceSize index_offset);
void backend_draw_arrays_instanced(int primitive, int first, int count, int primcount);
void backend_draw_indexed_instanced(int primitive, int count, int index_type,
                                    VkBuffer index_buffer, VkDeviceSize index_offset,
                                    int primcount);

/*
 * Push a vertex-stage push constant into the CURRENT program's pipeline
 * layout (root cause: gl_VertexID baseVertex semantics).
 *
 * Every vertex shader carries a `_MithrilBaseVertex` push-constant block
 * (offset 0, size 4 = int) injected by Shader.cpp; Drawing.cpp calls this
 * with g_state->currentBaseVertex on every draw so the shader's gl_VertexID
 * (defined as gl_VertexIndex + _mbv._mithrilBaseVertex) matches desktop GL.
 *
 * `program` selects the per-program VkPipelineLayout (via program_table());
 * a program with no descriptor bindings falls back to the process-wide empty
 * layout, which also declares the same push-constant range. Callable outside
 * a render pass (vkCmdPushConstants is a state command); typically invoked in
 * prepare_draw right after backend_bind_pipeline.
 */
void backend_push_constants(GLuint program, uint32_t offset, uint32_t size,
                            const void* data);

/*
 * Indirect draws (GL 4.0 ARB_draw_indirect).
 *
 * Draw parameters are read from a GPU buffer rather than passed in. GL's
 * parameter blocks are bit-identical to VkDrawIndirectCommand /
 * VkDrawIndexedIndirectCommand, so the buffer contents pass through
 * untranslated.
 *
 *   stride == 0 means tightly packed (16 / 20 bytes respectively).
 *   draw_count > 1 uses multiDrawIndirect where available and loops otherwise.
 */
void backend_draw_indirect(int primitive, VkBuffer indirect_buffer,
                           VkDeviceSize indirect_offset,
                           int draw_count, int stride);
void backend_draw_indexed_indirect(int primitive, int index_type,
                                   VkBuffer index_buffer, VkDeviceSize index_offset,
                                   VkBuffer indirect_buffer,
                                   VkDeviceSize indirect_offset,
                                   int draw_count, int stride);

/*
 * GL 4.6 ARB_indirect_parameters — glMultiDraw*IndirectCount.
 *
 * The draw COUNT is read by the GPU from `count_buffer` at `count_offset`
 * (a GLintptr offset into GL_DRAW_INDIRECT_BUFFER), clamped to
 * max_drawcount. Requires vkCmdDrawIndirectCount /
 * vkCmdDrawIndexedIndirectCount (Vulkan 1.2 core `drawIndirectCount`).
 * Callers must verify b->drawIndirectCountSupported first; when it is
 * unsupported they should fall back to a CPU readback (Sodium only uses
 * this on GL 4.6-capable devices, and MoltenVK 1.2.x reports it).
 */
void backend_draw_indirect_count(int primitive, VkBuffer indirect_buffer,
                                 VkDeviceSize indirect_offset,
                                 VkBuffer count_buffer, VkDeviceSize count_offset,
                                 int max_drawcount, int stride);
void backend_draw_indexed_indirect_count(int primitive, int index_type,
                                         VkBuffer index_buffer, VkDeviceSize index_offset,
                                         VkBuffer indirect_buffer, VkDeviceSize indirect_offset,
                                         VkBuffer count_buffer, VkDeviceSize count_offset,
                                         int max_drawcount, int stride);

/* ---- Buffers ---- */
VkBuffer backend_get_or_create_buffer(GLuint name, const void* data, size_t size);
/* GL_ARB_buffer_storage — immutable storage, optionally persistently & coherently
 * mapped. `persistent` keeps the host pointer live for the buffer's lifetime so
 * glMapBufferRange can return a direct slice of it. `extra_usage` adds Vulkan
 * usage bits beyond the default set (e.g. for indirect/SSBO already included). */
VkBuffer backend_create_buffer_storage(GLuint name, VkDeviceSize size,
                                       VkBufferUsageFlags extra_usage,
                                       bool persistent, bool coherent);
void     backend_buffer_upload(GLuint name, GLintptr offset, const void* data, size_t size);
/* Return the live host pointer of a persistently-mapped buffer (glBufferStorage
 * + MAP_PERSISTENT), or NULL if the buffer is not persistently mapped. */
void*    backend_get_buffer_mapped_pointer(GLuint name);
VkBuffer backend_get_buffer(GLuint name);
void     backend_delete_buffer(GLuint name);

/* Shared 16-byte zero-filled buffer for unbound vertex attribute slots. */
VkBuffer backend_get_zero_buffer(void);

/*
 * Generic vertex attribute values (root cause AQ).
 *
 * A single buffer of N vec4 slots backing the constants a shader reads when a
 * vertex array is DISABLED. Bind slot `i` at offset `i * 16` with stride 0.
 *
 * Needed because the zero buffer above hands out (0,0,0,0) while GL specifies
 * (0,0,0,1) — with alpha 0 a disabled colour array makes geometry vanish
 * under blending instead of drawing opaque black — and because
 * glVertexAttrib*() is allowed to change these constants at any time.
 *
 * `values` is `count` consecutive vec4s. Returns VK_NULL_HANDLE before the
 * first update.
 */
void     backend_update_generic_attribs(const float* values, int count);
VkBuffer backend_get_generic_attrib_buffer(void);

/* ---- Textures ---- */

/*
 * Unpack (pixel-store) parameters handed to backend_texture_upload. Mirrors the
 * glPixelStorei UNPACK_* state kept in g_state->pixelStore. The struct form lets
 * the MG_Impl layer build the params once and pass a stable pointer instead of
 * a single alignment integer, so the backend can honour
 * UNPACK_ROW_LENGTH / UNPACK_SKIP_* for sub-image uploads.
 */
struct MGUnpackParams {
    GLint unpackAlignment;
    GLint unpackRowLength;
    GLint unpackSkipPixels;
    GLint unpackSkipRows;
    GLint unpackImageHeight;
    GLint unpackSkipImages;
};

VkImage     backend_get_or_create_texture(GLuint name, int width, int height, int depth,
                                          int levels, GLenum internal_format, GLenum target,
                                          int samples);
void        backend_texture_upload(GLuint name, int level, int x, int y, int z,
                                   int w, int h, int d, GLenum format, GLenum type,
                                   const void* pixels, const MGUnpackParams* unpack,
                                   int is_full_upload);
/* Compressed texture upload. dataLen is the byte size of the compressed
 * payload. internalFormat is the GL compressed format enum (e.g.
 * GL_COMPRESSED_RGBA8_ETC2_EAC). The VkFormat is resolved via
 * backend_vk_format_for_gl in Resources.cpp. No pixel unpack/expand is
 * performed — compressed data is copied verbatim to the staging buffer and
 * vkCmdCopyBufferToImage copies it block-by-block. */
void        backend_texture_upload_compressed(GLuint name, int level, int x, int y, int z,
                                              int w, int h, int d,
                                              GLenum internalFormat,
                                              GLsizei dataLen, const void* pixels,
                                              int is_full_upload);
void        backend_texture_set_params(GLuint name, GLint min_filter, GLint mag_filter,
                                       GLint wrap_s, GLint wrap_t, GLint wrap_r,
                                       const float* border_color);
VkImageView backend_get_texture_view(GLuint name);
VkImage     backend_get_texture_image(GLuint name);
void        backend_delete_texture(GLuint name);
/* Drop any VkSampler cached for `name`. Called from glTexParameter*i* when a
 * texture's filtering/wrap state changes, so the next sampler fetch rebuilds a
 * VkSampler reflecting the new params instead of returning the stale cached one. */
void        backend_invalidate_sampler_cache(GLuint name);

/*
 * Transition the named texture's image into `target_layout` if it is not
 * already in that layout. Records an image-memory barrier on the active
 * command buffer (caller is responsible for committing). Used by glTexStorage*
 * to put freshly-allocated immutable storage into SHADER_READ_ONLY_OPTIMAL so
 * that subsequent sampler bindings / framebuffer attachment uses see a valid
 * layout instead of UNDEFINED. No-op if the texture doesn't exist or is
 * already in the target layout.
 *
 *   name          : GL texture name
 *   target_layout : VkImageLayout to transition to (e.g.
 *                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
 */
void        backend_transition_texture_layout(GLuint name, VkImageLayout target_layout);

/*
 * Generate mipmaps for the named texture via vkCmdBlitImage. Each level L>=1
 * is blitted from level L-1 (half-sampled, linear filter) and transitioned
 * to SHADER_READ_ONLY_OPTIMAL. The texture must have been created with
 * `levels` > 1 (glTexStorage2D / glTexImage2D with levels>1). Records into
 * the active command buffer; the caller is responsible for committing.
 */
void        backend_generate_mipmaps(GLuint name);

/*
 * Read a rectangular region of pixels back from the bound colour attachment
 * into `out_pixels`. The (format,type) pair selects the host pixel layout
 * (only GL_RGBA + GL_UNSIGNED_BYTE is fully implemented; other combos are
 * best-effort). Implementation: vkCmdCopyImage -> host-visible staging
 * buffer -> vkQueueSubmit + vkWaitForFences -> memcpy/convert into out.
 *
 *   x,y,w,h : source rectangle in the colour attachment (GL bottom-left origin)
 *   format/type : GL pixel format/type of out_pixels
 *   out_pixels : caller-allocated buffer of size w*h*bpp(format,type)
 *
 * Returns 1 on success, 0 on failure (e.g. no colour attachment bound).
 */
int         backend_read_pixels(int x, int y, int w, int h,
                                GLenum format, GLenum type, void* out_pixels);

/*
 * Blit a rectangular region from the source texture to the destination
 * texture, with optional colour/depth/stencil mask and linear/nearest
 * filtering. Mirrors glBlitFramebuffer's behaviour for the cases MC Java
 * exercises (resolve downsample, fullscreen post-process copies).
 *
 *   src_name / dst_name : GL texture names (must be 2D, same format)
 *   srcX0..srcY1 / dstX0..dstY1 : source + destination rectangles (GL coords)
 *   mask   : GLbitfield of GL_COLOR_BUFFER_BIT / GL_DEPTH_BUFFER_BIT / GL_STENCIL_BUFFER_BIT
 *   filter : GL_NEAREST or GL_LINEAR
 */
void        backend_blit_texture(GLuint src_name, GLuint dst_name,
                                 int srcX0, int srcY0, int srcX1, int srcY1,
                                 int dstX0, int dstY0, int dstX1, int dstY1,
                                 GLbitfield mask, GLenum filter);

/*
 * Blit a rectangular region between two raw VkImage handles (used by
 * glBlitFramebuffer when one or both sides is the EGL default framebuffer,
 * whose swapchain image is not in the GL texture table). Handles layout
 * transitions for both images (assumes both start/end in either
 * SHADER_READ_ONLY_OPTIMAL or COLOR_ATTACHMENT_OPTIMAL, and transitions to
 * TRANSFER_SRC/DST_OPTIMAL for the blit, then back to a sampling-friendly
 * layout). `src_format`/`dst_format` select the aspect mask (color/depth).
 *
 *   src_image / dst_image : VkImage handles (must not be VK_NULL_HANDLE)
 *   src_format / dst_format : VkFormat of each image (for aspect selection)
 *   srcX0..srcY1 / dstX0..dstY1 : source + destination rectangles (GL coords)
 *   mask   : GLbitfield of GL_COLOR_BUFFER_BIT (depth/stencil not yet supported)
 *   filter : GL_NEAREST or GL_LINEAR
 *   is_dst_default_fbo : 1 if dst_image is the EGL default framebuffer (swapchain
 *                        drawable), 0 if it is a user FBO texture. When 1, the
 *                        destination Y is flipped (vulkanDstY = dst_height - glDstY)
 *                        to convert GL bottom-left coords to Vulkan top-left coords.
 *                        Deep reference: MobileGL ApplyNativeBlitDefaultFramebufferTransform.
 *   dst_height : height of the destination framebuffer (in pixels). Used only when
 *                is_dst_default_fbo is 1 for the Y flip computation.
 */
void        backend_blit_images(VkImage src_image, VkFormat src_format,
                                VkImage dst_image, VkFormat dst_format,
                                int srcX0, int srcY0, int srcX1, int srcY1,
                                int dstX0, int dstY0, int dstX1, int dstY1,
                                GLbitfield mask, GLenum filter,
                                int is_dst_default_fbo, int dst_height);

/* ---- Samplers ---- */
VkSampler backend_get_or_create_sampler(GLuint name, GLint min_filter, GLint mag_filter,
                                        GLint wrap_s, GLint wrap_t, GLint wrap_r,
                                        const float* border_color,
                                        GLint compare_enable, GLint compare_op,
                                        GLfloat min_lod, GLfloat lod_bias);

/* ---- Format helpers ----
 * Map a GL internal format to the matching VkFormat. Returns VK_FORMAT_UNDEFINED
 * when the format is unsupported. Used by the drawing path to describe pipeline
 * color/depth attachment formats.
 */
VkFormat backend_vk_format_for_gl(GLenum internal_format);

/* ---- Pipeline cache ----
 * Description of one bound vertex attribute used to build the
 * VkPipelineVertexInputStateCreateInfo.
 */
struct MGVertexAttrib {
    int     location;     /* GL attribute index */
    int     size;         /* 1..4 */
    GLenum  type;         /* GL_FLOAT, GL_UNSIGNED_BYTE, etc. */
    int     normalized;   /* 0/1 */
    int     integer;      /* 0/1 (integer attribs) */
    int     stride;
    int     offset;       /* byte offset within the bound vertex buffer */
    int     enabled;      /* 0/1 */
    GLuint  buffer_name;  /* GL VBO name backing this attrib */
    int     divisor;      /* instance-step divisor (0 = per-vertex) */
};

/*
 * Build a VkShaderModule from SPIR-V words (cached on the program) and a
 * VkGraphicsPipeline matching the given vertex format and framebuffer
 * color/depth VkFormats. Blend state is part of the pipeline signature.
 *
 *   vertex_spirv / vertex_word_count   : vertex-stage SPIR-V
 *   fragment_spirv / fragment_word_count: fragment-stage SPIR-V (may be NULL/0)
 *   attribs / attrib_count             : enabled vertex attributes
 *   color_formats / color_count        : VkFormat values for color attachments
 *   depth_format                       : VkFormat for depth (VK_FORMAT_UNDEFINED = none)
 *   blend_enabled                      : 0/1
 *   blend_src / blend_dst              : GL blend factor enums (GL_SRC_ALPHA, etc.)
 *   gl_primitive_mode                  : GL primitive mode (cache key only)
 *   is_default_fbo                     : 1 when drawing to FBO 0 (selects the
 *                                        Y-flipped vertex module + pipeline hash)
 *
 * Returns a cached VkPipeline (VK_NULL_HANDLE on failure).
 */
VkPipeline backend_get_or_create_pipeline(GLuint program,
                                          const uint32_t* vertex_spirv, int vertex_word_count,
                                          const uint32_t* fragment_spirv, int fragment_word_count,
                                          const struct MGVertexAttrib* attribs, int attrib_count,
                                          const VkFormat* color_formats, int color_count,
                                          VkFormat depth_format,
                                          int blend_enabled, GLenum blend_src, GLenum blend_dst,
                                          GLenum blend_src_alpha, GLenum blend_dst_alpha,
                                          int color_write_mask,
                                          GLenum gl_primitive_mode,
                                          int is_default_fbo);

/*
 * Compute counterpart of backend_get_or_create_pipeline. Takes only the
 * program name: a compute pipeline has no vertex format, no attachments and
 * no blend state, so there is nothing to key a cache on and nothing for the
 * caller to supply. The SPIR-V is read from the linked Program.
 * Returns VK_NULL_HANDLE if the program is not a linked compute program or
 * pipeline creation fails.
 */
VkPipeline backend_get_or_create_compute_pipeline(GLuint program);

/*
 * Record a compute dispatch on the current command buffer. Ends the active
 * render pass first (Vulkan forbids vkCmdDispatch inside a render pass),
 * binds the current program's compute pipeline + descriptor set, then
 * vkCmdDispatch(groups_x, groups_y, groups_z). No-op if no compute program
 * is current. Backs glDispatchCompute.
 */
void backend_dispatch_compute(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

/*
 * Indirect form (backs glDispatchComputeIndirect). Identical setup to
 * backend_dispatch_compute, but the group counts are fetched by the GPU from
 * `buffer` at `offset` — a {x,y,z} uint32 triple, bit-identical to
 * VkDispatchIndirectCommand, so nothing needs repacking. The buffer comes
 * from the GL_DISPATCH_INDIRECT_BUFFER binding.
 */
void backend_dispatch_compute_indirect(VkBuffer buffer, VkDeviceSize offset);

/*
 * Record a global execution/memory barrier (backs glMemoryBarrier). GL states
 * the ordering guarantee in terms of the barrier bits; Vulkan needs explicit
 * src/dst access masks, so the bits are widened to a conservative
 * ALL_COMMANDS -> ALL_COMMANDS VkMemoryBarrier. Ends the active render pass
 * first. `barriers` is the GL bitfield (GL_SHADER_STORAGE_BARRIER_BIT etc.).
 */
void backend_memory_barrier(GLbitfield barriers);

/* Release all Vulkan resources owned by a program (shader modules + pipelines +
 * descriptor set layout / pipeline layout / descriptor pool). */
void backend_delete_program_resources(GLuint program);

/*
 * GL sync-object backing (glFenceSync / glClientWaitSync). Implemented in
 * MG_Backend/DirectVulkan/Device.cpp. Every GPU submission is stamped with a
 * monotonic serial; a sync object records the serial at fence-creation time and
 * glClientWaitSync blocks on the owning frame slot's VkFence until that
 * submission has actually completed. This lets a persistent mapped-buffer ring
 * (Sodium's chunk uploads) wait for the GPU before recycling a region it may
 * still be reading.
 */
/* 已确定完成的水位线：所有序号 <= 返回值的提交都已在 GPU 上执行完毕。 */
uint64_t backend_last_completed_serial(void);
/* 已发出的 vkQueueSubmit 总数。glFenceSync 用 `本值 + 1` 记录"我要等下一次
 * 提交"，因为 fence 之前的命令还在尚未提交的命令缓冲里。 */
uint64_t backend_current_submit_serial(void);
/* 等待某序号完成。timeout_ns == 0 为非阻塞探测，UINT64_MAX 为无限等待。
 * 注意：若该序号尚未提交，本函数【不会阻塞】而是直接返回 false —— 没有任何
 * fence 会为一次还没发生的提交亮起。调用方需自行决定是否先 backend_commit()。 */
bool     backend_wait_serial(uint64_t serial, uint64_t timeout_ns);

/*
 * Reflect the vertex + fragment SPIR-V of `program` (via SPIRV-Cross), merge
 * the VS/FS binding sets, and build + cache a VkDescriptorSetLayout /
 * VkPipelineLayout / VkDescriptorPool on the program. Idempotent (runs once
 * per program, from inside backend_get_or_create_pipeline). On a program with
 * no reflected bindings the pipeline layout is left null and the pipeline
 * builder falls back to a process-wide empty layout.
 *
 *   vs / vs_words : vertex-stage SPIR-V words (may be NULL/0)
 *   fs / fs_words : fragment-stage SPIR-V words (may be NULL/0)
 */
void backend_ensure_program_layouts(GLuint program,
                                    const uint32_t* vs, int vs_words,
                                    const uint32_t* fs, int fs_words);

/*
 * Allocate a fresh VkDescriptorSet (from the program's pool, reset once per
 * frame), populate it from the current Program.uniforms + g_state->boundTextures,
 * and vkCmdBindDescriptorSets it onto the active command buffer. No-op when the
 * program has no descriptor bindings. Must be called after backend_bind_pipeline()
 * and before the draw, with a recording command buffer active.
 */
void backend_bind_program_descriptors(GLuint program);

/*
 * Present + acquire helpers used by eglSwapBuffers. EGL owns the swapchain;
 * these forward into the per-frame command submission. The EGL layer calls
 * backend_end_render_pass() + backend_commit() before backend_present().
 */
void backend_present_and_acquire(void* swapchain_state);

/*
 * Create the Vulkan surface + swapchain for a native window. Returns an opaque
 * pointer the EGL layer holds onto. The depth VkImage/View is created here
 * (Depth32Float + Stencil8 -> VK_FORMAT_D32_SFLOAT_S8_UINT).
 *   native_window      : platform-native window handle
 *                        - Apple:   CAMetalLayer* (bridged void*)
 *   width/height       : drawable size
 *   want_depth_stencil : 1 to allocate a depth/stencil image
 *   platform_hint      : 0 = auto-detect via compile-time platform; or
 *                        EGL_PLATFORM_SURFACELESS_MESA for explicit dispatch
 *                        (forward-looking; the current split implementation
 *                        routes by CMake-selected TU, so the value is taken
 *                        as a hint and may be ignored by the impl).
 */
void* backend_create_swapchain(void* native_window, int width, int height,
                               int want_depth_stencil, int platform_hint);
void  backend_destroy_swapchain(void* swapchain_state);

/* Acquire the next swapchain image and return its color VkImageView (plus the
 * depth VkImageView if allocated). Used by eglSwapBuffers / eglMakeCurrent to
 * install the per-frame attachments on the GLState. */
VkImageView backend_swapchain_acquire_color(void* swapchain_state);
VkImageView backend_swapchain_acquire_depth(void* swapchain_state);
int          backend_swapchain_width(void* swapchain_state);
int          backend_swapchain_height(void* swapchain_state);

/*
 * Return the VkImage + VkFormat of the currently-acquired swapchain color
 * image (the one most recently returned by backend_swapchain_acquire_color).
 * Used by glBlitFramebuffer / glReadPixels to reference the on-screen
 * drawable at the image level (vkCmdBlitImage / vkCmdCopyImageToBuffer need
 * a VkImage, not a VkImageView). Returns VK_NULL_HANDLE if no image is
 * currently acquired.
 */
VkImage     backend_swapchain_current_color_image(void* swapchain_state);
/*
 * Tracked layout of the ACTIVE swapchain's current colour image.
 *
 * glReadPixels on the default framebuffer must transition the swapchain image
 * using its REAL current layout as oldLayout. commit_frame() leaves the image
 * in PRESENT_SRC_KHR (CommandStream.cpp) and records that in
 * Swapchain::currentColorLayout; a readback that hardcodes
 * COLOR_ATTACHMENT_OPTIMAL as oldLayout is a spec violation, MoltenVK treats
 * the transition as a no-op, and vkCmdCopyImageToBuffer then reads the image
 * in the wrong layout and yields all-zero pixels.
 */
VkImageLayout backend_active_swapchain_color_layout(void);
void          backend_set_active_swapchain_color_layout(VkImageLayout layout);
VkFormat    backend_swapchain_color_format(void* swapchain_state);
VkImage     backend_swapchain_current_depth_image(void* swapchain_state);
VkFormat    backend_swapchain_depth_format(void* swapchain_state);

/*
 * FIX (P1 - GL 上限硬编码):
 * 查询真实的 VkPhysicalDeviceLimits，供 glGetIntegerv 汇报 GL_MAX_* 使用。
 *
 * 之前 Getter.cpp 把 GL_MAX_TEXTURE_SIZE 写死成 16384，而 A9/A10 这类老
 * iOS GPU 的上限只有 8192（由 Metal 2 的 GPUFamily 决定）。上报一个设备做
 * 不到的值，Sodium/Iris 会照着分配超大纹理或阴影贴图 → vkCreateImage 失败
 * → 纹理丢失甚至崩溃。宁可少报，绝不能多报。
 *
 * 同理 GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS 之前写死 80，却和内部的
 * kMaxTextureUnits=32 自相矛盾 —— 上层按 80 个单位绑定，后端数组只有 32
 * 个槽位，越界部分被静默丢弃。
 *
 * `which` 取下面的 MITHRIL_LIMIT_* 常量。后端未初始化或该项无对应 Vulkan
 * 上限时返回 fallback，调用方因此不需要额外判空。
 */
#define MITHRIL_LIMIT_MAX_TEXTURE_SIZE            1
#define MITHRIL_LIMIT_MAX_3D_TEXTURE_SIZE         2
#define MITHRIL_LIMIT_MAX_CUBE_MAP_TEXTURE_SIZE   3
#define MITHRIL_LIMIT_MAX_ARRAY_TEXTURE_LAYERS    4
#define MITHRIL_LIMIT_MAX_RENDERBUFFER_SIZE       5
#define MITHRIL_LIMIT_MAX_VIEWPORT_WIDTH          6
#define MITHRIL_LIMIT_MAX_VIEWPORT_HEIGHT         7
#define MITHRIL_LIMIT_MAX_TEXTURE_IMAGE_UNITS     8   /* per-stage sampled images */
#define MITHRIL_LIMIT_MAX_COMBINED_TEX_UNITS      9
#define MITHRIL_LIMIT_MAX_UNIFORM_BLOCK_SIZE      10
#define MITHRIL_LIMIT_UNIFORM_BUFFER_ALIGNMENT    11
#define MITHRIL_LIMIT_MAX_UNIFORM_BUFFER_BINDINGS 12
#define MITHRIL_LIMIT_MAX_COLOR_ATTACHMENTS       13
#define MITHRIL_LIMIT_MAX_SAMPLES                 14
#define MITHRIL_LIMIT_MAX_VERTEX_ATTRIBS          15
#define MITHRIL_LIMIT_MAX_SSBO_BINDINGS           16
#define MITHRIL_LIMIT_MAX_SSBO_SIZE               17
#define MITHRIL_LIMIT_MAX_COMPUTE_WG_INVOCATIONS  18
#define MITHRIL_LIMIT_MAX_COMPUTE_WG_COUNT_X      19
#define MITHRIL_LIMIT_MAX_COMPUTE_WG_SIZE_X       20

int backend_device_limit(int which, int fallback);

#ifdef __cplusplus
}
#endif

#endif // MITHRIL_BACKEND_H
