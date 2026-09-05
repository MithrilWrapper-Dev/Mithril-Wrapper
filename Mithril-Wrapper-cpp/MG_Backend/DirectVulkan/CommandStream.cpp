// Mithril-Wrapper - MG_Backend/DirectVulkan/CommandStream.cpp
// Render-pass orchestration via VK_KHR_dynamic_rendering (vkCmdBeginRendering)
// + encoder dynamic-state setters + draw recording + per-frame submit.
#include "CommandStream.h"
#include "Device.h"
#include "Swapchain.h"
#include "Resources.h"  // texture_table() / TextureEntry (root cause Y: FBO layout barriers)
#include "DescriptorSet.h"  // bind_program_descriptors (compute dispatch path)
#include "Pipeline.h"       // clear_all_pipeline_caches (OOM recovery)
#include "UniformArena.h"   // ubo_arena_rewind (per-frame transient UBO storage)
#include "../Backend.h"
#include "../../MG_Impl/Log.h"
#include "../../MG_State/State.h"  // g_state (for scissorTest in clear_attachments +
                                  //  root cause AG: currentBaseVertex/currentBaseInstance +
                                  //  root cause Z: viewportH fallback)

#include <cstring>
#include <vector>

// glMemoryBarrier bit tested by backend_memory_barrier. The bundled
// GL/glcorearb.h in include/ predates ARB_shader_image_load_store's token
// block, so the one value we actually branch on is spelled out locally
// (prefixed to avoid ever colliding with a future header update).
#define MG_GL_COMMAND_BARRIER_BIT 0x00000040

namespace mithril {
namespace vk {

namespace {

// Root cause AA (depth-only format pStencilAttachment, VUID-06126):
// Returns true if `fmt` has a stencil aspect. Used by begin_render_pass to
// decide whether to bind pStencilAttachment: dynamic-rendering REQUIRES the
// stencil attachment's ImageView to contain a stencil aspect, but depth-only
// formats (D32_SFLOAT / D16_UNORM) only have a depth aspect. Binding a
// depth-only view as a stencil attachment is a spec violation
// (VUID-VkRenderingInfo-pStencilAttachment-06126) and may cause MoltenVK to
// drop the draw -> black screen. Mirrors MobileGL VkRenderPassManager's
// aspect-based stencil-attachment decision.
bool format_has_stencil(VkFormat fmt) {
    switch (fmt) {
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_S8_UINT:
            return true;
        default:
            return false;
    }
}

// 根因 G (深度参考 MobileGL VulkanRenderer.cpp:193-198 ResolveColorClearAlpha):
// 若 swapchain 颜色格式无 alpha 通道（如 R8G8B8_UNORM / B8G8R8_UNORM），
// clear color 的 alpha 必须强制为 1.0。否则合成器（compositor）会将
// alpha=0 的窗口视为透明 → 显示窗口后面的内容（黑屏）。当前 swapchain 恒
// BGRA8 + compositeAlpha=OPAQUE 未触发，但此 helper 是防御性加固，对标
// MobileGL，防止未来引入 RGB swapchain 或降级 compositeAlpha 时黑屏。
bool format_has_alpha(VkFormat fmt) {
    switch (fmt) {
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_B8G8R8_UNORM:
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_B8G8R8_SRGB:
        case VK_FORMAT_R5G6B5_UNORM_PACK16:
        case VK_FORMAT_R16G16B16_UNORM:
        case VK_FORMAT_R16G16B16_SFLOAT:
        case VK_FORMAT_R32G32B32_SFLOAT:
            return false;
        default:
            return true;  // 所有 RGBA / RG / R 格式及未知格式视为有 alpha
    }
}

// Encoder state carried between begin_render_pass() and the draw calls.
struct EncoderState {
    bool passActive = false;
    VkPipeline boundPipeline = VK_NULL_HANDLE;
    // True once a valid VkDescriptorSet has been bound into the current
    // command buffer (set by bind_program_descriptors just before
    // vkCmdBindDescriptorSets). backend_draw_* requires it: a vkCmdDraw with
    // an unbound descriptor set makes MoltenVK sample undefined memory -> pure
    // red geometry, and on A11 can fault the GPU at the next submit.
    bool descriptorsBound = false;

    // Pending clear values (applied to the load op of the next pass).
    float clearColor[4] = {0, 0, 0, 0};
    double clearDepth = 1.0;
    GLint clearStencil = 0;
    bool loadClear = false;   // true = CLEAR (glClear), false = LOAD (draw pass)

    // Color/depth attachment views for the active pass.
    VkImageView colorViews[8] = {};
    int colorCount = 0;
    VkImageView depthView = VK_NULL_HANDLE;
    int width = 0;
    int height = 0;

    // The swapchain whose currently-acquired image backs framebuffer 0.
    // nullptr when no EGLSurface is current (headless) or the active FBO is
    // a user-created framebuffer object. Set by EGL after each acquire; read
    // by begin_render_pass() / commit_frame() to record layout barriers and
    // signal the per-image renderFinished semaphore.
    Swapchain* activeSwapchain = nullptr;

    // True once any command has been recorded into the current command buffer
    // since the last commit_frame() / vkBeginCommandBuffer. Used by
    // commit_frame() to skip empty submits (eglWaitClient + eglSwapBuffers
    // both call backend_commit; the second call would otherwise submit an
    // empty command buffer, which is wasteful and — under resize/destruction
    // races — can submit against a destroyed swapchain's semaphore, triggering
    // MoltenVK / IOSurface UAF crashes).
    bool hasCommands = false;

    // ---- B1 first-frame diagnostic: per-frame recorded draw count ----
    // Incremented in draw_recording_allowed whenever a vkCmdDraw* is actually
    // recorded, reset at each fresh command-buffer begin (frame boundary).
    // Read by eglSwapBuffers' B1 present log to distinguish "draws dropped"
    // (count 0) from "draws recorded but fragments not visible" (count > 0).
    uint32_t drawCount = 0;

    // ---- Root cause Y (CRITICAL): user-FBO attachment layout transitions ----
    // VK_KHR_dynamic_rendering's vkCmdBeginRendering does NOT auto-transition
    // attachment image layouts — it only validates that each image is in the
    // layout declared by VkRenderingAttachmentInfo.imageLayout. The swapchain
    // path barrier transitions are handled above (activeSwapchain block). User
    // FBO color/depth textures are created with currentLayout=UNDEFINED and
    // become SHADER_READ_ONLY_OPTIMAL after upload; without an explicit
    // barrier to COLOR_ATTACHMENT_OPTIMAL / DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    // the actual layout != declared layout -> spec violation -> MoltenVK
    // drops the draw -> black screen.
    //
    // MobileGL's VkRenderPassManager (VkRenderPassManager.cpp:711-784)
    // barriers ALL attachments before render pass begin.
    //
    // The GL draw path (Drawing.cpp) calls backend_set_fbo_attachment_tex_ids
    // right before backend_begin_render_pass for each non-swapchain
    // attachment. begin_render_pass looks up the TextureEntry via tex_id and
    // barriers its image to attachment-optimal; end_render_pass barriers it
    // back to a read-only layout and updates TextureEntry::currentLayout.
    // Cleared in end_render_pass.
    GLuint fboColorTexIds[8] = {};
    int    fboColorTexCount = 0;
    GLuint fboDepthTexId = 0;

    // ---- Root cause AA (HIGH): tracked depth format for this pass ----
    // Set in begin_render_pass from the swapchain depth (D32_SFLOAT_S8_UINT,
    // always has stencil) or the registered user-FBO depth TextureEntry
    // (could be depth-only D32_SFLOAT / D16_UNORM, no stencil). Used to gate
    // pStencilAttachment (VUID-06126). Reset to VK_FORMAT_UNDEFINED at the
    // start of each begin_render_pass so a previous pass's depth format does
    // not leak into a pass that has no depth attachment.
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;

    // ---- GL 4.3 ARB_invalidate_subdata: per-attachment discard flags ----
    // Set by glInvalidateFramebuffer/glInvalidateSubFramebuffer via
    // backend_set_invalidate_attachments. Applied to storeOp in the NEXT
    // begin_render_pass, then cleared (invalidation is one-shot per GL spec).
    // On TBDR GPUs (Apple Silicon), storeOp=DONT_CARE means the tile memory
    // does NOT need to be written back to system memory — critical for
    // reducing memory bandwidth and VRAM pressure (MobileGL uses the same
    // pattern via VkAttachmentDescription.storeOp).
    uint32_t invalidateColorMask = 0;  // bit i = color attachment i discard
    bool invalidateDepth = false;
    bool invalidateStencil = false;
};

EncoderState& encoder() {
    static EncoderState s;
    return s;
}

/*
 * Record an image-memory barrier transitioning `image` from `oldLayout` to
 * `newLayout` on the active command buffer. Used by begin_render_pass() /
 * commit_frame() to put swapchain images into COLOR_ATTACHMENT_OPTIMAL before
 * dynamic rendering and back into PRESENT_SRC_KHR before present. Without
 * these barriers MoltenVK sees a PRESENT_SRC image used as a colour
 * attachment, which is spec-illegal and produces a black screen.
 *
 *   isDepthStencil : selects the aspect mask (color vs depth+stencil) and
 *                    the destination pipeline stage.
 */
void record_layout_barrier(VkCommandBuffer cb, VkImage image, VkFormat format,
                           VkImageLayout oldLayout, VkImageLayout newLayout,
                           bool isDepthStencil) {
    if (image == VK_NULL_HANDLE || oldLayout == newLayout) return;

    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;

    // FIX (root cause Y): derive the aspect mask from the format so depth-only
    // images (D32_SFLOAT / D16_UNORM → DEPTH_BIT) and stencil-only (S8_UINT →
    // STENCIL_BIT) get the correct mask. The previous hard-coded
    // DEPTH_BIT | STENCIL_BIT was only correct for D24_UNORM_S8_UINT /
    // D32_SFLOAT_S8_UINT (depth+stencil packed formats) and would emit a
    // barrier whose aspectMask has no matching image aspect for depth-only
    // formats -> spec-illegal (VUID-VkImageMemoryBarrier-aspectMask-0120).
    // aspect_for_format returns COLOR_BIT for color formats (matches the old
    // !isDepthStencil branch), DEPTH_BIT | STENCIL_BIT for packed depth+
    // stencil (matches the old isDepthStencil branch for the swapchain depth),
    // and the correct single-aspect mask for depth-only / stencil-only.
    // `isDepthStencil` is retained only to gate the legacy swapchain call
    // sites' expectation; the aspect mask is now format-driven.
    (void)isDepthStencil;
    b.subresourceRange.aspectMask = aspect_for_format(format);
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = 1;

    // Source stage mask: who wrote the image in oldLayout.
    VkPipelineStageFlags srcStage;
    VkAccessFlags srcAccess;
    switch (oldLayout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            // No prior writes; contents discarded.
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            srcAccess = 0;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            // Image came back from present; the presentation engine read it.
            srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            srcAccess = VK_ACCESS_MEMORY_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            srcAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            srcStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            srcAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        // FIX (root cause Y): proper src stage/access for the read-only
        // layouts used by end_render_pass when transitioning user FBO
        // attachments back. Previously fell through to the default case
        // (ALL_COMMANDS_BIT + MEMORY_WRITE), which is valid but overly
        // conservative and slows the fragment-shader visibility tracking.
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            srcAccess = VK_ACCESS_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            srcStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            srcAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                        VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            srcAccess = VK_ACCESS_MEMORY_WRITE_BIT;
            break;
    }

    // Destination stage mask: who will read/write the image in newLayout.
    VkPipelineStageFlags dstStage;
    VkAccessFlags dstAccess;
    switch (newLayout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dstAccess = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            // Present engine reads the image.
            // FIX (root cause R): Use COLOR_ATTACHMENT_OUTPUT instead of
            // BOTTOM_OF_PIPE. BOTTOM_OF_PIPE is legal per spec but MoltenVK
            // historically maps it to a no-op barrier, which can cause the
            // color-attachment writes from the render pass to not be fully
            // visible to the present engine — the present then reads stale/
            // incomplete pixels (black screen). COLOR_ATTACHMENT_OUTPUT is
            // the stage where the presentation engine reads the image on
            // MoltenVK/Metal, and is the stage MobileGL uses for this
            // transition. This guarantees the color attachment writes complete
            // before present.
            dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dstAccess = VK_ACCESS_MEMORY_READ_BIT;
            break;
        // FIX (root cause Y): proper dst stage/access for the read-only
        // layouts used by end_render_pass when transitioning user FBO
        // attachments back. Color attachments become SHADER_READ_ONLY_OPTIMAL
        // (sampled from the fragment shader), depth attachments become
        // DEPTH_STENCIL_READ_ONLY_OPTIMAL (shadow-map sampling or depth
        // compare). Previously fell through to the default case
        // (ALL_COMMANDS_BIT + MEMORY_READ|MEMORY_WRITE).
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dstAccess = VK_ACCESS_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                        VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            dstAccess = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            break;
    }

    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;

    vkCmdPipelineBarrier(cb, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &b);
}

} // namespace

bool render_pass_active() { return encoder().passActive; }

// Accessors for the "valid descriptor set bound in the current command buffer"
// flag (see EncoderState::descriptorsBound). Defined here because encoder() is
// an anonymous-namespace object in this TU; DescriptorSet.cpp sets it via
// set_descriptors_bound() and backend_draw_* reads it via descriptors_bound().
void set_descriptors_bound(bool bound) { encoder().descriptorsBound = bound; }
bool descriptors_bound() { return encoder().descriptorsBound; }

/*
 * ---- Root cause AI (CRITICAL, SIGSEGV inside MVKRenderSubpass) ----
 * Last line of defence before any vkCmdDraw* is recorded.
 *
 * A draw is only legal inside a render-pass instance and with a graphics
 * pipeline bound. Violating either is undefined behaviour, and MoltenVK's
 * reaction is not a dropped draw but a null dereference: MVKCommandEncoder
 * lazily opens the Metal render pass on the first draw, and with no active
 * render pass its _renderPass is null, so
 *
 *   MVKRenderSubpass::populateMTLRenderPassDescriptor()
 *     MVKPixelFormats* pixFmts = _renderPass->getPixelFormats();
 *
 * faults on its very first member access. That is the observed iPhone X
 * crash (SIGSEGV at populateMTLRenderPassDescriptor+0x3c).
 *
 * The GL layer already refuses to draw when prepare_draw() fails
 * (Drawing.cpp), which is the real fix. This check is deliberately
 * redundant: it keeps a single missed guard — in existing paths such as the
 * indirect draws, or in any path added later — from turning a recoverable
 * pipeline failure into a process abort. The cost is two predictable
 * branches per draw.
 */
bool draw_recording_allowed(const char* who) {
    EncoderState& e = encoder();
    // FIX (VK_NOT_READY storm): verify the command buffer is actually recording
    // before allowing any draw. passActive can be stale-true after a deviceLost
    // recovery; recording vkCmdDraw into a non-recording buffer spams VK_NOT_READY.
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->commandBuffer || !b->commandBufferRecording) {
        return false;
    }
    if (!e.passActive) {
        static uint32_t warned = 0;
        if (warned < 8) {
            ++warned;
            MITHRIL_LOG_WARN("vk", "%s: no active render pass — draw dropped "
                                   "(recording it would crash MoltenVK). This "
                                   "means a pipeline/pass setup step failed "
                                   "earlier; see prior warnings.", who);
        }
        return false;
    }
    if (e.boundPipeline == VK_NULL_HANDLE) {
        static uint32_t warned = 0;
        if (warned < 8) {
            ++warned;
            MITHRIL_LOG_WARN("vk", "%s: no graphics pipeline bound — draw "
                                   "dropped (undefined behaviour otherwise). "
                                   "Pipeline creation most likely failed; see "
                                   "prior warnings.", who);
        }
        return false;
    }
    // FIX (GPU page fault / pure-red from frame 1): a vkCmdDraw is only legal
    // when a valid descriptor set is bound for the current command buffer.
    // bind_program_descriptors() can bail early (uniform-arena upload failure,
    // incomplete-set guard, descriptor-pool growth retry failure) AFTER the
    // pipeline is bound but BEFORE issuing vkCmdBindDescriptorSets. Recording
    // the draw anyway makes MoltenVK sample an unbound / garbage descriptor
    // set -> geometry renders pure red and, on A11's Metal 2, can fault the
    // GPU at the next vkQueueSubmit (kIOGPUCommandBufferCallbackErrorPageFault,
    // the exact crash in the log: clean wrapper logs, then the first submit
    // faults). Drop the draw instead: it is safer to miss one draw than to
    // submit one referencing undefined memory. The flag is reset at every
    // command-buffer boundary (on_command_buffer_boundary / fresh begin).
    if (!e.descriptorsBound) {
        static uint32_t warned = 0;
        if (warned < 8) {
            ++warned;
            MITHRIL_LOG_WARN("vk", "%s: no valid descriptor set bound in this "
                                   "command buffer — draw dropped (recording it "
                                   "would sample undefined descriptors -> red / "
                                   "GPU page fault). bind_program_descriptors "
                                   "most likely bailed earlier; see prior "
                                   "warnings.", who);
        }
        return false;
    }
    // B1 first-frame diagnostic: a real draw was recorded this frame.
    e.drawCount++;
    return true;
}

/*
 * Root cause Z: returns the active render pass's framebuffer height (the
 * attachment extent set in begin_render_pass and clamped to the swapchain /
 * actual drawable dimensions). Used by backend_set_viewport /
 * backend_set_scissor to convert GL bottom-origin Y to Vulkan top-origin Y
 * (vk_y = fbHeight - gl_y - gl_h). Returns 0 when no pass is active (callers
 * fall back to g_state->viewportH). Lives in this TU so it can read the
 * anonymous-namespace encoder() without exposing EncoderState publicly.
 */
int encoder_height_for_yflip() {
    EncoderState& e = encoder();
    return e.passActive ? e.height : 0;
}

void set_clear_color(float r, float g, float b, float a) {
    auto& e = encoder();
    e.clearColor[0] = r; e.clearColor[1] = g; e.clearColor[2] = b; e.clearColor[3] = a;
}
void set_clear_depth(double d) { encoder().clearDepth = d; }
void set_clear_stencil(int s)  { encoder().clearStencil = s; }
void set_load_clear(bool clear){ encoder().loadClear = clear; }

unsigned int backend_get_recorded_draws() { return encoder().drawCount; }

// GL 4.3 ARB_invalidate_subdata: mark attachments for discard (storeOp=DONT_CARE)
// in the next begin_render_pass. One-shot: cleared after begin_render_pass applies.
void set_invalidate_attachments(uint32_t color_mask, bool depth, bool stencil) {
    EncoderState& e = encoder();
    e.invalidateColorMask = color_mask;
    e.invalidateDepth = depth;
    e.invalidateStencil = stencil;
}

Swapchain* active_swapchain() { return encoder().activeSwapchain; }

void set_active_swapchain(Swapchain* sc) {
    encoder().activeSwapchain = sc;
}

/*
 * Root cause Y (CRITICAL): register the GL texture names backing the upcoming
 * user-FBO render pass's color/depth attachments. The GL draw path
 * (Drawing.cpp) calls this IMMEDIATELY before backend_begin_render_pass for
 * each non-swapchain attachment (i.e. whenever the bound FBO is not 0).
 *
 * begin_render_pass can only see VkImageView handles — it cannot reverse-
 * resolve them back to GL texture names, and therefore cannot look up the
 * TextureEntry (which holds the VkImage, VkFormat, and tracked currentLayout
 * needed to emit a valid layout barrier). This registration bridges that gap:
 * the GL layer passes the tex_ids here, and begin_render_pass uses them to
 * look up TextureEntry via texture_table() and barrier each attachment to
 * attachment-optimal before vkCmdBeginRendering (which only VALIDATES, never
 * transitions, the declared imageLayout — see the EncoderState comment for
 * the full root-cause mechanism).
 *
 * end_render_pass reads the same tex_ids to barrier the attachments back to
 * a read-only layout and update TextureEntry::currentLayout, then clears the
 * registration (auto-clear, no separate backend_clear_fbo_attachments call
 * needed — the GL layer just re-registers before the next user-FBO pass).
 *
 * Contract:
 *   - color_tex_ids may be null when color_count == 0 (depth-only FBO).
 *   - Tex_id 0 (or any tex_id not in texture_table()) is silently skipped.
 *   - For swapchain rendering (FBO 0) the GL layer does NOT call this — the
 *     swapchain path's barriers are handled by the activeSwapchain block in
 *     begin_render_pass / commit_frame.
 *   - color_count is clamped to 8 (kMaxColorAttachments).
 *   - depth_tex_id == 0 means no user-FBO depth attachment (the depth view
 *     may still be the swapchain's depthView, handled separately).
 */
void set_fbo_attachment_tex_ids(GLuint* color_tex_ids, int color_count,
                                GLuint depth_tex_id) {
    auto& e = encoder();
    // Clear any stale registration from a previous pass (defensive —
    // end_render_pass already clears, but a stray set without a matching
    // end_render_pass should not leak tex_ids into the next pass).
    for (int i = 0; i < 8; ++i) e.fboColorTexIds[i] = 0;
    e.fboColorTexCount = 0;
    e.fboDepthTexId = 0;

    int n = color_count > 8 ? 8 : (color_count < 0 ? 0 : color_count);
    for (int i = 0; i < n; ++i) {
        e.fboColorTexIds[i] = color_tex_ids ? color_tex_ids[i] : 0;
    }
    e.fboColorTexCount = n;
    e.fboDepthTexId = depth_tex_id;
}

/*
 * Ensure the current frame slot's command buffer is in the RECORDING state.
 * See the header comment for the full rationale. In short: with per-slot
 * command buffers, after commit_frame() submits slot N and advances to
 * slot N+1, the next recording operation must switch the alias to slot N+1's
 * buffer, wait on its fence, reset it, and begin it. This function does all
 * of that lazily.
 *
 * This is the FIX for the black-screen-with-sound root cause: the old code
 * used a single shared command buffer and did vkResetCommandBuffer at the
 * end of commit_frame() — resetting a buffer the GPU was still executing,
 * which is Vulkan spec UB. The per-slot design + lazy ensure eliminates this
 * by never resetting a pending buffer.
 */
bool ensure_command_buffer_recording() {
    Backend* b = backend();
    if (!b->initialized) return false;
    if (b->commandBufferRecording) return true;  // fast path
    if (b->deviceLost) return false;
    if (!b->commandPool) return false;

    // Wait on the current slot's fence if a submit is pending on it. After
    // the wait, this slot's command buffer is no longer pending and can be
    // safely reset. The fence was created with VK_FENCE_CREATE_SIGNALED_BIT,
    // so the very first frame's wait (fencePending=false) is skipped.
    if (b->fencePending[b->currentFrame]) {
        VkFence fence = b->frameFences[b->currentFrame];
        VkResult wr = vkWaitForFences(b->device, 1, &fence, VK_TRUE, UINT64_MAX);
        if (wr != VK_SUCCESS) {
            // FIX (日志刷屏): 限流 — 首次 + 每 100 次打印一条
            static int waitFailCount = 0;
            waitFailCount++;
            if (waitFailCount <= 3 || waitFailCount % 100 == 0) {
                MITHRIL_LOG_ERROR("vk", "ensure_command_buffer_recording: "
                                  "vkWaitForFences(slot=%d) failed (rc=%d, "
                                  "fail #%d) — possible device lost",
                                  b->currentFrame, (int)wr, waitFailCount);
            }
            b->deviceLost = true;
            return false;
        }
        b->fencePending[b->currentFrame] = false;
        // The fence wait guarantees all GPU work submitted to this slot has
        // completed. Any resources deferred to this slot's disposal queue
        // (glDeleteBuffers / glBufferData orphan / glDeleteTextures from the
        // frame that last used this slot) are now safe to actually destroy.
        // Without this drain, the deferred VkBuffer/VkImage handles would
        // leak until the slot is reused again (kMaxFramesInFlight later),
        // and more critically, the drain would happen too late — the
        // disposal queue would accumulate unboundedly during rapid buffer
        // churn (e.g. MC's per-frame uniform buffer updates).
        drain_disposal_queue(b->currentFrame);
    }

    // Switch the alias to the current slot's buffer and reset+begin it.
    b->commandBuffer = b->commandBuffers[b->currentFrame];
    vkResetCommandBuffer(b->commandBuffer, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(b->commandBuffer, &bi) != VK_SUCCESS) {
        // FIX (日志刷屏): vkBeginCommandBuffer 持续失败时每帧都会调用此路径。
        // 限流：首次 + 每 100 次打印一条。持续失败通常意味着设备已挂起，
        // deviceLost 会在后续的 vkWaitForFences/vkQueueSubmit 失败时被置位。
        static int beginFailCount = 0;
        beginFailCount++;
        if (beginFailCount <= 3 || beginFailCount % 100 == 0) {
            MITHRIL_LOG_ERROR("vk", "ensure_command_buffer_recording: "
                              "vkBeginCommandBuffer failed (slot=%d, fail #%d)",
                              b->currentFrame, beginFailCount);
        }
        b->commandBufferRecording = false;
        // 持续失败时标记 deviceLost，避免无限重试刷屏
        b->deviceLost = true;
        return false;
    }
    b->commandBufferRecording = true;
    encoder().hasCommands = false;  // fresh buffer, no commands yet
    encoder().drawCount = 0;        // B1: new frame, draw counter starts at 0

    // FIX (Invalid Resource 根因 - per-frame transient staging arena rewind):
    // 到达这里意味着 command buffer 被重置+重新 begin（新帧开始）。
    // 之前的 fence wait（或 safe_device_wait_idle 的 vkDeviceWaitIdle）保证了
    // 该 slot 的所有 GPU 工作已完成，staging buffer 的数据不再被引用。
    // rewind offset 到 0，让本帧的纹理上传从 arena 头部重新 sub-allocate。
    // 参考 MobileGL TryDrainFrameTransients 的 transient arena rewind。
    if (b->frameStagingReady) {
        b->frameStagingOffset[b->currentFrame] = 0;
    }

    /* Same rewind, same justification, for the transient UNIFORM arena.
     *
     * This is the one place it is legal: the fence wait above proves every
     * command buffer ever submitted on this slot has finished executing, so
     * no in-flight draw can still be reading the bytes we are about to hand
     * out again. Rewinding anywhere else (e.g. at commit_frame time) would
     * recycle memory the GPU has not finished with. */
    ubo_arena_rewind(b->currentFrame);

    /* FIX (mid-frame flush 缓存失效 - P0): 到达这里意味着当前 slot 的 arena
     * 刚被 rewind（staging + UBO）。在"真帧边界"这无害 —— descMemo / UBO
     * plan 本来就会在下一帧通过 frameGeneration 检查作废。但在帧中间发生
     * flush（safe_device_wait_idle / drain_and_detach_swapchain 提交了当前
     * command buffer 后重新 begin）时，本帧前半段分配 arena 切片写成的
     * descriptor set 与 plan.lastOffset 仍指向这些刚被 rewind 的字节 ——
     * 后续同帧 draw 复用它们会读到被覆盖的数据。bump flushGeneration 让
     * bind_program_descriptors 作废这些缓存（见 DescriptorSet.cpp）。 */
    b->flushGeneration++;

    /* A freshly begun command buffer has no descriptor sets bound, so the
     * bind-dedup shadow inside DescriptorSet.cpp must be dropped — otherwise
     * the first draw of the frame would "recognise" a binding that only
     * existed in the previous buffer and skip a vkCmdBindDescriptorSets it
     * genuinely needs. */
    on_command_buffer_boundary();

    /* Root cause AI: pipeline bindings are command-buffer scoped. A reset +
     * re-begun buffer has no pipeline bound, so the tracking handle must be
     * cleared here or backend_draw_* would wrongly believe one is live. */
    encoder().boundPipeline = VK_NULL_HANDLE;
    // A freshly begun command buffer has no descriptor set bound either.
    // Reset the flag so backend_draw_* refuses a draw until bind_program_
    // descriptors() actually binds a set into this buffer.
    encoder().descriptorsBound = false;

    // FIX (GPU page fault root cause — re-entrant mid-frame purge): this is a
    // genuine command-buffer boundary: the buffer was just flushed (fence
    // waited above or via safe_device_wait_idle) and re-begun EMPTY — no render
    // pass open, no descriptor set bound. Any purge deferred earlier by
    // request_purge() (OOM / critical-pressure GC while a draw was mid-record)
    // is now safe to run: it cannot invalidate a set/pipeline the buffer
    // references. Run it before the first draw of the new buffer.
    process_pending_purge();

    return true;
}

void begin_render_pass(VkImageView* color_views, int color_count,
                       VkImageView depth_view, int width, int height, int samples) {
    (void)samples;
    Backend* b = backend();
    if (!b->initialized || !b->commandBuffer) return;
    EncoderState& e = encoder();
    if (e.passActive) return;  // coalesce draws into one pass

    // Root cause AA: reset the tracked depth format so a previous pass's
    // format does not leak into this pass. Set below when the swapchain or
    // user-FBO depth attachment is identified.
    e.depthFormat = VK_FORMAT_UNDEFINED;

    // Ensure the current frame slot's command buffer is in the recording state.
    // After commit_frame() submits slot N and advances to slot N+1, the
    // alias b->commandBuffer is stale (points at the pending slot N buffer).
    // This call lazily switches to slot N+1's buffer, waits on its fence,
    // resets it, and begins it. On the very first frame, the buffer is
    // already recording (begun by init_device), so this is a no-op.
    if (!ensure_command_buffer_recording()) {
        return;  // device lost or begin failed — skip this pass
    }

    // The command buffer is already in the recording state — either begun by
    // init_device() (first frame) or by commit_frame() (subsequent frames).
    // Pre-frame commands (layout transitions, texture uploads, etc.) recorded
    // before this point are preserved and will be submitted with this pass.
    // Do NOT reset the command buffer here; that would discard those records.

    // Record the per-frame attachments so draw commands can reference them.
    e.colorCount = color_count > 8 ? 8 : color_count;
    for (int i = 0; i < e.colorCount; ++i) e.colorViews[i] = color_views ? color_views[i] : VK_NULL_HANDLE;
    e.depthView = depth_view;
    e.width = width;
    e.height = height;
    int origW = width, origH = height;  // for clamp diagnostic log

    // FIX (IOSurfaceBindAccel SIGSEGV): Clamp the render area to the
    // swapchain image dimensions when rendering to FBO 0 (swapchain-bound).
    //
    // The GL viewport / eglDefaultWidth can exceed the swapchain image size
    // when GLFW or the host app sets a window size that differs from the
    // CAMetalLayer's drawableSize at swapchain creation time. On MoltenVK/iOS,
    // the IOSurface backing the swapchain image has the drawableSize, NOT the
    // GL viewport size. If renderArea.extent > IOSurface dimensions,
    // IOSurfaceBindAccel dereferences out-of-bounds memory → SIGSEGV.
    //
    // This mirrors MobileGL's VkRenderPassManager (VkRenderPassManager.cpp:760),
    // which clamps the render area to min(attachment dimensions, requested area).
    //
    // DOUBLE CLAMP: clamp to BOTH the swapchain creation-time size (sc->width)
    // AND the actual drawable size (sc->actualDrawableWidth). The drawable
    // size is updated by EGL after each acquire and reflects the ACTUAL
    // IOSurface dimensions (which may differ from the swapchain's imageExtent
    // if the drawableSize changed after swapchain creation). This is the
    // critical fix for the crash where GLFW resized the window between
    // swapchain creation and the first frame: the swapchain was created at
    // 2204x1696, but the drawable shrank to 1752x1696, and the render area
    // of 2204x1696 exceeded the 1752x1696 IOSurface → SIGSEGV.
    if (e.activeSwapchain) {
        Swapchain* sc = e.activeSwapchain;
        bool swapchainBound = false;
        for (int i = 0; i < e.colorCount; ++i) {
            if (sc->currentImage >= 0 && sc->currentImage < (int)sc->views.size() &&
                e.colorViews[i] == sc->views[sc->currentImage]) {
                swapchainBound = true;
                break;
            }
        }
        if (swapchainBound) {
            // Primary clamp: swapchain creation-time extent (VkImage size).
            if (e.width > sc->width) e.width = sc->width;
            if (e.height > sc->height) e.height = sc->height;
            // Secondary clamp: actual drawable size (IOSurface size at acquire
            // time). This catches the case where drawableSize shrank after
            // swapchain creation — the IOSurface is smaller than the VkImage.
            if (sc->actualDrawableWidth > 0 && e.width > sc->actualDrawableWidth)
                e.width = sc->actualDrawableWidth;
            if (sc->actualDrawableHeight > 0 && e.height > sc->actualDrawableHeight)
                e.height = sc->actualDrawableHeight;
        }
        // Diagnostic: log when the render area was clamped (helps verify the
        // IOSurfaceBindAccel fix is active). One-shot to avoid flooding the
        // log every frame (begin_render_pass is called per draw pass).
        if (e.width != origW || e.height != origH) {
            static bool clampedOnce = false;
            if (!clampedOnce) {
                clampedOnce = true;
                MITHRIL_LOG_WARN("vk", "begin_render_pass: clamped renderArea "
                                  "%dx%d -> %dx%d (swapchain=%dx%d, drawable=%dx%d)",
                                  origW, origH, e.width, e.height,
                                  sc->width, sc->height,
                                  sc->actualDrawableWidth, sc->actualDrawableHeight);
            }
        }
    }

    // ---- Layout barriers for the swapchain-backed default framebuffer ----
    // dynamic-rendering hard-codes imageLayout = COLOR_ATTACHMENT_OPTIMAL in
    // the VkRenderingAttachmentInfo below. The swapchain image comes back from
    // acquire in PRESENT_SRC_KHR (or UNDEFINED on first use); without an
    // explicit barrier transitioning it to COLOR_ATTACHMENT_OPTIMAL, MoltenVK
    // sees an illegal layout and renders nothing (black screen). The depth
    // image is created with initialLayout = UNDEFINED and needs the same
    // one-shot transition to DEPTH_STENCIL_ATTACHMENT_OPTIMAL on first use.
    //
    // swapchainColorWasUndefined: set when the colour image was transitioned
    // out of UNDEFINED this frame. Used below to pick DONT_CARE for the load
    // op (LOAD on an image whose contents were discarded is wasteful and
    // spec-discouraged; DONT_CARE matches the discard semantics).
    bool swapchainColorWasUndefined = false;
    bool swapchainDepthWasUndefined = false;
    // FIX (root cause — user-FBO depth first use): Minecraft renders the
    // loading screen / main menu / world geometry into USER FBOs with their
    // OWN depth texture (only the final composite goes to FBO 0). Those depth
    // textures are created with initialLayout=UNDEFINED and are NEVER cleared
    // to "far" on first use (see Resources.cpp:1407, Resources.h:58). The
    // swapchain-only far-init (44772c4) did not cover them, so a first-use
    // user-FBO depth was loaded as garbage (near/0) and every GL_LESS fragment
    // failed -> pure red from frame 1. Mirror the swapchain one-shot below.
    bool fboDepthWasUndefined = false;
    if (e.activeSwapchain) {
        Swapchain* sc = e.activeSwapchain;
        if (sc->currentImage >= 0 && sc->currentImage < (int)sc->views.size()) {
            // Only barrier the image if one of the bound colour attachments is
            // the swapchain's current view (i.e. we're rendering to FBO 0).
            bool swapchainBound = false;
            for (int i = 0; i < e.colorCount; ++i) {
                if (e.colorViews[i] == sc->views[sc->currentImage]) {
                    swapchainBound = true;
                    break;
                }
            }
            if (swapchainBound && sc->currentColorLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
                swapchainColorWasUndefined = (sc->currentColorLayout == VK_IMAGE_LAYOUT_UNDEFINED);
                record_layout_barrier(b->commandBuffer,
                                      sc->images[sc->currentImage], sc->format,
                                      sc->currentColorLayout,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                      /*isDepthStencil=*/false);
                sc->currentColorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
        }
        // Depth: one-shot UNDEFINED -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
        if (sc->depthImage != VK_NULL_HANDLE && sc->depthView != VK_NULL_HANDLE &&
            e.depthView == sc->depthView && !sc->depthLayoutInitialized) {
            record_layout_barrier(b->commandBuffer,
                                  sc->depthImage, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                  /*isDepthStencil=*/true);
            sc->depthLayoutInitialized = true;
            swapchainDepthWasUndefined = true;
        }
        // Root cause AA: track the swapchain depth format so the
        // pStencilAttachment decision below uses the real format. The
        // swapchain depth is always D32_SFLOAT_S8_UINT (has stencil), so
        // pStencilAttachment will be bound — same as before this fix.
        if (e.depthView == sc->depthView && e.depthView != VK_NULL_HANDLE) {
            e.depthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
        }
    }

    // ---- Root cause Y (CRITICAL): user-FBO attachment layout barriers ----
    // vkCmdBeginRendering does NOT auto-transition attachment image layouts.
    // It only VALIDATES that each image is in the layout declared by
    // VkRenderingAttachmentInfo.imageLayout. The swapchain path above
    // barriers the swapchain color/depth. User FBO color/depth textures
    // (registered via set_fbo_attachment_tex_ids by the GL draw path) are
    // created with currentLayout=UNDEFINED and become
    // SHADER_READ_ONLY_OPTIMAL after upload; without an explicit barrier
    // here, the actual layout != declared COLOR_ATTACHMENT_OPTIMAL /
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> spec violation -> MoltenVK drops
    // the draw -> black screen.
    //
    // Mirrors MobileGL VkRenderPassManager (VkRenderPassManager.cpp:711-784),
    // which barriers ALL attachments before render pass begin.
    //
    // The `isDepthStencil` parameter is now format-driven inside
    // record_layout_barrier (aspect_for_format); we pass true for the depth
    // attachment only to retain the legacy semantic of "depth-stencil stages"
    // for the srcStage/dstStage computation. record_layout_barrier itself
    // ignores the parameter for aspect-mask selection (see its comment).
    if (e.fboColorTexCount > 0 || e.fboDepthTexId != 0) {
        auto& tbl = texture_table();
        // Color attachments: barrier each registered tex_id (index-aligned
        // with e.colorViews[i]) to COLOR_ATTACHMENT_OPTIMAL. Skip indices
        // where the tex_id is 0 (unbound slot) or where the bound view is
        // the swapchain's current view (already barriered above).
        VkImageView swapchainView = VK_NULL_HANDLE;
        if (e.activeSwapchain && e.activeSwapchain->currentImage >= 0 &&
            e.activeSwapchain->currentImage < (int)e.activeSwapchain->views.size()) {
            swapchainView = e.activeSwapchain->views[e.activeSwapchain->currentImage];
        }
        for (int i = 0; i < e.fboColorTexCount && i < e.colorCount; ++i) {
            GLuint tex_id = e.fboColorTexIds[i];
            if (tex_id == 0) continue;
            // Skip if this slot is the swapchain color attachment (already
            // barriered above by the activeSwapchain block). This guards
            // against a misregistration where the GL layer passes the
            // swapchain tex_id (which is not in texture_table anyway, but
            // be defensive).
            if (swapchainView != VK_NULL_HANDLE && e.colorViews[i] == swapchainView) continue;
            auto it = tbl.find(tex_id);
            if (it == tbl.end()) continue;
            TextureEntry& tex = it->second;
            if (tex.image == VK_NULL_HANDLE) continue;
            if (tex.currentLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) continue;
            record_layout_barrier(b->commandBuffer,
                                  tex.image, tex.format,
                                  tex.currentLayout,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  /*isDepthStencil=*/false);
            tex.currentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        // Depth attachment: barrier to DEPTH_STENCIL_ATTACHMENT_OPTIMAL and
        // record the format for the pStencilAttachment decision (root cause AA).
        // Skip if the bound depth view is the swapchain's depth view (already
        // barriered above). The swapchain depth case sets e.depthFormat in
        // the activeSwapchain block above; the user-FBO case sets it here.
        if (e.fboDepthTexId != 0 && e.depthView != VK_NULL_HANDLE) {
            bool isSwapchainDepth = (e.activeSwapchain &&
                                     e.depthView == e.activeSwapchain->depthView);
            if (!isSwapchainDepth) {
                auto it = tbl.find(e.fboDepthTexId);
                if (it != tbl.end()) {
                    TextureEntry& tex = it->second;
                    if (tex.image != VK_NULL_HANDLE &&
                        tex.currentLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                        // Capture whether this is the depth's ONE-SHOT first use
                        // (UNDEFINED) so the loadOp below can CLEAR it to far.
                        // Must be read before the barrier overwrites the layout.
                        fboDepthWasUndefined =
                            (tex.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);
                        record_layout_barrier(b->commandBuffer,
                                              tex.image, tex.format,
                                              tex.currentLayout,
                                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                              /*isDepthStencil=*/true);
                        tex.currentLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    }
                    // Root cause AA: capture the user-FBO depth format so
                    // the pStencilAttachment decision below uses the real
                    // format. For depth-only formats (D32_SFLOAT / D16_UNORM)
                    // format_has_stencil returns false -> pStencilAttachment
                    // = nullptr (VUID-06126 compliance). For D24_UNORM_S8_UINT /
                    // D32_SFLOAT_S8_UINT it returns true -> pStencilAttachment
                    // = &depthAttach (preserves the existing swapchain behavior).
                    if (tex.format != VK_FORMAT_UNDEFINED) {
                        e.depthFormat = tex.format;
                    }
                }
            }
        }
    }

    // Begin dynamic rendering.
    // loadOp selection — aligned with MobileGL's VkRenderPassManager
    // (VkRenderPassManager.cpp:711-784). Priority order (hasClear first):
    //   1. hasClear (e.loadClear)  -> CLEAR   (discard prior contents)
    //   2. trackedLayout == UNDEFINED -> DONT_CARE (no valid contents to load;
    //      LOAD on UNDEFINED is spec-illegal and wastes tile bandwidth)
    //   3. otherwise               -> LOAD    (preserve existing contents)
    // MobileGL sets initialLayout=UNDEFINED for cases 1 & 2; we achieve the
    // same via the acquire->attachment barrier using oldLayout=UNDEFINED
    // (swapchain_acquire_color deliberately resets currentColorLayout to
    // UNDEFINED on every acquire, since post-present contents are undefined).
    VkRenderingAttachmentInfoKHR colorAttachs[8] = {};
    for (int i = 0; i < e.colorCount; ++i) {
        colorAttachs[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
        colorAttachs[i].imageView = e.colorViews[i];
        colorAttachs[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Color loadOp (MobileGL VkRenderPassManager.cpp:713,776-780):
        //   hasClear -> CLEAR; !hasClear && UNDEFINED -> DONT_CARE; else LOAD.
        // swapchainColorWasUndefined covers the swapchain image on its first
        // pass of the frame (currentColorLayout was UNDEFINED before the
        // acquire->attachment barrier). Subsequent passes in the same frame
        // see COLOR_ATTACHMENT_OPTIMAL and correctly use LOAD to preserve the
        // first pass's output.
        if (e.loadClear) {
            colorAttachs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        } else if (swapchainColorWasUndefined &&
                   e.activeSwapchain &&
                   e.colorViews[i] == e.activeSwapchain->views[e.activeSwapchain->currentImage]) {
            colorAttachs[i].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        } else {
            colorAttachs[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        }
        colorAttachs[i].storeOp = (e.invalidateColorMask & (1u << i))
                                  ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                                  : VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachs[i].clearValue.color.float32[0] = e.clearColor[0];
        colorAttachs[i].clearValue.color.float32[1] = e.clearColor[1];
        colorAttachs[i].clearValue.color.float32[2] = e.clearColor[2];
        // 根因 G: 若该 attachment 是 swapchain image 且格式无 alpha，强制 alpha=1.0
        // （对标 MobileGL ResolveColorClearAlpha），防止合成器视窗口透明 → 黑屏。
        bool attachHasAlpha = true;
        if (e.activeSwapchain &&
            e.colorViews[i] == e.activeSwapchain->views[e.activeSwapchain->currentImage]) {
            attachHasAlpha = format_has_alpha(e.activeSwapchain->format);
        }
        colorAttachs[i].clearValue.color.float32[3] = attachHasAlpha ? e.clearColor[3] : 1.0f;
    }
    // Depth/stencil loadOp (MobileGL ResolveDepthStencilAttachmentLoadInfo,
    // VkRenderPassManager.cpp:140-155). Same priority: hasClear -> CLEAR;
    // UNDEFINED (first use) -> DONT_CARE; else LOAD. The depth image is
    // persistent across frames (never presented), so after the one-shot
    // UNDEFINED->DEPTH_STENCIL_ATTACHMENT_OPTIMAL transition it stays valid
    // and subsequent passes use LOAD to preserve depth across passes.
    // NOTE: glClear now uses backend_clear_attachments (vkCmdClearAttachments)
    // to clear only the requested aspects, NOT loadOp=CLEAR. The loadClear
    // flag here only affects the initial loadOp of the pass, and glClear
    // sets it to LOAD (not CLEAR) before calling begin_render_pass.
    VkRenderingAttachmentInfoKHR depthAttach{};
    depthAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    depthAttach.imageView = e.depthView;
    depthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    // Clear value for the depth/stencil attachment. Honour the host's
    // glClearDepth for an explicit pass-start clear, EXCEPT on the ONE-SHOT
    // first use of the persistent swapchain depth buffer (UNDEFINED layout)
    // where we force far(1.0)/0 to initialise the buffer.
    //
    // ROOT CAUSE (systemic pure-red from frame 1): the persistent swapchain
    // depth buffer is created with initialLayout=UNDEFINED and was NEVER
    // initialized to "far" on its first use. With the old LOAD_OP_DONT_CARE
    // the depth buffer holds garbage (typically 0 / near) on the first
    // depth-tested draw; with the default depth func GL_LESS every fragment
    // (depth in (0,1]) compares against garbage==near and FAILS, so only the
    // swapchain clear color (red) survives -> pure red screen from the very
    // first frame (loading screen, main menu, in-game all red with sound).
    //
    // MobileGL initializes the swapchain depth to far on first use. Fix:
    // on the one-shot UNDEFINED->DEPTH_STENCIL_ATTACHMENT_OPTIMAL first use,
    // CLEAR the depth buffer to far (1.0) instead of DONT_CARE so the first
    // depth-tested draw's fragments (depth < 1.0) pass the GL_LESS compare.
    depthAttach.clearValue.depthStencil.depth = (float)e.clearDepth;
    depthAttach.clearValue.depthStencil.stencil = (uint32_t)e.clearStencil;
    // FIX (user-FBO depth first use): mirror the swapchain one-shot — if this
    // pass is the very first use of a freshly-created (UNDEFINED) depth buffer
    // — either the swapchain's persistent depth OR a user FBO's depth texture —
    // clear it to far(1.0)/0 so the first GL_LESS draw's fragments pass.
    const bool depthWasUndefined = swapchainDepthWasUndefined || fboDepthWasUndefined;
    if (depthWasUndefined) {
        depthAttach.clearValue.depthStencil.depth = 1.0f;
        depthAttach.clearValue.depthStencil.stencil = 0u;
    }
    if (e.loadClear) {
        depthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    } else if (depthWasUndefined) {
        depthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    } else {
        depthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    }
    depthAttach.storeOp = (e.invalidateDepth || e.invalidateStencil)
                          ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                          : VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfoKHR ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    ri.renderArea.offset.x = 0;
    ri.renderArea.offset.y = 0;
    // Use e.width/e.height (clamped to swapchain dimensions above) instead of
    // the raw caller-provided width/height. This ensures renderArea never
    // exceeds the swapchain image / IOSurface dimensions.
    ri.renderArea.extent.width = (uint32_t)e.width;
    ri.renderArea.extent.height = (uint32_t)e.height;
    ri.layerCount = 1;
    ri.colorAttachmentCount = (uint32_t)e.colorCount;
    ri.pColorAttachments = e.colorCount > 0 ? colorAttachs : nullptr;
    ri.pDepthAttachment = e.depthView ? &depthAttach : nullptr;
    // Root cause AA (HIGH, VUID-VkRenderingInfo-pStencilAttachment-06126):
    // pStencilAttachment's ImageView MUST contain a stencil aspect. For
    // depth-only formats (D32_SFLOAT / D16_UNORM) the view's aspect is
    // DEPTH_BIT only, so binding it as a stencil attachment is a spec
    // violation that may cause MoltenVK to drop the draw -> black screen.
    // Bind pStencilAttachment only when the depth format actually has a
    // stencil aspect (D24_UNORM_S8_UINT / D32_SFLOAT_S8_UINT / S8_UINT).
    // e.depthFormat is set above from the swapchain depth
    // (always D32_SFLOAT_S8_UINT, has stencil — preserves existing
    // swapchain behavior) or the registered user-FBO depth TextureEntry.
    // When no depth attachment is bound (e.depthView == null) or the depth
    // format is depth-only, pStencilAttachment is null.
    ri.pStencilAttachment = (e.depthView && format_has_stencil(e.depthFormat))
                            ? &depthAttach : nullptr;

    // Resolve the dynamic-rendering entry point (Vulkan 1.2 + extension).
    static PFN_vkCmdBeginRenderingKHR fn = nullptr;
    if (!fn) {
        fn = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(b->device, "vkCmdBeginRendering");
        if (!fn) fn = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(b->device, "vkCmdBeginRenderingKHR");
    }
    if (fn) fn(b->commandBuffer, &ri);

    e.passActive = true;
    e.hasCommands = true;  // begin_render_pass recorded real commands
    e.loadClear = false;  // subsequent passes within the frame use LOAD
    // GL 4.3 ARB_invalidate_subdata: invalidation is one-shot — clear after
    // applying so the next pass uses default STORE (unless re-invalidated).
    e.invalidateColorMask = 0;
    e.invalidateDepth = false;
    e.invalidateStencil = false;
}

void end_render_pass() {
    Backend* b = backend();
    EncoderState& e = encoder();
    // FIX (VK_NOT_READY storm): only call vkCmdEndRendering when the command
    // buffer is recording. If passActive is stale-true after a deviceLost
    // (commit_frame returned early without end_render_pass) and the buffer
    // is not recording, calling vkCmdEndRendering spams VK_NOT_READY.
    // Clear passActive regardless so the encoder state is consistent.
    if (!e.passActive) return;
    if (!b->commandBuffer || !b->commandBufferRecording) {
        e.passActive = false;
        return;
    }

    static PFN_vkCmdEndRenderingKHR fn = nullptr;
    if (!fn) {
        fn = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(b->device, "vkCmdEndRendering");
        if (!fn) fn = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(b->device, "vkCmdEndRenderingKHR");
    }
    if (fn) fn(b->commandBuffer);

    // ---- Root cause Y (CRITICAL): barrier user-FBO attachments back to ----
    // ---- read-only layouts and update TextureEntry::currentLayout.      --
    // vkCmdEndRendering leaves color attachments in COLOR_ATTACHMENT_OPTIMAL
    // and depth attachments in DEPTH_STENCIL_ATTACHMENT_OPTIMAL. For the
    // swapchain color this is fixed up by commit_frame's PRESENT_SRC_KHR
    // barrier; for the swapchain depth the layout stays at
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL across frames (one-shot transition,
    // never presented). For user FBO attachments, the texture is typically
    // sampled right after the FBO render completes (post-process pass,
    // shadow-map read, etc.) — it MUST be back in a read-only layout
    // (SHADER_READ_ONLY_OPTIMAL for color, DEPTH_STENCIL_READ_ONLY_OPTIMAL
    // for depth-stencil) before the next draw that samples it, otherwise
    // the descriptor's declared layout won't match the actual layout and
    // MoltenVK drops the sampling draw -> black screen.
    //
    // MobileGL's VkRenderPassManager barriers all attachments back to their
    // read-only layouts at render pass end (VkRenderPassManager.cpp:711-784).
    //
    // Use format_has_stencil to pick the right read-only layout for the depth
    // attachment: depth-stencil formats need DEPTH_STENCIL_READ_ONLY_OPTIMAL
    // (sampling depth or stencil, with depth-stencil aspect), depth-only
    // formats need SHADER_READ_ONLY_OPTIMAL (depth-compare sampler reads use
    // the depth aspect as a sampled image, layout is SHADER_READ_ONLY_OPTIMAL
    // for non-stencil depth textures).
    if (e.fboColorTexCount > 0 || e.fboDepthTexId != 0) {
        auto& tbl = texture_table();
        for (int i = 0; i < e.fboColorTexCount; ++i) {
            GLuint tex_id = e.fboColorTexIds[i];
            if (tex_id == 0) continue;
            auto it = tbl.find(tex_id);
            if (it == tbl.end()) continue;
            TextureEntry& tex = it->second;
            if (tex.image == VK_NULL_HANDLE) continue;
            // Use sampled_layout_for_format (the SAME helper DescriptorSet.cpp
            // and Resources.cpp use) so the post-pass layout exactly matches
            // the descriptor's declared imageLayout. A mismatch here would
            // re-introduce the black screen (root cause Y/AH interaction):
            // end_render_pass transitions to layout X, but the descriptor
            // declares layout Y -> MoltenVK drops the sampling draw.
            VkImageLayout readLayout = sampled_layout_for_format(tex.format);
            if (tex.currentLayout == readLayout) continue;
            record_layout_barrier(b->commandBuffer,
                                  tex.image, tex.format,
                                  tex.currentLayout,
                                  readLayout,
                                  /*isDepthStencil=*/false);
            tex.currentLayout = readLayout;
        }
        if (e.fboDepthTexId != 0) {
            auto it = tbl.find(e.fboDepthTexId);
            if (it != tbl.end()) {
                TextureEntry& tex = it->second;
                if (tex.image != VK_NULL_HANDLE) {
                    // Depth attachment read-only layout MUST match what
                    // sampled_layout_for_format returns (used by
                    // DescriptorSet.cpp for the descriptor's imageLayout and
                    // by Resources.cpp for the post-upload transition).
                    // depth-stencil formats -> DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    // depth-only formats -> DEPTH_READ_ONLY_OPTIMAL.
                    // Using a different layout here would mismatch the
                    // descriptor and drop the sampling draw (root cause AH).
                    VkImageLayout readLayout = sampled_layout_for_format(tex.format);
                    if (tex.currentLayout != readLayout) {
                        record_layout_barrier(b->commandBuffer,
                                              tex.image, tex.format,
                                              tex.currentLayout,
                                              readLayout,
                                              /*isDepthStencil=*/true);
                        tex.currentLayout = readLayout;
                    }
                }
            }
        }
        // Auto-clear the registration so a stale tex_id cannot leak into
        // the next pass (the GL layer re-registers before each user-FBO
        // begin_render_pass; for swapchain passes it leaves the
        // registration empty).
        for (int i = 0; i < 8; ++i) e.fboColorTexIds[i] = 0;
        e.fboColorTexCount = 0;
        e.fboDepthTexId = 0;
    }

    e.passActive = false;
}

/*
 * Clear specific aspects of the current framebuffer via vkCmdClearAttachments.
 * Must be called inside a render pass. This is the correct implementation of
 * glClear: it clears ONLY the buffers specified by `mask`, unlike the old
 * loadOp=CLEAR approach which cleared ALL attachments regardless of mask.
 *
 * MobileGL (VulkanRenderer.cpp:4230-4358) uses the same vkCmdClearAttachments
 * approach, respecting GL_SCISSOR_TEST for the clear rect.
 */
/*
 * Resolve the rectangle a clear applies to. GL's scissor test clips clears,
 * so an enabled scissor box wins; otherwise the whole render area is cleared.
 *
 * Everything is clamped to the render pass's effective dimensions
 * (e.width/e.height, themselves already clamped to the swapchain image in
 * begin_render_pass). A clear rect larger than the attachment violates
 * VUID-vkCmdClearAttachments-pRects-00016 and takes MoltenVK's
 * IOSurfaceBindAccel down with a SIGSEGV on iOS — so this clamp is load
 * bearing, not defensive tidiness.
 */
static VkClearRect compute_clear_rect(const EncoderState& e) {
    VkClearRect rect{};
    if (mithril::g_state && mithril::g_state->scissorTest) {
        int32_t sx = (int32_t)mithril::g_state->scissorX;
        int32_t sy = (int32_t)mithril::g_state->scissorY;
        int32_t sw = (int32_t)mithril::g_state->scissorW;
        int32_t sh = (int32_t)mithril::g_state->scissorH;
        if (sx < 0) { sw += sx; sx = 0; }
        if (sy < 0) { sh += sy; sy = 0; }
        if (sx + sw > e.width)  sw = e.width - sx;
        if (sy + sh > e.height) sh = e.height - sy;
        if (sw < 0) sw = 0;
        if (sh < 0) sh = 0;
        rect.rect.offset.x = sx;
        rect.rect.offset.y = sy;
        rect.rect.extent.width = (uint32_t)sw;
        rect.rect.extent.height = (uint32_t)sh;
    } else {
        rect.rect.offset.x = 0;
        rect.rect.offset.y = 0;
        rect.rect.extent.width = (uint32_t)e.width;
        rect.rect.extent.height = (uint32_t)e.height;
    }
    rect.baseArrayLayer = 0;
    rect.layerCount = 1;
    return rect;
}

void clear_attachments(uint32_t mask, int x, int y, int w, int h) {
    Backend* b = backend();
    EncoderState& e = encoder();
    // FIX (VK_NOT_READY storm): guard against recording into a command buffer
    // that is not in the RECORDING state. passActive can be stale-true after
    // a deviceLost recovery that did not reset the encoder, or a mid-frame
    // deviceLost where commit_frame returned early without end_render_pass.
    // Without this check vkCmdClearAttachments records into a non-recording
    // buffer -> MoltenVK spams VK_NOT_READY.
    if (!b->commandBuffer || !b->commandBufferRecording || !e.passActive) return;
    if (mask == 0) return;

    // Build the VkClearAttachment array for the requested aspects.
    std::vector<VkClearAttachment> attaches;
    VkClearValue cv{};

    if (mask & GL_COLOR_BUFFER_BIT) {
        cv.color.float32[0] = e.clearColor[0];
        cv.color.float32[1] = e.clearColor[1];
        cv.color.float32[2] = e.clearColor[2];
        // 根因 G: 若当前 render pass 的某个 color attachment 是 swapchain image
        // 且格式无 alpha，强制 clear alpha=1.0（对标 MobileGL ResolveColorClearAlpha）。
        // 对非 swapchain attachment（用户 FBO 纹理）原样使用 clearColor[3]。
        // 这里用 activeSwapchain 格式做统一判断（MRT 中 swapchain 通常是 attachment 0）。
        bool clearHasAlpha = !e.activeSwapchain || format_has_alpha(e.activeSwapchain->format);
        cv.color.float32[3] = clearHasAlpha ? e.clearColor[3] : 1.0f;
        // One clear attachment per color attachment (VK_IMAGE_ASPECT_COLOR_BIT
        // covers all color aspects, but per-attachment is safer with MRT).
        for (int i = 0; i < e.colorCount; ++i) {
            VkClearAttachment a{};
            a.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            a.colorAttachment = (uint32_t)i;
            a.clearValue = cv;
            attaches.push_back(a);
        }
    }
    if (mask & GL_DEPTH_BUFFER_BIT) {
        VkClearAttachment a{};
        a.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        a.clearValue.depthStencil.depth = (float)e.clearDepth;
        attaches.push_back(a);
    }
    if (mask & GL_STENCIL_BUFFER_BIT) {
        VkClearAttachment a{};
        a.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        a.clearValue.depthStencil.stencil = (uint32_t)e.clearStencil;
        attaches.push_back(a);
    }
    if (attaches.empty()) return;

    VkClearRect rect = compute_clear_rect(e);
    vkCmdClearAttachments(b->commandBuffer,
                          (uint32_t)attaches.size(), attaches.data(),
                          1, &rect);
    e.hasCommands = true;
}

/*
 * glClearBuffer{fv,iv,uiv,fi} — clear ONE attachment with an explicit value
 * (root cause AP).
 *
 * clear_attachments() above always clears every colour attachment using the
 * context-wide glClearColor. That is right for glClear(), but glClearBuffer*
 * targets a single draw buffer with a value passed at the call site — which
 * is how a deferred renderer wipes just its normal or velocity target between
 * passes. Without this entry point those calls did nothing at all, leaving
 * the previous frame's G-buffer contents to bleed through.
 *
 * `drawbuffer` indexes the colour attachment for GL_COLOR and must be 0 for
 * the depth/stencil targets.
 */
void clear_buffer_indexed(uint32_t buffer, int drawbuffer,
                          const float color[4], float depth, uint32_t stencil) {
    Backend* b = backend();
    EncoderState& e = encoder();
    // FIX (VK_NOT_READY storm): same commandBufferRecording guard as
    // clear_attachments — prevents recording into a non-recording buffer.
    if (!b->commandBuffer || !b->commandBufferRecording || !e.passActive) return;

    VkClearAttachment a{};
    switch (buffer) {
        case GL_COLOR:
            if (drawbuffer < 0 || drawbuffer >= e.colorCount) return;
            a.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            a.colorAttachment = (uint32_t)drawbuffer;
            a.clearValue.color.float32[0] = color[0];
            a.clearValue.color.float32[1] = color[1];
            a.clearValue.color.float32[2] = color[2];
            // Same swapchain-alpha rule as clear_attachments: a format
            // without alpha must be cleared to opaque or the compositor
            // blends the whole frame away.
            a.clearValue.color.float32[3] =
                (!e.activeSwapchain || format_has_alpha(e.activeSwapchain->format))
                ? color[3] : 1.0f;
            break;
        case GL_DEPTH:
            if (!e.depthView) return;
            a.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            a.clearValue.depthStencil.depth = depth;
            break;
        case GL_STENCIL:
            if (!e.depthView) return;
            a.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
            a.clearValue.depthStencil.stencil = stencil;
            break;
        case GL_DEPTH_STENCIL:
            if (!e.depthView) return;
            a.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            a.clearValue.depthStencil.depth = depth;
            a.clearValue.depthStencil.stencil = stencil;
            break;
        default:
            return;
    }

    VkClearRect rect = compute_clear_rect(e);
    vkCmdClearAttachments(b->commandBuffer, 1, &a, 1, &rect);
    e.hasCommands = true;
}

void commit_frame() {
    Backend* b = backend();
    if (b->deviceLost) {
        // 持久性故障已挂起，跳过 submit 避免死循环刷屏
        return;
    }
    if (!b->initialized || !b->commandBuffer) return;
    EncoderState& e = encoder();
    if (e.passActive) end_render_pass();

    // ---- Ensure the current slot's command buffer is recording ----
    // This is the async-pipeline wait point (mirrors MobileGL's
    // FrameContext::WaitAndAcquireNextImage, FrameContext.cpp:220, which
    // calls vkWaitForFences on the slot's imageInFlightFence before acquire).
    //
    // With per-slot command buffers, after the previous commit_frame()
    // submitted slot N and advanced to slot N+1, the alias b->commandBuffer
    // still points at slot N's pending buffer. This call lazily switches to
    // the current slot's buffer, waits on its fence (submitted
    // kMaxFramesInFlight frames ago — the deferred async wait that gives the
    // GPU kMaxFramesInFlight-1 frames of latency), resets it, and begins it.
    //
    // If the buffer is already recording (e.g. this is the first commit of
    // the process, or a previous commit's shouldSubmit was false so the
    // buffer was never submitted), this is a fast no-op.
    //
    // This replaces the old single-buffer design's "wait at top + reset+begin
    // at bottom" pattern, which reset a pending buffer (spec UB) at the bottom.
    if (!ensure_command_buffer_recording()) {
        return;  // device lost or begin failed
    }

    // ---- Decide whether we need to submit at all ----
    // MobileGL's Present() logic (VulkanRenderer.cpp:6912):
    //   shouldSubmit = hasCommandBufferRecorded || needsLayoutTransitionForPresent
    //
    // We MUST submit (and signal renderFinished) whenever:
    //   (a) commands were recorded this frame (normal draw pass), OR
    //   (b) the swapchain image is NOT already in PRESENT_SRC_KHR (it is in
    //       COLOR_ATTACHMENT_OPTIMAL from a previous render pass and needs a
    //       layout-transition barrier before present), OR
    //   (c) imageAvailable has not been consumed yet this frame (the very
    //       first commit after acquire — even with no draw commands, we must
    //       wait on imageAvailable so the GPU does not race ahead of the
    //       presentation engine; this is the MobileGL TransitionToPresent
    //       fallback path).
    //
    // Without this, a frame where the app only calls glClear (no draws) or a
    // frame where eglWaitClient already flushed the draws would skip submit,
    // leave renderFinished unsignaled, and present would wait on a semaphore
    // that is never signaled -> MoltenVK hangs / black screen.
    Swapchain* sc = e.activeSwapchain;
    bool hasCommands = e.hasCommands;
    bool needsLayoutTransition = false;
    bool needsImageAvailableWait = false;
    if (sc && sc->currentImage >= 0 && sc->currentImage < (int)sc->images.size()) {
        // FIX (root cause N): present requires the image to be in
        // PRESENT_SRC_KHR layout. The old code EXCLUDED UNDEFINED from the
        // transition check, assuming begin_render_pass would always transition
        // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL first. But when the swapchain
        // is created lazily (eglMakeCurrent failed because the window wasn't
        // sized yet, so eglSwapBuffers creates it), the first frame has NO
        // draw commands — the image stays UNDEFINED and is presented directly,
        // which is spec-illegal and crashes MoltenVK's IOSurfaceBindAccel
        // (SIGSEGV) because the IOSurface was never properly bound as a
        // render target. Now we transition to PRESENT_SRC_KHR whenever the
        // layout is not already PRESENT_SRC_KHR (UNDEFINED -> PRESENT_SRC is
        // a legal, content-discarding barrier).
        if (sc->currentColorLayout != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
            needsLayoutTransition = true;
        }
        needsImageAvailableWait = !sc->imageAvailableConsumed;
    }
    bool shouldSubmit = hasCommands || needsLayoutTransition || needsImageAvailableWait;

    if (!shouldSubmit) {
        return;
    }

    // FIX (root cause S): If we are about to present but NO draw commands were
    // recorded this frame (hasCommands=false), the swapchain image was never
    // used as a render target. On MoltenVK/iOS, the IOSurface backing the
    // swapchain image is lazily bound the first time the image is used as a
    // color attachment in a render pass. If we present without ever rendering
    // to it, MoltenVK presents a drawable whose IOSurface was never bound →
    // the Metal driver's IOSurfaceBindAccel dereferences an uninitialized
    // IOSurface → SIGSEGV or silent black screen.
    //
    // This happens on the first frame when the swapchain is created lazily
    // (eglMakeCurrent failed because the window wasn't sized, so eglSwapBuffers
    // creates the swapchain — but the app already finished rendering with no
    // color attachment, so no draw commands were recorded).
    //
    // Fix: if we need to present (needsLayoutTransition or needsImageAvailableWait)
    // but have no draw commands, insert a minimal dynamic-rendering pass that
    // touches the swapchain color attachment. This forces MoltenVK to bind the
    // IOSurface as a render target, making the subsequent present safe.
    // MobileGL avoids this because its TransitionToPresent path still records
    // a real command buffer with a layout barrier, and traditional VkRenderPass
    // ensures attachment binding happens during render pass begin.
    if (sc && !hasCommands && (needsLayoutTransition || needsImageAvailableWait)) {
        // Insert a dummy render pass: BeginRendering with DONT_CARE loadOp,
        // no draws, EndRendering. This is enough to trigger IOSurface binding
        // in MoltenVK without clearing or modifying the image contents.
        if (sc->currentImage >= 0 && sc->currentImage < (int)sc->views.size() &&
            sc->views[sc->currentImage] != VK_NULL_HANDLE) {
            // Ensure the image is in COLOR_ATTACHMENT_OPTIMAL first (if it
            // isn't already). The layout transition below (needsLayoutTransition)
            // will handle COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR.
            if (sc->currentColorLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
                sc->currentColorLayout != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
                record_layout_barrier(b->commandBuffer,
                                      sc->images[sc->currentImage], sc->format,
                                      sc->currentColorLayout,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                      /*isDepthStencil=*/false);
                sc->currentColorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
            // Minimal render pass to trigger IOSurface binding.
            // FIX (first-frame priming): use CLEAR with black instead of
            // DONT_CARE. DONT_CARE leaves the image contents undefined, which
            // can show garbage pixels on the first frame before any draws are
            // recorded. CLEAR ensures a clean black frame. MobileGL primes the
            // first swapchain image in Initialize() for the same reason.
            VkRenderingAttachmentInfoKHR dummyAttach{};
            dummyAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
            dummyAttach.imageView = sc->views[sc->currentImage];
            dummyAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            dummyAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            dummyAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            dummyAttach.clearValue.color.float32[0] = 0.0f;
            dummyAttach.clearValue.color.float32[1] = 0.0f;
            dummyAttach.clearValue.color.float32[2] = 0.0f;
            dummyAttach.clearValue.color.float32[3] = 1.0f;
            VkRenderingInfoKHR dummyRI{};
            dummyRI.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
            dummyRI.renderArea.offset.x = 0;
            dummyRI.renderArea.offset.y = 0;
            // FIX: clamp renderArea to min(swapchain extent, actual drawable
            // size). The swapchain extent (sc->width) may exceed the actual
            // IOSurface dimensions (sc->actualDrawableWidth) when drawableSize
            // changed after swapchain creation. An oversized renderArea causes
            // IOSurfaceBindAccel SIGSEGV (out-of-bounds IOSurface access).
            int dummyW = sc->width;
            int dummyH = sc->height;
            if (sc->actualDrawableWidth > 0 && sc->actualDrawableWidth < dummyW)
                dummyW = sc->actualDrawableWidth;
            if (sc->actualDrawableHeight > 0 && sc->actualDrawableHeight < dummyH)
                dummyH = sc->actualDrawableHeight;
            dummyRI.renderArea.extent.width = (uint32_t)dummyW;
            dummyRI.renderArea.extent.height = (uint32_t)dummyH;
            dummyRI.layerCount = 1;
            dummyRI.colorAttachmentCount = 1;
            dummyRI.pColorAttachments = &dummyAttach;
            static PFN_vkCmdBeginRenderingKHR beginFn = nullptr;
            static PFN_vkCmdEndRenderingKHR endFn = nullptr;
            if (!beginFn) {
                beginFn = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(b->device, "vkCmdBeginRendering");
                if (!beginFn) beginFn = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(b->device, "vkCmdBeginRenderingKHR");
            }
            if (!endFn) {
                endFn = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(b->device, "vkCmdEndRendering");
                if (!endFn) endFn = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(b->device, "vkCmdEndRenderingKHR");
            }
            if (beginFn) beginFn(b->commandBuffer, &dummyRI);
            if (endFn) endFn(b->commandBuffer);
            // After the dummy pass, the image is in COLOR_ATTACHMENT_OPTIMAL.
            // The needsLayoutTransition block below will transition it to
            // PRESENT_SRC_KHR for present.
            needsLayoutTransition = true;
        }
    }

    // The command buffer is guaranteed to be in the recording state here —
    // ensure_command_buffer_recording() at the top of this function took care
    // of it (including the fence wait, reset, and begin). The old defensive
    // recovery block that was here has been removed because it reset the
    // buffer WITHOUT waiting on the fence first, which could reset a pending
    // buffer (spec UB) under the per-slot design.

    // Transition the swapchain color image back to PRESENT_SRC_KHR before
    // vkEndCommandBuffer so vkQueuePresentKHR sees a legal layout. Without
    // this, present is spec-illegal and MoltenVK drops the frame (black screen).
    if (sc && needsLayoutTransition) {
        record_layout_barrier(b->commandBuffer,
                              sc->images[sc->currentImage], sc->format,
                              sc->currentColorLayout,
                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                              /*isDepthStencil=*/false);
        sc->currentColorLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    VkResult r = vkEndCommandBuffer(b->commandBuffer);
    if (r != VK_SUCCESS) {
        // FIX (日志刷屏): 限流 — 首次 + 每 100 次打印一条
        static int endFailCount = 0;
        endFailCount++;
        if (endFailCount <= 3 || endFailCount % 100 == 0) {
            MITHRIL_LOG_ERROR("vk", "vkEndCommandBuffer failed (rc=%d, fail #%d)",
                              (int)r, endFailCount);
        }
        // Buffer is now in an invalid state. Force reset+begin so the next
        // begin_render_pass has a recording buffer to write into; otherwise
        // the render thread spins forever issuing vkCmd* into a dead buffer.
        vkResetCommandBuffer(b->commandBuffer, 0);
        VkCommandBufferBeginInfo rbi{};
        rbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        rbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(b->commandBuffer, &rbi) == VK_SUCCESS) {
            b->commandBufferRecording = true;
        }
        // This reset+begin bypasses ensure_command_buffer_recording(), so the
        // descriptor bind shadow has to be dropped here too: it names a set
        // bound into a buffer that no longer exists, and believing it would
        // make the next draw skip a vkCmdBindDescriptorSets it needs.
        on_command_buffer_boundary();
        e.hasCommands = false;
        if (sc) sc->needsRebuild = true;
        // 持续失败时标记 deviceLost，避免无限重试刷屏
        b->deviceLost = true;
        return;
    }
    b->commandBufferRecording = false;  // executable, not recording

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &b->commandBuffer;

    // ---- Wait on imageAvailable (acquire semaphore) if not yet consumed ----
    // This is the CRITICAL fix for the black screen: vkAcquireNextImageKHR
    // signals imageAvailable, and the FIRST vkQueueSubmit of the frame MUST
    // wait on it (at COLOR_ATTACHMENT_OUTPUT stage) so the GPU does not start
    // writing to the swapchain image before the presentation engine releases
    // it. Without this wait, the GPU renders into a stale/owned-by-presenter
    // image and the rendered contents never reach the display -> black screen.
    //
    // Only the first commit_frame() per frame waits on imageAvailable (mid-
    // frame flushes via eglWaitClient consume it on the first submit). This
    // mirrors MobileGL's imageAvailableSemaphoreConsumed flag
    // (FrameContext.cpp:191-193).
    VkSemaphore waitSemaphore = VK_NULL_HANDLE;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (sc && sc->imageAvailable != VK_NULL_HANDLE && !sc->imageAvailableConsumed) {
        waitSemaphore = sc->imageAvailable;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &waitSemaphore;
        si.pWaitDstStageMask = &waitStage;
    }

    // ---- Signal renderFinished so present can wait on it ----
    // Signal the per-image render-finished semaphore for the image we just
    // rendered into. This mirrors MobileGL's GetSubmitInfo (FrameContext.cpp:196),
    // which signals m_swapchainImageRenderFinishedSemaphores[swapchainImageIndex].
    //
    // Per-image (not per-frame-slot or per-swapchain) signaling is essential:
    // present must wait on the exact semaphore signaled by the submit that
    // rendered into THIS image. A single shared semaphore races under triple
    // buffering (image A's submit signals it, then image B's submit re-signals
    // before present consumes A's signal → black screen or spec violation).
    //
    // Only signal if not already signaled for THIS image this frame (a binary
    // semaphore cannot be re-signaled while still signaled). renderFinished
    // SignaledPerImage[currentImage] is cleared by swapchain_present_and_acquire
    // after present consumes the signal.
    VkSemaphore signalSemaphore = VK_NULL_HANDLE;
    if (sc && sc->currentImage >= 0 &&
        (size_t)sc->currentImage < sc->renderFinishedPerImage.size() &&
        sc->renderFinishedPerImage[sc->currentImage] != VK_NULL_HANDLE &&
        (size_t)sc->currentImage < sc->renderFinishedSignaledPerImage.size() &&
        !sc->renderFinishedSignaledPerImage[sc->currentImage]) {
        signalSemaphore = sc->renderFinishedPerImage[sc->currentImage];
        sc->renderFinishedSignaledPerImage[sc->currentImage] = true;
    }
    if (signalSemaphore != VK_NULL_HANDLE) {
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &signalSemaphore;
    }

    VkFence fence = b->frameFences[b->currentFrame];
    vkResetFences(b->device, 1, &fence);
    r = vkQueueSubmit(b->graphicsQueue, 1, &si, fence);
    if (r != VK_SUCCESS) {
        // FIX (rendering suspended 根因): 彻底重新设计 submit 失败处理。
        //
        // 原实现：连续 3 次 submit 失败 → deviceLost=true → "rendering suspended"
        // 这导致 OOM 时渲染被永久挂起，即使后续显存已释放也无法恢复。
        //
        // 新策略（参考 MobileGL TryDrainFrameTransients）：
        // - VK_ERROR_OUT_OF_DATE_KHR / VK_SUBOPTIMAL_KHR → 重建 swapchain，不挂起
        // - VK_ERROR_DEVICE_LOST → 设置 deviceLost（真正的设备丢失，需要重建）
        // - VK_ERROR_OUT_OF_DEVICE_MEMORY → 触发 OOM GC，跳过当前帧，不挂起
        // - 其他错误 → 跳过当前帧，不挂起
        //
        // 关键改变：OOM 不再导致永久挂起。每帧 OOM 时触发 GC 释放资源，
        // 下一帧重试。只有真正的 VK_ERROR_DEVICE_LOST 才设置 deviceLost，
        // 且 deviceLost 可被 EGL 恢复路径清除。
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
            // swapchain 过期：标记重建，不计数，不挂起
            if (sc) sc->needsRebuild = true;
        } else if (r == VK_ERROR_DEVICE_LOST) {
            // 真正的设备丢失：设置 deviceLost，让 EGL 恢复路径处理
            b->deviceLost = true;
            static int deviceLostLogCount = 0;
            deviceLostLogCount++;
            if (deviceLostLogCount <= 3 || deviceLostLogCount % 100 == 0) {
                MITHRIL_LOG_ERROR("vk", "vkQueueSubmit returned VK_ERROR_DEVICE_LOST "
                                  "(occurrence #%d) — deviceLost set, EGL will "
                                  "attempt recovery", deviceLostLogCount);
            }
        } else {
            // OOM 或其他错误：在 Metal 上 OOM 经常意味着 GPU 已 fault
            //（MoltenVK 的 "Caused GPU Address Fault Error"）。
            // 旧代码只做 GC 不设 deviceLost，但 vkDeviceWaitIdle 后设备可能
            // 已半死状态 — 后续 vkBeginCommandBuffer 成功但所有 vkCmd* 报
            // VK_NOT_READY。现在也设 deviceLost，让 EGL 走恢复路径
            //（purge + rebuild swapchain），而不是在半死设备上继续渲染。
            b->consecutiveSubmitFailures++;
            b->deviceLost = true;  // FIX: OOM on Metal = likely faulted
            static int submitFailCount = 0;
            submitFailCount++;
            if (submitFailCount <= 3 || submitFailCount % 100 == 0) {
                MITHRIL_LOG_ERROR("vk", "vkQueueSubmit failed (rc=%d, occurrence "
                                  "#%d) — setting deviceLost (OOM on Metal likely "
                                  "means GPU fault), triggering recovery",
                                  (int)r, submitFailCount);
            }
            // OOM 主动 GC：等待 GPU 完成 + 释放所有延迟资源
            if (b->device) {
                vkDeviceWaitIdle(b->device);
            }
            drain_all_disposal_queues();
            clear_all_pipeline_caches();
            reset_all_descriptor_pools();
        }
        // vkQueueSubmit failure (e.g. VK_ERROR_OUT_OF_DEVICE_MEMORY /
        // VK_ERROR_DEVICE_LOST) means the command buffer was NOT consumed.
        // The fence will NOT be signaled, so we must NOT wait on it below
        // (vkWaitForFences would hang forever). Reset the fence manually and
        // mark the swapchain dead so EGL rebuilds it on the next swap.
        vkResetFences(b->device, 1, &fence);
        // Roll back the semaphore state: neither imageAvailable nor
        // renderFinished was actually consumed/signal'd (submit failed), so
        // the next commit must be allowed to wait/signal again. Without
        // these rollbacks, the state would be inconsistent (e.g. next submit
        // would skip the imageAvailable wait, racing the presenter again).
        if (sc) {
            // Roll back the per-image render-finished signal flag for the
            // image we tried (and failed) to render into, so the next
            // commit_frame() can signal it again.
            if (sc->currentImage >= 0 &&
                (size_t)sc->currentImage < sc->renderFinishedSignaledPerImage.size()) {
                sc->renderFinishedSignaledPerImage[sc->currentImage] = false;
            }
            // imageAvailableConsumed stays false (we never set it true on
            // failure) — correct, the next submit should still wait.
            sc->needsRebuild = true;
        }
        // Reset+begin so the next frame has a recording buffer.
        // FIX (VK_NOT_READY storm): When deviceLost is true (OOM / DEVICE_LOST
        // paths above), skip vkResetCommandBuffer + vkBeginCommandBuffer — the
        // device is in an error state and vkBeginCommandBuffer will likely fail
        // or produce a buffer that can't accept commands. Instead, clear the
        // encoder state (passActive, boundPipeline, etc.) and set
        // commandBufferRecording=false so that:
        //   1. end_render_pass won't call vkCmdEndRendering on a non-recording
        //      buffer (passActive=false → early return)
        //   2. draw_recording_allowed won't allow vkCmdDraw (commandBufferRecording
        //      =false → returns false)
        //   3. The recovery path (backend_reset_device_lost → reset_encoder_state)
        //      will handle the fresh vkBeginCommandBuffer after the device is
        //      actually recovered
        if (!b->deviceLost) {
            vkResetCommandBuffer(b->commandBuffer, 0);
            VkCommandBufferBeginInfo rbi{};
            rbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            rbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(b->commandBuffer, &rbi) == VK_SUCCESS) {
                b->commandBufferRecording = true;
            }
        } else {
            // Device is lost: clear encoder state to prevent stale passActive
            // from causing VK_NOT_READY errors on subsequent vkCmd* calls.
            e.passActive = false;
            e.boundPipeline = VK_NULL_HANDLE;
            e.hasCommands = false;
            b->commandBufferRecording = false;
        }
        // This reset+begin bypasses ensure_command_buffer_recording(), so the
        // descriptor bind shadow has to be dropped here too: it names a set
        // bound into a buffer that no longer exists, and believing it would
        // make the next draw skip a vkCmdBindDescriptorSets it needs.
        on_command_buffer_boundary();
        e.hasCommands = false;
        return;
    }

    // Success: a clean submit clears the consecutive-failure counter (only
    // persistent faults should keep it climbing toward the deviceLost threshold).
    b->consecutiveSubmitFailures = 0;

    // Submit succeeded: imageAvailable is now consumed (the wait was honored).
    if (sc) {
        sc->imageAvailableConsumed = true;
    }

    // Mark this slot's fence as pending. The NEXT commit_frame() that cycles
    // back to this slot (kMaxFramesInFlight frames later) will wait on
    // frameFences[currentFrame] via ensure_command_buffer_recording() before
    // reusing this slot's command buffer. This must be set BEFORE the
    // currentFrame advance below, so the flag lands on the slot we just
    // submitted (not the next one). Without this, the deferred wait in
    // ensure_command_buffer_recording() is a no-op (fencePending is always
    // false) and the next reset of this slot's buffer would reset a buffer
    // the GPU is still executing -> spec violation / UAF crash. The fences
    // start signaled (VK_FENCE_CREATE_SIGNALED_BIT), so the first frame's
    // wait is correctly skipped (flag starts false).
    b->fencePending[b->currentFrame] = true;

    // Stamp this submission with a monotonic serial so GL sync objects
    // (glFenceSync / glClientWaitSync) can tell when it has actually completed
    // on the GPU (see Device.cpp backend_wait_serial). Must run before the
    // currentFrame advance below, so the serial is pinned to the slot we just
    // submitted.
    backend_frame_serial_advance(b->currentFrame);

    // CRITICAL FIX: do NOT vkResetCommandBuffer here.
    //
    // The old single-buffer code reset+begin the command buffer at this point
    // so that inter-frame texture uploads (stage_and_copy_image) had a
    // recording buffer to write into. But with a single shared buffer, this
    // reset hit a buffer the GPU was still executing (spec UB) — the root
    // cause of the black-screen-with-sound issue.
    //
    // With per-slot command buffers, we advance to the NEXT slot's buffer and
    // leave the just-submitted slot's buffer alone (pending on the GPU). The
    // next slot's buffer is lazily reset+begin by ensure_command_buffer_recording()
    // when the next begin_render_pass / stage_and_copy_image / commit_frame
    // needs it. The lazy ensure waits on the next slot's fence (submitted
    // kMaxFramesInFlight frames ago) before resetting, so no pending buffer is
    // ever reset.
    //
    // The alias b->commandBuffer is NOT updated here — it still points at the
    // just-submitted (pending) buffer. ensure_command_buffer_recording() will
    // update it to point at the new slot's buffer on the next call. Code that
    // records into b->commandBuffer between now and the next ensure call MUST
    // call ensure_command_buffer_recording() first (stage_and_copy_image,
    // transition_image_layout, bind_program_descriptors).
    b->commandBufferRecording = false;  // just-submitted buffer is no longer recording
    b->currentFrame = (b->currentFrame + 1) % kMaxFramesInFlight;
    // Monotonic generation bump: descriptor pools are reset on first draw of
    // each generation (see DescriptorSet.cpp), so this must advance every frame
    // regardless of the cycling currentFrame value.
    b->frameGeneration++;

    e.hasCommands = false;  // fresh command buffer, no commands yet

    // FIX (显存耗尽根因 - 主动式 GC，深度参考 MobileGL):
    // 提交成功后立即非阻塞 poll 所有 slot 的 fence。刚提交的 slot 不会
    // 立即完成（GPU 还在执行），但 OTHER slot（kMaxFramesInFlight-1 帧前
    // 提交的）可能已经完成，其 disposalQueue 可以立即 drain。
    //
    // 这镜像 MobileGL Present() 末尾的 m_textureManager->BeginFrame() +
    // m_bufferManager.BeginFrame()，在帧边界释放已完成帧的延迟资源。
    // 相比只在 eglSwapBuffers 开头 poll，这里多一次机会：commit_frame
    // 可能在 eglWaitClient（mid-frame flush）中被调用，此时 poll 能更早
    // 释放资源，降低后续 stage_and_copy_image 的显存压力。
    //
    // 非阻塞：vkGetFenceStatus 立即返回，不影响渲染性能。
    backend_poll_completed_frames();
}

/*
 * Drain GPU work that references the active swapchain, then detach the
 * swapchain from the encoder. Called by EGL BEFORE backend_destroy_swapchain()
 * (eglDestroySurface / ensure_swapchain resize path).
 *
 * Sequence:
 *   1. end_render_pass() + commit_frame() — flush any pending commands that
 *      reference the swapchain's images into the GPU. This submit is safe
 *      because ensure_command_buffer_recording() at the top of commit_frame
 *      waits on the current slot's fence before resetting its buffer.
 *   2. vkDeviceWaitIdle() — block until the GPU has finished executing those
 *      commands (and any prior in-flight submits on other slots), so the
 *      swapchain's IOSurface-backed images are no longer referenced by the
 *      driver. Without this wait, vkDestroySwapchainKHR / vkDestroyImageView
 *      would free IOSurfaces that the GPU is still reading, and the next
 *      IOSurfaceBindAccel call in the Metal driver would crash with SIGSEGV
 *      (UAF).
 *   3. Clear fencePending[] — after vkDeviceWaitIdle, ALL fences are signaled.
 *      Clearing the flags ensures the next commit_frame doesn't waste a
 *      vkWaitForFences call on an already-signaled fence, and more importantly
 *      ensures consistent state for the new swapchain's first frame.
 *   4. Reset commandBufferRecording — the just-submitted buffer is no longer
 *      recording. The next ensure_command_buffer_recording() will lazily
 *      reset+begin the current slot's buffer (now safe, fence signaled).
 *   5. set_active_swapchain(nullptr) — clear the encoder's raw pointer to the
 *      swapchain so begin_render_pass() / commit_frame() cannot record layout
 *      barriers against its (soon-to-be-freed) images.
 */
void drain_and_detach_swapchain() {
    Backend* b = backend();
    if (!b->initialized || !b->device) {
        set_active_swapchain(nullptr);
        return;
    }
    end_render_pass();

    // FIX (IOSurfaceBindAccel SIGSEGV): Detach the swapchain BEFORE
    // commit_frame so commit_frame does NOT execute the dummy render pass
    // or layout transition on the newly-acquired image.
    //
    // When the swapchain is being rebuilt due to a size change,
    // swapchain_present_and_acquire (called by eglSwapBuffers just before
    // ensure_swapchain) already presented the old image and acquired the
    // next one. That newly-acquired image's IOSurface has the OLD swapchain
    // size, while the CAMetalLayer's drawableSize has already changed to
    // the NEW size. If commit_frame runs its dummy render pass against this
    // mismatched IOSurface, MoltenVK's IOSurfaceBindAccel dereferences an
    // invalid/stale IOSurface → SIGSEGV (crash log: IOSurface+0x19cc).
    //
    // Since we are about to destroy the swapchain anyway, there is no need
    // to transition layouts or bind IOSurfaces. Detaching first makes
    // commit_frame see sc=nullptr, which sets shouldSubmit=false (when
    // hasCommands is also false — the common case after eglSwapBuffers
    // already committed), so it returns immediately without recording
    // anything against the dying swapchain.
    //
    // This mirrors MobileGL's RecreateSwapchain (VulkanRenderer.cpp:7786+),
    // which calls vkDeviceWaitIdle unconditionally and never records new
    // commands against the dying swapchain.
    set_active_swapchain(nullptr);
    commit_frame();
    vkDeviceWaitIdle(b->device);

    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        b->fencePending[i] = false;
    }
    drain_all_disposal_queues();
    b->commandBufferRecording = false;
}

/*
 * FIX (VK_NOT_READY storm after deviceLost recovery):
 * Reset the encoder state to a clean "no pass active" baseline.
 *
 * When deviceLost is set mid-frame, commit_frame() returns early WITHOUT
 * calling end_render_pass() (it checks deviceLost at the very top), so
 * encoder().passActive stays true. The next frame's GL calls then see
 * passActive=true and record vkCmd* into b->commandBuffer — but that buffer
 * was never vkBeginCommandBuffer'd (ensure_command_buffer_recording returned
 * false during deviceLost). MoltenVK rejects every command with
 * "Command buffer cannot accept commands before vkBeginCommandBuffer() is
 * called" (VK_NOT_READY), producing thousands of identical errors per frame.
 *
 * backend_reset_device_lost() calls this after a successful swapchain rebuild
 * so the post-recovery frame starts clean: begin_render_pass() re-calls
 * ensure_command_buffer_recording() (now succeeds because deviceLost=false),
 * begins a fresh command buffer, and only then sets passActive=true.
 *
 * Also resets commandBufferRecording — the alias b->commandBuffer may point
 * at a stale/pending slot; forcing a re-begin via ensure_command_buffer_recording
 * on the next recording attempt avoids recording into a dead buffer.
 */
void reset_encoder_state() {
    EncoderState& e = encoder();
    e.passActive = false;
    e.boundPipeline = VK_NULL_HANDLE;
    e.hasCommands = false;
    e.colorCount = 0;
    e.depthView = VK_NULL_HANDLE;
    e.width = 0;
    e.height = 0;
    for (int i = 0; i < 8; ++i) e.colorViews[i] = VK_NULL_HANDLE;
    for (int i = 0; i < 8; ++i) e.fboColorTexIds[i] = 0;
    e.fboColorTexCount = 0;
    e.fboDepthTexId = 0;
    // Clear commandBufferRecording so the next ensure_command_buffer_recording()
    // lazily resets+begins the current slot's buffer rather than trusting a
    // stale recording flag left over from the pre-deviceLost frame.
    Backend* b = backend();
    if (b) b->commandBufferRecording = false;
}

} // namespace vk
} // namespace mithril

// ===========================================================================
// Public C API (declared in MG_Backend/Backend.h)
// ===========================================================================
extern "C" {

void backend_set_clear_color(float r, float g, float b, float a) {
    mithril::vk::set_clear_color(r, g, b, a);
}
void backend_set_clear_depth(double d) { mithril::vk::set_clear_depth(d); }
void backend_set_clear_stencil(int s)  { mithril::vk::set_clear_stencil(s); }
void backend_set_load_clear(void)      { mithril::vk::set_load_clear(true); }
void backend_set_load_load(void)       { mithril::vk::set_load_clear(false); }

void backend_set_invalidate_attachments(uint32_t color_mask, bool depth, bool stencil) {
    mithril::vk::set_invalidate_attachments(color_mask, depth, stencil);
}

void backend_clear_attachments(GLbitfield mask, int x, int y, int w, int h) {
    mithril::vk::clear_attachments(mask, x, y, w, h);
}

void backend_clear_buffer_indexed(GLenum buffer, GLint drawbuffer,
                                  const float color[4], float depth,
                                  GLuint stencil) {
    static const float kZero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mithril::vk::clear_buffer_indexed((uint32_t)buffer, drawbuffer,
                                      color ? color : kZero, depth,
                                      (uint32_t)stencil);
}

void backend_begin_render_pass(VkImageView* color_views, int color_count,
                               VkImageView depth_view, int width, int height, int samples) {
    mithril::vk::begin_render_pass(color_views, color_count, depth_view, width, height, samples);
}

/*
 * Root cause Y (CRITICAL): register the GL texture names backing the upcoming
 * user-FBO render pass's color/depth attachments. The GL draw path
 * (Drawing.cpp) calls this IMMEDIATELY before backend_begin_render_pass for
 * each non-swapchain attachment (i.e. whenever the bound FBO is not 0).
 *
 * begin_render_pass can only see VkImageView handles — it cannot reverse-
 * resolve them to GL texture names, and therefore cannot look up the
 * TextureEntry (which holds the VkImage, VkFormat, and tracked currentLayout
 * needed to emit a valid layout barrier). This registration bridges that gap.
 *
 * end_render_pass reads the same tex_ids to barrier the attachments back to
 * a read-only layout and update TextureEntry::currentLayout, then auto-clears
 * the registration (no separate clear call needed).
 *
 * Contract:
 *   - color_tex_ids may be NULL when color_count == 0 (depth-only FBO).
 *   - tex_id 0 (or any tex_id not in the texture table) is silently skipped.
 *   - For swapchain rendering (FBO 0) the GL layer does NOT call this — the
 *     swapchain path's barriers are handled by the activeSwapchain block in
 *     begin_render_pass / commit_frame.
 *   - color_count is clamped to 8 (kMaxColorAttachments).
 *   - depth_tex_id == 0 means no user-FBO depth attachment (the depth view
 *     may still be the swapchain's depthView, handled separately).
 *
 * This is a NEW C API entry point; all existing backend_* signatures are
 * unchanged. The corresponding namespace function is
 * mithril::vk::set_fbo_attachment_tex_ids (defined above).
 */
void backend_set_fbo_attachment_tex_ids(GLuint* color_tex_ids, int color_count,
                                        GLuint depth_tex_id) {
    mithril::vk::set_fbo_attachment_tex_ids(color_tex_ids, color_count, depth_tex_id);
}

void backend_end_render_pass(void) { mithril::vk::end_render_pass(); }
void backend_commit(void)          { mithril::vk::commit_frame(); }

void backend_set_active_swapchain(void* swapchain_state) {
    mithril::vk::set_active_swapchain((mithril::vk::Swapchain*)swapchain_state);
}

VkImageLayout backend_active_swapchain_color_layout(void) {
    mithril::vk::Swapchain* sc = mithril::vk::active_swapchain();
    return sc ? sc->currentColorLayout : VK_IMAGE_LAYOUT_UNDEFINED;
}

void backend_set_active_swapchain_color_layout(VkImageLayout layout) {
    mithril::vk::Swapchain* sc = mithril::vk::active_swapchain();
    if (sc) sc->currentColorLayout = layout;
}

void backend_drain_and_detach_swapchain(void) {
    mithril::vk::drain_and_detach_swapchain();
}

/*
 * Bind a graphics pipeline and track it (root cause AI).
 *
 * The tracking handle is what lets backend_draw_* refuse to record a draw
 * with no pipeline bound — recording one is undefined behaviour and makes
 * MoltenVK dereference a null MVKRenderPass (SIGSEGV in
 * MVKRenderSubpass::populateMTLRenderPassDescriptor). A null `pipeline` here
 * means creation failed upstream, so nothing is bound and the handle is
 * cleared rather than left pointing at a stale pipeline from an earlier draw.
 */
void backend_bind_pipeline(VkPipeline pipeline) {
    mithril::vk::Backend* b = mithril::vk::backend();
    // FIX (VK_NOT_READY storm): only record vkCmdBindPipeline when the command
    // buffer is actually recording. During deviceLost or before the first
    // begin_render_pass, b->commandBuffer may be non-null but not in the
    // RECORDING state — recording into it spams VK_NOT_READY.
    if (b->commandBuffer && b->commandBufferRecording && pipeline) {
        vkCmdBindPipeline(b->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        mithril::vk::encoder().boundPipeline = pipeline;
    } else {
        mithril::vk::encoder().boundPipeline = VK_NULL_HANDLE;
    }
}

/*
 * Push a vertex-stage push constant (root cause: gl_VertexID baseVertex
 * semantics). Every vertex shader carries a `_MithrilBaseVertex` block at
 * offset 0 / size 4 (Shader.cpp:inject_vertex_id_fixup); Drawing.cpp writes
 * g_state->currentBaseVertex here on every draw so the shader's gl_VertexID
 * (== gl_VertexIndex + _mbv._mithrilBaseVertex) matches desktop GL.
 *
 * Resolves the layout from the per-program table; a program with no descriptor
 * bindings falls back to the process-wide empty layout (both declare the same
 * VERTEX-stage range). vkCmdPushConstants is a state command and is valid both
 * inside and outside a render-pass instance.
 */
void backend_push_constants(GLuint program, uint32_t offset, uint32_t size,
                            const void* data) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b || !b->commandBuffer || !b->commandBufferRecording) return;
    if (!data) return;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    auto& tbl = mithril::vk::program_table();
    auto it = tbl.find(program);
    if (it != tbl.end() && it->second.pipelineLayout != VK_NULL_HANDLE) {
        layout = it->second.pipelineLayout;
    } else {
        layout = mithril::vk::backend_default_pipeline_layout();
    }
    if (layout == VK_NULL_HANDLE) return;

    vkCmdPushConstants(b->commandBuffer, layout, VK_SHADER_STAGE_VERTEX_BIT,
                       offset, size, data);
}

/* ---- Compute dispatch (glDispatchCompute) ----
 *
 * Mirrors MobileGL VulkanRenderer::DispatchCompute (VulkanRenderer.cpp:4492).
 * The render pass MUST be ended first: vkCmdDispatch is not a valid command
 * inside a render-pass instance, and under dynamic rendering there is no
 * subpass to hide in. Ending the pass here (rather than asking the GL
 * frontend to) keeps every caller — glDispatchCompute and
 * glDispatchComputeIndirect — from having to remember.
 */
static bool prepare_compute_dispatch() {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->initialized || b->deviceLost) return false;
    if (!mithril::g_state) return false;
    const GLuint program = mithril::g_state->currentProgram;
    if (program == 0) return false;

    if (mithril::vk::render_pass_active()) mithril::vk::end_render_pass();
    if (!mithril::vk::ensure_command_buffer_recording()) return false;

    VkPipeline pipe = backend_get_or_create_compute_pipeline(program);
    if (pipe == VK_NULL_HANDLE) return false;
    vkCmdBindPipeline(b->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    mithril::vk::bind_program_descriptors(program, VK_PIPELINE_BIND_POINT_COMPUTE);
    return true;
}

void backend_dispatch_compute(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    if (groups_x == 0 || groups_y == 0 || groups_z == 0) return;  // GL no-op
    if (!prepare_compute_dispatch()) return;
    vkCmdDispatch(mithril::vk::backend()->commandBuffer, groups_x, groups_y, groups_z);
}

void backend_dispatch_compute_indirect(VkBuffer buffer, VkDeviceSize offset) {
    if (buffer == VK_NULL_HANDLE) return;
    if (!prepare_compute_dispatch()) return;
    vkCmdDispatchIndirect(mithril::vk::backend()->commandBuffer, buffer, offset);
}

/* ---- glMemoryBarrier ----
 *
 * GL names the *kinds* of access that must be ordered; Vulkan wants explicit
 * src/dst access masks and pipeline stages. Rather than translate each bit
 * (and risk under-synchronising a case we did not enumerate), widen to a
 * single ALL_COMMANDS -> ALL_COMMANDS VkMemoryBarrier with a superset of
 * access flags, exactly as MobileGL does in BuildMemoryBarrierForGlBarriers /
 * MemoryBarrier (VulkanRenderer.cpp:4585-4620). glMemoryBarrier is called a
 * handful of times per frame at most, so the conservatism is free.
 */
void backend_memory_barrier(GLbitfield barriers) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->initialized || b->deviceLost) return;

    if (mithril::vk::render_pass_active()) mithril::vk::end_render_pass();
    if (!mithril::vk::ensure_command_buffer_recording()) return;

    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_SHADER_READ_BIT |
                       VK_ACCESS_TRANSFER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_HOST_WRITE_BIT |
                       VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                       VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_TRANSFER_WRITE_BIT |
                       VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                       VK_ACCESS_INDEX_READ_BIT |
                       VK_ACCESS_UNIFORM_READ_BIT |
                       VK_ACCESS_MEMORY_READ_BIT |
                       VK_ACCESS_MEMORY_WRITE_BIT;
    // GL_COMMAND_BARRIER_BIT orders writes against a subsequent
    // glDraw*Indirect / glDispatchComputeIndirect fetch, which Vulkan models
    // as its own access flag rather than folding into MEMORY_READ.
    if (barriers & MG_GL_COMMAND_BARRIER_BIT) {
        mb.dstAccessMask |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    }

    vkCmdPipelineBarrier(b->commandBuffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

void backend_set_viewport(int x, int y, int w, int h, double znear, double zfar) {
    mithril::vk::Backend* b = mithril::vk::backend();
    // FIX (VK_NOT_READY storm): guard against non-recording command buffer.
    if (!b->commandBuffer || !b->commandBufferRecording) return;
    // FIX (root cause Z): GL viewport Y is bottom-origin (Y grows upward
    // from the bottom-left of the framebuffer), but Vulkan viewport Y is
    // top-origin (Y grows downward from the top-left). The previous code
    // passed GL Y through directly, which is fine for a full-screen viewport
    // (y=0, h=H → vk_y would also be 0) but breaks for any non-zero Y:
    //   - GUI rendering that uses a non-full-screen viewport clips the wrong
    //     vertical region (the bottom region in Vulkan terms, which is the
    //     TOP region in GL terms).
    //   - Sub-viewports used by post-process passes are vertically flipped.
    //
    // MoltenVK does NOT do a Y-flip when translating Vulkan viewport to
    // Metal (the Y-flip is done in the vertex shader via gl_Position.y =
    // -gl_Position.y for the default framebuffer — see Drawing.cpp's
    // Y-flipped SPIR-V variant). So the Vulkan viewport's Y origin is the
    // true top-left of the framebuffer, and the GL→Vulkan Y conversion
    // must be: vk_y = framebufferHeight - gl_y - gl_h.
    //
    // Mirrors MobileGL VulkanRenderer (VulkanRenderer.cpp:
    // viewport.y = extent.height - glViewport.y - glViewport.height).
    //
    // framebufferHeight: prefer the active render pass height (e.height,
    // set in begin_render_pass and clamped to the swapchain/FBO dimensions)
    // so the viewport matches the actual attachment extent. When no pass is
    // active (rare — the GL frontend normally begins a pass before issuing
    // viewport state), fall back to g_state->viewportH (the GL-tracked
    // viewport height).
    int fbHeight = mithril::vk::encoder_height_for_yflip();
    if (fbHeight <= 0 && mithril::g_state) {
        fbHeight = mithril::g_state->viewportH;
    }
    int vk_y = fbHeight - y - h;
    VkViewport vp{};
    vp.x        = (float)x;
    vp.y        = (float)vk_y;
    vp.width    = (float)w;
    vp.height   = (float)h;
    vp.minDepth = (float)znear;
    vp.maxDepth = (float)zfar;
    vkCmdSetViewport(b->commandBuffer, 0, 1, &vp);
}

void backend_set_scissor(int x, int y, int w, int h) {
    mithril::vk::Backend* b = mithril::vk::backend();
    // FIX (VK_NOT_READY storm): guard against non-recording command buffer.
    if (!b->commandBuffer || !b->commandBufferRecording) return;
    // FIX (root cause Z): same Y-origin conversion as backend_set_viewport.
    // GL scissor Y is bottom-origin; Vulkan scissor Y is top-origin. Without
    // this conversion, a non-zero-Y scissor (e.g. GUI clipping) clips the
    // wrong vertical region -> partial black screen where the clipped-out
    // region should be drawn. MoltenVK does not Y-flip the scissor.
    //
    // Mirrors MobileGL VulkanRenderer's scissor.y = extent.height -
    // glScissor.y - glScissor.height.
    int fbHeight = mithril::vk::encoder_height_for_yflip();
    if (fbHeight <= 0 && mithril::g_state) {
        fbHeight = mithril::g_state->viewportH;
    }
    int vk_y = fbHeight - y - h;
    VkRect2D sc{};
    sc.offset.x = x; sc.offset.y = vk_y;
    sc.extent.width = (uint32_t)w; sc.extent.height = (uint32_t)h;
    vkCmdSetScissor(b->commandBuffer, 0, 1, &sc);
}

void backend_set_vertex_buffer(int slot, VkBuffer buffer, VkDeviceSize offset) {
    mithril::vk::Backend* b = mithril::vk::backend();
    // FIX (VK_NOT_READY storm): guard against non-recording command buffer.
    if (!b->commandBuffer || !b->commandBufferRecording || !buffer) return;
    VkDeviceSize offsets[1] = { offset };
    vkCmdBindVertexBuffers(b->commandBuffer, (uint32_t)slot, 1, &buffer, offsets);
}

void backend_set_fragment_buffer(int slot, VkBuffer buffer, VkDeviceSize offset) {
    // No-op: fragment-stage UBO binding is handled by descriptor sets built in
    // DescriptorSet.cpp (backend_bind_program_descriptors). There is no Vulkan
    // "bind buffer to fragment stage slot" command outside of descriptor sets,
    // so this entry point exists only to satisfy the C API contract.
    (void)slot; (void)buffer; (void)offset;
}

void backend_set_vertex_texture(int slot, VkImageView view, VkSampler sampler) {
    // No-op: see backend_set_fragment_buffer — descriptor binding is centralised
    // in DescriptorSet.cpp (backend_bind_program_descriptors).
    (void)slot; (void)view; (void)sampler;
}

void backend_set_fragment_texture(int slot, VkImageView view, VkSampler sampler) {
    // No-op: see backend_set_fragment_buffer — descriptor binding is centralised
    // in DescriptorSet.cpp (backend_bind_program_descriptors).
    (void)slot; (void)view; (void)sampler;
}

void backend_set_blend_color(float r, float g, float b, float a) {
    // NOTE: parameter `b` is the blue blend constant (float); the backend ptr
    // is renamed to avoid shadowing it.
    mithril::vk::Backend* bk = mithril::vk::backend();
    // FIX (VK_NOT_READY storm): guard against non-recording command buffer.
    if (!bk->commandBuffer || !bk->commandBufferRecording) return;
    float bc[4] = { r, g, b, a };
    vkCmdSetBlendConstants(bk->commandBuffer, bc);
}

void backend_set_depth_bias(float slope, float clamp) {
    mithril::vk::Backend* b = mithril::vk::backend();
    // FIX (VK_NOT_READY storm): guard against non-recording command buffer.
    if (!b->commandBuffer || !b->commandBufferRecording) return;
    vkCmdSetDepthBias(b->commandBuffer, slope, clamp, 0.0f);
}

void backend_set_cull_mode(int mode) {
    mithril::vk::Backend* b = mithril::vk::backend();
    // FIX (VK_NOT_READY storm): guard against non-recording command buffer.
    if (!b->commandBuffer || !b->commandBufferRecording) return;
    VkCullModeFlags cull = VK_CULL_MODE_NONE;
    if (mode == 1) cull = VK_CULL_MODE_FRONT_BIT;
    else if (mode == 2) cull = VK_CULL_MODE_BACK_BIT;
    else if (mode == 3) cull = VK_CULL_MODE_FRONT_AND_BACK;
    vkCmdSetCullMode(b->commandBuffer, cull);
}

void backend_set_front_face(int ccw) {
    mithril::vk::Backend* b = mithril::vk::backend();
    // FIX (VK_NOT_READY storm): guard against non-recording command buffer.
    if (!b->commandBuffer || !b->commandBufferRecording) return;
    vkCmdSetFrontFace(b->commandBuffer, ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE);
}

void backend_set_depth_test(int enabled, int write_mask, int compare_func) {
    mithril::vk::Backend* b = mithril::vk::backend();
    // FIX (VK_NOT_READY storm): guard against non-recording command buffer.
    if (!b->commandBuffer || !b->commandBufferRecording) return;
    vkCmdSetDepthTestEnable(b->commandBuffer, enabled ? VK_TRUE : VK_FALSE);
    vkCmdSetDepthWriteEnable(b->commandBuffer, write_mask ? VK_TRUE : VK_FALSE);
    VkCompareOp op = VK_COMPARE_OP_LESS;
    switch (compare_func) {
        case 0x200: op = VK_COMPARE_OP_NEVER; break;    // GL_NEVER
        case 0x201: op = VK_COMPARE_OP_LESS; break;     // GL_LESS
        case 0x202: op = VK_COMPARE_OP_EQUAL; break;    // GL_EQUAL
        case 0x203: op = VK_COMPARE_OP_LESS_OR_EQUAL; break;
        case 0x204: op = VK_COMPARE_OP_GREATER; break;
        case 0x205: op = VK_COMPARE_OP_NOT_EQUAL; break;
        case 0x206: op = VK_COMPARE_OP_GREATER_OR_EQUAL; break;
        case 0x207: op = VK_COMPARE_OP_ALWAYS; break;
        default: op = VK_COMPARE_OP_LESS; break;
    }
    vkCmdSetDepthCompareOp(b->commandBuffer, op);
}

void backend_set_color_write_mask(int r, int g, int b, int a) {
    (void)r; (void)g; (void)b; (void)a;
    // No-op by design: colorWriteMask is a STATIC pipeline state (part of
    // VkPipelineColorBlendAttachmentState), not a dynamic state. It is read
    // from g_state->colorMask at pipeline-creation time (Drawing.cpp packs it
    // into the pipeline signature), so changing glColorMask creates a new
    // pipeline rather than updating the bound one. VK_DYNAMIC_STATE_COLOR_WRITE
    // _ENABLE_EXT would allow dynamic toggling but requires an extension we do
    // not enable; the static approach is correct and matches MobileGL.
}

void backend_set_stencil_state(int enabled, int func, int ref, int mask,
                               int sfail, int dpfail, int dppass) {
    (void)enabled; (void)func; (void)ref; (void)mask;
    (void)sfail; (void)dpfail; (void)dppass;
    // Stencil dynamic state deferred (bring-up).
}

void backend_draw_arrays(int primitive, int first, int count) {
    (void)primitive;
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->commandBuffer) return;
    if (!mithril::vk::draw_recording_allowed("backend_draw_arrays")) return;
    // Root cause AG (CRITICAL): pass firstInstance from g_state. glDrawArrays
    // itself has no baseInstance, but glDrawArraysInstancedBaseInstance /
    // glDrawArraysInstancedBaseVertexBaseInstance (rare) set
    // g_state->currentBaseInstance before falling through to the
    // non-indexed draw path. The GL layer (Drawing.cpp, modified by another
    // agent) sets g_state->currentBaseInstance and resets it to 0 after the
    // draw returns — we only read it here.
    //   vkCmdDraw(cmdBuf, vertexCount, instanceCount, firstVertex, firstInstance)
    // Mirrors MobileGL drawParams.firstInstance.
    uint32_t firstInstance = 0;
    if (mithril::g_state) firstInstance = mithril::g_state->currentBaseInstance;
    vkCmdDraw(b->commandBuffer, (uint32_t)count, 1, (uint32_t)first, firstInstance);
}

void backend_draw_indexed(int primitive, int count, int index_type,
                          VkBuffer index_buffer, VkDeviceSize index_offset) {
    (void)primitive;
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->commandBuffer || !index_buffer) return;
    if (!mithril::vk::draw_recording_allowed("backend_draw_indexed")) return;
    // FIX (root cause AE, CRITICAL): GL_UNSIGNED_BYTE index support.
    // Drawing.cpp maps GL_UNSIGNED_BYTE → 2 (index_type_to_int), but the
    // previous code only handled 0 (UINT16) and 1 (UINT32), treating
    // GL_UNSIGNED_BYTE as UINT16 → 1-byte indices read as 2-byte → geometry
    // corruption → red screen. Map 2 to VK_INDEX_TYPE_UINT8_EXT (requires
    // VK_EXT_index_type_uint8, enabled in Device.cpp by another agent).
    // Mirrors MobileGL VulkanRenderer.cpp:3093-3109.
    VkIndexType t;
    if (index_type == 1)      t = VK_INDEX_TYPE_UINT32;
    else if (index_type == 2) t = VK_INDEX_TYPE_UINT8_EXT;  // GL_UNSIGNED_BYTE
    else                      t = VK_INDEX_TYPE_UINT16;
    vkCmdBindIndexBuffer(b->commandBuffer, index_buffer, index_offset, t);
    // FIX (root cause AG, CRITICAL): pass baseVertex + baseInstance from
    // g_state. glDrawElementsBaseVertex sets g_state->currentBaseVertex
    // (vertexOffset) and glDrawElementsInstancedBaseInstance sets
    // g_state->currentBaseInstance (firstInstance) before falling through to
    // glDrawElements. The previous hardcoded (vertexOffset=0, firstInstance=0)
    // discarded both -> all instanced draws read the same vertex range ->
    // geometry misalignment -> red/garbled screen.
    //   vkCmdDrawIndexed(cmdBuf, indexCount, instanceCount, firstIndex,
    //                    vertexOffset, firstInstance)
    // Mirrors MobileGL drawParams.baseVertex / drawParams.baseInstance.
    // The GL layer (Drawing.cpp) resets these to 0 after the draw returns.
    int32_t  vertexOffset = 0;
    uint32_t firstInstance = 0;
    if (mithril::g_state) {
        vertexOffset = mithril::g_state->currentBaseVertex;
        firstInstance = mithril::g_state->currentBaseInstance;
    }
    vkCmdDrawIndexed(b->commandBuffer, (uint32_t)count, 1, 0,
                     (int32_t)vertexOffset, firstInstance);
}

void backend_draw_arrays_instanced(int primitive, int first, int count, int primcount) {
    (void)primitive;
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->commandBuffer) return;
    if (!mithril::vk::draw_recording_allowed("backend_draw_arrays_instanced")) return;
    // Root cause AG (CRITICAL): pass firstInstance from g_state (see
    // backend_draw_arrays for rationale). glDrawArraysInstancedBaseInstance
    // sets g_state->currentBaseInstance before falling through to the
    // instanced draw path.
    uint32_t firstInstance = 0;
    if (mithril::g_state) firstInstance = mithril::g_state->currentBaseInstance;
    vkCmdDraw(b->commandBuffer, (uint32_t)count, (uint32_t)primcount,
              (uint32_t)first, firstInstance);
}

void backend_draw_indexed_instanced(int primitive, int count, int index_type,
                                    VkBuffer index_buffer, VkDeviceSize index_offset,
                                    int primcount) {
    (void)primitive;
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->commandBuffer || !index_buffer) return;
    if (!mithril::vk::draw_recording_allowed("backend_draw_indexed_instanced")) return;
    // FIX (root cause AE, CRITICAL): GL_UNSIGNED_BYTE index support — see
    // backend_draw_indexed for the full rationale.
    VkIndexType t;
    if (index_type == 1)      t = VK_INDEX_TYPE_UINT32;
    else if (index_type == 2) t = VK_INDEX_TYPE_UINT8_EXT;  // GL_UNSIGNED_BYTE
    else                      t = VK_INDEX_TYPE_UINT16;
    vkCmdBindIndexBuffer(b->commandBuffer, index_buffer, index_offset, t);
    // FIX (root cause AG, CRITICAL): pass baseVertex + baseInstance from
    // g_state — see backend_draw_indexed for the full rationale.
    // glDrawElementsInstancedBaseVertex sets currentBaseVertex,
    // glDrawElementsInstancedBaseInstance sets currentBaseInstance; both
    // fall through to glDrawElementsInstanced.
    int32_t  vertexOffset = 0;
    uint32_t firstInstance = 0;
    if (mithril::g_state) {
        vertexOffset = mithril::g_state->currentBaseVertex;
        firstInstance = mithril::g_state->currentBaseInstance;
    }
    vkCmdDrawIndexed(b->commandBuffer, (uint32_t)count, (uint32_t)primcount, 0,
                     (int32_t)vertexOffset, firstInstance);
}

/* ---- Indirect draws (GL 4.0 ARB_draw_indirect) ----
 *
 * The draw parameters live in a GPU buffer instead of the call arguments, so
 * the GPU can generate its own work. Metal has this natively
 * (drawPrimitives:indirectBuffer:) and MoltenVK maps vkCmdDrawIndirect onto
 * it, which makes this one of the few GL 4.0 features that costs almost
 * nothing here.
 *
 * The GL and Vulkan parameter blocks are laid out identically —
 * VkDrawIndirectCommand matches GL's {count, primCount, first, baseInstance}
 * and VkDrawIndexedIndirectCommand matches {count, primCount, firstIndex,
 * baseVertex, baseInstance} — so the buffer contents need no translation.
 *
 * multiDrawIndirect with drawCount > 1 requires the multiDrawIndirect
 * feature; the loop fallback keeps working without it.
 */
void backend_draw_indirect(int primitive, VkBuffer indirect_buffer,
                           VkDeviceSize indirect_offset,
                           int draw_count, int stride) {
    (void)primitive;
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->commandBuffer || !indirect_buffer || draw_count <= 0) return;
    if (!mithril::vk::draw_recording_allowed("backend_draw_indirect")) return;
    const uint32_t effStride = stride > 0 ? (uint32_t)stride : 16u;  // sizeof(VkDrawIndirectCommand)
    if (draw_count == 1 || b->multiDrawIndirectSupported) {
        vkCmdDrawIndirect(b->commandBuffer, indirect_buffer, indirect_offset,
                          (uint32_t)draw_count, effStride);
        return;
    }
    for (int i = 0; i < draw_count; ++i) {
        vkCmdDrawIndirect(b->commandBuffer, indirect_buffer,
                          indirect_offset + (VkDeviceSize)i * effStride, 1, effStride);
    }
}

void backend_draw_indexed_indirect(int primitive, int index_type,
                                   VkBuffer index_buffer, VkDeviceSize index_offset,
                                   VkBuffer indirect_buffer,
                                   VkDeviceSize indirect_offset,
                                   int draw_count, int stride) {
    (void)primitive;
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->commandBuffer || !index_buffer || !indirect_buffer || draw_count <= 0) return;
    if (!mithril::vk::draw_recording_allowed("backend_draw_indexed_indirect")) return;
    VkIndexType t;
    if (index_type == 1)      t = VK_INDEX_TYPE_UINT32;
    else if (index_type == 2) t = VK_INDEX_TYPE_UINT8_EXT;
    else                      t = VK_INDEX_TYPE_UINT16;
    vkCmdBindIndexBuffer(b->commandBuffer, index_buffer, index_offset, t);
    const uint32_t effStride = stride > 0 ? (uint32_t)stride : 20u;  // sizeof(VkDrawIndexedIndirectCommand)
    if (draw_count == 1 || b->multiDrawIndirectSupported) {
        vkCmdDrawIndexedIndirect(b->commandBuffer, indirect_buffer, indirect_offset,
                                 (uint32_t)draw_count, effStride);
        return;
    }
    for (int i = 0; i < draw_count; ++i) {
        vkCmdDrawIndexedIndirect(b->commandBuffer, indirect_buffer,
                                 indirect_offset + (VkDeviceSize)i * effStride, 1, effStride);
    }
}

/* ---- GL 4.6 ARB_indirect_parameters (_Count variants) ----
 *
 * vkCmdDrawIndirectCount / vkCmdDrawIndexedIndirectCount read the draw COUNT
 * from `count_buffer` at `count_offset` on the GPU, clamp it to maxDrawcount,
 * and issue that many draws — no CPU readback. This is exactly what
 * glMultiDrawArraysIndirectCount / glMultiDrawElementsIndirectCount need.
 *
 * Requires the Vulkan 1.2 `drawIndirectCount` core feature; the GL frontend
 * checks b->drawIndirectCountSupported and falls back to a CPU readback when
 * the device (or MoltenVK) does not report it.
 */
void backend_draw_indirect_count(int primitive, VkBuffer indirect_buffer,
                                 VkDeviceSize indirect_offset,
                                 VkBuffer count_buffer, VkDeviceSize count_offset,
                                 int max_drawcount, int stride) {
    (void)primitive;
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->commandBuffer || !indirect_buffer || !count_buffer || max_drawcount <= 0)
        return;
    if (!b->drawIndirectCountSupported) {
        // GL 4.6 ARB_indirect_parameters 无法用 vkCmdDrawIndirectCount。静默跳过
        // 会误导排查；记录一次。MoltenVK 1.2.x 正常路径不会到这里。
        static int loggedOnce = 0;
        if (loggedOnce++ < 1) {
            MITHRIL_LOG_WARN("vk", "backend_draw_indirect_count: device lacks "
                              "drawIndirectCount (GL 4.6 indirect_parameters "
                              "unavailable) — draw skipped");
        }
        return;
    }
    if (!mithril::vk::draw_recording_allowed("backend_draw_indirect_count")) return;
    const uint32_t effStride = stride > 0 ? (uint32_t)stride : 16u;  // sizeof(VkDrawIndirectCommand)
    vkCmdDrawIndirectCount(b->commandBuffer, indirect_buffer, indirect_offset,
                           count_buffer, count_offset,
                           (uint32_t)max_drawcount, effStride);
}

void backend_draw_indexed_indirect_count(int primitive, int index_type,
                                         VkBuffer index_buffer, VkDeviceSize index_offset,
                                         VkBuffer indirect_buffer, VkDeviceSize indirect_offset,
                                         VkBuffer count_buffer, VkDeviceSize count_offset,
                                         int max_drawcount, int stride) {
    (void)primitive;
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->commandBuffer || !index_buffer || !indirect_buffer || !count_buffer ||
        max_drawcount <= 0)
        return;
    if (!b->drawIndirectCountSupported) {
        static int loggedOnce = 0;
        if (loggedOnce++ < 1) {
            MITHRIL_LOG_WARN("vk", "backend_draw_indexed_indirect_count: device lacks "
                              "drawIndirectCount (GL 4.6 indirect_parameters "
                              "unavailable) — draw skipped");
        }
        return;
    }
    if (!mithril::vk::draw_recording_allowed("backend_draw_indexed_indirect_count")) return;
    VkIndexType t;
    if (index_type == 1)      t = VK_INDEX_TYPE_UINT32;
    else if (index_type == 2) t = VK_INDEX_TYPE_UINT8_EXT;
    else                      t = VK_INDEX_TYPE_UINT16;
    vkCmdBindIndexBuffer(b->commandBuffer, index_buffer, index_offset, t);
    const uint32_t effStride = stride > 0 ? (uint32_t)stride : 20u;  // sizeof(VkDrawIndexedIndirectCommand)
    vkCmdDrawIndexedIndirectCount(b->commandBuffer, indirect_buffer, indirect_offset,
                                  count_buffer, count_offset,
                                  (uint32_t)max_drawcount, effStride);
}

} // extern "C"
