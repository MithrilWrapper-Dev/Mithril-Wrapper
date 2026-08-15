// Mithril-Wrapper - MG_Backend/DirectVulkan/CommandStream.h
// Render-pass orchestration (dynamic rendering) + encoder dynamic-state setters
// + draw command recording + per-frame submit/present. Implements the
// backend_begin_render_pass / backend_end_render_pass / backend_commit /
// backend_set_* / backend_draw_* family declared in MG_Backend/Backend.h.
#ifndef MITHRIL_DIRECTVULKAN_COMMANDSTREAM_H
#define MITHRIL_DIRECTVULKAN_COMMANDSTREAM_H

#include <vulkan/vulkan.h>
#include <cstdint>  // uint32_t (used by clear_attachments mask in lieu of
                    // GLbitfield, to avoid pulling <GL/gl.h> into every TU
                    // that includes this header)

namespace mithril {
namespace vk {

struct Swapchain;

// True when a dynamic-rendering pass is currently active (between
// begin_render_pass() and end_render_pass()).
bool render_pass_active();

// True when a valid VkDescriptorSet has been bound into the CURRENT command
// buffer since the last command-buffer boundary (fresh begin / memo flush /
// pool growth). bind_program_descriptors() sets it true right before issuing
// vkCmdBindDescriptorSets; every command-buffer boundary resets it false.
// backend_draw_* refuses to record a vkCmdDraw when it is false, because a
// draw with an unbound / garbage descriptor set makes MoltenVK sample
// undefined memory (pure-red geometry) and, on A11's Metal 2, can fault the
// GPU (kIOGPUCommandBufferCallbackErrorPageFault) at the next vkQueueSubmit.
void set_descriptors_bound(bool bound);
bool descriptors_bound();

// Pending clear values applied to the load op of the next render pass.
void set_clear_color(float r, float g, float b, float a);
void set_clear_depth(double d);
void set_clear_stencil(int s);
void set_load_clear(bool clear);   // true = CLEAR (glClear), false = LOAD

// B1 first-frame diagnostic: number of vkCmdDraw* recorded in the current
// frame (reset at each fresh command-buffer begin). 0 => draws were dropped;
// >0 => draws were recorded but fragments may still be invisible (depth/
// viewport/shader). Read by eglSwapBuffers' B1 present log.
unsigned int backend_get_recorded_draws();

/*
 * Register the swapchain whose currently-acquired image is the render target
 * for framebuffer 0. Called by EGL (install_surface_on_state) after each
 * vkAcquireNextImageKHR. begin_render_pass() uses this to record the
 * PRESENT_SRC/UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL layout barrier on the
 * swapchain color image (and the one-shot UNDEFINED ->
 * DEPTH_STENCIL_ATTACHMENT_OPTIMAL barrier on the depth image); commit_frame()
 * uses it to record the reverse COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
 * barrier before vkEndCommandBuffer, and to signal the swapchain's
 * per-image renderFinished semaphore on vkQueueSubmit so vkQueuePresentKHR can
 * wait on it. Pass nullptr to detach (headless / no surface).
 */
void set_active_swapchain(Swapchain* sc);

// Begin a dynamic-rendering pass against the given attachments.
void begin_render_pass(VkImageView* color_views, int color_count,
                       VkImageView depth_view, int width, int height, int samples);

// End the active dynamic-rendering pass.
void end_render_pass();

// Clear specific aspects of the current framebuffer via vkCmdClearAttachments.
// Must be called inside a render pass. `mask` is a GLbitfield (uint32_t) of
// GL_COLOR_BUFFER_BIT / GL_DEPTH_BUFFER_BIT / GL_STENCIL_BUFFER_BIT.
void clear_attachments(uint32_t mask, int x, int y, int w, int h);

/*
 * Ensure the current frame slot's command buffer is in the RECORDING state.
 *
 * With per-slot command buffers (Backend::commandBuffers[]), after
 * commit_frame() submits slot N and advances to slot N+1, the alias
 * b->commandBuffer still points at slot N's buffer (now pending on the GPU).
 * The NEXT call to begin_render_pass / stage_and_copy_image / commit_frame
 * must switch to slot N+1's buffer, wait on its fence (submitted
 * kMaxFramesInFlight frames ago), reset it, and begin it before recording.
 *
 * This function does all of that lazily — only when the buffer is NOT already
 * recording. If it's already recording, it's a no-op (fast path). Call this
 * from ANY code path that records into b->commandBuffer outside of an active
 * render pass (texture uploads, layout transitions, commit_frame's present
 * barrier, etc.).
 *
 * Returns true if the buffer is recording and ready for vkCmd* calls.
 */
bool ensure_command_buffer_recording();

// Submit the recorded command buffer to the graphics queue and wait on the
// per-frame fence. Called by backend_commit() and backend_present_and_acquire().
void commit_frame();

/*
 * Reset the encoder state (passActive, boundPipeline, hasCommands, FBO
 * attachment registrations) to a clean "no pass active" state.
 *
 * Called by backend_reset_device_lost() after a successful swapchain rebuild.
 *
 * Root cause (VK_NOT_READY storm after deviceLost recovery):
 * When deviceLost is set mid-frame, commit_frame() returns early WITHOUT
 * calling end_render_pass(), so encoder().passActive stays true. The next
 * frame's GL calls (glClear -> clear_attachments -> vkCmdClearAttachments,
 * backend_bind_pipeline -> vkCmdBindPipeline) see passActive=true and record
 * into b->commandBuffer — but the command buffer was never vkBeginCommandBuffer'd
 * (ensure_command_buffer_recording returned false during deviceLost, and after
 * recovery the alias b->commandBuffer may point at a stale/pending slot).
 * MoltenVK rejects every vkCmd* with "Command buffer cannot accept commands
 * before vkBeginCommandBuffer() is called" (VK_NOT_READY), producing thousands
 * of identical error lines per frame.
 *
 * Resetting the encoder state here ensures the post-recovery frame starts
 * cleanly: begin_render_pass() will re-call ensure_command_buffer_recording()
 * (which now succeeds because deviceLost=false), begin a fresh command buffer,
 * and only then set passActive=true.
 */
void reset_encoder_state();

} // namespace vk
} // namespace mithril

#endif // MITHRIL_DIRECTVULKAN_COMMANDSTREAM_H
