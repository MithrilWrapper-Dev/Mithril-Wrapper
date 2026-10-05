// Mithril-Wrapper - egl/SwapchainHelper.cpp
// Vulkan swapchain lifecycle for the EGL layer. This is the ONLY module in
// the EGL layer that touches Vulkan/swapchain details (VkImageView,
// backend_*_swapchain, g_state's Vulkan fields). It implements
// ensure_swapchain() / install_surface_on_state() declared in EglInternal.h.
//
// Extracted verbatim from the former monolithic egl/egl.cpp so the EGL
// entry-point dispatch layer (egl.cpp) no longer carries these details inline.
// The two functions are pure "swapchain <-> GLState" plumbing and, unlike the
// rest of the EGL layer, do not set EGL errors (they return bool / void).
//
// NOTE: `is_current` replaces the old function's internal read of the
// thread-local t_currentDraw. egl.cpp computes `is_current == (t_currentDraw
// == surface)` and passes it in, so the behaviour is identical to the
// original code.
#include "EglInternal.h"

#include "../MG_Impl/includes.h"          // g_state, MITHRIL_LOG_*, backend_*
#include "../MG_Impl/Log.h"
#include "../MG_Backend/DirectVulkan/Device.h"

#include <cstdlib>

namespace mithril {
namespace egl {

// ---------------------------------------------------------------------------
// Vulkan swapchain helpers
// ---------------------------------------------------------------------------
// Build (or rebuild) the per-surface Vulkan swapchain against the native
// window. Returns true on success. The swapchain is owned by the EglSurface
// and freed in eglDestroySurface / when the window size changes.
//
// On rebuild (size changed), this drains GPU work that references the old
// swapchain BEFORE destroying it, so the Metal driver is no longer reading
// the old IOSurface-backed images when they are torn down. Without this
// drain, vkDestroySwapchainKHR frees IOSurfaces that the GPU is still
// accessing, and the next IOSurfaceBindAccel call crashes with SIGSEGV (UAF).
bool ensure_swapchain(EglSurface* s, bool is_current) {
    MITHRIL_LOG_WARN("egl", "ensure_swapchain enter");
    if (!s || !s->native_window) return false;
    int w = 0, h = 0;
    if (!surface_get_size(s->native_window, &w, &h)) return false;
    if (w <= 0 || h <= 0) {
        // Window not yet sized; defer swapchain creation to a later call.
        return false;
    }
    // FIX (swapchain 重建死循环): ensure_swapchain 失败后，如果每帧都重试，
    // 会形成死循环刷屏（vkCreateSwapchainKHR failed × N）。引入退避机制：
    // 失败后等待 N 帧再重试，给 GPU 时间释放资源。
    // 这个计数器是 per-surface 的，避免多 surface 互相干扰。
    if (s->swapchainRetryBackoff > 0) {
        s->swapchainRetryBackoff--;
        return false;
    }
    if (s->swapchain_state) {
        int cur_w = backend_swapchain_width(s->swapchain_state);
        int cur_h = backend_swapchain_height(s->swapchain_state);
        if (cur_w == w && cur_h == h) {
            s->width  = w;
            s->height = h;
            return true;
        }
        // Size changed: drain GPU work referencing the old swapchain, detach
        // it from the encoder, THEN tear down + recreate. The drain is
        // critical: it ensures the Metal driver has released the old
        // IOSurface-backed drawables before vkDestroySwapchainKHR frees them.
        // Skipping the drain causes IOSurfaceBindAccel UAF crashes on the
        // next present.
        if (is_current) {
            backend_drain_and_detach_swapchain();
        }
        backend_destroy_swapchain(s->swapchain_state);
        s->swapchain_state = nullptr;
    }
    // First-time creation: drain any in-flight GPU work (e.g. texture uploads
    // or shader compilation issued during context init) BEFORE creating the
    // swapchain. Without this, the first vkAcquireNextImageKHR races with
    // outstanding work that may touch the presentation engine's IOSurface
    // pool, and the first IOSurfaceBindAccel call crashes with SIGSEGV on
    // iPadOS 16.x. MobileGL's RecreateSwapchain (VulkanRenderer.cpp:7786)
    // calls vkDeviceWaitIdle unconditionally before swapchain creation; we
    // mirror that here. backend_drain_and_detach_swapchain() also calls
    // vkDeviceWaitIdle, so this is belt-and-suspenders even on the rebuild
    // path above.
    backend_drain_and_detach_swapchain();
    s->swapchain_state = backend_create_swapchain(
        s->native_window, w, h, s->wantDepthStencil ? 1 : 0, /*platform_hint=*/0);
    if (!s->swapchain_state) {
        // FIX: 退避机制。失败后等待 30 帧再重试（约 0.5 秒 @ 60fps），
        // 避免 eglSwapBuffers 每帧重试形成死循环刷屏。同时限流日志。
        s->swapchainRetryBackoff = 30;
        s->swapchainRetryCount++;
        if (s->swapchainRetryCount <= 3 || s->swapchainRetryCount % 50 == 0) {
            MITHRIL_LOG_WARN("egl", "backend_create_swapchain failed (window size = %dx%d, "
                              "retry #%d, backing off %d frames)",
                              w, h, s->swapchainRetryCount, s->swapchainRetryBackoff);
        }
        return false;
    }
    // 成功创建：重置退避计数器
    s->swapchainRetryBackoff = 0;
    s->swapchainRetryCount = 0;
    s->width  = backend_swapchain_width(s->swapchain_state);
    s->height = backend_swapchain_height(s->swapchain_state);
    return true;
}

// Push the surface's current swapchain image views into the active GLState so
// framebuffer-0 renders land on the on-screen drawable. Acquires the next
// swapchain image if none is currently acquired.
//
// Also registers the swapchain with the backend encoder (via
// backend_set_active_swapchain) so begin_render_pass()/commit_frame() can
// record the PRESENT_SRC/UNDEFINED <-> COLOR_ATTACHMENT_OPTIMAL layout barriers
// on the swapchain color image, the one-shot UNDEFINED ->
// DEPTH_STENCIL_ATTACHMENT_OPTIMAL barrier on the depth image, and signal the
// swapchain's per-image renderFinished semaphore on submit. Without this
// registration, dynamic rendering would hard-code COLOR_ATTACHMENT_OPTIMAL on
// an image that is actually in PRESENT_SRC_KHR (or UNDEFINED on first use),
// which MoltenVK treats as an illegal layout and renders nothing (black screen).
void install_surface_on_state(EglSurface* s, bool is_current) {
    if (!g_state) return;
    if (s && s->swapchain_state) {
        VkImageView color = backend_swapchain_acquire_color(s->swapchain_state);
        VkImageView depth = backend_swapchain_acquire_depth(s->swapchain_state);
        g_state->eglDefaultColor  = color;
        g_state->eglDefaultDepth  = depth;
        // Also expose the underlying VkImage handles + formats so image-level
        // operations (glBlitFramebuffer / glReadPixels involving FBO 0) can
        // reference the on-screen drawable directly.
        g_state->eglDefaultColorImage   = backend_swapchain_current_color_image(s->swapchain_state);
        g_state->eglDefaultColorFormat  = backend_swapchain_color_format(s->swapchain_state);
        g_state->eglDefaultDepthImage   = backend_swapchain_current_depth_image(s->swapchain_state);
        g_state->eglDefaultDepthFormat  = backend_swapchain_depth_format(s->swapchain_state);
        // FIX (IOSurfaceBindAccel SIGSEGV): Use the ACTUAL drawable size from
        // the native window (CAMetalLayer.drawableSize), NOT the swapchain's
        // creation-time size (s->width). On MoltenVK/iOS, the swapchain image
        // is backed by a CAMetalLayer drawable whose size = drawableSize at
        // acquire time, NOT the swapchain's imageExtent at creation time.
        // If the drawableSize changed after swapchain creation (e.g. GLFW
        // resized the window between swapchain creation and the first frame),
        // the IOSurface backing the acquired drawable has the NEW size, while
        // s->width still holds the OLD size. Setting eglDefaultWidth to the
        // stale s->width causes the render area to exceed the IOSurface
        // dimensions → IOSurfaceBindAccel dereferences out-of-bounds memory
        // → SIGSEGV.
        //
        // Use min(actual_drawable_size, swapchain_size) to handle both cases:
        //   - drawable shrank: clamp to the smaller drawable size (IOSurface)
        //   - drawable grew: clamp to the swapchain size (VkImage extent)
        int actualW = s->width;
        int actualH = s->height;
        int drawW = 0, drawH = 0;
        if (surface_get_size(s->native_window, &drawW, &drawH) && drawW > 0 && drawH > 0) {
            // Update the swapchain's tracked actual drawable size so
            // begin_render_pass() can clamp the render area as a safety net.
            backend_swapchain_set_drawable_size(s->swapchain_state, drawW, drawH);
            if (drawW != s->width || drawH != s->height) {
                // drawableSize changed after swapchain creation. This is the
                // ROOT CAUSE of the red/black screen: the swapchain's VkImage
                // extent (s->width) no longer matches the CAMetalLayer's
                // drawableSize (drawW), so the IOSurface backing the acquired
                // drawable has a different size than the VkImage. MoltenVK
                // presents a mismatched drawable → red screen or garbage.
                //
                // Mark the swapchain for rebuild so eglSwapBuffers recreates
                // it at the new drawableSize. For THIS frame, clamp
                // eglDefaultWidth to min(drawW, s->width) so the render area
                // does not exceed the IOSurface (prevents IOSurfaceBindAccel
                // SIGSEGV) while the rebuild is pending.
                static bool warnedOnce = false;
                if (!warnedOnce) {
                    warnedOnce = true;
                    MITHRIL_LOG_WARN("egl", "install_surface_on_state: drawableSize=%dx%d "
                                      "differs from swapchainSize=%dx%d — marking swapchain "
                                      "for rebuild, clamping eglDefault to %dx%d this frame",
                                      drawW, drawH, s->width, s->height,
                                      (drawW < s->width ? drawW : s->width),
                                      (drawH < s->height ? drawH : s->height));
                }
                // Mark for rebuild: eglSwapBuffers will call ensure_swapchain()
                // which detects needsRebuild and recreates the swapchain at
                // the current drawableSize.
                backend_swapchain_mark_rebuild(s->swapchain_state);
            }
            if (drawW < actualW) actualW = drawW;
            if (drawH < actualH) actualH = drawH;
        }
        g_state->eglDefaultWidth  = actualW;
        g_state->eglDefaultHeight = actualH;
        // Register the swapchain with the encoder so it can record layout
        // barriers and signal renderFinishedPerImage. Only register when the
        // surface actually has an acquired color view (color != VK_NULL_HANDLE)
        // — passing a swapchain whose acquire failed would crash the barrier
        // recorder, which dereferences sc->images[sc->currentImage].
        backend_set_active_swapchain(color != VK_NULL_HANDLE ? s->swapchain_state : nullptr);
    } else {
        g_state->eglDefaultColor  = VK_NULL_HANDLE;
        g_state->eglDefaultDepth  = VK_NULL_HANDLE;
        g_state->eglDefaultColorImage  = VK_NULL_HANDLE;
        g_state->eglDefaultColorFormat = VK_FORMAT_UNDEFINED;
        g_state->eglDefaultDepthImage  = VK_NULL_HANDLE;
        g_state->eglDefaultDepthFormat = VK_FORMAT_UNDEFINED;
        g_state->eglDefaultWidth  = 0;
        g_state->eglDefaultHeight = 0;
        // Detach the swapchain from the encoder so a headless / surfaceless
        // frame (or a frame against a user FBO) does not try to record layout
        // barriers against a destroyed swapchain.
        backend_set_active_swapchain(nullptr);
    }
    (void)is_current;   // retained for the API contract; logic above is size-driven
}

// ---------------------------------------------------------------------------
// Persistent device-loss recovery state machine.
//
// The EGL swap path (eglSwapBuffers) calls this once per frame. If the backend
// is not device-lost it returns false immediately and the caller proceeds with
// the normal swap. If it IS device-lost, the recovery state machine runs and
// returns true (caller must skip present/commit this frame).
//
// Recovery strategy (mirrors MobileGL): VK_ERROR_DEVICE_LOST is usually
// transient (a GPU timeout under memory pressure), so we periodically attempt a
// swapchain rebuild after vkDeviceWaitIdle + purging recreatable caches. On
// Metal a faulted device is permanent, so after a bound of consecutive failures
// we give up rather than spin forever.
bool swapchain_handle_device_lost(EglSurface* s, bool is_current) {
    if (!mithril::vk::backend_is_device_lost()) {
        return false;   // normal path
    }

    // 尝试恢复：每隔 10 帧尝试一次 swapchain 重建（约 0.17 秒 @ 60fps）。
    // 原 60 帧间隔太长，Minecraft 可能在等待期间检测到渲染失败而 exit(0)。
    static int recoveryAttemptCounter = 0;
    static int recoveryFailCount = 0;      // 连续重建失败计数（成功时清零）
    static bool recoveryGivenUp = false;   // 放弃恢复（Metal 设备已永久 fault）
    recoveryAttemptCounter++;

    // FIX (swapchain 重建死循环 - CRITICAL):
    // Metal 上 VK_ERROR_DEVICE_LOST 是不可恢复的 —— Metal device 已 fault，
    // 在同一个死掉的 VkDevice 上重建 swapchain 永远不会成功。旧代码无限重试
    // （日志显示 >150 次失败 + 数千行 VK_NOT_READY 警告），最终 Minecraft
    // 检测到渲染失败调用 exit(0)。
    // 修复：连续失败 8 次后停止重试，避免死循环刷屏。VkDevice 已死，只有
    // 销毁并重建整个 VkDevice 才能恢复（目前不可行 — 需要重启游戏）。
    // 首要防线是防止 device lost 发生（见 Device.cpp 的 VRAM 预算降低）。
    if (recoveryGivenUp) {
        return true;    // 静默返回，不再尝试
    }
    if (recoveryAttemptCounter >= 10 && s->native_window) {
        recoveryAttemptCounter = 0;
        // FIX (swapchain rebuild death loop - CRITICAL):
        // The old path only drained disposalQueue + attempted swapchain
        // rebuild. But rebuild kept failing because VkPipeline caches
        // (hundreds of MTLRenderPipelineState objects, each 1-5 MB) and
        // VkDescriptorSet pools (thousands of sets) were still consuming
        // memory. This was a chicken-and-egg: rebuild needs memory, but
        // caches are only freed AFTER rebuild succeeds.
        //
        // Fix: purge ALL recreatable cached resources BEFORE the rebuild
        // attempt. Pipelines are re-created on next draw, descriptor sets
        // on next bind — GL state (programs, textures, buffers) is preserved.
        //
        // Reference: MobileGL RecreateSwapchain (VulkanRenderer.cpp:8579)
        // calls pipelineFactory->DestroyAll() + uniformManager->ResetAllPools()
        // BEFORE creating the new swapchain.
        mithril::vk::backend_purge_cached_resources_for_recovery();
        // 强制重建 swapchain，如果成功则重置 deviceLost
        if (s->swapchain_state) {
            backend_destroy_swapchain(s->swapchain_state);
            s->swapchain_state = nullptr;
        }
        if (ensure_swapchain(s, is_current) && s->swapchain_state) {
            // 重建成功：重置 deviceLost，恢复渲染
            // backend_reset_device_lost 会再次 drain + 清除 pipeline 缓存
            mithril::vk::backend_reset_device_lost();
            if (is_current) {
                install_surface_on_state(s, true);
            }
            if (recoveryFailCount > 0) {
                MITHRIL_LOG_WARN("egl", "deviceLost recovery: swapchain rebuilt "
                                  "successfully after %d failed attempts, "
                                  "resuming rendering", recoveryFailCount);
            } else {
                MITHRIL_LOG_WARN("egl", "deviceLost recovery: swapchain rebuilt "
                                  "successfully, resuming rendering");
            }
            recoveryFailCount = 0;
            recoveryGivenUp = false;
        } else {
            // 重建失败：deviceLost 标志未清除，10 帧后重试。
            recoveryFailCount++;
            if (recoveryFailCount >= 8) {
                // Metal 设备已永久 fault — 停止重试，避免死循环。
                // VkDevice 上的 swapchain 重建无法恢复 faulted Metal device。
                MITHRIL_LOG_ERROR("egl", "deviceLost: swapchain rebuild failed "
                                  "%d times — Metal device is permanently "
                                  "faulted (VK_ERROR_DEVICE_LOST is "
                                  "unrecoverable on Metal). Stopping recovery "
                                  "attempts to avoid infinite loop. "
                                  "Prevention: see VRAM budget in Device.cpp.",
                                  recoveryFailCount);
                recoveryGivenUp = true;
            } else if (recoveryFailCount <= 3 || recoveryFailCount % 30 == 0) {
                MITHRIL_LOG_WARN("egl", "deviceLost recovery: swapchain rebuild "
                                  "failed (attempt #%d), will retry in 10 frames",
                                  recoveryFailCount);
            }
        }
    }
    return true;
}

// Frame-boundary GC. Non-blocking poll of every in-flight frame slot's fence;
// drains the deferred-release queue for slots whose fence has signaled. Call
// once per swap BEFORE allocating new frame resources. Mirrors MobileGL's
// BeginFrame() CollectAllDeferredReleases. Safe to call even when device-lost
// (only drains slots whose fence is already signaled).
void swapchain_poll_completed_frames() {
    mithril::vk::backend_poll_completed_frames();
}

// End the active render pass and submit the command buffer. commit_frame()'s
// empty-submit defense skips the submit if no commands were recorded since the
// last commit (e.g. eglWaitClient already flushed this frame).
void swapchain_flush_and_commit() {
    backend_end_render_pass();
    backend_commit();
}

// Present the frame we just rendered, then acquire the next image for the
// following frame. backend_present_and_acquire() calls vkQueuePresentKHR
// followed by vkAcquireNextImageKHR.
//
// IMPORTANT: present happens BEFORE the resize/rebuild check. If we rebuilt the
// swapchain before presenting, the vkQueuePresentKHR would reference a
// just-destroyed swapchain (UAF) — exactly the IOSurfaceBindAccel SIGSEGV seen
// in the field. The caller (eglSwapBuffers) handles the resize/rebuild AFTER
// this returns.
//
// ZERO-AREA GUARD: if the native window has collapsed to 0x0 (iOS app
// backgrounded / view minimized / snapshot not yet sized), skip present
// entirely. Mirrors MobileGL commit 7ab8386: presenting against an out-of-date
// swapchain from a zero-area window previously let Present submit on an
// already-signaled fence and present an image that was never properly acquired
// → black screen with sound.
void swapchain_present(EglSurface* s) {
    if (!s || !s->swapchain_state) return;
    bool zero_area = (s->width <= 0 || s->height <= 0);
    if (!zero_area) {
        backend_present_and_acquire(s->swapchain_state);
    }
}

void swapchain_refresh_default_views(EglSurface* s) {
    if (!g_state || !s || !s->swapchain_state) return;
    // backend_swapchain_acquire_color() is idempotent: with an index already
    // acquired it returns that index's view instead of acquiring again, so
    // calling it here (right after present acquired the next image) is safe
    // and cannot consume a second image.
    VkImageView color = backend_swapchain_acquire_color(s->swapchain_state);
    VkImageView depth = backend_swapchain_acquire_depth(s->swapchain_state);
    g_state->eglDefaultColor = color;
    g_state->eglDefaultDepth = depth;
    // Image handles too: glBlitFramebuffer / glReadPixels against FBO 0 go
    // through eglDefaultColorImage, and a stale one would read the image that
    // was already presented.
    g_state->eglDefaultColorImage = backend_swapchain_current_color_image(s->swapchain_state);
    g_state->eglDefaultDepthImage = backend_swapchain_current_depth_image(s->swapchain_state);
    // Keep the encoder's swapchain registration in step with the acquire. When
    // acquire failed there is no image to barrier or present, so detach —
    // otherwise begin_render_pass would record barriers against an image index
    // that is not owned by us.
    backend_set_active_swapchain(color != VK_NULL_HANDLE ? s->swapchain_state : nullptr);
}

// Query whether the backend has marked the swapchain dead (fatal Vulkan error
// from acquire/present/submit: GPU OOM / surface lost / device lost). The
// caller (eglSwapBuffers) rebuilds the swapchain when this returns true.
bool swapchain_needs_rebuild(EglSurface* s) {
    if (!s || !s->swapchain_state) return false;
    return backend_swapchain_needs_rebuild(s->swapchain_state) != 0;
}

// Destroy the surface's swapchain safely. If `is_current`, drain in-flight GPU
// work that references the swapchain and detach it from the encoder BEFORE
// tearing it down (prevents IOSurfaceBindAccel UAF when the GPU is still
// reading the IOSurface-backed images).
void swapchain_destroy(EglSurface* s, bool is_current) {
    if (!s || !s->swapchain_state) return;
    if (is_current) {
        backend_drain_and_detach_swapchain();
    }
    backend_destroy_swapchain(s->swapchain_state);
    s->swapchain_state = nullptr;
}

} // namespace egl
} // namespace mithril