// Mithril-Wrapper - MG_Backend/DirectVulkan/Device.cpp
// Vulkan 1.2 instance / physical-device / device / queue / command pool init.
// On Apple builds MoltenVK is linked as a dylib (host-bundled on iOS,
// Homebrew on macOS), so Vulkan entry points resolve at link time without a
// separate Vulkan loader or VK_ICD_FILENAMES.

// VK_EXT_metal_surface extension-name macro. The canonical definition lives in
// vulkan_metal.h, which vulkan.h only pulls in when VK_USE_PLATFORM_METAL_EXT
// is defined BEFORE the include. We intentionally do NOT define that macro in
// this .cpp (it drags in <Metal/Metal.h>, Objective-C only — would break the
// plain-C++ compile). SwapchainMetal.mm is the one TU that defines it. Here we
// supply the spec-mandated string literal directly so Device.cpp can request
// the instance extension by name. The value is fixed by the Vulkan spec and
// will never diverge; guarding with #ifndef keeps this a no-op if a future
// Vulkan header exposes the macro without the platform define.
#ifndef VK_EXT_METAL_SURFACE_EXTENSION_NAME
#define VK_EXT_METAL_SURFACE_EXTENSION_NAME "VK_EXT_metal_surface"
#endif

#include <cstdio>
#include "Device.h"
#include "Resources.h"
#include "CommandStream.h"  // end_render_pass, ensure_command_buffer_recording, render_pass_active
#include "Swapchain.h"      // acquire semaphore edge for out-of-band submits
#include "Pipeline.h"     // clear_all_pipeline_caches() for deviceLost recovery
#include "RenderPassCompat.h"  // 传统 VkRenderPass/VkFramebuffer 缓存（无 dynamic rendering 时的退路）
#include "DescriptorSet.h"  // reset_all_descriptor_pools() for swapchain rebuild recovery
#include "UniformArena.h"  // ubo_arena_shutdown() — transient UBO arena teardown
#include "../../MG_State/State.h"  // kMaxTextureUnits 等容量常量（backend_device_limit 用来夹紧上报值）
#include "../../MG_Impl/Log.h"

#include <cstring>
#include <ctime>
#include <cstdlib>
#include <vector>

// FIX (device lost 根因 - 真实内存上限): os_proc_available_memory() 返回进程
// 在被 Jetsam kill 之前还能分配的字节数。这是 iOS 上唯一可靠的内存上限指标
// （MoltenVK 报告的 heap size = 整个设备 RAM，完全不可信）。iOS 13+ / macOS 10.15+。
//
// 注意：os_proc_available_memory() 带 API_UNAVAILABLE(macos) 标注，macOS 原生
// 构建（CI test-macos-smoke 用来 dlopen 跑状态机冒烟）会因显式调用而编译失败。
// 因此必须用 TARGET_OS_IPHONE 而非 __APPLE__ 守卫，macOS 上降级走固定预算
// fallback（行为与 Android/其他平台一致）。
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if defined(__APPLE__) && TARGET_OS_IPHONE
#include <os/proc.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

namespace mithril {
namespace vk {

Backend* backend() {
    static Backend b;
    return &b;
}

// FIX (SIGBUS 根因 - vkDeviceWaitIdle 触发 deferred encoding):
// 见 Device.h 中的详细注释。此函数在 vkDeviceWaitIdle 前安全结束当前录制
// 的 command buffer，避免 MoltenVK 对未完成 command buffer 做 deferred
// encoding 时访问未对齐的命令池内存。
//
// 透明语义：如果调用时 command buffer 正在录制，函数会：
//   1. 结束 render pass（如有）— vkEndCommandBuffer 在 render pass 内会失败
//   2. vkEndCommandBuffer + vkQueueSubmit 提交当前录制的命令
//   3. vkDeviceWaitIdle 等待 GPU 完成
//   4. 清除所有 fencePending（wait 后所有 fence 已 signaled）
//   5. 重新 begin command buffer，让调用方可以继续录制（透明）
// 如果调用时 command buffer 未在录制，仅执行 vkDeviceWaitIdle（无状态变化）。
void safe_device_wait_idle() {
    Backend* b = backend();
    if (!b->initialized || !b->device) return;

    const bool wasRecording = b->commandBufferRecording && b->commandBuffer != VK_NULL_HANDLE;

    // 1. 如果 render pass 处于活动状态，先结束它。
    //    vkEndCommandBuffer 在 render pass 实例内调用会返回 VK_ERROR_UNKNOWN。
    //    纹理上传（stage_and_copy_image）通常在 render pass 外调用，
    //    但 draw 路径触发的分配可能在 render pass 内，需要安全处理。
    if (wasRecording && render_pass_active()) {
        end_render_pass();
        static int rpWarnCount = 0;
        rpWarnCount++;
        if (rpWarnCount <= 3) {
            MITHRIL_LOG_WARN("vk", "safe_device_wait_idle: ended active render "
                              "pass before vkDeviceWaitIdle (occurrence #%d) — "
                              "render pass will be re-established by next "
                              "begin_render_pass call", rpWarnCount);
        }
    }

    // 2. 结束并提交当前录制的 command buffer。
    //    vkEndCommandBuffer 让 MoltenVK 在安全上下文中完成 command buffer 的
    //    编码，然后 vkQueueSubmit 提交到 GPU。这样 vkDeviceWaitIdle 就不会
    //    触发对未完成 command buffer 的 deferred encoding。
    if (wasRecording) {
        VkResult endRc = vkEndCommandBuffer(b->commandBuffer);
        b->commandBufferRecording = false;
        if (endRc == VK_SUCCESS) {
            // 提交到当前 slot 的 fence，让 GPU 执行这个 command buffer。
            // 不等待 renderFinished semaphore（这是 GC 路径，不是正常帧提交）。
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &b->commandBuffer;

            // If this out-of-band flush contains work targeting the currently
            // acquired swapchain image, it is still the FIRST queue submit
            // that must consume vkAcquireNextImageKHR's binary semaphore.
            // Submitting without this wait races rendering against the
            // presentation engine; reusing the semaphore later is also invalid.
            Swapchain* sc = active_swapchain();
            VkSemaphore acquireWait = VK_NULL_HANDLE;
            VkPipelineStageFlags acquireStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            if (sc && sc->currentImage >= 0 && !sc->imageAvailableConsumed) {
                const int slot = sc->imageAvailableFrameSlot;
                if (slot >= 0 && slot < (int)sc->imageAvailablePerFrame.size() &&
                    sc->imageAvailablePerFrame[slot] != VK_NULL_HANDLE) {
                    acquireWait = sc->imageAvailablePerFrame[slot];
                    si.waitSemaphoreCount = 1;
                    si.pWaitSemaphores = &acquireWait;
                    si.pWaitDstStageMask = &acquireStage;
                } else {
                    MITHRIL_LOG_ERROR("vk", "safe_device_wait_idle: acquired image has no valid acquire semaphore");
                    sc->needsRebuild = true;
                }
            }

            VkFence fence = b->frameFences[b->currentFrame];
            // FIX: fence 可能在 signaled 状态（ensure_command_buffer_recording
            // 的 vkWaitForFences 不 reset fence，或上一次 safe_device_wait_idle
            // 的 vkDeviceWaitIdle 已 signal 它）。Vulkan spec 要求 vkQueueSubmit
            // 的 fence 必须是 unsignaled，否则 UB。commit_frame 也这样做。
            vkResetFences(b->device, 1, &fence);
            VkResult submitRc = vkQueueSubmit(b->graphicsQueue, 1, &si, fence);
            if (submitRc == VK_SUCCESS) {
                b->fencePending[b->currentFrame] = true;
                if (sc && acquireWait != VK_NULL_HANDLE) {
                    sc->imageAvailableConsumed = true;
                }
            }
            // 提交失败（OOM/deviceLost）时不设置 fencePending，后续的
            // vkDeviceWaitIdle 仍会等待其他 pending 工作。
        }
        // vkEndCommandBuffer 失败时（设备已挂起），command buffer 状态未定义，
        // 但 vkDeviceWaitIdle 仍可安全调用（它会等待其他 pending 工作）。
    }

    // 3. 等待 GPU 完成。
    VkResult waitRc = vkDeviceWaitIdle(b->device);
    // FIX (recovery loop): if vkDeviceWaitIdle returns DEVICE_LOST, the Metal
    // device is faulted — pending command buffers are discarded by Metal and
    // further wait/submit calls will also fail. Set deviceLost immediately to
    // prevent the recovery path from repeatedly calling vkDeviceWaitIdle (each
    // call re-triggers the page fault error in MoltenVK, flooding the log and
    // wasting CPU time). Deep reference: MobileGL handles device-lost by
    // marking the device as faulted and skipping all subsequent GPU operations.
    if (waitRc == VK_ERROR_DEVICE_LOST) {
        b->deviceLost = true;
    }

    // 4. vkDeviceWaitIdle 后所有 fence 已 signaled（或设备已 lost，fence 状态
    //    无意义）。清除 fencePending，让后续 ensure_command_buffer_recording
    //    跳过无意义的 vkWaitForFences。
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        b->fencePending[i] = false;
    }

    // 5. 如果之前在录制且设备未 lost，重新 begin command buffer。
    //    设备 lost 时不重新 begin（command buffer 录制会被 ensure_command_buffer_recording
    //    的 deviceLost 检查拦截）。
    if (wasRecording && !b->deviceLost) {
        ensure_command_buffer_recording();
    }
}

bool backend_is_device_lost() {
    return backend()->deviceLost;
}

void backend_reset_device_lost() {
    Backend* b = backend();
    // FIX (OOM 反馈循环): deviceLost 期间 disposalQueue 不会被排空
    // （ensure_command_buffer_recording 在 deviceLost 时早退），导致延迟
    // 释放的资源无限累积，显存占用不降反升。重置 deviceLost 时主动排空
    // 所有 slot 的 disposalQueue，释放被延迟销毁的 VkBuffer/VkImage/
    // VkDeviceMemory，给设备恢复腾出显存空间。
    if (b->device) {
        vkDeviceWaitIdle(b->device);
    }
    drain_all_disposal_queues();
    // FIX (红屏根因 - deviceLost 恢复后清除着色器负缓存):
    // deviceLost 期间 vkCreateGraphicsPipelines 可能因设备状态异常而失败，
    // 这些失败被缓存在 failedSignatures 中。即使设备恢复后着色器可以正常
    // 编译，draw 也会被永久跳过，导致物体消失/红屏。
    // 同时销毁已创建的 VkPipeline（可能引用了损坏的 MoltenVK 着色器缓存），
    // 让 get_or_create_pipeline 在下次 draw 时从干净状态重新创建。
    // 参考 MobileGL RecreateSwapchain（VulkanRenderer.cpp:8579）：
    // pipelineFactory->DestroyAll() 在 swapchain 重建时销毁全部 pipeline。
    clear_all_pipeline_caches();
    // FIX (VK_NOT_READY storm after deviceLost recovery):
    // Reset the encoder state (passActive, boundPipeline, hasCommands) and
    // commandBufferRecording. Without this, the post-recovery frame inherits
    // passActive=true from the pre-deviceLost frame (commit_frame returned
    // early without calling end_render_pass), and GL calls like glClear /
    // backend_bind_pipeline record vkCmd* into a command buffer that was never
    // vkBeginCommandBuffer'd -> MoltenVK spams "Command buffer cannot accept
    // commands before vkBeginCommandBuffer() is called" (VK_NOT_READY).
    reset_encoder_state();
    b->deviceLost = false;
    b->consecutiveSubmitFailures = 0;
}

void backend_reset_device_lost_pending_resources() {
    Backend* b = backend();
    if (!b->device) return;
    // vkDeviceWaitIdle 在 deviceLost 时可能返回 VK_ERROR_DEVICE_LOST，
    // 但仍会尝试等待已提交的 command buffer 完成或超时。之后 drain
    // 是安全的：即使 GPU 挂起，被引用的 Metal 资源也不会再被访问。
    VkResult waitResult = vkDeviceWaitIdle(b->device);
    if (waitResult != VK_SUCCESS) {
        // 设备挂起时 vkDeviceWaitIdle 失败是正常的，仍继续 drain
        // （资源已被 command buffer 引用过，GPU 挂起不会再次访问它们）
        static int drainFailLogCount = 0;
        drainFailLogCount++;
        if (drainFailLogCount <= 3) {
            MITHRIL_LOG_WARN("vk", "backend_reset_device_lost_pending_resources: "
                              "vkDeviceWaitIdle returned %d (expected during "
                              "deviceLost), draining disposal queues anyway",
                              (int)waitResult);
        }
    }
    drain_all_disposal_queues();
}

/*
 * FIX (swapchain rebuild death loop - CRITICAL):
 * Purge ALL recreatable cached Vulkan resources to free maximum memory before
 * a swapchain rebuild attempt during deviceLost recovery.
 *
 * Root cause of the death loop:
 *   The old recovery path only called clear_all_pipeline_caches() AFTER the
 *   swapchain rebuild succeeded (inside backend_reset_device_lost). But the
 *   rebuild FAILED because there wasn't enough memory — and the pipeline
 *   caches (holding hundreds of VkPipeline objects, each backed by a Metal
 *   MTLRenderPipelineState that can be several MB) and descriptor pools
 *   (holding thousands of VkDescriptorSet objects) were still consuming
 *   memory. This was a chicken-and-egg problem:
 *     - Swapchain rebuild needs memory → fails because caches hold memory
 *     - Caches are only cleared after rebuild succeeds → never clears
 *   Result: vkCreateSwapchainKHR failed 90+ times, game exited.
 *
 * Fix: call this BEFORE ensure_swapchain() in the deviceLost recovery path.
 * It frees:
 *   1. All cached VkPipeline objects (clear_all_pipeline_caches)
 *   2. All allocated VkDescriptorSet objects (reset_all_descriptor_pools)
 *   3. All deferred-destruction resources (drain_all_disposal_queues)
 *
 * These resources are all recreatable: pipelines are re-created on the next
 * draw via get_or_create_pipeline, descriptor sets via bind_program_descriptors.
 * The GL-level state (programs, textures, buffers) is preserved — only the
 * Vulkan-level cached objects are purged.
 *
 * Reference: MobileGL RecreateSwapchain (VulkanRenderer.cpp:8579) calls
 * pipelineFactory->DestroyAll() + uniformManager->ResetAllPools() BEFORE
 * creating the new swapchain.
 */
void backend_purge_cached_resources_for_recovery() {
    Backend* b = backend();
    if (!b->device) return;

    // 1. Wait for any in-flight GPU work to complete so we can safely destroy
    //    resources that may be referenced by pending command buffers.
    vkDeviceWaitIdle(b->device);

    // 2. Drain all deferred-destruction queues (staging buffers, old textures, etc.)
    drain_all_disposal_queues();

    // 3. Destroy all cached VkPipeline objects — the biggest memory consumers.
    //    Each VkPipeline is backed by a Metal MTLRenderPipelineState which holds
    //    compiled shader code + render state, typically 1-5 MB each.
    //    Minecraft creates dozens to hundreds of distinct pipelines during gameplay.
    clear_all_pipeline_caches();

    // 4. Reset all descriptor pools — frees every allocated VkDescriptorSet.
    //    Each set holds Metal descriptor resources; thousands can accumulate.
    reset_all_descriptor_pools();

    // 5. Reset the encoder state so the post-recovery frame starts clean.
    reset_encoder_state();

    MITHRIL_LOG_WARN("vk", "backend_purge_cached_resources_for_recovery: "
                      "purged pipelines + descriptor pools + disposal queues "
                      "(freeing memory for swapchain rebuild)");
}

void drain_disposal_queue(int slot) {
    Backend* b = backend();
    if (!b->device || slot < 0 || slot >= kMaxFramesInFlight) return;
    auto& q = b->disposalQueue[slot];
    if (q.empty()) return;
    for (auto& d : q) {
        if (d.buffer)  {  vkDestroyBuffer(b->device, d.buffer, nullptr); }
        if (d.image)   {  vkDestroyImage(b->device, d.image, nullptr); }
        if (d.view)    {  vkDestroyImageView(b->device, d.view, nullptr); }
        if (d.memory)  {
            
            vkFreeMemory(b->device, d.memory, nullptr);
            // FIX (P1): 递减分配计数器（诊断）
            if (b->currentAllocationCount > 0) b->currentAllocationCount--;
            // FIX (MobileGL-style VRAM monitoring): 递减已分配字节数
            if (d.memorySize > 0 && b->currentVramBytes >= d.memorySize) {
                b->currentVramBytes -= d.memorySize;
            } else if (d.memorySize > 0) {
                b->currentVramBytes = 0;  // 防止下溢
            }
        }
        if (d.sampler) {  vkDestroySampler(b->device, d.sampler, nullptr); }
        // FIX (descriptor pool UAF - P0): 次级池扩容时退役的 VkDescriptorPool。
        // 只能在此处（slot fence 已等待，所有引用其 set 的 command buffer 完成）
        // 销毁 —— vkDestroyDescriptorPool 隐式释放池内所有 set。
        if (d.pool) {  vkDestroyDescriptorPool(b->device, d.pool, nullptr); }
    }
    q.clear();
}

void drain_all_disposal_queues() {
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        drain_disposal_queue(i);
    }
}

// FIX (mid-frame UAF): drain every slot's disposal queue EXCEPT `skip`.
// Used by the proactive-GC path, which runs mid-frame while the current slot's
// command buffer is still being recorded. The current slot's deferred-destroyed
// resources may still be referenced by this frame's live descriptor sets, so
// they must not be freed here — they are freed by the normal fence-wait drain
// after the frame commits and the slot is recycled.
void drain_disposal_queues_except(int skip) {
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (i == skip) continue;
        drain_disposal_queue(i);
    }
}

// FIX (显存耗尽根因 - 主动式 GC，深度参考 MobileGL):
// 实现 backend_poll_completed_frames：非阻塞轮询所有帧槽位的 fence，
// 对已完成的 slot 立即 drain 其 disposalQueue。
//
// MobileGL 的 RefreshCompletedSubmits（VulkanRenderer.cpp:7199-7237）用
// vkGetFenceStatus 做 prefix-only scan，回收已完成的 pooled fence。
// 我们做类似的事：遍历所有 slot，对 fencePending[s]==true 的 slot 调用
// vkGetFenceStatus，若 VK_SUCCESS 则 drain 并清除 fencePending 标志。
//
// 关键：这是非阻塞的。vkGetFenceStatus 立即返回，不会 stall 渲染线程。
// 只有当 GPU 真正完成该 slot 的工作时才 drain，避免在 GPU 还在执行时
// 释放被引用的资源。
//
// 注意：drain 后 fencePending[s] 被清除，但 ensure_command_buffer_recording
// 复用该 slot 时仍会检查 fencePending——为 false 时跳过 vkWaitForFences，
// 这是正确的（fence 已 signaled，无需等待）。
int backend_poll_completed_frames() {
    Backend* b = backend();
    if (!b->initialized || !b->device) return 0;

    int drainedSlots = 0;
    for (int s = 0; s < kMaxFramesInFlight; ++s) {
        // FIX (GPU page fault root cause — defense-in-depth): NEVER drain the
        // CURRENT slot's disposal queue from a mid-frame poll. Every other
        // mid-frame drain site (try_allocate_memory_with_gc, proactive GC,
        // delete_program_resources, texture pre-create GC) excludes currentFrame;
        // this function is the only one that did not. Its safety previously
        // rested entirely on the invariant fencePending[currentFrame]==false
        // while recording. That holds today, but if it is ever broken (a submit
        // on the current slot that does not clear/advance — e.g. an out-of-band
        // safe_device_wait_idle path), draining the current slot here would free
        // views/descriptor-pools that the still-recording command buffer
        // references → kIOGPUCommandBufferCallbackErrorPageFault. Excluding
        // currentFrame removes the fragile invariant, matching every other site.
        if (s == b->currentFrame) continue;
        if (!b->fencePending[s]) continue;  // 无 pending submit，跳过
        if (b->disposalQueue[s].empty()) {
            // 队列为空：但 fence 可能仍 pending。检查 fence 状态以清除
            // fencePending 标志（让后续 ensure_command_buffer_recording 跳过等待）。
            // 不计入 drainedSlots（没有实际释放资源）。
            VkResult fr = vkGetFenceStatus(b->device, b->frameFences[s]);
            if (fr == VK_SUCCESS) {
                b->fencePending[s] = false;
            }
            continue;
        }
        // 有 pending submit 且 disposalQueue 非空：检查 fence
        VkResult fr = vkGetFenceStatus(b->device, b->frameFences[s]);
        if (fr == VK_SUCCESS) {
            // GPU 已完成该 slot 的所有工作，安全 drain
            drain_disposal_queue(s);
            b->fencePending[s] = false;
            drainedSlots++;
        }
        // VK_NOT_READY：GPU 还在执行，不 drain（避免 UAF）
        // 其他错误码（如 VK_ERROR_DEVICE_LOST）：不 drain，交给 deviceLost 恢复路径
    }

    if (drainedSlots > 0) {
        // 限流日志：仅在释放了大量资源时打印
        static int pollDrainLogCount = 0;
        pollDrainLogCount++;
        if (pollDrainLogCount <= 5 || pollDrainLogCount % 200 == 0) {
            MITHRIL_LOG_INFO("vk", "proactive poll: drained %d completed frame "
                              "slot(s) (occurrence #%d) — freed deferred "
                              "resources before new allocations",
                              drainedSlots, pollDrainLogCount);
        }
    }
    return drainedSlots;
}

// FIX (MobileGL-style VRAM monitoring): 主动 GC 基于实际已分配字节数，
// 而非 allocation count。MobileGL 的 TryDrainFrameTransients 在堆使用率高位时
// 主动 drain + rewind transient arena。我们镜像这个策略，但阈值更保守（50%）。
//
// 旧代码用 allocation count（currentAllocationCount >= 70% of
// maxMemoryAllocationCount）做阈值，但 MoltenVK 报告的 maxMemoryAllocationCount
// 不可信（10亿+），且 count 不能反映真实压力（16MB 纹理和 64B UBO 各算 1 次）。
// 旧字节级方案用 80% 阈值，但预算设为 1.5GB → 阈值 1.2GB，iPhone X GPU 在此
// 之前就 OOM fault。现在预算降到 ≤256MB，阈值降到 50%（≤128MB），确保 GC
// 在驱动 fault 前触发。
bool backend_proactive_gc_if_needed() {
    Backend* b = backend();
    if (!b->initialized || !b->device) return false;
    if (b->vramPressureThreshold == 0) return false;  // VRAM budget 未初始化

    // 字节级阈值检查：当前已分配字节数 >= 50% of totalVramBytes
    if (b->currentVramBytes < b->vramPressureThreshold) return false;

    // 检查是否有可释放的资源（任一 slot 的 disposalQueue 非空）
    bool hasDeferred = false;
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (!b->disposalQueue[i].empty()) { hasDeferred = true; break; }
    }
    if (!hasDeferred) return false;  // 没有可释放的，GC 无意义

    static int proactiveGcCount = 0;
    proactiveGcCount++;
    // 限流日志：首次 + 每 20 次
    if (proactiveGcCount <= 3 || proactiveGcCount % 20 == 0) {
        MITHRIL_LOG_WARN("vk", "proactive GC triggered (MobileGL-style): "
                          "VRAM %llu/%llu MB (>=50%% threshold %llu MB) — "
                          "draining before OOM (attempt #%d)",
                          (unsigned long long)(b->currentVramBytes / (1024*1024)),
                          (unsigned long long)(b->totalVramBytes / (1024*1024)),
                          (unsigned long long)(b->vramPressureThreshold / (1024*1024)),
                          proactiveGcCount);
    }

    // 先非阻塞 poll（可能已经完成，无需 vkDeviceWaitIdle 阻塞）
    backend_poll_completed_frames();

    // 如果 poll 后仍超阈值，且仍有 deferred 资源，才阻塞等待
    bool stillHasDeferred = false;
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (!b->disposalQueue[i].empty()) { stillHasDeferred = true; break; }
    }
    if (stillHasDeferred && b->currentVramBytes >= b->vramPressureThreshold) {
        // FIX (SIGBUS): 使用 safe_device_wait_idle 而非直接 vkDeviceWaitIdle。
        // proactive GC 可能在 stage_and_copy_image 录制期间触发（create_buffer →
        // try_allocate_memory_with_gc → 此函数）。此时 command buffer 正在录制，
        // 直接 vkDeviceWaitIdle 会触发 MoltenVK 的 deferred encoding →
        // MVKCmdBufferImageCopy::encode 访问未对齐内存 → SIGBUS。
        // safe_device_wait_idle 先结束+提交当前 command buffer，wait 后重新 begin，
        // 让 stage_and_copy_image 可以透明地继续录制。
        safe_device_wait_idle();
        // FIX (mid-frame UAF - GPU page fault root cause):
        // safe_device_wait_idle() submits the CURRENT slot's command buffer to
        // frameFences[currentFrame] WITHOUT advancing currentFrame, then waits
        // (vkDeviceWaitIdle) and re-begins the SAME slot's buffer. The descriptor
        // sets allocated for THIS frame still reference the current slot's
        // deferred-destroyed views (they were never invalidated, and
        // safe_device_wait_idle does NOT bump frameGeneration, so the descriptor
        // memo and setCursor are NOT rewound). If we drain disposalQueue[currentFrame]
        // here, we vkDestroy the very views that this frame's still-live
        // descriptor sets reference; the next draw can re-bind a stale set and
        // the next vkQueueSubmit hits kIOGPUCommandBufferCallbackErrorPageFault.
        //
        // Fix: drain ALL slots EXCEPT currentFrame. The current slot's queue is
        // left for the normal fence-wait drain (ensure_command_buffer_recording /
        // backend_poll_completed_frames) AFTER this frame commits and the slot is
        // recycled — by then no live descriptor set references those resources.
        drain_disposal_queues_except(b->currentFrame);
        // safe_device_wait_idle 已清除 fencePending，但重复清除无害（防御性）
        for (int i = 0; i < kMaxFramesInFlight; ++i) {
            b->fencePending[i] = false;
        }
        if (proactiveGcCount <= 3 || proactiveGcCount % 20 == 0) {
            MITHRIL_LOG_WARN("vk", "proactive GC: safe_device_wait_idle + "
                              "drain(except current) completed, VRAM now "
                              "%llu/%llu MB",
                              (unsigned long long)(b->currentVramBytes / (1024*1024)),
                              (unsigned long long)(b->totalVramBytes / (1024*1024)));
        }
    }

    // FIX (临界压力升级): 当 VRAM 仍 >= 65% 预算时（disposal queue 已空但内存
    // 仍高），驱逐 pipeline cache —— 每个 VkPipeline 背后是 Metal
    // MTLRenderPipelineState（1-5MB），Minecraft 会创建数百个。这是 disposal
    // queue 之外最大的可回收资源。pipeline 会在下次 draw 时自动重建。
    // 参考 MobileGL TryDrainFrameTransients：drain 后仍紧张时 DestroyAll
    // transient resources + pipeline cache。
    // 旧 80% 太晚：256MB 预算下 80%=205MB，此时 GPU 可能已接近 fault。
    // 65% of 256MB = 166MB → 在 GPU fault 前驱逐 pipeline。
    VkDeviceSize criticalThreshold = (b->totalVramBytes * 65) / 100;
    if (b->currentVramBytes >= criticalThreshold) {
        static int pipelinePurgeCount = 0;
        pipelinePurgeCount++;
        if (pipelinePurgeCount <= 3 || pipelinePurgeCount % 20 == 0) {
            MITHRIL_LOG_WARN("vk", "critical VRAM pressure: %llu/%llu MB "
                              "(>=65%%), purging pipeline cache to free memory "
                              "(attempt #%d)",
                              (unsigned long long)(b->currentVramBytes / (1024*1024)),
                              (unsigned long long)(b->totalVramBytes / (1024*1024)),
                              pipelinePurgeCount);
        }
        // FIX (GPU page fault root cause): clear_all_pipeline_caches() and
        // reset_all_descriptor_pools() run MID-FRAME here. reset_all_descriptor_pools
        // calls vkResetDescriptorPool on the current slot's pool, which invalidates
        // every descriptor set already vkCmdBindDescriptorSets'd into the still-
        // recording command buffer — including the current frame's live sets. On the
        // next vkQueueSubmit, the GPU reads invalid descriptor memory and faults
        // (kIOGPUCommandBufferCallbackErrorPageFault).
        //
        // Fix: flush the current command buffer + wait via safe_device_wait_idle()
        // FIRST. It re-begins the slot's buffer fresh (no descriptor sets bound yet),
        // so purging pipelines/pools afterwards cannot invalidate a bound set. The
        // pipeline/pool caches are recreated on the next draw.
        safe_device_wait_idle();
        // FIX (GPU page fault root cause — re-entrant purge): route through
        // request_purge() so the purge can never run while the buffer holds
        // live descriptor-set / pipeline binds. safe_device_wait_idle() above
        // normally flushes+re-begins it empty (safe), but request_purge()
        // guarantees the invariant and defers to the next safe boundary if not.
        request_purge();
    }
    return true;
}

namespace {

bool has_extension(const std::vector<VkExtensionProperties>& props, const char* name) {
    for (const auto& p : props) {
        if (std::strcmp(p.extensionName, name) == 0) return true;
    }
    return false;
}

bool has_layer(const std::vector<VkLayerProperties>& props, const char* name) {
    for (const auto& p : props) {
        if (std::strcmp(p.layerName, name) == 0) return true;
    }
    return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                              VkDebugUtilsMessageTypeFlagsEXT,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void*) {
    if (data && data->pMessage) {
        // FIX (日志刷屏根因): MoltenVK 在 OOM/deviceLost 时会通过 debug callback
        // 每帧输出大量重复的 MTLCommandBuffer/VK_ERROR 错误（日志中观察到的
        // ~3000 行循环就是来自这里）。对相同消息做去重限流：
        // - 首次：完整输出
        // - 后续相同消息：每 500 次输出一次（含计数）
        // - 不同消息：各自独立计数
        //
        // 使用简单的消息哈希（前 128 字节）做去重 key，避免存储完整消息。
        static thread_local struct {
            uint64_t hash;      // 上次消息的哈希
            int count;          // 相同消息的累计次数
        } dedup = { 0, 0 };

        // 简单 FNV-1a 哈希，只取前 128 字节（足以区分 MoltenVK 的不同错误类型）
        const char* msg = data->pMessage;
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < 128 && msg[i]; ++i) {
            h ^= (uint8_t)msg[i];
            h *= 1099511628211ULL;
        }

        if (h == dedup.hash) {
            dedup.count++;
            // 相同消息：首次后每 500 次才输出一次
            if (dedup.count % 500 == 0) {
                MITHRIL_LOG_WARN("vk", "%s (repeated %d times, suppressed)",
                                  msg, dedup.count);
            }
        } else {
            // 新消息：输出并重置计数
            dedup.hash = h;
            dedup.count = 1;
            MITHRIL_LOG_WARN("vk", "%s", msg);
        }
    }
    return VK_FALSE;
}

} // namespace

// MoltenVK reads MVK_CONFIG_* exactly once: at the first vkCreateInstance in
// the process. Everything after that is latched, so a later setenv() is ignored
// no matter who calls it.
//
// That made calling this from init_device() a gamble. It is fine when Mithril
// creates the instance itself - which is why macOS always looked correct - but
// when the host process has already created one, MoltenVK has latched its
// defaults and our flip settings never apply at all.
//
// Running it from a library constructor removes the gamble: the constructor
// runs while this dylib is being loaded, which necessarily precedes any Vulkan
// call the host or Mithril can make through it.
void apply_moltenvk_config() {
#if defined(__APPLE__)
    setenv("MVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS", "1", 1);
    setenv("MVK_CONFIG_PREFILL_METAL_COMMAND_BUFFERS", "0", 1);
    setenv("MVK_CONFIG_RESUME_LOST_DEVICE", "1", 1);
    // Mithril flips vertex Y itself for the default framebuffer. MoltenVK must
    // not flip as well, or the two cancel and the frame comes out mirrored.
    setenv("MVK_CONFIG_SHADER_CONVERSION_FLIP_VERTEX_Y", "0", 1);
    setenv("MVK_CONFIG_MAX_ACTIVE_METAL_COMMAND_BUFFERS_PER_QUEUE", "2", 1);
    setenv("MVK_CONFIG_VK_SEMAPHORE_SUPPORT_STYLE", "2", 1);
#endif
}

namespace {
struct MoltenVKConfigAtLoad {
    MoltenVKConfigAtLoad() { apply_moltenvk_config(); }
};
const MoltenVKConfigAtLoad g_moltenvk_config_at_load;
}

bool init_device() {
    Backend* b = backend();
    if (b->initialized) return true;

    // Not initialised, yet an instance is already sitting in the backend: a
    // previous call got as far as vkCreateInstance and then bailed out further
    // down (an extension check, no graphics queue family, ...). Every one of
    // those exits returns false without unwinding, and the function below
    // starts over from vkCreateInstance, so the old handle would simply be
    // overwritten. A leaked VkInstance is not a few bytes here — MoltenVK hangs
    // the Metal device, the physical-device objects and its whole queue
    // machinery off it, and none of that is reclaimed until the process exits.
    // Release it before rebuilding.
    if (b->instance != VK_NULL_HANDLE) {
        MITHRIL_LOG_WARN("vk", "init_device() retrying after an earlier partial "
                               "failure; destroying the leftover VkInstance first");
        vkDestroyInstance(b->instance, nullptr);
        b->instance = VK_NULL_HANDLE;
        b->physicalDevice = VK_NULL_HANDLE;
        b->createMetalSurfaceEXT = nullptr;
    }

    // ---- MoltenVK runtime configuration (root cause T) ----
    // Set critical MoltenVK environment variables BEFORE vkCreateInstance.
    // MoltenVK reads these once during instance creation; setting them after
    // has no effect. Amethyst-iOS sets these explicitly for stability.
    //
    // MVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS=1: vkQueueSubmit blocks until
    //   encoding is complete. Prevents submit races on iOS. (MoltenVK default
    //   is 1, but we set it explicitly for determinism.)
    //
    // MVK_CONFIG_PREFILL_METAL_COMMAND_BUFFERS=0 (SIGBUS 对齐崩溃根因修复):
    //   MoltenVK prefill 模式：
    //     0 = 提交时编码（vkQueueSubmit 时创建 Metal command buffer）— 默认
    //     1 = vkEndCommandBuffer() 时编码（deferred encoding）
    //     2 = 每条命令立即编码
    //
    //   原设为 1 是为了"避免 present 时 IOSurface 未绑定竞态"，但这是个
    //   误解：prefill=1 实际在 vkEndCommandBuffer() 时触发 deferred encoding，
    //   而 MoltenVK 命令池中的 MVKCmdBufferImageCopy 对象分配可能未满足
    //   8 字节对齐（ARMv8 ldr x9,[x9,#0xE0] 要求 8 字节对齐），导致 SIGBUS
    //   (BUS_ADRALN) 崩溃。崩溃栈：
    //     MVKCmdBufferImageCopy::encode → checkDeferredEncoding →
    //     MVKCommandBuffer::end() → vkEndCommandBuffer → commit_frame
    //
    //   改为 0（提交时编码）规避了 vkEndCommandBuffer 时的 deferred
    //   encoding 路径，从而避免命令池对齐崩溃。IOSurface 绑定竞态已通过
    //   commit_frame 中的 dummy render pass（CommandStream.cpp:706-768）
    //   和 imageAvailable semaphore 等待解决，不再依赖 prefill。
    //
    // MVK_CONFIG_RESUME_LOST_DEVICE=1: allow MoltenVK to continue using a Metal
    //   device after a recoverable loss when the underlying physical device
    //   remains usable. It does NOT recreate VkDevice; Mithril owns rebuild.
    //   Keeping this enabled avoids needlessly making recoverable faults sticky.
    //
    // MVK_CONFIG_SHADER_CONVERSION_FLIP_VERTEX_Y=0: MoltenVK's global vertex Y
    //   flip is DISABLED. Y flipping is now handled at the shader-translation
    //   layer (Shader.cpp inject_position_fixup with flip_y=true) ONLY for the
    //   default framebuffer (FBO 0). User-created FBOs use the non-flipped
    //   variant so their textures stay in GL Y-up orientation for correct
    //   sampling by GL shaders. The global MoltenVK flip applied Y inversion to
    //   ALL vertex shaders indiscriminately, which turned user-FBO texture
    //   content upside-down and was the root cause of the red/black screen.
    //   Deep reference: MobileGL GetShaderTransformFlags applies PositionYFlip
    //   only when currentDrawFBO->IsDefaultFramebuffer(), never globally.
    //
    // MVK_CONFIG_MAX_ACTIVE_METAL_COMMAND_BUFFERS_PER_QUEUE=2 (深度参考 MobileGL 缺口):
    //   限制每个 queue 同时未完成的 command buffer 数量。MoltenVK 默认
    //   是 64（即允许 64 个 command buffer 并发编码），每个 command buffer
    //   都会预分配 Metal 资源（编码器、IOSurface 引用等）。在 iPhone SE 3
    //   这种显存受限的设备上，64 个并发 command buffer 的 Metal 资源占用
    //   会加剧显存压力，导致 OOM。
    //   设为 kMaxFramesInFlight(2) 后，MoltenVK 最多维护 2 个活跃 command
    //   buffer（与我们的 per-slot 设计匹配），从默认 64 降到 2，显著降低
    //   峰值显存占用。
    //   注意：这个值必须 >= kMaxFramesInFlight（2），否则 submit 会被
    //   MoltenVK 阻塞等待前一个 command buffer 完成，可能导致死锁
    //   （若前一个的 fence 还没被 ensure_command_buffer_recording 等待）。
    apply_moltenvk_config();
    // MVK_CONFIG_VK_SEMAPHORE_SUPPORT_STYLE=2 (根因 E，深度参考 MoltenVK):
    //   强制使用 Metal 信号量（真实 GPU 侧同步）。Mithril 的同步设计完全
    //   依赖 Vulkan semaphore（imageAvailable: acquire→render；renderFinished:
    //   render→present）。MoltenVK 默认 VK_SEMAPHORE_SUPPORT_STYLE_METAL_EVENTS_WHERE_SAFE
    //   在 NVIDIA GPU 和 Rosetta2（x86_64 on Apple Silicon）上会退化为
    //   SingleQueue 模式，此时 vkQueueSubmit 的 semaphore wait/signal 全是 no-op
    //   （MVKSync.mm:87-101），GPU 侧无真实同步 → 渲染读取 stale image / present
    //   读取未完成像素 → 黑屏有声音。
    //   显式设为 2（METAL_EVENTS_WHERE_AVAILABLE）优先使用 MTLEvent，
    //   消除跨 MoltenVK 版本/平台的不确定性。参考 MoltenVK MVKDevice.mm:3621-3627。

    // ---- Instance ----
    std::vector<VkExtensionProperties> instExtProps;
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    instExtProps.resize(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, instExtProps.data());

    std::vector<const char*> instExts;
    // Portability enumeration is mandatory for MoltenVK-backed Vulkan.
    if (has_extension(instExtProps, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        instExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
    if (has_extension(instExtProps, VK_EXT_METAL_SURFACE_EXTENSION_NAME)) {
        instExts.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
    }
    if (has_extension(instExtProps, VK_KHR_SURFACE_EXTENSION_NAME)) {
        instExts.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    }
    // Android presentation surface. The name macro lives in vulkan_android.h,
    // which requires VK_USE_PLATFORM_ANDROID_KHR before <vulkan/vulkan.h>;
    // defining the string here keeps this TU platform-header-free, the same way
    // VK_EXT_METAL_SURFACE_EXTENSION_NAME is handled above.
#ifndef VK_KHR_ANDROID_SURFACE_EXTENSION_NAME
#define VK_KHR_ANDROID_SURFACE_EXTENSION_NAME "VK_KHR_android_surface"
#endif
    if (has_extension(instExtProps, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME)) {
        instExts.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
    }
    // Debug utils optional.
    bool wantDebugUtils = has_extension(instExtProps, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (wantDebugUtils) instExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    // Android WSI, asked for even when the driver does not admit to it.
    //
    // A Mesa HAL module driven directly - Turnip being the case in point -
    // reports its instance extensions as eight entries with no
    // VK_KHR_surface and no VK_KHR_android_surface, while still exporting the
    // WSI query commands (vkGetPhysicalDeviceSurfaceFormatsKHR and
    // vkGetPhysicalDeviceSurfaceSupportKHR are both present in its dynamic
    // table). That is not a contradiction: on Android those two instance
    // extensions are supplied by the loader, which is why Zink and ANGLE only
    // ever observe them through libvulkan.so. Driving the driver directly
    // means inheriting the driver's list, and with it no way to make a
    // presentable surface.
    //
    // So request them regardless. If they genuinely are not there,
    // vkCreateInstance answers VK_ERROR_EXTENSION_NOT_PRESENT and the second
    // variant below retries without them; nothing is lost by trying.
    std::vector<const char*> instExtsWsi = instExts;
#ifdef __ANDROID__
    if (!has_extension(instExtProps, VK_KHR_SURFACE_EXTENSION_NAME)) {
        instExtsWsi.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    }
    if (!has_extension(instExtProps, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME)) {
        instExtsWsi.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
    }
#endif

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Mithril-Wrapper";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Mithril-Wrapper";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    // Requesting an apiVersion the platform does not implement is fatal:
    // vkCreateInstance returns VK_ERROR_INCOMPATIBLE_DRIVER. Android devices
    // commonly expose only Vulkan 1.1 - the system loader on Snapdragon, even
    // when a Turnip driver is installed - so asking for 1.2 unconditionally
    // leaves the renderer unable to start there.
    //
    // Step down until a level is accepted and record which one, so the 1.2
    // feature structs and 1.3 commands below can gate on it.
    const bool portabilityEnumeration =
        has_extension(instExtProps, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    b->instanceApiVersion = VK_API_VERSION_1_1;

    // 1.3 first: on a 1.3 device (Turnip reports api 0x40316b) the features we
    // hard-depend on - dynamic rendering and extended dynamic state - are CORE,
    // so the driver is not obliged to advertise them as extensions. Pinning the
    // instance to 1.2 forces them to come from the extension list, and a driver
    // that promoted them to core may not report them there. Stepping down is
    // still safe: an unsupported level just fails and we try the next one.
    // 覆盖 1.4 / 1.3 / 1.2 / 1.1 / 1.0：从高到低逐个试，第一个被接受的即
    // 实例版本。新增 1.4（MoltenVK 1.2.9+ / 新驱动）与 1.0（老设备兜底）后，
    // 任何 Vulkan 版本都能拿到一个可用实例，不会因为「请求的版本平台不支持」
    // 而 VK_ERROR_INCOMPATIBLE_DRIVER 直接起不来。
    static const uint32_t kApiLevels[] = {
        VK_API_VERSION_1_4,
        VK_API_VERSION_1_3,
        VK_API_VERSION_1_2,
        VK_API_VERSION_1_1,
        VK_API_VERSION_1_0,
    };
    bool instanceCreated = false;
    const std::vector<const char*>* const variants[] = { &instExtsWsi, &instExts };
    const int variantCount = (instExtsWsi.size() != instExts.size()) ? 2 : 1;
    for (int variant = 0; variant < variantCount && !instanceCreated; ++variant) {
        const std::vector<const char*>& tryExts = *variants[variant];
        for (uint32_t level : kApiLevels) {
            appInfo.apiVersion = level;

            VkInstanceCreateInfo instCI{};
            instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            instCI.pApplicationInfo = &appInfo;
            instCI.enabledExtensionCount = (uint32_t)tryExts.size();
            instCI.ppEnabledExtensionNames = tryExts.data();
            // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR is REQUIRED so
            // vkEnumeratePhysicalDevices returns the MoltenVK ICD. It is only legal
            // together with VK_KHR_portability_enumeration: setting the flag without
            // the extension is invalid usage, and Android's loader has no such
            // extension at all, so it must be conditional.
            if (portabilityEnumeration) {
                instCI.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            }

            if (vkCreateInstance(&instCI, nullptr, &b->instance) == VK_SUCCESS) {
                b->instanceApiVersion = level;
                instanceCreated = true;
                MITHRIL_LOG_INFO("vk", "VkInstance created at apiVersion %u.%u",
                                 VK_VERSION_MAJOR(level), VK_VERSION_MINOR(level));
                break;
            }
            MITHRIL_LOG_WARN("vk", "vkCreateInstance failed at apiVersion %u.%u; stepping down",
                             VK_VERSION_MAJOR(level), VK_VERSION_MINOR(level));
        }
        if (!instanceCreated && variantCount == 2) {
            MITHRIL_LOG_WARN("vk", "retrying vkCreateInstance without the WSI extensions");
        }
    }
    if (!instanceCreated) {
        MITHRIL_LOG_ERROR("vk", "vkCreateInstance failed at every supported apiVersion");
        return false;
    }

    if (wantDebugUtils) {
        // Best-effort debug messenger (optional, never fatal).
        auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(b->instance, "vkCreateDebugUtilsMessengerEXT");
        if (fn) {
            VkDebugUtilsMessengerCreateInfoEXT mic{};
            mic.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            mic.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            mic.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            mic.pfnUserCallback = debug_callback;
            VkDebugUtilsMessengerEXT messenger;
            fn(b->instance, &mic, nullptr, &messenger);
            // Messenger intentionally leaked (process lifetime); the callback
            // stays armed for the whole session.
        }
    }

    // Resolve vkCreateMetalSurfaceEXT (used by SwapchainMetal.mm). Stored as
    // PFN_vkVoidFunction to avoid needing VK_USE_PLATFORM_METAL_EXT here
    // (that macro pulls in <Metal/Metal.h>, which is ObjC-only). The .mm
    // translation unit casts it to PFN_vkCreateMetalSurfaceEXT at the call.
    b->createMetalSurfaceEXT =
        vkGetInstanceProcAddr(b->instance, "vkCreateMetalSurfaceEXT");
    if (!b->createMetalSurfaceEXT) {
        // Diagnostic: this function is only exported when VK_EXT_metal_surface
        // is enabled at instance creation. Dump the enumerated instance
        // extensions so a null resolution is self-explaining on-device.
        MITHRIL_LOG_WARN("vk", "vkCreateMetalSurfaceEXT not resolved; "
                              "enumerated instance extensions:");
        for (const auto& e : instExtProps) {
            MITHRIL_LOG_WARN("vk", "  %s", e.extensionName);
        }
    }

#ifdef __ANDROID__
    // Whether presentation is possible at all is decided here, not at the
    // first eglCreateWindowSurface: if this entrypoint is absent the swapchain
    // can never be built, and the failure would otherwise surface much later
    // as a crash on a null surface handle.
    MITHRIL_LOG_INFO("vk", "vkCreateAndroidSurfaceKHR %s",
                     vkGetInstanceProcAddr(b->instance, "vkCreateAndroidSurfaceKHR")
                         ? "resolvable" : "NOT resolvable");
#endif

    // ---- Physical device ----
    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(b->instance, &gpuCount, nullptr);
    if (gpuCount == 0) {
        MITHRIL_LOG_ERROR("vk", "No Vulkan physical devices (MoltenVK not linked?)");
        return false;
    }
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(b->instance, &gpuCount, gpus.data());
    b->physicalDevice = gpus[0];  // iOS has exactly one Metal device.

    vkGetPhysicalDeviceProperties(b->physicalDevice, &b->props);
    MITHRIL_LOG_INFO("vk", "Physical device: %s (api 0x%x, driver 0x%x)",
                     b->props.deviceName, b->props.apiVersion, b->props.driverVersion);

    // FIX (MobileGL-style VRAM monitoring): 不再依赖 maxMemoryAllocationCount
    // 做压力判断。MoltenVK 报告的 maxMemoryAllocationCount 不可信（10亿+），
    // 且 allocation count 不能反映真实压力（16MB 纹理和 64B UBO 各算 1 次）。
    //
    // MobileGL 的做法（VulkanRenderer::GetMemoryUsage / TryDrainFrameTransients）：
    // 1. 查询 VkPhysicalDeviceMemoryProperties 获取实际堆大小
    // 2. 跟踪已分配字节数
    // 3. 在 80% 堆使用率时主动 drain + rewind transient arena
    //
    // 我们镜像这个策略：保留 maxMemoryAllocationCount 仅用于诊断日志，
    // 实际 GC 触发改为基于 currentVramBytes / totalVramBytes 的字节阈值。
    b->maxMemoryAllocationCount = b->props.limits.maxMemoryAllocationCount;
    MITHRIL_LOG_INFO("vk", "maxMemoryAllocationCount = %u (reported by driver, "
                     "diagnostic only — VRAM pressure tracked by bytes)",
                     (unsigned)b->maxMemoryAllocationCount);

    // ---- Query REAL available memory (MobileGL-style, iOS-native) ----
    // 深度参考 MobileGL VulkanRenderer::GetMemoryUsage：它用 OS 原生内存压力 API
    // 设定真实预算，而非依赖驱动报告的 heap size。
    //
    // 根因分析（device lost 持续崩溃）：
    //   旧代码把 VRAM 预算设为 1.5GB（80% 阈值 = 1.2GB）。但 iPhone X 只有 3GB
    //   统一内存：iOS 系统占 ~1GB，JVM + Minecraft CPU 端占 ~1GB，GPU 实际只能
    //   安全使用 ~400-500MB。MoltenVK 报告的 heap size = 整个设备 RAM（3GB），
    //   完全不可信。结果：GPU 在我们的 1.2GB 阈值触发前就已 OOM → GPU Address
    //   Fault → VK_ERROR_DEVICE_LOST（Metal 上不可恢复）→ swapchain 重建死循环。
    //
    // 修复：用 os_proc_available_memory() 获取进程真实可用内存，取保守比例给 GPU，
    // 并把 GC 阈值从 80% 降到 60%（在驱动 fault 之前就释放资源）。
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(b->physicalDevice, &memProps);
    b->memProps = memProps;
    VkDeviceSize maxHeapSize = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].size > maxHeapSize) {
            maxHeapSize = memProps.memoryHeaps[i].size;
        }
    }

    // 查询进程真实可用内存。
    //   iOS: os_proc_available_memory() = Jetsam kill 之前还能分配的字节数。
    //   macOS: 没有 Jetsam，os_proc_available_memory() API_UNAVAILABLE(macos)；
    //          改用 sysctl hw.memsize 取物理内存（Apple Silicon 为统一内存）。
    VkDeviceSize availableBytes = 0;
#if defined(__APPLE__) && TARGET_OS_IPHONE
    availableBytes = os_proc_available_memory();
#elif !defined(__APPLE__)
    // Android / Linux: no Jetsam and no sysctl hw.memsize, so the old code left
    // availableBytes at 0 and took the 256 MB fallback regardless of the
    // device. That is the same failure shape as the iOS 512 MB cap raised
    // earlier: when the budget is hit, image allocations fail, textures stay
    // unbacked, and sampling them paints the frame solid red. POSIX is enough
    // here - total physical pages times page size - and the existing
    // 25%-of-available rule then sizes the budget as it does on Apple.
    {
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long pageSize = sysconf(_SC_PAGESIZE);
        if (pages > 0 && pageSize > 0) {
            availableBytes = static_cast<VkDeviceSize>(pages) *
                             static_cast<VkDeviceSize>(pageSize);
        }
    }
#elif defined(__APPLE__)
    {
        int64_t phys = 0;
        size_t physLen = sizeof(phys);
        if (sysctlbyname("hw.memsize", &phys, &physLen, nullptr, 0) == 0 && phys > 0) {
            availableBytes = static_cast<VkDeviceSize>(phys);
        }
    }
#endif

    // 平台相关的硬上限 / 下限 / 回退预算。
#if defined(__APPLE__) && !TARGET_OS_IPHONE
    // macOS: 统一内存容量大、无 Jetsam；25% 物理内存，硬上限 4GB。
    constexpr VkDeviceSize kPlatformMaxVramBudget = 4096ULL * 1024 * 1024;  // 4 GB
    constexpr VkDeviceSize kPlatformMinVramBudget = 512ULL * 1024 * 1024;    // 512 MB
    constexpr VkDeviceSize kPlatformFallbackBudget = 1024ULL * 1024 * 1024; // 1 GB
#else
    // iOS / 其他平台。
    //
    // The cap used to be 512 MB - one eighth of the macOS cap. It was a guess,
    // not a measurement, and it is the only thing here that MobileGL has no
    // equivalent of: MobileGL sizes its budget from the device and otherwise
    // lets the 25%-of-available rule be the limit.
    //
    // The cap matters because it is a hard ceiling on image allocations, and
    // an image that cannot be allocated stays unbacked - sampling it paints
    // the frame solid red. That is the reported symptom: red on device only,
    // and macOS (4 GB cap) is clean. Raised to 1.5 GB so the 25% rule, which
    // already keeps three quarters of memory for the JVM, is what actually
    // limits the GPU instead of an arbitrary constant.
    //
    // MITHRIL_VRAM_BUDGET_MB still overrides this outright, for confirming or
    // ruling the hypothesis out in a single run without a rebuild.
    constexpr VkDeviceSize kPlatformMaxVramBudget = 1536ULL * 1024 * 1024;   // 1.5 GB
    constexpr VkDeviceSize kPlatformMinVramBudget = 96ULL * 1024 * 1024;     // 96 MB
    constexpr VkDeviceSize kPlatformFallbackBudget = 256ULL * 1024 * 1024;   // 256 MB
#endif

    VkDeviceSize gpuBudget = 0;
    if (availableBytes > 0) {
        // 取可用/物理内存的 25% 给 GPU — 留 75% 给 CPU/JVM 增长。
        gpuBudget = (availableBytes * 25) / 100;
        MITHRIL_LOG_INFO("vk", "available memory = %llu MB, "
                         "GPU budget = 25%% = %llu MB",
                         (unsigned long long)(availableBytes / (1024*1024)),
                         (unsigned long long)(gpuBudget / (1024*1024)));
    } else {
        gpuBudget = kPlatformFallbackBudget;
        MITHRIL_LOG_INFO("vk", "available memory unavailable, "
                         "using fallback budget %llu MB",
                         (unsigned long long)(gpuBudget / (1024*1024)));
    }

    if (gpuBudget > kPlatformMaxVramBudget) {
        MITHRIL_LOG_INFO("vk", "GPU budget %llu MB exceeds platform cap, "
                          "clamping to %llu MB",
                          (unsigned long long)(gpuBudget / (1024*1024)),
                          (unsigned long long)(kPlatformMaxVramBudget / (1024*1024)));
        gpuBudget = kPlatformMaxVramBudget;
    }
    if (gpuBudget < kPlatformMinVramBudget) {
        MITHRIL_LOG_WARN("vk", "GPU budget %llu MB below floor, "
                         "raising to %llu MB (may be unstable)",
                         (unsigned long long)(gpuBudget / (1024*1024)),
                         (unsigned long long)(kPlatformMinVramBudget / (1024*1024)));
        gpuBudget = kPlatformMinVramBudget;
    }

    // Diagnostic override (default behaviour unchanged).
    //
    // MobileGL has no equivalent cap, which is the reason to suspect this one:
    // when the budget is hit, image allocations fail, textures stay unbacked,
    // and sampling them is what paints the frame solid red. That matches the
    // reported symptom - red on device only, macOS clean.
    //
    // MITHRIL_VRAM_BUDGET_MB forces the budget in MB so the hypothesis can be
    // confirmed or ruled out in one run without a rebuild, and gives a real
    // workaround if it is confirmed.
    if (const char* e = std::getenv("MITHRIL_VRAM_BUDGET_MB")) {
        const long long mb = std::strtoll(e, nullptr, 10);
        if (mb > 0) {
            MITHRIL_LOG_INFO("vk", "VRAM budget overridden by "
                             "MITHRIL_VRAM_BUDGET_MB=%lld MB (auto was %llu MB)",
                             mb, (unsigned long long)(gpuBudget / (1024*1024)));
            gpuBudget = static_cast<VkDeviceSize>(mb) * 1024ULL * 1024ULL;
        }
    }
    // 不能超过物理 heap size
    if (maxHeapSize > 0 && gpuBudget > maxHeapSize) {
        gpuBudget = maxHeapSize;
    }

    b->totalVramBytes = gpuBudget;
    // GC 阈值：50% — 在驱动 fault 之前触发。
    // 旧 60% 仍太晚：大纹理上传可在单帧内从 40% 跳到 90%。
    // 50% of 256MB = 128MB → GC 在 128MB 时触发，留 128MB 余量。
    b->vramPressureThreshold = (b->totalVramBytes * 5) / 10;
    MITHRIL_LOG_INFO("vk", "VRAM budget: %llu MB, GC threshold: %llu MB (50%%)",
                     (unsigned long long)(b->totalVramBytes / (1024*1024)),
                     (unsigned long long)(b->vramPressureThreshold / (1024*1024)));

    // ---- Queue family ----
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(b->physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(b->physicalDevice, &qfCount, qfProps.data());
    b->graphicsFamily = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            b->graphicsFamily = i;
            break;
        }
    }
    if (b->graphicsFamily == 0xFFFFFFFFu) {
        MITHRIL_LOG_ERROR("vk", "No graphics queue family");
        return false;
    }

    // ---- Device ----
    // Verify the portability-subset + swapchain extensions are available.
    std::vector<VkExtensionProperties> devExtProps;
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(b->physicalDevice, nullptr, &devExtCount, nullptr);
    devExtProps.resize(devExtCount);
    vkEnumerateDeviceExtensionProperties(b->physicalDevice, nullptr, &devExtCount, devExtProps.data());

    MITHRIL_LOG_INFO("vk", "device extensions: %u reported; swapchain=%s",
                     devExtCount,
                     has_extension(devExtProps, VK_KHR_SWAPCHAIN_EXTENSION_NAME)
                         ? "yes" : "no");
    std::vector<const char*> devExts;
    if (has_extension(devExtProps, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        devExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    // VK_KHR_portability_subset MUST be enabled if present (MoltenVK always
    // advertises it).
    if (has_extension(devExtProps, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        devExts.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
    // VK_KHR_dynamic_rendering: lets us create pipelines + record render passes
    // without a VkRenderPass object (simpler than managing compat render passes).
    //
    // 注意这是**硬依赖**，不是可选优化。整个 CommandStream.cpp 只有
    // vkCmdBeginRendering 一条录制路径，没有传统 VkRenderPass/VkFramebuffer
    // 的退路。我们请求的是 Vulkan 1.2，而 dynamic_rendering 要到 1.3 才进核心，
    // 所以在 1.2 上它必须以扩展形式存在。
    //
    // 缺席时 vkGetDeviceProcAddr("vkCmdBeginRendering") 返回 nullptr，
    // CommandStream.cpp:826 的 `if (fn)` 会安静地跳过 —— pass 被标记成
    // active，draw 照常录制，但没有任何附件被绑定，最终什么都画不出来。
    // 那是最难排查的一类故障：没有报错、没有验证层警告，只有一块黑屏。
    //
    // 与其那样，不如在这里直接失败并说清原因。MoltenVK 从 1.1.0 起支持该
    // 扩展（对应 iOS 14+ / macOS 11+），低于此版本的环境本来也跑不动 MC。
    // Present as an extension OR promoted to core. VK_KHR_dynamic_rendering
    // became core in Vulkan 1.3, so a 1.3 device is not required to list it in
    // vkEnumerateDeviceExtensionProperties. gating purely on the extension name
    // rejects such a device outright and the renderer never starts.
    const bool dynRenderingExt =
        has_extension(devExtProps, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    // 非 const：init_device 末尾会根据 vkCmdBeginRendering 的实际可解析性
    // 再降级一次（声称支持但入口取不到 → 走传统路径）。
    bool hasDynamicRendering =
        dynRenderingExt || b->props.apiVersion >= VK_API_VERSION_1_3;
    if (dynRenderingExt) {
        devExts.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    } else if (!hasDynamicRendering) {
        // 不再是硬失败：CommandStream.cpp 的 begin_render_pass /
        // end_render_pass 在无该扩展时改走传统 VkRenderPass + VkFramebuffer
        // （RenderPassCompat.cpp），Pipeline.cpp 也据此把管线挂在真实
        // renderPass 上而不是 VkPipelineRenderingCreateInfo。
        //
        // 这正是 Zink / MobileGL 能吃下 turnip（Android HAL 构建不暴露 WSI，
        // 也不暴露 dynamic_rendering）的原因：它们从一开始就保留传统路径。
        // Adreno 619 上的系统驱动只有 Vulkan 1.1.128，同样落到这里。
        MITHRIL_LOG_WARN("vk",
            "VK_KHR_dynamic_rendering 不可用（%s，Vulkan %u.%u.%u）——"
            "命令录制改用传统 VkRenderPass/VkFramebuffer 路径。",
            b->props.deviceName,
            VK_VERSION_MAJOR(b->props.apiVersion),
            VK_VERSION_MINOR(b->props.apiVersion),
            VK_VERSION_PATCH(b->props.apiVersion));
        hasDynamicRendering = false;
    }
    // VK_EXT_extended_dynamic_state: vkCmdSetCullMode/FrontFace/DepthTestEnable/
    // DepthWriteEnable/DepthCompareOp etc. without rebuilding pipelines.
    //
    // 同样是硬依赖。CommandStream.cpp 的 backend_set_cull_mode /
    // set_front_face / set_depth_test 是**直接调用** vkCmdSetCullMode 那一族，
    // 而不是通过 vkGetDeviceProcAddr 取指针后判空。在 Vulkan 1.2 上这些符号
    // 由 loader 解析：扩展没启用时它们是空指针，直接调用 = 段错误崩溃，
    // 比黑屏还糟。而且 Pipeline.cpp:633 已经把 cullMode 设成 NONE 并声明为
    // 动态状态，管线里根本没有静态的剔除配置可回退。
    //
    // MoltenVK 从 1.1.0 起支持，与 dynamic_rendering 的门槛一致。
    // Same promotion story as dynamic rendering: VK_EXT_extended_dynamic_state
    // is core in Vulkan 1.3.
    const bool extDynStateExt =
        has_extension(devExtProps, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
    bool hasExtDynState =
        extDynStateExt || b->props.apiVersion >= VK_API_VERSION_1_3;
    if (extDynStateExt) {
        devExts.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
    } else if (!hasExtDynState) {
        // 同样不再是硬失败。缺少时剔除模式 / 正面朝向 / 深度测试全部烘进
        // 静态管线（Pipeline.cpp），并且纳入管线缓存键（hash_signature），
        // 所以状态切换仍然生效 —— 只是变成「换管线」而不是「下发动态状态」。
        // CommandStream.cpp 的 backend_set_cull_mode 那一族本来就有
        // extendedDynamicStateSupported 判空，不会调用空函数指针。
        MITHRIL_LOG_WARN("vk",
            "VK_EXT_extended_dynamic_state 不可用（%s，Vulkan %u.%u.%u）——"
            "剔除/正面朝向/深度状态改为烘进静态管线。",
            b->props.deviceName,
            VK_VERSION_MAJOR(b->props.apiVersion),
            VK_VERSION_MINOR(b->props.apiVersion),
            VK_VERSION_PATCH(b->props.apiVersion));
        hasExtDynState = false;
    }
    // FIX (root cause AE - GL_UNSIGNED_BYTE 索引支持):
    // VK_EXT_index_type_uint8 提供 VK_INDEX_TYPE_UINT8，让 GL_UNSIGNED_BYTE
    // 索引可以正确解释为 1 字节/索引。不启用则被当作 UINT16（2 字节）→
    // 索引值错乱 → 几何腐败 → 红屏。深度对照 MobileGL VulkanRenderer.cpp:3093-3109。
    if (has_extension(devExtProps, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME)) {
        devExts.push_back(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME);
    }
    // FIX (root cause AF - Primitive Restart):
    // VK_EXT_primitive_topology_list_restart 让 list topology（POINT_LIST/
    // LINE_LIST/TRIANGLE_LIST）也支持 primitiveRestartEnable。GL 的
    // GL_PRIMITIVE_RESTART / GL_PRIMITIVE_RESTART_FIXED_INDEX 在 strip/fan/list
    // 上都应生效；不启用则 strip 在 restart 索引处不断开 → 连接到无效顶点 →
    // 几何腐败 → 红屏。深度对照 MobileGL VulkanRenderer.cpp:3861-3877。
    if (has_extension(devExtProps, VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME)) {
        devExts.push_back(VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME);
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCI{};
    queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCI.queueFamilyIndex = b->graphicsFamily;
    queueCI.queueCount = 1;
    queueCI.pQueuePriorities = &queuePriority;

    // Feature chain: enable dynamic rendering + extended dynamic state so the
    // vkCmdBeginRendering / vkCmdSetCullMode etc. calls in CommandStream.cpp
    // are valid. Without these features a strict driver rejects pipeline
    // creation that carries VkPipelineRenderingCreateInfo and rejects the
    // dynamic-state vkCmdSet* calls. (MoltenVK is lenient but the spec
    // requires the features to be enabled.)
    /* FIX (P0 —— 设备创建失败，整个渲染器起不来):
     *
     * 这两个 feature 结构体原先是**无条件**挂进 pNext 且写死 VK_TRUE 的，
     * 而上面对应的两个扩展却是 has_extension() 条件启用。二者不一致：
     * 一旦设备不暴露 VK_KHR_dynamic_rendering 或 VK_EXT_extended_dynamic_state
     * （MoltenVK 1.1.x 及更早、iOS 13/14 时代的设备就是这样），扩展没进
     * devExts，但请求该 feature 的结构体仍在链上 → vkCreateDevice 按规范返回
     * VK_ERROR_FEATURE_NOT_PRESENT → init_device 直接 return false → 起不来。
     *
     * 同文件里 index_type_uint8、list_restart 以及那 24 个核心 feature 全都
     * 正确做了 gate，唯独最关键的这两个破了例。
     *
     * 改法：只有扩展确实可用时才把对应 feature 结构体链进去。链头从固定的
     * dynRenderFeat 改为动态的 featureChainHead，两个结构体都可能缺席。
     */
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extDynStateFeat{};
    extDynStateFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT;
    extDynStateFeat.extendedDynamicState = VK_TRUE;

    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynRenderFeat{};
    dynRenderFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
    dynRenderFeat.dynamicRendering = VK_TRUE;

    void*                featureChainHead = nullptr;
    VkBaseOutStructure*  chainTail        = nullptr;
    auto append_feature = [&](void* s) {
        VkBaseOutStructure* n = (VkBaseOutStructure*)s;
        n->pNext = nullptr;
        if (!featureChainHead) { featureChainHead = s; chainTail = n; }
        else                   { chainTail->pNext = n; chainTail = n; }
    };
    if (hasDynamicRendering) append_feature(&dynRenderFeat);
    if (hasExtDynState)      append_feature(&extDynStateFeat);

    // 记录下来供录制期分支使用：没有扩展就不能调 vkCmdSetCullMode 那一族，
    // 也不能用 vkCmdBeginRendering，必须走静态管线状态 / 传统 RenderPass。
    b->dynamicRenderingSupported   = hasDynamicRendering;
    b->extendedDynamicStateSupported = hasExtDynState;

    // FIX (root cause Q): When VK_KHR_portability_subset is enabled (it is, on
    // MoltenVK — Device.cpp:197-199), the Vulkan spec REQUIRES
    // VkPhysicalDevicePortabilitySubsetFeaturesKHR to be chained into
    // VkDeviceCreateInfo::pNext. Without it, all portability-subset features
    // default to VK_FALSE, which may cause pipelines using portability-
    // constrained features (triangleFans, separateStencilTextureFilter, etc.)
    // to silently fail on MoltenVK → black screen. MobileGL chains this struct.
    // Query the supported features first; only enable what the device supports.
    VkPhysicalDevicePortabilitySubsetFeaturesKHR portSubsetFeat{};
    portSubsetFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR;
    portSubsetFeat.pNext = nullptr;
    if (has_extension(devExtProps, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        // Query the supported portability-subset features. vkGetPhysicalDevice-
        // Features2 fills portSubsetFeat with VK_TRUE for features the device
        // supports; chaining the struct into VkDeviceCreateInfo::pNext enables
        // exactly those supported features. We do NOT force any feature on,
        // so unsupported features stay VK_FALSE (spec-safe).
        VkPhysicalDeviceFeatures2 feat2{};
        feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feat2.pNext = &portSubsetFeat;
        vkGetPhysicalDeviceFeatures2(b->physicalDevice, &feat2);
        // Chain portability-subset at the END of the feature chain
        // (after extended-dynamic-state). The order does not matter for
        // correctness; we just need all feature structs in the pNext chain.
        //
        // FIX: 必须走 append_feature，不能直接写 chainTail->pNext。
        // dynamic_rendering / extended_dynamic_state 现在是条件链入的，
        // 两者都缺席时 chainTail == nullptr，直接解引用就是空指针崩溃。
        append_feature(&portSubsetFeat);
    }

    // FIX (root cause AE - GL_UNSIGNED_BYTE 索引支持):
    // 启用 VK_EXT_index_type_uint8 的 feature，让 vkCmdBindIndexBuffer 接受
    // VK_INDEX_TYPE_UINT8。仅当设备支持该扩展时链入（spec-safe）。
    // 深度对照 MobileGL VulkanRenderer.cpp:3093-3109。
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT indexTypeUint8Feat{};
    indexTypeUint8Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT;
    indexTypeUint8Feat.indexTypeUint8 = VK_TRUE;
    indexTypeUint8Feat.pNext = nullptr;
    if (has_extension(devExtProps, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME)) {
        append_feature(&indexTypeUint8Feat);
    }

    // FIX (root cause AF - Primitive Restart):
    // 启用 VK_EXT_primitive_topology_list_restart 的 feature，让 list topology
    // 上的 primitiveRestartEnable=VK_TRUE 生效（strip/fan topology 在核心
    // 1.2 已支持，list 需此扩展）。仅当设备支持时链入。
    // 深度对照 MobileGL VulkanRenderer.cpp:3861-3877。
    VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT primTopoListRestartFeat{};
    primTopoListRestartFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT;
    primTopoListRestartFeat.primitiveTopologyListRestart = VK_TRUE;
    primTopoListRestartFeat.pNext = nullptr;
    if (has_extension(devExtProps, VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME)) {
        append_feature(&primTopoListRestartFeat);
        b->listRestartSupported = true;
    } else {
        // FIX (root cause AS): remember that we do NOT have it. Pipeline.cpp
        // must then suppress primitiveRestartEnable on list topologies —
        // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252 makes
        // that combination illegal, and an illegal pipeline is not created at
        // all, so every affected draw silently disappears. MoltenVK does not
        // currently expose this extension, making this the normal path.
        b->listRestartSupported = false;
        MITHRIL_LOG_INFO("vk",
                         "VK_EXT_primitive_topology_list_restart unavailable; "
                         "primitive restart will be suppressed on list topologies");
    }

    // ---- Core VkPhysicalDeviceFeatures ----
    // Without pEnabledFeatures, all core features default to VK_FALSE, which
    // means ANY pipeline/shader using them will fail vkCreateGraphicsPipelines
    // and get negatively cached in Pipeline.cpp's failedSignatures — the draw
    // is then permanently skipped, manifesting as black-screen-with-sound.
    // MobileGL (VulkanRenderer.cpp:7291-7348) explicitly enables ~17 features
    // here; we mirror that set, gated by what the physical device actually
    // supports so we never request an unsupported feature (which would cause
    // vkCreateDevice to fail).
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(b->physicalDevice, &supported);

    VkPhysicalDeviceFeatures enabled{};
    enabled.robustBufferAccess            = supported.robustBufferAccess;
    enabled.independentBlend              = supported.independentBlend;
    enabled.fillModeNonSolid              = supported.fillModeNonSolid;
    enabled.dualSrcBlend                  = supported.dualSrcBlend;
    enabled.logicOp                       = supported.logicOp;
    enabled.shaderClipDistance            = supported.shaderClipDistance;
    enabled.shaderCullDistance            = supported.shaderCullDistance;
    enabled.wideLines                     = supported.wideLines;
    enabled.largePoints                   = supported.largePoints;
    enabled.shaderInt64                   = supported.shaderInt64;
    enabled.vertexPipelineStoresAndAtomics= supported.vertexPipelineStoresAndAtomics;
    enabled.fragmentStoresAndAtomics      = supported.fragmentStoresAndAtomics;
    enabled.shaderStorageImageExtendedFormats    = supported.shaderStorageImageExtendedFormats;
    enabled.shaderStorageImageReadWithoutFormat  = supported.shaderStorageImageReadWithoutFormat;
    enabled.shaderStorageImageWriteWithoutFormat = supported.shaderStorageImageWriteWithoutFormat;
    enabled.drawIndirectFirstInstance     = supported.drawIndirectFirstInstance;
    enabled.multiDrawIndirect             = supported.multiDrawIndirect;
    enabled.samplerAnisotropy             = supported.samplerAnisotropy;
    enabled.depthBounds                   = supported.depthBounds;
    enabled.occlusionQueryPrecise         = supported.occlusionQueryPrecise;
    enabled.pipelineStatisticsQuery       = supported.pipelineStatisticsQuery;
    enabled.multiViewport                 = supported.multiViewport;
    enabled.imageCubeArray                = supported.imageCubeArray;
    enabled.alphaToOne                    = supported.alphaToOne;
    // GL 4.0 ARB_sample_shading: required before a pipeline may set
    // sampleShadingEnable. Without it, glMinSampleShading would build an
    // illegal pipeline rather than a slower one.
    enabled.sampleRateShading             = supported.sampleRateShading;
    // Remember whether one vkCmdDrawIndirect may issue drawCount > 1; the
    // indirect draw path falls back to a loop when it may not.
    b->multiDrawIndirectSupported = supported.multiDrawIndirect == VK_TRUE;

    // GL 4.6 ARB_indirect_parameters — glMultiDraw*IndirectCount 需要
    // vkCmdDrawIndirectCount / vkCmdDrawIndexedIndirectCount。这是 Vulkan 1.2
    // core 特性，位于 VkPhysicalDeviceVulkan12Features（不在 1.0 的
    // VkPhysicalDeviceFeatures 里）。必须链入 vkGetPhysicalDeviceFeatures2
    // 查询支持值，再链入 VkDeviceCreateInfo 启用。MoltenVK 1.2.x 报告它。
    // Declared at function scope on purpose: append_feature() stores this
    // address in the chain handed to vkCreateDevice, so it has to outlive the
    // gate below.
    VkPhysicalDeviceVulkan12Features vulkan12Feat{};
    vulkan12Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Feat.pNext = nullptr;
    // VkPhysicalDeviceVulkan12Features only exists from Vulkan 1.2. Handing its
    // sType to a 1.1 device asks the driver to interpret a structure it has
    // never seen - at best ignored, at worst the whole feature chain is
    // rejected. Gate on the level actually granted to the instance AND the one
    // the physical device reports, so a 1.1 device simply reports the feature
    // unsupported and keeps the existing fallback.
    if (b->instanceApiVersion >= VK_API_VERSION_1_2 &&
        b->props.apiVersion >= VK_API_VERSION_1_2) {
        {
            VkPhysicalDeviceFeatures2 feat2{};
            feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            feat2.pNext = &vulkan12Feat;
            vkGetPhysicalDeviceFeatures2(b->physicalDevice, &feat2);
            b->drawIndirectCountSupported = vulkan12Feat.drawIndirectCount == VK_TRUE;
        }
        // 仅当设备支持时链入启用；不支持则保持默认（spec-safe，启用不支持的特性
        // 会让 vkCreateDevice 返回 VK_ERROR_FEATURE_NOT_PRESENT）。
        if (b->drawIndirectCountSupported) append_feature(&vulkan12Feat);
    } else {
        b->drawIndirectCountSupported = false;
        MITHRIL_LOG_INFO("vk", "Vulkan 1.1 device: drawIndirectCount unavailable "
                         "(GL 4.6 indirect_parameters falls back to a loop)");
    }

    b->sampleRateShadingSupported = supported.sampleRateShading == VK_TRUE;

    VkDeviceCreateInfo devCI{};
    devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    // FIX (P0-2): 链头必须是动态构建的 featureChainHead，而不是写死的
    // dynRenderFeat。dynamic_rendering 扩展在老 MoltenVK / iOS 上可能缺席，
    // 此时把它的 feature 结构体强行链入会让 vkCreateDevice 直接返回
    // VK_ERROR_FEATURE_NOT_PRESENT，整个设备创建失败 → 渲染器起不来。
    // featureChainHead 可能为 nullptr（所有可选 feature 都缺席），这是合法的。
    devCI.pNext = featureChainHead;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos = &queueCI;
    devCI.enabledExtensionCount = (uint32_t)devExts.size();
    devCI.ppEnabledExtensionNames = devExts.data();
    devCI.pEnabledFeatures = &enabled;

    if (vkCreateDevice(b->physicalDevice, &devCI, nullptr, &b->device) != VK_SUCCESS) {
        MITHRIL_LOG_ERROR("vk", "vkCreateDevice failed");
        return false;
    }
    vkGetDeviceQueue(b->device, b->graphicsFamily, 0, &b->graphicsQueue);

    // ---- Vulkan 1.3 command pointers ----
    // Resolved rather than linked: Android's libvulkan.so does not export
    // these, so linking them broke the Android build even though every
    // translation unit compiled.
    //
    // Resolution needs a fallback chain. They are Vulkan 1.3 core, but the
    // device here is 1.2 (MoltenVK), and vkGetDeviceProcAddr may legitimately
    // return null for a core command of a version the device does not
    // implement - even when the implementation exports the symbol, which is
    // why linking worked on Apple. Querying the plain name and then the EXT
    // name, through both the device and the instance, covers all three cases:
    //   - 1.3 device:            core name via device
    //   - MoltenVK (1.2 + EXT):  EXT name, or core name via instance
    //   - Android loader:        core name via device dispatch trampoline
    auto resolve_cmd = [&](const char* coreName, const char* extName) -> PFN_vkVoidFunction {
        if (PFN_vkVoidFunction p = vkGetDeviceProcAddr(b->device, coreName)) return p;
        if (PFN_vkVoidFunction p = vkGetInstanceProcAddr(b->instance, coreName)) return p;
        if (extName) {
            if (PFN_vkVoidFunction p = vkGetDeviceProcAddr(b->device, extName)) return p;
            if (PFN_vkVoidFunction p = vkGetInstanceProcAddr(b->instance, extName)) return p;
        }
        return nullptr;
    };
    b->cmdSetCullMode              = resolve_cmd("vkCmdSetCullMode", "vkCmdSetCullModeEXT");
    b->cmdSetFrontFace             = resolve_cmd("vkCmdSetFrontFace", "vkCmdSetFrontFaceEXT");
    b->cmdSetDepthTestEnable       = resolve_cmd("vkCmdSetDepthTestEnable", "vkCmdSetDepthTestEnableEXT");
    b->cmdSetDepthWriteEnable      = resolve_cmd("vkCmdSetDepthWriteEnable", "vkCmdSetDepthWriteEnableEXT");
    b->cmdSetDepthCompareOp        = resolve_cmd("vkCmdSetDepthCompareOp", "vkCmdSetDepthCompareOpEXT");
    b->cmdDrawIndirectCount        = resolve_cmd("vkCmdDrawIndirectCount", "vkCmdDrawIndirectCountKHR");
    b->cmdDrawIndexedIndirectCount = resolve_cmd("vkCmdDrawIndexedIndirectCount", "vkCmdDrawIndexedIndirectCountKHR");

    // A device that advertises the feature but exports no entrypoint for the
    // command would crash on the first call, so tie the flags to what actually
    // resolved. Every call site is already gated on these flags.
    if (!b->cmdDrawIndirectCount || !b->cmdDrawIndexedIndirectCount) {
        MITHRIL_LOG_INFO("vk", "drawIndirectCount unavailable: entrypoints not resolved");
        b->drawIndirectCountSupported = false;
    }
    if (!b->cmdSetCullMode || !b->cmdSetFrontFace || !b->cmdSetDepthTestEnable ||
        !b->cmdSetDepthWriteEnable || !b->cmdSetDepthCompareOp) {
        MITHRIL_LOG_INFO("vk", "extended dynamic state unavailable: entrypoints not resolved");
        b->extendedDynamicStateSupported = false;
    }
    // Dynamic rendering 的二次校验：扩展/版本声称支持但入口取不到时，
    // begin_render_pass 里的 `if (fn)` 会安静跳过 —— pass 被标记成 active
    // 却没有绑定任何附件 → 纯黑屏、无报错、无验证层警告，是最难查的一类。
    // 这里主动降级到传统 VkRenderPass/VkFramebuffer 路径。
    b->cmdBeginRendering = nullptr;
    b->cmdEndRendering = nullptr;
    if (b->dynamicRenderingSupported) {
        PFN_vkVoidFunction br = resolve_cmd("vkCmdBeginRendering", "vkCmdBeginRenderingKHR");
        PFN_vkVoidFunction er = resolve_cmd("vkCmdEndRendering", "vkCmdEndRenderingKHR");
        if (!br || !er) {
            MITHRIL_LOG_WARN("vk",
                "dynamic rendering 声称可用但入口未解析到 —— 改用传统 "
                "VkRenderPass/VkFramebuffer 路径");
            b->dynamicRenderingSupported = false;
        } else {
            b->cmdBeginRendering = br;
            b->cmdEndRendering = er;
        }
    }
    if (!b->dynamicRenderingSupported) {
        MITHRIL_LOG_INFO("vk",
            "render path: classic VkRenderPass/VkFramebuffer (no dynamic rendering)");
    }

    // ---- Command pool + primary command buffer ----
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = b->graphicsFamily;
    if (vkCreateCommandPool(b->device, &poolCI, nullptr, &b->commandPool) != VK_SUCCESS) {
        MITHRIL_LOG_ERROR("vk", "vkCreateCommandPool failed");
        return false;
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = b->commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = kMaxFramesInFlight;
    if (vkAllocateCommandBuffers(b->device, &allocInfo, b->commandBuffers) != VK_SUCCESS) {
        MITHRIL_LOG_ERROR("vk", "vkAllocateCommandBuffers failed");
        return false;
    }
    // Alias points to the current slot's buffer (slot 0 at init).
    b->commandBuffer = b->commandBuffers[b->currentFrame];

    // ---- Per-frame fences ----
    VkFenceCreateInfo fenceCI{};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // start signalled so first wait is a no-op
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        vkCreateFence(b->device, &fenceCI, nullptr, &b->frameFences[i]);
    }

    // Begin slot 0's command buffer in the recording state so that pre-frame
    // commands (texture uploads, layout transitions from glTexStorage2D) have
    // somewhere to record before the first begin_render_pass. The other slots
    // are lazily begun by ensure_command_buffer_recording() when their turn
    // comes. This mirrors MobileGL's FrameContext::Initialize
    // (FrameContext.cpp:29-31) which begins each slot's buffer at init.
    VkCommandBufferBeginInfo cbBegin{};
    cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(b->commandBuffer, &cbBegin) == VK_SUCCESS) {
        b->commandBufferRecording = true;
    } else {
        MITHRIL_LOG_ERROR("vk", "initial vkBeginCommandBuffer failed");
        return false;
    }

    // ---- Pipeline cache ----
    VkPipelineCacheCreateInfo cacheCI{};
    cacheCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkCreatePipelineCache(b->device, &cacheCI, nullptr, &b->pipelineCache);

    // ---- Dummy vertex buffer ----
    // 16-byte zero buffer for vertex attributes the shader declares but GL
    // has not enabled (see Pipeline.cpp's get_or_create_pipeline). Provides
    // valid backing for dummy attribute descriptions so SPIRV-Cross emits
    // [[attribute(N)]] for every stage_in field, avoiding the Metal
    // "invalid type ... stage_in" compile error.
    {
        static const uint8_t zeros[16] = {0};
        BufferEntry tmp{};
        if (create_buffer(tmp, 16, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, zeros)) {
            b->dummyVertexBuffer = tmp.buffer;
            b->dummyVertexMemory = tmp.memory;
        } else {
            MITHRIL_LOG_WARN("vk", "failed to allocate dummy vertex buffer");
        }
    }

    // ---- Per-frame transient staging arena (MobileGL pattern) ----
    // 每个 frame slot 一个大的 host-visible VkBuffer，用于纹理上传的 staging。
    // 在 ensure_command_buffer_recording 的 fence wait 后 rewind offset 到 0。
    // 这消除了 per-texture staging buffer 的分配/销毁循环，根治 Metal
    // Invalid Resource (code 9) UAF 崩溃。
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        VkBufferCreateInfo sbci{};
        sbci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sbci.size = Backend::kFrameStagingSize;
        sbci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sbci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(b->device, &sbci, nullptr, &b->frameStagingBuffer[i]) != VK_SUCCESS) {
            MITHRIL_LOG_ERROR("vk", "failed to create frame staging buffer "
                              "(slot %d, size=%zu)", i, (size_t)Backend::kFrameStagingSize);
            break;
        }
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(b->device, b->frameStagingBuffer[i], &req);
        uint32_t mt = find_memory_type(req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (mt == 0xFFFFFFFFu) {
            vkDestroyBuffer(b->device, b->frameStagingBuffer[i], nullptr);
            b->frameStagingBuffer[i] = VK_NULL_HANDLE;
            MITHRIL_LOG_ERROR("vk", "no host-visible memory type for frame staging "
                              "buffer (slot %d)", i);
            break;
        }
        VkMemoryAllocateInfo smai{};
        smai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        smai.allocationSize = req.size;
        smai.memoryTypeIndex = mt;
        // 使用 try_allocate_memory_with_gc（虽然首次分配不应 OOM，但保持一致性）
        if (try_allocate_memory_with_gc(b->device, &smai, nullptr,
                                         &b->frameStagingMemory[i]) != VK_SUCCESS) {
            vkDestroyBuffer(b->device, b->frameStagingBuffer[i], nullptr);
            b->frameStagingBuffer[i] = VK_NULL_HANDLE;
            MITHRIL_LOG_ERROR("vk", "failed to allocate memory for frame staging "
                              "buffer (slot %d)", i);
            break;
        }
        // FIX (MobileGL-style VRAM monitoring): the per-frame transient staging
        // arena (16MB/slot) is a real device-memory allocation — track it so
        // currentVramBytes reflects true pressure. Mirrors MobileGL tracking all
        // allocator-owned memory. Freed at shutdown (no disposalQueue path), so
        // no decrement is needed in drain_disposal_queue.
        b->currentAllocationCount++;
        b->currentVramBytes += req.size;
        vkBindBufferMemory(b->device, b->frameStagingBuffer[i],
                           b->frameStagingMemory[i], 0);
        // Persistently map — keep mapped for the lifetime of the buffer.
        // Host-coherent memory: writes are automatically visible to GPU.
        if (vkMapMemory(b->device, b->frameStagingMemory[i], 0,
                        Backend::kFrameStagingSize, 0,
                        &b->frameStagingMapped[i]) != VK_SUCCESS) {
            MITHRIL_LOG_WARN("vk", "failed to persistently map frame staging "
                             "buffer (slot %d) — falling back to per-upload map",
                             i);
            b->frameStagingMapped[i] = nullptr;
        }
        b->frameStagingOffset[i] = 0;
    }
    // Check if all slots were created successfully
    b->frameStagingReady = true;
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (b->frameStagingBuffer[i] == VK_NULL_HANDLE) {
            b->frameStagingReady = false;
            break;
        }
    }
    if (b->frameStagingReady) {
        MITHRIL_LOG_INFO("vk", "Per-frame transient staging arena initialised "
                          "(%d slots x %zu MB, persistently mapped)",
                          kMaxFramesInFlight,
                          (size_t)Backend::kFrameStagingSize / (1024 * 1024));
    } else {
        MITHRIL_LOG_WARN("vk", "Per-frame staging arena NOT ready — falling "
                          "back to per-texture staging buffers (higher "
                          "allocation count, potential Invalid Resource risk)");
    }

    b->initialized = true;
    MITHRIL_LOG_INFO("vk", "Vulkan 1.2 backend initialised (MoltenVK static link)");

    // FIX (GPU page fault root cause): eagerly create the process-wide default
    // 1x1 black texture NOW, while the device is fresh and VRAM is plentiful.
    // It is lazily initialized on first descriptor bind otherwise; under the
    // VRAM pressure spike at world-load (block/entity atlas creation) that lazy
    // init could fail, leaving the descriptor completeness guard with NO valid
    // fallback and an unwritten COMBINED_IMAGE_SAMPLER binding → GPU samples an
    // uninitialized descriptor → kIOGPUCommandBufferCallbackErrorPageFault.
    // A tiny 1x1 texture is guaranteed-available when created up front.
    if (mithril::vk::ensure_default_texture_ready()) {
        MITHRIL_LOG_INFO("vk", "Default fallback texture initialised eagerly");
    } else {
        MITHRIL_LOG_WARN("vk", "WARNING: failed to eagerly create default "
                          "fallback texture — descriptor completeness fallback "
                          "may be unavailable under VRAM pressure");
    }
    return true;
}

// ---- GL sync object backing (glFenceSync / glClientWaitSync) ----

// 单调时钟（纳秒）。用于在多个 fence 上串行等待时收敛剩余超时，
// 避免"每个 fence 各等 timeout"导致总等待时间成倍放大。
static uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// 重算已完成水位线。
//
// 水位线 = (所有仍在飞行的 slot 的序号中的最小值) - 1；若没有任何 slot 在飞行，
// 则等于已发出的最大序号 submitSerial。
//
// 这样才是"严格已完成"：只要还有一个更早的提交没回来，就不能把更晚的序号判为
// 完成。旧实现正是在这里出错（见 Device.h 的 FIX 注释 (a)）。
static void refresh_completed_serials() {
    Backend* b = backend();
    if (!b->initialized) return;

    // 先轮询每个 pending slot 的 fence（非阻塞）。
    for (int s = 0; s < kMaxFramesInFlight; ++s) {
        if (!b->serialPending[s]) continue;
        if (vkGetFenceStatus(b->device, b->frameFences[s]) == VK_SUCCESS) {
            b->serialPending[s] = false;
        }
    }

    // 再取仍在飞行的最小序号，算出水位线。
    uint64_t oldestInFlight = UINT64_MAX;
    for (int s = 0; s < kMaxFramesInFlight; ++s) {
        if (b->serialPending[s] && b->slotSerial[s] < oldestInFlight) {
            oldestInFlight = b->slotSerial[s];
        }
    }
    const uint64_t watermark = (oldestInFlight == UINT64_MAX)
                             ? b->submitSerial          // 全部回收完
                             : oldestInFlight - 1;      // 最早未完成的前一个
    if (watermark > b->completedSerial) b->completedSerial = watermark;
}

// 每次真实 vkQueueSubmit 后调用：领取一个新序号并挂到该 slot 上。
uint64_t backend_frame_serial_advance(int frameSlot) {
    Backend* b = backend();
    const uint64_t s = ++b->submitSerial;
    if (frameSlot >= 0 && frameSlot < kMaxFramesInFlight) {
        b->slotSerial[frameSlot]   = s;
        b->serialPending[frameSlot] = true;
    }
    return s;
}

uint64_t backend_last_completed_serial() {
    refresh_completed_serials();
    return backend()->completedSerial;
}

// 已发出的提交总数。glFenceSync 用它判断"我这个 fence 要等的是哪一次提交"。
uint64_t backend_current_submit_serial() {
    return backend()->submitSerial;
}

bool backend_wait_serial(uint64_t serial, uint64_t timeout_ns) {
    Backend* b = backend();
    if (!b->initialized) return true;
    refresh_completed_serials();
    if (serial <= b->completedSerial) return true;

    // 该序号尚未提交（fence 建在一次提交之前，而那次提交还没发生）。
    // 这里不能阻塞 —— 没有任何 fence 会为它亮起。返回未完成，由调用方
    // (glClientWaitSync) 依据 GL_SYNC_FLUSH_COMMANDS_BIT 决定是否先 flush。
    if (serial > b->submitSerial) return false;

    // 等待所有序号 <= serial 且仍在飞行的 slot。可能不止一个。
    // 逐个 vkWaitForFences，并按剩余时间收敛 timeout，避免总时长翻倍。
    uint64_t remaining = timeout_ns;
    for (int s = 0; s < kMaxFramesInFlight; ++s) {
        if (!b->serialPending[s] || b->slotSerial[s] > serial) continue;
        const uint64_t t0 = now_ns();
        VkResult r = vkWaitForFences(b->device, 1, &b->frameFences[s], VK_TRUE, remaining);
        if (r != VK_SUCCESS) return false;   // VK_TIMEOUT / device lost
        b->serialPending[s] = false;
        if (remaining != UINT64_MAX) {
            const uint64_t spent = now_ns() - t0;
            remaining = (spent >= remaining) ? 0 : (remaining - spent);
        }
    }
    refresh_completed_serials();
    return serial <= b->completedSerial;
}

void shutdown_device() {
    Backend* b = backend();
    if (!b->initialized) return;
    if (b->device) vkDeviceWaitIdle(b->device);
    // After vkDeviceWaitIdle, all GPU work is complete — safe to destroy any
    // deferred resources still sitting in the disposal queues.
    drain_all_disposal_queues();
    if (b->pipelineCache) { vkDestroyPipelineCache(b->device, b->pipelineCache, nullptr); b->pipelineCache = VK_NULL_HANDLE; }
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (b->frameFences[i]) { vkDestroyFence(b->device, b->frameFences[i], nullptr); b->frameFences[i] = VK_NULL_HANDLE; }
    }
    if (b->commandPool && b->commandBuffers[0]) {
        vkFreeCommandBuffers(b->device, b->commandPool, kMaxFramesInFlight, b->commandBuffers);
        for (int i = 0; i < kMaxFramesInFlight; ++i) b->commandBuffers[i] = VK_NULL_HANDLE;
        b->commandBuffer = VK_NULL_HANDLE;
    }
    if (b->commandPool) { vkDestroyCommandPool(b->device, b->commandPool, nullptr); b->commandPool = VK_NULL_HANDLE; }
    if (b->dummyVertexBuffer) { vkDestroyBuffer(b->device, b->dummyVertexBuffer, nullptr); b->dummyVertexBuffer = VK_NULL_HANDLE; }
    if (b->dummyVertexMemory) { vkFreeMemory(b->device, b->dummyVertexMemory, nullptr); b->dummyVertexMemory = VK_NULL_HANDLE; }
    // Destroy per-frame transient staging arena
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (b->frameStagingMapped[i]) { vkUnmapMemory(b->device, b->frameStagingMemory[i]); b->frameStagingMapped[i] = nullptr; }
        if (b->frameStagingBuffer[i]) { vkDestroyBuffer(b->device, b->frameStagingBuffer[i], nullptr); b->frameStagingBuffer[i] = VK_NULL_HANDLE; }
        if (b->frameStagingMemory[i]) { vkFreeMemory(b->device, b->frameStagingMemory[i], nullptr); b->frameStagingMemory[i] = VK_NULL_HANDLE; }
        b->frameStagingOffset[i] = 0;
    }
    b->frameStagingReady = false;
    // Destroy the per-frame transient UNIFORM arena. Must run before
    // vkDestroyDevice (it unmaps/frees VkDeviceMemory and destroys VkBuffers
    // through b->device) and is safe here because of the vkDeviceWaitIdle at
    // the top of this function.
    ubo_arena_shutdown();
    // 传统路径缓存的 VkRenderPass / VkFramebuffer 必须在 vkDestroyDevice 之前销毁。
    mithril_vk_destroy_render_pass_cache();
    if (b->device) { vkDestroyDevice(b->device, nullptr); b->device = VK_NULL_HANDLE; }
    if (b->instance) { vkDestroyInstance(b->instance, nullptr); b->instance = VK_NULL_HANDLE; }
    b->initialized = false;
}

} // namespace vk
} // namespace mithril

// ===========================================================================
// Public C API lifecycle functions (declared in MG_Backend/Backend.h)
// ===========================================================================
extern "C" {

void backend_init(void) {
    mithril::vk::init_device();
}

void backend_shutdown(void) {
    mithril::vk::shutdown_device();
}

int backend_available(void) {
    return mithril::vk::backend()->initialized ? 1 : 0;
}

const char* backend_physical_device_name(void) {
    mithril::vk::Backend* b = mithril::vk::backend();
    return b->initialized ? b->props.deviceName : "Vulkan (MoltenVK)";
}

uint64_t backend_vram_bytes(void) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->initialized) return 0;
    // MoltenVK reports maxMemoryAllocationCount but not total VRAM reliably;
    // approximate using the heap sizes from memory properties.
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(b->physicalDevice, &mp);
    uint64_t total = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total += mp.memoryHeaps[i].size;
        }
    }
    return total;
}

// FIX (P1): 用真实的 VkPhysicalDeviceLimits 回答 GL_MAX_*，不再硬编码。
// 详见 Backend.h 中 MITHRIL_LIMIT_* 的说明。
int backend_device_limit(int which, int fallback) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b || !b->initialized) return fallback;
    const VkPhysicalDeviceLimits& L = b->props.limits;

    // Vulkan 的上限是 uint32_t，GL 侧是 GLint。某些实现会把「无限制」报成
    // 0xFFFFFFFF，直接转 int 会变成 -1，Sodium 拿到负的 GL_MAX_* 会算出
    // 负的数组大小。统一夹到 INT32_MAX。
    auto clamp_i = [](uint32_t v) -> int {
        return v > (uint32_t)0x7FFFFFFF ? 0x7FFFFFFF : (int)v;
    };

    switch (which) {
        case MITHRIL_LIMIT_MAX_TEXTURE_SIZE:          return clamp_i(L.maxImageDimension2D);
        case MITHRIL_LIMIT_MAX_3D_TEXTURE_SIZE:       return clamp_i(L.maxImageDimension3D);
        case MITHRIL_LIMIT_MAX_CUBE_MAP_TEXTURE_SIZE: return clamp_i(L.maxImageDimensionCube);
        case MITHRIL_LIMIT_MAX_ARRAY_TEXTURE_LAYERS:  return clamp_i(L.maxImageArrayLayers);
        case MITHRIL_LIMIT_MAX_RENDERBUFFER_SIZE:     return clamp_i(L.maxFramebufferWidth < L.maxFramebufferHeight
                                                                     ? L.maxFramebufferWidth : L.maxFramebufferHeight);
        case MITHRIL_LIMIT_MAX_VIEWPORT_WIDTH:        return clamp_i(L.maxViewportDimensions[0]);
        case MITHRIL_LIMIT_MAX_VIEWPORT_HEIGHT:       return clamp_i(L.maxViewportDimensions[1]);

        // 每级着色器可见的采样器数量。必须同时受内部 kMaxTextureUnits 数组
        // 容量约束 —— 报得比数组大，超出的绑定会被静默丢弃。
        case MITHRIL_LIMIT_MAX_TEXTURE_IMAGE_UNITS: {
            int v = clamp_i(L.maxPerStageDescriptorSampledImages);
            return v < mithril::kMaxTextureUnits ? v : mithril::kMaxTextureUnits;
        }
        case MITHRIL_LIMIT_MAX_COMBINED_TEX_UNITS: {
            int v = clamp_i(L.maxDescriptorSetSampledImages);
            return v < mithril::kMaxTextureUnits ? v : mithril::kMaxTextureUnits;
        }

        case MITHRIL_LIMIT_MAX_UNIFORM_BLOCK_SIZE:    return clamp_i(L.maxUniformBufferRange);
        case MITHRIL_LIMIT_UNIFORM_BUFFER_ALIGNMENT:
            return clamp_i((uint32_t)L.minUniformBufferOffsetAlignment);
        case MITHRIL_LIMIT_MAX_UNIFORM_BUFFER_BINDINGS: {
            int v = clamp_i(L.maxDescriptorSetUniformBuffers);
            return v < mithril::kMaxIndexedBindings ? v : mithril::kMaxIndexedBindings;
        }
        case MITHRIL_LIMIT_MAX_COLOR_ATTACHMENTS: {
            int v = clamp_i(L.maxColorAttachments);
            return v < mithril::kMaxColorAttachments ? v : mithril::kMaxColorAttachments;
        }

        // GL_MAX_SAMPLES：取 color+depth 都支持的最高 sample count。
        // 只看 framebufferColorSampleCounts 会在 depth 不支持 4x 时上报过高。
        case MITHRIL_LIMIT_MAX_SAMPLES: {
            VkSampleCountFlags f = L.framebufferColorSampleCounts
                                 & L.framebufferDepthSampleCounts;
            if (f & VK_SAMPLE_COUNT_16_BIT) return 16;
            if (f & VK_SAMPLE_COUNT_8_BIT)  return 8;
            if (f & VK_SAMPLE_COUNT_4_BIT)  return 4;
            if (f & VK_SAMPLE_COUNT_2_BIT)  return 2;
            return 1;
        }

        case MITHRIL_LIMIT_MAX_VERTEX_ATTRIBS: {
            int v = clamp_i(L.maxVertexInputAttributes);
            return v < mithril::kMaxVertexAttribs ? v : mithril::kMaxVertexAttribs;
        }
        case MITHRIL_LIMIT_MAX_SSBO_BINDINGS: {
            int v = clamp_i(L.maxDescriptorSetStorageBuffers);
            return v < mithril::kMaxIndexedBindings ? v : mithril::kMaxIndexedBindings;
        }
        case MITHRIL_LIMIT_MAX_SSBO_SIZE:             return clamp_i(L.maxStorageBufferRange);
        case MITHRIL_LIMIT_MAX_COMPUTE_WG_INVOCATIONS:
            return clamp_i(L.maxComputeWorkGroupInvocations);
        case MITHRIL_LIMIT_MAX_COMPUTE_WG_COUNT_X:    return clamp_i(L.maxComputeWorkGroupCount[0]);
        case MITHRIL_LIMIT_MAX_COMPUTE_WG_SIZE_X:     return clamp_i(L.maxComputeWorkGroupSize[0]);
        default:                                      return fallback;
    }
}

float backend_device_max_sampler_anisotropy(float fallback) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b || !b->initialized) return fallback;

    // VkPhysicalDeviceLimits is the authoritative device limit. A value below
    // one is not a legal GL anisotropy limit; retain the no-anisotropy floor
    // until Vulkan supplies a usable value.
    const float value = b->props.limits.maxSamplerAnisotropy;
    return value >= 1.0f ? value : fallback;
}

} // extern "C"
