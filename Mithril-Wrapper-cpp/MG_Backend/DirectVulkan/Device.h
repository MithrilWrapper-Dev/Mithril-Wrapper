// Mithril-Wrapper - MG_Backend/DirectVulkan/Device.h
// Global Vulkan 1.2 device state (VkInstance / VkPhysicalDevice / VkDevice /
// VkQueue / VkCommandPool / VkCommandBuffer). Accessed by the other
// DirectVulkan translation units via vk_backend().
//
// iOS surface creation uses VK_EXT_metal_surface (vkCreateMetalSurfaceEXT) —
// NOT the deprecated VK_MVK_metal_surface. Portability is required:
//   * VK_KHR_portability_enumeration instance extension +
//     VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
//   * VK_KHR_portability_subset device extension (always enabled alongside
//     VK_KHR_swapchain)
// Apple builds link MoltenVK directly as a dylib (host-bundled on iOS,
// Homebrew on macOS); there is no separate Vulkan loader / ICD file.
#ifndef MITHRIL_DIRECTVULKAN_DEVICE_H
#define MITHRIL_DIRECTVULKAN_DEVICE_H

#include <vulkan/vulkan.h>
#include <vector>

namespace mithril {
namespace vk {

// Minimum/maximum swapchain images in-flight (default 2; allows up to 3).
constexpr int kMaxFramesInFlight = 2;

// Deferred Vulkan resource destruction entry. When a GL buffer/texture/sampler
// is deleted or orphaned (glBufferData rename), its underlying VkBuffer/VkImage/
// VkDeviceMemory/VkImageView/VkSampler handles are pushed into a per-frame-slot
// disposal queue instead of being destroyed immediately. The queue for slot S
// is drained when ensure_command_buffer_recording() waits on fence[S] — at that
// point all command buffers submitted to slot S have completed on the GPU, so
// any Metal resources referenced by those command buffers are no longer in use.
//
// This fixes the Metal resource Use-After-Free crash (objc_retain of a zombie
// IOGPUMetalResource in MVKGraphicsResourcesCommandEncoderState::encodeBindings
// during vkEndCommandBuffer → MVKCmdDrawIndexed::encode). The crash occurred
// because glDeleteBuffers / glBufferData orphan destroyed VkBuffers immediately
// while MoltenVK's command encoding still held references to the corresponding
// MTLBuffer wrappers. With deferred destruction, the VkBuffer (and its MTLBuffer)
// survives until the GPU finishes all in-flight command buffers that reference it.
struct DeferredDestroy {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkImage        image = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkSampler      sampler = VK_NULL_HANDLE;
    // FIX (descriptor pool UAF - P0): a VkDescriptorPool whose sets may still
    // be referenced by an in-flight command buffer. When a per-program pool is
    // exhausted MID-RECORDING (see DescriptorSet.cpp), we now grow a secondary
    // pool instead of vkResetDescriptorPool — resetting frees sets that draws
    // already recorded in the current command buffer still reference, and when
    // the GPU executes those draws it reads freed descriptor memory → GPU page
    // fault (kIOGPUCommandBufferCallbackErrorPageFault). The retired pool is
    // deferred here and destroyed only after the slot's fence signals.
    VkDescriptorPool pool = VK_NULL_HANDLE;
    // FIX (MobileGL-style VRAM monitoring): byte size of the memory allocation
    // being freed, so drain_disposal_queue can decrement currentVramBytes.
    // Only set for entries that own a VkDeviceMemory (buffer/texture main alloc
    // + staging). Samplers/views have no device memory, so size stays 0.
    VkDeviceSize   memorySize = 0;
};

// Global Vulkan backend state. A single instance lives for the process; it is
// created by backend_init() and torn down by backend_shutdown().
struct Backend {
    bool initialized = false;

    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device = VK_NULL_HANDLE;
    VkQueue          graphicsQueue = VK_NULL_HANDLE;
    uint32_t         graphicsFamily = 0xFFFFFFFFu;

    VkCommandPool    commandPool = VK_NULL_HANDLE;
    // Per-frame-slot primary command buffers (mirrors MobileGL's
    // FrameContext::FrameData::commandBuffer). Each slot has its own buffer so
    // that submitting slot N's buffer and then recording into slot N+1's
    // buffer can happen concurrently without resetting a pending buffer
    // (which is Vulkan spec UB). With a single shared buffer, the reset+begin
    // at the end of commit_frame would reset a buffer the GPU is still
    // executing — the root cause of the black-screen-with-sound issue.
    VkCommandBuffer  commandBuffers[kMaxFramesInFlight] = {};
    // Alias pointing to commandBuffers[currentFrame]. Updated by
    // ensure_command_buffer_recording() whenever the current slot changes or
    // the buffer is reset+begin. All existing code that references
    // b->commandBuffer continues to work unchanged.
    VkCommandBuffer  commandBuffer = VK_NULL_HANDLE;

    // Tracks whether commandBuffer (= commandBuffers[currentFrame]) is in the
    // RECORDING state. Set false after vkEndCommandBuffer (buffer is now
    // executable/pending). Set true by ensure_command_buffer_recording().
    // Code that records into b->commandBuffer outside of begin_render_pass
    // (e.g. stage_and_copy_image for texture uploads) MUST call
    // ensure_command_buffer_recording() first.
    bool             commandBufferRecording = false;

    VkPipelineCache  pipelineCache = VK_NULL_HANDLE;

    // Physical-device properties (cached on init for the GPU name string).
    VkPhysicalDeviceProperties props{};

    // FIX (MobileGL-style VRAM monitoring): cached memory properties for
    // find_memory_type and VRAM heap size queries. Avoids repeated
    // vkGetPhysicalDeviceMemoryProperties calls.
    VkPhysicalDeviceMemoryProperties memProps{};

    // vkCreateMetalSurfaceEXT function pointer (resolved from the instance).
    // Stored as PFN_vkVoidFunction (always declared by vulkan_core.h) so this
    // header does NOT need VK_USE_PLATFORM_METAL_EXT — that macro pulls in
    // <Metal/Metal.h> (Objective-C only) and would break every .cpp that
    // includes this header. SwapchainMetal.mm casts to PFN_vkCreateMetalSurfaceEXT
    // at the call site, where the metal platform define is active.
    PFN_vkVoidFunction createMetalSurfaceEXT = nullptr;

    // ---- Vulkan 1.3 core commands, resolved at runtime ----
    // vkCmdSetCullMode / SetFrontFace / SetDepthTestEnable / SetDepthWriteEnable
    // / SetDepthCompareOp (extended dynamic state) and vkCmdDraw*IndirectCount
    // are 1.3 core. MoltenVK exports them so linking works on Apple, but
    // Android's libvulkan.so does not, which failed the link with
    // "undefined symbol". Resolving them with vkGetDeviceProcAddr keeps the
    // library loadable on both, and the matching *Supported flags below already
    // gate every call site, so a device that lacks them simply never calls in.
    // vkCmdBeginRendering / vkCmdEndRendering（VK_KHR_dynamic_rendering，
    // 1.3 起为核心）。存在 Backend 上而不是调用方的 static 局部量：
    // device-lost 重建后 static 会残留旧设备的函数指针 → 直接崩。
    PFN_vkVoidFunction cmdBeginRendering = nullptr;
    PFN_vkVoidFunction cmdEndRendering = nullptr;
    PFN_vkVoidFunction cmdSetCullMode = nullptr;
    PFN_vkVoidFunction cmdSetFrontFace = nullptr;
    PFN_vkVoidFunction cmdSetDepthTestEnable = nullptr;
    PFN_vkVoidFunction cmdSetDepthWriteEnable = nullptr;
    PFN_vkVoidFunction cmdSetDepthCompareOp = nullptr;
    PFN_vkVoidFunction cmdDrawIndirectCount = nullptr;
    PFN_vkVoidFunction cmdDrawIndexedIndirectCount = nullptr;


    // Per-frame sync: a fence per in-flight frame so we can wait on the GPU
    // before reusing the command buffer.
    VkFence frameFences[kMaxFramesInFlight] = {};
    int     currentFrame = 0;

    // Tracks whether a vkQueueSubmit has actually signaled each frame slot's
    // fence since the last wait on that slot. commit_frame() sets the flag
    // for the current slot after a successful submit; the NEXT commit_frame()
    // on the same slot waits on that fence (deferred async-pipeline wait,
    // mirroring MobileGL's FrameContext::WaitAndAcquireNextImage) and clears
    // the flag. Without this gate, commit_frame would wait on an
    // already-signaled or never-signaled fence. The fences are created with
    // VK_FENCE_CREATE_SIGNALED_BIT (Device.cpp), so the first frame's wait is
    // skipped — the flag starts false and is only set after a real submit.
    bool    fencePending[kMaxFramesInFlight] = {};

    // Monotonic frame generation counter, bumped once per commit_frame(). Used
    // by DescriptorSet.cpp to reset each program's descriptor pool exactly once
    // per frame (currentFrame cycles 0/1, so a program drawn only on every other
    // frame would never see a reset — the monotonic counter fixes that). The
    // value seen by every draw within a single frame is constant.
    uint64_t frameGeneration = 0;

    // FIX (mid-frame flush 缓存失效 - P0): bumped by safe_device_wait_idle()
    // every time it submits the current command buffer mid-frame and re-begins
    // it. ensure_command_buffer_recording() then rewinds the UBO arena / staging
    // arena, which invalidates:
    //   * per-program UBO plan caches (plan.lastOffset points at arena bytes
    //     that later uploads may overwrite after the rewind),
    //   * per-(program,slot) descriptor-set memos (sets written before the
    //     flush still name pre-flush arena offsets),
    //   * the descriptor bind shadow.
    // DescriptorSet.cpp compares each cache against this counter and drops it on
    // mismatch, so a post-flush draw never binds a descriptor whose dynamic
    // offset targets rewound (overwritten) arena memory.
    uint64_t flushGeneration = 0;

    // FIX (GPU page fault root cause — re-entrant mid-frame purge):
    // reset_all_descriptor_pools() / clear_all_pipeline_caches() are only safe
    // when the current command buffer has NO live descriptor-set / pipeline
    // binds (i.e. after a safe_device_wait_idle that flushed and re-begun an
    // empty buffer). When an OOM/critical-pressure GC path requests a purge
    // while a buffer is actively recording with binds, we must NOT run it
    // re-entrantly (it would vkResetDescriptorPool the current slot's pool,
    // invalidating the very sets the recording buffer already references ->
    // kIOGPUCommandBufferCallbackErrorPageFault at the next submit). Instead we
    // set this flag and defer the purge to the next safe point
    // (process_pending_purge, called after the command buffer is flushed/re-begun).
    bool pendingPurge = false;

    // FIX (GL sync correctness — glFenceSync / glClientWaitSync):
    // glFenceSync must not report "already signaled" immediately, otherwise
    // Sodium's persistent mapped-upload ring buffer overwrites a region the GPU
    // is still reading (corrupt geometry / flicker). We stamp every GPU
    // submission with a monotonic `frameSerial` and let a GL sync object record
    // the serial at fence-creation time. glClientWaitSync then checks whether
    // that serial's submission has actually completed (via the per-slot fence)
    // before reporting success.
    // FIX (serial 记账重写) —— 旧实现有四个会让 Sodium 撕裂或死锁的错误，
    // 对照 MobileGL 上游 VulkanRenderer 的 GetSyncPointSubmitIndex() 思路重做：
    //
    //  (a) 旧的 refresh 里写 `lastCompletedSerial = frameSerial`。某个 slot 的
    //      fence 完成时，它把"当前最大序号"整体判为已完成 —— 而那个最大序号
    //      往往属于另一个仍在飞行的 slot。Sodium 据此认为 GPU 已读完，提前回收
    //      持久映射 ring buffer 的区域并覆写 → 几何撕裂/闪烁。
    //      改法：每个 slot 记住自己那次提交的序号(slotSerial)，fence 亮了只推进
    //      到"所有仍 pending 的 slot 序号的最小值 - 1"，即严格已完成的水位线。
    //
    //  (b) 旧的 serialToFrame[serial & 63] 只有 64 槽，第 65 次提交就把旧映射
    //      覆盖掉，之后查到的 slot 是错的 → 等错 fence。
    //      改法：水位线模型不需要 serial→slot 反查表，直接扫描 pending slot。
    //
    //  (c) 一帧只 advance 一次，同帧内多个 glFenceSync 拿到同一序号，粒度退化。
    //      改法：serial 由每次真实 vkQueueSubmit 递增（见 commit_frame）。
    //
    //  (d) 详见 Drawing.cpp：glClientWaitSync 忽略 GL_SYNC_FLUSH_COMMANDS_BIT，
    //      未提交的工作永远等不到 → 死锁。
    //
    // 语义：serial N 表示"第 N 次 vkQueueSubmit"。completedSerial 是水位线，
    // 保证 <= 它的所有提交都已在 GPU 上执行完毕。
    uint64_t submitSerial = 0;                       // 已发出的提交总数（单调）
    uint64_t slotSerial[kMaxFramesInFlight] = {};    // 各 slot 最后一次提交的序号
    bool     serialPending[kMaxFramesInFlight] = {}; // 该 slot 是否仍在飞行
    uint64_t completedSerial = 0;                    // 已确定完成的水位线

    // 默认 16 字节 zero buffer，用于着色器声明但 GL 未 enable 的 vertex attribute
    // binding。Pipeline.cpp 的 get_or_create_pipeline 为这些 location 提供 dummy
    // attribute description 指向此 buffer，让 SPIRV-Cross 为每个 stage_in 字段生成
    // [[attribute(N)]] 限定符，避免 Metal 编译报错。
    VkBuffer         dummyVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory   dummyVertexMemory = VK_NULL_HANDLE;

    /* VK_EXT_primitive_topology_list_restart (root cause AS).
     *
     * Primitive restart on a LIST topology (points/lines/triangles) is only
     * legal when this feature is enabled —
     * VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252. Strip and
     * fan topologies never need it.
     *
     * Without the check, a program that leaves GL_PRIMITIVE_RESTART enabled
     * and then draws GL_TRIANGLES builds an illegal pipeline: creation fails,
     * get_or_create_pipeline returns VK_NULL_HANDLE, and every such draw is
     * dropped. MoltenVK does not currently expose this extension, so the
     * flag is normally false and the restart bit has to be suppressed for
     * list topologies. */
    bool             listRestartSupported = false;

    /* multiDrawIndirect core feature — lets one vkCmdDrawIndirect issue
     * drawCount > 1. Without it the indirect path loops one draw at a time,
     * which is correct but slower. */
    bool             multiDrawIndirectSupported = false;

    /* drawIndirectCount core feature (Vulkan 1.2) — enables
     * vkCmdDrawIndirectCount / vkCmdDrawIndexedIndirectCount, the backend of
     * GL 4.6 ARB_indirect_parameters (glMultiDraw*IndirectCount). Sodium's
     * chunk render issues these; when unsupported we must fall back rather
     * than record a vkCmd*Count call the device cannot execute. */
    /* Vulkan level actually granted to this instance. Requesting 1.2 on a
     * device that only implements 1.1 is fatal - vkCreateInstance returns
     * VK_ERROR_INCOMPATIBLE_DRIVER - so init_device() steps down until one
     * level is accepted and records it here. 1.2 feature structs and 1.3
     * commands are gated on this, so a 1.1 device gets a working instance
     * instead of none.
     */
    uint32_t         instanceApiVersion = VK_API_VERSION_1_1;

    bool             drawIndirectCountSupported = false;

    /* sampleRateShading core feature — required before a pipeline may set
     * sampleShadingEnable (GL 4.0 ARB_sample_shading). */
    bool             sampleRateShadingSupported = false;

    /* VK_KHR_dynamic_rendering — 决定录制期走 vkCmdBeginRendering 还是传统
     * VkRenderPass/VkFramebuffer。MoltenVK 1.2.x 之前不暴露该扩展，iOS 上
     * 常见缺席，因此必须运行时判定，不能假定为真。 */
    bool             dynamicRenderingSupported = false;

    /* VK_EXT_extended_dynamic_state — 决定能否调用 vkCmdSetCullMode /
     * SetFrontFace / SetDepthTestEnable 那一族。缺席时这些状态只能烘进
     * 静态管线，管线 key 必须把它们算进去。 */
    bool             extendedDynamicStateSupported = false;

    // GPU 故障状态。只有 VK_ERROR_DEVICE_LOST 才设置 deviceLost；
    // OOM 和其他错误不再设置 deviceLost（改为触发 OOM GC 后跳过当前帧）。
    // deviceLost 为真时 commit_frame / present / acquire / eglSwapBuffers 全部
    // 跳过实际 GPU 操作，由 EGL 恢复路径（10 帧间隔重建 swapchain）清除。
    bool             deviceLost = false;
    int              consecutiveSubmitFailures = 0;  // 统计用，不再触发 deviceLost

    // FIX (P1): 内存分配计数器。MoltenVK 的 maxMemoryAllocationCount 默认很小
    // （通常 128-4096），每张纹理/buffer 独立 vkAllocateMemory 会耗尽配额。
    // 跟踪当前已分配数量，接近上限时触发警告和主动释放。
    uint32_t         maxMemoryAllocationCount = 0;  // 从 limits 读取（仅诊断）
    uint32_t         currentAllocationCount = 0;    // 运行时计数（仅诊断）

    // FIX (MobileGL-style VRAM monitoring): MobileGL 不依赖 maxMemoryAllocationCount
    // 做压力判断，而是用 OS 原生内存 API（iOS: os_proc_available_memory）获取真实
    // 可用内存 + 跟踪已分配字节数，按字节阈值触发 GC。
    // MoltenVK 的 maxMemoryAllocationCount 报告 10亿+ 不可信，allocation count
    // 不能反映真实压力（一个 16MB 纹理和一个 64B UBO 各算 1 次）。改为字节级监控。
    //
    // 参考 MobileGL VulkanRenderer::GetMemoryUsage / TryDrainFrameTransients：
    // 它用 OS 原生 API 获取真实内存上限，跟踪已分配字节数，在高位时主动 drain +
    // rewind transient arena + DestroyAll pipeline cache。
    //
    // 预算来源：os_proc_available_memory() * 25%，硬上限 512MB，下限 96MB。
    // GC 阈值 50% — 在驱动 GPU Address Fault 之前触发。
    // 临界阈值 65% — 额外驱逐 pipeline cache + descriptor pools。
    // 注意：256MB 上限曾是"GPU Address Fault 之前安全"的保守估计，但那些
    // fault 的真正根因是资源生命周期错误（drawable 池不匹配 / descriptor UAF /
    // buffer 覆写竞争），已在 P0 修复中消除。iPhone X 有 ~2GB 统一内存可给
    // GPU，512MB 上限在修复后是安全的，且避免 256MB 下每帧触发 mid-frame
    // GC flush（safe_device_wait_idle 提交+重开 command buffer，本身有性能
    // 与一致性的代价）。
    VkDeviceSize     totalVramBytes = 0;       // GPU 显存预算（≤512MB）
    VkDeviceSize     currentVramBytes = 0;     // 运行时已分配字节数
    VkDeviceSize     vramPressureThreshold = 0; // GC 触发阈值（50% of totalVramBytes）

    // Deferred destruction queue: one bucket per frame slot. When a GL
    // buffer/texture/sampler is deleted or orphaned, its Vulkan handles are
    // pushed into disposalQueue[currentFrame] instead of being destroyed
    // immediately. The bucket is drained in ensure_command_buffer_recording()
    // after waiting on that slot's fence — at that point the GPU has finished
    // all work submitted to that slot, so the deferred resources are safe to
    // destroy. See DeferredDestroy above for the full rationale.
    std::vector<DeferredDestroy> disposalQueue[kMaxFramesInFlight];

    // FIX (Invalid Resource 根因 - per-frame transient staging arena):
    // 深度参考 MobileGL 的 transient staging arena 模式：每个 frame slot 持有
    // 一个大的 host-visible VkBuffer，纹理上传时从中 sub-allocate（bump offset）。
    // 在 ensure_command_buffer_recording 的 fence wait 之后 rewind offset 到 0。
    //
    // 这消除了 per-texture staging buffer 的创建/销毁循环：
    //   - 不再为每次 stage_and_copy_image 调用 vkCreateBuffer + vkAllocateMemory
    //   - 不再将 staging buffer 推入 disposalQueue
    //   - 不再因 disposal queue 生命周期管理不当导致 Metal Invalid Resource (code 9)
    //
    // arena 在 init_device 中创建（一次），在 shutdown_device 中销毁。
    // persistently mapped（vkMapMemory 一次，保持映射）。
    // overflow（单次上传超过剩余空间）回退到临时 staging buffer + deferred destroy。
    // FIX (VRAM 预算降低): 8MB/slot (was 16MB)。2 slots × 16MB = 32MB 占
    // 256MB 预算的 12.5%，降到 2 × 8MB = 16MB (6.25%)。Minecraft 单帧纹理
    // 上传极少超过 8MB（单张 2048×2048 RGBA = 16MB 才会 overflow，走临时
    // staging 路径）。overflow 路径已正确处理（defer_destroy + GC）。
    static constexpr VkDeviceSize kFrameStagingSize = 8 * 1024 * 1024;  // 8 MB per slot
    VkBuffer       frameStagingBuffer[kMaxFramesInFlight] = {};
    VkDeviceMemory frameStagingMemory[kMaxFramesInFlight] = {};
    VkDeviceSize   frameStagingOffset[kMaxFramesInFlight] = {};
    void*          frameStagingMapped[kMaxFramesInFlight] = {};
    bool           frameStagingReady = false;
};

// Access the singleton backend state. Allocated on first call.
Backend* backend();

// 返回 backend 是否进入持久性故障状态。一旦置位，渲染线程的 submit/present/
// acquire/swapchain-rebuild 全部跳过，避免死循环刷屏。
bool backend_is_device_lost();

// 重置 deviceLost 状态为 false，并清零 consecutiveSubmitFailures 计数器。
// 由 EGL 的 deviceLost 恢复路径调用：当 swapchain 重建成功后，给设备一次
// 恢复渲染的机会。GPU 超时通常是暂时的（资源压力释放后可恢复），不应永久
// 终止渲染（原实现一旦置位就永不恢复，导致进程永久卡死）。
void backend_reset_device_lost();

// FIX (黑屏根因): 在 deviceLost 期间（恢复尝试前）排空所有 disposalQueue，
// 释放被延迟销毁的 VkBuffer/VkImage/VkDeviceMemory。deviceLost 期间
// ensure_command_buffer_recording 早退，disposalQueue 不会被正常排空，
// 导致显存持续占用，swapchain 重建也因显存不足而失败。
// 此函数不重置 deviceLost 标志（仅释放资源），让 swapchain 重建有显存可用。
// 参考 MobileGL TryDrainFrameTransients（VulkanRenderer.cpp:7239-7313）。
void backend_reset_device_lost_pending_resources();

// FIX (swapchain rebuild death loop): Purge ALL recreatable cached Vulkan
// resources (VkPipeline objects, VkDescriptorSet objects, disposal queues)
// BEFORE a swapchain rebuild attempt. The old path only cleared caches AFTER
// rebuild success — but rebuild failed because the caches still held memory.
// This breaks the chicken-and-egg: free memory first, then rebuild succeeds.
// Reference: MobileGL RecreateSwapchain calls pipelineFactory->DestroyAll()
// + uniformManager->ResetAllPools() BEFORE creating the new swapchain.
void backend_purge_cached_resources_for_recovery();

// One-time init of the instance/device/queue/command pool/pipeline cache.
// Idempotent; sets Backend::initialized on success.
bool init_device();

// Tear down everything created by init_device() (instance-level resources).
// Resource/pipeline/swapchain objects are owned by their respective modules.
void shutdown_device();

// Drain the disposal queue for `slot`: actually call vkDestroy* / vkFreeMemory
// for every deferred entry. MUST only be called after the GPU has finished all
// work submitted to that slot (i.e. after vkWaitForFences on frameFences[slot]
// or vkDeviceWaitIdle). Called by ensure_command_buffer_recording() after the
// per-slot fence wait, and by drain_all_disposal_queues() after vkDeviceWaitIdle.
void drain_disposal_queue(int slot);

// Drain ALL disposal queue buckets. Called after vkDeviceWaitIdle (which
// guarantees all GPU work is complete) and during shutdown_device().
void drain_all_disposal_queues();

// FIX (mid-frame UAF): drain every slot's disposal queue EXCEPT `skip`. Used
// by the proactive-GC path, which runs mid-frame while the current slot's
// command buffer is still being recorded. The current slot's deferred resources
// may still be referenced by this frame's live descriptor sets, so they are
// freed later by the normal fence-wait drain after the frame commits.
void drain_disposal_queues_except(int skip);

// FIX (SIGBUS 根因 - vkDeviceWaitIdle 触发 deferred encoding):
// MoltenVK 的 vkDeviceWaitIdle 会等待所有 pending command buffer 被编码完成。
// 如果当前有正在录制的 command buffer（commandBufferRecording=true），其中
// 包含 vkCmdCopyBufferToImage 等命令，vkDeviceWaitIdle 会触发 MoltenVK 对
// 未完成 command buffer 的 deferred encoding → MVKCmdBufferImageCopy::encode
// 访问命令池中未对齐的内存 → SIGBUS (BUS_ADRALN)。
//
// safe_device_wait_idle() 在调用 vkDeviceWaitIdle 前，先检查并安全结束当前
// 录制的 command buffer（vkEndCommandBuffer + 提交到对应 slot 的 fence），
// 确保没有未完成的 command buffer 时才调用 vkDeviceWaitIdle。
//
// 所有需要在 GC/drain 路径中调用 vkDeviceWaitIdle 的地方都应使用此函数。
void safe_device_wait_idle();

// ---- GL sync object backing (glFenceSync / glClientWaitSync) ----
// Advance the per-submission serial. Called once per vkQueueSubmit in
// commit_frame(). Records the current frame slot so a sync object stamped with
// the returned serial can be completed exactly when that slot's fence signals.
uint64_t backend_frame_serial_advance(int frameSlot);
// Latest queue-submit serial. Exposed for GL/EGL fence synchronization.
extern "C" uint64_t backend_current_submit_serial();
// Returns the highest serial whose GPU submission has definitely completed.
extern "C" uint64_t backend_last_completed_serial();
// Block (or, with timeout==0, poll) until the submission bearing `serial` has
// completed. Returns true if completed, false if still pending (timeout==0) or
// the wait failed.
extern "C" bool backend_wait_serial(uint64_t serial, uint64_t timeout_ns);

// FIX (显存耗尽根因 - 主动式 GC，深度参考 MobileGL):
//
// MobileGL 在每帧 Present 末尾、FlushPendingCommands、WaitForFrameSerial 等
// 多个点调用 TryDrainFrameTransients（VulkanRenderer.cpp:7239-7313），其中：
//   1. RefreshCompletedSubmits() 用 vkGetFenceStatus 非阻塞轮询所有 in-flight
//      submit 的 fence，回收已完成的 pooled fence
//   2. CollectAllDeferredReleases() 释放所有已完成帧的延迟资源
//   3. 每 8 次 drain rewind transient arena + age cache
//
// 我们的 disposalQueue 是 per-frame-slot 的，只在 ensure_command_buffer_recording
// 复用某个 slot 时才 drain 该 slot。问题：kMaxFramesInFlight=2 时，slot 0 的
// GPU 工作可能在 slot 1 录制期间就完成了，但 slot 0 的 disposalQueue 要等到
// 下次循环回 slot 0 才 drain。这导致显存占用峰值偏高（2 帧 staging buffer 累积），
// 在 MC 纹理批量加载时容易触发 OOM。
//
// 本函数用 vkGetFenceStatus 非阻塞轮询所有 slot 的 fence，对已 signal 的 slot
// 立即 drain 其 disposalQueue。不会阻塞渲染线程。由 eglSwapBuffers 在每帧
// 开头调用，在新帧分配资源前释放已完成帧的资源。
//
// 返回 drain 的 slot 数（仅用于诊断日志）。
int backend_poll_completed_frames();

// FIX (显存耗尽根因 - 内存压力主动 GC):
//
// 在 vkAllocateMemory 失败之前就主动触发 GC。当 currentAllocationCount
// 达到 maxMemoryAllocationCount 的 70% 时，调用 vkDeviceWaitIdle +
// drain_all_disposal_queues，释放所有延迟销毁的资源。
//
// 这比 try_allocate_memory_with_gc 的失败后重试更有效：在 GPU 进入降级状态
// （timeout/device lost）之前就释放显存，避免级联失败。
//
// 参考 MobileGL 的策略：MobileGL 不会 OOM，因为它每帧主动 drain，但我们
// 的 per-slot drain 不够及时，所以需要这个压力触发的主动 GC。
//
// 返回 true 表示触发了 GC（调用方可记录日志）。
bool backend_proactive_gc_if_needed();

} // namespace vk
} // namespace mithril

#endif // MITHRIL_DIRECTVULKAN_DEVICE_H
