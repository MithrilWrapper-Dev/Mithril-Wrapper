// Mithril-Wrapper - MG_Backend/DirectVulkan/Resources.cpp
// VkBuffer / VkImage / VkImageView / VkSampler lifecycle + GL internalFormat
// -> VkFormat mapping + staging upload. Implements the backend_get_or_create_*
// family declared in MG_Backend/Backend.h.
#include "Resources.h"
#include "Device.h"
#include "CommandStream.h"  // ensure_command_buffer_recording
#include "Pipeline.h"       // clear_all_pipeline_caches — OOM 时驱逐 pipeline
#include "DescriptorSet.h"  // reset_all_descriptor_pools — OOM 时驱逐 descriptor
#include "../Backend.h"
#include "../../MG_Impl/Log.h"

#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace mithril {
namespace vk {

std::unordered_map<GLuint, BufferEntry>&  buffer_table()  { static std::unordered_map<GLuint, BufferEntry>  t; return t; }
std::unordered_map<GLuint, TextureEntry>& texture_table() { static std::unordered_map<GLuint, TextureEntry> t; return t; }
std::unordered_map<GLuint, SamplerEntry>& sampler_table() { static std::unordered_map<GLuint, SamplerEntry> t; return t; }

// ===========================================================================
// FIX (GPU page fault 根因 - buffer 覆写竞争, P0):
// CPU 侧覆写一个 GPU 仍在读取的 buffer（顶点/索引缓冲）会让 GPU 读到撕裂
// 的数据 —— 撕裂的索引值把顶点取址带到 buffer 之外 → GPU Address Fault
// (kIOGPUCommandBufferCallbackErrorPageFault)。Minecraft 每帧都会更新 VBO，
// 首帧安全（无在飞工作），第二帧起上一帧的 command buffer 还在执行时 CPU
// 就开始覆写 → 确定性崩溃，与线上症状完全吻合。
//
// 修复（深度对照 MobileGL VkBufferManager::IsResourceBusy / OnRespecify /
// OnSubData / StagedRangeCopy, VkBufferManager.cpp:200-427）:
//   - glBufferData（整块重指定，GL 有 discard 语义）: buffer 可能在飞时
//     orphan-rename（旧存储延迟销毁 + 全新分配），空闲时才原地更新。
//   - glBufferSubData / map-flush（部分写入）: 在飞时走 staged GPU copy
//     （数据先进 per-frame staging arena，再 vkCmdCopyBuffer 到目标），
//     保持 GL 帧内顺序且绝不覆写 GPU 正在取的字节。
//
// 在飞判定复用现有 submitSerial/completedSerial 水位线（Device.cpp 的
// refresh_completed_serials）：buffer 的最后一次写入将由第 N 次
// vkQueueSubmit 携带（lastWriteSerial = submitSerial+1），当且仅当
// completedSerial >= N（该次提交已在 GPU 完成）时才允许原地写入。
// ===========================================================================

// GL buffer 对象无类型 —— 同一个名字今天绑 GL_ARRAY_BUFFER，明天绑
// GL_SHADER_STORAGE_BUFFER，GL 从不提前告知。Vulkan 要求在创建时声明一切
// 可能的用途，因此这里把 GL 可能用到的 usage 全部加上（与
// backend_get_or_create_buffer 创建路径保持一致）。
static constexpr VkBufferUsageFlags kAllGlBufferUsage =
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

// 该 buffer 是否可能仍被 GPU 引用（其最近一次 CPU 写入尚未完成在飞）。
// 非阻塞：backend_last_completed_serial() 内部只做 vkGetFenceStatus 轮询。
static inline bool buffer_maybe_inflight(const BufferEntry& e) {
    return e.lastWriteSerial > backend_last_completed_serial();
}

// 标记一次 CPU 写入：它将被"下一次 vkQueueSubmit"携带。
static inline void stamp_buffer_write(BufferEntry& e) {
    Backend* b = backend();
    e.lastWriteSerial = b->submitSerial + 1;
}

// 在飞的部分写入：把 `data` 通过 per-frame staging arena（或临时 staging
// buffer）staged 成 GPU copy，记录进当前 command buffer。返回 true 表示
// 成功（调用方无需再做原地写入）。参照 MobileGL StagedRangeCopy。
static bool stage_buffer_range_copy(BufferEntry& e, VkDeviceSize offset,
                                    const void* data, VkDeviceSize size) {
    Backend* b = backend();
    if (!b->device || !data || size == 0) return false;
    // vkCmdCopyBuffer 不能在 render pass 实例内录制（dynamic rendering 同样
    // 禁止 transfer 命令在 pass 内）—— 先结束当前 pass，后续 draw 会重新开。
    if (render_pass_active()) end_render_pass();
    if (!ensure_command_buffer_recording()) return false;

    // Staging 源：优先从 per-frame staging arena sub-allocate（与纹理上传
    // 同一机制，arena 在 slot fence 后才 rewind，安全且零分配）。
    VkBuffer     stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    void*        stagingMapped = nullptr;
    bool         usedArena = false;
    if (b->frameStagingReady) {
        const VkDeviceSize aligned = (b->frameStagingOffset[b->currentFrame] + 255) & ~255;
        if (aligned + size <= Backend::kFrameStagingSize) {
            stagingBuffer = b->frameStagingBuffer[b->currentFrame];
            stagingOffset = aligned;
            stagingMapped = static_cast<uint8_t*>(b->frameStagingMapped[b->currentFrame]) + aligned;
            b->frameStagingOffset[b->currentFrame] = aligned + size;
            usedArena = true;
        }
    }
    if (!usedArena) {
        // Overflow / arena 未就绪：临时 staging buffer + 延迟销毁。
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(b->device, &bci, nullptr, &stagingBuffer) != VK_SUCCESS) return false;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(b->device, stagingBuffer, &mr);
        const uint32_t memType = find_memory_type(mr.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (memType == 0xFFFFFFFFu) {
            vkDestroyBuffer(b->device, stagingBuffer, nullptr);
            return false;
        }
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memType;
        VkDeviceMemory tmpMem = VK_NULL_HANDLE;
        if (try_allocate_memory_with_gc(b->device, &mai, nullptr, &tmpMem) != VK_SUCCESS) {
            vkDestroyBuffer(b->device, stagingBuffer, nullptr);
            return false;
        }
        vkBindBufferMemory(b->device, stagingBuffer, tmpMem, 0);
        if (vkMapMemory(b->device, tmpMem, 0, size, 0, &stagingMapped) != VK_SUCCESS) {
            vkFreeMemory(b->device, tmpMem, nullptr);
            vkDestroyBuffer(b->device, stagingBuffer, nullptr);
            return false;
        }
        b->currentAllocationCount++;
        b->currentVramBytes += mr.size;
        DeferredDestroy ds{};
        ds.buffer = stagingBuffer;
        ds.memory = tmpMem;
        ds.memorySize = mr.size;
        b->disposalQueue[b->currentFrame].push_back(ds);
    }

    std::memcpy(stagingMapped, data, (size_t)size);

    // 把 copy 排在目标 buffer 的所有先前读写之后（帧内在飞 + 同 CB 已录制的
    // 读取），以及所有后续使用之前。镜像 MobileGL StagedRangeCopy 的
    // before/after barrier。
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(b->commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         1, &before, 0, nullptr, 0, nullptr);

    VkBufferCopy region{};
    region.srcOffset = stagingOffset;
    region.dstOffset = offset;
    region.size = size;
    vkCmdCopyBuffer(b->commandBuffer, stagingBuffer, e.buffer, 1, &region);

    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(b->commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         1, &after, 0, nullptr, 0, nullptr);
    return true;
}

uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
    Backend* b = backend();
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(b->physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return 0xFFFFFFFFu;
}

// FIX (device lost + 红屏根因 - GC-only pre-check):
// 核心洞察：MoltenVK 在 vkAllocateMemory/vkCreateImage 失败时可能触发 Metal
// "GPU Address Fault Error"。但我们不能 REJECT 分配（拒绝必要纹理 → 红屏）。
// 解决方案：pre-check 只做 GC（释放空间），不 reject。让 vkAllocateMemory 自己
// 决定是否 OOM。如果 OOM，fallback 路径做三层恢复。
//
// 红屏根因：之前 90% gate 会 REJECT 分配，导致 Minecraft 启动时的必要纹理
//（字体图集 16MB、mojang logo 等）被拒绝 → 纹理缺失 → 红屏。
// 现在 95% gate 只 GC，不 reject，让分配尽可能成功。
VkResult try_allocate_memory_with_gc(VkDevice device, const VkMemoryAllocateInfo* info,
                                     const VkAllocationCallbacks* allocator,
                                     VkDeviceMemory* memory) {
    Backend* b = backend();
    VkDeviceSize reqSize = info ? info->allocationSize : 0;

    // ---- PRE-CHECK: 轻量 GC（只 drain disposal queue，不清 pipeline）----
    // 当前已用 + 请求大小 > 95% 预算时，先 drain disposal queue 释放空间。
    // 只做 drain（释放延迟销毁的资源），不做 clear_all_pipeline_caches
    //（那太激进 — 每次分配都清 pipeline 会导致每帧重建所有 pipeline，
    // 严重影响性能，且 safe_device_wait_idle 在帧中间提交+等待会造成卡顿）。
    // pipeline/descriptor purge 只在 OOM fallback 路径做（vkAllocateMemory 真的失败时）。
    if (b->totalVramBytes > 0 && reqSize > 0) {
        VkDeviceSize gcThreshold = (b->totalVramBytes * 95) / 100;
        if (b->currentVramBytes + reqSize > gcThreshold) {
            static int preGcCount = 0;
            preGcCount++;
            if (preGcCount <= 5 || preGcCount % 50 == 0) {
                MITHRIL_LOG_WARN("vk", "pre-alloc GC: %llu + %llu > %llu MB (95%%), "
                                  "draining disposal queue before vkAllocateMemory "
                                  "(attempt #%d)",
                                  (unsigned long long)(b->currentVramBytes / (1024*1024)),
                                  (unsigned long long)(reqSize / (1024*1024)),
                                  (unsigned long long)(gcThreshold / (1024*1024)),
                                  preGcCount);
            }
            // 轻量 GC：只 drain disposal queue（safe_device_wait_idle 提交当前
            // command buffer 后，所有延迟资源不再被 GPU 引用，可以安全释放）
            safe_device_wait_idle();
            // FIX (GPU page fault root cause): safe_device_wait_idle() re-begins
            // the CURRENT slot's buffer WITHOUT advancing currentFrame, and this
            // frame's descriptor sets still reference the current slot's deferred
            // views (memo not invalidated). Draining disposalQueue[currentFrame]
            // here frees views the current frame's live sets reference -> UAF on
            // the next vkQueueSubmit. Drain all slots EXCEPT the current one; the
            // current slot's queue drains after the frame commits and recycles.
            drain_disposal_queues_except(b->currentFrame);
            // 不清 pipeline/descriptor — 太激进，留到 OOM fallback
        }
    }

    // ---- ATTEMPT: 尝试分配 ----
    VkResult r = vkAllocateMemory(device, info, allocator, memory);
    if (r == VK_SUCCESS) return r;
    if (r != VK_ERROR_OUT_OF_DEVICE_MEMORY) return r;

    // ---- FALLBACK: vkAllocateMemory 返回 OOM ----
    // 只有真正 OOM 时才做激进 purge（pipeline + descriptor）
    static int gcTriggerCount = 0;
    gcTriggerCount++;

    safe_device_wait_idle();

    // Layer 1: drain disposal queues
    // FIX (GPU page fault root cause): drain all slots EXCEPT currentFrame. The
    // current slot's deferred views are still referenced by this frame's live
    // descriptor sets (safe_device_wait_idle re-begins the same slot without
    // invalidating them); freeing them here UAFs the next vkQueueSubmit.
    drain_disposal_queues_except(b->currentFrame);
    r = vkAllocateMemory(device, info, allocator, memory);
    if (r == VK_SUCCESS) {
        if (gcTriggerCount <= 5 || gcTriggerCount % 50 == 0) {
            MITHRIL_LOG_WARN("vk", "OOM recovery L1 (drain): succeeded "
                              "(attempt #%d, VRAM %llu MB)",
                              gcTriggerCount,
                              (unsigned long long)(b->currentVramBytes / (1024*1024)));
        }
        return r;
    }

    // Layer 2: purge pipeline cache + descriptor pools
    if (gcTriggerCount <= 5 || gcTriggerCount % 50 == 0) {
        MITHRIL_LOG_WARN("vk", "OOM recovery L2: purging pipeline+descriptor "
                          "(attempt #%d, VRAM %llu MB)",
                          gcTriggerCount,
                          (unsigned long long)(b->currentVramBytes / (1024*1024)));
    }
    // FIX (GPU page fault root cause — re-entrant purge): NEVER purge while the
    // command buffer is still recording with live descriptor-set / pipeline
    // binds. Even though safe_device_wait_idle() above flushed+re-begun the
    // buffer empty (making the purge safe in the common case), routing through
    // request_purge() guarantees it: if the buffer happens to still hold live
    // binds it defers to the next safe command-buffer boundary instead of
    // vkResetDescriptorPool'ing the very sets the recording buffer references
    // (-> kIOGPUCommandBufferCallbackErrorPageFault at the next submit).
    request_purge();
    r = vkAllocateMemory(device, info, allocator, memory);
    if (r == VK_SUCCESS) {
        if (gcTriggerCount <= 5 || gcTriggerCount % 50 == 0) {
            MITHRIL_LOG_WARN("vk", "OOM recovery L2 (purge): succeeded "
                              "(attempt #%d)", gcTriggerCount);
        }
        return r;
    }

    // Layer 3: 返回 OOM，调用方降级处理
    if (gcTriggerCount <= 5 || gcTriggerCount % 50 == 0) {
        MITHRIL_LOG_ERROR("vk", "OOM after L1+L2 (attempt #%d, "
                          "VRAM %llu/%llu MB) — caller must degrade",
                          gcTriggerCount,
                          (unsigned long long)(b->currentVramBytes / (1024*1024)),
                          (unsigned long long)(b->totalVramBytes / (1024*1024)));
    }
    return r;
}

bool create_buffer(BufferEntry& out, VkDeviceSize size,
                   VkBufferUsageFlags usage, const void* data, bool persistent) {
    Backend* b = backend();
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(b->device, &ci, nullptr, &out.buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(b->device, out.buffer, &req);
    uint32_t mt = find_memory_type(req.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == 0xFFFFFFFFu) { vkDestroyBuffer(b->device, out.buffer, nullptr); out.buffer = VK_NULL_HANDLE; return false; }
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    // FIX (OOM 主动 GC): 使用带 GC 的分配函数，OOM 时先排空延迟队列重试
    if (try_allocate_memory_with_gc(b->device, &ai, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyBuffer(b->device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }
    // FIX (P1): 递增分配计数器（仅诊断）+ 字节级 VRAM 跟踪（MobileGL-style）
    b->currentAllocationCount++;
    b->currentVramBytes += req.size;
    vkBindBufferMemory(b->device, out.buffer, out.memory, 0);
    out.persistentlyMapped = false;
    out.mapped = nullptr;
    void* dst = nullptr;
    if (data) {
        // A persistent buffer is permanently mapped so the app can keep writing
        // through the pointer (Sodium's chunk-upload ring buffer); a non-
        // persistent buffer is uploaded then unmapped.
        if (persistent) {
            if (vkMapMemory(b->device, out.memory, 0, size, 0, &dst) == VK_SUCCESS && dst) {
                std::memcpy(dst, data, (size_t)size);
                out.mapped = dst;
                out.persistentlyMapped = true;
            }
        } else {
            if (vkMapMemory(b->device, out.memory, 0, size, 0, &dst) == VK_SUCCESS && dst) {
                std::memcpy(dst, data, (size_t)size);
                vkUnmapMemory(b->device, out.memory);
            }
        }
    } else if (persistent) {
        // glBufferStorage with a NULL data pointer + MAP_PERSISTENT: keep the
        // mapping live so the app can write through glMapBufferRange later.
        if (vkMapMemory(b->device, out.memory, 0, size, 0, &dst) == VK_SUCCESS) {
            out.mapped = dst;
            out.persistentlyMapped = true;
        }
    }
    out.size = size;
    return true;
}

void destroy_buffer_entry(BufferEntry& e) {
    Backend* b = backend();
    if (!b->device) return;
    if (e.mapped) { vkUnmapMemory(b->device, e.memory); e.mapped = nullptr; }
    if (e.buffer) { vkDestroyBuffer(b->device, e.buffer, nullptr); e.buffer = VK_NULL_HANDLE; }
    if (e.memory) {
        vkFreeMemory(b->device, e.memory, nullptr); e.memory = VK_NULL_HANDLE;
        // FIX (MobileGL-style): decrement VRAM byte tracking
        if (b->currentAllocationCount > 0) b->currentAllocationCount--;
        if (e.size > 0 && b->currentVramBytes >= e.size) b->currentVramBytes -= e.size;
        else if (e.size > 0) b->currentVramBytes = 0;
    }
    e.size = 0;
}

void destroy_texture_entry(TextureEntry& e) {
    Backend* b = backend();
    if (!b->device) return;
    if (e.view)          { vkDestroyImageView(b->device, e.view, nullptr); e.view = VK_NULL_HANDLE; }
    if (e.image)         { vkDestroyImage(b->device, e.image, nullptr); e.image = VK_NULL_HANDLE; }
    if (e.memory)        {
        vkFreeMemory(b->device, e.memory, nullptr); e.memory = VK_NULL_HANDLE;
        // FIX (MobileGL-style): decrement VRAM byte tracking
        if (b->currentAllocationCount > 0) b->currentAllocationCount--;
        if (e.memorySize > 0 && b->currentVramBytes >= e.memorySize) b->currentVramBytes -= e.memorySize;
        else if (e.memorySize > 0) b->currentVramBytes = 0;
    }
    if (e.stagingBuffer) { vkDestroyBuffer(b->device, e.stagingBuffer, nullptr); e.stagingBuffer = VK_NULL_HANDLE; }
    if (e.stagingMemory) {
        vkFreeMemory(b->device, e.stagingMemory, nullptr); e.stagingMemory = VK_NULL_HANDLE;
        if (b->currentAllocationCount > 0) b->currentAllocationCount--;
        if (e.stagingSize > 0 && b->currentVramBytes >= e.stagingSize) b->currentVramBytes -= e.stagingSize;
        else if (e.stagingSize > 0) b->currentVramBytes = 0;
    }
    e.memorySize = 0;
    e.stagingSize = 0;
}

// ---- Deferred destruction (root cause U fix) ----
// Push the entry's Vulkan handles into disposalQueue[currentFrame] and null
// out the entry. The actual vkDestroy* / vkFreeMemory calls happen in
// drain_disposal_queue() after the slot's fence is waited — by then the GPU
// has finished all command buffers that might reference these resources.
// This prevents the Metal resource UAF crash where MoltenVK's command
// encoding retains MTLBuffer/MTLTexture wrappers that were already freed
// by an immediate vkDestroy*.
void defer_destroy_buffer_entry(BufferEntry& e) {
    Backend* b = backend();
    if (!b->device) return;
    if (e.mapped) { vkUnmapMemory(b->device, e.memory); e.mapped = nullptr; }
    if (e.buffer == VK_NULL_HANDLE && e.memory == VK_NULL_HANDLE) return;
    // FIX (GPU page fault root cause): a cached descriptor set (per-program memo)
    // may still reference this buffer/view. Invalidate the memo so no stale set
    // is re-bound after the resource is destroyed.
    mithril::vk::invalidate_descriptor_memo();
    DeferredDestroy d;
    d.buffer = e.buffer;
    d.memory = e.memory;
    d.memorySize = e.size;  // FIX: track bytes for VRAM monitoring
    b->disposalQueue[b->currentFrame].push_back(d);
    e.buffer = VK_NULL_HANDLE;
    e.memory = VK_NULL_HANDLE;
    e.size = 0;
    e.lastWriteSerial = 0;
}

void defer_destroy_texture_entry(TextureEntry& e) {
    Backend* b = backend();
    if (!b->device) return;
    if (e.view == VK_NULL_HANDLE && e.image == VK_NULL_HANDLE &&
        e.memory == VK_NULL_HANDLE && e.stagingBuffer == VK_NULL_HANDLE &&
        e.stagingMemory == VK_NULL_HANDLE) return;
    // FIX (GPU page fault root cause): a cached descriptor set (per-program memo)
    // may still reference this texture view. Invalidate the memo so no stale set
    // is re-bound after the view is destroyed.
    mithril::vk::invalidate_descriptor_memo();
    DeferredDestroy d;
    d.image = e.image;
    d.view  = e.view;
    d.memory = e.memory;
    d.memorySize = e.memorySize;  // FIX: track bytes for VRAM monitoring
    b->disposalQueue[b->currentFrame].push_back(d);
    // Staging buffer/memory go in a separate entry (they may be independently
    // non-null while the main image is null, e.g. during staging resize).
    if (e.stagingBuffer != VK_NULL_HANDLE || e.stagingMemory != VK_NULL_HANDLE) {
        DeferredDestroy ds;
        ds.buffer = e.stagingBuffer;
        ds.memory = e.stagingMemory;
        ds.memorySize = e.stagingSize;  // FIX: track staging bytes too
        b->disposalQueue[b->currentFrame].push_back(ds);
    }
    e.view = VK_NULL_HANDLE;
    e.image = VK_NULL_HANDLE;
    e.memory = VK_NULL_HANDLE;
    e.memorySize = 0;
    e.stagingBuffer = VK_NULL_HANDLE;
    e.stagingMemory = VK_NULL_HANDLE;
    e.stagingSize = 0;
}

void defer_destroy_sampler_entry(SamplerEntry& e) {
    Backend* b = backend();
    if (!b->device) return;
    // FIX (GPU page fault root cause): a cached descriptor set (per-program memo)
    // may still reference this sampler. Invalidate the memo so no stale set is
    // re-bound after the sampler is destroyed.
    mithril::vk::invalidate_descriptor_memo();
    // FIX (按参数缓存): 一个 SamplerEntry 可能持有多个按参数缓存的采样器，
    // 全部延迟销毁，防止任一仍被在途 command buffer 引用。
    for (auto& kv : e.byParams) {
        VkSampler s = kv.second;
        if (s == VK_NULL_HANDLE) continue;
        DeferredDestroy d;
        d.sampler = s;
        b->disposalQueue[b->currentFrame].push_back(d);
    }
    e.byParams.clear();
}

// ---- GL internalFormat -> VkFormat / host texel bytes / aspect mask ----
// (Moved to FormatMap.{h,cpp} so they can be unit-tested without linking the
// rest of the Vulkan backend. See FormatMap.h for the declarations.)

// FIX (compile order): stage_and_copy_image 需要在这几个 static helper 定义之前
// 调用它们，因此在此处前向声明。定义见下方 ~780 行。
static VkAccessFlags       src_access_for_layout(VkImageLayout layout);
static VkAccessFlags       dst_access_for_layout(VkImageLayout layout);
static VkPipelineStageFlags src_stage_for_layout(VkImageLayout layout);
static VkPipelineStageFlags dst_stage_for_layout(VkImageLayout layout);

void stage_and_copy_image(TextureEntry& tex, int level, int x, int y, int z,
                          int w, int h, int d, const void* pixels,
                          int unpack_alignment, GLenum format, GLenum type,
                          bool is_full_upload) {
    Backend* b = backend();
    if (!b->commandBuffer) return;
    // With per-slot command buffers, the alias b->commandBuffer may point at
    // a just-submitted (pending) buffer after commit_frame advanced the slot.
    // ensure_command_buffer_recording() lazily switches to the current slot's
    // buffer, waits on its fence, resets, and begins it. Without this, the
    // vkCmdPipelineBarrier / vkCmdCopyBufferToImage calls below would record
    // into a non-recording buffer (spec UB).
    if (!ensure_command_buffer_recording()) return;
    if (unpack_alignment <= 0) unpack_alignment = 4;  // GL default UNPACK_ALIGNMENT

    // Compute host-side bytes per pixel for this (format, type) pair and
    // honour GL_UNPACK_ALIGNMENT when computing the source row stride. The
    // staging buffer is repacked to be tightly packed, matching
    // VkBufferImageCopy.bufferRowLength == 0 below.
    int bpp = host_texel_bytes(format, type);
    if (bpp <= 0) bpp = 4;  // conservative fallback

    // ---- FIX (根因 W - CRITICAL): RGB→RGBA 逐像素展开 ----
    // Metal 的 MTLPixelFormat 枚举不含 3 分量格式（无 RGB8/RGB16F/RGB32F），
    // FormatMap.cpp 已将 GL_RGB8/GL_RGB/GL_RGB16F/GL_RGB32F 映射到 4 分量
    // VkFormat（R8G8B8A8_*）。但 GL 上传源数据（format=GL_RGB/GL_BGR/
    // GL_RGB_INTEGER）每像素仅 3 分量字节（3/6/12 字节），而 VkImage 期望
    // 4 分量（4/8/16 字节）。若直接 memcpy，GPU 按 4 字节/像素读取 3 字节/
    // 像素数据 → 颜色错位 → 红屏/花屏。
    // 深度参考 MobileGL ExpandRgbSourceToRgba (VkTextureManager.cpp:429+)：
    // 检测源 3 分量 + 目标 4 分量，逐像素复制 RGB 并补 alpha。
    // 对照 ResolveTextureFormatInfo (VkTextureManager.cpp:374-427)。
    //
    // 触发条件：源 format 是 3 分量（GL_RGB/GL_BGR/GL_RGB_INTEGER）且 type
    // 是非 packed 逐分量类型（packed 类型如 GL_UNSIGNED_SHORT_5_6_5 源数据
    // 已与目标 packed 格式匹配，host_texel_bytes 返回 2，无需展开）。
    // 不修改 tex.format（已由 FormatMap 映射为 4 分量）。
    bool expand_rgb = false;
    int src_bpp = bpp;   // 源每像素字节数
    int dst_bpp = bpp;   // 目标每像素字节数（展开后；非展开时 == src_bpp）
    if (format == GL_RGB || format == GL_BGR || format == GL_RGB_INTEGER) {
        switch (type) {
            case GL_UNSIGNED_BYTE:
            case GL_BYTE:
                src_bpp = 3;  dst_bpp = 4;  expand_rgb = true; break;
            case GL_UNSIGNED_SHORT:
            case GL_SHORT:
            case GL_HALF_FLOAT:
                src_bpp = 6;  dst_bpp = 8;  expand_rgb = true; break;
            case GL_UNSIGNED_INT:
            case GL_INT:
            case GL_FLOAT:
                src_bpp = 12; dst_bpp = 16; expand_rgb = true; break;
            default:
                // packed 类型（GL_UNSIGNED_SHORT_5_6_5 等）：源与目标均为
                // packed 字节，host_texel_bytes 已正确返回，无需展开。
                break;
        }
    }

    size_t mask = (size_t)unpack_alignment - 1;
    // staging 装的是展开后的 RGBA 数据（紧密排列）；tight_row 按 dst_bpp 计算。
    // 源行 stride 按 src_bpp 计算并受 GL_UNPACK_ALIGNMENT 约束。
    // 非展开路径下 src_bpp == dst_bpp == bpp，行为与原实现完全一致。
    size_t tight_row = (size_t)w * (size_t)dst_bpp;
    size_t src_tight_row = (size_t)w * (size_t)src_bpp;
    size_t src_stride = (src_tight_row + mask) & ~mask;
    size_t staging = tight_row * (size_t)h * (size_t)d;

    // ---- FIX (Invalid Resource 根因 - per-frame transient staging arena) ----
    // 深度参考 MobileGL 的 transient staging arena 模式：
    // 从当前 frame slot 的大 staging buffer 中 sub-allocate（bump offset），
    // 而不是为每张纹理创建/销毁独立的 staging buffer。
    //
    // 这消除了：
    //   1. per-texture vkCreateBuffer + vkAllocateMemory（降低 allocation count）
    //   2. staging buffer 的 disposalQueue 条目（消除 UAF 风险）
    //   3. vkMapMemory/vkUnmapMemory 的 per-upload 开销（arena persistently mapped）
    //
    // arena 在 ensure_command_buffer_recording 的 fence wait 后 rewind 到 0，
    // 保证 GPU 已完成对该 slot staging buffer 的引用。
    //
    // overflow（staging > 剩余空间）回退到临时 staging buffer + deferred destroy。

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    void* stagingMapped = nullptr;
    bool usedArena = false;

    if (b->frameStagingReady) {
        // 对齐到 256 字节（满足 VkBufferImageCopy.bufferOffset 的对齐要求，
        // 也满足 MoltenVK/Metal 的 MTLBuffer offset 对齐）
        VkDeviceSize alignedOffset = (b->frameStagingOffset[b->currentFrame] + 255) & ~255;
        if (alignedOffset + staging <= Backend::kFrameStagingSize) {
            // Fast path: sub-allocate from per-frame arena
            stagingBuffer = b->frameStagingBuffer[b->currentFrame];
            stagingOffset = alignedOffset;
            stagingMapped = b->frameStagingMapped[b->currentFrame];
            b->frameStagingOffset[b->currentFrame] = alignedOffset + staging;
            usedArena = true;
        }
        // else: overflow — fall through to temporary staging buffer path
    }

    if (!usedArena) {
        // Overflow path: arena 不存在或剩余空间不足。
        // 创建临时 staging buffer，上传后延迟销毁（disposalQueue）。
        // 这是罕见情况（单帧上传超过 16MB），不影响整体性能。
        // 对超大单张纹理（>16MB），也需要走此路径。
        if (tex.stagingSize < staging) {
            if (tex.stagingBuffer != VK_NULL_HANDLE || tex.stagingMemory != VK_NULL_HANDLE) {
                DeferredDestroy ds;
                ds.buffer = tex.stagingBuffer;
                ds.memory = tex.stagingMemory;
                ds.memorySize = tex.stagingSize;  // FIX: track bytes for VRAM monitoring
                b->disposalQueue[b->currentFrame].push_back(ds);
                tex.stagingBuffer = VK_NULL_HANDLE;
                tex.stagingMemory = VK_NULL_HANDLE;
            }
            BufferEntry tmp;
            if (create_buffer(tmp, staging,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT, nullptr)) {
                tex.stagingBuffer = tmp.buffer;
                tex.stagingMemory = tmp.memory;
                tex.stagingSize = staging;
            } else {
                return;
            }
        }
        stagingBuffer = tex.stagingBuffer;
        stagingOffset = 0;
        // Per-upload map for overflow path (arena 的 persistently mapped 不可用)
        vkMapMemory(b->device, tex.stagingMemory, 0, staging, 0, &stagingMapped);
    }

    // Copy pixel data into staging buffer at stagingOffset
    if (stagingMapped && pixels) {
        char* dst = (char*)stagingMapped + stagingOffset;
        if (expand_rgb) {
            // ---- FIX (根因 W): RGB→RGBA 逐像素展开 ----
            // 源每像素 src_bpp 字节（3/6/12 = RGB），目标每像素 dst_bpp 字节
            // （4/8/16 = RGBA）。逐像素复制 RGB 3 分量，第 4 分量（alpha）填
            // 1.0 的位模式。深度参考 MobileGL ExpandRgbSourceToRgba
            // (VkTextureManager.cpp:429+)。
            //
            // alpha 填充按 type 区分（对 sfloat 用 1.0 位模式，避免 0xFF→NaN）：
            //   - unorm byte:  0xFF       (== 1.0)
            //   - unorm short: 0xFFFF     (== 1.0)
            //   - half float:  0x3C00     (1.0 half)
            //   - float:       0x3F800000 (1.0f)
            //   - int/uint:    0x00000001 (integer 1；Minecraft 主流程不采样
            //                              RGB_INTEGER 的 alpha，可接受）
            // alpha_bytes == src_bpp/3 == 每分量字节数（1/2/4）；展开后 staging
            // 紧密排列（dst_bpp * w 每行），VkBufferImageCopy.bufferRowLength==0
            // 仍有效。
            uint32_t alpha_bits = 0x000000FFu;
            switch (type) {
                case GL_UNSIGNED_BYTE:
                case GL_BYTE:
                    alpha_bits = 0x000000FFu; break;       // unorm: 0xFF == 1.0
                case GL_UNSIGNED_SHORT:
                case GL_SHORT:
                    alpha_bits = 0x0000FFFFu; break;       // unorm: 0xFFFF == 1.0
                case GL_HALF_FLOAT:
                    alpha_bits = 0x00003C00u; break;       // 1.0 half-float
                case GL_FLOAT:
                    alpha_bits = 0x3F800000u; break;       // 1.0f
                case GL_UNSIGNED_INT:
                case GL_INT:
                    alpha_bits = 0x00000001u; break;       // integer 1
                default:
                    alpha_bits = 0x000000FFu; break;
            }
            const int alpha_bytes = src_bpp / 3;  // 每分量字节数: 1/2/4
            const char* s8 = (const char*)pixels;
            for (int layer = 0; layer < d; ++layer) {
                for (int row = 0; row < h; ++row) {
                    const char* src_row = s8;
                    for (int px = 0; px < w; ++px) {
                        std::memcpy(dst, src_row, src_bpp);         // 复制 RGB 3 分量
                        dst += src_bpp;
                        src_row += src_bpp;
                        std::memcpy(dst, &alpha_bits, alpha_bytes); // 填充 alpha 第 4 分量
                        dst += alpha_bytes;
                    }
                    s8 += src_stride;  // 源行按 GL_UNPACK_ALIGNMENT stride
                }
            }
        } else if (src_stride == tight_row) {
            // Source rows are already tightly packed — single memcpy.
            std::memcpy(dst, pixels, staging);
        } else {
            // Source rows carry GL_UNPACK_ALIGNMENT padding; repack to tight
            // so VkBufferImageCopy.bufferRowLength == 0 (== w) is valid.
            const char* s8 = (const char*)pixels;
            for (int layer = 0; layer < d; ++layer) {
                for (int row = 0; row < h; ++row) {
                    std::memcpy(dst, s8, tight_row);
                    dst += tight_row;
                    s8 += src_stride;
                }
            }
        }
    }

    if (!usedArena && stagingMapped) {
        // Unmap the per-upload mapping (overflow path only)
        vkUnmapMemory(b->device, tex.stagingMemory);
    }
    // Arena path: no unmap needed (persistently mapped)

    VkBufferImageCopy region{};
    region.bufferOffset = stagingOffset;  // 非 0 for arena sub-allocation
    region.bufferRowLength = 0;     // tightly packed
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    if (tex.format == VK_FORMAT_D16_UNORM || tex.format == VK_FORMAT_D32_SFLOAT)
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (tex.format == VK_FORMAT_S8_UINT)
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    if (tex.format == VK_FORMAT_D24_UNORM_S8_UINT || tex.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;  // depth aspect only
    region.imageSubresource.mipLevel = level;
    // FIX (Main-menu panorama cubemap GPU fault root cause - face→layer mapping):
    // For a cubemap the array layer IS the face (0-5). OpenGL faces are supplied
    // via the 6 face targets; the GL layer passes the face index in the `z`
    // parameter of glTexImage2D/glTexSubImage2D. The old code hardcoded
    // baseArrayLayer=0 and put z into imageOffset.z (a 3D-texture offset), so
    // every face was written into layer 0 and faces 1-5 were never initialized
    // -> sampling an undefined cubemap layer -> MoltenVK/A11 GPU Address Fault.
    // Fix: for cubemaps use baseArrayLayer = z (face), zero the image z-offset;
    // for 3D textures keep z as imageOffset.z and baseArrayLayer = 0.
    if (tex.target == GL_TEXTURE_CUBE_MAP && z >= 0 && z < 6) {
        region.imageSubresource.baseArrayLayer = (uint32_t)z;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { x, y, 0 };
    } else {
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { x, y, z };
    }
    region.imageExtent = { (uint32_t)w, (uint32_t)h, (uint32_t)d };

    // Transition the image layout to TRANSFER_DST for the copy.
    // 根因 F：部分上传（glTexSubImage*）必须用 tex.currentLayout 作为 oldLayout，
    // 保留未更新区域的既有内容；完整上传（glTexImage*）用 UNDEFINED 丢弃旧内容。
    // 无条件用 UNDEFINED 会导致 glTexSubImage2D 后纹理其余区域变 undefined →
    // 纹理损坏 → 物体黑色斑块。
    VkImageLayout oldLayout = is_full_upload ? VK_IMAGE_LAYOUT_UNDEFINED : tex.currentLayout;
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // FIX (GPU page fault root cause - 上传前 barrier 欠同步):
    // 旧代码硬编码 srcAccessMask = 0 / srcStage = TOP_OF_PIPE_BIT。对部分上传
    // （glTexSubImage*，oldLayout != UNDEFINED）而言，这意味着把 image 从
    // SHADER_READ_ONLY_OPTIMAL 过渡到 TRANSFER_DST 时，不等待此前已录制 draw
    // 对该 image 的着色器读取。MoltenVK 因此可能在 fragment/compute 仍在读取时
    // 就开始布局转换并覆盖数据 → 撕裂采样 / kIOGPUCommandBufferCallbackErrorPageFault。
    // 修复：用与 transition_image_layout 相同的 src_stage_for_layout /
    // src_access_for_layout 推导正确的源阶段与访问位。完整上传（UNDEFINED）时
    // 两者返回 0/TOP_OF_PIPE，行为与旧实现一致（无此前内容需等待）。
    barrier.srcAccessMask = src_access_for_layout(oldLayout);
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = tex.image;
    barrier.subresourceRange.aspectMask = region.imageSubresource.aspectMask;
    barrier.subresourceRange.baseMipLevel = level;
    barrier.subresourceRange.levelCount = 1;
    // FIX (cubemap face upload): the barrier subresourceRange must cover the
    // layer that the copy below writes into. The old code kept baseArrayLayer=0;
    // when a cubemap face is uploaded to layer z, the barrier only transitioned
    // layer 0, so layer z kept an undefined/untracked layout and was read back
    // without a proper transition -> MoltenVK/A11 GPU fault risk.
    barrier.subresourceRange.baseArrayLayer = region.imageSubresource.baseArrayLayer;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(b->commandBuffer,
                         src_stage_for_layout(oldLayout),
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    vkCmdCopyBufferToImage(b->commandBuffer, stagingBuffer, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // FIX (Root Cause AH - depth-stencil descriptor layout):
    // 上传后 layout transition 必须按纹理格式选择正确的 read-only 布局。
    // 旧代码硬编码 SHADER_READ_ONLY_OPTIMAL，对 depth-stencil 纹理不匹配
    // image 实际布局 → MoltenVK 验证错误或静默丢 draw → 黑屏。
    // 用 sampled_layout_for_format(tex.format) 选择正确布局：
    //   depth-stencil -> DEPTH_STENCIL_READ_ONLY_OPTIMAL
    //   depth-only    -> DEPTH_READ_ONLY_OPTIMAL
    //   color         -> SHADER_READ_ONLY_OPTIMAL
    // 对照 MobileGL ResolveSampledReadOnlyLayout (VkTextureManager.cpp:177)。
    //
    // 注意：dstAccessMask = SHADER_READ_BIT 覆盖 fragment+compute 的 SHADER_READ，
    // dstStage 现由 dst_stage_for_layout 推导（FRAGMENT|COMPUTE，见下方 FIX）。
    VkImageLayout sampledLayout = sampled_layout_for_format(tex.format);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = sampledLayout;
    // FIX (GPU page fault root cause - compute): 旧代码硬编码 dstStage =
    // FRAGMENT_SHADER_BIT，copy 的写入只对 fragment 后续读取可见。Sodium/Iris
    // 的 compute 管线会在上传后于计算着色器里采样该 image，但 dstStage 不含
    // COMPUTE_SHADER → copy 写入对 compute 采样不可见 → 采样到未初始化/旧数据。
    // 改用 dst_stage_for_layout(sampledLayout)（现含 FRAGMENT|COMPUTE）。
    VkPipelineStageFlags dstStage = dst_stage_for_layout(sampledLayout);
    vkCmdPipelineBarrier(b->commandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
    tex.currentLayout = sampledLayout;

    // Arena path: no cleanup needed — staging buffer 是永久的，offset 在
    // 下次 ensure_command_buffer_recording 时 rewind。
    //
    // Overflow path: 延迟释放临时 staging buffer（与原实现相同）。
    if (!usedArena) {
        if (tex.stagingBuffer != VK_NULL_HANDLE || tex.stagingMemory != VK_NULL_HANDLE) {
            DeferredDestroy ds;
            ds.buffer = tex.stagingBuffer;
            ds.memory = tex.stagingMemory;
            ds.memorySize = tex.stagingSize;  // FIX: track bytes for VRAM monitoring
            b->disposalQueue[b->currentFrame].push_back(ds);
            tex.stagingBuffer = VK_NULL_HANDLE;
            tex.stagingMemory = VK_NULL_HANDLE;
            tex.stagingSize = 0;
        }
    }
}

// aspect_for_format() moved to FormatMap.{h,cpp} (pure-logic helper).

// Stage masks for the source side of an image-memory barrier, keyed on the
// old layout. Returns 0 when the old layout is UNDEFINED or PREINITIALIZED
// (no prior work needs to be visible).
static VkAccessFlags src_access_for_layout(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return VK_ACCESS_SHADER_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return VK_ACCESS_TRANSFER_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_ACCESS_TRANSFER_WRITE_BIT;
        case VK_IMAGE_LAYOUT_GENERAL:
            return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        default:
            return 0;  // UNDEFINED / PREINITIALIZED
    }
}

// Stage masks for the destination side of an image-memory barrier, keyed on
// the new layout. Returns 0 when the new layout is PREINITIALIZED (invalid).
static VkAccessFlags dst_access_for_layout(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return VK_ACCESS_SHADER_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return VK_ACCESS_TRANSFER_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_ACCESS_TRANSFER_WRITE_BIT;
        case VK_IMAGE_LAYOUT_GENERAL:
            return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        default:
            return 0;
    }
}

// Source pipeline stage for an image-memory barrier, keyed on the old layout.
static VkPipelineStageFlags src_stage_for_layout(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            return VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            // FIX (GPU page fault root cause - compute): 纹理既被 fragment 也被
            // compute 着色器采样（Sodium/Iris 的 compute 管线在计算着色器里采样
            // 方块图集/光照贴图）。若此处只写 FRAGMENT_SHADER_BIT，那么 pre-copy
            // barrier 不会等待此前 compute 着色器对该 image 的读取 → 布局转换/覆盖
            // 上传可能在 compute 仍读取时发生 → 撕裂采样 / MoltenVK GPU page fault。
            // 同时覆盖两个 shader 阶段（访问位已含 SHADER_READ_BIT，两者共享）。
            return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_PIPELINE_STAGE_TRANSFER_BIT;
        default:
            return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
}

// Destination pipeline stage for an image-memory barrier, keyed on the new layout.
static VkPipelineStageFlags dst_stage_for_layout(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            // FIX (GPU page fault root cause - compute): 与 src_stage_for_layout 对称，
            // 上传后把 image 转回 SHADER_READ_ONLY 的 barrier 必须让 copy 的写入对
            // compute 着色器的后续读取可见（Sodium/Iris compute 管线）。
            return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_PIPELINE_STAGE_TRANSFER_BIT;
        default:
            return VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }
}

void transition_image_layout(TextureEntry& tex, VkImageLayout newLayout) {
    Backend* b = backend();
    if (!b->commandBuffer || tex.image == VK_NULL_HANDLE) return;
    if (tex.currentLayout == newLayout) return;  // already there
    // Ensure the current slot's command buffer is recording before we issue
    // vkCmdPipelineBarrier. See stage_and_copy_image for rationale.
    if (!ensure_command_buffer_recording()) return;

    VkImageAspectFlags aspect = aspect_for_format(tex.format);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access_for_layout(tex.currentLayout);
    barrier.dstAccessMask = dst_access_for_layout(newLayout);
    barrier.oldLayout = tex.currentLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = tex.image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = (uint32_t)tex.levels;
    barrier.subresourceRange.baseArrayLayer = 0;
    // FIX (Main-menu panorama cubemap GPU fault root cause): this whole-image
    // transition hardcoded layerCount=1. For a cubemap (arrayLayers=6) only
    // layer 0 was transitioned; the other 5 layers stayed in an undefined/
    // stale layout and were later sampled without a proper barrier -> MoltenVK/
    // A11 GPU address fault. Cover the full layer count for cubemaps (2D and
    // 3D images keep layerCount=1, so the blast radius is unchanged there).
    barrier.subresourceRange.layerCount = (tex.target == GL_TEXTURE_CUBE_MAP) ? 6u : 1u;

    vkCmdPipelineBarrier(b->commandBuffer,
                         src_stage_for_layout(tex.currentLayout),
                         dst_stage_for_layout(newLayout),
                         0,
                         0, nullptr, 0, nullptr, 1, &barrier);
    tex.currentLayout = newLayout;
}

// ---- GL filter/wrap -> VkFilter / VkSamplerAddressMode ----
static VkFilter to_vk_filter(GLenum f) {
    if (f == GL_NEAREST || f == GL_NEAREST_MIPMAP_NEAREST || f == GL_NEAREST_MIPMAP_LINEAR) return VK_FILTER_NEAREST;
    return VK_FILTER_LINEAR;
}
static VkSamplerMipmapMode to_vk_mipmap(GLenum f) {
    // FIX (纯红 + GPU page fault): 非 mipmap 的 min filter（GL_NEAREST / GL_LINEAR）
    // 必须映射为 NEAREST mip 模式，否则配合 maxLod=12 会请求跨不存在的 mip 层
    // 采样 → A11/MoltenVK 采样越界。mip 模式只在 LOD 范围跨多层时才有意义；
    // 非 mipmap filter 应在 backend_get_or_create_sampler 里把 minLod=maxLod=0。
    if (f == GL_NEAREST_MIPMAP_NEAREST || f == GL_LINEAR_MIPMAP_NEAREST ||
        f == GL_NEAREST || f == GL_LINEAR) return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    return VK_SAMPLER_MIPMAP_MODE_LINEAR;
}
static VkSamplerAddressMode to_vk_wrap(GLenum w) {
    switch (w) {
        case GL_CLAMP_TO_EDGE:        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case GL_CLAMP_TO_BORDER:      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case GL_MIRRORED_REPEAT:      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case GL_REPEAT:
        default:                      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

// ===========================================================================
// FIX (P0-1 / P0-3): 运行时格式能力回退
//
// FormatMap.cpp 是纯查表层（不碰 VkDevice，可单测），它把
// GL_DEPTH_COMPONENT24 / GL_DEPTH24_STENCIL8 一律映射为
// VK_FORMAT_D24_UNORM_S8_UINT。这在 Apple 平台上是错的：
//
//   Metal 的 MTLPixelFormatDepth24Unorm_Stencil8 在**所有 iOS/tvOS 设备上
//   都不可用**，在 Apple Silicon Mac 上同样不可用（只有部分 Intel Mac 的
//   独显支持）。MoltenVK 因此不为 D24_UNORM_S8_UINT 报告
//   DEPTH_STENCIL_ATTACHMENT_BIT。
//
// 后果：vkCreateImage 失败 → 深度附件为 VK_NULL_HANDLE → 深度测试整体失效
// 或 renderpass/pipeline 创建失败 → 黑屏。这是 iOS 上最典型的一类死法。
//
// 修复：按上游 MobileGL FindSupportedDepthStencilFormat
// (SwapchainObject.cpp:60-71) 的思路做候选链探测，但比上游更细 ——
// 上游只处理 swapchain 的那一个深度格式，我们要处理用户通过
// glTexImage2D / glRenderbufferStorage 传进来的**任意**深度格式，
// 所以按「是否需要 stencil」分成两条候选链，避免把纯深度请求
// 升级成 depth-stencil（那会浪费显存并改变 aspectMask）。
//
// 结果按 VkFormat 缓存，避免每次建纹理都调
// vkGetPhysicalDeviceFormatProperties。
// ===========================================================================
VkFormat resolve_supported_format(VkFormat requested, VkFormatFeatureFlags requiredFeatures) {
    if (requested == VK_FORMAT_UNDEFINED) return VK_FORMAT_UNDEFINED;

    // 缓存 key 要含 requiredFeatures：同一格式在「要求可采样」和
    // 「要求可作深度附件」两种场景下的回退结果可能不同。
    struct CacheKey {
        VkFormat fmt;
        VkFormatFeatureFlags feats;
        bool operator==(const CacheKey& o) const { return fmt == o.fmt && feats == o.feats; }
    };
    struct CacheHash {
        size_t operator()(const CacheKey& k) const {
            return (size_t)k.fmt * 1315423911u ^ (size_t)k.feats;
        }
    };
    static std::unordered_map<CacheKey, VkFormat, CacheHash> cache;

    CacheKey key{requested, requiredFeatures};
    auto cit = cache.find(key);
    if (cit != cache.end()) return cit->second;

    Backend* b = backend();
    if (!b || b->physicalDevice == VK_NULL_HANDLE) return requested;

    auto supports = [&](VkFormat f) -> bool {
        VkFormatProperties p{};
        vkGetPhysicalDeviceFormatProperties(b->physicalDevice, f, &p);
        return (p.optimalTilingFeatures & requiredFeatures) == requiredFeatures;
    };

    if (supports(requested)) {
        cache[key] = requested;
        return requested;
    }

    // ---- 候选链 ----
    // 原则：宁可提高精度/多一个 stencil，也不要降级到无法承载语义的格式。
    // 纯深度请求优先保持纯深度（省显存、aspectMask 不变），实在不行才
    // 退到 depth-stencil 组合格式。
    static const VkFormat kDepthStencilChain[] = {
        VK_FORMAT_D24_UNORM_S8_UINT,   // 桌面 GL 原生语义，Intel Mac 独显可用
        VK_FORMAT_D32_SFLOAT_S8_UINT,  // Apple 平台的实际主力（精度更高）
        VK_FORMAT_D16_UNORM_S8_UINT,   // 极少见，兜底
    };
    static const VkFormat kDepthOnlyChain[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_X8_D24_UNORM_PACK32,
        VK_FORMAT_D16_UNORM,
        // 纯深度全不可用时，才允许升级成带 stencil 的组合格式。
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
    };
    static const VkFormat kStencilChain[] = {
        VK_FORMAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
    };

    const VkFormat* chain = nullptr;
    size_t chainLen = 0;

    switch (requested) {
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_D16_UNORM_S8_UINT:
            chain = kDepthStencilChain; chainLen = 3; break;
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
            chain = kDepthOnlyChain; chainLen = 5; break;
        case VK_FORMAT_S8_UINT:
            chain = kStencilChain; chainLen = 3; break;
        default:
            break;
    }

    VkFormat chosen = VK_FORMAT_UNDEFINED;
    for (size_t i = 0; i < chainLen; ++i) {
        if (chain[i] != requested && supports(chain[i])) { chosen = chain[i]; break; }
    }

    if (chosen == VK_FORMAT_UNDEFINED) {
        // 非深度格式（例如 BC 压缩纹理在 iOS 上不可用）没有通用替代品。
        // 返回 UNDEFINED 让调用方决定：纹理路径会退回 RGBA8，
        // 附件路径会放弃该 usage bit，都比 vkCreateImage 硬失败好。
        static std::unordered_set<uint32_t> warned;
        if (warned.insert((uint32_t)requested).second) {
            MITHRIL_LOG_WARN("vk", "resolve_supported_format: VkFormat %d 不被设备支持"
                             "（需要 feature 位 0x%x），且无可用替代格式",
                             (int)requested, (unsigned)requiredFeatures);
        }
        cache[key] = VK_FORMAT_UNDEFINED;
        return VK_FORMAT_UNDEFINED;
    }

    static std::unordered_set<uint64_t> warnedSub;
    uint64_t wk = ((uint64_t)requested << 32) | (uint32_t)chosen;
    if (warnedSub.insert(wk).second) {
        MITHRIL_LOG_INFO("vk", "深度/模板格式回退：VkFormat %d 不受支持，改用 %d"
                         "（Apple GPU 普遍不支持 D24_UNORM_S8_UINT，属预期行为）",
                         (int)requested, (int)chosen);
    }
    cache[key] = chosen;
    return chosen;
}

bool format_is_depth_stencil(VkFormat fmt) {
    switch (fmt) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_S8_UINT:
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        default:
            return false;
    }
}

} // namespace vk
} // namespace mithril

// ===========================================================================
// Public C API (declared in MG_Backend/Backend.h). These thin wrappers map GL
// names to the vk::* tables above and create/destroy Vulkan resources.
// ===========================================================================
extern "C" {

VkBuffer backend_get_or_create_buffer(GLuint name, const void* data, size_t size) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->initialized || name == 0 || size == 0) return VK_NULL_HANDLE;
    auto& tbl = mithril::vk::buffer_table();
    auto it = tbl.find(name);
    // FIX (显存碎片化/持续增长): 当 buffer 已存在且大小足够时，优先原地更新
    // （vkMapMemory + memcpy），而不是 orphan 旧 buffer 重新分配。
    // 原实现要求 !data 才复用，但 glBufferData 和 UBO 更新都传 data != nullptr，
    // 导致每次调用都走 orphan 路径（destroy + realloc），每帧产生上百次
    // vkCreateBuffer/vkAllocateMemory 和延迟销毁，在 iPhone SE 3 等显存紧张
    // 设备上导致 VK_ERROR_OUT_OF_DEVICE_MEMORY。
    //
    // FIX (GPU page fault 根因 - 覆写竞争): 原地更新必须加"在飞"检查。
    // 若 GPU 仍在读取该 buffer（上一次写入的提交未完成），原地 memcpy 会让
    // GPU 读到撕裂的顶点/索引数据 → GPU Address Fault。在飞时改为
    // orphan-rename（MobileGL OnRespecify 的 busy 分支）：glBufferData 有
    // discard 语义，无需拷贝旧内容，直接换新存储。空闲时才走快速原地路径。
    if (it != tbl.end() && it->second.size >= (VkDeviceSize)size) {
        if (data && size > 0 && mithril::vk::buffer_maybe_inflight(it->second)) {
            // 可能仍在飞：orphan-rename
            mithril::vk::defer_destroy_buffer_entry(it->second);
            mithril::vk::BufferEntry e;
            if (!mithril::vk::create_buffer(e, size, mithril::vk::kAllGlBufferUsage, data, false)) {
                return VK_NULL_HANDLE;
            }
            mithril::vk::stamp_buffer_write(e);
            tbl[name] = e;
            // 注意：这里【不】调 invalidate_descriptor_memo()。backend_get_or_create_buffer
            // 主要服务 VBO/IBO/零缓冲/通用属性缓冲/临时索引缓冲 —— 这些【不】被
            // descriptor set 引用（它们经 vkCmdBindVertexBuffers / vkCmdBindIndexBuffer
            // 直接绑定，不经过 descriptor）。真正的 UBO/SSBO（会进 descriptor set、
            // 换新句柄时必须失效 memo 才能避免 UAF）走 backend_create_buffer_storage
            // （glBufferStorage 持久路径），已在那边失效。若在这里也全量调用
            // invalidate_descriptor_memo()，会经 on_command_buffer_boundary() 把
            // descriptorsBound 置 false，导致后续 draw 被 draw_recording_allowed 丢弃
            // （渲染冒烟测试：无 descriptor 的空 program 三角形全黑）。
            return e.buffer;
        }
        // Buffer 已存在且容量足够：原地更新数据（如果有）
        if (data && size > 0) {
            void* dst = nullptr;
            if (vkMapMemory(b->device, it->second.memory, 0, size, 0, &dst) == VK_SUCCESS && dst) {
                std::memcpy(dst, data, size);
                vkUnmapMemory(b->device, it->second.memory);
            }
        }
        mithril::vk::stamp_buffer_write(it->second);
        return it->second.buffer;
    }
    // Buffer 不存在或容量不足：orphan 旧的，创建新的
    if (it != tbl.end()) mithril::vk::defer_destroy_buffer_entry(it->second);
    mithril::vk::BufferEntry e;
    /* GL buffer objects are untyped — the same name can be bound to
     * GL_ARRAY_BUFFER today and GL_SHADER_STORAGE_BUFFER tomorrow, and GL
     * never tells us in advance. Vulkan wants every intended use declared at
     * creation, so every usage GL could possibly ask for goes on here.
     *
     * FIX (root cause AT): STORAGE_BUFFER and INDIRECT_BUFFER were missing.
     * Without INDIRECT_BUFFER the indirect draw path added in the previous
     * pass violates VUID-vkCmdDrawIndirect-buffer-02709 the moment it is
     * used; without STORAGE_BUFFER an SSBO binding is illegal the same way.
     * Both are spec violations that a validation layer rejects outright and
     * MoltenVK may turn into a dropped draw. */
    if (!mithril::vk::create_buffer(e, size, mithril::vk::kAllGlBufferUsage, data, false)) {
        return VK_NULL_HANDLE;
    }
    mithril::vk::stamp_buffer_write(e);
    tbl[name] = e;
    // 同理不在此全量 invalidate_descriptor_memo()（见上方在飞分支注释）：
    // 本函数服务的是不进 descriptor 的 VBO/IBO/临时缓冲。UBO/SSBO 的失效在
    // backend_create_buffer_storage 统一处理。
    return e.buffer;
}

/*
 * Immutable, possibly persistently-mapped storage (GL_ARB_buffer_storage).
 * Mirrors backend_get_or_create_buffer but keeps the host mapping live when
 * `persistent` is set, so the app can write through the pointer returned by
 * glMapBufferRange without re-mapping each frame. Backing memory is always
 * HOST_VISIBLE | HOST_COHERENT, so a persistent+coherent buffer needs no flush.
 */
VkBuffer backend_create_buffer_storage(GLuint name, VkDeviceSize size,
                                       VkBufferUsageFlags extra_usage,
                                       bool persistent, bool coherent) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->initialized || name == 0 || size == 0) return VK_NULL_HANDLE;
    auto& tbl = mithril::vk::buffer_table();
    auto it = tbl.find(name);
    if (it != tbl.end()) mithril::vk::defer_destroy_buffer_entry(it->second);
    mithril::vk::BufferEntry e;
    VkBufferUsageFlags usage =
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra_usage;
    if (!mithril::vk::create_buffer(e, size, usage, nullptr, persistent)) {
        return VK_NULL_HANDLE;
    }
    (void)coherent;  // HOST_COHERENT is always requested by create_buffer
    mithril::vk::stamp_buffer_write(e);
    tbl[name] = e;
    // FIX (GPU page fault 根因 - buffer storage orphan):
    // glBufferStorage 重新指定同样 orphan 旧 buffer 换新 VkBuffer，需作废
    // descriptor memo，避免缓存 set 引用已释放的旧 handle。
    mithril::vk::invalidate_descriptor_memo();
    return e.buffer;
}

void backend_buffer_upload(GLuint name, GLintptr offset, const void* data, size_t size) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->initialized) return;
    auto& tbl = mithril::vk::buffer_table();
    auto it = tbl.find(name);
    if (it == tbl.end()) return;
    if (data && size > 0 && mithril::vk::buffer_maybe_inflight(it->second)) {
        // FIX (GPU page fault 根因 - 覆写竞争): glBufferSubData /
        // glMapBufferRange-flush 在 buffer 可能在飞时改为 staged GPU copy
        // （MobileGL OnSubData / OnFlushMappedRange 的 busy 分支），保持
        // GL 帧内顺序，绝不原地覆写 GPU 正在取的字节。
        //
        // FIX (GPU page fault 剩余根因 - 危险回退): 原实现当 staging 失败时
        // （arena 溢出 + 临时 staging 分配在 VRAM 压力下 OOM）会【回退到原地
        // vkMapMemory + memcpy】。这在 buffer_maybe_inflight==true 时是真实的
        // UAF：GPU 仍在异步读取该 buffer（世界区块网格 ring buffer 每帧被写、
        // 上一帧数据仍在被 chunk 渲染读取），原地覆写会让 MoltenVK/Metal 对
        // 同一地址执行"先读旧值再被覆盖"的竞争，GPU 在 Metal 异步编码/执行
        // 时读到撕裂甚至已释放的内存 → kIOGPUCommandBufferCallbackErrorPageFault。
        // 此路径在世界加载（Sodium 批量上传区块网格 + VRAM 压力下 staging 分配
        // 失败）时恰好触发，且无任何日志告警 —— 与线上症状完全吻合。
        //
        // 修复：buffer 可能在飞且 staging 失败时，【不静默丢弃】。静默丢弃会让
        // GPU 继续读旧/未初始化的顶点数据 —— 对于 chunk 网格 / 全屏 quad 这类
        // 关键缓冲，缺数据可能让 GPU 在 vkQueueSubmit 执行期间引用无效内容，
        // 直接 kIOGPUCommandBufferCallbackErrorPageFault。参照 MobileGL
        // OnSubData / OnFlushMappedRange 的 busy 分支：staging 失败时强制
        // vkDeviceWaitIdle（让本 buffer 之前所有提交都完成、脱离在飞），再
        // 安全原地写入 —— 数据必达 GPU，且绝不覆写仍在飞的字节。
        if (!mithril::vk::stage_buffer_range_copy(it->second,
                                                  (VkDeviceSize)offset, data,
                                                  (VkDeviceSize)size)) {
            // FIX (staging 失败回退 - MobileGL OnSubData busy 分支):
            // 先强制同步：safe_device_wait_idle 提交当前录制的命令缓冲并等待
            // 全部完成，刷新 completedSerial 水位线，使本 buffer 不再在飞。
            // 与 Device.cpp 1498 注释的坑不同：这里【不 drain disposal queue】，
            // 只等 idle，因此不会释放被描述符引用的资源，无 UAF。代价是本次
            // mid-frame 同步（罕见：仅当 staging arena 溢出且临时 staging 分配
            // 在显存压力下 OOM），对性能影响可忽略。
            mithril::vk::safe_device_wait_idle();
            if (!mithril::vk::buffer_maybe_inflight(it->second)) {
                // 已脱离在飞：安全原地写入，数据必达 GPU。
                void* dst = nullptr;
                vkMapMemory(b->device, it->second.memory, (VkDeviceSize)offset,
                            (VkDeviceSize)size, 0, &dst);
                if (dst) { std::memcpy(dst, data, (size_t)size); vkUnmapMemory(b->device, it->second.memory); }
                mithril::vk::stamp_buffer_write(it->second);
                return;
            }
            // 极端兜底（idle 后仍异常在飞）：丢弃并告警，绝不原地覆写。
            static int uploadDropCount = 0;
            uploadDropCount++;
            if (uploadDropCount <= 5 || uploadDropCount % 200 == 0) {
                MITHRIL_LOG_WARN("vk", "backend_buffer_upload: buffer %u (offset %lld, "
                                  "%zu bytes) still in flight after safe_device_wait_idle "
                                  "and staging FAILED — DROPPING upload (drop #%d)",
                                  name, (long long)offset, size, uploadDropCount);
            }
            return;
        }
        mithril::vk::stamp_buffer_write(it->second);
        return;
    }
    // Buffer 不在飞：可以安全原地更新。
    void* dst = nullptr;
    if (data && size > 0) {
        vkMapMemory(b->device, it->second.memory, offset, size, 0, &dst);
        if (dst) { std::memcpy(dst, data, size); vkUnmapMemory(b->device, it->second.memory); }
    }
    mithril::vk::stamp_buffer_write(it->second);
}

VkBuffer backend_get_buffer(GLuint name) {
    auto& tbl = mithril::vk::buffer_table();
    auto it = tbl.find(name);
    return it == tbl.end() ? VK_NULL_HANDLE : it->second.buffer;
}

void* backend_get_buffer_mapped_pointer(GLuint name) {
    auto& tbl = mithril::vk::buffer_table();
    auto it = tbl.find(name);
    if (it == tbl.end()) return nullptr;
    return it->second.persistentlyMapped ? it->second.mapped : nullptr;
}

void backend_delete_buffer(GLuint name) {
    auto& tbl = mithril::vk::buffer_table();
    auto it = tbl.find(name);
    if (it == tbl.end()) return;
    mithril::vk::defer_destroy_buffer_entry(it->second);
    tbl.erase(it);
}

VkBuffer backend_get_zero_buffer(void) {
    static GLuint zero_name = 0x40000000u;  // sentinel name for the shared zero buffer
    static bool tried = false;
    if (!tried) {
        tried = true;
        static const uint8_t zeros[16] = {0};
        backend_get_or_create_buffer(zero_name, zeros, sizeof(zeros));
    }
    return backend_get_buffer(zero_name);
}

/*
 * Generic vertex attribute values (root cause AQ).
 *
 * One buffer holding kMaxVertexAttribs vec4 slots, so a disabled attribute
 * array can be bound to `buffer + index * 16` with stride 0 and read back the
 * constant the application set with glVertexAttrib*().
 *
 * This replaces binding the shared ZERO buffer to unenabled slots. That gave
 * every disabled attribute (0,0,0,0), but GL's documented default is
 * (0,0,0,1): a shader reading a disabled colour array is supposed to see
 * opaque black, and with alpha 0 it instead rendered nothing at all.
 */
static const GLuint kGenericAttribBufferName = 0x40000001u;

VkBuffer backend_get_generic_attrib_buffer(void) {
    return backend_get_buffer(kGenericAttribBufferName);
}

void backend_update_generic_attribs(const float* values, int count) {
    if (!values || count <= 0) return;
    const size_t bytes = (size_t)count * 4 * sizeof(float);
    VkBuffer existing = backend_get_buffer(kGenericAttribBufferName);
    if (existing == VK_NULL_HANDLE) {
        backend_get_or_create_buffer(kGenericAttribBufferName, values, bytes);
        return;
    }
    // FIX (容量防御): 若新数据超过当前分配（首帧 attrib count 小于后续帧），
    // 直接重建 buffer 而不是 map 越界写入。
    auto& tbl = mithril::vk::buffer_table();
    auto it = tbl.find(kGenericAttribBufferName);
    if (it != tbl.end() && it->second.size < (VkDeviceSize)bytes) {
        mithril::vk::defer_destroy_buffer_entry(it->second);
        mithril::vk::BufferEntry e;
        if (mithril::vk::create_buffer(e, bytes, mithril::vk::kAllGlBufferUsage, values, false)) {
            mithril::vk::stamp_buffer_write(e);
            tbl[kGenericAttribBufferName] = e;
        }
        return;
    }
    backend_buffer_upload(kGenericAttribBufferName, 0, values, bytes);
}

VkImage backend_get_or_create_texture(GLuint name, int width, int height, int depth,
                                      int levels, GLenum internal_format, GLenum target,
                                      int samples) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->initialized || name == 0 || width <= 0 || height <= 0) return VK_NULL_HANDLE;
    // FIX (Main-menu panorama cubemap GPU fault root cause): glTexImage2D may
    // pass any of the 6 cubemap face targets (GL_TEXTURE_CUBE_MAP_POSITIVE_X..
    // NEGATIVE_Z). Every downstream cubemap decision here (arrayLayers=6,
    // VIEW_TYPE_CUBE, upload face->layer) keys off `target == GL_TEXTURE_CUBE_MAP`,
    // so a face target must be normalized first, or the image is created with
    // arrayLayers=1 and the 6 faces get crammed into layer 0 + sampled out of
    // bounds -> MoltenVK/A11 GPU Address Fault.
    if (target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X &&
        target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) {
        target = GL_TEXTURE_CUBE_MAP;
    }
    VkFormat fmt = mithril::vk::gl_internal_to_vk(internal_format);
    if (fmt == VK_FORMAT_UNDEFINED) {
        // FIX (日志刷屏): 同一不支持的格式会被反复打印。用 static set 去重，
        // 每种格式只打印一次。
        static std::unordered_set<GLenum> warnedFormats;
        if (warnedFormats.find(internal_format) == warnedFormats.end()) {
            warnedFormats.insert(internal_format);
            MITHRIL_LOG_WARN("vk", "backend_get_or_create_texture: unsupported "
                              "internalFormat 0x%x (falling back to RGBA8, "
                              "further occurrences of this format suppressed)",
                              internal_format);
        }
        fmt = VK_FORMAT_R8G8B8A8_UNORM;
    }

    // FIX (P0-1): 查表得到的是「理想格式」，必须再过一遍设备能力。
    // 深度格式尤其关键 —— Apple GPU 不支持 D24_UNORM_S8_UINT，
    // 不回退就是 vkCreateImage 失败 → 无深度附件 → 黑屏。
    {
        const bool isDepth = mithril::vk::format_is_depth_stencil(fmt);
        // 深度纹理必须能当深度附件；颜色纹理至少要能被采样（MC 的纹理
        // 归根到底都是拿来采样的，采样不了就没有意义）。
        const VkFormatFeatureFlags need =
            isDepth ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
                    : VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        VkFormat resolved = mithril::vk::resolve_supported_format(fmt, need);
        if (resolved == VK_FORMAT_UNDEFINED) {
            // 无替代格式。压缩格式（BC1/2/3 在 iOS 上全不可用）会走到这里。
            // 退回 RGBA8：画面会丢失压缩纹理的内容，但至少资源能建出来、
            // 管线不会整条失败。真正的解法是在上层把 BCn 转码成 ASTC，
            // 那属于纹理转码器的范畴，不在本函数职责内。
            static std::unordered_set<uint32_t> warnedNoAlt;
            if (warnedNoAlt.insert((uint32_t)fmt).second) {
                MITHRIL_LOG_WARN("vk", "internalFormat 0x%x → VkFormat %d 在本设备"
                                 "完全不受支持且无替代（iOS 无 BC 压缩纹理支持），"
                                 "退回 RGBA8", internal_format, (int)fmt);
            }
            fmt = VK_FORMAT_R8G8B8A8_UNORM;
        } else {
            fmt = resolved;
        }
    }

    auto& tbl = mithril::vk::texture_table();
    auto it = tbl.find(name);
    // FIX (Root Cause AI - glTexImage2D mipmap uses base level dimensions):
    // 复用条件额外比较 levels（mip level count）。如果请求的 levels 与现有
    // VkImage 的 levels 不同（例如 glTexImage2D 逐级上传时 t->levels 递增），
    // 必须重建 VkImage 以匹配新的 mipLevels。否则 VkImage 的 mipLevels 不足，
    // 上传高 level 数据时 vkCmdCopyBufferToImage 会写入越界 mip level →
    // 纹理腐败 / 验证错误。
    // 对照 MobileGL CheckMipmapCompleteness (VkTextureManager.cpp:1918-1957)。
    int effective_levels = levels > 0 ? levels : 1;
    if (it != tbl.end() && it->second.image != VK_NULL_HANDLE &&
        it->second.format == fmt &&
        it->second.width == width && it->second.height == height &&
        it->second.depth == depth &&
        it->second.levels == effective_levels) {
        return it->second.image;
    }
    if (it != tbl.end()) mithril::vk::defer_destroy_texture_entry(it->second);

    mithril::vk::TextureEntry e;
    e.format = fmt;
    e.width = width; e.height = height; e.depth = depth;
    e.levels = levels > 0 ? levels : 1;
    e.target = target;

    VkImageType imgType = (target == GL_TEXTURE_3D) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = imgType;
    ici.format = fmt;
    ici.extent = { (uint32_t)width, (uint32_t)height, (uint32_t)(imgType == VK_IMAGE_TYPE_3D ? depth : 1) };
    ici.mipLevels = e.levels;
    ici.arrayLayers = (imgType == VK_IMAGE_TYPE_3D) ? 1 : (target == GL_TEXTURE_CUBE_MAP ? 6 : 1);
    ici.samples = (VkSampleCountFlagBits)(samples > 1 ? samples : 1);
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    // FIX (Root Cause I - 颜色纹理缺 COLOR_ATTACHMENT_BIT):
    // Minecraft 延迟渲染器通过 glFramebufferTexture2D 将颜色纹理绑定为 FBO 颜色附件。
    // Vulkan 规范要求颜色附件图像必须含 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT，
    // 否则渲染被 MoltenVK 静默丢弃（无验证错误，release 构建下不可见）→ FBO 渲染
    // 内容丢失 → swapchain 仅剩 clear color → 进游戏后黑屏。
    // 对所有颜色纹理无条件添加此 bit（Vulkan 允许设置未使用的 usage bit，无副作用），
    // 对标 MobileGL VulkanRenderer::CreateTexture 的纹理创建策略。
    // 注意：仅对颜色格式添加；depth/stencil 格式由下方 if 分支单独处理。
    //
    // FIX (GL 4.2 ARB_shader_image_load_store / GL 4.3 compute):
    // Iris/Sodium 的 compute 路径会用 glBindImageTexture 把普通 GL 纹理绑定为
    // storage image。Vulkan 要求该图像创建时带 VK_IMAGE_USAGE_STORAGE_BIT。
    // 但不能无条件加：压缩格式(BCn)、sRGB、部分 depth 格式不支持 STORAGE，
    // 无条件添加会让 vkCreateImage 直接失败 → 纹理全丢 → 比黑屏更糟。
    // 因此按 vkGetPhysicalDeviceFormatProperties 的 optimalTilingFeatures 逐位裁剪。
    // COLOR_ATTACHMENT_BIT 同理（BC7 之类不可作为颜色附件）。
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(b->physicalDevice, fmt, &fp);
    const VkFormatFeatureFlags feats = fp.optimalTilingFeatures;

    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (feats & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
        ici.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (feats & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
        ici.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (feats & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
        ici.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (feats & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
        ici.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    // FIX (P0-3): feats == 0 的含义被搞反了。
    //
    // 原注释写的是「驱动没报告任何 feature（不该发生）」，于是兜底强行加上
    // SAMPLED | COLOR_ATTACHMENT。但 optimalTilingFeatures == 0 在 Vulkan 里
    // 有明确语义：**该格式在 optimal tiling 下完全不被支持**。这恰恰是 iOS 上
    // BC1/BC2/BC3 压缩格式的正常返回值（Apple GPU 只支持 ASTC/ETC/PVRTC）。
    //
    // 把「不支持」当成「信息缺失」并强行加 COLOR_ATTACHMENT_BIT，会让
    // vkCreateImage 直接失败（VUID-VkImageCreateInfo-usage）→ 纹理创建返回
    // VK_NULL_HANDLE → 后续采样拿到空句柄。比不加 bit 糟糕得多。
    //
    // 正确做法：feats == 0 时只保留 TRANSFER 位（传输对任何格式都合法），
    // 让 vkCreateImage 有机会成功；能不能采样由上面的 resolve_supported_format
    // 决定 —— 走到这里说明它已经判定过该格式可用，或已回退成 RGBA8。
    if (feats == 0) {
        static std::unordered_set<uint32_t> warnedZeroFeat;
        if (warnedZeroFeat.insert((uint32_t)fmt).second) {
            MITHRIL_LOG_WARN("vk", "VkFormat %d 的 optimalTilingFeatures 为 0"
                             "（该格式在本设备不受支持，iOS 上 BC 压缩格式属正常情况）；"
                             "仅保留 TRANSFER usage", (int)fmt);
        }
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // FIX (device lost 预防 - vkCreateImage 预检查 GC-only):
    // vkCreateImage 本身在 MoltenVK 上可能触发 Metal GPU fault。所以调用前
    // 估算图像内存大小，如果会超过 95% 预算，先 GC。
    // 不 reject（避免红屏）：让 vkCreateImage + vkAllocateMemory 自己决定。
    if (b->totalVramBytes > 0) {
        // 粗略估算每像素字节数
        int estBpp = 4;  // 大多数格式 4 字节
        if (fmt == VK_FORMAT_R8_UNORM || fmt == VK_FORMAT_R8_SNORM ||
            fmt == VK_FORMAT_R8_UINT || fmt == VK_FORMAT_R8_SINT ||
            fmt == VK_FORMAT_S8_UINT) estBpp = 1;
        else if (fmt == VK_FORMAT_R8G8_UNORM || fmt == VK_FORMAT_R16_UNORM ||
                 fmt == VK_FORMAT_R16_SFLOAT || fmt == VK_FORMAT_D16_UNORM) estBpp = 2;
        else if (fmt == VK_FORMAT_R8G8B8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_UNORM ||
                 fmt == VK_FORMAT_R32_SFLOAT || fmt == VK_FORMAT_D32_SFLOAT ||
                 fmt == VK_FORMAT_R16G16B16A16_SFLOAT) estBpp = 4;
        else if (fmt == VK_FORMAT_R16G16B16A16_SFLOAT) estBpp = 8;
        else if (fmt == VK_FORMAT_R32G32B32A32_SFLOAT) estBpp = 16;

        // mip 链总大小约 = base_size * 4/3
        VkDeviceSize estSize = (VkDeviceSize)width * height * estBpp;
        if (ici.arrayLayers > 0) estSize *= ici.arrayLayers;
        if (ici.mipLevels > 1) estSize = (estSize * 4) / 3;  // mip 链

        VkDeviceSize gcThreshold = (b->totalVramBytes * 95) / 100;
        if (b->currentVramBytes + estSize > gcThreshold) {
            // 接近上限，先轻量 GC（只 drain disposal，不清 pipeline）
            static int imgGcCount = 0;
            imgGcCount++;
            if (imgGcCount <= 5 || imgGcCount % 50 == 0) {
                MITHRIL_LOG_WARN("vk", "pre-vkCreateImage GC: %dx%d fmt=%d "
                                  "est %llu MB + current %llu MB > 95%% threshold "
                                  "%llu MB — draining disposal (attempt #%d)",
                                  width, height, (int)fmt,
                                  (unsigned long long)(estSize / (1024*1024)),
                                  (unsigned long long)(b->currentVramBytes / (1024*1024)),
                                  (unsigned long long)(gcThreshold / (1024*1024)),
                                  imgGcCount);
            }
            mithril::vk::safe_device_wait_idle();
            // FIX (GPU page fault UAF): safe_device_wait_idle() submits the
            // CURRENT slot's buffer WITHOUT advancing currentFrame and re-begins
            // the SAME slot; this frame's descriptor sets still reference the
            // current slot's deferred-destroyed views (memo not invalidated).
            // drain_all_disposal_queues() here frees those views -> UAF on the
            // next vkQueueSubmit (kIOGPUCommandBufferCallbackErrorPageFault).
            // This fires in the texture re-spec / world-load path (pre-vkCreateImage
            // GC), exactly when Minecraft creates the block/entity atlases. Drain
            // all slots EXCEPT the current one; the current slot's queue is left
            // for the normal fence-wait drain after the frame commits and recycles.
            mithril::vk::drain_disposal_queues_except(b->currentFrame);
            // 不清 pipeline/descriptor — 留到 OOM fallback（try_allocate_memory_with_gc）
        }
    }

    if (vkCreateImage(b->device, &ici, nullptr, &e.image) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(b->device, e.image, &req);
    uint32_t mt = mithril::vk::find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == 0xFFFFFFFFu) mt = mithril::vk::find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    // FIX (OOM 主动 GC): 使用带 GC 的分配函数，OOM 时先排空延迟队列重试
    if (mithril::vk::try_allocate_memory_with_gc(b->device, &ai, nullptr, &e.memory) != VK_SUCCESS) {
        vkDestroyImage(b->device, e.image, nullptr);
        return VK_NULL_HANDLE;
    }
    // FIX (P1): 递增分配计数器（仅诊断）+ 字节级 VRAM 跟踪（MobileGL-style）
    b->currentAllocationCount++;
    b->currentVramBytes += req.size;
    e.memorySize = req.size;  // 记录大小，defer_destroy 时递减 currentVramBytes
    vkBindImageMemory(b->device, e.image, e.memory, 0);

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = e.image;
    vci.viewType = (target == GL_TEXTURE_3D) ? VK_IMAGE_VIEW_TYPE_3D :
                   (target == GL_TEXTURE_CUBE_MAP ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D);
    vci.format = fmt;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    if (fmt == VK_FORMAT_D16_UNORM || fmt == VK_FORMAT_D32_SFLOAT)
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    else if (fmt == VK_FORMAT_D24_UNORM_S8_UINT || fmt == VK_FORMAT_D32_SFLOAT_S8_UINT)
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    else if (fmt == VK_FORMAT_S8_UINT)
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = e.levels;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = ici.arrayLayers;
    vci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
    // FIX (GPU page fault root cause): the view creation return value was not
    // checked. On failure e.view stayed VK_NULL_HANDLE but the entry was still
    // stored in texture_table; a later descriptor bind would then feed that NULL
    // view into a descriptor, and MoltenVK would sample a garbage address on the
    // GPU → kIOGPUCommandBufferCallbackErrorPageFault. Under the VRAM spike at
    // world-load (atlas creation) view creation can genuinely fail. On failure we
    // now free the image+memory and do NOT store a broken entry; the GL layer's
    // next glTexImage2D retry (or the default-texture fallback) covers the bind.
    if (vkCreateImageView(b->device, &vci, nullptr, &e.view) != VK_SUCCESS) {
        static int viewFailCount = 0;
        viewFailCount++;
        if (viewFailCount <= 5 || viewFailCount % 100 == 0) {
            MITHRIL_LOG_WARN("vk", "vkCreateImageView failed (tex %u, %dx%d "
                              "fmt=%d) — freeing image/memory, entry not stored "
                              "(fail #%d)",
                              name, width, height, (int)fmt, viewFailCount);
        }
        vkDestroyImage(b->device, e.image, nullptr);
        if (e.memory != VK_NULL_HANDLE) {
            vkFreeMemory(b->device, e.memory, nullptr);
            if (b->currentAllocationCount > 0) b->currentAllocationCount--;
            if (e.memorySize > 0 && b->currentVramBytes >= e.memorySize) {
                b->currentVramBytes -= e.memorySize;
            } else if (e.memorySize > 0) {
                b->currentVramBytes = 0;
            }
        }
        return VK_NULL_HANDLE;
    }

    tbl[name] = e;
    return e.image;
}

void backend_texture_upload(GLuint name, int level, int x, int y, int z,
                            int w, int h, int d, GLenum format, GLenum type,
                            const void* pixels, const MGUnpackParams* unpack,
                            int is_full_upload) {
    auto& tbl = mithril::vk::texture_table();
    auto it = tbl.find(name);
    if (it == tbl.end() || !pixels) return;
    const int unpack_alignment = (unpack && unpack->unpackAlignment > 0)
                                     ? unpack->unpackAlignment : 4;
    mithril::vk::stage_and_copy_image(it->second, level, x, y, z, w, h, d,
                                      pixels, unpack_alignment, format, type,
                                      is_full_upload != 0);
}

/* Compressed texture upload. Compressed data is copied verbatim — no pixel
 * unpack/RGB-expand. The VkBufferImageCopy uses bufferRowLength=0 (tightly
 * packed) and imageExtent = (w,h,d). For block-compressed formats Vulkan
 * interprets the buffer as a sequence of compressed blocks, so a direct
 * memcpy is correct. */
void backend_texture_upload_compressed(GLuint name, int level, int x, int y, int z,
                                       int w, int h, int d,
                                       GLenum internalFormat,
                                       GLsizei dataLen, const void* pixels,
                                       int is_full_upload) {
    using namespace mithril::vk;
    auto& tbl = texture_table();
    auto it = tbl.find(name);
    if (it == tbl.end() || !pixels || dataLen <= 0) return;
    TextureEntry& tex = it->second;
    Backend* b = backend();
    if (!b->commandBuffer) return;
    if (!ensure_command_buffer_recording()) return;

    // Transition image to TRANSFER_DST_OPTIMAL if needed.
    if (tex.currentLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        transition_image_layout(tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }

    // Allocate staging buffer and copy compressed data verbatim.
    // Uses the same per-frame transient staging arena as stage_and_copy_image
    // to avoid per-texture vkCreateBuffer/vkAllocateMemory.
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    void* stagingMapped = nullptr;
    VkDeviceSize allocSize = dataLen;
    if (b->frameStagingReady) {
        VkDeviceSize alignedOffset = (b->frameStagingOffset[b->currentFrame] + 255) & ~255;
        if (alignedOffset + allocSize <= Backend::kFrameStagingSize) {
            stagingBuffer = b->frameStagingBuffer[b->currentFrame];
            stagingOffset = alignedOffset;
            stagingMapped = b->frameStagingMapped[b->currentFrame];
            b->frameStagingOffset[b->currentFrame] = alignedOffset + allocSize;
        }
    }
    if (stagingBuffer == VK_NULL_HANDLE) {
        // Fallback: temporary staging buffer with deferred destroy.
        // Reuses the same pattern as stage_and_copy_image's overflow path.
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = allocSize;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(b->device, &bci, nullptr, &stagingBuffer) != VK_SUCCESS) return;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(b->device, stagingBuffer, &mr);
        // Reuse the existing find_memory_type helper (queries b->memProps).
        uint32_t memType = find_memory_type(mr.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (memType == 0xFFFFFFFFu) {
            vkDestroyBuffer(b->device, stagingBuffer, nullptr);
            return;
        }
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memType;
        VkDeviceMemory tmpMem = VK_NULL_HANDLE;
        // FIX (OOM 主动 GC): use GC-aware allocator consistent with create_buffer.
        if (try_allocate_memory_with_gc(b->device, &mai, nullptr, &tmpMem) != VK_SUCCESS) {
            vkDestroyBuffer(b->device, stagingBuffer, nullptr);
            return;
        }
        vkBindBufferMemory(b->device, stagingBuffer, tmpMem, 0);
        vkMapMemory(b->device, tmpMem, 0, allocSize, 0, &stagingMapped);
        stagingOffset = 0;
        // FIX (MobileGL-style): track VRAM bytes for this temporary allocation
        b->currentAllocationCount++;
        b->currentVramBytes += mr.size;
        // Defer destroy until GPU finishes (same struct as stage_and_copy_image).
        DeferredDestroy ds{};
        ds.buffer = stagingBuffer;
        ds.memory = tmpMem;
        ds.memorySize = mr.size;  // track bytes for VRAM monitoring
        b->disposalQueue[b->currentFrame].push_back(ds);
    }
    std::memcpy(static_cast<uint8_t*>(stagingMapped) + stagingOffset, pixels, dataLen);

    VkBufferImageCopy region{};
    region.bufferOffset = stagingOffset;
    region.bufferRowLength = 0;  // tightly packed
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = level;
    // FIX (cubemap face upload, same as stage_and_copy_image): face index is
    // carried in the z parameter; for a cubemap map it to baseArrayLayer
    // (array layer == face) instead of treating z as a 3D image offset.
    if (tex.target == GL_TEXTURE_CUBE_MAP && z >= 0 && z < 6) {
        region.imageSubresource.baseArrayLayer = (uint32_t)z;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {x, y, 0};
    } else {
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {x, y, z};
    }
    region.imageExtent = {(uint32_t)w, (uint32_t)h, (uint32_t)d};

    vkCmdCopyBufferToImage(b->commandBuffer, stagingBuffer, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to SHADER_READ_ONLY if this was a full upload (typical for
    // glCompressedTexImage2D). Partial uploads (SubImage) leave it in
    // TRANSFER_DST_OPTIMAL so subsequent uploads don't re-transition.
    if (is_full_upload) {
        transition_image_layout(tex, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

void backend_texture_set_params(GLuint name, GLint min_filter, GLint mag_filter,
                                GLint wrap_s, GLint wrap_t, GLint wrap_r,
                                const float* border_color) {
    // Vulkan samplers are immutable; params are applied when the sampler is
    // fetched via backend_get_or_create_sampler (which caches per (name,param)).
    // Record nothing here — the sampler table is keyed by name and rebuilt on
    // demand. (See backend_get_or_create_sampler.)
    (void)name; (void)min_filter; (void)mag_filter;
    (void)wrap_s; (void)wrap_t; (void)wrap_r; (void)border_color;
}

VkImageView backend_get_texture_view(GLuint name) {
    auto& tbl = mithril::vk::texture_table();
    auto it = tbl.find(name);
    if (it == tbl.end()) return VK_NULL_HANDLE;
    // FIX (GPU page fault): 若纹理 image 已被销毁/重建成 NULL，但 entry 里还
    // 残留一个非 NULL 的 view（重建竞态：新 image 尚未创建或失败），绝不能把
    // 这个 stale view 交给 descriptor —— GPU 采样它会 kIOGPUCommandBufferCallback
    // ErrorPageFault。只有当 image 和 view 都有效时才认为该 view 可采样。
    if (it->second.image == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    return it->second.view;
}

VkImage backend_get_texture_image(GLuint name) {
    auto& tbl = mithril::vk::texture_table();
    auto it = tbl.find(name);
    return it == tbl.end() ? VK_NULL_HANDLE : it->second.image;
}

void backend_delete_texture(GLuint name) {
    auto& tbl = mithril::vk::texture_table();
    auto it = tbl.find(name);
    if (it == tbl.end()) return;
    mithril::vk::defer_destroy_texture_entry(it->second);
    tbl.erase(it);
}

void backend_invalidate_sampler_cache(GLuint name) {
    auto& tbl = mithril::vk::sampler_table();
    auto it = tbl.find(name);
    if (it == tbl.end()) return;
    mithril::vk::defer_destroy_sampler_entry(it->second);
    tbl.erase(it);
}

void backend_transition_texture_layout(GLuint name, VkImageLayout target_layout) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->initialized) return;
    auto& tbl = mithril::vk::texture_table();
    auto it = tbl.find(name);
    if (it == tbl.end()) return;
    mithril::vk::transition_image_layout(it->second, target_layout);
}

/* GL compare func -> VkCompareOp (local; avoids cross-TU linkage assumptions). */
static VkCompareOp mithril_gl_compare_to_vk(GLenum f) {
    switch (f) {
        case GL_NEVER:    return VK_COMPARE_OP_NEVER;
        case GL_LESS:     return VK_COMPARE_OP_LESS;
        case GL_EQUAL:    return VK_COMPARE_OP_EQUAL;
        case GL_LEQUAL:   return VK_COMPARE_OP_LESS_OR_EQUAL;
        case GL_GREATER:  return VK_COMPARE_OP_GREATER;
        case GL_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
        case GL_GEQUAL:   return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case GL_ALWAYS:   return VK_COMPARE_OP_ALWAYS;
        default:          return VK_COMPARE_OP_ALWAYS;
    }
}

VkSampler backend_get_or_create_sampler(GLuint name, GLint min_filter, GLint mag_filter,
                                        GLint wrap_s, GLint wrap_t, GLint wrap_r,
                                        const float* border_color,
                                        GLint compare_enable, GLint compare_op,
                                        GLfloat min_lod, GLfloat lod_bias) {
    mithril::vk::Backend* b = mithril::vk::backend();
    if (!b->initialized) return VK_NULL_HANDLE;

    // FIX (纯红 + GPU page fault): 采样器缓存必须按「纹理名 + 全部参数」，
    // 而非只按纹理名。旧实现把首个请求的参数永久冻结：同一纹理名稍后用
    // 不同的 filter/wrap 绑定时会拿到错误采样器 → mip/LOD 行为错误 →
    // A11/MoltenVK 采样越界 → 纯红 / GPU page fault。这里把 min/mag/wrap/
    // border 哈希进 key，同一纹理名下不同参数各持有一份采样器。
    bool borderWhite = false;
    if (border_color) {
        borderWhite = border_color[0] >= 1.0f && border_color[1] >= 1.0f &&
                      border_color[2] >= 1.0f && border_color[3] >= 1.0f;
    }
    // 参数哈希：仅纳入会影响 VkSamplerCreateInfo 的字段。
    uint64_t pkey = 0xcbf29ce484222325ull;  // FNV-1a 64 起点
    auto mix = [&pkey](uint64_t v) {
        pkey ^= v;
        pkey *= 0x100000001b3ull;
    };
    mix((uint64_t)(uint32_t)min_filter);
    mix((uint64_t)(uint32_t)mag_filter);
    mix((uint64_t)(uint32_t)wrap_s);
    mix((uint64_t)(uint32_t)wrap_t);
    mix((uint64_t)(uint32_t)wrap_r);
    mix(borderWhite ? 1ull : 0ull);
    // FIX (GPU page fault root cause — sampler maxLod stale across rebuild):
    // 采样器缓存 key 必须纳入该纹理当前实际拥有的 mip 层数。mipmap filter 的
    // maxLod 在采样器创建时被 clamp 到 tex.levels-1（见下）。若 levels 不参与
    // key，一旦纹理被 generate_mipmaps 重建 / glTexImage2D 重规范为不同的
    // mip 层数，旧采样器仍按旧 levels 缓存，新采样请求会命中过时的 maxLod：
    //   场景A: 先按 levels=1 缓存 maxLod=0，后重建为 11 层 → 永远只采 level0，
    //          图集糊成低清（不崩但质量崩）。
    //   场景B(致命): 先按 levels=11 缓存 maxLod=10，后重规范为 1 层，新 image/
    //          view 只有 1 个 mip，采样器却请求 level 1..10 → A11/MoltenVK 读
    //          未分配的 mip 层地址 → kIOGPUCommandBufferCallbackErrorPageFault，
    //          静默 GPU 崩溃（与线上「atlas 创建后第一帧纯红 + page fault」吻合）。
    // 修复：把 tex.levels 混入 pkey，levels 变化 → 自动 miss → 重建正确 maxLod
    // 的采样器；旧采样器留在 entry.byParams 中由 defer_destroy_sampler_entry 按
    // 需回收。这与 generate_mipmaps 重建后 tex.levels 更新严格同步，从根本上
    // 消除「采样器 maxLod 与 image view 层数不一致」这一整类越界崩溃。
    int keyLevels = 1;
    {
        auto& tex_tbl = mithril::vk::texture_table();
        auto tit = tex_tbl.find(name);
        if (tit != tex_tbl.end()) keyLevels = std::max(1, (int)tit->second.levels);
    }
    mix((uint64_t)(uint32_t)keyLevels);
    /* GL 3.3 sampler-object state must participate in the cache key, otherwise
     * a shadow (compare) sampler and a plain sampler on the same texture would
     * collide and one of them would sample with the wrong compare op. */
    mix((uint64_t)(uint32_t)compare_enable);
    mix(compare_enable ? (uint64_t)(uint32_t)compare_op : 0ull);
    mix((uint64_t)(uint32_t)(min_lod  * 1000.0f));
    mix((uint64_t)(uint32_t)(lod_bias * 1000.0f));

    auto& tbl = mithril::vk::sampler_table();
    mithril::vk::SamplerEntry& entry = tbl[name];
    entry.name = name;
    // 先查本参数缓存 + 当前 levels 的采样器；命中直接返回。按参数缓存保证同一
    // 纹理名的不同 filter/wrap 各持有一份采样器，互不干扰（旧实现只按纹理名缓存
    // 首个参数，正是纯红/GPU page fault 的根因）。
    auto pit = entry.byParams.find(pkey);
    if (pit != entry.byParams.end() && pit->second != VK_NULL_HANDLE) return pit->second;

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = mithril::vk::to_vk_filter(mag_filter);
    sci.minFilter = mithril::vk::to_vk_filter(min_filter);
    sci.mipmapMode = mithril::vk::to_vk_mipmap(min_filter);
    sci.addressModeU = mithril::vk::to_vk_wrap(wrap_s);
    sci.addressModeV = mithril::vk::to_vk_wrap(wrap_t);
    sci.addressModeW = mithril::vk::to_vk_wrap(wrap_r);
    sci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (border_color) {
        // Approximate border colour; only black/white matter for MoltenVK.
        sci.borderColor = borderWhite ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                                      : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
    sci.anisotropyEnable = VK_FALSE;
    sci.maxAnisotropy = 1.0f;
    /* GL 3.3 core: sampler objects carry TEXTURE_COMPARE_MODE / TEXTURE_COMPARE_FUNC
     * (shadow samplers) and TEXTURE_MIN_LOD / TEXTURE_LOD_BIAS. Previously these
     * were hardcoded off, so every samplerCompare / LOD bias the host set was
     * silently dropped. */
    sci.compareEnable = compare_enable ? VK_TRUE : VK_FALSE;
    sci.compareOp = compare_enable ? mithril_gl_compare_to_vk((GLenum)compare_op)
                                   : VK_COMPARE_OP_ALWAYS;
    // FIX (纯红 + GPU page fault): 非 mipmap 的 min filter（GL_NEAREST / GL_LINEAR）
    // 只采样 base level。旧实现无条件 minLod=0 / maxLod=12 + mipmapMode=LINEAR，
    // 会让只有 1 层 mip 的纹理被请求跨 0..12 层采样 → A11/MoltenVK 采样读越界
    // → 纯红 / GPU page fault。非 mipmap filter 把 LOD 范围收束到 0，保证只采
    // 第 0 层，与 GL 语义一致。mipmap filter 仍放开到 12。
    const bool mipmapped =
        min_filter == GL_NEAREST_MIPMAP_NEAREST || min_filter == GL_NEAREST_MIPMAP_LINEAR ||
        min_filter == GL_LINEAR_MIPMAP_NEAREST || min_filter == GL_LINEAR_MIPMAP_LINEAR;
    // 读取该纹理当前实际拥有的 mip 层数（glTexImage2D 只传 level0 时=1，
    // glGenerateMipmap 重建后才到完整层数）。
    int actualLevels = 1;
    {
        auto& tex_tbl = mithril::vk::texture_table();
        auto tit = tex_tbl.find(name);
        if (tit != tex_tbl.end()) actualLevels = tit->second.levels;
    }
    sci.mipLodBias = lod_bias;
    if (!mipmapped) {
        sci.minLod = 0.0f;
        sci.maxLod = 0.0f;
    } else {
        sci.minLod = min_lod;
        // FIX (加载界面即纯红 + GPU page fault 根因 - 单层 view + LINEAR mipmap):
        // 深度对照 MobileGL VkSamplerManager (ResolveSingleLevelMaxLod + BuildSamplerKey
        // 注释 :177-185)：当纹理当前只有 1 层 mip 时，若采样器仍用
        // VK_SAMPLER_MIPMAP_MODE_LINEAR，纹理单元在 A11/MoltenVK 上可能对单层 view
        // 发起 level+1 的取数，读进未初始化的越界页 → 静默 kIOGPUCommandBuffer
        // CallbackErrorPageFault。这正是「加载界面/Mojang logo 采 gui.png-atlas 时
        // 首帧即纯红 + page fault」的触发点：图集 base level 刚上传（tex.levels=1）
        // 就被加载界面用 GL_LINEAR_MIPMAP_LINEAR 采样。
        // 修复（与 MobileGL 完全一致）：actualLevels==1 时强制
        //   - mipmapMode = NEAREST（放弃对不存在的第 1 层的线性插值）
        //   - maxLod = 0.0f（只允许采第 0 层）
        // 保证单层 view 绝不请求 level+1。mip 链生成后 tex.levels 更新，keyLevels
        // 变化 → 重新取采样器得到正确的 LINEAR + 完整 maxLod。
        if (actualLevels <= 1) {
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.maxLod = 0.0f;
        } else {
            sci.maxLod = (float)std::max(0, actualLevels - 1);
        }
    }
    sci.unnormalizedCoordinates = VK_FALSE;

    VkSampler s = VK_NULL_HANDLE;
    if (vkCreateSampler(b->device, &sci, nullptr, &s) != VK_SUCCESS) return VK_NULL_HANDLE;
    entry.byParams[pkey] = s;
    return s;
}

VkFormat backend_vk_format_for_gl(GLenum internal_format) {
    return mithril::vk::gl_internal_to_vk(internal_format);
}

} // extern "C"
