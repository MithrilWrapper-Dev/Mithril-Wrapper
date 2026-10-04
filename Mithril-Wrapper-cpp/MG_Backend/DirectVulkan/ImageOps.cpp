// Mithril-Wrapper - MG_Backend/DirectVulkan/ImageOps.cpp
// Image-level operations that need their own command-buffer submission:
//   * backend_generate_mipmaps  — vkCmdBlitImage cascade across mip levels
//   * backend_read_pixels       — vkCmdCopyImage -> host-visible staging -> fence
//   * backend_blit_texture      — vkCmdBlitImage between two user textures
//
// These differ from the staging upload path in Resources.cpp because they
// cannot ride on the per-frame command buffer:
//   - glGenerateMipmap may be called outside a render pass and must complete
//     before the texture is sampled; recording onto the active command buffer
//     would delay the blits until eglSwapBuffers commits.
//   - glReadPixels must synchronously return host pixels, so it submits its
//     own one-shot command buffer and waits on a fence.
//   - glBlitFramebuffer between user FBOs similarly needs the blit to land
//     before subsequent draws read the destination.
//
// Each path allocates a transient VkCommandBuffer from the backend's pool,
// records + submits + waits on a dedicated fence, then frees the buffer.
#include "Device.h"
#include "Swapchain.h"
#include "CommandStream.h"
#include "Pipeline.h"
#include "RenderPassCompat.h"  // traditional VkRenderPass fallback (no dynamic rendering)
#include "Resources.h"
#include "../Backend.h"
#include "../../MG_State/State.h"
#include "../../MG_Impl/Log.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mithril {
namespace vk {

// host_texel_bytes is declared in Resources.h and defined in Resources.cpp.
// It is shared by the staging upload path and the readback path below.

namespace {

// One-shot command buffer + fence helper: records into a freshly allocated
// primary command buffer, submits it, waits for completion, and frees the
// buffer. The lambda returns false to abort the submit (e.g. recording error).
struct OneShotCtx {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool ok = false;
    // Optional binary semaphore this one-shot must wait before touching the
    // current swapchain drawable (used by a blit-to-default, which is recorded
    // out-of-band rather than through commit_frame's normal acquire wait).
    VkSemaphore waitSemaphore = VK_NULL_HANDLE;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    bool consumeAcquire = false;
};

bool begin_one_shot(OneShotCtx& c) {
    Backend* b = backend();
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = b->commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(b->device, &ai, &c.cmd) != VK_SUCCESS) return false;

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(b->device, &fi, nullptr, &c.fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(b->device, b->commandPool, 1, &c.cmd);
        c.cmd = VK_NULL_HANDLE;
        return false;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(c.cmd, &bi) != VK_SUCCESS) return false;
    c.ok = true;
    return true;
}

void end_one_shot(OneShotCtx& c) {
    Backend* b = backend();
    if (!c.ok) {
        if (c.fence) { vkDestroyFence(b->device, c.fence, nullptr); c.fence = VK_NULL_HANDLE; }
        if (c.cmd)   { vkFreeCommandBuffers(b->device, b->commandPool, 1, &c.cmd); c.cmd = VK_NULL_HANDLE; }
        return;
    }
    vkEndCommandBuffer(c.cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cmd;
    if (c.waitSemaphore != VK_NULL_HANDLE) {
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &c.waitSemaphore;
        si.pWaitDstStageMask = &c.waitStage;
    }
    VkResult submitRc = vkQueueSubmit(b->graphicsQueue, 1, &si, c.fence);
    if (submitRc == VK_SUCCESS && c.consumeAcquire) {
        Swapchain* sc0 = active_swapchain();
        if (sc0) sc0->imageAvailableConsumed = true;
    }
    vkWaitForFences(b->device, 1, &c.fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(b->device, c.fence, nullptr);
    vkFreeCommandBuffers(b->device, b->commandPool, 1, &c.cmd);
    c.cmd = VK_NULL_HANDLE;
    c.fence = VK_NULL_HANDLE;
}

// aspect_for_format is declared in Resources.h and defined in Resources.cpp;
// we share the single canonical implementation across the backend.

} // namespace

void generate_mipmaps(GLuint name) {
    Backend* b = backend();
    if (!b->initialized || name == 0) return;
    auto& tbl = mithril::vk::texture_table();
    auto it = tbl.find(name);
    if (it == tbl.end()) return;
    TextureEntry& tex = it->second;
    if (tex.image == VK_NULL_HANDLE) return;

    // 该纹理尺寸需要的完整 mip 链层数 (floor(log2(max(w,h)))+1)。
    int fullLevels = 1;
    {
        int m = tex.width > tex.height ? tex.width : tex.height;
        if (m < 1) m = 1;
        while (m > 1) { fullLevels++; m >>= 1; }
    }

    // FIX (真机主菜单 page fault 诊断 - 盲点: 用户在主菜单纯红状态下仍能点进世界,
    // 说明 fault 前的 draw 在持续, 但 GPU 在某次 atlas/字体/cubemap 采样时读无效地址):
    // 真机日志从未出现 REBUILD WARN —— 说明 atlas 走的是「非重建」分支
    // (tex.levels >= fullLevels, 即 MC 用 glTexStorage2D 分配了完整 mip 链)。
    // 该分支从未被 CI 覆盖 (render_smoke 的 mipmap 用例都用 2x2, tex.levels=1 走重建)。
    // 这里在入口打印每次 generateMipmap 调用 + 分支选择, 让真机日志一锤定音:
    //   走哪条分支 (rebuild vs cascade)、tex.levels vs fullLevels、当前布局。
    {
        static int genMipLog = 0;
        if (genMipLog <= 12 || genMipLog % 50 == 0) {
            const char* br = (tex.levels < fullLevels) ? "REBUILD" : "CASCADE";
            MITHRIL_LOG_WARN("vk", "generate_mipmaps tex=%u %dx%d "
                              "levels=%d fullLevels=%d curLayout=%d → %s",
                              name, tex.width, tex.height, (int)tex.levels,
                              fullLevels, (int)tex.currentLayout, br);
        }
        genMipLog++;
    }

    // FIX (Root Cause - 单层 image 被 mipmap 采样器越界采样 → page fault):
    // Minecraft 先 glTexImage2D(level=0) 上传 atlas 的 base level（Texture.cpp:145
    // 只把 t->levels 推到 level+1，故此时 tex.levels == 1，VkImage 只有 1 层 mip），
    // 再 glGenerateMipmap() 请求完整 mip 链。旧实现因 tex.levels<=1 直接 return，
    // VkImage 永远只有 1 层，而 blocks/terrain 等 atlas 的采样器是 mipmap filter
    // （minLod=0, maxLod 依 image 层数 clamp）。若 image 无 mip 数据，MoltenVK/Metal
    // 采样 level 1..11 会读未分配/未初始化的 mip 层地址 → kIOGPUCommandBuffer
    // CallbackErrorPageFault（GPU 执行期崩溃，完全静默，与线上「atlas 创建后第一帧
    // 渲染纯红 + page fault」吻合）。
    // 参照 MobileGL SyncTexture (VkTextureManager.cpp:1325)：检测到
    // mipLevels < ComputeFullMipLevelCount(extent) 时重建 image 为完整 mip 链。
    //
    // 实现：当当前 VkImage 层数不足完整链时，创建 fullLevels 层的新 image，在
    // one-shot 命令缓冲里把旧 image level 0 复制到新 image level 0，再对新 image
    // 逐级 vkCmdBlitImage 生成 mip。完成后替换 tex 的 image/view/memory，旧资源
    // 延迟释放（disposalQueue），并让本函数自包含地完成，不再走下方只针对已有多层
    // image 的 cascade。
    if (tex.levels < fullLevels) {
        const VkFormat fmt = tex.format;
        const VkImageAspectFlags aspect = aspect_for_format(fmt);
        const VkImageType imgType = (tex.target == GL_TEXTURE_3D) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        // Same 2D-array fix as Resources.cpp: the rebuild path must carry the
        // array's layer count through, or generate_mipmap rebuilds an N-layer
        // array as a 1-layer image and drops every layer but the first.
        const uint32_t arrayLayers = (imgType == VK_IMAGE_TYPE_3D) ? 1
                                   : (tex.target == GL_TEXTURE_CUBE_MAP ? 6
                                   : (tex.target == GL_TEXTURE_2D_ARRAY
                                      ? (uint32_t)(tex.depth > 0 ? tex.depth : 1) : 1));
        const bool isCube = (tex.target == GL_TEXTURE_CUBE_MAP);

        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(b->physicalDevice, fmt, &fp);
        const VkFormatFeatureFlags feats = fp.optimalTilingFeatures;

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = imgType;
        ici.format = fmt;
        ici.extent = { (uint32_t)tex.width, (uint32_t)tex.height,
                       (uint32_t)(imgType == VK_IMAGE_TYPE_3D ? tex.depth : 1) };
        ici.mipLevels = (uint32_t)fullLevels;
        ici.arrayLayers = arrayLayers;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (feats & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ici.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (feats & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) ici.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (feats & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ici.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (feats & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ici.usage |= VK_IMAGE_USAGE_STORAGE_BIT;

        VkImage newImage = VK_NULL_HANDLE;
        VkDeviceMemory newMem = VK_NULL_HANDLE;
        VkDeviceSize newMemSize = 0;
        if (vkCreateImage(b->device, &ici, nullptr, &newImage) == VK_SUCCESS) {
            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(b->device, newImage, &req);
            uint32_t mt = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (mt == 0xFFFFFFFFu)
                mt = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mt;
            if (try_allocate_memory_with_gc(b->device, &ai, nullptr, &newMem) == VK_SUCCESS) {
                vkBindImageMemory(b->device, newImage, newMem, 0);
                newMemSize = req.size;
            } else {
                vkDestroyImage(b->device, newImage, nullptr);
                newImage = VK_NULL_HANDLE;
            }
        }

        if (newImage != VK_NULL_HANDLE && newMem != VK_NULL_HANDLE) {
            // FIX (真机 page fault 诊断): 限流打印每次 mipmap 重建的纹理信息，
            // 便于在 MITHRIL_DEBUG 未开启（真机默认 Warning）时也能定位是哪张
            // atlas 触发 GPU address fault。
            {
                static int rebuildLogCount = 0;
                if (rebuildLogCount <= 5 || rebuildLogCount % 50 == 0) {
                    MITHRIL_LOG_WARN("vk", "generate_mipmaps REBUILD tex=%u %dx%d "
                                     "levels=%d->%d fmt=%d",
                                     name, tex.width, tex.height, (int)tex.levels,
                                     fullLevels, (int)fmt);
                }
                rebuildLogCount++;
            }
            // FIX (重建纹理采样黑屏 - CRITICAL cross-command-buffer 同步):
            // old image 的 level0 数据通常由 glTexImage2D 记录到「主 command
            // buffer」（backend_texture_upload → b->commandBuffer），并只在
            // draw / glFinish 时才提交。而下面重建分支用独立的 one-shot command
            // buffer 去 vkCmdCopyImage 读 old level0。若 one-shot 先于主 command
            // buffer 提交执行，old image 的 level0 在 GPU 上还【未初始化】，
            // copy 读到全 0 → 重建后的 mipmap 纹理 level0 即黑 → 采样黑屏。
            //   对照：render_smoke 判别 E（单级无 mipmap 纹理）直接 draw 后才
            // readPixels，upload 已被主 command buffer 提交，故采样白；判别
            // 4a 的 sampTex 在 texImage2D 后立即 generateMipmap，主 command
            // buffer 未提交，one-shot copy 读未初始化数据 → 采样黑。这正是
            // 根因。
            // 修复：在开始 one-shot copy 前先 safe_device_wait_idle() —— 它会
            // 提交并等待主 command buffer（含 pending upload）完成，再重新
            // begin 以便后续录制。这保证 old level0 数据 GPU 可见后 one-shot
            // 才执行，同 queue 顺序提交亦无竞态。
            safe_device_wait_idle();
            OneShotCtx c;
            bool rebuildOk = false;
            if (begin_one_shot(c)) {
                rebuildOk = true;
                const VkImageLayout oldLayout = tex.currentLayout;
                const VkPipelineStageFlags oldStage =
                    (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) ? VK_PIPELINE_STAGE_TRANSFER_BIT
                    : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                const VkAccessFlags oldAccess =
                    (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) ? VK_ACCESS_TRANSFER_WRITE_BIT
                    : VK_ACCESS_SHADER_READ_BIT;

                // old level0 -> TRANSFER_SRC_OPTIMAL
                VkImageMemoryBarrier oldSrc{};
                oldSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                oldSrc.srcAccessMask = oldAccess;
                oldSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                oldSrc.oldLayout = oldLayout;
                oldSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                oldSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                oldSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                oldSrc.image = tex.image;
                oldSrc.subresourceRange = { aspect, 0, 1, 0, arrayLayers };
                vkCmdPipelineBarrier(c.cmd, oldStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                     0, nullptr, 0, nullptr, 1, &oldSrc);

                // new level0 -> TRANSFER_DST_OPTIMAL
                VkImageMemoryBarrier newDst{};
                newDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                newDst.srcAccessMask = 0;
                newDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                newDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                newDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                newDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                newDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                newDst.image = newImage;
                newDst.subresourceRange = { aspect, 0, 1, 0, arrayLayers };
                vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                     0, nullptr, 0, nullptr, 1, &newDst);

                // copy old level0 -> new level0 (cubemap: arrayLayers=6, copy all layers at once)
                VkImageCopy copy{};
                copy.srcSubresource = { aspect, 0, 0, arrayLayers };
                copy.dstSubresource = { aspect, 0, 0, arrayLayers };
                copy.srcOffset = { 0, 0, 0 };
                copy.dstOffset = { 0, 0, 0 };
                copy.extent = { (uint32_t)tex.width, (uint32_t)tex.height,
                                (uint32_t)(imgType == VK_IMAGE_TYPE_3D ? tex.depth : 1) };
                vkCmdCopyImage(c.cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               newImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

                // new level0 -> TRANSFER_SRC_OPTIMAL for cascade
                VkImageMemoryBarrier n0{};
                n0.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                n0.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                n0.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                n0.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                n0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                n0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                n0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                n0.image = newImage;
                n0.subresourceRange = { aspect, 0, 1, 0, arrayLayers };
                vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                     0, nullptr, 0, nullptr, 1, &n0);

                // cascade: for each level L>=1, blit L-1 -> L, transition L to SRC.
                for (int L = 1; L < fullLevels; ++L) {
                    int32_t w = tex.width  >> L; if (w < 1) w = 1;
                    int32_t h = tex.height >> L; if (h < 1) h = 1;
                    VkImageMemoryBarrier toDst{};
                    toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    toDst.srcAccessMask = 0;
                    toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toDst.image = newImage;
                    toDst.subresourceRange = { aspect, (uint32_t)L, 1, 0, arrayLayers };
                    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                         0, nullptr, 0, nullptr, 1, &toDst);

                    VkImageBlit blit{};
                    blit.srcSubresource = { aspect, (uint32_t)(L - 1), 0, arrayLayers };
                    blit.dstSubresource = { aspect, (uint32_t)L, 0, arrayLayers };
                    blit.srcOffsets[0] = { 0, 0, 0 };
                    blit.srcOffsets[1] = { (tex.width >> (L - 1)) > 0 ? (tex.width >> (L - 1)) : 1,
                                           (tex.height >> (L - 1)) > 0 ? (tex.height >> (L - 1)) : 1, 1 };
                    blit.dstOffsets[0] = { 0, 0, 0 };
                    blit.dstOffsets[1] = { w, h, 1 };
                    VkFilter filt = (aspect == VK_IMAGE_ASPECT_COLOR_BIT) ? VK_FILTER_LINEAR
                                                                          : VK_FILTER_NEAREST;
                    vkCmdBlitImage(c.cmd, newImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   newImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, filt);

                    VkImageMemoryBarrier toSrc{};
                    toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    toSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    toSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toSrc.image = newImage;
                    toSrc.subresourceRange = { aspect, (uint32_t)L, 1, 0, arrayLayers };
                    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                         0, nullptr, 0, nullptr, 1, &toSrc);
                }

                // all levels -> SHADER_READ_ONLY_OPTIMAL
                VkImageMemoryBarrier toShader{};
                toShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                toShader.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                toShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toShader.image = newImage;
                toShader.subresourceRange = { aspect, 0, (uint32_t)fullLevels, 0, arrayLayers };
                // FIX (GPU page fault root cause - compute): 与下方非重建分支一致，
                // 同时覆盖 fragment + compute 采样阶段，确保 mip 写入对 compute
                // 着色器（Sodium/Iris 采样图集）可见。
                vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     0, nullptr, 0, nullptr, 1, &toShader);

                end_one_shot(c);
            } else {
                // FIX (真机 page fault): 若 one-shot 命令记录/提交失败，newImage
                // 从未被写入任何数据（也未 transition 布局）。旧代码仍无条件把
                // tex.image 替换成这个未初始化 image + 创建新 view → GPU 采样它
                // 读到无效/未初始化地址 → kIOGPUCommandBufferCallbackErrorPageFault。
                // 修复：失败时保留旧的（有效的单层）image，仅销毁 newImage，绝
                // 不把未初始化的资源接入采样链。
                if (newImage != VK_NULL_HANDLE) vkDestroyImage(b->device, newImage, nullptr);
                if (newMem != VK_NULL_HANDLE) vkFreeMemory(b->device, newMem, nullptr);
                newImage = VK_NULL_HANDLE;
                newMem = VK_NULL_HANDLE;
                rebuildOk = false;
                static int rebuildFailLog = 0;
                if (rebuildFailLog <= 3) {
                    MITHRIL_LOG_WARN("vk", "generate_mipmaps REBUILD FAILED tex=%u "
                                     "%dx%d — keeping old single-level image",
                                     name, tex.width, tex.height);
                }
                rebuildFailLog++;
            }

            if (rebuildOk) {
                // 重建新 view（fullLevels 层）。
                VkImageView newView = VK_NULL_HANDLE;
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = newImage;
                vci.viewType = (tex.target == GL_TEXTURE_3D) ? VK_IMAGE_VIEW_TYPE_3D
                             : (isCube ? VK_IMAGE_VIEW_TYPE_CUBE
                             : (tex.target == GL_TEXTURE_2D_ARRAY
                                ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D));
                vci.format = fmt;
                vci.subresourceRange.aspectMask = aspect;
                vci.subresourceRange.baseMipLevel = 0;
                vci.subresourceRange.levelCount = (uint32_t)fullLevels;
                vci.subresourceRange.baseArrayLayer = 0;
                vci.subresourceRange.layerCount = arrayLayers;
                vci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                   VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
                if (vkCreateImageView(b->device, &vci, nullptr, &newView) != VK_SUCCESS) {
                    newView = VK_NULL_HANDLE;
                }
                // 旧资源延迟释放（含 invalidate descriptor memo，防止 stale view 复用）。
                defer_destroy_texture_entry(tex);
                // 写入新资源。
                tex.image = newImage;
                tex.memory = newMem;
                tex.view  = newView;
                tex.levels = fullLevels;
                tex.memorySize = newMemSize;  // 记录新 image 的 VRAM 大小，供将来延迟释放时正确回收计数
                tex.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        } else {
            if (newImage != VK_NULL_HANDLE) vkDestroyImage(b->device, newImage, nullptr);
            if (newMem != VK_NULL_HANDLE) vkFreeMemory(b->device, newMem, nullptr);
        }
        if (tex.image == VK_NULL_HANDLE) return;
        // 重建分支已生成完整 mip 链，自包含完成。
        return;
    }

    if (tex.levels <= 1) return;

    // FIX (真机主菜单 page fault - 盲点根因 - CRITICAL cross-command-buffer 同步):
    // 这条「非重建」分支 (atlas 用 glTexStorage2D/逐级上传使 tex.levels==fullLevels
    // 时走这里) 与上方的重建分支有**完全同源**的同步缺陷，但此前只在重建分支
    // (line ~209) 加了 safe_device_wait_idle，这里漏了！
    //
    // 时序：MC 上传 atlas 用 glTexSubImage2D/glTexImage2D，这些 upload 记录到
    // 「主 command buffer」(stage_and_copy_image → b->commandBuffer)，只在 draw /
    // glFinish 时提交。随后 MC 立即 glGenerateMipmap 走本分支，用独立的 one-shot
    // command buffer 去 vkCmdBlitImage 读 old level0。若 one-shot 先于主 command
    // buffer 提交执行，level0 在 GPU 上还【未初始化】→ blit 生成的所有 mip 层都是
    // garbage → 采样器(mipmap filter)读取 level 1..N 的无效地址 →
    // kIOGPUCommandBufferCallbackErrorPageFault。
    //
    // 修复：与重建分支一致，在 begin_one_shot 前先 safe_device_wait_idle()——
    // 提交并等待主 command buffer (含 pending upload) 完成，再重新 begin。保证
    // level0 数据 GPU 可见后 one-shot 才执行，同 queue 顺序提交亦无竞态。
    safe_device_wait_idle();
    OneShotCtx c;
    if (!begin_one_shot(c)) {
        // FIX (日志刷屏): 限流 — deviceLost 时每张纹理的 mipmap 生成都会失败
        static int mipmapFailCount = 0;
        mipmapFailCount++;
        if (mipmapFailCount <= 3 || mipmapFailCount % 100 == 0) {
            MITHRIL_LOG_WARN("vk", "generate_mipmaps: failed to begin one-shot "
                              "cmd (fail #%d)", mipmapFailCount);
        }
        return;
    }

    const VkImageAspectFlags aspect = aspect_for_format(tex.format);

    // FIX (Main-menu panorama cubemap GPU fault root cause - CASCADE layer
    // coverage): the REBUILD branch allocates arrayLayers=6 for cubemaps, but
    // this CASCADE branch (tex.levels >= fullLevels, which Minecraft's
    // atlas/panorama paths take) issued every barrier/blit with layerCount=1,
    // so the mip chain was only generated for layer 0 and layers 1-5 stayed
    // uninitialized -> sampling an uninitialized mip layer of the cubemap ->
    // MoltenVK/A11 GPU Address Fault. Cover all layers (Vulkan allows a single
    // barrier/blit to address the whole array).
    const uint32_t blitLayers = (tex.target == GL_TEXTURE_CUBE_MAP) ? 6u : 1u;

    // Transition level 0 from its current layout (likely SHADER_READ_ONLY or
    // TRANSFER_DST after the most recent upload) to TRANSFER_SRC_OPTIMAL.
    //
    // FIX (GPU page fault / 图集重上传后花屏 - CRITICAL):
    // 旧代码把 oldLayout 硬编码为 VK_IMAGE_LAYOUT_UNDEFINED。按 Vulkan 规范，
    // 从 UNDEFINED 过渡意味着该 subresource 的**内容**变为 undefined（不只是
    // 布局）。Minecraft 的图集在 glGenerateMipmap 时，若纹理已有完整 mip 链
    // （上一轮 glTexImage2D 已把 tex.levels 推到 fullLevels），走的是这条非重建
    // 分支：这里把 level 0 用 UNDEFINED 过渡到 TRANSFER_SRC，会**丢弃刚上传的
    // base level 数据**，随后 vkCmdBlitImage 以 level 0 为源生成 level 1..N →
    // 所有 mip 层都采样到 garbage/undefined 内存 → MoltenVK GPU address fault
    // (kIOGPUCommandBufferCallbackErrorPageFault) 或花屏。
    // 修复：与重建分支（~182 行）一致，用 tex.currentLayout 作为 oldLayout 保留
    // level 0 内容，并按当前布局推导正确的源阶段与访问位——既要让此前 TRANSFER
    // 写入（上传）可见，也要等待此前 fragment/compute 着色器对该图集的采样完成，
    // 避免布局转换/覆盖与在途采样竞争。内联实现（src_stage_for_layout 等 helper
    // 位于 Resources.cpp 的 static 作用域，本文件不可见）。
    const VkImageLayout lvl0Layout = tex.currentLayout;
    VkAccessFlags lvl0SrcAccess = 0;
    VkPipelineStageFlags lvl0SrcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    if (lvl0Layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        lvl0SrcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
        lvl0SrcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (lvl0Layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        lvl0SrcAccess = VK_ACCESS_TRANSFER_READ_BIT;
        lvl0SrcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (lvl0Layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
               lvl0Layout == VK_IMAGE_LAYOUT_GENERAL) {
        // fragment + compute 都采样图集/光照贴图，源阶段需同时覆盖两者。
        lvl0SrcAccess = VK_ACCESS_SHADER_READ_BIT;
        lvl0SrcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (lvl0Layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        lvl0SrcAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        lvl0SrcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    VkImageMemoryBarrier b0{};
    b0.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b0.srcAccessMask = lvl0SrcAccess;
    b0.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b0.oldLayout = lvl0Layout;
    b0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.image = tex.image;
    b0.subresourceRange.aspectMask = aspect;
    b0.subresourceRange.baseMipLevel = 0;
    b0.subresourceRange.levelCount = 1;
    b0.subresourceRange.baseArrayLayer = 0;
    b0.subresourceRange.layerCount = blitLayers;
    vkCmdPipelineBarrier(c.cmd, lvl0SrcStage,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b0);

    // For each level L >= 1: blit from L-1 (TRANSFER_SRC_OPTIMAL) into L
    // (TRANSFER_DST_OPTIMAL), then transition L to SHADER_READ_ONLY_OPTIMAL.
    for (int L = 1; L < tex.levels; ++L) {
        int32_t w = tex.width  >> L; if (w < 1) w = 1;
        int32_t h = tex.height >> L; if (h < 1) h = 1;

        // Transition level L (currently UNDEFINED) to TRANSFER_DST_OPTIMAL.
        VkImageMemoryBarrier toDst{};
        toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toDst.srcAccessMask = 0;
        toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image = tex.image;
        toDst.subresourceRange.aspectMask = aspect;
        toDst.subresourceRange.baseMipLevel = (uint32_t)L;
        toDst.subresourceRange.levelCount = 1;
        toDst.subresourceRange.baseArrayLayer = 0;
        toDst.subresourceRange.layerCount = blitLayers;
        vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toDst);

        // Blit from level L-1 (TRANSFER_SRC_OPTIMAL) to level L.
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = aspect;
        blit.srcSubresource.mipLevel = (uint32_t)(L - 1);
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = blitLayers;
        blit.srcOffsets[0] = { 0, 0, 0 };
        blit.srcOffsets[1] = { tex.width >> (L - 1) > 0 ? (tex.width >> (L - 1)) : 1,
                               tex.height >> (L - 1) > 0 ? (tex.height >> (L - 1)) : 1, 1 };
        blit.dstSubresource.aspectMask = aspect;
        blit.dstSubresource.mipLevel = (uint32_t)L;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = blitLayers;
        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { w, h, 1 };

        VkFilter filt = (aspect == VK_IMAGE_ASPECT_COLOR_BIT) ? VK_FILTER_LINEAR
                                                              : VK_FILTER_NEAREST;
        vkCmdBlitImage(c.cmd,
                       tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, filt);

        // Transition level L to TRANSFER_SRC_OPTIMAL for the next iteration
        // (or to SHADER_READ_ONLY_OPTIMAL on the last level — handled below).
        VkImageMemoryBarrier toSrc{};
        toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image = tex.image;
        toSrc.subresourceRange.aspectMask = aspect;
        toSrc.subresourceRange.baseMipLevel = (uint32_t)L;
        toSrc.subresourceRange.levelCount = 1;
        toSrc.subresourceRange.baseArrayLayer = 0;
        toSrc.subresourceRange.layerCount = blitLayers;
        vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toSrc);
    }

    // After the cascade, transition ALL levels to SHADER_READ_ONLY_OPTIMAL so
    // the texture can be sampled. (Levels 0..N-1 are all TRANSFER_SRC_OPTIMAL
    // at this point, which is not a sampling layout.)
    VkImageMemoryBarrier toShader{};
    toShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toShader.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShader.image = tex.image;
    toShader.subresourceRange.aspectMask = aspect;
    toShader.subresourceRange.baseMipLevel = 0;
    toShader.subresourceRange.levelCount = (uint32_t)tex.levels;
    toShader.subresourceRange.baseArrayLayer = 0;
    toShader.subresourceRange.layerCount = blitLayers;
    // FIX (GPU page fault root cause - compute): Sodium/Iris 会在 compute
    // 着色器里采样图集/光照贴图。dstStage 若只含 FRAGMENT_SHADER_BIT，mipmap
    // 生成的写入对 compute 采样不可见 → 采样到未初始化 mip 层 → GPU address fault。
    // 与 src_stage_for_layout / stage_and_copy_image 的 compute 修复保持一致，
    // 同时覆盖 fragment + compute。
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &toShader);

    end_one_shot(c);

    // All mip levels are now in SHADER_READ_ONLY_OPTIMAL. Update the tracked
    // layout so subsequent operations (uploads, blits, attachment uses) emit
    // the correct oldLayout in their memory barriers.
    tex.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

int read_pixels(int x, int y, int w, int h, GLenum format, GLenum type, void* out_pixels) {
    Backend* b = backend();
    if (!b->initialized || !out_pixels || w <= 0 || h <= 0 || !g_state) return 0;

    const GLuint readFboName = g_state->currentReadFBO;

    // Device-proven Minecraft 26.2 compatibility: depth readback must use the
    // depth aspect and preserve the image's tracked layout. Treating a depth
    // attachment as color either returns zeros or drives MoltenVK into an
    // invalid copy/barrier path.
    if (format == GL_DEPTH_COMPONENT) {
        if (type != GL_FLOAT) return 0;

        VkImage srcImage = VK_NULL_HANDLE;
        VkFormat srcFmt = VK_FORMAT_UNDEFINED;
        VkImageLayout srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        GLuint srcTexId = 0;

        if (readFboName == 0) {
            srcImage = g_state->eglDefaultDepthImage;
            srcFmt = g_state->eglDefaultDepthFormat;
            srcLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        } else {
            mithril::Framebuffer* fbo = mithril::state_get_framebuffer(readFboName);
            if (!fbo || fbo->depth.texture == 0) return 0;
            srcTexId = fbo->depth.texture;
            auto& tbl = mithril::vk::texture_table();
            auto it = tbl.find(srcTexId);
            if (it == tbl.end() || it->second.image == VK_NULL_HANDLE) return 0;
            srcImage = it->second.image;
            srcFmt = it->second.format;
            srcLayout = it->second.currentLayout;
        }
        if (srcImage == VK_NULL_HANDLE || srcFmt == VK_FORMAT_UNDEFINED) return 0;

        backend_end_render_pass();
        backend_commit();

        if (readFboName != 0 && srcTexId != 0) {
            auto& tbl = mithril::vk::texture_table();
            auto it = tbl.find(srcTexId);
            if (it != tbl.end()) srcLayout = it->second.currentLayout;
        }
        if (srcLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            srcLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        int srcBpp = 0;
        switch (srcFmt) {
            case VK_FORMAT_D16_UNORM:          srcBpp = 2; break;
            case VK_FORMAT_D32_SFLOAT:         srcBpp = 4; break;
            case VK_FORMAT_D24_UNORM_S8_UINT:  srcBpp = 4; break;
            case VK_FORMAT_D32_SFLOAT_S8_UINT: srcBpp = 4; break;
            default: return 0;
        }

        const VkDeviceSize stagingSize =
            (VkDeviceSize)w * (VkDeviceSize)h * (VkDeviceSize)srcBpp;
        BufferEntry staging{};
        if (!create_buffer(staging, stagingSize,
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT, nullptr)) {
            return 0;
        }

        OneShotCtx oneShot;
        if (!begin_one_shot(oneShot)) {
            destroy_buffer_entry(staging);
            return 0;
        }

        VkImageMemoryBarrier bar{};
        bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bar.oldLayout = srcLayout;
        bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = srcImage;
        bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        bar.subresourceRange.baseMipLevel = 0;
        bar.subresourceRange.levelCount = 1;
        bar.subresourceRange.baseArrayLayer = 0;
        bar.subresourceRange.layerCount = 1;
        if (srcLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
            srcLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
            bar.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        } else {
            bar.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        }
        bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        const VkPipelineStageFlags srcStage =
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        vkCmdPipelineBarrier(oneShot.cmd, srcStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &bar);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {x, y, 0};
        region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
        vkCmdCopyImageToBuffer(oneShot.cmd, srcImage,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.buffer, 1, &region);

        bar.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        bar.dstAccessMask =
            (srcLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
             srcLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                ? (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                : (VK_ACCESS_SHADER_READ_BIT |
                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
        bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        bar.newLayout = srcLayout;
        const VkPipelineStageFlags dstStage =
            (srcLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
             srcLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                ? (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
                : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        vkCmdPipelineBarrier(oneShot.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage, 0,
                             0, nullptr, 0, nullptr, 1, &bar);
        end_one_shot(oneShot);

        void* mapped = nullptr;
        if (vkMapMemory(b->device, staging.memory, 0, stagingSize, 0, &mapped) !=
                VK_SUCCESS ||
            !mapped) {
            destroy_buffer_entry(staging);
            return 0;
        }

        const size_t count = (size_t)w * (size_t)h;
        float* dst = static_cast<float*>(out_pixels);
        if (srcFmt == VK_FORMAT_D32_SFLOAT ||
            srcFmt == VK_FORMAT_D32_SFLOAT_S8_UINT) {
            std::memcpy(dst, mapped, count * sizeof(float));
        } else if (srcFmt == VK_FORMAT_D16_UNORM) {
            const uint16_t* src = static_cast<const uint16_t*>(mapped);
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (float)src[i] / 65535.0f;
            }
        } else {
            const uint32_t* src = static_cast<const uint32_t*>(mapped);
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (float)(src[i] & 0x00FFFFFFu) / 16777215.0f;
            }
        }
        vkUnmapMemory(b->device, staging.memory);
        destroy_buffer_entry(staging);
        return 1;
    }

    // Source: the current colour attachment installed on g_state. This covers
    // both FBO 0 (the EGL default framebuffer's swapchain image) and user
    // FBOs (their GL_COLOR_ATTACHMENT0 texture).
    VkImage src_image = VK_NULL_HANDLE;
    VkFormat src_fmt = VK_FORMAT_UNDEFINED;
    // The source image's CURRENT tracked layout. The barrier below MUST use
    // this as oldLayout, not a hardcoded guess. After any prior render pass on
    // the FBO, end_render_pass() transitioned the color attachment back to a
    // read-only layout (SHADER_READ_ONLY_OPTIMAL for color) and recorded it in
    // TextureEntry::currentLayout. Issuing a barrier whose oldLayout claims
    // COLOR_ATTACHMENT_OPTIMAL while the image is actually in a read-only
    // layout is a Vulkan spec violation: MoltenVK treats the transition as a
    // no-op, so the subsequent vkCmdCopyImageToBuffer reads the image in the
    // wrong layout and returns garbage (observed as all-zero/black pixels on
    // the offscreen render smoke).
    VkImageLayout src_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    // User-FBO colour tex_id (0 for the swapchain default framebuffer). Used
    // to update TextureEntry::currentLayout after the reverse barrier below.
    GLuint src_tex_id = 0;

    if (readFboName == 0) {
        // EGL default framebuffer: read directly from the swapchain image.
        // The EGL layer installs both the VkImageView and the underlying
        // VkImage + format on g_state when a surface is made current. The
        // swapchain image is in COLOR_ATTACHMENT_OPTIMAL after a render pass
        // (transitioned to PRESENT_SRC_KHR only at present).
        src_image = g_state->eglDefaultColorImage;
        src_fmt   = g_state->eglDefaultColorFormat;
        src_layout = backend_active_swapchain_color_layout();
    } else {
        mithril::Framebuffer* fbo = mithril::state_get_framebuffer(readFboName);
        if (!fbo || !fbo->colors[0].texture) {
            MITHRIL_LOG_WARN("vk", "read_pixels: read FBO %u has no colour "
                              "attachment (fbo=%p) - returning no data",
                              readFboName, (void*)fbo);
            return 0;
        }
        src_tex_id = fbo->colors[0].texture;
        src_image = backend_get_texture_image(src_tex_id);
        mithril::Texture* t = mithril::state_get_texture(src_tex_id);
        if (t) src_fmt = gl_internal_to_vk((GLenum)t->internalFormat);
    }
    if (src_image == VK_NULL_HANDLE) {
        MITHRIL_LOG_WARN("vk", "read_pixels: no source image for read FBO %u "
                          "(default swapchain image=%p) - returning no data",
                          readFboName,
                          (void*)g_state->eglDefaultColorImage);
        return 0;
    }

    if (std::getenv("MITHRIL_RPIMG")) {
        MITHRIL_LOG_WARN("vk-diag","RPIMG fbo=%u tex=%u image=%p fmt=%d layout=%d",
          readFboName,src_tex_id,(void*)src_image,(int)src_fmt,(int)src_layout);
    }
    // Flush any pending rendering into the colour attachment so the readback
    // sees the latest pixels.
    backend_end_render_pass();
    backend_commit();

    // Resolve the source image's layout AFTER the flush above. If a render pass
    // was active when read_pixels() was entered, backend_end_render_pass() just
    // transitioned the user-FBO color attachment back to a read-only layout
    // (SHADER_READ_ONLY_OPTIMAL for color) and recorded it in
    // TextureEntry::currentLayout. Reading it here guarantees the barrier below
    // uses oldLayout == the image's REAL current layout.
    //
    // Hardcoding COLOR_ATTACHMENT_OPTIMAL would be a Vulkan spec violation when
    // the image is in a read-only layout: MoltenVK treats the transition as a
    // no-op, so the subsequent vkCmdCopyImageToBuffer reads the image in the
    // wrong layout and returns garbage (observed as all-zero/black pixels on
    // the offscreen render smoke).
    if (readFboName == 0) {
        // The swapchain image's REAL current layout, resolved AFTER the flush
        // above. backend_commit() runs commit_frame(), which transitions the
        // swapchain colour image to PRESENT_SRC_KHR and records that in
        // Swapchain::currentColorLayout -- so by the time we get here the image
        // is usually PRESENT_SRC_KHR, NOT COLOR_ATTACHMENT_OPTIMAL.
        //
        // Hardcoding COLOR_ATTACHMENT_OPTIMAL was the cause of an all-zero
        // readback: oldLayout then disagrees with the real layout, MoltenVK
        // treats the barrier as a no-op, and vkCmdCopyImageToBuffer reads the
        // image in the wrong layout. The defensive fallback below upgrades
        // UNDEFINED to COLOR_ATTACHMENT_OPTIMAL, matching the pre-existing
        // behaviour when nothing is tracked yet.
        src_layout = backend_active_swapchain_color_layout();
    } else {
        mithril::Framebuffer* fbo = mithril::state_get_framebuffer(readFboName);
        if (fbo) {
            auto& tbl = mithril::vk::texture_table();
            auto tit = tbl.find(fbo->colors[0].texture);
            if (tit != tbl.end()) src_layout = tit->second.currentLayout;
        }
    }
    // Defensive fallback: if the tracked layout is still unknown/undefined,
    // treat it as a colour-attachable image that a render pass just wrote to.
    // (oldLayout=UNDEFINED would discard contents on the transition, so we must
    // not leave it UNDEFINED here.)
    if (src_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        src_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    // Pick a host-visible staging buffer format. We always copy as RGBA8 on
    // the host side and convert to the requested (format,type) afterwards if
    // needed. MoltenVK supports VK_FORMAT_R8G8B8A8_UNORM as a transfer source
    // for any colour-attachable format.
    VkFormat staging_fmt = (src_fmt != VK_FORMAT_UNDEFINED) ? src_fmt : VK_FORMAT_R8G8B8A8_UNORM;
    int src_bpp = 4;
    switch (staging_fmt) {
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB: src_bpp = 4; break;
        case VK_FORMAT_R8G8B8_UNORM:  src_bpp = 3; break;
        case VK_FORMAT_R5G6B5_UNORM_PACK16: src_bpp = 2; break;
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:      src_bpp = 1; break;
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SNORM:   src_bpp = 2; break;
        case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_R16_UNORM:    src_bpp = 2; break;
        case VK_FORMAT_R32_SFLOAT:   src_bpp = 4; break;
        default: src_bpp = 4; break;
    }
    VkDeviceSize staging_size = (VkDeviceSize)w * (VkDeviceSize)h * (VkDeviceSize)src_bpp;

    // Create a host-visible staging buffer big enough for the copy.
    BufferEntry staging{};
    if (!create_buffer(staging, staging_size,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT, nullptr)) {
        return 0;
    }

    OneShotCtx c;
    if (!begin_one_shot(c)) {
        destroy_buffer_entry(staging);
        return 0;
    }

    // Transition the source image to TRANSFER_SRC_OPTIMAL.
    // NOTE: named `bar` (not `b`) to avoid clashing with the `Backend* b`
    // declared at the top of read_pixels(); the backend pointer is reused
    // below (b->device) after this barrier block.
    VkImageMemoryBarrier bar{};
    bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    // oldLayout MUST be the image's actual tracked layout (see src_layout
    // above). Hardcoding COLOR_ATTACHMENT_OPTIMAL here makes the transition a
    // no-op on MoltenVK when the image is in a read-only layout after a render
    // pass, returning garbage/black on readback.
    if (readFboName==0 && std::getenv("MITHRIL_DUMP_BLIT"))
        MITHRIL_LOG_WARN("vk-diag","read-fbo0 img=%p layout=%d defaultColorImg=%p",
          (void*)src_image,(int)src_layout,(void*)g_state->eglDefaultColorImage);
    bar.oldLayout = src_layout;
    bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = src_image;
    bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bar.subresourceRange.baseMipLevel = 0;
    bar.subresourceRange.levelCount = 1;
    bar.subresourceRange.baseArrayLayer = 0;
    bar.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &bar);

    // GL's origin is bottom-left; Vulkan's is top-left. Flip the Y axis by
    // adjusting the source offset: read from (x, imageHeight - y - h).
    // We don't know imageHeight here without an extra query, so we copy the
    // whole width x h strip starting at the top of the image and let the
    // caller interpret the rows. For the common MC Java case (reading the full
    // framebuffer) this is correct because y==0 and h==imageHeight.
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { x, y, 0 };
    region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };
    vkCmdCopyImageToBuffer(c.cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           staging.buffer, 1, &region);

    // Transition the source image back to COLOR_ATTACHMENT_OPTIMAL so
    // subsequent draws can render into it again.
    bar.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bar.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bar.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &bar);
    // Update the tracked layout so the next begin_render_pass / draw uses a
    // non-stale oldLayout (otherwise its barrier would claim SHADER_READ_ONLY_OPTIMAL
    // while the image is actually in COLOR_ATTACHMENT_OPTIMAL -> same no-op
    // transition -> dropped draw / black screen). Only the user-FBO colour
    // attachment lives in texture_table(); the swapchain image is handled by
    // the activeSwapchain path in begin_render_pass/commit_frame.
    if (src_tex_id != 0) {
        auto& tbl = mithril::vk::texture_table();
        auto tit = tbl.find(src_tex_id);
        if (tit != tbl.end()) tit->second.currentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    else if (readFboName == 0) {
        // The barrier above left the swapchain image in COLOR_ATTACHMENT_OPTIMAL.
        // Record that, or commit_frame()'s next barrier would use a stale
        // oldLayout (PRESENT_SRC_KHR) against an image that is no longer in it.
        backend_set_active_swapchain_color_layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    end_one_shot(c);

    // Map the staging buffer and convert/copy into out_pixels.
    void* mapped = nullptr;
    vkMapMemory(b->device, staging.memory, 0, staging_size, 0, &mapped);
    if (!mapped) {
        destroy_buffer_entry(staging);
        return 0;
    }

    int dst_bpp = host_texel_bytes(format, type);
    if (dst_bpp <= 0) dst_bpp = 4;

    if (src_bpp == dst_bpp) {
        std::memcpy(out_pixels, mapped, (size_t)staging_size);
    } else {
        // Best-effort: copy byte-for-byte up to the smaller of the two sizes.
        size_t n = std::min((size_t)staging_size, (size_t)w * h * dst_bpp);
        std::memcpy(out_pixels, mapped, n);
    }

    // Channel-order conversion. vkCmdCopyImageToBuffer preserves the image's
    // in-memory channel order; it does NOT reorder to the GL request. A BGRA8
    // swapchain read as GL_RGBA (or an RGBA8 image read as GL_BGRA) would
    // otherwise come back with R and B swapped (observed: a red corner read as
    // blue). Swap bytes 0/2 per texel for the cross-order 4-byte case.
    if (dst_bpp == 4) {
        const bool src_bgra = (src_fmt == VK_FORMAT_B8G8R8A8_UNORM ||
                               src_fmt == VK_FORMAT_B8G8R8A8_SRGB);
        const bool src_rgba = (src_fmt == VK_FORMAT_R8G8B8A8_UNORM ||
                               src_fmt == VK_FORMAT_R8G8B8A8_SRGB);
        const bool dst_bgra = (format == 0x80E1 /*GL_BGRA*/);
        const bool dst_rgba = (format == 0x1908 /*GL_RGBA*/);
        const bool swap_rb = (src_bgra && dst_rgba) || (src_rgba && dst_bgra);
        if (swap_rb) {
            unsigned char* px = static_cast<unsigned char*>(out_pixels);
            size_t count = (size_t)w * (size_t)h;
            for (size_t i = 0; i < count; ++i) {
                unsigned char t = px[i*4+0];
                px[i*4+0] = px[i*4+2];
                px[i*4+2] = t;
            }
        }
    }

    vkUnmapMemory(b->device, staging.memory);
    destroy_buffer_entry(staging);
    return 1;
}

void blit_texture(GLuint src_name, GLuint dst_name,
                  int srcX0, int srcY0, int srcX1, int srcY1,
                  int dstX0, int dstY0, int dstX1, int dstY1,
                  GLbitfield mask, GLenum filter) {
    Backend* b = backend();
    if (!b->initialized) return;
    auto& tbl = mithril::vk::texture_table();
    auto sit = tbl.find(src_name);
    auto dit = tbl.find(dst_name);
    if (sit == tbl.end() || dit == tbl.end()) return;
    TextureEntry& src = sit->second;
    TextureEntry& dst = dit->second;
    if (src.image == VK_NULL_HANDLE || dst.image == VK_NULL_HANDLE) return;

    // Only colour-buffer blits are implemented; depth/stencil blits would
    // need VK_IMAGE_ASPECT_DEPTH_BIT and a NEAREST filter (Vulkan forbids
    // linear filtering on depth formats).
    if (!(mask & GL_COLOR_BUFFER_BIT)) return;

    OneShotCtx c;
    if (!begin_one_shot(c)) return;

    VkImageAspectFlags srcAspect = aspect_for_format(src.format);
    VkImageAspectFlags dstAspect = aspect_for_format(dst.format);

    // Transition source to TRANSFER_SRC_OPTIMAL.
    VkImageMemoryBarrier sb{};
    sb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    sb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    sb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    sb.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sb.image = src.image;
    sb.subresourceRange.aspectMask = srcAspect;
    sb.subresourceRange.baseMipLevel = 0;
    sb.subresourceRange.levelCount = (uint32_t)src.levels;
    sb.subresourceRange.baseArrayLayer = 0;
    sb.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &sb);

    // Transition destination to TRANSFER_DST_OPTIMAL.
    VkImageMemoryBarrier db{};
    db.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    db.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    db.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    db.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    db.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    db.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    db.image = dst.image;
    db.subresourceRange.aspectMask = dstAspect;
    db.subresourceRange.baseMipLevel = 0;
    db.subresourceRange.levelCount = (uint32_t)dst.levels;
    db.subresourceRange.baseArrayLayer = 0;
    db.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &db);

    VkImageBlit blit{};
    blit.srcSubresource.aspectMask = srcAspect;
    blit.srcSubresource.mipLevel = 0;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0] = { srcX0, srcY0, 0 };
    blit.srcOffsets[1] = { srcX1, srcY1, 1 };
    blit.dstSubresource.aspectMask = dstAspect;
    blit.dstSubresource.mipLevel = 0;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[0] = { dstX0, dstY0, 0 };
    blit.dstOffsets[1] = { dstX1, dstY1, 1 };

    VkFilter filt = (filter == GL_LINEAR) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    vkCmdBlitImage(c.cmd,
                   src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, filt);

    // Transition both images back to SHADER_READ_ONLY_OPTIMAL.
    sb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    sb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    sb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &sb);

    db.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    db.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    db.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    db.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &db);

    end_one_shot(c);

    // Both images are back in SHADER_READ_ONLY_OPTIMAL after the blit.
    src.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dst.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

/*
 * Image-level blit between two raw VkImage handles. Used by glBlitFramebuffer
 * when one or both sides is the EGL default framebuffer (whose swapchain
 * image is not tracked in the GL texture table). The caller specifies the
 * initial and final layouts of each image so this helper can emit correct
 * memory barriers without needing to know whether each image is a swapchain
 * drawable (COLOR_ATTACHMENT_OPTIMAL) or a user texture (SHADER_READ_ONLY).
 */
void blit_images_impl(VkImage src_image, VkFormat src_format,
                      VkImageLayout src_initial, VkImageLayout src_final,
                      VkImage dst_image, VkFormat dst_format,
                      VkImageLayout dst_initial, VkImageLayout dst_final,
                      int srcX0, int srcY0, int srcX1, int srcY1,
                      int dstX0, int dstY0, int dstX1, int dstY1,
                      GLbitfield mask, GLenum filter,
                      bool is_dst_default_fbo, int dst_height) {
    Backend* b = backend();
    if (!b->initialized) return;
    if (src_image == VK_NULL_HANDLE || dst_image == VK_NULL_HANDLE) return;
    // Only colour-buffer blits are implemented; depth/stencil blits would
    // need VK_IMAGE_ASPECT_DEPTH_BIT and a NEAREST filter (Vulkan forbids
    // linear filtering on depth formats).
    if (!(mask & GL_COLOR_BUFFER_BIT)) return;

    // Y-flip the destination rectangle when blitting TO the EGL default
    // framebuffer (swapchain drawable). The draw path flips vertex Y in the
    // shader for default-FBO rendering (so on-screen content is in Vulkan's
    // top-left orientation), but blits bypass the vertex shader. GL's blit
    // coords are bottom-left origin; Vulkan's VkImageBlit offsets are top-left
    // origin. Without this flip, blits to the default framebuffer produce an
    // upside-down image (black screen / misaligned content).
    //
    // Deep reference: MobileGL ApplyNativeBlitDefaultFramebufferTransform
    // (VulkanRenderer.cpp:1650-1665, identity branch):
    //   blitRegion.dstOffsets[0].y = extent.y - blitRegion.dstOffsets[0].y;
    //   blitRegion.dstOffsets[1].y = extent.y - blitRegion.dstOffsets[1].y;
    //
    // Source Y is NOT flipped (MobileGL never flips src Y). The source
    // content's orientation is determined by the draw path: user FBO textures
    // are in GL orientation (GL bottom at Vulkan top), so GL src coords map
    // directly; default FBO content is in Vulkan orientation, but GL src coords
    // reading from it still produce correct results because the content
    // orientation and coordinate mapping are consistent within the image.
    if (is_dst_default_fbo && dst_height > 0) {
        // GL destination rect is bottom-origin; VkImageBlit offsets are
        // top-origin AND must satisfy offsets[0] <= offsets[1]. Mapping each
        // edge independently as (height - y0),(height - y1) reverses the order
        // (offsets[0].y > offsets[1].y), which is an invalid VkImageBlit and
        // makes MoltenVK blit nothing (uniform black). Convert as
        // (height - y1, height - y0) so the min/max ordering is preserved.
        int fy0 = dst_height - dstY1;
        int fy1 = dst_height - dstY0;
        dstY0 = fy0;
        dstY1 = fy1;
    }

    OneShotCtx c;
    if (!begin_one_shot(c)) return;

    VkImageAspectFlags srcAspect = aspect_for_format(src_format);
    VkImageAspectFlags dstAspect = aspect_for_format(dst_format);

    // Map an image's CURRENT layout to the correct (srcAccessMask, srcStage)
    // for the barrier that moves it into a TRANSFER layout. The old code only
    // distinguished COLOR_ATTACHMENT vs "everything else" (assumed
    // SHADER_READ_ONLY); a PRESENT_SRC_KHR image (which happens mid-frame
    // after a glReadPixels/prepresent commit ran commit_frame and transitioned
    // the swapchain image to PRESENT_SRC) was given SHADER_READ/FRAGMENT_SHADER,
    // an invalid source for leaving the present layout. MoltenVK treated that
    // barrier as a no-op, so vkCmdBlitImage wrote the destination in the wrong
    // layout -> uniform black.
    auto src_barrier_info = [](VkImageLayout lay)
        -> std::pair<VkAccessFlags, VkPipelineStageFlags> {
        switch (lay) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return {VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return {VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            return {0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return {VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return {VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
        default:
            return {0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        }
    };

    // Transition source to TRANSFER_SRC_OPTIMAL.
    VkImageMemoryBarrier sb{};
    sb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    VkAccessFlags s_in_acc; VkPipelineStageFlags s_in_stage;
    std::tie(s_in_acc, s_in_stage) = src_barrier_info(src_initial);
    sb.srcAccessMask = s_in_acc;
    sb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    sb.oldLayout = src_initial;
    sb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sb.image = src_image;
    sb.subresourceRange.aspectMask = srcAspect;
    sb.subresourceRange.baseMipLevel = 0;
    sb.subresourceRange.levelCount = 1;
    sb.subresourceRange.baseArrayLayer = 0;
    sb.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(c.cmd, s_in_stage,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &sb);

    // Transition destination to TRANSFER_DST_OPTIMAL.
    VkImageMemoryBarrier db{};
    db.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    VkAccessFlags d_in_acc; VkPipelineStageFlags d_in_stage;
    std::tie(d_in_acc, d_in_stage) = src_barrier_info(dst_initial);
    db.srcAccessMask = d_in_acc;
    db.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    db.oldLayout = dst_initial;
    db.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    db.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    db.image = dst_image;
    db.subresourceRange.aspectMask = dstAspect;
    db.subresourceRange.baseMipLevel = 0;
    db.subresourceRange.levelCount = 1;
    db.subresourceRange.baseArrayLayer = 0;
    db.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(c.cmd, d_in_stage,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &db);

    VkImageBlit blit{};
    blit.srcSubresource.aspectMask = srcAspect;
    blit.srcSubresource.mipLevel = 0;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0] = { srcX0, srcY0, 0 };
    blit.srcOffsets[1] = { srcX1, srcY1, 1 };
    blit.dstSubresource.aspectMask = dstAspect;
    blit.dstSubresource.mipLevel = 0;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[0] = { dstX0, dstY0, 0 };
    blit.dstOffsets[1] = { dstX1, dstY1, 1 };

    if (std::getenv("MITHRIL_DUMP_BLIT") || std::getenv("MITHRIL_DRAW_TRACE")) {
        MITHRIL_LOG_WARN("vk-diag","blit-exec srcImg=%p dstImg=%p srcInit=%d srcFinal=%d dstInit=%d dstFinal=%d src[%d,%d,%d,%d] dst[%d,%d,%d,%d] dstH=%d dstDefault=%d",
          (void*)src_image,(void*)dst_image,(int)src_initial,(int)src_final,(int)dst_initial,(int)dst_final,srcX0,srcY0,srcX1,srcY1,dstX0,dstY0,dstX1,dstY1,dst_height,(int)is_dst_default_fbo);
    }
    VkFilter filt = (filter == GL_LINEAR) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    vkCmdBlitImage(c.cmd,
                   src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, filt);

    // Transition both images back to their final layouts.
    sb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    sb.dstAccessMask = (src_final == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        ? (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
                        : VK_ACCESS_SHADER_READ_BIT;
    sb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sb.newLayout = src_final;
    VkPipelineStageFlags srcFinalStage = (src_final == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         srcFinalStage, 0,
                         0, nullptr, 0, nullptr, 1, &sb);

    db.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    db.dstAccessMask = (dst_final == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        ? (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
                        : VK_ACCESS_SHADER_READ_BIT;
    db.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    db.newLayout = dst_final;
    VkPipelineStageFlags dstFinalStage = (dst_final == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         dstFinalStage, 0,
                         0, nullptr, 0, nullptr, 1, &db);

    // TEMP (MITHRIL_BLIT_VERIFY): read the dst swapchain straight back on the
    // GPU right after the blit to decide whether vkCmdBlitImage wrote it.
    BufferEntry vstg{};
    bool vverify = is_dst_default_fbo && std::getenv("MITHRIL_BLIT_VERIFY");
    int vvw = dstX1 - dstX0, vvh = dst_height;
    if (vverify) {
        create_buffer(vstg, (VkDeviceSize)vvw*vvh*4,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, nullptr);
        VkImageMemoryBarrier vb0 = db;
        vb0.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vb0.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vb0.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        vb0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &vb0);
        VkBufferImageCopy vr{};
        vr.imageSubresource.aspectMask = dstAspect;
        vr.imageExtent = {(uint32_t)vvw,(uint32_t)vvh,1};
        vkCmdCopyImageToBuffer(c.cmd, dst_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               vstg.buffer, 1, &vr);
        vb0.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vb0.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vb0.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vb0.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &vb0);
    }

    end_one_shot(c);

    if (vverify) {
        void* vm = nullptr;
        vkMapMemory(b->device, vstg.memory, 0, (VkDeviceSize)vvw*vvh*4, 0, &vm);
        unsigned char* qp = (unsigned char*)vm;
        auto PP = [&](int x,int y)->unsigned char*{ return &qp[((size_t)y*vvw+x)*4]; };
        unsigned char* zc = PP(vvw/2,(int)(vvh*0.62));
        unsigned char* zk = PP(20,20);
        MITHRIL_LOG_WARN("vk-diag","BLIT-VERIFY center=(%d,%d,%d,%d) corner=(%d,%d,%d,%d)",
          zc[0],zc[1],zc[2],zc[3],zk[0],zk[1],zk[2],zk[3]);
        vkUnmapMemory(b->device, vstg.memory);
        destroy_buffer_entry(vstg);
    }
}

// ===========================================================================
// blit-to-default via a fullscreen textured quad
// ===========================================================================
// vkCmdBlitImage into a MoltenVK swapchain drawable writes no pixels (proven
// by an in-GPU BLIT-VERIFY: dst reads all-zero, no Metal validation error),
// while the normal render-pass draw path to the drawable works (CI Minecraft
// 1.21.1 menu). Implement a COLOUR glBlitFramebuffer whose destination is
// FBO 0 as a fullscreen textured quad drawn through a self-contained render
// pass. Depth/stencil blits still use backend_blit_images.
namespace {

const char* kBlitQuadVS = R"(#version 450
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUv;
layout(location=0) out vec2 vUv;
void main(){ vUv = aUv; gl_Position = vec4(aPos, 0.0, 1.0); }
)";

const char* kBlitQuadFS = R"(#version 450
layout(location=0) in vec2 vUv;
layout(location=0) out vec4 outColor;
layout(set=0,binding=0) uniform sampler2D uTex;
void main(){ outColor = texture(uTex, vUv); }
)";

bool bq_compile_stage(EShLanguage stage, const char* src,
                      std::vector<uint32_t>& out) {
    static std::once_flag s_init;
    std::call_once(s_init, []() { glslang::InitializeProcess(); });
    static std::mutex s_mu;
    std::lock_guard<std::mutex> lk(s_mu);
    glslang::TShader sh(stage);
    sh.setStrings(&src, 1);
    sh.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 450);
    sh.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
    sh.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);
    const TBuiltInResource* res = GetDefaultResources();
    if (!sh.parse(res, 450, false, (EShMessages)(EShMsgDefault))) {
        MITHRIL_LOG_WARN("blit-quad", "shader parse failed: %s", sh.getInfoLog());
        return false;
    }
    glslang::TProgram prog;
    prog.addShader(&sh);
    if (!prog.link(EShMsgDefault)) {
        MITHRIL_LOG_WARN("blit-quad", "shader link failed: %s", prog.getInfoLog());
        return false;
    }
    glslang::TIntermediate* inter = prog.getIntermediate(stage);
    if (!inter) return false;
    glslang::SpvOptions spv_opts;
    glslang::GlslangToSpv(*inter, out, &spv_opts);
    return !out.empty();
}

struct BqGpu {
    VkShaderModule      vs = VK_NULL_HANDLE;
    VkShaderModule      fs = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout    pipeLayout = VK_NULL_HANDLE;
    VkSampler           sampNearest = VK_NULL_HANDLE;
    VkSampler           sampLinear = VK_NULL_HANDLE;
    VkDescriptorPool    pool = VK_NULL_HANDLE;
    VkBuffer            vbo = VK_NULL_HANDLE;
    VkDeviceMemory      vboMem = VK_NULL_HANDLE;
    void*               vboMap = nullptr;
    std::unordered_map<uint32_t, VkRenderPass> rpByFmt;
    std::unordered_map<uint32_t, VkPipeline>   pipeByFmt;
    // Dynamic-rendering (VK_KHR_dynamic_rendering) blit pipelines, keyed by dst
    // format; renderPass=VK_NULL_HANDLE, compatible with the begin_render_pass()
    // frame path.
    std::unordered_map<uint32_t, VkPipeline>   drPipeByFmt;
    // Same blit pipelines compiled against a traditional VkRenderPass, for
    // devices without VK_KHR_dynamic_rendering. Compiled against the CANONICAL
    // pass (LOAD/STORE); loadOp/storeOp do not participate in render-pass
    // compatibility, so these stay usable with whatever variant
    // begin_render_pass() picks at recording time.
    std::unordered_map<uint32_t, VkPipeline>   classicPipeByFmt;
    // One descriptor set per swapchain image index; rebuilt only when that image
    // is re-acquired (prior present complete -> prior set not in GPU use).
    std::vector<VkDescriptorSet>               frameSets;
    bool                ready = false;
};
BqGpu& bq_gpu() { static BqGpu g; return g; }

bool bq_ensure_gpu() {
    BqGpu& g = bq_gpu();
    if (g.ready) return true;
    Backend* b = backend();
    std::vector<uint32_t> vspv, fspv;
    if (!bq_compile_stage(EShLangVertex, kBlitQuadVS, vspv)) return false;
    if (!bq_compile_stage(EShLangFragment, kBlitQuadFS, fspv)) return false;

    VkShaderModuleCreateInfo vm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    vm.codeSize = vspv.size() * 4; vm.pCode = vspv.data();
    if (vkCreateShaderModule(b->device, &vm, nullptr, &g.vs) != VK_SUCCESS) return false;
    vm.codeSize = fspv.size() * 4; vm.pCode = fspv.data();
    if (vkCreateShaderModule(b->device, &vm, nullptr, &g.fs) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding bind{};
    bind.binding = 0;
    bind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bind.descriptorCount = 1;
    bind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sl.bindingCount = 1; sl.pBindings = &bind;
    if (vkCreateDescriptorSetLayout(b->device, &sl, nullptr, &g.setLayout) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &g.setLayout;
    if (vkCreatePipelineLayout(b->device, &pl, nullptr, &g.pipeLayout) != VK_SUCCESS) return false;

    auto mk_sampler = [&](VkFilter f, VkSampler& out) -> bool {
        VkSamplerCreateInfo s{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        s.magFilter = f; s.minFilter = f;
        s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        s.addressModeU = s.addressModeV = s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        s.minLod = 0.0f; s.maxLod = 0.0f;
        return vkCreateSampler(b->device, &s, nullptr, &out) == VK_SUCCESS;
    };
    if (!mk_sampler(VK_FILTER_NEAREST, g.sampNearest)) return false;
    if (!mk_sampler(VK_FILTER_LINEAR, g.sampLinear)) return false;

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};
    VkDescriptorPoolCreateInfo pc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pc.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pc.maxSets = 4; pc.poolSizeCount = 1; pc.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(b->device, &pc, nullptr, &g.pool) != VK_SUCCESS) return false;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 6 * 16; bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (vkCreateBuffer(b->device, &bci, nullptr, &g.vbo) != VK_SUCCESS) return false;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(b->device, g.vbo, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(b->physicalDevice, &mp);
    uint32_t mi = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mi = i; break; }
    }
    if (mi == UINT32_MAX) return false;
    VkMemoryAllocateInfo mal{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mal.allocationSize = mr.size; mal.memoryTypeIndex = mi;
    if (vkAllocateMemory(b->device, &mal, nullptr, &g.vboMem) != VK_SUCCESS) return false;
    vkBindBufferMemory(b->device, g.vbo, g.vboMem, 0);
    vkMapMemory(b->device, g.vboMem, 0, mr.size, 0, &g.vboMap);

    g.ready = true;
    return true;
}

VkRenderPass bq_render_pass(VkFormat fmt) {
    BqGpu& g = bq_gpu();
    auto it = g.rpByFmt.find((uint32_t)fmt);
    if (it != g.rpByFmt.end()) return it->second;
    Backend* b = backend();
    VkAttachmentDescription a{};
    a.format = fmt; a.samples = VK_SAMPLE_COUNT_1_BIT;
    a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    a.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    a.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1; sp.pColorAttachments = &ref;
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = 1; rp.pAttachments = &a;
    rp.subpassCount = 1; rp.pSubpasses = &sp;
    rp.dependencyCount = 1; rp.pDependencies = &dep;
    VkRenderPass out = VK_NULL_HANDLE;
    if (vkCreateRenderPass(b->device, &rp, nullptr, &out) == VK_SUCCESS)
        g.rpByFmt[(uint32_t)fmt] = out;
    return out;
}

VkPipeline bq_pipeline(VkRenderPass rp, VkFormat fmt) {
    BqGpu& g = bq_gpu();
    auto it = g.pipeByFmt.find((uint32_t)fmt);
    if (it != g.pipeByFmt.end()) return it->second;
    Backend* b = backend();
    VkPipelineShaderStageCreateInfo st[2]{};
    st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   st[0].module = g.vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = g.fs; st[1].pName = "main";
    VkVertexInputBindingDescription vib{0, 16, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription via[2]{
        {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R32G32_SFLOAT, 8}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = via;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState ba{};
    ba.blendEnable = VK_FALSE;
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                      | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo bl{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    bl.attachmentCount = 1; bl.pAttachments = &ba;
    VkDynamicState dyn[2]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2; ds.pDynamicStates = dyn;
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2; gp.pStages = st;
    gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia; gp.pViewportState = &vp;
    gp.pRasterizationState = &rs; gp.pMultisampleState = &ms;
    gp.pColorBlendState = &bl; gp.pDynamicState = &ds;
    gp.layout = g.pipeLayout; gp.renderPass = rp; gp.subpass = 0;
    VkPipeline out = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(b->device, VK_NULL_HANDLE, 1, &gp, nullptr, &out) == VK_SUCCESS)
        g.pipeByFmt[(uint32_t)fmt] = out;
    return out;
}

} // namespace

void blit_to_default_quad(VkImage src_image, VkFormat src_format,
                          int src_w, int src_h,
                          int sx0, int sy0, int sx1, int sy1,
                          int dx0, int dy0, int dx1, int dy1,
                          GLenum filter) {
    Backend* b = backend();
    if (!b->initialized) return;
    Swapchain* sc = active_swapchain();
    if (!sc || sc->currentImage < 0 ||
        sc->currentImage >= (int)sc->views.size()) {
        MITHRIL_LOG_WARN("blit-quad", "no active acquired swapchain (sc=%p currentImage=%d views=%zu)",
            (void*)sc, sc?sc->currentImage:-99, sc?sc->views.size():0);
        return;
    }
    if (!bq_ensure_gpu()) {
        MITHRIL_LOG_WARN("blit-quad", "gpu resources unavailable");
        return;
    }
    BqGpu& g = bq_gpu();

    int DW = sc->actualDrawableWidth  > 0 ? sc->actualDrawableWidth  : sc->width;
    int DH = sc->actualDrawableHeight > 0 ? sc->actualDrawableHeight : sc->height;
    if (DW <= 0 || DH <= 0 || src_w <= 0 || src_h <= 0) {
        MITHRIL_LOG_WARN("blit-quad", "bad dims dst=%dx%d src=%dx%d", DW,DH,src_w,src_h);
        return;
    }
    VkImage     dst_image = sc->images[sc->currentImage];
    VkImageView dst_view  = sc->views[sc->currentImage];
    VkFormat    dst_fmt   = sc->format;

    // Source view + real tracked layout.
    VkImageView src_view = VK_NULL_HANDLE;
    VkImageLayout src_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bool src_tracked = false;
    for (auto& kv : texture_table()) {
        if (kv.second.image == src_image) {
            src_view = kv.second.view;
            src_layout = kv.second.currentLayout;
            src_tracked = true;
            break;
        }
    }
    bool created_src_view = false;
    if (src_view == VK_NULL_HANDLE) {
        VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vc.image = src_image; vc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vc.format = src_format;
        vc.subresourceRange = {aspect_for_format(src_format), 0, 1, 0, 1};
        if (vkCreateImageView(b->device, &vc, nullptr, &src_view) != VK_SUCCESS) return;
        created_src_view = true;
    }

    // Acquire-wait semaphore (only if the frame has not consumed it yet).
    VkSemaphore wait = VK_NULL_HANDLE;
    if (!sc->imageAvailableConsumed && sc->imageAvailableFrameSlot >= 0 &&
        sc->imageAvailableFrameSlot < (int)sc->imageAvailablePerFrame.size())
        wait = sc->imageAvailablePerFrame[sc->imageAvailableFrameSlot];

    OneShotCtx c;
    if (!begin_one_shot(c)) {
        if (created_src_view) vkDestroyImageView(b->device, src_view, nullptr);
        return;
    }
    if (wait != VK_NULL_HANDLE) {
        c.waitSemaphore = wait;
        c.waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        c.consumeAcquire = true;
    }

    // Source (real layout) -> SHADER_READ_ONLY_OPTIMAL.
    auto src_barrier_src = [](VkImageLayout l)
        -> std::pair<VkAccessFlags, VkPipelineStageFlags> {
        switch (l) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return {VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return {VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return {VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
        default:
            return {0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        }
    };
    VkAccessFlags s_acc; VkPipelineStageFlags s_stage;
    std::tie(s_acc, s_stage) = src_barrier_src(src_layout);
    VkImageMemoryBarrier sb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    sb.image = src_image;
    sb.subresourceRange = {aspect_for_format(src_format), 0, 1, 0, 1};
    sb.oldLayout = src_layout;
    sb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sb.srcAccessMask = s_acc;
    sb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(c.cmd, s_stage, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &sb);

    // Destination UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL (acquire => don't-care).
    VkImageMemoryBarrier db{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    db.image = dst_image;
    db.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    db.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    db.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    db.srcAccessMask = 0;
    db.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &db);

    // Fullscreen quad geometry. NDC maps the GL dst rect directly; the source
    // user texture is in GL orientation so V is flipped.
    struct BqVert { float x, y, u, v; };
    auto U = [&](float qx) -> float { return qx / (float)src_w; };
    auto VV = [&](float qy) -> float { return 1.0f - qy / (float)src_h; };
    auto NX = [&](float px) -> float { return 2.0f * px / (float)DW - 1.0f; };
    auto NY = [&](float py) -> float { return 2.0f * py / (float)DH - 1.0f; };
    BqVert corner[4] = {
        {NX(dx0), NY(dy0), U(sx0), VV(sy0)},
        {NX(dx1), NY(dy0), U(sx1), VV(sy0)},
        {NX(dx1), NY(dy1), U(sx1), VV(sy1)},
        {NX(dx0), NY(dy1), U(sx0), VV(sy1)}};
    BqVert verts[6] = {
        corner[0], corner[1], corner[2],
        corner[0], corner[2], corner[3]};
    std::memcpy(g.vboMap, verts, sizeof(verts));

    // Descriptor set (source combined image sampler).
    VkDescriptorSet set = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = g.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &g.setLayout;
        if (vkAllocateDescriptorSets(b->device, &ai, &set) != VK_SUCCESS) {
            c.ok = false; end_one_shot(c);
            if (created_src_view) vkDestroyImageView(b->device, src_view, nullptr);
            return;
        }
    }
    VkDescriptorImageInfo dii{};
    dii.sampler = (filter == GL_LINEAR) ? g.sampLinear : g.sampNearest;
    dii.imageView = src_view;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set; w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(b->device, 1, &w, 0, nullptr);

    VkRenderPass rp = bq_render_pass(dst_fmt);
    VkPipeline pipe = bq_pipeline(rp, dst_fmt);
    VkFramebuffer fb = VK_NULL_HANDLE;
    if (rp && pipe) {
        VkFramebufferCreateInfo fc{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fc.renderPass = rp; fc.attachmentCount = 1; fc.pAttachments = &dst_view;
        fc.width = (uint32_t)DW; fc.height = (uint32_t)DH; fc.layers = 1;
        if (vkCreateFramebuffer(b->device, &fc, nullptr, &fb) != VK_SUCCESS) fb = VK_NULL_HANDLE;
    }
    if (!rp || !pipe || !fb) {
        MITHRIL_LOG_WARN("blit-quad", "ABORT before pass rp=%p pipe=%p fb=%p dstFmt=%d",
            (void*)rp,(void*)pipe,(void*)fb,(int)dst_fmt);
        vkFreeDescriptorSets(b->device, g.pool, 1, &set);
        c.ok = false; end_one_shot(c);
        if (created_src_view) vkDestroyImageView(b->device, src_view, nullptr);
        return;
    }

    VkRenderPassBeginInfo rpb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpb.renderPass = rp; rpb.framebuffer = fb;
    rpb.renderArea = {{0, 0}, {(uint32_t)DW, (uint32_t)DH}};
    vkCmdBeginRenderPass(c.cmd, &rpb, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp{0.0f, 0.0f, (float)DW, (float)DH, 0.0f, 1.0f};
    vkCmdSetViewport(c.cmd, 0, 1, &vp);
    VkRect2D sr{{0, 0}, {(uint32_t)DW, (uint32_t)DH}};
    vkCmdSetScissor(c.cmd, 0, 1, &sr);
    vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    VkDeviceSize voff = 0;
    vkCmdBindVertexBuffers(c.cmd, 0, 1, &g.vbo, &voff);
    vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            g.pipeLayout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c.cmd, 6, 1, 0, 0);
    vkCmdEndRenderPass(c.cmd);

    MITHRIL_LOG_WARN("blit-quad", "recorded quad, submitting dstView=%p wait=%p",
        (void*)dst_view,(void*)wait);
    end_one_shot(c);
    MITHRIL_LOG_WARN("blit-quad", "quad submit complete");

    vkDestroyFramebuffer(b->device, fb, nullptr);
    vkFreeDescriptorSets(b->device, g.pool, 1, &set);
    if (created_src_view) vkDestroyImageView(b->device, src_view, nullptr);

    // Update tracked layouts so the next pass/present barrier is correct.
    sc->currentColorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    backend_set_active_swapchain_color_layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (src_tracked) {
        for (auto& kv : texture_table())
            if (kv.second.image == src_image)
                kv.second.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

// ---------------------------------------------------------------------------
// Frame-path blit: records the fullscreen textured quad into the backend's
// CURRENT per-frame command buffer and begins the swapchain pass through
// begin_render_pass(). commit_frame() (at eglSwapBuffers) then submits it with
// the acquire wait and presents. This is the only path whose writes to a
// MoltenVK drawable persist; an out-of-band one-shot (blit_to_default_quad)
// does not.
// ---------------------------------------------------------------------------

// Dynamic-rendering blit pipeline (renderPass=VK_NULL_HANDLE), cached by format.
VkPipeline bq_dr_pipeline(VkFormat dst_fmt) {
    BqGpu& g = bq_gpu();
    Backend* b = backend();
    if (b && !b->dynamicRenderingSupported) {
        auto cit = g.classicPipeByFmt.find((uint32_t)dst_fmt);
        if (cit != g.classicPipeByFmt.end()) return cit->second;
    } else {
        auto it = g.drPipeByFmt.find((uint32_t)dst_fmt);
        if (it != g.drPipeByFmt.end()) return it->second;
    }

    VkVertexInputBindingDescription bd{};
    bd.binding = 0; bd.stride = 16; bd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription ads[2]{};
    ads[0].location = 0; ads[0].binding = 0; ads[0].format = VK_FORMAT_R32G32_SFLOAT; ads[0].offset = 0;
    ads[1].location = 1; ads[1].binding = 0; ads[1].format = VK_FORMAT_R32G32_SFLOAT; ads[1].offset = 8;
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &bd;
    vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = ads;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_FALSE; cba.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    VkDynamicState dsts[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2; dyn.pDynamicStates = dsts;

    VkPipelineShaderStageCreateInfo stg[2]{};
    stg[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stg[0].module = g.vs; stg[0].pName = "main";
    stg[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stg[1].module = g.fs; stg[1].pName = "main";

    // A graphics pipeline carries EITHER VkPipelineRenderingCreateInfo in pNext
    // with renderPass == VK_NULL_HANDLE, OR a real VkRenderPass — never both.
    // Handing a driver that has no VK_KHR_dynamic_rendering the
    // rendering-create-info pNext it does not know is how Adreno died inside
    // vkCreateGraphicsPipelines (SIGSEGV, si_addr 0xe8) on the first
    // glBlitFramebuffer: the blit-quad pipeline was the one place that still
    // built itself unconditionally for dynamic rendering.
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1; rci.pColorAttachmentFormats = &dst_fmt;
    VkRenderPass compatRP = VK_NULL_HANDLE;
    if (!b->dynamicRenderingSupported) {
        // 与 Pipeline.cpp 保持一致：用「规范」pass（color load/store 传 nullptr
        // → LOAD/STORE，depth 无附件时 load/store 被忽略）。
        compatRP = mithril_vk_render_pass_for(&dst_fmt, 1, VK_FORMAT_UNDEFINED,
                                              VK_SAMPLE_COUNT_1_BIT,
                                              nullptr, nullptr,
                                              VK_ATTACHMENT_LOAD_OP_LOAD,
                                              VK_ATTACHMENT_STORE_OP_STORE);
        if (compatRP == VK_NULL_HANDLE) {
            MITHRIL_LOG_WARN("blit-quad", "canonical render pass unavailable fmt=%d",
                             (int)dst_fmt);
            return VK_NULL_HANDLE;
        }
    }
    VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    if (compatRP != VK_NULL_HANDLE) {
        gi.pNext = nullptr; gi.renderPass = compatRP;
    } else {
        gi.pNext = &rci; gi.renderPass = VK_NULL_HANDLE;
    }
    gi.stageCount = 2; gi.pStages = stg;
    gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia; gi.pViewportState = &vp;
    gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
    gi.pColorBlendState = &cb; gi.pDynamicState = &dyn;
    gi.subpass = 0; gi.layout = g.pipeLayout;

    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(b->device, b->pipelineCache, 1, &gi, nullptr, &pipe);
    if (r != VK_SUCCESS) {
        MITHRIL_LOG_WARN("blit-quad", "blit pipeline create failed r=%d fmt=%d classic=%d",
                         (int)r, (int)dst_fmt, (int)(compatRP != VK_NULL_HANDLE));
        return VK_NULL_HANDLE;
    }
    if (compatRP != VK_NULL_HANDLE) g.classicPipeByFmt[(uint32_t)dst_fmt] = pipe;
    else                            g.drPipeByFmt[(uint32_t)dst_fmt] = pipe;
    return pipe;
}

void blit_to_default_in_frame(VkImage src_image, VkFormat src_format, int src_w, int src_h,
                              int sx0, int sy0, int sx1, int sy1,
                              int dx0, int dy0, int dx1, int dy1, GLenum filter) {
    Backend* b = backend();
    if (!b->initialized) return;
    Swapchain* sc = active_swapchain();
    if (!sc || sc->currentImage < 0 || sc->currentImage >= (int)sc->views.size()) {
        MITHRIL_LOG_WARN("blit-quad", "inframe: no acquired swapchain");
        return;
    }
    if (!bq_ensure_gpu()) { MITHRIL_LOG_WARN("blit-quad", "inframe: gpu unavailable"); return; }
    BqGpu& g = bq_gpu();
    int DW = sc->actualDrawableWidth  > 0 ? sc->actualDrawableWidth  : sc->width;
    int DH = sc->actualDrawableHeight > 0 ? sc->actualDrawableHeight : sc->height;
    if (DW <= 0 || DH <= 0 || src_w <= 0 || src_h <= 0) {
        MITHRIL_LOG_WARN("blit-quad", "inframe: bad dims dst=%dx%d src=%dx%d",DW,DH,src_w,src_h);
        return;
    }
    VkImageView dst_view = sc->views[sc->currentImage];
    VkFormat    dst_fmt  = sc->format;

    // Source view + real tracked layout.
    VkImageView src_view = VK_NULL_HANDLE;
    VkImageLayout src_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool src_tracked = false;
    for (auto& kv : texture_table()) {
        if (kv.second.image == src_image && kv.second.view != VK_NULL_HANDLE) {
            src_view = kv.second.view; src_layout = kv.second.currentLayout; src_tracked = true;
            break;
        }
    }
    bool created_src_view = false;
    if (src_view == VK_NULL_HANDLE) {
        VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vc.image = src_image; vc.viewType = VK_IMAGE_VIEW_TYPE_2D; vc.format = src_format;
        vc.subresourceRange = {aspect_for_format(src_format), 0, 1, 0, 1};
        if (vkCreateImageView(b->device, &vc, nullptr, &src_view) != VK_SUCCESS) return;
        created_src_view = true;
    }

    // End any still-active offscreen pass (records end; transitions the scene
    // color attachment back to SHADER_READ). No commit.
    if (render_pass_active()) end_render_pass();
    if (!ensure_command_buffer_recording()) {
        MITHRIL_LOG_WARN("blit-quad", "inframe: no recording command buffer");
        if (created_src_view) vkDestroyImageView(b->device, src_view, nullptr);
        return;
    }
    VkCommandBuffer cmd = b->commandBuffer;

    // Barrier src -> SHADER_READ (unless end_render_pass already put it there).
    const VkImageLayout want = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (src_layout != want) {
        VkImageMemoryBarrier mb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        mb.oldLayout = src_layout; mb.newLayout = want;
        mb.image = src_image;
        mb.subresourceRange = {aspect_for_format(src_format), 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &mb);
        if (src_tracked)
            for (auto& kv : texture_table())
                if (kv.second.image == src_image) kv.second.currentLayout = want;
    }

    VkPipeline pipe = bq_dr_pipeline(dst_fmt);
    if (pipe == VK_NULL_HANDLE) {
        if (created_src_view) vkDestroyImageView(b->device, src_view, nullptr);
        return;
    }

    // Descriptor set ring keyed by swapchain image (prior use complete on reacquire).
    if ((int)g.frameSets.size() < (int)sc->images.size())
        g.frameSets.resize(sc->images.size(), VK_NULL_HANDLE);
    int slot = sc->currentImage;
    if (g.frameSets[slot] != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(b->device, g.pool, 1, &g.frameSets[slot]);
        g.frameSets[slot] = VK_NULL_HANDLE;
    }
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = g.pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &g.setLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(b->device, &ai, &set) != VK_SUCCESS) {
        MITHRIL_LOG_WARN("blit-quad", "inframe: descriptor alloc failed");
        if (created_src_view) vkDestroyImageView(b->device, src_view, nullptr);
        return;
    }
    g.frameSets[slot] = set;
    VkDescriptorImageInfo dii{};
    dii.sampler = (filter == GL_LINEAR) ? g.sampLinear : g.sampNearest;
    dii.imageView = src_view; dii.imageLayout = want;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set; w.dstBinding = 0; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &dii;
    vkUpdateDescriptorSets(b->device, 1, &w, 0, nullptr);

    // Fill the shared VBO with this rect (NDC; flip source V for GL orientation).
    float qx0=(float)sx0, qy0=(float)sy0, qx1=(float)sx1, qy1=(float)sy1;
    float px0=(float)dx0, py0=(float)dy0, px1=(float)dx1, py1=(float)dy1;
    float DWf=(float)DW, DHf=(float)DH, swf=(float)src_w, shf=(float)src_h;
    float u0=qx0/swf, u1=qx1/swf;
    // Same single-flip rule as the draw path: flip the source V only when
    // MoltenVK is not already flipping vertex Y, otherwise the blitted
    // frame comes out vertically mirrored.
    const bool flip_v = (backend_yflip_enabled() != 0);
    float v0 = flip_v ? (1.0f - qy1/shf) : (qy0/shf);
    float v1 = flip_v ? (1.0f - qy0/shf) : (qy1/shf);
    float nx0=2.0f*px0/DWf-1.0f, nx1=2.0f*px1/DWf-1.0f;
    float ny0=2.0f*py0/DHf-1.0f, ny1=2.0f*py1/DHf-1.0f;
    float verts[6][4] = {
        {nx0,ny0,u0,v0},{nx0,ny1,u0,v1},{nx1,ny1,u1,v1},
        {nx0,ny0,u0,v0},{nx1,ny1,u1,v1},{nx1,ny0,u1,v0},
    };
    std::memcpy(g.vboMap, verts, sizeof(verts));

    // Clear stale user-FBO attachment registration, then begin the swapchain
    // dynamic-rendering pass (PRESENT/UNDEFINED -> COLOR_ATTACHMENT barrier).
    backend_set_fbo_attachment_tex_ids(nullptr, 0, 0);
    begin_render_pass(&dst_view, 1, VK_NULL_HANDLE, DW, DH, 1);

    VkViewport vp{0.0f,0.0f,(float)DW,(float)DH,0.0f,1.0f};
    vkCmdSetViewport(cmd,0,1,&vp);
    VkRect2D sr{{0,0},{(uint32_t)DW,(uint32_t)DH}};
    vkCmdSetScissor(cmd,0,1,&sr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    VkDeviceSize voff = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g.vbo, &voff);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.pipeLayout,
                            0, 1, &set, 0, nullptr);
    vkCmdDraw(cmd, 6, 1, 0, 0);
    // Pass stays active; eglSwapBuffers ends it, commits (waits acquire), presents.
    MITHRIL_LOG_WARN("blit-quad", "inframe quad recorded into frame buffer");
    // A rare temporary source view stays referenced until present; not destroyed.
}

// Diagnostic (MITHRIL_FS_PROBE): constant-yellow fragment module used to
// decide whether the vertex stage rasterizes when the app's own fragment
// shader writes nothing. Not part of the shipping path.
VkShaderModule create_probe_fs_module() {
    static const char* kProbeFS =
        "#version 450\n"
        "layout(location=0) out vec4 outc;\n"
        "void main(){ outc = vec4(1.0,1.0,0.0,1.0); }\n";
    std::vector<uint32_t> spv;
    if (!bq_compile_stage(EShLangFragment, kProbeFS, spv)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo vm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    vm.codeSize = spv.size() * 4;
    vm.pCode = spv.data();
    VkShaderModule m = VK_NULL_HANDLE;
    Backend* bb = backend();
    if (vkCreateShaderModule(bb->device, &vm, nullptr, &m) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return m;
}

// Self-contained fixed fullscreen triangle (gl_VertexIndex) + constant yellow FS,
// empty pipeline layout, RGBA8 dynamic rendering. Verifies a user-FBO pass can
// rasterize draws independent of app vertex data / UBO.
const char* kFixedQVS = R"(#version 450
layout(location=0) out vec2 vUv;
void main(){
    vec2 p = vec2(float((gl_VertexIndex<<1)&2), float(gl_VertexIndex&2));
    gl_Position = vec4(p*2.0-1.0, 0.0, 1.0);
    vUv = p;
})";
const char* kFixedQFS = R"(#version 450
layout(location=0) in vec2 vUv;
layout(location=0) out vec4 outc;
void main(){ outc = vec4(1.0,1.0,0.0,1.0); })";

void probe_fixed_quad() {
    static VkPipeline pipe = VK_NULL_HANDLE;
    Backend* b = backend();
    // Diagnostic helper: it records with vkCmdBeginRendering, so it is
    // meaningless (and would crash) on a device without dynamic rendering.
    if (!b || !b->dynamicRenderingSupported) return;
    VkCommandBuffer cmd = b->commandBuffer;
    if (pipe == VK_NULL_HANDLE) {
        std::vector<uint32_t> vspv,fspv;
        if(!bq_compile_stage(EShLangVertex,kFixedQVS,vspv)) return;
        if(!bq_compile_stage(EShLangFragment,kFixedQFS,fspv)) return;
        VkShaderModule vsm,fsm;
        VkShaderModuleCreateInfo vm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        vm.codeSize=vspv.size()*4; vm.pCode=vspv.data();
        if(vkCreateShaderModule(b->device,&vm,nullptr,&vsm)!=VK_SUCCESS) return;
        vm.codeSize=fspv.size()*4; vm.pCode=fspv.data();
        if(vkCreateShaderModule(b->device,&vm,nullptr,&fsm)!=VK_SUCCESS) return;
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout lay; if(vkCreatePipelineLayout(b->device,&pl,nullptr,&lay)!=VK_SUCCESS)return;
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount=1; vp.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_NONE;
        rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState ca{}; ca.blendEnable=VK_FALSE; ca.colorWriteMask=0xF;
        cb.attachmentCount=1; cb.pAttachments=&ca;
        VkDynamicState dyns[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount=2; dyn.pDynamicStates=dyns;
        VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        VkFormat f=VK_FORMAT_R8G8B8A8_UNORM; rci.colorAttachmentCount=1; rci.pColorAttachmentFormats=&f;
        VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        VkPipelineShaderStageCreateInfo st[2]={{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        st[0].stage=VK_SHADER_STAGE_VERTEX_BIT; st[0].module=vsm; st[0].pName="main";
        st[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module=fsm; st[1].pName="main";
        gi.pNext=&rci; gi.stageCount=2; gi.pStages=st; gi.pVertexInputState=&vi; gi.pInputAssemblyState=&ia;
        gi.pViewportState=&vp; gi.pRasterizationState=&rs; gi.pMultisampleState=&ms; gi.pColorBlendState=&cb;
        gi.pDynamicState=&dyn; gi.layout=lay; gi.renderPass=VK_NULL_HANDLE;
        if(vkCreateGraphicsPipelines(b->device,b->pipelineCache,1,&gi,nullptr,&pipe)!=VK_SUCCESS) return;
    }
    VkViewport vpv{0,0,2048,2048,0,1}; vkCmdSetViewport(cmd,0,1,&vpv);
    VkRect2D sr{{0,0},{2048,2048}}; vkCmdSetScissor(cmd,0,1,&sr);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipe);
    vkCmdDraw(cmd,3,1,0,0);
}

// Fixed fullscreen verts transformed by the APP's Matrices UBO (set0 binding0),
// constant cyan FS, reusing the app program's pipeline layout + already-bound
// descriptor set. Distinguishes bad UBO matrix content (cyan clips -> nothing)
// from bad position attributes (cyan region appears).
const char* kUboQVS = R"(#version 450
layout(set=0,binding=0) uniform Matrices { mat4 ProjMat; mat4 ModelViewMat; };
void main(){
    vec2 p = vec2(float((gl_VertexIndex<<1)&2), float(gl_VertexIndex&2));
    gl_Position = ProjMat * ModelViewMat * vec4(p*2.0-1.0, 0.0, 1.0);
})";
const char* kUboQFS = R"(#version 450
layout(location=0) out vec4 outc;
void main(){ outc = vec4(0.0,1.0,1.0,1.0); })";

void probe_ubo_fixed(GLuint program) {
    Backend* b = backend();
    // Diagnostic helper: vkCmdBeginRendering only. See probe_fixed_quad().
    if (!b || !b->dynamicRenderingSupported) return;
    VkCommandBuffer cmd = b->commandBuffer;
    auto it = program_table().find(program);
    if (it == program_table().end()) return;
    VkPipelineLayout lay = it->second.pipelineLayout;
    if (lay == VK_NULL_HANDLE) return;
    static std::unordered_map<VkPipelineLayout,VkPipeline> cache;
    VkPipeline pipe = VK_NULL_HANDLE;
    auto ci = cache.find(lay);
    if (ci != cache.end()) { pipe = ci->second; }
    if (pipe == VK_NULL_HANDLE) {
        std::vector<uint32_t> vspv,fspv;
        if(!bq_compile_stage(EShLangVertex,kUboQVS,vspv)) return;
        if(!bq_compile_stage(EShLangFragment,kUboQFS,fspv)) return;
        VkShaderModule vsm,fsm;
        VkShaderModuleCreateInfo vm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        vm.codeSize=vspv.size()*4; vm.pCode=vspv.data();
        if(vkCreateShaderModule(b->device,&vm,nullptr,&vsm)!=VK_SUCCESS)return;
        vm.codeSize=fspv.size()*4; vm.pCode=fspv.data();
        if(vkCreateShaderModule(b->device,&vm,nullptr,&fsm)!=VK_SUCCESS)return;
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount=1; vp.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_NONE;
        rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState ca{}; ca.blendEnable=VK_FALSE; ca.colorWriteMask=0xF;
        cb.attachmentCount=1; cb.pAttachments=&ca;
        VkDynamicState dyns[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount=2; dyn.pDynamicStates=dyns;
        VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        VkFormat f=VK_FORMAT_R8G8B8A8_UNORM; rci.colorAttachmentCount=1; rci.pColorAttachmentFormats=&f;
        VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        VkPipelineShaderStageCreateInfo st[2]={{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        st[0].stage=VK_SHADER_STAGE_VERTEX_BIT; st[0].module=vsm; st[0].pName="main";
        st[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module=fsm; st[1].pName="main";
        gi.pNext=&rci; gi.stageCount=2; gi.pStages=st; gi.pVertexInputState=&vi; gi.pInputAssemblyState=&ia;
        gi.pViewportState=&vp; gi.pRasterizationState=&rs; gi.pMultisampleState=&ms; gi.pColorBlendState=&cb;
        gi.pDynamicState=&dyn; gi.layout=lay; gi.renderPass=VK_NULL_HANDLE;
        if(vkCreateGraphicsPipelines(b->device,b->pipelineCache,1,&gi,nullptr,&pipe)!=VK_SUCCESS)return;
        cache[lay]=pipe;
    }
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipe);
    vkCmdDraw(cmd,3,1,0,0);
}

} // namespace vk
} // namespace mithril

// ===========================================================================
// Public C API wrappers (declared in MG_Backend/Backend.h)
// ===========================================================================
extern "C" {

void backend_generate_mipmaps(GLuint name) {
    mithril::vk::generate_mipmaps(name);
}

int backend_read_pixels(int x, int y, int w, int h,
                        GLenum format, GLenum type, void* out_pixels) {
    return mithril::vk::read_pixels(x, y, w, h, format, type, out_pixels);
}

void backend_probe_fixed_quad(void) {
    mithril::vk::probe_fixed_quad();
}

void backend_probe_ubo_fixed(unsigned int program) {
    mithril::vk::probe_ubo_fixed(program);
}

void backend_blit_texture(GLuint src_name, GLuint dst_name,
                          int srcX0, int srcY0, int srcX1, int srcY1,
                          int dstX0, int dstY0, int dstX1, int dstY1,
                          GLbitfield mask, GLenum filter) {
    mithril::vk::blit_texture(src_name, dst_name,
                              srcX0, srcY0, srcX1, srcY1,
                              dstX0, dstY0, dstX1, dstY1,
                              mask, filter);
}

void backend_blit_to_default_quad(VkImage src_image, VkFormat src_format,
                                  int src_w, int src_h,
                                  int srcX0, int srcY0, int srcX1, int srcY1,
                                  int dstX0, int dstY0, int dstX1, int dstY1,
                                  GLenum filter) {
    mithril::vk::blit_to_default_in_frame(src_image, src_format, src_w, src_h,
                                          srcX0, srcY0, srcX1, srcY1,
                                          dstX0, dstY0, dstX1, dstY1, filter);
}

void backend_blit_images(VkImage src_image, VkFormat src_format,
                         VkImage dst_image, VkFormat dst_format,
                         int srcX0, int srcY0, int srcX1, int srcY1,
                         int dstX0, int dstY0, int dstX1, int dstY1,
                         GLbitfield mask, GLenum filter,
                         int is_dst_default_fbo, int dst_height) {
    // Resolve each image's REAL current layout instead of hardcoding
    // COLOR_ATTACHMENT_OPTIMAL. A user-FBO colour texture is in
    // SHADER_READ_ONLY_OPTIMAL after end_render_pass() (it is a sampling
    // source); a barrier whose oldLayout claims COLOR_ATTACHMENT_OPTIMAL
    // against an image actually in SHADER_READ_ONLY is a no-op on MoltenVK,
    // so vkCmdBlitImage reads the source in the wrong layout and produces a
    // uniform-black destination.
    auto layout_for_image = [](VkImage img) -> VkImageLayout {
        if (img == VK_NULL_HANDLE) return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        if (mithril::g_state && img == mithril::g_state->eglDefaultColorImage)
            return backend_active_swapchain_color_layout();
        auto& tbl = mithril::vk::texture_table();
        for (auto& kv : tbl) if (kv.second.image == img) return kv.second.currentLayout;
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    };

    VkImageLayout src_init = layout_for_image(src_image);
    VkImageLayout src_final = src_init;  // source is sampled again afterwards

    VkImageLayout dst_init = layout_for_image(dst_image);
    VkImageLayout dst_final;
    if (is_dst_default_fbo) {
        // The swapchain image was acquired at the end of the PREVIOUS
        // eglSwapBuffers. Its tracked layout is stale PRESENT_SRC_KHR (the
        // layout the prior frame ended in), but the blit is recorded into a
        // separate one-shot command buffer that does NOT wait on the
        // image-available semaphore. A barrier whose oldLayout claims
        // PRESENT_SRC against an image the presentation engine may still own
        // is treated as a no-op by MoltenVK, so vkCmdBlitImage writes the
        // drawable in the wrong layout -> uniform black.
        //
        // After acquire the image contents are don't-care, so oldLayout
        // UNDEFINED is legal regardless of the image's real layout and
        // correctly moves it into TRANSFER_DST for the blit.
        dst_init = VK_IMAGE_LAYOUT_UNDEFINED;
        // Leave the swapchain image in COLOR_ATTACHMENT_OPTIMAL: the bridge's
        // prepresent read and commit_frame()'s present barrier both expect the
        // pre-present state to be COLOR_ATTACHMENT_OPTIMAL.
        dst_final = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else {
        dst_final = dst_init;
    }

    mithril::vk::blit_images_impl(src_image, src_format, src_init, src_final,
                                  dst_image, dst_format, dst_init, dst_final,
                                  srcX0, srcY0, srcX1, srcY1,
                                  dstX0, dstY0, dstX1, dstY1,
                                  mask, filter,
                                  is_dst_default_fbo != 0, dst_height);

    if (is_dst_default_fbo) {
        // Keep Swapchain::currentColorLayout from going stale, or the next
        // begin_render_pass / commit_frame barrier would use a wrong oldLayout.
        backend_set_active_swapchain_color_layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
}

} // extern "C"
