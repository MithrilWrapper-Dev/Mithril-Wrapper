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
#include "Resources.h"
#include "../Backend.h"
#include "../../MG_State/State.h"
#include "../../MG_Impl/Log.h"

#include <cstring>
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
    vkQueueSubmit(b->graphicsQueue, 1, &si, c.fence);
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
    auto& tbl = texture_table();
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
    if (!b->initialized || !out_pixels || w <= 0 || h <= 0) return 0;

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

    if (g_state->currentDrawFBO == 0) {
        // EGL default framebuffer: read directly from the swapchain image.
        // The EGL layer installs both the VkImageView and the underlying
        // VkImage + format on g_state when a surface is made current. The
        // swapchain image is in COLOR_ATTACHMENT_OPTIMAL after a render pass
        // (transitioned to PRESENT_SRC_KHR only at present).
        src_image = g_state->eglDefaultColorImage;
        src_fmt   = g_state->eglDefaultColorFormat;
        src_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else {
        mithril::Framebuffer* fbo = mithril::state_get_framebuffer(g_state->currentDrawFBO);
        if (!fbo || !fbo->colors[0].texture) return 0;
        src_tex_id = fbo->colors[0].texture;
        src_image = backend_get_texture_image(src_tex_id);
        mithril::Texture* t = mithril::state_get_texture(src_tex_id);
        if (t) src_fmt = gl_internal_to_vk((GLenum)t->internalFormat);
    }
    if (src_image == VK_NULL_HANDLE) return 0;

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
    if (g_state->currentDrawFBO == 0) {
        // Swapchain image stays in COLOR_ATTACHMENT_OPTIMAL after a render pass
        // (transitioned to PRESENT_SRC_KHR only at present).
        src_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else {
        mithril::Framebuffer* fbo = mithril::state_get_framebuffer(g_state->currentDrawFBO);
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
    auto& tbl = texture_table();
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
        dstY0 = dst_height - dstY0;
        dstY1 = dst_height - dstY1;
    }

    OneShotCtx c;
    if (!begin_one_shot(c)) return;

    VkImageAspectFlags srcAspect = aspect_for_format(src_format);
    VkImageAspectFlags dstAspect = aspect_for_format(dst_format);

    // Transition source to TRANSFER_SRC_OPTIMAL.
    VkImageMemoryBarrier sb{};
    sb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    sb.srcAccessMask = (src_initial == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                        : VK_ACCESS_SHADER_READ_BIT;
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
    VkPipelineStageFlags srcStage = (src_initial == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    vkCmdPipelineBarrier(c.cmd, srcStage,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &sb);

    // Transition destination to TRANSFER_DST_OPTIMAL.
    VkImageMemoryBarrier db{};
    db.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    db.srcAccessMask = (dst_initial == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                        : VK_ACCESS_SHADER_READ_BIT;
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
    VkPipelineStageFlags dstStage = (dst_initial == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                        ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    vkCmdPipelineBarrier(c.cmd, dstStage,
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

    end_one_shot(c);
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

void backend_blit_texture(GLuint src_name, GLuint dst_name,
                          int srcX0, int srcY0, int srcX1, int srcY1,
                          int dstX0, int dstY0, int dstX1, int dstY1,
                          GLbitfield mask, GLenum filter) {
    mithril::vk::blit_texture(src_name, dst_name,
                              srcX0, srcY0, srcX1, srcY1,
                              dstX0, dstY0, dstX1, dstY1,
                              mask, filter);
}

void backend_blit_images(VkImage src_image, VkFormat src_format,
                         VkImage dst_image, VkFormat dst_format,
                         int srcX0, int srcY0, int srcX1, int srcY1,
                         int dstX0, int dstY0, int dstX1, int dstY1,
                         GLbitfield mask, GLenum filter,
                         int is_dst_default_fbo, int dst_height) {
    // Both images are assumed to be in a sampling or attachment layout before
    // the blit. The swapchain color image is in COLOR_ATTACHMENT_OPTIMAL
    // (it was just rendered into, or will be rendered into next frame); user
    // FBO textures are in SHADER_READ_ONLY_OPTIMAL (after an upload or a
    // previous blit). We transition them back to those same layouts after the
    // blit so subsequent rendering / sampling continues to work.
    //
    // Heuristic: if the format is a color format (not depth/stencil), assume
    // COLOR_ATTACHMENT_OPTIMAL for the initial/final layout. This matches the
    // common glBlitFramebuffer case (blitting between render targets that are
    // actively being rendered into). For depth/stencil formats we would use
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL, but depth blits are not yet supported.
    VkImageLayout src_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkImageLayout dst_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    mithril::vk::blit_images_impl(src_image, src_format, src_layout, src_layout,
                                  dst_image, dst_format, dst_layout, dst_layout,
                                  srcX0, srcY0, srcX1, srcY1,
                                  dstX0, dstY0, dstX1, dstY1,
                                  mask, filter,
                                  is_dst_default_fbo != 0, dst_height);
}

} // extern "C"
