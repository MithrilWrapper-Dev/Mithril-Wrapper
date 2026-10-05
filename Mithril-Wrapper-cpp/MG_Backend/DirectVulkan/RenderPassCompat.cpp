// Mithril-Wrapper - MG_Backend/DirectVulkan/RenderPassCompat.cpp
//
// 传统 VkRenderPass / VkFramebuffer 的实现。仅在设备不支持
// VK_KHR_dynamic_rendering 时被调用（Vulkan 1.0/1.1，或 1.2 但无该扩展）。
//
// 所有用到的 Vulkan 入口（vkCreateRenderPass / vkCreateFramebuffer /
// vkCmdBeginRenderPass…）都是 Vulkan 1.0 核心，任何驱动都有，
// 因此这条路径不依赖任何扩展。
#include "RenderPassCompat.h"
#include "Device.h"
#include "../../MG_Impl/Log.h"

#include <cstring>
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace mithril {
namespace vk {

namespace {

constexpr int kMaxCompatAttachments = 8;

// 缓存键：附件签名（格式 + 采样数 + 各附件的 loadOp/storeOp）。
// loadOp/storeOp 参与键，是为了让 draw 时能拿到语义正确的 pass；
// 它们不影响 render pass 之间的「兼容性」，所以管线仍可用规范 pass 创建。
struct PassKey {
    VkFormat            colorFormats[kMaxCompatAttachments];
    VkAttachmentLoadOp  colorLoad[kMaxCompatAttachments];
    VkAttachmentStoreOp colorStore[kMaxCompatAttachments];
    VkFormat            depthFormat;
    VkSampleCountFlagBits samples;
    VkAttachmentLoadOp  depthLoad;
    VkAttachmentStoreOp depthStore;
    // 附件「实际处于」的初始布局（MobileGL 模型）。参与缓存键：同一签名、
    // 不同初始布局的两个 pass 是不同的对象，不能复用。
    VkImageLayout       colorInitial[kMaxCompatAttachments];
    VkImageLayout       depthInitial;
    int                 colorCount;

    bool operator==(const PassKey& o) const {
        if (colorCount != o.colorCount) return false;
        if (depthFormat != o.depthFormat) return false;
        if (samples != o.samples) return false;
        if (depthLoad != o.depthLoad) return false;
        if (depthStore != o.depthStore) return false;
        if (depthInitial != o.depthInitial) return false;
        for (int i = 0; i < colorCount; ++i) {
            if (colorInitial[i] != o.colorInitial[i]) return false;
            if (colorFormats[i] != o.colorFormats[i]) return false;
            if (colorLoad[i] != o.colorLoad[i]) return false;
            if (colorStore[i] != o.colorStore[i]) return false;
        }
        return true;
    }
};

struct PassKeyHash {
    size_t operator()(const PassKey& k) const {
        uint64_t h = 1469598103934665603ULL;
        auto mixByte = [&](uint8_t v) { h ^= v; h *= 1099511628211ULL; };
        auto mixU32 = [&](uint32_t v) {
            for (int i = 0; i < 4; ++i) mixByte((uint8_t)((v >> (i * 8)) & 0xff));
        };
        mixU32((uint32_t)k.colorCount);
        mixU32((uint32_t)k.depthFormat);
        mixU32((uint32_t)k.samples);
        mixU32((uint32_t)k.depthLoad);
        mixU32((uint32_t)k.depthStore);
        mixU32((uint32_t)k.depthInitial);
        for (int i = 0; i < k.colorCount; ++i) {
            mixU32((uint32_t)k.colorFormats[i]);
            mixU32((uint32_t)k.colorInitial[i]);
            mixU32((uint32_t)k.colorLoad[i]);
            mixU32((uint32_t)k.colorStore[i]);
        }
        return (size_t)h;
    }
};

// Framebuffer 缓存键：视图集合 + 尺寸（同一组视图在不同尺寸下要不同的 fb）。
struct FbKey {
    VkImageView views[kMaxCompatAttachments + 1];  // 颜色 + 最后一个是 depth
    uint32_t    width;
    uint32_t    height;
    int         colorCount;

    bool operator==(const FbKey& o) const {
        if (width != o.width || height != o.height) return false;
        if (colorCount != o.colorCount) return false;
        for (int i = 0; i <= colorCount; ++i) {
            if (views[i] != o.views[i]) return false;
        }
        return true;
    }
};

struct FbKeyHash {
    size_t operator()(const FbKey& k) const {
        uint64_t h = 1469598103934665603ULL;
        auto mixU64 = [&](uint64_t v) {
            for (int i = 0; i < 8; ++i) { h ^= (uint8_t)((v >> (i * 8)) & 0xff); h *= 1099511628211ULL; }
        };
        mixU64(k.width);
        mixU64(k.height);
        mixU64((uint64_t)k.colorCount);
        for (int i = 0; i <= k.colorCount; ++i) mixU64((uint64_t)(uintptr_t)k.views[i]);
        return (size_t)h;
    }
};

std::unordered_map<PassKey, VkRenderPass, PassKeyHash>& pass_cache() {
    static std::unordered_map<PassKey, VkRenderPass, PassKeyHash> m;
    return m;
}

std::unordered_map<FbKey, VkFramebuffer, FbKeyHash>& fb_cache() {
    static std::unordered_map<FbKey, VkFramebuffer, FbKeyHash> m;
    return m;
}

} // namespace

VkRenderPass mithril_vk_render_pass_for(const VkFormat* color_formats,
                                        int color_count,
                                        VkFormat depth_format,
                                        VkSampleCountFlagBits samples,
                                        const VkAttachmentLoadOp* color_load,
                                        const VkAttachmentStoreOp* color_store,
                                        VkAttachmentLoadOp depth_load,
                                        VkAttachmentStoreOp depth_store,
                                        const VkImageLayout* color_initial_layouts,
                                        VkImageLayout depth_initial_layout) {
    Backend* b = backend();
    if (!b || !b->device) return VK_NULL_HANDLE;
    if (color_count < 0) color_count = 0;
    if (color_count > kMaxCompatAttachments) color_count = kMaxCompatAttachments;
    if (samples == 0) samples = VK_SAMPLE_COUNT_1_BIT;

    PassKey key{};
    key.colorCount = color_count;
    key.depthFormat = depth_format;
    key.samples = samples;
    key.depthLoad = depth_load;
    key.depthStore = depth_store;
    key.depthInitial = color_initial_layouts ? depth_initial_layout
                                             : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    for (int i = 0; i < kMaxCompatAttachments; ++i) {
        key.colorFormats[i] = VK_FORMAT_UNDEFINED;
        key.colorLoad[i] = VK_ATTACHMENT_LOAD_OP_LOAD;
        key.colorStore[i] = VK_ATTACHMENT_STORE_OP_STORE;
        key.colorInitial[i] = color_initial_layouts
            ? color_initial_layouts[i]
            : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    for (int i = 0; i < color_count; ++i) {
        key.colorFormats[i] = color_formats ? color_formats[i] : VK_FORMAT_UNDEFINED;
        key.colorLoad[i] = color_load ? color_load[i] : VK_ATTACHMENT_LOAD_OP_LOAD;
        key.colorStore[i] = color_store ? color_store[i] : VK_ATTACHMENT_STORE_OP_STORE;
    }

    auto it = pass_cache().find(key);
    if (it != pass_cache().end()) return it->second;

    // 附件初始布局 —— 对齐 MobileGL VkRenderPassManager.cpp 的
    // `rbDesc.initialLayout = (rbHasClear || trackedLayout == UNDEFINED)
    //      ? UNDEFINED : trackedLayout;`
    //
    // 之前这里无条件写 COLOR_ATTACHMENT_OPTIMAL，等于断言「begin_render_pass
    // 的 barrier 一定跑过了」。但那条 barrier 是条件性的：它要求附件被识别为
    // 交换链当前视图（currentImage >= 0，重建/恢复窗口内为 -1）或纹理表条目
    // （要求 GL 层注册过）。两条都不成立时图像实际仍在 PRESENT_SRC /
    // SHADER_READ_ONLY / UNDEFINED，而 pass 声明 COLOR_ATTACHMENT_OPTIMAL
    // —— tiler（Adreno）于是从错误的状态做 tile load：纯黑，无报错、无校验层
    // 提示。声明真实布局则永远合法；UNDEFINED 配 DONT_CARE/CLEAR 由调用方保证。
    VkAttachmentDescription attachments[kMaxCompatAttachments + 1];
    VkAttachmentReference colorRefs[kMaxCompatAttachments];
    VkAttachmentReference depthRef{};
    uint32_t attachmentCount = 0;
    bool hasDepth = (depth_format != VK_FORMAT_UNDEFINED);

    for (int i = 0; i < color_count; ++i) {
        VkAttachmentDescription& a = attachments[attachmentCount];
        a = {};
        a.flags = 0;
        a.format = key.colorFormats[i];
        a.samples = samples;
        a.loadOp = key.colorLoad[i];
        a.storeOp = key.colorStore[i];
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout = key.colorInitial[i];
        a.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorRefs[i].attachment = attachmentCount;
        colorRefs[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ++attachmentCount;
    }
    if (hasDepth) {
        VkAttachmentDescription& a = attachments[attachmentCount];
        a = {};
        a.flags = 0;
        a.format = depth_format;
        a.samples = samples;
        a.loadOp = depth_load;
        a.storeOp = depth_store;
        // 深度附件带 stencil 时不能用 DONT_CARE 丢弃 stencil，否则
        // D32_SFLOAT_S8_UINT 上的 stencil 内容未定义。统一 LOAD/STORE 最稳。
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
        a.initialLayout = key.depthInitial;
        a.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthRef.attachment = attachmentCount;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        ++attachmentCount;
    }

    if (attachmentCount == 0) return VK_NULL_HANDLE;  // 无附件：无意义的 pass

    VkSubpassDescription subpass{};
    subpass.flags = 0;
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = nullptr;
    subpass.colorAttachmentCount = (uint32_t)color_count;
    subpass.pColorAttachments = color_count > 0 ? colorRefs : nullptr;
    subpass.pResolveAttachments = nullptr;
    subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = nullptr;

    // 显式依赖：外部 → subpass 0（等屏障与附件写入可见），
    // subpass 0 → 外部（让后续 present 屏障看到结果）。
    // begin_render_pass 记录的布局屏障在 pass 之外，靠这两条依赖串起来。
    VkSubpassDependency deps[2];
    deps[0] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[0].dependencyFlags = 0;

    deps[1] = {};
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    // 对齐 MobileGL：dstStageMask 用真实阶段而不是 BOTTOM_OF_PIPE。
    // BOTTOM_OF_PIPE 意味着「没有阶段等待」，这条依赖实际上不起排序作用；
    // 而我们的整帧就是 pass -> end -> blit -> present 的链条，缺的正是
    // TRANSFER_BIT —— 即把结果搬到交换链的那一步。
    deps[1].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT;
    deps[1].dependencyFlags = 0;

    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = attachmentCount;
    ci.pAttachments = attachments;
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 2;
    ci.pDependencies = deps;

    VkRenderPass pass = VK_NULL_HANDLE;
    VkResult r = vkCreateRenderPass(b->device, &ci, nullptr, &pass);
    if (r != VK_SUCCESS) {
        MITHRIL_LOG_WARN("vk", "RenderPassCompat: vkCreateRenderPass failed (%d)", (int)r);
        return VK_NULL_HANDLE;
    }
    pass_cache().emplace(key, pass);
    return pass;
}

VkFramebuffer mithril_vk_framebuffer_for(VkRenderPass pass,
                                         const VkImageView* color_views,
                                         int color_count,
                                         VkImageView depth_view,
                                         uint32_t width,
                                         uint32_t height) {
    Backend* b = backend();
    if (!b || !b->device || pass == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    if (color_count < 0) color_count = 0;
    if (color_count > kMaxCompatAttachments) color_count = kMaxCompatAttachments;
    if (width == 0 || height == 0) return VK_NULL_HANDLE;

    FbKey key{};
    key.width = width;
    key.height = height;
    key.colorCount = color_count;
    for (int i = 0; i <= kMaxCompatAttachments; ++i) key.views[i] = VK_NULL_HANDLE;
    for (int i = 0; i < color_count; ++i) {
        key.views[i] = color_views ? color_views[i] : VK_NULL_HANDLE;
    }
    key.views[color_count] = depth_view;

    auto it = fb_cache().find(key);
    if (it != fb_cache().end()) return it->second;

    VkImageView attachments[kMaxCompatAttachments + 1];
    uint32_t count = 0;
    for (int i = 0; i < color_count; ++i) {
        if (!color_views || color_views[i] == VK_NULL_HANDLE) {
            // 该槽位未绑定：传统 render pass 不允许 NULL 附件，
            // 交给调用方保证 —— 这里直接放弃，回退到动态渲染/跳过。
            return VK_NULL_HANDLE;
        }
        attachments[count++] = color_views[i];
    }
    if (depth_view != VK_NULL_HANDLE) attachments[count++] = depth_view;
    if (count == 0) return VK_NULL_HANDLE;

    VkFramebufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.renderPass = pass;
    ci.attachmentCount = count;
    ci.pAttachments = attachments;
    ci.width = width;
    ci.height = height;
    ci.layers = 1;

    VkFramebuffer fb = VK_NULL_HANDLE;
    VkResult r = vkCreateFramebuffer(b->device, &ci, nullptr, &fb);
    if (r != VK_SUCCESS) {
        MITHRIL_LOG_WARN("vk", "RenderPassCompat: vkCreateFramebuffer failed (%d)", (int)r);
        return VK_NULL_HANDLE;
    }
    fb_cache().emplace(key, fb);
    return fb;
}

void mithril_vk_destroy_render_pass_cache() {
    Backend* b = backend();
    if (!b || !b->device) {
        fb_cache().clear();
        pass_cache().clear();
        return;
    }
    for (auto& kv : fb_cache()) {
        if (kv.second) vkDestroyFramebuffer(b->device, kv.second, nullptr);
    }
    fb_cache().clear();
    for (auto& kv : pass_cache()) {
        if (kv.second) vkDestroyRenderPass(b->device, kv.second, nullptr);
    }
    pass_cache().clear();
}

} // namespace vk
} // namespace mithril
