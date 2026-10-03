// RenderPassCompat.cpp — see RenderPassCompat.h for why this exists.

#include "RenderPassCompat.h"

#include "Device.h"
#include "Resources.h"

#include <cstdio>
#include <unordered_map>
#include <vector>

namespace mithril {
namespace vk {

namespace {

bool format_has_stencil(VkFormat fmt) {
    switch (fmt) {
        case VK_FORMAT_S8_UINT:
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        default:
            return false;
    }
}

// FNV-1a over a byte range; stable and dependency-free.
void hash_bytes(std::size_t& h, const void* p, std::size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<std::size_t>(b[i]);
        h *= static_cast<std::size_t>(1099511628211ull);
    }
}

struct PassCache {
    std::unordered_map<RenderPassKey, VkRenderPass, RenderPassKeyHash> passes;
    std::unordered_map<FramebufferKey, VkFramebuffer, FramebufferKeyHash> framebuffers;
};

PassCache& cache() {
    static PassCache c;
    return c;
}

}  // namespace

bool RenderPassKey::operator==(const RenderPassKey& o) const {
    if (colorCount != o.colorCount) return false;
    if (depthFormat != o.depthFormat) return false;
    if (samples != o.samples) return false;
    if (depthLoadOp != o.depthLoadOp || depthStoreOp != o.depthStoreOp) return false;
    if (stencilLoadOp != o.stencilLoadOp || stencilStoreOp != o.stencilStoreOp) return false;
    for (uint32_t i = 0; i < colorCount; ++i) {
        if (colorFormats[i] != o.colorFormats[i]) return false;
        if (colorLoadOps[i] != o.colorLoadOps[i]) return false;
        if (colorStoreOps[i] != o.colorStoreOps[i]) return false;
    }
    return true;
}

std::size_t RenderPassKeyHash::operator()(const RenderPassKey& k) const {
    std::size_t h = static_cast<std::size_t>(1469598103934665603ull);
    hash_bytes(h, &k.colorCount, sizeof(k.colorCount));
    hash_bytes(h, &k.depthFormat, sizeof(k.depthFormat));
    hash_bytes(h, &k.samples, sizeof(k.samples));
    hash_bytes(h, &k.depthLoadOp, sizeof(k.depthLoadOp));
    hash_bytes(h, &k.depthStoreOp, sizeof(k.depthStoreOp));
    hash_bytes(h, &k.stencilLoadOp, sizeof(k.stencilLoadOp));
    hash_bytes(h, &k.stencilStoreOp, sizeof(k.stencilStoreOp));
    for (uint32_t i = 0; i < k.colorCount; ++i) {
        hash_bytes(h, &k.colorFormats[i], sizeof(k.colorFormats[i]));
        hash_bytes(h, &k.colorLoadOps[i], sizeof(k.colorLoadOps[i]));
        hash_bytes(h, &k.colorStoreOps[i], sizeof(k.colorStoreOps[i]));
    }
    return h;
}

bool FramebufferKey::operator==(const FramebufferKey& o) const {
    if (renderPass != o.renderPass) return false;
    if (viewCount != o.viewCount) return false;
    if (width != o.width || height != o.height) return false;
    if (layers != o.layers) return false;
    for (uint32_t i = 0; i < viewCount; ++i) {
        if (views[i] != o.views[i]) return false;
    }
    return true;
}

std::size_t FramebufferKeyHash::operator()(const FramebufferKey& k) const {
    std::size_t h = static_cast<std::size_t>(1469598103934665603ull);
    hash_bytes(h, &k.renderPass, sizeof(k.renderPass));
    hash_bytes(h, &k.viewCount, sizeof(k.viewCount));
    hash_bytes(h, &k.width, sizeof(k.width));
    hash_bytes(h, &k.height, sizeof(k.height));
    hash_bytes(h, &k.layers, sizeof(k.layers));
    for (uint32_t i = 0; i < k.viewCount; ++i) {
        hash_bytes(h, &k.views[i], sizeof(k.views[i]));
    }
    return h;
}

namespace {

VkRenderPass create_render_pass(const RenderPassKey& key) {
    Backend* b = backend();
    if (!b || !b->initialized || b->device == VK_NULL_HANDLE) return VK_NULL_HANDLE;

    const bool hasDepth = key.depthFormat != VK_FORMAT_UNDEFINED;
    const bool hasStencil = hasDepth && format_has_stencil(key.depthFormat);

    VkAttachmentDescription attachments[kMaxRpColorAttachments + 1] = {};
    uint32_t attachmentCount = 0;

    for (uint32_t i = 0; i < key.colorCount; ++i) {
        VkAttachmentDescription& a = attachments[attachmentCount++];
        a.flags = 0;
        a.format = key.colorFormats[i];
        a.samples = key.samples;
        a.loadOp = key.colorLoadOps[i];
        a.storeOp = key.colorStoreOps[i];
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // The caller transitions attachments to COLOR_ATTACHMENT_OPTIMAL before
        // beginning the pass (begin_render_pass records those barriers), so
        // that is the honest initial layout here. Declaring UNDEFINED instead
        // would discard the contents the LOAD ops are meant to preserve.
        a.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        a.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if (hasDepth) {
        VkAttachmentDescription& a = attachments[attachmentCount++];
        a.flags = 0;
        a.format = key.depthFormat;
        a.samples = key.samples;
        a.loadOp = key.depthLoadOp;
        a.storeOp = key.depthStoreOp;
        a.stencilLoadOp = hasStencil ? key.stencilLoadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = hasStencil ? key.stencilStoreOp : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        a.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkAttachmentReference colorRefs[kMaxRpColorAttachments] = {};
    for (uint32_t i = 0; i < key.colorCount; ++i) {
        colorRefs[i].attachment = i;
        colorRefs[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    VkAttachmentReference depthRef{};
    depthRef.attachment = hasDepth ? key.colorCount : VK_ATTACHMENT_UNUSED;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.flags = 0;
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = nullptr;
    subpass.colorAttachmentCount = key.colorCount;
    subpass.pColorAttachments = key.colorCount > 0 ? colorRefs : nullptr;
    subpass.pResolveAttachments = nullptr;
    subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = nullptr;

    // The attachments are already in their optimal layouts when the pass
    // starts, so this dependency only orders the writes: it makes the previous
    // pass's color/depth writes visible to this pass's reads and writes.
    // Without it the load of a LOAD attachment races the store that produced it.
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dep.dependencyFlags = 0;

    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.pNext = nullptr;
    ci.flags = 0;
    ci.attachmentCount = attachmentCount;
    ci.pAttachments = attachmentCount > 0 ? attachments : nullptr;
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;

    VkRenderPass rp = VK_NULL_HANDLE;
    const VkResult r = vkCreateRenderPass(b->device, &ci, nullptr, &rp);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[mithril] render-pass: vkCreateRenderPass failed (%d)\n", (int)r);
        return VK_NULL_HANDLE;
    }
    return rp;
}

}  // namespace

VkRenderPass get_or_create_render_pass(const RenderPassKey& key) {
    PassCache& c = cache();
    auto it = c.passes.find(key);
    if (it != c.passes.end()) return it->second;

    VkRenderPass rp = create_render_pass(key);
    if (rp == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    c.passes.emplace(key, rp);
    return rp;
}

VkRenderPass get_or_create_canonical_render_pass(const VkFormat* colorFormats,
                                                 uint32_t colorCount,
                                                 VkFormat depthFormat,
                                                 VkSampleCountFlagBits samples) {
    RenderPassKey key;
    key.colorCount = colorCount;
    if (colorCount > kMaxRpColorAttachments) colorCount = kMaxRpColorAttachments;
    for (uint32_t i = 0; i < colorCount; ++i) {
        key.colorFormats[i] = colorFormats[i];
        key.colorLoadOps[i] = VK_ATTACHMENT_LOAD_OP_LOAD;
        key.colorStoreOps[i] = VK_ATTACHMENT_STORE_OP_STORE;
    }
    key.colorCount = colorCount;
    key.depthFormat = depthFormat;
    key.samples = samples;
    key.depthLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    key.depthStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    key.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    key.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    return get_or_create_render_pass(key);
}

VkFramebuffer get_or_create_framebuffer(VkRenderPass renderPass,
                                        const VkImageView* views, uint32_t viewCount,
                                        uint32_t width, uint32_t height) {
    if (renderPass == VK_NULL_HANDLE || viewCount == 0) return VK_NULL_HANDLE;
    Backend* b = backend();
    if (!b || !b->initialized || b->device == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    if (viewCount > kMaxRpColorAttachments + 1) return VK_NULL_HANDLE;

    FramebufferKey key;
    key.renderPass = renderPass;
    key.viewCount = viewCount;
    key.width = width;
    key.height = height;
    key.layers = 1;
    for (uint32_t i = 0; i < viewCount; ++i) key.views[i] = views[i];

    PassCache& c = cache();
    auto it = c.framebuffers.find(key);
    if (it != c.framebuffers.end()) return it->second;

    VkFramebufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.pNext = nullptr;
    ci.flags = 0;
    ci.renderPass = renderPass;
    ci.attachmentCount = viewCount;
    ci.pAttachments = views;
    ci.width = width;
    ci.height = height;
    ci.layers = 1;

    VkFramebuffer fb = VK_NULL_HANDLE;
    const VkResult r = vkCreateFramebuffer(b->device, &ci, nullptr, &fb);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[mithril] render-pass: vkCreateFramebuffer failed (%d)\n", (int)r);
        return VK_NULL_HANDLE;
    }
    c.framebuffers.emplace(key, fb);
    return fb;
}

void clear_render_pass_caches() {
    Backend* b = backend();
    VkDevice dev = (b && b->initialized) ? b->device : VK_NULL_HANDLE;
    PassCache& c = cache();
    // Framebuffers reference their render pass, so they must go first.
    if (dev != VK_NULL_HANDLE) {
        for (auto& kv : c.framebuffers) {
            if (kv.second != VK_NULL_HANDLE) vkDestroyFramebuffer(dev, kv.second, nullptr);
        }
        for (auto& kv : c.passes) {
            if (kv.second != VK_NULL_HANDLE) vkDestroyRenderPass(dev, kv.second, nullptr);
        }
    }
    c.framebuffers.clear();
    c.passes.clear();
}

void render_pass_cache_stats(uint32_t* outPasses, uint32_t* outFramebuffers) {
    PassCache& c = cache();
    if (outPasses) *outPasses = static_cast<uint32_t>(c.passes.size());
    if (outFramebuffers) *outFramebuffers = static_cast<uint32_t>(c.framebuffers.size());
}

VkFormat lookup_view_format(VkImageView view) {
    if (view == VK_NULL_HANDLE) return VK_FORMAT_UNDEFINED;

    static std::unordered_map<VkImageView, VkFormat> memo;
    auto it = memo.find(view);
    if (it != memo.end()) return it->second;

    // Miss. Scan the texture table once for this view. This only ever happens
    // for a view the classic path has not seen before, so it is not on the
    // steady-state per-frame cost.
    for (const auto& kv : texture_table()) {
        if (kv.second.view == view) {
            memo.emplace(view, kv.second.format);
            return kv.second.format;
        }
    }

    // Unknown view (a swapchain image view, which is not a texture-table entry,
    // or a renderbuffer). Callers compare against the active swapchain's views
    // first and only come here for genuinely untracked attachments.
    memo.emplace(view, VK_FORMAT_UNDEFINED);
    return VK_FORMAT_UNDEFINED;
}

}  // namespace vk
}  // namespace mithril
