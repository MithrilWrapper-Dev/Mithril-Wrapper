// RenderPassCompat.h — classic VkRenderPass / VkFramebuffer fallback.
//
// Why this exists: Mithril's command recording has exactly one path,
// vkCmdBeginRendering, which requires VK_KHR_dynamic_rendering (or Vulkan 1.3,
// where it is core). MobileGL does not need that extension because it keeps a
// traditional VkRenderPass + VkFramebuffer path, which is core Vulkan 1.0.
//
// This module supplies that path so the dynamic-rendering requirement can be
// dropped from a hard startup gate to a soft preference. Without it, merely
// removing the gate produces the worst possible failure mode: begin_render_pass
// would silently skip rendering (the `if (fn)` guard), leaving the pass marked
// active and every draw recorded against no attachments at all — a black screen
// with no validation error and no log line.
//
// Compatibility note (this is what makes the cache correct): two render passes
// are pipeline-compatible when their attachment *formats*, sample counts and
// attachment counts match. loadOp / storeOp do NOT participate. So the
// pipelines are always built against a canonical LOAD/STORE pass derived purely
// from the format signature, while begin_render_pass instantiates a separate,
// differently-loading pass per frame. Binding a pipeline created against the
// canonical one inside the other is legal.

#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>
#include <cstdint>

namespace mithril {
namespace vk {

// Maximum color attachments, matching EncoderState::colorViews.
constexpr uint32_t kMaxRpColorAttachments = 8;

struct RenderPassKey {
    VkFormat colorFormats[kMaxRpColorAttachments] = {};
    uint32_t colorCount = 0;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    // Set when depthFormat carries a stencil aspect; the depth and stencil of a
    // packed format are a single Vulkan attachment, so this never becomes a
    // second entry.
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    VkAttachmentLoadOp colorLoadOps[kMaxRpColorAttachments] = {};
    VkAttachmentStoreOp colorStoreOps[kMaxRpColorAttachments] = {};
    VkAttachmentLoadOp depthLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    VkAttachmentStoreOp depthStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkAttachmentLoadOp stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    VkAttachmentStoreOp stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;

    bool operator==(const RenderPassKey& o) const;
};

struct RenderPassKeyHash {
    std::size_t operator()(const RenderPassKey& k) const;
};

struct FramebufferKey {
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkImageView views[kMaxRpColorAttachments + 1] = {};  // colors, then depth
    uint32_t viewCount = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t layers = 1;

    bool operator==(const FramebufferKey& o) const;
};

struct FramebufferKeyHash {
    std::size_t operator()(const FramebufferKey& k) const;
};

// Canonical pass for a format signature: every attachment LOAD/STORE. Used for
// pipeline creation, where loadOp is irrelevant (and must not fragment the
// pipeline cache).
VkRenderPass get_or_create_canonical_render_pass(const VkFormat* colorFormats,
                                                 uint32_t colorCount,
                                                 VkFormat depthFormat,
                                                 VkSampleCountFlagBits samples);

// A pass with concrete load/store ops, for vkCmdBeginRenderPass.
VkRenderPass get_or_create_render_pass(const RenderPassKey& key);

VkFramebuffer get_or_create_framebuffer(VkRenderPass renderPass,
                                        const VkImageView* views, uint32_t viewCount,
                                        uint32_t width, uint32_t height);

// Drops every cached pass and framebuffer. Called on device loss / shutdown,
// after which no VkRenderPass handle this module handed out may be used again.
void clear_render_pass_caches();

// Diagnostics: number of live cached passes / framebuffers.
void render_pass_cache_stats(uint32_t* outPasses, uint32_t* outFramebuffers);

// A render pass needs each attachment's VkFormat, but begin_render_pass only
// has VkImageViews. This resolves one, memoised: the first lookup for a given
// view scans the texture table, later ones are a hash probe. Returns
// VK_FORMAT_UNDEFINED when the view is unknown, in which case the caller must
// not build a pass - an attachment with an undefined format is a hard error.
VkFormat lookup_view_format(VkImageView view);

}  // namespace vk
}  // namespace mithril
