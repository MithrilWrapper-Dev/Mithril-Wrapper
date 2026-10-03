// Mithril-Wrapper - MG_Backend/DirectVulkan/RenderPassCompat.h
//
// 传统 VkRenderPass / VkFramebuffer 兼容层（无 VK_KHR_dynamic_rendering 时的退路）。
//
// 背景：Mithril 的命令录制原本完全依赖 VK_KHR_dynamic_rendering
// (vkCmdBeginRendering / vkCmdEndRendering)。该扩展在 Vulkan 1.3 才进核心，
// 且 mesa/turnip 的 Android HAL 构建、以及 Adreno 619 上的系统驱动
// (Vulkan 1.1.128) 都不提供它。原本的做法是「检测不到就拒绝启动」，
// 于是这些设备永远起不来。
//
// Zink 和 MobileGL 都有传统 VkRenderPass + VkFramebuffer 路径，所以它们不受
// 这个限制。本模块为 Mithril 补上同一条路：按附件签名缓存 VkRenderPass，
// 按 (views, 尺寸) 缓存 VkFramebuffer。
//
// 兼容性依据（Vulkan spec "Render Pass Compatibility"）：两个 render pass
// 若对应的附件引用在 format / sampleCount 上一致，且除 initialLayout、
// finalLayout、loadOp、storeOp、附件引用里的 layout 之外完全相同，则互相兼容。
// 因此：
//   * 管线用「规范」pass（LOAD/STORE）创建
//   * 运行时用同一签名下不同 loadOp/storeOp 的 pass 开启
// 是合法的，缓存也不会因为 loadOp 组合而碎片化到不可控。
#ifndef MITHRIL_DIRECTVULKAN_RENDERPASSCOMPAT_H
#define MITHRIL_DIRECTVULKAN_RENDERPASSCOMPAT_H

#include <vulkan/vulkan.h>

namespace mithril {
namespace vk {

// 取得（或创建）一个传统 VkRenderPass。
//
// color_load / color_store 传 nullptr 时使用 LOAD/STORE —— 即管线创建所需的
// 「规范」形态。显式传入时按传入值创建（draw 时的实际 loadOp/storeOp）。
//
// 返回 VK_NULL_HANDLE 表示设备不可用或创建失败；调用方必须据此回退。
VkRenderPass mithril_vk_render_pass_for(const VkFormat* color_formats,
                                        int color_count,
                                        VkFormat depth_format,
                                        VkSampleCountFlagBits samples,
                                        const VkAttachmentLoadOp* color_load,
                                        const VkAttachmentStoreOp* color_store,
                                        VkAttachmentLoadOp depth_load,
                                        VkAttachmentStoreOp depth_store);

// 取得（或创建）一个 VkFramebuffer。内部用一个同签名的兼容 pass 创建它，
// 因此可与 mithril_vk_render_pass_for 返回的任意同签名 pass 配合使用。
VkFramebuffer mithril_vk_framebuffer_for(VkRenderPass pass,
                                         const VkImageView* color_views,
                                         int color_count,
                                         VkImageView depth_view,
                                         uint32_t width,
                                         uint32_t height);

// 销毁全部缓存的 VkFramebuffer / VkRenderPass。必须在 vkDestroyDevice 之前调用。
void mithril_vk_destroy_render_pass_cache();

} // namespace vk
} // namespace mithril

#endif // MITHRIL_DIRECTVULKAN_RENDERPASSCOMPAT_H
