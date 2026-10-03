// Mithril-Wrapper - MG_Backend/DirectVulkan/Pipeline.cpp
// VkShaderModule creation (from SPIR-V) + VkGraphicsPipeline caching keyed by
// a hash signature built from (program, vertex format, attachment formats,
// blend state, primitive mode). Uses VK_KHR_dynamic_rendering so pipelines
// are created against a VkPipelineRenderingCreateInfo instead of a VkRenderPass.
#include "Pipeline.h"
#include "Device.h"
#include "Resources.h"
#include "DescriptorSet.h"
#include "../Backend.h"
#include "../../MG_Impl/Log.h"
// FIX (root cause AF - Primitive Restart): 读取 g_state->primitiveRestart /
// primitiveRestartFixedIndex 以动态设置 ia.primitiveRestartEnable，并将其
// 纳入 hash_signature 缓存键。深度对照 MobileGL VulkanRenderer.cpp:3861-3877。
#include "../../MG_State/State.h"

#include <cstring>
#include <vector>

#include <spirv_cross.hpp>
// spirv_cross.hpp transitively pulls in SPIRV-Cross's bundled spirv.hpp,
// which defines the spv:: namespace (spv::DecorationLocation, etc.) used by
// reflect_vertex_input_locations below.

namespace mithril {
namespace vk {
VkShaderModule create_probe_fs_module();

std::unordered_map<GLuint, ProgramResources>& program_table() {
    static std::unordered_map<GLuint, ProgramResources> t;
    return t;
}

namespace {

// ---- GL blend factor -> VkBlendFactor ----
VkBlendFactor gl_blend_to_vk(GLenum f) {
    switch (f) {
        case GL_ZERO:                     return VK_BLEND_FACTOR_ZERO;
        case GL_ONE:                      return VK_BLEND_FACTOR_ONE;
        case GL_SRC_COLOR:                return VK_BLEND_FACTOR_SRC_COLOR;
        case GL_ONE_MINUS_SRC_COLOR:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case GL_DST_COLOR:                return VK_BLEND_FACTOR_DST_COLOR;
        case GL_ONE_MINUS_DST_COLOR:      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case GL_SRC_ALPHA:                return VK_BLEND_FACTOR_SRC_ALPHA;
        case GL_ONE_MINUS_SRC_ALPHA:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case GL_DST_ALPHA:                return VK_BLEND_FACTOR_DST_ALPHA;
        case GL_ONE_MINUS_DST_ALPHA:      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case GL_CONSTANT_COLOR:           return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case GL_ONE_MINUS_CONSTANT_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case GL_CONSTANT_ALPHA:           return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case GL_ONE_MINUS_CONSTANT_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        case GL_SRC_ALPHA_SATURATE:       return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case GL_SRC1_COLOR:               return VK_BLEND_FACTOR_SRC1_COLOR;
        case GL_ONE_MINUS_SRC1_COLOR:     return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
        case GL_SRC1_ALPHA:               return VK_BLEND_FACTOR_SRC1_ALPHA;
        case GL_ONE_MINUS_SRC1_ALPHA:     return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        default:                          return VK_BLEND_FACTOR_ONE;
    }
}

// ---- GL primitive mode -> VkPrimitiveTopology ----
VkPrimitiveTopology gl_prim_to_vk(GLenum m) {
    switch (m) {
        case GL_POINTS:         return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case GL_LINES:          return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case GL_LINE_STRIP:     return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case GL_LINE_LOOP:      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; // approx
        case GL_TRIANGLES:      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case GL_TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case GL_TRIANGLE_FAN:   return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        default:                return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

// ---- GL attribute type -> VkFormat ----
//
// FIX (根因 V - 3 分量顶点属性格式映射):
// Metal MTLVertexFormat 枚举对**非 float** 类型不含 3 分量变体
// （无 UChar3/Char3/UShort3/Short3/Half3，仅有 1/2/4 分量及 Float3）。
// MoltenVK 无法将 VK_FORMAT_R8G8B8_UNORM / R16G16B16_SFLOAT 等 3 分量顶点
// 格式映射到任何 MTLVertexFormat → vkCreateGraphicsPipelines 失败 → draw
// 被跳过 → 屏幕只剩 clear color（红屏）。
//
// 修复策略（最小影响域）：对 size==3 的**非 float** 类型，统一映射到对应的
// 4 分量 VkFormat（如 R8G8B8_UNORM→R8G8B8A8_UNORM）。shader 中 vec3 属性
// 仅取前 3 分量，第 4 分量从 stride 内的 padding 字节读取（Metal vertex
// fetch 对 padding 容忍）。不引入数据流重打包，仅改格式枚举。
//
// 保留不变：
//   - GL_FLOAT size==3 → R32G32B32_SFLOAT（Metal 支持 MTLVertexFormatFloat3）
//   - GL_DOUBLE size==3 → R64G64B64_SFLOAT（不动）
//   - GL_INT / GL_UNSIGNED_INT size==3 → R32G32B32_SINT/_UINT（Metal 支持
//     32-bit Int3/Uint3）
//   - GL_INT_2_10_10_10_REV / GL_UNSIGNED_INT_2_10_10_10_REV：A2B10G10R10
//     本身是 4 分量打包格式，无需转换
//
// 深度对照 MobileGL ConvertIntegerVertexStreamToFloat32 /
// RepackVertexStream (VulkanRenderer.cpp:602-666)：MobileGL 在 pipeline 创建
// 前将不支持格式的顶点数据流整体重打包。本修复采用更小影响域：仅改格式枚举，
// 不动顶点缓冲数据。
VkFormat attrib_type_to_vk_format(GLenum type, int size, bool normalized, bool integer) {
    if (integer) {
        switch (type) {
            // FIX (根因 V): UChar3 不存在于 MTLVertexFormat → 用 UChar4，
            // shader 取前 3 分量，第 4 分量读 stride 内 padding
            case GL_UNSIGNED_BYTE:  switch (size) { case 1: return VK_FORMAT_R8_UINT;   case 2: return VK_FORMAT_R8G8_UINT;   case 3: return VK_FORMAT_R8G8B8A8_UINT;   case 4: return VK_FORMAT_R8G8B8A8_UINT; }
            // GL_INT 32-bit：Metal 支持 Int3，保持不变
            case GL_INT:            switch (size) { case 1: return VK_FORMAT_R32_SINT;  case 2: return VK_FORMAT_R32G32_SINT; case 3: return VK_FORMAT_R32G32B32_SINT; case 4: return VK_FORMAT_R32G32B32A32_SINT; }
            // GL_UNSIGNED_INT 32-bit：Metal 支持 Uint3，保持不变
            case GL_UNSIGNED_INT:   switch (size) { case 1: return VK_FORMAT_R32_UINT;  case 2: return VK_FORMAT_R32G32_UINT; case 3: return VK_FORMAT_R32G32B32_UINT; case 4: return VK_FORMAT_R32G32B32A32_UINT; }
            // FIX (根因 V): Short3 不存在于 MTLVertexFormat → 用 Short4
            case GL_SHORT:          switch (size) { case 1: return VK_FORMAT_R16_SINT;  case 2: return VK_FORMAT_R16G16_SINT; case 3: return VK_FORMAT_R16G16B16A16_SINT; case 4: return VK_FORMAT_R16G16B16A16_SINT; }
            // FIX (根因 V): UShort3 不存在于 MTLVertexFormat → 用 UShort4
            case GL_UNSIGNED_SHORT: switch (size) { case 1: return VK_FORMAT_R16_UINT;  case 2: return VK_FORMAT_R16G16_UINT; case 3: return VK_FORMAT_R16G16B16A16_UINT; case 4: return VK_FORMAT_R16G16B16A16_UINT; }
            default: break;
        }
    }
    switch (type) {
        // GL_FLOAT：Metal 支持 MTLVertexFormatFloat3，保持 R32G32B32_SFLOAT
        case GL_FLOAT:          switch (size) { case 1: return VK_FORMAT_R32_SFLOAT;   case 2: return VK_FORMAT_R32G32_SFLOAT;   case 3: return VK_FORMAT_R32G32B32_SFLOAT;   case 4: return VK_FORMAT_R32G32B32A32_SFLOAT; }
        // FIX (根因 V): Half3 不存在于 MTLVertexFormat → 用 Half4
        case GL_HALF_FLOAT:     switch (size) { case 1: return VK_FORMAT_R16_SFLOAT;   case 2: return VK_FORMAT_R16G16_SFLOAT;   case 3: return VK_FORMAT_R16G16B16A16_SFLOAT;   case 4: return VK_FORMAT_R16G16B16A16_SFLOAT; }
        // ---- 已知限制 (P1)：GL_DOUBLE 顶点属性在 Apple 平台无法真正支持 ----
        //
        // Metal **完全没有** 64 位顶点格式：MTLVertexFormat 里没有 Double，
        // MSL 也不支持 double 类型。MoltenVK 无法映射 VK_FORMAT_R64*_SFLOAT，
        // vkCreateGraphicsPipelines 会失败 → 用到该属性的 draw 全部消失。
        //
        // 单纯把枚举换成 R32 是**错的**：缓冲区里每个分量占 8 字节，按 4 字节
        // 去取只会读到 double 的低半边，得到彻底的垃圾几何。要正确支持必须在
        // CPU 侧（或用 compute shader）把整条顶点流 double→float 重打包 ——
        // 这正是上游 MobileGL RepackVertexStream (VulkanRenderer.cpp:602-666)
        // 做的事，而本项目目前没有顶点流重打包基础设施。
        //
        // 现状判断：Minecraft / Sodium / Iris 的顶点格式只用 float / byte /
        // short / packed-2101010，从不使用 GL_DOUBLE 属性，因此这条路径在
        // 目标工作负载下不会被触发。保留 R64 映射（而不是伪装成 R32）是刻意
        // 选择：让它在管线创建时明确失败并留下日志，好过静默渲染出垃圾。
        //
        // TODO: 若将来要支持使用 double 属性的通用 GL 应用，需要先实现顶点流
        // 重打包，再把这里改成 R32 并在重打包层做转换。
        case GL_DOUBLE:
            MITHRIL_LOG_WARN("vk", "顶点属性使用 GL_DOUBLE（size=%d）——Metal 无 64 位"
                             "顶点格式，该管线将创建失败。需要顶点流重打包才能支持。",
                             size);
            switch (size) { case 1: return VK_FORMAT_R64_SFLOAT;   case 2: return VK_FORMAT_R64G64_SFLOAT;  case 3: return VK_FORMAT_R64G64B64_SFLOAT;  case 4: return VK_FORMAT_R64G64B64A64_SFLOAT; }
            break;
        case GL_UNSIGNED_BYTE:
            // FIX (根因 V): UChar3 normalized/unnormalized 均不存在 → 用 UChar4
            if (normalized) switch (size) { case 1: return VK_FORMAT_R8_UNORM;  case 2: return VK_FORMAT_R8G8_UNORM;  case 3: return VK_FORMAT_R8G8B8A8_UNORM;  case 4: return VK_FORMAT_R8G8B8A8_UNORM; }
            else            switch (size) { case 1: return VK_FORMAT_R8_UINT;   case 2: return VK_FORMAT_R8G8_UINT;   case 3: return VK_FORMAT_R8G8B8A8_UINT;   case 4: return VK_FORMAT_R8G8B8A8_UINT; }
        case GL_BYTE:
            // FIX (根因 V): Char3 normalized/unnormalized 均不存在 → 用 Char4
            if (normalized) switch (size) { case 1: return VK_FORMAT_R8_SNORM;  case 2: return VK_FORMAT_R8G8_SNORM;  case 3: return VK_FORMAT_R8G8B8A8_SNORM;  case 4: return VK_FORMAT_R8G8B8A8_SNORM; }
            else            switch (size) { case 1: return VK_FORMAT_R8_SINT;   case 2: return VK_FORMAT_R8G8_SINT;   case 3: return VK_FORMAT_R8G8B8A8_SINT;   case 4: return VK_FORMAT_R8G8B8A8_SINT; }
        case GL_UNSIGNED_SHORT:
            // FIX (根因 V): UShort3 normalized/unnormalized 均不存在 → 用 UShort4
            if (normalized) switch (size) { case 1: return VK_FORMAT_R16_UNORM; case 2: return VK_FORMAT_R16G16_UNORM; case 3: return VK_FORMAT_R16G16B16A16_UNORM; case 4: return VK_FORMAT_R16G16B16A16_UNORM; }
            else            switch (size) { case 1: return VK_FORMAT_R16_UINT;  case 2: return VK_FORMAT_R16G16_UINT;  case 3: return VK_FORMAT_R16G16B16A16_UINT;  case 4: return VK_FORMAT_R16G16B16A16_UINT; }
        case GL_SHORT:
            // FIX (根因 V): Short3 normalized/unnormalized 均不存在 → 用 Short4
            if (normalized) switch (size) { case 1: return VK_FORMAT_R16_SNORM; case 2: return VK_FORMAT_R16G16_SNORM; case 3: return VK_FORMAT_R16G16B16A16_SNORM; case 4: return VK_FORMAT_R16G16B16A16_SNORM; }
            else            switch (size) { case 1: return VK_FORMAT_R16_SINT;  case 2: return VK_FORMAT_R16G16_SINT;  case 3: return VK_FORMAT_R16G16B16A16_SINT;  case 4: return VK_FORMAT_R16G16B16A16_SINT; }
        // 打包格式 A2B10G10R10 本身是 4 分量，无需转换。
        //
        // FIX (根因 AM — 打包法线符号丢失):
        // GL_INT_2_10_10_10_REV 的三个 10-bit 分量是 *有符号补码*，
        // GL_UNSIGNED_INT_2_10_10_10_REV 才是无符号。旧代码把两者一律映射到
        // UNORM/UINT，等于把补码位模式当无符号读：
        //
        //   法线 (0,-1,0) 打包后按 SNORM 应解出 (0,-1.000,0)，
        //   按 UNORM 解出 (0,+0.501,0)  —— 符号翻转
        //   法线 (0,+1,0) 应为 +1.000，UNORM 下只有 +0.500 —— 亮度减半
        //
        // Sodium 正是用 GL_INT_2_10_10_10_REV 存方块/实体法线，5 个基准方向里
        // 4 个会翻转：方块底面被当作顶面照亮，向光面反而变黑。
        // 详见 verify/packed_norm.c 的逐位验证。
        case GL_INT_2_10_10_10_REV:
            if (normalized) return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
            return VK_FORMAT_A2B10G10R10_SINT_PACK32;
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            if (normalized) return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
            return VK_FORMAT_A2B10G10R10_UINT_PACK32;
        default: break;
    }
    return VK_FORMAT_R32G32B32A32_SFLOAT;
}

// GL compare func -> VkCompareOp
VkCompareOp gl_compare_to_vk(GLenum f) {
    switch (f) {
        case GL_NEVER:    return VK_COMPARE_OP_NEVER;
        case GL_LESS:     return VK_COMPARE_OP_LESS;
        case GL_EQUAL:    return VK_COMPARE_OP_EQUAL;
        case GL_LEQUAL:   return VK_COMPARE_OP_LESS_OR_EQUAL;
        case GL_GREATER:  return VK_COMPARE_OP_GREATER;
        case GL_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
        case GL_GEQUAL:   return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case GL_ALWAYS:   return VK_COMPARE_OP_ALWAYS;
        default:          return VK_COMPARE_OP_LESS;
    }
}

// 查询格式是否支持 VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT。
// 结果缓存到 static unordered_map，避免每次 pipeline 创建都调用
// vkGetPhysicalDeviceFormatProperties。
//
// FIX (根因 X - blend 格式校验):
// Vulkan VUID-VkGraphicsPipelineCreateInfo-blendEnable-04727 要求启用 blend
// 的颜色附件格式必须支持 COLOR_ATTACHMENT_BLEND_BIT。Mithril 原直接按 GL blend
// 状态设置 blendEnable 不校验格式，对不支持 blend 的格式（如某些 GPU 的
// RGBA16F、整数 FBO RGBA8UI）启用 blend 属非法 pipeline 状态 → MoltenVK
// 编译 Metal blend 状态失败或静默丢 draw → 红屏。
//
// 深度对照 MobileGL VulkanRenderer.cpp:4124-4157: pipeline 创建前查询格式
// 属性，不支持时强制 effectiveBlendEnabled = false。
bool format_supports_color_attachment_blend(VkFormat fmt) {
    if (fmt == VK_FORMAT_UNDEFINED) return false;
    static std::unordered_map<VkFormat, bool> cache;
    auto it = cache.find(fmt);
    if (it != cache.end()) return it->second;
    Backend* b = backend();
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(b->physicalDevice, fmt, &props);
    bool ok = (props.optimalTilingFeatures &
               VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
    cache[fmt] = ok;
    return ok;
}

/* ---- 实例化步进率（GL 4.3 两层顶点模型）----
 *
 * GL 的 divisor 挂在「顶点数据源」这一层上：glVertexAttribDivisor 和
 * glVertexBindingDivisor 写的都是 VertexBinding::divisor（见 MG_State/State.h）。
 * Vulkan 的对应物是 VkVertexInputBindingDescription::inputRate，同样是 per
 * binding —— 两者严格对得上。
 *
 * 之前这里是 `bd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;` 硬编码，divisor 被
 * 整个丢掉。后果不是画得略有偏差，而是实例化属性按顶点步进：本该每个实例读一
 * 次的数据变成每个顶点读一次，几何直接崩。Sodium 的实例化区块路径和 Iris 的粒
 * 子都踩这条。
 *
 * 为什么直接读 g_state 而不是从 MGVertexAttrib 拿：MGVertexAttrib 里没有
 * divisor 字段，而 MG_Backend/Backend.h 是跨组共享的 C ABI，本次不动它。
 * primitiveRestart（root cause AF）已经用同样的方式直读 g_state，此处沿用，
 * 并且 hash_signature 与下面建管线的代码调用同一个函数，缓存键不会漏。
 */
uint32_t attrib_divisor(int location) {
    if (!mithril::g_state) return 0;
    if (location < 0 || location >= mithril::kMaxVertexAttribs) return 0;
    mithril::VertexArray* vao = mithril::state_get_vao(mithril::g_state->currentVAO);
    if (!vao) vao = mithril::state_get_vao(0);
    if (!vao) return 0;
    const GLuint bi = vao->attribs[location].bindingIndex;
    if (bi >= (GLuint)mithril::kMaxVertexBindings) return 0;
    return (uint32_t)vao->bindings[bi].divisor;
}

// FNV-1a 64-bit hash over the pipeline signature.
uint64_t hash_signature(GLuint program, const MGVertexAttrib* attribs, int attrib_count,
                        const VkFormat* color_formats, int color_count,
                        VkFormat depth_format, int blend_enabled,
                        GLenum blend_src, GLenum blend_dst,
                        GLenum blend_src_alpha, GLenum blend_dst_alpha,
                        int color_write_mask, GLenum prim,
                        int is_default_fbo) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    mix(&program, sizeof(program));
    mix(&attrib_count, sizeof(attrib_count));
    for (int i = 0; i < attrib_count; ++i) {
        mix(&attribs[i].location, sizeof(attribs[i].location));
        mix(&attribs[i].size, sizeof(attribs[i].size));
        mix(&attribs[i].type, sizeof(attribs[i].type));
        mix(&attribs[i].normalized, sizeof(attribs[i].normalized));
        mix(&attribs[i].integer, sizeof(attribs[i].integer));
        mix(&attribs[i].stride, sizeof(attribs[i].stride));
        // FIX (Root Cause L - 缓存键遗漏 offset): offset 被烘焙进管线
        // (VkVertexInputAttributeDescription.offset, Pipeline.cpp:302 ad.offset)，
        // 必须参与缓存键哈希。否则不同 offset 的 VAO 复用同一管线 → 属性从错误
        // 字节偏移读取 → 顶点数据错位 → 红屏/花屏。
        mix(&attribs[i].offset, sizeof(attribs[i].offset));
        // divisor 决定 inputRate，同样被烘焙进管线，必须进缓存键 —— 否则
        // 同一份格式在实例化与非实例化之间切换会复用旧管线（与 root cause L
        // 的 offset 同理）。
        const uint32_t div = attrib_divisor(attribs[i].location);
        mix(&div, sizeof(div));
    }
    mix(&color_count, sizeof(color_count));
    for (int i = 0; i < color_count; ++i) mix(&color_formats[i], sizeof(color_formats[i]));
    mix(&depth_format, sizeof(depth_format));
    mix(&blend_enabled, sizeof(blend_enabled));
    mix(&blend_src, sizeof(blend_src));
    mix(&blend_dst, sizeof(blend_dst));
    mix(&blend_src_alpha, sizeof(blend_src_alpha));
    mix(&blend_dst_alpha, sizeof(blend_dst_alpha));
    mix(&color_write_mask, sizeof(color_write_mask));
    mix(&prim, sizeof(prim));
    mix(&is_default_fbo, sizeof(is_default_fbo));
    // FIX (root cause AF - Primitive Restart): 将 primitiveRestart 状态纳入
    // pipeline 缓存键。否则启用/禁用 restart 会复用同一管线 →
    // ia.primitiveRestartEnable 与 g_state 不一致 → strip 在 restart 索引处
    // 行为错误 → 几何腐败。直接读 g_state（与 get_or_create_pipeline 的
    // ia.primitiveRestartEnable 计算保持同一来源），g_state 为 null 时按
    // VK_FALSE 处理（与 get_or_create_pipeline 的 null 检查一致）。
    // 深度对照 MobileGL VulkanRenderer.cpp:3861-3877。
    bool pr = (mithril::g_state && mithril::g_state->primitiveRestart);
    bool prfi = (mithril::g_state && mithril::g_state->primitiveRestartFixedIndex);
    mix(&pr, sizeof(pr));
    mix(&prfi, sizeof(prfi));
    // Depth enable/write/func are baked into the pipeline (no dynamic depth-enable
    // in Vulkan 1.2), so they must be part of the cache key. Read g_state directly,
    // matching the pipeline-creation code below.
    bool dte = (mithril::g_state && mithril::g_state->depthTest);
    bool dwm = (mithril::g_state && mithril::g_state->depthMask);
    GLenum dfn = mithril::g_state ? mithril::g_state->depthFunc : GL_LESS;
    mix(&dte, sizeof(dte));
    mix(&dwm, sizeof(dwm));
    mix(&dfn, sizeof(dfn));
    return h;
}

VkShaderModule create_module(const uint32_t* spirv, int word_count) {
    if (!spirv || word_count <= 0) return VK_NULL_HANDLE;
    Backend* b = backend();
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = (size_t)word_count * sizeof(uint32_t);
    ci.pCode = spirv;
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(b->device, &ci, nullptr, &mod) != VK_SUCCESS) {
        // FIX (P2): 限流日志，防止显存不足时大量着色器编译失败刷屏
        static int shaderFailCount = 0;
        shaderFailCount++;
        if (shaderFailCount <= 3 || shaderFailCount % 50 == 0) {
            MITHRIL_LOG_WARN("vk", "vkCreateShaderModule failed (words=%d, fail #%d)",
                              word_count, shaderFailCount);
        }
        return VK_NULL_HANDLE;
    }
    return mod;
}

// Process-wide empty VkPipelineLayout used as a fallback for programs whose
// SPIR-V reflects no descriptor bindings (e.g. vertex-only / pass-through
// shaders). Created lazily on first use.
//
// Root cause: gl_VertexID baseVertex semantics. Even a binding-less vertex
// shader carries the injected _MithrilBaseVertex push-constant block (see
// Shader.cpp:inject_vertex_id_fixup), so this fallback layout must ALSO
// declare the VERTEX-stage push-constant range (offset 0, size 4) or MoltenVK
// rejects the pipeline at creation. Mirrors the per-program layout built in
// DescriptorSet.cpp:ensure_program_layouts.
VkPipelineLayout empty_pipeline_layout() {
    Backend* b = backend();
    static VkPipelineLayout layout = VK_NULL_HANDLE;
    if (layout == VK_NULL_HANDLE) {
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcr.offset = 0;
        pcr.size = 4;  // int _mithrilBaseVertex
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(b->device, &plci, nullptr, &layout);
    }
    return layout;
}

// Reflect all vertex-shader input locations from SPIR-V via SPIRV-Cross.
// Returns the set of locations the vertex shader declares as stage inputs
// (whether or not GL has enabled a corresponding vertex attrib). Used by
// get_or_create_pipeline to emit dummy VkVertexInputAttributeDescription
// entries for un-enabled locations so SPIRV-Cross generates [[attribute(N)]]
// for every stage_in field — Metal requires this and the MSL compiler rejects
// a [[stage_in]] struct whose fields lack [[attribute(N)]] with
// "invalid type ... of input declaration with attribute 'stage_in'".
std::vector<uint32_t> reflect_vertex_input_locations(const uint32_t* spirv, int words) {
    std::vector<uint32_t> out;
    if (!spirv || words <= 0) return out;
    try {
        spirv_cross::Compiler compiler(spirv, static_cast<size_t>(words));
        spirv_cross::ShaderResources res = compiler.get_shader_resources();
        for (auto& r : res.stage_inputs) {
            uint32_t loc = compiler.get_decoration(r.id, spv::DecorationLocation);
            out.push_back(loc);
        }
    } catch (const std::exception&) {
        // Malformed SPIR-V — return empty; pipeline creation will fail later.
    }
    return out;
}

} // namespace

// Exported accessor for the empty/fallback layout (used by
// CommandStream.cpp:backend_push_constants for binding-less programs).
//
// MUST live OUTSIDE the anonymous namespace above (which closes at the
// `} // namespace` line): functions inside an anonymous namespace have
// internal linkage, so a definition placed there would never be linkable
// from another translation unit (CommandStream.cpp). It may still call
// empty_pipeline_layout() (declared earlier in this TU, internal linkage is
// fine for a call site).
VkPipelineLayout backend_default_pipeline_layout() {
    return empty_pipeline_layout();
}

VkPipeline get_or_create_pipeline(GLuint program,
                                  const uint32_t* vertex_spirv, int vertex_word_count,
                                  const uint32_t* fragment_spirv, int fragment_word_count,
                                  const MGVertexAttrib* attribs, int attrib_count,
                                  const VkFormat* color_formats, int color_count,
                                  VkFormat depth_format,
                                  int blend_enabled, GLenum blend_src, GLenum blend_dst,
                                  GLenum blend_src_alpha, GLenum blend_dst_alpha,
                                  int color_write_mask,
                                  GLenum gl_primitive_mode,
                                  int is_default_fbo) {
    Backend* b = backend();
    if (!b->initialized || program == 0) return VK_NULL_HANDLE;

    auto& tbl = program_table();
    ProgramResources& pr = tbl[program];

    // Select the vertex shader module for this framebuffer type. The default
    // framebuffer (FBO 0) uses the Y-flipped variant so the on-screen drawable
    // matches Vulkan/Metal's Y-down coordinate system; user FBOs use the
    // non-flipped variant so their textures stay in GL Y-up orientation for
    // correct sampling. Deep reference: MobileGL GetShaderTransformFlags.
    VkShaderModule& vsModule = is_default_fbo ? pr.vertexModuleFlipped : pr.vertexModule;
    if (vsModule == VK_NULL_HANDLE && vertex_spirv && vertex_word_count > 0) {
        vsModule = create_module(vertex_spirv, vertex_word_count);
    }
    if (pr.fragmentModule == VK_NULL_HANDLE && fragment_spirv && fragment_word_count > 0) {
        pr.fragmentModule = create_module(fragment_spirv, fragment_word_count);
    }
    if (vsModule == VK_NULL_HANDLE) return VK_NULL_HANDLE;

    // Reflect SPIR-V + build VkDescriptorSetLayout / VkPipelineLayout /
    // VkDescriptorPool once per program (idempotent). The pipeline below binds
    // against pr.pipelineLayout (or the empty fallback for binding-less
    // shaders), and prepare_draw later calls backend_bind_program_descriptors
    // to populate the descriptor set from Program.uniforms + bound textures.
    // Y flip only modifies gl_Position (a builtin), so both SPIR-V variants
    // share the same descriptor layout — reflecting either is correct.
    ensure_program_layouts(program, vertex_spirv, vertex_word_count,
                           fragment_spirv, fragment_word_count,
                           nullptr, 0);   // graphics program: no compute stage

    uint64_t sig = hash_signature(program, attribs, attrib_count, color_formats,
                                  color_count, depth_format, blend_enabled,
                                  blend_src, blend_dst,
                                  blend_src_alpha, blend_dst_alpha,
                                  color_write_mask, gl_primitive_mode,
                                  is_default_fbo);
    auto it = pr.pipelines.find(sig);
    if (it != pr.pipelines.end() && it->second != VK_NULL_HANDLE) return it->second;

    // Negative cache: skip the creation attempt entirely for signatures that
    // have already failed. This avoids re-running vkCreateGraphicsPipelines
    // (and re-logging the warning) on every draw call for a persistently-
    // broken shader/format combo. The set is cleared on program relink
    // (delete_program_resources), so a fixed shader recovers on the next
    // glLinkProgram. Without this, a single broken shader producing 100
    // draws/frame would generate 100 WARN lines/frame = 6000/sec at 60 FPS.
    if (pr.failedSignatures.count(sig)) {
        return VK_NULL_HANDLE;
    }

    // ---- Vertex input state ----
    std::vector<VkVertexInputBindingDescription> bindDescs;
    std::vector<VkVertexInputAttributeDescription> attrDescs;
    /* 一个属性一个 binding（binding 号 == location）。
     *
     * 这是 GL 两层模型的一种退化投影，不是 bug：GL 允许多个属性共用一个
     * binding，投影到这里就是多个 Vulkan binding 绑同一个 VkBuffer、各自用
     * VkVertexInputAttributeDescription::offset 定位成员。kMaxVertexAttribs 是
     * 16，永远不会超过 maxVertexInputBindings，所以功能上是等价的。
     *
     * 之所以没改成「按 binding 分组」的真两层：分组的收益要靠 binding offset
     * 才能兑现（把 glBindVertexBuffer 的 offset 交给 pOffsets，而不是加进属性
     * offset），而 Drawing.cpp 目前恒传 offset 0（root cause H 的修法），
     * MGVertexAttrib 也没有字段把「binding 基址」和「成员内偏移」这两个数分开
     * 送过来。拆开需要动 MG_Backend/Backend.h 的 C ABI，那是跨组共享的。
     * 详见交付说明里的遗留项。
     */
    for (int i = 0; i < attrib_count; ++i) {
        const MGVertexAttrib& a = attribs[i];
        if (!a.enabled) continue;
        VkVertexInputBindingDescription bd{};
        bd.binding = (uint32_t)a.location;
        bd.stride = a.stride > 0 ? (uint32_t)a.stride : 0;

        /* divisor > 0 = 按实例步进。divisor > 1 需要
         * VK_EXT_vertex_attribute_divisor，Device.cpp 目前没启用，只能按 1 处
         * 理并报警 —— 静默画错比慢一点糟糕得多。实测 Sodium / Iris 用的都是
         * divisor == 1，走的是下面这条无需扩展的路径。 */
        const uint32_t divisor = attrib_divisor(a.location);
        bd.inputRate = divisor ? VK_VERTEX_INPUT_RATE_INSTANCE
                               : VK_VERTEX_INPUT_RATE_VERTEX;
        if (divisor > 1) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                MITHRIL_LOG_WARN("vk",
                    "vertex attrib %d has divisor %u; VK_EXT_vertex_attribute_divisor "
                    "is not enabled, treating it as 1 (instance data will repeat "
                    "every instance instead of every %u)",
                    a.location, divisor, divisor);
            }
        }
        bindDescs.push_back(bd);

        VkVertexInputAttributeDescription ad{};
        ad.location = (uint32_t)a.location;
        ad.binding = (uint32_t)a.location;
        ad.format = attrib_type_to_vk_format(a.type, a.size, a.normalized != 0, a.integer != 0);
        ad.offset = (uint32_t)a.offset;

        // FIX (P1): 校验该 VkFormat 真的能当顶点属性用。
        //
        // Vulkan 要求顶点属性格式的 bufferFeatures 含
        // VERTEX_BUFFER_BIT。MoltenVK 对 MTLVertexFormat 里不存在的格式
        // （R64 全家、部分 3 分量组合）不报告这个位。上面的映射表已经手工
        // 规避了已知的坑，但设备之间差异很大（A11 之前 / Apple Silicon /
        // Intel Mac 各不相同），漏一个就是整条管线创建失败、该 program 的
        // 所有 draw 静默消失 —— 这种故障极难从现象反推原因。
        //
        // 这里做一次运行时兜底：真遇到不支持的格式就打日志点名，让问题在
        // 日志里可见，而不是变成一块莫名其妙的黑屏。结果按格式缓存，
        // 不影响热路径（管线创建本来就不在每帧路径上）。
        {
            static std::unordered_map<uint32_t, bool> vtxFmtOk;
            auto vit = vtxFmtOk.find((uint32_t)ad.format);
            bool ok;
            if (vit != vtxFmtOk.end()) {
                ok = vit->second;
            } else {
                VkFormatProperties fp{};
                vkGetPhysicalDeviceFormatProperties(b->physicalDevice, ad.format, &fp);
                ok = (fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) != 0;
                vtxFmtOk[(uint32_t)ad.format] = ok;
                if (!ok) {
                    MITHRIL_LOG_WARN("vk",
                        "顶点属性 location=%d 的 VkFormat %d（GL type=0x%x size=%d）"
                        "不被本设备接受为顶点格式（缺 VERTEX_BUFFER_BIT）；"
                        "管线创建很可能失败，该 program 的 draw 会全部丢失。",
                        a.location, (int)ad.format, a.type, a.size);
                }
            }
        }

        attrDescs.push_back(ad);
    }

    // Reflect all vertex-shader input locations. SPIRV-Cross (via MoltenVK)
    // generates [[attribute(N)]] for every stage_in field based on the
    // VkVertexInputAttributeDescription array. Metal requires every field of
    // a [[stage_in]] struct to carry [[attribute(N)]] — if a shader-declared
    // location has no matching attribute description, Metal compilation fails.
    // For each shader location GL has NOT enabled, append a dummy binding +
    // attribute (stride 0, format R32G32B32A32_SFLOAT, offset 0) backed by
    // b->dummyVertexBuffer at draw time.
    std::vector<uint32_t> shaderLocations =
        reflect_vertex_input_locations(vertex_spirv, vertex_word_count);
    // FIX (MSL `main0_in stage_in` compile failure - reflection fallback):
    // 若 SPIRV-Cross 反射失败（畸形 SPIR-V / 异常被吞，返回空），上面一个 dummy
    // 属性都不会生成。此时任何「GL 未启用的 shader location」字段都会缺
    // [[attribute(N)]] → Metal 报 `invalid type 'main0_in' ... stage_in` →
    // vkCreateGraphicsPipelines 失败 → 该 program 所有 draw 静默消失 → 红屏。
    // 兜底：反射为空时覆盖 0..15 全部可能 location。多余的空 binding（stride 0）
    // 由 dummyVertexBuffer 兜底、无副作用；Metal 只要求每个 stage_in 字段都有
    // attribute，多给无害。这从根上消除整类 MSL stage_in 编译失败。
    if (shaderLocations.empty()) {
        for (uint32_t loc = 0; loc < 16; ++loc) shaderLocations.push_back(loc);
    }
    for (uint32_t loc : shaderLocations) {
        bool alreadyEnabled = false;
        for (int i = 0; i < attrib_count; ++i) {
            if (attribs[i].enabled && (uint32_t)attribs[i].location == loc) {
                alreadyEnabled = true;
                break;
            }
        }
        if (alreadyEnabled) continue;

        VkVertexInputBindingDescription bd{};
        bd.binding = loc;
        bd.stride = 0;
        bd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindDescs.push_back(bd);

        VkVertexInputAttributeDescription ad{};
        ad.location = loc;
        ad.binding = loc;
        ad.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ad.offset = 0;
        attrDescs.push_back(ad);
    }

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = (uint32_t)bindDescs.size();
    vertexInput.pVertexBindingDescriptions = bindDescs.data();
    vertexInput.vertexAttributeDescriptionCount = (uint32_t)attrDescs.size();
    vertexInput.pVertexAttributeDescriptions = attrDescs.data();

    // ---- Input assembly ----
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = gl_prim_to_vk(gl_primitive_mode);
    // FIX (root cause AF - Primitive Restart):
    // 原 `ia.primitiveRestartEnable = VK_FALSE` 硬编码，忽略 GL 的
    // GL_PRIMITIVE_RESTART / GL_PRIMITIVE_RESTART_FIXED_INDEX 状态。strip/fan
    // 几何在 restart 索引处不断开 → 连接到无效顶点 → 几何腐败 → 红屏。
    // 改为从 g_state 读取：两者任一启用时为 VK_TRUE。需 Device.cpp 启用
    // VK_EXT_primitive_topology_list_restart 扩展以支持 list topology 上的 restart。
    // 深度对照 MobileGL VulkanRenderer.cpp:3861-3877。
    ia.primitiveRestartEnable =
        (mithril::g_state && (mithril::g_state->primitiveRestart ||
                              mithril::g_state->primitiveRestartFixedIndex))
            ? VK_TRUE : VK_FALSE;

    /* FIX (root cause AS - list topology restart is conditional):
     *
     * Enabling the extension above is only half of it. When the device does
     * NOT have VK_EXT_primitive_topology_list_restart — which is the case on
     * MoltenVK today — combining primitiveRestartEnable with a LIST topology
     * violates VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252.
     * vkCreateGraphicsPipelines then fails, the signature lands in
     * failedSignatures, and every draw with that combination is dropped for
     * the rest of the session. An app that enables GL_PRIMITIVE_RESTART once
     * and later draws GL_TRIANGLES loses those draws entirely.
     *
     * Restart is meaningful on strips and fans; on a list each primitive is
     * already self-delimiting, so suppressing the bit there costs nothing
     * real. MobileGL throws instead — appropriate for a debug build, but here
     * dropping a corrupt-geometry edge case beats losing the draw outright.
     */
    if (ia.primitiveRestartEnable == VK_TRUE) {
        const bool isList =
            ia.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST ||
            ia.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
            ia.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ||
            ia.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY ||
            ia.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY ||
            ia.topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        if (isList && !b->listRestartSupported) {
            ia.primitiveRestartEnable = VK_FALSE;
            static bool warned = false;
            if (!warned) {
                warned = true;
                MITHRIL_LOG_WARN("vk",
                                 "primitive restart requested on a list topology but "
                                 "VK_EXT_primitive_topology_list_restart is unavailable; "
                                 "suppressing the restart bit so the pipeline can be created");
            }
        }
    }

    // ---- Viewport / scissor (dynamic) ----
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.pViewports = nullptr;   // dynamic
    vp.scissorCount = 1;
    vp.pScissors = nullptr;    // dynamic

    // ---- Rasterizer ----
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.depthClampEnable = VK_FALSE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    // These are dynamic when VK_EXT_extended_dynamic_state is present and the
    // values here are then ignored. Otherwise they are the real static state,
    // so they are read from the GL state rather than left at a placeholder -
    // cull mode NONE would silently disable back-face culling on any device
    // without the extension.
    if (!b->extendedDynamicStateSupported) {
        rs.cullMode = VK_CULL_MODE_NONE;
        if (mithril::g_state) {
            if (mithril::g_state->cullFace) {
                switch (mithril::g_state->cullMode) {
                    case GL_FRONT: rs.cullMode = VK_CULL_MODE_FRONT_BIT; break;
                    case GL_BACK:  rs.cullMode = VK_CULL_MODE_BACK_BIT; break;
                    case GL_FRONT_AND_BACK:
                        rs.cullMode = VK_CULL_MODE_FRONT_AND_BACK; break;
                    default: break;
                }
            }
            rs.frontFace = (mithril::g_state->frontFace == GL_CW)
                ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
        }
    } else {
        rs.cullMode = VK_CULL_MODE_NONE;             // replaced by vkCmdSetCullMode
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;  // replaced by vkCmdSetFrontFace
    }
    rs.depthBiasEnable = VK_FALSE;
    rs.lineWidth = 1.0f;

    // ---- Multisample ----
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ms.minSampleShading = 1.0f;
    /* GL 4.0 ARB_sample_shading. Only meaningful once the attachment is
     * actually multisampled — requesting sampleShadingEnable at
     * VK_SAMPLE_COUNT_1_BIT is legal but pointless, and needs the
     * sampleRateShading device feature to be enabled. */
    if (mithril::g_state && mithril::g_state->sampleShadingEnabled &&
        b->sampleRateShadingSupported &&
        ms.rasterizationSamples != VK_SAMPLE_COUNT_1_BIT) {
        ms.sampleShadingEnable = VK_TRUE;
        ms.minSampleShading = mithril::g_state->minSampleShading;
    }
    /* GL 4.0 glSampleMaski / GL_SAMPLE_MASK. VkPipelineMultisampleStateCreateInfo
     * takes the mask by pointer, so it must outlive this call — the state
     * lives in GLState, which does. */
    if (mithril::g_state && mithril::g_state->sampleMask) {
        ms.pSampleMask = (const VkSampleMask*)mithril::g_state->sampleMaskValue;
    }

    // ---- Depth / stencil ----
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    // Reflect GL depth state. Previously these were hardcoded VK_TRUE, which
    // depth-rejected GUI/HUD geometry (drawn after RenderSystem.disableDepthTest)
    // against the terrain depth buffer, so the HUD never appeared.
    ds.depthTestEnable = (mithril::g_state && mithril::g_state->depthTest) ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = (mithril::g_state && mithril::g_state->depthMask) ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = mithril::g_state ? gl_compare_to_vk(mithril::g_state->depthFunc)
                                         : VK_COMPARE_OP_LESS;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.stencilTestEnable = VK_FALSE;

    // ---- Color blend ----
    // FIX (root cause I+J): use independent alpha blend factors (blend_src_alpha/
    // blend_dst_alpha) and apply the GL colorWriteMask from g_state instead of
    // hardcoding RGBA all-on. glColorMask now takes effect via the pipeline
    // signature (color_write_mask is hashed above), so depth-only pre-passes
    // that disable color writes get a distinct pipeline that does not corrupt
    // the color buffer.
    //
    // FIX (根因 X - blend 格式校验):
    // 启用 blend 前校验颜色附件格式是否支持 COLOR_ATTACHMENT_BLEND_BIT。
    // 不支持时强制 blendEnable=VK_FALSE，避免非法 pipeline 状态导致
    // MoltenVK 编译失败或静默丢 draw → 红屏。深度对照 MobileGL
    // VulkanRenderer.cpp:4124-4157 effectiveBlendEnabled 逻辑。
    // VUID-VkGraphicsPipelineCreateInfo-blendEnable-04727。
    //
    // 注意：hash_signature 含 blend_enabled（原始值）而非 effective_blend_enabled，
    // 但 color_formats 也参与哈希，故两个不同格式的 FBO 会得到不同 sig，pipeline
    // 缓存不会冲突，无需修改 hash_signature。
    bool effective_blend_enabled = blend_enabled != 0;
    if (effective_blend_enabled && color_count > 0) {
        // 校验所有颜色附件格式（多附件时任一不支持则禁用 blend）
        for (int i = 0; i < color_count; ++i) {
            if (!format_supports_color_attachment_blend(color_formats[i])) {
                effective_blend_enabled = false;
                // 限流日志：每格式仅警告一次（cache 已保证查询一次，
                // 但 pipeline 创建可能多次走此路径）
                static std::unordered_map<VkFormat, bool> warned;
                if (!warned[color_formats[i]]) {
                    warned[color_formats[i]] = true;
                    MITHRIL_LOG_WARN("vk", "Root cause X: color attachment format "
                                      "(VkFormat=%d) does not support "
                                      "COLOR_ATTACHMENT_BLEND_BIT, forcing "
                                      "blendEnable=VK_FALSE to avoid illegal "
                                      "pipeline state (VUID-04727)",
                                      (int)color_formats[i]);
                }
                break;
            }
        }
    }

    VkPipelineColorBlendAttachmentState cbAttach{};
    cbAttach.blendEnable = effective_blend_enabled ? VK_TRUE : VK_FALSE;
    if (effective_blend_enabled) {
        cbAttach.srcColorBlendFactor = gl_blend_to_vk(blend_src);
        cbAttach.dstColorBlendFactor = gl_blend_to_vk(blend_dst);
        cbAttach.colorBlendOp = VK_BLEND_OP_ADD;
        cbAttach.srcAlphaBlendFactor = gl_blend_to_vk(blend_src_alpha);
        cbAttach.dstAlphaBlendFactor = gl_blend_to_vk(blend_dst_alpha);
        cbAttach.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    // colorWriteMask 部分保持不变
    VkColorComponentFlags cwm = 0;
    if (color_write_mask & 1) cwm |= VK_COLOR_COMPONENT_R_BIT;
    if (color_write_mask & 2) cwm |= VK_COLOR_COMPONENT_G_BIT;
    if (color_write_mask & 4) cwm |= VK_COLOR_COMPONENT_B_BIT;
    if (color_write_mask & 8) cwm |= VK_COLOR_COMPONENT_A_BIT;
    cbAttach.colorWriteMask = cwm;
    if (std::getenv("MITHRIL_BLEND_PROBE")) {
        // Diagnostic: force output = fragment RGB regardless of alpha.
        cbAttach.blendEnable = VK_TRUE;
        cbAttach.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;   // 1
        cbAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;  // 0
        cbAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cbAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cbAttach.colorBlendOp = VK_BLEND_OP_ADD;
        cbAttach.alphaBlendOp = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.logicOpEnable = VK_FALSE;
    // One blend record per color attachment. A depth-only pipeline has none.
    std::vector<VkPipelineColorBlendAttachmentState> cbAttachments;
    if (color_count > 0) cbAttachments.assign((size_t)color_count, cbAttach);
    cb.attachmentCount = (uint32_t)cbAttachments.size();
    cb.pAttachments = cbAttachments.empty() ? nullptr : cbAttachments.data();

    // ---- Dynamic state ----
    // VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT requires the
    // VK_EXT_color_write_enable extension which we do not enable; colour
    // write is therefore part of the pipeline's blend attachment state
    // (cbAttach.colorWriteMask above). The extended-dynamic-state extension
    // we DO enable covers cull/front-face/depth-test/depth-write/depth-compare.
    // VK_DYNAMIC_STATE_CULL_MODE and friends are only legal when
    // VK_EXT_extended_dynamic_state is enabled (or on Vulkan 1.3, where they
    // are core). Declaring them unconditionally made the pipeline depend on an
    // extension that Device.cpp treats as optional: with the extension absent
    // the pipeline still listed them, nothing ever set them, and the pipeline
    // itself was invalid.
    //
    // So the list is conditional. When the extension is missing the states stay
    // static and are taken from the rasterizer/depth-stencil structs above,
    // which is a valid pipeline - it just cannot track per-draw changes to
    // cull or depth compare. backend_set_* in CommandStream.cpp already bails
    // out when their entry points are unresolved, so the two agree.
    // Ordered core-first so the extended-dynamic-state entries can be dropped
    // by simply shortening the count: they are all at the tail.
    VkDynamicState dynStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        // --- tail: requires VK_EXT_extended_dynamic_state or Vulkan 1.3 ---
        VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    };
    constexpr uint32_t kCoreDynamicCount = 3;  // viewport, scissor, blend constants
    const bool ext_dyn = b->extendedDynamicStateSupported;
    const uint32_t dynCount = ext_dyn
        ? (uint32_t)(sizeof(dynStates) / sizeof(dynStates[0]))
        : kCoreDynamicCount;
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = dynCount;
    dyn.pDynamicStates = dynStates;

    // ---- Shader stages ----
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    VkPipelineShaderStageCreateInfo vsStage{};
    vsStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vsStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vsStage.module = vsModule;
    vsStage.pName = "main";
    stages.push_back(vsStage);
    if (pr.fragmentModule != VK_NULL_HANDLE) {
        VkPipelineShaderStageCreateInfo fsStage{};
        fsStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fsStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fsStage.module = pr.fragmentModule;
        fsStage.pName = "main";
        if (std::getenv("MITHRIL_FS_PROBE")) {
            static VkShaderModule probeMod = VK_NULL_HANDLE;
            if (probeMod == VK_NULL_HANDLE) probeMod = create_probe_fs_module();
            if (probeMod != VK_NULL_HANDLE) fsStage.module = probeMod;
        }
        stages.push_back(fsStage);
    }

    // ---- Dynamic rendering attachment info (Vulkan 1.2 + VK_KHR_dynamic_rendering) ----
    VkFormat colorFmts[8] = {};
    for (int i = 0; i < color_count && i < 8; ++i) colorFmts[i] = color_formats[i];
    VkPipelineRenderingCreateInfo renderingCI{};
    renderingCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingCI.colorAttachmentCount = color_count > 0 ? (uint32_t)color_count : 0;
    renderingCI.pColorAttachmentFormats = color_count > 0 ? colorFmts : nullptr;
    renderingCI.depthAttachmentFormat = depth_format;
    // FIX (root cause O): For packed depth-stencil formats (D32_SFLOAT_S8_UINT,
    // D24_UNORM_S8_UINT), the stencil attachment format MUST match the depth
    // format. begin_render_pass() binds the SAME VkImageView (depthView, which
    // has aspect DEPTH|STENCIL) as both pDepthAttachment and pStencilAttachment.
    // If the pipeline declares stencilAttachmentFormat = VK_FORMAT_UNDEFINED
    // while the render pass provides a stencil attachment, this is a
    // pipeline/render-pass incompatibility — MoltenVK may silently drop the
    // entire draw or fail to compile the Metal stencil state, producing a
    // black screen. MobileGL sets stencilAttachmentFormat = depth_format for
    // packed D32S8/D24S8 formats. For depth-only formats (D32_SFLOAT,
    // D16_UNORM) there is no stencil aspect, so UNDEFINED is correct.
    if (depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
        depth_format == VK_FORMAT_D24_UNORM_S8_UINT ||
        depth_format == VK_FORMAT_D16_UNORM_S8_UINT ||
        depth_format == VK_FORMAT_S8_UINT) {
        renderingCI.stencilAttachmentFormat = depth_format;
    } else {
        renderingCI.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    }

    // ---- Graphics pipeline ----
    VkGraphicsPipelineCreateInfo gi{};
    gi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gi.pNext = &renderingCI;
    gi.stageCount = (uint32_t)stages.size();
    gi.pStages = stages.data();
    gi.pVertexInputState = &vertexInput;
    gi.pInputAssemblyState = &ia;
    gi.pViewportState = &vp;
    gi.pRasterizationState = &rs;
    gi.pMultisampleState = &ms;
    gi.pDepthStencilState = &ds;
    gi.pColorBlendState = &cb;
    gi.pDynamicState = &dyn;
    gi.renderPass = VK_NULL_HANDLE;
    gi.subpass = 0;

    // Pipeline layout: use the program's reflected layout (built by
    // ensure_program_layouts above) so UBO / sampled-image bindings declared in
    // the SPIR-V are visible to the shader. Programs with no descriptor
    // bindings (reflection produced an empty set) fall back to the
    // process-wide empty layout so vkCreateGraphicsPipelines still succeeds.
    gi.layout = (pr.pipelineLayout != VK_NULL_HANDLE) ? pr.pipelineLayout
                                                      : empty_pipeline_layout();

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(b->device, b->pipelineCache, 1, &gi,
                                           nullptr, &pipeline);
    if (r != VK_SUCCESS) {
        {
            static int s_dbg=0;
            if (s_dbg < 4) {
                s_dbg++;
                
                for (auto& d : attrDescs)
                    std::fprintf(stderr,"   AD loc=%u binding=%u format=%d off=%u\n",
                        d.location,d.binding,(int)d.format,d.offset);
                std::fflush(stderr);
            }
        }
        // FIX (红屏根因 - 瞬态失败不可永久缓存):
        // vkCreateGraphicsPipelines 可能在设备处于异常状态时因瞬态原因失败：
        //   VK_ERROR_OUT_OF_DEVICE_MEMORY     (-2) 显存不足，设备恢复后可成功
        //   VK_ERROR_OUT_OF_HOST_MEMORY       (-1) 主机内存不足
        //   VK_ERROR_DEVICE_LOST              (-4) 设备丢失，恢复后可成功
        //   VK_ERROR_INITIALIZATION_FAILED    (-3) MoltenVK 着色器库编译失败
        //         （deviceLost 后 MoltenVK 内部 MSL 编译器状态异常）
        //
        // 这些失败不是着色器本身的永久性缺陷。如果将其加入 failedSignatures
        // 负缓存，即使设备恢复后着色器可以正常编译，draw 也会被永久跳过，
        // 导致物体消失/红屏（只剩 clear color）。
        //
        // 仅当失败是永久性着色器/格式问题时才加入负缓存：
        //   VK_ERROR_INVALID_SHADER_NV        (-1000072000) 着色器本身有缺陷
        //
        // MobileGL 不做负缓存（VulkanRenderer.cpp:4183 直接返回 null），
        // 但那样会导致每帧重试。我们保留负缓存但仅用于永久性失败。
        //
        // FIX (红屏根因 - VK_ERROR_DEVICE_LOST 时序窗口):
        // 原代码只检查 b->deviceLost 标志，但 vkCreateGraphicsPipelines 可能在
        // b->deviceLost 被 vkQueueSubmit/vkDeviceWaitIdle 设置之前就返回
        // VK_ERROR_DEVICE_LOST。在这个时序窗口中 isTransient=false，签名被永久
        // 缓存到 failedSignatures，后续所有 draw 被跳过 → 红屏。
        // 修复：显式检查 r == VK_ERROR_DEVICE_LOST，并在此时主动设置
        // b->deviceLost = true 触发 EGL 恢复路径（clear_all_pipeline_caches）。
        bool isTransient = (r == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                           r == VK_ERROR_OUT_OF_HOST_MEMORY ||
                           r == VK_ERROR_INITIALIZATION_FAILED ||
                           r == VK_ERROR_DEVICE_LOST ||
                           b->deviceLost);
        // 当 pipeline 创建返回 VK_ERROR_DEVICE_LOST 时，主动设置 deviceLost 标志，
        // 让 EGL 恢复路径（backend_reset_device_lost → clear_all_pipeline_caches）
        // 清除负缓存并重建 pipeline。否则设备已死但标志未设置，恢复永远不会触发。
        if (r == VK_ERROR_DEVICE_LOST && !b->deviceLost) {
            b->deviceLost = true;
        }
        if (!isTransient) {
            // 永久性失败：加入负缓存避免每帧重试刷屏
            pr.failedSignatures.insert(sig);
        }
        // 限流日志：瞬态失败首次 + 每 50 次打印一条；永久性失败首次 + 每 100 次
        static int transientFailCount = 0;
        static int permanentFailCount = 0;
        if (isTransient) {
            transientFailCount++;
            if (transientFailCount <= 3 || transientFailCount % 50 == 0) {
                MITHRIL_LOG_WARN("vk", "vkCreateGraphicsPipelines transient "
                                  "failure (rc=%d, program=%u, deviceLost=%d, "
                                  "fail #%d) — NOT cached, will retry next draw",
                                  (int)r, program, (int)b->deviceLost,
                                  transientFailCount);
            }
        } else {
            permanentFailCount++;
            if (permanentFailCount <= 3 || permanentFailCount % 100 == 0) {
                MITHRIL_LOG_WARN("vk", "vkCreateGraphicsPipelines failed (rc=%d, "
                                  "program=%u, colorCount=%d, colorFmt0=%d, "
                                  "depthFmt=%d, attribs=%d, blend=%d) — draws "
                                  "with this signature will be skipped until "
                                  "program relink (fail #%d)",
                                  (int)r, program, color_count,
                                  color_count > 0 ? (int)color_formats[0] : 0,
                                  (int)depth_format, attrib_count, blend_enabled,
                                  permanentFailCount);
            }
        }
        return VK_NULL_HANDLE;
    }
    pr.pipelines[sig] = pipeline;
    return pipeline;
}

VkPipeline get_or_create_compute_pipeline(GLuint program,
                                          const uint32_t* compute_spirv,
                                          int compute_word_count) {
    Backend* b = backend();
    if (!b->initialized || program == 0) return VK_NULL_HANDLE;
    if (!compute_spirv || compute_word_count <= 0) return VK_NULL_HANDLE;

    auto& tbl = program_table();
    ProgramResources& pr = tbl[program];

    // One program == one compute pipeline. Unlike the graphics path there is
    // no signature to key on (no vertex format, no attachments, no blend), so
    // the cached handle is returned directly. delete_program_resources()
    // clears it on relink.
    if (pr.computePipeline != VK_NULL_HANDLE) return pr.computePipeline;

    if (pr.computeModule == VK_NULL_HANDLE) {
        pr.computeModule = create_module(compute_spirv, compute_word_count);
    }
    if (pr.computeModule == VK_NULL_HANDLE) return VK_NULL_HANDLE;

    // Reflect the compute SPIR-V into the same per-program descriptor layout
    // machinery the graphics path uses, so bind_program_descriptors() can
    // write UBOs / samplers / SSBOs / storage images for this program.
    ensure_program_layouts(program, nullptr, 0, nullptr, 0,
                           compute_spirv, compute_word_count);

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = pr.computeModule;
    stage.pName  = "main";

    VkComputePipelineCreateInfo ci{};
    ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage  = stage;
    // A shader with no reflected bindings gets the process-wide empty layout,
    // exactly as get_or_create_pipeline does.
    ci.layout = pr.pipelineLayout ? pr.pipelineLayout : empty_pipeline_layout();

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult r = vkCreateComputePipelines(b->device, VK_NULL_HANDLE, 1, &ci,
                                          nullptr, &pipeline);
    if (r != VK_SUCCESS || pipeline == VK_NULL_HANDLE) {
        // FIX (VK_ERROR_DEVICE_LOST from compute pipeline creation):
        // Mirror the graphics path — set deviceLost so EGL recovery triggers
        // clear_all_pipeline_caches and rebuilds from a clean state.
        if (r == VK_ERROR_DEVICE_LOST && !b->deviceLost) {
            b->deviceLost = true;
        }
        // Rate-limited, same rationale as the graphics failure path: a broken
        // compute shader dispatched every frame must not flood the log.
        static int computeFailCount = 0;
        computeFailCount++;
        if (computeFailCount <= 3 || computeFailCount % 100 == 0) {
            MITHRIL_LOG_WARN("vk", "vkCreateComputePipelines failed (rc=%d, "
                              "program=%u, words=%d, fail #%d)",
                              (int)r, program, compute_word_count, computeFailCount);
        }
        return VK_NULL_HANDLE;
    }
    pr.computePipeline = pipeline;
    return pipeline;
}

// FIX (红屏根因 - deviceLost 恢复后清除负缓存):
// deviceLost 期间 vkCreateGraphicsPipelines 可能因设备状态异常而失败。
// 如果这些失败被加入 failedSignatures 负缓存，即使设备恢复后着色器
// 可以正常编译，draw 也会被永久跳过，导致物体消失/红屏。
//
// 此函数在 backend_reset_device_lost() 时调用，清除所有 program 的
// failedSignatures，让着色器在设备恢复后有机会重新编译。
// 同时也清除已创建的 pipeline 缓存（pipeline 可能引用了损坏的着色器），
// 让 get_or_create_pipeline 在下次 draw 时重新创建。
//
// 参考 MobileGL RecreateSwapchain（VulkanRenderer.cpp:8579）：
// swapchain 重建时 pipelineFactory->DestroyAll() 销毁全部 pipeline，
// 确保设备恢复后着色器从干净状态重新编译。
void clear_all_pipeline_caches() {
    Backend* b = backend();
    if (!b->device) return;
    auto& tbl = program_table();
    for (auto& kv : tbl) {
        ProgramResources& pr = kv.second;
        // 清除负缓存：让之前因瞬态失败被跳过的签名有机会重试
        pr.failedSignatures.clear();
        // 销毁已创建的 pipeline：deviceLost 后 MoltenVK 的着色器缓存
        // 可能已损坏，重建 pipeline 确保从干净状态编译
        for (auto& pkv : pr.pipelines) {
            if (pkv.second) {
                vkDestroyPipeline(b->device, pkv.second, nullptr);
            }
        }
        pr.pipelines.clear();
        // 计算管线同理：deviceLost 后必须重建，否则 dispatch 会引用损坏的
        // 着色器缓存。置空后 get_or_create_compute_pipeline 会重新创建。
        if (pr.computePipeline) {
            vkDestroyPipeline(b->device, pr.computePipeline, nullptr);
            pr.computePipeline = VK_NULL_HANDLE;
        }
    }
    MITHRIL_LOG_INFO("vk", "clear_all_pipeline_caches: cleared all "
                      "failedSignatures + destroyed all cached pipelines "
                      "(device recovery)");
}

void delete_program_resources(GLuint program) {
    Backend* b = backend();
    auto& tbl = program_table();
    auto it = tbl.find(program);
    if (it == tbl.end()) return;
    ProgramResources& pr = it->second;
    // FIX (SIGBUS): glDeleteProgram 可能在帧中间调用（command buffer 正在录制）。
    // 使用 safe_device_wait_idle 安全结束+提交当前 command buffer 后再 wait，
    // 避免 MoltenVK deferred encoding 触发 SIGBUS。
    safe_device_wait_idle();
    // safe_device_wait_idle 保证所有 GPU 工作已完成 — drain 任何之前帧延迟的
    // buffer/texture/sampler 销毁，确保没有过期的 Vulkan handle 比 program 的
    // shader module 存活更久。
    //
    // FIX (GPU page fault UAF): safe_device_wait_idle() submits the CURRENT slot's
    // buffer WITHOUT advancing currentFrame and re-begins the SAME slot, so OTHER
    // programs' descriptor sets (memo'd this frame) still reference the current
    // slot's deferred-destroyed views. drain_all_disposal_queues() here frees
    // those views -> UAF when a stale memo set is re-bound on the next submit
    // (kIOGPUCommandBufferCallbackErrorPageFault). glDeleteProgram fires during
    // world-render when Sodium/Iris swap shader variants. Drain all slots EXCEPT
    // the current one; the current slot's queue is left for the normal fence-wait
    // drain after the frame commits and recycles.
    drain_disposal_queues_except(b->currentFrame);
    for (auto& kv : pr.pipelines) {
        if (kv.second) vkDestroyPipeline(b->device, kv.second, nullptr);
    }
    pr.pipelines.clear();
    if (pr.computePipeline)     { vkDestroyPipeline(b->device, pr.computePipeline, nullptr);         pr.computePipeline = VK_NULL_HANDLE; }
    if (pr.vertexModule)        { vkDestroyShaderModule(b->device, pr.vertexModule, nullptr);        pr.vertexModule = VK_NULL_HANDLE; }
    if (pr.vertexModuleFlipped) { vkDestroyShaderModule(b->device, pr.vertexModuleFlipped, nullptr); pr.vertexModuleFlipped = VK_NULL_HANDLE; }
    if (pr.fragmentModule)      { vkDestroyShaderModule(b->device, pr.fragmentModule, nullptr);      pr.fragmentModule = VK_NULL_HANDLE; }
    if (pr.computeModule)       { vkDestroyShaderModule(b->device, pr.computeModule, nullptr);       pr.computeModule = VK_NULL_HANDLE; }
    // Descriptor resources built by ensure_program_layouts. Pools must be
    // destroyed before the set layout they were created from (Vulkan ordering);
    // destroying a pool implicitly frees all sets allocated from it, so the
    // per-slot allocatedSets caches just need to be cleared (no per-set
    // vkFreeDescriptorSets — the pool destruction reclaims them wholesale).
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        pr.allocatedSets[i].clear();
        pr.setCursor[i] = 0;
        pr.lastFrameGen[i] = 0;
        if (pr.descriptorPools[i]) {
            vkDestroyDescriptorPool(b->device, pr.descriptorPools[i], nullptr);
            pr.descriptorPools[i] = VK_NULL_HANDLE;
        }
    }
    // The descriptor bind shadow in DescriptorSet.cpp may still name a set (and
    // the pipeline layout) we just destroyed. Comparing a recycled handle
    // against it could make a later draw skip a bind it needs, so drop it.
    on_command_buffer_boundary();
    if (pr.pipelineLayout)      { vkDestroyPipelineLayout(b->device, pr.pipelineLayout, nullptr);      pr.pipelineLayout = VK_NULL_HANDLE; }
    if (pr.descriptorSetLayout) { vkDestroyDescriptorSetLayout(b->device, pr.descriptorSetLayout, nullptr); pr.descriptorSetLayout = VK_NULL_HANDLE; }
    pr.bindings.clear();
    pr.layoutsBuilt = false;
    tbl.erase(it);
}

} // namespace vk
} // namespace mithril

// ===========================================================================
// Public C API wrappers (declared in MG_Backend/Backend.h)
// ===========================================================================
extern "C" {

VkPipeline backend_get_or_create_pipeline(GLuint program,
                                          const uint32_t* vertex_spirv, int vertex_word_count,
                                          const uint32_t* fragment_spirv, int fragment_word_count,
                                          const MGVertexAttrib* attribs, int attrib_count,
                                          const VkFormat* color_formats, int color_count,
                                          VkFormat depth_format,
                                          int blend_enabled, GLenum blend_src, GLenum blend_dst,
                                          GLenum blend_src_alpha, GLenum blend_dst_alpha,
                                          int color_write_mask,
                                          GLenum gl_primitive_mode,
                                          int is_default_fbo) {
    return mithril::vk::get_or_create_pipeline(program, vertex_spirv, vertex_word_count,
                                               fragment_spirv, fragment_word_count,
                                               attribs, attrib_count,
                                               color_formats, color_count, depth_format,
                                               blend_enabled, blend_src, blend_dst,
                                               blend_src_alpha, blend_dst_alpha,
                                               color_write_mask,
                                               gl_primitive_mode,
                                               is_default_fbo);
}

// Unlike the graphics wrapper, the caller does not hand over the SPIR-V: a
// dispatch only knows the currently-bound GL program, and the compute stage
// has no per-draw variants (no Y-flip, no vertex format) to choose between.
// Fetch it straight from the linked program.
VkPipeline backend_get_or_create_compute_pipeline(GLuint program) {
    mithril::Program* p = mithril::state_get_program(program);
    if (!p || !p->linked || p->computeSpirv.empty()) return VK_NULL_HANDLE;
    return mithril::vk::get_or_create_compute_pipeline(
        program, p->computeSpirv.data(), (int)p->computeSpirv.size());
}

void backend_delete_program_resources(GLuint program) {
    mithril::vk::delete_program_resources(program);
}

} // extern "C"
