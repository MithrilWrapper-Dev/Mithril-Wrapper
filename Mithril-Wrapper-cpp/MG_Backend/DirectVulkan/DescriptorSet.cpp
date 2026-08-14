// Mithril-Wrapper - MG_Backend/DirectVulkan/DescriptorSet.cpp
// SPIR-V reflection (SPIRV-Cross) + VkDescriptorSetLayout / VkPipelineLayout /
// VkDescriptorPool construction + per-frame VkDescriptorSet allocation/write/bind.
//
// Reflection strategy: glslang compiles desktop GLSL with setAutoMapBindings(true),
// so each `uniform` global / `uniform sampler*` gets an auto-assigned Vulkan
// binding. We reflect both VS and FS, merge by (set,binding) OR-ing the stage
// masks, build one VkDescriptorSetLayout, and a VkPipelineLayout referencing it.
//
// UBO data sourcing: a reflected UBO is matched against Program.uniforms. If the
// UBO name matches a uniform directly (glslang emits one UBO per loose uniform)
// we upload that uniform's value. Otherwise (glslang aggregates loose uniforms
// into a single `$Global` block) we pack the block's members by name using the
// member offsets reported by SPIRV-Cross's get_active_buffer_ranges().
//
// Texture sourcing: a reflected combined-image-sampler at binding B is fed from
// g_state->boundTextureForUnit(B) (the GL texture bound to unit B), matching how
// the GL frontend binds textures by unit index.
#include "DescriptorSet.h"
#include "Device.h"
#include "Pipeline.h"
#include "CommandStream.h"  // ensure_command_buffer_recording
#include "Reflect.h"    // reflect_stage / merge_bindings (pure-logic, unit-tested)
#include "Std140.h"     // pack_std140 — tight glUniform data -> std140 slots
#include "Resources.h"  // texture_table() — for TextureEntry::format lookup (root cause AH)
#include "FormatMap.h"  // sampled_layout_for_format (root cause AH)
#include "UniformArena.h"  // transient per-frame UBO storage (see below)
#include "../Backend.h"
#include "../../MG_State/State.h"
#include "../../MG_Impl/Log.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace mithril {
namespace vk {

/*
 * The descriptorType a binding is declared with in the set layout — and
 * therefore the ONLY type a vkUpdateDescriptorSets write targeting it may
 * use. Both the layout builder and the per-draw write path call this, so the
 * two can never disagree (a mismatch is an immediate validation error and, on
 * MoltenVK, a dropped or mis-encoded draw).
 */
static VkDescriptorType descriptor_type_for(const ProgramResources& pr,
                                            const DescriptorBinding& db) {
    if (db.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && pr.uboDynamic) {
        for (uint32_t b : pr.dynamicUboBindings) {
            if (b == db.binding) return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        }
    }
    return db.type;
}

// Same question, by binding number, for the per-draw path.
static bool binding_is_dynamic_ubo(const ProgramResources& pr, uint32_t binding) {
    if (!pr.uboDynamic) return false;
    for (uint32_t b : pr.dynamicUboBindings) {
        if (b == binding) return true;
    }
    return false;
}

// Per-pool set capacity. Minecraft 1.21.1 每帧可能绑定超过 256 个不同描述符集
// （多种着色器 + 多个纹理单元），256 不够用导致 "descriptor pool exhausted"
// 日志刷屏并跳过 draw。1024 足够覆盖最复杂的场景，内存开销可忽略。池耗尽时
// bind_program_descriptors 会用 create_program_pool 扩容次级池。
constexpr uint32_t kMaxSetsPerPool = 1024;

/*
 * Create a VkDescriptorPool sized for `pr.bindings` with room for `maxSets`
 * sets. Pool sizing is per-SET, not per-type: a pool is consumed by both
 * counts at once, so a program with two UBO bindings needs 2*maxSets
 * uniform-buffer descriptors (see the sizing comment in ensure_program_layouts).
 */
static VkDescriptorPool create_program_pool(const ProgramResources& pr, uint32_t maxSets) {
    Backend* b = backend();
    if (!b->device) return VK_NULL_HANDLE;

    uint32_t uboPerSet = 0, imgPerSet = 0, ssboPerSet = 0, stImgPerSet = 0;
    for (const auto& db : pr.bindings) {
        const uint32_t n = db.descriptorCount ? db.descriptorCount : 1u;
        if (db.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)              uboPerSet += n;
        else if (db.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) imgPerSet += n;
        else if (db.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)         ssboPerSet += n;
        else if (db.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)          stImgPerSet += n;
    }
    std::vector<VkDescriptorPoolSize> poolSizes;
    if (uboPerSet) {
        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps.descriptorCount = maxSets * uboPerSet;
        poolSizes.push_back(ps);
    }
    if (imgPerSet) {
        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = maxSets * imgPerSet;
        poolSizes.push_back(ps);
    }
    if (ssboPerSet) {
        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = maxSets * ssboPerSet;
        poolSizes.push_back(ps);
    }
    if (stImgPerSet) {
        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps.descriptorCount = maxSets * stImgPerSet;
        poolSizes.push_back(ps);
    }

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = maxSets;
    dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    dpci.pPoolSizes = poolSizes.data();

    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(b->device, &dpci, nullptr, &pool) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return pool;
}

void ensure_program_layouts(GLuint program,
                            const uint32_t* vs, int vs_words,
                            const uint32_t* fs, int fs_words,
                            const uint32_t* cs, int cs_words) {
    Backend* b = backend();
    if (!b->initialized || program == 0) return;

    auto& tbl = program_table();
    ProgramResources& pr = tbl[program];
    if (pr.layoutsBuilt) return;

    // Reflect + merge every stage the program has. reflect_stage() returns an
    // empty vector for a null/zero-length module, so a graphics program simply
    // contributes nothing from the compute slot and vice versa.
    pr.bindings.clear();
    pr.bindings = reflect_stage(vs, vs_words, VK_SHADER_STAGE_VERTEX_BIT);
    merge_bindings(pr.bindings, reflect_stage(fs, fs_words, VK_SHADER_STAGE_FRAGMENT_BIT));
    merge_bindings(pr.bindings, reflect_stage(cs, cs_words, VK_SHADER_STAGE_COMPUTE_BIT));

    // Deterministic ordering for stable set-layout construction.
    std::sort(pr.bindings.begin(), pr.bindings.end(),
              [](const DescriptorBinding& a, const DescriptorBinding& c) {
                  if (a.set != c.set) return a.set < c.set;
                  return a.binding < c.binding;
              });

    pr.layoutsBuilt = true;  // set early so a reflection failure doesn't retry forever
    if (pr.bindings.empty()) {
        // No descriptors: caller falls back to the process-wide empty layout.
        return;
    }

    /* ---- Decide which UBO bindings become UNIFORM_BUFFER_DYNAMIC ----
     *
     * Only the bindings this layer sources from the transient arena are made
     * dynamic — i.e. the synthetic global block and any per-loose-uniform UBO.
     * An APPLICATION-declared block (glBindBufferBase) is deliberately left as
     * a plain VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
     *
     *   - Its descriptor already points straight at the app's own VkBuffer at
     *     the app's own offset, so it has no aliasing hazard to fix and
     *     nothing to gain from a dynamic offset.
     *   - It is written with range = VK_WHOLE_SIZE when the app used
     *     glBindBufferBase. A dynamic descriptor with VK_WHOLE_SIZE is a trap:
     *     vkCmdBindDescriptorSets requires effectiveOffset + range <= buffer
     *     size, which a non-zero dynamic offset would immediately violate.
     *   - Keeping it untouched means that path's behaviour is bit-for-bit
     *     what it was before this change.
     *
     * Restricting the dynamic set this way also keeps the count at 1 for
     * essentially every real program, which is what makes the MoltenVK cap
     * below a non-issue in practice rather than a lurking failure.
     */
    const uint32_t dynamicCap = ubo_arena_max_dynamic_ubos();
    mithril::Program* layoutProg = mithril::state_get_program(program);
    std::vector<uint32_t> wantDynamic;
    uint32_t wantDynamicDescriptors = 0;
    for (const auto& db : pr.bindings) {
        if (db.type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) continue;
        const bool appBlock =
            layoutProg &&
            layoutProg->blockIndexForDescriptor.count(db.binding) != 0;
        if (appBlock) continue;
        /* A UBO ARRAY would need one dynamic offset per element, in element
         * order, and the per-draw path only ever writes/offsets element 0.
         * Declaring it dynamic would make dynamicOffsetCount disagree with the
         * layout, which is a hard vkCmdBindDescriptorSets violation rather
         * than a wrong-pixels bug. Leave arrays non-dynamic. */
        if (db.descriptorCount > 1) continue;
        wantDynamic.push_back(db.binding);
        wantDynamicDescriptors += 1u;
    }
    /* MoltenVK / Metal 2 degradation path.
     *
     * maxDescriptorSetUniformBuffersDynamic is 8 on several Apple GPUs (it is
     * also the Vulkan spec floor). Exceeding it makes vkCreateDescriptorSet
     * Layout fail outright, which would leave the program with no layout and
     * silently drop every draw that uses it. So if the program wants more
     * dynamic UBOs than the device allows, fall back to plain uniform buffers
     * for ALL of them: the arena is still used (so the same-frame overwrite
     * bug stays fixed), but the slice offset travels in
     * VkDescriptorBufferInfo::offset instead of pDynamicOffsets. The only cost
     * is a lower descriptor-set reuse rate, because a moving offset now
     * changes the set's contents.
     */
    if (!wantDynamic.empty() && wantDynamicDescriptors <= dynamicCap) {
        pr.uboDynamic = true;
        pr.dynamicUboBindings = std::move(wantDynamic);
    } else {
        pr.uboDynamic = false;
        pr.dynamicUboBindings.clear();
        if (!wantDynamic.empty()) {
            MITHRIL_LOG_WARN("vk", "program %u wants %u dynamic UBO descriptors but the "
                             "device allows %u — falling back to non-dynamic uniform "
                             "buffers (still arena-backed, just fewer descriptor-set "
                             "cache hits)",
                             program, wantDynamicDescriptors, dynamicCap);
        }
    }

    // ---- VkDescriptorSetLayout ----
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.reserve(pr.bindings.size());
    for (const auto& db : pr.bindings) {
        VkDescriptorSetLayoutBinding lb{};
        lb.binding = db.binding;
        lb.descriptorType = descriptor_type_for(pr, db);
        lb.descriptorCount = db.descriptorCount;
        lb.stageFlags = db.stageMask;
        lb.pImmutableSamplers = nullptr;
        layoutBindings.push_back(lb);
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    dslci.pBindings = layoutBindings.data();
    if (vkCreateDescriptorSetLayout(b->device, &dslci, nullptr, &pr.descriptorSetLayout) != VK_SUCCESS) {
        MITHRIL_LOG_WARN("vk", "vkCreateDescriptorSetLayout failed (program %u)", program);
        pr.bindings.clear();
        return;
    }

    // ---- VkPipelineLayout (single set + gl_VertexID push constant) ----
    // Root cause: gl_VertexID baseVertex semantics. Every vertex shader now
    // carries a `_MithrilBaseVertex` push-constant block (see
    // Shader.cpp:inject_vertex_id_fixup) so gl_VertexID == gl_VertexIndex +
    // baseVertex. The pipeline layout MUST declare a matching VERTEX-stage
    // push-constant range (offset 0, size 4 = int _mithrilBaseVertex), or
    // MoltenVK rejects the shader at pipeline creation (VUID: the range must
    // cover every push-constant block the shader accesses). Declaring the
    // range unconditionally is safe — shaders that never touch the block
    // simply never read it, and Drawing.cpp writes 0 on every draw.
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &pr.descriptorSetLayout;
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.offset = 0;
    pcr.size = 4;  // int _mithrilBaseVertex
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(b->device, &plci, nullptr, &pr.pipelineLayout) != VK_SUCCESS) {
        MITHRIL_LOG_WARN("vk", "vkCreatePipelineLayout failed (program %u)", program);
        vkDestroyDescriptorSetLayout(b->device, pr.descriptorSetLayout, nullptr);
        pr.descriptorSetLayout = VK_NULL_HANDLE;
        pr.bindings.clear();
        return;
    }

    // ---- VkDescriptorPool (one per frame-in-flight slot) ----
    // maxSets=1024, descriptors per type sized per-set (see create_program_pool),
    // PER SLOT. Created with FREE_DESCRIPTOR_SET_BIT so individual sets CAN be
    // freed (useful if a layout is ever destroyed while cached sets survive);
    // in practice we never free individual sets — the pool is destroyed
    // wholesale on program deletion. Per-slot pools prevent the UAF where a
    // shared pool reset invalidated descriptor sets still referenced by an
    // in-flight command buffer on another slot.
    //
    // We do NOT call vkResetDescriptorPool in the normal path. Instead,
    // bind_program_descriptors rewinds a per-slot cursor at the start of each
    // frame (after the slot's fence wait) and reuses previously allocated
    // VkDescriptorSets. This avoids VUID-vkResetDescriptorPool-descriptorPool
    // -00313 entirely and mirrors MobileGL's UniformManager
    // (UniformManager.cpp:183,1026).
    const uint32_t kMaxSets = kMaxSetsPerPool;
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        pr.descriptorPools[i] = create_program_pool(pr, kMaxSets);
        if (pr.descriptorPools[i] == VK_NULL_HANDLE) {
            MITHRIL_LOG_WARN("vk", "vkCreateDescriptorPool failed (program %u, slot %d)", program, i);
            // Layout is still valid; bind_program_descriptors skips slots with a null pool.
        }
    }
}

/*
 * Process-wide default 1x1 opaque-black texture used to fill unbound sampler
 * bindings so the descriptor set stays complete. Without this, a shader that
 * declares a sampler but whose GL texture unit is unbound would leave the
 * corresponding descriptor binding unwritten — MoltenVK then samples
 * undefined memory and the geometry renders pure black (root cause L).
 *
 * MobileGL keeps a similar default texture for the same reason. Lazily
 * created on first use; lifetime is the process (never destroyed).
 */
struct DefaultTexture {
    VkImage    image  = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view   = VK_NULL_HANDLE;
    VkSampler  sampler = VK_NULL_HANDLE;
};

DefaultTexture& default_texture() {
    static DefaultTexture dt;
    if (dt.view != VK_NULL_HANDLE && dt.sampler != VK_NULL_HANDLE) return dt;
    Backend* b = backend();
    if (!b->device) return dt;

    // 1x1 R8G8B8A8_UNORM, opaque black (0,0,0,1).
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {1, 1, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(b->device, &ici, nullptr, &dt.image) != VK_SUCCESS) return dt;

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(b->device, dt.image, &req);
    // find_memory_type lives in Resources.cpp.
    uint32_t mt = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < 32; ++i) {
        if ((req.memoryTypeBits >> i) & 1) { mt = i; break; }
    }
    if (mt == 0xFFFFFFFFu) { vkDestroyImage(b->device, dt.image, nullptr); dt.image = VK_NULL_HANDLE; return dt; }

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    if (try_allocate_memory_with_gc(b->device, &ai, nullptr, &dt.memory) != VK_SUCCESS) {
        vkDestroyImage(b->device, dt.image, nullptr); dt.image = VK_NULL_HANDLE; return dt;
    }
    vkBindImageMemory(b->device, dt.image, dt.memory, 0);

    // Transition UNDEFINED -> SHADER_READ_ONLY_OPTIMAL and clear to black.
    // Use a one-shot command buffer on the graphics queue.
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = b->commandPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(b->device, &cai, &cb) != VK_SUCCESS) return dt;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    VkImageMemoryBarrier b2{};
    b2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b2.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.image = dt.image;
    b2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b2.subresourceRange.baseMipLevel = 0;
    b2.subresourceRange.levelCount = 1;
    b2.subresourceRange.baseArrayLayer = 0;
    b2.subresourceRange.layerCount = 1;
    b2.srcAccessMask = 0;
    b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b2);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VkFence fence;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(b->device, &fci, nullptr, &fence);
    vkQueueSubmit(b->graphicsQueue, 1, &si, fence);
    vkWaitForFences(b->device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(b->device, fence, nullptr);
    vkFreeCommandBuffers(b->device, b->commandPool, 1, &cb);

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = dt.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    if (vkCreateImageView(b->device, &vci, nullptr, &dt.view) != VK_SUCCESS) {
        // FIX (显存泄漏): view 创建失败时必须销毁已分配的 image+memory，
        // 否则下次重入 default_texture() 会被 vkCreateImage/vkAllocateMemory
        // 覆盖句柄，导致旧 VkImage+VkDeviceMemory 永久泄漏。每次重试都会
        // 累积一组泄漏，在显存紧张时加剧 OOM。
        vkDestroyImage(b->device, dt.image, nullptr); dt.image = VK_NULL_HANDLE;
        vkFreeMemory(b->device, dt.memory, nullptr); dt.memory = VK_NULL_HANDLE;
        return dt;
    }

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.minLod = 0.0f;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(b->device, &sci, nullptr, &dt.sampler) != VK_SUCCESS) {
        // FIX (显存泄漏): sampler 创建失败时同样必须销毁 image+memory+view。
        vkDestroyImageView(b->device, dt.view, nullptr); dt.view = VK_NULL_HANDLE;
        vkDestroyImage(b->device, dt.image, nullptr); dt.image = VK_NULL_HANDLE;
        vkFreeMemory(b->device, dt.memory, nullptr); dt.memory = VK_NULL_HANDLE;
        return dt;
    }
    return dt;
}

bool ensure_default_texture_ready() {
    DefaultTexture& dt = default_texture();
    return dt.view != VK_NULL_HANDLE && dt.sampler != VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Per-draw fast path: hashing, link-time packing plans, bind deduplication
// ---------------------------------------------------------------------------

namespace {

constexpr uint64_t kFnvBasis = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

// FNV-1a. Used for two independent memos — the packed-UBO content hash
// (decides whether to re-upload) and the descriptor-set signature (decides
// whether to re-allocate + re-write a set). Both are per-program and live for
// at most one frame, so a collision needs adversarial input rather than luck.
inline uint64_t fnv1a(const void* data, size_t n, uint64_t h = kFnvBasis) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}

inline uint64_t fnv1a_u64(uint64_t v, uint64_t h) { return fnv1a(&v, sizeof(v), h); }

/* Vulkan handles are pointers in a 64-bit build but plain uint64_t in a 32-bit
 * one, so neither a pointer cast nor an integer cast is portable on its own.
 * Copying the object representation works for both. */
template <typename H>
inline uint64_t handle_bits(H h) {
    static_assert(sizeof(H) <= sizeof(uint64_t), "unexpected Vulkan handle size");
    uint64_t v = 0;
    std::memcpy(&v, &h, sizeof(H));
    return v;
}

/* Resolve every UBO binding's data source ONCE per link.
 *
 * This is the whole point of UboBindingPlan: the old per-draw path did an
 * unordered_map<std::string, Uniform> lookup — string hash + bucket walk +
 * memcmp — for every member of every UBO on every draw. With Sodium issuing
 * thousands of draws a frame that is tens of thousands of string hashes per
 * frame, all of them recomputing an answer that only changes at link time.
 *
 * Holding `const Uniform*` is safe: unordered_map keeps pointers to elements
 * valid across inserts and rehashes, and Program::uniforms is only ever
 * cleared wholesale by glLinkProgram — which also calls
 * backend_delete_program_resources(), erasing the ProgramResources (and this
 * plan) outright. The linkVersion check below is the belt to that braces.
 */
void build_ubo_plans(ProgramResources& pr, const mithril::Program* prog) {
    pr.uboPlans.clear();
    for (const auto& db : pr.bindings) {
        if (db.type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) continue;
        pr.uboPlans.emplace_back();
        UboBindingPlan& plan = pr.uboPlans.back();
        plan.binding = db.binding;
        plan.size = db.bufferSize ? db.bufferSize : 16u;

        // (a) Application-declared block — the app owns the bytes; nothing to
        //     pack, only a buffer to look up at draw time.
        auto blk_it = prog->blockIndexForDescriptor.find(db.binding);
        if (blk_it != prog->blockIndexForDescriptor.end() &&
            blk_it->second < prog->blockInfos.size()) {
            plan.appBlock = true;
            plan.glBlockIndex = blk_it->second;
            continue;
        }

        // (b) One loose uniform whose value IS the block.
        auto uit = prog->uniforms.find(db.name);
        if (uit != prog->uniforms.end()) plan.directSrc = &uit->second;

        // (c) Aggregated $Global / mithril_GlobalBlock. Members with no
        //     matching uniform are dropped here instead of being re-looked-up
        //     and re-missed on every draw.
        plan.members.reserve(db.members.size());
        for (const auto& m : db.members) {
            auto mit = prog->uniforms.find(m.name);
            if (mit == prog->uniforms.end()) continue;
            UboMemberPlan mp;
            mp.slot.offset       = m.offset;
            mp.slot.size         = m.size;
            mp.slot.columns      = m.columns;
            mp.slot.rows         = m.rows;
            mp.slot.arraySize    = m.arraySize;
            mp.slot.arrayStride  = m.arrayStride;
            mp.slot.matrixStride = m.matrixStride;
            mp.src = &mit->second;
            plan.members.push_back(mp);
        }
        plan.scratch.assign(plan.size, 0);
    }
    pr.planLinkVersion = prog->linkVersion;
}

/* Shadow of the last vkCmdBindDescriptorSets issued, per bind point.
 *
 * Graphics and compute descriptor bindings are independent command-buffer
 * state, so one shared shadow would let a dispatch suppress a rebind the next
 * draw genuinely needs. Comparing the pipeline layout (not just the set) is
 * what makes the skip safe: a bind with a different layout is the only thing
 * that can disturb the existing binding, and every such bind goes through
 * this function and updates the shadow.
 */
struct BindShadow {
    VkCommandBuffer       cb = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkDescriptorSet       set = VK_NULL_HANDLE;
    std::vector<uint32_t> dynOffsets;
    bool                  valid = false;
};

BindShadow g_bindShadow[2];  // [0] = graphics, [1] = compute

inline int bind_point_index(VkPipelineBindPoint bp) {
    return bp == VK_PIPELINE_BIND_POINT_COMPUTE ? 1 : 0;
}

} // namespace

void on_command_buffer_boundary() {
    for (auto& sh : g_bindShadow) {
        sh.valid = false;
        sh.cb = VK_NULL_HANDLE;
        sh.layout = VK_NULL_HANDLE;
        sh.set = VK_NULL_HANDLE;
        sh.dynOffsets.clear();  // keeps capacity; only the contents are stale
    }
    // A command-buffer boundary means the buffer was flushed/re-begun or its
    // cached descriptor sets were dropped; no valid set is bound until
    // bind_program_descriptors() issues a fresh vkCmdBindDescriptorSets. Clear
    // the flag so backend_draw_* refuses a draw with unbound descriptors.
    mithril::vk::set_descriptors_bound(false);
}

void bind_program_descriptors(GLuint program, VkPipelineBindPoint bindPoint) {
    Backend* b = backend();
    if (!b->initialized || !b->commandBuffer || program == 0) return;
    // Ensure the current slot's command buffer is recording before we issue
    // vkCmdBindDescriptorSets. This is normally a no-op (bind_program_descriptors
    // is called during a render pass, when the buffer is already recording),
    // but the ensure call makes this function safe if called outside a pass.
    if (!ensure_command_buffer_recording()) return;

    auto& tbl = program_table();
    auto it = tbl.find(program);
    if (it == tbl.end()) return;
    ProgramResources& pr = it->second;
    int slot = b->currentFrame;
    // Empty-bindings programs FIRST: a program with no reflected descriptors
    // (no UBO, no sampler — e.g. a trivial `#version 330 core` shader that only
    // positions vertices and outputs a constant color) uses the process-wide
    // empty pipeline layout, so pr.pipelineLayout and pr.descriptorPools[] stay
    // VK_NULL_HANDLE (ensure_program_layouts returns early for empty bindings).
    // Those NULLs are NOT a "not ready" state — they are the normal, correct
    // state for such a program. If we let the layout/pool guard below run
    // first, it would wrongly treat this program as unready and bail without
    // calling set_descriptors_bound(true), leaving descriptorsBound=false so
    // backend_draw_* drops every draw for this program -> nothing rasterizes
    // (silent black framebuffer, no GL error). Handle it here instead.
    if (pr.bindings.empty()) {
        // No descriptor bindings reflected for this program -> its pipeline uses
        // the process-wide empty layout (zero descriptor sets). No
        // vkCmdBindDescriptorSets is needed or possible; a draw is valid with
        // nothing bound. Mark the descriptor requirement as satisfied so the
        // descriptors_bound() guard does not wrongly drop these draws.
        mithril::vk::set_descriptors_bound(true);
        return;
    }
    if (!pr.layoutsBuilt || pr.pipelineLayout == VK_NULL_HANDLE ||
        pr.descriptorPools[slot] == VK_NULL_HANDLE) {
        // Program not ready (layouts not built / no layout / no pool). Bail: the
        // descriptors_bound() guard in backend_draw_* will drop the draw, which is
        // safer than recording it with an unbound descriptor set (red + possible
        // GPU page fault).
        return;
    }

    mithril::Program* prog = mithril::state_get_program(program);
    if (!prog) return;

    /* Fixed-up packing plans. Built lazily here rather than in
     * ensure_program_layouts() because that runs during link, before the
     * frontend has necessarily registered every uniform; by the first draw the
     * uniform table is complete. Rebuilt only when the program is relinked. */
    if (pr.planLinkVersion != prog->linkVersion) build_ubo_plans(pr, prog);

    // Per-slot cursor rewind at the start of each frame. ensure_command_buffer
    // _recording() (called above) has ALREADY waited on frameFences[currentFrame]
    // before this point — so the command buffer submitted to this slot
    // kMaxFramesInFlight frames ago, and all of MoltenVK's Metal encoding of
    // its descriptor references, is guaranteed complete on the GPU. Rewinding
    // the cursor is therefore safe: no in-flight command buffer references
    // the sets we are about to reuse.
    //
    // We do NOT call vkResetDescriptorPool (which would invalidate all sets
    // and require re-allocation). Instead, previously allocated sets are
    // reused in order — vkUpdateDescriptorSets below rewrites their contents
    // with the current frame's uniform/texture bindings. This mirrors
    // MobileGL's UniformManager (UniformManager.cpp:183,1026), which also
    // avoids vkResetDescriptorPool in favor of cursor rewind + set reuse,
    // eliminating any VUID-vkResetDescriptorPool-descriptorPool-00313 hazard.
    //
    // FIX (mid-frame flush 缓存失效 - P0): ensure_command_buffer_recording 在
    // 帧中间被 safe_device_wait_idle 触发时会再次 rewind arena 并 bump
    // flushGeneration。本帧前半段写入的 descMemo 条目持有 flush 前的 set
    // 内容（arena 偏移已失效），必须一并作废，否则后续 draw 会复用指向
    // 已被覆盖 arena 内存的 set。
    if (pr.lastFrameGen[slot] != b->frameGeneration ||
        pr.lastFlushGen[slot] != b->flushGeneration) {
        pr.setCursor[slot] = 0;
        pr.lastFrameGen[slot] = b->frameGeneration;
        pr.lastFlushGen[slot] = b->flushGeneration;
        // FIX (mid-frame flush stale-set cache - mirror invalidate_descriptor_memo):
        // a safe_device_wait_idle() flush submits the current buffer, re-begins the
        // SAME slot, and (via drain_disposal_queues_except / the pre-vkCreateImage GC
        // and delete_program_resources paths) can FREE resources that previously-
        // allocated sets in this slot's cache still reference. Reusing such a set
        // (setCursor already rewound to 0 above) relies on the rewrite to refresh
        // it — which is skipped on a bail path (incomplete-set guard, DescriptorSet
        // ~1299). Drop the whole per-slot set cache so the next bind allocates a
        // BRAND-NEW set rewritten from current, still-valid resources.
        pr.allocatedSets[slot].clear();
        /* The cached sets this slot owns are about to be handed out again and
         * REWRITTEN by this frame's draws, so every memo entry naming one is
         * now a promise we can no longer keep. Dropping the memo here is what
         * makes the reuse below sound: within a frame the cursor only moves
         * forward, so a set already in the memo is never re-taken. */
        for (int i = 0; i < kDescriptorMemoSize; ++i) {
            pr.descMemo[slot][i] = DescriptorMemoEntry{};
        }
        pr.descMemoNext[slot] = 0;
    }

    /* Gather descriptor writes.
     *
     * static so the four vectors keep their capacity between draws — this
     * function runs thousands of times a frame and used to heap-allocate
     * (and free) them every single time. Safe because the GL context is
     * single-threaded and nothing here re-enters. The reserves guarantee no
     * reallocation, which matters: the VkWriteDescriptorSet structs hold raw
     * pointers into bufInfos/imgInfos.
     *
     * Note the set is NOT chosen yet — dstSet is filled in after the
     * signature below decides whether an existing set already holds exactly
     * this content.
     */
    static std::vector<VkWriteDescriptorSet>   writes;
    static std::vector<VkDescriptorBufferInfo> bufInfos;
    static std::vector<VkDescriptorImageInfo>  imgInfos;
    static std::vector<uint32_t>               dynOffsets;
    writes.clear();
    bufInfos.clear();
    imgInfos.clear();
    dynOffsets.clear();
    bufInfos.reserve(pr.bindings.size());
    imgInfos.reserve(pr.bindings.size());
    writes.reserve(pr.bindings.size());

    /* Running signature of the set's CONTENT. Seeded with the layout so two
     * programs can never share a memo entry, and deliberately NOT fed the
     * dynamic offsets: those travel in pDynamicOffsets, outside the set, which
     * is exactly what lets a set be reused across draws that only changed a
     * uniform value. */
    uint64_t sig = fnv1a_u64(handle_bits(pr.descriptorSetLayout), kFnvBasis);
    size_t uboPlanIdx = 0;

    for (const auto& db : pr.bindings) {
        if (db.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            // uboPlans is built from this very list with this very filter, so
            // a counter replaces any lookup. Running short can only mean the
            // plan is out of step with the reflection, which the linkVersion
            // gate makes impossible — abandon the bind rather than emit a set
            // with a hole in it (and, worse, a short pDynamicOffsets).
            if (uboPlanIdx >= pr.uboPlans.size()) return;
            UboBindingPlan& plan = pr.uboPlans[uboPlanIdx++];
            const bool dyn = binding_is_dynamic_ubo(pr, db.binding);

            VkBuffer     ubuf   = VK_NULL_HANDLE;
            VkDeviceSize uoff   = 0;
            VkDeviceSize urange = VK_WHOLE_SIZE;

            if (plan.appBlock) {
                /* ---- A uniform block the APPLICATION declared (root cause AL).
                 *
                 * Not synthesised from loose uniforms and never packed from
                 * Program::uniforms — the application owns the memory and
                 * fills it through glBufferData/glBufferSubData. All that is
                 * needed is to route the descriptor at the buffer the app
                 * bound, which takes two hops GL defines and Vulkan does not:
                 *
                 *   descriptor binding -> GL block index   (resolved at link)
                 *   GL block index     -> GL binding point (glUniformBlockBinding)
                 *   GL binding point   -> GL buffer        (glBindBufferBase)
                 *
                 * These are also never made dynamic (see
                 * ensure_program_layouts), so no dynamic offset is pushed for
                 * them and the arena is not involved at all.
                 */
                const mithril::UniformBlockInfo& info = prog->blockInfos[plan.glBlockIndex];

                GLuint point = info.bindingPoint;
                auto pit = prog->uniformBlockBindings.find(plan.glBlockIndex);
                if (pit != prog->uniformBlockBindings.end()) point = pit->second;

                if (point < (GLuint)mithril::kMaxIndexedBindings) {
                    const auto& sl = mithril::g_state->indexedBufferBindings
                                         [(int)mithril::IndexedBufferTarget::Uniform][point];
                    if (sl.name) {
                        ubuf = backend_get_buffer(sl.name);
                        if (ubuf != VK_NULL_HANDLE && sl.hasExplicitRange) {
                            uoff   = (VkDeviceSize)sl.offset;
                            urange = (VkDeviceSize)sl.size;
                        }
                    }
                }
                if (ubuf == VK_NULL_HANDLE) {
                    // Nothing bound to that point yet. Leaving the descriptor
                    // unwritten would make the set incomplete and MoltenVK
                    // would read undefined memory, so point it at a correctly
                    // sized zero buffer instead: the draw renders as if every
                    // block member were 0 rather than corrupting. The buffer
                    // is all zeros and never mutated, so unlike the old
                    // per-(program,binding) uniform buffer it carries no
                    // same-frame aliasing hazard.
                    uint32_t zsz = info.dataSize > 0 ? (uint32_t)info.dataSize
                                                     : (db.bufferSize ? db.bufferSize : 16u);
                    std::vector<uint8_t> zeros(zsz, 0);
                    GLuint zname = program * 1000000u + db.binding + 1u;
                    ubuf = backend_get_or_create_buffer(zname, zeros.data(), zeros.size());
                    uoff = 0;
                    urange = VK_WHOLE_SIZE;
                    static int warned = 0;
                    if (warned < 5) {
                        ++warned;
                        MITHRIL_LOG_WARN("vk", "uniform block '%s' (program %u, binding %u) "
                                         "has no buffer bound at GL binding point %u; "
                                         "using zeros", info.name.c_str(), program,
                                         db.binding, point);
                    }
                }
            } else {
                /* ---- Layer-synthesised block: pack, hash, maybe upload ----
                 *
                 * The pack itself is unavoidable (nothing tells us a uniform
                 * changed), but it now runs against link-time resolved
                 * `Uniform*` instead of re-hashing member names, and into a
                 * persistent scratch buffer instead of a fresh heap vector.
                 *
                 * The memset keeps this bit-identical to the old
                 * `std::vector<uint8_t> payload(sz, 0)`: a member with no
                 * value yet must read as zero, not as whatever the previous
                 * draw left in the scratch.
                 */
                if (plan.scratch.size() != plan.size) plan.scratch.assign(plan.size, 0);
                std::memset(plan.scratch.data(), 0, plan.scratch.size());
                if (plan.directSrc && !plan.directSrc->value.empty()) {
                    const size_t bytes = plan.directSrc->value.size() * sizeof(float);
                    std::memcpy(plan.scratch.data(), plan.directSrc->value.data(),
                                std::min(bytes, plan.scratch.size()));
                } else {
                    for (const auto& mp : plan.members) {
                        if (!mp.src || mp.src->value.empty()) continue;
                        pack_std140(plan.scratch.data(), plan.scratch.size(), mp.slot,
                                    mp.src->value.data(), mp.src->value.size());
                    }
                }

                /* Content hash decides whether this draw needs its own bytes.
                 *
                 * The previous slice is only reusable inside the same
                 * (slot, frame, flush): a slot's arena is rewound once its
                 * fence is signalled (after which those bytes belong to
                 * whoever bump-allocates them next), and ALSO when a mid-frame
                 * flush re-begins the command buffer (flushGeneration bump) —
                 * a stale slice there would be overwritten by later uploads.
                 */
                const uint64_t h = fnv1a(plan.scratch.data(), plan.scratch.size());
                const bool sliceLive = plan.lastValid &&
                                       plan.lastBuffer != VK_NULL_HANDLE &&
                                       plan.lastSlot == slot &&
                                       plan.lastFrameGen == b->frameGeneration &&
                                       plan.lastFlushGen == b->flushGeneration;
                if (!sliceLive || plan.lastHash != h) {
                    UboSlice sl;
                    if (!ubo_arena_upload(slot, plan.scratch.data(),
                                          (VkDeviceSize)plan.scratch.size(), sl)) {
                        /* Arena exhausted (≈51 MB of uniforms in one frame —
                         * a runaway, not a workload). Falling back to a shared
                         * buffer would reintroduce exactly the same-frame
                         * overwrite bug this change exists to remove, and
                         * skipping just this binding would leave
                         * dynamicOffsetCount disagreeing with the layout —
                         * a hard vkCmdBindDescriptorSets violation. Drop the
                         * whole bind instead. */
                        plan.lastValid = false;
                        static int arenaFailCount = 0;
                        if (++arenaFailCount <= 3) {
                            MITHRIL_LOG_WARN("vk", "ubo arena exhausted (program %u, binding %u, "
                                             "%zu bytes) — skipping the descriptor bind for "
                                             "this draw", program, db.binding,
                                             plan.scratch.size());
                        }
                        return;
                    }
                    plan.lastBuffer   = sl.buffer;
                    plan.lastOffset   = sl.offset;
                    plan.lastHash     = h;
                    plan.lastValid    = true;
                    plan.lastSlot     = slot;
                    plan.lastFrameGen = b->frameGeneration;
                    plan.lastFlushGen = b->flushGeneration;
                }
                ubuf   = plan.lastBuffer;
                uoff   = plan.lastOffset;
                // An exact range (never VK_WHOLE_SIZE) — a dynamic descriptor
                // requires offset + dynamicOffset + range <= buffer size, and
                // VK_WHOLE_SIZE would measure to the end of the whole arena
                // block instead of this slice.
                urange = plan.size;
            }

            if (ubuf == VK_NULL_HANDLE) {
                /* Only reachable on the app-block path (the arena path returns
                 * outright on failure), and app blocks are never dynamic — but
                 * if that ever changes, silently skipping the binding would
                 * make dynamicOffsetCount disagree with the layout, so make
                 * the assumption enforce itself. */
                if (dyn) return;
                continue;
            }

            VkDescriptorBufferInfo bi{};
            bi.buffer = ubuf;
            bi.range  = urange;
            if (dyn) {
                /* The moving part of a dynamic descriptor lives in
                 * pDynamicOffsets, NOT in the set — which is precisely what
                 * keeps the signature below stable across draws that only
                 * changed a uniform value. pDynamicOffsets is consumed in
                 * ascending binding order and pr.bindings is sorted that way,
                 * so appending here produces the right order. */
                bi.offset = 0;
                dynOffsets.push_back(static_cast<uint32_t>(uoff));
            } else {
                bi.offset = uoff;
            }
            bufInfos.push_back(bi);

            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = VK_NULL_HANDLE;  // filled in once the set is chosen
            w.dstBinding = db.binding;
            w.dstArrayElement = 0;
            w.descriptorCount = 1;
            // Must match what the layout declared, or the write is rejected.
            w.descriptorType = descriptor_type_for(pr, db);
            w.pBufferInfo = &bufInfos.back();
            writes.push_back(w);

            sig = fnv1a_u64(((uint64_t)db.binding << 32) | (uint64_t)w.descriptorType, sig);
            sig = fnv1a_u64(handle_bits(bi.buffer), sig);
            sig = fnv1a_u64((uint64_t)bi.offset, sig);
            sig = fnv1a_u64((uint64_t)bi.range, sig);
        } else if (db.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            /* Resolve descriptor binding -> GL texture unit.
             *
             * The old code assumed `binding == unit`. That was never a safe
             * assumption and became actively wrong once the global uniform
             * block took binding 0: Minecraft's Sampler0 (unit 0) landed on
             * binding 1 and therefore sampled whatever was bound to unit 1 —
             * usually the lightmap, or nothing at all. The real mapping is the
             * integer the application passed to glUniform1i(), which
             * store_uniform_int() now records per binding.
             */
            GLint unit = -1;
            auto uit_s = prog->samplerUnitForBinding.find(db.binding);
            if (uit_s != prog->samplerUnitForBinding.end()) {
                unit = uit_s->second;
            } else {
                unit = static_cast<GLint>(db.binding);  // legacy fallback
            }
            GLuint tex_id = 0;
            if (unit >= 0 && unit < mithril::kMaxTextureUnits) {
                // FIX (Main-menu panorama cubemap GPU fault root cause - sampler
                // type): pick the texture from the slot matching the sampler's
                // declared type (db.samplerTarget, reflected from SPIR-V
                // image.dim: samplerCube -> CubeMap slot). The old
                // boundTextureForUnit(unit) unconditionally preferred the 2D
                // slot: on the main menu unit 0 often still holds a GUI 2D
                // binding, so the panorama's samplerCube got a 2D view ->
                // MoltenVK viewType mismatch -> undefined sampling / GPU fault.
                mithril::TextureTarget tt = mithril::textureTargetFromGL(db.samplerTarget);
                if (tt != mithril::TextureTarget::Count) {
                    tex_id = mithril::g_state->boundTextureForUnit((GLuint)unit, tt);
                }
                if (tex_id == 0 && tt != mithril::TextureTarget::_2D) {
                    // Target slot unbound: fall back to any slot (rare shaders).
                    tex_id = mithril::g_state->boundTextureForUnit((GLuint)unit);
                }
            }
            /* No texture on that unit.
             *
             * The old code scanned for the first texture bound to ANY unit and
             * used that. It kept the descriptor set complete, which is why it
             * was written that way, but it substitutes an unrelated image:
             * with Minecraft's terrain shader an unbound Sampler0 would be fed
             * the lightmap, and the world renders in flat lighting colours
             * with no block textures — a plausible-looking picture that sends
             * you hunting in entirely the wrong place. The default 1x1 texture
             * below keeps the set complete without inventing data; a missing
             * texture should look missing. */
            if (!tex_id) {
                static int warned = 0;
                if (warned < 5) {
                    ++warned;
                    MITHRIL_LOG_WARN("vk", "sampler at binding %u resolves to texture unit %d, "
                                     "which has no texture bound (program %u); "
                                     "using the default texture",
                                     db.binding, unit, program);
                }
            }
            VkImageView view = VK_NULL_HANDLE;
            VkSampler samp = VK_NULL_HANDLE;
            if (tex_id) {
                view = backend_get_texture_view(tex_id);
                // FIX (Root Cause M - 采样器参数硬编码): 旧代码硬编码 GL_LINEAR/GL_REPEAT，
                // 完全忽略纹理通过 glTexParameteri 设置的真实 sampler 参数。
                // Minecraft 像素风纹理（GL_NEAREST）被双线性插值，图集纹理
                // （GL_CLAMP_TO_EDGE）被 REPEAT → 采样到错误纹素 → 颜色偏红/花屏。
                // 修复：从 Texture 结构体读取真实 minFilter/magFilter/wrapS/wrapT/wrapR。
                // 采样器按纹理名缓存（Resources.cpp:793-830），首次绑定读取的参数
                // 即为最终参数。Minecraft 通常在纹理创建后立即设置参数并不再修改，
                // 因此无需额外缓存失效逻辑。
                mithril::Texture* tex = mithril::state_get_texture(tex_id);
                GLenum minF = tex ? (GLenum)tex->minFilter : GL_NEAREST_MIPMAP_LINEAR;
                GLenum magF = tex ? (GLenum)tex->magFilter : GL_LINEAR;
                GLenum wrapS = tex ? (GLenum)tex->wrapS : GL_REPEAT;
                GLenum wrapT = tex ? (GLenum)tex->wrapT : GL_REPEAT;
                GLenum wrapR = tex ? (GLenum)tex->wrapR : GL_REPEAT;
                samp = backend_get_or_create_sampler(
                    tex_id, minF, magF, wrapS, wrapT, wrapR, nullptr);
            }
            // FIX (root cause L): if no texture is bound (or the bound texture
            // has no view/sampler), use the process-wide default 1x1 black
            // texture. Without this, the descriptor binding would be left
            // unwritten — MoltenVK then samples undefined memory and the
            // geometry renders pure black. A complete descriptor set with a
            // valid (black) texture is always preferable to an incomplete one.
            if (view == VK_NULL_HANDLE || samp == VK_NULL_HANDLE) {
                DefaultTexture& dt = default_texture();
                if (dt.view != VK_NULL_HANDLE && dt.sampler != VK_NULL_HANDLE) {
                    view = dt.view;
                    samp = dt.sampler;
                }
            }
            // DEBUG (sampler black-screen triage): log how each sampler binding
            // resolved so CI can tell a real-texture bind (tex_id != 0) from the
            // default-black fallback, and whether the view/sampler are valid.
            MITHRIL_LOG_DEBUG("vk", "sampler bind prog=%u binding=%u unit=%d tex=%u "
                              "viewValid=%d sampValid=%d (default view=%d samp=%d)",
                              program, db.binding, unit, tex_id,
                              view != VK_NULL_HANDLE, samp != VK_NULL_HANDLE,
                              default_texture().view != VK_NULL_HANDLE,
                              default_texture().sampler != VK_NULL_HANDLE);
            if (view != VK_NULL_HANDLE && samp != VK_NULL_HANDLE) {
                // FIX (Root Cause AH - depth-stencil descriptor layout):
                // 旧代码硬编码 imageLayout = SHADER_READ_ONLY_OPTIMAL，对 depth-stencil
                // 纹理不匹配。descriptor 声明的 imageLayout 必须与 image 实际布局一致，
                // 否则 MoltenVK 验证错误或静默丢 draw → 黑屏。
                // 通过 TextureEntry.format 查找纹理的 VkFormat，用 sampled_layout_for_format
                // 选择正确的 read-only 布局（depth-stencil -> DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                // depth-only -> DEPTH_READ_ONLY_OPTIMAL, color -> SHADER_READ_ONLY_OPTIMAL）。
                // 对照 MobileGL ResolveSampledReadOnlyLayout (VkTextureManager.cpp:177)。
                //
                // 注意：当 view 是 default fallback texture (R8G8B8A8_UNORM, color) 时，
                // tex_it->second.view != view，保持 SHADER_READ_ONLY_OPTIMAL（default 是 color 格式）。
                VkImageLayout sampledLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                if (tex_id) {
                    auto& tex_tbl = mithril::vk::texture_table();
                    auto tex_it = tex_tbl.find(tex_id);
                    if (tex_it != tex_tbl.end() && tex_it->second.view == view) {
                        sampledLayout = mithril::vk::sampled_layout_for_format(tex_it->second.format);

                        // FIX (真机主菜单 page fault 诊断 - CRITICAL):
                        // 真机在 atlas 创建后、主菜单首帧采样时触发 GPU address
                        // fault (kIOGPUCommandBufferCallbackErrorPageFault)，而 CI
                        // 全绿（2x2 小纹理不触发）。这里在 descriptor bind 时校验
                        // 即将被采样的真实纹理资源状态，若存在异常则 WARN（限流），
                        // 让用户无需开 MITHRIL_DEBUG（真机默认 Warning）也能定位
                        // fault 前最后一次绑定的 atlas/纹理是哪个、坏在哪一环。
                        //   a) image/view 为 NULL 或 view 与 descriptor 不一致 → 采
                        //      样无效 GPU 地址；
                        //   b) currentLayout 不是 read-only 采样布局（如停在
                        //      TRANSFER_DST）而 descriptor 声明 read-only → 布局
                        //      mismatch，MoltenVK/A11 读错地址。
                        bool resBad = false;
                        if (tex_it->second.image == VK_NULL_HANDLE) { resBad = true; }
                        if (tex_it->second.view == VK_NULL_HANDLE) { resBad = true; }
                        if (tex_it->second.currentLayout != sampledLayout) { resBad = true; }
                        if (resBad) {
                            static int layoutWarn = 0;
                            if (layoutWarn <= 8) {
                                MITHRIL_LOG_WARN("vk",
                                    "sampler layout/resource check tex=%u imgValid=%d "
                                    "viewValid=%d curLayout=%d declLayout=%d levels=%d",
                                    tex_id, tex_it->second.image != VK_NULL_HANDLE,
                                    tex_it->second.view != VK_NULL_HANDLE,
                                    (int)tex_it->second.currentLayout,
                                    (int)sampledLayout, (int)tex_it->second.levels);
                            }
                            layoutWarn++;
                        }
                    }
                }
                VkDescriptorImageInfo ii{};
                ii.sampler = samp;
                ii.imageView = view;
                ii.imageLayout = sampledLayout;
                imgInfos.push_back(ii);
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = VK_NULL_HANDLE;  // filled in once the set is chosen
                w.dstBinding = db.binding;
                w.dstArrayElement = 0;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo = &imgInfos.back();
                writes.push_back(w);

                sig = fnv1a_u64(((uint64_t)db.binding << 32) | (uint64_t)w.descriptorType, sig);
                sig = fnv1a_u64(handle_bits(ii.sampler), sig);
                sig = fnv1a_u64(handle_bits(ii.imageView), sig);
                sig = fnv1a_u64((uint64_t)ii.imageLayout, sig);
            }
        } else if (db.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            /* ---- Shader storage block (SSBO) ----
             *
             * 两跳间接，和应用声明的 uniform block 同构：
             *
             *   descriptor binding -> storage block index  (storageBlockInfos)
             *   storage block index -> GL binding point    (storageBlockBindings)
             *   GL binding point    -> GL buffer           (glBindBufferBase/Range)
             *
             * FIX（索引空间混用）：这里原先直接写
             *     storageBlockBindings.find(db.binding)
             * 把 descriptor binding 当成 storage block index 去查。但这两个是
             * 不同的编号空间：
             *   - storage block index 是 SSBO 在着色器里的**声明顺序**，由
             *     glLinkProgram 按 si 递增赋予，也是 glShaderStorageBlockBinding
             *     和 glGetProgramResourceIndex 使用的编号；
             *   - descriptor binding 是 glslang 生成 SPIR-V 时分配的 Vulkan 槽位。
             * 只有在"声明顺序恰好等于 binding 号"时两者才碰巧相等。一旦着色器
             * 写了非连续的 layout(binding=)（Iris 的 shadow/deferred pass 常见），
             * Iris 调 glShaderStorageBlockBinding 重定向就会绑到错误的 buffer，
             * 表现为 SSBO 读到别的 pass 的数据（画面错乱但不崩，极难定位）。
             *
             * 正确做法：先用 descriptorBinding 反查出 block index，再查绑定点。
             */
            GLuint blockIndex = db.binding;   // 兜底：反查不到时沿用旧行为
            for (size_t bi = 0; bi < prog->storageBlockInfos.size(); ++bi) {
                if (prog->storageBlockInfos[bi].descriptorBinding == db.binding) {
                    blockIndex = (GLuint)bi;
                    break;
                }
            }
            GLuint point = blockIndex;
            auto sit = prog->storageBlockBindings.find(blockIndex);
            if (sit != prog->storageBlockBindings.end()) point = sit->second;

            VkBuffer ssbo = VK_NULL_HANDLE;
            VkDeviceSize ssboOff = 0;
            VkDeviceSize ssboRange = VK_WHOLE_SIZE;
            if (point < (GLuint)mithril::kMaxIndexedBindings) {
                const auto& sl = mithril::g_state->indexedBufferBindings
                                     [(int)mithril::IndexedBufferTarget::ShaderStorage][point];
                if (sl.name) {
                    ssbo = backend_get_buffer(sl.name);
                    if (ssbo != VK_NULL_HANDLE && sl.hasExplicitRange) {
                        ssboOff   = (VkDeviceSize)sl.offset;
                        ssboRange = (VkDeviceSize)sl.size;
                    }
                }
            }
            if (ssbo == VK_NULL_HANDLE) {
                // Nothing bound at that point. Same reasoning as the UBO path:
                // an unwritten binding makes the whole set incomplete, so give
                // the shader a real (zeroed) buffer instead of undefined
                // memory. Writes land in scratch and are discarded.
                uint32_t zsz = db.bufferSize ? db.bufferSize : 16u;
                std::vector<uint8_t> zeros(zsz, 0);
                GLuint zname = program * 1000000u + db.binding + 1u;
                ssbo = backend_get_or_create_buffer(zname, zeros.data(), zeros.size());
                ssboOff = 0;
                ssboRange = VK_WHOLE_SIZE;
                static int warnedSsbo = 0;
                if (warnedSsbo < 5) {
                    ++warnedSsbo;
                    MITHRIL_LOG_WARN("vk", "storage block '%s' (program %u, binding %u) "
                                     "has no buffer bound at GL binding point %u; "
                                     "using zeros", db.name.c_str(), program,
                                     db.binding, point);
                }
            }
            if (ssbo != VK_NULL_HANDLE) {
                VkDescriptorBufferInfo bi{};
                bi.buffer = ssbo;
                bi.offset = ssboOff;
                bi.range  = ssboRange;
                bufInfos.push_back(bi);
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = VK_NULL_HANDLE;  // filled in once the set is chosen
                w.dstBinding = db.binding;
                w.dstArrayElement = 0;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w.pBufferInfo = &bufInfos.back();
                writes.push_back(w);

                sig = fnv1a_u64(((uint64_t)db.binding << 32) | (uint64_t)w.descriptorType, sig);
                sig = fnv1a_u64(handle_bits(bi.buffer), sig);
                sig = fnv1a_u64((uint64_t)bi.offset, sig);
                sig = fnv1a_u64((uint64_t)bi.range, sig);
            }
        } else if (db.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            /* ---- Storage image (glBindImageTexture) ----
             *
             * Resolved exactly like a combined image sampler — the binding is
             * mapped to a GL unit through samplerUnitForBinding, because an
             * `image2D` uniform is an opaque uniform and glUniform1i() points
             * it at an IMAGE unit the same way it points a sampler at a
             * texture unit. The one difference is the table it indexes:
             * glBindImageTexture maintains its own unit array, separate from
             * glBindTexture's.
             */
            GLint unit = -1;
            auto uit_i = prog->samplerUnitForBinding.find(db.binding);
            if (uit_i != prog->samplerUnitForBinding.end()) unit = uit_i->second;
            else                                            unit = (GLint)db.binding;

            GLuint tex_id = 0;
            if (unit >= 0 && unit < mithril::kMaxTextureUnits) {
                tex_id = mithril::g_state->imageTextureUnits[unit];
            }
            VkImageView view = VK_NULL_HANDLE;
            VkSampler samp = VK_NULL_HANDLE;
            if (tex_id) {
                view = backend_get_texture_view(tex_id);
                mithril::Texture* tex = mithril::state_get_texture(tex_id);
                GLenum minF = tex ? (GLenum)tex->minFilter : GL_NEAREST;
                GLenum magF = tex ? (GLenum)tex->magFilter : GL_NEAREST;
                GLenum wrapS = tex ? (GLenum)tex->wrapS : GL_CLAMP_TO_EDGE;
                GLenum wrapT = tex ? (GLenum)tex->wrapT : GL_CLAMP_TO_EDGE;
                GLenum wrapR = tex ? (GLenum)tex->wrapR : GL_CLAMP_TO_EDGE;
                samp = backend_get_or_create_sampler(
                    tex_id, minF, magF, wrapS, wrapT, wrapR, nullptr);
            }
            // A storage image ignores the sampler, but the descriptor write
            // still needs a view. With no image bound, fall back to the same
            // 1x1 default used for samplers so the set stays complete.
            VkImageLayout storageLayout = VK_IMAGE_LAYOUT_GENERAL;
            if (view == VK_NULL_HANDLE) {
                DefaultTexture& dt = default_texture();
                view = dt.view;
                samp = dt.sampler;
                // The fallback image was created SAMPLED-only, so GENERAL
                // would be a lie; declare the layout it is actually in.
                storageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                static int warnedImg = 0;
                if (warnedImg < 5) {
                    ++warnedImg;
                    MITHRIL_LOG_WARN("vk", "storage image at binding %u resolves to image "
                                     "unit %d, which has no texture bound (program %u); "
                                     "using the default texture", db.binding, unit, program);
                }
            }
            if (view != VK_NULL_HANDLE) {
                VkDescriptorImageInfo ii{};
                ii.sampler = samp;
                ii.imageView = view;
                ii.imageLayout = storageLayout;
                imgInfos.push_back(ii);
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = VK_NULL_HANDLE;  // filled in once the set is chosen
                w.dstBinding = db.binding;
                w.dstArrayElement = 0;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w.pImageInfo = &imgInfos.back();
                writes.push_back(w);

                sig = fnv1a_u64(((uint64_t)db.binding << 32) | (uint64_t)w.descriptorType, sig);
                sig = fnv1a_u64(handle_bits(ii.imageView), sig);
                sig = fnv1a_u64((uint64_t)ii.imageLayout, sig);
            }
        }
    }

    /* ---- Descriptor-set reuse memo ----
     *
     * Every resource this draw needs is now known, and `sig` is a hash of all
     * of it. If a recent draw resolved to the same content, its set already
     * holds exactly these descriptors: reusing it skips BOTH
     * vkAllocateDescriptorSets and vkUpdateDescriptorSets, which is the bulk
     * of the per-draw cost once the uniform packing is off the critical path.
     * Only the dynamic offsets can still differ, and those are supplied at
     * bind time rather than baked into the set.
     */
    VkDescriptorSet set = VK_NULL_HANDLE;
    for (int i = 0; i < kDescriptorMemoSize; ++i) {
        const DescriptorMemoEntry& e = pr.descMemo[slot][i];
        if (e.valid && e.set != VK_NULL_HANDLE && e.signature == sig) {
            set = e.set;
            break;
        }
    }

    if (set == VK_NULL_HANDLE) {
        // Reuse a cached set if the cursor still has one; otherwise allocate a
        // new one and append it to the per-slot cache.
        if (pr.setCursor[slot] < pr.allocatedSets[slot].size()) {
            set = pr.allocatedSets[slot][pr.setCursor[slot]++];
        } else {
            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool = pr.descriptorPools[slot];
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &pr.descriptorSetLayout;
            if (vkAllocateDescriptorSets(b->device, &dsai, &set) != VK_SUCCESS) {
                // Pool exhausted mid-frame (>1024 distinct sets this frame for
                // this program).
                //
                // FIX (descriptor pool UAF - GPU page fault 根因): 旧实现在这里
                // 调 vkResetDescriptorPool —— 但当前 command buffer 仍在录制，
                // 更早的 draw 已经通过 vkCmdBindDescriptorSets 引用了本池的 set。
                // reset 释放这些 set 后，GPU 执行该 command buffer 时读取已释放的
                // descriptor 内存 → GPU Address Fault (kIOGPUCommandBufferCallback
                // ErrorPageFault)。VUID-vkResetDescriptorPool-descriptorPool-00313
                // 明确禁止在 set 仍被录制中的 command buffer 引用时 reset。
                //
                // 修复（对照 MobileGL UniformManager 的增长策略）：不 reset 旧池，
                // 而是创建一个新的次级池继续分配，旧池（连同其所有 set）延迟到
                // slot fence 完成后才销毁。当前帧已绑定的 set 保持有效。
                static int poolExhaustedLogCount = 0;
                poolExhaustedLogCount++;
                if (poolExhaustedLogCount <= 3 || poolExhaustedLogCount % 100 == 0) {
                    MITHRIL_LOG_WARN("vk", "vkAllocateDescriptorSets failed (program %u, slot %d, "
                                      "gen %llu) — descriptor pool exhausted; growing a secondary "
                                      "pool (log %d)",
                                      program, slot, (unsigned long long)b->frameGeneration,
                                      poolExhaustedLogCount);
                }
                VkDescriptorPool oldPool = pr.descriptorPools[slot];
                VkDescriptorPool newPool = create_program_pool(pr, kMaxSetsPerPool);
                if (newPool == VK_NULL_HANDLE) {
                    return;  // 次级池创建失败，放弃本次 bind（下次 draw 重试）
                }
                // 旧池可能仍被当前 command buffer 引用 —— 延迟到 fence 后销毁。
                if (oldPool != VK_NULL_HANDLE) {
                    DeferredDestroy dd;
                    dd.pool = oldPool;
                    b->disposalQueue[b->currentFrame].push_back(dd);
                }
                pr.descriptorPools[slot] = newPool;
                // 新池没有任何已分配 set；旧池的缓存集合全部作废（它们属于旧池）。
                pr.allocatedSets[slot].clear();
                pr.setCursor[slot] = 0;
                // 旧池的 set 已随旧池延迟销毁，memo（和 bind shadow）里指向它们的
                // 记录必须作废，否则后续 draw 会复用已失效的 set。
                for (int i = 0; i < kDescriptorMemoSize; ++i) {
                    pr.descMemo[slot][i] = DescriptorMemoEntry{};
                }
                pr.descMemoNext[slot] = 0;
                on_command_buffer_boundary();
                // 从新池重试分配
                dsai.descriptorPool = newPool;
                if (vkAllocateDescriptorSets(b->device, &dsai, &set) != VK_SUCCESS) {
                    return;  // 重试仍失败，放弃本次 bind
                }
            }
            pr.allocatedSets[slot].push_back(set);
            pr.setCursor[slot]++;
        }

        for (auto& w : writes) w.dstSet = set;

        // ---- Descriptor completeness guard (root cause AK-2: PageFault) ----
        // VS/FS binding 分离后（inject_opaque_bindings），FS 有独立的
        // sampler binding（如 65）。如果该 binding 的 descriptor 因
        // view==NULL && default_texture 初始化失败而跳过写入，descriptor set
        // 里该 binding 保持未初始化 → MoltenVK 用随机地址 → GPU PageFault。
        // 补齐：检查 pr.bindings 里每个 COMBINED_IMAGE_SAMPLER binding 是否
        // 都有 write，没有的用 default texture 补齐。UBO 不补齐（UBO 路径
        // 在 arena upload 失败时 return 跳过整个 bind，不会产生半写入的 set）。
        //
        // FIX (GPU page fault root cause — second layer): if a sampler binding
        // still lacks a write AND the default texture is ALSO unavailable (its
        // eager init failed, or it was never created), the set would be committed
        // with an UNWRITTEN descriptor. MoltenVK samples that uninitialized slot
        // as a garbage GPU address → kIOGPUCommandBufferCallbackErrorPageFault at
        // the next vkQueueSubmit. We must NEVER bind an incomplete set: abort the
        // whole bind (return before vkUpdateDescriptorSets / vkCmdBindDescriptorSets)
        // so the set is not submitted. The set remains cached for a future retry;
        // once the default texture exists, a later draw re-enters and writes it.
        {
            bool incomplete = false;
            DefaultTexture& dt = default_texture();
            for (const auto& db : pr.bindings) {
                if (db.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
                bool found = false;
                for (const auto& w : writes) {
                    if (w.dstBinding == db.binding) { found = true; break; }
                }
                if (found) continue;
                // 该 sampler binding 没有被写入，用 default texture 补齐
                if (dt.view != VK_NULL_HANDLE && dt.sampler != VK_NULL_HANDLE) {
                    imgInfos.push_back({});
                    VkDescriptorImageInfo& ii = imgInfos.back();
                    ii.sampler = dt.sampler;
                    ii.imageView = dt.view;
                    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet w{};
                    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.dstSet = set;
                    w.dstBinding = db.binding;
                    w.dstArrayElement = 0;
                    w.descriptorCount = 1;
                    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w.pImageInfo = &ii;
                    writes.push_back(w);
                    // 补齐的 descriptor 也要参与 signature，否则缓存复用
                    // 会跳过补齐写入
                    sig = fnv1a_u64(((uint64_t)db.binding << 32) |
                                   (uint64_t)VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sig);
                    sig = fnv1a_u64(handle_bits(dt.sampler), sig);
                    sig = fnv1a_u64(handle_bits(dt.view), sig);
                    sig = fnv1a_u64((uint64_t)VkImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), sig);
                } else {
                    // 既没有真实纹理 view，也没有 default fallback —— 无法为
                    // 该 binding 提供有效 descriptor。绝不能提交不完整 set。
                    incomplete = true;
                    static int incompleteBindCount = 0;
                    incompleteBindCount++;
                    if (incompleteBindCount <= 5 || incompleteBindCount % 200 == 0) {
                        MITHRIL_LOG_WARN("vk", "descriptor set incomplete: binding %u "
                                          "(sampler) has no valid view AND default "
                                          "texture unavailable (program %u) — aborting "
                                          "bind, NOT submitting incomplete set (log %d)",
                                          db.binding, program, incompleteBindCount);
                    }
                    break;
                }
            }
            if (incomplete) return;  // do NOT commit/bind a half-written set
        }

        if (!writes.empty()) {
            vkUpdateDescriptorSets(b->device, static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }
        DescriptorMemoEntry& e = pr.descMemo[slot][pr.descMemoNext[slot]];
        e.signature = sig;
        e.set = set;
        e.valid = true;
        pr.descMemoNext[slot] = (pr.descMemoNext[slot] + 1) % kDescriptorMemoSize;
    }

    /* ---- Bind deduplication ----
     *
     * Consecutive draws that share a program, textures and uniform values
     * (Sodium's terrain batches are exactly this) would otherwise re-issue an
     * identical vkCmdBindDescriptorSets, which MoltenVK turns into real Metal
     * argument-buffer rebinding work. The shadow is command-buffer scoped and
     * cleared by on_command_buffer_boundary().
     *
     * Reaching this point means we hold a COMPLETE, valid `set` (all early
     * bails — layout missing, uniform-arena upload failure, pool-growth retry
     * failure, incomplete-set guard — are above). Whether we then bind it here
     * or a prior identical draw already bound it (sameAsBound below), the
     * current command buffer now has a valid descriptor set bound. Publish that
     * so backend_draw_* (via descriptors_bound()) refuses to record a vkCmdDraw
     * with an unbound descriptor set (the pure-red + GPU page-fault hazard).
     */
    mithril::vk::set_descriptors_bound(true);
    BindShadow& sh = g_bindShadow[bind_point_index(bindPoint)];
    const bool sameAsBound =
        sh.valid && sh.cb == b->commandBuffer &&
        sh.layout == pr.pipelineLayout && sh.set == set &&
        sh.dynOffsets.size() == dynOffsets.size() &&
        (dynOffsets.empty() ||
         std::memcmp(sh.dynOffsets.data(), dynOffsets.data(),
                     dynOffsets.size() * sizeof(uint32_t)) == 0);
    if (sameAsBound) return;

    vkCmdBindDescriptorSets(b->commandBuffer, bindPoint,
                            pr.pipelineLayout, 0, 1, &set,
                            static_cast<uint32_t>(dynOffsets.size()),
                            dynOffsets.empty() ? nullptr : dynOffsets.data());

    sh.valid = true;
    sh.cb = b->commandBuffer;
    sh.layout = pr.pipelineLayout;
    sh.set = set;
    sh.dynOffsets.assign(dynOffsets.begin(), dynOffsets.end());
}

void invalidate_descriptor_memo() {
    auto& tbl = program_table();
    for (auto& kv : tbl) {
        ProgramResources& pr = kv.second;
        for (int i = 0; i < kMaxFramesInFlight; ++i) {
            // Clear the memo entries so a cached set referencing a just-destroyed
            // resource is never returned by bind_program_descriptors' memo hit.
            //
            // FIX (GPU page fault / stale-set reuse): previously we deliberately
            // did NOT touch setCursor/allocatedSets, relying on the setCursor
            // reuse path (bind_program_descriptors line ~1166) to REWRITE a
            // reused set with fresh descriptors. That is only sound if the
            // rewrite is guaranteed to run. It is NOT: the rewrite can be
            // skipped when a uniform-arena upload fails (early return) or when
            // an incomplete-set guard bails out (DescriptorSet.cpp ~1299). In
            // those cases a reused set retains a descriptor pointing at the
            // just-deferred-destroyed VkImageView. The view is still alive now
            // (it sits in disposalQueue until the slot fence signals), but the
            // set is then submitted and, once the disposal drain frees the
            // view, the next submit that reuses/re-records that descriptor reads
            // freed memory -> kIOGPUCommandBufferCallbackErrorPageFault at a
            // scene transition where textures are re-specified.
            //
            // Fix (mirrors reset_all_descriptor_pools): drop the whole per-slot
            // set cache so the next bind allocates a BRAND-NEW set that is fully
            // written from the current, still-valid resources. A set that may
            // reference a destroyed view is never reused or re-bound.
            for (int j = 0; j < kDescriptorMemoSize; ++j) {
                pr.descMemo[i][j] = DescriptorMemoEntry{};
            }
            pr.descMemoNext[i] = 0;
            pr.allocatedSets[i].clear();
            pr.setCursor[i] = 0;
        }
    }
    // The memo-cleared program may be bound again before the next draw; make
    // sure the bind-dedup shadow is dropped so the re-issued bind re-runs the
    // full signature + memo lookup (which now misses and rewrites the set).
    on_command_buffer_boundary();
}

void reset_all_descriptor_pools() {
    Backend* b = backend();
    if (!b->device) return;
    auto& tbl = program_table();
    int resetCount = 0;
    for (auto& kv : tbl) {
        ProgramResources& pr = kv.second;
        for (int i = 0; i < kMaxFramesInFlight; ++i) {
            if (pr.descriptorPools[i] != VK_NULL_HANDLE) {
                vkResetDescriptorPool(b->device, pr.descriptorPools[i], 0);
                resetCount++;
            }
            pr.allocatedSets[i].clear();
            pr.setCursor[i] = 0;
            pr.lastFrameGen[i] = 0;
            pr.lastFlushGen[i] = 0;
            for (int j = 0; j < kDescriptorMemoSize; ++j) {
                pr.descMemo[i][j] = DescriptorMemoEntry{};
            }
            pr.descMemoNext[i] = 0;
        }
    }
    // Invalidate the bind shadow so the next draw re-binds descriptor sets
    // instead of trusting stale cached pointers to now-freed sets.
    on_command_buffer_boundary();
    if (resetCount > 0) {
        MITHRIL_LOG_INFO("vk", "reset_all_descriptor_pools: reset %d pools across "
                          "%zu programs (device recovery)", resetCount, tbl.size());
    }
}

// FIX (GPU page fault root cause — re-entrant mid-frame purge): see the
// declaration in DescriptorSet.h. reset_all_descriptor_pools() resets the
// CURRENT slot's pool, which frees any descriptor set already bound into the
// still-recording command buffer (VUID-vkResetDescriptorPool-descriptorPool
// -00313), and clear_all_pipeline_caches() destroys the pipeline that buffer
// references. Running either while a buffer is actively recording with live
// binds makes the next vkQueueSubmit read freed descriptor/pipeline memory ->
// kIOGPUCommandBufferCallbackErrorPageFault. So a purge requested from an OOM
// / critical-pressure GC path is deferred to a safe boundary when binds are
// live. The buffer's own contents are otherwise fence-gated, so at a genuine
// boundary (no pass open AND no descriptor set bound) the purge is safe.
void request_purge() {
    Backend* b = backend();
    if (!b->device) return;
    const bool liveBinds =
        b->commandBufferRecording && (render_pass_active() || descriptors_bound());
    if (liveBinds) {
        // Mid-draw: cannot purge now without faulting the GPU. Defer to the next
        // safe command-buffer boundary (process_pending_purge, called from
        // ensure_command_buffer_recording after the buffer is flushed/re-begun).
        b->pendingPurge = true;
        return;
    }
    clear_all_pipeline_caches();
    reset_all_descriptor_pools();
    b->pendingPurge = false;
}

void process_pending_purge() {
    Backend* b = backend();
    if (!b->device || !b->pendingPurge) return;
    if (b->commandBufferRecording && (render_pass_active() || descriptors_bound())) {
        return;  // still not safe; stay deferred
    }
    clear_all_pipeline_caches();
    reset_all_descriptor_pools();
    b->pendingPurge = false;
}

} // namespace vk
} // namespace mithril

// ===========================================================================
// Public C API wrappers (declared in MG_Backend/Backend.h)
// ===========================================================================
extern "C" {

void backend_ensure_program_layouts(GLuint program,
                                    const uint32_t* vs, int vs_words,
                                    const uint32_t* fs, int fs_words) {
    // The C ABI in MG_Backend/Backend.h is graphics-only (shared across groups,
    // not changed here); the compute stage is reflected from
    // get_or_create_compute_pipeline(), which calls the C++ entry point directly.
    mithril::vk::ensure_program_layouts(program, vs, vs_words, fs, fs_words,
                                        nullptr, 0);
}

void backend_bind_program_descriptors(GLuint program) {
    mithril::vk::bind_program_descriptors(program, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

} // extern "C"
