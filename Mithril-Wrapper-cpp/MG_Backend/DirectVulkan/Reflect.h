// Mithril-Wrapper - MG_Backend/DirectVulkan/Reflect.h
// SPIR-V descriptor reflection (via SPIRV-Cross) — pure-logic helpers extracted
// from DescriptorSet.cpp so they can be unit-tested without a VkDevice.
//
// reflect_stage() walks a SPIR-V module's uniform_buffers / sampled_images and
// returns a list of DescriptorBinding (set/binding/type/stageMask + UBO member
// layout). merge_bindings() unions VS+FS binding sets, OR-ing stageMask for
// matching (set,binding,type) triples.
//
// These functions depend only on SPIRV-Cross + Vulkan headers — no Device.h /
// Pipeline.h / Backend.h — so the test binary can link Reflect.cpp alone.
#ifndef MITHRIL_DIRECTVULKAN_REFLECT_H
#define MITHRIL_DIRECTVULKAN_REFLECT_H

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mithril {
namespace vk {

// One reflected descriptor binding member (UBO struct field).
struct DescriptorBindingMember {
    std::string name;     // struct member name (UBOs only)
    uint32_t offset = 0;  // byte offset within the UBO
    uint32_t size = 0;    // byte size of the member
    uint32_t columns = 0;       // matrix columns (1 for non-matrices)
    uint32_t rows = 0;          // matrix rows    (1 for non-matrices)
    uint32_t arraySize = 0;     // array length   (1 for non-arrays)
    uint32_t arrayStride = 0;   // bytes between array elements (std140-padded)
    uint32_t matrixStride = 0;  // bytes between matrix columns (std140-padded)
};

// One reflected descriptor binding (merged across vertex + fragment stages).
struct DescriptorBinding {
    uint32_t set = 0;
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    VkShaderStageFlags stageMask = 0;
    uint32_t bufferSize = 0;        // UBO size in bytes (0 for images)
    uint32_t descriptorCount = 1;   // array size (1 for non-array samplers)
    // FIX (Main-menu panorama cubemap GPU fault root cause): GL target type of a
    // sampled image (2D/Cube/3D/Array), derived from SPIR-V image.dim. The
    // descriptor binder uses it to pick the texture from the matching texture
    // unit slot (samplerCube -> CubeMap slot), instead of the old "always
    // prefer the 2D slot", which fed a 2D view to a samplerCube and triggered a
    // MoltenVK viewType mismatch. Stores the GLenum value (GL_TEXTURE_2D=0x0DE1
    // etc.) to keep the header free of a GL dependency.
    uint32_t samplerTarget = 0x0DE1;  // GL_TEXTURE_2D
    std::string name;               // reflected resource name (for UBO matching)
    std::vector<DescriptorBindingMember> members;  // UBO members (for packed $Global-style blocks)
};

// Reflect one stage's SPIR-V. `stage` tags every discovered binding so the
// caller can OR masks when merging VS+FS. Returns an empty vector on null
// input or if SPIRV-Cross throws (malformed SPIR-V).
std::vector<DescriptorBinding> reflect_stage(const uint32_t* spirv, int words,
                                             VkShaderStageFlags stage);

// Merge `src` bindings into `dst`. A binding with the same (set,binding,type)
// has its stageMask OR-ed in (members kept from the first occurrence; they are
// identical across stages for a well-formed program).
void merge_bindings(std::vector<DescriptorBinding>& dst,
                    const std::vector<DescriptorBinding>& src);

} // namespace vk
} // namespace mithril

#endif // MITHRIL_DIRECTVULKAN_REFLECT_H
