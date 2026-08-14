// Mithril-Wrapper - MG_Backend/DirectVulkan/Reflect.cpp
// SPIR-V descriptor reflection (SPIRV-Cross) — pure-logic helpers extracted
// from DescriptorSet.cpp. Depends only on SPIRV-Cross + Vulkan headers, so it
// can be linked into the unit-test binary without pulling in Device/Pipeline/
// Backend (which would require a real VkInstance).
#include "Reflect.h"

// samplerTarget stores GLenum values; pull the constants from the project's own
// GL header so Reflect.h can stay free of a GL include (it is linked standalone
// by the unit-test binary).
#include <GL/glcorearb.h>

#include <spirv_cross.hpp>
// spirv_cross.hpp transitively pulls in SPIRV-Cross's bundled spirv.hpp,
// which defines the spv:: namespace (spv::DecorationBinding, etc.) used below.

#include <cstdio>

namespace mithril {
namespace vk {

std::vector<DescriptorBinding> reflect_stage(const uint32_t* spirv, int words,
                                             VkShaderStageFlags stage) {
    std::vector<DescriptorBinding> out;
    if (!spirv || words <= 0) return out;
    try {
        spirv_cross::Compiler compiler(spirv, static_cast<size_t>(words));
        spirv_cross::ShaderResources res = compiler.get_shader_resources();

        for (auto& r : res.uniform_buffers) {
            DescriptorBinding b{};
            b.set = compiler.get_decoration(r.id, spv::DecorationDescriptorSet);
            b.binding = compiler.get_decoration(r.id, spv::DecorationBinding);
            b.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            b.stageMask = stage;
            b.name = r.name;
            const spirv_cross::SPIRType& t = compiler.get_type(r.base_type_id);
            b.bufferSize = static_cast<uint32_t>(compiler.get_declared_struct_size(t));
            // Member layout (offsets + names) for the aggregated-block case.
            for (auto& rng : compiler.get_active_buffer_ranges(r.id)) {
                DescriptorBindingMember m;
                m.name = compiler.get_member_name(r.base_type_id, rng.index);
                m.offset = static_cast<uint32_t>(rng.offset);
                m.size = static_cast<uint32_t>(rng.range);
                b.members.push_back(std::move(m));
            }
            out.push_back(std::move(b));
        }
        for (auto& r : res.sampled_images) {
            DescriptorBinding b{};
            b.set = compiler.get_decoration(r.id, spv::DecorationDescriptorSet);
            b.binding = compiler.get_decoration(r.id, spv::DecorationBinding);
            b.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.stageMask = stage;
            b.name = r.name;
            const spirv_cross::SPIRType& t = compiler.get_type(r.type_id);
            b.descriptorCount = t.array.empty() ? 1u : static_cast<uint32_t>(t.array[0]);
            if (b.descriptorCount == 0) b.descriptorCount = 1;
            // FIX (Main-menu panorama cubemap GPU fault root cause - sampler
            // type): record the sampler's GL target type (2D/Cube/3D/Array)
            // derived from SPIR-V image.dim. The descriptor binder picks the
            // texture from the matching texture-unit slot by this type. The old
            // code used boundTextureForUnit which unconditionally preferred the
            // 2D slot: on the main menu unit 0 often still holds a GUI 2D
            // binding, so the panorama's samplerCube got a 2D view -> MoltenVK
            // viewType mismatch -> undefined sampling / GPU fault. zink selects
            // the slot by the declared type, which is why it renders.
            switch (t.image.dim) {
                case spv::Dim1D:    b.samplerTarget = t.image.arrayed ? GL_TEXTURE_1D_ARRAY : GL_TEXTURE_1D; break;
                case spv::Dim3D:    b.samplerTarget = GL_TEXTURE_3D; break;
                case spv::DimCube:  b.samplerTarget = t.image.arrayed ? 0x9009u /* GL_TEXTURE_CUBE_MAP_ARRAY */ : GL_TEXTURE_CUBE_MAP; break;
                case spv::Dim2D:
                default:            b.samplerTarget = t.image.arrayed ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D; break;
            }
            out.push_back(std::move(b));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mithril: SPIRV-Cross reflection failed: %s\n", e.what());
    }
    return out;
}

void merge_bindings(std::vector<DescriptorBinding>& dst,
                    const std::vector<DescriptorBinding>& src) {
    for (const auto& s : src) {
        auto it = std::find_if(dst.begin(), dst.end(), [&](const DescriptorBinding& d) {
            return d.set == s.set && d.binding == s.binding && d.type == s.type;
        });
        if (it == dst.end()) {
            dst.push_back(s);
        } else {
            it->stageMask |= s.stageMask;
        }
    }
}

} // namespace vk
} // namespace mithril
