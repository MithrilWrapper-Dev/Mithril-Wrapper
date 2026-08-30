// Mithril-Wrapper - MG_State/Caps.cpp
#include "Caps.h"

namespace mithril {

// Master list of every extension string we could plausibly advertise. Entries
// whose entry points are still STUBS are listed in kUnsupported and filtered
// out below, so we never claim a capability we cannot back.
static const char* kAllExtensions[] = {
    "GL_ARB_vertex_buffer_object", "GL_ARB_vertex_array_object",
    "GL_ARB_framebuffer_object", "GL_ARB_shader_objects",
    "GL_ARB_vertex_shader", "GL_ARB_fragment_shader",
    "GL_ARB_uniform_buffer_object", "GL_ARB_draw_elements_base_vertex",
    "GL_ARB_instanced_arrays", "GL_ARB_texture_multisample",
    "GL_ARB_texture_buffer_object", "GL_ARB_texture_cube_map_array",
    "GL_ARB_texture_rg", "GL_ARB_texture_float", "GL_ARB_depth_buffer_float",
    "GL_ARB_depth_texture", "GL_ARB_depth_clamp", "GL_ARB_seamless_cube_map",
    "GL_ARB_seamless_cubemap_per_texture", "GL_ARB_sync",
    "GL_ARB_internalformat_query", "GL_ARB_internalformat_query2",
    "GL_ARB_invalidate_subdata", "GL_ARB_robustness", "GL_ARB_map_buffer_range",
    "GL_ARB_vertex_type_2_10_10_10_rev", "GL_ARB_half_float_vertex",
    "GL_ARB_half_float_pixel", "GL_ARB_texture_compression",
    "GL_ARB_vertex_array_bgra", "GL_ARB_explicit_attrib_location",
    "GL_ARB_conservative_depth", "GL_ARB_shading_language_420pack",
    "GL_ARB_draw_indirect", "GL_ARB_gpu_shader5", "GL_ARB_sample_shading",
    "GL_ARB_texture_gather", "GL_ARB_texture_query_lod",
    "GL_ARB_draw_buffers_blend", "GL_ARB_multi_draw_indirect",
    "GL_ARB_buffer_storage", "GL_ARB_clear_texture", "GL_ARB_enhanced_layouts",
    "GL_ARB_shader_image_load_store", "GL_ARB_shader_image_size",
    "GL_ARB_shader_storage_buffer_object", "GL_ARB_stencil_texturing",
    "GL_ARB_texture_buffer_range", "GL_ARB_texture_query_levels",
    "GL_ARB_texture_compression_bptc", "GL_ARB_texture_storage",
    "GL_ARB_texture_storage_multisample", "GL_ARB_vertex_attrib_binding",
    "GL_ARB_viewport_array", "GL_ARB_clip_control",
    "GL_ARB_conditional_render_inverted", "GL_ARB_cull_distance",
    "GL_ARB_derivative_control", "GL_ARB_ES2_compatibility",
    "GL_ARB_ES3_compatibility", "GL_ARB_fragment_layer_viewport",
    "GL_ARB_framebuffer_no_attachments", "GL_ARB_get_texture_sub_image",
    "GL_ARB_pipeline_statistics_query", "GL_ARB_query_buffer_object",
    "GL_ARB_shader_atomic_counters", "GL_ARB_shader_atomic_counter_ops",
    "GL_ARB_shader_clock", "GL_ARB_shader_draw_parameters",
    "GL_ARB_shader_group_vote", "GL_ARB_shader_precision",
    "GL_ARB_shader_texture_image_samples", "GL_ARB_shader_texture_lod",
    "GL_ARB_explicit_uniform_location", "GL_ARB_program_interface_query",
    "GL_ARB_shading_language_packing", "GL_ARB_texture_mirror_clamp_to_edge",
    "GL_ARB_ES3_1_compatibility", "GL_ARB_compute_shader",
};

// Extensions whose core entry points are still STUBS in this build. Claiming
// them makes hosts (Minecraft / Sodium / Iris) take paths that reference
// resources that were never created -> undefined sampling -> red screen.
static const char* kUnsupported[] = {
    "GL_ARB_sparse_texture",
    "GL_ARB_gpu_shader_fp64",
    "GL_ARB_tessellation_shader",
    "GL_ARB_transform_feedback2",
    "GL_ARB_transform_feedback3",
    "GL_ARB_shader_image_load_formats",
    "GL_ARB_shader_subroutine",
};

static bool is_unsupported(const char* n) {
    for (const char* u : kUnsupported) if (std::string(n) == u) return true;
    return false;
}

const Caps& caps() {
    static Caps c;   // GL 3.3 / GLSL 330 — raise only when implemented.
    return c;
}

const std::string& version_string() {
    static const std::string s = [] {
        const Caps& c = caps();
        return "OpenGL " + std::to_string(c.gl_major) + "." +
               std::to_string(c.gl_minor) + ".0 Mithril-Wrapper 1.0, Vulkan (MoltenVK) Backend";
    }();
    return s;
}

const std::string& glsl_version_string() {
    static const std::string s = [] {
        const Caps& c = caps();
        return std::to_string(c.glsl_major) + "." + std::to_string(c.glsl_minor) +
               " Mithril-Wrapper (glslang -> SPIR-V)";
    }();
    return s;
}

const std::vector<const char*>& extensions() {
    static const std::vector<const char*> v = [] {
        std::vector<const char*> out;
        for (const char* e : kAllExtensions)
            if (!is_unsupported(e)) out.push_back(e);
        return out;
    }();
    return v;
}

bool has_extension(const char* name) {
    if (!name) return false;
    for (const char* e : extensions()) if (std::string(e) == name) return true;
    return false;
}

} // namespace mithril
