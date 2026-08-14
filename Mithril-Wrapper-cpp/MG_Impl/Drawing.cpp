// Mithril-Wrapper - MG_Impl/Drawing.cpp
// Core drawing path: glDrawArrays / glDrawElements / instanced variants ->
// Vulkan dynamic-rendering + pipeline orchestration.
//
// Pipeline: resolve VAO + program + FBO attachments -> get-or-create
// VkGraphicsPipeline (backend_get_or_create_pipeline, SPIR-V + vertex format +
// attachment VkFormats + blend state as cache key) -> begin dynamic render
// pass (Load action) -> bind pipeline + set viewport/scissor/cull/depth/mask
// via vkCmdSet* -> bind vertex buffers + textures/samplers + uniform buffers
// -> issue draw -> end pass.
//
// This is the Vulkan/MoltenVK rewrite of the former gl/drawing.cpp. The Metal
// encoder calls (metal_encoder_*) are replaced with the Vulkan backend C API
// (backend_*) declared in MG_Backend/Backend.h. Render passes use Vulkan 1.2
// dynamic rendering (VK_KHR_dynamic_rendering) instead of Metal render
// encoders.
// Sync object + transform-feedback constants — standard GL values missing
// from our minimal glcorearb.h. Guarded so a future header update won't
// conflict. Defined before includes so State.h (which uses them as default
// field values) sees them.
#ifndef GL_SYNC_GPU_COMMANDS_COMPLETE
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#endif
#ifndef GL_SYNC_FENCE
#define GL_SYNC_FENCE                0x9116
#endif
#ifndef GL_SYNC_CONDITION
#define GL_SYNC_CONDITION            0x9118
#endif
#ifndef GL_SYNC_FLAGS
#define GL_SYNC_FLAGS                0x9115
#endif
#ifndef GL_SYNC_STATUS
#define GL_SYNC_STATUS               0x9119
#endif
#ifndef GL_SIGNALED
#define GL_SIGNALED                  0x911E
#endif
#ifndef GL_UNSIGNALED
#define GL_UNSIGNALED                0x911F
#endif
#ifndef GL_OBJECT_TYPE
#define GL_OBJECT_TYPE               0x9112
#endif
#ifndef GL_INTERLEAVED_ATTRIBS
#define GL_INTERLEAVED_ATTRIBS       0x8C8C
#endif

#include "includes.h"
#include "Framebuffer.h"

#include <algorithm>
#include <cstring>

extern "C" {

/*
 * Prepare everything a draw needs: pipeline, render pass, descriptors and
 * dynamic state.
 *
 * RETURNS true only when the render pass is open AND a graphics pipeline is
 * bound — i.e. only when it is legal to record a vkCmdDraw* afterwards.
 *
 * ---- Root cause AI (CRITICAL, SIGSEGV inside MVKRenderSubpass) ----
 * This used to return void, so every early-out below silently produced a
 * "bare draw": the caller went straight on to backend_draw_*(), which
 * recorded a vkCmdDraw into a command buffer that had NO active render pass
 * and NO bound pipeline. Recording a draw outside a render-pass instance is
 * undefined behaviour per the Vulkan spec, and MoltenVK reacts by
 * dereferencing its null MVKRenderPass:
 *
 *   MVKCommandEncoder::beginMetalRenderPass()
 *     -> getSubpass()                              // _renderPass == nullptr
 *     -> MVKRenderSubpass::populateMTLRenderPassDescriptor()
 *          MVKPixelFormats* pixFmts = _renderPass->getPixelFormats();  // BOOM
 *
 * which is exactly the crash observed on iPhone X / iOS 16.7.15:
 *   SIGSEGV at libmithril.dylib+0x280d20
 *   MVKRenderSubpass::populateMTLRenderPassDescriptor(...)+0x3c
 * (+0x3c is the function's very first member access, i.e. _renderPass.)
 *
 * The trigger on that device was a graphics-pipeline creation failure
 * ("vkCreateGraphicsPipelines transient failure (rc=-3)") caused by the
 * cross-stage descriptor binding collision fixed in Shader.cpp. That root
 * cause is gone, but ANY pipeline failure (unsupported shader, OOM, transient
 * driver error) must degrade to a dropped frame, never to a process abort.
 * Returning a status here — and re-checking it in the backend, see
 * backend_draw_* in CommandStream.cpp — makes that guarantee structural.
 */
static bool prepare_draw(GLenum mode) {
    // Resolve current program + its SPIR-V.
    mithril::Program* prog = mithril::state_get_program(g_state->currentProgram);
    if (!prog || !prog->linked) return false;

    // Determine whether we are drawing to the default framebuffer (FBO 0) or a
    // user-created FBO. This selects the Y-flipped vs non-flipped vertex SPIR-V
    // variant: the default framebuffer renders to the on-screen drawable
    // (Vulkan/Metal Y-down), so it needs the Y-flipped variant; user FBOs
    // render into textures sampled by GL shaders (GL Y-up), so they use the
    // non-flipped variant. Deep reference: MobileGL GetShaderTransformFlags.
    bool is_default_fbo = (g_state->currentDrawFBO == 0);
    const std::vector<uint32_t>& vs_spirv = is_default_fbo
        ? prog->vertexSpirvYFlipped : prog->vertexSpirv;

    // Defensive: skip draws whose shader translation produced no SPIR-V
    // (e.g. glslang failed on an unrecognised construct). Issuing the draw
    // would pass null/0 to backend_get_or_create_pipeline, which would
    // either crash on the SPIR-V pointer or fail pipeline creation silently
    // and leave the screen black. Logging once per program id keeps the log
    // readable when the host retries the same broken shader every frame.
    // Fallback: if Y-flipped variant is empty but non-flipped exists, use it
    // (wrong Y orientation but won't skip draws / leave only clear color).
    const std::vector<uint32_t>* vs_spirv_ptr = &vs_spirv;
    if (is_default_fbo && vs_spirv.empty() && !prog->vertexSpirv.empty()) {
        MITHRIL_LOG_WARN("gl", "prepare_draw: program %u Y-flipped SPIR-V empty, "
                          "falling back to non-flipped variant (%zu words)",
                          prog->id, prog->vertexSpirv.size());
        vs_spirv_ptr = &prog->vertexSpirv;
    }
    if (vs_spirv_ptr->empty() || prog->fragmentSpirv.empty()) {
        static GLuint last_warned = 0;
        if (last_warned != prog->id) {
            last_warned = prog->id;
            MITHRIL_LOG_WARN("gl", "prepare_draw: program %u has empty SPIR-V "
                              "(vertex=%zu vertexYFlip=%zu fragment=%zu words, "
                              "is_default_fbo=%d); skipping draw",
                              prog->id, prog->vertexSpirv.size(),
                              prog->vertexSpirvYFlipped.size(),
                              prog->fragmentSpirv.size(), (int)is_default_fbo);
        }
        return false;
    }

    // Resolve current draw FBO attachments (color + depth VkImageViews + size).
    VkImageView colors[8] = {VK_NULL_HANDLE};
    VkImageView depth_view = VK_NULL_HANDLE;
    int w = 0, h = 0;
    int color_count = mithril::collect_draw_fbo_attachments(colors, &depth_view, &w, &h);
    // Defensive: if no color attachment is bound at all (e.g. the EGL default
    // framebuffer has no swapchain yet because the surface isn't sized), skip
    // the draw. Beginning a render pass with all-null attachments produces a
    // validation error and a no-op pass on MoltenVK, so skipping is both
    // cheaper and avoids log spam.
    if (color_count <= 0) {
        bool any_color = false;
        for (int i = 0; i < 8; ++i) if (colors[i] != VK_NULL_HANDLE) { any_color = true; break; }
        if (!any_color) return false;
    }

    // Compute color attachment VkFormats.
    VkFormat color_formats[8] = {VK_FORMAT_UNDEFINED};
    mithril::Framebuffer* fbo = mithril::state_get_framebuffer(g_state->currentDrawFBO);
    if (fbo) {
        for (int i = 0; i < color_count; ++i) {
            GLuint t = fbo->colors[i].texture;
            mithril::Texture* tex = mithril::state_get_texture(t);
            if (tex) color_formats[i] = backend_vk_format_for_gl((GLenum)tex->internalFormat);
        }
    } else {
        // EGL default framebuffer: read the swapchain's actual color format
        // from g_state->eglDefaultColorFormat (set by install_surface_on_state
        // after each acquire). Hardcoding VK_FORMAT_B8G8R8A8_UNORM would
        // mismatch if MoltenVK picked a different surface format (e.g. RGBA8
        // or an sRGB variant), causing a pipeline-creation failure on the
        // first draw and a black screen.
        VkFormat swapchainFmt = g_state->eglDefaultColorFormat;
        if (swapchainFmt == VK_FORMAT_UNDEFINED) {
            // Fallback for headless / surfaceless mode where no swapchain is
            // attached. BGRA8Unorm matches MoltenVK's most common default.
            swapchainFmt = VK_FORMAT_B8G8R8A8_UNORM;
        }
        for (int i = 0; i < color_count; ++i) {
            if (colors[i] != VK_NULL_HANDLE) {
                color_formats[i] = swapchainFmt;
            }
        }
    }
    // Depth format from the bound depth texture.
    // For FBO 0 (EGL default framebuffer), the depth image is a raw VkImage
    // created by the EGL layer (VK_FORMAT_D32_SFLOAT_S8_UINT), not tracked in
    // the GL texture table. For user FBOs, derive the VkFormat from the GL
    // internalFormat.
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    if (fbo && fbo->depth.texture) {
        mithril::Texture* dt = mithril::state_get_texture(fbo->depth.texture);
        if (dt) depth_format = backend_vk_format_for_gl((GLenum)dt->internalFormat);
    } else if (depth_view != VK_NULL_HANDLE) {
        // EGL default framebuffer: depth is always D32_SFLOAT_S8_UINT.
        depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    }

    // Build the vertex attribute descriptor array for the pipeline signature.
    mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
    if (!vao) vao = mithril::state_get_vao(0);
    MGVertexAttrib attribs[mithril::kMaxVertexAttribs];
    int attrib_count = 0;
    for (int i = 0; i < mithril::kMaxVertexAttribs; ++i) {
        const mithril::VertexAttrib& a = vao->attribs[i];
        if (!a.enabled) continue;
        MGVertexAttrib& m = attribs[attrib_count++];
        m.location     = i;
        m.size         = a.size;
        m.type         = a.type;
        m.normalized   = a.normalized ? 1 : 0;
        m.integer      = a.integer ? 1 : 0;
        m.stride       = a.stride;
        m.offset       = (int)(intptr_t)a.pointer;
        m.enabled      = 1;
        m.buffer_name  = a.boundBuffer;
        m.divisor      = a.divisor;
    }

    // Get-or-create the VkGraphicsPipeline. Blend state + colorWriteMask are
    // part of the pipeline signature so that enabling/disabling GL_BLEND,
    // changing blend functions, or calling glColorMask creates a distinct
    // pipeline (root cause I+J: previously only blend_enabled/src/dst were in
    // the signature and colorWriteMask was hardcoded RGBA-all-on, so different
    // blend/mask configs collided in the cache and glColorMask was a no-op).
    int cwm_bits = 0;
    if (g_state->colorMask[0][0]) cwm_bits |= 1;
    if (g_state->colorMask[0][1]) cwm_bits |= 2;
    if (g_state->colorMask[0][2]) cwm_bits |= 4;
    if (g_state->colorMask[0][3]) cwm_bits |= 8;
    VkPipeline pipeline = backend_get_or_create_pipeline(
        prog->id,
        vs_spirv_ptr->data(),            (int)vs_spirv_ptr->size(),
        prog->fragmentSpirv.data(), (int)prog->fragmentSpirv.size(),
        attribs, attrib_count,
        color_formats, color_count,
        depth_format,
        g_state->blends[0].enabled ? 1 : 0,
        g_state->blends[0].srcRGB,
        g_state->blends[0].dstRGB,
        g_state->blends[0].srcA,
        g_state->blends[0].dstA,
        cwm_bits,
        mode,
        is_default_fbo ? 1 : 0);
    // Pipeline creation failed (shader compile error, OOM, transient driver
    // error). Returning false makes every caller skip its backend_draw_*
    // call — see the root cause AI comment on this function. Note that the
    // render pass has NOT been begun at this point (that happens below), so
    // a draw issued here would be recorded outside any render-pass instance.
    if (pipeline == VK_NULL_HANDLE) return false;

    // FIX (root cause Y, CRITICAL): Register user-FBO attachment tex_ids so
    // begin_render_pass can barrier their images to attachment-optimal and
    // end_render_pass can barrier them back to read-only + update
    // TextureEntry::currentLayout. VK_KHR_dynamic_rendering does NOT
    // auto-transition attachment layouts — without this registration, the
    // declared imageLayout (COLOR_ATTACHMENT_OPTIMAL) would mismatch the
    // actual layout (SHADER_READ_ONLY_OPTIMAL from a prior upload) → spec
    // violation → MoltenVK drops the draw → black screen.
    // For FBO 0 (swapchain), pass null/0 to clear any stale registration;
    // the swapchain path's barriers are handled by the activeSwapchain block.
    if (fbo) {
        GLuint color_tex_ids[8] = {0};
        for (int i = 0; i < color_count && i < 8; ++i) {
            color_tex_ids[i] = fbo->colors[i].texture;
        }
        GLuint depth_tex_id = fbo->depth.texture;
        backend_set_fbo_attachment_tex_ids(color_tex_ids, color_count, depth_tex_id);
    } else {
        backend_set_fbo_attachment_tex_ids(nullptr, 0, 0);
    }

    // Begin render pass (Load action preserves previous contents).
    backend_set_load_load();
    backend_begin_render_pass(colors, color_count, depth_view, w, h, 1);

    // Bind pipeline + set dynamic state via vkCmdSet*.
    backend_bind_pipeline(pipeline);
    // FIX (root cause: gl_VertexID baseVertex semantics): push
    // currentBaseVertex into the shader's _MithrilBaseVertex push-constant
    // block on EVERY draw. Shader.cpp defines gl_VertexID as
    // (gl_VertexIndex + _mbv._mithrilBaseVertex); writing
    // g_state->currentBaseVertex here (0 for the vast majority of Minecraft
    // draws, non-zero only under glDrawElementsBaseVertex / InstancedBaseVertex)
    // restores desktop GL's gl_VertexID == index + baseVertex semantics. Always
    // writing (not just when baseVertex != 0) guarantees the push constant is
    // never undefined when the shader reads it.
    backend_push_constants(prog->id, 0, 4, &g_state->currentBaseVertex);
    // Bind the program's descriptor set (UBOs + sampled images) immediately
    // after the pipeline so the shader's uniform/texture bindings are live for
    // the upcoming draw. The set is built per-draw from Program.uniforms +
    // g_state->boundTextures by DescriptorSet.cpp.
    backend_bind_program_descriptors(prog->id);
    backend_set_viewport(g_state->viewportX, g_state->viewportY,
                         g_state->viewportW, g_state->viewportH,
                         g_state->depthNear, g_state->depthFar);
    // FIX (root cause G): ALWAYS set the scissor. VK_DYNAMIC_STATE_SCISSOR is
    // a dynamic state (Pipeline.cpp), so it MUST be set via vkCmdSetScissor
    // before drawing. When scissorTest is disabled, the old code skipped the
    // call entirely, leaving the dynamic scissor at its undefined default
    // (0,0,0,0) — which clips ALL pixels → black screen. MobileGL always
    // sets a scissor (full viewport when GL_SCISSOR_TEST is off).
    if (g_state->scissorTest) {
        backend_set_scissor(g_state->scissorX, g_state->scissorY,
                            g_state->scissorW, g_state->scissorH);
    } else {
        backend_set_scissor(0, 0, g_state->viewportW, g_state->viewportH);
    }
    // FIX (root cause H + Y-flip winding fix): ALWAYS set cull mode.
    // VK_DYNAMIC_STATE_CULL_MODE is dynamic; skipping the call when cullFace is
    // disabled leaves the previous draw's cull mode active → stale culling
    // culls geometry incorrectly. When cullFace is off, explicitly set
    // VK_CULL_MODE_NONE.
    //
    // Y-flip winding adjustment (deep reference: MobileGL
    // ConvertCullFaceModeToVkEnum + VulkanRenderer frontFace=CLOCKWISE):
    // When the vertex Y is flipped (default framebuffer), triangle winding
    // inverts (CCW→CW, CW→CCW). To keep the GL-intended faces visible:
    //   - Swap the cull mode: GL_FRONT→VK_BACK, GL_BACK→VK_FRONT
    //   - Hardcode frontFace to CLOCKWISE (the inverted winding makes GL's
    //     CCW triangles appear as CW in Vulkan). MobileGL does the same.
    // User FBOs (no Y flip) keep the original cull mode and frontFace.
    if (g_state->cullFace) {
        // FIX (Root Cause K - Y翻转面剔除双重补偿):
        // Vulkan 面剔除由两个独立状态控制：frontFace（定义正面缠绕方向）+ cullMode（剔除哪面）。
        // Y 翻转（gl_Position.y = -y）反转缠绕：GL-CCW → Vulkan-CW。
        // 正确补偿（二选一，不可同时）：
        //   方案A: frontFace=CW（GL-CCW→Vulkan-CW="正面"），不交换 cull mode（GL_BACK→VK_BACK 剔除 GL-背面）
        //   方案B: frontFace=CCW（GL-CCW→Vulkan-CW="背面"），交换 cull mode（GL_BACK→VK_FRONT 剔除 GL-背面）
        // 旧代码同时执行 A+B → 双重补偿：frontFace=CW 使 GL-正面=Vulkan-正面，再交换 cull=VK_FRONT
        // 剔除 Vulkan-正面=GL-正面 → 所有正面几何被剔除，只剩 clear color（红色）→ 红屏。
        // 修复：采用方案A，仅 frontFace=CW 补偿，cull mode 直接按 GL 值映射不交换。
        // 参考 MobileGL VulkanRenderer ConvertCullFaceModeToVkEnum：不交换 cull mode，
        // 仅通过 frontFace=CLOCKWISE 补偿 Y 翻转。
        int vk_cull = 0;
        if (g_state->cullMode == GL_FRONT) {
            vk_cull = 1;  // VK_CULL_MODE_FRONT_BIT
        } else if (g_state->cullMode == GL_BACK) {
            vk_cull = 2;  // VK_CULL_MODE_BACK_BIT
        } else {  // GL_FRONT_AND_BACK
            vk_cull = 3;  // VK_CULL_MODE_FRONT_AND_BACK
        }
        backend_set_cull_mode(vk_cull);
        // Y 翻转使缠绕反转：GL-CCW → Vulkan-CW。设 frontFace=CW 补偿（仅默认帧缓冲）。
        // 用户 FBO 无 Y 翻转，frontFace 按 GL 值映射（CCW→1, CW→0）。
        backend_set_front_face(is_default_fbo ? 0 /*CW*/ :
                               (g_state->frontFace == GL_CCW ? 1 : 0));
    } else {
        backend_set_cull_mode(0);  // VK_CULL_MODE_NONE
    }
    backend_set_color_write_mask(
        g_state->colorMask[0][0], g_state->colorMask[0][1],
        g_state->colorMask[0][2], g_state->colorMask[0][3]);
    backend_set_depth_test(
        g_state->depthTest ? 1 : 0,
        g_state->depthMask ? 1 : 0,
        (int)g_state->depthFunc);
    // Apply dynamic pipeline state: depth bias + stencil.
    // 对照 MobileGL 动态状态应用.
    if (g_state->polygonOffsetFill) {
        backend_set_depth_bias(g_state->polygonOffsetFactor, g_state->polygonOffsetUnits);
    }
    if (g_state->stencilTest) {
        backend_set_stencil_state(1, (int)g_state->stencilFunc, g_state->stencilRef,
                                  (int)g_state->stencilValueMask,
                                  (int)g_state->stencilSfail, (int)g_state->stencilDpfail,
                                  (int)g_state->stencilDppass);
    }
    if (g_state->blends[0].enabled) {
        backend_set_blend_color(
            g_state->blendColor[0], g_state->blendColor[1],
            g_state->blendColor[2], g_state->blendColor[3]);
    }

    // Bind vertex buffers — one VkBuffer per enabled attribute, at index
    // == attribute location (matches the vertex input binding layout). For
    // attribute slots the VAO didn't enable, bind the shared zero buffer so
    // the unbound vertex input reads vec4(0) instead of dereferencing
    // unbound memory.
    VkBuffer zero_buf = backend_get_zero_buffer();
    bool bound_slots[16] = {false};
    for (int i = 0; i < attrib_count; ++i) {
        MGVertexAttrib& m = attribs[i];
        VkBuffer buf = backend_get_buffer(m.buffer_name);
        if (buf != VK_NULL_HANDLE) {
// FIX (Root Cause H - 顶点属性偏移双重应用):
// Vulkan 顶点寻址公式: buffer + pOffsets[binding] + vertexIndex*stride + attr.offset
// m.offset 是属性在顶点结构内的成员偏移，必须只由 VkVertexInputAttributeDescription::offset
// 处理（见 Pipeline.cpp:302 ad.offset = a.offset）。若同时作为 binding offset 传入，
// 偏移会被应用两次 → 有效地址 = buffer + 2*m.offset，导致交错顶点格式（如
// position@0/color@12/uv@24）的属性读取错位 → 加载界面红屏/花屏。
// 参考 MobileGL VkglVertexAttribBindingState：binding offset 恒为 0，偏移由属性描述处理。
            backend_set_vertex_buffer(m.location, buf, 0);
            if (m.location < 16) bound_slots[m.location] = true;
        }
    }
    // Bind the zero buffer to any slot 0..15 not covered above.
    if (zero_buf != VK_NULL_HANDLE) {
        for (int loc = 0; loc < 16; ++loc) {
            if (!bound_slots[loc]) {
                backend_set_vertex_buffer(loc, zero_buf, 0);
            }
        }
    }

    // Uniform buffers and sampled-image bindings are now sourced + bound via
    // the descriptor set in backend_bind_program_descriptors() (called above,
    // right after backend_bind_pipeline). It reflects the program's SPIR-V,
    // maps each UBO to Program.uniforms[name].value and each sampler binding B
    // to g_state->boundTextures[B], and writes + binds a fresh VkDescriptorSet
    // for this draw. The legacy backend_set_fragment_buffer /
    // backend_set_fragment_texture stubs are no-ops (kept only for the C API
    // contract) — descriptor binding is centralised in DescriptorSet.cpp.

    // Render pass is open and the pipeline + descriptors + dynamic state are
    // bound: it is now legal for the caller to record a vkCmdDraw*.
    return true;
}

static void end_draw(void) {
    // End the render pass but DON'T commit the command buffer here.
    // The command buffer is committed once per frame in eglSwapBuffers,
    // which presents the swapchain image. Committing per-draw would flush
    // the Vulkan pipeline hundreds of times per frame, causing severe perf
    // loss and present timing issues.
    backend_end_render_pass();
}

static int index_type_to_int(GLenum type) {
    // FIX (root cause AE - GL_UNSIGNED_BYTE 索引支持):
    // 0 = UINT16 (GL_UNSIGNED_SHORT), 1 = UINT32 (GL_UNSIGNED_INT),
    // 2 = UINT8 (GL_UNSIGNED_BYTE)。原代码仅返回 0/1，GL_UNSIGNED_BYTE
    // 被当作 UINT16 → 1 字节索引按 2 字节解释 → 索引值错乱 → 几何腐败 → 红屏。
    // backend_draw_indexed 的 case 2 映射到 VK_INDEX_TYPE_UINT8
    // （需 Device.cpp 启用 VK_EXT_index_type_uint8）。
    // 深度对照 MobileGL VulkanRenderer.cpp:3093-3109。
    if (type == GL_UNSIGNED_INT)   return 1;
    if (type == GL_UNSIGNED_BYTE)  return 2;
    return 0;  // GL_UNSIGNED_SHORT → UINT16
}

// P1-9: Validate primitive mode + vertex count for draw calls.
// Returns true if the draw may proceed; otherwise records a GL error and
// returns false. Mode must be one of the GL 3.3 Core primitive modes; count
// must be non-negative.
static bool validate_draw_call(GLenum mode, GLsizei count) {
    switch (mode) {
        case GL_POINTS:
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            break;
        default:
            mithril::state_set_error(GL_INVALID_ENUM);
            return false;
    }
    if (count < 0) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return false;
    }
    return true;
}

void glDrawArrays(GLenum mode, GLint first, GLsizei count) {
    MITHRIL_ENSURE_INIT();
    if (!validate_draw_call(mode, count)) return;
    // Root cause AI: a false return means no render pass was begun and no
    // pipeline was bound — issuing the draw anyway would record a vkCmdDraw
    // outside a render-pass instance and crash inside MoltenVK. Bail out
    // without calling end_draw(): there is no pass to end.
    if (!prepare_draw(mode)) return;
    backend_draw_arrays((int)mode, (int)first, (int)count);
    end_draw();
}

void glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei primcount) {
    MITHRIL_ENSURE_INIT();
    if (!prepare_draw(mode)) return;  // root cause AI — see glDrawArrays
    backend_draw_arrays_instanced((int)mode, (int)first, (int)count, (int)primcount);
    end_draw();
}

void glDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count,
                                       GLsizei primcount, GLuint baseinstance) {
    MITHRIL_ENSURE_INIT();
    // FIX (root cause AG - BaseInstance): 设置 currentBaseInstance 后调用 draw，
    // 完成后重置为 0。backend_draw_arrays_instanced 从 g_state 读取后传给
    // vkCmdDraw 的 firstInstance。深度对照 MobileGL drawParams.baseInstance。
    g_state->currentBaseInstance = baseinstance;
    // Root cause AI — see glDrawArrays. currentBaseInstance MUST be reset on
    // the early-out path too, otherwise it leaks into the next draw (which
    // expects firstInstance == 0) and misaddresses its instance data.
    if (!prepare_draw(mode)) { g_state->currentBaseInstance = 0; return; }
    backend_draw_arrays_instanced((int)mode, (int)first, (int)count, (int)primcount);
    g_state->currentBaseInstance = 0;
    end_draw();
}

void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
    MITHRIL_ENSURE_INIT();
    if (!validate_draw_call(mode, count)) return;
    if (!prepare_draw(mode)) return;  // root cause AI — see glDrawArrays
    // If a VBO is bound for GL_ELEMENT_ARRAY_BUFFER, indices is an offset into it.
    mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
    GLuint ib_name = vao ? vao->elementArrayBuffer : 0;
    VkBuffer ib = backend_get_buffer(ib_name);
    if (ib != VK_NULL_HANDLE) {
        backend_draw_indexed((int)mode, (int)count, index_type_to_int(type),
                             ib, (VkDeviceSize)(intptr_t)indices);
    } else if (indices) {
        // Client-space index pointer: stage into a transient VkBuffer.
        // FIX (root cause AE): GL_UNSIGNED_BYTE 索引按 1 字节/索引 staging，
        // 否则 staging 大小翻倍 → 越界读 + 索引错乱。
        size_t elem = (type == GL_UNSIGNED_INT) ? 4 : (type == GL_UNSIGNED_BYTE) ? 1 : 2;
        GLuint transient = (GLuint)(uintptr_t)indices; // use address as throwaway name
        VkBuffer staged = backend_get_or_create_buffer(transient | 0x80000000u,
                                                       indices, (size_t)count * elem);
        if (staged != VK_NULL_HANDLE) {
            backend_draw_indexed((int)mode, (int)count, index_type_to_int(type),
                                 staged, 0);
        }
    }
    end_draw();
}

void glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type,
                              const void* indices, GLint basevertex) {
    MITHRIL_ENSURE_INIT();
    // FIX (root cause AG - BaseVertex): 将 baseVertex 通过 g_state->currentBaseVertex
    // 传递给 backend_draw_indexed（vkCmdDrawIndexed 的 vertexOffset）。draw 完成后
    // 立即重置为 0，避免泄漏到后续无 BaseVertex 的 draw（应保持 vertexOffset=0）。
    // 深度对照 MobileGL drawParams.baseVertex。
    //
    // FIX (root cause: gl_VertexID baseVertex 语义，已实现): vertexOffset 只补偿
    // *顶点数据寻址*（buffer + (index + vertexOffset) * stride），不影响 shader 内
    // gl_VertexIndex。GL 的 gl_VertexID 在索引绘制中 == index + baseVertex（含
    // baseVertex），Vulkan 的 gl_VertexIndex == 原始 index（不含 vertexOffset）。
    //
    // 本函数及所有 BaseVertex/BaseInstance 入口都把 baseVertex 写入
    // g_state->currentBaseVertex；prepare_draw() 在每次 draw 前通过
    // backend_push_constants 把它推入注入的 _MithrilBaseVertex push-constant，
    // Shader.cpp 定义 gl_VertexID == gl_VertexIndex + _mbv._mithrilBaseVertex。
    // 因此用 gl_VertexID 做 SSBO/纹理数组查找的 vertex shader 在 baseVertex!=0
    // 时也得到正确的 GL 语义（不再偏移 baseVertex）。实现跨 Shader.cpp /
    // DescriptorSet.cpp / Pipeline.cpp / CommandStream.cpp / Drawing.cpp。
    g_state->currentBaseVertex = basevertex;
    glDrawElements(mode, count, type, indices);
    g_state->currentBaseVertex = 0;
}

void glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type,
                             const void* indices, GLsizei primcount) {
    MITHRIL_ENSURE_INIT();
    if (!prepare_draw(mode)) return;  // root cause AI — see glDrawArrays
    mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
    GLuint ib_name = vao ? vao->elementArrayBuffer : 0;
    VkBuffer ib = backend_get_buffer(ib_name);
    if (ib != VK_NULL_HANDLE) {
        backend_draw_indexed_instanced((int)mode, (int)count,
                                       index_type_to_int(type), ib,
                                       (VkDeviceSize)(intptr_t)indices, (int)primcount);
    } else if (indices) {
        // FIX (root cause AE): GL_UNSIGNED_BYTE 索引按 1 字节/索引 staging。
        size_t elem = (type == GL_UNSIGNED_INT) ? 4 : (type == GL_UNSIGNED_BYTE) ? 1 : 2;
        GLuint transient = (GLuint)(uintptr_t)indices;
        VkBuffer staged = backend_get_or_create_buffer(transient | 0x80000000u,
                                                       indices, (size_t)count * elem);
        if (staged != VK_NULL_HANDLE) {
            backend_draw_indexed_instanced((int)mode, (int)count,
                                           index_type_to_int(type), staged, 0,
                                           (int)primcount);
        }
    }
    end_draw();
}

void glDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type,
                                       const void* indices, GLsizei primcount,
                                       GLint basevertex) {
    MITHRIL_ENSURE_INIT();
    // FIX (root cause AG - BaseVertex): 设置 currentBaseVertex 后调用 draw，
    // 完成后重置为 0。深度对照 MobileGL drawParams.baseVertex。
    g_state->currentBaseVertex = basevertex;
    glDrawElementsInstanced(mode, count, type, indices, primcount);
    g_state->currentBaseVertex = 0;
}

void glDrawElementsInstancedBaseInstance(GLenum mode, GLsizei count, GLenum type,
                                         const void* indices, GLsizei primcount,
                                         GLuint baseinstance) {
    MITHRIL_ENSURE_INIT();
    // FIX (root cause AG - BaseInstance): 设置 currentBaseInstance 后调用 draw，
    // 完成后重置为 0。backend_draw_indexed_instanced 从 g_state 读取后传给
    // vkCmdDrawIndexed 的 firstInstance。深度对照 MobileGL drawParams.baseInstance。
    g_state->currentBaseInstance = baseinstance;
    glDrawElementsInstanced(mode, count, type, indices, primcount);
    g_state->currentBaseInstance = 0;
}

void glDrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count,
                         GLenum type, const void* indices) {
    MITHRIL_ENSURE_INIT();
    (void)start; (void)end;
    glDrawElements(mode, count, type, indices);
}

void glDrawRangeElementsBaseVertex(GLenum mode, GLuint start, GLuint end,
                                   GLsizei count, GLenum type,
                                   const void* indices, GLint basevertex) {
    MITHRIL_ENSURE_INIT();
    (void)start; (void)end;
    // Delegate to glDrawElementsBaseVertex (which handles vertexOffset).
    glDrawElementsBaseVertex(mode, count, type, indices, basevertex);
}

void glDrawElementsBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type,
                                          const void* indices, GLint basevertex,
                                          GLuint baseinstance) {
    MITHRIL_ENSURE_INIT();
    // FIX (root cause AG - BaseVertex + BaseInstance): 同时设置 currentBaseVertex
    // 与 currentBaseInstance，draw 完成后重置为 0。深度对照 MobileGL drawParams。
    g_state->currentBaseVertex = basevertex;
    g_state->currentBaseInstance = baseinstance;
    glDrawElements(mode, count, type, indices);
    g_state->currentBaseVertex = 0;
    g_state->currentBaseInstance = 0;
}

/* =========================================================================
 * MultiDraw — MobileGL 风格高性能模拟
 *
 * 参照 MobileGL VulkanRenderer::MultiDrawArrays / MultiDrawElements
 * (VulkanRenderer.cpp:6814 / 6844)：求 sub-draw 顶点范围并集 → 一次
 * SetupDraw（pipeline + render pass + descriptors + dynamic state + vertex
 * buffers）→ 循环 vkCmdDraw / vkCmdDrawIndexed → 一次 end_render_pass。
 *
 * 旧实现逐 sub-draw 调 glDrawArrays/glDrawElements，每次都重做 prepare_draw
 * + end_draw（pipeline 查找 + render pass 开 + 关 + descriptor 重绑）。
 * 对 Sodium 的数千 chunk draw，这把状态设置开销放大 drawcount 倍。
 *
 * 本实现把 prepare_draw/end_draw 提到循环外，drawcount 次 draw 共享同一
 * render-pass 实例与 pipeline 绑定，仅 vkCmdDraw 的 first/count 参数变化。
 * 这与 MobileGL 的"一次 SetupDraw + 循环 vkCmdDraw"完全等价。
 *
 * 不采用"打包成 indirect buffer + vkCmdDrawIndirect"路径的原因：
 *   1. MobileGL 自己也不打包（它循环 vkCmdDraw），GL spec 允许该等价。
 *   2. 打包需要把 CPU 端 first[]/count[] 拷进 GPU buffer，对纯 CPU 路径
 *      反而多一次上传 + 同步开销；真正的 GPU-side MultiDraw 由
 *      glMultiDraw*Indirect（见下）覆盖，那条路径数据本就在 GPU buffer。
 * ========================================================================= */
void glMultiDrawArrays(GLenum mode, const GLint* first, const GLsizei* count, GLsizei drawcount) {
    MITHRIL_ENSURE_INIT();
    if (!first || !count || drawcount <= 0) return;
    if (!prepare_draw(mode)) return;  // root cause AI — 一次 SetupDraw
    for (GLsizei i = 0; i < drawcount; ++i) {
        if (count[i] > 0) backend_draw_arrays((int)mode, (int)first[i], (int)count[i]);
    }
    end_draw();  // 一次 end_render_pass
}

void glMultiDrawElements(GLenum mode, const GLsizei* count, GLenum type,
                         const void* const* indices, GLsizei drawcount) {
    MITHRIL_ENSURE_INIT();
    if (!count || !indices || drawcount <= 0) return;
    // 解析索引缓冲一次（所有 sub-draw 共享同一 GL_ELEMENT_ARRAY_BUFFER，
    // 仅 offset 不同）。客户端指针路径逐 sub-draw staging。
    mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
    GLuint ib_name = vao ? vao->elementArrayBuffer : 0;
    VkBuffer ib = backend_get_buffer(ib_name);
    int idx_type = index_type_to_int(type);
    size_t elem = (type == GL_UNSIGNED_INT) ? 4 : (type == GL_UNSIGNED_BYTE) ? 1 : 2;
    if (!prepare_draw(mode)) return;  // root cause AI — 一次 SetupDraw
    if (ib != VK_NULL_HANDLE) {
        // VBO 路径：indices[i] 是 offset，零拷贝
        for (GLsizei i = 0; i < drawcount; ++i) {
            if (count[i] > 0)
                backend_draw_indexed((int)mode, (int)count[i], idx_type, ib,
                                     (VkDeviceSize)(intptr_t)indices[i]);
        }
    } else {
        // 客户端指针路径：逐 sub-draw staging 进 transient buffer
        for (GLsizei i = 0; i < drawcount; ++i) {
            if (count[i] > 0 && indices[i]) {
                GLuint transient = (GLuint)(uintptr_t)indices[i];
                VkBuffer staged = backend_get_or_create_buffer(transient | 0x80000000u,
                                                               indices[i], (size_t)count[i] * elem);
                if (staged != VK_NULL_HANDLE)
                    backend_draw_indexed((int)mode, (int)count[i], idx_type, staged, 0);
            }
        }
    }
    end_draw();
}

void glMultiDrawElementsBaseVertex(GLenum mode, const GLsizei* count, GLenum type,
                                   const void* const* indices, GLsizei drawcount,
                                   const GLint* basevertex) {
    MITHRIL_ENSURE_INIT();
    if (!count || !indices || drawcount <= 0) return;
    mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
    GLuint ib_name = vao ? vao->elementArrayBuffer : 0;
    VkBuffer ib = backend_get_buffer(ib_name);
    int idx_type = index_type_to_int(type);
    size_t elem = (type == GL_UNSIGNED_INT) ? 4 : (type == GL_UNSIGNED_BYTE) ? 1 : 2;
    if (!prepare_draw(mode)) return;
    if (ib != VK_NULL_HANDLE && basevertex) {
        for (GLsizei i = 0; i < drawcount; ++i) {
            if (count[i] > 0) {
                g_state->currentBaseVertex = basevertex[i];
                backend_draw_indexed((int)mode, (int)count[i], idx_type, ib,
                                     (VkDeviceSize)(intptr_t)indices[i]);
            }
        }
        g_state->currentBaseVertex = 0;
    } else if (basevertex) {
        for (GLsizei i = 0; i < drawcount; ++i) {
            if (count[i] > 0 && indices[i]) {
                g_state->currentBaseVertex = basevertex[i];
                GLuint transient = (GLuint)(uintptr_t)indices[i];
                VkBuffer staged = backend_get_or_create_buffer(transient | 0x80000000u,
                                                               indices[i], (size_t)count[i] * elem);
                if (staged != VK_NULL_HANDLE)
                    backend_draw_indexed((int)mode, (int)count[i], idx_type, staged, 0);
            }
        }
        g_state->currentBaseVertex = 0;
    }
    end_draw();
}

/* =========================================================================
 * Indirect draw (GL 4.0 ARB_draw_indirect + GL 4.3 ARB_multi_draw_indirect)
 *
 * 参数块在 GPU buffer（GL_DRAW_INDIRECT_BUFFER）中，bit-identical 于
 * VkDrawIndirectCommand / VkDrawIndexedIndirectCommand，直接传给
 * vkCmdDrawIndirect / vkCmdDrawIndexedIndirect，完全 GPU-side，无 CPU 回读。
 * 这是 Sodium 批量 chunk draw 的关键路径。
 *
 * 单个 glDrawArraysIndirect / glDrawElementsIndirect 复用 multi-draw 路径
 * （draw_count=1），与 MobileGL DirectVulkan::DrawElementsIndirect
 * (DirectVulkan.cpp:584) 的做法一致。
 * ========================================================================= */
void glDrawArraysIndirect(GLenum mode, const void* indirect) {
    MITHRIL_ENSURE_INIT();
    GLuint buf_name = g_state->bufferBindings[(int)mithril::BufferTarget::DrawIndirect].name;
    VkBuffer indirect_buf = backend_get_buffer(buf_name);
    if (indirect_buf == VK_NULL_HANDLE) return;
    if (!prepare_draw(mode)) return;  // root cause AI
    backend_draw_indirect((int)mode, indirect_buf,
                          (VkDeviceSize)(intptr_t)indirect, 1, 0);
    end_draw();
}

void glDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect) {
    MITHRIL_ENSURE_INIT();
    GLuint buf_name = g_state->bufferBindings[(int)mithril::BufferTarget::DrawIndirect].name;
    VkBuffer indirect_buf = backend_get_buffer(buf_name);
    if (indirect_buf == VK_NULL_HANDLE) return;
    mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
    GLuint ib_name = vao ? vao->elementArrayBuffer : 0;
    VkBuffer ib = backend_get_buffer(ib_name);
    if (ib == VK_NULL_HANDLE) return;
    if (!prepare_draw(mode)) return;
    backend_draw_indexed_indirect((int)mode, index_type_to_int(type), ib, 0,
                                  indirect_buf, (VkDeviceSize)(intptr_t)indirect,
                                  1, 0);
    end_draw();
}

void glMultiDrawArraysIndirect(GLenum mode, const void* indirect,
                               GLsizei drawcount, GLsizei stride) {
    MITHRIL_ENSURE_INIT();
    if (drawcount <= 0) return;
    GLuint buf_name = g_state->bufferBindings[(int)mithril::BufferTarget::DrawIndirect].name;
    VkBuffer indirect_buf = backend_get_buffer(buf_name);
    if (indirect_buf == VK_NULL_HANDLE) return;
    if (!prepare_draw(mode)) return;
    int s = stride ? stride : 16;  // sizeof(VkDrawIndirectCommand)
    backend_draw_indirect((int)mode, indirect_buf,
                          (VkDeviceSize)(intptr_t)indirect, drawcount, s);
    end_draw();
}

void glMultiDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect,
                                 GLsizei drawcount, GLsizei stride) {
    MITHRIL_ENSURE_INIT();
    if (drawcount <= 0) return;
    GLuint buf_name = g_state->bufferBindings[(int)mithril::BufferTarget::DrawIndirect].name;
    VkBuffer indirect_buf = backend_get_buffer(buf_name);
    if (indirect_buf == VK_NULL_HANDLE) return;
    mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
    GLuint ib_name = vao ? vao->elementArrayBuffer : 0;
    VkBuffer ib = backend_get_buffer(ib_name);
    if (ib == VK_NULL_HANDLE) return;
    if (!prepare_draw(mode)) return;
    int s = stride ? stride : 20;  // sizeof(VkDrawIndexedIndirectCommand)
    backend_draw_indexed_indirect((int)mode, index_type_to_int(type), ib, 0,
                                  indirect_buf, (VkDeviceSize)(intptr_t)indirect,
                                  drawcount, s);
    end_draw();
}

/* =========================================================================
 * GL 4.6 ARB_indirect_parameters — glMultiDraw*IndirectCount
 *
 * 与上面的 Indirect 变体唯一的差别：draw 数量（drawcount）不是由 CPU 传入，
 * 而是由 GPU 从 GL_DRAW_INDIRECT_BUFFER 的 `drawcount` 偏移处读取一个
 * uint32，并 clamp 到 maxdrawcount。Sodium 的 chunk 渲染正是用 compute
 * shader 在 GPU 端写好 indirect 命令 + 计数，再一次性提交 —— CPU 完全
 * 不知道最终 draw 数，因此绝不能 fallback 到 CPU 读回（会读到 stale 计数）。
 *
 * Vulkan 侧对应 vkCmdDrawIndirectCount / vkCmdDrawIndexedIndirectCount
 * （Vulkan 1.2 core `drawIndirectCount` 特性，MoltenVK 1.2.x 报告支持）。
 * backend_*_count 内部已检查 b->drawIndirectCountSupported；若不支持则静默
 * 跳过（保持与"旧 no-op"一致的行为），并在日志中提示 —— 比把 stale 计数
 * 交给 CPU 循环渲染更正确。
 * ========================================================================= */
void glMultiDrawArraysIndirectCount(GLenum mode, const void* indirect,
                                    GLintptr drawcount, GLint maxdrawcount,
                                    GLsizei stride) {
    MITHRIL_ENSURE_INIT();
    if (maxdrawcount <= 0) return;
    GLuint buf_name = g_state->bufferBindings[(int)mithril::BufferTarget::DrawIndirect].name;
    VkBuffer indirect_buf = backend_get_buffer(buf_name);
    if (indirect_buf == VK_NULL_HANDLE) return;
    // GL 规范：drawcount 是 GL_DRAW_INDIRECT_BUFFER 内的字节偏移，存储一个
    // uint32 的 draw 数量。Vulkan 的 count 参数正是 (buffer, offset)。
    VkBuffer count_buf = indirect_buf;
    VkDeviceSize count_off = (VkDeviceSize)drawcount;
    if (!prepare_draw(mode)) return;
    int s = stride ? stride : 16;  // sizeof(VkDrawIndirectCommand)
    backend_draw_indirect_count((int)mode, indirect_buf,
                                (VkDeviceSize)(intptr_t)indirect,
                                count_buf, count_off, maxdrawcount, s);
    end_draw();
}

void glMultiDrawElementsIndirectCount(GLenum mode, GLenum type,
                                      const void* indirect, GLintptr drawcount,
                                      GLint maxdrawcount, GLsizei stride) {
    MITHRIL_ENSURE_INIT();
    if (maxdrawcount <= 0) return;
    GLuint buf_name = g_state->bufferBindings[(int)mithril::BufferTarget::DrawIndirect].name;
    VkBuffer indirect_buf = backend_get_buffer(buf_name);
    if (indirect_buf == VK_NULL_HANDLE) return;
    mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
    GLuint ib_name = vao ? vao->elementArrayBuffer : 0;
    VkBuffer ib = backend_get_buffer(ib_name);
    if (ib == VK_NULL_HANDLE) return;
    // 同 Arrays 变体：count 在 GL_DRAW_INDIRECT_BUFFER 的 drawcount 偏移处。
    VkBuffer count_buf = indirect_buf;
    VkDeviceSize count_off = (VkDeviceSize)drawcount;
    if (!prepare_draw(mode)) return;
    int s = stride ? stride : 20;  // sizeof(VkDrawIndexedIndirectCommand)
    backend_draw_indexed_indirect_count((int)mode, index_type_to_int(type),
                                        ib, 0,
                                        indirect_buf, (VkDeviceSize)(intptr_t)indirect,
                                        count_buf, count_off, maxdrawcount, s);
    end_draw();
}

/* =========================================================================
 * Compute dispatch (GL 4.3 ARB_compute_shader)
 *
 * backend_dispatch_compute 已就绪：结束活动 render pass（Vulkan 禁止
 * render pass 内 vkCmdDispatch）+ 绑定 compute pipeline + descriptor set +
 * vkCmdDispatch。Iris 的 compute culling / shadow setup / 命令构建 shader
 * 由此调度。
 * ========================================================================= */
#ifndef GL_DISPATCH_INDIRECT_BUFFER
#define GL_DISPATCH_INDIRECT_BUFFER 0x90EE
#endif

void glDispatchCompute(GLuint groups_x, GLuint groups_y, GLuint groups_z) {
    MITHRIL_ENSURE_INIT();
    backend_dispatch_compute(groups_x, groups_y, groups_z);
}

void glDispatchComputeIndirect(GLintptr indirect) {
    MITHRIL_ENSURE_INIT();
    GLuint buf_name = g_state->bufferBindings[(int)mithril::BufferTarget::DispatchIndirect].name;
    VkBuffer indirect_buf = backend_get_buffer(buf_name);
    if (indirect_buf == VK_NULL_HANDLE) return;
    backend_dispatch_compute_indirect(indirect_buf, (VkDeviceSize)indirect);
}

/* =========================================================================
 * Memory barrier (GL 4.2 ARB_shader_image_load_store)
 *
 * backend_memory_barrier 已就绪：结束活动 render pass + 记录保守的
 * ALL_COMMANDS -> ALL_COMMANDS VkMemoryBarrier。Iris 在 compute 写完
 * image/SSBO 后必须调用，否则后续 draw 看不到 compute 的写入。
 * ========================================================================= */
#ifndef GL_ALL_BARRIER_BITS
#define GL_ALL_BARRIER_BITS 0xFFFFFFFF
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT
#define GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT  0x00000001
#define GL_ELEMENT_ARRAY_BARRIER_BIT        0x00000002
#define GL_UNIFORM_BARRIER_BIT              0x00000004
#define GL_TEXTURE_FETCH_BARRIER_BIT        0x00000008
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT  0x00000020
#define GL_COMMAND_BARRIER_BIT              0x00000040
#define GL_PIXEL_BUFFER_BARRIER_BIT         0x00000080
#define GL_TEXTURE_UPDATE_BARRIER_BIT       0x00000100
#define GL_BUFFER_UPDATE_BARRIER_BIT        0x00000200
#define GL_FRAMEBUFFER_BARRIER_BIT          0x00000400
#define GL_TRANSFORM_FEEDBACK_BARRIER_BIT   0x00000800
#define GL_ATOMIC_COUNTER_BARRIER_BIT       0x00001000
#define GL_SHADER_STORAGE_BARRIER_BIT       0x00002000
#define GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT 0x00004000
#endif

void glMemoryBarrier(GLbitfield barriers) {
    MITHRIL_ENSURE_INIT();
    backend_memory_barrier(barriers);
}

void glTextureBarrier(void) {
    MITHRIL_ENSURE_INIT();
    // GL 4.5 ARB_texture_barrier: 确保 framebuffer 读取看到之前 draw 的写入。
    // 保守实现为完整 memory barrier。
    backend_memory_barrier(GL_FRAMEBUFFER_BARRIER_BIT);
}

/* ---- Sync objects (P1-16 FIX) ---- */
// Real state tracking via g_state->syncObjects. Handles are allocated from
// g_state->nextSyncHandle (monotonic, avoids the sentinel 0x1). CPU-side
// fences are considered immediately signaled, matching the previous stub
// behaviour but with proper existence/identity checks.
GLsync glFenceSync(GLenum condition, GLbitfield flags) {
    MITHRIL_ENSURE_INIT();
    if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE) {
        mithril::state_set_error(GL_INVALID_ENUM);
        return nullptr;
    }
    mithril::Sync sync;
    sync.handle = g_state->nextSyncHandle;
    sync.condition = condition;
    sync.flags = flags;
    sync.signaled = true;  // CPU-side fence is immediately signaled
    sync.markedForDeletion = false;
    g_state->syncObjects[sync.handle] = sync;
    g_state->nextSyncHandle = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(g_state->nextSyncHandle) + 1);
    return reinterpret_cast<GLsync>(sync.handle);
}

void glDeleteSync(GLsync sync) {
    MITHRIL_ENSURE_INIT();
    if (!sync) return;
    void* handle = reinterpret_cast<void*>(sync);
    g_state->syncObjects.erase(handle);
}

GLenum glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
    MITHRIL_ENSURE_INIT();
    (void)flags; (void)timeout;
    if (!sync) return GL_WAIT_FAILED;
    void* handle = reinterpret_cast<void*>(sync);
    auto it = g_state->syncObjects.find(handle);
    if (it == g_state->syncObjects.end()) return GL_WAIT_FAILED;
    return it->second.signaled ? GL_ALREADY_SIGNALED : GL_TIMEOUT_EXPIRED;
}

void glWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
    MITHRIL_ENSURE_INIT();
    (void)sync; (void)flags; (void)timeout;
    // No-op: CPU-side fences are immediately signaled.
}

GLboolean glIsSync(GLsync sync) {
    MITHRIL_ENSURE_INIT();
    if (!sync) return GL_FALSE;
    void* handle = reinterpret_cast<void*>(sync);
    return g_state->syncObjects.find(handle) != g_state->syncObjects.end()
        ? GL_TRUE : GL_FALSE;
}

void glGetSynciv(GLsync sync, GLenum pname, GLsizei bufSize, GLsizei* length, GLint* values) {
    MITHRIL_ENSURE_INIT();
    if (length) *length = 0;
    if (bufSize < 0 || !values || bufSize == 0) return;
    if (!sync) return;
    void* handle = reinterpret_cast<void*>(sync);
    auto it = g_state->syncObjects.find(handle);
    if (it == g_state->syncObjects.end()) return;
    const mithril::Sync& s = it->second;
    GLint v = 0;
    switch (pname) {
        case GL_OBJECT_TYPE:    v = GL_SYNC_FENCE; break;
        case GL_SYNC_CONDITION: v = (GLint)s.condition; break;
        case GL_SYNC_FLAGS:     v = (GLint)s.flags; break;
        case GL_SYNC_STATUS:    v = s.signaled ? GL_SIGNALED : GL_UNSIGNALED; break;
        default: return;
    }
    values[0] = v;
    if (length) *length = 1;
}

} // extern "C"
