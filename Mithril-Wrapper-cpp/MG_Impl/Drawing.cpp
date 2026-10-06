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

namespace {
struct DrawRec { unsigned prog,fbo,t0,t1,t2,t3; unsigned char dt,dwm,bl,cull,sc; unsigned stride; };
constexpr int kDrawRingN = 1024;
DrawRec g_drawRing[kDrawRingN];
int g_drawRingIdx = 0;
}
extern "C" void mithril_dump_draw_ring(const char* path) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    int start = (g_drawRingIdx < kDrawRingN) ? 0 : g_drawRingIdx;
    for (int k=0;k<kDrawRingN;++k) {
        int i = (start + k) % kDrawRingN;
        const DrawRec& r = g_drawRing[i];
        if (r.prog==0 && r.fbo==0) continue;
        std::fprintf(f, "prog=%u fbo=%u tex[%u,%u,%u,%u] dt=%u dwm=%u bl=%u cull=%u sc=%u stride=%u\n", r.prog,r.fbo,r.t0,r.t1,r.t2,r.t3,(unsigned)r.dt,(unsigned)r.dwm,(unsigned)r.bl,(unsigned)r.cull,(unsigned)r.sc,r.stride);
    }
    std::fclose(f);
}

#include "Framebuffer.h"
#include "../MG_Backend/DirectVulkan/Device.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <algorithm>
#include <cstdio>
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

// ---- B1 first-frame diagnostic ----
// 真机首帧红屏/黑屏排障：在最初的 N 个已呈现帧内，把 prepare_draw 的每个
// 静默早退路径（缺 program / 空 SPIR-V / 无颜色附件 / 管线创建失败）都打一
// 条日志，便于一次性拿到「首几帧到底哪一环失败」。帧计数由 eglSwapBuffers
// 的 present 路径自增（g_state->presentedFrames），这里只读。
static bool first_frame_diag() {
    return g_state && g_state->presentedFrames <= 60;
}

// 把 VkFormat 数组压成一行可读字符串（避免逐条日志刷屏）。
static void diag_log_formats(const VkFormat fmts[8], int count, VkFormat depth) {
    char buf[256];
    int off = 0;
    off += snprintf(buf + off, sizeof(buf) - (size_t)off, "[");
    for (int i = 0; i < count && off < (int)sizeof(buf) - 32; ++i) {
        off += snprintf(buf + off, sizeof(buf) - (size_t)off, "%s%d",
                        i ? "," : "", (int)fmts[i]);
    }
    snprintf(buf + off, sizeof(buf) - (size_t)off, "] depth=%d", (int)depth);
    MITHRIL_LOG_ERROR("vk-diag", "%s", buf);
}

static bool prepare_draw(GLenum mode) {
    bool force_ccw_gui=false;

    bool pd_dump = std::getenv("MITHRIL_DUMP_BLIT") != nullptr;
    static uint64_t pd_att=0,pd_f_prog=0,pd_f_spirv=0,pd_f_att=0,pd_f_pipe=0;
    ++pd_att;
    #define PD_FAIL(which, why...) do { ++which; if (pd_dump && ((pd_att%120)==0)) { MITHRIL_LOG_WARN("vk-diag","prepareDrawFail att=%llu noprogram=%llu nospirv=%llu noattach=%llu nopipe=%llu drawFBO=%u " why, (unsigned long long)pd_att,(unsigned long long)pd_f_prog,(unsigned long long)pd_f_spirv,(unsigned long long)pd_f_att,(unsigned long long)pd_f_pipe, g_state->currentDrawFBO); } if (std::getenv("MITHRIL_PDFAIL")) { fprintf(stderr,"[PF] att=%llu fbo=%u prog=%u ",(unsigned long long)pd_att,g_state->currentDrawFBO,g_state->currentProgram); fprintf(stderr, why); fprintf(stderr,"\n"); } do { static bool once_ = false; if (!once_) { once_ = true; fprintf(stderr,"[PDFAIL] att=%llu fbo=%u prog=%u ",(unsigned long long)pd_att,g_state->currentDrawFBO,g_state->currentProgram); fprintf(stderr, why); fprintf(stderr,"\n"); } } while(0); return false; } while(0)
    // Resolve current program + its SPIR-V.
    mithril::Program* prog = mithril::state_get_program(g_state->currentProgram);
    if (!prog || !prog->linked) {
        if (first_frame_diag()) {
            MITHRIL_LOG_ERROR("vk-diag", "B1 draw skipped: program %u %s "
                              "(currentProgram=%u)", g_state->currentProgram,
                              (!prog) ? "not found" : "not linked",
                              g_state->currentProgram);
        }
        PD_FAIL(pd_f_prog, "reason=no-program");
    }
    // Front-face winding discriminator (root-cause fix for missing GUI/HUD).
    // Geometry authored in screen/Y-down coordinates (Minecraft GUI ortho,
    // ProjMat m[5] < 0) reaches the Metal framebuffer with the OPPOSITE winding
    // from Y-up world geometry: world front faces are CW in the (Y-down)
    // framebuffer; GUI front faces are CCW. Reading the current ProjMat loose
    // uniform m[5] (column-major, second row / second column) distinguishes
    // them without matching program ids. Programs that source ProjMat from an
    // explicit UBO block have no loose "ProjMat" uniform and keep the default
    // CW mapping (see invert_front_face below).
    {
        auto pu = prog->uniforms.find("ProjMat");
        if (pu != prog->uniforms.end() && pu->second.value.size() >= 16 &&
            pu->second.value[5] < 0.0f) {
            force_ccw_gui = true;
        }
    }

    // Determine whether we are drawing to the default framebuffer (FBO 0) or a
    // user-created FBO. This selects the Y-flipped vs non-flipped vertex SPIR-V
    // variant: the default framebuffer renders to the on-screen drawable
    // (Vulkan/Metal Y-down), so it needs the Y-flipped variant; user FBOs
    // render into textures sampled by GL shaders (GL Y-up), so they use the
    // non-flipped variant. Deep reference: MobileGL GetShaderTransformFlags.
    bool is_default_fbo = (g_state->currentDrawFBO == 0);
    // Y-flip selection. The frame must be flipped exactly once overall: if
    // MoltenVK is already flipping vertex Y (backend_yflip_enabled()==0, e.g.
    // on iOS where the host owned a VkInstance before our config was set),
    // adding our own flip would render the whole frame vertically mirrored.
    const bool want_yflip = is_default_fbo && (backend_yflip_enabled() != 0);
    const std::vector<uint32_t>& vs_spirv = want_yflip
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
    if (want_yflip && vs_spirv.empty() && !prog->vertexSpirv.empty()) {
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
        PD_FAIL(pd_f_spirv, "reason=no-spirv");
    }

    // Resolve current draw FBO attachments (color + depth VkImageViews + size).
    VkImageView colors[8] = {VK_NULL_HANDLE};
    VkImageView depth_view = VK_NULL_HANDLE;
    int w = 0, h = 0;
    int color_count = mithril::collect_draw_fbo_attachments(colors, &depth_view, &w, &h);
    // Depth-only FBOs are valid (shadow maps / depth prepasses). Reject only
    // a target with neither color nor depth attachments.
    if (color_count <= 0) {
        bool any_color = false;
        for (int i = 0; i < 8; ++i) if (colors[i] != VK_NULL_HANDLE) { any_color = true; break; }
        if (!any_color && depth_view == VK_NULL_HANDLE) {
            if (first_frame_diag()) {
                MITHRIL_LOG_ERROR("vk-diag", "B1 draw skipped: no framebuffer attachment "
                                  "(currentDrawFBO=%u size=%dx%d)", g_state->currentDrawFBO, w, h);
            }
            PD_FAIL(pd_f_att, "reason=no-attachment size=%dx%d", w, h);
        }
    }

    // Compute color attachment VkFormats.
    VkFormat color_formats[8] = {VK_FORMAT_UNDEFINED};
    mithril::Framebuffer* fbo = mithril::state_get_framebuffer(g_state->currentDrawFBO);
    // FIX (black-frame root cause): FBO 0 must ALWAYS take the swapchain-format
    // path. state_get_framebuffer(0) returns a non-null EMPTY Framebuffer entry;
    // branching on `fbo != null` routed the default framebuffer into the
    // user-FBO branch, where colors[i].texture == 0 left color_formats[] as
    // VK_FORMAT_UNDEFINED. The pipeline was then created with color attachment
    // format VK_FORMAT_UNDEFINED -> Metal MTLPixelFormatInvalid, which does not
    // match the framebuffer's MTLPixelFormatBGRA8Unorm: Metal validation aborts
    // (setRenderPipelineState pixelFormat assertion) and, without validation,
    // silently rasterizes nothing -> uniformly black frame.
    if (!is_default_fbo && fbo) {
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
    if (std::getenv("MITHRIL_FMT_TRACE")) {
        MITHRIL_LOG_WARN("vk-diag","SPVSZ prog=%u vsWords=%zu fsWords=%zu",
          prog->id,(size_t)(vs_spirv_ptr?vs_spirv_ptr->size():0),prog->fragmentSpirv.size());
        for (int i = 0; i < color_count; ++i) {
            GLuint tt = fbo ? fbo->colors[i].texture : 0;
            VkFormat allocFmt = tt ? backend_get_texture_format(tt) : VK_FORMAT_UNDEFINED;
            MITHRIL_LOG_WARN("vk-diag",
              "FMT fbo=%u slot=%d tex=%u glInternal=0x%x pipeFmt=%d allocFmt=%d match=%d",
              g_state->currentDrawFBO, i, tt,
              tt && mithril::state_get_texture(tt) ? (int)mithril::state_get_texture(tt)->internalFormat : 0,
              (int)color_formats[i], (int)allocFmt,
              (int)(color_formats[i]==allocFmt));
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
        // GL 4.3 separate-attribute-format API (glVertexAttribFormat +
        // glVertexAttribBinding + glBindVertexBuffer, used by modern MC's
        // GpuBuffer present path) stores stride/buffer/divisor on the named
        // binding, not on the attribute. Resolve them there; the legacy
        // glVertexAttribPointer path never populates bindings[], so fall back
        // to the attribute copies. Reading only the attribute copies under the
        // modern API bound a stale 16-byte buffer with a wrong stride, which
        // failed Metal's "argument length > buffer length" validation and made
        // MoltenVK drop every scene draw (uniform black).
        const GLuint bi = a.bindingIndex;
        const mithril::VertexBinding& vbn = vao->bindings[bi];
        MGVertexAttrib& m = attribs[attrib_count++];
        m.location     = i;
        m.size         = a.size;
        m.type         = a.type;
        m.normalized   = a.normalized ? 1 : 0;
        m.integer      = a.integer ? 1 : 0;
        m.stride       = vbn.stride > 0 ? vbn.stride : a.stride;
        m.offset       = (int)(intptr_t)a.pointer;
        m.enabled      = 1;
        m.buffer_name  = vbn.buffer ? vbn.buffer : a.boundBuffer;
        m.divisor      = vbn.divisor ? vbn.divisor : a.divisor;
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
    if (pipeline == VK_NULL_HANDLE) {
        if (first_frame_diag()) {
            MITHRIL_LOG_ERROR("vk-diag", "B1 draw skipped: pipeline creation FAILED "
                              "prog=%u is_default=%d colors=%d depth_format=%d",
                              prog->id, (int)is_default_fbo, color_count, (int)depth_format);
            diag_log_formats(color_formats, color_count, depth_format);
        }
        PD_FAIL(pd_f_pipe, "reason=no-pipeline colors=%d depth=%d", color_count, (int)depth_format);
    }

    // If a pass is already open from a previous (kept-open) draw but it
    // targets a different framebuffer, end it so we begin a fresh pass here.
    //
    // FIX (black screen / per-frame flicker, CRITICAL): this MUST happen
    // BEFORE the attachment registration below. end_render_pass() reads
    // e.fboColorTexIds[]/e.fboDepthTexId to barrier those images back to
    // read-only layouts and then CLEARS the registration — so registering
    // first made it act on the NEW target's textures (leaving them in a
    // read-only layout as they are about to become color/depth attachments)
    // and left begin_render_pass() with fboColorTexCount == 0, which is the
    // backend's unambiguous "this is the default framebuffer == swapchain"
    // marker (CommandStream.cpp swapchainBound / isSwapView). The
    // misclassified pass then declared the SWAPCHAIN's colour format
    // (B8G8R8A8_UNORM) instead of the texture's own, assumed a
    // D32_SFLOAT_S8_UINT depth instead of the attachment's real format, and
    // skipped both the barrier to COLOR_ATTACHMENT_OPTIMAL and the barrier
    // back to SHADER_READ_ONLY at pass end — so the final glBlitFramebuffer
    // composite sampled the scene texture from the wrong layout. Whether this
    // fires depends on whether a pass happened to still be open at the FBO
    // switch, which varies per frame: that is the difference between a steady
    // black screen and a rapidly flickering one.
    static GLuint s_passFBO = 0;
    GLuint wantFBO = (GLuint)g_state->currentDrawFBO;
    // True when the pass left open by the previous draw still targets this
    // draw's framebuffer, so begin_render_pass() will coalesce into it.
    const bool reuse_open_pass =
        backend_render_pass_active() && s_passFBO == wantFBO;
    if (backend_render_pass_active() && !reuse_open_pass) {
        backend_end_render_pass();
    }

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
    //
    // Only publish it when begin_render_pass() will actually run: while a pass
    // stays open across consecutive draws it coalesces them (begin_render_pass
    // early-returns on passActive), and re-registering every draw cleared
    // e.currentRenderPass — the handle Pipeline.cpp compiles a classic-path
    // pipeline against — so every draw after the first in a pass saw NULL and
    // silently fell back to a canonical pass. Skipping the redundant call keeps
    // the live registration (which end_render_pass still needs) intact.
    if (!reuse_open_pass) {
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
    }

    if (std::getenv("MITHRIL_DRAW_TRACE")) {
        GLuint ct = fbo ? fbo->colors[0].texture : 0;
        void* im = ct ? (void*)backend_get_texture_image(ct) : (void*)g_state->eglDefaultColorImage;
        MITHRIL_LOG_WARN("vk-diag","drawTrace fbo=%u tex=%u img=%p %dx%d mode=%d prog=%u",
          g_state->currentDrawFBO,ct,im,w,h,(int)mode,g_state->currentProgram);
    }

    // Begin render pass (Load action preserves previous contents). The
    // pass that was still open for the previous target was already ended
    // above, BEFORE this target's attachment ids were registered.
    backend_set_load_load();
    backend_begin_render_pass(colors, color_count, depth_view, w, h, 1);
    s_passFBO = wantFBO;

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
    {
      DrawRec rec{ prog->id,(unsigned)g_state->currentDrawFBO,
        g_state->boundTextureForUnit(0),g_state->boundTextureForUnit(1),
        g_state->boundTextureForUnit(2),g_state->boundTextureForUnit(3),
        (unsigned char)(g_state->depthTest?1:0),(unsigned char)(g_state->depthMask?1:0),
        (unsigned char)(g_state->blends[0].enabled?1:0),(unsigned char)(g_state->cullFace?1:0),
        (unsigned char)(g_state->scissorTest?1:0), 0u };
      g_drawRing[g_drawRingIdx % kDrawRingN] = rec;
      ++g_drawRingIdx;
    }
    if (getenv("MITHRIL_TOPO")) {
      static int tn=0; if(tn<400){++tn;
        MITHRIL_LOG_WARN("vk-diag","TOPO #%d fbo=%d mode=0x%x (STRIP=5,FAN=6,TRIS=4,QUADS=7)",tn,g_state->currentDrawFBO,(unsigned)mode);}
    }
    if (getenv("MITHRIL_MAT_DUMP") && g_state->currentDrawFBO==3){
      static int mn=0; ++mn; if(mn<=3||mn%400==0){
        for(int pt=0;pt<=1;++pt){
          mithril::IndexedBindingSlot& sl=g_state->indexedBufferBindings[0][pt];
          float m[16]={0};
          if(sl.name) backend_read_buffer_host(sl.name,(VkDeviceSize)sl.offset,64,m);
          MITHRIL_LOG_WARN("vk-diag","PT fbo3 #%d prog=%u strd=%d point=%d vbo=%u off=%ld range=%ld c0=(%.4f,%.4f,%.4f,%.4f) c1=(%.4f,%.4f,%.4f,%.4f) c3=(%.4f,%.4f,%.4f,%.4f)",
            mn,prog->id,(attrib_count?attribs[0].stride:-1),pt,sl.name,(long)sl.offset,(long)sl.size,
            m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[12],m[13],m[14],m[15]);
        }
      }
    }
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
    // TEMP DIAG (MITHRIL_FIXED_QUAD): draw a self-contained fullscreen yellow
    // triangle into the active user-FBO pass, then restore the app pipeline.
    // Tells whether the pass rasterizes draws independent of app vertex data.
    if (std::getenv("MITHRIL_FIXED_QUAD") && g_state->currentDrawFBO != 0) {
        backend_probe_fixed_quad();
        backend_bind_pipeline(pipeline);
    }
    if (std::getenv("MITHRIL_UBO_PROBE") && g_state->currentDrawFBO != 0) {
        backend_probe_ubo_fixed(prog->id);
        backend_bind_pipeline(pipeline);
    }
    // TEMP DIAG (MITHRIL_VP_DUMP): dump dynamic raster state for the first
    // draws into FBO 3 to find why fragments produce no tex6 pixels.
    if (std::getenv("MITHRIL_VP_DUMP") && g_state->currentDrawFBO != 0) {
        static int vp_n=0;
        if (vp_n < 40) {
            ++vp_n;
            MITHRIL_LOG_WARN("vk-diag",
              "VP fbo=%u #%d viewport=(%d,%d,%d,%d) scissorTest=%d scissor=(%d,%d,%d,%d) "
              "colorMask=(%d%d%d%d) blend=%d(%u,%u) depthTest=%d depthMask=%d depthFunc=%d cull=%d baseV=%d baseI=%u mode=%u",
              g_state->currentDrawFBO,
              vp_n, g_state->viewportX,g_state->viewportY,g_state->viewportW,g_state->viewportH,
              (int)g_state->scissorTest, g_state->scissorX,g_state->scissorY,g_state->scissorW,g_state->scissorH,
              (int)g_state->colorMask[0][0],(int)g_state->colorMask[0][1],
              (int)g_state->colorMask[0][2],(int)g_state->colorMask[0][3],
              (int)g_state->blends[0].enabled,
              (unsigned)g_state->blends[0].srcRGB,(unsigned)g_state->blends[0].dstRGB,
              (int)g_state->depthTest,
              (int)g_state->depthMask, (int)g_state->depthFunc, (int)g_state->cullFace,
              (int)g_state->currentBaseVertex, (unsigned)g_state->currentBaseInstance, (unsigned)mode);
        }
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
    if (std::getenv("MITHRIL_NO_CULL")) {
        backend_set_cull_mode(0);  // diagnostic
    } else
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
        // A positive-height Vulkan viewport reverses GL window-space winding
        // on the iOS MoltenVK path, including user FBOs. The physical-device
        // Minecraft 26.2 run exposed this on offscreen passes.
        // A positive-height Vulkan viewport has y DOWN; GL window space has
        // y UP. This reverses triangle winding on EVERY target (not just the
        // default FBO), so GL's CCW front faces appear CW in Vulkan and must be
        // classified with frontFace=CW whether or not the position Y was
        // flipped. The previous is_default_fbo gate left user-FBO front faces
        // culled -> uniform black fbo3 even with valid vertices/matrices.
        bool invert_front_face = true;
        (void)is_default_fbo;
        backend_set_front_face(
            force_ccw_gui ? 1 /*CCW*/ :
            invert_front_face ?
                (g_state->frontFace == GL_CCW ? 0 /*CW*/ : 1 /*CCW*/) :
                (g_state->frontFace == GL_CCW ? 1 /*CCW*/ : 0 /*CW*/));
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
    // Resolved vertex-binding vector, bound in ONE vkCmdBindVertexBuffers call
    // below instead of one call per attribute (and skipped entirely when this
    // command buffer already holds the identical vector — see
    // CommandStream.cpp). prepare_draw() re-resolves every attribute before
    // every draw, so this is the largest single cut in per-draw command
    // traffic available in the draw path.
    VkBuffer vb_bufs[16];
    VkDeviceSize vb_offs[16];
    for (int i = 0; i < attrib_count; ++i) {
        MGVertexAttrib& m = attribs[i];
        VkBuffer buf = backend_get_buffer(m.buffer_name);
        if (buf == VK_NULL_HANDLE && m.buffer_name != 0) {
            // An enabled attribute whose buffer has no backend VkBuffer used to
            // fall through to the zero buffer below with no diagnostic at all:
            // the shader then reads vec4(0) for every vertex of that attribute,
            // every draw collapses to a degenerate point, and the frame
            // presents at full rate with nothing on it. That is
            // indistinguishable from a viewport / shader / layout fault and is
            // the most expensive class of bug to chase. Say so once.
            static bool warnedNoBuf = false;
            if (!warnedNoBuf) {
                warnedNoBuf = true;
                MITHRIL_LOG_WARN("gl", "prepare_draw: vertex attrib %d (buffer %u, "
                                 "vao %u) has no backend VkBuffer - it reads "
                                 "zeros, geometry will collapse",
                                 m.location, m.buffer_name,
                                 (unsigned)(g_state ? g_state->currentVAO : 0u));
            }
        }
        if (buf != VK_NULL_HANDLE) {
            // pOffsets = the binding base offset (glBindVertexBuffer offset);
            // 0 for the legacy path (bindings untouched). The member/relative
            // offset is handled separately by ad.offset (Root Cause H).
            const GLuint mbi = vao->attribs[m.location].bindingIndex;
            VkDeviceSize binding_off = vao->bindings[mbi].offset;
// FIX (Root Cause H - 顶点属性偏移双重应用):
// Vulkan 顶点寻址公式: buffer + pOffsets[binding] + vertexIndex*stride + attr.offset
// m.offset 是属性在顶点结构内的成员偏移，必须只由 VkVertexInputAttributeDescription::offset
// 处理（见 Pipeline.cpp:302 ad.offset = a.offset）。若同时作为 binding offset 传入，
// 偏移会被应用两次 → 有效地址 = buffer + 2*m.offset，导致交错顶点格式（如
// position@0/color@12/uv@24）的属性读取错位 → 加载界面红屏/花屏。
// 参考 MobileGL VkglVertexAttribBindingState：binding offset 恒为 0，偏移由属性描述处理。
            if (m.location >= 0 && m.location < 16) {
                vb_bufs[m.location] = buf;
                vb_offs[m.location] = binding_off;
            }
            if (getenv("MITHRIL_VA_DUMP")) fprintf(stderr,"[LOW] slot=%d vkbuf=%p off=%llu\n",m.location,(void*)buf,(unsigned long long)binding_off);
            if (getenv("MITHRIL_GEO_DUMP") && m.location==0 && m.stride==16) {
                static int gn2=0; if(gn2<2){++gn2; unsigned char raw[120]={0};
                  backend_read_buffer_host(m.buffer_name, binding_off, 120, raw);
                  for(int vv=0;vv<4;++vv){ float x,y; memcpy(&x,raw+vv*16,4); memcpy(&y,raw+vv*16+4,4);
                    MITHRIL_LOG_WARN("vk-diag","GEO #%d v%d pos=(%.3f,%.3f)",gn2,vv,x,y);} }
            }
            if (getenv("MITHRIL_TEX_DUMP") && m.location==0 && g_state->currentDrawFBO==3 && m.stride==24) {
                static int tn=0;
                if (tn<6){ ++tn; unsigned char raw[200]={0};
                  backend_read_buffer_host(m.buffer_name, binding_off, 200, raw);
                  for(int vv=0; vv<2; ++vv){
                    float px,py,pz,ux,uy;
                    memcpy(&px,raw+vv*24,4); memcpy(&py,raw+vv*24+4,4); memcpy(&pz,raw+vv*24+8,4);
                    memcpy(&ux,raw+vv*24+12,4); memcpy(&uy,raw+vv*24+16,4);
                    MITHRIL_LOG_WARN("vk-diag","T24 #%d v%d pos=(%.2f,%.2f,%.2f) uv=(%.4f,%.4f)",
                       tn,vv,px,py,pz,ux,uy); }
                }
            }
            if (getenv("MITHRIL_TXT_DUMP") && m.location==0 && m.stride==24) {
                static int xn=0;
                if (xn<24){ ++xn;
                  FILE* tf=fopen("/tmp/mithril_txt_captures.txt","a");
                  if(tf){ fprintf(tf,"%u %llu %d %d %u\n",m.buffer_name,
                     (unsigned long long)binding_off,m.stride,
                     g_state->currentDrawFBO,prog->id); fclose(tf); }
                }
            }
            if (getenv("MITHRIL_POS_DUMP") && m.location==0 && g_state->currentDrawFBO==3) {
                static int pn=0;
                if (pn<3){ ++pn; unsigned char raw[160]={0};
                  backend_read_buffer_host(m.buffer_name, 0, 160, raw);
                  int any=0; for(int z=0;z<160;z++) if(raw[z]) any=1;
                  char hex[480]={0}; int hx=0; for(int z=0;z<96;z++) hx+=snprintf(hex+hx,480-hx,"%02x",raw[z]);
                  MITHRIL_LOG_WARN("vk-diag","BUF33 #%d anyNZ=%d first96=%s",pn,any,hex);
                }
            }
            if (m.location < 16) bound_slots[m.location] = true;
        }
    }
    // Any declared slot that resolved to no buffer reads the zero buffer, so an
    // unbound vertex input yields vec4(0) instead of dereferencing undefined
    // memory. Only slots below the highest declared location are filled: the
    // pipeline declares bindings exclusively for enabled attributes, so slots
    // above that are never fetched and binding them is wasted work.
    int vb_hi = 0;
    for (int i = 0; i < attrib_count; ++i) {
        int loc = attribs[i].location;
        if (loc >= 0 && loc < 16 && loc + 1 > vb_hi) vb_hi = loc + 1;
    }
    if (vb_hi > 0) {
        bool complete = true;
        for (int loc = 0; loc < vb_hi; ++loc) {
            if (!bound_slots[loc]) {
                // No zero buffer available (backend not initialised yet): a
                // null VkBuffer in the array is invalid, so bind only the
                // leading run of resolved slots instead of the whole vector.
                if (zero_buf == VK_NULL_HANDLE) { complete = false; break; }
                vb_bufs[loc] = zero_buf; vb_offs[loc] = 0;
            }
        }
        if (!complete) {
            int run = 0;
            while (run < vb_hi && bound_slots[run]) ++run;
            vb_hi = run;
        }
        if (vb_hi > 0) backend_bind_vertex_buffers(0, vb_hi, vb_bufs, vb_offs);
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
    // Keep the dynamic-render pass OPEN across consecutive draws so they share
    // ONE Metal render command encoder. MoltenVK does not guarantee the
    // store->load color dependency between separate encoders packed into one
    // Metal command buffer (verified: terrain accumulates only across command-
    // buffer boundaries or within one encoder). The pass is ended lazily on
    // FBO-target change (prepare_draw), FBO bind, state changes, texture
    // upload, readback, and at present. Set MITHRIL_NO_KEEPPASS to restore the
    // old end-pass-after-every-draw behavior.
    if (std::getenv("MITHRIL_NO_KEEPPASS")) backend_end_render_pass();
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
            // FIX (silent draw drop): this path returned false without emitting
            // anything, so a host issuing a primitive mode that core profile
            // removed (GL_QUADS=0x7, GL_QUAD_STRIP=0x8, GL_POLYGON=0x9) got
            // draws=0 on every frame and a black screen with a completely
            // clean log — indistinguishable from a rendering bug. Report the
            // offending mode so the drop is attributable. Rate limited: first
            // occurrences, each new mode, then 1-in-500.
            {
                static uint64_t badModeCount = 0;
                static GLenum   lastBadMode  = 0;
                ++badModeCount;
                if (lastBadMode != mode || badModeCount <= 3 || (badModeCount % 500) == 0) {
                    lastBadMode = mode;
                    MITHRIL_LOG_WARN("gl", "validate_draw_call: unsupported primitive "
                                     "mode=0x%x — draw dropped (occurrence #%llu)",
                                     (unsigned)mode, (unsigned long long)badModeCount);
                }
            }
            return false;
    }
    if (count < 0) {
        mithril::state_set_error(GL_INVALID_VALUE);
        {
            static uint64_t badCount = 0;
            ++badCount;
            if (badCount <= 3 || (badCount % 500) == 0) {
                MITHRIL_LOG_WARN("gl", "validate_draw_call: negative count=%d "
                                 "— draw dropped (occurrence #%llu)",
                                 (int)count, (unsigned long long)badCount);
            }
        }
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
    if (std::getenv("MITHRIL_DENT")) fprintf(stderr,"[DENT] fbo=%u prog=%u mode=%u count=%d type=%u\n",g_state->currentDrawFBO,g_state->currentProgram,mode,(int)count,type);
    if (!validate_draw_call(mode, count)) { if (std::getenv("MITHRIL_DENT")) fprintf(stderr,"[DENT] validate-REJECT count=%d\n",(int)count); return; }
    if (!prepare_draw(mode)) return;  // root cause AI — see glDrawArrays
    // If a VBO is bound for GL_ELEMENT_ARRAY_BUFFER, indices is an offset into it.
    mithril::VertexArray* vao = mithril::state_get_vao(g_state->currentVAO);
    GLuint ib_name = vao ? vao->elementArrayBuffer : 0;
    VkBuffer ib = backend_get_buffer(ib_name);
    if (ib != VK_NULL_HANDLE) {
        backend_draw_indexed((int)mode, (int)count, index_type_to_int(type),
                             ib, (VkDeviceSize)(intptr_t)indices);
    } else if (ib_name != 0) {
        // An element-array buffer IS bound, so `indices` is a byte OFFSET into
        // it — never a client-side pointer. Two things used to go wrong here:
        //   * offset 0 (by far the most common: "start of the EBO") made the old
        //     `else if (indices)` guard false, so the draw was dropped with no
        //     log line anywhere;
        //   * any other offset fell into the staging branch, which reinterpreted
        //     the offset as an address and uploaded arbitrary memory as indices.
        // Never take the client-pointer path while an EBO is bound; report it.
        static bool warned = false;
        if (!warned) {
            warned = true;
            MITHRIL_LOG_WARN("gl", "glDrawElements: EBO %u is bound but has no "
                             "backend VkBuffer (offset=%p count=%d type=0x%x "
                             "vao=%u) — draw dropped",
                             ib_name, indices, (int)count, (unsigned)type,
                             g_state->currentVAO);
        }
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
    if (getenv("MITHRIL_IDX_DUMP") && g_state->currentDrawFBO==3 && ib != VK_NULL_HANDLE){
      static int in_=0; if(in_<4){++in_;
        size_t el=(type==GL_UNSIGNED_INT)?4:(type==GL_UNSIGNED_BYTE)?1:2;
        long off=(long)(intptr_t)indices;
        unsigned char ibb[128]={0};
        backend_read_buffer_host(ib_name,(VkDeviceSize)off,128,ibb);
        if(el==2){ unsigned short* q=(unsigned short*)ibb;
          MITHRIL_LOG_WARN("vk-diag","IDX fbo3 #%d ib=%u off=%ld type=us count=%d first=%u,%u,%u,%u,%u,%u",
            in_,ib_name,off,count,q[0],q[1],q[2],q[3],q[4],q[5]); }
        else { unsigned int* q=(unsigned int*)ibb;
          MITHRIL_LOG_WARN("vk-diag","IDX fbo3 #%d ib=%u off=%ld type=ui count=%d first=%u,%u,%u,%u,%u,%u",
            in_,ib_name,off,count,q[0],q[1],q[2],q[3],q[4],q[5]); }
      }
    }
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

/* ---- Sync objects: real GPU completion semantics ------------------------- */
// The DirectVulkan backend already tracks a monotonically increasing serial for
// every vkQueueSubmit and associates it with the frame-slot fence.  GL syncs
// must use that mechanism: reporting a fence as signaled at creation lets
// persistent-mapped upload rings overwrite bytes the GPU is still consuming.
//
// Eager-flush semantics (validated on-device with MC 26.2): glFenceSync flushes
// the GL command stream FIRST, then stamps the fence with the exact submit
// serial containing all preceding commands. A lazy "next serial" scheme can
// mis-signal when an out-of-band one-shot submit (texture upload) advances the
// watermark before the frame's own commands reach the queue.
GLsync glFenceSync(GLenum condition, GLbitfield flags) {
    MITHRIL_ENSURE_INIT();
    if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE) {
        mithril::state_set_error(GL_INVALID_ENUM);
        return nullptr;
    }
    if (flags != 0) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return nullptr;
    }

    // Eagerly flush commands preceding the fence. Drivers are allowed to flush
    // earlier than required; doing it here gives the software fence an exact
    // Vulkan submission serial without inventing a second synchronization path.
    backend_end_render_pass();
    backend_commit();

    mithril::Sync sync;
    sync.handle = g_state->nextSyncHandle;
    sync.condition = condition;
    sync.flags = flags;
    sync.submitSerial = backend_current_submit_serial();
    sync.signaled = sync.submitSerial <= backend_last_completed_serial();
    if (sync.signaled) sync.submitSerial = 0;
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
    auto it = g_state->syncObjects.find(handle);
    if (it == g_state->syncObjects.end()) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return;
    }
    g_state->syncObjects.erase(it);
}

GLenum glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
    MITHRIL_ENSURE_INIT();
    if (!sync) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return GL_WAIT_FAILED;
    }
    if (flags & ~GL_SYNC_FLUSH_COMMANDS_BIT) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return GL_WAIT_FAILED;
    }
    void* handle = reinterpret_cast<void*>(sync);
    auto it = g_state->syncObjects.find(handle);
    if (it == g_state->syncObjects.end()) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return GL_WAIT_FAILED;
    }
    mithril::Sync& s = it->second;
    static uint64_t cw_att=0,cw_already=0,cw_cond=0,cw_timeout=0; ++cw_att;

    if (s.signaled || s.submitSerial == 0 ||
        s.submitSerial <= backend_last_completed_serial()) {
        s.signaled = true;
        s.submitSerial = 0;
        ++cw_already;
        if (std::getenv("MITHRIL_DUMP_BLIT")&&(cw_att%60==0)) MITHRIL_LOG_WARN("vk-diag","clientWait att=%llu already=%llu cond=%llu timeout=%llu",(unsigned long long)cw_att,(unsigned long long)cw_already,(unsigned long long)cw_cond,(unsigned long long)cw_timeout);
        return GL_ALREADY_SIGNALED;
    }

    // glFenceSync currently flushes eagerly, but honour the API flag as well so
    // this remains correct if fence creation becomes lazy in the future.
    if (flags & GL_SYNC_FLUSH_COMMANDS_BIT) {
        backend_end_render_pass();
        backend_commit();
    }

    if (backend_wait_serial(s.submitSerial, (uint64_t)timeout)) {
        s.signaled = true;
        s.submitSerial = 0;
        ++cw_cond;
        if (std::getenv("MITHRIL_DUMP_BLIT")&&(cw_att%60==0)) MITHRIL_LOG_WARN("vk-diag","clientWait att=%llu already=%llu cond=%llu timeout=%llu",(unsigned long long)cw_att,(unsigned long long)cw_already,(unsigned long long)cw_cond,(unsigned long long)cw_timeout);
        return GL_CONDITION_SATISFIED;
    }
    ++cw_timeout;
    if (std::getenv("MITHRIL_DUMP_BLIT")&&(cw_att%60==0)) MITHRIL_LOG_WARN("vk-diag","clientWait att=%llu already=%llu cond=%llu timeout=%llu serial=%llu completed=%llu",(unsigned long long)cw_att,(unsigned long long)cw_already,(unsigned long long)cw_cond,(unsigned long long)cw_timeout,(unsigned long long)s.submitSerial,(unsigned long long)backend_last_completed_serial());
    return GL_TIMEOUT_EXPIRED;
}

void glWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
    MITHRIL_ENSURE_INIT();
    if (!sync || flags != 0 || timeout != GL_TIMEOUT_IGNORED) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return;
    }
    void* handle = reinterpret_cast<void*>(sync);
    auto it = g_state->syncObjects.find(handle);
    if (it == g_state->syncObjects.end()) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return;
    }
    // Mithril uses one Vulkan graphics queue for all contexts. A host wait is
    // conservative but preserves GL server-wait semantics across thread/context
    // hand-offs until a native VkSemaphore-backed cross-context path exists.
    if (it->second.signaled || it->second.submitSerial == 0 ||
        backend_wait_serial(it->second.submitSerial, UINT64_MAX)) {
        it->second.signaled = true;
        it->second.submitSerial = 0;
    }
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
    if (!sync) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return;
    }
    if (bufSize < 0) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return;
    }
    if (bufSize == 0) return;
    if (!values) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return;
    }
    void* handle = reinterpret_cast<void*>(sync);
    auto it = g_state->syncObjects.find(handle);
    if (it == g_state->syncObjects.end()) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return;
    }
    mithril::Sync& s = it->second;
    if (s.submitSerial != 0 &&
        s.submitSerial <= backend_last_completed_serial()) {
        s.submitSerial = 0;
        s.signaled = true;
    }
    GLint v = 0;
    switch ((uint32_t)pname) {
        case 0x9112u: v = 0x9116; break; // GL_OBJECT_TYPE / GL_SYNC_FENCE
        case 0x9113u: v = (GLint)s.condition; break; // GL_SYNC_CONDITION
        case 0x9115u: v = (GLint)s.flags; break; // GL_SYNC_FLAGS
        case 0x9114u: v = (s.submitSerial == 0) ? 0x9119 : 0x9118; break;
        default:
            mithril::state_set_error(GL_INVALID_ENUM);
            return;
    }
    values[0] = v;
    if (length) *length = 1;
}

} // extern "C"
