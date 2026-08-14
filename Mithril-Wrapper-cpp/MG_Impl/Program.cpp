// Mithril-Wrapper - MG_Impl/Program.cpp
// Shader / program object lifecycle: source, compile (GLSL->SPIR-V), link,
// use, uniform reflection + setters. Linked SPIR-V is cached on the program
// for the Vulkan pipeline cache (backend_get_or_create_pipeline) to consume.
//
// This is the Vulkan/MoltenVK rewrite of the former gl/program.cpp. The Metal
// MSL fields (vertexMSL/fragmentMSL) are replaced with SPIR-V word vectors
// (vertexSpirv/fragmentSpirv); MoltenVK cross-translates the SPIR-V to MSL
// internally at vkCreateShaderModule time.
#include "includes.h"
#include "Shader.h"
#include "../MG_Backend/DirectVulkan/Reflect.h"  // reflect_stage / merge_bindings / DescriptorBinding

#include <algorithm>
#include <vector>

extern "C" {

GLuint glCreateShader(GLenum type) {
    MITHRIL_ENSURE_INIT();
    GLuint name = 0;
    mithril::state_gen_names("shader", 1, &name);
    mithril::Shader s{};
    s.id = name;
    s.type = type;
    g_state->shaders[name] = s;
    return name;
}

void glDeleteShader(GLuint shader) {
    MITHRIL_ENSURE_INIT();
    mithril::Shader* s = mithril::state_get_shader(shader);
    if (!s) return;
    // P1-5 deferred deletion: mark now, erase only when no program references
    // it. If attachCount > 0 the shader stays alive until the last detach
    // triggers the actual erase from glDetachShader.
    s->markedForDeletion = true;
    if (s->attachCount == 0) {
        g_state->shaders.erase(shader);
        g_state->shaderNames.release(shader);
    }
}

GLuint glCreateProgram(void) {
    MITHRIL_ENSURE_INIT();
    GLuint name = 0;
    mithril::state_gen_names("program", 1, &name);
    mithril::Program p{};
    p.id = name;
    g_state->programs[name] = p;
    return name;
}

void glDeleteProgram(GLuint program) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p) return;
    // P1-5 deferred deletion: mark now. If this program is NOT the current
    // program, erase immediately. If it IS current, keep it alive until
    // glUseProgram(0) (or another program) replaces it — glUseProgram triggers
    // the erase for the previously-current program.
    p->markedForDeletion = true;
    if (g_state->currentProgram != program) {
        // Release the Vulkan shader modules + cached pipelines owned by this program.
        backend_delete_program_resources(program);
        g_state->programs.erase(program);
        g_state->programNames.release(program);
    }
}

void glShaderSource(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length) {
    MITHRIL_ENSURE_INIT();
    mithril::Shader* s = mithril::state_get_shader(shader);
    if (!s || count <= 0 || !string) return;
    s->source.clear();
    for (GLsizei i = 0; i < count; ++i) {
        if (length && length[i] >= 0) {
            s->source.append(string[i], (size_t)length[i]);
        } else {
            s->source.append(string[i] ? string[i] : "");
        }
    }
}

void glShaderBinary(GLsizei, const GLuint*, GLenum, const void*, GLsizei) {
    MITHRIL_ENSURE_INIT();
    // Pre-compiled shader binaries are not supported.
}

void glCompileShader(GLuint shader) {
    MITHRIL_ENSURE_INIT();
    mithril::Shader* s = mithril::state_get_shader(shader);
    if (!s) return;
    std::string info;
    std::vector<uint32_t> spirv;
    bool ok = mithril::shader_translate(s->type, s->source, spirv, info);
    s->infoLog = info;
    if (ok) {
        s->compiled = true;
        s->spirv = std::move(spirv);
        MITHRIL_LOG_INFO("shader", "Compiled shader %u (%s) -> %zu SPIR-V words",
                         shader,
                         s->type == GL_VERTEX_SHADER ? "vertex" :
                         s->type == GL_FRAGMENT_SHADER ? "fragment" : "other",
                         s->spirv.size());
    } else {
        // FIX (red-screen): shader compilation failed. Instead of marking the
        // shader uncompiled (which prevents the program from linking and
        // causes ALL draws to be skipped — leaving only the application's
        // glClearColor visible), substitute a minimal fallback SPIR-V that
        // renders geometry in solid gray. This ensures the render pipeline
        // is exercised and geometry is visible, rather than a stuck clear
        // color. Deep reference: MobileGL VulkanRenderer.cpp fallback shader
        // substitution (GetFallbackShader).
        std::vector<uint32_t> fallback;
        if (mithril::get_fallback_spirv(s->type, /*flip_y=*/false, fallback)) {
            s->compiled = true;
            s->spirv = std::move(fallback);
            MITHRIL_LOG_WARN("shader", "Shader %u (%s) compilation failed — "
                              "using FALLBACK shader (gray output): %s",
                              shader,
                              s->type == GL_VERTEX_SHADER ? "vertex" :
                              s->type == GL_FRAGMENT_SHADER ? "fragment" : "other",
                              info.c_str());
        } else {
            s->compiled = false;
            MITHRIL_LOG_ERROR("shader", "Failed to compile shader %u: %s",
                              shader, info.c_str());
        }
    }
}

void glReleaseShaderCompiler(void) { MITHRIL_ENSURE_INIT(); }

void glAttachShader(GLuint program, GLuint shader) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p) return;
    for (GLuint id : p->attachedShaders) if (id == shader) return;
    p->attachedShaders.push_back(shader);
    // P1-5: track attach count so glDeleteShader's deferred deletion can fire
    // only when the last program detaches the shader.
    mithril::Shader* s = mithril::state_get_shader(shader);
    if (s) ++s->attachCount;
}

void glDetachShader(GLuint program, GLuint shader) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p) return;
    auto& v = p->attachedShaders;
    v.erase(std::remove(v.begin(), v.end(), shader), v.end());
    mithril::Shader* s = mithril::state_get_shader(shader);
    if (s && s->attachCount > 0) {
        --s->attachCount;
        // P1-5 deferred deletion: if this was the last detach AND the shader
        // was previously marked for deletion, finish the deletion now.
        if (s->attachCount == 0 && s->markedForDeletion) {
            g_state->shaders.erase(shader);
            g_state->shaderNames.release(shader);
        }
    }
}

void glLinkProgram(GLuint program) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p) { mithril::state_set_error(GL_INVALID_OPERATION); return; }

    // Release any previously-built Vulkan resources (shader modules, cached
    // pipelines, descriptor layouts, failed-signature negative cache) for
    // this program BEFORE rebuilding. Without this, a relink with new shader
    // source would keep using the OLD VkShaderModule (built from the OLD
    // SPIR-V) — get_or_create_pipeline only creates the module once (when
    // pr.vertexModule == VK_NULL_HANDLE) and the pipeline signature hash does
    // NOT include SPIR-V content, so the stale pipeline would be returned
    // from the cache on every subsequent draw. This manifests as "relink
    // did nothing" or black screen if the old shaders are incompatible with
    // the new render state. MobileGL rebuilds pipelines on every link
    // (ProgramObject::Link -> GenerateBinary -> PipelineFactory); we mirror
    // that by tearing down here so the next draw rebuilds from scratch.
    backend_delete_program_resources(program);

    p->vertexSpirv.clear();
    p->vertexSpirvYFlipped.clear();
    p->fragmentSpirv.clear();
    p->uniforms.clear();
    p->uniformByLocation.clear();
    p->attribs.clear();
    p->uniformBlocks.clear();

    // If the application called glBindAttribLocation before linking, re-translate
    // the vertex shader with those location overrides so the SPIR-V stage_input
    // locations match the app's vertex descriptor. Fragment shaders are
    // unaffected by attribute bindings.
    const bool has_attrib_bindings = !p->attribBindings.empty();

    bool missing = false;
    for (GLuint sid : p->attachedShaders) {
        mithril::Shader* s = mithril::state_get_shader(sid);
        if (!s) continue;
        if (!s->compiled || s->spirv.empty()) { missing = true; continue; }
        if (s->type == GL_VERTEX_SHADER) {
            // --- Non-flipped variant (for user-created FBOs) ---
            // s->spirv (from glCompileShader) already has Z remap injected but
            // no Y flip and no attrib bindings. Re-translate with bindings if
            // needed; otherwise reuse s->spirv directly.
            if (has_attrib_bindings) {
                std::vector<uint32_t> spirv;
                std::string info;
                if (mithril::shader_translate(s->type, s->source, spirv, info, &p->attribBindings, /*flip_y=*/false)) {
                    p->vertexSpirv = std::move(spirv);
                } else {
                    MITHRIL_LOG_ERROR("program", "Re-translation with attrib bindings "
                                      "failed for program %u: %s (using auto-mapped SPIR-V)",
                                      program, info.c_str());
                    p->vertexSpirv = s->spirv;
                }
            } else {
                p->vertexSpirv = s->spirv;
            }

            // --- Y-flipped variant (for default framebuffer / FBO 0) ---
            // Always re-translate with flip_y=true so draws to the on-screen
            // drawable get the Y inversion. Deep reference: MobileGL
            // GetShaderTransformFlags sets PositionYFlip when
            // currentDrawFBO->IsDefaultFramebuffer().
            {
                std::vector<uint32_t> spirv;
                std::string info;
                const auto* bindings_ptr = has_attrib_bindings ? &p->attribBindings : nullptr;
                if (mithril::shader_translate(s->type, s->source, spirv, info, bindings_ptr, /*flip_y=*/true)) {
                    p->vertexSpirvYFlipped = std::move(spirv);
                } else {
                    // FIX (red-screen): Y-flipped translation failed. If the
                    // non-flipped variant is also a fallback shader (i.e. the
                    // original shader source failed to compile), generate a
                    // Y-flipped fallback so default-FBO draws get correct Y
                    // orientation. Only fall back to the non-flipped variant
                    // if the Y-flipped fallback also fails.
                    std::vector<uint32_t> yflip_fallback;
                    if (mithril::get_fallback_spirv(GL_VERTEX_SHADER, /*flip_y=*/true, yflip_fallback)) {
                        p->vertexSpirvYFlipped = std::move(yflip_fallback);
                        MITHRIL_LOG_WARN("program", "Y-flipped vertex translation "
                                         "failed for program %u — using Y-flipped "
                                         "FALLBACK shader: %s", program, info.c_str());
                    } else {
                        // Last resort: non-flipped variant (wrong Y orientation
                        // but won't crash).
                        MITHRIL_LOG_ERROR("program", "Y-flipped vertex translation "
                                          "failed for program %u: %s (using non-flipped fallback)",
                                          program, info.c_str());
                        p->vertexSpirvYFlipped = p->vertexSpirv;
                    }
                }
            }
        } else if (s->type == GL_FRAGMENT_SHADER) {
            p->fragmentSpirv = s->spirv;
        }
    }
    // FIX (root cause K): link fails if EITHER stage is missing/empty (||),
    // not only if BOTH are empty (&&). The old && logic let single-stage
    // programs (VS-only or FS-only) "link successfully", but Drawing.cpp's
    // prepare_draw uses || to skip draws whose VS or FS is empty — so every
    // draw of a "linked" single-stage program was silently skipped, producing
    // a black screen. GL also requires a complete program (VS+FS) to link.
    if (missing || p->vertexSpirv.empty() || p->fragmentSpirv.empty()) {
        p->linked = false;
        p->infoLog = "link failed: a required stage was missing or uncompiled";
        MITHRIL_LOG_ERROR("program", "Link failed for program %u: missing/empty stage "
                          "(VS=%zu VS_yflip=%zu FS=%zu words)", program,
                          p->vertexSpirv.size(), p->vertexSpirvYFlipped.size(),
                          p->fragmentSpirv.size());
        return;
    }

    // Ensure vertexSpirvYFlipped is non-empty for default framebuffer (FBO 0) draws.
    // If Y-flipped translation failed and fell back to non-flipped variant which
    // also failed, vertexSpirvYFlipped could be empty while vertexSpirv is not.
    // In that case, use the non-flipped SPIR-V as last resort (wrong Y orientation
    // but won't skip draws / leave only clear color visible).
    if (p->vertexSpirvYFlipped.empty() && !p->vertexSpirv.empty()) {
        p->vertexSpirvYFlipped = p->vertexSpirv;
        MITHRIL_LOG_WARN("program", "vertexSpirvYFlipped was empty for program %u, "
                          "falling back to non-flipped variant (%zu words)",
                          program, p->vertexSpirv.size());
    }

    p->linked = true;
    p->infoLog.clear();

    // ---- Uniform reflection (CRITICAL for black screen) ----
    // Reflect SPIR-V to discover UBOs and their members, then populate
    // p->uniforms / p->uniformByLocation / p->uniformBlocks / p->uboBackingStore.
    // Without this, glGetUniformLocation returns -1 for ALL uniforms (even
    // ones that exist in the shader), so every glUniform* is a no-op →
    // shaders receive zero/uninitialized MVP matrices → geometry renders at
    // origin with identity transform → black screen.
    // 对照 MobileGL DirectVulkan.cpp:171-244 AddBufferVariablesRecursive.
    p->uboBackingStore.clear();
    p->uboSizes.clear();
    // App-block routing state is rebuilt from this reflection pass, so drop
    // any stale values from a previous link (block indices can shift between
    // links; uniformBlockBindings is re-set by the app after relink).
    p->blockIndexForDescriptor.clear();
    p->blockInfos.clear();
    p->uniformBlockBindings.clear();
    try {
        std::vector<mithril::vk::DescriptorBinding> bindings;
        if (!p->vertexSpirv.empty()) {
            auto b = mithril::vk::reflect_stage(p->vertexSpirv.data(),
                                                (int)p->vertexSpirv.size(),
                                                VK_SHADER_STAGE_VERTEX_BIT);
            mithril::vk::merge_bindings(bindings, b);
        }
        if (!p->fragmentSpirv.empty()) {
            auto b = mithril::vk::reflect_stage(p->fragmentSpirv.data(),
                                                (int)p->fragmentSpirv.size(),
                                                VK_SHADER_STAGE_FRAGMENT_BIT);
            mithril::vk::merge_bindings(bindings, b);
        }
        GLint nextLocation = 0;
        GLuint blockIndex = 0;
        for (const auto& db : bindings) {
            if (db.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                if (!db.name.empty()) {
                    p->uniformBlocks[db.name] = blockIndex;
                }
                p->uboSizes[db.binding] = db.bufferSize;
                auto& store = p->uboBackingStore[db.binding];
                store.assign(db.bufferSize ? db.bufferSize : 0, 0);
                for (const auto& m : db.members) {
                    if (m.name.empty()) continue;
                    mithril::Uniform u{};
                    u.name = m.name;
                    u.location = nextLocation++;
                    u.blockIndex = (GLint)blockIndex;
                    u.blockBinding = (GLint)db.binding;
                    u.offset = (GLint)m.offset;
                    switch (m.size) {
                        case 8:  u.type = GL_FLOAT_VEC2; break;
                        case 12: u.type = GL_FLOAT_VEC3; break;
                        case 16: u.type = GL_FLOAT_VEC4; break;
                        case 24: u.type = GL_FLOAT_MAT3; break;
                        case 32: u.type = GL_FLOAT_MAT2x4; break;
                        case 48: u.type = GL_FLOAT_MAT4x3; break;
                        case 64: u.type = GL_FLOAT_MAT4; break;
                        case 36: u.type = GL_FLOAT_MAT3x4; break;
                        default: u.type = GL_FLOAT; break;
                    }
                    u.size = 1;
                    p->uniforms[m.name] = u;
                    p->uniformByLocation[u.location] = m.name;
                }
                // FIX (root cause AL): route APPLICATION-declared blocks to the
                // backend's appBlock path. DescriptorSet.cpp build_ubo_plans
                // decides appBlock by looking up blockIndexForDescriptor; if it
                // is empty every block is misclassified as synthetic and read
                // from uboBackingStore (zeros) -> black screen.
                //
                // NOT every UBO is an app block though — only the ones the
                // application binds via glBindBufferBase/glUniformBlockBinding.
                // Mithril's own synthetic blocks must stay out:
                //   - mithril_GlobalBlock (the aggregated default-uniform block)
                //     is fed from the transient arena (UniformArena), never from
                //     a GL buffer; marking it appBlock would read zeros.
                //   - a per-loose-uniform UBO whose name matches a uniform is
                //     synthesized by glslang and read from the backing store.
                // Both are identified by name; app block names come from the
                // app's own GLSL and are never mithril_-prefixed nor match a
                // uniform.
                const bool synthGlobal = (db.name == "mithril_GlobalBlock");
                const bool synthLoose  = (p->uniforms.count(db.name) != 0);
                if (!synthGlobal && !synthLoose) {
                    p->blockIndexForDescriptor[(GLuint)db.binding] = blockIndex;
                    if (p->blockInfos.size() <= blockIndex) {
                        p->blockInfos.resize(blockIndex + 1);
                    }
                    mithril::UniformBlockInfo& info = p->blockInfos[blockIndex];
                    info.name = db.name;
                    info.dataSize = db.bufferSize ? (uint32_t)db.bufferSize : 0;
                    info.bindingPoint = blockIndex;  // GL default: binding == index
                }
                ++blockIndex;
            } else if (db.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                // Reflect sampler uniforms so glGetUniformLocation("Sampler0")
                // returns a valid location instead of -1. Without this, every
                // glUniform1i(samplerLoc, unit) is a no-op → the Vulkan
                // backend never learns which texture unit feeds each sampler →
                // multi-texture shaders (e.g. Minecraft) render black.
                // blockBinding stores the SPIR-V descriptor binding; the GL
                // texture unit is captured later in glUniform1i via
                // samplerUnitMap.
                if (!db.name.empty()) {
                    mithril::Uniform u{};
                    u.name = db.name;
                    u.location = nextLocation++;
                    u.blockIndex = -1;
                    u.blockBinding = (GLint)db.binding;
                    u.type = GL_SAMPLER_2D;
                    u.size = 1;
                    p->uniforms[db.name] = u;
                    p->uniformByLocation[u.location] = db.name;
                }
                p->samplerUnitMap[(GLuint)db.binding] = -1;
                // 同步初始化 samplerUnitForBinding（DescriptorSet.cpp 读这个 map）。
                // 之前只写 samplerUnitMap 不写 samplerUnitForBinding，靠
                // `unit = db.binding` 的 legacy fallback 碰巧工作（binding 0~31
                // == texture unit 0~31）。现在 inject_opaque_bindings 给 FS 的
                // sampler binding 加了 64 偏移，fallback 会取 texture unit 65 越界。
                // 在 link 时用 -1 初始化，glUniform1i 时再写入真实 unit。
                p->samplerUnitForBinding[(GLuint)db.binding] = -1;
            }
        }
    } catch (const std::exception& e) {
        MITHRIL_LOG_WARN("program", "Uniform reflection failed for program %u: %s",
                         program, e.what());
    }
    MITHRIL_LOG_INFO("program", "Linked program %u (VS=%zu VS_yflip=%zu FS=%zu SPIR-V words)",
                     program, p->vertexSpirv.size(), p->vertexSpirvYFlipped.size(),
                     p->fragmentSpirv.size());
}

void glUseProgram(GLuint program) {
    MITHRIL_ENSURE_INIT();
    if (program != 0 && !mithril::state_get_program(program)) {
        mithril::state_set_error(GL_INVALID_OPERATION);
        return;
    }
    // P1-5 deferred deletion: if the previously-current program was marked
    // for deletion and is being replaced (by 0 or another program), finish
    // the deletion now. This is the trigger for programs deleted while
    // current — glDeleteProgram left them alive precisely for this moment.
    GLuint prev = g_state->currentProgram;
    if (prev != 0 && prev != program) {
        mithril::Program* pp = mithril::state_get_program(prev);
        if (pp && pp->markedForDeletion) {
            backend_delete_program_resources(prev);
            g_state->programs.erase(prev);
            g_state->programNames.release(prev);
        }
    }
    g_state->currentProgram = program;
}

void glValidateProgram(GLuint program) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p) return;
    // Validation is a no-op for our purposes; report success if linked.
    (void)p;
}

void glGetShaderiv(GLuint shader, GLenum pname, GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Shader* s = mithril::state_get_shader(shader);
    if (!s) { *params = 0; return; }
    switch (pname) {
        case GL_SHADER_TYPE:        *params = (GLint)s->type; break;
        case GL_COMPILE_STATUS:     *params = s->compiled ? GL_TRUE : GL_FALSE; break;
        case GL_INFO_LOG_LENGTH:    *params = (GLint)s->infoLog.size(); break;
        case GL_SHADER_SOURCE_LENGTH:*params = (GLint)s->source.size() + 1; break;
        default:                    *params = 0; break;
    }
}

void glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog) {
    MITHRIL_ENSURE_INIT();
    mithril::Shader* s = mithril::state_get_shader(shader);
    if (!s || !infoLog || bufSize <= 0) { if (length) *length = 0; return; }
    GLsizei n = (GLsizei)s->infoLog.size();
    if (n > bufSize - 1) n = bufSize - 1;
    std::memcpy(infoLog, s->infoLog.data(), n);
    infoLog[n] = 0;
    if (length) *length = n;
}

void glGetShaderSource(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* source) {
    MITHRIL_ENSURE_INIT();
    mithril::Shader* s = mithril::state_get_shader(shader);
    if (!s || !source || bufSize <= 0) { if (length) *length = 0; return; }
    GLsizei n = (GLsizei)s->source.size();
    if (n > bufSize - 1) n = bufSize - 1;
    std::memcpy(source, s->source.data(), n);
    source[n] = 0;
    if (length) *length = n;
}

void glGetProgramiv(GLuint program, GLenum pname, GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Program* p = mithril::state_get_program(program);
    if (!p) { *params = 0; return; }
    switch (pname) {
        case GL_LINK_STATUS:     *params = p->linked ? GL_TRUE : GL_FALSE; break;
        case GL_VALIDATE_STATUS: *params = GL_TRUE; break;
        case GL_INFO_LOG_LENGTH: *params = (GLint)p->infoLog.size(); break;
        case GL_ACTIVE_UNIFORMS: *params = (GLint)p->uniforms.size(); break;
        case GL_ACTIVE_ATTRIBUTES: *params = (GLint)p->attribs.size(); break;
        case GL_ATTACHED_SHADERS: *params = (GLint)p->attachedShaders.size(); break;
        default:                 *params = 0; break;
    }
}

void glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p || !infoLog || bufSize <= 0) { if (length) *length = 0; return; }
    GLsizei n = (GLsizei)p->infoLog.size();
    if (n > bufSize - 1) n = bufSize - 1;
    std::memcpy(infoLog, p->infoLog.data(), n);
    infoLog[n] = 0;
    if (length) *length = n;
}

void glGetAttachedShaders(GLuint program, GLsizei maxCount, GLsizei* count, GLuint* shaders) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p || !shaders) { if (count) *count = 0; return; }
    GLsizei n = (GLsizei)p->attachedShaders.size();
    if (n > maxCount) n = maxCount;
    for (GLsizei i = 0; i < n; ++i) shaders[i] = p->attachedShaders[i];
    if (count) *count = n;
}

GLint glGetUniformLocation(GLuint program, const GLchar* name) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p || !name) return -1;
    // P1-6 FIX: querying a location on an unlinked program is GL_INVALID_OPERATION.
    // Never insert a synthetic uniform entry as a side effect of the query —
    // only return locations for uniforms that exist in the program's table.
    if (!p->linked) {
        mithril::state_set_error(GL_INVALID_OPERATION);
        return -1;
    }
    auto it = p->uniforms.find(name);
    if (it == p->uniforms.end()) return -1;
    return it->second.location;
}

void glGetActiveUniform(GLuint program, GLuint index, GLsizei bufSize,
                        GLsizei* length, GLint* size, GLenum* type, GLchar* name) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p || !name || bufSize <= 0) { if (length) *length = 0; return; }
    if (index >= p->uniforms.size()) { if (length) *length = 0; return; }
    // Linear scan to the index-th entry.
    GLuint i = 0;
    for (auto& kv : p->uniforms) {
        if (i == index) {
            GLsizei n = (GLsizei)kv.first.size();
            if (n > bufSize - 1) n = bufSize - 1;
            std::memcpy(name, kv.first.data(), n);
            name[n] = 0;
            if (length) *length = n;
            if (size) *size = 1;
            if (type) *type = GL_FLOAT;
            return;
        }
        ++i;
    }
    if (length) *length = 0;
}

void glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize,
                       GLsizei* length, GLint* size, GLenum* type, GLchar* name) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p || !name || bufSize <= 0) { if (length) *length = 0; return; }
    if (index >= p->attribs.size()) { if (length) *length = 0; return; }
    GLuint i = 0;
    for (auto& kv : p->attribs) {
        if (i == index) {
            GLsizei n = (GLsizei)kv.first.size();
            if (n > bufSize - 1) n = bufSize - 1;
            std::memcpy(name, kv.first.data(), n);
            name[n] = 0;
            if (length) *length = n;
            if (size) *size = 1;
            if (type) *type = GL_FLOAT;
            return;
        }
        ++i;
    }
    if (length) *length = 0;
}

void glGetUniformfv(GLuint program, GLint location, GLfloat* params) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p || !params) return;
    auto it = p->uniformByLocation.find(location);
    if (it == p->uniformByLocation.end()) { *params = 0; return; }
    auto& u = p->uniforms[it->second];
    if (!u.value.empty()) *params = u.value[0];
    else *params = 0;
}

void glGetUniformiv(GLuint program, GLint location, GLint* params) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p || !params) return;
    auto it = p->uniformByLocation.find(location);
    if (it == p->uniformByLocation.end()) { *params = 0; return; }
    auto& u = p->uniforms[it->second];
    *params = u.value.empty() ? 0 : (GLint)u.value[0];
}

GLuint glGetUniformBlockIndex(GLuint program, const GLchar* uniformBlockName) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p || !uniformBlockName) return 0xFFFFFFFFu;
    auto it = p->uniformBlocks.find(uniformBlockName);
    if (it == p->uniformBlocks.end()) return 0xFFFFFFFFu;
    return it->second;
}

void glGetActiveUniformBlockiv(GLuint program, GLuint uniformBlockIndex,
                               GLenum pname, GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Program* p = mithril::state_get_program(program);
    if (!p) { *params = 0; return; }
    // uniformBlockIndex is the binding value we stored (see uniformBlocks map).
    // Find the block by binding value and answer common queries.
    switch (pname) {
        case GL_UNIFORM_BLOCK_BINDING:
            *params = (GLint)uniformBlockIndex;
            break;
        case GL_UNIFORM_BLOCK_DATA_SIZE: {
            // Sum member sizes for this binding. Members are stored in
            // p->uniforms with blockBinding == uniformBlockIndex.
            auto type_size = [](GLenum type, GLint arrSize) -> GLint {
                GLint base = 4;
                switch (type) {
                    case GL_FLOAT: base = 4; break;
                    case GL_FLOAT_VEC2: base = 8; break;
                    case GL_FLOAT_VEC3: base = 12; break;
                    case GL_FLOAT_VEC4: base = 16; break;
                    case GL_INT: case GL_UNSIGNED_INT: case GL_BOOL: base = 4; break;
                    case GL_INT_VEC2: case GL_UNSIGNED_INT_VEC2: case GL_BOOL_VEC2: base = 8; break;
                    case GL_INT_VEC3: case GL_UNSIGNED_INT_VEC3: case GL_BOOL_VEC3: base = 12; break;
                    case GL_INT_VEC4: case GL_UNSIGNED_INT_VEC4: case GL_BOOL_VEC4: base = 16; break;
                    case GL_FLOAT_MAT2: base = 16; break;
                    case GL_FLOAT_MAT3: base = 36; break;
                    case GL_FLOAT_MAT4: base = 64; break;
                    default: base = 16; break;  // conservative
                }
                return base * std::max(1, arrSize);
            };
            GLint total = 0;
            for (const auto& kv : p->uniforms) {
                if (kv.second.blockBinding == (GLint)uniformBlockIndex) {
                    total += std::max(kv.second.arrayStride,
                                      type_size(kv.second.type, kv.second.size));
                }
            }
            *params = total;
            break;
        }
        case GL_UNIFORM_BLOCK_NAME_LENGTH: {
            // Find block name by binding.
            std::string name;
            for (const auto& kv : p->uniformBlocks) {
                if (kv.second == uniformBlockIndex) { name = kv.first; break; }
            }
            *params = (GLint)name.length() + 1;  // include null terminator
            break;
        }
        case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS: {
            GLint count = 0;
            for (const auto& kv : p->uniforms) {
                if (kv.second.blockBinding == (GLint)uniformBlockIndex) ++count;
            }
            *params = count;
            break;
        }
        case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER:
            *params = GL_TRUE;  // conservative: assume VS references it
            break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER:
            *params = GL_TRUE;  // conservative: assume FS references it
            break;
        default:
            *params = 0;
            break;
    }
}

void glUniformBlockBinding(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding) {
    MITHRIL_ENSURE_INIT();
    mithril::Program* p = mithril::state_get_program(program);
    if (!p) return;
    p->uniformBlockBindings[uniformBlockIndex] = uniformBlockBinding;
}

/* ---- Uniform setters ----
 * The Vulkan backend consumes uniform values via push constants or uniform
 * buffers bound by the draw path. Here we just cache the latest value on the
 * program so the draw path can push it into a uniform buffer.
 */
static mithril::Program* current_program() {
    return mithril::state_get_program(g_state->currentProgram);
}

static void store_uniform(GLint location, const GLfloat* v, int count, int comps) {
    mithril::Program* p = current_program();
    if (!p || location < 0 || !v) return;
    auto it = p->uniformByLocation.find(location);
    if (it == p->uniformByLocation.end()) return;  // unknown location: drop
    mithril::Uniform& u = p->uniforms[it->second];
    u.value.assign(v, v + (size_t)count * comps);
    // Write raw bytes into the UBO backing store at the reflected offset.
    // DescriptorSet.cpp memcpys the whole store into the UBO payload at draw.
    if (u.blockBinding >= 0 && u.offset >= 0) {
        auto bs = p->uboBackingStore.find((GLuint)u.blockBinding);
        if (bs != p->uboBackingStore.end()) {
            size_t bytes = (size_t)count * comps * sizeof(float);
            if ((size_t)u.offset + bytes <= bs->second.size()) {
                std::memcpy(bs->second.data() + u.offset, v, bytes);
            }
        }
    }
}

static void store_uniform_int(GLint location, const GLint* v, int count, int comps) {
    mithril::Program* p = current_program();
    if (!p || location < 0 || !v) return;
    auto it = p->uniformByLocation.find(location);
    if (it == p->uniformByLocation.end()) return;  // unknown location: drop
    mithril::Uniform& u = p->uniforms[it->second];
    // Store as floats for glGetUniform* compatibility.
    u.value.resize((size_t)count * comps);
    for (size_t i = 0; i < u.value.size(); ++i) u.value[i] = (GLfloat)v[i];
    // Write raw bytes into the UBO backing store at the reflected offset.
    if (u.blockBinding >= 0 && u.offset >= 0) {
        auto bs = p->uboBackingStore.find((GLuint)u.blockBinding);
        if (bs != p->uboBackingStore.end()) {
            size_t bytes = (size_t)count * comps * sizeof(GLint);
            if ((size_t)u.offset + bytes <= bs->second.size()) {
                std::memcpy(bs->second.data() + u.offset, v, bytes);
            }
        }
    }
}

void glUniform1f(GLint loc, GLfloat v0)                                    { store_uniform(loc, &v0, 1, 1); }
void glUniform2f(GLint loc, GLfloat v0, GLfloat v1)                        { GLfloat v[2] = {v0,v1}; store_uniform(loc, v, 1, 2); }
void glUniform3f(GLint loc, GLfloat v0, GLfloat v1, GLfloat v2)            { GLfloat v[3] = {v0,v1,v2}; store_uniform(loc, v, 1, 3); }
void glUniform4f(GLint loc, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3){ GLfloat v[4] = {v0,v1,v2,v3}; store_uniform(loc, v, 1, 4); }

void glUniform1i(GLint loc, GLint v0) {
    store_uniform_int(loc, &v0, 1, 1);
    // Sampler unit mapping: when the app calls glUniform1i(samplerLoc, unit)
    // to bind a sampler to a texture unit, record the unit in samplerUnitMap
    // keyed by the sampler's SPIR-V descriptor binding. The Vulkan backend
    // reads this map at draw time to wire VkDescriptorImageInfo for the
    // correct texture unit. Without this, multi-texture shaders (Minecraft)
    // get samplers pointing at the wrong (or no) texture → black screen.
    mithril::Program* p = current_program();
    if (p && loc >= 0) {
        auto it = p->uniformByLocation.find(loc);
        if (it != p->uniformByLocation.end()) {
            auto uit = p->uniforms.find(it->second);
            if (uit != p->uniforms.end()) {
                mithril::Uniform& u = uit->second;
                // Sampler types fall in 0x8B5E..0x8B60 (GL_SAMPLER_2D,
                // GL_SAMPLER_3D, GL_SAMPLER_CUBE). blockBinding holds the
                // SPIR-V descriptor binding for reflected samplers (>= 0).
                if (u.blockBinding >= 0 && u.type >= 0x8B5E && u.type <= 0x8B60) {
                    p->samplerUnitMap[(GLuint)u.blockBinding] = v0;
                    // 同步 samplerUnitForBinding（DescriptorSet.cpp 实际读的 map）。
                    p->samplerUnitForBinding[(GLuint)u.blockBinding] = v0;
                }
            }
        }
    }
}
void glUniform2i(GLint loc, GLint v0, GLint v1)                            { GLint v[2] = {v0,v1}; store_uniform_int(loc, v, 1, 2); }
void glUniform3i(GLint loc, GLint v0, GLint v1, GLint v2)                  { GLint v[3] = {v0,v1,v2}; store_uniform_int(loc, v, 1, 3); }
void glUniform4i(GLint loc, GLint v0, GLint v1, GLint v2, GLint v3)        { GLint v[4] = {v0,v1,v2,v3}; store_uniform_int(loc, v, 1, 4); }

void glUniform1ui(GLint loc, GLuint v0)                                    { GLint v = (GLint)v0; store_uniform_int(loc, &v, 1, 1); }
void glUniform2ui(GLint loc, GLuint v0, GLuint v1)                         { GLint v[2] = {(GLint)v0,(GLint)v1}; store_uniform_int(loc, v, 1, 2); }
void glUniform3ui(GLint loc, GLuint v0, GLuint v1, GLuint v2)              { GLint v[3] = {(GLint)v0,(GLint)v1,(GLint)v2}; store_uniform_int(loc, v, 1, 3); }
void glUniform4ui(GLint loc, GLuint v0, GLuint v1, GLuint v2, GLuint v3)   { GLint v[4] = {(GLint)v0,(GLint)v1,(GLint)v2,(GLint)v3}; store_uniform_int(loc, v, 1, 4); }

void glUniform1fv(GLint loc, GLsizei c, const GLfloat* v) { store_uniform(loc, v, c, 1); }
void glUniform2fv(GLint loc, GLsizei c, const GLfloat* v) { store_uniform(loc, v, c, 2); }
void glUniform3fv(GLint loc, GLsizei c, const GLfloat* v) { store_uniform(loc, v, c, 3); }
void glUniform4fv(GLint loc, GLsizei c, const GLfloat* v) { store_uniform(loc, v, c, 4); }
void glUniform1iv(GLint loc, GLsizei c, const GLint* v)   { store_uniform_int(loc, v, c, 1); }
void glUniform2iv(GLint loc, GLsizei c, const GLint* v)   { store_uniform_int(loc, v, c, 2); }
void glUniform3iv(GLint loc, GLsizei c, const GLint* v)   { store_uniform_int(loc, v, c, 3); }
void glUniform4iv(GLint loc, GLsizei c, const GLint* v)   { store_uniform_int(loc, v, c, 4); }
void glUniform1uiv(GLint loc, GLsizei c, const GLuint* v) {
    std::vector<GLint> tmp(v, v + c); store_uniform_int(loc, tmp.data(), c, 1);
}
void glUniform2uiv(GLint loc, GLsizei c, const GLuint* v) {
    std::vector<GLint> tmp(v, v + c*2); store_uniform_int(loc, tmp.data(), c, 2);
}
void glUniform3uiv(GLint loc, GLsizei c, const GLuint* v) {
    std::vector<GLint> tmp(v, v + c*3); store_uniform_int(loc, tmp.data(), c, 3);
}
void glUniform4uiv(GLint loc, GLsizei c, const GLuint* v) {
    std::vector<GLint> tmp(v, v + c*4); store_uniform_int(loc, tmp.data(), c, 4);
}

// Store a matrix uniform with optional transpose. OpenGL receives matrices
// in column-major order; when transpose == GL_TRUE the app supplied row-major
// data and we must transpose each matrix before writing it to the backing
// store so the Vulkan backend (which always expects column-major) sees the
// correct layout. `cols`/`rows` describe the original matrix dimensions
// (matCxR => cols=C, rows=R); the element count per matrix is cols*rows.
static void store_uniform_matrix(GLint location, GLsizei count, GLboolean transpose,
                                 const GLfloat* v, int cols, int rows) {
    int elems = cols * rows;
    if (transpose == GL_TRUE && v && elems > 0) {
        std::vector<GLfloat> tmp((size_t)count * elems);
        for (int m = 0; m < count; ++m) {
            const GLfloat* src = v + (size_t)m * elems;
            GLfloat* dst = tmp.data() + (size_t)m * elems;
            // Column-major transpose: transposed[j*cols + i] = original[i*rows + j].
            // Works for square (mat2/mat3/mat4) and non-square (mat2x3 etc.) matrices.
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < rows; ++j) {
                    dst[j * cols + i] = src[i * rows + j];
                }
            }
        }
        store_uniform(location, tmp.data(), count, elems);
    } else {
        store_uniform(location, v, count, elems);
    }
}

void glUniformMatrix2fv(GLint loc, GLsizei c, GLboolean t, const GLfloat* v)   { store_uniform_matrix(loc, c, t, v, 2, 2); }
void glUniformMatrix3fv(GLint loc, GLsizei c, GLboolean t, const GLfloat* v)   { store_uniform_matrix(loc, c, t, v, 3, 3); }
void glUniformMatrix4fv(GLint loc, GLsizei c, GLboolean t, const GLfloat* v)   { store_uniform_matrix(loc, c, t, v, 4, 4); }
void glUniformMatrix2x3fv(GLint loc, GLsizei c, GLboolean t, const GLfloat* v) { store_uniform_matrix(loc, c, t, v, 2, 3); }
void glUniformMatrix3x2fv(GLint loc, GLsizei c, GLboolean t, const GLfloat* v) { store_uniform_matrix(loc, c, t, v, 3, 2); }
void glUniformMatrix2x4fv(GLint loc, GLsizei c, GLboolean t, const GLfloat* v) { store_uniform_matrix(loc, c, t, v, 2, 4); }
void glUniformMatrix4x2fv(GLint loc, GLsizei c, GLboolean t, const GLfloat* v) { store_uniform_matrix(loc, c, t, v, 4, 2); }
void glUniformMatrix3x4fv(GLint loc, GLsizei c, GLboolean t, const GLfloat* v) { store_uniform_matrix(loc, c, t, v, 3, 4); }
void glUniformMatrix4x3fv(GLint loc, GLsizei c, GLboolean t, const GLfloat* v) { store_uniform_matrix(loc, c, t, v, 4, 3); }

GLboolean glIsProgram(GLuint program) {
    // P1-5: validity is O(1) via NameAllocator::valid(). This covers both
    // never-allocated names (valid_bits unset) and deleted-and-released names
    // (release() marks invalid + pushes to freeList for reuse).
    return (mithril::g_state && mithril::g_state->programNames.valid(program))
        ? GL_TRUE : GL_FALSE;
}

GLboolean glIsShader(GLuint shader) {
    return (mithril::g_state && mithril::g_state->shaderNames.valid(shader))
        ? GL_TRUE : GL_FALSE;
}

} // extern "C"
