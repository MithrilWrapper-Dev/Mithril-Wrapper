// Mithril-Wrapper - MG_Impl/Texture.cpp
// Texture object management: storage, upload, parameters, mipmap generation.
//
// This is the Vulkan/MoltenVK rewrite of the former gl/texture.cpp. The Metal
// MTLTexture calls are replaced with the Vulkan backend C API
// (backend_get_or_create_texture / backend_texture_upload /
// backend_texture_set_params / backend_delete_texture) declared in
// MG_Backend/Backend.h. Vulkan VkImage/VkImageView objects are owned by the
// backend and keyed by GL texture name.
//
// Binding model (rewritten state machine): textures are bound per-unit
// per-target via g_state->textureBindings[unit][target] (BindingSlot). The
// legacy flat boundTextures[] / boundTextureTargets[] arrays are gone.
#include "includes.h"

// Pnames that are standard OpenGL but absent from our minimal glcorearb.h.
#ifndef GL_TEXTURE_SWIZZLE_RGBA
#define GL_TEXTURE_SWIZZLE_RGBA       0x8E46
#endif
#ifndef GL_TEXTURE_LOD_BIAS
#define GL_TEXTURE_LOD_BIAS           0x8501
#endif
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif

extern "C" {

void glGenTextures(GLsizei n, GLuint* textures) {
    MITHRIL_ENSURE_INIT();
    // state_gen_names routes through the NameAllocator (free_list + valid_bits).
    mithril::state_gen_names("texture", n, textures);
    for (GLsizei i = 0; i < n; ++i) {
        mithril::Texture t{};
        t.id = textures[i];
        g_state->textures[textures[i]] = t;
    }
}

void glDeleteTextures(GLsizei n, const GLuint* textures) {
    MITHRIL_ENSURE_INIT();
    if (n <= 0 || !textures) return;
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = textures[i];
        if (name == 0) continue;
        // Unbind from every unit / every per-target slot.
        for (int u = 0; u < mithril::kMaxTextureUnits; ++u) {
            for (int t = 0; t < mithril::kTextureTargetCount; ++t) {
                if (g_state->textureBindings[u][t].name == name)
                    g_state->textureBindings[u][t].bind(0);
            }
        }
        backend_delete_texture(name);
        g_state->textures.erase(name);
        g_state->textureNames.release(name);
    }
}

void glBindTexture(GLenum target, GLuint texture) {
    MITHRIL_ENSURE_INIT();
    mithril::TextureTarget tt = mithril::textureTargetFromGL(target);
    if (tt == mithril::TextureTarget::Count) {
        mithril::state_set_error(GL_INVALID_ENUM);
        return;
    }
    GLuint unit = g_state->activeTextureUnit;
    if (unit >= mithril::kMaxTextureUnits) return;
    // Lazy-create on first bind (matches the GL name-allocator semantics).
    if (texture != 0 && !mithril::state_get_texture(texture)) {
        mithril::Texture t{};
        t.id = texture;
        g_state->textures[texture] = t;
    }
    g_state->textureBindings[unit][(int)tt].bind(texture);
    if ((int)unit > g_state->maxTouchedTextureUnit)
        g_state->maxTouchedTextureUnit = (int)unit;
    ++g_state->textureBindGeneration;
    if (mithril::Texture* t = mithril::state_get_texture(texture)) {
        t->target = target;
    }
}

/*
 * Look up the texture bound to the active unit for `target`.
 *
 * Replaces the old flat boundTextures[unit] lookup. Because the binding model
 * is now per-target, callers that already know the target (glTexImage2D etc.)
 * go straight to the matching slot. The lookup also enforces target
 * consistency: if the bound texture object already has a target assigned (set
 * by a prior glBindTexture) and it does not match the target passed here, the
 * call records GL_INVALID_OPERATION and returns nullptr (P0-5 / spec 4.2).
 *
 * For the "first non-zero texture on the unit" case (used by the backend
 * descriptor set binding), prefer g_state->boundTextureForUnit(unit).
 */
static mithril::Texture* bound_texture_for_target(GLenum target) {
    mithril::TextureTarget tt = mithril::textureTargetFromGL(target);
    if (tt == mithril::TextureTarget::Count) return nullptr;
    GLuint unit = g_state->activeTextureUnit;
    if (unit >= mithril::kMaxTextureUnits) return nullptr;
    GLuint id = g_state->textureBindings[unit][(int)tt].name;
    mithril::Texture* t = mithril::state_get_texture(id);
    if (t) {
        // FIX (Main-menu panorama cubemap GPU fault root cause): the target
        // consistency check must normalize the 6 cubemap face targets
        // (GL_TEXTURE_CUBE_MAP_POSITIVE_X .. NEGATIVE_Z). The texture was
        // bound as GL_TEXTURE_CUBE_MAP but a face is uploaded with a face
        // target, so a raw GLenum comparison rejects the upload with
        // GL_INVALID_OPERATION -> every cubemap face is silently dropped ->
        // the panorama is empty -> sampling uninitialized layers -> GPU fault.
        // Compare through textureTargetFromGL instead (face and CUBE_MAP both
        // resolve to TextureTarget::CubeMap).
        bool same = (t->target == target);
        if (!same) {
            mithril::TextureTarget t1 = mithril::textureTargetFromGL(t->target);
            same = (t1 != mithril::TextureTarget::Count && t1 == tt);
        }
        if (!same) {
            mithril::state_set_error(GL_INVALID_OPERATION);
            return nullptr;
        }
    }
    return t;
}

void glTexImage2D(GLenum target, GLint level, GLint internalFormat,
                  GLsizei width, GLsizei height, GLint border,
                  GLenum format, GLenum type, const void* pixels) {
    MITHRIL_ENSURE_INIT();
    if (border != 0) { mithril::state_set_error(GL_INVALID_VALUE); return; }

    // GL_PROXY_TEXTURE_2D: no real texture is created. Just record the
    // requested dimensions so glGetTexLevelParameteriv can report them.
    // Minecraft probes max texture size this way (GL_PROXY_TEXTURE_2D with
    // progressively larger sizes until the query returns 0).
    if (target == GL_PROXY_TEXTURE_2D) {
        // Accept the size if it's within our reported GL_MAX_TEXTURE_SIZE.
        // A size of 0 means "unsupported" per the GL spec.
        GLint maxSize = 16384; // matches GL_MAX_TEXTURE_SIZE in Getter.cpp
        if (width > 0 && height > 0 && width <= maxSize && height <= maxSize) {
            g_state->proxyTexture2D.width  = width;
            g_state->proxyTexture2D.height = height;
            g_state->proxyTexture2D.internalFormat = internalFormat;
            g_state->proxyTexture2D.valid = true;
        } else {
            g_state->proxyTexture2D.valid = false;
            g_state->proxyTexture2D.width = 0;
            g_state->proxyTexture2D.height = 0;
        }
        return;
    }

    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    if (level == 0) {
        t->internalFormat = internalFormat;
        t->width  = width;
        t->height = height;
        t->depth  = 1;
    }
    if (t->levels < level + 1) t->levels = level + 1;

    // FIX (Root Cause AI - glTexImage2D mipmap uses base level dimensions):
    // VkImage 的 extent 必须始终是 base level (level 0) 的尺寸，而非当前 level
    // 的尺寸。旧代码用当前 level 的 width/height 调用 backend_get_or_create_texture，
    // 上传 level 1 时 width/height = base/2 → Resources.cpp 的复用条件不满足 →
    // 重建 VkImage with extent=(base/2, base/2) → base level 数据丢失 + VkImage
    // extent 错误 → 纹理腐败 → 红屏/花屏。
    // 修复：始终用 t->width / t->height（level==0 时更新的 base level 尺寸）。
    // 当前 level 的数据仍通过 level 参数上传到正确的 mip level（imageExtent
    // 在 backend_texture_upload 内按当前 level 的 width/height 设置）。
    // 对照 MobileGL CheckMipmapCompleteness (VkTextureManager.cpp:1918-1957)：
    // MobileGL 始终用 base level 尺寸作为 VkImage extent。
    backend_get_or_create_texture(t->id, t->width, t->height, 1, t->levels,
                                  internalFormat, target, 1);
    if (pixels) {
        MGUnpackParams unpack{
            g_state->pixelStore.unpackAlignment,
            g_state->pixelStore.unpackRowLength,
            g_state->pixelStore.unpackSkipPixels,
            g_state->pixelStore.unpackSkipRows,
            g_state->pixelStore.unpackImageHeight,
            g_state->pixelStore.unpackSkipImages
        };
        // FIX (Main-menu panorama cubemap GPU fault root cause): when the target
        // is a cubemap face (GL_TEXTURE_CUBE_MAP_POSITIVE_X .. NEGATIVE_Z),
        // pass the face index (0-5) as z. The backend's stage_and_copy_image
        // maps z to VkBufferImageCopy's baseArrayLayer (cubemap array layer ==
        // face). The old code always passed z=0, cramming all 6 faces into
        // layer 0 (and, before the face-target fix, dropping the uploads
        // entirely) -> the panorama was empty.
        GLint uploadZ = 0;
        if (target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X &&
            target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) {
            uploadZ = (GLint)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X);
        }
        backend_texture_upload(t->id, level, 0, 0, uploadZ, width, height, 1,
                               format, type, pixels, &unpack,
                               /*is_full_upload=*/1);
    }
}

void glTexImage3D(GLenum target, GLint level, GLint internalFormat,
                  GLsizei width, GLsizei height, GLsizei depth, GLint border,
                  GLenum format, GLenum type, const void* pixels) {
    MITHRIL_ENSURE_INIT();
    if (border != 0) { mithril::state_set_error(GL_INVALID_VALUE); return; }
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    if (level == 0) {
        t->internalFormat = internalFormat;
        t->width  = width;
        t->height = height;
        t->depth  = depth;
    }
    if (t->levels < level + 1) t->levels = level + 1;

    backend_get_or_create_texture(t->id, width, height, depth, t->levels,
                                  internalFormat, target, 1);
    if (pixels) {
        MGUnpackParams unpack{
            g_state->pixelStore.unpackAlignment,
            g_state->pixelStore.unpackRowLength,
            g_state->pixelStore.unpackSkipPixels,
            g_state->pixelStore.unpackSkipRows,
            g_state->pixelStore.unpackImageHeight,
            g_state->pixelStore.unpackSkipImages
        };
        backend_texture_upload(t->id, level, 0, 0, 0, width, height, depth,
                               format, type, pixels, &unpack,
                               /*is_full_upload=*/1);
    }
}

/*
 * glTexStorage2D / glTexStorage3D: allocate immutable storage for a texture.
 * Minecraft 1.21 uses these (rather than glTexImage2D) to create framebuffer
 * attachments, especially depth/stencil textures. We set the GL-level metadata
 * (internalFormat, dimensions, levels) and create the Vulkan texture with the
 * correct VkFormat up front. No pixel data is uploaded (immutable storage
 * starts uninitialised, like glTexImage2D with pixels=NULL).
 *
 * The Vulkan image is created with initialLayout = UNDEFINED. We immediately
 * transition it to SHADER_READ_ONLY_OPTIMAL so that:
 *   - If the texture is sampled before being rendered into, the layout is valid.
 *   - If the texture is attached to an FBO and rendered into, dynamic rendering
 *     will transition it to COLOR/DEPTH_STENCIL_ATTACHMENT_OPTIMAL (and the
 *     tracked layout lets our barrier code emit the correct oldLayout).
 * Without this transition, the texture sits in UNDEFINED and a subsequent
 * vkCmdBindDescriptorSets + draw that samples it would be a validation error.
 */
void glTexStorage2D(GLenum target, GLsizei levels, GLenum internalFormat,
                    GLsizei width, GLsizei height) {
    MITHRIL_ENSURE_INIT();
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t || levels <= 0) return;
    t->internalFormat = internalFormat;
    t->width  = width;
    t->height = height;
    t->depth  = 1;
    t->levels = levels;
    t->immutable = true;
    t->immutableLevels = levels;

    backend_get_or_create_texture(t->id, width, height, 1, levels,
                                  internalFormat, target, 1);
    // Transition UNDEFINED -> SHADER_READ_ONLY_OPTIMAL so the texture is in a
    // valid sampling layout before any draw references it.
    backend_transition_texture_layout(t->id, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void glTexStorage3D(GLenum target, GLsizei levels, GLenum internalFormat,
                    GLsizei width, GLsizei height, GLsizei depth) {
    MITHRIL_ENSURE_INIT();
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t || levels <= 0) return;
    t->internalFormat = internalFormat;
    t->width  = width;
    t->height = height;
    t->depth  = depth;
    t->levels = levels;
    t->immutable = true;
    t->immutableLevels = levels;

    backend_get_or_create_texture(t->id, width, height, depth, levels,
                                  internalFormat, target, 1);
    backend_transition_texture_layout(t->id, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset,
                     GLsizei width, GLsizei height,
                     GLenum format, GLenum type, const void* pixels) {
    MITHRIL_ENSURE_INIT();
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t || !pixels) return;
    MGUnpackParams unpack{
        g_state->pixelStore.unpackAlignment,
        g_state->pixelStore.unpackRowLength,
        g_state->pixelStore.unpackSkipPixels,
        g_state->pixelStore.unpackSkipRows,
        g_state->pixelStore.unpackImageHeight,
        g_state->pixelStore.unpackSkipImages
    };
    // FIX (cubemap face upload, same as glTexImage2D): face target -> z = face
    // index (the backend maps it to baseArrayLayer).
    GLint subZ = 0;
    if (target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X &&
        target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) {
        subZ = (GLint)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X);
    }
    backend_texture_upload(t->id, level, xoffset, yoffset, subZ,
                           width, height, 1, format, type, pixels, &unpack,
                           /*is_full_upload=*/0);
}

void glTexSubImage3D(GLenum target, GLint level,
                     GLint xoffset, GLint yoffset, GLint zoffset,
                     GLsizei width, GLsizei height, GLsizei depth,
                     GLenum format, GLenum type, const void* pixels) {
    MITHRIL_ENSURE_INIT();
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t || !pixels) return;
    MGUnpackParams unpack{
        g_state->pixelStore.unpackAlignment,
        g_state->pixelStore.unpackRowLength,
        g_state->pixelStore.unpackSkipPixels,
        g_state->pixelStore.unpackSkipRows,
        g_state->pixelStore.unpackImageHeight,
        g_state->pixelStore.unpackSkipImages
    };
    backend_texture_upload(t->id, level, xoffset, yoffset, zoffset,
                           width, height, depth, format, type, pixels, &unpack,
                           /*is_full_upload=*/0);
}

void glTexImage2DMultisample(GLenum target, GLsizei samples, GLenum internalformat,
                             GLsizei width, GLsizei height,
                             GLboolean fixedsamplelocations) {
    MITHRIL_ENSURE_INIT();
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    t->internalFormat = internalformat;
    t->width  = width;
    t->height = height;
    t->depth  = 1;
    t->samples = samples;
    t->fixedSampleLocations = fixedsamplelocations != 0;
    backend_get_or_create_texture(t->id, width, height, 1, 1,
                                  internalformat, target, samples > 1 ? samples : 1);
}

// ---------------------------------------------------------------------------
// Compressed texture upload (GL 3.1+ / ARB_texture_compression).
//
// 压缩纹理数据直接 memcpy 到 staging buffer，vkCmdCopyBufferToImage 按块
// 拷贝。iOS/Metal 原生支持 ASTC/ETC2/EAC（Apple GPU），BC1-BC7 需要
// MoltenVK 1.2.9+ 的 emulate-default-* 选项（或硬件解码）。FormatMap.cpp
// 已映射所有这些格式到对应的 VkFormat。
//
// 深度参考 MobileGL VkTextureManager::UploadCompressedTexture：数据直接
// 拷贝到 staging，VkBufferImageCopy.bufferRowLength=0（紧密排列），不
// 做像素展开。本实现等价。
// ---------------------------------------------------------------------------

void glCompressedTexImage2D(GLenum target, GLint level, GLenum internalformat,
                            GLsizei width, GLsizei height, GLint border,
                            GLsizei imageSize, const void* data) {
    MITHRIL_ENSURE_INIT();
    if (border != 0) { mithril::state_set_error(GL_INVALID_VALUE); return; }
    if (imageSize <= 0 || !data) { mithril::state_set_error(GL_INVALID_VALUE); return; }
    if (target == GL_PROXY_TEXTURE_2D) {
        // Proxy query: accept if within max texture size.
        GLint maxSize = 16384;
        if (width > 0 && height > 0 && width <= maxSize && height <= maxSize) {
            g_state->proxyTexture2D.width = width;
            g_state->proxyTexture2D.height = height;
            g_state->proxyTexture2D.internalFormat = internalformat;
            g_state->proxyTexture2D.valid = true;
        } else {
            g_state->proxyTexture2D.valid = false;
        }
        return;
    }
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    if (level == 0) {
        t->internalFormat = internalformat;
        t->width  = width;
        t->height = height;
        t->depth  = 1;
        t->isCompressed = true;
    }
    if (t->levels < level + 1) t->levels = level + 1;
    backend_get_or_create_texture(t->id, t->width, t->height, 1, t->levels,
                                  internalformat, target, 1);
    // FIX (cubemap face upload, same as glTexImage2D): face target -> z = face
    // index (the backend maps it to baseArrayLayer).
    GLint cz = 0;
    if (target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X &&
        target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) {
        cz = (GLint)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X);
    }
    backend_texture_upload_compressed(t->id, level, 0, 0, cz, width, height, 1,
                                      internalformat, imageSize, data,
                                      /*is_full_upload=*/1);
}

void glCompressedTexImage3D(GLenum target, GLint level, GLenum internalformat,
                            GLsizei width, GLsizei height, GLsizei depth, GLint border,
                            GLsizei imageSize, const void* data) {
    MITHRIL_ENSURE_INIT();
    if (border != 0) { mithril::state_set_error(GL_INVALID_VALUE); return; }
    if (imageSize <= 0 || !data) { mithril::state_set_error(GL_INVALID_VALUE); return; }
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    if (level == 0) {
        t->internalFormat = internalformat;
        t->width  = width;
        t->height = height;
        t->depth  = depth;
        t->isCompressed = true;
    }
    if (t->levels < level + 1) t->levels = level + 1;
    backend_get_or_create_texture(t->id, width, height, depth, t->levels,
                                  internalformat, target, 1);
    backend_texture_upload_compressed(t->id, level, 0, 0, 0, width, height, depth,
                                      internalformat, imageSize, data,
                                      /*is_full_upload=*/1);
}

void glCompressedTexSubImage2D(GLenum target, GLint level,
                               GLint xoffset, GLint yoffset,
                               GLsizei width, GLsizei height,
                               GLenum format, GLsizei imageSize, const void* data) {
    MITHRIL_ENSURE_INIT();
    if (imageSize <= 0 || !data) { mithril::state_set_error(GL_INVALID_VALUE); return; }
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    // format parameter is the compressed format; pass as internalFormat to backend.
    backend_texture_upload_compressed(t->id, level, xoffset, yoffset, 0,
                                      width, height, 1, format, imageSize, data,
                                      /*is_full_upload=*/0);
}

void glCompressedTexSubImage3D(GLenum target, GLint level,
                               GLint xoffset, GLint yoffset, GLint zoffset,
                               GLsizei width, GLsizei height, GLsizei depth,
                               GLenum format, GLsizei imageSize, const void* data) {
    MITHRIL_ENSURE_INIT();
    if (imageSize <= 0 || !data) { mithril::state_set_error(GL_INVALID_VALUE); return; }
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    backend_texture_upload_compressed(t->id, level, xoffset, yoffset, zoffset,
                                      width, height, depth, format, imageSize, data,
                                      /*is_full_upload=*/0);
}

void glCompressedTexImage1D(GLenum target, GLint level, GLenum internalformat,
                            GLsizei width, GLint border,
                            GLsizei imageSize, const void* data) {
    // 1D textures are emulated as 2D with height=1.
    glCompressedTexImage2D(target, level, internalformat, width, 1, border,
                           imageSize, data);
}

void glCompressedTexSubImage1D(GLenum target, GLint level,
                               GLint xoffset, GLsizei width,
                               GLenum format, GLsizei imageSize, const void* data) {
    glCompressedTexSubImage2D(target, level, xoffset, 0, width, 1,
                              format, imageSize, data);
}

void glTexParameterf(GLenum target, GLenum pname, GLfloat param) {
    MITHRIL_ENSURE_INIT();
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    GLint p = (GLint)param;
    bool samplerChanged = false;
    switch (pname) {
        case GL_TEXTURE_MIN_FILTER:        t->minFilter = p; samplerChanged = true; break;
        case GL_TEXTURE_MAG_FILTER:        t->magFilter = p; samplerChanged = true; break;
        case GL_TEXTURE_WRAP_S:            t->wrapS = p; samplerChanged = true; break;
        case GL_TEXTURE_WRAP_T:            t->wrapT = p; samplerChanged = true; break;
        case GL_TEXTURE_WRAP_R:            t->wrapR = p; samplerChanged = true; break;
        case GL_TEXTURE_BASE_LEVEL:        t->baseLevel = p; break;
        case GL_TEXTURE_MAX_LEVEL:         t->maxLevel = p; break;
        case GL_TEXTURE_MIN_LOD:           t->minLod = param; break;
        case GL_TEXTURE_MAX_LOD:           t->maxLod = param; break;
        case GL_TEXTURE_LOD_BIAS:          t->lodBias = param; break;
        case GL_TEXTURE_MAX_ANISOTROPY_EXT:t->maxAnisotropy = param; break;
        case GL_TEXTURE_COMPARE_MODE:      t->compareMode = (GLenum)p; break;
        case GL_TEXTURE_COMPARE_FUNC:       t->compareFunc = (GLenum)p; break;
        case GL_TEXTURE_SWIZZLE_R:          t->swizzleR = (GLenum)p; break;
        case GL_TEXTURE_SWIZZLE_G:          t->swizzleG = (GLenum)p; break;
        case GL_TEXTURE_SWIZZLE_B:          t->swizzleB = (GLenum)p; break;
        case GL_TEXTURE_SWIZZLE_A:          t->swizzleA = (GLenum)p; break;
        default:
            mithril::state_set_error(GL_INVALID_ENUM);
            return;
    }
    if (samplerChanged) {
        // Vulkan samplers are immutable; invalidate cached VkSampler so it's
        // rebuilt on next use. 对照 MobileGL VkSamplerManager.
        backend_invalidate_sampler_cache(t->id);
    }
    ++t->paramsVersion;
    backend_texture_set_params(t->id, t->minFilter, t->magFilter,
                               t->wrapS, t->wrapT, t->wrapR, t->borderColor);
}

void glTexParameteri(GLenum target, GLenum pname, GLint param) {
    glTexParameterf(target, pname, (GLfloat)param);
}

void glTexParameterfv(GLenum target, GLenum pname, const GLfloat* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    switch (pname) {
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; ++i) t->borderColor[i] = params[i];
            backend_invalidate_sampler_cache(t->id);
            break;
        case GL_TEXTURE_SWIZZLE_RGBA:
            t->swizzleR = (GLenum)params[0];
            t->swizzleG = (GLenum)params[1];
            t->swizzleB = (GLenum)params[2];
            t->swizzleA = (GLenum)params[3];
            break;
        default:
            // Scalar pnames share the scalar path (which bumps version +
            // pushes params to the backend + records GL_INVALID_ENUM on unknown).
            glTexParameterf(target, pname, params[0]);
            return;
    }
    ++t->paramsVersion;
    backend_texture_set_params(t->id, t->minFilter, t->magFilter,
                               t->wrapS, t->wrapT, t->wrapR, t->borderColor);
}

void glTexParameteriv(GLenum target, GLenum pname, const GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    switch (pname) {
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; ++i) t->borderColor[i] = (GLfloat)params[i];
            backend_invalidate_sampler_cache(t->id);
            break;
        case GL_TEXTURE_SWIZZLE_RGBA:
            t->swizzleR = (GLenum)params[0];
            t->swizzleG = (GLenum)params[1];
            t->swizzleB = (GLenum)params[2];
            t->swizzleA = (GLenum)params[3];
            break;
        default:
            glTexParameterf(target, pname, (GLfloat)params[0]);
            return;
    }
    ++t->paramsVersion;
    backend_texture_set_params(t->id, t->minFilter, t->magFilter,
                               t->wrapS, t->wrapT, t->wrapR, t->borderColor);
}

void glTexParameterIiv(GLenum target, GLenum pname, const GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    switch (pname) {
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; ++i) t->borderColorI[i] = params[i];
            backend_invalidate_sampler_cache(t->id);
            break;
        case GL_TEXTURE_SWIZZLE_RGBA:
            t->swizzleR = (GLenum)params[0];
            t->swizzleG = (GLenum)params[1];
            t->swizzleB = (GLenum)params[2];
            t->swizzleA = (GLenum)params[3];
            break;
        default:
            glTexParameterf(target, pname, (GLfloat)params[0]);
            return;
    }
    ++t->paramsVersion;
    backend_texture_set_params(t->id, t->minFilter, t->magFilter,
                               t->wrapS, t->wrapT, t->wrapR, t->borderColor);
}

void glTexParameterIuiv(GLenum target, GLenum pname, const GLuint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    switch (pname) {
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; ++i) t->borderColorUI[i] = (GLint)params[i];
            backend_invalidate_sampler_cache(t->id);
            break;
        case GL_TEXTURE_SWIZZLE_RGBA:
            t->swizzleR = (GLenum)params[0];
            t->swizzleG = (GLenum)params[1];
            t->swizzleB = (GLenum)params[2];
            t->swizzleA = (GLenum)params[3];
            break;
        default:
            glTexParameterf(target, pname, (GLfloat)params[0]);
            return;
    }
    ++t->paramsVersion;
    backend_texture_set_params(t->id, t->minFilter, t->magFilter,
                               t->wrapS, t->wrapT, t->wrapR, t->borderColor);
}

void glGenerateMipmap(GLenum target) {
    MITHRIL_ENSURE_INIT();
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    // Compute mip level count if the app never called glTexStorage*(levels=N).
    // glGenerateMipmap is the legacy way to request a full mip chain: the
    // driver allocates log2(max(w,h))+1 levels. Our Vulkan texture was
    // created with whatever level count was last set on the GL object; if
    // it's still 1, the backend's generate_mipmaps becomes a no-op, so the
    // app sees the base level only. This is acceptable for MC Java (its
    // modern pipeline uses glTexStorage2D for mipmapped textures).
    t->generateMipmaps = true;
    backend_generate_mipmaps(t->id);
}

/* ---- Texture parameter queries (P1-4) ----
 * Return the REAL values tracked on the Texture struct (the previous stubs
 * unconditionally wrote 0). All pnames accepted by glTexParameter* are
 * accepted here. The Iiv/Iuiv variants return the integer border color from
 * borderColorI[] / borderColorUI[] respectively.
 */
void glGetTexParameteriv(GLenum target, GLenum pname, GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) { *params = 0; return; }
    switch (pname) {
        case GL_TEXTURE_MIN_FILTER:        *params = t->minFilter; break;
        case GL_TEXTURE_MAG_FILTER:        *params = t->magFilter; break;
        case GL_TEXTURE_WRAP_S:            *params = t->wrapS; break;
        case GL_TEXTURE_WRAP_T:            *params = t->wrapT; break;
        case GL_TEXTURE_WRAP_R:            *params = t->wrapR; break;
        case GL_TEXTURE_BASE_LEVEL:        *params = t->baseLevel; break;
        case GL_TEXTURE_MAX_LEVEL:         *params = t->maxLevel; break;
        case GL_TEXTURE_MIN_LOD:           *params = (GLint)t->minLod; break;
        case GL_TEXTURE_MAX_LOD:           *params = (GLint)t->maxLod; break;
        case GL_TEXTURE_LOD_BIAS:          *params = (GLint)t->lodBias; break;
        case GL_TEXTURE_MAX_ANISOTROPY_EXT:*params = (GLint)t->maxAnisotropy; break;
        case GL_TEXTURE_COMPARE_MODE:      *params = (GLint)t->compareMode; break;
        case GL_TEXTURE_COMPARE_FUNC:      *params = (GLint)t->compareFunc; break;
        case GL_TEXTURE_SWIZZLE_R:         *params = (GLint)t->swizzleR; break;
        case GL_TEXTURE_SWIZZLE_G:         *params = (GLint)t->swizzleG; break;
        case GL_TEXTURE_SWIZZLE_B:         *params = (GLint)t->swizzleB; break;
        case GL_TEXTURE_SWIZZLE_A:         *params = (GLint)t->swizzleA; break;
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; ++i) params[i] = (GLint)t->borderColor[i];
            break;
        case GL_TEXTURE_SWIZZLE_RGBA:
            params[0] = (GLint)t->swizzleR;
            params[1] = (GLint)t->swizzleG;
            params[2] = (GLint)t->swizzleB;
            params[3] = (GLint)t->swizzleA;
            break;
        default:
            mithril::state_set_error(GL_INVALID_ENUM);
            *params = 0;
            break;
    }
}

void glGetTexParameterfv(GLenum target, GLenum pname, GLfloat* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) { *params = 0; return; }
    switch (pname) {
        case GL_TEXTURE_MIN_FILTER:        *params = (GLfloat)t->minFilter; break;
        case GL_TEXTURE_MAG_FILTER:        *params = (GLfloat)t->magFilter; break;
        case GL_TEXTURE_WRAP_S:            *params = (GLfloat)t->wrapS; break;
        case GL_TEXTURE_WRAP_T:            *params = (GLfloat)t->wrapT; break;
        case GL_TEXTURE_WRAP_R:            *params = (GLfloat)t->wrapR; break;
        case GL_TEXTURE_BASE_LEVEL:        *params = (GLfloat)t->baseLevel; break;
        case GL_TEXTURE_MAX_LEVEL:         *params = (GLfloat)t->maxLevel; break;
        case GL_TEXTURE_MIN_LOD:           *params = t->minLod; break;
        case GL_TEXTURE_MAX_LOD:           *params = t->maxLod; break;
        case GL_TEXTURE_LOD_BIAS:          *params = t->lodBias; break;
        case GL_TEXTURE_MAX_ANISOTROPY_EXT:*params = t->maxAnisotropy; break;
        case GL_TEXTURE_COMPARE_MODE:      *params = (GLfloat)t->compareMode; break;
        case GL_TEXTURE_COMPARE_FUNC:      *params = (GLfloat)t->compareFunc; break;
        case GL_TEXTURE_SWIZZLE_R:         *params = (GLfloat)t->swizzleR; break;
        case GL_TEXTURE_SWIZZLE_G:         *params = (GLfloat)t->swizzleG; break;
        case GL_TEXTURE_SWIZZLE_B:         *params = (GLfloat)t->swizzleB; break;
        case GL_TEXTURE_SWIZZLE_A:         *params = (GLfloat)t->swizzleA; break;
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; ++i) params[i] = t->borderColor[i];
            break;
        case GL_TEXTURE_SWIZZLE_RGBA:
            params[0] = (GLfloat)t->swizzleR;
            params[1] = (GLfloat)t->swizzleG;
            params[2] = (GLfloat)t->swizzleB;
            params[3] = (GLfloat)t->swizzleA;
            break;
        default:
            mithril::state_set_error(GL_INVALID_ENUM);
            *params = 0;
            break;
    }
}

void glGetTexParameterIiv(GLenum target, GLenum pname, GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) { *params = 0; return; }
    switch (pname) {
        case GL_TEXTURE_BORDER_COLOR:
            // Integer border color (signed).
            for (int i = 0; i < 4; ++i) params[i] = t->borderColorI[i];
            break;
        case GL_TEXTURE_SWIZZLE_RGBA:
            params[0] = (GLint)t->swizzleR;
            params[1] = (GLint)t->swizzleG;
            params[2] = (GLint)t->swizzleB;
            params[3] = (GLint)t->swizzleA;
            break;
        default: {
            // Fall back to the plain iv query for non-integer-valued pnames.
            GLint iv = 0;
            glGetTexParameteriv(target, pname, &iv);
            *params = iv;
            break;
        }
    }
}

void glGetTexParameterIuiv(GLenum target, GLenum pname, GLuint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params) return;
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) { *params = 0; return; }
    switch (pname) {
        case GL_TEXTURE_BORDER_COLOR:
            // Unsigned integer border color.
            for (int i = 0; i < 4; ++i) params[i] = (GLuint)t->borderColorUI[i];
            break;
        case GL_TEXTURE_SWIZZLE_RGBA:
            params[0] = (GLuint)t->swizzleR;
            params[1] = (GLuint)t->swizzleG;
            params[2] = (GLuint)t->swizzleB;
            params[3] = (GLuint)t->swizzleA;
            break;
        default: {
            GLint iv = 0;
            glGetTexParameteriv(target, pname, &iv);
            *params = (GLuint)iv;
            break;
        }
    }
}

/*
 * glGetTexImage: basic CPU readback from shadow data.
 *
 * The current Texture struct does not shadow pixel data on the CPU side —
 * uploads go straight to the Vulkan VkImage via backend_texture_upload, and
 * no host copy is retained. Real GPU readback would require a
 * backend_read_texture_image() path (vkCmdCopyImageToBuffer on the texture's
 * VkImage) which is not wired up yet. Until that exists this leaves the
 * caller's buffer untouched (matching the previous stub behaviour) rather
 * than returning garbage. When shadow data is added to the Texture struct,
 * the readback path goes here.
 */
void glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void* pixels) {
    MITHRIL_ENSURE_INIT();
    if (!pixels) return;
    mithril::Texture* t = bound_texture_for_target(target);
    if (!t) return;
    (void)level; (void)format; (void)type;
    // TODO: once t->data shadow copy exists (or backend_read_texture_image is
    // added), perform the CPU readback here.
}

GLboolean glIsTexture(GLuint texture) {
    if (!g_state) return GL_FALSE;
    return g_state->textureNames.valid(texture) ? GL_TRUE : GL_FALSE;
}

void glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height,
                  GLenum format, GLenum type, void* pixels) {
    MITHRIL_ENSURE_INIT();
    if (!pixels || width <= 0 || height <= 0) return;
    // Delegate to the backend: it resolves the current colour attachment
    // (EGL default framebuffer or the user FBO's GL_COLOR_ATTACHMENT0),
    // transitions it to TRANSFER_SRC_OPTIMAL, copies into a host-visible
    // staging buffer via vkCmdCopyImageToBuffer, and synchronously maps +
    // memcpy's into the caller's buffer. Returns 0 if readback isn't
    // possible (e.g. no FBO bound to the default framebuffer).
    (void)backend_read_pixels((int)x, (int)y, (int)width, (int)height,
                              format, type, pixels);
}

void glCopyTexImage2D(GLenum target, GLint level, GLenum internalformat,
                      GLint x, GLint y, GLsizei width, GLsizei height, GLint border) {
    MITHRIL_ENSURE_INIT();
    (void)target; (void)level; (void)internalformat;
    (void)x; (void)y; (void)width; (void)height; (void)border;
}

void glCopyTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset,
                         GLint x, GLint y, GLsizei width, GLsizei height) {
    MITHRIL_ENSURE_INIT();
    (void)target; (void)level; (void)xoffset; (void)yoffset;
    (void)x; (void)y; (void)width; (void)height;
}

/* ---- GL 4.3 ARB_copy_image: glCopyImageSubData ----
 *
 * Copies a rectangular region of pixels from srcName to dstName. Both names
 * must refer to textures (not renderbuffers) with compatible internal formats
 * (same component count and size). The copy is pixel-exact (no scaling,
 * no filtering) — maps to Vulkan's vkCmdBlitImage with NEAREST filter and
 * identical src/dst rectangles, or vkCmdCopyImage for same-format copies.
 *
 * Used by Sodium (chunk mesh texture atlas updates) and Iris (shadow map
 * cascade copies). MC 1.21.1 may call this during texture atlas stitching.
 *
 * Implementation: uses backend_blit_images per-Z-slice. For 2D textures
 * (the common case), depth==1 so a single blit is issued. For 3D textures
 * and cube maps, each Z-slice is blitted separately.
 */
void glCopyImageSubData(GLuint srcName, GLenum srcTarget, GLint srcLevel,
                        GLint srcX, GLint srcY, GLint srcZ,
                        GLuint dstName, GLenum dstTarget, GLint dstLevel,
                        GLint dstX, GLint dstY, GLint dstZ,
                        GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth) {
    MITHRIL_ENSURE_INIT();
    (void)srcLevel; (void)dstLevel;  // mip level handled by texture creation

    if (srcWidth <= 0 || srcHeight <= 0 || srcDepth <= 0) return;
    if (srcName == 0 || dstName == 0) {
        mithril::state_set_error(GL_INVALID_VALUE);
        return;
    }

    mithril::Texture* srcTex = mithril::state_get_texture(srcName);
    mithril::Texture* dstTex = mithril::state_get_texture(dstName);
    if (!srcTex || !dstTex) {
        mithril::state_set_error(GL_INVALID_OPERATION);
        return;
    }

    VkImage srcImage = backend_get_texture_image(srcName);
    VkImage dstImage = backend_get_texture_image(dstName);
    if (srcImage == VK_NULL_HANDLE || dstImage == VK_NULL_HANDLE) return;

    VkFormat srcFmt = backend_vk_format_for_gl((GLenum)srcTex->internalFormat);
    VkFormat dstFmt = backend_vk_format_for_gl((GLenum)dstTex->internalFormat);
    if (srcFmt == VK_FORMAT_UNDEFINED) srcFmt = VK_FORMAT_R8G8B8A8_UNORM;
    if (dstFmt == VK_FORMAT_UNDEFINED) dstFmt = VK_FORMAT_R8G8B8A8_UNORM;

    // Flush pending rendering — the blit must see the latest src contents and
    // subsequent draws must see the blit's result.
    backend_end_render_pass();
    backend_commit();

    // Blit each Z-slice. For 2D textures, depth==1 → single iteration.
    // For 3D textures and cube map arrays, blit each slice separately.
    for (GLsizei z = 0; z < srcDepth; ++z) {
        int srcX0 = srcX;
        int srcY0 = srcY + (int)z * srcTex->height;  // linear slice offset (2D fallback)
        int srcX1 = srcX + srcWidth;
        int srcY1 = srcY0 + srcHeight;

        int dstX0 = dstX;
        int dstY0 = dstY + (int)z * dstTex->height;
        int dstX1 = dstX + srcWidth;
        int dstY1 = dstY0 + srcHeight;

        // For true 2D textures, z offset is 0 and we just blit the single slice.
        if (srcDepth == 1) {
            srcY0 = srcY; srcY1 = srcY + srcHeight;
            dstY0 = dstY; dstY1 = dstY + srcHeight;
        }

        backend_blit_images(srcImage, srcFmt, dstImage, dstFmt,
                            srcX0, srcY0, srcX1, srcY1,
                            dstX0, dstY0, dstX1, dstY1,
                            GL_COLOR_BUFFER_BIT, GL_NEAREST,
                            0, dstTex->height);
    }
}

/* ---- GL 4.3 ARB_internalformat_query2: glGetInternalformativ ----
 *
 * Queries implementation-supported properties of internal formats. MC/Sodium
 * uses this to check MSAA sample counts and renderability before allocating
 * renderbuffer storage.
 *
 * Returns reasonable defaults: all common color/depth/stencil formats are
 * supported, 1 sample count (0 = no MSAA) is available, and max dimensions
 * match the device's max texture size.
 */
#ifndef GL_TEXTURE_2D_MULTISAMPLE
#define GL_TEXTURE_2D_MULTISAMPLE 0x9100
#endif

void glGetInternalformativ(GLenum target, GLenum internalformat,
                           GLenum pname, GLsizei bufSize, GLint* params) {
    MITHRIL_ENSURE_INIT();
    if (!params || bufSize <= 0) return;
    (void)target;
    (void)internalformat;

    switch (pname) {
        case 0x826F: // GL_INTERNALFORMAT_SUPPORTED
            *params = GL_TRUE;
            return;
        case 0x9380: // GL_NUM_SAMPLE_COUNTS
            *params = 1;  // we support 1 sample count: 0 (no MSAA)
            return;
        case 0x8CAB: // GL_SAMPLES
            *params = 0;  // 0 = no MSAA
            return;
        case 0x8286: // GL_COLOR_RENDERABLE
        case 0x8287: // GL_DEPTH_RENDERABLE
        case 0x8288: // GL_STENCIL_RENDERABLE
        case 0x8289: // GL_FRAMEBUFFER_RENDERABLE
            *params = GL_TRUE;
            return;
        case 0x827E: // GL_MAX_WIDTH
        case 0x827F: // GL_MAX_HEIGHT
            *params = 16384;  // iOS A-series GPUs support up to 16384
            return;
        case 0x8280: // GL_MAX_DEPTH
            *params = 2048;
            return;
        case 0x8281: // GL_MAX_LAYERS
            *params = 2048;
            return;
        case 0x8293: // GL_MIPMAP
            *params = GL_TRUE;
            return;
        default:
            *params = 0;
            return;
    }
}

/* ---- GL 4.5 ARB_get_texture_sub_image: glGetTextureSubImage ----
 *
 * Reads a sub-region of a texture into client memory. Equivalent to binding
 * the texture to an FBO and calling glReadPixels, but operates directly on
 * the texture name without disturbing the current framebuffer binding.
 *
 * Used by MC's screenshot and debug overlay code.
 */
void glGetTextureSubImage(GLuint texture, GLint level,
                          GLint xoffset, GLint yoffset, GLint zoffset,
                          GLsizei width, GLsizei height, GLsizei depth,
                          GLenum format, GLenum type,
                          GLsizei bufSize, void* pixels) {
    MITHRIL_ENSURE_INIT();
    (void)level; (void)zoffset; (void)depth;

    if (!pixels || bufSize <= 0) return;
    if (width <= 0 || height <= 0) return;
    if (texture == 0) {
        mithril::state_set_error(GL_INVALID_OPERATION);
        return;
    }

    mithril::Texture* tex = mithril::state_get_texture(texture);
    if (!tex) {
        mithril::state_set_error(GL_INVALID_OPERATION);
        return;
    }

    // Save current read FBO + read buffer state.
    GLuint savedReadFBO = g_state->currentReadFBO;
    GLenum savedReadBuffer = GL_COLOR_ATTACHMENT0;
    mithril::Framebuffer* savedFbo = mithril::state_get_framebuffer(savedReadFBO);
    if (savedFbo) savedReadBuffer = savedFbo->readBuffer;

    // Create a temporary FBO, attach the texture, read pixels, restore.
    GLuint tmpFBO = 0;
    glGenFramebuffers(1, &tmpFBO);
    if (tmpFBO == 0) return;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, tmpFBO);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, texture, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    // Read the sub-region.
    glReadPixels(xoffset, yoffset, width, height, format, type, pixels);

    // Restore the original read FBO + read buffer.
    glBindFramebuffer(GL_READ_FRAMEBUFFER, savedReadFBO);
    if (savedReadFBO != 0) {
        glReadBuffer(savedReadBuffer);
    }

    // Delete the temporary FBO.
    glDeleteFramebuffers(1, &tmpFBO);
}

} // extern "C"
