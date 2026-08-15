// Mithril-Wrapper - egl/egl.cpp
// EGL 1.5 cross-platform core implementation backed by Vulkan 1.2.
//
// Platform-specific surface creation (CAMetalLayer coercion on Apple) is
// isolated in egl/Surface<Platform>.cpp/mm, selected by CMake based on the
// APPLE guard. This file stays pure C++ with no platform-specific includes.
//
// This is the layer that Amethyst-iOS' Natives/ctxbridges/gl_bridge.m dlsym's
// against libmithril.dylib. It exposes the 21 egl* entry points listed in
// Amethyst's `egl_library` struct (see Natives/ctxbridges/gl_bridge.h) plus a
// handful of EGL 1.5 helpers (eglCreatePbufferSurface, eglQuerySurface, ...).
//
// Mapping (Vulkan/MoltenVK rewrite of the former Metal-backed egl.mm):
//   EGLDisplay  -> singleton EglDisplay. The Vulkan instance/device live in
//                  the DirectVulkan backend (MG_Backend/DirectVulkan/Device.cpp);
//                  eglInitialize brings them up via backend_init().
//   EGLConfig   -> opaque pointer to one of a small set of pre-baked
//                  EglConfig records (RGBA8 + optional depth/stencil).
//   EGLSurface  -> EglSurface holding a void* native_window + an opaque
//                  swapchain_state pointer (a mithril::vk::Swapchain* created
//                  by backend_create_swapchain()). The swapchain owns the
//                  VkSurfaceKHR (via VK_EXT_metal_surface), VkSwapchainKHR,
//                  swapchain
//                  images/views, and the depth/stencil VkImage/View
//                  (VK_FORMAT_D32_SFLOAT_S8_UINT).
//   EGLContext  -> EglContext holding its own mithril::GLState* (allocated
//                  via state_create()) so multiple contexts do not share GL
//                  object tables. eglMakeCurrent swaps mithril::g_state to
//                  point at the chosen context's state.
//
// The render path:
//   eglMakeCurrent installs the surface's current swapchain image's
//   VkImageView on g_state->eglDefaultColor (and the depth VkImageView on
//   g_state->eglDefaultDepth). GL commands against framebuffer 0 then render
//   straight into the on-screen drawable (see collect_draw_fbo_attachments).
//   eglSwapBuffers flushes Mithril's pending Vulkan work, presents the
//   swapchain image via vkQueuePresentKHR, then acquires the next image for
//   the following frame.

// includes.h lives in MG_Impl/ (sibling of egl/); use a relative path since
// the egl/ directory is not on the include search path and the quote-include
// lookup only checks the current file's directory + -I dirs.
#include "../MG_Impl/includes.h"
#include "../MG_Impl/EGLConfig.h"
#include "../MG_Impl/Log.h"
#include "../MG_Backend/DirectVulkan/Device.h"
#include "../MG_Backend/DirectVulkan/CommandStream.h"  // backend_get_recorded_draws (B1)
#include <EGL/egl.h>

#include "EglInternal.h"   // shared internal handle types + state + swapchain helper decls

// Renderer/version strings built in MG_Impl/Getter.cpp + Getter_gpu.mm. Declared
// here (not in a shared header) so eglMakeCurrent can emit the once-per-process
// startup line mirroring MobileGL's "Using graphics backend/device" log.
#if defined(__APPLE__)
extern "C" const char* mithril_get_gpu_renderer_string(void);
extern "C" const char* mithril_get_version_string(void);
#endif

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

// Platform surface dispatch (surface_create / surface_get_size) is declared in
// EglInternal.h and implemented in egl/Surface<Platform>.cpp/mm (selected by
// CMake on the APPLE guard).

// ---------------------------------------------------------------------------
// Internal handle types
// ---------------------------------------------------------------------------
namespace {

// Bring the extracted config table + matching helpers from
// mithril::egl (see MG_Impl/EGLConfig.h) into this TU's anonymous namespace
// so egl.cpp can keep referring to EglConfig / g_configs / kNumConfigs /
// config_matches / config_get_attr unqualified, exactly as it did before the
// extraction. The internal handle types (EglDisplay / EglSurface / EglContext /
// EglSync / EglImage) now live in mithril::egl (EglInternal.h) and are pulled
// in the same way so the extern "C" entry points keep their unqualified names.
using mithril::egl::EglConfig;
using mithril::egl::EglDisplay;
using mithril::egl::EglSurface;
using mithril::egl::EglContext;
using mithril::egl::EglSync;
using mithril::egl::EglImage;
using mithril::egl::g_configs;
using mithril::egl::kNumConfigs;
using mithril::egl::config_matches;
using mithril::egl::config_get_attr;
using mithril::egl::set_error;
using mithril::egl::clear_error;
using mithril::egl::valid_display;
using mithril::egl::valid_config;
using mithril::egl::ensure_swapchain;
using mithril::egl::install_surface_on_state;
using mithril::egl::swapchain_handle_device_lost;
using mithril::egl::swapchain_poll_completed_frames;
using mithril::egl::swapchain_flush_and_commit;
using mithril::egl::swapchain_present;
using mithril::egl::swapchain_needs_rebuild;
using mithril::egl::swapchain_destroy;

} // namespace

// ---------------------------------------------------------------------------
// Shared EGL state (storage definitions for the extern declarations in
// EglInternal.h). These MUST live in namespace mithril::egl so they satisfy
// the extern declarations there; the anonymous-namespace using-declarations
// below re-expose them to the extern "C" entry points unqualified. (If they
// were left in the anonymous namespace, they would get anonymous-namespace
// internal linkage and the extern "C" entry points + SwapchainHelper.cpp
// would fail to link against mithril::egl::g_display / t_lastError / etc.)
// ---------------------------------------------------------------------------
namespace mithril {
namespace egl {

// Singleton display. Returned for every eglGetDisplay / eglGetPlatformDisplay.
EglDisplay g_display;

// Thread-local EGL current state.
thread_local EglContext* t_currentCtx    = nullptr;
thread_local EglSurface* t_currentDraw   = nullptr;
thread_local EglSurface* t_currentRead   = nullptr;
thread_local EGLint      t_lastError     = EGL_SUCCESS;
thread_local EGLenum     t_boundAPI      = EGL_OPENGL_ES_API;

std::mutex g_ctxMutex; // guards share-group refcount updates

// EGL 1.5 sync/image handle tables (shadow implementations). Handles are
// process-local integers cast to EGLSync/EGLImage; 0 is reserved for
// EGL_NO_SYNC / EGL_NO_IMAGE.
std::unordered_map<EGLSync, EglSync> g_syncs;
uintptr_t g_nextSyncHandle = 1;
std::unordered_map<EGLImage, EglImage> g_images;
uintptr_t g_nextImageHandle = 1;

} // namespace egl
} // namespace mithril

namespace {
using mithril::egl::g_display;
using mithril::egl::t_currentCtx;
using mithril::egl::t_currentDraw;
using mithril::egl::t_currentRead;
using mithril::egl::t_lastError;
using mithril::egl::t_boundAPI;
using mithril::egl::g_ctxMutex;
using mithril::egl::g_syncs;
using mithril::egl::g_nextSyncHandle;
using mithril::egl::g_images;
using mithril::egl::g_nextImageHandle;
} // namespace


// ===========================================================================
// Public EGL entry points (extern "C", exported by libmithril.dylib)
//
// Force default visibility at the source level regardless of the toolchain's
// global visibility policy. leetal/ios-cmake compiles .mm files as OBJCXX with
// -fvisibility=hidden by default; without this pragma the egl* entry points
// would be hidden, never enter the dylib's export table, and host launchers
// (Amethyst-iOS' egl_bridge.m) would see:
//     dlsym(handle, "eglCreateContext"): symbol not found
// followed by a NULL-pointer SIGSEGV in gl_make_current when the unresolved
// pointer is later called. The pragma below overrides hidden visibility so
// every egl* in this block is exported and dlsym-resolvable.
// ===========================================================================
#pragma GCC visibility push(default)
extern "C" {

EGLDisplay eglGetDisplay(EGLNativeDisplayType display_id) {
    clear_error();
    (void)display_id;   // we always return the singleton Vulkan-backed display
    return (EGLDisplay)&g_display;
}

EGLDisplay eglGetPlatformDisplay(EGLenum platform, void* native_display,
                                 const EGLint* attrib_list) {
    clear_error();
    (void)platform; (void)native_display; (void)attrib_list;
    // We are a single-display implementation; any platform token resolves to
    // the Vulkan-backed singleton. EGL_EXT_platform_base callers (Amethyst's
    //eglGetPlatformDisplay path) land here.
    return (EGLDisplay)&g_display;
}

EGLBoolean eglInitialize(EGLDisplay dpy, EGLint* major, EGLint* minor) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    // Bring up the Vulkan backend once. backend_init() is idempotent.
    backend_init();
    if (!backend_available()) {
        set_error(EGL_NOT_INITIALIZED);
        return EGL_FALSE;
    }
    g_display.initialized = true;
    if (major) *major = 1;
    if (minor) *minor = 5;
    return EGL_TRUE;
}

EGLBoolean eglTerminate(EGLDisplay dpy) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    // We do NOT destroy the Vulkan instance/device — the host process may call
    // eglInitialize again, and instance/device creation is expensive. Just
    // mark the display as not-initialized so callers must re-init per spec.
    g_display.initialized = false;
    return EGL_TRUE;
}

const char* eglQueryString(EGLDisplay dpy, EGLint name) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return nullptr; }
    switch (name) {
        case EGL_VENDOR:
            return "Mithril-Wrapper (EGL-on-Vulkan 1.2 / MoltenVK)";
        case EGL_VERSION:
            return "1.5 Mithril-Wrapper (Vulkan 1.2 backend)";
        case EGL_CLIENT_APIS:
            return "OpenGL";   // we expose OpenGL 3.3 Core Profile
        case EGL_EXTENSIONS:
            // Minimal but honest list of what we actually implement.
            return "EGL_EXT_platform_base "
                   "EGL_MESA_platform_surfaceless "
                   "EGL_KHR_swap_buffers_with_damage "
                   "EGL_KHR_fence_sync EGL_KHR_wait_sync EGL_KHR_image "
                   "EGL_KHR_image_base EGL_KHR_create_context "
                   "EGL_KHR_platform_base";
        default:
            set_error(EGL_BAD_PARAMETER);
            return nullptr;
    }
}

EGLBoolean eglBindAPI(EGLenum api) {
    clear_error();
    if (api != EGL_OPENGL_API && api != EGL_OPENGL_ES_API && api != EGL_OPENVG_API) {
        set_error(EGL_BAD_PARAMETER);
        return EGL_FALSE;
    }
    // We always expose OpenGL 3.3 Core Profile, but we accept OpenGL ES
    // requests too — the Mithril GL state machine is API-agnostic at the
    // surface level. Amethyst binds EGL_OPENGL_API for the Metal-ANGLE path
    // and EGL_OPENGL_ES_API for the LTW/GLES path; either works here.
    t_boundAPI = api;
    g_display.boundAPI = api;
    return EGL_TRUE;
}

EGLBoolean eglReleaseThread(void) {
    clear_error();
    // Drop the thread-local current context/surface references.
    t_currentCtx  = nullptr;
    t_currentDraw = nullptr;
    t_currentRead = nullptr;
    return EGL_TRUE;
}

EGLint eglGetError(void) {
    EGLint e = t_lastError;
    t_lastError = EGL_SUCCESS;
    return e;
}

// ---- Configs ----
EGLBoolean eglGetConfigs(EGLDisplay dpy, EGLConfig* configs,
                         EGLint config_size, EGLint* num_config) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    if (!num_config) { set_error(EGL_BAD_PARAMETER); return EGL_FALSE; }
    if (!configs || config_size <= 0) {
        *num_config = kNumConfigs;
        return EGL_TRUE;
    }
    EGLint n = kNumConfigs < config_size ? kNumConfigs : config_size;
    for (EGLint i = 0; i < n; ++i) configs[i] = (EGLConfig)&g_configs[i];
    *num_config = n;
    return EGL_TRUE;
}

EGLBoolean eglChooseConfig(EGLDisplay dpy, const EGLint* attrib_list,
                           EGLConfig* configs, EGLint config_size,
                           EGLint* num_config) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    if (!num_config) { set_error(EGL_BAD_PARAMETER); return EGL_FALSE; }

    EGLint matches[kNumConfigs];
    EGLint n = 0;
    for (int i = 0; i < kNumConfigs; ++i) {
        if (config_matches(&g_configs[i], attrib_list)) {
            matches[n++] = i;
        }
    }
    if (!configs || config_size <= 0) {
        *num_config = n;
        return EGL_TRUE;
    }
    EGLint out = n < config_size ? n : config_size;
    for (EGLint i = 0; i < out; ++i) configs[i] = (EGLConfig)&g_configs[matches[i]];
    *num_config = out;
    return EGL_TRUE;
}

EGLBoolean eglGetConfigAttrib(EGLDisplay dpy, EGLConfig config,
                              EGLint attribute, EGLint* value) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    if (!valid_config(config)) { set_error(EGL_BAD_CONFIG); return EGL_FALSE; }
    if (!value) { set_error(EGL_BAD_PARAMETER); return EGL_FALSE; }
    *value = config_get_attr((EglConfig*)config, attribute);
    return EGL_TRUE;
}

// ---- Surfaces ----
EGLSurface eglCreateWindowSurface(EGLDisplay dpy, EGLConfig config,
                                  EGLNativeWindowType win,
                                  const EGLint* attrib_list) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_NO_SURFACE; }
    if (!valid_config(config)) { set_error(EGL_BAD_CONFIG); return EGL_NO_SURFACE; }
    if (!win) { set_error(EGL_BAD_NATIVE_WINDOW); return EGL_NO_SURFACE; }

    // Platform-specific surface preparation. Returns a void* native_window
    // suitable for backend_create_swapchain (CAMetalLayer* on Apple),
    // or nullptr on failure.
    int w = 0, h = 0;
    void* native_window = surface_create((void*)win, &w, &h);
    if (!native_window) {
        set_error(EGL_BAD_NATIVE_WINDOW);
        return EGL_NO_SURFACE;
    }

    (void)attrib_list; // we ignore render-buffer / post-sub-buffer attribs

    EglSurface* s = new EglSurface{};
    s->native_window = native_window;
    s->config = config;
    EglConfig* cfg = (EglConfig*)config;
    s->wantDepthStencil = (cfg->depthSize > 0 || cfg->stencilSize > 0);
    // Build the Vulkan swapchain now if the window is already sized. If not,
    // defer to eglMakeCurrent / eglSwapBuffers which will retry. This surface
    // is not yet current on any thread, so pass is_current=false.
    if (!ensure_swapchain(s, false)) {
        MITHRIL_LOG_WARN("egl", "eglCreateWindowSurface: deferred swapchain (window size = %dx%d)", w, h);
    }
    return (EGLSurface)s;
}

EGLSurface eglCreatePbufferSurface(EGLDisplay dpy, EGLConfig config,
                                   const EGLint* attrib_list) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_NO_SURFACE; }
    if (!valid_config(config)) { set_error(EGL_BAD_CONFIG); return EGL_NO_SURFACE; }
    (void)attrib_list;
    // PBuffers are not actively used by MC Java; return a no-op surface so
    // EGL probes (LWJGL) succeed. We do not allocate a backing swapchain until
    // the surface is actually rendered to.
    EglSurface* s = new EglSurface{};
    s->config = config;
    return (EGLSurface)s;
}

EGLBoolean eglDestroySurface(EGLDisplay dpy, EGLSurface surface) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    if (surface == EGL_NO_SURFACE) { set_error(EGL_BAD_SURFACE); return EGL_FALSE; }
    EglSurface* s = (EglSurface*)surface;
    // If this surface is current on this thread, we must detach it from the
    // encoder and clear the active state before tearing down the swapchain.
    // swapchain_destroy() then drains GPU work referencing the swapchain and
    // destroys it; without the drain, vkDestroySwapchainKHR frees IOSurfaces
    // that the GPU may still be reading, and the next IOSurfaceBindAccel call
    // in the Metal driver crashes with SIGSEGV (UAF).
    const bool is_current = (t_currentDraw == s);
    if (is_current) {
        t_currentDraw = nullptr;
        install_surface_on_state(nullptr, false);
    }
    if (t_currentRead == s) { t_currentRead = nullptr; }
    swapchain_destroy(s, is_current);
    s->native_window = nullptr;
    delete s;
    return EGL_TRUE;
}

EGLBoolean eglQuerySurface(EGLDisplay dpy, EGLSurface surface,
                           EGLint attribute, EGLint* value) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    EglSurface* s = (EglSurface*)surface;
    if (!s) { set_error(EGL_BAD_SURFACE); return EGL_FALSE; }
    if (!value) { set_error(EGL_BAD_PARAMETER); return EGL_FALSE; }
    switch (attribute) {
        case EGL_WIDTH:           *value = s->width;  break;
        case EGL_HEIGHT:          *value = s->height; break;
        case EGL_CONFIG_ID:
            *value = s->config ? ((EglConfig*)s->config)->configId : 0; break;
        case EGL_RENDER_BUFFER:   *value = EGL_BACK_BUFFER; break;
        case EGL_SWAP_BEHAVIOR:   *value = EGL_BUFFER_DESTROYED; break;
        case EGL_MULTISAMPLE_RESOLVE: *value = EGL_MULTISAMPLE_RESOLVE_DEFAULT; break;
        default:                  *value = 0; break;
    }
    return EGL_TRUE;
}

// ---- Contexts ----
EGLContext eglCreateContext(EGLDisplay dpy, EGLConfig config,
                            EGLContext share_context,
                            const EGLint* attrib_list) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_NO_CONTEXT; }
    if (!valid_config(config)) { set_error(EGL_BAD_CONFIG); return EGL_NO_CONTEXT; }

    EglContext* ctx = new EglContext{};
    ctx->state = mithril::state_create();
    ctx->config = config;
    ctx->clientAPI = t_boundAPI;
    ctx->majorVer = 3;
    ctx->minorVer = 3;

    // Parse context attributes (EGL_CONTEXT_MAJOR_VERSION / _CLIENT_VERSION /
    // _MINOR_VERSION / _FLAGS_KHR / _OPENGL_PROFILE_MASK). We are an OpenGL
    // 3.3 Core Profile implementation, so we honor 3.3 / 4.x requests by
    // clamping to 3.3 (the highest Core Profile version Mithril speaks).
    if (attrib_list) {
        for (const EGLint* a = attrib_list; *a != EGL_NONE; a += 2) {
            EGLint name = a[0], value = a[1];
            if (name == EGL_CONTEXT_MAJOR_VERSION || name == EGL_CONTEXT_CLIENT_VERSION) {
                ctx->majorVer = value;
            } else if (name == EGL_CONTEXT_MINOR_VERSION) {
                ctx->minorVer = value;
            } else if (name == EGL_CONTEXT_OPENGL_PROFILE_MASK) {
                // We always report Core Profile; Compatibility is silently
                // honoured because our entry points don't differ.
            } else if (name == EGL_CONTEXT_FLAGS_KHR) {
                // No-op: we don't expose debug/robustness yet.
            }
        }
    }
    if (ctx->majorVer > 3 || (ctx->majorVer == 3 && ctx->minorVer > 3)) {
        ctx->majorVer = 3; ctx->minorVer = 3;
    }

    if (share_context != EGL_NO_CONTEXT) {
        EglContext* sh = (EglContext*)share_context;
        ctx->share = sh;
        std::lock_guard<std::mutex> lk(g_ctxMutex);
        sh->refcount.fetch_add(1);
    }
    return (EGLContext)ctx;
}

EGLBoolean eglDestroyContext(EGLDisplay dpy, EGLContext ctx) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    EglContext* c = (EglContext*)ctx;
    if (!c || c == (EglContext*)EGL_NO_CONTEXT) {
        set_error(EGL_BAD_CONTEXT); return EGL_FALSE;
    }
    // If this context is current on this thread, detach it first.
    if (t_currentCtx == c) {
        install_surface_on_state(nullptr, false);
        mithril::g_state = nullptr;
        t_currentCtx = nullptr;
        t_currentDraw = nullptr;
        t_currentRead = nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g_ctxMutex);
        if (c->refcount.fetch_sub(1) == 1) {
            mithril::state_destroy(c->state);
            delete c;
        }
    }
    return EGL_TRUE;
}

EGLBoolean eglMakeCurrent(EGLDisplay dpy, EGLSurface draw, EGLSurface read,
                          EGLContext ctx) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }

    // Detach case: ctx == EGL_NO_CONTEXT and draw/read == EGL_NO_SURFACE.
    if (ctx == EGL_NO_CONTEXT) {
        if (draw != EGL_NO_SURFACE || read != EGL_NO_SURFACE) {
            set_error(EGL_BAD_MATCH); return EGL_FALSE;
        }
        install_surface_on_state(nullptr, false);
        mithril::g_state = nullptr;
        t_currentCtx = nullptr;
        t_currentDraw = nullptr;
        t_currentRead = nullptr;
        return EGL_TRUE;
    }

    EglContext* c = (EglContext*)ctx;
    EglSurface* d = (EglSurface*)draw;
    EglSurface* r = (read == draw) ? d : (EglSurface*)read;
    if (!c) { set_error(EGL_BAD_CONTEXT); return EGL_FALSE; }

    // Make sure the Vulkan backend is up before any GL call lands.
    backend_init();

    // Emit the renderer/version identity once per process, at the first
    // make-current (Minecraft's Render thread context is current by the time the
    // launcher's first frame starts). Mirrors MobileGL's startup lines
    // "Using graphics backend OpenGL, using drivers: <version>" and
    // "Using graphics device: <renderer>" so crash logs and the console identify
    // the exact backend + build without digging into F3.
#if defined(__APPLE__)
    static std::once_flag s_rendererLogged;
    std::call_once(s_rendererLogged, [] {
        MITHRIL_LOG_INFO("egl", "Using graphics backend OpenGL, using drivers: %s",
                         mithril_get_version_string());
        MITHRIL_LOG_INFO("egl", "Using graphics device: %s",
                         mithril_get_gpu_renderer_string());
    });
#endif

    // Swap Mithril's global state pointer to this context's state.
    mithril::g_state = c->state;

    // Install the draw surface's swapchain image views on the (now current)
    // GLState so framebuffer-0 rendering lands on the on-screen surface.
    if (d) {
        if (!d->swapchain_state && d->native_window) {
            // First make-current on a freshly-created surface whose initial
            // swapchain creation failed (window wasn't sized yet). Retry now.
            ensure_swapchain(d, true);
        }
        install_surface_on_state(d, true);
        // Initialise the viewport to the surface size if the app hasn't yet.
        // Use g_state->eglDefaultWidth/Height (the actual drawable size, clamped
        // to the swapchain size by install_surface_on_state) instead of d->width
        // (the swapchain creation-time size, which may be stale if the window
        // was resized after swapchain creation).
        if (c->state->viewportW <= 0 || c->state->viewportH <= 0) {
            c->state->viewportX = 0;
            c->state->viewportY = 0;
            c->state->viewportW = g_state->eglDefaultWidth;
            c->state->viewportH = g_state->eglDefaultHeight;
        }
    } else {
        install_surface_on_state(nullptr, false);
    }

    t_currentCtx  = c;
    t_currentDraw = d;
    t_currentRead = r ? r : d;
    return EGL_TRUE;
}

EGLContext eglGetCurrentContext(void) {
    return (EGLContext)t_currentCtx;
}

EGLSurface eglGetCurrentSurface(EGLenum readdraw) {
    if (readdraw == EGL_READ) return (EGLSurface)t_currentRead;
    if (readdraw == EGL_DRAW) return (EGLSurface)t_currentDraw;
    set_error(EGL_BAD_PARAMETER);
    return EGL_NO_SURFACE;
}

EGLDisplay eglGetCurrentDisplay(void) {
    return t_currentCtx ? (EGLDisplay)&g_display : EGL_NO_DISPLAY;
}

EGLBoolean eglQueryContext(EGLDisplay dpy, EGLContext ctx,
                           EGLint attribute, EGLint* value) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    EglContext* c = (EglContext*)ctx;
    if (!c) { set_error(EGL_BAD_CONTEXT); return EGL_FALSE; }
    if (!value) { set_error(EGL_BAD_PARAMETER); return EGL_FALSE; }
    switch (attribute) {
        case EGL_CONFIG_ID:
            *value = c->config ? ((EglConfig*)c->config)->configId : 0; break;
        case EGL_CONTEXT_CLIENT_TYPE:
            *value = (t_boundAPI == EGL_OPENGL_ES_API) ? EGL_OPENGL_ES_API : EGL_OPENGL_API;
            break;
        // EGL_CONTEXT_CLIENT_VERSION and EGL_CONTEXT_MAJOR_VERSION are the
        // same token (0x3098) in the Khronos EGL spec (the latter is the EGL
        // 1.5 rename of the former); a single case label covers both.
        case EGL_CONTEXT_MAJOR_VERSION: *value = c->majorVer; break;
        case EGL_CONTEXT_MINOR_VERSION: *value = c->minorVer; break;
        case EGL_RENDER_BUFFER:         *value = EGL_BACK_BUFFER; break;
        default:                        *value = 0; break;
    }
    return EGL_TRUE;
}

// ---- Swap ----
EGLBoolean eglSwapBuffers(EGLDisplay dpy, EGLSurface surface) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    EglSurface* s = (EglSurface*)surface;
    if (!s) { set_error(EGL_BAD_SURFACE); return EGL_FALSE; }

    // FIX (显存耗尽根因 - 主动式 GC，深度参考 MobileGL):
    // 在每帧渲染开始前，非阻塞轮询所有帧槽位的 fence，对已完成的 slot
    // 立即 drain 其 disposalQueue。这在新帧分配资源前释放已完成帧的
    // staging buffer / 旧纹理 / orphaned buffer，降低显存占用峰值。
    //
    // MobileGL 在 Present() 末尾调用 m_textureManager->BeginFrame() +
    // m_bufferManager.BeginFrame() + m_uniformManager->BeginFrame()，这些
    // BeginFrame 会 CollectAllDeferredReleases 释放已完成帧的延迟资源。
    // 我们没有分层的 texture/buffer manager，所以用统一的
    // backend_poll_completed_frames 在帧边界做同样的事。
    //
    // 关键：这是非阻塞的（vkGetFenceStatus 立即返回），不会 stall 渲染线程。
    // 只有 GPU 真正完成的 slot 才 drain，避免 UAF。
    //
    // 注意：在 deviceLost 检查之前调用，确保即使 deviceLost 恢复路径也能
    // 释放已完成帧的资源（poll 只处理 fencePending=true 且 fence=signaled 的 slot）。
    swapchain_poll_completed_frames();

    // Persistent device-loss recovery, extracted to SwapchainHelper.cpp.
    // If the backend is device-lost, the recovery state machine runs (periodic
    // swapchain rebuild after purging recreatable caches) and returns true; the
    // caller must skip present/commit this frame. If not device-lost it returns
    // false and the normal swap path proceeds.
    if (swapchain_handle_device_lost(s, t_currentDraw == s)) {
        return EGL_TRUE;
    }

    // First-frame / deferred-swapchain retry: if the swapchain wasn't created
    // at eglCreateWindowSurface time (because the native window wasn't sized
    // yet — common on iOS where the CAMetalLayer gets its size asynchronously
    // after the view is laid out), retry now. Without this, the host would
    // call eglSwapBuffers on a surface with no swapchain, present nothing,
    // and the screen would stay black forever (the swapchain would never be
    // created because eglMakeCurrent's retry only fires on the very first
    // make-current). We retry on every swap until the swapchain comes up.
    if (s->native_window && !s->swapchain_state) {
        ensure_swapchain(s, t_currentDraw == s);
        if (s->swapchain_state && t_currentDraw == s) {
            // New swapchain just came up: install it on the current GLState so
            // the next frame's draws land on the on-screen drawable.
            install_surface_on_state(s, true);
        }
    }

    // Flush any pending Vulkan work into the current swapchain image view.
    // swapchain_flush_and_commit() = backend_end_render_pass() + backend_commit():
    // end the active render pass and submit the command buffer, so the encoded
    // draws land on the currently-acquired swapchain image before we present.
    // commit_frame()'s empty-submit defense skips the submit if no commands
    // were recorded since the last commit (e.g. eglWaitClient flushed this frame).
    swapchain_flush_and_commit();

    // Present the frame we just rendered, then acquire the next image for
    // the following frame. backend_present_and_acquire() calls
    // vkQueuePresentKHR followed by vkAcquireNextImageKHR.
    //
    // IMPORTANT: present happens BEFORE the resize/rebuild check below. If we
    // rebuilt the swapchain before presenting, the vkQueuePresentKHR would
    // reference a just-destroyed swapchain (UAF) — exactly the
    // IOSurfaceBindAccel SIGSEGV seen in the field. The correct order is:
    // present the already-rendered frame against the current (still-valid)
    // swapchain, THEN tear down + recreate for the next frame.
    //
    // Present the frame, then acquire the next image for the following frame.
    // swapchain_present() also applies the ZERO-AREA GUARD: if the native window
    // has collapsed to 0x0 (iOS app backgrounded / view minimized), it skips
    // present entirely. Mirrors MobileGL commit 7ab8386: presenting against an
    // out-of-date swapchain from a zero-area window previously let Present
    // submit on an already-signaled fence and present an image that was never
    // properly acquired → black screen with sound. Skipping present here lets
    // the resize/rebuild path below recreate the swapchain at the new (non-zero)
    // size on the next swap.
    swapchain_present(s);

    // B1 first-frame diagnostic: 记录已呈现帧计数 + 本帧 GL clear 颜色。
    // 用于真机确认首帧到底有没有出现过红色/黑色 clear 值，以及红屏是否由
    // glClearColor 驱动。仅前 60 帧输出，避免刷屏。
    if (mithril::g_state) {
        ++mithril::g_state->presentedFrames;
        if (mithril::g_state->presentedFrames <= 60) {
            // B1: log the frame's recorded draw count. draw>0 => draws reached
            // the command buffer but fragments aren't visible (depth/viewport/
            // shader); draw==0 => draws were dropped before recording.
            unsigned int draws = mithril::vk::backend_get_recorded_draws();
            MITHRIL_LOG_WARN("vk-diag", "B1 present frame #%u clearColor=(%.3f %.3f %.3f %.3f) draws=%u",
                             mithril::g_state->presentedFrames,
                             mithril::g_state->clearColor[0],
                             mithril::g_state->clearColor[1],
                             mithril::g_state->clearColor[2],
                             mithril::g_state->clearColor[3],
                             draws);
        }
    }

    // Rebuild the swapchain if (a) the native window was resized between
    // frames, or (b) the backend marked the swapchain dead via
    // backend_swapchain_needs_rebuild (fatal Vulkan error: GPU OOM, surface
    // lost, device lost). Case (b) is the recovery path for the VK_NOT_READY
    // death spiral: without rebuilding, the dead swapchain would keep
    // returning null from acquire and the render thread would spin forever.
    // ensure_swapchain() drains GPU work and detaches the old swapchain from
    // the encoder before destroying it, so this is safe even under OOM.
    if (s->native_window && s->swapchain_state) {
        int w = 0, h = 0;
        bool size_changed = (surface_get_size(s->native_window, &w, &h) &&
                             w > 0 && h > 0 &&
                             (w != s->width || h != s->height));
        bool needs_rebuild = swapchain_needs_rebuild(s);
        if (size_changed || needs_rebuild) {
            if (needs_rebuild) {
                // 限流：同一故障串内最多输出一次。首次故障时 consecutiveSubmitFailures
                // 从 0 自增到 1（≤1 仍输出），之后 ≥2 不再输出，直到计数器被成功提交清零。
                // 这样既保留首次诊断信息，又避免 deviceLost 置位前后的 rebuilding 死循环
                // 刷屏（latestlog.txt 中观察到的 ~3000 行重复日志）。
                mithril::vk::Backend* b = mithril::vk::backend();
                if (!b || b->consecutiveSubmitFailures <= 1) {
                    MITHRIL_LOG_WARN("egl", "eglSwapBuffers: swapchain marked dead by "
                                      "backend, rebuilding (GPU OOM / surface lost)");
                }
            }
            ensure_swapchain(s, t_currentDraw == s);
        }
    }

    // Re-install the (possibly new) swapchain's current image on the GLState
    // so the next frame's draws land on a valid drawable. This also re-
    // registers the swapchain with the encoder (backend_set_active_swapchain)
    // so layout barriers + per-image renderFinished signaling work next frame.
    if (s->swapchain_state && t_currentDraw == s) {
        install_surface_on_state(s, true);
    }
    return EGL_TRUE;
}

EGLBoolean eglSwapInterval(EGLDisplay dpy, EGLint interval) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    if (t_currentDraw) {
        t_currentDraw->swapInterval = interval > 1 ? 1 : (interval < 0 ? 0 : interval);
    }
    return EGL_TRUE;
}

// ---- Idle sync (no-ops; Mithril flushes work synchronously per draw) ----
EGLBoolean eglWaitClient(void)  { swapchain_flush_and_commit(); return EGL_TRUE; }
EGLBoolean eglWaitGL(void)      { swapchain_flush_and_commit(); return EGL_TRUE; }
EGLBoolean eglWaitNative(EGLint) { return EGL_TRUE; }

// ---- Extension function resolution ----
// eglGetProcAddress delegates to glXGetProcAddress which resolves symbols from
// this dylib's export table. LWJGL/GLFW use this to obtain GL function pointers.
// Any GL Core Profile entry point we export is returned; unknown names return
// NULL (per EGL spec).
void (*eglGetProcAddress(const char* procname))(void) {
    clear_error();
    if (!procname) return nullptr;
    // Delegate to glXGetProcAddress (same symbol resolution mechanism).
    extern void* glXGetProcAddress(const char*);
    return (void(*)(void))glXGetProcAddress(procname);
}

// EGL 1.5 surface attribute query (eglQuerySurface extension attributes).
EGLBoolean eglSurfaceAttrib(EGLDisplay dpy, EGLSurface surface,
                            EGLint attribute, EGLint value) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    EglSurface* s = (EglSurface*)surface;
    if (!s) { set_error(EGL_BAD_SURFACE); return EGL_FALSE; }
    (void)attribute; (void)value;
    return EGL_TRUE;
}

EGLBoolean eglBindTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) {
    clear_error();
    (void)dpy; (void)surface; (void)buffer;
    return EGL_TRUE;
}

EGLBoolean eglReleaseTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) {
    clear_error();
    (void)dpy; (void)surface; (void)buffer;
    return EGL_TRUE;
}

EGLBoolean eglCopyBuffers(EGLDisplay dpy, EGLSurface surface, EGLNativePixmapType target) {
    clear_error();
    (void)dpy; (void)surface; (void)target;
    return EGL_TRUE;
}

// ---- EGL 1.5 Sync (shadow implementation) ----
EGLSync eglCreateSync(EGLDisplay dpy, EGLenum type, const EGLAttrib* attrib_list) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_NO_SYNC; }
    if (type != EGL_SYNC_FENCE) { set_error(EGL_BAD_ATTRIBUTE); return EGL_NO_SYNC; }
    (void)attrib_list;  // EGL_SYNC_FENCE ignores attrib_list per spec

    EglSync sync{};
    sync.dpy = dpy;
    sync.type = EGL_SYNC_FENCE;
    sync.condition = EGL_SYNC_PRIOR_COMMANDS_COMPLETE;
    sync.status = EGL_SIGNALED;

    EGLSync handle = reinterpret_cast<EGLSync>(g_nextSyncHandle++);
    g_syncs[handle] = sync;
    return handle;
}

EGLBoolean eglDestroySync(EGLDisplay dpy, EGLSync sync) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    auto it = g_syncs.find(sync);
    if (it == g_syncs.end()) { set_error(EGL_BAD_SYNC_KHR); return EGL_FALSE; }
    g_syncs.erase(it);
    return EGL_TRUE;
}

EGLint eglClientWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags, EGLTime timeout) {
    clear_error();
    (void)flags; (void)timeout;
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    auto it = g_syncs.find(sync);
    if (it == g_syncs.end()) { set_error(EGL_BAD_SYNC_KHR); return EGL_FALSE; }
    // Shadow implementation: always signaled, return immediately.
    return EGL_CONDITION_SATISFIED;
}

EGLBoolean eglWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags) {
    clear_error();
    (void)flags;
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    auto it = g_syncs.find(sync);
    if (it == g_syncs.end()) { set_error(EGL_BAD_SYNC_KHR); return EGL_FALSE; }
    // Shadow implementation: no real GPU-side wait.
    return EGL_TRUE;
}

EGLBoolean eglGetSyncAttrib(EGLDisplay dpy, EGLSync sync, EGLint attribute, EGLAttrib* value) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    auto it = g_syncs.find(sync);
    if (it == g_syncs.end()) { set_error(EGL_BAD_SYNC_KHR); return EGL_FALSE; }
    if (!value) { set_error(EGL_BAD_PARAMETER); return EGL_FALSE; }
    const EglSync& s = it->second;
    switch (attribute) {
        case EGL_SYNC_TYPE:      *value = s.type;      break;
        case EGL_SYNC_STATUS:    *value = s.status;    break;
        case EGL_SYNC_CONDITION: *value = s.condition; break;
        default:                 set_error(EGL_BAD_ATTRIBUTE); return EGL_FALSE;
    }
    return EGL_TRUE;
}

// ---- EGL 1.5 Image (shadow implementation) ----
EGLImage eglCreateImage(EGLDisplay dpy, EGLContext ctx, EGLenum target,
                        EGLClientBuffer buffer, const EGLAttrib* attrib_list) {
    clear_error();
    (void)ctx; (void)attrib_list;
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_NO_IMAGE; }
    // Accept known targets; reject unknown.
    switch (target) {
        case EGL_GL_TEXTURE_2D:
        case EGL_GL_TEXTURE_3D:
        case EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case EGL_GL_RENDERBUFFER:
            break;
        default:
            set_error(EGL_BAD_PARAMETER);
            return EGL_NO_IMAGE;
    }

    EglImage img{};
    img.dpy = dpy;
    img.target = target;
    img.buffer = buffer;

    EGLImage handle = reinterpret_cast<EGLImage>(g_nextImageHandle++);
    g_images[handle] = img;
    return handle;
}

EGLBoolean eglDestroyImage(EGLDisplay dpy, EGLImage image) {
    clear_error();
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_FALSE; }
    auto it = g_images.find(image);
    if (it == g_images.end()) { set_error(EGL_BAD_PARAMETER); return EGL_FALSE; }
    g_images.erase(it);
    return EGL_TRUE;
}

// ---- EGL 1.5 Platform Surface ----
EGLSurface eglCreatePlatformWindowSurface(EGLDisplay dpy, EGLConfig config,
                                          void* native_window,
                                          const EGLAttrib* attrib_list) {
    clear_error();
    (void)attrib_list;

    // Surfaceless / headless mode: return a placeholder surface with no
    // swapchain. Used by EGL_MESA_platform_surfaceless with a null window.
    if (native_window == nullptr) {
        if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_NO_SURFACE; }
        if (!valid_config(config)) { set_error(EGL_BAD_CONFIG); return EGL_NO_SURFACE; }
        EglSurface* s = new EglSurface{};
        s->config = config;
        s->width = 0;
        s->height = 0;
        s->swapchain_state = nullptr;
        s->wantDepthStencil = false;
        s->swapInterval = 0;
        return (EGLSurface)s;
    }

    // Default: delegate to eglCreateWindowSurface (handles platform-specific
    // surface preparation via surface_create()).
    return eglCreateWindowSurface(dpy, config, (EGLNativeWindowType)native_window, nullptr);
}

EGLSurface eglCreatePlatformPixmapSurface(EGLDisplay dpy, EGLConfig config,
                                          void* native_pixmap,
                                          const EGLAttrib* attrib_list) {
    clear_error();
    (void)native_pixmap; (void)attrib_list;
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_NO_SURFACE; }
    if (!valid_config(config)) { set_error(EGL_BAD_CONFIG); return EGL_NO_SURFACE; }
    // Pixmap surfaces are recorded in the state layer only, no swapchain.
    EglSurface* s = new EglSurface{};
    s->config = config;
    s->width = 0;
    s->height = 0;
    s->swapchain_state = nullptr;
    s->wantDepthStencil = false;
    s->swapInterval = 0;
    return (EGLSurface)s;
}

EGLSurface eglCreatePixmapSurface(EGLDisplay dpy, EGLConfig config,
                                  EGLNativePixmapType pixmap,
                                  const EGLint* attrib_list) {
    clear_error();
    (void)pixmap; (void)attrib_list;
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_NO_SURFACE; }
    if (!valid_config(config)) { set_error(EGL_BAD_CONFIG); return EGL_NO_SURFACE; }
    // Pixmap surfaces are recorded in the state layer only, no swapchain.
    EglSurface* s = new EglSurface{};
    s->config = config;
    s->width = 0;
    s->height = 0;
    s->swapchain_state = nullptr;
    s->wantDepthStencil = false;
    s->swapInterval = 0;
    return (EGLSurface)s;
}

EGLSurface eglCreatePbufferFromClientBuffer(EGLDisplay dpy, EGLenum buftype,
                                            EGLClientBuffer buffer, EGLConfig config,
                                            const EGLint* attrib_list) {
    clear_error();
    (void)buffer; (void)config; (void)attrib_list;
    if (!valid_display(dpy)) { set_error(EGL_BAD_DISPLAY); return EGL_NO_SURFACE; }
    // OpenVG is not supported.
    if (buftype == EGL_OPENVG_IMAGE) { set_error(EGL_BAD_MATCH); return EGL_NO_SURFACE; }
    set_error(EGL_BAD_PARAMETER);
    return EGL_NO_SURFACE;
}

EGLenum eglQueryAPI(void) {
    return t_boundAPI;  // defaults to EGL_OPENGL_ES_API per EGL 1.5 spec
}

} // extern "C"
#pragma GCC visibility pop
