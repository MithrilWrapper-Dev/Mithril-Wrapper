// Mithril-Wrapper - egl/EglInternal.h
// Internal (non-exported) EGL state types + shared declarations for the
// modularised EGL layer.
//
// The former monolithic egl.cpp (EGL 1.5 core + Vulkan swapchain glue, ~1200
// lines) is split by responsibility so the EGL entry-point layer no longer
// carries the Vulkan/swapchain details inline:
//   * egl/egl.cpp            - extern "C" EGL entry points (the dispatch layer).
//                              Defines the process-wide + thread-local EGL state
//                              declared below, and calls the swapchain helper.
//   * egl/SwapchainHelper.cpp - Vulkan swapchain lifecycle (ensure_swapchain /
//                              install_surface_on_state). The only module that
//                              touches Vk* types / g_state's Vulkan fields.
//
// This header is compiled into the libmithril build but exposes NO exported
// symbols; it exists purely to share the internal handle types + state between
// the dispatch layer and the implementation modules. The EGL public API
// contract (the egl* entry points in egl.cpp) is unchanged by this split.
#ifndef MITHRIL_EGLINTERNAL_H
#define MITHRIL_EGLINTERNAL_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <EGL/egl.h>
#include "../MG_Impl/EGLConfig.h"
#include "../MG_State/State.h"   // mithril::GLState

// ---------------------------------------------------------------------------
// Platform surface dispatch (defined in egl/Surface<Platform>.cpp/mm, selected
// by CMake on the APPLE guard). Declared in the GLOBAL namespace (as the
// implementations in SurfaceMetal.mm are global extern "C") so both the EGL
// dispatch layer (egl.cpp) and SwapchainHelper.cpp can call them unqualified.
// ---------------------------------------------------------------------------
// surface_create() prepares the native window for use as a Vulkan surface and
// returns a void* native_window suitable for backend_create_swapchain():
//   - Apple:   CAMetalLayer* (after CALayer -> CAMetalLayer coercion)
// Returns nullptr on failure. out_w / out_h receive the window's current size
// (0 if undetermined).
//
// surface_get_size() queries the current size of a native_window previously
// returned by surface_create(). Returns false if the window is invalid or the
// size cannot be determined.
extern "C" void* surface_create(void* native_window, int* out_w, int* out_h);
extern "C" bool  surface_get_size(void* native_window, int* out_w, int* out_h);
extern "C" void  surface_destroy(void* native_window);

namespace mithril {
namespace egl {

// ---------------------------------------------------------------------------
// Internal handle types
// ---------------------------------------------------------------------------

// Singleton display record. A single Vulkan-backed EGLDisplay is returned for
// every eglGetDisplay / eglGetPlatformDisplay call.
struct EglDisplay {
    bool      initialized = false;
    EGLenum   boundAPI   = EGL_OPENGL_API;
};

// EGL surface. Holds the (host-owned, weak) native window plus an opaque
// swapchain_state pointer. The swapchain itself is owned by the backend
// (mithril::vk::Swapchain*) and created/destroyed via the swapchain helper.
struct EglSurface {
    void*         native_window    = nullptr;  // CAMetalLayer* (weak ref; owned by host)
    void*         swapchain_state  = nullptr;  // mithril::vk::Swapchain*
    EGLConfig     config           = nullptr;
    EGLint        width            = 0;
    EGLint        height           = 0;
    EGLint        swapInterval     = 1;
    bool          wantDepthStencil = false;
    // FIX (swapchain 重建死循环): 退避计数器。ensure_swapchain 失败后，
    // swapchainRetryBackoff 倒计时到 0 才允许下次重试，避免每帧重试刷屏。
    // swapchainRetryCount 记录连续失败次数，用于日志限流。
    int           swapchainRetryBackoff = 0;
    int           swapchainRetryCount   = 0;
};

// EGL context. Holds its own mithril::GLState* so multiple contexts do not
// share GL object tables. eglMakeCurrent swaps mithril::g_state to point at
// the chosen context's state.
struct EglContext {
    mithril::GLState*   state      = nullptr;
    EGLConfig           config     = nullptr;
    EglContext*         share      = nullptr;
    EGLenum             clientAPI  = EGL_OPENGL_API;
    EGLint              majorVer   = 3;   // we report OpenGL 3.3 Core Profile
    EGLint              minorVer   = 3;
    bool                lost       = false;
    std::atomic<int>    refcount{1};
};

// EGL 1.5 fence sync. The object records the DirectVulkan queue-submit
// serial containing all client commands that preceded eglCreateSync.
struct EglSync {
    EGLDisplay dpy       = EGL_NO_DISPLAY;
    EGLenum    type      = 0;
    EGLenum    condition = 0;
    EGLenum    status    = EGL_UNSIGNALED;
    uint64_t   submitSerial = 0;
};

// EGL 1.5 image object (shadow implementation: records target + buffer only,
// no real VkImage import). Real interop will land with the Vulkan Image bind.
struct EglImage {
    EGLDisplay      dpy    = EGL_NO_DISPLAY;
    EGLenum         target = 0;
    EGLClientBuffer buffer = nullptr;
};

// ---------------------------------------------------------------------------
// Shared process-wide / thread-local EGL state
//
// Defined (given storage) in egl.cpp, declared extern here so the split-out
// implementation modules can reference them. Thread-local state mirrors
// Khronos EGL semantics.
// ---------------------------------------------------------------------------

// Singleton display. Returned for every eglGetDisplay / eglGetPlatformDisplay.
extern EglDisplay g_display;

// Thread-local EGL current state.
extern thread_local EglContext* t_currentCtx;
extern thread_local EglSurface* t_currentDraw;
extern thread_local EglSurface* t_currentRead;
extern thread_local EGLint      t_lastError;
extern thread_local EGLenum     t_boundAPI;

// Guards share-group refcount updates.
extern std::mutex g_ctxMutex;

// EGL 1.5 sync/image handle tables (shadow implementations). Handles are
// process-local integers cast to EGLSync/EGLImage; 0 is reserved for
// EGL_NO_SYNC / EGL_NO_IMAGE.
extern std::unordered_map<EGLSync, EglSync> g_syncs;
extern uintptr_t g_nextSyncHandle;
extern std::unordered_map<EGLImage, EglImage> g_images;
extern uintptr_t g_nextImageHandle;

// ---------------------------------------------------------------------------
// Error helpers (used by the dispatch layer; cheap inline).
// ---------------------------------------------------------------------------
inline void set_error(EGLint e) { if (t_lastError == EGL_SUCCESS) t_lastError = e; }
inline void clear_error()       { t_lastError = EGL_SUCCESS; }

inline bool valid_display(EGLDisplay d) {
    return d == (EGLDisplay)&g_display;
}
inline bool valid_config(EGLConfig c) {
    if (!c) return false;
    for (int i = 0; i < kNumConfigs; ++i) {
        if ((EGLConfig)&g_configs[i] == c) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Swapchain lifecycle helpers (implemented in SwapchainHelper.cpp). These are
// the only functions that touch Vulkan/swapchain details; the EGL dispatch
// layer calls them without knowing Vk* types.
// ---------------------------------------------------------------------------
// Build (or rebuild) the per-surface Vulkan swapchain against the native
// window. Returns true on success. On rebuild (size changed) it drains GPU
// work referencing the old swapchain before tearing it down (prevents
// IOSurfaceBindAccel UAF). `is_current` must be true when this surface is the
// thread's current draw surface (so the swapchain can be drained/detached
// from the encoder before destroy).
bool ensure_swapchain(EglSurface* s, bool is_current);

// Push the surface's current swapchain image views into the active GLState so
// framebuffer-0 rendering lands on the on-screen drawable. Acquires the next
// swapchain image if none is currently acquired. `is_current` must be true
// when this surface is the thread's current draw surface.
void install_surface_on_state(EglSurface* s, bool is_current);

// Handle the persistent-device-loss guard for a frame. If the backend is not
// device-lost, returns false immediately (caller proceeds with the normal
// swap path). If it IS device-lost, runs the recovery state machine (periodic
// swapchain rebuild attempt with backoff / give-up after repeated failures)
// and returns true — the caller must skip present/commit for this frame (the
// original eglSwapBuffers returned EGL_TRUE in this case).
bool swapchain_handle_device_lost(EglSurface* s, bool is_current);

// Frame-boundary GC: non-blocking poll of every in-flight frame slot's fence;
// drains the deferred-release queue for slots whose fence has signaled. Call
// once per swap BEFORE allocating new frame resources. Mirrors MobileGL's
// BeginFrame() CollectAllDeferredReleases.
void swapchain_poll_completed_frames();

// End the active render pass and submit the command buffer (a no-op when no
// commands were recorded since the last commit). Used by eglSwapBuffers before
// present, and by eglWaitClient / eglWaitGL.
void swapchain_flush_and_commit();

// Present the current swapchain image (if one is acquired and the surface is
// non-zero-area), then acquire the next image for the following frame. The
// caller must have flushed via swapchain_flush_and_commit() first.
void swapchain_present(EglSurface* s);

// Re-sync the GLState's default-framebuffer handles to the swapchain image that
// is CURRENTLY acquired. Must be called after swapchain_present() (which
// presents one index and acquires the next), whenever the surface is the thread's
// current draw surface.
//
// Why this exists: install_surface_on_state() is only invoked when the swapchain
// is created/rebuilt, so it captured the image view acquired at that moment and
// cached it in g_state->eglDefaultColor. Every subsequent frame presents that
// index and acquires a DIFFERENT one, but g_state->eglDefaultColor was never
// updated — so the next frame's draws were recorded into the image that was
// just presented while the presentation engine displayed the freshly acquired,
// never-rendered one. With N images that shows a stale frame once every N
// presents and black on all the others, on every platform.
void swapchain_refresh_default_views(EglSurface* s);

// Query whether the backend has marked the swapchain dead (fatal Vulkan error:
// GPU OOM / surface lost / device lost from acquire/present/submit). The caller
// (eglSwapBuffers) rebuilds the swapchain when this returns true.
bool swapchain_needs_rebuild(EglSurface* s);

// Destroy the surface's swapchain safely: if `is_current`, drain in-flight GPU
// work that references the swapchain and detach it from the encoder BEFORE
// destroying (prevents IOSurfaceBindAccel UAF). Sets s->swapchain_state = null.
void swapchain_destroy(EglSurface* s, bool is_current);

} // namespace egl
} // namespace mithril

#endif // MITHRIL_EGLINTERNAL_H