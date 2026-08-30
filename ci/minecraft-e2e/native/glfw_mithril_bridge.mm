#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <EGL/egl.h>
#include <GL/gl.h>

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

extern "C" void mithril_e2e_capture_before_present(int width, int height, void* mithril_handle);

namespace {
struct ContextState {
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
};

struct MithrilApi {
    void* handle = nullptr;
    decltype(&eglGetDisplay) getDisplay = nullptr;
    decltype(&eglInitialize) initialize = nullptr;
    decltype(&eglChooseConfig) chooseConfig = nullptr;
    decltype(&eglBindAPI) bindAPI = nullptr;
    decltype(&eglCreateWindowSurface) createWindowSurface = nullptr;
    decltype(&eglCreateContext) createContext = nullptr;
    decltype(&eglMakeCurrent) makeCurrent = nullptr;
    decltype(&eglSwapBuffers) swapBuffers = nullptr;
    decltype(&eglSwapInterval) swapInterval = nullptr;
    decltype(&eglDestroySurface) destroySurface = nullptr;
    decltype(&eglDestroyContext) destroyContext = nullptr;
    const GLubyte* (*getString)(GLenum) = nullptr;
};

std::mutex g_mutex;
std::unordered_map<GLFWwindow*, ContextState> g_contexts;
thread_local GLFWwindow* g_current = nullptr;
std::atomic<unsigned long long> g_context_count{0};
std::atomic<unsigned long long> g_swap_count{0};
std::atomic<unsigned long long> g_proc_count{0};
std::mutex g_identity_mutex;
std::string g_identity_vendor;
std::string g_identity_renderer;
std::string g_identity_version;

std::string getenv_string(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

std::string json_escape(const char* s) {
    if (!s) return "";
    std::string out;
    for (; *s; ++s) {
        switch (*s) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += *s; break;
        }
    }
    return out;
}

void* delegate_handle() {
    static void* handle = [] {
        std::string path = getenv_string("MITHRIL_E2E_GLFW_DELEGATE");
        void* h = nullptr;
        if (!path.empty()) h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) h = dlopen("libglfw.3.dylib", RTLD_NOW | RTLD_LOCAL);
        if (!h) std::fprintf(stderr, "[mithril-e2e] failed to load delegate GLFW: %s\n", dlerror());
        return h;
    }();
    return handle;
}

template <typename T>
T real_glfw(const char* name) {
    void* h = delegate_handle();
    return h ? reinterpret_cast<T>(dlsym(h, name)) : nullptr;
}

MithrilApi& mithril() {
    static MithrilApi api = [] {
        MithrilApi m;
        std::string path = getenv_string("MITHRIL_E2E_MITHRIL_DYLIB");
        if (path.empty()) {
            std::fprintf(stderr, "[mithril-e2e] MITHRIL_E2E_MITHRIL_DYLIB is unset\n");
            return m;
        }
        m.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!m.handle) {
            std::fprintf(stderr, "[mithril-e2e] dlopen(%s) failed: %s\n", path.c_str(), dlerror());
            return m;
        }
#define LOAD_EGL(field, name) m.field = reinterpret_cast<decltype(m.field)>(dlsym(m.handle, #name))
        LOAD_EGL(getDisplay, eglGetDisplay);
        LOAD_EGL(initialize, eglInitialize);
        LOAD_EGL(chooseConfig, eglChooseConfig);
        LOAD_EGL(bindAPI, eglBindAPI);
        LOAD_EGL(createWindowSurface, eglCreateWindowSurface);
        LOAD_EGL(createContext, eglCreateContext);
        LOAD_EGL(makeCurrent, eglMakeCurrent);
        LOAD_EGL(swapBuffers, eglSwapBuffers);
        LOAD_EGL(swapInterval, eglSwapInterval);
        LOAD_EGL(destroySurface, eglDestroySurface);
        LOAD_EGL(destroyContext, eglDestroyContext);
#undef LOAD_EGL
        m.getString = reinterpret_cast<decltype(m.getString)>(dlsym(m.handle, "glGetString"));
        return m;
    }();
    return api;
}

bool mithril_ready() {
    auto& m = mithril();
    return m.handle && m.getDisplay && m.initialize && m.chooseConfig && m.bindAPI &&
           m.createWindowSurface && m.createContext && m.makeCurrent && m.swapBuffers &&
           m.swapInterval && m.destroySurface && m.destroyContext && m.getString;
}

const char* loaded_mithril_path() {
    Dl_info info{};
    auto& m = mithril();
    if (m.getDisplay && dladdr(reinterpret_cast<void*>(m.getDisplay), &info) && info.dli_fname) {
        return info.dli_fname;
    }
    return "";
}

void write_state(const char* stage) {
    std::string root = getenv_string("MITHRIL_E2E_ROOT");
    if (root.empty()) return;
    std::string path = root + "/bridge-state.json";
    std::string tmp = path + ".tmp";
    std::string vendor;
    std::string renderer;
    std::string version;
    auto& m = mithril();
    {
        std::lock_guard<std::mutex> identityLock(g_identity_mutex);
        if (g_current && m.getString) {
            const char* v = reinterpret_cast<const char*>(m.getString(GL_VENDOR));
            const char* r = reinterpret_cast<const char*>(m.getString(GL_RENDERER));
            const char* ver = reinterpret_cast<const char*>(m.getString(GL_VERSION));
            if (v && *v) g_identity_vendor = v;
            if (r && *r) g_identity_renderer = r;
            if (ver && *ver) g_identity_version = ver;
        }
        /* Keep the last authoritative identity after glfwMakeContextCurrent(NULL)
         * and glfwDestroyWindow().  The post-process oracle runs after JVM exit. */
        vendor = g_identity_vendor;
        renderer = g_identity_renderer;
        version = g_identity_version;
    }
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) return;
    std::fprintf(f,
        "{\n"
        "  \"schema_version\": \"1.0\",\n"
        "  \"stage\": \"%s\",\n"
        "  \"mithril_path\": \"%s\",\n"
        "  \"context_count\": %llu,\n"
        "  \"swap_count\": %llu,\n"
        "  \"get_proc_address_count\": %llu,\n"
        "  \"current_context\": %s,\n"
        "  \"gl_vendor\": \"%s\",\n"
        "  \"gl_renderer\": \"%s\",\n"
        "  \"gl_version\": \"%s\"\n"
        "}\n",
        json_escape(stage).c_str(), json_escape(loaded_mithril_path()).c_str(),
        g_context_count.load(), g_swap_count.load(), g_proc_count.load(),
        g_current ? "true" : "false",
        json_escape(vendor.c_str()).c_str(), json_escape(renderer.c_str()).c_str(), json_escape(version.c_str()).c_str());
    std::fclose(f);
    std::rename(tmp.c_str(), path.c_str());
}

void emit_event(const char* event, const char* message) {
    std::string root = getenv_string("MITHRIL_E2E_ROOT");
    if (root.empty()) return;
    FILE* f = std::fopen((root + "/native-events.jsonl").c_str(), "a");
    if (!f) return;
    std::fprintf(f,
        "{\"schema_version\":\"1.0\",\"producer\":\"glfw-mithril-bridge\","
        "\"event\":\"%s\",\"message\":\"%s\",\"mithril_path\":\"%s\"}\n",
        json_escape(event).c_str(), json_escape(message).c_str(),
        json_escape(loaded_mithril_path()).c_str());
    std::fclose(f);
}

bool ignored_context_hint(int hint) {
    switch (hint) {
        case GLFW_CLIENT_API:
        case GLFW_CONTEXT_VERSION_MAJOR:
        case GLFW_CONTEXT_VERSION_MINOR:
        case GLFW_CONTEXT_ROBUSTNESS:
        case GLFW_OPENGL_FORWARD_COMPAT:
        case GLFW_OPENGL_DEBUG_CONTEXT:
        case GLFW_OPENGL_PROFILE:
        case GLFW_CONTEXT_RELEASE_BEHAVIOR:
        case GLFW_CONTEXT_NO_ERROR:
        case GLFW_CONTEXT_CREATION_API:
            return true;
        default:
            return false;
    }
}
} // namespace

extern "C" {

void glfwWindowHint(int hint, int value) {
    using Fn = void (*)(int, int);
    Fn fn = real_glfw<Fn>("glfwWindowHint");
    if (!fn) return;
    if (ignored_context_hint(hint)) {
        if (hint == GLFW_CLIENT_API) fn(GLFW_CLIENT_API, GLFW_NO_API);
        return;
    }
    fn(hint, value);
}

GLFWwindow* glfwCreateWindow(int width, int height, const char* title,
                             GLFWmonitor* monitor, GLFWwindow* share) {
    using CreateFn = GLFWwindow* (*)(int, int, const char*, GLFWmonitor*, GLFWwindow*);
    using HintFn = void (*)(int, int);
    using CocoaFn = id (*)(GLFWwindow*);
    CreateFn create = real_glfw<CreateFn>("glfwCreateWindow");
    HintFn hint = real_glfw<HintFn>("glfwWindowHint");
    CocoaFn getCocoa = real_glfw<CocoaFn>("glfwGetCocoaWindow");
    if (!create || !hint || !getCocoa || !mithril_ready()) {
        emit_event("bridge_error", "missing delegate GLFW or Mithril EGL symbols");
        return nullptr;
    }
    hint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = create(width, height, title, monitor, nullptr);
    if (!window) {
        emit_event("bridge_error", "delegate GLFW could not create NO_API Cocoa window");
        return nullptr;
    }

    NSWindow* cocoa = (NSWindow*)getCocoa(window);
    NSView* view = cocoa.contentView;
    view.wantsLayer = YES;
    [view layoutSubtreeIfNeeded];
    if (!view.layer) {
        auto destroy = real_glfw<void (*)(GLFWwindow*)>("glfwDestroyWindow");
        if (destroy) destroy(window);
        emit_event("bridge_error", "Cocoa content view has no backing CALayer");
        return nullptr;
    }
    view.layer.contentsScale = cocoa.backingScaleFactor > 0.0 ? cocoa.backingScaleFactor : 1.0;

    auto& m = mithril();
    EGLDisplay display = m.getDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major = 0, minor = 0;
    if (display == EGL_NO_DISPLAY || m.initialize(display, &major, &minor) != EGL_TRUE ||
        m.bindAPI(EGL_OPENGL_API) != EGL_TRUE) {
        emit_event("bridge_error", "Mithril EGL display initialization failed");
        return nullptr;
    }
    const EGLint configAttribs[] = {
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };
    EGLConfig config = nullptr;
    EGLint count = 0;
    if (m.chooseConfig(display, configAttribs, &config, 1, &count) != EGL_TRUE ||
        count != 1 || !config) {
        emit_event("bridge_error", "Mithril EGL config selection failed");
        return nullptr;
    }
    EGLSurface surface = m.createWindowSurface(display, config,
                                                (__bridge void*)view.layer, nullptr);
    if (surface == EGL_NO_SURFACE) {
        emit_event("bridge_error", "Mithril EGL window surface creation failed");
        return nullptr;
    }
    const EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 6,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    EGLContext shareContext = EGL_NO_CONTEXT;
    if (share) {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_contexts.find(share);
        if (it != g_contexts.end()) shareContext = it->second.context;
    }
    EGLContext context = m.createContext(display, config, shareContext, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
        m.destroySurface(display, surface);
        emit_event("bridge_error", "Mithril EGL context creation failed");
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_contexts[window] = ContextState{display, context, surface};
    }
    g_context_count.fetch_add(1);
    emit_event("glfw_window_created", "NO_API Cocoa window bridged to Mithril EGL/CAMetalLayer");
    write_state("window_created");
    return window;
}

void glfwDestroyWindow(GLFWwindow* window) {
    ContextState state{};
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_contexts.find(window);
        if (it != g_contexts.end()) {
            state = it->second;
            g_contexts.erase(it);
            found = true;
        }
    }
    auto& m = mithril();
    if (found) {
        if (g_current == window) {
            m.makeCurrent(state.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            g_current = nullptr;
        }
        m.destroySurface(state.display, state.surface);
        m.destroyContext(state.display, state.context);
    }
    write_state("window_destroyed");
    auto destroy = real_glfw<void (*)(GLFWwindow*)>("glfwDestroyWindow");
    if (destroy) destroy(window);
}

void glfwMakeContextCurrent(GLFWwindow* window) {
    auto& m = mithril();
    if (!window) {
        if (g_current) {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto it = g_contexts.find(g_current);
            if (it != g_contexts.end())
                m.makeCurrent(it->second.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
        g_current = nullptr;
        write_state("context_detached");
        return;
    }
    ContextState state{};
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_contexts.find(window);
        if (it == g_contexts.end()) return;
        state = it->second;
    }
    if (m.makeCurrent(state.display, state.surface, state.surface, state.context) == EGL_TRUE) {
        g_current = window;
        emit_event("context_current", "Mithril EGL context made current for Minecraft GLFW window");
        write_state("context_current");
    } else {
        emit_event("bridge_error", "eglMakeCurrent failed");
    }
}

GLFWwindow* glfwGetCurrentContext(void) {
    return g_current;
}

void glfwSwapBuffers(GLFWwindow* window) {
    ContextState state{};
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_contexts.find(window);
        if (it == g_contexts.end()) return;
        state = it->second;
    }

    // This is the authoritative L4 capture seam: the frame is fully encoded by
    // Minecraft but eglSwapBuffers has not yet presented it or acquired the
    // next drawable. The helper is a no-op unless the Client GameTest has
    // atomically posted a one-shot capture request.
    auto getFramebufferSize = real_glfw<void (*)(GLFWwindow*, int*, int*)>("glfwGetFramebufferSize");
    if (getFramebufferSize) {
        int width = 0, height = 0;
        getFramebufferSize(window, &width, &height);
        mithril_e2e_capture_before_present(width, height, mithril().handle);
    }

    if (mithril().swapBuffers(state.display, state.surface) == EGL_TRUE) {
        auto count = g_swap_count.fetch_add(1) + 1;
        if (count <= 3 || count % 120 == 0) write_state("swap_buffers");
    } else {
        emit_event("bridge_error", "eglSwapBuffers failed");
        write_state("swap_failed");
    }
}

void glfwSwapInterval(int interval) {
    if (!g_current) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_contexts.find(g_current);
    if (it != g_contexts.end()) mithril().swapInterval(it->second.display, interval);
}

GLFWglproc glfwGetProcAddress(const char* procname) {
    if (!procname || !mithril().handle) return nullptr;
    g_proc_count.fetch_add(1);
    void* address = dlsym(mithril().handle, procname);
    return reinterpret_cast<GLFWglproc>(address);
}

int glfwGetWindowAttrib(GLFWwindow* window, int attrib) {
    switch (attrib) {
        case GLFW_CLIENT_API: return GLFW_OPENGL_API;
        case GLFW_CONTEXT_VERSION_MAJOR: return 4;
        case GLFW_CONTEXT_VERSION_MINOR: return 6;
        case GLFW_OPENGL_PROFILE: return GLFW_OPENGL_CORE_PROFILE;
        case GLFW_OPENGL_FORWARD_COMPAT: return GLFW_TRUE;
        case GLFW_CONTEXT_CREATION_API: return GLFW_NATIVE_CONTEXT_API;
        default: break;
    }
    auto fn = real_glfw<int (*)(GLFWwindow*, int)>("glfwGetWindowAttrib");
    return fn ? fn(window, attrib) : 0;
}

} // extern C
