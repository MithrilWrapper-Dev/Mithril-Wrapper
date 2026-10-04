// Mithril-Wrapper - MG_Backend/DirectVulkan/VulkanDispatchAndroid.cpp
// GENERATED FILE - do not edit by hand.
//
// Only the entrypoints Mithril actually calls are forwarded here, to keep the
// header surface small. If a new vk* call is introduced the link will fail
// with an undefined symbol and this file needs regenerating.
//
// Android-only Vulkan dispatch shim.
//
// Why: on Android we used to link the platform loader (libvulkan.so) directly.
// That hard-binds every vk* call to whatever driver the platform loader
// discovers. The Android loader finds drivers through hw_get_module only - it
// never reads VK_ICD_FILENAMES or VK_DRIVER_FILES (Khronos
// LoaderDriverInterface, "Driver Discovery on Android": "The Android loader
// lives in the system library folder. The location cannot be changed... Due to
// security policies in Android, none of this can be modified under normal
// use"). Devices whose stock driver is stuck at Vulkan 1.1 therefore have no
// way to reach an out-of-tree driver such as Turnip.
//
// By dlopen-ing the driver ourselves we choose it at runtime. Each vk* symbol
// below is a thin forwarder that lazily resolves the real entry point from the
// chosen library, so the rest of the codebase keeps calling vkCreateInstance
// etc. directly and needs no changes.
//
// Selection order (first match wins):
//   MITHRIL_VULKAN_LIBRARY  - explicit path to a driver .so
//   MITHRIL_TURNIP=1        - libvulkan_freedreno.so (Turnip / freedreno ICD)
//   otherwise               - libvulkan.so, the platform loader

#if defined(__ANDROID__)

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>

#include <android/dlext.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <setjmp.h>
#include <signal.h>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

// Declared here rather than including <vulkan/vk_icd.h>: that header is part of
// the loader's build interface and is not shipped by the NDK, so relying on it
// breaks the Android build. The signature is fixed by the loader/ICD contract.
typedef PFN_vkVoidFunction (VKAPI_PTR *PFN_vk_icdGetInstanceProcAddr)(VkInstance instance,
                                                                     const char* pName);

namespace {

void* g_handle = nullptr;
PFN_vkGetInstanceProcAddr g_gipa = nullptr;
PFN_vk_icdGetInstanceProcAddr g_icd_gipa = nullptr;
const char* g_driver_path = nullptr;
void* g_hal_device = nullptr;
void* g_fallback_handle = nullptr;
sigjmp_buf g_probe_jmp;
bool g_probe_active = false;
struct sigaction g_old_segv, g_old_bus, g_old_ill;
VkResult (*g_hal_create_instance)(const VkInstanceCreateInfo*, const VkAllocationCallbacks*,
                                  VkInstance*) = nullptr;
PFN_vkGetDeviceProcAddr g_gdpa = nullptr;
VkInstance g_instance = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
bool g_ready = false;
int g_failures = 0;

// ---------------------------------------------------------------------------
// Linker namespace escape
//
// Android N+ loads an app's native libraries into an isolated linker
// namespace. android_namespace_t::is_accessible() only allows paths under that
// namespace's ld_library_paths/default_library_paths/permitted_paths, so a plain
// dlopen() of another package's native library - which is where a Turnip driver
// plugin keeps libvulkan_freedreno.so - fails with "not accessible for the
// namespace". That is why the driver never loaded here.
//
// The fix is what libadrenotools/liblinkernsbypass does: call the linker's
// internal __loader_android_create_namespace() and pass an address inside the
// linker (here, &dlopen) as the caller. The linker then hands back a namespace
// it considers unrestricted, and we can load from it.
//
// Only the internal entrypoint has a caller parameter; the public
// android_create_namespace() always passes the real return address, which is
// inside our own restricted namespace.
// ---------------------------------------------------------------------------

// Not exposed by the NDK; matches bionic's linker.h.
enum { MITHRIL_NS_TYPE_SHARED = 2 };

typedef struct android_namespace_t* (*loader_create_ns_t)(
    const char* name, const char* ld_library_path, const char* default_library_path,
    uint64_t type, const char* permitted_when_isolated_path,
    struct android_namespace_t* parent_namespace, const void* caller_addr);

typedef bool (*loader_link_all_t)(struct android_namespace_t* from,
                                  struct android_namespace_t* to);

// __loader_dlopen differs from dlopen by taking the caller address, which is
// what lets us impersonate the linker.
typedef void* (*loader_dlopen_t)(const char* filename, int flags, const void* caller_addr);

loader_create_ns_t g_create_ns = nullptr;
loader_link_all_t g_link_all = nullptr;
loader_dlopen_t g_loader_dlopen = nullptr;
struct android_namespace_t* g_default_ns = nullptr;
struct android_namespace_t* g_escape_ns = nullptr;
bool g_ns_tried = false;

static void* align_ptr(void* ptr) {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) & ~(getpagesize() - 1));
}

// dlopen() in libdl is a thin wrapper: it loads the caller's return address and
// branches to __loader_dlopen, passing that address as the third argument. So
// the branch target is recoverable by scanning for the first BL instruction.
//
// This is how liblinkernsbypass reaches the linker internals. A plain
// dlopen("libdl_android.so") cannot work: that library is not on the
// classloader namespace's permitted path list, so it is rejected before the
// linker ever consults a caller address.
static loader_dlopen_t find_loader_dlopen() {
#if defined(__aarch64__)
    union BranchLinked {
        uint32_t raw;
        struct {
            int32_t offset : 26;
            uint8_t sig : 6;
        };
        bool verify() const { return sig == 0x25; }
    };

    // Some devices map executables execute-only; scanning needs read access.
    mprotect(align_ptr(reinterpret_cast<void*>(&dlopen)), getpagesize(),
             PROT_READ | PROT_WRITE | PROT_EXEC);

    auto* bl = reinterpret_cast<BranchLinked*>(&dlopen);
    for (int i = 0; i < 16 && !bl->verify(); ++i) ++bl;
    if (!bl->verify()) return nullptr;

    auto* fn = reinterpret_cast<loader_dlopen_t>(bl + bl->offset);
    // __loader_dlopen is an internal symbol and may carry BTI landing pads
    // that reject indirect branches; the linker does not need them here.
    mprotect(align_ptr(reinterpret_cast<void*>(fn)), getpagesize(),
             PROT_READ | PROT_WRITE | PROT_EXEC);
    return fn;
#else
    return nullptr;
#endif
}

void init_namespace_escape() {
    if (g_ns_tried) return;
    g_ns_tried = true;

    g_loader_dlopen = find_loader_dlopen();
    if (!g_loader_dlopen) {
        fprintf(stderr, "[mithril] vk-dispatch: could not locate __loader_dlopen"
                        " (namespace escape unavailable)\n");
        return;
    }

    // Passing &dlopen as the caller address is the whole trick: the linker
    // treats the call as originating inside itself and serves it from the
    // unrestricted namespace.
    void* ld = g_loader_dlopen("ld-android.so", RTLD_LAZY, reinterpret_cast<void*>(&dlopen));
    void* libdl = g_loader_dlopen("libdl_android.so", RTLD_LAZY, reinterpret_cast<void*>(&dlopen));

    if (ld) {
        g_link_all = (loader_link_all_t)dlsym(ld, "__loader_android_link_namespaces_all_libs");
    }
    if (libdl) {
        g_create_ns = (loader_create_ns_t)dlsym(libdl, "__loader_android_create_namespace");
    }
    if (!g_create_ns && ld) {
        g_create_ns = (loader_create_ns_t)dlsym(ld, "__loader_android_create_namespace");
    }
    if (!g_create_ns || !g_link_all) {
        fprintf(stderr, "[mithril] vk-dispatch: linker internals unavailable (ld=%s libdl=%s)\n",
                ld ? "yes" : "no", libdl ? "yes" : "no");
        g_create_ns = nullptr;
        g_link_all = nullptr;
        return;
    }
    fprintf(stderr, "[mithril] vk-dispatch: linker internals resolved, namespace escape available\n");
}

// A shared namespace with no parent is what gives us a handle on the default
// namespace - it is not exported, so it can only be reached by copying it.
struct android_namespace_t* make_default_ns() {
    if (g_default_ns) return g_default_ns;
    if (!g_create_ns) return nullptr;
    g_default_ns = g_create_ns("mithril-default-copy", nullptr, nullptr,
                               MITHRIL_NS_TYPE_SHARED, nullptr, nullptr,
                               reinterpret_cast<void*>(&dlopen));
    return g_default_ns;
}

struct android_namespace_t* make_escape_ns(const char* driver_dir) {
    if (g_escape_ns) return g_escape_ns;
    init_namespace_escape();
    if (!g_create_ns) return nullptr;

    static char paths[1024];
    if (driver_dir && driver_dir[0]) {
        snprintf(paths, sizeof(paths), "%s:/system/lib64:/vendor/lib64:/system/lib64/hw:"
                                       "/vendor/lib64/hw:/system_ext/lib64:/apex/com.android.runtime/lib64/bionic",
                 driver_dir);
    } else {
        snprintf(paths, sizeof(paths), "/system/lib64:/vendor/lib64:/system/lib64/hw:"
                                       "/vendor/lib64/hw:/system_ext/lib64");
    }

    // The caller address is what removes the restrictions; the public
    // android_create_namespace() always passes our own return address instead,
    // which is why it cannot escape.
    g_escape_ns = g_create_ns("mithril-vulkan", paths, paths, MITHRIL_NS_TYPE_SHARED, paths,
                              nullptr, reinterpret_cast<void*>(&dlopen));
    if (!g_escape_ns) {
        fprintf(stderr, "[mithril] vk-dispatch: namespace creation failed\n");
        return nullptr;
    }

    // Without this the driver's own DT_NEEDED entries (libhardware.so,
    // libcutils.so on Turnip) resolve to nothing, because the app namespace
    // cannot see /system/lib64.
    struct android_namespace_t* def = make_default_ns();
    if (def && !g_link_all(g_escape_ns, def)) {
        fprintf(stderr, "[mithril] vk-dispatch: could not link namespace to default\n");
    } else if (def) {
        fprintf(stderr, "[mithril] vk-dispatch: escape namespace linked to default\n");
    }
    fprintf(stderr, "[mithril] vk-dispatch: escape namespace created (paths=%s)\n", paths);

    // Bring the stock loader into the namespace before the driver. Its own
    // DT_NEEDED entries - libhardware.so, libcutils.so, libutils.so, and on
    // newer releases the gralloc/nativewindow stack - are precisely the ones
    // Turnip needs, and they live in /system/lib64 which the app namespace
    // cannot see. Loading them here registers them by SONAME, so the driver's
    // later DT_NEEDED lookups hit already-loaded libraries instead of having
    // to search the namespace path list.
    //
    // This is why Turnip has to arrive through libvulkan.so rather than being
    // dlopened on its own: the stock loader is what drags those system
    // dependencies into the process, and Turnip's kgsl backend reaches
    // /dev/kgsl-3d0 through them.
    android_dlextinfo dlext{};
    dlext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    dlext.library_namespace = g_escape_ns;
    const char* loader_paths[] = {"/system/lib64/libvulkan.so", "/vendor/lib64/libvulkan.so",
                                  "libvulkan.so"};
    for (const char* lp : loader_paths) {
        void* h = android_dlopen_ext(lp, RTLD_LOCAL | RTLD_NOW, &dlext);
        if (h) {
            fprintf(stderr, "[mithril] vk-dispatch: system loader %s loaded into escape namespace\n",
                    lp);
            break;
        }
        fprintf(stderr, "[mithril] vk-dispatch: system loader %s failed: %s\n", lp, dlerror());
    }

    return g_escape_ns;
}

// The driver's exported names are the ground truth for how to talk to it.
// Mesa's Android builds are not consistent about which discovery entrypoint
// they export, and dlsym can only guess, so read the loaded image's own
// dynamic symbol table and report what is actually there.
struct SymbolScan {
    const char* wanted_path;
    std::vector<std::string> vk_symbols;
    std::string gipa_name;
};

static PFN_vkGetInstanceProcAddr try_hal_open();

static int collect_symbols_cb(struct dl_phdr_info* info, size_t, void* data) {
    auto* scan = static_cast<SymbolScan*>(data);
    if (!info->dlpi_name || !scan->wanted_path) return 0;
    // dlpi_name is whatever the linker recorded, which is not necessarily the
    // path we asked for: a library opened by SONAME reports its SONAME, and one
    // opened by absolute path may report the resolved path. Match on the base
    // name so either form hits.
    const char* a = info->dlpi_name;
    const char* b = scan->wanted_path;
    const char* ab = strrchr(a, '/');
    const char* bb = strrchr(b, '/');
    a = ab ? ab + 1 : a;
    b = bb ? bb + 1 : b;
    if (strcmp(a, b) != 0) return 0;

    const ElfW(Dyn)* dyn = nullptr;
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
        if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
            dyn = reinterpret_cast<const ElfW(Dyn)*>(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
            break;
        }
    }
    if (!dyn) return 0;

    const ElfW(Sym)* symtab = nullptr;
    const char* strtab = nullptr;
    const uint32_t* hash = nullptr;      // DT_HASH
    const uint32_t* gnu_hash = nullptr;  // DT_GNU_HASH
    for (const ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; ++d) {
        if (d->d_tag == DT_SYMTAB) symtab = reinterpret_cast<const ElfW(Sym)*>(info->dlpi_addr + d->d_un.d_ptr);
        else if (d->d_tag == DT_STRTAB) strtab = reinterpret_cast<const char*>(info->dlpi_addr + d->d_un.d_ptr);
        else if (d->d_tag == DT_HASH) hash = reinterpret_cast<const uint32_t*>(info->dlpi_addr + d->d_un.d_ptr);
        else if (d->d_tag == 0x6ffffef5u) gnu_hash = reinterpret_cast<const uint32_t*>(info->dlpi_addr + d->d_un.d_ptr);
    }
    if (!symtab || !strtab) return 0;

    // Two hash formats exist and a given library ships only one. DT_GNU_HASH is
    // the default on modern Android (the NDK passes --hash-style=gnu), while
    // DT_HASH is the older SysV one. Reading only DT_HASH reported "0 vk*
    // symbols" for drivers that do export them, because gnu-hash-only images
    // have no DT_HASH entry at all and the guard above bailed out.
    uint32_t nsyms = 0;
    if (hash) {
        nsyms = hash[1];  // nchain equals the symbol count
    } else if (gnu_hash) {
        const uint32_t nbuckets = gnu_hash[0];
        const uint32_t symoffset = gnu_hash[1];
        const uint32_t bloom_size = gnu_hash[2];
        const uint32_t bloom_shift = gnu_hash[3];
        const unsigned char* bloom = reinterpret_cast<const unsigned char*>(gnu_hash + 4);
        const uint32_t* buckets = reinterpret_cast<const uint32_t*>(
            bloom + bloom_size * (sizeof(ElfW(Addr)) == 8 ? 8 : 4));
        const uint32_t* chain = buckets + nbuckets;

        nsyms = symoffset;
        uint32_t max_bucket = 0;
        for (uint32_t i = 0; i < nbuckets; ++i) {
            if (buckets[i] > max_bucket) max_bucket = buckets[i];
        }
        if (max_bucket >= symoffset) {
            // Walk the chain of the last symbol in the final bucket until the
            // terminator bit is set; its index plus one is the symbol count.
            uint32_t i = max_bucket - symoffset;
            while (!(chain[i] & 1u)) ++i;
            nsyms = symoffset + i + 1;
        }
        (void)bloom_shift;
    } else {
        return 0;
    }

    for (uint32_t i = 0; i < nsyms; ++i) {
        const char* name = strtab + symtab[i].st_name;
        if (!name || name[0] == '\0') continue;
        if (strncmp(name, "vk", 2) == 0) {
            scan->vk_symbols.emplace_back(name);
        } else if (strstr(name, "GetInstanceProcAddr")) {
            scan->gipa_name = name;
        }
    }
    return 0;
}

void report_driver_symbols(const char* path) {
    SymbolScan scan{path, {}, {}};
    dl_iterate_phdr(collect_symbols_cb, &scan);
    fprintf(stderr, "[mithril] vk-dispatch: driver exports %zu vk* symbols", scan.vk_symbols.size());
    for (size_t i = 0; i < scan.vk_symbols.size() && i < 12; ++i) {
        fprintf(stderr, " %s", scan.vk_symbols[i].c_str());
    }
    fprintf(stderr, "\n");
    if (!scan.gipa_name.empty()) {
        fprintf(stderr, "[mithril] vk-dispatch: non-standard discovery symbol \"%s\"\n",
                scan.gipa_name.c_str());
    }
}

void log_dir(const char* dir) {
    if (!dir || !dir[0]) return;
    DIR* d = opendir(dir);
    if (!d) {
        fprintf(stderr, "[mithril] vk-dispatch: cannot list driver dir %s: %s\n", dir, strerror(errno));
        return;
    }
    fprintf(stderr, "[mithril] vk-dispatch: driver dir %s contains:\n", dir);
    int n = 0;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr && n < 40) {
        if (e->d_name[0] == '.') continue;
        fprintf(stderr, "[mithril] vk-dispatch:   %s\n", e->d_name);
        ++n;
    }
    if (n == 0) fprintf(stderr, "[mithril] vk-dispatch:   (empty)\n");
    closedir(d);
}

static void log_dlerror(const char* path, const char* how) {
    // dlerror() clears its state on read, so it must be captured once -
    // reading it twice (once in the condition, once for the value) yields NULL
    // the second time and hides the real reason.
    const char* err = dlerror();
    fprintf(stderr, "[mithril] vk-dispatch: %s dlopen(\"%s\") failed: %s\n",
            how, path, err ? err : "unknown");
}

void* try_load(const char* path, const char* driver_dir) {
    // 1) Unrestricted namespace. This is the only route that reliably reaches
    //    another package's library directory on Android N+.
    if (!g_escape_ns) g_escape_ns = make_escape_ns(driver_dir);
    if (g_escape_ns) {
        android_dlextinfo ext{};
        ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
        ext.library_namespace = g_escape_ns;
        void* h = android_dlopen_ext(path, RTLD_NOW | RTLD_LOCAL, &ext);
        if (h) {
            fprintf(stderr, "[mithril] vk-dispatch: loaded \"%s\" via escape namespace\n", path);
            g_driver_path = path;
            return h;
        }
        log_dlerror(path, "namespace");
    }

    // 2) From an open descriptor: skips the path-based checks entirely.
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        android_dlextinfo ext{};
        ext.flags = ANDROID_DLEXT_USE_LIBRARY_FD | ANDROID_DLEXT_FORCE_LOAD;
        ext.library_fd = fd;
        void* h = android_dlopen_ext(path, RTLD_NOW | RTLD_LOCAL, &ext);
        close(fd);
        if (h) {
            fprintf(stderr, "[mithril] vk-dispatch: loaded \"%s\" via file descriptor\n", path);
            return h;
        }
        log_dlerror(path, "fd");
    }

    // 3) Plain dlopen. Works for system libraries, and for anything the
    //    launcher already put on LD_LIBRARY_PATH.
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (h) {
        fprintf(stderr, "[mithril] vk-dispatch: loaded \"%s\" via dlopen\n", path);
        return h;
    }
    log_dlerror(path, "plain");
    return nullptr;
}

// Candidate driver paths, most specific first. The last slots are reserved for
// the platform loader, so an over-long driver list can never crowd out the
// fallback - that is exactly what happened when Turnip's three name variants
// (joined + bare = six entries) filled the array and libvulkan.so was dropped,
// leaving the process with no Vulkan driver at all.
static const int kMaxCandidates = 12;
static const int kLoaderSlot = kMaxCandidates - 3;

// Does the platform loader already serve a Mesa/freedreno (Turnip) driver?
//
// This is how ANGLE drives Turnip, and why it works on every version: the
// application never opens a driver by name. It opens libvulkan.so and lets the
// loader - plus whatever the launcher or adrenotools installed in front of it -
// decide which driver backs it. WSI then comes from the loader, so the render
// path is the ordinary one with a real swapchain and nothing has to be
// presented offscreen.
//
// We can only take that route when the loader really is serving Turnip, which
// cannot be assumed: hw_get_module searches fixed system directories and never
// looks inside an app's library directory. So build a throwaway instance and
// read the physical device name. It is the only reliable way to tell the two
// drivers apart, because the GPU both of them report is the same Adreno.
static bool device_name_is_turnip(const char* name) {
    if (!name) return false;
    return strstr(name, "Turnip") != nullptr || strstr(name, "turnip") != nullptr ||
           strstr(name, "freedreno") != nullptr;
}

static bool loader_reports_turnip(PFN_vkGetInstanceProcAddr gipa) {
    if (!gipa) return false;
    auto* create_instance =
        reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!create_instance) return false;

    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;

    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = create_instance(&ci, nullptr, &inst);
    if (r != VK_SUCCESS || inst == VK_NULL_HANDLE) {
        fprintf(stderr, "[mithril] vk-dispatch: loader probe: vkCreateInstance failed (%d)\n",
                static_cast<int>(r));
        return false;
    }

    auto* enum_devs = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
        gipa(inst, "vkEnumeratePhysicalDevices"));
    auto* get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        gipa(inst, "vkGetPhysicalDeviceProperties"));
    auto* destroy_instance =
        reinterpret_cast<PFN_vkDestroyInstance>(gipa(inst, "vkDestroyInstance"));

    bool found = false;
    uint32_t count = 0;
    if (enum_devs && get_props && enum_devs(inst, &count, nullptr) == VK_SUCCESS && count > 0 &&
        count <= 16) {
        std::vector<VkPhysicalDevice> devs(count);
        if (enum_devs(inst, &count, devs.data()) == VK_SUCCESS) {
            for (uint32_t i = 0; i < count; ++i) {
                VkPhysicalDeviceProperties props{};
                get_props(devs[i], &props);
                fprintf(stderr, "[mithril] vk-dispatch: loader probe device %u: %s\n", i,
                        props.deviceName);
                if (device_name_is_turnip(props.deviceName)) found = true;
            }
        }
    }
    if (destroy_instance) destroy_instance(inst, nullptr);
    return found;
}

void add_candidate(const char* cands[], int& n, const char* dir, const char* name) {
    if (n >= kLoaderSlot) return;
    if (!name || !name[0]) return;
    if (name[0] == '/') {
        cands[n++] = name;
        return;
    }
    if (!dir || !dir[0]) {
        cands[n++] = name;
        return;
    }
    size_t len = strlen(dir) + strlen(name) + 2;
    char* buf = (char*)malloc(len);
    if (!buf) return;
    snprintf(buf, len, "%s/%s", dir, name);
    cands[n++] = buf;
    if (n < kLoaderSlot) cands[n++] = name;
}

// Route the platform loader at the chosen driver instead of driving the
// driver ourselves.
//
// Everything about the direct approach works - the HAL opens, the device
// enumerates, it reports Vulkan 1.3 - except presentation. Turnip reports no
// VK_KHR_surface, no VK_KHR_android_surface and no VK_KHR_swapchain, and
// vkCreateInstance rejects both of the former outright. Those belong to
// libvulkan.so on Android, which is why Zink and ANGLE only ever see WSI
// through it, and why a directly driven HAL can only ever be an offscreen
// device.
//
// So: load the hook object into the namespace first, hand it the driver, then
// load libvulkan.so into that same namespace. The loader keeps its WSI and its
// own view of the driver; when it asks for the vendor HAL module the hook
// answers with Turnip. libmithril.so then takes every entrypoint from the
// loader exactly as it would with the stock driver.
//
// Best effort throughout: on any failure we fall through to the existing
// paths, so a device without Turnip is unaffected.
static int g_hook_active = 0;

static void* try_hook_route(const char* driver_dir, const char* driver_name) {
    if (!g_escape_ns) g_escape_ns = make_escape_ns(driver_dir);
    if (!g_escape_ns) {
        fprintf(stderr, "[mithril] vk-dispatch: hook route unavailable (no namespace)\n");
        return nullptr;
    }

    // Our own library directory: the hook object ships beside libmithril.so.
    Dl_info self{};
    if (!dladdr((void*)&try_hook_route, &self) || !self.dli_fname) return nullptr;
    std::string self_path(self.dli_fname);
    size_t slash = self_path.find_last_of('/');
    if (slash == std::string::npos) return nullptr;
    std::string hook = self_path.substr(0, slash + 1) + "libmithril_vkhook.so";

    android_dlextinfo ext{};
    ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    ext.library_namespace = g_escape_ns;

    // RTLD_GLOBAL is the whole point: the hook's android_dlopen_ext /
    // android_load_sphal_library must be visible to libvulkan.so.
    void* h = android_dlopen_ext(hook.c_str(), RTLD_NOW | RTLD_GLOBAL, &ext);
    if (!h) {
        h = dlopen(hook.c_str(), RTLD_NOW | RTLD_GLOBAL);
    }
    if (!h) {
        fprintf(stderr, "[mithril] vk-dispatch: hook object not loaded (%s): %s\n",
                hook.c_str(), dlerror() ? dlerror() : "unknown");
        return nullptr;
    }
    fprintf(stderr, "[mithril] vk-dispatch: hook object loaded: %s\n", hook.c_str());

    void (*init_fn)(const char*, const char*, int) =
        (void (*)(const char*, const char*, int))dlsym(h, "mithril_vkhook_init");
    if (!init_fn) {
        fprintf(stderr, "[mithril] vk-dispatch: hook has no init entrypoint\n");
        return nullptr;
    }
    const char* trace = getenv("MITHRIL_DEBUG");
    init_fn(driver_dir ? driver_dir : "", driver_name ? driver_name : "",
            trace && trace[0] ? 1 : 0);

    // Now the loader, into the same namespace, after the hook.
    void* loader = android_dlopen_ext("libvulkan.so", RTLD_NOW | RTLD_LOCAL, &ext);
    if (!loader) {
        log_dlerror("libvulkan.so", "hook namespace");
        loader = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!loader) {
        fprintf(stderr, "[mithril] vk-dispatch: loader not loaded under hook\n");
        return nullptr;
    }
    g_hook_active = 1;
    fprintf(stderr, "[mithril] vk-dispatch: hook route active (loader redirected)\n");
    return loader;
}

void ensure_library() {
    if (g_ready) return;
    g_ready = true;

    // DRIVER_PATH is how the launcher points at an out-of-tree driver; FCL sets
    // it to the Turnip plugin's native library directory, which is not on
    // LD_LIBRARY_PATH and not inside our own linker namespace.
    const char* driver_dir = getenv("DRIVER_PATH");
    const char* explicit_path = getenv("MITHRIL_VULKAN_LIBRARY");
    const char* turnip = getenv("MITHRIL_TURNIP");
    const char* system_driver = getenv("VULKAN_DRIVER_SYSTEM");

    // Which driver to use is the launcher's decision, not ours. It exposes a
    // "use the system Vulkan driver" switch and a "use another driver" switch,
    // and expresses the result through MITHRIL_VULKAN_LIBRARY (an explicitly
    // named driver) or MITHRIL_TURNIP. Our job is only to make either choice
    // work - not to second-guess it.
    //
    // "Turnip requested" has to mean an actual enablement. The test that used
    // to gate the driver/hook paths was `turnip && turnip[0]`, which is also
    // true for MITHRIL_TURNIP=0 - so a launcher that had explicitly chosen the
    // system driver was still pushed down the custom-driver path. It is now the
    // same predicate that builds the candidate list, so the two can no longer
    // disagree.
    // VULKAN_DRIVER_SYSTEM does NOT mean "the launcher picked the system
    // driver". Two field runs on the same device show the opposite pairing:
    //
    //   launcher switch = Turnip   -> MITHRIL_TURNIP=1, VULKAN_DRIVER_SYSTEM=1
    //   launcher switch = system   -> neither variable set
    //
    // So the variable only says a system driver is *available*; FCL exports it
    // even while the switch points at Turnip. Letting it win here made every
    // Turnip run silently load the system driver instead, which is the one
    // thing this code must never do: the choice belongs to the launcher.
    //
    // Precedence is therefore:
    //   1. MITHRIL_TURNIP set to an enabling value          -> Turnip
    //   2. MITHRIL_TURNIP set to a disabling value          -> system driver
    //   3. MITHRIL_TURNIP absent, VULKAN_DRIVER_SYSTEM set   -> system driver
    //   4. neither set                                       -> system driver
    const bool turnip_explicit_on = turnip &&
                                    (turnip[0] == '1' || turnip[0] == 'y' ||
                                     turnip[0] == 'Y' || turnip[0] == 't' ||
                                     turnip[0] == 'T');
    const bool turnip_explicit_off = turnip &&
                                     (turnip[0] == '0' || turnip[0] == 'n' ||
                                      turnip[0] == 'N' || turnip[0] == 'f' ||
                                      turnip[0] == 'F');
    const bool force_system = !turnip_explicit_on &&
                              (turnip_explicit_off ||
                               (system_driver && system_driver[0]));

    const bool turnip_on = turnip_explicit_on;
    const bool explicit_driver = explicit_path && explicit_path[0];
    const bool driver_requested = explicit_driver || turnip_on;

    // Say which variable produced the choice: "system driver" alone is useless
    // when a launcher sets both, which is the exact case that was mis-resolved.
    const char* choice_desc =
        turnip_on        ? "turnip (MITHRIL_TURNIP)"
      : explicit_driver  ? "explicit driver (MITHRIL_VULKAN_LIBRARY)"
      : turnip_explicit_off ? "system driver (MITHRIL_TURNIP disabled)"
      : (system_driver && system_driver[0]) ? "system driver (VULKAN_DRIVER_SYSTEM)"
      : "system driver (default)";
    fprintf(stderr, "[mithril] vk-dispatch: launcher choice: %s\n", choice_desc);

    log_dir(driver_dir);

    const char* cands[kMaxCandidates];
    int n = 0;

    if (explicit_driver) {
        add_candidate(cands, n, driver_dir, explicit_path);
    } else if (turnip_on) {
        add_candidate(cands, n, driver_dir, "libvulkan_freedreno.so");
        add_candidate(cands, n, driver_dir, "libvulkan_adreno.so");
        add_candidate(cands, n, driver_dir, "vulkan.adreno.so");
    }
    // Platform loader fallbacks. These go at kLoaderSlot onward rather than at
    // n: the driver scan iterates 0..kLoaderSlot-1 and the loader scan
    // iterates kLoaderSlot..kMaxCandidates-1, so the two lists must not
    // overlap regardless of how many driver names were added above.
    cands[kLoaderSlot] = "libvulkan.so";
    cands[kLoaderSlot + 1] = "/system/lib64/libvulkan.so";
    cands[kLoaderSlot + 2] = "/vendor/lib64/libvulkan.so";
    for (int i = kLoaderSlot + 3; i < kMaxCandidates; ++i) cands[i] = nullptr;

    // Two different things can be loaded here and keeping them apart matters.
    //
    // A Mesa driver such as Turnip is a Vulkan HAL module: it exports HMI, and
    // its Vulkan entrypoints live behind hw_module_methods_t::open() rather
    // than in the dynamic symbol table (Mesa builds with hidden visibility and
    // exports only the module symbols, which is why dlsym("vkCreateInstance")
    // on it finds nothing). It is built to be opened by the platform loader.
    //
    // The platform loader, however, cannot reach it: hw_get_module("vulkan")
    // only searches fixed system directories such as /vendor/lib64/hw, and a
    // driver plugin lives under /data/app/.../lib/arm64. Loading Turnip first
    // and then letting the loader take over therefore just yields the stock
    // driver - which is what the previous attempt did.
    //
    // So talk to the driver directly when one was asked for, and only fall
    // back to the loader if that yields no usable entrypoints. On the driver we
    // try, in order: its own exported vk* symbols, the ICD discovery entry, and
    // finally the HAL handshake (HMI -> open -> hwvulkan_device_t), which is
    // the path a HAL module actually expects.
    // Drive an explicitly requested driver directly, and only try to route the
    // platform loader at it if the direct handshake yields nothing usable.
    //
    // The order used to be the other way round, on the theory that only the
    // loader has WSI. That theory was falsified by a real run on the reference
    // device: the hook reported "hook route active (loader redirected)", yet
    // /proc/maps contained no libvulkan_freedreno.so and the enumerated device
    // was still "Adreno (TM) 619 (api 0x401080)" - the stock 1.1 driver. The
    // loader resolves its HAL through hw_get_module well before our hook is in
    // place, so the redirect never fires; it only ever produced the stock
    // driver while looking like it had worked.
    //
    // The direct path is the one that has actually been observed to reach
    // Turnip ("Physical device: Turnip Adreno (TM) 619 (v32) (api 0x40316b)"),
    // so it goes first. It has no WSI, but that is not fatal: create_swapchain
    // detects the missing VK_KHR_swapchain and switches to the offscreen
    // present path, which is exactly how Zink drives a Mesa HAL driver.
    // Prefer the loader whenever it already serves the requested driver. Only
    // the loader has WSI, so this is the one route that ends in a real
    // swapchain instead of the offscreen present path.
    void* probe_handle = nullptr;
    bool loader_has_turnip = false;
    if (driver_requested) {
        for (int i = kLoaderSlot; i < kMaxCandidates; ++i) {
            if (!cands[i]) continue;
            probe_handle = try_load(cands[i], driver_dir);
            if (probe_handle) break;
        }
        if (probe_handle) {
            auto* probe_gipa =
                reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(probe_handle, "vkGetInstanceProcAddr"));
            loader_has_turnip = loader_reports_turnip(probe_gipa);
            fprintf(stderr, "[mithril] vk-dispatch: platform loader serves turnip: %s\n",
                    loader_has_turnip ? "yes" : "no");
        }
    }

    void* driver_handle = nullptr;
    if (driver_requested && !loader_has_turnip) {
        for (int i = 0; i < n && i < kLoaderSlot; ++i) {
            if (!cands[i]) continue;
            driver_handle = try_load(cands[i], driver_dir);
            if (driver_handle) {
                fprintf(stderr, "[mithril] vk-dispatch: driver loaded: %s\n", cands[i]);
                g_driver_path = cands[i];
                break;
            }
        }
    }

    // Only reached when the requested driver could not be driven directly.
    void* hook_loader = nullptr;
    if (!driver_handle && driver_requested && !loader_has_turnip) {
        const char* name = explicit_driver ? explicit_path : "libvulkan_freedreno.so";
        hook_loader = try_hook_route(driver_dir, name);
    }

    // The probe already opened the loader; reuse it instead of opening it twice.
    void* loader_handle = hook_loader ? hook_loader : probe_handle;
    for (int i = loader_handle ? kMaxCandidates : kLoaderSlot; i < kMaxCandidates; ++i) {
        if (!cands[i]) continue;
        loader_handle = try_load(cands[i], driver_dir);
        if (loader_handle) {
            fprintf(stderr, "[mithril] vk-dispatch: platform loader available: %s\n", cands[i]);
            break;
        }
    }

    // Prefer the custom driver; it is only usable if we can actually get
    // entrypoints out of it, and silently falling back is what lets a device
    // without Turnip still run.
    // A hook-mediated loader wins over a directly driven driver: the latter has
    // no way to present.
    g_handle = hook_loader ? hook_loader : (driver_handle ? driver_handle : loader_handle);
    if (!g_handle) {
        fprintf(stderr, "[mithril] vk-dispatch: no Vulkan driver could be loaded\n");
        return;
    }
    g_fallback_handle = (g_handle == driver_handle && !hook_loader) ? loader_handle : nullptr;

    // An ICD such as Turnip exposes discovery through vk_icdGetInstanceProcAddr
    // rather than the plain names; a loader exports the plain names.
    g_icd_gipa = (PFN_vk_icdGetInstanceProcAddr)dlsym(g_handle, "vk_icdGetInstanceProcAddr");
    g_gipa = (PFN_vkGetInstanceProcAddr)dlsym(g_handle, "vkGetInstanceProcAddr");
    fprintf(stderr, "[mithril] vk-dispatch: discovery entrypoints icd=%s gipa=%s\n",
            g_icd_gipa ? "yes" : "no", g_gipa ? "yes" : "no");
    if (!g_icd_gipa && !g_gipa) {
        // Ask the image itself what it exports rather than guessing. A Mesa
        // build may ship the discovery entrypoint under any name, and some
        // vendor drivers hide it inside a namespace (Adreno has shipped
        // qglinternal::vkGetInstanceProcAddr).
        // Ask dladdr for the path the linker actually recorded rather than
        // reusing the string we passed to dlopen - they differ whenever the
        // library was opened by SONAME, and the walk below matches on it.
        const char* recorded = g_driver_path;
        Dl_info self{};
        if (g_handle && dladdr(reinterpret_cast<void*>(g_icd_gipa ? (void*)g_icd_gipa : (void*)g_gipa), &self) == 0) {
            if (dladdr(g_handle, &self) && self.dli_fname) recorded = self.dli_fname;
        } else if (self.dli_fname) {
            recorded = self.dli_fname;
        }
        SymbolScan scan{recorded, {}, {}};
        dl_iterate_phdr(collect_symbols_cb, &scan);
        fprintf(stderr, "[mithril] vk-dispatch: driver exports %zu vk* symbols",
                scan.vk_symbols.size());
        for (size_t i = 0; i < scan.vk_symbols.size() && i < 16; ++i) {
            fprintf(stderr, " %s", scan.vk_symbols[i].c_str());
        }
        fprintf(stderr, "\n");
        if (!scan.gipa_name.empty()) {
            fprintf(stderr, "[mithril] vk-dispatch: trying non-standard discovery symbol \"%s\"\n",
                    scan.gipa_name.c_str());
            g_gipa = (PFN_vkGetInstanceProcAddr)dlsym(g_handle, scan.gipa_name.c_str());
        }
        if (!g_gipa) g_gipa = try_hal_open();
    }
    // Nothing usable from the driver. If a platform loader is available, switch
    // to it rather than leaving the process with no Vulkan at all - on a device
    // where Turnip cannot be driven this is the difference between running on
    // the stock driver and crashing.
    if (!g_icd_gipa && !g_gipa && g_fallback_handle) {
        fprintf(stderr, "[mithril] vk-dispatch: falling back to platform loader\n");
        g_handle = g_fallback_handle;
        g_fallback_handle = nullptr;
        g_driver_path = nullptr;
        g_icd_gipa = (PFN_vk_icdGetInstanceProcAddr)dlsym(g_handle, "vk_icdGetInstanceProcAddr");
        g_gipa = (PFN_vkGetInstanceProcAddr)dlsym(g_handle, "vkGetInstanceProcAddr");
        fprintf(stderr, "[mithril] vk-dispatch: loader discovery icd=%s gipa=%s\n",
                g_icd_gipa ? "yes" : "no", g_gipa ? "yes" : "no");
    }
    if (!g_icd_gipa && !g_gipa) {
        fprintf(stderr, "[mithril] vk-dispatch: driver exports no discovery entrypoint\n");
    }
}

// ---------------------------------------------------------------------------
// HAL path
//
// Turnip ships as a Vulkan HAL module, not as an ICD. It is built with
// -Dandroid-stub=true and the Adrenotools/Magisk packages then run
// patchelf --set-soname vulkan.<board>.so over it, so what the file exports is
// the HAL module symbol HMI - the Vulkan entrypoints sit behind
// hw_module_methods_t::open() rather than being exported directly.
//
// That is exactly what the platform loader walks through at
// frameworks/native/vulkan/libvulkan/driver.cpp: dlsym "HMI", check the module
// id is "vulkan", open HWVULKAN_DEVICE_0, and take vkGetInstanceProcAddr and
// vkCreateInstance off the returned hwvulkan_device_t. Nothing here needs
// inline hooking - the same handshake can be performed directly, which is why
// libvulkan.so is able to reach a driver that an app cannot dlopen meaningfully
// on its own.
//
// Window system integration is not lost by skipping the loader: Mesa's WSI
// layer implements VK_KHR_android_surface itself.
struct mithril_hw_module_t;
struct mithril_hw_device_t;

struct mithril_hw_module_methods_t {
    int (*open)(const struct mithril_hw_module_t* module, const char* id,
                struct mithril_hw_device_t** device);
};

struct mithril_hw_module_t {
    uint32_t tag;
    uint16_t module_api_version;
    uint16_t hal_api_version;
    const char* id;
    const char* name;
    const char* author;
    struct mithril_hw_module_methods_t* methods;
    void* dso;
    uint32_t reserved[32 - 7];
};

struct mithril_hw_device_t {
    uint32_t tag;
    uint32_t version;
    struct mithril_hw_module_t* module;
    uint32_t reserved[12];
    int (*close)(struct mithril_hw_device_t* device);
};

struct mithril_hwvulkan_device_t {
    struct mithril_hw_device_t common;
    VkResult (*EnumerateInstanceExtensionProperties)(const char* layer_name, uint32_t* count,
                                                     VkExtensionProperties* properties);
    VkResult (*CreateInstance)(const VkInstanceCreateInfo* create_info,
                               const VkAllocationCallbacks* allocator, VkInstance* instance);
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
};

static void probe_handler(int sig) {
    if (g_probe_active) siglongjmp(g_probe_jmp, 1);
    // Not our probe: restore and re-raise so the real crash handler runs.
    struct sigaction* old = (sig == SIGSEGV) ? &g_old_segv
                          : (sig == SIGBUS ? &g_old_bus : &g_old_ill);
    signal(sig, SIG_DFL);
    raise(sig);
}

static void install_probe_handlers() {
    struct sigaction sa{};
    sa.sa_handler = probe_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    sigaction(SIGSEGV, &sa, &g_old_segv);
    sigaction(SIGBUS, &sa, &g_old_bus);
    sigaction(SIGILL, &sa, &g_old_ill);
}

static PFN_vkGetInstanceProcAddr try_hal_open() {
    auto* hmi = static_cast<mithril_hw_module_t*>(dlsym(g_handle, "HMI"));
    if (!hmi) return nullptr;
    fprintf(stderr, "[mithril] vk-dispatch: HAL module symbol HMI present (id=%s)\n",
            hmi->id ? hmi->id : "(null)");
    if (!hmi->id || strcmp(hmi->id, "vulkan") != 0) return nullptr;
    if (!hmi->methods || !hmi->methods->open) return nullptr;

    install_probe_handlers();
    mithril_hw_device_t* raw = nullptr;
    if (hmi->methods->open(hmi, "vk0", &raw) != 0 || !raw) {
        fprintf(stderr, "[mithril] vk-dispatch: HAL open failed\n");
        return nullptr;
    }
    g_hal_device = raw;

    // The device struct's tail is driver-defined - the AOSP header and what a
    // Mesa build actually fills in have differed before, and reading the wrong
    // offset yields null pointers with no error. So dump the raw words rather
    // than trusting a fixed layout, and pick the GetInstanceProcAddr slot by
    // probing: a genuine one answers vkGetInstanceProcAddr for a null instance.
    auto* dev = reinterpret_cast<mithril_hwvulkan_device_t*>(raw);
    fprintf(stderr, "[mithril] vk-dispatch: HAL device opened (tag=%u version=%u)\n",
            raw->tag, raw->version);
    // Mesa's tu_hal_open() fills GetInstanceProcAddr/CreateInstance right after
    // hw_device_t, which is what these sizes encode. Print them so a mismatch
    // between our layout and the one the driver was built against is visible
    // rather than silent.
    fprintf(stderr, "[mithril] vk-dispatch: HAL sizes: hw_module_t=%zu hw_device_t=%zu hwvulkan_device_t=%zu\n",
            sizeof(mithril_hw_module_t), sizeof(mithril_hw_device_t),
            sizeof(mithril_hwvulkan_device_t));

    const void** words = reinterpret_cast<const void**>(raw);
    const size_t kWords = 24;
    fprintf(stderr, "[mithril] vk-dispatch: HAL device words");
    for (size_t i = 0; i < kWords; ++i) fprintf(stderr, " [%zu]=%p", i, const_cast<void*>(words[i]));
    fprintf(stderr, "\n");

    PFN_vkGetInstanceProcAddr found = nullptr;
    if (dev->GetInstanceProcAddr) {
        found = dev->GetInstanceProcAddr;
        fprintf(stderr, "[mithril] vk-dispatch: GetInstanceProcAddr at declared offset\n");
    } else {
        for (size_t i = 0; i < kWords; ++i) {
            auto* cand = reinterpret_cast<PFN_vkGetInstanceProcAddr>(const_cast<void*>(words[i]));
            if (!cand) continue;
            // Only slots that live in the device's function area are plausible;
            // the first words are integers and pointers we must not call.
            // Word 8 of hw_device_t is close(), which returns -1 and would
            // therefore look like a non-null answer to a single probe. Require
            // two things: a non-null answer for a name that must exist, and a
            // null answer for one that cannot. Only a real
            // GetInstanceProcAddr behaves that way.
            if (i < 4 || i == 8) continue;
            void* probe = nullptr;
            void* bogus = nullptr;
            if (sigsetjmp(g_probe_jmp, 1) == 0) {
                g_probe_active = true;
                probe = (void*)cand(VK_NULL_HANDLE, "vkGetInstanceProcAddr");
                bogus = (void*)cand(VK_NULL_HANDLE, "mithril_not_a_real_entrypoint");
                g_probe_active = false;
            } else {
                g_probe_active = false;
                fprintf(stderr, "[mithril] vk-dispatch: probe of word [%zu] trapped, skipping\n", i);
                continue;
            }
            if (probe && !bogus) {
                found = cand;
                fprintf(stderr, "[mithril] vk-dispatch: GetInstanceProcAddr found at word [%zu]\n", i);
                break;
            }
        }
    }

    if (found && !g_hal_create_instance) {
        // Ask the driver for its own vkCreateInstance through the instance
        // proc addr we just established; the HAL struct may not carry it.
        void* ci = (void*)found(VK_NULL_HANDLE, "vkCreateInstance");
        if (ci) g_hal_create_instance = reinterpret_cast<decltype(g_hal_create_instance)>(ci);
    }
    fprintf(stderr, "[mithril] vk-dispatch: HAL gipa=%s createInstance=%s\n",
            found ? "yes" : "no", g_hal_create_instance ? "yes" : "no");
    return found;
}

void* resolve(const char* name) {
    ensure_library();

    // The HAL device owns vkCreateInstance directly; it has to be served before
    // any instance exists, which is precisely when it is asked for.
    if (g_hal_create_instance && strcmp(name, "vkCreateInstance") == 0) {
        return reinterpret_cast<void*>(g_hal_create_instance);
    }

    // An ICD only exports vk_icdGetInstanceProcAddr; every other entrypoint has
    // to be fetched through the instance, and device-level ones through the
    // device. Resolving only against the library handle works for the platform
    // loader (which really does export all of them) but silently yields nothing
    // for Turnip, so the phases are tried most-specific first.
    if (g_gdpa && g_device) {
        void* p = (void*)g_gdpa(g_device, name);
        if (p) return p;
    }
    // Called with g_instance even when it is still null. Global commands -
    // vkEnumerateInstanceExtensionProperties, vkEnumerateInstanceVersion and
    // friends - are asked for BEFORE any instance exists, which is exactly the
    // case a null instance expresses: GetInstanceProcAddr(NULL, name) is the
    // documented way to reach them. Gating this on g_instance being non-null
    // made those lookups miss and left the driver unusable even after its
    // GetInstanceProcAddr had been found.
    if (g_gipa) {
        void* p = (void*)g_gipa(g_instance, name);
        if (p) return p;
    }
    if (g_icd_gipa) {
        void* p = (void*)g_icd_gipa(nullptr, name);
        if (p) return p;
    }
    if (g_handle) {
        void* p = dlsym(g_handle, name);
        if (p) return p;
    }
    if (g_failures++ < 12) {
        fprintf(stderr, "[mithril] vk-dispatch: unresolved entrypoint %s\n", name);
    }
    return nullptr;
}

// Called once the instance exists so instance- and device-level lookups become
// available. Without this an ICD stays unusable after vkCreateInstance.
void note_instance(VkInstance inst) {
    g_instance = inst;
    void* gipa = nullptr;
    if (g_icd_gipa) gipa = (void*)g_icd_gipa(inst, "vkGetInstanceProcAddr");
    if (!gipa && g_handle) gipa = dlsym(g_handle, "vkGetInstanceProcAddr");
    if (gipa && !g_gipa) g_gipa = (PFN_vkGetInstanceProcAddr)gipa;
    if (g_gipa && !g_gdpa) {
        g_gdpa = (PFN_vkGetDeviceProcAddr)g_gipa(inst, "vkGetDeviceProcAddr");
    }
    fprintf(stderr, "[mithril] vk-dispatch: instance created (gipa=%s gdpa=%s)\n",
            g_gipa ? "yes" : "no", g_gdpa ? "yes" : "no");
}

void note_device(VkDevice dev) {
    g_device = dev;
    fprintf(stderr, "[mithril] vk-dispatch: device created\n");
}

void clear_instance() {
    g_instance = VK_NULL_HANDLE;
    g_gdpa = nullptr;
}

void clear_device() {
    g_device = VK_NULL_HANDLE;
}

} // namespace

extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex) {
    static PFN_vkAcquireNextImageKHR fp = nullptr;
    if (!fp) fp = (PFN_vkAcquireNextImageKHR)resolve("vkAcquireNextImageKHR");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, swapchain, timeout, semaphore, fence, pImageIndex);
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo, VkCommandBuffer* pCommandBuffers) {
    static PFN_vkAllocateCommandBuffers fp = nullptr;
    if (!fp) fp = (PFN_vkAllocateCommandBuffers)resolve("vkAllocateCommandBuffers");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pAllocateInfo, pCommandBuffers);
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets) {
    static PFN_vkAllocateDescriptorSets fp = nullptr;
    if (!fp) fp = (PFN_vkAllocateDescriptorSets)resolve("vkAllocateDescriptorSets");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pAllocateInfo, pDescriptorSets);
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory) {
    static PFN_vkAllocateMemory fp = nullptr;
    if (!fp) fp = (PFN_vkAllocateMemory)resolve("vkAllocateMemory");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pAllocateInfo, pAllocator, pMemory);
}
VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo) {
    static PFN_vkBeginCommandBuffer fp = nullptr;
    if (!fp) fp = (PFN_vkBeginCommandBuffer)resolve("vkBeginCommandBuffer");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(commandBuffer, pBeginInfo);
}
VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset) {
    static PFN_vkBindBufferMemory fp = nullptr;
    if (!fp) fp = (PFN_vkBindBufferMemory)resolve("vkBindBufferMemory");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, buffer, memory, memoryOffset);
}
VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset) {
    static PFN_vkBindImageMemory fp = nullptr;
    if (!fp) fp = (PFN_vkBindImageMemory)resolve("vkBindImageMemory");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, image, memory, memoryOffset);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBeginQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query, VkQueryControlFlags flags) {
    static PFN_vkCmdBeginQuery fp = nullptr;
    if (!fp) fp = (PFN_vkCmdBeginQuery)resolve("vkCmdBeginQuery");
    if (!fp) { return; }
    fp(commandBuffer, queryPool, query, flags);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin, VkSubpassContents contents) {
    static PFN_vkCmdBeginRenderPass fp = nullptr;
    if (!fp) fp = (PFN_vkCmdBeginRenderPass)resolve("vkCmdBeginRenderPass");
    if (!fp) { return; }
    fp(commandBuffer, pRenderPassBegin, contents);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBeginRendering(VkCommandBuffer commandBuffer, const VkRenderingInfo* pRenderingInfo) {
    static PFN_vkCmdBeginRendering fp = nullptr;
    if (!fp) fp = (PFN_vkCmdBeginRendering)resolve("vkCmdBeginRendering");
    if (!fp) { return; }
    fp(commandBuffer, pRenderingInfo);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderingKHR(VkCommandBuffer commandBuffer, const VkRenderingInfo* pRenderingInfo) {
    static PFN_vkCmdBeginRenderingKHR fp = nullptr;
    if (!fp) fp = (PFN_vkCmdBeginRenderingKHR)resolve("vkCmdBeginRenderingKHR");
    if (!fp) { return; }
    fp(commandBuffer, pRenderingInfo);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets) {
    static PFN_vkCmdBindDescriptorSets fp = nullptr;
    if (!fp) fp = (PFN_vkCmdBindDescriptorSets)resolve("vkCmdBindDescriptorSets");
    if (!fp) { return; }
    fp(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkIndexType indexType) {
    static PFN_vkCmdBindIndexBuffer fp = nullptr;
    if (!fp) fp = (PFN_vkCmdBindIndexBuffer)resolve("vkCmdBindIndexBuffer");
    if (!fp) { return; }
    fp(commandBuffer, buffer, offset, indexType);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline) {
    static PFN_vkCmdBindPipeline fp = nullptr;
    if (!fp) fp = (PFN_vkCmdBindPipeline)resolve("vkCmdBindPipeline");
    if (!fp) { return; }
    fp(commandBuffer, pipelineBindPoint, pipeline);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount, const VkBuffer* pBuffers, const VkDeviceSize* pOffsets) {
    static PFN_vkCmdBindVertexBuffers fp = nullptr;
    if (!fp) fp = (PFN_vkCmdBindVertexBuffers)resolve("vkCmdBindVertexBuffers");
    if (!fp) { return; }
    fp(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageBlit* pRegions, VkFilter filter) {
    static PFN_vkCmdBlitImage fp = nullptr;
    if (!fp) fp = (PFN_vkCmdBlitImage)resolve("vkCmdBlitImage");
    if (!fp) { return; }
    fp(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
}
VKAPI_ATTR void VKAPI_CALL vkCmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount, const VkClearAttachment* pAttachments, uint32_t rectCount, const VkClearRect* pRects) {
    static PFN_vkCmdClearAttachments fp = nullptr;
    if (!fp) fp = (PFN_vkCmdClearAttachments)resolve("vkCmdClearAttachments");
    if (!fp) { return; }
    fp(commandBuffer, attachmentCount, pAttachments, rectCount, pRects);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions) {
    static PFN_vkCmdCopyBuffer fp = nullptr;
    if (!fp) fp = (PFN_vkCmdCopyBuffer)resolve("vkCmdCopyBuffer");
    if (!fp) { return; }
    fp(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy* pRegions) {
    static PFN_vkCmdCopyBufferToImage fp = nullptr;
    if (!fp) fp = (PFN_vkCmdCopyBufferToImage)resolve("vkCmdCopyBufferToImage");
    if (!fp) { return; }
    fp(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageCopy* pRegions) {
    static PFN_vkCmdCopyImage fp = nullptr;
    if (!fp) fp = (PFN_vkCmdCopyImage)resolve("vkCmdCopyImage");
    if (!fp) { return; }
    fp(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferImageCopy* pRegions) {
    static PFN_vkCmdCopyImageToBuffer fp = nullptr;
    if (!fp) fp = (PFN_vkCmdCopyImageToBuffer)resolve("vkCmdCopyImageToBuffer");
    if (!fp) { return; }
    fp(commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    static PFN_vkCmdDispatch fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDispatch)resolve("vkCmdDispatch");
    if (!fp) { return; }
    fp(commandBuffer, groupCountX, groupCountY, groupCountZ);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset) {
    static PFN_vkCmdDispatchIndirect fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDispatchIndirect)resolve("vkCmdDispatchIndirect");
    if (!fp) { return; }
    fp(commandBuffer, buffer, offset);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
    static PFN_vkCmdDraw fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDraw)resolve("vkCmdDraw");
    if (!fp) { return; }
    fp(commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    static PFN_vkCmdDrawIndexed fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDrawIndexed)resolve("vkCmdDrawIndexed");
    if (!fp) { return; }
    fp(commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride) {
    static PFN_vkCmdDrawIndexedIndirect fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDrawIndexedIndirect)resolve("vkCmdDrawIndexedIndirect");
    if (!fp) { return; }
    fp(commandBuffer, buffer, offset, drawCount, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride) {
    static PFN_vkCmdDrawIndexedIndirectCount fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDrawIndexedIndirectCount)resolve("vkCmdDrawIndexedIndirectCount");
    if (!fp) { return; }
    fp(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride) {
    static PFN_vkCmdDrawIndexedIndirectCountKHR fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDrawIndexedIndirectCountKHR)resolve("vkCmdDrawIndexedIndirectCountKHR");
    if (!fp) { return; }
    fp(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride) {
    static PFN_vkCmdDrawIndirect fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDrawIndirect)resolve("vkCmdDrawIndirect");
    if (!fp) { return; }
    fp(commandBuffer, buffer, offset, drawCount, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride) {
    static PFN_vkCmdDrawIndirectCount fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDrawIndirectCount)resolve("vkCmdDrawIndirectCount");
    if (!fp) { return; }
    fp(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride) {
    static PFN_vkCmdDrawIndirectCountKHR fp = nullptr;
    if (!fp) fp = (PFN_vkCmdDrawIndirectCountKHR)resolve("vkCmdDrawIndirectCountKHR");
    if (!fp) { return; }
    fp(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdEndQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query) {
    static PFN_vkCmdEndQuery fp = nullptr;
    if (!fp) fp = (PFN_vkCmdEndQuery)resolve("vkCmdEndQuery");
    if (!fp) { return; }
    fp(commandBuffer, queryPool, query);
}
VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass(VkCommandBuffer commandBuffer) {
    static PFN_vkCmdEndRenderPass fp = nullptr;
    if (!fp) fp = (PFN_vkCmdEndRenderPass)resolve("vkCmdEndRenderPass");
    if (!fp) { return; }
    fp(commandBuffer);
}
VKAPI_ATTR void VKAPI_CALL vkCmdEndRendering(VkCommandBuffer commandBuffer) {
    static PFN_vkCmdEndRendering fp = nullptr;
    if (!fp) fp = (PFN_vkCmdEndRendering)resolve("vkCmdEndRendering");
    if (!fp) { return; }
    fp(commandBuffer);
}
VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderingKHR(VkCommandBuffer commandBuffer) {
    static PFN_vkCmdEndRenderingKHR fp = nullptr;
    if (!fp) fp = (PFN_vkCmdEndRenderingKHR)resolve("vkCmdEndRenderingKHR");
    if (!fp) { return; }
    fp(commandBuffer);
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers) {
    static PFN_vkCmdPipelineBarrier fp = nullptr;
    if (!fp) fp = (PFN_vkCmdPipelineBarrier)resolve("vkCmdPipelineBarrier");
    if (!fp) { return; }
    fp(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
}
VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues) {
    static PFN_vkCmdPushConstants fp = nullptr;
    if (!fp) fp = (PFN_vkCmdPushConstants)resolve("vkCmdPushConstants");
    if (!fp) { return; }
    fp(commandBuffer, layout, stageFlags, offset, size, pValues);
}
VKAPI_ATTR void VKAPI_CALL vkCmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount) {
    static PFN_vkCmdResetQueryPool fp = nullptr;
    if (!fp) fp = (PFN_vkCmdResetQueryPool)resolve("vkCmdResetQueryPool");
    if (!fp) { return; }
    fp(commandBuffer, queryPool, firstQuery, queryCount);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetBlendConstants(VkCommandBuffer commandBuffer, const float blendConstants[4]) {
    static PFN_vkCmdSetBlendConstants fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetBlendConstants)resolve("vkCmdSetBlendConstants");
    if (!fp) { return; }
    fp(commandBuffer, blendConstants);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetCullMode(VkCommandBuffer commandBuffer, VkCullModeFlags cullMode) {
    static PFN_vkCmdSetCullMode fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetCullMode)resolve("vkCmdSetCullMode");
    if (!fp) { return; }
    fp(commandBuffer, cullMode);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetCullModeEXT(VkCommandBuffer commandBuffer, VkCullModeFlags cullMode) {
    static PFN_vkCmdSetCullModeEXT fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetCullModeEXT)resolve("vkCmdSetCullModeEXT");
    if (!fp) { return; }
    fp(commandBuffer, cullMode);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBias(VkCommandBuffer commandBuffer, float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) {
    static PFN_vkCmdSetDepthBias fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetDepthBias)resolve("vkCmdSetDepthBias");
    if (!fp) { return; }
    fp(commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthCompareOp(VkCommandBuffer commandBuffer, VkCompareOp depthCompareOp) {
    static PFN_vkCmdSetDepthCompareOp fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetDepthCompareOp)resolve("vkCmdSetDepthCompareOp");
    if (!fp) { return; }
    fp(commandBuffer, depthCompareOp);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthCompareOpEXT(VkCommandBuffer commandBuffer, VkCompareOp depthCompareOp) {
    static PFN_vkCmdSetDepthCompareOpEXT fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetDepthCompareOpEXT)resolve("vkCmdSetDepthCompareOpEXT");
    if (!fp) { return; }
    fp(commandBuffer, depthCompareOp);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthTestEnable(VkCommandBuffer commandBuffer, VkBool32 depthTestEnable) {
    static PFN_vkCmdSetDepthTestEnable fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetDepthTestEnable)resolve("vkCmdSetDepthTestEnable");
    if (!fp) { return; }
    fp(commandBuffer, depthTestEnable);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthTestEnableEXT(VkCommandBuffer commandBuffer, VkBool32 depthTestEnable) {
    static PFN_vkCmdSetDepthTestEnableEXT fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetDepthTestEnableEXT)resolve("vkCmdSetDepthTestEnableEXT");
    if (!fp) { return; }
    fp(commandBuffer, depthTestEnable);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthWriteEnable(VkCommandBuffer commandBuffer, VkBool32 depthWriteEnable) {
    static PFN_vkCmdSetDepthWriteEnable fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetDepthWriteEnable)resolve("vkCmdSetDepthWriteEnable");
    if (!fp) { return; }
    fp(commandBuffer, depthWriteEnable);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthWriteEnableEXT(VkCommandBuffer commandBuffer, VkBool32 depthWriteEnable) {
    static PFN_vkCmdSetDepthWriteEnableEXT fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetDepthWriteEnableEXT)resolve("vkCmdSetDepthWriteEnableEXT");
    if (!fp) { return; }
    fp(commandBuffer, depthWriteEnable);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetFrontFace(VkCommandBuffer commandBuffer, VkFrontFace frontFace) {
    static PFN_vkCmdSetFrontFace fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetFrontFace)resolve("vkCmdSetFrontFace");
    if (!fp) { return; }
    fp(commandBuffer, frontFace);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetFrontFaceEXT(VkCommandBuffer commandBuffer, VkFrontFace frontFace) {
    static PFN_vkCmdSetFrontFaceEXT fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetFrontFaceEXT)resolve("vkCmdSetFrontFaceEXT");
    if (!fp) { return; }
    fp(commandBuffer, frontFace);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(VkCommandBuffer commandBuffer, uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors) {
    static PFN_vkCmdSetScissor fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetScissor)resolve("vkCmdSetScissor");
    if (!fp) { return; }
    fp(commandBuffer, firstScissor, scissorCount, pScissors);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetViewport(VkCommandBuffer commandBuffer, uint32_t firstViewport, uint32_t viewportCount, const VkViewport* pViewports) {
    static PFN_vkCmdSetViewport fp = nullptr;
    if (!fp) fp = (PFN_vkCmdSetViewport)resolve("vkCmdSetViewport");
    if (!fp) { return; }
    fp(commandBuffer, firstViewport, viewportCount, pViewports);
}
VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage, VkQueryPool queryPool, uint32_t query) {
    static PFN_vkCmdWriteTimestamp fp = nullptr;
    if (!fp) fp = (PFN_vkCmdWriteTimestamp)resolve("vkCmdWriteTimestamp");
    if (!fp) { return; }
    fp(commandBuffer, pipelineStage, queryPool, query);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateAndroidSurfaceKHR(VkInstance instance, const VkAndroidSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) {
    static PFN_vkCreateAndroidSurfaceKHR fp = nullptr;
    if (!fp) fp = (PFN_vkCreateAndroidSurfaceKHR)resolve("vkCreateAndroidSurfaceKHR");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(instance, pCreateInfo, pAllocator, pSurface);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer) {
    static PFN_vkCreateBuffer fp = nullptr;
    if (!fp) fp = (PFN_vkCreateBuffer)resolve("vkCreateBuffer");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pBuffer);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool) {
    static PFN_vkCreateCommandPool fp = nullptr;
    if (!fp) fp = (PFN_vkCreateCommandPool)resolve("vkCreateCommandPool");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pCommandPool);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
    static PFN_vkCreateComputePipelines fp = nullptr;
    if (!fp) fp = (PFN_vkCreateComputePipelines)resolve("vkCreateComputePipelines");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pMessenger) {
    static PFN_vkCreateDebugUtilsMessengerEXT fp = nullptr;
    if (!fp) fp = (PFN_vkCreateDebugUtilsMessengerEXT)resolve("vkCreateDebugUtilsMessengerEXT");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(instance, pCreateInfo, pAllocator, pMessenger);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool) {
    static PFN_vkCreateDescriptorPool fp = nullptr;
    if (!fp) fp = (PFN_vkCreateDescriptorPool)resolve("vkCreateDescriptorPool");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pDescriptorPool);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout) {
    static PFN_vkCreateDescriptorSetLayout fp = nullptr;
    if (!fp) fp = (PFN_vkCreateDescriptorSetLayout)resolve("vkCreateDescriptorSetLayout");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pSetLayout);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    static PFN_vkCreateDevice fp = nullptr;
    if (!fp) fp = (PFN_vkCreateDevice)resolve("vkCreateDevice");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    VkResult r = fp(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (r == VK_SUCCESS && pDevice && *pDevice) note_device(*pDevice);
    return r;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice device, const VkFenceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence) {
    static PFN_vkCreateFence fp = nullptr;
    if (!fp) fp = (PFN_vkCreateFence)resolve("vkCreateFence");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pFence);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFramebuffer* pFramebuffer) {
    static PFN_vkCreateFramebuffer fp = nullptr;
    if (!fp) fp = (PFN_vkCreateFramebuffer)resolve("vkCreateFramebuffer");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pFramebuffer);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkGraphicsPipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
    static PFN_vkCreateGraphicsPipelines fp = nullptr;
    if (!fp) fp = (PFN_vkCreateGraphicsPipelines)resolve("vkCreateGraphicsPipelines");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice device, const VkImageCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImage* pImage) {
    static PFN_vkCreateImage fp = nullptr;
    if (!fp) fp = (PFN_vkCreateImage)resolve("vkCreateImage");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pImage);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice device, const VkImageViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImageView* pView) {
    static PFN_vkCreateImageView fp = nullptr;
    if (!fp) fp = (PFN_vkCreateImageView)resolve("vkCreateImageView");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pView);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance) {
    static PFN_vkCreateInstance fp = nullptr;
    if (!fp) fp = (PFN_vkCreateInstance)resolve("vkCreateInstance");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    VkResult r = fp(pCreateInfo, pAllocator, pInstance);
    if (r == VK_SUCCESS && pInstance && *pInstance) note_instance(*pInstance);
    return r;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(VkDevice device, const VkPipelineCacheCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineCache* pPipelineCache) {
    static PFN_vkCreatePipelineCache fp = nullptr;
    if (!fp) fp = (PFN_vkCreatePipelineCache)resolve("vkCreatePipelineCache");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pPipelineCache);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout) {
    static PFN_vkCreatePipelineLayout fp = nullptr;
    if (!fp) fp = (PFN_vkCreatePipelineLayout)resolve("vkCreatePipelineLayout");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pPipelineLayout);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool) {
    static PFN_vkCreateQueryPool fp = nullptr;
    if (!fp) fp = (PFN_vkCreateQueryPool)resolve("vkCreateQueryPool");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pQueryPool);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass) {
    static PFN_vkCreateRenderPass fp = nullptr;
    if (!fp) fp = (PFN_vkCreateRenderPass)resolve("vkCreateRenderPass");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pRenderPass);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler) {
    static PFN_vkCreateSampler fp = nullptr;
    if (!fp) fp = (PFN_vkCreateSampler)resolve("vkCreateSampler");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pSampler);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSemaphore* pSemaphore) {
    static PFN_vkCreateSemaphore fp = nullptr;
    if (!fp) fp = (PFN_vkCreateSemaphore)resolve("vkCreateSemaphore");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pSemaphore);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule) {
    static PFN_vkCreateShaderModule fp = nullptr;
    if (!fp) fp = (PFN_vkCreateShaderModule)resolve("vkCreateShaderModule");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pShaderModule);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) {
    static PFN_vkCreateSwapchainKHR fp = nullptr;
    if (!fp) fp = (PFN_vkCreateSwapchainKHR)resolve("vkCreateSwapchainKHR");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, pCreateInfo, pAllocator, pSwapchain);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyBuffer fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyBuffer)resolve("vkDestroyBuffer");
    if (!fp) { return; }
    fp(device, buffer, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyCommandPool fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyCommandPool)resolve("vkDestroyCommandPool");
    if (!fp) { return; }
    fp(device, commandPool, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyDescriptorPool fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyDescriptorPool)resolve("vkDestroyDescriptorPool");
    if (!fp) { return; }
    fp(device, descriptorPool, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyDescriptorSetLayout fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyDescriptorSetLayout)resolve("vkDestroyDescriptorSetLayout");
    if (!fp) { return; }
    fp(device, descriptorSetLayout, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyDevice fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyDevice)resolve("vkDestroyDevice");
    clear_device();
    if (!fp) { return; }
    fp(device, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice device, VkFence fence, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyFence fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyFence)resolve("vkDestroyFence");
    if (!fp) { return; }
    fp(device, fence, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyFramebuffer(VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyFramebuffer fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyFramebuffer)resolve("vkDestroyFramebuffer");
    if (!fp) { return; }
    fp(device, framebuffer, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyImage fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyImage)resolve("vkDestroyImage");
    if (!fp) { return; }
    fp(device, image, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyRenderPass(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyRenderPass fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyRenderPass)resolve("vkDestroyRenderPass");
    if (!fp) { return; }
    fp(device, renderPass, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice device, VkImageView imageView, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyImageView fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyImageView)resolve("vkDestroyImageView");
    if (!fp) { return; }
    fp(device, imageView, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyInstance fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyInstance)resolve("vkDestroyInstance");
    clear_instance();
    if (!fp) { return; }
    fp(instance, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyPipeline fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyPipeline)resolve("vkDestroyPipeline");
    if (!fp) { return; }
    fp(device, pipeline, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineCache(VkDevice device, VkPipelineCache pipelineCache, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyPipelineCache fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyPipelineCache)resolve("vkDestroyPipelineCache");
    if (!fp) { return; }
    fp(device, pipelineCache, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyPipelineLayout fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyPipelineLayout)resolve("vkDestroyPipelineLayout");
    if (!fp) { return; }
    fp(device, pipelineLayout, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyQueryPool fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyQueryPool)resolve("vkDestroyQueryPool");
    if (!fp) { return; }
    fp(device, queryPool, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroySampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroySampler fp = nullptr;
    if (!fp) fp = (PFN_vkDestroySampler)resolve("vkDestroySampler");
    if (!fp) { return; }
    fp(device, sampler, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroySemaphore fp = nullptr;
    if (!fp) fp = (PFN_vkDestroySemaphore)resolve("vkDestroySemaphore");
    if (!fp) { return; }
    fp(device, semaphore, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroyShaderModule fp = nullptr;
    if (!fp) fp = (PFN_vkDestroyShaderModule)resolve("vkDestroyShaderModule");
    if (!fp) { return; }
    fp(device, shaderModule, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroySurfaceKHR(VkInstance instance, VkSurfaceKHR surface, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroySurfaceKHR fp = nullptr;
    if (!fp) fp = (PFN_vkDestroySurfaceKHR)resolve("vkDestroySurfaceKHR");
    if (!fp) { return; }
    fp(instance, surface, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkDestroySwapchainKHR fp = nullptr;
    if (!fp) fp = (PFN_vkDestroySwapchainKHR)resolve("vkDestroySwapchainKHR");
    if (!fp) { return; }
    fp(device, swapchain, pAllocator);
}
VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice device) {
    static PFN_vkDeviceWaitIdle fp = nullptr;
    if (!fp) fp = (PFN_vkDeviceWaitIdle)resolve("vkDeviceWaitIdle");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device);
}
VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer commandBuffer) {
    static PFN_vkEndCommandBuffer fp = nullptr;
    if (!fp) fp = (PFN_vkEndCommandBuffer)resolve("vkEndCommandBuffer");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(commandBuffer);
}
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties) {
    static PFN_vkEnumerateDeviceExtensionProperties fp = nullptr;
    if (!fp) fp = (PFN_vkEnumerateDeviceExtensionProperties)resolve("vkEnumerateDeviceExtensionProperties");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(physicalDevice, pLayerName, pPropertyCount, pProperties);
}
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties) {
    static PFN_vkEnumerateInstanceExtensionProperties fp = nullptr;
    if (!fp) fp = (PFN_vkEnumerateInstanceExtensionProperties)resolve("vkEnumerateInstanceExtensionProperties");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(pLayerName, pPropertyCount, pProperties);
}
VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices) {
    static PFN_vkEnumeratePhysicalDevices fp = nullptr;
    if (!fp) fp = (PFN_vkEnumeratePhysicalDevices)resolve("vkEnumeratePhysicalDevices");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(instance, pPhysicalDeviceCount, pPhysicalDevices);
}
VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers) {
    static PFN_vkFreeCommandBuffers fp = nullptr;
    if (!fp) fp = (PFN_vkFreeCommandBuffers)resolve("vkFreeCommandBuffers");
    if (!fp) { return; }
    fp(device, commandPool, commandBufferCount, pCommandBuffers);
}
VKAPI_ATTR VkResult VKAPI_CALL vkFreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets) {
    static PFN_vkFreeDescriptorSets fp = nullptr;
    if (!fp) fp = (PFN_vkFreeDescriptorSets)resolve("vkFreeDescriptorSets");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, descriptorPool, descriptorSetCount, pDescriptorSets);
}
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator) {
    static PFN_vkFreeMemory fp = nullptr;
    if (!fp) fp = (PFN_vkFreeMemory)resolve("vkFreeMemory");
    if (!fp) { return; }
    fp(device, memory, pAllocator);
}
VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemoryRequirements) {
    static PFN_vkGetBufferMemoryRequirements fp = nullptr;
    if (!fp) fp = (PFN_vkGetBufferMemoryRequirements)resolve("vkGetBufferMemoryRequirements");
    if (!fp) { return; }
    fp(device, buffer, pMemoryRequirements);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    static PFN_vkGetDeviceProcAddr fp = nullptr;
    if (!fp) fp = (PFN_vkGetDeviceProcAddr)resolve("vkGetDeviceProcAddr");
    if (!fp) { return (PFN_vkVoidFunction)0; }
    return fp(device, pName);
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue) {
    static PFN_vkGetDeviceQueue fp = nullptr;
    if (!fp) fp = (PFN_vkGetDeviceQueue)resolve("vkGetDeviceQueue");
    if (!fp) { return; }
    fp(device, queueFamilyIndex, queueIndex, pQueue);
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice device, VkFence fence) {
    static PFN_vkGetFenceStatus fp = nullptr;
    if (!fp) fp = (PFN_vkGetFenceStatus)resolve("vkGetFenceStatus");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, fence);
}
VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice device, VkImage image, VkMemoryRequirements* pMemoryRequirements) {
    static PFN_vkGetImageMemoryRequirements fp = nullptr;
    if (!fp) fp = (PFN_vkGetImageMemoryRequirements)resolve("vkGetImageMemoryRequirements");
    if (!fp) { return; }
    fp(device, image, pMemoryRequirements);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    static PFN_vkGetInstanceProcAddr fp = nullptr;
    if (!fp) fp = (PFN_vkGetInstanceProcAddr)resolve("vkGetInstanceProcAddr");
    if (!fp) { return (PFN_vkVoidFunction)0; }
    return fp(instance, pName);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures* pFeatures) {
    static PFN_vkGetPhysicalDeviceFeatures fp = nullptr;
    if (!fp) fp = (PFN_vkGetPhysicalDeviceFeatures)resolve("vkGetPhysicalDeviceFeatures");
    if (!fp) { return; }
    fp(physicalDevice, pFeatures);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures) {
    static PFN_vkGetPhysicalDeviceFeatures2 fp = nullptr;
    if (!fp) fp = (PFN_vkGetPhysicalDeviceFeatures2)resolve("vkGetPhysicalDeviceFeatures2");
    if (!fp) { return; }
    fp(physicalDevice, pFeatures);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties* pFormatProperties) {
    static PFN_vkGetPhysicalDeviceFormatProperties fp = nullptr;
    if (!fp) fp = (PFN_vkGetPhysicalDeviceFormatProperties)resolve("vkGetPhysicalDeviceFormatProperties");
    if (!fp) { return; }
    fp(physicalDevice, format, pFormatProperties);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties) {
    static PFN_vkGetPhysicalDeviceMemoryProperties fp = nullptr;
    if (!fp) fp = (PFN_vkGetPhysicalDeviceMemoryProperties)resolve("vkGetPhysicalDeviceMemoryProperties");
    if (!fp) { return; }
    fp(physicalDevice, pMemoryProperties);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties* pProperties) {
    static PFN_vkGetPhysicalDeviceProperties fp = nullptr;
    if (!fp) fp = (PFN_vkGetPhysicalDeviceProperties)resolve("vkGetPhysicalDeviceProperties");
    if (!fp) { return; }
    fp(physicalDevice, pProperties);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties) {
    static PFN_vkGetPhysicalDeviceQueueFamilyProperties fp = nullptr;
    if (!fp) fp = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)resolve("vkGetPhysicalDeviceQueueFamilyProperties");
    if (!fp) { return; }
    fp(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR* pSurfaceCapabilities) {
    static PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR fp = nullptr;
    if (!fp) fp = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)resolve("vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(physicalDevice, surface, pSurfaceCapabilities);
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pSurfaceFormatCount, VkSurfaceFormatKHR* pSurfaceFormats) {
    static PFN_vkGetPhysicalDeviceSurfaceFormatsKHR fp = nullptr;
    if (!fp) fp = (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)resolve("vkGetPhysicalDeviceSurfaceFormatsKHR");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats);
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pPresentModeCount, VkPresentModeKHR* pPresentModes) {
    static PFN_vkGetPhysicalDeviceSurfacePresentModesKHR fp = nullptr;
    if (!fp) fp = (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)resolve("vkGetPhysicalDeviceSurfacePresentModesKHR");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(physicalDevice, surface, pPresentModeCount, pPresentModes);
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags) {
    static PFN_vkGetQueryPoolResults fp = nullptr;
    if (!fp) fp = (PFN_vkGetQueryPoolResults)resolve("vkGetQueryPoolResults");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages) {
    static PFN_vkGetSwapchainImagesKHR fp = nullptr;
    if (!fp) fp = (PFN_vkGetSwapchainImagesKHR)resolve("vkGetSwapchainImagesKHR");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, swapchain, pSwapchainImageCount, pSwapchainImages);
}
VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** ppData) {
    static PFN_vkMapMemory fp = nullptr;
    if (!fp) fp = (PFN_vkMapMemory)resolve("vkMapMemory");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, memory, offset, size, flags, ppData);
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
    static PFN_vkQueuePresentKHR fp = nullptr;
    if (!fp) fp = (PFN_vkQueuePresentKHR)resolve("vkQueuePresentKHR");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(queue, pPresentInfo);
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence) {
    static PFN_vkQueueSubmit fp = nullptr;
    if (!fp) fp = (PFN_vkQueueSubmit)resolve("vkQueueSubmit");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(queue, submitCount, pSubmits, fence);
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue queue) {
    static PFN_vkQueueWaitIdle fp = nullptr;
    if (!fp) fp = (PFN_vkQueueWaitIdle)resolve("vkQueueWaitIdle");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(queue);
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
    static PFN_vkResetCommandBuffer fp = nullptr;
    if (!fp) fp = (PFN_vkResetCommandBuffer)resolve("vkResetCommandBuffer");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(commandBuffer, flags);
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorPoolResetFlags flags) {
    static PFN_vkResetDescriptorPool fp = nullptr;
    if (!fp) fp = (PFN_vkResetDescriptorPool)resolve("vkResetDescriptorPool");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, descriptorPool, flags);
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences) {
    static PFN_vkResetFences fp = nullptr;
    if (!fp) fp = (PFN_vkResetFences)resolve("vkResetFences");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, fenceCount, pFences);
}
VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice device, VkDeviceMemory memory) {
    static PFN_vkUnmapMemory fp = nullptr;
    if (!fp) fp = (PFN_vkUnmapMemory)resolve("vkUnmapMemory");
    if (!fp) { return; }
    fp(device, memory);
}
VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies) {
    static PFN_vkUpdateDescriptorSets fp = nullptr;
    if (!fp) fp = (PFN_vkUpdateDescriptorSets)resolve("vkUpdateDescriptorSets");
    if (!fp) { return; }
    fp(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
}
VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout) {
    static PFN_vkWaitForFences fp = nullptr;
    if (!fp) fp = (PFN_vkWaitForFences)resolve("vkWaitForFences");
    if (!fp) { return VK_ERROR_UNKNOWN; }
    return fp(device, fenceCount, pFences, waitAll, timeout);
}
} // extern "C"

#endif // __ANDROID__
