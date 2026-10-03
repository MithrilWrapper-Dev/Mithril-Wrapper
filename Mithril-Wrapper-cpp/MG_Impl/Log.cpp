// Mithril-Wrapper - MG_Impl/Log.cpp
// Logging implementation (stderr-based). Ported from the former gl/log.cpp.
#include "Log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if defined(__ANDROID__) || defined(__linux__) || defined(__APPLE__)
#define MITHRIL_CRASH_HANDLER 1
#include <signal.h>
#include <unistd.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <initializer_list>
#endif

namespace mithril {

namespace {
LogLevel g_level = LogLevel::Warning;

const char* level_str(LogLevel l) {
    switch (l) {
        case LogLevel::Verbose: return "V";
        case LogLevel::Debug:   return "D";
        case LogLevel::Info:    return "I";
        case LogLevel::Warning: return "W";
        case LogLevel::Error:   return "E";
    }
    return "?";
}

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}
} // namespace

void log_set_level(LogLevel level) { g_level = level; }
LogLevel log_get_level() { return g_level; }

void log_write(LogLevel level, const char* tag, const char* fmt, ...) {
    if (static_cast<int>(level) < static_cast<int>(g_level)) return;

    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    std::tm tm{};
    localtime_r(&ts.tv_sec, &tm);

    std::fprintf(stderr, "[mithril %s %02d:%02d:%02d.%03ld %s] ",
                 level_str(level), tm.tm_hour, tm.tm_min, tm.tm_sec,
                 ts.tv_nsec / 1000000, tag ? tag : "");

    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);

    std::fputc('\n', stderr);
}

// ---- Crash location ------------------------------------------------------
//
// stdout/stderr is unbuffered (set below), so anything written before the
// fault is already on the wire. What has been missing is the fault itself:
// without a handler a SIGSEGV kills the process silently, and the log simply
// stops at whatever the last line happened to be. Every round of this
// investigation then had to infer the crash site from that last line, which
// is guesswork dressed up as diagnosis.
//
// Install a handler that names the signal, the faulting address and a symbol
// resolved backtrace, so the next log says where it died instead of implying
// it. Async-signal-safety: only write(2) and the execinfo entry points are
// used; backtrace() itself is documented as unsafe on some implementations,
// which is an accepted risk for a process that is about to die anyway.
#if defined(MITHRIL_CRASH_HANDLER)
// execinfo.h declares backtrace() unconditionally, but bionic has not always
// shipped an implementation (libunwind was removed from the NDK), and on such
// a target a strong reference is a link error in a module that exists purely
// to make failures legible. Re-declare it weak there and probe at run time: a
// missing implementation degrades to "signal + fault address", which is still
// enough to locate the fault, instead of taking the whole build down.
#if defined(__ANDROID__)
extern "C" int backtrace(void** buffer, int size) __attribute__((weak));
#endif

namespace crash {
constexpr int kMaxFrames = 48;

// Whoever owned each signal before we installed ourselves.
//
// A JVM installs handlers for SIGSEGV / SIGBUS / SIGFPE and uses them as
// ordinary control flow: implicit null checks, stack banging, safepoint
// polling. Those fire constantly and are not crashes. Taking them over and
// calling _exit() kills the process the first time Java raises one - which is
// exactly how the macOS / Minecraft E2E started failing at this change: the
// reported stack has JavaCalls::call_helper and JIT-region frames on top, with
// no Mithril frame anywhere in it. So record the previous disposition and hand
// the signal on once we have printed what we need.
struct SigSlot {
    int signo;
    struct sigaction prev;
};
static SigSlot g_slots[] = {
    {SIGSEGV, {}}, {SIGBUS, {}}, {SIGILL, {}}, {SIGABRT, {}}, {SIGFPE, {}},
};
constexpr std::size_t kSlotCount = sizeof(g_slots) / sizeof(g_slots[0]);

void write_str(const char* s) {
    if (!s) return;
    std::size_t n = std::strlen(s);
    while (n > 0) {
        ssize_t w = ::write(2, s, n);
        if (w <= 0) break;
        s += static_cast<std::size_t>(w);
        n -= static_cast<std::size_t>(w);
    }
}

void handler(int sig, siginfo_t* info, void* context) {
    (void)context;
    char buf[256];

    std::snprintf(buf, sizeof(buf),
                  "\n[mithril] FATAL: signal %d (%s), fault addr %p\n", sig,
                  sig == SIGSEGV ? "SIGSEGV"
                  : sig == SIGBUS ? "SIGBUS"
                  : sig == SIGILL  ? "SIGILL"
                  : sig == SIGABRT ? "SIGABRT"
                  : sig == SIGFPE  ? "SIGFPE" : "other",
                  info ? info->si_addr : nullptr);
    write_str(buf);

    void* frames[kMaxFrames];
    int n = 0;
    if (backtrace) n = backtrace(frames, kMaxFrames);
    if (n == 0) write_str("[mithril]   (no backtrace available on this target)\n");
    for (int i = 0; i < n; ++i) {
        Dl_info d{};
        const char* sym = nullptr;
        void* sym_addr = nullptr;
        if (::dladdr(frames[i], &d) && d.dli_sname) {
            sym = d.dli_sname;
            sym_addr = d.dli_saddr;
        }
        std::snprintf(buf, sizeof(buf), "[mithril]   #%02d %p %s", i, frames[i],
                      d.dli_fname ? d.dli_fname : "??");
        write_str(buf);
        if (sym) {
            std::snprintf(buf, sizeof(buf), " (%s+0x%lx)\n", sym,
                          static_cast<unsigned long>(
                              static_cast<const char*>(frames[i]) -
                              static_cast<const char*>(sym_addr)));
        } else {
            std::snprintf(buf, sizeof(buf), " (+0x%lx)\n",
                          static_cast<unsigned long>(
                              static_cast<const char*>(frames[i]) -
                              static_cast<const char*>(d.dli_fbase)));
        }
        write_str(buf);
    }
    write_str("[mithril] end of backtrace\n");

    // Hand the signal to the handler we displaced, if there was one, and let it
    // decide what the signal means. Only when nobody owned the signal do we
    // treat it as fatal: restore the default disposition and re-raise, so the
    // platform produces its own crash report rather than a silent _exit().
    for (std::size_t i = 0; i < kSlotCount; ++i) {
        if (g_slots[i].signo != sig) continue;
        const struct sigaction& prev = g_slots[i].prev;
        if ((prev.sa_flags & SA_SIGINFO) && prev.sa_sigaction &&
            prev.sa_sigaction != &handler) {
            prev.sa_sigaction(sig, info, context);
            return;
        }
        if (!(prev.sa_flags & SA_SIGINFO) && prev.sa_handler &&
            prev.sa_handler != SIG_DFL && prev.sa_handler != SIG_IGN) {
            prev.sa_handler(sig);
            return;
        }
        break;
    }
    ::signal(sig, SIG_DFL);
    ::raise(sig);
    ::_exit(128 + sig);
}

struct Install {
    Install() {
        struct sigaction sa{};
        sa.sa_sigaction = &handler;
        sa.sa_flags = SA_SIGINFO;
        sigemptyset(&sa.sa_mask);
        // oldact is kept per signal so the handler can chain to it.
        for (std::size_t i = 0; i < kSlotCount; ++i)
            ::sigaction(g_slots[i].signo, &sa, &g_slots[i].prev);
    }
} g_crash_handler;
} // namespace crash
#endif // MITHRIL_CRASH_HANDLER

// Initialise level from environment on first use.
namespace {
struct LogLevelInit {
    LogLevelInit() {
        // Unbuffered. stderr is normally line-buffered on a tty and fully
        // buffered otherwise; the launcher captures it through a pipe, so it
        // is the latter. A SIGSEGV then discards everything written since the
        // last flush, which routinely hides the last few hundred lines - the
        // ones naming the call that actually crashed. Diagnostics that go
        // missing exactly when they matter are worse than useless.
        std::setvbuf(stderr, nullptr, _IONBF, 0);
        if (env_flag("MITHRIL_VERBOSE")) log_set_level(LogLevel::Verbose);
        else if (env_flag("MITHRIL_DEBUG")) log_set_level(LogLevel::Debug);
        else if (env_flag("MITHRIL_INFO")) log_set_level(LogLevel::Info);
    }
} g_log_init;
} // namespace

} // namespace mithril
