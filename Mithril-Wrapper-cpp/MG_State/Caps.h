// Mithril-Wrapper - MG_State/Caps.h
// Single source of truth for the GL version / GLSL version / extension set we
// advertise to the host. Modelled on MobileGL's RendererGLInfo (TargetGLVersion
// + dynamic Extensions): NEVER hardcode a version in the getters. The host is
// told ONLY what is actually implemented, so applications like Minecraft take
// code paths we can service instead of silently sampling undefined resources
// (which renders pure red).
#pragma once
#include <string>
#include <vector>

namespace mithril {

struct Caps {
    int gl_major   = 3;
    int gl_minor   = 3;
    int glsl_major = 3;
    int glsl_minor = 30;   // GLSL 330 (matches GL 3.3)
};

// Process-wide capability set. Safe to call from any thread after init.
const Caps& caps();

// "OpenGL <major>.<minor>.0 Mithril-Wrapper ..." built from caps().
const std::string& version_string();
const std::string& glsl_version_string();

// Extensions we actually implement (curated; unsupported entries removed).
const std::vector<const char*>& extensions();
bool has_extension(const char* name);

} // namespace mithril
