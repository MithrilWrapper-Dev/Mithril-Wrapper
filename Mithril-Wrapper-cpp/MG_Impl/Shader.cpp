// Mithril-Wrapper - MG_Impl/Shader.cpp
// GLSL (desktop Core Profile) -> Vulkan SPIR-V translation via glslang.
//
// Pipeline (single EShClientOpenGL input dialect; no Vulkan-client dialect):
//   1. Preprocess: inject MG_MITHRIL / MG_MITHRIL_VERSION macros so host
//      shaders can branch on the Mithril backend (mirrors MobileGlues'
//      MG_MOBILEGLUES injection). Upgrade GLSL versions below 330 (the Vulkan
//      GLSL minimum) so desktop GLSL 150 shaders like Minecraft's blit_screen
//      compile under the Vulkan client.
//   2. Inject layout(location=N) into vertex `in` declarations from
//      glBindAttribLocation mappings so the SPIR-V stage_input locations match
//      the application's vertex descriptor.
//   3. Normalize Vulkan-incompatible layout qualifiers: rewrite UBO
//      `layout(packed)` / `layout(shared)` to `layout(std140)` and SSBO
//      equivalents to `layout(std430)` so strict EShClientOpenGL +
//      EShMsgVulkanRules compiles them directly (previously saved by the
//      removed relaxed fallback).
//   4. Normalize GL legacy constructs: in fragment shaders using
//      `gl_FragColor`, inject a synthetic `layout(location = 0) out vec4
//      _mithril_FragColor;` declaration and rewrite all references (other GL
//      legacy constructs are added on demand; Minecraft core shaders use only
//      gl_FragColor).
//   5. Inject GL->Vulkan position fixups (Z remap always; Y flip when flip_y)
//      by wrapping main() (vertex shaders only).
//   6. Preprocess: fold loose non-opaque uniforms into a synthetic
//      `mithril_GlobalBlock` UBO (mirroring ANGLE's ANGLE_DefaultUniformBlock).
//      Handles precision qualifiers, multi-dimensional arrays, multiple
//      declarators, named and anonymous struct uniforms, and skips
//      declarations inside comments. The #define renames let the shader body
//      reference members by their original names without a block prefix.
//      This avoids the "non-opaque uniforms outside a block" error that some
//      glslang versions emit even with EShClientOpenGL + EShMsgVulkanRules.
//   7. Per-stage explicit binding injection for opaque uniforms and UBO
//      blocks (root cause AK fix: prevents VS/FS binding collisions).
//   8. glslang compiles the GLSL to Vulkan SPIR-V via the GL_KHR_vulkan_glsl
//      path: EShClientOpenGL + EShMsgVulkanRules. Two-level strict fallback:
//      wrapped source (after step 6) -> unwrapped source (pre-step-6 backup,
//      still normalized by steps 3-5). Both attempts use EShClientOpenGL. The
//      previous third-level relaxed fallback is removed — its coverage
//      (layout(packed/shared) UBOs, gl_FragColor) is replaced by steps 3+4.
//      Because step 6 already wrapped loose uniforms, glslang never needs the
//      auto-wrap path — it sees only block uniforms and opaque samplers. The
//      emitted SPIR-V stays Vulkan-conformant (MoltenVK accepts it). The
//      synthetic block name "mithril_GlobalBlock" does not match any GL
//      uniform name, so DescriptorSet.cpp falls through to member-by-member
//      packing (same $Global convention), which works identically.
//      setAutoMapLocations(true) + setAutoMapBindings(true) auto-assign any
//      remaining locations/bindings.
//   9. The SPIR-V words are returned directly — MoltenVK cross-translates
//      Vulkan SPIR-V to MSL internally at vkCreateShaderModule time, so no
//      SPIRV-Cross stage is needed here.
#include "Shader.h"
#include "Log.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>

#include <set>
#include <cstdint>
#include <mutex>
#include <regex>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mithril {
namespace {

struct GlslangInit {
    GlslangInit()  { glslang::InitializeProcess(); }
    ~GlslangInit() { /* process-lifetime; no finalize needed */ }
};
GlslangInit& glslang_init() {
    static GlslangInit g;
    return g;
}

EShLanguage to_esh_stage(GLenum gl) {
    switch (gl) {
        case GL_VERTEX_SHADER:          return EShLangVertex;
        case GL_FRAGMENT_SHADER:        return EShLangFragment;
        case GL_GEOMETRY_SHADER:        return EShLangGeometry;
        case GL_TESS_CONTROL_SHADER:    return EShLangTessControl;
        case GL_TESS_EVALUATION_SHADER: return EShLangTessEvaluation;
        case GL_COMPUTE_SHADER:         return EShLangCompute;
        default:                        return EShLangCount;
    }
}

// Extract the GLSL #version number. Returns -1 if not found.
int get_glsl_version(const std::string& src) {
    static std::regex version_pattern(R"(#version\s+(\d{3}))");
    std::smatch match;
    if (std::regex_search(src, match, version_pattern)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

/*
 * Ensure the GLSL source has a version usable by the Vulkan client. Vulkan
 * GLSL requires #version 330 minimum (GL_KHR_vulkan_glsl). 但本渲染器在
 * GLSL 源码层面注入 layout(binding=N) 来分离 VS/FS 的 descriptor binding
 * （见 inject_opaque_bindings），而 layout(binding=) 要到 #version 420 才
 * 进核心（GL_ARB_shading_language_420pack）。所以统一升到 420 core：
 *  - 满足 Vulkan GLSL 最低要求（330）
 *  - 让 layout(binding=) 合法（420）
 * Minecraft 的 blit_screen 用 #version 150，升到 420 只是放宽版本声明，不
 * 改变任何已写代码的语义。如果源码没有 #version，前置 #version 420 core。
 *
 * Returns the resolved GLSL version number.
 */
int ensure_glsl_version(std::string& src) {
    int ver = get_glsl_version(src);
    if (ver == -1) {
        ver = 420;
        src.insert(0, "#version 420 core\n");
        return ver;
    }
    if (ver < 420) {
        // Replace the existing #version line with #version 420 core. The
        // 'core' profile is required for Vulkan GLSL; 'compatibility' would
        // pull in deprecated fixed-function symbols that Vulkan rejects.
        size_t pos = src.find("#version");
        size_t line_end = src.find('\n', pos);
        if (line_end == std::string::npos) line_end = src.length();
        // Preserve any trailing profile token that was on the line by
        // replacing the whole line with the upgraded version + core.
        src.replace(pos, line_end - pos, "#version 420 core");
        ver = 420;
    } else {
        // Ensure a profile token is present; Vulkan GLSL requires core.
        size_t pos = src.find("#version");
        size_t line_end = src.find('\n', pos);
        if (line_end == std::string::npos) line_end = src.length();
        std::string line = src.substr(pos, line_end - pos);
        if (line.find("core") == std::string::npos &&
            line.find("compatibility") == std::string::npos &&
            line.find("es") == std::string::npos) {
            // No profile token; append ' core'.
            src.replace(pos, line_end - pos, line + " core");
        }
    }
    return ver;
}

/*
 * Rewrite desktop-GLSL built-in identifiers that Vulkan GLSL either does not
 * declare or declares under a different name. Without this rewrite, glslang
 * (configured with EShClientOpenGL + EShMsgVulkanRules) rejects shaders that
 * reference these identifiers with "'gl_VertexID' : undeclared identifier",
 * which crashes Minecraft 1.21's rendertype_lines vertex shader at startup.
 *
 * Mapping (as of the gl_VertexID baseVertex fix):
 *   gl_InstanceID   -> gl_InstanceIndex    (Vulkan GLSL builtin.)
 *
 *   gl_VertexID     -> NOT renamed here. It is instead handled by
 *                      inject_vertex_id_fixup(), which injects a
 *                      `#define gl_VertexID (gl_VertexIndex + _mbv._mithrilBaseVertex)`
 *                      macro into every vertex shader. That single macro
 *                      serves TWO purposes that a plain rename cannot:
 *                        1. Compilation: glslang's preprocessor expands the
 *                           macro to the Vulkan builtin `gl_VertexIndex`, so
 *                           the parser never sees a bare `gl_VertexID` token
 *                           (which is not a Vulkan builtin) -> the shader
 *                           compiles under Vulkan GLSL.
 *                        2. Semantic fidelity (root cause: baseVertex):
 *                           desktop GL defines gl_VertexID in an indexed draw
 *                           as (index + baseVertex) — it INCLUDES baseVertex.
 *                           Vulkan's gl_VertexIndex == the raw index value
 *                           and does NOT include vkCmdDrawIndexed's
 *                           vertexOffset (which only offsets vertex *fetch*,
 *                           not the shader-visible index). Adding
 *                           _mbv._mithrilBaseVertex (set == currentBaseVertex
 *                           on every draw) restores the GL semantics, so a
 *                           vertex shader that uses gl_VertexID for a lookup
 *                           (texture-array index, per-vertex SSBO fetch) is
 *                           no longer off by baseVertex under
 *                           glDrawElementsBaseVertex /
 *                           glDrawElementsInstancedBaseVertex.
 *
 *   The macro is injected into EVERY vertex shader (whether or not it uses
 *   gl_VertexID) and _mbv._mithrilBaseVertex is written to 0 on every draw
 *   whose baseVertex == 0, so gl_VertexID is ALWAYS defined and equals the
 *   correct GL value. This removes the old SEMANTIC MISMATCH described in
 *   the previous revision of this comment (the push-constant compensation is
 *   now implemented across Shader.cpp / Drawing.cpp / DescriptorSet.cpp /
 *   Pipeline.cpp / CommandStream.cpp).
 *
 *   gl_InstanceID note (unchanged): Vulkan's InstanceIndex is 0-based and
 *   does NOT include the firstInstance offset; desktop GL's InstanceID is
 *   1-based. Minecraft's rendertype_lines only uses gl_VertexID, so the
 *   InstanceID semantic shift (and the analogous baseInstance gap) is
 *   irrelevant for the shaders we currently see — left as a follow-up.
 *
 * The rewrite is word-boundary scoped (regex \b) so it does not touch
 * identifiers like myGl_InstanceID_foo. It also skips occurrences inside
 * string literals and line comments — though Minecraft's core shaders do not
 * embed those in expression contexts, this keeps the rewrite safe for
 * third-party shader packs.
 *
 * Only vertex shaders are affected; fragment/compute shaders do not reference
 * these builtins. (gl_FragCoord and friends are already Vulkan-compatible.)
 *
 * Reference: this mirrors the approach MobileGlues takes for the same class
 * of desktop-vs-Vulkan builtin mismatch.
 */
void rewrite_desktop_builtins(std::string& src, GLenum gl_stage) {
    if (gl_stage != GL_VERTEX_SHADER) return;
    // Word-boundary replace. gl_InstanceID -> gl_InstanceIndex. gl_VertexID is
    // deliberately NOT rewritten (it is handled by inject_vertex_id_fixup's
    // macro, which references the Vulkan builtin gl_VertexIndex directly).
    static const std::regex re(R"(\bgl_InstanceID\b)", std::regex::optimize);
    std::string out;
    out.reserve(src.size());
    std::string::const_iterator it = src.cbegin();
    std::smatch m;
    while (std::regex_search(it, src.cend(), m, re)) {
        out.append(it, m[0].first);
        out.append("gl_InstanceIndex");
        it = m[0].second;
    }
    out.append(it, src.cend());
    src = std::move(out);
}

/*
 * Inject a vertex-shader push-constant compensation block + gl_VertexID macro
 * (root cause: gl_VertexID baseVertex semantics).
 *
 * Background:
 *   Desktop GL's gl_VertexID in an indexed draw == index + baseVertex; Vulkan's
 *   gl_VertexIndex == the raw index (vkCmdDrawIndexed's vertexOffset does NOT
 *   feed the shader-visible index). Under glDrawElementsBaseVertex /
 *   glDrawElementsInstancedBaseVertex, a shader that uses gl_VertexID for a
 *   data lookup reads the wrong index (off by baseVertex) -> geometry
 *   misalignment -> garbled/red screen. This is exactly the gap documented in
 *   fix-minecraft-black-screen/spec.md ("gl_VertexID 语义保留" SHALL
 *   requirement) that was previously deferred.
 *
 * Injection (per vertex shader, both Y orientations — glsl_to_spirv runs once
 * per variant):
 *   layout(push_constant) uniform _MithrilBaseVertex {
 *       int _mithrilBaseVertex;
 *   } _mbv;
 *   #define gl_VertexID (gl_VertexIndex + _mbv._mithrilBaseVertex)
 *
 *   - The push-constant block uses the Vulkan `push_constant` storage class,
 *     which glslang accepts under EShClientOpenGL + EShMsgVulkanRules (verified
 *     against the pinned glslang). It is NOT a descriptor-backed uniform block,
 *     so SPIRV-Cross's reflect_stage() does NOT surface it as a UBO binding
 *     (it lives in res.push_constant_buffers, not res.uniform_buffers) — it
 *     cannot corrupt the descriptor layout.
 *   - The `#define` expands every gl_VertexID occurrence (post-rewrite they are
 *     untouched by rewrite_desktop_builtins) to the compensated expression.
 *     glslang's preprocessor expands it BEFORE the parser, so the parser only
 *     ever sees valid Vulkan GLSL (gl_VertexIndex + _mbv._mithrilBaseVertex).
 *   - _mbv._mithrilBaseVertex is written by Drawing.cpp (backend_push_constants)
 *     to g_state->currentBaseVertex on EVERY draw; it defaults to 0 so
 *     baseVertex==0 draws (the vast majority in Minecraft) are unaffected and
 *     the push constant is never undefined when read.
 *
 * The block + #define are inserted immediately after the #version directive
 * (GLSL requires #version to be the first non-comment line). The block is a
 * top-level declaration and the #define a preprocessor directive — both valid
 * there. Must run AFTER ensure_glsl_version() (so #version exists) and is
 * idempotent (guards on an existing _MithrilBaseVertex marker).
 */
void inject_vertex_id_fixup(std::string& src, GLenum gl_stage) {
    if (gl_stage != GL_VERTEX_SHADER) return;
    if (src.find("_MithrilBaseVertex") != std::string::npos) return;  // idempotent

    // Insert right after the first #version line.
    size_t pos = src.find("#version");
    if (pos == std::string::npos) return;  // ensure_glsl_version guarantees this
    size_t insert_at = src.find('\n', pos);
    if (insert_at == std::string::npos) insert_at = src.length();
    insert_at += 1;  // after the newline

    const std::string snippet =
        "layout(push_constant) uniform _MithrilBaseVertex {\n"
        "    int _mithrilBaseVertex;\n"
        "} _mbv;\n"
        "#define gl_VertexID (gl_VertexIndex + _mbv._mithrilBaseVertex)\n";
    src.insert(insert_at, snippet);
}

/*
 * Inject `layout(location=N)` qualifiers into GLSL `in` declarations based on
 * the application's glBindAttribLocation() mappings. Minecraft 1.21 shaders
 * use bare `in vec3 Position;` declarations and rely on glBindAttribLocation
 * to assign locations at runtime. Even with setAutoMapLocations(true), we pin
 * the locations explicitly so the SPIR-V stage_input locations match the
 * application's vertex descriptor.
 *
 * Only vertex shaders are affected. The rewrite is conservative: it matches
 * declarations of the form `in <type> <name>;` and skips lines that already
 * have a layout() qualifier.
 */
void apply_attrib_bindings(std::string& src, GLenum gl_stage,
                           const std::unordered_map<std::string, GLuint>* bindings) {
    if (gl_stage != GL_VERTEX_SHADER) return;

    // Allow optional precision / interpolation qualifiers between the
    // interface keyword and the type (e.g. `in highp vec4 Position;`), plus an
    // optional trailing array suffix. Group 1 = type, 2 = name, 3 = array.
    static std::regex in_decl_re(
        R"(^\s*(?:layout\s*\([^)]*\)\s*)?(?:in|attribute)\s+(?:(?:highp|mediump|lowp|flat|smooth|noperspective|centroid|sample|patch)\s+)*(\w+)\s+(\w+)\s*(\[[^\]]*\])?\s*;)",
        std::regex::optimize | std::regex::multiline);

    // Locations the application pinned with glBindAttribLocation().
    std::unordered_map<std::string, GLuint> explicit_locs;
    std::set<GLuint> used;
    if (bindings) {
        for (const auto& kv : *bindings) {
            explicit_locs[kv.first] = kv.second;
            used.insert(kv.second);
        }
    }
    // Auto-assign the remaining (or all) attributes in declaration order,
    // skipping any location the application already claimed.
    GLuint next_auto = 0;
    auto take_auto = [&]() -> GLuint {
        while (used.count(next_auto)) ++next_auto;
        const GLuint loc = next_auto++;
        used.insert(loc);
        return loc;
    };

    std::string out;
    out.reserve(src.size() + 256);
    std::string::const_iterator search_start(src.cbegin());
    std::smatch m;
    size_t last_pos = 0;

    while (std::regex_search(search_start, src.cend(), m, in_decl_re)) {
        size_t match_pos = m.position(0) + (search_start - src.cbegin());
        out.append(src, last_pos, match_pos - last_pos);

        const std::string& vartype = m[1].str();
        const std::string& varname = m[2].str();
        const std::string& array_suffix = m[3].matched ? m[3].str() : std::string();

        GLuint loc = 0;
        auto it = explicit_locs.find(varname);
        if (it != explicit_locs.end()) {
            loc = it->second;
        } else {
            loc = take_auto();
        }

        out += "layout(location=";
        out += std::to_string(loc);
        out += ") in ";
        out += vartype;
        out += ' ';
        out += varname;
        if (!array_suffix.empty()) out += array_suffix;
        out += ';';

        last_pos = match_pos + m[0].length();
        search_start = m.suffix().first;
    }
    out.append(src, last_pos, std::string::npos);
    src.swap(out);
}

// ---------------------------------------------------------------------------
// Opaque GLSL type detection: types that MUST remain as standalone uniform
// declarations (samplers, images, atomic counters, subpass inputs) because
// they cannot be placed inside a uniform block.
// ---------------------------------------------------------------------------
static bool is_opaque_glsl_type(const std::string& name) {
    // All sampler types contain "sampler" in the name (sampler2D, isampler2D,
    // usampler2D, samplerCube, sampler2DArray, sampler2DShadow, etc.).
    if (name.find("sampler") != std::string::npos) return true;
    // All image types contain "image" (image2D, iimage2D, uimage2D, etc.).
    if (name.find("image")   != std::string::npos) return true;
    // Atomic counters and subpass inputs.
    if (name == "atomic_uint") return true;
    if (name.find("subpass") != std::string::npos) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Loose-uniform collection: fold every non-block non-opaque uniform into a
// synthetic UBO, mirroring how ANGLE's CollectVariables folds default uniforms
// into ANGLE_DefaultUniformBlock. The GL_KHR_vulkan_glsl path *should* do
// this automatically via EShClientOpenGL + EShMsgVulkanRules wrapping into
// $Global, but some glslang versions still reject the shader with
// "non-opaque uniforms outside a block". We stay deterministic by doing the
// wrapping ourselves as a preprocessor pass before glslang sees the source.
//
// Forms folded (this is the superset ANGLE collects, minus interface blocks
// and SSBOs which stay as-is):
//   uniform mat4 M;                    single declaration
//   uniform mat4 A, B;                 multiple declarators on one line
//   uniform highp float f;             precision qualifier
//   uniform mat4 arr[8];               (multi-dimensional) arrays
//   uniform MyStruct s;                named struct-typed uniform
//   uniform struct { mat4 a; } u;      anonymous inline struct uniform
//
// All collected declarations are removed from their original locations and
// re-emitted inside a single `mithril_GlobalBlock` UBO injected right after
// #version, with `#define <name> mithril_GlobalBlock.<name>` so the body keeps
// using the original identifier. Opaque uniforms (samplers, images,
// atomic_uint, subpass inputs) remain standalone — they cannot sit in a block.
// Shaders with no loose non-opaque uniforms are left untouched (no-op scan).
// Declarations inside // or /* */ comments are ignored.
// ---------------------------------------------------------------------------

// Find the '}' that closes the '{' at open_idx (brace-depth scan).
static size_t find_matching_brace(const std::string& s, size_t open_idx) {
    int depth = 0;
    for (size_t i = open_idx; i < s.size(); ++i) {
        if (s[i] == '{')       ++depth;
        else if (s[i] == '}') { --depth; if (depth == 0) return i; }
    }
    return std::string::npos;
}

// True if offset `off` sits inside a // or /* */ comment.
static bool is_in_comment(const std::string& s, size_t off) {
    // Line comment: any "//" between line start and off?
    size_t line_start = s.rfind('\n', off);
    line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
    if (s.find("//", line_start) < off) return true;
    // Block comment: count unterminated /* before off.
    int depth = 0;
    for (size_t i = 0; i < off; ++i) {
        if (i + 1 < off && s[i] == '/' && s[i + 1] == '*') { ++depth; ++i; }
        else if (i + 1 < off && s[i] == '*' && s[i + 1] == '/') { if (depth) --depth; ++i; }
    }
    return depth > 0;
}

// Split a declarator list ("a, b[2], c") into individual trimmed declarators,
// tracking [] depth so commas inside array sizes are ignored.
static void split_declarators(const std::string& list,
                              std::vector<std::string>& out) {
    std::string cur;
    int depth = 0;
    for (char c : list) {
        if (c == '[')      ++depth;
        else if (c == ']') { if (depth) --depth; }
        if (c == ',' && depth == 0) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
}

// Extract (name, arraySuffix) from a single declarator "arr[2][3]".
static bool parse_declarator(const std::string& d, std::string& name,
                             std::string& arr) {
    static const std::regex nm_re(R"(^\s*(\w+)\s*((?:\[[^\]]*\])*)\s*$)");
    std::smatch m;
    if (!std::regex_match(d, m, nm_re)) return false;
    name = m[1].str();
    arr  = m[2].str();
    return true;
}
static void wrap_loose_uniforms(std::string& source) {
    struct Member { std::string decl; std::string name; };
    struct Erase  { size_t pos; size_t len; };
    std::vector<Member> members;
    std::vector<Erase>  erases;

    // --- Pass A: simple / named-struct / precision / array / multi-declarator.
    //     uniform [layout(...)] [prec] <type> <decllist> ;
    static const std::regex simple_re(
        R"(^[ \t]*((?:layout\s*\([^)]*\)\s*)?uniform\s+(?!struct\b)(?:(?:highp|mediump|lowp)\s+)?(\w+)\s+([^;]+?)\s*;))",
        std::regex::multiline | std::regex::optimize);

    {
        auto cur = source.cbegin(), end = source.cend();
        std::smatch m;
        while (std::regex_search(cur, end, m, simple_re)) {
            size_t off  = m.position(0) + (cur - source.cbegin());
            size_t full = m[1].str().size();
            std::string type      = m[2].str();
            std::string decllist  = m[3].str();
            cur = m.suffix().first;

            if (is_in_comment(source, off)) continue;
            if (is_opaque_glsl_type(type))  continue;

            std::vector<std::string> names;
            split_declarators(decllist, names);
            for (const auto& n : names) {
                std::string var, arr;
                if (parse_declarator(n, var, arr)) {
                    members.push_back({type + " " + var + arr, var});
                    erases.push_back({off, full});
                }
            }
        }
    }

    // --- Pass B: anonymous / inline / named struct uniforms.
    //     uniform struct [Name] { ... } <decllist> ;
    static const std::regex struct_re(
        R"(^[ \t]*(uniform\s+struct\s+(\w+)?\s*\{))",
        std::regex::multiline | std::regex::optimize);

    {
        auto cur = source.cbegin(), end = source.cend();
        std::smatch m;
        while (std::regex_search(cur, end, m, struct_re)) {
            size_t off     = m.position(0) + (cur - source.cbegin());
            size_t brace   = off + m[1].str().size() - 1;
            bool  named    = m[2].matched;
            std::string struct_name = named ? m[2].str() : "";
            size_t close   = find_matching_brace(source, brace);
            cur = m.suffix().first;
            if (close == std::string::npos) continue;
            if (is_in_comment(source, off)) continue;

            std::string struct_def = source.substr(brace, close - brace + 1);
            size_t after = close + 1;
            size_t semi  = source.find(';', after);
            if (semi == std::string::npos) continue;
            std::string decllist = source.substr(after, semi - after);

            std::vector<std::string> names;
            split_declarators(decllist, names);
            for (const auto& n : names) {
                std::string var, arr;
                if (!parse_declarator(n, var, arr)) continue;
                if (named) {
                    // Keep "struct Name { ... };" globally; drop "uniform " and
                    // the trailing declarator so only the type definition survives.
                    erases.push_back({off, 7});                  // "uniform "
                    erases.push_back({after, semi - after + 1}); // " u;"
                    members.push_back({struct_name + " " + var + arr, var});
                } else {
                    erases.push_back({off, semi - off + 1});
                    members.push_back({struct_def + " " + var + arr, var});
                }
            }
        }
    }

    if (members.empty()) return;

    // Insertion point: right after #version (ensure_glsl_version guarantees it).
    size_t version_end = 0;
    {
        auto vp = source.find("#version");
        if (vp != std::string::npos) {
            auto nl = source.find('\n', vp);
            version_end = (nl != std::string::npos) ? nl + 1 : source.size();
        }
    }

    std::string injection;
    injection += "\nuniform mithril_GlobalBlock {\n";
    for (const auto& u : members)
        injection += "    " + u.decl + ";\n";
    injection += "} _m;\n\n";
    for (const auto& u : members)
        injection += "#define " + u.name + " _m." + u.name + "\n";
    injection += "\n";

    // Erase originals (descending position so earlier offsets stay valid).
    std::sort(erases.begin(), erases.end(),
              [](const Erase& a, const Erase& b) { return a.pos > b.pos; });
    for (const auto& e : erases)
        source.erase(e.pos, e.len);

    source.insert(version_end, injection);
}

// ---------------------------------------------------------------------------
// Per-stage explicit binding injection for opaque uniforms and UBO blocks.
//
// 背景（root cause AK — Sampler0Smplr undeclared + 找不到 uniform）：
//   glsl_to_spirv 每个 stage 单独编译，VS/FS 各自建独立 TProgram，从不调
//   program->mapIO()，glslang 的跨 stage 共享 binding 槽位机制没启用。
//   setAutoMapBindings(true) 让每个 stage 的 binding 都从 0 重新分配，
//   结果 VS 和 FS 的 mithril_GlobalBlock 都拿到 binding 0、Sampler0 都拿到
//   binding 1。merge_bindings() 按 (set,binding,type) 去重，把 FS 的条目
//   折叠进 VS 的条目 → FS 的 ColorModulator/Sampler0 从 uniform 表消失，
//   MoltenVK 给 FS 生成的 MSL 引用了 Sampler0Smplr 但 descriptor 没提供 →
//   "use of undeclared identifier 'Sampler0Smplr'"。
//
//   setShiftBinding(EResTexture, base) 本意给 FS 的 sampler 加偏移避开冲突，
//   但实测（glslang probe）对 auto-mapped combined sampler 完全不生效——
//   binding 仍是 0。setShiftBinding 只作用于带显式 layout(binding=) 的变量。
//
//   MobileGL 的正确做法是单 TProgram + mapIO() 让 glslang 全局分配 binding。
//   但 Mithril 的 glsl_to_spirv 是单 stage 接口 + 三级 fallback，改成跨 stage
//   编译会破坏缓存和 fallback，风险大。
//
//   本 pass 采用更直接的等价方案：在 GLSL 源码层面给每个 opaque uniform
//   (sampler*/image*) 和 UBO block 注入显式 layout(binding=N)，N = stage_base
//   + 出现顺序计数器。VS 用 base 0，FS 用 base 64，compute 用 base 128。
//   这样 SPIR-V 里的 OpDecorate Binding 就是正确的，VS/FS 永不冲突，MoltenVK
//   生成的 MSL 引用各自独立的 sampler 符号。setShiftBinding 保留无害。
//
//   注入规则：
//     1. UBO block: `uniform <Name> { ... } <inst>;` → 前置 layout(binding=N)
//     2. opaque uniform: `uniform sampler2D S;` → 前置 layout(binding=N)
//     3. 已有 layout(binding=) 的不重复注入（幂等）
//     4. 跳过注释内的声明
//   binding 号在同一 stage 内全局递增（UBO 和 sampler 共享计数器），保证唯一。
// ---------------------------------------------------------------------------
// 前向声明：binding_base_for_stage 定义在下方 glsl_to_spirv 附近，这里先用。
unsigned binding_base_for_stage(EShLanguage stage);

void inject_opaque_bindings(std::string& source, GLenum gl_stage) {
    EShLanguage stage = to_esh_stage(gl_stage);
    if (stage == EShLangCount) return;
    unsigned binding = binding_base_for_stage(stage);

    // ---- Pass 1: UBO blocks (named + anonymous) ----
    // 匹配 `uniform <Word> {` —— 命中 named block (mithril_GlobalBlock) 和
    // anonymous block (uniform { ... })。用找到 '{' 来确认是 block 而非
    // 普通 uniform 变量。
    static const std::regex block_re(
        R"((^[ \t]*)uniform\s+(?:layout\s*\([^)]*\)\s*)?(\w+)?\s*\{)",
        std::regex::multiline | std::regex::optimize);
    {
        auto cur = source.cbegin();
        std::smatch m;
        std::string out;
        out.reserve(source.size() + 64);
        size_t last = 0;
        while (std::regex_search(cur, source.cend(), m, block_re)) {
            size_t pos = m.position(0) + (cur - source.cbegin());
            // 检查该位置是否在注释内
            if (is_in_comment(source, pos)) {
                out.append(source, last, pos - last);
                out += m[0].str();
                last = pos + m[0].length();
                cur = m.suffix().first;
                continue;
            }
            // 检查是否已有 layout(binding=) —— 幂等
            std::string full_match = m[0].str();
            if (full_match.find("layout(") != std::string::npos &&
                full_match.find("binding=") != std::string::npos) {
                out.append(source, last, pos - last);
                out += m[0].str();
                last = pos + m[0].length();
                cur = m.suffix().first;
                continue;
            }
            std::string indent = m[1].str();
            std::string blockname = m[2].matched ? m[2].str() : std::string();
            out.append(source, last, pos - last);
            out += indent;
            out += "layout(binding=";
            out += std::to_string(binding++);
            out += ") uniform ";
            if (!blockname.empty()) { out += blockname; out += ' '; }
            out += "{";
            last = pos + m[0].length();
            cur = m.suffix().first;
        }
        out.append(source, last, std::string::npos);
        source.swap(out);
    }

    // ---- Pass 2: opaque uniforms (sampler*/image*) ----
    // 匹配 `uniform <sampler|image...> <name>[<array>];`。is_opaque_glsl_type
    // 复用上面的判定。跳过已有 layout(binding=) 的声明。
    static const std::regex opaque_re(
        R"((^[ \t]*)uniform\s+(?:layout\s*\([^)]*\)\s*)?(\w+)\s+(\w+)\s*((?:\[[^\]]*\])*)\s*;)",
        std::regex::multiline | std::regex::optimize);
    {
        auto cur = source.cbegin();
        std::smatch m;
        std::string out;
        out.reserve(source.size() + 64);
        size_t last = 0;
        while (std::regex_search(cur, source.cend(), m, opaque_re)) {
            size_t pos = m.position(0) + (cur - source.cbegin());
            std::string vartype = m[2].str();
            if (!is_opaque_glsl_type(vartype)) {
                cur = m.suffix().first;
                continue;
            }
            if (is_in_comment(source, pos)) {
                cur = m.suffix().first;
                continue;
            }
            std::string full_match = m[0].str();
            if (full_match.find("layout(") != std::string::npos &&
                full_match.find("binding=") != std::string::npos) {
                cur = m.suffix().first;
                continue;
            }
            std::string indent = m[1].str();
            std::string varname = m[3].str();
            std::string arr = m[4].matched ? m[4].str() : std::string();
            out.append(source, last, pos - last);
            out += indent;
            out += "layout(binding=";
            out += std::to_string(binding++);
            out += ") uniform ";
            out += vartype;
            out += ' ';
            out += varname;
            out += arr;
            out += ';';
            last = pos + m[0].length();
            cur = m.suffix().first;
        }
        out.append(source, last, std::string::npos);
        source.swap(out);
    }
}

// ---------------------------------------------------------------------------
// Position fixup injection (GL -> Vulkan NDC adjustment).
//
// Deep reference: MobileGL ProgramFactory::InsertPositionFixup
// (ProgramFactory.cpp:789-857) applies two position transforms at the SPIR-V
// level via SPIRV-Tools IRBuilder. Mithril does not link SPIRV-Tools, so we
// achieve the same result via GLSL source injection before glslang compiles.
//
// Transforms applied (vertex shader only):
//   1. Z remap (ALWAYS): gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5
//      GL NDC Z is [-1,1]; Vulkan NDC Z is [0,1]. Without this, Vulkan's
//      clip stage rejects geometry with z/w < 0 (GL's near plane) and depth
//      testing compares wrong values. Must happen in-shader (before clip),
//      NOT via viewport minDepth/maxDepth (clip runs before viewport transform).
//   2. Y flip (only when flip_y=true): gl_Position.y = -gl_Position.y
//      GL framebuffer origin is bottom-left (Y up); Vulkan/Metal is top-left
//      (Y down). The default framebuffer (FBO 0) renders directly to the
//      on-screen drawable, so its image must be Y-flipped to match the
//      screen's coordinate system. User-created FBOs render into textures
//      that are subsequently sampled by GL shaders using GL texture coords
//      (Y up), so they must NOT be flipped — flipping them would make the
//      sampled content upside-down (root cause of the red/black screen).
//
// Mechanism: rename `void main(` -> `void _mithril_original_main(` and append
// a wrapper main() that calls the original then applies the fixups. This
// mirrors MobileGL's approach of post-processing the position output, just at
// the GLSL level instead of the SPIR-V level.
//
// Safety: if no `void main(` match is found, the function returns without
// modifying the source (the shader compiles without fixups — depth testing
// may be wrong but no crash). Minecraft Java vertex shaders all use the
// standard `void main()` signature.
// ---------------------------------------------------------------------------
void inject_position_fixup(std::string& src, GLenum gl_stage, bool flip_y) {
    if (gl_stage != GL_VERTEX_SHADER) return;
    static const std::regex main_re(R"(\bvoid\s+main\s*\()");
    if (!std::regex_search(src, main_re)) return;
    src = std::regex_replace(src, main_re, "void _mithril_original_main(");
    src += "\nvoid main() {\n    _mithril_original_main();\n";
    if (flip_y) {
        src += "    gl_Position.y = -gl_Position.y;\n";
    }
    src += "    gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;\n}\n";
}

// ---------------------------------------------------------------------------
// Vulkan-incompatible layout qualifier normalization.
//
// glslang's strict EShClientOpenGL + EShMsgVulkanRules path rejects the
// legacy `layout(packed)` / `layout(shared)` UBO/SSBO packing qualifiers
// (Vulkan only allows std140 for UBOs and std430 for SSBOs). Previously
// these were saved by the third-level relaxed fallback; now that the
// relaxed fallback is removed, this preprocessor pass rewrites them at the
// source level so the strict path accepts them directly.
//
// Rewrites:
//   layout(packed)  uniform { ... }   -> layout(std140) uniform { ... }
//   layout(shared) uniform { ... }   -> layout(std140) uniform { ... }
//   layout(packed)  buffer  { ... }   -> layout(std430) buffer  { ... }
//   layout(shared) buffer  { ... }   -> layout(std430) buffer  { ... }
//
// Idempotent: layout(std140), layout(std430), layout(row_major),
// layout(column_major) and other qualifiers are left untouched. Multi-arg
// forms like `layout(std140, packed)` are handled by replacing packed/shared
// in-place with the appropriate std version. Occurrences inside // or /* */
// comments are skipped (uses the existing is_in_comment helper).
//
// Common Minecraft/mod shaders use the simple single-arg `layout(packed)`
// form. The pass is a no-op if the source contains no `packed`/`shared`
// tokens at all (fast-path guard).
// ---------------------------------------------------------------------------
void normalize_vulkan_incompatible_layouts(std::string& source) {
    // Fast path: skip if source has no packed/shared qualifiers at all.
    if (source.find("packed") == std::string::npos &&
        source.find("shared") == std::string::npos) {
        return;
    }

    // Match `layout(<args>) <storage>` where <storage> is uniform (UBO) or
    // buffer (SSBO). Capture the args and storage so we can rewrite
    // packed/shared to std140 (UBO) or std430 (SSBO) based on storage class.
    // [^)]* is safe: GLSL layout() never contains nested parens.
    static const std::regex re(
        R"(layout\s*\(([^)]*)\)\s*(uniform|buffer)\b)",
        std::regex::optimize | std::regex::multiline);

    // Helper: trim whitespace from a single arg, and if it equals "packed" or
    // "shared", return the appropriate std replacement; otherwise return the
    // trimmed arg unchanged. Sets *changed=true if a replacement happened.
    auto process_arg = [](const std::string& arg, const std::string& replacement,
                          bool* changed) -> std::string {
        size_t b = arg.find_first_not_of(" \t");
        size_t e = arg.find_last_not_of(" \t");
        if (b == std::string::npos) return std::string();  // whitespace-only
        std::string trimmed = arg.substr(b, e - b + 1);
        if (trimmed == "packed" || trimmed == "shared") {
            *changed = true;
            return replacement;
        }
        return trimmed;
    };

    std::string out;
    out.reserve(source.size());
    std::string::const_iterator it = source.cbegin();
    std::smatch m;
    while (std::regex_search(it, source.cend(), m, re)) {
        size_t match_pos = m.position(0) + (it - source.cbegin());
        // Skip occurrences inside // or /* */ comments.
        if (is_in_comment(source, match_pos)) {
            out.append(it, m[0].second);
            it = m[0].second;
            continue;
        }

        std::string args = m[1].str();
        std::string storage = m[2].str();
        std::string replacement = (storage == "uniform") ? "std140" : "std430";

        // Split args by comma; replace packed/shared with the appropriate std
        // version. Track whether any replacement happened so we can leave
        // non-matching layout() qualifiers untouched (idempotent).
        std::string new_args;
        bool changed = false;
        std::string cur;
        auto flush = [&]() {
            std::string processed = process_arg(cur, replacement, &changed);
            if (!processed.empty()) {
                if (!new_args.empty()) new_args += ", ";
                new_args += processed;
            }
            cur.clear();
        };
        for (char c : args) {
            if (c == ',') flush();
            else cur += c;
        }
        flush();

        if (!changed) {
            // No packed/shared in this layout() — leave it untouched.
            out.append(it, m[0].second);
        } else {
            // Reconstruct: `layout(<new_args>) <storage>`.
            out.append(it, m[0].first);
            out += "layout(";
            out += new_args;
            out += ") ";
            out += storage;
        }
        it = m[0].second;
    }
    out.append(it, source.cend());
    source.swap(out);
}

// ---------------------------------------------------------------------------
// GL legacy construct normalization.
//
// glslang's strict EShClientOpenGL + EShMsgVulkanRules path rejects GL
// legacy constructs that the previous third-level relaxed fallback used to
// accept. With the relaxed fallback removed, this pass rewrites the
// constructs at the source level so the strict path accepts them directly.
//
// Currently handled (fragment shaders only):
//   gl_FragColor: inject a synthetic `layout(location = 0) out vec4
//   _mithril_FragColor;` declaration right after the #version line, and
//   rewrite all word-boundary gl_FragColor references (outside comments) to
//   _mithril_FragColor. Idempotent: if gl_FragColor is not used, the pass is
//   a no-op and a shader that already declares a location=0 output is
//   untouched.
//
// Out of scope (left as follow-up if a shader pack actually needs them):
//   gl_FragData[0] — Minecraft core shaders don't use it.
// ---------------------------------------------------------------------------
void normalize_gl_legacy_constructs(std::string& source, GLenum gl_stage) {
    // Only fragment shaders use gl_FragColor; vertex shaders are unaffected.
    if (gl_stage != GL_FRAGMENT_SHADER) return;

    // Fast path: if gl_FragColor doesn't appear at all, no work to do.
    if (source.find("gl_FragColor") == std::string::npos) return;

    static const std::regex re(R"(\bgl_FragColor\b)", std::regex::optimize);

    // First pass: rewrite all word-boundary gl_FragColor references outside
    // comments to _mithril_FragColor. Track whether any rewrite happened so
    // we can skip the declaration injection if gl_FragColor only appears in
    // comments (idempotent no-op).
    std::string out;
    out.reserve(source.size());
    std::string::const_iterator it = source.cbegin();
    std::smatch m;
    bool rewritten = false;
    while (std::regex_search(it, source.cend(), m, re)) {
        size_t match_pos = m.position(0) + (it - source.cbegin());
        out.append(it, m[0].first);
        if (is_in_comment(source, match_pos)) {
            out.append(m[0].str());
        } else {
            out.append("_mithril_FragColor");
            rewritten = true;
        }
        it = m[0].second;
    }
    out.append(it, source.cend());
    if (!rewritten) return;
    source.swap(out);

    // Inject synthetic declaration right after the #version line so the
    // rewritten references resolve to a Vulkan-legal named output. Uses the
    // same find-#version-insertion-point pattern as wrap_loose_uniforms.
    // (gl_FragData[0] is out of scope — Minecraft core shaders don't use it;
    // left as a follow-up if a shader pack actually needs it.)
    size_t vp = source.find("#version");
    size_t insert_at = 0;
    if (vp != std::string::npos) {
        size_t nl = source.find('\n', vp);
        insert_at = (nl != std::string::npos) ? nl + 1 : source.size();
    }
    source.insert(insert_at, "layout(location = 0) out vec4 _mithril_FragColor;\n");
}

// FNV-1a 64-bit hash for cache keying.
uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (char c : s) { h ^= (uint8_t)c; h *= 1099511628211ULL; }
    return h;
}

struct Cache {
    std::mutex mu;
    std::unordered_map<uint64_t, std::vector<uint32_t>> entries; // key -> SPIR-V
};
Cache& cache() { static Cache c; return c; }

// ---------------------------------------------------------------------------
// Per-stage descriptor binding spaces.
//
// setAutoMapBindings() numbers bindings per TShader, always restarting at 0.
// Every GL stage is compiled as its own TShader here — there is no single
// TProgram spanning both stages before SPIR-V generation — so glslang has no
// way to coordinate the two, and does not try to. A Minecraft program such as
// rendertype_text comes out of the compiler like this:
//
//   vertex:    mithril_GlobalBlock -> set 0, binding 0   (ModelViewMat, ProjMat)
//              Sampler2            -> set 0, binding 1   (lightmap)
//   fragment:  mithril_GlobalBlock -> set 0, binding 0   (ColorModulator, Fog*)
//              Sampler0            -> set 0, binding 1   (glyph atlas)
//
// Both stages claim (set 0, binding 0) and (set 0, binding 1) for entirely
// different resources. merge_bindings() keys a descriptor on
// (set, binding, type) and so reads each colliding pair as ONE descriptor: the
// fragment entries are dropped and only their stageMask is OR-ed into the
// vertex entry. Everything downstream inherits the damage:
//
//   * Program.cpp builds the uniform table from the merged list, so every
//     fragment-only uniform disappears. That is precisely what the device log
//     shows — ColorModulator / FogStart / FogEnd / FogColor / Sampler0 all
//     report "could not find uniform named ...", while not one vertex uniform
//     does. glUniform1i(Sampler0, 0) is then never issued, so the backend
//     never learns which texture unit feeds the sampler.
//   * The surviving descriptor carries the vertex block's size and member
//     offsets, so the fragment stage reads matrix bytes where a colour belongs.
//   * The sampler descriptor resolves to the vertex stage's texture (Sampler2)
//     while the fragment shader samples it as Sampler0.
//   * MoltenVK is handed a single binding that the two stages disagree about,
//     and the MSL it generates for the fragment stage uses a sampler it never
//     declared:  error: use of undeclared identifier 'Sampler0Smplr'
//     — which fails pipeline creation outright (rc=-3 in the device log).
//
// Giving each stage its own range makes the collision arithmetically
// impossible. glslang exposes exactly this through setShiftBinding(), the same
// mechanism HLSL uses to separate register spaces: the shift becomes the
// starting point for auto-assignment and is also added to any explicit
// layout(binding=) a shader declares, so both kinds of shader stay separated.
//
// The span is 64 while kMaxTextureUnits is 32, so a stage would have to declare
// 32 samplers AND 32 buffers before it could reach the next stage's range.
// Vulkan places no requirement on binding numbers being dense, and descriptor
// pool sizes are computed from descriptor COUNTS rather than from the largest
// binding, so the sparse layout costs nothing.
constexpr unsigned kBindingSpacePerStage = 64;

unsigned binding_base_for_stage(EShLanguage stage) {
    switch (stage) {
        case EShLangVertex:   return 0u * kBindingSpacePerStage;
        case EShLangFragment: return 1u * kBindingSpacePerStage;
        case EShLangCompute:  return 2u * kBindingSpacePerStage;
        default:              return 3u * kBindingSpacePerStage;
    }
}

// Must run on every TShader before parse(), the unwrapped fallback included:
// a shader that only survives the second attempt would otherwise slip
// back to colliding bindings — and would do it silently, since nothing fails
// until MoltenVK rejects the generated MSL several frames later.
//
// NOTE: setShiftBinding 经 glslang probe 验证对 auto-mapped combined sampler
// (uniform sampler2D Sampler0; 无显式 layout) 完全不生效——binding 仍是 0。
// 真正的 binding 分离由 inject_opaque_bindings 在 GLSL 源码层面注入
// layout(binding=N) 实现。本函数保留调用（对带显式 layout(binding=) 的变量
// 仍有效，且与 inject_opaque_bindings 幂等不冲突），作为辅助而非主力。
void apply_stage_binding_shift(glslang::TShader& sh, EShLanguage stage) {
    const unsigned base = binding_base_for_stage(stage);
    if (base == 0) return;  // vertex already starts at 0
    sh.setShiftBinding(glslang::EResUbo,     base);
    sh.setShiftBinding(glslang::EResTexture, base);  // combined image samplers
    sh.setShiftBinding(glslang::EResSampler, base);
    sh.setShiftBinding(glslang::EResImage,   base);
    sh.setShiftBinding(glslang::EResSsbo,    base);
}

bool glsl_to_spirv(GLenum gl_stage, const std::string& src,
                   std::vector<uint32_t>& spirv, std::string& info,
                   const std::unordered_map<std::string, GLuint>* attrib_bindings,
                   bool flip_y) {
    glslang_init();
    EShLanguage stage = to_esh_stage(gl_stage);
    if (stage == EShLangCount) { info = "unsupported shader stage"; return false; }

    // Preprocess: upgrade GLSL version (Vulkan requires 330+), rewrite
    // desktop-GLSL builtins that Vulkan GLSL renames (gl_InstanceID ->
    // gl_InstanceIndex), inject the gl_VertexID push-constant compensation,
    // inject attribute location bindings, and wrap loose non-opaque uniforms
    // into a synthetic UBO so glslang produces Vulkan-conformant SPIR-V.
    std::string source = src;
    int glsl_version = ensure_glsl_version(source);
    rewrite_desktop_builtins(source, gl_stage);
    // Root cause: gl_VertexID baseVertex semantics. After ensure_glsl_version
    // (so #version exists) inject the push-constant block + #define that
    // rewrites gl_VertexID -> gl_VertexIndex + _mbv._mithrilBaseVertex. Placed
    // BEFORE the source_unwrapped backup below so BOTH the wrapped and
    // unwrapped compile-fallback paths inherit the injection.
    inject_vertex_id_fixup(source, gl_stage);
    apply_attrib_bindings(source, gl_stage, attrib_bindings);

    // Normalize Vulkan-incompatible layout qualifiers (layout(packed) ->
    // layout(std140) for UBOs, layout(std430) for SSBOs) and GL legacy
    // constructs (gl_FragColor -> synthetic named output). Both passes are
    // source-level and run BEFORE wrap_loose_uniforms / inject_opaque_bindings
    // so the UBO blocks they generate see already-normalized layout
    // qualifiers. Placed here (after apply_attrib_bindings, before
    // inject_position_fixup) so the source_unwrapped backup below captures
    // the normalized source and both compile attempts (wrapped + unwrapped)
    // inherit the normalization. The passes are idempotent and comment-aware.
    normalize_vulkan_incompatible_layouts(source);
    normalize_gl_legacy_constructs(source, gl_stage);

    // Inject GL->Vulkan position fixups (Z remap always; Y flip when flip_y).
    // Done AFTER ensure_glsl_version/rewrite_builtins/apply_attrib_bindings/
    // normalize_* but BEFORE the source_unwrapped backup below, so both
    // compile fallback paths (wrapped / unwrapped) inherit the injection.
    // The wrapper main() appended here is a function definition —
    // wrap_loose_uniforms only touches `uniform` declarations, so it never
    // interferes with the injection.
    // Deep reference: MobileGL GetShaderTransformFlags + InsertPositionFixup.
    inject_position_fixup(source, gl_stage, flip_y);

    // wrap_loose_uniforms() uses std::regex which can throw std::regex_error
    // on pathological inputs (e.g. catastrophic backtracking on a deeply
    // nested declarator list). Wrap it in a try-catch so a single bad shader
    // never crashes the host process — fall back to the un-wrapped source
    // and let glslang's EShMsgVulkanRules auto-wrap path handle loose
    // uniforms instead. The auto-wrap is less robust (some glslang versions
    // still reject non-block uniforms), but it is strictly better than
    // crashing.
    std::string source_unwrapped = source;  // backup for fallback
    bool wrapped = false;
    try {
        wrap_loose_uniforms(source);
        wrapped = true;
    } catch (const std::exception& e) {
        MITHRIL_LOG_WARN("shader", "wrap_loose_uniforms threw: %s; falling back "
                          "to unwrapped source (glslang auto-wrap will run)",
                          e.what());
        source = source_unwrapped;
    }

    // Per-stage explicit binding injection (root cause AK fix).
    // 必须在 wrap_loose_uniforms 之后（UBO block 已生成）执行，给每个
    // UBO block 和 opaque uniform 注入 layout(binding=N)，N 按 stage base
    // 偏移，避免 VS/FS binding 冲突导致 Sampler0Smplr undeclared。
    // 对 source_unwrapped 也注入，保证 fallback 路径同样正确。
    inject_opaque_bindings(source, gl_stage);
    inject_opaque_bindings(source_unwrapped, gl_stage);

    glslang::TShader shader(stage);
    const char* s = source.c_str();
    shader.setStrings(&s, 1);

    // GL_KHR_vulkan_glsl path: parse as OpenGL GLSL but emit Vulkan SPIR-V.
    // EShClientOpenGL is required because the Vulkan client dialect forbids
    // non-block uniforms outright. However, the loose uniforms have already
    // been wrapped into a synthetic block by wrap_loose_uniforms() above
    // (step 6 of the pipeline comment), so glslang never encounters them
    // unadorned. The EShMsgVulkanRules flag + EShClientOpenGL pair is kept as
    // a belt-and-suspenders safety net: if any loose non-opaque uniform slips
    // through (e.g. a type the regex did not recognise), the auto-wrap code
    // path will still protect it. The OpenGL client also keeps desktop GLSL
    // builtin semantics (e.g. gl_VertexID 1-based, gl_InstanceID 1-based).
    // Two-level strict fallback below: if this parse fails AND wrap_loose_uniforms
    // was applied, retry with the unwrapped (but still normalized) source using
    // the same EShClientOpenGL dialect. There is no third relaxed fallback —
    // its coverage is replaced by normalize_vulkan_incompatible_layouts() and
    // normalize_gl_legacy_constructs() in the preprocessor.
    // Target OpenGL 4.50 feature level (a superset of Minecraft's GLSL 150-330)
    // and emit SPIR-V 1.5 (paired with Vulkan 1.2).
    shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientOpenGL, glsl_version);
    shader.setEnvClient(glslang::EShClientOpenGL, glslang::EShTargetOpenGL_450);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);

    // Auto-assign locations/bindings for any `in`/`out`/uniform declarations
    // that lack explicit layout() qualifiers. This lets desktop GLSL 330
    // shaders written without Vulkan qualifiers compile.
    shader.setAutoMapLocations(true);
    shader.setAutoMapBindings(true);

    // Keep this stage's descriptor bindings out of every other stage's range.
    // Auto-assignment restarts at 0 for each TShader, so without the shift the
    // vertex and fragment stages hand identical (set, binding) pairs to two
    // unrelated resources and merge_bindings() folds them into one.
    apply_stage_binding_shift(shader, stage);

    // Inject the Mithril backend identification macros so host shaders can
    // branch on the backend (mirrors MobileGlues' MG_MOBILEGLUES injection).
    // MG_MITHRIL_VERSION encodes major/minor/patch as MMMNNPPP decimal.
    shader.setPreamble(
        "#define MG_MITHRIL 1\n"
        "#define MG_MITHRIL_VERSION 1000000\n"
    );

    // EShMsgVulkanRules IS set as a belt-and-suspenders safety net. If any
    // loose non-opaque uniform somehow bypasses wrap_loose_uniforms() (e.g.
    // an unrecognised type), the glslang auto-wrap path will still catch it
    // and wrap it into the synthetic `$Global` UBO. Without EShMsgVulkanRules,
    // glslang would not enforce Vulkan rules and the emitted SPIR-V might
    // contain non-block uniforms that MoltenVK rejects at module-creation
    // time. The client stays EShClientOpenGL so the emitted SPIR-V remains
    // Vulkan-conformant for MoltenVK. DescriptorSet.cpp uploads the `$Global`
    // UBO's members by name via SPIRV-Cross reflection.
    const EShMessages messages = static_cast<EShMessages>(
        EShMsgDefault | EShMsgSpvRules | EShMsgVulkanRules);

    if (!shader.parse(GetDefaultResources(), glsl_version, true, messages)) {
        info = shader.getInfoLog();
        info += shader.getInfoDebugLog();
        // Retry without wrap_loose_uniforms if we wrapped: the regex-based
        // wrapper can occasionally mangle edge-case declarations (e.g.
        // multi-line declarators, macros in type positions) in ways that
        // make glslang reject a shader that it would otherwise accept via
        // the auto-wrap path. Falling back to the unwrapped source gives
        // glslang one more chance to compile the shader with its own
        // (more conservative) uniform-wrapping logic.
        if (wrapped) {
            MITHRIL_LOG_WARN("shader", "glslang parse failed after wrap; retrying "
                              "with unwrapped source");
            source = source_unwrapped;
            glslang::TShader shader2(stage);
            const char* s2 = source.c_str();
            shader2.setStrings(&s2, 1);
            shader2.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientOpenGL, glsl_version);
            shader2.setEnvClient(glslang::EShClientOpenGL, glslang::EShTargetOpenGL_450);
            shader2.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);
            shader2.setAutoMapLocations(true);
            shader2.setAutoMapBindings(true);
            apply_stage_binding_shift(shader2, stage);
            shader2.setPreamble(
                "#define MG_MITHRIL 1\n"
                "#define MG_MITHRIL_VERSION 1000000\n"
            );
            if (!shader2.parse(GetDefaultResources(), glsl_version, true, messages)) {
                // Two-level strict fallback exhausted: both wrapped and
                // unwrapped EShClientOpenGL attempts failed. The previous
                // third-level relaxed fallback is removed — its coverage
                // (layout(packed/shared) UBOs, gl_FragColor) is now handled
                // by normalize_vulkan_incompatible_layouts() and
                // normalize_gl_legacy_constructs() in the preprocessor
                // above, so there is no need for a relaxed dialect retry.
                return false;
            }
            glslang::TProgram program2;
            program2.addShader(&shader2);
            if (!program2.link(messages)) {
                info = program2.getInfoLog();
                info += program2.getInfoDebugLog();
                return false;
            }
            glslang::TIntermediate* inter2 = program2.getIntermediate(stage);
            if (!inter2) { info = "no intermediate after link (unwrapped retry)"; return false; }
            glslang::SpvOptions spv_opts2;
            spv_opts2.disableOptimizer = false;
            glslang::GlslangToSpv(*inter2, spirv, &spv_opts2);
            if (spirv.empty()) { info = "SPIR-V generation produced no words (unwrapped retry)"; return false; }
            return true;
        }
        return false;
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        info = program.getInfoLog();
        info += program.getInfoDebugLog();
        return false;
    }

    glslang::TIntermediate* inter = program.getIntermediate(stage);
    if (!inter) { info = "no intermediate after link"; return false; }

    glslang::SpvOptions spv_opts;
    spv_opts.disableOptimizer = false;
    glslang::GlslangToSpv(*inter, spirv, &spv_opts);
    if (spirv.empty()) { info = "SPIR-V generation produced no words"; return false; }
    return true;
}

} // namespace

bool shader_translate(GLenum gl_stage, const std::string& glsl_source,
                      std::vector<uint32_t>& out_spirv, std::string& out_info_log,
                      const std::unordered_map<std::string, GLuint>* attrib_bindings,
                      bool flip_y) {
    const char* stage_name =
        gl_stage == GL_VERTEX_SHADER ? "vertex" :
        gl_stage == GL_FRAGMENT_SHADER ? "fragment" : "other";

    // Cache key includes the bindings so that re-linking with different
    // attribute bindings (e.g. a different VertexFormat) produces fresh SPIR-V.
    // flip_y is also part of the key so the Y-flipped variant (for default
    // framebuffer) and the non-flipped variant (for user FBOs) get distinct
    // cache entries — without this they would collide and the wrong SPIR-V
    // would be returned for one of the two framebuffer types.
    uint64_t key = fnv1a(glsl_source) ^ (uint64_t)gl_stage * 0x9E3779B97F4A7C15ULL;
    if (flip_y) key ^= 0xABCD1234567890ABULL;
    if (attrib_bindings) {
        for (const auto& kv : *attrib_bindings) {
            key ^= fnv1a(kv.first) ^ ((uint64_t)kv.second * 0x100000001B3ULL);
        }
    }
    {
        std::lock_guard<std::mutex> lk(cache().mu);
        auto it = cache().entries.find(key);
        if (it != cache().entries.end()) {
            out_spirv = it->second;
            MITHRIL_LOG_DEBUG("shader", "Cache hit for %s shader (hash %016llx)",
                              stage_name, (unsigned long long)key);
            return true;
        }
    }

    MITHRIL_LOG_INFO("shader", "Translating %s shader (%zu bytes GLSL, flip_y=%d)",
                     stage_name, glsl_source.size(), (int)flip_y);

    std::vector<uint32_t> spirv;
    if (!glsl_to_spirv(gl_stage, glsl_source, spirv, out_info_log, attrib_bindings, flip_y)) {
        MITHRIL_LOG_ERROR("shader", "GLSL->SPIR-V failed for %s shader: %s",
                          stage_name, out_info_log.c_str());
        return false;
    }

    MITHRIL_LOG_INFO("shader", "Translated %s shader: %zu SPIR-V words",
                     stage_name, spirv.size());

    out_spirv = spirv;
    std::lock_guard<std::mutex> lk(cache().mu);
    cache().entries[key] = std::move(spirv);
    return true;
}

// ---- Fallback error shader (red-screen fix) ----
// When a shader fails to compile, the program cannot link and every draw is
// skipped — leaving only the application's glClearColor visible (Minecraft's
// red loading screen). Instead of skipping draws, we substitute a minimal
// fallback shader that renders geometry in solid gray. This proves the render
// pipeline works and avoids the "stuck on clear color" failure mode.
//
// The fallback is compiled via the same glsl_to_spirv path so the SPIR-V is
// always valid. Results are cached (thread-safe) so repeated failures of the
// same stage don't re-invoke glslang.
bool get_fallback_spirv(GLenum gl_stage, bool flip_y,
                        std::vector<uint32_t>& out_spirv) {
    // Cache key: stage + flip_y
    static std::mutex fallback_mu;
    static std::unordered_map<uint64_t, std::vector<uint32_t>> fallback_cache;
    uint64_t key = (uint64_t)gl_stage | (flip_y ? (1ULL << 32) : 0);
    {
        std::lock_guard<std::mutex> lk(fallback_mu);
        auto it = fallback_cache.find(key);
        if (it != fallback_cache.end()) {
            out_spirv = it->second;
            return true;
        }
    }

    std::string source;
    if (gl_stage == GL_VERTEX_SHADER) {
        // CRITICAL: fallback VS must NOT read any vertex attributes.
        // Reading layout(location=0) in vec4 a_position when no vertex buffer
        // is bound at binding 0 causes a GPU page fault
        // (kIOGPUCommandBufferCallbackErrorPageFault) on Metal/MoltenVK →
        // VK_ERROR_DEVICE_LOST → permanent device death.
        // Instead, output a fixed degenerate position so all triangles
        // collapse to a single point (nothing drawn, but no crash).
        // The app's clear color remains visible — this is intentional:
        // it proves the pipeline works and avoids the fatal page fault.
        (void)flip_y;  // no position to flip
        source = "#version 330\n"
                 "void main() {\n"
                 "    gl_Position = vec4(0.0, 0.0, 0.5, 1.0);\n"
                 "}\n";
    } else if (gl_stage == GL_FRAGMENT_SHADER) {
        // Solid gray output — visible against any clear color.
        source = "#version 330\n"
                 "layout(location = 0) out vec4 fragColor;\n"
                 "void main() {\n"
                 "    fragColor = vec4(0.5, 0.5, 0.5, 1.0);\n"
                 "}\n";
    } else {
        // No fallback for compute/geometry/tessellation stages.
        return false;
    }

    std::vector<uint32_t> spirv;
    std::string info;
    if (!glsl_to_spirv(gl_stage, source, spirv, info, nullptr, flip_y)) {
        MITHRIL_LOG_ERROR("shader", "FATAL: fallback shader compilation failed: %s",
                          info.c_str());
        return false;
    }

    MITHRIL_LOG_WARN("shader", "Generated fallback %s shader (flip_y=%d): %zu SPIR-V words",
                     gl_stage == GL_VERTEX_SHADER ? "vertex" : "fragment",
                     (int)flip_y, spirv.size());

    out_spirv = spirv;
    std::lock_guard<std::mutex> lk(fallback_mu);
    fallback_cache[key] = spirv;
    return true;
}

} // namespace mithril
