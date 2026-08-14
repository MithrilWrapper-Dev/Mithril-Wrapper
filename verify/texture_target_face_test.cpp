// verify/texture_target_face_test.cpp
//
// Unit test for the 6-face cubemap target mapping (cubemap panorama root cause #1):
//   textureTargetFromGL() must recognise GL_TEXTURE_CUBE_MAP_POSITIVE_X ..
//   GL_TEXTURE_CUBE_MAP_NEGATIVE_Z as CubeMap, so that glTexImage2D/glTexSubImage2D
//   on each cubemap face:
//     - passes the backend's target-consistency check (bound_texture_for_target
//       compares via textureTargetFromGL), i.e. face uploads are NOT silently dropped,
//     - is classified as CubeMap (arrayLayers=6) rather than 2D (arrayLayers=1),
//       so every face gets its own Vulkan array layer.
//   Before the fix the 6 face targets fell through to TextureTarget::Count, so face
//   uploads were dropped and the panorama background was pure red / black.
//
// Single-TU build (no linking surprises) — includes the real State.cpp:
//   g++ -std=c++17 -I Mithril-Wrapper-cpp/include -I Mithril-Wrapper-cpp \
//       -o /tmp/texture_target_face_test verify/texture_target_face_test.cpp \
//       -I/usr/include/vulkan && /tmp/texture_target_face_test
// Requires -I for system vulkan/vulkan.h and glcorearb.h (State.h pulls them in).
#include <cstdio>
#include "MG_State/State.h"
#include "../MG_State/State.cpp"   // bring in the real textureTargetFromGL definition

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (cond) { printf("ok : " __VA_ARGS__ "\n"); } \
    else      { printf("FAIL: " __VA_ARGS__ "\n"); ++failures; } \
} while (0)

int main() {
    using mithril::TextureTarget;

    // The 6 cubemap face targets must map to CubeMap (0x8515..0x851A).
    const GLenum faces[6] = {
        GL_TEXTURE_CUBE_MAP_POSITIVE_X,  // 0x8515
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X,  // 0x8516
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y,  // 0x8517
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,  // 0x8518
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z,  // 0x8519
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,  // 0x851A
    };
    for (int i = 0; i < 6; ++i) {
        TextureTarget t = mithril::textureTargetFromGL(faces[i]);
        CHECK(t == TextureTarget::CubeMap,
              "textureTargetFromGL(face 0x%x) == CubeMap (got enum int %d)", faces[i], (int)t);
    }

    // The generic bind target also maps to CubeMap.
    CHECK(mithril::textureTargetFromGL(GL_TEXTURE_CUBE_MAP) == TextureTarget::CubeMap,
          "textureTargetFromGL(GL_TEXTURE_CUBE_MAP) == CubeMap");

    // 2D must remain 2D (regression guard; sampler2D path must not be disturbed).
    CHECK(mithril::textureTargetFromGL(GL_TEXTURE_2D) == TextureTarget::_2D,
          "textureTargetFromGL(GL_TEXTURE_2D) == _2D");

    // Unknown/other targets must not crash or misclassify as CubeMap.
    TextureTarget unk = mithril::textureTargetFromGL(0xDEADBEFFu);
    CHECK(unk == TextureTarget::Count, "textureTargetFromGL(0xDEADBEEF) == Count (got int %d)", (int)unk);

    printf("\nTEXTURE TARGET FACE TEST: %s (%d failure(s))\n",
           failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}