/*
 * DirectMetal layered-texture regression control.
 *
 * This intentionally exercises the CPU->MTLTexture upload boundary that is
 * difficult to cover with a normal 2D render smoke: cubemap face selection
 * and 2D-array layer selection.  With MTL_DEBUG_LAYER=1, a wrong mapping of
 * GL face/layer indices to Metal region.depth aborts the process before the
 * GL error checks below, so a clean exit is a useful negative control for the
 * exact validation failure seen in Minecraft resource loading.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glcorearb.h>

#ifndef GL_TEXTURE_CUBE_MAP
#define GL_TEXTURE_CUBE_MAP 0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X 0x8516
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif

typedef void (*genTextures_fn)(GLsizei, GLuint*);
typedef void (*bindTexture_fn)(GLenum, GLuint);
typedef void (*texImage2D_fn)(GLenum, GLint, GLint, GLsizei, GLsizei,
                              GLint, GLenum, GLenum, const void*);
typedef void (*texImage3D_fn)(GLenum, GLint, GLint, GLsizei, GLsizei, GLsizei,
                              GLint, GLenum, GLenum, const void*);
typedef void (*texSubImage3D_fn)(GLenum, GLint, GLint, GLint, GLint,
                                 GLsizei, GLsizei, GLsizei,
                                 GLenum, GLenum, const void*);
typedef void (*finish_fn)(void);
typedef GLenum (*getError_fn)(void);
typedef const GLubyte* (*getString_fn)(GLenum);
typedef void (*deleteTextures_fn)(GLsizei, const GLuint*);

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (cond) printf("ok  : " fmt "\n", ##__VA_ARGS__); \
    else { printf("FAIL: " fmt "\n", ##__VA_ARGS__); ++failures; } \
} while (0)

#define LOAD(type, name) \
    type name = (type)dlsym(handle, #name); \
    CHECK(name != NULL, "resolve %s", #name)

static void fill_rgba(unsigned char* dst, int pixels, unsigned char r,
                      unsigned char g, unsigned char b) {
    for (int i = 0; i < pixels; ++i) {
        dst[i * 4 + 0] = (unsigned char)(r + i);
        dst[i * 4 + 1] = (unsigned char)(g + i * 3);
        dst[i * 4 + 2] = (unsigned char)(b + i * 5);
        dst[i * 4 + 3] = 255;
    }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "./build/libmithril.dylib";
    void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    CHECK(handle != NULL, "dlopen %s", path);
    if (!handle) {
        fprintf(stderr, "dlerror: %s\n", dlerror());
        return 1;
    }

    LOAD(genTextures_fn, glGenTextures);
    LOAD(bindTexture_fn, glBindTexture);
    LOAD(texImage2D_fn, glTexImage2D);
    LOAD(texImage3D_fn, glTexImage3D);
    LOAD(texSubImage3D_fn, glTexSubImage3D);
    LOAD(finish_fn, glFinish);
    LOAD(getError_fn, glGetError);
    LOAD(getString_fn, glGetString);
    LOAD(deleteTextures_fn, glDeleteTextures);
    if (failures) return failures;

    const char* version = (const char*)glGetString(GL_VERSION);
    CHECK(version && strstr(version, "Metal 3 (DirectMetal)"),
          "forced backend is DirectMetal (%s)", version ? version : "null");

    unsigned char face0[4 * 4 * 4];
    unsigned char face1[4 * 4 * 4];
    fill_rgba(face0, 16, 11, 37, 73);
    fill_rgba(face1, 16, 101, 17, 53);

    GLuint cube = 0;
    glGenTextures(1, &cube);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA8,
                 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, face0);
    CHECK(glGetError() == GL_NO_ERROR,
          "cubemap face 0 upload has no GL error");
    /* Face index 1 is the critical regression: treating it as MTL depth z=1
     * on a 2D texture produces (origin.z + depth) == 2 > texture.depth == 1. */
    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA8,
                 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, face1);
    CHECK(glGetError() == GL_NO_ERROR,
          "cubemap face 1 maps to Metal slice, not texture depth");

    unsigned char layers[4 * 4 * 2 * 4];
    fill_rgba(layers, 32, 7, 29, 83);
    GLuint arrayTex = 0;
    glGenTextures(1, &arrayTex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, arrayTex);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
                 4, 4, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, layers);
    CHECK(glGetError() == GL_NO_ERROR,
          "2D-array two-layer allocation/upload has no GL error");

    unsigned char replacement[4 * 4 * 2 * 4];
    fill_rgba(replacement, 32, 151, 31, 5);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
                    4, 4, 2, GL_RGBA, GL_UNSIGNED_BYTE, replacement);
    CHECK(glGetError() == GL_NO_ERROR,
          "2D-array two-layer subimage maps layers to Metal slices");

    glFinish();
    CHECK(glGetError() == GL_NO_ERROR,
          "layered uploads finish without deferred GL error");

    glDeleteTextures(1, &arrayTex);
    glDeleteTextures(1, &cube);
    dlclose(handle);

    printf("DIRECTMETAL LAYERED TEXTURE SMOKE: %s\n",
           failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
