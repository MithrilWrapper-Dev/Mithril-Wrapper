#ifndef __khrplatform_h_
#define __khrplatform_h_

/*
 * Minimal Khronos platform header for Mithril-Wrapper.
 * Provides the fixed-width GL scalar types and the calling-convention macros
 * used by the GL headers. Deliberately small and self-contained.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void             GLvoid;
typedef char             GLchar;
typedef unsigned int     GLenum;
typedef unsigned char    GLboolean;
typedef unsigned int     GLbitfield;
typedef signed char      GLbyte;
typedef short            GLshort;
typedef int              GLint;
typedef int              GLsizei;
typedef unsigned char    GLubyte;
typedef unsigned short   GLushort;
typedef unsigned int     GLuint;
typedef float            GLfloat;
typedef float            GLclampf;
typedef double           GLdouble;
typedef double           GLclampd;
typedef int64_t          GLint64;
typedef uint64_t         GLuint64;
typedef ptrdiff_t        GLintptr;
typedef ptrdiff_t        GLsizeiptr;
typedef ptrdiff_t        GLintptrARB;
typedef ptrdiff_t        GLsizeiptrARB;
typedef char             GLcharARB;
/* GLhandleARB is specified by Khronos as `unsigned int`, but Apple's
 * <OpenGL/gltypes.h> defines it as `void *`. As soon as AppKit/Cocoa is
 * included (NSOpenGL pulls gltypes.h in), the two typedefs collide and the
 * translation unit fails with "typedef redefinition with different types".
 * That is exactly what broke the macOS-hosted E2E GLFW bridge build, while the
 * iOS build stayed green (no OpenGL framework there).
 *
 * When the Apple SDK headers are visible, adopt the SDK's definition instead of
 * ours. Nothing in Mithril uses GLhandleARB (it belongs to the legacy
 * ARB_shader_objects handle API, which is not part of the Core Profile surface
 * we implement), so this costs us nothing.
 */
#if !defined(__APPLE__) || !defined(__has_include) || !__has_include(<OpenGL/gltypes.h>)
typedef unsigned int     GLhandleARB;
#endif
typedef struct __GLsync* GLsync;
typedef void (*GLDEBUGPROC)(GLenum, GLenum, GLuint, GLenum, GLsizei, const GLchar*, const void*);
typedef void (*GLDEBUGPROCARB)(GLenum, GLenum, GLuint, GLenum, GLsizei, const GLchar*, const void*);
typedef void (*GLDEBUGPROCAMD)(GLuint, GLenum, GLenum, GLsizei, const GLchar*, void*);

/* GLAPI / GLAPIENTRY macros (Khronos-compatible) */
#if defined(_WIN32) && !defined(APIENTRY) && !defined(__CYGWIN__)
#define APIENTRY __stdcall
#else
#define APIENTRY
#endif
#ifndef APIENTRYP
#define APIENTRYP APIENTRY *
#endif
#ifndef GLAPI
#define GLAPI extern
#endif
#ifndef GLAPIENTRY
#define GLAPIENTRY APIENTRY
#endif

#ifdef __cplusplus
}
#endif

#endif /* __khrplatform_h_ */
