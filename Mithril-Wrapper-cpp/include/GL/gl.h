#ifndef __gl_h_
#define __gl_h_

/*
 * Mithril-Wrapper: OpenGL header.
 * Pulls in the Core Profile surface (glcorearb.h) and additionally declares the
 * legacy fixed-function entry points that are provided as no-op stubs so that
 * applications still dlsym-ing GL 1.x/2.x symbols resolve cleanly.
 */

#include <KHR/khrplatform.h>
#include "glcorearb.h"

/*
 * Texture level parameter constants used by glGetTexLevelParameteriv and
 * glTexImage2D(GL_PROXY_TEXTURE_2D). These are standard GL values but
 * missing from our minimal glcorearb.h.
 */
#ifndef GL_PROXY_TEXTURE_2D
#define GL_PROXY_TEXTURE_2D          0x8064
#endif
#ifndef GL_TEXTURE_WIDTH
#define GL_TEXTURE_WIDTH             0x1000
#endif
#ifndef GL_TEXTURE_HEIGHT
#define GL_TEXTURE_HEIGHT            0x1001
#endif
#ifndef GL_TEXTURE_DEPTH
#define GL_TEXTURE_DEPTH             0x8071
#endif

/*
 * Pixel-pack (glReadPixels) state constants. GL_PIXEL_PACK_BUFFER_BINDING is
 * standard GL 2.1/3.0 (0x88ED) but absent from our minimal glcorearb.h, which
 * made any consumer that saves/restores the bound PBO before a readback fail
 * to compile with "use of undeclared identifier".
 */
#ifndef GL_PIXEL_PACK_BUFFER_BINDING
#define GL_PIXEL_PACK_BUFFER_BINDING 0x88ED
#endif
#ifndef GL_TEXTURE_INTERNAL_FORMAT
#define GL_TEXTURE_INTERNAL_FORMAT   0x1003
#endif

/*
 * Pixel-store constants missing from our minimal glcorearb.h. The matching
 * pack/unpack state lives in mithril::PixelStoreState (g_state->pixelStore).
 */
#ifndef GL_PACK_IMAGE_HEIGHT
#define GL_PACK_IMAGE_HEIGHT                     0x806C
#endif
#ifndef GL_PACK_SKIP_IMAGES
#define GL_PACK_SKIP_IMAGES          0x806B
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Legacy fixed-function (stubbed, see src/stubs.cpp) ---- */

/* Miscellaneous legacy */
GLAPI void GLAPIENTRY glClearIndex(GLfloat c);
GLAPI void GLAPIENTRY glIndexMask(GLuint mask);
GLAPI void GLAPIENTRY glAlphaFunc(GLenum func, GLclampf ref);
GLAPI void GLAPIENTRY glLogicOp(GLenum opcode);
GLAPI void GLAPIENTRY glLineStipple(GLint factor, GLushort pattern);
GLAPI void GLAPIENTRY glPolygonStipple(const GLubyte* mask);
GLAPI void GLAPIENTRY glGetPolygonStipple(GLubyte* mask);
GLAPI void GLAPIENTRY glEdgeFlag(GLboolean flag);
GLAPI void GLAPIENTRY glEdgeFlagv(const GLboolean* flag);
GLAPI void GLAPIENTRY glClipPlane(GLenum plane, const GLdouble* equation);
GLAPI void GLAPIENTRY glGetClipPlane(GLenum plane, GLdouble* equation);

/* Accumulation buffer */
GLAPI void GLAPIENTRY glClearAccum(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
GLAPI void GLAPIENTRY glAccum(GLenum op, GLfloat value);

/* Transformation (matrix stack) */
GLAPI void GLAPIENTRY glMatrixMode(GLenum mode);
GLAPI void GLAPIENTRY glOrtho(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble near_val, GLdouble far_val);
GLAPI void GLAPIENTRY glFrustum(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble near_val, GLdouble far_val);
GLAPI void GLAPIENTRY glPushMatrix(void);
GLAPI void GLAPIENTRY glPopMatrix(void);
GLAPI void GLAPIENTRY glLoadIdentity(void);
GLAPI void GLAPIENTRY glLoadMatrixd(const GLdouble* m);
GLAPI void GLAPIENTRY glLoadMatrixf(const GLfloat* m);
GLAPI void GLAPIENTRY glMultMatrixd(const GLdouble* m);
GLAPI void GLAPIENTRY glMultMatrixf(const GLfloat* m);
GLAPI void GLAPIENTRY glRotated(GLdouble angle, GLdouble x, GLdouble y, GLdouble z);
GLAPI void GLAPIENTRY glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
GLAPI void GLAPIENTRY glScaled(GLdouble x, GLdouble y, GLdouble z);
GLAPI void GLAPIENTRY glScalef(GLfloat x, GLfloat y, GLfloat z);
GLAPI void GLAPIENTRY glTranslated(GLdouble x, GLdouble y, GLdouble z);
GLAPI void GLAPIENTRY glTranslatef(GLfloat x, GLfloat y, GLfloat z);

/* Display lists */
GLAPI GLboolean GLAPIENTRY glIsList(GLuint list);
GLAPI void GLAPIENTRY glDeleteLists(GLuint list, GLsizei range);
GLAPI GLuint GLAPIENTRY glGenLists(GLsizei range);
GLAPI void GLAPIENTRY glNewList(GLuint list, GLenum mode);
GLAPI void GLAPIENTRY glEndList(void);
GLAPI void GLAPIENTRY glCallList(GLuint list);
GLAPI void GLAPIENTRY glCallLists(GLsizei n, GLenum type, const GLvoid* lists);
GLAPI void GLAPIENTRY glListBase(GLuint base);

/* Immediate mode */
GLAPI void GLAPIENTRY glBegin(GLenum mode);
GLAPI void GLAPIENTRY glEnd(void);
GLAPI void GLAPIENTRY glVertex2d(GLdouble x, GLdouble y);
GLAPI void GLAPIENTRY glVertex2f(GLfloat x, GLfloat y);
GLAPI void GLAPIENTRY glVertex2i(GLint x, GLint y);
GLAPI void GLAPIENTRY glVertex3f(GLfloat x, GLfloat y, GLfloat z);
GLAPI void GLAPIENTRY glVertex3fv(const GLfloat* v);
GLAPI void GLAPIENTRY glVertex4f(GLfloat x, GLfloat y, GLfloat z, GLfloat w);
GLAPI void GLAPIENTRY glNormal3f(GLfloat nx, GLfloat ny, GLfloat nz);
GLAPI void GLAPIENTRY glColor3f(GLfloat red, GLfloat green, GLfloat blue);
GLAPI void GLAPIENTRY glColor3ub(GLubyte r, GLubyte g, GLubyte b);
GLAPI void GLAPIENTRY glColor4f(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
GLAPI void GLAPIENTRY glColor4ub(GLubyte r, GLubyte g, GLubyte b, GLubyte a);
GLAPI void GLAPIENTRY glTexCoord2f(GLfloat s, GLfloat t);
GLAPI void GLAPIENTRY glTexCoord4f(GLfloat s, GLfloat t, GLfloat r, GLfloat q);
GLAPI void GLAPIENTRY glRasterPos2i(GLint x, GLint y);
GLAPI void GLAPIENTRY glRasterPos3f(GLfloat x, GLfloat y, GLfloat z);
GLAPI void GLAPIENTRY glWindowPos2i(GLint x, GLint y);

/* Lighting / material / fog */
GLAPI void GLAPIENTRY glShadeModel(GLenum mode);
GLAPI void GLAPIENTRY glMaterialf(GLenum face, GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glMaterialfv(GLenum face, GLenum pname, const GLfloat* params);
GLAPI void GLAPIENTRY glMateriali(GLenum face, GLenum pname, GLint param);
GLAPI void GLAPIENTRY glLightf(GLenum light, GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glLightfv(GLenum light, GLenum pname, const GLfloat* params);
GLAPI void GLAPIENTRY glLightModelf(GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glLightModelfv(GLenum pname, const GLfloat* params);
GLAPI void GLAPIENTRY glFogf(GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glFogi(GLenum pname, GLint param);
GLAPI void GLAPIENTRY glFogfv(GLenum pname, const GLfloat* params);
GLAPI void GLAPIENTRY glFogiv(GLenum pname, const GLint* params);
GLAPI void GLAPIENTRY glColorMaterial(GLenum face, GLenum mode);

/* Tex gen / env */
GLAPI void GLAPIENTRY glTexGend(GLenum coord, GLenum pname, GLdouble param);
GLAPI void GLAPIENTRY glTexGenf(GLenum coord, GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glTexGeni(GLenum coord, GLenum pname, GLint param);
GLAPI void GLAPIENTRY glTexGenfv(GLenum coord, GLenum pname, const GLfloat* params);
GLAPI void GLAPIENTRY glTexGeniv(GLenum coord, GLenum pname, const GLint* params);
GLAPI void GLAPIENTRY glTexEnvf(GLenum target, GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glTexEnvi(GLenum target, GLenum pname, GLint param);
GLAPI void GLAPIENTRY glTexEnvfv(GLenum target, GLenum pname, const GLfloat* params);
GLAPI void GLAPIENTRY glTexEnviv(GLenum target, GLenum pname, const GLint* params);

/* Pixel transfer / copy */
GLAPI void GLAPIENTRY glPixelTransferf(GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glPixelTransferi(GLenum pname, GLint param);
GLAPI void GLAPIENTRY glPixelMapfv(GLenum map, GLsizei mapsize, const GLfloat* values);
GLAPI void GLAPIENTRY glPixelMapuiv(GLenum map, GLsizei mapsize, const GLuint* values);
GLAPI void GLAPIENTRY glPixelMapusv(GLenum map, GLsizei mapsize, const GLushort* values);
GLAPI void GLAPIENTRY glPixelZoom(GLfloat xfactor, GLfloat yfactor);
GLAPI void GLAPIENTRY glCopyPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum type);

/* Selection / feedback */
GLAPI void GLAPIENTRY glInitNames(void);
GLAPI void GLAPIENTRY glLoadName(GLuint name);
GLAPI void GLAPIENTRY glPushName(GLuint name);
GLAPI void GLAPIENTRY glPopName(void);
GLAPI GLint GLAPIENTRY glRenderMode(GLenum mode);
GLAPI void GLAPIENTRY glSelectBuffer(GLsizei size, GLuint* buffer);
GLAPI void GLAPIENTRY glFeedbackBuffer(GLsizei size, GLenum type, GLfloat* buffer);
GLAPI void GLAPIENTRY glPassThrough(GLfloat token);

/* Evaluator */
GLAPI void GLAPIENTRY glMap1d(GLenum target, GLdouble u1, GLdouble u2, GLint stride, GLint order, const GLdouble* points);
GLAPI void GLAPIENTRY glMap2d(GLenum target, GLdouble u1, GLdouble u2, GLint ustride, GLint uorder, GLdouble v1, GLdouble v2, GLint vstride, GLint vorder, const GLdouble* points);
GLAPI void GLAPIENTRY glEvalCoord1d(GLdouble u);
GLAPI void GLAPIENTRY glEvalCoord2d(GLdouble u, GLdouble v);
GLAPI void GLAPIENTRY glMapGrid1d(GLint un, GLdouble u1, GLdouble u2);
GLAPI void GLAPIENTRY glMapGrid2d(GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2);
GLAPI void GLAPIENTRY glEvalMesh1(GLenum mode, GLint i1, GLint i2);
GLAPI void GLAPIENTRY glEvalMesh2(GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2);
GLAPI void GLAPIENTRY glEvalPoint1(GLint i);
GLAPI void GLAPIENTRY glEvalPoint2(GLint i, GLint j);

/* Rect */
GLAPI void GLAPIENTRY glRectd(GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2);
GLAPI void GLAPIENTRY glRectf(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2);
GLAPI void GLAPIENTRY glRecti(GLint x1, GLint y1, GLint x2, GLint y2);

/* Attrib stack */
GLAPI void GLAPIENTRY glPushAttrib(GLbitfield mask);
GLAPI void GLAPIENTRY glPopAttrib(void);
GLAPI void GLAPIENTRY glPushClientAttrib(GLbitfield mask);
GLAPI void GLAPIENTRY glPopClientAttrib(void);

/* Client vertex arrays (legacy) */
GLAPI void GLAPIENTRY glEnableClientState(GLenum cap);
GLAPI void GLAPIENTRY glDisableClientState(GLenum cap);
GLAPI void GLAPIENTRY glVertexPointer(GLint size, GLenum type, GLsizei stride, const void* pointer);
GLAPI void GLAPIENTRY glNormalPointer(GLenum type, GLsizei stride, const void* pointer);
GLAPI void GLAPIENTRY glColorPointer(GLint size, GLenum type, GLsizei stride, const void* pointer);
GLAPI void GLAPIENTRY glTexCoordPointer(GLint size, GLenum type, GLsizei stride, const void* pointer);
GLAPI void GLAPIENTRY glIndexPointer(GLenum type, GLsizei stride, const void* pointer);
GLAPI void GLAPIENTRY glEdgeFlagPointer(GLsizei stride, const void* pointer);
GLAPI void GLAPIENTRY glInterleavedArrays(GLenum format, GLsizei stride, const void* pointer);
GLAPI void GLAPIENTRY glArrayElement(GLint i);

/* Misc legacy getters/queries */
GLAPI void GLAPIENTRY glGetDoublev(GLenum pname, GLdouble* params);  /* also legacy-compat */
GLAPI void GLAPIENTRY glGetPointerv(GLenum pname, void** params);
GLAPI void GLAPIENTRY glGetTexLevelParameteriv(GLenum target, GLint level, GLenum pname, GLint* params);
GLAPI void GLAPIENTRY glGetTexParameteriv(GLenum target, GLenum pname, GLint* params);
GLAPI void GLAPIENTRY glGetTexParameterfv(GLenum target, GLenum pname, GLfloat* params);
GLAPI void GLAPIENTRY glGetTexParameterIiv(GLenum target, GLenum pname, GLint* params);
GLAPI void GLAPIENTRY glGetTexParameterIuiv(GLenum target, GLenum pname, GLuint* params);
GLAPI void GLAPIENTRY glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void* pixels);
GLAPI void GLAPIENTRY glTexParameterIiv(GLenum target, GLenum pname, const GLint* params);
GLAPI void GLAPIENTRY glTexParameterIuiv(GLenum target, GLenum pname, const GLuint* params);
GLAPI GLboolean GLAPIENTRY glIsTexture(GLuint texture);
GLAPI GLboolean GLAPIENTRY glAreTexturesResident(GLsizei n, const GLuint* textures, GLboolean* residences);
GLAPI void GLAPIENTRY glPrioritizeTextures(GLsizei n, const GLuint* textures, const GLclampf* priorities);
GLAPI void GLAPIENTRY glDepthBoundsEXT(GLclampd zmin, GLclampd zmax);

#ifdef __cplusplus
}
#endif

/* ----------------------------------------------------------------------------
 * Mithril-Wrapper: GL 3.1 / 3.3 Core Profile enums missing from our minimal
 * glcorearb.h. These are referenced by MG_Impl/Getter.cpp (glGet*v limits) and
 * must be declared so the dylib compiles on platforms where the system OpenGL
 * headers are not used (e.g. iOS / MoltenVK cross-builds). Values are taken
 * verbatim from the Khronos OpenGL API glcorearb.h registry.
 * ------------------------------------------------------------------------- */
#ifndef GL_FRAGMENT_INTERPOLATION_OFFSET_BITS
#define GL_FRAGMENT_INTERPOLATION_OFFSET_BITS 0x8E5D
#endif
#ifndef GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS
#define GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS 0x92DC
#endif
#ifndef GL_MAX_COLOR_TEXTURE_SAMPLES
#define GL_MAX_COLOR_TEXTURE_SAMPLES      0x910E
#endif
#ifndef GL_MAX_COMBINED_ATOMIC_COUNTERS
#define GL_MAX_COMBINED_ATOMIC_COUNTERS   0x92D7
#endif
#ifndef GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES
#define GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES 0x82FA
#endif
#ifndef GL_MAX_COMBINED_IMAGE_UNIFORMS
#define GL_MAX_COMBINED_IMAGE_UNIFORMS    0x90CF
#endif
#ifndef GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS
#define GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS 0x8F39
#endif
#ifndef GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS
#define GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS 0x90DC
#endif
#ifndef GL_MAX_COMBINED_UNIFORM_BLOCKS
#define GL_MAX_COMBINED_UNIFORM_BLOCKS    0x8A2E
#endif
#ifndef GL_MAX_COMPUTE_IMAGE_UNIFORMS
#define GL_MAX_COMPUTE_IMAGE_UNIFORMS     0x91BD
#endif
#ifndef GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS
#define GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS 0x90DB
#endif
#ifndef GL_MAX_COMPUTE_WORK_GROUP_COUNT
#define GL_MAX_COMPUTE_WORK_GROUP_COUNT   0x91BE
#endif
#ifndef GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB
#endif
#ifndef GL_MAX_COMPUTE_WORK_GROUP_SIZE
#define GL_MAX_COMPUTE_WORK_GROUP_SIZE    0x91BF
#endif
#ifndef GL_MAX_CULL_DISTANCES
#define GL_MAX_CULL_DISTANCES             0x82F9
#endif
#ifndef GL_MAX_DEPTH_TEXTURE_SAMPLES
#define GL_MAX_DEPTH_TEXTURE_SAMPLES      0x910F
#endif
#ifndef GL_MAX_FRAGMENT_ATOMIC_COUNTERS
#define GL_MAX_FRAGMENT_ATOMIC_COUNTERS   0x92D6
#endif
#ifndef GL_MAX_FRAGMENT_IMAGE_UNIFORMS
#define GL_MAX_FRAGMENT_IMAGE_UNIFORMS    0x90CE
#endif
#ifndef GL_MAX_FRAGMENT_INPUT_COMPONENTS
#define GL_MAX_FRAGMENT_INPUT_COMPONENTS  0x9125
#endif
#ifndef GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS
#define GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS 0x90DA
#endif
#ifndef GL_MAX_GEOMETRY_UNIFORM_BLOCKS
#define GL_MAX_GEOMETRY_UNIFORM_BLOCKS    0x8A2C
#endif
#ifndef GL_MAX_IMAGE_SAMPLES
#define GL_MAX_IMAGE_SAMPLES              0x906D
#endif
#ifndef GL_MAX_IMAGE_UNITS
#define GL_MAX_IMAGE_UNITS                0x8F38
#endif
#ifndef GL_MAX_INTEGER_SAMPLES
#define GL_MAX_INTEGER_SAMPLES            0x9110
#endif
#ifndef GL_MAX_SAMPLES
#define GL_MAX_SAMPLES                    0x8D57
#endif
#ifndef GL_MAX_SERVER_WAIT_TIMEOUT
#define GL_MAX_SERVER_WAIT_TIMEOUT        0x9111
#endif
#ifndef GL_MAX_SHADER_STORAGE_BLOCK_SIZE
#define GL_MAX_SHADER_STORAGE_BLOCK_SIZE  0x90DE
#endif
#ifndef GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS
#define GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS 0x90DD
#endif
#ifndef GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS
#define GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS 0x8C8A
#endif
#ifndef GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS 0x8C8B
#endif
#ifndef GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS 0x8C80
#endif
#ifndef GL_MAX_UNIFORM_BUFFER_BINDINGS
#define GL_MAX_UNIFORM_BUFFER_BINDINGS    0x8A2F
#endif
#ifndef GL_MAX_UNIFORM_LOCATIONS
#define GL_MAX_UNIFORM_LOCATIONS          0x826E
#endif
#ifndef GL_MAX_VERTEX_ATOMIC_COUNTERS
#define GL_MAX_VERTEX_ATOMIC_COUNTERS     0x92D2
#endif
#ifndef GL_MAX_VERTEX_ATTRIB_BINDINGS
#define GL_MAX_VERTEX_ATTRIB_BINDINGS     0x82DA
#endif
#ifndef GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET
#define GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET 0x82D9
#endif
#ifndef GL_MAX_VERTEX_ATTRIB_STRIDE
#define GL_MAX_VERTEX_ATTRIB_STRIDE       0x82E5
#endif
#ifndef GL_MAX_VERTEX_IMAGE_UNIFORMS
#define GL_MAX_VERTEX_IMAGE_UNIFORMS      0x90CA
#endif
#ifndef GL_MAX_VERTEX_OUTPUT_COMPONENTS
#define GL_MAX_VERTEX_OUTPUT_COMPONENTS   0x9122
#endif
#ifndef GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS
#define GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS 0x90D6
#endif
#ifndef GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT
#define GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT 0x8A34
#endif

/* ---------------------------------------------------------------------------
 * Mithril-Wrapper: remaining GL enums (1.0 -> 4.x) referenced across the
 * MG_Impl / MG_State / MG_Backend / EGL translation units but absent from
 * our minimal glcorearb.h. Added so the dylib is self-contained and the
 * iOS / MoltenVK cross-build no longer depends on the host's legacy
 * <OpenGL/gl.h> (which lacks the 3.1+ constants). Values verbatim from the
 * Khronos OpenGL API glcorearb.h registry; each guarded by #ifndef.
 * ------------------------------------------------------------------------- */
#ifndef GL_ANY_SAMPLES_PASSED
#define GL_ANY_SAMPLES_PASSED 0x8C2F
#endif
#ifndef GL_BACK_LEFT
#define GL_BACK_LEFT 0x0402
#endif
#ifndef GL_BACK_RIGHT
#define GL_BACK_RIGHT 0x0403
#endif
#ifndef GL_BUFFER_MAPPED
#define GL_BUFFER_MAPPED 0x88BC
#endif
#ifndef GL_BUFFER_MAP_LENGTH
#define GL_BUFFER_MAP_LENGTH 0x9120
#endif
#ifndef GL_BUFFER_MAP_OFFSET
#define GL_BUFFER_MAP_OFFSET 0x9121
#endif
#ifndef GL_CLIP_DISTANCE0
#define GL_CLIP_DISTANCE0 0x3000
#endif
#ifndef GL_CLIP_DISTANCE1
#define GL_CLIP_DISTANCE1 0x3001
#endif
#ifndef GL_CLIP_DISTANCE2
#define GL_CLIP_DISTANCE2 0x3002
#endif
#ifndef GL_CLIP_DISTANCE3
#define GL_CLIP_DISTANCE3 0x3003
#endif
#ifndef GL_CLIP_DISTANCE4
#define GL_CLIP_DISTANCE4 0x3004
#endif
#ifndef GL_CLIP_DISTANCE5
#define GL_CLIP_DISTANCE5 0x3005
#endif
#ifndef GL_CLIP_DISTANCE6
#define GL_CLIP_DISTANCE6 0x3006
#endif
#ifndef GL_CLIP_DISTANCE7
#define GL_CLIP_DISTANCE7 0x3007
#endif
#ifndef GL_COLOR
#define GL_COLOR 0x1800
#endif
#ifndef GL_COMMAND_BARRIER_BIT
#define GL_COMMAND_BARRIER_BIT 0x00000040
#endif
#ifndef GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT
#define GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT 0x00000001
#endif
#ifndef GL_CONTEXT_LOST
#define GL_CONTEXT_LOST 0x0507
#endif
#ifndef GL_CURRENT_QUERY
#define GL_CURRENT_QUERY 0x8865
#endif
#ifndef GL_CURRENT_VERTEX_ATTRIB
#define GL_CURRENT_VERTEX_ATTRIB 0x8626
#endif
#ifndef GL_DEPTH
#define GL_DEPTH 0x1801
#endif
#ifndef GL_DEPTH_CLAMP
#define GL_DEPTH_CLAMP 0x864F
#endif
#ifndef GL_DISPATCH_INDIRECT_BUFFER
#define GL_DISPATCH_INDIRECT_BUFFER 0x90EE
#endif
#ifndef GL_FLOAT_MAT2
#define GL_FLOAT_MAT2 0x8B5A
#endif
#ifndef GL_FLOAT_MAT3
#define GL_FLOAT_MAT3 0x8B5B
#endif
#ifndef GL_FLOAT_MAT4
#define GL_FLOAT_MAT4 0x8B5C
#endif
// Minimal GL 2.0/3.0 scalar + vector types used by uniform reflection
// (glGetActiveUniformBlockiv, Program.cpp). Standard GL enum values.
#ifndef GL_INT
#define GL_INT 0x1404
#endif
#ifndef GL_UNSIGNED_INT
#define GL_UNSIGNED_INT 0x1405
#endif
#ifndef GL_BOOL
#define GL_BOOL 0x8B56
#endif
#ifndef GL_INT_VEC2
#define GL_INT_VEC2 0x8B53
#endif
#ifndef GL_INT_VEC3
#define GL_INT_VEC3 0x8B54
#endif
#ifndef GL_INT_VEC4
#define GL_INT_VEC4 0x8B55
#endif
#ifndef GL_BOOL_VEC2
#define GL_BOOL_VEC2 0x8B57
#endif
#ifndef GL_BOOL_VEC3
#define GL_BOOL_VEC3 0x8B58
#endif
#ifndef GL_BOOL_VEC4
#define GL_BOOL_VEC4 0x8B59
#endif
#ifndef GL_UNSIGNED_INT_VEC2
#define GL_UNSIGNED_INT_VEC2 0x8DC6
#endif
#ifndef GL_UNSIGNED_INT_VEC3
#define GL_UNSIGNED_INT_VEC3 0x8DC7
#endif
#ifndef GL_UNSIGNED_INT_VEC4
#define GL_UNSIGNED_INT_VEC4 0x8DC8
#endif
#ifndef GL_FLOAT_VEC2
#define GL_FLOAT_VEC2 0x8B50
#endif
#ifndef GL_FLOAT_VEC3
#define GL_FLOAT_VEC3 0x8B51
#endif
#ifndef GL_FLOAT_VEC4
#define GL_FLOAT_VEC4 0x8B52
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE
#define GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE 0x8215
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE
#define GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE 0x8214
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING
#define GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING 0x8210
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE
#define GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE 0x8216
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE
#define GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE 0x8213
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_LAYERED
#define GL_FRAMEBUFFER_ATTACHMENT_LAYERED 0x8DA7
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME
#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME 0x8CD1
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE
#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE 0x8CD0
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE
#define GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE 0x8212
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE
#define GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE 0x8217
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE 0x8CD3
#endif
#ifndef GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL 0x8CD2
#endif
#ifndef GL_FRONT_LEFT
#define GL_FRONT_LEFT 0x0400
#endif
#ifndef GL_FRONT_RIGHT
#define GL_FRONT_RIGHT 0x0401
#endif
#ifndef GL_INTERLEAVED_ATTRIBS
#define GL_INTERLEAVED_ATTRIBS 0x8C8C
#endif
#ifndef GL_LEFT
#define GL_LEFT 0x0406
#endif
#ifndef GL_OBJECT_TYPE
#define GL_OBJECT_TYPE 0x9112
#endif
#ifndef GL_PARAMETER_BUFFER
#define GL_PARAMETER_BUFFER 0x80EE
#endif
#ifndef GL_PRIMITIVES_GENERATED
#define GL_PRIMITIVES_GENERATED 0x8C87
#endif
#ifndef GL_QUERY_BUFFER
#define GL_QUERY_BUFFER 0x9192
#endif
#ifndef GL_QUERY_COUNTER_BITS
#define GL_QUERY_COUNTER_BITS 0x8864
#endif
#ifndef GL_QUERY_RESULT
#define GL_QUERY_RESULT 0x8866
#endif
#ifndef GL_QUERY_RESULT_AVAILABLE
#define GL_QUERY_RESULT_AVAILABLE 0x8867
#endif
#ifndef GL_RENDERBUFFER_ALPHA_SIZE
#define GL_RENDERBUFFER_ALPHA_SIZE 0x8D53
#endif
#ifndef GL_RENDERBUFFER_BLUE_SIZE
#define GL_RENDERBUFFER_BLUE_SIZE 0x8D52
#endif
#ifndef GL_RENDERBUFFER_DEPTH_SIZE
#define GL_RENDERBUFFER_DEPTH_SIZE 0x8D54
#endif
#ifndef GL_RENDERBUFFER_GREEN_SIZE
#define GL_RENDERBUFFER_GREEN_SIZE 0x8D51
#endif
#ifndef GL_RENDERBUFFER_HEIGHT
#define GL_RENDERBUFFER_HEIGHT 0x8D43
#endif
#ifndef GL_RENDERBUFFER_INTERNAL_FORMAT
#define GL_RENDERBUFFER_INTERNAL_FORMAT 0x8D44
#endif
#ifndef GL_RENDERBUFFER_RED_SIZE
#define GL_RENDERBUFFER_RED_SIZE 0x8D50
#endif
#ifndef GL_RENDERBUFFER_SAMPLES
#define GL_RENDERBUFFER_SAMPLES 0x8CAB
#endif
#ifndef GL_RENDERBUFFER_STENCIL_SIZE
#define GL_RENDERBUFFER_STENCIL_SIZE 0x8D55
#endif
#ifndef GL_RENDERBUFFER_WIDTH
#define GL_RENDERBUFFER_WIDTH 0x8D42
#endif
#ifndef GL_RG32I
#define GL_RG32I 0x823B
#endif
#ifndef GL_RG32UI
#define GL_RG32UI 0x823C
#endif
#ifndef GL_RIGHT
#define GL_RIGHT 0x0407
#endif
#ifndef GL_SAMPLER_2D
#define GL_SAMPLER_2D 0x8B5E
#endif
#ifndef GL_SAMPLER_BINDING
#define GL_SAMPLER_BINDING 0x8919
#endif
#ifndef GL_SAMPLES
#define GL_SAMPLES 0x80A9
#endif
#ifndef GL_SAMPLES_PASSED
#define GL_SAMPLES_PASSED 0x8914
#endif
#ifndef GL_SAMPLE_BUFFERS
#define GL_SAMPLE_BUFFERS 0x80A8
#endif
#ifndef GL_SAMPLE_MASK
#define GL_SAMPLE_MASK 0x8E51
#endif
#ifndef GL_SIGNALED
#define GL_SIGNALED 0x9119
#endif
#ifndef GL_SRGB
#define GL_SRGB 0x8C40
#endif
#ifndef GL_STENCIL
#define GL_STENCIL 0x1802
#endif
#ifndef GL_SYNC_CONDITION
#define GL_SYNC_CONDITION 0x9113
#endif
#ifndef GL_SYNC_FENCE
#define GL_SYNC_FENCE 0x9116
#endif
#ifndef GL_SYNC_FLAGS
#define GL_SYNC_FLAGS 0x9115
#endif
#ifndef GL_SYNC_GPU_COMMANDS_COMPLETE
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#endif
#ifndef GL_SYNC_STATUS
#define GL_SYNC_STATUS 0x9114
#endif
#ifndef GL_TEXTURE
#define GL_TEXTURE 0x1702
#endif
#ifndef GL_TEXTURE_CUBE_MAP_ARRAY
#define GL_TEXTURE_CUBE_MAP_ARRAY 0x9009
#endif
#ifndef GL_TEXTURE_LOD_BIAS
#define GL_TEXTURE_LOD_BIAS 0x8501
#endif
#ifndef GL_TEXTURE_SWIZZLE_RGBA
#define GL_TEXTURE_SWIZZLE_RGBA 0x8E46
#endif
#ifndef GL_TEXTURE_TARGET
#define GL_TEXTURE_TARGET 0x1006
#endif
#ifndef GL_TIMESTAMP
#define GL_TIMESTAMP 0x8E28
#endif
#ifndef GL_TIME_ELAPSED
#define GL_TIME_ELAPSED 0x88BF
#endif
#ifndef GL_TRANSFORM_FEEDBACK
#define GL_TRANSFORM_FEEDBACK 0x8E22
#endif
#ifndef GL_TRANSFORM_FEEDBACK_BINDING
#define GL_TRANSFORM_FEEDBACK_BINDING 0x8E25
#endif
#ifndef GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN
#define GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN 0x8C88
#endif
#ifndef GL_UNSIGNALED
#define GL_UNSIGNALED 0x9118
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING
#define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING 0x889F
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_DIVISOR
#define GL_VERTEX_ATTRIB_ARRAY_DIVISOR 0x88FE
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_ENABLED
#define GL_VERTEX_ATTRIB_ARRAY_ENABLED 0x8622
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_INTEGER
#define GL_VERTEX_ATTRIB_ARRAY_INTEGER 0x88FD
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_NORMALIZED
#define GL_VERTEX_ATTRIB_ARRAY_NORMALIZED 0x886A
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_POINTER
#define GL_VERTEX_ATTRIB_ARRAY_POINTER 0x8645
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_SIZE
#define GL_VERTEX_ATTRIB_ARRAY_SIZE 0x8623
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_STRIDE
#define GL_VERTEX_ATTRIB_ARRAY_STRIDE 0x8624
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_TYPE
#define GL_VERTEX_ATTRIB_ARRAY_TYPE 0x8625
#endif

#endif /* __gl_h_ */
