/*
 * render_smoke.c — Mithril-Wrapper 真正的离屏渲染冒烟测试。
 *
 * 与 gl_smoke.c（纯状态机断言）不同，本测试驱动完整 GL 4.6 Core 渲染管线：
 *   离屏 FBO（纹理 attachment）→ 纹理上传 → shader 编译/链接 → VAO/VBO →
 *   glDrawArrays → glFinish（backend_commit → vkQueueSubmit）→ glReadPixels
 *   读回像素，验证 GPU 真正执行了绘制（而非仅状态机 no-op）。
 *
 * 依据 backend headless 调研结论：
 *   - proc_init() 在无 EGL 时直接 backend_init()，可 headless 创建完整
 *     Vulkan device（Device.cpp:711/1177/1181/1188），无需 surface/swapchain。
 *   - 管线创建用 VK_KHR_dynamic_rendering（Pipeline.cpp:815 renderPass=NULL），
 *     headless 可行，无需 EGL。
 *   - 离屏渲染必须绑定用户 FBO：glBindFramebuffer(0) 在无 swapchain 时没有
 *     color attachment（Framebuffer.cpp:971-977 eglDefaultColor 为 NULL）。
 *     故全程绑定自建纹理 FBO。
 *   - glFinish → backend_end_render_pass + backend_commit → vkQueueSubmit
 *     （CommandStream.cpp:1583），GL 冒烟因此会真正上 GPU。
 *   - glReadPixels → backend_read_pixels（ImageOps.cpp:886 → read_pixels:481），
 *     内部 vkWaitForFences 同步，可确认 GPU 执行完成。
 *
 * 运行（在有 libmithril.dylib 产物的 macOS 原生构建上）：
 *   clang -std=c11 -O0 -Wall -Wextra -I Mithril-Wrapper-cpp/include \
 *         -o tests/render_smoke tests/render_smoke.c -ldl
 *   DYLD_LIBRARY_PATH="$(brew --prefix)/lib" ./tests/render_smoke build/libmithril.dylib
 * 通过条件：退出码 0 且 stdout 含 "RENDER SMOKE ALL PASSED"。
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glcorearb.h>

/* 项目自带 glcorearb.h 未定义 GL_INVALID_INDEX（Khronos 头在 GL 3.0 块才定义） */
#ifndef GL_INVALID_INDEX
#define GL_INVALID_INDEX 0xFFFFFFFFu
#endif

/* ---- 依赖的 GL 函数指针 typedef（与 glcorearb.h 签名一致） -------------- */
typedef void      (*genTextures_fn)(GLsizei, GLuint*);
typedef void      (*bindTexture_fn)(GLenum, GLuint);
typedef void      (*texImage2D_fn)(GLenum, GLint, GLint, GLsizei, GLsizei,
                                   GLint, GLenum, GLenum, const void*);
typedef void      (*genFramebuffers_fn)(GLsizei, GLuint*);
typedef void      (*bindFramebuffer_fn)(GLenum, GLuint);
typedef void      (*framebufferTex2D_fn)(GLenum, GLenum, GLenum, GLuint, GLint);
typedef GLenum    (*checkFBO_fn)(GLenum);
typedef void      (*genVertexArrays_fn)(GLsizei, GLuint*);
typedef void      (*bindVertexArray_fn)(GLuint);
typedef void      (*genBuffers_fn)(GLsizei, GLuint*);
typedef void      (*bindBuffer_fn)(GLenum, GLuint);
typedef void      (*bufferData_fn)(GLenum, GLsizeiptr, const void*, GLenum);
typedef void      (*vertexAttribPtr_fn)(GLuint, GLint, GLenum, GLboolean,
                                        GLsizei, const void*);
typedef void      (*enableAttrib_fn)(GLuint);
typedef GLuint    (*createShader_fn)(GLenum);
typedef void      (*shaderSource_fn)(GLuint, GLsizei, const GLchar* const*,
                                     const GLint*);
typedef void      (*compileShader_fn)(GLuint);
typedef void      (*getShaderiv_fn)(GLuint, GLenum, GLint*);
typedef GLuint    (*createProgram_fn)(void);
typedef void      (*attachShader_fn)(GLuint, GLuint);
typedef void      (*linkProgram_fn)(GLuint);
typedef void      (*getProgramiv_fn)(GLuint, GLenum, GLint*);
typedef void      (*deleteShader_fn)(GLuint);
typedef void      (*deleteProgram_fn)(GLuint);
typedef void      (*useProgram_fn)(GLuint);
typedef void      (*viewport_fn)(GLint, GLint, GLsizei, GLsizei);
typedef void      (*clearColor_fn)(GLfloat, GLfloat, GLfloat, GLfloat);
typedef void      (*clear_fn)(GLbitfield);
typedef void      (*drawArrays_fn)(GLenum, GLint, GLsizei);
typedef void      (*finish_fn)(void);
typedef void      (*readPixels_fn)(GLint, GLint, GLsizei, GLsizei,
                                   GLenum, GLenum, void*);
typedef GLenum    (*getError_fn)(void);
typedef void      (*getIntegerv_fn)(GLenum, GLint*);
typedef const GLubyte* (*getString_fn)(GLenum);
/* ---- 扩展测试（采样 + mipmap + 动态 UBO + 多帧 glFinish）所需 ---- */
typedef void      (*texParameteri_fn)(GLenum, GLenum, GLint);
typedef void      (*texStorage2D_fn)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLsizei);
typedef void      (*texSubImage2D_fn)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*);
typedef void      (*generateMipmap_fn)(GLenum);
typedef void      (*activeTexture_fn)(GLenum);
typedef void      (*bindBufferBase_fn)(GLenum, GLuint, GLuint);
typedef GLint     (*getUniformLocation_fn)(GLuint, const GLchar*);
typedef GLuint    (*getUniformBlockIndex_fn)(GLuint, const GLchar*);
typedef void      (*uniformBlockBinding_fn)(GLuint, GLuint, GLuint);
typedef void      (*uniform1i_fn)(GLint, GLint);
typedef void      (*uniform4f_fn)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);

/* ---- 断言基础设施（Uniaball 风格） -------------------------------------- */
static int failures = 0;
static int checks = 0;
#define CHECK(cond, fmt, ...) do { \
    ++checks; \
    if (cond) { printf("ok : " fmt "\n", ##__VA_ARGS__); } \
    else      { printf("FAIL: " fmt "\n", ##__VA_ARGS__); ++failures; } \
} while (0)

/* ---- dlopen 库路径解析（与 gl_smoke.c 一致） ----------------------------- */
static void* open_libmithril(int argc, char** argv) {
    const char* candidates[4];
    int n = 0;
    if (argc > 1) candidates[n++] = argv[1];
    candidates[n++] = "./output/libmithril.dylib";
    candidates[n++] = "./build/libmithril.dylib";
    candidates[n++] = "./build/output/libmithril.dylib";
    for (int i = 0; i < n; ++i) {
        void* h = dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
        if (h) { printf("loaded: %s\n", candidates[i]); return h; }
    }
    fprintf(stderr, "dlopen failed (tried %d candidates):\n", n);
    for (int i = 0; i < n; ++i) fprintf(stderr, "  %s\n", candidates[i]);
    fprintf(stderr, "  last dlerror: %s\n", dlerror());
    return NULL;
}

#define RESOLVE(fn, sym) \
    fn = (fn##_fn)dlsym(h, sym); \
    if (!fn) { printf("FAIL: missing symbol %s\n", sym); ++failures; }

/* 离屏渲染尺寸 */
#define R 64   /* render width  */
#define C 64   /* render height */

int main(int argc, char** argv) {
    void* h = open_libmithril(argc, argv);
    if (!h) return 2;

    genTextures_fn        genTextures        = NULL;
    bindTexture_fn        bindTexture        = NULL;
    texImage2D_fn         texImage2D         = NULL;
    genFramebuffers_fn    genFramebuffers    = NULL;
    bindFramebuffer_fn    bindFramebuffer    = NULL;
    framebufferTex2D_fn framebufferTex2D = NULL;
    checkFBO_fn checkFBO       = NULL;
    genVertexArrays_fn    genVertexArrays    = NULL;
    bindVertexArray_fn    bindVertexArray    = NULL;
    genBuffers_fn         genBuffers         = NULL;
    bindBuffer_fn         bindBuffer         = NULL;
    bufferData_fn         bufferData         = NULL;
    vertexAttribPtr_fn vertexAttribPtr   = NULL;
    enableAttrib_fn enableAttrib  = NULL;
    createShader_fn       createShader       = NULL;
    shaderSource_fn       shaderSource       = NULL;
    compileShader_fn      compileShader      = NULL;
    getShaderiv_fn        getShaderiv        = NULL;
    createProgram_fn      createProgram      = NULL;
    attachShader_fn       attachShader       = NULL;
    linkProgram_fn        linkProgram        = NULL;
    getProgramiv_fn       getProgramiv       = NULL;
    deleteShader_fn       deleteShader       = NULL;
    deleteProgram_fn      deleteProgram      = NULL;
    useProgram_fn         useProgram         = NULL;
    viewport_fn           viewport           = NULL;
    clearColor_fn         clearColor         = NULL;
    clear_fn              clear              = NULL;
    drawArrays_fn         drawArrays         = NULL;
    finish_fn             finish             = NULL;
    readPixels_fn         readPixels         = NULL;
    getError_fn           getError           = NULL;
    getIntegerv_fn        getIntegerv        = NULL;
    getString_fn          getString          = NULL;
    texParameteri_fn      texParameteri      = NULL;
    texStorage2D_fn       texStorage2D       = NULL;
    texSubImage2D_fn      texSubImage2D      = NULL;
    generateMipmap_fn     generateMipmap     = NULL;
    activeTexture_fn      activeTexture      = NULL;
    bindBufferBase_fn     bindBufferBase     = NULL;
    getUniformLocation_fn getUniformLocation = NULL;
    getUniformBlockIndex_fn getUniformBlockIndex = NULL;
    uniformBlockBinding_fn uniformBlockBinding = NULL;
    uniform1i_fn          uniform1i          = NULL;
    uniform4f_fn          uniform4f          = NULL;

    RESOLVE(genTextures, "glGenTextures");
    RESOLVE(bindTexture, "glBindTexture");
    RESOLVE(texImage2D, "glTexImage2D");
    RESOLVE(genFramebuffers, "glGenFramebuffers");
    RESOLVE(bindFramebuffer, "glBindFramebuffer");
    RESOLVE(framebufferTex2D, "glFramebufferTexture2D");
    RESOLVE(checkFBO, "glCheckFramebufferStatus");
    RESOLVE(genVertexArrays, "glGenVertexArrays");
    RESOLVE(bindVertexArray, "glBindVertexArray");
    RESOLVE(genBuffers, "glGenBuffers");
    RESOLVE(bindBuffer, "glBindBuffer");
    RESOLVE(bufferData, "glBufferData");
    RESOLVE(vertexAttribPtr, "glVertexAttribPointer");
    RESOLVE(enableAttrib, "glEnableVertexAttribArray");
    RESOLVE(createShader, "glCreateShader");
    RESOLVE(shaderSource, "glShaderSource");
    RESOLVE(compileShader, "glCompileShader");
    RESOLVE(getShaderiv, "glGetShaderiv");
    RESOLVE(createProgram, "glCreateProgram");
    RESOLVE(attachShader, "glAttachShader");
    RESOLVE(linkProgram, "glLinkProgram");
    RESOLVE(getProgramiv, "glGetProgramiv");
    RESOLVE(deleteShader, "glDeleteShader");
    RESOLVE(deleteProgram, "glDeleteProgram");
    RESOLVE(useProgram, "glUseProgram");
    RESOLVE(viewport, "glViewport");
    RESOLVE(clearColor, "glClearColor");
    RESOLVE(clear, "glClear");
    RESOLVE(drawArrays, "glDrawArrays");
    RESOLVE(finish, "glFinish");
    RESOLVE(readPixels, "glReadPixels");
    RESOLVE(getError, "glGetError");
    RESOLVE(getIntegerv, "glGetIntegerv");
    RESOLVE(getString, "glGetString");
    RESOLVE(texParameteri, "glTexParameteri");
    RESOLVE(texStorage2D, "glTexStorage2D");
    RESOLVE(texSubImage2D, "glTexSubImage2D");
    RESOLVE(generateMipmap, "glGenerateMipmap");
    RESOLVE(activeTexture, "glActiveTexture");
    RESOLVE(bindBufferBase, "glBindBufferBase");
    RESOLVE(getUniformLocation, "glGetUniformLocation");
    RESOLVE(getUniformBlockIndex, "glGetUniformBlockIndex");
    RESOLVE(uniformBlockBinding, "glUniformBlockBinding");
    RESOLVE(uniform1i, "glUniform1i");
    RESOLVE(uniform4f, "glUniform4f");
    if (failures) { printf("RENDER SMOKE FAILED (missing symbols)\n"); dlclose(h); return 1; }

    /* ---- 版本（走 backend 起来后的 glGetIntegerv，验证偏移修复） ---------- */
    GLint major = 0, minor = 0;
    getIntegerv(GL_MAJOR_VERSION, &major);
    getIntegerv(GL_MINOR_VERSION, &minor);
    CHECK(major == 4 && minor == 6,
          "GL version %d.%d (backend-up, glGetIntegerv un-hijacked)", major, minor);
    const char* ver = (const char*)getString(GL_VERSION);
    CHECK(ver && strstr(ver, "4.6"), "glGetString(GL_VERSION): %s", ver ? ver : "(null)");
    if (getError() != GL_NO_ERROR) { printf("FAIL: GL error before setup\n"); ++failures; }

    /* ---- 离屏 FBO：RGBA8 纹理作为 color attachment ------------------------ */
    GLuint colorTex = 0, fbo = 0;
    genTextures(1, &colorTex);
    CHECK(colorTex != 0, "genTextures allocated color texture (%u)", colorTex);
    bindTexture(GL_TEXTURE_2D, colorTex);
    texImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, R, C, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    genFramebuffers(1, &fbo);
    bindFramebuffer(GL_FRAMEBUFFER, fbo);
    framebufferTex2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);
    GLenum fboStatus = checkFBO(GL_FRAMEBUFFER);
    CHECK(fboStatus == GL_FRAMEBUFFER_COMPLETE,
          "offscreen FBO complete (status=0x%x)", fboStatus);

    viewport(0, 0, R, C);
    clearColor(0.0f, 0.0f, 0.0f, 0.0f);
    clear(GL_COLOR_BUFFER_BIT);
    CHECK(getError() == GL_NO_ERROR, "clear on offscreen FBO leaves no error");

    /* ---- 控制组诊断：clear 到已知非零颜色并读回，隔离 readback/clear 路径 ----
     * 若此读回返回 (25,51,76,255)（=0.1/0.2/0.3*255），说明 glClear+glReadPixels
     * 整条 GPU 写读路径正常，后续 draw 读回全黑就指向 draw 本身；若此读回也是全 0，
     * 说明 clear/readback 或离屏渲染目标本身有问题。 */
    clearColor(0.1f, 0.2f, 0.3f, 1.0f);
    clear(GL_COLOR_BUFFER_BIT);
    finish();  /* 提交 clear 的 command buffer，确保 GPU 执行完毕 */
    unsigned char ctrl[4] = {0,0,0,0};
    readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, ctrl);
    /* float→uint8 转换会四舍五入：0.1*255=25.5→26, 0.3*255=76.5→77。故对
     * (0.1,0.2,0.3,1.0) 期望区间为 r∈[24,27], g==51±1, b∈[75,78], a==255。 */
    CHECK((ctrl[0] >= 24 && ctrl[0] <= 27) &&
          (ctrl[1] >= 50 && ctrl[1] <= 52) &&
          (ctrl[2] >= 75 && ctrl[2] <= 78) &&
          ctrl[3] == 255,
          "CONTROL clear-to-(0.1,0.2,0.3,1.0) readback=(%d,%d,%d,%d) — clear+readback path %s",
          ctrl[0], ctrl[1], ctrl[2], ctrl[3],
          ((ctrl[0]>=24&&ctrl[0]<=27)&&(ctrl[1]>=50&&ctrl[1]<=52)&&
           (ctrl[2]>=75&&ctrl[2]<=78)&&ctrl[3]==255) ? "OK" : "BROKEN");
    /* 恢复黑色背景再继续 draw 测试 */
    clearColor(0.0f, 0.0f, 0.0f, 0.0f);
    clear(GL_COLOR_BUFFER_BIT);
    finish();
    CHECK(getError() == GL_NO_ERROR, "control clear/readback leaves no error");

    /* ---- shader 编译 + program 链接（glslang -> SPIR-V -> pipeline）-------- */
    const char* vsSrc = "#version 330 core\n"
                        "layout(location=0) in vec2 aPos;\n"
                        "void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }\n";
    const char* fsSrc = "#version 330 core\n"
                        "out vec4 fragColor;\n"
                        "void main(){ fragColor = vec4(1.0, 0.0, 0.0, 1.0); }\n";
    GLuint vs = createShader(GL_VERTEX_SHADER);
    GLuint fs = createShader(GL_FRAGMENT_SHADER);
    shaderSource(vs, 1, &vsSrc, NULL);
    shaderSource(fs, 1, &fsSrc, NULL);
    compileShader(vs);
    compileShader(fs);
    GLint vsOk = 0, fsOk = 0;
    getShaderiv(vs, GL_COMPILE_STATUS, &vsOk);
    getShaderiv(fs, GL_COMPILE_STATUS, &fsOk);
    CHECK(vsOk == GL_TRUE, "vertex shader compiled");
    CHECK(fsOk == GL_TRUE, "fragment shader compiled");
    GLuint prog = createProgram();
    attachShader(prog, vs);
    attachShader(prog, fs);
    linkProgram(prog);
    GLint linkOk = 0;
    getProgramiv(prog, GL_LINK_STATUS, &linkOk);
    CHECK(linkOk == GL_TRUE, "program linked (GL_LINK_STATUS=%d)", linkOk);
    useProgram(prog);
    deleteShader(vs);
    deleteShader(fs);
    CHECK(getError() == GL_NO_ERROR, "shader compile/link leaves no error");

    /* ---- VAO + VBO：全屏三角形（三个顶点覆盖中心区） ---------------------- */
    GLuint vao = 0, vbo = 0;
    const GLfloat verts[6] = {
        -1.0f, -1.0f,
         3.0f, -1.0f,
        -1.0f,  3.0f,
    };
    genVertexArrays(1, &vao);
    bindVertexArray(vao);
    genBuffers(1, &vbo);
    bindBuffer(GL_ARRAY_BUFFER, vbo);
    bufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    vertexAttribPtr(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (const void*)0);
    enableAttrib(0);
    CHECK(getError() == GL_NO_ERROR, "VAO/VBO setup leaves no error");

    /* ---- 绘制 + 提交（glFinish -> backend_commit -> vkQueueSubmit）--------- */
    drawArrays(GL_TRIANGLES, 0, 3);
    CHECK(getError() == GL_NO_ERROR, "glDrawArrays leaves no error");
    finish();
    CHECK(getError() == GL_NO_ERROR, "glFinish leaves no error");

    /* ---- 读回像素，验证 GPU 真正执行绘制 ---------------------------------- */
    /* 全屏三角形覆盖整个 viewport：中心应是红色，四角也应是红色（三角形盖满）。 */
    unsigned char px[4] = {0,0,0,0};
    readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    CHECK(px[0] > 128 && px[3] > 128,
          "center pixel is red (r=%d g=%d b=%d a=%d) — GPU draw executed", px[0], px[1], px[2], px[3]);
    readPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    CHECK(px[0] > 128 && px[3] > 128,
          "corner pixel is red (r=%d g=%d b=%d a=%d) — full-screen triangle rasterized",
          px[0], px[1], px[2], px[3]);
    CHECK(getError() == GL_NO_ERROR, "glReadPixels leaves no error");

    /* =====================================================================
     * 扩展测试：贴图采样 + mipmap + 动态 UBO + 多帧 glFinish 稳定性
     * =====================================================================
     * 目的（对齐 MC 真实动态 3D 场景的渲染路径，而非仅画一个纯色三角形）：
     *   1) 贴图采样：绑定一张 2D 纹理并在 fragment shader 里 texture() 采样，
     *      验证 glActiveTexture/glBindTexture/glTexImage2D/采样器链路。
     *   2) mipmap 危险路径：min filter 用 GL_LINEAR_MIPMAP_LINEAR 并对单级
     *      视图调用 glGenerateMipmap —— 这正是 MC 加载屏 GUI 图集采样触发
     *      GPU page fault（kIOGPUCommandBufferCallbackErrorPageFault）的路径。
     *      后端须对该单级视图强制 VK_SAMPLER_MIPMAP_MODE_NEAREST + maxLod=0
     *      （对齐 MobileGL），否则 texture unit 会越界取 level+1。
     *   3) 动态 UBO：用 glBindBufferBase 绑定一个 uniform block，每帧改写
     *      glBufferData（GL_DYNAMIC_DRAW）触发后端 orphan-rename + descriptor
     *      memo 失效，验证动态 uniform 每帧生效、无 page fault。
     *   4) 多帧 glFinish：连续 N 帧各执行 draw + glFinish（vkQueueSubmit）
     *      + readback，验证跨帧描述符复用/缓冲生命周期稳定。
     * ===================================================================== */
    {
        /* 共享的 attrib-0-only 顶点着色器：由 4e 判别测试创建，供 4f UBO-only
         * 复用，以隔离"顶点 shader 带 UV attrib"这一变量（见 4f 注释）。 */
        GLuint attrib0Vs = 0;

        /* ---- 4a) 生成全白采样纹理（2x2）------------------------------- */
        GLuint sampTex = 0;
        genTextures(1, &sampTex);
        CHECK(sampTex != 0, "genTextures allocated sample texture (%u)", sampTex);
        activeTexture(GL_TEXTURE0);
        bindTexture(GL_TEXTURE_2D, sampTex);
        const GLubyte white[16] = { 255,255,255,255, 255,255,255,255,
                                    255,255,255,255, 255,255,255,255 };
        texImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
        /* 关键：min filter 走 mipmap 线性，mip 链用 glGenerateMipmap 生成。
         * 这命中单级视图 + LINEAR mipmap 的页错误路径。 */
        texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        generateMipmap(GL_TEXTURE_2D);
        CHECK(getError() == GL_NO_ERROR, "texture upload + mipmap generation leaves no error");

        /* ---- 4a.5) 判别 F：glTexStorage2D 完整 mip 链 → 「非重建」mipmap 分支 -----
         * 4a 用 glTexImage2D(level0) → tex.levels=1<fullLevels，走 generate_mipmaps
         * 的【重建】分支（已由 CI 覆盖）。但真机 atlas 用 glTexStorage2D 分配完整
         * mip 链（tex.levels==fullLevels），generateMipmap 走【非重建】分支——该分支
         * 此前未在 CI 覆盖，且漏了「one-shot blit 前 flush 主 command buffer 同步
         * pending upload」的修复（与重建分支同源 bug）：
         *   texSubImage2D(level0) 记录到主 buffer 未提交 → glGenerateMipmap 用
         *   one-shot 读 level0 → 读到未初始化数据 → mip 全 garbage → 采样越界 page
         *   fault（正是真机主菜单 atlas 纯红 + kIOGPUCommandBufferCallbackErrorPageFault）。
         * 本用例精确命中该路径：glTexStorage2D(levels=2) + texSubImage2D(level0) +
         * generateMipmap → 采样读回应为白。修复后（safe_device_wait_idle）读回白，
         * 否则黑/崩。 */
        {
            GLuint stTex = 0;
            genTextures(1, &stTex);
            bindTexture(GL_TEXTURE_2D, stTex);
            texStorage2D(GL_TEXTURE_2D, 2, GL_RGBA8, 2, 2, 0);  /* 完整 2 级 mip 链 */
            CHECK(getError() == GL_NO_ERROR, "disc-F texStorage2D allocated full 2-level chain");
            const GLubyte stwhite[16] = { 255,255,255,255, 255,255,255,255,
                                          255,255,255,255, 255,255,255,255 };
            /* level0 经主 command buffer 上传（未 flush），随即 generateMipmap 走
             * 非重建分支——正是要验证的同步点。 */
            texSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2, 2, GL_RGBA, GL_UNSIGNED_BYTE, stwhite);
            /* 判别 A 前先不用 mipmap filter，单独验证 texStorage2D 的 level0 上传+采样 */
            texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

            /* 采样读回用已有的 attrib0 全屏三角形 vao/vbo。两个程序共用 attrib0 vs，
             * fs 不同：fprogA 采样 level0（texture()，UV 恒定 LOD→level0）；
             * fprogB 用 textureLod 强制采样 level1（CASCADE blit 生成的目标）。 */
            GLuint fvs = createShader(GL_VERTEX_SHADER);
            shaderSource(fvs, 1, &(const char*){
                "#version 330 core\n"
                "layout(location = 0) in vec2 aPos;\n"
                "void main() { gl_Position = vec4(aPos, 0.0, 1.0); }\n" }, NULL);
            compileShader(fvs);
            GLuint ffsA = createShader(GL_FRAGMENT_SHADER);
            shaderSource(ffsA, 1, &(const char*){
                "#version 330 core\n"
                "uniform sampler2D uTex;\n"
                "out vec4 fragColor;\n"
                "void main() { fragColor = texture(uTex, vec2(0.5, 0.5)); }\n" }, NULL);
            compileShader(ffsA);
            GLuint ffsB = createShader(GL_FRAGMENT_SHADER);
            shaderSource(ffsB, 1, &(const char*){
                "#version 330 core\n"
                "uniform sampler2D uTex;\n"
                "out vec4 fragColor;\n"
                "void main() { fragColor = textureLod(uTex, vec2(0.5, 0.5), 1.0); }\n" }, NULL);
            compileShader(ffsB);
            GLuint fprogA = createProgram();
            attachShader(fprogA, fvs);
            attachShader(fprogA, ffsA);
            linkProgram(fprogA);
            GLuint fprogB = createProgram();
            attachShader(fprogB, fvs);
            attachShader(fprogB, ffsB);
            linkProgram(fprogB);
            deleteShader(fvs);
            deleteShader(ffsA);
            deleteShader(ffsB);
            activeTexture(GL_TEXTURE0);
            bindTexture(GL_TEXTURE_2D, stTex);
            bindVertexArray(vao);

            /* 判别 A：finish() flush level0 upload 后，无 mipmap 直接采样 level0。
             * 隔离「texStorage2D 的 level0 上传」是否正常（若 A 黑，问题在上传，与
             * mipmap 无关；若 A 白，问题在 CASCADE blit 生成的 level1）。 */
            finish();
            useProgram(fprogA);
            GLint uTexLocA = getUniformLocation(fprogA, "uTex");
            CHECK(uTexLocA >= 0, "disc-F getUniformLocation(uTex)=%d", uTexLocA);
            if (uTexLocA >= 0) uniform1i(uTexLocA, 0);   /* sampler2D 使用 texture unit 0 */
            drawArrays(GL_TRIANGLES, 0, 3);
            finish();
            unsigned char fa[4] = {0,0,0,0};
            readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, fa);
            CHECK(getError() == GL_NO_ERROR, "disc-F(A) level0 sampling readback leaves no error");
            CHECK(fa[0] > 200 && fa[1] > 200 && fa[2] > 200 && fa[3] > 128,
                  "disc-F(A) texStorage2D level0 upload+sample is WHITE "
                  "(r=%d g=%d b=%d a=%d) — upload path OK", fa[0], fa[1], fa[2], fa[3]);

            /* 判别 B：切到 mipmap filter 触发 CASCADE 非重建分支生成 level1，再
             * textureLod(level1) 验证 blit 结果。若 A 白而 B 黑 → CASCADE blit 生成
             * 黑 level1；若 B 也白 → CASCADE 修复正确。 */
            texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            generateMipmap(GL_TEXTURE_2D);
            CHECK(getError() == GL_NO_ERROR, "disc-F storage mipmap generation leaves no error");
            useProgram(fprogB);
            GLint uTexLocB = getUniformLocation(fprogB, "uTex");
            CHECK(uTexLocB >= 0, "disc-F(B) getUniformLocation(uTex)=%d", uTexLocB);
            if (uTexLocB >= 0) uniform1i(uTexLocB, 0);   /* sampler2D 使用 texture unit 0 */
            drawArrays(GL_TRIANGLES, 0, 3);
            finish();
            unsigned char fb[4] = {0,0,0,0};
            readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, fb);
            CHECK(getError() == GL_NO_ERROR, "disc-F(B) level1 sampling readback leaves no error");
            CHECK(fb[0] > 200 && fb[1] > 200 && fb[2] > 200 && fb[3] > 128,
                  "disc-F(B) CASCADE non-rebuild mipmap level1 is WHITE "
                  "(r=%d g=%d b=%d a=%d) — cross-buffer sync + blit OK", fb[0], fb[1], fb[2], fb[3]);
            deleteProgram(fprogA);
            deleteProgram(fprogB);
            /* stTex 走 defer_destroy（后端持有跨帧存活），此处不手动删除 */
        }

        /* ---- 4b) 动态 UBO（每帧改写，触发 orphan-rename）--------------- */
        GLuint ubo = 0;
        genBuffers(1, &ubo);
        bindBuffer(GL_UNIFORM_BUFFER, ubo);
        GLfloat initColor[4] = { 0.0f, 0.0f, 1.0f, 1.0f };  /* 初始蓝色 */
        bufferData(GL_UNIFORM_BUFFER, sizeof(initColor), initColor, GL_DYNAMIC_DRAW);
        CHECK(getError() == GL_NO_ERROR, "dynamic UBO initial bufferData leaves no error");

        /* ---- 4c) 采样 + UBO 的 fragment shader -------------------------- */
        const char* texVsSrc = "#version 330 core\n"
                               "layout(location=0) in vec2 aPos;\n"
                               "layout(location=1) in vec2 aUV;\n"
                               "out vec2 vUV;\n"
                               "void main(){ vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }\n";
        const char* texFsSrc = "#version 330 core\n"
                               "in vec2 vUV;\n"
                               "out vec4 fragColor;\n"
                               "layout(std140) uniform DynColor { vec4 tint; };\n"
                               "uniform sampler2D uTex;\n"
                               "void main(){ fragColor = texture(uTex, vUV) * tint; }\n";
        GLuint texVs = createShader(GL_VERTEX_SHADER);
        GLuint texFs = createShader(GL_FRAGMENT_SHADER);
        shaderSource(texVs, 1, &texVsSrc, NULL);
        shaderSource(texFs, 1, &texFsSrc, NULL);
        compileShader(texVs);
        compileShader(texFs);
        GLint tvsOk = 0, tfsOk = 0;
        getShaderiv(texVs, GL_COMPILE_STATUS, &tvsOk);
        getShaderiv(texFs, GL_COMPILE_STATUS, &tfsOk);
        CHECK(tvsOk == GL_TRUE, "sampled vertex shader compiled");
        CHECK(tfsOk == GL_TRUE, "sampled fragment shader compiled");
        GLuint texProg = createProgram();
        attachShader(texProg, texVs);
        attachShader(texProg, texFs);
        linkProgram(texProg);
        GLint texLinkOk = 0;
        getProgramiv(texProg, GL_LINK_STATUS, &texLinkOk);
        CHECK(texLinkOk == GL_TRUE, "sampled program linked (GL_LINK_STATUS=%d)", texLinkOk);
        /* texVs 在后续隔离子测试 4e/4f 复用，故此处不 deleteShader(texVs)。
         * texFs 也不再单独 delete——shader 会在进程退出时统一回收，避免误删。 */

        /* 查询 UBO 块索引 + 采样器 uniform location */
        GLuint blockIdx = getUniformBlockIndex(texProg, "DynColor");
        CHECK(blockIdx != GL_INVALID_INDEX, "getUniformBlockIndex(DynColor)=%u", blockIdx);
        GLint texLoc = getUniformLocation(texProg, "uTex");
        CHECK(texLoc >= 0, "getUniformLocation(uTex)=%d", texLoc);
        GLint tintLoc = getUniformLocation(texProg, "tint");
        CHECK(tintLoc >= 0, "getUniformLocation(tint)=%d", tintLoc);

        useProgram(texProg);
        uniformBlockBinding(texProg, blockIdx, 1);   /* 绑定到 binding point 1 */
        bindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);   /* UBO 挂到 binding point 1 */
        activeTexture(GL_TEXTURE0);
        bindTexture(GL_TEXTURE_2D, sampTex);
        if (texLoc >= 0) uniform1i(texLoc, 0);        /* sampler2D 使用 texture unit 0 */
        CHECK(getError() == GL_NO_ERROR, "sampled program UBO+sampler binding leaves no error");

        /* ---- 4d) 带 UV 的全屏三角形 VAO（aPos=0, aUV=1）----------------- */
        GLuint texVao = 0, texVbo = 0;
        const GLfloat texVerts[20] = {
            /* aPos (2) + aUV (2) */
            -1.0f, -1.0f,  0.0f, 0.0f,
             3.0f, -1.0f,  2.0f, 0.0f,
            -1.0f,  3.0f,  0.0f, 2.0f,
        };
        genVertexArrays(1, &texVao);
        bindVertexArray(texVao);
        genBuffers(1, &texVbo);
        bindBuffer(GL_ARRAY_BUFFER, texVbo);
        bufferData(GL_ARRAY_BUFFER, sizeof(texVerts), texVerts, GL_STATIC_DRAW);
        vertexAttribPtr(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (const void*)0);
        enableAttrib(0);
        vertexAttribPtr(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                        (const void*)(2 * sizeof(GLfloat)));
        enableAttrib(1);
        CHECK(getError() == GL_NO_ERROR, "sampled VAO/VBO (pos+uv) setup leaves no error");

        /* ---- 4e) 判别测试 C：空 program（无 descriptor，常量绿）+ texVao
         * 关键判别：红色三角形（prog）用 vao（只有 attrib 0）能画；所有带
         * descriptor 的新 program 都用 texVao（attrib 0+1）。此测试用【空
         * program + 常量绿 + texVao】，隔离两个变量：
         *   - 若绿 → texVao/texVbo 多 attrib VAO 与【新 program】pipeline 都
         *           正常，黑屏问题 100% 出在 descriptor 绑定（UBO/贴图）。
         *   - 若黑 → 问题在 texVao 多 attrib VAO 或【新 program】的 pipeline
         *           创建/反射本身，与 UBO/贴图 descriptor 无关。
         * 绿色以便与红色控制组区分。 */
        {
            const char* discVs = "#version 330 core\n"
                                 "layout(location=0) in vec2 aPos;\n"
                                 "void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }\n";
            const char* discFs = "#version 330 core\n"
                                 "out vec4 fragColor;\n"
                                 "void main(){ fragColor = vec4(0.0, 1.0, 0.0, 1.0); }\n";
            GLuint dVs = createShader(GL_VERTEX_SHADER);
            GLuint dFs = createShader(GL_FRAGMENT_SHADER);
            shaderSource(dVs, 1, &discVs, NULL);
            shaderSource(dFs, 1, &discFs, NULL);
            compileShader(dVs);
            compileShader(dFs);
            GLuint discProg = createProgram();
            attachShader(discProg, dVs);
            attachShader(discProg, dFs);
            linkProgram(discProg);
            GLint discLink = 0;
            getProgramiv(discProg, GL_LINK_STATUS, &discLink);
            CHECK(discLink == GL_TRUE, "discriminant empty program linked (GL_LINK_STATUS=%d)", discLink);
            /* 保留 dVs 供 4f UBO-only 复用，不 deleteShader(dVs)。 */
            attrib0Vs = dVs;
            deleteShader(dFs);
            useProgram(discProg);
            bindVertexArray(texVao);              /* 复用 4d 的 pos+uv VAO */
            clearColor(0.0f, 0.0f, 0.0f, 0.0f);
            clear(GL_COLOR_BUFFER_BIT);
            drawArrays(GL_TRIANGLES, 0, 3);
            finish();
            unsigned char dpx[4] = {0,0,0,0};
            readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, dpx);
            CHECK(dpx[1] > 128 && dpx[3] > 128 && dpx[0] < 40,
                  "DISCRIMINANT empty-program + texVao = green (g=%d r=%d b=%d a=%d)",
                  dpx[1], dpx[0], dpx[2], dpx[3]);
        }

        /* ---- 4f) 隔离诊断 A：仅 UBO（无贴图），用 attrib-0-only 顶点 shader ----
         * 关键判别：上一轮 UBO-only 用 texVs（attrib 0+1）读回黑；判别测试用
         * discVs（attrib 0 only）绿。为区分"UBO descriptor"与"texVs UV 顶点
         * shader 的 pipeline"两个变量，此测试改用已验证能画的 attrib0Vs：
         *   - 若变红 → texVs（带 UV attrib）的 pipeline 是黑屏根因，UBO 正常。
         *   - 若仍黑 → 问题确在 UBO descriptor 绑定（读了 0），与顶点 shader 无关。 */
        {
            const char* uboFsSrc = "#version 330 core\n"
                                   "out vec4 fragColor;\n"
                                   "layout(std140) uniform DynColor { vec4 tint; };\n"
                                   "void main(){ fragColor = tint; }\n";
            GLuint uboFs = createShader(GL_FRAGMENT_SHADER);
            shaderSource(uboFs, 1, &uboFsSrc, NULL);
            compileShader(uboFs);
            GLint uboFsOk = 0;
            getShaderiv(uboFs, GL_COMPILE_STATUS, &uboFsOk);
            CHECK(uboFsOk == GL_TRUE, "UBO-only fragment shader compiled");
            GLuint uboProg = createProgram();
            attachShader(uboProg, attrib0Vs);   /* attrib-0-only，已验证能画 */
            attachShader(uboProg, uboFs);
            linkProgram(uboProg);
            GLint uboLinkOk = 0;
            getProgramiv(uboProg, GL_LINK_STATUS, &uboLinkOk);
            CHECK(uboLinkOk == GL_TRUE, "UBO-only program linked (GL_LINK_STATUS=%d)", uboLinkOk);
            deleteShader(uboFs);
            GLuint uboBlk = getUniformBlockIndex(uboProg, "DynColor");
            uniformBlockBinding(uboProg, uboBlk, 1);
            useProgram(uboProg);
            bindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);   /* 复用 4b 的 UBO */
            bindVertexArray(texVao);                      /* 复用 4d 的 pos+uv VAO */
            GLfloat uTint[4] = { 0.5f, 0.0f, 0.0f, 1.0f };
            bindBuffer(GL_UNIFORM_BUFFER, ubo);
            bufferData(GL_UNIFORM_BUFFER, sizeof(uTint), uTint, GL_DYNAMIC_DRAW);
            clearColor(0.0f, 0.0f, 0.0f, 0.0f);
            clear(GL_COLOR_BUFFER_BIT);
            drawArrays(GL_TRIANGLES, 0, 3);
            finish();
            unsigned char ubopx[4] = {0,0,0,0};
            readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, ubopx);
            /* tint=(0.5,0,0,1) → 期望 r≈128±15, g,b≈0, a=255 */
            CHECK(ubopx[0] >= 110 && ubopx[0] <= 145 &&
                  ubopx[1] < 20 && ubopx[2] < 20 && ubopx[3] > 128,
                  "UBO-only path: tint=(0.5,0,0,1) readback=(%d,%d,%d,%d)",
                  ubopx[0], ubopx[1], ubopx[2], ubopx[3]);
        }

        /* ---- 4f) 隔离诊断 B：仅贴图采样（无 UBO tint，乘白色常量）--------
         * fragColor = texture(uTex, vUV) * vec4(1)，全白 2x2 纹理 → 读回应全白。
         * 若此读回为白，则贴图采样链路正确，黑屏出在 UBO。 */
        {
            const char* texOnlyFs = "#version 330 core\n"
                                    "in vec2 vUV;\n"
                                    "out vec4 fragColor;\n"
                                    "uniform sampler2D uTex;\n"
                                    "void main(){ fragColor = texture(uTex, vUV); }\n";
            GLuint toFs = createShader(GL_FRAGMENT_SHADER);
            shaderSource(toFs, 1, &texOnlyFs, NULL);
            compileShader(toFs);
            GLint toFsOk = 0;
            getShaderiv(toFs, GL_COMPILE_STATUS, &toFsOk);
            CHECK(toFsOk == GL_TRUE, "texture-only fragment shader compiled");
            GLuint toProg = createProgram();
            attachShader(toProg, texVs);
            attachShader(toProg, toFs);
            linkProgram(toProg);
            GLint toLinkOk = 0;
            getProgramiv(toProg, GL_LINK_STATUS, &toLinkOk);
            CHECK(toLinkOk == GL_TRUE, "texture-only program linked (GL_LINK_STATUS=%d)", toLinkOk);
            deleteShader(toFs);
            GLint toLoc = getUniformLocation(toProg, "uTex");
            useProgram(toProg);
            activeTexture(GL_TEXTURE0);
            bindTexture(GL_TEXTURE_2D, sampTex);
            if (toLoc >= 0) uniform1i(toLoc, 0);
            bindVertexArray(texVao);
            clearColor(0.0f, 0.0f, 0.0f, 0.0f);
            clear(GL_COLOR_BUFFER_BIT);
            drawArrays(GL_TRIANGLES, 0, 3);
            finish();
            unsigned char topx[4] = {0,0,0,0};
            readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, topx);
            /* 全白纹理 → 期望 r,g,b≥200, a=255 */
            CHECK(topx[0] >= 200 && topx[1] >= 200 && topx[2] >= 200 && topx[3] > 128,
                  "texture-only path: white 2x2 sampled readback=(%d,%d,%d,%d)",
                  topx[0], topx[1], topx[2], topx[3]);
        }

        /* ---- 4h) 判别 C：texVs（attrib 0+1）program 能否画纯色 ----------
         * UBO-only 已用 attrib0Vs（attrib 0 only）验证红；本测试用 texVs
         * （attrib 0+1）+ 纯色 FS（无 sampler、无 UBO）画蓝色。若蓝 → texVs
         * 的 attrib-1 pipeline 正常，黑屏出在 sampler descriptor；若黑 →
         * attrib-1 顶点 pipeline 是根因。 */
        {
            const char* blueFs = "#version 330 core\n"
                                 "out vec4 fragColor;\n"
                                 "void main(){ fragColor = vec4(0.0, 0.0, 1.0, 1.0); }\n";
            GLuint bFs = createShader(GL_FRAGMENT_SHADER);
            shaderSource(bFs, 1, &blueFs, NULL);
            compileShader(bFs);
            GLuint blueProg = createProgram();
            attachShader(blueProg, texVs);   /* attrib 0+1 */
            attachShader(blueProg, bFs);
            linkProgram(blueProg);
            deleteShader(bFs);
            useProgram(blueProg);
            bindVertexArray(texVao);
            clearColor(0.0f, 0.0f, 0.0f, 0.0f);
            clear(GL_COLOR_BUFFER_BIT);
            drawArrays(GL_TRIANGLES, 0, 3);
            finish();
            unsigned char bp[4] = {0,0,0,0};
            readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, bp);
            CHECK(bp[2] > 128 && bp[3] > 128 && bp[0] < 40,
                  "disc-C texVs attrib-1 pipeline: blue readback=(%d,%d,%d,%d)",
                  bp[0], bp[1], bp[2], bp[3]);
        }

        /* ---- 4i) 判别 D：attrib0Vs + sampler（常量 UV）采样 ------------
         * 隔离 sampler descriptor 链路：attrib0Vs（已验证能画）+ 一个用常量
         * UV 采样全白纹理的 FS。若白 → sampler descriptor + attrib0Vs 正常；
         * 若黑 → sampler descriptor 链路（纹理 view / unit 映射 / 描述符写入）
         * 是根因。同时打印 uTex location 诊断映射是否建立。 */
        {
            const char* sampFs = "#version 330 core\n"
                                 "out vec4 fragColor;\n"
                                 "uniform sampler2D uTex;\n"
                                 "void main(){ fragColor = texture(uTex, vec2(0.5, 0.5)); }\n";
            GLuint sFs = createShader(GL_FRAGMENT_SHADER);
            shaderSource(sFs, 1, &sampFs, NULL);
            compileShader(sFs);
            GLuint sampProg = createProgram();
            attachShader(sampProg, attrib0Vs);   /* attrib 0 only */
            attachShader(sampProg, sFs);
            linkProgram(sampProg);
            deleteShader(sFs);
            GLint sLoc = getUniformLocation(sampProg, "uTex");
            printf("disc-D getUniformLocation(uTex)=%d\n", sLoc);
            useProgram(sampProg);
            activeTexture(GL_TEXTURE0);
            bindTexture(GL_TEXTURE_2D, sampTex);
            if (sLoc >= 0) uniform1i(sLoc, 0);
            bindVertexArray(texVao);
            clearColor(0.0f, 0.0f, 0.0f, 0.0f);
            clear(GL_COLOR_BUFFER_BIT);
            drawArrays(GL_TRIANGLES, 0, 3);
            finish();
            unsigned char sp[4] = {0,0,0,0};
            readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, sp);
            CHECK(sp[0] > 200 && sp[1] > 200 && sp[2] > 200 && sp[3] > 128,
                  "disc-D sampler+attrib0Vs: white readback=(%d,%d,%d,%d)",
                  sp[0], sp[1], sp[2], sp[3]);
        }

        /* ---- 4j) 判别 E：无 mipmap 单级纹理采样 -------------------------
         * 隔离「基础 upload+采样」 vs 「generateMipmap 重建」两个环节。用一张
         * 单独的单级白色纹理（不 generateMipmap，min filter=GL_LINEAR），配
         * attrib0Vs + 常量 UV 采样。
         *   - 若白 → 基础 upload+采样正常，黑屏出在 generateMipmap 重建链；
         *   - 若黑 → 基础 upload/采样链路本身有问题（与 mipmap 无关）。 */
        {
            GLuint singleTex = 0;
            genTextures(1, &singleTex);
            activeTexture(GL_TEXTURE0);
            bindTexture(GL_TEXTURE_2D, singleTex);
            const GLubyte swhite[16] = { 255,255,255,255, 255,255,255,255,
                                         255,255,255,255, 255,255,255,255 };
            texImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, swhite);
            texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); /* 非 mipmap */
            texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            const char* eFs = "#version 330 core\n"
                              "out vec4 fragColor;\n"
                              "uniform sampler2D uTex;\n"
                              "void main(){ fragColor = texture(uTex, vec2(0.5, 0.5)); }\n";
            GLuint eFsObj = createShader(GL_FRAGMENT_SHADER);
            shaderSource(eFsObj, 1, &eFs, NULL);
            compileShader(eFsObj);
            GLuint eProg = createProgram();
            attachShader(eProg, attrib0Vs);
            attachShader(eProg, eFsObj);
            linkProgram(eProg);
            deleteShader(eFsObj);
            GLint eLoc = getUniformLocation(eProg, "uTex");
            useProgram(eProg);
            activeTexture(GL_TEXTURE0);
            bindTexture(GL_TEXTURE_2D, singleTex);
            if (eLoc >= 0) uniform1i(eLoc, 0);
            bindVertexArray(texVao);
            clearColor(0.0f, 0.0f, 0.0f, 0.0f);
            clear(GL_COLOR_BUFFER_BIT);
            drawArrays(GL_TRIANGLES, 0, 3);
            finish();
            unsigned char ep[4] = {0,0,0,0};
            readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, ep);
            CHECK(ep[0] > 200 && ep[1] > 200 && ep[2] > 200 && ep[3] > 128,
                  "disc-E single-level (no mipmap) sampler: white readback=(%d,%d,%d,%d)",
                  ep[0], ep[1], ep[2], ep[3]);
        }

        /* ---- 4g) 多帧动态 UBO + 贴图 + glFinish 稳定性 ------------------- */
        /* 每帧：改写 UBO（orphan-rename + descriptor memo 失效）→ draw →
         * glFinish(vkQueueSubmit) → readback。连续 N 帧验证：
         *   - 每帧读回颜色随 UBO tint 变化而变（动态 uniform 真正生效）；
         *   - 全白纹理 × tint 的乘积 = tint 本身（验证采样链路颜色正确）；
         *   - 全程无 GPU page fault / draw 丢弃（读回非黑即证明 draw 未被丢）。 */
        const int NFRAMES = 8;
        int stableFrames = 0;
        /* 前面 4e/4f 换了 bound program/texture，这里恢复 texProg + 采样器 +
         * UBO 的完整绑定状态后再进入多帧循环。 */
        useProgram(texProg);
        uniformBlockBinding(texProg, blockIdx, 1);
        bindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);
        activeTexture(GL_TEXTURE0);
        bindTexture(GL_TEXTURE_2D, sampTex);
        if (texLoc >= 0) uniform1i(texLoc, 0);
        bindVertexArray(texVao);
        for (int f = 0; f < NFRAMES; ++f) {
            /* 动态改写 UBO：第 f 帧 tint = (f 递增的 R, 0, 0, 1) */
            GLfloat tint[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
            tint[0] = (float)(30 + f * 20) / 255.0f;  /* 30,50,...,170 */
            bindBuffer(GL_UNIFORM_BUFFER, ubo);
            bufferData(GL_UNIFORM_BUFFER, sizeof(tint), tint, GL_DYNAMIC_DRAW);
            clearColor(0.0f, 0.0f, 0.0f, 0.0f);
            clear(GL_COLOR_BUFFER_BIT);
            drawArrays(GL_TRIANGLES, 0, 3);
            finish();  /* vkQueueSubmit + 等待，验证提交稳定性 */
            if (getError() != GL_NO_ERROR) { ++failures; break; }

            unsigned char fp[4] = {0,0,0,0};
            readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, fp);
            int wantR = 30 + f * 20;
            /* 允许 ±12 的线性采样/量化容差（float→uint8 四舍五入） */
            if (fp[0] >= wantR - 12 && fp[0] <= wantR + 12 &&
                fp[3] > 128 && fp[1] < 20 && fp[2] < 20) {
                ++stableFrames;
            } else {
                printf("frame %d: sampled tint wantR~%d got=(%d,%d,%d,%d)\n",
                       f, wantR, fp[0], fp[1], fp[2], fp[3]);
            }
        }
        CHECK(stableFrames == NFRAMES,
              "multi-frame sampled+UBO stability: %d/%d frames rendered correct color after glFinish",
              stableFrames, NFRAMES);
        CHECK(getError() == GL_NO_ERROR, "multi-frame test leaves no error");

        /* =====================================================================
         * 4k) cubemap 判别测试：6 face 各色上传 + samplerCube 采样
         * =====================================================================
         * 目标：验证 panorama（主菜单背景）cubemap 全链路的 4 个修复点——
         *   1) textureTargetFromGL 识别 6 个 face target（face 上传不被静默丢弃）；
         *   2) face 数据 z(face 索引) 正确映射到 Vulkan baseArrayLayer（layer1-5
         *      被初始化，不再全落 layer 0）；
         *   3) mipmap 覆盖全部 6 层（本用例单级 GL_LINEAR 隔离 face→layer）；
         *   4) samplerCube 按 samplerTarget=CubeMap 从 CubeMap slot 绑定纹理。
         * 判别：+X 面传红色、-X 面传绿色。采样 +X 应得红、采样 -X 应得绿。
         *   - 修复前（face 全落 layer0 / 上传被丢弃）：两方向采样同一层数据，
         *     或纹理全空 → -X 读回不是绿（黑或红）→ 本用例 FAIL。
         *   - 修复后：两方向读回各自 face 颜色 → 本用例 PASS。 */
        {
            GLuint cubeTex = 0;
            genTextures(1, &cubeTex);
            CHECK(cubeTex != 0, "genTextures allocated cubemap texture (%u)", cubeTex);
            activeTexture(GL_TEXTURE0);
            bindTexture(GL_TEXTURE_CUBE_MAP, cubeTex);
            /* 6 个 face 各 2x2 纯色：+X 红 / -X 绿 / +Y 蓝 / -Y 黄 / +Z 白 / -Z 品红 */
            static const GLenum faces[6] = {
                GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
            };
            static const GLubyte faceColors[6][4] = {
                { 255,  0,  0, 255 },  /* +X red  */
                {   0,255,  0, 255 },  /* -X green*/
                {   0,  0,255, 255 },  /* +Y blue */
                { 255,255,  0, 255 },  /* -Y yellow */
                { 255,255,255, 255 },  /* +Z white */
                { 255,  0,255, 255 },  /* -Z magenta */
            };
            for (int f = 0; f < 6; ++f) {
                const GLubyte quad[16] = { faceColors[f][0],faceColors[f][1],faceColors[f][2],faceColors[f][3],
                                           faceColors[f][0],faceColors[f][1],faceColors[f][2],faceColors[f][3],
                                           faceColors[f][0],faceColors[f][1],faceColors[f][2],faceColors[f][3],
                                           faceColors[f][0],faceColors[f][1],faceColors[f][2],faceColors[f][3] };
                texImage2D(faces[f], 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, quad);
            }
            texParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            texParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            texParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            texParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            CHECK(getError() == GL_NO_ERROR, "cubemap 6-face upload + params leaves no error");

            /* 三个 program 分别采样 +X、-X、+Z 方向（attrib0Vs 已验证能画）。 */
            static const struct { const char* tag; const char* dir; } dirs[3] = {
                { "+X", "vec3( 1.0, 0.0, 0.0)" },
                { "-X", "vec3(-1.0, 0.0, 0.0)" },
                { "+Z", "vec3( 0.0, 0.0, 1.0)" },
            };
            static const GLubyte expect[3][4] = {
                { 255,  0,  0, 255 },  /* +X red */
                {   0,255,  0, 255 },  /* -X green */
                { 255,255,255, 255 },  /* +Z white */
            };
            for (int d = 0; d < 3; ++d) {
                char fsSrc[256];
                snprintf(fsSrc, sizeof(fsSrc),
                         "#version 330 core\n"
                         "out vec4 fragColor;\n"
                         "uniform samplerCube uCube;\n"
                         "void main(){ fragColor = texture(uCube, %s); }\n",
                         dirs[d].dir);
                GLuint cFs = createShader(GL_FRAGMENT_SHADER);
                shaderSource(cFs, 1, &(const char*){ fsSrc }, NULL);
                compileShader(cFs);
                GLuint cProg = createProgram();
                attachShader(cProg, attrib0Vs);
                attachShader(cProg, cFs);
                linkProgram(cProg);
                deleteShader(cFs);
                GLint cLoc = getUniformLocation(cProg, "uCube");
                CHECK(cLoc >= 0, "cubemap(%s) getUniformLocation(uCube)=%d", dirs[d].tag, cLoc);
                useProgram(cProg);
                activeTexture(GL_TEXTURE0);
                bindTexture(GL_TEXTURE_CUBE_MAP, cubeTex);
                if (cLoc >= 0) uniform1i(cLoc, 0);
                bindVertexArray(texVao);
                clearColor(0.0f, 0.0f, 0.0f, 0.0f);
                clear(GL_COLOR_BUFFER_BIT);
                drawArrays(GL_TRIANGLES, 0, 3);
                finish();
                unsigned char cp[4] = {0,0,0,0};
                readPixels(R / 2, C / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, cp);
                int dr = expect[d][0], dg = expect[d][1], db = expect[d][2];
                CHECK(cp[0] >= dr - 20 && cp[0] <= dr + 20 &&
                      cp[1] >= dg - 20 && cp[1] <= dg + 20 &&
                      cp[2] >= db - 20 && cp[2] <= db + 20 && cp[3] > 128,
                      "cubemap(%s) face sample=%s readback=(%d,%d,%d,%d) — face->layer + samplerCube OK",
                      dirs[d].tag, dirs[d].dir, cp[0], cp[1], cp[2], cp[3]);
                deleteProgram(cProg);
            }
            CHECK(getError() == GL_NO_ERROR, "cubemap sample tests leave no error");
        }
    }

    dlclose(h);

    printf("\nRENDER SMOKE: %d/%d checks passed, %d failure(s)\n", checks - failures, checks, failures);
    if (failures == 0) {
        printf("RENDER SMOKE ALL PASSED\n");
        return 0;
    }
    printf("RENDER SMOKE FAILED\n");
    return 1;
}