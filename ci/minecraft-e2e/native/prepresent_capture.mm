#include <GL/gl.h>

#include <dlfcn.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
std::string env_string(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

void append_event(const std::string& root, const char* event, int frame, const char* detail) {
    if (root.empty()) return;
    FILE* f = std::fopen((root + "/native-events.jsonl").c_str(), "a");
    if (!f) return;
    std::fprintf(f,
        "{\"schema_version\":\"1.0\",\"producer\":\"prepresent-capture\","
        "\"event\":\"%s\",\"frame_id\":%d,\"message\":\"%s\"}\n",
        event, frame, detail ? detail : "");
    std::fclose(f);
}

template <typename T>
T sym(void* handle, const char* name) {
    return reinterpret_cast<T>(handle ? dlsym(handle, name) : nullptr);
}

bool write_atomic(const std::string& path, const void* data, size_t size) {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(data, 1, size, f) == size;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) {
        std::remove(tmp.c_str());
        return false;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}
} // namespace

extern "C" void mithril_e2e_capture_before_present(int width, int height, void* mithril_handle) {
    const std::string root = env_string("MITHRIL_E2E_ROOT");
    if (root.empty() || width <= 1 || height <= 1 || !mithril_handle) return;

    const std::string request = root + "/render/prepresent-request.txt";
    FILE* request_file = std::fopen(request.c_str(), "r");
    if (!request_file) return;

    int frame = 0;
    if (std::fscanf(request_file, "%d", &frame) != 1 || frame <= 0) {
        std::fclose(request_file);
        std::remove(request.c_str());
        append_event(root, "prepresent_capture_bad_request", frame, "invalid capture request");
        return;
    }
    std::fclose(request_file);
    // Claim the request before touching GL so one request maps to exactly one swap.
    std::remove(request.c_str());

    using GetIntegerv = void (*)(GLenum, GLint*);
    using BindFramebuffer = void (*)(GLenum, GLuint);
    using BindBuffer = void (*)(GLenum, GLuint);
    using PixelStorei = void (*)(GLenum, GLint);
    using ReadPixels = void (*)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*);
    using GetError = GLenum (*)(void);

    GetIntegerv getIntegerv = sym<GetIntegerv>(mithril_handle, "glGetIntegerv");
    BindFramebuffer bindFramebuffer = sym<BindFramebuffer>(mithril_handle, "glBindFramebuffer");
    BindBuffer bindBuffer = sym<BindBuffer>(mithril_handle, "glBindBuffer");
    PixelStorei pixelStorei = sym<PixelStorei>(mithril_handle, "glPixelStorei");
    ReadPixels readPixels = sym<ReadPixels>(mithril_handle, "glReadPixels");
    GetError getError = sym<GetError>(mithril_handle, "glGetError");
    if (!getIntegerv || !bindFramebuffer || !bindBuffer || !pixelStorei || !readPixels || !getError) {
        append_event(root, "prepresent_capture_failed", frame, "required Mithril GL symbol missing");
        return;
    }

    GLint read_fbo = 0, pack_pbo = 0;
    GLint pack_alignment = 4, pack_row_length = 0, pack_skip_rows = 0, pack_skip_pixels = 0;
    GLint pack_image_height = 0, pack_skip_images = 0;
    getIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo);
    getIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &pack_pbo);
    getIntegerv(GL_PACK_ALIGNMENT, &pack_alignment);
    getIntegerv(GL_PACK_ROW_LENGTH, &pack_row_length);
    getIntegerv(GL_PACK_SKIP_ROWS, &pack_skip_rows);
    getIntegerv(GL_PACK_SKIP_PIXELS, &pack_skip_pixels);
    getIntegerv(GL_PACK_IMAGE_HEIGHT, &pack_image_height);
    getIntegerv(GL_PACK_SKIP_IMAGES, &pack_skip_images);

    bindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    bindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    pixelStorei(GL_PACK_ALIGNMENT, 1);
    pixelStorei(GL_PACK_ROW_LENGTH, 0);
    pixelStorei(GL_PACK_SKIP_ROWS, 0);
    pixelStorei(GL_PACK_SKIP_PIXELS, 0);
    pixelStorei(GL_PACK_IMAGE_HEIGHT, 0);
    pixelStorei(GL_PACK_SKIP_IMAGES, 0);

    std::vector<unsigned char> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
    // glReadPixels is synchronous by GL contract. In Mithril this path ends the
    // active render pass, submits the DirectMetal command buffer, blits the
    // current default-color texture into CPU-visible storage, and waits for the
    // copy. Crucially this runs before eglSwapBuffers presents/acquires the next
    // drawable, so it observes the frame that Minecraft just rendered.
    readPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    GLenum error = getError();

    bindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(read_fbo));
    bindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(pack_pbo));
    pixelStorei(GL_PACK_ALIGNMENT, pack_alignment);
    pixelStorei(GL_PACK_ROW_LENGTH, pack_row_length);
    pixelStorei(GL_PACK_SKIP_ROWS, pack_skip_rows);
    pixelStorei(GL_PACK_SKIP_PIXELS, pack_skip_pixels);
    pixelStorei(GL_PACK_IMAGE_HEIGHT, pack_image_height);
    pixelStorei(GL_PACK_SKIP_IMAGES, pack_skip_images);

    if (error != GL_NO_ERROR) {
        char detail[96];
        std::snprintf(detail, sizeof(detail), "glReadPixels error 0x%04x", error);
        append_event(root, "prepresent_capture_failed", frame, detail);
        return;
    }

    // Do not put the absolute path in a fixed-size C buffer: GitHub hosted
    // workspaces are long enough that a 96-byte stem silently truncates
    // "prepresent-frame-0001" into "prepres", making the producer and Java
    // consumer disagree even though readback itself succeeded.
    char frame_name[40];
    std::snprintf(frame_name, sizeof(frame_name), "prepresent-frame-%04d", frame);
    const std::string stem = root + "/render/" + frame_name;
    const std::string raw_path = stem + ".rgba";
    const std::string meta_path = stem + ".meta";
    if (!write_atomic(raw_path, rgba.data(), rgba.size())) {
        append_event(root, "prepresent_capture_failed", frame, "could not persist RGBA readback");
        return;
    }

    char meta[128];
    int n = std::snprintf(meta, sizeof(meta), "%d %d %zu\n", width, height, rgba.size());
    if (n <= 0 || !write_atomic(meta_path, meta, static_cast<size_t>(n))) {
        append_event(root, "prepresent_capture_failed", frame, "could not persist capture metadata");
        return;
    }
    append_event(root, "prepresent_capture_completed", frame, "captured default framebuffer before eglSwapBuffers");
}
