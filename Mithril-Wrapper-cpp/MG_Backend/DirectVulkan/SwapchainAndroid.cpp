// Mithril-Wrapper - MG_Backend/DirectVulkan/SwapchainAndroid.cpp
// Android platform entry: creates the VkSurfaceKHR via VK_KHR_android_surface
// (vkCreateAndroidSurfaceKHR) and then delegates to
// create_swapchain_post_surface() (in SwapchainCommon.cpp) for the rest of the
// swapchain pipeline - identical to what SwapchainMetal.mm does on Apple, just
// with a different surface creation call.
//
// VK_USE_PLATFORM_ANDROID_KHR must be defined before <vulkan/vulkan.h> so
// vulkan_android.h is visible. As with the Metal TU, it is deliberately not a
// global CMake compile-definition.
#if defined(__ANDROID__)

#define VK_USE_PLATFORM_ANDROID_KHR 1

#include "Swapchain.h"
#include "Device.h"
#include "../../MG_Impl/Log.h"

#include <android/native_window.h>
#include <android/hardware_buffer.h>
#include <cstring>

#include "Resources.h"
#include "Device.h"

namespace mithril {
namespace vk {

namespace {

// True when the physical device reports `wanted`. This is the authoritative
// test for whether the driver can present: Device.cpp only enables
// VK_KHR_swapchain when it is present, and a driver built as an Android HAL
// (Turnip) does not report it.
bool device_has_extension(const char* wanted) {
    Backend* b = backend();
    if (!b->physicalDevice) return false;
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(b->physicalDevice, nullptr, &n, nullptr) != VK_SUCCESS)
        return false;
    if (n == 0) return false;
    std::vector<VkExtensionProperties> props(n);
    if (vkEnumerateDeviceExtensionProperties(b->physicalDevice, nullptr, &n, props.data()) != VK_SUCCESS)
        return false;
    for (const auto& p : props) {
        if (strncmp(p.extensionName, wanted, sizeof(p.extensionName)) == 0) return true;
    }
    return false;
}

// Blit a tightly packed RGBA8 source into whatever pixel layout the window
// buffer happens to use. ANativeWindow is free to hand back RGBX_8888 or
// RGB_565 regardless of what we requested, so the conversion is mandatory
// rather than a fast path.
void blit_to_window(const void* src, int w, int h, int srcStride,
                    ANativeWindow_Buffer* wb) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = static_cast<uint8_t*>(wb->bits);
    const int dw = wb->width > w ? w : wb->width;
    const int dh = wb->height > h ? h : wb->height;

    if (wb->format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM ||
        wb->format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM) {
        for (int y = 0; y < dh; ++y) {
            const uint8_t* sp = s + (size_t)y * srcStride;
            uint8_t* dp = d + (size_t)y * wb->stride * 4;
            memcpy(dp, sp, (size_t)dw * 4);
            if (wb->format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM) {
                for (int x = 0; x < dw; ++x) dp[x * 4 + 3] = 0xFF;
            }
        }
        return;
    }
    if (wb->format == AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM) {
        for (int y = 0; y < dh; ++y) {
            const uint8_t* sp = s + (size_t)y * srcStride;
            uint16_t* dp = reinterpret_cast<uint16_t*>(d + (size_t)y * wb->stride * 2);
            for (int x = 0; x < dw; ++x) {
                const uint8_t r = sp[x * 4], g = sp[x * 4 + 1], bl = sp[x * 4 + 2];
                dp[x] = (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (bl >> 3));
            }
        }
        return;
    }
    MITHRIL_LOG_WARN("vk", "offscreen present: unsupported window format %d", wb->format);
}

// Build the offscreen swapchain: N plain color images plus staging buffers,
// no VkSurfaceKHR and no VkSwapchainKHR anywhere.
Swapchain* create_swapchain_offscreen(ANativeWindow* win, int width, int height,
                                      int want_depth_stencil) {
    Backend* b = backend();
    if (!b->initialized) return nullptr;

    const uint32_t kImageCount = 2;
    const VkFormat colorFmt = VK_FORMAT_R8G8B8A8_UNORM;

    Swapchain* sc = new Swapchain{};
    sc->width = width;
    sc->height = height;
    sc->format = colorFmt;
    sc->offscreen = new Swapchain::Offscreen{};
    sc->offscreen->window = win;

    // Keep the window alive for as long as these images reference it. The
    // launcher can drop its own reference as soon as the surface is created.
    ANativeWindow_acquire(win);

    // Pin the window's buffer geometry to exactly what we render and what
    // blit_to_window() knows how to convert. There is no VkSwapchainKHR here
    // to negotiate a format with, so ANativeWindow is free to hand back
    // anything on lock - including formats the blit does not cover, which
    // shows up as a permanently black screen rather than an error. Declaring
    // RGBA8 up front removes that failure mode entirely and also resizes the
    // window's buffer pool to match the drawable.
    ANativeWindow_setBuffersGeometry(win, width, height,
                                     AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM);

    for (uint32_t i = 0; i < kImageCount; ++i) {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = colorFmt;
        ci.extent = {(uint32_t)width, (uint32_t)height, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage img = VK_NULL_HANDLE;
        if (vkCreateImage(b->device, &ci, nullptr, &img) != VK_SUCCESS) {
            MITHRIL_LOG_ERROR("vk", "offscreen: vkCreateImage failed");
            break;
        }
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(b->device, img, &req);
        const uint32_t mt = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mt == 0xFFFFFFFFu) {
            vkDestroyImage(b->device, img, nullptr);
            MITHRIL_LOG_ERROR("vk", "offscreen: no device-local memory type");
            break;
        }
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = mt;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (vkAllocateMemory(b->device, &ai, nullptr, &mem) != VK_SUCCESS) {
            vkDestroyImage(b->device, img, nullptr);
            MITHRIL_LOG_ERROR("vk", "offscreen: vkAllocateMemory failed");
            break;
        }
        if (vkBindImageMemory(b->device, img, mem, 0) != VK_SUCCESS) {
            vkFreeMemory(b->device, mem, nullptr);
            vkDestroyImage(b->device, img, nullptr);
            MITHRIL_LOG_ERROR("vk", "offscreen: vkBindImageMemory failed");
            break;
        }
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = colorFmt;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 1;
        VkImageView view = VK_NULL_HANDLE;
        if (vkCreateImageView(b->device, &vci, nullptr, &view) != VK_SUCCESS) {
            vkFreeMemory(b->device, mem, nullptr);
            vkDestroyImage(b->device, img, nullptr);
            MITHRIL_LOG_ERROR("vk", "offscreen: vkCreateImageView failed");
            break;
        }

        // Staging buffer for the readback. Host-visible and coherent so the
        // pixels are there the moment the fence signals, with no
        // vkInvalidateMappedMemoryRanges round trip.
        const VkDeviceSize bufSize = (VkDeviceSize)width * height * 4;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = bufSize;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory bmem = VK_NULL_HANDLE;
        void* mapped = nullptr;
        bool bufOk = false;
        if (vkCreateBuffer(b->device, &bci, nullptr, &buf) == VK_SUCCESS) {
            VkMemoryRequirements breq{};
            vkGetBufferMemoryRequirements(b->device, buf, &breq);
            const uint32_t bmt = find_memory_type(breq.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (bmt != 0xFFFFFFFFu) {
                VkMemoryAllocateInfo bai{};
                bai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                bai.allocationSize = breq.size;
                bai.memoryTypeIndex = bmt;
                if (vkAllocateMemory(b->device, &bai, nullptr, &bmem) == VK_SUCCESS) {
                    if (vkBindBufferMemory(b->device, buf, bmem, 0) == VK_SUCCESS &&
                        vkMapMemory(b->device, bmem, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
                        bufOk = true;
                    }
                }
            }
        }
        if (!bufOk) {
            if (bmem) vkFreeMemory(b->device, bmem, nullptr);
            if (buf) vkDestroyBuffer(b->device, buf, nullptr);
            vkDestroyImageView(b->device, view, nullptr);
            vkFreeMemory(b->device, mem, nullptr);
            vkDestroyImage(b->device, img, nullptr);
            MITHRIL_LOG_ERROR("vk", "offscreen: staging buffer setup failed");
            break;
        }

        sc->images.push_back(img);
        sc->offscreen->ownedImages.push_back(img);
        sc->offscreen->imageMemories.push_back(mem);
        sc->offscreen->buffers.push_back(buf);
        sc->offscreen->memories.push_back(bmem);
        sc->offscreen->mapped.push_back(mapped);
        sc->offscreen->size = bufSize;
        sc->views.push_back(view);
    }

    if (sc->images.empty()) {
        swapchain_offscreen_destroy(sc);
        destroy_swapchain(sc);
        return nullptr;
    }

    // Depth attachment, mirroring create_swapchain_post_surface().
    if (want_depth_stencil) {
        const VkFormat depthFmt = VK_FORMAT_D32_SFLOAT_S8_UINT;
        VkImageCreateInfo dici{};
        dici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        dici.imageType = VK_IMAGE_TYPE_2D;
        dici.format = depthFmt;
        dici.extent = {(uint32_t)width, (uint32_t)height, 1};
        dici.mipLevels = 1;
        dici.arrayLayers = 1;
        dici.samples = VK_SAMPLE_COUNT_1_BIT;
        dici.tiling = VK_IMAGE_TILING_OPTIMAL;
        dici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        dici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        dici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(b->device, &dici, nullptr, &sc->depthImage) == VK_SUCCESS) {
            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(b->device, sc->depthImage, &req);
            const uint32_t mt = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (mt != 0xFFFFFFFFu) {
                VkMemoryAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                ai.allocationSize = req.size;
                ai.memoryTypeIndex = mt;
                if (vkAllocateMemory(b->device, &ai, nullptr, &sc->depthMemory) == VK_SUCCESS &&
                    vkBindImageMemory(b->device, sc->depthImage, sc->depthMemory, 0) == VK_SUCCESS) {
                    VkImageViewCreateInfo dvci{};
                    dvci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                    dvci.image = sc->depthImage;
                    dvci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    dvci.format = depthFmt;
                    dvci.subresourceRange.aspectMask =
                        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
                    dvci.subresourceRange.baseMipLevel = 0;
                    dvci.subresourceRange.levelCount = 1;
                    dvci.subresourceRange.baseArrayLayer = 0;
                    dvci.subresourceRange.layerCount = 1;
                    if (vkCreateImageView(b->device, &dvci, nullptr, &sc->depthView) != VK_SUCCESS) {
                        sc->depthView = VK_NULL_HANDLE;
                    }
                }
            }
        }
    }

    sc->imageAvailablePerFrame.assign(kMaxFramesInFlight, VK_NULL_HANDLE);
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(b->device, &si, nullptr, &sc->imageAvailablePerFrame[i]) != VK_SUCCESS) {
            sc->imageAvailablePerFrame[i] = VK_NULL_HANDLE;
        }
    }
    sc->renderFinishedPerImage.assign(sc->images.size(), VK_NULL_HANDLE);
    sc->renderFinishedSignaledPerImage.assign(sc->images.size(), false);
    for (size_t i = 0; i < sc->images.size(); ++i) {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(b->device, &si, nullptr, &sc->renderFinishedPerImage[i]) != VK_SUCCESS) {
            sc->renderFinishedPerImage[i] = VK_NULL_HANDLE;
        }
    }

    MITHRIL_LOG_INFO("vk", "offscreen swapchain: %ux%u, %zu images, no Vulkan WSI",
                     (unsigned)width, (unsigned)height, sc->images.size());
    return sc;
}

} // namespace

Swapchain* create_swapchain(void* native_window, int width, int height,
                            int want_depth_stencil, int platform_hint) {
    Backend* b = backend();
    MITHRIL_LOG_WARN("vk", "create_swapchain enter %dx%d (init=%d win=%p)",
                     width, height, (int)b->initialized, native_window);
    if (!b->initialized || !native_window || width <= 0 || height <= 0) return nullptr;
    // Only one surface path is compiled into this TU, so an explicit hint for
    // another platform is simply ignored.
    (void)platform_hint;

    ANativeWindow* win = static_cast<ANativeWindow*>(native_window);
    if (!win) {
        MITHRIL_LOG_ERROR("vk", "create_swapchain: null ANativeWindow");
        return nullptr;
    }

    // Whether the driver can present at all is decided by the device's own
    // extension list, NOT by whether vkCreateAndroidSurfaceKHR resolves.
    //
    // The dispatcher keeps the platform libvulkan.so loaded as a fallback, so
    // vkGetInstanceProcAddr happily returns the *system loader's* entry point
    // even when the instance we are running on belongs to a different driver
    // (Turnip). Calling that loader entry point with a foreign VkInstance is
    // undefined behaviour and crashes outright — which is exactly what
    // happened: the process died here and the offscreen path below never ran.
    //
    // VK_KHR_swapchain is what Device.cpp enables when the device supports
    // presentation; if the device never reported it, there is no WSI to use.
    if (!device_has_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        MITHRIL_LOG_WARN("vk", "device lacks VK_KHR_swapchain - "
                               "using offscreen present path (no Vulkan WSI)");
        return create_swapchain_offscreen(win, width, height, want_depth_stencil);
    }

    auto createAndroidSurfaceKHR = reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(
        vkGetInstanceProcAddr(b->instance, "vkCreateAndroidSurfaceKHR"));
    if (!createAndroidSurfaceKHR) {
        // No WSI at all. This is the normal case for Turnip built as an
        // Android HAL module: it renders fine but cannot hand an image to the
        // presentation engine. Fall back to rendering offscreen and posting
        // the pixels ourselves, which is how Zink drives such a driver.
        MITHRIL_LOG_WARN("vk", "vkCreateAndroidSurfaceKHR not available - "
                               "using offscreen present path (no Vulkan WSI)");
        return create_swapchain_offscreen(win, width, height, want_depth_stencil);
    }

    VkAndroidSurfaceCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    sci.pNext = nullptr;
    sci.flags = 0;
    sci.window = win;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const VkResult r = createAndroidSurfaceKHR(b->instance, &sci, nullptr, &surface);
    if (r != VK_SUCCESS) {
        MITHRIL_LOG_WARN("vk", "vkCreateAndroidSurfaceKHR failed (%d) - "
                               "using offscreen present path", (int)r);
        return create_swapchain_offscreen(win, width, height, want_depth_stencil);
    }

    // Delegate the rest (format query / vkCreateSwapchainKHR / image views /
    // depth image / acquire semaphore) to the platform-independent path. On
    // failure post_surface does NOT destroy the surface - we own it here until
    // post_surface signals success by returning a non-null Swapchain.
    Swapchain* sc = create_swapchain_post_surface(surface, width, height, want_depth_stencil);
    if (!sc) {
        vkDestroySurfaceKHR(b->instance, surface, nullptr);
    }
    return sc;
}

} // namespace vk
} // namespace mithril

// ===========================================================================
// Public C API wrappers (declared in MG_Backend/Backend.h)
// ===========================================================================
extern "C" {

void* backend_create_swapchain(void* native_window, int width, int height,
                               int want_depth_stencil, int platform_hint) {
    return mithril::vk::create_swapchain(native_window, width, height,
                                         want_depth_stencil, platform_hint);
}

void backend_destroy_swapchain(void* swapchain_state) {
    mithril::vk::destroy_swapchain((mithril::vk::Swapchain*)swapchain_state);
}

VkImageView backend_swapchain_acquire_color(void* swapchain_state) {
    return mithril::vk::swapchain_acquire_color((mithril::vk::Swapchain*)swapchain_state);
}

VkImageView backend_swapchain_acquire_depth(void* swapchain_state) {
    return mithril::vk::swapchain_acquire_depth((mithril::vk::Swapchain*)swapchain_state);
}

int backend_swapchain_width(void* swapchain_state) {
    auto* sc = (mithril::vk::Swapchain*)swapchain_state;
    return sc ? sc->width : 0;
}

int backend_swapchain_height(void* swapchain_state) {
    auto* sc = (mithril::vk::Swapchain*)swapchain_state;
    return sc ? sc->height : 0;
}

void backend_present_and_acquire(void* swapchain_state) {
    mithril::vk::swapchain_present_and_acquire((mithril::vk::Swapchain*)swapchain_state);
}

int backend_swapchain_needs_rebuild(void* swapchain_state) {
    auto* sc = (mithril::vk::Swapchain*)swapchain_state;
    return sc && sc->needsRebuild ? 1 : 0;
}

void backend_swapchain_set_drawable_size(void* swapchain_state, int w, int h) {
    auto* sc = (mithril::vk::Swapchain*)swapchain_state;
    if (!sc) return;
    sc->actualDrawableWidth = w;
    sc->actualDrawableHeight = h;
}

void backend_swapchain_mark_rebuild(void* swapchain_state) {
    auto* sc = (mithril::vk::Swapchain*)swapchain_state;
    if (!sc) return;
    sc->needsRebuild = true;
}

VkImage backend_swapchain_current_color_image(void* swapchain_state) {
    auto* sc = (mithril::vk::Swapchain*)swapchain_state;
    if (!sc || sc->currentImage < 0 || sc->currentImage >= (int)sc->images.size())
        return VK_NULL_HANDLE;
    return sc->images[sc->currentImage];
}

VkFormat backend_swapchain_color_format(void* swapchain_state) {
    auto* sc = (mithril::vk::Swapchain*)swapchain_state;
    return sc ? sc->format : VK_FORMAT_UNDEFINED;
}

VkImage backend_swapchain_current_depth_image(void* swapchain_state) {
    auto* sc = (mithril::vk::Swapchain*)swapchain_state;
    return sc ? sc->depthImage : VK_NULL_HANDLE;
}

VkFormat backend_swapchain_depth_format(void* swapchain_state) {
    auto* sc = (mithril::vk::Swapchain*)swapchain_state;
    // The depth image is always created as VK_FORMAT_D32_SFLOAT_S8_UINT in
    // create_swapchain_post_surface(); there is no per-swapchain field
    // tracking it.
    (void)sc;
    return VK_FORMAT_D32_SFLOAT_S8_UINT;
}

} // extern "C"

namespace mithril {
namespace vk {

void swapchain_offscreen_present(mithril::vk::Swapchain* sc) {
    if (!sc || !sc->offscreen || !sc->offscreen->window) return;
    const int idx = sc->currentImage;
    if (idx < 0 || idx >= (int)sc->images.size()) return;

    Backend* b = backend();
    auto* o = sc->offscreen;

    // Copy the rendered image into its staging buffer. Done on a throwaway
    // command buffer because commit_frame() has already submitted this frame.
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = b->commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(b->device, &ai, &cmd) != VK_SUCCESS) return;

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(b->device, &fi, nullptr, &fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(b->device, b->commandPool, 1, &cmd);
        return;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    bool ok = vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS;
    if (ok) {
        VkImageMemoryBarrier toSrc{};
        toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image = sc->images[idx];
        toSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toSrc.subresourceRange.baseMipLevel = 0;
        toSrc.subresourceRange.levelCount = 1;
        toSrc.subresourceRange.baseArrayLayer = 0;
        toSrc.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toSrc);

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {(uint32_t)sc->width, (uint32_t)sc->height, 1};
        vkCmdCopyImageToBuffer(cmd, sc->images[idx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               o->buffers[idx], 1, &region);

        VkImageMemoryBarrier back{};
        back.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        back.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        back.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        back.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        back.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        back.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        back.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        back.image = sc->images[idx];
        back.subresourceRange = toSrc.subresourceRange;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &back);
        if (vkEndCommandBuffer(cmd) == VK_SUCCESS) {
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            if (vkQueueSubmit(b->graphicsQueue, 1, &si, fence) == VK_SUCCESS) {
                vkWaitForFences(b->device, 1, &fence, VK_TRUE, UINT64_MAX);
            }
        }
    }
    vkDestroyFence(b->device, fence, nullptr);
    vkFreeCommandBuffers(b->device, b->commandPool, 1, &cmd);

    // Hand the pixels to the window.
    ANativeWindow* win = static_cast<ANativeWindow*>(o->window);
    ANativeWindow_Buffer wb{};
    if (ANativeWindow_lock(win, &wb, nullptr) == 0) {
        blit_to_window(o->mapped[idx], sc->width, sc->height,
                       (int)(o->size / sc->height), &wb);
        ANativeWindow_unlockAndPost(win);
    } else {
        static int lockFail = 0;
        if (++lockFail <= 3 || lockFail % 100 == 0) {
            MITHRIL_LOG_ERROR("vk", "offscreen present: ANativeWindow_lock failed (%d)", lockFail);
        }
    }
}

void swapchain_offscreen_destroy(mithril::vk::Swapchain* sc) {
    if (!sc || !sc->offscreen) return;
    Backend* b = backend();
    auto* o = sc->offscreen;
    if (b->device) {
        for (size_t i = 0; i < o->buffers.size(); ++i) {
            if (o->mapped[i]) vkUnmapMemory(b->device, o->memories[i]);
            if (o->memories[i]) vkFreeMemory(b->device, o->memories[i], nullptr);
            if (o->buffers[i]) vkDestroyBuffer(b->device, o->buffers[i], nullptr);
        }
        // Unlike swapchain images, which belong to the VkSwapchainKHR, these
        // were allocated here and must be released explicitly.
        for (size_t i = 0; i < o->ownedImages.size(); ++i) {
            if (o->ownedImages[i]) vkDestroyImage(b->device, o->ownedImages[i], nullptr);
            if (i < o->imageMemories.size() && o->imageMemories[i])
                vkFreeMemory(b->device, o->imageMemories[i], nullptr);
        }
    }
    if (o->window) {
        ANativeWindow_release(static_cast<ANativeWindow*>(o->window));
        o->window = nullptr;
    }
    delete o;
    sc->offscreen = nullptr;
}

} // namespace vk
} // namespace mithril

#endif // __ANDROID__
