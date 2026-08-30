cmake_minimum_required(VERSION 3.22.1)
project("mithril" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build identity: stamp the git commit id so the dylib prints it on every game
# startup (crash-log triage). Source it directly from GITHUB_SHA — the commit
# id GitHub Actions injects as an environment variable on every run. Local
# (non-CI) builds fall back to "unknown".
if(DEFINED ENV{GITHUB_SHA} AND NOT "$ENV{GITHUB_SHA}" STREQUAL "")
    set(MITHRIL_COMMIT_ID "$ENV{GITHUB_SHA}")
else()
    set(MITHRIL_COMMIT_ID "unknown")
endif()
message(STATUS "Mithril build commit id: ${MITHRIL_COMMIT_ID}")

# Objective-C++ (.mm) standard for the EGL surface layer + Apple GPU info +
# Vulkan Metal-backed swapchain. The .mm files in the cross-platform split are
# egl/SurfaceMetal.mm (CAMetalLayer coercion), MG_Impl/Getter_gpu.mm (NSString
# / NSLog helpers), and MG_Backend/DirectVulkan/SwapchainMetal.mm (defines
# VK_USE_PLATFORM_METAL_EXT so it can call vkCreateMetalSurfaceEXT). All of
# them are Foundation/QuartzCore/Metal — no other platforms build .mm files.
if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

# GL/EGL symbols must be exported (the dylib/so is a drop-in OpenGL/EGL
# replacement). IMPORTANT: .mm files (EGL surface, Getter_gpu, SwapchainMetal)
# are compiled as OBJCXX; leetal/ios-cmake defaults OBJCXX to
# -fvisibility=hidden, which would hide the egl* entry points and make host
# dlsym("eglCreateContext") fail. Mirror the CXX visibility on OBJCXX so
# egl.cpp's extern "C" functions are exported. (The egl.cpp translation unit
# also guards itself with #pragma GCC visibility push(default); both layers
# are intentional belt-and-suspenders.)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_C_VISIBILITY_PRESET default)
set(CMAKE_OBJCXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
if(APPLE)
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fvisibility=default")
endif()

# ---------------------------------------------------------------------------
# MoltenVK: Vulkan 1.2 implementation backed by Metal 2.
# Linked dynamically via libMoltenVK.dylib (v1.2.9, from Amethyst).
# Lives at Mithril-Wrapper-cpp/3rdparty/MoltenVK/libMoltenVK.dylib.
# Post-build patches the install_name reference to match Amethyst's layout.
# Vulkan headers come from MoltenVK-Headers (extracted from the CI tarball
# or from the MoltenVK.xcframework/Headers fallback).
# ---------------------------------------------------------------------------
if(APPLE)
    # NOTE: NO_CMAKE_FIND_ROOT_PATH is required here. leetal/ios-cmake sets
    # CMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY, which re-roots every search path
    # under the iOS sysroot. MoltenVK is staged at the REPO ROOT by the CI
    # workflow (MoltenVK-Headers/, MoltenVK.xcframework/...), NOT inside the
    # sysroot, so without this flag find_path would look for
    # <sysroot>/.../MoltenVK-Headers and fail with REQUIRED. This mirrors the
    # NO_CMAKE_FIND_ROOT_PATH already present on the find_library call below.
    find_path(MOLTENVK_INCLUDE_DIR
        NAMES vulkan/vulkan.h
        PATHS
            "${MOLTENVK_ROOT}/MoltenVK-Headers"
            "${CMAKE_SOURCE_DIR}/MoltenVK-Headers"
            "${CMAKE_SOURCE_DIR}/3rdparty/MoltenVK/MoltenVK-Headers"
            "${MOLTENVK_ROOT}/MoltenVK.xcframework/Headers"
            "${CMAKE_SOURCE_DIR}/MoltenVK.xcframework/Headers"
            "${CMAKE_SOURCE_DIR}/3rdparty/MoltenVK/MoltenVK.xcframework/Headers"
            "${CMAKE_SOURCE_DIR}/Mithril-Wrapper-cpp/3rdparty/MoltenVK/MoltenVK.xcframework/Headers"
            "${CMAKE_SOURCE_DIR}/MoltenVK.xcframework/ios-arm64/Headers"
            "${CMAKE_SOURCE_DIR}/MoltenVK.xcframework/ios-arm64_x86_64-simulator/Headers"
        NO_CMAKE_FIND_ROOT_PATH
        REQUIRED
    )
    message(STATUS "MoltenVK headers: ${MOLTENVK_INCLUDE_DIR}")

    # Link against the MoltenVK static library. The CI workflow downloads the
    # MoltenVK iOS tarball and stages it at the REPO ROOT as
    #   MoltenVK.xcframework/ios-arm64/libMoltenVK.a   (static lib)
    #   MoltenVK-Headers/vulkan/vulkan.h               (headers)
    # (see .github/workflows/build.yml "Download MoltenVK" step). The legacy
    # Mithril-Wrapper-cpp/3rdparty/MoltenVK location is kept only as a fallback
    # for local checkouts that vendor the dylib there. NAMES covers both
    # libMoltenVK.a (CI) and a bare MoltenVK.framework (local).
    find_library(MOLTENVK_LIBRARY
        NAMES MoltenVK libMoltenVK
        PATHS
            "${CMAKE_SOURCE_DIR}/MoltenVK.xcframework/ios-arm64"
            "${CMAKE_SOURCE_DIR}/MoltenVK.xcframework"
            "${CMAKE_SOURCE_DIR}/Mithril-Wrapper-cpp/3rdparty/MoltenVK"
        NO_CMAKE_FIND_ROOT_PATH
        REQUIRED
    )
    message(STATUS "MoltenVK library: ${MOLTENVK_LIBRARY}")
endif()

# ---------------------------------------------------------------------------
# glslang: GLSL front-end -> SPIR-V  (submodule under 3rdparty/, mirrors the
# MobileGlues layout which uses add_subdirectory(3rdparty/glslang) instead of
# FetchContent so the sources are available for local hacking).
#
# SPIRV-Cross is used at runtime for SPIR-V *reflection* (not translation):
# DescriptorSet.cpp reflects each linked program's VS+FS SPIR-V to discover
# UBO / sampled-image bindings, then builds VkDescriptorSetLayout /
# VkPipelineLayout / VkDescriptorPool. MoltenVK still cross-translates the
# Vulkan SPIR-V to MSL internally at vkCreateShaderModule time, so SPIRV-Cross
# is NOT used for shader translation — only for the binding reflection needed
# to wire uniforms + textures into the shader.
# ---------------------------------------------------------------------------
set(ENABLE_RTTI                ON  CACHE BOOL "" FORCE)
set(ENABLE_HLSL                OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT                 OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES    OFF CACHE BOOL "" FORCE)
set(BUILD_EXTERNAL             OFF CACHE BOOL "" FORCE)
set(SKIP_GLSLANG_INSTALL       ON  CACHE BOOL "" FORCE)
set(GLSLANG_TESTS              OFF CACHE BOOL "" FORCE)
add_subdirectory(Mithril-Wrapper-cpp/3rdparty/glslang)

# SPIRV-Cross: used for SPIR-V reflection (spirv_cross::Compiler +
# get_shader_resources) in MG_Backend/DirectVulkan/DescriptorSet.cpp. Only the
# core library is needed (no GLSL/HLSL/MSL/CPP backends, no CLI, no tests).
# SPIRV-Headers (sibling submodule) provides <spirv.h> for the spv:: enums.
set(SPIRV_CROSS_STATIC                 ON  CACHE BOOL "" FORCE)
set(SPIRV_CROSS_CLI                    OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS           OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_GLSL            OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_HLSL            OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_MSL             OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_CPP             OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_REFLECT         OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_C_API           OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SKIP_INSTALL           ON  CACHE BOOL "" FORCE)
set(SPIRV_HEADERS_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/Mithril-Wrapper-cpp/3rdparty/SPIRV-Headers/include"
    CACHE PATH "" FORCE)
add_subdirectory(Mithril-Wrapper-cpp/3rdparty/SPIRV-Cross)

# ---------------------------------------------------------------------------
# Mithril-Wrapper shared library (libmithril.dylib on Apple)
#
# Source layout (cross-platform Vulkan/MoltenVK rewrite):
#   Mithril-Wrapper-cpp/
#     MG_Impl/         — OpenGL 3.3 Core Profile entry points (GL calls into
#                        the Vulkan backend via the C API in MG_Backend/Backend.h)
#     MG_State/        — GL state machine (per-EGLContext GLState)
#     MG_Backend/      — abstract backend C API (Backend.h)
#       DirectVulkan/  — Vulkan 1.2 + MoltenVK implementation (split into
#                        platform-specific surface files + a shared common core)
#     egl/             — EGL 1.5 (cross-platform core egl.cpp + platform-specific
#                        Surface<Platform>.cpp/mm; the platform file is picked
#                        by the APPLE guard below)
#     include/         — public GL/EGL headers
# ---------------------------------------------------------------------------
set(MITHRIL_SOURCES
    # MG_Impl: OpenGL 3.3 Core Profile entry points
    Mithril-Wrapper-cpp/MG_Impl/init.cpp
    Mithril-Wrapper-cpp/MG_Impl/Log.cpp
    Mithril-Wrapper-cpp/MG_Impl/gl.cpp
    Mithril-Wrapper-cpp/MG_Impl/Getter.cpp
    Mithril-Wrapper-cpp/MG_Impl/Program.cpp
    Mithril-Wrapper-cpp/MG_Impl/Shader.cpp
    Mithril-Wrapper-cpp/MG_Impl/Buffer.cpp
    Mithril-Wrapper-cpp/MG_Impl/VertexArray.cpp
    Mithril-Wrapper-cpp/MG_Impl/Texture.cpp
    Mithril-Wrapper-cpp/MG_Impl/Framebuffer.cpp
    Mithril-Wrapper-cpp/MG_Impl/Drawing.cpp
    Mithril-Wrapper-cpp/MG_Impl/Stubs.cpp
    Mithril-Wrapper-cpp/MG_Impl/GL46_Compat.cpp
    Mithril-Wrapper-cpp/MG_Impl/Debug.cpp
    Mithril-Wrapper-cpp/MG_Impl/lookup.cpp

    # MITHRIL_SOURCES historically but the source files were never committed to
    # the repo (and no other translation unit references their symbols). They are
    # out of scope for the GL 3.3 Core Profile target and caused
    # "Cannot find source file" at configure time. Removed from the build until
    # the corresponding GL 4.x features are actually implemented.

    # MG_State: GL state machine
    Mithril-Wrapper-cpp/MG_State/State.cpp
    Mithril-Wrapper-cpp/MG_State/Caps.cpp

    # MG_Backend: Vulkan 1.2 + MoltenVK direct backend (platform-agnostic core)
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/Device.cpp
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/Resources.cpp
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/FormatMap.cpp
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/Reflect.cpp
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/Std140.cpp
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/UniformArena.cpp
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/Pipeline.cpp
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/DescriptorSet.cpp
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/CommandStream.cpp
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/ImageOps.cpp
    # SwapchainCommon.cpp: shared post-surface swapchain creation logic
    # (format query / vkCreateSwapchainKHR / image views / depth-stencil /
    # destroy / acquire / present). Platform-specific files
# (SwapchainMetal.mm) call create_swapchain_post_surface
    # after creating the VkSurfaceKHR.
    Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/SwapchainCommon.cpp

    # MG_Impl: pure-logic EGL config table extracted from the EGL layer so it
    # can be unit-tested without an ObjC++ toolchain. Compiled into every
    # platform's dylib/so (egl.cpp #includes EGLConfig.h and treats g_configs
    # as its config table).
    Mithril-Wrapper-cpp/MG_Impl/EGLConfig.cpp

    # egl/egl.cpp: cross-platform EGL 1.5 core (all egl* entry points +
    # shadow sync/image objects). No platform-specific includes — surface
    # creation dispatches to surface_create() / surface_get_size() provided
    # by the platform file (SurfaceMetal.mm) selected by
    # the APPLE guard below.
    Mithril-Wrapper-cpp/egl/egl.cpp

    # egl/SwapchainHelper.cpp: Vulkan swapchain lifecycle (ensure_swapchain /
    # install_surface_on_state) extracted out of the former monolithic egl.cpp
    # so the EGL entry-point layer no longer carries swapchain/Vulkan details.
    # Plain C++ (the swapchain is owned by the backend), built on every platform.
    # Shared internal types live in egl/EglInternal.h (header only).
    Mithril-Wrapper-cpp/egl/SwapchainHelper.cpp
)

# ---- Platform-specific sources ----
# Each platform contributes:
#   - One egl/Surface<Platform>.{cpp,mm} providing surface_create/surface_get_size
#   - One MG_Backend/DirectVulkan/Swapchain<Platform>.{cpp,mm} providing
#     create_swapchain + the backend_* C API wrappers (the wrappers are
#     identical across platforms; CMake picks one TU per build).
if(APPLE)
    list(APPEND MITHRIL_SOURCES
        # EGL surface creation: CAMetalLayer coercion + drawable size query.
        # Compiled as Objective-C++ so it can talk to CAMetalLayer.
        Mithril-Wrapper-cpp/egl/SurfaceMetal.mm
        # GPU info string builder (queries VkPhysicalDeviceProperties for
        # GL_RENDERER; .mm so it can use NSString/NSLog without an #ifdef dance).
        Mithril-Wrapper-cpp/MG_Impl/Getter_gpu.mm
        # Vulkan swapchain: VkSurfaceKHR via VK_EXT_metal_surface. Compiled as
        # Objective-C++ because it #defines VK_USE_PLATFORM_METAL_EXT before
        # including <vulkan/vulkan.h>, which pulls in <Metal/Metal.h>
        # (ObjC-only) for the VkMetalSurfaceCreateInfoEXT /
        # PFN_vkCreateMetalSurfaceEXT declarations. The other DirectVulkan
        # .cpp files stay plain C++ — Device.h stores the function pointer as
        # PFN_vkVoidFunction to avoid needing the metal platform define there.
        Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/SwapchainMetal.mm
    )
endif()

add_library(${PROJECT_NAME} SHARED ${MITHRIL_SOURCES})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Mithril-Wrapper-cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Mithril-Wrapper-cpp/include
)

# glslang (GLSL -> SPIR-V). NOTE: glslang-default-resource-limits provides
# GetDefaultResources() which lives in ResourceLimits/ResourceLimits.cpp.
# Without it the link fails with "Undefined symbols: GetDefaultResources()"
# when ENABLE_GLSLANG_BINARIES=OFF (the standalone glslangValidator that used
# to host it is disabled).
#
# spirv-cross-core: SPIR-V reflection API (spirv_cross::Compiler,
# get_shader_resources) used by DescriptorSet.cpp to build descriptor set
# layouts. The SPIRV-Headers include dir supplies <spirv.h> (spv::Decoration*).
target_link_libraries(${PROJECT_NAME} PRIVATE
    glslang::glslang
    glslang::SPIRV
    glslang::glslang-default-resource-limits
    spirv-cross-core
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Mithril-Wrapper-cpp/3rdparty/SPIRV-Cross
    ${CMAKE_CURRENT_SOURCE_DIR}/Mithril-Wrapper-cpp/3rdparty/SPIRV-Headers/include
)

if(APPLE)
    # MoltenVK static framework + Vulkan headers. The framework transitively
    # links Metal.framework / QuartzCore.framework / Foundation.framework, but
    # we still list them explicitly to satisfy -framework lookup when the
    # static link strips MoltenVK's lazy references. UIKit is required because
    # libMoltenVK.a references UIScreen / UIView (iOS surface/scale handling).
    #
    # -force_load: forces the linker to load ALL object files from the MoltenVK
    # static archive, including C++ static initializers that register Vulkan
    # instance extensions (VK_EXT_metal_surface, VK_KHR_surface, etc.) at
    # library load time. Without this flag, the linker may strip object files
    # whose symbols are not directly referenced — even with -ObjC, which only
    # preserves Objective-C classes/categories but NOT pure C++ global
    # constructors. When those initializers are missing,
    # vkEnumerateInstanceExtensionProperties returns 0 extensions,
    # VK_EXT_metal_surface is never enabled at instance creation,
    # vkCreateMetalSurfaceEXT resolves to null, and swapchain creation fails.
    #
    # -ObjC: forces the linker to load all Objective-C class/category symbols
    # from static archives. MoltenVK's libMoltenVK.a contains ObjC++ code
    # (Metal interop); without this flag, categories may be stripped and cause
    # runtime "unrecognized selector" crashes. Standard Apple recommendation
    # for static libraries containing Objective-C. This flag COMBINED with
    # -force_load above ensures both ObjC and C++ initializers are loaded.
    target_include_directories(${PROJECT_NAME} PRIVATE ${MOLTENVK_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "${MOLTENVK_LIBRARY}"
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
        "-framework CoreGraphics"
        "-framework IOSurface"
    )
    # UIKit is an iOS-only framework (absent on macOS). libMoltenVK.a references
    # UIScreen / UIView for iOS surface/scale handling, so it is required on iOS
    # but must NOT be linked on a native macOS build (test-macos-smoke) or the
    # linker fails with "framework 'UIKit' not found". The iOS toolchain
    # (ios.toolchain.cmake) defines the CMake IOS variable; a native macOS build
    # does not.
    if(IOS)
        target_link_libraries(${PROJECT_NAME} PRIVATE "-framework UIKit")
    endif()
    # The bundled libMoltenVK.dylib has install name
    # "@rpath/MoltenVK.framework/MoltenVK", but it's a raw dylib, not a
    # framework bundle. Patch the load reference so libmithril.dylib looks
    # for "@rpath/libMoltenVK.dylib" instead — which is where Amethyst
    # places it at runtime.
    #
    # Only relevant on iOS: the iOS MoltenVK ships as a .framework bundle with
    # that install name. A native macOS build (test-macos-smoke) links the
    # Homebrew libMoltenVK.dylib, whose load reference is already the real
    # dylib path — patching it to @rpath/libMoltenVK.dylib would break dyld
    # resolution. install_name_tool -change would silently no-op (old string
    # absent) but we keep the guard explicit so macOS keeps the true path.
    if(IOS)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND install_name_tool -change
                "@rpath/MoltenVK.framework/MoltenVK"
                "@rpath/libMoltenVK.dylib"
                "$<TARGET_FILE:${PROJECT_NAME}>"
            COMMENT "Patching MoltenVK load path: @rpath/MoltenVK.framework/MoltenVK → @rpath/libMoltenVK.dylib"
        )
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE MITHRIL_VULKAN=1)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})
endif()

# VK_ENABLE_BETA_EXTENSIONS: vulkan_core.h only includes vulkan_beta.h (which
# defines VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME / VK_KHR_portability_subset
# structs) when this macro is defined. MoltenVK on Apple AND conformant Linux
# drivers (Mesa / NVIDIA with portability subset) advertise
# VK_KHR_portability_subset; Device.cpp enables it by name on every platform,
# so the macro MUST be visible across the whole build. Safe to define globally
# — it only gates beta-extension declarations, no codegen effect.
target_compile_definitions(${PROJECT_NAME} PRIVATE VK_ENABLE_BETA_EXTENSIONS=1)

# Inject the build's git commit id (captured from GITHUB_SHA at the top of this
# file) so proc_init() can print it on every startup for crash-log triage. The
# value is a quoted string so it concatenates into a printf format literal.
target_compile_definitions(${PROJECT_NAME} PRIVATE
    MITHRIL_COMMIT_ID="${MITHRIL_COMMIT_ID}"
)

# ARC for Objective-C++ translation units (EGL surface, GPU info, Vulkan swapchain).
if(APPLE)
    set_source_files_properties(
        Mithril-Wrapper-cpp/egl/SurfaceMetal.mm
        Mithril-Wrapper-cpp/MG_Impl/Getter_gpu.mm
        Mithril-Wrapper-cpp/MG_Backend/DirectVulkan/SwapchainMetal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
endif()

# Reasonable warnings, no -Werror (CI must be tolerant during bring-up)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "mithril"
    PREFIX "lib"
)
# Apple produces libmithril.dylib.
# CMake's default SHARED suffix already differs by platform (.dylib on Apple,
# .so elsewhere), but we set it explicitly so the value is
# self-documenting and survives any global SUFFIX overrides from toolchain
# files (leetal/ios-cmake in particular forces .dylib unconditionally).
if(APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".dylib")
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")
endif()

