cmake_minimum_required(VERSION 3.16)

project(Kernelix
    VERSION 0.1.0
    DESCRIPTION "Header-only C++ library for automatic tensor kernel optimization"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(KERNELIX_BUILD_TESTS "Build unit tests" ON)
option(KERNELIX_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(KERNELIX_BUILD_EXAMPLES "Build examples" ON)
option(KERNELIX_ENABLE_AVX2 "Enable AVX2 intrinsics" ON)
option(KERNELIX_ENABLE_AVX512 "Enable AVX-512 intrinsics" OFF)

# Auto-detect ARM NEON
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    option(KERNELIX_ENABLE_NEON "Enable ARM NEON intrinsics" ON)
else()
    option(KERNELIX_ENABLE_NEON "Enable ARM NEON intrinsics" OFF)
endif()

# -----------------------------------------------------------------------------
# C++ Standard
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Find Dependencies
# -----------------------------------------------------------------------------

# TBB (optional - fallback to sequential if not found)
find_package(TBB QUIET)
if(TBB_FOUND)
    set(KERNELIX_HAS_TBB ON)
    message(STATUS "TBB found - parallel execution enabled")
else()
    set(KERNELIX_HAS_TBB OFF)
    message(STATUS "TBB not found - using sequential fallback")
endif()

# -----------------------------------------------------------------------------
# Header-only library target
# -----------------------------------------------------------------------------
add_library(kernelix INTERFACE)
add_library(Kernelix::kernelix ALIAS kernelix)

target_include_directories(kernelix INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(KERNELIX_HAS_TBB)
    target_link_libraries(kernelix INTERFACE TBB::tbb)
    target_compile_definitions(kernelix INTERFACE KERNELIX_HAS_TBB=1)
endif()

target_compile_features(kernelix INTERFACE cxx_std_20)

# -----------------------------------------------------------------------------
# Compiler-specific flags
# -----------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Common flags for GCC and Clang
    target_compile_options(kernelix INTERFACE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
        $<$<CONFIG:Debug>:-O0 -g>
    )

    # SIMD flags
    if(KERNELIX_ENABLE_AVX2 AND NOT KERNELIX_ENABLE_NEON)
        target_compile_options(kernelix INTERFACE -mavx2 -mfma)
        target_compile_definitions(kernelix INTERFACE KERNELIX_AVX2=1)
    endif()

    if(KERNELIX_ENABLE_AVX512 AND NOT KERNELIX_ENABLE_NEON)
        target_compile_options(kernelix INTERFACE -mavx512f -mavx512vl -mavx512bw -mavx512dq)
        target_compile_definitions(kernelix INTERFACE KERNELIX_AVX512=1)
    endif()

    if(KERNELIX_ENABLE_NEON)
        target_compile_definitions(kernelix INTERFACE KERNELIX_NEON=1)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
            # ARM64 has NEON by default
        else()
            target_compile_options(kernelix INTERFACE -mfpu=neon)
        endif()
    endif()

elseif(MSVC)
    target_compile_options(kernelix INTERFACE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi>
        $<$<CONFIG:Debug>:/Od /Zi>
        /W4
    )

    if(KERNELIX_ENABLE_AVX2)
        target_compile_options(kernelix INTERFACE /arch:AVX2)
        target_compile_definitions(kernelix INTERFACE KERNELIX_AVX2=1)
    endif()

    if(KERNELIX_ENABLE_AVX512)
        target_compile_options(kernelix INTERFACE /arch:AVX512)
        target_compile_definitions(kernelix INTERFACE KERNELIX_AVX512=1)
    endif()
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(KERNELIX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
if(KERNELIX_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------
if(KERNELIX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install target
install(TARGETS kernelix
    EXPORT KernelixTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT KernelixTargets
    FILE KernelixTargets.cmake
    NAMESPACE Kernelix::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kernelix
)

# Create config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/KernelixConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/KernelixConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kernelix
)

# Create version file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/KernelixConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/KernelixConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/KernelixConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kernelix
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Kernelix ${PROJECT_VERSION} Configuration Summary")
message(STATUS "======================================")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  Build tests:     ${KERNELIX_BUILD_TESTS}")
message(STATUS "  Build benchmarks:${KERNELIX_BUILD_BENCHMARKS}")
message(STATUS "  Build examples:  ${KERNELIX_BUILD_EXAMPLES}")
message(STATUS "")
message(STATUS "SIMD:")
message(STATUS "  AVX2:            ${KERNELIX_ENABLE_AVX2}")
message(STATUS "  AVX-512:         ${KERNELIX_ENABLE_AVX512}")
message(STATUS "  NEON:            ${KERNELIX_ENABLE_NEON}")
message(STATUS "")
message(STATUS "Parallelism:")
if(KERNELIX_HAS_TBB)
    message(STATUS "  TBB:             Found")
else()
    message(STATUS "  TBB:             Not found (sequential fallback)")
endif()
message(STATUS "")
