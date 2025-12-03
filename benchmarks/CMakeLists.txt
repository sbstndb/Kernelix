# Kernelix Benchmarks
# Uses Google Benchmark for comprehensive performance testing

# Try to find Google Benchmark, or fetch it
find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, fetching from GitHub...")

    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )

    # Don't build benchmark tests
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(benchmark)
endif()

message(STATUS "Building Kernelix benchmarks with Google Benchmark")

# Main benchmark executable
add_executable(kernelix_bench
    bench_gemm.cpp
    bench_kernels.cpp
    bench_elementwise.cpp
    bench_comparison.cpp
)

target_link_libraries(kernelix_bench PRIVATE
    Kernelix::kernelix
    benchmark::benchmark
    benchmark::benchmark_main
)

# Ensure optimizations for benchmarks
target_compile_options(kernelix_bench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -DNDEBUG>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)
