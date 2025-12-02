# Kernelix Benchmarks

find_package(benchmark QUIET)

if(benchmark_FOUND)
    message(STATUS "Found Google Benchmark, building benchmarks")

    add_executable(kernelix_bench
        bench_gemm.cpp
        bench_gemv.cpp
        bench_parallel.cpp
    )

    target_link_libraries(kernelix_bench PRIVATE
        Kernelix::kernelix
        benchmark::benchmark
        benchmark::benchmark_main
    )

else()
    message(STATUS "Google Benchmark not found, building simple benchmarks")

    add_executable(kernelix_bench
        bench_simple.cpp
    )

    target_link_libraries(kernelix_bench PRIVATE
        Kernelix::kernelix
    )
endif()
