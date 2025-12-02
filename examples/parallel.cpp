#include <kernelix/kernelix.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

using namespace kernelix;

void benchmark_parallel_for() {
    std::cout << "\n--- Benchmarking parallel_for ---" << std::endl;

    constexpr std::size_t N = 10'000'000;
    std::vector<float> data(N);

    // Warm up
    parallel::parallel_for(0, N, [&data](std::size_t i) {
        data[i] = static_cast<float>(i) * 0.5f;
    });

    // Benchmark parallel
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        parallel::parallel_for(0, N, [&data](std::size_t i) {
            data[i] = std::sin(static_cast<float>(i) * 0.001f) *
                      std::cos(static_cast<float>(i) * 0.001f);
        });
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Parallel time (10 iterations): " << parallel_time << " ms" << std::endl;
    std::cout << "Per iteration: " << parallel_time / 10.0 << " ms" << std::endl;
    std::cout << "Throughput: " << (N * 10.0 / parallel_time * 1000.0 / 1e9)
              << " billion elements/sec" << std::endl;
}

void benchmark_parallel_reduce() {
    std::cout << "\n--- Benchmarking parallel_reduce ---" << std::endl;

    constexpr std::size_t N = 100'000'000;

    // Warm up
    auto sum = parallel::parallel_reduce(
        std::size_t{0}, N,
        0.0,
        [](std::size_t i) { return static_cast<double>(i); },
        [](double a, double b) { return a + b; }
    );

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        sum = parallel::parallel_reduce(
            std::size_t{0}, N,
            0.0,
            [](std::size_t i) { return std::sin(static_cast<double>(i) * 1e-9); },
            [](double a, double b) { return a + b; }
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto reduce_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Parallel reduce time (10 iterations): " << reduce_time << " ms" << std::endl;
    std::cout << "Per iteration: " << reduce_time / 10.0 << " ms" << std::endl;
    std::cout << "Result (last): " << sum << std::endl;
}

void benchmark_parallel_2d() {
    std::cout << "\n--- Benchmarking parallel_for_2d ---" << std::endl;

    constexpr std::size_t M = 1000;
    constexpr std::size_t N = 1000;
    std::vector<float> matrix(M * N);

    // Warm up
    parallel::parallel_for_2d(0, M, 0, N, [&matrix, N](std::size_t i, std::size_t j) {
        matrix[i * N + j] = static_cast<float>(i + j);
    });

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        parallel::parallel_for_2d(0, M, 0, N, [&matrix, N](std::size_t i, std::size_t j) {
            matrix[i * N + j] = std::sin(static_cast<float>(i * 0.01f)) *
                                std::cos(static_cast<float>(j * 0.01f));
        });
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time_2d = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Parallel 2D time (100 iterations): " << time_2d << " ms" << std::endl;
    std::cout << "Per iteration: " << time_2d / 100.0 << " ms" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << " = " << M * N << " elements" << std::endl;
}

int main() {
    std::cout << "Kernelix v" << Version::string << " - TBB Parallelism Example" << std::endl;
    std::cout << "============================================================" << std::endl;

    std::cout << "Number of threads: " << num_threads() << std::endl;

    benchmark_parallel_for();
    benchmark_parallel_reduce();
    benchmark_parallel_2d();

    std::cout << "\n============================================================" << std::endl;
    std::cout << "Example completed successfully!" << std::endl;

    return 0;
}
