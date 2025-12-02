#include <kernelix/kernelix.hpp>
#include <iostream>
#include <random>
#include <chrono>

using namespace kernelix;

int main() {
    std::cout << "Kernelix v" << Version::string << " - Simple GEMM Example" << std::endl;
    std::cout << "=====================================================" << std::endl;

    // Create matrices with static dimensions
    constexpr std::size_t M = 256;
    constexpr std::size_t K = 512;
    constexpr std::size_t N = 256;

    Tensor<float, M, K> A;
    Tensor<float, K, N> B;
    Tensor<float, M, N> C;

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& val : A) val = dist(gen);
    for (auto& val : B) val = dist(gen);

    std::cout << "Matrix dimensions:" << std::endl;
    std::cout << "  A: " << M << " x " << K << std::endl;
    std::cout << "  B: " << K << " x " << N << std::endl;
    std::cout << "  C: " << M << " x " << N << std::endl;

    // Warm up
    eval(gemm(A, B), C.data());

    // Benchmark
    constexpr int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        eval(gemm(A, B), C.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / num_iterations;

    // Calculate GFLOPS
    double flops = 2.0 * M * N * K;
    double gflops = (flops / avg_ms) / 1e6;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Time per GEMM: " << avg_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << gflops << std::endl;
    std::cout << "  Estimated FLOPs: " << ExprInfo<decltype(gemm(A, B))>::flops() << std::endl;

    // Verify a sample value
    float c00_ref = 0;
    for (std::size_t k = 0; k < K; ++k) {
        c00_ref += A(0, k) * B(k, 0);
    }
    std::cout << "\nVerification:" << std::endl;
    std::cout << "  C[0,0] computed: " << C(0, 0) << std::endl;
    std::cout << "  C[0,0] expected: " << c00_ref << std::endl;

    std::cout << "\n=====================================================" << std::endl;
    std::cout << "Example completed successfully!" << std::endl;

    return 0;
}
