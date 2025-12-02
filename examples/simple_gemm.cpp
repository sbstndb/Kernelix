#include <kernelix/kernelix.hpp>
#include <iostream>
#include <random>

using namespace kernelix;

int main() {
    std::cout << "Kernelix v" << Version::string << " - Simple GEMM Example" << std::endl;
    std::cout << "=====================================================" << std::endl;

    // Create matrices with static dimensions
    constexpr std::size_t M = 64;
    constexpr std::size_t K = 128;
    constexpr std::size_t N = 64;

    Tensor<float, M, K> A;
    Tensor<float, K, N> B;
    Tensor<float, M, N> C;

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& val : A) val = dist(gen);
    for (auto& val : B) val = dist(gen);
    C.fill(0.0f);

    // Create GEMM expression
    auto gemm_expr = gemm(A, B);

    std::cout << "Created GEMM expression for " << M << "x" << K << " @ " << K << "x" << N << std::endl;
    std::cout << "Matrix A size: " << A.size() << " elements" << std::endl;
    std::cout << "Matrix B size: " << B.size() << " elements" << std::endl;
    std::cout << "Matrix C size: " << C.size() << " elements" << std::endl;

    // Expression with activation
    auto relu_gemm = relu(gemm(A, B));
    std::cout << "Created ReLU(GEMM) expression" << std::endl;

    // Note: eval() is a placeholder - full implementation coming soon
    // eval(gemm_expr, C.data());

    std::cout << "=====================================================" << std::endl;
    std::cout << "Example completed successfully!" << std::endl;

    return 0;
}
