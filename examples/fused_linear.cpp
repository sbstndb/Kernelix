#include <kernelix/kernelix.hpp>
#include <iostream>
#include <chrono>

using namespace kernelix;

int main() {
    std::cout << "Kernelix v" << Version::string << " - Fused Linear Example" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // Simulating a simple MLP layer
    constexpr std::size_t batch_size = 32;
    constexpr std::size_t input_dim = 512;
    constexpr std::size_t hidden_dim = 1024;
    constexpr std::size_t output_dim = 512;

    // Input and weights
    Tensor<float, batch_size, input_dim> x;
    Tensor<float, input_dim, hidden_dim> W1;
    Tensor<float, hidden_dim> bias1;
    Tensor<float, hidden_dim, output_dim> W2;
    Tensor<float, output_dim> bias2;

    // Initialize with simple values
    x.fill(0.1f);
    W1.fill(0.01f);
    bias1.fill(0.0f);
    W2.fill(0.01f);
    bias2.fill(0.0f);

    // Intermediate and output buffers
    Tensor<float, batch_size, hidden_dim> hidden;
    Tensor<float, batch_size, output_dim> output;

    std::cout << "\nNetwork architecture:" << std::endl;
    std::cout << "  Input:  " << batch_size << " x " << input_dim << std::endl;
    std::cout << "  Hidden: " << batch_size << " x " << hidden_dim << " (ReLU)" << std::endl;
    std::cout << "  Output: " << batch_size << " x " << output_dim << std::endl;

    // Layer 1: Linear + ReLU (fused)
    std::cout << "\nExecuting Layer 1 (Linear + ReLU)..." << std::endl;
    auto layer1_expr = relu(linear(x, W1, bias1));

    auto start = std::chrono::high_resolution_clock::now();
    eval(layer1_expr, hidden.data());
    auto end = std::chrono::high_resolution_clock::now();

    double layer1_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << layer1_ms << " ms" << std::endl;

    // Layer 2: Linear
    std::cout << "\nExecuting Layer 2 (Linear)..." << std::endl;
    auto layer2_expr = linear(hidden, W2, bias2);

    start = std::chrono::high_resolution_clock::now();
    eval(layer2_expr, output.data());
    end = std::chrono::high_resolution_clock::now();

    double layer2_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << layer2_ms << " ms" << std::endl;

    // Total time
    double total_ms = layer1_ms + layer2_ms;
    std::cout << "\nTotal forward pass: " << total_ms << " ms" << std::endl;

    // Calculate total FLOPs
    std::size_t layer1_flops = 2 * batch_size * input_dim * hidden_dim;
    std::size_t layer2_flops = 2 * batch_size * hidden_dim * output_dim;
    double total_gflops = (layer1_flops + layer2_flops) / (total_ms * 1e6);
    std::cout << "Throughput: " << total_gflops << " GFLOPS" << std::endl;

    // Verify output is reasonable (should be positive due to ReLU in layer 1)
    std::cout << "\nSample outputs:" << std::endl;
    std::cout << "  hidden[0,0]: " << hidden(0, 0) << " (should be >= 0)" << std::endl;
    std::cout << "  output[0,0]: " << output(0, 0) << std::endl;

    std::cout << "\n=======================================================" << std::endl;
    std::cout << "Example completed successfully!" << std::endl;

    return 0;
}
