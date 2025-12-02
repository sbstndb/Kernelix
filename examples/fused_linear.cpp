#include <kernelix/kernelix.hpp>
#include <iostream>

using namespace kernelix;

int main() {
    std::cout << "Kernelix v" << Version::string << " - Fused Linear Example" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // Simulating a transformer-like layer
    constexpr std::size_t batch_size = 32;
    constexpr std::size_t hidden_dim = 4096;

    Tensor<float, batch_size, hidden_dim> x;
    Tensor<float, hidden_dim, hidden_dim> W_q, W_k, W_v;
    Tensor<float, hidden_dim> bias_q, bias_k, bias_v;

    // Initialize
    x.fill(0.1f);
    W_q.fill(0.01f);
    W_k.fill(0.01f);
    W_v.fill(0.01f);
    bias_q.fill(0.0f);
    bias_k.fill(0.0f);
    bias_v.fill(0.0f);

    // Build expression tree for Q, K, V projections
    // In a full implementation, these would be fused automatically
    auto Q = linear(x, W_q, bias_q);
    auto K = linear(x, W_k, bias_k);
    auto V = linear(x, W_v, bias_v);

    std::cout << "Created QKV projection expressions" << std::endl;
    std::cout << "Input shape: " << batch_size << " x " << hidden_dim << std::endl;
    std::cout << "Weight shapes: " << hidden_dim << " x " << hidden_dim << std::endl;

    // Expression with activation (SwiGLU-like pattern)
    Tensor<float, hidden_dim, hidden_dim * 4> W_gate, W_up;
    Tensor<float, hidden_dim * 4, hidden_dim> W_down;
    W_gate.fill(0.01f);
    W_up.fill(0.01f);
    W_down.fill(0.01f);

    // SwiGLU: silu(x @ W_gate) * (x @ W_up) @ W_down
    auto gate_out = silu(linear(x, W_gate));
    auto up_out = linear(x, W_up);
    // Note: elementwise multiplication would be: gate_out * up_out
    // Then: linear(gate_out * up_out, W_down)

    std::cout << "Created SwiGLU-like FFN expressions" << std::endl;

    std::cout << "=======================================================" << std::endl;
    std::cout << "Example completed successfully!" << std::endl;

    return 0;
}
