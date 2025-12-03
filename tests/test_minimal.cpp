#include <kernelix/kernelix.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <chrono>

using namespace kernelix;

// Helper for float comparison
bool approx_equal(float a, float b, float rel_tol = 1e-4f, float abs_tol = 1e-6f) {
    return std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol);
}

void test_tensor_creation() {
    std::cout << "Testing tensor creation... ";

    // Static tensor
    Tensor<float, 2, 3> t1;
    t1.fill(1.0f);
    assert(t1.size() == 6);
    assert(t1[0] == 1.0f);

    // Tensor from initializer list
    Tensor<float, 2, 2> t2 = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(t2(0, 0) == 1.0f);
    assert(t2(1, 1) == 4.0f);

    std::cout << "PASSED" << std::endl;
}

void test_dynamic_tensor() {
    std::cout << "Testing dynamic tensor... ";

    DynamicTensor<float, 2> dt({4, 4});
    dt.fill(2.0f);
    assert(dt.size() == 16);
    assert(dt.dim(0) == 4);
    assert(dt.dim(1) == 4);
    assert(dt[0] == 2.0f);

    std::cout << "PASSED" << std::endl;
}

void test_tensor_view() {
    std::cout << "Testing tensor view... ";

    Tensor<float, 4, 4> t;
    t.fill(3.0f);

    TensorView<float, 4, 4> view(t);
    assert(view.size() == 16);
    assert(view[0] == 3.0f);

    // Modify through view
    view[0] = 5.0f;
    assert(t[0] == 5.0f);

    std::cout << "PASSED" << std::endl;
}

void test_gemm_small() {
    std::cout << "Testing small GEMM (2x3 @ 3x2)... ";

    // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
    // B = [[1, 2], [3, 4], [5, 6]]  (3x2)
    // C = A @ B = [[22, 28], [49, 64]]  (2x2)

    Tensor<float, 2, 3> A = {1, 2, 3, 4, 5, 6};
    Tensor<float, 3, 2> B = {1, 2, 3, 4, 5, 6};
    Tensor<float, 2, 2> C;

    auto expr = gemm(A, B);
    eval(expr, C.data());

    assert(approx_equal(C(0, 0), 22.0f));
    assert(approx_equal(C(0, 1), 28.0f));
    assert(approx_equal(C(1, 0), 49.0f));
    assert(approx_equal(C(1, 1), 64.0f));

    std::cout << "PASSED" << std::endl;
}

void test_gemm_medium() {
    std::cout << "Testing medium GEMM (32x64 @ 64x32)... ";

    constexpr std::size_t M = 32, K = 64, N = 32;

    Tensor<float, M, K> A;
    Tensor<float, K, N> B;
    Tensor<float, M, N> C;
    Tensor<float, M, N> C_ref;

    // Initialize with simple pattern
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            A(i, k) = static_cast<float>(i + k) * 0.01f;
        }
    }
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t j = 0; j < N; ++j) {
            B(k, j) = static_cast<float>(k - j) * 0.01f;
        }
    }

    // Compute reference (naive)
    C_ref.fill(0.0f);
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t j = 0; j < N; ++j) {
                C_ref(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    // Compute with Kernelix
    auto expr = gemm(A, B);
    eval(expr, C.data());

    // Verify
    bool correct = true;
    for (std::size_t i = 0; i < M * N; ++i) {
        if (!approx_equal(C[i], C_ref[i])) {
            std::cerr << "\nMismatch at " << i << ": got " << C[i] << ", expected " << C_ref[i] << std::endl;
            correct = false;
            break;
        }
    }
    assert(correct);

    std::cout << "PASSED" << std::endl;
}

void test_gemm_large() {
    std::cout << "Testing large GEMM (128x256 @ 256x128)... ";

    constexpr std::size_t M = 128, K = 256, N = 128;

    Tensor<float, M, K> A;
    Tensor<float, K, N> B;
    Tensor<float, M, N> C;

    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    // Compute reference (sample a few points)
    auto start = std::chrono::high_resolution_clock::now();
    eval(gemm(A, B), C.data());
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Verify a few sample points
    float c00_ref = 0, c_mid_ref = 0, c_last_ref = 0;
    for (std::size_t k = 0; k < K; ++k) {
        c00_ref += A(0, k) * B(k, 0);
        c_mid_ref += A(M/2, k) * B(k, N/2);
        c_last_ref += A(M-1, k) * B(k, N-1);
    }

    assert(approx_equal(C(0, 0), c00_ref));
    assert(approx_equal(C(M/2, N/2), c_mid_ref));
    assert(approx_equal(C(M-1, N-1), c_last_ref));

    double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
    std::cout << "PASSED (" << time_ms << " ms, " << gflops << " GFLOPS)" << std::endl;
}

void test_gemm_with_bias() {
    std::cout << "Testing GEMM with bias... ";

    Tensor<float, 2, 3> A = {1, 2, 3, 4, 5, 6};
    Tensor<float, 3, 2> B = {1, 2, 3, 4, 5, 6};
    Tensor<float, 2> bias = {10, 20};
    Tensor<float, 2, 2> C;

    // C = A @ B + bias
    // [[22, 28], [49, 64]] + [10, 20] = [[32, 48], [59, 84]]
    auto expr = gemm(A, B, bias);
    eval(expr, C.data());

    assert(approx_equal(C(0, 0), 32.0f));
    assert(approx_equal(C(0, 1), 48.0f));
    assert(approx_equal(C(1, 0), 59.0f));
    assert(approx_equal(C(1, 1), 84.0f));

    std::cout << "PASSED" << std::endl;
}

void test_gemv() {
    std::cout << "Testing GEMV... ";

    // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
    // x = [1, 2, 3]  (3)
    // y = A @ x = [14, 32]  (2)

    Tensor<float, 2, 3> A = {1, 2, 3, 4, 5, 6};
    Tensor<float, 3> x = {1, 2, 3};
    Tensor<float, 2> y;

    auto expr = gemv(A, x);
    eval(expr, y.data());

    assert(approx_equal(y[0], 14.0f));
    assert(approx_equal(y[1], 32.0f));

    std::cout << "PASSED" << std::endl;
}

void test_dot_product() {
    std::cout << "Testing dot product... ";

    Tensor<float, 4> a = {1, 2, 3, 4};
    Tensor<float, 4> b = {4, 3, 2, 1};

    // dot = 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
    auto expr = dot(a, b);
    float result;
    eval(expr, &result);

    assert(approx_equal(result, 20.0f));

    std::cout << "PASSED" << std::endl;
}

void test_relu_activation() {
    std::cout << "Testing ReLU activation... ";

    Tensor<float, 2, 3> input = {-1, 2, -3, 4, -5, 6};
    Tensor<float, 2, 3> output;

    auto expr = relu(input);
    eval(expr, output.data());

    assert(approx_equal(output[0], 0.0f));
    assert(approx_equal(output[1], 2.0f));
    assert(approx_equal(output[2], 0.0f));
    assert(approx_equal(output[3], 4.0f));
    assert(approx_equal(output[4], 0.0f));
    assert(approx_equal(output[5], 6.0f));

    std::cout << "PASSED" << std::endl;
}

void test_silu_activation() {
    std::cout << "Testing SiLU activation... ";

    Tensor<float, 4> input = {-2, -1, 0, 1};
    Tensor<float, 4> output;

    auto expr = silu(input);
    eval(expr, output.data());

    // SiLU(x) = x * sigmoid(x)
    auto silu_ref = [](float x) { return x / (1.0f + std::exp(-x)); };

    assert(approx_equal(output[0], silu_ref(-2.0f)));
    assert(approx_equal(output[1], silu_ref(-1.0f)));
    assert(approx_equal(output[2], 0.0f));
    assert(approx_equal(output[3], silu_ref(1.0f)));

    std::cout << "PASSED" << std::endl;
}

void test_fused_gemm_relu() {
    std::cout << "Testing fused GEMM + ReLU... ";

    Tensor<float, 2, 3> A = {1, 2, 3, -1, -2, -3};
    Tensor<float, 3, 2> B = {1, -1, 1, -1, 1, -1};
    Tensor<float, 2, 2> C;

    // GEMM result: [[6, -6], [-6, 6]]
    // After ReLU: [[6, 0], [0, 6]]
    auto expr = relu(gemm(A, B));
    eval(expr, C.data());

    assert(approx_equal(C(0, 0), 6.0f));
    assert(approx_equal(C(0, 1), 0.0f));
    assert(approx_equal(C(1, 0), 0.0f));
    assert(approx_equal(C(1, 1), 6.0f));

    std::cout << "PASSED" << std::endl;
}

void test_compute_returns_tensor() {
    std::cout << "Testing compute() returns correct tensor... ";

    Tensor<float, 2, 3> A = {1, 2, 3, 4, 5, 6};
    Tensor<float, 3, 2> B = {1, 2, 3, 4, 5, 6};

    // compute() should return a Tensor<float, 2, 2>
    auto result = compute(gemm(A, B));

    static_assert(std::is_same_v<decltype(result), Tensor<float, 2, 2>>);

    assert(approx_equal(result(0, 0), 22.0f));
    assert(approx_equal(result(1, 1), 64.0f));

    std::cout << "PASSED" << std::endl;
}

void test_expr_traits() {
    std::cout << "Testing expression traits... ";

    using A = Tensor<float, 4, 8>;
    using B = Tensor<float, 8, 16>;
    using GemmType = decltype(gemm(std::declval<A&>(), std::declval<B&>()));

    using Traits = traits::ExprTraits<GemmType>;

    static_assert(Traits::M == 4);
    static_assert(Traits::K == 8);
    static_assert(Traits::N == 16);
    static_assert(Traits::output_shape[0] == 4);
    static_assert(Traits::output_shape[1] == 16);
    static_assert(Traits::output_size == 64);
    static_assert(Traits::estimated_flops == 2 * 4 * 16 * 8);

    std::cout << "PASSED" << std::endl;
}

void test_parallel_for() {
    std::cout << "Testing parallel_for... ";

    constexpr std::size_t N = 10000;
    std::vector<float> data(N, 0.0f);

    parallel::parallel_for(0, N, [&data](std::size_t i) {
        data[i] = static_cast<float>(i);
    });

    bool correct = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (data[i] != static_cast<float>(i)) {
            correct = false;
            break;
        }
    }
    assert(correct);

    std::cout << "PASSED" << std::endl;
}

void test_parallel_reduce() {
    std::cout << "Testing parallel_reduce... ";

    constexpr std::size_t N = 10000;

    auto sum = parallel::parallel_reduce(
        std::size_t{0}, N,
        0.0,
        [](std::size_t i) { return static_cast<double>(i); },
        [](double a, double b) { return a + b; }
    );

    double expected = static_cast<double>(N) * (N - 1) / 2.0;
    assert(std::abs(sum - expected) < 1e-6);

    std::cout << "PASSED" << std::endl;
}

void test_simd_basic() {
    std::cout << "Testing SIMD basics... ";

    using V = simd::Vec<float>;

    alignas(64) float a[V::width];
    alignas(64) float b[V::width];
    alignas(64) float c[V::width];

    for (std::size_t i = 0; i < V::width; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    auto va = V::load_aligned(a);
    auto vb = V::load_aligned(b);
    auto vc = V::add(va, vb);
    V::store_aligned(c, vc);

    for (std::size_t i = 0; i < V::width; ++i) {
        assert(c[i] == 3.0f);
    }

    std::cout << "PASSED" << std::endl;
}

void test_config() {
    std::cout << "Testing configuration... ";

    Config cfg;
    cfg.hardware.num_threads = 4;
    cfg.enable_parallel = true;
    cfg.parallel_threshold = 1000;

    set_config(cfg);

    auto& current = get_config();
    assert(current.hardware.num_threads == 4);
    assert(current.enable_parallel == true);
    assert(current.parallel_threshold == 1000);

    std::cout << "PASSED" << std::endl;
}

void test_rmsnorm() {
    std::cout << "Testing RMSNorm... ";

    constexpr std::size_t batch = 2;
    constexpr std::size_t hidden = 4;

    // Input: simple values for manual verification
    Tensor<float, batch, hidden> x = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor<float, hidden> weight = {1, 1, 1, 1};  // Identity weight
    Tensor<float, batch, hidden> output;

    // RMSNorm formula: y = x * weight / sqrt(mean(x^2) + eps)
    // Row 0: x = [1, 2, 3, 4], mean(x^2) = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5) ≈ 2.739
    // Row 1: x = [5, 6, 7, 8], mean(x^2) = (25+36+49+64)/4 = 43.5, rms = sqrt(43.5) ≈ 6.595

    auto expr = rmsnorm(x, weight, 1e-6f);
    eval(expr, output.data());

    // Verify row 0
    float rms0 = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + 1e-6f);
    assert(approx_equal(output(0, 0), 1.0f / rms0));
    assert(approx_equal(output(0, 1), 2.0f / rms0));
    assert(approx_equal(output(0, 2), 3.0f / rms0));
    assert(approx_equal(output(0, 3), 4.0f / rms0));

    // Verify row 1
    float rms1 = std::sqrt((25.0f + 36.0f + 49.0f + 64.0f) / 4.0f + 1e-6f);
    assert(approx_equal(output(1, 0), 5.0f / rms1));
    assert(approx_equal(output(1, 1), 6.0f / rms1));

    std::cout << "PASSED" << std::endl;
}

void test_rmsnorm_with_weight() {
    std::cout << "Testing RMSNorm with weights... ";

    constexpr std::size_t batch = 2;
    constexpr std::size_t hidden = 4;

    Tensor<float, batch, hidden> x = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor<float, hidden> weight = {2, 2, 2, 2};  // Scale by 2
    Tensor<float, batch, hidden> output;

    eval(rmsnorm(x, weight), output.data());

    // Result should be 2x the identity weight case
    float rms0 = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + 1e-6f);
    assert(approx_equal(output(0, 0), 2.0f * 1.0f / rms0));
    assert(approx_equal(output(0, 1), 2.0f * 2.0f / rms0));

    std::cout << "PASSED" << std::endl;
}

void test_layernorm() {
    std::cout << "Testing LayerNorm... ";

    constexpr std::size_t batch = 2;
    constexpr std::size_t hidden = 4;

    // Simple test: input where mean=2.5 for each row
    Tensor<float, batch, hidden> x = {1, 2, 3, 4, 1, 2, 3, 4};
    Tensor<float, hidden> gamma = {1, 1, 1, 1};  // Identity scale
    Tensor<float, hidden> beta = {0, 0, 0, 0};   // Zero shift
    Tensor<float, batch, hidden> output;

    // LayerNorm formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
    // Row mean = 2.5, var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
    //          = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25

    auto expr = layernorm(x, gamma, beta, 1e-5f);
    eval(expr, output.data());

    float mean = 2.5f;
    float var = 1.25f;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);

    // Verify row 0
    assert(approx_equal(output(0, 0), (1.0f - mean) * inv_std));
    assert(approx_equal(output(0, 1), (2.0f - mean) * inv_std));
    assert(approx_equal(output(0, 2), (3.0f - mean) * inv_std));
    assert(approx_equal(output(0, 3), (4.0f - mean) * inv_std));

    // Row 1 should be identical to row 0
    assert(approx_equal(output(1, 0), output(0, 0)));
    assert(approx_equal(output(1, 3), output(0, 3)));

    std::cout << "PASSED" << std::endl;
}

void test_layernorm_with_params() {
    std::cout << "Testing LayerNorm with gamma/beta... ";

    constexpr std::size_t batch = 1;
    constexpr std::size_t hidden = 4;

    Tensor<float, batch, hidden> x = {1, 2, 3, 4};
    Tensor<float, hidden> gamma = {2, 2, 2, 2};  // Scale by 2
    Tensor<float, hidden> beta = {1, 1, 1, 1};   // Shift by 1
    Tensor<float, batch, hidden> output;

    eval(layernorm(x, gamma, beta), output.data());

    float mean = 2.5f;
    float var = 1.25f;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);

    // y = (x - mean) * inv_std * gamma + beta
    assert(approx_equal(output(0, 0), (1.0f - mean) * inv_std * 2.0f + 1.0f));
    assert(approx_equal(output(0, 3), (4.0f - mean) * inv_std * 2.0f + 1.0f));

    std::cout << "PASSED" << std::endl;
}

void test_rmsnorm_large() {
    std::cout << "Testing RMSNorm (large)... ";

    constexpr std::size_t batch = 32;
    constexpr std::size_t hidden = 4096;

    Tensor<float, batch, hidden> x;
    Tensor<float, hidden> weight;
    Tensor<float, batch, hidden> output;

    // Initialize
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
    for (auto& v : weight) v = dist(rng) * 0.1f + 1.0f;  // Around 1.0

    auto start = std::chrono::high_resolution_clock::now();
    eval(rmsnorm(x, weight), output.data());
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Verify one sample row
    float sum_sq = 0;
    for (std::size_t j = 0; j < hidden; ++j) {
        sum_sq += x(0, j) * x(0, j);
    }
    float rms = std::sqrt(sum_sq / hidden + 1e-6f);
    float expected = x(0, 0) / rms * weight[0];
    assert(approx_equal(output(0, 0), expected, 1e-3f));

    std::cout << "PASSED (" << time_ms << " ms)" << std::endl;
}

void test_fused_rmsnorm_linear() {
    std::cout << "Testing fused RMSNorm + Linear... ";

    constexpr std::size_t batch = 4;
    constexpr std::size_t hidden_in = 8;
    constexpr std::size_t hidden_out = 8;

    Tensor<float, batch, hidden_in> x;
    Tensor<float, hidden_in> norm_weight;
    Tensor<float, hidden_in, hidden_out> linear_weight;
    Tensor<float, batch, hidden_out> output;

    // Initialize
    x.fill(1.0f);
    norm_weight.fill(1.0f);
    linear_weight.fill(0.1f);

    // Fused: linear(rmsnorm(x, w), W)
    auto expr = linear(rmsnorm(x, norm_weight), linear_weight);
    eval(expr, output.data());

    // RMSNorm of all-ones: rms = 1.0, so normalized = 1.0
    // Linear: each output = sum of 8 * 0.1 = 0.8
    assert(approx_equal(output(0, 0), 0.8f, 1e-3f));
    assert(approx_equal(output(batch-1, hidden_out-1), 0.8f, 1e-3f));

    std::cout << "PASSED" << std::endl;
}

void test_softmax() {
    std::cout << "Testing Softmax... ";

    constexpr std::size_t batch = 2;
    constexpr std::size_t dim = 4;

    // Simple test: softmax over known values
    Tensor<float, batch, dim> x = {1, 2, 3, 4, 0, 0, 0, 0};
    Tensor<float, batch, dim> output;

    auto expr = softmax(x);
    eval(expr, output.data());

    // Row 0: softmax([1, 2, 3, 4])
    // exp values: e^1 ≈ 2.718, e^2 ≈ 7.389, e^3 ≈ 20.086, e^4 ≈ 54.598
    // sum ≈ 84.791
    float e1 = std::exp(1.0f), e2 = std::exp(2.0f), e3 = std::exp(3.0f), e4 = std::exp(4.0f);
    float sum0 = e1 + e2 + e3 + e4;

    assert(approx_equal(output(0, 0), e1 / sum0));
    assert(approx_equal(output(0, 1), e2 / sum0));
    assert(approx_equal(output(0, 2), e3 / sum0));
    assert(approx_equal(output(0, 3), e4 / sum0));

    // Row 1: softmax([0, 0, 0, 0]) = [0.25, 0.25, 0.25, 0.25]
    assert(approx_equal(output(1, 0), 0.25f));
    assert(approx_equal(output(1, 1), 0.25f));
    assert(approx_equal(output(1, 2), 0.25f));
    assert(approx_equal(output(1, 3), 0.25f));

    // Sum of each row should be 1.0
    float sum_row0 = output(0, 0) + output(0, 1) + output(0, 2) + output(0, 3);
    float sum_row1 = output(1, 0) + output(1, 1) + output(1, 2) + output(1, 3);
    assert(approx_equal(sum_row0, 1.0f));
    assert(approx_equal(sum_row1, 1.0f));

    std::cout << "PASSED" << std::endl;
}

void test_softmax_numerical_stability() {
    std::cout << "Testing Softmax numerical stability... ";

    constexpr std::size_t batch = 1;
    constexpr std::size_t dim = 4;

    // Large values that would overflow without max subtraction
    Tensor<float, batch, dim> x = {1000, 1001, 1002, 1003};
    Tensor<float, batch, dim> output;

    eval(softmax(x), output.data());

    // Despite large values, should still produce valid probabilities
    // After max subtraction: [0, 1, 2, 3] -> same ratios as test_softmax row 0
    float e0 = std::exp(0.0f), e1 = std::exp(1.0f), e2 = std::exp(2.0f), e3 = std::exp(3.0f);
    float sum = e0 + e1 + e2 + e3;

    assert(approx_equal(output(0, 0), e0 / sum));
    assert(approx_equal(output(0, 1), e1 / sum));
    assert(approx_equal(output(0, 2), e2 / sum));
    assert(approx_equal(output(0, 3), e3 / sum));

    // Sum should still be 1.0
    float total = output(0, 0) + output(0, 1) + output(0, 2) + output(0, 3);
    assert(approx_equal(total, 1.0f));

    // No NaN or Inf
    for (std::size_t i = 0; i < dim; ++i) {
        assert(!std::isnan(output(0, i)));
        assert(!std::isinf(output(0, i)));
    }

    std::cout << "PASSED" << std::endl;
}

void test_softmax_1d() {
    std::cout << "Testing Softmax 1D... ";

    constexpr std::size_t dim = 5;

    Tensor<float, dim> x = {1, 2, 3, 4, 5};
    Tensor<float, dim> output;

    eval(softmax(x), output.data());

    // Verify probabilities sum to 1
    float sum = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += output[i];
        assert(output[i] > 0.0f);  // All positive
        assert(output[i] < 1.0f);  // All less than 1
    }
    assert(approx_equal(sum, 1.0f));

    // Verify ordering (larger input -> larger probability)
    for (std::size_t i = 1; i < dim; ++i) {
        assert(output[i] > output[i-1]);
    }

    std::cout << "PASSED" << std::endl;
}

void test_softmax_large() {
    std::cout << "Testing Softmax (large)... ";

    constexpr std::size_t batch = 16;
    constexpr std::size_t dim = 4096;  // Reasonable size for stack allocation

    Tensor<float, batch, dim> x;
    Tensor<float, batch, dim> output;

    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : x) v = dist(rng);

    auto start = std::chrono::high_resolution_clock::now();
    eval(softmax(x), output.data());
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Verify a few sample rows sum to 1
    for (std::size_t row = 0; row < batch; row += batch/2) {
        float sum = 0;
        for (std::size_t j = 0; j < dim; ++j) {
            sum += output(row, j);
            assert(!std::isnan(output(row, j)));
            assert(!std::isinf(output(row, j)));
        }
        assert(approx_equal(sum, 1.0f, 1e-3f));
    }

    std::cout << "PASSED (" << time_ms << " ms)" << std::endl;
}

int main() {
    std::cout << "Kernelix v" << Version::string << " - Tests" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "SIMD width: " << simd::Vec<float>::width << " floats" << std::endl;
    std::cout << "Threads: " << num_threads() << std::endl;
    std::cout << "=============================================" << std::endl;

    // Core tests
    test_tensor_creation();
    test_dynamic_tensor();
    test_tensor_view();

    // GEMM tests
    test_gemm_small();
    test_gemm_medium();
    test_gemm_large();
    test_gemm_with_bias();

    // Other contractions
    test_gemv();
    test_dot_product();

    // Activations
    test_relu_activation();
    test_silu_activation();

    // Fusion
    test_fused_gemm_relu();

    // Normalizations
    test_rmsnorm();
    test_rmsnorm_with_weight();
    test_layernorm();
    test_layernorm_with_params();
    test_rmsnorm_large();
    test_fused_rmsnorm_linear();

    // Softmax
    test_softmax();
    test_softmax_numerical_stability();
    test_softmax_1d();
    test_softmax_large();

    // API tests
    test_compute_returns_tensor();
    test_expr_traits();

    // Utilities
    test_parallel_for();
    test_parallel_reduce();
    test_simd_basic();
    test_config();

    std::cout << "=============================================" << std::endl;
    std::cout << "All tests PASSED!" << std::endl;

    return 0;
}
