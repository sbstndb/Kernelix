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
