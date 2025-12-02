#include <kernelix/kernelix.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace kernelix;

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

void test_expression_creation() {
    std::cout << "Testing expression creation... ";

    Tensor<float, 4, 4> A, B;
    A.fill(1.0f);
    B.fill(2.0f);

    // GEMM expression
    auto gemm_expr = gemm(A, B);
    (void)gemm_expr;  // Just test creation

    // Linear expression
    auto linear_expr = linear(A, B);
    (void)linear_expr;

    // Chained expressions
    auto relu_expr = relu(gemm(A, B));
    (void)relu_expr;

    std::cout << "PASSED" << std::endl;
}

void test_parallel_for() {
    std::cout << "Testing parallel_for... ";

    constexpr std::size_t N = 10000;
    std::vector<float> data(N, 0.0f);

    // Use parallel_for to fill data
    parallel::parallel_for(0, N, [&data](std::size_t i) {
        data[i] = static_cast<float>(i);
    });

    // Verify
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

    // Sum 0 to N-1
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
    std::cout << "Kernelix v" << Version::string << " - Minimal Tests" << std::endl;
    std::cout << "================================================" << std::endl;

    test_tensor_creation();
    test_dynamic_tensor();
    test_tensor_view();
    test_expression_creation();
    test_parallel_for();
    test_parallel_reduce();
    test_simd_basic();
    test_config();

    std::cout << "================================================" << std::endl;
    std::cout << "All tests PASSED!" << std::endl;

    return 0;
}
