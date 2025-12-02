#include <kernelix/kernelix.hpp>
#include <iostream>
#include <chrono>
#include <random>

using namespace kernelix;

template<typename Func>
double benchmark(const std::string& name, int iterations, Func&& func) {
    // Warm up
    for (int i = 0; i < 3; ++i) {
        func();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = time_ms / iterations;

    std::cout << name << ": " << avg_ms << " ms/iter (" << iterations << " iterations)" << std::endl;
    return avg_ms;
}

void bench_tensor_creation() {
    std::cout << "\n=== Tensor Creation ===" << std::endl;

    benchmark("Static tensor 1024x1024", 1000, []() {
        Tensor<float, 1024, 1024> t;
        t.fill(1.0f);
    });

    benchmark("Dynamic tensor 1024x1024", 1000, []() {
        DynamicTensor<float, 2> t({1024, 1024});
        t.fill(1.0f);
    });
}

void bench_parallel_operations() {
    std::cout << "\n=== Parallel Operations ===" << std::endl;

    constexpr std::size_t N = 1'000'000;
    std::vector<float> data(N);

    benchmark("parallel_for 1M elements", 100, [&data, N]() {
        parallel::parallel_for(0, N, [&data](std::size_t i) {
            data[i] = static_cast<float>(i) * 2.0f;
        });
    });

    benchmark("parallel_reduce 1M elements", 100, [N]() {
        auto sum = parallel::parallel_reduce(
            std::size_t{0}, N,
            0.0f,
            [](std::size_t i) { return static_cast<float>(i); },
            [](float a, float b) { return a + b; }
        );
        (void)sum;
    });
}

void bench_simd_operations() {
    std::cout << "\n=== SIMD Operations ===" << std::endl;

    using V = simd::Vec<float>;
    constexpr std::size_t N = 1'000'000;
    alignas(64) std::vector<float> a(N), b(N), c(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : a) v = dist(gen);
    for (auto& v : b) v = dist(gen);

    benchmark("SIMD add 1M elements", 100, [&a, &b, &c, N]() {
        for (std::size_t i = 0; i < N; i += V::width) {
            auto va = V::load(&a[i]);
            auto vb = V::load(&b[i]);
            V::store(&c[i], V::add(va, vb));
        }
    });

    benchmark("SIMD fmadd 1M elements", 100, [&a, &b, &c, N]() {
        auto scale = V::set1(2.0f);
        for (std::size_t i = 0; i < N; i += V::width) {
            auto va = V::load(&a[i]);
            auto vb = V::load(&b[i]);
            V::store(&c[i], V::fmadd(va, scale, vb));
        }
    });
}

int main() {
    std::cout << "Kernelix v" << Version::string << " - Simple Benchmarks" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Threads: " << num_threads() << std::endl;
    std::cout << "SIMD width: " << simd::Vec<float>::width << " floats" << std::endl;

    bench_tensor_creation();
    bench_parallel_operations();
    bench_simd_operations();

    std::cout << "\n=============================================" << std::endl;
    std::cout << "Benchmarks completed!" << std::endl;

    return 0;
}
