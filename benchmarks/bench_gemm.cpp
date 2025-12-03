/// @file bench_gemm.cpp
/// @brief GEMM benchmarks with GFLOPS reporting

#include <kernelix/kernelix.hpp>
#include <benchmark/benchmark.h>
#include <random>

using namespace kernelix;

// =============================================================================
// Benchmark Utilities
// =============================================================================

template<typename T>
void fill_random(T* data, std::size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

// GEMM FLOPS: 2 * M * N * K (multiply-add per element)
constexpr double gemm_flops(std::size_t M, std::size_t N, std::size_t K) {
    return 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
}

// =============================================================================
// GEMM Benchmarks - Various Sizes
// =============================================================================

template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM(benchmark::State& state) {
    Tensor<float, M, K> A;
    Tensor<float, K, N> B;
    Tensor<float, M, N> C;

    fill_random(A.data(), M * K);
    fill_random(B.data(), K * N);

    for (auto _ : state) {
        eval(gemm(A, B), C.data());
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    // Report GFLOPS
    const double flops = gemm_flops(M, N, K);
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["M"] = M;
    state.counters["N"] = N;
    state.counters["K"] = K;
}

// Small matrices (L1 cache resident)
BENCHMARK(BM_GEMM<32, 32, 32>)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM<64, 64, 64>)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM<128, 128, 128>)->Unit(benchmark::kMicrosecond);

// Medium matrices (L2 cache)
BENCHMARK(BM_GEMM<256, 256, 256>)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM<512, 512, 512>)->Unit(benchmark::kMillisecond);

// Large matrices (L3 cache / memory bound)
BENCHMARK(BM_GEMM<1024, 1024, 1024>)->Unit(benchmark::kMillisecond);
// 2048x2048 skipped - static tensors exceed stack size limit

// LLM-style shapes (reduced sizes for stack allocation safety)
BENCHMARK(BM_GEMM<32, 1024, 1024>)->Name("GEMM/LLM_style_32")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GEMM<64, 1024, 1024>)->Name("GEMM/LLM_style_64")->Unit(benchmark::kMillisecond);
// Full LLM sizes (4096, 11008) skipped - static tensors exceed stack size

// =============================================================================
// GEMM + Activation Fusion Benchmarks
// =============================================================================

template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_ReLU(benchmark::State& state) {
    Tensor<float, M, K> A;
    Tensor<float, K, N> B;
    Tensor<float, M, N> C;

    fill_random(A.data(), M * K);
    fill_random(B.data(), K * N);

    for (auto _ : state) {
        eval(relu(gemm(A, B)), C.data());
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const double flops = gemm_flops(M, N, K) + M * N;  // GEMM + ReLU
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_SiLU(benchmark::State& state) {
    Tensor<float, M, K> A;
    Tensor<float, K, N> B;
    Tensor<float, M, N> C;

    fill_random(A.data(), M * K);
    fill_random(B.data(), K * N);

    for (auto _ : state) {
        eval(silu(gemm(A, B)), C.data());
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const double flops = gemm_flops(M, N, K) + 4 * M * N;  // GEMM + SiLU (approx)
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_GEMM_ReLU<256, 256, 256>)->Name("GEMM+ReLU/256x256x256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_SiLU<256, 256, 256>)->Name("GEMM+SiLU/256x256x256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_ReLU<512, 512, 512>)->Name("GEMM+ReLU/512x512x512")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GEMM_SiLU<512, 512, 512>)->Name("GEMM+SiLU/512x512x512")->Unit(benchmark::kMillisecond);

// =============================================================================
// Linear Layer Benchmarks
// =============================================================================

template<std::size_t Batch, std::size_t InDim, std::size_t OutDim>
static void BM_Linear(benchmark::State& state) {
    Tensor<float, Batch, InDim> X;
    Tensor<float, InDim, OutDim> W;
    Tensor<float, Batch, OutDim> Y;

    fill_random(X.data(), Batch * InDim);
    fill_random(W.data(), InDim * OutDim);

    for (auto _ : state) {
        eval(linear(X, W), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    const double flops = gemm_flops(Batch, OutDim, InDim);
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_Linear<1, 512, 512>)->Name("Linear/Batch1_512x512")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Linear<32, 512, 512>)->Name("Linear/Batch32_512x512")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Linear<128, 512, 512>)->Name("Linear/Batch128_512x512")->Unit(benchmark::kMillisecond);

// =============================================================================
// GEMV Benchmarks
// =============================================================================

template<std::size_t M, std::size_t N>
static void BM_GEMV(benchmark::State& state) {
    Tensor<float, M, N> A;
    Tensor<float, N> x;
    Tensor<float, M> y;

    fill_random(A.data(), M * N);
    fill_random(x.data(), N);

    for (auto _ : state) {
        eval(gemv(A, x), y.data());
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * N;  // GEMV: 2*M*N
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_GEMV<512, 512>)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMV<1024, 1024>)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMV<512, 1024>)->Name("GEMV/LLM_style")->Unit(benchmark::kMicrosecond);

// =============================================================================
// Dot Product Benchmarks
// =============================================================================

template<std::size_t N>
static void BM_Dot(benchmark::State& state) {
    Tensor<float, N> a, b;
    float result;

    fill_random(a.data(), N);
    fill_random(b.data(), N);

    for (auto _ : state) {
        eval(dot(a, b), &result);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * N;  // N muls + N adds
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_Dot<1024>)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Dot<4096>)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Dot<16384>)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Dot<65536>)->Unit(benchmark::kMicrosecond);
