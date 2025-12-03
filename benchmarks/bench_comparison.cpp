/// @file bench_comparison.cpp
/// @brief Comparative benchmarks: fusion vs separate, parallel vs sequential, tiled vs naive

#include <kernelix/kernelix.hpp>
#include <kernelix/kernel/gemm/fused.hpp>
#include <kernelix/kernel/gemm/chained.hpp>
#include <benchmark/benchmark.h>
#include <random>
#include <cmath>

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

// =============================================================================
// FUSION COMPARISON: GEMM + Activation
// =============================================================================

/// Fused GEMM + ReLU (single kernel)
template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_ReLU_Fused(benchmark::State& state) {
    Tensor<float, M, K> A;
    Tensor<float, K, N> B;
    Tensor<float, M, N> C;

    fill_random(A.data(), M * K);
    fill_random(B.data(), K * N);

    for (auto _ : state) {
        // Fused: single eval call, ReLU applied during GEMM output
        eval(relu(gemm(A, B)), C.data());
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * N * K + M * N;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

/// Separate GEMM then ReLU (two passes over memory)
template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_ReLU_Separate(benchmark::State& state) {
    Tensor<float, M, K> A;
    Tensor<float, K, N> B;
    Tensor<float, M, N> C, D;

    fill_random(A.data(), M * K);
    fill_random(B.data(), K * N);

    for (auto _ : state) {
        // Separate: GEMM first
        eval(gemm(A, B), C.data());
        // Then ReLU (second pass over memory)
        eval(relu(C), D.data());
        benchmark::DoNotOptimize(D.data());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * N * K + M * N;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_GEMM_ReLU_Fused<256, 256, 256>)->Name("Fusion/GEMM+ReLU_Fused/256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_ReLU_Separate<256, 256, 256>)->Name("Fusion/GEMM+ReLU_Separate/256")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_GEMM_ReLU_Fused<512, 512, 512>)->Name("Fusion/GEMM+ReLU_Fused/512")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GEMM_ReLU_Separate<512, 512, 512>)->Name("Fusion/GEMM+ReLU_Separate/512")->Unit(benchmark::kMillisecond);

// Note: 1024x1024 skipped - requires heap allocation or increased stack size

// =============================================================================
// KERNEL-LEVEL FUSION COMPARISON: Direct kernel calls
// =============================================================================

/// Truly fused GEMM+ReLU (activation during tile write)
template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_ReLU_KernelFused(benchmark::State& state) {
    auto A = memory::make_aligned<float>(M * K);
    auto B = memory::make_aligned<float>(K * N);
    auto C = memory::make_aligned<float>(M * N);

    fill_random(A.get(), M * K);
    fill_random(B.get(), K * N);

    for (auto _ : state) {
        // Truly fused: activation applied during final tile write
        kernel::gemm_relu_impl(A.get(), B.get(), C.get(), M, N, K, K, N, N);
        benchmark::DoNotOptimize(C.get());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * N * K + M * N;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

/// GEMM + separate ReLU pass
template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_ReLU_KernelSeparate(benchmark::State& state) {
    auto A = memory::make_aligned<float>(M * K);
    auto B = memory::make_aligned<float>(K * N);
    auto C = memory::make_aligned<float>(M * N);

    fill_random(A.get(), M * K);
    fill_random(B.get(), K * N);

    for (auto _ : state) {
        // Separate: GEMM first
        kernel::gemm_impl(A.get(), B.get(), C.get(), M, N, K, K, N, N, false);
        // Then separate ReLU pass (extra memory traffic)
        for (std::size_t i = 0; i < M * N; ++i) {
            C[i] = std::max(0.0f, C[i]);
        }
        benchmark::DoNotOptimize(C.get());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * N * K + M * N;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

/// Truly fused GEMM+SiLU (activation during tile write)
template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_SiLU_KernelFused(benchmark::State& state) {
    auto A = memory::make_aligned<float>(M * K);
    auto B = memory::make_aligned<float>(K * N);
    auto C = memory::make_aligned<float>(M * N);

    fill_random(A.get(), M * K);
    fill_random(B.get(), K * N);

    for (auto _ : state) {
        kernel::gemm_silu_impl(A.get(), B.get(), C.get(), M, N, K, K, N, N);
        benchmark::DoNotOptimize(C.get());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * N * K + 4 * M * N;  // SiLU is more expensive
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

/// GEMM + separate SiLU pass
template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_SiLU_KernelSeparate(benchmark::State& state) {
    auto A = memory::make_aligned<float>(M * K);
    auto B = memory::make_aligned<float>(K * N);
    auto C = memory::make_aligned<float>(M * N);

    fill_random(A.get(), M * K);
    fill_random(B.get(), K * N);

    for (auto _ : state) {
        kernel::gemm_impl(A.get(), B.get(), C.get(), M, N, K, K, N, N, false);
        for (std::size_t i = 0; i < M * N; ++i) {
            float x = C[i];
            C[i] = x / (1.0f + std::exp(-x));  // SiLU
        }
        benchmark::DoNotOptimize(C.get());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * N * K + 4 * M * N;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Kernel-level fusion benchmarks (using heap allocation for larger sizes)
BENCHMARK(BM_GEMM_ReLU_KernelFused<256, 256, 256>)->Name("KernelFusion/GEMM+ReLU_Fused/256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_ReLU_KernelSeparate<256, 256, 256>)->Name("KernelFusion/GEMM+ReLU_Separate/256")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_GEMM_ReLU_KernelFused<512, 512, 512>)->Name("KernelFusion/GEMM+ReLU_Fused/512")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GEMM_ReLU_KernelSeparate<512, 512, 512>)->Name("KernelFusion/GEMM+ReLU_Separate/512")->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GEMM_ReLU_KernelFused<1024, 1024, 1024>)->Name("KernelFusion/GEMM+ReLU_Fused/1024")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GEMM_ReLU_KernelSeparate<1024, 1024, 1024>)->Name("KernelFusion/GEMM+ReLU_Separate/1024")->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GEMM_SiLU_KernelFused<512, 512, 512>)->Name("KernelFusion/GEMM+SiLU_Fused/512")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GEMM_SiLU_KernelSeparate<512, 512, 512>)->Name("KernelFusion/GEMM+SiLU_Separate/512")->Unit(benchmark::kMillisecond);

// =============================================================================
// FUSION COMPARISON: RMSNorm + Linear
// =============================================================================

/// Fused RMSNorm + Linear
template<std::size_t Batch, std::size_t Dim, std::size_t OutDim>
static void BM_RMSNorm_Linear_Fused(benchmark::State& state) {
    Tensor<float, Batch, Dim> X;
    Tensor<float, Dim> weight;
    Tensor<float, Dim, OutDim> W;
    Tensor<float, Batch, OutDim> Y;

    fill_random(X.data(), Batch * Dim);
    fill_random(weight.data(), Dim);
    fill_random(W.data(), Dim * OutDim);

    for (auto _ : state) {
        // Fused expression
        eval(linear(rmsnorm(X, weight), W), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    const double flops = 4 * Batch * Dim + 2.0 * Batch * Dim * OutDim;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

/// Separate RMSNorm then Linear
template<std::size_t Batch, std::size_t Dim, std::size_t OutDim>
static void BM_RMSNorm_Linear_Separate(benchmark::State& state) {
    Tensor<float, Batch, Dim> X;
    Tensor<float, Dim> weight;
    Tensor<float, Dim, OutDim> W;
    Tensor<float, Batch, Dim> normed;
    Tensor<float, Batch, OutDim> Y;

    fill_random(X.data(), Batch * Dim);
    fill_random(weight.data(), Dim);
    fill_random(W.data(), Dim * OutDim);

    for (auto _ : state) {
        // Separate: RMSNorm first
        eval(rmsnorm(X, weight), normed.data());
        // Then Linear
        eval(linear(normed, W), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    const double flops = 4 * Batch * Dim + 2.0 * Batch * Dim * OutDim;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Smaller sizes to avoid stack overflow with static tensors
BENCHMARK(BM_RMSNorm_Linear_Fused<32, 512, 512>)->Name("Fusion/RMSNorm+Linear_Fused/32x512")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RMSNorm_Linear_Separate<32, 512, 512>)->Name("Fusion/RMSNorm+Linear_Separate/32x512")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_RMSNorm_Linear_Fused<64, 1024, 1024>)->Name("Fusion/RMSNorm+Linear_Fused/64x1024")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RMSNorm_Linear_Separate<64, 1024, 1024>)->Name("Fusion/RMSNorm+Linear_Separate/64x1024")->Unit(benchmark::kMillisecond);

// =============================================================================
// PARALLELISM COMPARISON: TBB vs Sequential
// =============================================================================

/// Parallel element-wise operation (uses TBB if available)
template<std::size_t N>
static void BM_Elementwise_Parallel(benchmark::State& state) {
    auto a = memory::make_aligned<float>(N);
    auto b = memory::make_aligned<float>(N);
    auto c = memory::make_aligned<float>(N);

    fill_random(a.get(), N, 1);
    fill_random(b.get(), N, 2);

    for (auto _ : state) {
        parallel::parallel_for(std::size_t{0}, N, [&](std::size_t i) {
            c[i] = a[i] * b[i] + a[i];
        });
        benchmark::DoNotOptimize(c.get());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 3 * N * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

/// Sequential element-wise operation (explicit sequential loop)
template<std::size_t N>
static void BM_Elementwise_Sequential(benchmark::State& state) {
    auto a = memory::make_aligned<float>(N);
    auto b = memory::make_aligned<float>(N);
    auto c = memory::make_aligned<float>(N);

    fill_random(a.get(), N, 1);
    fill_random(b.get(), N, 2);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            c[i] = a[i] * b[i] + a[i];
        }
        benchmark::DoNotOptimize(c.get());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 3 * N * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

BENCHMARK(BM_Elementwise_Parallel<1024>)->Name("Parallel/Elementwise_TBB/1K")->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Elementwise_Sequential<1024>)->Name("Parallel/Elementwise_Seq/1K")->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Elementwise_Parallel<65536>)->Name("Parallel/Elementwise_TBB/64K")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Elementwise_Sequential<65536>)->Name("Parallel/Elementwise_Seq/64K")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Elementwise_Parallel<1048576>)->Name("Parallel/Elementwise_TBB/1M")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Elementwise_Sequential<1048576>)->Name("Parallel/Elementwise_Seq/1M")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Elementwise_Parallel<16777216>)->Name("Parallel/Elementwise_TBB/16M")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Elementwise_Sequential<16777216>)->Name("Parallel/Elementwise_Seq/16M")->Unit(benchmark::kMillisecond);

// =============================================================================
// PARALLELISM COMPARISON: Parallel Reduce
// =============================================================================

template<std::size_t N>
static void BM_Reduce_Parallel(benchmark::State& state) {
    auto data = memory::make_aligned<float>(N);
    fill_random(data.get(), N);

    for (auto _ : state) {
        float sum = parallel::parallel_reduce(
            std::size_t{0}, N,
            0.0f,
            [&data](std::size_t i) { return data[i]; },
            [](float a, float b) { return a + b; }
        );
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N * sizeof(float));
}

template<std::size_t N>
static void BM_Reduce_Sequential(benchmark::State& state) {
    auto data = memory::make_aligned<float>(N);
    fill_random(data.get(), N);

    for (auto _ : state) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < N; ++i) {
            sum += data[i];
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N * sizeof(float));
}

BENCHMARK(BM_Reduce_Parallel<65536>)->Name("Parallel/Reduce_TBB/64K")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Reduce_Sequential<65536>)->Name("Parallel/Reduce_Seq/64K")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Reduce_Parallel<1048576>)->Name("Parallel/Reduce_TBB/1M")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Reduce_Sequential<1048576>)->Name("Parallel/Reduce_Seq/1M")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Reduce_Parallel<16777216>)->Name("Parallel/Reduce_TBB/16M")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Reduce_Sequential<16777216>)->Name("Parallel/Reduce_Seq/16M")->Unit(benchmark::kMillisecond);

// =============================================================================
// TILING COMPARISON: Tiled GEMM vs Naive GEMM
// =============================================================================

/// Naive GEMM implementation (no tiling, row-major)
template<typename T>
void naive_gemm(const T* A, const T* B, T* C, std::size_t M, std::size_t K, std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (std::size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/// Naive GEMM with loop reordering (better cache behavior, still no tiling)
template<typename T>
void naive_gemm_ikj(const T* A, const T* B, T* C, std::size_t M, std::size_t K, std::size_t N) {
    std::fill(C, C + M * N, T{0});
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            T a_ik = A[i * K + k];
            for (std::size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_Tiled(benchmark::State& state) {
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

    const double flops = 2.0 * M * N * K;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_Naive(benchmark::State& state) {
    auto A = memory::make_aligned<float>(M * K);
    auto B = memory::make_aligned<float>(K * N);
    auto C = memory::make_aligned<float>(M * N);

    fill_random(A.get(), M * K);
    fill_random(B.get(), K * N);

    for (auto _ : state) {
        naive_gemm(A.get(), B.get(), C.get(), M, K, N);
        benchmark::DoNotOptimize(C.get());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * N * K;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

template<std::size_t M, std::size_t K, std::size_t N>
static void BM_GEMM_Naive_IKJ(benchmark::State& state) {
    auto A = memory::make_aligned<float>(M * K);
    auto B = memory::make_aligned<float>(K * N);
    auto C = memory::make_aligned<float>(M * N);

    fill_random(A.get(), M * K);
    fill_random(B.get(), K * N);

    for (auto _ : state) {
        naive_gemm_ikj(A.get(), B.get(), C.get(), M, K, N);
        benchmark::DoNotOptimize(C.get());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * N * K;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Small: fits in L1 cache - tiling overhead may hurt
BENCHMARK(BM_GEMM_Tiled<64, 64, 64>)->Name("Tiling/GEMM_Tiled/64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_Naive<64, 64, 64>)->Name("Tiling/GEMM_Naive_IJK/64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_Naive_IKJ<64, 64, 64>)->Name("Tiling/GEMM_Naive_IKJ/64")->Unit(benchmark::kMicrosecond);

// Medium: L2/L3 - tiling starts to help
BENCHMARK(BM_GEMM_Tiled<256, 256, 256>)->Name("Tiling/GEMM_Tiled/256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_Naive<256, 256, 256>)->Name("Tiling/GEMM_Naive_IJK/256")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GEMM_Naive_IKJ<256, 256, 256>)->Name("Tiling/GEMM_Naive_IKJ/256")->Unit(benchmark::kMillisecond);

// Large: memory-bound - tiling essential
BENCHMARK(BM_GEMM_Tiled<512, 512, 512>)->Name("Tiling/GEMM_Tiled/512")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GEMM_Naive_IKJ<512, 512, 512>)->Name("Tiling/GEMM_Naive_IKJ/512")->Unit(benchmark::kMillisecond);
// Skip Naive IJK for 512+ as it's extremely slow

// Note: 1024x1024 skipped - static tensors exceed stack size limit

// =============================================================================
// SIMD COMPARISON: SIMD vs Scalar
// =============================================================================

template<std::size_t N>
static void BM_VectorAdd_SIMD(benchmark::State& state) {
    using V = simd::Vec<float>;
    auto a = memory::make_aligned<float>(N);
    auto b = memory::make_aligned<float>(N);
    auto c = memory::make_aligned<float>(N);

    fill_random(a.get(), N, 1);
    fill_random(b.get(), N, 2);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; i += V::width) {
            auto va = V::load(a.get() + i);
            auto vb = V::load(b.get() + i);
            V::store(c.get() + i, V::add(va, vb));
        }
        benchmark::DoNotOptimize(c.get());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * 3 * N * sizeof(float));
}

template<std::size_t N>
static void BM_VectorAdd_Scalar(benchmark::State& state) {
    auto a = memory::make_aligned<float>(N);
    auto b = memory::make_aligned<float>(N);
    auto c = memory::make_aligned<float>(N);

    fill_random(a.get(), N, 1);
    fill_random(b.get(), N, 2);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            c[i] = a[i] + b[i];
        }
        benchmark::DoNotOptimize(c.get());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * 3 * N * sizeof(float));
}

BENCHMARK(BM_VectorAdd_SIMD<65536>)->Name("SIMD/Add_SIMD/64K")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_VectorAdd_Scalar<65536>)->Name("SIMD/Add_Scalar/64K")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_VectorAdd_SIMD<1048576>)->Name("SIMD/Add_SIMD/1M")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_VectorAdd_Scalar<1048576>)->Name("SIMD/Add_Scalar/1M")->Unit(benchmark::kMicrosecond);

template<std::size_t N>
static void BM_FMA_SIMD(benchmark::State& state) {
    using V = simd::Vec<float>;
    auto a = memory::make_aligned<float>(N);
    auto b = memory::make_aligned<float>(N);
    auto c = memory::make_aligned<float>(N);

    fill_random(a.get(), N, 1);
    fill_random(b.get(), N, 2);

    auto scale = V::set1(2.5f);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; i += V::width) {
            auto va = V::load(a.get() + i);
            auto vb = V::load(b.get() + i);
            V::store(c.get() + i, V::fmadd(va, scale, vb));
        }
        benchmark::DoNotOptimize(c.get());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * 3 * N * sizeof(float));
}

template<std::size_t N>
static void BM_FMA_Scalar(benchmark::State& state) {
    auto a = memory::make_aligned<float>(N);
    auto b = memory::make_aligned<float>(N);
    auto c = memory::make_aligned<float>(N);

    fill_random(a.get(), N, 1);
    fill_random(b.get(), N, 2);

    float scale = 2.5f;

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            c[i] = a[i] * scale + b[i];
        }
        benchmark::DoNotOptimize(c.get());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * 3 * N * sizeof(float));
}

BENCHMARK(BM_FMA_SIMD<1048576>)->Name("SIMD/FMA_SIMD/1M")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FMA_Scalar<1048576>)->Name("SIMD/FMA_Scalar/1M")->Unit(benchmark::kMicrosecond);

// =============================================================================
// CHAINED GEMM: Result = A @ B @ C
// =============================================================================

/// Naive chained GEMM (full intermediate materialization)
template<std::size_t M, std::size_t K1, std::size_t K2, std::size_t N>
static void BM_ChainedGEMM_Naive(benchmark::State& state) {
    auto A = memory::make_aligned<float>(M * K1);
    auto B = memory::make_aligned<float>(K1 * K2);
    auto C = memory::make_aligned<float>(K2 * N);
    auto Result = memory::make_aligned<float>(M * N);

    fill_random(A.get(), M * K1, 1);
    fill_random(B.get(), K1 * K2, 2);
    fill_random(C.get(), K2 * N, 3);

    for (auto _ : state) {
        kernel::gemm_chain_naive(
            A.get(), B.get(), C.get(), Result.get(),
            M, K1, K2, N,
            K1, K2, N, N
        );
        benchmark::DoNotOptimize(Result.get());
        benchmark::ClobberMemory();
    }

    // FLOPs: M*K1*K2 + M*K2*N multiplications, same for additions
    const double flops = 2.0 * M * K1 * K2 + 2.0 * M * K2 * N;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    // Memory for intermediate: M * K2 * sizeof(float)
    state.counters["IntermediateMB"] = (M * K2 * sizeof(float)) / (1024.0 * 1024.0);
}

/// Fused chained GEMM (panel-based, no full intermediate)
template<std::size_t M, std::size_t K1, std::size_t K2, std::size_t N>
static void BM_ChainedGEMM_Fused(benchmark::State& state) {
    auto A = memory::make_aligned<float>(M * K1);
    auto B = memory::make_aligned<float>(K1 * K2);
    auto C = memory::make_aligned<float>(K2 * N);
    auto Result = memory::make_aligned<float>(M * N);

    fill_random(A.get(), M * K1, 1);
    fill_random(B.get(), K1 * K2, 2);
    fill_random(C.get(), K2 * N, 3);

    for (auto _ : state) {
        kernel::gemm_chain_impl(
            A.get(), B.get(), C.get(), Result.get(),
            M, K1, K2, N,
            K1, K2, N, N
        );
        benchmark::DoNotOptimize(Result.get());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * K1 * K2 + 2.0 * M * K2 * N;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    // Buffer size: MC * KC2 (fits in L2)
    state.counters["BufferKB"] = (128 * 256 * sizeof(float)) / 1024.0;
}

/// Auto-associating chained GEMM
template<std::size_t M, std::size_t K1, std::size_t K2, std::size_t N>
static void BM_ChainedGEMM_Auto(benchmark::State& state) {
    auto A = memory::make_aligned<float>(M * K1);
    auto B = memory::make_aligned<float>(K1 * K2);
    auto C = memory::make_aligned<float>(K2 * N);
    auto Result = memory::make_aligned<float>(M * N);

    fill_random(A.get(), M * K1, 1);
    fill_random(B.get(), K1 * K2, 2);
    fill_random(C.get(), K2 * N, 3);

    for (auto _ : state) {
        kernel::gemm_chain_auto_impl(
            A.get(), B.get(), C.get(), Result.get(),
            M, K1, K2, N,
            K1, K2, N, N
        );
        benchmark::DoNotOptimize(Result.get());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * M * K1 * K2 + 2.0 * M * K2 * N;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Small square matrices (256x256)
BENCHMARK(BM_ChainedGEMM_Naive<256, 256, 256, 256>)->Name("ChainedGEMM/Naive/256")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ChainedGEMM_Fused<256, 256, 256, 256>)->Name("ChainedGEMM/Fused/256")->Unit(benchmark::kMillisecond);

// Medium square matrices (512x512)
BENCHMARK(BM_ChainedGEMM_Naive<512, 512, 512, 512>)->Name("ChainedGEMM/Naive/512")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ChainedGEMM_Fused<512, 512, 512, 512>)->Name("ChainedGEMM/Fused/512")->Unit(benchmark::kMillisecond);

// MLP-like: expand then project (batch=256, hidden=256, intermediate=1024)
// Intermediate AB: 256 × 1024 = 1 MB
BENCHMARK(BM_ChainedGEMM_Naive<256, 256, 1024, 256>)->Name("ChainedGEMM/Naive/MLP_256_1024")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ChainedGEMM_Fused<256, 256, 1024, 256>)->Name("ChainedGEMM/Fused/MLP_256_1024")->Unit(benchmark::kMillisecond);

// Larger MLP-like: batch=512, hidden=512, intermediate=2048
// Intermediate AB: 512 × 2048 = 4 MB
BENCHMARK(BM_ChainedGEMM_Naive<512, 512, 2048, 512>)->Name("ChainedGEMM/Naive/MLP_512_2048")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ChainedGEMM_Fused<512, 512, 2048, 512>)->Name("ChainedGEMM/Fused/MLP_512_2048")->Unit(benchmark::kMillisecond);

// LLaMA-like dimensions (smaller scale for benchmark): batch=1024, hidden=512, intermediate=2048
// Intermediate AB: 1024 × 2048 = 8 MB
BENCHMARK(BM_ChainedGEMM_Naive<1024, 512, 2048, 512>)->Name("ChainedGEMM/Naive/LLaMA_1024")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ChainedGEMM_Fused<1024, 512, 2048, 512>)->Name("ChainedGEMM/Fused/LLaMA_1024")->Unit(benchmark::kMillisecond);

// Test auto-association on asymmetric case
// Left: M*K2 = 1024*2048 = 8 MB, Right: K1*N = 512*512 = 1 MB → should pick right
BENCHMARK(BM_ChainedGEMM_Auto<1024, 512, 2048, 512>)->Name("ChainedGEMM/Auto/LLaMA_1024")->Unit(benchmark::kMillisecond);
