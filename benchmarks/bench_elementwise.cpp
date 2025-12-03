/// @file bench_elementwise.cpp
/// @brief Benchmarks for element-wise operations and activations

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

// =============================================================================
// Element-wise Binary Operations
// =============================================================================

template<std::size_t Rows, std::size_t Cols>
static void BM_Mul(benchmark::State& state) {
    Tensor<float, Rows, Cols> A, B, C;
    fill_random(A.data(), Rows * Cols, 1);
    fill_random(B.data(), Rows * Cols, 2);

    for (auto _ : state) {
        eval(mul(A, B), C.data());
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 3 * Rows * Cols * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

template<std::size_t Rows, std::size_t Cols>
static void BM_Add(benchmark::State& state) {
    Tensor<float, Rows, Cols> A, B, C;
    fill_random(A.data(), Rows * Cols, 1);
    fill_random(B.data(), Rows * Cols, 2);

    for (auto _ : state) {
        eval(add(A, B), C.data());
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 3 * Rows * Cols * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

BENCHMARK(BM_Mul<256, 256>)->Name("Mul/256x256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Mul<1024, 1024>)->Name("Mul/1024x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Mul<32, 1024>)->Name("Mul/32x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Mul<128, 1024>)->Name("Mul/128x1024")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Add<256, 256>)->Name("Add/256x256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Add<1024, 1024>)->Name("Add/1024x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Add<128, 1024>)->Name("Add/128x1024")->Unit(benchmark::kMicrosecond);

// =============================================================================
// Activation Functions
// =============================================================================

template<std::size_t Rows, std::size_t Cols>
static void BM_ReLU(benchmark::State& state) {
    Tensor<float, Rows, Cols> X, Y;
    fill_random(X.data(), Rows * Cols);

    for (auto _ : state) {
        eval(relu(X), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 2 * Rows * Cols * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

template<std::size_t Rows, std::size_t Cols>
static void BM_SiLU(benchmark::State& state) {
    Tensor<float, Rows, Cols> X, Y;
    fill_random(X.data(), Rows * Cols);

    for (auto _ : state) {
        eval(silu(X), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 2 * Rows * Cols * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

template<std::size_t Rows, std::size_t Cols>
static void BM_GELU(benchmark::State& state) {
    Tensor<float, Rows, Cols> X, Y;
    fill_random(X.data(), Rows * Cols);

    for (auto _ : state) {
        eval(gelu(X), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 2 * Rows * Cols * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

BENCHMARK(BM_ReLU<256, 256>)->Name("ReLU/256x256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ReLU<1024, 1024>)->Name("ReLU/1024x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ReLU<128, 1024>)->Name("ReLU/128x1024")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SiLU<256, 256>)->Name("SiLU/256x256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SiLU<1024, 1024>)->Name("SiLU/1024x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SiLU<128, 1024>)->Name("SiLU/128x1024")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_GELU<256, 256>)->Name("GELU/256x256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GELU<1024, 1024>)->Name("GELU/1024x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GELU<128, 1024>)->Name("GELU/128x1024")->Unit(benchmark::kMicrosecond);

// =============================================================================
// SwiGLU Activation
// =============================================================================

template<std::size_t Batch, std::size_t Dim>
static void BM_SwiGLU(benchmark::State& state) {
    Tensor<float, Batch, Dim> gate, up, output;
    fill_random(gate.data(), Batch * Dim, 1);
    fill_random(up.data(), Batch * Dim, 2);

    for (auto _ : state) {
        eval(swiglu(gate, up), output.data());
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    // SwiGLU: read gate, up; compute silu(gate); multiply; write output
    const std::size_t bytes = 3 * Batch * Dim * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

BENCHMARK(BM_SwiGLU<32, 1024>)->Name("SwiGLU/32x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SwiGLU<128, 1024>)->Name("SwiGLU/128x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SwiGLU<64, 512>)->Name("SwiGLU/64x512")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SwiGLU<256, 512>)->Name("SwiGLU/256x512")->Unit(benchmark::kMicrosecond);

// =============================================================================
// Chained Operations (Expression Evaluation)
// =============================================================================

template<std::size_t Rows, std::size_t Cols>
static void BM_ChainedOps(benchmark::State& state) {
    Tensor<float, Rows, Cols> A, B, C, D;
    fill_random(A.data(), Rows * Cols, 1);
    fill_random(B.data(), Rows * Cols, 2);
    fill_random(C.data(), Rows * Cols, 3);

    for (auto _ : state) {
        // D = (A + B) * C
        eval(mul(add(A, B), C), D.data());
        benchmark::DoNotOptimize(D.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 4 * Rows * Cols * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

template<std::size_t Rows, std::size_t Cols>
static void BM_ActivationChain(benchmark::State& state) {
    Tensor<float, Rows, Cols> A, B, output;
    fill_random(A.data(), Rows * Cols, 1);
    fill_random(B.data(), Rows * Cols, 2);

    for (auto _ : state) {
        // output = relu(A + B)
        eval(relu(add(A, B)), output.data());
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 3 * Rows * Cols * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

BENCHMARK(BM_ChainedOps<256, 256>)->Name("ChainedOps/(A+B)*C/256x256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChainedOps<1024, 1024>)->Name("ChainedOps/(A+B)*C/1024x1024")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_ActivationChain<256, 256>)->Name("ActivationChain/relu(A+B)/256x256")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ActivationChain<1024, 1024>)->Name("ActivationChain/relu(A+B)/1024x1024")->Unit(benchmark::kMicrosecond);

// =============================================================================
// Memory Bandwidth Test
// =============================================================================

template<std::size_t N>
static void BM_MemoryBandwidth(benchmark::State& state) {
    Tensor<float, N> A, B;
    fill_random(A.data(), N, 1);

    for (auto _ : state) {
        // Simple copy through add(A, 0)
        eval(add(A, A), B.data());
        benchmark::DoNotOptimize(B.data());
        benchmark::ClobberMemory();
    }

    // Read A twice, write B
    const std::size_t bytes = 3 * N * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

BENCHMARK(BM_MemoryBandwidth<1024>)->Name("MemBW/4KB")->Unit(benchmark::kNanosecond);
BENCHMARK(BM_MemoryBandwidth<8192>)->Name("MemBW/32KB")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MemoryBandwidth<65536>)->Name("MemBW/256KB")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MemoryBandwidth<524288>)->Name("MemBW/2MB")->Unit(benchmark::kMicrosecond);
// 16MB skipped - static tensors exceed stack size limit
