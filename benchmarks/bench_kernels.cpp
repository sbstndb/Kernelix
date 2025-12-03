/// @file bench_kernels.cpp
/// @brief Benchmarks for normalization, softmax, attention, and RoPE kernels

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
// RMSNorm Benchmarks
// =============================================================================

template<std::size_t Batch, std::size_t Dim>
static void BM_RMSNorm(benchmark::State& state) {
    Tensor<float, Batch, Dim> X;
    Tensor<float, Dim> weight;
    Tensor<float, Batch, Dim> Y;

    fill_random(X.data(), Batch * Dim);
    fill_random(weight.data(), Dim);

    for (auto _ : state) {
        eval(rmsnorm(X, weight), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    // Bytes: read X + weight, write Y
    const std::size_t bytes = (Batch * Dim + Dim + Batch * Dim) * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
    state.counters["Batch"] = Batch;
    state.counters["Dim"] = Dim;
}

BENCHMARK(BM_RMSNorm<32, 1024>)->Name("RMSNorm/32x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RMSNorm<128, 1024>)->Name("RMSNorm/128x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RMSNorm<256, 1024>)->Name("RMSNorm/256x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RMSNorm<32, 2048>)->Name("RMSNorm/32x2048")->Unit(benchmark::kMicrosecond);

// =============================================================================
// LayerNorm Benchmarks
// =============================================================================

template<std::size_t Batch, std::size_t Dim>
static void BM_LayerNorm(benchmark::State& state) {
    Tensor<float, Batch, Dim> X;
    Tensor<float, Dim> gamma, beta;
    Tensor<float, Batch, Dim> Y;

    fill_random(X.data(), Batch * Dim);
    fill_random(gamma.data(), Dim);
    fill_random(beta.data(), Dim);

    for (auto _ : state) {
        eval(layernorm(X, gamma, beta), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = (Batch * Dim + 2 * Dim + Batch * Dim) * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

BENCHMARK(BM_LayerNorm<32, 1024>)->Name("LayerNorm/32x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_LayerNorm<128, 1024>)->Name("LayerNorm/128x1024")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_LayerNorm<256, 1024>)->Name("LayerNorm/256x1024")->Unit(benchmark::kMicrosecond);

// =============================================================================
// Fused RMSNorm + Linear Benchmarks
// =============================================================================

template<std::size_t Batch, std::size_t InDim, std::size_t OutDim>
static void BM_RMSNorm_Linear(benchmark::State& state) {
    Tensor<float, Batch, InDim> X;
    Tensor<float, InDim> weight;
    Tensor<float, InDim, OutDim> W;
    Tensor<float, Batch, OutDim> Y;

    fill_random(X.data(), Batch * InDim);
    fill_random(weight.data(), InDim);
    fill_random(W.data(), InDim * OutDim);

    for (auto _ : state) {
        auto normed = rmsnorm(X, weight);
        eval(linear(normed, W), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    // GEMM flops + RMSNorm
    const double flops = 2.0 * Batch * InDim * OutDim + 4 * Batch * InDim;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_RMSNorm_Linear<32, 1024, 1024>)->Name("Fused_RMSNorm+Linear/32")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RMSNorm_Linear<64, 1024, 1024>)->Name("Fused_RMSNorm+Linear/64")->Unit(benchmark::kMillisecond);

// =============================================================================
// Softmax Benchmarks
// =============================================================================

template<std::size_t Batch, std::size_t Dim>
static void BM_Softmax(benchmark::State& state) {
    Tensor<float, Batch, Dim> X;
    Tensor<float, Batch, Dim> Y;

    fill_random(X.data(), Batch * Dim);

    for (auto _ : state) {
        eval(softmax(X), Y.data());
        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 2 * Batch * Dim * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

BENCHMARK(BM_Softmax<32, 32>)->Name("Softmax/32x32")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Softmax<32, 128>)->Name("Softmax/32x128")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Softmax<32, 512>)->Name("Softmax/32x512")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Softmax<32, 2048>)->Name("Softmax/32x2048")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Softmax<128, 128>)->Name("Softmax/128x128")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Softmax<512, 512>)->Name("Softmax/512x512")->Unit(benchmark::kMicrosecond);

// =============================================================================
// Attention Benchmarks
// =============================================================================

template<std::size_t SeqLen, std::size_t HeadDim>
static void BM_Attention(benchmark::State& state) {
    Tensor<float, SeqLen, HeadDim> Q, K, V;
    Tensor<float, SeqLen, HeadDim> output;

    fill_random(Q.data(), SeqLen * HeadDim, 1);
    fill_random(K.data(), SeqLen * HeadDim, 2);
    fill_random(V.data(), SeqLen * HeadDim, 3);

    for (auto _ : state) {
        eval(attention(Q, K, V), output.data());
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    // Attention FLOPS: Q@K^T (2*S*S*D) + softmax (5*S*S) + scores@V (2*S*S*D)
    const double flops = 4.0 * SeqLen * SeqLen * HeadDim + 5.0 * SeqLen * SeqLen;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["SeqLen"] = SeqLen;
    state.counters["HeadDim"] = HeadDim;
}

template<std::size_t SeqLen, std::size_t HeadDim>
static void BM_CausalAttention(benchmark::State& state) {
    Tensor<float, SeqLen, HeadDim> Q, K, V;
    Tensor<float, SeqLen, HeadDim> output;

    fill_random(Q.data(), SeqLen * HeadDim, 1);
    fill_random(K.data(), SeqLen * HeadDim, 2);
    fill_random(V.data(), SeqLen * HeadDim, 3);

    for (auto _ : state) {
        eval(causal_attention(Q, K, V), output.data());
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    // Causal attention has ~half the compute due to mask
    const double flops = 2.0 * SeqLen * SeqLen * HeadDim + 2.5 * SeqLen * SeqLen;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Small sequences (interactive/chat)
BENCHMARK(BM_Attention<32, 64>)->Name("Attention/32x64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Attention<64, 64>)->Name("Attention/64x64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Attention<128, 64>)->Name("Attention/128x64")->Unit(benchmark::kMicrosecond);

// Medium sequences
BENCHMARK(BM_Attention<256, 64>)->Name("Attention/256x64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Attention<512, 64>)->Name("Attention/512x64")->Unit(benchmark::kMillisecond);

// Large sequences
BENCHMARK(BM_Attention<1024, 64>)->Name("Attention/1024x64")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Attention<2048, 64>)->Name("Attention/2048x64")->Unit(benchmark::kMillisecond);

// Causal attention
BENCHMARK(BM_CausalAttention<128, 64>)->Name("CausalAttention/128x64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CausalAttention<256, 64>)->Name("CausalAttention/256x64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CausalAttention<512, 64>)->Name("CausalAttention/512x64")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CausalAttention<1024, 64>)->Name("CausalAttention/1024x64")->Unit(benchmark::kMillisecond);

// Different head dimensions
BENCHMARK(BM_Attention<128, 128>)->Name("Attention/128x128")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Attention<256, 128>)->Name("Attention/256x128")->Unit(benchmark::kMillisecond);

// =============================================================================
// RoPE Benchmarks
// =============================================================================

template<std::size_t SeqLen, std::size_t HeadDim>
static void BM_RoPE(benchmark::State& state) {
    constexpr std::size_t half_dim = HeadDim / 2;

    Tensor<float, SeqLen, HeadDim> input;
    Tensor<float, SeqLen, half_dim> cos_cache, sin_cache;
    Tensor<float, SeqLen, HeadDim> output;

    fill_random(input.data(), SeqLen * HeadDim);
    rope_precompute_freqs(cos_cache.data(), sin_cache.data(), SeqLen, HeadDim);

    for (auto _ : state) {
        eval(rope(input, cos_cache, sin_cache), output.data());
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    // RoPE: 6 ops per pair, HeadDim/2 pairs per position
    const double flops = SeqLen * HeadDim * 3.0;
    const std::size_t bytes = (3 * SeqLen * HeadDim + SeqLen * HeadDim) * sizeof(float);

    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetBytesProcessed(state.iterations() * bytes);
}

template<std::size_t MaxSeqLen, std::size_t HeadDim>
static void BM_RoPE_Precompute(benchmark::State& state) {
    constexpr std::size_t half_dim = HeadDim / 2;
    Tensor<float, MaxSeqLen, half_dim> cos_cache, sin_cache;

    for (auto _ : state) {
        rope_precompute_freqs(cos_cache.data(), sin_cache.data(), MaxSeqLen, HeadDim);
        benchmark::DoNotOptimize(cos_cache.data());
        benchmark::DoNotOptimize(sin_cache.data());
        benchmark::ClobberMemory();
    }

    const std::size_t bytes = 2 * MaxSeqLen * half_dim * sizeof(float);
    state.SetBytesProcessed(state.iterations() * bytes);
}

BENCHMARK(BM_RoPE<32, 64>)->Name("RoPE/32x64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RoPE<128, 64>)->Name("RoPE/128x64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RoPE<512, 64>)->Name("RoPE/512x64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RoPE<2048, 64>)->Name("RoPE/2048x64")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_RoPE<128, 128>)->Name("RoPE/128x128")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RoPE<512, 128>)->Name("RoPE/512x128")->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_RoPE_Precompute<2048, 64>)->Name("RoPE_Precompute/2048x64")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RoPE_Precompute<1024, 128>)->Name("RoPE_Precompute/1024x128")->Unit(benchmark::kMicrosecond);
