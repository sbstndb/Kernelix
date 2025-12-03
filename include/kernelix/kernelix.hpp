#pragma once

/// @file kernelix.hpp
/// @brief Main header for Kernelix - Automatic Tensor Kernel Optimization Library
///
/// Kernelix is a C++ header-only library for automatic tensor kernel optimization
/// targeting CPU inference, primarily designed for LLM workloads.
///
/// @code
/// #include <kernelix/kernelix.hpp>
///
/// using namespace kernelix;
///
/// Tensor<float, 32, 4096> x;
/// Tensor<float, 4096, 4096> W;
/// Tensor<float, 32, 4096> output;
///
/// auto result = linear(x, W);
/// eval(result, output.data());
/// @endcode

// Version information
#define KERNELIX_VERSION_MAJOR 0
#define KERNELIX_VERSION_MINOR 6
#define KERNELIX_VERSION_PATCH 0
#define KERNELIX_VERSION_STRING "0.6.0"

// Core components
#include "core/types.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/tensor.hpp"

// Expression templates
#include "expr/base.hpp"
#include "expr/binary.hpp"
#include "expr/unary.hpp"
#include "expr/contraction.hpp"
#include "expr/norm.hpp"
#include "expr/attention.hpp"
#include "expr/rope.hpp"

// Traits system
#include "traits/expr_traits.hpp"

// Utilities
#include "util/simd.hpp"
#include "util/memory.hpp"
#include "util/parallel.hpp"

// Kernels and evaluator
#include "kernel/gemm/micro_kernel.hpp"
#include "kernel/gemm/pack.hpp"
#include "kernel/gemm/impl.hpp"
#include "kernel/evaluator.hpp"

namespace kernelix {

/// Library version as struct
struct Version {
    static constexpr int major = KERNELIX_VERSION_MAJOR;
    static constexpr int minor = KERNELIX_VERSION_MINOR;
    static constexpr int patch = KERNELIX_VERSION_PATCH;
    static constexpr const char* string = KERNELIX_VERSION_STRING;
};

// ============================================================================
// API Functions
// ============================================================================

/// Create GEMM expression: C = A @ B
template<TensorLike A, TensorLike B>
auto gemm(const A& a, const B& b) {
    return expr::GemmExpr<const A&, const B&>(a, b);
}

/// Create GEMM expression with bias: C = A @ B + bias
template<TensorLike A, TensorLike B, TensorLike Bias>
auto gemm(const A& a, const B& b, const Bias& bias) {
    return expr::GemmExpr<const A&, const B&, const Bias&>(a, b, bias);
}

/// Create GEMV expression: y = A @ x
template<TensorLike A, TensorLike X>
auto gemv(const A& a, const X& x) {
    return expr::GemvExpr<const A&, const X&>(a, x);
}

/// Create dot product expression
template<TensorLike A, TensorLike B>
auto dot(const A& a, const B& b) {
    return expr::DotExpr<const A&, const B&>(a, b);
}

/// Linear transformation: y = x @ W (TensorLike inputs)
template<TensorLike X, TensorLike W>
auto linear(const X& x, const W& w) {
    return gemm(x, w);
}

/// Linear transformation: y = expr @ W (Expression as first input for fusion)
/// Stores expression by value for proper pattern matching in evaluator
template<ExprLike X, TensorLike W>
auto linear(X&& x, const W& w) {
    return expr::GemmExpr<std::remove_cvref_t<X>, const W&, void>(
        std::forward<X>(x), w);
}

/// Linear transformation with bias: y = x @ W + b
template<TensorLike X, TensorLike W, TensorLike B>
auto linear(const X& x, const W& w, const B& bias) {
    return gemm(x, w, bias);
}

/// ReLU activation
template<typename E>
auto relu(E&& expr) {
    return expr::make_relu(std::forward<E>(expr));
}

/// SiLU (Swish) activation
template<typename E>
auto silu(E&& expr) {
    return expr::make_silu(std::forward<E>(expr));
}

/// GELU activation
template<typename E>
auto gelu(E&& expr) {
    return expr::make_gelu(std::forward<E>(expr));
}

/// Tanh activation
template<typename E>
auto tanh_act(E&& expr) {
    return expr::make_tanh(std::forward<E>(expr));
}

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise multiply: c = a * b
template<typename A, typename B>
auto mul(A&& a, B&& b) {
    return expr::make_mul(std::forward<A>(a), std::forward<B>(b));
}

/// Element-wise add: c = a + b
template<typename A, typename B>
auto add(A&& a, B&& b) {
    return expr::make_add(std::forward<A>(a), std::forward<B>(b));
}

/// Element-wise subtract: c = a - b
template<typename A, typename B>
auto sub(A&& a, B&& b) {
    return expr::make_sub(std::forward<A>(a), std::forward<B>(b));
}

/// Element-wise divide: c = a / b
template<typename A, typename B>
auto div(A&& a, B&& b) {
    return expr::make_div(std::forward<A>(a), std::forward<B>(b));
}

/// SwiGLU activation: silu(gate) * up
/// Used in LLaMA/Mistral FFN: output = silu(linear_gate(x)) * linear_up(x)
template<typename Gate, typename Up>
auto swiglu(Gate&& gate, Up&& up) {
    return mul(silu(std::forward<Gate>(gate)), std::forward<Up>(up));
}

// ============================================================================
// Normalizations
// ============================================================================

/// RMSNorm: y = x * weight / sqrt(mean(x^2) + eps)
/// Normalizes over the last dimension
/// @param x Input tensor [batch, hidden_dim] or [seq_len, hidden_dim]
/// @param weight Scale weights [hidden_dim]
/// @param eps Small constant for numerical stability (default: 1e-6)
template<TensorLike X, TensorLike Weight>
auto rmsnorm(const X& x, const Weight& weight, float eps = 1e-6f) {
    return expr::RMSNormExpr<const X&, const Weight&>(x, weight, eps);
}

/// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
/// Normalizes over the last dimension
/// @param x Input tensor [batch, hidden_dim] or [seq_len, hidden_dim]
/// @param gamma Scale parameter [hidden_dim]
/// @param beta Shift parameter [hidden_dim]
/// @param eps Small constant for numerical stability (default: 1e-5)
template<TensorLike X, TensorLike Gamma, TensorLike Beta>
auto layernorm(const X& x, const Gamma& gamma, const Beta& beta, float eps = 1e-5f) {
    return expr::LayerNormExpr<const X&, const Gamma&, const Beta&>(x, gamma, beta, eps);
}

/// Softmax: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
/// Applies softmax over the last dimension (numerically stable)
/// @param x Input tensor [batch, dim] or [dim]
template<TensorLike X>
auto softmax(const X& x) {
    return expr::SoftmaxExpr<const X&>(x);
}

// ============================================================================
// Attention
// ============================================================================

/// Scaled Dot-Product Attention: Output = softmax(Q @ K^T / sqrt(head_dim)) @ V
///
/// Implements the standard attention mechanism used in Transformers.
/// If scale is 0, it will be computed as 1/sqrt(head_dim).
///
/// @param query Query tensor [seq_len_q, head_dim]
/// @param key Key tensor [seq_len_k, head_dim]
/// @param value Value tensor [seq_len_k, head_dim]
/// @param scale Scaling factor (default: 0 = auto-compute 1/sqrt(head_dim))
/// @return Attention output [seq_len_q, head_dim]
template<TensorLike Q, TensorLike K, TensorLike V>
auto attention(const Q& query, const K& key, const V& value, float scale = 0.0f) {
    return expr::make_attention(query, key, value, scale);
}

/// Causal (Masked) Attention: Output = softmax(mask(Q @ K^T / sqrt(head_dim))) @ V
///
/// Implements causal attention where positions can only attend to earlier positions.
/// Used in autoregressive models (GPT-style). Future positions are masked with -inf.
///
/// @param query Query tensor [seq_len_q, head_dim]
/// @param key Key tensor [seq_len_k, head_dim]
/// @param value Value tensor [seq_len_k, head_dim]
/// @param scale Scaling factor (default: 0 = auto-compute 1/sqrt(head_dim))
/// @return Attention output [seq_len_q, head_dim]
template<TensorLike Q, TensorLike K, TensorLike V>
auto causal_attention(const Q& query, const K& key, const V& value, float scale = 0.0f) {
    return expr::make_causal_attention(query, key, value, scale);
}

// ============================================================================
// Rotary Position Embeddings (RoPE)
// ============================================================================

/// Apply Rotary Position Embeddings to input tensor
///
/// RoPE applies rotation based on position for each pair of dimensions:
///   x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
///   x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
///
/// @param input Input tensor [seq_len, head_dim]
/// @param cos_cache Precomputed cosines [max_seq_len, head_dim/2]
/// @param sin_cache Precomputed sines [max_seq_len, head_dim/2]
/// @param position_offset Starting position (for KV cache continuation)
/// @return RoPE-transformed tensor [seq_len, head_dim]
template<TensorLike Input, TensorLike CosSin>
auto rope(const Input& input, const CosSin& cos_cache, const CosSin& sin_cache,
          std::size_t position_offset = 0) {
    return expr::make_rope(input, cos_cache, sin_cache, position_offset);
}

/// Precompute RoPE frequency caches (cos and sin)
///
/// Fills cos_cache and sin_cache with precomputed values for positions 0..max_seq_len-1
/// θ_i = position / (base^(2i/head_dim))
///
/// @param cos_cache Output cos cache [max_seq_len, head_dim/2]
/// @param sin_cache Output sin cache [max_seq_len, head_dim/2]
/// @param max_seq_len Maximum sequence length
/// @param head_dim Head dimension (must be even)
/// @param base Base frequency (default: 10000.0)
template<typename T>
void rope_precompute_freqs(T* cos_cache, T* sin_cache,
                           std::size_t max_seq_len, std::size_t head_dim,
                           T base = T{10000}) {
    kernel::RoPEKernel<T>::precompute_freqs(cos_cache, sin_cache, max_seq_len, head_dim, base);
}

// ============================================================================
// Evaluation
// ============================================================================

/// Evaluate expression and write result to output pointer
template<typename Expr, typename T>
void eval(const Expr& expr, T* output) {
    kernel::Evaluator<std::remove_cvref_t<Expr>>::run(expr, output);
}

/// Evaluate expression and return result in a new tensor (static dimensions)
template<typename Expr>
auto compute(const Expr& expr) {
    using ExprType = std::remove_cvref_t<Expr>;
    using Traits = traits::ExprTraits<ExprType>;
    using T = typename Traits::value_type;

    // Create output tensor with correct shape
    if constexpr (Traits::rank == 2) {
        constexpr auto shape = Traits::output_shape;
        Tensor<T, shape[0], shape[1]> result;
        eval(expr, result.data());
        return result;
    } else if constexpr (Traits::rank == 1) {
        constexpr auto shape = Traits::output_shape;
        Tensor<T, shape[0]> result;
        eval(expr, result.data());
        return result;
    } else if constexpr (Traits::rank == 0) {
        T result;
        eval(expr, &result);
        return result;
    }
}

// ============================================================================
// Expression Info (for debugging/introspection)
// ============================================================================

/// Get compile-time info about an expression
template<typename Expr>
struct ExprInfo {
    using Traits = traits::ExprTraits<std::remove_cvref_t<Expr>>;

    static constexpr std::size_t rank = Traits::rank;
    static constexpr auto output_shape = Traits::output_shape;
    static constexpr std::size_t output_size = Traits::output_size;
    static constexpr bool is_terminal = Traits::is_terminal;

    // GEMM-specific info
    template<typename E = Expr>
    static constexpr std::size_t flops() {
        if constexpr (requires { Traits::estimated_flops; }) {
            return Traits::estimated_flops;
        } else {
            return 0;
        }
    }
};

} // namespace kernelix
