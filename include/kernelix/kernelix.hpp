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
#define KERNELIX_VERSION_MINOR 4
#define KERNELIX_VERSION_PATCH 0
#define KERNELIX_VERSION_STRING "0.4.0"

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
