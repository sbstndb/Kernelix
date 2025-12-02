#pragma once

#include "../traits/expr_traits.hpp"
#include "../expr/contraction.hpp"
#include "../expr/unary.hpp"
#include "../expr/binary.hpp"
#include "../expr/norm.hpp"
#include "gemm/impl.hpp"
#include "norm/rmsnorm.hpp"
#include "norm/layernorm.hpp"
#include "../util/parallel.hpp"
#include <cmath>
#include <type_traits>

namespace kernelix::kernel {

/// Primary evaluator template - dispatches based on expression type
template<typename Expr>
struct Evaluator;

// ============================================================================
// Terminal evaluator (just copy from tensor)
// ============================================================================

template<typename T, std::size_t... Dims>
struct Evaluator<Tensor<T, Dims...>> {
    static void run(const Tensor<T, Dims...>& tensor, T* output) {
        const std::size_t n = tensor.size();
        const T* src = tensor.data();
        parallel::parallel_for(0, n, [src, output](std::size_t i) {
            output[i] = src[i];
        });
    }
};

template<typename T, std::size_t... Dims>
struct Evaluator<const Tensor<T, Dims...>&> : Evaluator<Tensor<T, Dims...>> {};

// ============================================================================
// GEMM Evaluator
// ============================================================================

template<typename A, typename B, typename Bias>
struct Evaluator<expr::GemmExpr<A, B, Bias>> {
    using Traits = traits::ExprTraits<expr::GemmExpr<A, B, Bias>>;
    using T = typename Traits::value_type;

    static void run(const expr::GemmExpr<A, B, Bias>& e, T* output) {
        constexpr std::size_t M = Traits::M;
        constexpr std::size_t K = Traits::K;
        constexpr std::size_t N = Traits::N;

        const T* a_data = get_data(e.a);
        const T* b_data = get_data(e.b);

        if constexpr (Traits::has_bias) {
            const T* bias_data = get_data(e.bias);
            gemm_bias_impl(a_data, b_data, bias_data, output, M, N, K, K, N, N);
        } else {
            gemm_impl(a_data, b_data, output, M, N, K, K, N, N, false);
        }
    }

private:
    template<typename Tensor>
    static const T* get_data(const Tensor& t) {
        return t.data();
    }
};

// ============================================================================
// GEMV Evaluator
// ============================================================================

template<typename A, typename X>
struct Evaluator<expr::GemvExpr<A, X>> {
    using Traits = traits::ExprTraits<expr::GemvExpr<A, X>>;
    using T = typename Traits::value_type;

    static void run(const expr::GemvExpr<A, X>& e, T* output) {
        constexpr std::size_t M = Traits::M;
        constexpr std::size_t K = Traits::K;

        const T* a_data = e.a.data();
        const T* x_data = e.x.data();

        // Parallel over rows
        parallel::parallel_for(0, M, [=](std::size_t i) {
            T sum = T{0};
            for (std::size_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * x_data[k];
            }
            output[i] = sum;
        });
    }
};

// ============================================================================
// Dot Product Evaluator
// ============================================================================

template<typename A, typename B>
struct Evaluator<expr::DotExpr<A, B>> {
    using Traits = traits::ExprTraits<expr::DotExpr<A, B>>;
    using T = typename Traits::value_type;

    static void run(const expr::DotExpr<A, B>& e, T* output) {
        constexpr std::size_t K = Traits::K;

        const T* a_data = e.a.data();
        const T* b_data = e.b.data();

        // Parallel reduction
        T result = parallel::parallel_reduce(
            std::size_t{0}, K,
            T{0},
            [a_data, b_data](std::size_t k) { return a_data[k] * b_data[k]; },
            [](T a, T b) { return a + b; }
        );

        *output = result;
    }
};

// ============================================================================
// Unary Expression Evaluators (Activations)
// ============================================================================

/// ReLU evaluator
template<typename Input>
struct Evaluator<expr::UnaryExpr<Input, expr::ReluOp>> {
    using InputTraits = traits::ExprTraits<std::remove_cvref_t<Input>>;
    using T = typename InputTraits::value_type;

    static void run(const expr::UnaryExpr<Input, expr::ReluOp>& e, T* output) {
        constexpr std::size_t N = InputTraits::output_size;

        // First evaluate input
        auto input_buf = memory::make_aligned<T>(N);
        evaluate_input(e.input, input_buf.get());

        // Apply ReLU
        parallel::parallel_for(0, N, [&input_buf, output](std::size_t i) {
            output[i] = std::max(T{0}, input_buf[i]);
        });
    }

private:
    template<typename E>
    static void evaluate_input(const E& expr, T* buf) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            const T* src = expr.data();
            std::copy(src, src + InputTraits::output_size, buf);
        } else {
            Evaluator<std::remove_cvref_t<E>>::run(expr, buf);
        }
    }
};

/// SiLU evaluator: x * sigmoid(x)
template<typename Input>
struct Evaluator<expr::UnaryExpr<Input, expr::SiluOp>> {
    using InputTraits = traits::ExprTraits<std::remove_cvref_t<Input>>;
    using T = typename InputTraits::value_type;

    static void run(const expr::UnaryExpr<Input, expr::SiluOp>& e, T* output) {
        constexpr std::size_t N = InputTraits::output_size;

        auto input_buf = memory::make_aligned<T>(N);
        evaluate_input(e.input, input_buf.get());

        parallel::parallel_for(0, N, [&input_buf, output](std::size_t i) {
            T x = input_buf[i];
            T sigmoid_x = T{1} / (T{1} + std::exp(-x));
            output[i] = x * sigmoid_x;
        });
    }

private:
    template<typename E>
    static void evaluate_input(const E& expr, T* buf) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            const T* src = expr.data();
            std::copy(src, src + InputTraits::output_size, buf);
        } else {
            Evaluator<std::remove_cvref_t<E>>::run(expr, buf);
        }
    }
};

/// GELU evaluator (approximate)
template<typename Input>
struct Evaluator<expr::UnaryExpr<Input, expr::GeluOp>> {
    using InputTraits = traits::ExprTraits<std::remove_cvref_t<Input>>;
    using T = typename InputTraits::value_type;

    static void run(const expr::UnaryExpr<Input, expr::GeluOp>& e, T* output) {
        constexpr std::size_t N = InputTraits::output_size;
        constexpr T sqrt_2_over_pi = T{0.7978845608028654};
        constexpr T coeff = T{0.044715};

        auto input_buf = memory::make_aligned<T>(N);
        evaluate_input(e.input, input_buf.get());

        parallel::parallel_for(0, N, [&input_buf, output](std::size_t i) {
            T x = input_buf[i];
            T inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            output[i] = T{0.5} * x * (T{1} + std::tanh(inner));
        });
    }

private:
    template<typename E>
    static void evaluate_input(const E& expr, T* buf) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            const T* src = expr.data();
            std::copy(src, src + InputTraits::output_size, buf);
        } else {
            Evaluator<std::remove_cvref_t<E>>::run(expr, buf);
        }
    }
};

/// Tanh evaluator
template<typename Input>
struct Evaluator<expr::UnaryExpr<Input, expr::TanhOp>> {
    using InputTraits = traits::ExprTraits<std::remove_cvref_t<Input>>;
    using T = typename InputTraits::value_type;

    static void run(const expr::UnaryExpr<Input, expr::TanhOp>& e, T* output) {
        constexpr std::size_t N = InputTraits::output_size;

        auto input_buf = memory::make_aligned<T>(N);
        evaluate_input(e.input, input_buf.get());

        parallel::parallel_for(0, N, [&input_buf, output](std::size_t i) {
            output[i] = std::tanh(input_buf[i]);
        });
    }

private:
    template<typename E>
    static void evaluate_input(const E& expr, T* buf) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            const T* src = expr.data();
            std::copy(src, src + InputTraits::output_size, buf);
        } else {
            Evaluator<std::remove_cvref_t<E>>::run(expr, buf);
        }
    }
};

// ============================================================================
// Fused GEMM + Activation Evaluator
// ============================================================================

/// Detect and fuse GEMM + ReLU pattern
template<typename A, typename B, typename Bias>
struct Evaluator<expr::UnaryExpr<expr::GemmExpr<A, B, Bias>, expr::ReluOp>> {
    using GemmTraits = traits::ExprTraits<expr::GemmExpr<A, B, Bias>>;
    using T = typename GemmTraits::value_type;

    static void run(const expr::UnaryExpr<expr::GemmExpr<A, B, Bias>, expr::ReluOp>& e, T* output) {
        constexpr std::size_t M = GemmTraits::M;
        constexpr std::size_t K = GemmTraits::K;
        constexpr std::size_t N = GemmTraits::N;

        const auto& gemm_expr = e.input;
        const T* a_data = gemm_expr.a.data();
        const T* b_data = gemm_expr.b.data();

        // Execute GEMM
        if constexpr (GemmTraits::has_bias) {
            const T* bias_data = gemm_expr.bias.data();
            gemm_bias_impl(a_data, b_data, bias_data, output, M, N, K, K, N, N);
        } else {
            gemm_impl(a_data, b_data, output, M, N, K, K, N, N, false);
        }

        // Fused ReLU (in-place)
        parallel::parallel_for(0, M * N, [output](std::size_t i) {
            output[i] = std::max(T{0}, output[i]);
        });
    }
};

/// Detect and fuse GEMM + SiLU pattern
template<typename A, typename B, typename Bias>
struct Evaluator<expr::UnaryExpr<expr::GemmExpr<A, B, Bias>, expr::SiluOp>> {
    using GemmTraits = traits::ExprTraits<expr::GemmExpr<A, B, Bias>>;
    using T = typename GemmTraits::value_type;

    static void run(const expr::UnaryExpr<expr::GemmExpr<A, B, Bias>, expr::SiluOp>& e, T* output) {
        constexpr std::size_t M = GemmTraits::M;
        constexpr std::size_t K = GemmTraits::K;
        constexpr std::size_t N = GemmTraits::N;

        const auto& gemm_expr = e.input;
        const T* a_data = gemm_expr.a.data();
        const T* b_data = gemm_expr.b.data();

        if constexpr (GemmTraits::has_bias) {
            const T* bias_data = gemm_expr.bias.data();
            gemm_bias_impl(a_data, b_data, bias_data, output, M, N, K, K, N, N);
        } else {
            gemm_impl(a_data, b_data, output, M, N, K, K, N, N, false);
        }

        // Fused SiLU (in-place)
        parallel::parallel_for(0, M * N, [output](std::size_t i) {
            T x = output[i];
            T sigmoid_x = T{1} / (T{1} + std::exp(-x));
            output[i] = x * sigmoid_x;
        });
    }
};

// ============================================================================
// Normalization Evaluators
// ============================================================================

/// RMSNorm evaluator
template<typename Input, typename Weight>
struct Evaluator<expr::RMSNormExpr<Input, Weight>> {
    using Traits = traits::ExprTraits<expr::RMSNormExpr<Input, Weight>>;
    using T = typename Traits::value_type;

    static void run(const expr::RMSNormExpr<Input, Weight>& e, T* output) {
        constexpr std::size_t num_rows = Traits::num_rows;
        constexpr std::size_t hidden_dim = Traits::hidden_dim;

        const T* input_data = get_input_data(e.input);
        const T* weight_data = e.weight.data();

        RMSNormKernel<T>::run(
            input_data, weight_data, output,
            num_rows, hidden_dim, static_cast<T>(e.eps)
        );
    }

private:
    template<typename E>
    static const T* get_input_data(const E& expr) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            return expr.data();
        } else {
            // For non-terminal inputs, we'd need to evaluate first
            // For now, assume input is always a tensor
            return expr.data();
        }
    }
};

/// LayerNorm evaluator
template<typename Input, typename Gamma, typename Beta>
struct Evaluator<expr::LayerNormExpr<Input, Gamma, Beta>> {
    using Traits = traits::ExprTraits<expr::LayerNormExpr<Input, Gamma, Beta>>;
    using T = typename Traits::value_type;

    static void run(const expr::LayerNormExpr<Input, Gamma, Beta>& e, T* output) {
        constexpr std::size_t num_rows = Traits::num_rows;
        constexpr std::size_t hidden_dim = Traits::hidden_dim;

        const T* input_data = get_input_data(e.input);
        const T* gamma_data = e.gamma.data();
        const T* beta_data = e.beta.data();

        LayerNormKernel<T>::run(
            input_data, gamma_data, beta_data, output,
            num_rows, hidden_dim, static_cast<T>(e.eps)
        );
    }

private:
    template<typename E>
    static const T* get_input_data(const E& expr) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            return expr.data();
        } else {
            return expr.data();
        }
    }
};

// ============================================================================
// Fused RMSNorm + Linear Pattern
// ============================================================================

/// Detect RMSNorm followed by Linear (GEMM): linear(rmsnorm(x, w), W)
template<typename Input, typename NormWeight, typename LinearWeight, typename Bias>
struct Evaluator<expr::GemmExpr<expr::RMSNormExpr<Input, NormWeight>, LinearWeight, Bias>> {
    using NormExpr = expr::RMSNormExpr<Input, NormWeight>;
    using GemmExpr = expr::GemmExpr<NormExpr, LinearWeight, Bias>;
    using NormTraits = traits::ExprTraits<NormExpr>;
    using GemmTraits = traits::ExprTraits<GemmExpr>;
    using T = typename GemmTraits::value_type;

    static void run(const GemmExpr& e, T* output) {
        constexpr std::size_t M = GemmTraits::M;
        constexpr std::size_t K = GemmTraits::K;
        constexpr std::size_t N = GemmTraits::N;

        const auto& norm_expr = e.a;
        const T* input_data = norm_expr.input.data();
        const T* norm_weight = norm_expr.weight.data();
        const T* linear_weight = e.b.data();
        T eps = norm_expr.eps;

        // Allocate buffer for normalized input
        auto normed = memory::make_aligned<T>(M * K);

        // First: RMSNorm
        RMSNormKernel<T>::run(input_data, norm_weight, normed.get(), M, K, eps);

        // Then: Linear (GEMM)
        if constexpr (GemmTraits::has_bias) {
            const T* bias_data = e.bias.data();
            gemm_bias_impl(normed.get(), linear_weight, bias_data, output, M, N, K, K, N, N);
        } else {
            gemm_impl(normed.get(), linear_weight, output, M, N, K, K, N, N, false);
        }
    }
};

/// Detect LayerNorm followed by Linear (GEMM): linear(layernorm(x, g, b), W)
template<typename Input, typename Gamma, typename Beta, typename LinearWeight, typename Bias>
struct Evaluator<expr::GemmExpr<expr::LayerNormExpr<Input, Gamma, Beta>, LinearWeight, Bias>> {
    using NormExpr = expr::LayerNormExpr<Input, Gamma, Beta>;
    using GemmExpr = expr::GemmExpr<NormExpr, LinearWeight, Bias>;
    using NormTraits = traits::ExprTraits<NormExpr>;
    using GemmTraits = traits::ExprTraits<GemmExpr>;
    using T = typename GemmTraits::value_type;

    static void run(const GemmExpr& e, T* output) {
        constexpr std::size_t M = GemmTraits::M;
        constexpr std::size_t K = GemmTraits::K;
        constexpr std::size_t N = GemmTraits::N;

        const auto& norm_expr = e.a;
        const T* input_data = norm_expr.input.data();
        const T* gamma_data = norm_expr.gamma.data();
        const T* beta_data = norm_expr.beta.data();
        const T* linear_weight = e.b.data();
        T eps = norm_expr.eps;

        // Allocate buffer for normalized input
        auto normed = memory::make_aligned<T>(M * K);

        // First: LayerNorm
        LayerNormKernel<T>::run(input_data, gamma_data, beta_data, normed.get(), M, K, eps);

        // Then: Linear (GEMM)
        if constexpr (GemmTraits::has_bias) {
            const T* bias_data = e.bias.data();
            gemm_bias_impl(normed.get(), linear_weight, bias_data, output, M, N, K, K, N, N);
        } else {
            gemm_impl(normed.get(), linear_weight, output, M, N, K, K, N, N, false);
        }
    }
};

} // namespace kernelix::kernel
