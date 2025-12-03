#pragma once

#include "../traits/expr_traits.hpp"
#include "../expr/contraction.hpp"
#include "../expr/unary.hpp"
#include "../expr/binary.hpp"
#include "../expr/norm.hpp"
#include "gemm/impl.hpp"
#include "gemm/fused.hpp"
#include "norm/rmsnorm.hpp"
#include "norm/layernorm.hpp"
#include "norm/softmax.hpp"
#include "attention/impl.hpp"
#include "rope/impl.hpp"
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
// Binary Expression Evaluators (Element-wise ops)
// ============================================================================

/// Element-wise multiply evaluator
template<typename LHS, typename RHS>
struct Evaluator<expr::BinaryExpr<LHS, RHS, expr::MulOp>> {
    using Traits = traits::ExprTraits<expr::BinaryExpr<LHS, RHS, expr::MulOp>>;
    using T = typename Traits::value_type;

    static void run(const expr::BinaryExpr<LHS, RHS, expr::MulOp>& e, T* output) {
        constexpr std::size_t N = Traits::output_size;

        auto lhs_buf = memory::make_aligned<T>(N);
        auto rhs_buf = memory::make_aligned<T>(N);

        evaluate_operand(e.lhs, lhs_buf.get());
        evaluate_operand(e.rhs, rhs_buf.get());

        parallel::parallel_for(0, N, [&lhs_buf, &rhs_buf, output](std::size_t i) {
            output[i] = lhs_buf[i] * rhs_buf[i];
        });
    }

private:
    template<typename E>
    static void evaluate_operand(const E& expr, T* buf) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            const T* src = expr.data();
            std::copy(src, src + Traits::output_size, buf);
        } else {
            Evaluator<std::remove_cvref_t<E>>::run(expr, buf);
        }
    }
};

/// Element-wise add evaluator
template<typename LHS, typename RHS>
struct Evaluator<expr::BinaryExpr<LHS, RHS, expr::AddOp>> {
    using Traits = traits::ExprTraits<expr::BinaryExpr<LHS, RHS, expr::AddOp>>;
    using T = typename Traits::value_type;

    static void run(const expr::BinaryExpr<LHS, RHS, expr::AddOp>& e, T* output) {
        constexpr std::size_t N = Traits::output_size;

        auto lhs_buf = memory::make_aligned<T>(N);
        auto rhs_buf = memory::make_aligned<T>(N);

        evaluate_operand(e.lhs, lhs_buf.get());
        evaluate_operand(e.rhs, rhs_buf.get());

        parallel::parallel_for(0, N, [&lhs_buf, &rhs_buf, output](std::size_t i) {
            output[i] = lhs_buf[i] + rhs_buf[i];
        });
    }

private:
    template<typename E>
    static void evaluate_operand(const E& expr, T* buf) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            const T* src = expr.data();
            std::copy(src, src + Traits::output_size, buf);
        } else {
            Evaluator<std::remove_cvref_t<E>>::run(expr, buf);
        }
    }
};

/// Element-wise subtract evaluator
template<typename LHS, typename RHS>
struct Evaluator<expr::BinaryExpr<LHS, RHS, expr::SubOp>> {
    using Traits = traits::ExprTraits<expr::BinaryExpr<LHS, RHS, expr::SubOp>>;
    using T = typename Traits::value_type;

    static void run(const expr::BinaryExpr<LHS, RHS, expr::SubOp>& e, T* output) {
        constexpr std::size_t N = Traits::output_size;

        auto lhs_buf = memory::make_aligned<T>(N);
        auto rhs_buf = memory::make_aligned<T>(N);

        evaluate_operand(e.lhs, lhs_buf.get());
        evaluate_operand(e.rhs, rhs_buf.get());

        parallel::parallel_for(0, N, [&lhs_buf, &rhs_buf, output](std::size_t i) {
            output[i] = lhs_buf[i] - rhs_buf[i];
        });
    }

private:
    template<typename E>
    static void evaluate_operand(const E& expr, T* buf) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            const T* src = expr.data();
            std::copy(src, src + Traits::output_size, buf);
        } else {
            Evaluator<std::remove_cvref_t<E>>::run(expr, buf);
        }
    }
};

/// Element-wise divide evaluator
template<typename LHS, typename RHS>
struct Evaluator<expr::BinaryExpr<LHS, RHS, expr::DivOp>> {
    using Traits = traits::ExprTraits<expr::BinaryExpr<LHS, RHS, expr::DivOp>>;
    using T = typename Traits::value_type;

    static void run(const expr::BinaryExpr<LHS, RHS, expr::DivOp>& e, T* output) {
        constexpr std::size_t N = Traits::output_size;

        auto lhs_buf = memory::make_aligned<T>(N);
        auto rhs_buf = memory::make_aligned<T>(N);

        evaluate_operand(e.lhs, lhs_buf.get());
        evaluate_operand(e.rhs, rhs_buf.get());

        parallel::parallel_for(0, N, [&lhs_buf, &rhs_buf, output](std::size_t i) {
            output[i] = lhs_buf[i] / rhs_buf[i];
        });
    }

private:
    template<typename E>
    static void evaluate_operand(const E& expr, T* buf) {
        if constexpr (traits::ExprTraits<std::remove_cvref_t<E>>::is_terminal) {
            const T* src = expr.data();
            std::copy(src, src + Traits::output_size, buf);
        } else {
            Evaluator<std::remove_cvref_t<E>>::run(expr, buf);
        }
    }
};

// ============================================================================
// Fused GEMM + Activation Evaluator (True Fusion - applies during tile write)
// ============================================================================

/// Detect and fuse GEMM + ReLU pattern
/// Uses truly fused kernel that applies ReLU during final tile write
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

        // Use truly fused GEMM+ReLU kernel (activation applied during tile write)
        if constexpr (GemmTraits::has_bias) {
            // For bias case, do GEMM+bias then ReLU in-place
            const T* bias_data = gemm_expr.bias.data();
            gemm_bias_impl(a_data, b_data, bias_data, output, M, N, K, K, N, N);
            parallel::parallel_for(0, M * N, [output](std::size_t i) {
                output[i] = std::max(T{0}, output[i]);
            });
        } else {
            // Use fused kernel - no intermediate memory traffic
            gemm_relu_impl(a_data, b_data, output, M, N, K, K, N, N);
        }
    }
};

/// Detect and fuse GEMM + SiLU pattern
/// Uses truly fused kernel that applies SiLU during final tile write
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

        // Use truly fused GEMM+SiLU kernel
        if constexpr (GemmTraits::has_bias) {
            const T* bias_data = gemm_expr.bias.data();
            gemm_bias_impl(a_data, b_data, bias_data, output, M, N, K, K, N, N);
            parallel::parallel_for(0, M * N, [output](std::size_t i) {
                T x = output[i];
                T sigmoid_x = T{1} / (T{1} + std::exp(-x));
                output[i] = x * sigmoid_x;
            });
        } else {
            // Use fused kernel - no intermediate memory traffic
            gemm_silu_impl(a_data, b_data, output, M, N, K, K, N, N);
        }
    }
};

/// Detect and fuse GEMM + GELU pattern
/// Uses truly fused kernel that applies GELU during final tile write
template<typename A, typename B, typename Bias>
struct Evaluator<expr::UnaryExpr<expr::GemmExpr<A, B, Bias>, expr::GeluOp>> {
    using GemmTraits = traits::ExprTraits<expr::GemmExpr<A, B, Bias>>;
    using T = typename GemmTraits::value_type;

    static void run(const expr::UnaryExpr<expr::GemmExpr<A, B, Bias>, expr::GeluOp>& e, T* output) {
        constexpr std::size_t M = GemmTraits::M;
        constexpr std::size_t K = GemmTraits::K;
        constexpr std::size_t N = GemmTraits::N;

        const auto& gemm_expr = e.input;
        const T* a_data = gemm_expr.a.data();
        const T* b_data = gemm_expr.b.data();

        // Use truly fused GEMM+GELU kernel
        if constexpr (GemmTraits::has_bias) {
            constexpr T sqrt_2_over_pi = T{0.7978845608028654};
            constexpr T coeff = T{0.044715};
            const T* bias_data = gemm_expr.bias.data();
            gemm_bias_impl(a_data, b_data, bias_data, output, M, N, K, K, N, N);
            parallel::parallel_for(0, M * N, [output](std::size_t i) {
                T x = output[i];
                T inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                output[i] = T{0.5} * x * (T{1} + std::tanh(inner));
            });
        } else {
            // Use fused kernel - no intermediate memory traffic
            gemm_gelu_impl(a_data, b_data, output, M, N, K, K, N, N);
        }
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

/// Softmax evaluator
template<typename Input>
struct Evaluator<expr::SoftmaxExpr<Input>> {
    using Traits = traits::ExprTraits<expr::SoftmaxExpr<Input>>;
    using T = typename Traits::value_type;

    static void run(const expr::SoftmaxExpr<Input>& e, T* output) {
        constexpr std::size_t num_rows = Traits::num_rows;
        constexpr std::size_t softmax_dim = Traits::softmax_dim;

        const T* input_data = get_input_data(e.input);

        SoftmaxKernel<T>::run(
            input_data, output,
            num_rows, softmax_dim
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

// ============================================================================
// Attention Evaluators
// ============================================================================

/// Scaled Dot-Product Attention evaluator
template<typename Query, typename Key, typename Value>
struct Evaluator<expr::ScaledDotProductAttentionExpr<Query, Key, Value>> {
    using Traits = traits::ExprTraits<expr::ScaledDotProductAttentionExpr<Query, Key, Value>>;
    using T = typename Traits::value_type;

    static void run(const expr::ScaledDotProductAttentionExpr<Query, Key, Value>& e, T* output) {
        constexpr std::size_t seq_len_q = Traits::seq_len_q;
        constexpr std::size_t seq_len_k = Traits::seq_len_k;
        constexpr std::size_t head_dim = Traits::head_dim;

        const T* q_data = e.query.data();
        const T* k_data = e.key.data();
        const T* v_data = e.value.data();

        // Compute scale: 1/sqrt(head_dim) if not provided
        T scale = e.scale;
        if (scale == T{0}) {
            scale = T{1} / std::sqrt(static_cast<T>(head_dim));
        }

        AttentionKernel<T>::run_optimized(
            q_data, k_data, v_data, output,
            seq_len_q, seq_len_k, head_dim, scale
        );
    }
};

/// Causal Attention evaluator
template<typename Query, typename Key, typename Value>
struct Evaluator<expr::CausalAttentionExpr<Query, Key, Value>> {
    using Traits = traits::ExprTraits<expr::CausalAttentionExpr<Query, Key, Value>>;
    using T = typename Traits::value_type;

    static void run(const expr::CausalAttentionExpr<Query, Key, Value>& e, T* output) {
        constexpr std::size_t seq_len_q = Traits::seq_len_q;
        constexpr std::size_t seq_len_k = Traits::seq_len_k;
        constexpr std::size_t head_dim = Traits::head_dim;

        const T* q_data = e.query.data();
        const T* k_data = e.key.data();
        const T* v_data = e.value.data();

        T scale = e.scale;
        if (scale == T{0}) {
            scale = T{1} / std::sqrt(static_cast<T>(head_dim));
        }

        AttentionKernel<T>::run_causal_optimized(
            q_data, k_data, v_data, output,
            seq_len_q, seq_len_k, head_dim, scale
        );
    }
};

// ============================================================================
// RoPE Evaluator
// ============================================================================

/// RoPE (Rotary Position Embedding) evaluator
template<typename Input, typename CosSin>
struct Evaluator<expr::RoPEExpr<Input, CosSin>> {
    using Traits = traits::ExprTraits<expr::RoPEExpr<Input, CosSin>>;
    using T = typename Traits::value_type;

    static void run(const expr::RoPEExpr<Input, CosSin>& e, T* output) {
        constexpr std::size_t seq_len = Traits::seq_len;
        constexpr std::size_t head_dim = Traits::head_dim;

        const T* input_data = get_input_data(e.input);
        const T* cos_data = e.cos_cache.data();
        const T* sin_data = e.sin_cache.data();

        RoPEKernel<T>::run_optimized(
            input_data, cos_data, sin_data, output,
            seq_len, head_dim, e.position_offset
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

} // namespace kernelix::kernel
