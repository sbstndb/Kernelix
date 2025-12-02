#pragma once

#include "base.hpp"
#include <utility>

namespace kernelix::expr {

/// RMSNorm expression: y = x * weight / sqrt(mean(x^2) + eps)
/// Normalizes over the last dimension
template<typename Input, typename Weight>
struct RMSNormExpr : ExprBase {
    using input_type = Input;
    using weight_type = Weight;

    Input input;
    Weight weight;
    float eps;

    constexpr RMSNormExpr(Input in, Weight w, float epsilon = 1e-6f)
        : input(std::move(in)), weight(std::move(w)), eps(epsilon) {}
};

/// LayerNorm expression: y = (x - mean) / sqrt(var + eps) * gamma + beta
/// Normalizes over the last dimension
template<typename Input, typename Gamma, typename Beta>
struct LayerNormExpr : ExprBase {
    using input_type = Input;
    using gamma_type = Gamma;
    using beta_type = Beta;

    Input input;
    Gamma gamma;
    Beta beta;
    float eps;

    constexpr LayerNormExpr(Input in, Gamma g, Beta b, float epsilon = 1e-5f)
        : input(std::move(in)), gamma(std::move(g)), beta(std::move(b)), eps(epsilon) {}
};

/// Helper to create RMSNorm expression
template<typename Input, typename Weight>
constexpr auto make_rmsnorm(Input&& input, Weight&& weight, float eps = 1e-6f) {
    return RMSNormExpr<std::decay_t<Input>, std::decay_t<Weight>>{
        std::forward<Input>(input), std::forward<Weight>(weight), eps
    };
}

/// Helper to create LayerNorm expression
template<typename Input, typename Gamma, typename Beta>
constexpr auto make_layernorm(Input&& input, Gamma&& gamma, Beta&& beta, float eps = 1e-5f) {
    return LayerNormExpr<std::decay_t<Input>, std::decay_t<Gamma>, std::decay_t<Beta>>{
        std::forward<Input>(input), std::forward<Gamma>(gamma), std::forward<Beta>(beta), eps
    };
}

/// Softmax expression: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
/// Applies softmax over the last dimension (numerically stable version)
template<typename Input>
struct SoftmaxExpr : ExprBase {
    using input_type = Input;

    Input input;

    constexpr explicit SoftmaxExpr(Input in)
        : input(std::move(in)) {}
};

/// Helper to create Softmax expression
template<typename Input>
constexpr auto make_softmax(Input&& input) {
    return SoftmaxExpr<std::decay_t<Input>>{
        std::forward<Input>(input)
    };
}

} // namespace kernelix::expr
