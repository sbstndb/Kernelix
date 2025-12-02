#pragma once

#include "base.hpp"
#include <utility>

namespace kernelix::expr {

/// Unary expression template (for activations, etc.)
template<typename Input, typename Op>
struct UnaryExpr : ExprBase {
    using input_type = Input;
    using op_type = Op;

    Input input;

    constexpr explicit UnaryExpr(Input in) : input(std::move(in)) {}
};

/// Helper to create unary expressions
template<typename Input>
constexpr auto make_relu(Input&& input) {
    return UnaryExpr<std::decay_t<Input>, ReluOp>{std::forward<Input>(input)};
}

template<typename Input>
constexpr auto make_silu(Input&& input) {
    return UnaryExpr<std::decay_t<Input>, SiluOp>{std::forward<Input>(input)};
}

template<typename Input>
constexpr auto make_gelu(Input&& input) {
    return UnaryExpr<std::decay_t<Input>, GeluOp>{std::forward<Input>(input)};
}

template<typename Input>
constexpr auto make_tanh(Input&& input) {
    return UnaryExpr<std::decay_t<Input>, TanhOp>{std::forward<Input>(input)};
}

} // namespace kernelix::expr
