#pragma once

#include "base.hpp"
#include <utility>

namespace kernelix::expr {

/// Binary expression template
template<typename LHS, typename RHS, typename Op>
struct BinaryExpr : ExprBase {
    using lhs_type = LHS;
    using rhs_type = RHS;
    using op_type = Op;

    LHS lhs;
    RHS rhs;

    constexpr BinaryExpr(LHS l, RHS r) : lhs(std::move(l)), rhs(std::move(r)) {}
};

/// Deduction guides
template<typename L, typename R>
BinaryExpr(L, R) -> BinaryExpr<L, R, MulOp>;

/// Helper to create binary expressions
template<typename L, typename R>
constexpr auto make_add(L&& lhs, R&& rhs) {
    return BinaryExpr<std::decay_t<L>, std::decay_t<R>, AddOp>{
        std::forward<L>(lhs), std::forward<R>(rhs)
    };
}

template<typename L, typename R>
constexpr auto make_sub(L&& lhs, R&& rhs) {
    return BinaryExpr<std::decay_t<L>, std::decay_t<R>, SubOp>{
        std::forward<L>(lhs), std::forward<R>(rhs)
    };
}

template<typename L, typename R>
constexpr auto make_mul(L&& lhs, R&& rhs) {
    return BinaryExpr<std::decay_t<L>, std::decay_t<R>, MulOp>{
        std::forward<L>(lhs), std::forward<R>(rhs)
    };
}

template<typename L, typename R>
constexpr auto make_div(L&& lhs, R&& rhs) {
    return BinaryExpr<std::decay_t<L>, std::decay_t<R>, DivOp>{
        std::forward<L>(lhs), std::forward<R>(rhs)
    };
}

} // namespace kernelix::expr
