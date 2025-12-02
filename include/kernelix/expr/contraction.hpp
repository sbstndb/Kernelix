#pragma once

#include "base.hpp"
#include "../core/concepts.hpp"
#include <utility>

namespace kernelix::expr {

/// Contraction dimensions specification
template<std::size_t... Dims>
struct ContractionDims {};

/// General contraction expression (GEMM, GEMV, dot product)
template<typename A, typename B, typename ContrDims = ContractionDims<>>
struct ContractionExpr : ExprBase {
    using a_type = A;
    using b_type = B;
    using contraction_dims = ContrDims;

    A a;
    B b;

    constexpr ContractionExpr(A lhs, B rhs) : a(std::move(lhs)), b(std::move(rhs)) {}
};

/// GEMM-specific expression with optional bias
template<typename A, typename B, typename Bias = void>
struct GemmExpr : ExprBase {
    using a_type = A;
    using b_type = B;
    using bias_type = Bias;
    static constexpr bool has_bias = !std::is_void_v<Bias>;

    A a;
    B b;
    [[no_unique_address]] Bias bias;

    constexpr GemmExpr(A mat_a, B mat_b)
        requires std::is_void_v<Bias>
        : a(std::move(mat_a)), b(std::move(mat_b)), bias{} {}

    constexpr GemmExpr(A mat_a, B mat_b, Bias mat_bias)
        requires (!std::is_void_v<Bias>)
        : a(std::move(mat_a)), b(std::move(mat_b)), bias(std::move(mat_bias)) {}
};

/// GEMV-specific expression
template<typename A, typename X>
struct GemvExpr : ExprBase {
    using a_type = A;
    using x_type = X;

    A a;
    X x;

    constexpr GemvExpr(A mat, X vec) : a(std::move(mat)), x(std::move(vec)) {}
};

/// Dot product expression
template<typename A, typename B>
struct DotExpr : ExprBase {
    using a_type = A;
    using b_type = B;

    A a;
    B b;

    constexpr DotExpr(A lhs, B rhs) : a(std::move(lhs)), b(std::move(rhs)) {}
};

} // namespace kernelix::expr
