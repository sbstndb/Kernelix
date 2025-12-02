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

/// Empty type for no-bias case
struct NoBias {};

/// GEMM-specific expression with optional bias (primary template with bias)
template<typename A, typename B, typename Bias = void>
struct GemmExpr : ExprBase {
    using a_type = A;
    using b_type = B;
    using bias_type = Bias;
    static constexpr bool has_bias = true;

    A a;
    B b;
    Bias bias;

    constexpr GemmExpr(A mat_a, B mat_b, Bias mat_bias)
        : a(std::move(mat_a)), b(std::move(mat_b)), bias(std::move(mat_bias)) {}
};

/// Specialization for no-bias case
template<typename A, typename B>
struct GemmExpr<A, B, void> : ExprBase {
    using a_type = A;
    using b_type = B;
    using bias_type = void;
    static constexpr bool has_bias = false;

    A a;
    B b;

    constexpr GemmExpr(A mat_a, B mat_b)
        : a(std::move(mat_a)), b(std::move(mat_b)) {}
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
