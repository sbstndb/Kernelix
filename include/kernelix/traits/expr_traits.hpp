#pragma once

#include "../core/types.hpp"
#include "../expr/base.hpp"
#include "../expr/contraction.hpp"
#include "../expr/unary.hpp"
#include "../expr/binary.hpp"
#include <array>
#include <type_traits>

namespace kernelix::traits {

// Forward declaration
template<typename E>
struct ExprTraits;

// ============================================================================
// Shape extraction helpers
// ============================================================================

/// Extract shape from tensor types
template<typename T>
struct ShapeOf {
    static constexpr std::size_t rank = 0;
    static constexpr std::array<std::size_t, 0> shape = {};
};

/// Specialization for static Tensor
template<typename T, std::size_t... Dims>
struct ShapeOf<Tensor<T, Dims...>> {
    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::array<std::size_t, rank> shape = {Dims...};
    using value_type = T;
};

/// Specialization for const ref to Tensor
template<typename T, std::size_t... Dims>
struct ShapeOf<const Tensor<T, Dims...>&> : ShapeOf<Tensor<T, Dims...>> {};

/// Specialization for TensorView
template<typename T, std::size_t... Dims>
struct ShapeOf<TensorView<T, Dims...>> {
    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::array<std::size_t, rank> shape = {Dims...};
    using value_type = T;
};

template<typename T, std::size_t... Dims>
struct ShapeOf<const TensorView<T, Dims...>&> : ShapeOf<TensorView<T, Dims...>> {};

// ============================================================================
// Value type extraction
// ============================================================================

template<typename T>
struct ValueTypeOf {
    using type = typename std::remove_cvref_t<T>::value_type;
};

template<typename T>
using value_type_t = typename ValueTypeOf<T>::type;

// ============================================================================
// Expression Traits
// ============================================================================

/// Base traits - should be specialized for each expression type
template<typename E>
struct ExprTraits {
    using op_category = void;
    static constexpr bool is_expression = false;
    static constexpr bool is_terminal = true;  // Leaf node (tensor)
};

/// Traits for tensor terminals (leaf nodes)
template<typename T, std::size_t... Dims>
struct ExprTraits<Tensor<T, Dims...>> {
    using op_category = void;
    using value_type = T;
    static constexpr bool is_expression = false;
    static constexpr bool is_terminal = true;
    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::array<std::size_t, rank> output_shape = {Dims...};
    static constexpr std::size_t output_size = (Dims * ...);
};

template<typename T, std::size_t... Dims>
struct ExprTraits<const Tensor<T, Dims...>&> : ExprTraits<Tensor<T, Dims...>> {};

/// Traits for GEMM expression: C[M,N] = A[M,K] @ B[K,N]
template<typename A, typename B, typename Bias>
struct ExprTraits<expr::GemmExpr<A, B, Bias>> {
    using op_category = op_category::Contraction;
    using value_type = value_type_t<A>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;
    static constexpr bool has_bias = !std::is_void_v<Bias>;

    // Extract dimensions: A is [M, K], B is [K, N]
    static constexpr auto a_shape = ShapeOf<A>::shape;
    static constexpr auto b_shape = ShapeOf<B>::shape;

    static constexpr std::size_t M = a_shape[0];
    static constexpr std::size_t K = a_shape[1];
    static constexpr std::size_t N = b_shape[1];

    static constexpr std::size_t rank = 2;
    static constexpr std::array<std::size_t, 2> output_shape = {M, N};
    static constexpr std::size_t output_size = M * N;

    // Cost estimation
    static constexpr std::size_t estimated_flops = 2 * M * N * K;  // multiply-add
    static constexpr std::size_t estimated_bytes = (M * K + K * N + M * N) * sizeof(value_type);
    static constexpr double arithmetic_intensity =
        static_cast<double>(estimated_flops) / estimated_bytes;

    // Optimization hints
    static constexpr bool is_memory_bound = arithmetic_intensity < 10.0;
    static constexpr bool allows_fusion = true;
};

/// Traits for GEMV expression: y[M] = A[M,K] @ x[K]
template<typename A, typename X>
struct ExprTraits<expr::GemvExpr<A, X>> {
    using op_category = op_category::Contraction;
    using value_type = value_type_t<A>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;

    static constexpr auto a_shape = ShapeOf<A>::shape;
    static constexpr std::size_t M = a_shape[0];
    static constexpr std::size_t K = a_shape[1];

    static constexpr std::size_t rank = 1;
    static constexpr std::array<std::size_t, 1> output_shape = {M};
    static constexpr std::size_t output_size = M;

    static constexpr std::size_t estimated_flops = 2 * M * K;
    static constexpr std::size_t estimated_bytes = (M * K + K + M) * sizeof(value_type);
    static constexpr bool is_memory_bound = true;  // GEMV is always memory-bound
};

/// Traits for dot product: scalar = a[K] . b[K]
template<typename A, typename B>
struct ExprTraits<expr::DotExpr<A, B>> {
    using op_category = op_category::Contraction;
    using value_type = value_type_t<A>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;

    static constexpr auto a_shape = ShapeOf<A>::shape;
    static constexpr std::size_t K = a_shape[0];

    static constexpr std::size_t rank = 0;
    static constexpr std::array<std::size_t, 0> output_shape = {};
    static constexpr std::size_t output_size = 1;

    static constexpr std::size_t estimated_flops = 2 * K;
    static constexpr bool is_memory_bound = true;
};

/// Traits for unary expressions (activations)
template<typename Input, typename Op>
struct ExprTraits<expr::UnaryExpr<Input, Op>> {
    using op_category = op_category::Elementwise;
    using value_type = value_type_t<Input>;
    using input_traits = ExprTraits<std::remove_cvref_t<Input>>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;

    static constexpr std::size_t rank = input_traits::rank;
    static constexpr auto output_shape = input_traits::output_shape;
    static constexpr std::size_t output_size = input_traits::output_size;

    static constexpr bool is_memory_bound = true;
    static constexpr bool allows_fusion = true;  // Can fuse with preceding op
};

/// Traits for binary expressions (elementwise ops)
template<typename LHS, typename RHS, typename Op>
struct ExprTraits<expr::BinaryExpr<LHS, RHS, Op>> {
    using op_category = op_category::Elementwise;
    using value_type = value_type_t<LHS>;
    using lhs_traits = ExprTraits<std::remove_cvref_t<LHS>>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;

    static constexpr std::size_t rank = lhs_traits::rank;
    static constexpr auto output_shape = lhs_traits::output_shape;
    static constexpr std::size_t output_size = lhs_traits::output_size;

    static constexpr bool is_memory_bound = true;
    static constexpr bool allows_fusion = true;
};

// ============================================================================
// Helper concepts
// ============================================================================

template<typename E>
concept HasExprTraits = requires {
    typename ExprTraits<std::remove_cvref_t<E>>::op_category;
    { ExprTraits<std::remove_cvref_t<E>>::output_size } -> std::convertible_to<std::size_t>;
};

template<typename E>
concept IsContraction = std::is_same_v<
    typename ExprTraits<std::remove_cvref_t<E>>::op_category,
    op_category::Contraction
>;

template<typename E>
concept IsElementwise = std::is_same_v<
    typename ExprTraits<std::remove_cvref_t<E>>::op_category,
    op_category::Elementwise
>;

} // namespace kernelix::traits
