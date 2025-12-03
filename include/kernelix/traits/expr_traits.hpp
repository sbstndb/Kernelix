#pragma once

#include "../core/types.hpp"
#include "../expr/base.hpp"
#include "../expr/contraction.hpp"
#include "../expr/unary.hpp"
#include "../expr/binary.hpp"
#include "../expr/norm.hpp"
#include "../expr/attention.hpp"
#include "../expr/rope.hpp"
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

// Forward declaration for expression shape extraction
template<typename E>
struct ExprTraits;

/// ShapeOf for expression types - uses ExprTraits to get output shape
template<typename Input, typename Weight>
struct ShapeOf<expr::RMSNormExpr<Input, Weight>> {
    using Traits = ExprTraits<expr::RMSNormExpr<Input, Weight>>;
    static constexpr std::size_t rank = Traits::rank;
    static constexpr auto shape = Traits::output_shape;
    using value_type = typename Traits::value_type;
};

template<typename Input, typename Gamma, typename Beta>
struct ShapeOf<expr::LayerNormExpr<Input, Gamma, Beta>> {
    using Traits = ExprTraits<expr::LayerNormExpr<Input, Gamma, Beta>>;
    static constexpr std::size_t rank = Traits::rank;
    static constexpr auto shape = Traits::output_shape;
    using value_type = typename Traits::value_type;
};

template<typename Input>
struct ShapeOf<expr::SoftmaxExpr<Input>> {
    using Traits = ExprTraits<expr::SoftmaxExpr<Input>>;
    static constexpr std::size_t rank = Traits::rank;
    static constexpr auto shape = Traits::output_shape;
    using value_type = typename Traits::value_type;
};

// ============================================================================
// Value type extraction
// ============================================================================

/// Helper to recursively extract value_type from nested types
template<typename T>
struct ValueTypeOf {
    using type = typename std::remove_cvref_t<T>::value_type;
};

/// Specialization for RMSNormExpr - extract from input
template<typename Input, typename Weight>
struct ValueTypeOf<expr::RMSNormExpr<Input, Weight>> {
    using type = typename ValueTypeOf<std::remove_cvref_t<Input>>::type;
};

/// Specialization for LayerNormExpr - extract from input
template<typename Input, typename Gamma, typename Beta>
struct ValueTypeOf<expr::LayerNormExpr<Input, Gamma, Beta>> {
    using type = typename ValueTypeOf<std::remove_cvref_t<Input>>::type;
};

/// Specialization for SoftmaxExpr - extract from input
template<typename Input>
struct ValueTypeOf<expr::SoftmaxExpr<Input>> {
    using type = typename ValueTypeOf<std::remove_cvref_t<Input>>::type;
};

/// Specialization for GemmExpr - extract from first operand
template<typename A, typename B, typename Bias>
struct ValueTypeOf<expr::GemmExpr<A, B, Bias>> {
    using type = typename ValueTypeOf<std::remove_cvref_t<A>>::type;
};

/// Specialization for BinaryExpr - extract from LHS
template<typename LHS, typename RHS, typename Op>
struct ValueTypeOf<expr::BinaryExpr<LHS, RHS, Op>> {
    using type = typename ValueTypeOf<std::remove_cvref_t<LHS>>::type;
};

/// Specialization for UnaryExpr - extract from input
template<typename Input, typename Op>
struct ValueTypeOf<expr::UnaryExpr<Input, Op>> {
    using type = typename ValueTypeOf<std::remove_cvref_t<Input>>::type;
};

/// Specialization for ScaledDotProductAttentionExpr - extract from query
template<typename Q, typename K, typename V>
struct ValueTypeOf<expr::ScaledDotProductAttentionExpr<Q, K, V>> {
    using type = typename ValueTypeOf<std::remove_cvref_t<Q>>::type;
};

/// Specialization for CausalAttentionExpr - extract from query
template<typename Q, typename K, typename V>
struct ValueTypeOf<expr::CausalAttentionExpr<Q, K, V>> {
    using type = typename ValueTypeOf<std::remove_cvref_t<Q>>::type;
};

/// Specialization for RoPEExpr - extract from input
template<typename Input, typename CosSin>
struct ValueTypeOf<expr::RoPEExpr<Input, CosSin>> {
    using type = typename ValueTypeOf<std::remove_cvref_t<Input>>::type;
};

template<typename T>
using value_type_t = typename ValueTypeOf<std::remove_cvref_t<T>>::type;

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

/// Traits for RMSNorm expression
/// Input: [batch, hidden_dim], Weight: [hidden_dim], Output: [batch, hidden_dim]
template<typename Input, typename Weight>
struct ExprTraits<expr::RMSNormExpr<Input, Weight>> {
    using op_category = op_category::Normalization;
    using value_type = value_type_t<Input>;
    using input_traits = ExprTraits<std::remove_cvref_t<Input>>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;

    // Output shape is same as input shape
    static constexpr std::size_t rank = input_traits::rank;
    static constexpr auto output_shape = input_traits::output_shape;
    static constexpr std::size_t output_size = input_traits::output_size;

    // For 2D input [batch, hidden_dim]
    static constexpr std::size_t num_rows = (rank >= 1) ? output_shape[0] : 1;
    static constexpr std::size_t hidden_dim = (rank >= 2) ? output_shape[1] : output_shape[0];

    // Cost estimation: read input twice (sum_sq, normalize), read weight once, write output
    static constexpr std::size_t estimated_flops = 3 * output_size;  // sq, div, mul
    static constexpr std::size_t estimated_bytes = 3 * output_size * sizeof(value_type);

    static constexpr bool is_memory_bound = true;
    static constexpr bool allows_fusion = true;  // Can fuse with following Linear
};

/// Traits for LayerNorm expression
/// Input: [batch, hidden_dim], Gamma/Beta: [hidden_dim], Output: [batch, hidden_dim]
template<typename Input, typename Gamma, typename Beta>
struct ExprTraits<expr::LayerNormExpr<Input, Gamma, Beta>> {
    using op_category = op_category::Normalization;
    using value_type = value_type_t<Input>;
    using input_traits = ExprTraits<std::remove_cvref_t<Input>>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;

    // Output shape is same as input shape
    static constexpr std::size_t rank = input_traits::rank;
    static constexpr auto output_shape = input_traits::output_shape;
    static constexpr std::size_t output_size = input_traits::output_size;

    // For 2D input [batch, hidden_dim]
    static constexpr std::size_t num_rows = (rank >= 1) ? output_shape[0] : 1;
    static constexpr std::size_t hidden_dim = (rank >= 2) ? output_shape[1] : output_shape[0];

    // Cost estimation: read input 3x (mean, var, normalize), read gamma/beta, write output
    static constexpr std::size_t estimated_flops = 5 * output_size;  // sub, sq, div, mul, add
    static constexpr std::size_t estimated_bytes = 4 * output_size * sizeof(value_type);

    static constexpr bool is_memory_bound = true;
    static constexpr bool allows_fusion = true;
};

/// Traits for Softmax expression
/// Input: [batch, seq_len] or [batch, seq_len, vocab_size], Output: same shape
/// Softmax is applied over the last dimension
template<typename Input>
struct ExprTraits<expr::SoftmaxExpr<Input>> {
    using op_category = op_category::Normalization;
    using value_type = value_type_t<Input>;
    using input_traits = ExprTraits<std::remove_cvref_t<Input>>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;

    // Output shape is same as input shape
    static constexpr std::size_t rank = input_traits::rank;
    static constexpr auto output_shape = input_traits::output_shape;
    static constexpr std::size_t output_size = input_traits::output_size;

    // For 2D input [batch, dim] - softmax over dim
    // For 1D input [dim] - softmax over entire vector
    static constexpr std::size_t num_rows = (rank >= 2) ? output_shape[0] : 1;
    static constexpr std::size_t softmax_dim = (rank >= 2) ? output_shape[1] : output_shape[0];

    // Cost estimation: max (1 pass), exp+sub (1 pass), sum (1 pass), div (1 pass)
    // ~4N flops per row (max, exp, sum, div)
    static constexpr std::size_t estimated_flops = 4 * output_size;
    static constexpr std::size_t estimated_bytes = 3 * output_size * sizeof(value_type);

    static constexpr bool is_memory_bound = true;
    static constexpr bool allows_fusion = true;  // Can fuse with preceding matmul (attention)
};

/// Traits for Scaled Dot-Product Attention expression
/// Q: [seq_len_q, head_dim], K: [seq_len_k, head_dim], V: [seq_len_k, head_dim]
/// Output: [seq_len_q, head_dim]
template<typename Query, typename Key, typename Value>
struct ExprTraits<expr::ScaledDotProductAttentionExpr<Query, Key, Value>> {
    using op_category = op_category::Contraction;
    using value_type = value_type_t<Query>;

    using query_traits = ExprTraits<std::remove_cvref_t<Query>>;
    using key_traits = ExprTraits<std::remove_cvref_t<Key>>;
    using value_traits = ExprTraits<std::remove_cvref_t<Value>>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;

    // Extract dimensions from Q, K, V
    static constexpr auto q_shape = query_traits::output_shape;
    static constexpr auto k_shape = key_traits::output_shape;
    static constexpr auto v_shape = value_traits::output_shape;

    // For 2D: Q[seq_len_q, head_dim], K[seq_len_k, head_dim], V[seq_len_k, head_dim]
    static constexpr std::size_t seq_len_q = q_shape[0];
    static constexpr std::size_t seq_len_k = k_shape[0];
    static constexpr std::size_t head_dim = q_shape[1];

    // Output shape is [seq_len_q, head_dim]
    static constexpr std::size_t rank = 2;
    static constexpr std::array<std::size_t, 2> output_shape = {seq_len_q, head_dim};
    static constexpr std::size_t output_size = seq_len_q * head_dim;

    // Cost estimation:
    // Q @ K^T: 2 * seq_len_q * seq_len_k * head_dim
    // Softmax: 4 * seq_len_q * seq_len_k
    // Attn @ V: 2 * seq_len_q * seq_len_k * head_dim
    static constexpr std::size_t estimated_flops =
        4 * seq_len_q * seq_len_k * head_dim + 4 * seq_len_q * seq_len_k;

    // Memory: Q, K, V, attention scores, output
    static constexpr std::size_t estimated_bytes =
        (seq_len_q * head_dim + 2 * seq_len_k * head_dim +
         seq_len_q * seq_len_k + seq_len_q * head_dim) * sizeof(value_type);

    static constexpr bool is_memory_bound = false;  // Usually compute-bound
    static constexpr bool allows_fusion = false;    // Complex operation, difficult to fuse
};

/// Traits for Causal Attention expression (same shape as standard attention)
template<typename Query, typename Key, typename Value>
struct ExprTraits<expr::CausalAttentionExpr<Query, Key, Value>>
    : ExprTraits<expr::ScaledDotProductAttentionExpr<Query, Key, Value>> {
    // Inherits everything from standard attention
    // Causal mask doesn't change output shape or significantly change cost
};

/// Traits for RoPE expression
/// Input: [seq_len, head_dim], Output: [seq_len, head_dim]
template<typename Input, typename CosSin>
struct ExprTraits<expr::RoPEExpr<Input, CosSin>> {
    using op_category = op_category::Elementwise;
    using value_type = value_type_t<Input>;
    using input_traits = ExprTraits<std::remove_cvref_t<Input>>;

    static constexpr bool is_expression = true;
    static constexpr bool is_terminal = false;

    // Output shape is same as input shape
    static constexpr std::size_t rank = input_traits::rank;
    static constexpr auto output_shape = input_traits::output_shape;
    static constexpr std::size_t output_size = input_traits::output_size;

    // For 2D input [seq_len, head_dim]
    static constexpr std::size_t seq_len = output_shape[0];
    static constexpr std::size_t head_dim = output_shape[1];

    // Cost: 4 multiplies + 2 adds per pair = 6 ops per pair, head_dim/2 pairs per position
    static constexpr std::size_t estimated_flops = seq_len * head_dim * 3;
    static constexpr std::size_t estimated_bytes = 4 * output_size * sizeof(value_type);

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
