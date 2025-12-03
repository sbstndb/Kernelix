#pragma once

#include "base.hpp"
#include <utility>
#include <cmath>

namespace kernelix::expr {

/// Scaled Dot-Product Attention expression
/// Output = softmax(Q @ K^T / sqrt(head_dim)) @ V
///
/// Q: [seq_len_q, head_dim] - Query matrix
/// K: [seq_len_k, head_dim] - Key matrix
/// V: [seq_len_k, head_dim] - Value matrix
/// Output: [seq_len_q, head_dim]
///
/// For batched attention:
/// Q: [batch, seq_len_q, head_dim]
/// K: [batch, seq_len_k, head_dim]
/// V: [batch, seq_len_k, head_dim]
/// Output: [batch, seq_len_q, head_dim]
template<typename Query, typename Key, typename Value>
struct ScaledDotProductAttentionExpr : ExprBase {
    using query_type = Query;
    using key_type = Key;
    using value_type = Value;

    Query query;
    Key key;
    Value value;
    float scale;  // Usually 1/sqrt(head_dim)

    constexpr ScaledDotProductAttentionExpr(Query q, Key k, Value v, float s)
        : query(std::move(q)), key(std::move(k)), value(std::move(v)), scale(s) {}
};

/// Helper to create Scaled Dot-Product Attention expression
/// Automatically computes scale = 1/sqrt(head_dim) from query dimensions
template<typename Q, typename K, typename V>
constexpr auto make_attention(Q&& q, K&& k, V&& v) {
    // Scale will be determined at evaluation time from head_dim
    return ScaledDotProductAttentionExpr<std::decay_t<Q>, std::decay_t<K>, std::decay_t<V>>{
        std::forward<Q>(q), std::forward<K>(k), std::forward<V>(v), 0.0f  // 0 = auto-compute
    };
}

/// Helper with explicit scale factor
template<typename Q, typename K, typename V>
constexpr auto make_attention(Q&& q, K&& k, V&& v, float scale) {
    return ScaledDotProductAttentionExpr<std::decay_t<Q>, std::decay_t<K>, std::decay_t<V>>{
        std::forward<Q>(q), std::forward<K>(k), std::forward<V>(v), scale
    };
}

/// Masked attention expression (for causal/autoregressive models)
/// Applies a causal mask before softmax: positions can only attend to earlier positions
template<typename Query, typename Key, typename Value>
struct CausalAttentionExpr : ExprBase {
    using query_type = Query;
    using key_type = Key;
    using value_type = Value;

    Query query;
    Key key;
    Value value;
    float scale;

    constexpr CausalAttentionExpr(Query q, Key k, Value v, float s)
        : query(std::move(q)), key(std::move(k)), value(std::move(v)), scale(s) {}
};

/// Helper to create Causal Attention expression
template<typename Q, typename K, typename V>
constexpr auto make_causal_attention(Q&& q, K&& k, V&& v) {
    return CausalAttentionExpr<std::decay_t<Q>, std::decay_t<K>, std::decay_t<V>>{
        std::forward<Q>(q), std::forward<K>(k), std::forward<V>(v), 0.0f
    };
}

template<typename Q, typename K, typename V>
constexpr auto make_causal_attention(Q&& q, K&& k, V&& v, float scale) {
    return CausalAttentionExpr<std::decay_t<Q>, std::decay_t<K>, std::decay_t<V>>{
        std::forward<Q>(q), std::forward<K>(k), std::forward<V>(v), scale
    };
}

} // namespace kernelix::expr
