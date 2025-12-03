#pragma once

#include "base.hpp"
#include <utility>
#include <cmath>

namespace kernelix::expr {

/// Rotary Position Embedding expression
/// Applies rotary embeddings to input tensor based on position
/// For each pair (2i, 2i+1):
///   x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
///   x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
/// Where θ = position / (base^(2i/d))
template<typename Input, typename CosSin>
struct RoPEExpr : ExprBase {
    using input_type = Input;
    using cossin_type = CosSin;

    Input input;      // [seq_len, head_dim] or [batch, seq_len, head_dim]
    CosSin cos_cache; // [max_seq_len, head_dim/2]
    CosSin sin_cache; // [max_seq_len, head_dim/2]
    std::size_t position_offset;  // For KV cache continuation

    constexpr RoPEExpr(Input in, CosSin cos, CosSin sin, std::size_t pos_offset = 0)
        : input(std::move(in)), cos_cache(std::move(cos)), sin_cache(std::move(sin)),
          position_offset(pos_offset) {}
};

/// In-place RoPE expression (when cos/sin are provided per-token)
/// Input: [seq_len, head_dim], Cos/Sin: [seq_len, head_dim/2]
template<typename Input, typename CosSin>
constexpr auto make_rope(Input&& input, CosSin&& cos_cache, CosSin&& sin_cache,
                         std::size_t position_offset = 0) {
    return RoPEExpr<std::decay_t<Input>, std::decay_t<CosSin>>(
        std::forward<Input>(input),
        std::forward<CosSin>(cos_cache),
        std::forward<CosSin>(sin_cache),
        position_offset
    );
}

} // namespace kernelix::expr
