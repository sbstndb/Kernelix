#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

namespace kernelix {

/// Sentinel value for dynamic dimensions
inline constexpr std::size_t Dynamic = std::numeric_limits<std::size_t>::max();

/// Check if a dimension is dynamic
constexpr bool is_dynamic(std::size_t dim) noexcept {
    return dim == Dynamic;
}

/// Type tags for operation categories
namespace op_category {
    struct Contraction {};
    struct Elementwise {};
    struct Reduction {};
    struct Normalization {};
}

/// Type list for metaprogramming
template<typename... Ts>
struct type_list {
    static constexpr std::size_t size = sizeof...(Ts);
};

/// Index sequence helpers
template<std::size_t... Is>
using index_sequence = std::index_sequence<Is...>;

template<std::size_t N>
using make_index_sequence = std::make_index_sequence<N>;

/// Common type aliases
using f32 = float;
using f64 = double;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

} // namespace kernelix
