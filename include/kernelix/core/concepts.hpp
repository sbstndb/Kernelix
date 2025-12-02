#pragma once

#include <concepts>
#include <iterator>
#include <type_traits>

namespace kernelix {

// Forward declarations
namespace expr {
    struct ExprBase;
}

/// Concept for numeric types
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

/// Concept for floating point types
template<typename T>
concept FloatingPoint = std::floating_point<T>;

/// Concept for tensor-like types
template<typename T>
concept TensorLike = requires(T t) {
    typename T::value_type;
    { t.data() } -> std::convertible_to<typename T::value_type*>;
    { t.size() } -> std::convertible_to<std::size_t>;
};

/// Concept for expression types
template<typename T>
concept ExprLike = std::is_base_of_v<expr::ExprBase, std::remove_cvref_t<T>>;

/// Concept for types that can be evaluated
template<typename T>
concept Evaluable = TensorLike<T> || ExprLike<T>;

/// Concept for contiguous memory
template<typename T>
concept ContiguousRange = requires(T t) {
    { t.data() } -> std::contiguous_iterator;
    { t.size() } -> std::convertible_to<std::size_t>;
};

} // namespace kernelix
