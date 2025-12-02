#pragma once

#include "types.hpp"
#include "concepts.hpp"
#include <array>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace kernelix {

/// Static tensor with compile-time dimensions
template<typename T, std::size_t... Dims>
class Tensor {
public:
    using value_type = T;
    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::array<std::size_t, rank> static_dims = {Dims...};
    static constexpr std::size_t static_size = (Dims * ...);

    /// Default constructor - zero initialized
    Tensor() : data_{} {}

    /// Construct from data pointer (copy)
    explicit Tensor(const T* src) {
        std::copy(src, src + static_size, data_.data());
    }

    /// Construct from initializer list
    Tensor(std::initializer_list<T> init) {
        if (init.size() != static_size) {
            throw std::invalid_argument("Initializer list size mismatch");
        }
        std::copy(init.begin(), init.end(), data_.data());
    }

    // Accessors
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    static constexpr std::size_t size() noexcept { return static_size; }
    static constexpr std::array<std::size_t, rank> shape() noexcept { return static_dims; }

    // Element access
    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    template<typename... Indices>
    T& operator()(Indices... indices) noexcept {
        static_assert(sizeof...(Indices) == rank, "Wrong number of indices");
        return data_[linear_index(indices...)];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == rank, "Wrong number of indices");
        return data_[linear_index(indices...)];
    }

    // Iterator support
    auto begin() noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }

    // Fill with value
    void fill(T value) noexcept {
        std::fill(data_.begin(), data_.end(), value);
    }

private:
    template<typename... Indices>
    static constexpr std::size_t linear_index(Indices... indices) noexcept {
        std::array<std::size_t, rank> idx = {static_cast<std::size_t>(indices)...};
        std::size_t result = 0;
        std::size_t stride = 1;
        for (std::size_t i = rank; i-- > 0;) {
            result += idx[i] * stride;
            stride *= static_dims[i];
        }
        return result;
    }

    alignas(64) std::array<T, static_size> data_;
};

/// Dynamic tensor with runtime dimensions
template<typename T, std::size_t Rank>
class DynamicTensor {
public:
    using value_type = T;
    static constexpr std::size_t rank = Rank;

    /// Construct with given dimensions
    explicit DynamicTensor(std::array<std::size_t, Rank> dims)
        : dims_(dims)
        , size_(std::accumulate(dims.begin(), dims.end(), std::size_t{1}, std::multiplies<>{}))
        , data_(size_)
    {}

    /// Construct from data pointer (copy)
    DynamicTensor(std::array<std::size_t, Rank> dims, const T* src)
        : dims_(dims)
        , size_(std::accumulate(dims.begin(), dims.end(), std::size_t{1}, std::multiplies<>{}))
        , data_(src, src + size_)
    {}

    // Accessors
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    std::size_t size() const noexcept { return size_; }
    std::array<std::size_t, Rank> shape() const noexcept { return dims_; }
    std::size_t dim(std::size_t i) const noexcept { return dims_[i]; }

    // Element access
    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    // Fill with value
    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Resize
    void resize(std::array<std::size_t, Rank> new_dims) {
        dims_ = new_dims;
        size_ = std::accumulate(dims_.begin(), dims_.end(), std::size_t{1}, std::multiplies<>{});
        data_.resize(size_);
    }

private:
    std::array<std::size_t, Rank> dims_;
    std::size_t size_;
    std::vector<T> data_;
};

/// Non-owning view over tensor data
template<typename T, std::size_t... Dims>
class TensorView {
public:
    using value_type = T;
    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::array<std::size_t, rank> static_dims = {Dims...};
    static constexpr std::size_t static_size = (Dims * ...);

    /// Construct from raw pointer
    explicit TensorView(T* ptr) noexcept : data_(ptr) {}

    /// Construct from Tensor
    explicit TensorView(Tensor<std::remove_const_t<T>, Dims...>& t) noexcept
        : data_(t.data()) {}

    // Accessors
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    static constexpr std::size_t size() noexcept { return static_size; }
    static constexpr std::array<std::size_t, rank> shape() noexcept { return static_dims; }

    // Element access
    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

private:
    T* data_;
};

} // namespace kernelix
