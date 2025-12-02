#pragma once

#include <cstdlib>
#include <cstddef>
#include <memory>
#include <new>

namespace kernelix::memory {

/// Default alignment (cache line size)
constexpr std::size_t default_alignment = 64;

/// Aligned allocation
inline void* aligned_alloc(std::size_t size, std::size_t alignment = default_alignment) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        throw std::bad_alloc();
    }
    return ptr;
#endif
}

/// Aligned deallocation
inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

/// Aligned deleter for unique_ptr
struct AlignedDeleter {
    void operator()(void* ptr) const {
        aligned_free(ptr);
    }
};

/// Aligned unique_ptr type
template<typename T>
using aligned_ptr = std::unique_ptr<T[], AlignedDeleter>;

/// Create aligned array
template<typename T>
aligned_ptr<T> make_aligned(std::size_t count, std::size_t alignment = default_alignment) {
    void* ptr = aligned_alloc(count * sizeof(T), alignment);
    return aligned_ptr<T>(static_cast<T*>(ptr));
}

/// Prefetch data for read
inline void prefetch_read(const void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);  // Read, high temporal locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#endif
}

/// Prefetch data for write
inline void prefetch_write(const void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 1, 3);  // Write, high temporal locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#endif
}

/// Prefetch with specified distance
template<std::size_t Distance = 64>
inline void prefetch_ahead(const void* base, std::size_t offset) {
    prefetch_read(static_cast<const char*>(base) + offset + Distance);
}

/// Check if pointer is aligned
inline bool is_aligned(const void* ptr, std::size_t alignment = default_alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

} // namespace kernelix::memory
