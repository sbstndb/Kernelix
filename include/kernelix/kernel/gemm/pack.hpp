#pragma once

#include "../../util/memory.hpp"
#include <cstddef>
#include <algorithm>

namespace kernelix::kernel {

/// Pack panel of A into column-major micro-panels for micro-kernel
/// Input:  A[M x K] in row-major
/// Output: packed[K x MR] blocks, where each block has MR consecutive values
template<std::size_t MR, typename T>
void pack_a(
    const T* A,
    T* packed,
    std::size_t M,
    std::size_t K,
    std::size_t lda
) {
    const std::size_t num_panels = (M + MR - 1) / MR;

    for (std::size_t panel = 0; panel < num_panels; ++panel) {
        const std::size_t i_start = panel * MR;
        const std::size_t i_end = std::min(i_start + MR, M);
        const std::size_t panel_height = i_end - i_start;

        T* pack_ptr = packed + panel * MR * K;

        for (std::size_t k = 0; k < K; ++k) {
            // Pack MR elements from column k
            for (std::size_t i = 0; i < panel_height; ++i) {
                pack_ptr[k * MR + i] = A[(i_start + i) * lda + k];
            }
            // Zero-pad if panel is not full
            for (std::size_t i = panel_height; i < MR; ++i) {
                pack_ptr[k * MR + i] = T{0};
            }
        }
    }
}

/// Pack panel of B into row-major micro-panels for micro-kernel
/// Input:  B[K x N] in row-major
/// Output: packed[K x NR] blocks, where each block has NR consecutive values
template<std::size_t NR, typename T>
void pack_b(
    const T* B,
    T* packed,
    std::size_t K,
    std::size_t N,
    std::size_t ldb
) {
    const std::size_t num_panels = (N + NR - 1) / NR;

    for (std::size_t panel = 0; panel < num_panels; ++panel) {
        const std::size_t j_start = panel * NR;
        const std::size_t j_end = std::min(j_start + NR, N);
        const std::size_t panel_width = j_end - j_start;

        T* pack_ptr = packed + panel * NR * K;

        for (std::size_t k = 0; k < K; ++k) {
            // Pack NR elements from row k
            for (std::size_t j = 0; j < panel_width; ++j) {
                pack_ptr[k * NR + j] = B[k * ldb + j_start + j];
            }
            // Zero-pad if panel is not full
            for (std::size_t j = panel_width; j < NR; ++j) {
                pack_ptr[k * NR + j] = T{0};
            }
        }
    }
}

/// Calculate packed buffer size for A
template<std::size_t MR>
constexpr std::size_t packed_a_size(std::size_t M, std::size_t K) {
    const std::size_t num_panels = (M + MR - 1) / MR;
    return num_panels * MR * K;
}

/// Calculate packed buffer size for B
template<std::size_t NR>
constexpr std::size_t packed_b_size(std::size_t K, std::size_t N) {
    const std::size_t num_panels = (N + NR - 1) / NR;
    return num_panels * NR * K;
}

} // namespace kernelix::kernel
