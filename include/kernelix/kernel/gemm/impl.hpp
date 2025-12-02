#pragma once

#include "micro_kernel.hpp"
#include "pack.hpp"
#include "../../core/config.hpp"
#include "../../util/parallel.hpp"
#include "../../util/memory.hpp"
#include <algorithm>
#include <cstring>

namespace kernelix::kernel {

/// Tile sizes for cache blocking
template<typename T>
struct GemmTileConfig {
    // L2 cache blocking (256 KB typical)
    static constexpr std::size_t MC = 128;  // Tile height for A
    static constexpr std::size_t NC = 256;  // Tile width for B
    static constexpr std::size_t KC = 256;  // Tile depth (K dimension)

    // Micro-kernel sizes
    static constexpr std::size_t MR = MicroKernelConfig<T>::MR;
    static constexpr std::size_t NR = MicroKernelConfig<T>::NR;
};

/// Main GEMM implementation: C = A @ B
/// A: [M x K], B: [K x N], C: [M x N], all row-major
template<typename T>
void gemm_impl(
    const T* A, const T* B, T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc,
    bool accumulate = false  // If false, zero C first
) {
    using Cfg = GemmTileConfig<T>;

    // For very small matrices, use naive implementation
    if (M * N * K < 4096) {
        if (!accumulate) {
            std::memset(C, 0, M * ldc * sizeof(T));
        }
        GemmEdgeKernel<T>::run(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }

    // Zero output if not accumulating
    if (!accumulate) {
        parallel::parallel_for(0, M, [C, ldc, N](std::size_t i) {
            std::memset(C + i * ldc, 0, N * sizeof(T));
        });
    }

    // Allocate packing buffers (per-thread in parallel region)
    const std::size_t packed_a_sz = packed_a_size<Cfg::MR>(Cfg::MC, Cfg::KC);
    const std::size_t packed_b_sz = packed_b_size<Cfg::NR>(Cfg::KC, Cfg::NC);

    // Loop over K tiles (sequential - accumulation dependency)
    for (std::size_t k0 = 0; k0 < K; k0 += Cfg::KC) {
        const std::size_t kb = std::min(Cfg::KC, K - k0);

        // Loop over N tiles (can parallelize)
        for (std::size_t j0 = 0; j0 < N; j0 += Cfg::NC) {
            const std::size_t nb = std::min(Cfg::NC, N - j0);

            // Pack B panel once for this (k, j) tile
            auto packed_b = memory::make_aligned<T>(packed_b_sz);
            pack_b<Cfg::NR>(B + k0 * ldb + j0, packed_b.get(), kb, nb, ldb);

            // Loop over M tiles (parallelize)
            parallel::parallel_for(
                std::size_t{0}, (M + Cfg::MC - 1) / Cfg::MC,
                [&](std::size_t m_tile) {
                    const std::size_t i0 = m_tile * Cfg::MC;
                    const std::size_t mb = std::min(Cfg::MC, M - i0);

                    // Pack A panel for this (m, k) tile
                    auto packed_a = memory::make_aligned<T>(packed_a_sz);
                    pack_a<Cfg::MR>(A + i0 * lda + k0, packed_a.get(), mb, kb, lda);

                    // Macro-kernel: process MC x NC tile using micro-kernels
                    gemm_macro_kernel<T, Cfg::MR, Cfg::NR>(
                        packed_a.get(), packed_b.get(),
                        C + i0 * ldc + j0,
                        mb, nb, kb, ldc
                    );
                }
            );
        }
    }
}

/// Macro-kernel: process a MC x NC tile using MR x NR micro-kernels
template<typename T, std::size_t MR, std::size_t NR>
void gemm_macro_kernel(
    const T* packed_a,  // [MC/MR][K][MR] packed
    const T* packed_b,  // [NC/NR][K][NR] packed
    T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t ldc
) {
    const std::size_t m_panels = (M + MR - 1) / MR;
    const std::size_t n_panels = (N + NR - 1) / NR;

    for (std::size_t jp = 0; jp < n_panels; ++jp) {
        const std::size_t j = jp * NR;
        const std::size_t curr_nr = std::min(NR, N - j);
        const T* b_panel = packed_b + jp * NR * K;

        for (std::size_t ip = 0; ip < m_panels; ++ip) {
            const std::size_t i = ip * MR;
            const std::size_t curr_mr = std::min(MR, M - i);
            const T* a_panel = packed_a + ip * MR * K;

            if (curr_mr == MR && curr_nr == NR) {
                // Full micro-kernel
                GemmMicroKernel<MR, NR, T>::run(
                    a_panel, b_panel,
                    C + i * ldc + j,
                    K, ldc
                );
            } else {
                // Edge case - use scalar kernel
                // First unpack to temp buffer, then use edge kernel
                gemm_edge_case<T, MR, NR>(
                    a_panel, b_panel,
                    C + i * ldc + j,
                    curr_mr, curr_nr, K, ldc
                );
            }
        }
    }
}

/// Handle edge cases where tile doesn't fill MR x NR
template<typename T, std::size_t MR, std::size_t NR>
void gemm_edge_case(
    const T* packed_a,  // [K][MR] packed
    const T* packed_b,  // [K][NR] packed
    T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t ldc
) {
    // Simple scalar implementation for edge tiles
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T sum = C[i * ldc + j];
            for (std::size_t k = 0; k < K; ++k) {
                sum += packed_a[k * MR + i] * packed_b[k * NR + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

/// Naive GEMM for reference/testing: C = A @ B
template<typename T>
void gemm_naive(
    const T* A, const T* B, T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc
) {
    // Zero output
    for (std::size_t i = 0; i < M; ++i) {
        std::memset(C + i * ldc, 0, N * sizeof(T));
    }

    // Triple loop
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const T a_ik = A[i * lda + k];
            for (std::size_t j = 0; j < N; ++j) {
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
}

/// GEMM with bias: C = A @ B + bias
template<typename T>
void gemm_bias_impl(
    const T* A, const T* B, const T* bias, T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc
) {
    // First do GEMM
    gemm_impl(A, B, C, M, N, K, lda, ldb, ldc, false);

    // Then add bias (broadcast over rows)
    parallel::parallel_for(0, M, [C, bias, ldc, N](std::size_t i) {
        for (std::size_t j = 0; j < N; ++j) {
            C[i * ldc + j] += bias[j];
        }
    });
}

} // namespace kernelix::kernel
