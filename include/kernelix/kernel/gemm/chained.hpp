/// @file chained.hpp
/// @brief Chained GEMM: Result = A @ B @ C without full intermediate materialization
///
/// Instead of computing Temp = A @ B (full M × K2 matrix) then Result = Temp @ C,
/// we compute panels of (A@B) that fit in L2 cache and immediately use them.
///
/// Memory traffic comparison for LLaMA MLP (batch=16384, hidden=4096, intermediate=11008):
///   Naive:  720 MB intermediate (2× read/write)
///   Fused:  256 KB buffer reused in L2 cache

#pragma once

#include "impl.hpp"
#include "../../util/memory.hpp"
#include "../../util/parallel.hpp"
#include <algorithm>
#include <cstring>

namespace kernelix::kernel {

/// Configuration for chained GEMM
template<typename T>
struct ChainedGemmConfig {
    // Reuse standard GEMM tile sizes
    using GemmCfg = GemmTileConfig<T>;

    // Panel size for intermediate (A@B) - must fit in L2
    static constexpr std::size_t MC = GemmCfg::MC;   // 128
    static constexpr std::size_t KC2 = GemmCfg::KC;  // 256 - tile size on K2 dimension
    static constexpr std::size_t NC = GemmCfg::NC;   // 256

    // Buffer size: MC × KC2 ≈ 128 KB for float (fits in L2)
    static constexpr std::size_t panel_size = MC * KC2;
};

/// Chained GEMM: Result = A @ B @ C
///
/// @param A      Input matrix [M × K1], row-major
/// @param B      Input matrix [K1 × K2], row-major
/// @param C      Input matrix [K2 × N], row-major
/// @param Result Output matrix [M × N], row-major
/// @param M      Rows of A and Result
/// @param K1     Cols of A, Rows of B
/// @param K2     Cols of B, Rows of C (intermediate dimension)
/// @param N      Cols of C and Result
/// @param lda    Leading dimension of A
/// @param ldb    Leading dimension of B
/// @param ldc    Leading dimension of C
/// @param ldr    Leading dimension of Result
///
/// Algorithm:
///   for each output tile Result[i0:MC, j0:NC]:
///     for each K2 tile [k2_0:KC2]:
///       ab_panel[MC × KC2] = A[i0:MC, :] @ B[:, k2_0:KC2]   // In L2 cache
///       Result[i0:MC, j0:NC] += ab_panel @ C[k2_0:KC2, j0:NC]
///
template<typename T>
void gemm_chain_impl(
    const T* A, const T* B, const T* C, T* Result,
    std::size_t M, std::size_t K1, std::size_t K2, std::size_t N,
    std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t ldr
) {
    using Cfg = ChainedGemmConfig<T>;

    // For very small matrices, use naive approach
    if (M * K1 * K2 + K2 * N < 8192) {
        // Allocate full intermediate
        auto temp = memory::make_aligned<T>(M * K2);
        gemm_impl(A, B, temp.get(), M, K2, K1, lda, ldb, K2, false);
        gemm_impl(temp.get(), C, Result, M, N, K2, K2, ldc, ldr, false);
        return;
    }

    // Zero the result matrix
    parallel::parallel_for(0, M, [Result, ldr, N](std::size_t i) {
        std::memset(Result + i * ldr, 0, N * sizeof(T));
    });

    // Loop over output column tiles
    for (std::size_t j0 = 0; j0 < N; j0 += Cfg::NC) {
        const std::size_t nb = std::min(Cfg::NC, N - j0);

        // Parallelize over row tiles
        const std::size_t num_m_tiles = (M + Cfg::MC - 1) / Cfg::MC;

        parallel::parallel_for(
            std::size_t{0}, num_m_tiles,
            [&](std::size_t m_tile) {
                const std::size_t i0 = m_tile * Cfg::MC;
                const std::size_t mb = std::min(Cfg::MC, M - i0);

                // Thread-local buffer for (A@B) panel
                // Size: MC × KC2 - fits in L2 cache!
                auto ab_panel = memory::make_aligned<T>(Cfg::panel_size);

                // Loop over K2 dimension (intermediate)
                for (std::size_t k2_0 = 0; k2_0 < K2; k2_0 += Cfg::KC2) {
                    const std::size_t kb2 = std::min(Cfg::KC2, K2 - k2_0);

                    // Step 1: Compute panel of (A @ B)
                    // ab_panel[mb × kb2] = A[i0:i0+mb, :] @ B[:, k2_0:k2_0+kb2]
                    gemm_impl(
                        A + i0 * lda,           // A starting at row i0
                        B + k2_0,               // B starting at column k2_0
                        ab_panel.get(),         // Output to local buffer
                        mb, kb2, K1,            // Dimensions: mb × kb2, reduction on K1
                        lda, ldb, kb2,          // Strides (ab_panel is tightly packed)
                        false                   // Don't accumulate - fresh computation
                    );

                    // Step 2: Accumulate into Result
                    // Result[i0:i0+mb, j0:j0+nb] += ab_panel @ C[k2_0:k2_0+kb2, j0:j0+nb]
                    gemm_impl(
                        ab_panel.get(),         // Panel computed above
                        C + k2_0 * ldc + j0,    // C starting at (k2_0, j0)
                        Result + i0 * ldr + j0, // Result tile
                        mb, nb, kb2,            // Dimensions: mb × nb, reduction on kb2
                        kb2, ldc, ldr,          // Strides
                        true                    // ACCUMULATE into Result
                    );
                }
            }
        );
    }
}

/// Chained GEMM with alternative association: Result = A @ (B @ C)
/// Use when K1 × N < M × K2 (smaller intermediate)
template<typename T>
void gemm_chain_right_impl(
    const T* A, const T* B, const T* C, T* Result,
    std::size_t M, std::size_t K1, std::size_t K2, std::size_t N,
    std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t ldr
) {
    using Cfg = ChainedGemmConfig<T>;

    // For very small matrices, use naive approach
    if (K1 * K2 * N + M * K1 < 8192) {
        auto temp = memory::make_aligned<T>(K1 * N);
        gemm_impl(B, C, temp.get(), K1, N, K2, ldb, ldc, N, false);
        gemm_impl(A, temp.get(), Result, M, N, K1, lda, N, ldr, false);
        return;
    }

    // Zero the result matrix
    parallel::parallel_for(0, M, [Result, ldr, N](std::size_t i) {
        std::memset(Result + i * ldr, 0, N * sizeof(T));
    });

    // Panel size for (B @ C): KC1 × NC
    constexpr std::size_t KC1 = Cfg::KC2;  // Tile on K1 dimension

    // Loop over output column tiles
    for (std::size_t j0 = 0; j0 < N; j0 += Cfg::NC) {
        const std::size_t nb = std::min(Cfg::NC, N - j0);

        // Loop over K1 dimension
        for (std::size_t k1_0 = 0; k1_0 < K1; k1_0 += KC1) {
            const std::size_t kb1 = std::min(KC1, K1 - k1_0);

            // Compute panel of (B @ C)
            // bc_panel[kb1 × nb] = B[k1_0:k1_0+kb1, :] @ C[:, j0:j0+nb]
            auto bc_panel = memory::make_aligned<T>(KC1 * Cfg::NC);
            gemm_impl(
                B + k1_0 * ldb,
                C + j0,
                bc_panel.get(),
                kb1, nb, K2,
                ldb, ldc, nb,
                false
            );

            // Accumulate: Result[:, j0:j0+nb] += A[:, k1_0:k1_0+kb1] @ bc_panel
            parallel::parallel_for(
                std::size_t{0}, (M + Cfg::MC - 1) / Cfg::MC,
                [&](std::size_t m_tile) {
                    const std::size_t i0 = m_tile * Cfg::MC;
                    const std::size_t mb = std::min(Cfg::MC, M - i0);

                    gemm_impl(
                        A + i0 * lda + k1_0,
                        bc_panel.get(),
                        Result + i0 * ldr + j0,
                        mb, nb, kb1,
                        lda, nb, ldr,
                        true  // Accumulate
                    );
                }
            );
        }
    }
}

/// Auto-select best association based on intermediate size
template<typename T>
void gemm_chain_auto_impl(
    const T* A, const T* B, const T* C, T* Result,
    std::size_t M, std::size_t K1, std::size_t K2, std::size_t N,
    std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t ldr
) {
    // Compare intermediate sizes:
    // Left association (A @ B) @ C: intermediate is M × K2
    // Right association A @ (B @ C): intermediate is K1 × N

    const std::size_t left_size = M * K2;
    const std::size_t right_size = K1 * N;

    if (right_size < left_size) {
        gemm_chain_right_impl(A, B, C, Result, M, K1, K2, N, lda, ldb, ldc, ldr);
    } else {
        gemm_chain_impl(A, B, C, Result, M, K1, K2, N, lda, ldb, ldc, ldr);
    }
}

/// Naive chained GEMM for comparison (full intermediate materialization)
template<typename T>
void gemm_chain_naive(
    const T* A, const T* B, const T* C, T* Result,
    std::size_t M, std::size_t K1, std::size_t K2, std::size_t N,
    std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t ldr
) {
    // Allocate full intermediate: M × K2
    auto temp = memory::make_aligned<T>(M * K2);

    // Step 1: Temp = A @ B
    gemm_impl(A, B, temp.get(), M, K2, K1, lda, ldb, K2, false);

    // Step 2: Result = Temp @ C
    gemm_impl(temp.get(), C, Result, M, N, K2, K2, ldc, ldr, false);
}

} // namespace kernelix::kernel
