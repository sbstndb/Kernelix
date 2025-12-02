#pragma once

#include "../../util/simd.hpp"
#include <cstddef>

namespace kernelix::kernel {

/// Micro-kernel configuration
template<typename T>
struct MicroKernelConfig {
    // Default tile sizes for micro-kernel (fits in registers)
    static constexpr std::size_t MR = 6;   // Rows processed per micro-kernel call
    static constexpr std::size_t NR = 16;  // Columns processed (2x AVX-256 width for float)
};

/// Generic micro-kernel interface
template<std::size_t MR, std::size_t NR, typename T>
struct GemmMicroKernel {
    /// Compute C[MR x NR] += A[MR x K] @ B[K x NR]
    /// A is packed column-major (MR consecutive), B is packed row-major (NR consecutive)
    static void run(
        const T* A, const T* B, T* C,
        std::size_t K,
        std::size_t ldc
    );
};

// ============================================================================
// Scalar fallback implementation
// ============================================================================

template<std::size_t MR, std::size_t NR>
struct GemmMicroKernel<MR, NR, float> {
    static void run(
        const float* A, const float* B, float* C,
        std::size_t K,
        std::size_t ldc
    ) {
        // Accumulator registers (would be in actual registers with SIMD)
        float acc[MR][NR] = {};

        // Load existing C values
        for (std::size_t i = 0; i < MR; ++i) {
            for (std::size_t j = 0; j < NR; ++j) {
                acc[i][j] = C[i * ldc + j];
            }
        }

        // Main computation loop
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t i = 0; i < MR; ++i) {
                const float a_val = A[k * MR + i];  // Packed format
                for (std::size_t j = 0; j < NR; ++j) {
                    acc[i][j] += a_val * B[k * NR + j];  // Packed format
                }
            }
        }

        // Store back to C
        for (std::size_t i = 0; i < MR; ++i) {
            for (std::size_t j = 0; j < NR; ++j) {
                C[i * ldc + j] = acc[i][j];
            }
        }
    }
};

// ============================================================================
// AVX2 optimized micro-kernel (6x16 for float)
// ============================================================================

#if defined(KERNELIX_AVX2)

/// Specialized 6x16 micro-kernel using AVX2
/// Uses 12 YMM registers for accumulators (6 rows x 2 vectors of 8 floats)
template<>
struct GemmMicroKernel<6, 16, float> {
    static void run(
        const float* A, const float* B, float* C,
        std::size_t K,
        std::size_t ldc
    ) {
        // 12 accumulator registers (6 rows x 2 AVX vectors)
        __m256 c00 = _mm256_loadu_ps(C + 0 * ldc + 0);
        __m256 c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
        __m256 c10 = _mm256_loadu_ps(C + 1 * ldc + 0);
        __m256 c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
        __m256 c20 = _mm256_loadu_ps(C + 2 * ldc + 0);
        __m256 c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
        __m256 c30 = _mm256_loadu_ps(C + 3 * ldc + 0);
        __m256 c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
        __m256 c40 = _mm256_loadu_ps(C + 4 * ldc + 0);
        __m256 c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
        __m256 c50 = _mm256_loadu_ps(C + 5 * ldc + 0);
        __m256 c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

        for (std::size_t k = 0; k < K; ++k) {
            // Load B row (16 elements = 2 AVX vectors)
            __m256 b0 = _mm256_loadu_ps(B + k * 16 + 0);
            __m256 b1 = _mm256_loadu_ps(B + k * 16 + 8);

            // Load A column (6 elements) and broadcast each
            __m256 a0 = _mm256_set1_ps(A[k * 6 + 0]);
            __m256 a1 = _mm256_set1_ps(A[k * 6 + 1]);
            __m256 a2 = _mm256_set1_ps(A[k * 6 + 2]);
            __m256 a3 = _mm256_set1_ps(A[k * 6 + 3]);
            __m256 a4 = _mm256_set1_ps(A[k * 6 + 4]);
            __m256 a5 = _mm256_set1_ps(A[k * 6 + 5]);

            // FMA: c[i] += a[i] * b
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);
        }

        // Store results
        _mm256_storeu_ps(C + 0 * ldc + 0, c00);
        _mm256_storeu_ps(C + 0 * ldc + 8, c01);
        _mm256_storeu_ps(C + 1 * ldc + 0, c10);
        _mm256_storeu_ps(C + 1 * ldc + 8, c11);
        _mm256_storeu_ps(C + 2 * ldc + 0, c20);
        _mm256_storeu_ps(C + 2 * ldc + 8, c21);
        _mm256_storeu_ps(C + 3 * ldc + 0, c30);
        _mm256_storeu_ps(C + 3 * ldc + 8, c31);
        _mm256_storeu_ps(C + 4 * ldc + 0, c40);
        _mm256_storeu_ps(C + 4 * ldc + 8, c41);
        _mm256_storeu_ps(C + 5 * ldc + 0, c50);
        _mm256_storeu_ps(C + 5 * ldc + 8, c51);
    }
};

#endif // KERNELIX_AVX2

// ============================================================================
// Edge case micro-kernels (for cleanup)
// ============================================================================

/// Generic edge-case handler for partial tiles
template<typename T>
struct GemmEdgeKernel {
    static void run(
        const T* A, const T* B, T* C,
        std::size_t M, std::size_t N, std::size_t K,
        std::size_t lda, std::size_t ldb, std::size_t ldc
    ) {
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                T sum = C[i * ldc + j];
                for (std::size_t k = 0; k < K; ++k) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                C[i * ldc + j] = sum;
            }
        }
    }
};

} // namespace kernelix::kernel
