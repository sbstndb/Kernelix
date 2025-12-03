/// @file fused.hpp
/// @brief Fused GEMM + Activation kernels
///
/// Applies activation function during the final write phase of GEMM,
/// avoiding an extra memory pass for the activation.

#pragma once

#include "micro_kernel.hpp"
#include "pack.hpp"
#include "../../core/config.hpp"
#include "../../util/parallel.hpp"
#include "../../util/memory.hpp"
#include "../../util/simd.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace kernelix::kernel {

// =============================================================================
// Activation Functors
// =============================================================================

/// Identity activation (no-op) - for unfused GEMM
struct IdentityActivation {
    template<typename T>
    static T apply(T x) { return x; }

#if defined(KERNELIX_AVX2)
    static __m256 apply_avx(__m256 x) { return x; }
#endif
};

/// ReLU activation: max(0, x)
struct ReLUActivation {
    template<typename T>
    static T apply(T x) { return x > T{0} ? x : T{0}; }

#if defined(KERNELIX_AVX2)
    static __m256 apply_avx(__m256 x) {
        __m256 zero = _mm256_setzero_ps();
        return _mm256_max_ps(x, zero);
    }
#endif
};

/// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
struct SiLUActivation {
    template<typename T>
    static T apply(T x) {
        return x / (T{1} + std::exp(-x));
    }

#if defined(KERNELIX_AVX2)
    static __m256 apply_avx(__m256 x) {
        // Fast SiLU approximation using polynomial for sigmoid
        // sigmoid(x) ≈ 0.5 + 0.5 * tanh(x * 0.5)
        // But for accuracy, use exp approximation

        __m256 one = _mm256_set1_ps(1.0f);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);

        // exp(-x) approximation using polynomial
        // exp(x) ≈ (1 + x/256)^256, but we use a simpler fast exp
        // For production, use a proper fast_exp implementation

        // Clamp to avoid overflow
        neg_x = _mm256_max_ps(neg_x, _mm256_set1_ps(-88.0f));
        neg_x = _mm256_min_ps(neg_x, _mm256_set1_ps(88.0f));

        // Fast exp using Schraudolph's method (less accurate but fast)
        // exp(x) ≈ 2^(x / ln2) = 2^(x * 1.4426950408889634)
        __m256 exp_scale = _mm256_set1_ps(1.4426950408889634f * (1 << 23));
        __m256 exp_bias = _mm256_set1_ps(127.0f * (1 << 23));

        __m256 exp_neg_x = _mm256_castsi256_ps(
            _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(neg_x, exp_scale), exp_bias))
        );

        // sigmoid = 1 / (1 + exp(-x))
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));

        // SiLU = x * sigmoid(x)
        return _mm256_mul_ps(x, sigmoid);
    }
#endif
};

/// GELU activation (approximate): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
struct GELUActivation {
    template<typename T>
    static T apply(T x) {
        const T sqrt_2_over_pi = T{0.7978845608028654};
        const T coeff = T{0.044715};
        T x3 = x * x * x;
        T inner = sqrt_2_over_pi * (x + coeff * x3);
        return T{0.5} * x * (T{1} + std::tanh(inner));
    }

#if defined(KERNELIX_AVX2)
    static __m256 apply_avx(__m256 x) {
        // GELU approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        __m256 sqrt_2_over_pi = _mm256_set1_ps(0.7978845608028654f);
        __m256 coeff = _mm256_set1_ps(0.044715f);
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 one = _mm256_set1_ps(1.0f);

        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 inner = _mm256_mul_ps(sqrt_2_over_pi,
                        _mm256_add_ps(x, _mm256_mul_ps(coeff, x3)));

        // tanh approximation using exp
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        __m256 two_inner = _mm256_add_ps(inner, inner);

        // Fast exp approximation
        two_inner = _mm256_max_ps(two_inner, _mm256_set1_ps(-88.0f));
        two_inner = _mm256_min_ps(two_inner, _mm256_set1_ps(88.0f));
        __m256 exp_scale = _mm256_set1_ps(1.4426950408889634f * (1 << 23));
        __m256 exp_bias = _mm256_set1_ps(127.0f * (1 << 23));
        __m256 exp_2x = _mm256_castsi256_ps(
            _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(two_inner, exp_scale), exp_bias))
        );

        __m256 tanh_val = _mm256_div_ps(
            _mm256_sub_ps(exp_2x, one),
            _mm256_add_ps(exp_2x, one)
        );

        return _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_val)));
    }
#endif
};

// =============================================================================
// Fused Micro-Kernel Interface
// =============================================================================

/// Fused micro-kernel: computes C = activation(A @ B + C_prev) when is_final=true
template<std::size_t MR, std::size_t NR, typename T, typename Activation>
struct FusedGemmMicroKernel {
    static void run(
        const T* A, const T* B, T* C,
        std::size_t K,
        std::size_t ldc,
        bool is_final  // Apply activation only on final K tile
    );
};

// =============================================================================
// Scalar Fused Implementation
// =============================================================================

template<std::size_t MR, std::size_t NR, typename Activation>
struct FusedGemmMicroKernel<MR, NR, float, Activation> {
    static void run(
        const float* A, const float* B, float* C,
        std::size_t K,
        std::size_t ldc,
        bool is_final
    ) {
        float acc[MR][NR] = {};

        // Load existing C values
        for (std::size_t i = 0; i < MR; ++i) {
            for (std::size_t j = 0; j < NR; ++j) {
                acc[i][j] = C[i * ldc + j];
            }
        }

        // Main computation
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t i = 0; i < MR; ++i) {
                const float a_val = A[k * MR + i];
                for (std::size_t j = 0; j < NR; ++j) {
                    acc[i][j] += a_val * B[k * NR + j];
                }
            }
        }

        // Store with optional activation
        if (is_final) {
            for (std::size_t i = 0; i < MR; ++i) {
                for (std::size_t j = 0; j < NR; ++j) {
                    C[i * ldc + j] = Activation::apply(acc[i][j]);
                }
            }
        } else {
            for (std::size_t i = 0; i < MR; ++i) {
                for (std::size_t j = 0; j < NR; ++j) {
                    C[i * ldc + j] = acc[i][j];
                }
            }
        }
    }
};

// =============================================================================
// AVX2 Fused Implementation (6x16)
// =============================================================================

#if defined(KERNELIX_AVX2)

template<typename Activation>
struct FusedGemmMicroKernel<6, 16, float, Activation> {
    static void run(
        const float* A, const float* B, float* C,
        std::size_t K,
        std::size_t ldc,
        bool is_final
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
            __m256 b0 = _mm256_loadu_ps(B + k * 16 + 0);
            __m256 b1 = _mm256_loadu_ps(B + k * 16 + 8);

            __m256 a0 = _mm256_set1_ps(A[k * 6 + 0]);
            __m256 a1 = _mm256_set1_ps(A[k * 6 + 1]);
            __m256 a2 = _mm256_set1_ps(A[k * 6 + 2]);
            __m256 a3 = _mm256_set1_ps(A[k * 6 + 3]);
            __m256 a4 = _mm256_set1_ps(A[k * 6 + 4]);
            __m256 a5 = _mm256_set1_ps(A[k * 6 + 5]);

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

        // Apply activation if this is the final K tile
        if (is_final) {
            c00 = Activation::apply_avx(c00);
            c01 = Activation::apply_avx(c01);
            c10 = Activation::apply_avx(c10);
            c11 = Activation::apply_avx(c11);
            c20 = Activation::apply_avx(c20);
            c21 = Activation::apply_avx(c21);
            c30 = Activation::apply_avx(c30);
            c31 = Activation::apply_avx(c31);
            c40 = Activation::apply_avx(c40);
            c41 = Activation::apply_avx(c41);
            c50 = Activation::apply_avx(c50);
            c51 = Activation::apply_avx(c51);
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

// =============================================================================
// Fused Macro-Kernel
// =============================================================================

template<typename T, std::size_t MR, std::size_t NR, typename Activation>
void fused_gemm_macro_kernel(
    const T* packed_a,
    const T* packed_b,
    T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t ldc,
    bool is_final
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
                // Full micro-kernel with fusion
                FusedGemmMicroKernel<MR, NR, T, Activation>::run(
                    a_panel, b_panel,
                    C + i * ldc + j,
                    K, ldc, is_final
                );
            } else {
                // Edge case - scalar with activation
                for (std::size_t ii = 0; ii < curr_mr; ++ii) {
                    for (std::size_t jj = 0; jj < curr_nr; ++jj) {
                        T sum = C[(i + ii) * ldc + (j + jj)];
                        for (std::size_t k = 0; k < K; ++k) {
                            sum += a_panel[k * MR + ii] * b_panel[k * NR + jj];
                        }
                        if (is_final) {
                            sum = Activation::apply(sum);
                        }
                        C[(i + ii) * ldc + (j + jj)] = sum;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Main Fused GEMM Implementation
// =============================================================================

/// Fused GEMM + Activation: C = activation(A @ B)
/// Activation is applied during the final write, avoiding extra memory pass
template<typename T, typename Activation>
void gemm_activation_impl(
    const T* A, const T* B, T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc
) {
    using Cfg = GemmTileConfig<T>;

    // For very small matrices, use simple implementation
    if (M * N * K < 4096) {
        std::memset(C, 0, M * ldc * sizeof(T));
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                T sum = T{0};
                for (std::size_t k = 0; k < K; ++k) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                C[i * ldc + j] = Activation::apply(sum);
            }
        }
        return;
    }

    // Zero output
    parallel::parallel_for(0, M, [C, ldc, N](std::size_t i) {
        std::memset(C + i * ldc, 0, N * sizeof(T));
    });

    // Packing buffer sizes
    const std::size_t packed_a_sz = packed_a_size<Cfg::MR>(Cfg::MC, Cfg::KC);
    const std::size_t packed_b_sz = packed_b_size<Cfg::NR>(Cfg::KC, Cfg::NC);

    // Calculate total K tiles
    const std::size_t num_k_tiles = (K + Cfg::KC - 1) / Cfg::KC;

    // Loop over K tiles
    for (std::size_t k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const std::size_t k0 = k_tile * Cfg::KC;
        const std::size_t kb = std::min(Cfg::KC, K - k0);
        const bool is_final_k = (k_tile == num_k_tiles - 1);

        // Loop over N tiles
        for (std::size_t j0 = 0; j0 < N; j0 += Cfg::NC) {
            const std::size_t nb = std::min(Cfg::NC, N - j0);

            // Pack B panel
            auto packed_b = memory::make_aligned<T>(packed_b_sz);
            pack_b<Cfg::NR>(B + k0 * ldb + j0, packed_b.get(), kb, nb, ldb);

            // Loop over M tiles (parallel)
            parallel::parallel_for(
                std::size_t{0}, (M + Cfg::MC - 1) / Cfg::MC,
                [&](std::size_t m_tile) {
                    const std::size_t i0 = m_tile * Cfg::MC;
                    const std::size_t mb = std::min(Cfg::MC, M - i0);

                    // Pack A panel
                    auto packed_a = memory::make_aligned<T>(packed_a_sz);
                    pack_a<Cfg::MR>(A + i0 * lda + k0, packed_a.get(), mb, kb, lda);

                    // Fused macro-kernel
                    fused_gemm_macro_kernel<T, Cfg::MR, Cfg::NR, Activation>(
                        packed_a.get(), packed_b.get(),
                        C + i0 * ldc + j0,
                        mb, nb, kb, ldc,
                        is_final_k
                    );
                }
            );
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// GEMM + ReLU: C = relu(A @ B)
template<typename T>
void gemm_relu_impl(
    const T* A, const T* B, T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc
) {
    gemm_activation_impl<T, ReLUActivation>(A, B, C, M, N, K, lda, ldb, ldc);
}

/// GEMM + SiLU: C = silu(A @ B)
template<typename T>
void gemm_silu_impl(
    const T* A, const T* B, T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc
) {
    gemm_activation_impl<T, SiLUActivation>(A, B, C, M, N, K, lda, ldb, ldc);
}

/// GEMM + GELU: C = gelu(A @ B)
template<typename T>
void gemm_gelu_impl(
    const T* A, const T* B, T* C,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc
) {
    gemm_activation_impl<T, GELUActivation>(A, B, C, M, N, K, lda, ldb, ldc);
}

} // namespace kernelix::kernel
