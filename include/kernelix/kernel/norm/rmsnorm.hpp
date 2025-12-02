#pragma once

#include "../../util/simd.hpp"
#include "../../util/parallel.hpp"
#include "../../util/memory.hpp"
#include <cmath>
#include <cstddef>

namespace kernelix::kernel {

/// RMSNorm implementation: y = x * weight / sqrt(mean(x^2) + eps)
/// Input shape: [batch, hidden_dim] or [seq_len, hidden_dim]
/// Weight shape: [hidden_dim]
/// Normalizes over the last dimension (hidden_dim)
template<typename T>
struct RMSNormKernel {

    /// Scalar implementation
    static void run_scalar(
        const T* input,
        const T* weight,
        T* output,
        std::size_t num_rows,
        std::size_t hidden_dim,
        T eps
    ) {
        parallel::parallel_for(0, num_rows, [=](std::size_t row) {
            const T* x = input + row * hidden_dim;
            T* y = output + row * hidden_dim;

            // Compute sum of squares
            T sum_sq = T{0};
            for (std::size_t i = 0; i < hidden_dim; ++i) {
                sum_sq += x[i] * x[i];
            }

            // RMS = sqrt(mean(x^2) + eps)
            T rms = std::sqrt(sum_sq / static_cast<T>(hidden_dim) + eps);
            T inv_rms = T{1} / rms;

            // Normalize and scale by weight
            for (std::size_t i = 0; i < hidden_dim; ++i) {
                y[i] = x[i] * inv_rms * weight[i];
            }
        });
    }

#if defined(KERNELIX_AVX2)
    /// AVX2 vectorized implementation
    static void run_avx2(
        const float* input,
        const float* weight,
        float* output,
        std::size_t num_rows,
        std::size_t hidden_dim,
        float eps
    ) {
        parallel::parallel_for(0, num_rows, [=](std::size_t row) {
            const float* x = input + row * hidden_dim;
            float* y = output + row * hidden_dim;

            // Compute sum of squares using AVX2
            __m256 sum_sq_vec = _mm256_setzero_ps();
            std::size_t i = 0;

            // Main vectorized loop
            for (; i + 8 <= hidden_dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                sum_sq_vec = _mm256_fmadd_ps(x_vec, x_vec, sum_sq_vec);
            }

            // Horizontal sum
            float sum_sq = simd::Vec<float>::hsum(sum_sq_vec);

            // Handle remainder
            for (; i < hidden_dim; ++i) {
                sum_sq += x[i] * x[i];
            }

            // RMS and inverse
            float rms = std::sqrt(sum_sq / static_cast<float>(hidden_dim) + eps);
            float inv_rms = 1.0f / rms;
            __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

            // Normalize and scale
            i = 0;
            for (; i + 8 <= hidden_dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 w_vec = _mm256_loadu_ps(weight + i);
                __m256 y_vec = _mm256_mul_ps(_mm256_mul_ps(x_vec, inv_rms_vec), w_vec);
                _mm256_storeu_ps(y + i, y_vec);
            }

            // Handle remainder
            for (; i < hidden_dim; ++i) {
                y[i] = x[i] * inv_rms * weight[i];
            }
        });
    }
#endif

    /// Dispatch to best available implementation
    static void run(
        const T* input,
        const T* weight,
        T* output,
        std::size_t num_rows,
        std::size_t hidden_dim,
        T eps
    ) {
#if defined(KERNELIX_AVX2)
        if constexpr (std::is_same_v<T, float>) {
            run_avx2(input, weight, output, num_rows, hidden_dim, eps);
            return;
        }
#endif
        run_scalar(input, weight, output, num_rows, hidden_dim, eps);
    }
};

/// Fused RMSNorm + Linear: y = rmsnorm(x) @ W
/// This is a common pattern in transformers
template<typename T>
struct FusedRMSNormLinearKernel {

    static void run(
        const T* input,       // [batch, hidden_in]
        const T* norm_weight, // [hidden_in]
        const T* linear_w,    // [hidden_in, hidden_out]
        T* output,            // [batch, hidden_out]
        std::size_t batch,
        std::size_t hidden_in,
        std::size_t hidden_out,
        T eps
    ) {
        // Allocate temporary for normalized input
        auto normed = memory::make_aligned<T>(batch * hidden_in);

        // First: RMSNorm
        RMSNormKernel<T>::run(input, norm_weight, normed.get(), batch, hidden_in, eps);

        // Then: Linear (GEMM)
        // This could be further optimized by fusing into the GEMM loop
        // For now, we do it in two steps
        parallel::parallel_for(0, batch, [&](std::size_t b) {
            const T* x = normed.get() + b * hidden_in;
            T* y = output + b * hidden_out;

            for (std::size_t j = 0; j < hidden_out; ++j) {
                T sum = T{0};
                for (std::size_t k = 0; k < hidden_in; ++k) {
                    sum += x[k] * linear_w[k * hidden_out + j];
                }
                y[j] = sum;
            }
        });
    }
};

} // namespace kernelix::kernel
