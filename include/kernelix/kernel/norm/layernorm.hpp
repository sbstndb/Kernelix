#pragma once

#include "../../util/simd.hpp"
#include "../../util/parallel.hpp"
#include <cmath>
#include <cstddef>

namespace kernelix::kernel {

/// LayerNorm implementation: y = (x - mean) / sqrt(var + eps) * gamma + beta
/// Input shape: [batch, hidden_dim] or [seq_len, hidden_dim]
/// Gamma, Beta shape: [hidden_dim]
/// Normalizes over the last dimension (hidden_dim)
template<typename T>
struct LayerNormKernel {

    /// Scalar implementation
    static void run_scalar(
        const T* input,
        const T* gamma,
        const T* beta,
        T* output,
        std::size_t num_rows,
        std::size_t hidden_dim,
        T eps
    ) {
        parallel::parallel_for(0, num_rows, [=](std::size_t row) {
            const T* x = input + row * hidden_dim;
            T* y = output + row * hidden_dim;

            // Compute mean
            T sum = T{0};
            for (std::size_t i = 0; i < hidden_dim; ++i) {
                sum += x[i];
            }
            T mean = sum / static_cast<T>(hidden_dim);

            // Compute variance
            T var_sum = T{0};
            for (std::size_t i = 0; i < hidden_dim; ++i) {
                T diff = x[i] - mean;
                var_sum += diff * diff;
            }
            T var = var_sum / static_cast<T>(hidden_dim);

            // Normalize: (x - mean) / sqrt(var + eps)
            T inv_std = T{1} / std::sqrt(var + eps);

            // Apply gamma and beta
            for (std::size_t i = 0; i < hidden_dim; ++i) {
                y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
            }
        });
    }

#if defined(KERNELIX_AVX2)
    /// AVX2 vectorized implementation
    static void run_avx2(
        const float* input,
        const float* gamma,
        const float* beta,
        float* output,
        std::size_t num_rows,
        std::size_t hidden_dim,
        float eps
    ) {
        parallel::parallel_for(0, num_rows, [=](std::size_t row) {
            const float* x = input + row * hidden_dim;
            float* y = output + row * hidden_dim;

            // Compute mean using AVX2
            __m256 sum_vec = _mm256_setzero_ps();
            std::size_t i = 0;

            for (; i + 8 <= hidden_dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                sum_vec = _mm256_add_ps(sum_vec, x_vec);
            }

            float sum = simd::Vec<float>::hsum(sum_vec);

            // Handle remainder for mean
            for (; i < hidden_dim; ++i) {
                sum += x[i];
            }

            float mean = sum / static_cast<float>(hidden_dim);
            __m256 mean_vec = _mm256_set1_ps(mean);

            // Compute variance using AVX2
            __m256 var_vec = _mm256_setzero_ps();
            i = 0;

            for (; i + 8 <= hidden_dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 diff = _mm256_sub_ps(x_vec, mean_vec);
                var_vec = _mm256_fmadd_ps(diff, diff, var_vec);
            }

            float var_sum = simd::Vec<float>::hsum(var_vec);

            // Handle remainder for variance
            for (; i < hidden_dim; ++i) {
                float diff = x[i] - mean;
                var_sum += diff * diff;
            }

            float var = var_sum / static_cast<float>(hidden_dim);
            float inv_std = 1.0f / std::sqrt(var + eps);
            __m256 inv_std_vec = _mm256_set1_ps(inv_std);

            // Normalize and apply gamma/beta
            i = 0;
            for (; i + 8 <= hidden_dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 gamma_vec = _mm256_loadu_ps(gamma + i);
                __m256 beta_vec = _mm256_loadu_ps(beta + i);

                // (x - mean) * inv_std
                __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x_vec, mean_vec), inv_std_vec);

                // normalized * gamma + beta
                __m256 y_vec = _mm256_fmadd_ps(normalized, gamma_vec, beta_vec);

                _mm256_storeu_ps(y + i, y_vec);
            }

            // Handle remainder
            for (; i < hidden_dim; ++i) {
                y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
            }
        });
    }
#endif

    /// Dispatch to best available implementation
    static void run(
        const T* input,
        const T* gamma,
        const T* beta,
        T* output,
        std::size_t num_rows,
        std::size_t hidden_dim,
        T eps
    ) {
#if defined(KERNELIX_AVX2)
        if constexpr (std::is_same_v<T, float>) {
            run_avx2(input, gamma, beta, output, num_rows, hidden_dim, eps);
            return;
        }
#endif
        run_scalar(input, gamma, beta, output, num_rows, hidden_dim, eps);
    }
};

/// Welford's online algorithm for numerically stable mean/variance
/// Useful for very long sequences or when precision is critical
template<typename T>
struct LayerNormWelfordKernel {

    static void run(
        const T* input,
        const T* gamma,
        const T* beta,
        T* output,
        std::size_t num_rows,
        std::size_t hidden_dim,
        T eps
    ) {
        parallel::parallel_for(0, num_rows, [=](std::size_t row) {
            const T* x = input + row * hidden_dim;
            T* y = output + row * hidden_dim;

            // Welford's algorithm for mean and variance in one pass
            T mean = T{0};
            T M2 = T{0};

            for (std::size_t i = 0; i < hidden_dim; ++i) {
                T delta = x[i] - mean;
                mean += delta / static_cast<T>(i + 1);
                T delta2 = x[i] - mean;
                M2 += delta * delta2;
            }

            T var = M2 / static_cast<T>(hidden_dim);
            T inv_std = T{1} / std::sqrt(var + eps);

            // Apply normalization
            for (std::size_t i = 0; i < hidden_dim; ++i) {
                y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
            }
        });
    }
};

} // namespace kernelix::kernel
