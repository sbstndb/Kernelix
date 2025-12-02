#pragma once

#include "../../util/simd.hpp"
#include "../../util/parallel.hpp"
#include <cmath>
#include <cstddef>
#include <limits>

namespace kernelix::kernel {

/// Softmax implementation: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
/// Numerically stable version using max subtraction
/// Input shape: [num_rows, softmax_dim] - applies softmax over the last dimension
template<typename T>
struct SoftmaxKernel {

    /// Scalar implementation
    static void run_scalar(
        const T* input,
        T* output,
        std::size_t num_rows,
        std::size_t softmax_dim
    ) {
        parallel::parallel_for(0, num_rows, [=](std::size_t row) {
            const T* x = input + row * softmax_dim;
            T* y = output + row * softmax_dim;

            // Step 1: Find max for numerical stability
            T max_val = std::numeric_limits<T>::lowest();
            for (std::size_t i = 0; i < softmax_dim; ++i) {
                max_val = std::max(max_val, x[i]);
            }

            // Step 2: Compute exp(x_i - max) and sum
            T sum = T{0};
            for (std::size_t i = 0; i < softmax_dim; ++i) {
                y[i] = std::exp(x[i] - max_val);
                sum += y[i];
            }

            // Step 3: Normalize by sum
            T inv_sum = T{1} / sum;
            for (std::size_t i = 0; i < softmax_dim; ++i) {
                y[i] *= inv_sum;
            }
        });
    }

#if defined(KERNELIX_AVX2)
    /// AVX2 vectorized implementation
    static void run_avx2(
        const float* input,
        float* output,
        std::size_t num_rows,
        std::size_t softmax_dim
    ) {
        parallel::parallel_for(0, num_rows, [=](std::size_t row) {
            const float* x = input + row * softmax_dim;
            float* y = output + row * softmax_dim;

            // Step 1: Find max using SIMD
            __m256 max_vec = _mm256_set1_ps(std::numeric_limits<float>::lowest());
            std::size_t i = 0;

            for (; i + 8 <= softmax_dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                max_vec = _mm256_max_ps(max_vec, x_vec);
            }

            // Horizontal max
            float max_val = simd::Vec<float>::hmax(max_vec);

            // Handle remainder for max
            for (; i < softmax_dim; ++i) {
                max_val = std::max(max_val, x[i]);
            }

            __m256 max_broadcast = _mm256_set1_ps(max_val);

            // Step 2: Compute exp(x - max) and sum using SIMD
            // Note: Using fast approximation for exp or calling std::exp element-wise
            __m256 sum_vec = _mm256_setzero_ps();
            i = 0;

            for (; i + 8 <= softmax_dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 shifted = _mm256_sub_ps(x_vec, max_broadcast);

                // Compute exp element-wise (no native AVX2 exp instruction)
                // Store shifted values, compute exp, and accumulate
                alignas(32) float shifted_arr[8];
                alignas(32) float exp_arr[8];
                _mm256_store_ps(shifted_arr, shifted);

                for (int j = 0; j < 8; ++j) {
                    exp_arr[j] = std::exp(shifted_arr[j]);
                }

                __m256 exp_vec = _mm256_load_ps(exp_arr);
                _mm256_storeu_ps(y + i, exp_vec);
                sum_vec = _mm256_add_ps(sum_vec, exp_vec);
            }

            float sum = simd::Vec<float>::hsum(sum_vec);

            // Handle remainder
            for (; i < softmax_dim; ++i) {
                y[i] = std::exp(x[i] - max_val);
                sum += y[i];
            }

            // Step 3: Normalize by sum
            float inv_sum = 1.0f / sum;
            __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
            i = 0;

            for (; i + 8 <= softmax_dim; i += 8) {
                __m256 y_vec = _mm256_loadu_ps(y + i);
                y_vec = _mm256_mul_ps(y_vec, inv_sum_vec);
                _mm256_storeu_ps(y + i, y_vec);
            }

            // Handle remainder
            for (; i < softmax_dim; ++i) {
                y[i] *= inv_sum;
            }
        });
    }
#endif

#if defined(KERNELIX_AVX512)
    /// AVX-512 vectorized implementation
    static void run_avx512(
        const float* input,
        float* output,
        std::size_t num_rows,
        std::size_t softmax_dim
    ) {
        parallel::parallel_for(0, num_rows, [=](std::size_t row) {
            const float* x = input + row * softmax_dim;
            float* y = output + row * softmax_dim;

            // Step 1: Find max using SIMD
            __m512 max_vec = _mm512_set1_ps(std::numeric_limits<float>::lowest());
            std::size_t i = 0;

            for (; i + 16 <= softmax_dim; i += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + i);
                max_vec = _mm512_max_ps(max_vec, x_vec);
            }

            // Horizontal max
            float max_val = _mm512_reduce_max_ps(max_vec);

            // Handle remainder
            for (; i < softmax_dim; ++i) {
                max_val = std::max(max_val, x[i]);
            }

            __m512 max_broadcast = _mm512_set1_ps(max_val);

            // Step 2: Compute exp(x - max) and sum
            __m512 sum_vec = _mm512_setzero_ps();
            i = 0;

            for (; i + 16 <= softmax_dim; i += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + i);
                __m512 shifted = _mm512_sub_ps(x_vec, max_broadcast);

                // Compute exp element-wise
                alignas(64) float shifted_arr[16];
                alignas(64) float exp_arr[16];
                _mm512_store_ps(shifted_arr, shifted);

                for (int j = 0; j < 16; ++j) {
                    exp_arr[j] = std::exp(shifted_arr[j]);
                }

                __m512 exp_vec = _mm512_load_ps(exp_arr);
                _mm512_storeu_ps(y + i, exp_vec);
                sum_vec = _mm512_add_ps(sum_vec, exp_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);

            // Handle remainder
            for (; i < softmax_dim; ++i) {
                y[i] = std::exp(x[i] - max_val);
                sum += y[i];
            }

            // Step 3: Normalize
            float inv_sum = 1.0f / sum;
            __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
            i = 0;

            for (; i + 16 <= softmax_dim; i += 16) {
                __m512 y_vec = _mm512_loadu_ps(y + i);
                y_vec = _mm512_mul_ps(y_vec, inv_sum_vec);
                _mm512_storeu_ps(y + i, y_vec);
            }

            // Handle remainder
            for (; i < softmax_dim; ++i) {
                y[i] *= inv_sum;
            }
        });
    }
#endif

    /// Dispatch to best available implementation
    static void run(
        const T* input,
        T* output,
        std::size_t num_rows,
        std::size_t softmax_dim
    ) {
#if defined(KERNELIX_AVX512)
        if constexpr (std::is_same_v<T, float>) {
            run_avx512(input, output, num_rows, softmax_dim);
            return;
        }
#elif defined(KERNELIX_AVX2)
        if constexpr (std::is_same_v<T, float>) {
            run_avx2(input, output, num_rows, softmax_dim);
            return;
        }
#endif
        run_scalar(input, output, num_rows, softmax_dim);
    }
};

/// Online softmax using a single pass (Milakov & Gimelshein algorithm)
/// More cache-friendly for large sequences
template<typename T>
struct OnlineSoftmaxKernel {

    static void run(
        const T* input,
        T* output,
        std::size_t num_rows,
        std::size_t softmax_dim
    ) {
        parallel::parallel_for(0, num_rows, [=](std::size_t row) {
            const T* x = input + row * softmax_dim;
            T* y = output + row * softmax_dim;

            // Online computation of max and sum in a single pass
            T max_val = std::numeric_limits<T>::lowest();
            T sum = T{0};

            for (std::size_t i = 0; i < softmax_dim; ++i) {
                T new_max = std::max(max_val, x[i]);
                // Adjust sum for new max: sum * exp(old_max - new_max)
                sum = sum * std::exp(max_val - new_max) + std::exp(x[i] - new_max);
                max_val = new_max;
            }

            // Second pass: compute normalized values
            T inv_sum = T{1} / sum;
            for (std::size_t i = 0; i < softmax_dim; ++i) {
                y[i] = std::exp(x[i] - max_val) * inv_sum;
            }
        });
    }
};

} // namespace kernelix::kernel
