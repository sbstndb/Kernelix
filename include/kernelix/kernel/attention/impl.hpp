#pragma once

#include "../../util/simd.hpp"
#include "../../util/parallel.hpp"
#include "../../util/memory.hpp"
#include "../norm/softmax.hpp"
#include <cmath>
#include <cstddef>
#include <limits>
#include <cstring>

namespace kernelix::kernel {

/// Scaled Dot-Product Attention implementation
/// Output = softmax(Q @ K^T / scale) @ V
///
/// Q: [seq_len_q, head_dim]
/// K: [seq_len_k, head_dim]
/// V: [seq_len_k, head_dim]
/// Output: [seq_len_q, head_dim]
template<typename T>
struct AttentionKernel {

    /// Standard attention (non-causal)
    static void run(
        const T* Q, const T* K, const T* V, T* output,
        std::size_t seq_len_q, std::size_t seq_len_k, std::size_t head_dim,
        T scale  // 1/sqrt(head_dim)
    ) {
        // Allocate attention scores buffer [seq_len_q, seq_len_k]
        auto attn_scores = memory::make_aligned<T>(seq_len_q * seq_len_k);

        // Step 1: Compute Q @ K^T with scaling
        // attn_scores[i, j] = sum_d(Q[i, d] * K[j, d]) * scale
        compute_qk_scaled(Q, K, attn_scores.get(), seq_len_q, seq_len_k, head_dim, scale);

        // Step 2: Apply softmax row-wise
        SoftmaxKernel<T>::run(attn_scores.get(), attn_scores.get(), seq_len_q, seq_len_k);

        // Step 3: Compute attn_scores @ V
        // output[i, d] = sum_j(attn_scores[i, j] * V[j, d])
        compute_attn_v(attn_scores.get(), V, output, seq_len_q, seq_len_k, head_dim);
    }

    /// Causal (masked) attention
    /// Positions can only attend to earlier positions (i >= j)
    static void run_causal(
        const T* Q, const T* K, const T* V, T* output,
        std::size_t seq_len_q, std::size_t seq_len_k, std::size_t head_dim,
        T scale
    ) {
        auto attn_scores = memory::make_aligned<T>(seq_len_q * seq_len_k);

        // Step 1: Compute Q @ K^T with scaling and causal mask
        compute_qk_scaled_causal(Q, K, attn_scores.get(), seq_len_q, seq_len_k, head_dim, scale);

        // Step 2: Apply softmax row-wise (masked positions are -inf, become 0 after softmax)
        SoftmaxKernel<T>::run(attn_scores.get(), attn_scores.get(), seq_len_q, seq_len_k);

        // Step 3: Compute attn_scores @ V
        compute_attn_v(attn_scores.get(), V, output, seq_len_q, seq_len_k, head_dim);
    }

private:
    /// Compute Q @ K^T with scaling
    static void compute_qk_scaled(
        const T* Q, const T* K, T* scores,
        std::size_t seq_len_q, std::size_t seq_len_k, std::size_t head_dim,
        T scale
    ) {
        parallel::parallel_for(0, seq_len_q, [=](std::size_t i) {
            const T* q_row = Q + i * head_dim;

            for (std::size_t j = 0; j < seq_len_k; ++j) {
                const T* k_row = K + j * head_dim;

                // Dot product Q[i] · K[j]
                T dot = T{0};
                for (std::size_t d = 0; d < head_dim; ++d) {
                    dot += q_row[d] * k_row[d];
                }
                scores[i * seq_len_k + j] = dot * scale;
            }
        });
    }

    /// Compute Q @ K^T with scaling and causal mask
    static void compute_qk_scaled_causal(
        const T* Q, const T* K, T* scores,
        std::size_t seq_len_q, std::size_t seq_len_k, std::size_t head_dim,
        T scale
    ) {
        const T neg_inf = -std::numeric_limits<T>::infinity();

        parallel::parallel_for(0, seq_len_q, [=](std::size_t i) {
            const T* q_row = Q + i * head_dim;

            for (std::size_t j = 0; j < seq_len_k; ++j) {
                if (j > i) {
                    // Future position - mask with -inf
                    scores[i * seq_len_k + j] = neg_inf;
                } else {
                    const T* k_row = K + j * head_dim;

                    T dot = T{0};
                    for (std::size_t d = 0; d < head_dim; ++d) {
                        dot += q_row[d] * k_row[d];
                    }
                    scores[i * seq_len_k + j] = dot * scale;
                }
            }
        });
    }

    /// Compute attention_scores @ V
    static void compute_attn_v(
        const T* attn_scores, const T* V, T* output,
        std::size_t seq_len_q, std::size_t seq_len_k, std::size_t head_dim
    ) {
        parallel::parallel_for(0, seq_len_q, [=](std::size_t i) {
            const T* attn_row = attn_scores + i * seq_len_k;
            T* out_row = output + i * head_dim;

            // Initialize output row to zero
            std::memset(out_row, 0, head_dim * sizeof(T));

            // output[i, d] = sum_j(attn[i, j] * V[j, d])
            for (std::size_t j = 0; j < seq_len_k; ++j) {
                const T attn_ij = attn_row[j];
                const T* v_row = V + j * head_dim;

                for (std::size_t d = 0; d < head_dim; ++d) {
                    out_row[d] += attn_ij * v_row[d];
                }
            }
        });
    }

public:
#if defined(KERNELIX_AVX2)
    /// AVX2 optimized Q @ K^T
    static void compute_qk_scaled_avx2(
        const float* Q, const float* K, float* scores,
        std::size_t seq_len_q, std::size_t seq_len_k, std::size_t head_dim,
        float scale
    ) {
        __m256 scale_vec = _mm256_set1_ps(scale);

        parallel::parallel_for(0, seq_len_q, [=](std::size_t i) {
            const float* q_row = Q + i * head_dim;

            for (std::size_t j = 0; j < seq_len_k; ++j) {
                const float* k_row = K + j * head_dim;

                __m256 sum_vec = _mm256_setzero_ps();
                std::size_t d = 0;

                // Vectorized dot product
                for (; d + 8 <= head_dim; d += 8) {
                    __m256 q_vec = _mm256_loadu_ps(q_row + d);
                    __m256 k_vec = _mm256_loadu_ps(k_row + d);
                    sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                }

                float dot = simd::Vec<float>::hsum(sum_vec);

                // Handle remainder
                for (; d < head_dim; ++d) {
                    dot += q_row[d] * k_row[d];
                }

                scores[i * seq_len_k + j] = dot * scale;
            }
        });
    }

    /// AVX2 optimized attention @ V
    static void compute_attn_v_avx2(
        const float* attn_scores, const float* V, float* output,
        std::size_t seq_len_q, std::size_t seq_len_k, std::size_t head_dim
    ) {
        parallel::parallel_for(0, seq_len_q, [=](std::size_t i) {
            const float* attn_row = attn_scores + i * seq_len_k;
            float* out_row = output + i * head_dim;

            // Initialize output row to zero
            for (std::size_t d = 0; d < head_dim; d += 8) {
                if (d + 8 <= head_dim) {
                    _mm256_storeu_ps(out_row + d, _mm256_setzero_ps());
                } else {
                    for (std::size_t dd = d; dd < head_dim; ++dd) {
                        out_row[dd] = 0.0f;
                    }
                }
            }

            // Accumulate: output[d] += attn[j] * V[j, d]
            for (std::size_t j = 0; j < seq_len_k; ++j) {
                __m256 attn_broadcast = _mm256_set1_ps(attn_row[j]);
                const float* v_row = V + j * head_dim;

                std::size_t d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m256 out_vec = _mm256_loadu_ps(out_row + d);
                    __m256 v_vec = _mm256_loadu_ps(v_row + d);
                    out_vec = _mm256_fmadd_ps(attn_broadcast, v_vec, out_vec);
                    _mm256_storeu_ps(out_row + d, out_vec);
                }

                // Handle remainder
                for (; d < head_dim; ++d) {
                    out_row[d] += attn_row[j] * v_row[d];
                }
            }
        });
    }
#endif

    /// Optimized run with SIMD dispatch
    static void run_optimized(
        const T* Q, const T* K, const T* V, T* output,
        std::size_t seq_len_q, std::size_t seq_len_k, std::size_t head_dim,
        T scale
    ) {
#if defined(KERNELIX_AVX2)
        if constexpr (std::is_same_v<T, float>) {
            auto attn_scores = memory::make_aligned<float>(seq_len_q * seq_len_k);

            compute_qk_scaled_avx2(Q, K, attn_scores.get(), seq_len_q, seq_len_k, head_dim, scale);
            SoftmaxKernel<float>::run(attn_scores.get(), attn_scores.get(), seq_len_q, seq_len_k);
            compute_attn_v_avx2(attn_scores.get(), V, output, seq_len_q, seq_len_k, head_dim);
            return;
        }
#endif
        run(Q, K, V, output, seq_len_q, seq_len_k, head_dim, scale);
    }

    /// Optimized causal run with SIMD dispatch
    static void run_causal_optimized(
        const T* Q, const T* K, const T* V, T* output,
        std::size_t seq_len_q, std::size_t seq_len_k, std::size_t head_dim,
        T scale
    ) {
        // For now, use standard implementation
        // TODO: Add AVX2 optimized causal version
        run_causal(Q, K, V, output, seq_len_q, seq_len_k, head_dim, scale);
    }
};

} // namespace kernelix::kernel
