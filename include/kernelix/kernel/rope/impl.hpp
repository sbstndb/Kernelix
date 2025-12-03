#pragma once

#include "../../util/simd.hpp"
#include "../../util/parallel.hpp"
#include <cmath>
#include <cstddef>

namespace kernelix::kernel {

/// RoPE (Rotary Position Embedding) kernel
/// Applies rotational transformation based on position
template<typename T>
struct RoPEKernel {
    /// Apply RoPE to input tensor
    /// Input: [seq_len, head_dim]
    /// cos_cache, sin_cache: [seq_len, head_dim/2] (precomputed for positions)
    static void run(
        const T* input,
        const T* cos_cache,
        const T* sin_cache,
        T* output,
        std::size_t seq_len,
        std::size_t head_dim,
        std::size_t position_offset = 0
    ) {
        const std::size_t half_dim = head_dim / 2;

        parallel::parallel_for(0, seq_len, [&](std::size_t pos) {
            const std::size_t cache_pos = pos + position_offset;
            const T* cos_row = cos_cache + cache_pos * half_dim;
            const T* sin_row = sin_cache + cache_pos * half_dim;
            const T* in_row = input + pos * head_dim;
            T* out_row = output + pos * head_dim;

            for (std::size_t i = 0; i < half_dim; ++i) {
                T x0 = in_row[2 * i];
                T x1 = in_row[2 * i + 1];
                T cos_val = cos_row[i];
                T sin_val = sin_row[i];

                // Rotation:
                // x'[2i]   = x[2i] * cos - x[2i+1] * sin
                // x'[2i+1] = x[2i] * sin + x[2i+1] * cos
                out_row[2 * i] = x0 * cos_val - x1 * sin_val;
                out_row[2 * i + 1] = x0 * sin_val + x1 * cos_val;
            }
        });
    }

    /// Optimized RoPE with SIMD (when available)
    static void run_optimized(
        const T* input,
        const T* cos_cache,
        const T* sin_cache,
        T* output,
        std::size_t seq_len,
        std::size_t head_dim,
        std::size_t position_offset = 0
    ) {
#if defined(__AVX2__) || defined(__AVX__)
        if constexpr (std::is_same_v<T, float>) {
            run_avx2(input, cos_cache, sin_cache, output, seq_len, head_dim, position_offset);
            return;
        }
#endif
        run(input, cos_cache, sin_cache, output, seq_len, head_dim, position_offset);
    }

#if defined(__AVX2__) || defined(__AVX__)
    static void run_avx2(
        const float* input,
        const float* cos_cache,
        const float* sin_cache,
        float* output,
        std::size_t seq_len,
        std::size_t head_dim,
        std::size_t position_offset
    ) {
        const std::size_t half_dim = head_dim / 2;

        parallel::parallel_for(0, seq_len, [&](std::size_t pos) {
            const std::size_t cache_pos = pos + position_offset;
            const float* cos_row = cos_cache + cache_pos * half_dim;
            const float* sin_row = sin_cache + cache_pos * half_dim;
            const float* in_row = input + pos * head_dim;
            float* out_row = output + pos * head_dim;

            std::size_t i = 0;
            // Process 4 pairs at a time (8 floats)
            for (; i + 4 <= half_dim; i += 4) {
                // Load 8 input values as 4 pairs
                __m256 x = _mm256_loadu_ps(in_row + 2 * i);

                // Load cos/sin values
                __m128 cos4 = _mm_loadu_ps(cos_row + i);
                __m128 sin4 = _mm_loadu_ps(sin_row + i);

                // Interleave cos and sin for paired multiplication
                // cos_interleaved = [c0, c0, c1, c1, c2, c2, c3, c3]
                __m256 cos_lo = _mm256_castps128_ps256(_mm_unpacklo_ps(cos4, cos4));
                __m256 cos_hi = _mm256_castps128_ps256(_mm_unpackhi_ps(cos4, cos4));
                __m256 cos_interleaved = _mm256_insertf128_ps(cos_lo, _mm_unpackhi_ps(cos4, cos4), 1);
                cos_interleaved = _mm256_insertf128_ps(
                    _mm256_castps128_ps256(_mm_unpacklo_ps(cos4, cos4)),
                    _mm_unpackhi_ps(cos4, cos4), 1);

                // Actually easier approach: duplicate each element
                __m256 cos_vec = _mm256_set_ps(
                    cos_row[i+3], cos_row[i+3], cos_row[i+2], cos_row[i+2],
                    cos_row[i+1], cos_row[i+1], cos_row[i+0], cos_row[i+0]
                );
                __m256 sin_vec = _mm256_set_ps(
                    sin_row[i+3], sin_row[i+3], sin_row[i+2], sin_row[i+2],
                    sin_row[i+1], sin_row[i+1], sin_row[i+0], sin_row[i+0]
                );

                // x = [x0, x1, x2, x3, x4, x5, x6, x7]
                // We need: [x0*c - x1*s, x0*s + x1*c, x2*c - x3*s, x2*s + x3*c, ...]

                // Shuffle to get: x_even = [x0, x0, x2, x2, x4, x4, x6, x6]
                //                 x_odd  = [x1, x1, x3, x3, x5, x5, x7, x7]
                __m256 x_even = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(2, 2, 0, 0));
                __m256 x_odd = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 1, 1));

                // result_even = x_even * cos - x_odd * sin (for positions 0, 2, 4, 6)
                // result_odd  = x_even * sin + x_odd * cos (for positions 1, 3, 5, 7)
                __m256 term1 = _mm256_mul_ps(x_even, cos_vec);
                __m256 term2 = _mm256_mul_ps(x_odd, sin_vec);
                __m256 term3 = _mm256_mul_ps(x_even, sin_vec);
                __m256 term4 = _mm256_mul_ps(x_odd, cos_vec);

                __m256 res_even = _mm256_sub_ps(term1, term2);  // x*cos - y*sin
                __m256 res_odd = _mm256_add_ps(term3, term4);   // x*sin + y*cos

                // Blend: take even-indexed from res_even, odd-indexed from res_odd
                // blend mask: 0b10101010 = 0xAA
                __m256 result = _mm256_blend_ps(res_even, res_odd, 0xAA);

                _mm256_storeu_ps(out_row + 2 * i, result);
            }

            // Handle remaining pairs
            for (; i < half_dim; ++i) {
                float x0 = in_row[2 * i];
                float x1 = in_row[2 * i + 1];
                float cos_val = cos_row[i];
                float sin_val = sin_row[i];

                out_row[2 * i] = x0 * cos_val - x1 * sin_val;
                out_row[2 * i + 1] = x0 * sin_val + x1 * cos_val;
            }
        });
    }
#endif

    /// Precompute cos/sin caches for RoPE
    /// Output: cos_cache[max_seq_len, head_dim/2], sin_cache[max_seq_len, head_dim/2]
    static void precompute_freqs(
        T* cos_cache,
        T* sin_cache,
        std::size_t max_seq_len,
        std::size_t head_dim,
        T base = T{10000}
    ) {
        const std::size_t half_dim = head_dim / 2;

        parallel::parallel_for(0, max_seq_len, [&](std::size_t pos) {
            T* cos_row = cos_cache + pos * half_dim;
            T* sin_row = sin_cache + pos * half_dim;

            for (std::size_t i = 0; i < half_dim; ++i) {
                // freq = 1.0 / (base^(2i/d))
                T freq = T{1} / std::pow(base, static_cast<T>(2 * i) / static_cast<T>(head_dim));
                T theta = static_cast<T>(pos) * freq;
                cos_row[i] = std::cos(theta);
                sin_row[i] = std::sin(theta);
            }
        });
    }
};

} // namespace kernelix::kernel
