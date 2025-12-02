#pragma once

#include <cstddef>
#include <type_traits>

// SIMD intrinsics headers
#if defined(KERNELIX_AVX512)
    #include <immintrin.h>
    #define KERNELIX_SIMD_WIDTH 16  // 16 floats
#elif defined(KERNELIX_AVX2)
    #include <immintrin.h>
    #define KERNELIX_SIMD_WIDTH 8   // 8 floats
#elif defined(KERNELIX_NEON)
    #include <arm_neon.h>
    #define KERNELIX_SIMD_WIDTH 4   // 4 floats
#else
    #define KERNELIX_SIMD_WIDTH 1   // Scalar fallback
#endif

namespace kernelix::simd {

/// SIMD vector width for given type
template<typename T>
constexpr std::size_t vector_width() {
    if constexpr (std::is_same_v<T, float>) {
        return KERNELIX_SIMD_WIDTH;
    } else if constexpr (std::is_same_v<T, double>) {
        return KERNELIX_SIMD_WIDTH / 2;
    } else {
        return 1;
    }
}

/// Portable SIMD vector type
template<typename T>
struct Vec;

#if defined(KERNELIX_AVX2) || defined(KERNELIX_AVX512)

template<>
struct Vec<float> {
    using type = __m256;
    static constexpr std::size_t width = 8;

    static type load(const float* ptr) { return _mm256_loadu_ps(ptr); }
    static type load_aligned(const float* ptr) { return _mm256_load_ps(ptr); }
    static void store(float* ptr, type v) { _mm256_storeu_ps(ptr, v); }
    static void store_aligned(float* ptr, type v) { _mm256_store_ps(ptr, v); }
    static type set1(float val) { return _mm256_set1_ps(val); }
    static type zero() { return _mm256_setzero_ps(); }
    static type add(type a, type b) { return _mm256_add_ps(a, b); }
    static type sub(type a, type b) { return _mm256_sub_ps(a, b); }
    static type mul(type a, type b) { return _mm256_mul_ps(a, b); }
    static type div(type a, type b) { return _mm256_div_ps(a, b); }
    static type fmadd(type a, type b, type c) { return _mm256_fmadd_ps(a, b, c); }
    static type max(type a, type b) { return _mm256_max_ps(a, b); }
    static type min(type a, type b) { return _mm256_min_ps(a, b); }

    // Horizontal sum
    static float hsum(type v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }
};

template<>
struct Vec<double> {
    using type = __m256d;
    static constexpr std::size_t width = 4;

    static type load(const double* ptr) { return _mm256_loadu_pd(ptr); }
    static type load_aligned(const double* ptr) { return _mm256_load_pd(ptr); }
    static void store(double* ptr, type v) { _mm256_storeu_pd(ptr, v); }
    static void store_aligned(double* ptr, type v) { _mm256_store_pd(ptr, v); }
    static type set1(double val) { return _mm256_set1_pd(val); }
    static type zero() { return _mm256_setzero_pd(); }
    static type add(type a, type b) { return _mm256_add_pd(a, b); }
    static type sub(type a, type b) { return _mm256_sub_pd(a, b); }
    static type mul(type a, type b) { return _mm256_mul_pd(a, b); }
    static type fmadd(type a, type b, type c) { return _mm256_fmadd_pd(a, b, c); }
};

#elif defined(KERNELIX_NEON)

template<>
struct Vec<float> {
    using type = float32x4_t;
    static constexpr std::size_t width = 4;

    static type load(const float* ptr) { return vld1q_f32(ptr); }
    static type load_aligned(const float* ptr) { return vld1q_f32(ptr); }
    static void store(float* ptr, type v) { vst1q_f32(ptr, v); }
    static void store_aligned(float* ptr, type v) { vst1q_f32(ptr, v); }
    static type set1(float val) { return vdupq_n_f32(val); }
    static type zero() { return vdupq_n_f32(0.0f); }
    static type add(type a, type b) { return vaddq_f32(a, b); }
    static type sub(type a, type b) { return vsubq_f32(a, b); }
    static type mul(type a, type b) { return vmulq_f32(a, b); }
    static type fmadd(type a, type b, type c) { return vfmaq_f32(c, a, b); }
    static type max(type a, type b) { return vmaxq_f32(a, b); }
    static type min(type a, type b) { return vminq_f32(a, b); }

    static float hsum(type v) {
        float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
        sum = vpadd_f32(sum, sum);
        return vget_lane_f32(sum, 0);
    }
};

#else

// Scalar fallback
template<>
struct Vec<float> {
    using type = float;
    static constexpr std::size_t width = 1;

    static type load(const float* ptr) { return *ptr; }
    static type load_aligned(const float* ptr) { return *ptr; }
    static void store(float* ptr, type v) { *ptr = v; }
    static void store_aligned(float* ptr, type v) { *ptr = v; }
    static type set1(float val) { return val; }
    static type zero() { return 0.0f; }
    static type add(type a, type b) { return a + b; }
    static type sub(type a, type b) { return a - b; }
    static type mul(type a, type b) { return a * b; }
    static type fmadd(type a, type b, type c) { return a * b + c; }
    static type max(type a, type b) { return a > b ? a : b; }
    static type min(type a, type b) { return a < b ? a : b; }
    static float hsum(type v) { return v; }
};

template<>
struct Vec<double> {
    using type = double;
    static constexpr std::size_t width = 1;

    static type load(const double* ptr) { return *ptr; }
    static type load_aligned(const double* ptr) { return *ptr; }
    static void store(double* ptr, type v) { *ptr = v; }
    static void store_aligned(double* ptr, type v) { *ptr = v; }
    static type set1(double val) { return val; }
    static type zero() { return 0.0; }
    static type add(type a, type b) { return a + b; }
    static type sub(type a, type b) { return a - b; }
    static type mul(type a, type b) { return a * b; }
    static type fmadd(type a, type b, type c) { return a * b + c; }
    static double hsum(type v) { return v; }
};

#endif

} // namespace kernelix::simd
