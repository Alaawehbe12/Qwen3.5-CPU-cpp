#include "forward_pass.h"
#include <cstring>
#include <immintrin.h>

// AVX2 dot product with 4 independent accumulators to hide FMA latency.
static inline float dot_avx2(const float* a, const float* b, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float r = _mm_cvtss_f32(s);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
}

// y = W @ x, W row-major [out_dim, in_dim]. Parallelized over output rows
// (each row is an independent dot product) and vectorized with AVX2.
void matmul(const float* W, const float* x, float* y, int out_dim, int in_dim) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_dim; ++o)
        y[o] = dot_avx2(W + (size_t)o * in_dim, x, in_dim);
}

// Widen 8 bf16 values (as uint16) to an __m256 of fp32: bf16 is the top 16
// bits of the fp32, so zero-extend to 32 bits and shift left 16.
static inline __m256 load8_bf16(const uint16_t* p) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
    return _mm256_castsi256_ps(w);
}

// Dot product of a bf16 weight row with an fp32 activation, widening in-register.
static inline float dot_bf16(const uint16_t* w, const float* b, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(load8_bf16(w + i),      _mm256_loadu_ps(b + i),      a0);
        a1 = _mm256_fmadd_ps(load8_bf16(w + i + 8),  _mm256_loadu_ps(b + i + 8),  a1);
        a2 = _mm256_fmadd_ps(load8_bf16(w + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(load8_bf16(w + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(load8_bf16(w + i), _mm256_loadu_ps(b + i), acc);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float r = _mm_cvtss_f32(s);
    for (; i < n; ++i) { uint32_t bits = (uint32_t)w[i] << 16; float wf; std::memcpy(&wf, &bits, 4); r += wf * b[i]; }
    return r;
}

void matmul(const WeightView& W, const float* x, float* y, int out_dim, int in_dim) {
    if (W.bf16) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < out_dim; ++o)
            y[o] = dot_bf16(W.bf16 + (size_t)o * in_dim, x, in_dim);
    } else {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < out_dim; ++o)
            y[o] = dot_avx2(W.f32 + (size_t)o * in_dim, x, in_dim);
    }
}

void rmsnorm(const float* x, const float* weight, float* y, int n, float eps) {
    // Compute in double for the reduction to match torch's float32 accumulation
    // closely; the model normalizes in fp32 regardless of weight dtype.
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) sumsq += (double)x[i] * x[i];
    float inv = 1.0f / std::sqrt((float)(sumsq / n) + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (1.0f + weight[i]);
}

void rmsnorm_gated(const float* x, const float* weight, const float* gate,
                   float* y, int n, float eps) {
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) sumsq += (double)x[i] * x[i];
    float inv = 1.0f / std::sqrt((float)(sumsq / n) + eps);
    for (int i = 0; i < n; ++i)
        y[i] = weight[i] * (x[i] * inv) * silu(gate[i]);
}

void l2norm(float* x, int n, float eps) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)x[i] * x[i];
    float inv = 1.0f / std::sqrt((float)s + eps);
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

void apply_rope(float* vec, int rotary_dim, int pos, double theta) {
    int half = rotary_dim / 2;
    for (int i = 0; i < half; ++i) {
        double inv_freq = std::pow(theta, -(double)(2 * i) / rotary_dim);
        double angle = (double)pos * inv_freq;
        float c = (float)std::cos(angle), s = (float)std::sin(angle);
        float x1 = vec[i], x2 = vec[i + half];
        vec[i]        = x1 * c - x2 * s;
        vec[i + half] = x2 * c + x1 * s;
    }
}
