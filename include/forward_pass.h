#pragma once
#include <cstddef>
#include <cstdint>
#include <cmath>
#include "weight_view.h"

// ---------------------------------------------------------------------------
// Core fp32 kernels for the Qwen3.5 text forward pass.
//
// Weight-matrix convention (matches PyTorch nn.Linear.weight): a linear layer
// with `in_dim` inputs and `out_dim` outputs stores W as row-major
// [out_dim, in_dim], and computes  y[o] = sum_i W[o*in_dim + i] * x[i].
// This is exactly how the safetensors tensors are laid out, so weight pointers
// from SafeTensors::get_f32() feed straight in.
//
// Each kernel is validated against transformers-dumped reference activations
// (see scratchpad/oracle). No allocation happens inside these; callers pass
// output buffers.
// ---------------------------------------------------------------------------

inline float silu(float x) { return x / (1.0f + expf(-x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
// Numerically stable softplus: log(1+exp(x)).
inline float softplusf(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// L2 normalize x (length n) in place: x /= sqrt(sum(x^2) + eps). Used on GDN
// q/k (NOTE: sum, not mean — this is not RMSNorm).
void l2norm(float* x, int n, float eps);

// y = W @ x, with W row-major [out_dim, in_dim]. y has out_dim elements.
void matmul(const float* W, const float* x, float* y, int out_dim, int in_dim);

// Same, but W is a WeightView read in place: bf16 weights are widened to fp32
// inside the AVX2 kernel (no fp32 weight cache). fp32 weights take the plain
// path. Numerically identical to converting up front.
void matmul(const WeightView& W, const float* x, float* y, int out_dim, int in_dim);

// Qwen3_5RMSNorm: y = x * rsqrt(mean(x^2)+eps) * (1 + weight).
// NOTE the (1 + weight) form — the stored weight is a delta from 1.0, init 0.
// Used by input_layernorm, post_attention_layernorm, q_norm, k_norm, and the
// final model norm.
void rmsnorm(const float* x, const float* weight, float* y, int n, float eps);

// Qwen3_5RMSNormGated (inside the GDN block): y = weight * (x*rsqrt(...)) * silu(gate).
// Plain `weight` (init 1), gate applied via SiLU after normalization.
void rmsnorm_gated(const float* x, const float* weight, const float* gate,
                   float* y, int n, float eps);

// Partial rotary position embedding, applied in place to one head's vector of
// length head_dim. Only the first `rotary_dim` dims are rotated (the rest pass
// through unchanged); pairing is (i, i + rotary_dim/2) per rotate_half.
//   inv_freq_i = theta^(-2i/rotary_dim),  angle = pos * inv_freq_i
// Matches transformers apply_rotary_pos_emb with mRoPE collapsed to text.
void apply_rope(float* vec, int rotary_dim, int pos, double theta);
