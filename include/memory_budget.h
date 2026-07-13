#pragma once
#include <cstddef>
#include "model_config.h"

// Result of applying the memory equation for a given model + context choice.
struct MemoryBudget {
    size_t weight_bytes = 0;
    size_t gdn_state_bytes = 0;
    size_t gdn_conv_bytes = 0;   // GDN causal-conv1d rolling state
    size_t kv_cache_bytes = 0;
    size_t activation_bytes = 0;
    size_t vision_scratch_bytes = 0;

    size_t total() const {
        return weight_bytes + gdn_state_bytes + gdn_conv_bytes + kv_cache_bytes
             + activation_bytes + vision_scratch_bytes;
    }
};

// GDN conv-state bytes for the whole model: one rolling window of (kernel-1)
// past inputs per conv channel, per linear-attention layer, kept fp32.
//   conv_dim = 2*key_dim + value_dim  (q,k,v channels fed through the conv)
inline size_t gdn_conv_state_bytes(const ModelConfig& cfg) {
    size_t key_dim = (size_t)cfg.gdn_k_heads * cfg.gdn_k_head_dim;
    size_t value_dim = (size_t)cfg.gdn_v_heads * cfg.gdn_v_head_dim;
    size_t conv_dim = key_dim * 2 + value_dim;
    return (size_t)cfg.n_layers_gdn * conv_dim * (cfg.gdn_conv_kernel - 1) * sizeof(float);
}

// Forward direction: given a model, quantization, and chosen max context,
// compute how much memory each arena needs.
//
//   weight_bytes    = n_params * bytes_per_weight
//   gdn_state_bytes = n_layers_gdn * heads * head_dim^2 * sizeof(float)
//   kv_cache_bytes  = n_layers_attn * max_context * 2 * kv_heads * head_dim * kv_elem_bytes
//
MemoryBudget compute_budget(const ModelConfig& cfg,
                             int bytes_per_weight,
                             int max_context,
                             int kv_bytes_per_elem,
                             int max_vision_tiles = 4,
                             int vit_dim = 1024);

// Reverse direction: given a total RAM ceiling and everything else fixed,
// solve for the largest max_context that still fits.
// Returns -1 if the budget can't even cover weights + GDN state + margin.
int64_t solve_max_context(const ModelConfig& cfg,
                           size_t ram_budget_bytes,
                           size_t weight_bytes,
                           size_t gdn_state_bytes,
                           size_t activation_bytes,
                           size_t vision_scratch_bytes,
                           size_t safety_margin_bytes,
                           int kv_bytes_per_elem);

// Estimate the scratch buffer big enough to hold the largest single-layer
// activation (FFN intermediate is normally the largest tensor per layer).
size_t estimate_activation_scratch(const ModelConfig& cfg);

// Estimate scratch space for the ViT patch encoder, sized for a worst-case
// number of image tiles (e.g. high-res tiling schemes emit several tiles).
size_t estimate_vision_scratch(int max_tiles, int vit_dim, int patches_per_tile = 256);
