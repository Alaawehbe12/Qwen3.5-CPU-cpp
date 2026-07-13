#include "memory_budget.h"
#include <algorithm>

size_t estimate_activation_scratch(const ModelConfig& cfg) {
    // Largest per-layer tensor is generally the FFN intermediate activation:
    // [hidden_size -> ffn_intermediate] projection output, single token.
    // Sized generously (x2) to cover temporary buffers used mid-computation
    // (e.g. gate + up projections live simultaneously in SwiGLU-style FFNs).
    size_t ffn_activation = (size_t)cfg.ffn_intermediate * sizeof(float) * 2;

    // Attention layer needs Q/K/V projection scratch for one token.
    size_t attn_activation = (size_t)(cfg.attn_q_heads * cfg.attn_head_dim
                            + 2 * cfg.attn_kv_heads * cfg.attn_head_dim) * sizeof(float);

    // GDN layer needs Q/K/V projection scratch at its own head dimensions.
    size_t gdn_activation = (size_t)(3 * cfg.gdn_n_heads * cfg.gdn_head_dim) * sizeof(float);

    return std::max({ffn_activation, attn_activation, gdn_activation});
}

size_t estimate_vision_scratch(int max_tiles, int vit_dim, int patches_per_tile) {
    // Patch embeddings for the worst-case number of tiles, held simultaneously
    // until the projection step folds them into the token stream.
    size_t patch_embeddings = (size_t)max_tiles * patches_per_tile * vit_dim * sizeof(float);
    // Small constant overhead for intermediate ViT self-attention buffers.
    size_t vit_working_set = (size_t)patches_per_tile * vit_dim * sizeof(float) * 2;
    return patch_embeddings + vit_working_set;
}

MemoryBudget compute_budget(const ModelConfig& cfg,
                             int bytes_per_weight,
                             int max_context,
                             int kv_bytes_per_elem,
                             int max_vision_tiles,
                             int vit_dim) {
    MemoryBudget b;

    b.weight_bytes = (size_t)cfg.n_params * (size_t)bytes_per_weight;

    b.gdn_state_bytes = (size_t)cfg.n_layers_gdn
        * cfg.gdn_v_heads * cfg.gdn_k_head_dim * cfg.gdn_v_head_dim
        * sizeof(float); // one [k_head_dim, v_head_dim] state per value head

    b.gdn_conv_bytes = gdn_conv_state_bytes(cfg);

    b.kv_cache_bytes = (size_t)cfg.n_layers_attn * (size_t)max_context
        * 2 /* K and V */ * cfg.attn_kv_heads * cfg.attn_head_dim
        * kv_bytes_per_elem;

    b.activation_bytes = estimate_activation_scratch(cfg);
    b.vision_scratch_bytes = estimate_vision_scratch(max_vision_tiles, vit_dim);

    return b;
}

int64_t solve_max_context(const ModelConfig& cfg,
                           size_t ram_budget_bytes,
                           size_t weight_bytes,
                           size_t gdn_state_bytes,
                           size_t activation_bytes,
                           size_t vision_scratch_bytes,
                           size_t safety_margin_bytes,
                           int kv_bytes_per_elem) {
    size_t fixed_costs = weight_bytes + gdn_state_bytes
                        + activation_bytes + vision_scratch_bytes
                        + safety_margin_bytes;

    if (fixed_costs >= ram_budget_bytes) {
        return -1; // doesn't even fit the fixed costs
    }

    size_t remaining = ram_budget_bytes - fixed_costs;

    size_t per_token_bytes = (size_t)cfg.n_layers_attn
        * 2 * cfg.attn_kv_heads * cfg.attn_head_dim * kv_bytes_per_elem;

    int64_t context = (int64_t)(remaining / per_token_bytes);

    // Never propose more than the model natively supports.
    return std::min<int64_t>(context, cfg.native_max_context);
}
