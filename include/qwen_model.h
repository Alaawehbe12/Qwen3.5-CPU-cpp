#pragma once
#include <string>
#include <vector>
#include "model_config.h"
#include "memory_budget.h"
#include "safetensors_loader.h"

// ---------------------------------------------------------------------------
// Stateful, single-sequence incremental decoder — arena/memory-first edition.
//
// All memory is planned once (by the memory_budget equation) and allocated in
// the constructor:
//   - a KV cache per attention layer (fp32)
//   - a recurrent state matrix + a causal-conv rolling state per GDN layer
//   - ONE shared scratch arena that every per-step temporary is carved from
//     via a bump pointer that resets each step
// Every layer's weight pointers are resolved once up front. As a result there
// is ZERO heap allocation during the token loop — reset() rewinds state rather
// than freeing, exactly as the four-arena design intended.
// ---------------------------------------------------------------------------
class QwenModel {
public:
    QwenModel(SafeTensors& weights, const ModelConfig& cfg, int max_context);

    void reset();                          // clear all state, position -> 0
    const std::vector<float>& step(int token_id); // returns logits [vocab]
    int position() const { return pos_; }

    std::vector<int> generate_greedy(const std::vector<int>& prompt, int n_new);

    const MemoryBudget& budget() const { return budget_; }
    bool ok() const { return ok_; }

private:
    // All weight pointers for one decoder layer, resolved once at construction.
    // Big projection weights are WeightViews (read in place, bf16→fp32 in the
    // kernel); small elementwise weights stay fp32.
    struct LayerWeights {
        bool is_attn = false;
        int  mixer_idx = 0;            // index into kv caches / gdn state
        const float* input_ln = nullptr;
        const float* post_ln = nullptr;
        WeightView mlp_gate, mlp_up, mlp_down;
        // attention
        WeightView q_proj, k_proj, v_proj, o_proj;
        const float* q_norm = nullptr; const float* k_norm = nullptr;
        // gated deltanet
        WeightView in_qkv, in_z, in_b, in_a, out_proj;
        const float* conv1d = nullptr; const float* dt_bias = nullptr;
        const float* A_log = nullptr;  const float* gdn_norm = nullptr;
    };

    void attn_step(const LayerWeights& lw, float* out);
    void gdn_step(const LayerWeights& lw, float* out);

    // Scratch bump allocator (no heap traffic; scratch_ is pre-sized).
    float* sb(int n) { float* p = scratch_.data() + sbpos_; sbpos_ += (size_t)n; return p; }
    void sbreset() { sbpos_ = 0; }

    SafeTensors& w_;
    ModelConfig cfg_;
    int max_context_;
    int pos_ = 0;
    bool ok_ = true;

    std::vector<LayerWeights> lw_;
    std::string emb_name_;
    WeightView head_;                    // output projection (lm_head, or tied embed)
    const float* final_norm_ = nullptr;

    // Persistent arenas (allocated once, rewound by reset()).
    std::vector<std::vector<float>> kcache_, vcache_;      // per attention layer
    std::vector<std::vector<float>> gdn_state_, conv_state_; // per GDN layer

    // Per-step working buffers (fixed) + the shared scratch arena.
    std::vector<float> hidden_, normed_, mix_, ff_, logits_;
    std::vector<float> scratch_;
    size_t sbpos_ = 0;

    MemoryBudget budget_;
};
