#pragma once
#include <string>
#include <vector>
#include "model_config.h"
#include "safetensors_loader.h"

// ---------------------------------------------------------------------------
// Qwen3.5 text forward pass (fp32, CPU, single sequence).
//
// This is the reference-validated implementation: every layer and the final
// logits match a transformers fp32 run to rel_l2 ~ 1e-6 (see scratchpad tests
// / oracle). It runs directly against safetensors weights.
//
// `prefill` runs the whole prompt at once (the path validated against the
// oracle). Weights are addressed by HuggingFace tensor names, e.g.
// "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".
// ---------------------------------------------------------------------------
class QwenForward {
public:
    QwenForward(SafeTensors& weights, const ModelConfig& cfg)
        : w_(weights), cfg_(cfg) {}

    // Runs the full stack over `ids` and writes [seq * vocab_size] logits into
    // `logits_out`. Returns false if a required weight is missing.
    bool prefill(const std::vector<int>& ids, std::vector<float>& logits_out);

private:
    void mix_attention(int layer, const std::vector<float>& in,
                       std::vector<float>& out, int seq);
    void mix_gdn(int layer, const std::vector<float>& in,
                 std::vector<float>& out, int seq);
    const float* W(const std::string& name, bool* ok);

    SafeTensors& w_;
    ModelConfig cfg_;
    bool missing_ = false;
};
