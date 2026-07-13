#pragma once
#include <cstdint>
#include <string>

// Architecture constants for a hybrid GDN + Gated-Attention model.
//
// Values below are the REAL Qwen3.5-0.8B config, taken from the shipped
// models/Qwen3.5-0.8B/config.json (text_config). Earlier revisions of this
// file guessed at a "2B" variant; these numbers are now ground-truth for the
// model actually present in this repo.
struct ModelConfig {
    std::string name = "qwen3.5-0.8b";

    int64_t n_params = 800'000'000; // approx; derived exactly at load time

    int hidden_size = 1024;         // config: hidden_size

    int n_layers_total = 24;        // config: num_hidden_layers
    int n_layers_gdn = 18;          // 18 "linear_attention" layers
    int n_layers_attn = 6;          // 6 "full_attention" layers (every 4th)

    // Gated DeltaNet (linear attention) dimensions.
    // NOTE: the real block is a Mamba2/Gated-DeltaNet design (conv1d + dt +
    // A_log gating, fused in_proj_qkv/a/b/z), not a plain delta rule.
    // Key and value heads/dims can differ in principle (num_v_heads is a
    // multiple of num_k_heads; q/k are repeated); for the 0.8B they're equal.
    int gdn_k_heads = 16;           // config: linear_num_key_heads
    int gdn_v_heads = 16;           // config: linear_num_value_heads
    int gdn_k_head_dim = 128;       // config: linear_key_head_dim
    int gdn_v_head_dim = 128;       // config: linear_value_head_dim
    int gdn_conv_kernel = 4;        // config: linear_conv_kernel_dim
    // Back-compat aliases (== value-side) used by the older GGUF scaffold.
    int gdn_n_heads = 16;
    int gdn_head_dim = 128;

    // Gated Attention (standard softmax attention) dimensions.
    int attn_q_heads = 8;           // config: num_attention_heads
    int attn_kv_heads = 2;          // config: num_key_value_heads (GQA)
    int attn_head_dim = 256;        // config: head_dim
    bool attn_output_gate = true;   // config: attn_output_gate
    bool attn_qk_norm = true;       // self_attn.{q,k}_norm present per layer

    int ffn_intermediate = 3584;    // config: intermediate_size
    int vocab_size = 248320;        // config: vocab_size

    int native_max_context = 262144; // config: max_position_embeddings

    float rms_norm_eps = 1e-6f;      // config: rms_norm_eps
    bool tie_word_embeddings = true; // config: tie_word_embeddings (no lm_head)

    // RoPE (partial + interleaved mRoPE). Only partial_rotary_factor of each
    // head's dims are rotated; theta is large for long-context.
    double rope_theta = 10'000'000.0; // config: rope_parameters.rope_theta
    float partial_rotary_factor = 0.25f; // config: partial_rotary_factor

    int attn_interval = 4;           // config: full_attention_interval

    // Returns true if layer index `layer` (0-based) is a full-attention layer:
    // every `attn_interval`-th layer (config: full_attention_interval).
    bool is_attention_layer(int layer) const {
        return ((layer + 1) % attn_interval == 0);
    }
};

// Populates `cfg` from a Hugging Face config.json (reads its `text_config`).
// Returns false if the file can't be read; leaves fields at their defaults for
// any key that's absent, so a partial config still yields a usable ModelConfig.
bool load_model_config(const std::string& config_json_path, ModelConfig& cfg);

// Bytes-per-element for common quantization / precision choices.
enum class Precision {
    FP32 = 4,
    FP16 = 2,
    INT8 = 1,
    INT4_PACKED = 0 // handled specially: 0.5 bytes/param, see memory_budget.cpp
};
