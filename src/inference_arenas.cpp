#include "inference_arenas.h"
#include "gguf_loader.h"
#include <cstring>
#include <iostream>

InferenceArenas::~InferenceArenas() = default;

bool InferenceArenas::allocate(const std::string& gguf_path,
                                const ModelConfig& cfg,
                                const MemoryBudget& budget,
                                int max_context,
                                int kv_bytes_per_elem) {
    cfg_ = cfg;
    max_context_ = max_context;
    kv_bytes_per_elem_ = (size_t)kv_bytes_per_elem;

    // --- Arena 1: weights ---
    // GGUF parsing lives in gguf_loader.cpp; this call reads the file into
    // weight_data_ and fills tensor_offsets_ with name -> byte-offset.
    if (!load_gguf(gguf_path, weight_data_, tensor_offsets_)) {
        std::cerr << "Failed to load weights from " << gguf_path << "\n";
        return false;
    }

    // --- Arena 2: GDN state ---
    // One fixed-size [heads, head_dim, head_dim] matrix per GDN layer.
    gdn_state_bytes_per_layer_ =
        (size_t)cfg_.gdn_n_heads * cfg_.gdn_head_dim * cfg_.gdn_head_dim;
    gdn_state_.assign(gdn_state_bytes_per_layer_ * cfg_.n_layers_gdn, 0.0f);

    // --- Arena 3: KV cache ---
    // [n_layers_attn, max_context, 2 (K/V), kv_heads, head_dim], stored as
    // raw bytes so total size always matches budget.kv_cache_bytes exactly,
    // regardless of kv_bytes_per_elem (fp16, int8, ...).
    kv_cache_stride_bytes_per_layer_ =
        (size_t)max_context_ * 2 * cfg_.attn_kv_heads * cfg_.attn_head_dim
        * kv_bytes_per_elem_;
    kv_cache_.assign(kv_cache_stride_bytes_per_layer_ * cfg_.n_layers_attn, 0);

    // Sanity check: this must equal budget.kv_cache_bytes, or the equation
    // and the arena have drifted out of sync again.
    if (kv_cache_.size() != budget.kv_cache_bytes) {
        std::cerr << "[inference_arenas] WARNING: kv_cache size mismatch - "
                  << "arena=" << kv_cache_.size() << " bytes, budget="
                  << budget.kv_cache_bytes << " bytes\n";
    }

    // --- Arena 4: shared scratch ---
    size_t scratch_floats = (budget.activation_bytes + budget.vision_scratch_bytes)
                             / sizeof(float);
    scratch_.assign(scratch_floats, 0.0f);

    std::cout << "Arenas allocated:\n"
              << "  weights:  " << weight_data_.size() / (1024 * 1024) << " MB (mapped)\n"
              << "  gdn_state:" << (gdn_state_.size() * sizeof(float)) / 1024 << " KB\n"
              << "  kv_cache: " << kv_cache_.size() / (1024 * 1024) << " MB\n"
              << "  scratch:  " << (scratch_.size() * sizeof(float)) / 1024 << " KB\n";

    return true;
}

void InferenceArenas::reset_for_new_request() {
    std::fill(gdn_state_.begin(), gdn_state_.end(), 0.0f);
    // kv_cache_ is intentionally NOT zeroed: stale bytes beyond the new
    // write position are never read, since attention only looks back to
    // kv_write_pos_. Rewinding the pointer is sufficient and avoids an
    // unnecessary full-buffer memset every request.
    kv_write_pos_ = 0;
}

const float* InferenceArenas::get_weight(const std::string& tensor_name) const {
    auto it = tensor_offsets_.find(tensor_name);
    if (it == tensor_offsets_.end()) {
        return nullptr; // caller should check; see TODO in forward pass
    }
    return reinterpret_cast<const float*>(weight_data_.data() + it->second);
}

float* InferenceArenas::gdn_state_slot(int gdn_layer_idx) {
    return gdn_state_.data() + (size_t)gdn_layer_idx * gdn_state_bytes_per_layer_;
}

uint8_t* InferenceArenas::kv_cache_slot(int attn_layer_idx) {
    return kv_cache_.data() + (size_t)attn_layer_idx * kv_cache_stride_bytes_per_layer_;
}

float* InferenceArenas::scratch() {
    return scratch_.data();
}
