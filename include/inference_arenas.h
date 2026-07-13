#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "model_config.h"
#include "memory_budget.h"

// Owns the four fixed-size memory regions for single-sequence inference:
//   1. weights_   - static, loaded once, read-only for process lifetime
//   2. gdn_state_ - fixed-size recurrent state, reset (zeroed) per request
//   3. kv_cache_  - fixed-size (sized for max_context), reused per request
//   4. scratch_   - per-layer activation + vision working space, reused
//
// No allocation happens outside allocate(); no arena grows or shrinks
// during a request. reset_for_new_request() rewinds state instead of
// freeing/reallocating, since context is dropped between turns.
class InferenceArenas {
public:
    ~InferenceArenas();

    // Loads model weights from a GGUF file into weights_ and allocates the
    // other three arenas according to `budget`.
    bool allocate(const std::string& gguf_path,
                  const ModelConfig& cfg,
                  const MemoryBudget& budget,
                  int max_context,
                  int kv_bytes_per_elem);

    // Zeroes GDN state and rewinds the KV cache write position.
    // Called once per new prompt (context is dropped between turns).
    void reset_for_new_request();

    // Weight lookup by GGUF tensor name, e.g. "blk.0.gdn.q_proj.weight".
    const float* get_weight(const std::string& tensor_name) const;

    // Slot accessors used by the forward pass (see forward_pass.cpp, not yet
    // implemented — this header only exposes the memory layout).
    float* gdn_state_slot(int gdn_layer_idx);
    uint8_t* kv_cache_slot(int attn_layer_idx); // caller casts per kv_bytes_per_elem_
    float* scratch();

    size_t kv_write_pos() const { return kv_write_pos_; }
    void advance_kv_write_pos() { kv_write_pos_++; }

    const ModelConfig& config() const { return cfg_; }
    int max_context() const { return max_context_; }
    size_t kv_bytes_per_elem() const { return kv_bytes_per_elem_; }

private:
    ModelConfig cfg_;
    int max_context_ = 0;

    // Arena 1: weights (owns the raw file buffer; see gguf_loader.cpp)
    std::vector<uint8_t> weight_data_;
    std::unordered_map<std::string, size_t> tensor_offsets_;

    // Arena 2: GDN state, one fixed slot per GDN layer
    std::vector<float> gdn_state_;
    size_t gdn_state_bytes_per_layer_ = 0;

    // Arena 3: KV cache, one growing-but-capped slot per Attention layer.
    // Stored as raw bytes (not float) so its size always matches whatever
    // kv_bytes_per_elem the budget equation was computed with (fp16, int8,
    // etc.) instead of silently assuming 4-byte float, as it did before.
    std::vector<uint8_t> kv_cache_;
    size_t kv_cache_stride_bytes_per_layer_ = 0;
    size_t kv_bytes_per_elem_ = 2; // must match compute_budget()'s argument
    size_t kv_write_pos_ = 0;

    // Arena 4: shared scratch, reused by every layer in sequence
    std::vector<float> scratch_;
};
