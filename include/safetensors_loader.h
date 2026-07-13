#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include "weight_view.h"

// ---------------------------------------------------------------------------
// Safetensors loader
//
// The HuggingFace .safetensors format:
//   uint64 header_len (little-endian)
//   header_len bytes of JSON: { "<name>": {"dtype","shape","data_offsets"}, ...,
//                               "__metadata__": {...} }
//   raw tensor data (each tensor at 8 + header_len + data_offsets[0])
//
// This model ships weights as BF16 (a few small tensors are F32). We convert
// to fp32 on demand and cache the result, so the forward pass sees plain
// float* buffers. bf16->f32 is exact: take the 16 bits as the high half of a
// 32-bit float (bf16 == truncated fp32).
// ---------------------------------------------------------------------------

class SafeTensors {
public:
    struct Entry {
        std::string dtype;              // "BF16" or "F32"
        std::vector<int64_t> shape;
        int shard = 0;                  // which shard buffer holds the data
        size_t abs_offset = 0;          // byte offset into that shard buffer
        size_t nbytes = 0;              // on-disk byte size
        int64_t numel() const;
    };

    // Loads every *.safetensors file in `model_dir` (handles both single-file
    // and sharded models) and merges their tensor tables.
    bool load_model(const std::string& model_dir);

    // Reads a single .safetensors file and parses its header.
    bool load(const std::string& path);

    const Entry* find(const std::string& name) const;
    bool has(const std::string& name) const { return find(name) != nullptr; }

    // Returns an fp32 view of the named tensor (converting bf16 once, cached).
    // Returns nullptr if the tensor is missing. `out_numel` (optional) receives
    // the element count.
    const float* get_f32(const std::string& name, int64_t* out_numel = nullptr);

    // In-place view of a tensor (no copy, no conversion) as bf16 or fp32 — used
    // by the matmul so big weights never get a separate fp32 cache.
    WeightView view(const std::string& name) const;

    // Copies a single row (row-major, last dim = row_len) of a 2-D tensor into
    // `dst` as fp32, without materializing the whole tensor. Used for embedding
    // lookups on the huge [vocab, hidden] table.
    bool get_row_f32(const std::string& name, int64_t row, float* dst) const;

    const std::unordered_map<std::string, Entry>& entries() const { return entries_; }

private:
    bool parse_shard(const std::string& path); // append one shard's tensors
    std::vector<std::vector<uint8_t>> shards_;
    std::unordered_map<std::string, Entry> entries_;
    std::unordered_map<std::string, std::vector<float>> f32_cache_;
};

// Exact bf16 -> fp32 widening (bf16 is the top 16 bits of an fp32).
inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
