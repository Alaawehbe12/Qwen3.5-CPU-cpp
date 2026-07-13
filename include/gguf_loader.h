#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// GGUF loader
//
// Parses the GGUF container format (v2/v3) used by llama.cpp / ggml:
//   [header] [metadata KV pairs] [tensor descriptors] [padding] [tensor data]
//
// Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//
// This reads the whole file into memory (portable; no mmap so it builds the
// same on Windows/MSVC as on POSIX) and builds a name -> byte-offset table
// pointing into that buffer, plus typed metadata and per-tensor descriptors.
// ---------------------------------------------------------------------------

// ggml tensor data types we recognize (subset of the full enum, covering the
// formats a Qwen3.5 GGUF actually ships in). Values match ggml_type.
enum class GgmlType : uint32_t {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    Q2_K  = 10,
    Q3_K  = 11,
    Q4_K  = 12,
    Q5_K  = 13,
    Q6_K  = 14,
    Q8_K  = 15,
    I8    = 24,
    I16   = 25,
    I32   = 26,
    I64   = 27,
    F64   = 28,
    UNKNOWN = 0xFFFFFFFFu
};

// (block size in elements, bytes per block) for a ggml type. Returns false for
// a type we don't have traits for, so the caller can fail loudly instead of
// computing a wrong tensor size.
bool ggml_type_traits(uint32_t type, uint32_t& block_size, uint32_t& type_size);
const char* ggml_type_name(uint32_t type);

// One entry from the tensor descriptor table.
struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;   // ne[]: element counts per dimension
    uint32_t type = 0;             // GgmlType
    uint64_t rel_offset = 0;       // offset from start of the data section
    size_t   abs_offset = 0;       // offset into the loaded data buffer
    size_t   n_bytes = 0;          // computed on-disk size of this tensor
    uint64_t n_elements() const;
};

// Typed metadata value. GGUF values are heterogeneous; we keep a printable
// string form for everything and expose typed getters for the keys that
// matter to the memory planner.
struct GgufModel {
    bool     ok = false;
    bool     is_dummy = false;      // true when no file was found (fallback)
    uint32_t version = 0;
    uint32_t alignment = 32;        // general.alignment (default 32)
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    size_t   data_section_start = 0;

    // Raw metadata as strings, keyed by GGUF key (e.g. "general.architecture").
    std::unordered_map<std::string, std::string> meta;

    std::vector<GgufTensorInfo> tensors;
    std::unordered_map<std::string, size_t> tensor_index; // name -> index

    // Aggregates derived from the tensor table.
    int64_t n_params = 0;       // summed element count across all tensors
    size_t  total_tensor_bytes = 0; // summed on-disk bytes (the real weight_bytes)

    // Convenience lookups (return fallback if key absent / wrong type).
    std::string get_string(const std::string& key, const std::string& fallback = "") const;
    int64_t     get_int(const std::string& key, int64_t fallback = 0) const;

    const GgufTensorInfo* find(const std::string& name) const;
};

// Full parse. On success returns a populated GgufModel; `out_data` owns the
// raw file bytes that the tensor offsets point into.
//
// If the file is missing, returns a model with is_dummy=true and a small
// zero-filled out_data, so the rest of the pipeline can still run without a
// real model (matches the previous placeholder behavior).
GgufModel parse_gguf(const std::string& path, std::vector<uint8_t>& out_data);

// Backwards-compatible thin wrapper used by InferenceArenas::allocate():
// loads the file and fills a name -> absolute-byte-offset map.
bool load_gguf(const std::string& path,
               std::vector<uint8_t>& out_data,
               std::unordered_map<std::string, size_t>& out_offsets);
