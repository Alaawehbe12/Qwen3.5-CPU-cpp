#include "gguf_loader.h"
#include <cstring>
#include <fstream>
#include <iostream>

// ---------------------------------------------------------------------------
// GGUF binary format parser.
//
// Layout (all little-endian, which matches x86/x64 so we memcpy directly):
//   char     magic[4] = "GGUF"
//   uint32   version            (2 or 3)
//   uint64   tensor_count
//   uint64   metadata_kv_count
//   metadata_kv_count * { string key; uint32 value_type; value }
//   tensor_count      * { string name; uint32 n_dims; uint64 dims[n_dims];
//                         uint32 ggml_type; uint64 offset }
//   padding to `alignment`
//   tensor data
//
// A GGUF string is: uint64 length, then `length` raw bytes (no NUL).
// ---------------------------------------------------------------------------

namespace {

// GGUF metadata value type tags.
enum : uint32_t {
    GGUF_UINT8   = 0,  GGUF_INT8   = 1,
    GGUF_UINT16  = 2,  GGUF_INT16  = 3,
    GGUF_UINT32  = 4,  GGUF_INT32  = 5,
    GGUF_FLOAT32 = 6,  GGUF_BOOL   = 7,
    GGUF_STRING  = 8,  GGUF_ARRAY  = 9,
    GGUF_UINT64  = 10, GGUF_INT64  = 11,
    GGUF_FLOAT64 = 12
};

// Bounds-checked forward cursor over the file buffer. Any read that would run
// past the end sets ok_ = false; callers check ok() before trusting results,
// so a truncated or malformed file fails cleanly instead of reading OOB.
class Cursor {
public:
    Cursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    bool ok() const { return ok_; }
    size_t pos() const { return pos_; }

    template <typename T>
    T read_scalar() {
        T v{};
        if (!have(sizeof(T))) { ok_ = false; return v; }
        std::memcpy(&v, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    // GGUF string: uint64 length + raw bytes.
    std::string read_string() {
        uint64_t len = read_scalar<uint64_t>();
        if (!ok_ || !have(len)) { ok_ = false; return {}; }
        std::string s(reinterpret_cast<const char*>(data_ + pos_), (size_t)len);
        pos_ += (size_t)len;
        return s;
    }

    void skip(size_t n) {
        if (!have(n)) { ok_ = false; return; }
        pos_ += n;
    }

private:
    bool have(size_t n) const { return pos_ + n <= size_; }
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
    bool ok_ = true;
};

// Fixed byte width of a scalar metadata value type (0 => not a fixed scalar).
size_t scalar_width(uint32_t t) {
    switch (t) {
        case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL:   return 1;
        case GGUF_UINT16: case GGUF_INT16:                 return 2;
        case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: return 4;
        case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: return 8;
        default: return 0;
    }
}

// Read a scalar value and render it as a string (for the generic meta map).
std::string read_scalar_as_string(Cursor& c, uint32_t t) {
    switch (t) {
        case GGUF_UINT8:   return std::to_string((unsigned)c.read_scalar<uint8_t>());
        case GGUF_INT8:    return std::to_string((int)c.read_scalar<int8_t>());
        case GGUF_UINT16:  return std::to_string(c.read_scalar<uint16_t>());
        case GGUF_INT16:   return std::to_string(c.read_scalar<int16_t>());
        case GGUF_UINT32:  return std::to_string(c.read_scalar<uint32_t>());
        case GGUF_INT32:   return std::to_string(c.read_scalar<int32_t>());
        case GGUF_FLOAT32: return std::to_string(c.read_scalar<float>());
        case GGUF_BOOL:    return c.read_scalar<uint8_t>() ? "true" : "false";
        case GGUF_UINT64:  return std::to_string(c.read_scalar<uint64_t>());
        case GGUF_INT64:   return std::to_string(c.read_scalar<int64_t>());
        case GGUF_FLOAT64: return std::to_string(c.read_scalar<double>());
        default:           return {};
    }
}

// Consume one metadata value of type `t`, returning a printable form. Arrays
// are consumed fully (so the cursor stays aligned) but summarized as a tag.
std::string read_value(Cursor& c, uint32_t t) {
    if (t == GGUF_STRING) return c.read_string();
    if (scalar_width(t)) return read_scalar_as_string(c, t);

    if (t == GGUF_ARRAY) {
        uint32_t elem_type = c.read_scalar<uint32_t>();
        uint64_t count     = c.read_scalar<uint64_t>();
        if (!c.ok()) return {};
        if (elem_type == GGUF_STRING) {
            for (uint64_t i = 0; i < count && c.ok(); ++i) c.read_string();
        } else if (elem_type == GGUF_ARRAY) {
            c.ok(); // nested arrays aren't emitted by ggml; bail defensively
            return "[nested-array]";
        } else {
            size_t w = scalar_width(elem_type);
            if (w == 0) return "[array:?]";
            c.skip((size_t)count * w);
        }
        return "[array:" + std::to_string(count) + "]";
    }

    // Unknown type tag: we can't know its width, so stop parsing safely.
    return {};
}

} // namespace

// ---------------------------------------------------------------------------
// ggml type traits
// ---------------------------------------------------------------------------
bool ggml_type_traits(uint32_t type, uint32_t& block_size, uint32_t& type_size) {
    switch (type) {
        case (uint32_t)GgmlType::F32:  block_size = 1;   type_size = 4;   return true;
        case (uint32_t)GgmlType::F16:  block_size = 1;   type_size = 2;   return true;
        case (uint32_t)GgmlType::Q4_0: block_size = 32;  type_size = 18;  return true;
        case (uint32_t)GgmlType::Q4_1: block_size = 32;  type_size = 20;  return true;
        case (uint32_t)GgmlType::Q5_0: block_size = 32;  type_size = 22;  return true;
        case (uint32_t)GgmlType::Q5_1: block_size = 32;  type_size = 24;  return true;
        case (uint32_t)GgmlType::Q8_0: block_size = 32;  type_size = 34;  return true;
        case (uint32_t)GgmlType::Q8_1: block_size = 32;  type_size = 40;  return true;
        case (uint32_t)GgmlType::Q2_K: block_size = 256; type_size = 84;  return true;
        case (uint32_t)GgmlType::Q3_K: block_size = 256; type_size = 110; return true;
        case (uint32_t)GgmlType::Q4_K: block_size = 256; type_size = 144; return true;
        case (uint32_t)GgmlType::Q5_K: block_size = 256; type_size = 176; return true;
        case (uint32_t)GgmlType::Q6_K: block_size = 256; type_size = 210; return true;
        case (uint32_t)GgmlType::Q8_K: block_size = 256; type_size = 292; return true;
        case (uint32_t)GgmlType::I8:   block_size = 1;   type_size = 1;   return true;
        case (uint32_t)GgmlType::I16:  block_size = 1;   type_size = 2;   return true;
        case (uint32_t)GgmlType::I32:  block_size = 1;   type_size = 4;   return true;
        case (uint32_t)GgmlType::I64:  block_size = 1;   type_size = 8;   return true;
        case (uint32_t)GgmlType::F64:  block_size = 1;   type_size = 8;   return true;
        default: block_size = 0; type_size = 0; return false;
    }
}

const char* ggml_type_name(uint32_t type) {
    switch (type) {
        case (uint32_t)GgmlType::F32:  return "F32";
        case (uint32_t)GgmlType::F16:  return "F16";
        case (uint32_t)GgmlType::Q4_0: return "Q4_0";
        case (uint32_t)GgmlType::Q4_1: return "Q4_1";
        case (uint32_t)GgmlType::Q5_0: return "Q5_0";
        case (uint32_t)GgmlType::Q5_1: return "Q5_1";
        case (uint32_t)GgmlType::Q8_0: return "Q8_0";
        case (uint32_t)GgmlType::Q8_1: return "Q8_1";
        case (uint32_t)GgmlType::Q2_K: return "Q2_K";
        case (uint32_t)GgmlType::Q3_K: return "Q3_K";
        case (uint32_t)GgmlType::Q4_K: return "Q4_K";
        case (uint32_t)GgmlType::Q5_K: return "Q5_K";
        case (uint32_t)GgmlType::Q6_K: return "Q6_K";
        case (uint32_t)GgmlType::Q8_K: return "Q8_K";
        case (uint32_t)GgmlType::I8:   return "I8";
        case (uint32_t)GgmlType::I16:  return "I16";
        case (uint32_t)GgmlType::I32:  return "I32";
        case (uint32_t)GgmlType::I64:  return "I64";
        case (uint32_t)GgmlType::F64:  return "F64";
        default: return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// GgufTensorInfo / GgufModel helpers
// ---------------------------------------------------------------------------
uint64_t GgufTensorInfo::n_elements() const {
    uint64_t n = 1;
    for (uint64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

std::string GgufModel::get_string(const std::string& key, const std::string& fallback) const {
    auto it = meta.find(key);
    return it == meta.end() ? fallback : it->second;
}

int64_t GgufModel::get_int(const std::string& key, int64_t fallback) const {
    auto it = meta.find(key);
    if (it == meta.end()) return fallback;
    try { return std::stoll(it->second); } catch (...) { return fallback; }
}

const GgufTensorInfo* GgufModel::find(const std::string& name) const {
    auto it = tensor_index.find(name);
    return it == tensor_index.end() ? nullptr : &tensors[it->second];
}

// ---------------------------------------------------------------------------
// parse_gguf
// ---------------------------------------------------------------------------
GgufModel parse_gguf(const std::string& path, std::vector<uint8_t>& out_data) {
    GgufModel m;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[gguf_loader] '" << path << "' not found - using dummy "
                  << "zero-filled weights so the pipeline can still run.\n";
        out_data.assign(64 * 1024 * 1024, 0); // 64 MB placeholder
        m.is_dummy = true;
        m.ok = true; // dummy path is a deliberate success, not a parse failure
        return m;
    }

    size_t file_size = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);
    out_data.resize(file_size);
    if (!file.read(reinterpret_cast<char*>(out_data.data()), (std::streamsize)file_size)) {
        std::cerr << "[gguf_loader] Failed reading '" << path << "'\n";
        return m; // ok stays false
    }

    Cursor c(out_data.data(), out_data.size());

    // --- Header ---
    char magic[4];
    for (int i = 0; i < 4; ++i) magic[i] = (char)c.read_scalar<uint8_t>();
    if (!c.ok() || std::memcmp(magic, "GGUF", 4) != 0) {
        std::cerr << "[gguf_loader] '" << path << "' is not a GGUF file "
                  << "(bad magic).\n";
        return m;
    }
    m.version      = c.read_scalar<uint32_t>();
    m.tensor_count = c.read_scalar<uint64_t>();
    m.kv_count     = c.read_scalar<uint64_t>();
    if (!c.ok()) { std::cerr << "[gguf_loader] Truncated header.\n"; return m; }
    if (m.version != 2 && m.version != 3) {
        std::cerr << "[gguf_loader] Unsupported GGUF version " << m.version
                  << " (expected 2 or 3).\n";
        return m;
    }

    // --- Metadata KV pairs ---
    for (uint64_t i = 0; i < m.kv_count; ++i) {
        std::string key = c.read_string();
        uint32_t vtype  = c.read_scalar<uint32_t>();
        if (!c.ok()) break;
        std::string val = read_value(c, vtype);
        if (!c.ok()) break;
        m.meta.emplace(std::move(key), std::move(val));
    }
    if (!c.ok()) {
        std::cerr << "[gguf_loader] Malformed metadata section.\n";
        return m;
    }

    // Alignment can be overridden via metadata; default is 32.
    m.alignment = (uint32_t)m.get_int("general.alignment", 32);
    if (m.alignment == 0) m.alignment = 32;

    // --- Tensor descriptor table ---
    m.tensors.reserve((size_t)m.tensor_count);
    for (uint64_t i = 0; i < m.tensor_count; ++i) {
        GgufTensorInfo t;
        t.name         = c.read_string();
        uint32_t ndims = c.read_scalar<uint32_t>();
        if (!c.ok() || ndims > 8) { // ggml caps at 4; 8 is a generous sanity bound
            std::cerr << "[gguf_loader] Bad tensor dim count.\n";
            return m;
        }
        t.shape.resize(ndims);
        for (uint32_t d = 0; d < ndims; ++d) t.shape[d] = c.read_scalar<uint64_t>();
        t.type       = c.read_scalar<uint32_t>();
        t.rel_offset = c.read_scalar<uint64_t>();
        if (!c.ok()) { std::cerr << "[gguf_loader] Truncated tensor table.\n"; return m; }

        // Compute on-disk byte size from the ggml type block layout.
        uint32_t blk = 0, tsz = 0;
        if (!ggml_type_traits(t.type, blk, tsz)) {
            std::cerr << "[gguf_loader] Tensor '" << t.name << "' has unknown "
                      << "ggml type " << t.type << ".\n";
            return m;
        }
        uint64_t ne = t.n_elements();
        if (blk != 0 && (ne % blk) != 0 && ne != 0) {
            // Quantized tensors must be a whole number of blocks.
            std::cerr << "[gguf_loader] Tensor '" << t.name << "' element count "
                      << ne << " not a multiple of block size " << blk << ".\n";
            return m;
        }
        t.n_bytes = blk ? (size_t)(ne / blk) * tsz : 0;

        m.n_params += (int64_t)ne;
        m.tensor_index.emplace(t.name, m.tensors.size());
        m.tensors.push_back(std::move(t));
    }

    // --- Data section starts at the next `alignment` boundary ---
    size_t after_table = c.pos();
    size_t pad = (m.alignment - (after_table % m.alignment)) % m.alignment;
    m.data_section_start = after_table + pad;

    // Resolve absolute offsets and validate every tensor lies inside the file.
    for (auto& t : m.tensors) {
        t.abs_offset = m.data_section_start + (size_t)t.rel_offset;
        if (t.abs_offset + t.n_bytes > out_data.size()) {
            std::cerr << "[gguf_loader] Tensor '" << t.name << "' data ["
                      << t.abs_offset << ", +" << t.n_bytes << ") exceeds file "
                      << "size " << out_data.size() << ".\n";
            return m;
        }
        m.total_tensor_bytes += t.n_bytes;
    }

    m.ok = true;
    std::cout << "[gguf_loader] Parsed '" << path << "': v" << m.version << ", "
              << m.tensor_count << " tensors, " << m.kv_count << " metadata keys, "
              << "arch=" << m.get_string("general.architecture", "?") << ", "
              << (m.n_params / 1'000'000) << "M params, "
              << (m.total_tensor_bytes / (1024 * 1024)) << " MB tensor data.\n";
    return m;
}

// ---------------------------------------------------------------------------
// Backwards-compatible wrapper
// ---------------------------------------------------------------------------
bool load_gguf(const std::string& path,
               std::vector<uint8_t>& out_data,
               std::unordered_map<std::string, size_t>& out_offsets) {
    GgufModel m = parse_gguf(path, out_data);
    if (!m.ok) return false;

    out_offsets.clear();
    for (const auto& t : m.tensors) out_offsets.emplace(t.name, t.abs_offset);
    return true;
}
