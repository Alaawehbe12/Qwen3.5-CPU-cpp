#include "safetensors_loader.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <system_error>

// ---------------------------------------------------------------------------
// Minimal JSON parser, just enough for a safetensors header:
//   objects, arrays, strings, numbers (int64), plus skipping of bool/null.
// The header is small (~60 KB) and well-formed, so this stays simple.
// ---------------------------------------------------------------------------
namespace {

struct JsonParser {
    const char* p;
    const char* end;
    bool ok = true;

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    bool expect(char c) {
        skip_ws();
        if (p < end && *p == c) { ++p; return true; }
        ok = false;
        return false;
    }

    std::string parse_string() {
        std::string s;
        skip_ws();
        if (p >= end || *p != '"') { ok = false; return s; }
        ++p;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) { // handle simple escapes
                ++p;
                switch (*p) {
                    case 'n': s.push_back('\n'); break;
                    case 't': s.push_back('\t'); break;
                    case '"': s.push_back('"'); break;
                    case '\\': s.push_back('\\'); break;
                    case '/': s.push_back('/'); break;
                    default: s.push_back(*p); break;
                }
            } else {
                s.push_back(*p);
            }
            ++p;
        }
        if (p >= end) { ok = false; return s; }
        ++p; // closing quote
        return s;
    }

    int64_t parse_int() {
        skip_ws();
        const char* start = p;
        if (p < end && (*p == '-' || *p == '+')) ++p;
        while (p < end && *p >= '0' && *p <= '9') ++p;
        if (p == start) { ok = false; return 0; }
        return std::strtoll(std::string(start, p).c_str(), nullptr, 10);
    }

    std::vector<int64_t> parse_int_array() {
        std::vector<int64_t> v;
        if (!expect('[')) return v;
        skip_ws();
        if (p < end && *p == ']') { ++p; return v; }
        while (ok) {
            v.push_back(parse_int());
            skip_ws();
            if (p < end && *p == ',') { ++p; continue; }
            break;
        }
        expect(']');
        return v;
    }

    // Skip any value (used for __metadata__ and unknown fields).
    void skip_value() {
        skip_ws();
        if (p >= end) { ok = false; return; }
        char c = *p;
        if (c == '"') { parse_string(); }
        else if (c == '{') { skip_object(); }
        else if (c == '[') {
            ++p;
            skip_ws();
            if (p < end && *p == ']') { ++p; return; }
            while (ok) { skip_value(); skip_ws();
                        if (p < end && *p == ',') { ++p; continue; } break; }
            expect(']');
        } else if (c == 't' || c == 'f') { // true / false
            while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
        } else if (c == 'n') { // null
            while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
        } else { // number
            if (p < end && (*p == '-' || *p == '+')) ++p;
            while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' ||
                               *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) ++p;
        }
    }

    void skip_object() {
        if (!expect('{')) return;
        skip_ws();
        if (p < end && *p == '}') { ++p; return; }
        while (ok) {
            parse_string();
            expect(':');
            skip_value();
            skip_ws();
            if (p < end && *p == ',') { ++p; continue; }
            break;
        }
        expect('}');
    }
};

} // namespace

int64_t SafeTensors::Entry::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

bool SafeTensors::parse_shard(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[safetensors] cannot open '" << path << "'\n";
        return false;
    }
    size_t file_size = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);
    if (file_size < 8) { std::cerr << "[safetensors] file too small\n"; return false; }

    const int shard_idx = (int)shards_.size();
    shards_.emplace_back(file_size);
    std::vector<uint8_t>& data_ = shards_.back();
    if (!file.read(reinterpret_cast<char*>(data_.data()), (std::streamsize)file_size)) {
        std::cerr << "[safetensors] read failed\n";
        return false;
    }

    uint64_t header_len = 0;
    std::memcpy(&header_len, data_.data(), 8);
    if (8 + header_len > file_size) {
        std::cerr << "[safetensors] header length exceeds file\n";
        return false;
    }
    const size_t data_base = 8 + (size_t)header_len;

    JsonParser jp;
    jp.p = reinterpret_cast<const char*>(data_.data()) + 8;
    jp.end = jp.p + header_len;

    if (!jp.expect('{')) { std::cerr << "[safetensors] bad header\n"; return false; }
    jp.skip_ws();
    if (jp.p < jp.end && *jp.p == '}') return true; // empty
    while (jp.ok) {
        std::string key = jp.parse_string();
        jp.expect(':');
        if (key == "__metadata__") {
            jp.skip_object();
        } else {
            // Tensor object: {"dtype":..,"shape":[..],"data_offsets":[a,b]}
            Entry e;
            int64_t off_start = 0, off_end = 0;
            jp.expect('{');
            while (jp.ok) {
                std::string field = jp.parse_string();
                jp.expect(':');
                if (field == "dtype") {
                    e.dtype = jp.parse_string();
                } else if (field == "shape") {
                    e.shape = jp.parse_int_array();
                } else if (field == "data_offsets") {
                    auto v = jp.parse_int_array();
                    if (v.size() == 2) { off_start = v[0]; off_end = v[1]; }
                } else {
                    jp.skip_value();
                }
                jp.skip_ws();
                if (jp.p < jp.end && *jp.p == ',') { ++jp.p; continue; }
                break;
            }
            jp.expect('}');

            e.shard = shard_idx;
            e.abs_offset = data_base + (size_t)off_start;
            e.nbytes = (size_t)(off_end - off_start);
            if (e.abs_offset + e.nbytes > file_size) {
                std::cerr << "[safetensors] tensor '" << key << "' out of range\n";
                return false;
            }
            entries_.emplace(std::move(key), std::move(e));
        }
        jp.skip_ws();
        if (jp.p < jp.end && *jp.p == ',') { ++jp.p; continue; }
        break;
    }
    jp.expect('}');
    if (!jp.ok) { std::cerr << "[safetensors] header parse error\n"; return false; }
    std::cout << "[safetensors] shard '" << path << "': "
              << (file_size / (1024 * 1024)) << " MB\n";
    return true;
}

bool SafeTensors::load(const std::string& path) {
    shards_.clear(); entries_.clear(); f32_cache_.clear();
    return parse_shard(path);
}

bool SafeTensors::load_model(const std::string& model_dir) {
    shards_.clear(); entries_.clear(); f32_cache_.clear();
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto& de : std::filesystem::directory_iterator(model_dir, ec)) {
        if (de.path().extension() == ".safetensors") files.push_back(de.path().string());
    }
    if (files.empty()) {
        std::cerr << "[safetensors] no .safetensors files in '" << model_dir << "'\n";
        return false;
    }
    std::sort(files.begin(), files.end()); // deterministic shard order
    for (const auto& f : files) if (!parse_shard(f)) return false;
    std::cout << "[safetensors] " << files.size() << " shard(s), "
              << entries_.size() << " tensors\n";
    return true;
}

const SafeTensors::Entry* SafeTensors::find(const std::string& name) const {
    auto it = entries_.find(name);
    return it == entries_.end() ? nullptr : &it->second;
}

WeightView SafeTensors::view(const std::string& name) const {
    WeightView v;
    const Entry* e = find(name);
    if (!e) return v;
    const uint8_t* p = shards_[e->shard].data() + e->abs_offset;
    v.numel = e->numel();
    if (e->dtype == "BF16")      v.bf16 = reinterpret_cast<const uint16_t*>(p);
    else if (e->dtype == "F32")  v.f32  = reinterpret_cast<const float*>(p);
    return v;
}

const float* SafeTensors::get_f32(const std::string& name, int64_t* out_numel) {
    auto cit = f32_cache_.find(name);
    if (cit != f32_cache_.end()) {
        if (out_numel) *out_numel = (int64_t)cit->second.size();
        return cit->second.data();
    }
    const Entry* e = find(name);
    if (!e) return nullptr;

    int64_t n = e->numel();
    std::vector<float> buf((size_t)n);
    const uint8_t* src = shards_[e->shard].data() + e->abs_offset;
    if (e->dtype == "F32") {
        std::memcpy(buf.data(), src, (size_t)n * 4);
    } else if (e->dtype == "BF16") {
        const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
        for (int64_t i = 0; i < n; ++i) buf[(size_t)i] = bf16_to_f32(s[i]);
    } else {
        std::cerr << "[safetensors] unsupported dtype " << e->dtype
                  << " for '" << name << "'\n";
        return nullptr;
    }

    auto res = f32_cache_.emplace(name, std::move(buf));
    if (out_numel) *out_numel = (int64_t)res.first->second.size();
    return res.first->second.data();
}

bool SafeTensors::get_row_f32(const std::string& name, int64_t row, float* dst) const {
    const Entry* e = find(name);
    if (!e || e->shape.size() != 2) return false;
    int64_t rows = e->shape[0], cols = e->shape[1];
    if (row < 0 || row >= rows) return false;

    const uint8_t* base = shards_[e->shard].data() + e->abs_offset;
    if (e->dtype == "F32") {
        const float* s = reinterpret_cast<const float*>(base);
        std::memcpy(dst, s + row * cols, (size_t)cols * 4);
    } else if (e->dtype == "BF16") {
        const uint16_t* s = reinterpret_cast<const uint16_t*>(base);
        s += row * cols;
        for (int64_t i = 0; i < cols; ++i) dst[i] = bf16_to_f32(s[i]);
    } else {
        return false;
    }
    return true;
}
