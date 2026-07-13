#include "model_config.h"
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

// Lightweight config.json reader. We only need scalar fields out of the
// `text_config` object, so rather than a full JSON parser we extract that
// object's substring and scan for "key": value pairs within it.
namespace {

// Return the substring of the JSON object that follows "key": { ... },
// brace-matched. Empty if not found.
std::string extract_object(const std::string& s, const std::string& key) {
    size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return {};
    size_t b = s.find('{', k);
    if (b == std::string::npos) return {};
    int depth = 0;
    for (size_t i = b; i < s.size(); ++i) {
        if (s[i] == '{') ++depth;
        else if (s[i] == '}') { if (--depth == 0) return s.substr(b, i - b + 1); }
    }
    return {};
}

// Find "key": then parse the following JSON number. Returns false if absent.
bool find_number(const std::string& s, const std::string& key, double& out) {
    size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return false;
    size_t c = s.find(':', k);
    if (c == std::string::npos) return false;
    size_t i = c + 1;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    size_t start = i;
    while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '-' || s[i] == '+'
                            || s[i] == '.' || s[i] == 'e' || s[i] == 'E')) ++i;
    if (i == start) return false;
    try { out = std::stod(s.substr(start, i - start)); } catch (...) { return false; }
    return true;
}

bool find_bool(const std::string& s, const std::string& key, bool& out) {
    size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return false;
    size_t c = s.find(':', k);
    if (c == std::string::npos) return false;
    size_t e = s.find(',', c); if (e == std::string::npos) e = s.size();
    out = s.find("true", c) < e; // "true" appears before the next comma
    return true;
}

int as_int(const std::string& s, const std::string& key, int fallback) {
    double v; return find_number(s, key, v) ? (int)(v + 0.5) : fallback;
}

} // namespace

bool load_model_config(const std::string& config_json_path, ModelConfig& cfg) {
    std::ifstream f(config_json_path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[config] cannot open " << config_json_path
                  << " — using built-in defaults\n";
        return false;
    }
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::string tc = extract_object(all, "text_config");
    if (tc.empty()) tc = all; // some configs are flat

    cfg.hidden_size      = as_int(tc, "hidden_size", cfg.hidden_size);
    cfg.n_layers_total   = as_int(tc, "num_hidden_layers", cfg.n_layers_total);
    cfg.attn_interval    = as_int(tc, "full_attention_interval", cfg.attn_interval);
    cfg.attn_q_heads     = as_int(tc, "num_attention_heads", cfg.attn_q_heads);
    cfg.attn_kv_heads    = as_int(tc, "num_key_value_heads", cfg.attn_kv_heads);
    cfg.attn_head_dim    = as_int(tc, "head_dim", cfg.attn_head_dim);
    cfg.gdn_k_heads      = as_int(tc, "linear_num_key_heads", cfg.gdn_k_heads);
    cfg.gdn_v_heads      = as_int(tc, "linear_num_value_heads", cfg.gdn_v_heads);
    cfg.gdn_k_head_dim   = as_int(tc, "linear_key_head_dim", cfg.gdn_k_head_dim);
    cfg.gdn_v_head_dim   = as_int(tc, "linear_value_head_dim", cfg.gdn_v_head_dim);
    cfg.gdn_conv_kernel  = as_int(tc, "linear_conv_kernel_dim", cfg.gdn_conv_kernel);
    cfg.ffn_intermediate = as_int(tc, "intermediate_size", cfg.ffn_intermediate);
    cfg.vocab_size       = as_int(tc, "vocab_size", cfg.vocab_size);
    cfg.native_max_context = as_int(tc, "max_position_embeddings", cfg.native_max_context);

    cfg.gdn_n_heads  = cfg.gdn_v_heads;   // keep aliases consistent
    cfg.gdn_head_dim = cfg.gdn_v_head_dim;

    // Derived layer split (every attn_interval-th layer is full attention).
    cfg.n_layers_attn = cfg.n_layers_total / cfg.attn_interval;
    cfg.n_layers_gdn  = cfg.n_layers_total - cfg.n_layers_attn;

    double d;
    if (find_number(tc, "rms_norm_eps", d)) cfg.rms_norm_eps = (float)d;
    if (find_number(tc, "rope_theta", d)) cfg.rope_theta = d;
    if (find_number(tc, "partial_rotary_factor", d)) cfg.partial_rotary_factor = (float)d;
    bool b;
    if (find_bool(tc, "tie_word_embeddings", b)) cfg.tie_word_embeddings = b;

    // Rough parameter count for the budget display (weights dominate).
    cfg.n_params = (int64_t)cfg.vocab_size * cfg.hidden_size
                 + (int64_t)cfg.n_layers_total * 12 * cfg.hidden_size * cfg.hidden_size;

    std::cout << "[config] " << config_json_path << ": hidden=" << cfg.hidden_size
              << " layers=" << cfg.n_layers_total << " (" << cfg.n_layers_gdn << " gdn / "
              << cfg.n_layers_attn << " attn) ffn=" << cfg.ffn_intermediate
              << " vocab=" << cfg.vocab_size << " tie=" << cfg.tie_word_embeddings << "\n";
    return true;
}
