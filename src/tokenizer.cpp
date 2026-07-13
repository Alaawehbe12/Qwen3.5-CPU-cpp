#include "tokenizer.h"
#include <fstream>
#include <iostream>

namespace {

// Append the UTF-8 encoding of codepoint cp to out.
void utf8_append(std::string& out, int cp) {
    if (cp < 0x80) out.push_back((char)cp);
    else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

// GPT-2 bytes_to_unicode: fill codepoint->byte and byte->byte-level-UTF8 maps.
void build_byte_maps(std::unordered_map<int, int>& dec, std::string enc[256]) {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool present = false;
        for (int x : bs) if (x == b) { present = true; break; }
        if (!present) { bs.push_back(b); cs.push_back(256 + n); ++n; }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        dec[cs[i]] = bs[i];
        std::string s; utf8_append(s, cs[i]); enc[bs[i]] = s;
    }
}

// Codepoint classification for the pre-tokenizer (ASCII-exact; non-ASCII
// treated as letters, which covers \p{L}\p{M} for typical text).
bool is_ws(int c)     { return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c==0x0B||c==0xA0||c==0x85; }
bool is_num(int c)    { return c>='0' && c<='9'; }
bool is_letter(int c) { return (c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>=0x80 && !is_ws(c)); }

// Decode a UTF-8 string into codepoints.
std::vector<int> utf8_decode(const std::string& s) {
    std::vector<int> cps;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        int cp, len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
        else { cp = c; len = 1; }
        for (int k = 1; k < len && i + k < s.size(); ++k)
            cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        i += len;
        cps.push_back(cp);
    }
    return cps;
}

// Minimal JSON string parser: reads a "..."-delimited string starting at p
// (which must point at the opening quote), handling \" \\ \/ \b\f\n\r\t \uXXXX.
// Advances p past the closing quote. Returns the decoded UTF-8 string.
std::string parse_json_string(const char*& p, const char* end) {
    std::string s;
    ++p; // opening quote
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            ++p;
            switch (*p) {
                case 'n': s.push_back('\n'); break;
                case 't': s.push_back('\t'); break;
                case 'r': s.push_back('\r'); break;
                case 'b': s.push_back('\b'); break;
                case 'f': s.push_back('\f'); break;
                case '"': s.push_back('"'); break;
                case '\\': s.push_back('\\'); break;
                case '/': s.push_back('/'); break;
                case 'u': {
                    int cp = 0;
                    for (int k = 0; k < 4 && p + 1 < end; ++k) {
                        ++p; char h = *p; int d = 0;
                        if (h >= '0' && h <= '9') d = h - '0';
                        else if (h >= 'a' && h <= 'f') d = h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') d = h - 'A' + 10;
                        cp = (cp << 4) | d;
                    }
                    utf8_append(s, cp);
                    break;
                }
                default: s.push_back(*p); break;
            }
            ++p;
        } else {
            s.push_back(*p++);
        }
    }
    if (p < end) ++p; // closing quote
    return s;
}

} // namespace

bool Tokenizer::load(const std::string& vocab_json_path,
                     const std::string& merges_txt_path) {
    std::ifstream f(vocab_json_path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "[tokenizer] cannot open " << vocab_json_path << "\n"; return false; }
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    build_byte_maps(byte_decoder_, byte_encoder_);

    const char* p = buf.data();
    const char* end = p + buf.size();
    while (p < end && *p != '{') ++p;
    if (p < end) ++p;
    // Parse "token": id pairs.
    while (p < end) {
        while (p < end && *p != '"' && *p != '}') ++p;
        if (p >= end || *p == '}') break;
        std::string token = parse_json_string(p, end);
        while (p < end && *p != ':') ++p;
        if (p < end) ++p;
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
        long id = 0; bool neg = false;
        if (p < end && *p == '-') { neg = true; ++p; }
        while (p < end && *p >= '0' && *p <= '9') { id = id * 10 + (*p - '0'); ++p; }
        if (neg) id = -id;
        if ((long)id2token_.size() <= id) id2token_.resize(id + 1);
        token2id_[token] = (int)id;
        id2token_[id] = std::move(token);
        while (p < end && *p != ',' && *p != '}') ++p;
        if (p < end && *p == ',') ++p;
    }

    if (!merges_txt_path.empty()) {
        std::ifstream mf(merges_txt_path, std::ios::binary);
        if (!mf.is_open()) { std::cerr << "[tokenizer] cannot open " << merges_txt_path << "\n"; return false; }
        std::string line; int rank = 0;
        while (std::getline(mf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            merge_ranks_[line] = rank++; // key is exactly "A B"
        }
        std::cout << "[tokenizer] loaded " << merge_ranks_.size() << " merges\n";
    }

    std::cout << "[tokenizer] loaded " << id2token_.size() << " tokens\n";
    return !id2token_.empty();
}

// Apply BPE merges to the byte-level chars of one pre-token piece.
std::vector<std::string> Tokenizer::bpe(const std::string& piece_bytes) const {
    std::vector<std::string> word;
    for (unsigned char b : piece_bytes) word.push_back(byte_encoder_[b]);
    if (word.size() < 2) return word;

    while (true) {
        int best_rank = 0x7fffffff, best_i = -1;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = merge_ranks_.find(word[i] + " " + word[i + 1]);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second; best_i = (int)i;
            }
        }
        if (best_i < 0) break;
        word[best_i] = word[best_i] + word[best_i + 1];
        word.erase(word.begin() + best_i + 1);
    }
    return word;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    if (merge_ranks_.empty()) return ids;

    // UTF-8 -> codepoints with byte offsets, so we can slice raw bytes later.
    std::vector<int> cp; std::vector<size_t> off;
    { size_t i = 0; while (i < text.size()) {
        off.push_back(i);
        unsigned char c = (unsigned char)text[i]; int v, len;
        if (c < 0x80) { v = c; len = 1; }
        else if ((c >> 5) == 0x6) { v = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { v = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { v = c & 0x07; len = 4; }
        else { v = c; len = 1; }
        for (int k = 1; k < len && i + k < text.size(); ++k) v = (v << 6) | ((unsigned char)text[i + k] & 0x3F);
        i += len; cp.push_back(v);
    } off.push_back(text.size()); }

    const int n = (int)cp.size();
    int i = 0;
    auto emit = [&](int a, int b) {
        std::string piece = text.substr(off[a], off[b] - off[a]);
        for (const std::string& sym : bpe(piece)) {
            auto it = token2id_.find(sym);
            if (it != token2id_.end()) ids.push_back(it->second);
        }
    };
    auto contraction = [&](int p0) -> int { // returns end index or -1
        if (cp[p0] != '\'') return -1;
        auto lc = [&](int idx){ int c = idx < n ? cp[idx] : -1; return (c>='A'&&c<='Z')? c+32 : c; };
        int a = lc(p0+1), b = lc(p0+2);
        if (a=='s'||a=='t'||a=='m'||a=='d') return p0+2;
        if ((a=='r'&&b=='e')||(a=='v'&&b=='e')||(a=='l'&&b=='l')) return p0+3;
        return -1;
    };

    while (i < n) {
        int e;
        // 1. contractions
        if ((e = contraction(i)) > 0) { emit(i, e); i = e; continue; }
        // 2. [^\r\n\p{L}\p{N}]? [\p{L}\p{M}]+
        {
            int k = i;
            if (k < n && cp[k] != '\r' && cp[k] != '\n' && !is_letter(cp[k]) && !is_num(cp[k])) k++;
            int ls = k; while (k < n && is_letter(cp[k])) k++;
            if (k > ls) { emit(i, k); i = k; continue; }
        }
        // 3. \p{N} (single digit)
        if (is_num(cp[i])) { emit(i, i + 1); i = i + 1; continue; }
        // 4. ' '? [^\s\p{L}\p{M}\p{N}]+ [\r\n]*
        {
            int k = i;
            if (cp[k] == ' ' && k + 1 < n && !is_ws(cp[k+1]) && !is_letter(cp[k+1]) && !is_num(cp[k+1])) k++;
            int ps = k; while (k < n && !is_ws(cp[k]) && !is_letter(cp[k]) && !is_num(cp[k])) k++;
            if (k > ps) { while (k < n && (cp[k]=='\r'||cp[k]=='\n')) k++; emit(i, k); i = k; continue; }
        }
        // 5. \s* [\r\n]+
        {
            int k = i; while (k < n && is_ws(cp[k]) && cp[k] != '\r' && cp[k] != '\n') k++;
            if (k < n && (cp[k]=='\r'||cp[k]=='\n')) { while (k < n && (cp[k]=='\r'||cp[k]=='\n')) k++; emit(i, k); i = k; continue; }
        }
        // 6. \s+(?!\S) : whitespace run, leaving one space if followed by non-space
        {
            int k = i; while (k < n && is_ws(cp[k])) k++;
            if (k > i) {
                int endw = (k < n) ? k - 1 : k; // leave last space for the following word
                if (endw > i) { emit(i, endw); i = endw; continue; }
            }
        }
        // 7. \s+
        {
            int k = i; while (k < n && is_ws(cp[k])) k++;
            if (k > i) { emit(i, k); i = k; continue; }
        }
        // fallback: emit single codepoint
        emit(i, i + 1); i = i + 1;
    }
    return ids;
}

bool Tokenizer::load_specials(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t k = s.find("\"added_tokens_decoder\"");
    if (k == std::string::npos) return false;
    size_t start = s.find('{', k);
    if (start == std::string::npos) return false;
    int depth = 0; size_t end = start;
    for (size_t i = start; i < s.size(); ++i) {
        if (s[i] == '{') ++depth;
        else if (s[i] == '}') { if (--depth == 0) { end = i; break; } }
    }
    size_t i = start + 1;
    while (i < end) {
        size_t q = s.find('"', i); if (q == std::string::npos || q >= end) break;
        size_t q2 = s.find('"', q + 1); if (q2 == std::string::npos) break;
        std::string idstr = s.substr(q + 1, q2 - q - 1);
        size_t ob = s.find('{', q2); if (ob == std::string::npos || ob >= end) break;
        int d = 0; size_t oe = ob;
        for (size_t j = ob; j <= end; ++j) { if (s[j] == '{') ++d; else if (s[j] == '}') { if (--d == 0) { oe = j; break; } } }
        size_t ck = s.find("\"content\"", ob);
        if (ck != std::string::npos && ck < oe) {
            size_t cs = s.find('"', s.find(':', ck) + 1);
            size_t ce = s.find('"', cs + 1);
            std::string content = s.substr(cs + 1, ce - cs - 1);
            try { special2id_[content] = std::stoi(idstr); } catch (...) {}
        }
        i = oe + 1;
    }
    return !special2id_.empty();
}

int Tokenizer::special_id(const std::string& content) const {
    auto it = special2id_.find(content);
    return it == special2id_.end() ? -1 : it->second;
}

std::vector<int> Tokenizer::encode_chat(const std::string& user_msg) const {
    // Special-token ids: looked up from the loaded tokenizer_config, with the
    // Qwen3.5 defaults as a fallback if load_specials() wasn't called.
    auto sid = [&](const std::string& c, int fb) { int v = special_id(c); return v < 0 ? fb : v; };
    const int IM_START = sid("<|im_start|>", 248045), IM_END = sid("<|im_end|>", 248046);
    const int THINK = sid("<think>", 248068), THINK_END = sid("</think>", 248069);
    std::vector<int> out;
    auto add = [&](const std::vector<int>& v) { out.insert(out.end(), v.begin(), v.end()); };
    out.push_back(IM_START); add(encode("user\n" + user_msg)); out.push_back(IM_END);
    add(encode("\n"));
    out.push_back(IM_START); add(encode("assistant\n"));
    out.push_back(THINK); add(encode("\n\n")); out.push_back(THINK_END); add(encode("\n\n"));
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)id2token_.size()) continue;
        for (int cp : utf8_decode(id2token_[id])) {
            auto it = byte_decoder_.find(cp);
            if (it != byte_decoder_.end()) out.push_back((char)(unsigned char)it->second);
        }
    }
    return out;
}
