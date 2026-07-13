#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Byte-level BPE tokenizer (GPT-2 / Qwen family), decode side.
//
// Tokens in vocab.json are stored in a "byte-level" unicode encoding: each raw
// byte 0..255 is mapped to a printable codepoint (bytes_to_unicode). Decoding a
// token means reversing that map codepoint->byte and concatenating the bytes,
// which yields the original UTF-8 text.
//
// Encode implements the Qwen2 byte-level BPE: a hand-written pre-tokenizer
// (the tokenizer.json regex, ASCII-exact with a non-ASCII=letter fallback)
// followed by merge-rank BPE from merges.txt.
// ---------------------------------------------------------------------------
class Tokenizer {
public:
    // Loads token->id from vocab.json. If merges_txt_path is non-empty, also
    // loads BPE merges so encode() works.
    bool load(const std::string& vocab_json_path,
              const std::string& merges_txt_path = "");

    // Text -> token ids (byte-level BPE). Empty if merges weren't loaded.
    std::vector<int> encode(const std::string& text) const;

    // Wraps a user message in this model's chat template (Qwen im_start/im_end
    // with an empty think block) and returns the token ids with the assistant
    // turn primed. Makes the instruct model answer instead of continuing text.
    std::vector<int> encode_chat(const std::string& user_msg) const;

    // Concatenated decode of a sequence of token ids into UTF-8 text.
    std::string decode(const std::vector<int>& ids) const;

    // Loads special/added tokens (content <-> id) from tokenizer_config.json so
    // chat wrapping and stop tokens are read from the model, not hardcoded.
    bool load_specials(const std::string& tokenizer_config_path);
    // Id for a special token by content (e.g. "<|im_end|>"), or -1 if absent.
    int special_id(const std::string& content) const;

    int vocab_size() const { return (int)id2token_.size(); }

private:
    std::vector<std::string> bpe(const std::string& piece_bytes) const;

    std::vector<std::string> id2token_;         // id -> byte-level token (UTF-8)
    std::unordered_map<std::string, int> token2id_;
    std::unordered_map<int, int> byte_decoder_; // codepoint -> raw byte
    std::string byte_encoder_[256];             // raw byte -> byte-level UTF-8 char
    std::unordered_map<std::string, int> merge_ranks_; // "A B" -> rank
    std::unordered_map<std::string, int> special2id_;  // "<|im_end|>" -> id
};
