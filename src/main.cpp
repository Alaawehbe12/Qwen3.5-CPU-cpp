#include <iostream>
#include <string>
#include <vector>
#include "model_config.h"
#include "safetensors_loader.h"
#include "qwen_model.h"
#include "tokenizer.h"

// End-to-end text generation on the real Qwen3.5-0.8B weights:
//   text -> BPE encode -> incremental decode (greedy) -> BPE decode -> text
// The forward pass and generation are validated against transformers to
// rel_l2 ~1e-6 / exact token match.
//
// Usage: mllm_engine [model_dir] ["prompt"] [n_new_tokens]
int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "models/Qwen3.5-0.8B";
    std::string prompt = (argc > 2) ? argv[2] : "The capital of France is";
    int n_new = (argc > 3) ? std::stoi(argv[3]) : 24;

    // Everything is read from the model folder — no hardcoded dimensions or
    // filenames, so any Qwen3.5 size works by pointing at its directory.
    ModelConfig cfg;
    load_model_config(model_dir + "/config.json", cfg);

    SafeTensors weights;
    if (!weights.load_model(model_dir)) {
        std::cerr << "Could not load weights from '" << model_dir << "'.\n";
        return 1;
    }
    Tokenizer tok;
    if (!tok.load(model_dir + "/vocab.json", model_dir + "/merges.txt")) {
        std::cerr << "Could not load tokenizer from '" << model_dir << "'.\n";
        return 1;
    }
    tok.load_specials(model_dir + "/tokenizer_config.json");

    std::vector<int> prompt_ids = tok.encode(prompt);
    std::cout << "\nPrompt: \"" << prompt << "\"  (" << prompt_ids.size() << " tokens)\n";

    QwenModel model(weights, cfg, (int)prompt_ids.size() + n_new + 2);
    std::vector<int> gen = model.generate_greedy(prompt_ids, n_new);

    std::vector<int> full = prompt_ids;
    full.insert(full.end(), gen.begin(), gen.end());
    std::cout << "\nGenerated text:\n" << tok.decode(full) << "\n";
    return 0;
}
