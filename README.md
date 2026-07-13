# MLLM C++ inference framework

A CPU-only, single-sequence inference engine for the hybrid GDN + Gated-Attention
Qwen3.5 family. Built around the principle that memory should be planned by equation,
not managed by a general-purpose allocator at runtime.

**It generates text end-to-end today** — pure C++, on the real Qwen3.5-0.8B weights:

```
$ mllm_engine models/Qwen3.5-0.8B "The capital of France is" 24
The capital of France is Paris.
The capital of France is Paris.
...
```

The fp32 forward pass is numerically validated against a transformers reference
(every one of the 24 layers + logits match to rel_l2 ~1e-6), and the incremental
decode reproduces transformers' greedy generation token-for-token. The byte-level
BPE tokenizer (encode + decode) matches the HuggingFace tokenizer exactly.

Everything is driven from the model folder — `config.json` sets the dimensions,
all `.safetensors` shards are loaded, embeddings can be tied or untied, and the
tokenizer's special tokens are read from `tokenizer_config.json`. So other sizes
in the Qwen3.5 family work by pointing the engine at their directory; no code
changes.

## What's implemented

- `model_config.h` — architecture constants (defaults to Qwen3.5-2B: 24 layers,
  18 GDN + 6 Attention in a 3:1 repeating pattern).
- `memory_budget.{h,cpp}` — the memory equation: `compute_budget()` (model + context ->
  bytes per arena) and `solve_max_context()` (RAM ceiling -> max context, solved in reverse).
- `inference_arenas.{h,cpp}` — four fixed-size arenas (weights, GDN state, KV cache,
  scratch), allocated once, reset (not freed/reallocated) between requests.
- `gguf_loader.{h,cpp}` — **real GGUF parser** (v2/v3). Parses the header, typed
  metadata KV pairs, and the tensor descriptor table; computes each tensor's on-disk
  byte size from its ggml dtype (F32/F16/Q4_0…Q6_K/Q8_0, etc.), resolves absolute
  data offsets, and validates every tensor lies inside the file. Reads bounds-checked
  so a truncated/malformed file fails cleanly instead of crashing. `parse_gguf()` returns
  a `GgufModel` (metadata, tensors, `n_params`, `total_tensor_bytes`); `load_gguf()` is a
  thin name→offset wrapper for the arena. Still falls back to a zero-filled dummy buffer
  if no file is found, so the pipeline runs without a real model.
- `config_loader.{h,cpp}` — reads `config.json` into `ModelConfig`, so all
  dimensions come from the model folder, not hardcoded constants.
- `safetensors_loader.{h,cpp}` — parses the HuggingFace `.safetensors` container
  (JSON header + tensor table), converts bf16→fp32 on demand, and `load_model()`
  merges **all shards** in a directory (single- or multi-file models).
- `forward_pass.{h,cpp}` — the validated fp32 kernels: `matmul`, `rmsnorm`
  (the `(1+weight)` form), `rmsnorm_gated`, `apply_rope` (partial RoPE), `l2norm`,
  `silu`/`sigmoid`/`softplus`.
- `qwen_forward.{h,cpp}` — batch/prefill forward pass: embedding → 24 hybrid
  layers (Gated-DeltaNet + Gated-Attention mixers, SwiGLU FFN) → final norm →
  tied-embedding logits. `QwenForward::prefill()` (used for validation).
- `qwen_model.{h,cpp}` — the **stateful incremental decoder** (`QwenModel`),
  arena/memory-first: all memory is planned by the `memory_budget` equation and
  allocated once (KV cache per attention layer; recurrent + conv-1d state per GDN
  layer; one shared scratch arena carved by a bump pointer). Weight pointers are
  resolved up front, so `step()` does **zero heap allocation** — proven by an
  `operator new` counter (0 allocations across steady-state steps). `budget()`
  reports the planned arenas.
- `tokenizer.{h,cpp}` — byte-level BPE `encode()`/`decode()` (hand-written
  pre-tokenizer + merge-rank BPE), validated exact vs the HuggingFace tokenizer.
- `main.cpp` — CLI: text in → encode → generate → decode → text out.
- `server.cpp` — a tiny Winsock HTTP server (`mllm_server`) that loads the
  model once and serves a browser test UI at `http://localhost:8080`, streaming
  generated tokens live. Build target is Windows-only.

## What's NOT implemented yet (next steps)

1. **Sampling** — only greedy (argmax) decoding today; temperature/top-p/top-k
   would make it a general generator.
2. **Vision path** — ViT patch encoder + projection feeding into `hidden_state`
   before layer 0 when an image is present in the prompt.
3. **Quantized matmul kernels** — int8/int4 weight formats need dequant-on-the-fly
   or quantized GEMM (the model also ships bf16; fp32 upcast works today).
4. **Performance** — the matmul is multi-threaded (OpenMP over output rows) and
   AVX2-vectorized (FMA, 4 accumulators), and big weights are read **bf16 in
   place** and widened to fp32 in-register (no fp32 weight cache). Net decode:
   ~1 → ~32 tok/s (0.8B) / ~10 tok/s (2B) on 8 cores, at ~half the memory.
   Decode is memory-bandwidth bound, so the remaining lever is **int8/int4
   quantized weights** (~2-4x fewer bytes again).

## Build

Requires a C++17 compiler. CMake recommended (not available in the sandbox this was
built in, but the CMakeLists.txt is included and untested — verify on your machine):

```
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

Or compile directly:
```
g++ -std=c++17 -O2 -Iinclude src/*.cpp -o mllm_engine
```

Run: `./mllm_engine [path-to-model.gguf]` (works without a real file — falls back to
dummy weights so you can test the memory-planning logic today).
