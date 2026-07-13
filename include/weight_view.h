#pragma once
#include <cstdint>

// A non-owning view of a weight tensor as it sits in the loaded file buffer —
// either bf16 (the shipped format) or fp32 — so the matmul can read it in place
// and widen bf16 to fp32 in-register, avoiding a separate fp32 weight cache.
struct WeightView {
    const uint16_t* bf16 = nullptr; // set if the tensor is stored BF16
    const float*    f32  = nullptr; // set if the tensor is stored F32
    int64_t numel = 0;
    bool ok() const { return bf16 != nullptr || f32 != nullptr; }
};
