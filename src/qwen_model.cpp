#include "qwen_model.h"
#include "forward_pass.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
const char* LM = "model.language_model.";
std::string layer_pfx(int L) { return std::string(LM) + "layers." + std::to_string(L) + "."; }
} // namespace

QwenModel::QwenModel(SafeTensors& weights, const ModelConfig& cfg, int max_context)
    : w_(weights), cfg_(cfg), max_context_(max_context) {
    const int H = cfg_.hidden_size, I = cfg_.ffn_intermediate;
    const int HD = cfg_.attn_head_dim, NQ = cfg_.attn_q_heads, NKV = cfg_.attn_kv_heads;
    const int KH = cfg_.gdn_k_heads, VH = cfg_.gdn_v_heads;
    const int DK = cfg_.gdn_k_head_dim, DV = cfg_.gdn_v_head_dim, KER = cfg_.gdn_conv_kernel;
    const int KD = KH * DK, VD = VH * DV, CONV = KD * 2 + VD;

    // Small elementwise weights -> fp32 (tiny). Big matmul weights -> in-place
    // WeightView (bf16 read in the kernel; no fp32 cache).
    auto get = [&](const std::string& name) -> const float* {
        const float* p = w_.get_f32(name);
        if (!p) { ok_ = false; std::cerr << "[qwen] missing weight: " << name << "\n"; }
        return p;
    };
    auto getv = [&](const std::string& name) -> WeightView {
        WeightView v = w_.view(name);
        if (!v.ok()) { ok_ = false; std::cerr << "[qwen] missing weight: " << name << "\n"; }
        return v;
    };

    // --- Persistent per-layer arenas + resolved weight pointers ---
    lw_.resize(cfg_.n_layers_total);
    int ai = 0, gi = 0;
    for (int L = 0; L < cfg_.n_layers_total; ++L) {
        std::string P = layer_pfx(L);
        LayerWeights& lw = lw_[L];
        lw.input_ln = get(P + "input_layernorm.weight");
        lw.post_ln  = get(P + "post_attention_layernorm.weight");
        lw.mlp_gate = getv(P + "mlp.gate_proj.weight");
        lw.mlp_up   = getv(P + "mlp.up_proj.weight");
        lw.mlp_down = getv(P + "mlp.down_proj.weight");
        if (cfg_.is_attention_layer(L)) {
            lw.is_attn = true; lw.mixer_idx = ai++;
            std::string A = P + "self_attn.";
            lw.q_proj = getv(A + "q_proj.weight"); lw.k_proj = getv(A + "k_proj.weight");
            lw.v_proj = getv(A + "v_proj.weight"); lw.o_proj = getv(A + "o_proj.weight");
            lw.q_norm = get(A + "q_norm.weight");  lw.k_norm = get(A + "k_norm.weight");
            kcache_.emplace_back((size_t)max_context_ * NKV * HD, 0.0f);
            vcache_.emplace_back((size_t)max_context_ * NKV * HD, 0.0f);
        } else {
            lw.is_attn = false; lw.mixer_idx = gi++;
            std::string G = P + "linear_attn.";
            lw.in_qkv = getv(G + "in_proj_qkv.weight"); lw.in_z = getv(G + "in_proj_z.weight");
            lw.in_b = getv(G + "in_proj_b.weight");     lw.in_a = getv(G + "in_proj_a.weight");
            lw.conv1d = get(G + "conv1d.weight");       lw.dt_bias = get(G + "dt_bias");
            lw.A_log = get(G + "A_log");                lw.gdn_norm = get(G + "norm.weight");
            lw.out_proj = getv(G + "out_proj.weight");
            gdn_state_.emplace_back((size_t)VH * DK * DV, 0.0f);
            conv_state_.emplace_back((size_t)CONV * (KER - 1), 0.0f);
        }
    }
    emb_name_ = std::string(LM) + "embed_tokens.weight";
    final_norm_ = get(std::string(LM) + "norm.weight");
    // Output projection: dedicated lm_head if present (untied), else tied embed.
    if (w_.has("lm_head.weight"))                        head_ = getv("lm_head.weight");
    else if (w_.has(std::string(LM) + "lm_head.weight")) head_ = getv(std::string(LM) + "lm_head.weight");
    else                                                 head_ = getv(emb_name_);

    // --- Fixed per-step buffers ---
    hidden_.assign(H, 0.0f); normed_.assign(H, 0.0f);
    mix_.assign(H, 0.0f); ff_.assign(H, 0.0f);
    logits_.assign(cfg_.vocab_size, 0.0f);

    // --- Shared scratch arena: sized to the largest single-step working set ---
    size_t attn_ws = (size_t)NQ*HD*2 + NQ*HD + NQ*HD + NKV*HD + NQ*HD + (size_t)max_context_;
    size_t gdn_ws  = (size_t)CONV + VD + VH + VH + CONV + KD + KD + VD + VD + DV + DV;
    size_t ffn_ws  = (size_t)I * 3;
    scratch_.assign(std::max({attn_ws, gdn_ws, ffn_ws}) + 64, 0.0f);

    // --- Memory budget (plan by equation), reconciled with real allocations ---
    budget_ = compute_budget(cfg_, /*bytes_per_weight=*/2, max_context_, /*kv_bytes=*/4);
    budget_.activation_bytes =
        (scratch_.size() + hidden_.size() + normed_.size() + mix_.size()
         + ff_.size() + logits_.size()) * sizeof(float);
    budget_.vision_scratch_bytes = 0;

    std::cout << "[qwen] arenas planned (max_context=" << max_context_ << "):\n"
              << "   weights (bf16, in place): " << budget_.weight_bytes / (1024*1024) << " MB\n"
              << "   gdn_state:       " << budget_.gdn_state_bytes / 1024 << " KB\n"
              << "   gdn_conv_state:  " << budget_.gdn_conv_bytes / 1024 << " KB\n"
              << "   kv_cache:        " << budget_.kv_cache_bytes / 1024 << " KB\n"
              << "   scratch+step:    " << budget_.activation_bytes / 1024 << " KB\n"
              << "   TOTAL (ex-file): " << budget_.total() / (1024*1024) << " MB\n";
}

void QwenModel::reset() {
    pos_ = 0;
    for (auto& v : kcache_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : vcache_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : gdn_state_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : conv_state_) std::fill(v.begin(), v.end(), 0.0f);
}

void QwenModel::attn_step(const LayerWeights& lw, float* out) {
    const int H = cfg_.hidden_size, HD = cfg_.attn_head_dim;
    const int NQ = cfg_.attn_q_heads, NKV = cfg_.attn_kv_heads, GRP = NQ / NKV;
    const int ROT = (int)(HD * cfg_.partial_rotary_factor);
    const double TH = cfg_.rope_theta;
    const float eps = cfg_.rms_norm_eps, sc = 1.0f / std::sqrt((float)HD);
    const int idx = lw.mixer_idx;
    const float* in = normed_.data();

    float* qkv = sb(NQ*HD*2); float* Q = sb(NQ*HD); float* gate = sb(NQ*HD);
    float* kv = sb(NKV*HD);   float* ac = sb(NQ*HD); float* sco = sb(pos_ + 1);

    matmul(lw.q_proj, in, qkv, NQ*HD*2, H);
    for (int h = 0; h < NQ; ++h) {
        const float* qg = qkv + (size_t)h*HD*2;
        float* qd = Q + (size_t)h*HD;
        rmsnorm(qg, lw.q_norm, qd, HD, eps); apply_rope(qd, ROT, pos_, TH);
        for (int d = 0; d < HD; ++d) gate[(size_t)h*HD+d] = qg[HD+d];
    }
    float* kslot = kcache_[idx].data() + (size_t)pos_ * NKV * HD;
    float* vslot = vcache_[idx].data() + (size_t)pos_ * NKV * HD;
    matmul(lw.k_proj, in, kv, NKV*HD, H);
    for (int kh = 0; kh < NKV; ++kh) {
        float* kd = kslot + (size_t)kh*HD;
        rmsnorm(kv + (size_t)kh*HD, lw.k_norm, kd, HD, eps); apply_rope(kd, ROT, pos_, TH);
    }
    matmul(lw.v_proj, in, vslot, NKV*HD, H);

    for (int h = 0; h < NQ; ++h) {
        int kh = h / GRP;
        const float* q = Q + (size_t)h*HD;
        float mx = -1e30f;
        for (int j = 0; j <= pos_; ++j) {
            const float* k = kcache_[idx].data() + ((size_t)j*NKV+kh)*HD;
            float s = 0; for (int d = 0; d < HD; ++d) s += q[d]*k[d];
            s *= sc; sco[j] = s; if (s > mx) mx = s;
        }
        float sm = 0; for (int j = 0; j <= pos_; ++j) { sco[j] = std::exp(sco[j]-mx); sm += sco[j]; }
        float* oh = ac + (size_t)h*HD; for (int d = 0; d < HD; ++d) oh[d] = 0;
        for (int j = 0; j <= pos_; ++j) {
            float p = sco[j]/sm; const float* v = vcache_[idx].data() + ((size_t)j*NKV+kh)*HD;
            for (int d = 0; d < HD; ++d) oh[d] += p*v[d];
        }
    }
    for (int i = 0; i < NQ*HD; ++i) ac[i] *= sigmoidf(gate[i]);
    matmul(lw.o_proj, ac, out, H, NQ*HD);
}

void QwenModel::gdn_step(const LayerWeights& lw, float* out) {
    const int H = cfg_.hidden_size;
    const int KH = cfg_.gdn_k_heads, VH = cfg_.gdn_v_heads;
    const int DK = cfg_.gdn_k_head_dim, DV = cfg_.gdn_v_head_dim, KER = cfg_.gdn_conv_kernel;
    const int KD = KH*DK, VD = VH*DV, CONV = KD*2 + VD;
    const int rep = VH / KH;                       // q/k heads repeated per value head
    const float eps = cfg_.rms_norm_eps, qs = 1.0f / std::sqrt((float)DK);
    const int idx = lw.mixer_idx;
    const float* in = normed_.data();

    float* qkv = sb(CONV); float* z = sb(VD); float* bt = sb(VH); float* at = sb(VH);
    float* conv = sb(CONV); float* Q = sb(KD); float* K = sb(KD); float* V = sb(VD);
    float* gn = sb(VD); float* km = sb(DV); float* dl = sb(DV);

    matmul(lw.in_qkv, in, qkv, CONV, H);
    matmul(lw.in_z, in, z, VD, H);
    matmul(lw.in_b, in, bt, VH, H);
    matmul(lw.in_a, in, at, VH, H);

    float* cst = conv_state_[idx].data(); // [CONV * (KER-1)]
    for (int c = 0; c < CONV; ++c) {
        const float* w4 = lw.conv1d + (size_t)c*KER;
        float* st = cst + (size_t)c*(KER-1);
        float acc = 0;
        for (int k = 0; k < KER-1; ++k) acc += w4[k] * st[k];
        acc += w4[KER-1] * qkv[c];
        conv[c] = silu(acc);
        for (int k = 0; k < KER-2; ++k) st[k] = st[k+1];
        st[KER-2] = qkv[c];
    }
    // q,k: KH heads of DK (l2-normed, q scaled).  v: VH heads of DV.
    for (int h = 0; h < KH; ++h) {
        float* q = Q+(size_t)h*DK; float* k = K+(size_t)h*DK;
        for (int d = 0; d < DK; ++d) { q[d]=conv[h*DK+d]; k[d]=conv[KD+h*DK+d]; }
        l2norm(q, DK, 1e-6f); l2norm(k, DK, 1e-6f);
        for (int d = 0; d < DK; ++d) q[d] *= qs;
    }
    for (int i = 0; i < VD; ++i) V[i] = conv[KD*2 + i];

    for (int vh = 0; vh < VH; ++vh) {
        int kh = vh / rep;
        float g = -std::exp(lw.A_log[vh]) * softplusf(at[vh] + lw.dt_bias[vh]);
        float gt = std::exp(g), b = sigmoidf(bt[vh]);
        const float* q = Q+(size_t)kh*DK; const float* k = K+(size_t)kh*DK;
        const float* v = V+(size_t)vh*DV;
        float* S = gdn_state_[idx].data() + (size_t)vh*DK*DV; // [DK x DV]
        for (size_t i = 0; i < (size_t)DK*DV; ++i) S[i] *= gt;
        for (int vv = 0; vv < DV; ++vv) { float m = 0; for (int kk = 0; kk < DK; ++kk) m += S[(size_t)kk*DV+vv]*k[kk]; km[vv] = m; }
        for (int vv = 0; vv < DV; ++vv) dl[vv] = (v[vv]-km[vv])*b;
        for (int kk = 0; kk < DK; ++kk) { float kv = k[kk]; float* Sr = &S[(size_t)kk*DV]; for (int vv = 0; vv < DV; ++vv) Sr[vv] += kv*dl[vv]; }
        float* o = gn+(size_t)vh*DV;
        for (int vv = 0; vv < DV; ++vv) { float a = 0; for (int kk = 0; kk < DK; ++kk) a += S[(size_t)kk*DV+vv]*q[kk]; o[vv] = a; }
        rmsnorm_gated(o, lw.gdn_norm, z+(size_t)vh*DV, o, DV, eps);
    }
    matmul(lw.out_proj, gn, out, H, VD);
}

const std::vector<float>& QwenModel::step(int token_id) {
    const int H = cfg_.hidden_size, I = cfg_.ffn_intermediate;
    const float eps = cfg_.rms_norm_eps;
    const int64_t vocab = cfg_.vocab_size;

    w_.get_row_f32(emb_name_, token_id, hidden_.data());

    for (const LayerWeights& lw : lw_) {
        rmsnorm(hidden_.data(), lw.input_ln, normed_.data(), H, eps);
        sbreset();
        if (lw.is_attn) attn_step(lw, mix_.data());
        else            gdn_step(lw, mix_.data());
        for (int i = 0; i < H; ++i) hidden_[i] += mix_[i];

        rmsnorm(hidden_.data(), lw.post_ln, normed_.data(), H, eps);
        sbreset();
        float* g = sb(I); float* u = sb(I); float* hh = sb(I);
        matmul(lw.mlp_gate, normed_.data(), g, I, H);
        matmul(lw.mlp_up, normed_.data(), u, I, H);
        for (int i = 0; i < I; ++i) hh[i] = silu(g[i]) * u[i];
        matmul(lw.mlp_down, hh, ff_.data(), H, I);
        for (int i = 0; i < H; ++i) hidden_[i] += ff_[i];
    }

    rmsnorm(hidden_.data(), final_norm_, normed_.data(), H, eps);
    matmul(head_, normed_.data(), logits_.data(), (int)vocab, H);
    ++pos_;
    return logits_;
}

std::vector<int> QwenModel::generate_greedy(const std::vector<int>& prompt, int n_new) {
    reset();
    const int64_t vocab = cfg_.vocab_size;
    const std::vector<float>* lg = nullptr;
    for (int id : prompt) lg = &step(id);

    std::vector<int> outp;
    for (int i = 0; i < n_new && lg; ++i) {
        int am = 0;
        for (int64_t v = 1; v < vocab; ++v) if ((*lg)[v] > (*lg)[am]) am = (int)v;
        outp.push_back(am);
        lg = &step(am);
    }
    return outp;
}
