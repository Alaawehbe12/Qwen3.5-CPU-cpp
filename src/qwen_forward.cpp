#include "qwen_forward.h"
#include "forward_pass.h"
#include <cmath>
#include <iostream>

namespace {
const char* LM = "model.language_model.";
std::string layer_pfx(int L) {
    return std::string(LM) + "layers." + std::to_string(L) + ".";
}
} // namespace

const float* QwenForward::W(const std::string& name, bool* ok) {
    const float* p = w_.get_f32(name);
    if (!p) { missing_ = true; std::cerr << "[qwen] missing weight: " << name << "\n"; if (ok) *ok = false; }
    return p;
}

// --- Gated Attention: partial RoPE + QK-norm + GQA + sigmoid output gate ---
void QwenForward::mix_attention(int L, const std::vector<float>& in,
                                std::vector<float>& out, int seq) {
    const int H = cfg_.hidden_size, HD = cfg_.attn_head_dim;
    const int NQ = cfg_.attn_q_heads, NKV = cfg_.attn_kv_heads;
    const int GRP = NQ / NKV;
    const int ROT = (int)(HD * cfg_.partial_rotary_factor);
    const double TH = cfg_.rope_theta;
    const float eps = cfg_.rms_norm_eps, sc = 1.0f / std::sqrt((float)HD);
    std::string P = layer_pfx(L) + "self_attn.";
    bool ok = true;
    const float* qw = W(P + "q_proj.weight", &ok); const float* kw = W(P + "k_proj.weight", &ok);
    const float* vw = W(P + "v_proj.weight", &ok); const float* ow = W(P + "o_proj.weight", &ok);
    const float* qn = W(P + "q_norm.weight", &ok); const float* kn = W(P + "k_norm.weight", &ok);
    if (!ok) return;

    std::vector<float> Q((size_t)seq*NQ*HD), Kc((size_t)seq*NKV*HD),
                       Vc((size_t)seq*NKV*HD), gt((size_t)seq*NQ*HD);
    std::vector<float> qkv((size_t)NQ*HD*2), kv((size_t)NKV*HD);
    for (int t = 0; t < seq; ++t) {
        const float* x = in.data() + (size_t)t*H;
        matmul(qw, x, qkv.data(), NQ*HD*2, H);
        for (int h = 0; h < NQ; ++h) {
            const float* qg = qkv.data() + (size_t)h*HD*2;
            float* qd = Q.data() + ((size_t)t*NQ+h)*HD;
            rmsnorm(qg, qn, qd, HD, eps); apply_rope(qd, ROT, t, TH);
            float* gd = gt.data() + ((size_t)t*NQ+h)*HD;
            for (int d = 0; d < HD; ++d) gd[d] = qg[HD+d];
        }
        matmul(kw, x, kv.data(), NKV*HD, H);
        for (int kh = 0; kh < NKV; ++kh) {
            float* kd = Kc.data() + ((size_t)t*NKV+kh)*HD;
            rmsnorm(kv.data()+(size_t)kh*HD, kn, kd, HD, eps); apply_rope(kd, ROT, t, TH);
        }
        matmul(vw, x, kv.data(), NKV*HD, H);
        for (int kh = 0; kh < NKV; ++kh)
            for (int d = 0; d < HD; ++d)
                Vc[((size_t)t*NKV+kh)*HD+d] = kv[(size_t)kh*HD+d];
    }
    std::vector<float> ac((size_t)NQ*HD), sco((size_t)seq);
    for (int t = 0; t < seq; ++t) {
        for (int h = 0; h < NQ; ++h) {
            int kh = h / GRP;
            const float* q = Q.data() + ((size_t)t*NQ+h)*HD;
            float mx = -1e30f;
            for (int j = 0; j <= t; ++j) {
                const float* k = Kc.data() + ((size_t)j*NKV+kh)*HD;
                float s = 0; for (int d = 0; d < HD; ++d) s += q[d]*k[d];
                s *= sc; sco[j] = s; if (s > mx) mx = s;
            }
            float sm = 0; for (int j = 0; j <= t; ++j) { sco[j] = std::exp(sco[j]-mx); sm += sco[j]; }
            float* oh = ac.data() + (size_t)h*HD; for (int d = 0; d < HD; ++d) oh[d] = 0;
            for (int j = 0; j <= t; ++j) {
                float p = sco[j]/sm; const float* v = Vc.data() + ((size_t)j*NKV+kh)*HD;
                for (int d = 0; d < HD; ++d) oh[d] += p*v[d];
            }
        }
        const float* g = gt.data() + (size_t)t*NQ*HD;
        for (int i = 0; i < NQ*HD; ++i) ac[i] *= sigmoidf(g[i]);
        matmul(ow, ac.data(), out.data() + (size_t)t*H, H, NQ*HD);
    }
}

// --- Gated DeltaNet: causal conv1d + dt/A_log gating + recurrent delta rule ---
void QwenForward::mix_gdn(int L, const std::vector<float>& in,
                          std::vector<float>& out, int seq) {
    const int H = cfg_.hidden_size, NH = cfg_.gdn_n_heads, D = cfg_.gdn_head_dim;
    const int KD = NH*D, VD = NH*D, CONV = KD*2 + VD, KER = cfg_.gdn_conv_kernel;
    const float eps = cfg_.rms_norm_eps, qs = 1.0f / std::sqrt((float)D);
    std::string P = layer_pfx(L) + "linear_attn.";
    bool ok = true;
    const float* wq = W(P+"in_proj_qkv.weight",&ok); const float* wz = W(P+"in_proj_z.weight",&ok);
    const float* wb = W(P+"in_proj_b.weight",&ok);   const float* wa = W(P+"in_proj_a.weight",&ok);
    const float* wc = W(P+"conv1d.weight",&ok);      const float* dt = W(P+"dt_bias",&ok);
    const float* Al = W(P+"A_log",&ok);              const float* wn = W(P+"norm.weight",&ok);
    const float* wo = W(P+"out_proj.weight",&ok);
    if (!ok) return;

    std::vector<float> qkv((size_t)seq*CONV), zb((size_t)seq*VD),
                       beta((size_t)seq*NH), g((size_t)seq*NH), bt((size_t)NH), at((size_t)NH);
    for (int t = 0; t < seq; ++t) {
        const float* x = in.data() + (size_t)t*H;
        matmul(wq, x, qkv.data()+(size_t)t*CONV, CONV, H);
        matmul(wz, x, zb.data()+(size_t)t*VD, VD, H);
        matmul(wb, x, bt.data(), NH, H); matmul(wa, x, at.data(), NH, H);
        for (int h = 0; h < NH; ++h) {
            beta[(size_t)t*NH+h] = sigmoidf(bt[h]);
            g[(size_t)t*NH+h] = -std::exp(Al[h]) * softplusf(at[h] + dt[h]);
        }
    }
    // depthwise causal conv1d (left pad KER-1) + silu
    std::vector<float> conv((size_t)seq*CONV);
    for (int c = 0; c < CONV; ++c) {
        const float* w4 = wc + (size_t)c*KER;
        for (int t = 0; t < seq; ++t) {
            float a = 0;
            for (int k = 0; k < KER; ++k) { int s = t-(KER-1)+k; if (s >= 0) a += w4[k]*qkv[(size_t)s*CONV+c]; }
            conv[(size_t)t*CONV+c] = silu(a);
        }
    }
    std::vector<float> Q((size_t)seq*KD), K((size_t)seq*KD), V((size_t)seq*VD);
    for (int t = 0; t < seq; ++t) {
        float* c = conv.data() + (size_t)t*CONV;
        for (int h = 0; h < NH; ++h) {
            float* q = Q.data()+((size_t)t*NH+h)*D; float* k = K.data()+((size_t)t*NH+h)*D; float* v = V.data()+((size_t)t*NH+h)*D;
            for (int d = 0; d < D; ++d) { q[d]=c[h*D+d]; k[d]=c[KD+h*D+d]; v[d]=c[KD*2+h*D+d]; }
            l2norm(q, D, 1e-6f); l2norm(k, D, 1e-6f);
            for (int d = 0; d < D; ++d) q[d] *= qs;
        }
    }
    std::vector<float> core((size_t)seq*VD), S((size_t)D*D), km((size_t)D), dl((size_t)D);
    for (int h = 0; h < NH; ++h) {
        std::fill(S.begin(), S.end(), 0.0f);
        for (int t = 0; t < seq; ++t) {
            float gtt = std::exp(g[(size_t)t*NH+h]), b = beta[(size_t)t*NH+h];
            const float* q = Q.data()+((size_t)t*NH+h)*D; const float* k = K.data()+((size_t)t*NH+h)*D; const float* v = V.data()+((size_t)t*NH+h)*D;
            for (size_t i = 0; i < (size_t)D*D; ++i) S[i] *= gtt;
            for (int vv = 0; vv < D; ++vv) { float m = 0; for (int kk = 0; kk < D; ++kk) m += S[(size_t)kk*D+vv]*k[kk]; km[vv] = m; }
            for (int vv = 0; vv < D; ++vv) dl[vv] = (v[vv]-km[vv])*b;
            for (int kk = 0; kk < D; ++kk) { float kv = k[kk]; float* Sr = &S[(size_t)kk*D]; for (int vv = 0; vv < D; ++vv) Sr[vv] += kv*dl[vv]; }
            float* o = core.data()+((size_t)t*NH+h)*D;
            for (int vv = 0; vv < D; ++vv) { float a = 0; for (int kk = 0; kk < D; ++kk) a += S[(size_t)kk*D+vv]*q[kk]; o[vv] = a; }
        }
    }
    std::vector<float> gn((size_t)seq*VD);
    for (int t = 0; t < seq; ++t) {
        for (int h = 0; h < NH; ++h) {
            const float* c = core.data()+((size_t)t*NH+h)*D;
            const float* z = zb.data()+((size_t)t*NH+h)*D;
            rmsnorm_gated(c, wn, z, gn.data()+((size_t)t*NH+h)*D, D, eps);
        }
        matmul(wo, gn.data()+(size_t)t*VD, out.data()+(size_t)t*H, H, VD);
    }
}

bool QwenForward::prefill(const std::vector<int>& ids, std::vector<float>& logits_out) {
    missing_ = false;
    const int H = cfg_.hidden_size, I = cfg_.ffn_intermediate;
    const int seq = (int)ids.size();
    const float eps = cfg_.rms_norm_eps;

    const std::string EMB = std::string(LM) + "embed_tokens.weight";
    std::vector<float> hidden((size_t)seq*H);
    for (int t = 0; t < seq; ++t)
        if (!w_.get_row_f32(EMB, ids[t], hidden.data()+(size_t)t*H)) {
            std::cerr << "[qwen] embedding lookup failed (id " << ids[t] << ")\n"; return false;
        }

    std::vector<float> normed((size_t)seq*H), mix((size_t)seq*H), ff((size_t)seq*H),
                       gbuf((size_t)I), ubuf((size_t)I), hbuf((size_t)I);
    for (int L = 0; L < cfg_.n_layers_total; ++L) {
        std::string LP = layer_pfx(L);
        bool ok = true;
        const float* iln = W(LP+"input_layernorm.weight",&ok);
        const float* pln = W(LP+"post_attention_layernorm.weight",&ok);
        const float* gw = W(LP+"mlp.gate_proj.weight",&ok);
        const float* uw = W(LP+"mlp.up_proj.weight",&ok);
        const float* dw = W(LP+"mlp.down_proj.weight",&ok);
        if (!ok) return false;

        for (int t = 0; t < seq; ++t) rmsnorm(hidden.data()+(size_t)t*H, iln, normed.data()+(size_t)t*H, H, eps);
        if (cfg_.is_attention_layer(L)) mix_attention(L, normed, mix, seq);
        else                           mix_gdn(L, normed, mix, seq);
        if (missing_) return false;
        for (size_t i = 0; i < (size_t)seq*H; ++i) hidden[i] += mix[i];

        for (int t = 0; t < seq; ++t) {
            rmsnorm(hidden.data()+(size_t)t*H, pln, normed.data()+(size_t)t*H, H, eps);
            const float* x = normed.data()+(size_t)t*H;
            matmul(gw, x, gbuf.data(), I, H); matmul(uw, x, ubuf.data(), I, H);
            for (int i = 0; i < I; ++i) hbuf[i] = silu(gbuf[i]) * ubuf[i];
            matmul(dw, hbuf.data(), ff.data()+(size_t)t*H, H, I);
        }
        for (size_t i = 0; i < (size_t)seq*H; ++i) hidden[i] += ff[i];
    }

    const float* fn = W(std::string(LM)+"norm.weight", nullptr);
    int64_t en = 0; const float* E = w_.get_f32(EMB, &en);
    if (!fn || !E) return false;
    int64_t vocab = en / H;
    logits_out.assign((size_t)seq * vocab, 0.0f);
    std::vector<float> fno((size_t)H);
    for (int t = 0; t < seq; ++t) {
        rmsnorm(hidden.data()+(size_t)t*H, fn, fno.data(), H, eps);
        matmul(E, fno.data(), logits_out.data()+(size_t)t*vocab, (int)vocab, H);
    }
    return !missing_;
}
