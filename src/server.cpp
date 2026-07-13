// Minimal HTTP server exposing the C++ Qwen3.5 engine for a browser test UI.
// Loads the model + tokenizer once, then streams generated tokens (chunked
// transfer) so the slow naive-matmul generation feels live in the browser.
//
// Usage: mllm_server [model_dir] [port]   (default port 8080)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "model_config.h"
#include "safetensors_loader.h"
#include "qwen_model.h"
#include "tokenizer.h"
#pragma comment(lib, "ws2_32.lib")

static const int MAX_CTX = 1024;

static void send_all(SOCKET s, const char* data, int len) {
    int sent = 0;
    while (sent < len) {
        int n = send(s, data + sent, len - sent, 0);
        if (n <= 0) return;
        sent += n;
    }
}
static void send_str(SOCKET s, const std::string& str) { send_all(s, str.data(), (int)str.size()); }

// One HTTP chunk (Transfer-Encoding: chunked): <hex-len>CRLF<data>CRLF.
static void send_chunk(SOCKET s, const std::string& data) {
    char hdr[32];
    int n = std::snprintf(hdr, sizeof(hdr), "%zx\r\n", data.size());
    send_all(s, hdr, n);
    send_str(s, data);
    send_all(s, "\r\n", 2);
}

static std::string url_decode(const std::string& in) {
    std::string out;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '+') out.push_back(' ');
        else if (in[i] == '%' && i + 2 < in.size()) {
            auto hex = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
            out.push_back((char)((hex(in[i+1]) << 4) | hex(in[i+2]))); i += 2;
        } else out.push_back(in[i]);
    }
    return out;
}

static std::string query_param(const std::string& query, const std::string& key) {
    std::string pat = key + "=";
    size_t p = query.find(pat);
    if (p == std::string::npos) return "";
    p += pat.size();
    size_t e = query.find('&', p);
    return url_decode(query.substr(p, e == std::string::npos ? std::string::npos : e - p));
}

static const char* HTML = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><title>Qwen3.5-0.8B — C++ engine</title>
<style>
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:0 auto;padding:28px 20px;
 background:#faf9f7;color:#1a1a1a}
@media(prefers-color-scheme:dark){body{background:#16161a;color:#e8e8ea}}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#888;font-size:13px;margin:0 0 20px}
.badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;background:#2b7a3d22;color:#2b9a4d;
 border:1px solid #2b9a4d55;margin-left:8px;vertical-align:middle}
textarea{width:100%;min-height:80px;padding:11px 13px;border:1px solid #ccc4;border-radius:10px;font:inherit;
 resize:vertical;background:#fff;color:inherit}
@media(prefers-color-scheme:dark){textarea,.out{background:#1f1f25;border-color:#3a3a42}}
.row{display:flex;gap:12px;align-items:center;margin:12px 0}
label{font-size:13px;color:#888}
input[type=number]{width:72px;padding:7px 9px;border:1px solid #ccc4;border-radius:8px;font:inherit;background:#fff;color:inherit}
button{padding:9px 20px;border:0;border-radius:9px;background:#2b6cff;color:#fff;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.out{white-space:pre-wrap;word-wrap:break-word;margin-top:18px;padding:16px 18px;border:1px solid #e5e2dc;
 border-radius:12px;background:#fff;min-height:60px}
.gen{color:#2b6cff}
@media(prefers-color-scheme:dark){.gen{color:#7aa2ff}}
.meta{color:#999;font-size:12px;margin-top:8px;height:16px}
</style></head><body>
<h1>Qwen3.5-0.8B <span class="badge">pure C++ engine</span></h1>
<p class="sub">Text-to-text via the hand-written C++ forward pass + byte-level BPE tokenizer. Greedy decoding.</p>
<textarea id="prompt">What is 2 + 19?</textarea>
<div class="row">
  <label>max new tokens</label><input type="number" id="n" value="48" min="1" max="512">
  <label><input type="checkbox" id="chat" checked> chat mode</label>
  <button id="go">Generate</button>
</div>
<p class="sub" style="margin:-6px 0 0">Chat mode wraps your text in the instruct template so the model answers; uncheck for raw text continuation.</p>
<div class="out" id="out"></div>
<div class="meta" id="meta"></div>
<script>
const go=document.getElementById('go'),out=document.getElementById('out'),meta=document.getElementById('meta');
go.onclick=async()=>{
  const prompt=document.getElementById('prompt').value, n=document.getElementById('n').value;
  const chat=document.getElementById('chat').checked?1:0;
  go.disabled=true; out.textContent=''; meta.textContent='running…';
  const t0=performance.now();
  try{
    const r=await fetch('/gen?chat='+chat+'&n='+encodeURIComponent(n)+'&prompt='+encodeURIComponent(prompt));
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='',tokens=0;
    while(true){
      const {done,value}=await rd.read(); if(done)break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\x1e'); const last=parts[parts.length-1];
      const f1=last.indexOf('\x1f'), f2=last.indexOf('\x1f',f1+1);
      if(f1>=0&&f2>=0){ tokens=+last.slice(0,f1); const pl=+last.slice(f1+1,f2); const text=last.slice(f2+1);
        out.innerHTML=escapeHtml(text.slice(0,pl))+'<span class="gen">'+escapeHtml(text.slice(pl))+'</span>'; }
      const dt=(performance.now()-t0)/1000;
      meta.textContent=tokens+' tokens · '+dt.toFixed(1)+'s · '+(tokens/dt||0).toFixed(1)+' tok/s';
    }
  }catch(e){ meta.textContent='error: '+e; }
  go.disabled=false;
};
function escapeHtml(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
</script></body></html>)HTML";

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "models/Qwen3.5-0.8B";
    int port = (argc > 2) ? std::stoi(argv[2]) : 8080;

    ModelConfig cfg;
    load_model_config(model_dir + "/config.json", cfg); // dims from the model folder

    SafeTensors weights;
    if (!weights.load_model(model_dir)) return 1;       // all shards
    Tokenizer tok;
    if (!tok.load(model_dir + "/vocab.json", model_dir + "/merges.txt")) return 1;
    tok.load_specials(model_dir + "/tokenizer_config.json");

    std::cout << "Planning arenas + converting weights to fp32...\n";
    QwenModel model(weights, cfg, MAX_CTX); // constructor resolves+warms all weights

    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
    SOCKET srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((u_short)port);
    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) != 0) { std::cerr << "bind failed on port " << port << "\n"; return 1; }
    listen(srv, 4);
    std::cout << "\n  Ready -> open http://localhost:" << port << "  (Ctrl+C to stop)\n\n";

    const int64_t vocab = cfg.vocab_size;
    int eos = tok.special_id("<|endoftext|>"); if (eos < 0) eos = 248044;
    int im_end = tok.special_id("<|im_end|>");  if (im_end < 0) im_end = 248046;

    for (;;) {
        SOCKET cli = accept(srv, nullptr, nullptr);
        if (cli == INVALID_SOCKET) continue;

        std::string req; char buf[4096]; int n;
        while ((n = recv(cli, buf, sizeof(buf), 0)) > 0) {
            req.append(buf, n);
            if (req.find("\r\n\r\n") != std::string::npos) break;
        }
        // Request line: METHOD PATH HTTP/1.1
        size_t sp1 = req.find(' '), sp2 = req.find(' ', sp1 + 1);
        std::string path = (sp1 != std::string::npos && sp2 != std::string::npos)
            ? req.substr(sp1 + 1, sp2 - sp1 - 1) : "/";

        if (path == "/" ) {
            std::string body(HTML);
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: "
                + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
            send_str(cli, resp);
        } else if (path.rfind("/gen", 0) == 0) {
            size_t q = path.find('?');
            std::string query = q == std::string::npos ? "" : path.substr(q + 1);
            std::string prompt = query_param(query, "prompt");
            bool chat = query_param(query, "chat") == "1";
            int nnew = 24; try { nnew = std::stoi(query_param(query, "n")); } catch (...) {}

            std::vector<int> ids = chat ? tok.encode_chat(prompt) : tok.encode(prompt);
            if ((int)ids.size() >= MAX_CTX - 2) ids.resize(MAX_CTX - 2);
            if (nnew > MAX_CTX - (int)ids.size() - 1) nnew = MAX_CTX - (int)ids.size() - 1;

            send_str(cli, "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\n"
                          "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");

            model.reset();
            const std::vector<float>* lg = nullptr;
            for (int id : ids) lg = &model.step(id);

            // In chat mode we show only the assistant's answer; in raw mode we
            // echo the prompt (prefix) + continuation. prefix_len = byte length
            // of the non-generated portion the client renders in the base color.
            std::string prefix = chat ? "" : tok.decode(ids);
            std::vector<int> gen;
            auto snap = [&](int count) {
                std::string disp = chat ? tok.decode(gen) : (prefix + tok.decode(gen));
                send_chunk(cli, std::string("\x1e") + std::to_string(count) + "\x1f"
                                + std::to_string(prefix.size()) + "\x1f" + disp);
            };
            snap(0);
            for (int i = 0; i < nnew && lg; ++i) {
                int am = 0; const std::vector<float>& L = *lg;
                for (int64_t v = 1; v < vocab; ++v) if (L[v] > L[am]) am = (int)v;
                if (am == eos || am == im_end) break;
                gen.push_back(am);
                snap((int)gen.size());
                lg = &model.step(am);
            }
            send_all(cli, "0\r\n\r\n", 5); // end of chunked stream
        } else {
            send_str(cli, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        }
        closesocket(cli);
    }
}
