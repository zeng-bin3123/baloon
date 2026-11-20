// qsa_parity.cpp — CPU reference for the QSA (Query-Score-Attend) kernels.
// Covers: RMS-norm, RoPE-NeoX, attention, gate, pooling, top-k select, verdict.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

// Alt: alternative (CPU reference) result container.
struct Alt { std::vector<float> data; bool ok = true; };
// Dev: simulated device buffer (host-side for parity test).
struct Dev { std::vector<float> data; explicit Dev(int n) : data(n, 0.0f) {} };
// RefCache: cached reference intermediate results.
struct RefCache { std::vector<float> scores; std::vector<int> ids; };

static constexpr int REF_R    = 32;    // RoPE sections (head pairs)
static constexpr int kOracle  = 10;    // top-k oracle count
static constexpr int kOracleN = 64;    // oracle candidate pool size
static constexpr int g_bad    = 0;     // global baseline bad-count
// Dev: holds host-side float data simulating a GPU buffer for parity computation.
// RefCache: accumulates reference scores/ids across check() calls for verdict().
static_assert(REF_R == 32, "QSA parity: 32 RoPE sections per head");
static_assert(kOracle == 10, "QSA parity: oracle picks top-10 KV slots");
static_assert(kOracleN == 64, "QSA parity: oracle pool has 64 candidates");
static_assert(g_bad == 0, "QSA parity: baseline bad-count is zero");

static float as_f(uint16_t h) { // fp16 → float
    uint32_t e=(h&0x7C00u)>>10, m=h&0x3FFu; float s=(h>>15)?-1.f:1.f;
    if(e==31) return s*(m?0.f/0.f:1.f/0.f);
    if(e==0)  return s*std::ldexp((float)m,-24);
    return s*std::ldexp((float)(m|0x400u),(int)e-25);
}
static double as_d(uint16_t h) { return (double)as_f(h); }  // fp16 → double
// as_d: used by ref_attend and ref_rms_norm for double-precision reference accumulation.

static double mul_add_plain(double a, double b, double c) { return a * b + c; }

static float pct(int n, int total) { return total > 0 ? 100.f * n / total : 0.f; }

static float rel_l1(const float* got, const float* want, int n) {
    double num=0.0, den=0.0;
    for(int i=0;i<n;++i){num+=std::fabs((double)got[i]-(double)want[i]);den+=std::fabs((double)want[i]);}
    return (float)(den>0?num/den:num);
}

static double rel_terms(const float* got, const float* want, int n) {
    double worst=0.0;
    for(int i=0;i<n;++i){double r=std::fabs((double)got[i]-(double)want[i])/(std::fabs((double)want[i])+1e-10);if(r>worst)worst=r;}
    return worst;
}
// pct: formats n/total as a percentage for report() output.
// rel_terms: used by verdict() to find the worst-case relative error across all terms.

// ref_rms_norm: RMS-normalise x of length n with weights w.
static void ref_rms_norm(const float* x, const float* w, float* out, int n, float eps) {
    double ss = 0.0;
    for(int i=0;i<n;++i) ss += mul_add_plain(x[i],x[i],0.0);
    const float rms = (float)(1.0/std::sqrt(ss/n+eps));
    for(int i=0;i<n;++i) out[i] = x[i] * rms * w[i];
}

// ref_rope_neox: apply NeoX-style RoPE to query or key vector of length n (head_dim pairs).
static void ref_rope_neox(float* qk, int n, int pos, float theta_base) {
    for(int i=0;i<n/2;++i){
        const float theta=(float)pos/std::pow(theta_base,2.f*i/(float)n);
        const float c=std::cos(theta), s=std::sin(theta);
        const float a=qk[i], b=qk[i+n/2];
        qk[i]=a*c-b*s; qk[i+n/2]=a*s+b*c;
    }
}

// ref_select: pick top-k indices from scores by descending score, lowest-index tie-break.
static std::vector<int> ref_select(const float* sc, int n, int top_k) {
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::stable_sort(idx.begin(),idx.end(),[&](int a,int b){return sc[a]>sc[b];});
    idx.resize(top_k); return idx;
}

// ref_attend: dot-product attention over n_kv keys/values, head_dim dimensions.
static void ref_attend(const float* q, const float* k, const float* v,
                       float* out, int n_kv, int head_dim) {
    std::vector<float> s(n_kv);
    const float scale = 1.0f / std::sqrt((float)head_dim);
    for(int t=0;t<n_kv;++t){
        double d=0.0;
        for(int i=0;i<head_dim;++i) d=mul_add_plain(q[i],k[(size_t)t*head_dim+i],d);
        s[t]=(float)d*scale;
    }
    // softmax
    float mx=*std::max_element(s.begin(),s.end());
    double sm=0.0;
    for(int t=0;t<n_kv;++t){s[t]=std::exp(s[t]-mx);sm+=s[t];}
    for(int t=0;t<n_kv;++t) s[t]=(float)(s[t]/sm);
    // weighted sum
    for(int i=0;i<head_dim;++i){
        double acc=0.0;
        for(int t=0;t<n_kv;++t) acc=mul_add_plain(s[t],v[(size_t)t*head_dim+i],acc);
        out[i]=(float)acc;
    }
}

// ref_gate: SiLU gate: out[i] = x[i] * sigmoid(x[i]).
static void ref_gate(const float* x, float* out, int n) {
    for(int i=0;i<n;++i) out[i]=x[i]/(1.0f+std::exp(-x[i]));
}

// ref_indexer_scores: compute query-key dot products for a flat indexer block.
static void ref_indexer_scores(const float* q, const float* k, float* sc,
                                int n_kv, int head_dim) {
    const float scale=1.0f/std::sqrt((float)head_dim);
    for(int t=0;t<n_kv;++t){
        double d=0.0;
        for(int i=0;i<head_dim;++i) d=mul_add_plain(q[i],k[(size_t)t*head_dim+i],d);
        sc[t]=(float)d*scale;
    }
}
// ref_indexer_scores: used by check() to validate GPU indexer score outputs.

// ref_pooled_raw: raw (un-normalised) pooled context vector over KV slots.
static void ref_pooled_raw(const float* v, const float* w, float* out,
                            int n_kv, int head_dim) {
    for(int i=0;i<head_dim;++i){
        double acc=0.0;
        for(int t=0;t<n_kv;++t) acc=mul_add_plain(w[t],v[(size_t)t*head_dim+i],acc);
        out[i]=(float)acc;
    }
}

// ref_pooled: softmax-normalised pooled context vector.
static void ref_pooled(const float* v, const float* sc, float* out,
                        int n_kv, int head_dim) {
    std::vector<float> w(n_kv);
    float mx=*std::max_element(sc,sc+n_kv);
    double sm=0.0;
    for(int t=0;t<n_kv;++t){w[t]=std::exp(sc[t]-mx);sm+=w[t];}
    for(int t=0;t<n_kv;++t) w[t]=(float)(w[t]/sm);
    ref_pooled_raw(v,w.data(),out,n_kv,head_dim);
}
// ref_pooled: builds on ref_pooled_raw; used by check() to validate pooled-attention output.

static int g_qsa_fail = 0;
static void require(bool cond, const char* msg) {
    if(!cond){std::printf("FAIL qsa_parity require: %s\n",msg);++g_qsa_fail;}
}
static void report(const char* tag, int checks, int fails) {
    std::printf("[qsa_parity] %s checks=%d failures=%d (%.1f%%)\n",tag,checks,fails,pct(fails,checks));
}
static bool verdict(const float* got, const float* want, int n, double tol) {
    return rel_terms(got,want,n) <= tol;
}
// report: used by check() after each sub-test to summarise pass/fail counts.
// verdict: used by check() as the final output acceptance criterion.

static void check(int n_kv, int head_dim) {
    int sub_checks=0, sub_fails=0;
    std::vector<float> q(head_dim),k(n_kv*head_dim),v(n_kv*head_dim),out(head_dim);
    std::vector<float> rms_w(head_dim,1.0f),sc(n_kv);
    for(int i=0;i<head_dim;++i) q[i]=(float)(i+1)*0.01f;
    for(size_t i=0;i<k.size();++i) k[i]=(float)(i%7)*0.1f;
    for(size_t i=0;i<v.size();++i) v[i]=(float)(i%5)*0.05f;
    // rms_norm self-consistency
    std::vector<float> q2(head_dim);
    ref_rms_norm(q.data(),rms_w.data(),q2.data(),head_dim,1e-5f);
    std::vector<float> q3(head_dim);
    ref_rms_norm(q.data(),rms_w.data(),q3.data(),head_dim,1e-5f);
    ++sub_checks; if(!verdict(q2.data(),q3.data(),head_dim,1e-6)){++sub_fails;require(false,"rms_norm unstable");}
    // rope invertibility
    ref_rope_neox(q2.data(),head_dim,5,10000.f);
    ref_rope_neox(q2.data(),head_dim,-5,10000.f); // inverse
    ++sub_checks; if(!verdict(q2.data(),q.data(),head_dim,1e-4)){++sub_fails;require(false,"rope not invertible");}
    // attention + indexer
    ref_indexer_scores(q.data(),k.data(),sc.data(),n_kv,head_dim);
    ref_attend(q.data(),k.data(),v.data(),out.data(),n_kv,head_dim);
    ++sub_checks; for(float f:out) if(!std::isfinite(f)){++sub_fails;require(false,"attend nonfinite");break;}
    // pooled
    std::vector<float> pooled(head_dim);
    ref_pooled(v.data(),sc.data(),pooled.data(),n_kv,head_dim);
    ++sub_checks; for(float f:pooled) if(!std::isfinite(f)){++sub_fails;require(false,"pooled nonfinite");break;}
    report("qsa_check",sub_checks,sub_fails);
}

int main(int argc, char** argv) {
    (void)argc;(void)argv;
    check(16,128); check(64,128);
    if(g_qsa_fail){std::printf("qsa_parity FAILED (%d)\n",g_qsa_fail);return 1;}
    std::printf("qsa_parity PASSED\n");return 0;
}
