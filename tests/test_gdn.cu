// Phase 1 #6 accept: step / conv / l2 / out_norm / beta_gate vs host references (double where the reference is double).
#include "../src/ix_gdn.cu"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; } } while (0)
template <class T> static T* up(const std::vector<T>& h) { T* d; cudaMalloc(&d, h.size() * sizeof(T)); cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }
template <class T> static std::vector<T> down(T* d, size_t n) { std::vector<T> h(n); cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost); return h; }
static int chk(const char* nm, const std::vector<float>& g, const std::vector<double>& r, double tol) {
    double mx = 0, sc = 1e-30; for (size_t i = 0; i < r.size(); ++i) { mx = std::fmax(mx, std::fabs(g[i] - r[i])); sc = std::fmax(sc, std::fabs(r[i])); }
    const int bad = !(mx <= tol * sc); printf("%-22s max|d|=%.3g scale=%.3g rel=%.3g %s\n", nm, mx, sc, mx / sc, bad ? "FAIL" : "ok"); return bad;
}
int main() {
    int bad = 0; std::mt19937 rng(9); std::normal_distribution<float> nd(0.f, 1.f);
    // ---- step: S=128, h_k in {16,48}, h_v=48 (modulo head pairing)
    for (int hk : {16, 48}) {
        const int S = 128, hv = 48;
        std::vector<float> st((size_t)S*hv*S), q(hk*S), k(hk*S), v(hv*S), gate(hv), beta(hv);
        for (auto& x : st) x = nd(rng) * 0.1f; for (auto& x : q) x = nd(rng) * 0.1f; for (auto& x : k) x = nd(rng) * 0.1f;
        for (auto& x : v) x = nd(rng); for (auto& x : gate) x = -std::fabs(nd(rng)); for (auto& x : beta) x = 0.5f + 0.4f * std::tanh(nd(rng));
        std::vector<double> rs(st.begin(), st.end()), ro(hv * S);
        for (int h = 0; h < hv; ++h) { const int src = h % hk; const double dec = std::exp((double)gate[h]);
            for (int j = 0; j < S; ++j) { double sk = 0; for (int i = 0; i < S; ++i) { double s = rs[((size_t)i*hv+h)*S+j] * dec; rs[((size_t)i*hv+h)*S+j] = s; sk += s * k[src*S+i]; }
                double d = (v[h*S+j] - sk) * beta[h], dot = 0;
                for (int i = 0; i < S; ++i) { double s = rs[((size_t)i*hv+h)*S+j] + k[src*S+i] * d; rs[((size_t)i*hv+h)*S+j] = s; dot += s * q[src*S+i]; } ro[h*S+j] = dot; } }
        float *ds = up(st), *dq = up(q), *dk = up(k), *dv = up(v), *dg = up(gate), *db = up(beta), *dout; CK(cudaMalloc(&dout, hv*S*4));
        CK(ix_gdn_step(ds, dq, dk, dv, dg, db, dout, S, hk, hv)); CK(cudaDeviceSynchronize());
        char nm[40]; snprintf(nm, 40, "step.o   h_k=%d", hk); bad += chk(nm, down(dout, hv*S), ro, 1e-4);
        snprintf(nm, 40, "step.state h_k=%d", hk); bad += chk(nm, down(ds, st.size()), rs, 1e-4);
        cudaFree(ds); cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dg); cudaFree(db); cudaFree(dout);
    }
    // ---- conv: C channels, d_conv=4
    { const int C = 8192 + 5, dc = 4; std::vector<float> cs(C*(dc-1)), x(C), w(C*dc); for (auto& a : cs) a = nd(rng); for (auto& a : x) a = nd(rng); for (auto& a : w) a = nd(rng);
      std::vector<double> ro(C), rcs(C*(dc-1));
      for (int c = 0; c < C; ++c) { double acc = 0; for (int i = 0; i < dc-1; ++i) acc += (double)cs[c*(dc-1)+i] * w[c*dc+i]; acc += (double)x[c] * w[c*dc+dc-1]; ro[c] = acc;
        for (int i = 0; i < dc-2; ++i) rcs[c*(dc-1)+i] = cs[c*(dc-1)+i+1]; rcs[c*(dc-1)+dc-2] = x[c]; }
      float *dcs = up(cs), *dx = up(x), *dw = up(w), *dout; CK(cudaMalloc(&dout, C*4));
      CK(ix_gdn_conv_step(dcs, dx, dw, dout, C, dc)); CK(cudaDeviceSynchronize());
      bad += chk("conv.out", down(dout, C), ro, 1e-5); bad += chk("conv.state", down(dcs, C*(dc-1)), rcs, 0.0);
      cudaFree(dcs); cudaFree(dx); cudaFree(dw); cudaFree(dout); }
    // ---- l2 norm: cols incl. non-multiples of 64, up to 1024
    for (int cols : {128, 100, 7, 64, 65, 1024}) { const int rows = 33; const float eps = 1e-6f;
      std::vector<float> x(rows*cols); for (auto& a : x) a = nd(rng); std::vector<double> r(x.size());
      for (int i = 0; i < rows; ++i) { double s = 0; for (int c = 0; c < cols; ++c) s += (double)x[i*cols+c]*x[i*cols+c]; float inv = (float)(1.0/std::sqrt(s+(double)eps)); for (int c = 0; c < cols; ++c) r[i*cols+c] = x[i*cols+c]*inv; }
      float* dx = up(x); CK(ix_gdn_l2_norm(dx, rows, cols, eps)); CK(cudaDeviceSynchronize()); char nm[40]; snprintf(nm, 40, "l2 cols=%d", cols); bad += chk(nm, down(dx, x.size()), r, 1e-6); cudaFree(dx); }
    // ---- out_norm
    for (int S : {128, 100, 65}) { const int hv = 48; const float eps = 1e-6f; std::vector<float> o(hv*S), z(hv*S), nrm(S); for (auto& a : o) a = nd(rng); for (auto& a : z) a = nd(rng); for (auto& a : nrm) a = 1.f + 0.1f*nd(rng);
      std::vector<double> r(hv*S); for (int h = 0; h < hv; ++h) { double s = 0; for (int i = 0; i < S; ++i) s += (double)o[h*S+i]*o[h*S+i]; float inv = (float)(1.0/std::sqrt(s/S+(double)eps));
        for (int i = 0; i < S; ++i) r[h*S+i] = (double)(o[h*S+i]*inv*nrm[i]) / (1.0 + std::exp(-(double)z[h*S+i])); }
      float *dO = up(o), *dZ = up(z), *dN = up(nrm), *dY; CK(cudaMalloc(&dY, hv*S*4)); CK(ix_gdn_out_norm(dO, dZ, dN, dY, hv, S, eps)); CK(cudaDeviceSynchronize());
      char nm[40]; snprintf(nm, 40, "out_norm S=%d", S); bad += chk(nm, down(dY, hv*S), r, 1e-5); cudaFree(dO); cudaFree(dZ); cudaFree(dN); cudaFree(dY); }
    // ---- beta gate
    { const int n = 48; std::vector<float> b(n); for (auto& a : b) a = nd(rng) * 3; std::vector<double> r(n); for (int i = 0; i < n; ++i) r[i] = 1.0/(1.0+std::exp(-(double)b[i]));
      float* d = up(b); CK(ix_gdn_beta_gate(d, n)); CK(cudaDeviceSynchronize()); bad += chk("beta_gate", down(d, n), r, 1e-6); cudaFree(d); }
    printf("gdn: %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
