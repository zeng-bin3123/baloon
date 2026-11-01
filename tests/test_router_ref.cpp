#include "router_ref.h"
#include <cstdio>
#include <random>
int main() {
    std::mt19937 rng(1); std::normal_distribution<float> nd(0.f, 3.f); int bad = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        const int n = 256, k = 8; std::vector<float> l(n);
        for (auto& v : l) v = nd(rng);
        if (trial % 7 == 0) for (int i = 0; i < 40; ++i) l[i] = 1.5f;           // force ties
        int ids[8]; float w[8]; router_ref(l.data(), n, k, ids, w);
        double s = 0; for (int i = 0; i < k; ++i) s += w[i];
        bool ok = std::fabs(s - 1.0) < 1e-6;
        for (int i = 1; i < k; ++i) ok &= (w[i-1] >= w[i]);
        for (int i = 0; i < k; ++i) for (int j = i+1; j < k; ++j) ok &= (ids[i] != ids[j]);
        if (trial % 7 == 0) for (int i = 1; i < k; ++i) if (w[i-1] == w[i]) ok &= (ids[i-1] < ids[i]); // tie -> lowest idx first
        bad += !ok;
    }
    printf("router_ref property test: %s (%d bad / 2000)\n", bad ? "FAIL" : "PASS", bad);
    return bad != 0;
}
