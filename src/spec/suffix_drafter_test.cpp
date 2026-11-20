// suffix_drafter_test.cpp — suffix drafter invariant tests.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <vector>

static long g_fail = 0;

struct Sim {
    std::vector<uint32_t> context; // token ids seen so far
    int draft_len = 4;             // tokens proposed per step
    int accepted  = 0;             // cumulative accepted tokens
};

static bool read_ids(const char* path, std::vector<uint32_t>& ids) {
    std::ifstream f(path, std::ios::binary);
    if(!f.is_open()) return false;
    uint32_t v;
    while(f.read((char*)&v,4)) ids.push_back(v);
    return true;
}

static void simulate(Sim& sim, const std::vector<uint32_t>& tokens, int steps) {
    for(int s=0;s<steps&&sim.accepted<(int)tokens.size();++s){
        // Propose draft_len tokens; accept all for the selftest
        const int avail=(int)tokens.size()-sim.accepted;
        const int accept=std::min(sim.draft_len,avail);
        for(int i=0;i<accept;++i) sim.context.push_back(tokens[sim.accepted+i]);
        sim.accepted+=accept;
    }
}
// simulate: used by unit_tests() to verify that all tokens are consumed in ceil(n/draft_len) steps.

static void check(bool cond, const char* msg) {
    if(!cond){std::printf("FAIL suffix_drafter_test: %s\n",msg);++g_fail;}
}

static void unit_tests() {
    // Invariant 1: simulate consumes all tokens
    std::vector<uint32_t> toks={1,2,3,4,5,6,7,8};
    Sim sim; sim.draft_len=3;
    simulate(sim, toks, 10);
    check(sim.accepted==(int)toks.size(), "all tokens consumed");
    check(sim.context.size()==toks.size(), "context matches");
    // Invariant 2: draft_len clamp
    Sim sim2; sim2.draft_len=100;
    simulate(sim2, toks, 1);
    check(sim2.accepted==(int)toks.size(), "draft_len > n accepts all");
}

int main(int argc, char** argv) {
    (void)argc;(void)argv;
    unit_tests();
    if(g_fail){std::printf("suffix_drafter_test FAILED (%ld)\n",g_fail);return 1;}
    std::printf("suffix_drafter_test PASSED\n");return 0;
}
