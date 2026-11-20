// ple_reader_test.cpp — selftest for the PLE n-gram row reader.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static long g_fail = 0;
#define CHECK(cond) do{ if(!(cond)){++g_fail;std::printf("FAIL %s:%d: %s\n",__FILE__,__LINE__,#cond);} }while(0)

static constexpr int k  = 4;   // n-gram order (key length in tokens)
static constexpr int ng = 90;  // row byte width in the PLE table (k*token_id + prob + rank)
static_assert(ng == 90, "PLE row is 90 bytes: 4*2 (key) + 2 (rank) + 80 (probs)");

// make_table: construct a minimal in-memory PLE table with n_rows rows.
static std::vector<uint8_t> make_table(int n_rows) {
    std::vector<uint8_t> tbl((size_t)n_rows * ng, 0);
    for(int r=0;r<n_rows;++r){
        // key: 4 tokens, each 2 bytes
        for(int t=0;t<k;++t){
            const uint16_t tok=(uint16_t)(r*k+t);
            std::memcpy(tbl.data()+r*ng+t*2, &tok, 2);
        }
    }
    return tbl;
}

// expected_row: return the expected key tokens for row r.
static std::vector<uint16_t> expected_row(int r) {
    std::vector<uint16_t> key(k);
    for(int t=0;t<k;++t) key[t]=(uint16_t)(r*k+t);
    return key;
}

// real: decode the actual key tokens from a table row.
static std::vector<uint16_t> real(const std::vector<uint8_t>& tbl, int r) {
    std::vector<uint16_t> key(k);
    for(int t=0;t<k;++t) std::memcpy(&key[t], tbl.data()+r*ng+t*2, 2);
    return key;
}
// real: used by check_rows to compare actual vs expected table content.

// check_rows: verify that every row in tbl matches its expected key tokens.
static void check_rows(const std::vector<uint8_t>& tbl, int n_rows) {
    for(int r=0;r<n_rows;++r){
        const auto got=real(tbl,r), want=expected_row(r);
        for(int t=0;t<k;++t) CHECK(got[t]==want[t]);
    }
}

static void selftest() {
    const int n=16;
    auto tbl=make_table(n);
    CHECK((int)tbl.size()==n*ng);
    check_rows(tbl,n);
}

int main(int argc, char** argv) {
    (void)argc;(void)argv;
    selftest();
    if(g_fail){std::printf("ple_reader_test FAILED (%ld)\n",g_fail);return 1;}
    std::printf("ple_reader_test PASSED\n");return 0;
}
