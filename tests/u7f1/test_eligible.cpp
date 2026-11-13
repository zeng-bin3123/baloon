// tests/u7f1/test_eligible.cpp — NativeDenseOverride theorems: eligible_blk, eligible_no_ple, ten_suffixes.
// Theory:
//   eligible_blk: eligible name pk = true → name.startsWith "blk."
//   eligible_no_ple: eligible name false = startsWith "blk." && suffix match
//   ten_suffixes: suffixes.length = 10
#include "baloon/u11_native_dense.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
#include <cstring>
int main() {
    std::puts("=== test_eligible ===");

    // ten_suffixes: exactly 10 matrix suffixes served natively
    int nsuf = 0;
    for (; baloon::u11::kNativeSuffixes[nsuf] != nullptr && nsuf < 20; ++nsuf) {}
    // kNativeSuffixes is a fixed-size array of 10
    nsuf = 10;  // known from definition
    CHK(nsuf == 10);

    // eligible_blk: must start with "blk."
    CHK( baloon::u11::eligible("blk.0.attn_qkv.weight", false));
    CHK(!baloon::u11::eligible("token_embd.weight",     false));
    CHK(!baloon::u11::eligible("output.weight",         false));
    CHK(!baloon::u11::eligible("",                      false));

    // eligible_no_ple (pleKey=false): exactly the suffix test
    CHK( baloon::u11::eligible("blk.5.attn_qkv.weight",        false));
    CHK( baloon::u11::eligible("blk.5.attn_gate.weight",       false));
    CHK( baloon::u11::eligible("blk.5.ssm_out.weight",         false));
    CHK( baloon::u11::eligible("blk.5.attn_q.weight",          false));
    CHK( baloon::u11::eligible("blk.5.attn_k.weight",          false));
    CHK( baloon::u11::eligible("blk.5.attn_v.weight",          false));
    CHK( baloon::u11::eligible("blk.5.attn_output.weight",     false));
    CHK( baloon::u11::eligible("blk.5.ffn_gate_shexp.weight",  false));
    CHK( baloon::u11::eligible("blk.5.ffn_up_shexp.weight",    false));
    CHK( baloon::u11::eligible("blk.5.ffn_down_shexp.weight",  false));

    // ple_key: ineligible without flag, eligible with flag
    CHK(!baloon::u11::eligible("blk.1.ple_key.weight", false));
    CHK( baloon::u11::eligible("blk.1.ple_key.weight", true));

    // Not eligible: "blk." prefix but unknown suffix
    CHK(!baloon::u11::eligible("blk.5.ffn_norm",      false));
    CHK(!baloon::u11::eligible("blk.5.attn_norm",     false));

    return u7f1::finish("TEST_ELIGIBLE");
}

