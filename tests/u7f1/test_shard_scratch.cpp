// tests/u7f1/test_shard_scratch.cpp — NativeDenseOverride theorems: shardOk_number,
// first_not_additional, scratch_wide_enough.
// Theory:
//   shardOk_number       : shardOk ... → 0 < n ∧ n < fc
//   first_not_additional : shardOk c 0 t fc ft = false (shard 0 is never "additional")
//   scratch_wide_enough  : w ∈ widths → w ≤ maxIn widths
#include "u7f1/check.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>
int main() {
    std::puts("=== test_shard_scratch ===");

    // ---- shardOk: shard metadata matching ----
    // shardOk count no tensors firstCount firstTensors =
    //   firstCount != 0 && count == firstCount && no != 0 && no < firstCount && tensors == firstTensors
    auto shardOk = [](uint64_t count, uint64_t no, uint64_t tensors,
                      uint64_t fc, uint64_t ft) -> bool {
        return fc != 0 && count == fc && no != 0 && no < fc && tensors == ft;
    };

    // first_not_additional: shard number 0 is always rejected
    CHK(!shardOk(3, 0, 10, 3, 10));  // no=0: first shard, not additional

    // shardOk_number: accepted shard has 0 < no < fc
    CHK( shardOk(3, 1, 10, 3, 10));  // no=1 ∈ (0, 3): OK
    CHK( shardOk(3, 2, 10, 3, 10));  // no=2 ∈ (0, 3): OK
    CHK(!shardOk(3, 3, 10, 3, 10));  // no=3 not < fc=3: rejected

    // mismatched count or tensor count
    CHK(!shardOk(4, 1, 10, 3, 10));  // count ≠ firstCount
    CHK(!shardOk(3, 1, 11, 3, 10));  // tensors ≠ firstTensors
    CHK(!shardOk(3, 1, 10, 0, 10));  // firstCount = 0: rejected

    // ---- maxIn (scratch sizing): max over a list of widths ----
    // foldl_max_ge_mem: every element ≤ maxIn
    auto maxIn = [](const std::vector<uint64_t>& ws) -> uint64_t {
        uint64_t m = 0;
        for (auto w : ws) m = std::max(m, w);
        return m;
    };

    std::vector<uint64_t> widths = {512, 1024, 2048, 4096, 1280};
    uint64_t mi = maxIn(widths);
    CHK(mi == 4096u);  // scratch must hold the widest input

    // scratch_wide_enough: every width ≤ maxIn
    for (auto w : widths) {
        CHK(w <= mi);
    }

    // empty list: maxIn = 0
    CHK(maxIn({}) == 0u);

    return u7f1::finish("TEST_SHARD_SCRATCH");
}

