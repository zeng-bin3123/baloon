// GenerateDriver theory § speculative acceptance
// Contracts: accept, accept_le_left, accept_le_right, accept_self.
// accept = length of longest equal prefix of drafts and verifier outputs
// accept_le_left: accept ds os <= ds.length
// accept_le_right: accept ds os <= os.length
// accept_self: accept xs xs = xs.length
#include "check.hpp"
#include <vector>
#include <cstddef>

// C++ model of the Lean accept function (recursive on the lists)
static int64_t accept_fn(const std::vector<int64_t>& ds, const std::vector<int64_t>& os) {
    size_t i = 0;
    while (i < ds.size() && i < os.size() && ds[i] == os[i]) ++i;
    return (int64_t)i;
}

void test_accept() {
    // accept_self: accept xs xs = xs.length
    for (size_t n = 0; n <= 15; ++n) {
        std::vector<int64_t> xs;
        for (size_t i = 0; i < n; ++i) xs.push_back((int64_t)i * 3 + 1);
        CHECK_MSG(accept_fn(xs, xs) == (int64_t)n, "n=%zu", n);
    }
    // accept_le_left and accept_le_right
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> cases = {
        {{},      {}},
        {{1},     {}},
        {{},      {1}},
        {{1,2,3}, {1,2,3}},
        {{1,2,3}, {1,9,3}},
        {{9,2,3}, {1,2,3}},
        {{1,2,3}, {1,2}},
        {{1},     {1,2,3}},
    };
    for (auto& [ds, os] : cases) {
        const int64_t a = accept_fn(ds, os);
        CHECK_MSG(a <= (int64_t)ds.size(), "accept_le_left: a=%lld ds.size=%zu", (long long)a, ds.size());
        CHECK_MSG(a <= (int64_t)os.size(), "accept_le_right: a=%lld os.size=%zu", (long long)a, os.size());
    }
    // specific values
    CHECK(accept_fn({}, {}) == 0);
    CHECK(accept_fn({1,2,3}, {1,2,3}) == 3);
    CHECK(accept_fn({1,2,3}, {1,9,3}) == 1);
    CHECK(accept_fn({9,2,3}, {1,2,3}) == 0);
    CHECK(accept_fn({1,2,3}, {1,2}) == 2);
    CHECK(accept_fn({1}, {1,2,3}) == 1);
}
