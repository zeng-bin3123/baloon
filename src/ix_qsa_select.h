// QsaBlockSelect: block scoring (relu'd head dots, +1e9 for incomplete tail) and radix top-k over weighted blocks -> cell ids.
#pragma once
#include "ix_u4_common.h"
IX4_HD int qsa_block_weight(int n_kv, int n_bid, int b) { return b < n_bid ? 4 : n_kv - n_bid * 4; }
IX4_HD int qsa_cell_id(int b, int c) { return b * 4 + c; }
IX4_HD float qsa_block_score(const float* head_dots, int n_head, bool incomplete_tail) {
    float s = 0.f; for (int h = 0; h < n_head; ++h) s += head_dots[h] > 0.f ? head_dots[h] : 0.f;
    return incomplete_tail ? s + 1e9f : s;
}
IX4_HD uint32_t qsa_ordered_key(float f) { uint32_t u; memcpy(&u, &f, 4); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
// Select `width` cells from blocks (key per block, weight per block). If n_kv <= width everything is selected.
// Writes selected cell ids (ascending by selection order: all above-threshold in index order, then equal keys lowest-index-first merged ascending).
// Returns number of cells written. `sel` must hold at least min(width, n_kv).
inline int qsa_radix_select(const uint32_t* key, int n_blocks, int n_kv, int width, int* sel) {
    int n_bid = n_kv / 4;
    auto wt = [&](int b) { return qsa_block_weight(n_kv, n_bid, b); };
    int total = 0;
    if (n_kv <= width) { for (int c = 0; c < n_kv; ++c) sel[total++] = c; return total; }
    uint32_t prefix = 0; long above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        long hist[256] = {0}; uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int b = 0; b < n_blocks; ++b) if ((key[b] & hi_mask) == (prefix & hi_mask)) hist[(key[b] >> shift) & 255] += wt(b);
        int d = 255; long acc = above;
        for (; d >= 0; --d) { if (acc + hist[d] >= width) break; acc += hist[d]; }
        if (d < 0) d = 0;
        above = acc; prefix |= (uint32_t)d << shift;
    }
    long budget = width - above;
    for (int b = 0; b < n_blocks; ++b) {
        int w = wt(b); bool take = key[b] > prefix; int n = w;
        if (!take && key[b] == prefix && budget > 0) { take = true; n = (int)(budget < w ? budget : w); budget -= n; }
        if (take) for (int c = 0; c < n; ++c) sel[total++] = qsa_cell_id(b, c);
    }
    return total;
}
