// Shared by the oracle generator and the test: the synthetic table image and the oracle-vector layout.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "baloon/ple_ngram_kernels.hpp"

namespace k30 {
constexpr uint64_t kSynthStart = 4096, kSynthRows = 2048;
inline uint8_t synth_byte(uint64_t row, uint64_t j) { return (uint8_t)((row * 131 + j * 7 + 13 + (row >> 3) * 5) & 0xFF); }
inline bool write_synth(const std::string& path, uint64_t start = kSynthStart, uint64_t rows = kSynthRows) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::vector<uint8_t> hdr(start, 0xAB);
    bool ok = std::fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size();
    uint8_t r[baloon::PLE_ROW_BYTES];
    for (uint64_t i = 0; i < rows && ok; ++i) {
        for (uint64_t j = 0; j < (uint64_t)baloon::PLE_ROW_BYTES; ++j) r[j] = synth_byte(i, j);
        ok = std::fwrite(r, 1, sizeof r, f) == sizeof r;
    }
    return std::fclose(f) == 0 && ok;
}
}  // namespace k30
