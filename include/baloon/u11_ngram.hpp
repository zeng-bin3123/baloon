// u11_ngram.hpp - the n-gram / PLE table surface of the reference layout (src/kernels/ngram.cpp): hash rows, IQ4_NL row
// dequantisation, the prefetch switch and a file-backed row table with the one-token-in-flight protocol.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "baloon/ple_ngram_kernels.hpp"

namespace baloon {
namespace u11 {

extern const int8_t kIq4Nl[16];
int iq4nl_code(int nibble);
void iq4nl_dequant_row(const uint8_t* row, float* out);   // 160 values from PLE_ROW_BYTES (= 90) bytes: 5 x (fp16 d + 16 nibble bytes)
uint64_t ngram_mixed(const int64_t* ctx, int n);
void ngram_rows(const int64_t* window, int n, int64_t eos, uint64_t rows[PLE_N_HEADS]);
PleConsts ple_artifact_consts();
void ple_prefetch_enable(bool on);
bool ple_prefetch_enabled();

class PleTable {
public:
    PleTable();
    ~PleTable();
    PleTable(const PleTable&) = delete;
    PleTable& operator=(const PleTable&) = delete;
    bool open(const std::string& path, uint64_t data_start, uint64_t n_rows, const PleIoOptions& o = {});
    void close();
    bool is_open() const;
    uint64_t rows() const;
    uint64_t bytes_read() const;
    const char* mode() const;
    void set_injected_delay_us(uint64_t us);
    bool read_row(uint64_t row, uint8_t* out);
    bool gather(const uint64_t rows16[PLE_N_HEADS], uint8_t* out);
    bool gather_batch(const uint64_t* rows, size_t n, uint8_t* out);
    bool issue();
    bool collect();
    bool issue(const uint64_t rows16[PLE_N_HEADS]);
    bool collect(uint8_t* out);
    std::string io_report() const;
private:
    std::unique_ptr<baloon::PleTable> t_;
};

}  // namespace u11
}  // namespace baloon
