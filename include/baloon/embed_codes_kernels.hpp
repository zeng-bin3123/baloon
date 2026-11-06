#pragma once
// GPU decode of quantized embedding rows: the code of element i is the code_bits-wide field at bit (i % perbyte) * code_bits
// of byte i / perbyte (perbyte = 8 / code_bits, low field first). Same contract as core::embed_row (quantized overload).
#include <cuda_runtime.h>
#include <cstdint>
namespace baloon {
namespace embed_codes {
// ids[k] outside [0, vocab) yields a row of zero codes (and ok flag 0 in row_ok when non-null). n must be a multiple of
// perbyte; code_bits must be 2, 4 or 8. Returns cudaErrorInvalidValue for violated preconditions, no launch is made then.
cudaError_t embed_codes_rows(const uint8_t* table, int code_bits, int64_t vocab, int64_t n, const int32_t* ids, int n_ids,
                             uint32_t* out, uint8_t* row_ok, cudaStream_t s = 0);
}  // namespace embed_codes
}  // namespace baloon
