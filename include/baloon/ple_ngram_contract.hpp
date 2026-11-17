// ple_ngram_contract.hpp — compile-time contracts for every ported definition in
// ple_ngram_kernels.hpp.  One static_assert per port entry; the kernel checks the
// TSV against these names and types.  Build this header to prove every symbol is
// present with the right value or callable signature.
#pragma once
#include "baloon/ple_ngram_kernels.hpp"

namespace baloon {

// § 1/25 — const HEADS_PER_NGRAM
// Number of independent hash heads per n-gram order (bigram or trigram).
// Eight Iluvatar BI-V100 warps each handle one head simultaneously.
static_assert(HEADS_PER_NGRAM == 8,
              "port u89be81de0b: HEADS_PER_NGRAM must equal 8");

// § 2/25 — const NGRAM_SIZE
// Context window length in tokens: the current token plus up to two predecessors
// (unigram is not modelled; bigram uses 2 and trigram uses 3 of these slots).
static_assert(NGRAM_SIZE == 3,
              "port u89be81de0b: NGRAM_SIZE must equal 3");

// § 3/25 — const NG_HC
// Hyper-connection count: number of parallel residual streams that the n-gram
// block writes into the embedding.  Governs the conv channel count.
static_assert(NG_HC == 4,
              "port u89be81de0b: NG_HC must equal 4");

// § 4/25 — const NG_N_EMBD
// Hidden width of the n-gram model: 2560 floats per token, matching the main
// DeepSeek-V3 embedding dimension so the n-gram output adds directly into it.
static_assert(NG_N_EMBD == 2560,
              "port u89be81de0b: NG_N_EMBD must equal 2560");

// § 5/25 — const NG_HC_DIM
// Total hyper-connection dimension: NG_HC * NG_N_EMBD = 4 * 2560 = 10240.
// This is the flat width of the full HC injection buffer per token position.
static_assert(NG_HC_DIM == 10240,
              "port u89be81de0b: NG_HC_DIM must equal 10240");
static_assert(NG_HC_DIM == NG_HC * NG_N_EMBD,
              "port u89be81de0b: NG_HC_DIM must equal NG_HC * NG_N_EMBD");

// § 6/25 — const NG_HIST
// History depth fed into the temporal convolution: (PLE_CONV_KERNEL - 1) * NGRAM_SIZE
// = 3 * 3 = 9.  The ring buffer in the n-gram block holds exactly this many past rows.
static_assert(NG_HIST == 9,
              "port u89be81de0b: NG_HIST must equal 9");
static_assert(NG_HIST == (PLE_CONV_KERNEL - 1) * NGRAM_SIZE,
              "port u89be81de0b: NG_HIST must equal (PLE_CONV_KERNEL-1)*NGRAM_SIZE");

// § 7/25 — const NG_RMS_EPS
// Epsilon added to the variance before the square root in RMSNorm.  1e-6f matches
// the reference model's RMSNorm epsilon and avoids division by zero for zero-norm rows.
static_assert(NG_RMS_EPS == 1e-6f,
              "port u89be81de0b: NG_RMS_EPS must equal 1e-6f");

// § 8/25 — const PLE_CONV_KERNEL
// Temporal-conv kernel size over token positions: 4 taps.  Governs NG_HIST;
// increasing it widens the history ring without changing the hash itself.
static_assert(PLE_CONV_KERNEL == 4,
              "port u89be81de0b: PLE_CONV_KERNEL must equal 4");

// § 9/25 — const PLE_EOS_TOKEN_ID
// Token id that marks end-of-sequence in the DeepSeek-V3 vocabulary (248044).
// When a predecessor equals this id the EOS cut propagates; the token's own EOS
// does NOT cut its own context row, only predecessors.
static_assert(PLE_EOS_TOKEN_ID == 248044,
              "port u89be81de0b: PLE_EOS_TOKEN_ID must equal 248044");

// § 10/25 — const PLE_HEAD_DIM
// Output dimensionality of one hash head: 160 floats decoded from one 90-byte IQ4_NL
// row (5 blocks × 32 values).  PLE_N_HEADS × PLE_HEAD_DIM == NG_N_EMBD must hold.
static_assert(PLE_HEAD_DIM == 160,
              "port u89be81de0b: PLE_HEAD_DIM must equal 160");
static_assert(PLE_N_HEADS * PLE_HEAD_DIM == NG_N_EMBD,
              "port u89be81de0b: 16 heads * 160 dims must span NG_N_EMBD=2560");

// § 11/25 — const PLE_N_HEADS
// Total hash heads: (NGRAM_SIZE-1)*HEADS_PER_NGRAM = 2*8 = 16.
// Heads 0-7 are bigram heads; heads 8-15 are trigram heads.
// Each head issues one independent table lookup per decoded token.
static_assert(PLE_N_HEADS == 16,
              "port u89be81de0b: PLE_N_HEADS must equal 16");
static_assert(PLE_N_HEADS == (NGRAM_SIZE - 1) * HEADS_PER_NGRAM,
              "port u89be81de0b: PLE_N_HEADS must equal (NGRAM_SIZE-1)*HEADS_PER_NGRAM");

// § 12/25 — const PLE_ROW_BYTES
// Byte width of one table row: PLE_HEAD_DIM/32 * 18 = 5 * 18 = 90.
// Each row is 5 IQ4_NL blocks; each block = 2-byte fp16 scale + 16 code bytes.
// pread() always reads exactly PLE_ROW_BYTES bytes per lookup.
static_assert(PLE_ROW_BYTES == 90,
              "port u89be81de0b: PLE_ROW_BYTES must equal 90");
static_assert(PLE_ROW_BYTES == PLE_HEAD_DIM / 32 * 18,
              "port u89be81de0b: PLE_ROW_BYTES must equal PLE_HEAD_DIM/32*18");

// § 13/25 — const PLE_TABLE_ROWS
// Total rows in the PLE embedding table: 320,001,536.
// File footprint = 320001536 * 90 = 28,800,138,240 bytes (~28.8 GB) on the BI-V100 host.
// All 16 head bands lie within [0, PLE_TABLE_ROWS).
static_assert(PLE_TABLE_ROWS == 320001536ull,
              "port u89be81de0b: PLE_TABLE_ROWS must equal 320001536");
static_assert(PLE_TABLE_ROWS * 90ull == 28800138240ull,
              "port u89be81de0b: table footprint must be 28.8 GB");

// § 14/25 — const TOKEN_NULL
// Sentinel for a missing predecessor token: -1 (int64_t).
// Passed as p_new or p_old to ngram_rows when no predecessor exists in the window.
// Any negative predecessor value triggers the EOS cut, so TOKEN_NULL < 0 is required.
static_assert(TOKEN_NULL == -1,
              "port u89be81de0b: TOKEN_NULL must equal -1");
static_assert(TOKEN_NULL < 0,
              "port u89be81de0b: TOKEN_NULL must be negative (triggers EOS cut)");

// § 15/25 — type PleConsts
// Artifact-time constants extracted from the .ple file header:
//   head_vocab_sizes[16], head_offsets[16], mult[3], table_rows.
// Populated once at open time by ple_artifact_consts(); read-only after that.
static_assert(sizeof(PleConsts::head_vocab_sizes) == 16 * sizeof(uint64_t),
              "port u89be81de0b: PleConsts::head_vocab_sizes must be [16]uint64");
static_assert(sizeof(PleConsts::head_offsets)     == 16 * sizeof(uint64_t),
              "port u89be81de0b: PleConsts::head_offsets must be [16]uint64");
static_assert(sizeof(PleConsts::mult)             ==  3 * sizeof(uint64_t),
              "port u89be81de0b: PleConsts::mult must be [3]uint64");

// § 16/25 — type PleIoOptions
// Open-time options for a PleTable: direct I/O flag, prefetch flag and
// max in-flight slot count.  Passed to PleTable::open(); defaults are
// {direct=false, prefetch=false, max_inflight=1}.
static_assert(sizeof(PleIoOptions) > 0,
              "port u89be81de0b: PleIoOptions must be a complete type");
namespace { constexpr PleIoOptions _opt_default{}; }
static_assert(!_opt_default.direct   && !_opt_default.prefetch &&
              _opt_default.max_inflight == 1,
              "port u89be81de0b: PleIoOptions default: direct=false prefetch=false max_inflight=1");

// § 17/25 — type PleIo
// Open-time validation record: ndims=2, fast-dim=160, row count, file size and
// data-start offset.  PleIo::open() calls ple_io::open_check() internally;
// returns false for wrong ndims, wrong fast dim or size mismatch.
static_assert(sizeof(PleIo) > 0,
              "port u89be81de0b: PleIo must be a complete type");
static_assert(sizeof(PleIo::rows) == sizeof(uint64_t) &&
              sizeof(PleIo::data_start) == sizeof(uint64_t),
              "port u89be81de0b: PleIo::rows and data_start must be uint64_t");

// § 18/25 — type PleTable
// File-backed row table with pread() I/O, optional O_DIRECT and async prefetch.
// Protocol: issue(rows16) → collect(out) with exactly one token in flight.
// Verified via static callable checks: open, close, is_open, rows, gather,
// issue, collect, io_report must all exist as member functions.
static_assert(sizeof(PleTable) > 0,
              "port u89be81de0b: PleTable must be a complete type");
namespace {
  // Verify the core method signatures exist and are callable on a PleTable value.
  // Using decltype + void_t pattern to trigger substitution failure if absent.
  using _PleT = PleTable;
  static_assert(std::is_same_v<decltype(std::declval<_PleT>().is_open()), bool>,
                "port u89be81de0b: PleTable::is_open() must return bool");
  static_assert(std::is_same_v<decltype(std::declval<_PleT>().rows()), uint64_t>,
                "port u89be81de0b: PleTable::rows() must return uint64_t");
  static_assert(std::is_same_v<decltype(std::declval<_PleT>().pending()), bool>,
                "port u89be81de0b: PleTable::pending() must return bool");
}

// § 19/25 — func iq4nl_code
// Returns the signed dequantised integer for a 4-bit IQ4_NL nibble (0-15).
// Maps nibble → kIq4Table[nibble]; the codebook is strictly increasing in [-127, 113].
// Used by iq4nl_dequant_row to decode each nibble before fp16-scaling.
static_assert(std::is_same_v<decltype(iq4nl_code(0)), int>,
              "port u89be81de0b: iq4nl_code must return int");
// kIq4Table is constexpr — verify codebook boundaries used by iq4nl_code.
static_assert(ple_hash::kIq4Table[0]  == -127 && ple_hash::kIq4Table[15] == 113,
              "port u89be81de0b: iq4nl_code codebook: first=-127 last=113");

// § 20/25 — func iq4nl_dequant_row
// Decodes one 90-byte IQ4_NL row to 160 fp32 values by calling dequant_block on
// each of the 5 blocks.  Byte j of a block carries elements j and j+16; low nibble
// maps to element j, high nibble to element j+16 (split-half encoding).
// Signature: (const uint8_t* row, float* out) -> void.
static_assert(
    std::is_same_v<
        decltype(&iq4nl_dequant_row),
        void (*)(const uint8_t*, float*)>,
    "port u89be81de0b: iq4nl_dequant_row must have signature void(const uint8_t*,float*)");

// § 21/25 — func ngram_mixed
// XOR of ctx[i]*mult[i] over i in [0,n) (n ≤ NGRAM_SIZE).
// Equal inputs at positions ≥1 cancel under XOR; depends only on the first n entries.
// Signature: (const int64_t* ctx, int n) -> uint64_t.
static_assert(
    std::is_same_v<
        decltype(&ngram_mixed),
        uint64_t (*)(const int64_t*, int)>,
    "port u89be81de0b: ngram_mixed must have signature uint64_t(const int64_t*,int)");

// § 22/25 — func ngram_rows
// Computes all 16 head row indices for the newest token in `window` (OLDEST FIRST,
// length n ≥ 1; negative or TOKEN_NULL = missing predecessor).  EOS cut propagates
// to older positions; a token's own EOS does not cut its own context.
// Signature: (const int64_t* window, int n, int64_t eos, uint64_t rows[16]) -> void.
static_assert(
    std::is_same_v<
        decltype(&ngram_rows),
        void (*)(const int64_t*, int, int64_t, uint64_t*)>,
    "port u89be81de0b: ngram_rows must have signature void(const int64_t*,int,int64_t,uint64_t*)");

// § 23/25 — func ple_artifact_consts
// Constructs a PleConsts from the compile-time ple_hash tables: fills
// head_vocab_sizes[h]=kVocab[h], head_offsets[h]=kOffset[h], mult[0..2]=kMult{0,1,2},
// table_rows=PLE_TABLE_ROWS.  Called once at open time and stored for the lifetime
// of the PleTable.  Returns PleConsts by value.
static_assert(
    std::is_same_v<
        decltype(&ple_artifact_consts),
        PleConsts (*)()>,
    "port u89be81de0b: ple_artifact_consts must have signature PleConsts()");

// § 24/25 — func ple_prefetch_enable
// Sets the global prefetch flag (std::atomic<bool> g_ple_prefetch).
// When true, PleTable::issue() launches the row-reads asynchronously via std::async
// so the host CPU can proceed while the BI-V100 PCIe DMA is in flight.
// Signature: (bool on) -> void.
static_assert(
    std::is_same_v<
        decltype(&ple_prefetch_enable),
        void (*)(bool)>,
    "port u89be81de0b: ple_prefetch_enable must have signature void(bool)");

// § 25/25 — func ple_prefetch_enabled
// Reads the global prefetch flag (std::atomic<bool> g_ple_prefetch, memory_order
// relaxed load).  Called by PleTable::issue() to decide between synchronous and
// asynchronous row-read dispatch.  Signature: () -> bool.
static_assert(
    std::is_same_v<
        decltype(&ple_prefetch_enabled),
        bool (*)()>,
    "port u89be81de0b: ple_prefetch_enabled must have signature bool()");

}  // namespace baloon


