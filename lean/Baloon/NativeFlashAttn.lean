-- Baloon.NativeFlashAttn: independent proofs for the short-context flash-attention decode kernel.
-- Specialised to Q 24x256 / KV 2x256 fp16, 128 threads, 1 block per query head, two 128-cell passes.

namespace Baloon.NativeFlashAttn

/-- Step predicate: width in [1, maxCtx], n_kv = width, pos = width-1, n_bid = width/4. -/
structure StepOk (width nkv pos nbid maxCtx : Nat) : Prop where
  lo : 1 <= width
  hi : width <= maxCtx
  kv_eq : nkv = width
  pos_eq : pos + 1 = width
  bid_eq : nbid = width / 4

/-- A step filled from pos passes StepOk whenever pos+1 <= maxCtx. -/
theorem stepOk_of_pos (pos maxCtx : Nat) (h : pos + 1 <= maxCtx) :
    StepOk (pos + 1) (pos + 1) pos ((pos + 1) / 4) maxCtx :=
  { lo := by omega, hi := h, kv_eq := rfl, pos_eq := rfl, bid_eq := rfl }

/-- Short contexts (n <= 256) see all cells: min n 2051 = n. -/
theorem shortCtxIdentity (n : Nat) (hn : n <= 256) : min n 2051 = n := by omega

/-- GQA: query head h maps to kv head h/12; result stays below 2 for 24 heads. -/
theorem kvHeadLt (head : Nat) (hh : head < 24) : head / 12 < 2 := by omega

/-- Every cell index < 256 decomposes into (pass, group, row-bit, row) uniquely. -/
theorem cellDecomp (cell : Nat) (hc : cell < 256) :
    cell = (cell / 128) * 128 + ((cell % 128) / 32) * 32 + ((cell % 32) / 8) * 8 + cell % 8 /\
    cell / 128 < 2 /\ (cell % 128) / 32 < 4 /\ cell % 8 < 8 := by omega

/-- Thread with lane t keeps its own cell at warp*32 + t. -/
theorem laneKeepsCell (warp lane row : Nat) (hl : lane < 32) (hr : row = lane % 8) :
    warp * 32 + (lane / 8) * 8 + row = warp * 32 + lane := by omega

/-- Two 128-wide passes cover all 256 output dimensions. -/
theorem outputCover (d : Nat) (hd : d < 256) :
    d = (d / 128) * 128 + d % 128 /\ d / 128 < 2 /\ d % 128 < 128 := by omega

/-- Online-softmax rescale exponent: running_max - new_max <= 0. -/
theorem rescaleNonpos (m x : Int) : m - max m x <= 0 := by omega

/-- Running maximum is non-decreasing. -/
theorem maxNonDecr (m x : Int) : m <= max m x := by omega

/-- Accumulator staging index (warp, lane, i0, j) stays inside 4*4*256 shared tile. -/
theorem stageIdx (warp lane i0 j : Nat) (hw : warp < 4) (hl : lane < 32)
    (hi : i0 = 0 \/ i0 = 32 \/ i0 = 64 \/ i0 = 96) (hj : j < 4) :
    warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j + 1 < 4 * 4 * 256 := by
  omega

/-- Shared memory: tile (4*4*256) + two 32-float reductions fits 48 KiB. -/
theorem smemBound : (4 * 4 * 256 + 32 + 32) * 4 <= 49152 := by decide

/-- Scale factor 1/16: 16*16 = 256 so scale = 1/sqrt(256). -/
theorem scaleCheck : 16 * 16 = (256 : Nat) := by decide

/-- KV layout: each cell uses 2 heads * 256 dims * 2 bytes = 1024 bytes. -/
theorem kvBytesPerCell : 2 * 256 * 2 = (1024 : Nat) := by decide

/-- KV element index (cell*2+kv)*256+d stays inside cap cells. -/
theorem kvElemInRange (cell kv d cap : Nat) (hc : cell < cap) (hk : kv < 2) (hd : d < 256) :
    (cell * 2 + kv) * 256 + d < cap * 2 * 256 := by
  have h1 : (cell + 1) * 2 <= cap * 2 := Nat.mul_le_mul_right 2 (by omega)
  have h2 : (cell * 2 + kv + 1) * 256 <= cap * 2 * 256 := Nat.mul_le_mul_right 256 (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1 h2; omega

end Baloon.NativeFlashAttn
