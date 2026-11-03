/-!
# MODULE: prefill_chunk_moe
Chunking, allocation rounding, expert-ring rotation and the host counting sort that groups (token, k) pairs by expert.

## Source semantics
Prompt tokens are processed in chunks of at most `m.T`. Every device allocation is rounded up to 256 bytes with 256
bytes of padding. Experts stream through `STAGE = 8` staging slots with lookahead `STAGE - 1` and dequantize through
`DQ = 2` rings. Pairs `i ∈ [0, T*K)` map to source token `i / K`, and the expert offsets are prefix sums of counts.

## Form template identifiers
- allocBytes ← `Alloc::take` byte count, chunkLen ← `std::min(m.T, n - c0)`,
- nextSlot ← `stage_next`, dqSlot ← `j % DQ`, srcToken ← `i / K`.
-/

namespace Strata.Prefill

-- § allocation

/-- Bytes charged for one allocation of `b = n * sizeof(T)`: round `b + 256` up to 256. -/
def allocBytes (b : Nat) : Nat := (b + 256 + 255) / 256 * 256

/-- An allocation holds its request. -/
theorem allocBytes_ge (b : Nat) : b ≤ allocBytes b := by
  unfold allocBytes; omega

/-- Every allocation leaves at least 256 bytes of padding. -/
theorem allocBytes_padded (b : Nat) : b + 256 ≤ allocBytes b := by
  unfold allocBytes; omega

/-- Allocation sizes are multiples of 256. -/
theorem allocBytes_mod (b : Nat) : allocBytes b % 256 = 0 := by
  unfold allocBytes; omega

/-- Allocations waste fewer than 512 bytes. -/
theorem allocBytes_lt (b : Nat) : allocBytes b < b + 512 := by
  unfold allocBytes; omega

/-- Allocation size is monotone, so a larger request never charges less. -/
theorem allocBytes_mono {a b : Nat} (h : a ≤ b) : allocBytes a ≤ allocBytes b := by
  unfold allocBytes; omega

-- § chunking

/-- Length of the chunk starting at `c0`. -/
def chunkLen (chunk n c0 : Nat) : Nat := min chunk (n - c0)

/-- A chunk is nonempty while tokens remain. -/
theorem chunkLen_pos (chunk n c0 : Nat) (h : 1 ≤ chunk) (hc : c0 < n) : 0 < chunkLen chunk n c0 := by
  unfold chunkLen; omega

/-- A chunk fits the preallocated buffers. -/
theorem chunkLen_le (chunk n c0 : Nat) : chunkLen chunk n c0 ≤ chunk := by
  unfold chunkLen; omega

/-- The next chunk start is within or at the end of the prompt. -/
theorem chunk_next_le (chunk n c0 : Nat) (hc : c0 < n) : c0 + chunkLen chunk n c0 ≤ n := by
  unfold chunkLen; omega

/-- Selection batching: a sub-batch starting at `t0 < T` stays inside the chunk. -/
theorem sub_batch_within (batch T t0 : Nat) (h : t0 < T) : t0 + min batch (T - t0) ≤ T := by
  omega

-- § expert staging rings

/-- Staging ring advance. -/
def nextSlot (s : Nat) : Nat := (s + 1) % 8

/-- The staging slot stays in range. -/
theorem nextSlot_lt (s : Nat) : nextSlot s < 8 := by
  unfold nextSlot; omega

/-- The dequantized-expert ring slot of the `j`-th expert. -/
def dqSlot (j : Nat) : Nat := j % 2

/-- The dequant slot stays in range. -/
theorem dqSlot_lt (j : Nat) : dqSlot j < 2 := by
  unfold dqSlot; omega

/-- Consecutive experts use different dequant slots, so one can dequantize while the other multiplies. -/
theorem dqSlot_alternates (j : Nat) : dqSlot j ≠ dqSlot (j + 1) := by
  unfold dqSlot; omega

/-- Lookahead `STAGE - 1 = 7` keeps the staged-but-unused experts below the ring size. -/
theorem lookahead_fits (j staged : Nat) (h : staged ≤ j + 7) (hj : j < staged ∨ j = staged) :
    staged - j ≤ 7 := by
  omega

-- § counting sort

/-- Source token of the routed pair `i` (K = 10 experts per token). -/
def srcToken (i : Nat) : Nat := i / 10

/-- Pairs of a chunk of `T` tokens map to tokens inside the chunk. -/
theorem srcToken_lt (T i : Nat) (h : i < T * 10) : srcToken i < T := by
  unfold srcToken; omega

/-- Pair `10*t + j` (j < 10) maps back to token `t`. -/
theorem srcToken_pair (t j : Nat) (hj : j < 10) : srcToken (10 * t + j) = t := by
  unfold srcToken; omega

/-- Prefix-sum offsets: the next expert's offset is this offset plus this expert's count. -/
def nextOff (off cnt : Nat) : Nat := off + cnt

/-- An expert's slot range `[off, off+cnt)` ends where the next begins. -/
theorem slots_contiguous (off cnt : Nat) : nextOff off cnt - off = cnt := by
  unfold nextOff; omega

/-- A slot filled from an expert's cursor stays inside its range. -/
theorem slot_in_range (off cnt filled : Nat) (h : filled < cnt) : off + filled < nextOff off cnt := by
  unfold nextOff; omega

end Strata.Prefill
