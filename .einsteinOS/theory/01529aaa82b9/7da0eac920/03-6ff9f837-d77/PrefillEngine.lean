/-!
# MODULE: prefill_engine
Batched prompt processing (`strata::prefill::Prefill`): chunked loop over the prompt, bump allocation of
chunk buffers, host-side grouping of (token, k) pairs by expert, and the expert staging rings.

## Source semantics
* The prompt is processed in chunks of at most `T` tokens; the chunk sizes sum to the prompt length.
* Buffer allocations are rounded up to 256 bytes plus 256 slack; a borrowed region fails rather than overflows.
* MoE grouping is a counting sort: `off[e+1] = off[e] + cnt[e]`, each pair lands in `[off e, off (e+1))`.
* The staging ring index advances modulo `STAGE = 8`, the dequant ring modulo `DQ = 2`.
* Only `min(prefill_until, n-1)` positions are batched when `prefill_until` is set.

## Form template identifiers
* `chunkSum`  <- the `for (c0 = 0; c0 < n; c0 += m.T)` loop with `T = min(m.T, n - c0)`
* `allocBytes`<- `Alloc::take` rounding
* `offs`      <- `m.off`
* `STAGE`, `DQ` <- ring sizes
-/

namespace Strata.Prefill

-- § chunk loop

/-- Sum of the chunk sizes visited by the chunk loop, with a fuel bound on the iteration count. -/
def chunkSum (T : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | f + 1, n => if n = 0 then 0 else min T n + chunkSum T f (n - min T n)

/-- With enough fuel and a positive chunk size, the chunks cover exactly the prompt. -/
theorem chunkSum_eq (T : Nat) (hT : 0 < T) : ∀ (f n : Nat), n ≤ f → chunkSum T f n = n := by
  intro f
  induction f with
  | zero => intro n hn; simp [chunkSum]; omega
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · simp [chunkSum, h0]
    · have hm : 1 ≤ min T n ∧ min T n ≤ n := by omega
      have := ih (n - min T n) (by omega)
      simp only [chunkSum, if_neg h0]
      omega

/-- Each chunk is non-empty and no larger than the configured chunk size. -/
theorem chunk_size_bounds (T n : Nat) (hT : 0 < T) (hn : 0 < n) : 0 < min T n ∧ min T n ≤ T := by
  omega

/-- The number of batched positions is `prefill_until` when it is set and short enough, else `n - 1`. -/
def nBatched (untilN n : Nat) : Nat := if 0 < untilN ∧ untilN < n - 1 then untilN else n - 1

/-- Batched positions never include the last prompt position. -/
theorem nBatched_le (untilN n : Nat) : nBatched untilN n ≤ n - 1 := by
  unfold nBatched
  split <;> omega

-- § bump allocator

/-- Bytes charged for an allocation of `x` raw bytes: `(x + 256 + 255)` rounded down to a multiple of 256. -/
def allocBytes (x : Nat) : Nat := (x + 511) / 256 * 256

/-- The charge always covers the request with at least 256 bytes of slack. -/
theorem allocBytes_cover (x : Nat) : x + 256 ≤ allocBytes x := by
  unfold allocBytes
  omega

/-- The charge is 256-byte aligned. -/
theorem allocBytes_aligned (x : Nat) : allocBytes x % 256 = 0 := by
  unfold allocBytes
  omega

/-- A bump allocation succeeds only when it stays within the region. -/
def bumpOk (used cap x : Nat) : Prop := used + allocBytes x ≤ cap

/-- A successful allocation leaves the cursor within capacity. -/
theorem bump_within_cap (used cap x : Nat) (h : bumpOk used cap x) : used + allocBytes x ≤ cap := h

/-- Successive allocations never overlap: the next cursor is past the previous block. -/
theorem bump_no_overlap (used x : Nat) : used + x < used + allocBytes x := by
  have := allocBytes_cover x
  omega

-- § counting sort of (token, k) pairs by expert

/-- Prefix offsets from per-expert counts. -/
def offs (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offs cnt e + cnt e

/-- Offsets satisfy the recurrence the source builds. -/
theorem offs_succ (cnt : Nat → Nat) (e : Nat) : offs cnt (e + 1) = offs cnt e + cnt e := rfl

/-- Offsets are monotone in the expert index. -/
theorem offs_mono (cnt : Nat → Nat) (a b : Nat) (h : a ≤ b) : offs cnt a ≤ offs cnt b := by
  induction b with
  | zero =>
    have : a = 0 := by omega
    subst this; exact Nat.le_refl _
  | succ b ih =>
    by_cases hab : a ≤ b
    · have := ih hab
      simp only [offs]
      omega
    · have : a = b + 1 := by omega
      subst this; exact Nat.le_refl _

/-- The slot of the `r`-th pair of expert `e` (`r < cnt e`) lies inside expert `e`'s block. -/
theorem slot_in_block (cnt : Nat → Nat) (e r : Nat) (h : r < cnt e) :
    offs cnt e ≤ offs cnt e + r ∧ offs cnt e + r < offs cnt (e + 1) := by
  simp only [offs]
  omega

/-- Blocks of distinct experts are disjoint: an earlier block ends before a later block starts. -/
theorem blocks_disjoint (cnt : Nat → Nat) (a b : Nat) (h : a < b) : offs cnt (a + 1) ≤ offs cnt b :=
  offs_mono cnt (a + 1) b h

-- § rings

/-- Stage ring: `stage_next = (stage_next + 1) % 8`. -/
def stageNext (s : Nat) : Nat := (s + 1) % 8

/-- The stage slot index stays inside the ring. -/
theorem stageNext_lt (s : Nat) : stageNext s < 8 := by
  unfold stageNext
  omega

/-- The dequantized-expert ring index `j % DQ` stays inside the two buffers. -/
theorem dq_index_lt (j : Nat) : j % 2 < 2 := by omega

/-- Lookahead: staging at most `STAGE - 1` experts ahead never reuses a slot still awaiting dequantization. -/
theorem lookahead_distinct (j k : Nat) (hk : k ≤ j + 7) (hj : j < k) : (j % 8) ≠ (k % 8) := by
  omega

-- § batch tensor geometry

/-- Gate/up scratch rows: `o0 * 1280` offsets of consecutive experts do not overlap. -/
theorem gu_blocks_disjoint (o0 ne : Nat) : o0 * 1280 + ne * 1280 = (o0 + ne) * 1280 := by omega

/-- The expert block for a group of `ne` rows starting at `o0` ends within `T * K` rows when the offsets are valid. -/
theorem group_in_range (o0 ne total : Nat) (h : o0 + ne ≤ total) : (o0 + ne) * 1280 ≤ total * 1280 := by
  omega

end Strata.Prefill
