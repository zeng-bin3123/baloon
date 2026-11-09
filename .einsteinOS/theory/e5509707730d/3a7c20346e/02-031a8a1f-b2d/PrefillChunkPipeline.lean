/-!
# MODULE: prefill_chunk_pipeline
SOURCES: src/prefill/prefill.cpp
Chunked prefill: arena sizing by a counting pass, the per-chunk loop, host-side grouping of (token, k) pairs
by expert, the expert staging ring and the dequantized-expert ring.

## Source semantics
* `Alloc::take` pads each region to `bytes + 256` rounded up to 256; the count-only pass sizes the borrow.
* Chunks cover `[0, n)` in steps of `T`; each chunk has `min(T, n - c0) >= 1` tokens.
* Experts are grouped: `off[e+1] = off[e] + cnt[e]`, slot of the j-th pair of expert `e` is `off[e] + j`,
  source token of pair `i` is `i / K`.
* Staging ring has 8 slots with lookahead 7; dequant ring has 2 slots.

## Form template identifiers
takeBytes ↦ `Alloc::take` rounding; offs ↦ `m.off`; stage slot ↦ `stage_next`; q ↦ `j % DQ`.
-/
namespace Strata.PrefillChunkPipeline

-- § allocator

/-- Padded size of one region of `bytes` payload. -/
def takeBytes (bytes : Nat) : Nat := (bytes + 256 + 255) / 256 * 256

/-- Each region carries at least its payload plus 256 slack. -/
theorem takeBytes_ge (b : Nat) : b + 256 ≤ takeBytes b := by unfold takeBytes; omega

/-- Regions are 256-multiples. -/
theorem takeBytes_mod (b : Nat) : takeBytes b % 256 = 0 := by unfold takeBytes; omega

/-- Total used after a list of regions. -/
def used : List Nat → Nat → Nat
  | [], s => s
  | b :: bs, s => used bs (s + takeBytes b)

/-- Usage is monotone. -/
theorem used_ge : ∀ (bs : List Nat) (s : Nat), s ≤ used bs s
  | [], s => Nat.le_refl s
  | b :: bs, s => by
      rw [used]
      have h := used_ge bs (s + takeBytes b)
      omega

/-- Usage splits over concatenation. -/
theorem used_append : ∀ (xs ys : List Nat) (s : Nat), used (xs ++ ys) s = used ys (used xs s)
  | [], ys, s => by simp [used]
  | x :: xs, ys, s => by
      simp only [List.cons_append, used]
      exact used_append xs ys _

/-- If the counted total fits the borrowed capacity, every region fits when taken (no `ok = false`). -/
theorem take_fits (pre post : List Nat) (x cap : Nat) (h : used (pre ++ x :: post) 0 ≤ cap) :
    used pre 0 + takeBytes x ≤ cap := by
  rw [used_append] at h
  have h2 := used_ge post (used pre 0 + takeBytes x)
  simp only [used] at h
  omega

-- § chunk loop

/-- Tokens in the chunk starting at `c0`. -/
def chunkTokens (T n c0 : Nat) : Nat := min T (n - c0)

/-- Every chunk is non-empty and stays inside `[0, n)`. -/
theorem chunk_bounds (T n c0 : Nat) (hT : 0 < T) (h : c0 < n) :
    0 < chunkTokens T n c0 ∧ c0 + chunkTokens T n c0 ≤ n := by
  unfold chunkTokens; omega

/-- Every pair's source token `i / K` is a valid token of the chunk. -/
theorem src_token_lt (i T K : Nat) (hK : 0 < K) (h : i < T * K) : i / K < T := by
  rw [Nat.div_lt_iff_lt_mul hK]
  exact h

-- § grouping by expert

/-- Prefix offsets of the per-expert counts. -/
def offs (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offs cnt e + cnt e

/-- Prefix offsets are monotone. -/
theorem offs_mono (cnt : Nat → Nat) {a b : Nat} (h : a ≤ b) : offs cnt a ≤ offs cnt b := by
  induction b with
  | zero =>
      have : a = 0 := by omega
      subst this
      exact Nat.le_refl _
  | succ b ih =>
      rcases Nat.lt_or_ge a (b + 1) with h1 | h1
      · have := ih (by omega)
        simp only [offs]
        omega
      · have : a = b + 1 := by omega
        subst this
        exact Nat.le_refl _

/-- Expert `e`'s slot range ends before any later expert's begins: slot ranges are disjoint. -/
theorem slots_disjoint (cnt : Nat → Nat) (e e' : Nat) (h : e < e') : offs cnt e + cnt e ≤ offs cnt e' := by
  have := offs_mono cnt (show e + 1 ≤ e' by omega)
  simpa [offs] using this

/-- The j-th pair of expert `e` gets a slot inside its own range. -/
theorem slot_in_range (cnt : Nat → Nat) (e j : Nat) (h : j < cnt e) :
    offs cnt e ≤ offs cnt e + j ∧ offs cnt e + j < offs cnt (e + 1) := by
  simp only [offs]
  omega

-- § staging ring

/-- Staging slot reuse is safe: blob `i` (staged while computing `j`, so `i ≤ j + 7`) reuses the slot of
blob `i - 8`, which is strictly earlier than `j` and therefore already dequantized. -/
theorem stage_slot_reuse_safe (i j : Nat) (h : i ≤ j + 7) (h8 : 8 ≤ i) :
    i - 8 < j + 0 ∧ (i - 8) % 8 = i % 8 := by omega

/-- The slot index is always a valid ring slot. -/
theorem stage_slot_lt (n : Nat) : n % 8 < 8 := Nat.mod_lt _ (by omega)

/-- The dequant ring alternates between two buffers: consecutive experts use different ones. -/
theorem dq_alternates (j : Nat) : (j + 1) % 2 ≠ j % 2 := by omega

/-- Lookahead never stages beyond the order list: `staged ≤ j + 7` while `staged < len`. -/
theorem lookahead_bounded (staged j len : Nat) (h1 : staged < len) (h2 : staged ≤ j + 7) :
    staged < len ∧ staged - j ≤ 7 := by omega

end Strata.PrefillChunkPipeline
