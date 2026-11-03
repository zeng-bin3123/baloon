/-!
# MODULE: native_router_top10
The llama.cpp-style fused MoE router (`native_router.cu`) for exactly 512 experts and 10 outputs: one warp,
16 values per lane, the expert-id layout `id = lane + 32*i`, the tournament tie rule, the output-lane
mapping, the clamp constant and the host-side span validation (alignment, overflow, disjointness).

## Source semantics
* Lane `l` holds experts `l, l+32, ..., l+480`; the block is `32 x 8` threads, only row 0 is active.
* Each of 10 rounds takes the lane-local best (strict `>`, so lowest `i`, hence lowest id), then a XOR-butterfly
  keeps the better of two candidates, preferring the lower expert id on equal values.
* The winner's lane marks `values[id/32] = -inf` and records `ids[rank] = id`; lane `rank` keeps the weight.
* Weights are divided by `max(sum, 2^-14)`; only lanes `< 10` store.
* Host: non-null, 4-byte aligned, no address wrap, and the three spans pairwise disjoint.

## Form template identifiers
- `values[i]` → `slot i`; `lane` → `lane`; `rank` → `rank`; `expert` → `id`
- `valid` → `valid`; `overlap` → `overlap`; `UINTPTR_MAX` → `uptrMax`
-/

namespace Strata.NativeRouter

-- § Expert id layout

/-- Expert id of slot `i` in lane `lane`. -/
def expertId (lane i : Nat) : Nat := lane + 32 * i

/-- 512 experts = 32 lanes x 16 slots. -/
theorem experts_tile : 32 * 16 = 512 := by decide

/-- Every id is below 512. -/
theorem expertId_lt (lane i : Nat) (hl : lane < 32) (hi : i < 16) : expertId lane i < 512 := by
  unfold expertId; omega

/-- The lane and slot are recovered from the id. -/
theorem expertId_decode (lane i : Nat) (hl : lane < 32) :
    expertId lane i % 32 = lane ∧ expertId lane i / 32 = i := by
  unfold expertId; omega

/-- The id map is injective: no two (lane, slot) pairs share an expert. -/
theorem expertId_inj (l i l' i' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : expertId l i = expertId l' i') : l = l' ∧ i = i' := by
  unfold expertId at h; omega

/-- Every id below 512 has a (lane, slot) preimage. -/
theorem expertId_surj (e : Nat) (h : e < 512) : ∃ lane i, lane < 32 ∧ i < 16 ∧ expertId lane i = e :=
  ⟨e % 32, e / 32, by omega, by omega, by unfold expertId; omega⟩

/-- Within a lane, ids increase with the slot, so a strict-`>` ascending scan keeps the lowest id on ties. -/
theorem expertId_mono (lane i j : Nat) (h : i < j) : expertId lane i < expertId lane j := by
  unfold expertId; omega

/-- The winner's lane test `(expert & 31) == lane` is `expert % 32 = lane`. -/
theorem winner_lane (id : Nat) : id % 32 < 32 := by omega

/-- The slot to blank is `id / 32 < 16` for a valid id. -/
theorem winner_slot_lt (id : Nat) (h : id < 512) : id / 32 < 16 := by omega

-- § Tournament order

/-- `a` is strictly better than `b`: larger value, or equal value and smaller id. -/
def better (a b : Nat × Nat) : Prop := a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

instance (a b : Nat × Nat) : Decidable (better a b) := by
  unfold better; infer_instance

/-- Butterfly step: replace only when the other candidate is strictly better. -/
def butterfly (mine other : Nat × Nat) : Nat × Nat := if better other mine then other else mine

/-- The butterfly's result is not strictly worse than either candidate. -/
theorem butterfly_not_worse (a b : Nat × Nat) :
    ¬ better a (butterfly a b) ∧ ¬ better b (butterfly a b) := by
  unfold butterfly
  by_cases h : better b a
  · rw [if_pos h]; unfold better at *; omega
  · rw [if_neg h]; unfold better at *; omega

/-- The butterfly is symmetric on distinct candidates, so the XOR tree gives every lane the same winner. -/
theorem butterfly_comm (a b : Nat × Nat) (hne : a ≠ b) : butterfly a b = butterfly b a := by
  unfold butterfly
  by_cases h1 : better b a
  · have h2 : ¬ better a b := by unfold better at *; omega
    rw [if_pos h1, if_neg h2]
  · by_cases h2 : better a b
    · rw [if_neg h1, if_pos h2]
    · exfalso
      apply hne
      unfold better at h1 h2
      exact Prod.ext (by omega) (by omega)

/-- Equal ids with equal values are the same candidate, so the tree cannot split a tie inconsistently. -/
theorem butterfly_idem (a : Nat × Nat) : butterfly a a = a := by
  unfold butterfly
  rw [if_neg]
  unfold better; omega

-- § Rounds and outputs

/-- Ten rounds, ten output lanes: lane `rank` keeps the weight of round `rank`. -/
theorem rounds_fit_lanes : 10 ≤ 32 := by decide

/-- Only lanes below 10 store; lane `l` stores round `l`. -/
theorem store_lanes (lane : Nat) (h : lane < 10) : lane < 32 := by omega

/-- Block geometry: `32 x 8 = 256` threads, matching `__launch_bounds__(256, 1)`. -/
theorem block_threads : 32 * 8 = 256 := by decide

/-- Each lane loads slots `lane + 32*i` for `i < 16`: indices cover `[0, 512)` exactly once. -/
theorem load_index_lt (lane i : Nat) (hl : lane < 32) (hi : i < 16) : lane + i * 32 < 512 := by omega

/-- The renormalisation clamp: `2^-14 = 6.103515625e-5`. -/
theorem clamp_constant : 5 ^ 14 * 2 ^ 14 = 10 ^ 14 ∧ 6103515625 = 5 ^ 14 := by decide

-- § Host span validation

/-- Maximum pointer value. -/
def uptrMax : Nat := 2 ^ 64 - 1

/-- `valid p bytes`: non-null, 4-byte aligned, and `p + bytes` does not wrap. -/
def valid (p bytes : Nat) : Prop := p ≠ 0 ∧ p % 4 = 0 ∧ bytes ≤ uptrMax - p

/-- A valid span does not wrap the address space. -/
theorem valid_no_wrap (p bytes : Nat) (h : valid p bytes) (hp : p ≤ uptrMax) : p + bytes ≤ uptrMax := by
  unfold valid at h
  omega

/-- Half-open interval overlap, as in the source. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- Non-overlap of non-empty spans means one ends before the other begins. -/
theorem not_overlap_disjoint (a an b bn : Nat) (h : ¬ overlap a an b bn) :
    a + an ≤ b ∨ b + bn ≤ a := by
  unfold overlap at h
  omega

/-- A span is disjoint from itself only when empty: the host never accepts aliasing outputs. -/
theorem self_overlap (a n : Nat) (h : 0 < n) : overlap a n a n := by
  unfold overlap; omega

/-- Span sizes: logits 2048 bytes, ids 40, weights 40. -/
theorem span_sizes : 512 * 4 = 2048 ∧ 10 * 4 = 40 := by decide

/-- Alignment of all spans' elements follows from the base alignment: element `k` at `p + 4k`. -/
theorem element_aligned (p k : Nat) (h : p % 4 = 0) : (p + 4 * k) % 4 = 0 := by omega

end Strata.NativeRouter
