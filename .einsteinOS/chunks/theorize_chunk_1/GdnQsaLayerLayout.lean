/-!
# MODULE: gdn_qsa_layer_layout
Buffer-carving arithmetic and stage selection of the layer composition (`layer.cpp`).

## Source semantics
Q8_K buffers are `n/256` blocks of 292 bytes. Every carved region is 16-byte aligned (`(v+15)&~15`), so the total
is at least the sum of the parts. A paged KV state with page size `p` holds `ceil(max_cells/p)*p >= max_cells` cells.
RoPE row `pos_base+pos` must be below `max_cells`. `block_layer_pre` runs stage `i` iff the stage prefix covers it, or
(prefix 0) according to `half`: stages 0-2 for halves 0/1, stages 3-4 for halves 0/2.

## Form template identifiers
- q8k_bytes            -> q8kBytes
- align16 / Cursor     -> align16 / takeUsed
- qsa_state_bytes      -> ceilDivMul
- block_layer_pre run0..run4 -> runStage
-/
namespace Strata.GdnQsaLayerLayout

def q8kBytes (n : Nat) : Nat := n / 256 * 292
def align16 (v : Nat) : Nat := (v + 15) / 16 * 16

/-- Q8_K blocks carry 256 elements, so `n` a multiple of 256 is exactly `n/256` blocks. -/
theorem q8k_blocks (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by omega

/-- Alignment never shrinks a region. -/
theorem align16_ge (v : Nat) : v ≤ align16 v := by unfold align16; omega
/-- Aligned regions are multiples of 16. -/
theorem align16_mod (v : Nat) : align16 v % 16 = 0 := by unfold align16; omega
/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (v : Nat) : align16 v < v + 16 := by unfold align16; omega

/-- The carved total covers the sum of all parts. -/
theorem carve_total_ge : ∀ ps : List Nat, ps.sum ≤ (ps.map align16).sum
  | [] => by simp
  | p :: ps => by
    have h := carve_total_ge ps
    have := align16_ge p
    simp only [List.map_cons, List.sum_cons]
    omega

/-- A cursor `take` advances by at least the requested bytes and keeps 16-byte alignment. -/
theorem take_advances (used count sz : Nat) :
    used + count * sz ≤ align16 (used + count * sz) ∧ align16 (used + count * sz) % 16 = 0 :=
  ⟨align16_ge _, align16_mod _⟩

/-- Pages needed to hold `a` cells never undercount. -/
theorem ceilDivMul (a b : Nat) (hb : 0 < b) : a ≤ b * ((a + b - 1) / b) := by
  have h := Nat.div_add_mod (a + b - 1) b
  have h2 := Nat.mod_lt (a + b - 1) hb
  omega

/-- Row index `pos_base + pos < max_cells` keeps every RoPE-table access (`row*h + j`, `j < h`) in range. -/
theorem rope_index_in_table {row mc h j : Nat} (hr : row < mc) (hj : j < h) : row * h + j < mc * h := by
  have := Nat.mul_le_mul_right h (show row + 1 ≤ mc by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- Stage `i` (0..4) runs under a positive stage prefix `k` iff `k ≥ i+1`; with prefix 0 it follows `half`. -/
def runStage (pfx half i : Nat) : Bool :=
  if pfx > 0 then decide (pfx ≥ i + 1)
  else if i < 3 then decide (half = 0 ∨ half = 1) else decide (half = 0 ∨ half = 2)

/-- `half = 0` with no prefix runs the whole block. -/
theorem runStage_all (i : Nat) : runStage 0 0 i = true := by
  unfold runStage
  split <;> simp_all

/-- The full prefix 5 runs every stage. -/
theorem runStage_prefix5 (i : Nat) (h : i < 5) : runStage 5 0 i = true := by
  unfold runStage
  simp
  omega

/-- A positive prefix is downward closed: if stage `j` runs, every earlier stage runs. -/
theorem runStage_prefix_closed {k i j : Nat} (hk : 0 < k) (hij : i ≤ j) (h : runStage k 0 j = true) :
    runStage k 0 i = true := by
  unfold runStage at h ⊢
  rw [if_pos hk] at h ⊢
  simp only [decide_eq_true_eq] at h ⊢
  omega

/-- Half 2 skips the mixer stages 0-2. -/
theorem runStage_half2_skips_mixer (i : Nat) (h : i < 3) : runStage 0 2 i = false := by
  unfold runStage
  simp
  omega

/-- Half 1 skips the FFN-front and router stages 3-4. -/
theorem runStage_half1_skips_router (i : Nat) (h1 : 3 ≤ i) : runStage 0 1 i = false := by
  unfold runStage
  simp
  omega

/-- The dump region of one layer ends inside the whole dump buffer. -/
theorem dump_slot_in_buffer {l nl n : Nat} (h : l < nl) : (l + 1) * n + n ≤ (nl + 1) * n := by
  have h1 : (l + 2) * n ≤ (nl + 1) * n := Nat.mul_le_mul_right n (by omega)
  have h2 : (l + 2) * n = (l + 1) * n + n := by ring
  omega

end Strata.GdnQsaLayerLayout
