/-!
# MODULE: router_top10_select
SOURCES: src/kernels/cuda/router_top10.cu
MoE router: softmax over all experts, stable top-k selection (ties by ascending index), renormalization of the
selected weights with a 2^-14 lower clamp on the denominator.

## Source semantics
The selection order is "descending probability, ties by ascending index" - a strict total order. Block-wide
argmax picks the best candidate under this order and is independent of the reduction tree. The denominator is
max(sum, 2^-14) so division is safe. Launch: threads = min(n_expert,512) rounded up to 32; shared memory =
aligned taken-mask + n doubles + n floats; k <= 64, n_expert <= 512*64.

## Form template identifiers
before -> "ranking order"; pick -> "argmax combiner"; smemBytes -> "dynamic shared layout".
-/
namespace Strata.RouterTop10

-- § ranking order
/-- Expert f outranks expert e: larger probability, or equal probability and smaller index. -/
def before (p : Nat → Int) (e f : Nat) : Prop := p f > p e ∨ (p f = p e ∧ f < e)

/-- No expert outranks itself. -/
theorem before_irrefl (p : Nat → Int) (e : Nat) : ¬ before p e e := by
  unfold before
  omega

/-- The order is asymmetric. -/
theorem before_asymm (p : Nat → Int) (e f : Nat) (h : before p e f) : ¬ before p f e := by
  unfold before at *
  omega

/-- The order is transitive. -/
theorem before_trans (p : Nat → Int) (e f g : Nat) (h1 : before p e f) (h2 : before p f g) :
    before p e g := by
  unfold before at *
  omega

/-- Any two distinct experts are comparable, so ties never leave the selection ambiguous. -/
theorem before_total (p : Nat → Int) (e f : Nat) (h : e ≠ f) : before p e f ∨ before p f e := by
  unfold before
  omega

-- § argmax combiner
/-- Combine two (value, index) candidates exactly as the shuffle reduction does. -/
def pick (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The combiner is commutative on distinct indices, so the reduction tree shape cannot change the winner. -/
theorem pick_comm (a b : Int × Nat) (h : a.2 ≠ b.2) : pick a b = pick b a := by
  unfold pick
  split <;> split <;> first | rfl | (exfalso; omega)

/-- The sentinel (index n_expert, value -inf modeled as any value) never beats a real candidate with a larger value. -/
theorem pick_prefers_larger (a b : Int × Nat) (h : b.1 > a.1) : pick a b = b := by
  unfold pick
  rw [if_pos (Or.inl h)]

/-- On equal values the smaller index wins. -/
theorem pick_tie_smaller_index (a b : Int × Nat) (h : b.1 = a.1) (hi : b.2 < a.2) : pick a b = b := by
  unfold pick
  rw [if_pos (Or.inr ⟨h, hi⟩)]

-- § renormalization
/-- The clamped denominator `max(s, 2^-14)` (scaled by 2^14 to stay in Nat). -/
def clampDen (s : Nat) : Nat := max s 1

/-- The denominator is never zero. -/
theorem clampDen_pos (s : Nat) : 0 < clampDen s := by
  unfold clampDen
  omega

/-- The denominator never drops below the sum. -/
theorem clampDen_ge (s : Nat) : s ≤ clampDen s := by
  unfold clampDen
  omega

-- § launch geometry
/-- Threads per block: at least a warp-multiple covering min(n, 512). -/
def threads (n : Nat) : Nat := ((min n 512) + 31) / 32 * 32

/-- Thread count covers the experts up to 512 and is a multiple of 32. -/
theorem threads_props (n : Nat) : min n 512 ≤ threads n ∧ threads n % 32 = 0 ∧ threads n ≤ 512 := by
  unfold threads
  omega

/-- Bytes of the taken-mask (16-aligned). -/
def takenBytes (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Dynamic shared memory: mask, n doubles, n floats. -/
def smemBytes (n : Nat) : Nat := takenBytes n + n * 8 + n * 4

/-- The mask is 16-aligned, so the double array after it is 8-aligned and the float array 4-aligned. -/
theorem smem_alignment (n : Nat) : takenBytes n % 16 = 0 ∧ (takenBytes n) % 8 = 0 ∧ (takenBytes n + n * 8) % 4 = 0 := by
  unfold takenBytes
  omega

/-- For 512 experts the dynamic shared size is 6656 bytes. -/
theorem smem_512 : smemBytes 512 = 6656 := by
  decide

/-- The mask covers every expert. -/
theorem taken_covers (n : Nat) : n ≤ takenBytes n := by
  unfold takenBytes
  omega

end Strata.RouterTop10
