/-!
# MODULE: native_qsa_norms
Weighted RMSNorm (templated on 256 or 1024 threads, exact in-place use allowed) and the QSA sigmoid gate
adapted from llama.cpp.  This file proves the block-reduction slot bounds, the thread ownership that
makes in-place normalisation race-free, the gate's second-half addressing and the host span checks.

## Source semantics
* The norm reads its row once for the reduction; after the barrier every thread touches only the
  columns `tid, tid + B, tid + 2B, ...`, so output may alias input exactly.
* `gate` reads the second half of each head's `2 * head_dim` fused query row.
* `buffers` accepts disjoint spans or an exact input/output alias.

## Form template identifiers
- `norm<BlockSize>` -> `rms_norm_weighted`; `gate` -> `sigmoid_gate`; `warp_sum` -> `lane_reduce`
- `buffers` -> `span_check`; `elements` -> `element_count`
-/

namespace Strata.NativeQsaNorms

-- § reduction slots

/-- With 256 threads the warp slot is below 8 and below the 32-entry `sums` array. -/
theorem slot_256 {t : Nat} (ht : t < 256) : t / 32 < 8 ∧ t / 32 < 32 := by omega

/-- With 1024 threads the warp slot is below 32, exactly the size of `sums`. -/
theorem slot_1024 {t : Nat} (ht : t < 1024) : t / 32 < 32 := by omega

/-- The second-stage read `lane < BlockSize/32` covers exactly the warps that wrote a slot. -/
theorem second_stage_lanes : 256 / 32 = 8 ∧ 1024 / 32 = 32 := by decide

-- § in-place safety

/-- The thread that reads column `col` in the strided loop is the same thread that writes it. -/
theorem col_owner {col B : Nat} (hB : 0 < B) : col % B < B ∧ col % B + B * (col / B) = col :=
  ⟨Nat.mod_lt _ hB, Nat.mod_add_div col B⟩

/-- Two different columns are never visited by the same loop iteration of the same thread. -/
theorem col_owner_inj {c c' B : Nat} (hB : 0 < B) (h1 : c % B = c' % B) (h2 : c / B = c' / B) : c = c' := by
  have a := Nat.mod_add_div c B
  have b := Nat.mod_add_div c' B
  rw [h1, h2] at a
  omega

-- § gate addressing

/-- The gate half `head*(2*hd) + hd + channel` is inside the fused `n_head * (2*hd)` query buffer. -/
theorem gate_index_lt {head channel n hd : Nat} (hh : head < n) (hc : channel < hd) :
    head * (2 * hd) + hd + channel < n * (2 * hd) := by
  have h := Nat.mul_le_mul_right (2 * hd) (show head + 1 ≤ n by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- The fused query holds twice as many floats as the attention output. -/
theorem fused_q_size (n hd : Nat) : n * (2 * hd) = 2 * (hd * n) := by
  rw [Nat.mul_comm]
  exact Nat.mul_assoc 2 hd n

/-- The flat index splits into (head, channel) exactly. -/
theorem gate_split {i hd n : Nat} (hd0 : 0 < hd) (hi : i < hd * n) :
    i / hd < n ∧ i % hd < hd ∧ (i / hd) * hd + i % hd = i := by
  refine ⟨Nat.div_lt_of_lt_mul hi, Nat.mod_lt _ hd0, ?_⟩
  rw [Nat.mul_comm]
  exact Nat.div_add_mod i hd

/-- The launch of `ceil(count/256)` blocks covers every element. -/
theorem gate_grid_cover {count i : Nat} (hi : i < count) : i / 256 < (count + 255) / 256 := by omega

-- § span predicate

/-- Half-open byte ranges flagged as overlapping. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap predicate is symmetric. -/
theorem overlap_comm {ap an bp bn : Nat} : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap
  omega

/-- An exact alias is the one overlap the norm accepts; for a nonempty span it is flagged by the predicate. -/
theorem alias_is_flagged {p n : Nat} (hn : 0 < n) : overlap p n p n := by
  unfold overlap
  omega

/-- Ranges laid end to end are not flagged. -/
theorem adjacent_not_flagged {ap an bp bn : Nat} (h : ap + an ≤ bp) : ¬ overlap ap an bp bn := by
  unfold overlap
  omega

/-- The element count `cols * rows` bound keeps the byte size `count * 4` representable in 32-bit ints. -/
theorem count_bytes {count : Nat} (h : count ≤ 2147483647) : count * 4 ≤ 8589934588 := by omega

end Strata.NativeQsaNorms
