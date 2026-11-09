/-!
# MODULE: native_qsa_norm_gate
SOURCES: src/kernels/cuda/native_qsa.cu
llama.cpp-derived weighted RMS norm (in-place capable) and the sigmoid gate that multiplies attention output by the
second half of the fused q projection. This module models the validation and indexing contracts.

## Source semantics
* `elements(cols, rows)` requires positive dimensions whose product fits in int.
* The norm kernel uses a block of 256 threads for n_cols < 1024 and 1024 otherwise; thread t owns columns
  t, t + BS, t + 2 BS, ...; per-warp sums sit in a 32-entry shared array.
* `gate`: head = i / head_dim, channel = i % head_dim, and the gate value is q_full[head * 2 * head_dim + head_dim + channel].
* Buffer validation: byte spans must be 4-aligned, bounded and disjoint except for exact input/output aliasing.

## Form template identifiers
- `BlockSize` : 256 or 1024
- `overlap`   : half-open interval intersection test
- `gateIdx`   : head * (2 * head_dim) + (head_dim + channel)
-/

namespace Strata.NativeQsaNorm

-- § validation

/-- Positive dimensions give a positive element count. -/
theorem elements_pos (cols rows : Nat) (hc : 0 < cols) (hr : 0 < rows) : 0 < cols * rows :=
  Nat.mul_pos hc hr

/-- Interval overlap test, as in `overlap` of the source. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap test is symmetric. -/
theorem overlap_symm (ap an bp bn : Nat) : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap; omega

/-- A nonempty span overlaps itself, so a non-exact alias is always rejected. -/
theorem overlap_self (p n : Nat) (h : 0 < n) : overlap p n p n := by
  unfold overlap; omega

/-- A span of length 0 does not overlap itself. -/
theorem overlap_self_empty (p : Nat) : ¬ overlap p 0 p 0 := by
  unfold overlap; omega

/-- Disjoint adjacent spans never overlap. -/
theorem overlap_adjacent (p n m : Nat) : ¬ overlap p n (p + n) m := by
  unfold overlap; omega

-- § block geometry

/-- The block-size choice always gives at most 32 warps, so `sums[32]` suffices. -/
theorem warps_le_32 (n_cols : Nat) : (if n_cols < 1024 then 256 else 1024) / 32 ≤ 32 := by
  split <;> omega

/-- Each column is owned by exactly the thread with matching residue. -/
theorem stride_cover (col tid bs : Nat) (h : col % bs = tid) : col = tid + bs * (col / bs) := by
  have := Nat.mod_add_div col bs
  omega

/-- Columns visited by thread t (t + bs * k) have residue t. -/
theorem stride_residue (tid k bs : Nat) (h : tid < bs) : (tid + bs * k) % bs = tid := by
  rw [Nat.add_comm, Nat.mul_add_mod]
  exact Nat.mod_eq_of_lt h

-- § gate indexing

/-- Mixed-radix index bound. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m ha
  have h2 : (a + 1) * m = a * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The gate value read for (head, channel) lies inside the n_head * 2 * head_dim q_full buffer. -/
theorem gate_index_lt (head ch n_head hd : Nat) (hh : head < n_head) (hc : ch < hd) :
    head * (2 * hd) + (hd + ch) < n_head * (2 * hd) :=
  idx_lt head (hd + ch) n_head (2 * hd) hh (by omega)

/-- The gate buffer length n_head * (2 * hd) is twice the output length, matching the `count * 8` byte span. -/
theorem gate_span_double (n_head hd : Nat) : n_head * (2 * hd) = 2 * (n_head * hd) :=
  Nat.mul_left_comm n_head 2 hd

/-- The gate reads the second half: the offset hd + ch is at least head_dim. -/
theorem gate_reads_second_half (hd ch : Nat) : hd ≤ hd + ch := by
  omega

/-- Grid size covers every output element. -/
theorem gate_grid_covers (i count : Nat) (h : i < count) : i / 256 < (count + 255) / 256 := by
  omega

end Strata.NativeQsaNorm
