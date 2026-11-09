/-!
# MODULE: native_gdn_step
SOURCES: src/kernels/cuda/native_gdn.cu
Warp-per-(head, column) gated delta-net step adapted from llama.cpp: each warp keeps four state rows per
lane in registers, reduces with xor shuffles, and the host wrapper validates alignment/disjointness of every span.

## Source semantics
* Grid `(h_v, 1, S/4)`, block `(32, 4)`: `col = blockIdx.z*4 + threadIdx.y`; lane `l` owns rows `r*32 + l`.
* `q_head = head % h_k`; `h_v % h_k = 0` is required.
* Output scale is `1/sqrt(S)`; `S = 128`.
* A span is valid iff nonnull, 4-aligned and `address + bytes` does not overflow; spans overlap iff intervals intersect.

## Form template identifiers
* `S` -> state_size, `overlap` -> interval_overlap, `validSpan` -> aligned_bounded_span.
-/

namespace Ref.NativeGdn

/-- State size. -/
def S : Nat := 128

/-- 32 blocks along z of 4 columns each cover the 128 columns. -/
theorem z_blocks : S / 4 = 32 := by decide

/-- Every column belongs to exactly one (blockIdx.z, threadIdx.y). -/
theorem col_cover (c : Nat) (h : c < 128) : ∃ z y, z < 32 ∧ y < 4 ∧ z * 4 + y = c := by
  refine ⟨c / 4, c % 4, ?_, ?_, ?_⟩ <;> omega

/-- Column ownership is unique. -/
theorem col_unique (z y z' y' : Nat) (hy : y < 4) (hy' : y' < 4)
    (h : z * 4 + y = z' * 4 + y') : z = z' ∧ y = y' := by omega

/-- Lane `l` and register `r` address row `r*32 + l`, covering all 128 rows exactly once. -/
theorem row_cover (i : Nat) (h : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ r * 32 + l = i := by
  refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> omega

/-- Row ownership is unique. -/
theorem row_unique (r l r' l' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by omega

/-- Block shape: 32 x 4 = 128 threads (matches `__launch_bounds__(128, 2)`). -/
theorem block_threads : 32 * 4 = 128 := by decide

/-- Head pairing is a valid q/k head. -/
theorem q_head_lt (head h_k : Nat) (h : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ h

/-- State byte size is `65536 * h_v`. -/
theorem state_bytes (h_v : Nat) : 128 * 128 * h_v * 4 = 65536 * h_v := by omega

/-- Output byte size is `512 * h_v`. -/
theorem output_bytes (h_v : Nat) : 128 * h_v * 4 = 512 * h_v := by omega

/-- Interval overlap of two byte spans. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; constructor <;> intro h <;> omega

/-- The source's test has no emptiness guard: an empty span at `a` is reported as overlapping only when it
lies strictly inside the other span. -/
theorem overlap_empty_inside (a b bn : Nat) (h : overlap a 0 b bn) : b < a ∧ a < b + bn := by
  unfold overlap at h; omega

/-- Spans that end before the other begins are disjoint. -/
theorem disjoint_of_le (a an b bn : Nat) (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap; omega

/-- Every span overlaps itself when nonempty. -/
theorem overlap_self (a n : Nat) (h : 0 < n) : overlap a n a n := by
  unfold overlap; omega

/-- A valid span: nonnull, 4-aligned, and `address + bytes` fits in the address space. -/
def validSpan (maxAddr addr bytes : Nat) : Prop := addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ maxAddr - addr

/-- A valid span with `addr <= maxAddr` ends within the address space. -/
theorem validSpan_end (maxAddr addr bytes : Nat) (h : validSpan maxAddr addr bytes)
    (hm : addr ≤ maxAddr) : addr + bytes ≤ maxAddr := by
  unfold validSpan at h; omega

/-- A launch grid of `h_v` blocks in x must be `<= 65535`; the check rejects larger. -/
theorem grid_x_ok (h_v : Nat) (h : h_v ≤ 65535) : h_v < 65536 := by omega

/-- `h_v % h_k = 0` makes the pairing balanced in count (every k head used `h_v / h_k` times at head 0 residue). -/
theorem divisible_pairing (h_v h_k : Nat) (hk : 0 < h_k) (h : h_v % h_k = 0) : h_v / h_k * h_k = h_v := by
  have := Nat.div_add_mod h_v h_k
  rw [h] at this
  rw [Nat.mul_comm]; omega

end Ref.NativeGdn
