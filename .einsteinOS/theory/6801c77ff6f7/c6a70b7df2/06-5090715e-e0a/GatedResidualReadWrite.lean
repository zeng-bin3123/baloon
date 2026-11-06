/-!
# MODULE: gated_residual_read_write
SOURCES: src/kernels/cuda/gr.cu
`gr_read` (norm, down, gate, mean, inject) and `gr_write` of the gated residual / hyper-connection, plus the
workspace carving of `gr_workspace_init`.

## Source semantics
* Workspace regions xn, xq, lq, gated, lo are carved in one table with 16-byte alignment; the earlier bug
  sized `xq` and `lq` in a different order from the pointer assignments.
* `gr_down_kernel`: one block per low-rank row, 8 warps splitting the `hc_dim` reduction.
* `gr_gate_kernel`: one warp per output row; `gr_mean_kernel`: one thread per embedding column.
* `gr_write_kernel`: `out[i] = R[i] + block_out[i % n_embd] * w[i / n_embd]`.

## Form template identifiers
gr_norm_kernel -> stream_rms_norm ; gr_down_kernel -> low_rank_down ; gr_gate_kernel -> low_rank_up_gate ;
gr_mean_kernel -> stream_mean ; gr_write_kernel -> residual_write ; GrWorkspace -> scratch_layout
-/

namespace Strata.Gr

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § workspace layout

/-- Round up to 16 bytes. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks and gains less than 16 bytes. -/
theorem align16_bounds (x : Nat) : x ≤ align16 x ∧ align16 x < x + 16 := by
  unfold align16; omega

/-- The aligned size is a multiple of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16; omega

/-- Region offsets of the five-entry table. -/
def offsets (a0 a1 a2 a3 : Nat) : Nat × Nat × Nat × Nat × Nat := (0, a0, a0 + a1, a0 + a1 + a2, a0 + a1 + a2 + a3)

/-- Total bytes of the five aligned regions. -/
def totalBytes (a0 a1 a2 a3 a4 : Nat) : Nat := a0 + a1 + a2 + a3 + a4

/-- Consecutive regions are disjoint and contained in the workspace. -/
theorem regions_disjoint (a0 a1 a2 a3 a4 : Nat) :
    (offsets a0 a1 a2 a3).1 + a0 ≤ (offsets a0 a1 a2 a3).2.1 ∧
    (offsets a0 a1 a2 a3).2.1 + a1 ≤ (offsets a0 a1 a2 a3).2.2.1 ∧
    (offsets a0 a1 a2 a3).2.2.1 + a2 ≤ (offsets a0 a1 a2 a3).2.2.2.1 ∧
    (offsets a0 a1 a2 a3).2.2.2.1 + a3 ≤ (offsets a0 a1 a2 a3).2.2.2.2 ∧
    (offsets a0 a1 a2 a3).2.2.2.2 + a4 = totalBytes a0 a1 a2 a3 a4 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> simp [offsets, totalBytes] <;> omega

/-- Real geometry sizes (xn, xq, lq, gated, lo) in bytes. -/
theorem real_sizes :
    (10240 * 4, 10240 * 2, 320 * 2, 10240 * 4, 320 * 4) = (40960, 20480, 640, 40960, 1280) := by decide

/-- All five real sizes are already 16-byte aligned, total 104,320 bytes. -/
theorem real_workspace :
    align16 40960 = 40960 ∧ align16 20480 = 20480 ∧ align16 640 = 640 ∧ align16 1280 = 1280 ∧
    totalBytes 40960 20480 640 40960 1280 = 104320 := by decide

/-- The historical mix-up: `xq` needs `hc_dim*2` bytes, far more than the `hc_lr*2` once assigned to it. -/
theorem xq_size_regression : 320 * 2 < 10240 * 2 := by decide

-- § reduction-axis geometry

/-- In `gr_down_kernel` the 8 warps stride by 256: each `i < hc_dim` is owned by one (warp, lane). -/
theorem down_split_owner (i : Nat) :
    (i % 256) / 32 < 8 ∧ i % 256 = (i % 256) / 32 * 32 + i % 32 := by omega

/-- Block count of the gate kernel covers every output row. -/
theorem gate_rows_cover (i hc_dim : Nat) (h : i < hc_dim) : i / 8 < (hc_dim + 7) / 8 := by omega

/-- Each lane of the gate kernel strides `hc_lr` by 32; every `k` has owner `k % 32`. -/
theorem gate_lane_owner (k : Nat) : k % 32 < 32 ∧ k = k % 32 + 32 * (k / 32) := by omega

/-- Norm loop: 2560 elements strided by 256 threads is 10 iterations. -/
theorem norm_trip_count : 2560 / 256 = 10 ∧ 2560 % 256 = 0 := by decide

/-- Block-wide scratch: `(blockDim+31)>>5` warps, at most 8 for 256 threads. -/
theorem block_scratch_warps : (256 + 31) / 32 = 8 := by decide

/-- A `float[8]` scratch fits inside the `double[8]` it is reinterpreted from. -/
theorem scratch_reinterpret_fits : 8 * 4 ≤ 8 * 8 := by decide

-- § mean and write kernels

/-- Gated activation index `c*n_embd + d` is in `hc*n_embd`. -/
theorem gated_index_lt {c hc d n_embd : Nat} (hc' : c < hc) (hd : d < n_embd) :
    c * n_embd + d < hc * n_embd := rowmajor_lt hc' hd

/-- In `gr_write_kernel`, a flat index splits into stream `< hc` and column `< n_embd`. -/
theorem write_split {i hc n_embd : Nat} (h : i < hc * n_embd) :
    i / n_embd < hc ∧ i % n_embd < n_embd ∧ i = (i / n_embd) * n_embd + i % n_embd := by
  have hpos : 0 < n_embd := by
    rcases Nat.eq_zero_or_pos n_embd with h0 | h0
    · subst h0; simp at h
    · exact h0
  have h1 : i / n_embd < hc := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  have h2 : i % n_embd < n_embd := Nat.mod_lt _ hpos
  have h3 := Nat.div_add_mod i n_embd
  refine ⟨h1, h2, ?_⟩
  rw [Nat.mul_comm]
  exact h3.symm

/-- The shared weight array `w` has `hc` slots, written by threads `< hc`; needs `hc ≤ blockDim`. -/
theorem write_weights_written {t hc : Nat} (ht : t < hc) (hb : hc ≤ 256) : t < 256 := by omega

/-- Grid-stride coverage of the write kernel: for stride `s > 0`, `i = i % s + s*(i/s)`. -/
theorem grid_stride_owner (i s : Nat) : i % s + s * (i / s) = i := Nat.mod_add_div i s

/-- Real geometry: 10240 elements, 40 blocks of 256 threads. -/
theorem write_blocks_real : (10240 + 255) / 256 = 40 := by decide

end Strata.Gr
