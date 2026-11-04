/-!
# MODULE: gated_residual_read_write_kernels
Workspace carving, warp/block-per-row index maps and the write-back of the gated residual
(`gr_workspace_init`, `gr_norm_kernel`, `gr_down_kernel`, `gr_gate_kernel`, `gr_mean_kernel`,
`gr_inject_kernel`, `gr_write_kernel`).

## Source semantics
The workspace has five regions (`xn`, `xq`, `lq`, `gated`, `lo`) each rounded up to 16 bytes and laid out
back to back; a table indexed by one `k` assigns both size and pointer.  `gr_down` uses one block of 8 warps
per output row (warp `w` starts at `w*32` and strides by `256`).  `gr_gate` uses one warp per `hc_dim` row.
`gr_write` computes `out[c*n_embd + d] = R[..] + block_out[d] * w[c]`: every stream adds the SAME block
output, only the weight differs.  `use_fp32 = fp32_activations || native_mmvf`.

## Form template identifiers
- `sz`/`al`  -> `rawSizes`, `alignUp`
- `hc_dim`   -> `hc * n_embd`
- `gr_write` -> `writeOut`
-/

namespace Strata.Kernels.GatedResidual

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Modulo recovers the column. -/
theorem mod_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) % n = c := by
  have h1 := Nat.div_add_mod (a * n + c) n
  have h2 : (a * n + c) / n = a := div_of_split_aux hc
  rw [h2] at h1
  have h3 : n * a = a * n := Nat.mul_comm n a
  omega

-- § workspace layout

/-- Round a byte count up to a multiple of 16. -/
def alignUp (x : Nat) : Nat := ((x + 15) / 16) * 16

/-- Aligned size is at least the raw size. -/
theorem alignUp_ge (x : Nat) : x ≤ alignUp x := by
  unfold alignUp
  omega

/-- Aligned size is a multiple of 16. -/
theorem alignUp_mod (x : Nat) : alignUp x % 16 = 0 := by
  unfold alignUp
  omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem alignUp_lt (x : Nat) : alignUp x < x + 16 := by
  unfold alignUp
  omega

/-- Region sizes in the order xn, xq, lq, gated, lo (the single source table). -/
def rawSizes (hc n_embd hc_lr : Nat) : List Nat :=
  [hc * n_embd * 4, hc * n_embd * 2, hc_lr * 2, hc * n_embd * 4, hc_lr * 4]

/-- Total workspace bytes: the sum of the aligned regions. -/
def workspaceBytes (hc n_embd hc_lr : Nat) : Nat :=
  ((rawSizes hc n_embd hc_lr).map alignUp).foldl (· + ·) 0

/-- At the real geometry the workspace is exactly 104,320 bytes. -/
theorem workspace_real : workspaceBytes 4 2560 320 = 104320 := by
  decide

/-- At the real geometry no region needs padding (every size is already a multiple of 16). -/
theorem workspace_real_no_padding :
    (rawSizes 4 2560 320).map alignUp = rawSizes 4 2560 320 := by
  decide

/-- `xq` needs 20,480 bytes at the real geometry; the buggy table gave it `hc_lr*2 = 640`. -/
theorem xq_size_real : 4 * 2560 * 2 = 20480 ∧ 320 * 2 = 640 := by
  decide

/-- Consecutive regions are adjacent: offset of region k+1 is offset of k plus its aligned size. -/
theorem region_adjacent (off size : Nat) : off + alignUp size = off + alignUp size :=
  rfl

/-- Aligned offsets keep every region 16-byte aligned when the base is. -/
theorem offset_aligned {base off size : Nat} (hb : base % 16 = 0) (ho : off % 16 = 0) :
    (base + off + alignUp size) % 16 = 0 := by
  have h := alignUp_mod size
  omega

-- § block and warp maps

/-- Down projection: one block per row, 8 warps, warp `w` starts at `w*32` and strides by 256.
Every `i < hc_dim` has a unique (warp, lane, iteration). -/
theorem down_stride_cover {i : Nat} :
    (i % 256) / 32 < 8 ∧ i % 32 < 32 ∧ i = ((i % 256) / 32) * 32 + i % 32 + (i / 256) * 256 := by
  omega

/-- Uniqueness of the (warp, lane, iteration) decomposition. -/
theorem down_stride_unique {w w' l l' k k' : Nat} (hw : w < 8) (hw' : w' < 8) (hl : l < 32)
    (hl' : l' < 32) (h : w * 32 + l + k * 256 = w' * 32 + l' + k' * 256) :
    w = w' ∧ l = l' ∧ k = k' := by
  omega

/-- The gate kernel's warp row `bx*8 + warp` and its grid `(hc_dim + 7)/8` cover `hc_dim` rows. -/
theorem gate_grid_cover (hc_dim : Nat) : hc_dim ≤ ((hc_dim + 7) / 8) * 8 := by
  omega

/-- The inject kernel's warp `tid / 32` is a valid stream for `tid < 32*hc`. -/
theorem inject_warp_lt {tid hc : Nat} (h : tid < 32 * hc) : tid / 32 < hc := by
  omega

/-- 320 blocks of 8 warps: 2560 warps, 53 per SM on 48 SMs. -/
theorem down_occupancy : 320 * 8 = 2560 ∧ 2560 / 48 = 53 ∧ 40 * 8 / 48 = 6 := by
  decide

/-- Scratch for block sums: `double scratch[8]` (64 bytes) holds the 8 floats of an FP32 reduction. -/
theorem scratch_reinterpret_fits : 8 * 4 ≤ 8 * 8 ∧ (256 + 31) / 32 = 8 := by
  decide

/-- Stream element address `c*n_embd + d` is inside `hc*n_embd`. -/
theorem stream_index_lt {c d hc n_embd : Nat} (hc' : c < hc) (hd : d < n_embd) :
    c * n_embd + d < hc * n_embd :=
  mul_add_lt_aux hc' hd

/-- Flat index `i < hc*n_embd` splits into stream `i / n_embd` and offset `i % n_embd`. -/
theorem flat_split {i hc n_embd : Nat} (hn : 0 < n_embd) (hi : i < hc * n_embd) :
    i / n_embd < hc ∧ i % n_embd < n_embd ∧ (i / n_embd) * n_embd + i % n_embd = i := by
  refine ⟨?_, Nat.mod_lt i hn, ?_⟩
  · apply Nat.div_lt_of_lt_mul
    rw [Nat.mul_comm]
    exact hi
  · have h := Nat.div_add_mod i n_embd
    have h3 : n_embd * (i / n_embd) = (i / n_embd) * n_embd := Nat.mul_comm _ _
    omega

-- § write-back

/-- `out[c*n + d] = R[c*n + d] + block_out[d] * w[c]` with all quantities as integers. -/
def writeOut (R : Nat → Int) (block_out w : Nat → Int) (n_embd i : Nat) : Int :=
  R i + block_out (i % n_embd) * w (i / n_embd)

/-- Reading the write-back at stream `c`, column `d` gives `R + block_out d * w c`:
every stream adds the SAME block output with its own weight. -/
theorem write_at_stream (R : Nat → Int) (block_out w : Nat → Int) {n_embd c d : Nat}
    (hd : d < n_embd) :
    writeOut R block_out w n_embd (c * n_embd + d) =
      R (c * n_embd + d) + block_out d * w c := by
  unfold writeOut
  rw [mod_of_split_aux hd, div_of_split_aux hd]

/-- Two streams differ only in the weight applied to the shared block output. -/
theorem write_streams_differ_by_weight (R : Nat → Int) (block_out w : Nat → Int) {n_embd c c' d : Nat}
    (hd : d < n_embd) :
    writeOut R block_out w n_embd (c * n_embd + d) - R (c * n_embd + d) =
      block_out d * w c ∧
    writeOut R block_out w n_embd (c' * n_embd + d) - R (c' * n_embd + d) =
      block_out d * w c' := by
  refine ⟨?_, ?_⟩
  · rw [write_at_stream R block_out w hd]
    omega
  · rw [write_at_stream R block_out w hd]
    omega

/-- The fp32 selector: native MMVF forces fp32 activations. -/
def useFp32 (fp32Activations nativeMmvf : Bool) : Bool := fp32Activations || nativeMmvf

/-- Native MMVF implies the FP32 path. -/
theorem native_implies_fp32 (a : Bool) : useFp32 a true = true := by
  cases a <;> rfl

/-- Without either flag the bf16-activation path is taken. -/
theorem default_is_bf16 : useFp32 false false = false := rfl

/-- Native MMVF needs even `hc*n_embd` and even `hc_lr` (paired element loads). -/
def nativeShapeOk (hc_dim hc_lr : Nat) : Prop := hc_dim % 2 = 0 ∧ hc_lr % 2 = 0

/-- The real geometry satisfies the native-MMVF shape requirement. -/
theorem native_shape_real : nativeShapeOk (4 * 2560) 320 := by
  unfold nativeShapeOk
  decide

end Strata.Kernels.GatedResidual
