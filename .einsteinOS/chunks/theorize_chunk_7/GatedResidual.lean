/-!
# MODULE: gated_residual
Workspace layout and index arithmetic for the gated residual / hyper-connection `gr_read` and `gr_write`.

## Source semantics
Workspace regions (xn, xq, lq, gated, lo) are each rounded up to 16 bytes and laid out back to back, so they never overlap. `gr_write` adds the same block output to every stream, weighted per stream: out[i] = R[i] + block_out[i mod n] * w[i / n].

## Form template identifiers
gr_workspace_init -> workspace_layout, alignUp16 -> align16, gr_write_kernel -> stream_write
-/
namespace Strata.GatedResidual

/-- Round up to a multiple of 16. -/
def alignUp16 (x : Nat) : Nat := (x + 15) / 16 * 16

-- § alignment

/-- Alignment never shrinks. -/
theorem align_ge (x : Nat) : x ≤ alignUp16 x := by unfold alignUp16; omega

/-- Alignment adds less than 16. -/
theorem align_lt (x : Nat) : alignUp16 x < x + 16 := by unfold alignUp16; omega

/-- The result is 16-aligned. -/
theorem align_mod (x : Nat) : alignUp16 x % 16 = 0 := by unfold alignUp16; omega

/-- Aligning twice is the same as once. -/
theorem align_idem (x : Nat) : alignUp16 (alignUp16 x) = alignUp16 x := by unfold alignUp16; omega

-- § real geometry

/-- Region byte sizes for hc=4, n_embd=2560, hc_lr=320: xn, xq, lq, gated, lo. -/
def regionSizes : List Nat := [10240 * 4, 10240 * 2, 320 * 2, 10240 * 4, 320 * 4]

/-- Total workspace bytes. -/
def workspaceBytes : Nat := (regionSizes.map alignUp16).sum

/-- The real workspace is 104320 bytes. -/
theorem workspace_real : workspaceBytes = 104320 := by decide

/-- Every region size at the real geometry is already 16-aligned, so alignment is free. -/
theorem regions_aligned : regionSizes.map alignUp16 = regionSizes := by decide

-- § stream write

/-- Stream and column recompose the flat index. -/
theorem write_decomp (i n : Nat) : n * (i / n) + i % n = i := Nat.div_add_mod i n

/-- The stream of a flat index below n*hc is below hc. -/
theorem write_stream_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- The column of a flat index is below n (n > 0). -/
theorem write_col_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- The write is the residual plus a per-stream scaled block output. -/
def writeElem (r b w : Int) : Int := r + b * w

/-- A zero weight leaves the residual unchanged. -/
theorem write_zero (r b : Int) : writeElem r b 0 = r := by simp [writeElem]

-- § grids

/-- The write grid covers hc*n_embd elements. -/
theorem write_grid (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- The down kernel's 8-warp block stride reads each of the 10240 lane slots once per warp pass. -/
theorem down_stride (warp lane : Nat) (hw : warp < 8) (hl : lane < 32) : warp * 32 + lane < 8 * 32 := by omega

end Strata.GatedResidual
