/-!
# MODULE: gr_gated_residual
Workspace layout and index arithmetic of `gr_read` / `gr_write`, the gated residual (hyper-connection).

## Source semantics
`gr_workspace_init` lays five regions (xn, xq, lq, gated, lo) each rounded up to 16 bytes so none overlaps.
`gr_write`: out[i] = R[i] + block_out[d] * w[c] with i = c*n_embd + d.
`down` is one block per output row; `gate` one warp per row (8 rows/block); `mean` one thread per column.

## Form template identifiers
n_embd -> embedding width; hc -> streams; hc_lr -> bottleneck; hc_dim = hc*n_embd; al -> 16-byte round up.
-/
namespace Strata.Gr

/-- Round up to a multiple of 16. -/
def al (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding never shrinks. -/
theorem al_ge (n : Nat) : n ≤ al n := by unfold al; omega

/-- Rounded sizes are 16-byte aligned. -/
theorem al_mod (n : Nat) : al n % 16 = 0 := by unfold al; omega

/-- Rounding adds less than 16 bytes. -/
theorem al_lt (n : Nat) : al n < n + 16 := by unfold al; omega

/-- A region that starts aligned and has an aligned size leaves the next region aligned. -/
theorem next_aligned (p n : Nat) (h : p % 16 = 0) : (p + al n) % 16 = 0 := by
  have := al_mod n; omega

/-- Consecutive regions never overlap: the next start is at least start + size. -/
theorem region_disjoint (p n : Nat) : p + n ≤ p + al n := by
  have := al_ge n; omega

/-- Total workspace is the sum of the rounded sizes, hence at least the sum of raw sizes. -/
theorem total_ge (a b c d e : Nat) : a + b + c + d + e ≤ al a + al b + al c + al d + al e := by
  have := al_ge a; have := al_ge b; have := al_ge c; have := al_ge d; have := al_ge e; omega

/-- Flat index decomposition for gr_write: i = n_embd*(i/n_embd) + i%n_embd. -/
theorem write_decomp (i n : Nat) : n * (i / n) + i % n = i := Nat.div_add_mod i n

/-- The stream index of a flat index below n*hc is below hc. -/
theorem write_stream_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- The column index is below n_embd. -/
theorem write_col_lt (i n : Nat) (h : 0 < n) : i % n < n := Nat.mod_lt _ h

/-- Gate kernel grid: ceil(hc_dim/8) blocks of 8 warps cover all rows. -/
theorem gate_cover (row hcdim : Nat) (h : row < hcdim) : row / 8 < (hcdim + 7) / 8 := by omega

/-- Gate row from (block, warp). -/
theorem gate_row_decomp (b w : Nat) (hw : w < 8) : (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by omega

/-- Mean kernel grid covers n_embd columns with 256 threads. -/
theorem mean_cover (d n : Nat) (h : d < n) : d / 256 < (n + 255) / 256 := by omega

/-- Down kernel: warp w of nw starting at w*32 with stride nw*32 partitions the strided index set (nw=8). -/
theorem down_partition (i : Nat) : ∃ w l k, w < 8 ∧ l < 32 ∧ i = w * 32 + l + 256 * k :=
  ⟨i / 32 % 8, i % 32, i / 256, by omega, by omega, by omega⟩

/-- Distinct (warp, lane, step) triples give distinct indices. -/
theorem down_unique (w l k w' l' k' : Nat) (hw : w < 8) (hl : l < 32) (hw' : w' < 8) (hl' : l' < 32)
    (e : w * 32 + l + 256 * k = w' * 32 + l' + 256 * k') : w = w' ∧ l = l' ∧ k = k' := by omega

/-- The down reduction stage reads at most nw=8 warp partials. -/
theorem partials_bound (lane : Nat) (h : lane < 8) : lane < 8 := h

/-- block_sum scratch of 8 doubles (64 bytes) is at least as large as 8 floats (32 bytes). -/
theorem scratch_fits : 8 * 4 ≤ 8 * 8 := by decide

/-- Evenness precondition for the native MMVF path is preserved by multiplication by hc=4. -/
theorem even_hc_dim (n : Nat) : (4 * n) % 2 = 0 := by omega

/-- Inject block uses 32*hc threads: hc warps. -/
theorem inject_warps (hc : Nat) : 32 * hc / 32 = hc := by omega

end Strata.Gr
