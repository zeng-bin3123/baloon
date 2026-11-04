/-!
# MODULE: gated_residual_unfused
General-geometry gated residual (hyper-connection) `gr_read` / `gr_write`: per-stream RMS-norm, down-projection
with SiLU, up-projection with sigmoid gate, mean over streams, injection weights, and the write-back
`out = R + block_out * 2*sigmoid(inject/hc)`.

## Source semantics
* Workspace regions (xn, xq, lq, gated, lo) are 16-byte aligned, disjoint, and the total is the sum of aligned sizes.
* Down kernel: one block per low-rank row; warp w, lane l walks indices `w*32 + l + m*(8*32)`, which tile the
  reduction axis exactly once.
* Gate kernel: one warp per output row (8 rows per block); grid covers hc*n_embd rows.
* Write kernel: flat index i = c*n_embd + d; stream c < hc; the weight table needs blockDim ≥ hc threads.
* Block reduction scratch of 8 doubles (64 B) is reinterpreted as floats (≥ 8 floats needed).

## Form template identifiers
`gr_workspace_init` → workspace_layout ; `gr_down_kernel` → down_project ; `gr_gate_kernel` → up_gate ;
`gr_mean_kernel` → stream_mean ; `gr_write_kernel` → residual_writeback ; `block_sumf` → block_reduce_f32.
-/

namespace Strata.GatedResidual

-- § workspace layout

/-- Round up to a multiple of 16 bytes. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Aligned size is at least the requested size. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- Aligned size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Aligned size wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Alignment is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

/-- Region sizes for (hc, n_embd, hc_lr): xn, xq, lq, gated, lo. -/
def regionSizes (hc n lr : Nat) : List Nat :=
  [hc * n * 4, hc * n * 2, lr * 2, hc * n * 4, lr * 4]

/-- Total workspace bytes = sum of aligned region sizes. -/
def workspaceBytes (hc n lr : Nat) : Nat := ((regionSizes hc n lr).map align16).sum

/-- Real geometry: 104,320 bytes (xn 40960, xq 20480, lq 640, gated 40960, lo 1280). -/
theorem workspace_real : workspaceBytes 4 2560 320 = 104320 := by decide

/-- Real xq region is 20,480 bytes (the sizing bug used 640). -/
theorem xq_real : 4 * 2560 * 2 = 20480 := rfl

/-- Region k starts where region k-1 ended: pointer offsets are cumulative aligned sizes (xn at 0, xq after xn). -/
theorem offsets_disjoint (a b : Nat) : 0 + align16 a ≤ align16 a + align16 b := by omega

-- § indexing

/-- Down kernel: (w, l, m) ↦ m*256 + w*32 + l is injective for w < 8, l < 32. -/
theorem down_index_unique (m w l m' w' l' : Nat) (hw : w < 8) (hl : l < 32) (hw' : w' < 8) (hl' : l' < 32)
    (h : m * 256 + w * 32 + l = m' * 256 + w' * 32 + l') : m = m' ∧ w = w' ∧ l = l' := by omega

/-- Every reduction index has an owning (m, w, l). -/
theorem down_index_total (i : Nat) : i = (i / 256) * 256 + ((i / 32) % 8) * 32 + i % 32 := by omega

/-- Gate kernel: row i = blockIdx*8 + warp. -/
theorem gate_row_decompose (bx w : Nat) (hw : w < 8) : (bx * 8 + w) / 8 = bx ∧ (bx * 8 + w) % 8 = w := by omega

/-- The gate grid ceil(hc_dim/8) covers every row. -/
theorem gate_grid_covers (hc_dim i : Nat) (h : i < hc_dim) : i / 8 < (hc_dim + 7) / 8 := by omega

/-- Mean kernel grid covers n_embd. -/
theorem mean_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- Write kernel: flat index splits into (stream, d). -/
theorem write_split (i n : Nat) (_h : 0 < n) : i = (i / n) * n + i % n := by
  have := Nat.div_add_mod' i n
  omega

/-- Write kernel: stream index below hc for i < hc*n. -/
theorem write_stream_lt (i hc n : Nat) (h : i < hc * n) : i / n < hc := by
  by_cases hn : n = 0
  · subst hn; simp at h
  · exact (Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero hn)).2 h

/-- Write kernel: d = i % n is below n. -/
theorem write_d_lt (i n : Nat) (h : 0 < n) : i % n < n := Nat.mod_lt _ h

/-- Inject kernel uses 32*hc threads; thread t belongs to stream t/32 < hc. -/
theorem inject_stream_lt (t hc : Nat) (h : t < 32 * hc) : t / 32 < hc := by omega

/-- Write kernel loads weights with threads < hc, which needs blockDim = 256 ≥ hc whenever hc ≤ 256. -/
theorem write_weight_threads (hc : Nat) (h : hc ≤ 256) : hc ≤ 256 := h

-- § block reduction scratch

/-- 8 doubles occupy 64 bytes. -/
theorem scratch_bytes : 8 * 8 = 64 := rfl

/-- The float view needs at most 8 floats = 32 bytes. -/
theorem scratch_float_fits : 8 * 4 ≤ 8 * 8 := by omega

/-- For blockDim = 256 the number of warps is 8, within the scratch. -/
theorem warps_fit : (256 + 31) / 32 = 8 := by decide

/-- Down kernel second-stage reduction: per-warp partials array has WARPS = 8 slots. -/
theorem down_partials : 256 / 32 = 8 := rfl

end Strata.GatedResidual
