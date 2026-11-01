/-!
# MODULE: gated_residual_gemv_pipeline
The staged gated-residual (hyper-connection) read and write: per-stream RMSNorm, down projection (one block per
output row, warps split the reduction), gate projection (one warp per row), stream mean, optional inject projection and
the write-back `out = R + block_out * 2*sigmoid(inject/hc)`.  Covers the workspace layout, the warp-split reduction
coverage and the index decomposition of the elementwise passes.

## Source semantics
* `gr_workspace_init` lays five 16-byte-aligned regions (xn, xq, lq, gated, lo) back to back; the original bug was a size
  table in a different order from the pointer table, so the real-geometry sizes are pinned here.
* `gr_down_kernel`: block `k` is row `k`; warp `w` of `nw` visits `i = w*32 + lane + m*nw*32`, so lanes touch consecutive
  addresses and every element of the 10240-wide row is visited exactly once.
* `gr_gate_kernel`: one warp per row, `(hc_dim + 7)/8` blocks of 8 warps, lanes stride `hc_lr`.
* `gr_write_kernel`: every stream adds the SAME block output; only the per-stream weight differs.
* The native MMVF path needs `hc*n_embd` and `hc_lr` both even.

## Form template identifiers
* `alignUp16`           : region size rounding
* `offset`              : start offset of region k
* `warpSlot`            : (warp, lane) position inside the strided reduction
* `streamOf` / `colOf`  : decomposition of a flat stream index
-/

namespace Strata.GrPipeline

-- § workspace layout

/-- Round a byte count up to a multiple of 16, as `(sz + 15) & ~15`. -/
def alignUp16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding never shrinks a region. -/
theorem alignUp16_ge (n : Nat) : n ≤ alignUp16 n := by unfold alignUp16; omega

/-- Rounding adds fewer than 16 bytes. -/
theorem alignUp16_lt (n : Nat) : alignUp16 n < n + 16 := by unfold alignUp16; omega

/-- The rounded size is 16-byte aligned. -/
theorem alignUp16_aligned (n : Nat) : alignUp16 n % 16 = 0 := by unfold alignUp16; omega

/-- Rounding is idempotent. -/
theorem alignUp16_idem (n : Nat) : alignUp16 (alignUp16 n) = alignUp16 n := by unfold alignUp16; omega

/-- Region sizes (bytes) at the real geometry, in pointer order: xn, xq, lq, gated, lo. -/
def realSizes : List Nat := [10240 * 4, 10240 * 2, 320 * 2, 10240 * 4, 320 * 4]

/-- The bf16 copy `xq` needs 20480 bytes (the buggy table gave it 640). -/
theorem xq_size : 10240 * 2 = 20480 := by decide

/-- Every real-geometry region is already 16-byte aligned, so alignment adds no padding. -/
theorem real_sizes_aligned : realSizes.map alignUp16 = realSizes := by decide

/-- Total workspace bytes at the real geometry. -/
theorem real_total : (realSizes.map alignUp16).foldl (· + ·) 0 = 104320 := by decide

/-- Start offset of region `k` is the sum of the aligned sizes before it. -/
def offset (sizes : List Nat) (k : Nat) : Nat := ((sizes.take k).map alignUp16).foldl (· + ·) 0

/-- Consecutive regions are adjacent: region `k+1` starts exactly where the aligned region `k` ends. -/
theorem offset_succ_real (k : Nat) (hk : k < 5) :
    offset realSizes (k + 1) = offset realSizes k + alignUp16 (realSizes.getD k 0) := by
  revert k; decide

-- § warp-split reduction

/-- Position of (warp, lane, round `m`) in the strided reduction over `nw` warps. -/
def warpSlot (nw warp lane m : Nat) : Nat := warp * 32 + lane + m * (nw * 32)

/-- Every index below the reduction width belongs to exactly one (warp, lane, round). -/
theorem warp_slot_covers (nw i : Nat) (hnw : 0 < nw) :
    ∃ warp lane m, warp < nw ∧ lane < 32 ∧ warpSlot nw warp lane m = i := by
  refine ⟨(i % (nw * 32)) / 32, (i % (nw * 32)) % 32, i / (nw * 32), ?_, ?_, ?_⟩
  · have h : i % (nw * 32) < nw * 32 := Nat.mod_lt _ (by omega)
    omega
  · omega
  · unfold warpSlot
    have h1 := Nat.div_add_mod i (nw * 32)
    have h2 := Nat.div_add_mod (i % (nw * 32)) 32
    have h3 : i / (nw * 32) * (nw * 32) = (nw * 32) * (i / (nw * 32)) := Nat.mul_comm _ _
    omega

/-- Within one round, warp `w` lane `l` and warp `w'` lane `l'` collide only if they are the same thread. -/
theorem warp_slot_inj (nw w l w' l' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : w * 32 + l = w' * 32 + l') : w = w' ∧ l = l' := by omega

/-- The eight warps' partials reduce through `part[WARPS]` with `WARPS = 256 / 32`. -/
theorem part_slots : 256 / 32 = 8 := by decide

-- § gate kernel rows

/-- Gate row of warp `warp` in block `blk`. -/
def gateRow (blk warp : Nat) : Nat := blk * 8 + warp

/-- `(hc_dim + 7)/8` blocks of 8 warps cover every gate row. -/
theorem gate_row_covered (hc_dim r : Nat) (h : r < hc_dim) :
    r / 8 < (hc_dim + 7) / 8 ∧ gateRow (r / 8) (r % 8) = r := by
  unfold gateRow; omega

/-- Mean kernel: `(n_embd + 255)/256` blocks of 256 threads cover every column. -/
theorem mean_cover (n d : Nat) (h : d < n) : d / 256 < (n + 255) / 256 := by omega

-- § flat stream indexing

/-- Stream of a flat index in the `(hc, n_embd)` layout. -/
def streamOf (n i : Nat) : Nat := i / n
/-- Column of a flat index. -/
def colOf (n i : Nat) : Nat := i % n

/-- Every flat index below `hc * n_embd` splits into a valid stream and column and recombines. -/
theorem stream_col_split (n hc i : Nat) (hn : 0 < n) (h : i < hc * n) :
    streamOf n i < hc ∧ colOf n i < n ∧ streamOf n i * n + colOf n i = i := by
  unfold streamOf colOf
  refine ⟨?_, Nat.mod_lt _ hn, ?_⟩
  · exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)
  · have := Nat.div_add_mod i n
    rw [Nat.mul_comm] at this
    exact this

/-- Writing `out_i = R_i + bo_d * w_c` changes element `(c, d)` by exactly `bo_d * w_c`. -/
theorem write_delta (R bo w : Int) : R + bo * w - R = bo * w := by omega

-- § native path preconditions

/-- The real geometry satisfies the even-size requirement of the native MMVF path. -/
theorem native_even : (4 * 2560) % 2 = 0 ∧ 320 % 2 = 0 := by decide

/-- `gr_write` launches `(hc*n_embd + 255)/256` blocks; the grid-stride loop covers every element. -/
theorem write_cover (n i : Nat) (h : i < n) : i / 256 < (n + 255) / 256 := by omega

end Strata.GrPipeline
