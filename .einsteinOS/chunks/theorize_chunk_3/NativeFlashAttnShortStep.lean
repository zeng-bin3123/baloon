/-!
# MODULE: native_flash_attn_short_step
Index, tiling and validity contracts of the specialised single-token FlashAttention kernel `attend`
(Q 24 heads x 256, KV 2 heads x 256, f16 K/V, context 1..256, 128 threads per head).

## Source semantics
* Block `head` handles one query head; its KV head is `head / 12` (24 query heads share 2 KV heads).
* K/V cell `c` of KV head `kv` starts at `(c * 2 + kv) * 256`.
* The step buffer is valid only if `1 <= width <= max_context`, `nkv = width`, `pos = width - 1` and
  `nbid = width / 4`; otherwise the output is NaN and status reports unsupported-step.
* The 128 threads form 4 warps; each warp scores 32 cells per iteration, `cell = base + warp*32 + (lane & ~7) + row`
  with `row < 8`, so the 4 warps cover 128 cells per iteration and 2 iterations cover the padded length 256.
* Each lane holds `float2 qreg[16]` and `vkq[16]`: slot `i0/8 + j` with `i0 in {0,32,64,96}` and `j < 4`, covering
  head dims `2*(i0 + 4*(lane%8) + j) + {0,1}` = all 256 dims over the 8 lanes sharing a cell.
* The final cross-warp reduction goes through a `4*4*256` float tile.

## Form template identifiers
* `kvHead` / `cellBase`    : KV head and K/V cell start
* `cellOf`                 : cell scored by (base, warp, lane, row)
* `dimOf`                  : head dim covered by (i0 slot, lane%8, j)
* `StepValid`              : the validity predicate of the step buffer
-/

namespace Strata.NativeFlashAttn

-- § head mapping

/-- KV head of query head `head` (12 query heads per KV head). -/
def kvHead (head : Nat) : Nat := head / 12

/-- All 24 query heads map to one of the 2 KV heads. -/
theorem kvHead_lt (head : Nat) (h : head < 24) : kvHead head < 2 := by unfold kvHead; omega

/-- Heads `0..11` use KV head 0 and `12..23` use KV head 1. -/
theorem kvHead_split (head : Nat) (h : head < 24) : (head < 12 → kvHead head = 0) ∧ (12 ≤ head → kvHead head = 1) := by
  unfold kvHead; constructor <;> intro hh <;> omega

/-- Start of K/V cell `c` of KV head `kv` (element offset). -/
def cellBase (c kv : Nat) : Nat := (c * 2 + kv) * 256

/-- A valid cell and KV head give an offset whose 256-wide row fits in `capacity * 2 * 256` elements. -/
theorem cellBase_row_lt (c kv capacity : Nat) (hc : c < capacity) (hk : kv < 2) :
    cellBase c kv + 256 ≤ capacity * 2 * 256 := by
  unfold cellBase
  have h : (c + 1) * 2 ≤ capacity * 2 := Nat.mul_le_mul_right 2 hc
  have h2 : (c * 2 + kv + 1) * 256 ≤ capacity * 2 * 256 := Nat.mul_le_mul_right 256 (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- The KV buffers are `capacity * 1024` bytes: `capacity * 2 heads * 256 dims * 2 bytes`. -/
theorem kv_bytes (capacity : Nat) : capacity * 2 * 256 * 2 = capacity * 1024 := by omega

-- § step validity

/-- The step-buffer validity predicate checked by the kernel. -/
def StepValid (width nkv pos nbid max_context : Nat) : Prop :=
  1 ≤ width ∧ width ≤ max_context ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

/-- A valid step names the last cell of the context as the current position. -/
theorem stepValid_pos (width nkv pos nbid m : Nat) (h : StepValid width nkv pos nbid m) : pos < width := by
  unfold StepValid at h; omega

/-- With `max_context ≤ 256` a valid width never exceeds the padded length 256. -/
theorem stepValid_within_padded (width nkv pos nbid m : Nat) (h : StepValid width nkv pos nbid m) (hm : m ≤ 256) :
    width ≤ 256 := by
  unfold StepValid at h; omega

-- § cell scoring geometry

/-- Cell scored by (base, warp, lane, row): `base + warp*32 + 8*(lane/8) + row`. -/
def cellOf (base warp lane row : Nat) : Nat := base + warp * 32 + 8 * (lane / 8) + row

/-- Within one warp the (lane group, row) pairs cover the 32 local cells exactly once. -/
theorem local_cell_bijective (j : Nat) (h : j < 32) : 8 * (j / 8) + j % 8 = j ∧ j / 8 < 4 ∧ j % 8 < 8 := by omega

/-- Distinct (lane group, row) pairs score distinct local cells. -/
theorem local_cell_inj (g r g' r' : Nat) (hr : r < 8) (hr' : r' < 8) (h : 8 * g + r = 8 * g' + r') :
    g = g' ∧ r = r' := by omega

/-- The 4 warps cover 128 consecutive cells per iteration. -/
theorem warps_cover_iteration (warp local_ : Nat) (hw : warp < 4) (hl : local_ < 32) : warp * 32 + local_ < 128 := by omega

/-- Two 128-cell iterations cover the padded length 256, hence every cell `< width ≤ 256`. -/
theorem iterations_cover (cell : Nat) (h : cell < 256) : cell / 128 < 2 ∧ cell % 128 < 128 := by omega

-- § head-dim register map

/-- Head dim (first of the pair) covered by slot `m` (`i0 = 32 m`), lane group `l = lane % 8` and `j`. -/
def dimOf (m l j : Nat) : Nat := 2 * (32 * m + 4 * l + j)

/-- The 16 register slots `4*m + j` fit `float2 qreg[16]`. -/
theorem reg_slot_lt (m j : Nat) (hm : m < 4) (hj : j < 4) : 4 * m + j < 16 := by omega

/-- Each dim pair `(d, d+1)` stays inside 256. -/
theorem dim_pair_lt (m l j : Nat) (hm : m < 4) (hl : l < 8) (hj : j < 4) : dimOf m l j + 1 < 256 := by
  unfold dimOf; omega

/-- The register map is a bijection between (m, l, j) and the 128 dim pairs. -/
theorem dim_map_inj (m l j m' l' j' : Nat) (hl : l < 8) (hl' : l' < 8) (hj : j < 4) (hj' : j' < 4)
    (h : dimOf m l j = dimOf m' l' j') : m = m' ∧ l = l' ∧ j = j' := by
  unfold dimOf at h; omega

/-- Every pair index `i < 128` is reached from some (m, l, j). -/
theorem dim_map_surj (i : Nat) (h : i < 128) :
    i / 32 < 4 ∧ (i % 32) / 4 < 8 ∧ i % 4 < 4 ∧ 32 * (i / 32) + 4 * ((i % 32) / 4) + i % 4 = i := by omega

-- § output tile

/-- Shared tile `4*4*256 = 4096` floats. -/
def tileSize : Nat := 4 * 4 * 256

/-- The exact bound for lane group `l < 8`: the store index stays below 4096. -/
theorem tile_store_exact (warp q l m j : Nat) (hw : warp < 4) (hq : q < 4) (hl : l < 8) (hm : m < 4) (hj : j < 4) :
    warp * 4 * 256 + q * 256 + 2 * (32 * m + 4 * l + j) + 1 < 4096 := by omega

/-- The final reduction reads `tile[w*1024 + group*256 + i0 + tid]` for `w, group < 4`, `i0 ∈ {0,128}`, `tid < 128`. -/
theorem tile_read_lt (w group i0 tid : Nat) (hw : w < 4) (hg : group < 4) (hi : i0 = 0 ∨ i0 = 128) (ht : tid < 128) :
    w * 4 * 256 + group * 256 + i0 + tid < 4096 := by omega

/-- Output index `head*256 + i0 + tid` with `i0 ∈ {0,128}` stays inside the `24 * 256` output. -/
theorem out_index_lt (head i0 tid : Nat) (hh : head < 24) (hi : i0 = 0 ∨ i0 = 128) (ht : tid < 128) :
    head * 256 + i0 + tid < 24 * 256 := by omega

-- § span validation

/-- Two byte ranges overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric, so the pairwise `j < i` check is enough. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> intro h <;> omega

end Strata.NativeFlashAttn
