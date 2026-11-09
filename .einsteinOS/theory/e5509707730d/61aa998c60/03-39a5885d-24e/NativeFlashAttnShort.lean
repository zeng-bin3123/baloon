/-!
# MODULE: native_flash_attn_short
SOURCES: src/kernels/cuda/native_flash_attn.cu
Specialised decode FlashAttention (D=256, Q 24 heads, KV 2 heads, fp16 KV, context <= 256, one query column) adapted
from the pinned llama.cpp vector kernel. The thread/cell/dimension layouts, step validity gate and host validation
are formalised.

## Source semantics
* grid = 24 blocks (one per query head), block = (32, 4) = 128 threads; kv head = head / 12.
* The step is valid iff 1 <= width <= max_context, n_kv = width, pos = width-1 and n_bid = width/4; otherwise the
  output is NaN and status = unsupported.
* 128 cells per iteration: cell = base + warp*32 + (lane & ~7) + row; padded length 256 => two iterations.
* Each lane handles dimensions d = 2*(32a + 4*(lane%8) + j) and d+1 (a < 4, j < 4).
* The epilogue sums 16 per-thread partial tiles (4 warps x 4 lane groups) and divides by the softmax sum.

## Form template identifiers
stepValid -> step_validity, kvIdx -> kv_element_index, tileIdx -> partial_tile_slot
-/
namespace Strata.NativeFlashAttnShort

-- § step validity
/-- The validity predicate checked on-device. -/
def stepValid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos = width - 1 ∧ nbid = width / 4

/-- A valid step addresses only cells strictly below the width. -/
theorem valid_pos_lt (w n p b m : Nat) (h : stepValid w n p b m) : p < w := by
  unfold stepValid at h
  omega

/-- With max context <= 256 a valid step fits the padded length 256. -/
theorem valid_fits_padded (w n p b m : Nat) (h : stepValid w n p b m) (hm : m ≤ 256) : w ≤ 256 := by
  unfold stepValid at h
  omega

/-- A valid step's width is at least one (non-empty context). -/
theorem valid_nonempty (w n p b m : Nat) (h : stepValid w n p b m) : 0 < w := by
  unfold stepValid at h
  omega

/-- Width 1..max_context: the dense step of cell c is valid exactly when c+1 <= max_context. -/
theorem dense_step_valid_iff (c m : Nat) :
    stepValid (c + 1) (c + 1) c ((c + 1) / 4) m ↔ c + 1 ≤ m := by
  unfold stepValid
  omega

-- § head and kv layout
/-- KV head of a query head. -/
def kvHead (head : Nat) : Nat := head / 12

/-- All 24 query heads map to one of the two KV heads. -/
theorem kvHead_lt (head : Nat) (h : head < 24) : kvHead head < 2 := by
  unfold kvHead
  omega

/-- Heads 0..11 use KV head 0 and 12..23 use KV head 1. -/
theorem kvHead_split (head : Nat) (h : head < 24) :
    (head < 12 → kvHead head = 0) ∧ (12 ≤ head → kvHead head = 1) := by
  unfold kvHead
  constructor <;> intro hh <;> omega

/-- Element index of K/V: (cell*2 + kv)*256 + d. -/
def kvIdx (cell kv d : Nat) : Nat := (cell * 2 + kv) * 256 + d

/-- K/V reads stay in capacity*512 halves, which is exactly the validated `capacity*1024` bytes. -/
theorem kvIdx_lt (cell kv d cap : Nat) (hc : cell < cap) (hkv : kv < 2) (hd : d < 256) :
    kvIdx cell kv d < cap * 512 := by
  unfold kvIdx
  omega

/-- K/V element indexing is injective. -/
theorem kvIdx_inj (c k d c' k' d' : Nat) (hk : k < 2) (hk' : k' < 2) (hd : d < 256) (hd' : d' < 256)
    (h : kvIdx c k d = kvIdx c' k' d') : c = c' ∧ k = k' ∧ d = d' := by
  unfold kvIdx at h
  omega

/-- Two bytes per half: capacity*512 halves = capacity*1024 bytes. -/
theorem kv_bytes (cap : Nat) : cap * 512 * 2 = cap * 1024 := by omega

/-- Query/output row of a head: head*256 + d is in 24*256. -/
theorem q_idx_lt (head d : Nat) (hh : head < 24) (hd : d < 256) : head * 256 + d < 24 * 256 := by omega

-- § dimension coverage per lane
/-- Pair index handled by (a, lane%8, j): 32a + 4*l8 + j. -/
def pairIdx (a l8 j : Nat) : Nat := 32 * a + 4 * l8 + j

/-- Pair indices cover 0..127, so dimensions d = 2p and 2p+1 cover 0..255. -/
theorem pair_cover (p : Nat) (hp : p < 128) : ∃ a l8 j, a < 4 ∧ l8 < 8 ∧ j < 4 ∧ pairIdx a l8 j = p :=
  ⟨p / 32, (p % 32) / 4, p % 4, by omega, by omega, by omega, by unfold pairIdx; omega⟩

/-- Each pair index is produced by exactly one (a, lane%8, j). -/
theorem pair_inj (a l j a' l' j' : Nat) (hl : l < 8) (hl' : l' < 8) (hj : j < 4) (hj' : j' < 4)
    (h : pairIdx a l j = pairIdx a' l' j') : a = a' ∧ l = l' ∧ j = j' := by
  unfold pairIdx at h
  omega

/-- The d+1 element read next to d = 2p stays in range. -/
theorem dim_pair_lt (p : Nat) (hp : p < 128) : 2 * p + 1 < 256 := by omega

/-- qreg slot 4a + j is below 16. -/
theorem qreg_slot_lt (a j : Nat) (ha : a < 4) (hj : j < 4) : 4 * a + j < 16 := by omega

-- § cell coverage per iteration
/-- Cell offset inside a 128-cell iteration: warp*32 + (lane/8)*8 + row (the `lane & ~7` term is (lane/8)*8). -/
def cellOff (warp lane row : Nat) : Nat := warp * 32 + (lane / 8) * 8 + row

/-- `lane & ~7` equals lane minus lane mod 8. -/
theorem lane_mask_eq (lane : Nat) : (lane / 8) * 8 = lane - lane % 8 := by omega

/-- Cell offsets stay inside the 128 cells of an iteration. -/
theorem cellOff_lt (warp lane row : Nat) (hw : warp < 4) (hl : lane < 32) (hr : row < 8) :
    cellOff warp lane row < 128 := by
  unfold cellOff
  omega

/-- Every cell of the iteration is handled by some (warp, lane group, row). -/
theorem cellOff_cover (x : Nat) (hx : x < 128) :
    ∃ warp g row, warp < 4 ∧ g < 4 ∧ row < 8 ∧ warp * 32 + g * 8 + row = x :=
  ⟨x / 32, (x % 32) / 8, x % 8, by omega, by omega, by omega, by omega⟩

/-- Two iterations of 128 cover the padded length 256. -/
theorem iterations_cover : 2 * 128 = 256 := by decide

/-- Lane groups of 8 are closed under the xor offsets 1, 2, 4 of `warp_sum<8>`. -/
theorem xor_group_closed : ∀ l : Fin 32, ∀ o ∈ [1, 2, 4], ((l.val ^^^ o) / 8 = l.val / 8) := by
  decide

-- § epilogue
/-- Output slot of the epilogue: i0 in {0, 128} plus tid < 128 covers 0..255 exactly. -/
theorem out_cover (d : Nat) (hd : d < 256) :
    ∃ i0 tid, (i0 = 0 ∨ i0 = 128) ∧ tid < 128 ∧ i0 + tid = d := by
  by_cases h : d < 128
  · exact ⟨0, d, Or.inl rfl, h, by omega⟩
  · exact ⟨128, d - 128, Or.inr rfl, by omega, by omega⟩

/-- Partial-tile slot written before the final sum: warp*1024 + group*256 + d. -/
def tileIdx (warp g d : Nat) : Nat := warp * 4 * 256 + g * 256 + d

/-- Tile slots stay inside the 16*256 = 4096 shared floats. -/
theorem tileIdx_lt (warp g d : Nat) (hw : warp < 4) (hg : g < 4) (hd : d < 256) : tileIdx warp g d < 4 * 4 * 256 := by
  unfold tileIdx
  omega

/-- Tile slots are injective in (warp, group, d). -/
theorem tileIdx_inj (w g d w' g' d' : Nat) (hg : g < 4) (hg' : g' < 4) (hd : d < 256) (hd' : d' < 256)
    (h : tileIdx w g d = tileIdx w' g' d') : w = w' ∧ g = g' ∧ d = d' := by
  unfold tileIdx at h
  omega

/-- Sixteen partial tiles are summed per output dimension: 4 warps x 4 groups. -/
theorem partial_tiles : 4 * 4 = 16 := by decide

/-- The score scale 1/16 squared times D = 256 is one: scale = 1/sqrt(256). -/
theorem scale_sq : 16 * 16 = 256 := by decide

-- § host validation
/-- Launch shape: 24 blocks of 32x4 = 128 threads, matching `__launch_bounds__(128, 1)`. -/
theorem block_threads : 32 * 4 = 128 := by decide

/-- Each of the 128 threads writes two output dimensions: 128 * 2 = 256. -/
theorem outputs_per_head : 128 * 2 = 256 := by decide

/-- Byte-span overlap used by the host check. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> intro h <;> omega

/-- Adjacent spans never overlap. -/
theorem adjacent_disjoint (a n m : Nat) : ¬ overlap a n (a + n) m := by
  unfold overlap
  omega

end Strata.NativeFlashAttnShort
