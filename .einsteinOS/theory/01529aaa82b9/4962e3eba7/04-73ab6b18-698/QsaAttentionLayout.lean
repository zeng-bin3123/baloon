/-!
# MODULE: qsa_attention_layout
Decode-time bookkeeping of the QSA (sparse full-attention) layer: selection width, per-token step
state, tail/spare-slot cell mapping, GQA head map, paged KV addressing, top-k tie order, gate split,
memory budgets.

## Source semantics
Selection width = min(n_kv, idx_top_k + idx_block - 1). The step buffer holds (pos, n_kv = pos+1,
n_bid = n_kv / r, width). Cells of the incomplete tail map to the spare slot at row n_bid. The q to
kv head map is integer division. KV rows are `[page][kv_head][page_size][head_dim]`. Top-k ties
break by ascending index.

## Form template identifiers
selWidth -> qsa_selection_width ; stepFill -> qsa_step_fill ; cellBlock -> cell_block ;
before -> topk_order ; physRow -> kv_physical_row
-/

set_option maxRecDepth 100000

namespace Strata.Qsa

-- § Selection width

/-- `min(n_kv, idx_top_k + idx_block - 1)`. -/
def selWidth (n k r : Nat) : Nat := min n (k + r - 1)

/-- Never selects more cells than exist. -/
theorem sel_le_nkv (n k r : Nat) : selWidth n k r ≤ n := by unfold selWidth; omega

/-- Never exceeds the budget plus the incomplete tail. -/
theorem sel_le_cap (n k r : Nat) : selWidth n k r ≤ k + r - 1 := by unfold selWidth; omega

/-- Below the bound the selection is the identity: every cell is selected. -/
theorem sel_identity (n k r : Nat) (h : n ≤ k + r - 1) : selWidth n k r = n := by
  unfold selWidth; omega

/-- Width is monotone in the context length. -/
theorem sel_mono (n m k r : Nat) (h : n ≤ m) : selWidth n k r ≤ selWidth m k r := by
  unfold selWidth; omega

/-- Real artifact: the bound is 2051, not the plan's 2048. -/
theorem sel_real : selWidth 100000 2048 4 = 2051 ∧ selWidth 2051 2048 4 = 2051 ∧
    selWidth 2050 2048 4 = 2050 ∧ selWidth 2048 2048 4 = 2048 := by decide

/-- The capturable top-k capacity is the width at the 32768-cell cap. -/
theorem sel_cap_at_max : selWidth 32768 2048 4 = 2051 := by decide

-- § Step state

/-- Per-token device step buffer (pos, n_kv, n_bid, width). -/
structure Step where
  pos : Nat
  nkv : Nat
  nbid : Nat
  width : Nat

/-- All four entries are derived from `pos`, by one function. -/
def stepFill (pos k r : Nat) : Step := ⟨pos, pos + 1, (pos + 1) / r, selWidth (pos + 1) k r⟩

/-- n_kv = pos + 1. -/
theorem step_nkv (pos k r : Nat) : (stepFill pos k r).nkv = pos + 1 := rfl

/-- The width never exceeds n_kv. -/
theorem step_width_le (pos k r : Nat) : (stepFill pos k r).width ≤ (stepFill pos k r).nkv := by
  unfold stepFill; exact sel_le_nkv _ _ _

/-- Completed blocks cover at most n_kv cells, and fewer than r cells remain. -/
theorem step_blocks (pos k r : Nat) (hr : 0 < r) :
    (stepFill pos k r).nbid * r ≤ (stepFill pos k r).nkv ∧
    (stepFill pos k r).nkv < (stepFill pos k r).nbid * r + r := by
  unfold stepFill
  simp only
  have h1 := Nat.div_add_mod (pos + 1) r
  have h2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.mul_comm ((pos + 1) / r) r]
  omega

/-- The buffer is four int32: 16 bytes. -/
theorem step_bytes : 4 * 4 = 16 := rfl

/-- A block completes exactly when pos % 4 = 3. -/
theorem complete_iff (pos : Nat) : (pos + 1) % 4 = 0 ↔ pos % 4 = 3 := by omega

/-- n_bid at position pos is (pos+1)/4; it grows from pos-1 to pos exactly when pos % 4 = 3. -/
theorem nbid_grows (pos : Nat) (h : pos % 4 = 3) : (pos + 1) / 4 = pos / 4 + 1 := by omega

/-- At every other position n_bid is unchanged from the previous token. -/
theorem nbid_stable (pos : Nat) (h : pos % 4 ≠ 3) : (pos + 1) / 4 = pos / 4 := by omega

/-- The pooled block is rotated at its FIRST cell: pos - 3 = (pos / 4) * 4 on completion. -/
theorem rotate_first_cell (pos : Nat) (h : pos % 4 = 3) : pos - 3 = pos / 4 * 4 := by omega

/-- First cell and last cell differ by 3, so rotating at the last would be a different position. -/
theorem first_ne_last (pos : Nat) (h : pos % 4 = 3) : pos - 3 ≠ pos := by omega

-- § Cell to block map

/-- Cell j maps to block j / r, clamped to the spare slot at row n_bid. -/
def cellBlock (j nbid r : Nat) : Nat := min (j / r) nbid

/-- Cells of completed blocks map to their own block. -/
theorem cell_completed (j nbid r : Nat) (hr : 0 < r) (h : j < nbid * r) :
    cellBlock j nbid r = j / r := by
  have : j / r < nbid := (Nat.div_lt_iff_lt_mul hr).2 h
  unfold cellBlock; omega

/-- Tail cells (those in the last incomplete block) all map to the spare slot at row n_bid. -/
theorem cell_tail (j nkv r : Nat) (hr : 0 < r) (h1 : nkv / r * r ≤ j) (h2 : j < nkv) :
    cellBlock j (nkv / r) r = nkv / r := by
  have hlt : nkv < (nkv / r + 1) * r := by
    have a := Nat.div_add_mod nkv r
    have b := Nat.mod_lt nkv hr
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm (nkv / r) r]
    omega
  have : j / r = nkv / r := Nat.div_eq_of_lt_le h1 (by omega)
  unfold cellBlock; omega

/-- Fewer than r cells are in the tail; with r = 4 the raw tail holds idx_block - 1 = 3 rows. -/
theorem tail_size (nkv : Nat) : nkv % 4 ≤ 3 := by omega

-- § GQA head map

/-- Query head q maps to kv head q / group, which is in range. -/
theorem gqa_in_range (q nh nkv g : Nat) (hg : 0 < g) (hn : nh = nkv * g) (hq : q < nh) :
    q / g < nkv := by
  apply (Nat.div_lt_iff_lt_mul hg).2
  rw [hn] at hq
  exact hq

/-- Real shapes: 24 heads, 2 kv heads, group 12; every q/12 is 0 or 1. -/
theorem gqa_real : ∀ q < 24, q / 12 < 2 := by intro q h; omega

/-- Integer division and modulo give different maps (head 1 differs): the choice is observable. -/
theorem gqa_div_ne_mod : ∃ q < 24, q / 12 ≠ q % 2 := ⟨1, by omega, by omega⟩

-- § Paged KV addressing

/-- Row-major bound used by every paged index. -/
theorem rowmajor_lt {i n j m : Nat} (hi : i < n) (hj : j < m) : i * m + j < n * m := by
  have h1 : (i + 1) * m ≤ n * m := Nat.mul_le_mul_right m hi
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Physical row of (page, head, in-page offset). -/
def physRow (page h off nkv ps : Nat) : Nat := (page * nkv + h) * ps + off

/-- Every physical row lies in the pool of nPages pages. -/
theorem phys_row_lt (page h off nPages nkv ps : Nat) (hp : page < nPages) (hh : h < nkv)
    (ho : off < ps) : physRow page h off nkv ps < nPages * nkv * ps := by
  unfold physRow
  have h1 : page * nkv + h < nPages * nkv := rowmajor_lt hp hh
  exact rowmajor_lt h1 ho

/-- Within a page one head's cells are consecutive: offset +1 is row +1. -/
theorem phys_row_consec (page h off nkv ps : Nat) :
    physRow page h (off + 1) nkv ps = physRow page h off nkv ps + 1 := by
  unfold physRow; omega

/-- A logical cell splits into (page, offset) and recombines. -/
theorem cell_split (t ps : Nat) : (t / ps) * ps + t % ps = t := by
  rw [Nat.mul_comm]; exact Nat.div_add_mod t ps

-- § Top-k order

/-- i precedes j: higher score first, ties by ascending index. -/
def before (s : Nat → Int) (i j : Nat) : Prop := s j < s i ∨ (s i = s j ∧ i < j)

/-- Irreflexive. -/
theorem before_irrefl (s : Nat → Int) (i : Nat) : ¬ before s i i := by
  unfold before; omega

/-- Total on distinct cells: the selection is deterministic. -/
theorem before_total (s : Nat → Int) (i j : Nat) (h : i ≠ j) : before s i j ∨ before s j i := by
  unfold before; omega

/-- Asymmetric. -/
theorem before_asymm (s : Nat → Int) (i j : Nat) (h : before s i j) : ¬ before s j i := by
  unfold before at *; omega

/-- Transitive. -/
theorem before_trans (s : Nat → Int) (i j k : Nat) (h1 : before s i j) (h2 : before s j k) :
    before s i k := by
  unfold before at *; omega

-- § Gate split and budgets

/-- Raw q projection is per head `[q | gate]`: q at h*2hd + d, gate at h*2hd + hd + d, disjoint. -/
theorem gate_split (h d hd : Nat) (hdd : d < hd) :
    h * (2 * hd) + d < h * (2 * hd) + hd + d := by omega

/-- Both halves stay inside n_head * 2 * head_dim. -/
theorem gate_in_range (h d hd nh : Nat) (hh : h < nh) (hdd : d < hd) :
    h * (2 * hd) + hd + d < nh * (2 * hd) := by
  have := rowmajor_lt (m := 2 * hd) hh (show hd + d < 2 * hd by omega)
  omega

/-- Real raw projection width: 24 * 512. -/
theorem qfull_width : 24 * (2 * 256) = 12288 := rfl

/-- KV cache at 32K: 805,306,368 bytes, which is 768 MiB (the header says "805 MiB": 805 MB). -/
theorem kv_bytes : 12 * 2 * 256 * 2 * 2 * 32768 = 805306368 ∧ 805306368 = 768 * 1048576 := by decide

/-- The same in f32 is exactly double. -/
theorem kv_bytes_f32 : 12 * 2 * 256 * 2 * 4 * 32768 = 2 * 805306368 := by decide

/-- Indexer keys: 128 floats = 512 B per block, 128 B per token, 1536 B per token over 12 layers,
and 50,331,648 B = 48 MiB at 32K. -/
theorem indexer_bytes : 128 * 4 = 512 ∧ 512 / 4 = 128 ∧ 128 * 12 = 1536 ∧
    1536 * 32768 = 50331648 ∧ 50331648 = 48 * 1048576 := by decide

/-- Real geometry sanity: heads divide, head_dim multiple of 4, rotary dims fit both widths. -/
theorem real_shapes : 24 % 2 = 0 ∧ 256 % 4 = 0 ∧ 64 ≤ 256 ∧ 64 ≤ 128 ∧ 2048 % 4 = 0 := by decide

end Strata.Qsa
