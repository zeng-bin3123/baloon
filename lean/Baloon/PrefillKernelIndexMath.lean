/-!
# MODULE: prefill_kernel_index_math
Index and layout arithmetic of the prefill CUDA kernels: expert blob dequantizer, router tie-break and lane layout,
KV-append addressing, RoPE pairing and launch sizing.

## Source semantics
* Blob: gate/up codes `[1280][640 B]`, down codes `[2560][160 B]`, gate/up scales `[1280][40]` f16, down scales `[2560][10]` f16; 1,382,400 bytes.
* Router scans 512 experts as 32 lanes x 16 values (`expert = lane + 32*i`), ties go to the smaller index.
* KV append: `d = g*64 + tid`. RoPE rotates pairs `(p, p+32)`.

## Form template identifiers
* `O_D_CODES`, `O_GU_SC`, `O_D_SC` ↔ constants in `blob_dequant_kernel`; `better` ↔ `(ob > best || (ob == best && oi < ex))`; `blocksFor` ↔ `blocks_for`
-/

namespace Strata.PrefillKernels

/-- Offset of down codes. -/
def O_D_CODES : Nat := 1280 * 640
/-- Offset of gate/up scales. -/
def O_GU_SC : Nat := O_D_CODES + 2560 * 160
/-- Offset of down scales. -/
def O_D_SC : Nat := O_GU_SC + 1280 * 40 * 2
/-- Total blob size. -/
def BLOB : Nat := O_D_SC + 2560 * 10 * 2

/-- Every expert blob is 1,382,400 bytes. -/
theorem blob_total : BLOB = 1382400 := by decide

/-- Regions are laid out back to back without overlap. -/
theorem regions_ordered : O_D_CODES ≤ O_GU_SC ∧ O_GU_SC ≤ O_D_SC ∧ O_D_SC ≤ BLOB := by decide

/-- Gate/up thread: byte index and scale index stay in range. -/
theorem gu_thread_bounds (i : Nat) (h : i < 1280 * 640) :
    i / 640 < 1280 ∧ i % 640 < 640 ∧ (i / 640) * 40 + (i % 640 * 4) / 64 < 1280 * 40 := by
  have h1 : i / 640 < 1280 := by omega
  have h2 : i % 640 < 640 := by omega
  have h3 : (i % 640 * 4) / 64 < 40 := by omega
  refine ⟨h1, h2, ?_⟩
  omega

/-- Gate/up output index stays inside `[1280][2560]`. -/
theorem gu_out_bounds (i k : Nat) (h : i < 1280 * 640) (hk : k < 4) :
    (i / 640) * 2560 + (i % 640) * 4 + k < 1280 * 2560 := by
  have h1 : i / 640 < 1280 := by omega
  have h2 : i % 640 < 640 := by omega
  omega

/-- Down thread: scale and output indices stay in range. -/
theorem d_thread_bounds (j k : Nat) (h : j < 2560 * 160) (hk : k < 4) :
    (j / 160) * 10 + (j % 160 * 4) / 64 < 2560 * 10 ∧
    (j / 160) * 640 + (j % 160) * 4 + k < 2560 * 640 := by
  have h1 : j / 160 < 2560 := by omega
  have h2 : j % 160 < 160 := by omega
  have h3 : (j % 160 * 4) / 64 < 10 := by omega
  constructor <;> omega

/-- Both dequantized layouts hold one fp16 per weight. -/
theorem weights_per_expert : 1280 * 2560 = 3276800 ∧ 2560 * 640 = 1638400 := by decide

/-- A code decodes to `(c-1)` in `[-1, 2]`. -/
theorem code_offset_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by omega

/-- The scan covers 512 experts as 32 lanes of 16 values. -/
theorem router_cover : 32 * 16 = 512 := by decide

/-- Expert index from (lane, i) is in range. -/
theorem expert_index_lt (lane i : Nat) (hl : lane < 32) (hi : i < 16) : lane + i * 32 < 512 := by omega

/-- And decomposes back. -/
theorem expert_index_decompose (lane i : Nat) (hl : lane < 32) :
    (lane + i * 32) / 32 = i ∧ (lane + i * 32) % 32 = lane := by omega

/-- Tie-break order: larger value wins, equal values prefer the smaller index. -/
def better (ob : Int) (oi : Nat) (b : Int) (e : Nat) : Prop := ob > b ∨ (ob = b ∧ oi < e)

/-- `better` is asymmetric. -/
theorem better_asymm (ob b : Int) (oi e : Nat) (h : better ob oi b e) : ¬ better b e ob oi := by
  unfold better at *
  omega

/-- Distinct candidates are always strictly ordered one way. -/
theorem better_total (ob b : Int) (oi e : Nat) (hne : oi ≠ e) : better ob oi b e ∨ better b e ob oi := by
  unfold better
  omega

/-- Top-k is at most the expert count. -/
theorem topk_le : 10 ≤ 512 := by decide

/-- `d = g*64 + tid` covers 0..255. -/
theorem kv_dim_in_range (g tid : Nat) (hg : g < 4) (ht : tid < 64) : g * 64 + tid < 256 := by omega

/-- The z-grid of 8 splits into 4 groups x (K,V). -/
theorem kv_grid_split (z : Nat) (h : z < 8) : z / 2 < 4 ∧ z % 2 < 2 := by omega

/-- The position within a page is below the page size. -/
theorem pos_in_page (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt pos h

/-- The scale index `row*4 + g` is below four per row. -/
theorem scale_idx_lt (row g rows : Nat) (hr : row < rows) (hg : g < 4) : row * 4 + g < rows * 4 := by omega

/-- Rotated pair partners are distinct and inside the 64-wide rotary span. -/
theorem rope_pair (p : Nat) (h : p < 32) : p ≠ p + 32 ∧ p + 32 < 64 := by omega

/-- The pair indices cover every index below 64. -/
theorem rope_cover (k : Nat) (h : k < 64) : (k < 32 ∧ k = k) ∨ (32 ≤ k ∧ k - 32 < 32) := by omega

/-- Number of blocks for `n` elements at 256 threads per block. -/
def blocksFor (n : Nat) : Nat := (n + 255) / 256

/-- The launch covers every element. -/
theorem blocksFor_covers (n : Nat) : n ≤ blocksFor n * 256 := by
  unfold blocksFor
  omega

/-- It never launches a whole extra block of idle threads. -/
theorem blocksFor_tight (n : Nat) : blocksFor n * 256 < n + 256 := by
  unfold blocksFor
  omega

/-- GDN channel count: q + k + v. -/
theorem gdn_channels : 16 * 128 + 16 * 128 + 48 * 128 = 10240 := by decide

/-- Hyper-connection width. -/
theorem hc_width : 2560 * 4 = 10240 := by decide

/-- The recurrence splits 128 rows across 4 groups of 32. -/
theorem rec_split : 128 / 4 = 32 ∧ 4 * 32 = 128 := by decide

end Strata.PrefillKernels
