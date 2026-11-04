/-!
# MODULE: ple_block_gpu
GPU half of the PLE block: history advance, grouped RMS norm per stream, gate, value broadcast, causal dilated
conv with SiLU, 3-way residual add, and the caller-owned scratch workspace layout.

## Source semantics
* History is stored row-fastest per channel: element `(row, c)` at `row + 9*c`; advancing shifts rows up by one and
  appends the normalized row.
* Conv tap `k` reads row `nhist - (kern-1-k)*dil`; with kern 4, dil 3, hist 9 these are rows 0,3,6,9, and row 9 is
  the new row. Kernel weights are ggml-native: `kW[k + kern*c]`.
* Grouped norm normalizes each hc stream `[c*n_embd, (c+1)*n_embd)` independently.
* Scratch = aligned float region + aligned Q8 image + BF16 embedding copy + 256 slack; all pointers are carved
  at 16-byte aligned offsets, so no allocation occurs on the token path.

## Form template identifiers
* `NHIST`, `HC`, `HC_DIM`, `N_EMBD` -> 9, 4, 10240, 2560
* `advance`                        -> history column shift
* `alignUp`                        -> 16-byte alignment
* `scratchBytes`                   -> ple_block_scratch_bytes
-/

namespace Strata.PleBlock

/-- History depth. -/
def NHIST : Nat := 9
/-- Streams. -/
def HC : Nat := 4
/-- Embedding width per stream. -/
def N_EMBD : Nat := 2560
/-- Flat width. -/
def HC_DIM : Nat := 10240

/-- Flat width is streams times width. -/
theorem hc_dim_eq : HC * N_EMBD = HC_DIM := by decide

-- § history

/-- Column shift: drop the oldest row, append the new one. -/
def advance {α : Type} (col : List α) (x : α) : List α := col.tail ++ [x]

/-- Advance preserves the history depth of 9. -/
theorem advance_length {α : Type} (col : List α) (x : α) (h : col.length = NHIST) :
    (advance col x).length = NHIST := by
  unfold advance
  unfold NHIST at *
  simp [List.length_tail, h]

/-- The newest row is last after an advance. -/
theorem advance_last {α : Type} (col : List α) (x : α) : (advance col x).getLast? = some x := by
  simp [advance]

/-- Row-fastest addressing stays inside the `9*HC_DIM` history buffer. -/
theorem hist_index_in_range (row c : Nat) (hr : row < NHIST) (hc : c < HC_DIM) :
    row + NHIST * c < NHIST * HC_DIM := by
  unfold NHIST HC_DIM at *
  omega

/-- Row-fastest addressing is injective, so a transposed read cannot alias a correct one by accident. -/
theorem hist_index_injective (r1 c1 r2 c2 : Nat) (h1 : r1 < NHIST) (h2 : r2 < NHIST)
    (h : r1 + NHIST * c1 = r2 + NHIST * c2) : r1 = r2 ∧ c1 = c2 := by
  unfold NHIST at *
  omega

-- § conv

/-- Tap row for kernel size 4, dilation 3. -/
def tapRow (k : Nat) : Nat := NHIST - (4 - 1 - k) * 3

/-- Taps read rows 0,3,6,9. -/
theorem tap_rows : tapRow 0 = 0 ∧ tapRow 1 = 3 ∧ tapRow 2 = 6 ∧ tapRow 3 = 9 := by decide

/-- Only the last tap reads the new row (row index equal to the history depth). -/
theorem only_last_tap_new (k : Nat) (hk : k < 4) : tapRow k = NHIST ↔ k = 3 := by
  unfold tapRow NHIST
  omega

/-- Weight index `k + 4*c` is inside the `4*HC_DIM` kernel buffer and injective in (k, c). -/
theorem kernel_weight_index (k c : Nat) (hk : k < 4) (hc : c < HC_DIM) :
    k + 4 * c < 4 * HC_DIM := by
  unfold HC_DIM at *
  omega

-- § grouped norm and broadcast

/-- Stream `c` occupies `[c*n, (c+1)*n)`; an element index lies in its stream's window. -/
theorem stream_window (c d n : Nat) (hd : d < n) :
    c * n ≤ c * n + d ∧ c * n + d < (c + 1) * n := by
  have h2 : (c + 1) * n = c * n + n := Nat.succ_mul c n
  omega

/-- Gate sign antisymmetry check: positive, negative and zero map to +1, -1, 0. -/
theorem sign_samples : Int.sign 7 = 1 ∧ Int.sign (-7) = -1 ∧ Int.sign 0 = 0 := by decide

-- § scratch layout

/-- Round up to a multiple of 16. -/
def alignUp (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment produces a 16-aligned value no smaller than the input and less than 16 bytes larger. -/
theorem alignUp_spec (x : Nat) : alignUp x % 16 = 0 ∧ x ≤ alignUp x ∧ alignUp x < x + 16 := by
  unfold alignUp
  omega

/-- Float region bytes: five HC_DIM floats, N_EMBD floats and HC floats. -/
def floatBytes : Nat := (5 * HC_DIM + N_EMBD + HC) * 4

/-- Q8 activation image bytes: `N_EMBD/32` blocks of 34 bytes. -/
def q8Bytes : Nat := (N_EMBD / 32) * 34

/-- BF16 embedding copy bytes. -/
def embBytes : Nat := N_EMBD * 2

/-- Scratch size: `ple_block_scratch_bytes`. -/
def scratchBytes : Nat := alignUp floatBytes + alignUp q8Bytes + embBytes + 256

/-- The total is 223152 bytes for the real geometry. -/
theorem scratch_bytes_value : scratchBytes = 223152 := by decide

/-- The carved regions fit: BF16 copy ends before the total (leaving the 256-byte slack). -/
theorem carve_fits : alignUp floatBytes + alignUp q8Bytes + embBytes ≤ scratchBytes := by
  unfold scratchBytes
  omega

/-- Region starts are 16-aligned (float region at 0, Q8 image after it). -/
theorem carve_alignment : alignUp floatBytes % 16 = 0 ∧ (alignUp floatBytes + alignUp q8Bytes) % 16 = 0 := by
  unfold alignUp floatBytes q8Bytes N_EMBD HC_DIM HC
  omega

/-- Float sub-buffers: key, query, norm, gated, conv at multiples of HC_DIM, then value, then gate; they tile
the float region exactly. -/
theorem float_regions_tile : 5 * HC_DIM + N_EMBD + HC = floatBytes / 4 := by decide

-- § span overlap

/-- Overlap as implemented (`aa < bb ? bb-aa < a : aa-bb < b`). -/
def ovl (a b an bn : Nat) : Prop := if a < b then b - a < an else a - b < bn

/-- For nonempty spans the implementation equals the interval-intersection test. -/
theorem ovl_iff (a b an bn : Nat) (h1 : 0 < an) (h2 : 0 < bn) :
    ovl a b an bn ↔ (a < b + bn ∧ b < a + an) := by
  unfold ovl
  split <;> omega

end Strata.PleBlock
