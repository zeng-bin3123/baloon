/-!
# MODULE: native_gdn_preprocess
GDN preprocessing kernels adapted from llama.cpp: 4-tap conv + SiLU with a 3-deep history, L2/RMS norm
over 128-wide rows, beta sigmoid, gate softplus and the output RMSNorm with sigmoid gate.  This file
proves the history-window semantics, the warp-slot arithmetic of the block reductions, the launch
coverage and the host span checks.

## Source semantics
* History holds the last three inputs; the four taps are `[h0,h1,h2,x]` and the new history is `[h1,h2,x]`.
* Norm kernels run 256 threads per row on a width-128 row; threads 128..255 contribute zero.
* `count_and_stream` accepts counts in `[1, 65535]`.

## Form template identifiers
- `conv_silu` -> `conv_silu_step`; `l2_norm` -> `row_l2`; `out_norm` -> `out_rms_gate`
- `norm_sum` -> `block_sum`; `gate_softplus` -> `gate_from_alpha`; `beta_sigmoid` -> `beta_sigmoid`
-/

namespace Strata.NativeGdnPreprocess

-- § history window

/-- History of three inputs. -/
abbrev Hist := Int × Int × Int

/-- New history after one step: shift left, append the new input. -/
def slide (h : Hist) (x : Int) : Hist := (h.2.1, h.2.2, x)

/-- The four-tap sum, oldest history entry on tap 0 and the new input on tap 3. -/
def taps (h : Hist) (x w0 w1 w2 w3 : Int) : Int := h.1 * w0 + h.2.1 * w1 + h.2.2 * w2 + x * w3

/-- After three consecutive inputs the history holds exactly those inputs, oldest first. -/
theorem slide_three (h : Hist) (x0 x1 x2 : Int) : slide (slide (slide h x0) x1) x2 = (x0, x1, x2) := rfl

/-- The newest input becomes the last history entry. -/
theorem slide_last (h : Hist) (x : Int) : (slide h x).2.2 = x := rfl

/-- The oldest entry is dropped and the middle entries move down. -/
theorem slide_shift (h : Hist) (x : Int) : (slide h x).1 = h.2.1 ∧ (slide h x).2.1 = h.2.2 := ⟨rfl, rfl⟩

/-- The tap sum spelled out for the all-ones weights. -/
theorem taps_ones (h : Hist) (x : Int) : taps h x 1 1 1 1 = h.1 + h.2.1 + h.2.2 + x := by
  unfold taps
  omega

-- § addressing

/-- History addressing `history[c*3 + tap]` stays in the `3 * channels` buffer. -/
theorem history_index_lt {c tap channels : Nat} (hc : c < channels) (ht : tap < 3) :
    c * 3 + tap < channels * 3 := by omega

/-- Weight addressing `weights[c*4 + tap]` stays in the `4 * channels` buffer. -/
theorem weight_index_lt {c tap channels : Nat} (hc : c < channels) (ht : tap < 4) :
    c * 4 + tap < channels * 4 := by omega

/-- A 256-thread grid of `ceil(channels/256)` blocks covers every channel. -/
theorem channel_grid_cover {channels c : Nat} (hc : c < channels) : c / 256 < (channels + 255) / 256 := by omega

-- § block reduction slots

/-- With 256 threads the warp slot is below 8, matching the `lane < 8` read of `sums`. -/
theorem warp_slot_lt {t : Nat} (ht : t < 256) : t / 32 < 8 := by omega

/-- The `sums[32]` array is large enough for every warp of a 256-thread block. -/
theorem sums_capacity {t : Nat} (ht : t < 256) : t / 32 < 32 := by omega

/-- Threads beyond the row width (128..255) are in warps 4..7, which contribute zero partial sums. -/
theorem padded_threads_warps {t : Nat} (h1 : 128 ≤ t) (h2 : t < 256) : 4 ≤ t / 32 ∧ t / 32 < 8 := by omega

-- § launch validation

/-- A count accepted by `count_and_stream`. -/
def validCount (c : Nat) : Prop := 0 < c ∧ c ≤ 65535

/-- Accepted counts need between 1 and 256 blocks of 256 threads. -/
theorem valid_count_blocks {c : Nat} (h : validCount c) : 1 ≤ (c + 255) / 256 ∧ (c + 255) / 256 ≤ 256 := by
  unfold validCount at h
  omega

/-- Rows of width 128 occupy `rows * 128` floats; the byte size is `rows * 512`. -/
theorem norm_bytes_eq (rows : Nat) : rows * 128 * 4 = rows * 512 := by omega

-- § span overlap predicate

/-- Half-open byte ranges flagged as overlapping. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap predicate is symmetric. -/
theorem overlap_comm {ap an bp bn : Nat} : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap
  omega

/-- Ranges laid end to end are never flagged. -/
theorem not_overlap_of_adjacent {ap an bp bn : Nat} (h : ap + an ≤ bp) : ¬ overlap ap an bp bn := by
  unfold overlap
  omega

/-- A history span of `3 * bytes` placed directly before a span of `bytes` is accepted. -/
theorem history_then_raw_disjoint (p bytes : Nat) : ¬ overlap p (3 * bytes) (p + 3 * bytes) bytes := by
  unfold overlap
  omega

-- § gate softplus branch

/-- Overflow-safe softplus selector: above 20 the input is returned unchanged. -/
def softplusSel (f : Int → Int) (v : Int) : Int := if v > 20 then v else f v

/-- Above the threshold the large-argument branch is taken. -/
theorem softplusSel_large (f : Int → Int) {v : Int} (h : 20 < v) : softplusSel f v = v := by
  unfold softplusSel
  split <;> omega

end Strata.NativeGdnPreprocess
