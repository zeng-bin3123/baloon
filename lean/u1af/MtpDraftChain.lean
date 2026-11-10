/-!
# MODULE: mtp_draft_chain
SOURCES: include/ref/core/mtp.hpp
The MTP draft layer on the GPU: draft cell placement, the attention window, and the graph-array bounds.

## Source semantics
* A round drafts `T-1` tokens at cells `p+a+1, p+a+2, ...` after the last accepted row `a`.
* Attention is over the last `window` cells (`0` = every cell).
* Per-`T` graph arrays have 9 entries, so `T <= 8`.
* All 512 routed experts of the draft layer sit in VRAM (about 708 MB).

## Form template identifiers
MtpDrafter -> draft_engine; draft -> draft_round; draft_first -> first_round; prefill -> prompt_catchup;
window_ -> attention_window; step_exec_ -> per_step_graphs.
-/

namespace MtpDraftChain

-- § draft cells

/-- Cells of the draft chain after accepted row `a` of a window starting at `p`, for a window of `T` tokens. -/
def draftCells (p a T : Nat) : List Nat := (List.range (T - 1)).map (fun j => p + a + 1 + j)

/-- A round of window size `T` produces exactly `T - 1` draft cells. -/
theorem draftCells_length (p a T : Nat) : (draftCells p a T).length = T - 1 := by
  simp [draftCells]

/-- Every draft cell is strictly after the last accepted cell `p + a`. -/
theorem draftCells_gt {p a T c : Nat} (h : c ∈ draftCells p a T) : p + a < c := by
  simp [draftCells] at h
  obtain ⟨j, _, hj⟩ := h
  omega

-- § attention window

/-- First cell attended when `n` cells exist and the window is `w` (0 = all). -/
def winLo (n w : Nat) : Nat := if w = 0 then 0 else n - w

/-- The window never starts after the end. -/
theorem winLo_le (n w : Nat) : winLo n w ≤ n := by
  unfold winLo
  split <;> omega

/-- A positive window attends at most `w` cells. -/
theorem window_bound {n w : Nat} (hw : 0 < w) : n - winLo n w ≤ w := by
  unfold winLo
  have : w ≠ 0 := by omega
  simp [this]
  omega

/-- A zero window attends every cell. -/
theorem window_zero_all (n : Nat) : winLo n 0 = 0 := by
  simp [winLo]

-- § bounds

/-- `exec_[9]` is indexed by `T`; with `T <= 8` the index is in range. -/
theorem exec_index_lt {T : Nat} (h : T ≤ 8) : T < 9 := by omega

/-- The 512 resident experts take under 708,000,000 bytes... -/
theorem experts_vram : 512 * 1382400 < 708000000 := by decide

/-- ...and no less than 707,000,000 (the quoted 708 MB). -/
theorem experts_vram_lower : 707000000 < 512 * 1382400 := by decide

end MtpDraftChain
