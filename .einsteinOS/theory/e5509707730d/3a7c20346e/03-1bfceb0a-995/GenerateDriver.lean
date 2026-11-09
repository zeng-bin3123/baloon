/-!
# MODULE: generate_driver
SOURCES: src/program/generate.cpp
The `strata generate` driver: option validation, the per-token decode loop with teacher forcing, the
speculative verify-window loop, the VRAM expert-cache sizing, the routing trace, and the M-RoPE position
table used by the vision path of `--serve`.

## Source semantics
* A prompt is a non-empty list of token ids in `[0, 2^31)`; `--max-new`/`--max-context` must be positive and
  the prompt plus generation must fit the context, which itself is below `2^31 - 8`.
* The logits dump has `n_prompt - 1 + max_new` positions, because sampling only starts being recorded at the
  last prompt position.
* The routing trace records the layer as `d.layers - 1` (the adapter has already advanced it) and must lie in
  `0..47`.
* `--expert-cache auto` grants `(free - reserve) / max_blob` slots (never negative).
* The prompt path lends the last `k >= 1` cache slots and needs `k + 128 <= slots`.
* A speculative round has `1 <= T <= S` tokens, accepts the longest matching draft prefix `a <= T - 1`, commits
  `a + 1` tokens, and never overshoots `max_new` or `max_context`.
* Image rows get M-RoPE positions `(p, p + y, p + x)` with `y = j / nx`, `x = j % nx`; text after the image
  continues at `p + max(nx, ny)`.

## Form template identifiers
validConfig -> the `--max-new`/`--max-context` checks in `main`
dumpPositions -> `n_prompt - 1 + max_new`
layerIdx -> `(int32_t)(t->d.layers - 1)` in `drive_pool`
autoSlots -> `(free_b - reserve) / expert_layout().max_blob`
pickTAux -> the `while (T < S && dprob[T-1] >= min_p) ++T` loop
accPrefix -> the `while (a < T - 1 && window[a+1] == outv[a]) ++a` loop
imgPos -> `put(i + j, p, p + y, p + x)`
-/

namespace Strata.Program

-- § argument validation

/-- A token id accepted by `parse_i64_list`: a non-negative int32. -/
def validId (n : Int) : Prop := 0 ≤ n ∧ n ≤ 2147483647

/-- Negative ids are rejected. -/
theorem negative_id_rejected (n : Int) (h : n < 0) : ¬ validId n := by
  unfold validId
  omega

/-- Ids above int32 range are rejected. -/
theorem huge_id_rejected (n : Int) (h : 2147483647 < n) : ¬ validId n := by
  unfold validId
  omega

/-- The size checks `main` performs before anything is allocated. -/
def validConfig (promptLen maxNew maxCtx : Nat) : Prop :=
  0 < maxNew ∧ 0 < maxCtx ∧ maxNew ≤ maxCtx ∧ promptLen ≤ maxCtx - maxNew

/-- A valid configuration holds the prompt and every generated token. -/
theorem config_fits (p n c : Nat) (h : validConfig p n c) : p + n ≤ c := by
  unfold validConfig at h
  omega

/-- With `max_context < 2^31 - 8`, positions and cells stay within int32 on the device. -/
theorem positions_fit_int32 (p n c : Nat) (h : validConfig p n c) (hc : c ≤ 2147483647 - 8) :
    p + n + 8 ≤ 2147483647 := by
  unfold validConfig at h
  omega

-- § logits dump header

/-- Number of positions the logits dump covers. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- The header count equals the rows actually written: one per processed position. -/
theorem dumpPositions_eq (nPrompt maxNew : Nat) (h : 1 ≤ nPrompt) :
    dumpPositions nPrompt maxNew + 1 = nPrompt + maxNew := by
  unfold dumpPositions
  omega

-- § routing trace

/-- Layer index recorded by `drive_pool`: the adapter has advanced `d.layers` already. -/
def layerIdx (layersAfter : Int) : Int := layersAfter - 1

/-- The trace's bounds check accepts exactly the post-advance counts `1..48`. -/
theorem layerIdx_valid_iff (l : Int) : (0 ≤ layerIdx l ∧ layerIdx l < 48) ↔ (1 ≤ l ∧ l ≤ 48) := by
  unfold layerIdx
  omega

-- § VRAM expert cache sizing

/-- Slots granted by `--expert-cache auto`. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto-sized cache never needs more VRAM than is free. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free := by
  unfold autoSlots
  have := Nat.div_mul_le_self (free - reserve) blob
  omega

/-- Lending `k` slots to the prompt path leaves the first lent index valid and at least 128 slots resident. -/
theorem lend_first_bounds (slots k : Nat) (hk : 1 ≤ k) (h : k + 128 ≤ slots) :
    128 ≤ slots - k ∧ slots - k < slots := by
  omega

/-- The number of lent slots for a non-empty buffer is at least one. -/
theorem lent_count_pos (need blob : Nat) (hn : 0 < need) (hb : 0 < blob) :
    1 ≤ (need + blob - 1) / blob := by
  apply Nat.div_pos
  · omega
  · exact hb

-- § PCIe share

/-- `pcie_num`: the fraction in 1/256ths, clamped to `[0, 256]`. -/
def pcieNum (x : Int) : Int := max 0 (min 256 x)

/-- The clamped share is always a valid fraction numerator. -/
theorem pcieNum_bounds (x : Int) : 0 ≤ pcieNum x ∧ pcieNum x ≤ 256 := by
  unfold pcieNum
  omega

-- § speculative rounds

/-- Verify-window size: grows while the drafts stay above the probability threshold. -/
def pickTAux (ok : Nat → Bool) (S : Nat) : Nat → Nat → Nat
  | 0, T => T
  | f + 1, T => if T < S ∧ ok (T - 1) = true then pickTAux ok S f (T + 1) else T

/-- The window never shrinks and never exceeds `S`. -/
theorem pickTAux_bounds (ok : Nat → Bool) (S f : Nat) :
    ∀ T, T ≤ S → T ≤ pickTAux ok S f T ∧ pickTAux ok S f T ≤ S := by
  induction f with
  | zero =>
    intro T h
    simp only [pickTAux]
    exact ⟨Nat.le_refl _, h⟩
  | succ f ih =>
    intro T h
    simp only [pickTAux]
    split
    · rename_i hc
      obtain ⟨c1, _⟩ := hc
      have := ih (T + 1) (by omega)
      omega
    · exact ⟨Nat.le_refl _, h⟩

/-- Longest prefix on which the drafts equal the model's outputs. -/
def accPrefix : List Int → List Int → Nat
  | w :: ws, o :: os => if w = o then 1 + accPrefix ws os else 0
  | _, _ => 0

/-- The accepted prefix is no longer than either list. -/
theorem accPrefix_le (ws os : List Int) : accPrefix ws os ≤ min ws.length os.length := by
  induction ws generalizing os with
  | nil => simp [accPrefix]
  | cons w ws ih =>
    cases os with
    | nil => simp [accPrefix]
    | cons o os =>
      have := ih os
      simp only [accPrefix, List.length_cons]
      split <;> omega

/-- Accepted drafts plus the one verified token never exceed the window: `a + 1 <= T`. -/
theorem round_commit_le_window (a T : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) : a + 1 ≤ T := by
  omega

/-- Committing `a + 1` tokens keeps the position inside the context checked before the window ran. -/
theorem round_position_in_context (p a T ctx : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) (hc : p + T ≤ ctx) :
    p + a + 1 ≤ ctx := by
  omega

/-- Tokens emitted in a round are capped by what is still wanted. -/
def emitCount (a produced maxNew : Nat) : Nat := min (a + 1) (maxNew - produced)

/-- A round never overshoots `--max-new`. -/
theorem emit_no_overshoot (a produced maxNew : Nat) (h : produced ≤ maxNew) :
    produced + emitCount a produced maxNew ≤ maxNew := by
  unfold emitCount
  omega

-- § M-RoPE positions for images

/-- The `(h, w)` offsets of image row `j` in an `nx`-wide grid. -/
def imgPos (nx j : Nat) : Nat × Nat := (j / nx, j % nx)

/-- Row offsets stay inside the grid, so positions never exceed `p + max(nx, ny)`. -/
theorem imgPos_bounds (nx ny j : Nat) (hx : 0 < nx) (hj : j < nx * ny) :
    (imgPos nx j).1 < ny ∧ (imgPos nx j).2 < nx := by
  unfold imgPos
  exact ⟨Nat.div_lt_of_lt_mul hj, Nat.mod_lt _ hx⟩

/-- Every image cell's `h` and `w` position is strictly below where text resumes. -/
theorem img_text_resumes_after (p nx ny j : Nat) (hx : 0 < nx) (hj : j < nx * ny) :
    p + (imgPos nx j).1 < p + max nx ny ∧ p + (imgPos nx j).2 < p + max nx ny := by
  have ⟨h1, h2⟩ := imgPos_bounds nx ny j hx hj
  omega

/-- Text cells advance the position by exactly one. -/
theorem text_position_advances (p : Nat) : p + 1 > p := by
  omega

end Strata.Program
