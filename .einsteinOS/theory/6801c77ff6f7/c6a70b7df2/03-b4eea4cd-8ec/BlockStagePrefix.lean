/-!
# MODULE: block_stage_prefix
SOURCES: include/strata/core/layer.hpp, src/core/layer.cpp
Which of the five stages of `block_layer_pre` run for a given (`half`, `stage_prefix`), when the layer-1 PLE
fires, when the fused hyper-connection path is active, and the 12/36 QSA/GDN split of the 48 layers.

## Source semantics
stages: 0 gr_read(attn), 1 mixer, 2 gr_write(attn), 3 gr_read(ffn), 4 moe_route.
* `stage_prefix = 0`: `half = 0` runs all five, `half = 1` runs 0..2, `half = 2` runs 3..4.
* `stage_prefix = k > 0`: runs stages 0..k-1 and ignores `half`.
* The PLE runs only at layer 1 and only when the mixer half is requested (`half` 0 or 1).
* The fused GR path needs `stage_prefix = 0` and `half = 0`; the fused post only writes R on the last layer.
* `is_qsa_layer`: every fourth layer (l % 4 = 3), so 12 QSA and 36 GDN layers among 48.

## Form template identifiers
runStage <- `run0..run4`; fusedActive <- `fused`; pleApplies <- the PLE `if`; isQsa <- `is_qsa_layer`.
-/

namespace Strata.BlockStagePrefix

-- § stage selection

/-- Whether stage `i` (0..4) runs. -/
def runStage (sp half i : Nat) : Bool :=
  if 0 < sp then decide (i < sp)
  else if i ≤ 2 then decide (half = 0 ∨ half = 1)
  else decide (half = 0 ∨ half = 2)

/-- The default call (half 0, no prefix) runs every stage. -/
theorem whole_runs_all : ∀ i, i < 5 → runStage 0 0 i = true := by
  decide

/-- `half = 1` is exactly the mixer: stages 0, 1, 2. -/
theorem half1_is_mixer : ∀ i, i < 5 → runStage 0 1 i = decide (i ≤ 2) := by
  decide

/-- `half = 2` is exactly the FFN front and router: stages 3, 4. -/
theorem half2_is_ffn : ∀ i, i < 5 → runStage 0 2 i = decide (3 ≤ i) := by
  decide

/-- A prefix of length k runs exactly the first k stages, whatever `half` is. -/
theorem prefix_runs_first_k :
    ∀ k, k < 6 → 1 ≤ k → ∀ half, half < 3 → ∀ i, i < 5 → runStage k half i = decide (i < k) := by
  decide

/-- Prefixes are downward closed: if a stage runs, so does the one before it. -/
theorem prefix_downward_closed :
    ∀ k, k < 6 → 1 ≤ k → ∀ half, half < 3 → ∀ i, i < 4 →
      runStage k half (i + 1) = true → runStage k half i = true := by
  decide

/-- The two halves partition the five stages. -/
theorem halves_partition : ∀ i, i < 5 →
    (runStage 0 1 i = true ∨ runStage 0 2 i = true) ∧ ¬ (runStage 0 1 i = true ∧ runStage 0 2 i = true) := by
  decide

-- § the layer-1 PLE and the fused path

/-- The PLE block runs only at layer 1, when ready, in the mixer half or a whole call. -/
def pleApplies (ready : Bool) (layer half : Nat) : Bool :=
  ready && decide (layer = 1) && decide (half = 0 ∨ half = 1)

/-- No PLE outside layer 1. -/
theorem ple_only_layer1 (ready : Bool) (layer half : Nat) (h : layer ≠ 1) :
    pleApplies ready layer half = false := by
  simp [pleApplies, h]

/-- No PLE in the FFN-front half. -/
theorem ple_not_in_half2 (ready : Bool) (layer : Nat) : pleApplies ready layer 2 = false := by
  simp [pleApplies]

/-- An unwired PLE is skipped entirely. -/
theorem ple_skipped_when_not_ready (layer half : Nat) : pleApplies false layer half = false := by
  simp [pleApplies]

/-- The fused GR path's activation condition. -/
def fusedActive (flag : Bool) (sp half : Nat) (supported : Bool) : Bool :=
  flag && decide (sp = 0) && decide (half = 0) && supported

/-- The fused path never runs for a prefix or a half-graph. -/
theorem fused_needs_whole (flag : Bool) (sp half : Nat) (sup : Bool)
    (h : fusedActive flag sp half sup = true) : sp = 0 ∧ half = 0 := by
  simp [fusedActive] at h
  refine ⟨?_, ?_⟩ <;> omega

/-- In the fused post graph R is written only on the last layer; otherwise every layer writes it. -/
def postWritesR (fused : Bool) (layer nLayers : Nat) : Bool :=
  !fused || decide (layer + 1 = nLayers)

/-- Unfused: every layer writes R. -/
theorem unfused_always_writes (layer n : Nat) : postWritesR false layer n = true := by
  simp [postWritesR]

/-- Fused: only the last layer materialises R, so the head sees a complete residual. -/
theorem fused_writes_only_last (layer n : Nat) (h : layer + 1 ≠ n) : postWritesR true layer n = false := by
  simp [postWritesR, h]

-- § layer kinds

/-- `is_qsa_layer` for the real model. -/
def isQsa (l : Nat) : Bool := l % 4 == 3

/-- 12 of the 48 layers are QSA. -/
theorem qsa_count : ((List.range 48).filter isQsa).length = 12 := by
  decide

/-- 36 of the 48 layers are GDN. -/
theorem gdn_count : ((List.range 48).filter (fun l => !isQsa l)).length = 36 := by
  decide

end Strata.BlockStagePrefix
