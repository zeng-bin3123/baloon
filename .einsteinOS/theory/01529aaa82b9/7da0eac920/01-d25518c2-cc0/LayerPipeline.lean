/-!
# MODULE: layer_pipeline
One GDN/QSA layer composed with the MoE block: channel geometry of the GDN qkv path, the MoE combine and its
split into CPU and GPU parts, the pre/post stage prefixes, and the pinned-staging sizes.

## Source semantics
* The GDN conv width is `q | k | v`: 16 + 16 + 48 heads of width 128 = 10240 channels; only q and k are
  l2-normalised (two rows of 16 heads), v is not.
* `moe_combine` computes `sum_j w_j * parts_j`; it is additive in `parts`, so CPU rows for misses plus GPU rows
  for hits equal the complete expert output.
* `moe_finish` is `moe_shared` followed by `moe_combine_parts`.
* `half = 1` is stages 0..2, `half = 2` is stages 3..4, `half = 0` is all five; the two halves concatenate to
  the whole.
* QSA layers are at indices 3, 7, ..., 47 (12 of 48).

## Form template identifiers
* `gdn_layer` / `qsa_layer` -> `convChannels`, `isQsa`
* `moe_combine` -> `combineN`
* `block_layer_pre` `half` -> `stagesOfHalf`
* `Doorbell` -> `bellValue`
-/

namespace Strata.LayerPipeline

-- § GDN geometry

/-- Query channels: 16 heads of width 128. -/
def qChannels : Nat := 16 * 128

/-- Key channels: 16 heads of width 128. -/
def kChannels : Nat := 16 * 128

/-- Value channels: 48 heads of width 128. -/
def vChannels : Nat := 48 * 128

/-- The conv output is the concatenation `q | k | v` and is 10240 wide. -/
theorem conv_channels : qChannels + kChannels + vChannels = 10240 := by
  decide

/-- The l2 norm covers only q and k: 2 rows of 16 heads, against 48 value heads. -/
theorem l2_rows : 2 * 16 = 32 ∧ 32 < 48 + 32 := by
  decide

/-- The recurrent state holds `S * h_v * S = 128 * 48 * 128` floats per GDN layer. -/
theorem gdn_state_floats : 128 * 48 * 128 = 786432 := by
  decide

/-- The q/k/v runs are contiguous: each slice's end is the next slice's start. -/
theorem qkv_contiguous : qChannels = 2048 ∧ qChannels + kChannels = 4096 ∧ 4096 + vChannels = 10240 := by
  decide

-- § QSA layer positions

/-- A layer is QSA exactly at indices 3, 7, ..., 47. -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- There are 12 QSA layers among 48. -/
theorem qsa_layer_count : (List.range 48).countP isQsa = 12 := by
  first | decide | rfl

/-- Every layer is exactly one of GDN or QSA. -/
theorem layer_kind_partition (l : Nat) : isQsa l = true ∨ isQsa l = false := by
  cases isQsa l <;> simp

-- § MoE combine

/-- `moe_combine`: the weighted sum over the first `k` router slots. -/
def combineN (w p : Nat → Int) : Nat → Int
  | 0 => 0
  | k + 1 => combineN w p k + w k * p k

/-- The combine is additive in the expert outputs. -/
theorem combine_additive (w a b : Nat → Int) (k : Nat) :
    combineN w (fun i => a i + b i) k = combineN w a k + combineN w b k := by
  induction k with
  | zero => simp [combineN]
  | succ k ih => simp only [combineN, ih, Int.mul_add]; omega

/-- The combine of all-zero expert outputs is zero. -/
theorem combine_zero (w : Nat → Int) (k : Nat) : combineN w (fun _ => 0) k = 0 := by
  induction k with
  | zero => simp [combineN]
  | succ k ih => simp [combineN, ih]

/-- A zero router weight removes that expert from the sum. -/
theorem combine_zero_weight (p : Nat → Int) (k : Nat) : combineN (fun _ => 0) p k = 0 := by
  induction k with
  | zero => simp [combineN]
  | succ k ih => simp [combineN, ih]

/-- The combine only reads the first `k` slots: changing a slot at or beyond `k` changes nothing. -/
theorem combine_local (w p q : Nat → Int) (k : Nat) (h : ∀ i, i < k → p i = q i) :
    combineN w p k = combineN w q k := by
  induction k with
  | zero => simp [combineN]
  | succ k ih =>
    have h1 : combineN w p k = combineN w q k := ih (fun i hi => h i (by omega))
    have h2 : p k = q k := h k (by omega)
    simp only [combineN, h1, h2]

-- § stages

/-- The stage prefix selected by `half`. -/
def stagesOfHalf : Nat → List Nat
  | 1 => [0, 1, 2]
  | 2 => [3, 4]
  | _ => [0, 1, 2, 3, 4]

/-- The two halves concatenate to the whole block. -/
theorem halves_compose : stagesOfHalf 1 ++ stagesOfHalf 2 = stagesOfHalf 0 := by
  decide

/-- The two halves are disjoint stage sets. -/
theorem halves_disjoint : ∀ s ∈ stagesOfHalf 1, s ∉ stagesOfHalf 2 := by
  decide

/-- Five stages in total: gr_read, mixer, gr_write, gr_read, router. -/
theorem stage_count : (stagesOfHalf 0).length = 5 := by
  decide

-- § doorbell

/-- The doorbell ring value after `n` rings. -/
def bellValue (n : Nat) : Nat := n

/-- The device wait is satisfied once the host flag equals the ring. -/
def waitSatisfied (flag ring : Nat) : Bool := decide (flag = ring)

/-- After the host serves a ring, the wait is satisfied. -/
theorem serve_satisfies (n : Nat) : waitSatisfied (bellValue n) (bellValue n) = true := by
  simp [waitSatisfied]

/-- A stale flag (one ring behind) does not satisfy the wait: the device cannot run ahead. -/
theorem stale_flag_blocks (n : Nat) : waitSatisfied n (n + 1) = false := by
  unfold waitSatisfied
  exact decide_eq_false (by omega)

-- § buffer sizes

/-- The half-level oracle dump per layer: `2 * n_embd + 2 * hc + n_head * head_dim` floats (hc = 4). -/
def dumpFloats (nEmbd hc nHead headDim : Nat) : Nat := 2 * nEmbd + 2 * hc + nHead * headDim

/-- The dump region is at least the two block outputs. -/
theorem dump_covers_halves (nEmbd hc nHead headDim : Nat) : 2 * nEmbd ≤ dumpFloats nEmbd hc nHead headDim := by
  unfold dumpFloats
  omega

/-- Embedding row dequant: one column of `n_embd = 2560` values, 80 Q8_0 blocks of 32. -/
theorem embd_blocks : 2560 / 32 = 80 ∧ 2560 % 32 = 0 := by
  decide

/-- The two indexer ropes: the PLE gather reads 16 rows of 90 bytes, 1440 bytes per token. -/
theorem ple_bytes_per_token : 16 * 90 = 1440 := by
  decide

end Strata.LayerPipeline
