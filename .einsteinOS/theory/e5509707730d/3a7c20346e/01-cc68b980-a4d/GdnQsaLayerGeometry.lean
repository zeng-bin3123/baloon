/-!
# MODULE: gdn_qsa_layer_geometry
SOURCES: include/strata/core/layer.hpp
One GDN/QSA block composed from kernels: layer-type split (12 QSA / 36 GDN), the q|k|v conv channel
layout, the QSA q/gate halves split, INT8 KV sizing, the pre/post (and half) graph stage split, and PLE row bytes.

## Source semantics
* QSA layers are at indices 3, 7, ... 47; the other 36 are GDN.
* `conv_channels = 10240 = q (16x128) | k (16x128) | v (48x128)`; l2-norm applies to q and k only.
* `q_full` is split per head into halves: q = first `head_dim`, gate = second `head_dim`.
* `half = 1` is stages 0..2, `half = 2` is stages 3..4, `half = 0` is both; `preP[k]` is stages 0..k-1.
* INT8 KV stores one byte per value plus one fp16 scale per 64 values.

## Form template identifiers
block_layer_pre -> front_half; block_layer_post -> back_half; is_qsa_layer -> is_attention_layer;
GdnBuffers -> recurrent_scratch; QsaState -> kv_state; Doorbell -> handoff_cell; PleRun -> layer1_embedding.
-/

namespace GdnQsaLayerGeometry

-- § layer-type split

/-- QSA layers sit at indices congruent to 3 mod 4. -/
def isQsa (i : Nat) : Bool := i % 4 == 3

/-- Exactly 12 of the 48 layers are QSA. -/
theorem qsa_count : ((List.range 48).filter isQsa).length = 12 := by decide

/-- Exactly 36 of the 48 layers are GDN. -/
theorem gdn_count : ((List.range 48).filter (fun i => !isQsa i)).length = 36 := by decide

-- § conv channels

/-- q, k are 16 heads of width 128 each; v is 48 heads of width 128. -/
theorem conv_channels : 16 * 128 + 16 * 128 + 48 * 128 = 10240 := by decide

/-- The l2 norm covers the 32 q/k heads, not the 48 value heads. -/
theorem l2norm_heads : 16 + 16 = 32 := by decide

-- § QSA q/gate halves

/-- Index of q element `d` of head `h` inside `q_full` (first half of the head's `2*hd` block). -/
def qIdx (hd h d : Nat) : Nat := h * (2 * hd) + d

/-- Index of gate element `d` of head `h` inside `q_full` (second half of the block). -/
def gIdx (hd h d : Nat) : Nat := h * (2 * hd) + hd + d

/-- Gate indices stay inside `q_full` (`n_head * 2 * head_dim` floats). -/
theorem gIdx_lt {nh hd h d : Nat} (hh : h < nh) (hd' : d < hd) : gIdx hd h d < nh * (2 * hd) := by
  unfold gIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- q indices stay inside `q_full`. -/
theorem qIdx_lt {nh hd h d : Nat} (hh : h < nh) (hd' : d < hd) : qIdx hd h d < nh * (2 * hd) := by
  unfold qIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Within one head, q and gate never collide (contiguous halves, not interleaved). -/
theorem q_gate_disjoint {hd h d d' : Nat} (hd' : d < hd) : qIdx hd h d ≠ gIdx hd h d' := by
  unfold qIdx gIdx
  omega

-- § graph stage split

/-- Stages run by `half` (0 = both). -/
def stages : Nat → List Nat
  | 1 => [0, 1, 2]
  | 2 => [3, 4]
  | _ => [0, 1, 2, 3, 4]

/-- The two halves concatenate to the whole layer. -/
theorem halves_cover : stages 1 ++ stages 2 = stages 0 := rfl

/-- The two halves share no stage. -/
theorem halves_disjoint : ∀ s ∈ stages 1, s ∉ stages 2 := by decide

/-- `preP[k]` runs the first `k` stages. -/
def prefixStages (k : Nat) : List Nat := List.range k

/-- Consecutive prefixes differ by exactly one stage, so differencing times isolates it. -/
theorem prefix_step (k : Nat) : prefixStages (k + 1) = prefixStages k ++ [k] := List.range_succ

/-- Prefix `k` has `k` stages. -/
theorem prefix_length (k : Nat) : (prefixStages k).length = k := by
  simp [prefixStages]

-- § sizing

/-- INT8 KV with one fp16 scale per 64 values never exceeds the fp16 cache size (for whole groups). -/
theorem int8_kv_smaller (m : Nat) : 64 * m + 2 * m ≤ 2 * (64 * m) := by omega

/-- PLE traffic per token: sixteen 90-byte rows = 1440 bytes. -/
theorem ple_row_bytes : 16 * 90 = 1440 := by decide

end GdnQsaLayerGeometry
