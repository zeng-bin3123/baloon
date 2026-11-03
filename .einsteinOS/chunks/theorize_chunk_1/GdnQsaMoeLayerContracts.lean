import Mathlib

/-!
# MODULE: gdn_qsa_moe_layer_contracts
Shape and ordering contracts of one decoder block: GDN mixer geometry (q|k|v packing), arena carving with
16-byte alignment, the QSA `q | gate` half-split, the MoE split at the router (`pre` / `post` halves), the
doorbell ring/flag handshake, and the RoPE table size.

## Source semantics
* GDN `conv_channels = 10240` is q (16 heads x 128) | k (16 x 128) | v (48 x 128); v is not normalised.
* Buffers are carved from one arena with 16-byte alignment; the total is at least the sum of the sizes.
* `q_full` holds, per head, `q` in the first half and `gate` in the second half of a `2*head_dim` block.
* `moe_finish` is exactly `moe_shared` followed by `moe_combine_parts`.
* `half = 1` runs stages 0..2, `half = 2` stages 3..4, `half = 0` all five.
* The device releases the doorbell wait only when `flag = ring`.
* The RoPE table (max_cells x n_rot/2 x 2 floats) is 64 MiB at 262,144 cells for n_rot = 64.

## Form template identifiers
* `align16/carveEnd`      -> `ARENA_CARVE`
* `qIdx/gateIdx`          -> `QSA_Q_GATE_HALVES`
* `stagesOf`              -> `BLOCK_STAGE_SPLIT`
* `waitReleased`          -> `DOORBELL_WAIT`
-/

namespace StrataLayerContracts

-- § GDN geometry

/-- q|k|v packing of the conv channels: two 16-head rows of 128 plus 48 value heads of 128. -/
theorem gdn_conv_channels : 16 * 128 + 16 * 128 + 48 * 128 = 10240 := by decide

/-- The value dimension is 48 heads of 128. -/
theorem gdn_value_dim : 48 * 128 = 6144 := by decide

-- § arena carving

/-- Round up to a multiple of 16 bytes. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks a region. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- Each carved region is 16-byte aligned in size. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- End offset after carving regions of the given sizes in order. -/
def carveEnd : Nat → List Nat → Nat
  | acc, [] => acc
  | acc, s :: ss => carveEnd (acc + align16 s) ss

/-- The arena is at least as large as the sum of its regions. -/
theorem carveEnd_ge : ∀ (ss : List Nat) (acc : Nat), acc + ss.sum ≤ carveEnd acc ss := by
  intro ss
  induction ss with
  | nil => intro acc; simp [carveEnd]
  | cons s ss ih =>
    intro acc
    simp only [carveEnd, List.sum_cons]
    have h := ih (acc + align16 s)
    have h2 := align16_ge s
    omega

/-- Carving from an aligned base keeps the end aligned. -/
theorem carveEnd_aligned : ∀ (ss : List Nat) (acc : Nat), acc % 16 = 0 → carveEnd acc ss % 16 = 0 := by
  intro ss
  induction ss with
  | nil => intro acc h; simpa [carveEnd] using h
  | cons s ss ih =>
    intro acc h
    simp only [carveEnd]
    exact ih _ (by unfold align16; omega)

-- § QSA q | gate halves

/-- Index of head `h`'s query element `i` in `q_full` (first half of the head's block). -/
def qIdx (hd h i : Nat) : Nat := h * (2 * hd) + i

/-- Index of head `h`'s gate element `i` in `q_full` (second half). -/
def gateIdx (hd h i : Nat) : Nat := h * (2 * hd) + hd + i

/-- The gate half comes strictly after the query half within a head. -/
theorem qIdx_lt_gateIdx (hd h i j : Nat) (hi : i < hd) : qIdx hd h i < gateIdx hd h j := by
  unfold qIdx gateIdx; omega

/-- Both halves of every head stay inside `n_head * 2 * head_dim`. -/
theorem gateIdx_lt (nh hd h i : Nat) (hh : h < nh) (hi : i < hd) : gateIdx hd h i < nh * (2 * hd) := by
  unfold gateIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ hh
  have h2 : (h + 1) * (2 * hd) = h * (2 * hd) + 2 * hd := Nat.succ_mul h (2 * hd)
  omega

/-- The query half is in bounds too. -/
theorem qIdx_lt (nh hd h i : Nat) (hh : h < nh) (hi : i < hd) : qIdx hd h i < nh * (2 * hd) := by
  have := qIdx_lt_gateIdx hd h i i hi
  have h2 := gateIdx_lt nh hd h i hh hi
  omega

-- § MoE split at the router

/-- `moe_finish` as the composition of its two parts (shared expert, then combine). -/
def finish {S : Type} (shared combine : S → S) : S → S := combine ∘ shared

/-- `moe_finish` is exactly `moe_shared` followed by `moe_combine_parts`. -/
theorem finish_eq {S : Type} (shared combine : S → S) (s : S) : finish shared combine s = combine (shared s) := rfl

/-- The stages (0..4) covered by a `half` selector. -/
def stagesOf : Nat → List Nat
  | 1 => [0, 1, 2]
  | 2 => [3, 4]
  | _ => [0, 1, 2, 3, 4]

/-- Replaying half 1 then half 2 covers exactly the whole block. -/
theorem halves_cover : stagesOf 1 ++ stagesOf 2 = stagesOf 0 := by decide

-- § doorbell handshake

/-- The device's wait releases only when the host's flag equals the ring value. -/
def waitReleased (flag ring : Nat) : Prop := flag = ring

/-- A stale flag (an earlier ring) does not release the next layer's wait. -/
theorem stale_flag_blocks (r : Nat) : ¬ waitReleased r (r + 1) := by
  unfold waitReleased; omega

/-- Answering the current ring releases it. -/
theorem answered_releases (r : Nat) : waitReleased r r := rfl

-- § RoPE table

/-- 262,144 cells x 32 pair-slots x 2 floats x 4 B is 64 MiB (n_rot = 64). -/
theorem rope_table_64MiB : 262144 * 32 * 2 * 4 = 64 * 1048576 := by decide

end StrataLayerContracts
