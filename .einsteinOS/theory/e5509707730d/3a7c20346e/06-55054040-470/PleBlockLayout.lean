/-!
# MODULE: ple_block_layout
SOURCES: src/kernels/cuda/ple.cu
GPU half of the PLE block: key/value projections, grouped RMS norm, gate, broadcast, depthwise causal dilated
convolution over a carried history, and the three-way residual sum, all inside one caller-owned scratch arena.

## Source semantics
* The history is a per-channel column of `NG_HIST` entries; `ple_history_advance` drops the oldest and
  appends the new normalised value.
* Conv tap `k` reads row `nhist - (kern-1-k)*dil` of `[history | new]`; only tap `kern-1` reads the new row.
  The history is row-fastest: flat index `row + nhist * channel`.
* Stream `c` of the grouped norm occupies `[c*n_embd, (c+1)*n_embd)`; the gate is broadcast with
  `value[i % n_embd] * gate[i / n_embd]`.
* The scratch arena is five hc_dim buffers + `n_embd` + `hc` floats, then the Q8 activation image, then a
  BF16 embedding copy, each region 16-byte aligned, plus 256 bytes of slack.

## Form template identifiers
`ple_block` -> `ple_layer`; `hist_rows` -> `conv_history`; `d_scratch` -> `float_workspace`;
`NG_HC_DIM` -> `hc_dim`; `NG_N_EMBD` -> `n_embd`; `NG_HC` -> `n_stream`; `NG_HIST` -> `n_hist`.
-/

namespace Strata.Ple

-- § history advance (indexed form of "drop oldest, append newest")

/-- Entry `i` of the advanced column: shifted down by one, with the new value at the end. -/
def advIdx {α : Type} (col : Nat → α) (n : Nat) (x : α) (i : Nat) : α :=
  if i + 1 < n then col (i + 1) else x

/-- Every non-final slot receives its successor's value. -/
theorem advIdx_shift {α : Type} (col : Nat → α) (n : Nat) (x : α) (i : Nat) (h : i + 1 < n) :
    advIdx col n x i = col (i + 1) := by
  unfold advIdx
  rw [if_pos h]

/-- The last slot receives the new normalised value. -/
theorem advIdx_last {α : Type} (col : Nat → α) (n : Nat) (x : α) (hn : 1 ≤ n) :
    advIdx col n x (n - 1) = x := by
  unfold advIdx
  have hc : ¬ (n - 1 + 1 < n) := by omega
  rw [if_neg hc]

-- § conv taps and the history layout

/-- Row read by a tap that is `d = kern-1-k` positions before the newest one. -/
def convRow (nhist dil d : Nat) : Nat := nhist - d * dil

/-- Real geometry (kern 4, dilation 3, history 9): the four taps read rows 0, 3, 6 and 9. -/
theorem convRow_real : [0, 1, 2, 3].map (fun k => convRow 9 3 (4 - 1 - k)) = [0, 3, 6, 9] := by
  decide

/-- With positive dilation only the newest tap (`d = 0`) reads the new row `nhist`. -/
theorem convRow_eq_new (n dil d : Nat) (hdil : 0 < dil) (hle : d * dil ≤ n) :
    convRow n dil d = n ↔ d = 0 := by
  unfold convRow
  constructor
  · intro h
    by_contra hd
    have : 0 < d * dil := Nat.mul_pos (Nat.pos_of_ne_zero hd) hdil
    omega
  · intro h
    subst h
    simp

/-- Row-fastest flat index of the conv state. -/
def histIdx (row nhist c : Nat) : Nat := row + nhist * c

/-- A valid (row, channel) pair stays inside `nhist * hc` entries. -/
theorem histIdx_lt (row nhist c hc : Nat) (hr : row < nhist) (hcc : c < hc) :
    histIdx row nhist c < nhist * hc := by
  unfold histIdx
  have h1 : nhist * (c + 1) ≤ nhist * hc := Nat.mul_le_mul_left nhist (by omega)
  have h2 : nhist * (c + 1) = nhist * c + nhist := by rw [Nat.mul_add, Nat.mul_one]
  omega

/-- The row-fastest layout is injective: reading it channel-slowest would be a transposed state. -/
theorem histIdx_inj (n r r' c c' : Nat) (hn : 0 < n) (hr : r < n) (hr' : r' < n)
    (h : histIdx r n c = histIdx r' n c') : r = r' ∧ c = c' := by
  unfold histIdx at h
  have h1 : (r + n * c) % n = r := by
    rw [Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt hr
  have h2 : (r' + n * c') % n = r' := by
    rw [Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt hr'
  have h3 : (r + n * c) / n = c := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt hr]
    simp
  have h4 : (r' + n * c') / n = c' := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt hr']
    simp
  constructor
  · calc r = (r + n * c) % n := h1.symm
      _ = (r' + n * c') % n := by rw [h]
      _ = r' := h2
  · calc c = (r + n * c) / n := h3.symm
      _ = (r' + n * c') / n := by rw [h]
      _ = c' := h4

-- § grouped norm and broadcast indexing

/-- Real geometry: 4 streams of 2560 make hc_dim = 10240. -/
theorem hc_dim_real : 4 * 2560 = 10240 := by decide

/-- Broadcast index: element `i < n_embd * hc` reads gate `i / n_embd`, which is a valid stream. -/
theorem bcast_gate_lt (i n_embd hc : Nat) (h : i < n_embd * hc) : i / n_embd < hc := by
  have h' : i < n_embd * hc := h
  exact Nat.div_lt_of_lt_mul h'

/-- Broadcast index: the value index `i % n_embd` is in range when `n_embd > 0`. -/
theorem bcast_value_lt (i n_embd : Nat) (h : 0 < n_embd) : i % n_embd < n_embd :=
  Nat.mod_lt i h

/-- The block-sum scratch of 8 doubles holds one entry per warp of a 256-thread block. -/
theorem block_sum_scratch : (256 + 31) / 32 = 8 := by decide

-- § scratch arena

/-- Round up to a multiple of 16: `(x + 15) & ~15`. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by
  unfold align16
  omega

/-- Alignment gives a multiple of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16
  omega

/-- Alignment adds fewer than 16 bytes. -/
theorem align16_lt (x : Nat) : align16 x < x + 16 := by
  unfold align16
  omega

/-- Alignment is idempotent. -/
theorem align16_idem (x : Nat) : align16 (align16 x) = align16 x := by
  unfold align16
  omega

/-- Total arena size from the three region sizes (floats, Q8 image, BF16 embedding) plus slack. -/
def scratchTotal (f q e : Nat) : Nat := align16 f + align16 q + e + 256

/-- `ple_block_scratch_bytes` for a given geometry. -/
def scratchBytes (nEmbd hc hcDim : Nat) : Nat :=
  scratchTotal ((5 * hcDim + nEmbd + hc) * 4) ((nEmbd / 32) * 34) (nEmbd * 2)

/-- The real geometry needs 223152 bytes of scratch. -/
theorem scratchBytes_real : scratchBytes 2560 4 10240 = 223152 := by decide

/-- The float region ends before the Q8 activation image begins. -/
theorem float_region_before_act (f : Nat) : f ≤ align16 f := align16_ge f

/-- The Q8 image ends before the BF16 embedding copy begins. -/
theorem act_region_before_emb (q : Nat) : q ≤ align16 q := align16_ge q

/-- The BF16 embedding copy fits inside the arena, with the 256-byte slack to spare. -/
theorem emb_region_fits (f q e : Nat) : align16 f + align16 q + e + 256 ≤ scratchTotal f q e + 0 := by
  unfold scratchTotal
  omega

/-- The float sub-buffers tile the float region exactly: key, query, norm, gated, conv, value, gate. -/
theorem float_buffers_tile (hcDim nEmbd hc : Nat) :
    5 * hcDim + nEmbd + hc = hcDim + hcDim + hcDim + hcDim + hcDim + nEmbd + hc := by
  omega

end Strata.Ple
