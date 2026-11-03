/-!
# MODULE: prefill_kernel_indexing
Index arithmetic of the batched prefill CUDA kernels: hyper-connection streams, 2-bit blob dequantization,
Q/gate splitting, and KV append grid decomposition.

## Source semantics
- A residual element index i over T*D, D = N*HC (N = 2560, HC = 4), decomposes as t = i/D, c = (i%D)/N, d = i%N.
- A 2-bit code c in 0..3 decodes to (c-1) * scale, an integer multiplier in -1..2.
- Blob gate/up rows have 640 code bytes (2560 weights) and 40 scale groups of 64.
- q_full has 24 heads of 512 floats: the first 256 are the query, the next 256 the gate.
- kv_append grid z in 0..7 selects (group z/2, K-or-V z%2).

## Form template identifiers
gr_write_kernel indexing -> splitResidual ; blob_dequant_kernel -> blobGuIndex ;
split_q_kernel / gate_attn_kernel -> qfullIndex ; kv_append_kernel blockIdx.z -> kvZ
-/

namespace Strata.PrefillKernels

-- § residual stream indexing (N = 2560, HC = 4, D = 10240)

/-- (token, stream, dim) of a flat residual index. -/
def splitResidual (i : Nat) : Nat × Nat × Nat := (i / 10240, (i % 10240) / 2560, i % 2560)

/-- The stream index is below HC = 4. -/
theorem stream_lt (i : Nat) : (splitResidual i).2.1 < 4 := by
  unfold splitResidual; simp; omega

/-- The dim index is below N. -/
theorem dim_lt (i : Nat) : (splitResidual i).2.2 < 2560 := by
  unfold splitResidual; simp; omega

/-- The decomposition reconstructs the flat index. -/
theorem splitResidual_recon (i : Nat) :
    (splitResidual i).1 * 10240 + (splitResidual i).2.1 * 2560 + (splitResidual i).2.2 = i := by
  unfold splitResidual; simp; omega

/-- Broadcast reads the per-token embedding at dim i % N, equal to the dim of the decomposition. -/
theorem broadcast_dim (i : Nat) : i % 2560 = (splitResidual i).2.2 := rfl

-- § 2-bit dequantization

/-- Decoded integer multiplier of a 2-bit code. -/
def decode2 (c : Nat) : Int := (c : Int) - 1

/-- Codes 0..3 decode to -1..2. -/
theorem decode2_range (c : Nat) (h : c < 4) : -1 ≤ decode2 c ∧ decode2 c ≤ 2 := by
  unfold decode2; omega

/-- A zero scale decodes everything to zero. -/
theorem decode2_zero_scale (c : Nat) : decode2 c * 0 = 0 := by simp

/-- Output index of weight k of code byte `byte` in row `row` of a gate/up blob. -/
def blobGuIndex (row byte k : Nat) : Nat := row * 2560 + byte * 4 + k

/-- The output index stays inside its row. -/
theorem blobGuIndex_in_row (row byte k : Nat) (hb : byte < 640) (hk : k < 4) :
    row * 2560 ≤ blobGuIndex row byte k ∧ blobGuIndex row byte k < (row + 1) * 2560 := by
  unfold blobGuIndex; omega

/-- The scale group of a code byte is one of the 40 groups. -/
theorem blob_scale_group_lt (byte : Nat) (hb : byte < 640) : (byte * 4) / 64 < 40 := by omega

/-- A flat thread index splits into a valid (row, byte) pair. -/
theorem blob_thread_split (i : Nat) : i / 640 * 640 + i % 640 = i ∧ i % 640 < 640 := by omega

-- § Q / gate splitting

/-- Index of element d of head h of token t in q_full. -/
def qfullIndex (t h d : Nat) : Nat := t * 12288 + h * 512 + d

/-- The query half and the gate half do not overlap. -/
theorem q_gate_disjoint (t h d d' : Nat) (hd : d < 256) (hd' : d' < 256) :
    qfullIndex t h d < qfullIndex t h (256 + d') := by
  unfold qfullIndex; omega

/-- The gate element stays inside its head. -/
theorem gate_in_head (t h d : Nat) (hd : d < 256) :
    qfullIndex t h (256 + d) < t * 12288 + (h + 1) * 512 := by
  unfold qfullIndex; omega

/-- A head index below 24 stays inside the token's row. -/
theorem head_in_token (t h d : Nat) (hh : h < 24) (hd : d < 512) :
    qfullIndex t h d < (t + 1) * 12288 := by
  unfold qfullIndex; omega

-- § kv_append grid

/-- Group of blockIdx.z. -/
def kvGroup (z : Nat) : Nat := z / 2

/-- K-or-V selector of blockIdx.z. -/
def kvIsV (z : Nat) : Bool := z % 2 == 1

/-- The group is one of the four 64-wide groups of a 256-wide head. -/
theorem kvGroup_lt (z : Nat) (h : z < 8) : kvGroup z < 4 := by unfold kvGroup; omega

/-- Element index d = g*64 + lane is below 256. -/
theorem kv_dim_lt (z lane : Nat) (hz : z < 8) (hl : lane < 64) : kvGroup z * 64 + lane < 256 := by
  unfold kvGroup; omega

/-- Distinct z in 0..7 give distinct (group, K/V) pairs. -/
theorem kvZ_inj (z w : Nat) (hz : z < 8) (hw : w < 8) (hg : kvGroup z = kvGroup w)
    (hv : kvIsV z = kvIsV w) : z = w := by
  unfold kvGroup at hg
  unfold kvIsV at hv
  have : (z % 2 == 1) = (w % 2 == 1) := hv
  have h2 : z % 2 = w % 2 := by
    by_cases a : z % 2 = 1 <;> by_cases b : w % 2 = 1 <;> simp [a, b] at this <;> omega
  omega

end Strata.PrefillKernels
