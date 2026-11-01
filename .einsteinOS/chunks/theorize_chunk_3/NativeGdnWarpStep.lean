/-!
# MODULE: native_gdn_warp_step
Geometry, ownership and validation contracts of the warp-per-column gated-delta-net step (`native_gdn.cu`,
arithmetic taken from the pinned llama.cpp `gated_delta_net.cu`).

## Source semantics
* The grid is `(h_v, 1, S/4)` blocks of `(32, 4)` threads; block `(head, 0, z)`, thread `(lane, ty)` owns state column
  `col = z * 4 + ty` of head `head`, and the 32 lanes of a warp hold the 128 rows as 4 registers each: row `i = r*32 + lane`.
* The column contraction `kv = sum_i s_i k_i` and the read-out `attn = sum_i s'_i q_i` are warp xor-reductions;
  update `s' = g * s + k * delta`, `delta = (v - g * kv) * beta`, output scaled by `1/sqrt(S)`.
* `q`/`k` are read through `head % h_k`; `h_v % h_k = 0` and `h_v <= 65535` are required.
* Validation: all spans aligned to 4 bytes, state and output disjoint, every input disjoint from state and output.

## Form template identifiers
* `colOf`          : state column of (block z, thread ty)
* `rowOf`          : state row of (register r, lane)
* `stateIndex`     : flat state index `(row*h_v + head)*S + col`
* `overlap`        : byte-range overlap used by the validators
-/

namespace Strata.NativeGdn

-- § ownership

/-- State column owned by thread row `ty` of block-z `z`. -/
def colOf (z ty : Nat) : Nat := z * 4 + ty

/-- The `(S/4) x 4` thread arrangement covers all 128 columns. -/
theorem col_lt (z ty : Nat) (hz : z < 32) (ht : ty < 4) : colOf z ty < 128 := by unfold colOf; omega

/-- Each column has exactly one owner. -/
theorem col_owner_unique (z ty z' ty' : Nat) (ht : ty < 4) (ht' : ty' < 4) (h : colOf z ty = colOf z' ty') :
    z = z' ∧ ty = ty' := by
  unfold colOf at h; omega

/-- Every column has an owner. -/
theorem col_owner_exists (col : Nat) (h : col < 128) : col / 4 < 32 ∧ colOf (col / 4) (col % 4) = col := by
  unfold colOf; omega

/-- State row held in register `r` of lane `lane`. -/
def rowOf (r lane : Nat) : Nat := r * 32 + lane

/-- The 4 registers x 32 lanes cover the 128 rows. -/
theorem row_lt (r lane : Nat) (hr : r < 4) (hl : lane < 32) : rowOf r lane < 128 := by unfold rowOf; omega

/-- Distinct (register, lane) pairs hold distinct rows. -/
theorem row_inj (r l r' l' : Nat) (hl : l < 32) (hl' : l' < 32) (h : rowOf r l = rowOf r' l') : r = r' ∧ l = l' := by
  unfold rowOf at h; omega

/-- Every row is held by exactly one (register, lane). -/
theorem row_exists (i : Nat) (h : i < 128) : i / 32 < 4 ∧ rowOf (i / 32) (i % 32) = i := by
  unfold rowOf; omega

-- § addressing

/-- A product bound for flat indices of the shape `(a*B + c)*D + e`. -/
theorem idx3_lt (A B D a c e : Nat) (ha : a < A) (hc : c < B) (he : e < D) :
    (a * B + c) * D + e < A * B * D := by
  have h1 : a * B + c < A * B := by
    have h : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B ha
    rw [Nat.add_mul, Nat.one_mul] at h
    omega
  have h2 : (a * B + c + 1) * D ≤ A * B * D := Nat.mul_le_mul_right D h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- Flat index of state element `(row, head, col)` with `S = 128`. -/
def stateIndex (h_v row head col : Nat) : Nat := (row * h_v + head) * 128 + col

/-- Every in-range access stays inside the state buffer of `S * S * h_v` floats. -/
theorem stateIndex_lt (h_v row head col : Nat) (hr : row < 128) (hh : head < h_v) (hc : col < 128) :
    stateIndex h_v row head col < 128 * h_v * 128 := by
  unfold stateIndex
  exact idx3_lt 128 h_v 128 row head col hr hh hc

/-- The q/k head is valid and the q/k read `q_head*128 + i` stays inside the `h_k * 128` buffers. -/
theorem qk_index_lt (head h_k i : Nat) (hk : 0 < h_k) (hi : i < 128) : (head % h_k) * 128 + i < h_k * 128 := by
  have h : head % h_k < h_k := Nat.mod_lt _ hk
  omega

/-- Output index `head*128 + col` stays inside the `S * h_v` output. -/
theorem output_index_lt (h_v head col : Nat) (hh : head < h_v) (hc : col < 128) : head * 128 + col < 128 * h_v := by
  have h : (head + 1) * 128 ≤ h_v * 128 := Nat.mul_le_mul_right 128 hh
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § launch limits

/-- The block has `32 * 4 = 128` threads, matching `__launch_bounds__(128, 2)`. -/
theorem block_threads : 32 * 4 = 128 := by decide

/-- The grid's z extent `S / 4` is 32 and the product of block columns covers all of `S`. -/
theorem grid_z : 128 / 4 = 32 ∧ 32 * 4 = 128 := by decide

/-- Valid shape: `h_v` divisible by `h_k`, both positive, `h_v ≤ 65535`. -/
def ValidShape (h_k h_v : Nat) : Prop := 0 < h_k ∧ 0 < h_v ∧ h_v ≤ 65535 ∧ h_v % h_k = 0

/-- A valid shape has `h_k ≤ h_v`, so every q/k head is used by at least one value head. -/
theorem validShape_hk_le (h_k h_v : Nat) (h : ValidShape h_k h_v) : h_k ≤ h_v := by
  obtain ⟨hk, hv, _, hmod⟩ := h
  have h1 := Nat.div_add_mod h_v h_k
  rw [hmod] at h1
  rcases Nat.eq_zero_or_pos (h_v / h_k) with hz | hp
  · rw [hz] at h1
    omega
  · have h2 : h_k * 1 ≤ h_k * (h_v / h_k) := Nat.mul_le_mul_left h_k hp
    omega

-- § span validation

/-- Two byte ranges overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> intro h <;> omega

/-- A span that ends before another begins is disjoint from it. -/
theorem disjoint_of_before (a an b bn : Nat) (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap; omega

/-- A span is valid when its address is 4-aligned and `address + bytes` does not wrap. -/
def ValidSpan (addr bytes max : Nat) : Prop := addr % 4 = 0 ∧ bytes ≤ max - addr

/-- A valid span ends within the address space. -/
theorem validSpan_end (addr bytes max : Nat) (h : ValidSpan addr bytes max) (ha : addr ≤ max) : addr + bytes ≤ max := by
  unfold ValidSpan at h; omega

/-- Byte sizes of the five inputs: `q`,`k` are `S*h_k`, `v` is `S*h_v`, gate/beta are `h_v` floats. -/
theorem input_bytes (h_k h_v : Nat) : (128 * h_k) * 4 = 512 * h_k ∧ (128 * h_v) * 4 = 512 * h_v ∧ h_v * 4 = 4 * h_v := by
  omega

/-- State bytes: `S * S * h_v * 4`. -/
theorem state_bytes (h_v : Nat) : 128 * 128 * h_v * 4 = 65536 * h_v := by omega

end Strata.NativeGdn
