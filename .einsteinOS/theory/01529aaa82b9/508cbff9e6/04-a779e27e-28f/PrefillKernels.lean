/-!
# MODULE: prefill_kernels
Index arithmetic and layout contracts of the batched-prompt CUDA kernels (hyper-connection, GDN,
router, blob dequantization, KV append, QSA helpers).

## Source semantics
Flat thread indices are split into (token, channel, dim); every split must be a bijection onto the
buffer, every ceil-division launch must cover its range, and each lane/byte index must stay within
the layout (N = 2560, HC = 4, D = 10240, 2-bit Q2_0 codes, expert blob 1,382,400 B).

## Form template identifiers
* `N` → 2560, `D` → 10240, `HC` → 4, `blocks_for` → `blocksFor`, `bf`/`hf` rounding omitted
* `O_D_CODES`, `O_GU_SC`, `O_D_SC` → blob offsets
-/

namespace StrataPrefillKernels

-- § hyper-connection

/-- `gr_write`: the flat index splits into (token, channel, dim) bijectively. -/
theorem gr_write_decomp (i : Nat) :
    (i / 10240) * 10240 + ((i % 10240) / 2560) * 2560 + i % 2560 = i := by omega

/-- The channel index is below `HC = 4`. -/
theorem gr_write_chan_lt (i : Nat) : (i % 10240) / 2560 < 4 := by omega

/-- `d = i % N` agrees with the dim inside the token row. -/
theorem gr_write_d_consistent (i : Nat) : (i % 10240) % 2560 = i % 2560 := by omega

/-- `gr_broadcast` reads `e[t*N + d]`, which is inside `T*N`. -/
theorem gr_broadcast_index_lt (i T : Nat) (h : i < T * 10240) :
    (i / 10240) * 2560 + i % 2560 < T * 2560 := by omega

/-- `gr_mix` writes all four channel positions inside `T*D`. -/
theorem gr_mix_index_lt (i T c : Nat) (hc : c < 4) (h : i < T * 2560) :
    (i / 2560) * 10240 + c * 2560 + i % 2560 < T * 10240 := by omega

-- § launch geometry

/-- Ceil-division block count. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- A ceil-division launch covers every element. -/
theorem blocks_for_cover (n t : Nat) (ht : 0 < t) : n ≤ (n + t - 1) / t * t := by
  have h1 := Nat.div_add_mod (n + t - 1) t
  have h2 := Nat.mod_lt (n + t - 1) ht
  rw [Nat.mul_comm]
  generalize t * ((n + t - 1) / t) = p at *
  generalize (n + t - 1) % t = r at *
  omega

-- § GDN

/-- `gdn_rec_kernel`: 4 row groups of 32 rows tile the 128 state rows bijectively. -/
theorem gdn_row_tiling (rg r : Nat) (h1 : rg < 4) (h2 : r < 32) :
    rg * 32 + r < 128 ∧ (rg * 32 + r) / 32 = rg ∧ (rg * 32 + r) % 32 = r := by omega

-- § MoE

/-- Router lanes: expert id `lane + 32*i` splits back into lane and slot, below 512. -/
theorem route_lane_id (lane i : Nat) (hl : lane < 32) (hi : i < 16) :
    (lane + i * 32) % 32 = lane ∧ (lane + i * 32) / 32 = i ∧ lane + i * 32 < 512 := by omega

/-- Blob layout: offset of the down codes. -/
theorem blob_o_d_codes : 1280 * 640 = 819200 := by decide

/-- Blob layout: gate/up scales start after the down codes. -/
theorem blob_o_gu_sc : 1280 * 640 + 2560 * 160 = 1228800 := by decide

/-- Blob layout: down scales start after the gate/up scales. -/
theorem blob_o_d_sc : 1228800 + 1280 * 40 * 2 = 1331200 := by decide

/-- Blob layout: the down scales end exactly at the 1,382,400 byte blob size. -/
theorem blob_end : 1331200 + 2560 * 10 * 2 = 1382400 := by decide

/-- A 2-bit code minus the offset lies in `{-1, 0, 1, 2}`. -/
theorem dequant_code_range (c : Int) : -1 ≤ c % 4 - 1 ∧ c % 4 - 1 ≤ 2 := by omega

/-- Gate/up dequant: each code byte yields 4 weights inside a 2560-wide row. -/
theorem dequant_gu_index (byte k : Nat) (hb : byte < 640) (hk : k < 4) :
    byte * 4 + k < 2560 ∧ (byte * 4) / 64 < 40 := by omega

/-- Down dequant: each code byte yields 4 weights inside a 640-wide row. -/
theorem dequant_down_index (byte k : Nat) (hb : byte < 160) (hk : k < 4) :
    byte * 4 + k < 640 ∧ (byte * 4) / 64 < 10 := by omega

-- § QSA helpers

/-- `split_q` decomposition of the flat index into (token, head, dim). -/
theorem split_q_decomp (i : Nat) :
    (i / 6144) * 6144 + ((i / 256) % 24) * 256 + i % 256 = i := by omega

/-- RoPE rotates pairs `(p, p+32)` inside a 64-wide head. -/
theorem rope_pair_lt (pair : Nat) (h : pair < 32) : pair + 32 < 64 := by omega

/-- `kv_append`: `blockIdx.z` encodes (group, is_v) bijectively. -/
theorem kv_z_decomp (z : Nat) (h : z < 8) : z / 2 < 4 ∧ (z / 2) * 2 + z % 2 = z := by omega

/-- `kv_append`: the value index stays below 256. -/
theorem kv_dim_lt (g tid : Nat) (hg : g < 4) (ht : tid < 64) : g * 64 + tid < 256 := by omega

/-- Paged KV: position splits into page and in-page offset. -/
theorem kv_page_decomp (pos ps : Nat) (hps : 0 < ps) :
    ps * (pos / ps) + pos % ps = pos ∧ pos % ps < ps :=
  ⟨Nat.div_add_mod pos ps, Nat.mod_lt pos hps⟩

/-- Clamp used for int8 KV codes. -/
def clamp127 (q : Int) : Int := max (-127) (min 127 q)

/-- Quantized KV codes lie in `[-127, 127]`. -/
theorem clamp127_bounds (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127; omega

end StrataPrefillKernels
