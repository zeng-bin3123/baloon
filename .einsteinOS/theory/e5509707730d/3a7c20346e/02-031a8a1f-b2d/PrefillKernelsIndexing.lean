/-!
# MODULE: prefill_kernels_indexing
SOURCES: src/prefill/kernels.cu
CUDA prefill kernels: BF16 round-to-nearest-even, router tie-break order, residual-stream index decomposition,
2-bit expert blob layout and dequant indexing, KV append lane mapping, QSA q/gate split and launch sizing.

## Source semantics
* `bf(f)`: add `0x7fff + ((u>>16)&1)` and keep the high 16 bits (round to nearest, ties to even).
* Router: candidate `(v, i)` beats `(v', i')` iff `v > v'` or `v = v'` and `i < i'`; expert id is `lane + 32*i`.
* Residual `R[i]` with `N = 2560`, `HC = 4`, `D = N*HC`: `t = i/D`, `c = (i%D)/N`, `d = i%N`.
* Blob: gate/up codes `[1280][640 B]`, down codes `[2560][160 B]`, gate/up scales `[1280][40] f16`, down scales
  `[2560][10] f16`; each code byte holds four 2-bit weights, one scale per 64 weights.

## Form template identifiers
bf ↦ `bf`; better ↦ router `(ob > best || (ob == best && oi < ex))`; blocksFor ↦ `blocks_for`.
-/
namespace Strata.PrefillKernelsIndexing

-- § bf16 rounding

/-- BF16 rounding of the 32-bit pattern `u` (no overflow modelling needed below `2^32 - 2^16`). -/
def bf (u : Nat) : Nat := (u + 32767 + (u / 65536) % 2) / 65536

/-- Below the halfway point the result truncates. -/
theorem bf_below (u : Nat) (h : u % 65536 < 32768) : bf u = u / 65536 := by unfold bf; omega

/-- Above the halfway point the result rounds up. -/
theorem bf_above (u : Nat) (h : 32768 < u % 65536) : bf u = u / 65536 + 1 := by unfold bf; omega

/-- Exactly at the halfway point the result is even (ties to even). -/
theorem bf_tie_even (u : Nat) (h : u % 65536 = 32768) : bf u % 2 = 0 := by
  unfold bf
  have hu : u = 65536 * (u / 65536) + 32768 := by omega
  rcases Nat.mod_two_eq_zero_or_one (u / 65536) with h0 | h1
  · have e : (u + 32767 + u / 65536 % 2) / 65536 = u / 65536 := by omega
    rw [e]
    exact h0
  · have e : (u + 32767 + u / 65536 % 2) / 65536 = u / 65536 + 1 := by omega
    rw [e]
    omega

/-- The result is within one unit of truncation. -/
theorem bf_within_one (u : Nat) : u / 65536 ≤ bf u ∧ bf u ≤ u / 65536 + 1 := by unfold bf; omega

-- § router order

/-- Strict preference with ties broken toward the smaller index. -/
def better (v i v' i' : Nat) : Prop := v > v' ∨ (v = v' ∧ i < i')

/-- The order is irreflexive. -/
theorem better_irrefl (v i : Nat) : ¬ better v i v i := by unfold better; omega

/-- The order is asymmetric. -/
theorem better_asymm (v i v' i' : Nat) (h : better v i v' i') : ¬ better v' i' v i := by
  unfold better at *; omega

/-- Distinct candidates are always comparable (a unique winner exists). -/
theorem better_total (v i v' i' : Nat) (h : i ≠ i') : better v i v' i' ∨ better v' i' v i := by
  unfold better; omega

/-- Expert id `lane + 32*i` recovers `(lane, i)`. -/
theorem expert_id_roundtrip (lane i : Nat) (h : lane < 32) :
    (lane + 32 * i) % 32 = lane ∧ (lane + 32 * i) / 32 = i := by omega

/-- The 16 registers of 32 lanes cover exactly ids `0..511`. -/
theorem expert_id_lt (lane i : Nat) (hl : lane < 32) (hi : i < 16) : lane + 32 * i < 512 := by omega

/-- Surjective: every id below 512 is some `(lane, i)`. -/
theorem expert_id_surj (e : Nat) (h : e < 512) : e % 32 < 32 ∧ e / 32 < 16 ∧ e % 32 + 32 * (e / 32) = e := by
  omega

-- § residual stream indices

/-- `i = t*D + c*N + d` with `c < HC` for the real constants. -/
theorem residual_decomp (i : Nat) :
    i = (i / 10240) * 10240 + ((i % 10240) / 2560) * 2560 + i % 2560 ∧ (i % 10240) / 2560 < 4 := by
  omega

/-- Broadcast reads `e[t*N + i % N]`, the same element for every one of the four streams. -/
theorem broadcast_same_d (i : Nat) : i % 2560 = (i % 10240) % 2560 := by omega

-- § launch sizing

/-- `blocks_for n = (n + 255) / 256`. -/
def blocksFor (n : Nat) : Nat := (n + 255) / 256

/-- The grid covers all `n` threads. -/
theorem blocksFor_cover (n : Nat) : n ≤ blocksFor n * 256 := by unfold blocksFor; omega

/-- And launches fewer than one extra block. -/
theorem blocksFor_tight (n : Nat) : blocksFor n * 256 < n + 256 := by unfold blocksFor; omega

-- § blob layout

/-- Gate/up code bytes. -/
theorem blob_off_d_codes : 1280 * 640 = 819200 := by decide

/-- Gate/up scale plane begins after the down codes. -/
theorem blob_off_gu_sc : 1280 * 640 + 2560 * 160 = 1228800 := by decide

/-- Down scale plane begins after the gate/up scales (`[1280][40]` f16). -/
theorem blob_off_d_sc : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 = 1331200 := by decide

/-- The blob ends after the down scales (`[2560][10]` f16). -/
theorem blob_total : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

/-- Gate/up dequant: code byte, scale pair and 4 output weights are all in range. -/
theorem gu_dequant_in_range (row byte : Nat) (hr : row < 1280) (hb : byte < 640) :
    row * 640 + byte < 1280 * 640 ∧
    (row * 40 + byte * 4 / 64) * 2 + 2 ≤ 1280 * 40 * 2 ∧
    row * 2560 + byte * 4 + 3 < 1280 * 2560 := by omega

/-- Down dequant: same, with 160 bytes and 10 groups per row. -/
theorem d_dequant_in_range (row byte : Nat) (hr : row < 2560) (hb : byte < 160) :
    row * 160 + byte < 2560 * 160 ∧
    (row * 10 + byte * 4 / 64) * 2 + 2 ≤ 2560 * 10 * 2 ∧
    row * 640 + byte * 4 + 3 < 2560 * 640 := by omega

/-- Each 2-bit code is below 4, so `code - 1 ∈ {-1,0,1,2}`. -/
theorem code2_lt (c k : Nat) : (c / 2 ^ (2 * k)) % 4 < 4 := Nat.mod_lt _ (by omega)

/-- The thread index space splits at `n_gu` into gate/up and down bytes. -/
theorem thread_split : 1280 * 640 + 2560 * 160 = 1228800 := by decide

-- § KV append and QSA helpers

/-- `blockIdx.z` encodes (group, is_v) bijectively over 0..7. -/
theorem kv_z_decode (z : Nat) (h : z < 8) : z / 2 < 4 ∧ (z / 2) * 2 + z % 2 = z := by omega

/-- Each 64-thread block writes lanes `g*64 + t < 256`. -/
theorem kv_lane_lt (g t : Nat) (hg : g < 4) (ht : t < 64) : g * 64 + t < 256 := by omega

/-- Scale slot `row*4 + g` is inside the 4-per-row scale array. -/
theorem kv_scale_slot (row g : Nat) (hg : g < 4) : row * 4 + g < (row + 1) * 4 := by omega

/-- q half and gate half of a head's 512 floats are disjoint and cover it. -/
theorem qgate_halves (h d : Nat) (hd : d < 256) :
    h * 512 + d < h * 512 + 256 ∧ h * 512 + 256 + d < (h + 1) * 512 := by omega

/-- SwiGLU interleaved reads `gu[r*1280 + 2k]` and `+1` for `k < 640`, inside the row. -/
theorem swiglu_il_in_row (r k : Nat) (hk : k < 640) : r * 1280 + 2 * k + 1 < (r + 1) * 1280 := by omega

/-- GDN recurrence row groups: `rg * 32 + r` covers `0..127` for `rg < 4`, `r < 32`. -/
theorem rec_rows (rg r : Nat) (hg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- RoPE pairs `(p, p+32)` stay in a 64-wide rotary block. -/
theorem rope_pair_in_block (p : Nat) (h : p < 32) : p + 32 < 64 := by omega

/-- `gather_rows16` moves whole `uint4` units when the width is a multiple of 8. -/
theorem gather_units (width : Nat) (h : width % 8 = 0) : width / 8 * 8 = width := by omega

end Strata.PrefillKernelsIndexing
