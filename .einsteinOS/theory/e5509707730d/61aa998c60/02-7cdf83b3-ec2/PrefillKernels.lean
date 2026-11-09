/-!
# MODULE: prefill_kernels
SOURCES: src/prefill/kernels.cu
Index arithmetic and rounding contracts of the batched-prefill CUDA kernels: the Strata expert blob layout and its
2-bit dequantization, BF16 round-to-nearest-even, hyper-connection indexing, the router lane layout, the QSA
split/gate indexing, KV-pool row addressing and the GDN channel regions.

## Source semantics
* A blob is `gate/up codes | down codes | gate/up scales (f16) | down scales (f16)` = 1,382,400 bytes.
* A 2-bit code `q` decodes to `(q - 1) * scale`, i.e. one of -1, 0, 1, 2 times the scale.
* `bf(x)` rounds a 32-bit float pattern to its top 16 bits, ties to even.
* All thread/block index decompositions stay in bounds and are exact (no element is skipped or doubled).

## Form template identifiers
N=2560, HC=4, D=10240, LR=320, S=128, HK=16, HV=48, C=10240; blob offsets O_D_CODES, O_GU_SC, O_D_SC;
`bf` -> bfRound; `blocks_for` -> blocksFor.
-/

namespace Strata.PrefillKernels

-- § expert blob layout

/-- Section offsets and the total blob size. -/
theorem blob_offsets :
    1280 * 640 = 819200 ∧ 819200 + 2560 * 160 = 1228800 ∧
    1228800 + 1280 * 40 * 2 = 1331200 ∧ 1331200 + 2560 * 10 * 2 = 1382400 := by decide

/-- Codes pack 4 weights per byte and scales cover groups of 64. -/
theorem blob_packing : 640 * 4 = 2560 ∧ 160 * 4 = 640 ∧ 2560 / 64 = 40 ∧ 640 / 64 = 10 := by decide

/-- Gate/up thread index split is exact and in range. -/
theorem gu_split (i : Nat) (hi : i < 1280 * 640) : i / 640 < 1280 ∧ i % 640 < 640 ∧ i = i / 640 * 640 + i % 640 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Gate/up output element and scale indices stay inside their planes. -/
theorem gu_decode_in_bounds (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 ∧ row * 40 + byte * 4 / 64 < 1280 * 40 := by
  constructor <;> omega

/-- Down-projection thread index split is exact and in range. -/
theorem d_split (j : Nat) (hj : j < 2560 * 160) : j / 160 < 2560 ∧ j % 160 < 160 ∧ j = j / 160 * 160 + j % 160 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Down output element and scale indices stay inside their planes. -/
theorem d_decode_in_bounds (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 ∧ row * 10 + byte * 4 / 64 < 2560 * 10 := by
  constructor <;> omega

/-- A 2-bit code decodes to a small multiple of the scale: `q - 1 ∈ [-1, 2]`. -/
theorem code_value_range (q : Nat) (h : q < 4) : -1 ≤ (q : Int) - 1 ∧ (q : Int) - 1 ≤ 2 := by omega

/-- Number of blocks of `t` threads covering `n` items. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- The launch grid covers every item. -/
theorem blocksFor_covers (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h := Nat.div_add_mod (n + t - 1) t
  have h2 := Nat.mod_lt (n + t - 1) ht
  have e : (n + t - 1) / t * t = t * ((n + t - 1) / t) := Nat.mul_comm _ _
  omega

-- § BF16 rounding

/-- Round-to-nearest-even of a 32-bit pattern to its upper 16 bits (as Nat arithmetic). -/
def bfRound (u : Nat) : Nat := (u + 32767 + (u / 65536) % 2) / 65536

/-- The rounding error is at most half a unit in the last place (32768 of 65536). -/
theorem bfRound_err (u : Nat) : u ≤ bfRound u * 65536 + 32768 ∧ bfRound u * 65536 ≤ u + 32768 := by
  unfold bfRound
  constructor <;> omega

/-- Exact ties round to an even mantissa. -/
theorem bfRound_tie_even (q : Nat) : bfRound (q * 65536 + 32768) % 2 = 0 := by
  unfold bfRound
  have h1 : (q * 65536 + 32768) / 65536 = q := by omega
  rw [h1]
  have hr : q % 2 < 2 := Nat.mod_lt _ (by decide)
  have h2 : ∀ r : Nat, r < 2 → (q * 65536 + 32768 + 32767 + r) / 65536 = q + r := by
    intro r hr'
    have hnum : q * 65536 + 32768 + 32767 + r = 65536 * q + (65535 + r) := by omega
    rw [hnum, Nat.mul_add_div (by decide : 65536 > 0)]
    rcases (by omega : r = 0 ∨ r = 1) with h | h <;> subst h <;> simp
  rw [h2 (q % 2) hr]
  omega

/-- Values already representable are unchanged. -/
theorem bfRound_exact (q : Nat) : bfRound (q * 65536) = q := by
  unfold bfRound
  omega

-- § hyper-connection indexing

/-- `gr_write` decomposition of a flat index into token, stream and channel is exact and in range. -/
theorem gr_write_index (i : Nat) :
    i / 10240 * 10240 + (i % 10240) / 2560 * 2560 + i % 2560 = i ∧ (i % 10240) / 2560 < 4 ∧ i % 2560 < 2560 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- `gr_mix` reads the four streams of one token at `t*D + c*N + d`, all below the token's row end. -/
theorem gr_mix_in_bounds (t c d : Nat) (hc : c < 4) (hd : d < 2560) : t * 10240 + c * 2560 + d < (t + 1) * 10240 := by
  omega

-- § router lane layout

/-- Lane `l` holds experts `l + 32 i`, a bijection onto `0..511`. -/
theorem route_lane_cover (lane i : Nat) (hl : lane < 32) (hi : i < 16) :
    (lane + i * 32) % 32 = lane ∧ (lane + i * 32) / 32 = i ∧ lane + i * 32 < 512 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- The 10 routed ids of token `t` occupy `[10 t, 10 t + 10)`. -/
theorem route_out_in_bounds (t r : Nat) (hr : r < 10) : t * 10 + r < (t + 1) * 10 := by omega

-- § QSA helpers

/-- `rope_kernel` row decomposition (`row = t * heads + h`) is exact and in range. -/
theorem rope_row_split (row heads T : Nat) (hh : 0 < heads) (hr : row < heads * T) :
    row / heads < T ∧ row % heads < heads := by
  exact ⟨Nat.div_lt_of_lt_mul hr, Nat.mod_lt row hh⟩

/-- Both rotated halves of a 64-dim rope head stay inside it. -/
theorem rope_pair_in_head (pair : Nat) (h : pair < 32) : pair < 64 ∧ pair + 32 < 64 := by
  constructor <;> omega

/-- `split_q` reads the first half of each 512-wide head slot: inside the token's `24*512` row. -/
theorem split_q_in_bounds (T i : Nat) (hi : i < T * 6144) :
    i / 6144 * 12288 + (i / 256 % 24) * 512 + i % 256 < T * 12288 := by omega

/-- `gate_attn` reads the second half (`+256`) of the same slot, still in bounds. -/
theorem gate_attn_in_bounds (T i : Nat) (hi : i < T * 6144) :
    i / 6144 * 12288 + (i / 256 % 24) * 512 + 256 + i % 256 < T * 12288 := by omega

/-- Interleaved SwiGLU reads `gu[r*1280 + 2k + 1]`, inside row `r`. -/
theorem swiglu_il_in_bounds (r k : Nat) (hk : k < 640) : r * 1280 + 2 * k + 1 < (r + 1) * 1280 := by omega

/-- Alpha/beta gates read `ab[t*96 + h]` and `ab[t*96 + 48 + h]`, inside token `t`'s 96 values. -/
theorem gates_in_bounds (t h : Nat) (hh : h < 48) : t * 96 + h < (t + 1) * 96 ∧ t * 96 + 48 + h < (t + 1) * 96 := by
  constructor <;> omega

/-- MoE combine reads expert row `slot * 2560 + d` for `slot < T * 10`. -/
theorem combine_in_bounds (T slot d : Nat) (hs : slot < T * 10) (hd : d < 2560) : slot * 2560 + d < T * 10 * 2560 := by
  omega

-- § GDN channel regions

/-- q (32 heads of 128 over 16 key heads x 2) and k occupy 4096 channels, v the remaining 6144 of 10240. -/
theorem gdn_regions : 2 * 16 * 128 = 4096 ∧ 4096 + 48 * 128 = 10240 := by decide

/-- The recurrence reads q at `qh*128`, k at `2048 + qh*128`, v at `4096 + head*128 + col`, all inside 10240. -/
theorem gdn_rec_reads (qh head col tid : Nat) (hq : qh < 16) (hh : head < 48) (hc : col < 128) (ht : tid < 128) :
    qh * 128 + tid < 2048 ∧ 2048 + qh * 128 + tid < 4096 ∧ 4096 + head * 128 + col < 10240 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

-- § KV append addressing

/-- Pool row of position `pos` for kv head `kvh` in page `page`: it stays inside that page's `2 * ps` rows. -/
theorem kv_row_in_page (page kvh pos ps : Nat) (hk : kvh < 2) (hps : 0 < ps) :
    (page * 2 + kvh) * ps + pos % ps < ((page + 1) * 2) * ps := by
  have hm : pos % ps < ps := Nat.mod_lt pos hps
  have h1 : (page * 2 + kvh + 1) * ps ≤ ((page + 1) * 2) * ps := Nat.mul_le_mul_right ps (by omega)
  have h2 : (page * 2 + kvh) * ps + ps = (page * 2 + kvh + 1) * ps := (Nat.succ_mul _ _).symm
  omega

/-- `blockIdx.z` encodes (group, K-or-V) exactly: groups 0..3, flag 0..1. -/
theorem kv_z_split (z : Nat) (h : z < 8) : z / 2 < 4 ∧ z % 2 < 2 ∧ 2 * (z / 2) + z % 2 = z := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Element index within a head: group * 64 + lane stays below 256. -/
theorem kv_elem_in_head (g lane : Nat) (hg : g < 4) (hl : lane < 64) : g * 64 + lane < 256 := by omega

end Strata.PrefillKernels
