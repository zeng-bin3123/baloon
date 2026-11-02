/-!
# MODULE: prefill_gpu_kernels
Index arithmetic, memory layouts and rounding contracts of the batched-prefill CUDA kernels
(hyper-connection, GDN, router, expert blob dequantisation, SwiGLU, QSA helpers, KV append).

## Source semantics
* Geometry: N = 2560, HC = 4, D = N*HC = 10240, LR = 320, S = 128, HK = 16, HV = 48,
  C = 2*HK*S + HV*S = 10240.
* `blocks_for(n, t) = (n + t - 1)/t` blocks; the guard `i >= n` makes the surplus threads idle.
* `bf` is round-to-nearest-even on the upper 16 bits of the fp32 pattern.
* A Strata expert blob is gate/up codes `[1280][640]`, down codes `[2560][160]`, then fp16 scales
  `[1280][40]` and `[2560][10]`; each code byte holds four 2-bit weights, value `(code - 1) * d`.
* `route_kernel`: one warp per token, 16 logits per lane (`lane + 32 i`), top-10 with ties to the lowest
  expert index.
* `kv_append`: grid `(T, 2, 8)`; `z>>1` picks the 64-value group, `z&1` picks K or V.

## Form template identifiers
blocks_for -> grid_blocks ; bf -> bf16_round ; blob_dequant_kernel -> blob_unpack ; route_kernel -> top10_route ;
warp_sum -> warp_xor_reduce ; kv_append_kernel -> kv_page_write ; gdn_rec_kernel -> gdn_recurrence
-/

namespace Strata.Prefill.Kernels

/-- `x < x/P*P + P`. -/
theorem lt_div_mul_add' (x P : Nat) (hP : 0 < P) : x < x / P * P + P := by
  have h1 : x / P * P + x % P = x := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod x P
  have h2 := Nat.mod_lt x hP
  omega

-- § geometry

/-- The model dimensions are mutually consistent. -/
theorem geometry_consistent :
    (2560 : Nat) * 4 = 10240 ∧ 2 * 16 * 128 + 48 * 128 = 10240 ∧ 48 * 128 = 6144 ∧
      80 * 128 = 10240 ∧ 128 / 4 = 32 ∧ 4 * 128 = 512 := by decide

-- § launch sizing

/-- The grid covers all `n` work items. -/
theorem blocks_for_covers (n t : Nat) (ht : 0 < t) : n ≤ (n + t - 1) / t * t := by
  have := lt_div_mul_add' (n + t - 1) t ht
  omega

/-- The grid overshoots by less than one block. -/
theorem blocks_for_tight (n t : Nat) : (n + t - 1) / t * t ≤ n + t - 1 :=
  Nat.div_mul_le_self _ _

/-- A block-wide reduction needs at most 32 warp partials for blockDim ≤ 1024. -/
theorem block_sum_warps (b : Nat) (h : b ≤ 1024) : (b + 31) / 32 ≤ 32 := by omega

-- § hyper-connection kernels

/-- `gr_mix`: thread index decomposes into (token, channel). -/
theorem decomp_N (i T : Nat) (h : i < T * 2560) :
    i / 2560 < T ∧ i / 2560 * 2560 + i % 2560 = i ∧ i % 2560 < 2560 := by omega

/-- `gr_mix`: reading stream `c` of token `t` stays inside the `T*D` residual. -/
theorem gr_mix_index (t c d T : Nat) (ht : t < T) (hc : c < 4) (hd : d < 2560) :
    t * 10240 + c * 2560 + d < T * 10240 := by omega

/-- `gr_write`: (token, stream, channel) decomposition of a residual element. -/
theorem gr_write_decomp (i T : Nat) (h : i < T * 10240) :
    i / 10240 < T ∧ i % 10240 / 2560 < 4 ∧ i % 10240 % 2560 < 2560 ∧ i % 2560 = i % 10240 % 2560 := by
  omega

/-- `gr_norm`: one block per (token, stream) row; the row slice is inside the residual. -/
theorem gr_norm_row (row T : Nat) (h : row < T * 4) :
    row / 4 < T ∧ row % 4 < 4 ∧ row * 2560 + 2559 < T * 10240 := by omega

-- § GDN kernels

/-- `gdn_gates`: both ab halves are inside the `T * 2*HV` buffer. -/
theorem gdn_gate_index (t h T : Nat) (ht : t < T) (hh : h < 48) :
    t * 96 + h < T * 96 ∧ t * 96 + 48 + h < T * 96 := by omega

/-- `gdn_gates`: flat index decomposition. -/
theorem gdn_gate_decomp (i T : Nat) (h : i < T * 48) : i / 48 < T ∧ i % 48 < 48 := by omega

/-- `gdn_conv`: 80 blocks of 128 threads are exactly the 10240 channels. -/
theorem gdn_conv_grid : (10240 : Nat) / 128 * 128 = 10240 := by decide

/-- `gdn_conv`: the history and weight slices of channel `c` are inside their arrays. -/
theorem gdn_conv_index (c : Nat) (hc : c < 10240) : c * 3 + 2 < 30720 ∧ c * 4 + 3 < 40960 := by
  omega

/-- `gdn_rec`: q, k and v reads of a head stay in their own regions of the 10240-wide row. -/
theorem gdn_rec_regions (head col : Nat) (hh : head < 48) (hc : col < 128) :
    head % 16 * 128 + col < 2048 ∧ 2048 + head % 16 * 128 + col < 4096 ∧
      4096 + head * 128 + col < 10240 := by omega

/-- `gdn_rec`: the row groups tile the 128 state rows. -/
theorem gdn_rec_rows (rg r : Nat) (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- `gdn_rec`: every state row has exactly one (group, offset). -/
theorem gdn_rec_rows_unique (row : Nat) (h : row < 128) :
    row / 32 < 4 ∧ row % 32 < 32 ∧ row / 32 * 32 + row % 32 = row := by omega

/-- `gdn_rec`: state element address stays inside the `[128][HV][S]` state. -/
theorem gdn_rec_state_index (row head col : Nat) (hrow : row < 128) (hh : head < 48) (hc : col < 128) :
    row * 6144 + head * 128 + col < 128 * 6144 := by omega

-- § router

/-- Lane `l`, slot `i` addresses expert `l + 32 i < 512`. -/
theorem route_lane_index (l i : Nat) (hl : l < 32) (hi : i < 16) : l + i * 32 < 512 := by omega

/-- The lane/slot to expert map is injective: every expert is seen once. -/
theorem route_lane_inj (l i l' i' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : l + i * 32 = l' + i' * 32) : l = l' ∧ i = i' := by omega

/-- Candidate `(b', i')` replaces `(b, i)`: larger value, or equal value with a smaller index. -/
def better (b i b' i' : Int) : Prop := b < b' ∨ (b = b' ∧ i' < i)

/-- The shuffle tie-break is irreflexive. -/
theorem better_irrefl (b i : Int) : ¬ better b i b i := by
  unfold better
  omega

/-- The shuffle tie-break is asymmetric (a deterministic winner). -/
theorem better_asymm (b i b' i' : Int) (h : better b i b' i') : ¬ better b' i' b i := by
  unfold better at *
  omega

/-- The shuffle tie-break is transitive (the butterfly merge order does not matter). -/
theorem better_trans (b i b' i' b'' i'' : Int) (h1 : better b i b' i') (h2 : better b' i' b'' i'') :
    better b i b'' i'' := by
  unfold better at *
  omega

-- § warp reduction

/-- XOR with the same offset twice is the identity (butterfly partners are mutual). -/
theorem xor_involution (i o : Nat) : (i ^^^ o) ^^^ o = i := by
  simp [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- Butterfly partners stay inside the warp. -/
theorem butterfly_in_range : ∀ i : Fin 32, ∀ o ∈ [16, 8, 4, 2, 1], (i.val ^^^ o) < 32 := by
  decide

/-- The path of the five xor steps from lane `i` guided by `d`. -/
def butterflyPath (i d : Nat) : Nat :=
  ((((i ^^^ (d &&& 16)) ^^^ (d &&& 8)) ^^^ (d &&& 4)) ^^^ (d &&& 2)) ^^^ (d &&& 1)

set_option maxRecDepth 100000 in
/-- Every lane's value can reach every other lane through the five xor steps: full all-reduce coverage. -/
theorem butterfly_reaches_all :
    ∀ i : Fin 32, ∀ j : Fin 32, butterflyPath i.val (i.val ^^^ j.val) = j.val := by
  unfold butterflyPath
  decide

-- § blob layout and dequantisation

/-- Byte offset of the down codes. -/
def O_D_CODES : Nat := 1280 * 640
/-- Byte offset of the gate/up scales. -/
def O_GU_SC : Nat := O_D_CODES + 2560 * 160
/-- Byte offset of the down scales. -/
def O_D_SC : Nat := O_GU_SC + 1280 * 40 * 2
/-- Total blob size. -/
def BLOB_BYTES : Nat := O_D_SC + 2560 * 10 * 2

/-- The four sections are contiguous and total 1,382,400 bytes. -/
theorem blob_layout :
    O_D_CODES = 819200 ∧ O_GU_SC = 1228800 ∧ O_D_SC = 1331200 ∧ BLOB_BYTES = 1382400 := by decide

/-- Thread index split for gate/up code bytes. -/
theorem blob_gu_split (i : Nat) (h : i < 1280 * 640) :
    i / 640 < 1280 ∧ i % 640 < 640 ∧ i / 640 * 640 + i % 640 = i := by omega

/-- Gate/up scale index of a byte is inside the 40 scales of its row (one scale per 64 weights). -/
theorem blob_gu_scale_index (byte : Nat) (h : byte < 640) : byte * 4 / 64 < 40 := by omega

/-- Down scale index of a byte is inside the 10 scales of its row. -/
theorem blob_d_scale_index (byte : Nat) (h : byte < 160) : byte * 4 / 64 < 10 := by omega

/-- Gate/up output write of four weights stays inside `[1280][2560]`. -/
theorem blob_gu_out_index (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by omega

/-- Down output write of four weights stays inside `[2560][640]`. -/
theorem blob_d_out_index (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by omega

/-- Four 2-bit fields reconstruct the byte. -/
theorem code_pack (c : Nat) (hc : c < 256) :
    c % 4 + 4 * (c / 4 % 4) + 16 * (c / 16 % 4) + 64 * (c / 64 % 4) = c := by omega

/-- Each 2-bit field is a valid code. -/
theorem code_field_lt (c k : Nat) : c / 4 ^ k % 4 < 4 := Nat.mod_lt _ (by decide)

/-- The decoded multiplier `code - 1` lies in {-1, 0, 1, 2}. -/
theorem code_multiplier_range (x : Nat) (hx : x < 4) : -1 ≤ (x : Int) - 1 ∧ (x : Int) - 1 ≤ 2 := by
  omega

-- § SwiGLU and gather

/-- Interleaved SwiGLU reads `gu[r*1280 + 2k]` and `+1`, inside row `r`. -/
theorem swiglu_il_index (r k : Nat) (hk : k < 640) : r * 1280 + 2 * k + 1 < (r + 1) * 1280 := by
  omega

-- § QSA helpers

/-- `split_q`: the query half and the gate half of a 512-wide head row are disjoint and in bounds. -/
theorem split_q_index (t h d T : Nat) (ht : t < T) (hh : h < 24) (hd : d < 256) :
    t * 12288 + h * 512 + d < T * 12288 ∧ t * 12288 + h * 512 + 256 + d < T * 12288 := by omega

/-- `gate_attn`/`split_q` output index. -/
theorem q_out_index (t h d T : Nat) (ht : t < T) (hh : h < 24) (hd : d < 256) :
    t * 6144 + h * 256 + d < T * 6144 := by omega

-- § KV append

/-- Grid z picks the 64-value group (4 groups) and the 256-wide head row is covered. -/
theorem kv_grid (z tid : Nat) (hz : z < 8) (ht : tid < 64) :
    z / 2 < 4 ∧ z / 2 * 64 + tid < 256 := by omega

/-- (group, K/V) is a bijection with z. -/
theorem kv_grid_bijective (z : Nat) (hz : z < 8) : z / 2 * 2 + z % 2 = z := by omega

/-- The pool row of a position stays inside its page. -/
theorem kv_row_in_page (m ps r : Nat) (hr : r < ps) : m * ps + r < (m + 1) * ps := by
  have e : (m + 1) * ps = m * ps + ps := Nat.succ_mul m ps
  omega

/-- The scale index of a pool row stays in its row. -/
theorem kv_scale_index (row g : Nat) (hg : g < 4) : row * 4 + g < (row + 1) * 4 := by omega

/-- int8 clamp of the quantised value. -/
def clamp127 (q : Int) : Int := max (-127) (min 127 q)

/-- The clamped code is in [-127, 127] (never -128). -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  omega

-- § bf16 rounding

/-- Round-to-nearest-even on the upper 16 bits (Nat model of `bf`). -/
def bfRound (u : Nat) : Nat := (u + 0x7fff + (u / 65536) % 2) / 65536

/-- A value that is already bf16 (low 16 bits zero) is unchanged. -/
theorem bfRound_exact (u : Nat) (h : u % 65536 = 0) : bfRound u = u / 65536 := by
  unfold bfRound
  omega

/-- The rounding error is at most half a unit in the last place. -/
theorem bfRound_error (u : Nat) :
    u ≤ bfRound u * 65536 + 32768 ∧ bfRound u * 65536 ≤ u + 32768 := by
  unfold bfRound
  omega

/-- Ties go to the even mantissa. -/
theorem bfRound_tie_even (u : Nat) (h : u % 65536 = 32768) : bfRound u % 2 = 0 := by
  unfold bfRound
  have hx : u = 65536 * (u / 65536) + 32768 := by omega
  generalize u / 65536 = x at hx ⊢
  subst hx
  have hm : x % 2 < 2 := Nat.mod_lt _ (by decide)
  generalize hmd : x % 2 = m at hm ⊢
  have e : (65536 * x + 32768 + 32767 + m) / 65536 = x + m := by omega
  rw [e]
  omega

/-- No carry out of 16 bits for patterns away from the top of the range. -/
theorem bfRound_no_overflow (u : Nat) (h : u + 32768 < 4294967296) : bfRound u < 65536 := by
  unfold bfRound
  omega

end Strata.Prefill.Kernels
