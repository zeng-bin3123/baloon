/-!
# MODULE: prefill_cuda_kernels
Index arithmetic and integer-level contracts of the prefill CUDA kernels: hyper-connection read/write,
GDN conv / recurrence layout, top-10 router, 2-bit expert blob dequantisation, RoPE pairing, Q/K/V splitting,
paged KV append with INT8 quantisation, and the bf16 round-to-nearest-even bit trick.

## Source semantics
* `bf(f)` adds `0x7fff + ((u >> 16) & 1)` to the float bits and keeps the top 16 bits (round to nearest even).
* Streams are laid out `[token][stream c<4][dim d<2560]`, flat index `t*D + c*N + d`, `D = 10240`.
* GDN `h` rows are `[q: 16*128 | k: 16*128 | v: 48*128]`, `C = 10240`; q/k head is `head % 16`.
* Strata expert blob: gate/up codes `[1280][640 B]`, down codes `[2560][160 B]`, gate/up scales `[1280][40] f16`,
  down scales `[2560][10] f16`; total 1,382,400 bytes; one thread decodes one code byte (4 weights, value
  `(code - 1) * d`).
* Router: lane `l`, slot `i` owns expert `l + 32*i`; ten ranks select the top ten.
* RoPE rotates pairs `(p, p+32)` of the first 64 dims of a head.
* KV pool row = `(page*2 + kvh) * page_size + pos % page_size`; INT8 scale is `amax/127` per 64 values.

## Form template identifiers
bf16_round := bf ; stream_index := t*D + c*N + d ; gdn_row := C ; blob_codes_gu := O_GU_CODES ;
blob_codes_down := O_D_CODES ; blob_scale_gu := O_GU_SC ; blob_scale_down := O_D_SC ;
expert_of_lane := lane + 32*i ; rope_pair := (p, p+32) ; kv_row := (page*2+kvh)*page_size + pos%page_size
-/

namespace Strata.PrefillKernels

-- § constants
def N : Nat := 2560
def HC : Nat := 4
def D : Nat := N * HC
def S : Nat := 128
def HK : Nat := 16
def HV : Nat := 48
def C : Nat := 10240

/-- The GDN row is `q (16 heads) + k (16 heads) + v (48 heads)`, each head 128 wide. -/
theorem gdn_row_width : C = HK * S + HK * S + HV * S := by
  decide

/-- The q, k and v regions of a GDN row are laid out back to back without overlap. -/
theorem gdn_regions_disjoint (qh h : Nat) (a b : Nat) (hq : qh < HK) (ha : a < S) (hh : h < HV) (hb : b < S) :
    qh * S + a < HK * S ∧ HK * S + qh * S + a < 2 * HK * S ∧ 2 * HK * S + h * S + b < C := by
  unfold HK S HV C at *
  omega

/-- The q/k head index of a value head (`head % HK`) is a valid q/k head. -/
theorem qk_head_in_range (head : Nat) : head % HK < HK := by
  unfold HK
  omega

-- § hyper-connection

/-- Flat index of `(t, c, d)` in the four-stream residual. -/
def streamIndex (t c d : Nat) : Nat := t * D + c * N + d

/-- The write kernel recovers token, stream and dim from the flat index. -/
theorem stream_decompose (t c d : Nat) (hc : c < HC) (hd : d < N) :
    streamIndex t c d / D = t ∧ (streamIndex t c d % D) / N = c ∧ streamIndex t c d % N = d := by
  unfold streamIndex D HC N at *
  omega

/-- The stream index is within `T * D` for every token of a chunk. -/
theorem stream_index_lt (T t c d : Nat) (ht : t < T) (hc : c < HC) (hd : d < N) :
    streamIndex t c d < T * D := by
  unfold streamIndex D HC N at *
  have h : (t + 1) * 10240 ≤ T * 10240 := Nat.mul_le_mul_right 10240 ht
  omega

/-- The norm kernel's per-row stream index `row % HC` is a valid stream. -/
theorem norm_row_stream (row : Nat) : row % HC < HC := by
  unfold HC
  omega

-- § bf16 rounding

/-- Round-to-nearest-even of a 32-bit float pattern to its top 16 bits, as in `bf()`. -/
def bfBits (u : Nat) : Nat := (u + 32767 + (u / 65536) % 2) / 65536

/-- For every pattern below the infinity pattern the result fits 16 bits (no overflow of the add). -/
theorem bfBits_lt (u : Nat) (h : u < 2139095040) : bfBits u < 65536 := by
  unfold bfBits
  omega

/-- The rounded value is within half a bf16 ulp above the input. -/
theorem bfBits_upper (u : Nat) : bfBits u * 65536 ≤ u + 32768 := by
  unfold bfBits
  omega

/-- ... and within half a bf16 ulp below it. -/
theorem bfBits_lower (u : Nat) : u ≤ bfBits u * 65536 + 32768 := by
  unfold bfBits
  omega

/-- An exactly representable bf16 pattern (low 16 bits zero) is unchanged. -/
theorem bfBits_exact (hi : Nat) (h : hi < 32768) : bfBits (hi * 65536) = hi := by
  unfold bfBits
  omega

-- § MoE router

/-- Expert id owned by lane `l`, slot `i`. -/
def expertOf (l i : Nat) : Nat := l + 32 * i

/-- Every (lane, slot) pair names a valid expert of 512. -/
theorem expertOf_lt (l i : Nat) (hl : l < 32) (hi : i < 16) : expertOf l i < 512 := by
  unfold expertOf
  omega

/-- The selection loop recovers the slot as `ex / 32` and the lane as `ex % 32`. -/
theorem expertOf_inverse (l i : Nat) (hl : l < 32) : expertOf l i / 32 = i ∧ expertOf l i % 32 = l := by
  unfold expertOf
  omega

/-- Distinct (lane, slot) pairs give distinct experts, so no expert is read twice. -/
theorem expertOf_injective (l1 i1 l2 i2 : Nat) (h1 : l1 < 32) (h2 : l2 < 32)
    (h : expertOf l1 i1 = expertOf l2 i2) : l1 = l2 ∧ i1 = i2 := by
  unfold expertOf at h
  omega

/-- Ten ranks per token are written at `t*10 + rank`, inside `T*10`. -/
theorem route_out_index (T t r : Nat) (ht : t < T) (hr : r < 10) : t * 10 + r < T * 10 := by
  have h : (t + 1) * 10 ≤ T * 10 := Nat.mul_le_mul_right 10 ht
  omega

-- § expert blob decode

def O_GU_CODES : Nat := 0
def O_D_CODES : Nat := 1280 * 640
def O_GU_SC : Nat := O_D_CODES + 2560 * 160
def O_D_SC : Nat := O_GU_SC + 1280 * 40 * 2
def BLOB_BYTES : Nat := O_D_SC + 2560 * 10 * 2

/-- The four planes add up to exactly the architecture's 1,382,400-byte expert blob. -/
theorem blob_bytes_value : BLOB_BYTES = 1382400 := by
  decide

/-- The planes are laid out in increasing order without gaps. -/
theorem blob_planes_ordered : O_GU_CODES < O_D_CODES ∧ O_D_CODES < O_GU_SC ∧ O_GU_SC < O_D_SC ∧ O_D_SC < BLOB_BYTES := by
  decide

/-- One thread per code byte: gate/up and down code bytes together are the launch size. -/
theorem blob_threads : 1280 * 640 + 2560 * 160 = 1228800 := by
  decide

/-- Launch size is an exact multiple of the 256-thread block, so no thread is wasted or missing. -/
theorem blob_blocks_exact : 1228800 % 256 = 0 ∧ 1228800 / 256 = 4800 := by
  decide

/-- Gate/up scale index of a code byte is within the 40 scales of its row. -/
theorem gu_scale_index (byte : Nat) (h : byte < 640) : (byte * 4) / 64 < 40 := by
  omega

/-- Down scale index of a code byte is within the 10 scales of its row. -/
theorem down_scale_index (byte : Nat) (h : byte < 160) : (byte * 4) / 64 < 10 := by
  omega

/-- The four weights of one code byte share one scale (a 64-weight block). -/
theorem four_weights_share_scale (byte k : Nat) (hk : k < 4) : (byte * 4 + k) / 64 = (byte * 4) / 64 := by
  omega

/-- The decoded 2-bit code (`v` is the byte shifted right by `2k`) minus one lies in `{-1, 0, 1, 2}`. -/
theorem q2_value_range (v : Nat) : -1 ≤ ((v % 4 : Nat) : Int) - 1 ∧ ((v % 4 : Nat) : Int) - 1 ≤ 2 := by
  omega

/-- Output element index of weight `k` of byte `byte` in a gate/up row stays inside the 2560-wide row. -/
theorem gu_out_index (byte k : Nat) (hb : byte < 640) (hk : k < 4) : byte * 4 + k < 2560 := by
  omega

/-- ... and in a down row, inside the 640-wide row. -/
theorem down_out_index (byte k : Nat) (hb : byte < 160) (hk : k < 4) : byte * 4 + k < 640 := by
  omega

-- § SwiGLU and gathers

/-- The interleaved gate/up pair of hidden unit `k` is `(2k, 2k+1)` and stays inside the 1280 columns. -/
theorem swiglu_pair_index (k : Nat) (h : k < 640) : 2 * k + 1 < 1280 := by
  omega

/-- A gathered row is moved as `width / 8` 16-byte vectors; for width 2560 that is 320 and exact. -/
theorem gather_vectors : 2560 / 8 = 320 ∧ 2560 % 8 = 0 := by
  decide

-- § RoPE and Q/K/V helpers

/-- A rotary pair `(p, p+32)` stays inside the 64 rotated dims and the two halves never collide. -/
theorem rope_pair_disjoint (p : Nat) (h : p < 32) : p < 32 ∧ 32 ≤ p + 32 ∧ p + 32 < 64 := by
  omega

/-- The gate half of a fused query head sits right after its 256-dim query half. -/
theorem qgate_layout (h d : Nat) (hh : h < 24) (hd : d < 256) :
    h * 512 + d < h * 512 + 256 ∧ h * 512 + 256 + d < (h + 1) * 512 := by
  omega

/-- Split-q output index covers `T*24*256` exactly. -/
theorem split_q_total : 24 * 256 = 6144 := by
  decide

-- § paged KV append

/-- Pool row of a KV cell. -/
def kvRow (page kvh ps pos : Nat) : Nat := (page * 2 + kvh) * ps + pos % ps

/-- The row lies inside its page's two-head block. -/
theorem kvRow_lt (page kvh ps pos : Nat) (hkv : kvh < 2) (hps : 0 < ps) :
    kvRow page kvh ps pos < (page + 1) * 2 * ps := by
  unfold kvRow
  have h1 : pos % ps < ps := Nat.mod_lt pos hps
  have h2 : (page * 2 + kvh + 1) * ps = (page * 2 + kvh) * ps + ps := Nat.succ_mul _ _
  have h3 : (page + 1) * 2 * ps ≥ (page * 2 + kvh + 1) * ps := by
    apply Nat.mul_le_mul_right
    omega
  omega

/-- Clamp used by the INT8 quantiser. -/
def clamp127 (q : Int) : Int := max (-127) (min 127 q)

/-- Quantised codes are within `[-127, 127]`, so `-128` is never produced. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  omega

/-- In-range values are unchanged by the clamp. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  omega

/-- A 64-value group maps to a scale slot `row*4 + g` with `g < 4` for a 256-wide head. -/
theorem kv_scale_slot (row g d : Nat) (hg : g < 4) (hd : d < 64) : (g * 64 + d) / 64 = g ∧ row * 4 + g < (row + 1) * 4 := by
  omega

end Strata.PrefillKernels
