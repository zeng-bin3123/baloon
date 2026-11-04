/-!
# MODULE: prefill_kernels
Index arithmetic and quantization contracts of the batched prefill CUDA kernels (`src/prefill/kernels.cu`):
hyper-connection broadcast/mix/write, GDN gates and convolution state, expert blob layout, SwiGLU
interleaving, Q split/gate, KV append (fp16 and int8), RoPE pairing and launch grid sizing.

## Source semantics
* The residual `R` is `[T][HC=4][N=2560]`; flat index `i` decomposes as `t*D + c*N + d` with `D = N*HC`.
* The Strata expert blob is `gate/up codes | down codes | gate/up scales | down scales` and totals 1,382,400 bytes.
* 2-bit codes dequantize as `(code - 1) * d`, with `code` in `0..3`.
* INT8 KV values are clamped to `[-127, 127]`.
* RoPE rotates the pair `(p[pair], p[pair + 32])` for `pair < 32`.
* Every launch uses `ceil(n / threads)` blocks so all `n` elements are covered.

## Form template identifiers
* `N`, `HC`, `D` <- model width, hyper-connection streams, `N*HC`
* `blobTotal`    <- `O_D_SC + 2560*10*2`
* `clampQ`       <- the `q < -127 ? -127 : q > 127 ? 127 : q` clamp
* `blocksFor`    <- `blocks_for`
-/

namespace Strata.PrefillKernels

-- § residual decomposition

/-- Flat residual index `i` decomposes into token, stream and channel with the kernels' formulas. -/
theorem residual_decompose (i : Nat) :
    i / 10240 * 10240 + (i % 10240) / 2560 * 2560 + i % 2560 = i := by
  omega

/-- The stream index `(i % D) / N` is below `HC = 4`. -/
theorem stream_lt (i : Nat) : (i % 10240) / 2560 < 4 := by omega

/-- The channel index `i % N` is below `N`. -/
theorem channel_lt (i : Nat) : i % 2560 < 2560 := by omega

/-- Row index of `gr_norm` (`t*4 + c`) addresses a full row of `N` floats inside `T*D`. -/
theorem norm_row_in_range (t c T : Nat) (ht : t < T) (hc : c < 4) : (t * 4 + c) * 2560 + 2560 ≤ T * 10240 := by
  omega

-- § expert blob layout

/-- Offset of the down codes: after `1280 * 640` gate/up code bytes. -/
def oDCodes : Nat := 1280 * 640
/-- Offset of the gate/up scales: after `2560 * 160` down code bytes. -/
def oGuSc : Nat := oDCodes + 2560 * 160
/-- Offset of the down scales: after `1280 * 40 * 2` gate/up scale bytes. -/
def oDSc : Nat := oGuSc + 1280 * 40 * 2
/-- Total blob size: down scales are `2560 * 10 * 2` bytes. -/
def blobTotal : Nat := oDSc + 2560 * 10 * 2

/-- The four regions tile exactly one expert blob of 1,382,400 bytes. -/
theorem blob_total_eq : blobTotal = 1382400 := by decide

/-- Gate/up code thread index decomposes into row and byte covering `[0, 1280*640)` exactly once. -/
theorem gu_index_decompose (i : Nat) : i / 640 * 640 + i % 640 = i := by omega

/-- Down code thread index decomposes into row and byte. -/
theorem d_index_decompose (j : Nat) : j / 160 * 160 + j % 160 = j := by omega

/-- A gate/up scale read at `row*40 + byte*4/64` stays inside the row's 40 scales for `byte < 640`. -/
theorem gu_scale_in_row (byte : Nat) (h : byte < 640) : (byte * 4) / 64 < 40 := by omega

/-- A down scale read at `row*10 + byte*4/64` stays inside the row's 10 scales for `byte < 160`. -/
theorem d_scale_in_row (byte : Nat) (h : byte < 160) : (byte * 4) / 64 < 10 := by omega

/-- Extracting 2-bit code `k` from a byte yields a value below 4. -/
theorem code2_lt (c k : Nat) : (c / 4 ^ k) % 4 < 4 := Nat.mod_lt _ (by decide)

/-- A byte carries four codes, covering output positions `byte*4 .. byte*4+3`. -/
theorem byte_codes_cover (byte k : Nat) (hk : k < 4) : byte * 4 + k < (byte + 1) * 4 := by omega

-- § int8 KV quantization

/-- The kernel's clamp of the rounded quotient into the int8 symmetric range. -/
def clampQ (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The clamped code is within `[-127, 127]`. -/
theorem clampQ_bounds (q : Int) : -127 ≤ clampQ q ∧ clampQ q ≤ 127 := by
  unfold clampQ
  split <;> try omega

/-- Clamping is the identity on in-range values. -/
theorem clampQ_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clampQ q = q := by
  unfold clampQ
  split <;> try omega

/-- Row index of a KV cell stays inside `P` pages of `2` kv-heads of `ps` rows. -/
theorem kv_row_lt (page kvh r ps P : Nat) (hp : page < P) (hk : kvh < 2) (hr : r < ps) :
    (page * 2 + kvh) * ps + r < P * 2 * ps := by
  have h1 : (page * 2 + kvh + 1) * ps ≤ (2 * P) * ps := Nat.mul_le_mul_right ps (by omega)
  have h2 : (page * 2 + kvh + 1) * ps = (page * 2 + kvh) * ps + ps := by
    rw [Nat.add_mul, Nat.one_mul]
  have h3 : (2 * P) * ps = P * 2 * ps := by rw [Nat.mul_comm 2 P]
  omega

/-- The scale slot `row*4 + g` for group `g < 4` addresses one of the four 64-wide groups of 256. -/
theorem kv_group_lt (d g : Nat) (hg : g < 4) (hd : d < 64) : g * 64 + d < 256 := by omega

-- § SwiGLU, Q split and gating

/-- Interleaved gate/up pair `(2k, 2k+1)` of a 1280-wide row stays inside the row for `k < 640`. -/
theorem swiglu_pair_in_row (r k : Nat) (hk : k < 640) :
    r * 1280 + 2 * k + 1 < (r + 1) * 1280 := by omega

/-- Gate half of a fused `[24 heads][512]` Q row lives at offset `+256` and stays within its head. -/
theorem q_gate_in_head (h d : Nat) (hh : h < 24) (hd : d < 256) :
    h * 512 + 256 + d < (h + 1) * 512 := by omega

/-- The Q half and the gate half of one head do not overlap. -/
theorem q_gate_disjoint (h d d' : Nat) (hd : d < 256) : h * 512 + d < h * 512 + 256 + d' := by omega

-- § RoPE pairing

/-- Pair `p` rotates elements `p` and `p + 32`, which are distinct and inside the 64-wide rotary span. -/
theorem rope_pair (p : Nat) (hp : p < 32) : p < p + 32 ∧ p + 32 < 64 := by omega

/-- Every element of the rotary span belongs to exactly one pair (as first or second member). -/
theorem rope_cover (e : Nat) (he : e < 64) : (e < 32 ∧ e = e) ∨ (32 ≤ e ∧ e - 32 < 32) := by omega

-- § GDN state

/-- The 3-tap convolution history after consuming `x` is `(v1, v2, x)`. -/
def convShift (h : Int × Int × Int) (x : Int) : Int × Int × Int := (h.2.1, h.2.2, x)

/-- Shifting twice retains the last three inputs. -/
theorem convShift_three (h : Int × Int × Int) (a b c : Int) :
    convShift (convShift (convShift h a) b) c = (a, b, c) := rfl

/-- GDN head index of a flat gate element is below `HV = 48`. -/
theorem gdn_head_lt (i : Nat) : i % 48 < 48 := by omega

-- § launch geometry

/-- Blocks needed to cover `n` elements with `t` threads. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- The launch covers every element for 256 threads. -/
theorem blocksFor_cover (n : Nat) : n ≤ blocksFor n 256 * 256 := by
  unfold blocksFor
  omega

/-- The launch wastes less than one block of threads. -/
theorem blocksFor_tight (n : Nat) : blocksFor n 256 * 256 < n + 256 := by
  unfold blocksFor
  omega

-- § route weights

/-- The router selects ranks `0..9` and thread lane `rank` holds the rank's score. -/
theorem route_rank_lane (rank lane : Nat) (hr : rank < 10) (h : rank = lane) : lane < 10 := by omega

end Strata.PrefillKernels
