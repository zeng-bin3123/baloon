/-!
# MODULE: prefill_kernels
Index arithmetic and layout contracts of the batched-prefill CUDA kernels: hyper-connection write, 2-bit expert
blob dequantization, interleaved SwiGLU, QSA query split and gate, KV append, top-10 router lane layout.

## Source semantics
* `N = 2560`, `HC = 4`, `D = N*HC`; element `i` of `T*D` decomposes as token `i/D`, stream `(i%D)/N`, column `i%N`.
* The Strata expert blob is gate/up codes, down codes, gate/up scales (f16), down scales (f16): 1,382,400 bytes.
* A code byte holds four 2-bit weights; each decodes to `code - 1`.
* One block scale covers 64 weights; row-local scale indices stay in range.
* The router thread lane `l` owns experts `l + 32 i` for `i < 16`, covering all 512.

## Form template identifiers
* `gr_write_kernel` -> `grWrite*`, `blob_dequant_kernel` -> `blob*`, `route_kernel` -> `route*`
* `O_D_CODES`, `O_GU_SC`, `O_D_SC` -> `oDCodes`, `oGuSc`, `oDSc`
-/

namespace Strata.PrefillKernels

-- § hyper-connection

/-- Hidden width times four residual streams. -/
theorem d_value : 2560 * 4 = 10240 := by decide

/-- Decomposition of a flat index into token, stream and column is exact. -/
theorem grWrite_decompose (i : Nat) :
    (i / 10240) * 10240 + ((i % 10240) / 2560) * 2560 + i % 2560 = i := by
  omega

/-- The stream index of a flat element is one of the four streams. -/
theorem grWrite_stream_lt (i : Nat) : (i % 10240) / 2560 < 4 := by
  omega

/-- The column of a flat element is below the hidden width. -/
theorem grWrite_col_lt (i : Nat) : i % 2560 < 2560 := by
  omega

-- § blob layout

/-- Offset of the down codes: after `1280 * 640` gate/up code bytes. -/
def oDCodes : Nat := 1280 * 640

/-- Offset of the gate/up scales: after the down codes (`2560 * 160`). -/
def oGuSc : Nat := oDCodes + 2560 * 160

/-- Offset of the down scales: after `1280 * 40` f16 gate/up scales. -/
def oDSc : Nat := oGuSc + 1280 * 40 * 2

/-- Total bytes of one expert blob. -/
def blobBytes : Nat := oDSc + 2560 * 10 * 2

/-- The blob is exactly the 1,382,400 bytes the memory planner budgets per slot. -/
theorem blobBytes_value : blobBytes = 1382400 := by
  unfold blobBytes oDSc oGuSc oDCodes
  decide

/-- Scale offsets are laid out in strictly increasing order without overlap. -/
theorem blob_offsets_increasing : oDCodes < oGuSc ∧ oGuSc < oDSc ∧ oDSc < blobBytes := by
  unfold blobBytes oDSc oGuSc oDCodes
  decide

/-- Gate/up scale index for code byte `b` of a row stays inside the row's 40 scales. -/
theorem gu_scale_in_row (row b : Nat) (hb : b < 640) :
    row * 40 + (b * 4) / 64 < (row + 1) * 40 := by
  omega

/-- Down scale index for code byte `b` of a row stays inside the row's 10 scales. -/
theorem d_scale_in_row (row b : Nat) (hb : b < 160) :
    row * 10 + (b * 4) / 64 < (row + 1) * 10 := by
  omega

/-- Gate/up output position of weight `k` of byte `b` stays in the 2560-wide row. -/
theorem gu_out_in_row (b k : Nat) (hb : b < 640) (hk : k < 4) : b * 4 + k < 2560 := by
  omega

/-- Down output position of weight `k` of byte `b` stays in the 640-wide row. -/
theorem d_out_in_row (b k : Nat) (hb : b < 160) (hk : k < 4) : b * 4 + k < 640 := by
  omega

/-- A 2-bit code is at most three. -/
theorem code_le_three (x : Nat) : x &&& 3 ≤ 3 := Nat.and_le_right

set_option maxRecDepth 100000 in
/-- The four 2-bit fields of a byte reassemble the byte: the unpacking loses nothing. -/
theorem ternary_roundtrip : ∀ c : Fin 256,
    ((c.val &&& 3) + 4 * ((c.val >>> 2) &&& 3) + 16 * ((c.val >>> 4) &&& 3)
      + 64 * ((c.val >>> 6) &&& 3)) = c.val := by
  decide

/-- Decoded integer weight `code - 1` lies in `[-1, 2]`. -/
theorem decoded_range (q : Nat) (h : q ≤ 3) : (-1 : Int) ≤ (q : Int) - 1 ∧ (q : Int) - 1 ≤ 2 := by
  omega

-- § SwiGLU, QSA, KV

/-- Interleaved gate/up pair `(2k, 2k+1)` stays in the 1280-wide row for `k < 640`. -/
theorem swiglu_pair_in_row (k : Nat) (h : k < 640) : 2 * k + 1 < 1280 := by
  omega

/-- The gate half of a query head row starts after the 256 query values and stays in the 512-wide head. -/
theorem gate_offset_in_head (d : Nat) (h : d < 256) : 256 + d < 512 := by
  omega

/-- A block index `(z >> 1, z & 1)` over `z < 8` addresses four scale groups and K or V. -/
theorem kv_group_split (z : Nat) (h : z < 8) : z / 2 < 4 ∧ z % 2 < 2 := by
  omega

/-- Element index within a head of a (group, lane) block is below 256. -/
theorem kv_elem_lt (g t : Nat) (hg : g < 4) (ht : t < 64) : g * 64 + t < 256 := by
  omega

/-- The two rotary halves `pair` and `pair + 32` are both inside the 64-wide rotary span. -/
theorem rope_pairs_in_span (pair : Nat) (h : pair < 32) : pair < 64 ∧ pair + 32 < 64 := by
  omega

-- § router lanes

/-- Lane `l` owns expert `l + 32 i`; the owner of expert `e` is lane `e % 32`, slot `e / 32`. -/
theorem route_lane_inverse (lane i : Nat) (hl : lane < 32) :
    (lane + i * 32) % 32 = lane ∧ (lane + i * 32) / 32 = i := by
  omega

/-- Lane slots cover all 512 experts. -/
theorem route_cover (e : Nat) (h : e < 512) : e % 32 < 32 ∧ e / 32 < 16 := by
  omega

-- § GDN layout

/-- The channel count of the GDN qkv projection: q and k heads (2 * 16 * 128) plus v heads (48 * 128). -/
theorem gdn_c_value : 2 * 16 * 128 + 48 * 128 = 10240 := by decide

/-- Recurrence thread layout: four row groups of 32 rows cover the 128 state rows. -/
theorem gdn_row_groups : 4 * (128 / 4) = 128 := by decide

/-- A state row index `rg * 32 + r` with `rg < 4` and `r < 32` is below 128. -/
theorem gdn_row_lt (rg r : Nat) (h1 : rg < 4) (h2 : r < 32) : rg * 32 + r < 128 := by
  omega

end Strata.PrefillKernels
