/-!
# MODULE: prefill_kernels
SOURCES: src/prefill/kernels.cu
Index and layout contracts of the prefill CUDA kernels: hyper-connection stream indexing, the GDN convolution
history and recurrence tiling, MoE routing lane layout and tie-break, the 2-bit expert blob layout and its
dequantisation, QSA helper indexing, the int8 K/V append, and launch-grid coverage.

## Source semantics
* A residual is `T` rows of `HC = 4` streams of `N = 2560` floats; element `i` has token `i / 10240`, stream
  `(i % 10240) / 2560`, and channel `i % 2560`.
* GDN channels: `C = 10240 = 2*HK*S + HV*S` (q, k, v) with `HK = 16`, `HV = 48`, `S = 128`.
* The recurrence splits the 128 state rows over `RG = 4` thread rows of `RPG = 32`.
* The router keeps 16 logits per lane (`expert = lane + 32*i`), takes the top 10 with ties broken to the lower id.
* An expert blob is gate/up 2-bit codes, down codes, gate/up f16 scales, down f16 scales; one code byte holds 4
  weights with value `(code - 1) * scale`, one scale per 64 weights.
* The int8 K/V append quantises each 64-value group to `[-127, 127]` with an f16 scale.

## Form template identifiers
role mapping (source names -> abstract names):
  gr_norm/gr_mix/gr_write/gr_broadcast -> stream_kernels
  gdn_conv_kernel / gdn_rec_kernel     -> linear_recurrence_kernels
  route_kernel                         -> router_topk
  blob_dequant_kernel                  -> expert_blob_dequant
  kv_append_kernel                     -> kv_quant_append
  blocks_for                           -> grid_ceil
-/

namespace Strata.PrefillKernels

-- § shape relations

/-- `D = N * HC`. -/
theorem stream_width : 2560 * 4 = 10240 := by decide

/-- The GDN channel count splits into q, k and v parts. -/
theorem gdn_channels : 10240 = 2 * 16 * 128 + 48 * 128 := by decide

/-- The value part of the channels is the GDN value dimension `ZV`. -/
theorem gdn_value_dim : 48 * 128 = 6144 := by decide

/-- The q/k heads (32 of them) occupy the first 4096 channels. -/
theorem qk_prefix : 2 * 16 * 128 = 4096 := by decide

-- § stream indexing

/-- Token, stream and channel recompose to the flat index. -/
theorem stream_index_decomp (i : Nat) :
    (i / 10240) * 10240 + ((i % 10240) / 2560) * 2560 + i % 2560 = i := by
  omega

/-- The stream index is below 4 and the channel below 2560. -/
theorem stream_index_bounds (i : Nat) : (i % 10240) / 2560 < 4 ∧ i % 2560 < 2560 := by
  omega

/-- The token index of an in-range element is below `T`. -/
theorem token_index_lt (i T : Nat) (h : i < T * 10240) : i / 10240 < T := by
  omega

/-- Broadcast reads `e[t*N + d]`, which is inside the embedding buffer. -/
theorem broadcast_src_lt (i T : Nat) (h : i < T * 10240) :
    (i / 10240) * 2560 + i % 2560 < T * 2560 := by
  omega

/-- Elements of the same stream and channel share their source (stream-independence of the broadcast). -/
theorem broadcast_same_source (t c1 c2 d : Nat) (h1 : c1 < 4) (h2 : c2 < 4) (hd : d < 2560) :
    ((t * 10240 + c1 * 2560 + d) / 10240) * 2560 + (t * 10240 + c1 * 2560 + d) % 2560 =
    ((t * 10240 + c2 * 2560 + d) / 10240) * 2560 + (t * 10240 + c2 * 2560 + d) % 2560 := by
  omega

/-- The mixer reads the four stream copies of one channel at indices inside the token's row. -/
theorem mix_index_lt (i T c : Nat) (hi : i < T * 2560) (hc : c < 4) :
    (i / 2560) * 10240 + c * 2560 + i % 2560 < T * 10240 := by
  omega

/-- Norm rows: row `r = t*4 + c` has stream `r % 4` and token `r / 4`. -/
theorem norm_row_decomp (r : Nat) : (r / 4) * 4 + r % 4 = r ∧ r % 4 < 4 := by
  omega

-- § rotary pairs

/-- The two members of a rotary pair are distinct and both inside the 64 rotated dimensions. -/
theorem rope_pair_range (pair : Nat) (h : pair < 32) : pair < 64 ∧ pair + 32 < 64 ∧ pair ≠ pair + 32 := by
  omega

/-- A rotation preserves the squared norm of a pair when `c^2 + s^2 = 1` (exact over the integers). -/
theorem rope_preserves_norm (a b c s : Int) (h : c * c + s * s = 1) :
    (a * c - b * s) * (a * c - b * s) + (a * s + b * c) * (a * s + b * c) = a * a + b * b := by
  have key : (a * c - b * s) * (a * c - b * s) + (a * s + b * c) * (a * s + b * c) =
      (a * a + b * b) * (c * c + s * s) := by
    grind
  rw [key, h]
  omega

-- § GDN convolution history

/-- One convolution step shifts the 3-entry history and appends the new input. -/
def convStep (h : Nat × Nat × Nat) (x : Nat) : Nat × Nat × Nat := (h.2.1, h.2.2, x)

/-- History after a list of inputs. -/
def convRun (h : Nat × Nat × Nat) (xs : List Nat) : Nat × Nat × Nat := xs.foldl convStep h

/-- Appending an input shifts the newest two entries down. -/
theorem convRun_append (h : Nat × Nat × Nat) (xs : List Nat) (x : Nat) :
    convRun h (xs ++ [x]) = ((convRun h xs).2.1, (convRun h xs).2.2, x) := by
  simp [convRun, List.foldl_append, convStep]

/-- After a chunk the newest history entry is the last input. -/
theorem convRun_last (h : Nat × Nat × Nat) (xs : List Nat) (x : Nat) :
    (convRun h (xs ++ [x])).2.2 = x := by
  simp [convRun_append]

/-- Running two chunks in sequence equals running their concatenation. -/
theorem convRun_concat (h : Nat × Nat × Nat) (xs ys : List Nat) :
    convRun (convRun h xs) ys = convRun h (xs ++ ys) := by
  simp [convRun, List.foldl_append]

-- § recurrence tiling

/-- The four thread rows of 32 state rows cover the 128 state rows. -/
theorem rec_rows_cover : 4 * 32 = 128 := by decide

/-- State row of thread row `rg`, local row `r`. -/
theorem rec_row_lt (rg r : Nat) (h1 : rg < 4) (h2 : r < 32) : rg * 32 + r < 128 := by
  omega

/-- The kernel's base-plus-stride state addressing equals the direct (row, head, col) address. -/
theorem rec_state_addr (rg r head col : Nat) :
    ((rg * 32) * 48 + head) * 128 + col + r * (48 * 128) = ((rg * 32 + r) * 48 + head) * 128 + col := by
  omega

/-- Every state address is inside the `S * HV * S` state tensor of one layer. -/
theorem rec_state_addr_lt (rg r head col : Nat) (h1 : rg < 4) (h2 : r < 32) (h3 : head < 48) (h4 : col < 128) :
    ((rg * 32 + r) * 48 + head) * 128 + col < 128 * 48 * 128 := by
  omega

-- § router lanes

/-- Expert held by `lane` in slot `i`. -/
def expertOf (lane i : Nat) : Nat := lane + i * 32

/-- Lane and slot recover from the expert id. -/
theorem expert_lane_slot (lane i : Nat) (hl : lane < 32) :
    expertOf lane i % 32 = lane ∧ expertOf lane i / 32 = i := by
  unfold expertOf
  omega

/-- The 32 lanes times 16 slots cover exactly the 512 experts. -/
theorem expert_lt (lane i : Nat) (hl : lane < 32) (hi : i < 16) : expertOf lane i < 512 := by
  unfold expertOf
  omega

/-- Every expert id below 512 is held by some lane and slot. -/
theorem expert_covered (e : Nat) (h : e < 512) : e % 32 < 32 ∧ e / 32 < 16 ∧ expertOf (e % 32) (e / 32) = e := by
  unfold expertOf
  omega

/-- Candidate order of the butterfly reduction: larger value wins, ties go to the lower id. -/
def better (v1 : Int) (i1 : Nat) (v2 : Int) (i2 : Nat) : Prop := v1 > v2 ∨ (v1 = v2 ∧ i1 < i2)

/-- The order is asymmetric, so the reduction cannot flip-flop. -/
theorem better_asymm (v1 v2 : Int) (i1 i2 : Nat) (h : better v1 i1 v2 i2) : ¬ better v2 i2 v1 i1 := by
  unfold better at *
  omega

/-- Distinct experts are always strictly ordered, so the winner does not depend on the shuffle order. -/
theorem better_total (v1 v2 : Int) (i1 i2 : Nat) (h : i1 ≠ i2) :
    better v1 i1 v2 i2 ∨ better v2 i2 v1 i1 := by
  unfold better
  omega

/-- The router output slot `t*10 + rank` of one token is inside that token's 10 slots. -/
theorem route_slot_bounds (t rank : Nat) (h : rank < 10) :
    t * 10 ≤ t * 10 + rank ∧ t * 10 + rank < (t + 1) * 10 := by
  omega

-- § SwiGLU and Q/gate splitting

/-- Gate and up of the interleaved layout are adjacent and distinct inside a 1280-wide row. -/
theorem swiglu_il_bounds (k : Nat) (h : k < 640) : 2 * k + 1 < 1280 ∧ 2 * k ≠ 2 * k + 1 := by
  omega

/-- The q half and the gate half of a head's 512 values do not overlap. -/
theorem qgate_halves (h d : Nat) (hd : d < 256) :
    h * 512 + d < h * 512 + 256 + d ∧ h * 512 + 256 + d < (h + 1) * 512 := by
  omega

/-- The q index of the split kernel is inside the q buffer. -/
theorem split_q_lt (t h d T : Nat) (ht : t < T) (hh : h < 24) (hd : d < 256) :
    t * 6144 + h * 256 + d < T * 6144 := by
  omega

-- § 2-bit expert blob

/-- Codes of the gate/up matrices: 1280 rows of 640 bytes. -/
theorem blob_gu_codes : 1280 * 640 = 819200 := by decide

/-- Offset of the gate/up scales: after both code arrays. -/
theorem blob_gu_scale_off : 1280 * 640 + 2560 * 160 = 1228800 := by decide

/-- Offset of the down scales. -/
theorem blob_down_scale_off : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 = 1331200 := by decide

/-- Total blob size. -/
theorem blob_total : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

/-- Dequantisation threads are one per code byte: exactly the code arrays. -/
theorem dequant_threads : 1280 * 640 + 2560 * 160 = 1228800 := by decide

/-- Four weights per code byte give the full gate/up row width of 2560. -/
theorem gu_row_weights : 640 * 4 = 2560 := by decide

/-- Four weights per code byte give the full down row width of 640. -/
theorem down_row_weights : 160 * 4 = 640 := by decide

/-- A gate/up code byte maps to a scale group below 40 (one scale per 64 weights). -/
theorem gu_scale_group_lt (byte : Nat) (h : byte < 640) : (byte * 4) / 64 < 40 := by
  omega

/-- A down code byte maps to a scale group below 10. -/
theorem down_scale_group_lt (byte : Nat) (h : byte < 160) : (byte * 4) / 64 < 10 := by
  omega

/-- The four outputs of a code byte stay inside the destination row. -/
theorem gu_out_in_row (row byte k : Nat) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < (row + 1) * 2560 := by
  omega

/-- Dequantised value of a 2-bit code before scaling: `code - 1`. -/
def codeVal (x : Nat) : Int := ((x % 4 : Nat) : Int) - 1

/-- A 2-bit code decodes to one of `-1, 0, 1, 2`. -/
theorem codeVal_range (x : Nat) : -1 ≤ codeVal x ∧ codeVal x ≤ 2 := by
  unfold codeVal
  omega

-- § int8 K/V append

/-- Clamp of the rounded quotient to `[-127, 127]`. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The quantised value is always an int8 without `-128`. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  split <;> try split
  all_goals omega

/-- Clamping an in-range value changes nothing. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  split
  · omega
  · split
    · omega
    · rfl

/-- The grid's z coordinate encodes (group, is_v): both halves are in range. -/
theorem kv_z_decode (z : Nat) (h : z < 8) : z / 2 < 4 ∧ z % 2 < 2 := by
  omega

/-- The channel handled by a thread is below the 256 head dimension. -/
theorem kv_channel_lt (g tid : Nat) (hg : g < 4) (ht : tid < 64) : g * 64 + tid < 256 := by
  omega

/-- The row of a (page, kv head, position) triple is inside that page-head block. -/
theorem kv_row_in_block (page kvh pos ps : Nat) (hps : 0 < ps) :
    (page * 2 + kvh) * ps + pos % ps < (page * 2 + kvh + 1) * ps := by
  have h := Nat.mod_lt pos hps
  have e := Nat.add_mul (page * 2 + kvh) 1 ps
  omega

/-- The scale index `row * 4 + g` of a 256-channel row has room for 4 groups. -/
theorem kv_scale_index_lt (row g : Nat) (hg : g < 4) : row * 4 + g < (row + 1) * 4 := by
  omega

-- § launch grids

/-- Number of blocks of `t` threads to cover `n` elements. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- The launched threads cover every element. -/
theorem blocksFor_covers (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h := Nat.div_add_mod (n + t - 1) t
  have h2 : (n + t - 1) % t < t := Nat.mod_lt _ ht
  rw [Nat.mul_comm ((n + t - 1) / t) t]
  omega

/-- At most one block is wasted. -/
theorem blocksFor_tight (n t : Nat) (ht : 0 < t) : blocksFor n t * t < n + t := by
  unfold blocksFor
  have h := Nat.div_add_mod (n + t - 1) t
  rw [Nat.mul_comm ((n + t - 1) / t) t]
  omega

end Strata.PrefillKernels
