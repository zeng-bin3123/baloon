/-!
# MODULE: cpu_expert_kernel
SOURCES: src/kernels/cpu/expert.cpp
The Q2_0 expert kernel on x86 (AVX-512 VNNI/VBMI): fp16 scale decoding, the Q8_1 activation quantizer, the
2-bit unpack, the offset-corrected dot product, the 8-block scale masks, token-count dispatch, and the CPU
feature gate. This module models the arithmetic identities and index contracts these routines rely on.

## Source semantics
* `sum_j (c_j - 1) d_w xhat_j d_x == d_w * (d_x * sum(c*xhat) - d_x * sum(xhat))`, split per 32-chunk.
* Quantized activations are clamped to [-127, 127]; a 32-chunk sum therefore fits easily in int32.
* The fp16 subnormal path normalizes the mantissa with at most 10 shifts and a biased exponent `113 - s`.
* A 2-bit code is `(x % 256) % 4 = x % 4` of the multishifted word; codes are at most 3, so `maddubs`
  never saturates against an int8 activation.
* Token counts 1..7 select that template; everything else selects 8; the multi path needs `1 <= nt <= 8`.

## Form template identifiers
* `normShift`     : f16_subnormal_normalize
* `clampQ`        : q8_clamp
* `dotOff`, `dotRaw` : offset_dot, raw_dot
* `maskOf`        : block_tail_mask
* `dispatchNT`    : token_template_dispatch
* `usable`, `missing` : cpu_feature_gate
-/

namespace Strata.CpuExpert

-- § fp16 decoding

/-- Number of left shifts the subnormal loop performs (bounded by `fuel`). -/
def normShift : Nat → Nat → Nat
  | _, 0 => 0
  | m, f + 1 => if 1024 ≤ m then 0 else 1 + normShift (2 * m) f

/-- The shift count never exceeds the fuel. -/
theorem normShift_le (m f : Nat) : normShift m f ≤ f := by
  induction f generalizing m with
  | zero => simp [normShift]
  | succ f ih =>
    have e : normShift m (f + 1) = if 1024 ≤ m then 0 else 1 + normShift (2 * m) f := rfl
    rw [e]
    split
    · omega
    · have := ih (2 * m)
      omega

/-- After the shifts the mantissa has bit 10 set and stays below 2048. -/
theorem normShift_spec (f : Nat) : ∀ m, 0 < m → m < 2048 → 1024 ≤ m * 2 ^ f →
    1024 ≤ m * 2 ^ normShift m f ∧ m * 2 ^ normShift m f < 2048 := by
  induction f with
  | zero =>
    intro m h0 h1 h2
    have e : normShift m 0 = 0 := rfl
    rw [e]
    simp only [Nat.pow_zero, Nat.mul_one] at h2 ⊢
    exact ⟨h2, h1⟩
  | succ f ih =>
    intro m h0 h1 h2
    have e : normShift m (f + 1) = if 1024 ≤ m then 0 else 1 + normShift (2 * m) f := rfl
    rw [e]
    by_cases hm : 1024 ≤ m
    · rw [if_pos hm]
      simp only [Nat.pow_zero, Nat.mul_one]
      exact ⟨hm, h1⟩
    · rw [if_neg hm]
      have h2' : 1024 ≤ (2 * m) * 2 ^ f := by
        have e3 : m * 2 ^ (f + 1) = (2 * m) * 2 ^ f := by
          rw [Nat.pow_succ]
          ac_rfl
        rw [e3] at h2
        exact h2
      have ih' := ih (2 * m) (by omega) (by omega) h2'
      have e2 : m * 2 ^ (1 + normShift (2 * m) f) = (2 * m) * 2 ^ normShift (2 * m) f := by
        rw [Nat.pow_add, Nat.pow_one]
        ac_rfl
      rw [e2]
      exact ih'

/-- A nonzero fp16 subnormal mantissa (`0 < man < 1024`) normalizes within 10 shifts. -/
theorem f16_subnormal_normalizes (man : Nat) (h0 : 0 < man) (h1 : man < 1024) :
    normShift man 10 ≤ 10 ∧ 1024 ≤ man * 2 ^ normShift man 10 ∧ man * 2 ^ normShift man 10 < 2048 := by
  have hb : 1024 ≤ man * 2 ^ 10 := by omega
  have hs := normShift_spec 10 man h0 (by omega) hb
  exact ⟨normShift_le man 10, hs.1, hs.2⟩

/-- The biased f32 exponent `113 - s` of a subnormal stays a normal f32 exponent. -/
theorem subnormal_exp_range (s : Nat) (h : s ≤ 10) : 0 < 113 - s ∧ 113 - s < 255 := by
  omega

/-- The three fp16 fields reassemble the 16-bit word. -/
theorem f16_fields (h : Nat) (hh : h < 65536) :
    (h / 32768) * 32768 + ((h / 1024) % 32) * 1024 + h % 1024 = h := by
  omega

-- § activation quantization

/-- Clamp of the rounded quotient to the symmetric int8 range used by Q8_1. -/
def clampQ (v : Int) : Int := max (-127) (min 127 v)

/-- Clamped values lie in [-127, 127]. -/
theorem clampQ_bounds (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ
  omega

/-- Clamping is idempotent. -/
theorem clampQ_idem (v : Int) : clampQ (clampQ v) = clampQ v := by
  unfold clampQ
  omega

/-- The range is symmetric, so clamping commutes with negation. -/
theorem clampQ_neg (v : Int) : clampQ (-v) = - clampQ v := by
  unfold clampQ
  omega

/-- In-range values are untouched. -/
theorem clampQ_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clampQ v = v := by
  unfold clampQ
  omega

/-- A list of in-range values has sum bounded by 127 per element. -/
theorem sum_abs_bound : ∀ l : List Int, (∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) →
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | x :: xs, h => by
    have hx := h x (by simp)
    have ih := sum_abs_bound xs (fun y hy => h y (by simp [hy]))
    simp only [List.sum_cons, List.length_cons]
    push_cast
    omega

/-- A 32-value chunk sum is within ±4064, far inside int32. -/
theorem chunk_sum_fits (l : List Int) (hl : l.length = 32) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -4064 ≤ l.sum ∧ l.sum ≤ 4064 := by
  have hb := sum_abs_bound l h
  rw [hl] at hb
  obtain ⟨hb1, hb2⟩ := hb
  exact ⟨by omega, by omega⟩

-- § offset-corrected dot product

/-- Dot product with the Q2_0 code offset removed: `sum (c-1)*x`. -/
def dotOff : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | c :: cs, x :: xs => (c - 1) * x + dotOff cs xs

/-- Raw dot product `sum c*x`. -/
def dotRaw : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | c :: cs, x :: xs => c * x + dotRaw cs xs

/-- The offset dot equals the raw dot minus the activation sum: the `- d_x*sum(xhat)` correction. -/
theorem dotOff_eq : ∀ (cs xs : List Int), cs.length = xs.length →
    dotOff cs xs = dotRaw cs xs - xs.sum
  | [], [], _ => by simp [dotOff, dotRaw]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | c :: cs, x :: xs, h => by
    have ih := dotOff_eq cs xs (by simpa using h)
    simp only [dotOff, dotRaw, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega

/-- The offset dot is additive over chunk splits, which justifies the per-32-chunk scale handling. -/
theorem dotOff_append : ∀ (c1 x1 c2 x2 : List Int), c1.length = x1.length →
    dotOff (c1 ++ c2) (x1 ++ x2) = dotOff c1 x1 + dotOff c2 x2
  | [], [], c2, x2, _ => by simp [dotOff]
  | [], _ :: _, _, _, h => by simp at h
  | _ :: _, [], _, _, h => by simp at h
  | c :: cs, x :: xs, c2, x2, h => by
    have ih := dotOff_append cs xs c2 x2 (by simpa using h)
    simp only [List.cons_append, dotOff]
    rw [ih, Int.add_assoc]

-- § 2-bit unpack

/-- Masking the multishifted byte with 3 equals masking the whole word with 3. -/
theorem code_byte_mask (x : Nat) : (x % 256) % 4 = x % 4 := by
  omega

/-- A 2-bit code is at most 3. -/
theorem code_lt_four (x : Nat) : x % 4 < 4 :=
  Nat.mod_lt _ (by decide)

/-- Products of a code (0..3) with an int8 activation are at most 381 in magnitude. -/
theorem code_mul_bound (c q : Int) (h0 : 0 ≤ c) (h3 : c ≤ 3) (hq1 : -127 ≤ q) (hq2 : q ≤ 127) :
    -381 ≤ c * q ∧ c * q ≤ 381 := by
  interval_cases c <;> omega

/-- A pair-sum of two such products fits an int16 lane, so `maddubs` never saturates. -/
theorem maddubs_no_saturation (c1 c2 q1 q2 : Int) (h1 : 0 ≤ c1) (h2 : c1 ≤ 3) (h3 : 0 ≤ c2)
    (h4 : c2 ≤ 3) (hq1 : -127 ≤ q1) (hq2 : q1 ≤ 127) (hq3 : -127 ≤ q2) (hq4 : q2 ≤ 127) :
    -32768 ≤ c1 * q1 + c2 * q2 ∧ c1 * q1 + c2 * q2 ≤ 32767 := by
  have a := code_mul_bound c1 q1 h1 h2 hq1 hq2
  have b := code_mul_bound c2 q2 h3 h4 hq3 hq4
  exact ⟨by omega, by omega⟩

/-- Low four codes of a lane word `b0 + 256*b1` come from byte `b0`. -/
theorem lane_low (b0 b1 i : Nat) (hb : b0 < 256) (hi : i < 4) :
    (b0 + 256 * b1) / 4 ^ i % 4 = b0 / 4 ^ i % 4 := by
  interval_cases i <;> omega

/-- High four codes come from byte `b1`. -/
theorem lane_high (b0 b1 j : Nat) (hb : b0 < 256) (hj : j < 4) :
    (b0 + 256 * b1) / 4 ^ (j + 4) % 4 = b1 / 4 ^ j % 4 := by
  interval_cases j <;> omega

-- § block tail masks and scale permutes

/-- Lane mask for the last group of `nb <= 8` blocks (two lanes per block). -/
def maskOf (nb : Nat) : Nat := if 8 ≤ nb then 65535 else 2 ^ (2 * nb) - 1

/-- The mask has exactly `2*nb` low bits set. -/
theorem maskOf_bits (nb : Nat) (h : nb ≤ 8) : maskOf nb + 1 = 2 ^ (2 * nb) := by
  interval_cases nb <;> decide

/-- A group never runs past the end: `b0 + nb <= nblocks` and `nb >= 1` while blocks remain. -/
theorem group_bounds (b0 nblocks : Nat) (h : b0 < nblocks) :
    1 ≤ min 8 (nblocks - b0) ∧ b0 + min 8 (nblocks - b0) ≤ nblocks := by
  refine ⟨?_, ?_⟩ <;> omega

/-- Scale and correction loads index `2*(b0+i)+h`, which stays below `2*nblocks`. -/
theorem scale_index_lt (b0 i h nb nblocks : Nat) (hi : i < nb) (hnb : b0 + nb ≤ nblocks) (hh : h < 2) :
    2 * (b0 + i) + h < 2 * nblocks := by
  omega

/-- The permute index `lane/8 + 2*i` selects the product for block `i`, half `lane/8`, within 16 lanes. -/
theorem permute_index (i l : Nat) (hi : i < 8) (hl : l < 16) :
    l / 8 + 2 * i < 16 ∧ (l / 8 + 2 * i) / 2 = i ∧ (l / 8 + 2 * i) % 2 = l / 8 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- The gate/up scale arrays at hidden size 2560 hold 40 blocks, a whole number of 8-block groups. -/
theorem real_gu_blocks : 2560 / 64 = 40 ∧ 40 % 8 = 0 := by
  decide

-- § token-count dispatch

/-- Template argument chosen for `n_tokens`: 1..7 select that value, anything else selects 8. -/
def dispatchNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- The selected template never exceeds 8 and is at least 1. -/
theorem dispatchNT_range (n : Nat) : 1 ≤ dispatchNT n ∧ dispatchNT n ≤ 8 := by
  unfold dispatchNT
  split <;> omega

/-- For every legal token count `1..8` the template equals the count. -/
theorem dispatchNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : dispatchNT n = n := by
  unfold dispatchNT
  split <;> omega

/-- The GGUF Q2_0 block stride 18 = 2 scale bytes + 16 code bytes, and 16 bytes hold 64 codes. -/
theorem gguf_block_layout : 2 + 16 = 18 ∧ 16 * 4 = 64 := by
  decide

-- § CPU feature gate

/-- All five required features present. -/
def usable (f bw vl vnni vbmi : Bool) : Bool := f && bw && vl && vnni && vbmi

/-- Names of the missing features, in the order the diagnostic prints them. -/
def missing (f bw vl vnni vbmi : Bool) : List String :=
  (if f then [] else ["AVX512F"]) ++ (if bw then [] else ["AVX512BW"]) ++
  (if vl then [] else ["AVX512VL"]) ++ (if vnni then [] else ["AVX512-VNNI"]) ++
  (if vbmi then [] else ["AVX512-VBMI"])

/-- The CPU is usable exactly when nothing is reported missing. -/
theorem usable_iff_missing_nil (f bw vl vnni vbmi : Bool) :
    usable f bw vl vnni vbmi = true ↔ missing f bw vl vnni vbmi = [] := by
  cases f <;> cases bw <;> cases vl <;> cases vnni <;> cases vbmi <;> simp [usable, missing]

end Strata.CpuExpert
