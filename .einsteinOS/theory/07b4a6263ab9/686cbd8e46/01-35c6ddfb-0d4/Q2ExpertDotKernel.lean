/-!
# MODULE: q2_expert_dot_kernel
SOURCES: src/kernels/cpu/expert.cpp
The Q2_0 x Q8_1 expert kernel: activation quantization, the VNNI row dot with the code-offset
correction, the 512-bit block batching, the expert blob geometry and the CPU feature gate.

## Source semantics
* Row identity: sum_j (c_j - 1) d_w xq_j d_x = d_w (d_x sum c_j xq_j - d_x sum xq_j); the second term is
  the per-chunk `hx = scale * sum`, applied ONCE per chunk (not broadcast into 8 lanes).
* `chunk_dot_oracle` removes the offset in exact integer arithmetic: dot - sum = sum (c - 1) xq.
* Activations are clamped to [-127, 127]; codes are in [0, 3]; so a VNNI 4-product lane is at most
  1524 in magnitude and a 32-chunk integer sum at most 12192 < 2^24: the int -> f32 convert is exact.
* The multishift control 0x0E0C0A0806040200 selects shifts 0,2,..,14: output byte i of a u16 lane is
  the i-th 2-bit code in LSB-first order, matching the Q2_0 dequantizer.
* Blocks are batched 8 at a time with a tail `nb = min(8, n - b0)`; lane masks have 2*nb bits; the
  batches cover every block exactly once.
* Blob geometry (H = 2560, FF = 640): gate/up rows interleaved (gate 2r, up 2r+1), 40 blocks each;
  down rows 10 blocks each; total 1,382,400 bytes.
* The CPU gate requires AVX512 F, BW, VL, VNNI and VBMI (CPUID.7.0 EBX bits 16, 30, 31; ECX 11, 1).

## Form template identifiers
row_dot / row_dot_z -> rowDot, act_quant_q8_1 -> clampQ, ActQ.hx -> chunkHx, chunk_dot_oracle ->
chunkDotOracle, scales8 mask -> laneMask, b0/nb loop -> batchSizes, CpuFeatures::usable -> cpuUsable
-/

namespace Strata.Q2Expert

-- § the correction identity

/-- Sum of `f c x` over two equal-length lists. -/
def zsum (f : Int → Int → Int) : List Int → List Int → Int
  | c :: cs, x :: xs => f c x + zsum f cs xs
  | _, _ => 0

/-- Plain sum of a list. -/
def lsum : List Int → Int
  | [] => 0
  | x :: xs => x + lsum xs

/-- The code offset splits off as the plain activation sum: `sum (c-1) x = sum c x - sum x`. -/
theorem offset_identity (cs xs : List Int) (hl : cs.length = xs.length) :
    zsum (fun c x => (c - 1) * x) cs xs = zsum (fun c x => c * x) cs xs - lsum xs := by
  induction cs generalizing xs with
  | nil => cases xs with
    | nil => simp [zsum, lsum]
    | cons _ _ => simp at hl
  | cons c cs ih =>
    cases xs with
    | nil => simp at hl
    | cons x xs =>
      simp only [List.length_cons, Nat.add_right_cancel_iff] at hl
      simp only [zsum, lsum, ih xs hl]
      rw [Int.sub_mul, Int.one_mul]
      omega

/-- The chunk's correction term `hx = d_x * sum(xq)`. -/
def chunkHx (dx : Int) (xs : List Int) : Int := dx * lsum xs

/-- One chunk of the row dot as the kernel computes it equals the exact dequantized product
(over exact integer scales): `dw * (dx * sum c x - hx) = sum ((c-1) dw) (xq dx)`. -/
theorem chunk_kernel_exact (cs xs : List Int) (hl : cs.length = xs.length) (dw dx : Int) :
    dw * (dx * zsum (fun c x => c * x) cs xs - chunkHx dx xs) =
      dw * dx * zsum (fun c x => (c - 1) * x) cs xs := by
  rw [offset_identity cs xs hl, chunkHx, ← Int.mul_sub, Int.mul_assoc]

/-- `chunk_dot_oracle`: the integer VNNI dot minus the stored activation sum. -/
def chunkDotOracle (cs xs : List Int) : Int := zsum (fun c x => c * x) cs xs - lsum xs

/-- The oracle's integer result is exactly `sum (c - 1) xq` — no rounding before the scale. -/
theorem chunkDotOracle_exact (cs xs : List Int) (hl : cs.length = xs.length) :
    chunkDotOracle cs xs = zsum (fun c x => (c - 1) * x) cs xs := by
  unfold chunkDotOracle; rw [offset_identity cs xs hl]

-- § activation quantization ranges

/-- The clamp applied after rounding. -/
def clampQ (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Clamped activations lie in `[-127, 127]` and the clamp is idempotent. -/
theorem clampQ_range (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 ∧ clampQ (clampQ v) = clampQ v := by
  unfold clampQ
  split <;> split <;> simp_all <;> omega

/-- A VNNI lane adds four `u8 code * s8 act` products: bounded by 1524, far from int32 overflow. -/
theorem vnni_lane_bound (c0 c1 c2 c3 x0 x1 x2 x3 : Int)
    (hc : 0 ≤ c0 ∧ c0 ≤ 3 ∧ 0 ≤ c1 ∧ c1 ≤ 3 ∧ 0 ≤ c2 ∧ c2 ≤ 3 ∧ 0 ≤ c3 ∧ c3 ≤ 3)
    (hx : -127 ≤ x0 ∧ x0 ≤ 127 ∧ -127 ≤ x1 ∧ x1 ≤ 127 ∧ -127 ≤ x2 ∧ x2 ≤ 127 ∧ -127 ≤ x3 ∧ x3 ≤ 127) :
    -1524 ≤ c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3 ∧ c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3 ≤ 1524 := by
  have b : ∀ c x : Int, 0 ≤ c → c ≤ 3 → -127 ≤ x → x ≤ 127 → -381 ≤ c * x ∧ c * x ≤ 381 := by
    intro c x h1 h2 h3 h4
    have : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
    rcases this with rfl | rfl | rfl | rfl <;> omega
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := hc
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := hx
  have e0 := b c0 x0 a0 a1 y0 y1
  have e1 := b c1 x1 a2 a3 y2 y3
  have e2 := b c2 x2 a4 a5 y4 y5
  have e3 := b c3 x3 a6 a7 y6 y7
  omega

/-- Bound of a list of code-times-activation products: `|sum| <= 381 * length`. -/
theorem zsum_bound (cs xs : List Int) (hc : ∀ c ∈ cs, 0 ≤ c ∧ c ≤ 3) (hx : ∀ x ∈ xs, -127 ≤ x ∧ x ≤ 127) :
    -(381 * (cs.length : Int)) ≤ zsum (fun c x => c * x) cs xs ∧
      zsum (fun c x => c * x) cs xs ≤ 381 * (cs.length : Int) := by
  induction cs generalizing xs with
  | nil => simp [zsum]
  | cons c cs ih =>
    cases xs with
    | nil => simp [zsum]; omega
    | cons x xs =>
      have h1 := hc c (by simp)
      have h2 := hx x (by simp)
      have ihx := ih xs (fun c' h => hc c' (by simp [h])) (fun x' h => hx x' (by simp [h]))
      have hcx : -381 ≤ c * x ∧ c * x ≤ 381 := by
        have : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
        rcases this with rfl | rfl | rfl | rfl <;> omega
      simp only [zsum, List.length_cons]
      push_cast
      omega

/-- A 32-element chunk's integer dot is at most 12192 < 2^24, so its f32 conversion is exact. -/
theorem chunk_dot_exact_in_f32 (cs xs : List Int) (hlen : cs.length = 32)
    (hc : ∀ c ∈ cs, 0 ≤ c ∧ c ≤ 3) (hx : ∀ x ∈ xs, -127 ≤ x ∧ x ≤ 127) :
    -16777216 < zsum (fun c x => c * x) cs xs ∧ zsum (fun c x => c * x) cs xs < 16777216 := by
  have := zsum_bound cs xs hc hx
  rw [hlen] at this
  omega

-- § 2-bit unpack order

/-- Byte `i` of the multishift control constant. -/
def ctrlByte (i : Nat) : Nat := 0x0E0C0A0806040200 / 256 ^ i % 256

/-- The control selects shifts `0, 2, 4, ..., 14`. -/
theorem ctrl_shifts : (List.range 8).map ctrlByte = [0, 2, 4, 6, 8, 10, 12, 14] := by decide

/-- Output byte `i` of a u16 lane `b0 + 256 b1`, masked with 3, is the `i`-th code in LSB-first
order: codes 0..3 from byte 0, codes 4..7 from byte 1 (as in `dequantize_q2_0`). -/
theorem multishift_code_order (b0 b1 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    (b0 + 256 * b1) / 2 ^ 0 % 4 = b0 % 4 ∧ (b0 + 256 * b1) / 2 ^ 2 % 4 = b0 / 4 % 4 ∧
    (b0 + 256 * b1) / 2 ^ 4 % 4 = b0 / 16 % 4 ∧ (b0 + 256 * b1) / 2 ^ 6 % 4 = b0 / 64 % 4 ∧
    (b0 + 256 * b1) / 2 ^ 8 % 4 = b1 % 4 ∧ (b0 + 256 * b1) / 2 ^ 10 % 4 = b1 / 4 % 4 ∧
    (b0 + 256 * b1) / 2 ^ 12 % 4 = b1 / 16 % 4 ∧ (b0 + 256 * b1) / 2 ^ 14 % 4 = b1 / 64 % 4 := by
  simp only [Nat.reducePow]
  omega

-- § block batching in the 512-bit kernel

/-- Lane mask of `scales8`: `0xFFFF` for a full batch, else `(1 << 2nb) - 1`. -/
def laneMask (nb : Nat) : Nat := if nb ≥ 8 then 0xFFFF else 2 ^ (2 * nb) - 1

/-- The mask always enables exactly the `2 nb` chunk lanes of the batch. -/
theorem laneMask_spec : ∀ nb ∈ List.range 9, laneMask nb = 2 ^ (2 * nb) - 1 := by decide

/-- Batch sizes of the `b0 += 8` loop with tail `min(8, n - b0)` (fuel-bounded). -/
def batchSizes : Nat → Nat → List Nat
  | 0, _ => []
  | _, 0 => []
  | fuel + 1, n => min 8 n :: batchSizes fuel (n - min 8 n)

/-- Batches cover every block exactly once and none exceeds 8. -/
theorem batchSizes_cover (fuel n : Nat) (hf : n ≤ 8 * fuel) :
    (batchSizes fuel n).foldr (· + ·) 0 = n ∧ ∀ b ∈ batchSizes fuel n, 0 < b ∧ b ≤ 8 := by
  induction fuel generalizing n with
  | zero => have : n = 0 := by omega
            subst this; simp [batchSizes]
  | succ f ih =>
    cases n with
    | zero => simp [batchSizes]
    | succ m =>
      obtain ⟨h1, h2⟩ := ih (m + 1 - min 8 (m + 1)) (by omega)
      simp only [batchSizes, List.foldr, List.mem_cons]
      refine ⟨by omega, ?_⟩
      intro b hb
      rcases hb with rfl | hb
      · omega
      · exact h2 b hb

/-- Block `b` of a row uses activation chunks `2b` and `2b+1`, at int8 offsets `64b` and `64b + 32`:
the weight block spans two activation scale chunks. -/
theorem block_chunk_map (b : Nat) : (2 * b) * 32 = 64 * b ∧ (2 * b + 1) * 32 = 64 * b + 32 := by omega

-- § expert blob geometry

/-- Hidden and intermediate sizes (H from the qwen4exp guard, FF from the 1,382,400-byte blob). -/
def H : Nat := 2560
/-- See `H`. -/
def FF : Nat := 640

/-- Blocks per gate/up row and per down row, and the code/scale bytes per row. -/
theorem row_geometry : H / 64 = 40 ∧ FF / 64 = 10 ∧ 40 * 16 = 640 ∧ 10 * 16 = 160 := by decide

/-- Gate/up: 2 FF rows of (640 code + 80 scale) bytes; down: H rows of (160 + 20); sum 1,382,400. -/
theorem blob_size : 2 * FF * (640 + 80) + H * (160 + 20) = 1382400 := by decide

/-- Interleaved gate/up rows: row `r`'s gate is `2r` and up is `2r+1`; distinct and in range. -/
theorem gate_up_interleave (r : Nat) (hr : r < FF) :
    2 * r < 2 * FF ∧ 2 * r + 1 < 2 * FF ∧ (2 * r) / 2 = r ∧ (2 * r + 1) / 2 = r ∧ (2 * r) % 2 ≠ (2 * r + 1) % 2 := by
  unfold FF at *; omega

/-- Token-count dispatch of the `_multi` entry points: cases 1..7, default 8. -/
def dispatchNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- For 1..8 tokens the dispatched template width equals the token count. -/
theorem dispatchNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : dispatchNT n = n := by
  unfold dispatchNT; split <;> omega

-- § CPU feature gate

/-- Bit `k` of a CPUID register. -/
def bit (r k : Nat) : Bool := r / 2 ^ k % 2 = 1

/-- `CpuFeatures::usable`: AVX512 F (EBX.16), BW (EBX.30), VL (EBX.31), VNNI (ECX.11), VBMI (ECX.1). -/
def cpuUsable (ebx ecx : Nat) : Bool :=
  bit ebx 16 && bit ebx 30 && bit ebx 31 && bit ecx 11 && bit ecx 1

/-- Usable iff all five bits are set; dropping VBMI alone (an AVX512-VNNI part without VBMI)
makes the CPU unusable, which `reason()` reports individually. -/
theorem cpuUsable_iff (ebx ecx : Nat) :
    cpuUsable ebx ecx = true ↔
      (bit ebx 16 ∧ bit ebx 30 ∧ bit ebx 31 ∧ bit ecx 11 ∧ bit ecx 1) := by
  unfold cpuUsable; simp [Bool.and_eq_true, and_assoc]

/-- Witness: the minimal EBX/ECX words with exactly those bits are accepted; without VBMI rejected. -/
theorem cpuUsable_witness :
    cpuUsable (2 ^ 16 + 2 ^ 30 + 2 ^ 31) (2 ^ 11 + 2) = true ∧
    cpuUsable (2 ^ 16 + 2 ^ 30 + 2 ^ 31) (2 ^ 11) = false := by decide

end Strata.Q2Expert
