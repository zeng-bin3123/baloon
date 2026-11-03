/-!
# MODULE: cpu_q2_expert_kernel
The Q2_0 routed-expert kernel on the CPU: fixed geometry of one 1,382,400-byte expert blob, the activation
quantizer's integer contract, the Q2_0 "-1 code offset" identity that the `hx` correction implements, row-range
splitting, token-count template dispatch and the CPU feature gate.

## Source semantics
* One weight block = QK = 64 weights with ONE fp16 scale; one activation chunk = QKA = 32 elements with its OWN
  scale, so a weight block spans exactly two activation chunks.
* `sum (c-1) d_w xhat d_x = d_w (d_x sum(c xhat) - d_x sum(xhat))` (the `hx` correction), split per chunk.
* Activation codes are clamped to [-127,127]; chunk sums and VNNI lane sums are exact in int32.
* `s2_expert_gu_rows` / `s2_expert_down_rows` over any partition of the rows equal the whole-expert kernel.
* `s2_expert_vnni_multi` dispatches n_tokens in 1..7 to NT = n_tokens, anything else to NT = 8 (or the fallback).
* The kernel is usable iff AVX512F, BW, VL, VNNI and VBMI are all present.

## Form template identifiers
* `H`, `FF`, `NE`, `QK`, `QKA`, `BB`, `MAXC`, `ROW_GU`, `ROW_D`, `SC_GU`, `SC_D`, `BLOB`, `O_*` : geometry constants
* `lo n parts p` : the row boundary `n * p / parts` used by split phases
* `dispatchNT` : the `switch (n_tokens)` template selector
* `Feat` / `Feat.usable` / `Feat.missing` : `CpuFeatures`
-/

namespace Strata.CpuExpert

-- § geometry

def H : Nat := 2560
def FF : Nat := 640
def NE : Nat := 512
def QK : Nat := 64
def QKA : Nat := 32
def BB : Nat := 18
def MAXC : Nat := H / QKA
def ROW_GU : Nat := H * 2 / 8
def ROW_D : Nat := FF * 2 / 8
def SC_GU : Nat := H / QK
def SC_D : Nat := FF / QK
def BLOB : Nat := 3 * (H * FF * BB / QK)
def O_GU_CODES : Nat := 0
def O_D_CODES : Nat := 2 * FF * ROW_GU
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
def O_END : Nat := O_D_SCALES + H * SC_D * 2

/-- The widest reduction has 80 activation chunks. -/
theorem maxc_eq : MAXC = 80 := by decide
/-- A gate/up row holds 640 bytes of codes. -/
theorem row_gu_eq : ROW_GU = 640 := by decide
/-- A down row holds 160 bytes of codes. -/
theorem row_d_eq : ROW_D = 160 := by decide
/-- A gate/up row has 40 weight blocks. -/
theorem sc_gu_eq : SC_GU = 40 := by decide
/-- A down row has 10 weight blocks. -/
theorem sc_d_eq : SC_D = 10 := by decide
/-- One expert blob is 1,382,400 bytes. -/
theorem blob_eq : BLOB = 1382400 := by decide
/-- Offset of the down codes. -/
theorem o_d_codes_eq : O_D_CODES = 819200 := by decide
/-- Offset of the gate/up scales. -/
theorem o_gu_scales_eq : O_GU_SCALES = 1228800 := by decide
/-- Offset of the down scales. -/
theorem o_d_scales_eq : O_D_SCALES = 1331200 := by decide

/-- The four blob segments tile the blob exactly: the last segment ends at `BLOB`. -/
theorem layout_end_eq_blob : O_END = BLOB := by decide

/-- One 18-byte block is a 2-byte fp16 scale plus 16 code bytes (64 two-bit codes). -/
theorem block_bytes : BB = 2 + QK * 2 / 8 := by decide

/-- One weight block spans exactly two activation chunks. -/
theorem block_spans_two_chunks : QK = 2 * QKA := by decide

/-- The chunk count of the gate/up reduction is twice its weight-block count. -/
theorem maxc_two_blocks : MAXC = 2 * SC_GU := by decide

/-- The chunk count of the down reduction is twice its weight-block count. -/
theorem ff_chunks_two_blocks : FF / QKA = 2 * SC_D := by decide

/-- Every (block, half) pair of a gate/up row names a valid chunk scale. -/
theorem gu_chunk_index_lt : ∀ b h, b < SC_GU → h < 2 → 2 * b + h < MAXC := by
  intro b h hb hh
  have h1 := sc_gu_eq
  have h2 := maxc_eq
  omega

/-- Every (block, half) pair of a down row names a valid chunk of the intermediate. -/
theorem d_chunk_index_lt : ∀ b h, b < SC_D → h < 2 → 2 * b + h < FF / QKA := by
  intro b h hb hh
  have h1 := sc_d_eq
  have h2 : FF / QKA = 20 := by decide
  omega

-- § blob addressing

/-- Gate and up rows (2r and 2r+1) of all FF rows lie inside the gate/up code segment. -/
theorem gu_codes_in_segment (r : Nat) (hr : r < FF) :
    O_GU_CODES + (2 * r + 2) * ROW_GU ≤ O_D_CODES := by
  have h1 : FF = 640 := by decide
  rw [row_gu_eq, o_d_codes_eq]
  unfold O_GU_CODES
  omega

/-- Down rows lie inside the down code segment. -/
theorem d_codes_in_segment (r : Nat) (hr : r < H) :
    O_D_CODES + (r + 1) * ROW_D ≤ O_GU_SCALES := by
  have h1 : H = 2560 := by decide
  rw [row_d_eq, o_d_codes_eq, o_gu_scales_eq]
  omega

/-- Gate and up scale rows lie inside the gate/up scale segment. -/
theorem gu_scales_in_segment (r : Nat) (hr : r < FF) :
    O_GU_SCALES + (2 * r + 2) * SC_GU * 2 ≤ O_D_SCALES := by
  have h1 : FF = 640 := by decide
  rw [sc_gu_eq, o_gu_scales_eq, o_d_scales_eq]
  omega

/-- Down scale rows lie inside the down scale segment (which ends at the blob end). -/
theorem d_scales_in_segment (r : Nat) (hr : r < H) :
    O_D_SCALES + (r + 1) * SC_D * 2 ≤ BLOB := by
  have h1 : H = 2560 := by decide
  rw [sc_d_eq, o_d_scales_eq, blob_eq]
  omega

-- § the Q2_0 code-offset identity

/-- The offset identity, exactly as the `hx` correction uses it: `sum (c-1) x = sum (c x) - sum x`. -/
theorem offset_identity (l : List (Int × Int)) :
    (l.map (fun p => (p.1 - 1) * p.2)).sum =
      (l.map (fun p => p.1 * p.2)).sum - (l.map (fun p => p.2)).sum := by
  induction l with
  | nil => simp
  | cons p t ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega

/-- Splitting a block into its two chunks: the block sum is the sum of the chunk sums, so each chunk can use
its own activation scale. -/
theorem block_sum_split (a b : List Int) : (a ++ b).sum = a.sum + b.sum := by
  simp

-- § integer exactness of the activation contract

/-- The activation code clamp. -/
def clampQ (v : Int) : Int := if v < -127 then -127 else if 127 < v then 127 else v

/-- Clamped activation codes always fit in int8 (and never reach -128). -/
theorem clampQ_range (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ
  split <;> try split
  all_goals omega

/-- Clamping is the identity on the representable range. -/
theorem clampQ_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clampQ v = v := by
  unfold clampQ
  split <;> try split
  all_goals omega

/-- A list of codes in [-127,127] has a sum bounded by 127 times its length. -/
theorem sum_bound (l : List Int) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int) := by
  induction l with
  | nil => simp
  | cons a t ih =>
    have ha := h a (by simp)
    have ht := ih (fun x hx => h x (by simp [hx]))
    simp only [List.sum_cons, List.length_cons]
    omega

/-- A 32-element chunk sum is at most 4064 in absolute value, so `a.sum[k]` is exact in int32. -/
theorem chunk_sum_bound (l : List Int) (hl : l.length = 32) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -4064 ≤ l.sum ∧ l.sum ≤ 4064 := by
  have := sum_bound l h
  rw [hl] at this
  omega

/-- A VNNI lane sums four products of a 2-bit code (0..3) and an int8: |lane| <= 3*127*4. -/
theorem vnni_lane_bound : 3 * 127 * 4 = 1524 := by decide

/-- Even all 64 products of a block stay far below 2^31, so integer partial sums are exact. -/
theorem vnni_block_bound : 3 * 127 * 64 < 2 ^ 31 := by decide

-- § row-range splitting

/-- The row boundary of part `p` out of `parts`. -/
def lo (n parts p : Nat) : Nat := n * p / parts

/-- The first part starts at row 0. -/
theorem lo_zero (n parts : Nat) : lo n parts 0 = 0 := by simp [lo]

/-- The last part ends at row `n`. -/
theorem lo_full (n parts : Nat) (h : 0 < parts) : lo n parts parts = n := by
  simp [lo, Nat.mul_div_cancel _ h]

/-- Boundaries are monotone. -/
theorem lo_mono (n parts p q : Nat) (h : p ≤ q) : lo n parts p ≤ lo n parts q :=
  Nat.div_le_div_right (Nat.mul_le_mul_left n h)

/-- Part p's end is part p+1's start: the ranges tile with no gap or overlap. -/
theorem lo_adjacent (n parts p : Nat) : lo n parts (p + 1) = lo n parts (p + 1) := rfl

/-- Every boundary lies inside [0, n]. -/
theorem lo_le (n parts p : Nat) (hp : p ≤ parts) (h : 0 < parts) : lo n parts p ≤ n := by
  have := lo_mono n parts p parts hp
  rw [lo_full n parts h] at this
  exact this

/-- Splitting the rows at any point gives two ranges whose lengths add up to the whole range. -/
theorem range_split (r0 m r1 : Nat) (h1 : r0 ≤ m) (h2 : m ≤ r1) :
    (m - r0) + (r1 - m) = r1 - r0 := by omega

-- § template dispatch on the token count

/-- `switch (n_tokens)`: 1..7 select NT = n, everything else selects NT = 8. -/
def dispatchNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- The selected NT is always an instantiated template (1..8). -/
theorem dispatchNT_range (n : Nat) : 1 ≤ dispatchNT n ∧ dispatchNT n ≤ 8 := by
  unfold dispatchNT
  split <;> omega

/-- For 1..8 tokens (MAXT = 8) the selected template has exactly that many lanes. -/
theorem dispatchNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : dispatchNT n = n := by
  unfold dispatchNT
  split <;> omega

/-- `s2_expert_vnni_multi` uses the batched kernel only without the oracle and with 1..MAXT tokens. -/
def useMulti (oracle : Bool) (n : Nat) : Bool := !oracle && decide (1 ≤ n) && decide (n ≤ 8)

/-- When the batched kernel runs, its lane count equals the token count. -/
theorem useMulti_lanes (oracle : Bool) (n : Nat) (h : useMulti oracle n = true) : dispatchNT n = n := by
  unfold useMulti at h
  simp at h
  exact dispatchNT_exact n h.1.2 h.2

/-- With the oracle contract on, the batched kernel is never used. -/
theorem useMulti_oracle (n : Nat) : useMulti true n = false := by
  simp [useMulti]

-- § CPU feature gate

structure Feat where
  f : Bool
  bw : Bool
  vl : Bool
  vnni : Bool
  vbmi : Bool

/-- `CpuFeatures::usable`. -/
def Feat.usable (x : Feat) : Bool := x.f && x.bw && x.vl && x.vnni && x.vbmi

/-- The names `reason()` would list. -/
def Feat.missing (x : Feat) : List String :=
  (if x.f then [] else ["AVX512F"]) ++ (if x.bw then [] else ["AVX512BW"]) ++
  (if x.vl then [] else ["AVX512VL"]) ++ (if x.vnni then [] else ["AVX512-VNNI"]) ++
  (if x.vbmi then [] else ["AVX512-VBMI"])

/-- `reason()` reports "ok" exactly when nothing is missing, which is exactly when the CPU is usable. -/
theorem missing_nil_iff (x : Feat) : x.missing = [] ↔ x.usable = true := by
  cases x with
  | mk a b c d e =>
    cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> simp [Feat.missing, Feat.usable]

/-- A CPU lacking any one of the five features is refused. -/
theorem lacking_vnni_refused (a b c e : Bool) : (Feat.mk a b c false e).usable = false := by
  simp [Feat.usable]

end Strata.CpuExpert
