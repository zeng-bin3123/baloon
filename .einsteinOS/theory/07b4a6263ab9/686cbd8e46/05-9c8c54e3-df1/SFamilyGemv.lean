/-!
# MODULE: s_family_gemv
SOURCES: src/kernels/cuda/s_gemv.cu
The S-family GEMV `y[o] = Σ_i (decode(code_oi)·scale_og + offset_og) · x_i` over packed 2/4/8-bit code planes,
with FP16, Q8_K or Q8_0 activations.  Kernels: thread-per-row naive (`s_gemv_kernel`, `s_gemv_q8k_kernel`),
block-split with quads (`s_gemv_split_kernel`), and warp-per-row with 16 elements per lane-iteration
(`s_gemv_q8_split_kernel<CB, Q8K>`).  This module proves the multi-code word reads use the plane's bit order,
the group index as a shift, the "one scale per quad/octet" and "one activation block per lane-iteration"
invariants, the lane coverage of the warp kernel, and records the host gates — including one gate that is
weaker than its kernel needs.

## Source semantics
* Plane convention: element `e` at byte `e / (8/CB)`, bits `(e % (8/CB))·CB`.  A 4-code word is `QW = CB/2`
  bytes read little-endian; code `k` is bits `[k·CB, (k+1)·CB)`.
* Group index `g = e >> group_shift`, `group_shift = min{s | 2^s ≥ group_elems}`; hosts refuse non-powers of two.
* Split kernel: thread `tid` takes quads at `i = 4·tid + 4·tpr·m`; one scale per quad, so `group_elems % 4 = 0`.
* Warp Q8 kernel: lane takes `i = 16·lane + 512·m` (QE = 16), ONE scale for 16 elements (needs
  `group_elems % 16 = 0`) and ONE activation block (`i % 256` and `i % 32` are multiples of 16).
* Hosts: `s_gemv` CB ∈ {2,4,8}; Q8_K and Q8_0 paths CB ∈ {4,8} (S2 uses Q8_0 via other kernels);
  Q8_K needs `n_in % 256 = 0`, `n_in % group = 0`; Q8_0-split needs `n_in % 32 = 0` and checks only
  `group % 4 = 0`.
* Weight decode `w = decode(code)·d + b` BEFORE the activation multiply (exact model: distributes; FP: rounds
  differently, which is why the order is fixed).

## Form template identifiers
`CODE_BITS` → `cb`; `PER_BYTE` → `pb`; packed accessor → `codeAt`; `group_shift` → `groupShift`;
`QE` → 4 (split) / 16 (warp); `q8k_form_ok` → `q8kFormOk`; Q8_0 split gate → `q8_0SplitGate`;
lane start → `laneStart`; `acc0..acc15` combine → `combine16`.
-/

namespace Strata.SFamilyGemv

-- § plane access

/-- Plane code of element `e` with `pb` codes of `cb` bits per byte. -/
def codeAt (byte : Nat → Nat) (pb cb e : Nat) : Nat := (byte (e / pb) >>> ((e % pb) * cb)) &&& (2 ^ cb - 1)

/-- Shift-and-mask is division-and-modulo by powers of two. -/
theorem extract_eq (x s m : Nat) : (x >>> s) &&& (2 ^ m - 1) = x / 2 ^ s % 2 ^ m := by
  rw [Nat.shiftRight_eq_div_pow, Nat.and_two_pow_sub_one_eq_mod]

/-- **S2 word read** (`QW = 1`): code `k` of byte `i/4` is plane element `i + k` (for `i = 4a`). -/
theorem s2_word (byte : Nat → Nat) (a k : Nat) (hk : k < 4) :
    (byte a >>> (k * 2)) &&& (2 ^ 2 - 1) = codeAt byte 4 2 (4 * a + k) := by
  unfold codeAt
  rw [show (4 * a + k) / 4 = a by omega, show (4 * a + k) % 4 = k by omega]

/-- **S4 word read** (`QW = 2`): the little-endian `uint16` at byte `2a` yields plane elements `4a … 4a+3`. -/
theorem s4_word (byte : Nat → Nat) (hb : ∀ j, byte j < 256) (a k : Nat) (hk : k < 4) :
    ((byte (2 * a) + 256 * byte (2 * a + 1)) >>> (k * 4)) &&& (2 ^ 4 - 1) = codeAt byte 2 4 (4 * a + k) := by
  unfold codeAt
  rw [extract_eq, extract_eq]
  have h0 := hb (2 * a)
  have h1 := hb (2 * a + 1)
  have hk' : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  rcases hk' with rfl | rfl | rfl | rfl
  · rw [show (4 * a + 0) / 2 = 2 * a by omega, show (4 * a + 0) % 2 = 0 by omega]; simp; omega
  · rw [show (4 * a + 1) / 2 = 2 * a by omega, show (4 * a + 1) % 2 = 1 by omega]; simp; omega
  · rw [show (4 * a + 2) / 2 = 2 * a + 1 by omega, show (4 * a + 2) % 2 = 0 by omega]; simp; omega
  · rw [show (4 * a + 3) / 2 = 2 * a + 1 by omega, show (4 * a + 3) % 2 = 1 by omega]; simp; omega

/-- **S8 word read** (`QW = 4`): the little-endian `uint32` at byte `4a` yields plane elements `4a … 4a+3`. -/
theorem s8_word (byte : Nat → Nat) (hb : ∀ j, byte j < 256) (a k : Nat) (hk : k < 4) :
    ((byte (4 * a) + 256 * byte (4 * a + 1) + 65536 * byte (4 * a + 2) + 16777216 * byte (4 * a + 3))
        >>> (k * 8)) &&& (2 ^ 8 - 1) = codeAt byte 1 8 (4 * a + k) := by
  unfold codeAt
  rw [extract_eq, extract_eq]
  have h0 := hb (4 * a)
  have h1 := hb (4 * a + 1)
  have h2 := hb (4 * a + 2)
  have h3 := hb (4 * a + 3)
  have hk' : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  rcases hk' with rfl | rfl | rfl | rfl <;> simp [Nat.mod_one, Nat.div_one] <;> omega

/-- The 16-element lane-iteration reads FOUR consecutive words: word `w < 4` starts at plane element
`i + 4w`, so code `k` of word `w` is element `i + 4w + k` (the bit order of the four single-word reads). -/
theorem octet_words (i w k : Nat) (hw : w < 4) (hk : k < 4) : i + 4 * w + k < i + 16 ∧ (4 * w + k) / 4 = w := by
  omega

/-- Word size: `QW = 4·CB/8` bytes for the three widths. -/
theorem word_bytes : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

-- § the group index

/-- `while ((1 << s) < group_elems) ++s`, with fuel. -/
def shiftLoop (g : Nat) : Nat → Nat → Nat
  | 0, s => s
  | fuel + 1, s => if 2 ^ s < g then shiftLoop g fuel (s + 1) else s

/-- `group_shift`. -/
def groupShift (g : Nat) : Nat := shiftLoop g 64 0

/-- For every power-of-two group size up to 1024 the loop returns its logarithm. -/
theorem groupShift_pow2 : ∀ k : Fin 11, groupShift (2 ^ k.val) = k.val := by decide

/-- Hence the shift computes the group index: `e >> group_shift = e / group_elems`. -/
theorem shift_is_div (e : Nat) (k : Fin 11) : e >>> groupShift (2 ^ k.val) = e / 2 ^ k.val := by
  rw [groupShift_pow2, Nat.shiftRight_eq_div_pow]

/-- A non-power-of-two group is detected: 48 rounds up to shift 6, and `1 << 6 ≠ 48` (the Q8_K-split refusal);
using the shift anyway would index by 64, a WRONG group rather than a slow one. -/
theorem groupShift_48 : groupShift 48 = 6 ∧ 2 ^ groupShift 48 ≠ 48 ∧ 100 / 48 ≠ 100 >>> groupShift 48 := by
  decide

-- § one scale per quad / octet

/-- `n / (c·h) = (n / c) / h`. -/
theorem div_mul_split (n c h : Nat) : n / (c * h) = n / c / h := (Nat.div_div_eq_div_mul n c h).symm

/-- **Quad invariant** (split kernel): with `i ≡ 0 (mod 4)` and `group % 4 = 0`, the four elements of a quad
share a group, so the single `s[g]` load is right for all four. -/
theorem quad_same_group (i k g : Nat) (hi : i % 4 = 0) (hg : g % 4 = 0) (hk : k < 4) : (i + k) / g = i / g := by
  obtain ⟨h, rfl⟩ : ∃ h, g = 4 * h := ⟨g / 4, by omega⟩
  rw [div_mul_split, div_mul_split, show (i + k) / 4 = i / 4 by omega]

/-- **Octet invariant** (warp Q8 kernels): with `i ≡ 0 (mod 16)` and `group % 16 = 0`, all sixteen elements of
a lane-iteration share a group. -/
theorem octet_same_group (i k g : Nat) (hi : i % 16 = 0) (hg : g % 16 = 0) (hk : k < 16) :
    (i + k) / g = i / g := by
  obtain ⟨h, rfl⟩ : ∃ h, g = 16 * h := ⟨g / 16, by omega⟩
  rw [div_mul_split, div_mul_split, show (i + k) / 16 = i / 16 by omega]

/-- Split-kernel quads start on multiples of 4. -/
theorem split_start_aligned (tid tpr m : Nat) : (4 * tid + 4 * tpr * m) % 4 = 0 := by
  rw [Nat.mul_assoc]; omega

/-- Warp-kernel lane-iterations start at `16·lane + 512·m`. -/
def laneStart (lane m : Nat) : Nat := 16 * lane + 512 * m

/-- Lane starts are multiples of 16. -/
theorem laneStart_aligned (lane m : Nat) : laneStart lane m % 16 = 0 := by unfold laneStart; omega

/-- `s_gemv_q8k_split`'s gate (group part): power of two and multiple of 16. -/
def q8kSplitGroupOk (g : Nat) : Bool := 0 < g && g &&& (g - 1) == 0 && g % 16 == 0

/-- `s_gemv_q8_0_split`'s gate (group part): power of two and multiple of FOUR. -/
def q8_0SplitGate (g : Nat) : Bool := 0 < g && g &&& (g - 1) == 0 && g % 4 == 0

/-- The Q8_K gate is sound for the warp kernel: any accepted group keeps a lane-iteration in one group. -/
theorem q8kSplit_gate_sound (g lane m k : Nat) (h : q8kSplitGroupOk g = true) (hk : k < 16) :
    (laneStart lane m + k) / g = laneStart lane m / g := by
  simp [q8kSplitGroupOk] at h
  exact octet_same_group _ k g (laneStart_aligned lane m) h.2 hk

/-- **The Q8_0-split gate is weaker than its kernel.**  It checks `group % 4` (the comment points to the Q8_K
path's checks), but the warp kernel consumes 16 elements per scale load.  A group of 8 passes the gate while
one lane-iteration spans two groups — the second half would be scaled by the first group's `s[g]`.  Every group
size the pack format defines today (16, 32, 64) is safe, so this is latent; the fix is `% 16`. -/
theorem q8_0Split_gate_gap :
    q8_0SplitGate 8 = true ∧ q8kSplitGroupOk 8 = false ∧ (laneStart 0 0 + 8) / 8 ≠ laneStart 0 0 / 8 := by
  decide

/-- Every group size of the format passes both gates. -/
theorem format_groups_pass : ∀ g ∈ [16, 32, 64], q8kSplitGroupOk g = true ∧ q8_0SplitGate g = true := by
  decide

-- § one activation block per lane-iteration

/-- **Q8_K hoist is exact**: the 16 elements of a lane-iteration lie in one 256-element `block_q8_K`. -/
theorem q8k_block_hoist (lane m k : Nat) (hk : k < 16) :
    (laneStart lane m + k) / 256 = laneStart lane m / 256 ∧ laneStart lane m % 256 + 16 ≤ 256 := by
  unfold laneStart; omega

/-- **Q8_0 hoist is exact**: the 16 elements lie in one 32-element `block_q8_0`. -/
theorem q8_0_block_hoist (lane m k : Nat) (hk : k < 16) :
    (laneStart lane m + k) / 32 = laneStart lane m / 32 ∧ laneStart lane m % 32 + 16 ≤ 32 := by
  unfold laneStart; omega

/-- The hoisted pointer `xq = qs(block) + i % blk` indexes element `i + k` at offset `(i % blk) + k`. -/
theorem hoisted_offset (lane m k : Nat) (hk : k < 16) :
    (laneStart lane m + k) % 256 = laneStart lane m % 256 + k ∧
    (laneStart lane m + k) % 32 = laneStart lane m % 32 + k := by
  unfold laneStart; omega

-- § lane coverage of the warp kernel

/-- **Every element has exactly one owner** `(lane, m, k)` with `lane < 32`, `k < 16`. -/
theorem warp_owner_exists (e : Nat) : ∃ lane m k, lane < 32 ∧ k < 16 ∧ laneStart lane m + k = e :=
  ⟨e % 512 / 16, e / 512, e % 16, by omega, by omega, by unfold laneStart; omega⟩

/-- Ownership is unique. -/
theorem warp_owner_unique (l1 m1 k1 l2 m2 k2 : Nat) (hl1 : l1 < 32) (hl2 : l2 < 32) (hk1 : k1 < 16)
    (hk2 : k2 < 16) (h : laneStart l1 m1 + k1 = laneStart l2 m2 + k2) : l1 = l2 ∧ m1 = m2 ∧ k1 = k2 := by
  unfold laneStart at h; omega

-- § exact arithmetic

/-- `rsum f n = f 0 + … + f (n-1)`. -/
def rsum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => rsum f n + f n

/-- Unfolding step. -/
theorem rsum_succ (f : Nat → Int) (n : Nat) : rsum f (n + 1) = rsum f n + f n := rfl

/-- Splitting a range. -/
theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n := by
  induction n with
  | zero => exact (Int.add_zero _).symm
  | succ n ih => rw [← Nat.add_assoc, rsum_succ, ih, rsum_succ]; omega

/-- **Naive kernel loop structure**: the per-group double loop sums the same terms as the flat row sum. -/
theorem group_loop_eq_flat (f : Nat → Int) (ge ng : Nat) :
    rsum (fun g => rsum (fun j => f (g * ge + j)) ge) ng = rsum f (ng * ge) := by
  induction ng with
  | zero => simp [rsum]
  | succ m ih => rw [Nat.succ_mul, rsum_split f (m * ge) ge, ← ih, rsum_succ]

/-- The weight decode with offset, `w = c·d + b`, then `w·x` (exact model). -/
def term (c d b x : Int) : Int := (c * d + b) * x

/-- In exact arithmetic the source's order equals the distributed form it rejects; the two differ only in
FP rounding (one multiply-add then one multiply vs two multiplies and an add), which only Q4_K exposed. -/
theorem term_distributes (c d b x : Int) : term c d b x = c * d * x + b * x := by
  unfold term; rw [Int.add_mul]

/-- The 16-accumulator combine of the warp kernel is the plain sum. -/
def combine16 (a : Fin 16 → Int) : Int :=
  (((a 0 + a 1) + (a 2 + a 3)) + ((a 4 + a 5) + (a 6 + a 7))) +
  (((a 8 + a 9) + (a 10 + a 11)) + ((a 12 + a 13) + (a 14 + a 15)))

/-- Exactly, the combine order is irrelevant. -/
theorem combine16_eq (a : Fin 16 → Int) :
    combine16 a = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 +
      a 14 + a 15 := by
  unfold combine16; omega

-- § host gates

/-- Code widths the FP16 entry points (`s_gemv`, `s_gemv_split`) launch. -/
def fp16Widths : List Nat := [2, 4, 8]

/-- Code widths with a Q8_K / Q8_0 warp contract. -/
def q8Widths : List Nat := [4, 8]

/-- S2 has no Q8_K contract: `s_gemv_q8k*` refuse 2-bit codes. -/
theorem s2_no_q8k : 2 ∈ fp16Widths ∧ 2 ∉ q8Widths := by decide

/-- `q8k_form_ok`. -/
def q8kFormOk (g nIn : Nat) : Bool := 0 < g && nIn % g == 0 && nIn % 256 == 0

/-- The Q8_0-split activation gate. -/
def q8_0ActOk (nIn : Nat) : Bool := nIn % 32 == 0

/-- `ffn_down_shexp` (`n_in = 640`) can only be served by the Q8_0 path: Q8_K is structurally impossible. -/
theorem shexp_needs_q8_0 : q8kFormOk 32 640 = false ∧ q8_0ActOk 640 = true := by decide

/-- `s_gemv_split`'s thread gate: power of two in `1 … 1024`. -/
def splitTprOk (t : Nat) : Bool := 1 ≤ t && t &&& (t - 1) == 0 && t ≤ 1024

/-- The gate admits exactly the eleven powers of two `1 … 1024` among `t ≤ 2048`. -/
theorem splitTpr_examples : splitTprOk 1 = true ∧ splitTprOk 1024 = true ∧ splitTprOk 2048 = false ∧
    splitTprOk 96 = false ∧ splitTprOk 0 = false := by decide

end Strata.SFamilyGemv
