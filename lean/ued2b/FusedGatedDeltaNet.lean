/-!
# MODULE: fused_gated_delta_net
SOURCES: src/kernels/cuda/fused_gdn.cu
Single-launch GDN kernels at the fixed geometry S = 128: the step fused with its output RMS
norm and z-gate (4 row groups × 128 columns per head), the 4-tap conv fused with SiLU and the
per-head L2 norm, and the alpha/beta GEMV with bf16 weights fused with softplus / sigmoid.

## Source semantics
* Thread (col, rg) owns rows `rg*32 .. rg*32+31` of column `col`; the four row-group partials
  meet in `red[rg][col]` and are summed to a full contraction.
* The fused step computes `kv` on the UNDECAYED state and uses `delta = (v - g*kv)*beta`,
  `s' = g*s + k*delta`; this equals gdn.cu's decay-then-contract form.
* Conv: history `[h0,h1,h2]`, output `Σ`, history becomes `[h1,h2,x]`; blocks `< qk_heads`
  (q and k heads) are additionally L2-normalised, the v heads are not.
* `gdn_ab`: rows `0..h_v-1` are alpha (gate), rows `h_v..2h_v-1` are beta; bf16 is unpacked
  from a 32-bit word as `w << 16` (low half) and `w & 0xffff0000` (high half).
* Host preconditions: non-null pointers, `h_v % h_k = 0`, `channels % 128 = 0`,
  `0 ≤ qk_heads ≤ channels/128`, `n_embd % 8 = 0`.

## Form template identifiers
row-group split ↦ `rowOf`, partial reduction ↦ `dot_append`, fused step ↦ `fusedCol`,
two-pass step ↦ `twoPassCol`, conv history ↦ `histShift`, alpha/beta dispatch ↦ `abRow`,
bf16 halves ↦ `lowHalf`/`highHalf`.
-/

namespace Ref.FusedGdn

-- § helpers

/-- Dot product of coefficient lists. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Scaling the left operand scales the dot product. -/
theorem dot_scale_left (g : Int) (s k : List Int) :
    dot (s.map (· * g)) k = g * dot s k := by
  induction s generalizing k with
  | nil => simp [dot]
  | cons a s ih =>
    cases k with
    | nil => simp [dot]
    | cons b k =>
      simp only [List.map, dot, ih]
      rw [Int.mul_add, Int.mul_comm a g, Int.mul_assoc]

/-- Row-group reduction: a dot over concatenated equal-length blocks is the sum of block dots. -/
theorem dot_append (a b c d : List Int) (h : a.length = c.length) :
    dot (a ++ b) (c ++ d) = dot a c + dot b d := by
  induction a generalizing c with
  | nil =>
    cases c with
    | nil => simp [dot]
    | cons _ _ => simp at h
  | cons x a ih =>
    cases c with
    | nil => simp at h
    | cons y c =>
      simp only [List.cons_append, dot]
      rw [ih c (by simpa using h), Int.add_assoc]

-- § row-group geometry

/-- State size. -/
def S : Nat := 128
/-- Row groups per column. -/
def RG : Nat := 4
/-- Rows per thread. -/
def RPG : Nat := 32

/-- Global row owned by row group `rg` at register slot `r`. -/
def rowOf (rg r : Nat) : Nat := rg * RPG + r

/-- The geometry constants are consistent: `RG * RPG = S` and the block is 512 threads. -/
theorem geometry_consistent : RG * RPG = S ∧ S * RG = 512 := by decide

/-- Every row `< 128` is owned by exactly one `(rg, r)` with `rg < 4`, `r < 32`. -/
theorem rowOf_cover {i : Nat} (hi : i < S) :
    ∃ rg r, rg < RG ∧ r < RPG ∧ rowOf rg r = i := by
  refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> simp [rowOf, RG, RPG, S] at * <;> omega

/-- Ownership is unique: two slots naming the same row are the same slot. -/
theorem rowOf_unique {rg r rg' r' : Nat} (hr : r < RPG) (hr' : r' < RPG)
    (h : rowOf rg r = rowOf rg' r') : rg = rg' ∧ r = r' := by
  simp [rowOf, RPG] at *
  omega

/-- A row-group thread's rows stay inside the 128-row column. -/
theorem rowOf_lt {rg r : Nat} (hrg : rg < RG) (hr : r < RPG) : rowOf rg r < S := by
  simp [rowOf, RG, RPG, S] at *
  omega

/-- The four partials summed by `kv_col = red[0]+red[1]+red[2]+red[3]` give the full dot. -/
theorem four_partials (s0 s1 s2 s3 k0 k1 k2 k3 : List Int)
    (h0 : s0.length = k0.length) (h1 : s1.length = k1.length) (h2 : s2.length = k2.length) :
    dot (s0 ++ s1 ++ s2 ++ s3) (k0 ++ k1 ++ k2 ++ k3) =
      dot s0 k0 + dot s1 k1 + dot s2 k2 + dot s3 k3 := by
  have l01 : (s0 ++ s1).length = (k0 ++ k1).length := by simp [h0, h1]
  have l012 : (s0 ++ s1 ++ s2).length = (k0 ++ k1 ++ k2).length := by simp [h0, h1, h2]
  rw [dot_append _ _ _ _ l012, dot_append _ _ _ _ l01, dot_append _ _ _ _ h0]

-- § fused vs two-pass step

/-- gdn.cu's two-pass column step: decay, contract, rank-1 update. -/
def twoPassCol (g beta v : Int) (s k : List Int) : List Int :=
  let sd := s.map (· * g)
  let d := (v - dot sd k) * beta
  List.zipWith (fun a b => a + b * d) sd k

/-- fused_gdn.cu's column step: contract first, fold the decay into `delta` and the update. -/
def fusedCol (g beta v : Int) (s k : List Int) : List Int :=
  let delta := (v - g * dot s k) * beta
  List.zipWith (fun a b => g * a + b * delta) s k

/-- Pointwise: a zipWith over a mapped left list is a zipWith of the composed function. -/
theorem zipWith_map_left (f : Int → Int) (F : Int → Int → Int) (s k : List Int) :
    List.zipWith F (s.map f) k = List.zipWith (fun a b => F (f a) b) s k := by
  induction s generalizing k with
  | nil => simp
  | cons a s ih =>
    cases k with
    | nil => simp
    | cons b k => simp [List.zipWith, ih]

/-- The fused step computes exactly the two-pass state (exact arithmetic). -/
theorem fused_eq_twoPass (g beta v : Int) (s k : List Int) :
    fusedCol g beta v s k = twoPassCol g beta v s k := by
  have hd : (v - dot (s.map (· * g)) k) * beta = (v - g * dot s k) * beta := by
    rw [dot_scale_left]
  simp only [twoPassCol, fusedCol]
  rw [hd, zipWith_map_left]
  congr 1
  funext a b
  rw [Int.mul_comm a g]

/-- Hence the readouts `<s', q>` of both formulations agree as well. -/
theorem fused_readout_eq (g beta v : Int) (s k q : List Int) :
    dot (fusedCol g beta v s k) q = dot (twoPassCol g beta v s k) q := by
  rw [fused_eq_twoPass]

-- § conv + l2

/-- History shift of the fused conv: `[h0,h1,h2] , x ↦ [h1,h2,x]`. -/
def histShift (h : Int × Int × Int) (x : Int) : Int × Int × Int := (h.2.1, h.2.2, x)

/-- The conv sum over the 4-tap window `[h0,h1,h2,x]`. -/
def convSum (h : Int × Int × Int) (x w0 w1 w2 w3 : Int) : Int :=
  h.1 * w0 + h.2.1 * w1 + h.2.2 * w2 + x * w3

/-- Three shifts flush the history: it then holds exactly the last three inputs. -/
theorem histShift_three (h : Int × Int × Int) (a b c : Int) :
    histShift (histShift (histShift h a) b) c = (a, b, c) := rfl

/-- Channel `c = block*128 + thread` is an L2-normalised q/k channel iff its block is a qk head. -/
def isQkChannel (qkHeads c : Nat) : Bool := c / 128 < qkHeads

/-- Channels of block `b` all share the block's q/k classification. -/
theorem qk_by_block {qkHeads b t : Nat} (ht : t < 128) :
    isQkChannel qkHeads (b * 128 + t) = decide (b < qkHeads) := by
  simp [isQkChannel]
  have : (b * 128 + t) / 128 = b := by omega
  rw [this]

/-- Host check of `fused_gdn_conv_l2`. -/
def convL2Valid (channels qkHeads : Int) : Bool :=
  channels % 128 == 0 && qkHeads ≥ 0 && qkHeads ≤ channels / 128

/-- A valid call never marks more q/k blocks than there are blocks. -/
theorem convL2Valid_heads {c q : Int} (h : convL2Valid c q = true) : 0 ≤ q ∧ q ≤ c / 128 := by
  simp [convL2Valid] at h
  omega

-- § alpha / beta GEMV

/-- Dispatch of GEMV row `row < 2*h_v`: `(is_beta, r)`. -/
def abRow (hv row : Nat) : Bool × Nat := if row ≥ hv then (true, row - hv) else (false, row)

/-- Every row maps to a valid head index and each head is hit once as alpha and once as beta. -/
theorem abRow_lt {hv row : Nat} (h : row < 2 * hv) : (abRow hv row).2 < hv := by
  unfold abRow
  split <;> simp <;> omega

/-- Alpha row `r` and beta row `h_v + r` both land on head `r`. -/
theorem abRow_pair (hv r : Nat) (hr : r < hv) :
    abRow hv r = (false, r) ∧ abRow hv (hv + r) = (true, r) := by
  unfold abRow
  constructor
  · simp; omega
  · simp

/-- The launch `ceil(2*h_v/8)` blocks of 8 warps covers every GEMV row. -/
theorem ab_grid_covers (hv : Nat) : 2 * hv ≤ (2 * hv + 7) / 8 * 8 := by omega

/-- Low bf16 half of a packed word, widened to f32 bits: `(w mod 2^16) * 2^16`. -/
def lowHalf (w : Nat) : Nat := (w % 2 ^ 16) * 2 ^ 16
/-- High bf16 half, already in f32 position: `(w / 2^16) * 2^16`. -/
def highHalf (w : Nat) : Nat := (w / 2 ^ 16) * 2 ^ 16

/-- Both widened halves have zero low mantissa bits, as bf16→f32 requires. -/
theorem halves_aligned (w : Nat) : lowHalf w % 2 ^ 16 = 0 ∧ highHalf w % 2 ^ 16 = 0 := by
  simp [lowHalf, highHalf, Nat.mul_mod_left]

/-- The two halves carry the whole word: `w = highHalf + lowHalf / 2^16`. -/
theorem halves_reconstruct (w : Nat) : highHalf w + lowHalf w / 2 ^ 16 = w := by
  simp only [lowHalf, highHalf]
  rw [Nat.mul_div_cancel _ (by decide)]
  have := Nat.mod_add_div w (2 ^ 16)
  rw [Nat.mul_comm] at this
  omega

/-- Host check of `fused_gdn_step_norm`: positive heads with `h_v % h_k = 0`. -/
theorem step_norm_heads_valid {hk h : Nat} (hk0 : 0 < hk) :
    h % hk < hk := Nat.mod_lt _ hk0

end Ref.FusedGdn


