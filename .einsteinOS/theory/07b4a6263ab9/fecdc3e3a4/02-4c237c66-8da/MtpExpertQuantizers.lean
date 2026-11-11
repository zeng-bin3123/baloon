/-!
# MODULE: mtp_expert_quantizers
SOURCES: tools/mtp_pack.py
Packs the MTP draft block into a GGUF: dense tensors stay BF16 (1-D norms F32), and the routed experts
(fused `gate_up_proj` [512,1280,2560], `down_proj` [512,2560,640]) are quantized along the input axis to
Q2_0 (MSE-searched scale), Q4_0 or Q8_0.

## Source semantics
* Q2_0 block: 64 values → fp16 `d` (2 B) + 16 B of 2-bit codes; code = clip(round(x/d), −1, 2) + 1 ∈ 0..3,
  packed four per byte as `c0 | c1<<2 | c2<<4 | c3<<6`; dequant = (code − 1)·d.
* Q2_0 scale search: 17 candidates `d = amax·f`, f ∈ linspace(0.5, 1.0); a candidate replaces the best only if
  its error is STRICTLY smaller. f = 1.0 is the ggml reference scale, so the searched error is never worse
  than the reference (for the unrounded scale; the final codes use the fp16-rounded scale).
* Q4_0: 32 values → fp16 d + 16 B; `q = min(15, trunc(x/d + 8.5))`, low nibble = q[i], high = q[i+16].
* Q8_0: 32 values → fp16 d + 32 int8.
* The tensor's raw shape replaces the inner dim n by `n/block·block_bytes`, which requires `n % block = 0`.

## Form template identifiers
q2_0 packing → pack2bit / unpack2bit, clip+1 → q2Code, scale search → pickScale, q4_0 nibbles →
packNibbles, block sizes → blockElems / blockBytes, expert tensor bytes → expertBytes, QUANT table → Kind
-/

namespace Strata.MtpPack

-- § Q2_0 codes

/-- `clip(q, -1, 2) + 1`: the stored code. -/
def q2Code (q : Int) : Int := max (-1) (min q 2) + 1

/-- Codes always fit in two bits. -/
theorem q2Code_range (q : Int) : 0 ≤ q2Code q ∧ q2Code q ≤ 3 := by
  unfold q2Code; omega

/-- Dequantized levels are exactly the grid {−1, 0, 1, 2}·d (the +2 level is reachable). -/
theorem q2_levels (q : Int) : q2Code q - 1 = -1 ∨ q2Code q - 1 = 0 ∨ q2Code q - 1 = 1 ∨ q2Code q - 1 = 2 := by
  unfold q2Code; omega

/-- Values already on the grid are reproduced exactly. -/
theorem q2Code_on_grid (q : Int) (h1 : -1 ≤ q) (h2 : q ≤ 2) : q2Code q - 1 = q := by
  unfold q2Code; omega

/-- `c0 | c1<<2 | c2<<4 | c3<<6` (disjoint bit fields, so `|` is `+`). -/
def pack2bit (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- `(byte >> 2s) & 3`. -/
def unpack2bit (b s : Nat) : Nat := (b / 4 ^ s) % 4

/-- Four codes fit in one byte. -/
theorem pack2bit_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack2bit c0 c1 c2 c3 < 256 := by
  unfold pack2bit; omega

/-- The dequantizer's shifts recover each packed code: pack and unpack are inverse. -/
theorem unpack_pack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    unpack2bit (pack2bit c0 c1 c2 c3) 0 = c0 ∧ unpack2bit (pack2bit c0 c1 c2 c3) 1 = c1 ∧
    unpack2bit (pack2bit c0 c1 c2 c3) 2 = c2 ∧ unpack2bit (pack2bit c0 c1 c2 c3) 3 = c3 := by
  unfold unpack2bit pack2bit
  simp only [Nat.pow_zero, Nat.pow_succ, Nat.one_mul]
  omega

-- § Q2_0 scale search

/-- One step of the search: keep `(d, err)` unless the candidate's error is strictly smaller. -/
def pickScale (best cand : Nat × Nat) : Nat × Nat := if cand.2 < best.2 then cand else best

/-- The search result's error is no larger than the starting best (∞ in the source). -/
theorem search_le_init : ∀ (cands : List (Nat × Nat)) (init : Nat × Nat),
    (cands.foldl pickScale init).2 ≤ init.2
  | [], _ => Nat.le_refl _
  | c :: cs, init => by
    simp only [List.foldl]
    have h := search_le_init cs (pickScale init c)
    have hp : (pickScale init c).2 ≤ init.2 := by unfold pickScale; split <;> omega
    exact Nat.le_trans h hp

/-- The search result's error is no larger than ANY candidate's error. -/
theorem search_le_all : ∀ (cands : List (Nat × Nat)) (init : Nat × Nat),
    ∀ c ∈ cands, (cands.foldl pickScale init).2 ≤ c.2
  | [], _, _, h => by simp at h
  | c :: cs, init, x, h => by
    simp only [List.foldl]
    simp only [List.mem_cons] at h
    cases h with
    | inl he =>
      subst he
      have h1 := search_le_init cs (pickScale init x)
      have hp : (pickScale init x).2 ≤ x.2 := by unfold pickScale; split <;> omega
      exact Nat.le_trans h1 hp
    | inr hm => exact search_le_all cs (pickScale init c) x hm

/-- Hence: since f = 1.0 (the ggml reference scale) is a candidate, the searched scale's grid error is
never worse than the reference's. -/
theorem search_beats_reference (cands : List (Nat × Nat)) (init ref : Nat × Nat) (h : ref ∈ cands) :
    (cands.foldl pickScale init).2 ≤ ref.2 :=
  search_le_all cands init ref h

-- § Q4_0 nibbles

/-- `min(15, t)`: the clamp that catches `x/d = 8` (x equal to −max). -/
def q4Code (t : Nat) : Nat := min 15 t

/-- Q4 codes fit in a nibble. -/
theorem q4Code_lt (t : Nat) : q4Code t < 16 := by unfold q4Code; omega

/-- `q[i] | q[i+16] << 4`. -/
def packNibbles (lo hi : Nat) : Nat := lo + 16 * hi

/-- `qs & 15` and `qs >> 4` recover both halves. -/
theorem unpack_nibbles (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    packNibbles lo hi % 16 = lo ∧ packNibbles lo hi / 16 = hi ∧ packNibbles lo hi < 256 := by
  unfold packNibbles; omega

-- § block geometry and tensor sizes

/-- The three expert formats. -/
inductive Kind | q2_0 | q4_0 | q8_0

/-- Elements per block. -/
def blockElems : Kind → Nat
  | .q2_0 => 64 | .q4_0 => 32 | .q8_0 => 32

/-- Bytes per block: fp16 scale + payload. -/
def blockBytes : Kind → Nat
  | .q2_0 => 18 | .q4_0 => 18 | .q8_0 => 34

/-- The payload is exactly the code bits: 2-bit, 4-bit, 8-bit. -/
theorem block_payloads :
    blockBytes .q2_0 = 2 + blockElems .q2_0 * 2 / 8 ∧ blockBytes .q4_0 = 2 + blockElems .q4_0 * 4 / 8 ∧
    blockBytes .q8_0 = 2 + blockElems .q8_0 := by decide

/-- Bytes of a [rows, n] slab once quantized: `rows · (n / block · block_bytes)`. -/
def rowBytes (k : Kind) (rows n : Nat) : Nat := rows * (n / blockElems k * blockBytes k)

/-- Both fused expert tensors (512 experts) quantized with format `k`. -/
def expertBytes (k : Kind) : Nat := 512 * rowBytes k 1280 2560 + 512 * rowBytes k 2560 640

/-- Both inner dims are block multiples for every format, so the tool never raises. -/
theorem inner_dims_divisible : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

/-- The docstring's sizes: Q2_0 ≈ 0.71 GB, Q4_0 ≈ 1.42 GB, Q8_0 ≈ 2.67 GB. -/
theorem expert_sizes :
    expertBytes .q2_0 = 707788800 ∧ expertBytes .q4_0 = 1415577600 ∧ expertBytes .q8_0 = 2673868800 := by
  decide

/-- Q2_0 is exactly half of Q4_0 here (same 18-byte block, twice the elements). -/
theorem q2_half_of_q4 : 2 * expertBytes .q2_0 = expertBytes .q4_0 := by decide

/-- 2.25 bits per weight for Q2_0: 18 bytes · 8 = 64 elements · 9/4. -/
theorem q2_bits_per_weight : blockBytes .q2_0 * 8 * 4 = blockElems .q2_0 * 9 := by decide

end Strata.MtpPack
