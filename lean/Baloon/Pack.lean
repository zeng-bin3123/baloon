-- Baloon.Pack: 2-bit and 4-bit packing, nibble splitting, integer code bias and
-- bit-mask index proofs for Q2_0/Q4_0/Q5_0/Q3_K/Q5_K/Q6_K/Q8_0.
-- Independent implementation proving the same mathematical facts as Ref.Pack.

namespace Baloon.Pack

def pack2 (c0 c1 c2 c3 : Nat) : Nat := (c0 ||| (c1 <<< 2) ||| (c2 <<< 4) ||| (c3 <<< 6))
def pack4 (a b : Nat) : Nat := (a ||| (b <<< 4))
def field (byte k bits : Nat) : Nat := ((byte >>> (k * bits)) &&& ((1 <<< bits) - 1))

theorem pack2_roundtrip : ∀ c0 : Nat, c0 < 4 → ∀ c1 : Nat, c1 < 4 → ∀ c2 : Nat, c2 < 4 → ∀ c3 : Nat, c3 < 4 →
    field (pack2 c0 c1 c2 c3) 0 2 = c0 ∧ field (pack2 c0 c1 c2 c3) 1 2 = c1 ∧
    field (pack2 c0 c1 c2 c3) 2 2 = c2 ∧ field (pack2 c0 c1 c2 c3) 3 2 = c3 := by { decide }

theorem pack2_lt : ∀ c0 : Nat, c0 < 4 → ∀ c1 : Nat, c1 < 4 → ∀ c2 : Nat, c2 < 4 → ∀ c3 : Nat, c3 < 4 →
    pack2 c0 c1 c2 c3 < 256 := by { decide }

theorem pack4_roundtrip : ∀ a : Nat, a < 16 → ∀ b : Nat, b < 16 →
    field (pack4 a b) 0 4 = a ∧ field (pack4 a b) 1 4 = b ∧ pack4 a b < 256 := by { decide }

set_option maxRecDepth 100000 in
theorem nibble_split : ∀ b : Nat, b < 256 →
    (b >>> 4) * 16 + (b &&& 15) = b ∧ (b &&& 15) < 16 ∧ (b >>> 4) < 16 := by { decide }

def biased (code : Nat) (bias : Int) : Int := ((code : Int) + bias)

theorem q2_codebook (c : Nat) (h : c < 4) : -1 ≤ biased c (-1) ∧ biased c (-1) ≤ 2 := by { unfold biased; omega }
theorem q4_codebook (c : Nat) (h : c < 16) : -8 ≤ biased c (-8) ∧ biased c (-8) ≤ 7 := by { unfold biased; omega }
theorem q5_codebook (c : Nat) (h : c < 32) : -16 ≤ biased c (-16) ∧ biased c (-16) ≤ 15 := by { unfold biased; omega }
theorem q3k_codebook (c : Nat) (h : c < 8) : -4 ≤ biased c (-4) ∧ biased c (-4) ≤ 3 := by { unfold biased; omega }
theorem q6k_codebook (c : Nat) (h : c < 64) : -32 ≤ biased c (-32) ∧ biased c (-32) ≤ 31 := by { unfold biased; omega }
theorem q8_codebook (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) :
    ∃ c : Nat, c < 256 ∧ biased c (-128) = q := by
  refine ⟨(q + 128).toNat, by omega, ?_⟩; unfold biased; omega

theorem q3k_mask_bits : ∀ half : Nat, half < 2 → ∀ j : Nat, j < 4 → 4 * half + j < 8 := by { decide }

end Baloon.Pack
