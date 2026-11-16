-- Baloon.SGemv: s-GEMV activation addressing (Q8_K and Q8_0 buffer offsets),
-- lane-iteration alignment, hoisting, and same-group proofs.
-- Independent implementation proving the same mathematical facts as Ref.SGemv.

namespace Baloon.SGemv

def q8kOff (i : Nat) : Nat := ((i / 256) * 292 + 4 + i % 256)
def q80Off (i : Nat) : Nat := ((i / 32) * 34 + 2 + i % 32)

theorem q8kOff_inj (i j : Nat) (h : q8kOff i = q8kOff j) : i = j := by
  unfold q8kOff at h; omega

theorem q80Off_inj (i j : Nat) (h : q80Off i = q80Off j) : i = j := by
  unfold q80Off at h; omega

theorem q8kOff_before_bsums (i : Nat) : q8kOff i + 32 < (i / 256 + 1) * 292 := by
  unfold q8kOff; omega

theorem q8kOff_in_buffer (i nb : Nat) (h : i < 256 * nb) : q8kOff i < 292 * nb := by
  unfold q8kOff; omega

theorem q80Off_in_buffer (i nb : Nat) (h : i < 32 * nb) : q80Off i < 34 * nb := by
  unfold q80Off; omega

theorem hoist_q8k (i k : Nat) (h16 : i % 16 = 0) (hk : k < 16) :
    (i + k) / 256 = i / 256 ∧ (i + k) % 256 = i % 256 + k ∧ i % 256 + 16 ≤ 256 := by { omega }

theorem hoist_q8_0 (i k : Nat) (h16 : i % 16 = 0) (hk : k < 16) :
    (i + k) / 32 = i / 32 ∧ (i + k) % 32 = i % 32 + k ∧ i % 32 + 16 ≤ 32 := by { omega }

theorem lane_iter_aligned (lane t : Nat) : (lane * 16 + 512 * t) % 16 = 0 := by { omega }

theorem lane_iter_inj (lane t lane' t' : Nat) (hl : lane < 32) (hl' : lane' < 32)
    (h : lane * 16 + 512 * t = lane' * 16 + 512 * t') : lane = lane' ∧ t = t' := by { omega }

theorem lane_payload_bound (i : Nat) (h16 : i % 16 = 0) : 4 + i % 256 + 15 < 4 + 256 := by { omega }

theorem tail_empty (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) (h : ¬ (i + 16 ≤ n)) : ¬ (i < n) := by { omega }
theorem same_group (m a b k : Nat) (hm : 0 < m) (hk : k < m) : (m * a + k) / (m * b) = a / b := by
  have h1 : (m * a + k) / m = a := by { simp [Nat.mul_add_div hm, Nat.div_eq_of_lt hk] }
  rw [← Nat.div_div_eq_div_mul, h1]

-- ── word-level code packing and launch grids ─────────────────────────────────

/-- S8: the 32-bit word of four codes yields the four bytes. -/
theorem s8_word (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256) :
    let v := b0 + 256 * b1 + 65536 * b2 + 16777216 * b3
    (v % 256 = b0) ∧ ((v / 256) % 256 = b1) ∧ ((v / 65536) % 256 = b2) ∧
    ((v / 16777216) % 256 = b3) := by
  intro v; simp only [v]; omega

/-- Four lane words of QW bytes cover exactly 16 codes. -/
theorem word_bytes : ∀ B ∈ [2, 4, 8], 4 * (4 * B / 8) = 16 * B / 8 := by { decide }

/-- Alignment of the wide code loads. -/
theorem code_load_alignment (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) :
    (i / 1) % 4 = 0 ∧ (i / 2) % 2 = 0 ∧ (n / 1) % 4 = 0 ∧ (n / 2) % 2 = 0 := by omega

/-- 128 threads per block: the grid covers all rows. -/
theorem grid_cover_128 (n : Nat) : n ≤ ((n + 127) / 128) * 128 := by { omega }

/-- 8 warps per block: the grid covers all rows. -/
theorem grid_cover_warps (n : Nat) : n ≤ ((n + 7) / 8) * 8 := by { omega }

end Baloon.SGemv
