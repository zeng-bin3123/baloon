/-!
# MODULE: layer_buffer_arena
Carving of the per-layer scratch arenas (GDN buffers, MoE buffers, QSA/block buffers) out of one allocation:
each region is rounded up to 16 bytes and starts where the previous one ended.

## Source semantics
- `*_buffers_bytes` sums the 16-byte-aligned sizes of the parts; `*_buffers_init` hands out the same
  regions in the same order, so the sizes agree by construction.
- Every region start is a multiple of 16 when the base is.
- The GDN state region is S*Hv*S*4 bytes and the Q8_K image of n_embd=2560 is 10 blocks of 292 bytes.

## Form template identifiers
parts[] -> parts ; (v + 15) & ~15 -> align16 ; ptr[i] -> starts ; gdn_buffers_bytes -> arenaTotal
-/

namespace Strata.LayerArena

-- § alignment

/-- Round up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment yields a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Alignment never shrinks. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

-- § region starts

/-- Start offsets of consecutive regions beginning at `acc`. -/
def starts (acc : Nat) : List Nat → List Nat
  | [] => []
  | p :: ps => acc :: starts (acc + align16 p) ps

/-- Total bytes: sum of aligned parts. -/
def arenaTotal (parts : List Nat) : Nat := (parts.map align16).sum

/-- One start per part. -/
theorem starts_length (acc : Nat) (parts : List Nat) : (starts acc parts).length = parts.length := by
  induction parts generalizing acc with
  | nil => simp [starts]
  | cons p ps ih => simp [starts, ih]

/-- Every region start is 16-aligned when the base is. -/
theorem starts_aligned (parts : List Nat) (acc : Nat) (h : acc % 16 = 0) :
    ∀ s ∈ starts acc parts, s % 16 = 0 := by
  induction parts generalizing acc with
  | nil => simp [starts]
  | cons p ps ih =>
    intro s hs
    simp only [starts, List.mem_cons] at hs
    rcases hs with hs | hs
    · rw [hs]; exact h
    · exact ih (acc + align16 p) (by have := align16_mod p; omega) s hs

/-- Every region start is at least the base. -/
theorem starts_ge (parts : List Nat) (acc : Nat) : ∀ s ∈ starts acc parts, acc ≤ s := by
  induction parts generalizing acc with
  | nil => simp [starts]
  | cons p ps ih =>
    intro s hs
    simp only [starts, List.mem_cons] at hs
    rcases hs with hs | hs
    · omega
    · have := ih (acc + align16 p) s hs
      omega

/-- The first region starts at the base. -/
theorem starts_head (acc p : Nat) (ps : List Nat) : (starts acc (p :: ps)).head? = some acc := by
  simp [starts]

/-- The arena covers the raw sizes of its parts. -/
theorem arenaTotal_ge (parts : List Nat) : parts.sum ≤ arenaTotal parts := by
  unfold arenaTotal
  induction parts with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.sum_cons]
    have := align16_ge a
    omega

/-- Appending parts adds totals. -/
theorem arenaTotal_append (a b : List Nat) : arenaTotal (a ++ b) = arenaTotal a + arenaTotal b := by
  simp [arenaTotal, List.map_append, List.sum_append]

-- § the artifact's concrete sizes

/-- The GDN recurrent state: 128*48*128*4 bytes, already 16-aligned. -/
theorem gdn_state_region : align16 (128 * 48 * 128 * 4) = 3145728 := by decide

/-- Q8_K image of n_embd = 2560: 10 blocks of 292 bytes, aligned to 2928. -/
theorem q8k_embd_region : align16 (2560 / 256 * 292) = 2928 := by decide

/-- Q8_0 image of n_embd = 2560: 80 blocks of 34 bytes, already aligned. -/
theorem q8_0_embd_region : align16 (2560 / 32 * 34) = 2720 := by decide

end Strata.LayerArena
