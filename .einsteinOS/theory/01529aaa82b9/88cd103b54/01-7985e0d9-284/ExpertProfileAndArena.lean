/-!
# MODULE: expert_profile_and_arena
Validation of the STRP expert-profile file and the 256-byte-aligned slot arena of `ExpertCache::open_sized`.

## Source semantics
- A profile is accepted only when its header geometry equals the model's, the ranked count does not exceed the
  requested slot count, and every ranked pair is in range.
- `open_sized` rounds every slot up to a multiple of 256 bytes and lays them out back to back, so every blob
  starts 256-aligned.
- The allocation is refused when the requested bytes exceed the free VRAM.

## Form template identifiers
- `headerOk`   <- the geometry/`n_ranked > want` checks of `read_expert_profile`
- `pairOk`, `profileOk` <- the per-pair range check on `(layer, expert)`
- `align256`   <- `(slot_bytes[i] + 255) / 256 * 256`
- `arenaTotal` <- `off.back()` (sum of aligned slot sizes)
- `allocOk`    <- `free_b >= want` test of `ExpertCache::open`
-/

namespace Strata.ExpertProfile

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Sum of aligned slot sizes (the final offset). -/
def arenaTotal : List Nat → Nat
  | [] => 0
  | b :: t => align256 b + arenaTotal t

/-- A ranked pair is valid when both coordinates are in range. -/
def pairOk (nl ne l e : Nat) : Prop := l < nl ∧ e < ne

/-- Every ranked pair of a profile is valid. -/
def profileOk (nl ne : Nat) (r : List (Nat × Nat)) : Prop := ∀ p ∈ r, pairOk nl ne p.1 p.2

/-- Header acceptance: same geometry and not more ranked pairs than slots. -/
def headerOk (nl ne gnl gne want nr : Nat) : Prop := nl = gnl ∧ ne = gne ∧ nr ≤ want

/-- Allocation gate against the card's free memory. -/
def allocOk (free want : Nat) : Bool := decide (want ≤ free)

/-- Alignment never shrinks a slot. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by
  unfold align256; omega

/-- Aligned sizes are multiples of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256; omega

/-- Alignment wastes fewer than 256 bytes. -/
theorem align256_lt (x : Nat) : align256 x < x + 256 := by
  unfold align256; omega

/-- Every slot offset (and the total) is 256-aligned. -/
theorem arenaTotal_mod (l : List Nat) : arenaTotal l % 256 = 0 := by
  induction l with
  | nil => simp [arenaTotal]
  | cons b t ih =>
    have h := align256_mod b
    simp only [arenaTotal]
    omega

/-- The arena is at least as large as the raw slot bytes. -/
theorem arenaTotal_ge (l : List Nat) : l.sum ≤ arenaTotal l := by
  induction l with
  | nil => simp [arenaTotal]
  | cons b t ih =>
    have h := align256_ge b
    simp only [arenaTotal, List.sum_cons]
    omega

/-- A header that passes cannot claim more ranked pairs than slots. -/
theorem headerOk_ranked_le (nl ne gnl gne want nr : Nat) (h : headerOk nl ne gnl gne want nr)
    (ranked : List (Nat × Nat)) (hl : ranked.length = nr) : ranked.length ≤ want := by
  rw [hl]
  exact h.2.2

/-- A header that passes has the model's exact geometry (a different artifact's profile is refused). -/
theorem headerOk_geometry (nl ne gnl gne want nr : Nat) (h : headerOk nl ne gnl gne want nr) :
    nl = gnl ∧ ne = gne := ⟨h.1, h.2.1⟩

/-- Consing a valid pair onto a valid profile is valid, and conversely. -/
theorem profileOk_cons (nl ne : Nat) (p : Nat × Nat) (r : List (Nat × Nat)) :
    profileOk nl ne (p :: r) ↔ pairOk nl ne p.1 p.2 ∧ profileOk nl ne r := by
  simp [profileOk]

/-- The raw ranked payload is `4 * n_ranked` bytes (two `uint16` per pair). -/
theorem ranked_payload_bytes (nr : Nat) : 2 * (nr * 2) = 4 * nr := by
  omega

/-- If the allocation gate passes, the request fits in the free VRAM. -/
theorem allocOk_fits (free want : Nat) (h : allocOk free want = true) : want ≤ free := by
  simpa [allocOk] using h

end Strata.ExpertProfile
