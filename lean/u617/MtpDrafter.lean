/-!
# MODULE: mtp_drafter
SOURCES: src/core/mtp.cpp
The multi-token-prediction draft layer: bump-allocated device arena, windowed attention capacity, the
step/position row staging used by captured prefill / round / chain-step graphs, and the draft chain length.

## Source semantics
* `Bump::take` pads every region to 256 bytes; the counting pass and the real pass use the same sequence.
* Step/pos staging has `2*max_t` rows: catch-up rows `[0,T)`, the full-layer row `2*max_t-1`, chain rows
  `max_t + j - 1` for `j = 1..max_t-2`.
* `record_forward(T, s)`: `s >= 0` is the full layer on row `s`; `s < 0` is K/V-only on rows from `-1 - s`.
* The draft chain emits at most `max_t - 1` drafts.

## Form template identifiers
align256 ↦ `(n*sizeof(T)+255)&~255`; capRound ↦ `cap_`; ra ↦ `2*max_t_-1`; chainRow ↦ `max_t_+j-1`.
-/
namespace Ref.MtpDrafter

-- § bump allocator

/-- 256-byte padding of one region. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Padding never shrinks. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- Padded regions are 256-multiples. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- Padding wastes fewer than 256 bytes. -/
theorem align256_lt (n : Nat) : align256 n < n + 256 := by unfold align256; omega

/-- Total after carving a list of byte sizes. -/
def bump : List Nat → Nat → Nat
  | [], s => s
  | p :: ps, s => bump ps (s + align256 p)

/-- Starting aligned, every region base is 256-aligned (device pointers stay aligned). -/
theorem bump_aligned : ∀ (parts : List Nat) (s : Nat), s % 256 = 0 → bump parts s % 256 = 0
  | [], s, h => by simpa [bump] using h
  | p :: ps, s, h => by
      rw [bump]
      apply bump_aligned ps
      have := align256_mod p
      omega

/-- Bumping is monotone. -/
theorem bump_ge : ∀ (parts : List Nat) (s : Nat), s ≤ bump parts s
  | [], s => Nat.le_refl s
  | p :: ps, s => by
      rw [bump]
      have h := bump_ge ps (s + align256 p)
      omega

-- § windowed capacity

/-- `cap_ = ((window or max_cells) + 63) / 64 * 64`. -/
def capRound (w : Nat) : Nat := (w + 63) / 64 * 64

/-- The capacity covers the window. -/
theorem capRound_ge (w : Nat) : w ≤ capRound w := by unfold capRound; omega

/-- The capacity is a multiple of 64. -/
theorem capRound_mod (w : Nat) : capRound w % 64 = 0 := by unfold capRound; omega

/-- The capacity wastes fewer than 64 cells. -/
theorem capRound_lt (w : Nat) : capRound w < w + 64 := by unfold capRound; omega

-- § step rows

/-- The full-layer row of a round. -/
def ra (maxT : Nat) : Nat := 2 * maxT - 1

/-- The chain-step row for draft `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- The full-layer row is inside the `2*max_t` rows allocated. -/
theorem ra_lt (maxT : Nat) (h : 1 ≤ maxT) : ra maxT < 2 * maxT := by unfold ra; omega

/-- Chain rows lie strictly between the catch-up rows and the full-layer row. -/
theorem chainRow_between (maxT j : Nat) (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) :
    maxT ≤ chainRow maxT j ∧ chainRow maxT j < ra maxT := by
  unfold chainRow ra; omega

/-- Catch-up rows `[0,T)` never collide with a chain row. -/
theorem catchup_disjoint (maxT T j : Nat) (hT : T ≤ maxT) (h1 : 1 ≤ j) :
    ∀ t, t < T → t ≠ chainRow maxT j := by
  intro t ht; unfold chainRow; omega

/-- Writing four step ints for any row below `R2` stays inside `R2*4*4` bytes. -/
theorem step_row_bytes (row R2 : Nat) (h : row < R2) : (row * 4 + 3) * 4 + 4 ≤ R2 * 4 * 4 := by omega

/-- K/V-only staging: the first row of `step_row0 < 0` is non-negative. -/
theorem kv_only_row0 (s : Int) (h : s < 0) : 0 ≤ -1 - s := by omega

/-- `step_row0 = -1` maps to row 0 (the prefill graph reads rows `[0,T)`). -/
theorem kv_only_row0_prefill : (-1 : Int) - (-1) = 0 := by decide

-- § draft chain

/-- Number of drafts after `steps` accepted chain steps: the first plus one per step. -/
def draftCount (steps : Nat) : Nat := 1 + steps

/-- With chain index `j < max_t - 1` and `j >= 1` at most `max_t - 2` steps run, so at most `max_t - 1` drafts. -/
theorem draftCount_le (maxT steps : Nat) (h : steps + 2 ≤ maxT) : draftCount steps ≤ maxT - 1 := by
  unfold draftCount; omega

/-- The zero-fill range `[n, max_t-1)` is disjoint from emitted drafts `[0,n)` and fits the output array. -/
theorem fill_range (n maxT j : Nat) (h : n ≤ j) (hj : j < maxT - 1) : n ≤ j ∧ j < maxT := by omega

-- § prefill skipping

/-- A chunk entirely at or below `first_needed` is skipped; every cell it would write is below that mark. -/
theorem skipped_chunk_cells (cell0 c T firstNeeded : Int) (h : cell0 + c + T ≤ firstNeeded) :
    ∀ t : Int, 0 ≤ t → t < T → cell0 + c + t < firstNeeded := by
  intro t _ ht; omega

/-- A skipped cell is more than `window + 64` before the prompt end. -/
theorem skipped_cell_outside_window (cell promptLen window firstNeeded : Int)
    (hf : firstNeeded = promptLen - window - 64) (h : cell < firstNeeded) :
    cell + window + 64 < promptLen := by omega

/-- Third step field `(cell+1)/4` never exceeds the second. -/
theorem step_pool_field (cell : Nat) : (cell + 1) / 4 ≤ cell + 1 := Nat.div_le_self _ _

end Ref.MtpDrafter
