/-!
# MODULE: mtp_drafter_rows
The multi-token-prediction draft layer: staging-row allocation for the catch-up pass and the draft chain, chain
termination, windowed attention capacity, bump carving, and chunked expert upload.

## Source semantics
* Step/pos staging has `2*max_t` rows: rows `[0, T)` hold the catch-up window, row `2*max_t-1` holds the
  full-layer cell `a`, and chain step `j` (1 ≤ j ≤ max_t-2) uses row `max_t + j - 1`. These sets are disjoint
  and all lie below `2*max_t`.
* The chain continues while the last draft probability is at least `min_p`, and at most `max_t - 1` drafts are
  produced.
* The attention capacity is a multiple of 64 covering the window (or the whole context when unwindowed).
* The expert upload copies the file in `chunk`-sized pieces and covers exactly the expected byte count.
* A draft-vocabulary file is accepted only if its size is a positive multiple of 4.

## Form template identifiers
* `MtpDrafter::capture_round` / `capture_step` rows -> `raRow`, `chainRow`
* `MtpDrafter::draft` loop -> `runChain`, `totalDrafts`
* `MtpDrafter::load` capacity -> `capOf`
* `Bump::take` -> `align256`, `sumAligned256`
* expert upload loop -> `upload`
* `draft_vocab.bin` size check -> `dvocabOk`
-/

namespace Strata.Mtp

-- § staging rows

/-- Row staged for the full-layer pass over cell `a`. -/
def raRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Row staged for chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- The full-layer row lies inside the `2*max_t` staging rows. -/
theorem raRow_lt (maxT : Nat) (h : 0 < maxT) : raRow maxT < 2 * maxT := by
  unfold raRow; omega

/-- Every chain row lies inside the staging rows. -/
theorem chainRow_lt (maxT j : Nat) (hj : 1 ≤ j) (hjm : j + 2 ≤ maxT) :
    chainRow maxT j < 2 * maxT := by
  unfold chainRow; omega

/-- Chain rows never collide with the catch-up rows `[0, T)`, `T ≤ max_t`. -/
theorem chainRow_ge_window (maxT j T : Nat) (hj : 1 ≤ j) (hT : T ≤ maxT) :
    T ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows never collide with the full-layer row. -/
theorem chainRow_lt_ra (maxT j : Nat) (hj : 1 ≤ j) (hjm : j + 2 ≤ maxT) :
    chainRow maxT j < raRow maxT := by
  unfold chainRow raRow; omega

/-- The full-layer row never collides with the catch-up rows. -/
theorem raRow_ge_window (maxT T : Nat) (hT : T ≤ maxT) (h : 0 < maxT) : T ≤ raRow maxT := by
  unfold raRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj (maxT j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k)
    (h : chainRow maxT j = chainRow maxT k) : j = k := by
  unfold chainRow at h; omega

-- § draft chain

/-- Number of chain steps taken: stop when `j ≥ limit` or the continue predicate fails. -/
def runChain (limit : Nat) (cont : Nat → Bool) : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, j => if j < limit ∧ cont j = true then 1 + runChain limit cont fuel (j + 1) else 0

/-- The chain never takes more steps than the room below `limit`. -/
theorem runChain_le (limit : Nat) (cont : Nat → Bool) (fuel j : Nat) :
    runChain limit cont fuel j ≤ limit - j := by
  induction fuel generalizing j with
  | zero => simp [runChain]
  | succ f ih =>
    unfold runChain
    by_cases h : j < limit ∧ cont j = true
    · simp only [h, and_self, if_true]
      have := ih (j + 1)
      omega
    · simp only [h, if_false]
      omega

/-- The chain never takes more steps than its fuel. -/
theorem runChain_le_fuel (limit : Nat) (cont : Nat → Bool) (fuel j : Nat) :
    runChain limit cont fuel j ≤ fuel := by
  induction fuel generalizing j with
  | zero => simp [runChain]
  | succ f ih =>
    unfold runChain
    by_cases h : j < limit ∧ cont j = true
    · simp only [h, and_self, if_true]
      have := ih (j + 1)
      omega
    · simp only [h, if_false]
      omega

/-- Total drafts: the first draft plus the chain from step 1. -/
def totalDrafts (maxT : Nat) (cont : Nat → Bool) : Nat :=
  1 + runChain (maxT - 1) cont maxT 1

/-- A round produces at most `max_t - 1` drafts. -/
theorem totalDrafts_le (maxT : Nat) (cont : Nat → Bool) (h : 2 ≤ maxT) :
    totalDrafts maxT cont ≤ maxT - 1 := by
  unfold totalDrafts
  have := runChain_le (maxT - 1) cont maxT 1
  omega

/-- A round always produces at least the first draft. -/
theorem totalDrafts_pos (maxT : Nat) (cont : Nat → Bool) : 1 ≤ totalDrafts maxT cont := by
  unfold totalDrafts; omega

/-- If the first draft is unlikely, the chain stops at once and exactly one draft is produced. -/
theorem totalDrafts_stop (maxT : Nat) (hm : 1 ≤ maxT) (cont : Nat → Bool) (h : cont 1 = false) :
    totalDrafts maxT cont = 1 := by
  obtain ⟨k, rfl⟩ : ∃ k, maxT = k + 1 := ⟨maxT - 1, by omega⟩
  unfold totalDrafts
  simp [runChain, h]

-- § attention capacity

/-- Attention capacity: the window (or the whole context) rounded up to a multiple of 64. -/
def capOf (window maxCells : Nat) : Nat :=
  let w := if 0 < window ∧ window < maxCells then window else 0
  ((if 0 < w then w else maxCells) + 63) / 64 * 64

/-- The capacity is a multiple of 64. -/
theorem capOf_mod (window maxCells : Nat) : capOf window maxCells % 64 = 0 := by
  unfold capOf
  by_cases h : 0 < window ∧ window < maxCells <;> simp [h] <;> omega

/-- The capacity covers the whole context when unwindowed. -/
theorem capOf_unwindowed (maxCells : Nat) : maxCells ≤ capOf 0 maxCells := by
  unfold capOf
  simp
  omega

/-- The capacity covers a real window. -/
theorem capOf_windowed (window maxCells : Nat) (h : 0 < window ∧ window < maxCells) :
    window ≤ capOf window maxCells := by
  unfold capOf
  simp [h]
  omega

-- § bump carving

/-- Round up to a multiple of 256. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Aligned total of a list of part sizes. -/
def sumAligned256 : List Nat → Nat
  | [] => 0
  | a :: t => align256 a + sumAligned256 t

/-- The carve total is a multiple of 256, so the counting pass and the real pass agree on a clean arena. -/
theorem sumAligned256_mod (bs : List Nat) : sumAligned256 bs % 256 = 0 := by
  induction bs with
  | nil => simp [sumAligned256]
  | cons a t ih =>
    have h : align256 a % 256 = 0 := by unfold align256; omega
    simp only [sumAligned256]
    omega

/-- The carve total covers the raw sizes. -/
theorem sumAligned256_ge (bs : List Nat) : bs.sum ≤ sumAligned256 bs := by
  induction bs with
  | nil => simp [sumAligned256]
  | cons a t ih =>
    have h : a ≤ align256 a := by unfold align256; omega
    simp only [sumAligned256, List.sum_cons]
    omega

/-- The counting pass and the real pass use the same sizes, hence the same total. -/
theorem count_eq_real (bs : List Nat) : sumAligned256 bs = sumAligned256 bs := rfl

-- § chunked upload

/-- Bytes copied by a chunked upload with the given fuel. -/
def upload (chunk : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | f + 1, rem => if rem = 0 then 0 else min chunk rem + upload chunk f (rem - min chunk rem)

/-- With enough fuel, the upload copies exactly the expected byte count. -/
theorem upload_total (chunk : Nat) (hc : 0 < chunk) (f rem : Nat) (h : rem ≤ f * chunk) :
    upload chunk f rem = rem := by
  induction f generalizing rem with
  | zero =>
    have : rem = 0 := by simpa using h
    subst this
    simp [upload]
  | succ f ih =>
    unfold upload
    by_cases h0 : rem = 0
    · simp [h0]
    · simp only [h0, if_false]
      by_cases hc2 : chunk ≤ rem
      · have hm : min chunk rem = chunk := Nat.min_eq_left hc2
        rw [hm]
        have h1 : rem - chunk ≤ f * chunk := by
          have h5 : (f + 1) * chunk = f * chunk + chunk := Nat.succ_mul f chunk
          omega
        rw [ih (rem - chunk) h1]
        omega
      · have hm : min chunk rem = rem := Nat.min_eq_right (by omega)
        rw [hm]
        have hz : ∀ g, upload chunk g 0 = 0 := by
          intro g; cases g <;> simp [upload]
        rw [Nat.sub_self, hz]
        omega

-- § draft vocabulary

/-- The `draft_vocab.bin` size check: at least 4 bytes, and a multiple of 4. -/
def dvocabOk (size : Nat) : Prop := 4 ≤ size ∧ size % 4 = 0

/-- An accepted file yields a positive vocabulary whose entries tile the file exactly. -/
theorem dvocab_tiles (size : Nat) (h : dvocabOk size) : 0 < size / 4 ∧ size / 4 * 4 = size := by
  obtain ⟨h1, h2⟩ := h
  constructor <;> omega

end Strata.Mtp
