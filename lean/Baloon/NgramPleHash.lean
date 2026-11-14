/-!
# NgramPleHash — per-layer-embedding n-gram row hash
Formalises the XOR mixing hash of ngram_mixed, the EOS-cut context window,
head/row addressing, the IQ4_NL codebook geometry, and the single-token-in-flight
I/O discipline of the PLE table reader.

Sources: src/kernels/ngram.cpp, include/baloon/ple_ngram_kernels.hpp
-/

namespace Baloon.NgramPle

-- ── 64-bit arithmetic ────────────────────────────────────────────────────────

/-- Reduce to 64-bit range: `x mod 2^64`. -/
def w64 (x : Nat) : Nat := x % 2 ^ 64

/-- `w64` always fits in 64 bits. -/
theorem w64_bound (x : Nat) : w64 x < 2 ^ 64 :=
  Nat.mod_lt _ (Nat.two_pow_pos 64)

-- ── bigram / trigram mixing hash ─────────────────────────────────────────────

/-- Bigram hash: XOR of two 64-bit-wrapped products. -/
def hashBi (c0 c1 m0 m1 : Nat) : Nat := w64 (c0 * m0) ^^^ w64 (c1 * m1)

/-- Trigram hash extends the bigram hash with one more XOR. -/
def hashTri (c0 c1 c2 m0 m1 m2 : Nat) : Nat :=
  hashBi c0 c1 m0 m1 ^^^ w64 (c2 * m2)

/-- `hashTri` decomposes as `hashBi ^^^ w64(third product)`. -/
theorem hashTri_ext (c0 c1 c2 m0 m1 m2 : Nat) :
    hashTri c0 c1 c2 m0 m1 m2 = hashBi c0 c1 m0 m1 ^^^ w64 (c2 * m2) := rfl

-- ── EOS-cut predecessor context ──────────────────────────────────────────────

/-- Context window (newest, oldest) with sticky EOS cut.
    `eos` = EOS token id; `p_new`/`p_old` are predecessor tokens (oldest first). -/
def ngramCtx (eos p_old p_new : Int) : Int × Int :=
  let cut_new := decide (p_new < 0) || decide (p_new = eos)
  let c_new   := if cut_new then eos else p_new
  let cut_old := cut_new || decide (p_old < 0) || decide (p_old = eos)
  let c_old   := if cut_old then eos else p_old
  (c_new, c_old)

/-- No boundary: context is `(newest, oldest)`. -/
theorem ngramCtx_clean {eos p_old p_new : Int}
    (h_new : 0 ≤ p_new) (h_old : 0 ≤ p_old) (en : p_new ≠ eos) (eo : p_old ≠ eos) :
    ngramCtx eos p_old p_new = (p_new, p_old) := by
  simp [ngramCtx, show ¬ p_new < 0 from by omega, en, show ¬ p_old < 0 from by omega, eo]

/-- Newest boundary → both slots become EOS. -/
theorem ngramCtx_cut_new {eos p_old p_new : Int} (h : p_new < 0 ∨ p_new = eos) :
    ngramCtx eos p_old p_new = (eos, eos) := by
  rcases h with h | h <;> simp [ngramCtx, h]

/-- Oldest-only boundary → newest kept, oldest clamped to EOS. -/
theorem ngramCtx_cut_old {eos p_old p_new : Int}
    (hn : 0 ≤ p_new) (en : p_new ≠ eos) (ho : p_old < 0) :
    ngramCtx eos p_old p_new = (p_new, eos) := by
  simp [ngramCtx, show ¬ p_new < 0 from by omega, en, ho]

-- ── vocabulary and offset tables ─────────────────────────────────────────────

/-- Per-head vocabulary sizes (one per head, 16 heads). -/
def headVocab : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets (exclusive prefix sums of `headVocab`). -/
def headOffset : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Both tables have exactly 16 entries. -/
theorem headTables_len : headVocab.length = 16 ∧ headOffset.length = 16 := by decide

/-- `headOffset[h+1] = headOffset[h] + headVocab[h]` for `h < 15`. -/
theorem headOffset_prefix : ∀ h, h < 15 → headOffset[h+1]! = headOffset[h]! + headVocab[h]! := by
  decide

/-- The final band ends at 320 001 446, which fits in the 320 001 536-row table. -/
theorem headBands_fit : headOffset[15]! + headVocab[15]! = 320001446 ∧ 320001446 ≤ 320001536 := by
  decide

/-- Every vocabulary is positive so the modulo is well-defined. -/
theorem headVocab_pos : ∀ h, h < 16 → 0 < headVocab[h]! := by decide

-- ── head index formula ────────────────────────────────────────────────────────

/-- `headIdx n g = (n-2)*8 + g` maps (order, group) → head index. -/
def headIdx (n g : Nat) : Nat := (n - 2) * 8 + g

/-- Orders 2–3 and groups 0–7 tile heads 0–15 bijectively. -/
theorem headIdx_bij {n g n' g' : Nat}
    (hn : 2 ≤ n ∧ n ≤ 3) (hn' : 2 ≤ n' ∧ n' ≤ 3) (hg : g < 8) (hg' : g' < 8) :
    headIdx n g < 16 ∧ (headIdx n g = headIdx n' g' → n = n' ∧ g = g') := by
  unfold headIdx
  obtain ⟨a, b⟩ := hn; obtain ⟨c, d⟩ := hn'
  refine ⟨by omega, fun h => ?_⟩
  have hn2 : n = 2 ∨ n = 3 := by omega
  have hn2' : n' = 2 ∨ n' = 3 := by omega
  omega

/-- Row in the table for head `h` with mixed hash `hv`. -/
def tableRow (h hv : Nat) : Nat := hv % headVocab[h]! + headOffset[h]!

/-- A head's row lies inside its own band `[offset[h], offset[h] + vocab[h])`. -/
theorem tableRow_inBand {h : Nat} (hh : h < 16) (hv : Nat) :
    headOffset[h]! ≤ tableRow h hv ∧ tableRow h hv < headOffset[h]! + headVocab[h]! := by
  have := Nat.mod_lt hv (headVocab_pos h hh)
  unfold tableRow; omega

/-- Every row produced is within the 320 001 536-row table. -/
theorem tableRow_inTable {h : Nat} (hh : h < 16) (hv : Nat) : tableRow h hv < 320001536 := by
  have h2 := (tableRow_inBand hh hv).2
  have : ∀ i, i < 16 → headOffset[i]! + headVocab[i]! ≤ 320001446 := by decide
  omega

-- ── IQ4_NL codebook ──────────────────────────────────────────────────────────

/-- The 16-entry IQ4_NL dequantisation codebook. -/
def iq4Codes : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook is strictly increasing. -/
theorem iq4Codes_strict : ∀ i, i < 15 → iq4Codes[i]! < iq4Codes[i+1]! := by decide

/-- All entries are within signed 8-bit range. -/
theorem iq4Codes_range : ∀ i, i < 16 → (-127 : Int) ≤ iq4Codes[i]! ∧ iq4Codes[i]! ≤ 113 := by
  decide

/-- `c &&& 15` is always a valid codebook index. -/
theorem nibbleIdx_lt (c : Nat) : c &&& 15 < 16 := by
  rw [Nat.and_two_pow_sub_one_eq_mod c 4]; omega

/-- Row geometry: 5 blocks × 18 bytes = 90 bytes; 5 blocks × 32 elements = 160 per row. -/
theorem iq4Row_geom : 5 * 18 = 90 ∧ 5 * 32 = 160 ∧ 16 * 160 = 2560 := by decide

/-- Low nibble → element `j`; high nibble → element `j + 16`. -/
def nibbleElem (j : Nat) (hi : Bool) : Nat := if hi then j + 16 else j

/-- The 16 bytes' two nibbles cover the 32 block elements exactly once. -/
theorem nibbleElem_cover (e : Nat) (he : e < 32) : ∃ j, j < 16 ∧ ∃ hi, nibbleElem j hi = e := by
  by_cases h : e < 16
  · exact ⟨e, h, false, by simp [nibbleElem]⟩
  · exact ⟨e - 16, by omega, true, by simp [nibbleElem]; omega⟩

/-- The (byte, nibble) → element map is injective. -/
theorem nibbleElem_inj {j j' : Nat} {hi hi' : Bool} (hj : j < 16) (hj' : j' < 16)
    (h : nibbleElem j hi = nibbleElem j' hi') : j = j' ∧ hi = hi' := by
  cases hi <;> cases hi' <;> simp only [nibbleElem, Bool.false_eq_true, ↓reduceIte] at h
  · exact ⟨h, rfl⟩
  · exfalso; omega
  · exfalso; omega
  · exact ⟨by omega, rfl⟩

/-- Head-slowest layout: `h * 160 + d` is in `[0, 2560)` and injective in `(h, d)`. -/
theorem gatherPos_valid {h d h' d' : Nat} (hh : h < 16) (hd : d < 160) (hd' : d' < 160) :
    h * 160 + d < 2560 ∧ (h * 160 + d = h' * 160 + d' → h = h' ∧ d = d') := by
  exact ⟨by omega, fun e => by omega⟩

-- ── single-token-in-flight I/O state machine ─────────────────────────────────

/-- PLE table I/O state: is a token's row-fetch in flight? -/
structure PleIoState where
  inFlight : Bool

/-- `ioIssue`: start a fetch; refused when already in flight. -/
def ioIssue (st : PleIoState) : Bool × PleIoState :=
  if st.inFlight then (false, st) else (true, ⟨true⟩)

/-- `ioCollect`: finish a fetch; requires a prior issue, always clears the flag. -/
def ioCollect (st : PleIoState) : Bool × PleIoState := (st.inFlight, ⟨false⟩)

/-- `ioBatchOk`: batch gather is blocked while a fetch is in flight. -/
def ioBatchOk (st : PleIoState) : Bool := !st.inFlight

/-- Happy path: idle → issue → collect returns (true, true, idle). -/
theorem ioRoundTrip :
    (ioIssue ⟨false⟩).1 = true ∧
    (ioCollect (ioIssue ⟨false⟩).2).1 = true ∧
    (ioCollect (ioIssue ⟨false⟩).2).2.inFlight = false := by decide

/-- Double-issue is refused. -/
theorem ioDoubleIssue_refused : (ioIssue (ioIssue ⟨false⟩).2).1 = false := by decide

/-- Collecting without a prior issue returns false. -/
theorem ioCollect_empty : (ioCollect ⟨false⟩).1 = false := by decide

/-- Batch is blocked once a token is in flight. -/
theorem ioBatch_blocked : ioBatchOk (ioIssue ⟨false⟩).2 = false := by decide

end Baloon.NgramPle
