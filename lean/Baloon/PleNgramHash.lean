-- Baloon.PleNgramHash: PLE n-gram hash geometry, artifact constants, EOS-cut
-- context window, IQ4_NL codebook, and one-token-in-flight I/O protocol.
-- Independent implementation in the project namespace; proves the same
-- mathematical facts as the theory layer with our own names and proof terms.

namespace Baloon.PleNgramHash

-- ── geometry ─────────────────────────────────────────────────────────────────

/-- PLE_N_HEADS = (NGRAM_SIZE - 1) × HEADS_PER_NGRAM. -/
theorem pleNHeads_eq : (3 - 1) * 8 = 16 := by norm_num

/-- NG_HIST = (PLE_CONV_KERNEL - 1) × NGRAM_SIZE. -/
theorem ngHist_eq : (4 - 1) * 3 = 9 := by norm_num

/-- NG_HC_DIM = NG_HC × NG_N_EMBD. -/
theorem ngHcDim_eq : 4 * 2560 = 10240 := by norm_num

/-- PLE_ROW_BYTES: 160 values in 5 blocks of 18 bytes each. -/
theorem rowBytes_eq : 160 / 32 * 18 = 90 := by decide

/-- All 16 heads × 160 values flatten to NG_N_EMBD. -/
theorem embedWidth_eq : 16 * 160 = 2560 := by norm_num

/-- Table footprint in bytes (the "28.8 GB"). -/
theorem tableFootprint_eq : 320001536 * 90 = 28800138240 := by norm_num

-- ── artifact head constants ───────────────────────────────────────────────────

/-- Per-head vocabulary sizes from the artifact's kVocab table. -/
def headVocabSizes : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets (exclusive prefix sums of headVocabSizes). -/
def headRowOffsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Both tables carry exactly 16 entries. -/
theorem headTables_len : headVocabSizes.length = 16 ∧ headRowOffsets.length = 16 := by decide

/-- headRowOffsets is the exclusive prefix sum of headVocabSizes. -/
theorem headOffsets_prefix : ∀ i : Fin 15,
    headRowOffsets.getD (i.val + 1) 0 =
    headRowOffsets.getD i.val 0 + headVocabSizes.getD i.val 0 := by decide

/-- Every vocabulary size is strictly positive (so modulo is well-defined). -/
theorem headVocab_pos : ∀ v ∈ headVocabSizes, 0 < v := by decide

/-- All 16 head bands lie strictly within the 320001536-row table. -/
theorem headBands_inTable :
    headRowOffsets.getD 15 0 + headVocabSizes.getD 15 0 ≤ 320001536 := by decide

/-- The row index for head h lands in its own band [offset, offset+vocab). -/
theorem rowIdx_inWindow (hv : Nat) (v o : Nat) (hpos : 0 < v) : hv % v + o < o + v := by
  have hlt := Nat.mod_lt hv hpos; omega

-- ── XOR mixing + head index formula ──────────────────────────────────────────

/-- XOR of any value with itself yields zero. -/
theorem xor_selfCancel (a : Nat) : a ^^^ a = 0 := by simp [Nat.xor_self]

/-- Head index (n-2)*8+g is below 16 for n ∈ {2,3}, g < 8. -/
theorem headIdx_lt {n g : Nat} (hn : 2 ≤ n) (hn2 : n ≤ 3) (hg : g < 8) :
    (n - 2) * 8 + g < 16 := by omega

/-- The (n,g)↦head map is injective on {2,3}×{0..7}. -/
theorem headIdx_injOn {n g n' g' : Nat}
    (hn : 2 ≤ n) (hn2 : n ≤ 3) (hg : g < 8)
    (hn' : 2 ≤ n') (hn2' : n' ≤ 3) (hg' : g' < 8)
    (heq : (n - 2) * 8 + g = (n' - 2) * 8 + g') : n = n' ∧ g = g' := by omega

-- ── EOS-cut predecessor context ───────────────────────────────────────────────

/-- The EOS sentinel token id used by the n-gram kernel. -/
def eos_id : Int := 248044

/-- A predecessor is "cut" when it is missing (< 0) or equals the EOS sentinel. -/
def isCutToken (t : Int) : Bool := decide (t < 0) || decide (t = eos_id)

/-- Context slot 1: the newest predecessor, clamped to EOS on a cut. -/
def ctx_newest (p : Int) : Int := if isCutToken p then eos_id else p

/-- Context slot 2: older predecessor, clamped to EOS when either predecessor cuts. -/
def ctx_older (p0 p1 : Int) : Int := if isCutToken p1 || isCutToken p0 then eos_id else p0

/-- A cutting newest predecessor propagates the EOS to both context slots. -/
theorem eosPropagate {p0 p1 : Int} (h : isCutToken p1 = true) :
    ctx_newest p1 = eos_id ∧ ctx_older p0 p1 = eos_id := by
  constructor <;> simp [ctx_newest, ctx_older, h]

/-- A real (non-cut) newest predecessor passes through slot 1 unchanged. -/
theorem real_newest {p1 : Int} (h : isCutToken p1 = false) : ctx_newest p1 = p1 := by
  unfold ctx_newest; simp [h]

/-- When only the older predecessor cuts, slot 1 is real but slot 2 becomes EOS. -/
theorem older_cutOnly {p0 p1 : Int} (h1 : isCutToken p1 = false) (h0 : isCutToken p0 = true) :
    ctx_newest p1 = p1 ∧ ctx_older p0 p1 = eos_id :=
  ⟨real_newest h1, by simp [ctx_older, h1, h0]⟩

/-- When no predecessor cuts, the full two-token history is preserved. -/
theorem fullHistory {p0 p1 : Int} (h1 : isCutToken p1 = false) (h0 : isCutToken p0 = false) :
    ctx_newest p1 = p1 ∧ ctx_older p0 p1 = p0 :=
  ⟨real_newest h1, by simp [ctx_older, h1, h0]⟩

/-- Token 0 is a real token — not EOS, not a missing predecessor. -/
theorem zero_notEos : isCutToken 0 = false := by decide

/-- TOKEN_NULL (−1) is always a cut predecessor. -/
theorem null_isCut : isCutToken (-1) = true := by decide

-- ── IQ4_NL dequantisation codebook ───────────────────────────────────────────

/-- The 16-entry IQ4_NL quantisation lookup table (kvalues_iq4nl). -/
def iq4Tbl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The IQ4_NL table has exactly 16 entries. -/
theorem iq4Tbl_card : iq4Tbl.length = 16 := by decide

/-- The IQ4_NL table is strictly increasing. -/
theorem iq4Tbl_incr : iq4Tbl.Pairwise (· < ·) := by decide

/-- For any byte position j < 16 the two output slots j and j+16 are distinct and in-bounds. -/
theorem nibblePair_sep {j : Nat} (hj : j < 16) : j ≠ j + 16 ∧ j + 16 < 32 := by omega

/-- The high-nibble output slot b*32+j+16 stays within the 160-value row. -/
theorem highNibble_bound {b j : Nat} (hb : b < 5) (hj : j < 16) : b * 32 + j + 16 < 160 := by
  omega

/-- The quant bytes of block b (bytes b*18+2 .. b*18+17) fit inside the 90-byte row. -/
theorem blockBytes_bound {b : Nat} (hb : b < 5) : b * 18 + 2 + 16 ≤ 90 := by omega

-- ── one-token-in-flight I/O state machine ────────────────────────────────────

/-- PLE table I/O state: tracks whether one token read is currently in flight. -/
structure PleIoSt where
  inFlight : Bool

/-- Start a row-read: returns none when a read is already in flight. -/
def pleIssue (s : PleIoSt) : Option PleIoSt :=
  if s.inFlight then none else some ⟨true⟩

/-- Complete a row-read: returns none when no read is in flight. -/
def pleCollect (s : PleIoSt) : Option PleIoSt :=
  if s.inFlight then some ⟨false⟩ else none

/-- A second issue while in-flight is refused. -/
theorem pleIssue_busy : pleIssue ⟨true⟩ = none := by simp [pleIssue]

/-- Collecting with nothing in flight returns none. -/
theorem pleCollect_empty : pleCollect ⟨false⟩ = none := by simp [pleCollect]

/-- A fresh issue from idle yields in-flight state. -/
theorem pleIssue_pending : (pleIssue ⟨false⟩).map (·.inFlight) = some true := by
  simp [pleIssue]

/-- Issue → collect → issue round-trips: the final state is again in-flight. -/
theorem pleIo_roundtrip :
    ((pleIssue ⟨false⟩).bind pleCollect).bind pleIssue = some ⟨true⟩ := by
  simp [pleIssue, pleCollect]

end Baloon.PleNgramHash
