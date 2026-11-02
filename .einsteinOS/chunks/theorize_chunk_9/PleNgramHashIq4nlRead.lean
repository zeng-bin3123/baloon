/-!
# MODULE: ple_ngram_hash_iq4nl_read
The PLE n-gram hash (token context window → 16 table row indices, one per head) and the IQ4_NL
table read (block dequantisation, row layout, gather layout, issue/collect protocol).

## Source semantics
* `ngram_rows`: context `ctx[0]` is the current token; `ctx[s]` is the predecessor `s` positions back,
  read from an OLDEST-FIRST window at index `n_prev - s`. The EOS cut is evaluated BEFORE storing, and is
  sticky: once a predecessor is negative or EOS, it and every older position are EOS.
* `ngram_mixed`: first product is assigned, remaining products are XORed in, all mod 2^64.
* Head `h = (n-2)*HEADS_PER_NGRAM + g`; index = `mixed % vocab[h] + offset[h]`, so every head lands
  in its own disjoint slice of the table.
* IQ4_NL block: 18 bytes (2-byte scale + 16 bytes of nibbles); `qs[j]` holds elements `j` (low nibble)
  and `j+16` (high nibble). A row is `PLE_HEAD_DIM/32 = 5` blocks = 90 bytes.
* `gather` is head-slowest: head `h` occupies `[h*160, (h+1)*160)` of the 2560-float output.
* `PleTable` in-flight protocol: `collect` without `issue` fails; in Direct mode a second `issue`
  while pending fails.

## Form template identifiers
* `ngram_mixed`            → `hash_mix_xor_fold`
* `ngram_rows`             → `window_to_head_indices`
* `ple_artifact_consts`    → `hash_constants`  (`mult`, `vocab`, `offset`)
* `kIq4Nl` / `iq4nl_code`  → `codebook_16`, `codebook_lookup`
* `iq4nl_dequant_row`      → `block_dequant_split_halves`
* `PleTable::gather`       → `head_slowest_gather`
* `PleTable::issue/collect`→ `two_phase_read_protocol`
-/

namespace PleNgram

-- § constants

/-- Number of heads. -/
def nHeads : Nat := 16
/-- Heads per n-gram order. -/
def headsPerNgram : Nat := 8
/-- Largest n-gram order (context length). -/
def ngramSize : Nat := 3
/-- Per-head embedding width. -/
def headDim : Nat := 160
/-- Bytes in one IQ4_NL block. -/
def blockBytes : Nat := 18
/-- Bytes in one table row. -/
def rowBytes : Nat := (headDim / 32) * blockBytes

/-- A row is 90 bytes. -/
theorem rowBytes_eq : rowBytes = 90 := by decide

/-- Concatenated heads fill 2560 floats. -/
theorem nEmbd_eq : nHeads * headDim = 2560 := by decide

/-- Heads are covered by the n-gram orders 2..ngramSize. -/
theorem heads_cover : (ngramSize - 1) * headsPerNgram = nHeads := by decide

/-- Per-head vocabulary sizes. -/
def vocabL : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets. -/
def offsetL : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Head vocabulary (0 outside range). -/
def vocabAt (i : Nat) : Nat := vocabL.getD i 0
/-- Head offset (0 outside range). -/
def offsetAt (i : Nat) : Nat := offsetL.getD i 0

/-- Both constant arrays have one entry per head. -/
theorem consts_length : vocabL.length = nHeads ∧ offsetL.length = nHeads := by decide

/-- Every vocabulary is positive, so `% vocab` is well-behaved. -/
theorem vocab_pos : ∀ i, i < nHeads → 0 < vocabAt i := by decide

/-- The offsets are the exclusive prefix sums of the vocabularies (the slices tile the table). -/
theorem offset_prefix_sum : ∀ i, i < 15 → offsetAt (i + 1) = offsetAt i + vocabAt i := by decide

/-- The first slice starts at row 0. -/
theorem offset_zero : offsetAt 0 = 0 := by decide

/-- Total rows addressed by all heads. -/
theorem total_rows : offsetAt 15 + vocabAt 15 = 320001446 := by decide

/-- Slices are strictly ordered, hence disjoint. -/
theorem slices_ordered : ∀ i, i < 15 → offsetAt i + vocabAt i ≤ offsetAt (i + 1) := by
  intro i hi
  have := offset_prefix_sum i hi
  omega

/-- The hash index of a head always lies inside that head's slice. -/
theorem index_in_slice (mixed i : Nat) (hi : i < nHeads) :
    offsetAt i ≤ mixed % vocabAt i + offsetAt i ∧
    mixed % vocabAt i + offsetAt i < offsetAt i + vocabAt i := by
  have hp := vocab_pos i hi
  have := Nat.mod_lt mixed hp
  omega

/-- Indices of different heads never collide. -/
theorem index_heads_disjoint (m1 m2 i j : Nat) (hi : i < nHeads) (hj : j < nHeads) (hij : i < j) :
    m1 % vocabAt i + offsetAt i < m2 % vocabAt j + offsetAt j := by
  have h1 := (index_in_slice m1 i hi).2
  -- chain: offset i + vocab i ≤ offset (i+1) ≤ ... ≤ offset j
  have chain : ∀ k, i < k → k < nHeads → offsetAt i + vocabAt i ≤ offsetAt k := by
    intro k hk hkn
    induction k with
    | zero => omega
    | succ k ih =>
      by_cases hik : i < k
      · have := ih hik (by omega)
        have h2 := slices_ordered k (by unfold nHeads at hkn; omega)
        have h3 : offsetAt k ≤ offsetAt k + vocabAt k := Nat.le_add_right _ _
        omega
      · have : i = k := by omega
        subst this
        exact slices_ordered i (by unfold nHeads at hkn; omega)
  have := chain j hij hj
  omega

-- § mixing hash

/-- Fold of the products: first is assigned, the rest XORed (mod 2^64 via `UInt64`). -/
def mixedList : List UInt64 → List UInt64 → UInt64
  | c :: cs, m :: ms => cs.zip ms |>.foldl (fun acc p => acc ^^^ (p.1 * p.2)) (c * m)
  | _, _ => 0

/-- Mixing for order 2. -/
def mixed2 (c0 c1 m0 m1 : UInt64) : UInt64 := (c0 * m0) ^^^ (c1 * m1)

/-- Mixing for order 3. -/
def mixed3 (c0 c1 c2 m0 m1 m2 : UInt64) : UInt64 := ((c0 * m0) ^^^ (c1 * m1)) ^^^ (c2 * m2)

/-- The list fold agrees with the unrolled order-2 formula. -/
theorem mixedList_two (c0 c1 m0 m1 : UInt64) :
    mixedList [c0, c1] [m0, m1] = mixed2 c0 c1 m0 m1 := by
  simp [mixedList, mixed2]

/-- The list fold agrees with the unrolled order-3 formula. -/
theorem mixedList_three (c0 c1 c2 m0 m1 m2 : UInt64) :
    mixedList [c0, c1, c2] [m0, m1, m2] = mixed3 c0 c1 c2 m0 m1 m2 := by
  simp [mixedList, mixed3]

/-- Order 3 extends order 2 by one more XORed product. -/
theorem mixed3_extends (c0 c1 c2 m0 m1 m2 : UInt64) :
    mixed3 c0 c1 c2 m0 m1 m2 = mixed2 c0 c1 m0 m1 ^^^ (c2 * m2) := rfl

/-- If the third context entry has product zero, order 3 equals order 2. -/
theorem mixed3_of_zero (c0 c1 c2 m0 m1 m2 : UInt64) (h : c2 * m2 = 0) :
    mixed3 c0 c1 c2 m0 m1 m2 = mixed2 c0 c1 m0 m1 := by
  simp [mixed3, mixed2, h]

-- § head index layout

/-- Head index for order `n` (2..) and slot `g`. -/
def headIdx (n g : Nat) : Nat := (n - 2) * headsPerNgram + g

/-- Head indices are in range for orders 2..3 and slots below 8. -/
theorem headIdx_lt (n g : Nat) (hn2 : 2 ≤ n) (hn3 : n ≤ ngramSize) (hg : g < headsPerNgram) :
    headIdx n g < nHeads := by
  unfold headIdx nHeads ngramSize headsPerNgram at *
  omega

/-- Distinct (order, slot) pairs give distinct heads. -/
theorem headIdx_injective (n g n' g' : Nat) (hg : g < headsPerNgram) (hg' : g' < headsPerNgram)
    (h : headIdx n g = headIdx n' g') (hn : 2 ≤ n) (hn' : 2 ≤ n') : n = n' ∧ g = g' := by
  unfold headIdx headsPerNgram at *
  omega

/-- Every head is reached by some (order, slot) pair. -/
theorem headIdx_surjective (h : Nat) (hh : h < nHeads) :
    ∃ n g, 2 ≤ n ∧ n ≤ ngramSize ∧ g < headsPerNgram ∧ headIdx n g = h := by
  refine ⟨h / headsPerNgram + 2, h % headsPerNgram, by omega, ?_, ?_, ?_⟩
  · unfold nHeads ngramSize headsPerNgram at *; omega
  · unfold headsPerNgram; omega
  · unfold headIdx headsPerNgram; omega

-- § context window with EOS cut

/-- Context entry at distance 1 (NEWEST predecessor) after the cut. -/
def ctx1 (eos p1 : Int) : Int := if p1 < 0 ∨ p1 = eos then eos else p1

/-- Context entry at distance 2 after the sticky cut. -/
def ctx2 (eos p1 p2 : Int) : Int :=
  if (p1 < 0 ∨ p1 = eos) ∨ p2 < 0 ∨ p2 = eos then eos else p2

/-- Window index of the predecessor `s` back: `s = 1` reads the NEWEST (last) entry. -/
def prevIdx (nPrev s : Nat) : Nat := nPrev - s

/-- With three-gram context (`nPrev = 2`), distance 1 reads index 1 (newest), distance 2 reads index 0. -/
theorem prevIdx_newest_first : prevIdx 2 1 = 1 ∧ prevIdx 2 2 = 0 := by decide

/-- The window is walked newest to oldest and never leaves range. -/
theorem prevIdx_lt (nPrev s : Nat) (hs : 1 ≤ s) (hsn : s ≤ nPrev) : prevIdx nPrev s < nPrev := by
  unfold prevIdx; omega

/-- A real, non-EOS newest predecessor is kept verbatim. -/
theorem ctx1_keep (eos p1 : Int) (h0 : 0 ≤ p1) (hne : p1 ≠ eos) : ctx1 eos p1 = p1 := by
  unfold ctx1
  have : ¬ (p1 < 0 ∨ p1 = eos) := by omega
  simp [this]

/-- A cut at distance 1 forces EOS at distance 2 (the cut is sticky). -/
theorem cut_sticky (eos p1 p2 : Int) (h : p1 < 0 ∨ p1 = eos) : ctx2 eos p1 p2 = eos := by
  unfold ctx2
  simp [h]

/-- If the newest predecessor is EOS the cut already happened at distance 1: both entries are EOS,
    never one token of extra history. -/
theorem eos_cuts_both (eos p1 p2 : Int) (h : p1 = eos) :
    ctx1 eos p1 = eos ∧ ctx2 eos p1 p2 = eos := by
  subst h
  constructor
  · simp [ctx1]
  · simp [ctx2]

/-- Without any cut, both predecessors are kept verbatim. -/
theorem no_cut_keeps (eos p1 p2 : Int) (h1 : 0 ≤ p1) (h1e : p1 ≠ eos) (h2 : 0 ≤ p2) (h2e : p2 ≠ eos) :
    ctx1 eos p1 = p1 ∧ ctx2 eos p1 p2 = p2 := by
  have n1 : ¬ (p1 < 0 ∨ p1 = eos) := by omega
  have n2 : ¬ (p2 < 0 ∨ p2 = eos) := by omega
  constructor
  · simp [ctx1, n1]
  · simp [ctx2, n1, n2]

/-- Every context entry is either the source token or EOS (never anything invented). -/
theorem ctx2_cases (eos p1 p2 : Int) : ctx2 eos p1 p2 = eos ∨ ctx2 eos p1 p2 = p2 := by
  unfold ctx2
  split <;> simp

/-- A negative (null) predecessor behaves like EOS. -/
theorem null_is_eos (eos p1 p2 : Int) (h : p1 < 0) : ctx1 eos p1 = eos ∧ ctx2 eos p1 p2 = eos := by
  constructor
  · simp [ctx1, h]
  · simp [ctx2, h]

-- § IQ4_NL codebook

/-- The 16-entry non-linear codebook. -/
def codebook : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook lookup, masking the code to 4 bits like `code & 15`. -/
def code (c : Nat) : Int := codebook.getD (c % 16) 0

/-- Sixteen entries. -/
theorem codebook_length : codebook.length = 16 := by decide

/-- Strictly increasing codebook. -/
theorem codebook_sorted : codebook.Pairwise (· < ·) := by decide

/-- All entries lie in the int8 range with the stated extremes. -/
theorem codebook_range : ∀ c, c < 16 → -127 ≤ code c ∧ code c ≤ 113 := by decide

/-- Lookup only depends on the low four bits. -/
theorem code_mask (c : Nat) : code (c + 16) = code c := by
  unfold code
  have : (c + 16) % 16 = c % 16 := by omega
  rw [this]

/-- Only entries 8.. are positive: code 8 is the first positive. -/
theorem codebook_sign_split : code 7 < 0 ∧ 0 < code 8 := by decide

-- § block dequantisation

/-- Element index written by low nibble of byte `j` in block `b`. -/
def lowIdx (b j : Nat) : Nat := b * 32 + j
/-- Element index written by high nibble of byte `j` in block `b`. -/
def highIdx (b j : Nat) : Nat := b * 32 + j + 16

/-- Split halves: low and high nibble targets are `j` and `j+16`, never `2j` / `2j+1`. -/
theorem split_halves_gap (b j : Nat) : highIdx b j = lowIdx b j + 16 := by
  unfold highIdx lowIdx; omega

/-- All targets are inside the 160-element row. -/
theorem dequant_in_range (b j : Nat) (hb : b < headDim / 32) (hj : j < 16) :
    lowIdx b j < headDim ∧ highIdx b j < headDim := by
  unfold lowIdx highIdx headDim at *
  omega

/-- Writes do not collide: the low/high targets of different (block, byte) pairs differ. -/
theorem dequant_no_collision (b j b' j' : Nat) (hj : j < 16) (hj' : j' < 16) :
    (lowIdx b j = lowIdx b' j' ↔ b = b' ∧ j = j') ∧
    (highIdx b j = highIdx b' j' ↔ b = b' ∧ j = j') ∧
    lowIdx b j ≠ highIdx b' j' := by
  unfold lowIdx highIdx
  refine ⟨⟨fun h => by omega, fun h => by omega⟩, ⟨fun h => by omega, fun h => by omega⟩, ?_⟩
  intro h
  omega

/-- Every element of the row is written by exactly one nibble (full coverage). -/
theorem dequant_covers (i : Nat) (hi : i < headDim) :
    ∃ b j, b < headDim / 32 ∧ j < 16 ∧ (lowIdx b j = i ∨ highIdx b j = i) := by
  unfold headDim at *
  by_cases h : i % 32 < 16
  · exact ⟨i / 32, i % 32, by omega, h, Or.inl (by unfold lowIdx; omega)⟩
  · exact ⟨i / 32, i % 32 - 16, by omega, by omega, Or.inr (by unfold highIdx; omega)⟩

/-- Byte offset of block `b` inside a row. -/
def blockOff (b : Nat) : Nat := b * blockBytes

/-- The last block ends exactly at the end of the row. -/
theorem blocks_fill_row : blockOff (headDim / 32 - 1) + blockBytes = rowBytes := by decide

/-- Scale occupies bytes `[off, off+2)` and nibbles `[off+2, off+18)`. -/
theorem block_layout (b : Nat) : blockOff b + 2 + 16 = blockOff b + blockBytes := by
  unfold blockBytes; omega

-- § table geometry

/-- Byte offset of a row. -/
def rowOff (r : Nat) : Nat := r * rowBytes

/-- Sixteen row reads for a token are exactly 1440 bytes (the 0.5 MB/s figure). -/
theorem token_bytes : nHeads * rowBytes = 1440 := by decide

/-- Rows do not overlap. -/
theorem rows_disjoint (r r' : Nat) (h : r < r') : rowOff r + rowBytes ≤ rowOff r' := by
  unfold rowOff
  have : (r + 1) * rowBytes ≤ r' * rowBytes := Nat.mul_le_mul_right _ h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- A table exactly filling the data section: size check forces the data offset. -/
theorem size_check_determines_start (nRows fileSize start1 start2 : Nat)
    (h1 : nRows * rowBytes = fileSize - start1) (h2 : nRows * rowBytes = fileSize - start2)
    (hs1 : start1 ≤ fileSize) (hs2 : start2 ≤ fileSize) : start1 = start2 := by
  omega

/-- A row index below `nRows` has its whole row inside the table bytes. -/
theorem row_in_bounds (nRows r : Nat) (h : r < nRows) : rowOff r + rowBytes ≤ nRows * rowBytes := by
  have := rows_disjoint r nRows h
  unfold rowOff at *
  have h2 : r * rowBytes + rowBytes ≤ nRows * rowBytes := by
    have := Nat.mul_le_mul_right rowBytes (Nat.succ_le_of_lt h)
    rw [Nat.succ_mul] at this
    exact this
  exact h2

-- § gather layout

/-- Head-slowest output position of element `d` of head `h`. -/
def outPos (h d : Nat) : Nat := h * headDim + d

/-- Output positions lie in the 2560-float buffer. -/
theorem outPos_lt (h d : Nat) (hh : h < nHeads) (hd : d < headDim) : outPos h d < nHeads * headDim := by
  unfold outPos nHeads headDim at *
  omega

/-- The gather layout is injective (head-slowest, no overlap between heads). -/
theorem outPos_injective (h d h' d' : Nat) (hd : d < headDim) (hd' : d' < headDim)
    (e : outPos h d = outPos h' d') : h = h' ∧ d = d' := by
  unfold outPos headDim at *
  omega

/-- The two layouts really differ on a concrete element, so the check is non-vacuous. -/
theorem layouts_differ : outPos 1 0 ≠ 0 * nHeads + 1 := by decide

/-- Gather covers the whole output buffer. -/
theorem outPos_surjective (p : Nat) (hp : p < nHeads * headDim) :
    ∃ h d, h < nHeads ∧ d < headDim ∧ outPos h d = p := by
  unfold nHeads headDim at *
  exact ⟨p / 160, p % 160, by omega, by omega, by unfold outPos headDim; omega⟩

-- § issue / collect protocol

/-- Abstract state of one table: whether a token is in flight, and whether Direct mode is on. -/
structure St where
  pending : Bool
  direct : Bool

/-- `issue`: Direct mode refuses a second in-flight token; other modes always accept. -/
def issue (s : St) : St × Bool :=
  if s.direct then
    (if s.pending then (s, false) else ({ s with pending := true }, true))
  else ({ s with pending := true }, true)

/-- `collect`: fails without a prior issue; otherwise clears `pending` first. -/
def collect (s : St) : St × Bool :=
  if s.pending then ({ s with pending := false }, true) else (s, false)

/-- Collect without issue fails and changes nothing. -/
theorem collect_without_issue (s : St) (h : s.pending = false) : collect s = (s, false) := by
  simp [collect, h]

/-- Issue followed by collect succeeds and leaves nothing in flight. -/
theorem issue_collect (s : St) (h : s.pending = false) :
    (collect (issue s).1).2 = true ∧ (collect (issue s).1).1.pending = false := by
  cases hd : s.direct <;> simp [issue, collect, hd, h]

/-- Direct mode: a second issue while pending is rejected. -/
theorem direct_double_issue (s : St) (hd : s.direct = true) (hp : s.pending = true) :
    (issue s).2 = false := by
  simp [issue, hd, hp]

/-- A successful issue always leaves a token pending. -/
theorem issue_sets_pending (s : St) (h : (issue s).2 = true) : (issue s).1.pending = true := by
  cases hd : s.direct <;> cases hp : s.pending <;> simp [issue, hd, hp] at h ⊢

/-- A second collect after a successful one fails. -/
theorem collect_twice (s : St) : (collect (collect s).1).2 = false := by
  cases hp : s.pending <;> simp [collect, hp]

end PleNgram
