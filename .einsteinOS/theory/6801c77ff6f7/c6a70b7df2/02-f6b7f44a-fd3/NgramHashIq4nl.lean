/-!
# MODULE: ngram_hash_iq4nl
SOURCES: include/strata/kernels/ngram.hpp, src/kernels/ngram.cpp
The PLE n-gram hash (sixteen row indices per token from the last three token ids, XOR of 64-bit products reduced per head
by the artifact's vocab sizes and offsets) and the IQ4_NL row dequantisation and gather layout.

## Source semantics
* The context is `(tok, p1, p0)` with predecessors read newest-first. A predecessor that is null (`< 0`) or EOS cuts the
  context: it and every OLDER position become EOS. The token's own EOS never cuts its own context. Token id 0 is real.
* After the cut every context entry is non-negative, so the cast to uint64 is safe.
* The trigram mix extends the bigram mix by one XOR term (prefixes agree). Equal terms cancel under XOR.
* A head's row is `mixed % vocab[h] + offset[h]`, inside `[offset[h], offset[h]+vocab[h])`; the 16 head ranges tile
  `0..300001275+20000171` and all lie inside the 320,001,536-row table.
* IQ4_NL nibbles are SPLIT-HALF: byte `j` carries elements `j` and `j+16`. A row is 5 blocks of 18 bytes = 90 bytes for
  160 elements; 16 rows flatten head-slowest into exactly 2560 floats.
* The codebook has no `-8` shift: it is strictly increasing from -127 to 113.

## Form template identifiers
ngram_rows -> ctx3 / headRow ; ngram_mixed -> mixed2 / mixed3 ; PLE_EOS_TOKEN_ID -> EOS ; TOKEN_NULL -> -1 ;
head_vocab_sizes -> vocabL ; head_offsets -> offL ; kvalues_iq4nl -> kv ; iq4nl_dequant_row -> splitIdx ;
PLE_ROW_BYTES -> 90 ; gather -> headSlowest
-/

namespace Strata.Ngram

-- § context construction

/-- The end-of-sequence token id. -/
def EOS : Int := 248044

/-- A predecessor cuts the context when it is null (negative) or EOS. -/
def isCut (t : Int) : Bool := decide (t < 0) || decide (t = EOS)

/-- Context `(token, newest predecessor, older predecessor)` after the forward-propagating cut. -/
def ctx3 (tok p0 p1 : Int) : Int × Int × Int :=
  let cut1 := isCut p1
  let cut2 := cut1 || isCut p0
  (tok, if cut1 then EOS else p1, if cut2 then EOS else p0)

/-- The token's own id is never altered by the cut. -/
theorem ctx3_token (tok p0 p1 : Int) : (ctx3 tok p0 p1).1 = tok := rfl

/-- Token id 0 is a real token, not a cut. -/
theorem zero_not_cut : isCut 0 = false := by decide

/-- The null predecessor (-1) cuts. -/
theorem null_cuts : isCut (-1) = true := by decide

/-- EOS cuts. -/
theorem eos_cuts : isCut EOS = true := by decide

/-- THE CUT PROPAGATES FORWARD: if the newest predecessor cuts, the older position is EOS regardless of its value. -/
theorem cut_forward (tok p0 p1 : Int) (h : isCut p1 = true) : (ctx3 tok p0 p1).2.2 = EOS := by
  simp [ctx3, h]

/-- With no cut the context is the real history. -/
theorem no_cut_keeps_history (tok p0 p1 : Int) (h1 : isCut p1 = false) (h0 : isCut p0 = false) :
    ctx3 tok p0 p1 = (tok, p1, p0) := by
  simp [ctx3, h1, h0]

/-- The token's own EOS does not influence the predecessors' handling. -/
theorem own_eos_no_cut (p0 p1 : Int) : (ctx3 EOS p0 p1).2 = (ctx3 0 p0 p1).2 := rfl

private theorem pick_nonneg (c : Bool) (p : Int) (h : c = false → 0 ≤ p) :
    0 ≤ (if c = true then EOS else p) := by
  cases c
  · simpa using h rfl
  · simp [EOS]

/-- After the cut every history entry is non-negative, so the uint64 cast is value-preserving. -/
theorem ctx3_nonneg (tok p0 p1 : Int) : 0 ≤ (ctx3 tok p0 p1).2.1 ∧ 0 ≤ (ctx3 tok p0 p1).2.2 := by
  refine ⟨pick_nonneg _ _ ?_, pick_nonneg _ _ ?_⟩
  · intro h
    simp [isCut] at h
    omega
  · intro h
    simp [isCut] at h
    omega

-- § mixing

/-- Product reduced mod 2^64. -/
def mixU (a b : Nat) : Nat := (a * b) % 18446744073709551616

/-- Bigram mix: XOR of the first two products. -/
def mixed2 (c0 c1 m0 m1 : Nat) : Nat := mixU c0 m0 ^^^ mixU c1 m1

/-- Trigram mix: the bigram mix XORed with the third product. -/
def mixed3 (c0 c1 c2 m0 m1 m2 : Nat) : Nat := mixed2 c0 c1 m0 m1 ^^^ mixU c2 m2

/-- Bigram and trigram heads agree about recent history: the trigram mix is the bigram mix plus one term. -/
theorem mixed3_extends_mixed2 (c0 c1 c2 m0 m1 m2 : Nat) :
    mixed3 c0 c1 c2 m0 m1 m2 = mixed2 c0 c1 m0 m1 ^^^ mixU c2 m2 := rfl

/-- Why XOR and not sum: equal terms cancel under XOR. -/
theorem xor_cancels (x : Nat) : x ^^^ x = 0 := Nat.xor_self x

/-- Products are reduced to 64 bits. -/
theorem mixU_lt (a b : Nat) : mixU a b < 18446744073709551616 :=
  Nat.mod_lt _ (by decide)

/-- Two equal tokens with equal multipliers cancel in the bigram mix. -/
theorem mixed2_equal_cancels (c m : Nat) : mixed2 c c m m = 0 := by
  unfold mixed2
  exact Nat.xor_self _

-- § head ranges

/-- The sixteen vocab sizes (NOT an arithmetic progression). -/
def vocabL : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- The sixteen head offsets. -/
def offL : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- The offsets are the running sum of the vocab sizes, so the head ranges tile the index space with no gap or overlap. -/
theorem offsets_are_running_sum : offL = (vocabL.scanl (· + ·) 0).take 16 := by decide

/-- Every head range lies inside the 320,001,536-row table. -/
theorem heads_inside_table :
    ∀ h, h < 16 → offL.getD h 0 + vocabL.getD h 0 ≤ 320001536 := by decide

/-- The artifact's vocab sizes are not `20000003 + 10 i`: the first gap is 20. -/
theorem first_gap_is_twenty : vocabL.getD 1 0 - vocabL.getD 0 0 = 20 := by decide

/-- The reference selftest's constants differ from the artifact's. -/
theorem not_selftest_progression : vocabL.getD 1 0 ≠ 20000003 + 10 * 1 := by decide

/-- A head's row index. -/
def headRow (mixed vocab off : Nat) : Nat := mixed % vocab + off

/-- A head's row is inside its own range. -/
theorem headRow_bounds (mixed vocab off : Nat) (h : 0 < vocab) :
    off ≤ headRow mixed vocab off ∧ headRow mixed vocab off < off + vocab := by
  unfold headRow
  have := Nat.mod_lt mixed h
  omega

/-- Heads 0..7 use the bigram and 8..15 the trigram: `(n-2)*8 + g` enumerates heads uniquely. -/
theorem head_decomposition (h : Nat) (hh : h < 16) :
    (2 + h / 8 - 2) * 8 + h % 8 = h := by
  omega

-- § IQ4_NL

/-- The codebook `kvalues_iq4nl`. -/
def kv : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook is strictly increasing. -/
theorem kv_strictly_increasing : ∀ i, i < 15 → kv.getD i 0 < kv.getD (i + 1) 0 := by decide

/-- All codes lie in [-127, 113]. -/
theorem kv_range : ∀ i, i < 16 → -127 ≤ kv.getD i 0 ∧ kv.getD i 0 ≤ 113 := by decide

/-- There is no `-8` shift: entry 8 is 1, not 1 - 8. -/
theorem kv_no_shift : kv.getD 8 0 = 1 := by decide

/-- A byte splits into two 4-bit codes. -/
theorem nibble_roundtrip (b : Nat) (h : b < 256) : b / 16 * 16 + b % 16 = b := by omega

/-- Both nibbles index the 16-entry codebook. -/
theorem nibbles_in_range (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 := by omega

/-- SPLIT-HALF layout: nibble `half` of byte `j` is element `j + 16 * half` of the block. -/
def splitIdx (j half : Nat) : Nat := j + 16 * half

/-- The split-half map is injective on bytes `j < 16` and halves `< 2`. -/
theorem splitIdx_injective (j h j' h' : Nat) (hj : j < 16) (hj' : j' < 16)
    (hh : h < 2) (hh' : h' < 2) (e : splitIdx j h = splitIdx j' h') : j = j' ∧ h = h' := by
  unfold splitIdx at e; omega

/-- ...and onto the 32 block elements. -/
theorem splitIdx_surjective (i : Nat) (hi : i < 32) :
    ∃ j h, j < 16 ∧ h < 2 ∧ splitIdx j h = i :=
  ⟨i % 16, i / 16, by omega, by omega, by unfold splitIdx; omega⟩

/-- The interleaved reading (2j, 2j+1) is a different map: byte 1's high nibble lands at 17, not 3. -/
theorem split_ne_interleaved : splitIdx 1 1 ≠ 2 * 1 + 1 := by decide

-- § row geometry and gather

/-- Geometry: 160 elements = 5 blocks of 32; a row is 5 * 18 = 90 bytes. -/
theorem row_geometry : 160 / 32 = 5 ∧ 5 * 18 = 90 := by decide

/-- Sixteen heads of 160 make exactly n_embd = 2560. -/
theorem heads_make_embd : 16 * 160 = 2560 := by decide

/-- Head-slowest flatten index. -/
def headSlowest (h d : Nat) : Nat := h * 160 + d

/-- The flatten stays inside the 2560-float output. -/
theorem headSlowest_lt (h d : Nat) (hh : h < 16) (hd : d < 160) : headSlowest h d < 2560 := by
  unfold headSlowest; omega

/-- ...and is injective, so no two (head, dim) pairs collide. -/
theorem headSlowest_injective (h d h' d' : Nat) (hd : d < 160) (hd' : d' < 160)
    (e : headSlowest h d = headSlowest h' d') : h = h' ∧ d = d' := by
  unfold headSlowest at e; omega

/-- The head-fastest layout `d * 16 + h` is a different function (so a reshape check is not vacuous). -/
theorem headFastest_differs : (1 * 160 + 0) ≠ (0 * 16 + 1) := by decide

/-- One token's 16 row reads touch 1,440 bytes. -/
theorem bytes_per_token_gather : 16 * 90 = 1440 := by decide

/-- Table fit check (shard also holding other tensors): the table's end must be inside the data section. -/
theorem table_end_inside_file (dataStart off need fileSize : Nat) (hd : dataStart ≤ fileSize)
    (h : off + need ≤ fileSize - dataStart) : dataStart + off + need ≤ fileSize := by
  omega

end Strata.Ngram
