/-!
# MODULE: ngram_hash
SOURCES: src/kernels/ngram.cpp
The PLE n-gram hash (64-bit multiply/xor mixing, per-head modulus and offset), the EOS-cut predecessor context,
the IQ4_NL row layout (5 blocks of 18 bytes, split-halves nibbles), and the table size / issue-collect protocol.

## Source semantics
* `mixed = ctx[0]*m[0] ^ ctx[1]*m[1] ^ ...`, each product wrapping mod 2^64.
* Head `h` maps `mixed` to `mixed % vocab[h] + offset[h]`; the 16 head ranges are adjacent and fit in the table.
* Once a predecessor is EOS/null the cut persists: all older context slots become EOS.
* A row is 160 values = 5 blocks of (2-byte scale + 16 bytes of nibbles) = 90 bytes; element `b*32 + j` is the low
  nibble of `qs[j]`, `b*32 + j + 16` the high nibble.
* `issue` is refused while a token is in flight; `collect` requires one.

## Form template identifiers
PleConsts.mult/vocab/offset -> multL/vocabL/offsetL; `ngram_mixed` -> mixedN; cut loop -> ctxRest;
`kIq4Nl` -> iq4Tbl; `pending` -> issue/collect state.
-/

namespace Strata.NgramHash

-- § hash

/-- Wrap to 64 bits. -/
def wrap (x : Nat) : Nat := x % 2 ^ 64

/-- Mixed hash of two context tokens. -/
def mixed2 (c0 c1 m0 m1 : Nat) : Nat := wrap (c0 * m0) ^^^ wrap (c1 * m1)

/-- Mixed hash of three context tokens (assignment then xors). -/
def mixed3 (c0 c1 c2 m0 m1 m2 : Nat) : Nat := wrap (c0 * m0) ^^^ wrap (c1 * m1) ^^^ wrap (c2 * m2)

/-- Wrapped products fit in 64 bits. -/
theorem wrap_lt (x : Nat) : wrap x < 2 ^ 64 := Nat.mod_lt _ (Nat.two_pow_pos 64)

/-- The two-token hash is a 64-bit value. -/
theorem mixed2_lt (c0 c1 m0 m1 : Nat) : mixed2 c0 c1 m0 m1 < 2 ^ 64 :=
  Nat.xor_lt_two_pow (wrap_lt _) (wrap_lt _)

/-- The three-token hash is a 64-bit value. -/
theorem mixed3_lt (c0 c1 c2 m0 m1 m2 : Nat) : mixed3 c0 c1 c2 m0 m1 m2 < 2 ^ 64 :=
  Nat.xor_lt_two_pow (Nat.xor_lt_two_pow (wrap_lt _) (wrap_lt _)) (wrap_lt _)

-- § per-head ranges

/-- Head moduli (from the artifact's metadata). -/
def vocabL : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Head offsets (from the artifact's metadata). -/
def offsetL : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Modulus of head `i`. -/
def vocab (i : Nat) : Nat := vocabL.getD i 0

/-- Offset of head `i`. -/
def offset (i : Nat) : Nat := offsetL.getD i 0

/-- Every head modulus is positive, so `% vocab` is a genuine reduction. -/
theorem vocab_pos : ∀ i, i < 16 → 0 < vocab i := by decide

/-- Head ranges are adjacent: `offset (i+1) = offset i + vocab i`. -/
theorem offset_succ : ∀ i, i < 15 → offset (i + 1) = offset i + vocab i := by decide

/-- Every head's range ends inside the 320,001,536-row table. -/
theorem range_in_table : ∀ i, i < 16 → offset i + vocab i ≤ 320001536 := by decide

/-- A hashed row index lies in its head's range, hence in the table. -/
theorem row_in_range (mixed i : Nat) (hi : i < 16) : mixed % vocab i + offset i < offset i + vocab i ∧
    mixed % vocab i + offset i < 320001536 + 1 := by
  have h1 := Nat.mod_lt mixed (vocab_pos i hi)
  have h2 := range_in_table i hi
  constructor <;> omega

/-- Heads of one n-gram order are `base + g`, with `base = (n-2)*8` and `g < 8`: always below 16 for `n ∈ {2,3}`. -/
theorem head_index_lt (n g : Nat) (hn1 : 2 ≤ n) (hn2 : n ≤ 3) (hg : g < 8) : (n - 2) * 8 + g < 16 := by omega

/-- Output position `i*16 + h` stays inside the token's 16-row slice. -/
theorem out_index_lt (i h : Nat) (hh : h < 16) : i * 16 + h < (i + 1) * 16 := by omega

/-- Predecessor `s ∈ {1,2}` is read at `i*2 + (2 - s)`: newest at slot 1, inside the token's two-entry window. -/
theorem prev_index_lt (i s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) : i * 2 + (2 - s) < (i + 1) * 2 := by omega

-- § EOS cut

/-- Context after the first token, newest predecessor first; `cut` latches on null (<0) or EOS. -/
def ctxRest (eos : Int) : List Int → Bool → List Int
  | [], _ => []
  | t :: ts, cut =>
    let cut' := cut || decide (t < 0) || decide (t = eos)
    (if cut' then eos else t) :: ctxRest eos ts cut'

/-- One context slot per predecessor. -/
theorem ctxRest_length (eos : Int) (ps : List Int) : ∀ cut, (ctxRest eos ps cut).length = ps.length := by
  induction ps with
  | nil => intro cut; rfl
  | cons t ts ih => intro cut; simp [ctxRest, ih]

/-- Once cut, everything older is EOS. -/
theorem ctxRest_cut (eos : Int) (ps : List Int) : ctxRest eos ps true = List.replicate ps.length eos := by
  induction ps with
  | nil => rfl
  | cons t ts ih => simp [ctxRest, ih, List.replicate_succ]

-- § IQ4_NL row layout

/-- 90 bytes = 5 blocks of 18 bytes. -/
theorem row_bytes : 5 * 18 = 90 := by decide

/-- A block byte `b*18 + 2 + j` (`b<5`, `j<16`) is inside the 90-byte row. -/
theorem qs_byte_in_row (b j : Nat) (hb : b < 5) (hj : j < 16) : b * 18 + 2 + j < 90 := by omega

/-- Low/high nibble outputs `b*32 + j` and `b*32 + j + 16` are inside the 160 values. -/
theorem out_element_lt (b j half : Nat) (hb : b < 5) (hj : j < 16) (hh : half < 2) : b * 32 + j + 16 * half < 160 := by
  omega

/-- Every output index `idx < 160` is written exactly once, by (block, j, half). -/
theorem out_element_bijective (idx : Nat) (h : idx < 160) :
    idx / 32 < 5 ∧ idx % 16 < 16 ∧ idx % 32 / 16 < 2 ∧ idx / 32 * 32 + idx % 16 + 16 * (idx % 32 / 16) = idx := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- A packed byte splits into two valid 4-bit codes. -/
theorem nibble_split (q : Nat) (h : q < 256) : q % 16 < 16 ∧ q / 16 < 16 ∧ q = 16 * (q / 16) + q % 16 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- The IQ4_NL codebook. -/
def iq4L : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook entry. -/
def iq4Tbl (i : Nat) : Int := iq4L.getD i 0

/-- The codebook is strictly increasing. -/
theorem iq4_increasing : ∀ i, i < 15 → iq4Tbl i < iq4Tbl (i + 1) := by decide

-- § table size and gather layout

/-- Row `row < rows` ends within `rows * 90` bytes. -/
theorem table_row_end (row rows : Nat) (h : row < rows) : row * 90 + 90 ≤ rows * 90 := by omega

/-- Head-slowest gather: `h*160 + d` is inside the 2560-float output. -/
theorem gather_index_lt (h d : Nat) (hh : h < 16) (hd : d < 160) : h * 160 + d < 2560 := by omega

-- § issue / collect protocol

/-- `issue` on a table with a token in flight is refused. -/
def issue (pending : Bool) : Option Bool := if pending then none else some true

/-- `collect` without a pending issue is refused. -/
def collect (pending : Bool) : Option Bool := if pending then some false else none

/-- Issue then collect round-trips to an idle table. -/
theorem issue_collect_roundtrip : (issue false).bind collect = some false := by decide

/-- A second issue before collect is refused. -/
theorem double_issue_refused : (issue false).bind issue = none := by decide

end Strata.NgramHash
