/-!
# MODULE: ngram_ple_hash
SOURCES: src/kernels/ngram.cpp
The PLE n-gram hash that maps a token and its two predecessors to sixteen table rows, the artifact's
per-head vocabularies and offsets, the IQ4_NL row decode, and the one-token-in-flight protocol of the
direct-I/O table reader.

## Source semantics
* `ngram_mixed`: `mixed = ctx[0]·m[0]` (assignment) then `mixed ^= ctx[j]·m[j]`, all mod 2^64.
* Context: `ctx[0] = tok`; predecessor `s` back is `prev[n_prev - s]` (`prev` oldest first, so `s = 1`
  is the newest).  The cut flag is sticky and evaluated before storing: once a predecessor is negative
  or EOS, that position and every older one are EOS.
* Head `h = (n-2)·8 + g` for n-gram order `n ∈ {2, 3}`, `g < 8`; its row is
  `mixed % vocab[h] + offset[h]`, inside head `h`'s band `[offset[h], offset[h] + vocab[h])`.
  `offset[h+1] = offset[h] + vocab[h]`, and the bands end at 320,001,446 ≤ the table's 320,001,536 rows.
* IQ4_NL row: 160 values in 5 blocks of 18 bytes (f16 scale + 16 bytes); byte `j` holds elements `j`
  and `j + 16` (split halves).  Rows are 90 bytes; the gather is head-slowest: head `h` occupies
  `[160h, 160h + 160)` of the 2560-wide output.
* Direct mode allows one token in flight: `issue` while pending fails, `collect` without `issue` fails,
  and `gather_batch` refuses while a token is in flight.

## Form template identifiers
ngram_mixed → mixed2 / mixed3 · the ctx loop → ctx3 · head index → headOf · row → rowOf ·
vocab/offset → vocabs / offsets · kIq4Nl → iq4nl · split halves → halfElem · PleTable pending → Io
-/

namespace Strata.Kernels.Ngram

-- § the mixing hash

/-- 64-bit wrapping. -/
def wrap (x : Nat) : Nat := x % 2 ^ 64

/-- Order-2 mix: assignment of the first product, XOR of the second. -/
def mixed2 (c0 c1 m0 m1 : Nat) : Nat := wrap (c0 * m0) ^^^ wrap (c1 * m1)

/-- Order-3 mix. -/
def mixed3 (c0 c1 c2 m0 m1 m2 : Nat) : Nat := mixed2 c0 c1 m0 m1 ^^^ wrap (c2 * m2)

/-- The trigram hash extends the bigram hash by one XOR: the source's single loop computes both. -/
theorem mixed3_extends (c0 c1 c2 m0 m1 m2 : Nat) :
    mixed3 c0 c1 c2 m0 m1 m2 = mixed2 c0 c1 m0 m1 ^^^ wrap (c2 * m2) := rfl

/-- Every wrapped product is a 64-bit value. -/
theorem wrap_lt (x : Nat) : wrap x < 2 ^ 64 := Nat.mod_lt _ (Nat.two_pow_pos 64)

-- § context with the EOS cut

/-- `(ctx[1], ctx[2])` from `prev = [p0 (oldest), p1 (newest)]` with the sticky cut. -/
def ctx3 (eos p0 p1 : Int) : Int × Int :=
  let cut1 := decide (p1 < 0) || decide (p1 = eos)
  let c1 := if cut1 then eos else p1
  let cut2 := cut1 || decide (p0 < 0) || decide (p0 = eos)
  let c2 := if cut2 then eos else p0
  (c1, c2)

/-- Without a boundary the context is `[tok, newest, oldest]`: the window is read newest first. -/
theorem ctx3_plain {eos p0 p1 : Int} (h0 : 0 ≤ p0) (h1 : 0 ≤ p1) (e0 : p0 ≠ eos) (e1 : p1 ≠ eos) :
    ctx3 eos p0 p1 = (p1, p0) := by
  have a1 : ¬ p1 < 0 := by omega
  have a0 : ¬ p0 < 0 := by omega
  simp [ctx3, a1, a0, e1, e0]

/-- A boundary at the newest predecessor cuts the whole history: both slots read EOS. -/
theorem ctx3_cut_newest {eos p0 p1 : Int} (h : p1 < 0 ∨ p1 = eos) : ctx3 eos p0 p1 = (eos, eos) := by
  rcases h with h | h
  · simp [ctx3, h]
  · simp [ctx3, h]

/-- A boundary only at the oldest predecessor keeps the newest one. -/
theorem ctx3_cut_oldest {eos p0 p1 : Int} (h1 : 0 ≤ p1) (e1 : p1 ≠ eos) (h : p0 < 0) :
    ctx3 eos p0 p1 = (p1, eos) := by
  have a1 : ¬ p1 < 0 := by omega
  simp [ctx3, a1, e1, h]

-- § heads and rows

/-- The per-head table vocabularies (docs/gguf-dump-shard1.txt). -/
def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- The per-head row offsets. -/
def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Sixteen heads in both tables. -/
theorem tables_len : vocabs.length = 16 ∧ offsets.length = 16 := by decide

/-- The metadata's offsets are the running sum of the vocabularies (the parity test's check). -/
theorem offsets_prefix_sum : ∀ i, i < 15 → offsets[i + 1]! = offsets[i]! + vocabs[i]! := by decide

/-- The last band ends inside the 320,001,536-row table. -/
theorem bands_fit_table : offsets[15]! + vocabs[15]! = 320001446 ∧ 320001446 ≤ 320001536 := by decide

/-- Every vocabulary is positive, so `%` is well defined. -/
theorem vocabs_pos : ∀ i, i < 16 → 0 < vocabs[i]! := by decide

/-- Head index of n-gram order `n` and group `g`. -/
def headOf (n g : Nat) : Nat := (n - 2) * 8 + g

/-- Orders 2 and 3 with 8 groups each fill heads `0..15`, each exactly once. -/
theorem headOf_bij {n g n' g' : Nat} (hn : 2 ≤ n ∧ n ≤ 3) (hn' : 2 ≤ n' ∧ n' ≤ 3) (hg : g < 8) (hg' : g' < 8) :
    headOf n g < 16 ∧ (headOf n g = headOf n' g' → n = n' ∧ g = g') := by
  unfold headOf
  obtain ⟨a, b⟩ := hn
  obtain ⟨c, d⟩ := hn'
  constructor
  · omega
  · intro h
    have : n = 2 ∨ n = 3 := by omega
    have : n' = 2 ∨ n' = 3 := by omega
    omega

/-- The table row of head `h` for a mixed hash. -/
def rowOf (h mixed : Nat) : Nat := mixed % vocabs[h]! + offsets[h]!

/-- A head's row stays in its own band, so the sixteen heads never read each other's rows. -/
theorem rowOf_band {h : Nat} (hh : h < 16) (mixed : Nat) :
    offsets[h]! ≤ rowOf h mixed ∧ rowOf h mixed < offsets[h]! + vocabs[h]! := by
  have hv := vocabs_pos h hh
  have hm := Nat.mod_lt mixed hv
  unfold rowOf
  omega

/-- Every row the hash produces is inside the table. -/
theorem rowOf_in_table {h : Nat} (hh : h < 16) (mixed : Nat) : rowOf h mixed < 320001536 := by
  have hb := (rowOf_band hh mixed).2
  have hlast : ∀ i, i < 16 → offsets[i]! + vocabs[i]! ≤ 320001446 := by decide
  have := hlast h hh
  omega

-- § IQ4_NL rows

/-- The IQ4_NL codebook. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook is strictly increasing and within int8. -/
theorem iq4nl_sorted : ∀ i, i < 15 → iq4nl[i]! < iq4nl[i + 1]! := by decide

/-- Codebook bounds. -/
theorem iq4nl_bounds : ∀ i, i < 16 → (-127 : Int) ≤ iq4nl[i]! ∧ iq4nl[i]! ≤ (113 : Int) := by decide

/-- `code & 15` always indexes the 16-entry codebook. -/
theorem nibble_lt (c : Nat) : c &&& 15 < 16 := by
  have h : c &&& 15 = c % 16 := Nat.and_two_pow_sub_one_eq_mod c 4
  rw [h]
  omega

/-- Row geometry: 5 blocks of 18 bytes hold 160 values; 16 heads × 160 = 2560 outputs. -/
theorem row_geometry : 5 * 18 = 90 ∧ 5 * 32 = 160 ∧ 16 * 160 = 2560 := by decide

/-- Element of block-local byte `j`'s low (`hi = false`) or high nibble. -/
def halfElem (j : Nat) (hi : Bool) : Nat := if hi then j + 16 else j

/-- Split halves: the 16 bytes' two nibbles cover the 32 elements of a block exactly once. -/
theorem halfElem_cover (e : Nat) (he : e < 32) : ∃ j, j < 16 ∧ ∃ hi, halfElem j hi = e := by
  by_cases h : e < 16
  · exact ⟨e, h, false, by simp [halfElem]⟩
  · exact ⟨e - 16, by omega, true, by simp [halfElem]; omega⟩

/-- No two (byte, nibble) pairs decode to the same element. -/
theorem halfElem_inj {j j' : Nat} {hi hi' : Bool} (hj : j < 16) (hj' : j' < 16)
    (h : halfElem j hi = halfElem j' hi') : j = j' ∧ hi = hi' := by
  cases hi <;> cases hi' <;> simp only [halfElem, Bool.false_eq_true, ↓reduceIte] at h
  · exact ⟨h, rfl⟩
  · exfalso; omega
  · exfalso; omega
  · exact ⟨by omega, rfl⟩

/-- Head-slowest output: head `h`, element `d` lands at `160h + d`, inside the 2560-wide vector, injectively. -/
theorem gather_layout {h d h' d' : Nat} (hh : h < 16) (hd : d < 160) (hd' : d' < 160) :
    h * 160 + d < 2560 ∧ (h * 160 + d = h' * 160 + d' → h = h' ∧ d = d') := by
  constructor
  · omega
  · intro e; omega

-- § direct-I/O protocol

/-- Reader state: whether a token's sixteen reads are in flight. -/
structure Io where
  pending : Bool

/-- `issue` in direct mode: refused while a token is in flight. -/
def issue (s : Io) : Bool × Io := if s.pending then (false, s) else (true, ⟨true⟩)

/-- `collect`: requires a prior `issue`, clears the flag either way. -/
def collect (s : Io) : Bool × Io := (s.pending, ⟨false⟩)

/-- `gather_batch` refuses while a token is in flight. -/
def gatherBatchOk (s : Io) : Bool := !s.pending

/-- From idle, issue then collect succeeds and returns to idle. -/
theorem issue_collect_ok : (issue ⟨false⟩).1 = true ∧ (collect (issue ⟨false⟩).2).1 = true ∧
    (collect (issue ⟨false⟩).2).2.pending = false := by decide

/-- A second issue before collecting is refused (one token in flight per table). -/
theorem double_issue_refused : (issue (issue ⟨false⟩).2).1 = false := by decide

/-- Collecting without an issue is an error. -/
theorem collect_without_issue : (collect ⟨false⟩).1 = false := by decide

/-- The batched prompt read cannot interleave with an in-flight token. -/
theorem batch_blocked_in_flight : gatherBatchOk (issue ⟨false⟩).2 = false := by decide

end Strata.Kernels.Ngram
