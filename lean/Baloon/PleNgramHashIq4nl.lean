/-!
# MODULE: ple_ngram_hash_iq4nl
The PLE n-gram hash (token context -> 16 table row indices) and the IQ4_NL row dequantizer.

## Source semantics
* Context is current token then predecessors, newest first; a negative/EOS predecessor cuts stickily to EOS.
* Head `h` uses `mixed % vocab[h] + offset[h]`; offsets are the running sum of per-head vocab sizes.
* IQ4_NL block: 18 bytes (fp16 `d` + 16 bytes; byte `j` holds elements `j` and `j+16`); 160 values = 90 bytes.

## Form template identifiers
* `ctxTail` ↔ `cut` loop in `ngram_rows`; `vocabs`/`offsets` ↔ `PleConsts`; `kIq4Nl` ↔ dequant table
-/

namespace Strata.Ngram

/-- Build the predecessor context, newest first, with a sticky cut flag. -/
def ctxTail (eos : Int) : Bool → List Int → List Int
  | _, [] => []
  | cut, t :: ts =>
    let cut' := cut || decide (t < 0) || decide (t = eos)
    (if cut' then eos else t) :: ctxTail eos cut' ts

/-- The context keeps one entry per predecessor. -/
theorem ctxTail_length (eos : Int) (cut : Bool) (l : List Int) :
    (ctxTail eos cut l).length = l.length := by
  induction l generalizing cut with
  | nil => simp [ctxTail]
  | cons t ts ih => simp [ctxTail, ih]

/-- Once cut, every remaining position is EOS. -/
theorem ctxTail_cut (eos : Int) (l : List Int) :
    ctxTail eos true l = List.replicate l.length eos := by
  induction l with
  | nil => simp [ctxTail]
  | cons t ts ih => simp [ctxTail, ih, List.replicate_succ]

/-- With no cut and only valid tokens the context is the history unchanged. -/
theorem ctxTail_id (eos : Int) (l : List Int) (h : ∀ t ∈ l, 0 ≤ t ∧ t ≠ eos) :
    ctxTail eos false l = l := by
  induction l with
  | nil => simp [ctxTail]
  | cons t ts ih =>
    have ht := h t (by simp)
    have hts : ∀ u ∈ ts, 0 ≤ u ∧ u ≠ eos := fun u hu => h u (by simp [hu])
    have h1 : ¬ t < 0 := by omega
    have h2 : ¬ t = eos := ht.2
    simp [ctxTail, h1, h2, ih hts]

/-- The XOR-mix of a context with the multiplier list. -/
def mixed (ctx mult : List Nat) : Nat :=
  match ctx, mult with
  | c :: cs, m :: ms => (cs.zip ms).foldl (fun acc p => acc ^^^ (p.1 * p.2)) (c * m)
  | _, _ => 0

/-- A one-token context mixes to just the product. -/
theorem mixed_single (c m : Nat) : mixed [c] [m] = c * m := by
  simp [mixed]

/-- Head index lies in that head's slice of the table. -/
theorem head_index_bounds (mix vocab offset : Nat) (hv : 0 < vocab) :
    offset ≤ mix % vocab + offset ∧ mix % vocab + offset < offset + vocab := by
  have := Nat.mod_lt mix hv
  omega

/-- The artifact's per-head vocab sizes. -/
def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- The artifact's per-head offsets. -/
def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- There are sixteen heads in both tables. -/
theorem heads_sixteen : vocabs.length = 16 ∧ offsets.length = 16 := by
  decide

/-- Offsets are the running sum of vocab sizes, pair by pair. -/
theorem offsets_running_sum :
    (offsets.zip (vocabs.zip (offsets.drop 1))).all
      (fun p => p.1 + p.2.1 == p.2.2) = true := by
  decide

/-- Last head ends at 320,001,446, within the 320,001,536-row table. -/
theorem last_head_end : 300001275 + 20000171 = 320001446 ∧ 320001446 ≤ 320001536 := by
  decide

/-- Every head's index is below the table's row count. -/
theorem index_below_rows (h : Nat) (hh : h < 16) (mix : Nat) :
    mix % (vocabs.getD h 1) + offsets.getD h 0 < 320001536 ∨ vocabs.getD h 1 = 0 := by
  have hv : ∀ h, h < 16 → 0 < vocabs.getD h 1 := by decide
  have hb : ∀ h, h < 16 → offsets.getD h 0 + vocabs.getD h 1 ≤ 320001446 := by decide
  left
  have h1 := Nat.mod_lt mix (hv h hh)
  have h2 := hb h hh
  omega

/-- The IQ4_NL non-linear grid. -/
def kIq4Nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The table has 16 entries. -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by decide

/-- The grid is strictly increasing. -/
theorem kIq4Nl_sorted : (kIq4Nl.zip kIq4Nl.tail).all (fun p => decide (p.1 < p.2)) = true := by
  decide

/-- Every grid value fits in int8. -/
theorem kIq4Nl_int8 : kIq4Nl.all (fun v => decide (-128 ≤ v ∧ v ≤ 127)) = true := by
  decide

/-- A 160-value row is five 32-value blocks of 18 bytes, i.e. 90 bytes. -/
theorem row_bytes : 160 / 32 * 18 = 90 := by decide

/-- Split-halves layout: each output index `k < 32` is `j` or `j+16` for a byte `j < 16`. -/
theorem split_halves (k : Nat) (hk : k < 32) :
    ∃ j, j < 16 ∧ (k = j ∨ k = j + 16) := by
  by_cases h : k < 16
  · exact ⟨k, h, Or.inl rfl⟩
  · exact ⟨k - 16, by omega, Or.inr (by omega)⟩

/-- Low-nibble and high-nibble outputs of one byte never collide. -/
theorem split_halves_disjoint (j : Nat) : j ≠ j + 16 := by omega

/-- Nibble extraction is a valid table index. -/
theorem nibble_lt (b : Nat) : b % 16 < 16 ∧ b / 16 % 16 < 16 := by
  constructor <;> exact Nat.mod_lt _ (by decide)

/-- Row address is in bounds when `row < n_rows`. -/
theorem row_in_file (row n_rows rb : Nat) (h : row < n_rows) : row * rb + rb ≤ n_rows * rb := by
  have : (row + 1) * rb ≤ n_rows * rb := Nat.mul_le_mul_right rb h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

end Strata.Ngram
