/-!
# MODULE: suffix_lookup_drafter
SOURCES: src/spec/suffix_drafter.cpp
A prompt-lookup drafter: every 3-token suffix of the history is hashed into an open-addressed table
remembering its most recent `WAYS` end positions; `propose` extends the longest backward match of
the current suffix and returns the tokens that followed it.

## Source semantics
* Keys are `mix(...) | 1`, hence never 0, so 0 marks an empty slot.
* The table size is a power of two at least twice the nominal capacity; linear probing from
  `key & mask` visits `mask + 1` distinct slots, i.e. every slot once, before giving up.
* A slot keeps at most `WAYS` positions, newest first; `n = min(k + 1, WAYS)`.
* A match of length `len` requires `len ≤ max_match` and `len ≤ p` before comparing `hist[p-len]`;
  a proposal needs `len ≥ min_match`, and emits at most `max_k` tokens `hist[best_end+1 ..= cur]`.
* `min_match_ = max(3, min_match)` but `max_match_ = max(min_match, max_match)` uses the RAW
  `min_match`, so `min_match < 3` with `max_match < 3` gives `max_match_ < min_match_` and the
  drafter can never propose.
-/

namespace Ref.Suffix

-- § keys and probing

/-- The stored key: the hash with its low bit forced on. -/
def keyOf (h : Nat) : Nat := h ||| 1

/-- A stored key is never the empty-slot marker 0. -/
theorem keyOf_ne_zero (h : Nat) : keyOf h ≠ 0 := by
  intro hz
  have t := Nat.testBit_or h 1 0
  simp only [keyOf] at hz
  rw [hz] at t
  simp only at t

/-- The `p`-th slot probed for a key starting at `h` in a table of `c` slots. -/
def probe (cap hash pos : Nat) : Nat := (hash + pos) % cap

/-- Adding a shift that is nonzero and smaller than the modulus always changes a residue. -/
theorem shift_ne (r d c : Nat) (hr : r < c) (hd0 : 0 < d) (hdc : d < c) : (r + d) % c ≠ r := by
  by_cases hlt : r + d < c
  · rw [Nat.mod_eq_of_lt hlt]; exact (by omega)
  · have hge : c ≤ r + d := by exact (by omega)
    rw [Nat.mod_eq_sub_mod hge, Nat.mod_eq_of_lt (by omega)]; exact (by omega)

/-- **Probing never revisits a slot** within one sweep, so `probes <= mask` covers the whole table. -/
theorem probe_injective (c h p q : Nat) (hpq : p < q) (hq : q < c) : probe c h p ≠ probe c h q := by
  intro he
  simp only [probe] at he
  have hc : 0 < c := by exact (by omega)
  have hd : (q - p) % c = q - p := Nat.mod_eq_of_lt (by omega)
  have e1 : h + q = (h + p) + (q - p) := by exact (by omega)
  have key : (h + q) % c = ((h + p) % c + (q - p)) % c := by
    rw [e1, Nat.add_mod (h + p) (q - p) c, hd]
  exact shift_ne ((h + p) % c) (q - p) c (Nat.mod_lt _ hc) (by omega) (by omega) (by rw [← key, he])

/-- The constructor's doubling loop, `while (cap < target) cap <<= 1`, with fuel. -/
def grow : Nat → Nat → Nat → Nat
  | 0, sz, _ => sz
  | f + 1, sz, t => if sz < t then grow f (sz * 2) t else sz

/-- The table size is always a power of two. -/
theorem grow_pow2 (fuel k t : Nat) : ∃ j, grow fuel (2 ^ k) t = 2 ^ j := by
  induction fuel generalizing k with
  | zero => exact ⟨k, rfl⟩
  | succ fuel ih =>
    simp only [grow]
    split
    · rw [← Nat.pow_succ]; exact ih (k + 1)
    · exact ⟨k, rfl⟩

-- § ways

/-- The newest-first shift of a slot's positions, keeping `W` ways. -/
def pushWay (w e : Nat) (ps : List Nat) : List Nat := (e :: ps).take w

/-- After a push the slot holds `min(k + 1, W)` positions, newest first. -/
theorem pushWay_spec (W e : Nat) (pos : List Nat) (hW : 0 < W) :
    (pushWay W e pos).length = min W (pos.length + 1) ∧ (pushWay W e pos).head? = some e := by
  simp only [pushWay]
  constructor
  · simp only [List.length_take]
  · obtain ⟨w, rfl⟩ : ∃ w, W = w + 1 := ⟨W - 1, by omega⟩
    simp only

-- § matching

/-- The backward match loop with `f` steps of fuel and current length `len`. -/
def matchGo (hist : Nat → Int) (pos cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | f + 1, len => if len ≤ p ∧ hist (p - len) = hist (cur - len) then matchGo hist pos cur f (len + 1) else len

/-- The match length found for an earlier end position `p`. -/
def matchLen (hist : Nat → Int) (p cur maxM : Nat) : Nat := matchGo hist p cur maxM (0 : Nat)

/-- The loop adds at most its fuel. -/
theorem matchGo_le_fuel (hist : Nat → Int) (p cur : Nat) (f len : Nat) :
    matchGo hist p cur f len ≤ len + f := by
  induction f generalizing len with
  | zero => simp only [matchGo]
  | succ fuel ih =>
    simp only [matchGo]
    split
    · have := ih (len + 1); exact (by omega)
    · exact (by omega)

/-- The loop never reads before the start of the history: the length stays ≤ `p + 1`. -/
theorem matchGo_le_pos (hist : Nat → Int) (p cur : Nat) (f len : Nat) (h : len ≤ p + 1) :
    matchGo hist p cur f len ≤ p + 1 := by
  induction f generalizing len with
  | zero => simpa [matchGo]
  | succ fuel ih =>
    simp only [matchGo]
    split
    · next hc => exact ih (len + 1) (by omega)
    · exact h

/-- A match is bounded by `max_match` and by `p + 1`. -/
theorem matchLen_bounds (hist : Nat → Int) (p cur maxM : Nat) :
    matchLen hist p cur maxM ≤ maxM ∧ matchLen hist p cur maxM ≤ p + 1 := by
  simp only [matchLen]
  refine ⟨?_, matchGo_le_pos hist p cur maxM 0 (by omega)⟩
  have := matchGo_le_fuel hist p cur maxM 0
  exact (by omega)

-- § proposals

/-- The tokens `propose` emits: `hist[e+1 ..= cur]`, at most `k` of them. -/
def continuation (h : Nat → Int) (p q n : Nat) : List Int :=
  ((List.range (q - p)).map (fun i => h (p + 1 + i))).take n

/-- A proposal never exceeds `max_k` tokens nor runs past the current position. -/
theorem continuation_length (hist : Nat → Int) (e cur k : Nat) :
    (continuation hist e cur k).length = min k (cur - e) := by
  simp only [continuation, List.length_take]

/-- The effective minimum: never below 3. -/
def minMatchEff (mn : Nat) : Nat := max 3 mn

/-- The effective maximum, as written: built from the RAW `min_match`. -/
def maxMatchEff (mn mx : Nat) : Nat := max mn mx

/-- Whether a match of this length yields a proposal. -/
def proposes (mn l : Nat) : Bool := decide (minMatchEff mn ≤ l)

/-- **Finding**: with `min_match = 1, max_match = 2` the effective window is empty
(`max_match_ = 2 < 3 = min_match_`), so no history ever produces a draft. -/
theorem degenerate_config_never_proposes (hist : Nat → Int) (p cur : Nat) :
    maxMatchEff 1 2 < minMatchEff 1 ∧
    proposes 1 (matchLen hist p cur (maxMatchEff 1 2)) = false := by
  have hb := (matchLen_bounds hist p cur (maxMatchEff 1 2)).1
  refine ⟨by decide, ?_⟩
  simp only [proposes]
  simp only [decide_eq_false_iff_not]
  simp only [maxMatchEff, minMatchEff] at hb ⊢
  exact (by omega)

/-- When `max_match ≥ 3`, the effective window is nonempty for every `min_match ≤ max_match`. -/
theorem sane_config (minM maxM : Nat) (h3 : 3 ≤ maxM) (h : minM ≤ maxM) :
    minMatchEff minM ≤ maxMatchEff minM maxM := by
  simp only [minMatchEff] maxMatchEff; exact (by omega)

end Ref.Suffix

