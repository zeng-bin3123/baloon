/-!
# MODULE: kv_q8_cache
SOURCES: src/kernels/cuda/kv_q8.cu
INT8-quantised paged KV cache: append one token (per-64-value-group max-abs scale stored as fp16, values quantised
against the STORED scale, clamped to [-127,127]) and gather cells back to fp16 scratch.

## Source semantics
* head_dim must be a multiple of KV_Q8_GROUP = 64; one block = one 64-value group of one KV head of K or V.
* row = (page*kv_heads + h)*page_size + pos % page_size; scale slot = row*groups + g, groups = head_dim/64.
* gather: thread = 4 consecutive values of one (cell, head); all four share one scale group.

## Form template identifiers
KV_Q8_GROUP -> quant_group, rowIdx -> cache_row, clamp127 -> int8_clamp
-/
namespace Strata.KvQ8Cache

-- § mixed-radix lemmas
/-- Row-major pair bound. -/
theorem pair_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : a + 1 ≤ A := ha
  have h2 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- Mixed-radix recovery by division. -/
theorem pair_div (a b B : Nat) (hB : 0 < B) (hb : b < B) : (a * B + b) / B = a := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hb, Nat.zero_add]

/-- Mixed-radix recovery by modulo. -/
theorem pair_mod (a b B : Nat) (hb : b < B) : (a * B + b) % B = b := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- Mixed-radix injectivity. -/
theorem pair_inj (a b a' b' B : Nat) (hB : 0 < B) (hb : b < B) (hb' : b' < B)
    (h : a * B + b = a' * B + b') : a = a' ∧ b = b' := by
  have h1 := congrArg (fun x => x / B) h
  have h2 := congrArg (fun x => x % B) h
  simp only [pair_div a b B hB hb, pair_div a' b' B hB hb', pair_mod a b B hb,
    pair_mod a' b' B hb'] at h1 h2
  exact ⟨h1, h2⟩

-- § constants and clamp
/-- Quantisation group size. -/
def KV_Q8_GROUP : Nat := 64

/-- Symmetric int8 clamp. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if 127 < q then 127 else q

/-- The clamp always lands in [-127, 127] (never -128). -/
theorem clamp127_bounds (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  split <;> try split
  all_goals omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  split <;> try split
  all_goals omega

/-- The clamp is idempotent. -/
theorem clamp127_idem (q : Int) : clamp127 (clamp127 q) = clamp127 q :=
  clamp127_id _ (clamp127_bounds q).1 (clamp127_bounds q).2

-- § cache row addressing
/-- Cache row of (page, head, in-page offset). -/
def rowIdx (kvh ps page h off : Nat) : Nat := (page * kvh + h) * ps + off

/-- Rows stay inside `P` pages. -/
theorem rowIdx_lt (kvh ps P page h off : Nat) (hp : page < P) (hh : h < kvh) (ho : off < ps) :
    rowIdx kvh ps page h off < P * kvh * ps := by
  unfold rowIdx
  exact pair_lt (page * kvh + h) off (P * kvh) ps (pair_lt page h P kvh hp hh) ho

/-- Distinct (page, head, offset) triples never share a cache row. -/
theorem rowIdx_inj (kvh ps page h off page' h' off' : Nat) (hps : 0 < ps) (hkvh : 0 < kvh)
    (hh : h < kvh) (hh' : h' < kvh) (ho : off < ps) (ho' : off' < ps)
    (e : rowIdx kvh ps page h off = rowIdx kvh ps page' h' off') :
    page = page' ∧ h = h' ∧ off = off' := by
  unfold rowIdx at e
  obtain ⟨e1, e2⟩ := pair_inj (page * kvh + h) off (page' * kvh + h') off' ps hps ho ho' e
  obtain ⟨e3, e4⟩ := pair_inj page h page' h' kvh hkvh hh hh' e1
  exact ⟨e3, e4, e2⟩

/-- In-page offset of a position is below page_size. -/
theorem offset_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

-- § group addressing
/-- A value's slot inside a row (group g, lane t) is below head_dim. -/
theorem value_slot_lt (hd g t : Nat) (hd64 : hd % 64 = 0) (hg : g < hd / 64) (ht : t < 64) :
    g * 64 + t < hd := by omega

/-- A thread's four values start at d = 4*q4 < head_dim. -/
theorem gather_d_lt (hd q4 : Nat) (hq : q4 < hd / 4) : q4 * 4 < hd := by omega

/-- The four values of a gather thread share one scale group (head_dim % 64 = 0 is not even needed). -/
theorem gather_same_group (q4 : Nat) : (q4 * 4) / 64 = (q4 * 4 + 3) / 64 := by omega

/-- The scale group of a gather thread is a valid group index. -/
theorem gather_group_lt (hd q4 : Nat) (hd64 : hd % 64 = 0) (hq : q4 < hd / 4) : (q4 * 4) / 64 < hd / 64 := by
  omega

/-- The 64 threads of a block cover one group: thread t handles value g*64+t. -/
theorem append_cover (hd v : Nat) (hd64 : hd % 64 = 0) (hv : v < hd) :
    ∃ g t, g < hd / 64 ∧ t < 64 ∧ g * 64 + t = v :=
  ⟨v / 64, v % 64, by omega, by omega, by omega⟩

/-- Gather decomposition: flat i = id*(kvh*per) + rem with rem < kvh*per. -/
theorem gather_decomp (i kvh per : Nat) : (kvh * per) * (i / (kvh * per)) + i % (kvh * per) = i :=
  Nat.div_add_mod i (kvh * per)

/-- The remainder splits into (head, q4) with head = rem/per, q4 = rem%per. -/
theorem gather_head_split (rem per : Nat) : per * (rem / per) + rem % per = rem := Nat.div_add_mod rem per

end Strata.KvQ8Cache
