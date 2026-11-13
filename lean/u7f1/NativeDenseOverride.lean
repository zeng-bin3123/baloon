/-!
# MODULE: native_dense_override
SOURCES: src/core/native_dense.cpp
Attaching native (GGUF block-quantized) copies of the GDN/QSA dense matrices to the canonical
weight table: tensor-name eligibility, the GGUF directory span checks (no truncation, no overlap,
no byte-count overflow), split-shard metadata acceptance, scratch sizing, and all-or-nothing
publication of the overrides.

## Source semantics
* Eligible: a `blk.` tensor whose name ends with one of ten attention/SSM/shared-expert suffixes,
  or `blk.1.ple_key.weight` when requested.
* Every directory entry (uploaded or not) must have block-divisible first dimension, a payload
  span inside the file, and must end before the next tensor's offset.
* Overflow guards: `x <= MAX / d` implies `x * d <= MAX`.
* An additional shard is accepted only with the first shard's split count/tensor count and a
  shard number in `1 .. count-1`.
* The scratch is sized by the largest input width, which bounds every override's input.
* All checks and uploads finish before any reference is published; any failure leaves the table
  untouched.

## Form template identifiers
`eligible` → `eligible`; payload checks → `spanOk`/`nextOk`; overflow guard → `mul_le_of_le_div`;
split metadata check → `shardOk`; `max_in` → `maxIn`; `pending` + publish loop → `checkAll`/`publish`.
-/

namespace Ref.NativeDenseOverride

-- § Eligibility

/-- The ten matrix suffixes served natively. -/
def suffixes : List String :=
  [".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight", ".attn_q.weight", ".attn_k.weight",
   ".attn_v.weight", ".attn_output.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight",
   ".ffn_down_shexp.weight"]

/-- `eligible(name, include_ple_key)`. -/
def eligible (name : String) (pleKey : Bool) : Bool :=
  name.startsWith "blk." &&
    ((pleKey && name == "blk.1.ple_key.weight") || suffixes.any (fun s => name.endsWith s))

/-- Only per-block tensors are ever eligible. -/
theorem eligible_blk {name : String} {pk : Bool} (h : eligible name pk = true) :
    name.startsWith "blk." = true := by
  simp only [eligible, Bool.and_eq_true] at h
  exact h.1

/-- Without the PLE key flag, eligibility is exactly the suffix test. -/
theorem eligible_no_ple (name : String) :
    eligible name false = (name.startsWith "blk." && suffixes.any (fun s => name.endsWith s)) := by
  simp [eligible]

/-- Ten suffixes are served. -/
theorem ten_suffixes : suffixes.length = 10 := rfl

-- § Directory span checks

/-- The tensor's payload lies inside the file's data section (no overflowing addition). -/
def spanOk (payload off bytes : Nat) : Bool := decide (off ≤ payload) && decide (bytes ≤ payload - off)

/-- An accepted span ends inside the payload. -/
theorem spanOk_end {p o b : Nat} (h : spanOk p o b = true) : o + b ≤ p := by
  simp only [spanOk, Bool.and_eq_true, decide_eq_true_eq] at h
  omega

/-- The tensor ends before the next tensor's offset. -/
def nextOk (off bytes next : Nat) : Bool := decide (bytes ≤ next - off)

/-- An accepted entry ends at or before its successor starts. -/
theorem nextOk_end {o b n : Nat} (hlt : o < n) (h : nextOk o b n = true) : o + b ≤ n := by
  simp only [nextOk, decide_eq_true_eq] at h
  omega

/-- Two checked tensors never overlap: the earlier one ends before its successor, which is at
or before the later one. -/
theorem spans_disjoint {o1 b1 nx o2 : Nat} (h1 : o1 < nx) (hnx : nx ≤ o2)
    (h : nextOk o1 b1 nx = true) : o1 + b1 ≤ o2 := by
  have := nextOk_end h1 h
  omega

/-- The overflow guard: `x ≤ MAX / d` implies `x * d ≤ MAX`. -/
theorem mul_le_of_le_div {x d M : Nat} (hd : 0 < d) (h : x ≤ M / d) : x * d ≤ M :=
  (Nat.le_div_iff_mul_le hd).1 h

/-- Block divisibility of the first dimension makes the whole element count block divisible. -/
theorem elements_block_divisible {d0 rest be : Nat} (h : d0 % be = 0) : (d0 * rest) % be = 0 :=
  Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_of_mod_eq_zero h) (Nat.dvd_mul_right _ _))

-- § Split shards

/-- An additional shard's split metadata matches the first shard. -/
def shardOk (count no tensors firstCount firstTensors : Nat) : Bool :=
  (firstCount != 0) && (count == firstCount) && (no != 0) && decide (no < firstCount) &&
    (tensors == firstTensors)

/-- An accepted additional shard has a number strictly between 0 and the split count. -/
theorem shardOk_number {c n t fc ft : Nat} (h : shardOk c n t fc ft = true) : 0 < n ∧ n < fc := by
  simp only [shardOk, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq] at h
  omega

/-- The first shard (number 0) can never pass as an additional shard. -/
theorem first_not_additional (c t fc ft : Nat) : shardOk c 0 t fc ft = false := by
  simp [shardOk]

-- § Scratch sizing

/-- `max_in`: the widest input of any override. -/
def maxIn (widths : List Nat) : Nat := widths.foldl max 0

/-- Helper: the running maximum dominates its accumulator. -/
theorem foldl_max_ge_acc : ∀ (l : List Nat) (acc : Nat), acc ≤ l.foldl max acc
  | [], acc => Nat.le_refl _
  | a :: t, acc => by
    simp only [List.foldl_cons]
    have := foldl_max_ge_acc t (max acc a)
    omega

/-- Helper: the running maximum dominates every element. -/
theorem foldl_max_ge_mem : ∀ (l : List Nat) (acc x : Nat), x ∈ l → x ≤ l.foldl max acc
  | [], _, _, h => by cases h
  | a :: t, acc, x, h => by
    simp only [List.foldl_cons]
    rcases List.mem_cons.1 h with rfl | h
    · have := foldl_max_ge_acc t (max acc x)
      omega
    · exact foldl_max_ge_mem t _ x h

/-- The shared Q8_1 scratch is wide enough for every override's input. -/
theorem scratch_wide_enough {widths : List Nat} {w : Nat} (h : w ∈ widths) : w ≤ maxIn widths :=
  foldl_max_ge_mem widths 0 w h

-- § All-or-nothing publication

/-- Check and upload every eligible tensor; the first failure aborts. -/
def checkAll {α ε β : Type} (check : α → Except ε β) : List α → Except ε (List β)
  | [] => .ok []
  | a :: t =>
    match check a with
    | .error e => .error e
    | .ok b =>
      match checkAll check t with
      | .error e => .error e
      | .ok bs => .ok (b :: bs)

/-- Publish only a fully successful batch. -/
def publish {τ ε β : Type} (tbl : τ) (r : Except ε (List β)) (apply : τ → List β → τ) : τ :=
  match r with
  | .ok xs => apply tbl xs
  | .error _ => tbl

/-- One failing tensor fails the whole batch. -/
theorem checkAll_fails {α ε β : Type} (check : α → Except ε β) :
    ∀ (items : List α) (a : α) (e0 : ε), a ∈ items → check a = .error e0 →
      ∃ e, checkAll check items = .error e
  | [], _, _, h, _ => by cases h
  | x :: t, a, e0, hm, hc => by
    simp only [checkAll]
    split
    · rename_i e _; exact ⟨e, rfl⟩
    · rename_i b hb
      rcases List.mem_cons.1 hm with rfl | hm
      · rw [hc] at hb; cases hb
      · obtain ⟨e, he⟩ := checkAll_fails check t a e0 hm hc
        rw [he]
        exact ⟨e, rfl⟩

/-- A successful batch has one override per eligible tensor. -/
theorem checkAll_length {α ε β : Type} (check : α → Except ε β) :
    ∀ (items : List α) (bs : List β), checkAll check items = .ok bs → bs.length = items.length
  | [], bs, h => by simp only [checkAll, Except.ok.injEq] at h; subst h; rfl
  | x :: t, bs, h => by
    simp only [checkAll] at h
    split at h
    · cases h
    · split at h
      · cases h
      · rename_i b _ bs' hbs
        simp only [Except.ok.injEq] at h
        subst h
        simp [checkAll_length check t bs' hbs]

/-- A failed load leaves the canonical table exactly as it was. -/
theorem failed_load_untouched {τ ε β : Type} (tbl : τ) (e : ε) (apply : τ → List β → τ) :
    publish tbl (Except.error e : Except ε (List β)) apply = tbl := rfl

end Ref.NativeDenseOverride



