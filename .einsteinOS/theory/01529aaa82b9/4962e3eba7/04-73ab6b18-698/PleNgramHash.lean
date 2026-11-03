/-!
# MODULE: ple_ngram_hash
PLE n-gram hash row indices and the per-layer token table: XOR mixing, modulus and offsets, EOS cut,
null predecessor test, IQ4_NL row layout, table size accounting.

## Source semantics
`mixed_n = (ctx[0]*m[0]) ^ ... ^ (ctx[n-1]*m[n-1])` in uint64, row = (mixed % vocab_h) + offset_h.
Vocab sizes are not powers of two, so `%` cannot be a mask. A missing predecessor is TOKEN_NULL = -1
(tested `t < 0`; token 0 is real). An EOS predecessor cuts itself and every older position; the
token's own EOS cuts nothing. A 160-wide IQ4_NL row is 5 blocks of 18 bytes; 16 rows flatten
head-slowest to n_embd = 2560.

## Form template identifiers
mixed -> ngram_mixed ; rowOf -> row index ; cut -> eos_cut ; isNull -> t < 0
-/

set_option maxRecDepth 100000

namespace Strata.Ple

-- § Geometry

/-- Heads: (ngram - 1) * heads_per_ngram = 16. -/
theorem n_heads : (3 - 1) * 8 = 16 := rfl

/-- 16 heads x 160 = n_embd 2560. -/
theorem flatten_width : 16 * 160 = 2560 := rfl

/-- History length (conv_kernel - 1) * ngram = 9; hyper-connection width 4 * 2560 = 10240. -/
theorem hist_hc : (4 - 1) * 3 = 9 ∧ 4 * 2560 = 10240 := ⟨rfl, rfl⟩

/-- Head-slowest flatten: element i of head h lands at h*160 + i, in range and invertible. -/
theorem flatten_index (h i : Nat) (hh : h < 16) (hi : i < 160) :
    h * 160 + i < 2560 ∧ (h * 160 + i) / 160 = h ∧ (h * 160 + i) % 160 = i := by omega

/-- A row is 160 / 32 = 5 blocks of 18 bytes = 90 bytes. -/
theorem row_bytes : 160 / 32 * 18 = 90 := rfl

/-- Table: 320,001,536 rows x 160 = 51.2e9 elements; x 90 B = 28.8e9 bytes. -/
theorem table_size : 320001536 * 160 = 51200245760 ∧ 320001536 * 90 = 28800138240 := by decide

/-- Row cache of 2^20 rows x 90 B = 94,371,840 B (about 95 MB). -/
theorem cache_bytes : (1 <<< 20) * 90 = 94371840 := by decide

-- § Hash

/-- 64-bit product, modelled as a Nat modulo 2^64. -/
def mul64 (x m : Nat) : Nat := (x * m) % 18446744073709551616

/-- XOR-fold of the products. -/
def mixed : List Nat → List Nat → Nat
  | c :: cs, m :: ms => mul64 c m ^^^ mixed cs ms
  | _, _ => 0

/-- Two equal terms cancel under XOR. -/
theorem xor_cancel (x : Nat) : x ^^^ x = 0 := Nat.xor_self x

/-- Under sum the same two terms would double instead: a different function. -/
theorem sum_doubles (x : Nat) (h : 0 < x) : x + x ≠ 0 := by omega

/-- XOR cancellation, concretely: equal tokens with equal multipliers contribute nothing. -/
theorem mixed_equal_pair (c m : Nat) : mixed [c, c] [m, m] = 0 := by
  simp [mixed, Nat.xor_self]

/-- Products stay below 2^64. -/
theorem mul64_lt (x m : Nat) : mul64 x m < 18446744073709551616 := by
  unfold mul64
  exact Nat.mod_lt _ (by decide)

/-- Row index of head h: `(mixed % vocab) + offset`. -/
def rowOf (mix vocab off : Nat) : Nat := mix % vocab + off

/-- A row lies inside its head's slice [off, off + vocab). -/
theorem row_in_slice (mix vocab off : Nat) (hv : 0 < vocab) :
    off ≤ rowOf mix vocab off ∧ rowOf mix vocab off < off + vocab := by
  have := Nat.mod_lt mix hv
  unfold rowOf; omega

/-- Slices are disjoint when offsets are the running sum of vocab sizes. -/
theorem slices_disjoint (mix vocab off nextOff : Nat) (hv : 0 < vocab)
    (hn : off + vocab ≤ nextOff) : rowOf mix vocab off < nextOff := by
  have := (row_in_slice mix vocab off hv).2
  omega

/-- Artifact constants: first offsets are 0, 20000003, 40000026 = 20000003 + 20000023. -/
theorem offsets_running_sum : 0 + 20000003 = 20000003 ∧ 20000003 + 20000023 = 40000026 := ⟨rfl, rfl⟩

/-- The vocab sizes are not an arithmetic sequence of step 10: differences are 20, then 10. -/
theorem vocab_not_arith : 20000023 - 20000003 = 20 ∧ 20000033 - 20000023 = 10 := ⟨rfl, rfl⟩

/-- A mask would be a different function: 20000003 is not a power of two. -/
theorem vocab_not_pow2 : (20000003 &&& (20000003 - 1)) ≠ 0 := by decide

-- § Null predecessor and EOS cut

/-- TOKEN_NULL test: negative ids are null; id 0 is a real token. -/
def isNull (t : Int) : Bool := decide (t < 0)

/-- Token id 0 is not null; -1 is. -/
theorem null_test : isNull 0 = false ∧ isNull (-1) = true := by decide

/-- EOS cut over the two predecessors (oldest p0, newest p1): an EOS replaces itself and all older
positions by TOKEN_NULL. The current token is not an input. -/
def cut (eos p0 p1 : Int) : Int × Int :=
  if p1 = eos then (-1, -1) else if p0 = eos then (-1, p1) else (p0, p1)

/-- An EOS at the newest predecessor nulls both. -/
theorem cut_newest (eos p0 : Int) : cut eos p0 eos = (-1, -1) := by
  unfold cut; simp

/-- An EOS at the oldest predecessor nulls only itself. -/
theorem cut_oldest (eos p1 : Int) (h : p1 ≠ eos) : cut eos eos p1 = (-1, p1) := by
  unfold cut; simp [h]

/-- No EOS: nothing changes. -/
theorem cut_none (eos p0 p1 : Int) (h0 : p0 ≠ eos) (h1 : p1 ≠ eos) : cut eos p0 p1 = (p0, p1) := by
  unfold cut; simp [h0, h1]

/-- Idempotent when eos is non-negative: applying the cut twice equals once. -/
theorem cut_idem (eos p0 p1 : Int) (he : 0 ≤ eos) :
    cut eos (cut eos p0 p1).1 (cut eos p0 p1).2 = cut eos p0 p1 := by
  unfold cut
  split <;> (try split) <;> simp <;> omega

-- § IQ4_NL row decode layout

/-- Split-half order: byte j carries elements j and j + 16, covering 0..31 exactly once. -/
theorem split_half_cover (i : Nat) (hi : i < 32) : ∃ j, j < 16 ∧ (i = j ∨ i = j + 16) :=
  ⟨i % 16, by omega, by omega⟩

/-- The interleaved reading (2j, 2j+1) differs: byte 1 would carry 2 and 3, not 1 and 17. -/
theorem not_interleaved : (1 : Nat) ≠ 2 * 1 ∧ (17 : Nat) ≠ 2 * 1 + 1 := by decide

end Strata.Ple
