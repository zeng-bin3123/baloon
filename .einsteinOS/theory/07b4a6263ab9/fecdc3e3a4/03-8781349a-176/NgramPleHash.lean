/-!
# MODULE: ngram_ple_hash
SOURCES: include/strata/kernels/ngram.hpp, src/kernels/ngram.cpp
The host-side PLE n-gram hash (sixteen row indices per token from the last three token ids), the
IQ4_NL row dequantisation with its split-half nibble order, and the head-slowest gather layout and
table geometry checks of `PleTable`.

## Source semantics
* `mixed_n = (ctx[0]*m[0]) ^ ... ^ (ctx[n-1]*m[n-1])` in wrapping uint64: XOR, not sum.
* Context: `ctx[0]` is the token; predecessor `s` back is `prev[n_prev - s]` (oldest-first input).
  A predecessor that is NULL (`< 0`) or EOS cuts, and the cut propagates to every OLDER position,
  which becomes EOS.  The token's own EOS does not cut.  Token id 0 is real.
* Heads `0..7` hash the bigram prefix, `8..15` the trigram; row = `mixed % vocab[h] + offset[h]`,
  where `offset` is the running sum of `vocab`, so each head owns a disjoint row range below
  `PLE_TABLE_ROWS`.
* IQ4_NL: codebook has no `-8` shift; byte `j` of a block carries elements `j` and `j + 16`.
* Gather: head `h`'s 160 values occupy `[h*160, (h+1)*160)` of the 2560-vector.
* Table check: alone in its shard the table must fill the data section exactly, else fit in it.

## Form template identifiers
`ngram_mixed` → `mixed`; context construction in `ngram_rows` → `ctx1`/`ctx2`; row index →
`rowIdx`; `ple_artifact_consts` → `vocab`/`offset`; `kIq4Nl` → `cb`; dequant index → `elemIdx`;
gather index → `gatherIdx`; table size check → `tableOk`.
-/

namespace Strata.NgramPleHash

-- § geometry constants

/-- `PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM`. -/
theorem n_heads : (3 - 1) * 8 = 16 := rfl

/-- Sixteen 160-wide heads flatten to exactly `n_embd`. -/
theorem heads_fill_embd : 16 * 160 = 2560 := rfl

/-- One IQ4_NL row is 5 blocks of 18 bytes. -/
theorem row_bytes : 160 / 32 * 18 = 90 := rfl

/-- `NG_HIST = (PLE_CONV_KERNEL - 1) * NGRAM_SIZE`. -/
theorem hist_len : (4 - 1) * 3 = 9 := rfl

/-- `NG_HC_DIM = NG_HC * n_embd`. -/
theorem hc_dim : 4 * 2560 = 10240 := rfl

-- § the XOR mix

/-- The 64-bit wrap. -/
def W : Nat := 2 ^ 64

/-- `ngram_mixed` over the first terms: assignment of the first product, XOR of the rest. -/
def mixed : List Nat → List Nat → Nat
  | c :: cs, m :: ms => cs.zip ms |>.foldl (fun acc p => acc ^^^ (p.1 * p.2 % W)) (c * m % W)
  | _, _ => 0

/-- The bigram mix is the XOR of the two wrapped products. -/
theorem mixed_two (a b m0 m1 : Nat) : mixed [a, b] [m0, m1] = (a * m0 % W) ^^^ (b * m1 % W) := rfl

/-- XOR, not sum: two equal products cancel to zero. -/
theorem equal_terms_cancel (x m : Nat) : mixed [x, x] [m, m] = 0 := by
  rw [mixed_two]; exact Nat.xor_self _

/-- A sum would NOT cancel: the rival reading gives a different index (concrete witness). -/
theorem sum_differs : (1 * 3 % W) ^^^ (1 * 3 % W) ≠ (1 * 3 % W) + (1 * 3 % W) := by decide

-- § context construction with the EOS cut

/-- `PLE_EOS_TOKEN_ID`. -/
def EOS : Int := 248044

/-- Does a predecessor cut the context?  NULL is `< 0`; EOS cuts too. -/
def cuts (t : Int) : Bool := decide (t < 0) || decide (t = EOS)

/-- `ctx[1]`: the NEWEST predecessor `p1`, or EOS if it cuts. -/
def ctx1 (p1 : Int) : Int := if cuts p1 then EOS else p1

/-- `ctx[2]`: the OLDEST predecessor `p0`, EOS if it or anything newer cut. -/
def ctx2 (p0 p1 : Int) : Int := if cuts p1 || cuts p0 then EOS else p0

/-- The cut propagates forward: a cut at the newest predecessor makes the older position EOS. -/
theorem cut_propagates (p0 p1 : Int) (h : cuts p1 = true) : ctx1 p1 = EOS ∧ ctx2 p0 p1 = EOS := by
  simp [ctx1, ctx2, h]

/-- A missing predecessor (`TOKEN_NULL = -1`) is replaced by EOS. -/
theorem null_is_eos (p0 : Int) : ctx1 (-1) = EOS ∧ ctx2 p0 (-1) = EOS := by
  simp [ctx1, ctx2, cuts]

/-- Token id 0 is a real token, not a cut. -/
theorem zero_is_real (p0 : Int) (h : cuts p0 = false) : ctx1 0 = 0 ∧ ctx2 p0 0 = p0 := by
  have h0 : cuts 0 = false := by decide
  simp [ctx1, ctx2, h0, h]

/-- An older cut does not reach the newer position. -/
theorem old_cut_local (p1 : Int) (hp : cuts p1 = false) : ctx1 p1 = p1 ∧ ctx2 (-1) p1 = EOS := by
  have hn : cuts (-1) = true := by decide
  exact ⟨by simp [ctx1, hp], by simp [ctx2, hn]⟩

/-- The token's own value never cuts its context (only predecessors are examined). -/
theorem own_eos_no_cut (p0 p1 : Int) (h1 : cuts p1 = false) (h0 : cuts p0 = false) :
    [EOS, ctx1 p1, ctx2 p0 p1] = [EOS, p1, p0] := by
  simp [ctx1, ctx2, h1, h0]

-- § row indices and head ranges

/-- The artifact's sixteen head vocab sizes. -/
def vocab : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- The artifact's sixteen head offsets, transcribed separately rather than derived. -/
def offset : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Running prefix sums. -/
def prefixSums : Nat → List Nat → List Nat
  | _, [] => []
  | acc, v :: vs => acc :: prefixSums (acc + v) vs

/-- The transcribed offsets ARE the running sum of the vocab sizes (the parity check's content). -/
theorem offsets_running_sum : prefixSums 0 vocab = offset := by decide

/-- The vocab sizes are not the arithmetic sequence `20000003 + 10 i` of the reference selftest. -/
theorem vocab_not_arithmetic : vocab ≠ (List.range 16).map (fun i => 20000003 + 10 * i) := by decide

/-- The last head ends inside the 320,001,536-row table. -/
theorem heads_fit_table : 300001275 + 20000171 ≤ 320001536 := by decide

/-- Row index for head `h`. -/
def rowIdx (mix v o : Nat) : Nat := mix % v + o

/-- A head's row lies in its own range `[offset, offset + vocab)`. -/
theorem rowIdx_in_head (mix v o : Nat) (hv : 0 < v) : o ≤ rowIdx mix v o ∧ rowIdx mix v o < o + v := by
  have := Nat.mod_lt mix hv
  unfold rowIdx; omega

/-- Every head's range ends where the next begins, so heads never collide. -/
theorem head_ranges_contiguous :
    ∀ h : Fin 15, offset[h.val]! + vocab[h.val]! = offset[h.val + 1]! := by decide

/-- Every row index of every head is inside the table. -/
theorem rowIdx_lt_table (mix h : Nat) (hh : h < 16) : rowIdx mix vocab[h]! offset[h]! < 320001536 := by
  have hv : ∀ h : Fin 16, 0 < vocab[h.val]! ∧ offset[h.val]! + vocab[h.val]! ≤ 320001536 := by decide
  have h1 : 0 < vocab[h]! := (hv ⟨h, hh⟩).1
  have h2 : offset[h]! + vocab[h]! ≤ 320001536 := (hv ⟨h, hh⟩).2
  have := (rowIdx_in_head mix vocab[h]! offset[h]! h1).2
  unfold rowIdx at this ⊢
  omega

-- § IQ4_NL

/-- `kvalues_iq4nl`. -/
def cb : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook is strictly increasing. -/
theorem cb_sorted : ∀ i : Fin 15, cb[i.val]! < cb[i.val + 1]! := by decide

/-- No `-8` shift in this ggml version: code 8 decodes to 1, not -7. -/
theorem cb_no_shift : cb[8]! = 1 := by decide

/-- Element index of nibble `hi` (0 low, 1 high) of byte `j` in block `b`: split halves. -/
def elemIdx (b j hi : Nat) : Nat := b * 32 + j + 16 * hi

/-- Every decoded element lands inside the 160-wide row. -/
theorem elemIdx_lt (b j hi : Nat) (hb : b < 5) (hj : j < 16) (hh : hi < 2) : elemIdx b j hi < 160 := by
  unfold elemIdx; omega

/-- The split-half layout is a bijection: distinct nibbles go to distinct elements. -/
theorem elemIdx_inj (b j hi b' j' hi' : Nat) (hj : j < 16) (hj' : j' < 16) (hh : hi < 2) (hh' : hi' < 2)
    (h : elemIdx b j hi = elemIdx b' j' hi') : b = b' ∧ j = j' ∧ hi = hi' := by
  unfold elemIdx at h; omega

/-- Byte `j`'s two nibbles are elements `j` and `j+16`, NOT `2j` and `2j+1`. -/
theorem split_not_interleaved : elemIdx 0 1 0 = 1 ∧ elemIdx 0 1 1 = 17 ∧ elemIdx 0 1 1 ≠ 2 * 1 + 1 := by decide

-- § gather layout and table check

/-- Head-slowest index of value `d` of head `h`. -/
def gatherIdx (h d : Nat) : Nat := h * 160 + d

/-- The gather writes exactly the 2560-vector, each slot once. -/
theorem gatherIdx_lt (h d : Nat) (hh : h < 16) (hd : d < 160) : gatherIdx h d < 2560 := by
  unfold gatherIdx; omega

/-- Distinct (head, value) pairs never share a slot. -/
theorem gatherIdx_inj (h d h' d' : Nat) (hd : d < 160) (hd' : d' < 160)
    (e : gatherIdx h d = gatherIdx h' d') : h = h' ∧ d = d' := by
  unfold gatherIdx at e; omega

/-- `PleTable::open`'s size check: exact fill when alone, otherwise fit at its offset. -/
def tableOk (alone : Bool) (nRows tOff have_ : Nat) : Bool :=
  if alone then nRows * 90 == have_ else decide (tOff + nRows * 90 ≤ have_)

/-- A table alone in its shard is accepted only if it fills the data section exactly. -/
theorem tableOk_alone (n t hv : Nat) (h : tableOk true n t hv = true) : n * 90 = hv := by
  simpa [tableOk] using h

/-- The original shard: 320,001,536 rows fill exactly 28,800,138,240 bytes after the header. -/
theorem original_shard : tableOk true 320001536 0 28800138240 = true := by decide

end Strata.NgramPleHash
