/-!
# MODULE: cpu_expert_kernel
Arithmetic contracts of the AVX-512 Q2_0 expert kernel: the code-offset identity that removes the `-1` bias,
the two-chunk split of a weight block, batching of eight blocks per scale vector, multi-token dispatch, and the
VPMULTISHIFTQB 2-bit unpack.

## Source semantics
* `sum_j (c_j - 1) d_w xhat_j d_x = d_w (d_x sum(c xhat) - d_x sum(xhat))`: the dot is done on raw codes 0..3
  and the weight-independent correction `hx = scale * sum` is subtracted once.
* A 64-weight block is two 32-element activation chunks, so a dot over the concatenation is the sum of the
  per-chunk dots.
* Scale vectors are built for `nb = min 8 (nblocks - b0)` blocks at a time, with a 2*nb-lane mask.
* The multi-token entry points handle `1..8` tokens by template; any other count falls back to one-by-one.
* Unpack: output byte `i` of a qword lane is `(u16 >> 2i) & 0xFF`, and `& 3` yields the 2-bit code.

## Form template identifiers
row_dot → quantized_row_dot, ActQ → quantized_activation, act_quant_q8_1 → activation_quantizer,
unpack_q2_0/unpack64_q2_0 → code_unpack, scales8 → batched_scale_vector, MAXT → max_tokens_per_expert,
s2_expert_vnni_multi → multi_token_expert, expert_oracle_q8_0 → oracle_contract_path.
-/

namespace Strata.Cpu.ExpertKernel

-- § the code-offset identity

/-- Integer dot product of two lists (extra elements of the longer list are ignored). -/
def dot : List Int → List Int → Int
  | c :: cs, x :: xs => c * x + dot cs xs
  | _, _ => 0

/-- Sum of a list of integers. -/
def sumL : List Int → Int
  | [] => 0
  | x :: xs => x + sumL xs

/-- Subtract the code offset 1 from every code. -/
def offs : List Int → List Int
  | [] => []
  | c :: cs => (c - 1) :: offs cs

/-- Dot with offset-removed codes equals the raw-code dot minus the activation sum. -/
theorem dot_offs (cs xs : List Int) (h : cs.length = xs.length) :
    dot (offs cs) xs = dot cs xs - sumL xs := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [dot, offs, sumL]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have key := ih xs h'
      have e : (c - 1) * x = c * x - x := by rw [Int.sub_mul, Int.one_mul]
      simp only [dot, offs, sumL]
      rw [key]
      omega

/-- A dot over a concatenation is the sum of the dots (the first chunk's lengths must match). -/
theorem dot_append (a b x y : List Int) (h : a.length = x.length) :
    dot (a ++ b) (x ++ y) = dot a x + dot b y := by
  induction a generalizing x with
  | nil =>
    cases x with
    | nil => simp [dot]
    | cons u us => simp at h
  | cons c cs ih =>
    cases x with
    | nil => simp at h
    | cons u us =>
      have h' : cs.length = us.length := by simpa using h
      simp only [List.cons_append, dot]
      rw [ih us h']
      omega

/-- The sum of a concatenation is the sum of the sums (used for `hx` per chunk, then per block). -/
theorem sumL_append (a b : List Int) : sumL (a ++ b) = sumL a + sumL b := by
  induction a with
  | nil => simp [sumL]
  | cons c cs ih => simp only [List.cons_append, sumL]; rw [ih]; omega

/-- The two-chunk form of the identity: block dot minus block correction, chunk by chunk. -/
theorem block_two_chunks (c1 c2 x1 x2 : List Int) (h1 : c1.length = x1.length) (h2 : c2.length = x2.length) :
    dot (offs (c1 ++ c2)) (x1 ++ x2) =
      (dot c1 x1 - sumL x1) + (dot c2 x2 - sumL x2) := by
  have hl : (c1 ++ c2).length = (x1 ++ x2).length := by simp [h1, h2]
  rw [dot_offs _ _ hl, dot_append _ _ _ _ h1, sumL_append]
  omega

-- § batching eight blocks per scale vector

/-- Number of blocks processed in the batch starting at `b0`. -/
def batchSize (nblocks b0 : Nat) : Nat := if nblocks - b0 < 8 then nblocks - b0 else 8

/-- Every batch starting before `nblocks` is non-empty, at most 8 wide, and stays in range. -/
theorem batch_in_range (nblocks b0 : Nat) (h : b0 < nblocks) :
    0 < batchSize nblocks b0 ∧ batchSize nblocks b0 ≤ 8 ∧ b0 + batchSize nblocks b0 ≤ nblocks := by
  unfold batchSize
  split <;> omega

/-- A batch is either the final one (reaches `nblocks`) or a full 8-block one. -/
theorem batch_full_or_last (nblocks b0 : Nat) (h : b0 < nblocks) :
    b0 + batchSize nblocks b0 = nblocks ∨ batchSize nblocks b0 = 8 := by
  unfold batchSize
  split <;> omega

/-- Number of batches needed for `n` blocks. -/
theorem batch_count (n : Nat) : (n + 7) / 8 * 8 ≥ n := by omega

/-- Lane `l` (0..15) of the permute index at block offset `i` selects `l/8 + 2i`, which stays in 0..15. -/
theorem permute_index_in_range (i l : Nat) (hi : i < 8) (hl : l < 16) : l / 8 + 2 * i < 16 := by omega

/-- Global chunk `2(b0+i)+h` is local lane `2i+h` of the batch whose base chunk is `2*b0`. -/
theorem chunk_local_lane (b0 i h : Nat) : 2 * (b0 + i) + h = 2 * b0 + (2 * i + h) := by omega

/-- A full batch uses all 16 mask lanes: `2^(2*8) - 1 = 0xFFFF`. -/
theorem full_mask_is_ffff : 2 ^ (2 * 8) - 1 = 65535 := by decide

/-- A partial batch of `nb < 8` blocks masks in exactly `2*nb` low lanes (value `2^(2nb) - 1 < 2^16`). -/
theorem partial_mask_lt (nb : Nat) (h : nb < 8) : 2 ^ (2 * nb) - 1 < 65536 := by
  have : 2 ^ (2 * nb) ≤ 2 ^ 14 := Nat.pow_le_pow_right (by decide) (by omega)
  omega

-- § multi-token dispatch

/-- Maximum tokens per expert. -/
def MAXT : Nat := 8

/-- The template instantiation chosen by the `switch`: 1..7 are exact, everything else becomes 8. -/
def effNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- For every legal token count the template width equals the token count. -/
theorem effNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ MAXT) : effNT n = n := by
  unfold effNT MAXT at *
  split <;> omega

/-- The selected width never exceeds `MAXT` (so the stack arrays `acc[NT]` fit the buffers). -/
theorem effNT_le_maxt (n : Nat) : effNT n ≤ MAXT := by
  unfold effNT MAXT
  split <;> omega

/-- `n = 0` silently selects the width-8 template: callers must guard the lower bound (the source does, in
`s2_expert_vnni_multi`, only through its fallback test). -/
theorem effNT_zero : effNT 0 = 8 := by decide

/-- The multi-token path is taken exactly when the oracle is off and `1 ≤ n ≤ MAXT`. -/
def takesMultiPath (oracle : Bool) (n : Nat) : Bool := !oracle && decide (1 ≤ n) && decide (n ≤ MAXT)

/-- On the multi-token path the dispatched width is exactly the token count. -/
theorem multi_path_exact (oracle : Bool) (n : Nat) (h : takesMultiPath oracle n = true) : effNT n = n := by
  unfold takesMultiPath at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at h
  exact effNT_exact n h.1.2 h.2

/-- With the oracle on, the multi-token path is never taken. -/
theorem oracle_disables_multi (n : Nat) : takesMultiPath true n = false := by
  simp [takesMultiPath]

-- § VPMULTISHIFTQB unpack (control, data) order

/-- 2-bit code `i` of a 16-bit word, as `(w >> 2i) & 3`. -/
def code2 (w i : Nat) : Nat := (w / 4 ^ i) % 4

/-- Within a u16 made of two bytes, codes 0..3 come from the low byte, in order. -/
theorem code2_low_byte (b0 b1 i : Nat) (hb : b0 < 256) (hi : i < 4) :
    code2 (b0 + 256 * b1) i = (b0 / 4 ^ i) % 4 := by
  unfold code2
  have : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by omega
  rcases this with rfl | rfl | rfl | rfl <;> simp <;> omega

/-- Codes 4..7 come from the high byte, in order. -/
theorem code2_high_byte (b0 b1 i : Nat) (hb : b0 < 256) (hi : 4 ≤ i) (hi2 : i < 8) :
    code2 (b0 + 256 * b1) i = (b1 / 4 ^ (i - 4)) % 4 := by
  unfold code2
  have : i = 4 ∨ i = 5 ∨ i = 6 ∨ i = 7 := by omega
  rcases this with rfl | rfl | rfl | rfl <;> simp <;> omega

/-- A 2-bit code lies in 0..3, so `c - 1` lies in {-1,0,1,2}. -/
theorem code2_lt_four (w i : Nat) : code2 w i < 4 := by
  unfold code2
  exact Nat.mod_lt _ (by decide)

-- § oracle-path reduction

/-- The integer reduction tree of `chunk_dot_oracle` (two horizontal adds) sums all eight lanes. -/
theorem hadd_tree_sums_lanes (l0 l1 l2 l3 l4 l5 l6 l7 : Int) :
    let h0 := l0 + l4
    let h1 := l1 + l5
    let h2 := l2 + l6
    let h3 := l3 + l7
    ((h0 + h1) + (h2 + h3)) = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 := by
  intro h0 h1 h2 h3
  omega

end Strata.Cpu.ExpertKernel
