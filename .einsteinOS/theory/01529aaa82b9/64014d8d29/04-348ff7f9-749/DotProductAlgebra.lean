/-!
# MODULE: dot_product_algebra
SOURCES: src/kernels/cuda/s2_expert_grouped.cu, src/kernels/cuda/s_gemv.cu, src/kernels/cuda/native_mmvq.cu
Exact-arithmetic identities behind the integer dot products and block reductions: chunked dot products,
the Q2_0 bias identity `sum (c-1) x = sum c x - sum x`, the 2-bit code expansion into dp4a bytes, and the
shuffle / shared-memory tree reductions preserving the total.

## Source semantics
* `s - hx` with `s = sum c*x` (dp4a with the expanded codes) and `hx = sum x` (dp4a with ones) equals the
  sum of `(c-1)*x`.
* A dot product over concatenated chunks equals the sum of the chunk dot products (dp4a accumulation).
* A tree reduction with halving strides (`step = 16, 8, 4, 2, 1`, or `tpr/2 ... 1`) conserves the sum.

## Form template identifiers
dp4a -> int8x4_dot_accumulate ; hx -> activation_sum ; warp_sum -> warp_tree_sum ; partial -> block_partials
-/

namespace Strata.DotAlgebra

-- § chunked dot products

/-- Integer dot product of two lists (truncating to the shorter, as zip does). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Dot product distributes over concatenation of equal-length prefixes. -/
theorem dot_append (as as' bs bs' : List Int) (h : as.length = bs.length) :
    dot (as ++ as') (bs ++ bs') = dot as bs + dot as' bs' := by
  induction as generalizing bs with
  | nil =>
    cases bs with
    | nil => simp [dot]
    | cons _ _ => simp at h
  | cons a as ih =>
    cases bs with
    | nil => simp at h
    | cons b bs =>
      have h' : as.length = bs.length := by simpa using h
      simp only [List.cons_append, dot]
      rw [ih bs h']
      omega

/-- The Q2_0 bias identity: shifting codes by one equals subtracting the activation sum. -/
theorem dot_bias (cs xs : List Int) (h : cs.length = xs.length) :
    dot (cs.map (· - 1)) xs = dot cs xs - xs.sum := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [dot]
    | cons _ _ => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      simp only [List.map_cons, dot, List.sum_cons]
      rw [ih xs h', Int.sub_mul, Int.one_mul]
      omega

-- § 2-bit code expansion for dp4a

/-- A code byte with four 2-bit fields is expanded to one byte per field. -/
def expand (c : Nat) : Nat :=
  c % 4 + 256 * ((c / 4) % 4) + 65536 * ((c / 16) % 4) + 16777216 * ((c / 64) % 4)

/-- Packing four fields below 4 into bytes of a word keeps them separate. -/
theorem pack4 (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    (a + 256 * b + 65536 * c + 16777216 * d) % 256 = a ∧
    ((a + 256 * b + 65536 * c + 16777216 * d) / 256) % 256 = b ∧
    ((a + 256 * b + 65536 * c + 16777216 * d) / 65536) % 256 = c ∧
    (a + 256 * b + 65536 * c + 16777216 * d) / 16777216 = d ∧
    a + 256 * b + 65536 * c + 16777216 * d < 4294967296 := by
  omega

/-- Each byte of the expanded word holds exactly one field (value <= 3, so no sign/carry issues). -/
theorem expand_fields (c : Nat) (h : c < 256) :
    expand c % 256 = c % 4 ∧ (expand c / 256) % 256 = (c / 4) % 4 ∧
    (expand c / 65536) % 256 = (c / 16) % 4 ∧ expand c / 16777216 = (c / 64) % 4 ∧
    expand c < 4294967296 := by
  unfold expand
  exact pack4 _ _ _ _ (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))
    (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))

/-- The chunk's weight scale index is `c / 2` and stays below the row's scale count. -/
theorem chunk_scale_index (c : Nat) (h : c < 80) : c / 2 < 40 := by omega

-- § tree reductions

/-- `sumTo f n = f 0 + ... + f (n-1)`. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

/-- A sum splits at any point. -/
theorem sumTo_split (f : Nat → Int) (a b : Nat) :
    sumTo f (a + b) = sumTo f a + sumTo (fun t => f (a + t)) b := by
  induction b with
  | zero => simp [sumTo]
  | succ b ih =>
    show sumTo f (a + b) + f (a + b) = sumTo f a + (sumTo (fun t => f (a + t)) b + f (a + b))
    rw [ih]
    omega

/-- A sum of pointwise sums is the sum of sums. -/
theorem sumTo_add (f g : Nat → Int) (n : Nat) :
    sumTo (fun t => f t + g t) n = sumTo f n + sumTo g n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    show sumTo (fun t => f t + g t) n + (f n + g n) = sumTo f n + f n + (sumTo g n + g n)
    rw [ih]
    omega

/-- One tree step: lane `t < s` adds lane `t + s`. -/
def rstep (p : Nat → Int) (s : Nat) : Nat → Int := fun t => p t + p (t + s)

/-- A halving step conserves the total. -/
theorem tree_step (p : Nat → Int) (s : Nat) : sumTo p (s + s) = sumTo (rstep p s) s := by
  have h1 : (fun t => p (s + t)) = (fun t => p (t + s)) := funext fun t => by rw [Nat.add_comm]
  rw [sumTo_split, h1]
  unfold rstep
  rw [sumTo_add]

/-- The 32-lane shuffle reduction (strides 16, 8, 4, 2, 1) leaves the exact total in lane 0. -/
theorem warp32 (p : Nat → Int) :
    sumTo p 32 = rstep (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 1 0 := by
  have h1 : sumTo p 32 = sumTo (rstep p 16) 16 := tree_step p 16
  have h2 : sumTo (rstep p 16) 16 = sumTo (rstep (rstep p 16) 8) 8 := tree_step _ 8
  have h3 : sumTo (rstep (rstep p 16) 8) 8 = sumTo (rstep (rstep (rstep p 16) 8) 4) 4 := tree_step _ 4
  have h4 : sumTo (rstep (rstep (rstep p 16) 8) 4) 4 =
      sumTo (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 2 := tree_step _ 2
  have h5 : sumTo (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 2 =
      sumTo (rstep (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 1) 1 := tree_step _ 1
  have h6 : sumTo (rstep (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 1) 1 =
      rstep (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 1 0 := by
    simp [sumTo]
  exact h1.trans (h2.trans (h3.trans (h4.trans (h5.trans h6))))

/-- The shared-memory block reduction (`step = tpr/2` downwards) uses the same step; a power-of-two
`tpr = 2 * s` therefore folds exactly. -/
theorem block_fold (p : Nat → Int) (s : Nat) : sumTo p (2 * s) = sumTo (rstep p s) s := by
  have : 2 * s = s + s := by omega
  rw [this]
  exact tree_step p s

end Strata.DotAlgebra
