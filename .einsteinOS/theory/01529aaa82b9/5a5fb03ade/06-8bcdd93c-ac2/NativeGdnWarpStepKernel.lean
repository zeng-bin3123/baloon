/-!
# MODULE: native_gdn_warp_step_kernel
Warp-per-column delta-net step (`native_gdn_step`) and its argument validation (aligned, disjoint spans).

## Source semantics
Each warp owns one `(head, col)` state column; lane `l` holds rows `r*32 + l` for `r = 0..3` in registers.
`kv_col` and `attn_col` are `warp_sum` over the 32 lanes of the per-lane 4-row partials, so they equal the
sequential 128-row contraction.  Only lane 0 writes `output[head*S + col]`; every lane writes back its own
four state rows.  The host wrapper requires a stream, `S = 128`, `h_v % h_k = 0`, `h_v <= 65535`, and all
spans aligned, non-wrapping and pairwise disjoint from state/output.

## Form template identifiers
- `state`, `output`  -> `stateIdx`, `outIdx`
- `valid_span`, `overlap` -> `validSpan`, `overlap`
- `warp_sum`          -> `sumRange` over lanes
-/

namespace Strata.Kernels.NativeGdn

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Row-major addresses are unique. -/
theorem split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
    (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
  have h1 : (a * n + c) / n = a := div_of_split_aux hc
  have h2 : (a' * n + c') / n = a' := div_of_split_aux hc'
  have ha : a = a' := by rw [← h1, ← h2, h]
  subst ha
  exact ⟨rfl, by omega⟩

-- § reductions

/-- Sequential sum of `f 0 .. f (n-1)`. -/
def sumRange (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumRange f n + f n

/-- A range sum splits at any point. -/
theorem sumRange_add (f : Nat → Int) (n m : Nat) :
    sumRange f (n + m) = sumRange f n + sumRange (fun i => f (n + i)) m := by
  induction m with
  | zero => simp [sumRange]
  | succ m ih =>
    show sumRange f (n + m) + f (n + m) =
      sumRange f n + (sumRange (fun i => f (n + i)) m + f (n + m))
    rw [ih]
    omega

/-- Termwise sums add. -/
theorem sumRange_pointwise_add (u v : Nat → Int) (m : Nat) :
    sumRange (fun b => u b + v b) m = sumRange u m + sumRange v m := by
  induction m with
  | zero => simp [sumRange]
  | succ m ih =>
    show sumRange (fun b => u b + v b) m + (u m + v m) =
      (sumRange u m + u m) + (sumRange v m + v m)
    omega

/-- A sum of zeros is zero. -/
theorem sumRange_zero (f : Nat → Int) (n : Nat) (h : ∀ i, i < n → f i = 0) :
    sumRange f n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show sumRange f n + f n = 0
    have h1 := ih (fun i hi => h i (by omega))
    have h2 := h n (by omega)
    omega

/-- Order of a double sum can be swapped. -/
theorem sumRange_swap (g : Nat → Nat → Int) (n m : Nat) :
    sumRange (fun a => sumRange (g a) m) n = sumRange (fun b => sumRange (fun a => g a b) n) m := by
  induction n with
  | zero =>
    show (0 : Int) = sumRange (fun _ => (0 : Int)) m
    exact (sumRange_zero _ m (fun _ _ => rfl)).symm
  | succ n ih =>
    show sumRange (fun a => sumRange (g a) m) n + sumRange (g n) m =
      sumRange (fun b => sumRange (fun a => g a b) n + g n b) m
    rw [sumRange_pointwise_add, ih]

/-- A sum over four terms. -/
theorem sumRange_four (P : Nat → Int) : sumRange P 4 = P 0 + P 1 + P 2 + P 3 := by
  show 0 + P 0 + P 1 + P 2 + P 3 = P 0 + P 1 + P 2 + P 3
  omega

/-- Four 32-row groups add up to 128 rows. -/
theorem rows_four_way (f : Nat → Int) :
    sumRange f 128 =
      sumRange (fun r => f (0 * 32 + r)) 32 + sumRange (fun r => f (1 * 32 + r)) 32 +
      sumRange (fun r => f (2 * 32 + r)) 32 + sumRange (fun r => f (3 * 32 + r)) 32 := by
  have h1 : sumRange f 128 = sumRange f 96 + sumRange (fun r => f (3 * 32 + r)) 32 :=
    sumRange_add f 96 32
  have h2 : sumRange f 96 = sumRange f 64 + sumRange (fun r => f (2 * 32 + r)) 32 :=
    sumRange_add f 64 32
  have h3 : sumRange f 64 = sumRange f 32 + sumRange (fun r => f (1 * 32 + r)) 32 :=
    sumRange_add f 32 32
  have h4 : sumRange f 32 = sumRange f 0 + sumRange (fun r => f (0 * 32 + r)) 32 :=
    sumRange_add f 0 32
  have h0 : sumRange f 0 = 0 := rfl
  omega

/-- THE warp-sharding contract: summing, over the 32 lanes, each lane's four registers `f (r*32 + lane)`
equals the sequential contraction over all 128 rows. -/
theorem lane_shard_sum (f : Nat → Int) :
    sumRange (fun l => sumRange (fun r => f (r * 32 + l)) 4) 32 = sumRange f 128 := by
  refine ((sumRange_swap (fun l r => f (r * 32 + l)) 32 4).trans ?_).trans (rows_four_way f).symm
  exact sumRange_four _

-- § indexing

/-- Register `r` of lane `l` is row `r*32 + l`; rows decompose back uniquely. -/
theorem lane_row_roundtrip {r lane : Nat} (hr : r < 4) (hl : lane < 32) :
    r * 32 + lane < 128 ∧ (r * 32 + lane) / 32 = r ∧ (r * 32 + lane) % 32 = lane := by
  omega

/-- Column `blockIdx.z*4 + threadIdx.y` (grid z = 32, block y = 4) is in `0..127` and unique. -/
theorem col_cover {z ty : Nat} (hz : z < 32) (hty : ty < 4) :
    z * 4 + ty < 128 ∧ (z * 4 + ty) / 4 = z ∧ (z * 4 + ty) % 4 = ty := by
  omega

/-- One warp per (head, column): `h_v * 32 * 4` threads in warps of 32 is `h_v * 128` warps. -/
theorem warp_count (h_v : Nat) : h_v * 32 * 4 / 32 = h_v * 4 := by
  omega

/-- Address of `state[i][head][col]`. -/
def stateIdx (h_v i head col : Nat) : Nat := (i * h_v + head) * 128 + col

/-- Every state address is inside the `128*128*h_v` allocation. -/
theorem stateIdx_lt {h_v i head col : Nat} (hi : i < 128) (hh : head < h_v) (hc : col < 128) :
    stateIdx h_v i head col < 128 * h_v * 128 := by
  unfold stateIdx
  have h1 : i * h_v + head < 128 * h_v := mul_add_lt_aux hi hh
  exact mul_add_lt_aux h1 hc

/-- Address of `output[head][col]`. -/
def outIdx (head col : Nat) : Nat := head * 128 + col

/-- Output writes (lane 0 only) are unique per `(head, col)`. -/
theorem outIdx_unique {head head' col col' : Nat} (hc : col < 128) (hc' : col' < 128)
    (e : outIdx head col = outIdx head' col') : head = head' ∧ col = col' :=
  split_unique_aux hc hc' e

/-- No two threads write the same state address: the (head, col, register, lane) tuple is recovered. -/
theorem state_write_unique {h_v head head' col col' r r' l l' : Nat}
    (hh : head < h_v) (hh' : head' < h_v) (hc : col < 128) (hc' : col' < 128)
    (hl : l < 32) (hl' : l' < 32)
    (e : stateIdx h_v (r * 32 + l) head col = stateIdx h_v (r' * 32 + l') head' col') :
    r = r' ∧ l = l' ∧ head = head' ∧ col = col' := by
  unfold stateIdx at e
  obtain ⟨e1, e2⟩ := split_unique_aux hc hc' e
  obtain ⟨e3, e4⟩ := split_unique_aux hh hh' e1
  obtain ⟨e5, e6⟩ := split_unique_aux hl hl' e3
  exact ⟨e5, e6, e4, e2⟩

/-- The shared q/k row `(head % h_k)*128 + i` stays in the `h_k*128` buffer. -/
theorem qk_index_lt {head h_k i : Nat} (hk : 0 < h_k) (hi : i < 128) :
    (head % h_k) * 128 + i < h_k * 128 :=
  mul_add_lt_aux (Nat.mod_lt head hk) hi

-- § span validation

/-- A span is valid when non-null, 4-aligned and non-wrapping. -/
def validSpan (ptr bytes max : Nat) : Prop := ptr ≠ 0 ∧ ptr % 4 = 0 ∧ bytes ≤ max - ptr

/-- Interval overlap as written in the source. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  exact ⟨fun h => ⟨h.2, h.1⟩, fun h => ⟨h.2, h.1⟩⟩

/-- A non-empty span overlaps itself. -/
theorem overlap_self {a n : Nat} (h : 0 < n) : overlap a n a n := by
  unfold overlap
  omega

/-- Adjacent spans are disjoint. -/
theorem adjacent_disjoint {a an b bn : Nat} (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap
  omega

/-- A valid span `[p, p+bytes)` cannot wrap the address space. -/
theorem validSpan_no_wrap {ptr bytes max : Nat} (hp : ptr ≤ max) (h : validSpan ptr bytes max) :
    ptr + bytes ≤ max := by
  unfold validSpan at h
  omega

/-- The host-side byte counts are all multiples of 4, so 4-byte alignment of the base suffices. -/
theorem span_sizes_float_multiple (h_v h_k : Nat) :
    (128 * 128 * h_v * 4) % 4 = 0 ∧ (128 * h_k * 4) % 4 = 0 ∧ (128 * h_v * 4) % 4 = 0 ∧
      (h_v * 4) % 4 = 0 := by
  omega

/-- The grid limit `h_v <= 65535` is what the host check enforces (`blockIdx.x` has 16-bit range here). -/
theorem grid_limit {h_v : Nat} (h : h_v ≤ 65535) : h_v < 65536 := by
  omega

end Strata.Kernels.NativeGdn
