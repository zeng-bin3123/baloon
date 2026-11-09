/-!
# MODULE: native_gdn_step
SOURCES: src/kernels/cuda/native_gdn.cu
Warp-per-column gated-delta-net step (adapted from pinned llama.cpp): each warp owns one Strata (head, column) and
retains its four rows (r*32 + lane) in registers; two warp-sum reductions give kv and the attention readout.

## Source semantics
* grid = (h_v, 1, S/4), block = (32, 4); column = blockIdx.z*4 + threadIdx.y; row = r*32 + lane, r < 4.
* Preconditions: stream non-null, S = 128, h_k > 0, h_v % h_k = 0, h_v <= 65535; spans aligned and pairwise disjoint
  between state/output and every input.

## Form template identifiers
overlap -> span_overlap, S -> state_size
-/
namespace Strata.NativeGdnStep

-- § exact-arithmetic reduction library
/-- Sum of `f 0 .. f (n-1)` over the integers. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n+1 => sumTo f n + f n

/-- Additivity of reductions. -/
theorem sum_add_fn (f g : Nat → Int) (n : Nat) :
    sumTo (fun i => f i + g i) n = sumTo f n + sumTo g n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih => simp only [sumTo, ih]; omega

/-- The zero function reduces to zero. -/
theorem sumTo_zero_fn (m : Nat) : sumTo (fun _ => (0 : Int)) m = 0 := by
  induction m with
  | zero => simp [sumTo]
  | succ m ih => simp [sumTo, ih]

/-- Double reductions commute (reduction order is irrelevant in exact arithmetic). -/
theorem sumTo_comm (g : Nat → Nat → Int) (m n : Nat) :
    sumTo (fun a => sumTo (g a) m) n = sumTo (fun b => sumTo (fun a => g a b) n) m := by
  induction n with
  | zero => simp [sumTo, sumTo_zero_fn]
  | succ n ih =>
    simp only [sumTo]
    rw [sum_add_fn, ← ih]

/-- Splitting a reduction at an arbitrary point. -/
theorem sumTo_add (f : Nat → Int) (n m : Nat) :
    sumTo f (n+m) = sumTo f n + sumTo (fun r => f (n+r)) m := by
  induction m with
  | zero => simp [sumTo]
  | succ m ih =>
    have e : sumTo f (n + (m+1)) = sumTo f (n+m) + f (n+m) := rfl
    have e2 : sumTo (fun r => f (n+r)) (m+1) = sumTo (fun r => f (n+r)) m + f (n+m) := rfl
    rw [e, ih, e2]
    omega

/-- A reduction over `a*b` items equals the reduction of `a` block reductions of `b` items. -/
theorem sum_blocks (f : Nat → Int) (b a : Nat) :
    sumTo f (a*b) = sumTo (fun r => sumTo (fun c => f (r*b+c)) b) a := by
  induction a with
  | zero => simp [sumTo]
  | succ a ih =>
    have h : (a+1)*b = a*b + b := Nat.succ_mul a b
    rw [h, sumTo_add f (a*b) b, ih]
    try rfl

/-- The warp-sum of per-lane register partials (4 rows each) equals the contraction over all 128 rows. -/
theorem warp_partials_total (f : Nat → Int) :
    sumTo (fun lane => sumTo (fun r => f (r * 32 + lane)) 4) 32 = sumTo f 128 := by
  have h1 : sumTo (fun lane => sumTo (fun r => f (r * 32 + lane)) 4) 32
      = sumTo (fun r => sumTo (fun lane => f (r * 32 + lane)) 32) 4 :=
    sumTo_comm (fun lane r => f (r * 32 + lane)) 4 32
  rw [h1]
  exact (sum_blocks f 32 4).symm

-- § thread/row layout
/-- Every state row is owned by exactly one (register slot, lane). -/
theorem row_owner (i : Nat) (hi : i < 128) :
    ∃ r lane, r < 4 ∧ lane < 32 ∧ r * 32 + lane = i :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- Row ownership is injective. -/
theorem row_owner_inj (r lane r' lane' : Nat) (hl : lane < 32) (hl' : lane' < 32)
    (h : r * 32 + lane = r' * 32 + lane') : r = r' ∧ lane = lane' := by omega

/-- Every column is owned by exactly one (blockIdx.z, threadIdx.y). -/
theorem col_owner (c : Nat) (hc : c < 128) :
    ∃ z y, z < 128 / 4 ∧ y < 4 ∧ z * 4 + y = c :=
  ⟨c / 4, c % 4, by omega, by omega, by omega⟩

/-- Column ownership is injective. -/
theorem col_owner_inj (z y z' y' : Nat) (hy : y < 4) (hy' : y' < 4)
    (h : z * 4 + y = z' * 4 + y') : z = z' ∧ y = y' := by omega

/-- Block size 32*4 matches `__launch_bounds__(128, 2)`. -/
theorem block_size : 32 * 4 = 128 := by decide

/-- The q-head index of a head is below h_k, so q/k reads stay inside the h_k*128 floats. -/
theorem qk_read_lt (head hk i : Nat) (hk0 : 0 < hk) (hi : i < 128) : (head % hk) * 128 + i < 128 * hk := by
  have h : head % hk < hk := Nat.mod_lt _ hk0
  omega

-- § span validation
/-- Byte-span overlap test used by the host wrapper. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> intro h <;> omega

/-- Adjacent spans never overlap. -/
theorem adjacent_disjoint (a n m : Nat) : ¬ overlap a n (a + n) m := by
  unfold overlap
  omega

/-- A span never overlaps one that ends where it begins. -/
theorem adjacent_disjoint' (a n m : Nat) : ¬ overlap (a + n) m a n := by
  unfold overlap
  omega

/-- A nonempty span overlaps itself. -/
theorem overlap_self (a n : Nat) (h : 0 < n) : overlap a n a n := by
  unfold overlap
  omega

/-- Float spans are a multiple of the float size. -/
theorem float_span_aligned (n : Nat) : (n * 4) % 4 = 0 := by omega

/-- State bytes are 128*128*h_v*4. -/
theorem state_bytes_eq (hv : Nat) : 128 * 128 * hv * 4 = 65536 * hv := by omega

end Strata.NativeGdnStep
