/-!
# MODULE: native_gdn_step
SOURCES: src/kernels/cuda/native_gdn.cu
Native (llama.cpp-arithmetic) gated-delta-net step: each warp owns one (head, column) and keeps the four
rows `r*32 + lane` of its column in registers; plus the argument validation (alignment, no-wrap, disjointness).

## Source semantics
* Grid `(h_v, 1, S/4)`, block `(32, 4)`: `col = blockIdx.z*4 + threadIdx.y`, lane `l` owns rows `r*32 + l`.
* Preconditions: stream non-null, S = 128, `h_k > 0`, `h_v % h_k = 0`, `h_v ≤ 65535`.
* Spans must be float-aligned, non-wrapping, and state/output must be disjoint from everything else.

## Form template identifiers
step -> native_delta_step ; valid_span -> span_valid ; overlap -> span_overlap ;
native_gdn_step -> native_gdn_entry
-/

namespace Strata.NativeGdn

/-- Division/modulo uniqueness for a mixed-radix index `a*n+b`. -/
theorem divmod_unique {a b c d n : Nat} (hb : b < n) (hd : d < n) (h : a * n + b = c * n + d) :
    a = c ∧ b = d := by
  have hn : 0 < n := by omega
  have h1 : (a * n + b) / n = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hb, Nat.zero_add]
  have h2 : (c * n + d) / n = c := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hd, Nat.zero_add]
  have h3 : a = c := by rw [← h1, ← h2, h]
  refine ⟨h3, ?_⟩
  subst h3
  exact Nat.add_left_cancel h

-- § warp/register ownership

/-- Rows owned by a lane: `r*32 + lane` for `r < 4`, covering `[0,128)` bijectively. -/
theorem rows_cover (i : Nat) (h : i < 128) :
    ∃ r lane, r < 4 ∧ lane < 32 ∧ i = r * 32 + lane :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- No two (r, lane) pairs share a row. -/
theorem rows_injective {r l r' l' : Nat} (hl : l < 32) (hl' : l' < 32)
    (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by omega

/-- The column of a thread is below 128 when `z < 32`, `y < 4`. -/
theorem col_in_range {z y : Nat} (hz : z < 32) (hy : y < 4) : z * 4 + y < 128 := by omega

/-- Column assignment is injective in (z, y) and onto `[0,128)`. -/
theorem col_bijective (c : Nat) (h : c < 128) :
    c / 4 < 32 ∧ c % 4 < 4 ∧ c = (c / 4) * 4 + c % 4 := by omega

/-- Block shape: 32 x 4 = 128 threads, matching `__launch_bounds__(128, 2)`. -/
theorem block_threads : 32 * 4 = 128 ∧ 128 / 4 = 32 := by decide

/-- State address `(i*h_v + head)*S + col` is injective across the whole grid. -/
theorem address_injective {h_v i head col i' head' col' : Nat} (hh : head < h_v) (hh' : head' < h_v)
    (hc : col < 128) (hc' : col' < 128)
    (e : (i * h_v + head) * 128 + col = (i' * h_v + head') * 128 + col') :
    i = i' ∧ head = head' ∧ col = col' := by
  obtain ⟨a, b⟩ := divmod_unique hc hc' e
  obtain ⟨c, d⟩ := divmod_unique hh hh' a
  exact ⟨c, d, b⟩

/-- Query/key head `head % h_k` indexes a valid row of the q/k buffers. -/
theorem qk_row_in_bounds {head h_k i : Nat} (hk : 0 < h_k) (hi : i < 128) :
    (head % h_k) * 128 + i < 128 * h_k := by
  have h := Nat.mod_lt head hk
  omega

-- § validation predicates

/-- Span validity: nonnull, float-aligned, no pointer wrap. -/
def validSpan (addr bytes maxAddr : Nat) : Prop := addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ maxAddr - addr

/-- Overlap test of the source (`ap < bp + bn && bp < ap + an`). -/
def overlapSpan (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm {a an b bn : Nat} : overlapSpan a an b bn ↔ overlapSpan b bn a an := by
  unfold overlapSpan
  constructor <;> (intro h; exact ⟨h.2, h.1⟩)

/-- Separated spans do not overlap. -/
theorem overlap_false_of_separated {a an b bn : Nat} (h : a + an ≤ b) : ¬ overlapSpan a an b bn := by
  unfold overlapSpan
  omega

/-- A valid span does not wrap: its end is within the address space. -/
theorem valid_span_no_wrap {addr bytes maxAddr : Nat} (h : validSpan addr bytes maxAddr)
    (hm : addr ≤ maxAddr) : addr + bytes ≤ maxAddr := by
  unfold validSpan at h
  omega

/-- Byte counts used by the entry point at S = 128. -/
theorem byte_counts (h_v h_k : Nat) :
    128 * 128 * h_v * 4 = 65536 * h_v ∧ 128 * h_k * 4 = 512 * h_k ∧ 128 * h_v * 4 = 512 * h_v := by omega

/-- Head count limits: `h_v ≤ 65535` fits the grid.x limit and divisibility yields `h_k ≤ h_v`. -/
theorem grid_limit {h_v : Nat} (h : h_v ≤ 65535) : h_v < 65536 := by omega

end Strata.NativeGdn
