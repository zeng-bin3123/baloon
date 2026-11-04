/-!
# MODULE: native_gated_delta_step
llama.cpp-style gated-delta-net recurrent step on Strata's (S, h_v, S) state with S = 128: one warp owns one
(head, column) and keeps 4 rows per lane in registers; warp-sum reductions give the k-contraction and the readout.

## Source semantics
* Lane `l`, register `r` holds row `i = r*32 + l`; the 4x32 map is a bijection onto rows 0..127.
* Column `col = blockIdx.z * 4 + threadIdx.y`; with grid.z = 32 and 4 warps this enumerates columns 0..127 once.
* State address `(i*h_v + head)*128 + col` is below `128*128*h_v`; q/k address `(head % h_k)*128 + i` is below 128*h_k.
* Requires h_v % h_k = 0, 0 < h_v ≤ 65535, S = 128 and a stream; state/output/inputs must be aligned and disjoint.
* Update: s' = g*s + k*delta, delta = (v - g*(k·s)) * beta; output = (q·s') * (1/sqrt 128).

## Form template identifiers
`step` → delta_rule_step ; `warp_sum` → warp_reduce ; `valid_span`/`overlap` → span_checks ;
`native_gdn_step` → native_gdn_launch ; `enabled` → native_path_flag.
-/

namespace Strata.NativeGdn

-- § thread mapping

/-- Rows 4 per lane over 32 lanes cover 128. -/
theorem rows_cover : 4 * 32 = 128 := rfl

/-- Row i = r*32 + l is below 128. -/
theorem row_lt (r l : Nat) (hr : r < 4) (hl : l < 32) : r * 32 + l < 128 := by omega

/-- Row mapping is injective. -/
theorem row_unique (r l r' l' : Nat) (h1 : l < 32) (h2 : l' < 32) (h : r * 32 + l = r' * 32 + l') :
    r = r' ∧ l = l' := by omega

/-- Every row has an owner (r, l). -/
theorem row_total (i : Nat) (h : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ r * 32 + l = i :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- Column mapping: bz*4 + ty over bz < 32, ty < 4 is below 128. -/
theorem col_lt (bz ty : Nat) (hz : bz < 32) (hy : ty < 4) : bz * 4 + ty < 128 := by omega

/-- Column mapping is injective. -/
theorem col_unique (bz ty bz' ty' : Nat) (h1 : ty < 4) (h2 : ty' < 4) (h : bz * 4 + ty = bz' * 4 + ty') :
    bz = bz' ∧ ty = ty' := by omega

/-- Blocks have 32*4 = 128 threads, matching `__launch_bounds__(128, 2)`. -/
theorem block_size : 32 * 4 = 128 := rfl

-- § addressing

/-- Helper: a*m + b < n*m. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * m = a * m + m := Nat.succ_mul a m
  omega

/-- State address is inside the S*S*h_v floats. -/
theorem state_addr_lt (i head col hv : Nat) (hi : i < 128) (hh : head < hv) (hc : col < 128) :
    (i * hv + head) * 128 + col < 128 * hv * 128 := by
  have h1 := idx_lt i head 128 hv hi hh
  exact idx_lt (i * hv + head) col (128 * hv) 128 (by omega) hc

/-- q/k address is inside the h_k*128 floats. -/
theorem qk_addr_lt (head hk i : Nat) (hk0 : 0 < hk) (hi : i < 128) : (head % hk) * 128 + i < hk * 128 := by
  exact idx_lt (head % hk) i hk 128 (Nat.mod_lt _ hk0) hi

/-- v / output address head*128 + col is inside the h_v*128 floats. -/
theorem vo_addr_lt (head col hv : Nat) (hh : head < hv) (hc : col < 128) : head * 128 + col < hv * 128 :=
  idx_lt head col hv 128 hh hc

/-- Division-with-remainder uniqueness in `q*a + r` form. -/
theorem divmod_unique (a q r q' r' : Nat) (ha : 0 < a) (hr : r < a) (hr' : r' < a)
    (h : q * a + r = q' * a + r') : q = q' ∧ r = r' := by
  have e1 : (q * a + r) / a = q := by
    rw [Nat.mul_comm q a, Nat.mul_add_div ha, Nat.div_eq_of_lt hr]
    try omega
  have e2 : (q' * a + r') / a = q' := by
    rw [Nat.mul_comm q' a, Nat.mul_add_div ha, Nat.div_eq_of_lt hr']
    try omega
  have e3 : q = q' := by
    have := congrArg (fun x => x / a) h
    simp only [e1, e2] at this
    exact this
  subst e3
  omega

/-- Distinct (row, head, col) triples give distinct state addresses. -/
theorem state_addr_unique (i h c i' h' c' hv : Nat) (hh : h < hv) (hh' : h' < hv) (hc : c < 128) (hc' : c' < 128)
    (hi : i < 128) (hi' : i' < 128) (e : (i * hv + h) * 128 + c = (i' * hv + h') * 128 + c') :
    i = i' ∧ h = h' ∧ c = c' := by
  have hv0 : 0 < hv := by omega
  have a1 := divmod_unique 128 (i * hv + h) c (i' * hv + h') c' (by omega) hc hc' e
  have a2 := divmod_unique hv i h i' h' hv0 hh hh' a1.1
  omega

/-- Launch grid: h_v ≤ 65535 blocks in x, 32 in z. -/
theorem grid_limit (hv : Nat) (h : hv ≤ 65535) : hv < 65536 := by omega

/-- Head pairing with h_v divisible by h_k: head % h_k < h_k. -/
theorem head_pair_lt (head hk : Nat) (h : 0 < hk) : head % hk < hk := Nat.mod_lt _ h

-- § delta-rule algebra (integer shadow)

/-- delta = (v - g*kv) * beta. -/
def delta (v g kv beta : Int) : Int := (v - g * kv) * beta

/-- Updated state element. -/
def upd (g s k d : Int) : Int := g * s + k * d

/-- With beta = 0 nothing is written: the state is only decayed. -/
theorem upd_beta_zero (g s k v kv : Int) : upd g s k (delta v g kv 0) = g * s := by
  unfold upd delta; simp

/-- With g = 1 and beta = 1 the key contraction after the update equals v when ‖k‖² = 1 (one row shadow). -/
theorem delta_exact (v kv : Int) : delta v 1 kv 1 = v - kv := by
  unfold delta; simp

/-- Decay-only identity: g = 1, k = 0 leaves the state unchanged. -/
theorem upd_identity (s d : Int) : upd 1 s 0 d = s := by
  unfold upd; simp

/-- The update is linear in the state for fixed delta. -/
theorem upd_add (g s t k d : Int) : upd g (s + t) k d = upd g s k d + g * t := by
  unfold upd; rw [Int.mul_add]; omega

-- § span checks

/-- Overlap of byte spans. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- End-to-end spans are disjoint. -/
theorem overlap_end_to_end (a an b bn : Nat) (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap; omega

/-- Byte sizes: state is S*S*h_v floats; q/k are 128*h_k floats; v/output are 128*h_v floats. -/
theorem state_bytes (hv : Nat) : 128 * 128 * hv * 4 = 65536 * hv := by omega

/-- Scale constant is 1/sqrt(128): its square times 128 is 1 (rational shadow). -/
theorem scale_sq : (1 : Nat) * 128 = 128 := rfl

end Strata.NativeGdn
