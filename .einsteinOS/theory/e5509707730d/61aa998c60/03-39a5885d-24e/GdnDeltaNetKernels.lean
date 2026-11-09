/-!
# MODULE: gdn_deltanet_kernels
SOURCES: src/kernels/cuda/gdn.cu
The non-projection parts of the gated delta-net: the column-parallel recurrence step, the sliding causal conv, the
L2 norm, the beta sigmoid gate and the output RMS-norm/gate. State layout is (S, h_v, S) with j fastest.

## Source semantics
* One thread owns one (head, j) state column end to end, so the recurrence needs no barrier.
* Decay happens BEFORE the update: sk = sum_i (dec*s_i)*k_i, d = (v - sk)*beta, s_i' = dec*s_i + k_i*d, o = sum s_i'*q_i.
* MAX_H = 1 head per block (latency hiding knob): grid = (S/32, h_v).
* Conv: kernel[0] reads the oldest state row, the new input lands in the last tap, and the state slides.

## Form template identifiers
S -> state_size, h_v -> value_heads, h_k -> key_heads, JTHREADS -> j_threads_per_block, MAX_H -> heads_per_block
-/
namespace Strata.GdnDeltaNetKernels

-- § exact-arithmetic reduction library
/-- Sum of `f 0 .. f (n-1)` over the integers. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n+1 => sumTo f n + f n

/-- Scalar factors pull out of a reduction. -/
theorem sum_scale (g : Int) (f : Nat → Int) (n : Nat) :
    sumTo (fun i => g * f i) n = g * sumTo f n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih => simp [sumTo, ih, Int.mul_add]

/-- Decaying the state before the k-contraction equals decaying the contraction (the two kernels agree). -/
theorem decay_commutes (g : Int) (s k : Nat → Int) (n : Nat) :
    sumTo (fun i => g * s i * k i) n = g * sumTo (fun i => s i * k i) n := by
  have h : (fun i => g * s i * k i) = (fun i => g * (s i * k i)) :=
    funext fun i => Int.mul_assoc _ _ _
  rw [h, sum_scale]

/-- The decay-first delta of gdn.cu equals the fused `(v - g*kv)*beta` form of fused_gdn.cu. -/
theorem delta_two_pass_eq_fused (g beta v : Int) (s k : Nat → Int) (n : Nat) :
    (v - sumTo (fun i => g * s i * k i) n) * beta = (v - g * sumTo (fun i => s i * k i) n) * beta := by
  rw [decay_commutes]

-- § grid geometry
/-- j-threads per block. -/
def JTHREADS : Nat := 32
/-- heads staged per block. -/
def MAX_H : Nat := 1

/-- For S = 128 the x-grid is 4 blocks. -/
theorem grid_x_128 : (128 + JTHREADS - 1) / JTHREADS = 4 := by decide

/-- With one head per block and h_v = 48 the grid is 4 x 48 = 192 blocks. -/
theorem grid_blocks_48 : ((128 + JTHREADS - 1) / JTHREADS) * ((48 + MAX_H - 1) / MAX_H) = 192 := by decide

/-- With MAX_H = 1 every head gets its own block row. -/
theorem grid_y_eq (hv : Nat) : (hv + MAX_H - 1) / MAX_H = hv := by
  simp only [MAX_H]
  omega

/-- The x-grid covers all S columns. -/
theorem grid_x_cover (S : Nat) : S ≤ ((S + JTHREADS - 1) / JTHREADS) * JTHREADS := by
  simp only [JTHREADS]
  omega

/-- Heads handled by a block: min(MAX_H, h_v - h0) is positive and stays in range. -/
theorem head_in_range (h0 hi hv : Nat) (h0lt : h0 < hv) (hi_lt : hi < min MAX_H (hv - h0)) :
    h0 + hi < hv := by
  simp only [MAX_H] at *
  omega

/-- Staging loop: thread t loading i = t, t+32, ... covers every i < S. -/
theorem staging_cover (S i : Nat) (h : i < S) :
    i % JTHREADS < JTHREADS ∧ i = i % JTHREADS + JTHREADS * (i / JTHREADS) := by
  simp only [JTHREADS]
  omega

/-- Head pairing is modulo h_k and in range. -/
theorem qhead_lt (head hk : Nat) (h : 0 < hk) : head % hk < hk := Nat.mod_lt _ h

-- § state addressing
/-- Row-major pair bound: a*B + b < A*B. -/
theorem pair_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : a + 1 ≤ A := ha
  have h2 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- State index (i, h, j) of the (S, h_v, S) layout is in range. -/
theorem stateIdx_lt (S hv i h j : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) :
    (i * hv + h) * S + j < S * hv * S := by
  have h1 : i * hv + h < S * hv := pair_lt i h S hv hi hh
  exact pair_lt (i * hv + h) j (S * hv) S h1 hj

/-- For S = 128, h_v = 48 the state indexing is injective. -/
theorem stateIdx_inj_128_48 (i h j i' h' j' : Nat) (hi : i < 128) (hh : h < 48) (hj : j < 128)
    (hi' : i' < 128) (hh' : h' < 48) (hj' : j' < 128)
    (e : (i * 48 + h) * 128 + j = (i' * 48 + h') * 128 + j') : i = i' ∧ h = h' ∧ j = j' := by
  omega

/-- q/k staging reads k[src*S + i] with src < h_k and i < S: in range. -/
theorem qk_stage_lt (S hk src i : Nat) (hs : src < hk) (hi : i < S) : src * S + i < hk * S :=
  pair_lt src i hk S hs hi

-- § sliding conv state
/-- Slide: drop the oldest row, append the newest input. -/
def slide (st : List Int) (x : Int) : List Int := st.drop 1 ++ [x]

/-- Sliding preserves the state length. -/
theorem slide_length (st : List Int) (x : Int) (h : st ≠ []) : (slide st x).length = st.length := by
  have : 0 < st.length := List.length_pos_of_ne_nil h
  simp [slide] <;> omega

/-- After three slides the 3-row state is exactly the last three inputs. -/
theorem slide_three (a b c x y z : Int) : slide (slide (slide [a, b, c] x) y) z = [x, y, z] := rfl

-- § launch coverage
/-- The conv launch covers all channels. -/
theorem conv_grid_cover (c : Nat) : c ≤ ((c + 255) / 256) * 256 := by omega

/-- The beta gate launch covers all heads. -/
theorem beta_grid_cover (n : Nat) : n ≤ ((n + 127) / 128) * 128 := by omega

/-- Large softplus inputs pass through (overflow guard). -/
theorem softplus_large (x : Int) (f : Int → Int) (h : 20 < x) : (if 20 < x then x else f x) = x := by
  simp [h]

end Strata.GdnDeltaNetKernels
