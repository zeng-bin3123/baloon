/-!
# MODULE: gated_delta_net_step
SOURCES: src/kernels/cuda/gdn.cu
The non-projection parts of the gated delta-net (GDN) decode step: the per-column delta-rule
recurrence over a (S, h_v, S) state, the 4-tap causal depthwise conv with its sliding state,
the L2 / RMS output norms, and the sigmoid applied to `beta` by the layer glue.

## Source semantics
* State layout is (S, h_v, S) with `j` fastest: element (i, h, j) lives at `(i*h_v + h)*S + j`;
  the kernel walks column (h, j) as `col[i*stride]` with `col = h*S + j`, `stride = h_v*S`.
* One thread owns one (h, j) column end to end, so the recurrence needs no barrier.
* Head pairing is MODULO: value head `h` reads q/k head `h % h_k`.
* Pass 1 decays the column (`s := s*dec`) BEFORE contracting against k (PROPERTY 5);
  `d = (v - sk)*beta`; pass 2 applies the rank-1 update `s := s + k*d` and reads out against q
  using the UPDATED state.
* `beta` reaching the kernel must already be `sigmoid(beta_pre)` (the C1 bug was omitting it).
* Conv: `out = Σ_{i<dc-1} st[i]*w[i] + x*w[dc-1]`, then the state drops its oldest row and
  appends `x`.
* `gdn_step` rejects `S > 128` (shared staging width) and no-ops on non-positive shapes.

## Form template identifiers
state ↦ `State`, column update ↦ `colUpdate`, decay ↦ `decay`, contraction ↦ `dot`,
conv step ↦ `convOut`/`convSlide`, launch grid ↦ `gridX`/`gridY`, head pairing ↦ `srcHead`.
Arithmetic is modelled exactly over `Int` (the float kernel is an approximation of this).
-/

namespace Ref.GdnStep

-- § arithmetic helpers

/-- Dot product of two coefficient lists (truncating to the shorter one). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Linear bound on a two-level row-major index. -/
theorem lin_bound {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at h1
  omega

/-- Scaling the left operand of a dot product scales the result. -/
theorem dot_scale_left (g : Int) (s k : List Int) :
    dot (s.map (· * g)) k = g * dot s k := by
  induction s generalizing k with
  | nil => simp [dot]
  | cons a s ih =>
    cases k with
    | nil => simp [dot]
    | cons b k =>
      simp only [List.map, dot, ih]
      rw [Int.mul_add, Int.mul_comm a g, Int.mul_assoc]

-- § state layout

/-- Flat address of state element `(i, h, j)` in the `(S, h_v, S)` layout. -/
def stateIdx (S hv i h j : Nat) : Nat := (i * hv + h) * S + j

/-- The kernel's column addressing `h*S + j + i*(h_v*S)` is exactly the layout address. -/
theorem col_addr_eq (S hv i h j : Nat) :
    h * S + j + i * (hv * S) = stateIdx S hv i h j := by
  unfold stateIdx
  rw [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Every in-range state element lies inside the `S*h_v*S` buffer. -/
theorem stateIdx_lt {S hv i h j : Nat} (hi : i < S) (hh : h < hv) (hj : j < S) :
    stateIdx S hv i h j < S * hv * S := by
  have h1 : i * hv + h < S * hv := lin_bound hi hh
  exact lin_bound h1 hj

/-- Distinct in-range `(i, h, j)` give distinct addresses (no two threads share a cell). -/
theorem stateIdx_inj {S hv i h j i' h' j' : Nat} (hh : h < hv) (hh' : h' < hv)
    (hj : j < S) (hj' : j' < S)
    (heq : stateIdx S hv i h j = stateIdx S hv i' h' j') : i = i' ∧ h = h' ∧ j = j' := by
  unfold stateIdx at heq
  have hS : 0 < S := by omega
  have m1 : ((i * hv + h) * S + j) % S = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
  have m2 : ((i' * hv + h') * S + j') % S = j' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj']
  have hjj : j = j' := by rw [← m1, ← m2, heq]
  subst hjj
  have e1 : (i * hv + h) * S = (i' * hv + h') * S := by omega
  have e2 : i * hv + h = i' * hv + h' := Nat.eq_of_mul_eq_mul_right hS e1
  have n1 : (h + i * hv) % hv = h := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hh]
  have n2 : (h' + i' * hv) % hv = h' := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hh']
  have hhh : h = h' := by
    rw [← n1, ← n2, Nat.add_comm h, Nat.add_comm h', e2]
  subst hhh
  have hv0 : 0 < hv := by omega
  have e3 : i * hv = i' * hv := by omega
  exact ⟨Nat.eq_of_mul_eq_mul_right hv0 e3, rfl, rfl⟩

-- § launch geometry

/-- Threads along `j` per block (`JTHREADS`). -/
def JTHREADS : Nat := 32
/-- Heads per block (`MAX_H`, set to 1 as a latency-hiding fix). -/
def MAX_H : Nat := 1

/-- Grid x extent: `ceil(S / 32)`. -/
def gridX (S : Nat) : Nat := (S + JTHREADS - 1) / JTHREADS
/-- Grid y extent: `ceil(h_v / MAX_H)`. -/
def gridY (hv : Nat) : Nat := (hv + MAX_H - 1) / MAX_H

/-- Every column index `j < S` is reached by some block `bx < gridX S` and lane `tx < 32`. -/
theorem grid_covers_columns {S j : Nat} (hj : j < S) :
    ∃ bx tx, bx < gridX S ∧ tx < JTHREADS ∧ bx * JTHREADS + tx = j := by
  refine ⟨j / 32, j % 32, ?_, ?_, ?_⟩ <;> simp [gridX, JTHREADS] <;> omega

/-- With `MAX_H = 1`, there is exactly one block row per value head. -/
theorem gridY_eq_heads (hv : Nat) : gridY hv = hv := by
  simp [gridY, MAX_H]

/-- The thread count is `gridX*32 ≥ S` columns per head: the `j >= S` guard is needed only for
    the tail block. -/
theorem gridX_threads_ge (S : Nat) : S ≤ gridX S * JTHREADS := by
  simp [gridX, JTHREADS]; omega

/-- At the real geometry (S = 128, h_v = 48) the grid is (4, 48) = 192 blocks of 32 threads. -/
theorem real_geometry_blocks : gridX 128 = 4 ∧ gridY 48 = 48 ∧ gridX 128 * gridY 48 = 192 := by
  decide

/-- Modulo head pairing always selects a valid q/k head. -/
def srcHead (h hk : Nat) : Nat := h % hk

/-- The paired source head is in range whenever there is at least one q/k head. -/
theorem srcHead_lt {h hk : Nat} (hk0 : 0 < hk) : srcHead h hk < hk := Nat.mod_lt _ hk0

-- § recurrence (one column)

/-- Pass 1: decay every row of the column by `dec`. -/
def decay (dec : Int) (s : List Int) : List Int := s.map (· * dec)

/-- The rank-1 update of pass 2: `s_i + k_i * d`. -/
def rank1 (s k : List Int) (d : Int) : List Int := List.zipWith (fun a b => a + b * d) s k

/-- One column step exactly as `gdn_step_kernel` performs it: decayed state, write strength
    `d = (v - <decayed s, k>)*beta`, updated state and its readout against `q`. -/
def colUpdate (dec beta v : Int) (s k q : List Int) : List Int × Int :=
  let sd := decay dec s
  let d := (v - dot sd k) * beta
  let s' := rank1 sd k d
  (s', dot s' q)

/-- PROPERTY 5: contracting the decayed state equals decaying the contraction. -/
theorem decay_before_contract (dec : Int) (s k : List Int) :
    dot (decay dec s) k = dec * dot s k := dot_scale_left dec s k

/-- Decay preserves the column length. -/
theorem decay_length (dec : Int) (s : List Int) : (decay dec s).length = s.length := by
  simp [decay]

/-- The rank-1 update preserves the column length when `k` covers the column. -/
theorem rank1_length (s k : List Int) (d : Int) (h : s.length = k.length) :
    (rank1 s k d).length = s.length := by
  simp [rank1, h]

/-- The readout uses the UPDATED state: it is the dot of the returned state with `q`. -/
theorem readout_uses_updated (dec beta v : Int) (s k q : List Int) :
    (colUpdate dec beta v s k q).2 = dot (colUpdate dec beta v s k q).1 q := rfl

/-- `zipWith` that keeps its left argument is the identity on equal-length lists. -/
theorem zipWith_fst (t k : List Int) (h : t.length = k.length) :
    List.zipWith (fun a _ => a) t k = t := by
  induction t generalizing k with
  | nil => simp
  | cons a t ih =>
    cases k with
    | nil => simp at h
    | cons b k => simp [List.zipWith, ih k (by simpa using h)]

/-- With `beta = 0` the step is pure decay (the write strength vanishes). -/
theorem zero_beta_is_decay (dec v : Int) (s k q : List Int) (h : s.length = k.length) :
    (colUpdate dec 0 v s k q).1 = decay dec s := by
  have hl : (decay dec s).length = k.length := by rw [decay_length]; exact h
  simp only [colUpdate, rank1, Int.mul_zero, Int.add_zero]
  exact zipWith_fst _ _ hl

-- § conv step

/-- Conv output: state taps against `w[0..dc-2]`, newest input on the last tap. -/
def convOut (st w : List Int) (x : Int) : Int := dot (st ++ [x]) w

/-- Conv state slide: drop the oldest row, append the newest input. -/
def convSlide (st : List Int) (x : Int) : List Int := st.tail ++ [x]

/-- The slide keeps a non-empty conv state at constant length `dc - 1`. -/
theorem convSlide_length {st : List Int} (h : st ≠ []) (x : Int) :
    (convSlide st x).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a t => simp [convSlide]

/-- After the slide the newest row is the input just consumed. -/
theorem convSlide_last (st : List Int) (x : Int) :
    (convSlide st x).getLast? = some x := by
  simp [convSlide]

/-- For `d_conv = 4`: the output is the 4-tap dot with the oldest row on `w[0]`. -/
theorem convOut_four (a b c x w0 w1 w2 w3 : Int) :
    convOut [a, b, c] [w0, w1, w2, w3] x = a * w0 + (b * w1 + (c * w2 + (x * w3 + 0))) := rfl

-- § activations and guards

/-- Softplus large-argument branch: above 20 it is the identity (an abstract model). -/
def softplusBranch (f : Int → Int) (x : Int) : Int := if x > 20 then x else f x

/-- Above the threshold the overflow-avoiding branch returns its argument. -/
theorem softplus_large {f : Int → Int} {x : Int} (h : x > 20) : softplusBranch f x = x := by
  simp [softplusBranch, h]

/-- Host guard of `gdn_step`: launches only when all shapes are positive and `S ≤ 128`. -/
def stepLaunches (S hk hv : Int) : Bool := S > 0 && hk > 0 && hv > 0 && S ≤ 128

/-- A launched step always has a non-empty, stage-able state width. -/
theorem stepLaunches_bounds {S hk hv : Int} (h : stepLaunches S hk hv = true) :
    0 < S ∧ S ≤ 128 ∧ 0 < hk ∧ 0 < hv := by
  simp [stepLaunches] at h
  omega

/-- `gdn_l2_norm` refuses rows wider than the 1024-wide reduction. -/
def l2Launches (rows cols : Int) : Bool := rows > 0 && cols > 0 && cols ≤ 1024

/-- A launched L2 norm has `0 < cols ≤ 1024`. -/
theorem l2Launches_bounds {r c : Int} (h : l2Launches r c = true) : 0 < c ∧ c ≤ 1024 := by
  simp [l2Launches] at h
  omega

/-- The beta-gate grid `ceil(n/128)` blocks of 128 covers all heads. -/
theorem beta_gate_covers (n : Nat) : n ≤ (n + 127) / 128 * 128 := by omega

end Ref.GdnStep


