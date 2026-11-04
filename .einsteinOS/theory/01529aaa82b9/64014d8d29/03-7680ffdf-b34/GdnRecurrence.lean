/-!
# MODULE: gdn_recurrence
SOURCES: src/kernels/cuda/gdn.cu, src/kernels/cuda/fused_gdn.cu
Gated delta-net decode step: state layout (S, h_v, S) with j fastest, per-column recurrence, causal conv with a
sliding 3-row history, l2 / out norms, and the fused step+norm kernel that splits the 128 state rows over 4 row
groups; plus the fused alpha/beta projection and conv+l2 kernels.

## Source semantics
* State element (i, h, j) lives at ((i*h_v + h)*S + j); one thread owns one (h, j) column end to end.
* q/k heads are paired to value heads by MODULO: head h reads q/k head h % h_k.
* Fused step: thread (col, rg) owns rows rg*32 .. rg*32+31 of column col; the four partial dots are summed in
  fixed order.
* Conv: out = sum_i hist[i]*w[i] + x*w[3]; the history slides by one (drop oldest, append x).
* Alpha/beta: row < h_v is alpha (gate = softplus(acc+dt)*a), row >= h_v is beta (sigmoid).

## Form template identifiers
S -> 128; RG -> 4; RPG -> 32; state index -> `stIdx`; history -> `hist`.
-/

namespace GdnRecurrence

-- § helpers

/-- Decoding a mixed-radix pair. -/
theorem decode_div_mod (a b n : Nat) (hb : b < n) : (a*n+b)/n = a ∧ (a*n+b)%n = b := by
  have hn : 0 < n := by omega
  have e : a*n+b = b + a*n := Nat.add_comm _ _
  have h1 := Nat.add_mul_div_right b a hn
  rw [Nat.div_eq_of_lt hb] at h1
  constructor
  · rw [e]; omega
  · rw [e, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § state layout

/-- Linear index of state element (i, h, j) in the (S, h_v, S) layout. -/
def stIdx (hv S i h j : Nat) : Nat := (i * hv + h) * S + j

/-- Every state index is inside the `S * h_v * S` buffer. -/
theorem stIdx_lt (hv S i h j : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) : stIdx hv S i h j < S * hv * S := by
  unfold stIdx
  have h1 : i * hv + h < S * hv := idx2_lt i h S hv hi hh
  exact idx2_lt (i * hv + h) j (S * hv) S h1 hj

/-- Distinct (i, h, j) address distinct state elements, so columns never alias. -/
theorem stIdx_injective (hv S i1 i2 h1 h2 j1 j2 : Nat) (hh1 : h1 < hv) (hh2 : h2 < hv)
    (hj1 : j1 < S) (hj2 : j2 < S) (heq : stIdx hv S i1 h1 j1 = stIdx hv S i2 h2 j2) :
    i1 = i2 ∧ h1 = h2 ∧ j1 = j2 := by
  unfold stIdx at heq
  obtain ⟨d1, m1⟩ := decode_div_mod (i1*hv+h1) j1 S hj1
  obtain ⟨d2, m2⟩ := decode_div_mod (i2*hv+h2) j2 S hj2
  rw [heq] at d1 m1
  have e1 : i1*hv+h1 = i2*hv+h2 := d1.symm.trans d2
  have e2 : j1 = j2 := m1.symm.trans m2
  obtain ⟨a1, b1⟩ := decode_div_mod i1 h1 hv hh1
  obtain ⟨a2, b2⟩ := decode_div_mod i2 h2 hv hh2
  rw [e1] at a1 b1
  exact ⟨a1.symm.trans a2, b1.symm.trans b2, e2⟩

/-- The fused kernel's row-group stride: row `rg*32 + r` is `r` row-strides after row `rg*32`. -/
theorem fused_row_stride (A r hv head S col : Nat) :
    ((A * hv + head) * S + col) + r * (hv * S) = (((A + r) * hv + head) * S + col) := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Four row groups of 32 rows cover all 128 state rows exactly once. -/
theorem row_groups_cover (i : Nat) (h : i < 128) : i = (i / 32) * 32 + i % 32 ∧ i / 32 < 4 ∧ i % 32 < 32 := by
  omega

/-- The unfused grid covers all 128 columns with 4 blocks of 32 threads. -/
theorem columns_cover (j : Nat) (h : j < 128) : j / 32 < 4 ∧ j = (j / 32) * 32 + j % 32 := by omega

/-- Thread count is fixed by the state, independent of MAX_H: 128 * 48 = 6144. -/
theorem thread_count : 128 * 48 = 6144 := by decide

/-- Head pairing is modulo: always a valid q/k head. -/
theorem head_pair_lt (h hk : Nat) (hpos : 0 < hk) : h % hk < hk := Nat.mod_lt _ hpos

-- § convolution history

/-- Sequence `[hist0, hist1, hist2, x0, x1, ...]` as a function of position. -/
def seqAt {α : Type} (hist x : Nat → α) (i : Nat) : α := if i < 3 then hist i else x (i - 3)

/-- The window of the first-token step is the old history followed by the new input. -/
theorem window_first {α : Type} (hist x : Nat → α) :
    seqAt hist x 0 = hist 0 ∧ seqAt hist x 1 = hist 1 ∧ seqAt hist x 2 = hist 2 ∧ seqAt hist x 3 = x 0 := by
  simp [seqAt]

/-- Sliding the history by one token: new history = [h1, h2, x0]. -/
theorem slide_one {α : Type} (hist x : Nat → α) :
    seqAt hist x (1 + 0) = hist 1 ∧ seqAt hist x (1 + 1) = hist 2 ∧ seqAt hist x (1 + 2) = x 0 := by
  simp [seqAt]

/-- The window of token t+1 is the window of token t shifted by one. -/
theorem window_shift {α : Type} (hist x : Nat → α) (t j : Nat) :
    seqAt hist x ((t + 1) + j) = seqAt hist x (t + (j + 1)) := by
  have : (t + 1) + j = t + (j + 1) := by omega
  rw [this]

/-- After at least 3 tokens the history is made only of new inputs. -/
theorem history_after_many {α : Type} (hist x : Nat → α) (n j : Nat) (hn : 3 ≤ n) :
    seqAt hist x (n + j) = x (n + j - 3) := by
  unfold seqAt
  have : ¬ (n + j < 3) := by omega
  simp [this]

-- § alpha/beta projection

/-- A bf16 pair word: the low half becomes the high half of an f32 (`w << 16`), the high half stays in place. -/
theorem bf16_pair_split (w : Nat) (h : w < 4294967296) :
    w = (w / 65536) * 65536 + w % 65536 ∧ (w % 65536) * 65536 < 4294967296 := by omega

/-- Rows `[0, 2*h_v)` map to alpha (`row < h_v`) or beta, with `r < h_v`. -/
theorem ab_row_split (row hv : Nat) (h : row < 2 * hv) :
    (row < hv ∧ row < hv) ∨ (hv ≤ row ∧ row - hv < hv) := by omega

/-- The (2*h_v+7)/8 grid with 8 warps per block covers every row. -/
theorem ab_grid_covers (row hv : Nat) (h : row < 2 * hv) :
    row / 8 < (2 * hv + 7) / 8 ∧ row = (row / 8) * 8 + row % 8 := by omega

end GdnRecurrence
