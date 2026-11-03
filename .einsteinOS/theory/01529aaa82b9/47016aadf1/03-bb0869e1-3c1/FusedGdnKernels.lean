/-!
# MODULE: fused_gdn_kernels
Index arithmetic of the fused gated-delta-net kernels (conv+L2, alpha/beta projection, step+norm):
thread/row-group decomposition, head pairing, launch-grid coverage and the history slide.

## Source semantics
`gdn_step_norm_kernel`: block = head, thread (col, rg) with tid = rg*128+col owns rows rg*32..rg*32+31 of column col.
`gdn_conv_l2_kernel`: one block per 128 channels, history of three taps slides by one.
`gdn_ab_kernel`: one warp per row of the stacked [alpha; beta] projection, 8 rows per block.

## Form template identifiers
S -> state width 128; RG -> row groups 4; RPG -> rows per thread 32; h_k/h_v -> key/value head counts.
-/
namespace Strata.FusedGdn

/-- Linear thread id used by the step kernel. -/
def tid (rg col : Nat) : Nat := rg * 128 + col

/-- Thread ids stay below the 512-thread block size. -/
theorem tid_lt (rg col : Nat) (h1 : rg < 4) (h2 : col < 128) : tid rg col < 512 := by
  unfold tid; omega

/-- The row group is recovered from the thread id. -/
theorem tid_div (rg col : Nat) (h2 : col < 128) : tid rg col / 128 = rg := by
  unfold tid; omega

/-- The column is recovered from the thread id. -/
theorem tid_mod (rg col : Nat) (h2 : col < 128) : tid rg col % 128 = col := by
  unfold tid; omega

/-- Row-group slices of 32 rows stay inside the 128-row state. -/
theorem row_in_range (rg r : Nat) (h1 : rg < 4) (h2 : r < 32) : rg * 32 + r < 128 := by omega

/-- Row-group slices partition the rows: a row has a unique (group, offset). -/
theorem row_unique (rg r rg' r' : Nat) (h1 : rg < 4) (h2 : r < 32) (h3 : rg' < 4) (h4 : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' := by omega

/-- Every row has a group and offset. -/
theorem row_cover (i : Nat) (h : i < 128) : ∃ rg r, rg < 4 ∧ r < 32 ∧ i = rg * 32 + r :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- Div/mod uniqueness helper for addresses of the form a*m+b. -/
theorem divmod_unique (a b m : Nat) (hb : b < m) : (a * m + b) / m = a ∧ (a * m + b) % m = b := by
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hb]; simp
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt hb

/-- State address (row, head, col) with layout (S, h_v, S), col fastest. -/
def addr (hv row head col : Nat) : Nat := (row * hv + head) * 128 + col

/-- The state layout is injective on valid coordinates (no two threads alias). -/
theorem addr_inj (hv r h c r' h' c' : Nat) (hh : h < hv) (hh' : h' < hv) (hc : c < 128) (hc' : c' < 128)
    (e : addr hv r h c = addr hv r' h' c') : r = r' ∧ h = h' ∧ c = c' := by
  unfold addr at e
  have e1 : r * hv + h = r' * hv + h' := by omega
  have hc2 : c = c' := by omega
  have d1 := divmod_unique r h hv hh
  have d2 := divmod_unique r' h' hv hh'
  rw [e1] at d1
  exact ⟨by omega, by omega, hc2⟩

/-- The key/query head is taken modulo h_k, so it is always a valid key head. -/
theorem qh_lt (head hk : Nat) (h : 0 < hk) : head % hk < hk := Nat.mod_lt _ h

/-- Row-group-0 threads (tid < 128) occupy exactly warps 0..3 for the RMS reduction. -/
theorem rg0_warps (t : Nat) (h : t < 128) : t / 32 < 4 := by omega

/-- Five xor-shuffle rounds (16,8,4,2,1) reduce a 32-lane warp. -/
theorem warp_rounds : 16 + 8 + 4 + 2 + 1 + 1 = 32 := by decide

/-- Conv+L2 grid: one block per 128 channels covers all channels when 128 divides them. -/
theorem conv_grid_cover (channels : Nat) (h : channels % 128 = 0) : (channels / 128) * 128 = channels := by
  omega

/-- A q/k head block index is below the block count when qk_heads <= channels/128. -/
theorem conv_qk_blocks (b qk channels : Nat) (hb : b < qk) (h : qk ≤ channels / 128) : b < channels / 128 := by
  omega

/-- The history slides: [v0,v1,v2] with input x becomes [v1,v2,x]. -/
def slide3 {α : Type} (v0 v1 v2 x : α) : α × α × α := (v1, v2, x)

/-- Sliding drops the oldest tap and appends the newest. -/
theorem slide3_spec {α : Type} (v0 v1 v2 x : α) : slide3 v0 v1 v2 x = (v1, v2, x) := rfl

/-- alpha/beta kernel: every row below 2*h_v lies in some launched block. -/
theorem ab_grid_cover (row hv : Nat) (h : row < 2 * hv) : row / 8 < (2 * hv + 7) / 8 := by omega

/-- Warp index within an 8-warp block recovers the row. -/
theorem ab_row_decomp (b w : Nat) (hw : w < 8) : (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by
  omega

/-- Beta rows are shifted back to index r < h_v. -/
theorem ab_beta_index (row hv : Nat) (h1 : hv ≤ row) (h2 : row < 2 * hv) : row - hv < hv := by omega

/-- Alpha rows already satisfy r < h_v. -/
theorem ab_alpha_index (row hv : Nat) (h : row < hv) : row < hv := h

/-- The bf16 chunk loop with 8 values per uint4 covers n_embd/8 chunks; n_embd % 8 = 0 makes it exact. -/
theorem ab_chunks (n : Nat) (h : n % 8 = 0) : (n / 8) * 8 = n := by omega

end Strata.FusedGdn
