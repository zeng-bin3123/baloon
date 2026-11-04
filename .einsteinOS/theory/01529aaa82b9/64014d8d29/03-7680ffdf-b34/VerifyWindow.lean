/-!
# MODULE: verify_window
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token (speculative verify window) helpers: conv+l2 over a token window, history commit, alpha/beta for T
tokens, GDN step+norm over T tokens with replay and keep-count, device-side gathers (embedding, blobs, rows),
stream broadcasts, step-state builders (`dense_steps`, `window_ids`) and the MTP selector.

## Source semantics
* Token t's conv window is entries t..t+2 of [hist0,hist1,hist2,x0,x1,...] plus x_t.
* `gdn_conv_commit` keeps the last three entries of [hist | x_0..x_{n-1}], and is a no-op for n <= 0.
* `dense_steps`: steps = (c, c+1, (c+1)/4, c+1); consistent with the QSA step when no truncation happens.
* `window_ids`: start = max(0, n_kv - window), ids = start .. n_kv-1, width = n_kv - start.
* Embedding gather unpacks `8/bits` codes per byte; q/k/v offsets of the fused conv channels are contiguous.

## Form template identifiers
kVerifyMaxT -> T bound; steps record -> (pos, n_kv, n_bid, width); window -> `window`.
-/

namespace VerifyWindow

-- § helpers

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § conv window and commit

/-- Sequence `[hist0, hist1, hist2, x0, x1, ...]`. -/
def seqAt {α : Type} (hist x : Nat → α) (i : Nat) : α := if i < 3 then hist i else x (i - 3)

/-- Window entries of token t come from the history while `t + j < 3`, else from the inputs. -/
theorem window_entry {α : Type} (hist x : Nat → α) (t j : Nat) :
    seqAt hist x (t + j) = if t + j < 3 then hist (t + j) else x (t + j - 3) := rfl

/-- Commit of zero tokens leaves the history unchanged. -/
theorem commit_zero {α : Type} (hist x : Nat → α) (j : Nat) : seqAt hist x (0 + j) = seqAt hist x j := by
  simp

/-- Commit of one token slides the history: [h1, h2, x0]. -/
theorem commit_one {α : Type} (hist x : Nat → α) :
    seqAt hist x (1 + 0) = hist 1 ∧ seqAt hist x (1 + 1) = hist 2 ∧ seqAt hist x (1 + 2) = x 0 := by
  simp [seqAt]

/-- Commit of at least three tokens only reads inputs. -/
theorem commit_many {α : Type} (hist x : Nat → α) (n j : Nat) (hn : 3 ≤ n) :
    seqAt hist x (n + j) = x (n + j - 3) := by
  unfold seqAt
  have : ¬ (n + j < 3) := by omega
  simp [this]

/-- Consecutive token windows overlap by two entries. -/
theorem window_slides {α : Type} (hist x : Nat → α) (t j : Nat) :
    seqAt hist x ((t + 1) + j) = seqAt hist x (t + (j + 1)) := by
  have : (t + 1) + j = t + (j + 1) := by omega
  rw [this]

-- § fused qkv channel layout

/-- q occupies [0, qk), k occupies [qk, 2qk), v starts at 2qk: head/dim offsets stay in their segment. -/
theorem q_offset_lt (qh tid hk : Nat) (h1 : qh < hk) (h2 : tid < 128) : qh * 128 + tid < hk * 128 :=
  idx2_lt qh tid hk 128 h1 h2

/-- v offset lies in [2qk, 2qk + value_dim). -/
theorem v_offset_range (head col hv hk : Nat) (h1 : head < hv) (h2 : col < 128) :
    2 * (128 * hk) ≤ 2 * (128 * hk) + (head * 128 + col) ∧
    2 * (128 * hk) + (head * 128 + col) < 2 * (128 * hk) + hv * 128 := by
  have := idx2_lt head col hv 128 h1 h2
  omega

-- § step-state builders

/-- `dense_steps` record for cell c. -/
def denseStep (c : Nat) : Nat × Nat × Nat × Nat := (c, c + 1, (c + 1) / 4, c + 1)

/-- Below the selection bound the dense step equals the QSA step fill (width = min n 2051 = n). -/
theorem dense_matches_fill (c : Nat) (h : c + 1 ≤ 2051) :
    denseStep c = (c, c + 1, (c + 1) / 4, min (c + 1) 2051) := by
  unfold denseStep
  have : min (c + 1) 2051 = c + 1 := by omega
  rw [this]

/-- `window_ids`: start of the window. -/
def winStart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- The window width is `min n_kv window`. -/
theorem win_width (nKv window : Nat) : nKv - winStart nKv window = min nKv window := by
  unfold winStart; split <;> omega

/-- Every window id is a valid cell. -/
theorem win_ids_valid (nKv window j : Nat) (hj : j < nKv - winStart nKv window) :
    winStart nKv window + j < nKv := by
  by_cases c : nKv > window
  · have e : winStart nKv window = nKv - window := by unfold winStart; simp [c]
    rw [e] at hj ⊢; omega
  · have e : winStart nKv window = 0 := by unfold winStart; simp [c]
    rw [e] at hj ⊢; omega

/-- When the context fits the window the ids are the identity from 0. -/
theorem win_identity (nKv window : Nat) (h : nKv ≤ window) : winStart nKv window = 0 := by
  unfold winStart; simp; omega

-- § gathers and decodes

/-- Blob / row gathers: flat index i over `n * per` decodes to (row i/per < n, offset i%per < per). -/
theorem gather_decode (i n per : Nat) (hp : 0 < per) (h : i < n * per) :
    i / per < n ∧ i % per < per ∧ (i / per) * per + i % per = i := by
  refine ⟨(Nat.div_lt_iff_lt_mul hp).2 h, Nat.mod_lt _ hp, ?_⟩
  have := Nat.div_add_mod i per
  rw [Nat.mul_comm per (i / per)] at this
  exact this

/-- Source offset of a gathered row stays inside a table of `rows` rows. -/
theorem gather_src_lt (id o rows row16 : Nat) (h1 : id < rows) (h2 : o < row16) :
    id * row16 + o < rows * row16 := idx2_lt id o rows row16 h1 h2

/-- Rebased pointers are evenly spaced: ptr(k+1) = ptr(k) + blob_bytes. -/
theorem rebase_spacing (base k bytes : Nat) : (base + (k + 1) * bytes) = (base + k * bytes) + bytes := by
  rw [Nat.add_mul, Nat.one_mul]; omega

/-- 4-bit codes: two per byte, byte index in range for n codes. -/
theorem nibble_index (i n : Nat) (h : i < n) : i / 2 < (n + 1) / 2 := by omega

/-- A 4-bit code extracted from a byte is below 16. -/
theorem nibble_lt (byte sh : Nat) : (byte / sh) % 16 < 16 := Nat.mod_lt _ (by decide)

/-- Two nibbles recompose the byte. -/
theorem nibbles_recompose (byte : Nat) (h : byte < 256) : (byte / 16) % 16 * 16 + byte % 16 = byte := by omega

/-- 2-bit codes: four per byte. -/
theorem crumb_index (i n : Nat) (h : i < n) : i / 4 < (n + 3) / 4 := by omega

/-- Stream broadcast: `R[t*n*hc + i] = x[t*n + i % n]`, with `i % n < n`. -/
theorem broadcast_source_lt (i n hc : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- Token row offsets are disjoint across tokens. -/
theorem token_rows_disjoint (t1 t2 stride : Nat) (h : t1 < t2) : t1 * stride + stride ≤ t2 * stride := by
  have h1 : (t1 + 1) * stride ≤ t2 * stride := Nat.mul_le_mul_right stride (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

-- § keep-count and replay

/-- The state is committed only when a keep-count exists and is positive. -/
def commits (hasKeep : Bool) (n : Nat) : Bool := hasKeep && decide (0 < n)

/-- Without a keep-count the single-pass kernel never commits state. -/
theorem no_commit_without_keep (n : Nat) : commits false n = false := by
  unfold commits; simp

/-- Tokens whose output is written: indices t in [t_out_begin, n). Their count is at most `n`. -/
theorem outputs_le (n tOut : Nat) : n - tOut ≤ n := Nat.sub_le n tOut

/-- Replayed tokens (t < t_out_begin) are exactly those whose output is skipped. -/
theorem replayed_skip (t tOut : Nat) : (t < tOut) ↔ ¬ (tOut ≤ t) := by omega

/-- MTP chain: a draft row index is selected from device memory, so a row `< n_rows` gives an in-range copy. -/
theorem mtp_row_copy_lt (row i stride nRows : Nat) (hr : row < nRows) (hi : i < stride) :
    row * stride + i < nRows * stride := idx2_lt row i nRows stride hr hi

end VerifyWindow
