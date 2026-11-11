/-!
# MODULE: native_flash_attn_contract
SOURCES: src/kernels/cuda/native_flash_attn.cu
The pinned llama.cpp `fattn-vec` specialisation (D = 256, Q 24 heads / KV 2 heads, one column) used for
short contexts: a device-side step guard, the dense K/V addressing `(cell*2 + kv)*256 + d`, the
padded 256-cell sweep, and the host span validation.

## Source semantics
* The kernel accepts a step iff `1 ≤ width ≤ max_context`, `n_kv = width`, `pos = width - 1`,
  `n_bid = width / 4`; otherwise it writes NaN and reports `kNativeFlashAttnUnsupportedStep`.
* For `max_context ≤ 256 ≤ top_k` every `qsa_step_fill` with `pos < max_context` is accepted (below the
  selection bound the selection is the identity), and every accepted step IS such a fill.
* Head `h` reads KV head `h / 12`; K/V element `(cell*2 + kv)*256 + d` with `cell < capacity` is inside
  `capacity * 512` halves (`kv_bytes = capacity * 1024`).
* The padded sweep (`base` in steps of 128 up to 256) visits each cell `< 256` exactly once.
* Host spans must be non-null, aligned and pairwise disjoint.

## Form template identifiers
`valid` → `stepValid`, `width/nkv/pos/nbid` → step fields, `kv = head / 12` → `kvOf`,
`(cell*2+kv)*256+d` → `kvIdx`, `validate_spans` overlap → `spansOverlap`.
-/

namespace Strata.NativeFlash

/-- The device-side step guard. -/
def stepValid (maxCtx pos nKv nBid width : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nKv = width ∧ pos = width - 1 ∧ nBid = width / 4

/-- `qsa_step_fill` with `r = 4` and `top_k ≥ 256` passes the guard for every position in context. -/
theorem stepFill_valid (maxCtx topK pos : Nat) (hctx : maxCtx ≤ 256) (hk : 256 ≤ topK)
    (hp : pos < maxCtx) :
    stepValid maxCtx pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) (topK + 4 - 1)) := by
  have hw : min (pos + 1) (topK + 4 - 1) = pos + 1 := Nat.min_eq_left (by omega)
  rw [hw]
  unfold stepValid
  omega

/-- Every accepted step is the fill of `width - 1`. -/
theorem valid_is_fill (maxCtx pos nKv nBid width : Nat) (h : stepValid maxCtx pos nKv nBid width) :
    nKv = pos + 1 ∧ nBid = (pos + 1) / 4 ∧ width = pos + 1 := by
  unfold stepValid at h
  obtain ⟨h1, _, h3, h4, h5⟩ := h
  refine ⟨by omega, ?_, by omega⟩
  have : pos + 1 = width := by omega
  rw [this, h5]

/-- A step beyond the context is refused. -/
theorem refuses_long (maxCtx pos nKv nBid width : Nat) (h : maxCtx < width) :
    ¬ stepValid maxCtx pos nKv nBid width := by
  unfold stepValid
  omega

-- § addressing

/-- KV head of query head `h` (24 query heads over 2 KV heads). -/
def kvOf (h : Nat) : Nat := h / 12

/-- All 24 query heads map onto the 2 KV heads. -/
theorem kvOf_lt (h : Nat) (hh : h < 24) : kvOf h < 2 := by
  unfold kvOf; omega

/-- Dense K/V half index. -/
def kvIdx (cell kv d : Nat) : Nat := (cell * 2 + kv) * 256 + d

/-- Every K/V read of a cell inside the capacity stays inside `capacity * 512` halves. -/
theorem kvIdx_lt (cap cell kv d : Nat) (hc : cell < cap) (hk : kv < 2) (hd : d < 256) :
    kvIdx cell kv d < cap * 2 * 256 := by
  unfold kvIdx
  have h1 : (cell + 1) * 2 ≤ cap * 2 := Nat.mul_le_mul_right 2 (Nat.succ_le_of_lt hc)
  rw [Nat.succ_mul] at h1
  have h2 : cell * 2 + kv < cap * 2 := by omega
  have h3 : (cell * 2 + kv + 1) * 256 ≤ cap * 2 * 256 := Nat.mul_le_mul_right 256 (Nat.succ_le_of_lt h2)
  rw [Nat.succ_mul] at h3
  omega

/-- The padded sweep's cell `base + warp*32 + (lane & ~7) + row` decomposes every cell `< 256`. -/
theorem sweep_covers (cell : Nat) (hc : cell < 256) :
    ∃ base warp hi row, base < 256 ∧ base % 128 = 0 ∧ warp < 4 ∧ hi < 4 ∧ row < 8 ∧
      cell = base + warp * 32 + hi * 8 + row :=
  ⟨cell / 128 * 128, cell % 128 / 32, cell % 32 / 8, cell % 8, by omega, by omega, by omega,
    by omega, by omega, by omega⟩

-- § spans

/-- Two half-open byte spans overlap. -/
def spansOverlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric, so the pairwise check needs only `j < i`. -/
theorem spansOverlap_symm (a an b bn : Nat) : spansOverlap a an b bn ↔ spansOverlap b bn a an := by
  unfold spansOverlap; omega

/-- A non-empty span always overlaps itself: aliasing two arguments is rejected. -/
theorem spansOverlap_self (a n : Nat) (hn : 0 < n) : spansOverlap a n a n := by
  unfold spansOverlap; omega

/-- Adjacent spans do not overlap. -/
theorem spansOverlap_adjacent (a n m : Nat) : ¬ spansOverlap a n (a + n) m := by
  unfold spansOverlap; omega

end Strata.NativeFlash
