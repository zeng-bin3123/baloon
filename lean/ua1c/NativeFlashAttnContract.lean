-- NativeFlashAttnContract.lean
-- Formal model of src/kernels/cuda/native_flash_attn.cu step contract and addressing.
namespace Baloon.NativeFlash

def stepValid (maxCtx pos nKv nBid width : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nKv = width ∧ pos = width - 1 ∧ nBid = width / 4

theorem stepFill_valid (maxCtx topK pos : Nat) (hctx : maxCtx ≤ 256) (hk : 256 ≤ topK)
    (hp : pos < maxCtx) :
    stepValid maxCtx pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) (topK + 4 - 1)) := by
  have hw : min (pos + 1) (topK + 4 - 1) = pos + 1 := Nat.min_eq_left (by omega)
  rw [hw]; unfold stepValid; omega

theorem valid_is_fill (maxCtx pos nKv nBid width : Nat) (h : stepValid maxCtx pos nKv nBid width) :
    nKv = pos + 1 ∧ nBid = (pos + 1) / 4 ∧ width = pos + 1 := by
  unfold stepValid at h
  obtain ⟨h1, _, h3, h4, h5⟩ := h
  refine ⟨by omega, ?_, by omega⟩
  have : pos + 1 = width := by omega; rw [this, h5]

theorem refuses_long (maxCtx pos nKv nBid width : Nat) (h : maxCtx < width) :
    ¬ stepValid maxCtx pos nKv nBid width := by
  unfold stepValid; omega

def kvOf (h : Nat) : Nat := h / 12

theorem kvOf_lt (h : Nat) (hh : h < 24) : kvOf h < 2 := by unfold kvOf; omega

def kvIdx (cell kv d : Nat) : Nat := (cell * 2 + kv) * 256 + d

theorem kvIdx_lt (cap cell kv d : Nat) (hc : cell < cap) (hk : kv < 2) (hd : d < 256) :
    kvIdx cell kv d < cap * 2 * 256 := by
  unfold kvIdx
  have h1 : (cell + 1) * 2 ≤ cap * 2 := Nat.mul_le_mul_right 2 (Nat.succ_le_of_lt hc)
  rw [Nat.succ_mul] at h1
  have h2 : cell * 2 + kv < cap * 2 := by omega
  have h3 : (cell * 2 + kv + 1) * 256 ≤ cap * 2 * 256 := Nat.mul_le_mul_right 256 (Nat.succ_le_of_lt h2)
  rw [Nat.succ_mul] at h3; omega

theorem sweep_covers (cell : Nat) (hc : cell < 256) :
    ∃ base warp hi row, base < 256 ∧ base % 128 = 0 ∧ warp < 4 ∧ hi < 4 ∧ row < 8 ∧
      cell = base + warp * 32 + hi * 8 + row :=
  ⟨cell / 128 * 128, cell % 128 / 32, cell % 32 / 8, cell % 8, by omega, by omega, by omega,
    by omega, by omega, by omega⟩

def spansOverlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

theorem spansOverlap_symm (a an b bn : Nat) : spansOverlap a an b bn ↔ spansOverlap b bn a an := by
  unfold spansOverlap; omega

theorem spansOverlap_self (a n : Nat) (hn : 0 < n) : spansOverlap a n a n := by
  unfold spansOverlap; omega

theorem spansOverlap_adjacent (a n m : Nat) : ¬ spansOverlap a n (a + n) m := by
  unfold spansOverlap; omega

-- Span type for validate_spans function
structure Span where
  addr : Nat
  bytes : Nat

def validate_spans (spans : List Span) : Bool :=
  let pairs := (List.range spans.length).bind (fun i =>
    (List.range i).map (fun j => (spans.getD i ⟨0, 0⟩, spans.getD j ⟨0, 0⟩)))
  pairs.all (fun p => !spansOverlap p.1.addr p.1.bytes p.2.addr p.2.bytes)

theorem validate_spans_disjoint (spans : List Span) (h : validate_spans spans = true)
    (i j : Nat) (hi : i < spans.length) (hj : j < spans.length) (hij : i ≠ j) :
    ¬ spansOverlap (spans.get ⟨i, hi⟩).addr (spans.get ⟨i, hi⟩).bytes
                   (spans.get ⟨j, hj⟩).addr (spans.get ⟨j, hj⟩).bytes := by
  unfold spansOverlap; omega

end Baloon.NativeFlash
