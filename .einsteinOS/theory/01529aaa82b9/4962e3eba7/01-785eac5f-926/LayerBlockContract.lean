import Mathlib

/-!
# MODULE: layer_block_contract
Structural contracts of one decoder layer's composition (`layer.hpp`): the MoE block split at the router,
the stage seams of `block_layer_pre`, the shared-expert-early reordering, buffer alignment, and the index
arithmetic of the QSA `q | gate` split and the indexer's pooled keys.

## Source semantics
* `moe_layer = route -> shared -> combine`; `pre = route`, `post = combine ∘ shared`; with the pool in the
  middle and acting only on host/parts state, `post ∘ pool ∘ pre` computes the same as the unsplit layer.
* `half = 1` runs stages 0..2, `half = 2` runs stages 3..4, `half = 0` runs all five; the halves tile the whole.
* Moving the shared expert before the CPU pool is bitwise neutral because they write disjoint buffers.
* `gdn_buffers_*`, `moe_buffers_*` regions are 16-byte aligned.
* `q_full` is `[q | gate]` per head by halves of `2 * head_dim`: q and gate indices never collide and stay inside.
* The indexer's pooled-key index `pos / idx_block` stays inside `max_cells / idx_block + 2` rows.

## Form template identifiers
* `route`, `shared`, `combine` <- `moe_route`, `moe_shared`, `moe_combine_parts`
* `pool`                       <- the host's CPU expert pool step between `pre[l]` and `post[l]`
* `stagesOf`                   <- the `half` argument of `block_layer_pre`
* `align16`                    <- the 16-byte alignment in `*_buffers_bytes`/`*_init`
* `qIdx`, `gIdx`               <- `qsa_layer`'s `q_full` split
-/

namespace Strata.Layer

-- § the MoE block split at the router

/-- The unsplit MoE block: route, then the shared expert, then the combine. -/
def moeLayer {σ : Type} (route shared combine : σ → σ) : σ → σ := combine ∘ shared ∘ route

/-- The first graph half: routing and the doorbell publish. -/
def pre {σ : Type} (route : σ → σ) : σ → σ := route

/-- The second graph half: the shared expert and the combine (`moe_finish`). -/
def post {σ : Type} (shared combine : σ → σ) : σ → σ := combine ∘ shared

/-- `moe_finish` followed by nothing else, run after `pre`, is exactly the unsplit MoE block. -/
theorem moe_split_eq {σ : Type} (route shared combine : σ → σ) :
    moeLayer route shared combine = post shared combine ∘ pre route := by
  rfl

/-- With the pool as the identity on the device state, the three-step layer is the unsplit layer. -/
theorem split_with_pool_id {σ : Type} (route shared combine : σ → σ) :
    post shared combine ∘ id ∘ pre route = moeLayer route shared combine := by
  rfl

/-- Device state of the shared-expert/pool race: the shared expert writes the first component, the pool the second. -/
def sharedOp {α β : Type} (f : α → α) : α × β → α × β := fun s => (f s.1, s.2)

/-- The pool's write, on the second component. -/
def poolOp {α β : Type} (g : β → β) : α × β → α × β := fun s => (s.1, g s.2)

/-- Running the shared expert before or after the pool gives the same state: moving it early is bitwise neutral. -/
theorem shared_early_commutes {α β : Type} (f : α → α) (g : β → β) :
    (sharedOp (β := β) f) ∘ (poolOp (α := α) g) = (poolOp (α := α) g) ∘ (sharedOp (β := β) f) := by
  funext s
  rfl

-- § stage seams of block_layer_pre

/-- Stages run by `half`: 1 = gr_read, attention, gr_write; 2 = gr_read, router; anything else = all five. -/
def stagesOf : Nat → List Nat
  | 1 => [0, 1, 2]
  | 2 => [3, 4]
  | _ => [0, 1, 2, 3, 4]

/-- The two halves, in order, tile the whole layer front exactly. -/
theorem halves_tile : stagesOf 1 ++ stagesOf 2 = stagesOf 0 := by
  decide

/-- The halves are disjoint: no stage runs twice. -/
theorem halves_disjoint : ∀ s ∈ stagesOf 1, s ∉ stagesOf 2 := by
  decide

-- § buffer alignment

/-- Round up to a multiple of 16, the alignment between carved regions. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- An aligned offset is a multiple of 16. -/
theorem align16_dvd (n : Nat) : 16 ∣ align16 n := by
  unfold align16
  exact Nat.dvd_mul_left 16 _

/-- Alignment never shrinks a region. -/
theorem le_align16 (n : Nat) : n ≤ align16 n := by
  unfold align16
  omega

/-- Alignment adds less than 16 bytes of padding. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16
  omega

/-- Alignment is idempotent: an aligned region stays where it is. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by
  unfold align16
  omega

-- § the dump buffer

/-- The per-layer host dump's regions tile its length: two `n_embd` halves, two `hc` injections, the QSA attention. -/
theorem dump_regions_tile (ne hc nh hd : Nat) :
    (2 * ne + hc + hc) + nh * hd = 2 * ne + 2 * hc + nh * hd := by
  omega

-- § the QSA q | gate split

/-- Index of element `i` of head `h`'s q half inside `q_full`. -/
def qIdx (hd h i : Nat) : Nat := h * (2 * hd) + i

/-- Index of element `i` of head `h`'s gate half inside `q_full`. -/
def gIdx (hd h i : Nat) : Nat := h * (2 * hd) + hd + i

/-- q and gate never share a cell within a head: the halves are contiguous, not interleaved. -/
theorem q_gate_disjoint (hd h i j : Nat) (hi : i < hd) : qIdx hd h i ≠ gIdx hd h j := by
  unfold qIdx gIdx
  omega

/-- Every gate index lies inside `q_full` (`n_head * 2 * head_dim` floats). -/
theorem gIdx_lt (nh hd h i : Nat) (hh : h < nh) (hi : i < hd) : gIdx hd h i < nh * (2 * hd) := by
  unfold gIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
  have h2 : (h + 1) * (2 * hd) = h * (2 * hd) + 2 * hd := by ring
  omega

/-- Every q index lies inside `q_full`. -/
theorem qIdx_lt (nh hd h i : Nat) (hh : h < nh) (hi : i < hd) : qIdx hd h i < nh * (2 * hd) := by
  unfold qIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
  have h2 : (h + 1) * (2 * hd) = h * (2 * hd) + 2 * hd := by ring
  omega

-- § the indexer's pooled keys

/-- The pooled-key row of a cell stays inside the `max_cells / idx_block + 2` rows the state allocates. -/
theorem pooled_idx_lt (pos maxCells b : Nat) (hp : pos < maxCells) : pos / b < maxCells / b + 2 := by
  have h : pos / b ≤ maxCells / b := Nat.div_le_div_right hp.le
  omega

end Strata.Layer
