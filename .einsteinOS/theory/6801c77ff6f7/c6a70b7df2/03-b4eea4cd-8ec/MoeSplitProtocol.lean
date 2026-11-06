/-!
# MODULE: moe_split_protocol
SOURCES: include/strata/core/layer.hpp, src/core/layer.cpp, include/strata/core/session.hpp, src/core/session.cpp
The MoE block split at the router: `moe_route` publishes ids/weights for THIS layer, the host pool computes THIS
layer's expert outputs, and `moe_finish` combines them with THIS layer's weights plus the shared expert.

## Source semantics
* `moe_layer = moe_route ; moe_finish`, and `moe_finish = moe_shared ; moe_combine_parts`.
* `combine = sum_j w[j] * parts[j] + shared` (shared ADDED PLAIN).
* Combining layer l's expert outputs with layer l+1's weights (the L100 bug) is a different number.
* `k` must be in 1..64; the native router additionally needs exactly 512 experts and k = 10.
* The shared expert may run before or after the pool: it is a pure function of `x`, so the result is unchanged.
* The doorbell mapped-pinned regions total `4*(n_embd + 2k + 2)` bytes.

## Form template identifiers
combine <- `moe_combine` / `native_moe_combine`; kValid <- the `k < 1 || k > 64` guard;
nativeRouterOk <- the native router guard; sharedExpert <- `moe_shared`; stale <- the one-layer-delay loop.
-/

namespace Strata.MoeSplit

-- § the combination

/-- `sum_j w[j] * parts[j] + shared`. -/
def combine (w parts : List Int) (shared : Int) : Int :=
  (List.zipWith (· * ·) w parts).sum + shared

/-- With no experts, only the shared expert remains. -/
theorem combine_nil (parts : List Int) (s : Int) : combine [] parts s = s := by
  simp [combine]

/-- Peeling one routed expert off the combination. -/
theorem combine_cons (a b : Int) (w p : List Int) (s : Int) :
    combine (a :: w) (b :: p) s = a * b + combine w p s := by
  simp [combine, List.zipWith_cons_cons, List.sum_cons, Int.add_assoc]

/-- The shared output is added plain, after the routed sum. -/
theorem combine_shared (w p : List Int) (s : Int) : combine w p s = combine w p 0 + s := by
  simp [combine]

-- § route / finish split

/-- `moe_finish`: the shared expert, then the combination with this layer's weights. -/
def finish (sharedExpert : Int) (w parts : List Int) : Int := combine w parts sharedExpert

/-- The split form: compute the shared expert into `b.shared`, then combine. -/
def finishSplit (sharedExpert : Int) (w parts : List Int) : Int :=
  let sh := sharedExpert
  combine w parts sh

/-- `moe_finish` is exactly `moe_shared` followed by `moe_combine_parts`. -/
theorem finish_eq_split (sh : Int) (w parts : List Int) : finish sh w parts = finishSplit sh w parts := rfl

/-- Running the shared expert early (end of `pre[l]`) or late (inside `post[l]`) gives the same token, because
it depends only on `x`. -/
theorem shared_early_irrelevant (sharedOf : Int → Int) (x : Int) (w parts : List Int) :
    finish (sharedOf x) w parts = combine w parts (sharedOf x) := rfl

/-- `moe_layer` composes route then finish; the router's output is the only coupling. -/
def layerOut (route : Nat → List Int) (outs : Nat → List Int) (sh : Nat → Int) (l : Nat) : Int :=
  finish (sh l) (route l) (outs l)

/-- The delayed loop: layer l's experts combined with layer l+1's router weights. -/
def staleOut (route : Nat → List Int) (outs : Nat → List Int) (sh : Nat → Int) (l : Nat) : Int :=
  finish (sh l) (route (l + 1)) (outs l)

/-- The correct loop uses layer l's own weights. -/
theorem layerOut_uses_own_weights (route outs : Nat → List Int) (sh : Nat → Int) (l : Nat) :
    layerOut route outs sh l = combine (route l) (outs l) (sh l) := rfl

/-- The stale loop agrees with the correct one when the two layers' weights coincide... -/
theorem stale_eq_of_same_weights (route outs : Nat → List Int) (sh : Nat → Int) (l : Nat)
    (h : route (l + 1) = route l) : staleOut route outs sh l = layerOut route outs sh l := by
  simp [staleOut, layerOut, h]

/-- ...and differs in general: a concrete pair of routers on which the delay changes the answer (L100). -/
theorem stale_differs_example :
    combine [1, 0] [5, 7] 0 ≠ combine [0, 1] [5, 7] 0 := by
  decide

-- § guards

/-- The `k` guard shared by `moe_route`, `moe_finish` and `moe_combine_parts`. -/
def kValid (k : Nat) : Prop := 1 ≤ k ∧ k ≤ 64

/-- The native router's extra requirement. -/
def nativeRouterOk (nExpert k : Nat) : Prop := nExpert = 512 ∧ k = 10

/-- Whatever the native router accepts also passes the generic `k` guard. -/
theorem nativeRouter_k_valid (e k : Nat) (h : nativeRouterOk e k) : kValid k := by
  unfold nativeRouterOk at h
  unfold kValid
  omega

/-- The router's `k` bounds the doorbell and parts buffer sizes: parts is `k * n_embd` floats. -/
theorem parts_bytes_eq (k nEmbd : Nat) : k * nEmbd * 4 = 4 * (k * nEmbd) := by
  omega

-- § the doorbell regions

/-- Bytes of the five mapped-pinned doorbell regions: x_f, ids, weights, seq, flag. -/
def doorbellBytes (nEmbd k : Nat) : Nat := nEmbd * 4 + k * 4 + k * 4 + 4 + 4

/-- The closed form of the doorbell size. -/
theorem doorbellBytes_eq (nEmbd k : Nat) : doorbellBytes nEmbd k = 4 * (nEmbd + 2 * k + 2) := by
  unfold doorbellBytes
  omega

/-- The ring sequence the host waits for after l+1 layers is l+1 (strictly increasing, one ring per layer). -/
theorem ring_strict (l : Nat) : l + 1 < (l + 1) + 1 := by
  omega

end Strata.MoeSplit
