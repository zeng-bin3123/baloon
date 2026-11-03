/-!
# MODULE: expert_dispatch_contract
The adapter contract in `expert_source.hpp`: the all-or-nothing `hits_ready()` predicate, the latching `failed`
flag, the geometry refusal, and the GPU-plan list layout.

## Source semantics
* `hits_ready()` is true only if every one of the nine hit-path pointers is wired.
* `failed` latches: once set, the adapter short-circuits and the layer counter does not advance.
* A mismatched `n_embd` is refused rather than mis-indexed.
* The GPU plan (`GpuPlanSink`) holds at most `cap` entries; group starts are monotone.

## Form template identifiers
`ExpertDispatch::failed` -> failed, `hits_ready` -> hitsReady, `H` -> hWidth, `GpuPlanSink::cap` -> cap.
-/

namespace Strata.ExpertDispatch

-- § hits_ready

/-- Which of the hit-path members are set. -/
structure HitCfg where
  cache : Bool
  cacheBase : Bool
  partsOut : Bool
  hitOut : Bool
  mixed : Bool
  scratch : Bool
  xq : Bool
  xqScale : Bool
  dSlot : Bool
  dDst : Bool

/-- The all-or-nothing readiness predicate. -/
def hitsReady (c : HitCfg) : Bool :=
  c.cache && c.cacheBase && c.partsOut && c.hitOut && c.mixed && c.scratch && c.xq && c.xqScale && c.dSlot && c.dDst

/-- Readiness requires the fp32 activation scales (R4.2h). -/
theorem hitsReady_needs_scales (c : HitCfg) (h : hitsReady c = true) : c.xqScale = true := by
  unfold hitsReady at h
  simp at h
  tauto

/-- Readiness requires the cache itself. -/
theorem hitsReady_needs_cache (c : HitCfg) (h : hitsReady c = true) : c.cache = true := by
  unfold hitsReady at h
  simp at h
  tauto

/-- Readiness requires the device slot and destination lists. -/
theorem hitsReady_needs_lists (c : HitCfg) (h : hitsReady c = true) : c.dSlot = true ∧ c.dDst = true := by
  unfold hitsReady at h
  simp at h
  tauto

-- § the latching failure flag

/-- The flag after a step that may report a new failure. -/
def latch (failed new : Bool) : Bool := failed || new

/-- Once failed, always failed: clearing the message cannot re-arm the adapter. -/
theorem latch_sticky (new : Bool) : latch true new = true := by simp [latch]

/-- A failure is never lost. -/
theorem latch_keeps (failed new : Bool) (h : failed = true) : latch failed new = true := by simp [latch, h]

/-- The layer counter advances only when the adapter has not failed. -/
def stepLayers (failed : Bool) (layers : Nat) : Nat := if failed then layers else layers + 1

/-- After a failure the counter is frozen. -/
theorem stepLayers_frozen (layers : Nat) : stepLayers true layers = layers := by simp [stepLayers]

/-- Without failure the counter advances by one. -/
theorem stepLayers_advance (layers : Nat) : stepLayers false layers = layers + 1 := by simp [stepLayers]

-- § geometry refusal

/-- The kernel accepts only the compiled-in activation width. -/
def geometryOk (nEmbd hWidth : Nat) : Bool := decide (nEmbd = hWidth)

/-- A different width is refused. -/
theorem geometry_refuses (nEmbd hWidth : Nat) (h : nEmbd ≠ hWidth) : geometryOk nEmbd hWidth = false := by
  unfold geometryOk; simp [h]

/-- The compiled width is accepted. -/
theorem geometry_accepts (hWidth : Nat) : geometryOk hWidth hWidth = true := by
  unfold geometryOk; simp

-- § GPU plan layout

/-- Plan group starts are monotone and the entry count is bounded by `cap`. -/
structure PlanState where
  groups : Nat
  entries : Nat
  fetches : Nat

/-- Adding a group that holds `m` entries keeps `entries ≤ cap` when there is room. -/
theorem plan_add_group (st : PlanState) (cap m : Nat) (h : st.entries + m ≤ cap) :
    (PlanState.mk (st.groups + 1) (st.entries + m) st.fetches).entries ≤ cap := by
  simpa using h

/-- A fetch is only issued while below both the staging capacity and the 64-slot arrays. -/
def mayFetch (fetches stagingCap : Nat) : Bool := decide (fetches < stagingCap ∧ fetches < 64)

/-- Fetch count stays below the array size after an allowed fetch. -/
theorem mayFetch_bound (fetches stagingCap : Nat) (h : mayFetch fetches stagingCap = true) :
    fetches + 1 ≤ 64 := by
  unfold mayFetch at h
  simp at h
  omega

end Strata.ExpertDispatch
