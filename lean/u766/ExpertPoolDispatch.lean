/-!
# MODULE: expert_pool_dispatch
SOURCES: include/ref/core/expert_source.hpp, src/core/expert_source.cpp
The adapter between the host loop and the CPU expert pool: which router indices become CPU jobs,
that the router weight is applied exactly once (by `moe_combine`, not here), that a failure
latches, and the index arithmetic of the multi-token verify window and its PCIe share.

## Source semantics
* Clause 1: the activation is re-quantized every call (no cache keyed on the `x_f` address).
* Clause 2: `ExpertJob::weight = 1`; `moe_combine` multiplies by the router weight once.
* Clause 3: `n_embd != 2560` is refused; an out-of-range id or a null blob is refused.
* Job index `njobs` and output row `i` differ when there are hits: the jobs are exactly the
  non-hit rows, each once, and never a hit row.
* `failed` LATCHES: once set, dispatch is a no-op, and clearing the message does not re-arm it.
* Multi-token: entry `i` belongs to token `i / k`; the PCIe share `(nmiss * pcie_num) >> 8` never
  exceeds the misses when `pcie_num <= 256`.

## Form template identifiers
`expert_pool_dispatch` → `step`; `ExpertDispatch::failed/fail` → `DState.failed/msg`;
`is_hit` → `hit`; job rows → `jobRows`; hit rows → `hitRows`; `moe_combine` → `combine`;
`pcie_num` share → `pcieShare`.
-/

namespace Ref.ExpertPoolDispatch

-- § Geometry refusal

/-- The width the expert kernel is compiled for. -/
def H : Nat := 2560

/-- Clause 3: the dispatch accepts exactly one activation width. -/
def geometryOk (nEmbd : Nat) : Bool := nEmbd == H

/-- Any other width is refused. -/
theorem geometry_refused {n : Nat} (h : n ≠ 2560) : geometryOk n = false := by
  simp [geometryOk, H, h]

-- § The hit/miss split

/-- Output rows the CPU computes (non-hits). -/
def jobRows (hit : List Bool) : List Nat :=
  (List.range hit.length).filter (fun i => !(hit.getD i false))

/-- Output rows the GPU owns (hits). -/
def hitRows (hit : List Bool) : List Nat :=
  (List.range hit.length).filter (fun i => hit.getD i false)

/-- Helper: a predicate and its negation partition any list. -/
theorem filter_not_add (p : Nat → Bool) : ∀ l : List Nat,
    (l.filter (fun i => !p i)).length + (l.filter p).length = l.length
  | [] => rfl
  | a :: t => by
    have ih := filter_not_add p t
    cases h : p a <;> simp [List.filter_cons, h] <;> omega

/-- Every routed index is either a CPU job or a GPU hit: `njobs + n_hits = k`. -/
theorem jobs_add_hits (hit : List Bool) :
    (jobRows hit).length + (hitRows hit).length = hit.length := by
  have := filter_not_add (fun i => hit.getD i false) (List.range hit.length)
  simpa [jobRows, hitRows] using this

/-- No output row receives two CPU jobs. -/
theorem jobRows_nodup (hit : List Bool) : (jobRows hit).Nodup :=
  List.Nodup.sublist List.filter_sublist List.nodup_range

/-- A hit row is never given a CPU job (its row is zeroed instead). -/
theorem job_not_hit {hit : List Bool} {i : Nat} (h : i ∈ jobRows hit) : i ∉ hitRows hit := by
  simp only [jobRows, hitRows, List.mem_filter, List.mem_range] at h ⊢
  intro h2
  have := h.2
  rw [h2.2] at this
  cases this

/-- With no hits, job index and output row coincide: every row is a job. -/
theorem no_hits_all_jobs (k : Nat) : jobRows (List.replicate k false) = List.range k := by
  simp [jobRows, List.getD_eq_getElem?_getD, List.getElem?_replicate]
  intro i hi
  simp [hi]

-- § Clause 2: the router weight is applied once

/-- `moe_combine`: `sum w[i] * parts[i]`. -/
def combine : List (Int × Int) → Int
  | [] => 0
  | (w, p) :: t => w * p + combine t

/-- The part the pool writes for one expert, scaled by the job's diagnostic weight. -/
def poolPart (jobWeight y : Int) : Int := jobWeight * y

/-- With `ExpertJob::weight = 1` the combine applies each router weight exactly once. -/
theorem weight_applied_once : ∀ (wy : List (Int × Int)),
    combine (wy.map fun p => (p.1, poolPart 1 p.2)) = combine wy
  | [] => rfl
  | (w, y) :: t => by
    simp only [List.map_cons, combine]
    rw [weight_applied_once t]
    simp [poolPart]

/-- Setting the job weight to the router weight would apply it twice (a wrong token, not a fault). -/
theorem double_weight_differs : combine [(2, poolPart 2 1)] ≠ combine [(2, 1)] := by decide

-- § Failure latching

/-- Dispatch state: the latch, its message and the counters. -/
structure DState where
  failed : Bool
  msg : Option String
  layers : Nat
  experts : Nat

/-- What a layer's dispatch ran into. -/
inductive Outcome
  | ok
  | badGeometry
  | badId
  | noBlob

/-- One `expert_pool_dispatch` call. -/
def step (s : DState) (k : Nat) (o : Outcome) : DState :=
  if s.failed then s else
  match o with
  | .ok => { s with layers := s.layers + 1, experts := s.experts + k }
  | .badGeometry => { s with failed := true, msg := some "geometry" }
  | .badId => { s with failed := true, msg := some "id" }
  | .noBlob => { s with failed := true, msg := some "blob" }

/-- A failed adapter does nothing at all. -/
theorem failed_is_fixed {s : DState} (h : s.failed = true) (k : Nat) (o : Outcome) :
    step s k o = s := by
  simp [step, h]

/-- Running a whole sequence of layers. -/
def run (s : DState) (xs : List (Nat × Outcome)) : DState :=
  xs.foldl (fun s p => step s p.1 p.2) s

/-- Once failed, every later layer is a no-op: no resumption into a plausible-looking token. -/
theorem run_after_failure {s : DState} (h : s.failed = true) : ∀ xs, run s xs = s
  | [] => rfl
  | p :: t => by
    simp only [run, List.foldl_cons]
    rw [failed_is_fixed h]
    exact run_after_failure h t

/-- Every non-`ok` outcome sets the latch. -/
theorem bad_outcome_fails (s : DState) (k : Nat) (o : Outcome) (ho : o ≠ .ok) :
    (step s k o).failed = true := by
  unfold step
  split
  · assumption
  · cases o <;> simp_all

/-- Clearing the message does not re-arm a failed adapter. -/
theorem clear_msg_keeps_latch (s : DState) (h : s.failed = true) :
    ({ s with msg := none } : DState).failed = true := h

/-- A healthy layer advances the layer counter by one and the expert counter by `k`. -/
theorem ok_counts (s : DState) (h : s.failed = false) (k : Nat) :
    (step s k .ok).layers = s.layers + 1 ∧ (step s k .ok).experts = s.experts + k := by
  simp [step, h]

-- § Multi-token verify window

/-- An entry `i < n_tok * k` belongs to a token inside the window. -/
theorem entry_token_lt {i nTok k : Nat} (hk : 0 < k) (hi : i < nTok * k) : i / k < nTok :=
  (Nat.div_lt_iff_lt_mul hk).2 hi

/-- An entry's routed index within its token is below `k`. -/
theorem entry_slot_lt {i k : Nat} (hk : 0 < k) : i % k < k := Nat.mod_lt _ hk

/-- Entry decomposition: `i = token * k + j`, the row layout of `out`. -/
theorem entry_decompose (i k : Nat) : i = (i / k) * k + i % k := by
  have := Nat.div_add_mod i k
  rw [Nat.mul_comm] at this
  omega

/-- The PCIe share of the distinct misses, `pcie_num / 256` of them. -/
def pcieShare (nmiss pcieNum : Nat) : Nat := (nmiss * pcieNum) >>> 8

/-- The GPU's PCIe share never exceeds the misses when `pcie_num ≤ 256`. -/
theorem pcieShare_le {nmiss p : Nat} (hp : p ≤ 256) : pcieShare nmiss p ≤ nmiss := by
  unfold pcieShare
  rw [Nat.shiftRight_eq_div_pow]
  have h1 : nmiss * p ≤ nmiss * 256 := Nat.mul_le_mul_left nmiss hp
  have h2 : (2 : Nat) ^ 8 = 256 := rfl
  rw [h2]
  apply Nat.div_le_of_le_mul
  omega

/-- The misses left on the CPU are the complement of the PCIe share. -/
theorem cpu_misses_complement {nmiss p : Nat} (hp : p ≤ 256) :
    (nmiss - pcieShare nmiss p) + pcieShare nmiss p = nmiss := by
  have := pcieShare_le (nmiss := nmiss) hp
  omega

end Ref.ExpertPoolDispatch
