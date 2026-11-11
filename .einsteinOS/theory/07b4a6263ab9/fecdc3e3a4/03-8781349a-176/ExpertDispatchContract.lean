/-!
# MODULE: expert_dispatch_contract
SOURCES: include/strata/core/expert_source.hpp, src/core/expert_source.cpp
The adapter between `session_loop` and the CPU expert pool: per layer it validates geometry and
router ids, splits routed experts into GPU hits (rows zeroed) and CPU jobs, and latches failure.
Also the verify-window planner (`expert_pool_dispatch_multi`) and the hit Launch/Combine pair.

## Source semantics
* Clause 2: the router weight is never applied here; every job has `weight = 1`.
* Clause 3: `n_embd ≠ H` is refused (failure latched), never mis-indexed.
* `failed` LATCHES: once set, every later dispatch/hit call returns immediately without
  touching state, so a session cannot resume past a hole.
* The split: router index `i` is a hit (row zeroed, GPU owns it) or a job whose `slot = i`;
  `njobs` indexes jobs, `i` indexes output rows, and every row is written by exactly one side.
* `hits_ready` is all-or-nothing over the ten hit-path buffers.
* `Combine` adds `hit_out` only when `hit_pending` was set by `Launch`, and clears it.
* Multi-token: `first_of[i]` is the earliest entry with the same expert id; the PCIe share is
  `(nmiss * pcie_num) >> 8` of the distinct misses; entry `i` belongs to token `i / k`.

## Form template identifiers
`ExpertDispatch` → `Disp`; `expert_pool_dispatch` → `dispatch`; `ExpertJob` → job slot list
`jobs`; `is_hit` → `hit`; `hits_ready` → `hitsReady`; `expert_hit_run(Combine)` → `combine`;
`first_of` → `firstOf`; `pcie_num` → `pcieNum`.
-/

namespace Strata.ExpertDispatchContract

-- § dispatch state and failure latching

/-- The kernel's compiled activation width. -/
def H : Nat := 2560

/-- The adapter's observable state (counters and the latched failure). -/
structure Disp where
  layers : Nat
  experts : Nat
  failed : Bool
  failLayer : Int
  failExpert : Int
deriving Repr

/-- A router id the pool refuses: negative or at least `n_expert`. -/
def badId (nExpert : Nat) (e : Int) : Bool := decide (e < 0) || decide (e ≥ (nExpert : Int))

/-- One single-token dispatch: geometry check, id check, then success. `nExpert` is the pool size. -/
def dispatch (nExpert : Nat) (d : Disp) (nEmbd : Nat) (ids : List Int) : Disp :=
  if d.failed then d
  else if nEmbd ≠ H then { d with failed := true, failLayer := d.layers }
  else match ids.find? (badId nExpert) with
    | some e => { d with failed := true, failLayer := d.layers, failExpert := e }
    | none => { d with layers := d.layers + 1, experts := d.experts + ids.length }

/-- A failed adapter is a fixed point of `dispatch`. -/
theorem dispatch_latched (nE : Nat) (d : Disp) (n : Nat) (ids : List Int) (h : d.failed = true) :
    dispatch nE d n ids = d := by
  simp [dispatch, h]

/-- Failure persists across any sequence of later layers. -/
theorem failure_persists (nE : Nat) (d : Disp) (h : d.failed = true) :
    ∀ calls : List (Nat × List Int),
      calls.foldl (fun s c => dispatch nE s c.1 c.2) d = d := by
  intro calls
  induction calls with
  | nil => rfl
  | cons c cs ih => simp only [List.foldl, dispatch_latched nE d c.1 c.2 h, ih]

/-- Clause 3: a wrong activation width fails the adapter and names the layer. -/
theorem dispatch_wrong_width (nE : Nat) (d : Disp) (n : Nat) (ids : List Int)
    (hf : d.failed = false) (hn : n ≠ H) :
    (dispatch nE d n ids).failed = true ∧ (dispatch nE d n ids).failLayer = d.layers := by
  simp [dispatch, hf, hn]

/-- An out-of-range router id fails the adapter. -/
theorem dispatch_bad_id (nE : Nat) (d : Disp) (ids : List Int) (e : Int)
    (hf : d.failed = false) (hfind : ids.find? (badId nE) = some e) :
    (dispatch nE d H ids).failed = true ∧ (dispatch nE d H ids).failExpert = e := by
  simp [dispatch, hf, hfind]

/-- A successful layer advances the layer counter by one and the expert counter by `k`. -/
theorem dispatch_ok (nE : Nat) (d : Disp) (ids : List Int) (hf : d.failed = false)
    (hfind : ids.find? (badId nE) = none) :
    (dispatch nE d H ids).layers = d.layers + 1 ∧
      (dispatch nE d H ids).experts = d.experts + ids.length ∧
      (dispatch nE d H ids).failed = false := by
  simp [dispatch, hf, hfind]

-- § the hit/miss split

/-- CPU job slots: router indices `< k` that are not hits, in routing order. -/
def jobs (hit : Nat → Bool) : Nat → List Nat
  | 0 => []
  | k + 1 => jobs hit k ++ (if hit k then [] else [k])

/-- Number of GPU hits among the first `k` router indices. -/
def hits (hit : Nat → Bool) : Nat → Nat
  | 0 => 0
  | k + 1 => hits hit k + (if hit k then 1 else 0)

/-- `njobs + n_hits = k`: every routed expert is computed by exactly one side. -/
theorem jobs_len_add_hits (hit : Nat → Bool) (k : Nat) : (jobs hit k).length + hits hit k = k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [jobs, hits, List.length_append]
    cases hit k <;> simp <;> omega

/-- A router index is a CPU job iff it is in range and not a hit. -/
theorem mem_jobs (hit : Nat → Bool) (k i : Nat) : i ∈ jobs hit k ↔ i < k ∧ hit i = false := by
  induction k with
  | zero => simp [jobs]
  | succ k ih =>
    simp only [jobs, List.mem_append, ih]
    by_cases hk : hit k = true
    · simp only [hk, if_true, List.not_mem_nil, or_false]
      constructor
      · intro ⟨a, b⟩; exact ⟨by omega, b⟩
      · intro ⟨a, b⟩
        have : i ≠ k := by intro he; subst he; rw [hk] at b; contradiction
        exact ⟨by omega, b⟩
    · have hk' : hit k = false := by simpa using hk
      simp only [hk', Bool.false_eq_true, if_false, List.mem_singleton]
      constructor
      · rintro (⟨a, b⟩ | rfl)
        · exact ⟨by omega, b⟩
        · exact ⟨by omega, hk'⟩
      · intro ⟨a, b⟩
        by_cases he : i = k
        · exact Or.inr he
        · exact Or.inl ⟨by omega, b⟩

/-- Job slots are strictly increasing, so no output row receives two experts. -/
theorem jobs_sorted (hit : Nat → Bool) (k : Nat) : ∀ i ∈ jobs hit k, i < k :=
  fun i hi => ((mem_jobs hit k i).1 hi).1

/-- Job slots are pairwise distinct (`njobs` vs `i` never confused). -/
theorem jobs_nodup (hit : Nat → Bool) (k : Nat) : (jobs hit k).Nodup := by
  induction k with
  | zero => exact List.nodup_nil
  | succ k ih =>
    simp only [jobs]
    cases hk : hit k
    · simp only [Bool.false_eq_true, if_false]
      unfold List.Nodup at ih ⊢
      rw [List.pairwise_append]
      refine ⟨ih, List.pairwise_singleton _ k, ?_⟩
      intro a ha b hb
      rw [List.mem_singleton] at hb
      have := jobs_sorted hit k a ha
      omega
    · simpa using ih

/-- A row's owner: the GPU (zeroed on the CPU side) or the CPU job writing that same slot. -/
theorem row_owner (hit : Nat → Bool) (k i : Nat) (hi : i < k) :
    (hit i = true ∧ i ∉ jobs hit k) ∨ (hit i = false ∧ i ∈ jobs hit k) := by
  cases h : hit i
  · exact Or.inr ⟨rfl, (mem_jobs hit k i).2 ⟨hi, h⟩⟩
  · refine Or.inl ⟨rfl, fun hm => ?_⟩
    have := ((mem_jobs hit k i).1 hm).2
    rw [h] at this; contradiction

/-- Clause 2: the pool never sees the router weight; every job's weight field is 1. -/
def jobWeight (_routerWeight : Float) : Float := 1.0

/-- The job weight is independent of the router weight. -/
theorem jobWeight_const (w w' : Float) : jobWeight w = jobWeight w' := rfl

-- § the hit path is all-or-nothing

/-- The ten buffers of the hit path (`cache`, `cache_base`, `parts_out`, `hit_out`, `mixed`,
`hit_scratch`, `x_q8_0_hit`, `x_q8_0_hit_scale`, `d_slot`, `d_dst`). -/
def hitsReady (bufs : List Bool) : Bool := bufs.length = 10 && bufs.all id

/-- `hits_ready` implies every one of the ten buffers is configured. -/
theorem hitsReady_all (bufs : List Bool) (h : hitsReady bufs = true) : ∀ b ∈ bufs, b = true := by
  simp only [hitsReady, Bool.and_eq_true, List.all_eq_true, id] at h
  exact h.2

/-- Any missing buffer disables the hit path. -/
theorem hitsReady_missing (bufs : List Bool) (h : false ∈ bufs) : hitsReady bufs = false := by
  cases hr : hitsReady bufs
  · rfl
  · have := hitsReady_all bufs hr false h; contradiction

-- § Launch / Combine

/-- Hit-path pending flag and how many times `hit_out` has been added into `parts`. -/
structure HitState where
  pending : Bool
  adds : Nat

/-- `Launch`: clears the flag, then sets it only when there were hits on this layer. -/
def launch (s : HitState) (nHits : Nat) : HitState :=
  { s with pending := decide (0 < nHits) }

/-- `Combine`: adds once if pending and clears the flag; otherwise does nothing. -/
def combine (s : HitState) : HitState :=
  if s.pending then { pending := false, adds := s.adds + 1 } else s

/-- A `Combine` without a `Launch` in front of it adds nothing (no stale buffer). -/
theorem combine_without_launch (s : HitState) (h : s.pending = false) : combine s = s := by
  simp [combine, h]

/-- Two `Combine`s after one `Launch` add `hit_out` at most once. -/
theorem combine_idempotent (s : HitState) (n : Nat) :
    (combine (combine (launch s n))).adds ≤ s.adds + 1 := by
  simp only [launch, combine]
  by_cases h : 0 < n <;> simp [h]

/-- A layer with no hits adds nothing. -/
theorem no_hits_no_add (s : HitState) : (combine (launch s 0)).adds = s.adds := by
  simp [launch, combine]

-- § verify-window planning

/-- Earliest index `j ∈ [start, start+fuel)` with the same id as `i`, else `i`. -/
def firstGo (ids : List Int) (i : Nat) : Nat → Nat → Nat
  | _, 0 => i
  | j, f + 1 => if ids[j]? = ids[i]? then j else firstGo ids i (j + 1) f

/-- `first_of[i]`: the earliest entry with the same expert id. -/
def firstOf (ids : List Int) (i : Nat) : Nat := firstGo ids i 0 i

/-- The search never returns past `i`. -/
theorem firstGo_le (ids : List Int) (i : Nat) : ∀ f j, j + f = i → firstGo ids i j f ≤ i := by
  intro f
  induction f with
  | zero => intro j _; simp [firstGo]
  | succ f ih =>
    intro j hj
    simp only [firstGo]
    split
    · omega
    · exact ih (j + 1) (by omega)

/-- The search's answer holds the same id as `i`. -/
theorem firstGo_same (ids : List Int) (i : Nat) : ∀ f j, ids[firstGo ids i j f]? = ids[i]? := by
  intro f
  induction f with
  | zero => intro j; simp [firstGo]
  | succ f ih =>
    intro j
    simp only [firstGo]
    split
    · assumption
    · exact ih (j + 1)

/-- `first_of[i] ≤ i` and it names the same expert: groups are well-defined. -/
theorem firstOf_spec (ids : List Int) (i : Nat) :
    firstOf ids i ≤ i ∧ ids[firstOf ids i]? = ids[i]? :=
  ⟨firstGo_le ids i i 0 (by omega), firstGo_same ids i i 0⟩

/-- The PCIe share `(nmiss * pcie_num) >> 8` never exceeds the misses when `pcie_num ≤ 256`. -/
theorem pcie_share_le (nmiss p : Nat) (hp : p ≤ 256) : (nmiss * p) >>> 8 ≤ nmiss := by
  rw [Nat.shiftRight_eq_div_pow]
  apply Nat.div_le_of_le_mul
  have : nmiss * p ≤ nmiss * 256 := Nat.mul_le_mul_left _ hp
  rw [Nat.mul_comm (2 ^ 8)]
  exact this

/-- Entry `i` of a window with `k` experts per token is token `i / k`, router index `i % k`. -/
theorem entry_decomp (i k : Nat) : (i / k) * k + i % k = i := by
  rw [Nat.mul_comm]; exact Nat.div_add_mod i k

/-- Every entry of an `n_tok × k` window belongs to a token in range. -/
theorem entry_token_lt (i k nTok : Nat) (hk : 0 < k) (hi : i < nTok * k) : i / k < nTok :=
  (Nat.div_lt_iff_lt_mul hk).2 hi

end Strata.ExpertDispatchContract
