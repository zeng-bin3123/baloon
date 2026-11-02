/-!
# MODULE: expert_dispatch_split
The hit/miss split of one layer's routed experts between GPU (VRAM tier / PCIe) and CPU pool.

## Source semantics
Every routed index is exactly one of: hit (GPU computes, CPU row zeroed) or miss (CPU job). `njobs` indexes the job array,
`i` the output rows, and they coincide only without hits. The PCIe share is `(nmiss * pcie_num) >> 8` of the missed
experts, never more than all of them for `pcie_num <= 256`.

## Form template identifiers
- ExpertDispatch::is_hit / host_res -> hit
- expert_pool_dispatch (njobs)      -> missCount
- GpuPlanSink PCIe share            -> pcieShare
-/
namespace Strata.ExpertDispatchSplit

/-- Every id is counted once as a miss job or once as a GPU hit. -/
theorem split_count (hit : Nat → Bool) :
    ∀ ids : List Nat, (ids.filter (fun e => !hit e)).length + (ids.filter hit).length = ids.length
  | [] => rfl
  | e :: t => by
    have ih := split_count hit t
    by_cases h : hit e <;> simp [List.filter_cons, h] <;> omega

/-- Jobs never outnumber routed experts. -/
theorem jobs_le (hit : Nat → Bool) (ids : List Nat) : (ids.filter (fun e => !hit e)).length ≤ ids.length := by
  have := split_count hit ids
  omega

/-- With nothing resident every routed expert is a job, so job index equals output index. -/
theorem no_hits_all_jobs (ids : List Nat) : (ids.filter (fun _ => !false)).length = ids.length := by
  simp

def pcieShare (nmiss pcie_num : Nat) : Nat := nmiss * pcie_num / 256

/-- The PCIe share never exceeds the number of missed experts when `pcie_num ≤ 256`. -/
theorem pcieShare_le {nmiss p : Nat} (hp : p ≤ 256) : pcieShare nmiss p ≤ nmiss := by
  unfold pcieShare
  have := Nat.mul_le_mul_left nmiss hp
  omega

/-- The miss ranks handed to PCIe are `[nmiss - m, nmiss)`, exactly `m` of them. -/
theorem pcie_rank_count {nmiss m : Nat} (h : m ≤ nmiss) : nmiss - (nmiss - m) = m := by omega

/-- With `pcie_num = 0` nothing goes over PCIe. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by simp [pcieShare]

end Strata.ExpertDispatchSplit
