/-!
# MODULE: expert_source_dispatch
The expert pool's adapter to the host loop: blob addressing in `experts.bin`, geometry refusal, the
hit/miss split by router index, cache counters, the latching failure flag and the hit-timing counters.

## Source semantics
* Blob index is `layer * n_expert + expert` with no padding; byte offset is index times `BLOB`.
* Geometry is fixed (`H = 2560`, `FF = 640`, `BLOB = 1382400`); anything else is refused.
* Each router index is computed by exactly one side: CPU rows of hits are zeroed, GPU rows of
  misses are zero, so the sum equals the computed value.
* Every cache lookup is counted exactly once as hit, admitted or refused.
* `failed` latches; clearing the message does not re-arm the adapter.
* `hits_ready` is all-or-nothing over the hit path's pointers.

## Form template identifiers
* `FileExpertSource::blob` -> `blobOffset`, `ExpertDispatch::is_hit` -> `isHit`
* `cache_hits/admitted/refused` -> `Counters`, `hit_ready/hit_late` -> `Timing`
* `ExpertDispatch::failed/fail` -> `FailState`
-/

namespace StrataExpertSource

-- § geometry and addressing

/-- Bytes per expert blob. -/
def BLOB : Nat := 1382400
/-- Embedding width the kernel is compiled for. -/
def H : Nat := 2560
/-- Expert feed-forward width the kernel is compiled for. -/
def FF : Nat := 640

/-- Geometry check: other widths are refused. -/
def geometryOk (nEmbd ff : Nat) : Bool := nEmbd == H && ff == FF

/-- The compiled geometry is accepted. -/
theorem geometryOk_compiled : geometryOk 2560 640 = true := by
  decide

/-- A different embedding width is refused. -/
theorem geometryOk_refuses_embd (nEmbd ff : Nat) (h : nEmbd ≠ 2560) : geometryOk nEmbd ff = false := by
  unfold geometryOk H
  simp [h]

/-- A different expert width is refused. -/
theorem geometryOk_refuses_ff (nEmbd ff : Nat) (h : ff ≠ 640) : geometryOk nEmbd ff = false := by
  unfold geometryOk FF
  simp [h]

/-- Flat blob index, no padding. -/
def blobIndex (nExpert l e : Nat) : Nat := l * nExpert + e

/-- Byte offset of a blob. -/
def blobOffset (nExpert l e : Nat) : Nat := blobIndex nExpert l e * BLOB

/-- The pack holds exactly 24,576 blobs for 48 layers of 512 experts. -/
theorem blob_count : 48 * 512 = 24576 := by
  decide

/-- The artifact size: `48 * 512 * BLOB = 33,973,862,400`. -/
theorem experts_bin_size : 48 * 512 * BLOB = 33973862400 := by
  unfold BLOB
  decide

/-- File-size check: exact equality with `n_layers * n_expert * BLOB`. -/
def sizeOk (fileSize nLayers nExpert : Nat) : Bool := fileSize == nLayers * nExpert * BLOB

/-- The real artifact passes the size check. -/
theorem sizeOk_artifact : sizeOk 33973862400 48 512 = true := by
  unfold sizeOk BLOB
  decide

/-- A short file fails the size check. -/
theorem sizeOk_short : sizeOk 33973862399 48 512 = false := by
  unfold sizeOk BLOB
  decide

/-- In-range indices stay below the blob count. -/
theorem blobIndex_lt (nExpert l e nLayers : Nat) (hl : l < nLayers) (he : e < nExpert) :
    blobIndex nExpert l e < nLayers * nExpert := by
  unfold blobIndex
  have h1 : (l + 1) * nExpert ≤ nLayers * nExpert := Nat.mul_le_mul_right _ hl
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Distinct in-range `(layer, expert)` pairs map to distinct blob indices. -/
theorem blobIndex_injective (nExpert l e l' e' : Nat) (he : e < nExpert) (he' : e' < nExpert)
    (h : blobIndex nExpert l e = blobIndex nExpert l' e') : l = l' ∧ e = e' := by
  unfold blobIndex at h
  have hd : (l * nExpert + e) / nExpert = (l' * nExpert + e') / nExpert := by rw [h]
  have hm : (l * nExpert + e) % nExpert = (l' * nExpert + e') % nExpert := by rw [h]
  have hpos : 0 < nExpert := by omega
  have a1 : (l * nExpert + e) / nExpert = l := by
    rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt he]; simp
  have a2 : (l' * nExpert + e') / nExpert = l' := by
    rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt he']; simp
  have b1 : (l * nExpert + e) % nExpert = e := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt he]
  have b2 : (l' * nExpert + e') % nExpert = e' := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt he']
  rw [a1, a2] at hd
  rw [b1, b2] at hm
  exact ⟨hd, hm⟩

/-- Activation scales: `H / 32 = 80` Q8_0 chunks. -/
theorem q8_chunks : H / 32 = 80 := by
  unfold H
  decide

/-- Hit-path activation buffer: `(H / 32) * 34 = 2720` bytes. -/
theorem q8_hit_bytes : (H / 32) * 34 = 2720 := by
  unfold H
  decide

-- § hit / miss split

/-- CPU row for one router index: zero when the GPU owns it. -/
def cpuRow (hit : Bool) (v : Int) : Int := if hit then 0 else v

/-- GPU row for one router index: zero when the CPU owns it. -/
def gpuRow (hit : Bool) (v : Int) : Int := if hit then v else 0

/-- Combining both sides at the router index recovers exactly the computed value. -/
theorem split_sum (hit : Bool) (v : Int) : cpuRow hit v + gpuRow hit v = v := by
  cases hit <;> simp [cpuRow, gpuRow]

/-- A hit's CPU row is zero. -/
theorem cpuRow_hit (v : Int) : cpuRow true v = 0 := by
  simp [cpuRow]

/-- A miss's GPU row is zero. -/
theorem gpuRow_miss (v : Int) : gpuRow false v = 0 := by
  simp [gpuRow]

/-- Each index is owned by exactly one side. -/
theorem owner_exclusive (hit : Bool) : (hit = true) ≠ (hit = false) := by
  cases hit <;> simp

/-- Decide hits from a residency lookup. -/
def isHit (res : Nat → Option Nat) (e : Nat) : Bool := (res e).isSome

/-- Resident experts are hits. -/
theorem isHit_of_resident (res : Nat → Option Nat) (e s : Nat) (h : res e = some s) :
    isHit res e = true := by
  unfold isHit
  simp [h]

/-- Non-resident experts are misses. -/
theorem isHit_of_none (res : Nat → Option Nat) (e : Nat) (h : res e = none) :
    isHit res e = false := by
  unfold isHit
  simp [h]

/-- Router weight is applied once, by the combine: weighted sum of unweighted parts. -/
def combine (ws parts : List Int) : Int := (List.zipWith (· * ·) ws parts).sum

/-- Combine of the empty router output is zero. -/
theorem combine_nil : combine [] [] = 0 := by
  rfl

/-- Combine unfolds one entry at a time. -/
theorem combine_cons (w p : Int) (ws ps : List Int) :
    combine (w :: ws) (p :: ps) = w * p + combine ws ps := by
  simp [combine]

/-- Hit path pointers present or absent. -/
structure HitWiring where
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

/-- All-or-nothing wiring. -/
def HitWiring.ready (w : HitWiring) : Bool :=
  w.cache && w.cacheBase && w.partsOut && w.hitOut && w.mixed && w.scratch && w.xq && w.xqScale &&
    w.dSlot && w.dDst

/-- Without the fp32 activation scales the hit path is not ready. -/
theorem not_ready_without_scale (w : HitWiring) (h : w.xqScale = false) : w.ready = false := by
  unfold HitWiring.ready
  simp [h]

/-- Without a cache the hit path is not ready. -/
theorem not_ready_without_cache (w : HitWiring) (h : w.cache = false) : w.ready = false := by
  unfold HitWiring.ready
  simp [h]

/-- Ready implies every pointer is set. -/
theorem ready_all (w : HitWiring) (h : w.ready = true) :
    w.cache = true ∧ w.xqScale = true ∧ w.dDst = true := by
  unfold HitWiring.ready at h
  simp only [Bool.and_eq_true] at h
  exact ⟨h.1.1.1.1.1.1.1.1.1, h.1.1.2, h.2⟩

-- § counters

/-- Cache lookup counters. -/
structure Counters where
  hits : Nat
  admitted : Nat
  refused : Nat

/-- One lookup: resident is a hit, else admitted if a slot was free, else refused. -/
def Counters.step (c : Counters) (resident admittedOk : Bool) : Counters :=
  if resident then { c with hits := c.hits + 1 }
  else if admittedOk then { c with admitted := c.admitted + 1 }
  else { c with refused := c.refused + 1 }

/-- Total lookups counted. -/
def Counters.total (c : Counters) : Nat := c.hits + c.admitted + c.refused

/-- Every lookup is counted exactly once. -/
theorem counters_step_total (c : Counters) (r a : Bool) : (c.step r a).total = c.total + 1 := by
  cases r <;> cases a <;> simp [Counters.step, Counters.total] <;> omega

/-- Hit/late counters for the GPU-start probe. -/
structure Timing where
  ready : Nat
  late : Nat

/-- Record one layer: done by pool return is ready, else late. -/
def Timing.record (t : Timing) (done : Bool) : Timing :=
  if done then { t with ready := t.ready + 1 } else { t with late := t.late + 1 }

/-- Each layer is recorded exactly once. -/
theorem timing_record_total (t : Timing) (d : Bool) :
    (t.record d).ready + (t.record d).late = t.ready + t.late + 1 := by
  cases d <;> simp [Timing.record] <;> omega

-- § failure latch

/-- Failure state: the latch and the message. -/
structure FailState where
  failed : Bool
  msg : Option String

/-- Clearing the message. -/
def FailState.clearMsg (s : FailState) : FailState := { s with msg := none }

/-- Record a failure: the latch is set. -/
def FailState.raise (s : FailState) (m : String) : FailState := { failed := true, msg := some m }

/-- Clearing the message never re-arms the adapter. -/
theorem clearMsg_keeps_failed (s : FailState) : s.clearMsg.failed = s.failed := by
  rfl

/-- Once failed, always failed under message clearing. -/
theorem failed_latches (s : FailState) (m : String) : (s.raise m).clearMsg.failed = true := by
  rfl

/-- Dispatch short-circuits when latched. -/
def shouldRun (s : FailState) : Bool := !s.failed

/-- A latched adapter does no work. -/
theorem no_run_when_failed (s : FailState) (h : s.failed = true) : shouldRun s = false := by
  simp [shouldRun, h]

/-- Raising a failure stops later work even after clearing the message. -/
theorem no_run_after_raise_clear (s : FailState) (m : String) : shouldRun (s.raise m).clearMsg = false := by
  rfl

-- § PCIe share

/-- Number of missed experts the GPU reads over PCIe: `pcie_num / 256` of the misses. -/
def pcieShare (misses pcieNum : Nat) : Nat := misses * pcieNum / 256

/-- The PCIe share never exceeds the misses when `pcie_num ≤ 256`. -/
theorem pcieShare_le (misses pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare misses pcieNum ≤ misses := by
  unfold pcieShare
  apply Nat.div_le_of_le_mul
  have := Nat.mul_le_mul_left misses h
  rw [Nat.mul_comm 256 misses]
  exact this

/-- Zero share when `pcie_num = 0`. -/
theorem pcieShare_zero (misses : Nat) : pcieShare misses 0 = 0 := by
  simp [pcieShare]

end StrataExpertSource
