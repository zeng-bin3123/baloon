/-!
# MODULE: expert_source_dispatch
SOURCES: include/strata/core/expert_source.hpp
The adapter between the host loop and the expert pool: where an expert blob's bytes come from (file-mapped or arena),
the GPU-hit / CPU-miss split by router index, the all-or-nothing hit-path wiring, and the latching failure flag.

## Source semantics
* `experts.bin` holds exactly `48 x 512` blobs of 1,382,400 bytes, indexed `layer*512 + expert`, no padding; a file of
  any other size is refused.
* A routed index is computed by exactly one of CPU (miss) or GPU (hit): the CPU row of a hit is zeroed and the hit
  row of a miss is zero, so the combined row equals the expert's output exactly once.
* `hits_ready` is all-or-nothing, and in particular requires the fp32 activation-scale buffer (R4.2h).
* `failed` latches: clearing the message never re-arms the adapter.
* The PCIe share of a verify window is `pcie_num/256` of the missed experts, never more than all of them.

## Form template identifiers
FileExpertSource::blob -> blobIndex ; kBlob -> BLOB ; ExpertDispatch::is_hit -> hit flags ;
ExpertDispatch::hits_ready -> HitWiring.ready ; ExpertDispatch::failed -> Failure.failed ; pcie_num -> pcieNum
-/

namespace Strata.ExpertSource

-- § geometry and file indexing

/-- Bytes per expert blob. -/
def BLOB : Nat := 1382400
/-- Index of `(layer, expert)` inside `experts.bin`. -/
def blobIndex (l e : Nat) : Nat := l * 512 + e

/-- The file holds exactly 24,576 blobs. -/
theorem blob_count : 48 * 512 = 24576 := by decide

/-- The 34 GB expert file size quoted in the header. -/
theorem experts_file_size : 48 * 512 * 1382400 = 33973862400 := by decide

/-- Indices stay inside the file. -/
theorem blobIndex_lt (l e : Nat) (hl : l < 48) (he : e < 512) : blobIndex l e < 24576 := by
  unfold blobIndex; omega

/-- Distinct pairs have distinct indices (no padding, no aliasing). -/
theorem blobIndex_inj (l e l' e' : Nat) (he : e < 512) (he' : e' < 512)
    (h : blobIndex l e = blobIndex l' e') : l = l' ∧ e = e' := by
  unfold blobIndex at h; omega

/-- Every blob ends inside the file, so a mapped read cannot run past the end. -/
theorem blob_end_le_size (l e : Nat) (hl : l < 48) (he : e < 512) :
    (blobIndex l e + 1) * 1382400 ≤ 33973862400 := by
  unfold blobIndex; omega

/-- The size check of `FileExpertSource::open`. -/
def fileAccepts (size nLayers nExpert : Nat) : Bool := decide (size = nLayers * nExpert * BLOB)

/-- A short file is refused. -/
theorem short_file_refused (size nLayers nExpert : Nat) (h : size < nLayers * nExpert * BLOB) :
    fileAccepts size nLayers nExpert = false := by
  unfold fileAccepts
  simp
  omega

/-- An over-long file is refused. -/
theorem long_file_refused (size nLayers nExpert : Nat) (h : nLayers * nExpert * BLOB < size) :
    fileAccepts size nLayers nExpert = false := by
  unfold fileAccepts
  simp
  omega

/-- The 663.6 MB of expert bytes per token: 48 layers x 10 routed experts. -/
theorem bytes_per_token : 48 * 10 * 1382400 = 663552000 := by decide

/-- The hit path's activation buffer: 80 blocks of 34 bytes. -/
theorem q8_buffer_bytes : 2560 / 32 * 34 = 2720 := by decide

/-- A Q8_0 block is a 2-byte scale plus 32 quants. -/
theorem q8_block_bytes : 2 + 32 = 34 := by decide

-- § the hit/miss split

/-- CPU contribution of one router index: zero for a hit. -/
def cpuPart (hit : Bool) (v : Int) : Int := if hit then 0 else v
/-- GPU contribution of one router index: zero for a miss. -/
def gpuPart (hit : Bool) (v : Int) : Int := if hit then v else 0

/-- Each routed expert is counted exactly once, whichever side computed it. -/
theorem split_partition (hit : Bool) (v : Int) : cpuPart hit v + gpuPart hit v = v := by
  cases hit <;> simp [cpuPart, gpuPart]

/-- Weighted combine over router indices, as `moe_combine` does on the device. -/
def combine : List Int → List Int → Int
  | w :: ws, p :: ps => w * p + combine ws ps
  | _, _ => 0

/-- The split never changes the combined output: summing CPU and GPU rows weights each expert once. -/
theorem combine_split (hits : List Bool) (ws vs : List Int) (hlen : hits.length = vs.length) :
    combine ws ((hits.zip vs).map (fun x => cpuPart x.1 x.2)) +
      combine ws ((hits.zip vs).map (fun x => gpuPart x.1 x.2)) =
    combine ws ((hits.zip vs).map (fun x => x.2)) := by
  clear hlen
  induction hits generalizing ws vs with
  | nil => cases ws <;> simp [combine]
  | cons h hs ih =>
    cases ws with
    | nil => simp [combine]
    | cons w ws =>
      cases vs with
      | nil => simp [combine]
      | cons v vs =>
        simp only [List.zip_cons_cons, List.map_cons, combine]
        have := ih ws vs
        have e := split_partition h v
        calc w * cpuPart h v + combine ws (List.map (fun x => cpuPart x.1 x.2) (hs.zip vs)) +
              (w * gpuPart h v + combine ws (List.map (fun x => gpuPart x.1 x.2) (hs.zip vs)))
            = w * (cpuPart h v + gpuPart h v) +
              (combine ws (List.map (fun x => cpuPart x.1 x.2) (hs.zip vs)) +
               combine ws (List.map (fun x => gpuPart x.1 x.2) (hs.zip vs))) := by
                rw [Int.mul_add]; omega
          _ = w * v + combine ws (List.map (fun x => x.2) (hs.zip vs)) := by
                rw [e, this]

-- § hit-path wiring

/-- Which pieces of the hit path are configured. -/
structure HitWiring where
  cache : Bool
  cacheBase : Bool
  partsOut : Bool
  hitOut : Bool
  mixed : Bool
  scratch : Bool
  xq8 : Bool
  xq8Scale : Bool
  dSlot : Bool
  dDst : Bool

/-- All of it or none of it. -/
def HitWiring.ready (w : HitWiring) : Bool :=
  w.cache && w.cacheBase && w.partsOut && w.hitOut && w.mixed && w.scratch && w.xq8 && w.xq8Scale &&
    w.dSlot && w.dDst

/-- The fp32 activation scales are required: without them the GPU hits use fp16 scales and the tokens change. -/
theorem ready_requires_scale (w : HitWiring) (h : w.ready = true) : w.xq8Scale = true := by
  unfold HitWiring.ready at h
  simp only [Bool.and_eq_true] at h
  exact h.1.1.2

/-- A missing device staging buffer disables the hit path (never half-configured). -/
theorem not_ready_of_missing_slot (w : HitWiring) (h : w.dSlot = false) : w.ready = false := by
  unfold HitWiring.ready
  simp [h]

-- § failure latch

/-- Failure state: the flag latches, the message is only text. -/
structure Failure where
  failed : Bool
  msg : Option String

/-- Record a failure. -/
def Failure.record (f : Failure) (m : String) : Failure := { failed := true, msg := some m }
/-- Clear the message only. -/
def Failure.clearMsg (f : Failure) : Failure := { f with msg := none }

/-- Clearing the message does not re-arm the adapter. -/
theorem clearMsg_keeps_failed (f : Failure) : f.clearMsg.failed = f.failed := rfl

/-- Once failed, always failed after recording. -/
theorem record_failed (f : Failure) (m : String) : (f.record m).failed = true := rfl

-- § PCIe share

/-- Number of missed experts the GPU reads over PCIe: `misses * pcie_num / 256`. -/
def pcieShare (misses pcieNum : Nat) : Nat := misses * pcieNum / 256

/-- The PCIe share never exceeds the misses when `pcie_num <= 256`. -/
theorem pcieShare_le (misses pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare misses pcieNum ≤ misses := by
  unfold pcieShare
  apply Nat.div_le_of_le_mul
  have := Nat.mul_le_mul_left misses h
  omega

end Strata.ExpertSource
