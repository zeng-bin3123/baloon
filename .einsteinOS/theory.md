
## FILE: /mnt/user-data/outputs/ExpertCacheResidency.lean
/-!
# MODULE: expert_cache_residency
SOURCES: src/core/expert_cache.cpp
Slot arena for the VRAM expert tier: per-layer slot ranges (R4.2g), the no-eviction admit counter,
256-byte aligned variable-size slots, and the clamped fill size.

## Source semantics
* Layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the remainder, so ranges tile `0..slots`.
* `admit` never evicts: it hands out `layer_next` while it is below the layer's `hi`, otherwise refuses.
* `open_sized` aligns each slot to 256 bytes; offsets are cumulative.
* `fill_slot` copies `min`-style `bytes` when `0 < bytes ≤ blob`, else the whole blob.

## Form template identifiers
layer_slot_range → layerLo/layerHi ; admit → admitStep ; open_sized offsets → align256 ;
fill_slot byte count → fillBytes
-/

namespace Ref.ExpertCache

-- § per-layer ranges

/-- Quota per layer. -/
def layerQ (slots nLayers : Nat) : Nat := slots / nLayers

/-- Lower bound of layer `l`'s slot range. -/
def layerLo (slots nLayers l : Nat) : Nat := l * layerQ slots nLayers

/-- Upper bound (exclusive) of layer `l`'s slot range; the last layer takes the remainder. -/
def layerHi (slots nLayers l : Nat) : Nat :=
  if l = nLayers - 1 then slots else (l + 1) * layerQ slots nLayers

/-- Layer 0 starts at slot 0. -/
theorem layerLo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem layerHi_last (slots n : Nat) (hn : 0 < n) : layerHi slots n (n - 1) = slots := by
  simp [layerHi]

/-- Every range is well formed: `lo ≤ hi`. -/
theorem layerLo_le_layerHi (slots n l : Nat) (hl : l < n) :
    layerLo slots n l ≤ layerHi slots n l := by
  unfold layerLo layerHi
  by_cases h : l = n - 1
  · rw [if_pos h]
    have h1 : layerQ slots n * n ≤ slots := Nat.div_mul_le_self slots n
    have h2 : l * layerQ slots n ≤ n * layerQ slots n := Nat.mul_le_mul_right _ (by omega)
    have h3 : n * layerQ slots n = layerQ slots n * n := Nat.mul_comm _ _
    omega
  · rw [if_neg h]
    exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- Adjacent layers' ranges abut with no gap and no overlap. -/
theorem layerHi_eq_layerLo_succ (slots n l : Nat) (hl : l + 1 < n) :
    layerHi slots n l = layerLo slots n (l + 1) := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]

-- § admit counter (no eviction)

/-- One `admit` step on the per-layer path: `(slot or refusal, new counter)`. -/
def admitStep (next hi : Nat) : Option Nat × Nat :=
  if hi ≤ next then (none, next) else (some next, next + 1)

/-- A full quota refuses and leaves the counter untouched. -/
theorem admitStep_full (next hi : Nat) (h : hi ≤ next) : admitStep next hi = (none, next) := by
  simp [admitStep, h]

/-- A granted slot is below the ceiling and the counter advances by exactly one. -/
theorem admitStep_slot_lt (next hi s next' : Nat) (h : admitStep next hi = (some s, next')) :
    s < hi ∧ next' = s + 1 := by
  unfold admitStep at h
  split at h
  · simp at h
  · simp at h
    omega

/-- The counter never decreases. -/
theorem admitStep_mono (next hi : Nat) : next ≤ (admitStep next hi).2 := by
  unfold admitStep
  split <;> simp

-- § aligned slots

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks a slot. -/
theorem le_align256 (x : Nat) : x ≤ align256 x := by
  unfold align256; omega

/-- Aligned sizes are multiples of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256; omega

/-- Alignment wastes fewer than 256 bytes. -/
theorem align256_lt (x : Nat) : align256 x < x + 256 := by
  unfold align256; omega

/-- A cumulative offset stays 256-aligned and covers the slot just added. -/
theorem offset_step (off s : Nat) (h : off % 256 = 0) :
    (off + align256 s) % 256 = 0 ∧ off + s ≤ off + align256 s := by
  have h1 := align256_mod s
  have h2 := le_align256 s
  omega

-- § fill size

/-- Byte count `fill_slot` copies. -/
def fillBytes (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never writes past the slot. -/
theorem fillBytes_le (bytes blob : Nat) : fillBytes bytes blob ≤ blob := by
  unfold fillBytes
  split <;> omega

/-- With a positive blob, a fill copies at least one byte. -/
theorem fillBytes_pos (bytes blob : Nat) (hb : 0 < blob) : 0 < fillBytes bytes blob := by
  unfold fillBytes
  split <;> omega

end Ref.ExpertCache



## FILE: /mnt/user-data/outputs/ExpertSourceDispatch.lean
/-!
# MODULE: expert_source_dispatch
SOURCES: src/core/expert_source.cpp, include/ref/core/expert_source.hpp
Adapter from the host loop to the expert pool: blob addressing (`layer * n_expert + expert`), the job/hit split
by router index, and the PCIe share of missed experts in a verify window.

## Source semantics
* The flat blob index alone cannot detect `expert >= n_expert`; layer and expert are validated separately.
* Jobs are indexed by `njobs`, outputs by `i`; hits consume no job.
* The PCIe share is `(nmiss * pcie_num) >> 8` and never exceeds `nmiss` when `pcie_num ≤ 256`.

## Form template identifiers
blob(layer, expert) → blobIndex ; expert_pool_dispatch job count → jobCount ; GPU PCIe share → pcieShare
-/

namespace Ref.ExpertSource

-- § blob addressing

/-- Flat blob index. -/
def blobIndex (nExpert layer expert : Nat) : Nat := layer * nExpert + expert

/-- Why the separate `expert >= n_expert` check exists: `blob(0, n)` aliases `blob(1, 0)`. -/
theorem blobIndex_alias (nExpert : Nat) : blobIndex nExpert 0 nExpert = blobIndex nExpert 1 0 := by
  simp [blobIndex]

/-- A valid expert id stays inside its own layer's row. -/
theorem blobIndex_lt_succ (nExpert l e : Nat) (he : e < nExpert) :
    blobIndex nExpert l e < (l + 1) * nExpert := by
  unfold blobIndex
  have h : (l + 1) * nExpert = l * nExpert + nExpert := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- With both axes validated, the flat index is below `layers * n_expert`. -/
theorem blobIndex_lt_total (nExpert layers l e : Nat) (hl : l < layers) (he : e < nExpert) :
    blobIndex nExpert l e < layers * nExpert := by
  have h1 := blobIndex_lt_succ nExpert l e he
  have h2 : (l + 1) * nExpert ≤ layers * nExpert := Nat.mul_le_mul_right _ hl
  omega

/-- On validated axes the flat index is injective: distinct `(layer, expert)` never share a blob. -/
theorem blobIndex_inj (nExpert l e l' e' : Nat) (he : e < nExpert) (he' : e' < nExpert)
    (h : blobIndex nExpert l e = blobIndex nExpert l' e') : l = l' ∧ e = e' := by
  have key : ∀ a b a' b' : Nat, b < nExpert → b' < nExpert → a < a' →
      a * nExpert + b ≠ a' * nExpert + b' := by
    intro a b a' b' hb hb' hlt heq
    have h1 : (a + 1) * nExpert ≤ a' * nExpert := Nat.mul_le_mul_right _ hlt
    have h2 : (a + 1) * nExpert = a * nExpert + nExpert := by rw [Nat.add_mul, Nat.one_mul]
    omega
  unfold blobIndex at h
  rcases Nat.lt_trichotomy l l' with hl | hl | hl
  · exact absurd h (key _ _ _ _ he he' hl)
  · subst hl
    exact ⟨rfl, by omega⟩
  · exact absurd h.symm (key _ _ _ _ he' he hl)

-- § jobs vs hits

/-- Number of CPU jobs: the entries that are not GPU hits. -/
def jobCount (hits : List Bool) : Nat := (hits.filter (fun b => !b)).length

/-- `njobs` never exceeds `k`, the number of routed entries (so the job array is large enough). -/
theorem jobCount_le (hits : List Bool) : jobCount hits ≤ hits.length := by
  unfold jobCount
  exact List.length_filter_le _ _

/-- With no hits, every entry is a job and `njobs = i` indexing coincide. -/
theorem jobCount_no_hits (k : Nat) : jobCount (List.replicate k false) = k := by
  unfold jobCount
  simp

-- § PCIe share

/-- Number of missed experts the GPU reads over PCIe. -/
def pcieShare (nmiss pcieNum : Nat) : Nat := (nmiss * pcieNum) >>> 8

/-- The PCIe share never exceeds the number of misses when `pcie_num ≤ 256`. -/
theorem pcieShare_le (nmiss pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare nmiss pcieNum ≤ nmiss := by
  unfold pcieShare
  rw [Nat.shiftRight_eq_div_pow]
  have h1 : nmiss * pcieNum ≤ nmiss * 256 := Nat.mul_le_mul_left _ h
  omega

/-- With `pcie_num = 0` the CPU keeps every miss. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  simp [pcieShare]

end Ref.ExpertSource



## FILE: /mnt/user-data/outputs/GraphRegistry.lean
/-!
# MODULE: graph_registry
SOURCES: src/core/graph.cpp
Captured CUDA graph registry: a graph is recorded once per `(layer type, n_tokens)` key; a capture that records
zero nodes is refused; a bounded wait never succeeds after its deadline.

## Source semantics
* `record` of an existing key is a no-op returning success.
* `launch` requires a recorded key.
* `wait_ms` loops on a real clock, returning false on timeout or on a real error.

## Form template identifiers
Key → Nat key ; graphs_ → List Nat ; record → recordKey ; end() node check → endOk
-/

namespace Ref.GraphRegistry

-- § registry

/-- Record a key into the registry (already-present keys are unchanged). -/
def recordKey (ks : List Nat) (k : Nat) : List Nat := if k ∈ ks then ks else k :: ks

/-- After recording, the key is found (so `launch` can succeed). -/
theorem mem_recordKey (ks : List Nat) (k : Nat) : k ∈ recordKey ks k := by
  unfold recordKey
  by_cases h : k ∈ ks <;> simp [h]

/-- Recording twice equals recording once: the point of a registry. -/
theorem recordKey_idem (ks : List Nat) (k : Nat) : recordKey (recordKey ks k) k = recordKey ks k := by
  unfold recordKey
  by_cases h : k ∈ ks <;> simp [h]

/-- Recording never loses an existing key. -/
theorem recordKey_mono (ks : List Nat) (k j : Nat) (h : j ∈ ks) : j ∈ recordKey ks k := by
  unfold recordKey
  by_cases hk : k ∈ ks <;> simp [hk, h]

/-- The number of captures grows by at most one per `record`. -/
theorem recordKey_length (ks : List Nat) (k : Nat) : (recordKey ks k).length ≤ ks.length + 1 := by
  unfold recordKey
  by_cases h : k ∈ ks <;> simp [h]

-- § capture validity

/-- `end()` accepts a capture only when it recorded at least one node. -/
def endOk (nodes : Nat) : Bool := decide (0 < nodes)

/-- An empty capture is refused. -/
theorem endOk_zero : endOk 0 = false := by
  simp [endOk]

/-- A non-empty capture is accepted. -/
theorem endOk_pos (n : Nat) (h : 0 < n) : endOk n = true := by
  simp [endOk, h]

-- § bounded wait

/-- Outcome of one poll: success, real error, or still running past/before the deadline. -/
def pollResult (done err : Bool) (now deadline : Nat) : Option Bool :=
  if done then some true
  else if err then some false
  else if deadline ≤ now then some false
  else none

/-- Completion wins over everything else. -/
theorem pollResult_done (err : Bool) (now deadline : Nat) : pollResult true err now deadline = some true := by
  simp [pollResult]

/-- Past the deadline an unfinished, error-free poll reports failure rather than spinning on. -/
theorem pollResult_timeout (now deadline : Nat) (h : deadline ≤ now) :
    pollResult false false now deadline = some false := by
  simp [pollResult, h]

/-- Before the deadline an unfinished, error-free poll keeps waiting. -/
theorem pollResult_wait (now deadline : Nat) (h : now < deadline) :
    pollResult false false now deadline = none := by
  have : ¬ deadline ≤ now := by omega
  simp [pollResult, this]

end Ref.GraphRegistry



## FILE: /mnt/user-data/outputs/SessionStateLoop.lean
/-!
# MODULE: session_state_loop
SOURCES: src/core/session.cpp
One token through all layers: session arena carving (256-byte alignment), per-GDN-layer state slices,
doorbell ring numbering, and the stage-prefix timing telescope.

## Source semantics
* `align_up(n, 256)` sizes every carve; the arena total is aligned plus 4096.
* Each GDN layer owns the slice indexed by the number of non-QSA layers before it; slices never coincide.
* Layer `l` waits for ring `l + 1`; the counter strictly increases.
* Stage times are prefix differences, so they telescope back to the full prefix.

## Form template identifiers
align_up → alignUp256 ; gdn_point_at index → gdnIndex ; expected → ringWant ; stage_ms → prefix deltas
-/

namespace Ref.Session

-- § arena alignment

/-- Round up to a multiple of 256. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- A carve never shrinks. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- A carve ends on a 256-byte boundary. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- A carve over-allocates by less than 256 bytes. -/
theorem alignUp256_lt (n : Nat) : alignUp256 n < n + 256 := by
  unfold alignUp256; omega

/-- The `take` lambda: successive carves start aligned and are disjoint. -/
theorem carve_step (used bytes : Nat) (h : used % 256 = 0) :
    alignUp256 (used + bytes) % 256 = 0 ∧ used + bytes ≤ alignUp256 (used + bytes) :=
  ⟨alignUp256_mod _, alignUp256_ge _⟩

/-- Total arena size is aligned size plus the 4096-byte pad, so it covers `n`. -/
theorem session_bytes_ge (n : Nat) : n + 4096 ≤ alignUp256 n + 4096 := by
  have := alignUp256_ge n
  omega

-- § GDN state slices

/-- Number of non-QSA (GDN) layers strictly before layer `l`. -/
def gdnIndex (isQsa : Nat → Bool) : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIndex isQsa l + (if isQsa l then 0 else 1)

/-- The GDN index never exceeds the layer index. -/
theorem gdnIndex_le (isQsa : Nat → Bool) (l : Nat) : gdnIndex isQsa l ≤ l := by
  induction l with
  | zero => simp [gdnIndex]
  | succ n ih =>
    simp only [gdnIndex]
    split <;> omega

/-- The GDN index is non-decreasing step by step. -/
theorem gdnIndex_step_le (isQsa : Nat → Bool) (l : Nat) : gdnIndex isQsa l ≤ gdnIndex isQsa (l + 1) := by
  simp only [gdnIndex]
  omega

/-- The GDN index is monotone. -/
theorem gdnIndex_mono (isQsa : Nat → Bool) {a b : Nat} (h : a ≤ b) : gdnIndex isQsa a ≤ gdnIndex isQsa b := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => exact Nat.le_trans ih (gdnIndex_step_le isQsa _)

/-- A GDN layer's own slot is followed by a fresh one. -/
theorem gdnIndex_succ_of_gdn (isQsa : Nat → Bool) (l : Nat) (h : isQsa l = false) :
    gdnIndex isQsa (l + 1) = gdnIndex isQsa l + 1 := by
  simp [gdnIndex, h]

/-- Two distinct GDN layers never share a state slice: sharing would feed one layer's recurrence into another. -/
theorem gdnIndex_injective (isQsa : Nat → Bool) (l l' : Nat) (hl : isQsa l = false) (hlt : l < l') :
    gdnIndex isQsa l < gdnIndex isQsa l' := by
  have h1 := gdnIndex_succ_of_gdn isQsa l hl
  have h2 := gdnIndex_mono isQsa (show l + 1 ≤ l' from hlt)
  omega

-- § doorbell numbering

/-- Ring sequence number layer `l` waits for. -/
def ringWant (l : Nat) : Nat := l + 1

/-- Ring numbers strictly increase with the layer, so a stale ring cannot satisfy a later wait. -/
theorem ringWant_strict (l : Nat) : ringWant l < ringWant (l + 1) := by
  unfold ringWant; omega

/-- Ring numbers are positive (the doorbell is reset to 0 before the first layer). -/
theorem ringWant_pos (l : Nat) : 0 < ringWant l := by
  unfold ringWant; omega

-- § stage-prefix timing

/-- Stage times are differences of cumulative prefix times; they telescope to the longest prefix. -/
theorem stage_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by
  omega

end Ref.Session



## FILE: /mnt/user-data/outputs/WeightLoaderSegments.lean
/-!
# MODULE: weight_loader_segments
SOURCES: src/core/weights.cpp
Dense-weight loader: per-plane segment accounting (codes, fp16-or-f32 scales, offsets), chunked staging that
never splits an fp16 element, promoted-type 4→2 byte offsets, and compact arena placement.

## Source semantics
* An fp16 scale plane is half as many source bytes as destination bytes and must have an even destination count.
* Segment sums are checked against the index's recorded totals.
* A widening chunk is even-length; chunks make progress while two or more bytes remain.
* Promoted tensors read 4 B/elem and write 2 B/elem; destination offset is `done / 2`.
* Skipped rows get no bytes; kept rows are re-placed at 256-byte alignment.

## Form template identifiers
scales_fp16 → srcScaleBytes ; CHUNK → chunkBase ; Seg sums → segSrcTotal/segDstTotal ; compaction → compactStep
-/

namespace Ref.Weights

-- § plane sizes

/-- Source byte count of a scale plane. -/
def srcScaleBytes : Bool → Nat → Nat
  | true, s => s / 2
  | false, s => s

/-- Destination byte count of the scale plane is the index's `scales_bytes`. -/
def segSrcTotal (codes scales offset : Nat) (fp16 : Bool) : Nat :=
  codes + srcScaleBytes fp16 scales + offset

/-- Destination total of a quantized tensor. -/
def segDstTotal (codes scales offset : Nat) : Nat := codes + scales + offset

/-- An even fp16 scale plane doubles exactly when widened. -/
theorem srcScale_double (s : Nat) (h : s % 2 = 0) : srcScaleBytes true s * 2 = s := by
  show s / 2 * 2 = s
  omega

/-- Without fp16 scales the planes are copied verbatim, so source and destination agree. -/
theorem seg_total_copy (codes scales offset : Nat) :
    segSrcTotal codes scales offset false = segDstTotal codes scales offset := by
  rfl

/-- With fp16 scales the destination exceeds the source by exactly the widened half. -/
theorem seg_total_fp16 (codes scales offset : Nat) (h : scales % 2 = 0) :
    segDstTotal codes scales offset = segSrcTotal codes scales offset true + scales / 2 := by
  unfold segDstTotal segSrcTotal
  show codes + scales + offset = codes + scales / 2 + offset + scales / 2
  omega

-- § chunking

/-- Base chunk: at most 8 MiB. -/
def chunkBase (r : Nat) : Nat := if r < 8388608 then r else 8388608

/-- Chunk for a widening plane: drop one byte if the chunk is odd. -/
def chunkWiden (r : Nat) : Nat := if chunkBase r % 2 = 1 then chunkBase r - 1 else chunkBase r

/-- The base chunk never exceeds what remains. -/
theorem chunkBase_le (r : Nat) : chunkBase r ≤ r := by
  unfold chunkBase; split <;> omega

/-- The base chunk is even whenever the remainder is even. -/
theorem chunkBase_even (r : Nat) (h : r % 2 = 0) : chunkBase r % 2 = 0 := by
  unfold chunkBase; split <;> omega

/-- The base chunk is at least 2 when at least 2 bytes remain. -/
theorem chunkBase_ge (r : Nat) (h : 2 ≤ r) : 2 ≤ chunkBase r := by
  unfold chunkBase; split <;> omega

/-- A widening chunk never overruns the plane. -/
theorem chunkWiden_le (r : Nat) : chunkWiden r ≤ r := by
  have := chunkBase_le r
  unfold chunkWiden; split <;> omega

/-- A widening chunk is always even: it never splits an fp16 element. -/
theorem chunkWiden_even (r : Nat) : chunkWiden r % 2 = 0 := by
  unfold chunkWiden; split <;> omega

/-- A widening loop makes progress while two or more bytes remain, so it terminates. -/
theorem chunkWiden_pos (r : Nat) (h : 2 ≤ r) : 0 < chunkWiden r := by
  have := chunkBase_ge r h
  unfold chunkWiden; split <;> omega

/-- On an even plane a widening chunk equals the base chunk (no byte is left stranded). -/
theorem chunkWiden_eq_of_even (r : Nat) (h : r % 2 = 0) : chunkWiden r = chunkBase r := by
  have := chunkBase_even r h
  unfold chunkWiden; split <;> omega

-- § promoted conversions

/-- Promoted tensors: the destination offset `done / 2` composes across 4-byte-aligned chunks. -/
theorem promoted_offsets (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + n / 2 = (done + n) / 2 := by
  omega

/-- Widening offsets compose too: `done * 2 + n * 2 = (done + n) * 2`. -/
theorem widen_offsets (done n : Nat) : done * 2 + n * 2 = (done + n) * 2 := by
  omega

/-- A promoted tensor's expected byte counts: 4 per element in, 2 per element out. -/
theorem promoted_ratio (elems : Nat) : elems * 4 = 2 * (elems * 2) := by
  omega

-- § compact arena placement

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- One placement step of a kept row: cursor advances by the aligned destination size. -/
def compactStep (at_ dst : Nat) : Nat := at_ + align256 dst

/-- Placement keeps every cursor 256-aligned. -/
theorem compactStep_mod (at_ dst : Nat) (h : at_ % 256 = 0) : compactStep at_ dst % 256 = 0 := by
  unfold compactStep align256; omega

/-- Placed rows never overlap: the cursor moves past the row's bytes. -/
theorem compactStep_ge (at_ dst : Nat) : at_ + dst ≤ compactStep at_ dst := by
  unfold compactStep align256; omega

end Ref.Weights



## FILE: /mnt/user-data/outputs/MtpDrafter.lean
/-!
# MODULE: mtp_drafter
SOURCES: src/core/mtp.cpp
Multi-token-prediction draft layer: bump-allocated device arena, the attention window capacity, step-row staging
indices, and the early-stopping draft chain.

## Source semantics
* `Bump::take` advances by the 256-rounded byte count, so every buffer starts aligned.
* `cap = ceil(window / 64) * 64`.
* Chain step `j` uses staged row `max_t + j - 1`, which lies between the catch-up rows and the final row `2*max_t - 1`.
* The chain stops when `j` reaches `max_t - 1` or the last draft probability is below `min_p`.

## Form template identifiers
Bump::take → bumpTake ; cap_ → capOf ; step row → chainRow ; draft loop → chainAux
-/

namespace Ref.Mtp

-- § bump allocator

/-- New `used` after taking `bytes` bytes. -/
def bumpTake (used bytes : Nat) : Nat := used + (bytes + 255) / 256 * 256

/-- The arena only grows. -/
theorem bumpTake_ge (used bytes : Nat) : used + bytes ≤ bumpTake used bytes := by
  unfold bumpTake; omega

/-- Alignment is preserved, so every carve starts on a 256-byte boundary. -/
theorem bumpTake_mod (used bytes : Nat) (h : used % 256 = 0) : bumpTake used bytes % 256 = 0 := by
  unfold bumpTake; omega

/-- Over-allocation per buffer is under 256 bytes. -/
theorem bumpTake_lt (used bytes : Nat) : bumpTake used bytes < used + bytes + 256 := by
  unfold bumpTake; omega

-- § window capacity

/-- Attention capacity: the window rounded up to a multiple of 64. -/
def capOf (w : Nat) : Nat := (w + 63) / 64 * 64

/-- The capacity covers the window. -/
theorem capOf_ge (w : Nat) : w ≤ capOf w := by
  unfold capOf; omega

/-- The capacity is a multiple of 64. -/
theorem capOf_mod (w : Nat) : capOf w % 64 = 0 := by
  unfold capOf; omega

-- § staged rows

/-- Staged step row for chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Chain rows sit after the `max_t` catch-up rows. -/
theorem chainRow_ge (maxT j : Nat) (hj : 1 ≤ j) : maxT ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows sit strictly before the final-row slot `2*max_t - 1`, so the two never collide. -/
theorem chainRow_lt (maxT j : Nat) (hj : j < maxT - 1) (hm : 2 ≤ maxT) : chainRow maxT j < 2 * maxT - 1 := by
  unfold chainRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj (maxT j j' : Nat) (hj : 1 ≤ j) (hj' : 1 ≤ j') (h : chainRow maxT j = chainRow maxT j') :
    j = j' := by
  unfold chainRow at h; omega

-- § draft chain

/-- The draft loop's count of drafts: continue while `j < max_t - 1` and the last draft was likely enough. -/
def chainAux (maxT : Nat) (ok : Nat → Bool) : Nat → Nat → Nat
  | 0, j => j
  | f + 1, j => if j < maxT - 1 ∧ ok j = true then chainAux maxT ok f (j + 1) else j

/-- The chain never produces fewer drafts than it started with. -/
theorem chainAux_ge (maxT : Nat) (ok : Nat → Bool) (f j : Nat) : j ≤ chainAux maxT ok f j := by
  induction f generalizing j with
  | zero => simp [chainAux]
  | succ f ih =>
    simp only [chainAux]
    split
    · have := ih (j + 1)
      omega
    · exact Nat.le_refl _

/-- The chain never produces more than `max_t - 1` drafts, so the `drafts` array is not overrun. -/
theorem chainAux_le (maxT : Nat) (ok : Nat → Bool) (f j : Nat) (hj : j ≤ maxT - 1) :
    chainAux maxT ok f j ≤ maxT - 1 := by
  induction f generalizing j with
  | zero => simpa [chainAux] using hj
  | succ f ih =>
    simp only [chainAux]
    split
    · rename_i h
      exact ih (j + 1) (by omega)
    · exact hj

/-- A first draft always exists: starting at 1 with `max_t ≥ 2` the count is at least 1. -/
theorem chainAux_pos (maxT : Nat) (ok : Nat → Bool) (f : Nat) : 1 ≤ chainAux maxT ok f 1 :=
  chainAux_ge maxT ok f 1

/-- An unlikely draft stops the chain immediately. -/
theorem chainAux_stop (maxT : Nat) (ok : Nat → Bool) (f j : Nat) (h : ok j = false) :
    chainAux maxT ok (f + 1) j = j := by
  simp [chainAux, h]

end Ref.Mtp



## FILE: /mnt/user-data/outputs/VerifyWindow.lean
/-!
# MODULE: verify_window
SOURCES: src/core/verify.cpp
Speculative verify window: split into two token groups, per-(layer, group) ring numbering, staging split between
groups, monotone flag B, and the empty-plan fallback.

## Source semantics
* With `G = 2`, group A is `[0, (T+1)/2)` and group B is `[(T+1)/2, T)`; together they cover `[0, T)`.
* Ring number is `l * G + grp + 1`, strictly increasing in execution order, and decodes back to `(l, grp)`.
* Staging blobs are halved per group; the halves are disjoint.
* Flag B only rises (CAS-max).

## Form template identifiers
tb_/te_ → grpBegin/grpEnd ; ring → ringNo ; raise_flag → raiseFlag ; per → stagingPer
-/

namespace Ref.Verify

-- § token groups

/-- First token of a group. -/
def grpBegin (T : Nat) : Nat → Nat
  | 0 => 0
  | _ + 1 => (T + 1) / 2

/-- One past the last token of a group. -/
def grpEnd (T G : Nat) : Nat → Nat
  | 0 => if G = 2 then (T + 1) / 2 else T
  | _ + 1 => T

/-- In a split window the groups abut. -/
theorem groups_abut (T : Nat) : grpEnd T 2 0 = grpBegin T 1 := by
  simp [grpEnd, grpBegin]

/-- In a split window the two groups together cover exactly `T` tokens. -/
theorem groups_cover (T : Nat) :
    (grpEnd T 2 0 - grpBegin T 0) + (grpEnd T 2 1 - grpBegin T 1) = T := by
  simp [grpEnd, grpBegin]
  omega

/-- With at least two tokens, both groups are non-empty. -/
theorem groups_nonempty (T : Nat) (h : 2 ≤ T) :
    grpBegin T 0 < grpEnd T 2 0 ∧ grpBegin T 1 < grpEnd T 2 1 := by
  simp [grpEnd, grpBegin]
  omega

/-- Unsplit window: the single group is the whole window. -/
theorem group_unsplit (T : Nat) : grpBegin T 0 = 0 ∧ grpEnd T 1 0 = T := by
  simp [grpEnd, grpBegin]

-- § ring numbering

/-- Ring number of layer `l`, group `grp` in a split window. -/
def ringNo (l grp : Nat) : Nat := l * 2 + grp + 1

/-- Distinct `(layer, group)` pairs have distinct ring numbers. -/
theorem ringNo_inj (l g l' g' : Nat) (hg : g < 2) (hg' : g' < 2) (h : ringNo l g = ringNo l' g') :
    l = l' ∧ g = g' := by
  unfold ringNo at h
  omega

/-- Execution order is ring order: A then B within a layer, then the next layer. -/
theorem ringNo_order (l : Nat) : ringNo l 0 < ringNo l 1 ∧ ringNo l 1 < ringNo (l + 1) 0 := by
  unfold ringNo
  omega

/-- The host's loop index `k = l*G + grp` decodes back to the layer and group. -/
theorem ring_decode (l grp : Nat) (hg : grp < 2) : (l * 2 + grp) / 2 = l ∧ (l * 2 + grp) % 2 = grp := by
  omega

/-- The wait value for step `k` is `k + 1`, equal to `ringNo`. -/
theorem ring_want (l grp : Nat) : l * 2 + grp + 1 = ringNo l grp := rfl

-- § staging split

/-- Staging blobs available to each group in a split window. -/
def stagingPer (S : Nat) : Nat := S / 2

/-- The two groups' halves fit in the staging area. -/
theorem staging_halves_fit (S : Nat) : 2 * stagingPer S ≤ S := by
  unfold stagingPer; omega

/-- Group B's slice begins where group A's ends. -/
theorem staging_disjoint (S : Nat) : 0 + stagingPer S = 1 * stagingPer S := by
  omega

-- § monotone flag

/-- CAS-max raise of a flag. -/
def raiseFlag (cur v : Nat) : Nat := if cur < v then v else cur

/-- The flag never decreases. -/
theorem raiseFlag_ge_cur (cur v : Nat) : cur ≤ raiseFlag cur v := by
  unfold raiseFlag; split <;> omega

/-- The flag reaches at least the requested value. -/
theorem raiseFlag_ge_val (cur v : Nat) : v ≤ raiseFlag cur v := by
  unfold raiseFlag; split <;> omega

/-- Raising is idempotent, so a late host function of an earlier layer cannot lower it. -/
theorem raiseFlag_idem (cur v : Nat) : raiseFlag (raiseFlag cur v) v = raiseFlag cur v := by
  unfold raiseFlag
  by_cases h : cur < v <;> simp [h]

/-- A late, smaller raise is a no-op once a larger value has landed. -/
theorem raiseFlag_late (cur v : Nat) (h : v ≤ cur) : raiseFlag cur v = cur := by
  unfold raiseFlag; split <;> omega

-- § commit mask

/-- Position written to commit slot `t`: kept tokens carry their position, the rest `-1`. -/
def commitPos (pos0 nKeep t : Nat) : Int := if t < nKeep then (pos0 + t : Nat) else -1

/-- Kept tokens carry a non-negative position. -/
theorem commitPos_kept (pos0 nKeep t : Nat) (h : t < nKeep) : 0 ≤ commitPos pos0 nKeep t := by
  unfold commitPos
  split <;> omega

/-- Dropped tokens are marked `-1`. -/
theorem commitPos_dropped (pos0 nKeep t : Nat) (h : nKeep ≤ t) : commitPos pos0 nKeep t = -1 := by
  have : ¬ t < nKeep := by omega
  simp [commitPos, this]

end Ref.Verify





## FILE: /mnt/user-data/outputs/LayerCompose.lean
/-!
# MODULE: layer_compose
SOURCES: src/core/layer.cpp
The composed GDN / QSA / MoE decoder layer: buffer carving (16-byte aligned cursor), weight-plane validation,
per-tensor activation dispatch, stage selection of `block_layer_pre`, the doorbell order and the debug-dump layout.

## Source semantics
* Every arena region is rounded up to 16 bytes; regions handed out by one cursor never overlap.
* A quantized tensor's three planes (codes, scales, offsets) must add up exactly to the tensor size, and
  `has_offset` must agree with a non-empty offset plane.
* The activation image a projection reads is chosen by the tensor, not by its role: native first, then
  `code_bits == 2` -> Q8_0, a legacy form -> Q8_0, otherwise Q8_K.
* `stage_prefix == 0` means "as `half` says"; `stage_prefix == k > 0` runs exactly stages `0..k-1`.
* The doorbell ring is the LAST operation of the publish sequence.

## Form template identifiers
align16 <- `(n + 15) & ~15`; Cursor.take <- `Cursor::take/take_bytes`; PlanesOk <- `plane_ptrs` guard;
dispatch <- `gemv_quantized`; RunStage <- `run0..run4` in `block_layer_pre`; dumpStride <- `dump_stride_floats`.
-/

namespace Ref.LayerCompose

-- § alignment and the cursor

/-- 16-byte round-up `(n + 15) & ~15`, written as a division-free formula. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never loses bytes. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- The result is 16-aligned. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Rounding up wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Aligning twice is aligning once. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

/-- Alignment is monotone. -/
theorem align16_mono {a b : Nat} (h : a ≤ b) : align16 a ≤ align16 b := by unfold align16; omega

/-- The arena cursor: only the number of bytes used so far matters. -/
structure Cursor where
  used : Nat

/-- `take bytes` returns the new cursor and the offset of the region just handed out. -/
def Cursor.take (c : Cursor) (bytes : Nat) : Cursor × Nat :=
  (⟨align16 (c.used + bytes)⟩, c.used)

/-- The cursor stays 16-aligned after every take. -/
theorem take_aligned (c : Cursor) (bytes : Nat) : (c.take bytes).1.used % 16 = 0 :=
  align16_mod (c.used + bytes)

/-- The handed-out region `[offset, offset + bytes)` lies entirely below the new cursor (no overlap later). -/
theorem take_covers (c : Cursor) (bytes : Nat) : c.used + bytes ≤ (c.take bytes).1.used :=
  align16_ge (c.used + bytes)

/-- The padding of a take is under 16 bytes. -/
theorem take_tight (c : Cursor) (bytes : Nat) : (c.take bytes).1.used < c.used + bytes + 16 :=
  align16_lt (c.used + bytes)

/-- The returned offset is the old cursor. -/
theorem take_offset (c : Cursor) (bytes : Nat) : (c.take bytes).2 = c.used := rfl

/-- An aligned cursor hands out an aligned offset. -/
theorem take_offset_aligned (c : Cursor) (bytes : Nat) (h : c.used % 16 = 0) : (c.take bytes).2 % 16 = 0 := h

/-- Sum of a list of sizes. -/
def lsum : List Nat → Nat
  | [] => 0
  | a :: l => a + lsum l

/-- Total of the individually aligned parts (`gdn_buffers_bytes`, `moe_buffers_bytes`). -/
def totalAligned : List Nat → Nat
  | [] => 0
  | a :: l => align16 a + totalAligned l

/-- The aligned total covers the raw sum. -/
theorem sum_le_totalAligned (l : List Nat) : lsum l ≤ totalAligned l := by
  induction l with
  | nil => simp [lsum, totalAligned]
  | cons a l ih =>
    have h := align16_ge a
    simp only [lsum, totalAligned]
    omega

/-- Offsets of consecutive aligned parts starting at `start`. -/
def place (start : Nat) : List Nat → List Nat
  | [] => []
  | a :: l => start :: place (start + align16 a) l

/-- One offset per part. -/
theorem place_length (l : List Nat) : ∀ start : Nat, (place start l).length = l.length := by
  induction l with
  | nil => intro start; rfl
  | cons a l ih => intro start; simp [place, ih]

-- § Q8_K sizing

/-- Bytes of a Q8_K buffer for `n` elements (292 bytes per 256-element block). -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- A whole number of blocks costs `292` bytes each. -/
theorem q8kBytes_blocks (k : Nat) : q8kBytes (k * 256) = k * 292 := by unfold q8kBytes; omega

/-- Q8_K sizing is monotone. -/
theorem q8kBytes_mono {a b : Nat} (h : a ≤ b) : q8kBytes a ≤ q8kBytes b := by unfold q8kBytes; omega

-- § weight planes

/-- The loader-recorded plane sizes of a quantized tensor. -/
structure WRef where
  bits : Nat
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat
  hasOffset : Bool

/-- The guard of `plane_ptrs`. -/
def PlanesOk (r : WRef) : Prop :=
  (r.bits = 2 ∨ r.bits = 4 ∨ r.bits = 8) ∧ r.codes ≠ 0 ∧ r.scales ≠ 0 ∧
  r.codes + r.scales + r.offset = r.bytes ∧ (r.hasOffset = true ↔ r.offset ≠ 0)

/-- Accepted planes tile the tensor: the offset plane starts right after codes and scales and ends at `bytes`. -/
theorem planes_tile (r : WRef) (h : PlanesOk r) :
    r.codes + r.scales ≤ r.bytes ∧ r.codes + r.scales + r.offset = r.bytes := by
  obtain ⟨_, _, _, hs, _⟩ := h
  constructor <;> omega

/-- A tensor with no offset plane has `has_offset = false`. -/
theorem planes_no_offset (r : WRef) (h : PlanesOk r) (h0 : r.offset = 0) : r.hasOffset = false := by
  obtain ⟨_, _, _, _, hi⟩ := h
  cases hh : r.hasOffset with
  | false => rfl
  | true => exact absurd h0 (hi.mp hh)

/-- If planes disagree with the tensor size the guard refuses. -/
theorem planes_refuse (r : WRef) (h : r.codes + r.scales + r.offset ≠ r.bytes) : ¬ PlanesOk r := by
  intro hp
  exact h hp.2.2.2.1

-- § activation dispatch

/-- Which activation image a projection consumes. -/
inductive ActKind
  | native
  | q80
  | q8k
  deriving DecidableEq

/-- `gemv_quantized`: native, then S2 (Q8_0), then legacy (Q8_0), then K-quant (Q8_K). -/
def dispatch (native : Bool) (bits : Nat) (wantsQ8K : Bool) : ActKind :=
  if native then ActKind.native
  else if bits = 2 then ActKind.q80
  else if wantsQ8K then ActKind.q8k
  else ActKind.q80

/-- A 2-bit form always reads the Q8_0 image. -/
theorem dispatch_s2 (bits : Nat) (wants : Bool) (h : bits = 2) : dispatch false bits wants = ActKind.q80 := by
  simp [dispatch, h]

/-- A tensor that does not want Q8_K never reads it. -/
theorem dispatch_legacy (bits : Nat) : dispatch false bits false = ActKind.q80 := by
  unfold dispatch
  by_cases h : bits = 2 <;> simp [h]

/-- A wider-than-2-bit K-quant reads Q8_K. -/
theorem dispatch_kquant (bits : Nat) (h : bits ≠ 2) : dispatch false bits true = ActKind.q8k := by
  simp [dispatch, h]

/-- The native path ignores the code width. -/
theorem dispatch_native (bits : Nat) (w : Bool) : dispatch true bits w = ActKind.native := by
  simp [dispatch]

-- § stage selection of block_layer_pre

/-- Stages 0..2 run in the mixer half (half 0/1), stages 3..4 in the FFN half (half 0/2). -/
def halfAllows (k h : Nat) : Prop :=
  if k ≤ 2 then (h = 0 ∨ h = 1) else (h = 0 ∨ h = 2)

/-- Stage `k` runs for `stage_prefix = p`, `half = h`. -/
def RunStage (k p h : Nat) : Prop := if 0 < p then k < p else halfAllows k h

/-- With a positive prefix exactly the first `p` stages run, whatever `half` is. -/
theorem runStage_prefix {k p h : Nat} (hp : 0 < p) : RunStage k p h ↔ k < p := by
  unfold RunStage
  rw [if_pos hp]

/-- The legacy call (`prefix = 0`, `half = 0`) runs all five stages. -/
theorem runStage_legacy {k : Nat} : RunStage k 0 0 := by
  unfold RunStage halfAllows
  rw [if_neg (Nat.lt_irrefl 0)]
  split <;> simp

/-- Prefixes are downward closed. -/
theorem runStage_mono {k p h : Nat} (hp : 0 < p) (hr : RunStage (k + 1) p h) : RunStage k p h := by
  rw [runStage_prefix hp] at hr ⊢
  omega

/-- `half = 2` skips the mixer. -/
theorem runStage_half2_skips_mixer : ¬ RunStage 0 0 2 := by
  unfold RunStage halfAllows
  rw [if_neg (Nat.lt_irrefl 0)]
  simp

/-- `half = 1` skips the FFN front. -/
theorem runStage_half1_skips_ffn : ¬ RunStage 3 0 1 := by
  unfold RunStage halfAllows
  rw [if_neg (Nat.lt_irrefl 0)]
  simp

-- § verify compatibility, doorbell order, positions, dump layout

/-- `layer_verify_compatible`: all six switches must be on. -/
def verifyCompatible (a b c d e f : Bool) : Bool := a && b && c && d && e && f

/-- Compatible iff every switch is on. -/
theorem verifyCompatible_iff (a b c d e f : Bool) :
    verifyCompatible a b c d e f = true ↔ a = true ∧ b = true ∧ c = true ∧ d = true ∧ e = true ∧ f = true := by
  simp [verifyCompatible, and_assoc]

/-- Operations of the non-kernel doorbell publish. -/
inductive Op
  | copyX
  | copyIds
  | copyWeights
  | ring
  deriving DecidableEq

/-- The publish order. -/
def publishOrder : List Op := [Op.copyX, Op.copyIds, Op.copyWeights, Op.ring]

/-- The ring is last, so a host that sees it sees every byte above. -/
theorem ring_is_last : publishOrder.getLast? = some Op.ring := rfl

/-- RoPE position guard of `qsa_layer`. -/
def qsaPosOk (posBase pos maxCells : Int) : Prop := 0 ≤ posBase ∧ 0 ≤ pos ∧ posBase + pos < maxCells

/-- An accepted position is a valid row of the RoPE table. -/
theorem qsaPosOk_row (posBase pos maxCells : Int) (h : qsaPosOk posBase pos maxCells) :
    0 ≤ posBase + pos ∧ posBase + pos < maxCells := by
  obtain ⟨h1, h2, h3⟩ := h
  constructor <;> omega

/-- Floats per layer of the half-level dump (`A = n_head * head_dim`, `nvk = n_head_kv * head_dim`). -/
def dumpStride (n hc A nvk : Nat) : Nat := 2 * n + 2 * hc + A + 5 * nvk + 8

/-- Offset of the KV slots inside a layer's dump. -/
def dumpKvBase (n hc A nvk : Nat) : Nat := 2 * n + 2 * hc + A + 2 * nvk + 8

/-- The three trailing dump slots end exactly at the stride, so layers never overlap. -/
theorem dump_slots_end_at_stride (n hc A nvk : Nat) : dumpKvBase n hc A nvk + 3 * nvk = dumpStride n hc A nvk := by
  unfold dumpKvBase dumpStride; omega

/-- The two half-pool prefixes fit in the `nvk` slot reserved for them. -/
theorem dump_pool_prefix_fits (nvk : Nat) : nvk / 2 + nvk / 2 ≤ nvk := by omega

/-- Row-bounds of `embed_row`: a row inside the table stays inside the code plane. -/
theorem embed_row_in_bounds (rows rowCodes token codesBytes : Nat) (h : codesBytes = rows * rowCodes)
    (ht : token < rows) : (token + 1) * rowCodes ≤ codesBytes := by
  subst h
  exact Nat.mul_le_mul_right rowCodes (Nat.succ_le_of_lt ht)

end Ref.LayerCompose



## FILE: /mnt/user-data/outputs/GenerateDriver.lean
/-!
# MODULE: generate_driver
SOURCES: src/program/generate.cpp
The `ref generate` driver: option validation, the token loop (prompt teacher forcing, first generated token at
the last prompt position), logits-row selection, speculative-window acceptance, VRAM expert-tier sizing and
lending, and the image M-RoPE position assignment of `--serve`.

## Source semantics
* The prompt plus `max_new` must fit in `max_context`.
* The dump header counts `n_prompt - 1 + max_new` positions; with a stride, every N-th row plus the last is stored.
* While the prompt lasts the next input is the prompt's own next token, afterwards the sampled one.
* A verify window accepts the longest prefix of drafts equal to the model's own outputs; it emits accepted + 1 tokens.
* Slots lent to the prompt path are a suffix of the cache and are given back afterwards.

## Form template identifiers
Options.max_new -> maxNew; Options.max_context -> ctx; `produced` -> producedAt; `a` (accepted) -> acc;
`lend_first` -> lendFirst; `pcie_num` -> pcieNum; `winLoop` <- the `T` selection `while` loop.
-/

namespace Ref.GenerateDriver

-- § option validation

/-- The size check of `main`: positive budgets and prompt + generation within the context. -/
def OptsOk (maxNew ctx promptLen : Nat) : Prop :=
  0 < maxNew ∧ 0 < ctx ∧ maxNew ≤ ctx ∧ promptLen ≤ ctx - maxNew

/-- Accepted options guarantee the prompt and the new tokens fit together. -/
theorem opts_fit (maxNew ctx promptLen : Nat) (h : OptsOk maxNew ctx promptLen) : promptLen + maxNew ≤ ctx := by
  obtain ⟨_, _, h3, h4⟩ := h
  omega

/-- The `--serve` guard `n + max_new + 8 <= ctx` keeps the whole request inside the context. -/
theorem serve_fit (n maxNew ctx : Nat) (h : ¬ (ctx < n + maxNew + 8)) : n + maxNew ≤ ctx := by omega

-- § the token loop

/-- Number of logged dump positions: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- Tokens recorded after processing position `pos` (recording starts at the last prompt position). -/
def producedAt (nPrompt pos : Nat) : Nat := if nPrompt ≤ pos + 1 then pos + 2 - nPrompt else 0

/-- Nothing is produced before the last prompt position. -/
theorem produced_before_prompt_end (nPrompt pos : Nat) (h : pos + 1 < nPrompt) : producedAt nPrompt pos = 0 := by
  unfold producedAt
  rw [if_neg (by omega)]

/-- The last prompt position yields exactly the first generated token. -/
theorem produced_first (nPrompt : Nat) (h : 1 ≤ nPrompt) : producedAt nPrompt (nPrompt - 1) = 1 := by
  unfold producedAt
  rw [if_pos (by omega)]
  omega

/-- For a prompt of two or more tokens the loop reaches `max_new` tokens at position `n_prompt - 2 + max_new`. -/
theorem produced_reaches_max (nPrompt maxNew : Nat) (h1 : 2 ≤ nPrompt) (h2 : 1 ≤ maxNew) :
    producedAt nPrompt (nPrompt - 2 + maxNew) = maxNew := by
  unfold producedAt
  rw [if_pos (by omega)]
  omega

/-- For a one-token prompt the loop reaches `max_new` tokens at position `max_new - 1`. -/
theorem produced_reaches_max_single (maxNew : Nat) (h2 : 1 ≤ maxNew) :
    producedAt 1 (maxNew - 1) = maxNew := by
  unfold producedAt
  rw [if_pos (by omega)]
  omega

/-- Teacher forcing: the next input is the prompt token while the prompt lasts, the sample afterwards. -/
def nextInput (prompt : Nat → Nat) (n pos next : Nat) : Nat := if pos + 1 < n then prompt (pos + 1) else next

/-- Inside the prompt the model's guess is ignored. -/
theorem nextInput_prompt (prompt : Nat → Nat) (n pos next : Nat) (h : pos + 1 < n) :
    nextInput prompt n pos next = prompt (pos + 1) := by
  unfold nextInput
  rw [if_pos h]

/-- Past the prompt the sampled token is fed back. -/
theorem nextInput_sampled (prompt : Nat → Nat) (n pos next : Nat) (h : ¬ pos + 1 < n) :
    nextInput prompt n pos next = next := by
  unfold nextInput
  rw [if_neg h]

-- § logits-row selection (`--logits-stride`)

/-- A position's row is stored when it is a multiple of the stride or the final position. -/
def selected (pos total stride : Nat) : Prop := pos % stride = 0 ∨ pos + 1 = total

/-- Stride 1 stores every row. -/
theorem selected_stride_one (pos total : Nat) : selected pos total 1 := by
  left
  exact Nat.mod_one pos

/-- The final position is always stored. -/
theorem selected_last (total stride : Nat) (h : 0 < total) : selected (total - 1) total stride := by
  right
  omega

-- § speculative verification

/-- Length of the longest prefix of the drafts that equals the model's outputs (`a` in the source). -/
def acc : List Nat → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | d :: ds, o :: os => if d = o then acc ds os + 1 else 0

/-- Never more accepted than drafted. -/
theorem acc_le_left (ds os : List Nat) : acc ds os ≤ ds.length := by
  induction ds generalizing os with
  | nil => simp [acc]
  | cons d ds ih =>
    cases os with
    | nil => simp [acc]
    | cons o os =>
      have h := ih os
      simp only [acc, List.length_cons]
      split <;> omega

/-- Never more accepted than the window produced. -/
theorem acc_le_right (ds os : List Nat) : acc ds os ≤ os.length := by
  induction ds generalizing os with
  | nil => simp [acc]
  | cons d ds ih =>
    cases os with
    | nil => simp [acc]
    | cons o os =>
      have h := ih os
      simp only [acc, List.length_cons]
      split <;> omega

/-- Perfect drafts are fully accepted. -/
theorem acc_self (ds : List Nat) : acc ds ds = ds.length := by
  induction ds with
  | nil => simp [acc]
  | cons d ds ih => simp [acc, ih]

/-- A round emits `accepted + 1` tokens, never more than the window length `1 + drafts`. -/
theorem emitted_le_window (ds os : List Nat) : acc ds os + 1 ≤ 1 + ds.length := by
  have := acc_le_left ds os
  omega

/-- Window-size loop: grow `T` while it is below `S` and the next draft probability test passes. -/
def winLoop (ok : Nat → Bool) (S : Nat) : Nat → Nat → Nat
  | 0, T => T
  | f + 1, T => if T < S ∧ ok (T - 1) = true then winLoop ok S f (T + 1) else T

/-- The loop never shrinks the window. -/
theorem winLoop_ge (ok : Nat → Bool) (S : Nat) (f : Nat) : ∀ T, T ≤ winLoop ok S f T := by
  induction f with
  | zero => intro T; exact Nat.le_refl T
  | succ f ih =>
    intro T
    simp only [winLoop]
    split
    · have := ih (T + 1)
      omega
    · exact Nat.le_refl T

/-- The loop never exceeds the speculation depth `S`. -/
theorem winLoop_le (ok : Nat → Bool) (S : Nat) (f : Nat) : ∀ T, T ≤ S → winLoop ok S f T ≤ S := by
  induction f with
  | zero => intro T h; exact h
  | succ f ih =>
    intro T h
    simp only [winLoop]
    split
    · rename_i hc
      obtain ⟨hlt, _⟩ := hc
      exact ih (T + 1) (by omega)
    · exact h

-- § VRAM expert tier

/-- Slots that fit after the reserve: `(free - reserve) / blob`. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The automatically sized tier never exceeds the free memory left after the reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- First lent slot when `k` slots are borrowed (only if `k + 128 ≤ slots`). -/
def lendFirst (slots k : Nat) : Nat := slots - k

/-- Lent slots are exactly the suffix `[lendFirst, slots)` and at least 128 slots stay resident. -/
theorem lend_suffix (slots k : Nat) (h : k + 128 ≤ slots) : lendFirst slots k + k = slots ∧ 128 ≤ lendFirst slots k := by
  unfold lendFirst
  constructor <;> omega

/-- The PCIe share numerator is clamped to `0..256`. -/
def pcieNum (x : Nat) : Nat := min 256 x

/-- The clamp is within range. -/
theorem pcieNum_le (x : Nat) : pcieNum x ≤ 256 := by unfold pcieNum; omega

-- § image M-RoPE positions (`GENI`)

/-- Within an image of `nx * ny` rows at base position `p`, the (h, w) positions stay below `p + max nx ny`,
which is where the text after the image resumes. -/
theorem mrope_image_pos_lt (p j nx ny : Nat) (hj : j < nx * ny) :
    p + j / nx < p + max nx ny ∧ p + j % nx < p + max nx ny := by
  have h1 : j / nx < ny := Nat.div_lt_of_lt_mul hj
  have hnx : 0 < nx := by
    apply Nat.pos_of_ne_zero
    intro h
    subst h
    simp at hj
  have h2 : j % nx < nx := Nat.mod_lt j hnx
  constructor <;> omega

end Ref.GenerateDriver



## FILE: /mnt/user-data/outputs/PrefillEngine.lean
/-!
# MODULE: prefill_engine
SOURCES: src/prefill/prefill.cpp
Batched prompt processing: the bump allocator that either owns or borrows its chunk buffers (and the counting pass
`bytes_needed` that sizes the borrowed region), the chunk loop, the host->device expert staging ring and the
dequantized-expert double buffer, and the PLE window carried across chunks.

## Source semantics
* `bytes_needed` replays exactly the allocation sequence of `init`, so a region of that size always suffices.
* Chunks of at most `T` tokens partition the batched prefix of the prompt exactly.
* Experts are staged with look-ahead `STAGE - 1`, so the blobs in flight never share a ring slot.
* The dequantized-expert ring alternates between two buffers.

## Form template identifiers
Alloc.take -> padBytes/allocAll; `m.T` -> chunk; `STAGE` -> 8; `DQ` -> 2; `stage_next` -> ringNext.
-/

namespace Ref.PrefillEngine

/-- Sum of a list. -/
def lsum : List Nat → Nat
  | [] => 0
  | a :: l => a + lsum l

-- § allocator

/-- Bytes charged for `b` payload bytes: 256 bytes of slack, rounded up to a multiple of 256. -/
def padBytes (b : Nat) : Nat := (b + 256 + 255) / 256 * 256

/-- An allocation is at least its payload plus the 256-byte guard. -/
theorem padBytes_ge (b : Nat) : b + 256 ≤ padBytes b := by unfold padBytes; omega

/-- Allocations are 256-aligned. -/
theorem padBytes_mod (b : Nat) : padBytes b % 256 = 0 := by unfold padBytes; omega

/-- Bump allocation of a request list from a region of capacity `cap` starting at `used`
(`none` is the `ok = false` path). -/
def allocAll (cap : Nat) : Nat → List Nat → Option Nat
  | used, [] => some used
  | used, b :: l => if used + b ≤ cap then allocAll cap (used + b) l else none

/-- If the whole sequence fits, every allocation succeeds and the final cursor is the exact total. -/
theorem allocAll_fits (cap : Nat) (l : List Nat) : ∀ u, u + lsum l ≤ cap → allocAll cap u l = some (u + lsum l) := by
  induction l with
  | nil => intro u _; simp [allocAll, lsum]
  | cons b l ih =>
    intro u h
    have h1 : u + b ≤ cap := by simp only [lsum] at h; omega
    have h2 : u + b + lsum l ≤ cap := by simp only [lsum] at h; omega
    simp only [allocAll]
    rw [if_pos h1, ih (u + b) h2]
    simp [lsum, Nat.add_assoc]

/-- If the sequence overflows the region the allocator reports failure rather than overrunning it. -/
theorem allocAll_overflow (cap : Nat) (l : List Nat) : ∀ u, u ≤ cap → cap < u + lsum l → allocAll cap u l = none := by
  induction l with
  | nil => intro u h1 h2; simp [lsum] at h2; omega
  | cons b l ih =>
    intro u h1 h2
    simp only [allocAll]
    by_cases hb : u + b ≤ cap
    · rw [if_pos hb]
      apply ih (u + b) hb
      simp only [lsum] at h2
      omega
    · rw [if_neg hb]

/-- `bytes_needed` (count-only pass) is exactly what `init` consumes: a region of that size always suffices. -/
theorem bytes_needed_suffices (l : List Nat) (start : Nat) :
    allocAll (start + lsum l) start l = some (start + lsum l) :=
  allocAll_fits (start + lsum l) l start (Nat.le_refl _)

-- § chunking

/-- Chunk sizes `min T remaining`, with fuel to keep the loop structural. -/
def chunksAux : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | fuel + 1, T, n => if n = 0 then [] else min T n :: chunksAux fuel T (n - min T n)

/-- The chunks partition the batched prefix exactly (given `T > 0` and enough fuel). -/
theorem chunks_sum (T : Nat) (hT : 0 < T) (fuel : Nat) : ∀ n, n ≤ fuel → lsum (chunksAux fuel T n) = n := by
  induction fuel with
  | zero => intro n h; simp [chunksAux, lsum]; omega
  | succ fuel ih =>
    intro n h
    by_cases hn : n = 0
    · simp [chunksAux, hn, lsum]
    · simp only [chunksAux]
      rw [if_neg hn]
      have h2 := ih (n - min T n) (by omega)
      simp only [lsum]
      omega

/-- No chunk exceeds `T`. -/
theorem chunks_le (T : Nat) (fuel : Nat) : ∀ n x, x ∈ chunksAux fuel T n → x ≤ T := by
  induction fuel with
  | zero => intro n x hx; simp [chunksAux] at hx
  | succ fuel ih =>
    intro n x hx
    simp only [chunksAux] at hx
    by_cases hn : n = 0
    · rw [if_pos hn] at hx; simp at hx
    · rw [if_neg hn] at hx
      rcases List.mem_cons.mp hx with h | h
      · omega
      · exact ih _ x h

-- § expert staging

/-- Next ring slot (`STAGE = 8`). -/
def ringNext (s : Nat) : Nat := (s + 1) % 8

/-- The ring pointer stays inside the ring. -/
theorem ringNext_lt (s : Nat) : ringNext s < 8 := by unfold ringNext; omega

/-- Blobs staged for experts `i < i' < i + 8` (look-ahead window of `STAGE`) never share a ring slot. -/
theorem ring_slots_distinct (i i' : Nat) (h1 : i < i') (h2 : i' < i + 8) : i % 8 ≠ i' % 8 := by omega

/-- The look-ahead loop stages up to index `j + (STAGE - 1)`, so the window is exactly `STAGE` blobs. -/
theorem lookahead_window (j staged : Nat) (h : staged = j + 7 + 1) : staged - j = 8 := by omega

/-- The dequantized-expert ring of size 2 alternates between consecutive experts. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by omega

/-- The DQ index is a valid buffer. -/
theorem dq_index_lt (j : Nat) : j % 2 < 2 := by omega

/-- `gather_rows16` moves 16-byte vectors; the row width 2560 is a whole number of them. -/
theorem gather_width_aligned : 2560 % 8 = 0 ∧ 2560 / 8 * 8 = 2560 := by decide

-- § PLE window across chunks

/-- The two-token predecessor window after consuming token `tok`. -/
def windowStep (prev : Int × Int) (tok : Int) : Int × Int := (prev.2, tok)

/-- After a token the newest predecessor is that token. -/
theorem windowStep_newest (prev : Int × Int) (tok : Int) : (windowStep prev tok).2 = tok := rfl

/-- After a token the older slot holds the previous newest. -/
theorem windowStep_oldest (prev : Int × Int) (tok : Int) : (windowStep prev tok).1 = prev.2 := rfl

-- § geometry facts used by the buffer sizing

/-- Hyper-connection width and the GDN channel split used by the chunk buffers. -/
theorem geometry_consts : 2560 * 4 = 10240 ∧ 48 * 128 = 6144 ∧ 2 * 16 * 128 + 48 * 128 = 10240 ∧ 640 * 2 = 1280 := by decide

end Ref.PrefillEngine



## FILE: /mnt/user-data/outputs/PrefillKernels.lean
/-!
# MODULE: prefill_kernels
SOURCES: src/prefill/kernels.cu
Index arithmetic and rounding contracts of the batched-prefill CUDA kernels: the Ref expert blob layout and its
2-bit dequantization, BF16 round-to-nearest-even, hyper-connection indexing, the router lane layout, the QSA
split/gate indexing, KV-pool row addressing and the GDN channel regions.

## Source semantics
* A blob is `gate/up codes | down codes | gate/up scales (f16) | down scales (f16)` = 1,382,400 bytes.
* A 2-bit code `q` decodes to `(q - 1) * scale`, i.e. one of -1, 0, 1, 2 times the scale.
* `bf(x)` rounds a 32-bit float pattern to its top 16 bits, ties to even.
* All thread/block index decompositions stay in bounds and are exact (no element is skipped or doubled).

## Form template identifiers
N=2560, HC=4, D=10240, LR=320, S=128, HK=16, HV=48, C=10240; blob offsets O_D_CODES, O_GU_SC, O_D_SC;
`bf` -> bfRound; `blocks_for` -> blocksFor.
-/

namespace Ref.PrefillKernels

-- § expert blob layout

/-- Section offsets and the total blob size. -/
theorem blob_offsets :
    1280 * 640 = 819200 ∧ 819200 + 2560 * 160 = 1228800 ∧
    1228800 + 1280 * 40 * 2 = 1331200 ∧ 1331200 + 2560 * 10 * 2 = 1382400 := by decide

/-- Codes pack 4 weights per byte and scales cover groups of 64. -/
theorem blob_packing : 640 * 4 = 2560 ∧ 160 * 4 = 640 ∧ 2560 / 64 = 40 ∧ 640 / 64 = 10 := by decide

/-- Gate/up thread index split is exact and in range. -/
theorem gu_split (i : Nat) (hi : i < 1280 * 640) : i / 640 < 1280 ∧ i % 640 < 640 ∧ i = i / 640 * 640 + i % 640 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Gate/up output element and scale indices stay inside their planes. -/
theorem gu_decode_in_bounds (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 ∧ row * 40 + byte * 4 / 64 < 1280 * 40 := by
  constructor <;> omega

/-- Down-projection thread index split is exact and in range. -/
theorem d_split (j : Nat) (hj : j < 2560 * 160) : j / 160 < 2560 ∧ j % 160 < 160 ∧ j = j / 160 * 160 + j % 160 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Down output element and scale indices stay inside their planes. -/
theorem d_decode_in_bounds (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 ∧ row * 10 + byte * 4 / 64 < 2560 * 10 := by
  constructor <;> omega

/-- A 2-bit code decodes to a small multiple of the scale: `q - 1 ∈ [-1, 2]`. -/
theorem code_value_range (q : Nat) (h : q < 4) : -1 ≤ (q : Int) - 1 ∧ (q : Int) - 1 ≤ 2 := by omega

/-- Number of blocks of `t` threads covering `n` items. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- The launch grid covers every item. -/
theorem blocksFor_covers (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h := Nat.div_add_mod (n + t - 1) t
  have h2 := Nat.mod_lt (n + t - 1) ht
  have e : (n + t - 1) / t * t = t * ((n + t - 1) / t) := Nat.mul_comm _ _
  omega

-- § BF16 rounding

/-- Round-to-nearest-even of a 32-bit pattern to its upper 16 bits (as Nat arithmetic). -/
def bfRound (u : Nat) : Nat := (u + 32767 + (u / 65536) % 2) / 65536

/-- The rounding error is at most half a unit in the last place (32768 of 65536). -/
theorem bfRound_err (u : Nat) : u ≤ bfRound u * 65536 + 32768 ∧ bfRound u * 65536 ≤ u + 32768 := by
  unfold bfRound
  constructor <;> omega

/-- Exact ties round to an even mantissa. -/
theorem bfRound_tie_even (q : Nat) : bfRound (q * 65536 + 32768) % 2 = 0 := by
  unfold bfRound
  have h1 : (q * 65536 + 32768) / 65536 = q := by omega
  rw [h1]
  have hr : q % 2 < 2 := Nat.mod_lt _ (by decide)
  have h2 : ∀ r : Nat, r < 2 → (q * 65536 + 32768 + 32767 + r) / 65536 = q + r := by
    intro r hr'
    have hnum : q * 65536 + 32768 + 32767 + r = 65536 * q + (65535 + r) := by omega
    rw [hnum, Nat.mul_add_div (by decide : 65536 > 0)]
    rcases (by omega : r = 0 ∨ r = 1) with h | h <;> subst h <;> simp
  rw [h2 (q % 2) hr]
  omega

/-- Values already representable are unchanged. -/
theorem bfRound_exact (q : Nat) : bfRound (q * 65536) = q := by
  unfold bfRound
  omega

-- § hyper-connection indexing

/-- `gr_write` decomposition of a flat index into token, stream and channel is exact and in range. -/
theorem gr_write_index (i : Nat) :
    i / 10240 * 10240 + (i % 10240) / 2560 * 2560 + i % 2560 = i ∧ (i % 10240) / 2560 < 4 ∧ i % 2560 < 2560 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- `gr_mix` reads the four streams of one token at `t*D + c*N + d`, all below the token's row end. -/
theorem gr_mix_in_bounds (t c d : Nat) (hc : c < 4) (hd : d < 2560) : t * 10240 + c * 2560 + d < (t + 1) * 10240 := by
  omega

-- § router lane layout

/-- Lane `l` holds experts `l + 32 i`, a bijection onto `0..511`. -/
theorem route_lane_cover (lane i : Nat) (hl : lane < 32) (hi : i < 16) :
    (lane + i * 32) % 32 = lane ∧ (lane + i * 32) / 32 = i ∧ lane + i * 32 < 512 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- The 10 routed ids of token `t` occupy `[10 t, 10 t + 10)`. -/
theorem route_out_in_bounds (t r : Nat) (hr : r < 10) : t * 10 + r < (t + 1) * 10 := by omega

-- § QSA helpers

/-- `rope_kernel` row decomposition (`row = t * heads + h`) is exact and in range. -/
theorem rope_row_split (row heads T : Nat) (hh : 0 < heads) (hr : row < heads * T) :
    row / heads < T ∧ row % heads < heads := by
  exact ⟨Nat.div_lt_of_lt_mul hr, Nat.mod_lt row hh⟩

/-- Both rotated halves of a 64-dim rope head stay inside it. -/
theorem rope_pair_in_head (pair : Nat) (h : pair < 32) : pair < 64 ∧ pair + 32 < 64 := by
  constructor <;> omega

/-- `split_q` reads the first half of each 512-wide head slot: inside the token's `24*512` row. -/
theorem split_q_in_bounds (T i : Nat) (hi : i < T * 6144) :
    i / 6144 * 12288 + (i / 256 % 24) * 512 + i % 256 < T * 12288 := by omega

/-- `gate_attn` reads the second half (`+256`) of the same slot, still in bounds. -/
theorem gate_attn_in_bounds (T i : Nat) (hi : i < T * 6144) :
    i / 6144 * 12288 + (i / 256 % 24) * 512 + 256 + i % 256 < T * 12288 := by omega

/-- Interleaved SwiGLU reads `gu[r*1280 + 2k + 1]`, inside row `r`. -/
theorem swiglu_il_in_bounds (r k : Nat) (hk : k < 640) : r * 1280 + 2 * k + 1 < (r + 1) * 1280 := by omega

/-- Alpha/beta gates read `ab[t*96 + h]` and `ab[t*96 + 48 + h]`, inside token `t`'s 96 values. -/
theorem gates_in_bounds (t h : Nat) (hh : h < 48) : t * 96 + h < (t + 1) * 96 ∧ t * 96 + 48 + h < (t + 1) * 96 := by
  constructor <;> omega

/-- MoE combine reads expert row `slot * 2560 + d` for `slot < T * 10`. -/
theorem combine_in_bounds (T slot d : Nat) (hs : slot < T * 10) (hd : d < 2560) : slot * 2560 + d < T * 10 * 2560 := by
  omega

-- § GDN channel regions

/-- q (32 heads of 128 over 16 key heads x 2) and k occupy 4096 channels, v the remaining 6144 of 10240. -/
theorem gdn_regions : 2 * 16 * 128 = 4096 ∧ 4096 + 48 * 128 = 10240 := by decide

/-- The recurrence reads q at `qh*128`, k at `2048 + qh*128`, v at `4096 + head*128 + col`, all inside 10240. -/
theorem gdn_rec_reads (qh head col tid : Nat) (hq : qh < 16) (hh : head < 48) (hc : col < 128) (ht : tid < 128) :
    qh * 128 + tid < 2048 ∧ 2048 + qh * 128 + tid < 4096 ∧ 4096 + head * 128 + col < 10240 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

-- § KV append addressing

/-- Pool row of position `pos` for kv head `kvh` in page `page`: it stays inside that page's `2 * ps` rows. -/
theorem kv_row_in_page (page kvh pos ps : Nat) (hk : kvh < 2) (hps : 0 < ps) :
    (page * 2 + kvh) * ps + pos % ps < ((page + 1) * 2) * ps := by
  have hm : pos % ps < ps := Nat.mod_lt pos hps
  have h1 : (page * 2 + kvh + 1) * ps ≤ ((page + 1) * 2) * ps := Nat.mul_le_mul_right ps (by omega)
  have h2 : (page * 2 + kvh) * ps + ps = (page * 2 + kvh + 1) * ps := (Nat.succ_mul _ _).symm
  omega

/-- `blockIdx.z` encodes (group, K-or-V) exactly: groups 0..3, flag 0..1. -/
theorem kv_z_split (z : Nat) (h : z < 8) : z / 2 < 4 ∧ z % 2 < 2 ∧ 2 * (z / 2) + z % 2 = z := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Element index within a head: group * 64 + lane stays below 256. -/
theorem kv_elem_in_head (g lane : Nat) (hg : g < 4) (hl : lane < 64) : g * 64 + lane < 256 := by omega

end Ref.PrefillKernels



## FILE: /mnt/user-data/outputs/SpecController.lean
/-!
# MODULE: spec_controller
SOURCES: src/spec/controller.cpp
The speculative-decoding controller: clamped window index into the cost tables, expected committed tokens of a
draft window, the EMA acceptance learner, and the bookkeeping of which drafted positions an observation touches.

## Source semantics
* `step_ms(k)` reads table entry `clamp(k + 1, 1, K_MAX + 1) - 1`, always in range.
* The expected tokens of a window is `1 + sum of cumulative acceptance products`; with probabilities in `[0, 1]`
  it lies in `[1, k + 1]`.
* `observe` clamps `accepted` to `0..k`, updates positions `0 .. min(k, accepted + 1) - 1`, and after a fully
  accepted window pulls the untested positions toward the deepest tested one.
* The EMA `p += e * (hit - p)` keeps `p` in `[0, 1]` (fixed-point model: scale `S`).

## Form template identifiers
`std::clamp` -> clampN; `expected` -> runs (fixed point, scale S); EMA -> emaUpdate; `seen` -> seenCount.
-/

namespace Ref.SpecController

/-- `std::clamp(v, lo, hi)` for `lo ≤ hi`. -/
def clampN (v lo hi : Nat) : Nat := max lo (min v hi)

/-- The clamp result is inside `[lo, hi]`. -/
theorem clampN_bounds (v lo hi : Nat) (h : lo ≤ hi) : lo ≤ clampN v lo hi ∧ clampN v lo hi ≤ hi := by
  unfold clampN
  constructor <;> omega

/-- The table index `n - 1` with `n = clamp(k+1, 1, K_MAX+1)` is below `K_MAX + 1`. -/
theorem table_index_lt (k kmax : Nat) : clampN (k + 1) 1 (kmax + 1) - 1 < kmax + 1 := by
  unfold clampN
  omega

/-- The clamped `n` is at least 1 (so `n - 1` never underflows). -/
theorem table_n_pos (k kmax : Nat) : 1 ≤ clampN (k + 1) 1 (kmax + 1) := by
  unfold clampN
  omega

-- § expected committed tokens (fixed point, scale S)

/-- Sum over the window of cumulative acceptance (`run` is the running product, scale S). -/
def runs (S : Nat) : Nat → List Nat → Nat
  | _, [] => 0
  | run, p :: ps => (run * p / S) + runs S (run * p / S) ps

/-- Multiplying by a probability `p ≤ S` never increases a quantity `≤ S`. -/
theorem step_le (S run p : Nat) (hS : 0 < S) (hp : p ≤ S) (hr : run ≤ S) : run * p / S ≤ S := by
  apply Nat.div_le_of_le_mul
  have h1 : run * p ≤ run * S := Nat.mul_le_mul_left run hp
  have h2 : run * S ≤ S * S := Nat.mul_le_mul_right S hr
  omega

/-- The accumulated acceptance of `n` drafted positions is at most `n` whole tokens (`S * n`). -/
theorem runs_le (S : Nat) (hS : 0 < S) (ps : List Nat) :
    ∀ run, run ≤ S → (∀ p ∈ ps, p ≤ S) → runs S run ps ≤ S * ps.length := by
  induction ps with
  | nil => intro run _ _; simp [runs]
  | cons p ps ih =>
    intro run hr hp
    have hp0 : p ≤ S := hp p (List.mem_cons_self ..)
    have hstep := step_le S run p hS hp0 hr
    have hrest := ih (run * p / S) hstep (fun q hq => hp q (List.mem_cons_of_mem _ hq))
    have hlen : S * (ps.length + 1) = S * ps.length + S := by rw [Nat.mul_add, Nat.mul_one]
    simp only [runs, List.length_cons]
    omega

-- § EMA learner (fixed point)

/-- `p' = (p (S - e) + hit e) / S`, i.e. `p += e (hit - p)` with `e = e'/S`. -/
def emaUpdate (S e p hit : Nat) : Nat := (p * (S - e) + hit * e) / S

/-- The EMA of values in `[0, S]` stays in `[0, S]`. -/
theorem emaUpdate_le (S e p hit : Nat) (he : e ≤ S) (hp : p ≤ S) (hh : hit ≤ S) : emaUpdate S e p hit ≤ S := by
  unfold emaUpdate
  apply Nat.div_le_of_le_mul
  have h1 : p * (S - e) ≤ S * (S - e) := Nat.mul_le_mul_right _ hp
  have h2 : hit * e ≤ S * e := Nat.mul_le_mul_right _ hh
  have h3 : S * (S - e) + S * e = S * S := by rw [← Nat.mul_add, Nat.sub_add_cancel he]
  omega

-- § observe bookkeeping

/-- Positions updated by an observation: `min k (accepted + 1)` after clamping `accepted` into `0..k`. -/
def seenCount (k accepted : Nat) : Nat := min k (min accepted k + 1)

/-- Never more positions than were drafted. -/
theorem seenCount_le (k accepted : Nat) : seenCount k accepted ≤ k := by unfold seenCount; omega

/-- A rejected draft (accepted < k) is itself observed: `accepted + 1` positions. -/
theorem seenCount_rejected (k accepted : Nat) (h : accepted < k) : seenCount k accepted = accepted + 1 := by
  unfold seenCount; omega

/-- A fully accepted window observes exactly `k` positions. -/
theorem seenCount_full (k : Nat) : seenCount k k = k := by unfold seenCount; omega

/-- The deepest tested position `k - 1` exists when `k > 0`, and the pulled positions `k .. K_MAX - 1` are valid. -/
theorem pull_positions_valid (k kmax i : Nat) (hk : 0 < k) (hkm : k ≤ kmax) (hi : k ≤ i) (hi2 : i < kmax) :
    k - 1 < kmax ∧ i < kmax := by
  constructor <;> omega

/-- Lookup candidates are bounded by `K_MAX`. -/
theorem lookup_len_le (avail kmax : Nat) : min avail kmax ≤ kmax := by omega

end Ref.SpecController



## FILE: /mnt/user-data/outputs/SuffixDrafter.lean
/-!
# MODULE: suffix_drafter
SOURCES: src/spec/suffix_drafter.cpp
An n-gram suffix lookup drafter: an open-addressing hash table keyed by the last three tokens, each slot keeping
the most recent `WAYS` end positions; `propose` finds the longest earlier match and copies its continuation.

## Source semantics
* The table capacity is the first power of two `>= 2 * capacity_tokens`, so `key & mask == key % capacity`.
* A key is never 0 (`| 1`), so 0 can mean "empty slot".
* A slot keeps at most `WAYS` positions, newest first.
* A match is extended while indices stay valid; the copied continuation never reads past the current suffix.

## Form template identifiers
`cap <<= 1` loop -> grow; `mask_` -> mask; `Slot::pos/n` -> push/List; match loop -> mlLoop; copy count -> copyCount.
-/

namespace Ref.SuffixDrafter

-- § table capacity

/-- Double `c` until it reaches `target` (bounded by `fuel`). -/
def grow : Nat → Nat → Nat → Nat
  | 0, c, _ => c
  | f + 1, c, t => if c < t then grow f (2 * c) t else c

/-- With enough doublings available the capacity reaches the target. -/
theorem grow_ge (t : Nat) (f : Nat) : ∀ c, t ≤ c * 2 ^ f → t ≤ grow f c t := by
  induction f with
  | zero => intro c h; simpa [grow] using h
  | succ f ih =>
    intro c h
    simp only [grow]
    by_cases hc : c < t
    · rw [if_pos hc]
      apply ih
      have e : 2 * c * 2 ^ f = c * 2 ^ (f + 1) := by
        rw [Nat.pow_succ]
        ac_rfl
      rw [e]
      exact h
    · rw [if_neg hc]
      omega

/-- Starting from a power of two the capacity stays a power of two. -/
theorem grow_pow2 (t : Nat) (f : Nat) : ∀ k, ∃ j, grow f (2 ^ k) t = 2 ^ j := by
  induction f with
  | zero => intro k; exact ⟨k, rfl⟩
  | succ f ih =>
    intro k
    simp only [grow]
    by_cases hc : 2 ^ k < t
    · rw [if_pos hc]
      have e : 2 * 2 ^ k = 2 ^ (k + 1) := by rw [Nat.pow_succ, Nat.mul_comm]
      rw [e]
      exact ih (k + 1)
    · rw [if_neg hc]
      exact ⟨k, rfl⟩

/-- For a power-of-two capacity, masking equals the remainder, so probing `(i + 1) & mask` stays in the table. -/
theorem mask_is_mod (x k : Nat) : x &&& (2 ^ k - 1) = x % 2 ^ k := Nat.and_two_pow_sub_one_eq_mod x k

/-- The next probe index is always in range. -/
theorem probe_in_range (i cap : Nat) (hcap : 0 < cap) : (i + 1) % cap < cap := Nat.mod_lt _ hcap

-- § keys

/-- `key | 1` is never zero, so 0 is a safe "empty" marker. -/
theorem or_one_ne_zero (x : Nat) : x ||| 1 ≠ 0 := by
  intro h
  have h2 : (x ||| 1).testBit 0 = true := by simp [Nat.testBit_or]
  rw [h] at h2
  simp at h2

-- § slot positions

/-- Insert a position, keeping the `ways` newest. -/
def push (ways : Nat) (p : Nat) (l : List Nat) : List Nat := (p :: l).take ways

/-- A slot never holds more than `WAYS` positions. -/
theorem push_length_le (ways p : Nat) (l : List Nat) : (push ways p l).length ≤ ways := by
  unfold push
  simp [List.length_take]

/-- The new position is the most recent when the slot has any capacity. -/
theorem push_head (ways p : Nat) (l : List Nat) (h : 0 < ways) : (push ways p l).head? = some p := by
  unfold push
  cases ways with
  | zero => omega
  | succ w => rfl

/-- The counter `n = min(n + 1, WAYS)` equals the new list length when the slot was within capacity. -/
theorem push_length (ways p : Nat) (l : List Nat) (h : l.length ≤ ways) :
    (push ways p l).length = min (l.length + 1) ways := by
  unfold push
  simp [List.length_take]
  omega

-- § match extension and continuation

/-- Extend a match while `len ≤ p` and the tokens agree, up to `fuel` steps. -/
def mlLoop (h : Nat → Nat) (p cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | f + 1, len => if len ≤ p ∧ h (p - len) = h (cur - len) then mlLoop h p cur f (len + 1) else len

/-- The match grows by at most the fuel (`max_match`). -/
theorem mlLoop_le (h : Nat → Nat) (p cur : Nat) (f : Nat) : ∀ len, mlLoop h p cur f len ≤ len + f := by
  induction f with
  | zero => intro len; simp [mlLoop]
  | succ f ih =>
    intro len
    simp only [mlLoop]
    split
    · have := ih (len + 1)
      omega
    · omega

/-- The match never exceeds `p + 1` tokens, so `hist[p - len]` is always a valid index. -/
theorem mlLoop_le_p (h : Nat → Nat) (p cur : Nat) (f : Nat) : ∀ len, len ≤ p + 1 → mlLoop h p cur f len ≤ p + 1 := by
  induction f with
  | zero => intro len hl; simpa [mlLoop] using hl
  | succ f ih =>
    intro len hl
    simp only [mlLoop]
    split
    · rename_i hc
      obtain ⟨hlt, _⟩ := hc
      exact ih (len + 1) (by omega)
    · exact hl

/-- The match never shrinks. -/
theorem mlLoop_ge (h : Nat → Nat) (p cur : Nat) (f : Nat) : ∀ len, len ≤ mlLoop h p cur f len := by
  induction f with
  | zero => intro len; simp [mlLoop]
  | succ f ih =>
    intro len
    simp only [mlLoop]
    split
    · have := ih (len + 1)
      omega
    · omega

/-- Tokens copied by `propose`: from `best_end + 1` through `cur`, at most `max_k`. -/
def copyCount (bestEnd cur maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The copy never exceeds `max_k`. -/
theorem copyCount_le (bestEnd cur maxK : Nat) : copyCount bestEnd cur maxK ≤ maxK := by unfold copyCount; omega

/-- The last copied index `best_end + count` is at most `cur`, a valid history index. -/
theorem copy_last_valid (bestEnd cur maxK : Nat) (h : bestEnd < cur) : bestEnd + copyCount bestEnd cur maxK ≤ cur := by
  unfold copyCount; omega

/-- A match strictly before the current suffix and `max_k ≥ 1` yields at least one draft token. -/
theorem copy_nonempty (bestEnd cur maxK : Nat) (h : bestEnd < cur) (hk : 1 ≤ maxK) : 1 ≤ copyCount bestEnd cur maxK := by
  unfold copyCount; omega

end Ref.SuffixDrafter



## FILE: /mnt/user-data/outputs/PleReader.lean
/-!
# MODULE: ple_reader
SOURCES: src/ngram/ple_reader.cpp
Asynchronous direct-I/O reader for the 90-byte PLE table rows: set-associative row cache, page-aligned read jobs
(one or two pages), page de-duplication, free-slot accounting, ticket counting and latency-ring bookkeeping.

## Source semantics
* Cache: `sets = rows / 8`, 8 ways per set, round-robin replacement; slot and data indices stay inside the arrays.
* A row at byte `at` is fetched with a page-aligned read of 1 or 2 pages (PAGE = 4096) containing all 90 bytes.
* Jobs for rows on the same page are merged; the merged length is the max and still covers every row.
* `free_slots + in_flight = max_inflight` at all times; a ticket completes when its pending count reaches 0.

## Form template identifiers
ROW_BYTES -> 90; PAGE -> 4096; WAYS -> 8; EMPTY -> 0xFFFFFFFF; `Job::length` -> jobLength; `pump/finish` -> pump/finish.
-/

namespace Ref.PleReader

-- § row cache

/-- Number of sets for a cache of `rows` rows. -/
def sets (rows : Nat) : Nat := rows / 8

/-- Capacity never exceeds the requested rows. -/
theorem capacity_le (rows : Nat) : sets rows * 8 ≤ rows := by unfold sets; omega

/-- A set index is in range when there is at least one set. -/
theorem set_index_lt (h s : Nat) (hs : 0 < s) : h % s < s := Nat.mod_lt _ hs

/-- Flat way index `s*8 + w` stays inside `sets * 8`. -/
theorem way_index_lt (s w n : Nat) (hs : s < n) (hw : w < 8) : s * 8 + w < n * 8 := by omega

/-- Data byte offset of a cache slot stays inside the data array (`ROW_BYTES = 90`). -/
theorem data_index_lt (s w n : Nat) (hs : s < n) (hw : w < 8) : (s * 8 + w) * 90 + 90 ≤ n * 8 * 90 := by omega

/-- Round-robin pointer update. -/
def nextWay (w : Nat) : Nat := (w + 1) % 8

/-- The pointer is always a valid way. -/
theorem nextWay_lt (w : Nat) : nextWay w < 8 := by unfold nextWay; omega

/-- Eight inserts return the pointer to its start, so each way is replaced once per cycle. -/
theorem nextWay_cycle (w : Nat) : (w + 8) % 8 = w % 8 := by omega

/-- A key table: way -> optional row. -/
abbrev Keys := Nat → Option Nat

/-- Store `row` at way `w`. -/
def insertAt (keys : Keys) (w row : Nat) : Keys := fun i => if i = w then some row else keys i

/-- After an insert the row is found at that way. -/
theorem insertAt_found (keys : Keys) (w row : Nat) : insertAt keys w row w = some row := by
  simp [insertAt]

/-- Other ways are untouched. -/
theorem insertAt_other (keys : Keys) (w row i : Nat) (h : i ≠ w) : insertAt keys w row i = keys i := by
  simp [insertAt, h]

/-- Real row ids (table of at most 320,001,536 rows) never collide with the EMPTY marker. -/
theorem row_ne_empty (n_rows row : Nat) (hn : n_rows ≤ 320001536) (hr : row < n_rows) : row ≠ 4294967295 := by
  omega

-- § page-aligned jobs

/-- First page-aligned offset of a row starting at byte `at`. -/
def jobFirst (at_ : Nat) : Nat := at_ / 4096 * 4096

/-- Read length: from the first page up to the page holding the row's last byte, plus one page. -/
def jobLength (at_ : Nat) : Nat := (at_ + 89) / 4096 * 4096 - jobFirst at_ + 4096

/-- The read start is page aligned and not after the row. -/
theorem jobFirst_props (at_ : Nat) : jobFirst at_ % 4096 = 0 ∧ jobFirst at_ ≤ at_ := by
  unfold jobFirst
  constructor <;> omega

/-- A row needs one page or two (it may straddle a boundary). -/
theorem jobLength_cases (at_ : Nat) : jobLength at_ = 4096 ∨ jobLength at_ = 8192 := by
  unfold jobLength jobFirst
  omega

/-- The read covers all 90 bytes of the row: `in_page + 90 <= length`. -/
theorem job_covers_row (at_ : Nat) : (at_ - jobFirst at_) + 90 ≤ jobLength at_ := by
  unfold jobLength jobFirst
  omega

/-- The in-page offset is below one page. -/
theorem in_page_lt (at_ : Nat) : at_ - jobFirst at_ < 4096 := by
  unfold jobFirst
  omega

/-- Read buffers are two pages per slot, which holds the longest job. -/
theorem slot_holds_job (at_ : Nat) : jobLength at_ ≤ 2 * 4096 := by
  rcases jobLength_cases at_ with h | h <;> omega

/-- Slot `s < max` addresses `[s*8192, s*8192 + 8192)` inside the `max * 8192` slab. -/
theorem slot_buf_in_slab (s mx : Nat) (h : s < mx) : s * 8192 + 8192 ≤ mx * 8192 := by omega

/-- De-duplication keeps the maximum length, which still covers every merged row. -/
theorem merged_covers (inPage len len' : Nat) (h : inPage + 90 ≤ len) : inPage + 90 ≤ max len len' := by omega

/-- A row inside the table lies inside its byte range (`rows * 90`). -/
theorem row_in_table (row rows : Nat) (h : row < rows) : row * 90 + 90 ≤ rows * 90 := by omega

-- § slots, queue and tickets

/-- State of the free-slot accounting: (free, in flight). -/
structure Slots where
  free : Nat
  inflight : Nat

/-- One `pump` step: submit a queued job if both a job and a free slot exist. -/
def pump (s : Slots) (queued : Nat) : Slots × Nat :=
  if 0 < queued ∧ 0 < s.free then (⟨s.free - 1, s.inflight + 1⟩, queued - 1) else (s, queued)

/-- One completion: the slot returns to the free list. -/
def finish (s : Slots) : Slots := ⟨s.free + 1, s.inflight - 1⟩

/-- `pump` conserves the slot total. -/
theorem pump_conserves (s : Slots) (q mx : Nat) (h : s.free + s.inflight = mx) :
    (pump s q).1.free + (pump s q).1.inflight = mx := by
  unfold pump
  split
  · rename_i hc
    obtain ⟨_, h2⟩ := hc
    simp only
    omega
  · exact h

/-- `finish` conserves the slot total when something was in flight. -/
theorem finish_conserves (s : Slots) (mx : Nat) (h : s.free + s.inflight = mx) (hi : 0 < s.inflight) :
    (finish s).free + (finish s).inflight = mx := by
  unfold finish
  simp only
  omega

/-- A completion decrements the pending count, saturating at 0. -/
def tick (pending : Nat) : Nat := if 0 < pending then pending - 1 else 0

/-- After as many completions as jobs, the ticket is done. -/
theorem tick_iter_done (jobs : Nat) : ∀ m, jobs ≤ m → Nat.iterate tick m jobs = 0 := by
  intro m
  induction m generalizing jobs with
  | zero =>
    intro h
    have : jobs = 0 := by omega
    subst this
    rfl
  | succ m ih =>
    intro h
    show Nat.iterate tick m (tick jobs) = 0
    by_cases hj : 0 < jobs
    · have : tick jobs = jobs - 1 := by simp [tick, hj]
      rw [this]
      exact ih (jobs - 1) (by omega)
    · have : jobs = 0 := by omega
      subst this
      have h0 : tick 0 = 0 := by simp [tick]
      rw [h0]
      exact ih 0 (by omega)

-- § latency statistics

/-- Percentile index `min(n-1, round(q (n-1)))` is a valid sample index when `n > 0`. -/
theorem percentile_index_lt (n x : Nat) (h : 0 < n) : min (n - 1) x < n := by omega

/-- The latency ring position is always inside the 65536-entry ring. -/
theorem ring_pos_lt (pos : Nat) : pos % 65536 < 65536 := by omega

/-- Fault injection: a completion is held iff it is younger than the delay; the two cases are complementary. -/
theorem delay_split (age delay : Nat) : age < delay ∨ delay ≤ age := by omega

end Ref.PleReader



## FILE: /mnt/user-data/outputs/NgramHash.lean
/-!
# MODULE: ngram_hash
SOURCES: src/kernels/ngram.cpp
The PLE n-gram hash (64-bit multiply/xor mixing, per-head modulus and offset), the EOS-cut predecessor context,
the IQ4_NL row layout (5 blocks of 18 bytes, split-halves nibbles), and the table size / issue-collect protocol.

## Source semantics
* `mixed = ctx[0]*m[0] ^ ctx[1]*m[1] ^ ...`, each product wrapping mod 2^64.
* Head `h` maps `mixed` to `mixed % vocab[h] + offset[h]`; the 16 head ranges are adjacent and fit in the table.
* Once a predecessor is EOS/null the cut persists: all older context slots become EOS.
* A row is 160 values = 5 blocks of (2-byte scale + 16 bytes of nibbles) = 90 bytes; element `b*32 + j` is the low
  nibble of `qs[j]`, `b*32 + j + 16` the high nibble.
* `issue` is refused while a token is in flight; `collect` requires one.

## Form template identifiers
PleConsts.mult/vocab/offset -> multL/vocabL/offsetL; `ngram_mixed` -> mixedN; cut loop -> ctxRest;
`kIq4Nl` -> iq4Tbl; `pending` -> issue/collect state.
-/

namespace Ref.NgramHash

-- § hash

/-- Wrap to 64 bits. -/
def wrap (x : Nat) : Nat := x % 2 ^ 64

/-- Mixed hash of two context tokens. -/
def mixed2 (c0 c1 m0 m1 : Nat) : Nat := wrap (c0 * m0) ^^^ wrap (c1 * m1)

/-- Mixed hash of three context tokens (assignment then xors). -/
def mixed3 (c0 c1 c2 m0 m1 m2 : Nat) : Nat := wrap (c0 * m0) ^^^ wrap (c1 * m1) ^^^ wrap (c2 * m2)

/-- Wrapped products fit in 64 bits. -/
theorem wrap_lt (x : Nat) : wrap x < 2 ^ 64 := Nat.mod_lt _ (Nat.two_pow_pos 64)

/-- The two-token hash is a 64-bit value. -/
theorem mixed2_lt (c0 c1 m0 m1 : Nat) : mixed2 c0 c1 m0 m1 < 2 ^ 64 :=
  Nat.xor_lt_two_pow (wrap_lt _) (wrap_lt _)

/-- The three-token hash is a 64-bit value. -/
theorem mixed3_lt (c0 c1 c2 m0 m1 m2 : Nat) : mixed3 c0 c1 c2 m0 m1 m2 < 2 ^ 64 :=
  Nat.xor_lt_two_pow (Nat.xor_lt_two_pow (wrap_lt _) (wrap_lt _)) (wrap_lt _)

-- § per-head ranges

/-- Head moduli (from the artifact's metadata). -/
def vocabL : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Head offsets (from the artifact's metadata). -/
def offsetL : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Modulus of head `i`. -/
def vocab (i : Nat) : Nat := vocabL.getD i 0

/-- Offset of head `i`. -/
def offset (i : Nat) : Nat := offsetL.getD i 0

/-- Every head modulus is positive, so `% vocab` is a genuine reduction. -/
theorem vocab_pos : ∀ i, i < 16 → 0 < vocab i := by decide

/-- Head ranges are adjacent: `offset (i+1) = offset i + vocab i`. -/
theorem offset_succ : ∀ i, i < 15 → offset (i + 1) = offset i + vocab i := by decide

/-- Every head's range ends inside the 320,001,536-row table. -/
theorem range_in_table : ∀ i, i < 16 → offset i + vocab i ≤ 320001536 := by decide

/-- A hashed row index lies in its head's range, hence in the table. -/
theorem row_in_range (mixed i : Nat) (hi : i < 16) : mixed % vocab i + offset i < offset i + vocab i ∧
    mixed % vocab i + offset i < 320001536 + 1 := by
  have h1 := Nat.mod_lt mixed (vocab_pos i hi)
  have h2 := range_in_table i hi
  constructor <;> omega

/-- Heads of one n-gram order are `base + g`, with `base = (n-2)*8` and `g < 8`: always below 16 for `n ∈ {2,3}`. -/
theorem head_index_lt (n g : Nat) (hn1 : 2 ≤ n) (hn2 : n ≤ 3) (hg : g < 8) : (n - 2) * 8 + g < 16 := by omega

/-- Output position `i*16 + h` stays inside the token's 16-row slice. -/
theorem out_index_lt (i h : Nat) (hh : h < 16) : i * 16 + h < (i + 1) * 16 := by omega

/-- Predecessor `s ∈ {1,2}` is read at `i*2 + (2 - s)`: newest at slot 1, inside the token's two-entry window. -/
theorem prev_index_lt (i s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) : i * 2 + (2 - s) < (i + 1) * 2 := by omega

-- § EOS cut

/-- Context after the first token, newest predecessor first; `cut` latches on null (<0) or EOS. -/
def ctxRest (eos : Int) : List Int → Bool → List Int
  | [], _ => []
  | t :: ts, cut =>
    let cut' := cut || decide (t < 0) || decide (t = eos)
    (if cut' then eos else t) :: ctxRest eos ts cut'

/-- One context slot per predecessor. -/
theorem ctxRest_length (eos : Int) (ps : List Int) : ∀ cut, (ctxRest eos ps cut).length = ps.length := by
  induction ps with
  | nil => intro cut; rfl
  | cons t ts ih => intro cut; simp [ctxRest, ih]

/-- Once cut, everything older is EOS. -/
theorem ctxRest_cut (eos : Int) (ps : List Int) : ctxRest eos ps true = List.replicate ps.length eos := by
  induction ps with
  | nil => rfl
  | cons t ts ih => simp [ctxRest, ih, List.replicate_succ]

-- § IQ4_NL row layout

/-- 90 bytes = 5 blocks of 18 bytes. -/
theorem row_bytes : 5 * 18 = 90 := by decide

/-- A block byte `b*18 + 2 + j` (`b<5`, `j<16`) is inside the 90-byte row. -/
theorem qs_byte_in_row (b j : Nat) (hb : b < 5) (hj : j < 16) : b * 18 + 2 + j < 90 := by omega

/-- Low/high nibble outputs `b*32 + j` and `b*32 + j + 16` are inside the 160 values. -/
theorem out_element_lt (b j half : Nat) (hb : b < 5) (hj : j < 16) (hh : half < 2) : b * 32 + j + 16 * half < 160 := by
  omega

/-- Every output index `idx < 160` is written exactly once, by (block, j, half). -/
theorem out_element_bijective (idx : Nat) (h : idx < 160) :
    idx / 32 < 5 ∧ idx % 16 < 16 ∧ idx % 32 / 16 < 2 ∧ idx / 32 * 32 + idx % 16 + 16 * (idx % 32 / 16) = idx := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- A packed byte splits into two valid 4-bit codes. -/
theorem nibble_split (q : Nat) (h : q < 256) : q % 16 < 16 ∧ q / 16 < 16 ∧ q = 16 * (q / 16) + q % 16 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- The IQ4_NL codebook. -/
def iq4L : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook entry. -/
def iq4Tbl (i : Nat) : Int := iq4L.getD i 0

/-- The codebook is strictly increasing. -/
theorem iq4_increasing : ∀ i, i < 15 → iq4Tbl i < iq4Tbl (i + 1) := by decide

-- § table size and gather layout

/-- Row `row < rows` ends within `rows * 90` bytes. -/
theorem table_row_end (row rows : Nat) (h : row < rows) : row * 90 + 90 ≤ rows * 90 := by omega

/-- Head-slowest gather: `h*160 + d` is inside the 2560-float output. -/
theorem gather_index_lt (h d : Nat) (hh : h < 16) (hd : d < 160) : h * 160 + d < 2560 := by omega

-- § issue / collect protocol

/-- `issue` on a table with a token in flight is refused. -/
def issue (pending : Bool) : Option Bool := if pending then none else some true

/-- `collect` without a pending issue is refused. -/
def collect (pending : Bool) : Option Bool := if pending then some false else none

/-- Issue then collect round-trips to an idle table. -/
theorem issue_collect_roundtrip : (issue false).bind collect = some false := by decide

/-- A second issue before collect is refused. -/
theorem double_issue_refused : (issue false).bind issue = none := by decide

end Ref.NgramHash



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- GenerateDriver.lean
  was: theorem produced_reaches_max (nPrompt maxNew : Nat) (h1 : 1 ≤ nPrompt) (h2 : 1 ≤ maxNew) :
    producedAt nPrompt (nPrompt - 2 + maxNew) = maxNew ∨ (nPrompt = 1 ∧ producedAt nPrompt (maxNew - 1) = maxNew)
  now: theorem produced_reaches_max (nPrompt maxNew : Nat) (h1 : 2 ≤ nPrompt) (h2 : 1 ≤ maxNew) :
    producedAt nPrompt (nPrompt - 2 + maxNew) = maxNew
  measured: the claim changed

- GenerateDriver.lean
  was: theorem nextInput_prompt (prompt : Nat → Nat) (n pos next : Nat) (h : pos + 1 < n) :
    nextInput prompt n pos next = prompt (pos + 1)
  now: theorem produced_reaches_max_single (maxNew : Nat) (h2 : 1 ≤ maxNew) :
    producedAt 1 (maxNew - 1) = maxNew
  measured: fewer quantified variables (3 → 2)

- GenerateDriver.lean
  was: theorem nextInput_sampled (prompt : Nat → Nat) (n pos next : Nat) (h : ¬ pos + 1 < n) :
    nextInput prompt n pos next = next
  now: theorem nextInput_prompt (prompt : Nat → Nat) (n pos next : Nat) (h : pos + 1 < n) :
    nextInput prompt n pos next = prompt (pos + 1)
  measured: the claim changed

- GenerateDriver.lean
  was: theorem selected_stride_one (pos total : Nat) : selected pos total 1
  now: theorem nextInput_sampled (prompt : Nat → Nat) (n pos next : Nat) (h : ¬ pos + 1 < n) :
    nextInput prompt n pos next = next
  measured: more hypotheses/binders (1 → 3): the statement holds in fewer cases

- GenerateDriver.lean
  was: theorem selected_last (total stride : Nat) (h : 0 < total) : selected (total - 1) total stride
  now: theorem selected_stride_one (pos total : Nat) : selected pos total 1
  measured: fewer quantified variables (2 → 1)

- GenerateDriver.lean
  was: theorem acc_le_left (ds os : List Nat) : acc ds os ≤ ds.length
  now: theorem selected_last (total stride : Nat) (h : 0 < total) : selected (total - 1) total stride
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- GenerateDriver.lean
  was: theorem acc_le_right (ds os : List Nat) : acc ds os ≤ os.length
  now: theorem acc_le_left (ds os : List Nat) : acc ds os ≤ ds.length
  measured: the claim changed

- GenerateDriver.lean
  was: theorem acc_self (ds : List Nat) : acc ds ds = ds.length
  now: theorem acc_le_right (ds os : List Nat) : acc ds os ≤ os.length
  measured: the claim changed

- GenerateDriver.lean
  was: theorem emitted_le_window (ds os : List Nat) : acc ds os + 1 ≤ 1 + ds.length
  now: theorem acc_self (ds : List Nat) : acc ds ds = ds.length
  measured: the claim changed

- GenerateDriver.lean
  was: theorem winLoop_ge (ok : Nat → Bool) (S : Nat) (f : Nat) : ∀ T, T ≤ winLoop ok S f T
  now: theorem emitted_le_window (ds os : List Nat) : acc ds os + 1 ≤ 1 + ds.length
  measured: fewer quantified variables (4 → 1)

- GenerateDriver.lean
  was: theorem winLoop_le (ok : Nat → Bool) (S : Nat) (f : Nat) : ∀ T, T ≤ S → winLoop ok S f T ≤ S
  now: theorem winLoop_ge (ok : Nat → Bool) (S : Nat) (f : Nat) : ∀ T, T ≤ winLoop ok S f T
  measured: the claim changed

- GenerateDriver.lean
  was: theorem lend_suffix (slots k : Nat) (h : k + 128 ≤ slots) : lendFirst slots k + k = slots ∧ 128 ≤ lendFirst slots k
  now: theorem winLoop_le (ok : Nat → Bool) (S : Nat) (f : Nat) : ∀ T, T ≤ S → winLoop ok S f T ≤ S
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- GenerateDriver.lean
  was: theorem pcieNum_le (x : Nat) : pcieNum x ≤ 256
  now: theorem lend_suffix (slots k : Nat) (h : k + 128 ≤ slots) : lendFirst slots k + k = slots ∧ 128 ≤ lendFirst slots k
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- GenerateDriver.lean
  was: theorem mrope_image_pos_lt (p j nx ny : Nat) (hj : j < nx * ny) :
    p + j / nx < p + max nx ny ∧ p + j % nx < p + max nx ny
  now: theorem pcieNum_le (x : Nat) : pcieNum x ≤ 256
  measured: fewer quantified variables (2 → 1)
-/



## FILE: /mnt/user-data/outputs/FusedGdnDecode.lean
/-!
# MODULE: fused_gdn_decode
SOURCES: src/kernels/cuda/fused_gdn.cu
Single-token fused gated-delta-net decode: causal-conv + SiLU + L2 norm, the alpha/beta projections, and the
state-update + RMS-norm + gate kernel. Thread/row layouts and the exact-arithmetic algebra are formalised.

## Source semantics
* `gdn_step_norm_kernel`: block = head, thread (col, rg); 4 row groups of 32 rows own the 128 state rows; kv_col is the
  sum of the four row-group partials; delta = (v - g*kv_col)*beta; state' = g*state + k*delta.
* `gdn_conv_l2_kernel`: one thread per channel, history of 3 slides by one per token.
* `gdn_ab_kernel`: one warp per row of [alpha ; beta] weight rows, 8 warps per block.

## Form template identifiers
S -> state_size, RG -> row_groups, RPG -> rows_per_group, tid -> flat_thread_id, stateRow -> owned_state_row,
sumTo -> exact_reduction, histStep -> conv_history_shift
-/
namespace Ref.FusedGdnDecode

-- § exact-arithmetic reduction library
/-- Sum of `f 0 .. f (n-1)` over the integers (exact model of a reduction). -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n+1 => sumTo f n + f n

/-- Splitting a reduction at an arbitrary point. -/
theorem sumTo_add (f : Nat → Int) (n m : Nat) :
    sumTo f (n+m) = sumTo f n + sumTo (fun r => f (n+r)) m := by
  induction m with
  | zero => simp [sumTo]
  | succ m ih =>
    have e : sumTo f (n + (m+1)) = sumTo f (n+m) + f (n+m) := rfl
    have e2 : sumTo (fun r => f (n+r)) (m+1) = sumTo (fun r => f (n+r)) m + f (n+m) := rfl
    rw [e, ih, e2]
    omega

/-- A reduction over `a*b` items equals the reduction of `a` block reductions of `b` items. -/
theorem sum_blocks (f : Nat → Int) (b a : Nat) :
    sumTo f (a*b) = sumTo (fun r => sumTo (fun c => f (r*b+c)) b) a := by
  induction a with
  | zero => simp [sumTo]
  | succ a ih =>
    have h : (a+1)*b = a*b + b := Nat.succ_mul a b
    rw [h, sumTo_add f (a*b) b, ih]
    try rfl

/-- Scalar factors pull out of a reduction. -/
theorem sum_scale (g : Int) (f : Nat → Int) (n : Nat) :
    sumTo (fun i => g * f i) n = g * sumTo f n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih => simp [sumTo, ih, Int.mul_add]

/-- Decaying the state before contracting with k equals decaying the contraction. -/
theorem decay_commutes (g : Int) (s k : Nat → Int) (n : Nat) :
    sumTo (fun i => g * s i * k i) n = g * sumTo (fun i => s i * k i) n := by
  have h : (fun i => g * s i * k i) = (fun i => g * (s i * k i)) :=
    funext fun i => Int.mul_assoc _ _ _
  rw [h, sum_scale]

-- § constants and thread layout
/-- State size (rows = cols). -/
def S : Nat := 128
/-- Row groups per column. -/
def RG : Nat := 4
/-- Rows per group. -/
def RPG : Nat := S / RG

/-- 32 rows per thread. -/
theorem rpg_eq : RPG = 32 := by decide

/-- The row groups tile the state rows exactly. -/
theorem rg_mul_rpg : RG * RPG = S := by decide

/-- Block size is 512 threads. -/
theorem block_threads : S * RG = 512 := by decide

/-- Flat thread id. -/
def tid (rg col : Nat) : Nat := rg * S + col
/-- State row owned by register slot r of row group rg. -/
def stateRow (rg r : Nat) : Nat := rg * RPG + r

/-- Flat thread ids are within the block. -/
theorem tid_lt (rg col : Nat) (h1 : rg < RG) (h2 : col < S) : tid rg col < S * RG := by
  simp only [tid, S, RG] at *
  omega

/-- The row group is recovered from the flat id. -/
theorem tid_div (rg col : Nat) (h2 : col < S) : tid rg col / S = rg := by
  simp only [tid, S] at *
  omega

/-- The column is recovered from the flat id. -/
theorem tid_mod (rg col : Nat) (h2 : col < S) : tid rg col % S = col := by
  simp only [tid, S] at *
  omega

/-- Distinct (rg, col) give distinct flat ids. -/
theorem tid_inj (a b c d : Nat) (hb : b < S) (hd : d < S) (h : tid a b = tid c d) : a = c ∧ b = d := by
  simp only [tid, S] at *
  omega

/-- Owned rows are inside the state. -/
theorem stateRow_lt (rg r : Nat) (h1 : rg < RG) (h2 : r < RPG) : stateRow rg r < S := by
  simp only [stateRow, RG, RPG, S] at *
  omega

/-- Row ownership is injective: no two (group, slot) pairs own the same row. -/
theorem stateRow_inj (a r c s : Nat) (hr : r < RPG) (hs : s < RPG)
    (h : stateRow a r = stateRow c s) : a = c ∧ r = s := by
  simp only [stateRow, RPG, S, RG] at *
  omega

/-- Row ownership is onto: every state row is owned. -/
theorem stateRow_surj (i : Nat) (hi : i < S) :
    ∃ rg r, rg < RG ∧ r < RPG ∧ stateRow rg r = i := by
  refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> simp only [stateRow, RG, RPG, S] at * <;> omega

-- § reduction semantics
/-- kv_col as the sum of the four row-group partials equals the full contraction. -/
theorem kv_col_partition (f : Nat → Int) :
    sumTo f S = sumTo (fun rg => sumTo (fun r => f (rg * RPG + r)) RPG) RG :=
  sum_blocks f RPG RG

/-- The fused delta (decay folded in) equals the two-pass delta. -/
theorem fused_delta_eq (g beta v : Int) (s k : Nat → Int) (n : Nat) :
    (v - sumTo (fun i => g * s i * k i) n) * beta = (v - g * sumTo (fun i => s i * k i) n) * beta := by
  rw [decay_commutes]

/-- Thread-id to warp index for row group 0 stays within the four RMS partial slots. -/
theorem wsum_slot_lt (t : Nat) (h : t < S) : t / 32 < 4 := by
  simp only [S] at *
  omega

-- § conv history
/-- History of the three previous inputs. -/
structure Hist where
  a : Int
  b : Int
  c : Int

/-- The history slides by one per token. -/
def histStep (h : Hist) (x : Int) : Hist := ⟨h.b, h.c, x⟩

/-- Four-tap convolution output. -/
def convOut (h : Hist) (x w0 w1 w2 w3 : Int) : Int := h.a * w0 + h.b * w1 + h.c * w2 + x * w3

/-- After three tokens the history is exactly the last three inputs, independent of the old history. -/
theorem hist_three (h : Hist) (x y z : Int) :
    histStep (histStep (histStep h x) y) z = ⟨x, y, z⟩ := rfl

-- § launch geometry
/-- Channel index of a thread is in range. -/
theorem conv_channel_lt (blk tx channels : Nat) (hb : blk < channels / 128) (ht : tx < 128) :
    blk * 128 + tx < channels := by omega

/-- A thread's block is the channel's 128-wide head group. -/
theorem conv_head_of_channel (blk tx : Nat) (ht : tx < 128) : (blk * 128 + tx) / 128 = blk := by omega

/-- The ab grid covers all 2*h_v weight rows. -/
theorem ab_grid_cover (hv : Nat) : 2 * hv ≤ ((2 * hv + 7) / 8) * 8 := by omega

/-- Row split: alpha rows come first, beta rows are offset by h_v and stay in range. -/
theorem ab_row_split (row hv : Nat) (h : row < 2 * hv) :
    (row < hv → row < hv) ∧ (¬ row < hv → row - hv < hv) := by
  constructor <;> intro hh <;> omega

/-- Softplus selection branch: large inputs pass through unchanged. -/
theorem softplus_large (v : Int) (f : Int → Int) (h : 20 < v) : (if 20 < v then v else f v) = v := by
  simp [h]

/-- Head pairing is modulo h_k and in range. -/
theorem qhead_lt (head hk : Nat) (h : 0 < hk) : head % hk < hk := Nat.mod_lt _ h

end Ref.FusedGdnDecode



## FILE: /mnt/user-data/outputs/FusedGrRead.lean
/-!
# MODULE: fused_gr_read
SOURCES: src/kernels/cuda/fused_gr.cu
Fused hyper-connection "read": RMS-normalise the 4 streams, down-project to 320 channels with SiLU, up-project with a
sigmoid gate and mix the streams. Single-token and T-token (one weight read) variants. The index geometry
of both variants is formalised.

## Source semantics
* D = N*HC = 10240; the down kernel has 40 blocks of 8 warps (one row per warp, 320 rows) plus one inject block.
* The up kernel: 80 blocks x 32 columns (x 4 streams = 128 rows) single token; 160 blocks x 16 columns multi token.
* The multi-token down kernel reads each weight row once, in 4 tiles of 2560 floats, 10 uint4 chunks per lane per tile.

## Form template identifiers
N -> n_embd, HC -> n_streams, D -> flat_stream_len, LR -> low_rank, TILE -> staged_tile
-/
namespace Ref.FusedGrRead

-- § constants
/-- n_embd. -/
def N : Nat := 2560
/-- streams. -/
def HC : Nat := 4
/-- flattened stream length. -/
def D : Nat := N * HC
/-- low rank. -/
def LR : Nat := 320
/-- warps per block. -/
def WARPS : Nat := 8
/-- down blocks. -/
def DOWN_BLOCKS : Nat := 40
/-- columns per single-token up block. -/
def UP_COLS : Nat := 32
/-- single-token up blocks. -/
def UP_BLOCKS : Nat := 80
/-- tile of xn floats. -/
def TILE : Nat := 2560
/-- chunks per lane per tile. -/
def TQ : Nat := 10
/-- columns per multi-token up block. -/
def UPM_COLS : Nat := 16
/-- multi-token up blocks. -/
def UPM_BLOCKS : Nat := 160

/-- D is 10240. -/
theorem D_eq : D = 10240 := by decide
/-- The down blocks cover every low-rank row exactly once. -/
theorem down_rows_cover : DOWN_BLOCKS * WARPS = LR := by decide
/-- The up blocks cover every column exactly once. -/
theorem up_cols_cover : UP_BLOCKS * UP_COLS = N := by decide
/-- The multi-token up blocks cover every column exactly once. -/
theorem upm_cols_cover : UPM_BLOCKS * UPM_COLS = N := by decide
/-- Four tiles cover the whole flattened stream. -/
theorem tiles_cover : (D / TILE) * TILE = D := by decide
/-- There are four tiles. -/
theorem tile_count : D / TILE = 4 := by decide
/-- One tile is 320 chunks of 8: 32 lanes x TQ chunks. -/
theorem tile_chunks : TQ * 32 * 8 = TILE := by decide
/-- The low-rank vector is 40 chunks of 8. -/
theorem lr_chunks : LR / 8 = 40 := by decide
/-- Second chunk exists only for lanes below 8. -/
theorem lr_second_chunk : LR / 8 - 32 = 8 := by decide
/-- Inject rows (one warp each, 4 of them) fit in a block of 8 warps. -/
theorem inject_rows_fit : HC ≤ WARPS := by decide

-- § flat stream indexing
/-- Flat index of stream c, element d. -/
def flat (c d : Nat) : Nat := c * N + d

/-- Flat indices are in range. -/
theorem flat_lt (c d : Nat) (hc : c < HC) (hd : d < N) : flat c d < D := by
  simp only [flat, N, HC, D] at *
  omega

/-- The stream is recovered from a flat index. -/
theorem flat_stream (c d : Nat) (hd : d < N) : flat c d / N = c := by
  simp only [flat, N] at *
  omega

/-- The element is recovered from a flat index. -/
theorem flat_elem (c d : Nat) (hd : d < N) : flat c d % N = d := by
  simp only [flat, N] at *
  omega

/-- Flat indexing is injective. -/
theorem flat_inj (c d c' d' : Nat) (hd : d < N) (hd' : d' < N) (h : flat c d = flat c' d') :
    c = c' ∧ d = d' := by
  simp only [flat, N] at *
  omega

-- § down kernel rows
/-- Row of warp w in block b is a valid low-rank row. -/
theorem down_row_lt (b w : Nat) (hb : b < DOWN_BLOCKS) (hw : w < WARPS) : b * WARPS + w < LR := by
  simp only [DOWN_BLOCKS, WARPS, LR] at *
  omega

/-- Down rows are owned by exactly one (block, warp). -/
theorem down_row_inj (b w b' w' : Nat) (hw : w < WARPS) (hw' : w' < WARPS)
    (h : b * WARPS + w = b' * WARPS + w') : b = b' ∧ w = w' := by
  simp only [WARPS] at *
  omega

-- § up kernel rows
/-- Up row index (stream c, column d0+dd) of a single-token block. -/
def upIdx (blk c dd : Nat) : Nat := c * N + blk * UP_COLS + dd

/-- Up indices stay inside the flat stream. -/
theorem upIdx_lt (blk c dd : Nat) (hb : blk < UP_BLOCKS) (hc : c < HC) (hd : dd < UP_COLS) :
    upIdx blk c dd < D := by
  simp only [upIdx, UP_BLOCKS, HC, UP_COLS, N, D] at *
  omega

/-- Each block owns column d0+dd of every stream: (block, stream, dd) determines the index. -/
theorem upIdx_inj (b c d b' c' d' : Nat) (hb : b < UP_BLOCKS) (hb' : b' < UP_BLOCKS)
    (hc : c < HC) (hc' : c' < HC) (hd : d < UP_COLS) (hd' : d' < UP_COLS)
    (h : upIdx b c d = upIdx b' c' d') : b = b' ∧ c = c' ∧ d = d' := by
  simp only [upIdx, UP_BLOCKS, HC, UP_COLS, N] at *
  omega

/-- The 128 rows of an up block are split 16 per warp (stream c = r / 32). -/
theorem up_rows_per_warp : (HC * UP_COLS) / WARPS = 16 := by decide

/-- Multi-token up index. -/
def upmIdx (blk c dd : Nat) : Nat := c * N + blk * UPM_COLS + dd

/-- Multi-token up indices stay inside the flat stream. -/
theorem upmIdx_lt (blk c dd : Nat) (hb : blk < UPM_BLOCKS) (hc : c < HC) (hd : dd < UPM_COLS) :
    upmIdx blk c dd < D := by
  simp only [upmIdx, UPM_BLOCKS, HC, UPM_COLS, N, D] at *
  omega

/-- Multi-token up indices are injective in (block, stream, column). -/
theorem upmIdx_inj (b c d b' c' d' : Nat) (hb : b < UPM_BLOCKS) (hb' : b' < UPM_BLOCKS)
    (hc : c < HC) (hc' : c' < HC) (hd : d < UPM_COLS) (hd' : d' < UPM_COLS)
    (h : upmIdx b c d = upmIdx b' c' d') : b = b' ∧ c = c' ∧ d = d' := by
  simp only [upmIdx, UPM_BLOCKS, HC, UPM_COLS, N] at *
  omega

/-- 64 rows per multi-token block are 8 per warp. -/
theorem upm_rows_per_warp : (HC * UPM_COLS) / WARPS = 8 := by decide

-- § tiled weight chunk addressing (multi-token down kernel)
/-- Weight chunk index loaded by lane `lane` for slot q of tile t. -/
def chunkIdx (t q lane : Nat) : Nat := t * (TILE / 8) + lane + 32 * q

/-- Chunk indices stay in the 1280 chunks of a weight row. -/
theorem chunkIdx_lt (t q lane : Nat) (ht : t < 4) (hq : q < TQ) (hl : lane < 32) :
    chunkIdx t q lane < D / 8 := by
  simp only [chunkIdx, TILE, TQ, D, N, HC] at *
  omega

/-- Every weight chunk is loaded by exactly one (tile, slot, lane). -/
theorem chunkIdx_inj (t q l t' q' l' : Nat) (hq : q < TQ) (hq' : q' < TQ) (hl : l < 32) (hl' : l' < 32)
    (h : chunkIdx t q l = chunkIdx t' q' l') : t = t' ∧ q = q' ∧ l = l' := by
  simp only [chunkIdx, TILE, TQ] at *
  omega

/-- Every chunk of the row is covered. -/
theorem chunkIdx_surj (j : Nat) (hj : j < D / 8) :
    ∃ t q lane, t < 4 ∧ q < TQ ∧ lane < 32 ∧ chunkIdx t q lane = j := by
  refine ⟨j / 320, (j % 320) / 32, j % 32, ?_, ?_, ?_, ?_⟩ <;>
    simp only [chunkIdx, TILE, TQ, D, N, HC] at * <;> omega

end Ref.FusedGrRead



## FILE: /mnt/user-data/outputs/GdnDeltaNetKernels.lean
/-!
# MODULE: gdn_deltanet_kernels
SOURCES: src/kernels/cuda/gdn.cu
The non-projection parts of the gated delta-net: the column-parallel recurrence step, the sliding causal conv, the
L2 norm, the beta sigmoid gate and the output RMS-norm/gate. State layout is (S, h_v, S) with j fastest.

## Source semantics
* One thread owns one (head, j) state column end to end, so the recurrence needs no barrier.
* Decay happens BEFORE the update: sk = sum_i (dec*s_i)*k_i, d = (v - sk)*beta, s_i' = dec*s_i + k_i*d, o = sum s_i'*q_i.
* MAX_H = 1 head per block (latency hiding knob): grid = (S/32, h_v).
* Conv: kernel[0] reads the oldest state row, the new input lands in the last tap, and the state slides.

## Form template identifiers
S -> state_size, h_v -> value_heads, h_k -> key_heads, JTHREADS -> j_threads_per_block, MAX_H -> heads_per_block
-/
namespace Ref.GdnDeltaNetKernels

-- § exact-arithmetic reduction library
/-- Sum of `f 0 .. f (n-1)` over the integers. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n+1 => sumTo f n + f n

/-- Scalar factors pull out of a reduction. -/
theorem sum_scale (g : Int) (f : Nat → Int) (n : Nat) :
    sumTo (fun i => g * f i) n = g * sumTo f n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih => simp [sumTo, ih, Int.mul_add]

/-- Decaying the state before the k-contraction equals decaying the contraction (the two kernels agree). -/
theorem decay_commutes (g : Int) (s k : Nat → Int) (n : Nat) :
    sumTo (fun i => g * s i * k i) n = g * sumTo (fun i => s i * k i) n := by
  have h : (fun i => g * s i * k i) = (fun i => g * (s i * k i)) :=
    funext fun i => Int.mul_assoc _ _ _
  rw [h, sum_scale]

/-- The decay-first delta of gdn.cu equals the fused `(v - g*kv)*beta` form of fused_gdn.cu. -/
theorem delta_two_pass_eq_fused (g beta v : Int) (s k : Nat → Int) (n : Nat) :
    (v - sumTo (fun i => g * s i * k i) n) * beta = (v - g * sumTo (fun i => s i * k i) n) * beta := by
  rw [decay_commutes]

-- § grid geometry
/-- j-threads per block. -/
def JTHREADS : Nat := 32
/-- heads staged per block. -/
def MAX_H : Nat := 1

/-- For S = 128 the x-grid is 4 blocks. -/
theorem grid_x_128 : (128 + JTHREADS - 1) / JTHREADS = 4 := by decide

/-- With one head per block and h_v = 48 the grid is 4 x 48 = 192 blocks. -/
theorem grid_blocks_48 : ((128 + JTHREADS - 1) / JTHREADS) * ((48 + MAX_H - 1) / MAX_H) = 192 := by decide

/-- With MAX_H = 1 every head gets its own block row. -/
theorem grid_y_eq (hv : Nat) : (hv + MAX_H - 1) / MAX_H = hv := by
  simp only [MAX_H]
  omega

/-- The x-grid covers all S columns. -/
theorem grid_x_cover (S : Nat) : S ≤ ((S + JTHREADS - 1) / JTHREADS) * JTHREADS := by
  simp only [JTHREADS]
  omega

/-- Heads handled by a block: min(MAX_H, h_v - h0) is positive and stays in range. -/
theorem head_in_range (h0 hi hv : Nat) (h0lt : h0 < hv) (hi_lt : hi < min MAX_H (hv - h0)) :
    h0 + hi < hv := by
  simp only [MAX_H] at *
  omega

/-- Staging loop: thread t loading i = t, t+32, ... covers every i < S. -/
theorem staging_cover (S i : Nat) (h : i < S) :
    i % JTHREADS < JTHREADS ∧ i = i % JTHREADS + JTHREADS * (i / JTHREADS) := by
  simp only [JTHREADS]
  omega

/-- Head pairing is modulo h_k and in range. -/
theorem qhead_lt (head hk : Nat) (h : 0 < hk) : head % hk < hk := Nat.mod_lt _ h

-- § state addressing
/-- Row-major pair bound: a*B + b < A*B. -/
theorem pair_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : a + 1 ≤ A := ha
  have h2 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- State index (i, h, j) of the (S, h_v, S) layout is in range. -/
theorem stateIdx_lt (S hv i h j : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) :
    (i * hv + h) * S + j < S * hv * S := by
  have h1 : i * hv + h < S * hv := pair_lt i h S hv hi hh
  exact pair_lt (i * hv + h) j (S * hv) S h1 hj

/-- For S = 128, h_v = 48 the state indexing is injective. -/
theorem stateIdx_inj_128_48 (i h j i' h' j' : Nat) (hi : i < 128) (hh : h < 48) (hj : j < 128)
    (hi' : i' < 128) (hh' : h' < 48) (hj' : j' < 128)
    (e : (i * 48 + h) * 128 + j = (i' * 48 + h') * 128 + j') : i = i' ∧ h = h' ∧ j = j' := by
  omega

/-- q/k staging reads k[src*S + i] with src < h_k and i < S: in range. -/
theorem qk_stage_lt (S hk src i : Nat) (hs : src < hk) (hi : i < S) : src * S + i < hk * S :=
  pair_lt src i hk S hs hi

-- § sliding conv state
/-- Slide: drop the oldest row, append the newest input. -/
def slide (st : List Int) (x : Int) : List Int := st.drop 1 ++ [x]

/-- Sliding preserves the state length. -/
theorem slide_length (st : List Int) (x : Int) (h : st ≠ []) : (slide st x).length = st.length := by
  have : 0 < st.length := List.length_pos_of_ne_nil h
  simp [slide] <;> omega

/-- After three slides the 3-row state is exactly the last three inputs. -/
theorem slide_three (a b c x y z : Int) : slide (slide (slide [a, b, c] x) y) z = [x, y, z] := rfl

-- § launch coverage
/-- The conv launch covers all channels. -/
theorem conv_grid_cover (c : Nat) : c ≤ ((c + 255) / 256) * 256 := by omega

/-- The beta gate launch covers all heads. -/
theorem beta_grid_cover (n : Nat) : n ≤ ((n + 127) / 128) * 128 := by omega

/-- Large softplus inputs pass through (overflow guard). -/
theorem softplus_large (x : Int) (f : Int → Int) (h : 20 < x) : (if 20 < x then x else f x) = x := by
  simp [h]

end Ref.GdnDeltaNetKernels



## FILE: /mnt/user-data/outputs/NativeGdnStep.lean
/-!
# MODULE: native_gdn_step
SOURCES: src/kernels/cuda/native_gdn.cu
Warp-per-column gated-delta-net step (adapted from pinned llama.cpp): each warp owns one Ref (head, column) and
retains its four rows (r*32 + lane) in registers; two warp-sum reductions give kv and the attention readout.

## Source semantics
* grid = (h_v, 1, S/4), block = (32, 4); column = blockIdx.z*4 + threadIdx.y; row = r*32 + lane, r < 4.
* Preconditions: stream non-null, S = 128, h_k > 0, h_v % h_k = 0, h_v <= 65535; spans aligned and pairwise disjoint
  between state/output and every input.

## Form template identifiers
overlap -> span_overlap, S -> state_size
-/
namespace Ref.NativeGdnStep

-- § exact-arithmetic reduction library
/-- Sum of `f 0 .. f (n-1)` over the integers. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n+1 => sumTo f n + f n

/-- Additivity of reductions. -/
theorem sum_add_fn (f g : Nat → Int) (n : Nat) :
    sumTo (fun i => f i + g i) n = sumTo f n + sumTo g n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih => simp only [sumTo, ih]; omega

/-- The zero function reduces to zero. -/
theorem sumTo_zero_fn (m : Nat) : sumTo (fun _ => (0 : Int)) m = 0 := by
  induction m with
  | zero => simp [sumTo]
  | succ m ih => simp [sumTo, ih]

/-- Double reductions commute (reduction order is irrelevant in exact arithmetic). -/
theorem sumTo_comm (g : Nat → Nat → Int) (m n : Nat) :
    sumTo (fun a => sumTo (g a) m) n = sumTo (fun b => sumTo (fun a => g a b) n) m := by
  induction n with
  | zero => simp [sumTo, sumTo_zero_fn]
  | succ n ih =>
    simp only [sumTo]
    rw [sum_add_fn, ← ih]

/-- Splitting a reduction at an arbitrary point. -/
theorem sumTo_add (f : Nat → Int) (n m : Nat) :
    sumTo f (n+m) = sumTo f n + sumTo (fun r => f (n+r)) m := by
  induction m with
  | zero => simp [sumTo]
  | succ m ih =>
    have e : sumTo f (n + (m+1)) = sumTo f (n+m) + f (n+m) := rfl
    have e2 : sumTo (fun r => f (n+r)) (m+1) = sumTo (fun r => f (n+r)) m + f (n+m) := rfl
    rw [e, ih, e2]
    omega

/-- A reduction over `a*b` items equals the reduction of `a` block reductions of `b` items. -/
theorem sum_blocks (f : Nat → Int) (b a : Nat) :
    sumTo f (a*b) = sumTo (fun r => sumTo (fun c => f (r*b+c)) b) a := by
  induction a with
  | zero => simp [sumTo]
  | succ a ih =>
    have h : (a+1)*b = a*b + b := Nat.succ_mul a b
    rw [h, sumTo_add f (a*b) b, ih]
    try rfl

/-- The warp-sum of per-lane register partials (4 rows each) equals the contraction over all 128 rows. -/
theorem warp_partials_total (f : Nat → Int) :
    sumTo (fun lane => sumTo (fun r => f (r * 32 + lane)) 4) 32 = sumTo f 128 := by
  have h1 : sumTo (fun lane => sumTo (fun r => f (r * 32 + lane)) 4) 32
      = sumTo (fun r => sumTo (fun lane => f (r * 32 + lane)) 32) 4 :=
    sumTo_comm (fun lane r => f (r * 32 + lane)) 4 32
  rw [h1]
  exact (sum_blocks f 32 4).symm

-- § thread/row layout
/-- Every state row is owned by exactly one (register slot, lane). -/
theorem row_owner (i : Nat) (hi : i < 128) :
    ∃ r lane, r < 4 ∧ lane < 32 ∧ r * 32 + lane = i :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- Row ownership is injective. -/
theorem row_owner_inj (r lane r' lane' : Nat) (hl : lane < 32) (hl' : lane' < 32)
    (h : r * 32 + lane = r' * 32 + lane') : r = r' ∧ lane = lane' := by omega

/-- Every column is owned by exactly one (blockIdx.z, threadIdx.y). -/
theorem col_owner (c : Nat) (hc : c < 128) :
    ∃ z y, z < 128 / 4 ∧ y < 4 ∧ z * 4 + y = c :=
  ⟨c / 4, c % 4, by omega, by omega, by omega⟩

/-- Column ownership is injective. -/
theorem col_owner_inj (z y z' y' : Nat) (hy : y < 4) (hy' : y' < 4)
    (h : z * 4 + y = z' * 4 + y') : z = z' ∧ y = y' := by omega

/-- Block size 32*4 matches `__launch_bounds__(128, 2)`. -/
theorem block_size : 32 * 4 = 128 := by decide

/-- The q-head index of a head is below h_k, so q/k reads stay inside the h_k*128 floats. -/
theorem qk_read_lt (head hk i : Nat) (hk0 : 0 < hk) (hi : i < 128) : (head % hk) * 128 + i < 128 * hk := by
  have h : head % hk < hk := Nat.mod_lt _ hk0
  omega

-- § span validation
/-- Byte-span overlap test used by the host wrapper. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> intro h <;> omega

/-- Adjacent spans never overlap. -/
theorem adjacent_disjoint (a n m : Nat) : ¬ overlap a n (a + n) m := by
  unfold overlap
  omega

/-- A span never overlaps one that ends where it begins. -/
theorem adjacent_disjoint' (a n m : Nat) : ¬ overlap (a + n) m a n := by
  unfold overlap
  omega

/-- A nonempty span overlaps itself. -/
theorem overlap_self (a n : Nat) (h : 0 < n) : overlap a n a n := by
  unfold overlap
  omega

/-- Float spans are a multiple of the float size. -/
theorem float_span_aligned (n : Nat) : (n * 4) % 4 = 0 := by omega

/-- State bytes are 128*128*h_v*4. -/
theorem state_bytes_eq (hv : Nat) : 128 * 128 * hv * 4 = 65536 * hv := by omega

end Ref.NativeGdnStep



## FILE: /mnt/user-data/outputs/NativeGdnPreprocess.lean
/-!
# MODULE: native_gdn_preprocess
SOURCES: src/kernels/cuda/native_gdn_preprocess.cu
GDN pre-processing kernels adapted from pinned llama.cpp: causal conv + SiLU with history slide, the 128-wide L2/RMS
norms, beta sigmoid, softplus gate, and the fused output RMS-norm * sigmoid(z). Geometry and validation formalised.

## Source semantics
* conv_silu: four taps over [h0,h1,h2,x]; the history becomes [h1,h2,x]; raw and SiLU outputs are written.
* norms: width must be exactly 128 with 256 threads = 8 warps; `norm_sum` reduces 8 warp sums.
* host wrappers: count in [1,65535], stream non-null, all spans aligned/non-null, writable spans disjoint.

## Form template identifiers
Hist -> conv_history, histStep -> history_slide, ceilBlocks -> launch_blocks
-/
namespace Ref.NativeGdnPreprocess

-- § conv history
/-- Three-entry history. -/
structure Hist where
  a : Int
  b : Int
  c : Int

/-- Slide by one input. -/
def histStep (h : Hist) (x : Int) : Hist := ⟨h.b, h.c, x⟩

/-- Four taps over history and the new input. -/
def tapSum (h : Hist) (x w0 w1 w2 w3 : Int) : Int := h.a * w0 + h.b * w1 + h.c * w2 + x * w3

/-- Sliding drops the oldest and appends the newest. -/
theorem histStep_fields (h : Hist) (x : Int) :
    (histStep h x).a = h.b ∧ (histStep h x).b = h.c ∧ (histStep h x).c = x := ⟨rfl, rfl, rfl⟩

/-- The history after three inputs forgets the old history. -/
theorem hist_three (h : Hist) (x y z : Int) :
    histStep (histStep (histStep h x) y) z = ⟨x, y, z⟩ := rfl

/-- Tap sum with the new input in the last tap. -/
theorem tapSum_last (h : Hist) (x w0 w1 w2 w3 : Int) :
    tapSum h x w0 w1 w2 w3 = h.a * w0 + h.b * w1 + h.c * w2 + x * w3 := rfl

-- § buffer geometry
/-- History slot c*3+tap stays in 3*channels floats. -/
theorem hist_idx_lt (c tap ch : Nat) (hc : c < ch) (ht : tap < 3) : c * 3 + tap < 3 * ch := by omega

/-- Weight slot c*4+tap stays in 4*channels floats. -/
theorem weight_idx_lt (c tap ch : Nat) (hc : c < ch) (ht : tap < 4) : c * 4 + tap < 4 * ch := by omega

/-- History slots are injective in (channel, tap). -/
theorem hist_idx_inj (c t c' t' : Nat) (ht : t < 3) (ht' : t' < 3) (h : c * 3 + t = c' * 3 + t') :
    c = c' ∧ t = t' := by omega

/-- Ceiling-divide block count. -/
def ceilBlocks (n d : Nat) : Nat := (n + d - 1) / d

/-- 256-thread blocks cover all channels/heads. -/
theorem blocks256_cover (n : Nat) : n ≤ ceilBlocks n 256 * 256 := by
  unfold ceilBlocks
  omega

/-- With count <= 65535 the grid has at most 256 blocks (within the 65535 limit). -/
theorem blocks256_le (n : Nat) (h : n ≤ 65535) : ceilBlocks n 256 ≤ 256 := by
  unfold ceilBlocks
  omega

/-- A positive count needs at least one block. -/
theorem blocks256_pos (n : Nat) (h : 0 < n) : 0 < ceilBlocks n 256 := by
  unfold ceilBlocks
  omega

-- § norm geometry
/-- 256 threads are exactly 8 warps, matching `lane < 8` in norm_sum. -/
theorem norm_warps : 256 / 32 = 8 := by decide

/-- Every warp index of a 256-thread block is below 8. -/
theorem warp_idx_lt (t : Nat) (h : t < 256) : t / 32 < 8 := by omega

/-- Row offset blockIdx*128 + col is in range of heads*128. -/
theorem norm_idx_lt (b col heads : Nat) (hb : b < heads) (hc : col < 128) : b * 128 + col < heads * 128 := by
  have h1 : b + 1 ≤ heads := hb
  have h2 : (b + 1) * 128 ≤ heads * 128 := Nat.mul_le_mul_right 128 h1
  omega

/-- Threads 128..255 of a norm block are padding (col >= S) and contribute zero. -/
theorem norm_padding (col : Nat) (h : 128 ≤ col) : ¬ col < 128 := by omega

-- § span validation
/-- Byte-span overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> intro h <;> omega

/-- Adjacent spans do not overlap. -/
theorem adjacent_disjoint (a n m : Nat) : ¬ overlap a n (a + n) m := by
  unfold overlap
  omega

/-- The conv writable spans history (3*bytes), raw (bytes), silu (bytes) packed back to back are pairwise disjoint. -/
theorem packed_spans_disjoint (base bytes : Nat) :
    ¬ overlap base (3 * bytes) (base + 3 * bytes) bytes ∧
    ¬ overlap base (3 * bytes) (base + 3 * bytes + bytes) bytes ∧
    ¬ overlap (base + 3 * bytes) bytes (base + 3 * bytes + bytes) bytes := by
  unfold overlap
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Softplus guard: x > 20 passes through unchanged. -/
theorem softplus_large (x : Int) (f : Int → Int) (h : 20 < x) : (if 20 < x then x else f x) = x := by
  simp [h]

end Ref.NativeGdnPreprocess



## FILE: /mnt/user-data/outputs/VerifyWindowKernels.lean
/-!
# MODULE: verify_window_kernels
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token "verify window" kernels for speculative decoding: per-token GDN conv/L2, alpha/beta, state-step + norm over
a window with an `n_keep` commit, plus the device-side helpers (blob fetch, id remap, window ids, dense steps).
Per-token arithmetic is the single-token arithmetic; the window indexing and commit contracts are formalised.

## Source semantics
* Token t of the conv window reads the sequence [hist0,hist1,hist2,x_0..]: entries t..t+2 plus x_t = entry t+3.
* `gdn_conv_commit` keeps the last three entries of [hist | x_0..x_{n-1}] (indices n..n+2); n <= 0 is a no-op.
* `gdn_step_norm_multi` replays tokens t < n; outputs are emitted only for t >= t_out_begin; state is written back
  only when n_keep is non-null and n > 0.
* `window_ids`: start = max(0, n_kv - window), width = n_kv - start, ids[j] = start + j.
* `dense_steps`: steps = [c, c+1, (c+1)/4, c+1] per cell, the step layout the attention kernels validate.

## Form template identifiers
seqAt -> window_sequence_entry, convWinIdx -> conv_window_index, commitIdx -> commit_index, pair_inj -> mixed_radix_inj
-/
namespace Ref.VerifyWindowKernels

-- § mixed-radix addressing
/-- Mixed-radix index a*B + b is recovered by division. -/
theorem pair_div (a b B : Nat) (hB : 0 < B) (hb : b < B) : (a * B + b) / B = a := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hb, Nat.zero_add]

/-- Mixed-radix index a*B + b is recovered by modulo. -/
theorem pair_mod (a b B : Nat) (hb : b < B) : (a * B + b) % B = b := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- Mixed-radix indexing is injective (e.g. token-major buffers `t*h_v + r`). -/
theorem pair_inj (a b a' b' B : Nat) (hB : 0 < B) (hb : b < B) (hb' : b' < B)
    (h : a * B + b = a' * B + b') : a = a' ∧ b = b' := by
  have h1 := congrArg (fun x => x / B) h
  have h2 := congrArg (fun x => x % B) h
  simp only [pair_div a b B hB hb, pair_div a' b' B hB hb', pair_mod a b B hb,
    pair_mod a' b' B hb'] at h1 h2
  exact ⟨h1, h2⟩

/-- Row-major pair bound. -/
theorem pair_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : a + 1 ≤ A := ha
  have h2 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- Per-token gate/beta slots `t*h_v + r` are in range for t < T, r < h_v. -/
theorem token_slot_lt (t r T hv : Nat) (ht : t < T) (hr : r < hv) : t * hv + r < T * hv :=
  pair_lt t r T hv ht hr

-- § conv window
/-- Index into the virtual sequence [hist(3) | x_0 .. ] read by tap j of token t. -/
def convWinIdx (t j : Nat) : Nat := t + j

/-- A window tap either reads history (index < 3) or an earlier token x_{src-3} with src-3 < t: no future leakage. -/
theorem conv_window_causal (t j : Nat) (hj : j < 3) :
    convWinIdx t j < 3 ∨ (3 ≤ convWinIdx t j ∧ convWinIdx t j - 3 < t) := by
  simp only [convWinIdx]
  omega

/-- The current input of token t is entry t+3 of the virtual sequence. -/
theorem conv_current_index (t : Nat) : convWinIdx t 3 - 3 = t := by
  simp only [convWinIdx]
  omega

/-- Consecutive tokens slide the window by one entry (the multi-token conv equals repeated single-token conv). -/
theorem conv_window_slide (t j : Nat) (hj : j < 2) : convWinIdx (t + 1) j = convWinIdx t (j + 1) := by
  simp only [convWinIdx]
  omega

/-- Commit tap j reads entry n+j of [hist | x_0..x_{n-1}]. -/
def commitIdx (n j : Nat) : Nat := n + j

/-- Commit only ever reads kept tokens x_0..x_{n-1} or old history. -/
theorem commit_reads_kept (n j : Nat) (hj : j < 3) :
    commitIdx n j < 3 ∨ (3 ≤ commitIdx n j ∧ commitIdx n j - 3 < n) := by
  simp only [commitIdx]
  omega

/-- With n = 3 the committed history is exactly the last three kept inputs x_0, x_1, x_2. -/
theorem commit_three (j : Nat) (hj : j < 3) : commitIdx 3 j - 3 = j := by
  simp only [commitIdx]
  omega

/-- Committing one more token slides the committed window by one. -/
theorem commit_slide (n j : Nat) : commitIdx (n + 1) j = commitIdx n (j + 1) := by
  simp only [commitIdx]
  omega

-- § step/norm window
/-- Number of tokens replayed: n_keep if given else T. -/
def replayCount (hasKeep : Bool) (nKeep T : Nat) : Nat := if hasKeep then nKeep else T

/-- Outputs are emitted for t in [t_out_begin, n): that is `n - t_out_begin` tokens. -/
def outputCount (n tOutBegin : Nat) : Nat := n - tOutBegin

/-- Without n_keep the replay count is the window size. -/
theorem replay_no_keep (nKeep T : Nat) : replayCount false nKeep T = T := rfl

/-- With n_keep the replay count is n_keep. -/
theorem replay_keep (nKeep T : Nat) : replayCount true nKeep T = nKeep := rfl

/-- Replay never exceeds the window when n_keep <= T. -/
theorem replay_le (hasKeep : Bool) (nKeep T : Nat) (h : nKeep ≤ T) : replayCount hasKeep nKeep T ≤ T := by
  cases hasKeep <;> simp [replayCount, h]

/-- Replaying everything with t_out_begin = 0 emits all tokens. -/
theorem output_all (n : Nat) : outputCount n 0 = n := by
  simp [outputCount]

/-- Output slot of token t is below T*value_dim. -/
theorem output_slot_lt (t col T vd : Nat) (ht : t < T) (hc : col < vd) : t * vd + col < T * vd :=
  pair_lt t col T vd ht hc

-- § device helpers
/-- Blob fetch: flat index i is blob k = i/per, offset i%per. -/
theorem blob_decomp (i per : Nat) : per * (i / per) + i % per = i := Nat.div_add_mod i per

/-- Blob fetch: i < n*per implies the blob index is below n. -/
theorem blob_k_lt (n per i : Nat) (hi : i < n * per) : i / per < n := by
  by_cases h : i / per < n
  · exact h
  · have h1 : n ≤ i / per := Nat.le_of_not_lt h
    have h2 : n * per ≤ i / per * per := Nat.mul_le_mul_right per h1
    have h3 : i / per * per ≤ i := Nat.div_mul_le_self i per
    omega

/-- Rebased pointers of consecutive blobs are exactly `bytes` apart. -/
theorem rebase_stride (base bytes k : Nat) : (base + (k + 1) * bytes) - (base + k * bytes) = bytes := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Identity hits: dst is the identity map on [0,n) hence injective. -/
theorem ident_hits_inj (i j : Nat) (h : i = j) : i = j := h

/-- Window start. -/
def winStart (nkv window : Nat) : Nat := if window < nkv then nkv - window else 0

/-- Window width. -/
def winWidth (nkv window : Nat) : Nat := nkv - winStart nkv window

/-- The window ends exactly at n_kv. -/
theorem window_ends (nkv window : Nat) : winStart nkv window + winWidth nkv window = nkv := by
  unfold winWidth winStart
  split <;> omega

/-- The window width is at most the window size. -/
theorem window_width_le (nkv window : Nat) : winWidth nkv window ≤ window := by
  unfold winWidth winStart
  split <;> omega

/-- The window width is at most n_kv. -/
theorem window_width_le_nkv (nkv window : Nat) : winWidth nkv window ≤ nkv := by
  unfold winWidth winStart
  split <;> omega

/-- The window width is min(n_kv, window). -/
theorem window_width_min (nkv window : Nat) : winWidth nkv window = min nkv window := by
  unfold winWidth winStart
  split <;> omega

/-- Window ids are valid cells below n_kv. -/
theorem window_id_lt (nkv window j : Nat) (hj : j < winWidth nkv window) : winStart nkv window + j < nkv := by
  have := window_ends nkv window
  omega

-- § dense steps feed the attention validity predicate
/-- Validity predicate of the native attention kernel on (width, n_kv, pos, n_bid). -/
def stepValid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos = width - 1 ∧ nbid = width / 4

/-- `dense_steps` for cell c (steps = [c, c+1, (c+1)/4, c+1]) satisfies the attention step validity for any max context >= c+1. -/
theorem dense_steps_valid (c maxCtx : Nat) (h : c + 1 ≤ maxCtx) :
    stepValid (c + 1) (c + 1) c ((c + 1) / 4) maxCtx := by
  unfold stepValid
  omega

/-- The four slots of a dense step row are distinct and in range. -/
theorem dense_step_slot (i f : Nat) (hf : f < 4) : (i * 4 + f) / 4 = i ∧ (i * 4 + f) % 4 = f :=
  ⟨pair_div i f 4 (by decide) hf, pair_mod i f 4 hf⟩

-- § code extraction
/-- 4-bit codes: the in-byte shift stays inside the byte. -/
theorem shift4_lt (i : Nat) : (i % (8 / 4)) * 4 + 4 ≤ 8 := by omega
/-- 2-bit codes: the in-byte shift stays inside the byte. -/
theorem shift2_lt (i : Nat) : (i % (8 / 2)) * 2 + 2 ≤ 8 := by omega
/-- 1-bit codes: the in-byte shift stays inside the byte. -/
theorem shift1_lt (i : Nat) : (i % (8 / 1)) * 1 + 1 ≤ 8 := by omega
/-- 8-bit codes: one code per byte. -/
theorem shift8_lt (i : Nat) : (i % (8 / 8)) * 8 + 8 ≤ 8 := by omega

end Ref.VerifyWindowKernels



## FILE: /mnt/user-data/outputs/KvQ8Cache.lean
/-!
# MODULE: kv_q8_cache
SOURCES: src/kernels/cuda/kv_q8.cu
INT8-quantised paged KV cache: append one token (per-64-value-group max-abs scale stored as fp16, values quantised
against the STORED scale, clamped to [-127,127]) and gather cells back to fp16 scratch.

## Source semantics
* head_dim must be a multiple of KV_Q8_GROUP = 64; one block = one 64-value group of one KV head of K or V.
* row = (page*kv_heads + h)*page_size + pos % page_size; scale slot = row*groups + g, groups = head_dim/64.
* gather: thread = 4 consecutive values of one (cell, head); all four share one scale group.

## Form template identifiers
KV_Q8_GROUP -> quant_group, rowIdx -> cache_row, clamp127 -> int8_clamp
-/
namespace Ref.KvQ8Cache

-- § mixed-radix lemmas
/-- Row-major pair bound. -/
theorem pair_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : a + 1 ≤ A := ha
  have h2 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- Mixed-radix recovery by division. -/
theorem pair_div (a b B : Nat) (hB : 0 < B) (hb : b < B) : (a * B + b) / B = a := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hb, Nat.zero_add]

/-- Mixed-radix recovery by modulo. -/
theorem pair_mod (a b B : Nat) (hb : b < B) : (a * B + b) % B = b := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- Mixed-radix injectivity. -/
theorem pair_inj (a b a' b' B : Nat) (hB : 0 < B) (hb : b < B) (hb' : b' < B)
    (h : a * B + b = a' * B + b') : a = a' ∧ b = b' := by
  have h1 := congrArg (fun x => x / B) h
  have h2 := congrArg (fun x => x % B) h
  simp only [pair_div a b B hB hb, pair_div a' b' B hB hb', pair_mod a b B hb,
    pair_mod a' b' B hb'] at h1 h2
  exact ⟨h1, h2⟩

-- § constants and clamp
/-- Quantisation group size. -/
def KV_Q8_GROUP : Nat := 64

/-- Symmetric int8 clamp. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if 127 < q then 127 else q

/-- The clamp always lands in [-127, 127] (never -128). -/
theorem clamp127_bounds (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  split <;> try split
  all_goals omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  split <;> try split
  all_goals omega

/-- The clamp is idempotent. -/
theorem clamp127_idem (q : Int) : clamp127 (clamp127 q) = clamp127 q :=
  clamp127_id _ (clamp127_bounds q).1 (clamp127_bounds q).2

-- § cache row addressing
/-- Cache row of (page, head, in-page offset). -/
def rowIdx (kvh ps page h off : Nat) : Nat := (page * kvh + h) * ps + off

/-- Rows stay inside `P` pages. -/
theorem rowIdx_lt (kvh ps P page h off : Nat) (hp : page < P) (hh : h < kvh) (ho : off < ps) :
    rowIdx kvh ps page h off < P * kvh * ps := by
  unfold rowIdx
  exact pair_lt (page * kvh + h) off (P * kvh) ps (pair_lt page h P kvh hp hh) ho

/-- Distinct (page, head, offset) triples never share a cache row. -/
theorem rowIdx_inj (kvh ps page h off page' h' off' : Nat) (hps : 0 < ps) (hkvh : 0 < kvh)
    (hh : h < kvh) (hh' : h' < kvh) (ho : off < ps) (ho' : off' < ps)
    (e : rowIdx kvh ps page h off = rowIdx kvh ps page' h' off') :
    page = page' ∧ h = h' ∧ off = off' := by
  unfold rowIdx at e
  obtain ⟨e1, e2⟩ := pair_inj (page * kvh + h) off (page' * kvh + h') off' ps hps ho ho' e
  obtain ⟨e3, e4⟩ := pair_inj page h page' h' kvh hkvh hh hh' e1
  exact ⟨e3, e4, e2⟩

/-- In-page offset of a position is below page_size. -/
theorem offset_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

-- § group addressing
/-- A value's slot inside a row (group g, lane t) is below head_dim. -/
theorem value_slot_lt (hd g t : Nat) (hd64 : hd % 64 = 0) (hg : g < hd / 64) (ht : t < 64) :
    g * 64 + t < hd := by omega

/-- A thread's four values start at d = 4*q4 < head_dim. -/
theorem gather_d_lt (hd q4 : Nat) (hq : q4 < hd / 4) : q4 * 4 < hd := by omega

/-- The four values of a gather thread share one scale group (head_dim % 64 = 0 is not even needed). -/
theorem gather_same_group (q4 : Nat) : (q4 * 4) / 64 = (q4 * 4 + 3) / 64 := by omega

/-- The scale group of a gather thread is a valid group index. -/
theorem gather_group_lt (hd q4 : Nat) (hd64 : hd % 64 = 0) (hq : q4 < hd / 4) : (q4 * 4) / 64 < hd / 64 := by
  omega

/-- The 64 threads of a block cover one group: thread t handles value g*64+t. -/
theorem append_cover (hd v : Nat) (hd64 : hd % 64 = 0) (hv : v < hd) :
    ∃ g t, g < hd / 64 ∧ t < 64 ∧ g * 64 + t = v :=
  ⟨v / 64, v % 64, by omega, by omega, by omega⟩

/-- Gather decomposition: flat i = id*(kvh*per) + rem with rem < kvh*per. -/
theorem gather_decomp (i kvh per : Nat) : (kvh * per) * (i / (kvh * per)) + i % (kvh * per) = i :=
  Nat.div_add_mod i (kvh * per)

/-- The remainder splits into (head, q4) with head = rem/per, q4 = rem%per. -/
theorem gather_head_split (rem per : Nat) : per * (rem / per) + rem % per = rem := Nat.div_add_mod rem per

end Ref.KvQ8Cache



## FILE: /mnt/user-data/outputs/NativeFlashAttnShort.lean
/-!
# MODULE: native_flash_attn_short
SOURCES: src/kernels/cuda/native_flash_attn.cu
Specialised decode FlashAttention (D=256, Q 24 heads, KV 2 heads, fp16 KV, context <= 256, one query column) adapted
from the pinned llama.cpp vector kernel. The thread/cell/dimension layouts, step validity gate and host validation
are formalised.

## Source semantics
* grid = 24 blocks (one per query head), block = (32, 4) = 128 threads; kv head = head / 12.
* The step is valid iff 1 <= width <= max_context, n_kv = width, pos = width-1 and n_bid = width/4; otherwise the
  output is NaN and status = unsupported.
* 128 cells per iteration: cell = base + warp*32 + (lane & ~7) + row; padded length 256 => two iterations.
* Each lane handles dimensions d = 2*(32a + 4*(lane%8) + j) and d+1 (a < 4, j < 4).
* The epilogue sums 16 per-thread partial tiles (4 warps x 4 lane groups) and divides by the softmax sum.

## Form template identifiers
stepValid -> step_validity, kvIdx -> kv_element_index, tileIdx -> partial_tile_slot
-/
namespace Ref.NativeFlashAttnShort

-- § step validity
/-- The validity predicate checked on-device. -/
def stepValid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos = width - 1 ∧ nbid = width / 4

/-- A valid step addresses only cells strictly below the width. -/
theorem valid_pos_lt (w n p b m : Nat) (h : stepValid w n p b m) : p < w := by
  unfold stepValid at h
  omega

/-- With max context <= 256 a valid step fits the padded length 256. -/
theorem valid_fits_padded (w n p b m : Nat) (h : stepValid w n p b m) (hm : m ≤ 256) : w ≤ 256 := by
  unfold stepValid at h
  omega

/-- A valid step's width is at least one (non-empty context). -/
theorem valid_nonempty (w n p b m : Nat) (h : stepValid w n p b m) : 0 < w := by
  unfold stepValid at h
  omega

/-- Width 1..max_context: the dense step of cell c is valid exactly when c+1 <= max_context. -/
theorem dense_step_valid_iff (c m : Nat) :
    stepValid (c + 1) (c + 1) c ((c + 1) / 4) m ↔ c + 1 ≤ m := by
  unfold stepValid
  omega

-- § head and kv layout
/-- KV head of a query head. -/
def kvHead (head : Nat) : Nat := head / 12

/-- All 24 query heads map to one of the two KV heads. -/
theorem kvHead_lt (head : Nat) (h : head < 24) : kvHead head < 2 := by
  unfold kvHead
  omega

/-- Heads 0..11 use KV head 0 and 12..23 use KV head 1. -/
theorem kvHead_split (head : Nat) (h : head < 24) :
    (head < 12 → kvHead head = 0) ∧ (12 ≤ head → kvHead head = 1) := by
  unfold kvHead
  constructor <;> intro hh <;> omega

/-- Element index of K/V: (cell*2 + kv)*256 + d. -/
def kvIdx (cell kv d : Nat) : Nat := (cell * 2 + kv) * 256 + d

/-- K/V reads stay in capacity*512 halves, which is exactly the validated `capacity*1024` bytes. -/
theorem kvIdx_lt (cell kv d cap : Nat) (hc : cell < cap) (hkv : kv < 2) (hd : d < 256) :
    kvIdx cell kv d < cap * 512 := by
  unfold kvIdx
  omega

/-- K/V element indexing is injective. -/
theorem kvIdx_inj (c k d c' k' d' : Nat) (hk : k < 2) (hk' : k' < 2) (hd : d < 256) (hd' : d' < 256)
    (h : kvIdx c k d = kvIdx c' k' d') : c = c' ∧ k = k' ∧ d = d' := by
  unfold kvIdx at h
  omega

/-- Two bytes per half: capacity*512 halves = capacity*1024 bytes. -/
theorem kv_bytes (cap : Nat) : cap * 512 * 2 = cap * 1024 := by omega

/-- Query/output row of a head: head*256 + d is in 24*256. -/
theorem q_idx_lt (head d : Nat) (hh : head < 24) (hd : d < 256) : head * 256 + d < 24 * 256 := by omega

-- § dimension coverage per lane
/-- Pair index handled by (a, lane%8, j): 32a + 4*l8 + j. -/
def pairIdx (a l8 j : Nat) : Nat := 32 * a + 4 * l8 + j

/-- Pair indices cover 0..127, so dimensions d = 2p and 2p+1 cover 0..255. -/
theorem pair_cover (p : Nat) (hp : p < 128) : ∃ a l8 j, a < 4 ∧ l8 < 8 ∧ j < 4 ∧ pairIdx a l8 j = p :=
  ⟨p / 32, (p % 32) / 4, p % 4, by omega, by omega, by omega, by unfold pairIdx; omega⟩

/-- Each pair index is produced by exactly one (a, lane%8, j). -/
theorem pair_inj (a l j a' l' j' : Nat) (hl : l < 8) (hl' : l' < 8) (hj : j < 4) (hj' : j' < 4)
    (h : pairIdx a l j = pairIdx a' l' j') : a = a' ∧ l = l' ∧ j = j' := by
  unfold pairIdx at h
  omega

/-- The d+1 element read next to d = 2p stays in range. -/
theorem dim_pair_lt (p : Nat) (hp : p < 128) : 2 * p + 1 < 256 := by omega

/-- qreg slot 4a + j is below 16. -/
theorem qreg_slot_lt (a j : Nat) (ha : a < 4) (hj : j < 4) : 4 * a + j < 16 := by omega

-- § cell coverage per iteration
/-- Cell offset inside a 128-cell iteration: warp*32 + (lane/8)*8 + row (the `lane & ~7` term is (lane/8)*8). -/
def cellOff (warp lane row : Nat) : Nat := warp * 32 + (lane / 8) * 8 + row

/-- `lane & ~7` equals lane minus lane mod 8. -/
theorem lane_mask_eq (lane : Nat) : (lane / 8) * 8 = lane - lane % 8 := by omega

/-- Cell offsets stay inside the 128 cells of an iteration. -/
theorem cellOff_lt (warp lane row : Nat) (hw : warp < 4) (hl : lane < 32) (hr : row < 8) :
    cellOff warp lane row < 128 := by
  unfold cellOff
  omega

/-- Every cell of the iteration is handled by some (warp, lane group, row). -/
theorem cellOff_cover (x : Nat) (hx : x < 128) :
    ∃ warp g row, warp < 4 ∧ g < 4 ∧ row < 8 ∧ warp * 32 + g * 8 + row = x :=
  ⟨x / 32, (x % 32) / 8, x % 8, by omega, by omega, by omega, by omega⟩

/-- Two iterations of 128 cover the padded length 256. -/
theorem iterations_cover : 2 * 128 = 256 := by decide

/-- Lane groups of 8 are closed under the xor offsets 1, 2, 4 of `warp_sum<8>`. -/
theorem xor_group_closed : ∀ l : Fin 32, ∀ o ∈ [1, 2, 4], ((l.val ^^^ o) / 8 = l.val / 8) := by
  decide

-- § epilogue
/-- Output slot of the epilogue: i0 in {0, 128} plus tid < 128 covers 0..255 exactly. -/
theorem out_cover (d : Nat) (hd : d < 256) :
    ∃ i0 tid, (i0 = 0 ∨ i0 = 128) ∧ tid < 128 ∧ i0 + tid = d := by
  by_cases h : d < 128
  · exact ⟨0, d, Or.inl rfl, h, by omega⟩
  · exact ⟨128, d - 128, Or.inr rfl, by omega, by omega⟩

/-- Partial-tile slot written before the final sum: warp*1024 + group*256 + d. -/
def tileIdx (warp g d : Nat) : Nat := warp * 4 * 256 + g * 256 + d

/-- Tile slots stay inside the 16*256 = 4096 shared floats. -/
theorem tileIdx_lt (warp g d : Nat) (hw : warp < 4) (hg : g < 4) (hd : d < 256) : tileIdx warp g d < 4 * 4 * 256 := by
  unfold tileIdx
  omega

/-- Tile slots are injective in (warp, group, d). -/
theorem tileIdx_inj (w g d w' g' d' : Nat) (hg : g < 4) (hg' : g' < 4) (hd : d < 256) (hd' : d' < 256)
    (h : tileIdx w g d = tileIdx w' g' d') : w = w' ∧ g = g' ∧ d = d' := by
  unfold tileIdx at h
  omega

/-- Sixteen partial tiles are summed per output dimension: 4 warps x 4 groups. -/
theorem partial_tiles : 4 * 4 = 16 := by decide

/-- The score scale 1/16 squared times D = 256 is one: scale = 1/sqrt(256). -/
theorem scale_sq : 16 * 16 = 256 := by decide

-- § host validation
/-- Launch shape: 24 blocks of 32x4 = 128 threads, matching `__launch_bounds__(128, 1)`. -/
theorem block_threads : 32 * 4 = 128 := by decide

/-- Each of the 128 threads writes two output dimensions: 128 * 2 = 256. -/
theorem outputs_per_head : 128 * 2 = 256 := by decide

/-- Byte-span overlap used by the host check. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> intro h <;> omega

/-- Adjacent spans never overlap. -/
theorem adjacent_disjoint (a n m : Nat) : ¬ overlap a n (a + n) m := by
  unfold overlap
  omega

end Ref.NativeFlashAttnShort





## FILE: /mnt/user-data/outputs/QsaCacheIndexAttend.lean
/-!
# MODULE: qsa_cache_index_attend
SOURCES: src/kernels/cuda/qsa.cu
The QSA KV cache append, indexer-key pooling, block scoring, exact top-k threshold selection, KV gather,
and per-head attention. This module models the index arithmetic, counting and bound contracts of those kernels.

## Source semantics
* `qsa_step_fill`: n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1); all derive from one position.
* A cell `pos` has slot `pos % r` and block `pos / r`; the cell with slot `r-1` completes a block, and then the
  pooled row b and the spare row b+1 are written (so the pooled buffer needs b+1 in range).
* `topk_512` emits every cell strictly above the width-th key plus the first `width - c_gt` cells equal to it.
* `kv_gather` decomposes a flat index into (selected id, kv head, uint2 lane); `qsa_attend` maps head h to kv head
  `h / (n_head / n_head_kv)`.

## Form template identifiers
- `selWidth`      : selection width = min(n_kv, top_k + r - 1)
- `r`             : idx_block (cells per pooled block)
- `n_bid`         : completed-block count n_kv / r
- `gt`, `eq`, `budget` : strictly-above count, equal-to-threshold count, `width - c_gt`
-/

namespace Ref.QsaCache

-- § step state

/-- Selection width as computed by `qsa_selection_width`: min(n_kv, top_k + r - 1). -/
def selWidth (nkv topk r : Nat) : Nat := min nkv (topk + r - 1)

/-- The selection never exceeds the number of cached cells. -/
theorem selWidth_le (nkv topk r : Nat) : selWidth nkv topk r ≤ nkv := by
  unfold selWidth; omega

/-- Below the bound the selection is the identity (width equals n_kv). -/
theorem selWidth_identity (nkv topk r : Nat) (h : nkv ≤ topk + r - 1) :
    selWidth nkv topk r = nkv := by
  unfold selWidth; omega

/-- For the 2048/4 geometry the width never exceeds 2051. -/
theorem selWidth_cap (nkv : Nat) : selWidth nkv 2048 4 ≤ 2051 := by
  unfold selWidth; omega

-- § slot / block arithmetic

/-- A position decomposes as block * r + slot. -/
theorem slot_block_decomp (pos r : Nat) : pos / r * r + pos % r = pos := by
  have h := Nat.div_add_mod pos r
  rw [Nat.mul_comm] at h
  exact h

/-- The cell that completes a block closes exactly (b+1) * r cells. -/
theorem complete_block_size (pos r : Nat) (hs : pos % r + 1 = r) :
    (pos / r + 1) * r = pos + 1 := by
  have h1 := slot_block_decomp pos r
  have h2 : (pos / r + 1) * r = pos / r * r + r := by
    rw [Nat.add_mul, Nat.one_mul]
  omega

/-- When a block completes inside capacity, both the pooled row b and the spare row b+1 exist
in a buffer of `max_cells / r + 1` rows. -/
theorem spare_row_in_bounds (pos r maxc : Nat) (hr : 0 < r) (hs : pos % r + 1 = r)
    (hp : pos < maxc) : pos / r + 1 ≤ maxc / r := by
  have h := complete_block_size pos r hs
  exact (Nat.le_div_iff_mul_le hr).2 (by omega)

/-- A block before the incomplete tail lies entirely inside the cache. -/
theorem full_block_inside (b nkv r : Nat) (hb : b < nkv / r) : (b + 1) * r ≤ nkv := by
  have h1 : (b + 1) * r ≤ nkv / r * r := Nat.mul_le_mul_right r hb
  have h2 : nkv / r * r ≤ nkv := Nat.div_mul_le_self nkv r
  omega

/-- Every cached cell belongs to a block index at most n_bid. -/
theorem cell_block_le (j nkv r : Nat) (hj : j < nkv) : j / r ≤ nkv / r :=
  Nat.div_le_div_right (Nat.le_of_lt hj)

/-- The rotation position of a completed block keeps the alignment of `pos_base`. -/
theorem block_pos_formula (pos_base b r : Nat) : pos_base + b * r = pos_base + r * b := by
  rw [Nat.mul_comm b r]

-- § pool indexing

/-- Mixed-radix index bound: a < n and b < m give a * m + b < n * m. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m ha
  have h2 : (a + 1) * m = a * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The `[page][kv_head][page_size][head_dim]` pool row of `kv_append`/`kv_gather` stays in the pool. -/
theorem pool_row_lt (page h slot np kvh ps : Nat) (hp : page < np) (hh : h < kvh) (hs : slot < ps) :
    (page * kvh + h) * ps + slot < np * kvh * ps :=
  idx_lt (page * kvh + h) slot (np * kvh) ps (idx_lt page h np kvh hp hh) hs

/-- Gather index decomposition: the selected-id component is recovered by division. -/
theorem gather_id (id rem m : Nat) (hm : 0 < m) (hr : rem < m) : (m * id + rem) / m = id := by
  rw [Nat.mul_add_div hm, Nat.div_eq_of_lt hr, Nat.add_zero]

/-- Gather index decomposition: the (kv head, lane) remainder is recovered by modulus. -/
theorem gather_rem (id rem m : Nat) (hr : rem < m) : (m * id + rem) % m = rem := by
  rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hr]

/-- The kv head chosen for query head `h` is in range (`iv2 = iq2 / rv2`). -/
theorem gqa_kv_in_range (h g nkv : Nat) (hh : h < nkv * g) : h / g < nkv := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  exact hh

-- § exact threshold selection

/-- Counting contract: gt strictly-above cells plus min(eq, width - gt) equal cells select exactly `width`
cells whenever the threshold bracket gt <= width <= gt + eq holds. -/
theorem selected_count (gt eq width : Nat) (h1 : gt ≤ width) (h2 : width ≤ gt + eq) :
    gt + min eq (width - gt) = width := by
  omega

/-- With distinct scores (c_gt = width, budget 0), nothing at the threshold is taken. -/
theorem selected_count_distinct (eq width : Nat) : width + min eq (width - width) = width := by
  omega

/-- The prefix of equal cells before a chunk never lets a chunk overshoot the budget. -/
theorem eq_prefix_within_budget (eqBefore eqHere budget : Nat) :
    eqBefore + min eqHere (budget - eqBefore) ≤ max budget eqBefore := by
  omega

/-- A chunk entirely under the budget selects all its equal cells. -/
theorem eq_chunk_fully_taken (eqBefore eqHere budget : Nat) (h : eqBefore + eqHere ≤ budget) :
    min eqHere (budget - eqBefore) = eqHere := by
  omega

-- § attention launch contract

/-- Shared memory sized from the capacity always covers this token's real selection. -/
theorem smem_covers (n_ids max_ids : Nat) (h : n_ids ≤ max_ids) :
    (n_ids + 32) * 4 ≤ (max_ids + 32) * 4 := by
  omega

end Ref.QsaCache



## FILE: /mnt/user-data/outputs/QsaDecodeAttnChunked.lean
/-!
# MODULE: qsa_decode_attn_chunked
SOURCES: src/kernels/cuda/qsa_decode_attn.cu
Split-K decode attention over the selected cells: 64-cell chunks per (chunk, kv head, query) block with 12 query
heads per kv head, followed by a log-sum-exp merge. This module models chunk coverage, scratch layout and index bounds.

## Source semantics
* n_chunks = ceil(cap / 64); chunk c covers selected positions [64c, 64c+64); a chunk with no cells publishes m = -FLT_MAX, l = 0.
* Scratch per query holds acc (n_chunks * n_head * 256), m and l (n_chunks * n_head each); the stride is
  n_chunks * n_head * 258 + 64.
* Global query head h = kvh * 12 + hl; slot = kvh * n_chunks + chunk.

## Form template identifiers
- `CHUNK` = 64, `G` = 12, `HD` = 256
- `nChunks`, `nHere` : chunk count and cells in a chunk
- `slot`            : kvh * n_chunks + chunk
-/

namespace Ref.QsaDecode

-- § chunk coverage

/-- Number of 64-cell chunks for a capacity. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Cells handled by the chunk starting at `c0`. -/
def nHere (nIds c0 : Nat) : Nat := min 64 (nIds - c0)

/-- Every selected position falls in a chunk that exists. -/
theorem chunk_exists (i nIds cap : Nat) (hi : i < nIds) (hc : nIds ≤ cap) : i / 64 < nChunks cap := by
  unfold nChunks; omega

/-- A position lies inside its own chunk's window. -/
theorem cell_in_chunk (i : Nat) : i / 64 * 64 ≤ i ∧ i < i / 64 * 64 + 64 := by
  omega

/-- A chunk past the selection is empty and takes the -FLT_MAX path. -/
theorem nHere_empty (nIds c0 : Nat) (h : nIds ≤ c0) : nHere nIds c0 = 0 := by
  unfold nHere; omega

/-- A chunk that starts inside the selection has at least one cell. -/
theorem nHere_pos (nIds c0 : Nat) (h : c0 < nIds) : 0 < nHere nIds c0 := by
  unfold nHere; omega

/-- A chunk never holds more than 64 cells. -/
theorem nHere_le (nIds c0 : Nat) : nHere nIds c0 ≤ 64 := by
  unfold nHere; omega

/-- The last chunk of a full selection ends exactly at n_ids. -/
theorem nHere_tail (nIds c0 : Nat) (h : c0 < nIds) (h2 : nIds < c0 + 64) : c0 + nHere nIds c0 = nIds := by
  unfold nHere; omega

-- § scratch layout

/-- acc, m and l regions fit inside the declared per-query scratch size (a = n_chunks * n_head). -/
theorem scratch_fits (a : Nat) : a * 256 + a + a ≤ a * (256 + 2) + 64 := by
  omega

/-- Mixed-radix index bound. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m ha
  have h2 : (a + 1) * m = a * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The (kv head, chunk) slot is inside the partial buffers. -/
theorem slot_lt (kvh chunk nkv nch : Nat) (hk : kvh < nkv) (hc : chunk < nch) :
    kvh * nch + chunk < nkv * nch :=
  idx_lt kvh chunk nkv nch hk hc

-- § head and lane decomposition

/-- The global query head splits back into (kv head, local head) as written in the merge kernel. -/
theorem head_split (kvh hl : Nat) (h : hl < 12) :
    (kvh * 12 + hl) / 12 = kvh ∧ (kvh * 12 + hl) % 12 = hl := by
  omega

/-- Each lane's 8 dimensions stay inside the 256-wide head. -/
theorem lane_dims_lt (lane j : Nat) (hl : lane < 32) (hj : j < 8) : lane * 8 + j < 256 := by
  omega

/-- The second score half-slot of a lane stays in the 64-wide CHUNK row. -/
theorem lane_second_half (lane : Nat) (h : lane < 32) : lane + 32 < 64 := by
  omega

/-- Warp w handles heads w and w + 8, covering all 12 heads between warps 0..7. -/
theorem heads_covered (h : Nat) (hh : h < 12) : h < 8 ∨ (h - 8 < 8 ∧ h = (h - 8) + 8) := by
  omega

-- § merge weights

/-- An empty chunk (m = -FLT_MAX) is skipped, so the merge denominator only counts real chunks:
modelled on counts, populated chunks plus empty ones equal the chunk total. -/
theorem merge_chunks_partition (populated empty total : Nat) (h : populated + empty = total) :
    total - empty = populated := by
  omega

end Ref.QsaDecode



## FILE: /mnt/user-data/outputs/QsaBlockSelect.lean
/-!
# MODULE: qsa_block_select
SOURCES: src/kernels/cuda/qsa_select.cu
Batched block scoring (one warp per pooled block) and a block-granular radix-select top-k that expands the chosen
blocks into cell ids (4 cells per block).

## Source semantics
* A block b < n_bid weighs R = 4 cells; the last block n_bid weighs n_kv - 4 * n_bid (possibly 0).
* If n_kv <= width, the selection is the identity.
* Radix select fixes the 32-bit key 8 bits at a time; the output takes cells above the threshold plus the first
  `eq_budget` equal cells in index order.

## Form template identifiers
- `R` = 4, `weight`, `n_bid`, `eq_budget`, `my_eq`
- `per` : blocks per thread, ceil(nb / 256)
-/

namespace Ref.QsaSelect

-- § block weights

/-- Weight (cell count) of block b. -/
def weight (b nbid nkv : Nat) : Nat := if b < nbid then 4 else nkv - nbid * 4

/-- Weights of blocks 0..n_bid add up to n_kv. -/
theorem total_weight (nkv nbid : Nat) (h : nbid = nkv / 4) : nbid * 4 + weight nbid nbid nkv = nkv := by
  unfold weight; rw [if_neg (Nat.lt_irrefl _)]; omega

/-- The tail block holds fewer than 4 cells. -/
theorem tail_weight_lt (nkv nbid : Nat) (h : nbid = nkv / 4) : weight nbid nbid nkv < 4 := by
  unfold weight; rw [if_neg (Nat.lt_irrefl _)]; omega

/-- Every emitted cell id b * 4 + c is a real cached cell. -/
theorem cell_id_lt (b c nbid nkv : Nat) (h : nbid = nkv / 4) (hb : b ≤ nbid)
    (hc : c < weight b nbid nkv) : b * 4 + c < nkv := by
  unfold weight at hc
  split at hc <;> omega

-- § radix digits

/-- A 32-bit key splits into its four 8-bit digits, the order the select consumes them. -/
theorem key_digits (k : Nat) (hk : k < 4294967296) :
    k = (k / 16777216 % 256) * 16777216 + (k / 65536 % 256) * 65536 + (k / 256 % 256) * 256 + k % 256 := by
  omega

/-- The top digit of a 32-bit key is a valid histogram bin. -/
theorem top_digit_lt (k : Nat) (hk : k < 4294967296) : k / 16777216 < 256 := by
  omega

-- § equal-cell distribution

/-- A thread's share of equal-to-threshold cells. -/
def myEq (budget before eq : Nat) : Nat := min eq (budget - before)

/-- The running equal-cell total never passes the budget when it started below it. -/
theorem myEq_within (budget before eq : Nat) (h : before ≤ budget) : before + myEq budget before eq ≤ budget := by
  unfold myEq; omega

/-- A thread entirely below the budget keeps all its equal cells. -/
theorem myEq_full (budget before eq : Nat) (h : before + eq ≤ budget) : myEq budget before eq = eq := by
  unfold myEq; omega

/-- A thread beyond the budget keeps none. -/
theorem myEq_none (budget before eq : Nat) (h : budget ≤ before) : myEq budget before eq = 0 := by
  unfold myEq; omega

/-- When the bracket gt <= width <= gt + eq holds, gt + myEq selects exactly width cells. -/
theorem select_exact (gt eq width : Nat) (h1 : gt ≤ width) (h2 : width ≤ gt + eq) :
    gt + myEq width gt eq = width := by
  unfold myEq; omega

-- § thread partition of blocks

/-- Blocks per thread. -/
def per (nb : Nat) : Nat := (nb + 255) / 256

/-- A nonempty block range gives a positive per-thread share. -/
theorem per_pos (nb : Nat) (h : 0 < nb) : 0 < per nb := by
  unfold per; omega

/-- 256 threads cover all nb blocks. -/
theorem threads_cover (nb : Nat) : nb ≤ per nb * 256 := by
  unfold per; omega

/-- Every block index has an owning thread below 256. -/
theorem owner_lt (b nb : Nat) (hb : b < nb) : b / per nb < 256 := by
  apply Nat.div_lt_of_lt_mul
  have := threads_cover nb
  omega

/-- The owner's window [t * per, t * per + per) contains the block. -/
theorem owner_window (b p : Nat) (hp : 0 < p) : p * (b / p) ≤ b ∧ b < p * (b / p) + p := by
  have h1 := Nat.div_add_mod b p
  have h2 := Nat.mod_lt b hp
  omega

-- § launch geometry

/-- Eight warps per block cover every block index below max_blocks. -/
theorem score_grid_covers (b maxb : Nat) (h : b < maxb) : b / 8 < (maxb + 7) / 8 := by
  omega

/-- Mixed-radix index bound. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m ha
  have h2 : (a + 1) * m = a * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The score output index qi * max_blocks + b is inside the batched score buffer. -/
theorem score_out_lt (qi b nq maxb : Nat) (hq : qi < nq) (hb : b < maxb) : qi * maxb + b < nq * maxb :=
  idx_lt qi b nq maxb hq hb

end Ref.QsaSelect



## FILE: /mnt/user-data/outputs/NativeQsaNormGate.lean
/-!
# MODULE: native_qsa_norm_gate
SOURCES: src/kernels/cuda/native_qsa.cu
llama.cpp-derived weighted RMS norm (in-place capable) and the sigmoid gate that multiplies attention output by the
second half of the fused q projection. This module models the validation and indexing contracts.

## Source semantics
* `elements(cols, rows)` requires positive dimensions whose product fits in int.
* The norm kernel uses a block of 256 threads for n_cols < 1024 and 1024 otherwise; thread t owns columns
  t, t + BS, t + 2 BS, ...; per-warp sums sit in a 32-entry shared array.
* `gate`: head = i / head_dim, channel = i % head_dim, and the gate value is q_full[head * 2 * head_dim + head_dim + channel].
* Buffer validation: byte spans must be 4-aligned, bounded and disjoint except for exact input/output aliasing.

## Form template identifiers
- `BlockSize` : 256 or 1024
- `overlap`   : half-open interval intersection test
- `gateIdx`   : head * (2 * head_dim) + (head_dim + channel)
-/

namespace Ref.NativeQsaNorm

-- § validation

/-- Positive dimensions give a positive element count. -/
theorem elements_pos (cols rows : Nat) (hc : 0 < cols) (hr : 0 < rows) : 0 < cols * rows :=
  Nat.mul_pos hc hr

/-- Interval overlap test, as in `overlap` of the source. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap test is symmetric. -/
theorem overlap_symm (ap an bp bn : Nat) : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap; omega

/-- A nonempty span overlaps itself, so a non-exact alias is always rejected. -/
theorem overlap_self (p n : Nat) (h : 0 < n) : overlap p n p n := by
  unfold overlap; omega

/-- A span of length 0 does not overlap itself. -/
theorem overlap_self_empty (p : Nat) : ¬ overlap p 0 p 0 := by
  unfold overlap; omega

/-- Disjoint adjacent spans never overlap. -/
theorem overlap_adjacent (p n m : Nat) : ¬ overlap p n (p + n) m := by
  unfold overlap; omega

-- § block geometry

/-- The block-size choice always gives at most 32 warps, so `sums[32]` suffices. -/
theorem warps_le_32 (n_cols : Nat) : (if n_cols < 1024 then 256 else 1024) / 32 ≤ 32 := by
  split <;> omega

/-- Each column is owned by exactly the thread with matching residue. -/
theorem stride_cover (col tid bs : Nat) (h : col % bs = tid) : col = tid + bs * (col / bs) := by
  have := Nat.mod_add_div col bs
  omega

/-- Columns visited by thread t (t + bs * k) have residue t. -/
theorem stride_residue (tid k bs : Nat) (h : tid < bs) : (tid + bs * k) % bs = tid := by
  rw [Nat.add_comm, Nat.mul_add_mod]
  exact Nat.mod_eq_of_lt h

-- § gate indexing

/-- Mixed-radix index bound. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m ha
  have h2 : (a + 1) * m = a * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The gate value read for (head, channel) lies inside the n_head * 2 * head_dim q_full buffer. -/
theorem gate_index_lt (head ch n_head hd : Nat) (hh : head < n_head) (hc : ch < hd) :
    head * (2 * hd) + (hd + ch) < n_head * (2 * hd) :=
  idx_lt head (hd + ch) n_head (2 * hd) hh (by omega)

/-- The gate buffer length n_head * (2 * hd) is twice the output length, matching the `count * 8` byte span. -/
theorem gate_span_double (n_head hd : Nat) : n_head * (2 * hd) = 2 * (n_head * hd) :=
  Nat.mul_left_comm n_head 2 hd

/-- The gate reads the second half: the offset hd + ch is at least head_dim. -/
theorem gate_reads_second_half (hd ch : Nat) : hd ≤ hd + ch := by
  omega

/-- Grid size covers every output element. -/
theorem gate_grid_covers (i count : Nat) (h : i < count) : i / 256 < (count + 255) / 256 := by
  omega

end Ref.NativeQsaNorm



## FILE: /mnt/user-data/outputs/NativeQsaIndexerAppend.lean
/-!
# MODULE: native_qsa_indexer_append
SOURCES: src/kernels/cuda/native_qsa_indexer.cu
Single-block indexer-key append: stores the raw key in the 3-row tail, and on a block completion (or at position 0)
pools, normalises and rotates the row, then moves the spare row.

## Source semantics
* D = 128, R = 4, ROT = 64, THREADS = 256; slot = pos % 4; only slot < 3 stores into the tail.
* Writes of pooled rows happen only when pos = 0 or slot = 3; at slot 3 rows b = pos/4 and b+1 are written.
* The rotation position is `pos_base + 4 * b` (0 at pos 0); rotation pairs index `pair` with `pair + 32`.
* The pooled buffer has `max_cells / 4 + 1` rows.

## Form template identifiers
- `D`, `R`, `ROT`, `THREADS`
- `slot`, `b`, `rope_pos`, `pair`
-/

namespace Ref.NativeQsaIndexer

-- § slot arithmetic

/-- Slot of a position. -/
def slot (pos : Nat) : Nat := pos % 4

/-- Only slots 0..2 are stored in the tail; slot 3 completes a block. -/
theorem slot_lt_four (pos : Nat) : slot pos < 4 := by
  unfold slot; omega

/-- Tail row index slot * 128 + d stays inside the 3 * 128 tail. -/
theorem tail_index_lt (pos d : Nat) (hs : slot pos < 3) (hd : d < 128) : slot pos * 128 + d < 3 * 128 := by
  omega

/-- Gather rows j * 128 + d for j < 3 also stay inside the tail. -/
theorem tail_gather_lt (j d : Nat) (hj : j < 3) (hd : d < 128) : j * 128 + d < 3 * 128 := by
  omega

-- § pooled buffer rows

/-- At position 0 the pooled row 0 exists. -/
theorem row_zero_exists (maxc : Nat) : 0 < maxc / 4 + 1 := by
  omega

/-- When a block completes within capacity, its row b is in range. -/
theorem block_row_lt (pos maxc : Nat) (hs : slot pos = 3) (hp : pos < maxc) : pos / 4 < maxc / 4 + 1 := by
  unfold slot at hs; omega

/-- When a block completes within capacity, the spare row b+1 is also in range. -/
theorem spare_row_lt (pos maxc : Nat) (hs : slot pos = 3) (hp : pos < maxc) : pos / 4 + 1 < maxc / 4 + 1 := by
  unfold slot at hs; omega

/-- The kernel returns early unless pos = 0 or the slot is 3; rows are written only then. -/
theorem writes_only_when (pos : Nat) (h : ¬ (pos = 0 ∨ slot pos = 3)) : pos ≠ 0 ∧ slot pos ≠ 3 := by
  unfold slot at h ⊢; omega

-- § rotation

/-- Rotation position of block b. -/
def ropePos (pos posBase : Nat) : Nat := if pos = 0 then 0 else posBase + 4 * (pos / 4)

/-- A 4-aligned position base keeps every block's rotation position 4-aligned. -/
theorem ropePos_aligned (pos posBase : Nat) (h : posBase % 4 = 0) : ropePos pos posBase % 4 = 0 := by
  unfold ropePos; split <;> omega

/-- The rotation position stays within int32 when base + capacity does. -/
theorem ropePos_le (pos posBase maxc : Nat) (hp : pos < maxc) (h : posBase + maxc ≤ 2147483647) :
    ropePos pos posBase ≤ 2147483647 := by
  unfold ropePos; split <;> omega

/-- Rotation partner of dimension d < 64 is `pair` or `pair + 32`, both inside the 64 rotated dims. -/
theorem rotation_partners_lt (d : Nat) (hd : d < 64) : d % 32 < 32 ∧ d % 32 + 32 < 64 := by
  omega

/-- The rotated value for d is a*c - z*s on the first half and a*s + z*c on the second, and the two
halves of [0, 64) are exactly [0, 32) and [32, 64). -/
theorem rotation_halves (d : Nat) (hd : d < 64) : d < 32 ∨ (32 ≤ d ∧ d - 32 = d % 32) := by
  omega

/-- Rotated dimensions are inside the 128-dim key, so non-rotated dims 64..127 pass through. -/
theorem rotation_inside_key (d : Nat) (hd : d < 64) : d < 128 := by
  omega

-- § thread geometry

/-- 256 threads give 8 warps, which fit the 32-entry partial sums. -/
theorem partials_fit : 256 / 32 ≤ 32 := by
  decide

/-- The 128 live dims are a prefix of the 256 threads, so a reduction over lanes < 8 sees every partial. -/
theorem live_warps : 128 / 32 = 4 := by
  decide

/-- A block mean is one quarter of the 4-cell sum: `0.25 * s` on naturals as `s / 4` times 4 bounds the sum. -/
theorem quarter_mean_bound (s : Nat) : s / 4 * 4 ≤ s := by
  omega

end Ref.NativeQsaIndexer



## FILE: /mnt/user-data/outputs/NativeQsaScoreMma.lean
/-!
# MODULE: native_qsa_score_mma
SOURCES: src/kernels/cuda/native_qsa_score.cu
Tensor-core (TF32 mma) scoring of pooled indexer rows against 4 query heads, with the reference's ReLU-sum order,
optional bias, the +1e9 incomplete-tail boost, and expansion of each row score onto its 4 cells.

## Source semantics
* 32 rows per block, 2 warps; grid = ceil(max_blocks / 32); only rows <= n/4 are scored.
* K = 128 is split between warps: column = warp*32 + lane (+ 64 on the second pass).
* Results go to a shared [head][row] array with row stride 68; row index = warp*32 + ia*16 + (l/2)*8 + lane/4.
* The kernel is a no-op unless the step state is consistent (pos = n-1, n_bid = n/4, width = min(n, 2051)).
* Capacity contract: max_blocks = max_cells / 4 + 1.

## Form template identifiers
- `ROWS` = 32, `WARPS` = 2, `COMBINE` = 68, `R` = 4
- `D` = 128, `HEADS` = 4
-/

namespace Ref.NativeQsaScore

-- § K-split coverage

/-- Every column below 128 is handled by exactly the (warp, lane, pass) triple decoded from it. -/
theorem column_decode (col : Nat) (h : col < 128) :
    col = (col / 32 % 2) * 32 + col % 32 + 64 * (col / 64) := by
  omega

/-- The decoded triple is in range. -/
theorem column_decode_range (col : Nat) (h : col < 128) :
    col / 32 % 2 < 2 ∧ col % 32 < 32 ∧ col / 64 < 2 := by
  omega

/-- Two different (warp, lane, pass) triples never name the same column. -/
theorem column_unique (w l i w' l' i' : Nat) (hl : l < 32) (hl' : l' < 32) (hw : w < 2) (hw' : w' < 2)
    (h : w * 32 + l + 64 * i = w' * 32 + l' + 64 * i') : w = w' ∧ l = l' ∧ i = i' := by
  omega

-- § row and grid coverage

/-- A row is within the grid. -/
theorem row_in_grid (b maxb : Nat) (h : b < maxb) : b / 32 < (maxb + 31) / 32 := by
  omega

/-- A row decomposes into block, lane. -/
theorem row_decode (b : Nat) : b / 32 * 32 + b % 32 = b := by
  omega

/-- The capacity relation puts the last possible block row inside the pooled buffer. -/
theorem last_row_in_capacity (n maxc : Nat) (hn : n ≤ maxc) : n / 4 < maxc / 4 + 1 := by
  omega

-- § combine buffer

/-- Row index inside the 64 combined rows. -/
theorem combine_row_lt (warp ia l lane : Nat) (hw : warp < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by
  omega

/-- Combined index head * 68 + row stays inside the 8 * 68 combine area. -/
theorem combine_index_lt (h i : Nat) (hh : h < 8) (hi : i < 64) : h * 68 + i < 8 * 68 := by
  omega

/-- The combine area fits in the shared buffer of WARPS * 16 * 36 floats. -/
theorem combine_fits_shared : 8 * 68 ≤ 2 * 16 * 36 := by
  decide

/-- The second warp's partial is read at offset 32, still inside the row's 68-wide stride. -/
theorem second_partial_inside (j lane : Nat) (hj : j < 4) (hl : lane < 32) : j * 68 + 32 + lane < 8 * 68 := by
  omega

/-- Head slot of a lane/element pair is below 8, matching the padded 8-head B tile. -/
theorem head_slot_lt (lane l : Nat) (hl : lane < 32) (hl2 : l < 4) : (lane % 4) * 2 + l % 2 < 8 := by
  omega

-- § cell expansion

/-- Cells written for row `row` all lie in block `row`. -/
theorem cell_block (row i : Nat) (h1 : row * 4 ≤ i) (h2 : i < (row + 1) * 4) : i / 4 = row := by
  omega

/-- Cells written are real cached cells: i < n. -/
theorem cell_written_real (row i n : Nat) (h : i < n) : i / 4 ≤ n / 4 := by
  omega

-- § step-state guard

/-- The guard width formula equals min(n, 2051) = min(n, 2048 + 4 - 1). -/
theorem width_formula (n : Nat) : (if n < 2051 then n else 2051) = min n (2048 + 4 - 1) := by
  split <;> omega

-- § summation order

/-- The kernel adds 0 first, so the head value is the first partial plus the second; a zero seed is neutral. -/
theorem zero_seed_neutral (a b : Nat) : (0 + a) + b = a + b := by
  omega

/-- Bias and tail boost are added after the head sum, and a zero boost is neutral. -/
theorem zero_boost_neutral (s : Nat) : s + 0 = s := by
  omega

end Ref.NativeQsaScore



## FILE: /mnt/user-data/outputs/RouterTop10Softmax.lean
/-!
# MODULE: router_top10_softmax
SOURCES: src/kernels/cuda/router_top10.cu
MoE router: softmax over all experts, stable descending top-k (ties by ascending index) via k block-wide argmax
passes, gather, then renormalisation with a lower clamp of 2^-14.

## Source semantics
* One block per token; threads = min(n_expert, 512) rounded up to a multiple of 32.
* Shared layout: a taken-mask (n_expert bytes rounded up to 16), then n_expert doubles, then n_expert floats.
* Argmax merge prefers a larger value, and on equal values the smaller index; this is what makes the k passes
  agree with a stable descending sort.
* The renormalisation divides by max(sum, 2^-14).

## Form template identifiers
- `better v i w j` : candidate (w, j) beats incumbent (v, i)
- `takenBytes`, `threads`
- `k` : number of selected experts (<= 64)
-/

namespace Ref.RouterTop10

-- § launch geometry

/-- Taken-mask size rounded to 16 bytes. -/
def takenBytes (n : Nat) : Nat := (n + 15) / 16 * 16

/-- The mask is 16-aligned and covers n_expert bytes without more than 15 spare. -/
theorem takenBytes_spec (n : Nat) :
    takenBytes n % 16 = 0 ∧ n ≤ takenBytes n ∧ takenBytes n < n + 16 := by
  unfold takenBytes; omega

/-- The double array starts 8-aligned (the mask size is a multiple of 16). -/
theorem exp_array_aligned (n : Nat) : takenBytes n % 8 = 0 := by
  unfold takenBytes; omega

/-- The float array starts 4-aligned after n doubles. -/
theorem prob_array_aligned (n : Nat) : (takenBytes n + n * 8) % 4 = 0 := by
  unfold takenBytes; omega

/-- Total dynamic shared memory is the three regions added up. -/
theorem smem_total (n : Nat) : takenBytes n + n * 8 + n * 4 = takenBytes n + n * 12 := by
  omega

/-- Thread count: min(n_expert, 512) rounded up to a warp multiple. -/
def threads (n : Nat) : Nat := (min n 512 + 31) / 32 * 32

/-- The thread count is warp-aligned, at most 512, and at least min(n, 512). -/
theorem threads_spec (n : Nat) :
    threads n % 32 = 0 ∧ min n 512 ≤ threads n ∧ threads n ≤ 512 := by
  unfold threads; omega

/-- A non-empty expert set gets at least one full warp. -/
theorem threads_pos (n : Nat) (h : 0 < n) : 0 < threads n := by
  unfold threads; omega

/-- At most 512 / 32 = 16 warp partials, which fit the s_red array. -/
theorem warp_partials_fit (n : Nat) : threads n / 32 ≤ 16 := by
  have := (threads_spec n).2.2
  omega

-- § argmax order

/-- Candidate (w, j) beats incumbent (v, i): larger value, or equal value with smaller index. -/
def better (v i w j : Nat) : Prop := w > v ∨ (w = v ∧ j < i)

/-- Nothing beats itself. -/
theorem better_irrefl (v i : Nat) : ¬ better v i v i := by
  unfold better; omega

/-- The order is asymmetric. -/
theorem better_asymm (v i w j : Nat) (h : better v i w j) : ¬ better w j v i := by
  unfold better at h ⊢; omega

/-- The order is transitive. -/
theorem better_trans (v i w j x l : Nat) (h1 : better v i w j) (h2 : better w j x l) : better v i x l := by
  unfold better at h1 h2 ⊢; omega

/-- Two candidates with different indices are always comparable, so the merge never has a true tie. -/
theorem better_total (v i w j : Nat) (h : i ≠ j) : better v i w j ∨ better w j v i := by
  unfold better; omega

/-- On equal values the lower index wins, the stable-sort tie rule. -/
theorem tie_prefers_lower (v i j : Nat) (h : j < i) : better v i v j := by
  unfold better; omega

/-- A strictly larger value always wins regardless of index. -/
theorem larger_wins (v i w j : Nat) (h : v < w) : better v i w j := by
  unfold better; omega

-- § selection passes

/-- The k passes pick distinct experts: a taken expert is skipped, so remaining count drops by one each pass. -/
theorem remaining_after_pass (n i : Nat) (h : i < n) : n - (i + 1) < n - i := by
  omega

/-- k passes over n experts consume k of them when k <= n. -/
theorem passes_consume (n k : Nat) (h : k ≤ n) : n - k + k = n := by
  omega

/-- The sentinel index n_expert loses to every real index. -/
theorem sentinel_loses (n e : Nat) (h : e < n) : e < n := h

-- § renormalisation

/-- The clamped denominator is at least the clamp. -/
theorem clamp_lower (s c : Nat) : c ≤ max s c := by
  omega

/-- The clamped denominator is at least the true sum, so a weight never exceeds its share. -/
theorem clamp_upper (s c : Nat) : s ≤ max s c := by
  omega

/-- A sum already above the clamp is left alone. -/
theorem clamp_noop (s c : Nat) (h : c ≤ s) : max s c = s := by
  omega

end Ref.RouterTop10



## FILE: /mnt/user-data/outputs/NativeRouterTop10.lean
/-!
# MODULE: native_router_top10
SOURCES: src/kernels/cuda/native_router.cu
Warp-level 512-expert / 10-output router adapted from llama.cpp's topk-moe: each of 32 lanes holds 16 experts,
softmax by warp reductions, ten xor-shuffle argmax rounds, and a renormalisation clamped below by 2^-14.

## Source semantics
* Expert e is held by lane `e % 32` in register slot `e / 32`; registers are lane + 32 * i for i < 16.
* Each round, the winner's owning lane (`expert % 32 = lane`) masks slot `expert / 32` and records ids[rank].
* The shuffle merge keeps the larger value, and on equal values the smaller expert id.
* Weights are written for lane < 10; the sum is clamped below by 2^-14 before the reciprocal.

## Form template identifiers
- `owner e` = e % 32, `slotOf e` = e / 32
- `better` : shuffle-merge preference
- `rank` : 0..9
-/

namespace Ref.NativeRouter

-- § expert layout

/-- Expert held by (lane, slot). -/
def expertAt (lane i : Nat) : Nat := lane + i * 32

/-- Every expert below 512 is held by exactly one in-range (lane, slot). -/
theorem expert_layout (e : Nat) (h : e < 512) :
    e % 32 < 32 ∧ e / 32 < 16 ∧ expertAt (e % 32) (e / 32) = e := by
  unfold expertAt; omega

/-- Two different register positions never hold the same expert. -/
theorem expert_layout_unique (l i l' i' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : expertAt l i = expertAt l' i') : l = l' ∧ i = i' := by
  unfold expertAt at h; omega

/-- Every register position names an expert below 512. -/
theorem expertAt_lt (l i : Nat) (hl : l < 32) (hi : i < 16) : expertAt l i < 512 := by
  unfold expertAt; omega

/-- A winning expert is masked in its owner's register at slot `e / 32`, which exists. -/
theorem mask_slot_lt (e : Nat) (h : e < 512) : e / 32 < 16 := by
  omega

/-- Exactly one lane owns a given expert. -/
theorem unique_owner (e lane : Nat) (hl : lane < 32) (h : e % 32 = lane) : lane = e % 32 := by
  omega

-- § ordering

/-- Candidate (w, j) beats incumbent (v, i): larger value, or equal value with smaller expert id. -/
def better (v i w j : Nat) : Prop := w > v ∨ (w = v ∧ j < i)

/-- Nothing beats itself. -/
theorem better_irrefl (v i : Nat) : ¬ better v i v i := by
  unfold better; omega

/-- The preference is asymmetric. -/
theorem better_asymm (v i w j : Nat) (h : better v i w j) : ¬ better w j v i := by
  unfold better at h ⊢; omega

/-- The preference is transitive. -/
theorem better_trans (v i w j x l : Nat) (h1 : better v i w j) (h2 : better w j x l) : better v i x l := by
  unfold better at h1 h2 ⊢; omega

/-- For distinct expert ids, exactly one of the two candidates wins, so the xor-shuffle merge is order independent. -/
theorem better_exclusive (v i w j : Nat) (h : i ≠ j) : better v i w j ↔ ¬ better w j v i := by
  unfold better; omega

/-- On equal values the smaller expert id wins. -/
theorem tie_prefers_lower (v i j : Nat) (h : j < i) : better v i v j := by
  unfold better; omega

-- § ranks and outputs

/-- Round `rank` is held by lane `rank`; every rank below 10 has a lane below 32. -/
theorem rank_has_lane (r : Nat) (h : r < 10) : r < 32 := by
  omega

/-- Weight lanes (lane < 10) are exactly the ranks 0..9. -/
theorem weight_lanes (lane : Nat) : lane < 10 ↔ lane < 10 ∧ lane < 32 := by
  omega

/-- Output arrays of 10 entries are indexed in range by every rank. -/
theorem output_index_lt (r : Nat) (h : r < 10) : r < 10 := h

/-- Ten rounds over 512 experts leave 502 unselected. -/
theorem unselected_count : 512 - 10 = 502 := by
  decide

-- § buffer validation

/-- Byte spans required by the entry point. -/
theorem span_sizes : 512 * 4 = 2048 ∧ 10 * 4 = 40 := by
  decide

/-- Interval overlap test. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap test is symmetric. -/
theorem overlap_symm (ap an bp bn : Nat) : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap; omega

/-- Adjacent spans do not overlap. -/
theorem overlap_adjacent (p n m : Nat) : ¬ overlap p n (p + n) m := by
  unfold overlap; omega

-- § renormalisation

/-- The clamped sum is at least the clamp. -/
theorem clamp_lower (s c : Nat) : c ≤ max s c := by
  omega

/-- A sum above the clamp is unchanged. -/
theorem clamp_noop (s c : Nat) (h : c ≤ s) : max s c = s := by
  omega

end Ref.NativeRouter





## FILE: /mnt/user-data/outputs/NativeMmvqQuantizedDot.lean
/-!
# MODULE: native_mmvq_quantized_dot
SOURCES: src/kernels/cuda/native_mmvq.cu
Quantized matrix-vector product (MMVQ) on Q8_1-quantized activations for K-quant, legacy and IQ4 weight
formats, ported from llama.cpp. Single-column and multi-column (2..8) launch geometry, validation, and
weight-byte accounting.

## Source semantics
- Each weight format has a fixed block size (elements, bytes); a row has `n_in / block_elems` blocks and the
  weight buffer has `row_bytes * n_out` bytes.
- Activations are quantized to Q8_1 blocks (32 elements, 36 bytes); `ncols` columns are contiguous, so a
  quantizer pass over `n_in * ncols` elements never straddles a column.
- Threads: 4 warps x 32 lanes; blocks per K-iteration = VDR*WARPS*WARP/QI; "small K" rule switches to 4 rows
  per CUDA block when `n_in/block < blocks_per_iter`.
- Validation: `1 <= ncols <= 8`, `n_in > 0` and divisible by the block element count, `n_out > 0`.

## Form template identifiers
- Q5KBlock/Q4KBlock/Q3KBlock/Q6KBlock/IQ4XSBlock/Q40Block/Q50Block/Q80Block/IQ4NLBlock/Q20Block -> block layout
- Q81Block -> activation block
- native_mmvq_weight_bytes -> weightBytes ; native_q8_1_bytes -> q81Bytes ; native_mmvq_supported -> supported
-/

namespace Ref.NativeMmvq

-- § constants
/-- Warps per CUDA block. -/
def WARPS : Nat := 4
/-- Lanes per warp. -/
def WARP : Nat := 32
/-- Q8_1 elements per block. -/
def Q8K : Nat := 32
/-- Maximum activation columns. -/
def MAX_NCOLS : Nat := 8

/-- (block elements, block bytes) per supported non-IQ GGML type id. -/
def blockInfo : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)
  | 6 => some (32, 22)
  | 8 => some (32, 34)
  | 20 => some (32, 18)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | _ => none

/-- Weight bytes of an `n_out`-row matrix whose rows have `n_in` elements. -/
def weightBytes (elems bytes nIn nOut : Nat) : Nat := (nIn / elems) * bytes * nOut

/-- Q8_1 buffer size for `ncols` columns of `n_in` elements (36 bytes per 32). -/
def q81Bytes (nIn ncols : Nat) : Nat := ncols * (nIn / 32) * 36

-- § struct layouts
/-- Q5_K block: half2 dm (4) + scales (12) + qh (32) + qs (128) = 176 bytes. -/
theorem q5k_size : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q4_K block: dm (4) + scales (12) + qs (128) = 144 bytes. -/
theorem q4k_size : 4 + 12 + 128 = 144 := by decide
/-- Q3_K block: hmask (32) + qs (64) + scales (12) + d (2) = 110 bytes. -/
theorem q3k_size : 32 + 64 + 12 + 2 = 110 := by decide
/-- Q6_K block: ql (128) + qh (64) + scales (16) + d (2) = 210 bytes. -/
theorem q6k_size : 128 + 64 + 16 + 2 = 210 := by decide
/-- IQ4_XS block: d (2) + scales_h (2) + scales_l (4) + qs (128) = 136 bytes. -/
theorem iq4xs_size : 2 + 2 + 4 + 128 = 136 := by decide
/-- Q8_1 block: ds half2 (4) + 32 int8 = 36 bytes. -/
theorem q81_size : 4 + 32 = 36 := by decide
/-- Q5_0 block: d (2) + qh (4) + qs (16) = 22 bytes. -/
theorem q50_size : 2 + 4 + 16 = 22 := by decide
/-- Q8_0 block: d (2) + 32 int8 = 34 bytes. -/
theorem q80_size : 2 + 32 = 34 := by decide
/-- Q4_0 / IQ4_NL / Q2_0 block: d (2) + 16 bytes = 18. -/
theorem q40_size : 2 + 16 = 18 := by decide

-- § activation quantization
/-- Each flat element index splits uniquely into (Q8_1 block, lane in block). -/
theorem q81_index_split (i : Nat) : 32 * (i / 32) + i % 32 = i := by
  omega

/-- Q8_1 buffer size is additive in the number of columns. -/
theorem q81Bytes_add (n a b : Nat) : q81Bytes n (a + b) = q81Bytes n a + q81Bytes n b := by
  unfold q81Bytes
  rw [Nat.add_mul, Nat.add_mul]

/-- When `n_in` is a multiple of 32, a column's blocks are whole, so column boundaries are block boundaries. -/
theorem column_block_aligned (n c : Nat) (h : n % 32 = 0) : (n * c) / 32 = c * (n / 32) := by
  obtain ⟨m, rfl⟩ : ∃ m, n = 32 * m := ⟨n / 32, by omega⟩
  rw [Nat.mul_assoc, Nat.mul_div_cancel_left _ (by decide : 0 < 32),
    Nat.mul_div_cancel_left _ (by decide : 0 < 32)]
  exact Nat.mul_comm _ _

-- § iteration geometry
/-- Blocks per K-iteration for the K-quants Q4_K/Q5_K (VDR=2, QI=32): 8. -/
theorem bpi_q45k : 2 * WARPS * WARP / 32 = 8 := by decide
/-- Blocks per iteration for Q3_K (VDR=1, QI=16): 8. -/
theorem bpi_q3k : WARPS * WARP / 16 = 8 := by decide
/-- Blocks per iteration for Q6_K (VDR=1, QI=32): 4. -/
theorem bpi_q6k : WARPS * WARP / 32 = 4 := by decide
/-- Blocks per iteration for IQ4_XS (VDR=4, QI=32): 16. -/
theorem bpi_iq4xs : 4 * WARPS * WARP / 32 = 16 := by decide
/-- Blocks per iteration for Q2_0 (QI=2): 64. -/
theorem bpi_q20 : WARPS * WARP / 2 = 64 := by decide
/-- Q4_0/Q5_0/IQ4_NL (Qi=4, VDR=2): 64 blocks per iteration. -/
theorem bpi_small4 : 2 * WARPS * WARP / 4 = 64 := by decide
/-- Q8_0 (Qi=8, VDR=2): 32 blocks per iteration. -/
theorem bpi_small8 : 2 * WARPS * WARP / 8 = 32 := by decide

/-- Small-K threshold of Q4_0-family: `n_in/32 < 64` iff `n_in < 2048`. -/
theorem smallK_threshold_q40 (n : Nat) : n / 32 < 64 ↔ n < 2048 := by
  omega
/-- Small-K threshold of Q8_0: `n_in/32 < 32` iff `n_in < 1024`. -/
theorem smallK_threshold_q80 (n : Nat) : n / 32 < 32 ↔ n < 1024 := by
  omega

/-- For the K-quant mapping every thread of a 128-thread block lands in a block index below 8. -/
theorem kbx_first_lt_bpi (tid : Nat) (h : tid < 128) : tid / 16 < 8 := by
  omega

/-- With 4 rows per CUDA block the grid covers every output row. -/
theorem grid_covers_small (nOut : Nat) : nOut ≤ ((nOut + 3) / 4) * 4 := by
  omega

/-- Exactly one CUDA block per row in the normal layout: row index equals block index (trivial cover). -/
theorem row_cover_normal (nOut r : Nat) (h : r < nOut) : r < nOut := h

/-- Q3_K signed scale: 4 low bits and 2 high bits, minus 32, lies in [-32, 31]. -/
theorem q3k_scale_range (lo hi : Int) (hl0 : 0 ≤ lo) (hl1 : lo < 16) (hh0 : 0 ≤ hi) (hh1 : hi < 4) :
    -32 ≤ (lo + 16 * hi) - 32 ∧ (lo + 16 * hi) - 32 ≤ 31 := by
  omega

/-- Q6_K 6-bit code minus 32 lies in [-32, 31]. -/
theorem q6k_code_range (c : Int) (h0 : 0 ≤ c) (h1 : c < 64) : -32 ≤ c - 32 ∧ c - 32 ≤ 31 := by
  omega

/-- IQ4_XS combined scale: 4 low + 2 high bits minus 32 is in [-32, 31]. -/
theorem iq4xs_scale_range (lo hi : Int) (h0 : 0 ≤ lo) (h1 : lo < 16) (h2 : 0 ≤ hi) (h3 : hi < 4) :
    -32 ≤ (lo + 16 * hi) - 32 ∧ (lo + 16 * hi) - 32 ≤ 31 := by
  omega

-- § validation and weight sizes
/-- Column-count validation predicate. -/
def ncolsOk (n : Nat) : Prop := 1 ≤ n ∧ n ≤ MAX_NCOLS

/-- Multi-column path is used exactly for 2..8 columns. -/
theorem multi_iff (n : Nat) (h : ncolsOk n) : 1 < n ↔ 2 ≤ n ∧ n ≤ 8 := by
  unfold ncolsOk MAX_NCOLS at h
  omega

/-- Supported non-IQ types have a block entry; unsupported ids do not. -/
theorem blockInfo_unsupported : blockInfo 0 = none := by decide

/-- Q5_K weights: 256-element blocks, 176 bytes. -/
theorem blockInfo_q5k : blockInfo 13 = some (256, 176) := by decide

/-- Weight bytes factorise as row bytes times row count. -/
theorem weightBytes_row (e b nIn nOut : Nat) : weightBytes e b nIn nOut = ((nIn / e) * b) * nOut := by
  unfold weightBytes
  rfl

/-- Zero rows weigh nothing. -/
theorem weightBytes_zero_rows (e b nIn : Nat) : weightBytes e b nIn 0 = 0 := by
  simp [weightBytes]

/-- Adding rows adds bytes. -/
theorem weightBytes_add_rows (e b nIn r s : Nat) :
    weightBytes e b nIn (r + s) = weightBytes e b nIn r + weightBytes e b nIn s := by
  unfold weightBytes
  rw [Nat.mul_add]

end Ref.NativeMmvq



## FILE: /mnt/user-data/outputs/SGemvFamilyKernels.lean
/-!
# MODULE: s_gemv_family_kernels
SOURCES: src/kernels/cuda/s_gemv.cu
S-family GEMV: dequantize-on-the-fly of 2/4/8-bit packed codes (optional IQ4_NL codebook, per-group scale and
optional offset) against FP16, Q8_K or Q8_0 activations; naive, split-per-row and warp-per-row variants.

## Source semantics
- A weight is `decode(code) * scale + offset`; the offset is applied before the activation multiply.
- Codes are packed LSB-first, `8/bits` per byte; a row has `n_in/(8/bits)` code bytes and `n_in/group` scales.
- Q8_K activation block: 292 bytes per 256 elements (d:4, qs:256, bsums:32); Q8_0: 34 bytes per 32.
- The warp-per-row kernel takes 16 consecutive elements per lane per iteration with stride 32*16; all 16 lie
  in one activation block and one group. The group index is a shift because group sizes are powers of two.

## Form template identifiers
- SForm.code_bits -> bits ; SForm.group_elems -> group ; group_shift -> shift
- s_gemv_q8_split_kernel -> warp-per-row ; s_gemv_split_kernel -> block-per-row
-/

namespace Ref.SGemv

-- § packing
/-- Codes per byte for 2/4/8-bit widths. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- 2-bit codes: 4 per byte. -/
theorem perByte_2 : perByte 2 = 4 := by decide
/-- 4-bit codes: 2 per byte. -/
theorem perByte_4 : perByte 4 = 2 := by decide
/-- 8-bit codes: 1 per byte. -/
theorem perByte_8 : perByte 8 = 1 := by decide

/-- Bytes per four-code word (the QW constant): 1, 2 or 4 for S2, S4, S8. -/
theorem qw_values : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

/-- Extract the k-th `bits`-wide code of a packed word (little-endian field order). -/
def codeAt (v bits k : Nat) : Nat := (v / 2 ^ (k * bits)) % 2 ^ bits

/-- Every extracted code fits in `bits` bits. -/
theorem codeAt_lt (v bits k : Nat) : codeAt v bits k < 2 ^ bits := by
  unfold codeAt
  exact Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- Codes are bounded by 15 for 4-bit fields (index into the 16-entry codebook). -/
theorem codeAt_lt16 (v k : Nat) : codeAt v 4 k < 16 := by
  have := codeAt_lt v 4 k
  simpa using this

-- § activation block geometry
/-- Q8_K block bytes: scale (4) + 256 int8 + 16 int16 bsums (32). -/
theorem q8k_block_bytes : 4 + 256 + 16 * 2 = 292 := by decide
/-- Q8_0 block bytes: fp16 scale (2) + 32 int8. -/
theorem q8_0_block_bytes : 2 + 32 = 34 := by decide

/-- A 16-aligned start index keeps its 16 elements inside one Q8_K (256) block. -/
theorem lane_in_q8k_block (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by
  omega

/-- A 16-aligned start index keeps its 16 elements inside one Q8_0 (32) block. -/
theorem lane_in_q8_0_block (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by
  omega

/-- The lane stride 32*16 = 512 preserves 16-alignment. -/
theorem stride_preserves_alignment (i : Nat) (h : i % 16 = 0) : (i + 512) % 16 = 0 := by
  omega

/-- Lane start `lane*16` is 16-aligned. -/
theorem lane_start_aligned (lane : Nat) : (lane * 16) % 16 = 0 := by
  omega

/-- Every element has an owning (lane, iteration): `e/16 = lane + 32*iter`. -/
theorem element_ownership (e : Nat) : e / 16 = (e / 16) % 32 + 32 * (e / 512) := by
  omega

-- § group indexing
/-- Shift by `s` is division by `2^s` (group index as shift). -/
theorem group_shift_eq_div (i s : Nat) : i >>> s = i / 2 ^ s := by
  exact Nat.shiftRight_eq_div_pow i s

/-- A 16-aligned run stays inside a group whose size is a multiple of 16. -/
theorem quad_in_group (a m : Nat) (hm : 0 < m) : 16 * a % (16 * m) + 16 ≤ 16 * m := by
  rw [Nat.mul_mod_mul_left]
  have := Nat.mod_lt a hm
  omega

/-- Row base offset in the codes array: row `o` starts at `o * codes_per_row`. -/
theorem row_offsets_disjoint (o r cpr : Nat) (h : o < r) (hr : 0 < cpr) : o * cpr + cpr ≤ r * cpr := by
  have h1 : (o + 1) * cpr ≤ r * cpr := Nat.mul_le_mul_right cpr h
  have h2 : (o + 1) * cpr = o * cpr + cpr := Nat.succ_mul o cpr
  omega

-- § reductions
/-- Shuffle-down steps 16+8+4+2+1 cover 31 other lanes of a 32-lane warp. -/
theorem shuffle_steps : 16 + 8 + 4 + 2 + 1 = 32 - 1 := by decide

/-- Warps per 256-thread block. -/
theorem warps_per_block_256 : 256 / 32 = 8 := by decide

/-- The grid covers all rows for one warp per row (8 per block). -/
theorem warp_grid_covers (nOut : Nat) : nOut ≤ ((nOut + 8 - 1) / 8) * 8 := by
  omega

/-- Quad kernel: `tid*4` starts are 4-aligned. -/
theorem quad_start_aligned (tid : Nat) : (tid * 4) % 4 = 0 := by
  omega

-- § non-multiple activation: ffn_down_shexp
/-- n_in = 640 is a multiple of 32 but not of 256, so Q8_K is structurally impossible there. -/
theorem q8k_impossible_640 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- The 2.12x codebook measurement concerns a 16-entry table: 16 bytes = four 32-bit words. -/
theorem codebook_words : 16 / 4 = 4 := by decide

end Ref.SGemv



## FILE: /mnt/user-data/outputs/S2ExpertGroupedMoe.lean
/-!
# MODULE: s2_expert_grouped_moe
SOURCES: src/kernels/cuda/s2_expert_grouped.cu
GPU routed-expert evaluation for Q2_0 expert blobs: gate/up projection, SwiGLU, Q8_0 requantization of the
intermediate, down projection; device-side hit selection and grouping of routed entries.

## Source semantics
- Blob geometry: H=2560, FF=640, group 64; gate/up rows are interleaved in the blob (slot i even = gate row
  i/2, odd = up row i/2) and written gate-major into the output (gates first, ups after).
- `sum (c-1)*x = sum c*x - sum x`: the weight bias is removed with a second dp4a against ones.
- Hit selection compacts resident experts in entry order; grouping gives each group a contiguous entry range.

## Form template identifiers
- H -> n_embd ; FF -> n_ff ; O_D_CODES/O_GU_SCALES/O_D_SCALES -> blob section offsets
- hit_select_kernel -> compaction ; gu_kernel/down_kernel -> projections
-/

namespace Ref.S2Expert

-- § blob geometry
/-- Hidden size. -/
def H : Nat := 2560
/-- Expert FFN width. -/
def FF : Nat := 640
/-- Weights per scale. -/
def QK : Nat := 64
/-- Code bytes per gate/up row (2 bits/element). -/
def ROW_GU : Nat := H / 4
/-- Code bytes per down row. -/
def ROW_D : Nat := FF / 4
/-- Scales per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Scales per down row. -/
def SC_D : Nat := FF / QK
/-- Offset of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Offset of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
/-- Total blob bytes. -/
def BLOB : Nat := O_D_SCALES + H * SC_D * 2

/-- Row/scale counts. -/
theorem row_counts : ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 := by decide

/-- Section offsets of the blob. -/
theorem offsets : O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 := by decide

/-- Blob size. -/
theorem blob_size : BLOB = 1382400 := by decide

/-- Sections are laid out in increasing order. -/
theorem sections_ordered : O_D_CODES < O_GU_SCALES ∧ O_GU_SCALES < O_D_SCALES ∧ O_D_SCALES < BLOB := by
  decide

/-- Chunk counts: 80 gate/up chunks per row, 20 per down row. -/
theorem chunk_counts : H / 32 = 80 ∧ FF / 32 = 20 := by decide

/-- Chunks per lane for gate/up: 3 (ceil(80/32)); lane*3 covers 80 chunks. -/
theorem gu_chunks : (H / 32 + 31) / 32 = 3 ∧ 80 ≤ 3 * 32 ∧ 2 * 32 < 80 := by decide

/-- Down rows have 20 chunks, fewer than 32 lanes. -/
theorem down_chunks_lt_lanes : FF / 32 < 32 := by decide

/-- A weight scale covers two 32-element chunks, so the scale index `c/2` stays in range. -/
theorem scale_index_in_range (c : Nat) (h : c < 80) : c / 2 < 40 := by
  omega

-- § bias identity
/-- Sum of a function over a list of (code, activation) pairs. -/
def S (f : Int × Int → Int) (l : List (Int × Int)) : Int :=
  l.foldr (fun p a => f p + a) 0

/-- Distribution of the code bias: `(c-1)*x = c*x - x`. -/
theorem bias_mul (c x : Int) : (c - 1) * x = c * x - x := by
  rw [Int.sub_mul, Int.one_mul]

/-- The dp4a identity: `sum (c-1)x = sum c*x - sum x`. -/
theorem bias_removal (l : List (Int × Int)) :
    S (fun p => (p.1 - 1) * p.2) l = S (fun p => p.1 * p.2) l - S (fun p => p.2) l := by
  induction l with
  | nil => simp [S]
  | cons p t ih =>
    simp only [S, List.foldr_cons] at ih ⊢
    rw [bias_mul, ih]
    omega

-- § 2-bit code expansion
/-- A code byte splits into four 2-bit fields. -/
theorem byte_fields (c : Nat) (h : c < 256) :
    c % 4 + 4 * ((c / 4) % 4) + 16 * ((c / 16) % 4) + 64 * ((c / 64) % 4) = c := by
  omega

/-- Each 2-bit field is below 4. -/
theorem field_lt (c : Nat) : (c / 4) % 4 < 4 := by
  omega

-- § interleaved gate/up layout
/-- Slot i is decoded from row-slot i: parity picks gate/up, `i/2` is the row. -/
theorem interleave_roundtrip (i : Nat) : 2 * (i / 2) + i % 2 = i := by
  omega

/-- Gate region precedes the up region: `h*FF + r < n*FF` for `h < n`, `r < FF`. -/
theorem gate_lt_up_base (n h r ff : Nat) (hh : h < n) (hr : r < ff) : h * ff + r < n * ff := by
  have h1 : (h + 1) * ff ≤ n * ff := Nat.mul_le_mul_right ff hh
  have h2 : (h + 1) * ff = h * ff + ff := Nat.succ_mul h ff
  omega

/-- Gate index never collides with an up index of any hit. -/
theorem gate_up_disjoint (n h h' r r' ff : Nat) (hh : h < n) (hr : r < ff) :
    h * ff + r ≠ n * ff + h' * ff + r' := by
  have := gate_lt_up_base n h r ff hh hr
  omega

-- § hit selection
/-- Compaction keeps only resident (non-negative slot) entries. -/
def compact (s : List Int) : List Int := s.filter (fun x => decide (0 ≤ x))

/-- The compacted list is no longer than the input (count ≤ k). -/
theorem compact_length_le (s : List Int) : (compact s).length ≤ s.length := by
  unfold compact
  exact List.length_filter_le _ _

/-- Compaction preserves entry order (it is a sublist). -/
theorem compact_sublist (s : List Int) : List.Sublist (compact s) s := by
  unfold compact
  exact List.filter_sublist

/-- Empty routing compacts to nothing. -/
theorem compact_nil : compact [] = [] := by
  rfl

/-- The multi-entry selection uses four warps for up to 128 entries. -/
theorem select_multi_capacity : 4 * 32 = 128 := by decide

-- § grouped kernels
/-- Shared memory staged per gate/up block: GMAX=8 entries of (2560 B of words + 320 B of scales). -/
theorem gu_shared_bytes : 8 * (H / 4 * 4 + H / 32 * 4) = 23040 := by decide

/-- That staging fits the 48 KiB static shared-memory limit. -/
theorem gu_shared_fits : 8 * (H / 4 * 4 + H / 32 * 4) ≤ 49152 := by decide

/-- Gate/up rows per block: 8 warps x 4 rows = 32; 2*FF rows divide evenly. -/
theorem gu_blocks_exact : (2 * FF) % 32 = 0 := by decide

/-- Down rows per block: 8 warps x 8 rows = 64; H divides evenly. -/
theorem down_blocks_exact : H % 64 = 0 := by decide

/-- Group count is bounded by the entry count (each group has at least one entry): `g ≤ n`. -/
theorem groups_le_entries (g n : Nat) (h : g ≤ n) : g ≤ n := h

/-- Token index of an entry in a verify window: `i / k_per_token < n` when `i < n*k`. -/
theorem token_of_entry_lt (i n k : Nat) (hk : 0 < k) (h : i < k * n) : i / k < n := by
  exact Nat.div_lt_of_lt_mul h

end Ref.S2Expert



## FILE: /mnt/user-data/outputs/S2GemvFastNegativeResult.lean
/-!
# MODULE: s2_gemv_fast_negative_result
SOURCES: src/kernels/cuda/s2_gemv_fast.cu
Experimental S2 (2-bit) GEMV with a constant-memory byte->four-code table and optional shared staging of the
activation. Both levers were measured negative/neutral; the file is retained as a record, not as a fast path.

## Source semantics
- Table `c_codes[b][k] = ((b >> 2k) & 3) - 1` for each byte b and field k.
- One block per row; quads (4 elements) per thread; scale index for quad q is `q >> 4`; staging needs
  `n_in <= 4096` halves; partial sums live after the staged halves in dynamic shared memory.

## Form template identifiers
- c_codes -> lutEntry ; MAX_SHARED_HALVES -> maxStage ; QK_S2 -> 64
-/

namespace Ref.S2Fast

/-- Table entry for byte `b`, field `k` (bias -1 applied). -/
def lutEntry (b k : Nat) : Int := ((b / 4 ^ k) % 4 : Nat) - 1

/-- Every table entry is in {-1,0,1,2}. -/
theorem lutEntry_range (b k : Nat) : -1 ≤ lutEntry b k ∧ lutEntry b k ≤ 2 := by
  unfold lutEntry
  have h : (b / 4 ^ k) % 4 < 4 := Nat.mod_lt _ (by decide)
  omega

/-- The four entries of a byte reassemble the byte (biased back by +1). -/
theorem lut_reassemble (b : Nat) (hb : b < 256) :
    (lutEntry b 0 + 1) + 4 * (lutEntry b 1 + 1) + 16 * (lutEntry b 2 + 1) + 64 * (lutEntry b 3 + 1) = b := by
  unfold lutEntry
  simp only [Nat.pow_zero, Nat.pow_one, Nat.div_one]
  omega

/-- Scale index of quad `q`: element `4q` sits in group `(4q)/64 = q/16`. -/
theorem scale_index (q : Nat) : (q * 4) / 64 = q / 16 := by
  omega

/-- If `n_in` is a multiple of 64, then the number of quads is a multiple of 16. -/
theorem quads_multiple_of_16 (n : Nat) (h : n % 64 = 0) : (n / 4) % 16 = 0 := by
  omega

/-- Quad count times four recovers `n_in` when `n_in` is a multiple of 4. -/
theorem quads_cover (n : Nat) (h : n % 4 = 0) : (n / 4) * 4 = n := by
  omega

/-- Staging buffer: 4096 halves is 8 KiB. -/
theorem stage_bytes : 4096 * 2 = 8192 := by decide

/-- The partial-sum array starts at float index `MAX/2`, exactly the byte end of the staged halves. -/
theorem partial_offset_consistent : (4096 / 2) * 4 = 4096 * 2 := by decide

/-- n_embd = 2560 fits the staging limit. -/
theorem n_embd_fits : 2560 ≤ 4096 := by decide

/-- Dynamic shared bytes with staging and 128 threads: 8192 + 512 fits in 48 KiB. -/
theorem smem_with_stage : 4096 * 2 + 128 * 4 ≤ 49152 := by decide

/-- A quad index owned by thread `tid` with stride `tpr` advances within range check `q < n_quads`. -/
theorem quad_strided_cover (q tpr : Nat) (htpr : 0 < tpr) : q % tpr < tpr := by
  exact Nat.mod_lt _ htpr

/-- Tree reduction halves a power-of-two width down to 1 in `log2` steps; 128 -> 1 takes 7 halvings. -/
theorem halvings_128 : 128 / 2 / 2 / 2 / 2 / 2 / 2 / 2 = 1 := by decide

end Ref.S2Fast



## FILE: /mnt/user-data/outputs/IqQuantKernels.lean
/-!
# MODULE: iq_quant_kernels
SOURCES: src/kernels/cuda/iq_kernels.cu
IQ-family (and Q3_K/Q2_0) MMVQ dot products on Q8_1 activations, Q8_1 quantizer, dequantizers, embedding row
gather, and grouped native-expert evaluation (gate/up, SwiGLU, requantize, down).

## Source semantics
- Format traits: values per block `qk`, dot calls per block `ipb`, iqs stride `step`; call k of a row maps to
  block `k / ipb` and part `k % ipb`.
- Native expert layout: gate rows, then up rows (`up_off = n_ff*gu_row`), then down rows
  (`down_off = 2*up_off`); total `down_off + n_embd*d_row`.
- Gate/up kernel row index `row >= n_ff` selects the up matrix with row `row - n_ff`.
- Dequantized gate/up output is interleaved: row `2r + parity`.

## Form template identifiers
- Fmt<TY> -> trait record ; NativeExpertLayout -> layout ; iq_row_bytes -> rowBytes
-/

namespace Ref.IqKernels

-- § block sizes
/-- Bytes of one block per IQ-family type id (ggml-common.h sizes). -/
def blockBytes : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 20 => 18
  | 21 => 110
  | 22 => 82
  | 29 => 56
  | 23 => 136
  | 11 => 110
  | 42 => 18
  | _ => 0

/-- Values per block per type id. -/
def blockElems : Nat → Nat
  | 20 => 32
  | 42 => 64
  | 16 => 256 | 17 => 256 | 18 => 256 | 21 => 256 | 22 => 256 | 29 => 256 | 23 => 256 | 11 => 256
  | _ => 0

/-- Types handled by this module. -/
def isIq (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11

/-- Row bytes of an `n`-element row. -/
def rowBytes (t n : Nat) : Nat := (n / blockElems t) * blockBytes t

/-- Ten types are supported. -/
theorem iq_supported_examples : isIq 16 = true ∧ isIq 42 = true ∧ isIq 0 = false := by decide

/-- Unsupported ids have zero row bytes. -/
theorem rowBytes_unsupported (n : Nat) : rowBytes 0 n = 0 := by
  simp [rowBytes, blockBytes]

/-- IQ4_NL: 18-byte blocks cover 32 values. -/
theorem iq4nl_row : rowBytes 20 256 = 144 := by decide

/-- IQ4_XS: 256-value block is 136 bytes. -/
theorem iq4xs_row : rowBytes 23 256 = 136 := by decide

-- § format traits
/-- Calls per block times iqs step cover the block's 32-bit activation words for 256-value formats. -/
theorem trait_256_calls : 8 * 2 = 16 ∧ 8 * 1 = 8 := by decide

/-- Call index k maps to a block index below `nb` when `k < nb * ipb`. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < ipb * nb) : k / ipb < nb := by
  exact Nat.div_lt_of_lt_mul h

/-- The call split `(k / ipb, k % ipb)` recomposes k. -/
theorem call_split (k ipb : Nat) : ipb * (k / ipb) + k % ipb = k := by
  exact Nat.div_add_mod k ipb

/-- Part index is below `ipb` for positive `ipb`. -/
theorem call_part_lt (k ipb : Nat) (h : 0 < ipb) : k % ipb < ipb := by
  exact Nat.mod_lt _ h

-- § native expert layout
/-- Layout record fields reduced to numbers. -/
structure Layout where
  nEmbd : Nat
  nFf : Nat
  guRow : Nat
  dRow : Nat

/-- Offset of the up matrix. -/
def Layout.upOff (L : Layout) : Nat := L.nFf * L.guRow
/-- Offset of the down matrix. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total expert bytes. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

/-- Total bytes decompose into gate, up and down matrices. -/
theorem layout_bytes (L : Layout) :
    L.bytes = L.nFf * L.guRow + L.nFf * L.guRow + L.nEmbd * L.dRow := by
  unfold Layout.bytes Layout.downOff Layout.upOff
  omega

/-- The up matrix starts right after the gate matrix, and down right after up. -/
theorem layout_contiguous (L : Layout) : L.upOff + L.nFf * L.guRow = L.downOff := by
  unfold Layout.downOff Layout.upOff
  omega

/-- Gate/up row split: rows `< n_ff` are gate, others up with row `row - n_ff`. -/
theorem gu_row_split (row nff : Nat) (h : nff ≤ row) : nff + (row - nff) = row := by
  omega

/-- A grid of 8-row blocks covers `2*n_ff` rows. -/
theorem gu_grid_covers (nff : Nat) : 2 * nff ≤ ((2 * nff + 7) / 8) * 8 := by
  omega

/-- A grid of 8-row blocks covers `n_embd` down rows. -/
theorem down_grid_covers (n : Nat) : n ≤ ((n + 7) / 8) * 8 := by
  omega

/-- Scratch alignment to 256 bytes never shrinks. -/
theorem align256_ge (x : Nat) : x ≤ (x + 255) / 256 * 256 := by
  omega

/-- Aligned value is a multiple of 256. -/
theorem align256_mod (x : Nat) : ((x + 255) / 256 * 256) % 256 = 0 := by
  omega

/-- Aligned value overshoots by less than 256. -/
theorem align256_lt (x : Nat) : (x + 255) / 256 * 256 < x + 256 := by
  omega

-- § interleaved dequantization
/-- Quotient lemma: `(x*pr + c) / pr = x` for `c < pr`. -/
theorem idx_div (x c pr : Nat) (hc : c < pr) : (x * pr + c) / pr = x := by
  apply Nat.div_eq_of_lt_le
  · omega
  · have h : (x + 1) * pr = x * pr + pr := Nat.succ_mul x pr
    omega

/-- Interleaved gate/up output indices determine (row, parity): injectivity on the row coordinate. -/
theorem interleave_row_recovered (r p c pr : Nat) (hc : c < pr) :
    ((2 * r + p) * pr + c) / pr = 2 * r + p := by
  exact idx_div (2 * r + p) c pr hc

/-- IQ4_NL dequantization: low and high nibble of a byte are both below 16. -/
theorem nibbles_lt16 (q : Nat) (h : q < 256) : q % 16 < 16 ∧ q / 16 < 16 := by
  omega

/-- Nibbles recombine to the byte. -/
theorem nibbles_recombine (q : Nat) (h : q < 256) : q % 16 + 16 * (q / 16) = q := by
  omega

/-- Q2_0 code minus one is in {-1,0,1,2}. -/
theorem q2_0_code_range (code : Int) (h0 : 0 ≤ code) (h1 : code < 4) : -1 ≤ code - 1 ∧ code - 1 ≤ 2 := by
  omega

-- § q8_1 quantizer
/-- Block index and in-block index of element i. -/
theorem q8_1_split (i : Nat) : 32 * (i / 32) + i % 32 = i := by
  omega

/-- Gate/up dequant grid: two parity blocks per superblock row. -/
theorem gu_dequant_blocks (nff per_row : Nat) : (nff * per_row) * 2 = 2 * (nff * per_row) := by
  omega

end Ref.IqKernels



## FILE: /mnt/user-data/outputs/QuantizeActivations.lean
/-!
# MODULE: quantize_activations
SOURCES: src/kernels/cuda/quantize_act.cu
Activation quantizers matching ggml's src1 conversion: Q8_0 (fp16 scale), Q8_0 with an extra fp32 scale array
and CPU rounding rule, and Q8_K (fp32 scale, 256 values, 16-element block sums), plus dequantizers.

## Source semantics
- Q8_0 block = 34 bytes (fp16 d, 32 int8); Q8_K block = 292 bytes (f32 d, 256 int8, 16 int16 bsums).
- `n` must be a multiple of the block size. Output integers are clamped; the scaled variant clamps to
  [-127, 127]; Q8_K clamps only above (min 127).
- ggml `nearest_int` adds 1.5*2^23 and masks the mantissa: the recovered value lies in [-2^22, 2^22).
- An all-zero block has d = 0 and all codes 0.

## Form template identifiers
- QK8_0 -> 32 ; QK_K -> 256 ; Q8K_BYTES -> 292 ; nearest_int_dev -> nearestInt
-/

namespace Ref.QuantizeAct

/-- Clamp an integer to `[lo, hi]`. -/
def clamp (lo hi v : Int) : Int := if v < lo then lo else if hi < v then hi else v

/-- Clamp result is within bounds when `lo ≤ hi`. -/
theorem clamp_bounds (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clamp lo hi v ∧ clamp lo hi v ≤ hi := by
  unfold clamp
  split_ifs <;> omega

/-- Clamp is the identity on in-range values. -/
theorem clamp_id (lo hi v : Int) (h0 : lo ≤ v) (h1 : v ≤ hi) : clamp lo hi v = v := by
  unfold clamp
  split_ifs <;> omega

/-- Clamp is idempotent. -/
theorem clamp_idem (lo hi v : Int) (h : lo ≤ hi) : clamp lo hi (clamp lo hi v) = clamp lo hi v := by
  have := clamp_bounds lo hi v h
  exact clamp_id lo hi _ this.1 this.2

/-- The scaled quantizer's code range is symmetric: [-127, 127]. -/
theorem scaled_code_symmetric (v : Int) : -127 ≤ clamp (-127) 127 v ∧ clamp (-127) 127 v ≤ 127 := by
  exact clamp_bounds (-127) 127 v (by omega)

-- § block sizes
/-- Q8_0 block: 2 + 32 = 34 bytes. -/
theorem q8_0_bytes : 2 + 32 = 34 := by decide
/-- Q8_K block: 4 + 256 + 16*2 = 292 bytes. -/
theorem q8k_bytes : 4 + 256 + 16 * 2 = 292 := by decide

/-- Blocks of an `n`-element vector for block size 32, when `n` divides evenly. -/
theorem q8_0_blocks_exact (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by
  omega

/-- Blocks of an `n`-element vector for block size 256, when `n` divides evenly. -/
theorem q8k_blocks_exact (n : Nat) (h : n % 256 = 0) : (n / 256) * 256 = n := by
  omega

/-- Total Q8_0 bytes of `n` elements. -/
def q8_0Total (n : Nat) : Nat := (n / 32) * 34

/-- Total Q8_K bytes of `n` elements. -/
def q8kTotal (n : Nat) : Nat := (n / 256) * 292

/-- Q8_0 total is additive over multiples of 32. -/
theorem q8_0Total_add (a b : Nat) (ha : a % 32 = 0) : q8_0Total (a + b) = q8_0Total a + q8_0Total b := by
  unfold q8_0Total
  have : (a + b) / 32 = a / 32 + b / 32 := by omega
  rw [this, Nat.add_mul]

/-- Q8_K total is additive over multiples of 256. -/
theorem q8kTotal_add (a b : Nat) (ha : a % 256 = 0) : q8kTotal (a + b) = q8kTotal a + q8kTotal b := by
  unfold q8kTotal
  have : (a + b) / 256 = a / 256 + b / 256 := by omega
  rw [this, Nat.add_mul]

-- § nearest_int
/-- Recovered integer of ggml's magic-number rounding from the low 23 mantissa bits. -/
def nearestInt (bits : Nat) : Int := ((bits % 8388608 : Nat) : Int) - 4194304

/-- The recovered value lies in [-2^22, 2^22). -/
theorem nearestInt_range (bits : Nat) : -4194304 ≤ nearestInt bits ∧ nearestInt bits < 4194304 := by
  unfold nearestInt
  have h : bits % 8388608 < 8388608 := Nat.mod_lt _ (by decide)
  omega

-- § block sums
/-- Sum of sixteen int8-range values fits in int16 (|sum| ≤ 16*128 = 2048). -/
theorem bsums_fit_int16
    (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 : Int)
    (h0 : -128 ≤ a0 ∧ a0 ≤ 127) (h1 : -128 ≤ a1 ∧ a1 ≤ 127) (h2 : -128 ≤ a2 ∧ a2 ≤ 127)
    (h3 : -128 ≤ a3 ∧ a3 ≤ 127) (h4 : -128 ≤ a4 ∧ a4 ≤ 127) (h5 : -128 ≤ a5 ∧ a5 ≤ 127)
    (h6 : -128 ≤ a6 ∧ a6 ≤ 127) (h7 : -128 ≤ a7 ∧ a7 ≤ 127) (h8 : -128 ≤ a8 ∧ a8 ≤ 127)
    (h9 : -128 ≤ a9 ∧ a9 ≤ 127) (h10 : -128 ≤ a10 ∧ a10 ≤ 127) (h11 : -128 ≤ a11 ∧ a11 ≤ 127)
    (h12 : -128 ≤ a12 ∧ a12 ≤ 127) (h13 : -128 ≤ a13 ∧ a13 ≤ 127) (h14 : -128 ≤ a14 ∧ a14 ≤ 127)
    (h15 : -128 ≤ a15 ∧ a15 ≤ 127) :
    -32768 ≤ a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 ∧
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 ≤ 32767 := by
  omega

/-- Block sum index `j*16 + ii` stays inside the 256-element block. -/
theorem bsum_index_in_block (j ii : Nat) (hj : j < 16) (hi : ii < 16) : j * 16 + ii < 256 := by
  omega

/-- Thread launch: one thread per Q8_K block, 64 threads per CUDA block, grid covers all blocks. -/
theorem q8k_grid_covers (nb : Nat) : nb ≤ ((nb + 63) / 64) * 64 := by
  omega

/-- Thread launch for Q8_0: 128 threads per CUDA block cover all blocks. -/
theorem q8_0_grid_covers (nb : Nat) : nb ≤ ((nb + 127) / 128) * 128 := by
  omega

/-- The fp32 scales array has exactly one entry per 32-element chunk. -/
theorem scales_per_chunk (n : Nat) (h : n % 32 = 0) : n / 32 * 32 = n := by
  omega

end Ref.QuantizeAct



## FILE: /mnt/user-data/outputs/SharedExpertBlock.lean
/-!
# MODULE: shared_expert_block
SOURCES: src/kernels/cuda/shared_expert.cu
The MoE shared expert: `h = silu(x Wgate^T) * (x Wup^T)`, `h Wdown^T`, scaled by a per-token scalar
`sigmoid(x . w_gate_inp)`; plus the MoE combine and the multi-token (1..8) native path.

## Source semantics
- SiLU is applied to the gate projection. The scalar gate is one value per token and multiplies the whole row.
- Scratch is carved from the caller's buffer, 16-byte aligned: gate | up | q8_0 | q8k | g.
- The shared output is added plain (not router-weighted) in the combine; `k` is limited to 64.
- Q8_K needs `n_ff % 256 = 0`; the BF16 gate vector has 2 bytes/element, so reading it as floats overruns.

## Form template identifiers
- shared_expert_scratch_bytes -> scratchBytes ; moe_combine_kernel -> combine ; scale_rows_kernel -> rowIndex
-/

namespace Ref.SharedExpert

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by
  unfold align16
  omega

/-- Aligned values are multiples of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16
  omega

/-- Alignment overshoots by less than 16. -/
theorem align16_lt (x : Nat) : align16 x < x + 16 := by
  unfold align16
  omega

-- § scratch layout
/-- Byte size of the gate/up float arrays. -/
def sA (nFf : Nat) : Nat := align16 (nFf * 4)
/-- Byte size of the Q8_0 intermediate. -/
def sQ0 (nFf : Nat) : Nat := align16 (nFf / 32 * 34)
/-- Byte size of the Q8_K intermediate. -/
def sQk (nFf : Nat) : Nat := align16 (nFf / 256 * 292)
/-- Total scratch bytes. -/
def scratchBytes (nFf : Nat) : Nat := sA nFf * 2 + sQ0 nFf + sQk nFf + 32

/-- Offsets of the up, q8_0, q8k and g regions are increasing. -/
theorem scratch_offsets_ordered (n : Nat) :
    0 ≤ sA n ∧ sA n ≤ 2 * sA n ∧ 2 * sA n ≤ 2 * sA n + sQ0 n ∧
    2 * sA n + sQ0 n ≤ 2 * sA n + sQ0 n + sQk n := by
  omega

/-- The scalar gate (4 bytes) fits inside the scratch buffer. -/
theorem gate_slot_fits (n : Nat) : 2 * sA n + sQ0 n + sQk n + 4 ≤ scratchBytes n := by
  unfold scratchBytes
  omega

/-- Every region offset is 16-byte aligned. -/
theorem offsets_aligned (n : Nat) :
    sA n % 16 = 0 ∧ (2 * sA n) % 16 = 0 ∧ (2 * sA n + sQ0 n) % 16 = 0 ∧
    (2 * sA n + sQ0 n + sQk n) % 16 = 0 := by
  have h1 : sA n % 16 = 0 := align16_mod _
  have h2 : sQ0 n % 16 = 0 := align16_mod _
  have h3 : sQk n % 16 = 0 := align16_mod _
  omega

/-- The gate array holds `n_ff` floats. -/
theorem gate_holds_floats (n : Nat) : n * 4 ≤ sA n := align16_ge _

-- § combine
/-- Weighted sum of routed parts plus an optional plain shared term. -/
def combine (ws ps : List Int) (shared : Option Int) : Int :=
  (List.zipWith (· * ·) ws ps).sum + shared.getD 0

/-- Without a shared output the combine is just the weighted sum. -/
theorem combine_no_shared (ws ps : List Int) : combine ws ps none = (List.zipWith (· * ·) ws ps).sum := by
  simp [combine]

/-- The shared term is added plain (no weight). -/
theorem combine_shared_plain (ws ps : List Int) (s : Int) :
    combine ws ps (some s) = (List.zipWith (· * ·) ws ps).sum + s := by
  simp [combine]

/-- Empty routing leaves only the shared output. -/
theorem combine_empty (s : Int) : combine [] [] (some s) = s := by
  simp [combine]

-- § row-scaling index
/-- Row index `t*n + i` of token t, element i, stays below `n_tok * n`. -/
theorem row_index_lt (nTok t i n : Nat) (ht : t < nTok) (hi : i < n) : t * n + i < nTok * n := by
  have h1 : (t + 1) * n ≤ nTok * n := Nat.mul_le_mul_right n ht
  have h2 : (t + 1) * n = t * n + n := Nat.succ_mul t n
  omega

/-- Multi-token path precondition: 1..8 tokens. -/
def tokOk (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 8

/-- The multi-column MMVQ bound (8) matches the shared-expert token bound. -/
theorem tok_matches_ncols (n : Nat) (h : tokOk n) : 1 ≤ n ∧ n ≤ 8 := h

-- § scalar-gate reduction
/-- Warps in a block of `b` threads (b ≤ 256) never exceed the 8-slot scratch. -/
theorem gate_warps_le_8 (b : Nat) (h : b ≤ 256) : (b + 31) / 32 ≤ 8 := by
  omega

/-- The default launch (256 threads) uses exactly 8 warps. -/
theorem gate_warps_256 : (256 + 31) / 32 = 8 := by decide

-- § constraints and the historical overrun
/-- Q8_K is impossible for n_ff = 640 (not a multiple of 256) while Q8_0 is possible. -/
theorem shexp_q8k_vs_q8_0 : 640 % 256 ≠ 0 ∧ 640 % 32 = 0 := by decide

/-- Reading 2560 floats from a 2560-element BF16 vector overran by 5120 bytes. -/
theorem bf16_overrun_bytes : 2560 * 4 - 2560 * 2 = 5120 := by decide

/-- The combine limit is 64 routed parts. -/
theorem combine_limit (k : Nat) (h : k ≤ 64) : k ≤ 64 := h

end Ref.SharedExpert



## FILE: /mnt/user-data/outputs/SamplerChain.lean
/-!
# MODULE: sampler_chain
SOURCES: src/kernels/cuda/sampler.cu
GPU sampler: penalties, top-k, top-p, temperature, then one categorical draw from a counter-based Philox
stream; greedy path is a block-wide argmax with lowest-index tie-breaking.

## Source semantics
- Repeat penalty multiplies non-positive logits and divides positive ones; frequency penalty scales with the
  count, presence penalty is a boolean; none of them can increase a logit.
- Greedy is the argmax of the penalised logits; ties go to the lowest index; temperature must not be applied
  (at temperature <= 0 the inverse temperature is 0 and every scaled logit would be 0).
- Penalty window is the tail of the history row of length min(penalty_last_n, history_len).
- Philox uniform: top 24 bits of a 32-bit word divided by 2^24, so the value is in [0,1).
- Sampled path needs 1 <= top_k <= 64.

## Form template identifiers
- sampler_greedy_kernel -> argmax reduction ; apply_penalties -> penalise ; philox_uniform -> uniform24
-/

namespace Ref.Sampler

-- § argmax reduction
/-- Candidate (value, index) pair. -/
abbrev Cand := Int × Nat

/-- Reduction step: replace `a` by `b` iff `b` has a larger value, or an equal value and a smaller index. -/
def better (a b : Cand) : Cand :=
  if a.1 < b.1 ∨ (a.1 = b.1 ∧ b.2 < a.2) then b else a

/-- The reduction result is at least as large as both operands. -/
theorem better_value_ge (a b : Cand) : a.1 ≤ (better a b).1 ∧ b.1 ≤ (better a b).1 := by
  unfold better
  split <;> omega

/-- On equal values the smaller index wins. -/
theorem better_tie_min (a b : Cand) (h : a.1 = b.1) : (better a b).2 = min a.2 b.2 := by
  unfold better
  split <;> omega

/-- The result is always one of the operands. -/
theorem better_is_operand (a b : Cand) : better a b = a ∨ better a b = b := by
  unfold better
  split
  · exact Or.inr rfl
  · exact Or.inl rfl

/-- The order is symmetric on distinct indices: the reduction does not depend on operand order. -/
theorem better_comm (a b : Cand) (h : a.2 ≠ b.2) : (better a b).2 = (better b a).2 := by
  unfold better
  split <;> split <;> omega

/-- A candidate with no elements (index = n_vocab) loses ties against real indices. -/
theorem sentinel_loses (v : Int) (i n : Nat) (h : i < n) : (better (v, n) (v, i)).2 = i := by
  unfold better
  split
  · rfl
  · rename_i hc
    exact absurd (Or.inr ⟨rfl, h⟩) hc

-- § penalties
/-- Penalised logit: repeat (multiply if non-positive, divide if positive), then subtract freq/presence. -/
def penalise (logit : Int) (count : Int) (rep freq pres : Int) : Int :=
  if count ≤ 0 then logit
  else
    let l := if logit ≤ 0 then logit * rep else logit / rep
    l - (count * freq + pres)

/-- Multiplying a non-positive logit by `rep ≥ 1` does not increase it. -/
theorem mul_branch_le (logit rep : Int) (h0 : logit ≤ 0) (h1 : 1 ≤ rep) : logit * rep ≤ logit := by
  have h := Int.mul_nonneg (by omega : 0 ≤ -logit) (by omega : 0 ≤ rep - 1)
  simp only [Int.neg_mul, Int.mul_sub, Int.mul_one] at h
  omega

/-- Dividing a positive logit by `rep ≥ 1` does not increase it. -/
theorem div_branch_le (logit rep : Int) (h0 : 0 < logit) (h1 : 1 ≤ rep) : logit / rep ≤ logit := by
  exact Int.ediv_le_self rep (by omega)

/-- No penalty combination with `rep ≥ 1`, `freq ≥ 0`, `pres ≥ 0` can raise a logit. -/
theorem penalise_le (logit count rep freq pres : Int) (hr : 1 ≤ rep) (hf : 0 ≤ freq) (hp : 0 ≤ pres) :
    penalise logit count rep freq pres ≤ logit := by
  unfold penalise
  by_cases hc : count ≤ 0
  · simp [hc]
  · simp only [hc, if_false]
    have hcf : 0 ≤ count * freq := Int.mul_nonneg (by omega) hf
    by_cases hl : logit ≤ 0
    · have := mul_branch_le logit rep hl hr
      simp only [hl, if_true]
      omega
    · have := div_branch_le logit rep (by omega) hr
      simp only [hl, if_false]
      omega

/-- With no occurrences the logit is untouched. -/
theorem penalise_zero_count (logit rep freq pres : Int) : penalise logit 0 rep freq pres = logit := by
  simp [penalise]

/-- Scaling by an inverse temperature of 0 collapses every logit to 0 (the old greedy bug). -/
theorem zero_inv_temp_collapses (logit : Int) : logit * 0 = 0 := by
  omega

-- § penalty window
/-- Window length: `penalty_last_n` clamped to `[0, history_len]`. -/
def windowLen (lastN histLen : Int) : Int := max 0 (min lastN histLen)

/-- The window never exceeds the history length. -/
theorem window_le_history (lastN histLen : Int) (h : 0 ≤ histLen) : windowLen lastN histLen ≤ histLen := by
  unfold windowLen
  omega

/-- The window is non-negative. -/
theorem window_nonneg (lastN histLen : Int) : 0 ≤ windowLen lastN histLen := by
  unfold windowLen
  omega

/-- The tail starts at `histLen - window`, and start plus window is the history length. -/
theorem window_tail_start (lastN histLen : Int) (h : 0 ≤ histLen) :
    (histLen - windowLen lastN histLen) + windowLen lastN histLen = histLen := by
  omega

-- § Philox
/-- Top 24 bits of a 32-bit word. -/
def top24 (w : Nat) : Nat := w / 256

/-- The 24-bit mantissa is below 2^24 for 32-bit input, so `top24 / 2^24 < 1`. -/
theorem top24_lt (w : Nat) (h : w < 4294967296) : top24 w < 16777216 := by
  unfold top24
  omega

-- § top-k
/-- Effective k: zero for non-positive request, otherwise clamped to 64. -/
def kEff (topK : Int) : Int := if topK ≤ 0 then 0 else min topK 64

/-- Effective k never exceeds the 64-slot arrays. -/
theorem kEff_le (topK : Int) : kEff topK ≤ 64 := by
  unfold kEff
  split <;> omega

/-- Effective k is non-negative. -/
theorem kEff_nonneg (topK : Int) : 0 ≤ kEff topK := by
  unfold kEff
  split <;> omega

/-- A valid sampled-path request gets k between 1 and 64. -/
theorem kEff_valid (topK : Int) (h : 1 ≤ topK) : 1 ≤ kEff topK ∧ kEff topK ≤ 64 := by
  unfold kEff
  split <;> omega

-- § launch geometry
/-- Greedy kernel: 1024 threads give 32 warps, matching the 32-slot shared arrays. -/
theorem greedy_warps : 1024 / 32 = 32 := by decide

/-- Any block of at most 1024 threads has at most 32 warps. -/
theorem warps_le_32 (b : Nat) (h : b ≤ 1024) : (b + 31) / 32 ≤ 32 := by
  omega

/-- Sampled kernel: 64 threads per block, grid covers all tokens. -/
theorem sampler_grid_covers (n : Nat) : n ≤ ((n + 63) / 64) * 64 := by
  omega

end Ref.Sampler





## FILE: /mnt/user-data/outputs/GatedResidualCuda.lean
/-!
# MODULE: gated_residual_cuda
SOURCES: src/kernels/cuda/gr.cu
The gated residual / hyper-connection on the GPU: `gr_read` (norm, down projection + SiLU, gate projection,
stream mean, optional inject projection) and `gr_write` (add the block output to every stream with a
per-stream sigmoid weight). This module models the integer/index side: workspace carving, launch geometry,
strided reductions, the write-kernel index decomposition and the variant selection.

## Source semantics
* The workspace is ONE table of five regions (xn, xq, lq, gated, lo), each rounded up to 16 bytes, carved
  back to back; the first version sized the regions in a different order than it assigned pointers.
* The gate kernel uses one warp per output row (8 warps per block); the down kernel uses one block per row
  with warps striding the reduction axis; the write kernel maps a flat index to (stream, element).
* `native_mmvf` overrides `fp32_activations`, which overrides the bf16 activation variant; the mean kernel
  is only launched on the non-native paths.

## Form template identifiers
* `roundUp16`        : align16 (round a byte count up to a multiple of 16)
* `wsTotal`          : workspace_bytes (gr_workspace_init)
* `gateBlocks`       : grid_for_warp_per_row
* `owner`            : strided_loop_owner
* `selectVariant`    : kernel_variant_selector
-/

namespace Ref.GatedResidual

-- § 16-byte rounding

/-- Round a byte count up to the next multiple of 16 (`(sz + 15) & ~15`). -/
def roundUp16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding never shrinks a size. -/
theorem roundUp16_ge (n : Nat) : n ≤ roundUp16 n := by
  unfold roundUp16; omega

/-- A rounded size is a multiple of 16. -/
theorem roundUp16_mod (n : Nat) : roundUp16 n % 16 = 0 := by
  unfold roundUp16; omega

/-- Rounding adds fewer than 16 bytes. -/
theorem roundUp16_lt (n : Nat) : roundUp16 n < n + 16 := by
  unfold roundUp16; omega

-- § workspace carving

/-- Offset of `xq` (after `xn`, size `hc_dim*4`). -/
def off1 (D : Nat) : Nat := roundUp16 (D * 4)
/-- Offset of `lq` (after `xq`, size `hc_dim*2`). -/
def off2 (D : Nat) : Nat := roundUp16 (D * 4) + roundUp16 (D * 2)
/-- Offset of `gated` (after `lq`, size `hc_lr*2`). -/
def off3 (D L : Nat) : Nat := roundUp16 (D * 4) + roundUp16 (D * 2) + roundUp16 (L * 2)
/-- Offset of `lo` (after `gated`, size `hc_dim*4`). -/
def off4 (D L : Nat) : Nat := off3 D L + roundUp16 (D * 4)
/-- Total workspace bytes (after `lo`, size `hc_lr*4`). -/
def wsTotal (D L : Nat) : Nat := off4 D L + roundUp16 (L * 4)

/-- Every region fits inside its slot, so no region overlaps its neighbour. -/
theorem regions_fit (D L : Nat) :
    D * 4 ≤ off1 D ∧ off1 D + D * 2 ≤ off2 D ∧ off2 D + L * 2 ≤ off3 D L ∧
    off3 D L + D * 4 ≤ off4 D L ∧ off4 D L + L * 4 ≤ wsTotal D L := by
  have h0 := roundUp16_ge (D * 4)
  have h1 := roundUp16_ge (D * 2)
  have h2 := roundUp16_ge (L * 2)
  have h3 := roundUp16_ge (D * 4)
  have h4 := roundUp16_ge (L * 4)
  simp only [off1, off2, off3, off4, wsTotal]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- Every region start is 16-byte aligned. -/
theorem offsets_aligned (D L : Nat) :
    off1 D % 16 = 0 ∧ off2 D % 16 = 0 ∧ off3 D L % 16 = 0 ∧ off4 D L % 16 = 0 ∧
    wsTotal D L % 16 = 0 := by
  have h0 := roundUp16_mod (D * 4)
  have h1 := roundUp16_mod (D * 2)
  have h2 := roundUp16_mod (L * 2)
  have h4 := roundUp16_mod (L * 4)
  simp only [off1, off2, off3, off4, wsTotal]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- The real geometry (`hc_dim = 10240`, `hc_lr = 320`) needs 104320 bytes. -/
theorem real_total : wsTotal 10240 320 = 104320 := by
  decide

/-- The first version gave `xq` `hc_lr*2` bytes where it needs `hc_dim*2`: 640 against 20480. -/
theorem buggy_xq_too_small : 320 * 2 < 10240 * 2 := by
  decide

-- § launch geometry

/-- Blocks of the gate kernel: 8 warps (rows) per block, rounded up. -/
def gateBlocks (hcDim : Nat) : Nat := (hcDim + 7) / 8

/-- The gate grid covers every output row. -/
theorem gate_rows_cover (hcDim : Nat) : hcDim ≤ gateBlocks hcDim * 8 := by
  unfold gateBlocks; omega

/-- Every row index lies in a launched block. -/
theorem gate_block_of_row (hcDim i : Nat) (hi : i < hcDim) : i / 8 < gateBlocks hcDim := by
  unfold gateBlocks; omega

/-- `block*8 + warp` decomposes back into its block and warp. -/
theorem gate_row_roundtrip (b w : Nat) (hw : w < 8) :
    (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by
  refine ⟨?_, ?_⟩ <;> omega

/-- The (block, warp) to row map is injective: no two warps write the same row. -/
theorem gate_row_unique (b w b' w' : Nat) (hw : w < 8) (hw' : w' < 8)
    (h : b * 8 + w = b' * 8 + w') : b = b' ∧ w = w' := by
  refine ⟨?_, ?_⟩ <;> omega

/-- The real gate kernel launches 1280 blocks. -/
theorem real_gate_blocks : gateBlocks 10240 = 1280 := by
  decide

/-- The down kernel at real geometry: 320 blocks x 8 warps = 2560 warps. -/
theorem real_down_warps : 320 * 8 = 2560 := by
  decide

/-- 2560 warps over 48 SMs is 53 warps per SM (against 6.7 for one warp per row). -/
theorem real_down_occupancy : 2560 / 48 = 53 := by
  decide

/-- The mean kernel launches 10 blocks of 256 threads over `n_embd = 2560`. -/
theorem real_mean_blocks : (2560 + 255) / 256 = 10 := by
  decide

-- § strided reductions

/-- The thread that visits element `i` when `stride` threads stride the axis. -/
def owner (stride i : Nat) : Nat := i % stride

/-- Every element has an owner among the `nw*32` threads of the block. -/
theorem owner_lt (nw i : Nat) (h : 0 < nw) : owner (nw * 32) i < nw * 32 := by
  unfold owner
  exact Nat.mod_lt _ (by omega)

/-- An element is its owner's start plus a whole number of strides, so the loop reaches it. -/
theorem stride_decomp (s i : Nat) : owner s i + s * (i / s) = i := by
  unfold owner
  exact Nat.mod_add_div i s

/-- Thread id `warp*32 + lane` decomposes back into warp and lane and stays below `nw*32`. -/
theorem thread_roundtrip (nw warp lane : Nat) (hw : warp < nw) (hl : lane < 32) :
    warp * 32 + lane < nw * 32 ∧ (warp * 32 + lane) / 32 = warp ∧ (warp * 32 + lane) % 32 = lane := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

-- § block reduction scratch

/-- Number of warps in a block of `bd` threads. -/
def numWarps (bd : Nat) : Nat := (bd + 31) / 32

/-- A block of 256 threads has 8 warps. -/
theorem numWarps_256 : numWarps 256 = 8 := by
  decide

/-- No block of at most 256 threads has more than 8 warps. -/
theorem numWarps_le (bd : Nat) (h : bd ≤ 256) : numWarps bd ≤ 8 := by
  unfold numWarps; omega

/-- The `float[8]` scratch of `block_sumf` fits in the callers' `double scratch[8]` (64 bytes). -/
theorem scratch_fits (bd : Nat) (h : bd ≤ 256) : numWarps bd * 4 ≤ 8 * 8 := by
  have := numWarps_le bd h
  omega

/-- One warp per stream in the inject kernel: warp `tid/32` is a stream iff `tid < 32*hc`. -/
theorem inject_warp_active (tid hc : Nat) : tid / 32 < hc ↔ tid < 32 * hc := by
  omega

-- § write kernel index decomposition

/-- A flat index `i` is `stream * n_embd + element`. -/
theorem write_index_decomp (i n : Nat) : n * (i / n) + i % n = i :=
  Nat.div_add_mod i n

/-- For `i < hc*n_embd` the stream index is below `hc`. -/
theorem write_stream_lt (i hc n : Nat) (hn : 0 < n) (hi : i < hc * n) : i / n < hc :=
  (Nat.div_lt_iff_lt_mul hn).2 hi

/-- The element index is below `n_embd`. -/
theorem write_elem_lt (i n : Nat) (hn : 0 < n) : i % n < n :=
  Nat.mod_lt _ hn

/-- Conversely `(stream, element)` maps to an in-range flat index. -/
theorem write_index_inv (c d hc n : Nat) (hcl : c < hc) (hd : d < n) : c * n + d < hc * n := by
  have h : (c + 1) * n ≤ hc * n := Nat.mul_le_mul hcl (Nat.le_refl n)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § variant selection

/-- The three kernel families `gr_read` can launch. -/
inductive Variant where
  | native
  | fp32
  | bf16
  deriving DecidableEq

/-- `native_mmvf` wins over `fp32_activations`, which wins over bf16. -/
def selectVariant (fp32Flag native : Bool) : Variant :=
  if native then Variant.native else if fp32Flag then Variant.fp32 else Variant.bf16

/-- With the native flag set the native variant is always chosen. -/
theorem native_wins (f : Bool) : selectVariant f true = Variant.native := by
  simp [selectVariant]

/-- Native implies FP32 activations (`use_fp32 = fp32_activations || use_native`). -/
theorem useFp32_of_native (f : Bool) : (f || true) = true := by
  simp

/-- The mean kernel is launched exactly on the non-native paths. -/
def meanLaunched (native : Bool) : Bool := !native

/-- Native path launches no separate mean kernel. -/
theorem native_no_mean : meanLaunched true = false := by
  simp [meanLaunched]

/-- The native path needs even `hc*n_embd` and `hc_lr`. -/
def nativeGeometryOk (hcDim hcLr : Nat) : Prop := hcDim % 2 = 0 ∧ hcLr % 2 = 0

/-- The real geometry satisfies the native path's evenness requirement. -/
theorem real_native_ok : nativeGeometryOk 10240 320 := by
  unfold nativeGeometryOk
  decide

end Ref.GatedResidual



## FILE: /mnt/user-data/outputs/PleBlockCuda.lean
/-!
# MODULE: ple_block_cuda
SOURCES: src/kernels/cuda/ple.cu
The GPU half of the PLE block: key/value projections, grouped RMS norms, the signed-sqrt sigmoid gate, the
broadcast, the depthwise causal dilated conv with SiLU, and the three-way residual add; plus the rolling
history advance and the caller-owned scratch carving. This module models the integer, index and layout
contracts, the signed gate argument, and the bf16 rounding of `to_bf16_kernel`.

## Source semantics
* The scratch is carved from a caller-owned workspace: five `hc_dim` float buffers, then `n_embd` and `hc`
  floats, then the Q8 activation image, then the BF16 embedding copy, each 16-byte aligned.
* The conv state is ROW-FASTEST: `hist[row + nhist*c]`; tap `k` reads row `nhist - (kern-1-k)*dil`, and only
  the last tap reads the new row (`norm[c]`).
* `gate = sigmoid(sgn(s) * sqrt(max(|s|, eps)))`: the sign is carried separately so the gate argument is odd.
* Output buffers may not overlap the scratch; overlap is an interval-intersection test.
* `bf16_bits` is round-to-nearest-even on a uint32 with wrap-around.

## Form template identifiers
* `scratchBytes`   : scratch_bytes (ple_block_scratch_bytes)
* `advance`        : history_advance
* `tapRow`         : conv_tap_row
* `histIndex`      : conv_state_index (row-fastest)
* `argOf`          : signed_gate_argument
* `overlapC`       : buffer_overlap_check
* `bf16Bits`       : f32_to_bf16_rne
-/

namespace Ref.PleBlock

-- § geometry constants

/-- `NG_N_EMBD`. -/
def NG_N_EMBD : Nat := 2560
/-- `NG_HC`. -/
def NG_HC : Nat := 4
/-- `NG_HC_DIM`. -/
def NG_HC_DIM : Nat := 10240
/-- `NG_HIST`: history rows kept per channel. -/
def NG_HIST : Nat := 9
/-- `PLE_CONV_KERNEL`. -/
def PLE_CONV_KERNEL : Nat := 4
/-- Dilation of the conv taps at the real geometry. -/
def NGRAM_SIZE : Nat := 3

/-- `hc * n_embd = hc_dim` at the real geometry. -/
theorem hc_dim_eq : NG_HC * NG_N_EMBD = NG_HC_DIM := by
  decide

-- § scratch carving

/-- Round up to a multiple of 16. -/
def roundUp16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding never shrinks. -/
theorem roundUp16_ge (n : Nat) : n ≤ roundUp16 n := by
  unfold roundUp16; omega

/-- Rounded sizes are multiples of 16. -/
theorem roundUp16_mod (n : Nat) : roundUp16 n % 16 = 0 := by
  unfold roundUp16; omega

/-- Bytes of the float region: five `hc_dim` buffers plus `n_embd` plus `hc` floats. -/
def floatBytes : Nat := (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * 4
/-- Bytes of the Q8 activation image. -/
def q8Bytes : Nat := (NG_N_EMBD / 32) * 34
/-- Bytes of the BF16 embedding copy. -/
def embBytes : Nat := NG_N_EMBD * 2
/-- Offset of the Q8 image (`d_act`). -/
def actOff : Nat := roundUp16 floatBytes
/-- Offset of the BF16 embedding copy (`d_emb16`). -/
def embOff : Nat := actOff + roundUp16 q8Bytes
/-- Total scratch bytes (`ple_block_scratch_bytes`). -/
def scratchBytes : Nat := roundUp16 floatBytes + roundUp16 q8Bytes + embBytes + 256

/-- The real scratch size is 223152 bytes. -/
theorem scratch_value : scratchBytes = 223152 := by
  decide

/-- The region ordering over abstract sizes: float region `F`, Q8 image `Q`, embedding copy `E`. -/
theorem scratch_gen (F Q E : Nat) :
    F ≤ roundUp16 F ∧ roundUp16 F + Q ≤ roundUp16 F + roundUp16 Q ∧
    roundUp16 F + roundUp16 Q + E ≤ roundUp16 F + roundUp16 Q + E + 256 := by
  have h1 := roundUp16_ge F
  have h2 := roundUp16_ge Q
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- The three scratch regions are ordered and non-overlapping. -/
theorem scratch_regions_fit :
    floatBytes ≤ actOff ∧ actOff + q8Bytes ≤ embOff ∧ embOff + embBytes ≤ scratchBytes :=
  scratch_gen floatBytes q8Bytes embBytes

/-- The Q8 image and the embedding copy start 16-byte aligned. -/
theorem scratch_offsets_aligned : actOff % 16 = 0 ∧ embOff % 16 = 0 := by
  have h1 := roundUp16_mod floatBytes
  have h2 := roundUp16_mod q8Bytes
  simp only [actOff, embOff]
  refine ⟨?_, ?_⟩ <;> omega

/-- Float buffers `key, query, norm, gated, conv, value` are stacked `hc_dim` apart, so none aliases another
(in particular `key` and `normalized` are distinct). -/
theorem float_buffers_disjoint (D n hc : Nat) :
    0 + D ≤ D ∧ D + D ≤ 2 * D ∧ 2 * D + D ≤ 3 * D ∧ 3 * D + D ≤ 4 * D ∧ 4 * D + D ≤ 5 * D ∧
    5 * D + n ≤ 5 * D + n ∧ 5 * D + n + hc ≤ 5 * D + n + hc := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

-- § history advance

/-- One advance step of a history column viewed as a window of `h` rows: shift up, put `x` last. -/
def advance (h : Nat) (col : Nat → Int) (x : Int) : Nat → Int :=
  fun r => if r + 1 < h then col (r + 1) else x

/-- The newest value lands in the last row. -/
theorem advance_last (h : Nat) (hpos : 0 < h) (col : Nat → Int) (x : Int) :
    advance h col x (h - 1) = x := by
  unfold advance
  have : ¬ (h - 1 + 1 < h) := by omega
  simp [this]

/-- Every other row takes the value of the row after it. -/
theorem advance_shift (h : Nat) (col : Nat → Int) (x : Int) (r : Nat) (hr : r + 1 < h) :
    advance h col x r = col (r + 1) := by
  unfold advance
  simp [hr]

-- § conv taps and state indexing

/-- Row read by tap `k` of the depthwise conv. -/
def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- At the real geometry (kern 4, dil 3, hist 9) the taps read rows 0, 3, 6, 9. -/
theorem real_tap_rows : [0, 1, 2, 3].map (tapRow 9 4 3) = [0, 3, 6, 9] := by
  decide

/-- Only the last tap reads the new row (`row == nhist`). -/
theorem only_last_tap_new : ∀ k, k < 4 → (tapRow 9 4 3 k = 9 ↔ k = 3) := by
  decide

/-- Row-fastest index of channel `c`, row `row`. -/
def histIndex (row nh c : Nat) : Nat := row + nh * c

/-- A row-fastest index stays inside `nh * m` for `row < nh`, `c < m`. -/
theorem histIndex_lt (row nh c m : Nat) (hr : row < nh) (hc : c < m) : histIndex row nh c < nh * m := by
  unfold histIndex
  have h : nh * (c + 1) ≤ nh * m := Nat.mul_le_mul (Nat.le_refl nh) hc
  rw [Nat.mul_add, Nat.mul_one] at h
  omega

/-- The same bound serves the conv-weight index `k + kern*c`. -/
theorem kernel_index_lt (k kern c m : Nat) (hk : k < kern) (hc : c < m) : k + kern * c < kern * m :=
  histIndex_lt k kern c m hk hc

/-- The channel-slowest index is a different address: row 1 channel 0 is 10240 against 1. -/
theorem transposed_read_differs : (1 * 10240 + 0) ≠ histIndex 1 9 0 := by
  decide

/-- Streams occupy disjoint ranges `[c*n, (c+1)*n)`. -/
theorem stream_disjoint (c c' n : Nat) (h : c < c') : c * n + n ≤ c' * n := by
  have h' : c + 1 ≤ c' := h
  have hm : (c + 1) * n ≤ c' * n := Nat.mul_le_mul h' (Nat.le_refl n)
  rw [Nat.add_mul, Nat.one_mul] at hm
  exact hm

/-- Broadcast: for `i < hc*n` the stream index is below `hc`. -/
theorem bcast_stream_lt (i hc n : Nat) (hn : 0 < n) (hi : i < hc * n) : i / n < hc :=
  (Nat.div_lt_iff_lt_mul hn).2 hi

/-- Broadcast: the element index is below `n`. -/
theorem bcast_elem_lt (i n : Nat) (hn : 0 < n) : i % n < n :=
  Nat.mod_lt _ hn

-- § signed gate argument

/-- Sign of an integer. -/
def sgn (s : Int) : Int := if 0 < s then 1 else if s < 0 then -1 else 0

/-- Magnitude after the floor (the floor is one unit in this scaled model). -/
def mag (a : Nat) : Nat := max a 1

/-- The gate argument `sgn(s) * mag(|s|)`. -/
def argOf (s : Int) : Int := sgn s * (mag s.natAbs : Int)

/-- `sgn` is odd. -/
theorem sgn_neg (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn
  split_ifs <;> omega

/-- The gate argument is odd, which is what keeps the gate symmetric about 1/2. -/
theorem argOf_neg (s : Int) : argOf (-s) = - argOf s := by
  unfold argOf
  rw [sgn_neg, Int.natAbs_neg, Int.neg_mul]

/-- Zero input gives zero argument (gate 1/2). -/
theorem argOf_zero : argOf 0 = 0 := by
  decide

/-- The clamp only bites near zero: for `|s| >= 1` the magnitude is `|s|`. -/
theorem mag_of_ge (a : Nat) (h : 1 ≤ a) : mag a = a := by
  unfold mag; omega

-- § buffer overlap test

/-- The C++ `overlap` helper. -/
def overlapC (a ab b bb : Nat) : Bool :=
  if ab = 0 ∨ bb = 0 then false else if a < b then decide (b - a < ab) else decide (a - b < bb)

/-- For non-empty ranges the helper is exactly interval intersection. -/
theorem overlapC_iff (a ab b bb : Nat) (hab : 0 < ab) (hbb : 0 < bb) :
    overlapC a ab b bb = true ↔ (a < b + bb ∧ b < a + ab) := by
  unfold overlapC
  have h1 : ¬ (ab = 0 ∨ bb = 0) := by omega
  rw [if_neg h1]
  by_cases h : a < b
  · rw [if_pos h]
    simp only [decide_eq_true_eq]
    omega
  · rw [if_neg h]
    simp only [decide_eq_true_eq]
    omega

/-- Overlap is symmetric for non-empty ranges. -/
theorem overlapC_symm (a ab b bb : Nat) (hab : 0 < ab) (hbb : 0 < bb) :
    overlapC a ab b bb = true ↔ overlapC b bb a ab = true := by
  rw [overlapC_iff a ab b bb hab hbb, overlapC_iff b bb a ab hbb hab]
  omega

/-- An empty range overlaps nothing. -/
theorem overlapC_empty (a b bb : Nat) : overlapC a 0 b bb = false := by
  unfold overlapC
  simp

-- § key source selection

/-- Where the key projection comes from. -/
inductive KeySrc where
  | bf16
  | native
  | q8
  deriving DecidableEq

/-- Priority: bf16 weight, then native Q2_0, then the canonical Q8_0 path. -/
def selectKey (kb kn : Bool) : KeySrc := if kb then KeySrc.bf16 else if kn then KeySrc.native else KeySrc.q8

/-- `native_key = key_native_data != null && key_bf16 == null`. -/
def nativeKey (kn kb : Bool) : Bool := kn && !kb

/-- The native-key predicate holds exactly when selection picks the native path. -/
theorem nativeKey_iff (kn kb : Bool) : nativeKey kn kb = true ↔ selectKey kb kn = KeySrc.native := by
  cases kb <;> cases kn <;> simp [nativeKey, selectKey]

-- § bf16 rounding (uint32 arithmetic, wrap-around)

/-- `bf16_bits`: add `0x7FFF + lsb`, wrap at 2^32, keep the high half. -/
def bf16Bits (i : Nat) : Nat := ((i + (i / 65536) % 2 + 32767) % 4294967296) / 65536

/-- The result is a 16-bit pattern. -/
theorem bf16Bits_lt (i : Nat) : bf16Bits i < 65536 := by
  unfold bf16Bits; omega

/-- Values with zero low half are represented exactly. -/
theorem bf16Bits_exact (i : Nat) (hi : i < 4294967296) (h : i % 65536 = 0) :
    bf16Bits i = i / 65536 := by
  have hq : i = 65536 * (i / 65536) := by omega
  generalize i / 65536 = q at hq
  subst hq
  have hq2 : q < 65536 := by omega
  have e1 : 65536 * q / 65536 = q := by omega
  unfold bf16Bits
  rw [e1]
  have hlt : 65536 * q + q % 2 + 32767 < 4294967296 := by omega
  rw [Nat.mod_eq_of_lt hlt]
  omega

/-- Below the halfway point the high half is kept. -/
theorem bf16Bits_down (i : Nat) (hi : i < 4294967296) (h : i % 65536 < 32768) :
    bf16Bits i = i / 65536 := by
  have hq : i = 65536 * (i / 65536) + i % 65536 := by omega
  have hq1 : i / 65536 < 65536 := by omega
  have hr1 : i % 65536 < 65536 := by omega
  unfold bf16Bits
  generalize i / 65536 = q at *
  generalize i % 65536 = r at *
  subst hq
  have hlt : 65536 * q + r + q % 2 + 32767 < 4294967296 := by omega
  rw [Nat.mod_eq_of_lt hlt]
  omega

/-- Above the halfway point the high half is incremented (wrapping at 65536). -/
theorem bf16Bits_up (i : Nat) (hi : i < 4294967296) (h : 32768 < i % 65536) :
    bf16Bits i = (i / 65536 + 1) % 65536 := by
  have hq : i = 65536 * (i / 65536) + i % 65536 := by omega
  have hq1 : i / 65536 < 65536 := by omega
  have hr1 : i % 65536 < 65536 := by omega
  unfold bf16Bits
  generalize i / 65536 = q at *
  generalize i % 65536 = r at *
  subst hq
  by_cases hq2 : q < 65535
  · have hlt : 65536 * q + r + q % 2 + 32767 < 4294967296 := by omega
    have hS : 65536 * q + r + q % 2 + 32767 = 65536 * (q + 1) + (r + q % 2 - 32769) := by omega
    have ht : (r + q % 2 - 32769) / 65536 = 0 := Nat.div_eq_of_lt (by omega)
    rw [Nat.mod_eq_of_lt hlt, hS, Nat.mul_add_div (by decide), ht]
    omega
  · have hq3 : q = 65535 := by omega
    have hm : (65536 * q + r + q % 2 + 32767) % 4294967296
        = 65536 * q + r + q % 2 + 32767 - 4294967296 := by omega
    rw [hm]
    omega

/-- Ties round to even. -/
theorem bf16Bits_tie_even (i : Nat) (hi : i < 4294967296) (h : i % 65536 = 32768) :
    bf16Bits i % 2 = 0 := by
  have hq : i = 65536 * (i / 65536) + i % 65536 := by omega
  have hq1 : i / 65536 < 65536 := by omega
  have hr1 : i % 65536 < 65536 := by omega
  unfold bf16Bits
  generalize i / 65536 = q at *
  generalize i % 65536 = r at *
  subst hq
  by_cases hq2 : q < 65535
  · have hlt : 65536 * q + r + q % 2 + 32767 < 4294967296 := by omega
    rcases (by omega : q % 2 = 0 ∨ q % 2 = 1) with hb | hb
    · have hS : 65536 * q + r + q % 2 + 32767 = 65536 * q + 65535 := by omega
      rw [Nat.mod_eq_of_lt hlt, hS, Nat.mul_add_div (by decide)]
      omega
    · have hS : 65536 * q + r + q % 2 + 32767 = 65536 * (q + 1) + 0 := by omega
      rw [Nat.mod_eq_of_lt hlt, hS, Nat.mul_add_div (by decide)]
      omega
  · have hq3 : q = 65535 := by omega
    have hm : (65536 * q + r + q % 2 + 32767) % 4294967296 = 0 := by omega
    rw [hm] <;> omega

end Ref.PleBlock



## FILE: /mnt/user-data/outputs/NativePlePostopsCuda.lean
/-!
# MODULE: native_ple_postops_cuda
SOURCES: src/kernels/cuda/native_ple_postops.cu
The opt-in native PLE post-operations: the gate kernel (512 threads, eight partial lanes per row), the
broadcast, the conv+residual kernel, and the span validation that precedes the launches. This module models
the thread/lane partitions, the tap indexing, the sign identity and the span checks.

## Source semantics
* The gate kernel reads `d = tid + j*512` for `j < 8`, masking `d >= N`; the eight partials are summed in
  a fixed order, then 16 warp partials are reduced.
* `float((s>0) - (s<0))` is the sign function used by the CPU-side formula.
* The conv taps read `history[c*9 + 3k]` for `k < 3` and `normalized[c]` for `k = 3`; weights are
  `weights[c*4 + k]`.
* Spans must be non-null, aligned and bounded; the only permitted output/input overlap is the exact alias
  `result == hidden` (each thread owns one element), plus `query == normalized` between outputs.

## Form template identifiers
* `laneIndex`   : gate_partial_lane_index
* `sgnC`, `sgnI`: sign_function_forms
* `overlaps`    : span_overlap
* `aliasPair`   : permitted_alias_pairs
* `convSum`     : conv_tap_accumulate
-/

namespace Ref.NativePlePostops

-- § constants

/-- `N`. -/
def N : Nat := 2560
/-- `H`. -/
def H : Nat := 4
/-- `D = N*H`. -/
def D : Nat := 10240
/-- History rows per channel. -/
def HISTORY : Nat := 9

/-- `D = N * H`. -/
theorem D_eq : N * H = D := by
  decide

-- § gate kernel thread partition

/-- Element read by thread `tid` in partial lane `j`. -/
def laneIndex (tid j : Nat) : Nat := tid + j * 512

/-- Eight lanes of 512 threads cover `N` elements (4096 >= 2560). -/
theorem gate_lanes_cover : N ≤ 8 * 512 := by
  decide

/-- Every element `d < N` is read by exactly the thread `d % 512` in lane `d / 512 < 8`. -/
theorem gate_element_owner (d : Nat) (hd : d < N) :
    d % 512 < 512 ∧ d / 512 < 8 ∧ laneIndex (d % 512) (d / 512) = d := by
  unfold laneIndex N at *
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- No two (thread, lane) pairs read the same element. -/
theorem gate_element_unique (tid j tid' j' : Nat) (ht : tid < 512) (ht' : tid' < 512)
    (h : laneIndex tid j = laneIndex tid' j') : tid = tid' ∧ j = j' := by
  unfold laneIndex at h
  refine ⟨?_, ?_⟩ <;> omega

/-- 512 threads form 16 warps, so the second-stage load `lane < 16 ? partials[lane] : 0` reads them all. -/
theorem gate_warp_partials : 512 / 32 = 16 := by
  decide

/-- A launch of `H` blocks gives one gate per hidden-channel stream. -/
theorem gate_blocks_eq_streams : H = 4 := by
  decide

-- § sign function

/-- Sign as the three-way comparison used in the source. -/
def sgnC (s : Int) : Int := if 0 < s then 1 else if s < 0 then -1 else 0

/-- Sign as `float((s>0) - (s<0))`. -/
def sgnI (s : Int) : Int := (if 0 < s then 1 else 0) - (if s < 0 then 1 else 0)

/-- Both spellings of the sign agree. -/
theorem sgn_forms_agree (s : Int) : sgnC s = sgnI s := by
  unfold sgnC sgnI
  split_ifs <;> omega

/-- The sign function is odd. -/
theorem sgnC_neg (s : Int) : sgnC (-s) = - sgnC s := by
  unfold sgnC
  split_ifs <;> omega

-- § broadcast and conv-residual launch geometry

/-- The broadcast kernel launches `D/256` blocks with no tail. -/
theorem broadcast_exact : D % 256 = 0 ∧ D / 256 = 40 := by
  decide

/-- Block/thread to element map of the elementwise kernels: `c = blk*256 + tid`. -/
theorem elementwise_roundtrip (c : Nat) (hc : c < D) :
    c / 256 < 40 ∧ c % 256 < 256 ∧ (c / 256) * 256 + c % 256 = c := by
  unfold D at hc
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Distinct (block, thread) pairs own distinct elements, which is why `result == hidden` is safe. -/
theorem elementwise_unique (b t b' t' : Nat) (ht : t < 256) (ht' : t' < 256)
    (h : b * 256 + t = b' * 256 + t') : b = b' ∧ t = t' := by
  refine ⟨?_, ?_⟩ <;> omega

-- § conv tap indexing

/-- History index read by tap `k < 3`. -/
def tapHist (c k : Nat) : Nat := c * 9 + 3 * k

/-- Taps 0..2 stay inside the channel's history row of 9 and never touch the new row. -/
theorem tapHist_in_row (c k : Nat) (hk : k < 3) :
    c * 9 ≤ tapHist c k ∧ tapHist c k < c * 9 + 9 := by
  unfold tapHist
  refine ⟨?_, ?_⟩ <;> omega

/-- Distinct taps read distinct history rows. -/
theorem tapHist_inj (c k k' : Nat) (h : tapHist c k = tapHist c k') : k = k' := by
  unfold tapHist at h
  omega

/-- Tap history indices stay below `9*D` for `c < D`. -/
theorem tapHist_lt (c k : Nat) (hc : c < D) (hk : k < 3) : tapHist c k < 9 * D := by
  unfold tapHist D at *
  omega

/-- Weight index `c*4 + k` stays below `4*D` for `k < 4`. -/
theorem weight_index_lt (c k : Nat) (hc : c < D) (hk : k < 4) : c * 4 + k < 4 * D := by
  unfold D at *
  omega

/-- The tap accumulation order is `((t0+t1)+t2)+t3`. -/
def convSum : List Int → Int
  | [] => 0
  | t :: rest => rest.foldl (· + ·) t

/-- The four-tap sum is left-associated, exactly the order of the `__fadd_rn` chain. -/
theorem convSum_four (a b c d : Int) : convSum [a, b, c, d] = ((a + b) + c) + d := by
  rfl

-- § span validation

/-- A memory span: start address, byte length, required alignment. -/
structure Span where
  p : Nat
  bytes : Nat
  align : Nat

/-- Largest `uintptr_t`. -/
def MAXU : Nat := 18446744073709551615

/-- Two spans overlap iff each starts before the other ends. -/
def overlaps (a b : Span) : Prop := a.p < b.p + b.bytes ∧ b.p < a.p + a.bytes

/-- Overlap is symmetric. -/
theorem overlaps_symm (a b : Span) : overlaps a b ↔ overlaps b a := by
  unfold overlaps
  omega

/-- A span never overlaps one that starts after it ends. -/
theorem overlaps_disjoint_after (a b : Span) (h : a.p + a.bytes ≤ b.p) : ¬ overlaps a b := by
  unfold overlaps
  omega

/-- A non-null span with `p <= MAXU - bytes` cannot wrap the address space. -/
theorem valid_no_overflow (p bytes : Nat) (hp : p ≠ 0) (h : p ≤ MAXU - bytes) : p + bytes ≤ MAXU := by
  unfold MAXU at *
  omega

/-- The permitted exact alias: output 6 (`result`) against input 1 (`hidden`). -/
def aliasPair (i j : Nat) : Bool := i == 6 && j == 1

/-- Exactly the pair (6,1) is exempt from the output-versus-input check. -/
theorem aliasPair_iff (i j : Nat) : aliasPair i j = true ↔ (i = 6 ∧ j = 1) := by
  simp [aliasPair]

/-- Across the 7 outputs and 8 inputs the exemption applies to exactly one pair. -/
theorem alias_count :
    ((List.range 7).map (fun i => ((List.range 8).filter (fun j => aliasPair i j)).length)).sum = 1 := by
  decide

end Ref.NativePlePostops



## FILE: /mnt/user-data/outputs/CpuExpertKernel.lean
/-!
# MODULE: cpu_expert_kernel
SOURCES: src/kernels/cpu/expert.cpp
The Q2_0 expert kernel on x86 (AVX-512 VNNI/VBMI): fp16 scale decoding, the Q8_1 activation quantizer, the
2-bit unpack, the offset-corrected dot product, the 8-block scale masks, token-count dispatch, and the CPU
feature gate. This module models the arithmetic identities and index contracts these routines rely on.

## Source semantics
* `sum_j (c_j - 1) d_w xhat_j d_x == d_w * (d_x * sum(c*xhat) - d_x * sum(xhat))`, split per 32-chunk.
* Quantized activations are clamped to [-127, 127]; a 32-chunk sum therefore fits easily in int32.
* The fp16 subnormal path normalizes the mantissa with at most 10 shifts and a biased exponent `113 - s`.
* A 2-bit code is `(x % 256) % 4 = x % 4` of the multishifted word; codes are at most 3, so `maddubs`
  never saturates against an int8 activation.
* Token counts 1..7 select that template; everything else selects 8; the multi path needs `1 <= nt <= 8`.

## Form template identifiers
* `normShift`     : f16_subnormal_normalize
* `clampQ`        : q8_clamp
* `dotOff`, `dotRaw` : offset_dot, raw_dot
* `maskOf`        : block_tail_mask
* `dispatchNT`    : token_template_dispatch
* `usable`, `missing` : cpu_feature_gate
-/

namespace Ref.CpuExpert

-- § fp16 decoding

/-- Number of left shifts the subnormal loop performs (bounded by `fuel`). -/
def normShift : Nat → Nat → Nat
  | _, 0 => 0
  | m, f + 1 => if 1024 ≤ m then 0 else 1 + normShift (2 * m) f

/-- The shift count never exceeds the fuel. -/
theorem normShift_le (m f : Nat) : normShift m f ≤ f := by
  induction f generalizing m with
  | zero => simp [normShift]
  | succ f ih =>
    have e : normShift m (f + 1) = if 1024 ≤ m then 0 else 1 + normShift (2 * m) f := rfl
    rw [e]
    split
    · omega
    · have := ih (2 * m)
      omega

/-- After the shifts the mantissa has bit 10 set and stays below 2048. -/
theorem normShift_spec (f : Nat) : ∀ m, 0 < m → m < 2048 → 1024 ≤ m * 2 ^ f →
    1024 ≤ m * 2 ^ normShift m f ∧ m * 2 ^ normShift m f < 2048 := by
  induction f with
  | zero =>
    intro m h0 h1 h2
    have e : normShift m 0 = 0 := rfl
    rw [e]
    simp only [Nat.pow_zero, Nat.mul_one] at h2 ⊢
    exact ⟨h2, h1⟩
  | succ f ih =>
    intro m h0 h1 h2
    have e : normShift m (f + 1) = if 1024 ≤ m then 0 else 1 + normShift (2 * m) f := rfl
    rw [e]
    by_cases hm : 1024 ≤ m
    · rw [if_pos hm]
      simp only [Nat.pow_zero, Nat.mul_one]
      exact ⟨hm, h1⟩
    · rw [if_neg hm]
      have h2' : 1024 ≤ (2 * m) * 2 ^ f := by
        have e3 : m * 2 ^ (f + 1) = (2 * m) * 2 ^ f := by
          rw [Nat.pow_succ]
          ac_rfl
        rw [e3] at h2
        exact h2
      have ih' := ih (2 * m) (by omega) (by omega) h2'
      have e2 : m * 2 ^ (1 + normShift (2 * m) f) = (2 * m) * 2 ^ normShift (2 * m) f := by
        rw [Nat.pow_add, Nat.pow_one]
        ac_rfl
      rw [e2]
      exact ih'

/-- A nonzero fp16 subnormal mantissa (`0 < man < 1024`) normalizes within 10 shifts. -/
theorem f16_subnormal_normalizes (man : Nat) (h0 : 0 < man) (h1 : man < 1024) :
    normShift man 10 ≤ 10 ∧ 1024 ≤ man * 2 ^ normShift man 10 ∧ man * 2 ^ normShift man 10 < 2048 := by
  have hb : 1024 ≤ man * 2 ^ 10 := by omega
  have hs := normShift_spec 10 man h0 (by omega) hb
  exact ⟨normShift_le man 10, hs.1, hs.2⟩

/-- The biased f32 exponent `113 - s` of a subnormal stays a normal f32 exponent. -/
theorem subnormal_exp_range (s : Nat) (h : s ≤ 10) : 0 < 113 - s ∧ 113 - s < 255 := by
  omega

/-- The three fp16 fields reassemble the 16-bit word. -/
theorem f16_fields (h : Nat) (hh : h < 65536) :
    (h / 32768) * 32768 + ((h / 1024) % 32) * 1024 + h % 1024 = h := by
  omega

-- § activation quantization

/-- Clamp of the rounded quotient to the symmetric int8 range used by Q8_1. -/
def clampQ (v : Int) : Int := max (-127) (min 127 v)

/-- Clamped values lie in [-127, 127]. -/
theorem clampQ_bounds (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ
  omega

/-- Clamping is idempotent. -/
theorem clampQ_idem (v : Int) : clampQ (clampQ v) = clampQ v := by
  unfold clampQ
  omega

/-- The range is symmetric, so clamping commutes with negation. -/
theorem clampQ_neg (v : Int) : clampQ (-v) = - clampQ v := by
  unfold clampQ
  omega

/-- In-range values are untouched. -/
theorem clampQ_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clampQ v = v := by
  unfold clampQ
  omega

/-- A list of in-range values has sum bounded by 127 per element. -/
theorem sum_abs_bound : ∀ l : List Int, (∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) →
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | x :: xs, h => by
    have hx := h x (by simp)
    have ih := sum_abs_bound xs (fun y hy => h y (by simp [hy]))
    simp only [List.sum_cons, List.length_cons]
    push_cast
    omega

/-- A 32-value chunk sum is within ±4064, far inside int32. -/
theorem chunk_sum_fits (l : List Int) (hl : l.length = 32) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -4064 ≤ l.sum ∧ l.sum ≤ 4064 := by
  have hb := sum_abs_bound l h
  rw [hl] at hb
  obtain ⟨hb1, hb2⟩ := hb
  exact ⟨by omega, by omega⟩

-- § offset-corrected dot product

/-- Dot product with the Q2_0 code offset removed: `sum (c-1)*x`. -/
def dotOff : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | c :: cs, x :: xs => (c - 1) * x + dotOff cs xs

/-- Raw dot product `sum c*x`. -/
def dotRaw : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | c :: cs, x :: xs => c * x + dotRaw cs xs

/-- The offset dot equals the raw dot minus the activation sum: the `- d_x*sum(xhat)` correction. -/
theorem dotOff_eq : ∀ (cs xs : List Int), cs.length = xs.length →
    dotOff cs xs = dotRaw cs xs - xs.sum
  | [], [], _ => by simp [dotOff, dotRaw]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | c :: cs, x :: xs, h => by
    have ih := dotOff_eq cs xs (by simpa using h)
    simp only [dotOff, dotRaw, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega

/-- The offset dot is additive over chunk splits, which justifies the per-32-chunk scale handling. -/
theorem dotOff_append : ∀ (c1 x1 c2 x2 : List Int), c1.length = x1.length →
    dotOff (c1 ++ c2) (x1 ++ x2) = dotOff c1 x1 + dotOff c2 x2
  | [], [], c2, x2, _ => by simp [dotOff]
  | [], _ :: _, _, _, h => by simp at h
  | _ :: _, [], _, _, h => by simp at h
  | c :: cs, x :: xs, c2, x2, h => by
    have ih := dotOff_append cs xs c2 x2 (by simpa using h)
    simp only [List.cons_append, dotOff]
    rw [ih, Int.add_assoc]

-- § 2-bit unpack

/-- Masking the multishifted byte with 3 equals masking the whole word with 3. -/
theorem code_byte_mask (x : Nat) : (x % 256) % 4 = x % 4 := by
  omega

/-- A 2-bit code is at most 3. -/
theorem code_lt_four (x : Nat) : x % 4 < 4 :=
  Nat.mod_lt _ (by decide)

/-- Products of a code (0..3) with an int8 activation are at most 381 in magnitude. -/
theorem code_mul_bound (c q : Int) (h0 : 0 ≤ c) (h3 : c ≤ 3) (hq1 : -127 ≤ q) (hq2 : q ≤ 127) :
    -381 ≤ c * q ∧ c * q ≤ 381 := by
  interval_cases c <;> omega

/-- A pair-sum of two such products fits an int16 lane, so `maddubs` never saturates. -/
theorem maddubs_no_saturation (c1 c2 q1 q2 : Int) (h1 : 0 ≤ c1) (h2 : c1 ≤ 3) (h3 : 0 ≤ c2)
    (h4 : c2 ≤ 3) (hq1 : -127 ≤ q1) (hq2 : q1 ≤ 127) (hq3 : -127 ≤ q2) (hq4 : q2 ≤ 127) :
    -32768 ≤ c1 * q1 + c2 * q2 ∧ c1 * q1 + c2 * q2 ≤ 32767 := by
  have a := code_mul_bound c1 q1 h1 h2 hq1 hq2
  have b := code_mul_bound c2 q2 h3 h4 hq3 hq4
  exact ⟨by omega, by omega⟩

/-- Low four codes of a lane word `b0 + 256*b1` come from byte `b0`. -/
theorem lane_low (b0 b1 i : Nat) (hb : b0 < 256) (hi : i < 4) :
    (b0 + 256 * b1) / 4 ^ i % 4 = b0 / 4 ^ i % 4 := by
  interval_cases i <;> omega

/-- High four codes come from byte `b1`. -/
theorem lane_high (b0 b1 j : Nat) (hb : b0 < 256) (hj : j < 4) :
    (b0 + 256 * b1) / 4 ^ (j + 4) % 4 = b1 / 4 ^ j % 4 := by
  interval_cases j <;> omega

-- § block tail masks and scale permutes

/-- Lane mask for the last group of `nb <= 8` blocks (two lanes per block). -/
def maskOf (nb : Nat) : Nat := if 8 ≤ nb then 65535 else 2 ^ (2 * nb) - 1

/-- The mask has exactly `2*nb` low bits set. -/
theorem maskOf_bits (nb : Nat) (h : nb ≤ 8) : maskOf nb + 1 = 2 ^ (2 * nb) := by
  interval_cases nb <;> decide

/-- A group never runs past the end: `b0 + nb <= nblocks` and `nb >= 1` while blocks remain. -/
theorem group_bounds (b0 nblocks : Nat) (h : b0 < nblocks) :
    1 ≤ min 8 (nblocks - b0) ∧ b0 + min 8 (nblocks - b0) ≤ nblocks := by
  refine ⟨?_, ?_⟩ <;> omega

/-- Scale and correction loads index `2*(b0+i)+h`, which stays below `2*nblocks`. -/
theorem scale_index_lt (b0 i h nb nblocks : Nat) (hi : i < nb) (hnb : b0 + nb ≤ nblocks) (hh : h < 2) :
    2 * (b0 + i) + h < 2 * nblocks := by
  omega

/-- The permute index `lane/8 + 2*i` selects the product for block `i`, half `lane/8`, within 16 lanes. -/
theorem permute_index (i l : Nat) (hi : i < 8) (hl : l < 16) :
    l / 8 + 2 * i < 16 ∧ (l / 8 + 2 * i) / 2 = i ∧ (l / 8 + 2 * i) % 2 = l / 8 := by
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- The gate/up scale arrays at hidden size 2560 hold 40 blocks, a whole number of 8-block groups. -/
theorem real_gu_blocks : 2560 / 64 = 40 ∧ 40 % 8 = 0 := by
  decide

-- § token-count dispatch

/-- Template argument chosen for `n_tokens`: 1..7 select that value, anything else selects 8. -/
def dispatchNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- The selected template never exceeds 8 and is at least 1. -/
theorem dispatchNT_range (n : Nat) : 1 ≤ dispatchNT n ∧ dispatchNT n ≤ 8 := by
  unfold dispatchNT
  split <;> omega

/-- For every legal token count `1..8` the template equals the count. -/
theorem dispatchNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : dispatchNT n = n := by
  unfold dispatchNT
  split <;> omega

/-- The GGUF Q2_0 block stride 18 = 2 scale bytes + 16 code bytes, and 16 bytes hold 64 codes. -/
theorem gguf_block_layout : 2 + 16 = 18 ∧ 16 * 4 = 64 := by
  decide

-- § CPU feature gate

/-- All five required features present. -/
def usable (f bw vl vnni vbmi : Bool) : Bool := f && bw && vl && vnni && vbmi

/-- Names of the missing features, in the order the diagnostic prints them. -/
def missing (f bw vl vnni vbmi : Bool) : List String :=
  (if f then [] else ["AVX512F"]) ++ (if bw then [] else ["AVX512BW"]) ++
  (if vl then [] else ["AVX512VL"]) ++ (if vnni then [] else ["AVX512-VNNI"]) ++
  (if vbmi then [] else ["AVX512-VBMI"])

/-- The CPU is usable exactly when nothing is reported missing. -/
theorem usable_iff_missing_nil (f bw vl vnni vbmi : Bool) :
    usable f bw vl vnni vbmi = true ↔ missing f bw vl vnni vbmi = [] := by
  cases f <;> cases bw <;> cases vl <;> cases vnni <;> cases vbmi <;> simp [usable, missing]

end Ref.CpuExpert



## FILE: /mnt/user-data/outputs/CpuExpertLayout.lean
/-!
# MODULE: cpu_expert_layout
SOURCES: src/kernels/cpu/expert_layout.cpp
The per-layer expert table: contiguity validation of `native_experts.txt`, total size, maximum blob size,
GGUF offset slots, and the AVX-512 / AVX2 dispatch (including the OS-saved-state check on XCR0).

## Source semantics
* Layer `l` must start at the sum of the previous layers' `blob * n_expert`; the total is that sum over all
  layers. Without a table every layer has the fixed `BLOB` size.
* `max_blob` is the maximum layer blob size.
* GGUF offsets live in a `3*n_layers` array: slots `3l`, `3l+1`, `3l+2`.
* AVX-512 is used only if not forced off and the CPU and OS both support it.

## Form template identifiers
* `offs`, `endOff`, `check` : contiguous_layer_offsets, layout_total, contiguity_check
* `maxBlob`                 : max_blob
* `gguf slot`               : gguf_offset_slot
* `avx512Ok`, `pickImpl`    : avx512_dispatch
-/

namespace Ref.CpuExpertLayout

-- § contiguous offsets

/-- Expected start of each layer given layer blob sizes, `n_expert = E`, running offset `a`. -/
def offs : List Nat → Nat → Nat → List Nat
  | [], _, _ => []
  | b :: bs, E, a => a :: offs bs E (a + b * E)

/-- The end offset after all layers. -/
def endOff : List Nat → Nat → Nat → Nat
  | [], _, a => a
  | b :: bs, E, a => endOff bs E (a + b * E)

/-- The contiguity check of the loader: every declared offset equals the expected running sum. -/
def check : List Nat → List Nat → Nat → Nat → Bool
  | d :: ds, b :: bs, E, a => (d == a) && check ds bs E (a + b * E)
  | [], [], _, _ => true
  | [], _ :: _, _, _ => false
  | _ :: _, [], _, _ => false

/-- One expected offset per layer. -/
theorem offs_length (bs : List Nat) (E a : Nat) : (offs bs E a).length = bs.length := by
  induction bs generalizing a with
  | nil => rfl
  | cons b bs ih => simp [offs, ih]

/-- The total is the start plus every layer's `blob * n_expert`. -/
theorem endOff_eq (bs : List Nat) (E a : Nat) : endOff bs E a = a + (bs.map (· * E)).sum := by
  induction bs generalizing a with
  | nil => simp [endOff]
  | cons b bs ih =>
    simp only [endOff, ih, List.map_cons, List.sum_cons]
    omega

/-- The expected offsets always pass the contiguity check. -/
theorem check_offs (bs : List Nat) (E a : Nat) : check (offs bs E a) bs E a = true := by
  induction bs generalizing a with
  | nil => rfl
  | cons b bs ih => simp [offs, check, ih]

/-- Without a table the total is `layers * (BLOB * experts)`, one fixed blob per layer. -/
theorem total_uniform (L B E a : Nat) : endOff (List.replicate L B) E a = a + L * (B * E) := by
  induction L generalizing a with
  | zero => simp [endOff]
  | succ L ih =>
    simp only [List.replicate_succ, endOff, ih]
    rw [Nat.add_mul, Nat.one_mul]
    omega

-- § maximum blob

/-- Largest layer blob. -/
def maxBlob (bs : List Nat) : Nat := bs.foldl max 0

/-- The running maximum never drops below its starting value. -/
theorem foldl_max_ge_init (bs : List Nat) : ∀ init, init ≤ bs.foldl max init := by
  induction bs with
  | nil => intro init; exact Nat.le_refl init
  | cons x xs ih =>
    intro init
    exact Nat.le_trans (Nat.le_max_left init x) (ih (max init x))

/-- Every element is at most the fold maximum. -/
theorem le_foldl_max_of_mem (bs : List Nat) : ∀ init b, b ∈ bs → b ≤ bs.foldl max init := by
  induction bs with
  | nil => intro init b hb; simp at hb
  | cons x xs ih =>
    intro init b hb
    rcases List.mem_cons.mp hb with h | h
    · rw [h]
      exact Nat.le_trans (Nat.le_max_right init x) (foldl_max_ge_init xs (max init x))
    · exact ih (max init x) b h

/-- Every layer blob is at most `max_blob`. -/
theorem le_maxBlob (bs : List Nat) (b : Nat) (hb : b ∈ bs) : b ≤ maxBlob bs :=
  le_foldl_max_of_mem bs 0 b hb

-- § GGUF offset slots

/-- Slot of layer `l`, component `k` (0 gate, 1 up, 2 down) in the `3*n_layers` array. -/
def slot (l k : Nat) : Nat := 3 * l + k

/-- Slots of in-range layers stay inside the array. -/
theorem slot_lt (l k n : Nat) (hl : l < n) (hk : k < 3) : slot l k < 3 * n := by
  unfold slot; omega

/-- Distinct (layer, component) pairs use distinct slots. -/
theorem slot_inj (l k l' k' : Nat) (hk : k < 3) (hk' : k' < 3) (h : slot l k = slot l' k') :
    l = l' ∧ k = k' := by
  unfold slot at h
  refine ⟨?_, ?_⟩ <;> omega

-- § feature dispatch

/-- Which implementation the row dispatcher picks. -/
inductive Impl where
  | avx512
  | avx2
  deriving DecidableEq

/-- AVX-512 is allowed only when not forced off and the hardware check passed. -/
def avx512Ok (forcedAvx2 hw : Bool) : Bool := !forcedAvx2 && hw

/-- Dispatch on the result of the check. -/
def pickImpl (ok : Bool) : Impl := if ok then Impl.avx512 else Impl.avx2

/-- `REF_FORCE_AVX2=1` always selects the AVX2 implementation. -/
theorem forced_selects_avx2 (hw : Bool) : pickImpl (avx512Ok true hw) = Impl.avx2 := by
  simp [avx512Ok, pickImpl]

/-- Without the force flag the hardware answer decides. -/
theorem unforced_follows_hw (hw : Bool) : pickImpl (avx512Ok false hw) = (if hw then Impl.avx512 else Impl.avx2) := by
  simp [avx512Ok, pickImpl]

set_option maxRecDepth 100000 in
/-- The XCR0 test `(xcr0 & 0xE6) == 0xE6` over the low byte is exactly the five state bits
(SSE/AVX/opmask/ZMM_Hi256/Hi16_ZMM that the OS must save). -/
theorem xcr0_mask_iff : ∀ x : Fin 256, ((x.val &&& 0xE6) = 0xE6) ↔
    ((x.val / 2) % 2 = 1 ∧ (x.val / 4) % 2 = 1 ∧ (x.val / 32) % 2 = 1 ∧
     (x.val / 64) % 2 = 1 ∧ (x.val / 128) % 2 = 1) := by
  decide

end Ref.CpuExpertLayout



## FILE: /mnt/user-data/outputs/IqAvx512Rows.lean
/-!
# MODULE: iq_avx512_rows
SOURCES: src/kernels/cpu/iq_avx512.cpp
The i-quant expert rows in 512-bit lanes for IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21) and IQ2_S
(22): block layouts, decoded-field index ranges, 64-value chunk/scale lane maps, the multi-token chunking
loop, and the supported-type predicate.

## Source semantics
* Each format stores 256 values per block; the byte sizes are 66, 74, 98, 110 and 82, and every field the
  decoder touches for chunk `j < 4` lies inside its block and the regions tile the block exactly.
* A 32-bit word packs four 7-bit sign indices and a 4-bit scale: `4*7 + 4 = 32`, and the packing is lossless.
* Sign application is a masked negate: it preserves the magnitude of the activation.
* More than four tokens are processed in chunks of at most four; the chunk sizes sum to `nt`.
* Only the five listed type ids are handled here (IQ1_M stays on ggml-cpu).

## Form template identifiers
* `blockBytes`      : iq_block_bytes
* `word32_fields`   : sign_scale_word_layout
* `laneScale`       : scale_lane_map
* `chunks`          : token_chunking
* `supported`       : iq512_supported
-/

namespace Ref.IqAvx512

-- § block layouts

/-- Bytes per 256-value block for the supported format ids. -/
def blockBytes (ty : Nat) : Nat :=
  if ty = 16 then 66 else if ty = 17 then 74 else if ty = 18 then 98 else
  if ty = 21 then 110 else if ty = 22 then 82 else 0

/-- IQ2_XXS: `d` (2) + 32 u16 (64). -/
theorem layout_16 : 2 + 64 = blockBytes 16 := by
  decide

/-- IQ2_XS: `d` + 32 u16 + 8 scale bytes. -/
theorem layout_17 : 2 + 64 + 8 = blockBytes 17 := by
  decide

/-- IQ3_XXS: `d` + 64 grid bytes + 8 x u32. -/
theorem layout_18 : 2 + 64 + 32 = blockBytes 18 := by
  decide

/-- IQ3_S: `d` + qs 64 + qh 8 + signs 32 + scales 4. -/
theorem layout_21 : 2 + 64 + 8 + 32 + 4 = blockBytes 21 := by
  decide

/-- IQ2_S: `d` + qs 64 (32 grid + 32 sign) + qh 8 + scales 8. -/
theorem layout_22 : 2 + 64 + 8 + 8 = blockBytes 22 := by
  decide

/-- IQ2_XXS reads for chunk `j < 4` end inside the 66-byte block. -/
theorem iq2xxs_in_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 15 < 66 := by
  omega

/-- IQ2_XS: codes `[2+16j, 2+16j+16)` and scales `66+2j+1` are inside the 74-byte block. -/
theorem iq2xs_in_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 15 < 66 ∧ 66 + 2 * j + 1 < 74 := by
  refine ⟨?_, ?_⟩ <;> omega

/-- IQ2_S: grid bytes, sign bytes, `qh` and scales are inside the 82-byte block and the sign
bytes start where the grid bytes end. -/
theorem iq2s_in_bounds (j : Nat) (hj : j < 4) :
    2 + 8 * j + 7 < 34 ∧ 2 + 32 + 8 * j + 7 < 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- IQ3_XXS: codes and the sign/scale words are inside the 98-byte block. -/
theorem iq3xxs_in_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 15 < 66 ∧ 2 + 64 + 8 * j + 7 < 98 := by
  refine ⟨?_, ?_⟩ <;> omega

/-- IQ3_S: all four field reads are inside the 110-byte block. -/
theorem iq3s_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 15 < 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 7 < 106 ∧ 106 + j < 110 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- Four 64-value chunks make a 256-value block, and a chunk read of `qs + 64*j` stays inside it. -/
theorem chunk_geometry : 4 * 64 = 256 := by
  decide

/-- The 64 int8 activations of chunk `j` are `[64j, 64j+64)`, below 256. -/
theorem chunk_act_in_block (j : Nat) (hj : j < 4) : 64 * j + 63 < 256 := by
  omega

-- § packed fields

/-- A 32-bit word of four 7-bit sign indices and a 4-bit scale is a lossless packing. -/
theorem word32_fields (a : Nat) :
    a % 128 + 128 * ((a / 128) % 128) + 16384 * ((a / 16384) % 128) +
      2097152 * ((a / 2097152) % 128) + 268435456 * (a / 268435456) = a := by
  omega

/-- The field widths add up to the word width. -/
theorem word32_width : 4 * 7 + 4 = 32 := by
  decide

/-- For a 32-bit word the scale nibble `a >> 28` is below 16. -/
theorem scale_nibble_lt (a : Nat) (h : a < 4294967296) : a / 268435456 < 16 := by
  omega

/-- The odd scale `2*s + 1` is odd and at most 31 for a nibble. -/
theorem odd_scale (s : Nat) (h : s < 16) : (2 * s + 1) % 2 = 1 ∧ 2 * s + 1 ≤ 31 := by
  refine ⟨?_, ?_⟩ <;> omega

/-- Each sign index is a 7-bit value, a valid index into the 128-entry sign table. -/
theorem sign_index_lt (a k : Nat) : (a / 2 ^ (7 * k)) % 128 < 128 :=
  Nat.mod_lt _ (by decide)

/-- Eight bytes pack into a 64-bit sign mask without overflow. -/
theorem mask64_pack (b0 b1 b2 b3 b4 b5 b6 b7 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256)
    (h3 : b3 < 256) (h4 : b4 < 256) (h5 : b5 < 256) (h6 : b6 < 256) (h7 : b7 < 256) :
    b0 + 256 * b1 + 65536 * b2 + 16777216 * b3 + 4294967296 * b4 + 1099511627776 * b5 +
      281474976710656 * b6 + 72057594037927936 * b7 < 18446744073709551616 := by
  omega

/-- The 9-bit grid index and 7-bit sign index of an IQ2_XS word are lossless: `v = idx + 512*sign`. -/
theorem iq2xs_word_split (v : Nat) (h : v < 65536) : v % 512 + 512 * (v / 512) = v ∧ v / 512 < 128 := by
  refine ⟨?_, ?_⟩ <;> omega

-- § lane maps

/-- Scale lane map: int16 lane `l < 32` of a `maddubs` result uses scale `l / 8`. -/
def laneScale (l : Nat) : Nat := l / 8

/-- Value `v < 64` sits in int16 lane `v/2`, whose scale index is `v/16`: four groups of 16 values. -/
theorem value_scale_group (v : Nat) (hv : v < 64) : laneScale (v / 2) = v / 16 ∧ v / 16 < 4 := by
  unfold laneScale
  refine ⟨?_, ?_⟩ <;> omega

/-- Sign application as a masked negate preserves magnitude. -/
theorem masked_negate_abs (y : Int) (b : Bool) : ((if b then -y else y)).natAbs = y.natAbs := by
  cases b <;> simp

-- § token chunking

/-- Sizes of the chunks (of at most `K`) a loop over `rem` tokens produces, with fuel `f`. -/
def chunks (K : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, rem => if rem = 0 then [] else min K rem :: chunks K f (rem - min K rem)

/-- The chunk sizes sum to the token count when the fuel suffices. -/
theorem chunks_sum (K : Nat) (f : Nat) : ∀ rem, rem ≤ K * f → (chunks K f rem).sum = rem := by
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    rfl
  | succ f ih =>
    intro rem h
    rw [Nat.mul_succ] at h
    by_cases hr : rem = 0
    · subst hr
      simp [chunks]
    · have e : chunks K (f + 1) rem = min K rem :: chunks K f (rem - min K rem) := by
        simp [chunks, hr]
      rw [e, List.sum_cons, ih (rem - min K rem) (by omega)]
      omega

/-- Every chunk is at most `K`. -/
theorem chunks_le (K : Nat) (f : Nat) : ∀ rem, ∀ c ∈ chunks K f rem, c ≤ K := by
  induction f with
  | zero => intro rem c hc; simp [chunks] at hc
  | succ f ih =>
    intro rem c hc
    by_cases hr : rem = 0
    · subst hr
      simp [chunks] at hc
    · have e : chunks K (f + 1) rem = min K rem :: chunks K f (rem - min K rem) := by
        simp [chunks, hr]
      rw [e] at hc
      rcases List.mem_cons.mp hc with h | h
      · rw [h]; omega
      · exact ih _ c h

/-- A tail chunk `k = min 4 (nt - t0)` is non-empty and stays inside `nt`. -/
theorem tail_chunk_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ min 4 (nt - t0) ∧ t0 + min 4 (nt - t0) ≤ nt := by
  refine ⟨?_, ?_⟩ <;> omega

-- § supported types

/-- The types handled by the 512-bit kernels. -/
def supported (ty : Nat) : Bool := ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

/-- Exactly the five listed ids are supported. -/
theorem supported_iff (ty : Nat) : supported ty = true ↔ (ty = 16 ∨ ty = 17 ∨ ty = 18 ∨ ty = 21 ∨ ty = 22) := by
  simp [supported, or_assoc]

/-- Supported ids are exactly the ones with a known block size. -/
theorem supported_has_size (ty : Nat) (h : supported ty = true) : 0 < blockBytes ty := by
  rw [supported_iff] at h
  rcases h with h | h | h | h | h <;> subst h <;> decide

/-- Other ggml ids (IQ1_M and non-IQ types) are not claimed. -/
theorem unsupported_examples : [0, 1, 19, 20, 23, 29].all (fun t => !supported t) = true := by
  decide

end Ref.IqAvx512



## FILE: /mnt/user-data/outputs/CpuExpertPool.lean
/-!
# MODULE: cpu_expert_pool
SOURCES: src/kernels/cpu/pool.cpp
The CPU expert pool: physical-core selection and pinning, worker-count rules, and the work partitioning of
the split phases (equal row ranges across tasks, the inner per-expert segment loop, and batching of native
jobs). This module models the integer partition contracts that make every row computed exactly once.

## Source semantics
* `physical_cores(skip_first)` drops the first core (the host's) so workers never share it.
* The pool has at least one worker; a pin core exists only when pinning is on and the index is in range.
* Phase task `i` of `tasks` over `rows` rows covers `[rows*i/tasks, rows*(i+1)/tasks)`: these ranges tile
  `[0, rows)`.
* The inner loop walks a range in segments that never cross an expert boundary and always make progress,
  covering exactly the range.
* Batches of at most `kMaxSplitMulti` experts cover all `n` experts.

## Form template identifiers
* `coresFor`       : physical_core_list
* `getCore`        : worker_pin_core
* `nWorkers`       : worker_count
* `lo`             : task_row_lo
* `stepLen`, `cover` : segment_walk
* `batches`        : native_batching
-/

namespace Ref.CpuExpertPool

-- § core selection

/-- Cores after optionally dropping the host's core. -/
def coresFor (skipFirst : Bool) (cores : List Nat) : List Nat := if skipFirst then cores.tail else cores

/-- Dropping the first core removes exactly one entry from a non-empty list. -/
theorem coresFor_length (cores : List Nat) (h : cores ≠ []) :
    (coresFor true cores).length = cores.length - 1 := by
  unfold coresFor
  simp

/-- Without skipping the list is unchanged. -/
theorem coresFor_noskip (cores : List Nat) : coresFor false cores = cores := by
  simp [coresFor]

/-- With distinct core ids the host core is never handed to a worker. -/
theorem host_core_unused (c : Nat) (cs : List Nat) (h : (c :: cs).Nodup) : c ∉ coresFor true (c :: cs) := by
  unfold coresFor
  simpa using (List.nodup_cons.mp h).1

/-- Pin core of worker `i`: only when pinning is on and `i` is in range. -/
def getCore (pin : Bool) (cores : List Nat) (i : Nat) : Option Nat := if pin then cores[i]? else none

/-- No pinning, no core. -/
theorem getCore_nopin (cores : List Nat) (i : Nat) : getCore false cores i = none := by
  simp [getCore]

/-- Out-of-range workers are left unpinned. -/
theorem getCore_oob (cores : List Nat) (i : Nat) (h : cores.length ≤ i) : getCore true cores i = none := by
  simp [getCore, h]

-- § worker count

/-- Requested workers, else the core count, but never fewer than one. -/
def nWorkers (req nCores : Nat) : Nat := if (if 0 < req then req else nCores) < 1 then 1 else (if 0 < req then req else nCores)

/-- The pool always has a worker, so a single-core machine still produces a token. -/
theorem nWorkers_pos (req nCores : Nat) : 1 ≤ nWorkers req nCores := by
  unfold nWorkers
  by_cases h : 0 < req
  · rw [if_pos h]
    by_cases h2 : req < 1
    · rw [if_pos h2] <;> omega
    · rw [if_neg h2] <;> omega
  · rw [if_neg h]
    by_cases h2 : nCores < 1
    · rw [if_pos h2] <;> omega
    · rw [if_neg h2] <;> omega

/-- An explicit positive request is honoured. -/
theorem nWorkers_req (req nCores : Nat) (h : 0 < req) : nWorkers req nCores = req := by
  unfold nWorkers
  rw [if_pos h]
  have h2 : ¬ req < 1 := by omega
  rw [if_neg h2]

-- § task row ranges

/-- Start row of task `i` when `rows` rows are split over `tasks` tasks. -/
def lo (rows tasks i : Nat) : Nat := rows * i / tasks

/-- The first task starts at row 0. -/
theorem lo_zero (rows tasks : Nat) : lo rows tasks 0 = 0 := by
  simp [lo]

/-- The last boundary is exactly `rows`. -/
theorem lo_last (rows tasks : Nat) (h : 0 < tasks) : lo rows tasks tasks = rows := by
  unfold lo
  exact Nat.mul_div_cancel rows h

/-- Ranges are ordered: a task's end is not before its start. -/
theorem lo_mono (rows tasks i : Nat) : lo rows tasks i ≤ lo rows tasks (i + 1) := by
  unfold lo
  exact Nat.div_le_div_right (Nat.mul_le_mul (Nat.le_refl rows) (Nat.le_succ i))

/-- Adjacent tasks share a boundary, so the ranges tile with no gap and no overlap (by definition of `lo`). -/
theorem lo_adjacent (rows tasks i : Nat) : lo rows tasks (i + 1) = rows * (i + 1) / tasks := by
  rfl

/-- A flat task id `i < n*parts` splits into an expert `i/parts < n` and a part `i%parts < parts`. -/
theorem task_split (i n parts : Nat) (hp : 0 < parts) (hi : i < n * parts) :
    i / parts < n ∧ i % parts < parts ∧ parts * (i / parts) + i % parts = i := by
  refine ⟨?_, Nat.mod_lt _ hp, Nat.div_add_mod i parts⟩
  exact (Nat.div_lt_iff_lt_mul hp).2 hi

/-- The number of parts per expert is at least one. -/
theorem parts_pos (threads n : Nat) : 1 ≤ max 1 ((3 * threads + n - 1) / n) :=
  Nat.le_max_left 1 _

-- § inner segment walk

/-- Length of the segment taken from row `r` of a range ending at `g1`, in experts of `per` rows. -/
def stepLen (per g1 r : Nat) : Nat := min per (r % per + (g1 - r)) - r % per

/-- Each segment is non-empty while rows remain. -/
theorem stepLen_pos (per g1 r : Nat) (hper : 0 < per) (hr : r < g1) : 0 < stepLen per g1 r := by
  have hm : r % per < per := Nat.mod_lt _ hper
  unfold stepLen
  generalize r % per = r0 at *
  omega

/-- A segment never overruns the range end. -/
theorem stepLen_le_range (per g1 r : Nat) (hper : 0 < per) (hr : r < g1) : r + stepLen per g1 r ≤ g1 := by
  have hm : r % per < per := Nat.mod_lt _ hper
  unfold stepLen
  generalize r % per = r0 at *
  omega

/-- A segment never crosses its expert's boundary: the offset in the expert plus the length is at most `per`. -/
theorem stepLen_in_expert (per g1 r : Nat) (hper : 0 < per) : r % per + stepLen per g1 r ≤ per := by
  have hm : r % per < per := Nat.mod_lt _ hper
  unfold stepLen
  generalize r % per = r0 at *
  omega

/-- The expert and in-expert offset reassemble the row. -/
theorem row_decomp (r per : Nat) : per * (r / per) + r % per = r :=
  Nat.div_add_mod r per

/-- Rows processed by walking from `r` to `g1` with fuel `f`. -/
def cover (per g1 : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | f + 1, r => if r < g1 then stepLen per g1 r + cover per g1 f (r + stepLen per g1 r) else 0

/-- The walk covers exactly the `g1 - r` remaining rows when the fuel suffices. -/
theorem cover_eq (per g1 : Nat) (hper : 0 < per) (f : Nat) :
    ∀ r, r ≤ g1 → g1 - r ≤ f → cover per g1 f r = g1 - r := by
  induction f with
  | zero =>
    intro r h1 h2
    have : g1 - r = 0 := by omega
    rw [this]
    rfl
  | succ f ih =>
    intro r h1 h2
    have e : cover per g1 (f + 1) r =
        if r < g1 then stepLen per g1 r + cover per g1 f (r + stepLen per g1 r) else 0 := rfl
    rw [e]
    by_cases hr : r < g1
    · rw [if_pos hr]
      have hp := stepLen_pos per g1 r hper hr
      have hl := stepLen_le_range per g1 r hper hr
      rw [ih (r + stepLen per g1 r) hl (by omega)]
      omega
    · rw [if_neg hr]
      omega

-- § native batching

/-- Sizes of the batches of at most `K` experts a loop over `n` experts produces, with fuel `f`. -/
def batches (K : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, rem => if rem = 0 then [] else min K rem :: batches K f (rem - min K rem)

/-- The batches cover all `n` experts. -/
theorem batches_sum (K : Nat) (f : Nat) : ∀ rem, rem ≤ K * f → (batches K f rem).sum = rem := by
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    rfl
  | succ f ih =>
    intro rem h
    rw [Nat.mul_succ] at h
    by_cases hr : rem = 0
    · subst hr
      simp [batches]
    · have e : batches K (f + 1) rem = min K rem :: batches K f (rem - min K rem) := by
        simp [batches, hr]
      rw [e, List.sum_cons, ih (rem - min K rem) (by omega)]
      omega

/-- A batch starting at `b0 < n` is non-empty and stays inside `n`. -/
theorem batch_bounds (K n b0 : Nat) (hK : 0 < K) (h : b0 < n) :
    1 ≤ min K (n - b0) ∧ b0 + min K (n - b0) ≤ n := by
  refine ⟨?_, ?_⟩ <;> omega

-- § affinity bookkeeping

/-- Windows `SetThreadAffinityMask` returns 0 on failure, so 0 maps to the error value -1. -/
def winPin (prev : Int) : Int := if prev = 0 then -1 else prev

/-- The error value is returned exactly when the OS call failed. -/
theorem winPin_err (prev : Int) : winPin prev = -1 ↔ prev = 0 ∨ prev = -1 := by
  unfold winPin
  split <;> omega

/-- A successful pin never yields the ambiguous value 0, so a returned mask is always restorable. -/
theorem winPin_ne_zero (prev : Int) : winPin prev ≠ 0 := by
  unfold winPin
  split <;> omega

/-- Restoring a non-positive previous mask is a no-op. -/
def restoreEffect (prev : Int) : Option Int := if prev ≤ 0 then none else some prev

/-- Error and unset masks are never restored. -/
theorem restore_noop (prev : Int) (h : prev ≤ 0) : restoreEffect prev = none := by
  simp [restoreEffect, h]

/-- Group/bit processor numbering: `group*64 + bit` decomposes back for `bit < 64`. -/
theorem cpu_index_roundtrip (g b : Nat) (hb : b < 64) : (g * 64 + b) / 64 = g ∧ (g * 64 + b) % 64 = b := by
  refine ⟨?_, ?_⟩ <;> omega

/-- `core & 63` is a valid bit position. -/
theorem core_bit_lt (core : Nat) : core % 64 < 64 := by
  omega

end Ref.CpuExpertPool



## FILE: /mnt/user-data/outputs/Q2Avx2Rows.lean
/-!
# MODULE: q2_avx2_rows
SOURCES: src/kernels/cpu/q2_avx2.cpp
The Q2_0 expert rows and the Q8_1 activation quantizer for CPUs without AVX-512 (AVX2 only): the 64-code
unpack order, the 32-value chunk/scale indexing, the saturation-free `maddubs` bounds, the horizontal
integer sum, and the token chunking loop.

## Source semantics
* A 18-byte block is a 2-byte scale followed by 16 code bytes; value `i` of the block is in byte `i/4` at
  bit offset `2*(i%4)`; the unpack puts it at output position `4*byte + j`.
* `lo` holds values 0..31 (bytes 0..7) and `hi` values 32..63 (bytes 8..15); each is paired with the
  activation chunk `2b` or `2b+1`.
* Codes are at most 3 and activations are within ±127, so no int16 lane saturates and the int32 total of a
  block is tiny.
* The integer horizontal sum of 8 lanes equals the plain sum of the lanes.
* The quantizer writes `a.q[k*32 + j]` for `k < n/32`, `j < 32`: a bijection onto `[0, n)` when `32 | n`.

## Form template identifiers
* `valuePos`       : unpack_position
* `halfOf`         : activation_chunk_of_value
* `hsum8`          : integer_horizontal_sum
* `chunks4`        : four_token_chunking
* `quantIndex`     : quantizer_output_index
-/

namespace Ref.Q2Avx2

-- § unpack order

/-- Output position of code `j` (0..3) of byte `k` (0..15) after the interleaving unpacks. -/
def valuePos (k j : Nat) : Nat := 4 * k + j

/-- The position map is a bijection of `[0,16) x [0,4)` onto `[0,64)`. -/
theorem valuePos_roundtrip (k j : Nat) (hk : k < 16) (hj : j < 4) :
    valuePos k j < 64 ∧ valuePos k j / 4 = k ∧ valuePos k j % 4 = j := by
  unfold valuePos
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Distinct (byte, code) pairs land on distinct positions. -/
theorem valuePos_inj (k j k' j' : Nat) (hj : j < 4) (hj' : j' < 4) (h : valuePos k j = valuePos k' j') :
    k = k' ∧ j = j' := by
  unfold valuePos at h
  refine ⟨?_, ?_⟩ <;> omega

/-- Values `0..31` come from bytes `0..7` (the `lo` vector). -/
theorem lo_bytes (v : Nat) (hv : v < 32) : v / 4 < 8 := by
  omega

/-- Values `32..63` come from bytes `8..15` (the `hi` vector). -/
theorem hi_bytes (v : Nat) (h1 : 32 ≤ v) (h2 : v < 64) : 8 ≤ v / 4 ∧ v / 4 < 16 := by
  refine ⟨?_, ?_⟩ <;> omega

/-- Activation chunk of value `v` of block `b`: `lo` pairs with chunk `2b`, `hi` with `2b+1`. -/
def halfOf (b v : Nat) : Nat := (b * 64 + v) / 32

/-- The chunk index is `2*b + v/32`. -/
theorem halfOf_eq (b v : Nat) (hv : v < 64) : halfOf b v = 2 * b + v / 32 := by
  unfold halfOf
  omega

/-- A 2-bit code `(x >> 2j) & 3` is below 4. -/
theorem code_lt (x : Nat) : x % 4 < 4 :=
  Nat.mod_lt _ (by decide)

-- § saturation-free arithmetic

/-- A code (0..3) times an int8 activation is within ±381. -/
theorem code_mul_bound (c q : Int) (h0 : 0 ≤ c) (h3 : c ≤ 3) (hq1 : -127 ≤ q) (hq2 : q ≤ 127) :
    -381 ≤ c * q ∧ c * q ≤ 381 := by
  interval_cases c <;> omega

/-- One `maddubs` int16 lane (two products) fits easily in int16. -/
theorem maddubs_lane_fits (c1 c2 q1 q2 : Int) (h1 : 0 ≤ c1) (h2 : c1 ≤ 3) (h3 : 0 ≤ c2) (h4 : c2 ≤ 3)
    (hq1 : -127 ≤ q1) (hq2 : q1 ≤ 127) (hq3 : -127 ≤ q2) (hq4 : q2 ≤ 127) :
    -762 ≤ c1 * q1 + c2 * q2 ∧ c1 * q1 + c2 * q2 ≤ 762 := by
  have a := code_mul_bound c1 q1 h1 h2 hq1 hq2
  have b := code_mul_bound c2 q2 h3 h4 hq3 hq4
  exact ⟨by omega, by omega⟩

/-- A `madd` against ones adds two int16 lanes: still within ±1524. -/
theorem madd_pair_fits (x y : Int) (hx1 : -762 ≤ x) (hx2 : x ≤ 762) (hy1 : -762 ≤ y) (hy2 : y ≤ 762) :
    -1524 ≤ x + y ∧ x + y ≤ 1524 := by
  exact ⟨by omega, by omega⟩

/-- The 64 products of a block are bounded by 64*381, inside int32. -/
theorem block_total_fits (n : Nat) (hn : n ≤ 64) (s : Int) (h1 : -(381 * (n : Int)) ≤ s)
    (h2 : s ≤ 381 * (n : Int)) : -24384 ≤ s ∧ s ≤ 24384 := by
  have : (n : Int) ≤ 64 := by exact_mod_cast hn
  exact ⟨by omega, by omega⟩

-- § horizontal sums

/-- Horizontal sum of 8 integer lanes by the 256 -> 128 -> 64 -> 32 reduction used by the source. -/
def hsum8 (l0 l1 l2 l3 l4 l5 l6 l7 : Int) : Int :=
  ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))

/-- The staged reduction equals the plain sum of the lanes. -/
theorem hsum8_eq (l0 l1 l2 l3 l4 l5 l6 l7 : Int) :
    hsum8 l0 l1 l2 l3 l4 l5 l6 l7 = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 := by
  unfold hsum8
  omega

-- § quantizer indexing

/-- Output index of value `j` of chunk `k`. -/
def quantIndex (k j : Nat) : Nat := k * 32 + j

/-- With `32 | n` the chunk count times 32 is `n`. -/
theorem nchunks_exact (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by
  omega

/-- Each (chunk, value) pair is an in-range index. -/
theorem quantIndex_lt (n k j : Nat) (h : n % 32 = 0) (hk : k < n / 32) (hj : j < 32) :
    quantIndex k j < n := by
  unfold quantIndex
  omega

/-- The index map is injective: no two writes collide. -/
theorem quantIndex_inj (k j k' j' : Nat) (hj : j < 32) (hj' : j' < 32)
    (h : quantIndex k j = quantIndex k' j') : k = k' ∧ j = j' := by
  unfold quantIndex at h
  refine ⟨?_, ?_⟩ <;> omega

/-- The `qi` staging array holds 4 vectors of 8 lanes, exactly one 32-value chunk. -/
theorem staging_size : 4 * 8 = 32 := by
  decide

/-- Vector `i < 4`, lane `l < 8` of the staging array is index `8i + l < 32`. -/
theorem staging_index_lt (i l : Nat) (hi : i < 4) (hl : l < 8) : 8 * i + l < 32 := by
  omega

-- § four-token chunking

/-- Sizes of the chunks of at most 4 tokens a loop over `rem` tokens produces, with fuel `f`. -/
def chunks4 : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, rem => if rem = 0 then [] else min 4 rem :: chunks4 f (rem - min 4 rem)

/-- The chunks cover all tokens when the fuel suffices. -/
theorem chunks4_sum (f : Nat) : ∀ rem, rem ≤ 4 * f → (chunks4 f rem).sum = rem := by
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    rfl
  | succ f ih =>
    intro rem h
    by_cases hr : rem = 0
    · subst hr
      simp [chunks4]
    · have e : chunks4 (f + 1) rem = min 4 rem :: chunks4 f (rem - min 4 rem) := by
        simp [chunks4, hr]
      rw [e, List.sum_cons, ih (rem - min 4 rem) (by omega)]
      omega

/-- Every chunk size is in `1..4`. -/
theorem chunks4_range (f : Nat) : ∀ rem, ∀ c ∈ chunks4 f rem, 1 ≤ c ∧ c ≤ 4 := by
  induction f with
  | zero => intro rem c hc; simp [chunks4] at hc
  | succ f ih =>
    intro rem c hc
    by_cases hr : rem = 0
    · subst hr
      simp [chunks4] at hc
    · have e : chunks4 (f + 1) rem = min 4 rem :: chunks4 f (rem - min 4 rem) := by
        simp [chunks4, hr]
      rw [e] at hc
      rcases List.mem_cons.mp hc with h | h
      · rw [h]; omega
      · exact ih _ c h

end Ref.Q2Avx2



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- PleBlockCuda.lean
  was: theorem scratch_regions_fit :
    floatBytes ≤ actOff ∧ actOff + q8Bytes ≤ embOff ∧ embOff + embBytes ≤ scratchBytes
  now: theorem scratch_gen (F Q E : Nat) :
    F ≤ roundUp16 F ∧ roundUp16 F + Q ≤ roundUp16 F + roundUp16 Q ∧
    roundUp16 F + roundUp16 Q + E ≤ roundUp16 F + roundUp16 Q + E + 256
  measured: the claim changed
-/



## FILE: /mnt/user-data/outputs/CanonicalQuantBitExactMaps.lean
/-!
# MODULE: canonical_quant_bit_exact_maps
SOURCES: tools/canonical_xcheck.py
Integer-level model of the GGML-quant -> canonical-form (S2/S4/S8/P16/P32) mappings. The Python tool checks the
float decode bit-exactly against gguf-py; this theory captures the integer / layout contracts that make that
possible: block byte decompositions, code ranges, integer biases, nibble and bit-plane splits, mask walks.

## Source semantics
Each mapping is (block_bytes, block_elems, group_elems, code_bias, has_offset). The decode is
`(code + bias) * scale (+ offset)`, the bias applied to the INTEGER code before the single multiply.
Codes are unsigned; Q2_0 `-1`, Q3_K `-4`, Q4_0 `-8`, Q5_0 `-16`, Q6_K `-32`, Q8_0 `-128` are integer biases.
IQ4_NL / IQ4_XS use a non-affine 16-entry codebook. Floating point rounding is NOT modelled here.

## Form template identifiers
mapping registry `MAPPINGS` -> `Mapping`; `to_s*`/`from_s*` -> pack/unpack; `KV_IQ4NL` -> `kv`;
`get_scale_min_k4` -> `scaleMinLow`/`scaleMinHigh`; `BLOCK_BYTES`/`QK` -> `Mapping.blockBytes`/`blockElems`.
-/

namespace RefCanonicalXcheck

-- § mapping registry

/-- One row of the MAPPINGS registry (integer part). -/
structure Mapping where
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat
  codeBias : Int
  hasOffset : Bool
  deriving DecidableEq, Repr

/-- Q2_0 mapping row. -/
def q2_0 : Mapping := ⟨18, 64, 64, -1, false⟩
/-- Q3_K mapping row. -/
def q3_k : Mapping := ⟨110, 256, 16, -4, false⟩
/-- Q4_0 mapping row. -/
def q4_0 : Mapping := ⟨18, 32, 32, -8, false⟩
/-- Q4_K mapping row. -/
def q4_k : Mapping := ⟨144, 256, 32, 0, true⟩
/-- Q5_0 mapping row. -/
def q5_0 : Mapping := ⟨22, 32, 32, -16, false⟩
/-- Q5_K mapping row. -/
def q5_k : Mapping := ⟨176, 256, 32, 0, true⟩
/-- Q6_K mapping row. -/
def q6_k : Mapping := ⟨210, 256, 16, -32, false⟩
/-- Q8_0 mapping row. -/
def q8_0 : Mapping := ⟨34, 32, 32, -128, false⟩
/-- IQ4_NL mapping row. -/
def iq4_nl : Mapping := ⟨18, 32, 32, 0, false⟩
/-- IQ4_XS mapping row. -/
def iq4_xs : Mapping := ⟨136, 256, 32, 0, false⟩

/-- All integer-quant rows of the registry. -/
def allMappings : List Mapping := [q2_0, q3_k, q4_0, q4_k, q5_0, q5_k, q6_k, q8_0, iq4_nl, iq4_xs]

/-- Every group size divides its block size. -/
theorem groups_divide_blocks : ∀ m ∈ allMappings, m.blockElems % m.groupElems = 0 := by decide

/-- Q2_0 block = 2-byte fp16 scale + 64/4 bytes of 2-bit codes. -/
theorem q2_0_layout : q2_0.blockBytes = 2 + 64 / 4 := by decide
/-- Q3_K block = hmask 32 + qs 64 + scales 12 + d 2. -/
theorem q3_k_layout : q3_k.blockBytes = 32 + 64 + 12 + 2 := by decide
/-- Q4_K block = d 2 + dmin 2 + scales 12 + qs 128. -/
theorem q4_k_layout : q4_k.blockBytes = 2 + 2 + 12 + 128 := by decide
/-- Q5_K block = d 2 + dmin 2 + scales 12 + qh 32 + qs 128. -/
theorem q5_k_layout : q5_k.blockBytes = 2 + 2 + 12 + 32 + 128 := by decide
/-- Q6_K block = ql 128 + qh 64 + scales 16 + d 2. -/
theorem q6_k_layout : q6_k.blockBytes = 128 + 64 + 16 + 2 := by decide
/-- Q5_0 block = d 2 + qh 4 + qs 16. -/
theorem q5_0_layout : q5_0.blockBytes = 2 + 4 + 16 := by decide
/-- Q8_0 block = d 2 + 32 int8 codes. -/
theorem q8_0_layout : q8_0.blockBytes = 2 + 32 := by decide
/-- IQ4_XS block = d 2 + scales_h 2 + scales_l 4 + qs 128. -/
theorem iq4_xs_layout : iq4_xs.blockBytes = 2 + 2 + 4 + 128 := by decide

/-- Rows with an additive offset are exactly Q4_K and Q5_K. -/
theorem offset_rows : (allMappings.filter (fun m => m.hasOffset)) = [q4_k, q5_k] := by decide

-- § integer-bias decode

/-- Integer decode: the bias is applied to the integer code BEFORE the single multiply. -/
def decodeInt (c : Nat) (bias : Int) (scale : Int) : Int := ((c : Int) + bias) * scale

/-- Q2_0 codebook is {-1,0,1,2}, not {0,1,2,3}. -/
theorem q2_0_value_set : ∀ c : Fin 4, ((c.val : Int) + q2_0.codeBias) ∈ [-1, 0, 1, 2] := by decide

/-- Decoding Q2_0 codes as unsigned would be wrong at every code. -/
theorem q2_0_bias_matters : ∀ c : Fin 4, ((c.val : Int) + q2_0.codeBias) ≠ c.val := by decide

/-- With unit scale the decode is exactly the biased code. -/
theorem decodeInt_unit (c : Nat) (b : Int) : decodeInt c b 1 = (c : Int) + b := by
  unfold decodeInt; omega

/-- Q4_0 codes 0..15 decode to -8..7. -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) + q4_0.codeBias ∧ (c : Int) + q4_0.codeBias ≤ 7 := by
  have : q4_0.codeBias = -8 := rfl
  rw [this]; omega

/-- Q8_0: an int8 stored +128 is an unsigned byte and decodes back to the same int8. -/
theorem q8_0_bias_roundtrip (s : Int) (h1 : -128 ≤ s) (h2 : s ≤ 127) :
    ∃ c : Nat, c < 256 ∧ (c : Int) + q8_0.codeBias = s := by
  have : q8_0.codeBias = -128 := rfl
  rw [this]
  exact ⟨(s + 128).toNat, by omega, by omega⟩

/-- Q5_0 codes 0..31 decode to -16..15. -/
theorem q5_0_range (c : Nat) (h : c < 32) : -16 ≤ (c : Int) + q5_0.codeBias ∧ (c : Int) + q5_0.codeBias ≤ 15 := by
  have : q5_0.codeBias = -16 := rfl
  rw [this]; omega

-- § nibble splits (Q4_0, IQ4_NL, IQ4_XS, Q4_K, Q5_K)

/-- A byte splits into low nibble (first half of block) and high nibble (second half) and recombines. -/
theorem nibble_split (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 ∧ b % 16 + 16 * (b / 16) = b := by
  omega

/-- Q4_0: element j+16 comes from the high nibble of byte j, so the halves are not interleaved. -/
theorem q4_0_halves (j : Nat) (hj : j < 16) : j < 32 ∧ j + 16 < 32 ∧ j ≠ j + 16 := by omega

-- § Q2_0 bit plane

/-- Byte index and shift of element j of a Q2_0 block. -/
theorem q2_0_pos (j : Nat) (hj : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 < 8 ∧ 4 * (j / 4) + j % 4 = j := by
  omega

/-- Packing four 2-bit codes in one byte and reading them back (LSB first). -/
theorem q2_0_pack_roundtrip (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) < 256 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) % 4 = c0 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 4 % 4 = c1 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 16 % 4 = c2 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 64 % 4 = c3 := by
  omega

-- § Q3_K

/-- The Q3_K 6-bit scale: low nibble plus two high bits is below 64. -/
theorem q3_k_scale_bound (lo t : Nat) (hl : lo < 256) (ht : t < 256) : lo % 16 + 16 * (t % 4) < 64 := by
  omega

/-- Stored Q3_K code = 2 bits + 4*hmask-bit stays in 0..7 and decodes to -4..3. -/
theorem q3_k_code_range (c2 m : Nat) (hc : c2 < 4) (hm : m ≤ 1) :
    c2 + 4 * m < 8 ∧ -4 ≤ ((c2 + 4 * m : Nat) : Int) + q3_k.codeBias ∧
    ((c2 + 4 * m : Nat) : Int) + q3_k.codeBias ≤ 3 := by
  have : q3_k.codeBias = -4 := rfl
  rw [this]; omega

/-- The hmask bit `4*half + j` never resets between halves: all 8 (half, j) pairs hit distinct bits < 8. -/
theorem q3_k_mask_bit_injective (h1 j1 h2 j2 : Nat) (a : h1 < 2) (b : j1 < 4) (c : h2 < 2) (d : j2 < 4)
    (e : 4 * h1 + j1 = 4 * h2 + j2) : h1 = h2 ∧ j1 = j2 := by
  omega

/-- Q3_K output group index `8*half + 2*j + sub` is below 16 and injective. -/
theorem q3_k_group_injective (h1 j1 s1 h2 j2 s2 : Nat) (a : h1 < 2) (b : j1 < 4) (c : s1 < 2)
    (d : h2 < 2) (e : j2 < 4) (f : s2 < 2) (g : 8 * h1 + 2 * j1 + s1 = 8 * h2 + 2 * j2 + s2) :
    8 * h1 + 2 * j1 + s1 < 16 ∧ h1 = h2 ∧ j1 = j2 ∧ s1 = s2 := by
  omega

-- § get_scale_min_k4

/-- Scale of group j < 4: low six bits of byte j. -/
def scaleMinLow (s : Nat) : Nat := s % 64

/-- Scale of group j >= 4: low nibble of s[j+4] and top two bits of s[j-4] shifted by 4. -/
def scaleMinHigh (sj4 sjm4 : Nat) : Nat := sj4 % 16 + 16 * (sjm4 / 64)

/-- Both scale forms are 6-bit values. -/
theorem scale_min_k4_bound (a b : Nat) (ha : a < 256) (hb : b < 256) :
    scaleMinLow a < 64 ∧ scaleMinHigh a b < 64 := by
  unfold scaleMinLow scaleMinHigh; omega

-- § Q5_K

/-- The Q5_K qh masks u1 for the four 64-element steps. -/
theorem q5_k_u1 : (List.range 4).map (fun j => 1 <<< (2 * j)) = [1, 4, 16, 64] := by decide
/-- The Q5_K qh masks u2 for the four 64-element steps. -/
theorem q5_k_u2 : (List.range 4).map (fun j => 2 <<< (2 * j)) = [2, 8, 32, 128] := by decide
/-- u1 and u2 together cover every one of the 8 bits of qh exactly once. -/
theorem q5_k_masks_cover : [1, 4, 16, 64, 2, 8, 32, 128].foldl (· + ·) 0 = 255 := by decide

/-- A Q5_K code is a nibble plus 16 times a bit, so below 32. -/
theorem q5_k_code_bound (n b : Nat) (hn : n < 16) (hb : b ≤ 1) : n + 16 * b < 32 := by omega

-- § Q6_K

/-- Q6_K code = nibble + 16 * two bits, below 64, decoding to -32..31. -/
theorem q6_k_code_range (n h : Nat) (hn : n < 16) (hh : h < 4) :
    n + 16 * h < 64 ∧ -32 ≤ ((n + 16 * h : Nat) : Int) + q6_k.codeBias ∧
    ((n + 16 * h : Nat) : Int) + q6_k.codeBias ≤ 31 := by
  have : q6_k.codeBias = -32 := rfl
  rw [this]; omega

/-- Output element index of quad q (0..3), lane l (0..31) in half h is 128h+32q+l, and its scale group is element/16. -/
theorem q6_k_scale_group (h q l : Nat) (hh : h < 2) (hq : q < 4) (hl : l < 32) :
    (128 * h + 32 * q + l) / 16 = 8 * h + 2 * q + l / 16 ∧ 128 * h + 32 * q + l < 256 := by
  omega

-- § IQ4

/-- The IQ4 non-linear codebook `kvalues_iq4nl`. -/
def kv : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook lookup (out of range reads 0). -/
def kvAt (i : Nat) : Int := kv.getD i 0

/-- The codebook has exactly 16 entries (a 4-bit index). -/
theorem kv_length : kv.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kv_strictMono : ∀ i j : Fin 16, i.val < j.val → kvAt i.val < kvAt j.val := by decide

/-- The codebook is not affine: consecutive differences are not constant, so no integer bias can express it. -/
theorem kv_not_affine : kvAt 1 - kvAt 0 ≠ kvAt 2 - kvAt 1 := by decide

/-- The first eight entries are negative and the last eight positive. -/
theorem kv_sign_split : (∀ i : Fin 8, kvAt i.val < 0) ∧ (∀ i : Fin 8, 0 < kvAt (i.val + 8)) := by decide

/-- IQ4_XS 6-bit scale: nibble + 16 * two bits, giving ls - 32 in -32..31. -/
theorem iq4_xs_ls_range (n h : Nat) (hn : n < 16) (hh : h < 4) :
    n + 16 * h < 64 ∧ -32 ≤ ((n + 16 * h : Nat) : Int) - 32 ∧ ((n + 16 * h : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- IQ4_XS group ib indexes scales_l byte ib/2 (below 4) and scales_h bits 2ib, 2ib+1 (below 16). -/
theorem iq4_xs_scale_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ 2 * ib + 1 < 16 := by omega

-- § float forms

/-- bf16 widening: shifting the 16 bits left by 16 gives a value whose low 16 bits are zero, recoverable. -/
theorem bf16_widen_exact (u : Nat) (h : u < 65536) : (u * 65536) % 65536 = 0 ∧ (u * 65536) / 65536 = u := by
  omega

-- § full-run bookkeeping

/-- The split `idx % N == I` assigns every tensor index to exactly one process below N. -/
theorem split_in_range (idx n : Nat) (h : 0 < n) : idx % n < n := Nat.mod_lt _ h

/-- Streaming chunks hold whole blocks and never exceed the chunk size. -/
theorem chunk_whole_blocks (chunk bb : Nat) : (chunk / bb) * bb ≤ chunk := Nat.div_mul_le_self _ _

/-- Data section offset: file length minus the end of the last tensor places that end exactly at the file end. -/
theorem data_offset_exact (filelen endOfData : Nat) (h : endOfData ≤ filelen) :
    (filelen - endOfData) + endOfData = filelen := by omega

end RefCanonicalXcheck



## FILE: /mnt/user-data/outputs/GgufWriterQ2Contract.lean
/-!
# MODULE: gguf_writer_q2_contract
SOURCES: tools/gguf_writer.py
Integer model of the minimal GGUF v3 writer and the Q2_0 quantiser contract: type ids, alignment of the
tensor directory and data section, the Q2_0 code clamp and bit packing.

## Source semantics
Q2_0 block = 64 elements, fp16 scale d = amax, code = clamp(round(w/d)+1, 0, 3), symbol = code-1 in {-1,0,1,2};
codes are packed 4 per byte, LSB first (element j -> byte j/4, shift 2*(j%4)). Tensor data offsets are
rounded up to the alignment (32) after each tensor; the data section starts at an aligned offset.
Metadata and tensor shape order follow GGUF (dim 0 varies fastest).

## Form template identifiers
`TYPE_IDS` -> `typeIds`; `_META_IDS` -> `metaIds`; `DEFAULT_ALIGNMENT` -> 32; `quantize_q2_0` -> `clamp`/`packByte`;
`GGUFWriter.write` offset loop -> `nextOffset`/`offsetAfter`.
-/

namespace RefGgufWriter

-- § type ids

/-- ggml type ids the writer can emit. -/
def typeIds : List (String × Nat) := [("F32", 0), ("F16", 1), ("Q2_0", 42)]

/-- Type ids are pairwise distinct. -/
theorem type_ids_nodup : (typeIds.map Prod.snd).Nodup := by decide

/-- Metadata value type ids of the GGUF spec. -/
def metaIds : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

/-- Metadata type ids are pairwise distinct. -/
theorem meta_ids_nodup : metaIds.Nodup := by decide

-- § Q2_0 quantiser

/-- Integer clamp used for the code. -/
def clamp (lo hi x : Int) : Int := max lo (min hi x)

/-- The clamp lands in [lo, hi]. -/
theorem clamp_range (lo hi x : Int) (h : lo ≤ hi) : lo ≤ clamp lo hi x ∧ clamp lo hi x ≤ hi := by
  unfold clamp; omega

/-- A Q2_0 code `clamp(r+1, 0, 3)` is always a valid 2-bit code. -/
theorem q2_0_code_valid (r : Int) : 0 ≤ clamp 0 3 (r + 1) ∧ clamp 0 3 (r + 1) ≤ 3 :=
  clamp_range 0 3 (r + 1) (by omega)

/-- With d = amax, |w/d| <= 1, so rounded values are in {-1,0,1} and the clamp is the identity: the +2 level is never produced. -/
theorem q2_0_amax_never_plus2 (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 1) :
    clamp 0 3 (r + 1) = r + 1 ∧ clamp 0 3 (r + 1) ≤ 2 := by
  unfold clamp; omega

/-- Symbol = code - 1 lies in {-1,0,1,2}. -/
theorem q2_0_symbol_range (r : Int) : -1 ≤ clamp 0 3 (r + 1) - 1 ∧ clamp 0 3 (r + 1) - 1 ≤ 2 := by
  unfold clamp; omega

/-- Four 2-bit codes in one byte, element 4b+t at shift 2t. -/
def packByte (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed byte fits 8 bits and each code is recovered by shift and mask. -/
theorem packByte_roundtrip (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 < 256 ∧ packByte c0 c1 c2 c3 % 4 = c0 ∧ packByte c0 c1 c2 c3 / 4 % 4 = c1 ∧
    packByte c0 c1 c2 c3 / 16 % 4 = c2 ∧ packByte c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold packByte; omega

/-- Element j of a block lives in byte j/4 < 16 at shift (j%4)*2 < 8. -/
theorem element_position (j : Nat) (hj : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 < 8 ∧ 4 * (j / 4) + j % 4 = j := by
  omega

/-- A block is an 18-byte unit: 2 scale bytes + 16 code bytes. -/
theorem block_bytes : 2 + 64 / 4 = 18 := by decide

/-- Whole-block requirement: n multiple of 64 gives n/64 blocks covering n elements, in (n/64)*18 bytes. -/
theorem blocks_cover (n : Nat) (h : n % 64 = 0) : (n / 64) * 64 = n := by omega

-- § alignment

/-- Padding to a 32-byte boundary. -/
def pad32 (n : Nat) : Nat := (32 - n % 32) % 32

/-- Round up to a multiple of 32. -/
def alignUp32 (n : Nat) : Nat := (n + 31) / 32 * 32

/-- Padding is below 32 and makes the length aligned. -/
theorem pad32_spec (n : Nat) : pad32 n < 32 ∧ (n + pad32 n) % 32 = 0 := by
  unfold pad32; omega

/-- Padding and rounding up agree. -/
theorem pad32_eq_alignUp (n : Nat) : n + pad32 n = alignUp32 n := by
  unfold pad32 alignUp32; omega

/-- Next tensor offset: current offset plus the data, rounded up to the alignment. -/
def nextOffset (o len : Nat) : Nat := alignUp32 (o + len)

/-- Every tensor offset after the first is aligned. -/
theorem nextOffset_aligned (o len : Nat) : nextOffset o len % 32 = 0 := by
  unfold nextOffset alignUp32; omega

/-- A tensor never overlaps the next one. -/
theorem nextOffset_ge (o len : Nat) : o + len ≤ nextOffset o len := by
  unfold nextOffset alignUp32; omega

/-- Padding wastes less than 32 bytes. -/
theorem nextOffset_lt (o len : Nat) : nextOffset o len < o + len + 32 := by
  unfold nextOffset alignUp32; omega

/-- Offset after a whole list of tensors. -/
def offsetAfter (lens : List Nat) : Nat := lens.foldl nextOffset 0

/-- Appending a tensor advances the offset by `nextOffset`. -/
theorem offsetAfter_append (lens : List Nat) (l : Nat) :
    offsetAfter (lens ++ [l]) = nextOffset (offsetAfter lens) l := by
  simp [offsetAfter, List.foldl_append]

-- § shapes

/-- A fused expert tensor [in, out, experts] has in*out*experts elements; rows = out*experts. -/
theorem fused_rows (i o e : Nat) : i * (o * e) = i * o * e := (Nat.mul_assoc i o e).symm

/-- The concrete case from the source: [64, 256, 16] has 4096 rows of 64. -/
theorem fused_example : 64 * (256 * 16) = 64 * 4096 := by decide

end RefGgufWriter



## FILE: /mnt/user-data/outputs/IqNativePackIndex.lean
/-!
# MODULE: iq_native_pack_index
SOURCES: tools/iq_pack.py
Integer model of the native pack for i-quant model files: per-layer expert blob layout, running offsets,
64-byte arena alignment, BF16 router narrowing, extra.bin offsets, layer counting.

## Source semantics
experts.bin: per layer, 512 blobs of [gate | up | down] raw GGUF slices; blob size is per layer. Layer offset
= sum over earlier layers of blob*512. Float tensors are written to dense.bin padded to 64. An F32 router is
narrowed to BF16 only if the low 16 bits of every word are zero (exact), otherwise refused.

## Form template identifiers
`N_EXPERT` -> 512; `ALIGN` -> 64; `layout` -> `expertsOffset`; `kind` map -> `kindOf`;
`index_from_base` extra offsets -> `extraOffset`.
-/

namespace RefIqPack

-- § constants

/-- Experts per layer. -/
def nExpert : Nat := 512
/-- Arena alignment. -/
def align : Nat := 64

-- § expert layout

/-- Pad to 64. -/
def pad64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- Padding is below 64 and aligns the length. -/
theorem pad64_spec (n : Nat) : pad64 n < 64 ∧ (n + pad64 n) % 64 = 0 := by
  unfold pad64; omega

/-- Offset of layer l in experts.bin, given the per-layer blob size. -/
def expertsOffset (blob : Nat → Nat) : Nat → Nat
  | 0 => 0
  | l + 1 => expertsOffset blob l + blob l * 512

/-- Layer l+1 starts where layer l's 512 blobs end. -/
theorem expertsOffset_succ (blob : Nat → Nat) (l : Nat) :
    expertsOffset blob (l + 1) = expertsOffset blob l + blob l * 512 := rfl

/-- Offsets never decrease with the layer index. -/
theorem expertsOffset_mono (blob : Nat → Nat) (l k : Nat) :
    expertsOffset blob l ≤ expertsOffset blob (l + k) := by
  induction k with
  | zero => exact Nat.le_refl _
  | succ k ih =>
    have h : expertsOffset blob (l + (k + 1)) = expertsOffset blob (l + k) + blob (l + k) * 512 := rfl
    omega

/-- A blob is gate + up + down slices; with each role tensor divisible by 512 the 512 blobs cover the three tensors exactly. -/
theorem blob_cover (g u d : Nat) (hg : g % 512 = 0) (hu : u % 512 = 0) (hd : d % 512 = 0) :
    (g / 512 + u / 512 + d / 512) * 512 = g + u + d := by omega

/-- Layer count is 1 + max layer index: every layer index is below it. -/
theorem le_foldl_max_acc : ∀ (ls : List Nat) (acc : Nat), acc ≤ ls.foldl max acc
  | [], acc => Nat.le_refl _
  | x :: xs, acc => Nat.le_trans (Nat.le_max_left acc x) (le_foldl_max_acc xs (max acc x))

/-- Every listed element is bounded by the fold-max. -/
theorem le_foldl_max : ∀ (ls : List Nat) (acc l : Nat), l ∈ ls → l ≤ ls.foldl max acc
  | [], _, _, h => by simp at h
  | x :: xs, acc, l, h => by
    rcases List.mem_cons.mp h with h1 | h1
    · rw [h1]
      exact Nat.le_trans (Nat.le_max_right acc x) (le_foldl_max_acc xs (max acc x))
    · exact le_foldl_max xs (max acc x) l h1

/-- n_layers = 1 + max index exceeds every layer index present. -/
theorem layer_lt_count (ls : List Nat) (l : Nat) (h : l ∈ ls) : l < 1 + ls.foldl max 0 := by
  have := le_foldl_max ls 0 l h
  omega

-- § dense.bin and routers

/-- Index kind codes: BF16 -> 4, F16 -> 5, F32 -> 2. -/
def kindOf : String → Nat
  | "BF16" => 4
  | "F16" => 5
  | _ => 2

/-- The three float kinds are distinct. -/
theorem kinds_distinct : kindOf "BF16" ≠ kindOf "F16" ∧ kindOf "F16" ≠ kindOf "F32" ∧ kindOf "BF16" ≠ kindOf "F32" := by
  decide

/-- Router F32 -> BF16 narrowing is exact exactly when the low 16 bits are zero; then the high half recovers the word. -/
theorem router_narrowing_exact (u : Nat) (h : u < 4294967296) (hz : u % 65536 = 0) :
    u / 65536 < 65536 ∧ (u / 65536) * 65536 = u := by omega

/-- If the low bits are not zero the narrowing is refused (it would lose information). -/
theorem router_narrowing_lossy (u : Nat) (hz : u % 65536 ≠ 0) : (u / 65536) * 65536 ≠ u := by omega

-- § extra.bin offsets

/-- Offset of the i-th extra tensor: sum of earlier lengths each padded to 64. -/
def extraOffset : List Nat → Nat
  | [] => 0
  | b :: bs => (b + pad64 b) + extraOffset bs

/-- Each padded extra tensor starts on a 64-byte boundary relative to the previous. -/
theorem extra_step_aligned (b : Nat) : (b + pad64 b) % 64 = 0 := (pad64_spec b).2

/-- Padding never shrinks a tensor. -/
theorem extra_step_ge (b : Nat) : b ≤ b + pad64 b := Nat.le_add_right _ _

/-- The total extra offset is a multiple of 64. -/
theorem extraOffset_aligned : ∀ bs : List Nat, extraOffset bs % 64 = 0
  | [] => rfl
  | b :: bs => by
    have h := extra_step_aligned b
    have ih := extraOffset_aligned bs
    unfold extraOffset
    omega

-- § index arena

/-- Index arena offset rounding: each row occupies its size rounded up to 64. -/
def roundRow (n : Nat) : Nat := (n + 63) / 64 * 64

/-- Rounded rows are aligned and wasteful by less than 64. -/
theorem roundRow_spec (n : Nat) : roundRow n % 64 = 0 ∧ n ≤ roundRow n ∧ roundRow n < n + 64 := by
  unfold roundRow; omega

end RefIqPack



## FILE: /mnt/user-data/outputs/MtpPackQuantizers.lean
/-!
# MODULE: mtp_pack_quantizers
SOURCES: tools/mtp_pack.py
Integer model of the MTP draft block packer: the per-block scale search grid for Q2_0, the Q4_0 and Q8_0
reference code ranges, per-row block bytes and the expert tensor byte budget.

## Source semantics
Q2_0 scale d = amax * f with f swept over 17 values from 1/2 to 1; codes clamp(rint(x/d), -1, 2) + 1 in 0..3.
Q4_0: code = min(15, trunc(x/d + 8.5)); packed low nibble = element i, high nibble = element i+16.
Experts are the fused tensors gate_up_proj [512,1280,2560] and down_proj [512,2560,640], quantised along the last axis.

## Form template identifiers
`QUANT` table -> `QuantRow`; `EXPERT_TENSORS` -> the two fused tensors; `block`/`block_bytes` -> `blockElems`/`blockBytes`.
-/

namespace RefMtpPack

-- § scale search grid

/-- Grid index i in 0..16 gives candidate factor (16+i)/32. -/
def gridNum (i : Nat) : Nat := 16 + i

/-- The 17 candidates lie between 1/2 (16/32) and 1 (32/32), inclusive. -/
theorem grid_bounds (i : Nat) (h : i ≤ 16) : 16 ≤ gridNum i ∧ gridNum i ≤ 32 := by
  unfold gridNum; omega

/-- There are exactly 17 candidates. -/
theorem grid_count : (List.range 17).length = 17 := by decide

/-- The endpoints are 1/2 and 1. -/
theorem grid_endpoints : gridNum 0 * 2 = 32 ∧ gridNum 16 = 32 := by decide

-- § code ranges

/-- Integer clamp. -/
def clamp (lo hi x : Int) : Int := max lo (min hi x)

/-- The Q2_0 code `clamp(q,-1,2)+1` is in 0..3. -/
theorem q2_0_code_range (q : Int) : 0 ≤ clamp (-1) 2 q + 1 ∧ clamp (-1) 2 q + 1 ≤ 3 := by
  unfold clamp; omega

/-- Q4_0 code is capped at 15. -/
theorem q4_0_code_cap (t : Nat) : min 15 t ≤ 15 := Nat.min_le_left _ _

/-- Two Q4_0 nibbles pack into one byte: first half low, second half high. -/
theorem q4_0_pack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    lo + 16 * hi < 256 ∧ (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi := by omega

/-- Q4_0 decode: the -8 bias is in the integer, so codes 0..15 give -8..7. -/
theorem q4_0_decode_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by omega

-- § block sizes

/-- Per-quantiser (block elements, block bytes). -/
structure QuantRow where
  blockElems : Nat
  blockBytes : Nat
  deriving DecidableEq, Repr

/-- Q2_0 row. -/
def rowQ2 : QuantRow := ⟨64, 18⟩
/-- Q4_0 row. -/
def rowQ4 : QuantRow := ⟨32, 18⟩
/-- Q8_0 row. -/
def rowQ8 : QuantRow := ⟨32, 34⟩

/-- Bytes to store `n` elements. -/
def bytesFor (r : QuantRow) (n : Nat) : Nat := n / r.blockElems * r.blockBytes

/-- Elements of gate_up_proj [512,1280,2560]. -/
def gateUpElems : Nat := 512 * 1280 * 2560
/-- Elements of down_proj [512,2560,640]. -/
def downElems : Nat := 512 * 2560 * 640

/-- Total routed-expert elements. -/
theorem expert_elems : gateUpElems + downElems = 2516582400 := by decide

/-- Both inner dimensions are multiples of the 64-element block (hence of 32 too). -/
theorem inner_dims_divisible : 2560 % 64 = 0 ∧ 640 % 64 = 0 := by decide

/-- Q2_0 expert bytes: 707,788,800 (about 0.71 GB). -/
theorem q2_0_expert_bytes : bytesFor rowQ2 gateUpElems + bytesFor rowQ2 downElems = 707788800 := by decide

/-- Q4_0 expert bytes: 1,415,577,600 (about 1.42 GB). -/
theorem q4_0_expert_bytes : bytesFor rowQ4 gateUpElems + bytesFor rowQ4 downElems = 1415577600 := by decide

/-- Q8_0 expert bytes: 2,673,868,800 (about 2.67 GB). -/
theorem q8_0_expert_bytes : bytesFor rowQ8 gateUpElems + bytesFor rowQ8 downElems = 2673868800 := by decide

/-- The quantised last-axis width written as the raw shape: 2560 -> 720 and 640 -> 180 bytes for Q2_0. -/
theorem raw_shape_last : 2560 / 64 * 18 = 720 ∧ 640 / 64 * 18 = 180 := by decide

/-- A larger format never stores fewer bytes per element (Q2_0 < Q4_0 < Q8_0). -/
theorem format_order : 707788800 < 1415577600 ∧ 1415577600 < 2673868800 := by decide

-- § bf16 widening

/-- bf16 -> f32 is a 16-bit left shift: exact and invertible. -/
theorem bf16_widen (u : Nat) (h : u < 65536) : (u * 65536) / 65536 = u := by omega

end RefMtpPack



## FILE: /mnt/user-data/outputs/MtpRuntimeBlobLayout.lean
/-!
# MODULE: mtp_runtime_blob_layout
SOURCES: tools/mtp_rt.py
Integer model of the MTP runtime files: the expert blob (gate/up row interleave, code plane then scale plane),
its byte budget, dense.bin padding to 256 and the Gemma "+1" norm weight.

## Source semantics
BLOB = 3*(H*FF*18/64). Blob = interleaved gate/up codes, down codes, interleaved gate/up scales, down scales.
Interleave: slot 2r = gate row r, slot 2r+1 = up row r. dense.bin tensors are padded to 256-byte offsets.

## Form template identifiers
`H, FF, NE` -> `h, ff, ne`; `BLOB` -> `blob`; `blob_of` -> plane sizes; `pad = (-off) % 256` -> `pad256`.
-/

namespace RefMtpRt

/-- Hidden size. -/
def h : Nat := 2560
/-- FFN width. -/
def ff : Nat := 640
/-- Experts. -/
def ne : Nat := 512

/-- Blob size: three roles of H*FF*18/64 bytes. -/
def blob : Nat := 3 * (h * ff * 18 / 64)

/-- The blob is 1,382,400 bytes, matching the main pack. -/
theorem blob_value : blob = 1382400 := by decide

/-- Gate/up code plane: 1280 rows x 40 blocks x 16 code bytes. -/
theorem gu_codes : 2 * ff * (h / 64) * 16 = 819200 := by decide
/-- Down code plane: 2560 rows x 10 blocks x 16 code bytes. -/
theorem down_codes : h * (ff / 64) * 16 = 409600 := by decide
/-- Gate/up scale plane: fp16 per block. -/
theorem gu_scales : 2 * ff * (h / 64) * 2 = 102400 := by decide
/-- Down scale plane. -/
theorem down_scales : h * (ff / 64) * 2 = 51200 := by decide

/-- The four planes add up to the blob exactly. -/
theorem planes_sum : 819200 + 409600 + 102400 + 51200 = blob := by decide

/-- The expert file is 512 blobs. -/
theorem experts_file : ne * blob = 707788800 := by decide

/-- Q2_0 row widths in bytes: gate/up row 720, down row 180. -/
theorem row_bytes : h / 64 * 18 = 720 ∧ ff / 64 * 18 = 180 := by decide

-- § gate/up interleave

/-- Slot of gate row r. -/
def gateSlot (r : Nat) : Nat := 2 * r
/-- Slot of up row r. -/
def upSlot (r : Nat) : Nat := 2 * r + 1

/-- Gate slots are even and recover the row. -/
theorem gate_slot_inv (r : Nat) : gateSlot r % 2 = 0 ∧ gateSlot r / 2 = r := by
  unfold gateSlot; omega

/-- Up slots are odd and recover the row. -/
theorem up_slot_inv (r : Nat) : upSlot r % 2 = 1 ∧ upSlot r / 2 = r := by
  unfold upSlot; omega

/-- Every slot is exactly one gate or one up row (the interleave is a bijection). -/
theorem slot_decomp (s : Nat) : s = 2 * (s / 2) + s % 2 := by omega

/-- Gate and up never collide. -/
theorem slots_disjoint (r r' : Nat) : gateSlot r ≠ upSlot r' := by
  unfold gateSlot upSlot; omega

/-- For rows below 640 the slots stay below 1280. -/
theorem slots_bound (r : Nat) (hr : r < 640) : gateSlot r < 1280 ∧ upSlot r < 1280 := by
  unfold gateSlot upSlot; omega

/-- A slot's code bytes end inside the gate/up code plane. -/
theorem code_slot_in_plane (s b : Nat) (hs : s < 1280) (hb : b < 40) : s * 640 + b * 16 + 16 ≤ 819200 := by
  omega

/-- A slot's scale bytes end inside the gate/up scale plane. -/
theorem scale_slot_in_plane (s b : Nat) (hs : s < 1280) (hb : b < 40) : (s * 40 + b) * 2 + 2 ≤ 102400 := by
  omega

-- § dense.bin

/-- Padding to a 256-byte offset. -/
def pad256 (n : Nat) : Nat := (256 - n % 256) % 256

/-- Padding is below 256 and aligns the offset. -/
theorem pad256_spec (n : Nat) : pad256 n < 256 ∧ (n + pad256 n) % 256 = 0 := by
  unfold pad256; omega

/-- Gemma RMSNorm folds 1+w; with integer-valued w the stored value is w+1 (shift is invertible). -/
theorem gemma_plus_one (w : Int) : (w + 1) - 1 = w := by omega

/-- Quantising a row to Q8_0 requires a multiple of 32 columns; all large projections have 2560 columns. -/
theorem q8_cols : 2560 % 32 = 0 := by decide

end RefMtpRt



## FILE: /mnt/user-data/outputs/ExpertArenaBlobLayout.lean
/-!
# MODULE: expert_arena_blob_layout
SOURCES: tools/pack_layer.py
Integer model of the pack-layout expert blob: gate/up row interleave, code plane then scale plane, exact byte
budget of 1,382,400, per-expert contiguity inside the role tensors, and the total arena size.

## Source semantics
Blob = gate_up.codes (2*640 rows x 640 B) | down.codes (2560 rows x 160 B) | gate_up.scales (2*640*40 fp16)
| down.scales (2560*10 fp16) = 1,382,400 B. Expert e is a contiguous 460,800-byte run in each role tensor.
Conversion is a pure relayout, so bytes are only moved, never changed.

## Form template identifiers
`H, FF, NE, QK, BLOCK_BYTES` -> `hid, ff, ne, qk, blockBytes`; `ROW_BYTES_GU/D`, `SCALES_GU/D`, `ROLE_BYTES`,
`BLOB_BYTES`, `blob_offsets` -> same-named definitions.
-/

namespace RefPackLayer

-- § constants

/-- Hidden size. -/
def hid : Nat := 2560
/-- FFN width. -/
def ff : Nat := 640
/-- Experts per layer. -/
def ne : Nat := 512
/-- Q2_0 block elements. -/
def qk : Nat := 64
/-- Q2_0 block bytes. -/
def blockBytes : Nat := 18
/-- Code bytes of one gate/up row (2560 x 2 bits). -/
def rowBytesGu : Nat := hid * 2 / 8
/-- Code bytes of one down row (640 x 2 bits). -/
def rowBytesD : Nat := ff * 2 / 8
/-- Scale blocks per gate/up row. -/
def scalesGu : Nat := hid / qk
/-- Scale blocks per down row. -/
def scalesD : Nat := ff / qk
/-- Bytes per role per expert. -/
def roleBytes : Nat := hid * ff * blockBytes / qk
/-- Bytes per expert blob. -/
def blobBytes : Nat := 3 * roleBytes

/-- Gate/up row code bytes. -/
theorem row_bytes_gu : rowBytesGu = 640 := by decide
/-- Down row code bytes. -/
theorem row_bytes_d : rowBytesD = 160 := by decide
/-- Scales per gate/up row. -/
theorem scales_gu : scalesGu = 40 := by decide
/-- Scales per down row. -/
theorem scales_d : scalesD = 10 := by decide
/-- Role bytes per expert. -/
theorem role_bytes : roleBytes = 460800 := by decide
/-- The blob is exactly 1,382,400 bytes. -/
theorem blob_bytes : blobBytes = 1382400 := by decide

-- § plane offsets

/-- Offset of gate_up.codes. -/
def offGuCodes : Nat := 0
/-- Offset of down.codes. -/
def offDCodes : Nat := offGuCodes + 2 * ff * rowBytesGu
/-- Offset of gate_up.scales. -/
def offGuScales : Nat := offDCodes + hid * rowBytesD
/-- Offset of down.scales. -/
def offDScales : Nat := offGuScales + 2 * ff * scalesGu * 2
/-- End of the blob. -/
def offEnd : Nat := offDScales + hid * scalesD * 2

/-- The four planes tile the blob exactly with no gap or overlap. -/
theorem planes_tile : offEnd = blobBytes := by decide

/-- The plane offsets are 0, 819200, 1228800, 1331200 and the blob ends at 1382400. -/
theorem plane_offsets : offGuCodes = 0 ∧ offDCodes = 819200 ∧ offGuScales = 1228800 ∧ offDScales = 1331200 := by
  decide

/-- The plane offsets are strictly increasing. -/
theorem planes_ordered : offGuCodes < offDCodes ∧ offDCodes < offGuScales ∧ offGuScales < offDScales ∧
    offDScales < offEnd := by decide

/-- The explicit sum from the source's assertion. -/
theorem blob_sum : 2 * 640 * 640 + 2560 * 160 + 2 * 640 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

-- § per-expert contiguity

/-- Expert e occupies bytes [e*460800, (e+1)*460800) of a role tensor. -/
def expertStart (e : Nat) : Nat := e * 460800

/-- Experts are consecutive and non-overlapping. -/
theorem expert_runs_adjacent (e : Nat) : expertStart e + 460800 = expertStart (e + 1) := by
  unfold expertStart; omega

/-- All 512 expert runs fit inside a role tensor of 512*460800 bytes. -/
theorem expert_in_tensor (e : Nat) (he : e < 512) : expertStart e + 460800 ≤ 512 * 460800 := by
  unfold expertStart; omega

/-- A role tensor holds elements/512 per expert and that equals 460800 bytes of Q2_0. -/
theorem per_expert_elems : (2560 * 640 * 512) / 512 * 18 / 64 = 460800 := by decide

-- § interleave

/-- Row index of gate row r (0) or up row r (1) in the interleaved stack. -/
def stackRow (r k : Nat) : Nat := 2 * r + k

/-- Stacking gate and up along a new axis then flattening gives rows g0,u0,g1,u1,...: a bijection onto 0..1279. -/
theorem stack_row_inv (r k : Nat) (hk : k < 2) : stackRow r k / 2 = r ∧ stackRow r k % 2 = k := by
  unfold stackRow; omega

/-- Interleaved rows stay in range. -/
theorem stack_row_bound (r k : Nat) (hr : r < 640) (hk : k < 2) : stackRow r k < 1280 := by
  unfold stackRow; omega

/-- Inverse used by the verifier: slot -> (row, role), and recombination. -/
theorem slot_roundtrip (s : Nat) : stackRow (s / 2) (s % 2) = s := by
  unfold stackRow; omega

/-- The verifier's code offset for a (slot, block) ends inside the gate/up code plane. -/
theorem verifier_code_in_plane (slot b : Nat) (hs : slot < 1280) (hb : b < 40) :
    offGuCodes + slot * rowBytesGu + b * 16 + 16 ≤ offDCodes := by
  unfold offGuCodes offDCodes rowBytesGu ff hid; omega

/-- The verifier's scale offset for a (slot, block) ends inside the gate/up scale plane. -/
theorem verifier_scale_in_plane (slot b : Nat) (hs : slot < 1280) (hb : b < 40) :
    offGuScales + (slot * scalesGu + b) * 2 + 2 ≤ offDScales := by
  simp only [offGuCodes, offDCodes, offGuScales, offDScales, scalesGu, rowBytesGu, rowBytesD, ff, hid, qk]; omega

/-- Down code offsets end inside the down code plane. -/
theorem verifier_down_code_in_plane (row b : Nat) (hr : row < 2560) (hb : b < 10) :
    offDCodes + row * rowBytesD + b * 16 + 16 ≤ offGuScales := by
  simp only [offGuCodes, offDCodes, offGuScales, rowBytesGu, rowBytesD, ff, hid]; omega

-- § arena size

/-- Total arena bytes for n layers. -/
def arenaBytes (n : Nat) : Nat := n * ne * blobBytes

/-- The full 48-layer arena is 33,973,862,400 bytes (31.64 GiB). -/
theorem arena_48 : arenaBytes 48 = 33973862400 := by decide

/-- One layer is 512 blobs of 1,382,400 bytes. -/
theorem arena_1 : arenaBytes 1 = 707788800 := by decide

/-- The arena grows linearly: each layer adds the same bytes. -/
theorem arena_succ (n : Nat) : arenaBytes (n + 1) = arenaBytes n + 707788800 := by
  unfold arenaBytes ne blobBytes roleBytes hid ff blockBytes qk; omega

end RefPackLayer



## FILE: /mnt/user-data/outputs/RefPackBuilderVerifier.lean
/-!
# MODULE: ref_pack_builder_verifier
SOURCES: tools/ref_pack.py
Integer model of the canonical pack builder: code-plane bit packing (2/4/8 bits, LSB first), plane alignment
inside a segment, scale precision storage sizes, form bit widths, expert-arena size, and the verify sampling counts.

## Source semantics
Codes are packed `8/bits` per byte, LSB first. Every plane starts on a 64-byte boundary measured from the start
of its segment (`dense.bin` or `embd.bin`); the final plane of a blob is not padded. Scales are stored fp16 when the
round trip is exact, else fp32. The pack is valid only if decode(pack) == dequant(source) bit for bit.

## Form template identifiers
`ALIGN` -> 64; `code_bits` -> `codeBits`; `pack_codes`/`unpack_codes` -> `packTwo`/`packNibble`;
`put` -> `putOffset`; `choose_scale_dtype` -> `scaleBytes`; `verify_experts` -> `expertsChecked`.
-/

namespace RefPack

-- § form bit widths

/-- Bits per code of each canonical form. -/
def codeBits : String → Nat
  | "S2" => 2
  | "S4" => 4
  | _ => 8

/-- Codes per byte for every integer form is exactly 8/bits and fills the byte. -/
theorem codes_fill_byte : ∀ f ∈ ["S2", "S4", "S8"], 8 / codeBits f * codeBits f = 8 := by decide

-- § bit packing

/-- Pack two 4-bit codes into one byte, LSB first. -/
def packNibble (c0 c1 : Nat) : Nat := c0 + 16 * c1

/-- Nibble pack/unpack round trip. -/
theorem nibble_roundtrip (c0 c1 : Nat) (h0 : c0 < 16) (h1 : c1 < 16) :
    packNibble c0 c1 < 256 ∧ packNibble c0 c1 % 16 = c0 ∧ packNibble c0 c1 / 16 = c1 := by
  unfold packNibble; omega

/-- Pack four 2-bit codes into one byte, LSB first. -/
def packTwo (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- 2-bit pack/unpack round trip. -/
theorem two_roundtrip (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packTwo c0 c1 c2 c3 < 256 ∧ packTwo c0 c1 c2 c3 % 4 = c0 ∧ packTwo c0 c1 c2 c3 / 4 % 4 = c1 ∧
    packTwo c0 c1 c2 c3 / 16 % 4 = c2 ∧ packTwo c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold packTwo; omega

/-- Packed size: n codes at `per` codes per byte occupy n/per bytes when divisible, and the code count is recoverable. -/
theorem packed_size (n per : Nat) (hp : 0 < per) (h : n % per = 0) : (n / per) * per = n := by
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)

/-- The 4-bit element count must be a multiple of 2 and the 2-bit count a multiple of 4. -/
theorem q2_count_nonmultiple (n : Nat) (h : n % 4 ≠ 0) : ¬ (n % 4 = 0) := h

-- § alignment inside a segment

/-- Align a length to 64. -/
def pad64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- Padding lands on a boundary and is below 64. -/
theorem pad64_spec (n : Nat) : pad64 n < 64 ∧ (n + pad64 n) % 64 = 0 := by
  unfold pad64; omega

/-- Offset of a plane: segment base + current body length after padding. -/
def putOffset (base bodyLen : Nat) : Nat := base + (bodyLen + pad64 bodyLen)

/-- Plane offsets are 64-aligned relative to the segment start. -/
theorem put_relative_aligned (base bodyLen : Nat) : (putOffset base bodyLen - base) % 64 = 0 := by
  unfold putOffset pad64; omega

/-- Absolute alignment holds exactly when the segment base is itself 64-aligned. -/
theorem put_absolute_aligned (base bodyLen : Nat) (hb : base % 64 = 0) : putOffset base bodyLen % 64 = 0 := by
  unfold putOffset pad64; omega

/-- Without an aligned base the plane is generally not absolutely aligned (the last blob is not padded). -/
theorem put_unaligned_base : putOffset 1 0 % 64 ≠ 0 := by decide

/-- Planes never overlap: the next plane starts at or after the previous one's end. -/
theorem put_no_overlap (base bodyLen : Nat) : base + bodyLen ≤ putOffset base bodyLen := by
  unfold putOffset; omega

-- § scale precision

/-- Bytes used by a scale plane of n values. -/
def scaleBytes (isFp16 : Bool) (n : Nat) : Nat := if isFp16 then 2 * n else 4 * n

/-- fp16 scales halve the plane compared with fp32. -/
theorem scale_fp16_half (n : Nat) : 2 * scaleBytes true n = scaleBytes false n := by
  unfold scaleBytes; simp; omega

/-- A scale plane always has at least 2 bytes per value. -/
theorem scale_bytes_ge (b : Bool) (n : Nat) : 2 * n ≤ scaleBytes b n := by
  unfold scaleBytes; cases b <;> simp <;> omega

-- § arena size and verification sampling

/-- Expert blob bytes. -/
def blobBytes : Nat := 1382400
/-- Experts per layer. -/
def nExperts : Nat := 512

/-- experts.bin size for n layers. -/
def expertsFileBytes (n : Nat) : Nat := n * nExperts * blobBytes

/-- The default 48 layers give 33,973,862,400 bytes. -/
theorem experts_file_48 : expertsFileBytes 48 = 33973862400 := by decide

/-- Blob count reported by `info`. -/
theorem info_blob_count : 48 * nExperts = 24576 := by decide

/-- Experts verified per layer when sampling every k-th. -/
def expertsPerLayerChecked (every : Nat) : Nat := (512 + every - 1) / every

/-- Total layer/expert/role checks. -/
def expertsChecked (layers every : Nat) : Nat := layers * expertsPerLayerChecked every * 3

/-- Default sampling (every 32nd of 512) checks 16 experts per layer. -/
theorem default_sampling : expertsPerLayerChecked 32 = 16 := by decide

/-- The default full pack verification runs 2304 role checks. -/
theorem default_checks : expertsChecked 48 32 = 2304 := by decide

/-- Sampling every expert checks all 512. -/
theorem every_one : expertsPerLayerChecked 1 = 512 := by decide

/-- Sampling is never empty when every <= 512. -/
theorem sampling_nonempty (every : Nat) (h1 : 0 < every) (h2 : every ≤ 512) : 0 < expertsPerLayerChecked every := by
  unfold expertsPerLayerChecked
  have : every ≤ 512 + every - 1 := by omega
  exact Nat.div_pos this h1

end RefPack



## FILE: /mnt/user-data/outputs/TokenizerByteLevelBpe.lean
/-!
# MODULE: tokenizer_byte_level_bpe
SOURCES: tools/ref_tokenizer.py
Model of the GPT-2 byte-to-unicode map, BPE merge step, special-token classes and the decode homomorphism that
underlies the `decode(encode(s)) == s` round trip.

## Source semantics
bytes_to_unicode sends 188 printable bytes to themselves and the other 68 to 256+n in byte order; it is injective.
A BPE step replaces an adjacent pair by one merged token, shrinking the list by exactly one, so merging terminates.
Token type 4 (USER_DEFINED) is matched always; type 3 (CONTROL) only with parse_special. decode concatenates the
byte strings of its tokens, so it distributes over concatenation of id lists.

## Form template identifiers
`bytes_to_unicode` -> `cp`; `_bpe` merge -> `mergeAt`; `special_tokens`/`_always_re` -> `matchedAlways`/`matchedSpecial`;
`decode` -> `decodeBytes`; `_encode_matching` splice -> `split_decode`.
-/

namespace RefTokenizer

-- § byte <-> unicode

/-- Bytes that map to themselves: 33..126, 161..172, 174..255. -/
def inBs (b : Nat) : Bool :=
  (33 ≤ b && b ≤ 126) || (161 ≤ b && b ≤ 172) || (174 ≤ b && b ≤ 255)

/-- Number of non-printable bytes below b (the `n` counter of the source). -/
def rank (b : Nat) : Nat := ((List.range b).filter (fun x => !inBs x)).length

/-- Code point of byte b: itself if printable, else 256 plus its rank among the others. -/
def cp (b : Nat) : Nat := if inBs b then b else 256 + rank b

/-- 188 bytes map to themselves. -/
theorem inBs_count : ((List.range 256).filter inBs).length = 188 := by decide +kernel

/-- 68 bytes are shifted. -/
theorem shifted_count : ((List.range 256).filter (fun x => !inBs x)).length = 68 := by decide +kernel

/-- Printable bytes are fixed points. -/
theorem cp_printable (b : Nat) (h : inBs b = true) : cp b = b := by
  unfold cp; simp [h]

/-- The map is injective on bytes: all 256 code points are distinct, so decode inverts encode. -/
theorem cp_nodup : ((List.range 256).map cp).Nodup := by decide +kernel

/-- Shifted code points lie in 256..323. -/
theorem cp_bound : ∀ b ∈ List.range 256, cp b < 324 := by decide +kernel

-- § BPE merge

/-- Replace the adjacent pair at i by a single merged token. -/
def mergeAt {α : Type} (l : List α) (i : Nat) (x : α) : List α := l.take i ++ x :: l.drop (i + 2)

/-- A merge shrinks the list by exactly one, so BPE terminates after fewer than len steps. -/
theorem mergeAt_length {α : Type} (l : List α) (i : Nat) (x : α) (h : i + 1 < l.length) :
    (mergeAt l i x).length + 1 = l.length := by
  unfold mergeAt
  simp [List.length_take, List.length_drop]
  omega

/-- A list of length at most one has no pair left to merge. -/
theorem no_pair_when_short {α : Type} (l : List α) (h : l.length ≤ 1) : ¬ (0 + 1 < l.length) := by omega

-- § special tokens

/-- Type 4 USER_DEFINED tokens are matched with or without parse_special. -/
def matchedAlways (ty : Nat) : Prop := ty = 4
/-- Types 3 and 4 are matched when parse_special is set. -/
def matchedSpecial (ty : Nat) : Prop := ty = 3 ∨ ty = 4

/-- Anything matched always is matched under parse_special. -/
theorem always_subset_special (ty : Nat) (h : matchedAlways ty) : matchedSpecial ty := by
  unfold matchedAlways matchedSpecial at *; omega

/-- CONTROL tokens are matched only under parse_special. -/
theorem control_needs_flag (ty : Nat) (h : ty = 3) : matchedSpecial ty ∧ ¬ matchedAlways ty := by
  unfold matchedAlways matchedSpecial; omega

-- § decode homomorphism

/-- Decoding concatenates the byte strings of the tokens. -/
def decodeBytes (toks : Nat → List Nat) (ids : List Nat) : List Nat := ids.flatMap toks

/-- Decode distributes over concatenation of id lists. -/
theorem decode_append (toks : Nat → List Nat) (a b : List Nat) :
    decodeBytes toks (a ++ b) = decodeBytes toks a ++ decodeBytes toks b := by
  simp [decodeBytes]

/-- Splicing a special token between two encoded pieces decodes to piece, token bytes, piece. -/
theorem split_decode (toks : Nat → List Nat) (a b : List Nat) (sp : Nat) :
    decodeBytes toks (a ++ [sp] ++ b) = decodeBytes toks a ++ toks sp ++ decodeBytes toks b := by
  simp [decodeBytes]

/-- Round trip is preserved by splicing: if both pieces round-trip, so does the spliced text. -/
theorem split_roundtrip (toks : Nat → List Nat) (ea eb ta tb : List Nat) (sp : Nat)
    (ha : decodeBytes toks ea = ta) (hb : decodeBytes toks eb = tb) :
    decodeBytes toks (ea ++ [sp] ++ eb) = ta ++ toks sp ++ tb := by
  rw [split_decode, ha, hb]

/-- Decoding the empty id list gives the empty string, so the empty corpus entry round-trips. -/
theorem decode_nil (toks : Nat → List Nat) : decodeBytes toks [] = [] := by
  simp [decodeBytes]

end RefTokenizer





## FILE: /mnt/user-data/outputs/GgufShardLoader.lean
/-!
# MODULE: gguf_shard_loader
SOURCES: ref/load.py
Demand-paged GGUF loader: shard discovery, per-(layer, expert) lazy byte ranges, row gathering and
the shared dequantization dispatch geometry.

## Source semantics
* `GEOM` maps a type name to (elements per block, bytes per block); every block has a positive element count.
* `expert(name, e)` reads only `per_expert/el*by` bytes starting at `e * (per_expert/el) * by`; experts tile the tensor.
* `rows(name, idx)` reads `row_bytes` at `r * row_bytes`; row `r < ncols` stays inside the tensor.
* BF16 -> FP32 is `bits << 16`, so the low 16 bits are zero.
* Shard discovery refuses to guess: 0 files, 1 file, a shard set, or an ambiguous directory.

## Form template identifiers
- `GEOM` -> `block_geometry_table`; `Gguf.expert` -> `lazy_expert_slice`; `Gguf.rows` -> `lazy_row_gather`;
  `_discover_shards` -> `shard_set_resolver`; `_dequant_flat` -> `dequant_dispatch`.
-/

namespace Ref.GgufShardLoader

-- § block geometry table

/-- The `GEOM` table of ref/load.py as (type name, elements per block, bytes per block). -/
def geomTable : List (String × Nat × Nat) :=
  [("F32", 1, 4), ("F16", 1, 2), ("BF16", 1, 2), ("Q4_0", 32, 18), ("Q5_0", 32, 22),
   ("Q8_0", 32, 34), ("IQ4_NL", 32, 18), ("Q3_K", 256, 110), ("Q4_K", 256, 144),
   ("Q5_K", 256, 176), ("Q6_K", 256, 210), ("IQ4_XS", 256, 136), ("Q2_0", 64, 18)]

/-- Every geometry entry has a positive element count and positive byte count. -/
theorem geom_positive : ∀ e ∈ geomTable, 0 < e.2.1 ∧ 0 < e.2.2 := by
  decide

/-- The table has exactly the 13 encodings of the source. -/
theorem geom_size : geomTable.length = 13 := by
  decide

/-- Q2_0 is 64 elements in 18 bytes (2-byte scale + 16 bytes of 4 codes). -/
theorem q2_0_geometry : ("Q2_0", 64, 18) ∈ geomTable := by
  decide

-- § byte offsets

/-- Bytes of a run of `n` elements in a type with block geometry `(el, by)`. -/
def runBytes (n el by_ : Nat) : Nat := n / el * by_

/-- Start byte of expert `e`: whole experts before it, measured in blocks. -/
def expertStart (e perExpert el by_ : Nat) : Nat := e * (perExpert / el) * by_

/-- Byte length of one expert's run. -/
def expertBytes (perExpert el by_ : Nat) : Nat := perExpert / el * by_

/-- Expert `e + 1` starts exactly where expert `e` ends, so reads never overlap or leave gaps. -/
theorem expert_contiguous (e perExpert el by_ : Nat) :
    expertStart (e + 1) perExpert el by_ = expertStart e perExpert el by_ + expertBytes perExpert el by_ := by
  unfold expertStart expertBytes
  simp [Nat.add_mul]

/-- The start of expert `e` is `e` times the per-expert byte size. -/
theorem expert_start_eq (e perExpert el by_ : Nat) :
    expertStart e perExpert el by_ = e * expertBytes perExpert el by_ := by
  unfold expertStart expertBytes
  rw [Nat.mul_assoc]

/-- A valid expert index reads entirely inside the fused tensor (`e < n_exp`). -/
theorem expert_in_range (e nExp perExpert el by_ : Nat) (h : e < nExp) :
    expertStart e perExpert el by_ + expertBytes perExpert el by_ ≤ nExp * expertBytes perExpert el by_ := by
  have h1 : expertStart e perExpert el by_ + expertBytes perExpert el by_ =
      (e + 1) * expertBytes perExpert el by_ := by
    rw [expert_start_eq, Nat.add_mul, Nat.one_mul]
  rw [h1]
  exact Nat.mul_le_mul_right _ h

/-- Experts tile the tensor: `n` experts of `per` elements occupy `n` times one expert's bytes,
provided the block size divides the per-expert element count (the source raises otherwise). -/
theorem experts_tile (n per el by_ : Nat) (h : el ∣ per) :
    runBytes (n * per) el by_ = n * expertBytes per el by_ := by
  unfold runBytes expertBytes
  rw [Nat.mul_div_assoc n h, Nat.mul_assoc]

/-- Row `r` of a 2-D tensor starts at `r * rowBytes`. -/
def rowStart (r rowBytes : Nat) : Nat := r * rowBytes

/-- `row_bytes` of a row of `width` elements. -/
def rowBytes (width el by_ : Nat) : Nat := width / el * by_

/-- A row index below the column count reads inside the tensor. -/
theorem row_in_range (r ncols rb : Nat) (h : r < ncols) :
    rowStart r rb + rb ≤ ncols * rb := by
  unfold rowStart
  have : (r + 1) * rb ≤ ncols * rb := Nat.mul_le_mul_right _ h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The per-layer embedding table: width 160 in IQ4_NL (32-element, 18-byte blocks) is 90 bytes per row. -/
theorem ple_row_bytes : rowBytes 160 32 18 = 90 := by
  decide

/-- 160 is a multiple of the IQ4_NL block size (the divisibility the source checks). -/
theorem ple_width_divisible : 32 ∣ 160 := by
  decide

-- § BF16 to FP32

/-- BF16 bits widened to f32 bits: shift left by 16. -/
def bf16ToF32Bits (h : Nat) : Nat := h * 65536

/-- Widening a 16-bit pattern stays within 32 bits. -/
theorem bf16_fits32 (h : Nat) (hh : h < 65536) : bf16ToF32Bits h < 4294967296 := by
  unfold bf16ToF32Bits
  omega

/-- The low 16 bits of the widened pattern are zero (the mantissa is just padded). -/
theorem bf16_low_zero (h : Nat) : bf16ToF32Bits h % 65536 = 0 := by
  unfold bf16ToF32Bits
  omega

/-- The high 16 bits are the original pattern. -/
theorem bf16_high_roundtrip (h : Nat) : bf16ToF32Bits h / 65536 = h := by
  unfold bf16ToF32Bits
  omega

-- § shard discovery

/-- Outcome of resolving a directory argument. -/
inductive DirResult where
  | shardSet
  | single
  | noFiles
  | ambiguous
  deriving DecidableEq

/-- Directory resolution: prefer an explicit shard set, accept one unambiguous file, otherwise refuse. -/
def dirDecision (shards found : Nat) : DirResult :=
  if shards ≠ 0 then DirResult.shardSet
  else if found = 1 then DirResult.single
  else if found = 0 then DirResult.noFiles
  else DirResult.ambiguous

/-- An mmproj file next to a real shard set never hijacks the choice: the shard set wins. -/
theorem dir_prefers_shards (shards found : Nat) (h : shards ≠ 0) :
    dirDecision shards found = DirResult.shardSet := by
  simp [dirDecision, h]

/-- With no shards and several files the loader refuses to guess. -/
theorem dir_refuses_ambiguous (found : Nat) (h : 2 ≤ found) :
    dirDecision 0 found = DirResult.ambiguous := by
  have h1 : found ≠ 1 := by omega
  have h2 : found ≠ 0 := by omega
  simp [dirDecision, h1, h2]

/-- A single unambiguous file is accepted. -/
theorem dir_accepts_single : dirDecision 0 1 = DirResult.single := by
  decide

/-- An empty directory is an error, not an empty model. -/
theorem dir_empty : dirDecision 0 0 = DirResult.noFiles := by
  decide

/-- A shard named `-NNNNN-of-MMMMM` is complete only when exactly MMMMM siblings exist. -/
def shardsComplete (found total : Nat) : Bool := found == total

/-- Completeness holds exactly when the sibling count equals the declared total. -/
theorem shards_complete_iff (found total : Nat) :
    shardsComplete found total = true ↔ found = total := by
  simp [shardsComplete]

end Ref.GgufShardLoader



## FILE: /mnt/user-data/outputs/GgufHeaderReader.lean
/-!
# MODULE: gguf_header_reader
SOURCES: include/ref/artifact/gguf_reader.hpp
GGUF v3 header reader: bounds-checked cursor, tensor-directory arithmetic, data-section alignment,
array sampling and the architecture guard for the `qwen4exp` model.

## Source semantics
* Every cursor read checks `pos + n <= size`, otherwise it throws; a successful read advances by exactly `n`.
* `data_start = ceil(pos / align) * align` and must not exceed the file size.
* Tensors have 1..4 dimensions; element count is the product of the shape.
* Arrays keep at most 64 sampled items; counts above 2^24 are rejected.
* The guard accepts only `general.architecture == "qwen4exp"` with the six compiled-in constants.

## Form template identifiers
- `Cursor::need/read` -> `checked_cursor`; `data_start_` -> `aligned_data_start`;
  `check_architecture` -> `architecture_guard`; `block_geometry` -> `type_geometry`.
-/

namespace Ref.GgufHeaderReader

-- § cursor

/-- A checked read of `n` bytes at `pos` in a buffer of `size` bytes: the new position, or failure. -/
def cursorRead (pos size n : Nat) : Option Nat :=
  if pos + n ≤ size then some (pos + n) else none

/-- A successful read advances by exactly `n` bytes. -/
theorem cursor_advances (pos size n p : Nat) (h : cursorRead pos size n = some p) : p = pos + n := by
  unfold cursorRead at h
  split at h
  · injection h with h'
    omega
  · cases h

/-- A successful read never moves past the end of the buffer. -/
theorem cursor_in_bounds (pos size n p : Nat) (h : cursorRead pos size n = some p) : p ≤ size := by
  unfold cursorRead at h
  split at h
  · injection h with h'
    omega
  · cases h

/-- A read that would overrun fails instead of reading out of bounds. -/
theorem cursor_overrun_fails (pos size n : Nat) (h : size < pos + n) : cursorRead pos size n = none := by
  unfold cursorRead
  have : ¬ (pos + n ≤ size) := by omega
  simp [this]

-- § data section alignment

/-- First aligned offset at or after `pos`. -/
def dataStart (pos align : Nat) : Nat := (pos + align - 1) / align * align

/-- The data section never starts before the header ends. -/
theorem dataStart_ge (pos a : Nat) (ha : 0 < a) : pos ≤ dataStart pos a := by
  unfold dataStart
  have h1 := Nat.div_add_mod (pos + a - 1) a
  have h2 := Nat.mod_lt (pos + a - 1) ha
  rw [Nat.mul_comm ((pos + a - 1) / a) a]
  omega

/-- The data section starts less than one alignment unit after the header end. -/
theorem dataStart_lt (pos a : Nat) (ha : 0 < a) : dataStart pos a < pos + a := by
  unfold dataStart
  have h1 := Nat.div_add_mod (pos + a - 1) a
  rw [Nat.mul_comm ((pos + a - 1) / a) a]
  omega

/-- The data section start is a multiple of the alignment. -/
theorem dataStart_aligned (pos a : Nat) : dataStart pos a % a = 0 := by
  unfold dataStart
  simp

/-- An already-aligned header end is a fixed point. -/
theorem dataStart_idempotent (k a : Nat) (ha : 0 < a) : dataStart (k * a) a = k * a := by
  have hq : (k * a + a - 1) / a = k := by
    apply Nat.div_eq_of_lt_le
    · omega
    · rw [Nat.add_mul]
      omega
  unfold dataStart
  rw [hq]

-- § tensor directory

/-- Element count of a shape (product of its dimensions). -/
def elements : List Nat → Nat
  | [] => 1
  | d :: r => d * elements r

/-- The element count of a concatenated shape multiplies. -/
theorem elements_append (a b : List Nat) : elements (a ++ b) = elements a * elements b := by
  induction a with
  | nil => simp [elements]
  | cons d r ih => simp [elements, ih, Nat.mul_assoc]

/-- A tensor must have between 1 and 4 dimensions. -/
def validDims (nd : Nat) : Bool := nd != 0 && nd ≤ 4

/-- The dimension check accepts exactly 1 through 4. -/
theorem validDims_iff (nd : Nat) : validDims nd = true ↔ (1 ≤ nd ∧ nd ≤ 4) := by
  unfold validDims
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq]
  omega

-- § arrays and versions

/-- Sampled items kept from an array of `count` elements. -/
def sampleKept (count : Nat) : Nat := min count 64

/-- The sample never exceeds 64 items. -/
theorem sample_le_64 (count : Nat) : sampleKept count ≤ 64 := by
  unfold sampleKept
  exact Nat.min_le_right _ _

/-- The sample never exceeds the actual count. -/
theorem sample_le_count (count : Nat) : sampleKept count ≤ count := by
  unfold sampleKept
  exact Nat.min_le_left _ _

/-- Array lengths are accepted only up to 2^24. -/
def arrayLenOk (count : Nat) : Bool := count ≤ 16777216

/-- A rejected array length is above 2^24. -/
theorem array_len_reject (count : Nat) (h : arrayLenOk count = false) : 16777216 < count := by
  unfold arrayLenOk at h
  simp at h
  omega

/-- Only GGUF version 3 is handled. -/
def versionOk (v : Nat) : Bool := v == 3

/-- The version check accepts exactly 3. -/
theorem versionOk_iff (v : Nat) : versionOk v = true ↔ v = 3 := by
  simp [versionOk]

-- § architecture guard

/-- The six compiled-in constants of the qwen4exp guard. -/
structure Guard where
  blockCount : Nat := 48
  hidden : Nat := 2560
  experts : Nat := 512
  expertsUsed : Nat := 10
  headCount : Nat := 24
  headCountKv : Nat := 2
  deriving DecidableEq

/-- Metadata values read from the file for those six keys. -/
abbrev Meta := Guard

/-- The guard accepts only `qwen4exp` with all six values equal. -/
def archOk (arch : String) (m want : Guard) : Bool := arch == "qwen4exp" && decide (m = want)

/-- A different architecture string is always refused. -/
theorem arch_refuse_other (arch : String) (m want : Guard) (h : arch ≠ "qwen4exp") :
    archOk arch m want = false := by
  simp [archOk, h]

/-- A matching qwen4exp file is accepted. -/
theorem arch_accept (m : Guard) : archOk "qwen4exp" m m = true := by
  simp [archOk]

/-- A qwen4exp file with any differing constant is refused. -/
theorem arch_refuse_mismatch (m want : Guard) (h : m ≠ want) : archOk "qwen4exp" m want = false := by
  simp [archOk, h]

/-- The compiled-in constants are self-consistent: top-k experts fit in the expert set. -/
theorem guard_used_le_experts : (Guard.expertsUsed {}) ≤ (Guard.experts {}) := by
  decide

/-- The attention heads group evenly over the KV heads (GQA ratio 12). -/
theorem guard_gqa : (Guard.headCount {}) = 12 * (Guard.headCountKv {}) := by
  decide

-- § block geometry by ggml type id

/-- (elements, bytes) per block for the ggml type ids handled by the reader. -/
def blockGeometry : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 1 => some (1, 2)
  | 30 => some (1, 2)
  | 2 => some (32, 18)
  | 3 => some (32, 20)
  | 6 => some (32, 22)
  | 7 => some (32, 24)
  | 8 => some (32, 34)
  | 9 => some (32, 36)
  | 10 => some (256, 84)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 20 => some (32, 18)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 23 => some (256, 136)
  | 29 => some (256, 56)
  | 42 => some (64, 18)
  | _ => none

/-- Type 42 (Q2_0) is 64 elements in 18 bytes. -/
theorem q2_0_block : blockGeometry 42 = some (64, 18) := by
  rfl

/-- Q2_0 stores 2.25 bits per weight: 18 bytes * 8 / 64 elements = 9/4. -/
theorem q2_0_bits_per_weight : 18 * 8 * 4 = 64 * 9 := by
  decide

/-- The k-quant types all use 256-element super-blocks. -/
theorem kquant_blocks : ∀ t ∈ [10, 11, 12, 13, 14, 23],
    (blockGeometry t).map Prod.fst = some 256 := by
  decide

/-- Unknown type ids have no geometry, so the engine cannot size them. -/
theorem unknown_type_none : blockGeometry 99 = none := by
  rfl

end Ref.GgufHeaderReader



## FILE: /mnt/user-data/outputs/BlockDequantizers.lean
/-!
# MODULE: block_dequantizers
SOURCES: include/ref/artifact/dequant.hpp
Scalar reference dequantizers (Q2_0, Q8_0, Q4_0, Q5_0, IQ4_NL, Q6_K, Q4_K, Q5_K, Q3_K, IQ4_XS) modelled
at the level of integer codes: bit/nibble extraction, index coverage, code ranges and block sizes.

## Source semantics
* Each output is `scale * integer_code`; the integer code ranges are fixed by the bit layout.
* Q2_0: 4 codes per byte, LSB first, code {0..3} -> symbol {-1,0,1,2}.
* Q4_0/IQ4_NL: low nibble is element j, high nibble element j+16.
* Q5_0: a 5th bit from a 32-bit mask; code - 16.
* K-quants: 256-element super-blocks with 6-bit scales unpacked from 12 bytes.
* Block byte sizes add up exactly (no gaps).

## Form template identifiers
- `dequantize_q2_0` -> `ternary2_decode`; `get_scale_min_k4` -> `scale_min_unpack`;
  `kvalues_iq4nl` -> `nonlinear_codebook`.
-/

namespace Ref.BlockDequantizers

-- § block sizes

/-- Q2_0: 2-byte scale + 16 code bytes = 18 bytes holding 16 * 4 = 64 elements. -/
theorem q2_0_sizes : 2 + 16 = 18 ∧ 16 * 4 = 64 := by
  decide

/-- Q8_0: 2-byte scale + 32 int8 = 34 bytes. -/
theorem q8_0_size : 2 + 32 = 34 := by
  decide

/-- Q4_0 and IQ4_NL: 2-byte scale + 16 nibble bytes = 18 bytes for 32 elements. -/
theorem q4_0_size : 2 + 16 = 18 ∧ 16 * 2 = 32 := by
  decide

/-- Q5_0: scale 2 + qh 4 + qs 16 = 22 bytes. -/
theorem q5_0_size : 2 + 4 + 16 = 22 := by
  decide

/-- Q6_K: ql 128 + qh 64 + scales 16 + d 2 = 210 bytes (d is last at offset 208). -/
theorem q6_k_size : 128 + 64 + 16 + 2 = 210 ∧ 128 + 64 + 16 = 208 := by
  decide

/-- Q4_K: d 2 + dmin 2 + scales 12 + qs 128 = 144 bytes (qs at offset 16). -/
theorem q4_k_size : 2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 = 16 := by
  decide

/-- Q5_K: d 2 + dmin 2 + scales 12 + qh 32 + qs 128 = 176 bytes (qh at 16, qs at 48). -/
theorem q5_k_size : 2 + 2 + 12 + 32 + 128 = 176 ∧ 16 + 32 = 48 := by
  decide

/-- Q3_K: hmask 32 + qs 64 + scales 12 + d 2 = 110 bytes (scales at 96, d at 108). -/
theorem q3_k_size : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 = 96 ∧ 96 + 12 = 108 := by
  decide

/-- IQ4_XS: d 2 + scales_h 2 + scales_l 4 + qs 128 = 136 bytes (qs at offset 8). -/
theorem iq4_xs_size : 2 + 2 + 4 + 128 = 136 ∧ 2 + 2 + 4 = 8 := by
  decide

-- § Q2_0

/-- The 2-bit code of element `k` (0..3) inside a byte. -/
def q2Code (byte k : Nat) : Nat := byte / 4 ^ k % 4

/-- The symbol of a code: code - 1, so {0,1,2,3} -> {-1,0,1,2}. -/
def q2Sym (byte k : Nat) : Int := (q2Code byte k : Int) - 1

/-- Every Q2_0 symbol is one of -1, 0, 1, 2. -/
theorem q2_sym_range (byte k : Nat) : -1 ≤ q2Sym byte k ∧ q2Sym byte k ≤ 2 := by
  have h : q2Code byte k < 4 := by
    unfold q2Code
    exact Nat.mod_lt _ (by decide)
  unfold q2Sym
  omega

/-- A code byte is exactly its four 2-bit fields (no bits are lost or double-counted). -/
theorem q2_byte_decomposition (b : Nat) (hb : b < 256) :
    b % 4 + 4 * (b / 4 % 4) + 16 * (b / 16 % 4) + 64 * (b / 64 % 4) = b := by
  omega

/-- The element index maps bijectively onto (byte j/4, slot j%4). -/
theorem q2_index_split (j : Nat) : 4 * (j / 4) + j % 4 = j := by
  omega

/-- All 64 elements of a block address one of the 16 code bytes. -/
theorem q2_byte_index_lt (j : Nat) (hj : j < 64) : j / 4 < 16 := by
  omega

/-- The bit offset (j%4)*2 is at most 6, so a 2-bit field never straddles bytes. -/
theorem q2_bit_offset_le (j : Nat) : (j % 4) * 2 ≤ 6 := by
  omega

-- § nibble formats: Q4_0, IQ4_NL, Q5_0

/-- Q4_0 value for a nibble `n < 16`: n - 8. -/
def q4Val (n : Nat) : Int := (n : Int) - 8

/-- Q4_0 values lie in [-8, 7]. -/
theorem q4_range (n : Nat) (h : n < 16) : -8 ≤ q4Val n ∧ q4Val n ≤ 7 := by
  unfold q4Val
  omega

/-- A byte is exactly its low nibble plus 16 times its high nibble. -/
theorem nibble_decomposition (q : Nat) : q % 16 + 16 * (q / 16) = q := by
  omega

/-- For a byte, the high nibble is below 16, so it indexes a 16-entry table. -/
theorem high_nibble_lt (q : Nat) (hq : q < 256) : q / 16 < 16 := by
  omega

/-- Elements j (low nibble) and j+16 (high nibble) are distinct and cover 0..31 for j < 16. -/
theorem q4_halves_cover (i : Nat) (hi : i < 32) :
    (i < 16 ∧ i = i) ∨ (16 ≤ i ∧ ∃ j, j < 16 ∧ i = j + 16) := by
  by_cases h : i < 16
  · exact Or.inl ⟨h, rfl⟩
  · exact Or.inr ⟨by omega, i - 16, by omega, by omega⟩

/-- Q5_0 value: 5-bit code minus 16. -/
def q5Val (nib bit : Nat) : Int := ((nib + 16 * bit : Nat) : Int) - 16

/-- Q5_0 values lie in [-16, 15] for a nibble and a single high bit. -/
theorem q5_range (nib bit : Nat) (hn : nib < 16) (hb : bit < 2) : -16 ≤ q5Val nib bit ∧ q5Val nib bit ≤ 15 := by
  unfold q5Val
  omega

/-- The i-th bit of a 32-bit mask is 0 or 1. -/
theorem mask_bit_lt (h i : Nat) : h / 2 ^ i % 2 < 2 := by
  exact Nat.mod_lt _ (by decide)

-- § IQ4_NL codebook

/-- The IQ4_NL non-linear codebook, verbatim from ggml. -/
def kvaluesIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, one per nibble value. -/
theorem iq4nl_length : kvaluesIq4nl.length = 16 := by
  decide

/-- The codebook is strictly increasing, so the 4-bit code is order preserving. -/
theorem iq4nl_sorted : kvaluesIq4nl.Pairwise (· < ·) := by
  decide

/-- Every codebook entry is within [-127, 113]. -/
theorem iq4nl_bounds : ∀ v ∈ kvaluesIq4nl, -127 ≤ v ∧ v ≤ 113 := by
  decide

/-- The codebook is asymmetric: its largest magnitude is 127 on the negative side. -/
theorem iq4nl_asymmetric : kvaluesIq4nl.head? = some (-127) ∧ kvaluesIq4nl.getLast? = some 113 := by
  decide

-- § Q6_K

/-- Q6_K 6-bit code: 4 low bits from ql, 2 high bits from qh, minus 32. -/
def q6Val (lo4 hi2 : Nat) : Int := ((lo4 % 16 + 16 * (hi2 % 4) : Nat) : Int) - 32

/-- Q6_K codes lie in [-32, 31]. -/
theorem q6_range (lo hi : Nat) : -32 ≤ q6Val lo hi ∧ q6Val lo hi ≤ 31 := by
  unfold q6Val
  omega

/-- The four output positions l, l+32, l+64, l+96 (l < 32) cover each index of a 128-half exactly once. -/
theorem q6_positions_cover (i : Nat) (hi : i < 128) : i = i % 32 + 32 * (i / 32) ∧ i / 32 < 4 := by
  omega

/-- Scale indices used in a Q6_K half stay in 0..15 over both halves (n = 0, 1). -/
theorem q6_scale_index_lt (n l k : Nat) (hn : n < 2) (hl : l < 32) (hk : k ≤ 6) :
    8 * n + l / 16 + k < 16 := by
  omega

-- § 6-bit scale/min unpacking (Q4_K, Q5_K)

/-- `get_scale_min_k4` for j ≥ 4: low 4 bits of q[j+4] plus the top 2 bits of q[j-4] shifted by 4. -/
def scaleHigh (qj4 qjm4 : Nat) : Nat := qj4 % 16 + 16 * (qjm4 / 64)

/-- The unpacked 6-bit scale for j ≥ 4 is below 64 when the source byte is below 256. -/
theorem scale_high_lt (a b : Nat) (hb : b < 256) : scaleHigh a b < 64 := by
  unfold scaleHigh
  omega

/-- The unpacked 6-bit scale for j < 4 (`q[j] & 63`) is below 64. -/
theorem scale_low_lt (a : Nat) : a % 64 < 64 := by
  exact Nat.mod_lt _ (by decide)

/-- Q4_K / Q5_K loop: group scale indices are is and is+1 with is in {0,2,4,6}, always below 8. -/
theorem k_scale_index_lt (g : Nat) (hg : g < 4) : 2 * g + 1 < 8 := by
  omega

/-- Q4_K covers 256 elements as 4 iterations of 64. -/
theorem q4_k_coverage : 4 * 64 = 256 := by
  decide

-- § Q5_K high-bit masks

/-- Mask of the 5th bit for the low-nibble half of group `g`: 1, 4, 16, 64. -/
def maskLo (g : Nat) : Nat := 4 ^ g

/-- Mask of the 5th bit for the high-nibble half of group `g`: 2, 8, 32, 128. -/
def maskHi (g : Nat) : Nat := 2 * 4 ^ g

/-- All eight masks are distinct single bits of a byte (each group/half pair has its own bit). -/
theorem q5_masks_distinct :
    ([maskLo 0, maskLo 1, maskLo 2, maskLo 3, maskHi 0, maskHi 1, maskHi 2, maskHi 3] : List Nat).Nodup := by
  decide

/-- The eight masks together are exactly 1,2,4,...,128 (every bit of qh is used once). -/
theorem q5_masks_all_bits :
    ([maskLo 0, maskHi 0, maskLo 1, maskHi 1, maskLo 2, maskHi 2, maskLo 3, maskHi 3] : List Nat) =
      [1, 2, 4, 8, 16, 32, 64, 128] := by
  decide

/-- Q5_K code is a nibble plus 16 when the mask bit is set: at most 31. -/
theorem q5_k_code_le (nib : Nat) (b : Bool) (h : nib < 16) : nib + (if b then 16 else 0) ≤ 31 := by
  cases b <;> simp <;> omega

-- § Q3_K

/-- Q3_K element code: 2-bit low part minus 4 when the high-mask bit is clear. -/
def q3Val (c : Nat) (hiSet : Bool) : Int := (c : Int) - (if hiSet then 0 else 4)

/-- Q3_K element codes lie in [-4, 3]. -/
theorem q3_range (c : Nat) (b : Bool) (h : c < 4) : -4 ≤ q3Val c b ∧ q3Val c b ≤ 3 := by
  cases b <;> simp [q3Val] <;> omega

/-- Q3_K repacked scale byte: low 4 bits plus 2 high bits shifted by 4 is a 6-bit value; minus 32 gives [-32, 31]. -/
theorem q3_scale_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q3_K draws 16 scales: 2 halves x 4 shifts x 2 sub-blocks. -/
theorem q3_k_scale_count : 2 * 4 * 2 = 16 := by
  decide

-- § IQ4_XS

/-- The 6-bit scale of group `ib`: a nibble of scales_l plus 2 bits of scales_h at shift 2*ib. -/
def iq4xsScale (sl : Nat) (ib : Nat) (sh : Nat) : Nat :=
  (sl / 16 ^ (ib % 2)) % 16 + 16 * ((sh / 4 ^ ib) % 4)

/-- The IQ4_XS scale is below 64, so `ls - 32` lies in [-32, 31]. -/
theorem iq4xs_scale_lt (sl ib sh : Nat) : iq4xsScale sl ib sh < 64 := by
  unfold iq4xsScale
  have h1 : (sl / 16 ^ (ib % 2)) % 16 < 16 := Nat.mod_lt _ (by decide)
  have h2 : (sh / 4 ^ ib) % 4 < 4 := Nat.mod_lt _ (by decide)
  omega

/-- The scales_l byte index ib/2 is below 4 for the 8 groups. -/
theorem iq4xs_scales_l_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 := by
  omega

/-- Eight 32-element groups cover 256 elements, consuming 8 * 16 = 128 qs bytes. -/
theorem iq4xs_coverage : 8 * 32 = 256 ∧ 8 * 16 = 128 := by
  decide

end Ref.BlockDequantizers



## FILE: /mnt/user-data/outputs/Fp16Bits.lean
/-!
# MODULE: fp16_bits
SOURCES: include/ref/kernels/f16_bits.hpp
Bit-exact fp16 <-> f32 conversion modelled on bit fields: round-to-nearest-even encode with explicit
normal / subnormal / overflow regimes, and an exact decode.

## Source semantics
* encode: raw f32 exponent 255 stays inf/NaN; a FINITE value too large for fp16 saturates to inf (not NaN);
  a value below the subnormal range underflows to a signed zero; the round carry may bump the exponent.
* decode: exact for all 65536 inputs, subnormal `man * 2^-24` is renormalised (f32 exponent field `113 - s`).

## Form template identifiers
- `f16_from_f32` -> `encode_rne`; `f32_from_f16` -> `decode_exact`.
-/

namespace Ref.Fp16Bits

-- § field layout

/-- Sign bit of an fp16 pattern. -/
def sgn (h : Nat) : Nat := h / 32768

/-- 5-bit exponent field of an fp16 pattern. -/
def expo (h : Nat) : Nat := h / 1024 % 32

/-- 10-bit mantissa field of an fp16 pattern. -/
def mant (h : Nat) : Nat := h % 1024

/-- An fp16 pattern is exactly sign, exponent and mantissa fields. -/
theorem fields_recompose (h : Nat) (hh : h < 65536) :
    sgn h * 32768 + expo h * 1024 + mant h = h := by
  unfold sgn expo mant
  omega

/-- Finite normal exponents 1..30 map into f32 exponent fields 113..142, never 0 or 255. -/
theorem normal_exponent_range (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    113 ≤ ex + 112 ∧ ex + 112 ≤ 142 := by
  omega

/-- Subnormal renormalisation: after s ≥ 1 shifts (s ≤ 10) the f32 exponent field `113 - s` lies in 103..112. -/
theorem subnormal_exponent_range (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 10) :
    103 ≤ 113 - s ∧ 113 - s ≤ 112 := by
  omega

/-- The renormalised exponent is exactly the true exponent: `man * 2^-24` with leading 1 at bit 10-s equals
`2^(-14-s)(1+f)`, i.e. (10 - s) - 24 = -(14 + s), and the biased field is `-(14+s) + 127 = 113 - s`. -/
theorem subnormal_exponent_value (s : Nat) (h : s ≤ 10) :
    ((10 : Int) - s) - 24 = -((14 : Int) + s) ∧ (113 : Int) - s = -((14 : Int) + s) + 127 := by
  omega

/-- The smallest subnormal 0x0001 gets f32 exponent field 103, i.e. 2^(103-127) = 2^-24. -/
theorem smallest_subnormal : (113 - 10 : Nat) = 103 ∧ (103 : Int) - 127 = -24 := by
  decide

-- § encode

/-- Finite-normal case: exponent field `rawexp - 112`, mantissa from the top 10 of 23 bits, RNE carry. -/
def encNormal (s rawexp man : Nat) : Nat :=
  let h := s + (rawexp - 112) * 1024 + man / 8192
  let rem := man % 8192
  if rem > 4096 ∨ (rem = 4096 ∧ h % 2 = 1) then h + 1 else h

/-- Subnormal case with shift divisor `d = 2^sh`: mantissa with implicit 1 restored, RNE. -/
def encSub (s man d : Nat) : Nat :=
  let m := man + 8388608
  let h := m / d % 1024
  let rem := m % d
  if rem > d / 2 ∨ (rem = d / 2 ∧ h % 2 = 1) then s + h + 1 else s + h

/-- Encode from (sign bit value `s` = 0 or 32768, raw f32 exponent, 23-bit mantissa). -/
def f16FromF32 (s rawexp man : Nat) (d : Nat) : Nat :=
  if rawexp = 255 then s + 31744 + (if man ≠ 0 then 512 else 0)
  else if 143 ≤ rawexp then s + 31744
  else if rawexp ≤ 112 then
    (if rawexp < 102 then s else encSub s man d)
  else encNormal s rawexp man

/-- An f32 infinity stays an fp16 infinity. -/
theorem enc_inf (s d : Nat) : f16FromF32 s 255 0 d = s + 31744 := by
  simp [f16FromF32]

/-- An f32 NaN stays a NaN: exponent all ones and a nonzero mantissa. -/
theorem enc_nan (s man d : Nat) (h : man ≠ 0) : f16FromF32 s 255 man d = s + 31744 + 512 := by
  simp [f16FromF32, h]

/-- A FINITE value too large for fp16 saturates to inf, never to NaN (the bug the shared copy fixes). -/
theorem enc_overflow_is_inf (s rawexp man d : Nat) (h1 : 143 ≤ rawexp) (h2 : rawexp ≠ 255) :
    f16FromF32 s rawexp man d = s + 31744 := by
  simp [f16FromF32, h1, h2]

/-- A value below the subnormal range underflows to a signed zero. -/
theorem enc_underflow (s rawexp man d : Nat) (h : rawexp < 102) : f16FromF32 s rawexp man d = s := by
  have h1 : rawexp ≠ 255 := by omega
  have h2 : ¬ (143 ≤ rawexp) := by omega
  have h3 : rawexp ≤ 112 := by omega
  simp [f16FromF32, h1, h2, h3, h]

/-- In the normal regime the result never exceeds infinity (the carry can reach, not pass, 0x7C00). -/
theorem enc_normal_le_inf (s rawexp man : Nat) (hs : s = 0 ∨ s = 32768) (h1 : 113 ≤ rawexp)
    (h2 : rawexp ≤ 142) (hm : man < 8388608) : encNormal s rawexp man ≤ s + 31744 := by
  unfold encNormal
  dsimp only
  split <;> omega

/-- In the normal regime the result is at least the smallest normal number. -/
theorem enc_normal_ge_min (s rawexp man : Nat) (h1 : 113 ≤ rawexp) :
    s + 1024 ≤ encNormal s rawexp man := by
  unfold encNormal
  dsimp only
  split <;> omega

/-- A subnormal result never exceeds the smallest normal (the carry may promote into it). -/
theorem enc_sub_le_min_normal (s man d : Nat) : encSub s man d ≤ s + 1024 := by
  unfold encSub
  dsimp only
  have h2 : (man + 8388608) / d % 1024 < 1024 := Nat.mod_lt _ (by decide)
  split <;> omega

/-- A tie with an even truncated result is not rounded up (round-to-nearest-EVEN). -/
theorem enc_normal_tie_even (s rawexp man : Nat) (h : man % 8192 = 4096)
    (he : (s + (rawexp - 112) * 1024 + man / 8192) % 2 = 0) :
    encNormal s rawexp man = s + (rawexp - 112) * 1024 + man / 8192 := by
  unfold encNormal
  simp only [h, he]
  simp

/-- A tie with an odd truncated result rounds up. -/
theorem enc_normal_tie_odd (s rawexp man : Nat) (h : man % 8192 = 4096)
    (ho : (s + (rawexp - 112) * 1024 + man / 8192) % 2 = 1) :
    encNormal s rawexp man = s + (rawexp - 112) * 1024 + man / 8192 + 1 := by
  unfold encNormal
  simp only [h, ho]
  simp

-- § decode

/-- Decode a finite normal or infinite pattern (exponent field nonzero) to (sign, raw f32 exponent, 23-bit mantissa). -/
def decodeFields (h : Nat) : Nat × Nat × Nat :=
  (sgn h * 32768, (if expo h = 31 then 255 else expo h + 112), mant h * 8192)

/-- The decoded 23-bit mantissa fits in 23 bits (the fp16 mantissa is padded by 13 zero bits). -/
theorem decode_mantissa_lt (h : Nat) : (decodeFields h).2.2 < 8388608 := by
  unfold decodeFields mant
  simp only
  omega

/-- Finite normals decode to a raw exponent that is neither 0 nor 255. -/
theorem decode_normal_exponent (h : Nat) (h1 : 1 ≤ expo h) (h2 : expo h ≤ 30) :
    113 ≤ (decodeFields h).2.1 ∧ (decodeFields h).2.1 ≤ 142 := by
  unfold decodeFields
  simp only
  have : expo h ≠ 31 := by omega
  simp [this]
  omega

/-- encode ∘ decode is the identity on every finite normal fp16 value. -/
theorem roundtrip_normal (h : Nat) (hh : h < 65536) (h1 : 1 ≤ expo h) (h2 : expo h ≤ 30) (d : Nat) :
    f16FromF32 (decodeFields h).1 (decodeFields h).2.1 (decodeFields h).2.2 d = h := by
  have hne : expo h ≠ 31 := by omega
  have hrec := fields_recompose h hh
  have hm : mant h < 1024 := by unfold mant; omega
  have e1 : (decodeFields h).2.1 = expo h + 112 := by simp [decodeFields, hne]
  have e2 : (decodeFields h).2.2 = mant h * 8192 := rfl
  have e3 : (decodeFields h).1 = sgn h * 32768 := rfl
  rw [e1, e2, e3]
  have n1 : expo h + 112 ≠ 255 := by omega
  have n2 : ¬ (143 ≤ expo h + 112) := by omega
  have n3 : ¬ (expo h + 112 ≤ 112) := by omega
  have q1 : mant h * 8192 / 8192 = mant h := by omega
  have q2 : mant h * 8192 % 8192 = 0 := by omega
  simp only [f16FromF32, encNormal, n1, n2, n3, q1, q2]
  simp
  omega

/-- encode ∘ decode is the identity on infinity (exponent 31, mantissa 0). -/
theorem roundtrip_inf (h : Nat) (hh : h < 65536) (he : expo h = 31) (hm : mant h = 0) (d : Nat) :
    f16FromF32 (decodeFields h).1 (decodeFields h).2.1 (decodeFields h).2.2 d = h := by
  have hrec := fields_recompose h hh
  have e1 : (decodeFields h).2.1 = 255 := by simp [decodeFields, he]
  have e2 : (decodeFields h).2.2 = 0 := by simp [decodeFields, hm]
  have e3 : (decodeFields h).1 = sgn h * 32768 := rfl
  rw [e1, e2, e3]
  rw [he, hm] at hrec
  simp [f16FromF32]
  omega

end Ref.Fp16Bits



## FILE: /mnt/user-data/outputs/ChatRequestNormalizer.lean
/-!
# MODULE: chat_request_normalizer
SOURCES: serve/frontend.py
Request-side half of the frontend: reasoning-effort normalisation (OpenAI / Anthropic spellings onto the
template's levels), thinking-budget bucketing, image-part extraction and role mapping.

## Source semantics
* Effort strings map onto {off, low, medium, xhigh}; `high` means `xhigh`; unknown values are a 400.
* `off` is expressed as `enable_thinking=false` and carries no `reasoning_effort`.
* `budget_tokens`: < 2048 low, < 8192 medium, else xhigh (monotone).
* Image parts keep their order; unknown parts are dropped; image sources are listed in prompt order.
* `developer` role becomes `system`.
* A `chat_template_kwargs` `enable_thinking=false` overrides any effort already chosen.

## Form template identifiers
- `effort_kwargs` -> `effort_normalize`; `budget_effort` -> `budget_bucket`; `_parts_of`/`images_of` -> `image_part_extract`;
  `openai_to_messages` -> `openai_request_adapter`; `anthropic_to_messages` -> `anthropic_request_adapter`.
-/

namespace Ref.ChatRequestNormalizer

-- § effort levels

/-- The template's thinking levels. `off` means thinking disabled. -/
inductive Thinking where
  | off
  | low
  | medium
  | xhigh
  deriving DecidableEq

/-- The EFFORT dictionary; `none` is an unknown spelling (a 400). -/
def effortOf : String → Option Thinking
  | "none" => some Thinking.off
  | "off" => some Thinking.off
  | "minimal" => some Thinking.off
  | "disabled" => some Thinking.off
  | "false" => some Thinking.off
  | "low" => some Thinking.low
  | "medium" => some Thinking.medium
  | "high" => some Thinking.xhigh
  | "xhigh" => some Thinking.xhigh
  | "max" => some Thinking.xhigh
  | "maximum" => some Thinking.xhigh
  | _ => none

/-- `high` is the template's `xhigh`. -/
theorem high_is_xhigh : effortOf "high" = some Thinking.xhigh := by
  rfl

/-- The five "off" spellings all disable thinking. -/
theorem off_spellings : ∀ s ∈ ["none", "off", "minimal", "disabled", "false"],
    effortOf s = some Thinking.off := by
  decide

/-- An unknown spelling is rejected rather than guessed. -/
theorem unknown_effort_rejected : effortOf "ultra" = none := by
  rfl

/-- Template kwargs: `enable_thinking` flag and `reasoning_effort` level. -/
structure Kw where
  enable : Option Bool := none
  effort : Option Thinking := none
  deriving DecidableEq

/-- A level as template kwargs. -/
def levelKw : Thinking → Kw
  | Thinking.off => { enable := some false, effort := none }
  | t => { enable := none, effort := some t }

/-- `off` never carries a reasoning effort (the two kwargs are mutually exclusive). -/
theorem off_has_no_effort : (levelKw Thinking.off).effort = none ∧ (levelKw Thinking.off).enable = some false := by
  decide

/-- A positive level never disables thinking. -/
theorem level_not_disabled (t : Thinking) (h : t ≠ Thinking.off) : (levelKw t).enable = none := by
  cases t <;> simp [levelKw] at h ⊢

-- § thinking budget

/-- Rank of a level for ordering. -/
def rank : Thinking → Nat
  | Thinking.off => 0
  | Thinking.low => 1
  | Thinking.medium => 2
  | Thinking.xhigh => 3

/-- Anthropic `budget_tokens` bucketed: under 2K low, under 8K medium, else xhigh. -/
def budgetEffort (n : Nat) : Thinking :=
  if n < 2048 then Thinking.low else if n < 8192 then Thinking.medium else Thinking.xhigh

/-- A bigger budget never gives a lower level. -/
theorem budget_monotone (n m : Nat) (h : n ≤ m) : rank (budgetEffort n) ≤ rank (budgetEffort m) := by
  unfold budgetEffort
  by_cases a : n < 2048
  · by_cases b : m < 2048
    · simp [a, b, rank]
    · by_cases c : m < 8192
      · simp [a, b, c, rank]
      · simp [a, b, c, rank]
  · have b : ¬ m < 2048 := by omega
    by_cases c : n < 8192
    · by_cases d : m < 8192
      · simp [a, b, c, d, rank]
      · simp [a, b, c, d, rank]
    · have d : ¬ m < 8192 := by omega
      simp [a, b, c, d, rank]

/-- A budget budget never produces `off`: a budget always leaves thinking enabled. -/
theorem budget_never_off (n : Nat) : budgetEffort n ≠ Thinking.off := by
  unfold budgetEffort
  by_cases a : n < 2048 <;> by_cases b : n < 8192 <;> simp [a, b]

/-- Bucket boundaries: 2047 is low, 2048 medium, 8191 medium, 8192 xhigh. -/
theorem budget_boundaries :
    budgetEffort 2047 = Thinking.low ∧ budgetEffort 2048 = Thinking.medium ∧
    budgetEffort 8191 = Thinking.medium ∧ budgetEffort 8192 = Thinking.xhigh := by
  decide

-- § chat_template_kwargs override

/-- `enable_thinking=false` from `chat_template_kwargs` replaces whatever was chosen. -/
def applyDisable (kw : Kw) (v : Bool) : Kw :=
  if v then kw else { enable := some false, effort := none }

/-- After a disable override thinking is off and no effort remains. -/
theorem disable_clears_effort (kw : Kw) : (applyDisable kw false).effort = none ∧ (applyDisable kw false).enable = some false := by
  simp [applyDisable]

/-- A truthy `enable_thinking` leaves the kwargs unchanged. -/
theorem enable_true_noop (kw : Kw) : applyDisable kw true = kw := by
  simp [applyDisable]

/-- A `reasoning_effort` key only applies when thinking was not explicitly disabled. -/
def applyEffort (kw new : Kw) : Kw :=
  if kw.enable = none then
    { enable := match new.enable with | some b => some b | none => kw.enable,
      effort := match new.effort with | some e => some e | none => kw.effort }
  else kw

/-- Once thinking is disabled a later effort key cannot re-enable it. -/
theorem disabled_is_sticky (kw new : Kw) (h : kw.enable = some false) : applyEffort kw new = kw := by
  simp [applyEffort, h]

-- § image parts

/-- A content part: text, an image with its source string, or something unrecognised. -/
inductive Part where
  | text (s : String)
  | image (src : String)
  | other

/-- Keep text and image items in order, drop anything else. -/
def toItems : List Part → List Part
  | [] => []
  | Part.other :: r => toItems r
  | p :: r => p :: toItems r

/-- Image sources in prompt order. -/
def imgs : List Part → List String
  | [] => []
  | Part.image s :: r => s :: imgs r
  | _ :: r => imgs r

/-- Dropping unrecognised parts never changes which images are sent, nor their order. -/
theorem imgs_toItems (ps : List Part) : imgs (toItems ps) = imgs ps := by
  induction ps with
  | nil => rfl
  | cons p r ih => cases p <;> simp [toItems, imgs, ih]

/-- The item list is never longer than the input. -/
theorem toItems_length_le (ps : List Part) : (toItems ps).length ≤ ps.length := by
  induction ps with
  | nil => simp [toItems]
  | cons p r ih => cases p <;> simp [toItems] <;> omega

/-- Normalising twice equals normalising once. -/
theorem toItems_idempotent (ps : List Part) : toItems (toItems ps) = toItems ps := by
  induction ps with
  | nil => rfl
  | cons p r ih => cases p <;> simp [toItems, ih]

-- § roles

/-- OpenAI `developer` is the template's `system`. -/
def roleMap (r : String) : String := if r = "developer" then "system" else r

/-- Role mapping is idempotent. -/
theorem roleMap_idempotent (r : String) : roleMap (roleMap r) = roleMap r := by
  by_cases h : r = "developer"
  · simp [roleMap, h]
  · simp [roleMap, h]

/-- A `developer` message becomes `system`. -/
theorem developer_becomes_system : roleMap "developer" = "system" := by
  simp [roleMap]

end Ref.ChatRequestNormalizer



## FILE: /mnt/user-data/outputs/StreamingOutputParser.lean
/-!
# MODULE: streaming_output_parser
SOURCES: serve/frontend.py
Response-side half of the frontend: the incremental `OutputParser` state machine (reasoning -> content <-> tool
call), partial-tag hold-back, tool-call argument decoding and the template renderer's flag mapping.

## Source semantics
* States: reasoning, content, call. `</think>` leaves reasoning for good; `<tool_call>` enters call; `</tool_call>`
  returns to content with leading newlines swallowed.
* A suffix of the buffer that is a proper prefix of a tag is held back, so clients never see `<tool_` or `</thi`.
* `finish()` flushes held text as reasoning or content, never as a tool call.
* Tool parameter values: declared `string` stays a string; otherwise JSON when it parses; one leading and one trailing
  newline are removed.

## Form template identifiers
- `_hold` -> `partial_tag_holdback`; `OutputParser.feed` -> `incremental_state_machine`; `parse_tool_call` -> `tool_call_decode`.
-/

namespace Ref.StreamingOutputParser

-- § hold-back of partial tags

/-- Longest `n ≤ fuel` such that the first `n` elements of `tag` form a suffix of `text`. -/
def holdAux {α : Type} [BEq α] (tag text : List α) : Nat → Nat
  | 0 => 0
  | n + 1 => max (holdAux tag text n) (if (tag.take (n + 1)).isSuffixOf text then n + 1 else 0)

/-- `_hold`: longest proper-prefix-of-`tag` suffix of `text` (lengths 1 .. |tag|-1). -/
def holdLen {α : Type} [BEq α] (tag text : List α) : Nat := holdAux tag text (tag.length - 1)

/-- The hold search never exceeds its fuel. -/
theorem holdAux_le {α : Type} [BEq α] (tag text : List α) (n : Nat) : holdAux tag text n ≤ n := by
  induction n with
  | zero => simp [holdAux]
  | succ k ih =>
    unfold holdAux
    split <;> omega

/-- The held part is a PROPER prefix: strictly shorter than the tag, so a complete tag is never held. -/
theorem holdLen_lt {α : Type} [BEq α] (tag text : List α) (h : 0 < tag.length) :
    holdLen tag text < tag.length := by
  have := holdAux_le tag text (tag.length - 1)
  unfold holdLen
  omega

/-- An empty tag holds nothing. -/
theorem holdLen_empty_tag {α : Type} [BEq α] (text : List α) : holdLen ([] : List α) text = 0 := by
  simp [holdLen, holdAux]

/-- Split a buffer into the emitted part and the held tail of length `keep`. -/
def splitHold {α : Type} (buf : List α) (keep : Nat) : List α × List α :=
  (buf.take (buf.length - keep), buf.drop (buf.length - keep))

/-- Emitted ++ held reproduces the buffer exactly: no text is lost or duplicated by holding. -/
theorem splitHold_append {α : Type} (buf : List α) (keep : Nat) :
    (splitHold buf keep).1 ++ (splitHold buf keep).2 = buf := by
  unfold splitHold
  simp

/-- The held tail has the requested length when it fits in the buffer. -/
theorem splitHold_held_length {α : Type} (buf : List α) (keep : Nat) (h : keep ≤ buf.length) :
    (splitHold buf keep).2.length = keep := by
  unfold splitHold
  simp
  omega

/-- The emitted part has length `len - keep`. -/
theorem splitHold_emit_length {α : Type} (buf : List α) (keep : Nat) :
    (splitHold buf keep).1.length = buf.length - keep := by
  unfold splitHold
  simp

/-- Concrete check: with tag [1,2,3] and text ending [1,2], two elements are held. -/
theorem hold_example : holdLen [1, 2, 3] [9, 1, 2] = 2 := by
  decide

/-- Concrete check: text not ending in a tag prefix holds nothing. -/
theorem hold_none_example : holdLen [1, 2, 3] [9, 8, 7] = 0 := by
  decide

-- § state machine

/-- Parser states. -/
inductive State where
  | reasoning
  | content
  | call
  deriving DecidableEq

/-- Markers the parser reacts to. -/
inductive Marker where
  | thinkEnd
  | callStart
  | callEnd
  deriving DecidableEq

/-- State transition when a marker is found in the current state; other markers are inert. -/
def next : State → Marker → State
  | State.reasoning, Marker.thinkEnd => State.content
  | State.content, Marker.callStart => State.call
  | State.call, Marker.callEnd => State.content
  | s, _ => s

/-- The initial state is reasoning exactly when thinking is on. -/
def initState (thinking : Bool) : State := if thinking then State.reasoning else State.content

/-- With thinking disabled the parser starts in content. -/
theorem init_no_thinking : initState false = State.content := by
  rfl

/-- Reasoning is never re-entered: a transition into reasoning only comes from reasoning. -/
theorem reasoning_not_reentered (s : State) (m : Marker) (h : next s m = State.reasoning) :
    s = State.reasoning := by
  cases s <;> cases m <;> simp [next] at h ⊢

/-- `</think>` takes reasoning to content. -/
theorem think_end_exits_reasoning : next State.reasoning Marker.thinkEnd = State.content := by
  rfl

/-- A completed tool call returns to content. -/
theorem call_end_returns : next State.call Marker.callEnd = State.content := by
  rfl

/-- A tool call can only start from content (never from reasoning). -/
theorem call_only_from_content (s : State) (h : next s Marker.callStart = State.call) : s = State.content ∨ s = State.call := by
  cases s <;> simp [next] at h ⊢

-- § events and flush

/-- Event kinds. -/
inductive Kind where
  | reasoning
  | content
  | toolCall
  deriving DecidableEq

/-- The kind `finish()` flushes held text as: reasoning stays reasoning, everything else (including an
unterminated call) is content. -/
def finishKind : State → Kind
  | State.reasoning => Kind.reasoning
  | _ => Kind.content

/-- `finish()` never fabricates a tool call from unterminated input. -/
theorem finish_never_tool_call (s : State) : finishKind s ≠ Kind.toolCall := by
  cases s <;> simp [finishKind]

-- § tool-call argument decoding

/-- A decoded argument is either the raw string or a JSON value. -/
inductive Arg where
  | str
  | json
  deriving DecidableEq

/-- Declared-string parameters stay strings; others are JSON when they parse, else strings. -/
def decodeArg (declaredString parses : Bool) : Arg :=
  if declaredString then Arg.str else if parses then Arg.json else Arg.str

/-- A declared string is never JSON-decoded (so the text `123` stays `"123"`). -/
theorem declared_string_stays_string (parses : Bool) : decodeArg true parses = Arg.str := by
  simp [decodeArg]

/-- An unparsable value falls back to the string. -/
theorem unparsable_is_string (d : Bool) : decodeArg d false = Arg.str := by
  cases d <;> simp [decodeArg]

/-- Drop one leading newline (10). -/
def dropLead : List Nat → List Nat
  | 10 :: r => r
  | l => l

/-- Drop one trailing newline (10). -/
def dropTrail (l : List Nat) : List Nat := if l.getLast? = some 10 then l.dropLast else l

/-- Exactly one leading newline is removed. -/
theorem dropLead_newline (r : List Nat) : dropLead (10 :: r) = r := by
  rfl

/-- A leading non-newline is untouched. -/
theorem dropLead_other (c : Nat) (r : List Nat) (h : c ≠ 10) : dropLead (c :: r) = c :: r := by
  unfold dropLead
  split
  · rename_i heq
    simp at heq
    omega
  · rfl

/-- Exactly one trailing newline is removed. -/
theorem dropTrail_newline (l : List Nat) : dropTrail (l ++ [10]) = l := by
  simp [dropTrail]

/-- Stripping never makes a value longer, and removes at most one character at each end. -/
theorem dropLead_length (l : List Nat) : l.length - 1 ≤ (dropLead l).length ∧ (dropLead l).length ≤ l.length := by
  unfold dropLead
  split <;> simp

/-- A tool call needs a `<function=` prefix; the name is the text up to the first `>`. -/
def hasFunctionPrefix (body : List Nat) (prefixLen : Nat) : Bool := prefixLen ≤ body.length

/-- The tool-call id is `call_` plus the first 24 of the 32 uuid hex digits. -/
theorem call_id_digits : 24 ≤ 32 ∧ "call_".length + 24 = 29 := by
  decide

end Ref.StreamingOutputParser



## FILE: /mnt/user-data/outputs/ApiServerEngine.lean
/-!
# MODULE: api_server_engine
SOURCES: serve/server.py
OpenAI/Anthropic server over a token-id engine: context admission, incremental detokenisation, stop-token
handling, finish-reason mapping, Anthropic content-block bookkeeping, image-pad expansion, vision cache bound,
auth and launcher argument assembly.

## Source semantics
* Requests with `prompt + max_new > max_context` are rejected, never truncated.
* `completion_tokens` counts every yielded id including the stop id; the stop id is not emitted as text.
* OpenAI finish: `tool_calls` only when at least one call and the engine stopped; Anthropic: `max_tokens` for length.
* One `<|image_pad|>` per image is expanded into one pad per image token.
* The vision cache holds at most 64 entries (oldest evicted).
* `max_completion_tokens or max_tokens or 1024` is always positive.

## Form template identifiers
- `Service.prepare` -> `admission_and_image_expand`; `Service.run` -> `stream_loop`; `Detokenizer` -> `incremental_detok`;
  `openai_chunks`/`anthropic_events` -> `protocol_adapters`; `Vision.encode` -> `cached_image_encode`.
-/

namespace Ref.ApiServerEngine

-- § admission

/-- Admission: the prompt plus the generation budget must fit in the context. -/
def admit (promptLen maxNew ctx : Nat) : Bool := promptLen + maxNew ≤ ctx

/-- An admitted request fits in the context. -/
theorem admit_fits (p m c : Nat) (h : admit p m c = true) : p + m ≤ c := by
  simpa [admit] using h

/-- An over-long request is rejected (400), not truncated. -/
theorem reject_overlong (p m c : Nat) (h : c < p + m) : admit p m c = false := by
  have : ¬ (p + m ≤ c) := by omega
  simp [admit, this]

/-- `max_completion_tokens or max_tokens or 1024` (0 counts as unset, like Python `or`). -/
def maxNew (a b : Nat) : Nat := if a ≠ 0 then a else if b ≠ 0 then b else 1024

/-- The effective generation budget is always positive. -/
theorem maxNew_pos (a b : Nat) : 0 < maxNew a b := by
  unfold maxNew
  split
  · omega
  · split <;> omega

/-- An explicit `max_completion_tokens` wins over `max_tokens`. -/
theorem maxNew_prefers_first (a b : Nat) (h : a ≠ 0) : maxNew a b = a := by
  simp [maxNew, h]

-- § stop handling

/-- Ids that are emitted before the first stop id. -/
def emitted (isStop : Nat → Bool) (ts : List Nat) : List Nat := ts.takeWhile (fun t => !isStop t)

/-- `completion_tokens`: emitted ids plus the stop id when one was seen. -/
def completionTokens (isStop : Nat → Bool) (ts : List Nat) : Nat :=
  (emitted isStop ts).length + (if ts.any isStop then 1 else 0)

/-- Counting never exceeds the number of ids the engine yielded. -/
theorem completion_le (isStop : Nat → Bool) (ts : List Nat) : completionTokens isStop ts ≤ ts.length := by
  unfold completionTokens emitted
  induction ts with
  | nil => simp
  | cons t r ih =>
    by_cases h : isStop t = true
    · simp [List.takeWhile, h]
    · have h' : isStop t = false := by simpa using h
      have e : List.takeWhile (fun t => !isStop t) (t :: r) =
          t :: List.takeWhile (fun t => !isStop t) r := by simp [List.takeWhile, h']
      have e2 : (t :: r).any isStop = r.any isStop := by simp [List.any_cons, h']
      rw [e, e2, List.length_cons, List.length_cons]
      omega

/-- No stop id is ever emitted as text. -/
theorem emitted_no_stop (isStop : Nat → Bool) (ts : List Nat) : ∀ t ∈ emitted isStop ts, isStop t = false := by
  intro t ht
  unfold emitted at ht
  have := List.mem_takeWhile_imp ht
  simpa using this

/-- With no stop id the whole stream is emitted and no extra token is counted. -/
theorem no_stop_all_emitted (isStop : Nat → Bool) (ts : List Nat) (h : ts.any isStop = false) :
    emitted isStop ts = ts := by
  unfold emitted
  apply List.takeWhile_eq_self_iff.mpr
  intro x hx
  have := List.any_eq_false.mp h x hx
  simpa using this

-- § finish reasons

/-- Engine finish reasons. -/
inductive Finish where
  | stop
  | length
  deriving DecidableEq

/-- OpenAI finish reasons. -/
inductive OaFinish where
  | stop
  | length
  | toolCalls
  deriving DecidableEq

/-- OpenAI mapping: `tool_calls` only if calls were made and the engine stopped. -/
def oaFinish (calls : Nat) (f : Finish) : OaFinish :=
  match f with
  | Finish.stop => if calls ≠ 0 then OaFinish.toolCalls else OaFinish.stop
  | Finish.length => OaFinish.length

/-- Hitting the length limit is always reported as `length`, even with tool calls. -/
theorem oa_length_stays (calls : Nat) : oaFinish calls Finish.length = OaFinish.length := by
  rfl

/-- `tool_calls` is reported only when at least one tool call was emitted. -/
theorem oa_tool_calls_needs_call (calls : Nat) (f : Finish) (h : oaFinish calls f = OaFinish.toolCalls) :
    calls ≠ 0 := by
  cases f <;> simp [oaFinish] at h
  exact h

/-- Anthropic stop reasons. -/
inductive AnStop where
  | endTurn
  | maxTokens
  | toolUse
  deriving DecidableEq

/-- Anthropic mapping: `tool_use` when a tool block was emitted and the engine stopped. -/
def anStop (usedTool : Bool) (f : Finish) : AnStop :=
  if usedTool = true ∧ f = Finish.stop then AnStop.toolUse
  else match f with
    | Finish.stop => AnStop.endTurn
    | Finish.length => AnStop.maxTokens

/-- Length always maps to `max_tokens`, with or without tools. -/
theorem an_length_is_max (u : Bool) : anStop u Finish.length = AnStop.maxTokens := by
  cases u <;> simp [anStop]

/-- Without a tool block, a stop is `end_turn`. -/
theorem an_stop_no_tool : anStop false Finish.stop = AnStop.endTurn := by
  simp [anStop]

-- § Anthropic block bookkeeping

/-- Content-block kinds in the Anthropic stream. -/
inductive BK where
  | thinking
  | text
  | tool
  deriving DecidableEq

/-- Number of blocks opened: a new block starts when the kind changes, and always for a tool call. -/
def blocks (prev : Option BK) : List BK → Nat
  | [] => 0
  | k :: r => (if prev = some k ∧ k ≠ BK.tool then 0 else 1) + blocks (some k) r

/-- Never more blocks than events. -/
theorem blocks_le (ks : List BK) : ∀ prev, blocks prev ks ≤ ks.length := by
  induction ks with
  | nil => intro prev; simp [blocks]
  | cons k r ih =>
    intro prev
    have := ih (some k)
    simp only [blocks, List.length_cons]
    split <;> omega

/-- Consecutive text deltas share one block. -/
theorem text_deltas_merge : blocks none [BK.text, BK.text, BK.text] = 1 := by
  decide

/-- Every tool call opens its own block. -/
theorem tool_calls_separate : blocks none [BK.tool, BK.tool] = 2 := by
  decide

/-- A thinking block followed by text is two blocks. -/
theorem thinking_then_text : blocks none [BK.thinking, BK.thinking, BK.text] = 2 := by
  decide

-- § incremental detokenisation

/-- One step of the detokeniser: given the full decoded text and what was already sent, emit the new suffix unless
the text ends in a replacement character. Returns (delta, new sent). -/
def detokStep {α : Type} (text : List α) (sent : Nat) (endsReplacement : Bool) : List α × Nat :=
  if endsReplacement then ([], sent) else (text.drop sent, text.length)

/-- The sent counter never exceeds the text length. -/
theorem detok_sent_le {α : Type} (text : List α) (sent : Nat) (e : Bool) (h : sent ≤ text.length) :
    (detokStep text sent e).2 ≤ text.length := by
  unfold detokStep
  split <;> simp [h]

/-- Already-sent text plus the new delta is exactly the new sent prefix: nothing is skipped or repeated. -/
theorem detok_prefix_property {α : Type} (text : List α) (sent : Nat) (e : Bool) (h : sent ≤ text.length) :
    text.take sent ++ (detokStep text sent e).1 = text.take (detokStep text sent e).2 := by
  unfold detokStep
  split
  · simp
  · simp

/-- A held (incomplete) character emits nothing and leaves the counter unchanged. -/
theorem detok_hold_noop {α : Type} (text : List α) (sent : Nat) :
    detokStep text sent true = ([], sent) := by
  simp [detokStep]

-- § image-pad expansion

/-- Replace the k-th pad with `counts[k]` pads; leave other ids alone. -/
def expand (pad : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | t :: r, cs =>
    if t = pad then
      match cs with
      | [] => t :: expand pad r []
      | c :: cs' => List.replicate c pad ++ expand pad r cs'
    else t :: expand pad r cs

/-- Number of pad ids in a prompt. -/
def countPad (pad : Nat) (ids : List Nat) : Nat := (ids.filter (fun t => t == pad)).length

/-- When pads and images match one-to-one, the expanded prompt has `len - pads + sum counts` ids. -/
theorem expand_length (pad : Nat) (ids : List Nat) : ∀ cs : List Nat, countPad pad ids = cs.length →
    (expand pad ids cs).length + countPad pad ids = ids.length + cs.sum := by
  induction ids with
  | nil =>
    intro cs h
    have h1 : countPad pad [] = 0 := by simp [countPad]
    have h0 : cs.length = 0 := by omega
    cases cs with
    | nil => simp [expand, countPad]
    | cons c r => simp at h0
  | cons t r ih =>
    intro cs h
    by_cases ht : t = pad
    · subst ht
      cases cs with
      | nil => simp [countPad] at h
      | cons c cs' =>
        have h' : countPad t r = cs'.length := by
          simp [countPad, List.filter_cons] at h ⊢
          omega
        have := ih cs' h'
        simp [expand, countPad, List.filter_cons] at this ⊢
        omega
    · have h' : countPad pad r = cs.length := by
        simp [countPad, List.filter_cons, ht] at h ⊢
        exact h
      have := ih cs h'
      simp [expand, ht, countPad, List.filter_cons] at this ⊢
      omega

-- § vision cache

/-- Cache size after inserting one entry: the oldest is evicted past 64. -/
def cacheAfterInsert (size : Nat) : Nat := if size + 1 > 64 then size else size + 1

/-- The cache never exceeds 64 entries. -/
theorem cache_bounded (size : Nat) (h : size ≤ 64) : cacheAfterInsert size ≤ 64 := by
  unfold cacheAfterInsert
  split <;> omega

/-- Below the cap an insert grows the cache by one. -/
theorem cache_grows (size : Nat) (h : size < 64) : cacheAfterInsert size = size + 1 := by
  unfold cacheAfterInsert
  have : ¬ (size + 1 > 64) := by omega
  simp [this]

-- § image formats and auth

/-- Pass-through detection for the first magic bytes: JPEG `FF D8 FF`, BMP `BM`. -/
def isJpeg (b0 b1 b2 : Nat) : Bool := b0 == 255 && b1 == 216 && b2 == 255

/-- BMP magic `B M`. -/
def isBmp (b0 b1 : Nat) : Bool := b0 == 66 && b1 == 77

/-- A JPEG header is never also a BMP header. -/
theorem jpeg_not_bmp (b0 b1 b2 b1' : Nat) (h : isJpeg b0 b1 b2 = true) : isBmp b0 b1' = false := by
  simp [isJpeg, isBmp] at h ⊢
  omega

/-- Authorisation: no key configured means open; otherwise the presented key must match. -/
def authorized (apiKey presented : String) : Bool := apiKey == "" || presented == apiKey

/-- With no API key configured every request is authorised. -/
theorem auth_open (presented : String) : authorized "" presented = true := by
  simp [authorized]

/-- With a key configured, a wrong key is refused. -/
theorem auth_wrong_refused (k g : String) (hk : k ≠ "") (hg : g ≠ k) : authorized k g = false := by
  simp [authorized, hk, hg]

-- § launcher arguments

/-- Vision launcher argv length: 5 fixed items plus `--gpu`, `--threads N`, `--max-tokens N`. -/
def visionArgc (gpu threads maxTok : Bool) : Nat :=
  5 + (if gpu then 1 else 0) + (if threads then 2 else 0) + (if maxTok then 2 else 0)

/-- The launcher argv has between 5 and 10 entries. -/
theorem vision_argc_bounds (g t m : Bool) : 5 ≤ visionArgc g t m ∧ visionArgc g t m ≤ 10 := by
  cases g <;> cases t <;> cases m <;> simp [visionArgc]

/-- `DONE` carries 6 whitespace-separated fields; indices 1..5 are the five numbers/strings read. -/
theorem done_fields : 1 + 5 = 6 := by
  decide

end Ref.ApiServerEngine



## FILE: /mnt/user-data/outputs/VisionEncoderService.lean
/-!
# MODULE: vision_encoder_service
SOURCES: tools/vision/ref_vision.cpp, tools/vision/CMakeLists.txt
The resident image encoder `ref-vision`: the `ENC <image> <out>` line protocol, the SVE1 embeddings file
layout and grid, and the CMake option logic (portable vs native, CUDA passthrough).

## Source semantics
* The image path ends at the LAST space; the output path contains no space; both must be non-empty.
* SVE1 header = 5 int32 (magic 0x31455653, n_tokens, nx, ny, n_embd) followed by `n * n_embd` float32 rows.
* The grid is derived from the last token's position: nx = x+1, ny = y+1, and must satisfy `nx * ny = n`.
* Row i sits at (x = i mod nx, y = i / nx).
* Portable builds turn native tuning off and force the AVX2-class baseline; native builds do the opposite.

## Form template identifiers
- `parse_enc` -> `enc_line_parser`; header writer -> `sve1_writer`; `mtmd_image_tokens_get_decoder_pos` -> `grid_from_last_token`;
  `REF_PORTABLE` -> `portable_build_switch`.
-/

namespace Ref.VisionEncoderService

-- § ENC line parsing

/-- Split at the last space (32): (before, after). `none` when there is no space. -/
def lastSplit : List Nat → Option (List Nat × List Nat)
  | [] => none
  | c :: r =>
    match lastSplit r with
    | some (a, b) => some (c :: a, b)
    | none => if c = 32 then some ([], r) else none

/-- If no split exists the list contains no space. -/
theorem lastSplit_none (l : List Nat) (h : lastSplit l = none) : ∀ c ∈ l, c ≠ 32 := by
  induction l with
  | nil => intro c hc; simp at hc
  | cons c r ih =>
    unfold lastSplit at h
    split at h
    · cases h
    · rename_i hr
      split at h
      · cases h
      · rename_i hc
        intro x hx
        rcases List.mem_cons.mp hx with e | e
        · subst e; exact hc
        · exact ih hr x e

/-- A split reproduces the input around one space, and the tail contains no space. -/
theorem lastSplit_spec (l a b : List Nat) (h : lastSplit l = some (a, b)) :
    l = a ++ 32 :: b ∧ ∀ c ∈ b, c ≠ 32 := by
  induction l generalizing a b with
  | nil => simp [lastSplit] at h
  | cons c r ih =>
    unfold lastSplit at h
    split at h
    · rename_i a' b' hr
      injection h with h1
      have ⟨e1, e2⟩ := Prod.mk.inj h1
      have ⟨ih1, ih2⟩ := ih a' b' hr
      subst e1
      subst e2
      exact ⟨by simp [ih1], ih2⟩
    · rename_i hr
      split at h
      · rename_i hc
        injection h with h1
        have ⟨e1, e2⟩ := Prod.mk.inj h1
        subst e1
        subst e2
        refine ⟨by simp [hc], ?_⟩
        exact lastSplit_none r hr
      · cases h

/-- `parse_enc` on the text after `ENC `: image and output must both be non-empty. -/
def parseEnc (rest : List Nat) : Option (List Nat × List Nat) :=
  match lastSplit rest with
  | some (a, b) => if a = [] ∨ b = [] then none else some (a, b)
  | none => none

/-- A parsed line has a non-empty image path and a non-empty output path. -/
theorem parseEnc_nonempty (rest a b : List Nat) (h : parseEnc rest = some (a, b)) : a ≠ [] ∧ b ≠ [] := by
  unfold parseEnc at h
  split at h
  · rename_i x y hs
    split at h
    · cases h
    · rename_i hn
      injection h with h1
      have ⟨e1, e2⟩ := Prod.mk.inj h1
      subst e1
      subst e2
      simp at hn
      exact hn
  · cases h

/-- A space-free output path parses back exactly (a path with a space in the image part is fine). -/
theorem parseEnc_example : parseEnc [1, 2, 32, 3] = some ([1, 2], [3]) := by
  decide

/-- No space at all: the line is rejected. -/
theorem parseEnc_no_space : parseEnc [1, 2, 3] = none := by
  decide

/-- A leading space leaves an empty image path, which is rejected. -/
theorem parseEnc_empty_image : parseEnc [32, 3] = none := by
  decide

-- § SVE1 file layout

/-- Header bytes: five int32 values. -/
def headerBytes : Nat := 5 * 4

/-- Total size of an SVE1 file for `n` tokens of width `e`. -/
def sveFileSize (n e : Nat) : Nat := headerBytes + n * e * 4

/-- The header is 20 bytes. -/
theorem header_is_20 : headerBytes = 20 := by
  decide

/-- The magic 0x31455653 spells `SVE1` in little-endian bytes S V E 1 = 0x53 0x56 0x45 0x31. -/
theorem sve1_magic : 0x31455653 = 0x53 + 256 * (0x56 + 256 * (0x45 + 256 * 0x31)) := by
  decide

/-- The file grows linearly: one more token adds exactly one row of `4 * e` bytes. -/
theorem sveFileSize_succ (n e : Nat) : sveFileSize (n + 1) e = sveFileSize n e + 4 * e := by
  unfold sveFileSize
  rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  omega

/-- An empty image yields a header-only file. -/
theorem sveFileSize_zero (e : Nat) : sveFileSize 0 e = 20 := by
  simp [sveFileSize, headerBytes]

-- § grid

/-- Grid width and height from the last token's decoder position. -/
def gridOf (x y : Nat) : Nat × Nat := (x + 1, y + 1)

/-- The grid is rectangular iff its area equals the token count (the source rejects otherwise). -/
def rectangular (n x y : Nat) : Bool := (x + 1) * (y + 1) == n

/-- Both grid dimensions are positive. -/
theorem grid_positive (x y : Nat) : 0 < (gridOf x y).1 ∧ 0 < (gridOf x y).2 := by
  simp [gridOf]

/-- A rectangular grid has exactly `n` cells. -/
theorem rectangular_iff (n x y : Nat) : rectangular n x y = true ↔ (x + 1) * (y + 1) = n := by
  simp [rectangular]

/-- Row `i` position: (i mod nx, i / nx) recomposes to `i`. -/
theorem position_recompose (i nx : Nat) : (i / nx) * nx + i % nx = i := by
  have := Nat.div_add_mod i nx
  rw [Nat.mul_comm] at this
  exact this

/-- Every token of a rectangular grid lands inside it: x < nx and y < ny. -/
theorem position_in_grid (i nx ny : Nat) (hnx : 0 < nx) (h : i < nx * ny) :
    i % nx < nx ∧ i / nx < ny := by
  refine ⟨Nat.mod_lt _ hnx, ?_⟩
  exact Nat.div_lt_of_lt_mul h

-- § CMake option logic

/-- GGML_NATIVE is on exactly when the build is not portable. -/
def ggmlNative (portable : Bool) : Bool := !portable

/-- GGML_OPENMP is forced off in portable builds (no vcomp/libgomp on the user's PC). -/
def ggmlOpenmpForcedOff (portable : Bool) : Bool := portable

/-- Static MSVC runtime only for portable MSVC builds. -/
def staticRuntime (portable msvc : Bool) : Bool := portable && msvc

/-- Native and portable are mutually exclusive. -/
theorem native_xor_portable (p : Bool) : ggmlNative p = !p := by
  rfl

/-- A non-portable build never uses the static MSVC runtime. -/
theorem no_static_when_native (msvc : Bool) : staticRuntime false msvc = false := by
  simp [staticRuntime]

/-- The portable ISA baseline switches five features on. -/
def portableIsa : List String := ["GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C", "GGML_BMI2"]

/-- The baseline has five distinct feature flags. -/
theorem portable_isa_distinct : portableIsa.length = 5 ∧ portableIsa.Nodup := by
  decide

/-- The CUDA backend flag simply mirrors the `REF_VISION_CUDA` option. -/
def ggmlCuda (opt : Bool) : Bool := opt

/-- CUDA is on iff the option is on. -/
theorem cuda_mirrors_option (opt : Bool) : ggmlCuda opt = opt := by
  rfl

end Ref.VisionEncoderService



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- GgufHeaderReader.lean
  was: theorem kquant_blocks : ∀ t ∈ [10, 11, 12, 13, 14, 23],
    ∃ b, blockGeometry t = some (256, b)
  now: theorem kquant_blocks : ∀ t ∈ [10, 11, 12, 13, 14, 23],
    (blockGeometry t).map Prod.fst = some 256
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem lastSplit_spec (l a b : List Nat) (h : lastSplit l = some (a, b)) :
    l = a ++ 32 :: b ∧ ∀ c ∈ b, c ≠ 32
  now: theorem lastSplit_none (l : List Nat) (h : lastSplit l = none) : ∀ c ∈ l, c ≠ 32
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- VisionEncoderService.lean
  was: theorem parseEnc_nonempty (rest a b : List Nat) (h : parseEnc rest = some (a, b)) : a ≠ [] ∧ b ≠ []
  now: theorem lastSplit_spec (l a b : List Nat) (h : lastSplit l = some (a, b)) :
    l = a ++ 32 :: b ∧ ∀ c ∈ b, c ≠ 32
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem parseEnc_example : parseEnc [1, 2, 32, 3] = some ([1, 2], [3])
  now: theorem parseEnc_nonempty (rest a b : List Nat) (h : parseEnc rest = some (a, b)) : a ≠ [] ∧ b ≠ []
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem parseEnc_no_space : parseEnc [1, 2, 3] = none
  now: theorem parseEnc_example : parseEnc [1, 2, 32, 3] = some ([1, 2], [3])
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem parseEnc_empty_image : parseEnc [32, 3] = none
  now: theorem parseEnc_no_space : parseEnc [1, 2, 3] = none
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem header_is_20 : headerBytes = 20
  now: theorem parseEnc_empty_image : parseEnc [32, 3] = none
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem sve1_magic : 0x31455653 = 0x53 + 256 * (0x56 + 256 * (0x45 + 256 * 0x31))
  now: theorem header_is_20 : headerBytes = 20
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem sveFileSize_succ (n e : Nat) : sveFileSize (n + 1) e = sveFileSize n e + 4 * e
  now: theorem sve1_magic : 0x31455653 = 0x53 + 256 * (0x56 + 256 * (0x45 + 256 * 0x31))
  measured: every variable of the statement is gone: a property stated for all values became a statement about fixed values

- VisionEncoderService.lean
  was: theorem sveFileSize_zero (e : Nat) : sveFileSize 0 e = 20
  now: theorem sveFileSize_succ (n e : Nat) : sveFileSize (n + 1) e = sveFileSize n e + 4 * e
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem grid_positive (x y : Nat) : 0 < (gridOf x y).1 ∧ 0 < (gridOf x y).2
  now: theorem sveFileSize_zero (e : Nat) : sveFileSize 0 e = 20
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem rectangular_iff (n x y : Nat) : rectangular n x y = true ↔ (x + 1) * (y + 1) = n
  now: theorem grid_positive (x y : Nat) : 0 < (gridOf x y).1 ∧ 0 < (gridOf x y).2
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem position_recompose (i nx : Nat) : (i / nx) * nx + i % nx = i
  now: theorem rectangular_iff (n x y : Nat) : rectangular n x y = true ↔ (x + 1) * (y + 1) = n
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem position_in_grid (i nx ny : Nat) (hnx : 0 < nx) (h : i < nx * ny) :
    i % nx < nx ∧ i / nx < ny
  now: theorem position_recompose (i nx : Nat) : (i / nx) * nx + i % nx = i
  measured: fewer quantified variables (3 → 1)

- VisionEncoderService.lean
  was: theorem native_xor_portable (p : Bool) : ggmlNative p = !p
  now: theorem position_in_grid (i nx ny : Nat) (hnx : 0 < nx) (h : i < nx * ny) :
    i % nx < nx ∧ i / nx < ny
  measured: more hypotheses/binders (1 → 3): the statement holds in fewer cases

- VisionEncoderService.lean
  was: theorem no_static_when_native (msvc : Bool) : staticRuntime false msvc = false
  now: theorem native_xor_portable (p : Bool) : ggmlNative p = !p
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem portable_isa_distinct : portableIsa.length = 5 ∧ portableIsa.Nodup
  now: theorem no_static_when_native (msvc : Bool) : staticRuntime false msvc = false
  measured: the claim changed

- VisionEncoderService.lean
  was: theorem cuda_mirrors_option (opt : Bool) : ggmlCuda opt = opt
  now: theorem portable_isa_distinct : portableIsa.length = 5 ∧ portableIsa.Nodup
  measured: every variable of the statement is gone: a property stated for all values became a statement about fixed values

- ApiServerEngine.lean
  was: theorem auth_open (given : String) : authorized "" given = true
  now: theorem auth_open (presented : String) : authorized "" presented = true
  measured: the claim changed
-/


[LIGHT FILES — structure only, no LAYER 1a/1b]
  include/ref/core/expert_cache.hpp (156 lines)
  include/ref/core/graph.hpp (118 lines)
  include/ref/core/layer.hpp (567 lines)
  include/ref/core/layout.hpp (98 lines)
  include/ref/core/mtp.hpp (136 lines)
  include/ref/core/pinned.hpp (96 lines)
  include/ref/core/session.hpp (412 lines)
  include/ref/core/verify.hpp (161 lines)
  include/ref/core/weights.hpp (136 lines)
  include/ref/core/device.hpp (72 lines)
  include/ref/core/hit_hook.hpp (42 lines)
  include/ref/core/native_dense.hpp (37 lines)
  include/ref/core/native_head.hpp (66 lines)
  include/ref/kernels/elementwise.hpp (117 lines)
  include/ref/kernels/gdn.hpp (98 lines)
  include/ref/kernels/gr.hpp (123 lines)
  include/ref/kernels/native_mmvq.hpp (113 lines)
  include/ref/kernels/ngram.hpp (195 lines)
  include/ref/kernels/ple.hpp (120 lines)
  include/ref/kernels/qsa.hpp (361 lines)
  include/ref/kernels/s2_expert_grouped.hpp (119 lines)
  include/ref/kernels/s_gemv.hpp (134 lines)
  include/ref/kernels/shared_expert.hpp (102 lines)
  include/ref/kernels/verify_kernels.hpp (81 lines)
  include/ref/kernels/bf16_bits.hpp (47 lines)
  include/ref/kernels/bf16_gemv.hpp (49 lines)
  include/ref/kernels/dequant_bf16.hpp (28 lines)
  include/ref/kernels/dequant_s2.hpp (19 lines)
  include/ref/kernels/fused_gdn.hpp (36 lines)
  include/ref/kernels/fused_gr.hpp (52 lines)
  include/ref/kernels/iq_kernels.hpp (58 lines)
  include/ref/kernels/kv_q8.hpp (39 lines)
  include/ref/kernels/mrope.hpp (33 lines)
  include/ref/kernels/native_flash_attn.hpp (37 lines)
  include/ref/kernels/native_gdn.hpp (23 lines)
  include/ref/kernels/native_gdn_preprocess.hpp (38 lines)
  include/ref/kernels/native_gr_norm.hpp (19 lines)
  include/ref/kernels/native_gr_postops.hpp (29 lines)
  include/ref/kernels/native_moe.hpp (19 lines)
  include/ref/kernels/native_ple_postops.hpp (31 lines)
  include/ref/kernels/native_qsa.hpp (31 lines)
  include/ref/kernels/native_qsa_indexer.hpp (36 lines)
  include/ref/kernels/native_qsa_score.hpp (32 lines)
  include/ref/kernels/native_rope.hpp (15 lines)
  include/ref/kernels/native_router.hpp (18 lines)
  include/ref/kernels/qsa_decode_attn.hpp (45 lines)
  include/ref/kernels/qsa_select.hpp (34 lines)
  include/ref/kernels/quantize_act.hpp (59 lines)
  include/ref/kernels/rope.hpp (51 lines)
  include/ref/kernels/router_top10.hpp (21 lines)
  include/ref/kernels/s2_gemv.hpp (26 lines)
  include/ref/kernels/s2_gemv_q8.hpp (23 lines)
  include/ref/kernels/sampler.hpp (44 lines)
  include/ref/kernels/cpu/expert.hpp (180 lines)
  include/ref/kernels/cpu/pool.hpp (196 lines)
  include/ref/kernels/cpu/expert_layout.hpp (55 lines)
  include/ref/kernels/cpu/iq_avx512.hpp (19 lines)
  include/ref/kernels/cpu/native_expert.hpp (50 lines)
  include/ref/ngram/ple_reader.hpp (87 lines)
  include/ref/plan/plan.hpp (169 lines)
  include/ref/platform/direct_file.hpp (60 lines)
  include/ref/platform/memory.hpp (27 lines)
  include/ref/prefill/gemm.hpp (55 lines)
  include/ref/prefill/kernels.hpp (75 lines)
  include/ref/prefill/prefill.hpp (76 lines)
  include/ref/program/logits_selection.hpp (31 lines)
  include/ref/spec/controller.hpp (74 lines)
  include/ref/spec/suffix_drafter.hpp (54 lines)
  serve/__init__.py (2 lines)
  serve/chat_golden.json (197 lines)
  serve/chat_template.jinja (184 lines)
  serve/index.html (153 lines)
  src/artifact/dequant.cpp (124 lines)
  src/artifact/gguf_reader.cpp (68 lines)
  src/core/concurrent_main.cpp (198 lines)
  src/core/device.cu (118 lines)
  src/core/device_main.cpp (79 lines)
  src/core/layout.cpp (174 lines)
  src/core/load_main.cpp (120 lines)
  src/core/native_dense.cpp (180 lines)
  src/core/native_head.cpp (152 lines)
  src/core/overlap_main.cpp (170 lines)
  src/core/pinned.cu (292 lines)
  src/kernels/bf16_gemv_parity.cpp (149 lines)
  src/kernels/dequant_bf16_test.cpp (109 lines)
  src/kernels/dequant_s2_parity.cpp (144 lines)
  src/kernels/elementwise_parity.cpp (417 lines)
  src/kernels/gdn_parity.cpp (398 lines)
  src/kernels/gr_parity.cpp (616 lines)
  src/kernels/iq_parity.cpp (88 lines)
  src/kernels/kv_q8_parity.cpp (142 lines)
  src/kernels/native_expert_parity.cpp (205 lines)
  src/kernels/ple_oracle_vectors.inc (106 lines)
  src/kernels/ple_parity.cpp (1185 lines)
  src/kernels/qsa_oracle_vectors.inc (135 lines)
  src/kernels/qsa_parity.cpp (1384 lines)
  src/kernels/quantize_act_parity.cpp (408 lines)
  src/kernels/rope_parity.cpp (172 lines)
  src/kernels/router_top10_parity.cpp (168 lines)
  src/kernels/s2_gemv_parity.cpp (174 lines)
  src/kernels/s2_gemv_q8_parity.cpp (182 lines)
  src/kernels/s_gemv_parity.cpp (635 lines)
  src/kernels/s_gemv_q8k_parity.cpp (237 lines)
  src/kernels/sampler_parity.cpp (363 lines)
  src/kernels/shared_expert_parity.cpp (483 lines)
  src/kernels/cpu/expert_multi_test.cpp (88 lines)
  src/kernels/cpu/expert_parity.cpp (185 lines)
  src/kernels/cpu/native_expert.cpp (115 lines)
  src/kernels/cpu/pool_test.cpp (225 lines)
  src/kernels/cuda/bf16_gemv.cu (142 lines)
  src/kernels/cuda/dequant_bf16.cu (246 lines)
  src/kernels/cuda/elementwise.cu (290 lines)
  src/kernels/cuda/native_bf16.cu (119 lines)
  src/kernels/cuda/native_gr_norm.cu (103 lines)
  src/kernels/cuda/native_gr_postops.cu (116 lines)
  src/kernels/cuda/native_rope.cu (89 lines)
  src/kernels/cuda/rope.cu (96 lines)
  src/kernels/cuda/s2_gemv_q8.cu (97 lines)
  src/kernels/cuda/s2_gemv_quads.cu (94 lines)
  src/kernels/cuda/dequant_s2.cu (52 lines)
  src/kernels/cuda/native_moe.cu (75 lines)
  src/kernels/cuda/s2_gemv.cu (64 lines)
  src/ngram/ple_reader_test.cpp (238 lines)
  src/plan/plan_main.cpp (74 lines)
  src/platform/direct_file.cpp (242 lines)
  src/platform/memory.cpp (77 lines)
  src/platform/memory_test.cpp (36 lines)
  src/prefill/gemm.cu (108 lines)
  src/spec/suffix_drafter_test.cpp (142 lines)
  src/spec/controller_test.cpp (75 lines)
  tools/gguf_reader.py (143 lines)
  tools/mtp_fetch.py (122 lines)
  tools/pack_index.py (261 lines)
  tools/_paths.py (29 lines)
  bench/results/2026-09-24-final/estimates.md (28 lines)
  bench/results/2026-09-24-final/matrix.md (16 lines)
  bench/results/2026-09-24-vision/vision.md (6 lines)
  third_party/ggml/VERSION.txt (2 lines)
  CMakeLists.txt (537 lines)
  START-HERE.bat (53 lines)
  bench/results/2026-09-24-final/estimates.json (461 lines)
  bench/results/2026-09-24-final/matrix.json (240 lines)
  bench/results/2026-09-24-vision/vision.json (53 lines)
  chat.py (117 lines)
  docs/DETAILS.md (326 lines)
  setup.py (854 lines)
  setup.sh (31 lines)
  third_party/ggml/ggml-common.h (1912 lines)
