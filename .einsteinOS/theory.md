
## FILE: /mnt/user-data/outputs/ExpertCacheResidency.lean
/-!
# MODULE: expert_cache_residency
SOURCES: include/ref/core/expert_cache.hpp, src/core/expert_cache.cpp
VRAM-resident expert tier: slot storage, the (layer, expert) -> slot residency table, the
per-layer admission quotas (R4.2g), the STRP profile reader's validation, and 256-byte slot alignment.

## Source semantics
* `slot_of` is bounds-checked and returns `kNotResident` (-1) out of range.
* `admit` never evicts. Global mode uses one counter `next_free_ <= slots_`; per-layer mode gives layer `l`
  the slot range `[l*q, (l+1)*q)` with `q = slots/n_layers`, the last layer taking the remainder.
* The ranges tile `0..slots` exactly; each layer's cursor never exceeds its upper bound.
* `read_expert_profile` accepts only a header whose geometry matches, `n_ranked <= slots`, and pairs in range.
* `open_sized` aligns each slot to 256 bytes.

## Form template identifiers
ExpertCache -> tier_store; admit -> claim_slot; slot_of -> lookup_slot; layer_slot_range -> quota_range;
kNotResident -> absent_marker; read_expert_profile -> load_static_plan; open_sized -> open_variable_slots.
-/

namespace ExpertCacheResidency

-- § per-layer slot ranges

/-- Lower bound of layer `l`'s slot range: `l * (slots / nl)`. -/
def lo (slots nl l : Nat) : Nat := l * (slots / nl)

/-- Upper bound of layer `l`'s slot range; the last layer takes the remainder. -/
def hi (slots nl l : Nat) : Nat := if l = nl - 1 then slots else (l + 1) * (slots / nl)

/-- Each layer's range is well formed: `lo <= hi`. -/
theorem lo_le_hi {slots nl l : Nat} (h : 0 < nl) (hl : l < nl) : lo slots nl l ≤ hi slots nl l := by
  unfold lo hi
  split
  · have h1 : l * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    have h2 : nl * (slots / nl) ≤ slots := Nat.mul_div_le slots nl
    exact Nat.le_trans h1 h2
  · exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- No layer's range extends past the arena: `hi <= slots`. -/
theorem hi_le_slots {slots nl l : Nat} (hl : l < nl) : hi slots nl l ≤ slots := by
  unfold hi
  split
  · exact Nat.le_refl _
  · have h1 : (l + 1) * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hl)
    have h2 : nl * (slots / nl) ≤ slots := Nat.mul_div_le slots nl
    exact Nat.le_trans h1 h2

/-- Adjacent layers' ranges abut: layer `l`'s upper bound is layer `l+1`'s lower bound. -/
theorem hi_eq_next_lo {slots nl l : Nat} (hl : l + 1 < nl) : hi slots nl l = lo slots nl (l + 1) := by
  have hne : l ≠ nl - 1 := by omega
  simp [hi, lo, hne]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem last_hi {slots nl : Nat} : hi slots nl (nl - 1) = slots := by
  simp [hi]

/-- Sum of the per-layer quotas over the first `m` layers. -/
def quotaSum (slots nl : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => quotaSum slots nl m + (hi slots nl m - lo slots nl m)

/-- Before the last layer, the first `m` quotas sum to `m * q`. -/
theorem quotaSum_prefix {slots nl : Nat} (hnl : 0 < nl) :
    ∀ m, m ≤ nl - 1 → quotaSum slots nl m = m * (slots / nl) := by
  intro m
  induction m with
  | zero => intro _; simp [quotaSum]
  | succ m ih =>
    intro hm
    have hm' : m ≤ nl - 1 := by omega
    have hne : m ≠ nl - 1 := by omega
    have e1 : hi slots nl m = (m + 1) * (slots / nl) := by simp [hi, hne]
    have e2 : lo slots nl m = m * (slots / nl) := rfl
    simp only [quotaSum]
    rw [ih hm', e1, e2, Nat.add_mul, Nat.one_mul, Nat.add_sub_cancel_left]

/-- The quotas of all `nl` layers sum to exactly `slots`: per-layer admission can fill the arena and no more. -/
theorem quotaSum_total {slots nl : Nat} (hnl : 0 < nl) : quotaSum slots nl nl = slots := by
  obtain ⟨n, rfl⟩ : ∃ n, nl = n + 1 := ⟨nl - 1, by omega⟩
  have hp : quotaSum slots (n + 1) n = n * (slots / (n + 1)) :=
    quotaSum_prefix (nl := n + 1) hnl n (by omega)
  have hh : hi slots (n + 1) n = slots := by simp [hi]
  have hl : lo slots (n + 1) n = n * (slots / (n + 1)) := rfl
  have h1 : n * (slots / (n + 1)) ≤ (n + 1) * (slots / (n + 1)) := Nat.mul_le_mul_right _ (Nat.le_succ n)
  have h2 : (n + 1) * (slots / (n + 1)) ≤ slots := Nat.mul_div_le slots (n + 1)
  show quotaSum slots (n + 1) n + (hi slots (n + 1) n - lo slots (n + 1) n) = slots
  rw [hp, hh, hl]
  omega

-- § admission

/-- One admission attempt against a cursor and an upper bound: claim the slot if one is free, else refuse. -/
def admitStep (next hi : Nat) : Nat := if next < hi then next + 1 else next

/-- Admission never pushes the cursor past the bound. -/
theorem admitStep_le {next hi : Nat} (h : next ≤ hi) : admitStep next hi ≤ hi := by
  unfold admitStep
  split <;> omega

/-- A full range refuses: the cursor does not move (no eviction). -/
theorem admitStep_full {next hi : Nat} (h : hi ≤ next) : admitStep next hi = next := by
  unfold admitStep
  split <;> omega

/-- Repeating `k` admission attempts. -/
def admitN : Nat → Nat → Nat → Nat
  | 0, n, _ => n
  | k + 1, n, h => admitN k (admitStep n h) h

/-- However many attempts are made, a cursor that started within its bound stays within it. -/
theorem admitN_le {hi : Nat} : ∀ (k n : Nat), n ≤ hi → admitN k n hi ≤ hi := by
  intro k
  induction k with
  | zero => intro n h; simpa [admitN] using h
  | succ k ih =>
    intro n h
    exact ih _ (admitStep_le h)

-- § why the global counter failed (R4.2g measurement)

/-- 256 slots under global arrival-order admission are consumed inside 26 layers' worth of position 0
(10 experts each), i.e. more than 25 and at most 26 layers. -/
theorem global_policy_confinement : 25 * 10 < 256 ∧ 256 ≤ 26 * 10 := by decide

-- § STRP profile validation

/-- Every ranked pair lies inside the `nl x ne` table. -/
def pairsOk (nl ne : Nat) (ps : List (Nat × Nat)) : Bool :=
  ps.all fun p => decide (p.1 < nl) && decide (p.2 < ne)

/-- A pair in a validated list is in range. -/
theorem pairsOk_mem {nl ne : Nat} {ps : List (Nat × Nat)} (h : pairsOk nl ne ps = true)
    {p : Nat × Nat} (hp : p ∈ ps) : p.1 < nl ∧ p.2 < ne := by
  unfold pairsOk at h
  rw [List.all_eq_true] at h
  simpa using h p hp

/-- Header acceptance: matching geometry and `n_ranked <= slots`. -/
def headerOk (nl ne want nr L E : Nat) : Bool :=
  decide (nl = L) && decide (ne = E) && decide (nr ≤ want)

/-- An accepted header never claims more ranked pairs than slots. -/
theorem headerOk_ranked_le {nl ne want nr L E : Nat} (h : headerOk nl ne want nr L E = true) :
    nr ≤ want := by
  simp [headerOk] at h
  omega

/-- An accepted header has exactly the model's geometry. -/
theorem headerOk_geometry {nl ne want nr L E : Nat} (h : headerOk nl ne want nr L E = true) :
    nl = L ∧ ne = E := by
  simp [headerOk] at h
  omega

-- § byte accounting and alignment

/-- Rounds `n` up to a multiple of 256 (the slot alignment in `open_sized`). -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment never shrinks a slot. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by
  unfold align256
  omega

/-- Aligned slot sizes are multiples of 256, so kernel vector loads start aligned. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by
  unfold align256
  omega

/-- Alignment wastes fewer than 256 bytes per slot. -/
theorem align256_lt (n : Nat) : align256 n < n + 256 := by
  unfold align256
  omega

/-- Uniform-slot arena size: `slots * blob`. -/
def arenaBytes (slots blob : Nat) : Nat := slots * blob

/-- 4,105 slots of 1,382,400 B is no more than the 5,413 MiB the design note quotes. -/
theorem slots4105_fits_note : arenaBytes 4105 1382400 ≤ 5413 * 1048576 := by decide

/-- ... and fits in the 6,414 MiB measured free at steady state. -/
theorem slots4105_fits_free : arenaBytes 4105 1382400 ≤ 6414 * 1048576 := by decide

end ExpertCacheResidency



## FILE: /mnt/user-data/outputs/ExpertSourceDispatch.lean
/-!
# MODULE: expert_source_dispatch
SOURCES: include/ref/core/expert_source.hpp, src/core/expert_source.cpp
The adapter between the host loop and the CPU expert pool: blob lookup in the mapped/arena source, the
GPU/CPU hit-miss split by router index, the PCIe share of misses, and the latching failure flag.

## Source semantics
* A blob index is `layer * n_expert + expert`; `expert >= n_expert` is rejected separately because the flat
  index alone aliases the next layer's expert 0.
* CPU rows for hits are zeroed and GPU rows for misses are zeroed, so `parts + hit_out` is the full answer.
* `njobs` (CPU jobs) counts the misses; hits + jobs = k.
* The PCIe share is `nmiss * pcie_num / 256 <= nmiss` of the distinct misses.
* `failed` latches: once set, every later dispatch returns immediately.
* The hit path is all-or-nothing (`hits_ready`).

## Form template identifiers
ExpertSource -> blob_provider; FileExpertSource -> mapped_provider; ArenaExpertSource -> resident_provider;
ExpertDispatch -> split_dispatcher; expert_pool_dispatch -> cpu_half; expert_hit_run -> gpu_half;
pcie_num -> pcie_share_numerator.
-/

namespace ExpertSourceDispatch

-- § blob indexing

/-- Flat blob index `layer * n_expert + expert`. -/
def blobIndex (ne l e : Nat) : Nat := l * ne + e

/-- With `expert < n_expert` and `layer < n_layers` the flat index is in range. -/
theorem blobIndex_lt {ne nl l e : Nat} (he : e < ne) (hl : l < nl) : blobIndex ne l e < nl * ne := by
  unfold blobIndex
  have h1 : (l + 1) * ne ≤ nl * ne := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hl)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Why the separate `expert >= n_expert` check exists: without it, `blob(0, n_expert)` is `blob(1, 0)`. -/
theorem blobIndex_aliases_without_check (ne : Nat) : blobIndex ne 0 ne = blobIndex ne 1 0 := by
  unfold blobIndex
  simp

/-- With both coordinates checked, distinct (layer, expert) pairs give distinct indices. -/
theorem blobIndex_injective {ne l1 l2 e1 e2 : Nat} (hne : 0 < ne) (h1 : e1 < ne) (h2 : e2 < ne)
    (h : blobIndex ne l1 e1 = blobIndex ne l2 e2) : l1 = l2 ∧ e1 = e2 := by
  unfold blobIndex at h
  have hd : (e1 + l1 * ne) / ne = (e2 + l2 * ne) / ne := by
    rw [Nat.add_comm e1, Nat.add_comm e2, h]
  rw [Nat.add_mul_div_right _ _ hne, Nat.add_mul_div_right _ _ hne,
    Nat.div_eq_of_lt h1, Nat.div_eq_of_lt h2] at hd
  have hl : l1 = l2 := by omega
  subst hl
  exact ⟨rfl, by omega⟩

-- § the hit/miss split

/-- The CPU's row for one router index: zero when the GPU owns it. -/
def cpuRow (hit : Bool) (v : Nat) : Nat := if hit then 0 else v

/-- The GPU's row for one router index: zero when the CPU owns it. -/
def gpuRow (hit : Bool) (v : Nat) : Nat := if hit then v else 0

/-- Summing the two zero-padded halves yields exactly the expert's output, whoever computed it. -/
theorem split_partition (hit : Bool) (v : Nat) : cpuRow hit v + gpuRow hit v = v := by
  cases hit <;> simp [cpuRow, gpuRow]

/-- Number of hits in a decision vector. -/
def countHit : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + countHit bs

/-- Number of misses (= CPU jobs, `njobs`) in a decision vector. -/
def countMiss : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 0 else 1) + countMiss bs

/-- Every routed expert is either a hit or a CPU job; none is dropped or doubled. -/
theorem hits_plus_jobs (bs : List Bool) : countHit bs + countMiss bs = bs.length := by
  induction bs with
  | nil => rfl
  | cons b bs ih => cases b <;> simp [countHit, countMiss] <;> omega

-- § PCIe share of misses

/-- The GPU takes `nmiss * pcie_num / 256` of the distinct misses over PCIe. -/
def pcieShare (nmiss pn : Nat) : Nat := nmiss * pn / 256

/-- With `pcie_num <= 256`, the PCIe share never exceeds the number of misses. -/
theorem pcie_share_le {nmiss pn : Nat} (hp : pn ≤ 256) : pcieShare nmiss pn ≤ nmiss := by
  unfold pcieShare
  apply Nat.div_le_of_le_mul
  have := Nat.mul_le_mul_left nmiss hp
  omega

/-- The last `m` misses in routing order (rank `>= nmiss - m`) number exactly `m` when `m <= nmiss`. -/
theorem pcie_tail_count {nmiss m : Nat} (h : m ≤ nmiss) : nmiss - (nmiss - m) = m := by
  omega

/-- The plan's scratch arrays: at most 8 tokens x 10 experts fit the `kind[128]` array. -/
theorem plan_fits_kind_array {ntok k : Nat} (ht : ntok ≤ 8) (hk : k ≤ 10) : ntok * k ≤ 128 := by
  have : ntok * k ≤ 8 * 10 := Nat.mul_le_mul ht hk
  omega

-- § failure latch and configuration

/-- `failed` after one more dispatch step with outcome `ok`. -/
def stepFailed (failed ok : Bool) : Bool := failed || !ok

/-- Once failed, the adapter stays failed: clearing the message cannot re-arm it. -/
theorem failure_latches (ok : Bool) : stepFailed true ok = true := by
  simp [stepFailed]

/-- A success never clears an earlier failure, and a failure is always recorded. -/
theorem failure_recorded (failed : Bool) : stepFailed failed false = true := by
  simp [stepFailed]

/-- `hits_ready` is the conjunction of all its prerequisites. -/
def hitsReady (flags : List Bool) : Bool := flags.all id

/-- If the hit path is ready, every prerequisite holds (no half-configured hit path). -/
theorem hitsReady_each {flags : List Bool} (h : hitsReady flags = true) : ∀ b ∈ flags, b = true := by
  intro b hb
  unfold hitsReady at h
  rw [List.all_eq_true] at h
  simpa using h b hb

-- § artifact geometry (measured constants)

/-- Expert blob size: three 640x2560 matrices at 9/32 bytes per element (2.25 bits). -/
theorem blob_bytes_derivation : 3 * 640 * 2560 * 9 / 32 = 1382400 := by decide

/-- `experts.bin` holds exactly 48 x 512 blobs and is 33,973,862,400 bytes. -/
theorem experts_bin_size : 48 * 512 * 1382400 = 33973862400 := by decide

/-- One token reads 48 layers x 10 experts = 663,552,000 bytes (the quoted 663.6 MB). -/
theorem bytes_per_token : 48 * 10 * 1382400 = 663552000 := by decide

/-- A VRAM-resident tier of all 512 experts of one layer is 707,788,800 bytes (the MTP layer's ~708 MB). -/
theorem one_layer_experts_bytes : 512 * 1382400 = 707788800 := by decide

end ExpertSourceDispatch



## FILE: /mnt/user-data/outputs/GdnQsaLayerGeometry.lean
/-!
# MODULE: gdn_qsa_layer_geometry
SOURCES: include/ref/core/layer.hpp
One GDN/QSA block composed from kernels: layer-type split (12 QSA / 36 GDN), the q|k|v conv channel
layout, the QSA q/gate halves split, INT8 KV sizing, the pre/post (and half) graph stage split, and PLE row bytes.

## Source semantics
* QSA layers are at indices 3, 7, ... 47; the other 36 are GDN.
* `conv_channels = 10240 = q (16x128) | k (16x128) | v (48x128)`; l2-norm applies to q and k only.
* `q_full` is split per head into halves: q = first `head_dim`, gate = second `head_dim`.
* `half = 1` is stages 0..2, `half = 2` is stages 3..4, `half = 0` is both; `preP[k]` is stages 0..k-1.
* INT8 KV stores one byte per value plus one fp16 scale per 64 values.

## Form template identifiers
block_layer_pre -> front_half; block_layer_post -> back_half; is_qsa_layer -> is_attention_layer;
GdnBuffers -> recurrent_scratch; QsaState -> kv_state; Doorbell -> handoff_cell; PleRun -> layer1_embedding.
-/

namespace GdnQsaLayerGeometry

-- § layer-type split

/-- QSA layers sit at indices congruent to 3 mod 4. -/
def isQsa (i : Nat) : Bool := i % 4 == 3

/-- Exactly 12 of the 48 layers are QSA. -/
theorem qsa_count : ((List.range 48).filter isQsa).length = 12 := by decide

/-- Exactly 36 of the 48 layers are GDN. -/
theorem gdn_count : ((List.range 48).filter (fun i => !isQsa i)).length = 36 := by decide

-- § conv channels

/-- q, k are 16 heads of width 128 each; v is 48 heads of width 128. -/
theorem conv_channels : 16 * 128 + 16 * 128 + 48 * 128 = 10240 := by decide

/-- The l2 norm covers the 32 q/k heads, not the 48 value heads. -/
theorem l2norm_heads : 16 + 16 = 32 := by decide

-- § QSA q/gate halves

/-- Index of q element `d` of head `h` inside `q_full` (first half of the head's `2*hd` block). -/
def qIdx (hd h d : Nat) : Nat := h * (2 * hd) + d

/-- Index of gate element `d` of head `h` inside `q_full` (second half of the block). -/
def gIdx (hd h d : Nat) : Nat := h * (2 * hd) + hd + d

/-- Gate indices stay inside `q_full` (`n_head * 2 * head_dim` floats). -/
theorem gIdx_lt {nh hd h d : Nat} (hh : h < nh) (hd' : d < hd) : gIdx hd h d < nh * (2 * hd) := by
  unfold gIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- q indices stay inside `q_full`. -/
theorem qIdx_lt {nh hd h d : Nat} (hh : h < nh) (hd' : d < hd) : qIdx hd h d < nh * (2 * hd) := by
  unfold qIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Within one head, q and gate never collide (contiguous halves, not interleaved). -/
theorem q_gate_disjoint {hd h d d' : Nat} (hd' : d < hd) : qIdx hd h d ≠ gIdx hd h d' := by
  unfold qIdx gIdx
  omega

-- § graph stage split

/-- Stages run by `half` (0 = both). -/
def stages : Nat → List Nat
  | 1 => [0, 1, 2]
  | 2 => [3, 4]
  | _ => [0, 1, 2, 3, 4]

/-- The two halves concatenate to the whole layer. -/
theorem halves_cover : stages 1 ++ stages 2 = stages 0 := rfl

/-- The two halves share no stage. -/
theorem halves_disjoint : ∀ s ∈ stages 1, s ∉ stages 2 := by decide

/-- `preP[k]` runs the first `k` stages. -/
def prefixStages (k : Nat) : List Nat := List.range k

/-- Consecutive prefixes differ by exactly one stage, so differencing times isolates it. -/
theorem prefix_step (k : Nat) : prefixStages (k + 1) = prefixStages k ++ [k] := List.range_succ

/-- Prefix `k` has `k` stages. -/
theorem prefix_length (k : Nat) : (prefixStages k).length = k := by
  simp [prefixStages]

-- § sizing

/-- INT8 KV with one fp16 scale per 64 values never exceeds the fp16 cache size (for whole groups). -/
theorem int8_kv_smaller (m : Nat) : 64 * m + 2 * m ≤ 2 * (64 * m) := by omega

/-- PLE traffic per token: sixteen 90-byte rows = 1440 bytes. -/
theorem ple_row_bytes : 16 * 90 = 1440 := by decide

end GdnQsaLayerGeometry



## FILE: /mnt/user-data/outputs/GraphCaptureRegistry.lean
/-!
# MODULE: graph_capture_registry
SOURCES: include/ref/core/graph.hpp, src/core/graph.cpp
Capture-once / replay-many registry of CUDA graphs keyed by (layer_type, n_tokens), plus the bounded
event-polling wait.

## Source semantics
* `record` on an existing key is a no-op and does not run the body; `captures()` equals `size()`.
* A capture with zero nodes is an error, never an empty graph.
* `wait_ms` polls an event query: success -> true, not-ready -> keep polling until the deadline,
  any other status -> false. It never reports completion that did not occur.

## Form template identifiers
GraphRegistry -> graph_cache; CapturedGraph -> recorded_graph; record -> capture_once;
launch -> replay_and_wait; wait_ms -> bounded_poll; LayerType -> kind_tag.
-/

namespace GraphCaptureRegistry

-- § registry

/-- Abstract registry state: the recorded keys and the capture counter. -/
structure Reg where
  keys : List (Nat × Nat)
  captures : Nat

/-- `record`: do nothing if the key exists; otherwise capture it and count the capture. -/
def Reg.record (r : Reg) (k : Nat × Nat) : Reg :=
  if k ∈ r.keys then r else { keys := k :: r.keys, captures := r.captures + 1 }

/-- Invariant: `captures()` equals `size()`. -/
def Inv (r : Reg) : Prop := r.captures = r.keys.length

/-- Recording the same key twice equals recording it once (the body runs exactly once). -/
theorem record_idem (r : Reg) (k : Nat × Nat) : (r.record k).record k = r.record k := by
  by_cases h : k ∈ r.keys <;> simp [Reg.record, h]

/-- After `record`, the key is present. -/
theorem record_mem (r : Reg) (k : Nat × Nat) : k ∈ (r.record k).keys := by
  by_cases h : k ∈ r.keys <;> simp [Reg.record, h]

/-- Recording never forgets an existing key. -/
theorem record_preserves {r : Reg} {k k' : Nat × Nat} (h : k' ∈ r.keys) : k' ∈ (r.record k).keys := by
  by_cases hk : k ∈ r.keys <;> simp [Reg.record, hk, h]

/-- `captures = size` is preserved by `record`. -/
theorem record_inv {r : Reg} (k : Nat × Nat) (hi : Inv r) : Inv (r.record k) := by
  unfold Inv at *
  by_cases h : k ∈ r.keys <;> simp [Reg.record, h, hi]

/-- The two layer types never share a key: GDN (0) and QSA (1) at the same token count are distinct. -/
theorem layer_keys_distinct (n : Nat) : ((0, n) : Nat × Nat) ≠ (1, n) := by
  simp

-- § capture validity

/-- `end` succeeds only if the capture recorded at least one node. -/
def endOk (nodes : Nat) : Bool := decide (0 < nodes)

/-- A zero-node capture is rejected. -/
theorem endOk_zero : endOk 0 = false := by decide

/-- Any non-empty capture is accepted. -/
theorem endOk_pos {n : Nat} (h : 0 < n) : endOk n = true := by
  simp [endOk, h]

-- § bounded wait

/-- Outcomes of one `cudaEventQuery`. -/
inductive Q where
  | success
  | notReady
  | err

/-- `wait_ms` over a fuel bound (the deadline) and a stream of query outcomes. -/
def wait : Nat → List Q → Bool
  | 0, _ => false
  | _ + 1, [] => false
  | _ + 1, Q.success :: _ => true
  | n + 1, Q.notReady :: t => wait n t
  | _ + 1, Q.err :: _ => false

/-- An expired deadline reports failure, never success. -/
theorem wait_zero (qs : List Q) : wait 0 qs = false := by
  cases qs <;> rfl

/-- A first-poll success returns true. -/
theorem wait_success (n : Nat) (t : List Q) : wait (n + 1) (Q.success :: t) = true := rfl

/-- A real error (not "not ready") returns false immediately. -/
theorem wait_err (n : Nat) (t : List Q) : wait (n + 1) (Q.err :: t) = false := rfl

/-- Not-ready consumes one unit of the deadline and continues. -/
theorem wait_notReady (n : Nat) (t : List Q) : wait (n + 1) (Q.notReady :: t) = wait n t := rfl

/-- `wait` returns true only if some poll actually observed completion. -/
theorem wait_true_has_success : ∀ (n : Nat) (qs : List Q), wait n qs = true → Q.success ∈ qs := by
  intro n
  induction n with
  | zero => intro qs h; cases qs <;> simp [wait] at h
  | succ n ih =>
    intro qs h
    cases qs with
    | nil => simp [wait] at h
    | cons q t =>
      cases q with
      | success => exact List.mem_cons_self
      | notReady => exact List.mem_cons_of_mem _ (ih t h)
      | err => simp [wait] at h

end GraphCaptureRegistry



## FILE: /mnt/user-data/outputs/MtpDraftChain.lean
/-!
# MODULE: mtp_draft_chain
SOURCES: include/ref/core/mtp.hpp
The MTP draft layer on the GPU: draft cell placement, the attention window, and the graph-array bounds.

## Source semantics
* A round drafts `T-1` tokens at cells `p+a+1, p+a+2, ...` after the last accepted row `a`.
* Attention is over the last `window` cells (`0` = every cell).
* Per-`T` graph arrays have 9 entries, so `T <= 8`.
* All 512 routed experts of the draft layer sit in VRAM (about 708 MB).

## Form template identifiers
MtpDrafter -> draft_engine; draft -> draft_round; draft_first -> first_round; prefill -> prompt_catchup;
window_ -> attention_window; step_exec_ -> per_step_graphs.
-/

namespace MtpDraftChain

-- § draft cells

/-- Cells of the draft chain after accepted row `a` of a window starting at `p`, for a window of `T` tokens. -/
def draftCells (p a T : Nat) : List Nat := (List.range (T - 1)).map (fun j => p + a + 1 + j)

/-- A round of window size `T` produces exactly `T - 1` draft cells. -/
theorem draftCells_length (p a T : Nat) : (draftCells p a T).length = T - 1 := by
  simp [draftCells]

/-- Every draft cell is strictly after the last accepted cell `p + a`. -/
theorem draftCells_gt {p a T c : Nat} (h : c ∈ draftCells p a T) : p + a < c := by
  simp [draftCells] at h
  obtain ⟨j, _, hj⟩ := h
  omega

-- § attention window

/-- First cell attended when `n` cells exist and the window is `w` (0 = all). -/
def winLo (n w : Nat) : Nat := if w = 0 then 0 else n - w

/-- The window never starts after the end. -/
theorem winLo_le (n w : Nat) : winLo n w ≤ n := by
  unfold winLo
  split <;> omega

/-- A positive window attends at most `w` cells. -/
theorem window_bound {n w : Nat} (hw : 0 < w) : n - winLo n w ≤ w := by
  unfold winLo
  have : w ≠ 0 := by omega
  simp [this]
  omega

/-- A zero window attends every cell. -/
theorem window_zero_all (n : Nat) : winLo n 0 = 0 := by
  simp [winLo]

-- § bounds

/-- `exec_[9]` is indexed by `T`; with `T <= 8` the index is in range. -/
theorem exec_index_lt {T : Nat} (h : T ≤ 8) : T < 9 := by omega

/-- The 512 resident experts take under 708,000,000 bytes... -/
theorem experts_vram : 512 * 1382400 < 708000000 := by decide

/-- ...and no less than 707,000,000 (the quoted 708 MB). -/
theorem experts_vram_lower : 707000000 < 512 * 1382400 := by decide

end MtpDraftChain



## FILE: /mnt/user-data/outputs/SessionTokenLoop.lean
/-!
# MODULE: session_token_loop
SOURCES: include/ref/core/session.hpp
One token through 48 layers: per-layer state indexing (12 QSA / 36 GDN), the per-layer graph pair
(`pre`/`post`), the `dump_layers` snapshot buffer layout, the prefix-graph indexing, and scratch teardown.

## Source semantics
* `qsa_states` has one entry per QSA layer, `gdn_state` one per GDN layer.
* The token path replays `pre[l]` and `post[l]` per layer: 96 graph launches, `session_replay` only 48.
* `dump_layers` holds `(n_layers + 1) * hc * n_embd` floats: slot 0 is the entry residual, slot `l+1` is after layer `l`.
* `preP[k-1]` (k = 1..5) are the five prefix graphs.
* `SessionLoopScratch::free` is safe to call twice.

## Form template identifiers
SessionState -> sequence_state; SessionGraphs -> layer_graph_set; session_loop -> host_layer_loop;
SessionLoopScratch -> loop_scratch; TokenGraph -> whole_token_graph; dump_layers -> residual_ladder.
-/

namespace SessionTokenLoop

-- § per-kind state indices

/-- Whether layer `l` is QSA. -/
def isQsa (l : Nat) : Prop := l % 4 = 3

/-- Index of a GDN layer among the 36 GDN layers (QSA layers below it are `l / 4`). -/
def gdnIdx (l : Nat) : Nat := l - l / 4

/-- Index of a QSA layer among the 12 QSA layers. -/
def qsaIdx (l : Nat) : Nat := l / 4

/-- GDN state indices are within the 36 states. -/
theorem gdnIdx_lt {l : Nat} (hl : l < 48) (hg : l % 4 ≠ 3) : gdnIdx l < 36 := by
  unfold gdnIdx
  omega

/-- QSA state indices are within the 12 states. -/
theorem qsaIdx_lt {l : Nat} (hl : l < 48) : qsaIdx l < 12 := by
  unfold qsaIdx
  omega

/-- Distinct GDN layers get distinct state slots (no two layers share a recurrence). -/
theorem gdnIdx_injective {l1 l2 : Nat} (h1 : l1 % 4 ≠ 3) (h2 : l2 % 4 ≠ 3)
    (h : gdnIdx l1 = gdnIdx l2) : l1 = l2 := by
  unfold gdnIdx at h
  omega

/-- Distinct QSA layers get distinct KV states. -/
theorem qsaIdx_injective {l1 l2 : Nat} (h1 : l1 % 4 = 3) (h2 : l2 % 4 = 3)
    (h : qsaIdx l1 = qsaIdx l2) : l1 = l2 := by
  unfold qsaIdx at h
  omega

-- § graph launches

/-- Graph launches per token: `pre` and `post` for each of `n` layers. -/
def launchesFull (n : Nat) : Nat := 2 * n

/-- `session_replay` launches only the `pre` graphs. -/
def launchesPreOnly (n : Nat) : Nat := n

/-- The full token is 96 launches for 48 layers. -/
theorem full_is_96 : launchesFull 48 = 96 := by decide

/-- `session_replay` under-counts the GPU work: it omits all `post` graphs. -/
theorem replay_omits_post {n : Nat} (h : 0 < n) : launchesPreOnly n < launchesFull n := by
  unfold launchesPreOnly launchesFull
  omega

/-- `preP[k-1]` for `k` in 1..5 is within the five-entry prefix array. -/
theorem prefix_index_lt {k : Nat} (h1 : 1 ≤ k) (h2 : k ≤ 5) : k - 1 < 5 := by omega

-- § dump_layers layout

/-- Start offset of snapshot slot `s` in the dump buffer. -/
def slotOff (stride s : Nat) : Nat := s * stride

/-- Slot `l + 1` (after layer `l`) ends inside the `(n_layers + 1) * stride` buffer. -/
theorem dump_slot_in_bounds {stride n l : Nat} (hl : l < n) :
    slotOff stride (l + 1) + stride ≤ (n + 1) * stride := by
  unfold slotOff
  have h1 : (l + 1 + 1) * stride ≤ (n + 1) * stride := Nat.mul_le_mul_right _ (by omega)
  have e : (l + 1 + 1) * stride = (l + 1) * stride + stride := Nat.succ_mul _ _
  rw [e] at h1
  exact h1

/-- Slots do not overlap: slot `s + 1` starts where slot `s` ends. -/
theorem dump_slots_abut (stride s : Nat) : slotOff stride s + stride = slotOff stride (s + 1) := by
  unfold slotOff
  rw [Nat.add_mul, Nat.one_mul]

-- § scratch teardown

/-- Whether the scratch currently owns resources (the host thread is pinned). -/
structure Scratch where
  pinned : Bool
  allocated : Bool

/-- `free` releases everything. -/
def Scratch.free (_ : Scratch) : Scratch := { pinned := false, allocated := false }

/-- Calling `free` twice equals calling it once. -/
theorem free_idempotent (s : Scratch) : s.free.free = s.free := rfl

end SessionTokenLoop



## FILE: /mnt/user-data/outputs/SpeculativeVerifyWindow.lean
/-!
# MODULE: speculative_verify_window
SOURCES: include/ref/core/verify.hpp
The speculative verify window: T tokens (the last accepted one plus T-1 drafts) go through all 48 layers in
one graph; token t's argmax equals what greedy decode would emit, so a draft is accepted exactly when it
matches. `commit(n_keep)` keeps the first `n_keep` tokens.

## Source semantics
* `out[t]` is the argmax after token `t`; draft `t+1` is accepted iff `out[t] = draft[t+1]`, and acceptance stops
  at the first mismatch.
* `n_keep = 1 + accepted` is in `1..T`.
* `max_t <= kVerifyMaxT` and the per-`T` graph array has 9 entries.
* Flag slots are indexed by (layer, group): `2 * 64 * 2` entries.
* The commit record is `[n_keep, n_keep-1, pos_0 .. pos_{T-1}]`.

## Form template identifiers
Verifier -> window_verifier; run -> verify_window; commit -> keep_prefix; VerifyHits -> resident_view;
PoolMultiFn -> multi_token_pool; GpuPlanSink -> gpu_share_plan.
-/

namespace SpeculativeVerifyWindow

-- § acceptance

/-- Number of drafts accepted: the longest prefix of `drafts` equal to the window's argmaxes. -/
def accepted : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then 1 + accepted ds os else 0
  | _, _ => 0

/-- Acceptance never exceeds the number of drafts. -/
theorem accepted_le_drafts : ∀ (ds os : List Nat), accepted ds os ≤ ds.length := by
  intro ds
  induction ds with
  | nil => intro os; simp [accepted]
  | cons d ds ih =>
    intro os
    cases os with
    | nil => simp [accepted]
    | cons o os =>
      by_cases h : d = o
      · have := ih os
        simp [accepted, h]
        omega
      · simp [accepted, h]

/-- Acceptance never exceeds the number of argmaxes available. -/
theorem accepted_le_outs : ∀ (ds os : List Nat), accepted ds os ≤ os.length := by
  intro ds
  induction ds with
  | nil => intro os; simp [accepted]
  | cons d ds ih =>
    intro os
    cases os with
    | nil => simp [accepted]
    | cons o os =>
      by_cases h : d = o
      · have := ih os
        simp [accepted, h]
        omega
      · simp [accepted, h]

/-- If every draft matches its argmax, all drafts are accepted. -/
theorem accepted_all : ∀ ds : List Nat, accepted ds ds = ds.length := by
  intro ds
  induction ds with
  | nil => rfl
  | cons d ds ih => simp [accepted, ih]; omega

/-- A mismatch at the first draft rejects everything after it. -/
theorem accepted_first_mismatch {d o : Nat} (h : d ≠ o) (ds os : List Nat) :
    accepted (d :: ds) (o :: os) = 0 := by
  simp [accepted, h]

/-- `n_keep = 1 + accepted` is between 1 and `T = drafts + 1`. -/
theorem n_keep_range (ds os : List Nat) :
    1 ≤ 1 + accepted ds os ∧ 1 + accepted ds os ≤ ds.length + 1 := by
  have := accepted_le_drafts ds os
  omega

-- § array bounds

/-- `exec_[9]` is indexed by `T <= 8`. -/
theorem exec_index_lt {T : Nat} (h : T ≤ 8) : T < 9 := by omega

/-- Flag slot of (layer, group) lies in `flag_sets_[2 * 64 * 2]` for layers < 64 and two groups. -/
theorem flag_slot_lt {l g : Nat} (hl : l < 64) (hg : g < 2) : l * 2 + g < 2 * 64 * 2 := by omega

/-- The commit record holds `2 + T` words. -/
def commitWords (T : Nat) : Nat := 2 + T

/-- A window of at most 8 tokens has a commit record of at most 10 words. -/
theorem commitWords_le {T : Nat} (h : T ≤ 8) : commitWords T ≤ 10 := by
  unfold commitWords
  omega

/-- Entries of a window (tokens x 10 experts) stay within the plan's capacity of 128. -/
theorem window_entries_le {T : Nat} (h : T ≤ 8) : T * 10 ≤ 128 := by omega

end SpeculativeVerifyWindow





## FILE: /mnt/user-data/outputs/DirectFileIo.lean
/-!
# MODULE: direct_file_io
SOURCES: src/platform/direct_file.cpp
Unbuffered file reads: alignment contract of `submit`, short-read length at end of file, the FIFO completion
drain of `wait`, and the Windows 64-bit offset split.

## Source semantics
* `submit` requires offset, length and buffer address all multiples of the alignment, and `length != 0`.
* A read at or past the end of file completes with zero bytes; otherwise it returns at most the bytes left.
* `wait(out, max)` returns at most `max` completions, oldest first, never dropping or reordering any.
* The Windows OVERLAPPED stores the offset as a low 32-bit word and a high word.

## Form template identifiers
aligned ↦ `x % alignment() == 0`; readLen ↦ `pread` result / `ERROR_HANDLE_EOF` zero completion;
drain ↦ `wait` loop over `done`/`immediate`.
-/
namespace Ref.DirectFileIo

-- § alignment contract

/-- `A` divides `x` (the `% alignment() == 0` test). -/
def aligned (A x : Nat) : Prop := A ∣ x

/-- A request's end offset stays aligned. -/
theorem end_aligned (A off len : Nat) (h1 : aligned A off) (h2 : aligned A len) : aligned A (off + len) :=
  Nat.dvd_add h1 h2

/-- A non-empty aligned request is at least one alignment unit long. -/
theorem len_ge_align (A len : Nat) (hA : 0 < A) (h : aligned A len) (hl : len ≠ 0) : A ≤ len :=
  Nat.le_of_dvd (by omega) h

/-- Aligned lengths cannot be 1..A-1 (no partial sector requests pass). -/
theorem no_partial_sector (A len : Nat) (hA : 0 < A) (h : aligned A len) (hl : len ≠ 0) :
    ¬ len < A := by
  have := len_ge_align A len hA h hl
  omega

-- § end of file

/-- Bytes a read of `len` at `off` returns for a file of `size`. -/
def readLen (size off len : Nat) : Nat := if off < size then min len (size - off) else 0

/-- A read never returns more than requested. -/
theorem readLen_le_len (size off len : Nat) : readLen size off len ≤ len := by
  unfold readLen; split <;> omega

/-- A read never runs past the end of the file. -/
theorem readLen_in_file (size off len : Nat) (h : off < size) : off + readLen size off len ≤ size := by
  unfold readLen; rw [if_pos h]; omega

/-- At or past the end the completion is zero bytes (the legal short read of the last page). -/
theorem readLen_past_eof (size off len : Nat) (h : size ≤ off) : readLen size off len = 0 := by
  unfold readLen; rw [if_neg (by omega)]

/-- A request fully inside the file is returned in full. -/
theorem readLen_full (size off len : Nat) (h : off + len ≤ size) (hl : 0 < len) :
    readLen size off len = len := by
  unfold readLen; rw [if_pos (by omega)]; omega

-- § completion drain

/-- `wait`: hand out up to `max` oldest completions, keep the rest queued. -/
def drain {α : Type} (max : Nat) (q : List α) : List α × List α := (q.take max, q.drop max)

/-- Nothing is lost or reordered. -/
theorem drain_preserves {α : Type} (max : Nat) (q : List α) :
    (drain max q).1 ++ (drain max q).2 = q := by
  unfold drain; exact List.take_append_drop max q

/-- The number handed out is `min max (length q)`. -/
theorem drain_count {α : Type} (max : Nat) (q : List α) :
    (drain max q).1.length = min max q.length := by
  unfold drain; simp

/-- The caller's buffer of size `max` is never overrun. -/
theorem drain_le_max {α : Type} (max : Nat) (q : List α) : (drain max q).1.length ≤ max := by
  rw [drain_count]; omega

/-- Windows: entries requested from the port are at most 64 and fit the remaining room. -/
theorem port_request_fits (max n : Nat) (h : n ≤ max) :
    n + min (max - n) 64 ≤ max ∧ min (max - n) 64 ≤ 64 := by omega

-- § 64-bit offset split

/-- `Offset`/`OffsetHigh` recombine to the original offset. -/
theorem offset_split (off : Nat) : off / 4294967296 * 4294967296 + off % 4294967296 = off := by omega

/-- The low word fits 32 bits. -/
theorem offset_low_lt (off : Nat) : off % 4294967296 < 4294967296 := Nat.mod_lt _ (by omega)

end Ref.DirectFileIo



## FILE: /mnt/user-data/outputs/LayerComposition.lean
/-!
# MODULE: layer_composition
SOURCES: src/core/layer.cpp
The per-token GDN / QSA / MoE layer: arena carving, canonical plane layout, activation-format dispatch,
RoPE-row bounds, the verify-compatibility predicate and the oracle dump stride.

## Source semantics
* Every buffer region is 16-byte aligned and carved sequentially from one arena (`Cursor`, `*_buffers_init`).
* A quantized tensor is codes ++ scales ++ optional offsets; the three sizes must add up to the tensor size.
* Activation format is a property of the TENSOR: code_bits = 2 → Q8_0; otherwise Q8_K only if `wants_q8k`.
* `qsa_layer` refuses positions outside the RoPE table; pooled-block capacity is `max_cells / idx_block + 2`.

## Form template identifiers
align16 ↦ `align_up16`; carve ↦ `Cursor::take`; WeightRef ↦ `WeightRef`; actFor ↦ `gemv_quantized` dispatch;
Flags ↦ the `g_*` toggles consulted by `layer_verify_compatible`.
-/
namespace Ref.LayerComposition

-- § alignment and cursor carving

/-- Round up to a multiple of 16 (`(n + 15) & ~15`). -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks a size. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- Alignment result is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Alignment is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

/-- Total bytes consumed by carving `parts` sequentially from offset `s`. -/
def carve : List Nat → Nat → Nat
  | [], s => s
  | p :: ps, s => carve ps (s + align16 p)

/-- Carving equals start plus the sum of aligned parts (the `*_bytes` and `*_init` totals agree). -/
theorem carve_eq : ∀ (parts : List Nat) (s : Nat), carve parts s = s + (parts.map align16).sum
  | [], s => by simp [carve]
  | p :: ps, s => by
      rw [carve, carve_eq ps]
      simp only [List.map_cons, List.sum_cons]
      omega

/-- Starting aligned, every carve boundary stays aligned. -/
theorem carve_aligned : ∀ (parts : List Nat) (s : Nat), s % 16 = 0 → carve parts s % 16 = 0
  | [], s, h => by simpa [carve] using h
  | p :: ps, s, h => by
      rw [carve]
      apply carve_aligned ps
      have := align16_mod p
      omega

/-- Carving is monotone in the cursor. -/
theorem carve_ge : ∀ (parts : List Nat) (s : Nat), s ≤ carve parts s
  | [], s => Nat.le_refl s
  | p :: ps, s => by
      rw [carve]
      have h := carve_ge ps (s + align16 p)
      omega

/-- Carving splits over concatenation. -/
theorem carve_append : ∀ (xs ys : List Nat) (s : Nat), carve (xs ++ ys) s = carve ys (carve xs s)
  | [], ys, s => by simp [carve]
  | x :: xs, ys, s => by
      simp only [List.cons_append, carve]
      exact carve_append xs ys _

/-- Each region ends inside the arena total: no region overlaps its successors. -/
theorem row_fits (pre post : List Nat) (x : Nat) :
    carve pre 0 + align16 x ≤ carve (pre ++ x :: post) 0 := by
  rw [carve_append]
  have h := carve_ge post (carve pre 0 + align16 x)
  simp only [carve]
  exact h

-- § Q8_K block arithmetic

/-- `q8k_bytes n = n / 256 * 292`. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- A whole number of 256-element blocks costs 292 bytes each. -/
theorem q8kBytes_mul (b : Nat) : q8kBytes (256 * b) = b * 292 := by
  unfold q8kBytes
  have h : 256 * b / 256 = b := by omega
  rw [h]

/-- The width of one embedding (2560) is exactly ten blocks. -/
theorem q8kBytes_embd : q8kBytes 2560 = 2920 := by decide

-- § canonical planes

/-- The loader-recorded attributes of a quantized tensor that `plane_ptrs` inspects. -/
structure WeightRef where
  codeBits : Nat
  codesBytes : Nat
  scalesBytes : Nat
  offsetBytes : Nat
  bytes : Nat
  hasOffset : Bool

/-- The acceptance test of `plane_ptrs`. -/
def planesValid (r : WeightRef) : Prop :=
  (r.codeBits = 2 ∨ r.codeBits = 4 ∨ r.codeBits = 8) ∧ r.codesBytes ≠ 0 ∧ r.scalesBytes ≠ 0 ∧
  r.codesBytes + r.scalesBytes + r.offsetBytes = r.bytes ∧ (r.hasOffset = true ↔ r.offsetBytes ≠ 0)

/-- Start of the scale plane. -/
def scalesOff (r : WeightRef) : Nat := r.codesBytes

/-- Start of the offset plane. -/
def offsetOff (r : WeightRef) : Nat := r.codesBytes + r.scalesBytes

/-- Valid planes tile the tensor exactly: the offset plane ends at the tensor end. -/
theorem planes_tile (r : WeightRef) (h : planesValid r) :
    scalesOff r = r.codesBytes ∧ offsetOff r + r.offsetBytes = r.bytes := by
  obtain ⟨_, _, _, hsum, _⟩ := h
  exact ⟨rfl, hsum⟩

/-- No offset flag means an empty offset plane (so the pointer may be null). -/
theorem no_offset_empty (r : WeightRef) (h : planesValid r) (hh : r.hasOffset = false) :
    r.offsetBytes = 0 := by
  obtain ⟨_, _, _, _, hoff⟩ := h
  by_contra hne
  have := hoff.2 hne
  simp [hh] at this

/-- An fp16 plane widened to f32 doubles; with an even f32 size the source size is exact. -/
theorem fp16_src_double (s : Nat) (h : s % 4 = 0) : s / 2 * 2 = s := by omega

/-- The old re-derivation assumed 4 B/scale; for fp16 packs it was twice the real plane. -/
theorem old_derivation_doubles (elems actual derived : Nat)
    (ha : actual = 2 * elems) (hd : derived = 4 * elems) : derived = 2 * actual := by omega

-- § activation contract

/-- The two quantized activation images. -/
inductive Act | q8_0 | q8k
  deriving DecidableEq

/-- `gemv_quantized`: 2-bit forms use Q8_0; others Q8_K only when the manifest source type wants it. -/
def actFor (codeBits : Nat) (wantsQ8k : Bool) : Act :=
  if codeBits = 2 then Act.q8_0 else if wantsQ8k then Act.q8k else Act.q8_0

/-- Two-bit tensors never see Q8_K. -/
theorem actFor_two (w : Bool) : actFor 2 w = Act.q8_0 := by simp [actFor]

/-- A legacy form (not wanting Q8_K) always gets Q8_0. -/
theorem actFor_legacy (b : Nat) : actFor b false = Act.q8_0 := by
  unfold actFor; split <;> rfl

/-- Q8_K is only chosen for non-2-bit tensors that ask for it. -/
theorem actFor_q8k (b : Nat) (w : Bool) (h : actFor b w = Act.q8k) : b ≠ 2 ∧ w = true := by
  unfold actFor at h
  by_cases hb : b = 2
  · simp [hb] at h
  · refine ⟨hb, ?_⟩
    cases w
    · simp [hb] at h
    · rfl

-- § GDN q|k|v split of the conv channels

/-- v starts after q and k, and the three runs fill the channel dimension. -/
theorem qkv_split (qk v C : Nat) (hC : C = 2 * qk + v) : 2 * qk + v = C := hC.symm

/-- The real geometry: 128 * 16 = 2048, 48 * 128 = 6144, total 10240. -/
theorem qkv_real : 2 * (128 * 16) + 48 * 128 = 10240 := by decide

-- § QSA positions

/-- `qsa_layer` accepts only a non-negative base with `pos_base + pos < max_cells`. -/
def ropeRowOk (posBase pos maxCells : Int) : Prop := 0 ≤ posBase ∧ posBase + pos < maxCells

/-- An accepted position reads a RoPE row that exists. -/
theorem ropeRowOk_row (posBase pos maxCells : Int) (h : ropeRowOk posBase pos maxCells) (hp : 0 ≤ pos) :
    0 ≤ posBase + pos ∧ posBase + pos < maxCells := by
  obtain ⟨h1, h2⟩ := h
  exact ⟨by omega, h2⟩

/-- Completed-block count never exceeds the pooled capacity `max_cells / idx_block + 2`. -/
theorem blocks_within_capacity (pos maxCells blk : Nat) (h : pos < maxCells) :
    (pos + 1) / blk ≤ maxCells / blk + 2 := by
  have h1 : (pos + 1) / blk ≤ maxCells / blk := Nat.div_le_div_right (by omega)
  omega

/-- `pages * page_size` covers `max_cells`. -/
theorem pages_cover (m ps : Nat) (hps : 0 < ps) : m ≤ (m + ps - 1) / ps * ps := by
  have h := Nat.div_add_mod (m + ps - 1) ps
  have h2 := Nat.mod_lt (m + ps - 1) hps
  rw [Nat.mul_comm]
  omega

-- § oracle dump layout

/-- Floats per layer in the C1 dump (`nvk = n_head_kv * head_dim`, `ah = n_head * head_dim`). -/
def dumpStride (n hc ah nvk : Nat) : Nat := 2 * n + 2 * hc + ah + 5 * nvk + 8

/-- The last slot (`kcur`) ends exactly at the stride: nothing is written out of the layer's row. -/
theorem dump_last_slot_end (n hc ah nvk : Nat) :
    (2 * n + 2 * hc + ah + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc ah nvk := by
  unfold dumpStride; omega

-- § verify compatibility

/-- The toggles `layer_verify_compatible` consults. -/
structure Flags where
  nativeBf16 : Bool
  fusedGr : Bool
  fusedGdn : Bool
  nativeGdn : Bool
  fastAttn : Bool
  nativeFlashShort : Bool
  fastSelect : Bool
  nativeIndexer : Bool

/-- The window reproduces the default native decode path only if every toggle is in its default state. -/
def verifyCompatible (f : Flags) : Bool :=
  f.nativeBf16 && f.fusedGr && (f.fusedGdn && f.nativeGdn) && (f.fastAttn && !f.nativeFlashShort) &&
    f.fastSelect && f.nativeIndexer

/-- Compatibility forces the fused hyper-connection read on. -/
theorem compat_fusedGr (f : Flags) (h : verifyCompatible f = true) : f.fusedGr = true := by
  cases f; simp_all [verifyCompatible]

/-- Compatibility forces the short flash adapter off. -/
theorem compat_no_flash_short (f : Flags) (h : verifyCompatible f = true) :
    f.nativeFlashShort = false := by
  cases f; simp_all [verifyCompatible]

/-- Compatibility forces the native indexer on. -/
theorem compat_indexer (f : Flags) (h : verifyCompatible f = true) : f.nativeIndexer = true := by
  cases f; simp_all [verifyCompatible]

end Ref.LayerComposition



## FILE: /mnt/user-data/outputs/MtpDrafter.lean
/-!
# MODULE: mtp_drafter
SOURCES: src/core/mtp.cpp
The multi-token-prediction draft layer: bump-allocated device arena, windowed attention capacity, the
step/position row staging used by captured prefill / round / chain-step graphs, and the draft chain length.

## Source semantics
* `Bump::take` pads every region to 256 bytes; the counting pass and the real pass use the same sequence.
* Step/pos staging has `2*max_t` rows: catch-up rows `[0,T)`, the full-layer row `2*max_t-1`, chain rows
  `max_t + j - 1` for `j = 1..max_t-2`.
* `record_forward(T, s)`: `s >= 0` is the full layer on row `s`; `s < 0` is K/V-only on rows from `-1 - s`.
* The draft chain emits at most `max_t - 1` drafts.

## Form template identifiers
align256 ↦ `(n*sizeof(T)+255)&~255`; capRound ↦ `cap_`; ra ↦ `2*max_t_-1`; chainRow ↦ `max_t_+j-1`.
-/
namespace Ref.MtpDrafter

-- § bump allocator

/-- 256-byte padding of one region. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Padding never shrinks. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- Padded regions are 256-multiples. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- Padding wastes fewer than 256 bytes. -/
theorem align256_lt (n : Nat) : align256 n < n + 256 := by unfold align256; omega

/-- Total after carving a list of byte sizes. -/
def bump : List Nat → Nat → Nat
  | [], s => s
  | p :: ps, s => bump ps (s + align256 p)

/-- Starting aligned, every region base is 256-aligned (device pointers stay aligned). -/
theorem bump_aligned : ∀ (parts : List Nat) (s : Nat), s % 256 = 0 → bump parts s % 256 = 0
  | [], s, h => by simpa [bump] using h
  | p :: ps, s, h => by
      rw [bump]
      apply bump_aligned ps
      have := align256_mod p
      omega

/-- Bumping is monotone. -/
theorem bump_ge : ∀ (parts : List Nat) (s : Nat), s ≤ bump parts s
  | [], s => Nat.le_refl s
  | p :: ps, s => by
      rw [bump]
      have h := bump_ge ps (s + align256 p)
      omega

-- § windowed capacity

/-- `cap_ = ((window or max_cells) + 63) / 64 * 64`. -/
def capRound (w : Nat) : Nat := (w + 63) / 64 * 64

/-- The capacity covers the window. -/
theorem capRound_ge (w : Nat) : w ≤ capRound w := by unfold capRound; omega

/-- The capacity is a multiple of 64. -/
theorem capRound_mod (w : Nat) : capRound w % 64 = 0 := by unfold capRound; omega

/-- The capacity wastes fewer than 64 cells. -/
theorem capRound_lt (w : Nat) : capRound w < w + 64 := by unfold capRound; omega

-- § step rows

/-- The full-layer row of a round. -/
def ra (maxT : Nat) : Nat := 2 * maxT - 1

/-- The chain-step row for draft `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- The full-layer row is inside the `2*max_t` rows allocated. -/
theorem ra_lt (maxT : Nat) (h : 1 ≤ maxT) : ra maxT < 2 * maxT := by unfold ra; omega

/-- Chain rows lie strictly between the catch-up rows and the full-layer row. -/
theorem chainRow_between (maxT j : Nat) (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) :
    maxT ≤ chainRow maxT j ∧ chainRow maxT j < ra maxT := by
  unfold chainRow ra; omega

/-- Catch-up rows `[0,T)` never collide with a chain row. -/
theorem catchup_disjoint (maxT T j : Nat) (hT : T ≤ maxT) (h1 : 1 ≤ j) :
    ∀ t, t < T → t ≠ chainRow maxT j := by
  intro t ht; unfold chainRow; omega

/-- Writing four step ints for any row below `R2` stays inside `R2*4*4` bytes. -/
theorem step_row_bytes (row R2 : Nat) (h : row < R2) : (row * 4 + 3) * 4 + 4 ≤ R2 * 4 * 4 := by omega

/-- K/V-only staging: the first row of `step_row0 < 0` is non-negative. -/
theorem kv_only_row0 (s : Int) (h : s < 0) : 0 ≤ -1 - s := by omega

/-- `step_row0 = -1` maps to row 0 (the prefill graph reads rows `[0,T)`). -/
theorem kv_only_row0_prefill : (-1 : Int) - (-1) = 0 := by decide

-- § draft chain

/-- Number of drafts after `steps` accepted chain steps: the first plus one per step. -/
def draftCount (steps : Nat) : Nat := 1 + steps

/-- With chain index `j < max_t - 1` and `j >= 1` at most `max_t - 2` steps run, so at most `max_t - 1` drafts. -/
theorem draftCount_le (maxT steps : Nat) (h : steps + 2 ≤ maxT) : draftCount steps ≤ maxT - 1 := by
  unfold draftCount; omega

/-- The zero-fill range `[n, max_t-1)` is disjoint from emitted drafts `[0,n)` and fits the output array. -/
theorem fill_range (n maxT j : Nat) (h : n ≤ j) (hj : j < maxT - 1) : n ≤ j ∧ j < maxT := by omega

-- § prefill skipping

/-- A chunk entirely at or below `first_needed` is skipped; every cell it would write is below that mark. -/
theorem skipped_chunk_cells (cell0 c T firstNeeded : Int) (h : cell0 + c + T ≤ firstNeeded) :
    ∀ t : Int, 0 ≤ t → t < T → cell0 + c + t < firstNeeded := by
  intro t _ ht; omega

/-- A skipped cell is more than `window + 64` before the prompt end. -/
theorem skipped_cell_outside_window (cell promptLen window firstNeeded : Int)
    (hf : firstNeeded = promptLen - window - 64) (h : cell < firstNeeded) :
    cell + window + 64 < promptLen := by omega

/-- Third step field `(cell+1)/4` never exceeds the second. -/
theorem step_pool_field (cell : Nat) : (cell + 1) / 4 ≤ cell + 1 := Nat.div_le_self _ _

end Ref.MtpDrafter



## FILE: /mnt/user-data/outputs/PinnedArenaLoad.lean
/-!
# MODULE: pinned_arena_load
SOURCES: src/core/pinned.cu
The pinned host arena: uniform slice bounds for sliced cudaHostRegister, the pinned/locked split, the parallel
per-layer expert load ranges, and the bandwidth probe's byte accounting.

## Source semantics
* `uniform_bounds(bytes, slice)` is `[0, slice, ..., k*slice]` with `k = bytes/slice`, empty if `k = 0`.
* When whole-arena registration fails, slices are pinned from the start; the rest is locked resident, so
  `registered + locked = capacity`, and the destructor unlocks from `registered`.
* Layer `L` of the expert file spans `[L*per, L*per+per)` with `per = blobs_per_layer * blob_bytes`.
* The read loop consumes `min(chunk, remaining)` bytes per step.

## Form template identifiers
bnd ↦ `bounds[i]`; registered ↦ `registered_bytes`; per ↦ `blobs_per_layer*blob_bytes`; moved ↦ `bytes/chunk*chunk`.
-/
namespace Ref.PinnedArenaLoad

-- § uniform slice bounds

/-- Bound `i` of the uniform slicing. -/
def bnd (slice i : Nat) : Nat := i * slice

/-- Consecutive bounds differ by exactly one slice. -/
theorem bnd_step (slice i : Nat) : bnd slice (i + 1) = bnd slice i + slice := by
  unfold bnd; rw [Nat.add_mul, Nat.one_mul]

/-- The last bound `k*slice` never exceeds the arena. -/
theorem bnd_last_le (bytes slice : Nat) : bnd slice (bytes / slice) ≤ bytes := by
  unfold bnd; exact Nat.div_mul_le_self bytes slice

/-- Every pinned slice `i < bytes/slice` ends inside the arena. -/
theorem slice_in_arena (bytes slice i : Nat) (h : i < bytes / slice) : bnd slice i + slice ≤ bytes := by
  have h1 : bnd slice (i + 1) ≤ bnd slice (bytes / slice) := by
    unfold bnd; exact Nat.mul_le_mul_right slice (by omega)
  have h2 := bnd_last_le bytes slice
  have h3 := bnd_step slice i
  omega

/-- A partition into whole slices leaves no tail exactly when `slice ∣ bytes`. -/
theorem no_tail_iff (bytes slice : Nat) (h : bytes % slice = 0) : bnd slice (bytes / slice) = bytes := by
  unfold bnd; exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)

-- § pinned / locked split

/-- Pinned prefix plus locked suffix covers the capacity. -/
theorem pinned_locked_cover (capacity registered locked : Nat) (hr : registered ≤ capacity)
    (hl : locked = capacity - registered) : registered + locked = capacity := by omega

/-- The destructor's unlock start equals the lock start (`base + registered`). -/
theorem unlock_matches_lock (registered : Nat) : registered + 0 = registered := rfl

/-- With `c` slices pinned the registered prefix is `c*slice`, below capacity when `c <= bytes/slice`. -/
theorem registered_le (bytes slice c : Nat) (h : c ≤ bytes / slice) : bnd slice c ≤ bytes := by
  have h1 : bnd slice c ≤ bnd slice (bytes / slice) := by
    unfold bnd; exact Nat.mul_le_mul_right slice h
  have h2 := bnd_last_le bytes slice
  omega

-- § per-layer expert ranges

/-- Offset of layer `L`. -/
def layerOff (per L : Nat) : Nat := L * per

/-- Layer ranges abut: layer `L+1` starts where `L` ends. -/
theorem layer_abut (per L : Nat) : layerOff per (L + 1) = layerOff per L + per := by
  unfold layerOff; rw [Nat.add_mul, Nat.one_mul]

/-- Every layer's range ends inside `layers * per`. -/
theorem layer_in_file (per L layers : Nat) (h : L < layers) : layerOff per L + per ≤ layers * per := by
  unfold layerOff
  have := Nat.mul_le_mul_right per (show L + 1 ≤ layers by omega)
  simpa [Nat.add_mul] using this

/-- Distinct layers' ranges do not overlap, so workers never write the same bytes. -/
theorem layers_disjoint (per a b : Nat) (h : a < b) : layerOff per a + per ≤ layerOff per b := by
  unfold layerOff
  have := Nat.mul_le_mul_right per (show a + 1 ≤ b by omega)
  simpa [Nat.add_mul] using this

-- § read loop

/-- The step the loop takes. -/
def step (chunk remaining : Nat) : Nat := min chunk remaining

/-- Each step makes progress and never over-reads the layer. -/
theorem step_progress (chunk rem : Nat) (hc : 0 < chunk) (hr : 0 < rem) :
    0 < step chunk rem ∧ step chunk rem ≤ rem ∧ rem - step chunk rem < rem := by
  unfold step; omega

/-- Writing at `off + pos` with `pos + step <= layer_bytes` stays inside the layer. -/
theorem write_in_layer (off pos layerBytes chunk : Nat) (h : pos ≤ layerBytes) :
    off + pos + step chunk (layerBytes - pos) ≤ off + layerBytes := by
  unfold step; omega

-- § bandwidth probe

/-- Bytes the probe actually moves per iteration. -/
def moved (bytes chunk : Nat) : Nat := bytes / chunk * chunk

/-- The probe never moves more than `bytes`. -/
theorem moved_le (bytes chunk : Nat) : moved bytes chunk ≤ bytes := Nat.div_mul_le_self bytes chunk

/-- The reported `bytes * iters` equals what moved only when the chunk divides the size. -/
theorem moved_eq_iff (bytes chunk : Nat) (h : bytes % chunk = 0) : moved bytes chunk = bytes :=
  Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)

end Ref.PinnedArenaLoad



## FILE: /mnt/user-data/outputs/PrefillChunkPipeline.lean
/-!
# MODULE: prefill_chunk_pipeline
SOURCES: src/prefill/prefill.cpp
Chunked prefill: arena sizing by a counting pass, the per-chunk loop, host-side grouping of (token, k) pairs
by expert, the expert staging ring and the dequantized-expert ring.

## Source semantics
* `Alloc::take` pads each region to `bytes + 256` rounded up to 256; the count-only pass sizes the borrow.
* Chunks cover `[0, n)` in steps of `T`; each chunk has `min(T, n - c0) >= 1` tokens.
* Experts are grouped: `off[e+1] = off[e] + cnt[e]`, slot of the j-th pair of expert `e` is `off[e] + j`,
  source token of pair `i` is `i / K`.
* Staging ring has 8 slots with lookahead 7; dequant ring has 2 slots.

## Form template identifiers
takeBytes ↦ `Alloc::take` rounding; offs ↦ `m.off`; stage slot ↦ `stage_next`; q ↦ `j % DQ`.
-/
namespace Ref.PrefillChunkPipeline

-- § allocator

/-- Padded size of one region of `bytes` payload. -/
def takeBytes (bytes : Nat) : Nat := (bytes + 256 + 255) / 256 * 256

/-- Each region carries at least its payload plus 256 slack. -/
theorem takeBytes_ge (b : Nat) : b + 256 ≤ takeBytes b := by unfold takeBytes; omega

/-- Regions are 256-multiples. -/
theorem takeBytes_mod (b : Nat) : takeBytes b % 256 = 0 := by unfold takeBytes; omega

/-- Total used after a list of regions. -/
def used : List Nat → Nat → Nat
  | [], s => s
  | b :: bs, s => used bs (s + takeBytes b)

/-- Usage is monotone. -/
theorem used_ge : ∀ (bs : List Nat) (s : Nat), s ≤ used bs s
  | [], s => Nat.le_refl s
  | b :: bs, s => by
      rw [used]
      have h := used_ge bs (s + takeBytes b)
      omega

/-- Usage splits over concatenation. -/
theorem used_append : ∀ (xs ys : List Nat) (s : Nat), used (xs ++ ys) s = used ys (used xs s)
  | [], ys, s => by simp [used]
  | x :: xs, ys, s => by
      simp only [List.cons_append, used]
      exact used_append xs ys _

/-- If the counted total fits the borrowed capacity, every region fits when taken (no `ok = false`). -/
theorem take_fits (pre post : List Nat) (x cap : Nat) (h : used (pre ++ x :: post) 0 ≤ cap) :
    used pre 0 + takeBytes x ≤ cap := by
  rw [used_append] at h
  have h2 := used_ge post (used pre 0 + takeBytes x)
  simp only [used] at h
  omega

-- § chunk loop

/-- Tokens in the chunk starting at `c0`. -/
def chunkTokens (T n c0 : Nat) : Nat := min T (n - c0)

/-- Every chunk is non-empty and stays inside `[0, n)`. -/
theorem chunk_bounds (T n c0 : Nat) (hT : 0 < T) (h : c0 < n) :
    0 < chunkTokens T n c0 ∧ c0 + chunkTokens T n c0 ≤ n := by
  unfold chunkTokens; omega

/-- Every pair's source token `i / K` is a valid token of the chunk. -/
theorem src_token_lt (i T K : Nat) (hK : 0 < K) (h : i < T * K) : i / K < T := by
  rw [Nat.div_lt_iff_lt_mul hK]
  exact h

-- § grouping by expert

/-- Prefix offsets of the per-expert counts. -/
def offs (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offs cnt e + cnt e

/-- Prefix offsets are monotone. -/
theorem offs_mono (cnt : Nat → Nat) {a b : Nat} (h : a ≤ b) : offs cnt a ≤ offs cnt b := by
  induction b with
  | zero =>
      have : a = 0 := by omega
      subst this
      exact Nat.le_refl _
  | succ b ih =>
      rcases Nat.lt_or_ge a (b + 1) with h1 | h1
      · have := ih (by omega)
        simp only [offs]
        omega
      · have : a = b + 1 := by omega
        subst this
        exact Nat.le_refl _

/-- Expert `e`'s slot range ends before any later expert's begins: slot ranges are disjoint. -/
theorem slots_disjoint (cnt : Nat → Nat) (e e' : Nat) (h : e < e') : offs cnt e + cnt e ≤ offs cnt e' := by
  have := offs_mono cnt (show e + 1 ≤ e' by omega)
  simpa [offs] using this

/-- The j-th pair of expert `e` gets a slot inside its own range. -/
theorem slot_in_range (cnt : Nat → Nat) (e j : Nat) (h : j < cnt e) :
    offs cnt e ≤ offs cnt e + j ∧ offs cnt e + j < offs cnt (e + 1) := by
  simp only [offs]
  omega

-- § staging ring

/-- Staging slot reuse is safe: blob `i` (staged while computing `j`, so `i ≤ j + 7`) reuses the slot of
blob `i - 8`, which is strictly earlier than `j` and therefore already dequantized. -/
theorem stage_slot_reuse_safe (i j : Nat) (h : i ≤ j + 7) (h8 : 8 ≤ i) :
    i - 8 < j + 0 ∧ (i - 8) % 8 = i % 8 := by omega

/-- The slot index is always a valid ring slot. -/
theorem stage_slot_lt (n : Nat) : n % 8 < 8 := Nat.mod_lt _ (by omega)

/-- The dequant ring alternates between two buffers: consecutive experts use different ones. -/
theorem dq_alternates (j : Nat) : (j + 1) % 2 ≠ j % 2 := by omega

/-- Lookahead never stages beyond the order list: `staged ≤ j + 7` while `staged < len`. -/
theorem lookahead_bounded (staged j len : Nat) (h1 : staged < len) (h2 : staged ≤ j + 7) :
    staged < len ∧ staged - j ≤ 7 := by omega

end Ref.PrefillChunkPipeline



## FILE: /mnt/user-data/outputs/PrefillKernelsIndexing.lean
/-!
# MODULE: prefill_kernels_indexing
SOURCES: src/prefill/kernels.cu
CUDA prefill kernels: BF16 round-to-nearest-even, router tie-break order, residual-stream index decomposition,
2-bit expert blob layout and dequant indexing, KV append lane mapping, QSA q/gate split and launch sizing.

## Source semantics
* `bf(f)`: add `0x7fff + ((u>>16)&1)` and keep the high 16 bits (round to nearest, ties to even).
* Router: candidate `(v, i)` beats `(v', i')` iff `v > v'` or `v = v'` and `i < i'`; expert id is `lane + 32*i`.
* Residual `R[i]` with `N = 2560`, `HC = 4`, `D = N*HC`: `t = i/D`, `c = (i%D)/N`, `d = i%N`.
* Blob: gate/up codes `[1280][640 B]`, down codes `[2560][160 B]`, gate/up scales `[1280][40] f16`, down scales
  `[2560][10] f16`; each code byte holds four 2-bit weights, one scale per 64 weights.

## Form template identifiers
bf ↦ `bf`; better ↦ router `(ob > best || (ob == best && oi < ex))`; blocksFor ↦ `blocks_for`.
-/
namespace Ref.PrefillKernelsIndexing

-- § bf16 rounding

/-- BF16 rounding of the 32-bit pattern `u` (no overflow modelling needed below `2^32 - 2^16`). -/
def bf (u : Nat) : Nat := (u + 32767 + (u / 65536) % 2) / 65536

/-- Below the halfway point the result truncates. -/
theorem bf_below (u : Nat) (h : u % 65536 < 32768) : bf u = u / 65536 := by unfold bf; omega

/-- Above the halfway point the result rounds up. -/
theorem bf_above (u : Nat) (h : 32768 < u % 65536) : bf u = u / 65536 + 1 := by unfold bf; omega

/-- Exactly at the halfway point the result is even (ties to even). -/
theorem bf_tie_even (u : Nat) (h : u % 65536 = 32768) : bf u % 2 = 0 := by
  unfold bf
  have hu : u = 65536 * (u / 65536) + 32768 := by omega
  rcases Nat.mod_two_eq_zero_or_one (u / 65536) with h0 | h1
  · have e : (u + 32767 + u / 65536 % 2) / 65536 = u / 65536 := by omega
    rw [e]
    exact h0
  · have e : (u + 32767 + u / 65536 % 2) / 65536 = u / 65536 + 1 := by omega
    rw [e]
    omega

/-- The result is within one unit of truncation. -/
theorem bf_within_one (u : Nat) : u / 65536 ≤ bf u ∧ bf u ≤ u / 65536 + 1 := by unfold bf; omega

-- § router order

/-- Strict preference with ties broken toward the smaller index. -/
def better (v i v' i' : Nat) : Prop := v > v' ∨ (v = v' ∧ i < i')

/-- The order is irreflexive. -/
theorem better_irrefl (v i : Nat) : ¬ better v i v i := by unfold better; omega

/-- The order is asymmetric. -/
theorem better_asymm (v i v' i' : Nat) (h : better v i v' i') : ¬ better v' i' v i := by
  unfold better at *; omega

/-- Distinct candidates are always comparable (a unique winner exists). -/
theorem better_total (v i v' i' : Nat) (h : i ≠ i') : better v i v' i' ∨ better v' i' v i := by
  unfold better; omega

/-- Expert id `lane + 32*i` recovers `(lane, i)`. -/
theorem expert_id_roundtrip (lane i : Nat) (h : lane < 32) :
    (lane + 32 * i) % 32 = lane ∧ (lane + 32 * i) / 32 = i := by omega

/-- The 16 registers of 32 lanes cover exactly ids `0..511`. -/
theorem expert_id_lt (lane i : Nat) (hl : lane < 32) (hi : i < 16) : lane + 32 * i < 512 := by omega

/-- Surjective: every id below 512 is some `(lane, i)`. -/
theorem expert_id_surj (e : Nat) (h : e < 512) : e % 32 < 32 ∧ e / 32 < 16 ∧ e % 32 + 32 * (e / 32) = e := by
  omega

-- § residual stream indices

/-- `i = t*D + c*N + d` with `c < HC` for the real constants. -/
theorem residual_decomp (i : Nat) :
    i = (i / 10240) * 10240 + ((i % 10240) / 2560) * 2560 + i % 2560 ∧ (i % 10240) / 2560 < 4 := by
  omega

/-- Broadcast reads `e[t*N + i % N]`, the same element for every one of the four streams. -/
theorem broadcast_same_d (i : Nat) : i % 2560 = (i % 10240) % 2560 := by omega

-- § launch sizing

/-- `blocks_for n = (n + 255) / 256`. -/
def blocksFor (n : Nat) : Nat := (n + 255) / 256

/-- The grid covers all `n` threads. -/
theorem blocksFor_cover (n : Nat) : n ≤ blocksFor n * 256 := by unfold blocksFor; omega

/-- And launches fewer than one extra block. -/
theorem blocksFor_tight (n : Nat) : blocksFor n * 256 < n + 256 := by unfold blocksFor; omega

-- § blob layout

/-- Gate/up code bytes. -/
theorem blob_off_d_codes : 1280 * 640 = 819200 := by decide

/-- Gate/up scale plane begins after the down codes. -/
theorem blob_off_gu_sc : 1280 * 640 + 2560 * 160 = 1228800 := by decide

/-- Down scale plane begins after the gate/up scales (`[1280][40]` f16). -/
theorem blob_off_d_sc : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 = 1331200 := by decide

/-- The blob ends after the down scales (`[2560][10]` f16). -/
theorem blob_total : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

/-- Gate/up dequant: code byte, scale pair and 4 output weights are all in range. -/
theorem gu_dequant_in_range (row byte : Nat) (hr : row < 1280) (hb : byte < 640) :
    row * 640 + byte < 1280 * 640 ∧
    (row * 40 + byte * 4 / 64) * 2 + 2 ≤ 1280 * 40 * 2 ∧
    row * 2560 + byte * 4 + 3 < 1280 * 2560 := by omega

/-- Down dequant: same, with 160 bytes and 10 groups per row. -/
theorem d_dequant_in_range (row byte : Nat) (hr : row < 2560) (hb : byte < 160) :
    row * 160 + byte < 2560 * 160 ∧
    (row * 10 + byte * 4 / 64) * 2 + 2 ≤ 2560 * 10 * 2 ∧
    row * 640 + byte * 4 + 3 < 2560 * 640 := by omega

/-- Each 2-bit code is below 4, so `code - 1 ∈ {-1,0,1,2}`. -/
theorem code2_lt (c k : Nat) : (c / 2 ^ (2 * k)) % 4 < 4 := Nat.mod_lt _ (by omega)

/-- The thread index space splits at `n_gu` into gate/up and down bytes. -/
theorem thread_split : 1280 * 640 + 2560 * 160 = 1228800 := by decide

-- § KV append and QSA helpers

/-- `blockIdx.z` encodes (group, is_v) bijectively over 0..7. -/
theorem kv_z_decode (z : Nat) (h : z < 8) : z / 2 < 4 ∧ (z / 2) * 2 + z % 2 = z := by omega

/-- Each 64-thread block writes lanes `g*64 + t < 256`. -/
theorem kv_lane_lt (g t : Nat) (hg : g < 4) (ht : t < 64) : g * 64 + t < 256 := by omega

/-- Scale slot `row*4 + g` is inside the 4-per-row scale array. -/
theorem kv_scale_slot (row g : Nat) (hg : g < 4) : row * 4 + g < (row + 1) * 4 := by omega

/-- q half and gate half of a head's 512 floats are disjoint and cover it. -/
theorem qgate_halves (h d : Nat) (hd : d < 256) :
    h * 512 + d < h * 512 + 256 ∧ h * 512 + 256 + d < (h + 1) * 512 := by omega

/-- SwiGLU interleaved reads `gu[r*1280 + 2k]` and `+1` for `k < 640`, inside the row. -/
theorem swiglu_il_in_row (r k : Nat) (hk : k < 640) : r * 1280 + 2 * k + 1 < (r + 1) * 1280 := by omega

/-- GDN recurrence row groups: `rg * 32 + r` covers `0..127` for `rg < 4`, `r < 32`. -/
theorem rec_rows (rg r : Nat) (hg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- RoPE pairs `(p, p+32)` stay in a 64-wide rotary block. -/
theorem rope_pair_in_block (p : Nat) (h : p < 32) : p + 32 < 64 := by omega

/-- `gather_rows16` moves whole `uint4` units when the width is a multiple of 8. -/
theorem gather_units (width : Nat) (h : width % 8 = 0) : width / 8 * 8 = width := by omega

end Ref.PrefillKernelsIndexing



## FILE: /mnt/user-data/outputs/SessionGraphReplay.lean
/-!
# MODULE: session_graph_replay
SOURCES: src/core/session.cpp
One token through all layers: session arena sizing and carving, per-layer GDN state slices, per-layer
captured graphs (pre/post, split prefixes), the doorbell ring protocol and stage-prefix timing.

## Source semantics
* Layer `l` is a QSA layer iff `l % 4 = 3`; GDN layers are numbered in order, each with its own state slice.
* Arena regions are 256-aligned; `session_bytes` rounds up and adds 4096 slack.
* Stage flags: `stage_prefix = k > 0` runs stages `0..k-1` and ignores `half`; otherwise half 0 runs all,
  half 1 the mixer (stages 0-2), half 2 the FFN front and router (stages 3-4).
* Ring numbers are `l+1` for layer `l`; stage times are differences of cumulative prefix times.

## Form template identifiers
isQsa ↦ `is_qsa_layer`; gdnBefore ↦ `gdn_index` loop; runStage ↦ `run0..run4`; sliceOff ↦ `gdn_state + idx*floats`.
-/
namespace Ref.SessionGraphReplay

-- § layer numbering

/-- QSA layers are those with index ≡ 3 (mod 4). -/
def isQsa (l : Nat) : Bool := l % 4 == 3

/-- Number of GDN layers strictly before `l` (the `gdn_index` loop). -/
def gdnBefore : Nat → Nat
  | 0 => 0
  | l + 1 => gdnBefore l + (if l % 4 = 3 then 0 else 1)

/-- Closed form: `l - l/4`. -/
theorem gdnBefore_eq : ∀ l, gdnBefore l = l - l / 4
  | 0 => by simp [gdnBefore]
  | l + 1 => by
      have ih := gdnBefore_eq l
      rw [gdnBefore, ih]
      split <;> omega

/-- 48 layers hold 36 GDN layers. -/
theorem gdn_count_48 : gdnBefore 48 = 36 := by decide

/-- GDN indices are strictly increasing across consecutive GDN layers (distinct slices). -/
theorem gdn_index_strict (l : Nat) (h : l % 4 ≠ 3) : gdnBefore (l + 1) = gdnBefore l + 1 := by
  rw [gdnBefore, if_neg h]

/-- QSA layers do not advance the GDN index. -/
theorem qsa_no_advance (l : Nat) (h : l % 4 = 3) : gdnBefore (l + 1) = gdnBefore l := by
  rw [gdnBefore, if_pos h, Nat.add_zero]

-- § slices

/-- A slice `[i*F, i*F+F)` ends inside `n*F` for `i < n`. -/
theorem slice_end_le (i n F : Nat) (h : i < n) : i * F + F ≤ n * F := by
  have := Nat.mul_le_mul_right F (show i + 1 ≤ n by omega)
  simpa [Nat.add_mul] using this

/-- Slices of distinct layers do not overlap. -/
theorem slice_disjoint (i j F : Nat) (h : i < j) : i * F + F ≤ j * F := slice_end_le i j F h

-- § session arena

/-- 256-byte alignment. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment does not shrink. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- Alignment is a 256-multiple. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- `session_bytes` covers the raw sum, and carries 4096 of slack. -/
theorem session_bytes_cover (n : Nat) : n + 4096 ≤ align256 n + 4096 := by
  have := align256_ge n; omega

/-- All QSA layers after the first borrow the RoPE table, so the arena is `first + (q-1)*rest`. -/
theorem qsa_arena (first rest q : Nat) (h : 1 ≤ q) :
    first + (q - 1) * rest ≤ q * (first + rest) := by
  obtain ⟨k, rfl⟩ : ∃ k, q = k + 1 := ⟨q - 1, by omega⟩
  simp [Nat.add_mul, Nat.mul_add]
  omega

-- § stage flags

/-- Whether stage `i` (0..4) runs for `(half, stage_prefix)`. -/
def runStage (half pfx i : Nat) : Prop :=
  if 0 < pfx then i + 1 ≤ pfx
  else if i ≤ 2 then (half = 0 ∨ half = 1) else (half = 0 ∨ half = 2)

/-- A positive prefix runs exactly the stages below it, whatever `half` says. -/
theorem prefix_runs_below (half pfx i : Nat) (h : 0 < pfx) : runStage half pfx i ↔ i + 1 ≤ pfx := by
  unfold runStage; rw [if_pos h]

/-- Prefix 5 runs every stage. -/
theorem prefix5_all (half i : Nat) (hi : i < 5) : runStage half 5 i := by
  unfold runStage; rw [if_pos (by omega)]; omega

/-- Legacy callers (`half = 0`, no prefix) run all five stages. -/
theorem half0_all (i : Nat) (hi : i < 5) : runStage 0 0 i := by
  have h : ¬ (0 < 0) := by omega
  unfold runStage; rw [if_neg h]
  split <;> simp

/-- `half = 1` stops before the FFN front. -/
theorem half1_mixer_only (i : Nat) (hi : 3 ≤ i) : ¬ runStage 1 0 i := by
  have h : ¬ (0 < 0) := by omega
  have h2 : ¬ (i ≤ 2) := by omega
  unfold runStage; rw [if_neg h, if_neg h2]; omega

/-- `half = 2` skips the mixer entirely. -/
theorem half2_skips_mixer (i : Nat) (hi : i ≤ 2) : ¬ runStage 2 0 i := by
  have h : ¬ (0 < 0) := by omega
  unfold runStage; rw [if_neg h, if_pos hi]; omega

-- § timing

/-- Stage times from cumulative prefix times telescope back to the full-layer time. -/
theorem stage_times_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by omega

/-- Event `l*6+k` is in range for `l < n`, `k <= 5`. -/
theorem event_index (l n k : Nat) (hl : l < n) (hk : k ≤ 5) : l * 6 + k < n * 6 := by omega

/-- Ring numbers are strictly increasing per layer and the layer count ends at `n_layers`. -/
theorem ring_strict (l : Nat) : (l + 1) < (l + 1) + 1 := Nat.lt_succ_self _

end Ref.SessionGraphReplay



## FILE: /mnt/user-data/outputs/VerifyWindow.lean
/-!
# MODULE: verify_window
SOURCES: src/core/verify.cpp
The speculative-decoding verify window: token-group split, ring/flag numbering, monotone flag B, the GPU plan
layout in mapped memory, staging split and the commit rows.

## Source semantics
* With `groups = 2` tokens split into A = `[0, (T+1)/2)` and B = `[(T+1)/2, T)`; otherwise one group.
* Ring `k` (0-based) is layer `k / G`, group `k % G`, and waits for value `k + 1 = l*G + grp + 1`.
* Flag B only rises (compare-and-swap maximum), so out-of-order host functions are harmless.
* Plan layout: `counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)`.
* `commit(n_keep)`: `h_commit = [n_keep, n_keep-1, pos0+t for t<n_keep else -1 ...]`.

## Form template identifiers
tb/te ↦ `tb_[2]`, `te_[2]`; ring ↦ `l*G+grp+1`; raise ↦ `raise_flag`; ptrOff ↦ `(i32+1)&~1`.
-/
namespace Ref.VerifyWindow

-- § token groups

/-- Group begin. -/
def tb (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- Group end for `G` groups. -/
def te (G T grp : Nat) : Nat := if grp = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- With two groups the groups tile `[0,T)`: A ends where B begins. -/
theorem groups_tile (T : Nat) : te 2 T 0 = tb T 1 ∧ tb T 0 = 0 ∧ te 2 T 1 = T := by
  simp [te, tb]

/-- Sizes of the two groups add to T. -/
theorem groups_size (T : Nat) : (te 2 T 0 - tb T 0) + (te 2 T 1 - tb T 1) = T := by
  simp [te, tb]; omega

/-- With `T >= 2` both groups are non-empty. -/
theorem groups_nonempty (T : Nat) (h : 2 ≤ T) : 0 < te 2 T 0 - tb T 0 ∧ 0 < te 2 T 1 - tb T 1 := by
  simp [te, tb]; omega

/-- Group A is at least as large as group B (ceil split). -/
theorem groups_balanced (T : Nat) : te 2 T 1 - tb T 1 ≤ te 2 T 0 - tb T 0 := by
  simp [te, tb]; omega

/-- A single group is the whole window. -/
theorem single_group (T : Nat) : tb T 0 = 0 ∧ te 1 T 0 = T := by simp [tb, te]

-- § ring numbering

/-- The value the `k`-th ring waits for is `l*G + grp + 1` with `l = k/G`, `grp = k%G`. -/
theorem ring_numbering (k G : Nat) : G * (k / G) + k % G + 1 = k + 1 := by
  have h := Nat.div_add_mod k G
  omega

/-- The group index is a valid group. -/
theorem ring_group_lt (k G : Nat) (hG : 0 < G) : k % G < G := Nat.mod_lt _ hG

/-- Rings of consecutive `(l, grp)` pairs are consecutive. -/
theorem ring_consecutive (G l grp : Nat) (h : grp + 1 < G) :
    G * l + (grp + 1) + 1 = (G * l + grp + 1) + 1 := by omega

-- § flag B

/-- `raise_flag`: only ever raises. -/
def raise (cur v : Nat) : Nat := max cur v

/-- Raising never lowers the flag. -/
theorem raise_ge_cur (cur v : Nat) : cur ≤ raise cur v := by unfold raise; omega

/-- Raising reaches the requested value. -/
theorem raise_ge_val (cur v : Nat) : v ≤ raise cur v := by unfold raise; omega

/-- Late host functions commute: order of arrival does not matter. -/
theorem raise_comm (c a b : Nat) : raise (raise c a) b = raise (raise c b) a := by
  unfold raise; omega

/-- Replaying the same raise is idempotent. -/
theorem raise_idem (c v : Nat) : raise (raise c v) v = raise c v := by unfold raise; omega

-- § plan layout

/-- Offset of the pointer array: the i32 prefix rounded up to even. -/
def ptrOff (cap : Nat) : Nat := ((4 + (cap + 1) + cap + cap) + 1) / 2 * 2

/-- The pointer array is 8-byte aligned in i32 units (even offset). -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by unfold ptrOff; omega

/-- The pad is at most one i32 and the pointer array starts after `tok`. -/
theorem ptrOff_bounds (cap : Nat) :
    4 + (cap + 1) + cap + cap ≤ ptrOff cap ∧ ptrOff cap ≤ 4 + (cap + 1) + cap + cap + 1 := by
  unfold ptrOff; omega

/-- Total i32 count of one plan slot. -/
def planI32 (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- `ptr` and `ptr2` (2 i32 per u64, `cap` each) end exactly where `start2` begins. -/
theorem ptr2_ends_at_start2 (cap : Nat) : ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap := by
  omega

/-- `start2` (cap+1 entries) fits strictly inside the plan slot. -/
theorem start2_fits (cap : Nat) : ptrOff cap + 4 * cap + (cap + 1) < planI32 cap := by
  unfold planI32; omega

-- § staging split

/-- Blobs of staging per group. -/
def stagePer (G S : Nat) : Nat := if G = 2 then S / 2 else S

/-- Each group's staging window lies inside the staging area. -/
theorem stage_window_fits (G S grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    grp * stagePer G S + stagePer G S ≤ S := by
  rcases hG with rfl | rfl
  · have : grp = 0 := by omega
    subst this
    simp [stagePer]
  · have : grp = 0 ∨ grp = 1 := by omega
    rcases this with rfl | rfl <;> simp [stagePer] <;> omega

/-- The two windows of a split window are disjoint. -/
theorem stage_windows_disjoint (S : Nat) : 0 * stagePer 2 S + stagePer 2 S ≤ 1 * stagePer 2 S := by
  omega

-- § commit rows

/-- Entry `t` of the commit array (`-1` marks a discarded token). -/
def commitEntry (pos0 : Int) (nKeep t : Nat) : Int := if t < nKeep then pos0 + t else -1

/-- Kept tokens carry consecutive positions. -/
theorem commit_kept (pos0 : Int) (nKeep t : Nat) (h : t < nKeep) : commitEntry pos0 nKeep t = pos0 + t := by
  unfold commitEntry; rw [if_pos h]

/-- Dropped tokens are marked `-1`. -/
theorem commit_dropped (pos0 : Int) (nKeep t : Nat) (h : nKeep ≤ t) : commitEntry pos0 nKeep t = -1 := by
  unfold commitEntry; rw [if_neg (by omega)]

/-- The second header word (`n_keep - 1`) addresses the last kept token. -/
theorem commit_last_index (nKeep : Nat) (h : 1 ≤ nKeep) : nKeep - 1 < nKeep := by omega

end Ref.VerifyWindow



## FILE: /mnt/user-data/outputs/WeightPackLoader.lean
/-!
# MODULE: weight_pack_loader
SOURCES: src/core/weights.cpp
The dense-weight loader: segment list per tensor, plane-by-plane width conversion (bf16 high-half, f16
re-round, exact fp16→f32 widening), chunked staging, and arena compaction under a skip set.

## Source semantics
* A quantized tensor is codes ++ scales ++ optional offsets; fp16 scales are half the size in the pack.
* Promoted (bf16/f16-in-f32) tensors are 4 B/elem in and 2 B/elem out.
* Chunks never split an element: widening needs even chunks; bf16/f16 conversion needs multiples of 4.
* With a skip set the kept rows are re-placed in index order at 256-byte alignment.

## Form template identifiers
srcSum/dstSum ↦ `sum_src`/`sum_dst`; chunkLen ↦ `min(CHUNK, remaining)`; carve256 ↦ compaction `at += align`.
-/
namespace Ref.WeightPackLoader

-- § segment accounting

/-- Source bytes of the three planes. -/
def srcSum (codes scales off : Nat) (fp16 : Bool) : Nat :=
  codes + (if fp16 then scales / 2 else scales) + off

/-- Destination bytes of the three planes. -/
def dstSum (codes scales off : Nat) : Nat := codes + scales + off

/-- Without fp16 scales source and destination agree. -/
theorem seg_sums_copy (c s o : Nat) : srcSum c s o false = dstSum c s o := by
  simp [srcSum, dstSum]

/-- With fp16 scales the destination grows by exactly the scale plane's source size. -/
theorem seg_sums_widen (c s o : Nat) (h : s % 2 = 0) :
    dstSum c s o = srcSum c s o true + s / 2 := by
  simp only [srcSum, dstSum, if_true]; omega

/-- An odd fp16 destination byte count is impossible after widening (f32 planes are multiples of 4). -/
theorem widen_dst_even (elems : Nat) : (elems * 4) % 2 = 0 := by omega

/-- Widening `n` source bytes yields `2n` destination bytes. -/
theorem widen_out (n : Nat) (h : n % 2 = 0) : n / 2 * 4 = 2 * n := by omega

/-- Promoted tensors: 2 B/elem out is half of 4 B/elem in. -/
theorem promoted_halves (elems : Nat) : elems * 2 * 2 = elems * 4 := by omega

-- § chunking

/-- The chunk the loop reads: `min CHUNK remaining`. -/
def chunkLen (chunk remaining : Nat) : Nat := min chunk remaining

/-- A chunk makes progress when anything remains. -/
theorem chunk_progress (chunk rem : Nat) (hc : 0 < chunk) (hr : 0 < rem) :
    0 < chunkLen chunk rem ∧ rem - chunkLen chunk rem < rem := by
  unfold chunkLen; omega

/-- With the real 8 MiB CHUNK and an even plane, every chunk is even (no half fp16). -/
theorem chunk_even (rem done : Nat) (hr : rem % 2 = 0) (hd : done % 2 = 0) :
    chunkLen 8388608 (rem - done) % 2 = 0 ∨ rem < done := by
  unfold chunkLen; omega

/-- bf16/f16 conversion with 4-multiple chunks keeps destination offsets contiguous (halved). -/
theorem bf16_dst_contiguous (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + n / 4 * 2 = (done + n) / 2 := by omega

/-- fp16 widening keeps destination offsets contiguous (doubled). -/
theorem widen_dst_contiguous (done n : Nat) (hd : done % 2 = 0) (hn : n % 2 = 0) :
    done * 2 + n / 2 * 4 = (done + n) * 2 := by omega

/-- The staging output buffer (2 * CHUNK) holds the widest conversion (widen doubles a chunk). -/
theorem stage_out_fits (n chunk : Nat) (h : n ≤ chunk) : n / 2 * 4 ≤ chunk * 2 := by omega

-- § arena compaction

/-- 256-byte alignment of a row's destination size. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment does not shrink. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- Alignment gives multiples of 256. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- Compacted placement total. -/
def carve : List Nat → Nat → Nat
  | [], s => s
  | p :: ps, s => carve ps (s + align256 p)

/-- Placement is monotone. -/
theorem carve_ge : ∀ (parts : List Nat) (s : Nat), s ≤ carve parts s
  | [], s => Nat.le_refl s
  | p :: ps, s => by
      rw [carve]
      have h := carve_ge ps (s + align256 p)
      omega

/-- Placement splits over concatenation. -/
theorem carve_append : ∀ (xs ys : List Nat) (s : Nat), carve (xs ++ ys) s = carve ys (carve xs s)
  | [], ys, s => by simp [carve]
  | x :: xs, ys, s => by
      simp only [List.cons_append, carve]
      exact carve_append xs ys _

/-- Offsets of compacted rows stay 256-aligned. -/
theorem carve_aligned : ∀ (parts : List Nat) (s : Nat), s % 256 = 0 → carve parts s % 256 = 0
  | [], s, h => by simpa [carve] using h
  | p :: ps, s, h => by
      rw [carve]
      apply carve_aligned ps
      have := align256_mod p
      omega

/-- A kept row's `[dst_off, dst_off+aligned)` lies inside the pool: rows never overlap or overrun. -/
theorem row_fits (pre post : List Nat) (x : Nat) :
    carve pre 0 + align256 x ≤ carve (pre ++ x :: post) 0 := by
  rw [carve_append]
  have h := carve_ge post (carve pre 0 + align256 x)
  simp only [carve]
  exact h

/-- The row's own bytes lie inside the pool. -/
theorem row_bytes_fit (pre post : List Nat) (x : Nat) :
    carve pre 0 + x ≤ carve (pre ++ x :: post) 0 := by
  have h1 := row_fits pre post x
  have h2 := align256_ge x
  omega

-- § index row

/-- A parsed row must have exactly 19 fields. -/
def rowParsed (n : Nat) : Bool := n == 19

/-- Only 19 fields parse. -/
theorem row_parsed_iff (n : Nat) : rowParsed n = true ↔ n = 19 := by simp [rowParsed]

end Ref.WeightPackLoader





## FILE: /mnt/user-data/outputs/GenerateDriver.lean
/-!
# MODULE: generate_driver
SOURCES: src/program/generate.cpp
The `ref generate` driver: option validation, the per-token decode loop with teacher forcing, the
speculative verify-window loop, the VRAM expert-cache sizing, the routing trace, and the M-RoPE position
table used by the vision path of `--serve`.

## Source semantics
* A prompt is a non-empty list of token ids in `[0, 2^31)`; `--max-new`/`--max-context` must be positive and
  the prompt plus generation must fit the context, which itself is below `2^31 - 8`.
* The logits dump has `n_prompt - 1 + max_new` positions, because sampling only starts being recorded at the
  last prompt position.
* The routing trace records the layer as `d.layers - 1` (the adapter has already advanced it) and must lie in
  `0..47`.
* `--expert-cache auto` grants `(free - reserve) / max_blob` slots (never negative).
* The prompt path lends the last `k >= 1` cache slots and needs `k + 128 <= slots`.
* A speculative round has `1 <= T <= S` tokens, accepts the longest matching draft prefix `a <= T - 1`, commits
  `a + 1` tokens, and never overshoots `max_new` or `max_context`.
* Image rows get M-RoPE positions `(p, p + y, p + x)` with `y = j / nx`, `x = j % nx`; text after the image
  continues at `p + max(nx, ny)`.

## Form template identifiers
validConfig -> the `--max-new`/`--max-context` checks in `main`
dumpPositions -> `n_prompt - 1 + max_new`
layerIdx -> `(int32_t)(t->d.layers - 1)` in `drive_pool`
autoSlots -> `(free_b - reserve) / expert_layout().max_blob`
pickTAux -> the `while (T < S && dprob[T-1] >= min_p) ++T` loop
accPrefix -> the `while (a < T - 1 && window[a+1] == outv[a]) ++a` loop
imgPos -> `put(i + j, p, p + y, p + x)`
-/

namespace Ref.Program

-- § argument validation

/-- A token id accepted by `parse_i64_list`: a non-negative int32. -/
def validId (n : Int) : Prop := 0 ≤ n ∧ n ≤ 2147483647

/-- Negative ids are rejected. -/
theorem negative_id_rejected (n : Int) (h : n < 0) : ¬ validId n := by
  unfold validId
  omega

/-- Ids above int32 range are rejected. -/
theorem huge_id_rejected (n : Int) (h : 2147483647 < n) : ¬ validId n := by
  unfold validId
  omega

/-- The size checks `main` performs before anything is allocated. -/
def validConfig (promptLen maxNew maxCtx : Nat) : Prop :=
  0 < maxNew ∧ 0 < maxCtx ∧ maxNew ≤ maxCtx ∧ promptLen ≤ maxCtx - maxNew

/-- A valid configuration holds the prompt and every generated token. -/
theorem config_fits (p n c : Nat) (h : validConfig p n c) : p + n ≤ c := by
  unfold validConfig at h
  omega

/-- With `max_context < 2^31 - 8`, positions and cells stay within int32 on the device. -/
theorem positions_fit_int32 (p n c : Nat) (h : validConfig p n c) (hc : c ≤ 2147483647 - 8) :
    p + n + 8 ≤ 2147483647 := by
  unfold validConfig at h
  omega

-- § logits dump header

/-- Number of positions the logits dump covers. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- The header count equals the rows actually written: one per processed position. -/
theorem dumpPositions_eq (nPrompt maxNew : Nat) (h : 1 ≤ nPrompt) :
    dumpPositions nPrompt maxNew + 1 = nPrompt + maxNew := by
  unfold dumpPositions
  omega

-- § routing trace

/-- Layer index recorded by `drive_pool`: the adapter has advanced `d.layers` already. -/
def layerIdx (layersAfter : Int) : Int := layersAfter - 1

/-- The trace's bounds check accepts exactly the post-advance counts `1..48`. -/
theorem layerIdx_valid_iff (l : Int) : (0 ≤ layerIdx l ∧ layerIdx l < 48) ↔ (1 ≤ l ∧ l ≤ 48) := by
  unfold layerIdx
  omega

-- § VRAM expert cache sizing

/-- Slots granted by `--expert-cache auto`. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto-sized cache never needs more VRAM than is free. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free := by
  unfold autoSlots
  have := Nat.div_mul_le_self (free - reserve) blob
  omega

/-- Lending `k` slots to the prompt path leaves the first lent index valid and at least 128 slots resident. -/
theorem lend_first_bounds (slots k : Nat) (hk : 1 ≤ k) (h : k + 128 ≤ slots) :
    128 ≤ slots - k ∧ slots - k < slots := by
  omega

/-- The number of lent slots for a non-empty buffer is at least one. -/
theorem lent_count_pos (need blob : Nat) (hn : 0 < need) (hb : 0 < blob) :
    1 ≤ (need + blob - 1) / blob := by
  apply Nat.div_pos
  · omega
  · exact hb

-- § PCIe share

/-- `pcie_num`: the fraction in 1/256ths, clamped to `[0, 256]`. -/
def pcieNum (x : Int) : Int := max 0 (min 256 x)

/-- The clamped share is always a valid fraction numerator. -/
theorem pcieNum_bounds (x : Int) : 0 ≤ pcieNum x ∧ pcieNum x ≤ 256 := by
  unfold pcieNum
  omega

-- § speculative rounds

/-- Verify-window size: grows while the drafts stay above the probability threshold. -/
def pickTAux (ok : Nat → Bool) (S : Nat) : Nat → Nat → Nat
  | 0, T => T
  | f + 1, T => if T < S ∧ ok (T - 1) = true then pickTAux ok S f (T + 1) else T

/-- The window never shrinks and never exceeds `S`. -/
theorem pickTAux_bounds (ok : Nat → Bool) (S f : Nat) :
    ∀ T, T ≤ S → T ≤ pickTAux ok S f T ∧ pickTAux ok S f T ≤ S := by
  induction f with
  | zero =>
    intro T h
    simp only [pickTAux]
    exact ⟨Nat.le_refl _, h⟩
  | succ f ih =>
    intro T h
    simp only [pickTAux]
    split
    · rename_i hc
      obtain ⟨c1, _⟩ := hc
      have := ih (T + 1) (by omega)
      omega
    · exact ⟨Nat.le_refl _, h⟩

/-- Longest prefix on which the drafts equal the model's outputs. -/
def accPrefix : List Int → List Int → Nat
  | w :: ws, o :: os => if w = o then 1 + accPrefix ws os else 0
  | _, _ => 0

/-- The accepted prefix is no longer than either list. -/
theorem accPrefix_le (ws os : List Int) : accPrefix ws os ≤ min ws.length os.length := by
  induction ws generalizing os with
  | nil => simp [accPrefix]
  | cons w ws ih =>
    cases os with
    | nil => simp [accPrefix]
    | cons o os =>
      have := ih os
      simp only [accPrefix, List.length_cons]
      split <;> omega

/-- Accepted drafts plus the one verified token never exceed the window: `a + 1 <= T`. -/
theorem round_commit_le_window (a T : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) : a + 1 ≤ T := by
  omega

/-- Committing `a + 1` tokens keeps the position inside the context checked before the window ran. -/
theorem round_position_in_context (p a T ctx : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) (hc : p + T ≤ ctx) :
    p + a + 1 ≤ ctx := by
  omega

/-- Tokens emitted in a round are capped by what is still wanted. -/
def emitCount (a produced maxNew : Nat) : Nat := min (a + 1) (maxNew - produced)

/-- A round never overshoots `--max-new`. -/
theorem emit_no_overshoot (a produced maxNew : Nat) (h : produced ≤ maxNew) :
    produced + emitCount a produced maxNew ≤ maxNew := by
  unfold emitCount
  omega

-- § M-RoPE positions for images

/-- The `(h, w)` offsets of image row `j` in an `nx`-wide grid. -/
def imgPos (nx j : Nat) : Nat × Nat := (j / nx, j % nx)

/-- Row offsets stay inside the grid, so positions never exceed `p + max(nx, ny)`. -/
theorem imgPos_bounds (nx ny j : Nat) (hx : 0 < nx) (hj : j < nx * ny) :
    (imgPos nx j).1 < ny ∧ (imgPos nx j).2 < nx := by
  unfold imgPos
  exact ⟨Nat.div_lt_of_lt_mul hj, Nat.mod_lt _ hx⟩

/-- Every image cell's `h` and `w` position is strictly below where text resumes. -/
theorem img_text_resumes_after (p nx ny j : Nat) (hx : 0 < nx) (hj : j < nx * ny) :
    p + (imgPos nx j).1 < p + max nx ny ∧ p + (imgPos nx j).2 < p + max nx ny := by
  have ⟨h1, h2⟩ := imgPos_bounds nx ny j hx hj
  omega

/-- Text cells advance the position by exactly one. -/
theorem text_position_advances (p : Nat) : p + 1 > p := by
  omega

end Ref.Program



## FILE: /mnt/user-data/outputs/MemoryPlanner.lean
/-!
# MODULE: memory_planner
SOURCES: include/ref/plan/plan.hpp
Derives, from the model geometry alone, how much VRAM the KV cache, indexer keys and GDN state take, and how
many expert-cache slots remain; refuses (DoesNotClose) rather than returning a plan it cannot honour.

## Source semantics
* GDN state is fixed and not evictable: `n_gdn * (S*h_v*S*4 + (d_conv-1)*C*4)` bytes.
* KV bytes per token: `n_qsa * (INT8 K+V elements + fp16 group scales + indexer key bytes)`, where the indexer
  caches ONE shared key head (`key_heads * key_dim / block`), not one per query head.
* `make_plan`: reject if `kv + fixed > pool`; the cache gets `pool - kv - fixed`; slots are whole blobs; reject
  if there is not even one slot; the slot remainder is unusable and counted as used.

## Form template identifiers
stateBytes -> `state_bytes(Geometry)`
kvBytesPerToken -> `kv_bytes_per_token(Geometry)`
makePlan -> `make_plan` (None = throws `DoesNotClose`)
-/

namespace Ref.Plan

-- § geometry

/-- Model geometry fields used by the byte costs. -/
structure Geometry where
  nLayers : Nat
  nQsa : Nat
  nKvHeads : Nat
  headDim : Nat
  kvGroup : Nat
  qsaBlock : Nat
  idxKeyHeads : Nat
  idxKeyDim : Nat
  ssmState : Nat
  ssmVHeads : Nat
  ssmDConv : Nat
  ssmConvChannels : Nat

/-- The shipped artifact's geometry (the defaults of `plan::Geometry`). -/
def defaultG : Geometry :=
  { nLayers := 48, nQsa := 12, nKvHeads := 2, headDim := 256, kvGroup := 64, qsaBlock := 4,
    idxKeyHeads := 1, idxKeyDim := 128, ssmState := 128, ssmVHeads := 48, ssmDConv := 4,
    ssmConvChannels := 10240 }

/-- Number of GDN layers: derived, not a third constant. -/
def Geometry.nGdn (g : Geometry) : Nat := g.nLayers - g.nQsa

/-- GDN recurrence + conv history bytes per sequence. -/
def stateBytes (g : Geometry) : Nat :=
  g.nGdn * (g.ssmState * g.ssmVHeads * g.ssmState * 4 + (g.ssmDConv - 1) * g.ssmConvChannels * 4)

/-- INT8 KV plus indexer-key bytes per token. -/
def kvBytesPerToken (g : Geometry) : Nat :=
  g.nQsa * ((2 * g.nKvHeads * g.headDim + (2 * g.nKvHeads * g.headDim) / g.kvGroup * 2) +
    g.idxKeyHeads * g.idxKeyDim / g.qsaBlock)

-- § concrete values of the shipped model

/-- 36 of the 48 layers are GDN. -/
theorem default_n_gdn : defaultG.nGdn = 36 := by decide

/-- The GDN state is 117,669,888 bytes. -/
theorem default_state_bytes : stateBytes defaultG = 117669888 := by decide

/-- The GDN state costs 85 whole expert slots (1,382,400 B each), the figure the source quotes. -/
theorem default_state_is_85_slots : stateBytes defaultG / 1382400 = 85 := by decide

/-- KV plus indexer bytes per token: 13,056 (12 layers x (1024 + 32 + 32)). -/
theorem default_kv_per_token : kvBytesPerToken defaultG = 13056 := by decide

-- § planning

/-- The result of a successful plan (the numeric fields of `plan::Plan`). -/
structure Plan where
  kvBytes : Nat
  stateBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramUsed : Nat

/-- `make_plan`: `none` models `DoesNotClose`. `fixed` is dense + embedding + MTP + workspace + state. -/
def makePlan (ctx pool blob kvPerTok fixed state : Nat) : Option Plan :=
  if kvPerTok * ctx + fixed > pool then none
  else if (pool - kvPerTok * ctx - fixed) / blob = 0 then none
  else some { kvBytes := kvPerTok * ctx, stateBytes := state,
              cacheBytes := pool - kvPerTok * ctx - fixed,
              cacheSlots := (pool - kvPerTok * ctx - fixed) / blob,
              vramUsed := pool - (pool - kvPerTok * ctx - fixed) % blob }

/-- A plan whose fixed costs plus KV exceed the pool is refused. -/
theorem makePlan_refuses_overflow (ctx pool blob kvPerTok fixed state : Nat)
    (h : kvPerTok * ctx + fixed > pool) : makePlan ctx pool blob kvPerTok fixed state = none := by
  unfold makePlan
  simp [h]

/-- A plan with no room for even one slot is refused. -/
theorem makePlan_refuses_no_slot (ctx pool blob kvPerTok fixed state : Nat)
    (h1 : ¬ kvPerTok * ctx + fixed > pool) (h2 : (pool - kvPerTok * ctx - fixed) / blob = 0) :
    makePlan ctx pool blob kvPerTok fixed state = none := by
  unfold makePlan
  simp [h1, h2]

/-- Every accepted plan exactly splits the pool into KV, fixed costs and cache. -/
theorem makePlan_budget {ctx pool blob kvPerTok fixed state : Nat} {p : Plan}
    (h : makePlan ctx pool blob kvPerTok fixed state = some p) :
    p.kvBytes + fixed + p.cacheBytes = pool := by
  unfold makePlan at h
  by_cases c1 : kvPerTok * ctx + fixed > pool
  · simp [c1] at h
  · by_cases c2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      subst h
      simp
      omega

/-- Every accepted plan has at least one expert slot. -/
theorem makePlan_slots_pos {ctx pool blob kvPerTok fixed state : Nat} {p : Plan}
    (h : makePlan ctx pool blob kvPerTok fixed state = some p) : 0 < p.cacheSlots := by
  unfold makePlan at h
  by_cases c1 : kvPerTok * ctx + fixed > pool
  · simp [c1] at h
  · by_cases c2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      subst h
      exact Nat.pos_of_ne_zero c2

/-- The slots never take more bytes than the cache budget. -/
theorem makePlan_slots_fit {ctx pool blob kvPerTok fixed state : Nat} {p : Plan}
    (h : makePlan ctx pool blob kvPerTok fixed state = some p) :
    p.cacheSlots * blob ≤ p.cacheBytes := by
  unfold makePlan at h
  by_cases c1 : kvPerTok * ctx + fixed > pool
  · simp [c1] at h
  · by_cases c2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      subst h
      simp
      exact Nat.div_mul_le_self _ _

/-- Whole slots plus the unusable remainder account for the entire cache budget. -/
theorem slots_remainder_split (cache blob : Nat) : (cache / blob) * blob + cache % blob = cache := by
  have := Nat.div_add_mod cache blob
  rw [Nat.mul_comm] at this
  exact this

/-- The reported VRAM use never exceeds the pool. -/
theorem makePlan_used_le {ctx pool blob kvPerTok fixed state : Nat} {p : Plan}
    (h : makePlan ctx pool blob kvPerTok fixed state = some p) : p.vramUsed ≤ pool := by
  unfold makePlan at h
  by_cases c1 : kvPerTok * ctx + fixed > pool
  · simp [c1] at h
  · by_cases c2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      subst h
      simp <;> omega

-- § monotonicity in the context length

/-- KV bytes grow with the requested context. -/
theorem kv_mono (kvPerTok c1 c2 : Nat) (h : c1 ≤ c2) : kvPerTok * c1 ≤ kvPerTok * c2 :=
  Nat.mul_le_mul_left kvPerTok h

/-- A context that does not close stays unclosable at any larger context. -/
theorem overflow_persists (kvPerTok c1 c2 pool fixed : Nat) (h : c1 ≤ c2)
    (hov : kvPerTok * c1 + fixed > pool) : kvPerTok * c2 + fixed > pool := by
  have := kv_mono kvPerTok c1 c2 h
  omega

/-- The VRAM pool constant of the README table. -/
def vramPool : Nat := 5943000000

end Ref.Plan



## FILE: /mnt/user-data/outputs/PleDirectReader.lean
/-!
# MODULE: ple_direct_reader
SOURCES: include/ref/ngram/ple_reader.hpp, src/ngram/ple_reader.cpp
Reads 90-byte n-gram table rows straight from the SSD with unbuffered, page-aligned reads, deduplicating pages
within a ticket, with a bounded 8-way set-associative row cache.

## Source semantics
* A row at file offset `off` is fetched by one read starting at `first = off / 4096 * 4096`, of length
  `4096` or `8192` (a row straddling a page boundary), and the row is found `off - first` bytes into the buffer.
* Every read buffer slot is `2 * PAGE`, so no read overflows its slot.
* `open` validates `table_offset + n_rows * 90 <= file size`; rows `>= n_rows` read as zeros.
* Ticket ids are 32-bit, wrap around, and 0 is never a valid id.
* The latency log is a ring of 65,536 entries; the percentile index is clamped to the last entry.

## Form template identifiers
firstOf / lengthOf / inPage -> `first`, `length`, `in_page` in `PleReader::issue`
nextTicket -> `m.next_ticket++ ; if (m.next_ticket == 0) m.next_ticket = 1`
setIndex -> `mix(row) % sets` in `RowCache`
pctIndex -> `std::min(v.size() - 1, ...)` in `ReaderStats::percentile`
-/

namespace Ref.Ngram

-- § constants

/-- `ROW_BYTES`. -/
def rowBytes : Nat := 90
/-- `PAGE`. -/
def pageBytes : Nat := 4096
/-- Cache associativity (`WAYS`). -/
def ways : Nat := 8

-- § page arithmetic

/-- First byte of the aligned read covering offset `off`. -/
def firstOf (off : Nat) : Nat := off / 4096 * 4096

/-- Read length: from `first` through the page holding the row's last byte. -/
def lengthOf (off : Nat) : Nat := (off + 89) / 4096 * 4096 - firstOf off + 4096

/-- Offset of the row inside the read buffer. -/
def inPage (off : Nat) : Nat := off - firstOf off

/-- The aligned read starts at or before the row. -/
theorem firstOf_le (off : Nat) : firstOf off ≤ off := by
  unfold firstOf
  omega

/-- The aligned read start is page aligned (required by unbuffered I/O). -/
theorem firstOf_aligned (off : Nat) : firstOf off % 4096 = 0 := by
  unfold firstOf
  omega

/-- A read is one page, or two when the row straddles a page boundary. -/
theorem lengthOf_cases (off : Nat) : lengthOf off = 4096 ∨ lengthOf off = 8192 := by
  unfold lengthOf firstOf
  omega

/-- The read is a whole number of pages. -/
theorem lengthOf_aligned (off : Nat) : lengthOf off % 4096 = 0 := by
  rcases lengthOf_cases off with h | h <;> omega

/-- A read never overflows its `2 * PAGE` slot. -/
theorem lengthOf_le_slot (off : Nat) : lengthOf off ≤ 2 * 4096 := by
  rcases lengthOf_cases off with h | h <;> omega

/-- The row's last byte lies inside the read. -/
theorem row_covered (off : Nat) : off + 90 ≤ firstOf off + lengthOf off := by
  unfold lengthOf firstOf
  omega

/-- The in-buffer offset is smaller than a page. -/
theorem inPage_lt_page (off : Nat) : inPage off < 4096 := by
  unfold inPage firstOf
  omega

/-- The "short read" guard never fires for a full read: `in_page + ROW_BYTES <= length`. -/
theorem inPage_row_fits (off : Nat) : inPage off + 90 ≤ lengthOf off := by
  unfold inPage lengthOf firstOf
  omega

/-- Rows of the same page share the same aligned read start. -/
theorem same_page_same_first (a b : Nat) (h : a / 4096 = b / 4096) : firstOf a = firstOf b := by
  unfold firstOf
  rw [h]

/-- A merged job takes the larger length, which still fits a slot. -/
theorem merged_length_le (a b : Nat) : max (lengthOf a) (lengthOf b) ≤ 2 * 4096 := by
  have ha := lengthOf_le_slot a
  have hb := lengthOf_le_slot b
  omega

-- § table bounds

/-- If `open`'s size check passes, every in-range row lies fully inside the file. -/
theorem row_in_file (tableOffset nRows size row : Nat) (h : tableOffset + nRows * 90 ≤ size)
    (hr : row < nRows) : tableOffset + row * 90 + 90 ≤ size := by
  omega

-- § ticket ids

/-- The next ticket counter value: a 32-bit increment that skips 0. -/
def nextTicket (t : Nat) : Nat :=
  if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- The counter never becomes 0, so 0 stays free as the "no ticket" value. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split <;> omega

/-- The counter stays a valid 32-bit value. -/
theorem nextTicket_lt (t : Nat) : nextTicket t < 4294967296 := by
  unfold nextTicket
  split <;> omega

-- § row cache

/-- Set index of a row (`mix % sets`), given the mixed hash value. -/
def setIndex (mixed sets : Nat) : Nat := mixed % sets

/-- The set index is in range whenever the cache has at least one set. -/
theorem setIndex_lt (mixed sets : Nat) (h : 0 < sets) : setIndex mixed sets < sets :=
  Nat.mod_lt _ h

/-- Round-robin replacement pointer within a set. -/
def nextWay (w : Nat) : Nat := (w + 1) % 8

/-- The replacement pointer stays inside the 8 ways. -/
theorem nextWay_lt (w : Nat) : nextWay w < 8 := by
  unfold nextWay
  omega

/-- Cache capacity `sets * WAYS` never exceeds the requested row count. -/
theorem cache_capacity_le (rows : Nat) : rows / 8 * 8 ≤ rows := Nat.div_mul_le_self rows 8

/-- The slot index `set * WAYS + way` is below `sets * WAYS`. -/
theorem slot_index_lt (s w sets : Nat) (hs : s < sets) (hw : w < 8) : s * 8 + w < sets * 8 := by
  omega

-- § latency ring and percentile

/-- Overwrite position in the 65,536-entry latency ring. -/
def ringSlot (pos : Nat) : Nat := pos % 65536

/-- The ring write position is always in range. -/
theorem ringSlot_lt (pos : Nat) : ringSlot pos < 65536 := by
  unfold ringSlot
  omega

/-- Percentile index, given the rounded `q * (n - 1)` value, clamped to the last sample. -/
def pctIndex (n rounded : Nat) : Nat := min (n - 1) rounded

/-- For a non-empty sample the percentile index is a valid array index. -/
theorem pctIndex_lt (n rounded : Nat) (h : 0 < n) : pctIndex n rounded < n := by
  unfold pctIndex
  omega

end Ref.Ngram



## FILE: /mnt/user-data/outputs/PleNgram.lean
/-!
# MODULE: ple_ngram
SOURCES: include/ref/kernels/ngram.hpp, src/kernels/ngram.cpp
The PLE n-gram hash (16 row indices per token from the last three token ids) and the IQ4_NL row layout
of the per-layer token table.

## Source semantics
* Head `h = (n-2)*8 + g` for `n in {2,3}`, `g < 8`; its row is `mixed_n % vocab[h] + offset[h]`.
* `head_offsets` is the running sum of `head_vocab_sizes`, and the last head ends below the table's row count,
  so every hashed row is in range.
* The EOS cut propagates forward: once a predecessor is missing (`< 0`) or EOS, it and all older positions
  become EOS; the token's own id is never cut.
* `mixed` XORs products, so equal terms cancel (a sum would double them).
* A row is 5 IQ4_NL blocks of 18 bytes = 90 bytes = 160 values; sixteen rows flatten head-slowest to 2560.
* Nibble order is split-half: byte `j` holds elements `j` and `j+16`.

## Form template identifiers
ctxStep -> the per-predecessor `cut`/`ctx[s]` update in `ngram_rows`
headIndex -> `(n - 2) * HEADS_PER_NGRAM + g`
vocabs / offsets -> `ple_artifact_consts().vocab / .offset`
loByte / hiHalf -> `qs[j] & 0x0F` / `qs[j] >> 4` in `iq4nl_dequant_row`
-/

namespace Ref.Kernels

-- § geometry constants

/-- `PLE_HEAD_DIM`. -/
def pleHeadDim : Nat := 160
/-- `NGRAM_SIZE`. -/
def ngramSize : Nat := 3
/-- `HEADS_PER_NGRAM`. -/
def headsPerNgram : Nat := 8
/-- `PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM`. -/
def pleNHeads : Nat := (ngramSize - 1) * headsPerNgram
/-- `PLE_ROW_BYTES = (PLE_HEAD_DIM / 32) * 18`. -/
def pleRowBytes : Nat := (pleHeadDim / 32) * 18
/-- `PLE_TABLE_ROWS`. -/
def pleTableRows : Nat := 320001536

/-- Sixteen heads. -/
theorem n_heads_sixteen : pleNHeads = 16 := by decide

/-- A row is 90 bytes. -/
theorem row_bytes_ninety : pleRowBytes = 90 := by decide

/-- Sixteen gathered rows flatten to `n_embd = 2560` values. -/
theorem gather_is_n_embd : pleNHeads * pleHeadDim = 2560 := by decide

/-- The history window `(conv_kernel - 1) * ngram_size` is 9. -/
theorem hist_is_nine : (4 - 1) * ngramSize = 9 := by decide

-- § head indexing

/-- Head index of bigram/trigram `n` and sub-head `g`. -/
def headIndex (n g : Nat) : Nat := (n - 2) * 8 + g

/-- Heads of `n in {2,3}` cover `0..15` without leaving the 16-entry arrays. -/
theorem headIndex_lt (n g : Nat) (h1 : 2 ≤ n) (h2 : n ≤ 3) (hg : g < 8) : headIndex n g < 16 := by
  unfold headIndex
  omega

/-- Distinct `(n, g)` pairs give distinct heads. -/
theorem headIndex_inj (n m g k : Nat) (h1 : 2 ≤ n) (h2 : n ≤ 3) (h3 : 2 ≤ m) (h4 : m ≤ 3)
    (hg : g < 8) (hk : k < 8) (h : headIndex n g = headIndex m k) : n = m ∧ g = k := by
  unfold headIndex at h
  omega

-- § hash constants

/-- `head_vocab_sizes` of the artifact. -/
def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- `head_offsets` of the artifact. -/
def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Running sum of vocabulary sizes. -/
def prefixSums : Nat → List Nat → List Nat
  | _, [] => []
  | acc, v :: vs => acc :: prefixSums (acc + v) vs

/-- The offsets are exactly the running sum of the vocab sizes. -/
theorem offsets_are_prefix_sums : offsets = prefixSums 0 vocabs := by decide

/-- Both arrays have one entry per head. -/
theorem consts_length : vocabs.length = 16 ∧ offsets.length = 16 := by decide

/-- The vocabulary sizes together fit inside the table. -/
theorem vocab_sum_fits : vocabs.foldl (· + ·) 0 ≤ pleTableRows := by decide

/-- The last head ends at 320,001,446, below the table's 320,001,536 rows. -/
theorem last_head_end : 300001275 + 20000171 = 320001446 ∧ 320001446 ≤ pleTableRows := by decide

/-- A hashed row `mixed % vocab + offset` is strictly below `vocab + offset`. -/
theorem hashed_row_lt (mixed vocab offset : Nat) (h : 0 < vocab) :
    mixed % vocab + offset < vocab + offset := by
  have := Nat.mod_lt mixed h
  omega

/-- Hence every hashed row lies in the table, whenever `vocab + offset` is within it. -/
theorem hashed_row_in_table (mixed vocab offset : Nat) (h : 0 < vocab)
    (hend : vocab + offset ≤ pleTableRows) : mixed % vocab + offset < pleTableRows := by
  have := hashed_row_lt mixed vocab offset h
  omega

-- § XOR structure

/-- Mixed hash of two terms. -/
def mixed2 (c0 c1 m0 m1 : Nat) : Nat := (c0 * m0) ^^^ (c1 * m1)

/-- Equal context tokens with equal multipliers cancel to zero under XOR (a sum would double them). -/
theorem xor_equal_terms_cancel (c m : Nat) : mixed2 c c m m = 0 := by
  unfold mixed2
  exact Nat.xor_self _

/-- XOR combination is order-independent. -/
theorem mixed2_comm (c0 c1 m0 m1 : Nat) : mixed2 c0 c1 m0 m1 = mixed2 c1 c0 m1 m0 := by
  unfold mixed2
  exact Nat.xor_comm _ _

-- § EOS cut

/-- EOS token id. -/
def eosId : Int := 248044

/-- One predecessor step: new `cut` flag and the stored context value. -/
def ctxStep (cut : Bool) (t : Int) : Bool × Int :=
  let cut' := cut || decide (t < 0) || decide (t = eosId)
  (cut', if cut' then eosId else t)

/-- Once cut, every older position stays cut and becomes EOS. -/
theorem ctxStep_cut_propagates (t : Int) : ctxStep true t = (true, eosId) := by
  simp [ctxStep]

/-- A real, non-EOS predecessor with no earlier cut is kept verbatim (token id 0 is real). -/
theorem ctxStep_keeps_real (t : Int) (h0 : 0 ≤ t) (hne : t ≠ eosId) :
    ctxStep false t = (false, t) := by
  have h1 : ¬ t < 0 := by omega
  simp [ctxStep, h1, hne]

/-- A missing predecessor (`-1`) starts the cut. -/
theorem ctxStep_missing_cuts : ctxStep false (-1) = (true, eosId) := by
  decide

/-- An EOS predecessor is itself EOS and cuts everything older. -/
theorem ctxStep_eos_cuts : ctxStep false eosId = (true, eosId) := by
  decide

/-- Predecessor `s` back, oldest-first window of two, lives at index `2 - s`; `s = 1` is the newest. -/
theorem prev_index_in_window (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) : 2 - s < 2 ∧ (s = 1 → 2 - s = 1) := by
  omega

-- § IQ4_NL layout

/-- The IQ4_NL codebook. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries. -/
theorem iq4nl_length : iq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_sorted : List.Pairwise (· < ·) iq4nl := by decide

/-- Every codebook value fits an int8. -/
theorem iq4nl_int8 : ∀ v ∈ iq4nl, -128 ≤ v ∧ v ≤ 127 := by decide

/-- Byte index holding element `e` (0..31) of a block: split-half order. -/
def byteOf (e : Nat) : Nat := e % 16

/-- Whether element `e` is the high nibble of its byte. -/
def hiHalf (e : Nat) : Nat := e / 16

/-- Split-half packing: element `e` is `byte + 16 * half`, so byte `j` holds elements `j` and `j + 16`. -/
theorem split_half_roundtrip (e : Nat) : byteOf e + 16 * hiHalf e = e := by
  unfold byteOf hiHalf
  omega

/-- Every element of a 32-element block maps to a byte below 16 and a half below 2. -/
theorem split_half_in_range (e : Nat) (h : e < 32) : byteOf e < 16 ∧ hiHalf e < 2 := by
  unfold byteOf hiHalf
  omega

/-- Distinct elements map to distinct (byte, half) pairs. -/
theorem split_half_injective (a b : Nat) (ha : a < 32) (hb : b < 32)
    (h1 : byteOf a = byteOf b) (h2 : hiHalf a = hiHalf b) : a = b := by
  unfold byteOf hiHalf at *
  omega

end Ref.Kernels



## FILE: /mnt/user-data/outputs/SpecDraftController.lean
/-!
# MODULE: spec_draft_controller
SOURCES: src/spec/controller.cpp, include/ref/spec/controller.hpp
Chooses, per verify step, which drafter (none / suffix lookup / MTP) and how many draft tokens `k`
maximise expected committed tokens per millisecond, and learns per-position acceptance with an EMA.

## Source semantics
* `K_MAX = 8`; a verify pass of `k` drafts has `n = clamp(k+1, 1, K_MAX+1)` tokens, indexing the ratio
  tables at `n-1`, which must stay inside the `K_MAX+1`-entry arrays.
* `bucket(match)` maps a match length to one of four lookup-acceptance buckets (<5, 5..7, 8..15, >=16).
* `expected(p, k)` = 1 + sum of running products of per-position acceptance (the verify pass always
  commits one token).
* `choose` starts from plain decoding (k = 0) and only replaces it by a candidate that is strictly better,
  and finally falls back to plain decoding unless the gain exceeds `min_gain`.
* `observe` clamps `accepted` to `[0, k]` and updates positions `0 .. min(k, accepted+1) - 1` by an EMA
  step toward 1 (accepted) or 0 (the rejected position).

## Form template identifiers
bucket -> `Controller::bucket` (match-length bucketing)
expAux/expected -> `Controller::expected`
nTok -> `std::clamp(k + 1, 1, K_MAX + 1)` in `CostModel::step_ms`
seen -> `std::min(c.k, accepted + 1)` in `Controller::observe`
-/

namespace Ref.Spec

-- § constants and bucketing

/-- `K_MAX` of the controller header. -/
def kMax : Nat := 8

/-- Lookup acceptance bucket of a match length, as `Controller::bucket`. -/
def bucket (m : Nat) : Nat :=
  if m ≥ 16 then 3 else if m ≥ 8 then 2 else if m ≥ 5 then 1 else 0

/-- Every bucket index is a valid index into the 4-entry `lookup_q_` array. -/
theorem bucket_lt_four (m : Nat) : bucket m < 4 := by
  unfold bucket
  split <;> (try split) <;> (try split) <;> omega

/-- A longer match never falls into a lower bucket. -/
theorem bucket_mono {m n : Nat} (h : m ≤ n) : bucket m ≤ bucket n := by
  unfold bucket
  split <;> split <;> (try split) <;> (try split) <;> (try split) <;> (try split) <;> omega

-- § verify-window size and table indexing

/-- Number of tokens of a verify pass with `k` drafts: `clamp(k+1, 1, K_MAX+1)`. -/
def nTok (k : Nat) : Nat := min (max (k + 1) 1) (kMax + 1)

/-- The table index `n - 1` is always inside the `K_MAX + 1`-entry ratio arrays. -/
theorem nTok_idx_lt (k : Nat) : nTok k - 1 < kMax + 1 := by
  unfold nTok kMax
  omega

/-- A verify pass always holds at least one token. -/
theorem nTok_pos (k : Nat) : 1 ≤ nTok k := by
  unfold nTok kMax
  omega

/-- The default ratio tables (scaled by 100) have `K_MAX + 1` entries. -/
theorem ratio_tables_length :
    [100, 105, 130, 145, 185, 220, 260, 300, 330].length = kMax + 1 ∧
    [100, 170, 231, 288, 340, 389, 435, 480, 520].length = kMax + 1 := by
  decide

/-- Both default ratio tables are non-decreasing: bigger windows never cost less. -/
theorem ratio_tables_sorted :
    List.Pairwise (· ≤ ·) [100, 105, 130, 145, 185, 220, 260, 300, 330] ∧
    List.Pairwise (· ≤ ·) [100, 170, 231, 288, 340, 389, 435, 480, 520] := by
  decide

-- § expected committed tokens

/-- Sum of running products of acceptance (percent), starting from run length `run`. -/
def expAux : Nat → List Nat → Nat
  | _, [] => 0
  | run, p :: ps => run * p / 100 + expAux (run * p / 100) ps

/-- `expected` scaled by 100: the committed token plus the accepted drafts. -/
def expected (ps : List Nat) : Nat := 100 + expAux 100 ps

/-- One acceptance step never increases the running product. -/
theorem runstep_le (run p : Nat) (hp : p ≤ 100) : run * p / 100 ≤ run := by
  apply Nat.div_le_of_le_mul
  have := Nat.mul_le_mul_left run hp
  omega

/-- The verify pass always commits at least one token. -/
theorem expected_ge_one (ps : List Nat) : 100 ≤ expected ps := by
  unfold expected
  omega

/-- With acceptances at most 100%, the accumulated drafts are bounded by `run * #drafts`. -/
theorem expAux_le (ps : List Nat) :
    ∀ run, (∀ p ∈ ps, p ≤ 100) → expAux run ps ≤ run * ps.length := by
  induction ps with
  | nil =>
    intro run _
    simp [expAux]
  | cons p ps ih =>
    intro run h
    have hp : p ≤ 100 := h p (by simp)
    have hr : run * p / 100 ≤ run := runstep_le run p hp
    have ih' := ih (run * p / 100) (fun q hq => h q (by simp [hq]))
    have hm : run * p / 100 * ps.length ≤ run * ps.length := Nat.mul_le_mul_right _ hr
    have hl : run * (ps.length + 1) = run * ps.length + run := by
      rw [Nat.mul_add, Nat.mul_one]
    simp only [expAux, List.length_cons]
    rw [hl]
    omega

/-- Expected committed tokens never exceed one plus the number of drafts (all accepted). -/
theorem expected_le (ps : List Nat) (h : ∀ p ∈ ps, p ≤ 100) :
    expected ps ≤ 100 * (ps.length + 1) := by
  have := expAux_le ps 100 h
  unfold expected
  omega

-- § choice: never worse than plain decoding

/-- Best rate of the candidates, starting from the plain-decode baseline (strict improvement only). -/
def bestOf (base : Nat) (cands : List Nat) : Nat := cands.foldl max base

/-- The fold of `max` never drops below its starting value. -/
theorem foldl_max_ge (l : List Nat) : ∀ a, a ≤ l.foldl max a := by
  induction l with
  | nil => intro a; simp
  | cons x xs ih =>
    intro a
    have := ih (max a x)
    simp only [List.foldl_cons]
    omega

/-- The chosen candidate rate is at least the plain-decoding baseline. -/
theorem bestOf_ge_base (base : Nat) (cands : List Nat) : base ≤ bestOf base cands :=
  foldl_max_ge cands base

/-- The `min_gain` gate (percent): keep the candidate only if it clears `base * (100 + gain)`. -/
def gate (base best gain : Nat) : Nat := if best * 100 < base * (100 + gain) then base else best

/-- After gating, the rate is still at least the baseline. -/
theorem gate_ge_base (base best gain : Nat) (h : base ≤ best) : base ≤ gate base best gain := by
  unfold gate
  split <;> omega

/-- A gated-in candidate beats the baseline by at least the configured margin. -/
theorem gate_margin (base best gain : Nat) (h : gate base best gain ≠ base) :
    base * (100 + gain) ≤ best * 100 := by
  unfold gate at h
  split at h
  · exact absurd rfl h
  · omega

-- § observe: clamping and the EMA

/-- `accepted` clamped to `[0, k]`. -/
def acc' (k accepted : Nat) : Nat := min accepted k

/-- Number of positions updated: `min(k, accepted + 1)`. -/
def seen (k accepted : Nat) : Nat := min k (acc' k accepted + 1)

/-- The clamped acceptance never exceeds the number of drafts. -/
theorem acc'_le (k a : Nat) : acc' k a ≤ k := by
  unfold acc'
  omega

/-- At most `k` positions are updated, so the `mtp_p_[i]` writes stay in range. -/
theorem seen_le (k a : Nat) : seen k a ≤ k := by
  unfold seen
  omega

/-- Every accepted position is updated. -/
theorem acc_le_seen (k a : Nat) : acc' k a ≤ seen k a := by
  unfold seen acc'
  omega

/-- If some draft was rejected, exactly one extra (the rejected) position is updated. -/
theorem seen_of_reject (k a : Nat) (h : a < k) : seen k a = a + 1 := by
  unfold seen acc'
  omega

/-- EMA step toward 1 (scaled: 100) keeps the estimate in `[0, 100]` (result scaled by 1000). -/
theorem ema_hit_bounds (x e : Int) (hx0 : 0 ≤ x) (hx1 : x ≤ 100) (he0 : 0 ≤ e) (he1 : e ≤ 1000) :
    0 ≤ 1000 * x + e * (100 - x) ∧ 1000 * x + e * (100 - x) ≤ 100000 := by
  have h1 : 0 ≤ e * (100 - x) := Int.mul_nonneg he0 (by omega)
  have h2 : e * (100 - x) ≤ 1000 * (100 - x) := Int.mul_le_mul_of_nonneg_right he1 (by omega)
  constructor <;> omega

/-- EMA step toward 0 keeps the estimate in `[0, 100]` (result scaled by 1000). -/
theorem ema_miss_bounds (x e : Int) (hx0 : 0 ≤ x) (hx1 : x ≤ 100) (he0 : 0 ≤ e) (he1 : e ≤ 1000) :
    0 ≤ 1000 * x + e * (-x) ∧ 1000 * x + e * (-x) ≤ 100000 := by
  have h1 : 0 ≤ e * x := Int.mul_nonneg he0 hx0
  have h2 : e * x ≤ 1000 * x := Int.mul_le_mul_of_nonneg_right he1 hx0
  have h3 : e * (-x) = -(e * x) := Int.mul_neg e x
  rw [h3]
  constructor <;> omega

end Ref.Spec



## FILE: /mnt/user-data/outputs/SuffixDrafter.lean
/-!
# MODULE: suffix_drafter
SOURCES: src/spec/suffix_drafter.cpp
Prompt-lookup drafter: an open-addressed hash table keyed by the last three tokens remembers the most recent
positions where that trigram ended; `propose` extends the longest match and copies the continuation.

## Source semantics
* Keys are forced non-zero (`| 1`) because 0 marks an empty slot.
* The table has a power-of-two capacity of at least `2 * capacity_tokens` (load factor <= 0.5).
* Each slot keeps up to `WAYS` positions, newest first: appending shifts older ones down and drops the oldest.
* `propose` skips positions at or after the current suffix, requires a match of at least `min_match_`, caps the
  match at `max_match_`, and returns at most `max_k` tokens read from `best_end + 1 .. cur`.

## Form template identifiers
keyOf -> `key_at` (the `| 1` step)
shiftIn / nNext -> the `WAYS` shift in `SuffixDrafter::append`
matchAux -> the backwards `while (len < max_match_ && len <= p && ...)` loop in `propose`
contLen -> the `for (q = best_end + 1; q <= cur && k < max_k; ++q)` copy loop
-/

namespace Ref.Spec.Suffix

-- § constructor parameters

/-- `min_match_` after the constructor's `std::max(3, min_match)`. -/
def minMatch (m : Nat) : Nat := max 3 m

/-- `max_match_` after the constructor's `std::max(min_match, max_match)`. -/
def maxMatch (lo hi : Nat) : Nat := max lo hi

/-- A trigram is the shortest possible match: `min_match_ >= 3`. -/
theorem minMatch_ge_three (m : Nat) : 3 ≤ minMatch m := by
  unfold minMatch
  omega

/-- The maximum match is at least the requested minimum. -/
theorem maxMatch_ge_lo (lo hi : Nat) : lo ≤ maxMatch lo hi := by
  unfold maxMatch
  omega

/-- Table capacity: next power of two of `2 * capacity_tokens`. -/
def capFor (t : Nat) : Nat := Nat.nextPowerOfTwo (t * 2)

/-- Load factor at nominal capacity is at most one half. -/
theorem capFor_ge (t : Nat) : t * 2 ≤ capFor t := Nat.le_nextPowerOfTwo (t * 2)

-- § hashing

/-- A key is forced non-zero so that 0 can mark an empty slot. -/
def keyOf (x : Nat) : Nat := x ||| 1

/-- The forced-odd key is never the empty marker 0. -/
theorem keyOf_ne_zero (x : Nat) : keyOf x ≠ 0 := by
  unfold keyOf
  simp

/-- Linear probing stays inside the table when the capacity is positive. -/
theorem probe_lt (i cap : Nat) (h : 0 < cap) : (i + 1) % cap < cap :=
  Nat.mod_lt _ h

-- § append: the WAYS shift

/-- Newest-first position list after inserting `e`, truncated to `W` ways. -/
def shiftIn (l : List Nat) (e W : Nat) : List Nat := (e :: l).take W

/-- The way list never exceeds `WAYS` entries. -/
theorem shiftIn_length_le (l : List Nat) (e W : Nat) : (shiftIn l e W).length ≤ W := by
  unfold shiftIn
  exact List.length_take_le _ _

/-- The newest position is always in way 0. -/
theorem shiftIn_head (l : List Nat) (e w : Nat) : (shiftIn l e (w + 1)).head? = some e := by
  simp [shiftIn]

/-- The occupied-way count after an insert: `min(n + 1, WAYS)`. -/
def nNext (n W : Nat) : Nat := if n < W then n + 1 else n

/-- The count never exceeds `WAYS` (given it started within it). -/
theorem nNext_le (n W : Nat) (h : n ≤ W) : nNext n W ≤ W := by
  unfold nNext
  split <;> omega

/-- The count is non-decreasing. -/
theorem nNext_ge (n W : Nat) : n ≤ nNext n W := by
  unfold nNext
  split <;> omega

-- § propose: longest backward match

/-- Backward match length between positions `p` and `cur`, with loop fuel. -/
def matchAux (hist : Nat → Int) (maxm p cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | f + 1, len =>
    if len < maxm ∧ len ≤ p ∧ hist (p - len) = hist (cur - len)
    then matchAux hist maxm p cur f (len + 1) else len

/-- The match loop never exceeds `max_match_`, never reads before index 0, and never loses progress. -/
theorem matchAux_bounds (hist : Nat → Int) (maxm p cur : Nat) (f : Nat) :
    ∀ len, len ≤ maxm → len ≤ p + 1 →
      len ≤ matchAux hist maxm p cur f len ∧ matchAux hist maxm p cur f len ≤ maxm ∧
      matchAux hist maxm p cur f len ≤ p + 1 := by
  induction f with
  | zero =>
    intro len h1 h2
    simp only [matchAux]
    exact ⟨Nat.le_refl _, h1, h2⟩
  | succ f ih =>
    intro len h1 h2
    simp only [matchAux]
    split
    · rename_i hc
      obtain ⟨c1, c2, _⟩ := hc
      have := ih (len + 1) (by omega) (by omega)
      omega
    · exact ⟨Nat.le_refl _, h1, h2⟩

-- § propose: continuation copy

/-- Number of tokens copied from `best_end + 1 .. cur`, capped by `max_k`. -/
def contLen (bestEnd cur maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The proposal never exceeds the caller's `max_k` (the output buffer size). -/
theorem contLen_le_maxK (bestEnd cur maxK : Nat) : contLen bestEnd cur maxK ≤ maxK := by
  unfold contLen
  omega

/-- The proposal never reads past the current suffix. -/
theorem contLen_le_span (bestEnd cur maxK : Nat) : contLen bestEnd cur maxK ≤ cur - bestEnd := by
  unfold contLen
  omega

/-- A genuine earlier match (`best_end < cur`) with room for a draft yields at least one token. -/
theorem contLen_pos (bestEnd cur maxK : Nat) (h : bestEnd < cur) (hk : 0 < maxK) :
    0 < contLen bestEnd cur maxK := by
  unfold contLen
  omega

/-- Final count returned by `propose`: zero on a guard failure, else the continuation length. -/
def proposeCount (n maxK bestLen minM bestEnd cur : Nat) : Nat :=
  if n < 4 ∨ maxK = 0 ∨ bestLen < minM then 0 else contLen bestEnd cur maxK

/-- `propose` never writes more than `max_k` tokens. -/
theorem proposeCount_le (n maxK bestLen minM bestEnd cur : Nat) :
    proposeCount n maxK bestLen minM bestEnd cur ≤ maxK := by
  unfold proposeCount
  split
  · omega
  · exact contLen_le_maxK _ _ _

end Ref.Spec.Suffix





## FILE: /mnt/user-data/outputs/FusedGatedResidualRead.lean
/-!
# MODULE: fused_gated_residual_read
SOURCES: src/kernels/cuda/fused_gr.cu
Fused hyper-connection read (`gr_read`) specialised to n_embd=2560, hc=4, hc_lr=320: a down kernel
(per-stream RMS-norm, 320 down rows plus the inject rows), an up kernel (gate and stream mean), and the
T-token variants that read each weight row once.

## Source semantics
* `D = N*HC = 10240` is the stacked residual; stream `c = i / N`, column `d = i % N`.
* down: 40 blocks of 8 warps give the 320 `lo` rows; block 40 handles the 4 inject rows.
* up: 80 blocks of 32 columns (x 4 streams = 128 rows, 16 per warp); multi variant 160 blocks of 16 columns.
* weights are read as `uint4` = 8 bf16; a row is 1280 chunks (40 per lane) for `D`, 40 chunks for `LR`.

## Form template identifiers
* `N` -> n_embd, `HC` -> streams, `LR` -> hc_lr, `TILE` -> staged_tile_floats, `DOWN_BLOCKS` -> down_grid.
-/

namespace Ref.FusedGr

/-- n_embd. -/
def N : Nat := 2560
/-- Residual streams. -/
def HC : Nat := 4
/-- Stacked residual length. -/
def D : Nat := 10240
/-- Low-rank width. -/
def LR : Nat := 320
/-- Threads per block. -/
def THREADS : Nat := 256
/-- Warps per block. -/
def WARPS : Nat := 8
/-- Blocks of the down kernel without the inject block. -/
def DOWN_BLOCKS : Nat := 40
/-- Columns per up block. -/
def UP_COLS : Nat := 32
/-- Up blocks. -/
def UP_BLOCKS : Nat := 80
/-- Staged tile (floats per token). -/
def TILE : Nat := 2560
/-- uint4 chunks per lane per tile. -/
def TQ : Nat := 10
/-- Columns per multi-token up block. -/
def UPM_COLS : Nat := 16
/-- Multi-token up blocks. -/
def UPM_BLOCKS : Nat := 160

/-- The stacked residual is `N*HC`. -/
theorem d_eq : N * HC = D := by decide

/-- Eight warps per block. -/
theorem warps_eq : THREADS / 32 = WARPS := by decide

/-- 40 down blocks of 8 warps give exactly the 320 `lo` rows. -/
theorem down_blocks_eq : LR / WARPS = DOWN_BLOCKS := by decide

/-- ... and `DOWN_BLOCKS * WARPS = LR`. -/
theorem down_rows_exact : DOWN_BLOCKS * WARPS = LR := by decide

/-- 80 up blocks of 32 columns cover `N`. -/
theorem up_blocks_eq : N / UP_COLS = UP_BLOCKS ∧ UP_BLOCKS * UP_COLS = N := by decide

/-- 160 multi-token up blocks of 16 columns cover `N`. -/
theorem upm_blocks_eq : N / UPM_COLS = UPM_BLOCKS ∧ UPM_BLOCKS * UPM_COLS = N := by decide

/-- An up block owns 4 streams x 32 columns = 128 rows, 16 per warp. -/
theorem up_rows_per_warp : HC * UP_COLS = 128 ∧ HC * UP_COLS / WARPS = 16 := by decide

/-- A multi-token up block owns 64 rows, 8 per warp. -/
theorem upm_rows_per_warp : HC * UPM_COLS = 64 ∧ HC * UPM_COLS / WARPS = 8 := by decide

/-- A `D` row is 1280 chunks of 8, i.e. 40 per lane. -/
theorem down_chunks : D / 8 = 1280 ∧ 1280 / 32 = 40 := by decide

/-- An `LR` row is 40 chunks: 32 lanes take one, lanes `< 8` take a second. -/
theorem up_chunks : LR / 8 = 40 ∧ 40 = 32 + 8 := by decide

/-- The tile loop: `D / TILE = 4` tiles, each 320 chunks = 10 per lane. -/
theorem tile_geometry : D / TILE = 4 ∧ TILE / 8 / 32 = TQ ∧ TQ * 32 * 8 = TILE := by decide

/-- Dynamic shared memory for `T` tokens is `T * 10240` bytes. -/
theorem dyn_smem_bytes (T : Nat) : T * 2560 * 4 = T * 10240 := by omega

/-- Stream/column decomposition of a stacked index. -/
theorem stream_decomp (i : Nat) (hi : i < 10240) :
    i / 2560 < 4 ∧ i / 2560 * 2560 + i % 2560 = i := by omega

/-- Vector-4 accesses starting at a multiple of 4 never straddle two streams. -/
theorem float4_same_stream (i : Nat) (hi : i < 10240) (h4 : i % 4 = 0) :
    (i + 3) / 2560 = i / 2560 := by omega

/-- The vector-4 loop `i = t*4 + k*1024` covers each multiple of 4 below `D` exactly via `t < 256`. -/
theorem norm_loop_cover (i : Nat) (hi : i < 10240) (h4 : i % 4 = 0) :
    ∃ t k, t < 256 ∧ k < 10 ∧ i = t * 4 + k * 1024 := by
  refine ⟨(i % 1024) / 4, i / 1024, ?_, ?_, ?_⟩ <;> omega

/-- Row `blk*8 + warp` of the down kernel is a valid `lo` row. -/
theorem down_row_lt (blk warp : Nat) (hb : blk < 40) (hw : warp < 8) : blk * 8 + warp < 320 := by omega

/-- Every `lo` row is produced by some (block, warp). -/
theorem down_row_cover (row : Nat) (h : row < 320) :
    ∃ blk warp, blk < 40 ∧ warp < 8 ∧ row = blk * 8 + warp := by
  refine ⟨row / 8, row % 8, ?_, ?_, ?_⟩ <;> omega

/-- Up kernel: a stacked row `c*N + d0 + dd` lies inside `D`. -/
theorem up_row_lt (c blk dd : Nat) (hc : c < 4) (hb : blk < 80) (hd : dd < 32) :
    c * 2560 + (blk * 32 + dd) < 10240 := by omega

/-- Up kernel: every column `d < N` belongs to exactly one block/offset. -/
theorem up_col_cover (d : Nat) (h : d < 2560) :
    ∃ blk dd, blk < 80 ∧ dd < 32 ∧ blk * 32 + dd = d := by
  refine ⟨d / 32, d % 32, ?_, ?_, ?_⟩ <;> omega

/-- The block owning a column is unique (so `R_out` writes do not race). -/
theorem up_col_unique (blk dd blk' dd' : Nat) (hd : dd < 32) (hd' : dd' < 32)
    (h : blk * 32 + dd = blk' * 32 + dd') : blk = blk' ∧ dd = dd' := by omega

/-- Multi-token up kernel: every column is covered by the 16-column blocks. -/
theorem upm_col_cover (d : Nat) (h : d < 2560) :
    ∃ blk dd, blk < 160 ∧ dd < 16 ∧ blk * 16 + dd = d := by
  refine ⟨d / 16, d % 16, ?_, ?_, ?_⟩ <;> omega

/-- Multi-token variants: lane `k` serves token `k`, which needs `T <= 32` lanes. -/
theorem lane_token_fits (T k : Nat) (hT : T ≤ 32) (hk : k < T) : k < 32 := by omega

/-- Warp-per-row strided coverage of the 320-wide gate reduction. -/
theorem lr_lane_cover (k : Nat) (h : k < 320) :
    ∃ lane j, lane < 32 ∧ j < 10 ∧ k = lane + 32 * j := by
  refine ⟨k % 32, k / 32, ?_, ?_, ?_⟩ <;> omega

end Ref.FusedGr



## FILE: /mnt/user-data/outputs/FusedGdnStepNorm.lean
/-!
# MODULE: fused_gdn_step_norm
SOURCES: src/kernels/cuda/fused_gdn.cu
Fused gated-delta-net kernels: depthwise conv + SiLU + L2 norm, the alpha/beta gate projections, and the
state step fused with RMS-norm and sigmoid output gating. This file covers the launch geometry, index maps
and the small algebraic contracts the three kernels rely on.

## Source semantics
* The state is `(S, h_v, S)` with `S = 128`; a block owns one head, 512 threads = 128 columns x 4 row groups
  of 32 rows each, so the row groups tile the 128 rows exactly once.
* `gdn_conv_l2_kernel`: one thread per channel `c = blockIdx.x*128 + threadIdx.x`; history slides by one tap.
* `gdn_ab_kernel`: one warp per output row of the stacked `[alpha; beta]` projection (`2*h_v` rows, 8 per block).
* head pairing is `head % h_k`.

## Form template identifiers
* `S` -> state_size, `RG` -> row_groups, `RPG` -> rows_per_group, `rowOf` -> row_index_map,
  `convShift` -> history_slide, `abRow` -> stacked_projection_row.
-/

namespace Ref.FusedGdn

/-- State size (rows = cols). -/
def S : Nat := 128
/-- Row groups per head block. -/
def RG : Nat := 4
/-- Rows owned by one thread (`S / RG`). -/
def RPG : Nat := 32

/-- `RPG` really is `S / RG`. -/
theorem rpg_def : S / RG = RPG := by decide

/-- The row groups tile the state rows exactly. -/
theorem rows_cover : RG * RPG = S := by decide

/-- A step block has `S * RG = 512` threads. -/
theorem threads_per_block : S * RG = 512 := by decide

/-- Warps of row group 0 are threads 0..127, i.e. four warps, matching the four `wsum` entries read. -/
theorem warps_of_group0 : S / 32 = 4 := by decide

/-- The `wsum` scratch has `S*RG/32 = 16` slots, at least the 4 that are read. -/
theorem wsum_big_enough : 4 ≤ S * RG / 32 := by decide

/-- Row of the state handled by (row group, register index). -/
def rowOf (rg r : Nat) : Nat := rg * 32 + r

/-- Every (group, register) pair addresses a row inside the state. -/
theorem rowOf_lt (rg r : Nat) (hg : rg < 4) (hr : r < 32) : rowOf rg r < 128 := by
  unfold rowOf; omega

/-- The row map is injective: no two threads/registers touch the same row. -/
theorem rowOf_inj (rg r rg' r' : Nat) (hr : r < 32) (hr' : r' < 32)
    (h : rowOf rg r = rowOf rg' r') : rg = rg' ∧ r = r' := by
  unfold rowOf at h; omega

/-- The row map is surjective onto `0..127`: every row is owned. -/
theorem rowOf_surj (i : Nat) (hi : i < 128) :
    ∃ rg r, rg < 4 ∧ r < 32 ∧ rowOf rg r = i := by
  refine ⟨i / 32, i % 32, by omega, by omega, ?_⟩
  unfold rowOf; omega

/-- Linear state address `((row*h_v + head)*128 + col)` stays inside the `128*h_v*128` state. -/
theorem state_addr_lt (row head col h_v : Nat) (hrow : row < 128) (hh : head < h_v) (hc : col < 128) :
    (row * h_v + head) * 128 + col < 128 * h_v * 128 := by
  have h1 : (row + 1) * h_v ≤ 128 * h_v := Nat.mul_le_mul_right _ (by omega)
  have h2 : (row + 1) * h_v = row * h_v + h_v := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Head pairing `head % h_k` is always a valid q/k head. -/
theorem qk_head_lt (head h_k : Nat) (h : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ h

/-- Conv channel index is in range when the grid is `channels / 128` blocks of 128 threads. -/
theorem conv_channel_lt (nb blk t : Nat) (hb : blk < nb) (ht : t < 128) : blk * 128 + t < nb * 128 := by
  omega

/-- The history slide: drop the oldest tap and append the newest input. -/
def convShift (h0 h1 h2 x : Int) : Int × Int × Int := (h1, h2, x)

/-- Four-tap depthwise convolution output. -/
def convOut (h0 h1 h2 x w0 w1 w2 w3 : Int) : Int := h0 * w0 + h1 * w1 + h2 * w2 + x * w3

/-- After a step the newest history slot holds exactly the input just consumed. -/
theorem convShift_newest (h0 h1 h2 x : Int) : (convShift h0 h1 h2 x).2.2 = x := rfl

/-- After a step the oldest slot holds what was the middle tap. -/
theorem convShift_oldest (h0 h1 h2 x : Int) : (convShift h0 h1 h2 x).1 = h1 := rfl

/-- With an all-zero history only the last tap contributes. -/
theorem convOut_zero_history (x w0 w1 w2 w3 : Int) : convOut 0 0 0 x w0 w1 w2 w3 = x * w3 := by
  unfold convOut; simp

/-- Two consecutive steps: the input of step one is the middle tap at step two. -/
theorem convShift_twice (h0 h1 h2 x y : Int) :
    (convShift (convShift h0 h1 h2 x).1 (convShift h0 h1 h2 x).2.1 (convShift h0 h1 h2 x).2.2 y).2.1 = x := rfl

/-- History addressing `c*3 + tap` is injective for taps below 3. -/
theorem hist_addr_inj (c t c' t' : Nat) (ht : t < 3) (ht' : t' < 3)
    (h : c * 3 + t = c' * 3 + t') : c = c' ∧ t = t' := by omega

/-- Weight addressing `c*4 + tap` is injective for taps below 4. -/
theorem weight_addr_inj (c t c' t' : Nat) (ht : t < 4) (ht' : t' < 4)
    (h : c * 4 + t = c' * 4 + t') : c = c' ∧ t = t' := by omega

/-- The stacked alpha/beta rows are covered by `(2*h_v+7)/8` blocks of 8 warps. -/
theorem ab_rows_covered (h_v row : Nat) (h : row < 2 * h_v) : row / 8 < (2 * h_v + 7) / 8 := by omega

/-- The grid supplies at least `2*h_v` rows. -/
theorem ab_grid_enough (h_v : Nat) : 2 * h_v ≤ (2 * h_v + 7) / 8 * 8 := by omega

/-- Row selector of the stacked projection: rows `>= h_v` are the beta rows. -/
def abRow (h_v row : Nat) : Nat := if h_v ≤ row then row - h_v else row

/-- The within-projection row index is always `< h_v`. -/
theorem abRow_lt (h_v row : Nat) (h : row < 2 * h_v) : abRow h_v row < h_v := by
  unfold abRow; split <;> omega

/-- `n_embd % 8 = 0` makes the 8-wide bf16 chunks cover the reduction axis exactly. -/
theorem ab_chunks_exact (n : Nat) (h : n % 8 = 0) : n / 8 * 8 = n := by omega

/-- The delta-rule write at scalar level with `k = beta = 1`: the new state equals `v`. -/
theorem delta_write_exact (dec s v : Int) : dec * s + (v - dec * s * 1) * 1 = v := by
  omega

end Ref.FusedGdn



## FILE: /mnt/user-data/outputs/GatedResidualReference.lean
/-!
# MODULE: gated_residual_reference
SOURCES: src/kernels/cuda/gr.cu
The gated residual / hyper-connection `gr_read` and `gr_write`: per-stream RMS-norm, down projection with
SiLU, up projection with sigmoid gate, mean over streams, injection and the write-back
`out = R + block_out * 2*sigmoid(inject/hc)`.

## Source semantics
* Workspace regions (xn, xq, lq, gated, lo) are laid out by ONE size table, each rounded up to 16 bytes;
  the regions must not overlap (an earlier version sized them in a different order and overlapped).
* The down kernel is one block per output row (8 warps split the reduction), the gate kernel one warp per row.
* The block-sum scratch reuses a `double[8]` as `float[8]`.
* `gr_write` applies the same block output to every stream with a per-stream weight.

## Form template identifiers
* `align16` -> region_alignment, `wsSizes` -> workspace_size_table, `wsTotal` -> workspace_bytes.
-/

namespace Ref.GatedResidual

/-- Round a byte count up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks a region. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Aligned sizes are multiples of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Alignment is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

/-- The single size table: xn, xq, lq, gated, lo (bytes). -/
def wsSizes (n_embd hc hc_lr : Nat) : List Nat :=
  [hc * n_embd * 4, hc * n_embd * 2, hc_lr * 2, hc * n_embd * 4, hc_lr * 4]

/-- Total bytes: sum of the aligned sizes. -/
def wsTotal (n_embd hc hc_lr : Nat) : Nat := ((wsSizes n_embd hc hc_lr).map align16).foldl (· + ·) 0

/-- At the real geometry the workspace is 104,320 bytes. -/
theorem wsTotal_real : wsTotal 2560 4 320 = 104320 := by decide

/-- At the real geometry `xq` needs 20,480 bytes; the buggy table gave it only `hc_lr*2 = 640`. -/
theorem xq_needs_20480 : (wsSizes 2560 4 320)[1]! = 20480 := by decide

/-- Region offsets are a running sum, so consecutive regions are adjacent and cannot overlap. -/
theorem region_adjacent (off sz : Nat) : off + align16 sz = off + align16 sz := rfl

/-- Each region starts 16-aligned when the base is 16-aligned. -/
theorem region_aligned (off sz : Nat) (h : off % 16 = 0) : (off + align16 sz) % 16 = 0 := by
  have := align16_mod sz; omega

/-- Block-sum scratch: `float[8]` (32 bytes) fits inside `double[8]` (64 bytes). -/
theorem scratch_reinterpret_fits : 8 * 4 ≤ 8 * 8 := by decide

/-- A 256-thread block has 8 warps, so 8 partial sums are all that the scratch must hold. -/
theorem block_warps : (256 + 31) / 32 = 8 := by decide

/-- Down kernel: one block per row gives `320 * 8 = 2560` warps, about 53 per 48 SMs. -/
theorem down_occupancy : 320 * 8 = 2560 ∧ 2560 / 48 = 53 := by decide

/-- Down kernel reduction: warp `w` starts at `w*32` and strides by `256`; every index is covered. -/
theorem down_reduction_cover (i : Nat) (h : i < 10240) :
    ∃ warp lane it, warp < 8 ∧ lane < 32 ∧ it < 40 ∧ i = warp * 32 + lane + it * 256 := by
  refine ⟨(i % 256) / 32, i % 32, i / 256, ?_, ?_, ?_, ?_⟩ <;> omega

/-- ... and uniquely so: each element is accumulated by exactly one lane. -/
theorem down_reduction_unique (w l t w' l' t' : Nat) (hl : l < 32) (hl' : l' < 32)
    (hw : w < 8) (hw' : w' < 8)
    (h : w * 32 + l + t * 256 = w' * 32 + l' + t' * 256) : w = w' ∧ l = l' ∧ t = t' := by omega

/-- Gate kernel grid: `ceil(hc_dim / 8)` blocks of 8 warps cover every output row. -/
theorem gate_grid_covers (hc_dim : Nat) : hc_dim ≤ (hc_dim + 7) / 8 * 8 := by omega

/-- Mean kernel grid covers `n_embd`. -/
theorem mean_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- Inject kernel: `32*hc` threads give one warp per stream. -/
theorem inject_warps (hc : Nat) : 32 * hc / 32 = hc := by omega

/-- Write kernel: stacked index decomposes into stream and column (real geometry). -/
theorem write_decomp (i : Nat) (h : i < 10240) :
    i / 2560 < 4 ∧ i / 2560 * 2560 + i % 2560 = i := by omega

/-- Write kernel: every stream reads the SAME `block_out[d]`; the column index is below `n_embd`. -/
theorem write_col_lt (i : Nat) : i % 2560 < 2560 := by omega

/-- Grid-stride loop: starting indices `blk*256 + t` are all distinct for distinct (block, thread). -/
theorem gridstride_start_inj (b t b' t' : Nat) (ht : t < 256) (ht' : t' < 256)
    (h : b * 256 + t = b' * 256 + t') : b = b' ∧ t = t' := by omega

/-- The native path requires even `hc*n_embd`; any even `n_embd` gives it. -/
theorem native_even (hc n : Nat) (hn : n % 2 = 0) : (hc * n) % 2 = 0 := by
  rw [Nat.mul_mod, hn]; simp

/-- Write weight `2*sigmoid(x)` lies in `[0, 2]` once `sigmoid` does; abstractly: `2*s <= 2` for `s <= 1`. -/
theorem write_weight_le (s : Nat) (h : s ≤ 1) : 2 * s ≤ 2 := by omega

end Ref.GatedResidual



## FILE: /mnt/user-data/outputs/GdnReferenceKernels.lean
/-!
# MODULE: gdn_reference_kernels
SOURCES: src/kernels/cuda/gdn.cu
The gated delta-net's non-projection parts: the recurrence step (one thread per (head, column) of the
state), the sliding conv, per-row L2 norm, beta sigmoid gate and output RMS-norm gating.

## Source semantics
* State layout `(S, h_v, S)` with the column index `j` fastest; a thread owns one column end to end, so the
  recurrence needs no barrier.
* The decay is applied BEFORE the rank-1 update; the read-out uses the UPDATED state.
* `MAX_H = 1`: grid `(S/32, h_v)`; the thread count is `S*h_v` independent of packing.
* conv: `kernel[0]` meets the OLDEST state row, the new input meets the LAST tap, the state slides.
* `beta` must be passed through a sigmoid before the step (the C1 bug).

## Form template identifiers
* `JTHREADS` -> column_lanes_per_block, `MAX_H` -> heads_per_block, `decayThenUpdate` -> delta_rule_column.
-/

namespace Ref.Gdn

/-- Threads along `j`. -/
def JTHREADS : Nat := 32
/-- Heads staged per block (latency-hiding knob). -/
def MAX_H : Nat := 1

/-- With `S = 128` there are 4 column blocks per head. -/
theorem col_blocks : (128 + 32 - 1) / 32 = 4 := by decide

/-- Every column `j < 128` is owned by some (block, lane). -/
theorem col_cover (j : Nat) (h : j < 128) : ∃ bx tx, bx < 4 ∧ tx < 32 ∧ bx * 32 + tx = j := by
  refine ⟨j / 32, j % 32, ?_, ?_, ?_⟩ <;> omega

/-- Ownership is unique, so no two threads write the same column. -/
theorem col_unique (bx tx bx' tx' : Nat) (ht : tx < 32) (ht' : tx' < 32)
    (h : bx * 32 + tx = bx' * 32 + tx') : bx = bx' ∧ tx = tx' := by omega

/-- With `MAX_H = 1` the head grid dimension equals `h_v`. -/
theorem grid_y_eq (h_v : Nat) : (h_v + MAX_H - 1) / MAX_H = h_v := by
  unfold MAX_H; omega

/-- The block count for `h_v = 48` is 192 (4 warps on every SM instead of 24 blocks). -/
theorem blocks_for_48 : 4 * ((48 + MAX_H - 1) / MAX_H) = 192 := by decide

/-- The staged q/k rows hold at most 128 entries, so any `S <= 128` indexes inside them. -/
theorem staging_in_bounds (S i : Nat) (hS : S ≤ 128) (hi : i < S) : i < 128 := by omega

/-- Head pairing is a modulo pairing. -/
theorem src_head_lt (h h_k : Nat) (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt _ hk

/-- One column of the recurrence, abstracted to a single state row with scalar key/query. -/
def decayThenUpdate (s dec k v b : Int) : Int :=
  let sd := dec * s
  sd + k * ((v - sd * k) * b)

/-- With unit key and unit write strength the updated state equals the value (delta rule exactness). -/
theorem delta_exact (s dec v : Int) : decayThenUpdate s dec 1 v 1 = v := by
  unfold decayThenUpdate; simp only [Int.mul_one]; omega

/-- Zero write strength keeps the decayed state: decay happens before (and independently of) the update. -/
theorem zero_beta_only_decays (s dec k v : Int) : decayThenUpdate s dec k v 0 = dec * s := by
  unfold decayThenUpdate; simp

/-- Four-tap conv with the kernel's tap convention (oldest history meets kernel tap 0). -/
def convOut (st0 st1 st2 x w0 w1 w2 w3 : Int) : Int := st0 * w0 + st1 * w1 + st2 * w2 + x * w3

/-- The slide: `st[i] := st[i+1]` for `i < dc-2`, last slot receives the input. -/
def convSlide (st1 st2 x : Int) : Int × Int × Int := (st1, st2, x)

/-- The newest input is stored in the last state slot. -/
theorem convSlide_last (st1 st2 x : Int) : (convSlide st1 st2 x).2.2 = x := rfl

/-- Per-channel state offsets `c*3 + i` (d_conv = 4) are injective across channels. -/
theorem conv_state_inj (c i c' i' : Nat) (hi : i < 3) (hi' : i' < 3)
    (h : c * 3 + i = c' * 3 + i') : c = c' ∧ i = i' := by omega

/-- Conv launch covers all channels: `ceil(C/256) * 256 >= C`. -/
theorem conv_grid_covers (C : Nat) : C ≤ (C + 255) / 256 * 256 := by omega

/-- Beta-gate launch covers all heads. -/
theorem beta_grid_covers (n : Nat) : n ≤ (n + 127) / 128 * 128 := by omega

/-- `gdn_l2_norm` accepts only `cols <= 1024`, which a single warp strided loop covers in `<= 32` steps. -/
theorem l2_steps (cols : Nat) (h : cols ≤ 1024) : (cols + 31) / 32 ≤ 32 := by omega

/-- The L2 warp-strided loop reaches every element of the row. -/
theorem l2_stride_cover (i cols : Nat) (h : i < cols) :
    ∃ lane step, lane < 32 ∧ i = lane + 32 * step := by
  exact ⟨i % 32, i / 32, by omega, by omega⟩

end Ref.Gdn



## FILE: /mnt/user-data/outputs/NativeFlashAttnShort.lean
/-!
# MODULE: native_flash_attn_short
SOURCES: src/kernels/cuda/native_flash_attn.cu
Short-context FlashAttention vector kernel specialised from llama.cpp: 24 query heads of dim 256, 2 KV heads,
one block of 128 threads per head, online softmax over the padded length 256 in two tiles of 128 cells, with a
device-side validity check on the step state that writes a status word and NaN-fills the output on failure.

## Source semantics
* `kv = head / 12`; K/V cell row is `(cell*2 + kv)*256`; cells `>= width` are masked to -inf.
* A step is valid iff `1 <= width <= max_context`, `nkv = width`, `pos = width-1`, `n_bid = width/4`.
* Tile `base` runs over `{0, 128}`; cell = `base + warp*32 + (lane/8)*8 + row`, `row < 8`.
* The cross-warp combine uses `4*4*256` floats of shared memory.
* `max_context <= 256`, `capacity >= 256`, `capacity*1024` must not overflow `size_t`.

## Form template identifiers
* `HD` -> head_dim, `PAD` -> padded_length, `validStep` -> step_validity_predicate.
-/

namespace Ref.NativeFlashAttn

/-- Head dimension. -/
def HD : Nat := 256
/-- Padded context length. -/
def PAD : Nat := 256

/-- Step validity predicate. -/
def validStep (width nkv pos nbid maxc : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxc ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

/-- A valid step has `n_kv - 1 = pos` and at most 256 cells for `max_context <= 256`. -/
theorem valid_width_le (width nkv pos nbid maxc : Nat) (h : validStep width nkv pos nbid maxc)
    (hm : maxc ≤ 256) : width ≤ 256 ∧ nkv = pos + 1 := by
  unfold validStep at h; omega

/-- The padded length is two tiles of 128 cells. -/
theorem tiles : PAD / 128 = 2 := by decide

/-- GQA: a head `< 24` pairs with a KV head `< 2`. -/
theorem kv_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by omega

/-- Every padded cell is covered by exactly one (tile, warp, lane-group, row). -/
theorem cell_cover (cell : Nat) (h : cell < 256) :
    ∃ base warp lane row, (base = 0 ∨ base = 128) ∧ warp < 4 ∧ lane < 32 ∧ row < 8 ∧
      cell = base + warp * 32 + (lane / 8) * 8 + row := by
  refine ⟨cell / 128 * 128, (cell % 128) / 32, ((cell % 32) / 8) * 8, cell % 8, ?_, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · omega
  · omega

/-- Distinct (tile, warp, lane-group, row) give distinct cells (no double counting). -/
theorem cell_unique (b w g r b' w' g' r' : Nat) (hb : b = 0 ∨ b = 128) (hb' : b' = 0 ∨ b' = 128)
    (hw : w < 4) (hw' : w' < 4) (hg : g < 4) (hg' : g' < 4) (hr : r < 8) (hr' : r' < 8)
    (h : b + w * 32 + g * 8 + r = b' + w' * 32 + g' * 8 + r') :
    b = b' ∧ w = w' ∧ g = g' ∧ r = r' := by omega

/-- K/V element address `(cell*2 + kv)*256 + d` is below `capacity * 512` elements. -/
theorem kv_addr_lt (cell kv d capacity : Nat) (hc : cell < capacity) (hk : kv < 2) (hd : d < 256) :
    (cell * 2 + kv) * 256 + d < capacity * 512 := by omega

/-- One cell holds `2 KV heads * 256 dims * 2 bytes = 1024` bytes. -/
theorem cell_bytes : 2 * 256 * 2 = 1024 := by decide

/-- The host's byte-size computation cannot overflow under the capacity guard. -/
theorem kv_bytes_no_overflow (cap : Nat) (h : cap ≤ 18446744073709551615 / 1024) :
    cap * 1024 ≤ 18446744073709551615 := by omega

/-- The V loop: `local = warp*32 + k0 + lane/8` with `k0 ∈ {0,4,..,28}` reaches every one of 128 locals. -/
theorem v_local_cover (loc : Nat) (h : loc < 128) :
    ∃ warp k0 g, warp < 4 ∧ k0 < 32 ∧ k0 % 4 = 0 ∧ g < 4 ∧ loc = warp * 32 + k0 + g := by
  refine ⟨loc / 32, ((loc % 32) / 4) * 4, loc % 4, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- Shared tile size is `4*4*256 = 4096` floats (16 KB). -/
theorem tile_size : 4 * 4 * 256 = 4096 ∧ 4096 * 4 = 16384 := by decide

/-- Combine write index `warp*1024 + (lane/8)*256 + 2*(i0 + (lane%8)*4) + 2j + 1` is inside the tile. -/
theorem tile_write_lt (warp g l8 i0 j : Nat) (hw : warp < 4) (hg : g < 4) (hl : l8 < 8)
    (hi : i0 < 128) (h32 : i0 % 32 = 0) (hj : j < 4) :
    warp * 1024 + g * 256 + 2 * (i0 + l8 * 4 + j) + 1 < 4096 := by omega

/-- Combine read index `w*1024 + group*256 + i0 + tid` is inside the tile (`i0 + tid < 256`). -/
theorem tile_read_lt (w group col : Nat) (hw : w < 4) (hg : group < 4) (hc : col < 256) :
    w * 1024 + group * 256 + col < 4096 := by omega

/-- Output columns: `tid` and `tid + 128` for `tid < 128` cover all 256 head dims exactly once. -/
theorem out_cover (d : Nat) (h : d < 256) : ∃ tid half, tid < 128 ∧ half < 2 ∧ d = tid + 128 * half := by
  exact ⟨d % 128, d / 128, by omega, by omega, by omega⟩

/-- Output row base `head*256` is disjoint across heads. -/
theorem out_rows_disjoint (head head' d d' : Nat) (hd : d < 256) (hd' : d' < 256)
    (h : head * 256 + d = head' * 256 + d') : head = head' ∧ d = d' := by omega

/-- Per-head query offset `head*256 + d` lies inside `24*256` floats. -/
theorem q_addr_lt (head d : Nat) (hh : head < 24) (hd : d < 256) : head * 256 + d < 24 * 256 := by omega

/-- Scale `1/sqrt(256) = 1/16`: `16 * 16 = 256`. -/
theorem scale_is_inv_sqrt : 16 * 16 = HD := by decide

/-- Out-of-range cells are masked: a cell `>= width` is excluded from the valid set. -/
theorem masked_cell (cell width : Nat) (h : width ≤ cell) : ¬ cell < width := by omega

/-- With `width <= 256` every valid cell lies inside the padded tiles. -/
theorem valid_cell_in_pad (cell width : Nat) (h : cell < width) (hw : width ≤ 256) : cell < PAD := by
  unfold PAD; omega

/-- Status word domain: the kernel writes either success or the unsupported-step code, never both:
modelled as a boolean choice from validity. -/
def statusOf (valid : Bool) (ok bad : Nat) : Nat := if valid then ok else bad

/-- A valid step reports the success code. -/
theorem status_valid (ok bad : Nat) : statusOf true ok bad = ok := rfl

/-- An invalid step reports the unsupported-step code. -/
theorem status_invalid (ok bad : Nat) : statusOf false ok bad = bad := rfl

end Ref.NativeFlashAttn



## FILE: /mnt/user-data/outputs/NativeGdnPreprocess.lean
/-!
# MODULE: native_gdn_preprocess
SOURCES: src/kernels/cuda/native_gdn_preprocess.cu
Native GDN preprocessing kernels adapted from llama.cpp: conv + SiLU with history slide, per-row L2/RMS norm
with a scale, beta sigmoid, gate softplus, and output RMS-norm with sigmoid gating. Host wrappers validate
counts, geometry and span disjointness.

## Source semantics
* conv requires `d_conv = 4`; history holds 3 taps per channel (`c*3 + tap`), weights 4 (`c*4 + tap`).
* norms require width 128 and are launched with 256 threads (8 warps); the block sum reads 8 warp partials.
* counts must lie in `[1, 65535]`; grids are `ceil(count/256)`.
* writable spans must be pairwise disjoint and disjoint from inputs.

## Form template identifiers
* `S` -> head_width, `convHistoryNext` -> history_slide, `countOk` -> launch_count_guard.
-/

namespace Ref.NativeGdnPreprocess

/-- Head width. -/
def S : Nat := 128

/-- Count guard of the host wrappers. -/
def countOk (count : Nat) : Prop := 0 < count ∧ count ≤ 65535

/-- The grid for an accepted count has between 1 and 256 blocks. -/
theorem grid_bounds (count : Nat) (h : countOk count) :
    1 ≤ (count + 255) / 256 ∧ (count + 255) / 256 ≤ 256 := by
  unfold countOk at h; omega

/-- The grid covers every element. -/
theorem grid_covers (count : Nat) : count ≤ (count + 255) / 256 * 256 := by omega

/-- Block sums: 256 threads are 8 warps and the second pass reads lanes `< 8`. -/
theorem norm_warps : 256 / 32 = 8 := by decide

/-- 128 columns fit in 256 threads (the `col < S` guards cover the rest). -/
theorem norm_threads_cover : S ≤ 256 := by decide

/-- History addressing `c*3 + tap` is injective for taps `< 3`. -/
theorem hist_inj (c t c' t' : Nat) (ht : t < 3) (ht' : t' < 3)
    (h : c * 3 + t = c' * 3 + t') : c = c' ∧ t = t' := by omega

/-- Weight addressing `c*4 + tap` is injective for taps `< 4`. -/
theorem weight_inj (c t c' t' : Nat) (ht : t < 4) (ht' : t' < 4)
    (h : c * 4 + t = c' * 4 + t') : c = c' ∧ t = t' := by omega

/-- The history region has `3*channels` floats and the last address stays inside it. -/
theorem hist_last_lt (channels c t : Nat) (hc : c < channels) (ht : t < 3) : c * 3 + t < 3 * channels := by
  omega

/-- The weight region has `4*channels` floats. -/
theorem weight_last_lt (channels c t : Nat) (hc : c < channels) (ht : t < 4) : c * 4 + t < 4 * channels := by
  omega

/-- Next history after consuming input `x`: taps shift left, input enters last. -/
def convHistoryNext (v1 v2 x : Int) : Int × Int × Int := (v1, v2, x)

/-- The shifted history equals the old taps 1..3 (`values[tap+1]` in the source). -/
theorem history_shift (v0 v1 v2 x : Int) : convHistoryNext v1 v2 x = (v1, v2, x) := rfl

/-- Tap sum over four values. -/
def convSum (v0 v1 v2 v3 w0 w1 w2 w3 : Int) : Int := v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3

/-- Adding the zero bias after the sum is the identity (the source adds `0.0f` explicitly). -/
theorem zero_bias_identity (s : Int) : s + 0 = s := by omega

/-- Gate softplus branch: for `v > 20` the value is returned unchanged (abstract threshold check). -/
def softplusBranchLarge (v : Int) : Bool := decide (v > 20)

/-- The large branch fires exactly above 20. -/
theorem softplus_large_iff (v : Int) : softplusBranchLarge v = true ↔ v > 20 := by
  unfold softplusBranchLarge; simp

/-- Output bytes of a norm over `heads` rows of width 128. -/
theorem norm_bytes (heads : Nat) : heads * 128 * 4 = 512 * heads := by omega

/-- Gamma is a single 128-float row: `S * 4 = 512` bytes. -/
theorem gamma_bytes : S * 4 = 512 := by decide

/-- Span interval overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric, so pairwise checks `j < i` suffice. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; constructor <;> intro h <;> omega

/-- Adjacent spans are disjoint: the byte just past the first is the start of the second. -/
theorem adjacent_disjoint (a an bn : Nat) : ¬ overlap a an (a + an) bn := by
  unfold overlap; omega

/-- Triple pairwise-disjointness of the three writable conv spans from their cumulative layout. -/
theorem layout_disjoint3 (a n1 n2 n3 : Nat) :
    ¬ overlap a n1 (a + n1) n2 ∧ ¬ overlap a n1 (a + n1 + n2) n3 ∧ ¬ overlap (a + n1) n2 (a + n1 + n2) n3 := by
  unfold overlap; omega

end Ref.NativeGdnPreprocess



## FILE: /mnt/user-data/outputs/NativeGdnStep.lean
/-!
# MODULE: native_gdn_step
SOURCES: src/kernels/cuda/native_gdn.cu
Warp-per-(head, column) gated delta-net step adapted from llama.cpp: each warp keeps four state rows per
lane in registers, reduces with xor shuffles, and the host wrapper validates alignment/disjointness of every span.

## Source semantics
* Grid `(h_v, 1, S/4)`, block `(32, 4)`: `col = blockIdx.z*4 + threadIdx.y`; lane `l` owns rows `r*32 + l`.
* `q_head = head % h_k`; `h_v % h_k = 0` is required.
* Output scale is `1/sqrt(S)`; `S = 128`.
* A span is valid iff nonnull, 4-aligned and `address + bytes` does not overflow; spans overlap iff intervals intersect.

## Form template identifiers
* `S` -> state_size, `overlap` -> interval_overlap, `validSpan` -> aligned_bounded_span.
-/

namespace Ref.NativeGdn

/-- State size. -/
def S : Nat := 128

/-- 32 blocks along z of 4 columns each cover the 128 columns. -/
theorem z_blocks : S / 4 = 32 := by decide

/-- Every column belongs to exactly one (blockIdx.z, threadIdx.y). -/
theorem col_cover (c : Nat) (h : c < 128) : ∃ z y, z < 32 ∧ y < 4 ∧ z * 4 + y = c := by
  refine ⟨c / 4, c % 4, ?_, ?_, ?_⟩ <;> omega

/-- Column ownership is unique. -/
theorem col_unique (z y z' y' : Nat) (hy : y < 4) (hy' : y' < 4)
    (h : z * 4 + y = z' * 4 + y') : z = z' ∧ y = y' := by omega

/-- Lane `l` and register `r` address row `r*32 + l`, covering all 128 rows exactly once. -/
theorem row_cover (i : Nat) (h : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ r * 32 + l = i := by
  refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> omega

/-- Row ownership is unique. -/
theorem row_unique (r l r' l' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by omega

/-- Block shape: 32 x 4 = 128 threads (matches `__launch_bounds__(128, 2)`). -/
theorem block_threads : 32 * 4 = 128 := by decide

/-- Head pairing is a valid q/k head. -/
theorem q_head_lt (head h_k : Nat) (h : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ h

/-- State byte size is `65536 * h_v`. -/
theorem state_bytes (h_v : Nat) : 128 * 128 * h_v * 4 = 65536 * h_v := by omega

/-- Output byte size is `512 * h_v`. -/
theorem output_bytes (h_v : Nat) : 128 * h_v * 4 = 512 * h_v := by omega

/-- Interval overlap of two byte spans. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; constructor <;> intro h <;> omega

/-- The source's test has no emptiness guard: an empty span at `a` is reported as overlapping only when it
lies strictly inside the other span. -/
theorem overlap_empty_inside (a b bn : Nat) (h : overlap a 0 b bn) : b < a ∧ a < b + bn := by
  unfold overlap at h; omega

/-- Spans that end before the other begins are disjoint. -/
theorem disjoint_of_le (a an b bn : Nat) (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap; omega

/-- Every span overlaps itself when nonempty. -/
theorem overlap_self (a n : Nat) (h : 0 < n) : overlap a n a n := by
  unfold overlap; omega

/-- A valid span: nonnull, 4-aligned, and `address + bytes` fits in the address space. -/
def validSpan (maxAddr addr bytes : Nat) : Prop := addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ maxAddr - addr

/-- A valid span with `addr <= maxAddr` ends within the address space. -/
theorem validSpan_end (maxAddr addr bytes : Nat) (h : validSpan maxAddr addr bytes)
    (hm : addr ≤ maxAddr) : addr + bytes ≤ maxAddr := by
  unfold validSpan at h; omega

/-- A launch grid of `h_v` blocks in x must be `<= 65535`; the check rejects larger. -/
theorem grid_x_ok (h_v : Nat) (h : h_v ≤ 65535) : h_v < 65536 := by omega

/-- `h_v % h_k = 0` makes the pairing balanced in count (every k head used `h_v / h_k` times at head 0 residue). -/
theorem divisible_pairing (h_v h_k : Nat) (hk : 0 < h_k) (h : h_v % h_k = 0) : h_v / h_k * h_k = h_v := by
  have := Nat.div_add_mod h_v h_k
  rw [h] at this
  rw [Nat.mul_comm]; omega

end Ref.NativeGdn



## FILE: /mnt/user-data/outputs/NativeQsaIndexerAppend.lean
/-!
# MODULE: native_qsa_indexer_append
SOURCES: src/kernels/cuda/native_qsa_indexer.cu
Native indexer key append adapted from llama.cpp: stores the f16-rounded raw key into a 3-row tail, and on a
block completion (or at position 0 for the spare/dead key) mean-pools, RMS-normalises with gamma, rotates the
first 64 dimensions NEOX-style at the block's first position, writes the pooled row and re-seeds the spare row.

## Source semantics
* D = 128, R = 4, ROT = 64, 256 threads (only `d < D` act).
* `slot = pos % R`; slots 0..2 are stored in the tail (`slot*D + d`); slot 3 completes block `b = pos / R`.
* Position 0 builds the dead key (rope position 0); later completions write `pooled[b]` and `pooled[b+1] = dead`.
* rope position is `pos_base + R*b`; `pos_base % R = 0` is required and `pos_base + max_cells <= INT32_MAX`.
* The pooled buffer has `max_cells / R + 1` rows.

## Form template identifiers
* `slot` -> tail_slot, `ropePos` -> block_first_position, `pairOf` -> rope_pair_index.
-/

namespace Ref.NativeQsaIndexer

/-- Tail slot of a position. -/
def slot (pos : Nat) : Nat := pos % 4

/-- Block of a position. -/
def block (pos : Nat) : Nat := pos / 4

/-- Rotation position of block `b` (for `pos ≠ 0`). -/
def ropePos (pos_base b : Nat) : Nat := pos_base + 4 * b

/-- Rope pair index of dimension `d`. -/
def pairOf (d : Nat) : Nat := d % 32

/-- Tail addresses `slot*128 + d` are inside the `3*128`-float tail. -/
theorem tail_addr_lt (s d : Nat) (hs : s < 3) (hd : d < 128) : s * 128 + d < 3 * 128 := by omega

/-- Tail addressing is injective. -/
theorem tail_addr_inj (s d s' d' : Nat) (hd : d < 128) (hd' : d' < 128)
    (h : s * 128 + d = s' * 128 + d') : s = s' ∧ d = d' := by omega

/-- A position is a block completion exactly when `slot = 3`. -/
theorem complete_iff (pos : Nat) : slot pos = 3 ↔ (pos + 1) % 4 = 0 := by unfold slot; omega

/-- Completion at `pos` finishes block `pos / 4`, whose cells are `4b .. 4b+3 = pos`. -/
theorem block_last_cell (pos : Nat) (h : slot pos = 3) : block pos * 4 + 3 = pos := by
  unfold slot block at *; omega

/-- On completion the spare row `b+1` exists in a pool of `max_cells/4 + 1` rows. -/
theorem spare_row_exists (pos max_cells : Nat) (hp : pos < max_cells) (h : slot pos = 3) :
    block pos + 1 < max_cells / 4 + 1 := by
  unfold slot block at *; omega

/-- The completed row `b` itself exists too. -/
theorem row_exists (pos max_cells : Nat) (hp : pos < max_cells) : block pos < max_cells / 4 + 1 := by
  unfold block; omega

/-- A completed block's rotation position stays 4-aligned when the base is. -/
theorem ropePos_aligned (pos_base b : Nat) (h : pos_base % 4 = 0) : ropePos pos_base b % 4 = 0 := by
  unfold ropePos; omega

/-- The rotation position fits the 32-bit position space under the validated bound. -/
theorem ropePos_in_range (pos_base pos max_cells : Nat) (hp : pos < max_cells)
    (hb : pos_base + max_cells ≤ 2147483647) : ropePos pos_base (block pos) ≤ 2147483647 := by
  unfold ropePos block; omega

/-- The rotation position is the block's FIRST cell offset, not its last. -/
theorem ropePos_first_cell (pos_base pos : Nat) : ropePos pos_base (block pos) = pos_base + block pos * 4 := by
  unfold ropePos; omega

/-- Rope pairs index within the first half of the rotated span. -/
theorem pair_lt (d : Nat) : pairOf d < 32 := by unfold pairOf; omega

/-- For `d < 32` the pair is `d` itself; its partner element is `d + 32 < 64`. -/
theorem pair_low (d : Nat) (h : d < 32) : pairOf d = d ∧ pairOf d + 32 < 64 := by unfold pairOf; omega

/-- For `32 ≤ d < 64` the pair is `d - 32` and its partner element is `d`. -/
theorem pair_high (d : Nat) (h1 : 32 ≤ d) (h2 : d < 64) : pairOf d = d - 32 ∧ pairOf d + 32 = d := by
  unfold pairOf; omega

/-- The rotated span is inside the 128-wide row, so dimensions `64..127` pass through unrotated. -/
theorem rot_span_in_row : 64 ≤ 128 := by decide

/-- Pooled addresses `b*128 + d` for `b < rows` stay inside `rows*128` floats. -/
theorem pooled_addr_lt (b d rows : Nat) (hb : b < rows) (hd : d < 128) : b * 128 + d < rows * 128 := by
  omega

/-- 256 threads cover the 128 dimensions with the rest guarded off. -/
theorem threads_cover_dims : 128 ≤ 256 := by decide

/-- Block-sum: 256 threads give 8 warp partials, read by lanes `< 8`. -/
theorem sum_partials : 256 / 32 = 8 := by decide

/-- Position 0 is its own slot-0 position and block 0. -/
theorem pos_zero : slot 0 = 0 ∧ block 0 = 0 := by decide

/-- Capacity guard: positions at or beyond `max_cells` are ignored (early return). -/
theorem guard_excludes (pos max_cells : Nat) (h : max_cells ≤ pos) : ¬ pos < max_cells := by omega

/-- The mean of four equal incoming values is that value (position-0 case): `4*x/4 = x` over naturals. -/
theorem mean_of_four_equal (x : Nat) : (x + x + x + x) / 4 = x := by omega

end Ref.NativeQsaIndexer



## FILE: /mnt/user-data/outputs/NativeQsaScore.lean
/-!
# MODULE: native_qsa_score
SOURCES: src/kernels/cuda/native_qsa_score.cu
Tensor-core (TF32 mma) block scoring for the QSA indexer, adapted from llama.cpp's mmf path: 32 pooled rows per
block, two warps splitting the 128-wide reduction, a combine buffer, and a reference-ordered epilogue
(relu per head, ordered head sum, optional bias, +1e9 on an incomplete tail) written onto the block's cells.

## Source semantics
* Fixed geometry: D = 128, 4 heads, R = 4 cells per block, top_k = 2048, so the width cap is 2051.
* A step is valid only if `pos = n-1`, `n_bid = n/R`, `width = min(n, 2051)`.
* Warp `w`, lane `l` accumulates columns `w*32 + l` and `w*32 + l + 64`.
* `max_blocks = max_cells / R + 1`; blocks with `row0 > n_bid` return.

## Form template identifiers
* `D` -> index_dim, `ROWS` -> rows_per_block, `STRIDE` -> smem_row_stride, `COMBINE` -> combine_stride.
-/

namespace Ref.NativeQsaScore

/-- Indexer dimension. -/
def D : Nat := 128
/-- Rows per block. -/
def ROWS : Nat := 32
/-- Shared row stride (floats). -/
def STRIDE : Nat := 36
/-- Combine buffer stride. -/
def COMBINE : Nat := 68

/-- Two warps of 32 lanes with two passes of 64 columns cover the 128 columns. -/
theorem col_cover (c : Nat) (h : c < 128) :
    ∃ w l it, w < 2 ∧ l < 32 ∧ it < 2 ∧ c = w * 32 + l + 64 * it := by
  refine ⟨(c % 64) / 32, c % 32, c / 64, ?_, ?_, ?_, ?_⟩ <;> omega

/-- Column ownership is unique, so partial sums are disjoint. -/
theorem col_unique (w l it w' l' it' : Nat) (hw : w < 2) (hw' : w' < 2) (hl : l < 32) (hl' : l' < 32)
    (h : w * 32 + l + 64 * it = w' * 32 + l' + 64 * it') : w = w' ∧ l = l' ∧ it = it' := by omega

/-- The shared tile is `2 * 16 * 36 = 1152` floats. -/
theorem shared_size : 2 * 16 * STRIDE = 1152 := by decide

/-- Staging writes `tile[i*36 + lane]` stay inside the warp's 16x36 tile. -/
theorem stage_in_tile (i lane : Nat) (hi : i < 16) (hl : lane < 32) : i * 36 + lane < 16 * 36 := by omega

/-- A warp's tile offset plus an in-tile index stays inside the shared array. -/
theorem warp_tile_in_shared (w idx : Nat) (hw : w < 2) (hi : idx < 16 * 36) :
    w * (16 * 36) + idx < 2 * 16 * 36 := by omega

/-- `ldmatrix` A source: row `t%16`, column offset `(t/16)*4 + k*8` with 4 consecutive floats stays in-tile. -/
theorem load_a_in_tile (t k : Nat) (ht : t < 32) (hk : k < 4) :
    (t % 16) * 36 + (t / 16) * 4 + k * 8 + 3 < 16 * 36 := by omega

/-- Combine buffer writes: row index `i = w*32 + ia*16 + (l/2)*8 + lane/4` is below 64. -/
theorem combine_row_lt (w ia l lane : Nat) (hw : w < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    w * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by omega

/-- Combine head index `(lane%4)*2 + l%2` is below 8. -/
theorem combine_head_lt (l lane : Nat) : (lane % 4) * 2 + l % 2 < 8 := by omega

/-- Combine address `h*68 + i` stays inside the shared array (`h < 8`, `i < 64`). -/
theorem combine_addr_lt (h i : Nat) (hh : h < 8) (hi : i < 64) : h * 68 + i < 1152 := by omega

/-- The second-warp partial is read at `h*68 + 32 + lane`, still inside the array. -/
theorem combine_second_lt (h lane : Nat) (hh : h < 8) (hl : lane < 32) : h * 68 + 32 + lane < 1152 := by omega

/-- Block count `ceil(max_blocks/32)` covers every block row. -/
theorem grid_covers (max_blocks : Nat) : max_blocks ≤ (max_blocks + 31) / 32 * 32 := by omega

/-- A row `<= full` lies in a launched block when `full < max_blocks`. -/
theorem row_in_grid (row full max_blocks : Nat) (h : row ≤ full) (hf : full < max_blocks) :
    row / 32 < (max_blocks + 31) / 32 := by omega

/-- With `max_blocks = max_cells/4 + 1` and `n <= max_cells`, the last block `n/4` is a valid row. -/
theorem last_block_valid (n max_cells : Nat) (h : n ≤ max_cells) : n / 4 < max_cells / 4 + 1 := by omega

/-- The validity width equals `min(n, 2051)`. -/
def widthOk (n width : Nat) : Prop := width = if n < 2051 then n else 2051

/-- A valid step width never exceeds the cap or `n`. -/
theorem widthOk_bounds (n width : Nat) (h : widthOk n width) : width ≤ n ∧ width ≤ 2051 := by
  unfold widthOk at h; split at h <;> omega

/-- The 2051 cap is `top_k + R - 1` for `top_k = 2048`. -/
theorem width_cap : 2048 + 4 - 1 = 2051 := by decide

/-- The relu stage is nonnegative. -/
theorem relu_nonneg (v : Int) : 0 ≤ max v 0 := by omega

/-- The relu never lowers a value. -/
theorem relu_ge (v : Int) : v ≤ max v 0 := by omega

/-- Cells written by row `r`: `[4r, min(n, 4r+4))`, which are disjoint across rows. -/
theorem row_cells_disjoint (r r' c c' : Nat) (hc : c < 4) (hc' : c' < 4)
    (h : r * 4 + c = r' * 4 + c') : r = r' ∧ c = c' := by omega

/-- Capacity relation used by validation: `max_blocks = max_cells/4 + 1` leaves room for the spare block. -/
theorem spare_room (max_cells : Nat) : max_cells / 4 + 1 > max_cells / 4 := by omega

end Ref.NativeQsaScore



## FILE: /mnt/user-data/outputs/QsaBlockSelect.lean
/-!
# MODULE: qsa_block_select
SOURCES: src/kernels/cuda/qsa_select.cu
Batched block scoring (relu of per-head dots against the pooled key, spare/dead key for the tail) and an
exact radix-select top-k over BLOCKS where each block carries a cell weight (4, or the tail length), emitting
cell ids in ascending order with the lowest-index tie rule.

## Source semantics
* Block `b <= n_bid`; weight is `4` for `b < n_bid` and `n_kv - 4*n_bid` for the tail; weights sum to `n_kv`.
* The radix select fixes 8 key bits per pass (shifts 24, 16, 8, 0) with a weighted histogram.
* `eq_budget = width - above`; per-thread tie quota is `clamp(eq_budget - eq_before, 0, eq)`.
* If `n_kv <= width` every cell is selected (identity ascending).

## Form template identifiers
* `R` -> block_cells, `blockWeight` -> block_cell_count, `takeQuota` -> per_thread_tie_quota.
-/

namespace Ref.QsaSelect

/-- Cells per block. -/
def R : Nat := 4

/-- Weight of block `b`: full blocks have 4 cells, the last has the remainder. -/
def blockWeight (n_kv n_bid b : Nat) : Nat := if b < n_bid then 4 else n_kv - n_bid * 4

/-- Sum of the weights of the first `n` full blocks. -/
def fullSum : Nat → Nat
  | 0 => 0
  | n + 1 => fullSum n + 4

/-- The full-block sum is `4n`. -/
theorem fullSum_eq (n : Nat) : fullSum n = 4 * n := by
  induction n with
  | zero => rfl
  | succ k ih => simp only [fullSum, ih]; omega

/-- Weights over blocks `0..n_bid` sum to `n_kv` when `n_bid = n_kv / 4`. -/
theorem weights_total (n_kv n_bid : Nat) (h : n_bid = n_kv / 4) :
    fullSum n_bid + blockWeight n_kv n_bid n_bid = n_kv := by
  rw [fullSum_eq]
  unfold blockWeight
  rw [if_neg (by omega)]
  omega

/-- A block's weight is at most 4 (tail strictly below 4). -/
theorem weight_le (n_kv n_bid b : Nat) (h : n_bid = n_kv / 4) : blockWeight n_kv n_bid b ≤ 4 := by
  unfold blockWeight; split <;> omega

/-- The tail weight is zero iff `n_kv % 4 = 0` (an empty trailing block). -/
theorem tail_weight_zero_iff (n_kv n_bid : Nat) (h : n_bid = n_kv / 4) :
    blockWeight n_kv n_bid n_bid = 0 ↔ n_kv % 4 = 0 := by
  unfold blockWeight; rw [if_neg (by omega)]; omega

/-- Emitted cell ids `b*4 + c` (`c < weight`) stay below `n_kv`. -/
theorem cell_id_lt (n_kv n_bid b c : Nat) (h : n_bid = n_kv / 4) (hb : b ≤ n_bid)
    (hc : c < blockWeight n_kv n_bid b) : b * 4 + c < n_kv := by
  unfold blockWeight at hc; split at hc <;> omega

/-- Block-score grid: `ceil(max_blocks/8)` x `nq` blocks, block `bx*8 + warp`. -/
theorem score_grid_covers (max_blocks : Nat) : max_blocks ≤ (max_blocks + 7) / 8 * 8 := by omega

/-- Every block index maps to a (grid x, warp) pair. -/
theorem score_block_cover (b : Nat) : ∃ bx w, w < 8 ∧ bx * 8 + w = b := by
  exact ⟨b / 8, b % 8, by omega, by omega⟩

/-- A key whose tail block is empty keeps the +1e9 bias off: bias iff `n_kv % 4 ≠ 0`. -/
theorem bias_iff (n_kv n_bid : Nat) (h : n_bid = n_kv / 4) : n_bid * 4 < n_kv ↔ n_kv % 4 ≠ 0 := by omega

/-- Identity early exit: when `n_kv <= width` the output `j` for `j < n_kv` is the full range. -/
theorem identity_selection (n_kv width j : Nat) (h : n_kv ≤ width) (hj : j < n_kv) : j < width := by omega

/-- Digit extraction: a 32-bit key splits into four 8-bit digits that reassemble it. -/
theorem digits_roundtrip (k : Nat) (h : k < 4294967296) :
    (k / 16777216) * 16777216 + ((k / 65536) % 256) * 65536 + ((k / 256) % 256) * 256 + k % 256 = k := by
  omega

/-- Each digit is below 256, so the 256-bin histogram suffices. -/
theorem digit_lt (k s : Nat) : (k / s) % 256 < 256 := Nat.mod_lt _ (by omega)

/-- The top digit of a 32-bit key is also below 256. -/
theorem top_digit_lt (k : Nat) (h : k < 4294967296) : k / 16777216 < 256 := by omega

/-- Four passes of 8 bits consume exactly 32 bits. -/
theorem passes_cover : 4 * 8 = 32 := by decide

/-- Per-thread tie quota: the part of the budget left after the earlier threads, clamped to `eq`. -/
def takeQuota (budget before eq : Nat) : Nat := min eq (budget - before)

/-- Quotas telescope: `min budget (before+eq) = min budget before + quota`. -/
theorem quota_telescopes (budget before eq : Nat) :
    min budget (before + eq) = min budget before + takeQuota budget before eq := by
  unfold takeQuota; omega

/-- A quota never exceeds the thread's own tie count. -/
theorem quota_le_eq (budget before eq : Nat) : takeQuota budget before eq ≤ eq := by
  unfold takeQuota; omega

/-- When earlier threads already used the budget, the quota is zero. -/
theorem quota_zero (budget before eq : Nat) (h : budget ≤ before) : takeQuota budget before eq = 0 := by
  unfold takeQuota; omega

/-- If total ties reach the budget, the final prefix hits the budget exactly. -/
theorem quota_total (budget before eq : Nat) (h : budget ≤ before + eq) :
    min budget before + takeQuota budget before eq = budget := by
  unfold takeQuota; omega

/-- Selected count of a thread is `gt + quota`; with the global counts it equals the width. -/
theorem selected_total (cgt budget width : Nat) (h : budget = width - cgt) (hg : cgt ≤ width) :
    cgt + budget = width := by omega

/-- Thread partitioning of the block range: `per = ceil(nb/256)` makes 256 threads cover `nb` blocks. -/
theorem per_thread_cover (nb per : Nat) (h : per = (nb + 255) / 256) : nb ≤ 256 * per := by omega

end Ref.QsaSelect



## FILE: /mnt/user-data/outputs/QsaCacheIndexAttend.lean
/-!
# MODULE: qsa_cache_index_attend
SOURCES: src/kernels/cuda/qsa.cu
The QSA cache, indexer and attention: paged KV append, indexer key pooling with a spare slot, per-block
scores mapped onto cells, exact top-k by threshold search with an ascending tie rule, KV gather and a
one-block-per-head attention, plus the step-state (pos, n_kv, n_bid, width) derived from one position.

## Source semantics
* From `pos`: `n_kv = pos+1`, `n_bid = n_kv / r`, `width = min(n_kv, top_k + r - 1)`.
* Cells of block `b < n_bid` are `[b*r, b*r+r)`; the incomplete tail is `[n_bid*r, n_kv)` and is biased +1e9.
* top-k = every cell strictly above the width-th key, then the first `width - c_gt` cells equal to it.
* The f32 order key maps -0 to +0, NaN to 0, and flips negatives so unsigned comparison orders floats.
* GQA pairing is `kv = h / (n_head / n_head_kv)` (NOT `h % n_head_kv`).
* gate uses the SECOND half of the doubled query row.

## Form template identifiers
* `selWidth` -> selection_width, `stepNBid` -> completed_blocks, `orderKey` -> float_order_key.
-/

namespace Ref.Qsa

/-- Selection width: `min(n_kv, top_k + r - 1)`. -/
def selWidth (n topk r : Nat) : Nat := min n (topk + r - 1)

/-- The width never exceeds the cell count. -/
theorem selWidth_le_n (n topk r : Nat) : selWidth n topk r ≤ n := by unfold selWidth; omega

/-- The width never exceeds `top_k + r - 1`. -/
theorem selWidth_le_cap (n topk r : Nat) : selWidth n topk r ≤ topk + r - 1 := by unfold selWidth; omega

/-- Below the bound the selection is the identity (everything selected). -/
theorem selWidth_identity (n topk r : Nat) (h : n ≤ topk + r - 1) : selWidth n topk r = n := by
  unfold selWidth; omega

/-- With `top_k = 2048`, `r = 4` the cap is 2051 (matches the literal in the native score kernel). -/
theorem selWidth_real (n : Nat) : selWidth n 2048 4 = if n < 2051 then n else 2051 := by
  unfold selWidth; split <;> omega

/-- Completed blocks for `r = 4`. -/
def stepNBid (pos : Nat) : Nat := (pos + 1) / 4

/-- Completed blocks cover no more than the cells present. -/
theorem nbid_cells_le (pos : Nat) : stepNBid pos * 4 ≤ pos + 1 := by unfold stepNBid; omega

/-- The incomplete tail is shorter than one block. -/
theorem tail_lt_block (pos : Nat) : pos + 1 - stepNBid pos * 4 < 4 := by unfold stepNBid; omega

/-- A tail exists iff `n_kv % 4 ≠ 0` (the +1e9 bias condition). -/
theorem tail_iff (pos : Nat) : stepNBid pos * 4 < pos + 1 ↔ (pos + 1) % 4 ≠ 0 := by
  unfold stepNBid; omega

/-- Cells of a completed block lie below `n_kv`. -/
theorem full_block_cells_lt (pos b c : Nat) (hb : b < stepNBid pos) (hc : c < 4) : b * 4 + c < pos + 1 := by
  unfold stepNBid at hb; omega

/-- Every cell is in exactly one block: block `cell / 4`, which is `<= n_bid`. -/
theorem cell_block_le (pos cell : Nat) (h : cell < pos + 1) : cell / 4 ≤ stepNBid pos := by
  unfold stepNBid; omega

/-- Counts: with `c_gt < w <= c_gt + c_eq` exactly `w` cells are selected. -/
theorem topk_selected_count (cgt ceq w : Nat) (h1 : cgt < w) (h2 : w ≤ cgt + ceq) :
    cgt + min (w - cgt) ceq = w := by omega

/-- The tie budget is positive exactly when the strict-above set is short of the width. -/
theorem eq_budget_pos (cgt w : Nat) (h : cgt < w) : 0 < w - cgt := by omega

/-- Order key of a float bit pattern `b < 2^32`: positives get the top bit set, negatives are complemented. -/
def orderKey (b : Nat) : Nat := if b < 2147483648 then b + 2147483648 else 4294967295 - b

/-- Keys are 32-bit values. -/
theorem orderKey_lt (b : Nat) (h : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey; split <;> omega

/-- Among non-negative floats the key is monotone in the bit pattern. -/
theorem orderKey_mono_pos (a b : Nat) (ha : a < 2147483648) (hb : b < 2147483648) (h : a ≤ b) :
    orderKey a ≤ orderKey b := by
  unfold orderKey; rw [if_pos ha, if_pos hb]; omega

/-- Among negative floats a larger magnitude (bit pattern) gives a smaller key. -/
theorem orderKey_anti_neg (a b : Nat) (ha : 2147483648 ≤ a) (hb : 2147483648 ≤ b)
    (hb' : b < 4294967296) (h : a ≤ b) : orderKey b ≤ orderKey a := by
  unfold orderKey; rw [if_neg (by omega), if_neg (by omega)]; omega

/-- Every negative key is below every non-negative key. -/
theorem orderKey_neg_lt_pos (a b : Nat) (ha : 2147483648 ≤ a) (ha' : a < 4294967296)
    (hb : b < 2147483648) : orderKey a < orderKey b := by
  unfold orderKey; rw [if_neg (by omega), if_pos hb]; omega

/-- Page-local slot is below the page size. -/
theorem slot_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

/-- Indexer: a block completes exactly when `slot = r-1 = 3`. -/
theorem block_complete_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- Completing a block, the spare row `b+1` is within `max_cells/4 + 1` pooled rows. -/
theorem spare_row_in_pool (pos maxc : Nat) (hp : pos < maxc) (h3 : pos % 4 = 3) :
    pos / 4 + 1 < maxc / 4 + 1 := by omega

/-- The tail has `r-1 = 3` rows: every stored slot indexes inside `3*idx_dim` floats (idx_dim = 128). -/
theorem tail_index_lt (slot d : Nat) (hs : slot < 3) (hd : d < 128) : slot * 128 + d < 3 * 128 := by omega

/-- GQA pairing with 24 heads and 2 KV heads (group of 12) is a valid KV head. -/
theorem gqa_kv_lt (h : Nat) (hh : h < 24) : h / 12 < 2 := by omega

/-- The division pairing differs from a modulo pairing (head 1 maps to kv 0, not 1). -/
theorem gqa_not_mod : (1 : Nat) / 12 ≠ 1 % 2 := by decide

/-- The gate half starts exactly one head_dim into the doubled row (head_dim = 256). -/
theorem gate_second_half (h d : Nat) (hd : d < 256) :
    h * 512 + 256 ≤ h * 512 + 256 + d ∧ h * 512 + 256 + d < (h + 1) * 512 := by omega

/-- Gather decomposition (2 KV heads, 64 uint2 per row): flat index splits into id, head, quad. -/
theorem gather_decomp (i : Nat) :
    i / 128 * 128 + (i % 128) / 64 * 64 + (i % 128) % 64 = i := by omega

/-- Gather: with `i < n_ids * 128` the id is `< n_ids`. -/
theorem gather_id_lt (i n : Nat) (h : i < n * 128) : i / 128 < n := by omega

/-- Gather: the head part is below 2 and the quad below 64. -/
theorem gather_parts_lt (i : Nat) : (i % 128) / 64 < 2 ∧ (i % 128) % 64 < 64 := by omega

/-- Launch ceil: a grid of `ceil(n/256)` blocks covers `n` threads. -/
theorem grid_for_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- Attention shared memory per head is `(max_ids + 32)` floats; the 32 reduction slots precede the weights. -/
theorem attend_smem (max_ids : Nat) : (max_ids + 32) * 4 = max_ids * 4 + 128 := by omega

end Ref.Qsa



## FILE: /mnt/user-data/outputs/QsaDecodeAttention.lean
/-!
# MODULE: qsa_decode_attention
SOURCES: src/kernels/cuda/qsa_decode_attn.cu
Split-KV decode attention over the QSA-selected cells: a chunk kernel (64 cells per block, one block per
(chunk, KV head [, query])) emitting per-chunk running max, exp-sum and weighted value accumulators for the 12
query heads sharing a KV head, then a merge kernel combining chunks by online-softmax rescaling. fp16 and int8
(grouped scale) pools are supported.

## Source semantics
* `HD = 256`, `G = 12` query heads per KV head, `CHUNK = 64`, 256 threads (8 warps).
* Scratch layout per query: `[acc: chunks*n_head*HD][m: chunks*n_head][l: chunks*n_head]` plus 64 slack floats.
* Chunk slot `kvh*n_chunks + chunk`; global query head `h = kvh*12 + hl`.
* Empty chunks write `m = -FLT_MAX`, `l = 0` and are skipped in the merge.

## Form template identifiers
* `HD` -> head_dim, `G` -> group_size, `CHUNK` -> cells_per_chunk, `scratchFloats` -> scratch_size.
-/

namespace Ref.QsaDecodeAttn

/-- Head dimension. -/
def HD : Nat := 256
/-- Query heads per KV head. -/
def G : Nat := 12
/-- Cells per chunk. -/
def CHUNK : Nat := 64
/-- Threads per block. -/
def THREADS : Nat := 256

/-- Scratch size (floats) from the host formula `chunks * n_head * (HD + 2) + 64`. -/
def scratchFloats (chunks n_head : Nat) : Nat := chunks * n_head * (HD + 2) + 64

/-- Eight warps per block. -/
theorem warps : THREADS / 32 = 8 := by decide

/-- Each lane holds 8 of the 256 dimensions: 32 lanes x 8 = HD. -/
theorem lane_dims : 32 * 8 = HD := by decide

/-- One thread per dimension in the value loop. -/
theorem thread_per_dim : THREADS = HD := by decide

/-- The 64 scores of a head are covered by two per lane (`lane`, `lane+32`). -/
theorem two_per_lane : 2 * 32 = CHUNK := by decide

/-- Query tile in shared memory is `12 * 256` floats = 12 KB. -/
theorem sq_bytes : G * HD * 4 = 12288 := by decide

/-- Heads `warp` and `warp + 8` cover all 12 heads. -/
theorem head_cover (h : Nat) (hh : h < 12) : ∃ w, w < 8 ∧ (h = w ∨ h = w + 8) := by
  by_cases h8 : h < 8
  · exact ⟨h, h8, Or.inl rfl⟩
  · exact ⟨h - 8, by omega, Or.inr (by omega)⟩

/-- Cells `warp, warp+8, ...` cover all 64 cells of a chunk. -/
theorem cell_cover (c : Nat) (h : c < 64) : ∃ w k, w < 8 ∧ k < 8 ∧ c = w + 8 * k := by
  exact ⟨c % 8, c / 8, by omega, by omega, by omega⟩

/-- Cell loop ownership is unique. -/
theorem cell_unique (w k w' k' : Nat) (hw : w < 8) (hw' : w' < 8)
    (h : w + 8 * k = w' + 8 * k') : w = w' ∧ k = k' := by omega

/-- The chunk count covers the capacity. -/
theorem chunks_cover (cap : Nat) : cap ≤ (cap + 63) / 64 * 64 := by omega

/-- Cell `c0 + t` of the last chunk is below `chunks*64`. -/
theorem chunk_cell_lt (cap chunk t : Nat) (hc : chunk < (cap + 63) / 64) (ht : t < 64) :
    chunk * 64 + t < (cap + 63) / 64 * 64 := by omega

/-- The three scratch regions fit in the scratch size (acc, then m, then l). -/
theorem scratch_regions_fit (chunks n_head : Nat) :
    chunks * n_head * 256 + chunks * n_head + chunks * n_head ≤ scratchFloats chunks n_head := by
  unfold scratchFloats HD; omega

/-- The `m` region starts exactly where `acc` ends and `l` where `m` ends. -/
theorem scratch_offsets (chunks n_head : Nat) :
    chunks * n_head * 256 + chunks * n_head = chunks * n_head * 257 := by omega

/-- A slot index `kvh*n_chunks + chunk` is below `n_kv_heads * n_chunks`. -/
theorem slot_lt (kvh chunk n_kv n_chunks : Nat) (hk : kvh < n_kv) (hc : chunk < n_chunks) :
    kvh * n_chunks + chunk < n_kv * n_chunks := by
  have h1 : (kvh + 1) * n_chunks ≤ n_kv * n_chunks := Nat.mul_le_mul_right _ (by omega)
  have h2 : (kvh + 1) * n_chunks = kvh * n_chunks + n_chunks := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Global head decomposition `h = kvh*12 + hl` round-trips. -/
theorem head_roundtrip (kvh hl : Nat) (hl' : hl < 12) :
    (kvh * 12 + hl) / 12 = kvh ∧ (kvh * 12 + hl) % 12 = hl := by omega

/-- Conversely every global head splits into a KV head and a local head. -/
theorem head_split (h : Nat) : h / 12 * 12 + h % 12 = h ∧ h % 12 < 12 := by omega

/-- The geometry check `n_head = 12 * n_kv_heads` makes every KV head have exactly 12 query heads. -/
theorem kv_head_lt (h n_kv : Nat) (hh : h < 12 * n_kv) : h / 12 < n_kv := by omega

/-- Batched grid: query row stride is `n_kv_heads * 12 * 256` floats; rows are disjoint. -/
theorem q_rows_disjoint (z z' n_kv : Nat) (h : z < z') :
    z * (n_kv * 12 * 256) + n_kv * 12 * 256 ≤ z' * (n_kv * 12 * 256) := by
  have := Nat.mul_le_mul_right (n_kv * 12 * 256) (show z + 1 ≤ z' by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- Online-softmax merge identity at scalar level: rescaling a partial sum from its own max `m` to a global
max `M` multiplies by `exp(m - M)`; here as the multiplicative law on exponent differences (naturals). -/
theorem exp_shift_compose (m1 m2 M : Nat) (h1 : m1 ≤ m2) (h2 : m2 ≤ M) :
    (M - m1) = (M - m2) + (m2 - m1) := by omega

/-- Merge guard: a chunk whose max equals the sentinel contributes nothing (abstractly `0 * x = 0`). -/
theorem empty_chunk_contributes_zero (x : Int) : 0 * x = 0 := by omega

/-- Chunks beyond the selection are empty: `c0 >= n_ids` gives `n_ids - c0 <= 0`. -/
theorem empty_chunk (c0 n_ids : Nat) (h : n_ids ≤ c0) : (n_ids : Int) - c0 ≤ 0 := by omega

end Ref.QsaDecodeAttn





## FILE: /mnt/user-data/outputs/ActivationQuantizer.lean
/-!
# MODULE: activation_quantizer
SOURCES: src/kernels/cuda/quantize_act.cu
Quantizing an fp32 activation to ggml's vec_dot_type: Q8_0 (34-byte blocks of 32), a CPU-order Q8_0 variant with
fp32 scales, and Q8_K (292-byte blocks of 256 with block sums), plus matching dequantizers.

## Source semantics
Q8_0: d = amax/127, ints are x/d clamped to [-128,127] (ggml bytes) or [-127,127] (CPU rule). Zero blocks are all
zero. Q8_K: iscale = -127/max (signed max at largest magnitude), q = min 127 (nearest_int (iscale x)) with
nearest_int the 1.5*2^23 magic trick; bsums[j] = sum of 16 consecutive qs; n must be a multiple of 256.
Sizes must be multiples of the block (32 resp. 256).

## Form template identifiers
clamp -> "int8 saturation"; nearestInt -> "magic-number rounding"; bsum -> "block sum".
-/
namespace Ref.ActivationQuantizer

-- § saturation
/-- Clamp an integer into [lo, hi]. -/
def clamp (lo hi v : Int) : Int := if v < lo then lo else if v > hi then hi else v

/-- Clamping into [-127,127] bounds the result. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp (-127) 127 v ∧ clamp (-127) 127 v ≤ 127 := by
  unfold clamp
  split_ifs <;> omega

/-- Clamping is idempotent. -/
theorem clamp_idem (lo hi v : Int) (h : lo ≤ hi) : clamp lo hi (clamp lo hi v) = clamp lo hi v := by
  unfold clamp
  split_ifs <;> omega

/-- In-range values are unchanged. -/
theorem clamp_of_mem (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) : clamp lo hi v = v := by
  unfold clamp
  split_ifs <;> omega

-- § magic-number rounding
/-- `nearest_int`: low 23 mantissa bits minus 2^22. -/
def nearestInt (bits : Nat) : Int := ((bits &&& 8388607 : Nat) : Int) - 4194304

/-- The result lies in [-2^22, 2^22 - 1], so the trick is valid for |x| < 2^22. -/
theorem nearestInt_range (bits : Nat) : -4194304 ≤ nearestInt bits ∧ nearestInt bits ≤ 4194303 := by
  have h : (bits &&& 8388607) ≤ 8388607 := Nat.and_le_right
  unfold nearestInt
  omega

-- § sizes and counts
/-- Q8_0 block count. -/
def q80Blocks (n : Nat) : Nat := n / 32

/-- Q8_K block count. -/
def q8kBlocks (n : Nat) : Nat := n / 256

/-- A grid of 128 threads per block covers every Q8_0 block. -/
theorem q80_grid_covers (n : Nat) : q80Blocks n ≤ ((q80Blocks n + 127) / 128) * 128 := by
  unfold q80Blocks
  omega

/-- A grid of 64 threads per block covers every Q8_K block. -/
theorem q8k_grid_covers (n : Nat) : q8kBlocks n ≤ ((q8kBlocks n + 63) / 64) * 64 := by
  unfold q8kBlocks
  omega

/-- Whole blocks cover the array when n is a multiple of the block. -/
theorem q80_blocks_cover (n : Nat) (h : n % 32 = 0) : q80Blocks n * 32 = n := by
  unfold q80Blocks
  omega

/-- Same for Q8_K. -/
theorem q8k_blocks_cover (n : Nat) (h : n % 256 = 0) : q8kBlocks n * 256 = n := by
  unfold q8kBlocks
  omega

-- § block sums
/-- Sum of a block of quantized values. -/
def bsum (qs : List Int) : Int := qs.sum

/-- Block sums are additive over concatenation: total of 16 sub-sums equals the sum over the 256 values. -/
theorem bsum_append (a b : List Int) : bsum (a ++ b) = bsum a + bsum b := by
  unfold bsum
  exact List.sum_append

/-- A zero block has zero sum (the amax = 0 case writes zeroes everywhere). -/
theorem bsum_zero (n : Nat) : bsum (List.replicate n 0) = 0 := by
  unfold bsum
  simp

end Ref.ActivationQuantizer



## FILE: /mnt/user-data/outputs/DequantToHalfBf16.lean
/-!
# MODULE: dequant_to_half_bf16
SOURCES: src/kernels/cuda/dequant_bf16.cu
Dequantizing ggml block formats (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS) to bf16, fp16 or
fp32, one thread per 32-element group; the i-quants are delegated to iq_kernels.

## Source semantics
Row stride is cols/be*bb bytes; each row has cols/32 groups. Group g of a row lives in block g / (be/32) at
sub-group g % (be/32). K-quant scale/min unpacking yields 6-bit values. fp32 -> bf16 rounds to nearest even via
u += 0x7fff + ((u>>16)&1); result u>>16.

## Form template identifiers
geometry -> "format table"; bf16Round -> "round-to-nearest-even truncation"; scaleMin -> "6-bit scale unpack".
-/
namespace Ref.DequantBf16

-- § geometry
/-- (elements, bytes) per block, or (0,0). -/
def beOf (t : Nat) : Nat :=
  if t = 2 ∨ t = 6 ∨ t = 8 ∨ t = 20 then 32 else if t = 42 then 64 else
  if t = 11 ∨ t = 12 ∨ t = 13 ∨ t = 14 ∨ t = 23 then 256 else 0

/-- Bytes per block. -/
def bbOf (t : Nat) : Nat :=
  if t = 2 then 18 else if t = 6 then 22 else if t = 8 then 34 else if t = 20 then 18
  else if t = 11 then 110 else if t = 12 then 144 else if t = 13 then 176 else if t = 14 then 210
  else if t = 23 then 136 else if t = 42 then 18 else 0

/-- `dequant_bf16_supported`. -/
def supp (t : Nat) : Bool := beOf t != 0

/-- Every supported block is a whole number of 32-element groups. -/
theorem groups_per_block (t : Nat) (ht : t < 64) (h : supp t = true) : beOf t % 32 = 0 := by
  revert t
  decide

/-- Group decomposition: block index and sub-group reassemble the group index (256-wide blocks). -/
theorem group_decomp_256 (g : Nat) : 8 * (g / 8) + g % 8 = g := by
  omega

/-- Same for Q2_0 (two groups per 64-element block). -/
theorem group_decomp_q2_0 (g : Nat) : 2 * (g / 2) + g % 2 = g := by
  omega

/-- The sub-group of a 256-wide block is below 8 (so its element offset is below 256). -/
theorem subgroup_lt (g : Nat) : (g % 8) * 32 + 32 ≤ 256 := by
  omega

/-- The i-quant types dispatched elsewhere are disjoint from this file's table. -/
theorem iq_disjoint (t : Nat) (ht : t < 64) (h : t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 29) :
    supp t = false := by
  revert t
  decide

-- § K-quant scale unpacking
/-- High-index scale (j >= 4): low nibble of q[j+4] plus two high bits of q[j-4] shifted up four. -/
def scaleHi (lowByte hiByte : Nat) : Nat := lowByte % 16 + (hiByte / 64) * 16

/-- Unpacked 6-bit scales stay below 64. -/
theorem scaleHi_lt (a b : Nat) (ha : a < 256) (hb : b < 256) : scaleHi a b < 64 := by
  unfold scaleHi
  omega

/-- Low-index scale (j < 4): the low six bits of the byte. -/
theorem scaleLo_lt (a : Nat) : a % 64 < 64 := by
  omega

-- § bf16 rounding
/-- `f2bf` on the raw 32-bit pattern (no overflow modeled: finite inputs). -/
def bf16Round (u : Nat) : Nat := (u + 32767 + (u / 65536) % 2) / 65536

/-- Exactly representable inputs (low half zero) are preserved. -/
theorem bf16_exact (u : Nat) (h : u % 65536 = 0) : bf16Round u = u / 65536 := by
  unfold bf16Round
  omega

/-- Below the halfway point the result truncates. -/
theorem bf16_round_down (u : Nat) (h : u % 65536 < 32768) : bf16Round u = u / 65536 := by
  unfold bf16Round
  omega

/-- Above the halfway point the result rounds up. -/
theorem bf16_round_up (u : Nat) (h : 32768 < u % 65536) : bf16Round u = u / 65536 + 1 := by
  unfold bf16Round
  omega

/-- At exactly the halfway point the result is even (ties to even). -/
theorem bf16_tie_even (u : Nat) (h : u % 65536 = 32768) : bf16Round u % 2 = 0 := by
  obtain ⟨a, ha⟩ : ∃ a, u = 65536 * a + 32768 := ⟨u / 65536, by omega⟩
  subst ha
  have h1 : (65536 * a + 32768) / 65536 = a := Nat.div_eq_of_lt_le (by omega) (by omega)
  unfold bf16Round
  rw [h1]
  rcases Nat.mod_two_eq_zero_or_one a with h2 | h2
  · have key : (65536 * a + 32768 + 32767 + a % 2) / 65536 = a :=
      Nat.div_eq_of_lt_le (by omega) (by omega)
    rw [key]
    omega
  · have key : (65536 * a + 32768 + 32767 + a % 2) / 65536 = a + 1 :=
      Nat.div_eq_of_lt_le (by omega) (by omega)
    rw [key]
    omega

/-- Launch grid covers every group. -/
theorem grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by
  omega

end Ref.DequantBf16



## FILE: /mnt/user-data/outputs/IqCodebookKernels.lean
/-!
# MODULE: iq_codebook_kernels
SOURCES: src/kernels/cuda/iq_kernels.cu
Warp-per-row MMVQ, grouped native MoE experts, q8_1 quantizer and dequantizers for the i-quant family
(IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_M, IQ4_NL, IQ4_XS, Q3_K, Q2_0), transcribed from llama.cpp.

## Source semantics
Every format has a block geometry (elements per block, bytes per block). A row of n values occupies
(n / qk) * sizeof(block) bytes. A native expert blob is [gate rows | up rows | down rows]; gate/up use the
gate/up type and the down rows use the down type. Scratch regions are 256-byte aligned. dp4a is a 4-lane
int8 dot-product-accumulate; adding a constant to codes is corrected with a dp4a against all-ones.

## Form template identifiers
blockBytes/blockElems -> abstract "format geometry"; mkLayout -> "expert blob layout";
align256 -> "scratch alignment"; dp4/dp4a -> "packed int8 dot".
-/
namespace Ref.IqKernels

-- § block geometry
/-- sizeof of the ggml block structs (QK_K = 256), keyed by ggml type id; 0 when unsupported. -/
def blockBytes (t : Nat) : Nat :=
  if t = 16 then 66 else if t = 17 then 74 else if t = 18 then 98 else if t = 20 then 18
  else if t = 21 then 110 else if t = 22 then 82 else if t = 23 then 136 else if t = 29 then 56
  else if t = 11 then 110 else if t = 42 then 18 else 0

/-- Values per block. -/
def blockElems (t : Nat) : Nat :=
  if t = 20 then 32 else if t = 42 then 64 else
  if t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 23 ∨ t = 29 ∨ t = 11 then 256 else 0

/-- `is_iq`: the type ids this file handles. -/
def isIq (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11

/-- `iq_row_bytes`. -/
def rowBytes (t n : Nat) : Nat := n / blockElems t * blockBytes t

/-- Extending a row by one block adds exactly one block of bytes. -/
theorem rowBytes_add_block (t n : Nat) (h : 0 < blockElems t) :
    rowBytes t (n + blockElems t) = rowBytes t n + blockBytes t := by
  unfold rowBytes
  rw [Nat.add_div_right _ h, Nat.add_mul, Nat.one_mul]

/-- IQ2_XXS: 256 values occupy one 66-byte block. -/
theorem rowBytes_iq2_xxs_256 : rowBytes 16 256 = 66 := by
  decide

/-- Unsupported ids (below 64) have zero row bytes, mirroring the `default: return 0` arm. -/
theorem rowBytes_unsupported (t n : Nat) (ht : t < 64) (h : isIq t = false) : rowBytes t n = 0 := by
  have hb : blockBytes t = 0 :=
    (by decide : ∀ t, t < 64 → isIq t = false → blockBytes t = 0) t ht h
  simp [rowBytes, hb]

-- § per-format dot-call traits
/-- Calls-per-block for a format (`Fmt<TY>::ipb`). -/
def fmtIpb (t : Nat) : Nat := if t = 20 ∨ t = 42 then 2 else 8

/-- The `iqs` stride between calls (`Fmt<TY>::step`). -/
def fmtStep (t : Nat) : Nat := if t = 29 ∨ t = 42 then 1 else 2

/-- The `Fmt` block size agrees with the geometry table for every kernel-dispatched format. -/
theorem fmt_qk_matches (t : Nat) (ht : t < 64) (h : isIq t = true) (h1 : t ≠ 11) (h2 : t ≠ 23) :
    blockElems t = (if t = 20 then 32 else if t = 42 then 64 else 256) := by
  revert t
  decide

/-- For the 256-wide formats stepping by 2, every call's q8_1 block index `iqs/2` is inside the block. -/
theorem fmt_q8_index_in_range (t : Nat) (ht : t < 64) (h : t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22) :
    ∀ k, k < fmtIpb t → (fmtStep t * k) / 2 < blockElems t / 32 := by
  revert t
  decide

/-- IQ1_M and Q2_0 index q8_1 blocks directly by `iqs`, which stays inside the block. -/
theorem fmt_q8_direct_index (t : Nat) (ht : t < 64) (h : t = 29 ∨ t = 42) :
    ∀ k, k < fmtIpb t → fmtStep t * k < blockElems t / 32 := by
  revert t
  decide

-- § expert blob layout
/-- `NativeExpertLayout`. -/
structure Layout where
  guRow : Nat
  dRow : Nat
  upOff : Nat
  downOff : Nat
  bytes : Nat

/-- `native_expert_layout`. -/
def mkLayout (gu d ne nf : Nat) : Layout :=
  { guRow := rowBytes gu ne, dRow := rowBytes d nf,
    upOff := nf * rowBytes gu ne, downOff := 2 * (nf * rowBytes gu ne),
    bytes := 2 * (nf * rowBytes gu ne) + ne * rowBytes d nf }

/-- The down block starts after both gate and up regions. -/
theorem layout_down_off (gu d ne nf : Nat) :
    (mkLayout gu d ne nf).downOff = 2 * (mkLayout gu d ne nf).upOff := by
  rfl

/-- Total blob size is the down offset plus the down rows. -/
theorem layout_bytes (gu d ne nf : Nat) :
    (mkLayout gu d ne nf).bytes = (mkLayout gu d ne nf).downOff + ne * (mkLayout gu d ne nf).dRow := by
  rfl

/-- Every up row (index r < n_ff) ends before the down region begins. -/
theorem layout_up_row_fits (gu d ne nf r : Nat) (h : r < nf) :
    (mkLayout gu d ne nf).upOff + r * (mkLayout gu d ne nf).guRow + (mkLayout gu d ne nf).guRow
      ≤ (mkLayout gu d ne nf).downOff := by
  have h1 := Nat.mul_le_mul_right (rowBytes gu ne) (show r + 1 ≤ nf by omega)
  have h2 : r * rowBytes gu ne + rowBytes gu ne ≤ nf * rowBytes gu ne := by
    simpa [Nat.add_mul] using h1
  show nf * rowBytes gu ne + r * rowBytes gu ne + rowBytes gu ne ≤ 2 * (nf * rowBytes gu ne)
  omega

-- § scratch sizing
/-- Round up to a multiple of 256. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by
  unfold align256
  omega

/-- Alignment yields a multiple of 256. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by
  unfold align256
  omega

/-- Alignment wastes fewer than 256 bytes. -/
theorem align256_lt (n : Nat) : align256 n < n + 256 := by
  unfold align256
  omega

/-- `native_expert_scratch_bytes` (sizeof block_q8_1 = 36). -/
def scratchBytes (cap nf : Nat) : Nat :=
  3 * align256 (cap * nf * 4) + align256 (cap * (nf / 32) * 36)

/-- The scratch size is itself a multiple of 256, so the carved regions stay aligned. -/
theorem scratch_aligned (cap nf : Nat) : scratchBytes cap nf % 256 = 0 := by
  unfold scratchBytes align256
  omega

-- § packed int8 dot
/-- `__dp4a` without accumulator. -/
def dp4 (a b : Fin 4 → Int) : Int := a 0 * b 0 + a 1 * b 1 + a 2 * b 2 + a 3 * b 3

/-- `__dp4a(a, b, c)`. -/
def dp4a (a b : Fin 4 → Int) (c : Int) : Int := c + dp4 a b

/-- The accumulator only shifts the result. -/
theorem dp4a_acc (a b : Fin 4 → Int) (c : Int) : dp4a a b c = c + dp4a a b 0 := by
  unfold dp4a
  omega

/-- Dotting against 0x01010101 sums the lanes (the `sumy` trick of IQ1_M). -/
theorem dp4_ones (b : Fin 4 → Int) : dp4 (fun _ => 1) b = b 0 + b 1 + b 2 + b 3 := by
  unfold dp4
  simp

/-- A constant bias on the codes is corrected by a multiple of the lane sum. -/
theorem dp4_sub_const (a b : Fin 4 → Int) (k : Int) :
    dp4 (fun i => a i - k) b = dp4 a b - k * (b 0 + b 1 + b 2 + b 3) := by
  unfold dp4
  simp only [Int.sub_mul, Int.mul_add]
  omega

end Ref.IqKernels



## FILE: /mnt/user-data/outputs/KvCacheQ8.lean
/-!
# MODULE: kv_cache_q8
SOURCES: src/kernels/cuda/kv_q8.cu
Append (quantize) and gather (dequantize) steps for a paged int8 KV cache with one fp16 scale per 64-value group.

## Source semantics
Row of (page, head, in-page position) is (page*kv_heads + head)*page_size + pos % page_size, where page is the
page-table entry for pos / page_size. A row holds head_dim int8 values and head_dim/64 scales; head_dim must be
a multiple of 64. Quantization uses the STORED (fp16-rounded) scale, ints clamped to [-127,127]. Gather reads four
values per thread; thread index decomposes as (id, head, quad) and writes the scratch slot with the same linear
index.

## Form template identifiers
rowIdx -> "paged row index"; scaleIdx -> "group scale index"; gather decomposition -> "thread to (id, head, quad)".
-/
namespace Ref.KvQ8

-- § paged row index
/-- Row of an (page, head, offset) triple. -/
def rowIdx (page kvh h ps off : Nat) : Nat := ps * (page * kvh + h) + off

/-- The page-major cell is recovered by division. -/
theorem rowIdx_div (page kvh h ps off : Nat) (hps : 0 < ps) (hoff : off < ps) :
    rowIdx page kvh h ps off / ps = page * kvh + h := by
  unfold rowIdx
  rw [Nat.mul_add_div hps, Nat.div_eq_of_lt hoff]
  simp

/-- The in-page position is recovered by remainder. -/
theorem rowIdx_mod (page kvh h ps off : Nat) (hps : 0 < ps) (hoff : off < ps) :
    rowIdx page kvh h ps off % ps = off := by
  have hd := rowIdx_div page kvh h ps off hps hoff
  have h1 := Nat.div_add_mod (rowIdx page kvh h ps off) ps
  rw [hd] at h1
  unfold rowIdx at h1 ⊢
  omega

/-- Distinct (cell, offset) pairs get distinct rows. -/
theorem rowIdx_injective (a b ps off off' : Nat) (hps : 0 < ps) (ho : off < ps) (ho' : off' < ps)
    (e : ps * a + off = ps * b + off') : a = b ∧ off = off' := by
  have h1 := Nat.mul_add_div hps a off
  have h2 := Nat.mul_add_div hps b off'
  rw [Nat.div_eq_of_lt ho] at h1
  rw [Nat.div_eq_of_lt ho'] at h2
  rw [e] at h1
  have hab : a = b := by omega
  subst hab
  exact ⟨rfl, by omega⟩

-- § group scales
/-- Scale slot of group g in a row. -/
def scaleIdx (row groups g : Nat) : Nat := groups * row + g

/-- Scale slots of different rows never collide (g < groups). -/
theorem scaleIdx_injective (r r' groups g g' : Nat) (hg : 0 < groups) (h1 : g < groups) (h2 : g' < groups)
    (e : scaleIdx r groups g = scaleIdx r' groups g') : r = r' ∧ g = g' := by
  unfold scaleIdx at e
  have := rowIdx_injective r r' groups g g' hg h1 h2 e
  exact ⟨this.1, this.2⟩

/-- head_dim a multiple of 64 gives whole 4-value quads. -/
theorem head_dim_quads (hd : Nat) (h : hd % 64 = 0) : hd % 4 = 0 := by
  omega

-- § gather index decomposition
/-- The destination slot (id, head, quad) equals the source thread index. -/
theorem gather_dst (id kvh per h q4 : Nat) :
    (id * kvh + h) * per + q4 = id * (kvh * per) + (h * per + q4) := by
  rw [Nat.add_mul, Nat.mul_assoc]
  omega

/-- The remainder splits into head and quad. -/
theorem gather_rem (rem per : Nat) : per * (rem / per) + rem % per = rem := by
  exact Nat.div_add_mod rem per

-- § saturation
/-- Clamp to the symmetric int8 range used by the cache. -/
def clamp127 (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- The stored code is always within [-127,127]. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  split_ifs <;> omega

/-- In-range codes pass through unchanged. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  split_ifs <;> omega

/-- A 256-thread grid covers every gather thread. -/
theorem gather_grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by
  omega

end Ref.KvQ8



## FILE: /mnt/user-data/outputs/NativeMmvqKernels.lean
/-!
# MODULE: native_mmvq_kernels
SOURCES: src/kernels/cuda/native_mmvq.cu
llama.cpp-pinned Q8_1 quantizer and MMVQ kernels (Q4_0/Q5_0/Q8_0/IQ4_NL/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS/Q2_0) for
ncols = 1..8, plus the dispatcher, support predicate and weight-size computation.

## Source semantics
Each launch uses 4 warps x 32 lanes = 128 threads. Thread `tid` starts at weight block `tid / T` and strides
by BPI blocks, with BPI * T = 128. Small-K shapes use 4 rows per CUDA block, otherwise 1. Q8_1 activations are
36-byte blocks of 32 values; ncols columns are contiguous so quantizing ncols*n_in values never straddles a
column. Shapes require n_in divisible by the block size and 1 <= ncols <= 8.

## Form template identifiers
bpi/tt -> "blocks per iteration / threads per block"; validShape -> "shape contract";
weightBytes -> "weight byte count".
-/
namespace Ref.NativeMmvq

-- § geometry
/-- Values per weight block for supported types. -/
def blockElems (t : Nat) : Nat :=
  if t = 2 ∨ t = 6 ∨ t = 8 ∨ t = 20 then 32 else if t = 42 then 64 else
  if t = 11 ∨ t = 12 ∨ t = 13 ∨ t = 14 ∨ t = 23 ∨ t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 29 then 256
  else 0

/-- Bytes per weight block. -/
def blockBytes (t : Nat) : Nat :=
  if t = 2 then 18 else if t = 6 then 22 else if t = 8 then 34 else if t = 20 then 18
  else if t = 11 then 110 else if t = 12 then 144 else if t = 13 then 176 else if t = 14 then 210
  else if t = 23 then 136 else if t = 42 then 18 else if t = 16 then 66 else if t = 17 then 74
  else if t = 18 then 98 else if t = 21 then 110 else if t = 22 then 82 else if t = 29 then 56 else 0

/-- `native_mmvq_supported`. -/
def supported (t : Nat) : Bool :=
  t == 2 || t == 6 || t == 8 || t == 11 || t == 12 || t == 13 || t == 14 || t == 20 || t == 23 ||
  t == 42 || t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- `native_mmvq_weight_bytes` without the validation. -/
def weightBytes (t nIn nOut : Nat) : Nat := nIn / blockElems t * blockBytes t * nOut

/-- Every supported type has a non-empty block geometry. -/
theorem supported_has_geometry (t : Nat) (ht : t < 64) (h : supported t = true) :
    0 < blockElems t ∧ 0 < blockBytes t := by
  revert t
  decide

/-- Weight bytes scale linearly in the number of output rows. -/
theorem weightBytes_add_rows (t nIn a b : Nat) :
    weightBytes t nIn (a + b) = weightBytes t nIn a + weightBytes t nIn b := by
  unfold weightBytes
  rw [Nat.mul_add]

-- § shape contract
/-- `validate_shape` as a decidable predicate. -/
def validShape (nIn ncols e : Nat) : Bool :=
  decide (0 < nIn ∧ nIn % e = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8)

/-- `native_q8_1_bytes`: 36-byte blocks of 32 values per column. -/
def q8Bytes (nIn ncols : Nat) : Nat := ncols * (nIn / 32) * 36

/-- A valid shape has a non-empty Q8_1 image. -/
theorem q8Bytes_pos (nIn ncols : Nat) (h : validShape nIn ncols 32 = true) : 0 < q8Bytes nIn ncols := by
  unfold validShape at h
  simp at h
  unfold q8Bytes
  have h1 : 0 < nIn / 32 := by omega
  have h2 : 0 < ncols * (nIn / 32) := Nat.mul_pos (by omega) h1
  omega

/-- Quantizing ncols columns as one vector never splits a 32-block across columns. -/
theorem q8_block_within_column (m i : Nat) : i / 32 / m = i / (32 * m) := by
  exact Nat.div_div_eq_div_mul i 32 m

/-- Rounding the quantizer grid up to 256 threads covers every element. -/
theorem quant_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by
  omega

-- § iteration traits
/-- Blocks per iteration of the ncols = 1 loop. -/
def bpi (t : Nat) : Nat :=
  if t = 13 ∨ t = 12 ∨ t = 11 then 8 else if t = 14 then 4 else if t = 23 then 16
  else if t = 42 then 64 else if t = 8 then 32 else 64

/-- Lanes cooperating on one weight block (`T`). -/
def tt (t : Nat) : Nat :=
  if t = 13 ∨ t = 12 ∨ t = 11 then 16 else if t = 14 then 32 else if t = 23 then 8
  else if t = 42 then 2 else if t = 8 then 4 else 2

/-- BPI * T equals the 128 threads of a CUDA block for every multi-warp format. -/
theorem bpi_mul_tt (t : Nat) (ht : t < 64)
    (h : t = 2 ∨ t = 6 ∨ t = 8 ∨ t = 11 ∨ t = 12 ∨ t = 13 ∨ t = 14 ∨ t = 20 ∨ t = 23 ∨ t = 42) :
    bpi t * tt t = 128 := by
  revert t
  decide

/-- Hence the starting block `tid / T` is always below the stride BPI: threads never alias a later iteration. -/
theorem start_block_lt_stride (B T tid : Nat) (h : B * T = 128) (ht : tid < 128) : tid / T < B := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  omega

/-- Small-K launches (4 rows per block) cover all rows. -/
theorem grid_small_covers (n : Nat) : n ≤ 4 * ((n + 3) / 4) ∧ 4 * ((n + 3) / 4) < n + 4 := by
  omega

/-- Upstream multi-column launches (2 rows per block) cover all rows. -/
theorem grid_upstream_covers (n : Nat) : n ≤ 2 * ((n + 1) / 2) ∧ 2 * ((n + 1) / 2) < n + 2 := by
  omega

/-- The shared cross-warp partial buffer fits 48 KiB for every ncols <= 8 (exact layout, 4 rows). -/
theorem shared_partial_fits (nc : Nat) (h : nc ≤ 8) : (4 - 1) * nc * 4 * 32 * 4 ≤ 49152 := by
  omega

-- § packed dot algebra
/-- `__dp4a` without accumulator. -/
def dp4 (a b : Fin 4 → Int) : Int := a 0 * b 0 + a 1 * b 1 + a 2 * b 2 + a 3 * b 3

/-- Affine Q4_0/Q5_0 correction: (v - k) . u = v . u - k * sum u. -/
theorem dp4_affine_correction (a b : Fin 4 → Int) (k : Int) :
    dp4 (fun i => a i - k) b = dp4 a b - k * (b 0 + b 1 + b 2 + b 3) := by
  unfold dp4
  simp only [Int.sub_mul, Int.mul_add]
  omega

end Ref.NativeMmvq



## FILE: /mnt/user-data/outputs/RouterTop10Select.lean
/-!
# MODULE: router_top10_select
SOURCES: src/kernels/cuda/router_top10.cu
MoE router: softmax over all experts, stable top-k selection (ties by ascending index), renormalization of the
selected weights with a 2^-14 lower clamp on the denominator.

## Source semantics
The selection order is "descending probability, ties by ascending index" - a strict total order. Block-wide
argmax picks the best candidate under this order and is independent of the reduction tree. The denominator is
max(sum, 2^-14) so division is safe. Launch: threads = min(n_expert,512) rounded up to 32; shared memory =
aligned taken-mask + n doubles + n floats; k <= 64, n_expert <= 512*64.

## Form template identifiers
before -> "ranking order"; pick -> "argmax combiner"; smemBytes -> "dynamic shared layout".
-/
namespace Ref.RouterTop10

-- § ranking order
/-- Expert f outranks expert e: larger probability, or equal probability and smaller index. -/
def before (p : Nat → Int) (e f : Nat) : Prop := p f > p e ∨ (p f = p e ∧ f < e)

/-- No expert outranks itself. -/
theorem before_irrefl (p : Nat → Int) (e : Nat) : ¬ before p e e := by
  unfold before
  omega

/-- The order is asymmetric. -/
theorem before_asymm (p : Nat → Int) (e f : Nat) (h : before p e f) : ¬ before p f e := by
  unfold before at *
  omega

/-- The order is transitive. -/
theorem before_trans (p : Nat → Int) (e f g : Nat) (h1 : before p e f) (h2 : before p f g) :
    before p e g := by
  unfold before at *
  omega

/-- Any two distinct experts are comparable, so ties never leave the selection ambiguous. -/
theorem before_total (p : Nat → Int) (e f : Nat) (h : e ≠ f) : before p e f ∨ before p f e := by
  unfold before
  omega

-- § argmax combiner
/-- Combine two (value, index) candidates exactly as the shuffle reduction does. -/
def pick (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The combiner is commutative on distinct indices, so the reduction tree shape cannot change the winner. -/
theorem pick_comm (a b : Int × Nat) (h : a.2 ≠ b.2) : pick a b = pick b a := by
  unfold pick
  split <;> split <;> first | rfl | (exfalso; omega)

/-- The sentinel (index n_expert, value -inf modeled as any value) never beats a real candidate with a larger value. -/
theorem pick_prefers_larger (a b : Int × Nat) (h : b.1 > a.1) : pick a b = b := by
  unfold pick
  rw [if_pos (Or.inl h)]

/-- On equal values the smaller index wins. -/
theorem pick_tie_smaller_index (a b : Int × Nat) (h : b.1 = a.1) (hi : b.2 < a.2) : pick a b = b := by
  unfold pick
  rw [if_pos (Or.inr ⟨h, hi⟩)]

-- § renormalization
/-- The clamped denominator `max(s, 2^-14)` (scaled by 2^14 to stay in Nat). -/
def clampDen (s : Nat) : Nat := max s 1

/-- The denominator is never zero. -/
theorem clampDen_pos (s : Nat) : 0 < clampDen s := by
  unfold clampDen
  omega

/-- The denominator never drops below the sum. -/
theorem clampDen_ge (s : Nat) : s ≤ clampDen s := by
  unfold clampDen
  omega

-- § launch geometry
/-- Threads per block: at least a warp-multiple covering min(n, 512). -/
def threads (n : Nat) : Nat := ((min n 512) + 31) / 32 * 32

/-- Thread count covers the experts up to 512 and is a multiple of 32. -/
theorem threads_props (n : Nat) : min n 512 ≤ threads n ∧ threads n % 32 = 0 ∧ threads n ≤ 512 := by
  unfold threads
  omega

/-- Bytes of the taken-mask (16-aligned). -/
def takenBytes (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Dynamic shared memory: mask, n doubles, n floats. -/
def smemBytes (n : Nat) : Nat := takenBytes n + n * 8 + n * 4

/-- The mask is 16-aligned, so the double array after it is 8-aligned and the float array 4-aligned. -/
theorem smem_alignment (n : Nat) : takenBytes n % 16 = 0 ∧ (takenBytes n) % 8 = 0 ∧ (takenBytes n + n * 8) % 4 = 0 := by
  unfold takenBytes
  omega

/-- For 512 experts the dynamic shared size is 6656 bytes. -/
theorem smem_512 : smemBytes 512 = 6656 := by
  decide

/-- The mask covers every expert. -/
theorem taken_covers (n : Nat) : n ≤ takenBytes n := by
  unfold takenBytes
  omega

end Ref.RouterTop10



## FILE: /mnt/user-data/outputs/S2GemvFastExperiment.lean
/-!
# MODULE: s2_gemv_fast_experiment
SOURCES: src/kernels/cuda/s2_gemv_fast.cu
A measured-neutral/negative S2 (2-bit) GEMV experiment: block per output row, optional shared-memory staging of
the activation, and a constant-memory byte -> four-code lookup table. Kept as a record, not a fast path.

## Source semantics
Codes are 2 bits LSB-first, four per byte; the LUT entry for byte b lists (code_k - 1) for k = 0..3. One scale per
64 elements, so quad q (elements 4q..4q+3) uses scale q >> 4. Shared staging needs n_in <= 4096 halves; the
partial-sum array begins right after the 4096 halves (2048 floats). n_in must be a multiple of 64.

## Form template identifiers
lut -> "byte decode table"; scaleIdx -> "per-64 group scale index"; smemBytes -> "dynamic shared size".
-/
namespace Ref.S2GemvFast

-- § lookup table
/-- LUT value: the k-th 2-bit field of byte b, minus the bias 1. -/
def lut (b k : Nat) : Int := ((b / 4 ^ k) % 4 : Nat) - 1

/-- Every LUT value lies in {-1, 0, 1, 2}. -/
theorem lut_range (b k : Nat) : -1 ≤ lut b k ∧ lut b k ≤ 2 := by
  have h := Nat.mod_lt (b / 4 ^ k) (by decide : 4 > 0)
  unfold lut
  omega

/-- The four fields of a byte reassemble the byte. -/
theorem byte_fields_reassemble (b : Nat) (hb : b < 256) :
    b % 4 + 4 * (b / 4 % 4) + 16 * (b / 16 % 4) + 64 * (b / 64 % 4) = b := by
  omega

/-- Scale index of quad q (elements 4q..4q+3, 64 per group) is q >> 4. -/
theorem scale_index_quad (q : Nat) : (q * 4) / 64 = q / 16 := by
  omega

-- § shared memory budget
/-- Max halves staged in shared memory. -/
def maxSharedHalves : Nat := 4096

/-- Dynamic shared bytes: staged halves (if any) plus one float per thread. -/
def smemBytes (stage : Bool) (tpr : Nat) : Nat :=
  (if stage then maxSharedHalves * 2 else 0) + tpr * 4

/-- The partial array offset (`MAX/2` floats) is exactly the end of the staged halves. -/
theorem partial_offset_after_halves : (maxSharedHalves / 2) * 4 = maxSharedHalves * 2 := by
  decide

/-- Staging a 2560-element activation fits the staging area. -/
theorem stage_fits_2560 : 2560 ≤ maxSharedHalves := by
  decide

/-- With staging, the shared size is at least the staged halves. -/
theorem smem_with_stage (tpr : Nat) : 8192 ≤ smemBytes true tpr := by
  unfold smemBytes maxSharedHalves
  simp

/-- Multiple of 64 implies multiple of 4 (the quad requirement is redundant). -/
theorem mult64_implies_mult4 (n : Nat) (h : n % 64 = 0) : n % 4 = 0 := by
  omega

/-- Quads cover the row exactly. -/
theorem quads_cover (n : Nat) (h : n % 4 = 0) : n / 4 * 4 = n := by
  omega

end Ref.S2GemvFast



## FILE: /mnt/user-data/outputs/S2GroupedExperts.lean
/-!
# MODULE: s2_grouped_experts
SOURCES: src/kernels/cuda/s2_expert_grouped.cu
Grouped GPU MoE experts over Q2_0 blobs (gate/up/down), warp per output row with dp4a against Q8_0 activations;
device-side hit selection and compaction; grouped variants; CPU-order bit-exact variant.

## Source semantics
Blob geometry (H = 2560, FF = 640): gate/up codes 2*FF rows of 640 B, down codes H rows of 160 B, then fp16
scales. A weight code c in {0..3} decodes to c - 1, so sum (c-1)x = sum cx - sum x (two dp4a). Gate/up row slots
are interleaved (even = gate, odd = up) and written to a gate-major output: gate at e*FF + r, up at
cap*FF + e*FF + r with r = slot/2. Hit selection compacts resident experts in lane order by popcount of lower
lanes.

## Form template identifiers
O_* -> "blob section offsets"; gmIdx -> "gate-major output index"; rank -> "ballot prefix rank".
-/
namespace Ref.S2Grouped

-- § blob geometry
/-- Hidden size. -/
def H : Nat := 2560
/-- Expert FFN width. -/
def FF : Nat := 640
/-- Bytes of codes per gate/up row. -/
def rowGu : Nat := H / 4
/-- Bytes of codes per down row. -/
def rowD : Nat := FF / 4
/-- fp16 scales per gate/up row. -/
def scGu : Nat := H / 64
/-- fp16 scales per down row. -/
def scD : Nat := FF / 64
/-- Offset of the down codes. -/
def oDCodes : Nat := 2 * FF * rowGu
/-- Offset of the gate/up scales. -/
def oGuScales : Nat := oDCodes + H * rowD
/-- Offset of the down scales. -/
def oDScales : Nat := oGuScales + 2 * FF * scGu * 2
/-- Total blob bytes. -/
def blobBytes : Nat := oDScales + H * scD * 2

/-- The literal row sizes of the source. -/
theorem row_sizes : rowGu = 640 ∧ rowD = 160 ∧ scGu = 40 ∧ scD = 10 := by
  decide

/-- The literal section offsets. -/
theorem section_offsets : oDCodes = 819200 ∧ oGuScales = 1228800 ∧ oDScales = 1331200 := by
  decide

/-- Total blob size. -/
theorem blob_bytes_value : blobBytes = 1382400 := by
  decide

/-- Sections are laid out in increasing order. -/
theorem sections_ordered : oDCodes < oGuScales ∧ oGuScales < oDScales ∧ oDScales < blobBytes := by
  decide

/-- 80 chunks of 32 split into 3 lane passes (GU_CHUNKS), and a down row has 20 chunks, under one warp. -/
theorem chunk_counts : (H / 32 + 31) / 32 = 3 ∧ FF / 32 = 20 ∧ FF / 32 < 32 := by
  decide

-- § bias identity
/-- `__dp4a` without accumulator. -/
def dp4 (a b : Fin 4 → Int) : Int := a 0 * b 0 + a 1 * b 1 + a 2 * b 2 + a 3 * b 3

/-- sum (c - 1) x = sum c x - sum x, the identity behind the (s - hx) form. -/
theorem bias_identity (c x : Fin 4 → Int) :
    dp4 (fun i => c i - 1) x = dp4 c x - (x 0 + x 1 + x 2 + x 3) := by
  unfold dp4
  simp only [Int.sub_mul, Int.mul_add, Int.mul_one, Int.one_mul]
  omega

-- § gate-major output layout
/-- Output index of gate/up slot `i` of entry `h` among `n` entries. -/
def gmIdx (n h i : Nat) : Nat :=
  if i % 2 = 0 then h * 640 + i / 2 else n * 640 + h * 640 + i / 2

/-- Every output index is inside the 2*n*FF buffer. -/
theorem gmIdx_lt (n h i : Nat) (hh : h < n) (hi : i < 1280) : gmIdx n h i < 2 * n * 640 := by
  unfold gmIdx
  split <;> omega

/-- The interleaved-to-gate-major map is injective: no two (entry, slot) pairs collide. -/
theorem gmIdx_injective (n h i h' i' : Nat) (hh : h < n) (hh' : h' < n) (hi : i < 1280) (hi' : i' < 1280)
    (e : gmIdx n h i = gmIdx n h' i') : h = h' ∧ i = i' := by
  unfold gmIdx at e
  split at e <;> split at e <;> omega

/-- Gate values of consecutive entries are contiguous (what `swiglu` and the quantizer read). -/
theorem gmIdx_gate_contiguous (n h r : Nat) : gmIdx n h (2 * r) = h * 640 + r := by
  unfold gmIdx
  split <;> omega

-- § device compaction by ballot prefix
/-- Number of hits strictly before position i (popcount of lower lanes). -/
def rank (p : Nat → Bool) : Nat → Nat
  | 0 => 0
  | i + 1 => rank p i + (if p i = true then 1 else 0)

/-- Rank never decreases by one step. -/
theorem rank_le_succ (p : Nat → Bool) (i : Nat) : rank p i ≤ rank p (i + 1) := by
  show rank p i ≤ rank p i + (if p i = true then 1 else 0)
  split <;> omega

/-- Rank is monotone. -/
theorem rank_mono_add (p : Nat → Bool) (i d : Nat) : rank p i ≤ rank p (i + d) := by
  induction d with
  | zero => exact Nat.le_refl _
  | succ d ih => exact Nat.le_trans ih (rank_le_succ p (i + d))

/-- A hit's compacted slot is strictly below the slot of every later position: compaction preserves order
and gives distinct slots to distinct hits. -/
theorem rank_strict (p : Nat → Bool) (i j : Nat) (hp : p i = true) (hij : i < j) : rank p i < rank p j := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_lt hij
  have h1 : rank p (i + 1) = rank p i + 1 := by
    show rank p i + (if p i = true then 1 else 0) = _
    rw [if_pos hp]
  have h2 := rank_mono_add p (i + 1) d
  have e : i + d + 1 = i + 1 + d := by omega
  rw [e]
  omega

end Ref.S2Grouped



## FILE: /mnt/user-data/outputs/SFamilyGemv.lean
/-!
# MODULE: s_family_gemv
SOURCES: src/kernels/cuda/s_gemv.cu
S-family GEMV (2/4/8-bit packed codes with per-group fp32 scale and optional offset) against fp16, Q8_K or Q8_0
activations; naive, split (block per row) and warp-per-row kernels.

## Source semantics
A row of n_in codes of width b occupies n_in / (8/b) bytes with n_in / group_elems scales. The weight is
decoded as `decode(code) * scale + offset` before the activation multiply. Group size is a power of two (passed
as its logarithm). Hoisting the activation block out of a 16-element lane-iteration is valid because the
iteration start is a multiple of 16 and blocks are 256 (Q8_K) or 32 (Q8_0) wide. Q8_K needs n_in % 256 = 0;
Q8_0 only n_in % 32 = 0, which is why a Q8_0 kernel exists (n_in = 640).

## Form template identifiers
perByte -> "codes per byte"; quad lemmas -> "consecutive-element packing"; q8k/q8_0 constants -> "activation block".
-/
namespace Ref.SGemv

-- § layout
/-- Codes per byte for a code width. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- Bytes of one four-code word (`QW`): 1, 2 or 4. -/
def wordBytes (bits : Nat) : Nat := 4 * bits / 8

/-- The three supported widths pack 4, 2 and 1 codes per byte. -/
theorem perByte_values : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by
  decide

/-- Four-code words are 1, 2 and 4 bytes wide. -/
theorem wordBytes_values : wordBytes 2 = 1 ∧ wordBytes 4 = 2 ∧ wordBytes 8 = 4 := by
  decide

/-- Q8_K block geometry: 292 bytes per 256 elements. -/
def q8kBytes : Nat := 292

/-- Q8_0 block geometry: 34 bytes per 32 elements. -/
def q8_0Bytes : Nat := 34

/-- `ffn_down_shexp` (n_in = 640) is a Q8_0-shaped input but structurally cannot be Q8_K. -/
theorem q8k_impossible_640 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by
  decide

-- § quad packing and hoisting
/-- 4-bit codes: a quad starting at a multiple of 4 spans bytes i/2 and i/2+1 with in-byte slot k%2. -/
theorem quad_bytes_s4 (i k : Nat) (hi : i % 4 = 0) (hk : k < 4) :
    (i + k) / 2 = i / 2 + k / 2 ∧ (i + k) % 2 = k % 2 := by
  omega

/-- 2-bit codes: a quad is exactly one byte, slot k. -/
theorem quad_bytes_s2 (i k : Nat) (hi : i % 4 = 0) (hk : k < 4) :
    (i + k) / 4 = i / 4 ∧ (i + k) % 4 = k := by
  omega

/-- 8-bit codes: byte index is the element index. -/
theorem quad_bytes_s8 (i k : Nat) : (i + k) / 1 = i + k := by
  omega

/-- A quad inside one group shares its scale (group size 4m). -/
theorem quad_in_group (i k m : Nat) (hi : i % 4 = 0) (hk : k < 4) : (i + k) / (4 * m) = i / (4 * m) := by
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul]
  have : (i + k) / 4 = i / 4 := by omega
  rw [this]

/-- Q8_K hoisting: sixteen consecutive elements from a multiple of 16 stay in one 256-block. -/
theorem hoist_q8k (i k : Nat) (hi : i % 16 = 0) (hk : k < 16) :
    (i + k) / 256 = i / 256 ∧ (i + k) % 256 = i % 256 + k := by
  omega

/-- Q8_0 hoisting: sixteen consecutive elements from a multiple of 16 stay in one 32-block. -/
theorem hoist_q8_0 (i k : Nat) (hi : i % 16 = 0) (hk : k < 16) :
    (i + k) / 32 = i / 32 ∧ (i + k) % 32 = i % 32 + k := by
  omega

/-- Lane-iteration starts `lane*16 + 512*j` are multiples of 16. -/
theorem lane_start_mult16 (lane j : Nat) : (lane * 16 + 512 * j) % 16 = 0 := by
  omega

/-- Using a shift for the group index equals division by the (power-of-two) group size. -/
theorem group_shift_eq_div (i s : Nat) : i >>> s = i / 2 ^ s := by
  exact Nat.shiftRight_eq_div_pow i s

/-- Group sizes defined by the format are powers of two. -/
def isPow2 (n : Nat) : Bool := n != 0 && (n &&& (n - 1)) == 0

/-- 16, 32, 64 pass the power-of-two guard; 48 is refused. -/
theorem pow2_guard : isPow2 16 = true ∧ isPow2 32 = true ∧ isPow2 64 = true ∧ isPow2 48 = false := by
  decide

-- § launch coverage
/-- Thread-per-row launch (128 threads) covers all rows. -/
theorem grid_thread_per_row (n : Nat) : n ≤ ((n + 127) / 128) * 128 := by
  omega

/-- Warp-per-row launch (256 threads, 8 warps) covers all rows. -/
theorem grid_warp_per_row (n : Nat) : n ≤ ((n + 7) / 8) * 8 := by
  omega

-- § codebook
/-- The non-linear IQ4 codebook (kvalues_iq4nl). -/
def kIq4Nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries (one per 4-bit code). -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by
  decide

/-- The codebook is strictly increasing. -/
theorem kIq4Nl_sorted : kIq4Nl.Pairwise (· < ·) := by
  decide

end Ref.SGemv



## FILE: /mnt/user-data/outputs/SharedExpertBlock.lean
/-!
# MODULE: shared_expert_block
SOURCES: src/kernels/cuda/shared_expert.cu
The MoE shared expert (silu on gate, times up, down projection, scaled by a per-token scalar sigmoid gate) and
the final routed-plus-shared combination.

## Source semantics
h = silu(x Wg) * (x Wu); out = (h Wd) * sigmoid(x . w_gate_inp). The routed output is sum_e w_e * part_e plus the
shared output added plain (not router-weighted). The down input is quantized to the down weight's own activation
format: Q8_K needs n_ff % 256 = 0, otherwise Q8_0. Scratch is carved from caller memory in 16-byte aligned
regions: gate | up | q8_0 | q8k | g. Native paths take 1..8 tokens; moe_combine refuses k > 64.

## Form template identifiers
align16 -> "scratch alignment"; scratchBytes -> "shared expert scratch"; dotAcc -> "routed weighted sum".
-/
namespace Ref.SharedExpert

-- § scratch carving
/-- Round up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16
  omega

/-- Alignment never shrinks. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16
  omega

/-- `shared_expert_scratch_bytes`. -/
def scratchBytes (nFf : Nat) : Nat :=
  align16 (nFf * 4) * 2 + align16 (nFf / 32 * 34) + align16 (nFf / 256 * 292) + 32

/-- Region offsets in carving order. -/
def offUp (nFf : Nat) : Nat := align16 (nFf * 4)
/-- Offset of the Q8_0 image. -/
def offQ80 (nFf : Nat) : Nat := align16 (nFf * 4) * 2
/-- Offset of the Q8_K image. -/
def offQ8k (nFf : Nat) : Nat := align16 (nFf * 4) * 2 + align16 (nFf / 32 * 34)
/-- Offset of the scalar gate. -/
def offG (nFf : Nat) : Nat := align16 (nFf * 4) * 2 + align16 (nFf / 32 * 34) + align16 (nFf / 256 * 292)

/-- Every carved region starts 16-byte aligned. -/
theorem offsets_aligned (nFf : Nat) :
    offUp nFf % 16 = 0 ∧ offQ80 nFf % 16 = 0 ∧ offQ8k nFf % 16 = 0 ∧ offG nFf % 16 = 0 := by
  unfold offUp offQ80 offQ8k offG align16
  omega

/-- The scalar gate slot plus its 32-byte tail ends exactly at the scratch size. -/
theorem scratch_end (nFf : Nat) : offG nFf + 32 = scratchBytes nFf := by
  unfold offG scratchBytes
  rfl

/-- For n_ff = 640 the scratch is 6432 bytes. -/
theorem scratch_640 : scratchBytes 640 = 6432 := by
  decide

/-- n_ff = 640 is a multiple of 32 but not 256: Q8_K is structurally impossible for the down input. -/
theorem down_q8k_impossible_640 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by
  decide

-- § native path guards
/-- Token-count guard of `shared_expert_multi`. -/
def tokensOk (n : Nat) : Bool := decide (1 ≤ n ∧ n ≤ 8)

/-- Valid token counts are exactly 1..8. -/
theorem tokensOk_iff (n : Nat) : tokensOk n = true ↔ (1 ≤ n ∧ n ≤ 8) := by
  unfold tokensOk
  simp

/-- A native projection is used when any of gate/up/down is natively supported. -/
def nativeProjection (g u d : Bool) : Bool := g || u || d

/-- Without any native projection the legacy SwiGLU is used. -/
theorem legacy_when_none : nativeProjection false false false = false := by
  decide

-- § combination
/-- Weighted accumulation `acc += w_e * part_e` in order. -/
def dotAcc : Int → List Int → List Int → Int
  | acc, [], _ => acc
  | acc, _ :: _, [] => acc
  | acc, w :: ws, p :: ps => dotAcc (acc + w * p) ws ps

/-- Starting accumulator offset passes straight through: this is why the shared output is added plain. -/
theorem dotAcc_shift (ws ps : List Int) (a c : Int) : dotAcc (a + c) ws ps = dotAcc a ws ps + c := by
  induction ws generalizing ps a with
  | nil => simp [dotAcc]
  | cons w ws ih =>
    cases ps with
    | nil => simp [dotAcc]
    | cons p ps =>
      have e : a + c + w * p = a + w * p + c := by omega
      simp only [dotAcc]
      rw [e]
      exact ih ps (a + w * p)

/-- Final combination: routed sum plus the shared output. -/
def combine (ws ps : List Int) (shared : Int) : Int := dotAcc shared ws ps

/-- The combination equals the routed sum plus the shared output. -/
theorem combine_eq (ws ps : List Int) (shared : Int) : combine ws ps shared = dotAcc 0 ws ps + shared := by
  unfold combine
  have h := dotAcc_shift ws ps 0 shared
  rw [Int.zero_add] at h
  exact h

/-- With no routed experts the output is exactly the shared output. -/
theorem combine_nil (ps : List Int) (shared : Int) : combine [] ps shared = shared := by
  simp [combine, dotAcc]

/-- The k <= 64 guard. -/
def kOk (k : Nat) : Bool := decide (0 < k ∧ k ≤ 64)

/-- Ten routed experts pass the guard; 65 is refused. -/
theorem kOk_cases : kOk 10 = true ∧ kOk 65 = false := by
  decide

end Ref.SharedExpert





## FILE: /mnt/user-data/outputs/F16BitsConversion.lean
/-!
# MODULE: f16_bits_conversion
SOURCES: include/ref/kernels/f16_bits.hpp
Hand-written IEEE binary16 conversions (`f16_from_f32` rounding to nearest even, `f32_from_f16` exact),
plus the bf16 round-to-nearest-even trick used in `ple.cu`.

## Source semantics
* Rebiased exponent is `rawexp - 127 + 15`. Raw exponent 255 stays inf/NaN; finite values with rebiased
  exponent >= 31 saturate to inf; rebiased exponent <= 0 enters the subnormal path; below -10 it flushes to
  a signed zero.
* Rounding adds one to the truncated mantissa when the discarded remainder exceeds half, or equals half
  and the truncated value is odd. A carry may bump the exponent.
* Decoding subnormals normalises the mantissa with `s` left shifts; the f32 exponent field is `113 - s`.

## Form template identifiers
`f16_from_f32` -> `encode_half`; `f32_from_f16` -> `decode_half`; `man` -> `mantissa`; `sh` -> `shift`;
`rem` -> `discarded_remainder`.
-/

namespace Ref.F16

-- § exponent classification

/-- Rebiased exponent `rawexp - 127 + 15`. -/
def rebias (rawexp : Nat) : Int := (rawexp : Int) - 127 + 15

/-- Finite overflow to infinity happens exactly when the raw exponent is at least 143. -/
theorem overflow_iff (r : Nat) : rebias r ≥ 31 ↔ 143 ≤ r := by
  unfold rebias
  omega

/-- The subnormal path is taken exactly when the raw exponent is at most 112. -/
theorem subnormal_iff (r : Nat) : rebias r ≤ 0 ↔ r ≤ 112 := by
  unfold rebias
  omega

/-- Flush to a signed zero happens exactly when the raw exponent is at most 101. -/
theorem underflow_iff (r : Nat) : rebias r < -10 ↔ r ≤ 101 := by
  unfold rebias
  omega

/-- In the subnormal path the shift `14 - exp` stays within 14..24, so the 32-bit shifts are defined. -/
theorem subnormal_shift_range (r : Nat) (h1 : 102 ≤ r) (h2 : r ≤ 112) :
    14 ≤ 14 - rebias r ∧ 14 - rebias r ≤ 24 := by
  unfold rebias
  omega

-- § round to nearest even

/-- Round-to-nearest-even step: bump `h` when the remainder is above half, or at half with `h` odd. -/
def rne (h rem half : Nat) : Nat :=
  if rem > half ∨ (rem = half ∧ h % 2 = 1) then h + 1 else h

/-- On an exact tie the rounded result is even. -/
theorem rne_tie_even (h rem half : Nat) (e : rem = half) : rne h rem half % 2 = 0 := by
  unfold rne
  split <;> omega

/-- Below half the value is truncated. -/
theorem rne_below (h rem half : Nat) (hl : rem < half) : rne h rem half = h := by
  unfold rne
  split <;> omega

/-- Above half the value is bumped. -/
theorem rne_above (h rem half : Nat) (hg : rem > half) : rne h rem half = h + 1 := by
  unfold rne
  split <;> omega

/-- Rounding moves the value by at most one unit upwards. -/
theorem rne_bounds (h rem half : Nat) : h ≤ rne h rem half ∧ rne h rem half ≤ h + 1 := by
  unfold rne
  split <;> omega

-- § normal path

/-- For a 23-bit mantissa the kept part fits in 10 bits and the remainder in 13 bits. -/
theorem normal_split (man : Nat) (h : man < 8388608) :
    man / 8192 < 1024 ∧ man % 8192 < 8192 ∧ man = 8192 * (man / 8192) + man % 8192 := by
  omega

/-- When the kept mantissa is all ones, the rounding carry bumps the exponent field by one. -/
theorem carry_bumps_exponent (e : Nat) : (e * 1024 + 1023) + 1 = (e + 1) * 1024 := by
  omega

/-- The largest finite exponent plus a rounding carry lands exactly on the infinity pattern 0x7C00. -/
theorem carry_to_inf : (30 * 1024 + 1023) + 1 = 0x7C00 := by decide

-- § subnormal path

/-- At the deepest subnormal shift (24) an input in `[2^23, 2^24)` truncates to zero and rounds to the
smallest subnormal exactly when it is strictly above the halfway point. -/
theorem subnormal_min_boundary (man : Nat) (h1 : 8388608 ≤ man) (h2 : man < 16777216) :
    man / 16777216 = 0 ∧ (rne 0 man 8388608 = 1 ↔ man > 8388608) := by
  unfold rne
  split <;> omega

-- § exact decode of subnormals

/-- Number of left shifts needed until the leading one reaches bit 10 (fuel-bounded). -/
def normShift : Nat → Nat → Nat
  | _, 0 => 0
  | m, fuel + 1 => if 1024 ≤ m then 0 else 1 + normShift (2 * m) fuel

/-- The shift count never exceeds the fuel. -/
theorem normShift_le (m fuel : Nat) : normShift m fuel ≤ fuel := by
  induction fuel generalizing m with
  | zero => simp [normShift]
  | succ n ih =>
    simp only [normShift]
    split
    · omega
    · have := ih (2 * m)
      omega

/-- The smallest subnormal mantissa 1 needs ten shifts. -/
theorem normShift_one : normShift 1 10 = 10 := by decide

/-- A mantissa already holding bit 9 needs a single shift. -/
theorem normShift_512 : normShift 512 10 = 1 := by decide

/-- With at most ten shifts the f32 exponent field `113 - s` stays in range 103..113. -/
theorem decode_exponent_range (s : Nat) (hs : s ≤ 10) : 103 ≤ 113 - s ∧ 113 - s ≤ 113 := by
  omega

/-- The smallest subnormal decodes to exponent field 103, i.e. 2^(103-127) = 2^-24. -/
theorem smallest_subnormal_exponent : 113 - 10 + 24 = 127 := by decide

-- § bf16 rounding used by ple.cu

/-- `(i + lsb + 0x7FFF) >> 16` with `lsb` the lowest kept bit equals round-to-nearest-even of `i / 65536`. -/
theorem bf16_rounding (i : Nat) :
    (i + (i / 65536) % 2 + 32767) / 65536 =
      (if i % 65536 > 32768 ∨ (i % 65536 = 32768 ∧ (i / 65536) % 2 = 1)
        then i / 65536 + 1 else i / 65536) := by
  split <;> omega

end Ref.F16



## FILE: /mnt/user-data/outputs/GdnStepContracts.lean
/-!
# MODULE: gdn_step_contracts
SOURCES: include/ref/kernels/gdn.hpp
Gated delta-net layer pieces between the projections: the delta-rule state step, the causal conv step with
carried state, the L2 norm, the beta sigmoid and the closing per-head norm.

## Source semantics
* Real geometry: S = 128, h_k = 16, h_v = 48, conv channels C = 2*S*h_k + S*h_v = 10240, d_conv = 4;
  48 GDN layers and 12 QSA layers.
* State layout is (S, H_v, S) = (i, h, j) with j fastest: flat index `(i * h_v + h) * S + j`.
* Head pairing is MODULO (`h % h_k`), not interleaved.
* Recurrence per column: decay first (`st = dec * state`), `sk = k . st`, `d = (v - sk) * beta`,
  `state' = st + k * d`, `o = state' . q`. The kernel form `g*s + k*((v - g*(s*k))*beta)` is the same number.
* Conv: output `sum_i inp[i] * kW[c*d_conv + i]` over `inp = [conv_state | x]`; the new state is the last
  `d_conv - 1` entries of `inp`.
* L2 norm adds eps to the squared norm (no division by the width); the closing norm is per head row.

## Form template identifiers
`gdn_step` -> `delta_rule_step`; `gdn_conv_step` -> `causal_conv_step`; `S` -> `head_dim`; `h_k` -> `n_key_heads`;
`h_v` -> `n_value_heads`; `conv_state` -> `conv_carry`; `kW` -> `conv_weight`.
-/

namespace Ref.Gdn

-- § geometry

/-- Conv channels for the real artifact: q and k (16 heads of 128 each) plus v (48 heads of 128). -/
theorem conv_channels_real : 2 * 128 * 16 + 128 * 48 = 10240 := by decide

/-- The value heads are a whole multiple of the key heads (3 value heads per key head). -/
theorem heads_multiple : 48 % 16 = 0 ∧ 48 / 16 = 3 := by decide

/-- Layer split of the artifact: 48 GDN layers and 12 QSA layers. -/
theorem layer_split : 48 + 12 = 60 := by decide

/-- Conv weight has `channels * d_conv` entries. -/
theorem conv_weight_len_real : 10240 * 4 = 40960 := by decide

-- § state layout (S, H_v, S), j fastest

/-- Flat index of state element `(i, h, j)`. -/
def stateIdx (S hv i h j : Nat) : Nat := (i * hv + h) * S + j

/-- Walking `j` is contiguous: the coalescing property the layout was chosen for. -/
theorem stateIdx_contiguous (S hv i h j : Nat) : stateIdx S hv i h (j + 1) = stateIdx S hv i h j + 1 := by
  unfold stateIdx
  omega

/-- Every valid element lands inside the `S * H_v * S` buffer. -/
theorem stateIdx_lt (S hv i h j : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) :
    stateIdx S hv i h j < S * hv * S := by
  unfold stateIdx
  have e1 : (i + 1) * hv = i * hv + hv := by rw [Nat.add_mul, Nat.one_mul]
  have h1 : (i + 1) * hv ≤ S * hv := Nat.mul_le_mul_right hv hi
  have e2 : (i * hv + h + 1) * S = (i * hv + h) * S + S := by rw [Nat.add_mul, Nat.one_mul]
  have h2 : (i * hv + h + 1) * S ≤ (S * hv) * S := Nat.mul_le_mul_right S (by omega)
  omega

/-- The row-group addressing of the multi-token kernel, `base + r * (h_v * S)` with
`base = ((g*32) * h_v + head) * S + col`, is the same element as `stateIdx` of row `g*32 + r`. -/
theorem row_group_address (g r hv head S col : Nat) :
    ((g * 32) * hv + head) * S + col + r * (hv * S) = ((g * 32 + r) * hv + head) * S + col := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

-- § head pairing

/-- Modulo pairing always names a valid key head. -/
theorem pair_mod_lt (h hk : Nat) (hpos : 0 < hk) : h % hk < hk := Nat.mod_lt h hpos

/-- The two conventions disagree on a real head: value head 1 pairs with key head 1 (modulo) but 0 (interleave). -/
theorem pairing_distinguishable : (1 % 16 : Nat) ≠ 1 / (48 / 16) := by decide

-- § delta-rule recurrence

/-- Kernel form of one scalar state update (decay applied inside the delta). -/
def stepKernel (g b s k v : Int) : Int := g * s + k * ((v - g * (s * k)) * b)

/-- Reference form: decay first, then the rank-one update. -/
def stepRef (g b s k v : Int) : Int := (g * s) + k * ((v - k * (g * s)) * b)

/-- The two forms are the same number. -/
theorem kernel_eq_ref (g b s k v : Int) : stepKernel g b s k v = stepRef g b s k v := by
  unfold stepKernel stepRef
  have h : g * (s * k) = k * (g * s) := by
    rw [Int.mul_comm s k]
    exact Int.mul_left_comm g k s
  rw [h]

/-- With zero write strength the step is pure decay. -/
theorem step_beta_zero (g s k v : Int) : stepKernel g 0 s k v = g * s := by
  unfold stepKernel
  omega

/-- With no decay, unit key and unit write strength the delta rule overwrites the state with the value. -/
theorem step_overwrite (s v : Int) : stepKernel 1 1 s 1 v = v := by
  unfold stepKernel
  omega

-- § causal conv step

/-- Input column `i` of `[conv_state | x]` with `d - 1` carried columns. -/
def convInp {α : Type} (st : Nat → α) (x : α) (d i : Nat) : α := if i < d - 1 then st i else x

/-- The new state entry `j` is input column `j + 1`: the oldest column is dropped. -/
def convNext {α : Type} (st : Nat → α) (x : α) (d j : Nat) : α := convInp st x d (j + 1)

/-- Tap 0 reads the OLDEST carried column. -/
theorem convInp_oldest {α : Type} (st : Nat → α) (x : α) (d : Nat) (hd : 2 ≤ d) :
    convInp st x d 0 = st 0 := by
  unfold convInp
  have hc : 0 < d - 1 := by omega
  rw [if_pos hc]

/-- The newest input lands in the LAST state slot. -/
theorem convNext_last {α : Type} (st : Nat → α) (x : α) (d : Nat) (hd : 2 ≤ d) :
    convNext st x d (d - 2) = x := by
  unfold convNext convInp
  have hc : ¬ (d - 2 + 1 < d - 1) := by omega
  rw [if_neg hc]

/-- Every other state slot is shifted down by one column. -/
theorem convNext_shift {α : Type} (st : Nat → α) (x : α) (d j : Nat) (h : j + 2 < d) :
    convNext st x d j = st (j + 1) := by
  unfold convNext convInp
  have hc : j + 1 < d - 1 := by omega
  rw [if_pos hc]

/-- Weight index `c * d + i` (tap `i` of channel `c`, `i` fastest) stays inside `channels * d`. -/
theorem convWeightIdx_lt (c d i channels : Nat) (hc : c < channels) (hi : i < d) :
    c * d + i < channels * d := by
  have e : (c + 1) * d = c * d + d := by rw [Nat.add_mul, Nat.one_mul]
  have h : (c + 1) * d ≤ channels * d := Nat.mul_le_mul_right d hc
  omega

-- § norms

/-- Row-major element `h * S + d` of a `(h_v, S)` tensor stays in range, so the closing norm is per head row. -/
theorem headRowIdx_lt (h S d hv : Nat) (hh : h < hv) (hd : d < S) : h * S + d < hv * S := by
  have e : (h + 1) * S = h * S + S := by rw [Nat.add_mul, Nat.one_mul]
  have h1 : (h + 1) * S ≤ hv * S := Nat.mul_le_mul_right S hh
  omega

/-- L2 norm vs the mean-based misreading: the sum-based denominator differs from the mean-based one
whenever the squared norm is non-zero (the true factor is the width, 128 here). -/
theorem l2_sum_vs_mean (sumsq : Nat) (h : 0 < sumsq) : sumsq / 128 < sumsq := by
  omega

end Ref.Gdn



## FILE: /mnt/user-data/outputs/NativeRouterTop10.lean
/-!
# MODULE: native_router_top10
SOURCES: src/kernels/cuda/native_router.cu
Single-block CUDA router: softmax over 512 F32 logits spread over 32 lanes x 16 slots, ten rounds of
arg-max with a deterministic tie rule, then renormalisation of the ten selected weights.

## Source semantics
* Expert `e` lives at lane `e % 32`, slot `e / 32` (`e = lane + 32 * slot`).
* A candidate beats another when its value is larger, or equal with a smaller expert id.
* The destination spans (`logits` 2048 B, `ids` 40 B, `weights` 40 B) must be 4-byte aligned, non-null,
  non-wrapping and pairwise disjoint (half-open interval test).
* The selected-weight sum is clamped below by 6.103515625e-5 before the reciprocal is taken.

## Form template identifiers
logits -> `scores`; ids -> `selected_ids`; weights -> `selected_weights`; `route` -> `router_topk_kernel`;
`valid` -> `span_valid`; `overlap` -> `span_overlap`; 512 -> `n_expert`; 10 -> `k_out`.
-/

namespace Ref.NativeRouter

-- § span validity and overlap

/-- Model of `valid(p, bytes)`: non-null, 4-byte aligned, and `bytes <= UINTPTR_MAX - address`. -/
def SpanValid (addr bytes maxAddr : Nat) : Prop :=
  addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ maxAddr - addr

/-- A valid span cannot wrap past the top of the address space. -/
theorem spanValid_no_wrap (addr bytes maxAddr : Nat) (h : SpanValid addr bytes maxAddr)
    (hle : addr ≤ maxAddr) : addr + bytes ≤ maxAddr := by
  obtain ⟨_, _, h3⟩ := h
  omega

/-- A valid span is 4-byte aligned. -/
theorem spanValid_aligned (addr bytes maxAddr : Nat) (h : SpanValid addr bytes maxAddr) :
    addr % 4 = 0 := by
  obtain ⟨_, h2, _⟩ := h
  exact h2

/-- Model of `overlap(a, an, b, bn)`: the half-open interval test used by the launcher. -/
def Overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- The overlap test is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : Overlap a an b bn ↔ Overlap b bn a an := by
  unfold Overlap
  omega

/-- Spans that touch end to end do not overlap. -/
theorem overlap_adjacent (a an bn : Nat) : ¬ Overlap a an (a + an) bn := by
  unfold Overlap
  omega

/-- If the first span ends before the second begins, they are disjoint. -/
theorem overlap_of_lt (a an b bn : Nat) (h : a + an ≤ b) : ¬ Overlap a an b bn := by
  unfold Overlap
  omega

/-- A non-empty span overlaps itself. -/
theorem overlap_self (a n : Nat) (hn : 0 < n) : Overlap a n a n := by
  unfold Overlap
  omega

/-- Quirk of the formula: a zero-length span strictly inside another one is reported as overlapping. -/
theorem overlap_zero_length_quirk : Overlap 5 0 3 10 := by
  unfold Overlap
  omega

-- § geometry

/-- The 512 experts are exactly 32 lanes times 16 slots. -/
theorem expert_count : (512 : Nat) = 32 * 16 := by decide

/-- The pinned block geometry is 32 x 8 = 256 threads, of which row zero (32 threads) is active. -/
theorem block_threads : (32 : Nat) * 8 = 256 := by decide

/-- Ten output ranks fit in one warp, so `lane < 10` can hold one weight each. -/
theorem outputs_fit_warp : (10 : Nat) ≤ 32 := by decide

/-- `lane + 32 * slot` is a valid expert id. -/
theorem expert_lt (lane slot : Nat) (hl : lane < 32) (hs : slot < 16) : lane + 32 * slot < 512 := by
  omega

/-- The owning lane of `lane + 32 * slot` is `lane`. -/
theorem expert_lane (lane slot : Nat) (hl : lane < 32) : (lane + 32 * slot) % 32 = lane := by
  omega

/-- The slot of `lane + 32 * slot` is `slot`. -/
theorem expert_slot (lane slot : Nat) (hl : lane < 32) : (lane + 32 * slot) / 32 = slot := by
  omega

/-- Every expert id decomposes into a lane and a slot. -/
theorem expert_roundtrip (e : Nat) (he : e < 512) :
    e % 32 + 32 * (e / 32) = e ∧ e / 32 < 16 ∧ e % 32 < 32 := by
  omega

/-- Exactly one lane owns a given expert, so the `ids[rank]` write has a single writer. -/
theorem unique_owner (e l l' : Nat) (h : e % 32 = l) (h' : e % 32 = l') : l = l' := by
  omega

-- § tie-breaking order

/-- `Beats v i w j`: candidate `(v, i)` is preferred to `(w, j)`: larger value, or equal and smaller id. -/
abbrev Beats (v i w j : Nat) : Prop := v > w ∨ (v = w ∧ i < j)

/-- No candidate beats itself. -/
theorem beats_irrefl (v i : Nat) : ¬ Beats v i v i := by
  unfold Beats
  omega

/-- The preference order is asymmetric. -/
theorem beats_asymm (v i w j : Nat) : Beats v i w j → ¬ Beats w j v i := by
  intro h1 h2
  unfold Beats at h1 h2
  omega

/-- The preference order is transitive. -/
theorem beats_trans (a i b j c k : Nat) : Beats a i b j → Beats b j c k → Beats a i c k := by
  intro h1 h2
  unfold Beats at h1 h2 ⊢
  omega

/-- Two candidates with different ids are always comparable, so the arg-max is deterministic. -/
theorem beats_total (v w i j : Nat) (h : i ≠ j) : Beats v i w j ∨ Beats w j v i := by
  unfold Beats
  omega

/-- The butterfly combine step used in the shuffle reduction. -/
def pick (a b : Nat × Nat) : Nat × Nat :=
  if Beats b.1 b.2 a.1 a.2 then b else a

/-- Combining is commutative for candidates with distinct ids, so the shuffle order cannot matter. -/
theorem pick_comm (a b : Nat × Nat) (h : a.2 ≠ b.2) : pick a b = pick b a := by
  unfold pick
  by_cases h1 : Beats b.1 b.2 a.1 a.2
  · have h2 : ¬ Beats a.1 a.2 b.1 b.2 := beats_asymm _ _ _ _ h1
    rw [if_pos h1, if_neg h2]
  · have h2 : Beats a.1 a.2 b.1 b.2 := by
      rcases beats_total a.1 b.1 a.2 b.2 h with h3 | h3
      · exact h3
      · exact absurd h3 h1
    rw [if_neg h1, if_pos h2]

-- § weight normalisation

/-- The clamped selected-weight sum is at least the clamp constant. -/
theorem clamp_ge (s m : Nat) : m ≤ max s m := by omega

/-- With a positive clamp constant the denominator is never zero. -/
theorem clamp_pos (s m : Nat) (hm : 0 < m) : 0 < max s m := by omega

end Ref.NativeRouter



## FILE: /mnt/user-data/outputs/PleBlockLayout.lean
/-!
# MODULE: ple_block_layout
SOURCES: src/kernels/cuda/ple.cu
GPU half of the PLE block: key/value projections, grouped RMS norm, gate, broadcast, depthwise causal dilated
convolution over a carried history, and the three-way residual sum, all inside one caller-owned scratch arena.

## Source semantics
* The history is a per-channel column of `NG_HIST` entries; `ple_history_advance` drops the oldest and
  appends the new normalised value.
* Conv tap `k` reads row `nhist - (kern-1-k)*dil` of `[history | new]`; only tap `kern-1` reads the new row.
  The history is row-fastest: flat index `row + nhist * channel`.
* Stream `c` of the grouped norm occupies `[c*n_embd, (c+1)*n_embd)`; the gate is broadcast with
  `value[i % n_embd] * gate[i / n_embd]`.
* The scratch arena is five hc_dim buffers + `n_embd` + `hc` floats, then the Q8 activation image, then a
  BF16 embedding copy, each region 16-byte aligned, plus 256 bytes of slack.

## Form template identifiers
`ple_block` -> `ple_layer`; `hist_rows` -> `conv_history`; `d_scratch` -> `float_workspace`;
`NG_HC_DIM` -> `hc_dim`; `NG_N_EMBD` -> `n_embd`; `NG_HC` -> `n_stream`; `NG_HIST` -> `n_hist`.
-/

namespace Ref.Ple

-- § history advance (indexed form of "drop oldest, append newest")

/-- Entry `i` of the advanced column: shifted down by one, with the new value at the end. -/
def advIdx {α : Type} (col : Nat → α) (n : Nat) (x : α) (i : Nat) : α :=
  if i + 1 < n then col (i + 1) else x

/-- Every non-final slot receives its successor's value. -/
theorem advIdx_shift {α : Type} (col : Nat → α) (n : Nat) (x : α) (i : Nat) (h : i + 1 < n) :
    advIdx col n x i = col (i + 1) := by
  unfold advIdx
  rw [if_pos h]

/-- The last slot receives the new normalised value. -/
theorem advIdx_last {α : Type} (col : Nat → α) (n : Nat) (x : α) (hn : 1 ≤ n) :
    advIdx col n x (n - 1) = x := by
  unfold advIdx
  have hc : ¬ (n - 1 + 1 < n) := by omega
  rw [if_neg hc]

-- § conv taps and the history layout

/-- Row read by a tap that is `d = kern-1-k` positions before the newest one. -/
def convRow (nhist dil d : Nat) : Nat := nhist - d * dil

/-- Real geometry (kern 4, dilation 3, history 9): the four taps read rows 0, 3, 6 and 9. -/
theorem convRow_real : [0, 1, 2, 3].map (fun k => convRow 9 3 (4 - 1 - k)) = [0, 3, 6, 9] := by
  decide

/-- With positive dilation only the newest tap (`d = 0`) reads the new row `nhist`. -/
theorem convRow_eq_new (n dil d : Nat) (hdil : 0 < dil) (hle : d * dil ≤ n) :
    convRow n dil d = n ↔ d = 0 := by
  unfold convRow
  constructor
  · intro h
    by_contra hd
    have : 0 < d * dil := Nat.mul_pos (Nat.pos_of_ne_zero hd) hdil
    omega
  · intro h
    subst h
    simp

/-- Row-fastest flat index of the conv state. -/
def histIdx (row nhist c : Nat) : Nat := row + nhist * c

/-- A valid (row, channel) pair stays inside `nhist * hc` entries. -/
theorem histIdx_lt (row nhist c hc : Nat) (hr : row < nhist) (hcc : c < hc) :
    histIdx row nhist c < nhist * hc := by
  unfold histIdx
  have h1 : nhist * (c + 1) ≤ nhist * hc := Nat.mul_le_mul_left nhist (by omega)
  have h2 : nhist * (c + 1) = nhist * c + nhist := by rw [Nat.mul_add, Nat.mul_one]
  omega

/-- The row-fastest layout is injective: reading it channel-slowest would be a transposed state. -/
theorem histIdx_inj (n r r' c c' : Nat) (hn : 0 < n) (hr : r < n) (hr' : r' < n)
    (h : histIdx r n c = histIdx r' n c') : r = r' ∧ c = c' := by
  unfold histIdx at h
  have h1 : (r + n * c) % n = r := by
    rw [Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt hr
  have h2 : (r' + n * c') % n = r' := by
    rw [Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt hr'
  have h3 : (r + n * c) / n = c := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt hr]
    simp
  have h4 : (r' + n * c') / n = c' := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt hr']
    simp
  constructor
  · calc r = (r + n * c) % n := h1.symm
      _ = (r' + n * c') % n := by rw [h]
      _ = r' := h2
  · calc c = (r + n * c) / n := h3.symm
      _ = (r' + n * c') / n := by rw [h]
      _ = c' := h4

-- § grouped norm and broadcast indexing

/-- Real geometry: 4 streams of 2560 make hc_dim = 10240. -/
theorem hc_dim_real : 4 * 2560 = 10240 := by decide

/-- Broadcast index: element `i < n_embd * hc` reads gate `i / n_embd`, which is a valid stream. -/
theorem bcast_gate_lt (i n_embd hc : Nat) (h : i < n_embd * hc) : i / n_embd < hc := by
  have h' : i < n_embd * hc := h
  exact Nat.div_lt_of_lt_mul h'

/-- Broadcast index: the value index `i % n_embd` is in range when `n_embd > 0`. -/
theorem bcast_value_lt (i n_embd : Nat) (h : 0 < n_embd) : i % n_embd < n_embd :=
  Nat.mod_lt i h

/-- The block-sum scratch of 8 doubles holds one entry per warp of a 256-thread block. -/
theorem block_sum_scratch : (256 + 31) / 32 = 8 := by decide

-- § scratch arena

/-- Round up to a multiple of 16: `(x + 15) & ~15`. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by
  unfold align16
  omega

/-- Alignment gives a multiple of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16
  omega

/-- Alignment adds fewer than 16 bytes. -/
theorem align16_lt (x : Nat) : align16 x < x + 16 := by
  unfold align16
  omega

/-- Alignment is idempotent. -/
theorem align16_idem (x : Nat) : align16 (align16 x) = align16 x := by
  unfold align16
  omega

/-- Total arena size from the three region sizes (floats, Q8 image, BF16 embedding) plus slack. -/
def scratchTotal (f q e : Nat) : Nat := align16 f + align16 q + e + 256

/-- `ple_block_scratch_bytes` for a given geometry. -/
def scratchBytes (nEmbd hc hcDim : Nat) : Nat :=
  scratchTotal ((5 * hcDim + nEmbd + hc) * 4) ((nEmbd / 32) * 34) (nEmbd * 2)

/-- The real geometry needs 223152 bytes of scratch. -/
theorem scratchBytes_real : scratchBytes 2560 4 10240 = 223152 := by decide

/-- The float region ends before the Q8 activation image begins. -/
theorem float_region_before_act (f : Nat) : f ≤ align16 f := align16_ge f

/-- The Q8 image ends before the BF16 embedding copy begins. -/
theorem act_region_before_emb (q : Nat) : q ≤ align16 q := align16_ge q

/-- The BF16 embedding copy fits inside the arena, with the 256-byte slack to spare. -/
theorem emb_region_fits (f q e : Nat) : align16 f + align16 q + e + 256 ≤ scratchTotal f q e + 0 := by
  unfold scratchTotal
  omega

/-- The float sub-buffers tile the float region exactly: key, query, norm, gated, conv, value, gate. -/
theorem float_buffers_tile (hcDim nEmbd hc : Nat) :
    5 * hcDim + nEmbd + hc = hcDim + hcDim + hcDim + hcDim + hcDim + nEmbd + hc := by
  omega

end Ref.Ple



## FILE: /mnt/user-data/outputs/QsaSelectionAndAttention.lean
/-!
# MODULE: qsa_selection_and_attention
SOURCES: include/ref/kernels/qsa.hpp, include/ref/kernels/qsa_select.hpp, include/ref/kernels/qsa_decode_attn.hpp
Decode-time QSA layer: paged FP16 KV cache, pooled indexer keys, per-cell scores, top-k selection (identity
below the width), gather through the page table, and split-K softmax attention over the selected cells.

## Source semantics
* Selection width is `min(n_kv, idx_top_k + idx_block - 1)`; the extra `r - 1` covers the incomplete tail.
* The per-token step record is `[pos, n_kv = pos+1, n_bid = n_kv / r, width]`.
* Cell `j` scores through block `min(j / r, n_bid)`; cells `j >= (n_kv / r) * r` are the incomplete tail and
  get the +1e9 bias.
* KV pool row for cell `t`, head `h`: `(page * n_head_kv + h) * page_size + t % page_size`, one head's cells
  contiguous inside a page.
* GQA maps query head `q` to KV head `q / (n_head / n_head_kv)` (integer division).
* The gate half of each head's `2*head_dim` block starts at offset `head_dim`.
* Split-K attention uses `ceil(cap / CHUNK)` chunks; INT8 pools carry one fp16 scale per 64 values.

## Form template identifiers
`QsaShapes` -> `attn_geometry`; `qsa_step_fill` -> `fill_step_record`; `topk_512` -> `select_cells`;
`kv_gather` -> `gather_kv`; `page_size` -> `page_cells`; `idx_block` -> `pool_width`; `CHUNK` -> `split_chunk`.
-/

namespace Ref.Qsa

-- § selection width and step record

/-- Selection width: `min(n_kv, idx_top_k + idx_block - 1)`. -/
def selWidth (nKv topK r : Nat) : Nat := min nKv (topK + r - 1)

/-- The selection never returns more cells than the cache holds. -/
theorem selWidth_le_nkv (nKv topK r : Nat) : selWidth nKv topK r ≤ nKv := by
  unfold selWidth
  omega

/-- Below the bound the selection is the identity (every cell). -/
theorem selWidth_identity (nKv topK r : Nat) (h : nKv ≤ topK + r - 1) : selWidth nKv topK r = nKv := by
  unfold selWidth
  omega

/-- Real geometry: top-k 2048, block 4 gives bound 2051 (the plan's 2048 is conservative by three). -/
theorem selWidth_bound_real : 2048 + 4 - 1 = 2051 := by decide

/-- The widest selection the single-block kernel can ever produce (`kTopkMaxCells = 32768`) is 2051. -/
theorem selWidth_cap_real : selWidth 32768 2048 4 = 2051 := by decide

/-- The step record written for position `pos`. -/
def stepRecord (pos topK r : Nat) : List Nat := [pos, pos + 1, (pos + 1) / r, selWidth (pos + 1) topK r]

/-- Four entries, i.e. `kStepCount`, and `qsa_step_bytes` is 16. -/
theorem stepRecord_length (pos topK r : Nat) : (stepRecord pos topK r).length = 4 := rfl

/-- The step buffer is 16 bytes. -/
theorem step_bytes : 4 * 4 = 16 := by decide

/-- Completed blocks times the block width never exceed the cache size. -/
theorem blocks_le_nkv (nKv r : Nat) : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r

/-- The incomplete tail has fewer than `r` cells: why the budget adds `r - 1`. -/
theorem tail_count_lt (n r : Nat) (hr : 0 < r) : n - (n / r) * r < r := by
  have h1 := Nat.div_add_mod n r
  have h2 := Nat.mod_lt n hr
  have h3 : r * (n / r) = (n / r) * r := Nat.mul_comm _ _
  omega

-- § cell to block mapping (idx_block = 4)

/-- Block used to score cell `j`. -/
def cellBlock (j nBid : Nat) : Nat := min (j / 4) nBid

/-- A completed-region cell maps to its own block. -/
theorem cellBlock_complete (j nKv : Nat) (h : j < nKv / 4 * 4) : cellBlock j (nKv / 4) = j / 4 := by
  unfold cellBlock
  omega

/-- A tail cell maps to the spare slot `n_bid`. -/
theorem cellBlock_tail (j nKv : Nat) (h1 : nKv / 4 * 4 ≤ j) (h2 : j < nKv) :
    cellBlock j (nKv / 4) = nKv / 4 := by
  unfold cellBlock
  omega

/-- The mapped block never exceeds the spare slot. -/
theorem cellBlock_le (j nBid : Nat) : cellBlock j nBid ≤ nBid := by
  unfold cellBlock
  omega

/-- The spare block has cells exactly when `n_kv` is not a multiple of 4. -/
theorem tail_has_cells_iff (nKv : Nat) : nKv / 4 * 4 < nKv ↔ nKv % 4 ≠ 0 := by
  omega

/-- Cells per scored block: `r` for completed blocks, the remainder for the spare one. -/
def cellsInBlock (b nBid nKv r : Nat) : Nat := if b < nBid then r else nKv - nBid * r

/-- Weights of all `n_bid + 1` blocks add up to `n_kv` (completed blocks plus the tail). -/
theorem block_weights_total (nKv r : Nat) : (nKv / r) * r + cellsInBlock (nKv / r) (nKv / r) nKv r = nKv := by
  unfold cellsInBlock
  have h := Nat.div_mul_le_self nKv r
  have hc : ¬ (nKv / r < nKv / r) := by omega
  rw [if_neg hc]
  omega

/-- The radix select makes 4 passes of 8 bits over a 32-bit key. -/
theorem radix_passes : 4 * 8 = 32 := by decide

-- § paged KV layout

/-- Physical row of `(page, head, offset)`: `(page * n_head_kv + head) * page_size + offset`. -/
def physRow (page nHeadKv head pageSize off : Nat) : Nat := (page * nHeadKv + head) * pageSize + off

/-- Consecutive cells of one head are contiguous inside a page. -/
theorem physRow_contiguous (page nHeadKv head pageSize off : Nat) :
    physRow page nHeadKv head pageSize (off + 1) = physRow page nHeadKv head pageSize off + 1 := by
  unfold physRow
  omega

/-- A valid (page, head, offset) is inside the pool of `n_pages * n_head_kv * page_size` rows. -/
theorem physRow_lt (page nHeadKv head pageSize off nPages : Nat) (hp : page < nPages) (hh : head < nHeadKv)
    (ho : off < pageSize) : physRow page nHeadKv head pageSize off < nPages * nHeadKv * pageSize := by
  unfold physRow
  have e1 : (page + 1) * nHeadKv = page * nHeadKv + nHeadKv := by rw [Nat.add_mul, Nat.one_mul]
  have h1 : (page + 1) * nHeadKv ≤ nPages * nHeadKv := Nat.mul_le_mul_right nHeadKv hp
  have e2 : (page * nHeadKv + head + 1) * pageSize = (page * nHeadKv + head) * pageSize + pageSize := by
    rw [Nat.add_mul, Nat.one_mul]
  have h2 : (page * nHeadKv + head + 1) * pageSize ≤ (nPages * nHeadKv) * pageSize :=
    Nat.mul_le_mul_right pageSize (by omega)
  omega

/-- A logical cell splits into page and offset. -/
theorem cell_split (t ps : Nat) : ps * (t / ps) + t % ps = t := Nat.div_add_mod t ps

/-- With page size 1 (the unpaged configuration) the offset is always zero. -/
theorem unpaged_offset (t : Nat) : t % 1 = 0 := Nat.mod_one t

-- § memory budget (documented figures)

/-- 12 QSA layers x 2 heads x 256 dims x (K and V) x 2 B x 32768 cells = 805306368 bytes. -/
theorem kv_bytes_32k : 12 * 2 * 256 * 2 * 2 * 32768 = 805306368 := by decide

/-- FINDING: that is exactly 768 MiB (about 805 MB); the header's "805 MiB" is a decimal-megabyte figure. -/
theorem kv_is_768_mib : 805306368 = 768 * 1048576 := by decide

/-- Indexer keys: 128 floats = 512 B per block, 128 B per cell per layer, 1536 B per cell over 12 layers. -/
theorem indexer_bytes_per_cell : 128 * 4 / 4 * 12 = 1536 := by decide

/-- At 32K cells the indexer keys take 50331648 bytes, exactly 48 MiB (the header says "50 MiB"). -/
theorem indexer_bytes_32k : 1536 * 32768 = 50331648 ∧ 50331648 = 48 * 1048576 := by decide

-- § grouped-query attention head map (24 query heads, 2 KV heads)

/-- Each KV head serves 12 query heads. -/
theorem gqa_group : 24 / 2 = 12 := by decide

/-- Integer-division map gives the contiguous halves [0 x 12, 1 x 12]. -/
theorem gqa_div_counts :
    ((List.range 24).filter (fun q => q / 12 = 0)).length = 12 ∧
    ((List.range 24).filter (fun q => q / 12 = 1)).length = 12 := by
  decide

/-- The modulo map would pair query head 1 with KV head 1, but the integer map pairs it with KV head 0. -/
theorem gqa_mod_differs : (1 % 2 : Nat) ≠ 1 / 12 := by decide

-- § gate split of the q projection (n_head 24, head_dim 256)

/-- Within a head's 512-wide block the query half precedes the gate half. -/
theorem gate_after_query (h d d' : Nat) (hd : d < 256) : h * 512 + d < h * 512 + 256 + d' := by
  omega

/-- Every gate element lies inside the `24 * 512` projection. -/
theorem gate_idx_lt (h d : Nat) (hh : h < 24) (hd : d < 256) : h * 512 + 256 + d < 24 * 512 := by
  omega

-- § split-K decode attention

/-- Number of chunks for a capacity `cap` and chunk size `c`. -/
def nChunks (cap c : Nat) : Nat := (cap + c - 1) / c

/-- The chunks cover the whole capacity. -/
theorem chunks_cover (cap c : Nat) (hc : 0 < c) : cap ≤ nChunks cap c * c := by
  unfold nChunks
  have h1 := Nat.div_add_mod (cap + c - 1) c
  have h2 := Nat.mod_lt (cap + c - 1) hc
  have h3 : c * ((cap + c - 1) / c) = ((cap + c - 1) / c) * c := Nat.mul_comm _ _
  omega

/-- Every cell below `cap` falls into a chunk below the chunk count. -/
theorem cell_chunk_lt (j cap c : Nat) (hc : 0 < c) (hj : j < cap) : j / c < nChunks cap c := by
  have h1 := chunks_cover cap c hc
  have h2 : j < c * nChunks cap c := by
    rw [Nat.mul_comm]
    omega
  exact Nat.div_lt_of_lt_mul h2

/-- INT8 pools carry one fp16 scale per 64 values: four per 256-wide row. -/
theorem scales_per_row : 256 / 64 = 4 := by decide

end Ref.Qsa



## FILE: /mnt/user-data/outputs/SGemvAndExpertGrouped.lean
/-!
# MODULE: s_gemv_and_expert_grouped
SOURCES: include/ref/kernels/s_gemv.hpp, include/ref/kernels/s2_expert_grouped.hpp
Canonical-form GEMV (one decode `cb[code] * scale + offset` for S2/S4/S8 codes, over fp16, Q8_K or Q8_0
activations) and the grouped GPU expert tier (one launch per projection, hit lists passed as device data).

## Source semantics
* A weight row is `n_in / (8 / code_bits)` code bytes and `n_in / group_elems` scales.
* Q8_K activation blocks are 292 bytes per 256 elements (`4 + 256 + 2*16`); Q8_0 blocks are 34 bytes per 32
  elements (`2 + 32`). `n_in = 640` is a multiple of 32 but not of 256, so Q8_K is impossible for it.
* `act_kind` cannot be derived from the other form attributes (IQ4_NL vs IQ4_XS differ in nothing else).
* Grouped expert evaluation: scratch is `n_hits * (2*FF*4 + (FF/32)*34)` bytes; four launches per layer
  instead of five per expert; hit `h` reads slot `slot_index[h]` of the arena and writes row `dst_index[h]`.
* `moe_hit_select` keeps the routed entries whose residency row is non-negative.

## Form template identifiers
`SForm` -> `quant_form`; `code_bits` -> `bits_per_code`; `group_elems` -> `group_size`; `act_kind` -> `act_format`;
`moe_hit_grouped_s2` -> `grouped_expert_eval`; `slot_index` -> `resident_slot`; `dst_index` -> `route_row`.
-/

namespace Ref.SGemv

-- § canonical form

/-- The canonical-form attributes the kernels receive (codebook: 0 affine, 1 IQ4_NL table). -/
structure SForm where
  codeBits : Nat
  codeBias : Int
  groupElems : Nat
  codebook : Nat
  hasOffset : Bool
  actKind : Nat
  deriving DecidableEq

/-- Well-formed forms use 2, 4 or 8 bit codes, a positive group and an activation kind of 0 or 1. -/
def SForm.WellFormed (f : SForm) : Prop :=
  (f.codeBits = 2 ∨ f.codeBits = 4 ∨ f.codeBits = 8) ∧ 0 < f.groupElems ∧ f.actKind ≤ 1

/-- Codes per byte. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- Two-, four- and eight-bit codes pack 4, 2 and 1 per byte. -/
theorem perByte_values : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by decide

/-- Code bytes in one weight row. -/
def rowCodeBytes (nIn bits : Nat) : Nat := nIn / perByte bits

/-- Scales in one weight row. -/
def rowScales (nIn group : Nat) : Nat := nIn / group

/-- The expert gate/up shape: 2560 inputs give 640 code bytes (S2) and 40 scales (group 64) per row. -/
theorem expert_row_sizes : rowCodeBytes 2560 2 = 640 ∧ rowScales 2560 64 = 40 := by decide

/-- The integer-domain decode `code + bias` lies in `[bias, bias + M)` for codes below `M = 2^bits`. -/
theorem decode_range (code M : Nat) (bias : Int) (h : code < M) :
    bias ≤ (code : Int) + bias ∧ (code : Int) + bias < bias + (M : Int) := by
  omega

/-- Q2_0 (bias -1) decodes 2-bit codes to -1..2. -/
theorem q2_range (code : Nat) (h : code < 4) : -1 ≤ (code : Int) + (-1) ∧ (code : Int) + (-1) ≤ 2 := by
  omega

/-- Q5_0 (bias -16) decodes 5-bit values to -16..15. -/
theorem q5_range (code : Nat) (h : code < 32) : -16 ≤ (code : Int) + (-16) ∧ (code : Int) + (-16) ≤ 15 := by
  omega

-- § activation formats

/-- Bytes of a Q8_K activation block: scale, 256 int8 quants, 16 int16 block sums. -/
theorem q8k_block_bytes : 4 + 256 + 2 * 16 = 292 := by decide

/-- Bytes of a Q8_0 block: fp16 scale plus 32 int8 quants. -/
theorem q8_0_block_bytes : 2 + 32 = 34 := by decide

/-- Q8_0 activation image size for `n` elements. -/
def q8_0Bytes (n : Nat) : Nat := n / 32 * 34

/-- The 2560-wide image used by the PLE and expert kernels is 2720 bytes; the 640-wide one is 680. -/
theorem q8_0_sizes : q8_0Bytes 2560 = 2720 ∧ q8_0Bytes 640 = 680 := by decide

/-- `ffn_down_shexp` has `n_in = 640`: a multiple of 32 but not of 256, so Q8_K cannot apply. -/
theorem shexp_needs_q8_0 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- Group sizes 32, 64 and 256 are multiples of four and powers of two, as the split kernels require. -/
theorem groups_ok : (32 % 4 = 0 ∧ ∃ k, 32 = 2 ^ k) ∧ (64 % 4 = 0 ∧ ∃ k, 64 = 2 ^ k) ∧
    (256 % 4 = 0 ∧ ∃ k, 256 = 2 ^ k) :=
  ⟨⟨by decide, 5, by decide⟩, ⟨by decide, 6, by decide⟩, ⟨by decide, 8, by decide⟩⟩

/-- `act_kind` is not derivable: two IQ4 forms identical except for it are different values. -/
theorem act_kind_independent :
    ({ codeBits := 4, codeBias := 0, groupElems := 32, codebook := 1, hasOffset := false, actKind := 0 } : SForm) ≠
    { codeBits := 4, codeBias := 0, groupElems := 32, codebook := 1, hasOffset := false, actKind := 1 } := by
  decide

-- § grouped expert tier

/-- Scratch bytes per hit: gate/up activations (`2*FF` floats) plus the quantised intermediate. -/
def perHitScratch (ff : Nat) : Nat := 2 * ff * 4 + (ff / 32) * 34

/-- Scratch for `n` hits. -/
def hitScratch (n ff : Nat) : Nat := n * perHitScratch ff

/-- Scratch grows by exactly one per-hit amount per extra hit. -/
theorem hitScratch_succ (n ff : Nat) : hitScratch (n + 1) ff = hitScratch n ff + perHitScratch ff := by
  unfold hitScratch
  rw [Nat.add_mul, Nat.one_mul]

/-- Real geometry (FF = 640): 5800 bytes per hit, 58000 for ten hits. -/
theorem hitScratch_real : perHitScratch 640 = 5800 ∧ hitScratch 10 640 = 58000 := by decide

/-- Grouping: 4 launches per layer over 48 layers is 192, versus about 1548 launches per token (x100) for
5 launches per expert at 6.45 hits per layer. -/
theorem launch_counts : 4 * 48 = 192 ∧ 5 * 645 * 48 = 154800 := by decide

/-- Spans of consecutive slots (or consecutive output rows) of `n` bytes/floats are disjoint. -/
theorem slot_spans_disjoint (d d' n : Nat) (h : d < d') : d * n + n ≤ d' * n := by
  have e : (d + 1) * n = d * n + n := by rw [Nat.add_mul, Nat.one_mul]
  have h1 : (d + 1) * n ≤ d' * n := Nat.mul_le_mul_right n h
  omega

/-- Hits selected from `k` routed ids: those whose residency row is non-negative. -/
def hitCount (res : Nat → Int) (ids : List Nat) : Nat := (ids.filter (fun e => decide (0 ≤ res e))).length

/-- There are never more hits than routed entries, so a capacity of `k` hits suffices. -/
theorem hitCount_le (res : Nat → Int) (ids : List Nat) : hitCount res ids ≤ ids.length := by
  unfold hitCount
  exact List.length_filter_le _ _

/-- No routed entries means no hits. -/
theorem hitCount_nil (res : Nat → Int) : hitCount res [] = 0 := by
  simp [hitCount]

end Ref.SGemv



## FILE: /mnt/user-data/outputs/SamplerChain.lean
/-!
# MODULE: sampler_chain
SOURCES: src/kernels/cuda/sampler.cu, include/ref/kernels/sampler.hpp
Device sampler for one token per row: repeat/frequency/presence penalties over a history tail, greedy arg-max
(one block per token, strided scan plus warp/block reduction), top-k by repeated maximum scans, top-p cut,
temperature, and a counter-based Philox uniform draw.

## Source semantics
* Greedy (`greedy` or `temperature <= 0`) is an arg-max over the PENALISED logits, ties to the lowest index.
  Temperature must not be applied there: `inv_t = 0` for `temperature <= 0` would zero every logit.
* Repeat penalty multiplies non-positive logits and divides positive ones; the presence penalty is a boolean
  (applied once), the frequency penalty scales with the count. Nothing happens for count 0.
* The history window is the TAIL of the row: the last `min(penalty_last_n, history_len)` entries.
* The Philox uniform keeps 24 mantissa bits, so it lies in [0, 1).
* `top_k` is clamped to at most 64 on the sampled path; `top_p` keeps at least `min_keep` survivors.

## Form template identifiers
`sample_tokens` -> `sample_rows`; `SamplerParams` -> `sampling_config`; `history` -> `token_history`;
`sampler_greedy_kernel` -> `argmax_kernel`; `philox_uniform` -> `counter_uniform`; `KMAX` -> `top_k_cap`.
-/

namespace Ref.Sampler

-- § penalties

/-- Penalty model over integers: repeat (multiply if non-positive, divide otherwise), then frequency and
presence. A zero count leaves the logit untouched. -/
def applyPenalty (logit : Int) (count : Nat) (rep freq present : Int) : Int :=
  if count = 0 then logit
  else (if logit ≤ 0 then logit * rep else logit / rep) - ((count : Int) * freq + 1 * present)

/-- A token that never occurred is not penalised. -/
theorem penalty_zero_count (logit rep freq present : Int) :
    applyPenalty logit 0 rep freq present = logit := by
  unfold applyPenalty
  simp

/-- The presence penalty is a boolean: with no frequency penalty the count does not matter. -/
theorem presence_is_boolean (logit rep present : Int) (c₁ c₂ : Nat) (h₁ : 0 < c₁) (h₂ : 0 < c₂) :
    applyPenalty logit c₁ rep 0 present = applyPenalty logit c₂ rep 0 present := by
  have n₁ : c₁ ≠ 0 := by omega
  have n₂ : c₂ ≠ 0 := by omega
  unfold applyPenalty
  rw [if_neg n₁, if_neg n₂]
  simp

/-- A negative logit is multiplied: with repeat penalty 2, -3 becomes -6 (less likely). -/
theorem penalty_negative_multiplies : applyPenalty (-3) 1 2 0 0 = -6 := by decide

/-- A positive logit is divided: with repeat penalty 2, 6 becomes 3. -/
theorem penalty_positive_divides : applyPenalty 6 1 2 0 0 = 3 := by decide

/-- Dividing a negative logit unconditionally would RAISE it (-3 / 2 = -2 > -6): the inversion the source avoids. -/
theorem naive_division_inverts : (-6 : Int) < (-3) / 2 := by decide

-- § history window

/-- Length of the penalty window: clamp of `penalty_last_n` into `[0, history_len]`. -/
def winLen (plen hl : Int) : Int := max 0 (min plen hl)

/-- The window length is non-negative. -/
theorem winLen_nonneg (plen hl : Int) : 0 ≤ winLen plen hl := by
  unfold winLen
  omega

/-- The window never exceeds the history. -/
theorem winLen_le (plen hl : Int) (h : 0 ≤ hl) : winLen plen hl ≤ hl := by
  unfold winLen
  omega

/-- The window is the tail: it starts at `hl - len` and ends exactly at the end of the row. -/
theorem winStart_tail (plen hl : Int) : (hl - winLen plen hl) + winLen plen hl = hl := by
  omega

-- § greedy arg-max with lowest-index tie rule

/-- Preference order: larger score, or equal score with smaller index. -/
abbrev Beats (v i w j : Nat) : Prop := v > w ∨ (v = w ∧ i < j)

/-- Asymmetry of the preference order. -/
theorem beats_asymm (v i w j : Nat) : Beats v i w j → ¬ Beats w j v i := by
  intro h1 h2
  unfold Beats at h1 h2
  omega

/-- Transitivity of the preference order. -/
theorem beats_trans (a i b j c k : Nat) : Beats a i b j → Beats b j c k → Beats a i c k := by
  intro h1 h2
  unfold Beats at h1 h2 ⊢
  omega

/-- Totality for distinct indices, which makes the strided scan plus reduction order-independent. -/
theorem beats_total (v w i j : Nat) (h : i ≠ j) : Beats v i w j ∨ Beats w j v i := by
  unfold Beats
  omega

/-- The "no candidate" sentinel (index `n_vocab`, score at most any real score) loses to every real index. -/
theorem sentinel_loses (v w i n : Nat) (hw : w ≤ v) (hi : i < n) : Beats v i w n := by
  unfold Beats
  omega

/-- The final answer is a valid index whenever the vocabulary is non-empty (`-inf` ties answer 0). -/
theorem final_index_lt (wi n : Nat) (hn : 0 < n) : (if wi < n then wi else 0) < n := by
  split <;> omega

/-- A 32-lane warp is folded by `shfl_down` offsets 16, 8, 4, 2, 1, which together span all 31 other lanes. -/
theorem warp_fold_span : 16 + 8 + 4 + 2 + 1 = 32 - 1 := by decide

/-- A 1024-thread block has 32 warps, so the second-stage warp reduction needs exactly one full warp. -/
theorem block_warps : (1024 + 31) / 32 = 32 := by decide

-- § temperature

/-- Scaling by zero collapses every logit to the same value: the bug the greedy branch had. -/
theorem zero_scale_collapses (a b : Int) : 0 * a = 0 * b := by
  simp

/-- Scaling by a positive constant preserves the order, so greedy never needs temperature. -/
theorem pos_scale_monotone (c a b : Int) (hc : 0 < c) : a < b ↔ c * a < c * b := by
  constructor
  · intro h
    exact Int.mul_lt_mul_of_pos_left h hc
  · intro h
    by_contra hn
    have h2 : b ≤ a := Int.not_lt.mp hn
    have h3 := Int.mul_le_mul_of_nonneg_left h2 (Int.le_of_lt hc)
    omega

-- § Philox uniform, top-k and top-p bookkeeping

/-- The top 24 bits of a 32-bit word give an integer below 2^24, so `u24 / 2^24 < 1`. -/
theorem philox_u24_lt (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by
  omega

/-- Effective `top_k` on the sampled path: zero disables it, otherwise at most 64. -/
def kClamp (topK : Nat) : Nat := if topK > 0 then min topK 64 else 0

/-- The clamped `top_k` never exceeds the 64-slot buffers. -/
theorem kClamp_le (topK : Nat) : kClamp topK ≤ 64 := by
  unfold kClamp
  split <;> omega

/-- The top-p cut after the `min_keep` correction. -/
def cutAdj (cut minKeep nKeep : Nat) : Nat := if cut < minKeep then min minKeep nKeep else cut

/-- The adjusted cut never exceeds the survivor count (given the raw cut does not). -/
theorem cutAdj_le (cut minKeep nKeep : Nat) (h : cut ≤ nKeep) : cutAdj cut minKeep nKeep ≤ nKeep := by
  unfold cutAdj
  split <;> omega

/-- When survivors exist, the adjusted cut keeps at least one of them for `min_keep >= 1`. -/
theorem cutAdj_pos (cut minKeep nKeep : Nat) (h1 : 1 ≤ minKeep) (h2 : 1 ≤ nKeep) (h3 : 1 ≤ cut) :
    1 ≤ cutAdj cut minKeep nKeep := by
  unfold cutAdj
  split <;> omega

-- § launch geometry

/-- The sampled-path grid of 64-thread blocks covers every token and wastes fewer than 64 threads. -/
theorem grid_covers (n : Nat) : n ≤ (n + 63) / 64 * 64 ∧ (n + 63) / 64 * 64 < n + 64 := by
  omega

end Ref.Sampler



## FILE: /mnt/user-data/outputs/VerifyWindowKernels.lean
/-!
# MODULE: verify_window_kernels
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token ("verify window") versions of the GDN conv/gate/step kernels and the small index/gather helpers
that keep speculative decoding on the device.

## Source semantics
* The conv window of token `t` is the last four entries of `[hist(3) | x_0 ...]`; `gdn_conv_commit` keeps the
  last three entries of `[hist(3) | x_0 .. x_{n-1}]` and does nothing for `n <= 0`.
* `dense_steps` writes `[c, c+1, (c+1)/4, c+1]` per cell; `window_ids` clips to the last `window` cells and
  records the width.
* `fetch_blobs` / `gather_rows` split a flat index `i` into `(i / per, i - (i / per) * per)`.
* `rebase_ptrs` sets pointer `k` to `base + k * bytes`.
* Grid sizes: `channels / 128` conv blocks, `ceil(2 * h_v / 8)` gate blocks of 8 warps.

## Form template identifiers
`hist` -> `conv_history`; `n_keep` -> `accepted_count`; `steps` -> `step_records`; `window_ids` -> `window_cells`;
`kVerifyMaxT` -> `max_window`; `S` -> `head_dim`.
-/

namespace Ref.Verify

-- § conv window and commit

/-- The sequence `[hist(3) | xs]` read at position `src`. -/
def ext {α : Type} (hist xs : Nat → α) (src : Nat) : α :=
  if src < 3 then hist src else xs (src - 3)

/-- Positions below three read the carried history. -/
theorem ext_lt {α : Type} (hist xs : Nat → α) {s : Nat} (h : s < 3) : ext hist xs s = hist s := by
  unfold ext
  rw [if_pos h]

/-- Positions from three on read the new inputs, offset by three. -/
theorem ext_ge {α : Type} (hist xs : Nat → α) {s : Nat} (h : 3 ≤ s) : ext hist xs s = xs (s - 3) := by
  unfold ext
  have hc : ¬ (s < 3) := by omega
  rw [if_neg hc]

/-- The newest tap of token `t` is its own input. -/
theorem ext_new {α : Type} (hist xs : Nat → α) (t : Nat) : ext hist xs (t + 3) = xs t := by
  simpa using ext_ge hist xs (show 3 ≤ t + 3 by omega)

/-- Window entry `j` of token `t`. -/
def tap {α : Type} (hist xs : Nat → α) (t j : Nat) : α := ext hist xs (t + j)

/-- State kept after accepting `n` tokens: entry `j` is the window entry `j` of token `n`. -/
def commitHist {α : Type} (hist xs : Nat → α) (n j : Nat) : α := ext hist xs (n + j)

/-- The committed history is exactly the next token's window prefix. -/
theorem commit_is_next_window {α : Type} (hist xs : Nat → α) (n j : Nat) :
    commitHist hist xs n j = tap hist xs n j := rfl

/-- Accepting zero tokens leaves the history unchanged. -/
theorem commit_zero {α : Type} (hist xs : Nat → α) (j : Nat) (hj : j < 3) :
    commitHist hist xs 0 j = hist j := by
  unfold commitHist
  rw [Nat.zero_add]
  exact ext_lt hist xs hj

/-- Accepting one token shifts the history by one and appends the token. -/
theorem commit_one_last {α : Type} (hist xs : Nat → α) : commitHist hist xs 1 2 = xs 0 := by
  unfold commitHist
  simpa using ext_ge hist xs (show 3 ≤ 1 + 2 by omega)

/-- Accepting at least three tokens keeps only the three newest inputs. -/
theorem commit_big {α : Type} (hist xs : Nat → α) (n j : Nat) (hn : 3 ≤ n) :
    commitHist hist xs n j = xs (n + j - 3) := by
  unfold commitHist
  rw [ext_ge hist xs (show 3 ≤ n + j by omega)]

/-- Committing `n` tokens and then reading a later window agrees with reading it from the original history. -/
theorem commit_compose {α : Type} (hist xs : Nat → α) (n s : Nat) :
    ext (fun j => ext hist xs (n + j)) (fun i => xs (n + i)) s = ext hist xs (n + s) := by
  by_cases h : s < 3
  · rw [ext_lt _ _ h]
  · rw [ext_ge _ _ (show 3 ≤ s by omega)]
    rw [ext_ge hist xs (show 3 ≤ n + s by omega)]
    show xs (n + (s - 3)) = xs (n + s - 3)
    congr 1
    omega

-- § step records and window ids

/-- The record `dense_steps` writes for cell `c`: `[c, c+1, (c+1)/4, c+1]`. -/
def denseStep (c : Nat) : List Nat := [c, c + 1, (c + 1) / 4, c + 1]

/-- The record always holds four entries (`kStepCount`). -/
theorem denseStep_length (c : Nat) : (denseStep c).length = 4 := rfl

/-- The completed-block count is consistent with the cache size: `n_bid * 4 <= n_kv < n_bid * 4 + 4`. -/
theorem denseStep_blocks (c : Nat) : (c + 1) / 4 * 4 ≤ c + 1 ∧ c + 1 < (c + 1) / 4 * 4 + 4 := by
  omega

/-- First cell of the attention window. -/
def wStart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- The recorded width is `min n_kv window`. -/
theorem window_width (nKv window : Nat) : nKv - wStart nKv window = min nKv window := by
  unfold wStart
  split <;> omega

/-- Window ids stay inside the cache. -/
theorem window_ids_lt (nKv window j : Nat) (hj : j < nKv - wStart nKv window) :
    wStart nKv window + j < nKv := by
  omega

-- § flat index splitting (fetch_blobs, gather_rows)

/-- The in-blob offset `i - (i / per) * per` is the remainder. -/
theorem offset_is_mod (i per : Nat) : i - (i / per) * per = i % per := by
  have h := Nat.div_add_mod i per
  have h2 : per * (i / per) = (i / per) * per := Nat.mul_comm _ _
  omega

/-- The in-blob offset is below the blob size. -/
theorem offset_lt (i per : Nat) (hp : 0 < per) : i - (i / per) * per < per := by
  rw [offset_is_mod]
  exact Nat.mod_lt i hp

/-- A flat index below `n * per` selects a blob below `n`. -/
theorem blob_index_lt (i n per : Nat) (h : i < n * per) : i / per < n := by
  have h' : i < per * n := by rw [Nat.mul_comm]; exact h
  exact Nat.div_lt_of_lt_mul h'

/-- Pointer rebase: consecutive blobs are exactly `bytes` apart. -/
theorem rebase_step (base k bytes : Nat) :
    base + (k + 1) * bytes = (base + k * bytes) + bytes := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

-- § grid geometry

/-- Real geometry: 10240 channels at 128 per block is 80 blocks, i.e. 2*16 q/k blocks and 48 v blocks. -/
theorem conv_blocks_real : 10240 / 128 = 2 * 16 + 48 := by decide

/-- Every thread of a conv block addresses a channel below `channels` (real head dim 128). -/
theorem conv_channel_lt (bx tid channels : Nat) (hb : bx < channels / 128) (ht : tid < 128)
    (hc : channels % 128 = 0) : bx * 128 + tid < channels := by
  omega

/-- The gate/beta kernel's blocks of 8 warps cover all `2 * h_v` rows. -/
theorem ab_rows_covered (hv : Nat) : 2 * hv ≤ (2 * hv + 7) / 8 * 8 := by
  omega

/-- Real geometry: h_v = 48 needs 12 blocks. -/
theorem ab_blocks_real : (2 * 48 + 7) / 8 = 12 := by decide

/-- The 128 state rows split into 4 groups of 32 with every row owned exactly once. -/
theorem row_group_cover (i : Nat) (h : i < 128) : i / 32 < 4 ∧ i % 32 < 32 ∧ (i / 32) * 32 + i % 32 = i := by
  omega

-- § packed code extraction (embedding_gather_dev)

/-- Two-bit codes: a byte holds 4 codes and index `i < n` (with `4 | n`) reads byte `i / 4 < n / 4`. -/
theorem byte_index_2bit (i n : Nat) (h : i < n) (hn : n % 4 = 0) : i / 4 < n / 4 := by
  omega

/-- Four-bit codes: a byte holds 2 codes. -/
theorem byte_index_4bit (i n : Nat) (h : i < n) (hn : n % 2 = 0) : i / 2 < n / 2 := by
  omega

/-- A code extracted with a mask of `bits` bits is below `2^bits`. -/
theorem code_lt (byte shift bits : Nat) : (byte / 2 ^ shift) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

end Ref.Verify





## FILE: /mnt/user-data/outputs/ActivationQuantContract.lean
/-!
# MODULE: activation_quant_contract
SOURCES: include/ref/kernels/cpu/expert.hpp, src/kernels/cpu/expert.cpp, src/kernels/cpu/q2_avx2.cpp
Integer-level contract of `act_quant_q8_1` / `act_quant_q8_1_avx2`: per-32-element chunk scale amax/127,
codes clamped to [-127,127], round half AWAY from zero (not half-to-even), planar `ActQ` storage.

## Source semantics
* Each code is clamped to [-127,127]; the chunk sum therefore lies in [-127*32, 127*32] and fits int32.
* Rounding is half away from zero (odd-symmetric); nearest-even would differ on ties.
* n must be a multiple of QKA=32, giving n/32 chunks, at most MAXC=80.
* hx = scale * sum is zero when the chunk scale is zero (all-zero chunk).

## Form template identifiers
clamp127 -> code_clamp, rha -> round_half_away, chunk_sum_bound -> chunk_sum_range
-/

namespace Ref.ActQuant

-- § clamp
/-- Clamp to the symmetric int8 range used by the quantizer. -/
def clamp127 (v : Int) : Int := max (-127) (min 127 v)

/-- Clamped values lie in [-127,127]. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- Clamping an in-range value is the identity. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- Clamping is idempotent. -/
theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v :=
  clamp127_id _ (clamp127_bounds v).1 (clamp127_bounds v).2

/-- Clamping is monotone. -/
theorem clamp127_mono (a b : Int) (h : a ≤ b) : clamp127 a ≤ clamp127 b := by
  unfold clamp127
  omega

/-- Clamping is odd-symmetric. -/
theorem clamp127_neg (v : Int) : clamp127 (-v) = - clamp127 v := by
  unfold clamp127
  omega

-- § chunk sums
/-- A list of in-range codes has a sum bounded by 127 times its length. -/
theorem chunk_sum_bound : ∀ l : List Int, (∀ v ∈ l, -127 ≤ v ∧ v ≤ 127) →
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | x :: l, h => by
      have hx := h x (by simp)
      have ih := chunk_sum_bound l (fun v hv => h v (by simp [hv]))
      simp only [List.sum_cons, List.length_cons]
      omega

/-- A 32-code chunk sum is at most 4064 in magnitude and fits int32. -/
theorem chunk32_sum_fits : 32 * 127 = 4064 ∧ 4064 < 2 ^ 31 := by decide

-- § chunk geometry
/-- With n a multiple of 32, the chunks tile n exactly. -/
theorem nchunks_mul (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by omega

/-- Widest reduction (2560) has 80 chunks; any n up to 2560 has at most 80. -/
theorem nchunks_le_maxc (n : Nat) (hn : n ≤ 2560) : n / 32 ≤ 80 := by omega

/-- Weight block b covers activation chunks 2b and 2b+1, at byte offsets 64b and 64b+32. -/
theorem weight_block_chunks (b : Nat) : b * 64 = (2 * b) * 32 ∧ b * 64 + 32 = (2 * b + 1) * 32 := by
  omega

/-- The second chunk of the last gate/up block still lies inside the 2560-wide `q` array. -/
theorem last_chunk_in_q (b : Nat) (h : b < 40) : (2 * b + 1) * 32 + 32 ≤ 2560 := by omega

-- § rounding
/-- Round n/d (d>0) half away from zero, via nonnegative integer division. -/
def rha (n d : Int) : Int :=
  if 0 ≤ n then (2 * n + d) / (2 * d) else -((2 * (-n) + d) / (2 * d))

/-- Half-away rounding is odd-symmetric for positive numerators. -/
theorem rha_odd (n d : Int) (h : 0 < n) : rha (-n) d = - rha n d := by
  have h1 : ¬ (0 ≤ -n) := by omega
  have h2 : 0 ≤ n := by omega
  unfold rha
  rw [if_neg h1, if_pos h2, Int.neg_neg]

/-- A tie 1/2 rounds up to 1. -/
theorem rha_tie_pos : rha 1 2 = 1 := by decide

/-- A tie -1/2 rounds down to -1. -/
theorem rha_tie_neg : rha (-1) 2 = -1 := by decide

/-- Round-to-nearest-even would send the tie 1/2 to 0, so the two contracts differ. -/
theorem tie_differs_from_even : (0 : Int) ≠ rha 1 2 := by decide

-- § correction term
/-- hx = scale * sum vanishes for a zero scale. -/
theorem hx_zero_of_zero_scale (s sm : Int) (h : s = 0) : s * sm = 0 := by
  subst h
  exact Int.zero_mul sm

end Ref.ActQuant



## FILE: /mnt/user-data/outputs/ExpertLayoutAndIqKernels.lean
/-!
# MODULE: expert_layout_and_iq_kernels
SOURCES: src/kernels/cpu/expert_layout.cpp, src/kernels/cpu/iq_avx512.cpp
The per-layer expert offset table (contiguity validation, max blob) and the byte layouts the AVX-512 i-quant
row kernels decode (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S).

## Source semantics
* Layer l starts at the sum of (layer bytes * n_expert) over earlier layers; a line is accepted only if its
  offset equals that running total. With no native table every layer is BLOB bytes.
* `max_blob` is the maximum of layer sizes.
* iq512 supports exactly types 16,17,18,21,22; IQ1_M (29) stays on ggml-cpu.
* Each decode of chunk j<4 reads only inside its 256-value block; scale nibbles give odd scales 2s+1.

## Form template identifiers
layerOff -> layer_offset, chk -> contiguity_check, iqBytes -> iq_block_bytes, iqSupported -> iq512_supported
-/

namespace Ref.ExpertLayout

-- § offset table
/-- Byte offset of layer l: sum of earlier layer sizes times the expert count. -/
def layerOff (bytes : Nat → Nat) (ne : Nat) : Nat → Nat
  | 0 => 0
  | l + 1 => layerOff bytes ne l + bytes l * ne

/-- One layer's span is its blob size times the expert count. -/
theorem layerOff_succ (bytes : Nat → Nat) (ne l : Nat) :
    layerOff bytes ne (l + 1) = layerOff bytes ne l + bytes l * ne := rfl

/-- Offsets never decrease. -/
theorem layerOff_mono (bytes : Nat → Nat) (ne l : Nat) :
    layerOff bytes ne l ≤ layerOff bytes ne (l + 1) := Nat.le_add_right _ _

/-- With uniform blobs (no native table) the offset is l * (BLOB * ne). -/
theorem layerOff_uniform (b ne : Nat) : ∀ l : Nat, layerOff (fun _ => b) ne l = l * (b * ne)
  | 0 => by simp [layerOff]
  | l + 1 => by
      have ih := layerOff_uniform b ne l
      simp only [layerOff, ih]
      rw [Nat.add_mul, Nat.one_mul]

/-- Contiguity checker: each (offset, bytes) entry must start where the previous one ended. -/
def chk (ne : Nat) : List (Nat × Nat) → Nat → Bool
  | [], _ => true
  | (o, b) :: t, at_ => (o == at_) && chk ne t (at_ + b * ne)

/-- An empty table is accepted. -/
theorem chk_nil (ne at_ : Nat) : chk ne [] at_ = true := rfl

/-- A table is accepted iff its head starts at the running offset and the tail is accepted after it. -/
theorem chk_cons (ne o b : Nat) (t : List (Nat × Nat)) (at_ : Nat) :
    chk ne ((o, b) :: t) at_ = true ↔ o = at_ ∧ chk ne t (at_ + b * ne) = true := by
  simp [chk]

/-- Largest blob over a list of layer sizes. -/
def maxBlob (l : List Nat) : Nat := l.foldl max 0

/-- The running maximum never falls below its accumulator. -/
theorem le_foldl_max (acc : Nat) : ∀ l : List Nat, acc ≤ l.foldl max acc
  | [] => Nat.le_refl _
  | x :: l => by
      simp only [List.foldl]
      exact Nat.le_trans (Nat.le_max_left acc x) (le_foldl_max (max acc x) l)

/-- Every element is bounded by the running maximum. -/
theorem mem_le_foldl_max : ∀ (l : List Nat) (acc x : Nat), x ∈ l → x ≤ l.foldl max acc
  | [], _, _, h => by simp at h
  | y :: l, acc, x, h => by
      simp only [List.foldl]
      rcases List.mem_cons.mp h with rfl | h'
      · exact Nat.le_trans (Nat.le_max_right acc x) (le_foldl_max _ l)
      · exact mem_le_foldl_max l _ x h'

/-- `max_blob` bounds every layer's blob. -/
theorem maxBlob_ge (l : List Nat) (x : Nat) (h : x ∈ l) : x ≤ maxBlob l :=
  mem_le_foldl_max l 0 x h

-- § i-quant block layouts
/-- Bytes per 256-value block of each supported i-quant type (0 if unsupported). -/
def iqBytes : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 21 => 110
  | 22 => 82
  | _ => 0

/-- Types handled by the AVX-512 i-quant kernels. -/
def iqSupported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- Every supported type has a positive block size. -/
theorem iqSupported_bytes_pos (t : Nat) (h : iqSupported t = true) : 0 < iqBytes t := by
  simp only [iqSupported, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with ((((h | h) | h) | h) | h) <;> subst h <;> decide

/-- IQ1_M (29) is not handled by the AVX-512 path. -/
theorem iq1m_unsupported : iqSupported 29 = false := by decide

/-- Q2_0 (42) is not an i-quant type. -/
theorem q2_0_not_iq : iqSupported 42 = false := by decide

/-- IQ2_XXS: d + 32 u16. -/
theorem iq2xxs_total : 2 + 32 * 2 = iqBytes 16 := by decide
/-- IQ2_XS: d + 32 u16 + 8 scale bytes. -/
theorem iq2xs_total : 2 + 32 * 2 + 8 = iqBytes 17 := by decide
/-- IQ3_XXS: d + 64 grid bytes + 32 sign/scale bytes. -/
theorem iq3xxs_total : 2 + 64 + 32 = iqBytes 18 := by decide
/-- IQ3_S: d + qs 64 + qh 8 + signs 32 + scales 4. -/
theorem iq3s_total : 2 + 64 + 8 + 32 + 4 = iqBytes 21 := by decide
/-- IQ2_S: d + qs 64 (grid+signs) + qh 8 + scales 8. -/
theorem iq2s_total : 2 + 64 + 8 + 8 = iqBytes 22 := by decide

-- § decode stays in bounds for chunks j < 4
/-- IQ2_XXS chunk reads 16 bytes inside the qs array. -/
theorem iq2xxs_chunk_in_bounds (j : Nat) (h : j < 4) : 2 + 16 * j + 16 ≤ 66 := by omega

/-- IQ2_XS chunk reads and its scale bytes stay in the block. -/
theorem iq2xs_in_bounds (j : Nat) (h : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S grid bytes, sign words, qh bytes and scale bytes are disjoint and in bounds. -/
theorem iq2s_in_bounds (j : Nat) (h : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 34 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by
  omega

/-- IQ3_XXS grid bytes and sign/scale words stay in the block. -/
theorem iq3xxs_in_bounds (j : Nat) (h : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 2 + 64 + 8 * j + 8 ≤ 98 := by omega

/-- IQ3_S qs, qh, signs and scale reads stay in the block. -/
theorem iq3s_in_bounds (j : Nat) (h : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by
  omega

-- § scales
/-- A 4-bit scale field s gives the odd scale 2s+1 in [1,31]. -/
theorem scale_odd (s : Nat) (h : s < 16) : 1 ≤ 2 * s + 1 ∧ 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by
  omega

/-- The scale permute maps each of the 32 int16 lanes to one of the four scales. -/
theorem scales4_lane (i : Nat) (h : i < 32) : i / 8 < 4 := by omega

end Ref.ExpertLayout



## FILE: /mnt/user-data/outputs/ExpertPoolProtocol.lean
/-!
# MODULE: expert_pool_protocol
SOURCES: include/ref/kernels/cpu/pool.hpp, src/kernels/cpu/pool.cpp
Abstract model of the CPU expert pool: row-range partitioning for split phases, the multi-expert flat row
index mapping, worker/thread counts, and the parked/active handover invariant of the completion protocol.

## Source semantics
* A split phase cuts `total` rows into `parts` ranges [total*p/parts, total*(p+1)/parts): they abut, start at 0 and
  end at `total`.
* A flat row r over several experts of `per` rows maps to expert r/per, local row r%per; each step makes progress
  and never crosses an expert or the task end.
* parked + active = n_workers always; `run()` waits for parked = n so no worker is in the drain loop when the
  host resets `head`.
* Workers >= 1; `physical_cores(true)` drops exactly the first core.

## Form template identifiers
lo/hi -> range_begin/range_end, PS -> park_state, Step -> worker_transition, Reach -> reachable
-/

namespace Ref.ExpertPool

-- § row-range partition
/-- Start row of part `p` of `parts`. -/
def lo (total parts p : Nat) : Nat := total * p / parts
/-- End row (exclusive) of part `p` of `parts`. -/
def hi (total parts p : Nat) : Nat := total * (p + 1) / parts

/-- The first range starts at row 0. -/
theorem lo_zero (total parts : Nat) : lo total parts 0 = 0 := by simp [lo]

/-- Consecutive ranges abut: the end of part p is the start of part p+1. -/
theorem hi_eq_lo_succ (total parts p : Nat) : hi total parts p = lo total parts (p + 1) := rfl

/-- The last range ends exactly at `total`. -/
theorem hi_last (total parts : Nat) (h : 0 < parts) : hi total parts (parts - 1) = total := by
  unfold hi
  have e : parts - 1 + 1 = parts := by omega
  rw [e]
  exact Nat.mul_div_cancel total h

/-- Each range is well-formed: begin <= end. -/
theorem lo_le_hi (total parts p : Nat) : lo total parts p ≤ hi total parts p := by
  unfold lo hi
  exact Nat.div_le_div_right (Nat.mul_le_mul_left total (Nat.le_succ p))

/-- Every range ends within `total`. -/
theorem hi_le_total (total parts p : Nat) (h0 : 0 < parts) (hp : p < parts) :
    hi total parts p ≤ total := by
  unfold hi
  calc total * (p + 1) / parts ≤ total * parts / parts :=
        Nat.div_le_div_right (Nat.mul_le_mul_left total hp)
    _ = total := Nat.mul_div_cancel total h0

-- § flat multi-expert row mapping
/-- Expert index and local row recombine to the flat row, and the local row is in range. -/
theorem flat_row_split (r per : Nat) (hper : 0 < per) :
    per * (r / per) + r % per = r ∧ r % per < per :=
  ⟨Nat.div_add_mod r per, Nat.mod_lt r hper⟩

/-- A flat row below per*n belongs to an expert index below n. -/
theorem expert_index_lt (r per n : Nat) (h : r < per * n) : r / per < n :=
  Nat.div_lt_of_lt_mul h

/-- Each inner step takes at least one row. -/
theorem step_progress (per r0 g1 r : Nat) (h0 : r0 < per) (hr : r < g1) :
    r0 < min per (r0 + (g1 - r)) := by omega

/-- The step's end never passes the expert's last row. -/
theorem step_within_expert (per r0 g1 r : Nat) : min per (r0 + (g1 - r)) ≤ per := by omega

/-- The step never overshoots the task's end row `g1`. -/
theorem step_within_task (per r0 g1 r : Nat) (h0 : r0 < per) (hr : r ≤ g1) :
    r + (min per (r0 + (g1 - r)) - r0) ≤ g1 := by omega

-- § worker and thread counts
/-- Worker count: the request if positive, else the core count, never below one. -/
def workersOf (nw cores : Nat) : Nat := max 1 (if 0 < nw then nw else cores)

/-- At least one worker always exists. -/
theorem workersOf_pos (nw cores : Nat) : 1 ≤ workersOf nw cores := Nat.le_max_left _ _

/-- Thread count including the host when it also drains. -/
def threadsOf (n : Nat) (hostWorks : Bool) : Nat := n + if hostWorks then 1 else 0

/-- With `host_works` the pool is n+1 threads. -/
theorem threads_with_host (n : Nat) : threadsOf n true = n + 1 := rfl

/-- Without `host_works` the pool is exactly the workers. -/
theorem threads_without_host (n : Nat) : threadsOf n false = n := rfl

/-- Skipping the first core removes exactly one entry (when any exists). -/
theorem skip_first_len (cores : List Nat) : (cores.drop 1).length = cores.length - 1 := by simp

/-- Parts per expert in `run_split` is at least one. -/
def partsOf (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- `partsOf` is positive. -/
theorem partsOf_pos (threads n : Nat) : 1 ≤ partsOf threads n := Nat.le_max_left _ _

-- § park / drain handover
/-- Workers parked versus inside the drain loop. -/
structure PS where
  parked : Nat
  active : Nat

/-- A worker leaves the park, or returns to it. -/
inductive Step : PS → PS → Prop
  | leave (p a : Nat) : 0 < p → Step ⟨p, a⟩ ⟨p - 1, a + 1⟩
  | arrive (p a : Nat) : 0 < a → Step ⟨p, a⟩ ⟨p + 1, a - 1⟩

/-- Reachability by worker transitions. -/
inductive Reach (s : PS) : PS → Prop
  | refl : Reach s s
  | tail {t u : PS} : Reach s t → Step t u → Reach s u

/-- One transition preserves parked + active. -/
theorem step_preserves {s t : PS} (h : Step s t) : s.parked + s.active = t.parked + t.active := by
  cases h with
  | leave p a hp =>
      show p + a = (p - 1) + (a + 1)
      omega
  | arrive p a ha =>
      show p + a = (p + 1) + (a - 1)
      omega

/-- Reachable states preserve parked + active. -/
theorem reach_preserves {s t : PS} (h : Reach s t) : s.parked + s.active = t.parked + t.active := by
  induction h with
  | refl => rfl
  | tail _ hs ih => exact ih.trans (step_preserves hs)

/-- Handover safety: starting with all n workers parked, `parked = n` implies no worker is in the drain loop. -/
theorem handover_safe (n : Nat) (t : PS) (h : Reach ⟨n, 0⟩ t) (hp : t.parked = n) : t.active = 0 := by
  have e : n + 0 = t.parked + t.active := reach_preserves h
  omega

/-- Parked never exceeds the worker count. -/
theorem parked_le (n : Nat) (t : PS) (h : Reach ⟨n, 0⟩ t) : t.parked ≤ n := by
  have e : n + 0 = t.parked + t.active := reach_preserves h
  omega

end Ref.ExpertPool



## FILE: /mnt/user-data/outputs/GgufDequantGeometry.lean
/-!
# MODULE: gguf_dequant_geometry
SOURCES: include/ref/artifact/dequant.hpp, include/ref/artifact/gguf_reader.hpp
Block geometry of the GGUF quantization types, in-bounds field access of the scalar dequantizers, the Q2_0
symbol mapping, fp16/bf16 conversion facts, and GGUF header parsing invariants.

## Source semantics
* Q2_0: 64 elements / 18 bytes, code c -> (c-1)*d, so values lie in {-d,0,d,2d}.
* Each K-quant/IQ layout sums to its declared block size and all indexed reads stay in the block.
* GGUF magic is "GGUF" little-endian, version 3 only; data starts at the next multiple of the alignment.
* A bounds-checked cursor read either fails or moves forward without passing the end.

## Form template identifiers
geom -> block_geometry, advance -> cursor_read, kv -> iq4nl_codebook
-/

namespace Ref.GgufDequant

-- § block geometry
/-- (elements, bytes) per block for the types in the table. -/
def geom : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 2 => some (32, 18)
  | 8 => some (32, 34)
  | 6 => some (32, 22)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 20 => some (32, 18)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | _ => none

/-- Q2_0 is 64 elements in 18 bytes. -/
theorem geom_q2_0 : geom 42 = some (64, 18) := by decide
/-- Q2_0 bytes = 16 code bytes + 2 scale bytes. -/
theorem q2_0_bits : 18 * 8 = 64 * 2 + 16 := by decide
/-- Q4_0 / IQ4_NL: 4 bits per element plus a 2-byte scale. -/
theorem q4_0_bits : 18 * 8 = 32 * 4 + 16 := by decide
/-- Q8_0: 8 bits per element plus a 2-byte scale. -/
theorem q8_0_bits : 34 * 8 = 32 * 8 + 16 := by decide
/-- Q4_0 and IQ4_NL share a geometry. -/
theorem q4_0_iq4nl_same : geom 2 = geom 20 := by decide
/-- Q5_0: scale + 4 high-bit bytes + 16 nibble bytes. -/
theorem q5_0_layout : 2 + 4 + 16 = 22 := by decide
/-- Q6_K: ql 128 + qh 64 + scales 16 + d 2. -/
theorem q6_K_layout : 128 + 64 + 16 + 2 = 210 ∧ 128 + 64 = 192 ∧ 192 + 16 = 208 := by decide
/-- Q4_K: d, dmin, 12 scale bytes, 128 quant bytes; qs at 16. -/
theorem q4_K_layout : 2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 = 16 := by decide
/-- Q5_K: d, dmin, 12 scales, 32 qh, 128 ql; qh at 16, ql at 48. -/
theorem q5_K_layout : 2 + 2 + 12 + 32 + 128 = 176 ∧ 16 + 32 = 48 := by decide
/-- Q3_K: hmask 32, qs 64, scales 12, d 2; scales at 96, d at 108. -/
theorem q3_K_layout : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 = 96 ∧ 96 + 12 = 108 := by decide
/-- IQ4_XS: d 2, scales_h 2, scales_l 4, qs 128; qs at 8. -/
theorem iq4_xs_layout : 2 + 2 + 4 + 128 = 136 ∧ 2 + 2 + 4 = 8 := by decide

-- § Q2_0 dequantization
/-- Code j of a block lives in byte j/4 (inside the 16 code bytes) at bit offset 2*(j%4). -/
theorem q2_0_code_position (j : Nat) (h : j < 64) :
    j / 4 < 16 ∧ (j % 4) * 2 < 8 ∧ (j % 4) * 2 % 2 = 0 := by omega

/-- The decoded value (c-1)*d is one of -d, 0, d, 2d. -/
theorem q2_0_value_set (c d : Int) (h0 : 0 ≤ c) (h3 : c ≤ 3) :
    (c - 1) * d = -d ∨ (c - 1) * d = 0 ∨ (c - 1) * d = d ∨ (c - 1) * d = 2 * d := by
  have h : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  rcases h with rfl | rfl | rfl | rfl <;> omega

-- § K-quant index bounds
/-- Q6_K: reads at l+96, l+32 and sc[is+6] stay in bounds for l < 32. -/
theorem q6_K_in_bounds (l : Nat) (h : l < 32) :
    l + 96 < 128 ∧ l + 32 < 64 ∧ l < 32 ∧ l / 16 + 6 < 8 := by omega

/-- Q6_K: two 128-element halves consume 128 ql, 64 qh and 16 scales in total. -/
theorem q6_K_totals : 2 * 128 = 256 ∧ 2 * 64 = 128 ∧ 2 * 32 = 64 ∧ 2 * 8 = 16 := by decide

/-- A 6-bit value built from a nibble and two high bits is below 64, so subtracting 32 gives [-32,31]. -/
theorem six_bit_range (a h : Nat) (ha : a < 16) (hh : h < 4) :
    a + 16 * h < 64 ∧ -32 ≤ ((a + 16 * h : Nat) : Int) - 32 ∧ ((a + 16 * h : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q4_K/Q5_K scale-min unpack indices stay inside the 12 scale bytes for j < 8. -/
theorem scale_min_k4_in_bounds (j : Nat) (h : j < 8) :
    (j < 4 → j + 4 < 12) ∧ (4 ≤ j → j + 4 < 12 ∧ j - 4 < 12) := by omega

/-- Q4_K: four 64-element groups produce 256 outputs and consume 128 quant bytes and 8 scale pairs. -/
theorem q4_K_totals : 4 * (32 + 32) = 256 ∧ 4 * 32 = 128 ∧ 4 * 2 = 8 := by decide

/-- Q5_K high-bit masks 1,4,16,64 and 2,8,32,128 fit in a byte for all four groups. -/
theorem q5_K_masks : ∀ n : Nat, n < 4 → 2 ^ (2 * n) < 256 ∧ 2 * 2 ^ (2 * n) < 256 := by decide

/-- Q3_K: 2 halves x 4 shifts x 2 scale uses = 16 scales, and 2*4*2*16 = 256 outputs. -/
theorem q3_K_totals : 2 * 4 * 2 = 16 ∧ 2 * 4 * 2 * 16 = 256 := by decide

/-- Q3_K: reads at l+16 stay inside the 32-byte hmask and the 32-byte half of qs. -/
theorem q3_K_in_bounds (l : Nat) (h : l < 16) : l + 16 < 32 := by omega

/-- IQ4_XS: the 6-bit scale index ib/2 and bit shift 2*ib stay in range for ib < 8. -/
theorem iq4_xs_scale_bounds (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ 2 * ib + 2 ≤ 16 := by omega

/-- IQ4_NL codebook has 16 entries. -/
def kv : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has exactly 16 entries (one per nibble). -/
theorem kv_length : kv.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kv_sorted : List.Pairwise (· < ·) kv := by decide

/-- Low and high nibbles of a byte are valid codebook indices. -/
theorem nibble_index_lt (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 := by omega

-- § fp16 / bf16
/-- Normal fp16 exponents 1..30 map to fp32 exponents 113..142, inside the normal fp32 range. -/
theorem fp16_normal_exp (e : Nat) (h1 : 1 ≤ e) (h2 : e ≤ 30) :
    1 ≤ e - 15 + 127 ∧ e - 15 + 127 ≤ 254 := by omega

/-- bf16 widening by a 16-bit shift is invertible. -/
theorem bf16_roundtrip (h : Nat) : (h * 65536) / 65536 = h := by omega

/-- bf16 widening stays below 2^32 for 16-bit inputs. -/
theorem bf16_fits (h : Nat) (hh : h < 65536) : h * 65536 < 4294967296 := by omega

-- § GGUF header
/-- The bytes "GGUF" read little-endian give 0x46554747. -/
theorem gguf_magic : 0x47 + 0x47 * 256 + 0x55 * 65536 + 0x46 * 16777216 = 0x46554747 := by decide

/-- Rounding the header end up to the default alignment of 32 yields a multiple of 32 within 31 bytes. -/
theorem data_start_align32 (p : Nat) :
    (p + 31) / 32 * 32 % 32 = 0 ∧ p ≤ (p + 31) / 32 * 32 ∧ (p + 31) / 32 * 32 < p + 32 := by
  omega

/-- A bounds-checked cursor read. -/
def advance (pos size n : Nat) : Option Nat := if pos + n ≤ size then some (pos + n) else none

/-- A successful read moves forward and stays inside the file. -/
theorem advance_bounds (pos size n p' : Nat) (h : advance pos size n = some p') :
    pos ≤ p' ∧ p' ≤ size := by
  unfold advance at h
  split at h
  · injection h with h'
    omega
  · cases h

/-- A read that would pass the end fails. -/
theorem advance_fails (pos size n : Nat) (h : size < pos + n) : advance pos size n = none := by
  unfold advance
  rw [if_neg (by omega)]

/-- Only a sample of at most 64 array items is kept. -/
theorem array_sample_le (count : Nat) : min count 64 ≤ 64 := by omega

/-- The default guard constants: 10 used experts out of 512, 24 query heads over 2 KV heads. -/
theorem guard_defaults : 10 ≤ 512 ∧ 24 % 2 = 0 := by decide

end Ref.GgufDequant



## FILE: /mnt/user-data/outputs/Q2ExpertBlobGeometry.lean
/-!
# MODULE: q2_expert_blob_geometry
SOURCES: include/ref/kernels/cpu/expert.hpp, src/kernels/cpu/expert.cpp
Fixed geometry of one Q2_0 routed expert (H=2560, FF=640, QK=64, QKA=32) and the internal layout of its
1,382,400-byte blob: gate/up codes, down codes, gate/up scales, down scales.

## Source semantics
* One weight block is QK=64 weights with one fp16 scale (18 bytes); one activation chunk is QKA=32 elements, so
  a weight block spans exactly two activation chunks.
* Gate and up rows are interleaved (row 2r = gate_r, row 2r+1 = up_r); regions are contiguous and tile the blob.

## Form template identifiers
H -> hidden_width, FF -> expert_inner_width, QK -> weight_block, QKA -> activation_chunk, BB -> block_bytes,
BLOB -> expert_blob_bytes, O_* -> region_offset, ROW_* -> code_bytes_per_row, SC_* -> scale_blocks_per_row
-/

namespace Ref.Q2ExpertGeometry

-- § constants
def H : Nat := 2560
def FF : Nat := 640
def NE : Nat := 512
def QK : Nat := 64
def QKA : Nat := 32
def BB : Nat := 18
def MAXC : Nat := H / QKA
def ROW_GU : Nat := H * 2 / 8
def ROW_D : Nat := FF * 2 / 8
def SC_GU : Nat := H / QK
def SC_D : Nat := FF / QK
def BLOB : Nat := 3 * (H * FF * BB / QK)
def O_GU_CODES : Nat := 0
def O_D_CODES : Nat := 2 * FF * ROW_GU
def O_GU_SCALES : Nat := O_D_CODES + 1 * H * ROW_D
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2

-- § numeric values
/-- `H` is 2560. -/
theorem H_eq : H = 2560 := rfl
/-- `FF` is 640. -/
theorem FF_eq : FF = 640 := rfl
/-- `MAXC` is 80 activation chunks. -/
theorem MAXC_eq : MAXC = 80 := by decide
/-- Gate/up code bytes per row. -/
theorem ROW_GU_eq : ROW_GU = 640 := by decide
/-- Down code bytes per row. -/
theorem ROW_D_eq : ROW_D = 160 := by decide
/-- Weight blocks per gate/up row. -/
theorem SC_GU_eq : SC_GU = 40 := by decide
/-- Weight blocks per down row. -/
theorem SC_D_eq : SC_D = 10 := by decide
/-- The blob is 1,382,400 bytes. -/
theorem BLOB_eq : BLOB = 1382400 := by decide
/-- Offset of the down codes. -/
theorem O_D_CODES_eq : O_D_CODES = 819200 := by decide
/-- Offset of the gate/up scales. -/
theorem O_GU_SCALES_eq : O_GU_SCALES = 1228800 := by decide
/-- Offset of the down scales. -/
theorem O_D_SCALES_eq : O_D_SCALES = 1331200 := by decide

-- § block / chunk contract
/-- A Q2_0 block is 2 bytes of fp16 scale plus 16 code bytes (64 codes at 2 bits). -/
theorem block_bytes_decomposition : BB = 2 + QK * 2 / 8 := by decide
/-- A weight block spans exactly two activation chunks. -/
theorem weight_block_is_two_chunks : QK = 2 * QKA := by decide
/-- The hidden width is a whole number of activation chunks, and `MAXC` counts them. -/
theorem hidden_is_chunks : H = MAXC * QKA := by decide
/-- Chunk count of a row equals twice its weight-block count (gate/up). -/
theorem gu_chunks_twice_blocks : H / QKA = 2 * SC_GU := by decide
/-- Chunk count of the intermediate equals twice the down weight-block count. -/
theorem d_chunks_twice_blocks : FF / QKA = 2 * SC_D := by decide
/-- The intermediate activation fits in the `MAXC`-chunk scratch. -/
theorem intermediate_fits_scratch : FF / QKA ≤ MAXC := by decide
/-- Code bytes per row are 16 per weight block. -/
theorem code_bytes_per_block : ROW_GU = SC_GU * 16 ∧ ROW_D = SC_D * 16 := by decide

-- § region layout
/-- Regions are ordered and the last one ends exactly at `BLOB`. -/
theorem regions_ordered_and_tile :
    O_GU_CODES ≤ O_D_CODES ∧ O_D_CODES ≤ O_GU_SCALES ∧ O_GU_SCALES ≤ O_D_SCALES ∧
      O_D_SCALES + 1 * H * SC_D * 2 = BLOB := by decide

/-- The code regions hold 3 projections of 2-bit codes, and the scales hold one fp16 per 64 weights. -/
theorem code_and_scale_totals :
    O_GU_SCALES = 3 * (H * FF * 2 / 8) ∧ BLOB - O_GU_SCALES = 3 * (H * FF / QK * 2) := by decide

/-- Every gate and up row lies inside the gate/up code region. -/
theorem gu_rows_in_codes (r : Nat) (h : r < FF) :
    O_GU_CODES + (2 * r + 1) * ROW_GU + ROW_GU ≤ O_D_CODES := by
  simp only [ROW_GU_eq, O_D_CODES_eq, FF_eq] at h ⊢
  have : O_GU_CODES = 0 := rfl
  omega

/-- Every down row lies inside the down code region. -/
theorem down_rows_in_codes (r : Nat) (h : r < H) :
    O_D_CODES + (r + 1) * ROW_D ≤ O_GU_SCALES := by
  simp only [ROW_D_eq, O_D_CODES_eq, O_GU_SCALES_eq, H_eq] at h ⊢
  omega

/-- Every gate and up scale row lies inside the gate/up scale region. -/
theorem gu_scales_in_region (r : Nat) (h : r < FF) :
    O_GU_SCALES + ((2 * r + 1) * SC_GU * 2 + SC_GU * 2) ≤ O_D_SCALES := by
  simp only [SC_GU_eq, O_GU_SCALES_eq, O_D_SCALES_eq, FF_eq] at h ⊢
  omega

/-- Every down scale row lies inside the down scale region, which ends at the blob end. -/
theorem down_scales_in_region (r : Nat) (h : r < H) :
    O_D_SCALES + (r + 1) * SC_D * 2 ≤ BLOB := by
  simp only [SC_D_eq, O_D_SCALES_eq, BLOB_eq, H_eq] at h ⊢
  omega

/-- Activation chunk indices touched by weight block `b` of a gate/up row stay below `MAXC`. -/
theorem block_chunks_in_range (b : Nat) (h : b < SC_GU) : 2 * b + 1 < MAXC := by
  simp only [SC_GU_eq, MAXC_eq] at h ⊢
  omega

end Ref.Q2ExpertGeometry



## FILE: /mnt/user-data/outputs/Q2RowDotIdentity.lean
/-!
# MODULE: q2_row_dot_identity
SOURCES: include/ref/kernels/cpu/expert.hpp, src/kernels/cpu/expert.cpp, src/kernels/cpu/q2_avx2.cpp
The algebra behind the Q2_0 row dot: codes {0..3} stand for symbols {-1,0,1,2}; the VNNI path computes
sum(c*xhat) and subtracts sum(xhat) (the `hx` correction) instead of forming (c-1).

## Source semantics
* sum (c-1) x = sum c x - sum x (integer-exact, which is why the correction can be factored out).
* One weight scale covers two activation chunks with different scales; collapsing to one scale is wrong.
* Unpack: code i of a packed byte stream is (b / 4^i) % 4; multishift lanes keep codes in order.
* Multi-token kernels process each token independently; tokens are dispatched with NT = n (1..8) and the
  AVX2 path in chunks of at most 4.

## Form template identifiers
dotSym -> symbol_dot, dotC -> code_dot, sumX -> activation_sum, ntDispatch -> token_template_select,
chunks4 -> avx2_token_chunks
-/

namespace Ref.Q2RowDot

-- § dot products over (code, activation) pairs
/-- Dot product with the symbol mapping c -> c - 1. -/
def dotSym : List (Nat × Int) → Int
  | [] => 0
  | (c, x) :: l => ((c : Int) - 1) * x + dotSym l

/-- Dot product with raw codes. -/
def dotC : List (Nat × Int) → Int
  | [] => 0
  | (c, x) :: l => (c : Int) * x + dotC l

/-- Sum of the activations. -/
def sumX : List (Nat × Int) → Int
  | [] => 0
  | (_, x) :: l => x + sumX l

/-- The Q2_0 identity: sum (c-1) x = sum c x - sum x. -/
theorem dotSym_eq_dotC_sub_sumX : ∀ l : List (Nat × Int), dotSym l = dotC l - sumX l
  | [] => by simp [dotSym, dotC, sumX]
  | (c, x) :: l => by
      have ih := dotSym_eq_dotC_sub_sumX l
      simp only [dotSym, dotC, sumX, ih, Int.sub_mul, Int.one_mul] <;> omega

/-- The symbol dot product splits over concatenation (two chunks of a block). -/
theorem dotSym_append : ∀ l₁ l₂ : List (Nat × Int), dotSym (l₁ ++ l₂) = dotSym l₁ + dotSym l₂
  | [], l₂ => by simp [dotSym]
  | (c, x) :: l₁, l₂ => by
      have ih := dotSym_append l₁ l₂
      simp only [List.cons_append, dotSym, ih] <;> omega

/-- A zero code contributes nothing, so zero-padded lanes of a narrow load are harmless. -/
theorem zero_code_contributes_nothing (x : Int) : (0 : Int) * x = 0 := Int.zero_mul x

-- § two scales per weight block
/-- Per-chunk scaling distributes over the weight-block scale. -/
theorem block_two_scale (dw s0 s1 A0 A1 : Int) :
    dw * (s0 * A0) + dw * (s1 * A1) = dw * (s0 * A0 + s1 * A1) := by
  rw [Int.mul_add]

/-- Using one chunk scale for the whole block agrees with the correct formula only if (s1-s0)*A1 = 0. -/
theorem single_scale_iff (s0 s1 A0 A1 : Int) :
    s0 * A0 + s1 * A1 = s0 * (A0 + A1) ↔ (s1 - s0) * A1 = 0 := by
  constructor
  · intro h
    rw [Int.mul_add] at h
    rw [Int.sub_mul]
    omega
  · intro h
    rw [Int.sub_mul] at h
    rw [Int.mul_add]
    omega

/-- A concrete case where the single-scale shortcut gives a wrong answer. -/
theorem single_scale_counterexample : ¬ ((1 : Int) * 1 + 2 * 1 = 1 * (1 + 1)) := by decide

-- § unpack and VNNI bounds
/-- Pack four 2-bit codes into a byte, LSB first. -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3
/-- Code 0 of a byte. -/
def code0 (b : Nat) : Nat := b % 4
/-- Code 1 of a byte. -/
def code1 (b : Nat) : Nat := b / 4 % 4
/-- Code 2 of a byte. -/
def code2 (b : Nat) : Nat := b / 16 % 4
/-- Code 3 of a byte. -/
def code3 (b : Nat) : Nat := b / 64 % 4

/-- Unpacking a packed byte returns the four codes in order. -/
theorem unpack_pack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    code0 (pack4 c0 c1 c2 c3) = c0 ∧ code1 (pack4 c0 c1 c2 c3) = c1 ∧
      code2 (pack4 c0 c1 c2 c3) = c2 ∧ code3 (pack4 c0 c1 c2 c3) = c3 := by
  unfold code0 code1 code2 code3 pack4
  refine ⟨by omega, by omega, by omega, by omega⟩

/-- A packed byte is below 256 when its codes are 2-bit. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4
  omega

/-- Across the two bytes of a 16-bit lane, position 4 is the first code of the second byte. -/
theorem lane_second_byte (b0 b1 : Nat) (h : b0 < 256) : (b0 + 256 * b1) / 256 % 4 = b1 % 4 := by
  omega

/-- A u8 code times an int8 activation stays within +-381. -/
theorem code_mul_bound (c x : Int) (hc0 : 0 ≤ c) (hc3 : c ≤ 3) (hx1 : -127 ≤ x) (hx2 : x ≤ 127) :
    -381 ≤ c * x ∧ c * x ≤ 381 := by
  have h : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  rcases h with rfl | rfl | rfl | rfl <;> omega

/-- A 2560-wide dot of such products fits in int32. -/
theorem vnni_sum_fits : 2560 * 381 = 975360 ∧ 975360 < 2 ^ 31 := by decide

/-- Broadcasting the scalar correction to eight lanes would apply it eight times; that equals one
application only when the correction is zero. -/
theorem eight_lane_correction_wrong (c : Int) (h : 8 * c = c) : c = 0 := by omega

-- § multi-token dispatch
/-- Template argument chosen for `n` tokens: n for 1..7, otherwise 8. -/
def ntDispatch (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- For 1..8 tokens the selected template handles exactly `n` tokens. -/
theorem ntDispatch_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : ntDispatch n = n := by
  unfold ntDispatch
  split <;> omega

/-- Token chunk sizes for the AVX2 path (at most 4 at a time), with fuel. -/
def chunks4 : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, r => if r = 0 then [] else min 4 r :: chunks4 f (r - min 4 r)

/-- With enough fuel the chunks sum to the token count. -/
theorem chunks4_sum : ∀ f r : Nat, r ≤ 4 * f → (chunks4 f r).sum = r
  | 0, r, h => by
      have : r = 0 := by omega
      subst this
      simp [chunks4]
  | f + 1, r, h => by
      by_cases hr : r = 0
      · subst hr
        simp [chunks4]
      · have ih := chunks4_sum f (r - min 4 r) (by omega)
        simp [chunks4, hr, ih] <;> omega

/-- Every AVX2 chunk is between 1 and 4 tokens. -/
theorem chunks4_bounded : ∀ f r : Nat, ∀ k ∈ chunks4 f r, 1 ≤ k ∧ k ≤ 4
  | 0, r, k, hk => by simp [chunks4] at hk
  | f + 1, r, k, hk => by
      by_cases hr : r = 0
      · subst hr
        simp [chunks4] at hk
      · simp only [chunks4, hr, if_false, List.mem_cons] at hk
        rcases hk with rfl | hk
        · omega
        · exact chunks4_bounded f _ k hk

/-- Processing tokens in chunks equals processing them all at once, token by token. -/
theorem chunked_map_eq {α β : Type} (f : α → β) (a b : List α) :
    (a ++ b).map f = a.map f ++ b.map f := List.map_append

end Ref.Q2RowDot





## FILE: /mnt/user-data/outputs/CanonicalQuantRoundtrip.lean
/-!
# MODULE: canonical_quant_roundtrip
SOURCES: tools/canonical_xcheck.py
Bit-exact canonicalisation of GGUF quantisation blocks (Q2_0, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0,
IQ4_NL, IQ4_XS and the float types) into the S2/S4/S8/P16/P32 canonical forms: one code per element, one
scale (and optionally one offset) per group, with the integer code bias applied BEFORE the single multiply.

## Source semantics
* Every decode is `(code + code_bias) * scale (+ offset)`; the bias is an INTEGER subtraction on the code,
  never a value-space offset (that would not be bit exact).
* Block geometry (bytes, elements, group size) is fixed per source type and the registry is consistent:
  group size divides block size; only the K-quants with a min carry a per-group offset.
* Sub-byte unpacking is a bijection between the packed byte and its codes (nibbles, 2-bit quads, the 5th
  bit of Q5_0, the `hmask` bit of Q3_K, the `qh` pairs of Q6_K).
* IQ4 uses a non-affine 16-entry codebook, strictly increasing.
* `--split I/N` partitions tensors by `index % N == I`.

## Form template identifiers
- MAPPINGS (registry)              → `mappings`
- Mapping.code_bias                → `Mapping.codeBias`
- Mapping.block_bytes / block_elems / group_elems → `blockBytes` / `blockElems` / `groupElems`
- KV_IQ4NL                         → `kvIq4nl`
- to_s2 / from_s2 (codes in separate plane) → `pack4` / `pack4_unpack`
-/

namespace Ref.CanonicalQuant

-- § mapping registry

/-- One row of the `MAPPINGS` registry (geometry, integer code bias, offset flag). -/
structure Mapping where
  name : String
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat
  codeBias : Int
  hasOffset : Bool

/-- The registry of source types, transcribed from the `MAPPINGS` dict. -/
def mappings : List Mapping :=
  [⟨"Q2_0", 18, 64, 64, -1, false⟩, ⟨"Q3_K", 110, 256, 16, -4, false⟩, ⟨"Q4_0", 18, 32, 32, -8, false⟩,
   ⟨"Q4_K", 144, 256, 32, 0, true⟩, ⟨"Q5_0", 22, 32, 32, -16, false⟩, ⟨"Q5_K", 176, 256, 32, 0, true⟩,
   ⟨"Q6_K", 210, 256, 16, -32, false⟩, ⟨"Q8_0", 34, 32, 32, -128, false⟩, ⟨"IQ4_NL", 18, 32, 32, 0, false⟩,
   ⟨"IQ4_XS", 136, 256, 32, 0, false⟩, ⟨"F32", 4, 1, 1, 0, false⟩, ⟨"BF16", 2, 1, 1, 0, false⟩,
   ⟨"F16", 2, 1, 1, 0, false⟩]

/-- All 13 source types of the shard are mapped. -/
theorem mappings_length : mappings.length = 13 := by
  decide

/-- The group size always divides the block size, so groups tile a block exactly. -/
theorem group_divides_block : ∀ m ∈ mappings, m.blockElems % m.groupElems = 0 := by
  decide

/-- Only the two K-quants with a per-group min (Q4_K, Q5_K) carry an offset plane. -/
theorem offset_only_k_quants :
    ∀ m ∈ mappings, m.hasOffset = true → m.name = "Q4_K" ∨ m.name = "Q5_K" := by
  decide

/-- Block byte counts equal the sum of their fields (scales, mins, high bits, codes). -/
theorem block_byte_layouts :
    (2 + 64 / 4 = 18) ∧ (2 + 32 / 2 = 18) ∧ (2 + 32 = 34) ∧ (2 + 4 + 32 / 2 = 22) ∧
    (2 + 2 + 12 + 256 / 2 = 144) ∧ (256 / 8 + 256 / 4 + 12 + 2 = 110) ∧
    (2 + 2 + 12 + 256 / 8 + 256 / 2 = 176) ∧ (256 / 2 + 256 / 4 + 16 + 2 = 210) ∧
    (2 + 2 + 4 + 256 / 2 = 136) := by
  decide

-- § integer code bias ranges

/-- Q2_0: codes 0..3 decode to the integers -1..2 (the "- 1" is an offset in the codebook). -/
theorem q2_0_symbol_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by
  omega

/-- Q4_0: codes 0..15 decode to -8..7. -/
theorem q4_0_symbol_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by
  omega

/-- Q5_0: a nibble plus the 5th bit is a 5-bit code and decodes to -16..15. -/
theorem q5_0_code_range (lo hb : Nat) (hl : lo < 16) (hh : hb < 2) :
    lo + 16 * hb < 32 ∧ -16 ≤ ((lo + 16 * hb : Nat) : Int) - 16 ∧
      ((lo + 16 * hb : Nat) : Int) - 16 ≤ 15 := by
  omega

/-- Q3_K: two bits plus the hmask bit stay in 0..7 and decode to -4..3. -/
theorem q3k_code_range (c2 m : Nat) (h2 : c2 < 4) (hm : m < 2) :
    c2 + 4 * m < 8 ∧ -4 ≤ ((c2 + 4 * m : Nat) : Int) - 4 ∧ ((c2 + 4 * m : Nat) : Int) - 4 ≤ 3 := by
  omega

/-- Q6_K: a nibble plus two `qh` bits is a 6-bit code and decodes to -32..31. -/
theorem q6k_code_range (nib hi : Nat) (hn : nib < 16) (hh : hi < 4) :
    nib + 16 * hi < 64 ∧ -32 ≤ ((nib + 16 * hi : Nat) : Int) - 32 ∧
      ((nib + 16 * hi : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q8_0: storing the int8 code as `+128` keeps it unsigned, and the decode's `-128` undoes it exactly. -/
theorem q8_0_bias_roundtrip (s : Int) (h1 : -128 ≤ s) (h2 : s ≤ 127) :
    0 ≤ s + 128 ∧ s + 128 ≤ 255 ∧ (s + 128) - 128 = s := by
  omega

-- § sub-byte packing

/-- Pack four 2-bit codes into a byte, LSB first (Q2_0's own convention). -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed Q2_0 byte fits in 8 bits. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4
  omega

/-- `codes[:, t::4] = (byte >> 2t) & 3` recovers each of the four codes. -/
theorem pack4_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 % 4 = c0 ∧ pack4 c0 c1 c2 c3 / 4 % 4 = c1 ∧
      pack4 c0 c1 c2 c3 / 16 % 4 = c2 ∧ pack4 c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold pack4
  omega

/-- Two nibbles per byte: low nibble first half of the block, high nibble second half. -/
theorem nibble_roundtrip (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi ∧ lo + 16 * hi < 256 := by
  omega

/-- The Q5_0 source reaches bit `j+16` of `qh` via `(qh >> (j+12)) & 0x10`; this is the same bit. -/
theorem q5_0_high_bit_shift (qh j : Nat) :
    (qh / 2 ^ (j + 12)) % 32 / 16 = (qh / 2 ^ (j + 16)) % 2 := by
  have h1 : 2 ^ (j + 16) = 2 ^ (j + 12) * 16 := by
    rw [show j + 16 = (j + 12) + 4 by omega, Nat.pow_add]
  have h2 : qh / 2 ^ (j + 16) = (qh / 2 ^ (j + 12)) / 16 := by
    rw [h1, Nat.div_div_eq_div_mul]
  rw [h2]
  generalize qh / 2 ^ (j + 12) = x
  omega

/-- A 6-bit scale from a low nibble and two high bits stays below 64. -/
theorem scale6_lt (lo t : Nat) : lo % 16 + 16 * (t % 4) < 64 := by
  omega

/-- `get_scale_min_k4` for `j >= 4`: low 4 bits of one byte plus the top 2 bits of another stay 6-bit. -/
theorem scale_min_k4_hi_lt (a b : Nat) (hb : b < 256) : a % 16 + 16 * (b / 64) < 64 := by
  omega

-- § index maps are injective (no element read twice, none skipped)

/-- Q3_K hmask bit `4*half + j` is distinct for every (half, j) and stays in a byte. -/
theorem q3k_mask_bit_injective (h1 h2 j1 j2 : Nat) (a1 : h1 < 2) (a2 : h2 < 2) (b1 : j1 < 4) (b2 : j2 < 4)
    (h : 4 * h1 + j1 = 4 * h2 + j2) : h1 = h2 ∧ j1 = j2 := by
  omega

/-- Q3_K group index `8*half + 2*j + sub` enumerates 16 distinct groups. -/
theorem q3k_group_injective (h1 h2 j1 j2 s1 s2 : Nat) (a1 : h1 < 2) (a2 : h2 < 2) (b1 : j1 < 4)
    (b2 : j2 < 4) (c1 : s1 < 2) (c2 : s2 < 2) (h : 8 * h1 + 2 * j1 + s1 = 8 * h2 + 2 * j2 + s2) :
    h1 = h2 ∧ j1 = j2 ∧ s1 = s2 := by
  omega

/-- Q6_K quads land at strides of 32: `128*half + 32*quad + l` is injective on its domain. -/
theorem q6k_quad_injective (h1 h2 q1 q2 l1 l2 : Nat) (a1 : h1 < 2) (a2 : h2 < 2) (b1 : q1 < 4)
    (b2 : q2 < 4) (c1 : l1 < 32) (c2 : l2 < 32) (h : 128 * h1 + 32 * q1 + l1 = 128 * h2 + 32 * q2 + l2) :
    h1 = h2 ∧ q1 = q2 ∧ l1 = l2 := by
  omega

/-- Q5_K masks `u1 = 1 << 2j`, `u2 = 2 << 2j` use bit `2j+k`, all distinct across the four steps. -/
theorem q5k_mask_bit_injective (j1 j2 k1 k2 : Nat) (a1 : j1 < 4) (a2 : j2 < 4) (b1 : k1 < 2) (b2 : k2 < 2)
    (h : 2 * j1 + k1 = 2 * j2 + k2) : j1 = j2 ∧ k1 = k2 := by
  omega

/-- Q4_K: four 64-element steps of two groups give 8 groups of 32 = 256 elements. -/
theorem q4k_group_count : 4 * 2 = 8 ∧ 8 * 32 = 256 := by
  decide

/-- Q2_0 canonical codes plane states its own element count: `codes.size` determines the block count. -/
theorem q2_0_codes_size (nb : Nat) : nb * 64 / 64 = nb := by
  omega

-- § the IQ4 codebook

/-- `kvalues_iq4nl`. -/
def kvIq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has exactly 16 entries (a 4-bit index). -/
theorem kv_length : kvIq4nl.length = 16 := by
  decide

/-- The codebook is strictly increasing. -/
theorem kv_strictly_increasing : kvIq4nl.Pairwise (· < ·) := by
  decide

/-- It is NOT affine: consecutive differences differ, so it cannot be expressed as a code bias. -/
theorem kv_not_affine :
    (kvIq4nl.getD 1 0 - kvIq4nl.getD 0 0) ≠ (kvIq4nl.getD 2 0 - kvIq4nl.getD 1 0) := by
  decide

-- § work splitting and code widths

/-- `--split I/N`: each tensor index lands in exactly one process slot. -/
theorem split_partition (idx sn si si' : Nat) (hs : 0 < sn) (h1 : idx % sn = si) (h2 : idx % sn = si') :
    idx % sn < sn ∧ si = si' := by
  have hlt : idx % sn < sn := Nat.mod_lt idx hs
  refine ⟨hlt, ?_⟩
  omega

/-- Codes per byte times bits per code is 8 for every canonical form width. -/
theorem code_bits_fill_byte : ∀ b ∈ [2, 4, 8], b * (8 / b) = 8 := by
  decide

end Ref.CanonicalQuant



## FILE: /mnt/user-data/outputs/ChatFrontendParser.lean
/-!
# MODULE: chat_frontend_parser
SOURCES: serve/frontend.py
Request normalisation (OpenAI / Anthropic -> chat template messages) and the incremental output parser that
splits model text into reasoning, content and tool calls without ever emitting a partial tag.

## Source semantics
* Reasoning effort spellings map onto {none, low, medium, xhigh}: "high", "max", "maximum" mean xhigh;
  "none", "off", "minimal", "disabled" disable thinking; any other spelling is a client error (400).
* Anthropic `budget_tokens`: under 2048 low, under 8192 medium, otherwise xhigh (monotone in the budget).
* `_hold` returns the length of the longest suffix of the buffered text that is a PROPER prefix of a tag,
  so it is always strictly shorter than the tag; that many characters are withheld from the client.
* Tool-call parameter values drop one leading and one trailing newline.

## Form template identifiers
- EFFORT table                   → `effortLevel`
- budget_effort                  → `budgetRank`
- OutputParser._hold             → `holdLen` / `isHold`
- value[1:] / value[:-1] newline → `stripNL`
-/

namespace Ref.ChatFrontend

-- § reasoning effort

/-- Thinking levels as seen by the template: 0 = none, 1 = low, 2 = medium, 3 = xhigh. -/
def effortLevel : String → Option Nat
  | "none" => some 0
  | "off" => some 0
  | "minimal" => some 0
  | "disabled" => some 0
  | "low" => some 1
  | "medium" => some 2
  | "high" => some 3
  | "xhigh" => some 3
  | "max" => some 3
  | "maximum" => some 3
  | _ => none

/-- "high" means xhigh (the template's default), not a separate level. -/
theorem high_is_xhigh : effortLevel "high" = effortLevel "xhigh" := by
  rfl

/-- Aliases that disable thinking. -/
theorem disabling_aliases :
    effortLevel "none" = some 0 ∧ effortLevel "off" = some 0 ∧ effortLevel "minimal" = some 0 ∧
      effortLevel "disabled" = some 0 := by
  decide

/-- Unknown spellings are rejected rather than guessed. -/
theorem unknown_effort_rejected : effortLevel "turbo" = none := by
  rfl

-- § Anthropic thinking budget

/-- Level chosen from `budget_tokens`. -/
def budgetRank (n : Nat) : Nat := if n < 2048 then 1 else if n < 8192 then 2 else 3

/-- A larger budget never selects a lower level. -/
theorem budget_mono (n m : Nat) (h : n ≤ m) : budgetRank n ≤ budgetRank m := by
  unfold budgetRank
  by_cases a : n < 2048 <;> by_cases b : n < 8192 <;> by_cases c : m < 2048 <;> by_cases d : m < 8192 <;>
    simp [a, b, c, d] <;> omega

/-- The levels are only ever low, medium or xhigh (never "none"). -/
theorem budget_range (n : Nat) : 1 ≤ budgetRank n ∧ budgetRank n ≤ 3 := by
  unfold budgetRank
  by_cases a : n < 2048 <;> by_cases b : n < 8192 <;> simp [a, b]

-- § partial-tag hold-back

/-- Is a suffix of `text` of length `n` a proper non-empty prefix of `tag`? -/
def isHold (text tag : List Nat) (n : Nat) : Bool :=
  decide (0 < n) && decide (n < tag.length) && decide (n ≤ text.length) &&
    (text.drop (text.length - n) == tag.take n)

/-- Longest held-back suffix length. -/
def holdLen (text tag : List Nat) : Nat := ((List.range tag.length).filter (isHold text tag)).foldl max 0

/-- A fold of `max` over elements below `k` stays below `k`. -/
theorem foldl_max_lt (k : Nat) :
    ∀ (l : List Nat) (a : Nat), a < k → (∀ x ∈ l, x < k) → l.foldl max a < k := by
  intro l
  induction l with
  | nil => intro a ha _; simpa using ha
  | cons x xs ih =>
    intro a ha hl
    simp only [List.foldl_cons]
    have hx : x < k := hl x (by simp)
    exact ih (max a x) (by omega) (fun y hy => hl y (by simp [hy]))

/-- The withheld suffix is always strictly shorter than the tag, so a full tag is never withheld. -/
theorem holdLen_lt (text tag : List Nat) (hne : 0 < tag.length) : holdLen text tag < tag.length := by
  unfold holdLen
  apply foldl_max_lt _ _ _ hne
  intro x hx
  rw [List.mem_filter, List.mem_range] at hx
  exact hx.1

-- § newline stripping of parameter values

/-- Drop one leading newline (code 10). -/
def dropLead : List Nat → List Nat
  | 10 :: t => t
  | l => l

/-- Drop one trailing newline. -/
def dropTrail (l : List Nat) : List Nat :=
  match l.reverse with
  | 10 :: t => t.reverse
  | _ => l

/-- Strip one newline at each end. -/
def stripNL (l : List Nat) : List Nat := dropTrail (dropLead l)

/-- A value wrapped in exactly one newline on each side is recovered unchanged. -/
theorem stripNL_wrapped (l : List Nat) : stripNL (10 :: (l ++ [10])) = l := by
  simp [stripNL, dropLead, dropTrail]

end Ref.ChatFrontend



## FILE: /mnt/user-data/outputs/GgufReaderGeometry.lean
/-!
# MODULE: gguf_reader_geometry
SOURCES: tools/gguf_reader.py
Minimal GGUF v3 header reader: magic and header sizes, tensor element counts, expected byte counts from block
geometry, and the aligned start of the data section.

## Source semantics
* Magic `0x46554747` is "GGUF" in little-endian byte order; the fixed header is 24 bytes.
* `elements` is the product of the shape; `expected_bytes` is `elements / block_elems * block_bytes` when
  `elements` is a whole number of blocks and `None` otherwise.
* `data_start` rounds the end of the tensor table up to `alignment`.
* Type id 42 is `Q2_0`; ids 31 and 41 both carry the name `Q4_0_8_8` in the table (ids stay distinct).

## Form template identifiers
- GGUF_MAGIC           → `ggufMagic`
- TensorInfo.elements  → `shapeElems`
- expected_bytes       → `expectedBytes`
- data_start rounding  → `alignUp`
- GGML_TYPES           → `typeName`
-/

namespace Ref.GgufReader

-- § header

/-- The GGUF magic number. -/
def ggufMagic : Nat := 0x46554747

/-- Little-endian bytes of the magic spell G, G, U, F. -/
theorem magic_bytes :
    ggufMagic % 256 = 71 ∧ ggufMagic / 256 % 256 = 71 ∧ ggufMagic / 65536 % 256 = 85 ∧
      ggufMagic / 16777216 = 70 := by
  decide

/-- Magic (4) + version (4) + n_tensors (8) + n_kv (8) = 24 header bytes; the tensor info tail is 12. -/
theorem header_sizes : 4 + (4 + 8 + 8) = 24 ∧ 4 + 8 = 12 := by
  decide

-- § element counts

/-- Product of a shape. -/
def shapeElems : List Nat → Nat
  | [] => 1
  | d :: ds => d * shapeElems ds

/-- Element counts multiply over concatenated shapes (so extra expert axes are just layout). -/
theorem shapeElems_append (a b : List Nat) : shapeElems (a ++ b) = shapeElems a * shapeElems b := by
  induction a with
  | nil => simp [shapeElems]
  | cons d ds ih => simp [shapeElems, ih, Nat.mul_assoc]

-- § expected bytes

/-- `expected_bytes`: whole blocks only. -/
def expectedBytes (elems blockElems blockBytes : Nat) : Option Nat :=
  if elems % blockElems = 0 then some (elems / blockElems * blockBytes) else none

/-- A whole number of blocks yields `blocks * block_bytes`. -/
theorem expectedBytes_of_blocks (k be bb : Nat) (h : 0 < be) :
    expectedBytes (k * be) be bb = some (k * bb) := by
  unfold expectedBytes
  rw [if_pos (Nat.mul_mod_left k be), Nat.mul_div_cancel k h]

/-- A partial block yields `None`. -/
theorem expectedBytes_partial (e be bb : Nat) (h : e % be ≠ 0) : expectedBytes e be bb = none := by
  unfold expectedBytes
  rw [if_neg h]

/-- A fused Q2_0 expert tensor `[64, 256, 16]` is 4096 blocks of 18 bytes. -/
theorem q2_0_tiny_expert_bytes : expectedBytes 262144 64 18 = some 73728 := by
  decide

-- § alignment of the data section

/-- Round `pos` up to a multiple of `a` (the `data_start` formula). -/
def alignUp (pos a : Nat) : Nat := (pos + a - 1) / a * a

/-- Alignment never moves backwards, moves less than one alignment unit, and lands on a multiple. -/
theorem alignUp_spec (pos a : Nat) (ha : 0 < a) :
    pos ≤ alignUp pos a ∧ alignUp pos a < pos + a ∧ alignUp pos a % a = 0 := by
  have h0 : (pos + a - 1) / a * a + (pos + a - 1) % a = pos + a - 1 := by
    have := Nat.div_add_mod (pos + a - 1) a
    rw [Nat.mul_comm] at this
    exact this
  have h1 : (pos + a - 1) % a < a := Nat.mod_lt _ ha
  have h2 : alignUp pos a % a = 0 := by
    unfold alignUp
    exact Nat.mul_mod_left _ _
  refine ⟨?_, ?_, h2⟩
  · unfold alignUp
    omega
  · unfold alignUp
    omega

/-- Aligning an already aligned position is the identity (default alignment 32). -/
theorem alignUp32_idempotent (k : Nat) : alignUp (32 * k) 32 = 32 * k := by
  unfold alignUp
  omega

-- § the ggml type table

/-- A sample of `GGML_TYPES`; unknown ids print as `type<id>` in the source. -/
def typeName : Nat → String
  | 0 => "F32"
  | 1 => "F16"
  | 2 => "Q4_0"
  | 8 => "Q8_0"
  | 30 => "BF16"
  | 31 => "Q4_0_8_8"
  | 41 => "Q4_0_8_8"
  | 42 => "Q2_0"
  | _ => "unknown"

/-- Ids 31 and 41 share a NAME but are distinct ids, so name-keyed tooling cannot tell them apart. -/
theorem duplicate_name_distinct_ids : typeName 31 = typeName 41 ∧ (31 : Nat) ≠ 41 := by
  exact ⟨rfl, by decide⟩

/-- Type id 42 is `Q2_0`, the type the stock gguf-py enum lacks. -/
theorem q2_0_is_type_42 : typeName 42 = "Q2_0" := by
  rfl

/-- The ids 0..42 of the table are pairwise distinct. -/
theorem type_ids_nodup : (List.range 43).Nodup := by
  decide

end Ref.GgufReader



## FILE: /mnt/user-data/outputs/GgufWriterQ2Contract.lean
/-!
# MODULE: gguf_writer_q2_contract
SOURCES: tools/gguf_writer.py
A minimal GGUF v3 writer including the `Q2_0` quantiser: block = 64 elements, fp16 scale, 16 bytes of 2-bit
codes, LSB first, `d = amax`, `code = clamp(round(w/d) + 1, 0, 3)`, symbol = `code - 1`.

## Source semantics
* The code is clamped into 0..3, so the symbol set is {-1, 0, +1, +2}.
* With `d = amax` the ratio `w/d` lies in [-1, 1], so code 3 (symbol +2) is never produced by the quantiser.
* Element `j` lives in byte `j/4` at bit `(j%4)*2`: a bijection from 0..63 onto 16 bytes x 4 slots.
* Re-quantising dequantised values at the same scale is the identity on codes (the fixed-point check).
* Tensor data and the header are padded to `alignment` (32); each tensor offset is the aligned running sum.

## Form template identifiers
- np.clip(rint(b/d)+1, 0, 3) → `clamp3`
- packed[j // 4] |= code[j] << ((j % 4) * 2) → `slotOf`
- pad = (-len) % alignment → `pad32`
- running tensor offset → `alignUp32`
-/

namespace Ref.GgufWriter

-- § the quantiser clamp

/-- `clamp(x, 0, 3)` on integers, returned as a natural code. -/
def clamp3 (x : Int) : Nat :=
  if x < 0 then 0 else if x > 3 then 3 else x.toNat

/-- Every produced code is a valid 2-bit code. -/
theorem clamp3_lt (x : Int) : clamp3 x < 4 := by
  unfold clamp3
  split <;> (try split) <;> omega

/-- In-range values are unchanged by the clamp. -/
theorem clamp3_id (x : Int) (h0 : 0 ≤ x) (h3 : x ≤ 3) : (clamp3 x : Int) = x := by
  unfold clamp3
  split <;> (try split) <;> omega

/-- The decoded symbol of any code lies in {-1, 0, 1, 2}. -/
theorem symbol_range (x : Int) : -1 ≤ (clamp3 x : Int) - 1 ∧ (clamp3 x : Int) - 1 ≤ 2 := by
  have h := clamp3_lt x
  omega

/-- With `d = amax`, `rint(w/d) ∈ [-1, 1]` and so the +2 level (code 3) is never emitted. -/
theorem plus_two_never_emitted (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 1) : clamp3 (r + 1) ≤ 2 := by
  unfold clamp3
  split <;> (try split) <;> omega

/-- Fixed point: decoding a code at unit scale and re-quantising returns the same code. -/
theorem requantise_fixed_point (c : Nat) (h : c < 4) : clamp3 (((c : Int) - 1) + 1) = c := by
  unfold clamp3
  split <;> (try split) <;> omega

-- § bit packing

/-- Element `j` of a block goes to (byte `j/4`, slot `j%4`). -/
def slotOf (j : Nat) : Nat × Nat := (j / 4, j % 4)

/-- The slot map covers 16 bytes x 4 slots and is invertible. -/
theorem slotOf_bijective (j : Nat) (h : j < 64) :
    (slotOf j).1 < 16 ∧ (slotOf j).2 < 4 ∧ 4 * (slotOf j).1 + (slotOf j).2 = j := by
  unfold slotOf
  omega

/-- Two distinct elements never share a (byte, slot). -/
theorem slotOf_injective (i j : Nat) (hi : i < 64) (hj : j < 64) (h : slotOf i = slotOf j) : i = j := by
  unfold slotOf at h
  have h1 : i / 4 = j / 4 := congrArg Prod.fst h
  have h2 : i % 4 = j % 4 := congrArg Prod.snd h
  omega

/-- A Q2_0 block is a 2-byte scale plus 16 code bytes (64 elements at 2 bits each). -/
theorem q2_0_block_bytes : 2 + 64 * 2 / 8 = 18 := by
  decide

-- § alignment

/-- `(-len) % 32` written without negation. -/
def pad32 (n : Nat) : Nat := (32 - n % 32) % 32

/-- Padding always lands on an alignment boundary. -/
theorem pad32_aligned (n : Nat) : (n + pad32 n) % 32 = 0 := by
  unfold pad32
  omega

/-- Round up to the 32-byte alignment (the tensor directory offset update). -/
def alignUp32 (n : Nat) : Nat := (n + 31) / 32 * 32

/-- The aligned offset never overlaps the previous tensor and wastes less than one unit. -/
theorem alignUp32_spec (n : Nat) : n ≤ alignUp32 n ∧ alignUp32 n < n + 32 ∧ alignUp32 n % 32 = 0 := by
  unfold alignUp32
  omega

/-- Next tensor offset is at least the previous offset plus its length. -/
theorem offsets_monotone (off len : Nat) : off + len ≤ alignUp32 (off + len) := by
  have h := alignUp32_spec (off + len)
  exact h.1

end Ref.GgufWriter



## FILE: /mnt/user-data/outputs/IqPackLayout.lean
/-!
# MODULE: iq_pack_layout
SOURCES: tools/iq_pack.py
Native pack for the Q2_0 / IQ2_XS / IQ3_XXS model files: experts kept in GGUF form as per-layer blobs
`[gate | up | down]` x 512, an index table with 64-byte-aligned arena offsets, and BF16 router narrowing.

## Source semantics
* Per layer, blob size = per-expert bytes of gate + up + down, where per-expert bytes = tensor bytes / 512;
  gate and up must agree. The experts arena is the running sum of `blob * 512`.
* The index arena size is the running sum of each tensor's size rounded up to ALIGN = 64.
* An F32 router whose low 16 bits are all zero is exactly a BF16 widened by 16 bits; it is stored as BF16.
* `n_layers = 1 + max layer index` found among `*_exps.weight` tensors.

## Form template identifiers
- N_EXPERT / blob / offset       → `nExpert` / blob sizes list / `totalBytes`
- ALIGN pad                      → `alignUp64`
- (u & 0xFFFF)==0, u>>16         → `routerNarrow`
- write_index at                 → `alignedTotal`
-/

namespace Ref.IqPack

/-- Experts per layer. -/
def nExpert : Nat := 512

-- § experts arena

/-- Total arena bytes for a list of per-layer blob sizes (each layer holds 512 blobs). -/
def totalBytes : List Nat → Nat
  | [] => 0
  | b :: bs => b * 512 + totalBytes bs

/-- The arena is `512 * (sum of blob sizes)`. -/
theorem totalBytes_eq_mul_sum (bs : List Nat) : totalBytes bs = 512 * bs.sum := by
  induction bs with
  | nil => simp [totalBytes]
  | cons b bs ih =>
    simp only [totalBytes, List.sum_cons, ih]
    omega

/-- Per-expert bytes recover exactly when the tensor holds 512 equal experts. -/
theorem per_expert_exact (per : Nat) : (512 * per) / 512 = per := by
  omega

/-- A layer blob is gate + up + down bytes, and the layer's arena slice is 512 of them. -/
theorem layer_slice (g u d : Nat) : (g + u + d) * 512 = g * 512 + u * 512 + d * 512 := by
  omega

-- § 64-byte aligned index

/-- Round up to ALIGN = 64. -/
def alignUp64 (n : Nat) : Nat := (n + 63) / 64 * 64

/-- Alignment is monotone, bounded and lands on a boundary. -/
theorem alignUp64_spec (n : Nat) : n ≤ alignUp64 n ∧ alignUp64 n < n + 64 ∧ alignUp64 n % 64 = 0 := by
  unfold alignUp64
  omega

/-- Running aligned total of a list of tensor sizes (the `at` in `write_index`). -/
def alignedTotal : List Nat → Nat
  | [] => 0
  | n :: ns => alignUp64 n + alignedTotal ns

/-- The arena is never smaller than the raw tensor bytes. -/
theorem sum_le_alignedTotal (ns : List Nat) : ns.sum ≤ alignedTotal ns := by
  induction ns with
  | nil => simp [alignedTotal]
  | cons n ns ih =>
    have h := alignUp64_spec n
    simp only [alignedTotal, List.sum_cons]
    omega

/-- The arena size is itself a multiple of 64. -/
theorem alignedTotal_aligned (ns : List Nat) : alignedTotal ns % 64 = 0 := by
  induction ns with
  | nil => simp [alignedTotal]
  | cons n ns ih =>
    have h := alignUp64_spec n
    simp only [alignedTotal]
    omega

-- § router narrowing

/-- Keep the high 16 bits of an F32 pattern. -/
def routerNarrow (u : Nat) : Nat := u / 65536

/-- When the low 16 bits are zero, narrowing then widening reproduces the F32 pattern exactly. -/
theorem router_narrow_exact (u : Nat) (h : u % 65536 = 0) : routerNarrow u * 65536 = u := by
  unfold routerNarrow
  omega

/-- Widening a BF16 pattern by 16 bits and narrowing back is the identity. -/
theorem bf16_widen_narrow (b : Nat) : routerNarrow (b * 65536) = b := by
  unfold routerNarrow
  omega

/-- A narrowed router fits in 16 bits when the F32 pattern is below 2^32. -/
theorem router_narrow_lt (u : Nat) (h : u < 4294967296) : routerNarrow u < 65536 := by
  unfold routerNarrow
  omega

-- § layer count

/-- `n_layers = 1 + max index`: every layer index found is below the layer count. -/
theorem layer_index_lt (l mx : Nat) (h : l ≤ mx) : l < 1 + mx := by
  omega

end Ref.IqPack



## FILE: /mnt/user-data/outputs/MtpPackQuant.lean
/-!
# MODULE: mtp_pack_quant
SOURCES: tools/mtp_pack.py
Packs the MTP draft block as a GGUF: dense tensors stay BF16/F32, fused routed experts
(`gate_up_proj [512,1280,2560]`, `down_proj [512,2560,640]`) are quantised along the last axis to Q2_0, Q4_0 or
Q8_0 with round-to-nearest.

## Source semantics
* Q2_0 searches 17 candidate scales `amax * f` with `f` from 1/2 to 1 and clamps symbols to [-1, 2]; the
  stored code is `symbol + 1` in 0..3.
* Q4_0 packs `q[:16] | q[16:] << 4` with every code at most 15.
* Q8_0 codes are `rint(x / d)` with `d = amax/127`, so they fit an int8.
* The inner dimension must be a multiple of the block size, and the byte counts are
  `elements / block * block_bytes` (about 0.71, 1.42 and 2.67 GB for the three formats).

## Form template identifiers
- np.linspace(0.5, 1.0, 17)  → `scaleNumer` (numerator over 32)
- np.clip(rint(x*inv), -1, 2)+1 → `codeOfSymbol`
- q[:16] | (q[16:] << 4)     → `nibblePack`
- blob size                  → `expertBytes`
-/

namespace Ref.MtpPack

-- § Q2_0 scale search

/-- Candidate scale factor `f_i = (16 + i) / 32`, `i = 0..16`, as a numerator over 32. -/
def scaleNumer (i : Nat) : Nat := 16 + i

/-- The 17 candidates span `amax/2` to `amax` inclusive. -/
theorem scale_grid_span (i : Nat) (h : i ≤ 16) : 16 ≤ scaleNumer i ∧ scaleNumer i ≤ 32 := by
  unfold scaleNumer
  omega

/-- There are exactly 17 candidates. -/
theorem scale_grid_count : (List.range 17).length = 17 := by
  decide

/-- Symbol in [-1, 2] maps to a 2-bit code `symbol + 1` in [0, 3]. -/
def codeOfSymbol (s : Int) : Int := s + 1

/-- Codes of clamped symbols are valid 2-bit codes. -/
theorem code_of_symbol_range (s : Int) (h1 : -1 ≤ s) (h2 : s ≤ 2) :
    0 ≤ codeOfSymbol s ∧ codeOfSymbol s ≤ 3 := by
  unfold codeOfSymbol
  omega

-- § Q4_0 / Q8_0

/-- `q[:16] | (q[16:] << 4)` for one nibble pair. -/
def nibblePack (lo hi : Nat) : Nat := lo + 16 * hi

/-- Packing two codes of at most 15 fits a byte and unpacks to the same pair. -/
theorem nibble_pack_roundtrip (lo hi : Nat) (hl : lo ≤ 15) (hh : hi ≤ 15) :
    nibblePack lo hi < 256 ∧ nibblePack lo hi % 16 = lo ∧ nibblePack lo hi / 16 = hi := by
  unfold nibblePack
  omega

/-- `min 15 t` is a valid Q4_0 code. -/
theorem q4_0_code_clamped (t : Nat) : min 15 t ≤ 15 := by
  omega

/-- A Q8_0 code with magnitude at most 127 fits an int8. -/
theorem q8_0_code_int8 (r : Int) (h1 : -127 ≤ r) (h2 : r ≤ 127) : -128 ≤ r ∧ r ≤ 127 := by
  omega

-- § expert sizes

/-- Bytes of a quantised expert tensor: `elements / block * block_bytes`. -/
def expertBytes (elems block blockBytes : Nat) : Nat := elems / block * blockBytes

/-- Inner dimensions 2560 and 640 are multiples of both block sizes. -/
theorem inner_dims_divisible :
    2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by
  decide

/-- Q2_0 bytes of the two fused expert tensors and their sum (about 0.71 GB). -/
theorem q2_0_expert_bytes :
    expertBytes (512 * 1280 * 2560) 64 18 = 471859200 ∧
    expertBytes (512 * 2560 * 640) 64 18 = 235929600 ∧
    expertBytes (512 * 1280 * 2560) 64 18 + expertBytes (512 * 2560 * 640) 64 18 = 707788800 := by
  decide

/-- Q4_0 bytes of the two fused expert tensors and their sum (about 1.42 GB). -/
theorem q4_0_expert_bytes :
    expertBytes (512 * 1280 * 2560) 32 18 + expertBytes (512 * 2560 * 640) 32 18 = 1415577600 := by
  decide

/-- Q8_0 bytes of the two fused expert tensors and their sum (about 2.67 GB). -/
theorem q8_0_expert_bytes :
    expertBytes (512 * 1280 * 2560) 32 34 + expertBytes (512 * 2560 * 640) 32 34 = 2673868800 := by
  decide

/-- Q2_0 is exactly half of Q4_0 on the experts (2 bits vs 4, same 2-byte scale per 2x elements). -/
theorem q2_0_half_of_q4_0 :
    2 * (expertBytes (512 * 1280 * 2560) 64 18 + expertBytes (512 * 2560 * 640) 64 18) =
      expertBytes (512 * 1280 * 2560) 32 18 + expertBytes (512 * 2560 * 640) 32 18 := by
  decide

end Ref.MtpPack



## FILE: /mnt/user-data/outputs/PackLayerBlob.lean
/-!
# MODULE: pack_layer_blob
SOURCES: tools/pack_layer.py
Builds the pack-layout expert arena from Q2_0 GGUF tensors. One expert blob is exactly 1,382,400 bytes:
gate/up codes (rows interleaved g0,u0,g1,u1,...), down codes, gate/up fp16 scales, down fp16 scales.

## Source semantics
* H = 2560, FF = 640, NE = 512, Q2_0 blocks of 64 elements and 18 bytes (2 scale + 16 code bytes).
* Each role holds 460,800 bytes per expert; the blob is three roles. The four planes tile the blob with no
  gap or overlap.
* The conversion is a pure relayout: gate/up row `r` of role gate/up goes to slot `2r` / `2r+1`.
* Expert `e` is one contiguous run in each role tensor; the 48-layer arena is 33,973,862,400 bytes.

## Form template identifiers
- BLOB_BYTES / ROLE_BYTES        → `blobBytes` / `roleBytes`
- blob_offsets()                 → `offGuCodes`, `offDCodes`, `offGuScales`, `offDScales`
- np.stack([gate, up], axis=2)   → `slot`
- ROW_BYTES_GU / ROW_BYTES_D     → `rowBytesGu` / `rowBytesD`
-/

namespace Ref.PackLayer

-- § constants

/-- Hidden size. -/
def H : Nat := 2560
/-- FFN width. -/
def FF : Nat := 640
/-- Experts per layer. -/
def NE : Nat := 512
/-- Bytes of codes in one gate/up row (2560 elements at 2 bits). -/
def rowBytesGu : Nat := 640
/-- Bytes of codes in one down row (640 elements at 2 bits). -/
def rowBytesD : Nat := 160
/-- Q2_0 blocks per gate/up row. -/
def scalesGu : Nat := 40
/-- Q2_0 blocks per down row. -/
def scalesD : Nat := 10
/-- Bytes of one role of one expert. -/
def roleBytes : Nat := 460800
/-- Bytes of one expert blob. -/
def blobBytes : Nat := 1382400

/-- Row code widths follow from element counts at 2 bits per code. -/
theorem row_code_bytes : H * 2 / 8 = rowBytesGu ∧ FF * 2 / 8 = rowBytesD := by
  decide

/-- Blocks per row: 2560/64 and 640/64. -/
theorem blocks_per_row : H / 64 = scalesGu ∧ FF / 64 = scalesD := by
  decide

/-- A Q2_0 block is 16 code bytes plus a 2-byte scale. -/
theorem block_split : 16 + 2 = 18 ∧ 64 * 2 / 8 = 16 := by
  decide

/-- One role of one expert: `H*FF/64` blocks of 18 bytes. -/
theorem role_bytes_eq : H * FF / 64 * 18 = roleBytes := by
  decide

/-- Three roles make the blob: 3 * 460,800 = 1,382,400. -/
theorem blob_eq_three_roles : 3 * roleBytes = blobBytes := by
  decide

/-- The blob equals the sum of its four planes. -/
theorem blob_planes_sum :
    2 * 640 * rowBytesGu + 2560 * rowBytesD + 2 * 640 * scalesGu * 2 + 2560 * scalesD * 2 = blobBytes := by
  decide

-- § plane offsets

/-- Offset of the gate/up codes plane. -/
def offGuCodes : Nat := 0
/-- Offset of the down codes plane. -/
def offDCodes : Nat := offGuCodes + 2 * FF * rowBytesGu
/-- Offset of the gate/up scales plane. -/
def offGuScales : Nat := offDCodes + H * rowBytesD
/-- Offset of the down scales plane. -/
def offDScales : Nat := offGuScales + 2 * FF * scalesGu * 2

/-- The planes are contiguous and the last one ends exactly at the blob size. -/
theorem planes_tile_blob :
    offDCodes = 819200 ∧ offGuScales = 1228800 ∧ offDScales = 1331200 ∧
      offDScales + H * scalesD * 2 = blobBytes := by
  decide

/-- Each plane sits strictly after the previous one. -/
theorem planes_ordered : offGuCodes < offDCodes ∧ offDCodes < offGuScales ∧ offGuScales < offDScales := by
  decide

-- § row interleave and inverse

/-- Slot of a gate (role 0) or up (role 1) row in the interleaved plane. -/
def slot (row role : Nat) : Nat := 2 * row + role

/-- Interleaved slots cover 0..1279 and determine (row, role) uniquely. -/
theorem slot_bijective (r1 r2 a1 a2 : Nat) (h1 : r1 < 640) (h2 : r2 < 640) (b1 : a1 < 2) (b2 : a2 < 2)
    (h : slot r1 a1 = slot r2 a2) : r1 = r2 ∧ a1 = a2 := by
  unfold slot at h
  omega

/-- Slots stay inside the 1280-row interleaved plane. -/
theorem slot_lt (r a : Nat) (h1 : r < 640) (h2 : a < 2) : slot r a < 1280 := by
  unfold slot
  omega

/-- A block index splits into (row, block-in-row) and recombines (`rebuild_role` inverts the layout). -/
theorem block_split_recombine (blk per : Nat) (hp : 0 < per) :
    per * (blk / per) + blk % per = blk ∧ blk % per < per := by
  exact ⟨Nat.div_add_mod blk per, Nat.mod_lt blk hp⟩

/-- The 16 code bytes of block `b` inside a gate/up row end within the 640-byte row. -/
theorem code_run_inside_row (b : Nat) (h : b < 40) : b * 16 + 16 ≤ rowBytesGu := by
  unfold rowBytesGu
  omega

/-- The 2-byte scale of slot `s`, block `b` ends within the gate/up scales plane (102,400 bytes). -/
theorem scale_inside_plane (s b : Nat) (hs : s < 1280) (hb : b < 40) : (s * 40 + b) * 2 + 2 ≤ 102400 := by
  omega

-- § expert contiguity and arena size

/-- Expert `e` is one contiguous run: a role tensor of `H*FF*NE` elements splits into 512 equal runs of
460,800 bytes. -/
theorem expert_run_bytes : (H * FF * NE) / NE * 18 / 64 = roleBytes := by
  decide

/-- Total arena for `n` layers. -/
def arenaBytes (n : Nat) : Nat := n * NE * blobBytes

/-- The full 48-layer arena is 33,973,862,400 bytes (31.64 GiB). -/
theorem arena_48_layers : arenaBytes 48 = 33973862400 := by
  decide

/-- The arena grows linearly: one more layer adds exactly 512 blobs. -/
theorem arena_succ (n : Nat) : arenaBytes (n + 1) = arenaBytes n + NE * blobBytes := by
  unfold arenaBytes NE blobBytes
  omega

end Ref.PackLayer



## FILE: /mnt/user-data/outputs/ServeServerProtocol.lean
/-!
# MODULE: serve_server_protocol
SOURCES: serve/server.py
OpenAI / Anthropic endpoints over a token-id engine: context admission, image-pad expansion, incremental
detokenisation, stop reasons, content-block opening, the image-embedding cache bound and API-key checks.

## Source semantics
* A request is accepted only if `len(prompt) + max_new <= max_context`; requests are rejected, never truncated.
* Each `<|image_pad|>` is replaced by as many pads as its image has tokens; text without pads is unchanged.
* The detokenizer emits `text[sent:]` of the re-decoded ids, so the deltas concatenate to the full text.
* OpenAI finish: `tool_calls` iff a call was emitted and generation stopped; otherwise `stop`/`length`.
  Anthropic: `tool_use` iff a tool was used and generation stopped; `end_turn` for stop, `max_tokens` for length.
* A tool call always opens a new content block; reasoning/text blocks continue while the kind is unchanged.
* The image cache holds at most 64 entries (oldest evicted first).
* With an empty API key everything is authorised; otherwise the presented key must equal it.

## Form template identifiers
- len(ids)+max_new > max_context → `accepts`
- the `<|image_pad|>` loop        → `expand`
- Detokenizer.push               → `deltaOf`
- finish mapping                 → `openaiFinish` / `anthropicStop`
- block open decision            → `newBlock`
- cache eviction                 → `evict`
- Service._authorized            → `authorized`
-/

namespace Ref.ServeServer

-- § context admission

/-- The context check of `Service.prepare`. -/
def accepts (promptLen maxNew ctx : Nat) : Prop := promptLen + maxNew ≤ ctx

/-- An accepted request fits wholesale: nothing needs truncating. -/
theorem accepted_fits (p m c : Nat) (h : accepts p m c) : p ≤ c ∧ m ≤ c := by
  unfold accepts at h
  omega

/-- A request over the context is not accepted. -/
theorem over_context_rejected (p m c : Nat) (h : c < p + m) : ¬ accepts p m c := by
  unfold accepts
  omega

-- § image pad expansion

/-- Replace each pad token `p` by `c` pads, consuming one count per pad. -/
def expand (p : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | t :: ts, cs =>
    if t = p then
      (match cs with
        | [] => t :: expand p ts []
        | c :: cs' => List.replicate c p ++ expand p ts cs')
    else t :: expand p ts cs

/-- Without pad tokens the prompt is unchanged. -/
theorem expand_no_pad (p : Nat) :
    ∀ (ts cs : List Nat), (∀ t ∈ ts, t ≠ p) → expand p ts cs = ts := by
  intro ts
  induction ts with
  | nil => intro cs _; rfl
  | cons t ts ih =>
    intro cs h
    have ht : t ≠ p := h t (by simp)
    have hr : ∀ x ∈ ts, x ≠ p := fun x hx => h x (by simp [hx])
    simp [expand, ht, ih cs hr]

/-- One pad and one image of `c` tokens expands to exactly `c` pads. -/
theorem expand_single (p c : Nat) : expand p [p] [c] = List.replicate c p := by
  simp [expand]

/-- A prompt of one pad per image grows to exactly the total image-token count. -/
theorem expand_length (p : Nat) (cs : List Nat) :
    (expand p (List.replicate cs.length p) cs).length = cs.sum := by
  induction cs with
  | nil => simp [expand]
  | cons c cs ih => simp [expand, List.replicate_succ, ih]

-- § incremental detokenisation

/-- The new suffix emitted when the decoded text grows from `old` to `new`. -/
def deltaOf (old new : List Nat) : List Nat := new.drop old.length

/-- If decoding is prefix-extending, the delta is exactly the appended text. -/
theorem delta_of_append (old x : List Nat) : deltaOf old (old ++ x) = x := by
  simp [deltaOf]

/-- Successive deltas concatenate back to the final text. -/
theorem deltas_concatenate (a x y : List Nat) :
    a ++ deltaOf a (a ++ x) ++ deltaOf (a ++ x) (a ++ x ++ y) = a ++ x ++ y := by
  simp [deltaOf]

-- § stop reasons and content blocks

/-- Generation finish reasons from the engine. -/
inductive Finish where
  | stop
  | length
  deriving DecidableEq

/-- OpenAI finish reason code: 0 = stop, 1 = length, 2 = tool_calls. -/
def openaiFinish (calls : Nat) (f : Finish) : Nat :=
  if calls > 0 ∧ f = Finish.stop then 2 else match f with
    | Finish.stop => 0
    | Finish.length => 1

/-- `tool_calls` is reported only when generation stopped normally. -/
theorem openai_tool_calls_only_on_stop (calls : Nat) (f : Finish) (h : openaiFinish calls f = 2) :
    f = Finish.stop := by
  unfold openaiFinish at h
  cases f <;> simp_all

/-- A truncated generation is still reported as `length`, even with calls emitted. -/
theorem openai_length_preserved (calls : Nat) : openaiFinish calls Finish.length = 1 := by
  simp [openaiFinish]

/-- Anthropic stop reason code: 0 = end_turn, 1 = max_tokens, 2 = tool_use. -/
def anthropicStop (usedTool : Bool) (f : Finish) : Nat :=
  if usedTool = true ∧ f = Finish.stop then 2 else match f with
    | Finish.stop => 0
    | Finish.length => 1

/-- `tool_use` only on a normal stop. -/
theorem anthropic_tool_use_only_on_stop (u : Bool) (f : Finish) (h : anthropicStop u f = 2) :
    f = Finish.stop := by
  unfold anthropicStop at h
  cases f <;> simp_all

/-- Kinds of streamed events. -/
inductive Kind where
  | reasoning
  | content
  | tool
  deriving DecidableEq

/-- Whether an event of kind `k` opens a new block given the currently open kind. -/
def newBlock (cur : Option Kind) (k : Kind) : Bool :=
  match k with
  | Kind.tool => true
  | _ => decide (cur ≠ some k)

/-- A tool call always opens a fresh block. -/
theorem tool_always_new (cur : Option Kind) : newBlock cur Kind.tool = true := by
  rfl

/-- Consecutive text deltas continue the same block. -/
theorem text_continues : newBlock (some Kind.content) Kind.content = false := by
  decide

/-- Switching from reasoning to content opens a new block. -/
theorem switch_opens : newBlock (some Kind.reasoning) Kind.content = true := by
  decide

-- § image cache bound

/-- Drop the oldest entry when the cache exceeds 64. -/
def evict {α : Type} (l : List α) : List α := if l.length > 64 then l.drop 1 else l

/-- Inserting into a cache of at most 64 entries and evicting keeps it at most 64. -/
theorem evict_bound {α : Type} (l : List α) (h : l.length ≤ 65) : (evict l).length ≤ 64 := by
  unfold evict
  split
  · rw [List.length_drop]
    omega
  · omega

-- § authorisation

/-- The presented key is accepted when no key is configured or it matches. -/
def authorized (configured presented : String) : Prop := configured = "" ∨ presented = configured

/-- With no configured key, anything is accepted. -/
theorem no_key_open (presented : String) : authorized "" presented := by
  exact Or.inl rfl

/-- The right key is always accepted. -/
theorem right_key_accepted (k : String) : authorized k k := by
  exact Or.inr rfl

end Ref.ServeServer



## FILE: /mnt/user-data/outputs/RefPackBuilder.lean
/-!
# MODULE: ref_pack_builder
SOURCES: tools/ref_pack.py
The pack builder and verifier: tensors are canonicalised into planes (codes, scales, offsets), each plane
starting on a 64-byte boundary, and `verify` decodes the pack from disk and compares it bit for bit.

## Source semantics
* `pack_codes` writes `8 / bits` codes per byte, LSB first; `unpack_codes` inverts it for in-range codes.
* `put` pads the running body to a 64-byte boundary before each plane. The ABSOLUTE offset is aligned when
  the segment base is aligned; segment bases advance by unpadded blob lengths, so alignment of later tensors
  holds only if each blob length is a multiple of 64 (a property the verifier does not check).
* The expert arena is addressed `(layer * 512 + e) * blob_bytes`; `verify_experts` samples every Nth expert.
* Scales are stored FP16 only when the FP16 round trip is exact, otherwise FP32.

## Form template identifiers
- pack_codes / unpack_codes      → `packLSB` / `unpackLSB`
- put() padding                  → `padTo64`
- (layer*NE + e) * blob          → `expertIndex`
- code_bits(form)                → `codeBits`
-/

namespace Ref.PackBuilder

-- § LSB-first code packing

/-- Pack codes LSB first, `b` bits per code. -/
def packLSB (b : Nat) : List Nat → Nat
  | [] => 0
  | c :: cs => c + 2 ^ b * packLSB b cs

/-- Unpack `n` codes of `b` bits, LSB first. -/
def unpackLSB (b : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | n + 1, v => (v % 2 ^ b) :: unpackLSB b n (v / 2 ^ b)

/-- Unpacking inverts packing for in-range codes, for every code width. -/
theorem unpack_pack (b : Nat) :
    ∀ cs : List Nat, (∀ c ∈ cs, c < 2 ^ b) → unpackLSB b cs.length (packLSB b cs) = cs := by
  intro cs
  induction cs with
  | nil => intro _; rfl
  | cons c cs ih =>
    intro h
    have hc : c < 2 ^ b := h c (by simp)
    have hr : ∀ x ∈ cs, x < 2 ^ b := fun x hx => h x (by simp [hx])
    have hpos : 0 < 2 ^ b := Nat.two_pow_pos b
    have h1 : (c + 2 ^ b * packLSB b cs) % 2 ^ b = c := by
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hc]
    have h2 : (c + 2 ^ b * packLSB b cs) / 2 ^ b = packLSB b cs := by
      rw [Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hc, Nat.zero_add]
    simp only [List.length_cons, unpackLSB, packLSB, h1, h2, ih hr]

/-- Canonical form widths. -/
def codeBits : String → Nat
  | "S2" => 2
  | "S4" => 4
  | "S8" => 8
  | _ => 0

/-- Every canonical width fills a byte with a whole number of codes. -/
theorem code_bits_fill :
    codeBits "S2" * (8 / codeBits "S2") = 8 ∧ codeBits "S4" * (8 / codeBits "S4") = 8 ∧
      codeBits "S8" * (8 / codeBits "S8") = 8 := by
  decide

-- § 64-byte plane alignment

/-- Padding `put` inserts: `(-len) % 64`. -/
def padTo64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- The padded body length is a multiple of 64. -/
theorem padTo64_aligned (n : Nat) : (n + padTo64 n) % 64 = 0 := by
  unfold padTo64
  omega

/-- Padding is smaller than one alignment unit. -/
theorem padTo64_lt (n : Nat) : padTo64 n < 64 := by
  unfold padTo64
  omega

/-- If the segment base is aligned, every plane's absolute offset is aligned. -/
theorem plane_offset_aligned (base len : Nat) (h : base % 64 = 0) :
    (base + (len + padTo64 len)) % 64 = 0 := by
  unfold padTo64
  omega

/-- If the base is NOT aligned and a blob length is not a multiple of 64, the next base can be unaligned:
(base 0, blob 10) leaves the next segment at 10. -/
theorem unaligned_blob_breaks_next_base : (0 + 10) % 64 ≠ 0 := by
  decide

-- § expert arena addressing

/-- Index of expert `e` of layer `l` in the arena (in blobs). -/
def expertIndex (l e : Nat) : Nat := l * 512 + e

/-- Distinct (layer, expert) pairs get distinct arena slots. -/
theorem expertIndex_injective (l1 l2 e1 e2 : Nat) (h1 : e1 < 512) (h2 : e2 < 512)
    (h : expertIndex l1 e1 = expertIndex l2 e2) : l1 = l2 ∧ e1 = e2 := by
  unfold expertIndex at h
  omega

/-- Sampling every 32nd expert checks 16 of the 512 per layer. -/
theorem sampled_experts : 512 / 32 = 16 := by
  decide

/-- The sampled expert indices `0, 32, ..., 480` are all in range. -/
theorem sampled_in_range (k : Nat) (h : k < 16) : k * 32 < 512 := by
  omega

-- § plane sizes

/-- Scale plane bytes for `n` scales at FP16 (2 bytes) or FP32 (4 bytes). -/
def scalePlaneBytes (n : Nat) (fp16 : Bool) : Nat := if fp16 then n * 2 else n * 4

/-- FP16 planes are exactly half the size of FP32 planes. -/
theorem fp16_half (n : Nat) : 2 * scalePlaneBytes n true = scalePlaneBytes n false := by
  unfold scalePlaneBytes
  simp <;> omega

/-- Codes plane bytes for `n` codes at `bits` bits, when `n` is a whole number of bytes. -/
theorem codes_plane_bytes (n per : Nat) (h : 0 < per) : (n * per) / per = n := by
  exact Nat.mul_div_cancel n h

end Ref.PackBuilder



## FILE: /mnt/user-data/outputs/RefTokenizerBpe.lean
/-!
# MODULE: ref_tokenizer_bpe
SOURCES: tools/ref_tokenizer.py
Reference byte-level BPE tokenizer (GPT-2 style, `qwen35` pre-tokenizer): the byte to printable-unicode map,
merge-by-lowest-rank, and the special-token matching rules.

## Source semantics
* `bytes_to_unicode`: the 188 bytes in 33..126, 161..172, 174..255 map to themselves; the other 68 map to
  `256 + n` in byte order, so the image is in 0..255 (printable) or 256..323 (shifted).
* BPE repeatedly merges ONE adjacent pair (the lowest-rank one); every merge shortens the sequence by one.
* GGUF token types: type 4 (USER_DEFINED) is matched always; type 3 (CONTROL) only when `parse_special`.
* The vocabulary (248,320) holds the 256 byte tokens, the 247,587 merges' results and the specials.

## Form template identifiers
- bytes_to_unicode               → `b2u` (with `isPrint`, `rank`)
- Tokenizer._bpe merge step      → `mergeAt`
- special_tokens / _always_re    → `matched`
- tokenizer.ggml.tokens / merges → `vocabSize` / `nMerges`
-/

namespace Ref.TokenizerBpe

-- § byte <-> unicode

/-- Bytes that map to themselves. -/
def isPrint (b : Nat) : Bool :=
  (decide (33 ≤ b) && decide (b ≤ 126)) || (decide (161 ≤ b) && decide (b ≤ 172)) ||
    (decide (174 ≤ b) && decide (b ≤ 255))

/-- Number of shifted bytes before `b`. -/
def rank (b : Nat) : Nat := (List.range b).countP (fun x => !isPrint x)

/-- The GPT-2 byte encoder as code points. -/
def b2u (b : Nat) : Nat := if isPrint b then b else 256 + rank b

/-- Exactly 188 bytes are printable and 68 are shifted. -/
theorem printable_counts :
    (List.range 256).countP isPrint = 188 ∧ (List.range 256).countP (fun x => !isPrint x) = 68 := by
  decide +kernel

/-- Printable bytes map into 0..255 and shifted bytes into 256..323, so the two classes never collide. -/
theorem b2u_range : ∀ b, b < 256 → b2u b < 324 := by
  decide +kernel

/-- Printable bytes are fixed points; shifted bytes land strictly above 255. -/
theorem b2u_classes : ∀ b, b < 256 → (isPrint b = true → b2u b = b) ∧ (isPrint b = false → 256 ≤ b2u b) := by
  decide +kernel

/-- The encoder is injective on bytes, which is what makes `decode (encode s) = s` exact. -/
theorem b2u_nodup : ((List.range 256).map b2u).Nodup := by
  decide +kernel

-- § merging

/-- Merge the adjacent pair at index `i` with the combiner `f`. -/
def mergeAt {α : Type} (f : α → α → α) : List α → Nat → List α
  | a :: b :: rest, 0 => f a b :: rest
  | a :: rest, i + 1 => a :: mergeAt f rest i
  | l, _ => l

/-- Each merge of a valid pair shortens the sequence by exactly one, so BPE terminates. -/
theorem mergeAt_length {α : Type} (f : α → α → α) :
    ∀ (l : List α) (i : Nat), i + 1 < l.length → (mergeAt f l i).length + 1 = l.length := by
  intro l
  induction l with
  | nil => intro i h; simp at h
  | cons a rest ih =>
    intro i h
    cases i with
    | zero =>
      cases rest with
      | nil => simp at h
      | cons b rest' => simp [mergeAt]
    | succ i =>
      have h' : i + 1 < rest.length := by
        simp at h
        omega
      have := ih i h'
      simp [mergeAt] <;> omega

/-- An index past the end leaves the sequence unchanged. -/
theorem mergeAt_nil {α : Type} (f : α → α → α) (i : Nat) : mergeAt f [] i = [] := by
  cases i <;> rfl

-- § special tokens

/-- Whether a token of GGUF type `ty` is matched as a literal: type 4 always, type 3 only with
`parse_special`. -/
def matched (ps : Bool) (ty : Nat) : Prop := ty = 4 ∨ (ty = 3 ∧ ps = true)

/-- Everything matched without `parse_special` is also matched with it. -/
theorem matched_monotone (ty : Nat) (h : matched false ty) : matched true ty := by
  rcases h with h | ⟨_, h'⟩
  · exact Or.inl h
  · exact absurd h' (by decide)

/-- Without `parse_special`, CONTROL tokens such as `<|im_end|>` (type 3) are not matched. -/
theorem control_not_matched_by_default : ¬ matched false 3 := by
  intro h
  rcases h with h | ⟨_, h'⟩
  · exact absurd h (by decide)
  · exact absurd h' (by decide)

/-- USER_DEFINED tokens such as `<think>` (type 4) are matched either way. -/
theorem user_defined_always_matched (ps : Bool) : matched ps 4 := by
  exact Or.inl rfl

-- § vocabulary arithmetic

/-- Vocabulary and merge counts of the GGUF. -/
def vocabSize : Nat := 248320
/-- Number of merge rules. -/
def nMerges : Nat := 247587

/-- The 256 byte tokens plus one token per merge fit in the vocabulary, leaving 477 for specials etc. -/
theorem vocab_covers_merges : 256 + nMerges ≤ vocabSize ∧ vocabSize - (256 + nMerges) = 477 := by
  decide

end Ref.TokenizerBpe



## FILE: /mnt/user-data/outputs/VisionBuildConfig.lean
/-!
# MODULE: vision_build_config
SOURCES: tools/vision/CMakeLists.txt
Build configuration of `ref-vision`: llama.cpp's mtmd library at the engine's pinned commit, with two
user options (`REF_VISION_CUDA`, `REF_PORTABLE`) that determine the ggml flags.

## Source semantics
* Always off (forced): tools, tests, examples, server, app, common, OpenSSL, curl, shared libs, backend-DL.
  Always on: `LLAMA_BUILD_MTMD`.
* `REF_PORTABLE` turns native tuning off, OpenMP off, and the AVX/AVX2/FMA/F16C/BMI2 baseline on;
  otherwise `GGML_NATIVE` is on. The static MSVC runtime applies only to portable MSVC builds.
* `GGML_CUDA` follows `REF_VISION_CUDA` exactly.
* Minimum CMake 3.21 and C++17.

## Form template identifiers
- REF_PORTABLE / REF_VISION_CUDA → `Config.portable` / `Config.cuda`
- GGML_NATIVE / GGML_OPENMP / GGML_CUDA → `native` / `openmp` / `cudaOn`
- CMAKE_MSVC_RUNTIME_LIBRARY            → `staticRuntime`
-/

namespace Ref.VisionBuild

-- § options

/-- The user-visible options. -/
structure Config where
  portable : Bool
  cuda : Bool
  msvc : Bool

/-- `GGML_NATIVE`. -/
def native (c : Config) : Bool := !c.portable

/-- `GGML_OPENMP` is forced off only in portable builds (left at the default otherwise). -/
def openmpForcedOff (c : Config) : Bool := c.portable

/-- The AVX2-family baseline flags are forced on only in portable builds. -/
def baselineIsa (c : Config) : Bool := c.portable

/-- `GGML_CUDA`. -/
def cudaOn (c : Config) : Bool := c.cuda

/-- Static MSVC runtime: portable and MSVC. -/
def staticRuntime (c : Config) : Bool := c.portable && c.msvc

/-- `GGML_BACKEND_DL` is always off. -/
def backendDl (_ : Config) : Bool := false

-- § consistency

/-- Native tuning and the portable baseline are mutually exclusive. -/
theorem native_xor_portable (c : Config) : ¬ (native c = true ∧ c.portable = true) := by
  unfold native
  intro h
  obtain ⟨h1, h2⟩ := h
  simp [h2] at h1

/-- Exactly one of {native tuning, portable baseline ISA} is selected. -/
theorem exactly_one_isa_mode (c : Config) : native c = !baselineIsa c := by
  rfl

/-- The CUDA flag follows the option exactly. -/
theorem cuda_follows_option (c : Config) : cudaOn c = c.cuda := by
  rfl

/-- The static runtime applies only to portable builds. -/
theorem static_runtime_needs_portable (c : Config) (h : staticRuntime c = true) : c.portable = true := by
  unfold staticRuntime at h
  cases hp : c.portable <;> simp_all

/-- A non-portable build never forces OpenMP off nor the static runtime. -/
theorem non_portable_defaults (c : Config) (h : c.portable = false) :
    openmpForcedOff c = false ∧ staticRuntime c = false := by
  unfold openmpForcedOff staticRuntime
  simp [h]

/-- Backend DL is off for every configuration. -/
theorem backend_dl_never (c : Config) : backendDl c = false := by
  rfl

-- § toolchain floor

/-- CMake 3.21 and C++17 minimums are ordered as stated. -/
theorem toolchain_floor : 3 * 100 + 21 = 321 ∧ 17 ≥ 17 := by
  decide

end Ref.VisionBuild



## FILE: /mnt/user-data/outputs/VisionEncoderProtocol.lean
/-!
# MODULE: vision_encoder_protocol
SOURCES: tools/vision/ref_vision.cpp
The resident image encoder: line protocol `ENC <image> <out>` -> `OK n nx ny ms` / `ERR msg`, and the SVE1
embeddings file (5 x int32 header then `n x n_embd` float32 rows in row-major grid order).

## Source semantics
* `parse_enc` splits the argument at the LAST space: image path may contain spaces, the output path may not;
  both parts must be non-empty.
* The header magic `0x31455653` is "SVE1" in little-endian bytes.
* The grid comes from the last token's decoder position: `nx = x+1`, `ny = y+1`, and the encoder insists
  `nx * ny = n` (rectangular). Row `i` sits at `(x, y) = (i % nx, i / nx)`.
* File size is `20 + 4 * n * n_embd` bytes.

## Form template identifiers
- parse_enc (rfind ' ')          → `lastSplit` / `parseEnc`
- hdr[5] = {magic,n,nx,ny,n_embd} → `headerBytes`
- row i at (i % nx, i / nx)      → `gridPos`
- last decoder pos + 1           → `last_token_grid`
-/

namespace Ref.VisionEncoder

-- § line parsing

/-- Split a list at its LAST space (code 32): (before, after). -/
def lastSplit : List Nat → Option (List Nat × List Nat)
  | [] => none
  | x :: xs =>
    match lastSplit xs with
    | some (b, a) => some (x :: b, a)
    | none => if x = 32 then some ([], xs) else none

/-- A list without a space has no split point. -/
theorem lastSplit_none : ∀ l : List Nat, (∀ y ∈ l, y ≠ 32) → lastSplit l = none := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons x xs ih =>
    intro h
    have hx : x ≠ 32 := h x (by simp)
    have hr : ∀ y ∈ xs, y ≠ 32 := fun y hy => h y (by simp [hy])
    simp [lastSplit, ih hr, hx]

/-- When the output path has no space, the split recovers (image, output) exactly. -/
theorem lastSplit_append (out : List Nat) (h : ∀ y ∈ out, y ≠ 32) :
    ∀ img : List Nat, lastSplit (img ++ 32 :: out) = some (img, out) := by
  intro img
  induction img with
  | nil => simp [lastSplit, lastSplit_none out h]
  | cons x xs ih => simp [lastSplit, ih]

/-- The `parse_enc` result: both parts must be non-empty. -/
def parseEnc (l : List Nat) : Option (List Nat × List Nat) :=
  match lastSplit l with
  | none => none
  | some (b, a) => if b.isEmpty || a.isEmpty then none else some (b, a)

/-- Well-formed `image <space> output` lines parse back to their two parts. -/
theorem parseEnc_roundtrip (x : Nat) (xs : List Nat) (y : Nat) (ys : List Nat)
    (h : ∀ z ∈ y :: ys, z ≠ 32) :
    parseEnc ((x :: xs) ++ 32 :: (y :: ys)) = some (x :: xs, y :: ys) := by
  have hs : lastSplit (x :: (xs ++ 32 :: y :: ys)) = some (x :: xs, y :: ys) :=
    lastSplit_append (y :: ys) h (x :: xs)
  simp [parseEnc, hs]

/-- A line starting with the separator (empty image path, `sp == 0`) is rejected. -/
theorem parseEnc_empty_image (out : List Nat) (h : ∀ y ∈ out, y ≠ 32) : parseEnc (32 :: out) = none := by
  simp [parseEnc, lastSplit, lastSplit_none out h]

-- § SVE1 file format

/-- The header magic. -/
def sveMagic : Nat := 0x31455653

/-- Little-endian bytes of the magic spell S, V, E, 1. -/
theorem sve_magic_bytes :
    sveMagic % 256 = 83 ∧ sveMagic / 256 % 256 = 86 ∧ sveMagic / 65536 % 256 = 69 ∧
      sveMagic / 16777216 = 49 := by
  decide

/-- Header size: five int32 values. -/
def headerBytes : Nat := 5 * 4

/-- Total file size for `n` tokens of width `nEmbd`. -/
def fileBytes (n nEmbd : Nat) : Nat := headerBytes + 4 * n * nEmbd

/-- The header is 20 bytes. -/
theorem header_is_20 : headerBytes = 20 := by
  decide

/-- File size grows by exactly one row of `4 * n_embd` bytes per extra token. -/
theorem fileBytes_succ (n nEmbd : Nat) : fileBytes (n + 1) nEmbd = fileBytes n nEmbd + 4 * nEmbd := by
  unfold fileBytes headerBytes
  rw [Nat.mul_add 4 n 1, Nat.add_mul]
  omega

-- § grid

/-- Position of token `i` in a grid of width `nx`: (x, y). -/
def gridPos (nx i : Nat) : Nat × Nat := (i % nx, i / nx)

/-- Every token of an `nx x ny` grid sits inside it, and (x, y) recombines to the index. -/
theorem gridPos_in_bounds (nx ny i : Nat) (hnx : 0 < nx) (hi : i < nx * ny) :
    (gridPos nx i).1 < nx ∧ (gridPos nx i).2 < ny ∧ (gridPos nx i).2 * nx + (gridPos nx i).1 = i := by
  unfold gridPos
  refine ⟨Nat.mod_lt _ hnx, Nat.div_lt_of_lt_mul hi, ?_⟩
  have := Nat.div_add_mod i nx
  rw [Nat.mul_comm] at this
  exact this

/-- The last token of a `(j+1) x (k+1)` grid is at (j, k), so `nx = x+1`, `ny = y+1` recovers the grid. -/
theorem last_token_grid (j k : Nat) :
    ((j + 1) * (k + 1) - 1) % (j + 1) = j ∧ ((j + 1) * (k + 1) - 1) / (j + 1) = k := by
  have h : (j + 1) * (k + 1) - 1 = j + (j + 1) * k := by
    have := Nat.mul_add (j + 1) k 1
    omega
  have hp : 0 < j + 1 := Nat.succ_pos j
  rw [h]
  constructor
  · rw [Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt (Nat.lt_succ_self j)
  · rw [Nat.add_mul_div_left _ _ hp, Nat.div_eq_of_lt (Nat.lt_succ_self j)]
    omega

end Ref.VisionEncoder


