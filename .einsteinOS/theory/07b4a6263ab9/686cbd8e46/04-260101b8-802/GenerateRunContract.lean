/-!
# MODULE: generate_run_contract
SOURCES: src/program/generate.cpp
The refusal rules `strata generate` applies before it allocates anything: token-id parsing, the
prompt/generation/context fit, the int32 position cap, the PCIe share clamp, the routing-trace record
and its layer index, the logits-dump row count, and the split-GGUF shard enumeration.

## Source semantics
* `parse_i64_list` accepts a non-empty list of decimal ids in `[0, 2^31-1]`; anything else is refused.
* A run is admitted only if `max_new > 0`, `max_context > 0`, `max_new ≤ max_context` and
  `n_prompt ≤ max_context - max_new`; `max_context` must stay below `2^31 - 8` because positions,
  cells and pooled-block indices are cast to int32 on the device.
* `pcie_num = clamp(round(pcie_frac * 256), 0, 256)`.
* The routing trace is written after the dispatch, when the adapter has already advanced `d.layers`,
  so the recorded layer is `d.layers - 1`, and only `0..47` is accepted.  A record is
  `int32 layer, int32 k, k int32 ids, k float weights` = `8 + 8k` bytes.
* The logits dump has `n_prompt - 1 + max_new` positions; with stride 1 every position is a row.
* `model_shards` probes shard numbers `1..min(total, 99)` and keeps the ones that exist.

## Form template identifiers
parse_i64_list → parseIds · Options{tokens,max_new,max_context} → RunShape · the fit check → fitsContext ·
pcie_num → pcieNum · drive_pool's trace → traceLayer / traceRecordBytes ·
logits_selection::selected → selected · model_shards → shardCandidates
-/

namespace Strata.Generate.RunContract

-- § token id parsing

/-- The largest id `std::from_chars<int32_t>` can return. -/
def idMax : Int := 2147483647

/-- `parse_i64_list`: accept a non-empty list of ids that are all in `[0, idMax]`. -/
def parseIds (xs : List Int) : Option (List Nat) :=
  if xs ≠ [] ∧ ∀ v ∈ xs, 0 ≤ v ∧ v ≤ idMax then some (xs.map Int.toNat) else none

/-- A successful parse is never empty and every id fits a non-negative int32. -/
theorem parseIds_spec {xs : List Int} {ys : List Nat} (h : parseIds xs = some ys) :
    ys ≠ [] ∧ ∀ y ∈ ys, y ≤ 2147483647 := by
  unfold parseIds at h
  split at h
  · rename_i hc
    obtain ⟨hne, hall⟩ := hc
    simp only [Option.some.injEq] at h
    subst h
    refine ⟨?_, ?_⟩
    · simpa using hne
    · intro y hy
      obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hy
      have hb := hall v hv
      simp only [idMax] at hb
      omega
  · cases h

/-- A negative id anywhere in the list refuses the whole list. -/
theorem parseIds_neg_refused {xs : List Int} {v : Int} (hv : v ∈ xs) (hneg : v < 0) :
    parseIds xs = none := by
  unfold parseIds
  have hbad : ¬ (xs ≠ [] ∧ ∀ v ∈ xs, 0 ≤ v ∧ v ≤ idMax) := by
    intro hc
    have := (hc.2 v hv).1
    omega
  simp only [hbad, if_false]

/-- The empty list is refused ("token list was empty"). -/
theorem parseIds_empty : parseIds [] = none := by
  unfold parseIds
  simp

-- § context fit

/-- The three numbers the fit check reads. -/
structure RunShape where
  nPrompt : Nat
  maxNew : Int
  maxContext : Int

/-- The admission predicate of the fit check in `main`. -/
def fitsContext (s : RunShape) : Prop :=
  0 < s.maxNew ∧ 0 < s.maxContext ∧ s.maxNew ≤ s.maxContext ∧
    (s.nPrompt : Int) ≤ s.maxContext - s.maxNew

/-- An admitted run holds the prompt and every generated token. -/
theorem fits_total {s : RunShape} (h : fitsContext s) :
    (s.nPrompt : Int) + s.maxNew ≤ s.maxContext := by
  obtain ⟨_, _, _, hp⟩ := h
  omega

/-- An admitted run leaves at least one cell after the prompt, so the last prompt position exists. -/
theorem fits_prompt_lt {s : RunShape} (h : fitsContext s) : (s.nPrompt : Int) < s.maxContext := by
  have ht := fits_total h
  have hm := h.1
  omega

/-- With `max_context ≤ 2^31 - 1 - 8`, every position below the context is a valid int32 and so is
the `+8` slack the serve path adds. -/
theorem positions_fit_int32 {ctx pos : Int} (hcap : ctx ≤ 2147483647 - 8) (hp : pos < ctx) :
    pos + 8 < 2147483647 := by
  omega

-- § PCIe share

/-- `drive.d.pcie_num`, clamped to `[0, 256]` after rounding. -/
def pcieNum (r : Int) : Int := max 0 (min 256 r)

/-- The clamped share is a valid numerator over 256. -/
theorem pcieNum_range (r : Int) : 0 ≤ pcieNum r ∧ pcieNum r ≤ 256 := by
  unfold pcieNum
  omega

/-- In-range values pass the clamp unchanged. -/
theorem pcieNum_id {r : Int} (h0 : 0 ≤ r) (h1 : r ≤ 256) : pcieNum r = r := by
  unfold pcieNum
  omega

-- § routing trace

/-- The layer recorded for a dispatch, read after the adapter advanced `d.layers`. -/
def traceLayer (layersAfter : Int) : Int := layersAfter - 1

/-- Whether the trace accepts the record (the bounds check that caught the off-by-one). -/
def traceAccepts (layersAfter : Int) : Prop := 0 ≤ traceLayer layersAfter ∧ traceLayer layersAfter < 48

/-- Dispatching layer `l` leaves `d.layers = l + 1`, and the record names layer `l`. -/
theorem traceLayer_after_dispatch (l : Int) : traceLayer (l + 1) = l := by
  unfold traceLayer
  omega

/-- Every layer of a 48-layer model is accepted when recorded after its own dispatch. -/
theorem trace_accepts_real_layers {l : Int} (h0 : 0 ≤ l) (h1 : l < 48) : traceAccepts (l + 1) := by
  unfold traceAccepts
  rw [traceLayer_after_dispatch]
  exact ⟨h0, h1⟩

/-- Reading the counter as the layer (the first version) would put the last layer out of range. -/
theorem untransformed_counter_overflows : ¬ ((48 : Int) < 48) := by decide

/-- Bytes of one routing record: two int32 headers, `k` int32 ids and `k` floats. -/
def traceRecordBytes (k : Nat) : Nat := 4 + 4 + 4 * k + 4 * k

/-- With the engine's `K = 10`, a record is 88 bytes. -/
theorem traceRecordBytes_k10 : traceRecordBytes 10 = 88 := by decide

-- § logits dump rows

/-- `logits_selection::selected`: every `stride`-th position plus the final one. -/
def selected (p positions stride : Nat) : Bool := p % stride == 0 || p + 1 == positions

/-- Number of positions the decode loop visits: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- With stride 1 every position is selected. -/
theorem selected_stride_one (p positions : Nat) : selected p positions 1 = true := by
  simp [selected, Nat.mod_one]

/-- With stride 1 the number of rows written equals the number of positions, which is what the
header must record. -/
theorem rows_stride_one (positions : Nat) :
    ((List.range positions).filter (fun p => selected p positions 1)).length = positions := by
  have h : ((List.range positions).filter (fun p => selected p positions 1)) = List.range positions := by
    apply List.filter_eq_self.mpr
    intro p _
    exact selected_stride_one p positions
  rw [h, List.length_range]

/-- The header count is one less than `n_prompt + max_new` whenever there is a prompt. -/
theorem dumpPositions_lt {nPrompt maxNew : Nat} (h : 1 ≤ nPrompt) :
    dumpPositions nPrompt maxNew + 1 = nPrompt + maxNew := by
  unfold dumpPositions
  omega

-- § split GGUF shards

/-- The shard numbers `model_shards` probes for a split of `total` shards. -/
def shardCandidates (total : Nat) : List Nat := List.range' 1 (min total 99)

/-- The probe never exceeds 99 shards, whatever the file name claims. -/
theorem shardCandidates_le_99 (total : Nat) (present : Nat → Bool) :
    ((shardCandidates total).filter present).length ≤ 99 := by
  have h1 := List.length_filter_le present (shardCandidates total)
  have h2 : (shardCandidates total).length = min total 99 := by
    simp [shardCandidates]
  omega

/-- Every probed shard number is in `1..99`. -/
theorem shardCandidates_range {total i : Nat} (h : i ∈ shardCandidates total) : 1 ≤ i ∧ i ≤ 99 := by
  unfold shardCandidates at h
  rw [List.mem_range'_1] at h
  omega

end Strata.Generate.RunContract
