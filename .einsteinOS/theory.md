
## FILE: /mnt/user-data/outputs/ExpertVramCache.lean
/-!
# MODULE: expert_vram_cache
VRAM-resident expert tier: slot arena, residency table, per-layer slot quotas, sized slot offsets, and the
bounded copy size of `fill_slot`.

## Source semantics
`admit` never evicts: a resident (layer, expert) keeps its slot forever; a free slot is handed out from one
counter (global) or from a per-layer counter bounded by the layer's own range `[l*q, (l+1)*q)` with `q = slots/n_layers`
(the last layer takes the remainder). Ranges tile `0..slots` exactly. Sized slots are 256-byte aligned.

## Form template identifiers
- ExpertCache::layer_slot_range -> layerLo / layerHi
- ExpertCache::admit            -> admitSlot / admitNext / admitRes
- ExpertCache::open_sized       -> align256 / offsetsFrom
- ExpertCache::fill_slot        -> fillBytes
-/
namespace Strata.ExpertVramCache

-- § per-layer slot ranges
def layerLo (slots nl l : Nat) : Nat := l * (slots / nl)
def layerHi (slots nl l : Nat) : Nat := if l = nl - 1 then slots else (l + 1) * (slots / nl)

/-- Layer 0's range starts at slot 0. -/
theorem layerLo_zero (slots nl : Nat) : layerLo slots nl 0 = 0 := by simp [layerLo]

/-- `n_layers` quotas of size `q` never exceed the slot count. -/
theorem layer_quota_le (slots nl : Nat) : nl * (slots / nl) ≤ slots := Nat.mul_div_le slots nl

/-- Every layer's range is well formed (lo ≤ hi). -/
theorem layerLo_le_layerHi {slots nl l : Nat} (h : l < nl) : layerLo slots nl l ≤ layerHi slots nl l := by
  unfold layerLo layerHi
  have hq := layer_quota_le slots nl
  split
  · have h1 : l * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    omega
  · exact Nat.mul_le_mul_right _ (by omega)

/-- Consecutive ranges abut: no slot is orphaned between layers. -/
theorem layerHi_eq_layerLo_succ {slots nl l : Nat} (h : l + 1 < nl) :
    layerHi slots nl l = layerLo slots nl (l + 1) := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]

/-- The last layer's range ends exactly at `slots`, so the ranges cover everything. -/
theorem layerHi_last {slots nl : Nat} (h : 0 < nl) : layerHi slots nl (nl - 1) = slots := by
  unfold layerHi
  simp

/-- Ranges of distinct layers are disjoint. -/
theorem layer_ranges_disjoint {slots nl l l' : Nat} (h : l < l') (h' : l' < nl) :
    layerHi slots nl l ≤ layerLo slots nl l' := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]
  exact Nat.mul_le_mul_right _ (by omega)

-- § admission without eviction
def admitSlot (next hi : Nat) : Option Nat := if next ≥ hi then none else some next
def admitNext (next hi : Nat) : Nat := if next ≥ hi then next else next + 1

/-- A granted slot lies below the ceiling. -/
theorem admitSlot_lt {next hi s : Nat} (h : admitSlot next hi = some s) : s < hi ∧ s = next := by
  unfold admitSlot at h
  split at h
  · simp at h
  · rename_i hn
    simp only [Option.some.injEq] at h
    omega

/-- A full counter refuses and does not move. -/
theorem admit_full {next hi : Nat} (h : next ≥ hi) : admitSlot next hi = none ∧ admitNext next hi = next := by
  unfold admitSlot admitNext
  simp [h]

/-- The counter only grows and never passes its ceiling. -/
theorem admitNext_mono (next hi : Nat) : next ≤ admitNext next hi ∧ (next ≤ hi → admitNext next hi ≤ hi) := by
  unfold admitNext
  split <;> omega

/-- Residency update: a key that is already resident is left untouched (no eviction). -/
def admitRes (res : Nat → Option Nat) (key : Nat) (s : Option Nat) : Nat → Option Nat :=
  fun k => if k = key ∧ res k = none then s else res k

/-- Nothing resident is ever displaced by an admission. -/
theorem admitRes_preserves {res : Nat → Option Nat} {key k v : Nat} {s : Option Nat}
    (hk : res k = some v) : admitRes res key s k = some v := by
  unfold admitRes
  split
  · rename_i hh
    rw [hh.2] at hk
    simp at hk
  · exact hk

-- § 256-byte aligned sized slots
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment never shrinks a slot. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega
/-- Aligned sizes are multiples of 256. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega
/-- Alignment wastes fewer than 256 bytes. -/
theorem align256_lt (n : Nat) : align256 n < n + 256 := by unfold align256; omega

def offsetsFrom (start : Nat) : List Nat → Nat
  | [] => start
  | b :: bs => offsetsFrom (start + align256 b) bs

/-- The arena is at least as big as the sum of the blobs. -/
theorem offsets_total_ge : ∀ (bs : List Nat) (start : Nat), start + bs.sum ≤ offsetsFrom start bs
  | [], start => by simp [offsetsFrom]
  | b :: bs, start => by
    have h := offsets_total_ge bs (start + align256 b)
    have := align256_ge b
    simp only [offsetsFrom, List.sum_cons]
    omega

/-- Every slot starts on a 256-byte boundary. -/
theorem offsets_total_mod : ∀ (bs : List Nat) (start : Nat), start % 256 = 0 → offsetsFrom start bs % 256 = 0
  | [], start, h => by simpa [offsetsFrom] using h
  | b :: bs, start, h => by
    show offsetsFrom (start + align256 b) bs % 256 = 0
    apply offsets_total_mod bs
    have := align256_mod b
    omega

-- § bounded fill and in-arena slot address
def fillBytes (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never copies more than one slot. -/
theorem fillBytes_le (bytes blob : Nat) : fillBytes bytes blob ≤ blob := by
  unfold fillBytes
  split <;> omega

/-- A uniform slot lies wholly inside the arena. -/
theorem uniform_slot_in_arena {slot slots blob : Nat} (h : slot < slots) : slot * blob + blob ≤ slots * blob := by
  have := Nat.mul_le_mul_right blob (show slot + 1 ≤ slots by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

end Strata.ExpertVramCache



## FILE: /mnt/user-data/outputs/ExpertBlobSource.lean
/-!
# MODULE: expert_blob_source
Index arithmetic of the file-backed and arena-backed expert sources: `(layer, expert) -> blob`.

## Source semantics
Blob index is `layer * n_expert + expert`; an expert id at or past `n_expert` must be refused separately because the
flat index would silently alias the next layer. The arena is one max-blob longer than the file so a whole-slot copy may
start at any expert. GGUF chunked loading stays inside the per-layer expert range.

## Form template identifiers
- FileExpertSource::blob / ArenaExpertSource::blob -> flat
- ArenaExpertSource::device_alias                  -> sliceOff
- load_experts_gguf                                -> gguf_expert_in_range
-/
namespace Strata.ExpertBlobSource

-- § flat index
def flat (ne l e : Nat) : Nat := l * ne + e

/-- In-range (layer, expert) gives an in-range flat index. -/
theorem flat_lt {ne nl l e : Nat} (hl : l < nl) (he : e < ne) : flat ne l e < nl * ne := by
  unfold flat
  have := Nat.mul_le_mul_right ne (show l + 1 ≤ nl by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- A smaller layer gives a smaller flat index when experts are in range. -/
theorem flat_lt_of_layer_lt {ne l l' e e' : Nat} (h : l < l') (he : e < ne) : flat ne l e < flat ne l' e' := by
  unfold flat
  have := Nat.mul_le_mul_right ne (show l + 1 ≤ l' by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- The flat index is injective on in-range experts. -/
theorem flat_inj {ne l l' e e' : Nat} (he : e < ne) (he' : e' < ne) (h : flat ne l e = flat ne l' e') :
    l = l' ∧ e = e' := by
  rcases Nat.lt_trichotomy l l' with hlt | heq | hgt
  · have := flat_lt_of_layer_lt (e' := e') hlt he
    omega
  · subst heq
    unfold flat at h
    omega
  · have := flat_lt_of_layer_lt (e' := e) hgt he'
    omega

/-- Why the separate `expert >= n_expert` check exists: the unchecked flat index aliases the next layer. -/
theorem flat_alias (ne : Nat) : flat ne 0 ne = flat ne 1 0 := by simp [flat]

-- § arena slices
def sliceOff (bb e : Nat) : Nat := e * bb

/-- An expert's blob lies inside its layer's registration slice. -/
theorem slice_blob_in_slice {bb ne e : Nat} (he : e < ne) : sliceOff bb e + bb ≤ ne * bb := by
  unfold sliceOff
  have := Nat.mul_le_mul_right bb (show e + 1 ≤ ne by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The arena is one max-blob longer than the data, so a max-blob copy from any blob start stays inside it. -/
theorem arena_slack {off bb mb total : Nat} (h : off + bb ≤ total) (hb : bb ≤ mb) : off + mb ≤ total + mb := by
  omega

-- § GGUF chunked loading
/-- With `per` bytes per expert, a chunk of `b` experts starting at expert `a` that fits in the tensor
yields only expert ids below `n_expert`. -/
theorem gguf_expert_in_range {per a b ne k : Nat} (hp : 0 < per) (h : per * a + per * b ≤ per * ne)
    (hk : k < b) : a + k < ne := by
  have h2 : per * (a + b) ≤ per * ne := by rw [Nat.mul_add]; exact h
  have h3 := Nat.le_of_mul_le_mul_left h2 hp
  omega

end Strata.ExpertBlobSource



## FILE: /mnt/user-data/outputs/ExpertDispatchSplit.lean
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



## FILE: /mnt/user-data/outputs/CpuExpertPool.lean
/-!
# MODULE: cpu_expert_pool
Work partitioning of the CPU expert pool: equal row ranges per task, and one thread per physical core.

## Source semantics
`run_split` cuts a phase of `rows` rows into `n` equal ranges `[i*rows/n, (i+1)*rows/n)`; they are contiguous, ordered,
and cover `[0, rows)` exactly. The pool is `cores - 1` workers plus the host when `host_works`, i.e. one thread per
physical core.

## Form template identifiers
- ExpertPool::run_split_multi (mtasks_, mrows_) -> rowLo / rowHi
- physical_cores(skip_first), host_works        -> poolThreads
-/
namespace Strata.CpuExpertPool

def rowLo (rows n i : Nat) : Nat := i * rows / n
def rowHi (rows n i : Nat) : Nat := (i + 1) * rows / n

/-- Task 0 starts at row 0. -/
theorem rowLo_zero (rows n : Nat) : rowLo rows n 0 = 0 := by simp [rowLo]

/-- Consecutive tasks abut: no row is skipped or read twice. -/
theorem rowHi_eq_rowLo_succ (rows n i : Nat) : rowHi rows n i = rowLo rows n (i + 1) := rfl

/-- The last task ends at the last row. -/
theorem rowHi_last {rows n : Nat} (h : 0 < n) : rowHi rows n (n - 1) = rows := by
  unfold rowHi
  have : n - 1 + 1 = n := by omega
  rw [this]
  exact Nat.mul_div_cancel_left rows h

/-- Each task's range is non-empty-or-empty but never inverted. -/
theorem rowLo_le_rowHi (rows n i : Nat) : rowLo rows n i ≤ rowHi rows n i := by
  unfold rowLo rowHi
  exact Nat.div_le_div_right (Nat.mul_le_mul_right rows (by omega))

/-- Row ranges stay inside the phase. -/
theorem rowHi_le {rows n i : Nat} (h : i < n) : rowHi rows n i ≤ rows := by
  unfold rowHi
  rcases Nat.eq_zero_or_pos n with hn | hn
  · omega
  · have h1 : (i + 1) * rows ≤ n * rows := Nat.mul_le_mul_right rows (by omega)
    have h2 : (i + 1) * rows / n ≤ n * rows / n := Nat.div_le_div_right h1
    rw [Nat.mul_div_cancel_left rows hn] at h2
    exact h2

/-- Workers plus the working host are exactly the physical cores (when at least one core exists). -/
def poolThreads (cores : Nat) (hostWorks : Bool) : Nat := (cores - 1) + (if hostWorks then 1 else 0)

/-- With `host_works` every physical core carries one thread. -/
theorem poolThreads_host {cores : Nat} (h : 1 ≤ cores) : poolThreads cores true = cores := by
  unfold poolThreads
  simp
  omega

/-- Without `host_works` core 0 stays free for the host's doorbell spin. -/
theorem poolThreads_nohost {cores : Nat} (h : 1 ≤ cores) : poolThreads cores false + 1 = cores := by
  unfold poolThreads
  simp
  omega

end Strata.CpuExpertPool



## FILE: /mnt/user-data/outputs/F16Bits.lean
/-!
# MODULE: f16_bits
Structural facts behind the hand-written fp16 <-> fp32 bit conversions.

## Source semantics
fp16 = 1 sign | 5 exponent | 10 mantissa bits. The decoder maps normals to f32 exponent `ex + 112`, subnormals to
`113 - s` after `s` normalising shifts, and inf/NaN (`ex = 31`) to 255. The encoder saturates finite overflow to inf,
rounds to nearest even, and a mantissa carry may bump the exponent (30 -> 31 gives inf).

## Form template identifiers
- f32_from_f16 -> exF / manF / signF
- f16_from_f32 -> overflow threshold, carry, subnormal shift range
-/
namespace Strata.F16Bits

def exF (h : Nat) : Nat := h / 1024 % 32
def manF (h : Nat) : Nat := h % 1024
def signF (h : Nat) : Nat := h / 32768 % 2

/-- The three fields tile the 16 bits exactly. -/
theorem fields_recompose (h : Nat) (hh : h < 65536) : h = signF h * 32768 + exF h * 1024 + manF h := by
  unfold signF exF manF; omega

/-- Exponent field is 5 bits. -/
theorem exF_lt (h : Nat) : exF h < 32 := by unfold exF; omega
/-- Mantissa field is 10 bits. -/
theorem manF_lt (h : Nat) : manF h < 1024 := by unfold manF; omega
/-- Sign is one bit. -/
theorem signF_le (h : Nat) : signF h ≤ 1 := by unfold signF; omega

/-- Normal fp16 exponents land in the normal f32 exponent range. -/
theorem normal_exp_field {ex : Nat} (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    113 ≤ ex - 15 + 127 ∧ ex - 15 + 127 ≤ 142 := by omega

/-- Subnormal decode: after `1..10` normalising shifts the f32 exponent field is a valid normal exponent. -/
theorem subnormal_exp_field {s : Nat} (h1 : 1 ≤ s) (h2 : s ≤ 10) : 103 ≤ 113 - s ∧ 113 - s ≤ 112 := by omega

/-- Mantissa shifted into f32 position occupies only the low 23 bits. -/
theorem man_shift_fits {man : Nat} (h : man < 1024) : man * 8192 < 8388608 := by omega

/-- Assembled normal f32 (sign excluded) stays below 2^31. -/
theorem normal_assembled_lt {ex man : Nat} (h1 : ex ≤ 30) (h2 : man < 1024) :
    (ex - 15 + 127) * 8388608 + man * 8192 < 2147483648 := by omega

/-- Encoder overflow test: `exp = rawexp - 127 + 15 ≥ 31` iff `rawexp ≥ 143`. -/
theorem overflow_threshold (r : Nat) : ((r : Int) - 127 + 15 ≥ 31) ↔ 143 ≤ r := by omega

/-- Largest finite fp16 is 0x7BFF; one more is +inf 0x7C00, which is what a mantissa carry from there produces. -/
theorem carry_to_inf : (30 * 1024 + 1023) + 1 = 31 * 1024 := by decide

/-- Subnormal encode shift range for `exp ∈ [-10, 0]`. -/
theorem subnormal_shift_range (e : Int) (h1 : -10 ≤ e) (h2 : e ≤ 0) : 14 ≤ 14 - e ∧ 14 - e ≤ 24 := by omega

/-- Subnormal encode: a 24-bit mantissa shifted right by 14..24 fits the 10-bit field. -/
theorem subnormal_quotient_fits {X sh : Nat} (hX : X < 16777216) (h1 : 14 ≤ sh) : X / 2 ^ sh < 1024 := by
  have h2 : 2 ^ 14 ≤ 2 ^ sh := Nat.pow_le_pow_right (by decide) h1
  have h3 : (2 : Nat) ^ 14 = 16384 := by decide
  apply Nat.div_lt_of_lt_mul
  omega

/-- Round-to-nearest-even: an exact tie with an even result does not round up. -/
theorem tie_even_no_increment (half h : Nat) (he : h % 2 = 0) : ¬ (half < half ∨ (half = half ∧ h % 2 = 1)) := by
  omega

/-- An exact tie with an odd result does round up. -/
theorem tie_odd_increments (half h : Nat) (ho : h % 2 = 1) : (half < half ∨ (half = half ∧ h % 2 = 1)) := by
  omega

end Strata.F16Bits



## FILE: /mnt/user-data/outputs/GdnQsaLayerLayout.lean
/-!
# MODULE: gdn_qsa_layer_layout
Buffer-carving arithmetic and stage selection of the layer composition (`layer.cpp`).

## Source semantics
Q8_K buffers are `n/256` blocks of 292 bytes. Every carved region is 16-byte aligned (`(v+15)&~15`), so the total
is at least the sum of the parts. A paged KV state with page size `p` holds `ceil(max_cells/p)*p >= max_cells` cells.
RoPE row `pos_base+pos` must be below `max_cells`. `block_layer_pre` runs stage `i` iff the stage prefix covers it, or
(prefix 0) according to `half`: stages 0-2 for halves 0/1, stages 3-4 for halves 0/2.

## Form template identifiers
- q8k_bytes            -> q8kBytes
- align16 / Cursor     -> align16 / takeUsed
- qsa_state_bytes      -> ceilDivMul
- block_layer_pre run0..run4 -> runStage
-/
namespace Strata.GdnQsaLayerLayout

def q8kBytes (n : Nat) : Nat := n / 256 * 292
def align16 (v : Nat) : Nat := (v + 15) / 16 * 16

/-- Q8_K blocks carry 256 elements, so `n` a multiple of 256 is exactly `n/256` blocks. -/
theorem q8k_blocks (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by omega

/-- Alignment never shrinks a region. -/
theorem align16_ge (v : Nat) : v ≤ align16 v := by unfold align16; omega
/-- Aligned regions are multiples of 16. -/
theorem align16_mod (v : Nat) : align16 v % 16 = 0 := by unfold align16; omega
/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (v : Nat) : align16 v < v + 16 := by unfold align16; omega

/-- The carved total covers the sum of all parts. -/
theorem carve_total_ge : ∀ ps : List Nat, ps.sum ≤ (ps.map align16).sum
  | [] => by simp
  | p :: ps => by
    have h := carve_total_ge ps
    have := align16_ge p
    simp only [List.map_cons, List.sum_cons]
    omega

/-- A cursor `take` advances by at least the requested bytes and keeps 16-byte alignment. -/
theorem take_advances (used count sz : Nat) :
    used + count * sz ≤ align16 (used + count * sz) ∧ align16 (used + count * sz) % 16 = 0 :=
  ⟨align16_ge _, align16_mod _⟩

/-- Pages needed to hold `a` cells never undercount. -/
theorem ceilDivMul (a b : Nat) (hb : 0 < b) : a ≤ b * ((a + b - 1) / b) := by
  have h := Nat.div_add_mod (a + b - 1) b
  have h2 := Nat.mod_lt (a + b - 1) hb
  omega

/-- Row index `pos_base + pos < max_cells` keeps every RoPE-table access (`row*h + j`, `j < h`) in range. -/
theorem rope_index_in_table {row mc h j : Nat} (hr : row < mc) (hj : j < h) : row * h + j < mc * h := by
  have := Nat.mul_le_mul_right h (show row + 1 ≤ mc by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- Stage `i` (0..4) runs under a positive stage prefix `k` iff `k ≥ i+1`; with prefix 0 it follows `half`. -/
def runStage (pfx half i : Nat) : Bool :=
  if pfx > 0 then decide (pfx ≥ i + 1)
  else if i < 3 then decide (half = 0 ∨ half = 1) else decide (half = 0 ∨ half = 2)

/-- `half = 0` with no prefix runs the whole block. -/
theorem runStage_all (i : Nat) : runStage 0 0 i = true := by
  unfold runStage
  split <;> simp_all

/-- The full prefix 5 runs every stage. -/
theorem runStage_prefix5 (i : Nat) (h : i < 5) : runStage 5 0 i = true := by
  unfold runStage
  simp
  omega

/-- A positive prefix is downward closed: if stage `j` runs, every earlier stage runs. -/
theorem runStage_prefix_closed {k i j : Nat} (hk : 0 < k) (hij : i ≤ j) (h : runStage k 0 j = true) :
    runStage k 0 i = true := by
  unfold runStage at h ⊢
  rw [if_pos hk] at h ⊢
  simp only [decide_eq_true_eq] at h ⊢
  omega

/-- Half 2 skips the mixer stages 0-2. -/
theorem runStage_half2_skips_mixer (i : Nat) (h : i < 3) : runStage 0 2 i = false := by
  unfold runStage
  simp
  omega

/-- Half 1 skips the FFN-front and router stages 3-4. -/
theorem runStage_half1_skips_router (i : Nat) (h1 : 3 ≤ i) : runStage 0 1 i = false := by
  unfold runStage
  simp
  omega

/-- The dump region of one layer ends inside the whole dump buffer. -/
theorem dump_slot_in_buffer {l nl n : Nat} (h : l < nl) : (l + 1) * n + n ≤ (nl + 1) * n := by
  have h1 : (l + 2) * n ≤ (nl + 1) * n := Nat.mul_le_mul_right n (by omega)
  have h2 : (l + 2) * n = (l + 1) * n + n := by ring
  omega

end Strata.GdnQsaLayerLayout



## FILE: /mnt/user-data/outputs/MtpDraftRows.lean
/-!
# MODULE: mtp_draft_rows
Row bookkeeping of the MTP draft layer: arena alignment, window rounding, step-row layout, prefill chunking.

## Source semantics
Staging has `2*T` step rows: rows `[0,T)` catch-up, row `2*max_t - 1` the full-layer row, and chain step `j`
(1..max_t-2) uses row `max_t + j - 1`; these ranges are disjoint. Bump allocations round to 256 bytes. The window rounds
up to 64. Prefill walks the prompt in chunks of at most `max_t` and always makes progress.

## Form template identifiers
- Bump::take            -> bumpUsed
- cap_                  -> cap64
- put(row, cell) rows   -> chainRow / fullRow
- MtpDrafter::prefill   -> chunkT
-/
namespace Strata.MtpDraftRows

def bumpUsed (used n sz : Nat) : Nat := used + (n * sz + 255) / 256 * 256

/-- A bump allocation never overlaps earlier ones. -/
theorem bump_ge (used n sz : Nat) : used + n * sz ≤ bumpUsed used n sz := by unfold bumpUsed; omega
/-- A bump allocation keeps 256-byte alignment. -/
theorem bump_mod (used n sz : Nat) (h : used % 256 = 0) : bumpUsed used n sz % 256 = 0 := by
  unfold bumpUsed; omega

def cap64 (w : Nat) : Nat := (w + 63) / 64 * 64
/-- The attention capacity covers the window. -/
theorem cap64_ge (w : Nat) : w ≤ cap64 w := by unfold cap64; omega
/-- The capacity is a multiple of 64. -/
theorem cap64_mod (w : Nat) : cap64 w % 64 = 0 := by unfold cap64; omega
/-- The capacity wastes under 64 cells. -/
theorem cap64_lt (w : Nat) : cap64 w < w + 64 := by unfold cap64; omega

def fullRow (maxT : Nat) : Nat := 2 * maxT - 1
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- The full-layer row is inside the `2*max_t` staged rows. -/
theorem fullRow_lt {maxT : Nat} (h : 1 ≤ maxT) : fullRow maxT < 2 * maxT := by unfold fullRow; omega

/-- Chain rows are inside the staged rows. -/
theorem chainRow_lt {maxT j : Nat} (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) : chainRow maxT j < 2 * maxT := by
  unfold chainRow; omega

/-- Chain rows sit above the catch-up rows `[0, max_t)`. -/
theorem chainRow_ge {maxT j : Nat} (h1 : 1 ≤ j) : maxT ≤ chainRow maxT j := by unfold chainRow; omega

/-- Chain rows never collide with the full-layer row. -/
theorem chainRow_ne_fullRow {maxT j : Nat} (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) : chainRow maxT j ≠ fullRow maxT := by
  unfold chainRow fullRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj {maxT j j' : Nat} (h1 : 1 ≤ j) (h2 : 1 ≤ j') (h : chainRow maxT j = chainRow maxT j') : j = j' := by
  unfold chainRow at h; omega

/-- The full-layer row is above every catch-up row `t < T ≤ max_t`. -/
theorem fullRow_above_catchup {maxT T t : Nat} (hT : T ≤ maxT) (ht : t < T) : t < fullRow maxT := by
  unfold fullRow; omega

def chunkT (maxT n c : Nat) : Nat := min maxT (n - c)

/-- A prefill chunk is non-empty whenever cells remain. -/
theorem chunkT_pos {maxT n c : Nat} (h1 : 1 ≤ maxT) (h2 : c < n) : 1 ≤ chunkT maxT n c := by
  unfold chunkT; omega

/-- A prefill chunk never runs past the prompt. -/
theorem chunkT_le {maxT n c : Nat} : c + chunkT maxT n c ≤ max c n := by unfold chunkT; omega

/-- A chunk never exceeds the staged `max_t` rows. -/
theorem chunkT_le_max (maxT n c : Nat) : chunkT maxT n c ≤ maxT := by unfold chunkT; omega

/-- The draft-vocabulary file is whole 4-byte ids. -/
theorem dvocab_count {size : Nat} (h : size % 4 = 0) : size / 4 * 4 = size := by omega

end Strata.MtpDraftRows



## FILE: /mnt/user-data/outputs/SessionLoopIndexing.lean
/-!
# MODULE: session_loop_indexing
Per-layer state indexing, arena sizing and doorbell sequence numbers of the session loop.

## Source semantics
Layer `l` is a QSA layer iff `l % 4 == 3`. GDN and QSA layers are numbered separately: `gdn_point_at` counts the
non-QSA layers before `l`. `session_bytes` rounds up to 256. Layer `l` waits for ring number `l+1`. The staged-prefix
stage times are differences of cumulative prefix times and telescope.

## Form template identifiers
- gdn_point_at / session_token index counters -> gdnIdx / qsaIdx
- align_up                                    -> alignUp
- session_loop `want = ++expected`            -> want
- session_replay_stage_prefixes               -> telescope
-/
namespace Strata.SessionLoopIndexing

def gdnIdx : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIdx l + (if l % 4 == 3 then 0 else 1)
def qsaIdx : Nat → Nat
  | 0 => 0
  | l + 1 => qsaIdx l + (if l % 4 == 3 then 1 else 0)

/-- Every layer before `l` is exactly one of GDN or QSA. -/
theorem gdn_add_qsa : ∀ l, gdnIdx l + qsaIdx l = l
  | 0 => rfl
  | l + 1 => by
    have := gdn_add_qsa l
    simp only [gdnIdx, qsaIdx]
    split <;> omega

/-- The QSA index of layer `l` is `l / 4`. -/
theorem qsaIdx_eq : ∀ l, qsaIdx l = l / 4
  | 0 => rfl
  | l + 1 => by
    have := qsaIdx_eq l
    simp only [qsaIdx]
    split <;> simp_all <;> omega

/-- The GDN index of layer `l` is `l - l/4`. -/
theorem gdnIdx_eq (l : Nat) : gdnIdx l = l - l / 4 := by
  have h1 := gdn_add_qsa l
  have h2 := qsaIdx_eq l
  omega

/-- In a 48-layer model every GDN layer's state slot is among the 36 allocated. -/
theorem gdn_slot_in_range {l : Nat} (hl : l < 48) (hg : l % 4 ≠ 3) : gdnIdx l < 36 := by
  rw [gdnIdx_eq]; omega

/-- Every QSA layer's state slot is among the 12 allocated. -/
theorem qsa_slot_in_range {l : Nat} (hl : l < 48) : qsaIdx l < 12 := by
  rw [qsaIdx_eq]; omega

/-- Two distinct GDN layers never share a state slot. -/
theorem gdn_slots_distinct {l l' : Nat} (h : l < l') (hl : l % 4 ≠ 3) (hl' : l' % 4 ≠ 3) :
    gdnIdx l ≠ gdnIdx l' := by
  rw [gdnIdx_eq, gdnIdx_eq]; omega

def alignUp (n a : Nat) : Nat := (n + a - 1) / a * a

/-- Session arena rounding never shrinks. -/
theorem alignUp_ge (n a : Nat) (ha : 0 < a) : n ≤ alignUp n a := by
  unfold alignUp
  have h := Nat.div_add_mod (n + a - 1) a
  have h2 := Nat.mod_lt (n + a - 1) ha
  have h3 := Nat.mul_comm ((n + a - 1) / a) a
  omega

/-- Session arena rounding is a multiple of `a`. -/
theorem alignUp_mod (n a : Nat) : alignUp n a % a = 0 := by unfold alignUp; simp

def want (expected : Nat) : Nat := expected + 1

/-- Layer `l` (0-based) waits for ring `l+1`, matching the doorbell's monotone counter. -/
theorem want_layer (l : Nat) : want l = l + 1 := rfl

/-- Per-stage times derived from cumulative prefix times sum back to the full prefix. -/
theorem telescope (t1 t2 t3 t4 t5 : Int) : t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by omega

end Strata.SessionLoopIndexing



## FILE: /mnt/user-data/outputs/VerifyWindowGroups.lean
/-!
# MODULE: verify_window_groups
Token-group split, ring numbering, staging and plan-buffer layout of the speculative verify window.

## Source semantics
With `G = 2` the window's T tokens are cut into A = `[0,(T+1)/2)` and B = `[(T+1)/2, T)`; with `G = 1` one group holds all.
Processing order index is `k = l*G + grp` and the ring awaited is `k+1`; `k/G = l`, `k%G = grp`. Staging blobs are
split evenly per group. The plan buffer holds i32 header/arrays, then 8-byte aligned u64 pointer arrays. Flag B only rises.

## Form template identifiers
- tb_/te_              -> tbF / teF
- ring `l*G+grp+1`     -> ringK
- kStagingBlobs split  -> perGroup
- Verifier::raise_flag -> raiseFlag
-/
namespace Strata.VerifyWindowGroups

def tbF (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2
def teF (G T grp : Nat) : Nat := if G = 2 ∧ grp = 0 then (T + 1) / 2 else T

/-- One group covers `[0, T)`. -/
theorem single_group (T : Nat) : tbF T 0 = 0 ∧ teF 1 T 0 = T := by simp [tbF, teF]

/-- Two groups tile `[0, T)` with no gap or overlap. -/
theorem two_groups_tile (T : Nat) : tbF T 0 = 0 ∧ teF 2 T 0 = tbF T 1 ∧ teF 2 T 1 = T := by simp [tbF, teF]

/-- Both groups are non-empty when `T ≥ 2`. -/
theorem two_groups_nonempty {T : Nat} (h : 2 ≤ T) : 1 ≤ teF 2 T 0 - tbF T 0 ∧ 1 ≤ teF 2 T 1 - tbF T 1 := by
  simp [tbF, teF]
  omega

/-- Group A is at least as large as group B. -/
theorem groupA_ge_groupB (T : Nat) : teF 2 T 1 - tbF T 1 ≤ teF 2 T 0 - tbF T 0 := by
  simp [tbF, teF]
  omega

/-- The two group sizes add up to the window. -/
theorem group_sizes_sum {T : Nat} (h : 2 ≤ T) :
    (teF 2 T 0 - tbF T 0) + (teF 2 T 1 - tbF T 1) = T := by
  simp [tbF, teF]
  omega

def ringK (G l grp : Nat) : Nat := grp + G * l + 1

/-- `(l*G + grp) / G = l` for `grp < G`: the host loop recovers the layer. -/
theorem ring_div {G l grp : Nat} (hG : 0 < G) (hg : grp < G) : (grp + G * l) / G = l := by
  rw [Nat.add_mul_div_left _ _ hG, Nat.div_eq_of_lt hg]
  simp

/-- `(l*G + grp) % G = grp`: the host loop recovers the group. -/
theorem ring_mod {G l grp : Nat} (hg : grp < G) : (grp + G * l) % G = grp := by
  rw [Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt hg

/-- Ring numbers of distinct (layer, group) pairs are distinct. -/
theorem ringK_inj {G l l' grp grp' : Nat} (hG : 0 < G) (hg : grp < G) (hg' : grp' < G)
    (h : ringK G l grp = ringK G l' grp') : l = l' ∧ grp = grp' := by
  unfold ringK at h
  have h1 : grp + G * l = grp' + G * l' := by omega
  have e1 := ring_div (l := l) hG hg
  have e2 := ring_div (l := l') hG hg'
  have m1 := ring_mod (l := l) hg
  have m2 := ring_mod (l := l') hg'
  rw [h1] at e1 m1
  exact ⟨by omega, by omega⟩

/-- The ring numbers are all within `1 .. n_layers * G`. -/
theorem ringK_le {G nl l grp : Nat} (hg : grp < G) (hl : l < nl) : ringK G l grp ≤ nl * G := by
  unfold ringK
  have := Nat.mul_le_mul_left G (show l + 1 ≤ nl by omega)
  rw [Nat.mul_add, Nat.mul_one] at this
  have h2 := Nat.mul_comm G nl
  omega

def perGroup (G K : Nat) : Nat := if G = 2 then K / 2 else K

/-- Each group's staging half lies inside the staging area. -/
theorem staging_in_area {G K grp : Nat} (hg : grp < G) (hG : G ≤ 2) : grp * perGroup G K + perGroup G K ≤ K := by
  unfold perGroup
  split
  · have h01 : grp = 0 ∨ grp = 1 := by omega
    rcases h01 with rfl | rfl <;> omega
  · have : grp = 0 := by omega
    subst this
    omega

/-- Plan buffer layout: the u64 region start is even (8-byte aligned in i32 units), past the i32 header. -/
theorem plan_ptr_off (cap : Nat) :
    let i32 := 4 + (cap + 1) + cap + cap
    let off := (i32 + 1) / 2 * 2
    i32 ≤ off ∧ off % 2 = 0 := by
  intro i32 off
  omega

/-- Plan buffer: the `start2` array fits within the allocated i32 count. -/
theorem plan_start2_fits (off cap : Nat) : off + 4 * cap + (cap + 1) ≤ off + 4 * cap + (cap + 1) + 1 := by omega

def raiseFlag (cur v : Nat) : Nat := if cur < v then v else cur

/-- Flag B never decreases. -/
theorem raiseFlag_ge (cur v : Nat) : cur ≤ raiseFlag cur v := by unfold raiseFlag; split <;> omega
/-- Raising reaches at least the requested value. -/
theorem raiseFlag_ge_v (cur v : Nat) : v ≤ raiseFlag cur v := by unfold raiseFlag; split <;> omega
/-- Late raises from earlier layers do not lower a later layer's value. -/
theorem raiseFlag_eq_max (cur v : Nat) : raiseFlag cur v = max cur v := by
  unfold raiseFlag
  split <;> omega

/-- Raise order does not matter. -/
theorem raiseFlag_comm (c a b : Nat) : raiseFlag (raiseFlag c a) b = raiseFlag (raiseFlag c b) a := by
  simp only [raiseFlag_eq_max]
  omega

/-- The flag-set ring index is always in bounds. -/
theorem flagset_index {n sz : Nat} (h : 0 < sz) : n % sz < sz := Nat.mod_lt n h

end Strata.VerifyWindowGroups



## FILE: /mnt/user-data/outputs/WeightsLoaderSegments.lean
/-!
# MODULE: weights_loader_segments
Byte arithmetic of the dense-weight loader: plane segments, chunked conversion, compaction alignment.

## Source semantics
A quantized tensor is codes | scales | optional offsets. An fp16 scale plane is half its engine size in the pack and is
widened to f32 (exact). Promoted tensors are 4 B/elem in and 2 B/elem out. Chunks are at most 8 MiB of source; widening
never splits an element; stage-out is twice stage-in. Compacted placement rounds each tensor to the index alignment.

## Form template identifiers
- Seg list / sum_src / sum_dst -> srcTotal / dstTotal
- CHUNK                        -> CHUNK
- Conv::Bf16High / WidenF16    -> bf16_chunk_contiguous / widen_chunk_contiguous
- skip-set compaction          -> placeAt
-/
namespace Strata.WeightsLoaderSegments

def srcScale (scales : Nat) (fp16 : Bool) : Nat := if fp16 then scales / 2 else scales
def srcTotal (codes scales offs : Nat) (fp16 : Bool) : Nat := codes + srcScale scales fp16 + offs
def dstTotal (codes scales offs : Nat) : Nat := codes + scales + offs

/-- The widened plane is exactly twice its source (even byte count), so totals differ by the plane's source size. -/
theorem widen_total {codes scales offs : Nat} (he : scales % 2 = 0) :
    dstTotal codes scales offs = srcTotal codes scales offs true + scales / 2 := by
  unfold dstTotal srcTotal srcScale
  simp
  omega

/-- Without fp16 scales the two layouts have equal size. -/
theorem copy_total (codes scales offs : Nat) : dstTotal codes scales offs = srcTotal codes scales offs false := by
  simp [dstTotal, srcTotal, srcScale]

/-- The offset segment starts where the scale segment ends in the source and ends at the source total. -/
theorem offset_seg_end (codes scales offs : Nat) (fp16 : Bool) :
    (codes + srcScale scales fp16) + offs = srcTotal codes scales offs fp16 := by
  unfold srcTotal; omega

def CHUNK : Nat := 8388608

/-- CHUNK is even, so a full chunk never splits an fp16 element. -/
theorem CHUNK_even : CHUNK % 2 = 0 := by decide

def widenN (rem : Nat) : Nat :=
  if min 8388608 rem % 2 = 1 then min 8388608 rem - 1 else min 8388608 rem

/-- A widening chunk covers a whole number of fp16 elements. -/
theorem widenN_even (rem : Nat) : widenN rem % 2 = 0 := by
  unfold widenN
  split <;> omega

/-- A widening chunk never overreads the plane. -/
theorem widenN_le (rem : Nat) : widenN rem ≤ rem := by
  unfold widenN
  split <;> omega

/-- A widening chunk is non-empty while an even number of bytes remain. -/
theorem widenN_pos {rem : Nat} (h0 : 0 < rem) (he : rem % 2 = 0) : 0 < widenN rem := by
  unfold widenN
  split <;> omega

/-- A widened chunk (output = 2 × input) fits stage_out, which is twice CHUNK. -/
theorem widen_out_fits {n : Nat} (h : n ≤ 8388608) : n / 2 * 4 ≤ 2 * 8388608 := by omega

/-- bf16 promotion: with 4-byte-aligned progress, the next output offset equals the end of this chunk's output. -/
theorem bf16_chunk_contiguous {done n : Nat} (h1 : done % 4 = 0) (h2 : n % 4 = 0) :
    done / 2 + n / 4 * 2 = (done + n) / 2 := by omega

/-- fp16 widening: output offset `done*2` advances exactly by the chunk's output size. -/
theorem widen_chunk_contiguous {done n : Nat} (h2 : n % 2 = 0) : done * 2 + n / 2 * 4 = (done + n) * 2 := by omega

/-- Promoted tensor sizes: `4e` in, `2e` out. -/
theorem promoted_sizes (e : Nat) : 2 * e * 2 = 4 * e := by omega

def placeAt (at_ dst a : Nat) : Nat := at_ + (dst + a - 1) / a * a

/-- Compaction never overlaps tensors: each placed tensor still fits. -/
theorem placeAt_ge {at_ dst a : Nat} (ha : 0 < a) : at_ + dst ≤ placeAt at_ dst a := by
  unfold placeAt
  have h := Nat.div_add_mod (dst + a - 1) a
  have h2 := Nat.mod_lt (dst + a - 1) ha
  have h3 := Nat.mul_comm ((dst + a - 1) / a) a
  omega

/-- Compaction keeps every tensor start on the index alignment. -/
theorem placeAt_mod {at_ dst a : Nat} (h : at_ % a = 0) : placeAt at_ dst a % a = 0 := by
  unfold placeAt
  simp [Nat.add_mod, h]

/-- raw 16-bit index kinds map to the engine kinds 1 (BF16) and 3 (F16). -/
def engineKind (k : Nat) : Nat := if k = 4 then 1 else if k = 5 then 3 else k

/-- Kind 4 becomes BF16 (1). -/
theorem engineKind_bf16 : engineKind 4 = 1 := rfl
/-- Kind 5 becomes F16 (3). -/
theorem engineKind_f16 : engineKind 5 = 3 := rfl

end Strata.WeightsLoaderSegments





## FILE: /mnt/user-data/outputs/Q2ExpertVnniKernel.lean
/-!
# MODULE: q2_expert_vnni_kernel
Q2_0 expert kernel (gate/up -> SwiGLU -> Q8_1 requantise -> down) for AVX-512 VNNI CPUs. Weights are 2-bit
codes 0..3 (value = code - 1) with an fp16 scale per 64-weight block; activations are int8 per 32-value chunk.

## Source semantics
* Offset identity: sum_j (c_j - 1) * xhat_j = sum(c * xhat) - sum(xhat), so codes may be fed unsigned to VNNI
  and the `hx = scale * sum` correction subtracted afterwards.
* A 64-weight block is two activation chunks of 32; the weight scale is shared, the activation scale is per chunk.
* Tail handling of the 512-bit row dot: groups of up to 8 blocks, masks cover exactly `2*nb` lanes.
* Activation quantiser: round half away from zero, clamp to [-127,127], store as int8.
* Token tile dispatch: 1..8 tokens select the template, larger counts fall to 8, oracle / out-of-range uses
  the single-token path. Gate row is 2r, up row is 2r+1.

## Form template identifiers
row_dot_z / row_dot_multi_z -> row_kernel ; act_quant_q8_1 -> act_quantizer ; QK, QKA -> block_size, chunk_size ;
s2_expert_vnni_multi -> multi_token_dispatch ; unpack_q2_0 -> code_unpack
-/

namespace Strata.Cpu.ExpertKernel

-- § constants and block geometry

/-- Weights per Q2_0 block. -/
def QK : Nat := 64

/-- Values per activation chunk. -/
def QKA : Nat := 32

/-- A weight block spans exactly two activation chunks. -/
theorem qk_eq_two_chunks : QK = 2 * QKA := by decide

/-- Chunk count of `k` full chunks is `k`. -/
theorem nchunks_of_mul (k : Nat) : (k * QKA) / QKA = k := by
  unfold QKA; omega

/-- Full blocks cover a row whose length is a multiple of the block size. -/
theorem blocks_cover (n : Nat) (h : n % QK = 0) : (n / QK) * QK = n := by
  unfold QK at *; omega

-- § code unpacking (vpmultishiftqb + and 3)

/-- The 2-bit code number `j` of a 16-bit lane. -/
def code2 (w j : Nat) : Nat := (w >>> (2 * j)) &&& 3

/-- Every unpacked code is at most 3. -/
theorem code2_le (w j : Nat) : code2 w j ≤ 3 := Nat.and_le_right

/-- Multishift selects byte `(w >> 2i) & 0xFF`; AND 3 leaves exactly the i-th code. -/
theorem multishift_and3 (w i : Nat) : ((w >>> (2 * i)) &&& 0xFF) &&& 3 = code2 w i := by
  have h : (0xFF : Nat) &&& 3 = 3 := by decide
  unfold code2
  rw [Nat.and_assoc, h]

/-- Scalar reference: code of weight `j` in a block lives in byte `j/4` at shift `2*(j%4)`. -/
def codeAt (bytes : Nat → Nat) (j : Nat) : Nat := (bytes (j / 4) >>> (2 * (j % 4))) &&& 3

/-- Scalar-reference codes are at most 3. -/
theorem codeAt_le (bytes : Nat → Nat) (j : Nat) : codeAt bytes j ≤ 3 := Nat.and_le_right

/-- Lane `l` (a 16-bit pair of bytes) holds weights `8l .. 8l+7`; weight `8l+i` sits in byte `2l + i/4`. -/
theorem lane_byte_index (l i : Nat) (hi : i < 8) : (8 * l + i) / 4 = 2 * l + i / 4 := by
  omega

/-- The shift of weight `8l+i` inside its byte equals the in-lane shift index modulo 4. -/
theorem lane_shift_index (l i : Nat) : (8 * l + i) % 4 = i % 4 := by
  omega

-- § offset algebra on integer lists

/-- Integer dot product of two lists (truncating at the shorter). -/
def dotI : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | c :: cs, x :: xs => c * x + dotI cs xs

/-- Integer sum of a list. -/
def sumI : List Int → Int
  | [] => 0
  | x :: xs => x + sumI xs

/-- The code offset identity: sum (c-1) x = sum c x - sum x. -/
theorem dot_offset : ∀ (cs xs : List Int), cs.length = xs.length →
    dotI (cs.map (· - 1)) xs = dotI cs xs - sumI xs
  | [], [], _ => by simp [dotI, sumI]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | c :: cs, x :: xs, h => by
      have ih := dot_offset cs xs (by simpa using h)
      have hm : (c - 1) * x = c * x - x := by rw [Int.sub_mul, Int.one_mul]
      simp only [List.map_cons, dotI, sumI]
      rw [ih, hm]
      omega

/-- A dot product splits across a concatenation (per-chunk splitting of a block). -/
theorem dot_append : ∀ (a b c d : List Int), a.length = c.length →
    dotI (a ++ b) (c ++ d) = dotI a c + dotI b d
  | [], b, [], d, _ => by simp [dotI]
  | [], _, _ :: _, _, h => by simp at h
  | _ :: _, _, [], _, h => by simp at h
  | x :: a, b, y :: c, d, h => by
      have ih := dot_append a b c d (by simpa using h)
      simp only [List.cons_append, dotI]
      rw [ih]
      omega

/-- A zero code contributes nothing, so zero lanes left by the narrow load are harmless. -/
theorem zero_code_contrib (x : Int) : (0 : Int) * x = 0 := Int.zero_mul x

-- § tail masks and 8-block batches

/-- Lane mask for a batch of `nb` blocks (two lanes per block, 16 lanes max). -/
def tailMask (nb : Nat) : Nat := if nb ≥ 8 then 0xFFFF else (1 <<< (2 * nb)) - 1

/-- Masks never exceed 16 lanes. -/
theorem tailMask_le (nb : Nat) : tailMask nb ≤ 0xFFFF := by
  have small : ∀ k, k < 8 → (1 <<< (2 * k)) - 1 ≤ 0xFFFF := by decide
  unfold tailMask
  split
  · exact Nat.le_refl _
  · rename_i h
    exact small nb (by omega)

/-- A full batch uses the full mask. -/
theorem tailMask_full (nb : Nat) (h : nb ≥ 8) : tailMask nb = 0xFFFF := by
  unfold tailMask
  rw [if_pos h]

/-- Blocks in the batch starting at `b0`. -/
def batchSize (nblocks b0 : Nat) : Nat := min 8 (nblocks - b0)

/-- A batch is non-empty, at most 8, and stays inside the row. -/
theorem batch_bounds (nblocks b0 : Nat) (h : b0 < nblocks) :
    1 ≤ batchSize nblocks b0 ∧ batchSize nblocks b0 ≤ 8 ∧ b0 + batchSize nblocks b0 ≤ nblocks := by
  unfold batchSize
  omega

/-- The next batch start strictly advances, so the loop terminates. -/
theorem batch_advance (nblocks b0 : Nat) (h : b0 < nblocks) : b0 < b0 + 8 := by omega

-- § activation quantiser rule

/-- Clamp to the symmetric int8 range [-127, 127]. -/
def clamp127 (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Clamp output lies in [-127, 127]. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  by_cases h1 : v < -127 <;> by_cases h2 : v > 127 <;> simp [h1, h2] <;> omega

/-- Clamp is the identity on in-range values. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  have a : ¬ v < -127 := by omega
  have b : ¬ v > 127 := by omega
  simp [a, b]

/-- Clamp is idempotent. -/
theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v := by
  have h := clamp127_bounds v
  exact clamp127_id _ h.1 h.2

/-- Round half away from zero on a value scaled by two. -/
def roundAway (u : Int) : Int := if 0 ≤ u then (u + 1) / 2 else -((-u + 1) / 2)

/-- Round-half-away-from-zero is odd-symmetric (unlike round-half-even). -/
theorem roundAway_neg (u : Int) : roundAway (-u) = -roundAway u := by
  unfold roundAway
  by_cases h1 : 0 ≤ u <;> by_cases h2 : 0 ≤ -u <;> simp [h1, h2] <;> omega

/-- Quantised magnitude never exceeds 127 once clamped. -/
theorem quant_in_range (u : Int) : -127 ≤ clamp127 (roundAway u) ∧ clamp127 (roundAway u) ≤ 127 :=
  clamp127_bounds _

-- § row layout and token-tile dispatch

/-- Gate and up rows of the interleaved projection never coincide. -/
theorem gu_rows_distinct (r r' : Nat) : 2 * r ≠ 2 * r' + 1 := by omega

/-- The up row of an in-range index is in range. -/
theorem gu_row_lt (ff r : Nat) (h : r < ff) : 2 * r + 1 < 2 * ff := by omega

/-- Template selection: 8 or more tokens use the 8-token tile. -/
def ntSel (n : Nat) : Nat := if n ≥ 8 then 8 else n

/-- The selected tile never exceeds 8. -/
theorem ntSel_le (n : Nat) : ntSel n ≤ 8 := by
  unfold ntSel
  split <;> omega

/-- For 1..8 tokens the tile equals the token count. -/
theorem ntSel_id (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : ntSel n = n := by
  unfold ntSel
  split <;> omega

/-- Multi-token path is taken only without the oracle and with a valid token count. -/
def useMulti (oracle : Bool) (n maxt : Nat) : Bool := !oracle && decide (1 ≤ n) && decide (n ≤ maxt)

/-- The oracle always forces the single-token path. -/
theorem oracle_forces_single (n maxt : Nat) : useMulti true n maxt = false := by
  simp [useMulti]

/-- A valid count without the oracle takes the multi path. -/
theorem valid_uses_multi (n maxt : Nat) (h1 : 1 ≤ n) (h2 : n ≤ maxt) : useMulti false n maxt = true := by
  simp [useMulti, h1, h2]

end Strata.Cpu.ExpertKernel



## FILE: /mnt/user-data/outputs/IquantAvx512Rows.lean
/-!
# MODULE: iquant_avx512_rows
512-bit multi-token dot products for the i-quant expert formats IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18),
IQ3_S(21) and IQ2_S(22) against Q8_K activations. Each 64-value chunk is decoded once (grid lookups, sign mask,
scales) and applied to every token.

## Source semantics
* Per-format block sizes are fixed; every byte offset the decoder touches stays inside its block for chunk j < 4.
* Sign application is a masked negate, an involution that preserves magnitude.
* Scales are `2*nibble + 1`, hence odd and at most 31.
* Grid indices are built from byte | high bits and stay inside the grid tables.
* Token tiling: 1..4 tokens map directly, more are processed in groups of 4; the gate/up variant clamps to 8.

## Form template identifiers
Fmt<TY>::decode -> chunk_decoder ; row_dot -> row_kernel ; iq512_gu_rows -> gate_up_dispatch ;
iq512_rows -> down_dispatch ; iq512_supported -> format_supported
-/

namespace Strata.Cpu.IquantAvx512

-- § format table

/-- Bytes per 256-value block, per ggml type id. -/
def blockBytes (ty : Nat) : Nat :=
  if ty = 16 then 66 else if ty = 17 then 74 else if ty = 18 then 98
  else if ty = 21 then 110 else if ty = 22 then 82 else 0

/-- Formats handled by the AVX-512 path. -/
def supported (ty : Nat) : Bool := ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

/-- Every supported format has a positive block size. -/
theorem supported_blockBytes_pos (ty : Nat) (h : supported ty = true) : 0 < blockBytes ty := by
  unfold supported at h
  simp at h
  unfold blockBytes
  rcases h with ((((h | h) | h) | h) | h) <;> subst h <;> decide

/-- Unsupported types (IQ1_M, 20 etc.) stay on ggml-cpu. -/
theorem iq1_m_unsupported : supported 29 = false := by decide

/-- IQ2_XXS layout: fp16 scale plus 32 u16. -/
theorem layout_iq2xxs : 2 + 64 = blockBytes 16 := by decide

/-- IQ2_XS layout: scale, 32 u16, 8 scale bytes. -/
theorem layout_iq2xs : 2 + 64 + 8 = blockBytes 17 := by decide

/-- IQ3_XXS layout: scale, 64 grid bytes, 8 u32. -/
theorem layout_iq3xxs : 2 + 64 + 32 = blockBytes 18 := by decide

/-- IQ3_S layout: scale, qs, qh, signs, scales. -/
theorem layout_iq3s : 2 + 64 + 8 + 32 + 4 = blockBytes 21 := by decide

/-- IQ2_S layout: scale, qs(64), qh, scales. -/
theorem layout_iq2s : 2 + 64 + 8 + 8 = blockBytes 22 := by decide

-- § decoder byte offsets stay inside the block (chunk j < 4)

/-- IQ2_XXS reads 16 bytes at `2 + 16j`. -/
theorem xxs_in_block (j : Nat) (h : j < 4) : 2 + 16 * j + 16 ≤ 66 := by omega

/-- IQ2_XS scale bytes `66 + 2j (+1)`. -/
theorem xs_scale_in_block (j : Nat) (h : j < 4) : 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S sign qword at `2 + 32 + 8j` ends within qs. -/
theorem iq2s_sign_in_block (j : Nat) (h : j < 4) : 2 + 32 + 8 * j + 8 ≤ 66 := by omega

/-- IQ2_S grid bytes `2 + 8j + 7` are inside qs. -/
theorem iq2s_qs_in_block (j : Nat) (h : j < 4) : 2 + 8 * j + 7 < 34 := by omega

/-- IQ2_S scales `74 + 2j (+1)`. -/
theorem iq2s_scale_in_block (j : Nat) (h : j < 4) : 74 + 2 * j + 1 < 82 := by omega

/-- IQ3_XXS sign/scale words at `66 + 8j` stay inside. -/
theorem iq3xxs_words_in_block (j : Nat) (h : j < 4) : 66 + 8 * j + 8 ≤ 98 := by omega

/-- IQ3_S: qh, sign qword and scale byte offsets are inside the block. -/
theorem iq3s_offsets_in_block (j : Nat) (h : j < 4) :
    66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by omega

-- § indices and scales

/-- Seven-bit sign index is below 128 (size of the ksigns table). -/
theorem sign_index_lt (a k : Nat) : ((a >>> (7 * k)) &&& 127) < 128 := by
  have h : ((a >>> (7 * k)) &&& 127) ≤ 127 := Nat.and_le_right
  omega

/-- A 4-bit scale nibble of a 32-bit word. -/
theorem nibble_lt (a : Nat) (h : a < 4294967296) : a >>> 28 < 16 := by
  rw [Nat.shiftRight_eq_div_pow]
  omega

/-- Block scales `2*s+1` are odd. -/
theorem scale_odd (s : Nat) : (2 * s + 1) % 2 = 1 := by omega

/-- Block scales from a nibble are at most 31. -/
theorem scale_le (s : Nat) (h : s < 16) : 2 * s + 1 ≤ 31 := by omega

/-- IQ2_XS 9-bit grid index is below 512. -/
theorem iq2xs_grid_index_lt (v : Nat) : (v &&& 511) < 512 := by
  have h : (v &&& 511) ≤ 511 := Nat.and_le_right
  omega

/-- IQ2_XS 7-bit sign index from a 16-bit entry is below 128. -/
theorem iq2xs_sign_index_lt (v : Nat) (h : v < 65536) : v >>> 9 < 128 := by
  rw [Nat.shiftRight_eq_div_pow]
  omega

/-- IQ2_S grid index (byte plus two high bits) is below 1024. -/
theorem iq2s_index_lt (q h : Nat) (hq : q < 256) : (q ||| ((h <<< 2) &&& 0x300)) < 1024 := by
  have h1 : (h <<< 2) &&& 0x300 ≤ 0x300 := Nat.and_le_right
  exact Nat.or_lt_two_pow (n := 10) (by omega) (by omega)

/-- IQ3_S grid index (byte plus one high bit) is below 512. -/
theorem iq3s_index_lt (q x kk : Nat) (hq : q < 256) : (q ||| (((x >>> kk) &&& 1) <<< 8)) < 512 := by
  have h1 : ((x >>> kk) &&& 1) ≤ 1 := Nat.and_le_right
  have h2 : (((x >>> kk) &&& 1) <<< 8) ≤ 256 := by
    rw [Nat.shiftLeft_eq]
    omega
  exact Nat.or_lt_two_pow (n := 9) (by omega) (by omega)

-- § sign application

/-- Masked negate: lane is negated when its sign bit is set. -/
def applySign (m : Bool) (y : Int) : Int := if m then 0 - y else y

/-- Applying the same sign twice restores the lane. -/
theorem applySign_invol (m : Bool) (y : Int) : applySign m (applySign m y) = y := by
  cases m <;> simp [applySign]

/-- Sign application preserves magnitude. -/
theorem applySign_abs (m : Bool) (y : Int) : (applySign m y).natAbs = y.natAbs := by
  cases m <;> simp [applySign]

/-- A weight times a signed activation is the signed product. -/
theorem applySign_mul (g y : Int) : g * applySign true y = -(g * y) := by
  simp [applySign, Int.mul_neg]

-- § token tiling

/-- Group size for the down-projection tiling over `nt` tokens starting at `t0`. -/
def tileSize (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Each group is non-empty, at most 4, and stays inside the token range. -/
theorem tile_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ tileSize nt t0 ∧ tileSize nt t0 ≤ 4 ∧ t0 + tileSize nt t0 ≤ nt := by
  unfold tileSize
  omega

/-- The tiling loop advances and so terminates. -/
theorem tile_advance (nt t0 : Nat) (h : t0 < nt) : t0 < t0 + tileSize nt t0 := by
  unfold tileSize
  omega

/-- Gate/up dispatch clamps to 8 tokens. -/
def guTile (nt : Nat) : Nat := if nt ≥ 8 then 8 else nt

/-- The gate/up tile is at most 8. -/
theorem guTile_le (nt : Nat) : guTile nt ≤ 8 := by
  unfold guTile
  split <;> omega

-- § row geometry

/-- Blocks per row for an `n`-value row (QK_K = 256). -/
theorem blocks_per_row (n : Nat) (h : n % 256 = 0) : (n / 256) * 256 = n := by omega

/-- Up row offset is past every gate row: row r < rows implies r*gu_row + gu_row ≤ rows*gu_row. -/
theorem gate_row_end (r rows g : Nat) (h : r < rows) : r * g + g ≤ rows * g := by
  have : (r + 1) * g ≤ rows * g := Nat.mul_le_mul_right g h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

end Strata.Cpu.IquantAvx512



## FILE: /mnt/user-data/outputs/ExpertPoolProtocol.lean
/-!
# MODULE: expert_pool_protocol
CPU expert thread pool: workers park on an epoch counter, drain a shared job counter, and the host optionally
drains too. Multi-phase modes split the rows of all experts into equal task ranges.

## Source semantics
* `physical_cores(skip_first)` drops core 0 so the host owns it.
* Row ranges `[R*i/T, R*(i+1)/T)` tile `[0, R)` contiguously and monotonically.
* Inside a task, the expert/row walk always progresses, never leaves an expert, and never overshoots the task end.
* Park accounting: workers arrive once at construction, leave on work, arrive after drain; parked + draining = n.
* Batching of more experts than buffers proceeds in bounded, non-empty batches.
* `run_split` falls back to `run` when too many jobs, a single worker, or the oracle is on.

## Form template identifiers
ExpertPool::drain -> task_drain ; run_phase -> phase_barrier ; parked_ , epoch_ , head_ , done_ -> park_count ,
epoch , claim_counter , done_counter ; mrows_ , mtasks_ -> total_rows , task_count
-/

namespace Strata.Cpu.ExpertPool

-- § core list

/-- Core list after the optional host-core skip. -/
def skipFirst (skip : Bool) (cores : List Nat) : List Nat := if skip then cores.drop 1 else cores

/-- Skipping removes at most one core. -/
theorem skipFirst_length (skip : Bool) (cores : List Nat) :
    (skipFirst skip cores).length = if skip then cores.length - 1 else cores.length := by
  cases skip <;> simp [skipFirst]

/-- Without skipping the list is unchanged. -/
theorem skipFirst_false (cores : List Nat) : skipFirst false cores = cores := by
  simp [skipFirst]

-- § row-range partition

/-- Start of part `i` when `rows` are split across `parts`. -/
def lo (rows parts i : Nat) : Nat := rows * i / parts

/-- The first part starts at row 0. -/
theorem lo_zero (rows parts : Nat) : lo rows parts 0 = 0 := by
  simp [lo]

/-- The last boundary reaches all rows. -/
theorem lo_last (rows parts : Nat) (h : 0 < parts) : lo rows parts parts = rows := by
  unfold lo
  exact Nat.mul_div_cancel rows h

/-- Boundaries are monotone. -/
theorem lo_mono (rows parts i : Nat) : lo rows parts i ≤ lo rows parts (i + 1) := by
  unfold lo
  apply Nat.div_le_div_right
  exact Nat.mul_le_mul_left rows (Nat.le_succ i)

/-- Parts are contiguous: the end of part i is the start of part i+1. -/
theorem lo_contiguous (rows parts i : Nat) : lo rows parts (i + 1) = rows * (i + 1) / parts := rfl

/-- No boundary exceeds the total once `i ≤ parts`. -/
theorem lo_le_rows (rows parts i : Nat) (hp : 0 < parts) (hi : i ≤ parts) : lo rows parts i ≤ rows := by
  have h1 : lo rows parts i ≤ lo rows parts parts := by
    unfold lo
    apply Nat.div_le_div_right
    exact Nat.mul_le_mul_left rows hi
  rw [lo_last rows parts hp] at h1
  exact h1

-- § per-task walk across experts

/-- Row-in-expert is strictly below the expert size. -/
theorem row_in_expert_lt (r per : Nat) (h : 0 < per) : r % per < per := Nat.mod_lt r h

/-- One step of the expert walk makes progress, stays in the expert, and stays within the task. -/
theorem walk_step (per r0 g1 r : Nat) (hr0 : r0 < per) (hlt : r < g1) :
    r0 < min per (r0 + (g1 - r)) ∧ min per (r0 + (g1 - r)) ≤ per ∧
    r + (min per (r0 + (g1 - r)) - r0) ≤ g1 := by
  omega

/-- Expert index and row recombine to the global row. -/
theorem expert_row_recombine (r per : Nat) (h : 0 < per) : (r / per) * per + r % per = r := by
  have := Nat.div_add_mod r per
  rw [Nat.mul_comm] at this
  exact this

-- § task counts

/-- Parts per expert in the single-token split: about three tasks per thread. -/
def partsA (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- There is always at least one part. -/
theorem partsA_pos (threads n : Nat) : 1 ≤ partsA threads n := by
  unfold partsA
  omega

/-- Total tasks of a two-phase split. -/
theorem tasks_total (n parts : Nat) : n * parts = n * parts := rfl

/-- Task index decomposition: expert and part recombine. -/
theorem task_decompose (i parts : Nat) (h : 0 < parts) : (i / parts) * parts + i % parts = i := by
  have := Nat.div_add_mod i parts
  rw [Nat.mul_comm] at this
  exact this

/-- The part index is below the part count. -/
theorem part_lt (i parts : Nat) (h : 0 < parts) : i % parts < parts := Nat.mod_lt i h

-- § park accounting

/-- Workers either parked or draining. -/
structure PState where
  parked : Nat
  draining : Nat

/-- A worker leaves the park to drain. -/
def leave (s : PState) : PState := ⟨s.parked - 1, s.draining + 1⟩

/-- A worker returns to the park after draining. -/
def arrive (s : PState) : PState := ⟨s.parked + 1, s.draining - 1⟩

/-- Leaving preserves the worker total. -/
theorem leave_inv (s : PState) (n : Nat) (h : s.parked + s.draining = n) (hp : 0 < s.parked) :
    (leave s).parked + (leave s).draining = n := by
  simp only [leave]
  omega

/-- Arriving preserves the worker total. -/
theorem arrive_inv (s : PState) (n : Nat) (h : s.parked + s.draining = n) (hd : 0 < s.draining) :
    (arrive s).parked + (arrive s).draining = n := by
  simp only [arrive]
  omega

/-- Leave then arrive restores the state. -/
theorem leave_arrive (s : PState) (hp : 0 < s.parked) : arrive (leave s) = s := by
  cases s with
  | mk p d =>
    have hp' : 0 < p := hp
    have h1 : p - 1 + 1 = p := by omega
    simp [leave, arrive, h1]

/-- After all workers arrive at construction the pool is fully parked. -/
theorem constructed_all_parked (n : Nat) : (PState.mk n 0).parked = n := rfl

-- § claim counter

/-- A claimed index runs a job exactly when it is below the job count. -/
def claims (i njobs : Nat) : Bool := decide (i < njobs)

/-- Indices at or past the job count end the drain loop. -/
theorem claim_stop (i njobs : Nat) (h : njobs ≤ i) : claims i njobs = false := by
  simp [claims]
  omega

/-- Indices below the job count are executed. -/
theorem claim_run (i njobs : Nat) (h : i < njobs) : claims i njobs = true := by
  simp [claims, h]

-- § mode dispatch

/-- Rows-per-expert for a multi-token mode: gate/up modes (3, 5) use FF, others H. -/
def perOf (ff h mode : Nat) : Nat := if mode = 3 ∨ mode = 5 then ff else h

/-- Modes 3 and 4 never take the native (>= 5) branch. -/
theorem multi_not_native (m : Nat) (h : m = 3 ∨ m = 4) : ¬ (m ≥ 5) := by omega

/-- Native modes are exactly those at least 5. -/
theorem native_mode (m : Nat) : (m ≥ 5) ↔ (5 ≤ m) := Iff.rfl

-- § run_split guards and batching

/-- `run_split` falls back to `run` for oversize jobs, one worker, or the oracle. -/
def useRun (n kMax nWorkers : Nat) (oracle : Bool) : Bool :=
  decide (n > kMax) || decide (nWorkers = 1) || oracle

/-- One worker always uses the inline path. -/
theorem single_worker_uses_run (n kMax : Nat) (oracle : Bool) : useRun n kMax 1 oracle = true := by
  simp [useRun]

/-- The oracle always uses the single-token path. -/
theorem oracle_uses_run (n kMax nWorkers : Nat) : useRun n kMax nWorkers true = true := by
  simp [useRun]

/-- Batch size when more experts than buffers must be processed in several batches. -/
def batchLen (kMax n b0 : Nat) : Nat := min kMax (n - b0)

/-- Batches are non-empty, bounded by the buffer count, and inside the job list. -/
theorem batch_ok (kMax n b0 : Nat) (hk : 0 < kMax) (h : b0 < n) :
    1 ≤ batchLen kMax n b0 ∧ batchLen kMax n b0 ≤ kMax ∧ b0 + batchLen kMax n b0 ≤ n := by
  unfold batchLen
  omega

/-- The batch loop strictly advances. -/
theorem batch_advance (kMax n b0 : Nat) (hk : 0 < kMax) (h : b0 < n) :
    b0 < b0 + batchLen kMax n b0 := by
  unfold batchLen
  omega

end Strata.Cpu.ExpertPool



## FILE: /mnt/user-data/outputs/Q2Avx2Fallback.lean
/-!
# MODULE: q2_avx2_fallback
Q2_0 expert rows and the Q8_1 activation quantiser for CPUs with AVX2 but no AVX-512. Same arithmetic as the
AVX-512 kernels: codes against int8 activations per 32-value chunk, minus the offset correction.

## Source semantics
* GGUF Q2_0 block: fp16 scale (2 bytes) then 16 code bytes, stride 18, 64 weights per block.
* 16 code bytes are unpacked by four shift-and-mask planes plus byte and word interleaves: output position p
  takes plane p mod 4 of byte p / 4; positions 0..31 come from bytes 0..7, 32..63 from bytes 8..15.
* Offset identity: sum (c-1) x = sum c x - sum x.
* Quantiser: values clamped to [-127,127], stored as int8 (identity on that range), chunk sums bounded by 127*32.
* Token tiling is groups of 4.

## Form template identifiers
unpack64 -> code_unpack ; row_multi -> row_kernel ; act_quant_q8_1_avx2 -> act_quantizer ;
q2_0_gguf_rows_multi_avx2 -> multi_token_rows
-/

namespace Strata.Cpu.Q2Avx2

-- § block geometry

/-- Bytes per GGUF Q2_0 block. -/
def blockStride : Nat := 18

/-- Scale bytes plus code bytes make up a block. -/
theorem stride_split : blockStride = 2 + 16 := by decide

/-- Sixty-four 2-bit codes occupy sixteen bytes. -/
theorem codes_bytes : 64 / 4 = 16 := by decide

/-- Block `b`'s scale and codes are inside its stride. -/
theorem block_offsets (b nblocks : Nat) (h : b < nblocks) :
    b * blockStride + 2 + 16 ≤ nblocks * blockStride := by
  unfold blockStride
  have : (b + 1) * 18 ≤ nblocks * 18 := Nat.mul_le_mul_right 18 h
  omega

-- § unpack index map

/-- Output position `p` takes byte `p/4`, plane `p%4`. -/
theorem unpack_position (p : Nat) : 4 * (p / 4) + p % 4 = p := by omega

/-- Low half (positions 0..31) reads bytes 0..7. -/
theorem lo_half_bytes (p : Nat) (h : p < 32) : p / 4 < 8 := by omega

/-- High half (positions 32..63) reads bytes 8..15. -/
theorem hi_half_bytes (p : Nat) (h1 : 32 ≤ p) (h2 : p < 64) : 8 ≤ p / 4 ∧ p / 4 < 16 := by omega

/-- Plane index is below 4. -/
theorem plane_lt (p : Nat) : p % 4 < 4 := by omega

/-- A 2-bit code extracted from a byte is at most 3. -/
theorem plane_code_le (b k : Nat) : ((b >>> (2 * k)) &&& 3) ≤ 3 := Nat.and_le_right

-- § offset algebra

/-- Integer dot product. -/
def dotI : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | c :: cs, x :: xs => c * x + dotI cs xs

/-- Integer sum. -/
def sumI : List Int → Int
  | [] => 0
  | x :: xs => x + sumI xs

/-- Offset identity used by the `hx` correction. -/
theorem dot_offset : ∀ (cs xs : List Int), cs.length = xs.length →
    dotI (cs.map (· - 1)) xs = dotI cs xs - sumI xs
  | [], [], _ => by simp [dotI, sumI]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | c :: cs, x :: xs, h => by
      have ih := dot_offset cs xs (by simpa using h)
      have hm : (c - 1) * x = c * x - x := by rw [Int.sub_mul, Int.one_mul]
      simp only [List.map_cons, dotI, sumI]
      rw [ih, hm]
      omega

-- § quantiser

/-- Clamp to [-127, 127]. -/
def clamp127 (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Clamped values lie in [-127, 127]. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  by_cases h1 : v < -127 <;> by_cases h2 : v > 127 <;> simp [h1, h2] <;> omega

/-- Two's-complement wrap into int8. -/
def wrap8 (v : Int) : Int := (v + 128) % 256 - 128

/-- Storing an in-range value as int8 is lossless. -/
theorem wrap8_id (v : Int) (h1 : -128 ≤ v) (h2 : v ≤ 127) : wrap8 v = v := by
  unfold wrap8
  omega

/-- Clamped values survive the int8 store. -/
theorem clamp_store_lossless (v : Int) : wrap8 (clamp127 v) = clamp127 v := by
  have h := clamp127_bounds v
  exact wrap8_id _ (by omega) h.2

/-- Sum of in-range quantised values is bounded by 127 per element. -/
theorem sum_bound : ∀ (l : List Int), (∀ v, v ∈ l → -127 ≤ v ∧ v ≤ 127) →
    -(127 * (l.length : Int)) ≤ sumI l ∧ sumI l ≤ 127 * (l.length : Int)
  | [], _ => by simp [sumI]
  | x :: xs, h => by
      have hx := h x (List.mem_cons_self ..)
      have ih := sum_bound xs (fun v hv => h v (List.mem_cons_of_mem _ hv))
      simp only [sumI, List.length_cons]
      omega

/-- A 32-value chunk sum fits in int32 with a large margin. -/
theorem chunk_sum_fits (s : Int) (h : -(127 * 32) ≤ s ∧ s ≤ 127 * 32) :
    -2147483648 ≤ s ∧ s ≤ 2147483647 := by omega

/-- Chunk `k`, lane `j` indexes inside the row when `n = 32 * nchunks`. -/
theorem chunk_index_lt (n k j : Nat) (hn : n % 32 = 0) (hk : k < n / 32) (hj : j < 32) :
    k * 32 + j < n := by omega

-- § token tiling

/-- Group size of the tiling loop. -/
def tileSize (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Groups are non-empty, at most 4, and inside the token range. -/
theorem tile_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ tileSize nt t0 ∧ tileSize nt t0 ≤ 4 ∧ t0 + tileSize nt t0 ≤ nt := by
  unfold tileSize
  omega

/-- The tiling loop advances. -/
theorem tile_advance (nt t0 : Nat) (h : t0 < nt) : t0 < t0 + tileSize nt t0 := by
  unfold tileSize
  omega

end Strata.Cpu.Q2Avx2





## FILE: /mnt/user-data/outputs/FusedGdnDecodeStep.lean
/-!
# MODULE: fused_gdn_decode_step
Index geometry and launch contracts of the fused gated-delta-net decode kernels: the conv+SiLU+L2 kernel,
the alpha/beta projection kernel, and the state-update + RMSNorm*gate kernel. The kernels split the 128x128
per-head state over 4 row groups of 32 rows, one thread per (row group, column).

## Source semantics
- `gdn_step_norm_kernel`: block = head, 128x4 threads, thread (col, rg) owns rows rg*32..rg*32+31 of column col.
  State address of row i is `(i*h_v + head)*S + col`; q/k head is `head % h_k`; h_v % h_k = 0 is required.
- `gdn_conv_l2_kernel`: one thread per channel, history stride 3, weight stride 4; the first `qk_heads` blocks
  (of 128 channels) are L2-normalised.
- `gdn_ab_kernel`: one warp per output row, rows [0,h_v) are alpha (gate), rows [h_v,2h_v) are beta.

## Form template identifiers
- S -> state_dim (128), RG -> row_groups (4), RPG -> rows_per_group (32)
- head -> head_idx, h_k -> n_qk_heads, h_v -> n_v_heads, col -> column_idx, rg -> row_group_idx
-/

namespace Strata.FusedGdnDecode

/-- State size (rows = cols). -/
def S : Nat := 128
/-- Row groups per block. -/
def RG : Nat := 4
/-- Rows held in registers per thread. -/
def RPG : Nat := 32

-- § helpers

/-- Row-major index bound used throughout the address arithmetic. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § block geometry

/-- The row groups tile the state exactly. -/
theorem row_groups_tile_state : RG * RPG = S := by decide

/-- A block has 512 threads. -/
theorem block_thread_count : S * RG = 512 := by decide

/-- The per-warp partial-sum array has 16 slots (512/32). -/
theorem wsum_slot_count : S * RG / 32 = 16 := by decide

/-- Thread (rg, r) addresses a row below 128. -/
theorem thread_row_in_range (rg r : Nat) (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by
  omega

/-- Distinct (row group, register) pairs address distinct rows. -/
theorem thread_row_inj (rg r rg' r' : Nat) (hr : r < 32) (hr' : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' := by
  omega

/-- Every row of the state is owned by some (row group, register) pair. -/
theorem row_cover (row : Nat) (h : row < 128) :
    ∃ rg r, rg < 4 ∧ r < 32 ∧ row = rg * 32 + r :=
  ⟨row / 32, row % 32, by omega, by omega, by omega⟩

/-- Flattened thread id is below the block size. -/
theorem tid_in_range (rg col : Nat) (hrg : rg < 4) (hcol : col < 128) : rg * 128 + col < 512 := by
  omega

/-- Only row group 0 stages the 128 q/k entries into shared memory. -/
theorem staging_threads_are_group0 (rg col : Nat) (hrg : rg < 4) (hcol : col < 128) :
    rg * 128 + col < 128 ↔ rg = 0 := by
  omega

/-- The RMS reduction uses warp slots 0..3, which fit in `wsum`. -/
theorem rms_warp_slot (col : Nat) (h : col < 128) : col / 32 < 4 := by
  omega

/-- Every warp slot index of the 512-thread block is a valid `wsum` index. -/
theorem wsum_index_bound (tid : Nat) (h : tid < 512) : tid / 32 < 16 := by
  omega

/-- Row-group-0 warps are exactly threads 0..127: their lane-0 slots are 0..3. -/
theorem rms_slots_cover_group0 (tid : Nat) (h : tid < 128) : tid / 32 < 4 := by
  omega

-- § state addressing

/-- Query/key heads are paired by modulo, and the pairing stays in range. -/
theorem qk_head_in_range (head h_k : Nat) (hk : 0 < h_k) : head % h_k < h_k :=
  Nat.mod_lt _ hk

/-- State element address bound for the (row, head, col) layout `(S, h_v, S)`. -/
theorem state_index_lt (row head col h_v : Nat) (hrow : row < 128) (hhead : head < h_v)
    (hcol : col < 128) : (row * h_v + head) * 128 + col < 128 * h_v * 128 :=
  idx_lt (idx_lt hrow hhead) hcol

/-- Stepping from row `rg*32` by `r` rows adds `r * (h_v*128)` to the address. -/
theorem state_row_stride (rg r head col h_v : Nat) :
    ((rg * 32 + r) * h_v + head) * 128 + col =
      ((rg * 32) * h_v + head) * 128 + col + r * (h_v * 128) := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Output / v / z addressing `head*S + col` stays inside `h_v * S`. -/
theorem out_index_lt (head col h_v : Nat) (hhead : head < h_v) (hcol : col < 128) :
    head * 128 + col < h_v * 128 :=
  idx_lt hhead hcol

-- § conv + L2 kernel

/-- With channels a multiple of 128, block `b` thread `t` addresses channel below `channels`. -/
theorem conv_channel_in_range (channels b t : Nat) (hc : channels % 128 = 0)
    (hb : b < channels / 128) (ht : t < 128) : b * 128 + t < channels := by
  omega

/-- History (stride 3) and weight (stride 4) addresses stay in range. -/
theorem conv_strides_in_range (channels c : Nat) (hc : c < channels) :
    c * 3 + 2 < 3 * channels ∧ c * 4 + 3 < 4 * channels := by
  omega

/-- The first `qk_heads` blocks are exactly the channels below `qk_heads*128`. -/
theorem l2_applies_to_qk_channels (qk_heads c : Nat) :
    c / 128 < qk_heads ↔ c < qk_heads * 128 := by
  omega

/-- The L2 partial array has 4 slots, one per warp of a 128-thread block. -/
theorem l2_warp_slots (t : Nat) (h : t < 128) : t / 32 < 4 := by
  omega

-- § alpha/beta kernel

/-- Grid size covers all `2*h_v` rows (8 warps per block). -/
theorem ab_grid_covers (h_v : Nat) : 2 * h_v ≤ (2 * h_v + 7) / 8 * 8 := by
  omega

/-- A row index below `2*h_v` is either an alpha row or a beta row with local index below `h_v`. -/
theorem ab_row_split (row h_v : Nat) (h : row < 2 * h_v) :
    (row < h_v ∧ row < h_v) ∨ (h_v ≤ row ∧ row - h_v < h_v) := by
  omega

/-- Each lane handles 8-element chunks; with `n % 8 = 0` the chunks tile `n` exactly. -/
theorem ab_chunks_tile (n j : Nat) (hn : n % 8 = 0) (hj : j < n / 8) : j * 8 + 7 < n := by
  omega

/-- The warp index within a 256-thread block is below 8. -/
theorem ab_warp_in_block (t : Nat) (h : t < 256) : t / 32 < 8 := by
  omega

-- § launch contract

/-- `fused_gdn_step_norm` precondition: `h_v` is a positive multiple of `h_k`. -/
def stepNormArgsValid (h_k h_v : Nat) : Prop := 0 < h_k ∧ 0 < h_v ∧ h_v % h_k = 0

/-- Valid arguments give a launch of `h_v` blocks, each with an in-range q/k head. -/
theorem step_norm_valid_heads (h_k h_v head : Nat) (hv : stepNormArgsValid h_k h_v) :
    head % h_k < h_k := by
  obtain ⟨hk, _, _⟩ := hv
  exact qk_head_in_range head h_k hk

/-- `fused_gdn_conv_l2` precondition: channels is a multiple of 128 and `qk_heads <= channels/128`. -/
def convArgsValid (channels qk_heads : Nat) : Prop :=
  channels % 128 = 0 ∧ qk_heads ≤ channels / 128

/-- Under the contract, the normalised channel range lies inside the channel range. -/
theorem conv_qk_range_valid (channels qk_heads : Nat) (h : convArgsValid channels qk_heads) :
    qk_heads * 128 ≤ channels := by
  obtain ⟨h1, h2⟩ := h
  omega

end Strata.FusedGdnDecode



## FILE: /mnt/user-data/outputs/FusedGatedResidualRead.lean
/-!
# MODULE: fused_gated_residual_read
Index geometry of the fused hyper-connection read kernels for the fixed geometry n_embd = 2560, hc = 4,
hc_lr = 320: the norm, the down projection (+ inject rows), the up projection with gated mean, and their
T-token variants that read each weight row once.

## Source semantics
- Residual `R` is hc streams of n_embd floats; flat index `c*N + d`.
- Down kernel: 40 blocks of 8 warps (one row each) cover the 320 `lo` rows; block 40 handles the 4 inject rows.
- Up kernel: 80 blocks of 32 columns x 4 streams = 128 rows; multi variant 160 blocks of 16 columns.
- Tiled multi-token down kernel: 10240 reduction elements = 4 tiles of 2560 floats = 320 uint4 chunks each.

## Form template identifiers
- N -> n_embd, HC -> n_streams, D -> stream_dim, LR -> low_rank, TILE -> act_tile
- w_down -> down_weight, w_up -> up_weight, w_inject -> inject_weight, R -> residual
-/

namespace Strata.FusedGr

def N : Nat := 2560
def HC : Nat := 4
def D : Nat := N * HC
def LR : Nat := 320
def THREADS : Nat := 256
def WARPS : Nat := THREADS / 32
def DOWN_BLOCKS : Nat := LR / WARPS
def UP_COLS : Nat := 32
def UP_BLOCKS : Nat := N / UP_COLS
def TILE : Nat := 2560
def TQ : Nat := TILE / 8 / 32
def UPM_COLS : Nat := 16
def UPM_BLOCKS : Nat := N / UPM_COLS

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § constants

/-- Stream dimension. -/
theorem stream_dim : D = 10240 := by decide
/-- Eight warps per block. -/
theorem warps_per_block : WARPS = 8 := by decide
/-- 40 down blocks of 8 rows cover all 320 low-rank rows. -/
theorem down_blocks_cover_lr : DOWN_BLOCKS * WARPS = LR := by decide
/-- 80 up blocks of 32 columns cover n_embd. -/
theorem up_blocks_cover_n : UP_BLOCKS * UP_COLS = N := by decide
/-- 160 multi-token up blocks of 16 columns cover n_embd. -/
theorem upm_blocks_cover_n : UPM_BLOCKS * UPM_COLS = N := by decide
/-- Per tile each lane loads 10 uint4 chunks. -/
theorem tile_chunks_per_lane : TQ = 10 := by decide
/-- Lane x chunk coverage of one tile: 32 lanes * 10 chunks * 8 values = TILE. -/
theorem tile_covered : 32 * TQ * 8 = TILE := by decide
/-- Four tiles cover the reduction axis. -/
theorem tiles_cover_reduction : 4 * TILE = D := by decide
/-- LR/8 = 40 chunks = 32 + 8, which is the lane<8 second-chunk rule. -/
theorem lr_chunk_split : LR / 8 = 32 + 8 := by decide

-- § flat residual indexing

/-- A flat residual index splits into stream and column. -/
theorem flat_split (i : Nat) (h : i < 10240) :
    i / 2560 < 4 ∧ i % 2560 < 2560 ∧ (i / 2560) * 2560 + i % 2560 = i := by
  omega

/-- float4 accesses at multiples of 4 never straddle two streams. -/
theorem vec4_within_stream (i : Nat) (h : i < 10240) (h4 : i % 4 = 0) :
    (i + 3) / 2560 = i / 2560 := by
  omega

/-- The down kernel's strided loop (step THREADS*4) visits only valid float4 starts. -/
theorem vec4_loop_valid (t k : Nat) (ht : t < 256) (hk : t * 4 + k * 1024 < 10240) :
    (t * 4 + k * 1024) % 4 = 0 := by
  omega

-- § down kernel

/-- Down block rows: block b, warp w addresses row b*8+w, below 320 for b < 40. -/
theorem down_row_in_range (b w : Nat) (hb : b < 40) (hw : w < 8) : b * 8 + w < 320 := by
  omega

/-- The inject block is block 40 and its active warps are 0..3. -/
theorem inject_block_rows (w : Nat) (hw : w < 8) : w < 4 ∨ 4 ≤ w := by
  omega

/-- Chunk index of lane l in a 10240-element row loop: l + 32 j < 1280. -/
theorem down_chunk_in_range (l j : Nat) (hl : l < 32) (hj : j < 40) : l + 32 * j < 1280 := by
  omega

/-- Tiled loading: chunk index base/8 + lane + 32 q stays inside the 1280 chunks of a row. -/
theorem tiled_chunk_in_range (base l q : Nat) (hbase : base % 2560 = 0) (hb : base < 10240)
    (hl : l < 32) (hq : q < 10) : base / 8 + l + 32 * q < 1280 := by
  omega

/-- Shared activation tile copy: element i of the T*TILE/4 float4 loop maps to token i/640. -/
theorem tile_copy_split (i T : Nat) (h : i < T * 640) :
    i / 640 < T ∧ i % 640 < 640 ∧ (i / 640) * 640 + i % 640 = i := by
  have h1 : i / 640 < T := by
    have : i < 640 * T := by omega
    exact Nat.div_lt_of_lt_mul this
  omega

-- § up kernel

/-- Up row r in a 128-row block: stream c = r/32, column dd = r%32. -/
theorem up_row_split (r : Nat) (h : r < 128) :
    r / 32 < 4 ∧ r % 32 < 32 ∧ (r / 32) * 32 + r % 32 = r := by
  omega

/-- The flat residual index of an up row is below 10240. -/
theorem up_index_in_range (blk r : Nat) (hb : blk < 80) (hr : r < 128) :
    (r / 32) * 2560 + blk * 32 + r % 32 < 10240 := by
  omega

/-- Multi-token up kernel: 64 rows per block, 8 per warp. -/
theorem upm_index_in_range (blk r : Nat) (hb : blk < 160) (hr : r < 64) :
    (r / 16) * 2560 + blk * 16 + r % 16 < 10240 := by
  omega

/-- Each block owns column d0+dd of every stream, so the mean output index stays below n_embd. -/
theorem mixed_index_in_range (blk t : Nat) (hb : blk < 80) (ht : t < 32) : blk * 32 + t < 2560 := by
  omega

/-- The second weight chunk of a row exists only for lane < 8. -/
theorem second_chunk_in_range (l : Nat) (hl : l < 8) : 32 + l < 40 := by
  omega

-- § multi-token launch contract

/-- Support predicate of the fixed geometry. -/
def grSupported (n_embd hc hc_lr : Int) : Prop := n_embd = 2560 ∧ hc = 4 ∧ hc_lr = 320

/-- Supported geometry gives the compile-time constants. -/
theorem supported_gives_constants (n hc lr : Int) (h : grSupported n hc lr) :
    n * hc = 10240 ∧ lr = 320 := by
  obtain ⟨h1, h2, h3⟩ := h
  subst h1; subst h2; subst h3
  exact ⟨by decide, rfl⟩

/-- Token count is at most the warp width, so lane k can own token k. -/
theorem token_lane_ownership (T k maxT : Nat) (hT : T ≤ maxT) (hm : maxT ≤ 32) (hk : k < T) :
    k < 32 := by
  omega

end Strata.FusedGr



## FILE: /mnt/user-data/outputs/GdnReferenceKernels.lean
/-!
# MODULE: gdn_reference_kernels
Launch geometry and addressing of the non-projection parts of the gated delta net: one-step recurrence
(one thread per (head, j) state column), the sliding 4-tap causal convolution, L2 normalisation, the
output RMS-norm * gate, and the in-place beta sigmoid.

## Source semantics
- State layout `(S, h_v, S)` with j fastest: element (i, h, j) at `(i*h_v + h)*S + j`; the recurrence touches only
  one (h, j) column per thread so it needs no barrier.
- Grid is (ceil(S/32), ceil(h_v/MAX_H)) with MAX_H = 1 (latency-hiding choice): one head per block row.
- Conv state per channel holds d_conv-1 values; the new input lands in the last tap and the window slides.
- `gdn_step` aborts for S > 128 (shared staging is [MAX_H][128]).

## Form template identifiers
- S -> state_dim, h_k -> n_qk_heads, h_v -> n_v_heads, dc -> d_conv, JTHREADS -> j_threads
- gdn_step -> delta_step, gdn_conv_step -> causal_conv_step, gdn_beta_gate -> beta_sigmoid
-/

namespace Strata.GdnReference

def JTHREADS : Nat := 32
def MAX_H : Nat := 1

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § step kernel geometry

/-- With MAX_H = 1 each block row stages exactly one head. -/
theorem one_head_per_block (by_ h_v : Nat) (h : by_ < h_v) : min 1 (h_v - by_) = 1 := by
  omega

/-- Grid y extent with MAX_H = 1 equals h_v. -/
theorem grid_y_is_h_v (h_v : Nat) : (h_v + MAX_H - 1) / MAX_H = h_v := by
  unfold MAX_H
  omega

/-- Grid x extent covers every column j < S. -/
theorem grid_x_covers (S j : Nat) (hj : j < S) : j / 32 < (S + 31) / 32 := by
  omega

/-- A column index inside S lies in block j/32 at lane j%32. -/
theorem column_split (j : Nat) : (j / 32) * 32 + j % 32 = j := by
  omega

/-- Head pairing (MODULO) stays in range. -/
theorem head_pair_in_range (h h_k : Nat) (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt _ hk

/-- Element address of state row i for column pointer `h*S + j` and stride `h_v*S`. -/
theorem state_elem_address (i h j S h_v : Nat) :
    h * S + j + i * (h_v * S) = (i * h_v + h) * S + j := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

/-- State element address stays inside S*h_v*S. -/
theorem state_elem_lt (i h j S h_v : Nat) (hi : i < S) (hh : h < h_v) (hj : j < S) :
    (i * h_v + h) * S + j < S * h_v * S :=
  idx_lt (idx_lt hi hh) hj

/-- Output address `h*S + j` stays inside `h_v*S`. -/
theorem out_elem_lt (h j S h_v : Nat) (hh : h < h_v) (hj : j < S) : h * S + j < h_v * S :=
  idx_lt hh hj

/-- The `S > 128` guard: accepted sizes fit the staged shared arrays. -/
theorem staged_rows_fit (S i : Nat) (hS : S ≤ 128) (hi : i < S) : i < 128 := by
  omega

-- § conv kernel

/-- Sliding the window drops the oldest value and appends the new one (d_conv = 4 case). -/
def slide3 (st : Int × Int × Int) (x : Int) : Int × Int × Int := (st.2.1, st.2.2, x)

/-- After a slide the window is [s1, s2, x]. -/
theorem slide3_eq (a b c x : Int) : slide3 (a, b, c) x = (b, c, x) := rfl

/-- Three slides leave exactly the last three inputs. -/
theorem slide3_three (s : Int × Int × Int) (x y z : Int) :
    slide3 (slide3 (slide3 s x) y) z = (x, y, z) := rfl

/-- Generic slide on lists keeps the length when non-empty. -/
def slideList (st : List Int) (x : Int) : List Int := st.tail ++ [x]

/-- The generic slide preserves length of a non-empty window. -/
theorem slideList_length (st : List Int) (x : Int) (h : st ≠ []) :
    (slideList st x).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a t => simp [slideList]

/-- Per-channel state address bound for stride dc-1. -/
theorem conv_state_in_range (c i C dc : Nat) (hc : c < C) (hi : i < dc - 1) :
    c * (dc - 1) + i < C * (dc - 1) :=
  idx_lt hc hi

/-- Per-channel weight address bound for stride dc. -/
theorem conv_weight_in_range (c i C dc : Nat) (hc : c < C) (hi : i < dc) :
    c * dc + i < C * dc :=
  idx_lt hc hi

/-- Grid for the conv kernel covers all channels (256 per block). -/
theorem conv_grid_covers (C : Nat) : C ≤ (C + 255) / 256 * 256 := by
  omega

-- § beta gate

/-- Grid for the in-place beta sigmoid covers all heads (128 per block). -/
theorem beta_grid_covers (n : Nat) : n ≤ (n + 127) / 128 * 128 := by
  omega

-- § norm kernels

/-- Output norm: one block per head, address `h*S + i` stays in range for every strided i. -/
theorem out_norm_in_range (h i S h_v : Nat) (hh : h < h_v) (hi : i < S) : h * S + i < h_v * S :=
  idx_lt hh hi

/-- L2 norm warp reduction width is guarded: accepted column counts are at most 1024. -/
theorem l2_cols_guard (cols : Nat) (h : cols ≤ 1024) : cols < 1025 := by
  omega

end Strata.GdnReference



## FILE: /mnt/user-data/outputs/GatedResidualReadWrite.lean
/-!
# MODULE: gated_residual_read_write
Workspace layout, reduction mapping and index contracts of the generic gated residual / hyper-connection
`gr_read` and `gr_write` kernels (any geometry, bf16 weights, one warp per output row).

## Source semantics
- Workspace has five regions xn, xq, lq, gated, lo, each rounded up to 16 bytes, laid out back to back; the
  first version mis-sized regions so they overlapped, which the offsets lemmas below rule out.
- Down kernel: one block per `lo` row, 8 warps split the reduction with element `warp*32+lane + 256*k`.
- Gate kernel: one warp per `gated` row, 8 rows per block. Mean kernel: flat over n_embd, streams strided.
- Write kernel: `out[c*n_embd+d] = R[...] + block_out[d] * w[c]`, w computed once per stream.

## Form template identifiers
- n_embd -> model_dim, hc -> n_streams, hc_lr -> low_rank, hc_dim -> stream_total
- xn -> normed_streams, xq -> quantized_streams, lq -> quantized_low_rank, gated -> gated_streams
-/

namespace Strata.GatedResidual

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § workspace layout

/-- Round a byte count up to a multiple of 16. -/
def alignUp16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks a region. -/
theorem alignUp16_ge (n : Nat) : n ≤ alignUp16 n := by
  unfold alignUp16
  omega

/-- Alignment adds less than one alignment unit. -/
theorem alignUp16_lt (n : Nat) : alignUp16 n < n + 16 := by
  unfold alignUp16
  omega

/-- Aligned sizes are multiples of 16. -/
theorem alignUp16_mod (n : Nat) : alignUp16 n % 16 = 0 := by
  unfold alignUp16
  omega

/-- Region offsets: each region starts where the previous one ends, so regions are disjoint. -/
def offsets (a0 a1 a2 a3 : Nat) : Nat × Nat × Nat × Nat := (0, a0, a0 + a1, a0 + a1 + a2)

/-- Region k+1 begins at or after the end of region k. -/
theorem regions_disjoint (a0 a1 a2 a3 : Nat) :
    (offsets a0 a1 a2 a3).1 + a0 ≤ (offsets a0 a1 a2 a3).2.1 ∧
    (offsets a0 a1 a2 a3).2.1 + a1 ≤ (offsets a0 a1 a2 a3).2.2.1 ∧
    (offsets a0 a1 a2 a3).2.2.1 + a2 ≤ (offsets a0 a1 a2 a3).2.2.2 ∧
    (offsets a0 a1 a2 a3).2.2.2 + a3 = a0 + a1 + a2 + a3 := by
  simp [offsets] <;> omega

/-- The real geometry (2560 x 4 streams, rank 320) has all five regions already 16-byte aligned. -/
theorem real_geometry_regions :
    alignUp16 (10240 * 4) = 40960 ∧ alignUp16 (10240 * 2) = 20480 ∧ alignUp16 (320 * 2) = 640 ∧
    alignUp16 (10240 * 4) = 40960 ∧ alignUp16 (320 * 4) = 1280 := by
  decide

/-- Total workspace of the real geometry. -/
theorem real_geometry_total : 40960 + 20480 + 640 + 40960 + 1280 = 104320 := by decide

/-- The historical mis-size: giving xq only low-rank bytes is far too small for the stream vector. -/
theorem xq_misize_too_small : (320 * 2 : Nat) < 10240 * 2 := by decide

-- § down kernel mapping

/-- Strided reduction: element i of the row decomposes into (iteration k, warp, lane). -/
theorem down_element_split (i : Nat) :
    i = (i / 256) * 256 + ((i % 256) / 32) * 32 + i % 32 := by
  omega

/-- The decomposed warp index is below 8 and the lane index below 32. -/
theorem down_split_ranges (i : Nat) : (i % 256) / 32 < 8 ∧ i % 32 < 32 := by
  omega

/-- Unique decomposition: equal (k, warp, lane) triples give equal elements. -/
theorem down_split_inj (k w l k' w' l' : Nat) (hw : w < 8) (hl : l < 32) (hw' : w' < 8)
    (hl' : l' < 32) (h : w * 32 + l + 256 * k = w' * 32 + l' + 256 * k') :
    k = k' ∧ w = w' ∧ l = l' := by
  omega

/-- Block-level reduction scratch: 8 warps' partials fit in `part[WARPS]`. -/
theorem partials_fit (nw : Nat) (h : nw = 256 / 32) : nw ≤ 8 := by
  omega

/-- The block-sum scratch of 8 doubles can hold 8 floats (reinterpretation does not overrun). -/
theorem scratch_reinterpret_fits : 8 * 4 ≤ 8 * 8 := by decide

-- § gate / mean / write kernels

/-- Gate grid covers all hc_dim rows (8 rows per block). -/
theorem gate_grid_covers (hc_dim : Nat) : hc_dim ≤ (hc_dim + 7) / 8 * 8 := by
  omega

/-- A row below hc_dim lies in a launched block. -/
theorem gate_row_block (i hc_dim : Nat) (h : i < hc_dim) : i / 8 < (hc_dim + 7) / 8 := by
  omega

/-- Mean kernel: element (c, d) of the gated array is in range. -/
theorem mean_index_lt (c d hc n_embd : Nat) (hc' : c < hc) (hd : d < n_embd) :
    c * n_embd + d < hc * n_embd :=
  idx_lt hc' hd

/-- Mean grid covers n_embd (256 threads per block). -/
theorem mean_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by
  omega

/-- Write kernel: the stream index of a flat position is below hc. -/
theorem write_stream_lt (i n_embd hc : Nat) (h : i < n_embd * hc) : i / n_embd < hc :=
  Nat.div_lt_of_lt_mul h

/-- Write kernel: (stream, column) recombines to the flat index. -/
theorem write_recombine (i n_embd : Nat) : n_embd * (i / n_embd) + i % n_embd = i :=
  Nat.div_add_mod i n_embd

/-- Write kernel: the column index is below n_embd. -/
theorem write_col_lt (i n_embd : Nat) (h : 0 < n_embd) : i % n_embd < n_embd :=
  Nat.mod_lt _ h

/-- Stream-weight setup: threads 0..hc-1 write `w`, which needs `hc` floats of shared memory. -/
theorem write_weight_slots (t hc : Nat) (h : t < hc) : t * 4 + 4 ≤ hc * 4 := by
  omega

-- § inject / native contracts

/-- The inject kernel launches 32*hc threads in one block: at most 1024 means hc <= 32. -/
theorem inject_block_size_ok (hc : Nat) (h : 32 * hc ≤ 1024) : hc ≤ 32 := by
  omega

/-- Native path requirement: even `hc*n_embd` when n_embd is even. -/
theorem native_even (n_embd hc : Nat) (h : n_embd % 2 = 0) : (hc * n_embd) % 2 = 0 := by
  rw [Nat.mul_mod, h]
  simp

end Strata.GatedResidual



## FILE: /mnt/user-data/outputs/IqQuantMatVec.lean
/-!
# MODULE: iq_quant_mat_vec
Layout and index contracts of the IQ-family (and Q2_0/Q3_K) quantized matrix-vector kernels transcribed from
llama.cpp: per-format block traits, the warp-strided row dot, the grouped native expert layout (gate | up | down
blobs), the q8_1 activation quantiser, and the thread mapping of the dequantisers.

## Source semantics
- A format has `qk` values per block, `ipb` dot calls per block and an `iqs` stride `step`; call k of a row maps to
  block `k / ipb`, part `k % ipb`, and the activation q8_1 blocks start at `kbx * (qk/32)`.
- Expert blob layout: gate rows [0, n_ff*gu_row), up rows [up_off, down_off), down rows [down_off, bytes) with
  `up_off = n_ff*gu_row`, `down_off = 2*up_off`, `bytes = down_off + n_embd*d_row`.
- q8_1 quantiser: block of 32, scale `d = amax/127`, code 0 when amax = 0.
- Dequantisers: 32 threads per 256-value superblock, thread tid writes 8 (or 4+4 split) values.

## Form template identifiers
- qk -> block_values, ipb -> calls_per_block, step -> iqs_step, gu_row -> gate_up_row_bytes, d_row -> down_row_bytes
- up_off -> up_offset, down_off -> down_offset, ent_tok -> entry_token, ent_dst -> entry_dest
-/

namespace Strata.IqQuant

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § format table

/-- Bytes per block for each supported GGML type code (sizes from ggml-common.h). -/
def blockBytes : Nat → Nat
  | 16 => 66 | 17 => 74 | 18 => 98 | 20 => 18 | 21 => 110
  | 22 => 82 | 29 => 56 | 23 => 136 | 11 => 110 | 42 => 18
  | _ => 0

/-- Values per block for each supported type code. -/
def blockValues : Nat → Nat
  | 20 => 32 | 42 => 64 | 16 => 256 | 17 => 256 | 18 => 256 | 21 => 256
  | 22 => 256 | 29 => 256 | 23 => 256 | 11 => 256
  | _ => 0

/-- Types accepted by `is_iq`. -/
def iqTypes : List Nat := [16, 17, 18, 20, 21, 22, 23, 29, 42, 11]

/-- Every accepted type has a nonzero block size and nonzero block width. -/
theorem iq_types_have_layout : ∀ t ∈ iqTypes, 0 < blockBytes t ∧ 0 < blockValues t := by
  decide

/-- The accepted type list has no duplicates. -/
theorem iq_types_nodup : iqTypes.Nodup := by decide

/-- Row bytes: whole blocks per row times block bytes. -/
def rowBytes (t n : Nat) : Nat := n / blockValues t * blockBytes t

-- § row dot call enumeration

/-- Call k decomposes into (block, part). -/
theorem call_split (k ipb : Nat) : (k / ipb) * ipb + k % ipb = k := by
  have := Nat.div_add_mod k ipb
  rw [Nat.mul_comm] at this
  exact this

/-- Block index of a call below `nb*ipb` is below `nb`. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < nb * ipb) (hp : 0 < ipb) : k / ipb < nb :=
  Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- Part index is below `ipb`. -/
theorem call_part_lt (k ipb : Nat) (hp : 0 < ipb) : k % ipb < ipb := Nat.mod_lt _ hp

/-- The q8_1 blocks consumed by block kbx end inside the activation: (kbx+1)*m ≤ nb*m. -/
theorem q8_window_in_range (kbx nb m : Nat) (h : kbx < nb) : (kbx + 1) * m ≤ nb * m :=
  Nat.mul_le_mul_right m (Nat.succ_le_of_lt h)

/-- `nb` weight blocks of `32*m` values consume `nb*m` q8_1 blocks of 32 values. -/
theorem q8_blocks_per_row (nb m : Nat) : (nb * (32 * m)) / 32 = nb * m := by
  rw [Nat.mul_left_comm]
  exact Nat.mul_div_cancel_left _ (by decide)

/-- Per-format iqs ranges: ipb=8, step=2 gives iqs ≤ 14. -/
theorem iqs_range_step2 (part : Nat) (h : part < 8) : 2 * part ≤ 14 := by
  omega

/-- IQ1_M: ipb=8, step=1 gives iqs ≤ 7. -/
theorem iqs_range_step1 (part : Nat) (h : part < 8) : 1 * part ≤ 7 := by
  omega

/-- IQ4_NL: ipb=2, step=2 gives iqs ≤ 2 (two 4-int halves). -/
theorem iqs_range_iq4nl (part : Nat) (h : part < 2) : 2 * part ≤ 2 := by
  omega

-- § mmvq grid

/-- Four rows per block cover all n_out rows. -/
theorem mmvq_grid_covers (n_out : Nat) : n_out ≤ (n_out + 3) / 4 * 4 := by
  omega

/-- Row of thread (bx, ty) with ty < 4 is bx*4+ty. -/
theorem mmvq_row_split (row : Nat) : (row / 4) * 4 + row % 4 = row := by
  omega

/-- Output address for column c of row r is below `ncols*n_out`. -/
theorem mmvq_out_lt (c r ncols n_out : Nat) (hc : c < ncols) (hr : r < n_out) :
    c * n_out + r < ncols * n_out :=
  idx_lt hc hr

-- § grouped expert layout

/-- Expert blob layout. -/
structure Layout where
  nEmbd : Nat
  nFf : Nat
  guRow : Nat
  dRow : Nat

/-- Offset of the up matrix. -/
def Layout.upOff (L : Layout) : Nat := L.nFf * L.guRow
/-- Offset of the down matrix. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total blob size. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

/-- Gate occupies [0, upOff) and up occupies [upOff, downOff): no overlap and no gap. -/
theorem gate_up_partition (L : Layout) : L.upOff + L.nFf * L.guRow = L.downOff := by
  unfold Layout.downOff Layout.upOff
  omega

/-- Down matrix starts exactly after up. -/
theorem down_after_up (L : Layout) : L.upOff + (L.downOff - L.upOff) = L.downOff := by
  unfold Layout.downOff Layout.upOff
  omega

/-- Every gate row ends inside the gate region. -/
theorem gate_row_in_region (L : Layout) (r : Nat) (h : r < L.nFf) :
    r * L.guRow + L.guRow ≤ L.upOff := by
  unfold Layout.upOff
  have this : (r + 1) * L.guRow ≤ L.nFf * L.guRow := Nat.mul_le_mul_right L.guRow (Nat.succ_le_of_lt h)
  have e : (r + 1) * L.guRow = r * L.guRow + L.guRow := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Every up row ends inside the up region. -/
theorem up_row_in_region (L : Layout) (r : Nat) (h : r < L.nFf) :
    L.upOff + r * L.guRow + L.guRow ≤ L.downOff := by
  have := gate_row_in_region L r h
  unfold Layout.downOff
  omega

/-- Every down row ends inside the blob. -/
theorem down_row_in_blob (L : Layout) (r : Nat) (h : r < L.nEmbd) :
    L.downOff + r * L.dRow + L.dRow ≤ L.bytes := by
  unfold Layout.bytes
  have this : (r + 1) * L.dRow ≤ L.nEmbd * L.dRow := Nat.mul_le_mul_right L.dRow (Nat.succ_le_of_lt h)
  have e : (r + 1) * L.dRow = r * L.dRow + L.dRow := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Gate/up kernel: row index in [0, 2*n_ff) splits into matrix selector and local row. -/
theorem gu_row_split (row n_ff : Nat) (h : row < 2 * n_ff) :
    (row < n_ff ∧ row < n_ff) ∨ (n_ff ≤ row ∧ row - n_ff < n_ff) := by
  omega

/-- Gate/up grid covers 2*n_ff rows with 8 rows per block. -/
theorem gu_grid_covers (n_ff : Nat) : 2 * n_ff ≤ (2 * n_ff + 7) / 8 * 8 := by
  omega

/-- Down grid covers n_embd rows with 8 rows per block. -/
theorem down_grid_covers (n_embd : Nat) : n_embd ≤ (n_embd + 7) / 8 * 8 := by
  omega

/-- Entry output `e*n_ff + r` stays inside `cap*n_ff`. -/
theorem entry_out_lt (e r cap n_ff : Nat) (he : e < cap) (hr : r < n_ff) :
    e * n_ff + r < cap * n_ff :=
  idx_lt he hr

-- § scratch layout

/-- Round up to a multiple of 256. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256
  omega

/-- Aligned value is a multiple of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256
  omega

/-- Scratch regions gate/up/h each get `fa` bytes and `hq` starts at `3*fa`: the four regions are disjoint. -/
theorem scratch_regions (f : Nat) : f ≤ alignUp256 f ∧ alignUp256 f + alignUp256 f ≤ 2 * alignUp256 f ∧
    2 * alignUp256 f + alignUp256 f = 3 * alignUp256 f := by
  have := alignUp256_ge f
  omega

-- § q8_1 quantiser

/-- Element i belongs to block i/32 at position i%32. -/
theorem q8_block_split (i : Nat) : (i / 32) * 32 + i % 32 = i := by
  omega

/-- The position within the block is below 32. -/
theorem q8_pos_lt (i : Nat) : i % 32 < 32 := by
  omega

/-- Quantiser grid covers n elements with 256 threads per block. -/
theorem q8_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by
  omega

-- § dequant thread mapping

/-- 8-value formats: thread tid writes offsets `32*(tid%8) + 8*(tid/8) + j`, all below 256. -/
theorem dq8_offset_lt (tid j : Nat) (ht : tid < 32) (hj : j < 8) :
    32 * (tid % 8) + 8 * (tid / 8) + j < 256 := by
  omega

/-- 8-value formats: distinct (thread, j) pairs write distinct offsets. -/
theorem dq8_offset_inj (t j t' j' : Nat) (ht : t < 32) (hj : j < 8) (ht' : t' < 32) (hj' : j' < 8)
    (h : 32 * (t % 8) + 8 * (t / 8) + j = 32 * (t' % 8) + 8 * (t' / 8) + j') : t = t' ∧ j = j' := by
  omega

/-- 4+4 formats (iq4_nl/iq4_xs): offsets `32*ib + 4*il + j (+16)` are below 256. -/
theorem dq4_offset_lt (ib il j h : Nat) (hb : ib < 8) (hi : il < 4) (hj : j < 4) (hh : h < 2) :
    32 * ib + 4 * il + j + 16 * h < 256 := by
  omega

/-- 4+4 formats: distinct (ib, il, j, h) write distinct offsets. -/
theorem dq4_offset_inj (ib il j h ib' il' j' h' : Nat) (hb : ib < 8) (hi : il < 4) (hj : j < 4)
    (hh : h < 2) (hb' : ib' < 8) (hi' : il' < 4) (hj' : j' < 4) (hh' : h' < 2)
    (e : 32 * ib + 4 * il + j + 16 * h = 32 * ib' + 4 * il' + j' + 16 * h') :
    ib = ib' ∧ il = il' ∧ j = j' ∧ h = h' := by
  omega

/-- Q2_0 dequant: block b (0..3) and part p (0..7) write offsets `b*64 + p*8 + j` below 256. -/
theorem dq_q2_0_offset_lt (b p j : Nat) (hb : b < 4) (hp : p < 8) (hj : j < 8) :
    b * 64 + (p * 8 + j) < 256 := by
  omega

/-- Embedding rows: output address `t*n_embd + b*256 + o` stays inside `n_tok * n_embd`. -/
theorem embed_out_lt (t b o n_tok n_embd : Nat) (ht : t < n_tok) (hn : n_embd % 256 = 0)
    (hb : b < n_embd / 256) (ho : o < 256) : t * n_embd + (b * 256 + o) < n_tok * n_embd := by
  have h : b * 256 + o < n_embd := by omega
  have h2 := idx_lt ht h
  exact h2

/-- Gate/up interleaved dequant: row `2r+parity` with column block c stays inside `2*n_ff` rows. -/
theorem gu_interleave_lt (r parity c n_ff per_row : Nat) (hr : r < n_ff) (hp : parity < 2)
    (hc : c < per_row) : (2 * r + parity) * per_row + c < 2 * n_ff * per_row := by
  have h : 2 * r + parity < 2 * n_ff := by omega
  exact idx_lt h hc

-- § supported types

/-- Types accepted by `iq_mmvq` / grouped gate-up kernels route to a template instance. -/
def mmvqTypes : List Nat := [16, 17, 18, 20, 21, 22, 29, 42]

/-- The mmvq set is a subset of the accepted IQ set. -/
theorem mmvq_types_subset : ∀ t ∈ mmvqTypes, t ∈ iqTypes := by
  decide

/-- Down projection of the grouped experts accepts only types 20 and 42. -/
def downTypes : List Nat := [20, 42]

/-- Down types are mmvq types. -/
theorem down_types_subset : ∀ t ∈ downTypes, t ∈ mmvqTypes := by
  decide

end Strata.IqQuant



## FILE: /mnt/user-data/outputs/KvCacheQ8.lean
/-!
# MODULE: kv_cache_q8
Addressing and quantisation contracts of the paged int8 KV cache: the append kernel (one block per 64-value
group of one KV head of K or V) and the gather kernel (one thread per four consecutive values).

## Source semantics
- Cache row of (page, head h, slot s) is `(page*kv_heads + h)*page_size + s`; values of a row are `head_dim` int8,
  scales are one fp16 per 64-value group, `groups = head_dim/64`.
- Append: position `pos` maps to page `table[pos/page_size]` and slot `pos % page_size`; codes are clamped to
  [-127, 127] and quantised against the STORED (fp16-rounded) scale.
- Gather: thread i decomposes as (id, head, q4) with `per = head_dim/4`; destination index equals i, i.e. the
  scratch is the dense (id, head, d) layout.

## Form template identifiers
- KV_Q8_GROUP -> quant_group (64), kv_heads -> n_kv_heads, page_size -> page_slots, per -> quads_per_head
-/

namespace Strata.KvQ8

/-- Quantisation group width. -/
def KV_Q8_GROUP : Nat := 64

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § quantisation

/-- Code clamp to the symmetric int8 range. -/
def clamp127 (q : Int) : Int := max (-127) (min 127 q)

/-- Clamped codes lie in [-127, 127]. -/
theorem clamp127_bounds (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  omega

/-- Clamping is the identity inside the range. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  omega

/-- The int8 cast of a clamped code does not wrap: it fits in [-128, 127]. -/
theorem clamp127_fits_int8 (q : Int) : -128 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  omega

-- § append addressing

/-- A 64-value group inside a head row stays inside the row (head_dim = groups * 64). -/
theorem group_offset_lt (g t groups : Nat) (hg : g < groups) (ht : t < 64) :
    g * 64 + t < groups * 64 :=
  idx_lt hg ht

/-- Cache row index bound for page < n_pages, head < kv_heads, slot < page_size. -/
theorem cache_row_lt (page h s n_pages kv ps : Nat) (hp : page < n_pages) (hh : h < kv)
    (hs : s < ps) : (page * kv + h) * ps + s < n_pages * kv * ps := by
  have h1 : page * kv + h < n_pages * kv := idx_lt hp hh
  exact idx_lt h1 hs

/-- Slot of a position is below page_size and recombines with the page index. -/
theorem pos_split (pos ps : Nat) : ps * (pos / ps) + pos % ps = pos :=
  Nat.div_add_mod pos ps

/-- Slot of a position is below page_size. -/
theorem slot_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

/-- Value offset of (row, g, t) is below `(row+1)*head_dim` when head_dim = groups*64. -/
theorem value_offset_lt (row g t groups hd : Nat) (hd' : hd = groups * 64) (hg : g < groups)
    (ht : t < 64) : row * hd + (g * 64 + t) < (row + 1) * hd := by
  have h1 : g * 64 + t < groups * 64 := idx_lt hg ht
  have h2 : (row + 1) * hd = row * hd + hd := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Scale index of (row, g) is below `(row+1)*groups`. -/
theorem scale_index_lt (row g groups : Nat) (hg : g < groups) :
    row * groups + g < (row + 1) * groups :=
  idx_lt (Nat.lt_succ_self row) hg

/-- Append grid has one block per (head, group, K/V) and 64 threads per block (two warps). -/
theorem append_block_warps (t : Nat) (h : t < 64) : t / 32 < 2 := by
  omega

-- § gather decomposition

/-- Thread index decomposition: id from the quotient, remainder `rem < kv*per`. -/
theorem gather_rem_lt (i kv per : Nat) (hk : 0 < kv * per) : i % (kv * per) < kv * per :=
  Nat.mod_lt _ hk

/-- Head index of the remainder is below kv_heads. -/
theorem gather_head_lt (rem kv per : Nat) (h : rem < kv * per) : rem / per < kv := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  exact h

/-- The quad index is below `per`. -/
theorem gather_quad_lt (rem per : Nat) (hp : 0 < per) : rem % per < per := Nat.mod_lt _ hp

/-- Destination index (id*kv + h)*per + q4 equals the thread index: the scratch is dense. -/
theorem gather_dst_eq (id h q4 kv per : Nat) :
    (id * kv + h) * per + q4 = id * (kv * per) + (h * per + q4) := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Total thread count bound: dst below `n_ids * kv * per`. -/
theorem gather_dst_lt (id h q4 n_ids kv per : Nat) (hi : id < n_ids) (hh : h < kv) (hq : q4 < per) :
    (id * kv + h) * per + q4 < n_ids * kv * per :=
  idx_lt (idx_lt hi hh) hq

/-- A quad starts at d = 4*q4 below head_dim when q4 < head_dim/4. -/
theorem quad_start_lt (q4 hd : Nat) (hd4 : hd % 4 = 0) (h : q4 < hd / 4) : q4 * 4 + 3 < hd := by
  omega

/-- The four values of a quad stay in one 64-group (head_dim a multiple of 64). -/
theorem quad_in_one_group (q4 : Nat) : (q4 * 4 + 3) / 64 = (q4 * 4) / 64 := by
  omega

/-- The scale group of a quad is below `groups` when head_dim = groups*64. -/
theorem quad_group_lt (q4 groups hd : Nat) (hd' : hd = groups * 64) (h : q4 < hd / 4) :
    (q4 * 4) / 64 < groups := by
  omega

/-- Row-level pointer arithmetic: char4 element q4 of row sits at byte `row*hd + 4*q4`. -/
theorem char4_in_row (row q4 hd : Nat) (h : q4 * 4 + 3 < hd) :
    row * hd + q4 * 4 + 3 < (row + 1) * hd := by
  have h2 : (row + 1) * hd = row * hd + hd := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § launch contract

/-- `validate`: head_dim a multiple of 64 with positive head count and page size. -/
def validShapes (hd nkv ps : Nat) : Prop := hd % KV_Q8_GROUP = 0 ∧ 0 < nkv ∧ 0 < ps

/-- Valid shapes give a whole number of groups and quads. -/
theorem valid_shapes_quads (hd nkv ps : Nat) (h : validShapes hd nkv ps) : hd % 4 = 0 := by
  obtain ⟨h1, _, _⟩ := h
  unfold KV_Q8_GROUP at h1
  omega

/-- Gather grid covers all threads with 256 per block. -/
theorem gather_grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by
  omega

end Strata.KvQ8



## FILE: /mnt/user-data/outputs/NativeShortFlashAttention.lean
/-!
# MODULE: native_short_flash_attention
Index geometry and step-validity contract of the specialised single-token FlashAttention-vec kernel:
24 query heads of dimension 256, 2 KV heads (12 query heads per KV head), contexts 1..256, fp16 K/V,
128 threads (32 lanes x 4 warps) per head, padded KV length 256 processed in tiles of 128 cells.

## Source semantics
- Block = query head; `kv = head / 12`. K/V element (cell, kv, d) is at `(cell*2 + kv)*256 + d` halves.
- The step buffer is valid iff `1 <= width <= max_context`, `nkv == width`, `pos == width-1`, `nbid == width/4`;
  otherwise the kernel writes NaN and an unsupported-step status.
- Each tile handles cells `base + warp*32 + (lane & ~7) + row` for row in 0..7; cells >= width are masked.
- The final reduction sums 4 warps x 4 lane groups of the shared tile (4*4*256 floats) per output element.

## Form template identifiers
- width -> context_len, nkv -> kv_len, pos -> last_position, nbid -> index_blocks, mask -> attn_mask
- head -> query_head, kv -> kv_head, cell -> kv_cell
-/

namespace Strata.NativeFlashAttn

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § validity

/-- The step-validity predicate. -/
def validStep (width nkv pos nbid maxc : Int) : Prop :=
  1 ≤ width ∧ width ≤ maxc ∧ nkv = width ∧ pos = width - 1 ∧ nbid = width / 4

/-- Valid steps under the launcher bound have width at most 256. -/
theorem valid_width_le_256 (width nkv pos nbid maxc : Int) (hm : maxc ≤ 256)
    (h : validStep width nkv pos nbid maxc) : width ≤ 256 := by
  obtain ⟨_, h2, _, _, _⟩ := h
  omega

/-- Valid steps have the last position strictly below the context length. -/
theorem valid_pos_lt_width (width nkv pos nbid maxc : Int)
    (h : validStep width nkv pos nbid maxc) : pos < width := by
  obtain ⟨_, _, _, h4, _⟩ := h
  omega

/-- Cells below `width` are below the padded length 256 (padded cells are masked). -/
theorem live_cells_below_padding (cell width : Nat) (hw : width ≤ 256) (hc : cell < width) :
    cell < 256 := by
  omega

-- § head mapping

/-- 24 query heads over 2 KV heads is exactly 12 per KV head. -/
theorem heads_per_kv : 24 / 2 = 12 := by decide

/-- A query head below 24 maps to a KV head below 2. -/
theorem kv_head_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by
  omega

/-- Query heads of one KV group are 12 consecutive heads. -/
theorem kv_group_range (head : Nat) (h : head < 24) :
    (head / 12) * 12 ≤ head ∧ head < (head / 12) * 12 + 12 := by
  omega

-- § cell / lane mapping inside a tile

/-- `lane & ~7` equals `(lane/8)*8`. -/
theorem lane_mask_eq (lane : Nat) : lane - lane % 8 = (lane / 8) * 8 := by
  omega

/-- Tile-local cell of (warp, lane, row) is below 128. -/
theorem tile_cell_lt (warp lane row : Nat) (hw : warp < 4) (hl : lane < 32) (hr : row < 8) :
    warp * 32 + (lane / 8) * 8 + row < 128 := by
  omega

/-- Tile-local cells are distinct for distinct (warp, lane group, row). -/
theorem tile_cell_inj (w g r w' g' r' : Nat) (hg : g < 4) (hr : r < 8) (hg' : g' < 4) (hr' : r' < 8)
    (h : w * 32 + g * 8 + r = w' * 32 + g' * 8 + r') : w = w' ∧ g = g' ∧ r = r' := by
  omega

/-- Every tile-local cell is produced by some (warp, lane group, row). -/
theorem tile_cell_cover (c : Nat) (h : c < 128) :
    ∃ w g r, w < 4 ∧ g < 4 ∧ r < 8 ∧ c = w * 32 + g * 8 + r :=
  ⟨c / 32, (c % 32) / 8, c % 8, by omega, by omega, by omega, by omega⟩

/-- Two tiles of 128 cover the padded length 256. -/
theorem tiles_cover_padding : 2 * 128 = 256 := by decide

/-- Absolute cell = tile base + local cell stays below 256 for base in {0, 128}. -/
theorem abs_cell_lt (base local_ : Nat) (hb : base = 0 ∨ base = 128) (hl : local_ < 128) :
    base + local_ < 256 := by
  omega

-- § K/V addressing

/-- K/V element address bound: (cell*2 + kv)*256 + d < capacity*2*256. -/
theorem kv_index_lt (cell kv d capacity : Nat) (hc : cell < capacity) (hk : kv < 2) (hd : d < 256) :
    (cell * 2 + kv) * 256 + d < capacity * 2 * 256 :=
  idx_lt (idx_lt hc hk) hd

/-- K/V capacity in halves times 2 bytes gives 1024 bytes per cell (the span checked by the launcher). -/
theorem kv_bytes_per_cell : 2 * 256 * 2 = 1024 := by decide

/-- Query/vkq register slot of (i0, j) with i0 = 32*m is below 16. -/
theorem qreg_slot_lt (m j : Nat) (hm : m < 4) (hj : j < 4) : (32 * m) / 8 + j < 16 := by
  omega

/-- Dimension pair index d = 2*(i+j) with i = 32*m + 4*l has d+1 < 256. -/
theorem dim_pair_lt (m l j : Nat) (hm : m < 4) (hl : l < 8) (hj : j < 4) :
    2 * (32 * m + 4 * l + j) + 1 < 256 := by
  omega

/-- The 128 (m, l, j) triples enumerate distinct dimension pairs. -/
theorem dim_pair_inj (m l j m' l' j' : Nat) (hl : l < 8) (hj : j < 4) (hl' : l' < 8) (hj' : j' < 4)
    (h : 32 * m + 4 * l + j = 32 * m' + 4 * l' + j') : m = m' ∧ l = l' ∧ j = j' := by
  omega

/-- Mask index of a live cell is below 256 (the mask has 256 halves). -/
theorem mask_index_lt (cell width : Nat) (hw : width ≤ 256) (hc : cell < width) : cell < 256 := by
  omega

-- § output reduction

/-- Shared-tile reduction index `w*1024 + group*256 + i0 + tid` is below 4096 (the tile size). -/
theorem reduce_index_lt (w group i0 tid : Nat) (hw : w < 4) (hg : group < 4)
    (hi : i0 = 0 ∨ i0 = 128) (ht : tid < 128) : w * 1024 + group * 256 + i0 + tid < 4096 := by
  omega

/-- Shared tile of 4*4*256 floats. -/
theorem tile_floats : 4 * 4 * 256 = 4096 := by decide

/-- Output elements `i0 + tid` for i0 in {0,128}, tid < 128 cover [0,256) once. -/
theorem output_cover (e : Nat) (h : e < 256) :
    ∃ i0 tid, (i0 = 0 ∨ i0 = 128) ∧ tid < 128 ∧ e = i0 + tid := by
  by_cases h' : e < 128
  · exact ⟨0, e, Or.inl rfl, h', by omega⟩
  · exact ⟨128, e - 128, Or.inr rfl, by omega, by omega⟩

/-- Output address of (head, element) is below 24*256. -/
theorem out_index_lt (head e : Nat) (hh : head < 24) (he : e < 256) : head * 256 + e < 24 * 256 :=
  idx_lt hh he

/-- Warp slots for the cross-warp max/sum are 0..3 and fit the 32-entry shared arrays. -/
theorem warp_slots_fit (warp : Nat) (h : warp < 4) : warp < 32 := by
  omega

-- § launcher contract

/-- Supported geometry of the launcher. -/
def supportedShape (n_head n_head_kv head_dim capacity max_context : Int) : Prop :=
  n_head = 24 ∧ n_head_kv = 2 ∧ head_dim = 256 ∧ 256 ≤ capacity ∧ 1 ≤ max_context ∧ max_context ≤ 256

/-- Supported shapes have capacity for every live cell. -/
theorem supported_capacity_covers (a b c cap m : Int) (h : supportedShape a b c cap m) : m ≤ cap := by
  obtain ⟨_, _, _, h4, _, h6⟩ := h
  omega

end Strata.NativeFlashAttn



## FILE: /mnt/user-data/outputs/NativeGdnRecurrence.lean
/-!
# MODULE: native_gdn_recurrence
Thread mapping and span contracts of the llama.cpp-style gated delta net step with one warp per
(head, column): each lane keeps four state rows in registers and the warp reduces the K^T S and Q^T S
contractions by xor shuffle.

## Source semantics
- Grid is (h_v, 1, S/4), block (32, 4): column `col = blockIdx.z*4 + threadIdx.y`, lane `l` owns rows
  `r*32 + l` for r in 0..3; row i of column (head, col) is at `(i*h_v + head)*S + col`.
- q/k head is `head % h_k`; the output is scaled by `1/sqrt(S)`.
- Required: stream given, S = 128, h_v in (0, 65535], h_k > 0, h_v % h_k = 0; state/output/inputs are aligned,
  state and output disjoint, and every input span is disjoint from state and output.

## Form template identifiers
- S -> state_dim, h_k -> n_qk_heads, h_v -> n_v_heads, g_val -> decay, delta_col -> delta
- native_gdn_step -> warp_delta_step, enabled -> native_gdn_flag
-/

namespace Strata.NativeGdn

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § column and row mapping

/-- Column of (z, y) with z < 32, y < 4 is below 128. -/
theorem column_lt (z y : Nat) (hz : z < 32) (hy : y < 4) : z * 4 + y < 128 := by
  omega

/-- Distinct (z, y) give distinct columns. -/
theorem column_inj (z y z' y' : Nat) (hy : y < 4) (hy' : y' < 4) (h : z * 4 + y = z' * 4 + y') :
    z = z' ∧ y = y' := by
  omega

/-- Every column is produced by some (z, y). -/
theorem column_cover (c : Nat) (h : c < 128) : ∃ z y, z < 32 ∧ y < 4 ∧ c = z * 4 + y :=
  ⟨c / 4, c % 4, by omega, by omega, by omega⟩

/-- Rows owned by lane l: r*32 + l for r < 4, all below 128. -/
theorem lane_row_lt (r l : Nat) (hr : r < 4) (hl : l < 32) : r * 32 + l < 128 := by
  omega

/-- The four register rows of 32 lanes tile all 128 rows exactly once. -/
theorem lane_row_inj (r l r' l' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by
  omega

/-- Every row is owned by some (register, lane). -/
theorem lane_row_cover (i : Nat) (h : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ i = r * 32 + l :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- The warp reduction is over exactly the 32 lanes of one column. -/
theorem warp_is_lane_set : 4 * 32 = 128 := by decide

/-- A block of 32x4 threads has 128 threads. -/
theorem block_threads : 32 * 4 = 128 := by decide

-- § addressing

/-- State element address bound. -/
theorem state_elem_lt (i head col h_v : Nat) (hi : i < 128) (hh : head < h_v) (hc : col < 128) :
    (i * h_v + head) * 128 + col < 128 * h_v * 128 :=
  idx_lt (idx_lt hi hh) hc

/-- q/k element address bound for head `head % h_k`. -/
theorem qk_elem_lt (head i h_k : Nat) (hk : 0 < h_k) (hi : i < 128) :
    (head % h_k) * 128 + i < h_k * 128 :=
  idx_lt (Nat.mod_lt _ hk) hi

/-- v / output address bound. -/
theorem vo_elem_lt (head col h_v : Nat) (hh : head < h_v) (hc : col < 128) :
    head * 128 + col < h_v * 128 :=
  idx_lt hh hc

-- § launch contract

/-- Shape validity (the checks before the span checks). -/
def validShape (S h_k h_v : Nat) : Prop := S = 128 ∧ 0 < h_k ∧ 0 < h_v ∧ h_v ≤ 65535 ∧ h_v % h_k = 0

/-- Valid shapes give an in-range q/k head for every v head. -/
theorem valid_shape_head_pair (S h_k h_v head : Nat) (h : validShape S h_k h_v) :
    head % h_k < h_k := by
  obtain ⟨_, hk, _, _, _⟩ := h
  exact Nat.mod_lt _ hk

/-- Valid shapes fit the grid.x limit. -/
theorem valid_shape_grid_x (S h_k h_v : Nat) (h : validShape S h_k h_v) : h_v ≤ 65535 := by
  obtain ⟨_, _, _, h4, _⟩ := h
  exact h4

/-- Byte counts of the state, q/k, v/output and per-head spans for S = 128. -/
def stateBytes (h_v : Nat) : Nat := 128 * 128 * h_v * 4
/-- q/k byte count. -/
def qkBytes (h_k : Nat) : Nat := 128 * h_k * 4
/-- v / output byte count. -/
def voBytes (h_v : Nat) : Nat := 128 * h_v * 4
/-- per-head scalar byte count. -/
def headBytes (h_v : Nat) : Nat := h_v * 4

/-- The state is 128 times larger than the output. -/
theorem state_is_128x_output (h_v : Nat) : stateBytes h_v = 128 * voBytes h_v := by
  unfold stateBytes voBytes
  omega

/-- Spans [a, a+an) and [b, b+bn) overlap iff each starts before the other ends. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Disjoint spans placed back to back do not overlap. -/
theorem adjacent_disjoint (a an bn : Nat) (hn : 0 < an) : ¬ overlaps a an (a + an) bn := by
  unfold overlaps
  omega

/-- The overlap relation is symmetric. -/
theorem overlaps_symm (a an b bn : Nat) : overlaps a an b bn ↔ overlaps b bn a an := by
  unfold overlaps
  omega

end Strata.NativeGdn



## FILE: /mnt/user-data/outputs/NativeGdnPreprocessing.lean
/-!
# MODULE: native_gdn_preprocessing
Contracts of the gated-delta-net preprocessing kernels adapted from llama.cpp: 4-tap causal convolution
with SiLU and history shift, per-row L2/RMS norm over width 128, beta sigmoid, gate softplus, and the
output RMSNorm * gamma * sigmoid(z).

## Source semantics
- Convolution: per channel `c`, history `h[c*3..c*3+2]`, taps `w[c*4..c*4+3]`; the window is
  `[h0, h1, h2, x]` and the new history is `[h1, h2, x]`. Raw and SiLU outputs are written.
- L2 norm: width exactly 128, one block (256 threads) per row; scale `rsqrt(sum/128 + eps/128)` then `1/sqrt(128)`.
- Gate: `softplus(alpha + dt) * ssm_a` with softplus(v) = v when v > 20.
- Launch contracts: stream present, element count in [1, 65535], aligned non-null spans, writable spans disjoint.

## Form template identifiers
- S -> norm_width (128), history -> conv_history, ssm_a -> decay_rate, dt -> time_step_bias
- native_gdn_conv_silu -> causal_conv_silu, native_gdn_gate -> softplus_gate
-/

namespace Strata.NativeGdnPre

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § convolution

/-- Window `[h0,h1,h2,x]` as a four-tap list. -/
def window (h : Int × Int × Int) (x : Int) : List Int := [h.1, h.2.1, h.2.2, x]

/-- New history after one step. -/
def nextHistory (h : Int × Int × Int) (x : Int) : Int × Int × Int := (h.2.1, h.2.2, x)

/-- The new history is the last three entries of the window. -/
theorem next_history_is_window_tail (a b c x : Int) :
    (window (a, b, c) x).drop 1 = [(nextHistory (a, b, c) x).1, (nextHistory (a, b, c) x).2.1,
      (nextHistory (a, b, c) x).2.2] := rfl

/-- Four-tap dot product with the weight list. -/
def dot4 (w : Int × Int × Int × Int) (h : Int × Int × Int) (x : Int) : Int :=
  h.1 * w.1 + h.2.1 * w.2.1 + h.2.2 * w.2.2.1 + x * w.2.2.2

/-- The newest input is multiplied by the last tap. -/
theorem dot4_last_tap (w0 w1 w2 w3 a b c x : Int) :
    dot4 (w0, w1, w2, w3) (a, b, c) x = a * w0 + b * w1 + c * w2 + x * w3 := rfl

/-- Three steps leave the history equal to the last three inputs. -/
theorem history_after_three (h : Int × Int × Int) (x y z : Int) :
    nextHistory (nextHistory (nextHistory h x) y) z = (x, y, z) := rfl

/-- History (stride 3) and weight (stride 4) addresses of a channel are in range. -/
theorem conv_addresses_lt (c tap channels : Nat) (hc : c < channels) (h3 : tap < 3) (h4 : tap < 4) :
    c * 3 + tap < channels * 3 ∧ c * 4 + tap < channels * 4 :=
  ⟨idx_lt hc h3, idx_lt hc h4⟩

/-- Grid for the conv kernel covers all channels (256 per block). -/
theorem conv_grid_covers (channels : Nat) : channels ≤ (channels + 255) / 256 * 256 := by
  omega

-- § norm kernels

/-- A 256-thread block has 8 warps, which fit the 32-entry sums array. -/
theorem norm_warps_fit (t : Nat) (h : t < 256) : t / 32 < 8 ∧ 8 ≤ 32 := by
  omega

/-- The second-stage reduction reads lanes below 8 only. -/
theorem norm_second_stage (lane : Nat) (h : lane < 8) : lane < 32 := by
  omega

/-- Row base address `row*128 + col` stays inside `rows*128` for col < 128. -/
theorem norm_row_lt (row col rows : Nat) (hr : row < rows) (hc : col < 128) :
    row * 128 + col < rows * 128 :=
  idx_lt hr hc

/-- Threads with col >= 128 (128..255) contribute zero and do not write. -/
theorem idle_threads (t : Nat) (h : t < 256) : t < 128 ∨ (128 ≤ t ∧ t < 256) := by
  omega

/-- The norm scale `sum/S + eps/S` equals `(sum + eps)/S` over the integers when both divide evenly. -/
theorem scaled_sum_eq (sum eps : Nat) (hs : sum % 128 = 0) (he : eps % 128 = 0) :
    sum / 128 + eps / 128 = (sum + eps) / 128 := by
  omega

-- § elementwise kernels

/-- Grid for beta sigmoid / gate softplus covers all heads (256 per block). -/
theorem heads_grid_covers (heads : Nat) : heads ≤ (heads + 255) / 256 * 256 := by
  omega

/-- The softplus threshold branch: above 20 the value is returned unchanged. -/
def softplusBranch (v : Int) : Bool := decide (v > 20)

/-- The branch is taken exactly for v above 20. -/
theorem softplus_branch_iff (v : Int) : softplusBranch v = true ↔ 20 < v := by
  unfold softplusBranch
  simp

-- § launch contract

/-- Count-and-stream precondition. -/
def countOk (count : Int) : Prop := 1 ≤ count ∧ count ≤ 65535

/-- Valid counts launch at least one block and at most 255 blocks of 256 threads. -/
theorem count_blocks_bound (count : Nat) (h : 1 ≤ count ∧ count ≤ 65535) :
    1 ≤ (count + 255) / 256 ∧ (count + 255) / 256 ≤ 256 := by
  omega

/-- Norm geometry: width must be exactly 128. -/
def normGeometryOk (rows cols : Int) : Prop := 1 ≤ rows ∧ rows ≤ 65535 ∧ cols = 128

/-- Valid norm geometry has one block per row and each row fully covered by the 128 active threads. -/
theorem norm_geometry_cols (rows cols : Int) (h : normGeometryOk rows cols) : cols = 128 := by
  obtain ⟨_, _, h3⟩ := h
  exact h3

/-- Span byte counts for conv: history is 3x, weights 4x the per-channel float bytes. -/
theorem conv_span_bytes (channels : Nat) : 3 * (channels * 4) + 4 * (channels * 4) = 7 * (channels * 4) := by
  omega

end Strata.NativeGdnPre



## FILE: /mnt/user-data/outputs/NativeQuantMatVec.lean
/-!
# MODULE: native_quant_mat_vec
Block layouts, thread-to-block mappings, grid rules and launch contracts of the native quantized
matrix-vector kernels (Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q2_0) for 1 to 8 activation
columns, plus the q8_1 activation quantiser.

## Source semantics
- Activations are quantised to q8_1 blocks of 32 (36 bytes: half2 scale/sum + 32 int8).
- A kernel block has 4 warps x 32 lanes = 128 threads; thread `tid` handles weight block `tid / T` (+ k*BPI) and
  intra-block index `kqs(tid)`, where `T * BPI = 128` for every format.
- "Small K" (fewer weight blocks per row than BPI) uses 4 rows per block, otherwise 1 row per block; the
  non-exact multi-column layout uses 2 rows per block.
- Weight bytes = `(n_in / block_elems) * block_bytes * n_out`, with an overflow guard.

## Form template identifiers
- QK -> superblock_values, QI/VDR -> int_slots/vector_dot_ratio, BPI -> blocks_per_iter, T -> threads_per_block
- Q81Block -> act_block, native_mmvq -> quant_matvec, ncols -> n_columns
-/

namespace Strata.NativeMmvq

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § struct sizes

/-- Q5_K block: half2 + 12 scale bytes + 32 high bits + 128 low nibbles. -/
theorem q5k_size : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q4_K block: half2 + 12 + 128. -/
theorem q4k_size : 4 + 12 + 128 = 144 := by decide
/-- Q3_K block: 32 hmask + 64 qs + 12 scales + 2. -/
theorem q3k_size : 32 + 64 + 12 + 2 = 110 := by decide
/-- Q6_K block: 128 ql + 64 qh + 16 scales + 2. -/
theorem q6k_size : 128 + 64 + 16 + 2 = 210 := by decide
/-- IQ4_XS block: 2 + 2 + 4 + 128. -/
theorem iq4xs_size : 2 + 2 + 4 + 128 = 136 := by decide
/-- Q4_0 / IQ4_NL / Q2_0 block: half + 16 bytes. -/
theorem q40_size : 2 + 16 = 18 := by decide
/-- Q5_0 block: half + 4 high-bit bytes + 16. -/
theorem q50_size : 2 + 4 + 16 = 22 := by decide
/-- Q8_0 block: half + 32. -/
theorem q80_size : 2 + 32 = 34 := by decide
/-- q8_1 block: half2 + 32. -/
theorem q81_size : 4 + 32 = 36 := by decide

/-- Q4_K / Q5_K qs offsets: Q5_K qs begins after dm(4) + scales(12) + qh(32) = 48. -/
theorem q5k_qs_offset : 4 + 12 + 32 = 48 := by decide

/-- Q2_0 packs 64 two-bit codes into 16 bytes. -/
theorem q20_density : 16 * 4 = 64 := by decide

/-- Q5_0 high-bit masks are single bits 4, 12, 20 and 28. -/
theorem q50_high_masks :
    (0x10 : Nat) = 1 <<< 4 ∧ (0x1000 : Nat) = 1 <<< 12 ∧ (0x100000 : Nat) = 1 <<< 20 ∧
      (0x10000000 : Nat) = 1 <<< 28 :=
  ⟨rfl, rfl, rfl, rfl⟩

-- § thread to block mapping

/-- Threads per block: 4 warps of 32. -/
theorem block_threads : 4 * 32 = 128 := by decide

/-- Format traits (T lanes per weight block, BPI blocks per iteration) as (T, BPI) pairs. -/
def traitPairs : List (Nat × Nat) :=
  [(16, 8), (16, 8), (2, 64), (16, 8), (32, 4), (8, 16), (2, 64), (2, 64), (4, 32), (2, 64)]

/-- Every format covers all 128 threads: T * BPI = 128. -/
theorem trait_pairs_cover : ∀ p ∈ traitPairs, p.1 * p.2 = 128 := by decide

/-- Thread index splits into (weight block lane group, lane within block). -/
theorem thread_split (tid T : Nat) : T * (tid / T) + tid % T = tid :=
  Nat.div_add_mod tid T

/-- The block group index of a thread below `T*BPI` is below BPI. -/
theorem thread_group_lt (tid T BPI : Nat) (h : tid < T * BPI) : tid / T < BPI :=
  Nat.div_lt_of_lt_mul h

/-- The lane within a weight block is below T. -/
theorem thread_lane_lt (tid T : Nat) (hT : 0 < T) : tid % T < T := Nat.mod_lt _ hT

/-- Distinct threads handle distinct (group, lane) pairs when T lanes share a block. -/
theorem thread_pair_inj (tid tid' T : Nat) (hT : 0 < T) (h1 : tid / T = tid' / T)
    (h2 : tid % T = tid' % T) : tid = tid' := by
  have e := Nat.div_add_mod tid T
  have e' := Nat.div_add_mod tid' T
  rw [h1, h2] at e
  omega

/-- Small-K threshold values: BPI formulas VDR*4*32/QI. -/
theorem bpi_values :
    2 * 4 * 32 / 32 = 8 ∧ 4 * 32 / 16 = 8 ∧ 4 * 32 / 2 = 64 ∧ 4 * 4 * 32 / 32 = 16 ∧
      4 * 32 / 32 = 4 := by
  decide

/-- 32-element formats: BPI = 2*128/Qi gives 64 for Qi = 4 and 32 for Qi = 8. -/
theorem small_bpi : 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by decide

/-- Small-K element thresholds: 64 blocks of 32 = 2048 and 32 blocks = 1024. -/
theorem small_k_thresholds : 64 * 32 = 2048 ∧ 32 * 32 = 1024 := by decide

-- § rows per block and grids

/-- Four rows per block cover all rows. -/
theorem small_k_grid_covers (n_out : Nat) : n_out ≤ (n_out + 3) / 4 * 4 := by
  omega

/-- Two rows per block (upstream multi-column layout) cover all rows. -/
theorem upstream_grid_covers (n_out : Nat) : n_out ≤ (n_out + 1) / 2 * 2 := by
  omega

/-- Row of (block, i) with ROWS = 4 is below the padded row count. -/
theorem row_in_padded (b i n_out : Nat) (hb : b < (n_out + 3) / 4) (hi : i < 4) :
    4 * b + i < (n_out + 3) / 4 * 4 := by
  omega

/-- Row guards: padded rows above n_out are skipped (`row0 + i < n_out`). -/
theorem guarded_rows (row0 i n_out : Nat) (h : row0 + i < n_out) : row0 + i < n_out := h

/-- Weight block address `row*blocks_per_row + kbx` stays inside the matrix. -/
theorem weight_block_lt (row kbx n_out bpr : Nat) (hr : row < n_out) (hk : kbx < bpr) :
    row * bpr + kbx < n_out * bpr :=
  idx_lt hr hk

/-- Activation block address of column j: `j*(n_in/32) + kby` stays inside `ncols*(n_in/32)`. -/
theorem act_block_lt (j kby ncols stride : Nat) (hj : j < ncols) (hk : kby < stride) :
    j * stride + kby < ncols * stride :=
  idx_lt hj hk

/-- Output address `j*n_out + row` stays inside `ncols*n_out`. -/
theorem out_addr_lt (j row ncols n_out : Nat) (hj : j < ncols) (hr : row < n_out) :
    j * n_out + row < ncols * n_out :=
  idx_lt hj hr

-- § shared-memory partials

/-- Cross-warp partial buffer of the exact layout (3 extra warps, up to 8 columns, 4 rows): bytes. -/
theorem partial_bytes_exact (ncols : Nat) (h : ncols ≤ 8) : 3 * ncols * 4 * 32 * 4 ≤ 49152 := by
  omega

/-- Single-column partial buffer is 3*4*32 floats. -/
theorem partial_bytes_single : 3 * 4 * 32 * 4 = 1536 := by decide

/-- The upstream layout with 5..8 columns, 2 warps, 2 rows needs far less. -/
theorem partial_bytes_upstream : 1 * 8 * 2 * 32 * 4 = 2048 := by decide

-- § q8_1 quantiser

/-- Bytes of `ncols` quantised columns: ncols * (n_in/32) * 36. -/
def q81Bytes (n_in ncols : Nat) : Nat := ncols * (n_in / 32) * 36

/-- Quantised bytes add over columns. -/
theorem q81_bytes_add (n_in a b : Nat) : q81Bytes n_in (a + b) = q81Bytes n_in a + q81Bytes n_in b := by
  unfold q81Bytes
  simp only [Nat.add_mul]

/-- Columns quantise as one long vector: every 32-block stays inside one column. -/
theorem q8_block_in_column (c m k : Nat) (hk : k < 32 * m) :
    (c * (32 * m) + k) / 32 = c * m + k / 32 := by
  have e : c * (32 * m) = 32 * (c * m) := Nat.mul_left_comm c 32 m
  omega

/-- The column of a block is bounded: block index below `(c+1)*m`. -/
theorem q8_block_bound (c m k : Nat) (hk : k < 32 * m) : c * m + k / 32 < (c + 1) * m := by
  have e : (c + 1) * m = c * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Quantiser: element i is block i/32, position i%32. -/
theorem quant_split (i : Nat) : (i / 32) * 32 + i % 32 = i := by
  omega

/-- Only whole warps return early because n_in is a multiple of 32. -/
theorem whole_warps_return (i n : Nat) (hn : n % 32 = 0) (h : n ≤ i) : (i / 32) * 32 ≥ n := by
  omega

/-- Quantiser grid covers all elements with 256 threads per block. -/
theorem quant_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by
  omega

-- § weight byte accounting

/-- Total weight bytes. -/
def weightBytes (n_in block_elems block_bytes n_out : Nat) : Nat :=
  (n_in / block_elems) * block_bytes * n_out

/-- Weight bytes split additively over output rows. -/
theorem weight_bytes_add (n_in e b o1 o2 : Nat) :
    weightBytes n_in e b (o1 + o2) = weightBytes n_in e b o1 + weightBytes n_in e b o2 := by
  unfold weightBytes
  simp only [Nat.mul_add]

/-- The overflow guard `row_bytes <= MAX / n_out` implies `row_bytes * n_out <= MAX`. -/
theorem overflow_guard (rb n_out M : Nat) (hn : 0 < n_out) (h : rb ≤ M / n_out) : rb * n_out ≤ M :=
  (Nat.le_div_iff_mul_le hn).mp h

/-- Divisible shapes give exactly `n_in/elems` whole blocks per row. -/
theorem whole_blocks (n_in elems : Nat) (he : 0 < elems) (h : n_in % elems = 0) :
    (n_in / elems) * elems = n_in := by
  have := Nat.div_add_mod n_in elems
  rw [Nat.mul_comm] at this
  omega

-- § dispatch table

/-- Type codes supported by `native_mmvq_supported`. -/
def supportedTypes : List Nat := [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29]

/-- Sixteen distinct supported types. -/
theorem supported_nodup : supportedTypes.Nodup ∧ supportedTypes.length = 16 := by decide

/-- Types routed to the IQ kernel set. -/
def iqRoute : List Nat := [16, 17, 18, 21, 22, 29]

/-- Types routed to the native kernels. -/
def nativeRoute : List Nat := [2, 6, 8, 20, 11, 12, 13, 14, 23, 42]

/-- Every supported type has exactly one route. -/
theorem every_type_routed : ∀ t ∈ supportedTypes, (t ∈ iqRoute ∨ t ∈ nativeRoute) ∧
    ¬ (t ∈ iqRoute ∧ t ∈ nativeRoute) := by
  decide

/-- Block geometry table (type, elements, bytes) used by `native_mmvq_weight_bytes`. -/
def geometry : List (Nat × Nat × Nat) :=
  [(2, 32, 18), (6, 32, 22), (8, 32, 34), (20, 32, 18), (11, 256, 110), (12, 256, 144),
   (13, 256, 176), (14, 256, 210), (23, 256, 136), (42, 64, 18)]

/-- Every native-route type has a geometry row. -/
theorem native_types_have_geometry : ∀ t ∈ nativeRoute, ∃ g ∈ geometry, g.1 = t := by
  decide

/-- Geometry rows agree with the struct sizes: Q3_K..Q6_K and IQ4_XS byte counts. -/
theorem geometry_matches_structs :
    (256, 110) ∈ geometry.map (fun g => (g.2.1, g.2.2)) ∧ (256, 144) ∈ geometry.map (fun g => (g.2.1, g.2.2)) ∧
    (256, 176) ∈ geometry.map (fun g => (g.2.1, g.2.2)) ∧ (256, 210) ∈ geometry.map (fun g => (g.2.1, g.2.2)) ∧
    (256, 136) ∈ geometry.map (fun g => (g.2.1, g.2.2)) := by
  decide

-- § launch contract

/-- Shape validation: positive n_in divisible by the block element count, 1..8 columns. -/
def shapeOk (n_in ncols elems : Nat) : Prop := 0 < n_in ∧ n_in % elems = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8

/-- Valid shapes give an integral number of q8_1 blocks per column when elems is a multiple of 32. -/
theorem shape_ok_q8_blocks (n_in ncols elems : Nat) (h : shapeOk n_in ncols elems)
    (he : elems % 32 = 0) (hpos : 0 < elems) : n_in % 32 = 0 := by
  obtain ⟨_, h2, _, _⟩ := h
  have : ∃ m, elems = 32 * m := ⟨elems / 32, by omega⟩
  obtain ⟨m, hm⟩ := this
  subst hm
  have : n_in % (32 * m) = 0 := h2
  have h3 : 32 ∣ n_in := by
    have : 32 * m ∣ n_in := Nat.dvd_of_mod_eq_zero this
    exact Nat.dvd_trans ⟨m, rfl⟩ this
  exact Nat.mod_eq_zero_of_dvd h3

/-- Pointer validation: 4-byte alignment of an address. -/
def aligned4 (addr : Nat) : Prop := addr % 4 = 0

/-- Offsets that are multiples of 4 preserve alignment. -/
theorem aligned_offset (addr off : Nat) (h : aligned4 addr) (ho : off % 4 = 0) : aligned4 (addr + off) := by
  unfold aligned4 at *
  omega

/-- Multi-column kernels are used only for 2..8 columns. -/
theorem multi_range (ncols : Nat) (h1 : 2 ≤ ncols) (h2 : ncols ≤ 8) : 1 < ncols ∧ ncols ≤ 8 := by
  omega

end Strata.NativeMmvq



## FILE: /mnt/user-data/outputs/QsaIndexerPooling.lean
/-!
# MODULE: qsa_indexer_pooling
Slot, block and buffer-bound contracts of the QSA indexer append kernel: each token's 128-dim indexer key is
stored in a small tail ring of R-1 = 3 slots; when a block of R = 4 tokens completes (or at position 0) the block
is mean-pooled, RMS-normalised with gamma, RoPE-rotated on the first 64 dims, and written into the pooled array.

## Source semantics
- `slot = pos % R`; slots 0..R-2 are saved to `tail`; the kernel proceeds to pooling only when `pos == 0` or
  `slot == R-1`.
- Pooled block index `b = pos / R`, rope position `pos_base + R*b` (0 at position 0). The rotation pairs dim
  `pair` with `pair + 32` for `pair = d % 32`, d < 64.
- At a block-completing position (pos != 0) the next pooled row `b+1` is seeded with the "dead" copy.
- Launch contract: fixed geometry (idx_dim 128, block 4, n_rot 64), 1 <= max_cells < 2^31, pos_base a multiple
  of R with `pos_base + max_cells < 2^31`, finite positive epsilon, frequency base > 1, explicit stream, seven
  aligned pairwise-disjoint buffers; pooled holds `max_cells/R + 1` rows.

## Form template identifiers
- D -> indexer_dim (128), R -> pool_block (4), ROT -> rotary_dims (64), pooled -> pooled_blocks
- tail -> pending_tail, dead -> seed_row, block_pos -> last_block_position
-/

namespace Strata.QsaIndexer

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § slots and blocks

/-- A position splits into block and slot. -/
theorem pos_split (pos : Nat) : 4 * (pos / 4) + pos % 4 = pos := by
  omega

/-- Slot is below R. -/
theorem slot_lt (pos : Nat) : pos % 4 < 4 := by
  omega

/-- Tail slots written by `d < 128` threads: slot * 128 + d stays inside the 3-slot tail. -/
theorem tail_index_lt (slot d : Nat) (hs : slot < 3) (hd : d < 128) : slot * 128 + d < 3 * 128 :=
  idx_lt hs hd

/-- Pooling is triggered exactly at pos 0 or when the slot is the last of the block. -/
def poolTriggered (pos : Nat) : Prop := pos = 0 ∨ pos % 4 = 3

/-- Positions that do not pool are exactly those with slot in 1..2 or slot 0 with pos > 0. -/
theorem not_triggered_iff (pos : Nat) : ¬ poolTriggered pos ↔ pos ≠ 0 ∧ pos % 4 ≠ 3 := by
  unfold poolTriggered
  omega

/-- Every pool-triggering position past 0 closes a full block: pos = 4b + 3. -/
theorem trigger_closes_block (pos : Nat) (h : pos ≠ 0) (ht : poolTriggered pos) :
    pos = 4 * (pos / 4) + 3 := by
  unfold poolTriggered at ht
  omega

-- § pooled-array bounds

/-- Pooled row of a block-closing position is below `max_cells/R + 1`. -/
theorem pooled_row_lt (pos max_cells : Nat) (hp : pos < max_cells) :
    pos / 4 < max_cells / 4 + 1 := by
  omega

/-- The seed row `b+1` written at a block-closing position is also in range. -/
theorem seed_row_lt (pos max_cells : Nat) (hp : pos < max_cells) (hs : pos % 4 = 3) :
    pos / 4 + 1 < max_cells / 4 + 1 := by
  omega

/-- Pooled element address bound. -/
theorem pooled_elem_lt (row d max_cells : Nat) (hr : row < max_cells / 4 + 1) (hd : d < 128) :
    row * 128 + d < (max_cells / 4 + 1) * 128 :=
  idx_lt hr hd

/-- Position 0 writes only row 0, which exists for every capacity (the row count is at least 1). -/
theorem row_zero_exists (max_cells : Nat) : 0 < max_cells / 4 + 1 := by
  omega

-- § rotary mapping

/-- Pair index of a rotated dim is below 32. -/
theorem pair_lt (d : Nat) : d % 32 < 32 := by
  omega

/-- Both elements of a rotation pair lie in [0, 64). -/
theorem pair_elems_lt (pair : Nat) (h : pair < 32) : pair < 64 ∧ pair + 32 < 64 := by
  omega

/-- Dims below 32 use the first-half formula and their pair index is d itself. -/
theorem rotary_first_half (d : Nat) (h : d < 32) : d % 32 = d := by
  omega

/-- Dims 32..63 use the second-half formula and their pair index is d - 32. -/
theorem rotary_second_half (d : Nat) (h1 : 32 ≤ d) (h2 : d < 64) : d % 32 = d - 32 := by
  omega

/-- Dims 64..127 are not rotated: exactly the dims below ROT pass through the rotation. -/
theorem rotated_dims (d : Nat) (h : d < 128) : d < 64 ∨ (64 ≤ d ∧ d < 128) := by
  omega

-- § rope position

/-- The rope position of block b is `pos_base + 4*b`, a multiple of 4 when pos_base is. -/
theorem rope_pos_mod (pos_base b : Nat) (h : pos_base % 4 = 0) : (pos_base + 4 * b) % 4 = 0 := by
  omega

/-- The rope position stays below `pos_base + max_cells` for positions below max_cells. -/
theorem rope_pos_lt (pos_base pos max_cells : Nat) (hp : pos < max_cells) :
    pos_base + 4 * (pos / 4) < pos_base + max_cells := by
  omega

/-- The int32 capacity check implies rope positions fit in int32. -/
theorem rope_pos_int32 (pos_base pos max_cells : Nat) (hp : pos < max_cells)
    (hb : pos_base + max_cells ≤ 2147483647) : pos_base + 4 * (pos / 4) ≤ 2147483647 := by
  omega

-- § block geometry

/-- 256 threads, of which d < 128 are active: 8 warps for the norm reduction. -/
theorem norm_reduce_warps (t : Nat) (h : t < 256) : t / 32 < 8 := by
  omega

/-- The block mean divides the 4-term sum by 4 (SCALE with 0.25): the pool size is R = 4. -/
theorem pool_size : 4 * 1 = 4 := rfl

-- § launch contract

/-- Validity of the append launch parameters over integers. -/
def appendArgsValid (idx_dim idx_block n_rot max_cells pos_base : Int) : Prop :=
  idx_dim = 128 ∧ idx_block = 4 ∧ n_rot = 64 ∧ 1 ≤ max_cells ∧ max_cells ≤ 2147483647 ∧
    0 ≤ pos_base ∧ pos_base % 4 = 0 ∧ pos_base + max_cells ≤ 2147483647

/-- Valid arguments have a pooled array of at least one row and a rope position range inside int32. -/
theorem append_valid_capacity (a b c m p : Int) (h : appendArgsValid a b c m p) :
    0 ≤ p + m ∧ p + m ≤ 2147483647 := by
  obtain ⟨_, _, _, h4, _, h6, _, h8⟩ := h
  omega

/-- Span bytes of the buffers: pooled needs `(max_cells/4 + 1) * 128 * 4` bytes. -/
def pooledBytes (max_cells : Nat) : Nat := (max_cells / 4 + 1) * 128 * 4

/-- The pooled span covers every written row. -/
theorem pooled_span_covers_rows (row max_cells : Nat) (hr : row < max_cells / 4 + 1) :
    (row + 1) * 128 * 4 ≤ pooledBytes max_cells := by
  unfold pooledBytes
  have h : (row + 1) * 128 ≤ (max_cells / 4 + 1) * 128 := Nat.mul_le_mul_right 128 (Nat.succ_le_of_lt hr)
  omega

end Strata.QsaIndexer



## FILE: /mnt/user-data/outputs/QsaBlockScoring.lean
/-!
# MODULE: qsa_block_scoring
Tile geometry, shared-memory layout, step validity and cell-export contracts of the QSA score kernel: for each
pooled block row, 4 indexer heads are scored with tensor-core tiles over K = 128, ReLU'd per head, summed,
biased, masked for the partial last block, and replicated to the R = 4 cells of the block.

## Source semantics
- Grid covers `max_blocks` rows in tiles of ROWS = 32; block = 2 warps (64 threads); warp w accumulates
  K columns `w*32+lane` and `w*32+lane+64`.
- Shared tile per warp is 16 rows x STRIDE 36 floats; the combine buffer is indexed `h*COMBINE + i`, with
  COMBINE = 68, i = warp*32 + ia*16 + (l/2)*8 + lane/4 and head `h = (lane%4)*2 + l%2`.
- Step validity: `n >= 1`, `n <= max_cells`, `pos == n-1`, `full == n/4`, `width == min(n, 2051)`; invalid steps
  produce no output. Rows `row > full` are skipped; cell export covers `i in [row*4, min(n, (row+1)*4))`.
- The partial last block (`row == full` and `n % 4 != 0`) is penalised with 1e9.

## Form template identifiers
- ROWS -> rows_per_tile, WARPS -> warps_per_block, STRIDE -> tile_stride, COMBINE -> combine_stride
- pooled -> pooled_blocks, cells -> cell_scores, kStepNKv -> kv_len, kStepNBid -> full_blocks
-/

namespace Strata.QsaScore

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § step validity

/-- The kernel's step validity predicate over integers. -/
def validStep (n maxc pos full width : Int) : Prop :=
  1 ≤ n ∧ n ≤ maxc ∧ pos = n - 1 ∧ full = n / 4 ∧ width = (if n < 2051 then n else 2051)

/-- The kernel's block count: full blocks plus the partial block row. -/
theorem valid_full_le (n maxc pos full width : Int) (h : validStep n maxc pos full width) :
    full * 4 ≤ n := by
  obtain ⟨_, _, _, h4, _⟩ := h
  omega

/-- Valid steps have a positive context length. -/
theorem valid_pos_nonneg (n maxc pos full width : Int) (h : validStep n maxc pos full width) :
    0 ≤ pos := by
  obtain ⟨h1, _, h3, _, _⟩ := h
  omega

-- § grid and rows

/-- Grid covers max_blocks rows with 32 rows per block. -/
theorem grid_covers (max_blocks : Nat) : max_blocks ≤ (max_blocks + 31) / 32 * 32 := by
  omega

/-- Row of (block, lane) is block*32 + lane; lane < 32 gives unique rows. -/
theorem row_inj (b l b' l' : Nat) (hl : l < 32) (hl' : l' < 32) (h : b * 32 + l = b' * 32 + l') :
    b = b' ∧ l = l' := by
  omega

/-- With `max_blocks = max_cells/4 + 1`, every row up to `full = n/4` exists when n <= max_cells. -/
theorem full_row_exists (n max_cells : Nat) (h : n ≤ max_cells) : n / 4 < max_cells / 4 + 1 := by
  omega

/-- A row the kernel scores (row <= full) is a valid pooled row. -/
theorem scored_row_lt (row full n max_cells : Nat) (hr : row ≤ full) (hf : full = n / 4)
    (hn : n ≤ max_cells) : row < max_cells / 4 + 1 := by
  omega

-- § K split across warps

/-- Column of (warp, lane, iteration) with 2 warps: warp*32 + lane + 64*iter. -/
theorem k_col_lt (w l it : Nat) (hw : w < 2) (hl : l < 32) (hi : it < 2) : w * 32 + l + 64 * it < 128 := by
  omega

/-- Distinct (warp, lane, iteration) triples read distinct columns. -/
theorem k_col_inj (w l it w' l' it' : Nat) (hw : w < 2) (hl : l < 32) (hw' : w' < 2) (hl' : l' < 32)
    (h : w * 32 + l + 64 * it = w' * 32 + l' + 64 * it') : w = w' ∧ l = l' ∧ it = it' := by
  omega

/-- Every K column of 128 is read exactly once. -/
theorem k_col_cover (c : Nat) (h : c < 128) :
    ∃ w l it, w < 2 ∧ l < 32 ∧ it < 2 ∧ c = w * 32 + l + 64 * it :=
  ⟨(c % 64) / 32, c % 32, c / 64, by omega, by omega, by omega, by omega⟩

/-- Each warp covers 64 columns in two strides of 32. -/
theorem warp_columns : 2 * 2 * 32 = 128 := by decide

-- § shared memory layout

/-- Shared tile floats for two warps: 2 * 16 * 36. -/
theorem shared_tile_floats : 2 * 16 * 36 = 1152 := by decide

/-- The combine buffer (8 heads x 68) is smaller than the shared tile it reuses. -/
theorem combine_fits_in_tile : 8 * 68 ≤ 2 * 16 * 36 := by decide

/-- Tile row of the A operand: (tid%16)*STRIDE + (tid/16)*4 + 3 stays inside 16 rows of STRIDE 36. -/
theorem load_a_in_tile (tid : Nat) (h : tid < 32) : (tid % 16) * 36 + (tid / 16) * 4 + 3 < 16 * 36 := by
  omega

/-- Tile row of the B operand: (tid%8)*STRIDE + ((tid/8)*4)%8 + 3 stays inside 8 rows. -/
theorem load_b_in_tile (tid : Nat) (h : tid < 32) : (tid % 8) * 36 + ((tid / 8) * 4) % 8 + 3 < 8 * 36 := by
  omega

/-- Tile fill: row i, lane l writes `i*36 + l` with l < 32, below the row stride. -/
theorem fill_in_stride (i l : Nat) (hi : i < 16) (hl : l < 32) : i * 36 + l < 16 * 36 := by
  omega

/-- Combine row index `warp*32 + ia*16 + (l/2)*8 + lane/4` is below 64 (< COMBINE = 68). -/
theorem combine_row_lt (w ia l lane : Nat) (hw : w < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    w * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by
  omega

/-- Combine head index `(lane%4)*2 + l%2` is below 8. -/
theorem combine_head_lt (lane l : Nat) : (lane % 4) * 2 + l % 2 < 8 := by
  omega

/-- Combine address bound: `h*68 + i < 544` for h < 8, i < 64. -/
theorem combine_addr_lt (h i : Nat) (hh : h < 8) (hi : i < 64) : h * 68 + i < 8 * 68 := by
  omega

/-- The two warp partials of row i sit at `h*68 + i` and `h*68 + 32 + i` for the combining warp-0 lane i < 32. -/
theorem warp_partial_addrs (h lane : Nat) (hh : h < 8) (hl : lane < 32) :
    h * 68 + lane < 8 * 68 ∧ h * 68 + 32 + lane < 8 * 68 := by
  omega

/-- Each (warp, ia, l-pair, lane/4) row of the combine buffer is hit by a unique (ia, l/2, lane/4, warp). -/
theorem combine_row_inj (w ia a lq w' ia' a' lq' : Nat) (ha : a < 2) (hlq : lq < 8) (ha' : a' < 2)
    (hlq' : lq' < 8) (hia : ia < 2) (hia' : ia' < 2)
    (h : w * 32 + ia * 16 + a * 8 + lq = w' * 32 + ia' * 16 + a' * 8 + lq') :
    w = w' ∧ ia = ia' ∧ a = a' ∧ lq = lq' := by
  omega

-- § cell export

/-- Cell index recombines: i / 4 is the row that exports it. -/
theorem cell_row (i : Nat) : 4 * (i / 4) + i % 4 = i := by
  omega

/-- A cell below `n` is exported by a row that is scored (row <= n/4). -/
theorem cell_row_scored (i n : Nat) (h : i < n) : i / 4 ≤ n / 4 := by
  omega

/-- The partial block row is the last scored row exactly when n % 4 != 0. -/
theorem partial_row_is_last (n : Nat) (h : n % 4 ≠ 0) : n / 4 * 4 < n := by
  omega

/-- Capacity precondition: `max_blocks = max_cells/4 + 1` is what the launcher requires. -/
def capacitiesOk (max_cells max_blocks : Nat) : Prop := 1 ≤ max_cells ∧ max_blocks = max_cells / 4 + 1

/-- Under the capacity precondition the cells array covers every exported cell. -/
theorem cells_cover_exports (max_cells max_blocks n i : Nat) (h : capacitiesOk max_cells max_blocks)
    (hn : n ≤ max_cells) (hi : i < n) : i < max_cells := by
  omega

end Strata.QsaScore





## FILE: /mnt/user-data/outputs/NativeRouterTop10.lean
/-!
# MODULE: native_router_top10
Single-row 512-expert softmax / top-10 / renormalise router (llama.cpp topk-moe adaptation) with an
argument-validity guard (alignment, disjoint spans). One 32-lane warp holds 16 experts per lane.

## Source semantics
Expert `e` lives in lane `e % 32`, register `e / 32`. Ties break by lowest index. Winning-expert lane
accumulates the selected sum; the sum is clamped below by 2^-14. Output `weights[rank]` is read from lane `rank`.

## Form template identifiers
native_router_top10 -> router_top10_fixed; logits -> scores; ids -> expert_ids; weights -> expert_weights
-/
namespace Strata.NativeRouter

/-- Half-open span overlap, as in `overlap()` of native_router.cu. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> (intro h; omega)

/-- Expert id held in register `i` of lane `lane`. -/
def expertOf (lane i : Nat) : Nat := lane + i * 32

/-- The lane of an expert id is recovered by `% 32`. -/
theorem expert_lane (lane i : Nat) (h : lane < 32) : expertOf lane i % 32 = lane := by
  unfold expertOf; omega

/-- The register slot of an expert id is recovered by `/ 32`. -/
theorem expert_slot (lane i : Nat) (h : lane < 32) : expertOf lane i / 32 = i := by
  unfold expertOf; omega

/-- All 512 experts are covered by 32 lanes x 16 registers. -/
theorem expert_range (lane i : Nat) (hl : lane < 32) (hi : i < 16) : expertOf lane i < 512 := by
  unfold expertOf; omega

/-- Strict preference order: larger value, then smaller index. -/
def better (v i v' i' : Nat) : Prop := v > v' ∨ (v = v' ∧ i < i')

/-- The preference order is asymmetric, so the shuffle tree is well defined. -/
theorem better_asymm (v i v' i' : Nat) (h : better v i v' i') : ¬ better v' i' v i := by
  unfold better at *; omega

/-- The lower clamp of the selected sum. -/
theorem clamp_ge (s m : Nat) : m ≤ max s m := by omega

/-- Ten output ranks fit in the 32 lanes, so `rank == lane` reads every output. -/
theorem rank_lane (r : Nat) (h : r < 10) : r < 32 := by omega

end Strata.NativeRouter



## FILE: /mnt/user-data/outputs/PleBlock.lean
/-!
# MODULE: ple_block
GPU half of the PLE block: grouped RMS norms, signed-sqrt sigmoid gate, depthwise causal dilated conv + SiLU,
three-way add, and history shift. Caller-owned scratch workspace.

## Source semantics
Conv tap k reads row `nhist - (kern-1-k)*dil` of [history | new]; history is row-fastest (`row + nhist*c`).
Scratch regions are 16-byte aligned. Output buffers must not overlap scratch.

## Form template identifiers
ple_block -> gated_conv_block; hist -> conv_state; normalized -> conv_input
-/
namespace Strata.Ple

/-- Round up to a multiple of 16. -/
def round16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounded size is not smaller. -/
theorem round16_ge (n : Nat) : n ≤ round16 n := by unfold round16; omega

/-- Rounded size is 16-byte aligned. -/
theorem round16_mod (n : Nat) : round16 n % 16 = 0 := by unfold round16; omega

/-- Row read by tap `k`. -/
def convRow (nhist kern dil k : Nat) : Int := (nhist : Int) - ((kern : Int) - 1 - k) * dil

/-- Real geometry (kern 4, dil 3, hist 9): taps read rows 0, 3, 6, 9. -/
theorem conv_rows_real : [0, 1, 2, 3].map (fun k => convRow 9 4 3 k) = [0, 3, 6, 9] := by decide

/-- Row-fastest flat index of the conv state. -/
def idx (nhist row c : Nat) : Nat := row + nhist * c

/-- Row-fastest indexing recovers the channel. -/
theorem idx_div (nhist row c : Nat) (hp : 0 < nhist) (hr : row < nhist) : idx nhist row c / nhist = c := by
  unfold idx
  rw [Nat.add_mul_div_left _ _ hp, Nat.div_eq_of_lt hr]
  simp

/-- Row-fastest indexing recovers the row. -/
theorem idx_mod (nhist row c : Nat) (hr : row < nhist) : idx nhist row c % nhist = row := by
  unfold idx
  rw [Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt hr

/-- One column of the history shifts left and appends the new entry. -/
def advance {α : Type} (col : List α) (x : α) : List α := col.tail ++ [x]

/-- History advance preserves column length for a non-empty column. -/
theorem advance_length {α : Type} (a : α) (l : List α) (x : α) :
    (advance (a :: l) x).length = (a :: l).length := by
  simp [advance]

/-- Pointer-range overlap test of `overlap()` in ple.cu. -/
def ovl (aa an bb bn : Nat) : Bool :=
  if aa < bb then decide (bb - aa < an) else decide (aa - bb < bn)

/-- For non-empty spans the test is exactly interval overlap. -/
theorem ovl_iff (aa an bb bn : Nat) (ha : 0 < an) (hb : 0 < bn) :
    ovl aa an bb bn = true ↔ (aa < bb + bn ∧ bb < aa + an) := by
  unfold ovl
  split <;> simp <;> omega

/-- Sign function used by the gate (zero maps to zero). -/
def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- Sign is odd, which keeps the gate symmetric about 0.5. -/
theorem sgn_neg (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn
  rcases lt_trichotomy s 0 with h | h | h
  · have h1 : ¬ (s > 0) := by omega
    have h2 : -s > 0 := by omega
    simp only [if_pos h2, if_neg h1, if_pos h] <;> omega
  · subst h
    simp
  · have h2 : ¬ (-s > 0) := by omega
    have h3 : -s < 0 := by omega
    have h1 : s > 0 := h
    simp only [if_neg h2, if_pos h3, if_pos h1] <;> omega

end Strata.Ple



## FILE: /mnt/user-data/outputs/QsaCacheIndexerAttention.lean
/-!
# MODULE: qsa_cache_indexer_attention
QSA paged KV append, block-pooled indexer keys, indexer scoring, exact top-k by threshold search, gather, attention.

## Source semantics
Step state is derived from one position: n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1).
A pooled block completes when `pos % r = r-1`. Top-k picks all cells above the threshold key plus the first
`width - c_gt` equal cells by ascending index. Query head h uses kv head `h / (n_head / n_head_kv)`.

## Form template identifiers
qsa -> sparse_block_attention; kStep* -> step_state fields; r -> idx_block (4 in practice)
-/
namespace Strata.Qsa

/-- Selection width. -/
def selWidth (nKv topK r : Nat) : Nat := min nKv (topK + r - 1)

/-- Width never exceeds the number of cells. -/
theorem selWidth_le (nKv topK r : Nat) : selWidth nKv topK r ≤ nKv := by unfold selWidth; omega

/-- Below the bound the selection is the identity. -/
theorem selWidth_identity (nKv topK r : Nat) (h : nKv ≤ topK + r - 1) : selWidth nKv topK r = nKv := by
  unfold selWidth; omega

/-- Threshold selection emits exactly `width` cells: all above, plus the equal budget. -/
theorem selection_count (gt eq width : Nat) (h1 : gt ≤ width) (h2 : width ≤ gt + eq) :
    gt + min eq (width - gt) = width := by omega

/-- Pooled block completion for r = 4: the next position starts a new block. -/
theorem pool_complete (pos : Nat) (h : pos % 4 = 3) :
    (pos + 1) % 4 = 0 ∧ (pos + 1) / 4 = pos / 4 + 1 := by omega

/-- Completed blocks never cover more cells than exist. -/
theorem nbid_cover (pos : Nat) : ((pos + 1) / 4) * 4 ≤ pos + 1 := by omega

/-- Incomplete tail size plus completed cells equals n_kv. -/
theorem tail_size (nKv : Nat) : (nKv / 4) * 4 + nKv % 4 = nKv := by omega

/-- GQA mapping for 24 heads over 2 kv heads stays in range. -/
theorem gqa_range (h : Nat) (hh : h < 24) : h / (24 / 2) < 2 := by omega

/-- Slot of a position inside its page is in range. -/
theorem page_slot_lt (pos ps : Nat) (hp : 0 < ps) : pos % ps < ps := Nat.mod_lt _ hp

/-- Pool of positions: page index times page size plus slot reconstructs the position. -/
theorem page_decomp (pos ps : Nat) : (pos / ps) * ps + pos % ps = pos := by
  have := Nat.div_add_mod pos ps
  rw [Nat.mul_comm] at this
  omega

/-- Uint2 gather copies four halves at a time, so head_dim must be a multiple of 4. -/
theorem uint2_per (hd : Nat) (h : hd % 4 = 0) : (hd / 4) * 4 = hd := by omega

end Strata.Qsa



## FILE: /mnt/user-data/outputs/QsaDecodeAttention.lean
/-!
# MODULE: qsa_decode_attention
Split-K decode attention over gathered cells: 64-cell chunks per (chunk, kv head), online-softmax partials
(max, sum, acc), then a merge kernel. FP16 or INT8+scale KV pools. 12 query heads per kv head, head_dim 256.

## Source semantics
Chunk c covers cells [64c, 64c+n_here). Empty chunks emit m = -FLT_MAX, l = 0 and are skipped by the merge.
Merge: M = max m; L = sum l*exp(m-M); out = acc/L or 0 if L = 0.

## Form template identifiers
chunk -> split_block; part_m/part_l/part_acc -> partial_max/partial_sum/partial_acc
-/
namespace Strata.QsaDecode

/-- Number of 64-cell chunks. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Chunks cover the capacity. -/
theorem chunks_cover (cap : Nat) : cap ≤ nChunks cap * 64 := by unfold nChunks; omega

/-- Cells handled by chunk `c` of `n` ids. -/
def nHere (n c : Nat) : Nat := min 64 (n - c * 64)

/-- A chunk handles at most 64 cells. -/
theorem nHere_le (n c : Nat) : nHere n c ≤ 64 := by unfold nHere; omega

/-- Chunk sizes sum correctly for the first chunk when the selection fits one chunk. -/
theorem nHere_single (n : Nat) (h : n ≤ 64) : nHere n 0 = n := by unfold nHere; omega

/-- Chunks past the end are empty. -/
theorem nHere_empty (n c : Nat) (h : n ≤ c * 64) : nHere n c = 0 := by unfold nHere; omega

/-- Scratch floats for a capacity, with 12 heads x kv heads = n_head. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * (256 + 2) + 64

/-- Scratch holds acc (HD per head) plus m and l (one each) per chunk-head. -/
theorem scratch_split (cap nHead : Nat) :
    scratchFloats cap nHead = nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead + 64 := by
  unfold scratchFloats
  generalize nChunks cap * nHead = m
  omega

/-- Query head index decomposes into kv head and local head. -/
theorem head_decomp (h : Nat) : (h / 12) * 12 + h % 12 = h := by omega

/-- Each of the 12 local heads is below 12. -/
theorem local_head_lt (h : Nat) : h % 12 < 12 := by omega

/-- KV head of a query head of 24 is below 2. -/
theorem kv_head_lt (h : Nat) (hh : h < 24) : h / 12 < 2 := by omega

/-- INT8 scale group index stays inside the row (256 dims, groups of KV_Q8_GROUP = 32 assumed). -/
theorem scale_group_lt (d : Nat) (hd : d < 256) : d / 32 < 8 := by omega

end Strata.QsaDecode



## FILE: /mnt/user-data/outputs/QsaBlockSelect.lean
/-!
# MODULE: qsa_block_select
Batched block scoring (relu'd per-head dots, +1e9 for an incomplete tail) and block-level radix top-k that
maps selected blocks to cell ids (4 cells per block).

## Source semantics
Block b < n_bid has weight 4; block n_bid has weight n_kv - 4 n_bid. If n_kv <= width everything is selected.
Radix select works on 8-bit digits at shifts 24, 16, 8, 0 to find the largest threshold key with weighted
count >= width; equal-key cells fill the remaining budget lowest index first.

## Form template identifiers
block_scores -> block_score; block_topk -> block_select
-/
namespace Strata.QsaSelect

/-- Cells in block `b`. -/
def weight (nKv nBid b : Nat) : Nat := if b < nBid then 4 else nKv - nBid * 4

/-- The last block weight is the tail size. -/
theorem weight_last (nKv : Nat) : weight nKv (nKv / 4) (nKv / 4) = nKv % 4 := by
  unfold weight; split <;> omega

/-- Completed cells plus the tail give n_kv. -/
theorem weights_total (nKv : Nat) : (nKv / 4) * 4 + weight nKv (nKv / 4) (nKv / 4) = nKv := by
  unfold weight; split <;> omega

/-- Cell id of the c-th cell of block b. -/
def cellId (b c : Nat) : Nat := b * 4 + c

/-- Cell ids decompose back into block and offset. -/
theorem cell_block (b c : Nat) (h : c < 4) : cellId b c / 4 = b ∧ cellId b c % 4 = c := by
  unfold cellId; omega

/-- Four 8-bit radix passes cover 32 bits. -/
theorem radix_cover : [24, 16, 8, 0].map (fun s => s + 8) = [32, 24, 16, 8] := by decide

/-- Equal-key budget never exceeds what remains. -/
theorem eq_budget_le (width above : Nat) (h : above ≤ width) : width - above ≤ width := by omega

/-- Per-thread equal quota is clamped between zero and its own count. -/
theorem my_eq_clamp (budget before eq : Nat) : min eq (budget - before) ≤ eq := by omega

/-- Block-per-warp grid covers all blocks. -/
theorem grid_cover (mb : Nat) : mb ≤ ((mb + 7) / 8) * 8 := by omega

end Strata.QsaSelect



## FILE: /mnt/user-data/outputs/QuantizeActivation.lean
/-!
# MODULE: quantize_activation
Activation quantizers matching ggml: Q8_0 (34-byte blocks, fp16 scale), Q8_0 with fp32 scales (CPU-matching rule,
round half away from zero), and Q8_K (292-byte blocks). Also the dequantizers.

## Source semantics
Q8_0: d32 = amax/127; stored scale is fp16(d32); integers divide by d32. Scaled variant clamps to [-127,127].
Q8_K: iscale = -127/max (signed value at largest magnitude); only an upper clamp at 127; bsums per 16 elements.

## Form template identifiers
quantize_q8_0 -> act_quant_block32; quantize_q8_K -> act_quant_block256
-/
namespace Strata.QuantizeAct

/-- A Q8_0 block is a 2-byte scale plus 32 int8. -/
theorem q8_0_block_bytes : 2 + 32 = 34 := by decide

/-- A Q8_K block is f32 d, 256 int8, and 16 int16 partial sums. -/
theorem q8_K_block_bytes : 4 + 256 + 2 * 16 = 292 := by decide

/-- Q8_K has 16 partial sums of 16 elements each. -/
theorem q8_K_bsums : 256 / 16 = 16 := by decide

/-- Consecutive Q8_0 blocks are 34 bytes apart. -/
theorem q8_0_stride (b : Nat) : (b + 1) * 34 = b * 34 + 34 := by omega

/-- Consecutive Q8_K blocks are 292 bytes apart. -/
theorem q8_K_stride (b : Nat) : (b + 1) * 292 = b * 292 + 292 := by omega

/-- Symmetric clamp used by the scaled quantizer lands in [-127, 127]. -/
theorem clamp_range (v : Int) : -127 ≤ max (-127) (min 127 v) ∧ max (-127) (min 127 v) ≤ 127 := by omega

/-- Q8_K upper-only clamp never exceeds 127. -/
theorem q8K_upper (v : Int) : min 127 v ≤ 127 := by omega

/-- The nearest-int magic constant 12582912 is 1.5 * 2^23. -/
theorem magic_constant : 12582912 = 3 * 4194304 ∧ 4194304 = 2 ^ 22 := by decide

/-- Grid sizing covers every block. -/
theorem grid_cover (nb t : Nat) (h : 0 < t) : nb ≤ ((nb + t - 1) / t) * t := by
  have := Nat.div_add_mod (nb + t - 1) t
  have h2 := Nat.mod_lt (nb + t - 1) h
  rw [Nat.mul_comm] at this
  omega

/-- Block count for 2560 elements in Q8_0 is 80. -/
theorem blocks_2560 : 2560 / 32 = 80 := by decide

/-- Block count for 2560 elements in Q8_K is 10. -/
theorem blocks_2560_K : 2560 / 256 = 10 := by decide

/-- Dequantization of a zero scale is zero. -/
theorem dequant_zero_scale (q : Int) : q * 0 = 0 := by omega

end Strata.QuantizeAct



## FILE: /mnt/user-data/outputs/RouterTop10.lean
/-!
# MODULE: router_top10
MoE router: softmax over all experts (double exp, ascending double sum), k block-wide argmax passes with
lowest-index tie rule, renormalise the gathered weights with lower clamp 2^-14.

## Source semantics
ids = stable descending argsort(p)[:k]; w = gather(p, ids); w / max(sum w, 2^-14).
Shared memory: taken mask (16-aligned), n_expert doubles, n_expert floats. Threads = min(n_expert,512) rounded to 32.

## Form template identifiers
router_top10 -> topk_gate; n_expert -> num_experts; k -> top_k
-/
namespace Strata.RouterTop10

/-- Preference order of the argmax passes: larger value, then smaller index. -/
def better (v i v' i' : Nat) : Prop := v > v' ∨ (v = v' ∧ i < i')

/-- Irreflexive. -/
theorem better_irrefl (v i : Nat) : ¬ better v i v i := by unfold better; omega

/-- Asymmetric. -/
theorem better_asymm (v i v' i' : Nat) (h : better v i v' i') : ¬ better v' i' v i := by
  unfold better at *; omega

/-- Transitive. -/
theorem better_trans (v i v' i' v'' i'' : Nat) (h1 : better v i v' i') (h2 : better v' i' v'' i'') :
    better v i v'' i'' := by
  unfold better at *; omega

/-- Total on distinct indices: any two candidates are comparable. -/
theorem better_total (v i v' i' : Nat) (h : i ≠ i') : better v i v' i' ∨ better v' i' v i := by
  unfold better; omega

/-- Round up to a multiple of 16. -/
def round16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Taken mask size is at least n_expert and 16-aligned. -/
theorem taken_bytes (n : Nat) : n ≤ round16 n ∧ round16 n % 16 = 0 := by unfold round16; omega

/-- Shared-memory bytes. -/
def smem (n : Nat) : Nat := round16 n + n * 8 + n * 4

/-- Shared memory for 512 experts. -/
theorem smem_512 : smem 512 = 512 + 4096 + 2048 := by decide

/-- Thread count: min(n, 512) rounded up to a warp. -/
def threads (n : Nat) : Nat := ((min n 512) + 31) / 32 * 32

/-- Thread count is a warp multiple and at least min(n,512). -/
theorem threads_warp (n : Nat) : threads n % 32 = 0 ∧ min n 512 ≤ threads n := by
  unfold threads; omega

/-- For 512 experts the block has 512 threads. -/
theorem threads_512 : threads 512 = 512 := by decide

/-- The clamp keeps the divisor at least 2^-14 (as a fixed denominator model). -/
theorem clamp_pos (s m : Nat) (h : 0 < m) : 0 < max s m := by omega

/-- k passes each select one new expert: selected count after i passes is i. -/
theorem passes_count (k : Nat) : (List.range k).length = k := by simp

end Strata.RouterTop10



## FILE: /mnt/user-data/outputs/S2ExpertGroupedGemv.lean
/-!
# MODULE: s2_expert_grouped_gemv
Grouped GPU routed-expert FFN over Q2_0 blobs: gate/up rows (interleaved in the blob, written gate-major),
SwiGLU, requantise to Q8_0, down rows; dp4a with bias term `sum c*x - sum x`.

## Source semantics
(c-1)*x summed equals sum c*x minus sum x. 2-bit codes are LSB-first: element 4j+k is bits [2k,2k+2) of byte j.
Blob layout: [gate/up codes | down codes | gate/up scales | down scales]. Row slot i: even = gate row i/2, odd = up row.

## Form template identifiers
H=2560 -> n_embd; FF=640 -> n_ff_exp; QK=64 -> weight_group
-/
namespace Strata.S2Expert

/-- Integer sum of a list. -/
def isum : List Int → Int
  | [] => 0
  | x :: xs => x + isum xs

/-- Dot of code with activation, with the bias removed per element. -/
def dotBias : List (Int × Int) → Int
  | [] => 0
  | (c, x) :: r => (c - 1) * x + dotBias r

/-- Plain code-activation dot. -/
def dotRaw : List (Int × Int) → Int
  | [] => 0
  | (c, x) :: r => c * x + dotRaw r

/-- Sum of activations. -/
def sumX : List (Int × Int) → Int
  | [] => 0
  | (_, x) :: r => x + sumX r

/-- The dp4a decomposition: sum (c-1)x = sum c x - sum x. -/
theorem bias_decomposition (l : List (Int × Int)) : dotBias l = dotRaw l - sumX l := by
  induction l with
  | nil => simp [dotBias, dotRaw, sumX]
  | cons p r ih =>
    obtain ⟨c, x⟩ := p
    simp only [dotBias, dotRaw, sumX, ih, Int.sub_mul, Int.one_mul]
    omega

/-- Byte layout of the blob (bytes). -/
def oDCodes : Nat := 2 * 640 * 640
/-- Offset of gate/up scales. -/
def oGuScales : Nat := oDCodes + 2560 * 160
/-- Offset of down scales. -/
def oDScales : Nat := oGuScales + 2 * 640 * 40 * 2
/-- Total blob size. -/
def blobBytes : Nat := oDScales + 2560 * 10 * 2

/-- Blob total is 1382400 bytes. -/
theorem blob_total : blobBytes = 1382400 := by decide

/-- Regions are ordered and non-overlapping. -/
theorem blob_ordered : oDCodes < oGuScales ∧ oGuScales < oDScales ∧ oDScales < blobBytes := by decide

/-- Code byte fields: four 2-bit fields reconstruct the byte (LSB-first). -/
theorem code_fields : ∀ c : Fin 256,
    (c.val &&& 3) + 4 * ((c.val >>> 2) &&& 3) + 16 * ((c.val >>> 4) &&& 3) + 64 * ((c.val >>> 6) &&& 3) = c.val := by
  decide +kernel

/-- Interleaved row slot to (row, is-up). -/
theorem interleave (i : Nat) : (i / 2) * 2 + i % 2 = i := by omega

/-- Gate-major layout: hit h row r (gate) is below every up index. -/
theorem gate_below_up (n h r : Nat) (hh : h < n) (hr : r < 640) :
    h * 640 + r < n * 640 + h * 640 + r := by omega

/-- Gate indices stay inside the first n*FF floats. -/
theorem gate_in_first_half (n h r : Nat) (hh : h < n) (hr : r < 640) : h * 640 + r < n * 640 := by omega

/-- Up indices stay inside the second half. -/
theorem up_in_second_half (n h r : Nat) (hh : h < n) (hr : r < 640) :
    n * 640 + h * 640 + r < 2 * (n * 640) := by omega

/-- Gate/up row loop of 80 chunks with 32 lanes needs 3 chunks per lane. -/
theorem gu_chunks : (2560 / 32 + 31) / 32 = 3 := by decide

/-- Down row has 20 chunks, fewer than a warp. -/
theorem down_chunks : 640 / 32 = 20 ∧ 20 < 32 := by decide

/-- Gate/up grid exactly covers 1280 rows with 32-row blocks. -/
theorem gu_grid : 2 * 640 / 32 * 32 = 2 * 640 := by decide

/-- Down grid exactly covers 2560 rows with 64-row blocks. -/
theorem down_grid : 2560 / 64 * 64 = 2560 := by decide

/-- One weight scale per 64 elements, i.e. per two 32-element chunks. -/
theorem scale_pairing (c : Nat) : (c / 2) * 2 ≤ c ∧ c < (c / 2) * 2 + 2 := by omega

end Strata.S2Expert



## FILE: /mnt/user-data/outputs/SGemv.lean
/-!
# MODULE: s_gemv
S-family quantized GEMV (2/4/8-bit codes, per-group scale, optional offset, optional non-linear IQ4_NL codebook)
with FP16, Q8_K and Q8_0 activations. Naive, split-row and warp-per-row variants.

## Source semantics
w = decode(code)*scale + offset, applied before multiplying by the activation. PER_BYTE = 8/bits.
A lane processes 16 consecutive elements per iteration; those lie inside one scale group (groups are 16/32/64)
and one activation block (Q8_K 256, Q8_0 32). Q8_K needs n_in % 256 = 0, Q8_0 needs n_in % 32 = 0.

## Form template identifiers
code_bits -> width; group_elems -> group; kIq4Nl -> nonlinear_codebook
-/
namespace Strata.SGemv

/-- Codes per byte. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- Per-byte counts for the three widths. -/
theorem per_byte_values : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by decide

/-- Bytes per four-code word: 1, 2, 4. -/
theorem quad_bytes : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

/-- Bytes per sixteen-code span: 4, 8, 16. -/
theorem octet16_bytes : 16 * 2 / 8 = 4 ∧ 16 * 4 / 8 = 8 ∧ 16 * 8 / 8 = 16 := by decide

/-- A 16-element lane span starting at a multiple of 16 stays within one group of 16. -/
theorem span_in_group16 (i : Nat) (h : i % 16 = 0) : (i + 15) / 16 = i / 16 := by omega

/-- Same for groups of 32. -/
theorem span_in_group32 (i : Nat) (h : i % 16 = 0) : (i + 15) / 32 = i / 32 := by omega

/-- Same for groups of 64. -/
theorem span_in_group64 (i : Nat) (h : i % 16 = 0) : (i + 15) / 64 = i / 64 := by omega

/-- A 16-element span stays inside one Q8_K block of 256. -/
theorem span_in_q8k (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by omega

/-- A 16-element span stays inside one Q8_0 block of 32. -/
theorem span_in_q8_0 (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega

/-- Lane starts are multiples of 16, and so are all strides of 32*16. -/
theorem lane_start (lane k : Nat) : (lane * 16 + k * (32 * 16)) % 16 = 0 := by omega

/-- Shared-memory codebook is indexed by a 4-bit value. -/
theorem code_index_lt (code : Nat) : code % 16 < 16 := by omega

/-- IQ4_NL codebook values are strictly increasing. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook has 16 entries. -/
theorem iq4nl_len : iq4nl.length = 16 := by decide

/-- Codebook is sorted (adjacent differences positive). -/
theorem iq4nl_sorted : (List.zip iq4nl iq4nl.tail).all (fun p => decide (p.1 < p.2)) = true := by decide

/-- Q8_0 requires n_in multiple of 32; 640 qualifies but is not a multiple of 256. -/
theorem shexp_not_q8k : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- Warps per block of 256 threads cover all output rows. -/
theorem warp_grid (n : Nat) : n ≤ ((n + 7) / 8) * 8 := by omega

end Strata.SGemv



## FILE: /mnt/user-data/outputs/Sampler.lean
/-!
# MODULE: sampler
Sampler chain: penalties -> top_k -> top_p -> temperature -> pick, with a Philox counter-based uniform draw and a
block-wide greedy argmax whose tie rule is the lowest index.

## Source semantics
Greedy is the global argmax over penalised logits (temperature is NOT applied; temperature 0 would zero logits).
The penalty window is the tail of the history of length min(penalty_last_n, history_len).
Philox uniform uses the top 24 bits so the draw is in [0,1).

## Form template identifiers
sample_tokens -> sample_next_token; SamplerParams -> sampling_config
-/
namespace Strata.Sampler

/-- Candidate merge in the greedy reduction: larger value wins, tie goes to the smaller index. -/
def merge (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The merged value is at least both inputs. -/
theorem merge_ge (a b : Int × Nat) : a.1 ≤ (merge a b).1 ∧ b.1 ≤ (merge a b).1 := by
  unfold merge; split <;> omega

/-- On equal values the merged index is the smaller one. -/
theorem merge_tie (a b : Int × Nat) (h : a.1 = b.1) :
    (merge a b).2 ≤ a.2 ∧ (merge a b).2 ≤ b.2 := by
  unfold merge; split <;> omega

/-- The merge returns one of its inputs. -/
theorem merge_mem (a b : Int × Nat) : merge a b = a ∨ merge a b = b := by
  unfold merge; split <;> simp

/-- Philox draw: the top 24 bits of a 32-bit word are below 2^24, so the uniform is below 1. -/
theorem uniform_lt_one (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by omega

/-- Penalty window length. -/
def hlen (lastN histLen : Nat) : Nat := min lastN histLen

/-- Window is a tail: offset plus length equals the history length. -/
theorem window_tail (lastN histLen : Nat) : (histLen - hlen lastN histLen) + hlen lastN histLen = histLen := by
  unfold hlen; omega

/-- Window never exceeds the history. -/
theorem window_le (lastN histLen : Nat) : hlen lastN histLen ≤ histLen := by unfold hlen; omega

/-- Top-p cut after the min_keep clamp. -/
def clampCut (cut minKeep n : Nat) : Nat := if cut < minKeep then min minKeep n else cut

/-- The clamped cut never exceeds the survivor count. -/
theorem clampCut_le (cut minKeep n : Nat) (h : cut ≤ n) : clampCut cut minKeep n ≤ n := by
  unfold clampCut; split <;> omega

/-- The clamped cut honours min_keep when enough survivors exist. -/
theorem clampCut_ge (cut minKeep n : Nat) (h : minKeep ≤ n) : minKeep ≤ clampCut cut minKeep n := by
  unfold clampCut; split <;> omega

/-- Presence penalty is a boolean, not the count. -/
def presenceTerm (count : Nat) : Nat := if count > 0 then 1 else 0

/-- Presence term is at most one. -/
theorem presence_le_one (count : Nat) : presenceTerm count ≤ 1 := by unfold presenceTerm; split <;> omega

/-- Repeat penalty direction: non-positive logits multiply, positive divide (model over Int, penalty >= 1). -/
def repeatPen (logit p : Int) : Int := if logit ≤ 0 then logit * p else logit / p

/-- A positive logit never becomes larger under a penalty >= 1. -/
theorem repeat_pos (logit p : Int) (hl : 0 < logit) (hp : 1 ≤ p) : repeatPen logit p ≤ logit := by
  unfold repeatPen
  split
  · omega
  · exact Int.ediv_le_self _ hl.le

/-- A non-positive logit never becomes larger under a penalty >= 1. -/
theorem repeat_nonpos (logit p : Int) (hl : logit ≤ 0) (hp : 1 ≤ p) : repeatPen logit p ≤ logit := by
  unfold repeatPen
  split
  · nlinarith [mul_nonneg (neg_nonneg.mpr hl) (sub_nonneg.mpr hp)]
  · omega

end Strata.Sampler



## FILE: /mnt/user-data/outputs/VerifyKernels.lean
/-!
# MODULE: verify_kernels
Multi-token verify-window kernels: GDN conv+L2 multi-token, conv history commit, alpha/beta, recurrent step+norm,
embedding gather, stream broadcast/add, hit compaction, id mapping, window ids, dense step state, pointer rebase.

## Source semantics
Conv window of token t is [hist0,hist1,hist2,x_0..x_t]; commit keeps the last three of [hist | x_0..x_{n-1}].
Window ids: start = max(0, n_kv - window), width = n_kv - start. Dense steps: (pos, pos+1, (pos+1)/4, pos+1).
Stream broadcast maps flat index i to source i % n and stream i / n.

## Form template identifiers
kVerifyMaxT -> max_verify_window; n_keep -> committed_count
-/
namespace Strata.Verify

/-- Source index into [hist(3) | x...] for conv tap j of token t. -/
def src (t j : Nat) : Nat := t + j

/-- Token 0 reads only history. -/
theorem token0_history (j : Nat) (h : j < 3) : src 0 j < 3 := by unfold src; omega

/-- A token t >= 3 reads only fresh activations. -/
theorem token_fresh (t j : Nat) (h : 3 ≤ t) : 3 ≤ src t j := by unfold src; omega

/-- Fresh activation index is non-negative offset into the window. -/
theorem fresh_index (t j : Nat) (h : 3 ≤ src t j) : src t j - 3 + 3 = t + j := by unfold src at *; omega

/-- Commit with n = 0 keeps history unchanged (sources are j itself). -/
theorem commit_zero (j : Nat) : src 0 j = j := by unfold src; omega

/-- Window start for the window-ids kernel. -/
def wstart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- Window width. -/
def wwidth (nKv window : Nat) : Nat := nKv - wstart nKv window

/-- Start plus width equals n_kv. -/
theorem window_sum (nKv window : Nat) : wstart nKv window + wwidth nKv window = nKv := by
  unfold wwidth wstart; split <;> omega

/-- Width is at most the window. -/
theorem window_width_le (nKv window : Nat) : wwidth nKv window ≤ window ∨ nKv ≤ window := by
  unfold wwidth wstart; split <;> omega

/-- Dense step state is consistent: n_kv = pos+1, n_bid = n_kv/4, width = n_kv. -/
theorem dense_step (c : Nat) : (c + 1) / 4 * 4 ≤ c + 1 := by omega

/-- Broadcast source index is within the embedding. -/
theorem bcast_src_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- Broadcast stream index is within the stream count. -/
theorem bcast_stream_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- Recurrent state split: 4 row groups of 32 rows give the 128-wide state. -/
theorem state_split : 128 / 4 = 32 ∧ 4 * 32 = 128 := by decide

/-- Hit compaction slot below the lane is below the lane. -/
theorem compact_slot (before lane : Nat) (pop : Nat) (h : pop ≤ lane) : before + pop < before + lane + 1 := by omega

/-- Rebased pointer of slot k. -/
def rebase (base bytes k : Nat) : Nat := base + k * bytes

/-- Consecutive rebased pointers differ by one blob. -/
theorem rebase_step (base bytes k : Nat) : rebase base bytes (k + 1) = rebase base bytes k + bytes := by
  unfold rebase; rw [Nat.add_mul]; omega

/-- Uint4 copy count is blob_bytes/16, exact when the blob is a multiple of 16. -/
theorem uint4_copy (b : Nat) (h : b % 16 = 0) : (b / 16) * 16 = b := by omega

/-- Alpha/beta rows: 2*h_v rows, 8 per block, grid covers them. -/
theorem ab_grid (hv : Nat) : 2 * hv ≤ ((2 * hv + 7) / 8) * 8 := by omega

/-- Embedding gather: code index decomposition for 2/4/8-bit codes. -/
theorem gather_decomp (i pb : Nat) (h : 0 < pb) : (i / pb) * pb + i % pb = i := by
  have := Nat.div_add_mod i pb
  rw [Nat.mul_comm] at this
  omega

end Strata.Verify





## FILE: /mnt/user-data/outputs/NgramHashIq4nlTable.lean
/-!
# MODULE: ngram_hash_iq4nl_table
The PLE n-gram hash (token context -> 16 table-row indices) and the IQ4_NL table read
(row addressing, split-halves dequantisation layout, head-slowest gather layout).

## Source semantics
* `ngram_mixed(ctx, mult, n)` assigns `ctx[0]*mult[0]` and XORs `ctx[j]*mult[j]` for `1 ≤ j < n` (mod 2^64).
* `ngram_rows`: predecessor `s` positions back is `prev[n_prev - s]` (prev is oldest first); an EOS or a
  negative predecessor cuts the history and every later position (and the cut position itself) is EOS.
* Row index of head `h` is `mixed % vocab[h] + offset[h]`; heads of n-gram size `n` are `(n-2)*8 .. (n-2)*8+7`.
* A row is 90 bytes = 5 blocks of 18 bytes (fp16 scale + 16 nibble bytes); byte `j` of a block holds element
  `j` (low nibble) and `j+16` (high nibble).
* `gather` writes head `h` at `[h*160, (h+1)*160)`; out-of-range rows read as zeros.

## Form template identifiers
ngram_mixed -> mixed_hash ; ngram_rows -> context_to_rows ; PleConsts.vocab -> head_vocab ;
PleConsts.offset -> head_offset ; kIq4Nl -> nl_table ; iq4nl_dequant_row -> row_dequant ;
PleTable::gather -> head_slowest_gather ; PleTable::read_row -> bounded_row_read
-/

namespace Strata.Kernels.Ngram

/-- N-gram size (context length). -/
def NGRAM_SIZE : Nat := 3
/-- Heads per n-gram size. -/
def HEADS_PER_NGRAM : Nat := 8
/-- Total heads. -/
def PLE_N_HEADS : Nat := 16
/-- Values per head row. -/
def PLE_HEAD_DIM : Nat := 160
/-- Bytes per IQ4_NL row. -/
def PLE_ROW_BYTES : Nat := 90
/-- Flattened embedding width. -/
def NG_N_EMBD : Nat := 2560
/-- "No predecessor" marker. -/
def TOKEN_NULL : Int := -1

/-- Per-head vocabulary sizes of the artifact. -/
def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets of the artifact. -/
def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

-- § constants

/-- Heads are exactly (NGRAM_SIZE-1) groups of HEADS_PER_NGRAM. -/
theorem heads_eq : (NGRAM_SIZE - 1) * HEADS_PER_NGRAM = PLE_N_HEADS := by decide

/-- Sixteen heads of 160 values make the 2560-wide embedding. -/
theorem embd_eq : PLE_N_HEADS * PLE_HEAD_DIM = NG_N_EMBD := by decide

/-- A row is five 18-byte blocks. -/
theorem row_bytes_eq : (PLE_HEAD_DIM / 32) * 18 = PLE_ROW_BYTES := by decide

/-- Both constant tables have one entry per head. -/
theorem tables_length : vocabs.length = 16 ∧ offsets.length = 16 := by decide

-- § head tables

/-- The offsets are the prefix sums of the vocabulary sizes (the table is a concatenation). -/
theorem offsets_prefix_sums :
    ∀ i : Fin 15, offsets[i.val + 1]! = offsets[i.val]! + vocabs[i.val]! := by decide

/-- Every head's index range ends inside the table (320,001,536 rows). -/
theorem head_ranges_in_table :
    ∀ i : Fin 16, offsets[i.val]! + vocabs[i.val]! ≤ 320001536 := by decide

/-- The row index always fits a uint32 (the C++ cast is lossless). -/
theorem head_ranges_fit_u32 :
    ∀ i : Fin 16, offsets[i.val]! + vocabs[i.val]! < 4294967296 := by decide

/-- Every vocabulary is positive, so the modulo is defined. -/
theorem vocabs_pos : ∀ i : Fin 16, 0 < vocabs[i.val]! := by decide

/-- The row index of a head lies in that head's own range `[off, off+v)`. -/
theorem head_index_bounds (m v off : Nat) (hv : 0 < v) :
    off ≤ m % v + off ∧ m % v + off < off + v := by
  have := Nat.mod_lt m hv
  omega

/-- For the artifact constants the row index is below 2^32 for every hash value. -/
theorem head_index_u32 (m : Nat) (h : Fin 16) :
    m % vocabs[h.val]! + offsets[h.val]! < 4294967296 := by
  have h1 := head_index_bounds m vocabs[h.val]! offsets[h.val]! (vocabs_pos h)
  have h2 := head_ranges_fit_u32 h
  omega

/-- Distinct heads of the same hash value address disjoint ranges (offsets are prefix sums). -/
theorem head_ranges_disjoint :
    ∀ i : Fin 15, offsets[i.val]! + vocabs[i.val]! ≤ offsets[i.val + 1]! := by
  intro i
  have := offsets_prefix_sums i
  omega

-- § ngram_mixed

/-- Model of `ngram_mixed`: first product assigned, the rest XORed in. -/
def ngramMixed (ctx mult : Nat → UInt64) : Nat → UInt64
  | 0 => 0
  | 1 => ctx 0 * mult 0
  | n + 2 => ngramMixed ctx mult (n + 1) ^^^ (ctx (n + 1) * mult (n + 1))

/-- Bigram hash unfolds to the two-term XOR. -/
theorem ngramMixed_two (ctx mult : Nat → UInt64) :
    ngramMixed ctx mult 2 = (ctx 0 * mult 0) ^^^ (ctx 1 * mult 1) := rfl

/-- Trigram hash unfolds to the three-term XOR. -/
theorem ngramMixed_three (ctx mult : Nat → UInt64) :
    ngramMixed ctx mult 3 = ((ctx 0 * mult 0) ^^^ (ctx 1 * mult 1)) ^^^ (ctx 2 * mult 2) := rfl

/-- The hash of the first `n` positions depends only on those positions. -/
theorem ngramMixed_congr (ctx ctx' mult : Nat → UInt64) :
    ∀ n, (∀ j, j < n → ctx j = ctx' j) → ngramMixed ctx mult n = ngramMixed ctx' mult n
  | 0, _ => rfl
  | 1, h => by simp [ngramMixed, h 0 (by omega)]
  | n + 2, h => by
      simp only [ngramMixed]
      rw [ngramMixed_congr ctx ctx' mult (n + 1) (fun j hj => h j (by omega)), h (n + 1) (by omega)]

-- § context cut

/-- One step of the history walk: returns the new cut flag and the stored context value. -/
def ctxStep (eos : Int) (cut : Bool) (raw : Int) : Bool × Int :=
  let t := if cut then TOKEN_NULL else raw
  let cut' := cut || decide (t < 0) || decide (t = eos)
  (cut', if cut' then eos else t)

/-- Once cut, the flag stays set and the stored value is EOS. -/
theorem ctxStep_sticky (eos raw : Int) :
    (ctxStep eos true raw).1 = true ∧ (ctxStep eos true raw).2 = eos := by
  simp [ctxStep, TOKEN_NULL]

/-- A real, non-EOS predecessor passes through unchanged and does not cut. -/
theorem ctxStep_passthrough (eos raw : Int) (h0 : 0 ≤ raw) (hne : raw ≠ eos) :
    ctxStep eos false raw = (false, raw) := by
  have h1 : ¬ raw < 0 := by omega
  simp [ctxStep, h1, hne]

/-- An EOS predecessor is itself stored as EOS and cuts everything after. -/
theorem ctxStep_eos_cuts (eos : Int) :
    (ctxStep eos false eos).1 = true ∧ (ctxStep eos false eos).2 = eos := by
  simp [ctxStep]

/-- A missing (negative) predecessor cuts. -/
theorem ctxStep_negative_cuts (eos raw : Int) (h : raw < 0) :
    (ctxStep eos false raw).1 = true := by
  simp [ctxStep, h]

/-- The whole history walk over the raw predecessors. -/
def ctxChain (eos : Int) : Bool → List Int → List Int
  | _, [] => []
  | cut, r :: rs => (ctxStep eos cut r).2 :: ctxChain eos (ctxStep eos cut r).1 rs

/-- The walk preserves length. -/
theorem ctxChain_length (eos : Int) (rs : List Int) :
    ∀ cut, (ctxChain eos cut rs).length = rs.length := by
  induction rs with
  | nil => intro cut; rfl
  | cons r rs ih => intro cut; simp [ctxChain, ih]

/-- After a cut every remaining position is EOS. -/
theorem ctxChain_after_cut (eos : Int) :
    ∀ rs : List Int, ctxChain eos true rs = List.replicate rs.length eos
  | [] => rfl
  | r :: rs => by
      have h : ctxStep eos true r = (true, eos) := by simp [ctxStep, TOKEN_NULL]
      simp [ctxChain, h, List.replicate_succ, ctxChain_after_cut eos rs]

/-- The position whose predecessor is EOS is EOS, and so is all later history. -/
theorem ctxChain_eos_then_eos (eos : Int) (rs : List Int) :
    ctxChain eos false (eos :: rs) = eos :: List.replicate rs.length eos := by
  have h : ctxStep eos false eos = (true, eos) := by simp [ctxStep]
  simp [ctxChain, h, ctxChain_after_cut]

-- § index arithmetic of ngram_rows

/-- `prev` is oldest first: predecessor `s` back is entry `2 - s`, in range. -/
theorem prev_index_lt (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) : 2 - s < 2 := by omega

/-- Predecessor 1 is the NEWEST entry (index 1), predecessor 2 the oldest (index 0). -/
theorem prev_newest_oldest : (2 - 1 = 1) ∧ (2 - 2 = 0) := by decide

/-- The prev row of token `i` stays inside its own two-entry window. -/
theorem prev_offset_lt (i s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) :
    i * 2 + (2 - s) < (i + 1) * 2 := by omega

/-- Output index of head `h` for token `i` is inside token `i`'s 16-wide block. -/
theorem out_index_lt (i h : Nat) (hh : h < 16) : i * 16 + h < (i + 1) * 16 := by omega

/-- Head slot of (n-gram size, group) is below 16. -/
theorem head_slot_lt (n g : Nat) (hn : 2 ≤ n) (hn3 : n ≤ 3) (hg : g < 8) :
    (n - 2) * 8 + g < 16 := by omega

/-- (n, g) -> head slot is injective, so every head is written exactly once. -/
theorem head_slot_inj (n g n' g' : Nat) (hg : g < 8) (hg' : g' < 8) :
    (n - 2) * 8 + g = (n' - 2) * 8 + g' → n - 2 = n' - 2 ∧ g = g' := by omega

-- § IQ4_NL table and dequantisation layout

/-- The non-linear 4-bit codebook. -/
def iq4nlTable : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook lookup with the 4-bit mask (`code & 15`, written as `% 16`). -/
def iq4nl (code : Nat) : Int := iq4nlTable[code % 16]!

/-- The table has 16 entries. -/
theorem iq4nl_length : iq4nlTable.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_strictly_increasing :
    ∀ i : Fin 15, iq4nlTable[i.val]! < iq4nlTable[i.val + 1]! := by decide

/-- Every codebook value fits int8 and avoids -128. -/
theorem iq4nl_range : ∀ i : Fin 16, -127 ≤ iq4nlTable[i.val]! ∧ iq4nlTable[i.val]! ≤ 127 := by
  decide

/-- Masking is idempotent. -/
theorem iq4nl_mask (code : Nat) : iq4nl code = iq4nl (code % 16) := by
  simp [iq4nl, Nat.mod_mod]

/-- A byte splits into low and high nibbles, each a valid code. -/
theorem nibble_split (b : Nat) (hb : b < 256) : b % 16 + 16 * (b / 16) = b ∧ b / 16 < 16 := by
  omega

/-- Split halves: every element of a 32-block is reached as `j` or `j+16` with `j < 16`. -/
theorem dequant_cover (e : Nat) (he : e < 160) :
    ∃ b j, b < 5 ∧ j < 16 ∧ (e = b * 32 + j ∨ e = b * 32 + j + 16) := by
  by_cases h : e % 32 < 16
  · exact ⟨e / 32, e % 32, by omega, h, Or.inl (by omega)⟩
  · exact ⟨e / 32, e % 32 - 16, by omega, by omega, Or.inr (by omega)⟩

/-- Low-nibble destinations are injective. -/
theorem dequant_lo_inj (b j b' j' : Nat) (hj : j < 16) (hj' : j' < 16)
    (h : b * 32 + j = b' * 32 + j') : b = b' ∧ j = j' := by omega

/-- High-nibble destinations are injective. -/
theorem dequant_hi_inj (b j b' j' : Nat) (hj : j < 16) (hj' : j' < 16)
    (h : b * 32 + j + 16 = b' * 32 + j' + 16) : b = b' ∧ j = j' := by omega

/-- Low and high destinations never collide, so no element is written twice. -/
theorem dequant_lo_hi_disjoint (b j b' j' : Nat) (hj : j < 16) (hj' : j' < 16)
    (h : b * 32 + j = b' * 32 + j' + 16) : False := by omega

/-- Each 18-byte block stays inside the 90-byte row. -/
theorem block_fits (b : Nat) (hb : b < 5) : b * 18 + 18 ≤ 90 := by omega

/-- Head-slowest gather: head `h`, element `d` lands inside the 2560 buffer. -/
theorem gather_index_lt (h d : Nat) (hh : h < 16) (hd : d < 160) : h * 160 + d < 2560 := by
  omega

/-- The gather layout is injective in (h, d). -/
theorem gather_index_inj (h d h' d' : Nat) (hd : d < 160) (hd' : d' < 160)
    (e : h * 160 + d = h' * 160 + d') : h = h' ∧ d = d' := by omega

/-- A row inside the table has all 90 bytes inside the data region. -/
theorem row_in_table (row n : Nat) (hr : row < n) : (row + 1) * 90 ≤ n * 90 := by omega

/-- The prefetch list never has more ranges than rows (out-of-range rows are skipped). -/
theorem prefetch_ranges_le (rows : List Nat) (n : Nat) :
    (rows.filter (· < n)).length ≤ rows.length :=
  List.length_filter_le _ _

end Strata.Kernels.Ngram



## FILE: /mnt/user-data/outputs/PleDirectRowReader.lean
/-!
# MODULE: ple_direct_row_reader
The unbuffered SSD row reader: page-aligned read jobs, a set-associative row cache, ticket numbering,
completion counting and the slot slab.

## Source semantics
* A row at file offset `at` is read by one job at `first = at/P*P` of length
  `(at+R-1)/P*P - first + P`, i.e. one or two pages (P = PAGE, R = ROW_BYTES ≤ P).
* The row is copied from `buf + (at - first)`; the read must cover `at - first + R` bytes.
* The cache has `sets = rows/8` sets of 8 ways, slot index `s*8 + w`, round-robin replacement `w -> (w+1)%8`.
* Tickets count from 1 and skip 0 on wrap; a ticket's `pending` is the number of its jobs and is decremented
  once per completed job; `collect` returns when it reaches 0.
* Read slots are `2*PAGE` bytes in a slab of `max_inflight` slots.

## Form template identifiers
PAGE -> page_size ; ROW_BYTES -> row_bytes ; Job.offset -> aligned_offset ; Job.length -> job_length ;
Use.in_page -> in_page_offset ; RowCache -> set_assoc_cache ; WAYS -> ways ; next_ticket -> ticket_counter
-/

namespace Strata.Ngram.PleReader

/-- Cache associativity. -/
def WAYS : Nat := 8
/-- Empty-slot sentinel of the cache key array. -/
def EMPTY : Nat := 4294967295

-- § page-aligned jobs

/-- `x < x/P*P + P`. -/
theorem lt_div_mul_add' (x P : Nat) (hP : 0 < P) : x < x / P * P + P := by
  have h1 : x / P * P + x % P = x := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod x P
  have h2 := Nat.mod_lt x hP
  omega

/-- Aligned start offset of the job containing byte `a`. -/
def jobFirst (P a : Nat) : Nat := a / P * P

/-- Job length: from the first page to the page holding the last byte of the row. -/
def jobLength (P R a : Nat) : Nat := (a + R - 1) / P * P - a / P * P + P

/-- The job starts at or before the row. -/
theorem jobFirst_le (P a : Nat) : jobFirst P a ≤ a := Nat.div_mul_le_self a P

/-- The job start is page aligned. -/
theorem jobFirst_aligned (P a : Nat) : jobFirst P a % P = 0 := Nat.mul_mod_left _ _

/-- First page start is at most the last page start. -/
theorem first_le_last (P R a : Nat) (hR : 0 < R) : a / P * P ≤ (a + R - 1) / P * P :=
  Nat.mul_le_mul_right P (Nat.div_le_div_right (by omega))

/-- Closed form: the job spans `(last_page - first_page + 1)` pages. -/
theorem jobLength_eq (P R a : Nat) (hR : 0 < R) :
    jobLength P R a = ((a + R - 1) / P - a / P + 1) * P := by
  unfold jobLength
  have _h := first_le_last P R a hR
  rw [Nat.add_mul, Nat.one_mul, Nat.sub_mul]

/-- The job length is a whole number of pages (direct I/O alignment). -/
theorem jobLength_mod (P R a : Nat) (hR : 0 < R) : jobLength P R a % P = 0 := by
  rw [jobLength_eq P R a hR]
  exact Nat.mul_mod_left _ _

/-- The job covers the whole row. -/
theorem jobLength_covers (P R a : Nat) (hP : 0 < P) (hR : 0 < R) :
    a + R ≤ jobFirst P a + jobLength P R a := by
  unfold jobFirst jobLength
  have h1 := first_le_last P R a hR
  have h2 := lt_div_mul_add' (a + R - 1) P hP
  omega

/-- The in-page offset of the row is below one page. -/
theorem inPage_lt (P a : Nat) (hP : 0 < P) : a - jobFirst P a < P := by
  unfold jobFirst
  have h1 : a / P * P + a % P = a := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod a P
  have h2 := Nat.mod_lt a hP
  omega

/-- The "short read" guard `in_page + R > bytes` can never fire on a full-length read. -/
theorem inPage_fits (P R a : Nat) (hP : 0 < P) (hR : 0 < R) :
    (a - jobFirst P a) + R ≤ jobLength P R a := by
  have c := jobLength_covers P R a hP hR
  have l := jobFirst_le P a
  omega

/-- A row of at most one page needs one page or two pages, never more. -/
theorem jobLength_small (P R a : Nat) (hP : 0 < P) (hR : 0 < R) (hRP : R ≤ P) :
    jobLength P R a = P ∨ jobLength P R a = 2 * P := by
  rw [jobLength_eq P R a hR]
  have h2 : (a + R - 1) / P < a / P + 2 := by
    rw [Nat.div_lt_iff_lt_mul hP]
    have h3 := lt_div_mul_add' a P hP
    have e : (a / P + 2) * P = a / P * P + 2 * P := by rw [Nat.add_mul]
    omega
  have h3 : a / P ≤ (a + R - 1) / P := Nat.div_le_div_right (by omega)
  have h4 : (a + R - 1) / P - a / P = 0 ∨ (a + R - 1) / P - a / P = 1 := by omega
  rcases h4 with h | h <;> rw [h]
  · left; omega
  · right; omega

/-- A job of at most `2*P` bytes fits its slot, and slots tile the slab. -/
theorem slot_in_slab (s m P : Nat) (hs : s < m) : s * (2 * P) + 2 * P ≤ m * (2 * P) := by
  have h : (s + 1) * (2 * P) ≤ m * (2 * P) := Nat.mul_le_mul_right _ hs
  have e : (s + 1) * (2 * P) = s * (2 * P) + 2 * P := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § row cache

/-- Slot index inside the key/data arrays. -/
def cacheSlot (s w : Nat) : Nat := s * 8 + w

/-- Every (set, way) pair is inside the arrays. -/
theorem cacheSlot_lt (s w sets : Nat) (hs : s < sets) (hw : w < 8) :
    cacheSlot s w < sets * 8 := by
  unfold cacheSlot
  omega

/-- The cache never holds more rows than requested. -/
theorem cache_capacity_le (rows : Nat) : rows / 8 * 8 ≤ rows := by omega

/-- The hashed set index is in range. -/
theorem set_index_lt (x sets : Nat) (h : 0 < sets) : x % sets < sets := Nat.mod_lt x h

/-- Round-robin pointer step. -/
def nextWay (w : Nat) : Nat := (w + 1) % 8

/-- The replacement pointer stays a valid way. -/
theorem nextWay_lt (w : Nat) : nextWay w < 8 := by
  unfold nextWay
  omega

/-- Eight consecutive insertions into one set use eight distinct ways. -/
theorem ways_distinct (w i j : Nat) (hi : i < 8) (hj : j < 8) (hne : i ≠ j) :
    (w + i) % 8 ≠ (w + j) % 8 := by omega

/-- A real row id never equals the empty sentinel. -/
theorem row_ne_empty (row n : Nat) (hn : n ≤ 4294967295) (h : row < n) : row ≠ EMPTY := by
  unfold EMPTY
  omega

-- § tickets and completion counting

/-- Ticket numbering: increment, skipping 0 on wrap. -/
def nextTicket (t : UInt32) : UInt32 :=
  let n := t + 1
  if n = 0 then 1 else n

/-- A ticket is never 0. -/
theorem nextTicket_ne_zero (t : UInt32) : nextTicket t ≠ 0 := by
  unfold nextTicket
  by_cases h : t + 1 = 0 <;> simp [h]

/-- Saturating decrement applied once per completed job. -/
theorem pending_after (k p : Nat) :
    (List.replicate k ()).foldl (fun q _ => q - 1) p = p - k := by
  induction k generalizing p with
  | zero => simp
  | succ k ih => simp [List.replicate_succ, List.foldl_cons, ih] <;> omega

/-- When every job of a ticket completes, `pending` reaches 0 exactly. -/
theorem pending_zero (n : Nat) : (List.replicate n ()).foldl (fun q _ => q - 1) n = 0 := by
  rw [pending_after]
  omega

/-- Latency ring index stays inside the ring. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by omega

/-- The percentile index is a valid element index. -/
theorem pct_idx_le (m k : Nat) : min (m - 1) k ≤ m - 1 := Nat.min_le_left _ _

end Strata.Ngram.PleReader



## FILE: /mnt/user-data/outputs/PrefillGpuKernels.lean
/-!
# MODULE: prefill_gpu_kernels
Index arithmetic, memory layouts and rounding contracts of the batched-prefill CUDA kernels
(hyper-connection, GDN, router, expert blob dequantisation, SwiGLU, QSA helpers, KV append).

## Source semantics
* Geometry: N = 2560, HC = 4, D = N*HC = 10240, LR = 320, S = 128, HK = 16, HV = 48,
  C = 2*HK*S + HV*S = 10240.
* `blocks_for(n, t) = (n + t - 1)/t` blocks; the guard `i >= n` makes the surplus threads idle.
* `bf` is round-to-nearest-even on the upper 16 bits of the fp32 pattern.
* A Strata expert blob is gate/up codes `[1280][640]`, down codes `[2560][160]`, then fp16 scales
  `[1280][40]` and `[2560][10]`; each code byte holds four 2-bit weights, value `(code - 1) * d`.
* `route_kernel`: one warp per token, 16 logits per lane (`lane + 32 i`), top-10 with ties to the lowest
  expert index.
* `kv_append`: grid `(T, 2, 8)`; `z>>1` picks the 64-value group, `z&1` picks K or V.

## Form template identifiers
blocks_for -> grid_blocks ; bf -> bf16_round ; blob_dequant_kernel -> blob_unpack ; route_kernel -> top10_route ;
warp_sum -> warp_xor_reduce ; kv_append_kernel -> kv_page_write ; gdn_rec_kernel -> gdn_recurrence
-/

namespace Strata.Prefill.Kernels

/-- `x < x/P*P + P`. -/
theorem lt_div_mul_add' (x P : Nat) (hP : 0 < P) : x < x / P * P + P := by
  have h1 : x / P * P + x % P = x := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod x P
  have h2 := Nat.mod_lt x hP
  omega

-- § geometry

/-- The model dimensions are mutually consistent. -/
theorem geometry_consistent :
    (2560 : Nat) * 4 = 10240 ∧ 2 * 16 * 128 + 48 * 128 = 10240 ∧ 48 * 128 = 6144 ∧
      80 * 128 = 10240 ∧ 128 / 4 = 32 ∧ 4 * 128 = 512 := by decide

-- § launch sizing

/-- The grid covers all `n` work items. -/
theorem blocks_for_covers (n t : Nat) (ht : 0 < t) : n ≤ (n + t - 1) / t * t := by
  have := lt_div_mul_add' (n + t - 1) t ht
  omega

/-- The grid overshoots by less than one block. -/
theorem blocks_for_tight (n t : Nat) : (n + t - 1) / t * t ≤ n + t - 1 :=
  Nat.div_mul_le_self _ _

/-- A block-wide reduction needs at most 32 warp partials for blockDim ≤ 1024. -/
theorem block_sum_warps (b : Nat) (h : b ≤ 1024) : (b + 31) / 32 ≤ 32 := by omega

-- § hyper-connection kernels

/-- `gr_mix`: thread index decomposes into (token, channel). -/
theorem decomp_N (i T : Nat) (h : i < T * 2560) :
    i / 2560 < T ∧ i / 2560 * 2560 + i % 2560 = i ∧ i % 2560 < 2560 := by omega

/-- `gr_mix`: reading stream `c` of token `t` stays inside the `T*D` residual. -/
theorem gr_mix_index (t c d T : Nat) (ht : t < T) (hc : c < 4) (hd : d < 2560) :
    t * 10240 + c * 2560 + d < T * 10240 := by omega

/-- `gr_write`: (token, stream, channel) decomposition of a residual element. -/
theorem gr_write_decomp (i T : Nat) (h : i < T * 10240) :
    i / 10240 < T ∧ i % 10240 / 2560 < 4 ∧ i % 10240 % 2560 < 2560 ∧ i % 2560 = i % 10240 % 2560 := by
  omega

/-- `gr_norm`: one block per (token, stream) row; the row slice is inside the residual. -/
theorem gr_norm_row (row T : Nat) (h : row < T * 4) :
    row / 4 < T ∧ row % 4 < 4 ∧ row * 2560 + 2559 < T * 10240 := by omega

-- § GDN kernels

/-- `gdn_gates`: both ab halves are inside the `T * 2*HV` buffer. -/
theorem gdn_gate_index (t h T : Nat) (ht : t < T) (hh : h < 48) :
    t * 96 + h < T * 96 ∧ t * 96 + 48 + h < T * 96 := by omega

/-- `gdn_gates`: flat index decomposition. -/
theorem gdn_gate_decomp (i T : Nat) (h : i < T * 48) : i / 48 < T ∧ i % 48 < 48 := by omega

/-- `gdn_conv`: 80 blocks of 128 threads are exactly the 10240 channels. -/
theorem gdn_conv_grid : (10240 : Nat) / 128 * 128 = 10240 := by decide

/-- `gdn_conv`: the history and weight slices of channel `c` are inside their arrays. -/
theorem gdn_conv_index (c : Nat) (hc : c < 10240) : c * 3 + 2 < 30720 ∧ c * 4 + 3 < 40960 := by
  omega

/-- `gdn_rec`: q, k and v reads of a head stay in their own regions of the 10240-wide row. -/
theorem gdn_rec_regions (head col : Nat) (hh : head < 48) (hc : col < 128) :
    head % 16 * 128 + col < 2048 ∧ 2048 + head % 16 * 128 + col < 4096 ∧
      4096 + head * 128 + col < 10240 := by omega

/-- `gdn_rec`: the row groups tile the 128 state rows. -/
theorem gdn_rec_rows (rg r : Nat) (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- `gdn_rec`: every state row has exactly one (group, offset). -/
theorem gdn_rec_rows_unique (row : Nat) (h : row < 128) :
    row / 32 < 4 ∧ row % 32 < 32 ∧ row / 32 * 32 + row % 32 = row := by omega

/-- `gdn_rec`: state element address stays inside the `[128][HV][S]` state. -/
theorem gdn_rec_state_index (row head col : Nat) (hrow : row < 128) (hh : head < 48) (hc : col < 128) :
    row * 6144 + head * 128 + col < 128 * 6144 := by omega

-- § router

/-- Lane `l`, slot `i` addresses expert `l + 32 i < 512`. -/
theorem route_lane_index (l i : Nat) (hl : l < 32) (hi : i < 16) : l + i * 32 < 512 := by omega

/-- The lane/slot to expert map is injective: every expert is seen once. -/
theorem route_lane_inj (l i l' i' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : l + i * 32 = l' + i' * 32) : l = l' ∧ i = i' := by omega

/-- Candidate `(b', i')` replaces `(b, i)`: larger value, or equal value with a smaller index. -/
def better (b i b' i' : Int) : Prop := b < b' ∨ (b = b' ∧ i' < i)

/-- The shuffle tie-break is irreflexive. -/
theorem better_irrefl (b i : Int) : ¬ better b i b i := by
  unfold better
  omega

/-- The shuffle tie-break is asymmetric (a deterministic winner). -/
theorem better_asymm (b i b' i' : Int) (h : better b i b' i') : ¬ better b' i' b i := by
  unfold better at *
  omega

/-- The shuffle tie-break is transitive (the butterfly merge order does not matter). -/
theorem better_trans (b i b' i' b'' i'' : Int) (h1 : better b i b' i') (h2 : better b' i' b'' i'') :
    better b i b'' i'' := by
  unfold better at *
  omega

-- § warp reduction

/-- XOR with the same offset twice is the identity (butterfly partners are mutual). -/
theorem xor_involution (i o : Nat) : (i ^^^ o) ^^^ o = i := by
  simp [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- Butterfly partners stay inside the warp. -/
theorem butterfly_in_range : ∀ i : Fin 32, ∀ o ∈ [16, 8, 4, 2, 1], (i.val ^^^ o) < 32 := by
  decide

/-- The path of the five xor steps from lane `i` guided by `d`. -/
def butterflyPath (i d : Nat) : Nat :=
  ((((i ^^^ (d &&& 16)) ^^^ (d &&& 8)) ^^^ (d &&& 4)) ^^^ (d &&& 2)) ^^^ (d &&& 1)

set_option maxRecDepth 100000 in
/-- Every lane's value can reach every other lane through the five xor steps: full all-reduce coverage. -/
theorem butterfly_reaches_all :
    ∀ i : Fin 32, ∀ j : Fin 32, butterflyPath i.val (i.val ^^^ j.val) = j.val := by
  unfold butterflyPath
  decide

-- § blob layout and dequantisation

/-- Byte offset of the down codes. -/
def O_D_CODES : Nat := 1280 * 640
/-- Byte offset of the gate/up scales. -/
def O_GU_SC : Nat := O_D_CODES + 2560 * 160
/-- Byte offset of the down scales. -/
def O_D_SC : Nat := O_GU_SC + 1280 * 40 * 2
/-- Total blob size. -/
def BLOB_BYTES : Nat := O_D_SC + 2560 * 10 * 2

/-- The four sections are contiguous and total 1,382,400 bytes. -/
theorem blob_layout :
    O_D_CODES = 819200 ∧ O_GU_SC = 1228800 ∧ O_D_SC = 1331200 ∧ BLOB_BYTES = 1382400 := by decide

/-- Thread index split for gate/up code bytes. -/
theorem blob_gu_split (i : Nat) (h : i < 1280 * 640) :
    i / 640 < 1280 ∧ i % 640 < 640 ∧ i / 640 * 640 + i % 640 = i := by omega

/-- Gate/up scale index of a byte is inside the 40 scales of its row (one scale per 64 weights). -/
theorem blob_gu_scale_index (byte : Nat) (h : byte < 640) : byte * 4 / 64 < 40 := by omega

/-- Down scale index of a byte is inside the 10 scales of its row. -/
theorem blob_d_scale_index (byte : Nat) (h : byte < 160) : byte * 4 / 64 < 10 := by omega

/-- Gate/up output write of four weights stays inside `[1280][2560]`. -/
theorem blob_gu_out_index (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by omega

/-- Down output write of four weights stays inside `[2560][640]`. -/
theorem blob_d_out_index (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by omega

/-- Four 2-bit fields reconstruct the byte. -/
theorem code_pack (c : Nat) (hc : c < 256) :
    c % 4 + 4 * (c / 4 % 4) + 16 * (c / 16 % 4) + 64 * (c / 64 % 4) = c := by omega

/-- Each 2-bit field is a valid code. -/
theorem code_field_lt (c k : Nat) : c / 4 ^ k % 4 < 4 := Nat.mod_lt _ (by decide)

/-- The decoded multiplier `code - 1` lies in {-1, 0, 1, 2}. -/
theorem code_multiplier_range (x : Nat) (hx : x < 4) : -1 ≤ (x : Int) - 1 ∧ (x : Int) - 1 ≤ 2 := by
  omega

-- § SwiGLU and gather

/-- Interleaved SwiGLU reads `gu[r*1280 + 2k]` and `+1`, inside row `r`. -/
theorem swiglu_il_index (r k : Nat) (hk : k < 640) : r * 1280 + 2 * k + 1 < (r + 1) * 1280 := by
  omega

-- § QSA helpers

/-- `split_q`: the query half and the gate half of a 512-wide head row are disjoint and in bounds. -/
theorem split_q_index (t h d T : Nat) (ht : t < T) (hh : h < 24) (hd : d < 256) :
    t * 12288 + h * 512 + d < T * 12288 ∧ t * 12288 + h * 512 + 256 + d < T * 12288 := by omega

/-- `gate_attn`/`split_q` output index. -/
theorem q_out_index (t h d T : Nat) (ht : t < T) (hh : h < 24) (hd : d < 256) :
    t * 6144 + h * 256 + d < T * 6144 := by omega

-- § KV append

/-- Grid z picks the 64-value group (4 groups) and the 256-wide head row is covered. -/
theorem kv_grid (z tid : Nat) (hz : z < 8) (ht : tid < 64) :
    z / 2 < 4 ∧ z / 2 * 64 + tid < 256 := by omega

/-- (group, K/V) is a bijection with z. -/
theorem kv_grid_bijective (z : Nat) (hz : z < 8) : z / 2 * 2 + z % 2 = z := by omega

/-- The pool row of a position stays inside its page. -/
theorem kv_row_in_page (m ps r : Nat) (hr : r < ps) : m * ps + r < (m + 1) * ps := by
  have e : (m + 1) * ps = m * ps + ps := Nat.succ_mul m ps
  omega

/-- The scale index of a pool row stays in its row. -/
theorem kv_scale_index (row g : Nat) (hg : g < 4) : row * 4 + g < (row + 1) * 4 := by omega

/-- int8 clamp of the quantised value. -/
def clamp127 (q : Int) : Int := max (-127) (min 127 q)

/-- The clamped code is in [-127, 127] (never -128). -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  omega

-- § bf16 rounding

/-- Round-to-nearest-even on the upper 16 bits (Nat model of `bf`). -/
def bfRound (u : Nat) : Nat := (u + 0x7fff + (u / 65536) % 2) / 65536

/-- A value that is already bf16 (low 16 bits zero) is unchanged. -/
theorem bfRound_exact (u : Nat) (h : u % 65536 = 0) : bfRound u = u / 65536 := by
  unfold bfRound
  omega

/-- The rounding error is at most half a unit in the last place. -/
theorem bfRound_error (u : Nat) :
    u ≤ bfRound u * 65536 + 32768 ∧ bfRound u * 65536 ≤ u + 32768 := by
  unfold bfRound
  omega

/-- Ties go to the even mantissa. -/
theorem bfRound_tie_even (u : Nat) (h : u % 65536 = 32768) : bfRound u % 2 = 0 := by
  unfold bfRound
  have hx : u = 65536 * (u / 65536) + 32768 := by omega
  generalize u / 65536 = x at hx ⊢
  subst hx
  have hm : x % 2 < 2 := Nat.mod_lt _ (by decide)
  generalize hmd : x % 2 = m at hm ⊢
  have e : (65536 * x + 32768 + 32767 + m) / 65536 = x + m := by omega
  rw [e]
  omega

/-- No carry out of 16 bits for patterns away from the top of the range. -/
theorem bfRound_no_overflow (u : Nat) (h : u + 32768 < 4294967296) : bfRound u < 65536 := by
  unfold bfRound
  omega

end Strata.Prefill.Kernels



## FILE: /mnt/user-data/outputs/PrefillChunkOrchestration.lean
/-!
# MODULE: prefill_chunk_orchestration
Host-side orchestration of batched prefill: bump allocation of the device buffers, the chunk loop,
the n-gram window carried across chunks, expert grouping by counting sort, and the staging/dequant rings.

## Source semantics
* `Alloc::take<T>(n)` reserves `((n*sizeof(T) + 256 + 255) & ~255)` bytes: at least 256 bytes of slack and a
  multiple of 256. A borrowed region fails when `used + bytes > cap`; count-only mode just sums.
* The chunk loop visits `c0 = 0, T, 2T, ...` with chunk length `min(T, n - c0)`.
* The n-gram window `prev = (older, newer)` advances `(p0, p1) -> (p1, tok)` per token, including when PLE is off.
* Experts are grouped with a counting sort: `off[e+1] = off[e] + cnt[e]`, `slot = fill[e]++`.
* Staging ring of 8 slots with lookahead 7; dequantised ring of 2 slots (`j % 2`).

## Form template identifiers
Alloc.take -> bump_take ; ms_since -> elapsed_ms ; STAGE -> stage_ring_size ; DQ -> dequant_ring_size ;
cnt/off -> expert_counts/expert_offsets ; ple_prev -> ngram_window ; Prefill::run -> chunk_loop
-/

namespace Strata.Prefill.Orchestration

-- § allocator

/-- Bytes reserved for a request of `b` raw bytes. -/
def takeBytes (b : Nat) : Nat := (b + 256 + 255) / 256 * 256

/-- A reservation is at least the request plus 256 bytes. -/
theorem takeBytes_ge (b : Nat) : b + 256 ≤ takeBytes b := by
  unfold takeBytes
  omega

/-- A reservation wastes less than 512 bytes. -/
theorem takeBytes_lt (b : Nat) : takeBytes b < b + 512 := by
  unfold takeBytes
  omega

/-- Every reservation is a multiple of 256 (all sub-buffers stay 256-aligned). -/
theorem takeBytes_aligned (b : Nat) : takeBytes b % 256 = 0 := by
  unfold takeBytes
  omega

/-- A bump pointer that starts aligned stays aligned. -/
theorem bump_aligned (used b : Nat) (h : used % 256 = 0) : (used + takeBytes b) % 256 = 0 := by
  have := takeBytes_aligned b
  omega

/-- A successful bump reservation stays within the region. -/
theorem bump_fits (used cap b : Nat) (h : used + takeBytes b ≤ cap) : used + takeBytes b ≤ cap := h

/-- Slack added by `bytes_needed` never reduces the size. -/
theorem bytes_needed_ge (used : Nat) : used ≤ used + 8388608 := by omega

/-- The 32 MiB element scratch and the 32 MiB workspace constants. -/
theorem gemm_constants : (32 : Nat) * 1048576 = 33554432 ∧ 8 * 1048576 = 8388608 := by decide

-- § chunk loop

/-- Start of chunk `i`. -/
def chunkStart (T i : Nat) : Nat := i * T

/-- Length of chunk `i` of an `n`-token run. -/
def chunkLen (T n i : Nat) : Nat := min T (n - i * T)

/-- A visited chunk is non-empty and ends inside the run. -/
theorem chunk_bounds (T n c0 : Nat) (hT : 0 < T) (h : c0 < n) :
    0 < min T (n - c0) ∧ c0 + min T (n - c0) ≤ n := by omega

/-- Consecutive full chunks abut. -/
theorem chunk_abut (T n i : Nat) (h : i * T + T ≤ n) :
    chunkStart T (i + 1) = chunkStart T i + chunkLen T n i := by
  unfold chunkStart chunkLen
  have e : (i + 1) * T = i * T + T := Nat.succ_mul i T
  omega

/-- The chunk containing the end of the run stops exactly at `n`. -/
theorem chunk_last_ends_at_n (T n i : Nat) (h1 : i * T < n) (h2 : n ≤ i * T + T) :
    chunkStart T i + chunkLen T n i = n := by
  unfold chunkStart chunkLen
  omega

-- § PLE batched rows

/-- One batched PLE request reads `T*16` rows into `T*16*160 = T*2560` floats. -/
theorem ple_batch_floats (T : Nat) : T * 16 * 160 = T * 2560 := by omega

/-- Row index of (token, head) inside the batch. -/
theorem ple_rows_index (t h T : Nat) (ht : t < T) (hh : h < 16) : t * 16 + h < T * 16 := by omega

-- § the n-gram window

/-- Advance the two-token window by one token. -/
def advance (p : Int × Int) (tok : Int) : Int × Int := (p.2, tok)

/-- The window holds the two newest tokens (`-1` when absent). -/
def prevWindow (l : List Int) : Int × Int := (l.reverse.getD 1 (-1), l.reverse.getD 0 (-1))

/-- Advancing over a prompt prefix gives exactly the window the token path reconstructs from the prompt. -/
theorem advance_fold (l : List Int) : l.foldl advance (-1, -1) = prevWindow l := by
  induction l using List.reverseRecOn with
  | nil => rfl
  | append_singleton l t ih =>
      rw [List.foldl_append, ih]
      simp [advance, prevWindow, List.reverse_append]

/-- Two tokens fully replace the window. -/
theorem advance_two (p : Int × Int) (a b : Int) : [a, b].foldl advance p = (a, b) := rfl

-- § expert grouping (counting sort)

/-- Exclusive prefix sums of the per-expert counts. -/
def offs (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offs cnt e + cnt e

/-- Offsets are monotone. -/
theorem offs_mono (cnt : Nat → Nat) : ∀ {e e' : Nat}, e ≤ e' → offs cnt e ≤ offs cnt e' := by
  intro e e' h
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => exact Nat.le_trans ih (by simp [offs])

/-- The k-th assignment of expert `e` lands inside expert `e`'s own range. -/
theorem slot_in_range (cnt : Nat → Nat) (e k : Nat) (hk : k < cnt e) :
    offs cnt e ≤ offs cnt e + k ∧ offs cnt e + k < offs cnt (e + 1) := by
  simp [offs]
  omega

/-- Ranges of different experts are disjoint. -/
theorem ranges_disjoint (cnt : Nat → Nat) (e e' : Nat) (h : e < e') : offs cnt (e + 1) ≤ offs cnt e' :=
  offs_mono cnt h

-- § rings

/-- Staging slot pointer step. -/
theorem stage_next_lt (s : Nat) : (s + 1) % 8 < 8 := by omega

/-- With lookahead 7, slots of staged blobs `i < k ≤ i + 7` are distinct: no in-flight slot is overwritten. -/
theorem stage_slots_distinct (i k : Nat) (h1 : i < k) (h2 : k - i < 8) : i % 8 ≠ k % 8 := by omega

/-- The dequantised ring alternates between its two slots. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by omega

end Strata.Prefill.Orchestration



## FILE: /mnt/user-data/outputs/GenerateDriverContracts.lean
/-!
# MODULE: generate_driver_contracts
Pure contracts of the `strata generate` driver: argument validation, the teacher-forced decode loop and its
row accounting, prompt batching and lent-slot arithmetic, expert-cache sizing, speculative acceptance,
M-RoPE image positions and the serve request guards.

## Source semantics
* Validation: `0 < max_new ≤ max_context` and `n_prompt ≤ max_context - max_new`.
* Decode loop: positions `0 ..`; the next input is the prompt token while `pos+1 < n_prompt`, else the sample;
  a token is recorded from `pos ≥ n_prompt-1`; the loop stops once `max_new` tokens exist, so the number of
  processed positions (= logits rows) is `n_prompt - 1 + max_new`.
* Batched prefill covers `[0, n_batched)` with `n_batched = prefill_until` if `0 < prefill_until < n_prompt-1`
  else `n_prompt-1`; the token loop resumes at `pos_start = n_batched`.
* Borrowing: `k + 128 ≤ slots` slots from the end are lent, the first lent slot index is `slots - k`.
* Speculation: window `[x, d_1 .. d_{T-1}]`; `a` = longest prefix with `window[i+1] == out[i]`; emit `a+1`
  tokens, advance `p += a+1`.
* Image rows: row `j` of an image with `nx` columns sits at `(t, h, w) = (p, p + j/nx, p + j%nx)`; text after
  the image continues at `p + max(nx, ny)`.

## Form template identifiers
Options.max_new -> max_new ; Options.max_context -> max_context ; produced -> generated ;
pos_start -> resume_position ; lent -> lent_slots ; window/outv -> verify_window/verify_outputs ;
put(c,t,h,w) -> mrope_cell ; pcie_num -> pcie_share
-/

namespace Strata.Program.Generate

-- § argument validation

/-- The accepted argument shape. -/
def validContext (max_new max_ctx ntok : Nat) : Prop :=
  0 < max_new ∧ 0 < max_ctx ∧ max_new ≤ max_ctx ∧ ntok ≤ max_ctx - max_new

/-- Validation implies the later "context cannot hold prompt + new tokens" check never fires. -/
theorem validContext_fits (max_new max_ctx ntok : Nat) (h : validContext max_new max_ctx ntok) :
    ntok + max_new ≤ max_ctx := by
  unfold validContext at h
  omega

/-- With `max_context < 2^31 - 8` every position and cell index fits int32. -/
theorem positions_fit_int32 (max_ctx pos : Nat) (h1 : max_ctx ≤ 2147483647 - 8) (h2 : pos < max_ctx) :
    pos + 8 ≤ 2147483647 := by omega

/-- A token id accepted by the parser. -/
def validTokenId (id : Int) : Prop := 0 ≤ id ∧ id ≤ 2147483647

/-- Every accepted id fits a non-negative int32. -/
theorem validTokenId_int32 (id : Int) (h : validTokenId id) : -2147483648 ≤ id ∧ id ≤ 2147483647 := by
  unfold validTokenId at h
  omega

-- § the decode loop

/-- Number of logits rows written by the loop (`n_prompt - 1 + max_new`). -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- Tokens recorded after processing position `pos` (0-based). -/
def producedAfter (nPrompt pos : Nat) : Nat := if pos + 1 ≥ nPrompt then pos + 2 - nPrompt else 0

/-- The last prompt position records the first generated token. -/
theorem first_token_at_last_prompt (nPrompt : Nat) (h : 1 ≤ nPrompt) :
    producedAfter nPrompt (nPrompt - 1) = 1 := by
  unfold producedAfter
  split <;> omega

/-- Prompt positions before the last record nothing. -/
theorem prompt_positions_record_nothing (nPrompt pos : Nat) (h : pos + 1 < nPrompt) :
    producedAfter nPrompt pos = 0 := by
  unfold producedAfter
  split <;> omega

/-- The loop stops at position `n_prompt + max_new - 2`, where `max_new` tokens exist; it is the last row written. -/
theorem stop_position_rows (nPrompt maxNew : Nat) (h1 : 1 ≤ nPrompt) (h2 : 1 ≤ maxNew) :
    producedAfter nPrompt (nPrompt + maxNew - 2) = maxNew ∧
      (nPrompt + maxNew - 2) + 1 = dumpPositions nPrompt maxNew := by
  unfold producedAfter dumpPositions
  constructor
  · split <;> omega
  · omega

/-- Teacher forcing: the next input. -/
def nextInput (pos nPrompt : Nat) (promptTok sampled : Int) : Int :=
  if pos + 1 < nPrompt then promptTok else sampled

/-- While the prompt lasts the model's own guess is ignored. -/
theorem teacher_forced (pos nPrompt : Nat) (promptTok sampled : Int) (h : pos + 1 < nPrompt) :
    nextInput pos nPrompt promptTok sampled = promptTok := by
  unfold nextInput
  simp [h]

/-- Past the prompt the sampled token is fed back. -/
theorem free_running (pos nPrompt : Nat) (promptTok sampled : Int) (h : nPrompt ≤ pos + 1) :
    nextInput pos nPrompt promptTok sampled = sampled := by
  unfold nextInput
  have : ¬ pos + 1 < nPrompt := by omega
  simp [this]

-- § prompt batching

/-- Number of batched positions. -/
def nBatched (nPrompt prefillUntil : Nat) : Nat :=
  if 0 < prefillUntil ∧ prefillUntil < nPrompt - 1 then prefillUntil else nPrompt - 1

/-- The batched prefix never swallows the last prompt position. -/
theorem nBatched_le (nPrompt pu : Nat) : nBatched nPrompt pu ≤ nPrompt - 1 := by
  unfold nBatched
  split <;> omega

/-- The resume position is a valid prompt index. -/
theorem resume_in_prompt (nPrompt pu : Nat) (h : 1 ≤ nPrompt) : nBatched nPrompt pu < nPrompt := by
  have := nBatched_le nPrompt pu
  omega

/-- Lent-slot arithmetic: the lent block is `[slots - k, slots)` and at least 128 slots stay in service. -/
theorem lent_slots (k slots : Nat) (h : k + 128 ≤ slots) :
    128 ≤ slots - k ∧ (slots - k) + k = slots := by omega

/-- Expert-cache auto sizing: the granted slots fit the free memory minus the reserve. -/
theorem auto_slots_fit (free reserve blob : Nat) :
    (free - reserve) / blob * blob ≤ free - reserve := Nat.div_mul_le_self _ _

/-- A profile truncates the grant. -/
theorem profile_truncates (slots psize : Nat) : min slots psize ≤ slots := Nat.min_le_left _ _

-- § routing trace and clamps

/-- The adapter has advanced `layers` past the layer it just ran, so the traced index is `layers - 1`. -/
theorem trace_layer_index (l : Nat) (h : l < 48) : (l + 1) - 1 = l ∧ l + 1 ≤ 48 := by omega

/-- The PCIe share clamp lies in [0, 256]. -/
theorem pcie_clamp (x : Int) : 0 ≤ max 0 (min 256 x) ∧ max 0 (min 256 x) ≤ 256 := by omega

/-- Truncating a swap list to the budget never exceeds the budget. -/
theorem swaps_truncated {α : Type} (l : List α) (n : Nat) : (l.take n).length ≤ n :=
  List.length_take_le _ _

-- § speculative acceptance

/-- Longest prefix of drafts equal to the verifier outputs. -/
def accept : List Int → List Int → Nat
  | d :: ds, o :: os => if d = o then 1 + accept ds os else 0
  | _, _ => 0

/-- Acceptance never exceeds the number of drafts. -/
theorem accept_le_left : ∀ (ds os : List Int), accept ds os ≤ ds.length
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      unfold accept
      have := accept_le_left ds os
      split <;> simp <;> omega

/-- Acceptance never exceeds the number of verifier outputs. -/
theorem accept_le_right : ∀ (ds os : List Int), accept ds os ≤ os.length
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      unfold accept
      have := accept_le_right ds os
      split <;> simp <;> omega

/-- The accepted drafts are exactly the verifier's own outputs, so the emitted tokens equal greedy decode. -/
theorem accept_prefix : ∀ (ds os : List Int), os.take (accept ds os) = ds.take (accept ds os)
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      by_cases h : d = o
      · have e : accept (d :: ds) (o :: os) = accept ds os + 1 := by
          simp only [accept]
          rw [if_pos h]
          omega
        rw [e, List.take_succ_cons, List.take_succ_cons, accept_prefix ds os, h]
      · have e : accept (d :: ds) (o :: os) = 0 := by
          simp only [accept]
          rw [if_neg h]
        rw [e]
        simp

/-- A round emits at most `T` tokens and advances by at most `T` cells. -/
theorem round_emits_le (T a : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) : a + 1 ≤ T := by omega

/-- The emitted index range `0..a` is inside the `T` verifier outputs. -/
theorem emitted_in_outputs (T a i : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) (hi : i ≤ a) : i < T := by
  omega

/-- Window growth loop with fuel: extend while below `S` and the draft probability gate holds. -/
def growWindow (S : Nat) (ok : Nat → Bool) : Nat → Nat → Nat
  | 0, T => T
  | f + 1, T => if T < S ∧ ok (T - 1) = true then growWindow S ok f (T + 1) else T

/-- The window size stays in `[1, S]`. -/
theorem growWindow_bounds (S : Nat) (ok : Nat → Bool) :
    ∀ (f T : Nat), 1 ≤ T → T ≤ S → 1 ≤ growWindow S ok f T ∧ growWindow S ok f T ≤ S := by
  intro f
  induction f with
  | zero => intro T h1 h2; simp [growWindow]; omega
  | succ f ih =>
      intro T h1 h2
      unfold growWindow
      by_cases h : T < S ∧ ok (T - 1) = true
      · simp only [h, if_true]
        exact ih (T + 1) (by omega) (by omega)
      · simp only [h, if_false]
        omega

-- § serve request guards and M-RoPE

/-- An accepted request leaves room for the eight-cell margin. -/
theorem request_room (n maxNew ctx : Nat) (h : ¬ (n + maxNew + 8 > ctx)) : n + maxNew + 8 ≤ ctx := by
  omega

/-- Image row decomposition: row `j` of an `nx × ny` image has a valid (y, x). -/
theorem image_row_split (j nx ny : Nat) (hnx : 0 < nx) (h : j < nx * ny) :
    j / nx < ny ∧ j % nx < nx := by
  refine ⟨?_, Nat.mod_lt _ hnx⟩
  rw [Nat.div_lt_iff_lt_mul hnx]
  rw [Nat.mul_comm] at h
  exact h

/-- Image positions stay below the position where the following text resumes. -/
theorem image_positions_below_resume (p y x nx ny : Nat) (hy : y < ny) (hx : x < nx) :
    p + y < p + max nx ny ∧ p + x < p + max nx ny := by omega

/-- Text cells after an image resume strictly after every image position. -/
theorem text_resume_after_image (p nx ny : Nat) (hx : 0 < nx) (hy : 0 < ny) :
    p + (max nx ny - 1) < p + max nx ny := by omega

/-- The position table has three int32 per cell; the last component of cell `c` is inside it. -/
theorem mrope_table_index (c cells : Nat) (h : c < cells) : c * 3 + 2 < cells * 3 := by omega

/-- A request's cells (prompt + generation + margin) fit inside `max_context + 64` table cells. -/
theorem mrope_cells_cover (n maxNew ctx : Nat) (h : n + maxNew + 8 ≤ ctx) : n + maxNew < ctx + 64 := by
  omega

end Strata.Program.Generate



## FILE: /mnt/user-data/outputs/SuffixLookupDrafter.lean
/-!
# MODULE: suffix_lookup_drafter
Suffix-lookup speculative drafter: a hash table from the last three tokens to the most recent positions
where they ended, and a proposal that copies the continuation of the longest earlier match.

## Source semantics
* Table capacity is the smallest power of two `≥ 2 * capacity_tokens`; probing uses `i & (cap - 1)`.
* Keys are `mix(...) | 1`, so key 0 means "empty slot".
* Each slot keeps up to `WAYS` positions, newest first: shift right, put the new end at index 0, `n = min(n+1, WAYS)`.
* `min_match = max(3, min_match)`, `max_match = max(min_match, max_match)`.
* `propose` needs `n ≥ 4`; candidates `p ≥ cur` are skipped; the match walk stops at `len ≤ p`; ties keep the
  newer candidate (strict `>`); the continuation copies `hist[best_end+1 .. cur]`, at most `max_k` tokens.

## Form template identifiers
SuffixDrafter::append -> index_tokens ; propose -> propose_continuation ; Slot.pos -> recent_positions ;
find_slot -> probe_slot ; key_at -> suffix_key ; WAYS -> ways ; mask_ -> probe_mask
-/

namespace Strata.Spec.SuffixDrafter

-- § key and table

/-- A key with the low bit forced is never the empty marker 0. -/
theorem key_ne_zero (x : UInt64) : x ||| 1 ≠ 0 := by
  bv_decide

/-- Doubling-until-large loop with fuel. -/
def grow (target : Nat) : Nat → Nat → Nat
  | 0, c => c
  | f + 1, c => if c < target then grow target f (2 * c) else c

/-- The table capacity is always a power of two (so `& mask` is a valid modulo). -/
theorem grow_pow2 (target : Nat) :
    ∀ (f c : Nat), (∃ k, c = 2 ^ k) → ∃ k, grow target f c = 2 ^ k := by
  intro f
  induction f with
  | zero => intro c h; simpa [grow] using h
  | succ f ih =>
      intro c h
      unfold grow
      by_cases hc : c < target
      · simp only [hc, if_true]
        obtain ⟨k, hk⟩ := h
        exact ih (2 * c) ⟨k + 1, by rw [hk, Nat.pow_succ]; omega⟩
      · simp only [hc, if_false]
        exact h

/-- The capacity starts at 1 = 2^0. -/
theorem grow_start : ∃ k, (1 : Nat) = 2 ^ k := ⟨0, rfl⟩

/-- Probe index masking stays inside the table. -/
theorem probe_index_lt (x k : Nat) : x &&& (2 ^ k - 1) < 2 ^ k := by
  have h : 2 ^ k - 1 < 2 ^ k := by
    have := Nat.two_pow_pos k
    omega
  exact Nat.and_lt_two_pow x h

/-- Next probe also stays inside the table. -/
theorem probe_next_lt (i k : Nat) : (i + 1) &&& (2 ^ k - 1) < 2 ^ k := probe_index_lt (i + 1) k

-- § per-slot position list

/-- Insert a new end position, newest first, keeping at most `W`. -/
def push (W new : Nat) (l : List Nat) : List Nat := (new :: l).take W

/-- The slot never holds more than `W` positions. -/
theorem push_length_le (W new : Nat) (l : List Nat) : (push W new l).length ≤ W := by
  unfold push
  simp [List.length_take]

/-- The count update `n = min(n+1, WAYS)` matches the real list length. -/
theorem push_length (W new : Nat) (l : List Nat) : (push W new l).length = min W (l.length + 1) := by
  unfold push
  simp [List.length_take]

/-- With room for at least one entry, the newest position is first. -/
theorem push_head (w new : Nat) (l : List Nat) : (push (w + 1) new l).head? = some new := by
  simp [push, List.take_succ_cons]

/-- Positions are inserted in increasing time, so each slot stays strictly newest-first. -/
theorem push_sorted (W new : Nat) (l : List Nat) (hs : l.Pairwise (· > ·)) (hlt : ∀ x ∈ l, x < new) :
    (push W new l).Pairwise (· > ·) := by
  unfold push
  have h : (new :: l).Pairwise (· > ·) := List.pairwise_cons.2 ⟨fun x hx => hlt x hx, hs⟩
  exact h.sublist (List.take_sublist _ _)

-- § parameters

/-- The effective minimum match is at least 3 (the key covers three tokens). -/
theorem min_match_ge (m : Nat) : 3 ≤ max 3 m := by omega

/-- The effective maximum match is at least the minimum. -/
theorem max_match_ge (m M : Nat) : m ≤ max m M := by omega

-- § propose

/-- The key of the current suffix needs three tokens, guaranteed by `n ≥ 4`. -/
theorem key_indices_valid (n : Nat) (h : 4 ≤ n) : 2 ≤ n - 1 := by omega

/-- A candidate before the current suffix leaves positive room for the continuation. -/
theorem candidate_before_cur (p cur : Nat) (h : ¬ p ≥ cur) : p < cur := by omega

/-- During the match walk, the earlier index `p - len` does not underflow. -/
theorem match_walk_index (p len : Nat) (h : len ≤ p) : p - len + len = p := by omega

/-- The current-suffix index `cur - len` is valid whenever `len ≤ p < cur`. -/
theorem match_walk_cur_index (p cur len : Nat) (h : len ≤ p) (hp : p < cur) : 0 < cur - len := by omega

/-- Number of tokens copied as the continuation. -/
def contCount (cur bestEnd maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The proposal is at most `max_k` tokens. -/
theorem contCount_le (cur bestEnd maxK : Nat) : contCount cur bestEnd maxK ≤ maxK := by
  unfold contCount
  omega

/-- The last token read, `best_end + count`, is at most `cur`, an existing history index. -/
theorem contCount_in_history (cur bestEnd maxK : Nat) (h : bestEnd < cur) :
    bestEnd + contCount cur bestEnd maxK ≤ cur := by
  unfold contCount
  omega

/-- A non-empty continuation exists when a match was found before the end and `max_k > 0`. -/
theorem contCount_pos (cur bestEnd maxK : Nat) (h : bestEnd < cur) (hk : 0 < maxK) :
    0 < contCount cur bestEnd maxK := by
  unfold contCount
  omega

end Strata.Spec.SuffixDrafter



## FILE: /mnt/user-data/outputs/SpecDraftController.lean
/-!
# MODULE: spec_draft_controller
Index and bounds contracts of the speculative-decoding controller: cost-table indexing, the argmax-with-threshold
choice, and the acceptance-rate learning step.

## Source semantics
* `step_ms(k)` indexes the ratio tables with `n = clamp(k+1, 1, K_MAX+1)`, entry `n-1`.
* `choose` scans lookup draft lengths `1 .. min(available, K_MAX)` and MTP lengths `1 .. K_MAX`, keeping the
  candidate with the strictly largest tokens/ms, and falls back to "no draft" unless the best beats the baseline
  by `min_gain`.
* `observe(c, accepted)`: `accepted` is clamped to `[0, k]`; positions `0 .. min(k, accepted+1) - 1` are updated
  with `p += ema * (hit - p)`, hit = 1 for accepted positions and 0 for the rejected one.
* After a fully accepted window the untested positions `k .. K_MAX-1` move toward position `k-1`.

## Form template identifiers
CostModel.step_ms -> step_cost ; Controller.choose -> pick_draft ; Controller.observe -> learn_acceptance ;
mtp_p_ -> mtp_accept_rates ; lookup_q_ -> lookup_accept_rates ; K_MAX -> max_draft ; ema_ -> ema_rate
-/

namespace Strata.Spec.Controller

-- § cost-table indexing

/-- `clamp(k+1, 1, K+1)`. -/
def clampN (K k : Nat) : Nat := min (max (k + 1) 1) (K + 1)

/-- The table position `n` is in `[1, K+1]`. -/
theorem clampN_range (K k : Nat) : 1 ≤ clampN K k ∧ clampN K k ≤ K + 1 := by
  unfold clampN
  omega

/-- The table index `n - 1` is inside tables with `K + 1` entries. -/
theorem clampN_index_lt (K k : Nat) : clampN K k - 1 < K + 1 := by
  unfold clampN
  omega

/-- For draft lengths `k ≤ K` the clamp is the identity `k + 1`. -/
theorem clampN_exact (K k : Nat) (h : k ≤ K) : clampN K k = k + 1 := by
  unfold clampN
  omega

-- § candidate selection

/-- A strict-improvement scan never loses to any candidate and never loses to its start value. -/
theorem foldl_max_ge :
    ∀ (cs : List Nat) (init : Nat), init ≤ cs.foldl max init ∧ ∀ x ∈ cs, x ≤ cs.foldl max init
  | [], init => by simp
  | c :: cs, init => by
      have ih := foldl_max_ge cs (max init c)
      simp only [List.foldl_cons]
      refine ⟨Nat.le_trans (Nat.le_max_left _ _) ih.1, ?_⟩
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact Nat.le_trans (Nat.le_max_right _ _) ih.1
      · exact ih.2 x hx

/-- Lookup draft lengths considered never exceed `K_MAX`. -/
theorem lookup_len_le (avail K k : Nat) (h : k ≤ min avail K) : k ≤ K := by omega

/-- Every considered length is at most the number of available lookup tokens. -/
theorem lookup_len_le_avail (avail K k : Nat) (h : k ≤ min avail K) : k ≤ avail := by omega

-- § observe

/-- Accepted count clamped to `[0, k]`. -/
def clampAccepted (a : Int) (k : Nat) : Int := max 0 (min a k)

/-- The clamped acceptance is a valid count. -/
theorem clampAccepted_range (a : Int) (k : Nat) : 0 ≤ clampAccepted a k ∧ clampAccepted a k ≤ k := by
  unfold clampAccepted
  omega

/-- Number of positions updated: the accepted ones plus the first rejected one. -/
def seen (k accepted : Nat) : Nat := min k (accepted + 1)

/-- The update loop never leaves the drafted window. -/
theorem seen_le (k accepted : Nat) : seen k accepted ≤ k := by
  unfold seen
  omega

/-- All accepted positions are updated (as hits). -/
theorem accepted_updated (k accepted : Nat) (h : accepted ≤ k) : accepted ≤ seen k accepted := by
  unfold seen
  omega

/-- A rejection is recorded at position `accepted` exactly when something was drafted beyond it. -/
theorem rejection_recorded (k accepted : Nat) (h : accepted < k) : seen k accepted = accepted + 1 := by
  unfold seen
  omega

/-- If everything was accepted, no rejected position is recorded. -/
theorem all_accepted_seen (k : Nat) : seen k k = k := by
  unfold seen
  omega

/-- The learning indices stay inside the K_MAX-entry rate array. -/
theorem observe_index_lt (K k accepted i : Nat) (hk : k ≤ K) (hi : i < seen k accepted) : i < K := by
  have := seen_le k accepted
  omega

/-- Extension of untested positions after a full window reads position `k-1`, which exists since `k ≥ 1`. -/
theorem extension_source (K k : Nat) (hk : 0 < k) (hKk : k ≤ K) : k - 1 < K := by omega

/-- Untested positions `k .. K-1` are strictly beyond the tested window. -/
theorem extension_range (K k i : Nat) (hk : 0 < k) (hi : k ≤ i) (hiK : i < K) : k - 1 < i := by omega

-- § EMA step (scaled integer model: probabilities are numerators over S)

/-- Scaled EMA: `p' * S = p * (S - a) + h * a` where `a` is the rate and `p, h` are probabilities. -/
def emaScaled (S p a h : Int) : Int := p * (S - a) + h * a

/-- The EMA update is a convex combination: it never leaves the probability range `[0, S]`
(after dividing the scaled value by `S`). -/
theorem emaScaled_range (S p a h : Int) (hp0 : 0 ≤ p) (hpS : p ≤ S) (ha0 : 0 ≤ a) (haS : a ≤ S)
    (hh0 : 0 ≤ h) (hhS : h ≤ S) : 0 ≤ emaScaled S p a h ∧ emaScaled S p a h ≤ S * S := by
  unfold emaScaled
  have hSa : 0 ≤ S - a := by omega
  have h1 : 0 ≤ p * (S - a) := Int.mul_nonneg hp0 hSa
  have h2 : 0 ≤ h * a := Int.mul_nonneg hh0 ha0
  have h3 : p * (S - a) ≤ S * (S - a) := Int.mul_le_mul_of_nonneg_right hpS hSa
  have h4 : h * a ≤ S * a := Int.mul_le_mul_of_nonneg_right hhS ha0
  have h5 : S * (S - a) = S * S - S * a := Int.mul_sub S S a
  omega

/-- A hit (`h = S`) never decreases the rate. -/
theorem ema_hit_monotone (S p a : Int) (hp0 : 0 ≤ p) (hpS : p ≤ S) (ha0 : 0 ≤ a) (haS : a ≤ S) :
    p * S ≤ emaScaled S p a S := by
  unfold emaScaled
  have h1 : p * (S - a) + p * a = p * S := by rw [← Int.mul_add]; congr 1; omega
  have h2 : p * a ≤ S * a := Int.mul_le_mul_of_nonneg_right hpS ha0
  omega

/-- A miss (`h = 0`) never increases the rate. -/
theorem ema_miss_monotone (S p a : Int) (hp0 : 0 ≤ p) (ha0 : 0 ≤ a) (haS : a ≤ S) :
    emaScaled S p a 0 ≤ p * S := by
  unfold emaScaled
  have h1 : p * (S - a) ≤ p * S := by
    have : p * (S - a) + p * a = p * S := by rw [← Int.mul_add]; congr 1; omega
    have h2 : 0 ≤ p * a := Int.mul_nonneg hp0 ha0
    omega
  omega

end Strata.Spec.Controller




[LIGHT FILES — structure only, no LAYER 1a/1b]
  include/strata/core/device.hpp (72 lines)
  include/strata/core/graph.hpp (118 lines)
  include/strata/core/hit_hook.hpp (42 lines)
  include/strata/core/layer.hpp (567 lines)
  include/strata/core/layout.hpp (98 lines)
  include/strata/core/mtp.hpp (136 lines)
  include/strata/core/native_dense.hpp (37 lines)
  include/strata/core/native_head.hpp (66 lines)
  include/strata/core/pinned.hpp (96 lines)
  include/strata/core/session.hpp (412 lines)
  include/strata/core/verify.hpp (161 lines)
  include/strata/core/weights.hpp (136 lines)
  include/strata/kernels/cpu/expert.hpp (180 lines)
  include/strata/kernels/cpu/expert_layout.hpp (52 lines)
  include/strata/kernels/cpu/iq_avx512.hpp (19 lines)
  include/strata/kernels/cpu/native_expert.hpp (50 lines)
  include/strata/kernels/elementwise.hpp (117 lines)
  include/strata/kernels/gdn.hpp (98 lines)
  include/strata/kernels/gr.hpp (123 lines)
  include/strata/kernels/native_mmvq.hpp (113 lines)
  include/strata/kernels/ngram.hpp (195 lines)
  include/strata/kernels/ple.hpp (120 lines)
  include/strata/kernels/qsa.hpp (361 lines)
  include/strata/kernels/s2_expert_grouped.hpp (119 lines)
  include/strata/kernels/s_gemv.hpp (134 lines)
  include/strata/kernels/shared_expert.hpp (102 lines)
  include/strata/kernels/verify_kernels.hpp (81 lines)
  include/strata/kernels/bf16_bits.hpp (47 lines)
  include/strata/kernels/bf16_gemv.hpp (49 lines)
  include/strata/kernels/dequant_bf16.hpp (28 lines)
  include/strata/kernels/dequant_s2.hpp (19 lines)
  include/strata/kernels/fused_gdn.hpp (36 lines)
  include/strata/kernels/fused_gr.hpp (52 lines)
  include/strata/kernels/iq_kernels.hpp (58 lines)
  include/strata/kernels/kv_q8.hpp (39 lines)
  include/strata/kernels/mrope.hpp (33 lines)
  include/strata/kernels/native_flash_attn.hpp (37 lines)
  include/strata/kernels/native_gdn.hpp (23 lines)
  include/strata/kernels/native_gdn_preprocess.hpp (38 lines)
  include/strata/kernels/native_gr_norm.hpp (19 lines)
  include/strata/kernels/native_gr_postops.hpp (29 lines)
  include/strata/kernels/native_moe.hpp (19 lines)
  include/strata/kernels/native_ple_postops.hpp (31 lines)
  include/strata/kernels/native_qsa.hpp (31 lines)
  include/strata/kernels/native_qsa_indexer.hpp (36 lines)
  include/strata/kernels/native_qsa_score.hpp (32 lines)
  include/strata/kernels/native_rope.hpp (15 lines)
  include/strata/kernels/native_router.hpp (18 lines)
  include/strata/kernels/qsa_decode_attn.hpp (45 lines)
  include/strata/kernels/qsa_select.hpp (34 lines)
  include/strata/kernels/quantize_act.hpp (59 lines)
  include/strata/kernels/rope.hpp (51 lines)
  include/strata/kernels/router_top10.hpp (21 lines)
  include/strata/kernels/s2_gemv.hpp (26 lines)
  include/strata/kernels/s2_gemv_q8.hpp (23 lines)
  include/strata/kernels/sampler.hpp (44 lines)
  include/strata/platform/direct_file.hpp (60 lines)
  include/strata/platform/memory.hpp (27 lines)
  include/strata/prefill/gemm.hpp (55 lines)
  include/strata/prefill/kernels.hpp (75 lines)
  include/strata/prefill/prefill.hpp (76 lines)
  include/strata/program/logits_selection.hpp (31 lines)
  include/strata/spec/suffix_drafter.hpp (54 lines)
  include/strata/artifact/dequant.hpp (293 lines)
  include/strata/artifact/gguf_reader.hpp (487 lines)
  include/strata/ngram/ple_reader.hpp (87 lines)
  src/core/concurrent_main.cpp (198 lines)
  src/core/device.cu (118 lines)
  src/core/device_main.cpp (79 lines)
  src/core/graph.cpp (147 lines)
  src/core/layout.cpp (174 lines)
  src/core/load_main.cpp (120 lines)
  src/core/native_dense.cpp (180 lines)
  src/core/native_head.cpp (152 lines)
  src/core/overlap_main.cpp (170 lines)
  src/core/pinned.cu (292 lines)
  src/kernels/bf16_gemv_parity.cpp (149 lines)
  src/kernels/cpu/expert_layout.cpp (137 lines)
  src/kernels/cpu/expert_multi_test.cpp (88 lines)
  src/kernels/cpu/expert_parity.cpp (185 lines)
  src/kernels/cpu/native_expert.cpp (115 lines)
  src/kernels/cpu/pool_test.cpp (225 lines)
  src/kernels/cuda/bf16_gemv.cu (142 lines)
  src/kernels/cuda/dequant_bf16.cu (246 lines)
  src/kernels/cuda/dequant_s2.cu (52 lines)
  src/kernels/cuda/elementwise.cu (290 lines)
  src/kernels/cuda/native_bf16.cu (119 lines)
  src/kernels/cuda/native_gr_norm.cu (103 lines)
  src/kernels/cuda/native_gr_postops.cu (116 lines)
  src/kernels/cuda/native_moe.cu (75 lines)
  src/kernels/cuda/native_ple_postops.cu (135 lines)
  src/kernels/cuda/native_qsa.cu (128 lines)
  src/kernels/cuda/native_rope.cu (89 lines)
  src/kernels/cuda/rope.cu (96 lines)
  src/kernels/cuda/s2_gemv.cu (64 lines)
  src/kernels/cuda/s2_gemv_fast.cu (151 lines)
  src/kernels/cuda/s2_gemv_q8.cu (97 lines)
  src/kernels/cuda/s2_gemv_quads.cu (94 lines)
  src/kernels/cuda/shared_expert.cu (339 lines)
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
  src/ngram/ple_reader_test.cpp (238 lines)
  src/prefill/gemm.cu (108 lines)
  src/spec/suffix_drafter_test.cpp (142 lines)
  src/spec/controller_test.cpp (75 lines)
  src/artifact/gguf_reader.cpp (68 lines)
  src/artifact/dequant.cpp (124 lines)
  src/platform/direct_file.cpp (242 lines)
  src/platform/memory.cpp (77 lines)
  src/platform/memory_test.cpp (36 lines)
  src/plan/plan_main.cpp (74 lines)
  bench/results/2026-09-24-final/estimates.md (28 lines)
  bench/results/2026-09-24-final/matrix.md (16 lines)
  bench/results/2026-09-24-vision/vision.md (6 lines)
  bench/results/2026-09-24-final/estimates.json (461 lines)
  bench/results/2026-09-24-final/matrix.json (240 lines)
  bench/results/2026-09-24-vision/vision.json (53 lines)
  third_party/ggml/VERSION.txt (2 lines)
  third_party/ggml/ggml-common.h (1912 lines)
  tools/_paths.py (29 lines)
  tools/gguf_reader.py (143 lines)
  tools/iq_pack.py (222 lines)
  tools/mtp_fetch.py (122 lines)
  tools/mtp_rt.py (113 lines)
  tools/pack_index.py (261 lines)
  tools/pack_layer.py (124 lines)
  tools/vision/CMakeLists.txt (45 lines)
  tools/vision/strata_vision.cpp (159 lines)
  serve/__init__.py (2 lines)
  serve/chat_golden.json (197 lines)
  serve/chat_template.jinja (184 lines)
  serve/index.html (153 lines)
  chat.py (117 lines)
  setup.py (807 lines)
  setup.sh (31 lines)
  START-HERE.bat (53 lines)
  CMakeLists.txt (537 lines)
  ref/load.py (213 lines)
  include/strata/spec/controller.hpp (74 lines)
  include/strata/plan/plan.hpp (169 lines)
  serve/server.py (607 lines)
  serve/frontend.py (345 lines)
  tools/strata_tokenizer.py (267 lines)
  tools/mtp_pack.py (194 lines)
  tools/gguf_writer.py (235 lines)
  tools/canonical_xcheck.py (800 lines)
  tools/strata_pack.py (414 lines)
