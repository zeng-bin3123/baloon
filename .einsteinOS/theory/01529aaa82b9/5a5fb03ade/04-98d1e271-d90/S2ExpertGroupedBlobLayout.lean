/-!
# MODULE: s2_expert_grouped_blob_layout
Grouped GPU expert for the Q2_0 (S2) MoE: gate/up/down projections over Q8_0 activations with `__dp4a`,
warp-per-row, device-side hit selection and grouping, and the CPU-order (bit-matching) variant.

## Source semantics
* Blob geometry for `H = 2560`, `FF = 640`, group 64: codes `[gate/up interleaved | down]`, then gate/up
  scales, then down scales; all offsets are fixed literals.
* `sum_j (c_j - 1) x_j = sum_j c_j x_j - sum_j x_j` (weight bias through the "ones" dot).
* Gate/up rows are interleaved (slot `i` even = gate row `i/2`, odd = up row `i/2`) but the output is
  gate-major: `[hit h gate rows | ... | hit h up rows]`.
* Scales: one fp16 per 64 weights = one per two 32-element chunks, index `c >> 1`.
* Hit compaction by ballot prefix popcount assigns dense slots in entry order.
* CPU-order lane reduction adds lanes in the order 4, 1, 2.

## Form template identifiers
- `gu_kernel` / `down_kernel`        → `projection_kernel`
- `hit_select_kernel`                → `resident_hit_compaction`
- `group_resident_kernel`            → `group_by_expert`
- `row_dot_s2_q8`                    → `s2_row_dot_q8`
- `O_D_CODES`, `O_GU_SCALES`, ...    → `blob_region_offsets`
-/

namespace Strata.S2Expert

-- § blob geometry

/-- Row and scale counts derived from `H = 2560`, `FF = 640`, `QK = 64`. -/
theorem row_constants :
    2560 / 4 = 640 ∧ 640 / 4 = 160 ∧ 2560 / 64 = 40 ∧ 640 / 64 = 10 := by decide

/-- Region offsets of the blob. -/
theorem blob_offsets :
    2 * 640 * 640 = 819200 ∧
    2 * 640 * 640 + 2560 * 160 = 1228800 ∧
    2 * 640 * 640 + 2560 * 160 + 2 * 640 * 40 * 2 = 1331200 ∧
    2 * 640 * 640 + 2560 * 160 + 2 * 640 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

/-- Regions are consecutive and non-overlapping: gate/up codes < down codes < gate/up scales < down scales. -/
theorem blob_regions_ordered : 819200 < 1228800 ∧ 1228800 < 1331200 ∧ 1331200 < 1382400 := by decide

/-- Scale index of 32-element chunk `c`: chunk pairs share one fp16 scale; gate/up has 80 chunks → 40 scales. -/
theorem chunk_scale_in_range (c : Nat) (h : c < 80) : c / 2 < 40 := by omega

/-- Down rows have 20 chunks → 10 scales, and lanes `0..19` each hold one chunk (`20 ≤ 32`). -/
theorem down_chunk_scale_in_range (c : Nat) (h : c < 20) : c / 2 < 10 ∧ 20 ≤ 32 := by omega

/-- Lanes `0..31` cover the 80 gate/up chunks in `GU_CHUNKS = 3` rounds, and 2 rounds do not suffice. -/
theorem gu_chunks_rounds : (80 + 31) / 32 = 3 ∧ 80 ≤ 3 * 32 ∧ 2 * 32 < 80 := by decide

/-- Block row tilings divide the output dimensions exactly. -/
theorem grouped_tilings : 2 * 640 / 32 = 40 ∧ 2 * 640 % 32 = 0 ∧ 2560 / 64 = 40 ∧ 2560 % 64 = 0 := by decide

/-- A Q8_0 chunk is 34 bytes (2 header + 32 data): data at offset 2 is never 4-aligned, hence `memcpy`. -/
theorem q8_0_data_misaligned : 2 + 32 = 34 ∧ 2 % 4 ≠ 0 ∧ 34 % 4 ≠ 0 := by decide

-- § integer identity

/-- The weight bias through `sum(c*x) - sum(x)`: `Σ (c-1)·x = Σ c·x − Σ x`. -/
theorem bias_through_ones (l : List (Int × Int)) :
    (l.map (fun p => (p.1 - 1) * p.2)).sum = (l.map (fun p => p.1 * p.2)).sum - (l.map (fun p => p.2)).sum := by
  induction l with
  | nil => simp
  | cons p l ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, Int.sub_mul]
    omega

-- § interleaved rows, gate-major output

/-- Output slot of gate/up row-slot `i` of hit `h` among `n` hits (gate-major layout). -/
def guSlot (n h i : Nat) : Nat :=
  if i % 2 = 1 then n * 640 + h * 640 + i / 2 else h * 640 + i / 2

/-- The output slot is inside the `2 * n * 640` gate/up buffer. -/
theorem guSlot_lt (n h i : Nat) (hh : h < n) (hi : i < 1280) : guSlot n h i < 2 * n * 640 := by
  unfold guSlot
  split <;> omega

/-- Distinct (hit, row-slot) pairs write distinct output slots: no two warps collide. -/
theorem guSlot_injective (n h i h' i' : Nat) (hh : h < n) (hh' : h' < n) (hi : i < 1280) (hi' : i' < 1280)
    (e : guSlot n h i = guSlot n h' i') : h = h' ∧ i = i' := by
  unfold guSlot at e
  split at e <;> split at e <;> omega

/-- Gate rows of hit `h` are contiguous from `h * 640`; up rows contiguous from `n * 640 + h * 640`, so the
swiglu kernel finds `up` at `+ n * 640` from `gate`. -/
theorem swiglu_pairs_with_own_up (n h r : Nat) (hr : r < 640) :
    guSlot n h (2 * r + 1) = guSlot n h (2 * r) + n * 640 := by
  unfold guSlot
  split <;> split <;> omega

-- § hit compaction

/-- Number of `true` entries strictly before position `i`. -/
def prefixCount (l : List Bool) (i : Nat) : Nat := (l.take i).count true

/-- A prefix count never exceeds the total count, so compacted slots stay below `count`. -/
theorem prefixCount_le (l : List Bool) (i : Nat) : prefixCount l i ≤ l.count true := by
  unfold prefixCount
  exact (List.take_sublist i l).count_le true

/-- Warp-ballot compaction across four warps: slot = earlier warps' counts + in-warp prefix. -/
theorem four_warp_slot_lt (c0 c1 c2 c3 p : Nat) (hp : p < c2) :
    c0 + c1 + p < c0 + c1 + c2 + c3 := by omega

/-- Entry position within a group: group start plus rank is below the next group start. -/
theorem group_entry_slot (start size rank : Nat) (h : rank < size) : start + rank < start + size := by omega

/-- Entry token is the routed index divided by `k_per_tok`; entries of one token have distinct destinations. -/
theorem token_of_entry (i k : Nat) (hk : 0 < k) : i = (i / k) * k + i % k := by
  rw [Nat.mul_comm]
  exact (Nat.div_add_mod i k).symm

-- § scratch carving

/-- 16-byte alignment of scratch regions. -/
theorem align16_ok (x : Nat) : x ≤ (x + 15) / 16 * 16 ∧ (x + 15) / 16 * 16 % 16 = 0 := by omega

/-- Down rows are indexed `row = h * H + r`; decomposition is unique for `r < H`. -/
theorem down_row_split (H h r : Nat) (hr : r < H) : (h * H + r) / H = h ∧ (h * H + r) % H = r := by
  have hpos : 0 < H := by omega
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hr, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt hr

-- § CPU-order lane reduction

/-- The CPU-order reduction (shuffle offsets 4, 1, 2 within groups of 8) sums all eight lanes; over exact
integers the order is immaterial, over floats it is exactly what is being matched. -/
theorem cpu_order_reduction_sum (a0 a1 a2 a3 a4 a5 a6 a7 : Int) :
    (((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7))) = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 := by
  omega

/-- Eight lanes handle one 64-element block: each lane's `dot4` covers 4 of 32 elements per half. -/
theorem cpu_order_lane_coverage : 8 * 4 = 32 ∧ 2 * 32 = 64 := by decide

end Strata.S2Expert
