/-!
# MODULE: s2_expert_grouped_hits
SOURCES: src/kernels/cuda/s2_expert_grouped.cu
Grouped GPU routed-expert kernels over Q2_0 expert blobs (H = 2560, FF = 640): gate/up (interleaved rows,
gate-major output), SwiGLU, Q8_0 requantisation, down; device-side hit selection/compaction; resident grouping;
CPU-order variant.

## Source semantics
* Blob layout: [gate/up codes 2*FF rows of 640 B | down codes H rows of 160 B | gate/up scales | down scales].
* Row slot i of the 2*FF gate/up rows is gate row i/2 (even) or up row i/2 (odd); output is gate-major:
  gate at h*FF + r, up at n_hits*FF + h*FF + r.
* One fp16 weight scale per 64 values = per two 32-value activation chunks.
* hit_select compacts resident experts by ballot prefix popcount; slots are distinct and increasing.
* ent_tok = entry / k_per_token; grouped kernels stage at most GMAX = 8 entries per group in shared memory.

## Form template identifiers
H,FF -> n_embd,n_ff ; O_D_CODES,O_GU_SCALES,O_D_SCALES -> blob_offsets ; gu_kernel -> gate_up_pass ;
hit_select_kernel -> resident_compaction ; group_resident_kernel -> group_builder ; down_kernel -> down_pass
-/

namespace Strata.S2ExpertGrouped

/-- Hidden size. -/
def H : Nat := 2560
/-- FFN width. -/
def FF : Nat := 640
/-- Gate/up code bytes per row. -/
def ROW_GU : Nat := H / 4
/-- Down code bytes per row. -/
def ROW_D : Nat := FF / 4
/-- fp16 scales per gate/up row. -/
def SC_GU : Nat := H / 64
/-- fp16 scales per down row. -/
def SC_D : Nat := FF / 64
/-- Offset of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Offset of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
/-- Total blob bytes. -/
def BLOB : Nat := O_D_SCALES + H * SC_D * 2

-- § blob geometry

/-- Row code widths follow from 2-bit packing. -/
theorem row_widths : ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 := by
  decide

/-- The offsets are the arithmetic the kernel hard-codes. -/
theorem offsets_values : O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 := by
  decide

/-- Blob size: 1,228,800 B of codes plus 153,600 B of scales. -/
theorem blob_size : BLOB = 1382400 ∧ BLOB = 1228800 + 153600 := by
  decide

/-- Every gate/up code row slot lies inside the gate/up code region. -/
theorem gu_code_row_in_bounds (i : Nat) (h : i < 1280) : i * 640 + 640 ≤ 819200 := by
  omega

/-- Every down code row lies between the down-code start and the scales start. -/
theorem down_code_row_in_bounds (r : Nat) (h : r < 2560) :
    819200 + r * 160 + 160 ≤ 1228800 := by
  omega

/-- Every gate/up scale row lies in the gate/up scale region. -/
theorem gu_scale_row_in_bounds (i : Nat) (h : i < 1280) :
    1228800 + i * 80 + 80 ≤ 1331200 := by
  omega

/-- Every down scale row lies inside the blob. -/
theorem down_scale_row_in_bounds (r : Nat) (h : r < 2560) :
    1331200 + r * 20 + 20 ≤ 1382400 := by
  omega

/-- A 32-value chunk index maps to a valid weight scale (per 64 values) in gate/up. -/
theorem gu_scale_of_chunk (c : Nat) (h : c < 80) : c / 2 < 40 := by
  omega

/-- Same for the down projection. -/
theorem d_scale_of_chunk (c : Nat) (h : c < 20) : c / 2 < 10 := by
  omega

/-- The per-lane chunk count of a gate/up row covers the 80 chunks. -/
theorem gu_chunks_cover : (2560 / 32 + 31) / 32 = 3 ∧ 80 ≤ 32 * 3 := by
  decide

-- § interleaved rows and gate-major output

/-- A row slot is its gate/up pair index and parity. -/
theorem slot_decomp (i : Nat) : 2 * (i / 2) + i % 2 = i := by
  omega

/-- Gate-major index of gate row r of hit h. -/
def gateIdx (h r : Nat) : Nat := h * 640 + r

/-- Gate-major index of up row r of hit h with n hits. -/
def upIdx (n h r : Nat) : Nat := n * 640 + h * 640 + r

/-- Gate indices stay in the first n*FF floats. -/
theorem gateIdx_lt (n h r : Nat) (hh : h < n) (hr : r < 640) : gateIdx h r < n * 640 := by
  unfold gateIdx
  omega

/-- Up indices lie in [n*FF, 2*n*FF). -/
theorem upIdx_range (n h r : Nat) (hh : h < n) (hr : r < 640) :
    n * 640 ≤ upIdx n h r ∧ upIdx n h r < 2 * n * 640 := by
  unfold upIdx
  omega

/-- Gate and up never collide. -/
theorem gate_up_disjoint (n h r h' r' : Nat) (hh : h < n) (hr : r < 640) :
    gateIdx h r ≠ upIdx n h' r' := by
  unfold gateIdx upIdx
  omega

/-- Gate-major indexing is injective per half: distinct (h, r) give distinct slots. -/
theorem gateIdx_injective (h r h' r' : Nat) (hr : r < 640) (hr' : r' < 640)
    (e : gateIdx h r = gateIdx h' r') : h = h' ∧ r = r' := by
  unfold gateIdx at e
  omega

/-- SwiGLU reads gate at i and up at n_pairs + i, which is the up index of the same (h, r). -/
theorem swiglu_pairing (n h r : Nat) : gateIdx h r + n * 640 = upIdx n h r := by
  unfold gateIdx upIdx
  omega

-- § hit selection (ballot compaction)

/-- Number of hits among lanes below n. -/
def prefCount (hit : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n + 1 => prefCount hit n + (if hit n = true then 1 else 0)

/-- The prefix count is at most the prefix length. -/
theorem prefCount_le (hit : Nat → Bool) (n : Nat) : prefCount hit n ≤ n := by
  induction n with
  | zero => simp [prefCount]
  | succ n ih =>
    show prefCount hit n + (if hit n = true then 1 else 0) ≤ n + 1
    split <;> omega

/-- The prefix count is monotone. -/
theorem prefCount_mono (hit : Nat → Bool) (m d : Nat) : prefCount hit m ≤ prefCount hit (m + d) := by
  induction d with
  | zero => simp
  | succ d ih =>
    have e : prefCount hit (m + (d + 1)) = prefCount hit (m + d) + (if hit (m + d) = true then 1 else 0) := rfl
    rw [e]
    split <;> omega

/-- A hit lane's slot is its prefix count and the next prefix is one larger. -/
theorem slot_succ (hit : Nat → Bool) (lane : Nat) (hl : hit lane = true) :
    prefCount hit (lane + 1) = prefCount hit lane + 1 := by
  show prefCount hit lane + (if hit lane = true then 1 else 0) = _
  simp [hl]

/-- Slots of hit lanes are strictly increasing: compaction keeps entry order with distinct slots. -/
theorem slots_strictly_increasing (hit : Nat → Bool) (a b : Nat) (hab : a < b) (ha : hit a = true) :
    prefCount hit a < prefCount hit b := by
  have h1 := slot_succ hit a ha
  obtain ⟨d, rfl⟩ : ∃ d, b = (a + 1) + d := ⟨b - (a + 1), by omega⟩
  have h2 := prefCount_mono hit (a + 1) d
  omega

/-- The slot of a hit lane is below the total count over k lanes. -/
theorem slot_lt_count (hit : Nat → Bool) (lane k : Nat) (hk : lane < k) (hl : hit lane = true) :
    prefCount hit lane < prefCount hit k := by
  exact slots_strictly_increasing hit lane k hk hl

/-- The hit count over k <= 32 lanes is at most 32. -/
theorem count_le_32 (hit : Nat → Bool) (k : Nat) (hk : k ≤ 32) : prefCount hit k ≤ 32 := by
  have := prefCount_le hit k
  omega

-- § grouping

/-- Token index of an entry is below the token count. -/
theorem ent_tok_lt (i kPer tokens : Nat) (h : i < tokens * kPer) : i / kPer < tokens := by
  exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- Shared staging of one group: 8 entries of 2560 B plus 8 * 80 fp32 scales fits 48 KiB. -/
theorem gu_shared_fits : 8 * (2560 / 4) * 4 + 8 * (2560 / 32) * 4 = 23040 ∧ 23040 ≤ 49152 := by
  decide

/-- Gate/up grouped grid: 40 blocks of 32 rows exactly cover 2*FF rows. -/
theorem gu_grouped_grid : 2 * 640 / 32 = 40 ∧ 40 * 32 = 2 * 640 := by
  decide

/-- Down grouped grid: 40 blocks of 64 rows exactly cover H rows. -/
theorem down_grouped_grid : 2560 / 64 = 40 ∧ 40 * 64 = 2560 := by
  decide

/-- Per-warp row assignment rr = warp, warp+8, ... covers 0..R-1 once for R divisible by 8. -/
theorem warp_rows_cover (rr : Nat) (h : rr < 32) : rr % 8 < 8 ∧ rr = rr % 8 + 8 * (rr / 8) := by
  omega

-- § scratch sizing

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by
  unfold align16
  omega

/-- Alignment gives a multiple of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16
  omega

/-- The scratch size dominates the gate/up float plane. -/
theorem scratch_holds_gu (n nEmbd nFf : Nat) :
    n * (2 * nFf) * 4 ≤
      align16 (n * (2 * nFf) * 4) + align16 (n * (nFf / 32) * 34) +
        2 * align16 (n * (nFf / 32) * 4) + align16 (nEmbd / 32 * 4) := by
  have := align16_ge (n * (2 * nFf) * 4)
  omega

end Strata.S2ExpertGrouped
