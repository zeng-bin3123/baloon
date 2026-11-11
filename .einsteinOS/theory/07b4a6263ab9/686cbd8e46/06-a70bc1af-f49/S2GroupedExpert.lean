/-!
# MODULE: s2_grouped_expert
SOURCES: src/kernels/cuda/s2_expert_grouped.cu
Model of the R4 grouped GPU expert over S2 (2-bit, Q2_0-style) blobs: the fixed blob
geometry, the `Σ (c-1)·x = Σ c·x − Σ x` DP4A identity, the gate-major gate/up output
layout, the device-side hit compaction and the CPU-order 8-lane reduction tree.

## Source semantics
* A blob is `[2·FF gate/up code rows | H down code rows | gate/up fp16 scales | down scales]`
  with H = 2560, FF = 640, one fp16 weight scale per 64 elements.
* Row-slot `i` of the gate/up rows is gate row `i/2` (even) or up row `i/2` (odd); output is
  GATE-MAJOR: hit `h` gate row `r` → `h·FF + r`, up row `r` → `n·FF + h·FF + r`.
* A code byte packs four LSB-first 2-bit fields; the weight value is `code − 1`.
* Q8_0 activation blocks are 34 bytes, so their int8 payload alternates between 2-mod-4 and
  0-mod-4 addresses — hence `memcpy` loads.
* `hit_select` compacts resident hits in lane order: the output slot of a hit is the number of
  hits before it.
* The CPU-order kernel reduces 8 lanes with shuffle offsets 4, 1, 2 (SSE `add`, `hadd`, `hadd`).

## Form template identifiers
`H/FF/ROW_GU/ROW_D/SC_GU/SC_D/O_*` → same names; `row_dot_s2_q8` → `s2dot`; gate/up write
index → `guOut`; `hit_select_kernel` → `slotOf`; `row_dot_cpu_order` tail → `cpuTree`;
`moe_hit_grouped_scratch_bytes` → `scratchBytes`.
-/

namespace S2GroupedExpert

-- § blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- Expert intermediate size. -/
def FF : Nat := 640
/-- Bytes of 2-bit codes per gate/up row. -/
def ROW_GU : Nat := H / 4
/-- Bytes of 2-bit codes per down row. -/
def ROW_D : Nat := FF / 4
/-- fp16 scales per gate/up row. -/
def SC_GU : Nat := H / 64
/-- fp16 scales per down row. -/
def SC_D : Nat := FF / 64
/-- Start of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Start of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Start of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
/-- Total blob size. -/
def BLOB : Nat := O_D_SCALES + H * SC_D * 2

/-- The literal geometry restated in the kernel: 640 / 160 code bytes, 40 / 10 scales per row,
    and the region offsets. -/
theorem geometry :
    ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 ∧ O_D_CODES = 819200 ∧
    O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 ∧ BLOB = 1382400 := by decide

/-- Density: the blob holds exactly 2 bits per weight plus one fp16 per 64 weights. -/
theorem blob_density : BLOB * 8 = (3 * H * FF) * 2 + (3 * H * FF / 64) * 16 := by decide

/-- A gate/up row `i < 2·FF` and its scales stay inside their regions. -/
theorem gu_row_regions (i : Nat) (h : i < 2 * FF) :
    i * ROW_GU + ROW_GU ≤ O_D_CODES ∧
    O_GU_SCALES + i * SC_GU * 2 + SC_GU * 2 ≤ O_D_SCALES := by
  simp only [O_D_CODES, O_GU_SCALES, O_D_SCALES, ROW_GU, SC_GU, H, FF, ROW_D] at *
  omega

/-- A down row `r < H` and its scales stay inside their regions. -/
theorem down_row_regions (r : Nat) (h : r < H) :
    O_D_CODES + r * ROW_D + ROW_D ≤ O_GU_SCALES ∧
    O_D_SCALES + r * SC_D * 2 + SC_D * 2 ≤ BLOB := by
  simp only [O_D_CODES, O_GU_SCALES, O_D_SCALES, BLOB, ROW_GU, ROW_D, SC_GU, SC_D, H, FF] at *
  omega

/-- 32-element chunk `c` uses weight scale `c >> 1`: one fp16 per two chunks (64 weights). -/
theorem chunk_scale (c : Nat) (h : c < H / 32) : c / 2 < SC_GU := by
  simp only [SC_GU, H] at *; omega

-- § integer arithmetic of a chunk

/-- Exact integer dot. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Integer sum. -/
def lsum : List Int → Int
  | [] => 0
  | a :: as => a + lsum as

/-- The header identity: `Σ (c_j − 1)·x_j = Σ c_j·x_j − Σ x_j`.  Both right-hand sums are
    INT8×INT8 DP4A chains, so the chunk is exact in integer before one float multiply. -/
theorem s2dot : ∀ (c x : List Int), c.length = x.length →
    dot (c.map (· - 1)) x = dot c x - lsum x
  | [], [], _ => by simp [dot, lsum]
  | a :: as, b :: bs, h => by
    have ih := s2dot as bs (by simpa using h)
    simp only [List.map, dot, lsum]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

/-- A code byte is exactly the LSB-first packing of its four 2-bit fields. -/
theorem code_byte_fields (b : Nat) (h : b < 256) :
    b = b % 4 + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) := by omega

/-- The bit expansion `(c & 3) | ((c>>2)&3)<<8 | …` of every code byte has four bytes `< 4`
    (each a valid non-negative int8 lane for DP4A). -/
theorem code_word_lanes : ∀ c, c < 256 →
    (c &&& 3) < 4 ∧ ((c >>> 2) &&& 3) < 4 ∧ ((c >>> 4) &&& 3) < 4 ∧ ((c >>> 6) &&& 3) < 4 := by
  intro c _
  refine ⟨?_, ?_, ?_, ?_⟩ <;> exact Nat.lt_of_le_of_lt Nat.and_le_right (by decide)

/-- Q8_0 int8 payload of block `c` (at `34c + 2`) is 4-byte aligned exactly for odd `c`; the
    first block's is not, which is the fault the `memcpy` load removes. -/
theorem q8_0_payload_alignment (c : Nat) : (34 * c + 2) % 4 = 0 ↔ c % 2 = 1 := by omega

-- § gate-major gate/up layout

/-- Output index of row-slot `i` of hit `h` among `n` hits. -/
def guOut (n h i : Nat) : Nat :=
  if i % 2 = 1 then n * FF + h * FF + i / 2 else h * FF + i / 2

/-- Every write lands inside the `2 · n · FF` gate/up buffer. -/
theorem guOut_bound (n h i : Nat) (hh : h < n) (hi : i < 2 * FF) : guOut n h i < 2 * n * FF := by
  unfold guOut; simp only [FF] at *; split <;> omega

/-- Gate rows of all hits occupy `[0, n·FF)` and up rows `[n·FF, 2·n·FF)`: the swiglu over the
    first half sees each hit's gate paired with the SAME hit's up at offset `n·FF`. -/
theorem guOut_pairing (n h r : Nat) :
    guOut n h (2 * r) = h * FF + r ∧ guOut n h (2 * r + 1) = n * FF + (h * FF + r) := by
  unfold guOut
  constructor <;> split <;> omega

/-- Distinct (hit, slot) pairs write distinct outputs. -/
theorem guOut_injective (n h1 h2 i1 i2 : Nat) (hh1 : h1 < n) (hh2 : h2 < n)
    (hi1 : i1 < 2 * FF) (hi2 : i2 < 2 * FF) (e : guOut n h1 i1 = guOut n h2 i2) :
    h1 = h2 ∧ i1 = i2 := by
  unfold guOut at e; simp only [FF] at *
  split at e <;> split at e <;> omega

-- § device-side hit compaction

/-- Output slot of lane `i`: the number of resident hits strictly before it (`popc(hit & mask)`). -/
def slotOf (hits : List Bool) (i : Nat) : Nat := (hits.take i).count true

/-- Prefix counts are monotone in the prefix length. -/
theorem slotOf_mono (hits : List Bool) {i j : Nat} (h : i ≤ j) : slotOf hits i ≤ slotOf hits j := by
  unfold slotOf
  have : hits.take i = (hits.take j).take i := by rw [List.take_take, Nat.min_eq_left h]
  rw [this]
  exact (List.take_sublist i _).count_le true

/-- A hit at lane `i` raises the prefix count by one. -/
theorem slotOf_succ (hits : List Bool) (i : Nat) (hi : i < hits.length) (hh : hits[i] = true) :
    slotOf hits (i + 1) = slotOf hits i + 1 := by
  unfold slotOf
  rw [List.take_add_one, List.getElem?_eq_getElem hi, hh]
  simp

/-- Compaction is order-preserving and injective: earlier hits get strictly smaller slots. -/
theorem slotOf_strict (hits : List Bool) {i j : Nat} (hij : i < j) (hi : i < hits.length)
    (hh : hits[i] = true) : slotOf hits i < slotOf hits j := by
  have h1 := slotOf_succ hits i hi hh
  have h2 := slotOf_mono hits (show i + 1 ≤ j by omega)
  omega

/-- Every hit's slot is below the hit count written to `*count` (`popc(hit)`). -/
theorem slotOf_lt_count (hits : List Bool) (i : Nat) (hi : i < hits.length)
    (hh : hits[i] = true) : slotOf hits i < hits.count true := by
  have h1 := slotOf_succ hits i hi hh
  have h2 : slotOf hits (i + 1) ≤ hits.count true := by
    unfold slotOf; exact (List.take_sublist _ _).count_le true
  omega

-- § CPU-order reduction tree

/-- `__shfl_down_sync(v, off, 8)`: lane `l` reads lane `l + off` when it is inside the 8-lane
    segment, otherwise keeps its own value. -/
def shflDown {α : Type} (v : Fin 8 → α) (off : Nat) (l : Fin 8) : α :=
  if h : l.val + off < 8 then v ⟨l.val + off, h⟩ else v l

/-- One reduction step `acc += shfl_down(acc, off)`. -/
def step {α : Type} [Add α] (v : Fin 8 → α) (off : Nat) : Fin 8 → α :=
  fun l => v l + shflDown v off l

/-- The source's tree: offsets 4, then 1, then 2. -/
def cpuTree {α : Type} [Add α] (v : Fin 8 → α) : α := step (step (step v 4) 1) 2 0

/-- Lane 0 of the 4,1,2 tree computes exactly `((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7))` — the
    `_mm_add_ps(low,high)` then two `_mm_hadd_ps` bracketing — for any `+`, including
    non-associative float addition. -/
theorem cpuTree_lane0 {α : Type} [Add α] (v : Fin 8 → α) :
    cpuTree v = ((v 0 + v 4) + (v 1 + v 5)) + ((v 2 + v 6) + (v 3 + v 7)) := rfl

/-- The standard 4,2,1 tree, for comparison. -/
def stdTree {α : Type} [Add α] (v : Fin 8 → α) : α := step (step (step v 4) 2) 1 0

/-- The standard tree brackets differently: `((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))`; with
    non-associative float `+` this is a different expression, as the source warns. -/
theorem stdTree_lane0 {α : Type} [Add α] (v : Fin 8 → α) :
    stdTree v = ((v 0 + v 4) + (v 2 + v 6)) + ((v 1 + v 5) + (v 3 + v 7)) := rfl

/-- Free syntax of additions, used to compare bracketings without assuming associativity. -/
inductive Sum8 where
  | leaf (n : Nat)
  | add (a b : Sum8)
  deriving DecidableEq

instance : Add Sum8 := ⟨Sum8.add⟩

/-- As syntax (i.e. for a non-associative `+` such as IEEE float addition) the 4,1,2 tree and
    the standard 4,2,1 tree are different expressions. -/
theorem trees_differ :
    cpuTree (fun l : Fin 8 => Sum8.leaf l.val) ≠ stdTree (fun l : Fin 8 => Sum8.leaf l.val) := by
  decide

-- § scratch

/-- 16-byte round-up. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- `moe_hit_grouped_scratch_bytes`. -/
def scratchBytes (n nEmbd nFF : Nat) : Nat :=
  align16 (n * (2 * nFF) * 4) + align16 (n * (nFF / 32) * 34) +
  2 * align16 (n * (nFF / 32) * 4) + align16 ((nEmbd / 32) * 4)

/-- The carve-out used by `moe_hit_grouped_s2` (gate_up | h_q8_0 | h_scales) fits the
    advertised scratch. -/
theorem scratch_fits (n nEmbd nFF : Nat) :
    align16 (n * (2 * nFF) * 4) + align16 (n * (nFF / 32) * 34) + n * (nFF / 32) * 4 ≤
      scratchBytes n nEmbd nFF := by
  unfold scratchBytes align16; omega

end S2GroupedExpert
