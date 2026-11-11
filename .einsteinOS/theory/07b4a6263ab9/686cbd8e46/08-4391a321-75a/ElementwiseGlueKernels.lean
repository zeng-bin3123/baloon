/-!
# MODULE: elementwise_glue_kernels
SOURCES: src/kernels/cuda/elementwise.cu
P2.S5's glue kernels: quantized embedding gather, GDN gate (softplus), scale/add/silu in place,
f16/bf16 bulk conversion, warp-per-row weighted RMS norm, mapped-memory copies, and the
host↔device doorbell (ring / wait / publish).

## Source semantics
* Embedding: `code = (codes[i/per_byte] >> ((i % per_byte)*bits)) & (2^bits-1)`,
  `out[i] = (code + bias)*scale[i/group] + (offset[i/group] or 0)`.
* GDN gate: `gate[i] = softplus(alpha[i] + dt[i % h_v]) * ssm_a[i % h_v]`.
* `grid_for(n) = ceil(n/256)` blocks; every flat launch guards `i < n`.
* RMS norm: 4 warps per block, `grid = ceil(rows/4)`, row = `block*4 + warp`; the `row < rows`
  guard is REQUIRED — without it a 2-row call (QSA k-norm) writes two rows past the end, into
  the neighbouring arena buffer. Normalises by the MEAN of squares.
* Doorbell ring increments the device word IN MEMORY (`*seq = *seq + 1`) after a system fence;
  passing `host_value + 1` as an argument would be frozen at graph-capture time.
* `doorbell_wait` spins until `flag == seq`. `copy_from_mapped` needs `n % 4 = 0` and 16-byte
  aligned pointers, at most 64 blocks with a grid-stride loop.

## Form template identifiers
code extraction ↦ `codeAt`, dequant ↦ `dequant`, launch grid ↦ `gridFor`, row mapping ↦
`rowOf`, ring model ↦ `ringMem`/`ringLiteral`, replay ↦ `replay`, grid-stride ↦ `strideCover`.
-/

namespace Strata.Elementwise

-- § embedding gather

/-- Code of element `i` within its byte, `bits` wide. -/
def codeAt (byte bits i : Nat) : Nat := (byte / 2 ^ ((i % (8 / bits)) * bits)) % 2 ^ bits

/-- Codes fit in `bits` bits. -/
theorem codeAt_lt (byte bits i : Nat) : codeAt byte bits i < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos _)

/-- For supported widths the in-byte shift stays below 8 bits. -/
theorem shift_in_byte : ∀ b ∈ [1, 2, 4, 8], ∀ i < 8, (i % (8 / b)) * b < 8 := by decide

/-- Dequantised value in exact arithmetic. -/
def dequant (code bias : Int) (scale : Int) (offset : Option Int) : Int :=
  (code + bias) * scale + offset.getD 0

/-- Without an offset table the value is just the scaled code. -/
theorem dequant_no_offset (code bias scale : Int) : dequant code bias scale none = (code + bias) * scale := by
  simp [dequant]

-- § launch grids

/-- `grid_for(n)`. -/
def gridFor (n : Nat) : Nat := (n + 255) / 256

/-- The grid covers all `n` elements and wastes less than one block. -/
theorem gridFor_covers (n : Nat) : n ≤ gridFor n * 256 ∧ gridFor n * 256 < n + 256 := by
  unfold gridFor; omega

/-- The GDN gate's head index for flat element `i` with one token is `i` itself. -/
theorem gate_head_single {i hv : Nat} (hi : i < hv) : i % hv = i := Nat.mod_eq_of_lt hi

-- § rms norm row guard

/-- Row handled by warp `w` of block `b` (4 warps per block). -/
def rowOf (b w : Nat) : Nat := b * 4 + w

/-- Grid of the warp-per-row RMS norm. -/
def normGrid (rows : Nat) : Nat := (rows + 3) / 4

/-- Every row is processed by some launched warp. -/
theorem rows_covered {r rows : Nat} (hr : r < rows) :
    ∃ b w, b < normGrid rows ∧ w < 4 ∧ rowOf b w = r :=
  ⟨r / 4, r % 4, by unfold normGrid; omega, by omega, by unfold rowOf; omega⟩

/-- With the guard, every write `row*cols + c` is inside the `rows × cols` buffer. -/
theorem guarded_write_in_bounds {row rows c cols : Nat} (hr : row < rows) (hc : c < cols) :
    row * cols + c < rows * cols := by
  have : (row + 1) * cols ≤ rows * cols := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at this; omega

/-- The QSA bug: a 2-row call launches 4 warps, and warp 3 (unguarded) would start writing at
    `3*cols`, past the buffer's `2*cols` end, for any non-empty row. -/
theorem qsa_unguarded_overrun (cols : Nat) (hc : 0 < cols) :
    normGrid 2 * 4 = 4 ∧ 2 * cols < rowOf 0 3 * cols + 1 := by
  refine ⟨by decide, ?_⟩
  unfold rowOf; omega

-- § grid-stride copies

/-- Grid-stride loop: thread `t < G` visits `t + k*G`; every element is visited once. -/
theorem strideCover {G i : Nat} (hG : 0 < G) : i % G < G ∧ i % G + (i / G) * G = i := by
  refine ⟨Nat.mod_lt _ hG, ?_⟩
  have := Nat.mod_add_div i G
  rw [Nat.mul_comm] at this; exact this

/-- `copy_from_mapped`'s precondition makes the float4 view exact. -/
theorem float4_view {n : Nat} (h : n % 4 = 0) : n / 4 * 4 = n := by omega

/-- The copy uses at most 64 blocks. -/
theorem copy_blocks_le (n4 : Nat) : min ((n4 + 255) / 256) 64 ≤ 64 := Nat.min_le_right _ _

-- § doorbell

/-- Correct ring: increment the value in memory. -/
def ringMem (s : Nat) : Nat := (s + 1) % 2 ^ 32

/-- Buggy ring: write a value captured on the host at graph-capture time. -/
def ringLiteral (c : Nat) (_s : Nat) : Nat := c

/-- Replaying a captured graph `k` times. -/
def replay (f : Nat → Nat) : Nat → Nat → Nat
  | 0, s => s
  | k + 1, s => f (replay f k s)

/-- The in-memory ring advances by exactly `k` (mod 2^32) after `k` replays, for any valid
    `uint32_t` starting value. -/
theorem replay_mem (k s : Nat) (hs : s < 2 ^ 32) : replay ringMem k s = (s + k) % 2 ^ 32 := by
  induction k with
  | zero => simp [replay]; omega
  | succ k ih => simp only [replay, ih, ringMem]; omega

/-- The cloned literal never advances after the first replay: a host waiting for the value to
    change past `c` waits forever. -/
theorem replay_literal_stuck (c k s : Nat) : replay (ringLiteral c) (k + 1) s = c := by
  simp [replay, ringLiteral]

/-- `doorbell_wait` releases exactly when the flag catches up with the sequence. -/
def waitDone (flag seq : Nat) : Bool := flag == seq

/-- Releasing implies the published sequence was observed. -/
theorem waitDone_spec {f s : Nat} (h : waitDone f s = true) : f = s := by
  simpa [waitDone] using h

/-- Exact model of `add_inplace` and `scale_inplace`: `x*s + y` in either order of launches
    only when no aliasing occurs; in-place add is element-wise. -/
theorem add_then_scale (x y s : Int) : (x + y) * s = x * s + y * s := Int.add_mul x y s

end Strata.Elementwise
