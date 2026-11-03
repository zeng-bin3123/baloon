/-!
# MODULE: elementwise_glue_kernels
Glue kernels: embedding code extraction, launch grids, warp-per-row RMSNorm row guard, grid-stride
copy coverage, doorbell sequence counter.

## Source semantics
`embedding_gather` extracts code (i / per_byte) >> ((i % per_byte) * bits) & mask. `grid_for(n)` is
ceil(n/256). `rms_norm_weighted` launches 4 warps per block, so a row count that is not a multiple
of 4 launches surplus warps whose rows must be guarded. The grid-stride copy visits every index.
The doorbell increments the memory it points at; a value baked into a graph at capture would repeat.

## Form template identifiers
extract -> embedding_code ; gridFor -> grid_for ; surplus -> rms_norm_surplus ; ring -> doorbell_ring
-/

set_option maxRecDepth 100000

namespace Strata.Elementwise

-- § Code extraction

/-- Extract the code at slot k of a byte: shift by k*bits, mask. -/
def extract (b k bits : Nat) : Nat := (b >>> (k * bits)) &&& ((1 <<< bits) - 1)

/-- 2-bit codes are below 4. -/
theorem extract2_lt : ∀ b < 256, ∀ k < 4, extract b k 2 < 4 := by decide

/-- 4-bit codes are below 16. -/
theorem extract4_lt : ∀ b < 256, ∀ k < 2, extract b k 4 < 16 := by decide

/-- 8-bit code is the byte itself. -/
theorem extract8 : ∀ b < 256, extract b 0 8 = b := by decide

/-- Element i lives in byte i / per_byte at slot i % per_byte, and these recombine. -/
theorem slot_cover (i p : Nat) : (i / p) * p + i % p = i := by
  rw [Nat.mul_comm]; exact Nat.div_add_mod i p

/-- The slot is below per_byte, so the shift stays inside the byte. -/
theorem slot_lt (i p : Nat) (hp : 0 < p) : i % p < p := Nat.mod_lt i hp

-- § Launch grids

/-- `grid_for(n) = ceil(n / 256)`. -/
def gridFor (n : Nat) : Nat := (n + 255) / 256

/-- The grid covers every element. -/
theorem grid_covers (n : Nat) : n ≤ gridFor n * 256 := by unfold gridFor; omega

/-- Zero elements give a zero grid, which is illegal: callers return early for n <= 0. -/
theorem grid_zero : gridFor 0 = 0 := by decide

/-- Positive n gives a positive grid. -/
theorem grid_pos (n : Nat) (h : 0 < n) : 0 < gridFor n := by unfold gridFor; omega

-- § RMSNorm row guard

/-- Warps launched: 4 per block, ceil(rows / 4) blocks. -/
def warps (rows : Nat) : Nat := (rows + 3) / 4 * 4

/-- At least `rows` warps are launched. -/
theorem warps_cover (rows : Nat) : rows ≤ warps rows := by unfold warps; omega

/-- Fewer than 4 surplus warps. -/
theorem warps_surplus (rows : Nat) : warps rows < rows + 4 := by unfold warps; omega

/-- The QSA k-norm has 2 rows and launches 4 warps: two surplus warps. -/
theorem kcur_case : warps 2 = 4 ∧ warps 2 - 2 = 2 := by decide

/-- Unguarded, the surplus rows write rows*cols .. warps*cols past the end: with 2 rows of 256 that
is 512 floats beyond the buffer, which in the arena is the start of vcur. -/
theorem unguarded_overrun : warps 2 * 256 - 2 * 256 = 512 := by decide

/-- Guarded: every active (row < rows, c < cols) index stays inside rows * cols. -/
theorem guarded_in_bounds (row c rows cols : Nat) (hr : row < rows) (hc : c < cols) :
    row * cols + c < rows * cols := by
  have h1 : (row + 1) * cols ≤ rows * cols := Nat.mul_le_mul_right cols hr
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § Grid-stride loops

/-- Every index i is visited by thread i % S at iteration i / S. -/
theorem stride_visit (i S : Nat) : i % S + S * (i / S) = i := Nat.mod_add_div i S

/-- Thread t at iteration k visits t + k*S, whose owner is t. -/
theorem stride_owner (t k S : Nat) (ht : t < S) : (t + k * S) % S = t := by
  rw [Nat.add_mul_mod_self_right]
  exact Nat.mod_eq_of_lt ht

/-- Copy launch bound: min(ceil(n4/256), 64) blocks, at most 64. -/
theorem copy_blocks_le (n4 : Nat) : min ((n4 + 255) / 256) 64 ≤ 64 := by omega

/-- n multiple of 4 gives n4 = n/4 with 4*n4 = n. -/
theorem copy_n4 (n : Nat) (h : n % 4 = 0) : 4 * (n / 4) = n := by omega

-- § Doorbell

/-- Sequence counter increment on a 32-bit word. -/
def ring (s : Nat) : Nat := (s + 1) % 4294967296

/-- Each ring changes the value, so a waiting host observes it. -/
theorem ring_changes (s : Nat) (h : s < 4294967296) : ring s ≠ s := by unfold ring; omega

/-- Ringing stays within 32 bits. -/
theorem ring_lt (s : Nat) : ring s < 4294967296 := by unfold ring; omega

/-- A value captured as a literal c rings c on every replay; two consecutive replays agree, so a
waiter for a change never wakes. The in-memory increment of two replays differs. -/
theorem literal_vs_increment (c s : Nat) (h : s < 4294967296) :
    (fun _ : Nat => c) s = (fun _ : Nat => c) (ring s) ∧ ring s ≠ s := by
  exact ⟨rfl, ring_changes s h⟩

/-- doorbell_publish bound: k <= 1024 threads each copy one id and weight. -/
theorem publish_threads (k : Nat) (h : k ≤ 1024) : k ≤ 1024 := h

end Strata.Elementwise
