/-!
# MODULE: elementwise_glue_kernels
Glue kernels: embedding dequantisation, gate indexing, warp-per-row RMS norm launch, bounded 1-D launches,
and the host/device doorbell protocol.

## Source semantics
`embedding_gather` unpacks `8/bits` codes per byte, adds the code bias, multiplies by the group scale and
adds the optional group offset.  `gdn_gate_kernel` is launched over `n_tokens*h_v` elements but guards
`i >= h_v`, so only the first token row is processed (exact for the real one-token call).
`rms_norm_weighted` runs 4 warps per block, one row per warp; the grid is rounded up, so the row guard is
needed (rows = 2 launches 8 warps).  `grid_for n` is `ceil(n/256)`; `n = 0` gives a zero grid, which is why
every launcher returns early.  The doorbell RING increments device memory; passing the incremented value as
a launch argument is evaluated once at capture time and makes every replay ring the same value.

## Form template identifiers
- `dequant`   -> `(code + bias) * scale + offset`
- `grid_for`  -> `gridFor`
- `ring`      -> `incRing` (correct) / `litRing` (capture-time literal)
-/

namespace Strata.Kernels.Elementwise

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Row-major addresses are unique. -/
theorem split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
    (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
  have h1 : (a * n + c) / n = a := div_of_split_aux hc
  have h2 : (a' * n + c') / n = a' := div_of_split_aux hc'
  have ha : a = a' := by rw [← h1, ← h2, h]
  subst ha
  exact ⟨rfl, by omega⟩

-- § embedding dequantisation

/-- `(code + bias) * scale + offset`, with a missing offset read as zero. -/
def dequant (code bias : Nat) (scale : Int) (offset : Option Int) : Int :=
  ((code + bias : Nat) : Int) * scale + offset.getD 0

/-- Without offsets the value is just `(code + bias) * scale`. -/
theorem dequant_no_offset (code bias : Nat) (scale : Int) :
    dequant code bias scale none = ((code + bias : Nat) : Int) * scale := by
  simp [dequant]

/-- A `bits`-bit code field is below `2^bits`. -/
theorem code_lt (byte shift bits : Nat) : (byte / 2 ^ shift) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- Codes per byte: 8 for 1-bit, 4 for 2-bit, 2 for 4-bit, 1 for 8-bit. -/
theorem codes_per_byte : 8 / 1 = 8 ∧ 8 / 2 = 4 ∧ 8 / 4 = 2 ∧ 8 / 8 = 1 := by
  decide

/-- Element `i` of a 2-bit stream: byte `i/4`, slot `i%4`. -/
theorem two_bit_split (i : Nat) : i = 4 * (i / 4) + i % 4 ∧ i % 4 < 4 := by
  omega

-- § gate indexing

/-- The gate kernel's guard `i < h_v` selects exactly the first token row. -/
theorem gate_first_row_only {i h_v : Nat} (hv : 0 < h_v) (hi : i < h_v) :
    i / h_v = 0 ∧ i % h_v = i :=
  ⟨Nat.div_eq_of_lt hi, Nat.mod_eq_of_lt hi⟩

/-- Conversely, a processed element lies in token row 0. -/
theorem gate_processed_in_row0 {i h_v : Nat} (hv : 0 < h_v) (h : i / h_v = 0) : i < h_v := by
  rcases Nat.lt_or_ge i h_v with h' | h'
  · exact h'
  · have := Nat.div_pos h' hv
    omega

/-- For the real single-token call every element is processed: `n = 1 * h_v`. -/
theorem gate_single_token_total {i h_v : Nat} (hi : i < 1 * h_v) : i < h_v := by
  omega

-- § bounded launches

/-- Blocks for `n` elements at 256 threads. -/
def gridFor (n : Nat) : Nat := (n + 255) / 256

/-- The launch covers all `n` elements. -/
theorem gridFor_cover (n : Nat) : n ≤ gridFor n * 256 := by
  unfold gridFor
  omega

/-- A zero-element call yields an illegal zero grid, hence the early return. -/
theorem gridFor_zero : gridFor 0 = 0 := by
  decide

/-- A positive-element call has a non-empty grid. -/
theorem gridFor_pos {n : Nat} (h : 0 < n) : 0 < gridFor n := by
  unfold gridFor
  omega

-- § warp-per-row rms norm launch

/-- Warps launched for `rows` rows at 4 warps per block. -/
def launchedWarps (rows : Nat) : Nat := ((rows + 3) / 4) * 4

/-- Every row has a warp: warp `r` lives in block `r/4`. -/
theorem rows_covered {rows r : Nat} (hr : r < rows) :
    r / 4 < (rows + 3) / 4 ∧ r / 4 * 4 + r % 4 = r := by
  omega

/-- The grid never launches fewer warps than rows, and fewer than four extra. -/
theorem launched_warps_bounds (rows : Nat) :
    rows ≤ launchedWarps rows ∧ launchedWarps rows < rows + 4 := by
  unfold launchedWarps
  omega

/-- THE QSA BUG WITNESS: two rows launch one 4-warp block, so exactly two warps are out of range
and the row guard is mandatory. -/
theorem qsa_guard_needed : launchedWarps 2 = 4 ∧ launchedWarps 2 - 2 = 2 := by
  decide

/-- With the guard, a row index `>= rows` writes nothing; without it row `rows..` would write `cols` floats
each past the buffer. -/
theorem out_of_range_rows {rows : Nat} (h : rows % 4 ≠ 0) : rows < launchedWarps rows := by
  unfold launchedWarps
  omega

-- § mapped copies

/-- `copy_from_mapped` moves `n/4` float4 when `n % 4 = 0`, which is all `n` floats. -/
theorem float4_total {n : Nat} (h : n % 4 = 0) : (n / 4) * 4 = n := by
  omega

/-- The capped grid `min((n4+255)/256, 64)` is non-empty for non-empty input. -/
theorem capped_grid_pos {n4 : Nat} (h : 0 < n4) : 0 < min ((n4 + 255) / 256) 64 := by
  omega

/-- `doorbell_publish` copies `k` ids with threads `< k`; the guard `k <= 1024` fits the block. -/
theorem publish_threads {k t : Nat} (hk : k ≤ 1024) (ht : t < k) : t < 1024 := by
  omega

-- § doorbell protocol

/-- Correct ring: increments the memory it reads at replay time. -/
def incRing (seq : Nat) : Nat := seq + 1

/-- Broken ring: the incremented value was computed on the host at capture time and baked in. -/
def litRing (c : Nat) (_seq : Nat) : Nat := c

/-- The correct ring changes the counter on every replay. -/
theorem incRing_changes (seq : Nat) : incRing seq ≠ seq := by
  unfold incRing
  omega

/-- After two replays the correct ring has advanced by two. -/
theorem incRing_twice (seq : Nat) : incRing (incRing seq) = seq + 2 := by
  unfold incRing
  omega

/-- The captured literal makes the second replay a no-op, so a host waiting for a change waits forever. -/
theorem litRing_stalls (c seq : Nat) : litRing c (litRing c seq) = litRing c seq :=
  rfl

/-- The device wait is satisfied when the flag reaches the sequence value it read. -/
def waitReady (flag want : Nat) : Prop := flag = want

/-- After the host copies the sequence into the flag the wait is released. -/
theorem wait_released (seq : Nat) : waitReady seq seq :=
  rfl

/-- `wait_flag_ge`: a counter that only grows keeps satisfying `flag >= value`. -/
theorem flag_ge_monotone {f1 f2 v : Nat} (hle : f1 ≤ f2) (h : v ≤ f1) : v ≤ f2 := by
  omega

end Strata.Kernels.Elementwise
