/-!
# MODULE: buffer_arena_sizing
Buffer allocation, alignment, and arena sizing for GDN, QSA, MoE,
and block subsystems. All regions are 16-byte aligned within arenas.

## Source semantics
Every session buffer is carved from a device arena sized once at startup.
Zero token-path allocations (P2.T10). Key sizing formulas:
- Q8_K: (n/256) * 292 bytes per n elements
- 16-byte alignment: (n + 15) / 16 * 16
- 256-byte alignment for expert slots: (n + 255) / 256 * 256
- Session state alignment: 256 bytes

The Cursor pattern advances a pointer through an arena, 16-byte aligned,
so every region starts where vector loads expect it.

## Form template identifiers
- gdn_buffers_bytes → gdnBufBytes
- qsa_buffers_bytes → qsaBufBytes
- moe_buffers_bytes → moeBufBytes
- block_buffers_bytes → blkBufBytes
- session_bytes → sessBytes
- align_up → alignUp
-/

-- ============================================================
-- § 16-byte alignment
-- ============================================================

/-- 16-byte alignment rounds up. -/
theorem align16_ge (n : Nat) : (n + 15) / 16 * 16 ≥ n := by omega

/-- 16-byte alignment produces a multiple of 16. -/
theorem align16_divisible (n : Nat) : ((n + 15) / 16 * 16) % 16 = 0 := by omega

-- ============================================================
-- § 256-byte alignment (expert slot alignment)
-- ============================================================

/-- 256-byte alignment rounds up. -/
theorem align256_ge (n : Nat) : (n + 255) / 256 * 256 ≥ n := by omega

/-- 256-byte alignment produces a multiple of 256. -/
theorem align256_divisible (n : Nat) : ((n + 255) / 256 * 256) % 256 = 0 := by omega

-- ============================================================
-- § Q8_K buffer sizing
-- (n/256) * 292 bytes for n elements (n must be multiple of 256)
-- ============================================================

/-- Q8_K buffer is at least as large as the element count when n is
    a multiple of 256, since 292 > 256. -/
theorem q8k_size (n : Nat) (h : n % 256 = 0) :
    n / 256 * 292 ≥ n := by omega

/-- Concrete Q8_K size: 2560 elements → 2920 bytes. -/
theorem q8k_concrete_2560 : 2560 / 256 * 292 = 2920 := by omega

/-- Concrete Q8_K size: 6144 elements (value_dim) → 7008 bytes. -/
theorem q8k_concrete_6144 : 6144 / 256 * 292 = 7008 := by omega

-- ============================================================
-- § Session state alignment
-- ============================================================

/-- Session state uses 256-byte alignment. -/
def sessionStateAlign : Nat := 256

/-- align_up rounds to a given alignment. -/
def alignUp (n a : Nat) : Nat := (n + a - 1) / a * a

/-- align_up is at least n for positive alignment. -/
theorem align_up_ge (n a : Nat) (ha : 0 < a) : alignUp n a ≥ n := by
  unfold alignUp; omega

/-- align_up produces a multiple of a for positive alignment. -/
theorem align_up_divisible (n a : Nat) (ha : 0 < a) : (alignUp n a) % a = 0 := by
  unfold alignUp; omega

-- ============================================================
-- § Arena sum: total is sum of aligned parts + safety margin
-- ============================================================

/-- The arena total includes all parts plus a safety margin. -/
theorem arena_with_margin (partsSum margin : Nat) (hm : 0 < margin) :
    partsSum < partsSum + margin := by omega

-- ============================================================
-- § Cursor pattern: monotonic pointer advance
-- ============================================================

/-- Each take advances the cursor by at least the requested amount. -/
theorem cursor_advances (used requested : Nat) :
    used + requested ≤ used + ((requested + 15) / 16 * 16) := by omega

/-- Cursor regions do not overlap (sequential allocation). -/
theorem cursor_no_overlap (used1 size1 used2 : Nat)
    (h : used2 = used1 + ((size1 + 15) / 16 * 16)) :
    used1 + size1 ≤ used2 := by omega

-- ============================================================
-- § Fold sum for arena sizing
-- ============================================================

/-- Sum of a list of aligned parts. -/
def sumAligned (parts : List Nat) : Nat :=
  parts.foldl (fun acc p => acc + (p + 15) / 16 * 16) 0

/-- Sum of aligned parts is non-negative. -/
theorem sumAligned_nonneg (parts : List Nat) :
    0 ≤ sumAligned parts := by
  exact Nat.zero_le _

-- ============================================================
-- § GDN state size per layer
-- ============================================================

/-- GDN state has two components: recurrence and conv history. -/
theorem gdn_state_two_parts (ssmS vHeads dConv convCh : Nat)
    (hs : 0 < ssmS) (hv : 0 < vHeads) (hd : 1 < dConv) (hc : 0 < convCh) :
    0 < ssmS * vHeads * ssmS + convCh * (dConv - 1) := by omega

-- ============================================================
-- § PLE history size
-- ============================================================

/-- PLE conv history: NG_HIST rows of hc_dim floats, 4 bytes each. -/
theorem ple_hist_positive (ngHist hcDim : Nat)
    (hh : 0 < ngHist) (hd : 0 < hcDim) :
    0 < ngHist * hcDim * 4 := by omega
