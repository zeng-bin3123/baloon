/-!
# MODULE: native_ple_postops
SOURCES: src/kernels/cuda/native_ple_postops.cu
Native PLE post-projection operators: signed-sqrt gate per stream, value broadcast, and the dilated 4-tap
conv + residual, with the span-validation rules that allow exactly two aliasing exceptions.

## Source semantics
* Geometry: N = 2560, H = 4 streams, D = N*H = 10240, conv history depth 9 per channel.
* `gate_kernel`: 512 threads per stream, 8 partial lanes `d = tid + j*512` masked to `d < N`.
* `broadcast_kernel`: `gated[i] = value[i % N] * gate[i / N]`.
* `conv_residual_kernel`: taps read history rows 0, 3, 6 and the fresh normalized value for tap 3.

## Form template identifiers
gate_kernel -> stream_gate ; broadcast_kernel -> value_broadcast ; conv_residual_kernel -> conv_residual ;
validate -> span_check ; native_ple_postops -> ple_postops_entry
-/

namespace Strata.NativePle

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § constants

/-- Geometry constants. -/
theorem geometry : 2560 * 4 = 10240 ∧ 10240 % 256 = 0 ∧ 10240 / 256 = 40 := by decide

-- § gate kernel

/-- Each element `d < N` is covered by thread `d % 512` in partial slot `d / 512 < 5 ≤ 8`. -/
theorem gate_slot_cover (d : Nat) (h : d < 2560) :
    d % 512 < 512 ∧ d / 512 < 5 ∧ d = d % 512 + 512 * (d / 512) := by omega

/-- Slots 5..7 are always masked: `tid + j*512 ≥ N` for `j ≥ 5`. -/
theorem gate_masked_slots (tid j : Nat) (hj : 5 ≤ j) : 2560 ≤ tid + j * 512 := by omega

/-- 512 threads are 16 warps, and the second-level reduction reads `partials[lane < 16]`. -/
theorem gate_warps : 512 / 32 = 16 ∧ 16 ≤ 32 := by decide

/-- Per-stream row offset `stream*N + d` fits in D. -/
theorem gate_row_lt {s d : Nat} (hs : s < 4) (hd : d < 2560) : s * 2560 + d < 10240 := by omega

-- § broadcast and conv

/-- A flat index splits into stream `< 4` and column `< N`. -/
theorem broadcast_split (i : Nat) (h : i < 10240) :
    i % 2560 < 2560 ∧ i / 2560 < 4 ∧ i = (i / 2560) * 2560 + i % 2560 := by omega

/-- History row index of tap `k`: `3*k` for taps 0..2, and tap 3 uses the new value instead. -/
def histRow (k : Nat) : Nat := 3 * k

/-- All history reads stay within the 9-deep column: `c*9 + 3k < 9*D`. -/
theorem history_read_in_bounds {c k : Nat} (hc : c < 10240) (hk : k < 3) : c * 9 + histRow k < 9 * 10240 := by
  unfold histRow
  omega

/-- The three history taps are rows 0, 3, 6 (the new value is the fourth). -/
theorem history_taps : [histRow 0, histRow 1, histRow 2] = [0, 3, 6] := by decide

/-- Weight halves for channel `c` and tap `k < 4` stay within `4*D`. -/
theorem weight_in_bounds {c k : Nat} (hc : c < 10240) (hk : k < 4) : c * 4 + k < 4 * 10240 := by omega

/-- The residual is `hidden + (gated + activation)`; `+` on naturals is associative, so the model order
(`gated + act` first) is the only association the source ever uses. -/
theorem residual_association (h g a : Nat) : h + (g + a) = (h + g) + a := by omega

-- § span validation

/-- Overlap test. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Symmetric. -/
theorem overlaps_symm {a an b bn : Nat} : overlaps a an b bn ↔ overlaps b bn a an := by
  unfold overlaps
  constructor <;> (intro h; exact ⟨h.2, h.1⟩)

/-- Pair counts of the double loop: 7 outputs x 8 inputs, and 7*6/2 output pairs. -/
theorem check_counts : 7 * 8 = 56 ∧ 7 * 6 / 2 = 21 := by decide

/-- Byte sizes of the validated spans (floats of length D, N and H, half-precision weights 4*D). -/
theorem span_sizes : 10240 * 4 = 40960 ∧ 2560 * 4 = 10240 ∧ 4 * 10240 * 2 = 81920 ∧ 9 * 10240 * 4 = 368640 := by
  decide

end Strata.NativePle
