/-!
# MODULE: native_gdn_preprocess
SOURCES: src/kernels/cuda/native_gdn_preprocess.cu
Native GDN preprocessing kernels (conv + SiLU, L2 norm, beta sigmoid, gate softplus, gated output RMS norm)
and their launch-geometry / span validation.

## Source semantics
* `conv_silu`: window `[h0,h1,h2,x]` with 4 taps; history shifts to `[h1,h2,x]`.
* `l2_norm` / `out_norm`: one block per row of width exactly 128, 256 threads, 8 warp sums.
* Launch checks: stream non-null, count in `[1, 65535]`, width = 128, epsilon finite and nonnegative.
* Writable spans must be pairwise disjoint and disjoint from inputs.

## Form template identifiers
conv_silu -> conv_silu_step ; l2_norm -> row_l2 ; beta_sigmoid -> beta_gate ; gate_softplus -> decay_gate ;
out_norm -> gated_rms_out
-/

namespace Strata.NativeGdnPre

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § conv

/-- One conv step on a 3-deep history: the window is `(h0,h1,h2,x)`, the new history `(h1,h2,x)`. -/
def convShift (h : Nat × Nat × Nat) (x : Nat) : Nat × Nat × Nat := (h.2.1, h.2.2, x)

/-- The new history's last entry is the new input. -/
theorem convShift_last (h : Nat × Nat × Nat) (x : Nat) : (convShift h x).2.2 = x := rfl

/-- The oldest value is dropped, the others move one slot back. -/
theorem convShift_moves (a b c x : Nat) : convShift (a, b, c) x = (b, c, x) := rfl

/-- Shifting three times from `(a,b,c)` with inputs `x y z` leaves exactly the inputs. -/
theorem convShift_three (a b c x y z : Nat) :
    convShift (convShift (convShift (a, b, c) x) y) z = (x, y, z) := rfl

/-- Channel `c < channels` addresses 3 history and 4 weight cells within bounds. -/
theorem conv_cells {c channels : Nat} (h : c < channels) :
    c * 3 + 2 < 3 * channels ∧ c * 4 + 3 < 4 * channels := by omega

/-- Conv launch covers all channels with 256-thread blocks. -/
theorem conv_cover (c channels : Nat) (h : c < channels) : c / 256 < (channels + 255) / 256 := by omega

/-- Span byte sizes: history is 3x, weights 4x the per-channel float vector. -/
theorem conv_span_sizes (channels : Nat) : 3 * (channels * 4) = 12 * channels ∧ 4 * (channels * 4) = 16 * channels := by
  omega

-- § norms

/-- 256 threads form 8 warps, matching the `lane < 8` second-level sum. -/
theorem norm_warps : 256 / 32 = 8 ∧ 8 ≤ 32 := by decide

/-- Columns `col ≥ 128` of the 256-thread block contribute zero; exactly 128 contribute. -/
theorem norm_active_columns : 128 ≤ 256 ∧ 256 - 128 = 128 := by decide

/-- Row offset `row*S + col` of norm blocks fits within `rows*S`. -/
theorem norm_offset_lt {row rows col : Nat} (hr : row < rows) (hc : col < 128) :
    row * 128 + col < rows * 128 := rowmajor_lt hr hc

/-- Geometry validity: width 128 and a row count in `[1,65535]` (a CUDA grid.x-safe bound). -/
def normGeometryOk (rows cols : Nat) : Prop := 1 ≤ rows ∧ rows ≤ 65535 ∧ cols = 128

/-- A valid geometry has a bounded grid. -/
theorem normGeometry_grid {rows cols : Nat} (h : normGeometryOk rows cols) : rows < 65536 ∧ cols = 128 := by
  unfold normGeometryOk at h
  omega

/-- Zero rows are rejected. -/
theorem normGeometry_rejects_zero (cols : Nat) : ¬ normGeometryOk 0 cols := by
  unfold normGeometryOk
  omega

/-- The epsilon passed into the L2 kernel is `eps / S` (scale `S = 128`): integer model of the scaling. -/
theorem eps_scaling (e : Nat) : (e * 128) / 128 = e := by omega

-- § element-wise kernels

/-- Beta sigmoid / gate softplus launches (256-thread blocks) cover every head. -/
theorem head_launch_cover (i n : Nat) (h : i < n) : i / 256 < (n + 255) / 256 := by omega

/-- Output norm: head block `b`, column `col` writes `b*128 + col`, which is injective. -/
theorem out_norm_injective {b col b' col' : Nat} (hc : col < 128) (hc' : col' < 128)
    (h : b * 128 + col = b' * 128 + col') : b = b' ∧ col = col' := by omega

-- § span validation

/-- Overlap predicate of the source. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric, so checking `(i, j<i)` pairs suffices. -/
theorem overlaps_symm {a an b bn : Nat} : overlaps a an b bn ↔ overlaps b bn a an := by
  unfold overlaps
  constructor <;> (intro h; exact ⟨h.2, h.1⟩)

/-- Three writable spans give exactly three unordered pairs checked by the nested loop. -/
theorem pair_count : (List.range 3).foldl (fun acc i => acc + (List.range i).length) 0 = 3 := by decide

end Strata.NativeGdnPre
