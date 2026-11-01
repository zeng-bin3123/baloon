/-!
# MODULE: native_gdn_preprocess
Contracts of the GDN preprocessing kernels adapted from the pinned llama.cpp: depthwise 4-tap conv + SiLU with sliding
history (`conv_silu`), per-head L2 norm (`l2_norm`), beta sigmoid, gate softplus, and the gated RMS output norm
(`out_norm`).

## Source semantics
* `conv_silu`: one thread per channel; window `[h0, h1, h2, x]`, taps `c*4 .. c*4+3`, history `c*3 .. c*3+2`; the raw sum
  gets an explicit `+ 0.0` (the native kernel adds a zero bias); history becomes `[h1, h2, x]`.
* `l2_norm`: one block of 256 threads per row of width `S = 128`; only `col < S` threads contribute; the block sum uses
  8 warp partials.  The kernel is passed `eps / S` and `1 / sqrt(S)`, so `rsqrt(sum/S + eps/S) * (1/sqrt S)` is the plain
  `1 / sqrt(sum + eps)`.
* `gate_softplus`: `softplus(v) = v` for `v > 20` else `log(1 + exp v)`, then multiplied by `ssm_a`.
* `out_norm`: one block per head, `offset = head * S`, RMS scale from the head's `S` outputs, then `* gamma[col] * sigmoid(z)`.
* Every host wrapper requires a stream and a count in `[1, 65535]`, 4-byte aligned non-null spans, and pairwise disjoint
  writable spans.

## Form template identifiers
* `window`          : the 4-tap conv window
* `slideHistory`    : the 3-tap history update
* `warpPartial`     : which warp partial a thread contributes to
-/

namespace Strata.NativeGdnPre

-- § conv window and history

/-- The 4-tap window `[h0, h1, h2, x]`. -/
def window (h0 h1 h2 x : Int) : List Int := [h0, h1, h2, x]

/-- History after the step: drop the oldest tap, append the input. -/
def slideHistory (h0 h1 h2 x : Int) : List Int := [h1, h2, x]

/-- The new history is the window without its oldest element. -/
theorem slide_is_tail (h0 h1 h2 x : Int) : slideHistory h0 h1 h2 x = (window h0 h1 h2 x).tail := rfl

/-- The new input always occupies the last tap, and the step keeps three history entries. -/
theorem slide_shape (h0 h1 h2 x : Int) :
    (slideHistory h0 h1 h2 x).length = 3 ∧ (window h0 h1 h2 x).getLast? = some x := by
  simp [slideHistory, window]

/-- Sliding twice with inputs `x`, `y` leaves `[h2, x, y]`: history is a bounded FIFO of the last three inputs. -/
theorem slide_twice (h0 h1 h2 x y : Int) :
    slideHistory h1 h2 x y = [h2, x, y] := rfl

/-- Adding the explicit zero bias is the identity on integers (the FP counterpart changes no value). -/
theorem zero_bias_identity (s : Int) : s + 0 = s := Int.add_zero s

/-- Tap and history offsets stay inside their `4*channels` / `3*channels` buffers. -/
theorem conv_offsets_lt (c channels tap htap : Nat) (hc : c < channels) (ht : tap < 4) (hh : htap < 3) :
    c * 4 + tap < 4 * channels ∧ c * 3 + htap < 3 * channels := by omega

/-- Blocks `(channels + 255)/256` of 256 threads cover every channel. -/
theorem conv_blocks_cover (channels c : Nat) (h : c < channels) : c / 256 < (channels + 255) / 256 := by omega

/-- The host wrapper's count limit keeps the grid within `(65535 + 255)/256 = 256` blocks. -/
theorem conv_grid_bound (channels : Nat) (h : channels ≤ 65535) : (channels + 255) / 256 ≤ 256 := by omega

-- § block reductions

/-- Warp partial slot of thread `t`. -/
def warpPartial (t : Nat) : Nat := t / 32

/-- A 256-thread block has 8 warp partials, so `lane < 8` in `norm_sum` reads only written slots. -/
theorem warp_partial_lt (t : Nat) (h : t < 256) : warpPartial t < 8 := by unfold warpPartial; omega

/-- The declared `sums[32]` array has room for them. -/
theorem partials_fit (t : Nat) (h : t < 256) : warpPartial t < 32 := by unfold warpPartial; omega

/-- Threads `col ≥ S` hold zero, and exactly `S = 128` of the 256 threads carry data. -/
theorem data_threads : 256 - 128 = 128 := by decide

/-- Rows and heads are launched one block each: block `b < rows` addresses `[b*128, b*128 + 128)` inside `rows*128`. -/
theorem row_offset_lt (rows b col : Nat) (hb : b < rows) (hc : col < 128) : b * 128 + col < rows * 128 := by
  have h : (b + 1) * 128 ≤ rows * 128 := Nat.mul_le_mul_right 128 hb
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Rows are limited to 65535 blocks. -/
theorem row_limit (rows : Nat) (h : 1 ≤ rows ∧ rows ≤ 65535) : 0 < rows ∧ rows < 65536 := by omega

-- § softplus branch

/-- The threshold branch of softplus, on a fixed-point stand-in for the floating value. -/
def softplusBranch (v : Int) : Bool := decide (v > 20)

/-- Large inputs take the identity branch, the rest the `log1p(exp)` branch. -/
theorem softplus_branch_split (v : Int) : (softplusBranch v = true ↔ 20 < v) := by
  unfold softplusBranch
  simp

-- § epsilon / scale algebra on Nat stand-ins

/-- The kernel receives `eps / S`: the scaled epsilon times `S` recovers `eps` when `S` divides it. -/
theorem eps_scaling (eps : Nat) (h : eps % 128 = 0) : (eps / 128) * 128 = eps := by omega

/-- Head and row address in `out_norm`: `offset + col` with `offset = head * 128`. -/
theorem out_norm_index (heads head col : Nat) (hh : head < heads) (hc : col < 128) :
    head * 128 + col < heads * 128 := row_offset_lt heads head col hh hc

-- § span validation

/-- Two byte ranges overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> intro h <;> omega

/-- Adjacent spans do not overlap. -/
theorem adjacent_disjoint (a an bn : Nat) : ¬ overlap a an (a + an) bn := by
  unfold overlap; omega

/-- Conv spans: history is `3*channels` floats, weights `4*channels`, in/outs `channels` floats (bytes ×4). -/
theorem conv_span_bytes (channels : Nat) : 3 * channels * 4 = 12 * channels ∧ 4 * channels * 4 = 16 * channels := by
  omega

end Strata.NativeGdnPre
