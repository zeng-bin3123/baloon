/-!
# MODULE: native_gdn_preprocess
SOURCES: src/kernels/cuda/native_gdn_preprocess.cu
GDN pre-processing kernels adapted from pinned llama.cpp: causal conv + SiLU with history slide, the 128-wide L2/RMS
norms, beta sigmoid, softplus gate, and the fused output RMS-norm * sigmoid(z). Geometry and validation formalised.

## Source semantics
* conv_silu: four taps over [h0,h1,h2,x]; the history becomes [h1,h2,x]; raw and SiLU outputs are written.
* norms: width must be exactly 128 with 256 threads = 8 warps; `norm_sum` reduces 8 warp sums.
* host wrappers: count in [1,65535], stream non-null, all spans aligned/non-null, writable spans disjoint.

## Form template identifiers
Hist -> conv_history, histStep -> history_slide, ceilBlocks -> launch_blocks
-/
namespace Strata.NativeGdnPreprocess

-- § conv history
/-- Three-entry history. -/
structure Hist where
  a : Int
  b : Int
  c : Int

/-- Slide by one input. -/
def histStep (h : Hist) (x : Int) : Hist := ⟨h.b, h.c, x⟩

/-- Four taps over history and the new input. -/
def tapSum (h : Hist) (x w0 w1 w2 w3 : Int) : Int := h.a * w0 + h.b * w1 + h.c * w2 + x * w3

/-- Sliding drops the oldest and appends the newest. -/
theorem histStep_fields (h : Hist) (x : Int) :
    (histStep h x).a = h.b ∧ (histStep h x).b = h.c ∧ (histStep h x).c = x := ⟨rfl, rfl, rfl⟩

/-- The history after three inputs forgets the old history. -/
theorem hist_three (h : Hist) (x y z : Int) :
    histStep (histStep (histStep h x) y) z = ⟨x, y, z⟩ := rfl

/-- Tap sum with the new input in the last tap. -/
theorem tapSum_last (h : Hist) (x w0 w1 w2 w3 : Int) :
    tapSum h x w0 w1 w2 w3 = h.a * w0 + h.b * w1 + h.c * w2 + x * w3 := rfl

-- § buffer geometry
/-- History slot c*3+tap stays in 3*channels floats. -/
theorem hist_idx_lt (c tap ch : Nat) (hc : c < ch) (ht : tap < 3) : c * 3 + tap < 3 * ch := by omega

/-- Weight slot c*4+tap stays in 4*channels floats. -/
theorem weight_idx_lt (c tap ch : Nat) (hc : c < ch) (ht : tap < 4) : c * 4 + tap < 4 * ch := by omega

/-- History slots are injective in (channel, tap). -/
theorem hist_idx_inj (c t c' t' : Nat) (ht : t < 3) (ht' : t' < 3) (h : c * 3 + t = c' * 3 + t') :
    c = c' ∧ t = t' := by omega

/-- Ceiling-divide block count. -/
def ceilBlocks (n d : Nat) : Nat := (n + d - 1) / d

/-- 256-thread blocks cover all channels/heads. -/
theorem blocks256_cover (n : Nat) : n ≤ ceilBlocks n 256 * 256 := by
  unfold ceilBlocks
  omega

/-- With count <= 65535 the grid has at most 256 blocks (within the 65535 limit). -/
theorem blocks256_le (n : Nat) (h : n ≤ 65535) : ceilBlocks n 256 ≤ 256 := by
  unfold ceilBlocks
  omega

/-- A positive count needs at least one block. -/
theorem blocks256_pos (n : Nat) (h : 0 < n) : 0 < ceilBlocks n 256 := by
  unfold ceilBlocks
  omega

-- § norm geometry
/-- 256 threads are exactly 8 warps, matching `lane < 8` in norm_sum. -/
theorem norm_warps : 256 / 32 = 8 := by decide

/-- Every warp index of a 256-thread block is below 8. -/
theorem warp_idx_lt (t : Nat) (h : t < 256) : t / 32 < 8 := by omega

/-- Row offset blockIdx*128 + col is in range of heads*128. -/
theorem norm_idx_lt (b col heads : Nat) (hb : b < heads) (hc : col < 128) : b * 128 + col < heads * 128 := by
  have h1 : b + 1 ≤ heads := hb
  have h2 : (b + 1) * 128 ≤ heads * 128 := Nat.mul_le_mul_right 128 h1
  omega

/-- Threads 128..255 of a norm block are padding (col >= S) and contribute zero. -/
theorem norm_padding (col : Nat) (h : 128 ≤ col) : ¬ col < 128 := by omega

-- § span validation
/-- Byte-span overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  constructor <;> intro h <;> omega

/-- Adjacent spans do not overlap. -/
theorem adjacent_disjoint (a n m : Nat) : ¬ overlap a n (a + n) m := by
  unfold overlap
  omega

/-- The conv writable spans history (3*bytes), raw (bytes), silu (bytes) packed back to back are pairwise disjoint. -/
theorem packed_spans_disjoint (base bytes : Nat) :
    ¬ overlap base (3 * bytes) (base + 3 * bytes) bytes ∧
    ¬ overlap base (3 * bytes) (base + 3 * bytes + bytes) bytes ∧
    ¬ overlap (base + 3 * bytes) bytes (base + 3 * bytes + bytes) bytes := by
  unfold overlap
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Softplus guard: x > 20 passes through unchanged. -/
theorem softplus_large (x : Int) (f : Int → Int) (h : 20 < x) : (if 20 < x then x else f x) = x := by
  simp [h]

end Strata.NativeGdnPreprocess
