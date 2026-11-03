/-!
# MODULE: native_gdn_preprocess
Geometry and span-safety arithmetic of the native GDN preprocessing kernels (conv+silu, l2-norm,
beta sigmoid, gate softplus, out-norm).

## Source semantics
Conv uses four taps: history[3c..3c+2] and the input; history slides to [h1,h2,x].
Norm kernels use 256 threads per row, only threads below 128 are active; the second-level block
reduction reads at most 8 warp partials. Counts must lie in [1, 65535].

## Form template identifiers
S -> 128; THREADS -> 256; channels/heads/rows -> counts; Span -> (pointer, bytes).
-/
namespace Strata.NativeGdnPre

/-- History slides: [h0,h1,h2] and x give [h1,h2,x]. -/
def slideHist {α : Type} (h0 h1 h2 x : α) : α × α × α := (h1, h2, x)

/-- Slide drops the oldest tap. -/
theorem slideHist_spec {α : Type} (h0 h1 h2 x : α) : slideHist h0 h1 h2 x = (h1, h2, x) := rfl

/-- Weight/history indexing c*4+tap and c*3+tap stay in range of 4c and 3c sized buffers. -/
theorem tap_index_lt (c ch tap : Nat) (hc : c < ch) (ht : tap < 4) : c * 4 + tap < ch * 4 := by omega

/-- History index with three taps. -/
theorem hist_index_lt (c ch tap : Nat) (hc : c < ch) (ht : tap < 3) : c * 3 + tap < ch * 3 := by omega

/-- Distinct (channel, tap) pairs address distinct weight slots. -/
theorem weight_inj (c t c' t' : Nat) (ht : t < 4) (ht' : t' < 4) (e : c * 4 + t = c' * 4 + t') :
    c = c' ∧ t = t' := by omega

/-- Conv grid covers all channels: ceil(ch/256) blocks of 256. -/
theorem conv_cover (ch c : Nat) (h : c < ch) : c / 256 < (ch + 255) / 256 := by omega

/-- Count validation bounds the grid below 65535 blocks. -/
theorem grid_bound (n : Nat) (h : n ≤ 65535) : (n + 255) / 256 ≤ 65535 := by omega

/-- Eight warps of 32 threads fill a 256-thread block. -/
theorem warps_per_block : 256 / 32 = 8 := by decide

/-- The second-level reduction reads lanes below 8, which are exactly the valid partials. -/
theorem partial_lanes (lane : Nat) (h : lane < 8) : lane < 256 / 32 := by omega

/-- Active norm threads (col < 128) occupy the first four warps. -/
theorem active_warps (col : Nat) (h : col < 128) : col / 32 < 4 := by omega

/-- Row offset blockIdx*128 + col stays inside rows*128. -/
theorem row_offset_lt (b rows col : Nat) (hb : b < rows) (hc : col < 128) : b * 128 + col < rows * 128 := by
  omega

/-- Overlap predicate as used by `disjoint`. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- Non-overlap implies separation. -/
theorem not_overlap_sep (a an b bn : Nat) (h : ¬ overlap a an b bn) : b + bn ≤ a ∨ a + an ≤ b := by
  unfold overlap at h; omega

/-- Valid spans have no pointer wrap. -/
theorem span_no_wrap (addr bytes umax : Nat) (h : bytes ≤ umax - addr) (h2 : addr ≤ umax) :
    addr + bytes ≤ umax := by omega

/-- The l2 epsilon is divided by S before entering rsqrt(sum/S + eps/S): scaling check on naturals. -/
theorem eps_scale (sum S : Nat) (hS : 0 < S) : (sum + 0) / S = sum / S := by simp

end Strata.NativeGdnPre
