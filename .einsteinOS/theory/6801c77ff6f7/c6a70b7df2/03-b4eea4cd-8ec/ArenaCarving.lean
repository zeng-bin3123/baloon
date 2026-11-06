/-!
# MODULE: arena_carving
SOURCES: include/strata/core/layer.hpp, src/core/layer.cpp, include/strata/core/session.hpp, src/core/session.cpp
Every per-layer buffer set (GDN, MoE, QSA, block) is carved ONCE from a device arena, each region rounded up to
16 bytes, and the `*_bytes` function must equal the bytes the matching `*_init` consumes.

## Source semantics
* `gdn_buffers_bytes` sums `(v + 15) & ~15` over the region list; `gdn_buffers_init` walks the same list with a
  moving pointer and returns the same total.
* Region offsets are 16-byte aligned whenever the base is, and consecutive regions never overlap.
* The session gives every GDN layer its own state slice (`gdn_state_floats`), and the 12 QSA layers share one
  RoPE table (the first state owns it, the others borrow it).

## Form template identifiers
align16 <- `(n + 15) & ~15`, `align_up16`; parts <- the `parts[]` array of `*_buffers_bytes/_init`;
offsets <- the `ptr[i]` cursor; total <- `*_buffers_bytes`; cursorEnd <- the value `*_init` returns;
Geometry <- `ModelGeometry` (n_embd, ssm_conv_channels, ssm_value_dim, ssm_v_heads, ssm_state_size, ssm_d_conv).
-/

namespace Strata.ArenaCarving

-- § alignment

/-- 16-byte round-up used between every region of every arena. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never loses bytes. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- Rounded sizes are multiples of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Rounding wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- Rounding is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by
  unfold align16; omega

-- § carving a list of regions

/-- `gdn_buffers_bytes`-style total: sum of the aligned region sizes. -/
def total (parts : List Nat) : Nat := (parts.map align16).sum

/-- The cursor offsets `gdn_buffers_init` assigns to the regions, starting at `base`. -/
def offsets : List Nat → Nat → List Nat
  | [], _ => []
  | p :: ps, base => base :: offsets ps (base + align16 p)

/-- The value `*_init` returns after walking every region. -/
def cursorEnd : List Nat → Nat → Nat
  | [], b => b
  | p :: ps, b => cursorEnd ps (b + align16 p)

/-- Peeling one region off the total. -/
theorem total_cons (p : Nat) (ps : List Nat) : total (p :: ps) = align16 p + total ps := by
  simp [total]

/-- The empty arena needs zero bytes. -/
theorem total_nil : total [] = 0 := by
  simp [total]

/-- One offset is produced per region. -/
theorem offsets_length : ∀ (ps : List Nat) (b : Nat), (offsets ps b).length = ps.length
  | [], _ => by simp [offsets]
  | p :: ps, b => by simp [offsets, offsets_length ps (b + align16 p)]

/-- Consecutive regions are separated by exactly the aligned size of the first one. -/
theorem offsets_step (p q : Nat) (ps : List Nat) (b : Nat) :
    offsets (p :: q :: ps) b = b :: (b + align16 p) :: offsets ps (b + align16 p + align16 q) := rfl

/-- A region never spills into its successor: the aligned size covers the requested size. -/
theorem region_no_overlap (p b : Nat) : b + p ≤ b + align16 p := by
  have := align16_ge p
  omega

/-- If the base is 16-byte aligned, every region handed out is 16-byte aligned. -/
theorem offsets_aligned : ∀ (ps : List Nat) (b : Nat), b % 16 = 0 → ∀ o ∈ offsets ps b, o % 16 = 0
  | [], b, _, o, ho => by simp [offsets] at ho
  | p :: ps, b, hb, o, ho => by
    simp only [offsets, List.mem_cons] at ho
    rcases ho with h | h
    · subst h; exact hb
    · exact offsets_aligned ps (b + align16 p) (by have := align16_mod p; omega) o h

/-- THE CONTRACT BETWEEN `*_bytes` AND `*_init`: the walk ends exactly `total` bytes after the base. -/
theorem cursorEnd_eq : ∀ (ps : List Nat) (b : Nat), cursorEnd ps b = b + total ps
  | [], b => by simp [cursorEnd, total_nil]
  | p :: ps, b => by
    rw [cursorEnd, cursorEnd_eq ps (b + align16 p), total_cons]
    omega

/-- `*_init` returns what `*_bytes` promised. -/
theorem init_returns_bytes (parts : List Nat) : cursorEnd parts 0 = total parts := by
  rw [cursorEnd_eq]; omega

/-- The aligned total is at least the raw sum, so a sized arena always holds the regions. -/
theorem total_ge_sum : ∀ ps : List Nat, ps.sum ≤ total ps
  | [] => by simp [total_nil]
  | p :: ps => by
    rw [total_cons, List.sum_cons]
    have := align16_ge p
    have := total_ge_sum ps
    omega

-- § the GDN region list

/-- The geometry fields the buffer sizes depend on. -/
structure Geometry where
  nEmbd : Nat
  convChannels : Nat
  valueDim : Nat
  vHeads : Nat
  stateSize : Nat
  dConv : Nat

/-- `q8k_bytes`: n/256 blocks of 292 bytes. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- The sixteen regions of `GdnBuffers`, in carve order. -/
def gdnParts (g : Geometry) : List Nat :=
  [q8kBytes g.nEmbd, g.nEmbd / 32 * 34, g.nEmbd * 2,
   g.convChannels * 4, g.convChannels * 4, g.convChannels * 4,
   g.vHeads * 4, g.vHeads * 4, g.vHeads * 4,
   g.vHeads * g.stateSize * 4, g.valueDim * 4, g.valueDim * 4,
   q8kBytes g.valueDim, g.valueDim / 32 * 34,
   g.stateSize * g.vHeads * g.stateSize * 4, g.convChannels * (g.dConv - 1) * 4]

/-- The real Qwen3.8-Flash-Next geometry used by the engine. -/
def realGeometry : Geometry := ⟨2560, 10240, 6144, 48, 128, 4⟩

/-- `GdnBuffers` has sixteen regions. -/
theorem gdnParts_length (g : Geometry) : (gdnParts g).length = 16 := rfl

/-- `gdn_buffers_init` consumes exactly `gdn_buffers_bytes`. -/
theorem gdn_init_eq_bytes (g : Geometry) : cursorEnd (gdnParts g) 0 = total (gdnParts g) :=
  init_returns_bytes _

/-- Recurrent state: S * h_v * S floats per GDN layer, in bytes. -/
theorem real_state_bytes : realGeometry.stateSize * realGeometry.vHeads * realGeometry.stateSize * 4 = 3145728 := by
  decide

/-- Conv history: conv_channels * (d_conv - 1) floats per GDN layer, in bytes. -/
theorem real_conv_bytes : realGeometry.convChannels * (realGeometry.dConv - 1) * 4 = 122880 := by
  decide

/-- The aligned size of one GDN layer's scratch plus state at the real geometry. -/
theorem real_gdn_bytes : total (gdnParts realGeometry) = 3490096 := by
  decide

-- § the MoE region list

/-- The nine regions of `MoEBuffers`; `sharedScratch` stands for `shared_expert_scratch_bytes(n_ff)`. -/
def moeParts (nEmbd nExpert k sharedScratch : Nat) : List Nat :=
  [nEmbd * 2, nEmbd * 2, nExpert * 4, k * 4, k * 4, nEmbd * 4, sharedScratch,
   nEmbd / 32 * 34, q8kBytes nEmbd]

/-- `MoEBuffers` has nine regions. -/
theorem moeParts_length (a b c d : Nat) : (moeParts a b c d).length = 9 := rfl

/-- `moe_buffers_init` consumes exactly `moe_buffers_bytes`. -/
theorem moe_init_eq_bytes (a b c d : Nat) :
    cursorEnd (moeParts a b c d) 0 = total (moeParts a b c d) :=
  init_returns_bytes _

-- § the shared RoPE table

/-- Sharing the RoPE table: the first QSA state owns `rope` extra bytes, the rest borrow it, so n = m+1 layers
cost `(m+1)*rest + rope` rather than `(m+1)*(rest+rope)`. -/
theorem rope_sharing_total (m rest rope : Nat) :
    (rest + rope) + m * rest = (m + 1) * rest + rope := by
  have := Nat.add_mul m 1 rest
  omega

/-- Unshared RoPE tables would cost exactly `m * rope` more. -/
theorem rope_sharing_saves (m rest rope : Nat) :
    (m + 1) * (rest + rope) = ((rest + rope) + m * rest) + m * rope := by
  have h1 := Nat.add_mul m 1 (rest + rope)
  have h2 := Nat.mul_add m rest rope
  omega

end Strata.ArenaCarving
