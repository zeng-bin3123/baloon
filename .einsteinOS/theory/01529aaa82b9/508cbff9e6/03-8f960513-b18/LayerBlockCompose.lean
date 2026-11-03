/-!
# MODULE: layer_block_compose
Arithmetic and control-flow contracts of the composed GDN/QSA/MoE layer: 16-byte arena carving, Q8_K sizing,
quantized-plane consistency, the stage-prefix/half selection of `block_layer_pre`, verify-compatibility, and the
per-layer dump stride.

## Source semantics
Every arena region is rounded up to a multiple of 16 bytes. A Q8_K buffer of `n` elements (n a multiple of 256) is
`n/256` blocks of 292 bytes. A quantized tensor's codes/scales/offset planes must sum exactly to the tensor size,
and an offset plane exists iff `has_offset`. `stage_prefix == 0` means "as `half` says"; `stage_prefix == k` runs
stages 0..k-1 and ignores `half`. The MoE `k` must be in 1..64.

## Form template identifiers
- align16 ← `(v + 15) & ~15`, q8kBytes ← `q8k_bytes`, planesOk ← `plane_ptrs` guards,
- runMix / runFfn ← `run0..run4`, verifyCompatible ← `layer_verify_compatible`, dumpStride ← `dump_stride_floats`.
-/

namespace Strata.Layer

-- § arena alignment

/-- Round up to a multiple of 16, as every `gdn_buffers_*`/`moe_buffers_*` part does. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks a region. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- Every aligned region size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- Alignment is idempotent, so re-carving a carved size is stable. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by
  unfold align16; omega

/-- Alignment is monotone, so a larger part never gets a smaller slot. -/
theorem align16_mono {a b : Nat} (h : a ≤ b) : align16 a ≤ align16 b := by
  unfold align16; omega

-- § Q8_K sizing

/-- Bytes of a Q8_K buffer for `n` elements: 292 bytes per 256-element block. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- A whole number of blocks costs exactly 292 bytes each. -/
theorem q8kBytes_blocks (b : Nat) : q8kBytes (256 * b) = 292 * b := by
  unfold q8kBytes; omega

/-- Q8_K size is monotone in the element count. -/
theorem q8kBytes_mono {a b : Nat} (h : a ≤ b) : q8kBytes a ≤ q8kBytes b := by
  unfold q8kBytes; omega

/-- The model's `n_embd = 2560` is a whole number of blocks: 10 blocks, 2920 bytes. -/
theorem q8kBytes_embd : q8kBytes 2560 = 2920 := by
  unfold q8kBytes; omega

-- § quantized planes

/-- The consistency conditions `plane_ptrs` enforces before reading a tensor's planes. -/
def planesOk (codes scales offset bytes : Nat) (hasOffset : Bool) : Prop :=
  codes ≠ 0 ∧ scales ≠ 0 ∧ codes + scales + offset = bytes ∧ (hasOffset = true ↔ offset ≠ 0)

/-- Accepted planes lie inside the tensor: the scale plane ends no later than the tensor does. -/
theorem planesOk_scales_inside {c s o b : Nat} {h : Bool} (p : planesOk c s o b h) : c + s ≤ b := by
  obtain ⟨_, _, hsum, _⟩ := p
  omega

/-- Accepted planes with no offset flag leave no room for an offset plane. -/
theorem planesOk_no_offset {c s o b : Nat} (p : planesOk c s o b false) : b = c + s := by
  obtain ⟨_, _, hsum, hiff⟩ := p
  have ho : o = 0 := by
    by_contra hne
    have := hiff.mpr hne
    exact absurd this (by decide)
  omega

/-- Accepted planes with the offset flag have a nonempty offset plane. -/
theorem planesOk_offset_pos {c s o b : Nat} (p : planesOk c s o b true) : 0 < o := by
  obtain ⟨_, _, _, hiff⟩ := p
  have := hiff.mp rfl
  omega

-- § stage selection of block_layer_pre

/-- Mixer stages (k = 1,2,3 ↔ run0,run1,run2): prefix `sp ≥ k`, or no prefix and half 0/1. -/
def runMix (k sp half : Nat) : Prop := k ≤ sp ∨ (sp = 0 ∧ (half = 0 ∨ half = 1))

/-- FFN stages (k = 4,5 ↔ run3,run4): prefix `sp ≥ k`, or no prefix and half 0/2. -/
def runFfn (k sp half : Nat) : Prop := k ≤ sp ∨ (sp = 0 ∧ (half = 0 ∨ half = 2))

/-- With the defaults (`stage_prefix = 0`, `half = 0`) every stage runs, so older callers are unaffected. -/
theorem default_runs_all :
    runMix 1 0 0 ∧ runMix 2 0 0 ∧ runMix 3 0 0 ∧ runFfn 4 0 0 ∧ runFfn 5 0 0 := by
  unfold runMix runFfn; omega

/-- `half = 2` skips the whole mixer and runs only the FFN front and the router. -/
theorem half2_only_ffn :
    ¬ runMix 1 0 2 ∧ ¬ runMix 2 0 2 ∧ ¬ runMix 3 0 2 ∧ runFfn 4 0 2 ∧ runFfn 5 0 2 := by
  unfold runMix runFfn; omega

/-- `half = 1` runs the mixer and stops before the FFN front. -/
theorem half1_only_mixer :
    runMix 1 0 1 ∧ runMix 2 0 1 ∧ runMix 3 0 1 ∧ ¬ runFfn 4 0 1 ∧ ¬ runFfn 5 0 1 := by
  unfold runMix runFfn; omega

/-- A nonzero prefix ignores `half`: the result depends only on `sp`. -/
theorem prefix_ignores_half {k sp h1 h2 : Nat} (hsp : 0 < sp) :
    runMix k sp h1 ↔ runMix k sp h2 := by
  unfold runMix; omega

/-- Prefix 5 runs every stage whatever `half` says. -/
theorem prefix5_runs_all (half : Nat) :
    runMix 1 5 half ∧ runMix 2 5 half ∧ runMix 3 5 half ∧ runFfn 4 5 half ∧ runFfn 5 5 half := by
  unfold runMix runFfn; omega

/-- Under a nonzero prefix the mixer stages are nested: stage 3 implies stage 2 implies stage 1. -/
theorem prefix_mixer_nested {sp half : Nat} (hsp : 0 < sp) :
    (runMix 3 sp half → runMix 2 sp half) ∧ (runMix 2 sp half → runMix 1 sp half) := by
  unfold runMix; omega

/-- Under a nonzero prefix the FFN stages follow the mixer: stage 4 implies stage 3, stage 5 implies 4. -/
theorem prefix_ffn_after_mixer {sp half : Nat} (hsp : 0 < sp) :
    (runFfn 4 sp half → runMix 3 sp half) ∧ (runFfn 5 sp half → runFfn 4 sp half) := by
  unfold runMix runFfn; omega

-- § verify compatibility and k range

/-- `layer_verify_compatible`: native BF16, fused GR, fused GDN (and native GDN), split-K attention
(fast attention on and the short flash path off), fast select, native indexer. -/
def verifyCompatible (nativeBf16 fusedGr fusedGdn nativeGdn fastAttn flashShort fastSelect indexer : Bool) : Bool :=
  nativeBf16 && fusedGr && fusedGdn && nativeGdn && fastAttn && !flashShort && fastSelect && indexer

/-- Compatibility implies native BF16 projections are on. -/
theorem verifyCompatible_native :
    ∀ a b c d e f g h : Bool, verifyCompatible a b c d e f g h = true → a = true := by
  decide

/-- Compatibility implies the short flash-attention path is off. -/
theorem verifyCompatible_no_flash :
    ∀ a b c d e f g h : Bool, verifyCompatible a b c d e f g h = true → f = false := by
  decide

/-- Compatibility implies both the fused and the native GDN kernels are on. -/
theorem verifyCompatible_gdn :
    ∀ a b c d e f g h : Bool, verifyCompatible a b c d e f g h = true → c = true ∧ d = true := by
  decide

/-- The router/combine guard on the routed-expert count. -/
def kValid (k : Nat) : Prop := 1 ≤ k ∧ k ≤ 64

/-- The model's k = 10 is accepted. -/
theorem kValid_ten : kValid 10 := by
  unfold kValid; omega

/-- Zero experts is rejected. -/
theorem kValid_zero : ¬ kValid 0 := by
  unfold kValid; omega

-- § per-layer dump stride

/-- Floats per layer in the C1 dump: block_out×2, inject×2, attention, 5·(n_kv·head_dim), 8 words of ids/step. -/
def dumpStride (n hc attn nvk : Nat) : Nat := 2 * n + 2 * hc + attn + 5 * nvk + 8

/-- The last written slot (post-rope `kcur`, at base + 2·nvk, width nvk) ends exactly at the stride. -/
theorem dump_last_slot_fits (n hc attn nvk : Nat) :
    (2 * n + 2 * hc + attn + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc attn nvk := by
  unfold dumpStride; omega

/-- The two fp16 pool slots (each `nvk/2`) sit before the raw `vcur` slot without overlap. -/
theorem dump_pool_slots_disjoint (n hc attn nvk : Nat) :
    (2 * n + 2 * hc + attn + 2 * nvk + 8) + nvk / 2 + nvk / 2 ≤
      (2 * n + 2 * hc + attn + 2 * nvk + 8) + nvk := by
  omega

end Strata.Layer
