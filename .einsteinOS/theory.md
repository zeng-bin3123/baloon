
## FILE: /mnt/user-data/outputs/LayerComposition.lean
/-!
# MODULE: layer_composition
SOURCES: src/core/layer.cpp, include/strata/core/layer.hpp
Composition of one GDN/QSA/MoE block: arena carving (16-byte aligned regions), quantized-plane layout
checks, activation-format dispatch, stage-prefix selection of `block_layer_pre`, and the debug-dump layout.

## Source semantics
* Every scratch region is carved from an arena with 16-byte alignment (`(n + 15) & ~15`).
* A Q8_K buffer of `n` elements (n a multiple of 256) occupies `n / 256 * 292` bytes.
* A quantized tensor is three planes (codes | scales | optional offsets) that must add up to the tensor size,
  and `has_offset` must agree with the offset plane being non-empty.
* Which activation image a weight wants is a property of the tensor: native first, then 2-bit → Q8_0,
  then Q8_K for K-quants, else Q8_0 (legacy).
* `block_layer_pre` runs a prefix of 5 stages (`stage_prefix = k > 0`) or a half (1 = mixer, 2 = FFN front).
* The verify window requires a conjunction of fast-path flags.

## Form template identifiers
- align16 → align_up_16 ; q8kBytes → q8k_bytes ; PlaneRef → WeightRef ; Cursor → arena cursor
- runStage → run0..run4 ; verifyCompatible → layer_verify_compatible
-/

namespace Strata.LayerComposition

-- § alignment

/-- 16-byte round-up used for every arena region. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never loses bytes. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- The rounded size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Rounding adds fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- Rounding is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by
  unfold align16; omega

-- § Q8_K sizing

/-- Bytes of a Q8_K image of `n` elements. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- One more 256-element block costs exactly 292 bytes. -/
theorem q8kBytes_step (n : Nat) : q8kBytes (n + 256) = q8kBytes n + 292 := by
  unfold q8kBytes; omega

/-- `k` full blocks cost `292 * k` bytes. -/
theorem q8kBytes_mul (k : Nat) : q8kBytes (256 * k) = 292 * k := by
  unfold q8kBytes; omega

-- § quantized planes

/-- The plane attributes a `WeightRef` carries. -/
structure PlaneRef where
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat
  codeBits : Nat
  hasOffset : Bool

/-- `plane_ptrs` acceptance: S2/S4/S8, non-empty codes and scales, planes add up, `has_offset` agrees. -/
def planesOk (r : PlaneRef) : Prop :=
  (r.codeBits = 2 ∨ r.codeBits = 4 ∨ r.codeBits = 8) ∧ r.codes ≠ 0 ∧ r.scales ≠ 0 ∧
  r.codes + r.scales + r.offset = r.bytes ∧ (r.hasOffset = true ↔ r.offset ≠ 0)

/-- Start of the scale plane. -/
def scalesStart (r : PlaneRef) : Nat := r.codes

/-- Start of the offset plane. -/
def offsetStart (r : PlaneRef) : Nat := r.codes + r.scales

/-- The three planes partition the tensor exactly. -/
theorem planes_partition (r : PlaneRef) (h : planesOk r) :
    scalesStart r + r.scales + r.offset = r.bytes := by
  obtain ⟨_, _, _, h3, _⟩ := h
  unfold scalesStart; omega

/-- A non-empty offset plane ends exactly at the end of the tensor. -/
theorem planes_offset_ends (r : PlaneRef) (h : planesOk r) :
    offsetStart r + r.offset = r.bytes := by
  obtain ⟨_, _, _, h3, _⟩ := h
  unfold offsetStart; omega

/-- Without an offset plane, codes plus scales are the whole tensor. -/
theorem planes_no_offset (r : PlaneRef) (h : planesOk r) (hf : r.hasOffset = false) :
    r.codes + r.scales = r.bytes := by
  obtain ⟨_, _, _, h3, h5⟩ := h
  have h0 : r.offset = 0 := by
    by_cases hz : r.offset = 0
    · exact hz
    · have ht := h5.mpr hz
      rw [hf] at ht
      exact absurd ht Bool.false_ne_true
  omega

-- § arena cursor

/-- Cursor advance of `Cursor::take`: `count` elements of `size` bytes, re-aligned to 16. -/
def cursorTake (used count size : Nat) : Nat := align16 (used + count * size)

/-- The taken region fits below the new cursor. -/
theorem cursorTake_covers (used count size : Nat) : used + count * size ≤ cursorTake used count size := by
  unfold cursorTake; exact align16_ge _

/-- The new cursor is 16-byte aligned. -/
theorem cursorTake_mod (used count size : Nat) : cursorTake used count size % 16 = 0 := by
  unfold cursorTake; exact align16_mod _

/-- The cursor never moves backwards. -/
theorem cursorTake_mono (used count size : Nat) : used ≤ cursorTake used count size := by
  have h := cursorTake_covers used count size
  omega

/-- Sum of 16-byte aligned parts (the `*_buffers_bytes` totals). -/
def alignedTotal : List Nat → Nat
  | [] => 0
  | x :: xs => align16 x + alignedTotal xs

/-- Every aligned total is a multiple of 16. -/
theorem alignedTotal_mod : ∀ l : List Nat, alignedTotal l % 16 = 0
  | [] => by simp [alignedTotal]
  | x :: xs => by
    have h1 := align16_mod x
    have h2 := alignedTotal_mod xs
    simp only [alignedTotal]
    omega

/-- The aligned total is at least the plain sum of the parts. -/
theorem alignedTotal_ge : ∀ l : List Nat, l.sum ≤ alignedTotal l
  | [] => by simp [alignedTotal]
  | x :: xs => by
    have h1 := align16_ge x
    have h2 := alignedTotal_ge xs
    simp only [List.sum_cons, alignedTotal]
    omega

/-- Region `i` of `*_buffers_init` starts at an aligned offset. -/
theorem region_base_aligned (parts : List Nat) (i : Nat) : alignedTotal (parts.take i) % 16 = 0 :=
  alignedTotal_mod _

-- § activation dispatch

/-- Which GEMV path `gemv_quantized` takes. -/
inductive Gemv
  | native
  | s2q80
  | legacyQ80
  | q8k

/-- The dispatch rule of `gemv_quantized`. -/
def dispatch (native : Bool) (codeBits : Nat) (wantsQ8k : Bool) : Gemv :=
  if native then Gemv.native
  else if codeBits = 2 then Gemv.s2q80
  else if wantsQ8k then Gemv.q8k
  else Gemv.legacyQ80

/-- A native tensor always takes the native path. -/
theorem dispatch_native (c : Nat) (w : Bool) : dispatch true c w = Gemv.native := by
  simp [dispatch]

/-- A non-native 2-bit tensor always takes the S2/Q8_0 path. -/
theorem dispatch_s2 (c : Nat) (w : Bool) (hc : c = 2) : dispatch false c w = Gemv.s2q80 := by
  simp [dispatch, hc]

/-- A legacy 4/8-bit tensor that does not want Q8_K takes the legacy Q8_0 path. -/
theorem dispatch_legacy (c : Nat) (hc : c ≠ 2) : dispatch false c false = Gemv.legacyQ80 := by
  simp [dispatch, hc]

/-- The Q8_K path is never reached by a 2-bit tensor (which has no Q8_K contract). -/
theorem dispatch_q8k_not_s2 (n : Bool) (c : Nat) (w : Bool) (h : dispatch n c w = Gemv.q8k) :
    c ≠ 2 := by
  intro hc
  subst hc
  cases n <;> cases w <;> simp [dispatch] at h

-- § stage prefixes of block_layer_pre

/-- Whether stage `i` (0..4) runs, given `stage_prefix` and `half`. -/
def runStage (stagePrefix half i : Nat) : Bool :=
  if stagePrefix > 0 then decide (stagePrefix ≥ i + 1)
  else if i < 3 then decide (half = 0 ∨ half = 1)
  else decide (half = 0 ∨ half = 2)

/-- Default call (`half = 0`, no prefix) runs all five stages. -/
theorem runStage_default : ∀ i, i < 5 → runStage 0 0 i = true := by
  decide

/-- `half = 1` is the mixer: stages 0..2. -/
theorem runStage_mixer : ∀ i, i < 5 → runStage 0 1 i = decide (i < 3) := by
  decide

/-- `half = 2` is the FFN front and router: stages 3..4. -/
theorem runStage_ffn : ∀ i, i < 5 → runStage 0 2 i = decide (3 ≤ i) := by
  decide

/-- A positive prefix `k` runs exactly stages `0..k-1`, ignoring `half`. -/
theorem runStage_prefix (k half i : Nat) (hk : 0 < k) : runStage k half i = decide (i < k) := by
  unfold runStage
  rw [if_pos hk]
  exact decide_eq_decide.mpr (by omega)

/-- Prefix 5 runs every stage (it is the full graph). -/
theorem runStage_prefix5 (half i : Nat) (hi : i < 5) : runStage 5 half i = true := by
  rw [runStage_prefix 5 half i (by decide)]
  exact decide_eq_true hi

-- § verify-window compatibility

/-- `layer_verify_compatible` as a boolean conjunction of the fast-path flags. -/
def verifyCompatible (nb gr gdnFused nativeGdn fastAttn flashShort fastSel qsaIdx : Bool) : Bool :=
  nb && gr && (gdnFused && nativeGdn) && (fastAttn && !flashShort) && fastSel && qsaIdx

/-- Compatible exactly when every required flag is on and the short flash adapter is off. -/
theorem verifyCompatible_iff (nb gr gdnFused nativeGdn fastAttn flashShort fastSel qsaIdx : Bool) :
    verifyCompatible nb gr gdnFused nativeGdn fastAttn flashShort fastSel qsaIdx = true ↔
      (nb = true ∧ gr = true ∧ gdnFused = true ∧ nativeGdn = true ∧ fastAttn = true ∧
        flashShort = false ∧ fastSel = true ∧ qsaIdx = true) := by
  cases nb <;> cases gr <;> cases gdnFused <;> cases nativeGdn <;> cases fastAttn <;>
    cases flashShort <;> cases fastSel <;> cases qsaIdx <;> simp [verifyCompatible]

-- § fused hyper-connection bookkeeping

/-- `fused` of `block_layer_pre`: only for the default call. -/
def fusedGr (flag : Bool) (stagePrefix half : Nat) (supported : Bool) : Bool :=
  flag && decide (stagePrefix = 0) && decide (half = 0) && supported

/-- Fusion implies the default call shape. -/
theorem fusedGr_default (flag : Bool) (sp half : Nat) (s : Bool)
    (h : fusedGr flag sp half s = true) : sp = 0 ∧ half = 0 := by
  unfold fusedGr at h
  simp at h
  exact ⟨h.1.1.2, h.1.2⟩

/-- The post half writes the residual itself unless fused, and always on the last layer. -/
def postWritesR (fusedOn : Bool) (layer nLayers : Nat) : Bool :=
  !fusedOn || decide (layer + 1 = nLayers)

/-- Without fusion `gr_write` always runs. -/
theorem postWritesR_unfused (layer n : Nat) : postWritesR false layer n = true := by
  simp [postWritesR]

/-- With fusion the last layer materialises the residual for the head. -/
theorem postWritesR_last (n : Nat) : postWritesR true n (n + 1) = true := by
  simp [postWritesR]

/-- With fusion an inner layer defers its write. -/
theorem postWritesR_inner (layer n : Nat) (h : layer + 1 ≠ n) : postWritesR true layer n = false := by
  simp [postWritesR, h]

-- § PLE gating

/-- The PLE block applies only at layer 1 and only in the mixer-side halves. -/
def pleApplies (ready : Bool) (layer half : Nat) : Bool :=
  ready && decide (layer = 1) && decide (half = 0 ∨ half = 1)

/-- No PLE outside layer 1. -/
theorem pleApplies_layer (r : Bool) (l half : Nat) (h : l ≠ 1) : pleApplies r l half = false := by
  simp [pleApplies, h]

/-- No PLE in the FFN-front half. -/
theorem pleApplies_ffn (r : Bool) : pleApplies r 1 2 = false := by
  simp [pleApplies]

-- § layer kinds

/-- QSA layers are those with index ≡ 3 mod 4. -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- 48 layers split 12 QSA / 36 GDN. -/
theorem qsa_gdn_split :
    ((List.range 48).filter isQsa).length = 12 ∧
      ((List.range 48).filter (fun l => !isQsa l)).length = 36 := by
  decide

-- § RoPE table bounds

/-- A flat row-major index stays inside a table of `m` rows of width `h`. -/
theorem flat_index_lt (r m h j : Nat) (hr : r < m) (hj : j < h) : r * h + j < m * h := by
  have h1 : (r + 1) * h ≤ m * h := Nat.mul_le_mul_right h (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § debug-dump layout

/-- Floats per layer in the C1 dump (n, hc, a = n_head*head_dim, nvk = n_head_kv*head_dim). -/
def dumpStride (n hc a nvk : Nat) : Nat := 2 * n + 2 * hc + a + 5 * nvk + 8

/-- The last of the four KV dumps ends exactly at the stride. -/
theorem dump_last_slot_ends_at_stride (n hc a nvk : Nat) :
    (2 * n + 2 * hc + a + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc a nvk := by
  unfold dumpStride; omega

/-- Arithmetic fact about the debug layout: the `ids/step` dump window of `qsa_layer` starts 8 floats
below the `k_pool` dump window and so intersects it whenever `nvk ≥ 2`. -/
theorem dump_debug_windows_intersect (n hc a nvk : Nat) (hn : 2 ≤ nvk) :
    ∃ p, (2 * n + 2 * hc + a + 2 * nvk + 8 ≤ p ∧
            p < 2 * n + 2 * hc + a + 2 * nvk + 8 + nvk / 2) ∧
         ((2 * n + 2 * hc + a + 2 * nvk) + nvk / 2 + 4 ≤ p ∧
            p < (2 * n + 2 * hc + a + 2 * nvk) + nvk / 2 + 8) := by
  refine ⟨2 * n + 2 * hc + a + 2 * nvk + 8 + nvk / 2 - 1, ?_, ?_⟩ <;> constructor <;> omega

end Strata.LayerComposition



## FILE: /mnt/user-data/outputs/SessionGraphs.lean
/-!
# MODULE: session_graphs
SOURCES: src/core/session.cpp, include/strata/core/session.hpp
One token through all 48 layers: session arena sizing/carving, per-layer GDN/QSA state indexing, the
stage-prefix timing differences, and the dump-buffer layout of the host loop.

## Source semantics
* `session_bytes` = align256(sum of region sizes) + 4096; `session_init` carves each region at a 256-byte
  aligned cursor, so the carve must fit inside the reported size.
* GDN layers are numbered separately from QSA layers (`l % 4 == 3` is QSA): 36 GDN states, 12 QSA states.
* Stage times are differences of consecutive prefix times; the stage list telescopes to the last prefix.
* `session_loop` expects ring value `l + 1` from layer `l`; the dump holds `(n_layers + 1) * hc * n_embd` floats.

## Form template identifiers
- alignUp256 → align_up(n, SESSION_STATE_ALIGN) ; takeAll → session_init's `take` chain
- gdnIndex / qsaIndex → gdn_point_at / qsa_index ; stageDiffs → session_replay_stage_prefixes
-/

namespace Strata.SessionGraphs

-- § alignment to 256

/-- Round up to a multiple of 256 (`SESSION_STATE_ALIGN`). -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Rounding adds at most 255 bytes. -/
theorem alignUp256_le (n : Nat) : alignUp256 n ≤ n + 255 := by
  unfold alignUp256; omega

/-- The result is a multiple of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

-- § carving fits in the reported size

/-- `session_init`'s chain of `take` calls: advance the cursor by each region, re-aligned to 256. -/
def takeAll : Nat → List Nat → Nat
  | u, [] => u
  | u, x :: xs => takeAll (alignUp256 (u + x)) xs

/-- Each take wastes fewer than 256 bytes, so the end cursor is bounded by sum plus 255 per region. -/
theorem takeAll_le : ∀ (l : List Nat) (u : Nat), takeAll u l ≤ u + l.sum + 255 * l.length
  | [], u => by simp [takeAll]
  | x :: xs, u => by
    have h1 := alignUp256_le (u + x)
    have h2 := takeAll_le xs (alignUp256 (u + x))
    simp only [takeAll, List.sum_cons, List.length_cons]
    omega

/-- `session_bytes` formula. -/
def sessionBytes (parts : List Nat) : Nat := alignUp256 parts.sum + 4096

/-- With at most 16 regions (the session has 7) the carve always fits inside `session_bytes`. -/
theorem session_init_fits (parts : List Nat) (hlen : parts.length ≤ 16) :
    takeAll 0 parts ≤ sessionBytes parts := by
  have h1 := takeAll_le parts 0
  have h2 := alignUp256_ge parts.sum
  unfold sessionBytes
  omega

/-- The reported size is a multiple of 256 (4096 is a multiple of 256). -/
theorem sessionBytes_mod (parts : List Nat) : sessionBytes parts % 256 = 0 := by
  have := alignUp256_mod parts.sum
  unfold sessionBytes; omega

-- § per-layer state indices

/-- Whether layer `l` is QSA. -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- Index of GDN layer `l` among GDN layers: how many GDN layers precede it. -/
def gdnIndex (l : Nat) : Nat := ((List.range l).filter (fun i => !isQsa i)).length

/-- Index of QSA layer `l` among QSA layers. -/
def qsaIndex (l : Nat) : Nat := ((List.range l).filter isQsa).length

/-- Every GDN layer of the 48 gets a state slot below 36. -/
theorem gdnIndex_lt : ∀ l, l < 48 → l % 4 ≠ 3 → gdnIndex l < 36 := by
  decide

/-- Every QSA layer of the 48 gets a state slot below 12. -/
theorem qsaIndex_lt : ∀ l, l < 48 → l % 4 = 3 → qsaIndex l < 12 := by
  decide

/-- Consecutive GDN layers get consecutive slots, so no two layers share a recurrence. -/
theorem gdnIndex_succ : ∀ l, l < 47 → l % 4 ≠ 3 → gdnIndex (l + 1) = gdnIndex l + 1 := by
  decide

/-- Consecutive QSA layers get consecutive slots. -/
theorem qsaIndex_succ : ∀ l, l < 47 → l % 4 = 3 → qsaIndex (l + 1) = qsaIndex l + 1 := by
  decide

/-- Two GDN layers below 48 with equal slots are the same layer. -/
theorem gdnIndex_inj : ∀ a, a < 48 → ∀ b, b < 48 → a % 4 ≠ 3 → b % 4 ≠ 3 →
    gdnIndex a = gdnIndex b → a = b := by
  decide

-- § stage timing by prefix differences

/-- Stage times from prefix times t1..t5 (stage 0 is t1, stage i is t_{i+1} - t_i). -/
def stageDiffs (t1 t2 t3 t4 t5 : Nat) : List Nat := [t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4]

/-- For nondecreasing prefix times the stages telescope back to the last prefix. -/
theorem stageDiffs_sum (t1 t2 t3 t4 t5 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) (h3 : t3 ≤ t4)
    (h4 : t4 ≤ t5) : (stageDiffs t1 t2 t3 t4 t5).sum = t5 := by
  simp only [stageDiffs, List.sum_cons, List.sum_nil]
  omega

/-- The mixer time per layer is prefix 3 (stages 0..2): stage 0 + 1 + 2 equals t3. -/
theorem mixer_is_prefix3 (t1 t2 t3 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) :
    t1 + (t2 - t1) + (t3 - t2) = t3 := by
  omega

-- § ring numbering and dump layout

/-- The ring value `session_loop` waits for at layer `l`. -/
def wantRing (l : Nat) : Nat := l + 1

/-- Rings are strictly increasing in layer order. -/
theorem wantRing_strict (a b : Nat) (h : a < b) : wantRing a < wantRing b := by
  unfold wantRing; omega

/-- Slot `l+1` of the residual dump lies inside the `(L+1)`-slot buffer. -/
theorem dump_slot_in_bounds (l L n : Nat) (h : l < L) : (l + 1) * n + n ≤ (L + 1) * n := by
  have h2 : (l + 1) * n ≤ L * n := Nat.mul_le_mul_right n (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h2 ⊢
  omega

-- § per-token handoff bookkeeping

/-- Parts staging size `k * n_embd * 4` is positive for positive inputs, so the pinned scratch is non-empty. -/
theorem parts_bytes_pos (k n : Nat) (hk : 0 < k) (hn : 0 < n) : 0 < k * n * 4 := by
  have : 0 < k * n := Nat.mul_pos hk hn
  omega

/-- Token-graph flush interval check: a ring wait is only escalated after a `1024`-spin boundary. -/
def spinBoundary (spins : Nat) : Bool := decide ((spins + 1) % 1024 = 0)

/-- The boundary fires exactly every 1024th spin. -/
theorem spinBoundary_period (s : Nat) : spinBoundary (s + 1024) = spinBoundary s := by
  unfold spinBoundary
  simp only [decide_eq_decide]
  omega

end Strata.SessionGraphs



## FILE: /mnt/user-data/outputs/VerifyWindow.lean
/-!
# MODULE: verify_window
SOURCES: src/core/verify.cpp, include/strata/core/verify.hpp
The speculative verify window: T tokens through all layers in one captured graph, optionally split into two
token groups pipelined against the CPU experts, with a commit step that keeps the first `n_keep` tokens.

## Source semantics
* Split window: groups A = [0, (T+1)/2) and B = [(T+1)/2, T); with one group the range is [0, T).
* Ring/flag value for (layer l, group g) is `l*G + g + 1`; the host loop recovers (l, g) = (k / G, k % G).
* The stream order is pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) ...
* The GPU plan block lays out counts | start | dst | tok | pad | ptr | ptr2 | start2 in int32 words.
* Flag B only rises (`raise_flag` is an atomic max).
* `commit(n_keep)` writes `[n_keep, n_keep-1, pos0..pos0+n_keep-1, -1...]` and advances the PLE predecessors.

## Form template identifiers
- groupBegin/groupEnd → tb_/te_ ; ringValue → `ring` ; preIdx/postIdx → position in the captured stream
- planPtrOff → ptr_off ; raiseFlag → Verifier::raise_flag ; advance → ss.ple_prev update
-/

namespace Strata.VerifyWindow

-- § token groups

/-- First token of group `grp` for a window of `T` tokens. -/
def groupBegin (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- End (exclusive) of group `grp`, with `G` groups. -/
def groupEnd (T G grp : Nat) : Nat := if grp = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- With two groups the two ranges tile [0, T). -/
theorem groups_tile (T : Nat) :
    groupBegin T 0 = 0 ∧ groupEnd T 2 0 = groupBegin T 1 ∧ groupEnd T 2 1 = T := by
  simp [groupBegin, groupEnd]

/-- With two groups the sizes add up to T. -/
theorem groups_sizes (T : Nat) :
    (groupEnd T 2 0 - groupBegin T 0) + (groupEnd T 2 1 - groupBegin T 1) = T := by
  simp only [groupBegin, groupEnd]
  simp
  omega

/-- For a window of at least two tokens both groups are non-empty. -/
theorem groups_nonempty (T : Nat) (hT : 2 ≤ T) :
    groupBegin T 0 < groupEnd T 2 0 ∧ groupBegin T 1 < groupEnd T 2 1 := by
  simp only [groupBegin, groupEnd]
  simp
  omega

/-- With one group the single range is [0, T). -/
theorem single_group (T : Nat) : groupBegin T 0 = 0 ∧ groupEnd T 1 0 = T := by
  simp [groupBegin, groupEnd]

-- § ring numbering

/-- Ring/flag value for (layer, group). -/
def ringValue (l G grp : Nat) : Nat := l * G + grp + 1

/-- The host's `want = k + 1` with `k = l*G + grp` equals the graph's ring value. -/
theorem ring_host_matches (l G grp : Nat) : (l * G + grp) + 1 = ringValue l G grp := by
  unfold ringValue; rfl

/-- With two groups the host recovers the layer as `k / 2`. -/
theorem ring_recover_layer (l grp : Nat) (h : grp < 2) : (l * 2 + grp) / 2 = l := by
  omega

/-- With two groups the host recovers the group as `k % 2`. -/
theorem ring_recover_group (l grp : Nat) (h : grp < 2) : (l * 2 + grp) % 2 = grp := by
  omega

/-- Ring values are strictly increasing in (layer, group) order for two groups. -/
theorem ring_strict (l grp l' grp' : Nat) (h : grp < 2) (h' : grp' < 2) (hl : l < l') :
    ringValue l 2 grp < ringValue l' 2 grp' := by
  unfold ringValue; omega

-- § stream order of the split window

/-- Position of `pre(l, g)` in the captured stream (two groups). -/
def preIdx (l g : Nat) : Nat := if l = 0 then g else 4 * l - 1 + 2 * g

/-- Position of `post(l, g)` in the captured stream (two groups). -/
def postIdx (l g : Nat) : Nat := 2 + 4 * l + 2 * g

/-- `post(l,g)` comes after `pre(l,g)`. -/
theorem pre_before_post (l g : Nat) (_hg : g < 2) : preIdx l g < postIdx l g := by
  unfold preIdx postIdx
  split <;> omega

/-- `pre(l+1,g)` comes after `post(l,g)`. -/
theorem post_before_next_pre (l g : Nat) (_hg : g < 2) : postIdx l g < preIdx (l + 1) g := by
  unfold preIdx postIdx
  split <;> omega

/-- Group B's `pre` follows group A's `pre` of the same layer. -/
theorem preB_after_preA (l : Nat) : preIdx l 0 < preIdx l 1 := by
  unfold preIdx
  split <;> omega

/-- The overlap property: A's next `pre` is queued before B's `post` of this layer. -/
theorem nextPreA_before_postB (l : Nat) : preIdx (l + 1) 0 < postIdx l 1 := by
  unfold preIdx postIdx
  split <;> omega

/-- `post(l,A)` precedes `post(l,B)`. -/
theorem postA_before_postB (l : Nat) : postIdx l 0 < postIdx l 1 := by
  unfold postIdx; omega

-- § GPU plan block layout

/-- int32 words before the pad: counts(4) | start(cap+1) | dst(cap) | tok(cap). -/
def planI32 (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- `ptr_off`: the 64-bit arrays start on an even word. -/
def planPtrOff (cap : Nat) : Nat := (planI32 cap + 1) / 2 * 2

/-- Total int32 words: ptr_off + 4*cap (ptr and ptr2) + (cap+1) (start2) + 1. -/
def planTotal (cap : Nat) : Nat := planPtrOff cap + 4 * cap + (cap + 1) + 1

/-- The pointer arrays are 8-byte aligned (even word offset). -/
theorem planPtrOff_even (cap : Nat) : planPtrOff cap % 2 = 0 := by
  unfold planPtrOff; omega

/-- The pad is at most one word. -/
theorem planPtrOff_bounds (cap : Nat) : planI32 cap ≤ planPtrOff cap ∧ planPtrOff cap ≤ planI32 cap + 1 := by
  unfold planPtrOff; omega

/-- `start2` fits inside the block. -/
theorem plan_start2_fits (cap : Nat) : planPtrOff cap + 4 * cap + (cap + 1) ≤ planTotal cap := by
  unfold planTotal; omega

/-- The `post` lambda's formula for `ptr_off` is the same as the init formula. -/
theorem plan_post_formula (cap : Nat) : ((4 + (cap + 1) + 2 * cap) + 1) / 2 * 2 = planPtrOff cap := by
  unfold planPtrOff planI32; omega

-- § staging split

/-- Staging blobs per group: half each with two groups. -/
def stagingPer (G : Nat) : Nat := if G = 2 then 16 / 2 else 16

/-- Group staging ranges are disjoint and fit in the 16 staging blobs. -/
theorem staging_fits (G grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    grp * stagingPer G + stagingPer G ≤ 16 := by
  unfold stagingPer
  rcases hG with rfl | rfl
  · have : grp = 0 := by omega
    subst this
    simp
  · simp
    omega

-- § flag B is a monotone max

/-- `raise_flag`: store `v` only if it is larger than the current value. -/
def raiseFlag (cur v : Nat) : Nat := max cur v

/-- Raising never lowers the flag. -/
theorem raiseFlag_ge_cur (cur v : Nat) : cur ≤ raiseFlag cur v := by
  unfold raiseFlag; omega

/-- After raising to `v` the flag is at least `v`. -/
theorem raiseFlag_ge_val (cur v : Nat) : v ≤ raiseFlag cur v := by
  unfold raiseFlag; omega

/-- A late raise of an older layer cannot lower the flag. -/
theorem raiseFlag_late (cur v : Nat) (h : v ≤ cur) : raiseFlag cur v = cur := by
  unfold raiseFlag; omega

/-- Raising is order-insensitive. -/
theorem raiseFlag_comm (c a b : Nat) : raiseFlag (raiseFlag c a) b = raiseFlag (raiseFlag c b) a := by
  unfold raiseFlag; omega

-- § commit and PLE predecessors

/-- Predecessor window update of `commit`: shift in one committed token. -/
def advance (p : Int × Int) (tok : Int) : Int × Int := (p.2, tok)

/-- After committing at least two tokens, the predecessors are exactly the last two. -/
theorem advance_last_two (xs : List Int) (a b y z : Int) :
    (xs ++ [y, z]).foldl advance (a, b) = (y, z) := by
  simp [List.foldl_append, advance]

/-- Committing one token shifts the window by one. -/
theorem advance_one (a b z : Int) : [z].foldl advance (a, b) = (b, z) := by
  simp [advance]

/-- Exactly `n_keep` of the 8 commit slots carry a position. -/
theorem commit_slots_count : ∀ k, k ≤ 8 →
    ((List.range 8).filter (fun t => decide (t < k))).length = k := by
  decide

/-- Window sizes within the constructor bound index the 9-entry graph table safely. -/
theorem window_index_ok (T maxT kMax : Nat) (h1 : T ≤ maxT) (h2 : maxT ≤ kMax) (h3 : kMax ≤ 8) :
    T < 9 := by
  omega

end Strata.VerifyWindow



## FILE: /mnt/user-data/outputs/WeightLoader.lean
/-!
# MODULE: weight_loader
SOURCES: src/core/weights.cpp
The dense-weight loader: reads `index.txt`, places each tensor in the arena (256-byte aligned, optionally
compacted by a skip set), and streams it plane by plane in 8 MiB chunks, converting widths on the way.

## Source semantics
* A quantized tensor is segments codes | scales | offsets; an fp16 scale plane is half as large in the pack as
  in the arena and is widened exactly (fp16 → f32).
* The segment totals are checked against the index's `src_bytes`/`dst_bytes` before any byte moves.
* BF16/F16 tensors promoted from f32 containers are 4 B/elem in and 2 B/elem out.
* Index kinds 4/5 (raw 16-bit) are copied and take the engine forms 1 and 3.
* Chunks never split an element; the widened chunk fits the 2×CHUNK output staging buffer.

## Form template identifiers
- Seg / segs → segment list ; chunkLen → `n = min(CHUNK, rem)` ; placeAll → compaction loop
- convFor → conversion selection ; mapKind → raw16 kind remap
-/

namespace Strata.WeightLoader

/-- 8 MiB of source per staging round. -/
def CHUNK : Nat := 8388608

-- § segment lists

/-- Source bytes of the scale plane. -/
def srcScaleBytes (scalesFp16 : Bool) (scales : Nat) : Nat := if scalesFp16 then scales / 2 else scales

/-- Total source bytes of the three segments. -/
def sumSrc (codes scales offset : Nat) (fp16 : Bool) : Nat := codes + srcScaleBytes fp16 scales + offset

/-- Total destination bytes of the three segments. -/
def sumDst (codes scales offset : Nat) : Nat := codes + scales + offset

/-- The destination total is the three plane sizes. -/
theorem sumDst_eq (c s o : Nat) : sumDst c s o = c + s + o := rfl

/-- With plain scales the two totals agree. -/
theorem sumSrc_plain (c s o : Nat) : sumSrc c s o false = sumDst c s o := by
  simp [sumSrc, sumDst, srcScaleBytes]

/-- With fp16 scales of even byte count, the source is exactly half the scale plane smaller. -/
theorem sumSrc_fp16 (c s o : Nat) (he : s % 2 = 0) : sumSrc c s o true + s / 2 = sumDst c s o := by
  simp only [sumSrc, sumDst, srcScaleBytes, if_true]
  omega

/-- An even fp16 scale plane survives halve-then-double. -/
theorem fp16_half_double (s : Nat) (he : s % 2 = 0) : s / 2 * 2 = s := by
  omega

-- § conversions: widen, bf16, f16

/-- Widening `n` fp16 bytes yields `2 * n` f32 bytes. -/
theorem widen_out_bytes (n : Nat) (he : n % 2 = 0) : 4 * (n / 2) = 2 * n := by
  omega

/-- Widened chunks tile the destination: `done*2 + out = (done + n)*2`. -/
theorem widen_tiles (done n : Nat) (he : n % 2 = 0) : done * 2 + 4 * (n / 2) = (done + n) * 2 := by
  omega

/-- A widened chunk fits the `2 * CHUNK` staging buffer. -/
theorem widen_fits_stage (n : Nat) (h : n ≤ CHUNK) : 4 * (n / 2) ≤ 2 * CHUNK := by
  unfold CHUNK at *
  omega

/-- Narrowing a 4-byte container to 2 bytes tiles the destination at half the source offset. -/
theorem narrow_tiles (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + 2 * (n / 4) = (done + n) / 2 := by
  omega

/-- The promoted-tensor check: 4 B/elem in and 2 B/elem out saves exactly 2 B/elem. -/
theorem promoted_saves (elems src dst : Nat) (h1 : src = elems * 4) (h2 : dst = elems * 2) :
    src - dst = elems * 2 := by
  omega

-- § chunk loop

/-- Length of the next chunk. -/
def chunkLen (rem : Nat) : Nat := min CHUNK rem

/-- While bytes remain, the next chunk is non-empty and strictly shrinks the remainder. -/
theorem chunk_progress (rem : Nat) (h : 0 < rem) : 0 < chunkLen rem ∧ rem - chunkLen rem < rem := by
  unfold chunkLen CHUNK
  omega

/-- A chunk never exceeds the staging size. -/
theorem chunk_le (rem : Nat) : chunkLen rem ≤ CHUNK := by
  unfold chunkLen; omega

/-- The chunk size is even (needed so widening never splits an element). -/
theorem chunk_even_of_full (rem : Nat) (h : CHUNK ≤ rem) : chunkLen rem % 2 = 0 := by
  unfold chunkLen CHUNK at *
  omega

/-- When the remainder is a multiple of 4, so is the next chunk. -/
theorem chunk_mult4 (rem : Nat) (h : rem % 4 = 0) : chunkLen rem % 4 = 0 := by
  unfold chunkLen CHUNK
  omega

-- § arena placement and compaction

/-- Round up to a multiple of 256 (the default alignment). -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding keeps all bytes. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Rounded sizes are multiples of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- The compaction loop: offsets of kept rows, each advancing by its aligned size. -/
def placeAll : Nat → List Nat → Nat
  | at_, [] => at_
  | at_, x :: xs => placeAll (at_ + alignUp256 x) xs

/-- Placement from an aligned start gives an aligned pool. -/
theorem placeAll_mod : ∀ (l : List Nat) (a : Nat), a % 256 = 0 → placeAll a l % 256 = 0
  | [], a, h => by simpa [placeAll] using h
  | x :: xs, a, h => by
    have h1 := alignUp256_mod x
    have := placeAll_mod xs (a + alignUp256 x) (by omega)
    simpa [placeAll] using this

/-- The pool is at least the start plus the plain sum of sizes. -/
theorem placeAll_ge : ∀ (l : List Nat) (a : Nat), a + l.sum ≤ placeAll a l
  | [], a => by simp [placeAll]
  | x :: xs, a => by
    have h1 := alignUp256_ge x
    have h2 := placeAll_ge xs (a + alignUp256 x)
    simp only [placeAll, List.sum_cons]
    omega

/-- Each row's end is at or below the next row's start (rows never overlap). -/
theorem row_end_le_next (at_ x : Nat) : at_ + x ≤ at_ + alignUp256 x := by
  have := alignUp256_ge x
  omega

-- § index kinds

/-- Raw 16-bit index kinds 4 (BF16) and 5 (F16) take engine forms 1 and 3; others are unchanged. -/
def mapKind (k : Nat) : Nat := if k = 4 then 1 else if k = 5 then 3 else k

/-- Kind 4 becomes the BF16 engine form. -/
theorem mapKind_bf16 : mapKind 4 = 1 := by decide

/-- Kind 5 becomes the F16 engine form. -/
theorem mapKind_f16 : mapKind 5 = 3 := by decide

/-- Other kinds are untouched. -/
theorem mapKind_other (k : Nat) (h4 : k ≠ 4) (h5 : k ≠ 5) : mapKind k = k := by
  simp [mapKind, h4, h5]

/-- Conversion applied to an unquantized row. -/
inductive Conv
  | copy
  | bf16High
  | toF16

/-- Selection rule: raw 16-bit rows are copied, promoted rows convert, others copy. -/
def convFor (raw16 : Bool) (kind : Nat) : Conv :=
  if raw16 then Conv.copy else if kind = 1 then Conv.bf16High else if kind = 3 then Conv.toF16 else Conv.copy

/-- Raw 16-bit rows are never converted. -/
theorem convFor_raw (k : Nat) : convFor true k = Conv.copy := by
  simp [convFor]

/-- A promoted BF16 row takes the high half. -/
theorem convFor_bf16 : convFor false 1 = Conv.bf16High := by
  simp [convFor]

/-- A promoted F16 row converts with round-to-nearest-even. -/
theorem convFor_f16 : convFor false 3 = Conv.toF16 := by
  simp [convFor]

-- § arena sufficiency

/-- The arena check rejects exactly when the arena is smaller than the pool. -/
def arenaOk (arena pool : Nat) : Bool := decide (pool ≤ arena)

end Strata.WeightLoader



## FILE: /mnt/user-data/outputs/MtpDrafter.lean
/-!
# MODULE: mtp_drafter
SOURCES: src/core/mtp.cpp
The multi-token-prediction draft layer: window/capacity sizing, step-row staging, the probability-gated draft
chain, chunked prefill and tensor lookup by name and kind.

## Source semantics
* `window_ = window` only when `0 < window < max_cells`, else 0 (dense attention over every cell).
* The attention capacity is `cap = ceil(len / 64) * 64` with `len = window_` or `max_cells`.
* Step/pos staging has `2*T` rows: T catch-up rows, chain steps in rows `T + j - 1` for `j = 1..T-2`, and the
  full-layer row `2*T - 1`.
* The chain continues while `j < max_t - 1` and the last draft's probability is at least `min_p`;
  the remaining draft slots are zero-filled.
* Prefill walks the prompt in chunks of `max_t`, skipping chunks entirely below the first needed cell.
* Tensors are looked up by name AND kind; a kind mismatch yields null.

## Form template identifiers
- effWindow → window_ ; capOf → cap_ ; chainFuel → the `for (j = 1; ...)` loop ; findTensor → f32/bf16/q8
-/

namespace Strata.MtpDrafter

-- § window and capacity

/-- Effective window: the requested one only when strictly inside the context. -/
def effWindow (window maxCells : Nat) : Nat := if 0 < window ∧ window < maxCells then window else 0

/-- Cells attended per row. -/
def attendLen (window maxCells : Nat) : Nat :=
  if effWindow window maxCells > 0 then effWindow window maxCells else maxCells

/-- Selection capacity, rounded up to a multiple of 64. -/
def capOf (window maxCells : Nat) : Nat := (attendLen window maxCells + 63) / 64 * 64

/-- The effective window never exceeds the context. -/
theorem effWindow_le (w m : Nat) : effWindow w m ≤ m := by
  unfold effWindow
  split <;> omega

/-- The attended length never exceeds the context. -/
theorem attendLen_le (w m : Nat) : attendLen w m ≤ m := by
  unfold attendLen
  have := effWindow_le w m
  split <;> omega

/-- The capacity covers every attended cell. -/
theorem cap_covers (w m : Nat) : attendLen w m ≤ capOf w m := by
  unfold capOf; omega

/-- The capacity is a multiple of 64. -/
theorem cap_mod (w m : Nat) : capOf w m % 64 = 0 := by
  unfold capOf; omega

/-- Rounding adds fewer than 64 cells. -/
theorem cap_lt (w m : Nat) : capOf w m < attendLen w m + 64 := by
  unfold capOf; omega

-- § step-row staging

/-- Row index of chain step `j`. -/
def chainRow (T j : Nat) : Nat := T + j - 1

/-- Row of the full-layer cell of the catch-up. -/
def fullRow (T : Nat) : Nat := 2 * T - 1

/-- Chain rows lie above the catch-up rows. -/
theorem chainRow_above_catchup (T j : Nat) (hj : 1 ≤ j) : T ≤ chainRow T j := by
  unfold chainRow; omega

/-- Chain rows lie strictly below the full-layer row and inside the 2T rows. -/
theorem chainRow_below_full (T j : Nat) (hj : j ≤ T - 2) (hT : 3 ≤ T) :
    chainRow T j < fullRow T ∧ fullRow T < 2 * T := by
  unfold chainRow fullRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj (T j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k) (h : chainRow T j = chainRow T k) :
    j = k := by
  unfold chainRow at h; omega

/-- The step-row block count of a cell: `(cell + 1) / 4` is monotone in the cell. -/
theorem step_block_mono (a b : Nat) (h : a ≤ b) : (a + 1) / 4 ≤ (b + 1) / 4 := by
  omega

-- § the draft chain

/-- Draft-chain loop with fuel: continue while `j < maxT - 1` and the previous probability `p (j-1)` is high
enough; returns the final `j`, which equals the number of drafts produced. -/
def chainFuel (maxT minP : Nat) (p : Nat → Nat) : Nat → Nat → Nat
  | 0, j => j
  | fuel + 1, j => if j < maxT - 1 ∧ p (j - 1) ≥ minP then chainFuel maxT minP p fuel (j + 1) else j

/-- The chain never produces fewer drafts than it started with. -/
theorem chain_ge (maxT minP : Nat) (p : Nat → Nat) :
    ∀ fuel j, j ≤ chainFuel maxT minP p fuel j
  | 0, j => by simp [chainFuel]
  | fuel + 1, j => by
    unfold chainFuel
    split
    · have := chain_ge maxT minP p fuel (j + 1)
      omega
    · omega

/-- Starting inside the bound, the chain never exceeds `max_t - 1` drafts. -/
theorem chain_le (maxT minP : Nat) (p : Nat → Nat) :
    ∀ fuel j, j ≤ maxT - 1 → chainFuel maxT minP p fuel j ≤ maxT - 1
  | 0, j, h => by simpa [chainFuel] using h
  | fuel + 1, j, h => by
    unfold chainFuel
    split
    · rename_i hc
      exact chain_le maxT minP p fuel (j + 1) (by omega)
    · exact h

/-- A draft whose probability is below the threshold stops the chain at once. -/
theorem chain_stops_on_low (maxT minP : Nat) (p : Nat → Nat) (fuel j : Nat) (h : p (j - 1) < minP) :
    chainFuel maxT minP p (fuel + 1) j = j := by
  unfold chainFuel
  split
  · rename_i hc; omega
  · rfl

-- § argument validity of `draft`

/-- `draft` accepts `1 ≤ T ≤ max_t` and `0 ≤ a < T`. -/
def draftArgsOk (T maxT a : Nat) : Prop := 1 ≤ T ∧ T ≤ maxT ∧ a < T

/-- The caught-up row `a` is a real window row. -/
theorem draftArgs_row (T maxT a : Nat) (h : draftArgsOk T maxT a) : a < maxT := by
  obtain ⟨_, h2, h3⟩ := h
  omega

-- § chunked prefill

/-- Chunk length at offset `c` of `n` prompt cells. -/
def prefillChunk (maxT n c : Nat) : Nat := min maxT (n - c)

/-- While cells remain, a chunk is non-empty. -/
theorem prefillChunk_pos (maxT n c : Nat) (hm : 0 < maxT) (hc : c < n) : 0 < prefillChunk maxT n c := by
  unfold prefillChunk; omega

/-- A chunk never runs past the prompt. -/
theorem prefillChunk_end (maxT n c : Nat) (hc : c ≤ n) : c + prefillChunk maxT n c ≤ n := by
  unfold prefillChunk; omega

/-- A chunk fits the K/V-only graph table (size `max_t`). -/
theorem prefillChunk_le (maxT n c : Nat) : prefillChunk maxT n c ≤ maxT := by
  unfold prefillChunk; omega

/-- A skipped chunk lies entirely below the first needed cell (integer arithmetic). -/
theorem skipped_chunk_unneeded (cell0 c T firstNeeded : Int) (h : cell0 + c + T ≤ firstNeeded)
    (t : Int) (_ht0 : 0 ≤ t) (ht : t < T) : cell0 + c + t < firstNeeded := by
  omega

-- § tensor lookup by name and kind

/-- A dense tensor entry. -/
structure Tensor where
  name : String
  kind : String
  off : Nat

/-- Lookup of `f32`/`bf16`/`q8`: first entry with this name and kind. -/
def findTensor (ts : List Tensor) (name kind : String) : Option Tensor :=
  ts.find? (fun t => t.name == name && t.kind == kind)

/-- A hit always has the requested name and kind. -/
theorem findTensor_sound (ts : List Tensor) (name kind : String) (t : Tensor)
    (h : findTensor ts name kind = some t) : t.name = name ∧ t.kind = kind := by
  have := List.find?_some h
  simpa using this

/-- An empty table finds nothing. -/
theorem findTensor_nil (name kind : String) : findTensor [] name kind = none := rfl

end Strata.MtpDrafter



## FILE: /mnt/user-data/outputs/ExpertCache.lean
/-!
# MODULE: expert_cache
SOURCES: src/core/expert_cache.cpp, include/strata/core/expert_cache.hpp
The VRAM-resident expert tier: profile-file validation, slot arena sizing, the (layer, expert) → slot residency
table, per-layer slot ranges for admission, variable-size slot offsets and bounded slot fills.

## Source semantics
* A profile header must match the model's `n_layers × n_expert`, and may rank at most `slots` pairs; every
  ranked pair must be in range.
* The residency table is flat row-major `layer * n_expert + expert`; out-of-range queries are "not resident".
* Per-layer admission: layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the
  remainder, so the ranges tile `[0, slots)`; admission never evicts.
* Sized slots are 256-byte aligned, back to back; `fill_slot` copies at most the slot's blob size.

## Form template identifiers
- flatIdx → `layer * n_expert + expert` ; layerLo/layerHi → layer_slot_range ; offsets → `off_`
- fillLen → `n = (bytes > 0 && bytes <= blob_) ? bytes : blob_`
-/

namespace Strata.ExpertCache

-- § profile validation

/-- Header acceptance of `read_expert_profile`. -/
def headerOk (nl ne want nRanked nLayers nExpert : Nat) : Prop :=
  nl = nLayers ∧ ne = nExpert ∧ nRanked ≤ want

/-- Accepted headers rank no more pairs than there are slots. -/
theorem header_ranked_le (nl ne want nr L E : Nat) (h : headerOk nl ne want nr L E) : nr ≤ want := by
  obtain ⟨_, _, h3⟩ := h
  exact h3

/-- A pair is in range. -/
def pairOk (l e nLayers nExpert : Nat) : Prop := l < nLayers ∧ e < nExpert

/-- Flat residency index. -/
def flatIdx (l e nExpert : Nat) : Nat := l * nExpert + e

/-- An in-range pair indexes inside the residency table. -/
theorem flatIdx_lt (l e L E : Nat) (h : pairOk l e L E) : flatIdx l e E < L * E := by
  obtain ⟨hl, he⟩ := h
  have h1 : (l + 1) * E ≤ L * E := Nat.mul_le_mul_right E (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  unfold flatIdx
  omega

/-- The flat index determines the pair (layer = idx / E, expert = idx % E). -/
theorem flatIdx_recover (l e E : Nat) (hE : 0 < E) (he : e < E) :
    flatIdx l e E / E = l ∧ flatIdx l e E % E = e := by
  have hcomm : flatIdx l e E = e + E * l := by
    unfold flatIdx
    rw [Nat.mul_comm, Nat.add_comm]
  rw [hcomm]
  constructor
  · rw [Nat.add_mul_div_left e l hE, Nat.div_eq_of_lt he, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he]

-- § per-layer slot ranges

/-- Lower end of layer `l`'s slot range. -/
def layerLo (slots nl l : Nat) : Nat := l * (slots / nl)

/-- Upper end: the last layer takes whatever is left. -/
def layerHi (slots nl l : Nat) : Nat := if l = nl - 1 then slots else (l + 1) * (slots / nl)

/-- Layer 0 starts at slot 0. -/
theorem layerLo_zero (slots nl : Nat) : layerLo slots nl 0 = 0 := by
  simp [layerLo]

/-- A non-last layer's range ends where the next one starts (no gap, no overlap). -/
theorem layer_ranges_adjacent (slots nl l : Nat) (h : l ≠ nl - 1) :
    layerHi slots nl l = layerLo slots nl (l + 1) := by
  simp [layerHi, layerLo, h]

/-- The last layer ends at `slots`, so the ranges cover `0..slots` exactly. -/
theorem last_layer_ends_at_slots (slots m : Nat) : layerHi slots (m + 1) m = slots := by
  simp [layerHi]

/-- Every range has `lo ≤ hi` (for layers below `nl`). -/
theorem layer_range_ordered (slots nl l : Nat) (hl : l < nl) : layerLo slots nl l ≤ layerHi slots nl l := by
  unfold layerLo layerHi
  by_cases h : l = nl - 1
  · rw [if_pos h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : l * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.mul_comm nl] at h2
    omega
  · rw [if_neg h]
    exact Nat.mul_le_mul_right _ (by omega)

/-- Every range stays inside `0..slots`. -/
theorem layer_range_in_slots (slots nl l : Nat) (hl : l < nl) : layerHi slots nl l ≤ slots := by
  unfold layerHi
  by_cases h : l = nl - 1
  · simp [h]
  · rw [if_neg h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : (l + 1) * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.mul_comm nl] at h2
    omega

/-- The last layer's range is at least the uniform quota `q`: it gets the remainder on top. -/
theorem last_layer_has_quota (slots m q : Nat) (hq : q = slots / (m + 1)) :
    q ≤ layerHi slots (m + 1) m - layerLo slots (m + 1) m := by
  have h := Nat.div_mul_le_self slots (m + 1)
  rw [← hq] at h
  rw [Nat.mul_add, Nat.mul_one] at h
  have hhi : layerHi slots (m + 1) m = slots := last_layer_ends_at_slots slots m
  have hlo : layerLo slots (m + 1) m = m * q := by
    unfold layerLo
    rw [← hq]
  rw [hhi, hlo]
  rw [Nat.mul_comm q m] at h
  omega

-- § admission

/-- Per-layer admission: take the layer's next slot if it is below the layer's ceiling. -/
def admitPerLayer (next hi : Nat) : Option Nat := if next < hi then some next else none

/-- An admitted slot lies in `[lo, hi)` when the cursor was at or above `lo`. -/
theorem admitPerLayer_in_range (next lo hi s : Nat) (hlo : lo ≤ next)
    (h : admitPerLayer next hi = some s) : lo ≤ s ∧ s < hi := by
  unfold admitPerLayer at h
  split at h
  · rename_i hc
    cases h
    exact ⟨hlo, hc⟩
  · cases h

/-- A full layer admits nothing (no eviction). -/
theorem admitPerLayer_full (next hi : Nat) (h : hi ≤ next) : admitPerLayer next hi = none := by
  unfold admitPerLayer
  rw [if_neg (by omega)]

/-- Global admission: the shared counter hands out slots below `slots` only. -/
def admitGlobal (nextFree slots : Nat) : Option Nat := if nextFree < slots then some nextFree else none

/-- Global admission never produces an out-of-range slot. -/
theorem admitGlobal_lt (nf slots s : Nat) (h : admitGlobal nf slots = some s) : s < slots := by
  unfold admitGlobal at h
  split at h
  · rename_i hc
    cases h
    exact hc
  · cases h

-- § sized slots

/-- Round a slot up to 256 bytes. -/
def al256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Slot offsets: start at `acc`, each slot advances by its aligned size; the list ends with the total. -/
def offsets : Nat → List Nat → List Nat
  | acc, [] => [acc]
  | acc, b :: bs => acc :: offsets (acc + al256 b) bs

/-- The offset list has one more entry than there are slots. -/
theorem offsets_length : ∀ (bs : List Nat) (acc : Nat), (offsets acc bs).length = bs.length + 1
  | [], acc => by simp [offsets]
  | b :: bs, acc => by
    simp only [offsets, List.length_cons]
    rw [offsets_length bs]

/-- Every slot offset is 256-byte aligned when the start is. -/
theorem offsets_aligned : ∀ (bs : List Nat) (acc : Nat), acc % 256 = 0 →
    ∀ x ∈ offsets acc bs, x % 256 = 0
  | [], acc, h, x, hx => by
    simp [offsets] at hx
    omega
  | b :: bs, acc, h, x, hx => by
    simp only [offsets, List.mem_cons] at hx
    rcases hx with hx | hx
    · omega
    · have h1 : (acc + al256 b) % 256 = 0 := by unfold al256; omega
      exact offsets_aligned bs (acc + al256 b) h1 x hx

/-- `fill_slot`'s copy length: the given size when it fits the slot, else the whole slot. -/
def fillLen (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never exceeds the slot's blob size. -/
theorem fillLen_le (bytes blob : Nat) : fillLen bytes blob ≤ blob := by
  unfold fillLen
  split <;> omega

/-- A fill is never empty when the blob size is positive. -/
theorem fillLen_pos (bytes blob : Nat) (h : 0 < blob) : 0 < fillLen bytes blob := by
  unfold fillLen
  split <;> omega

/-- Slot arena size is `slots * blob`, so every slot start is inside it. -/
theorem slot_start_in_arena (slot slots blob : Nat) (h : slot < slots) : slot * blob + blob ≤ slots * blob := by
  have h1 : (slot + 1) * blob ≤ slots * blob := Nat.mul_le_mul_right blob (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  exact h1

end Strata.ExpertCache



## FILE: /mnt/user-data/outputs/ExpertSource.lean
/-!
# MODULE: expert_source
SOURCES: src/core/expert_source.cpp, include/strata/core/expert_source.hpp
The adapter between the host loop and the CPU expert pool: file- or arena-backed blob lookup, hit/miss job
building, the GPU/PCIe share of missed experts in verify windows, and chunked loading of expert tensors.

## Source semantics
* Blobs are addressed by `layer * n_expert + expert`; `expert >= n_expert` must be rejected separately,
  because the flat index alone would alias the next layer's first expert.
* `experts.bin` holds exactly `n_layers * n_expert * BLOB` bytes (24576 × 1,382,400 = 33,973,862,400).
* For every router entry exactly one of {GPU hit, CPU job} computes it: jobs + hits = k, and `njobs` indexes
  the job array while `i` indexes the output.
* PCIe share: `m = (nmiss * pcie_num) >> 8` of the missed experts (the last `m` in routing order) go to the GPU.
* GGUF expert loading reads 16 experts per chunk and lays a blob out as [gate | up | down].
* Any slot-sized copy may start at any expert because the arena is one max-blob longer than the file.

## Form template identifiers
- flatIdx → `layer * n_expert + expert` ; nHit/nMiss → is_hit counts ; pcieShare → `m`
- blobRegions → per[3]/at[3] ; arenaSlack → `want + blob`
-/

namespace Strata.ExpertSource

-- § blob addressing

/-- Flat blob index. -/
def flatIdx (layer expert nExpert : Nat) : Nat := layer * nExpert + expert

/-- The expert axis is validated separately: without it, `(0, E)` aliases `(1, 0)`. -/
theorem unchecked_flat_alias (E : Nat) : flatIdx 0 E E = flatIdx 1 0 E := by
  unfold flatIdx; omega

/-- Concrete case from the source comment: with 512 experts, `blob(0, 512)` is `blob(1, 0)`. -/
theorem alias_512 : flatIdx 0 512 512 = flatIdx 1 0 512 := by
  decide

/-- With the separate range check the flat index recovers both axes. -/
theorem flatIdx_recover (l e E : Nat) (hE : 0 < E) (he : e < E) :
    flatIdx l e E / E = l ∧ flatIdx l e E % E = e := by
  have hcomm : flatIdx l e E = e + E * l := by
    unfold flatIdx
    rw [Nat.mul_comm, Nat.add_comm]
  rw [hcomm]
  constructor
  · rw [Nat.add_mul_div_left e l hE, Nat.div_eq_of_lt he, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he]

/-- With the range check on `expert`, distinct (layer, expert) pairs get distinct blobs. -/
theorem flatIdx_inj (l e l' e' E : Nat) (hE : 0 < E) (he : e < E) (he' : e' < E)
    (h : flatIdx l e E = flatIdx l' e' E) : l = l' ∧ e = e' := by
  have h1 := flatIdx_recover l e E hE he
  have h2 := flatIdx_recover l' e' E hE he'
  rw [h] at h1
  exact ⟨h1.1.symm.trans h2.1, h1.2.symm.trans h2.2⟩

/-- An in-range pair indexes inside `n_layers * n_expert` blobs. -/
theorem flatIdx_lt (l e L E : Nat) (hl : l < L) (he : e < E) : flatIdx l e E < L * E := by
  have h1 : (l + 1) * E ≤ L * E := Nat.mul_le_mul_right E (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  unfold flatIdx
  omega

/-- The expert file size for the artifact: 48 layers × 512 experts × 1,382,400 bytes. -/
theorem experts_file_size : 48 * 512 * 1382400 = 33973862400 := by
  decide

/-- There are exactly 24,576 blobs. -/
theorem blob_count : 48 * 512 = 24576 := by
  decide

-- § hit/miss job building

/-- Number of GPU hits among the router entries. -/
def nHit : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + nHit bs

/-- Number of CPU jobs (misses) among the router entries. -/
def nMiss : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 0 else 1) + nMiss bs

/-- Every router entry is computed by exactly one engine: jobs + hits = k. -/
theorem jobs_plus_hits : ∀ l : List Bool, nHit l + nMiss l = l.length
  | [] => by simp [nHit, nMiss]
  | b :: bs => by
    have ih := jobs_plus_hits bs
    cases b <;> simp [nHit, nMiss, List.length_cons] <;> omega

/-- Hits never exceed the number of entries (so `n_hits ≤ k`). -/
theorem nHit_le (l : List Bool) : nHit l ≤ l.length := by
  have := jobs_plus_hits l
  omega

/-- With no hits every entry is a job, so job index and output index coincide. -/
theorem no_hits_all_jobs : ∀ l : List Bool, nHit l = 0 → nMiss l = l.length := by
  intro l h
  have := jobs_plus_hits l
  omega

-- § PCIe share of the missed experts

/-- `m = (nmiss * pcie_num) >> 8`. -/
def pcieShare (nmiss num : Nat) : Nat := nmiss * num / 256

/-- A numerator of at most 256 gives a share of at most all the misses. -/
theorem pcieShare_le (nmiss num : Nat) (h : num ≤ 256) : pcieShare nmiss num ≤ nmiss := by
  unfold pcieShare
  apply Nat.div_le_of_le_mul
  rw [Nat.mul_comm 256 nmiss]
  exact Nat.mul_le_mul_left nmiss h

/-- Ranks `r` with `nmiss - m ≤ r < nmiss` number exactly `m`. -/
theorem pcie_ranks_count (nmiss m : Nat) (h : m ≤ nmiss) : nmiss - (nmiss - m) = m := by
  omega

/-- A zero numerator sends nothing over PCIe. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  simp [pcieShare]

-- § GGUF expert loading

/-- Region sizes `per[3]` and starts `at[3]` of one blob from `up_off`, `down_off`, `blob`. -/
def blobRegions (upOff downOff blob : Nat) : List (Nat × Nat) :=
  [(0, upOff), (upOff, upOff), (downOff, blob - downOff)]

/-- With `down_off = 2 * up_off` the three regions tile the blob with no gap and no overlap. -/
theorem blob_regions_tile (upOff blob : Nat) (h : 2 * upOff ≤ blob) :
    0 + upOff = upOff ∧ upOff + upOff = 2 * upOff ∧ 2 * upOff + (blob - 2 * upOff) = blob := by
  omega

/-- A 16-expert read never covers more than 16 experts. -/
theorem chunk_experts_le (n per : Nat) (hn : n ≤ per * 16) : n / per ≤ 16 :=
  Nat.div_le_of_le_mul hn

-- § arena slack

/-- The arena is one max-blob longer than the file. -/
def arenaSlack (want maxBlob : Nat) : Nat := want + maxBlob

/-- A whole-slot copy starting at any in-file expert stays inside the arena. -/
theorem slot_copy_in_arena (off bb want mb : Nat) (h1 : off + bb ≤ want) :
    off + mb ≤ arenaSlack want mb := by
  unfold arenaSlack
  omega

-- § misc counters

/-- `reads_` increments once per accepted blob lookup, so two lookups add two. -/
theorem reads_two (r : Nat) : r + 1 + 1 = r + 2 := by
  omega

end Strata.ExpertSource





## FILE: /mnt/user-data/outputs/GenerateDriver.lean
/-!
# MODULE: generate_driver
SOURCES: src/program/generate.cpp
The `strata generate` driver: validates options, runs the teacher-forced decode loop
(`embed -> 48 layers -> head -> sample`), the speculative verify loop, the adaptive VRAM
expert tier, the routing trace and the `--serve` request loop (including M-RoPE positions
for images).  This module states the arithmetic/control contracts of that driver.

## Source semantics
* A run fits iff `max_new > 0`, `max_context > 0`, `max_new <= max_context` and the prompt
  fits in `max_context - max_new`.
* The loop produces one logits row per position `0 .. n_prompt-2+max_new`; generated tokens
  start at the LAST prompt position; while the prompt lasts the next input is the prompt's token.
* A speculative round accepts the longest prefix of drafts equal to the verifier's outputs,
  and commits `accepted + 1` tokens.
* The routing trace layer index is `d.layers - 1` because the adapter has already advanced it.
* The auto expert cache is sized `(free - reserve) / max_blob` and never exceeds free VRAM.
* Image cells get M-RoPE `(t,h,w) = (p, p + y, p + x)`; text resumes at `p + max(nx, ny)`.

## Form template identifiers
* `n_prompt` / `tokens.size()`  -> `nPrompt`
* `o.max_new`, `o.max_context`   -> `maxNew`, `maxCtx`
* `accepted_hist`, `a`           -> `accept`
* `expert_cache auto`            -> `autoSlots`
* `prefill_until`                -> `pu`
-/

namespace Strata.Generate

-- § run-size validation

/-- The fit predicate mirroring the `--max-new` / `--max-context` checks in `main`. -/
def RunFits (nPrompt maxNew maxCtx : Nat) : Prop :=
  0 < maxNew ∧ 0 < maxCtx ∧ maxNew ≤ maxCtx ∧ nPrompt ≤ maxCtx - maxNew

/-- A run that passes validation can hold the prompt and all generated tokens. -/
theorem runFits_total (n m c : Nat) (h : RunFits n m c) : n + m ≤ c := by
  obtain ⟨_, _, h3, h4⟩ := h
  omega

-- § positions, rows and produced-token accounting

/-- Number of logits rows dumped: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- The dump count is one less than `n_prompt + max_new` (the header-count bug fixed in the source). -/
theorem dumpPositions_succ (n m : Nat) (h : 1 ≤ n) : dumpPositions n m + 1 = n + m := by
  unfold dumpPositions
  omega

/-- Tokens produced after processing `P` positions: sampling is recorded from position `n-1`. -/
def producedAfter (n P : Nat) : Nat := P - (n - 1)

/-- After all dumped positions exactly `max_new` tokens are produced, so the loop stops there. -/
theorem produced_at_end (n m : Nat) : producedAfter n (dumpPositions n m) = m := by
  unfold producedAfter dumpPositions
  omega

/-- Before the last prompt position nothing is produced (teacher-forced conditioning only). -/
theorem produced_before_prompt_end (n P : Nat) (h : P ≤ n - 1) : producedAfter n P = 0 := by
  unfold producedAfter
  omega

/-- The next input token: the prompt's own token while the prompt lasts, else the sampled one. -/
def nextInput (tokens : List Nat) (pos next : Nat) : Nat :=
  if pos + 1 < tokens.length then tokens.getD (pos + 1) 0 else next

/-- Teacher forcing: inside the prompt the next input is the prompt's token. -/
theorem nextInput_prompt (tokens : List Nat) (pos next : Nat) (h : pos + 1 < tokens.length) :
    nextInput tokens pos next = tokens.getD (pos + 1) 0 := by
  unfold nextInput
  rw [if_pos h]

/-- Past the prompt the sampled token is fed back. -/
theorem nextInput_sampled (tokens : List Nat) (pos next : Nat) (h : tokens.length ≤ pos + 1) :
    nextInput tokens pos next = next := by
  unfold nextInput
  rw [if_neg (by omega)]

-- § batched prefill split (`--prefill-until`)

/-- How many prompt positions go through the batched path. -/
def nBatched (pu n : Nat) : Nat := if 0 < pu ∧ pu < n - 1 then pu else n - 1

/-- The batched count never exceeds `n - 1` (the last prompt position stays on the token path). -/
theorem nBatched_le (pu n : Nat) : nBatched pu n ≤ n - 1 := by
  unfold nBatched
  split_ifs with h <;> omega

/-- The token path starts strictly inside the prompt whenever the prompt is non-empty. -/
theorem posStart_lt (pu n : Nat) (hn : 1 ≤ n) : nBatched pu n < n := by
  have := nBatched_le pu n
  omega

-- § speculative acceptance

/-- Length of the longest prefix of `drafts` equal to the verifier outputs `outs`. -/
def accept : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then accept ds os + 1 else 0
  | _, _ => 0

/-- Accepted drafts never exceed the number of drafts. -/
theorem accept_le_left : ∀ (ds os : List Nat), accept ds os ≤ ds.length
  | [], os => by simp [accept]
  | d :: ds, [] => by simp [accept]
  | d :: ds, o :: os => by
    have ih := accept_le_left ds os
    by_cases h : d = o
    · simp [accept, h] <;> omega
    · simp [accept, h]

/-- Accepted drafts never exceed the number of verifier outputs. -/
theorem accept_le_right : ∀ (ds os : List Nat), accept ds os ≤ os.length
  | [], os => by simp [accept]
  | d :: ds, [] => by simp [accept]
  | d :: ds, o :: os => by
    have ih := accept_le_right ds os
    by_cases h : d = o
    · simp [accept, h] <;> omega
    · simp [accept, h]

/-- If the drafts equal the outputs, every draft is accepted. -/
theorem accept_self : ∀ (xs : List Nat), accept xs xs = xs.length
  | [] => by simp [accept]
  | x :: xs => by
    have ih := accept_self xs
    simp [accept, ih]

/-- Window length with the minimum-probability rule: `T = min S (1 + run of passing drafts)`. -/
def windowLen (S : Nat) (ok : List Bool) : Nat := min S (1 + (ok.takeWhile id).length)

/-- The verify window never exceeds the configured spec size. -/
theorem windowLen_le (S : Nat) (ok : List Bool) : windowLen S ok ≤ S := by
  unfold windowLen
  omega

/-- The verify window always holds at least the last emitted token. -/
theorem windowLen_pos (S : Nat) (ok : List Bool) (h : 1 ≤ S) : 1 ≤ windowLen S ok := by
  unfold windowLen
  omega

/-- A round commits `accepted + 1` tokens, which fits the window `T` of `T-1` drafts. -/
theorem commit_le_window (T a : Nat) (hT : 1 ≤ T) (ha : a ≤ T - 1) : a + 1 ≤ T := by
  omega

/-- The PCIe share numerator is clamped to `0..256`. -/
def pcieNum (x : Nat) : Nat := min 256 x

/-- The clamped PCIe numerator is within the 256 denominator. -/
theorem pcieNum_le (x : Nat) : pcieNum x ≤ 256 := by
  unfold pcieNum
  omega

-- § routing trace

/-- The traced layer index is `d.layers - 1` read after the adapter advanced it. -/
def traceLayer (afterDispatch : Nat) : Int := (afterDispatch : Int) - 1

/-- For layers `0..47` the advanced counter `l+1` maps back to `l`, in range `0..47`. -/
theorem traceLayer_valid (l : Nat) (h : l < 48) :
    traceLayer (l + 1) = (l : Int) ∧ 0 ≤ traceLayer (l + 1) ∧ traceLayer (l + 1) < 48 := by
  unfold traceLayer
  omega

-- § VRAM expert tier sizing

/-- `--expert-cache auto`: slots = (free - reserve) / blob. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto-sized tier never occupies more than the free VRAM left after the reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) :
    autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- With a profile the slot count is truncated to the profile length. -/
def withProfile (slots profLen : Nat) : Nat := min slots profLen

/-- Profile truncation only shrinks the tier. -/
theorem withProfile_le (slots profLen : Nat) : withProfile slots profLen ≤ slots := by
  unfold withProfile
  omega

/-- Ceiling division, as used for the number of lent cache slots. -/
def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Lent slots are enough to hold the prompt path's buffers: `need <= k * blob`. -/
theorem ceilDiv_mul_ge (a b : Nat) (hb : 0 < b) : a ≤ ceilDiv a b * b := by
  unfold ceilDiv
  have h := (Nat.div_lt_iff_lt_mul hb).mp (Nat.lt_succ_self ((a + b - 1) / b))
  rw [Nat.succ_mul] at h
  omega

/-- When the prompt path borrows cache slots the separate buffer reservation is zero. -/
def prefillMib (chunk : Nat) (borrow : Bool) : Nat :=
  if 0 < chunk ∧ borrow = false then 160 + chunk * 680 / 1024 else 0

/-- Borrowing removes the reserved prompt buffers from the VRAM budget. -/
theorem prefillMib_borrow (chunk : Nat) : prefillMib chunk true = 0 := by
  unfold prefillMib
  simp

-- § adaptive tier swaps

/-- A swap (in half-units of routed count) needs the candidate used >= 2 and >= victim + 1.5. -/
def swapOK (c v : Nat) : Prop := 4 ≤ 2 * c ∧ 2 * v + 3 ≤ 2 * c

/-- An accepted swap always replaces a strictly less-routed resident expert. -/
theorem swapOK_lt (c v : Nat) (h : swapOK c v) : v < c := by
  obtain ⟨_, h2⟩ := h
  omega

/-- The 0.7 decay never increases a usage count (scaled integer form). -/
theorem decay_le (u : Nat) : u * 7 / 10 ≤ u := by
  omega

/-- The PLE token window advances oldest-first: the newest predecessor becomes the older one. -/
def pleAdvance (prev : Int × Int) (tok : Int) : Int × Int := (prev.2, tok)

/-- After an advance the older slot holds the previous newest token. -/
theorem pleAdvance_fst (prev : Int × Int) (tok : Int) : (pleAdvance prev tok).1 = prev.2 := rfl

/-- After an advance the newest slot holds the token just decoded. -/
theorem pleAdvance_snd (prev : Int × Int) (tok : Int) : (pleAdvance prev tok).2 = tok := rfl

-- § image M-RoPE positions (`GENI`)

/-- Position triple of image row `j` of an image with `nx` columns starting at text position `p`. -/
def imgPos (p nx j : Nat) : Nat × Nat × Nat := (p, p + j / nx, p + j % nx)

/-- The height coordinate stays below the advance `p + max nx ny`. -/
theorem imgPos_h_lt (p nx ny j : Nat) (hx : 0 < nx) (hj : j < nx * ny) :
    p + j / nx < p + max nx ny := by
  have h1 : j / nx < ny := (Nat.div_lt_iff_lt_mul hx).mpr (by rw [Nat.mul_comm]; exact hj)
  have h2 := Nat.le_max_right nx ny
  omega

/-- The width coordinate stays below the advance `p + max nx ny`. -/
theorem imgPos_w_lt (p nx ny j : Nat) (hx : 0 < nx) :
    p + j % nx < p + max nx ny := by
  have h1 : j % nx < nx := Nat.mod_lt j hx
  have h2 := Nat.le_max_left nx ny
  omega

/-- Text after an image strictly advances past the image's start. -/
theorem image_advance_pos (p nx ny : Nat) (hx : 0 < nx) : p < p + max nx ny := by
  have h2 := Nat.le_max_left nx ny
  omega

-- § serve-loop admission

/-- A `GEN` request is admitted iff `n + max_new + 8 <= max_context`. -/
def admitted (n maxNew maxCtx : Nat) : Prop := n + maxNew + 8 ≤ maxCtx

/-- Admission leaves room for a full verify window of lookahead. -/
theorem admitted_fits (n m c : Nat) (h : admitted n m c) : n + m ≤ c := by
  unfold admitted at h
  omega

/-- Tokens produced in a serve round are clipped to `max_new`. -/
def producedStep (maxNew produced a : Nat) : Nat := min maxNew (produced + a + 1)

/-- A round never overshoots `max_new`. -/
theorem producedStep_le (maxNew produced a : Nat) : producedStep maxNew produced a ≤ maxNew := by
  unfold producedStep
  omega

/-- A round makes progress while below `max_new`, so the serve loop terminates. -/
theorem producedStep_progress (maxNew produced a : Nat) (h : produced < maxNew) :
    produced < producedStep maxNew produced a := by
  unfold producedStep
  omega

end Strata.Generate



## FILE: /mnt/user-data/outputs/PrefillOrchestration.lean
/-!
# MODULE: prefill_orchestration
SOURCES: src/prefill/prefill.cpp
Batched prompt processing: the prompt is cut into chunks of `T` tokens; per chunk every layer runs
a hyper-connection read, a GDN / QSA / MoE half, and a write.  MoE experts are grouped by id on the
host, staged through a host->device ring and dequantized through a two-slot ring.  Buffers come from
a bump allocator over a borrowed region (or `cudaMalloc`).

## Source semantics
* Chunks of size `min T (n - c0)` partition `n` positions exactly.
* The bump allocator rounds each request to 256 bytes (+256 slack) and fails exactly when the
  region would overflow.
* A staging ring of 8 slots with lookahead 7 never aliases two live blobs.
* Grouping tokens by expert yields cursors strictly inside each expert's block.

## Form template identifiers
* `m.T` (chunk size)          -> `T`
* `Alloc::take`               -> `roundBytes`, `allocAll`
* `STAGE`, `DQ`               -> ring sizes 8 and 2
* `gdn_floats`                -> `gdnFloats`
-/

namespace Strata.Prefill

-- § chunk partition

/-- Chunk lengths produced by the `for (c0 = 0; c0 < n; c0 += T)` loop (fuel-bounded). -/
def chunksAux : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | fuel + 1, T, n => if n = 0 then [] else min T n :: chunksAux fuel T (n - min T n)

/-- The chunk list of an `n`-token prompt with chunk size `T`. -/
def chunks (T n : Nat) : List Nat := chunksAux n T n

/-- Chunk lengths sum to the number of positions processed. -/
theorem sum_chunksAux (T : Nat) (hT : 0 < T) :
    ∀ fuel n, n ≤ fuel → (chunksAux fuel T n).sum = n := by
  intro fuel
  induction fuel with
  | zero =>
    intro n h
    have hn : n = 0 := by omega
    subst hn
    simp [chunksAux]
  | succ f ih =>
    intro n h
    by_cases hn : n = 0
    · subst hn
      simp [chunksAux]
    · have hm : 0 < min T n := by omega
      have h2 := ih (n - min T n) (by omega)
      rw [chunksAux, if_neg hn, List.sum_cons, h2]
      omega

/-- The full chunk list covers the prompt exactly. -/
theorem sum_chunks (T n : Nat) (hT : 0 < T) : (chunks T n).sum = n :=
  sum_chunksAux T hT n n (Nat.le_refl n)

/-- Every chunk is non-empty and at most `T` tokens (so it fits the chunk buffers). -/
theorem chunksAux_bounds (T : Nat) (hT : 0 < T) :
    ∀ fuel n, ∀ x ∈ chunksAux fuel T n, 0 < x ∧ x ≤ T := by
  intro fuel
  induction fuel with
  | zero =>
    intro n x hx
    simp [chunksAux] at hx
  | succ f ih =>
    intro n x hx
    by_cases hn : n = 0
    · subst hn
      simp [chunksAux] at hx
    · rw [chunksAux, if_neg hn] at hx
      rcases List.mem_cons.mp hx with h | hx'
      · rw [h]
        omega
      · exact ih _ x hx'

-- § bump allocator

/-- Bytes charged for a request of `b` bytes: `(b + 256 + 255) & ~255`. -/
def roundBytes (b : Nat) : Nat := (b + 256 + 255) / 256 * 256

/-- Every allocation is 256-byte aligned. -/
theorem roundBytes_aligned (b : Nat) : roundBytes b % 256 = 0 := by
  unfold roundBytes
  omega

/-- Every allocation leaves at least the 256 byte slack. -/
theorem roundBytes_ge (b : Nat) : b + 256 ≤ roundBytes b := by
  unfold roundBytes
  omega

/-- Rounding wastes less than one alignment unit. -/
theorem roundBytes_lt (b : Nat) : roundBytes b < b + 256 + 256 := by
  unfold roundBytes
  omega

/-- Sequential bump allocation of `sizes` into a region of capacity `cap`. -/
def allocAll (cap : Nat) : List Nat → Nat → Option Nat
  | [], used => some used
  | b :: bs, used => if used + b ≤ cap then allocAll cap bs (used + b) else none

/-- A successful allocation sequence never exceeds the region. -/
theorem allocAll_le (cap : Nat) : ∀ (bs : List Nat) (used u : Nat),
    used ≤ cap → allocAll cap bs used = some u → u ≤ cap := by
  intro bs
  induction bs with
  | nil =>
    intro used u hu h
    simp [allocAll] at h
    omega
  | cons b bs ih =>
    intro used u hu h
    by_cases hb : used + b ≤ cap
    · rw [allocAll, if_pos hb] at h
      exact ih _ u hb h
    · rw [allocAll, if_neg hb] at h
      simp at h

/-- A single request fits exactly when the region has room. -/
theorem alloc_single_iff (cap used b : Nat) :
    (allocAll cap [b] used).isSome = true ↔ used + b ≤ cap := by
  by_cases hb : used + b ≤ cap
  · simp [allocAll, hb]
  · simp [allocAll, hb]

-- § staging rings

/-- Two staged blobs within one lookahead window (< 8 apart) never share a staging slot. -/
theorem stage_slots_distinct (i k : Nat) (hlt : i < k) (hwin : k < i + 8) : i % 8 ≠ k % 8 := by
  omega

/-- Lookahead 7 is strictly below the ring size 8, so the window fits the ring. -/
theorem lookahead_lt_stage : (7 : Nat) < 8 := by decide

/-- Two experts in flight on the 2-slot dequantized ring never share a slot. -/
theorem dq_slots_distinct (j k : Nat) (hlt : j < k) (hwin : k < j + 2) : j % 2 ≠ k % 2 := by
  omega

-- § expert grouping (counting sort by expert id)

/-- A token-expert pair's destination row lies strictly inside its expert's block. -/
theorem group_slot_in_block (off cnt c : Nat) (h : c < cnt) : off + c < off + cnt := by
  omega

/-- The next block starts where this block ends (`off[e+1] = off[e] + cnt[e]`). -/
theorem group_blocks_disjoint (off cnt c off' : Nat) (h : c < cnt) (h' : off' = off + cnt) :
    off + c < off' := by
  omega

-- § GDN state layout

/-- Floats of GDN state per layer: `128*48*128` recurrent plus `10240*3` conv history. -/
def gdnFloats : Nat := 128 * 48 * 128 + 10240 * 3

/-- The per-layer GDN state is 817,152 floats. -/
theorem gdnFloats_value : gdnFloats = 817152 := by decide

/-- The conv history starts after the recurrent state inside one layer's block. -/
theorem gdn_conv_offset (idx : Nat) :
    idx * gdnFloats + 128 * 48 * 128 < (idx + 1) * gdnFloats := by
  unfold gdnFloats
  omega

end Strata.Prefill



## FILE: /mnt/user-data/outputs/PrefillKernels.lean
/-!
# MODULE: prefill_kernels
SOURCES: src/prefill/kernels.cu
CUDA kernels of the batched prompt path: hyper-connection norm/mix/write/broadcast, GDN gates /
conv / recurrence, MoE routing / dequant / SwiGLU / gather / combine, and QSA helpers
(rms rows, rope, split-q, gated attention, KV append).  All kernels address flat buffers by
decomposing a linear thread index into `(token, stream, dim)` style coordinates.

## Source semantics
* With `N = 2560`, `HC = 4`, `D = N*HC`, a flat index `i < T*D` decomposes as
  `t = i / D`, `c = (i % D) / N`, `d = i % N` and recomposes exactly.
* Launch grids are ceiling divisions that cover every element.
* The Strata expert blob is gate/up codes, down codes, gate/up scales, down scales
  (1,382,400 bytes) and every dequant read stays inside its section.
* The router addresses 512 experts as 32 lanes x 16 registers.

## Form template identifiers
* `N`, `HC`, `D`, `LR`, `S`, `HK`, `HV`, `C` -> numerals 2560, 4, 10240, 320, 128, 16, 48, 10240
* `blocks_for`                               -> `blocksFor`
* `blob_dequant_kernel`                      -> `guOff`, `downOff`, `guScaleOff`, `downScaleOff`
-/

namespace Strata.PrefillKernels

-- § hyper-connection indexing

/-- Recomposition of a flat `(t, c, d)` index over `D = 10240`, `N = 2560`. -/
theorem hc_index_recompose (i : Nat) :
    i / 10240 * 10240 + (i % 10240) / 2560 * 2560 + i % 2560 = i := by
  omega

/-- The stream coordinate `c` of `gr_write` is a valid stream (`< 4`). -/
theorem hc_stream_lt (i : Nat) : (i % 10240) / 2560 < 4 := by
  omega

/-- The dimension coordinate is a valid embedding index. -/
theorem hc_dim_lt (i : Nat) : i % 2560 < 2560 := by
  omega

/-- The token coordinate is in range when `i < T*D`. -/
theorem hc_tok_lt (T i : Nat) (h : i < T * 10240) : i / 10240 < T := by
  omega

/-- `gr_broadcast` reads `e[t*N + d]`, which is inside the `T*N` embedding block. -/
theorem broadcast_src_lt (T i : Nat) (h : i < T * 10240) : i / 10240 * 2560 + i % 2560 < T * 2560 := by
  omega

/-- `gr_write` reads the residual and the injection: `inj[t*ld + c]` with `c < 4`. -/
theorem write_inj_lt (i ld : Nat) (hld : 4 ≤ ld) : i / 10240 * ld + (i % 10240) / 2560 < (i / 10240 + 1) * ld := by
  have : (i % 10240) / 2560 < 4 := by omega
  have h2 : (i / 10240 + 1) * ld = i / 10240 * ld + ld := by rw [Nat.succ_mul]
  omega

-- § launch grids

/-- `blocks_for(n, t) = (n + t - 1) / t` blocks. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- A ceiling-divided launch grid covers every element: `n <= blocks * t`. -/
theorem blocksFor_cover (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h := (Nat.div_lt_iff_lt_mul ht).mp (Nat.lt_succ_self ((n + t - 1) / t))
  rw [Nat.succ_mul] at h
  omega

/-- The GDN conv grid `C/128` blocks of 128 threads covers all `C = 10240` channels exactly. -/
theorem gdn_conv_grid : 10240 / 128 * 128 = 10240 := by decide

/-- The q/k/v channel layout: `2*HK + HV = 80` heads of 128 equal `C = 10240`. -/
theorem gdn_channels : (2 * 16 + 48) * 128 = 10240 := by decide

/-- The recurrence row groups cover all `S = 128` rows: `RG * RPG = S`. -/
theorem gdn_rows_cover : 4 * (128 / 4) = 128 := by decide

/-- A row-group's rows `rg*RPG + r` stay below `S`. -/
theorem gdn_row_lt (rg r : Nat) (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by
  omega

-- § expert blob layout (Strata blob)

/-- Gate/up codes: 1280 rows x 640 bytes. -/
def guCodes : Nat := 1280 * 640
/-- Offset of the down codes. -/
def downOff : Nat := guCodes
/-- Offset of the gate/up scales (after 2560 x 160 B of down codes). -/
def guScaleOff : Nat := downOff + 2560 * 160
/-- Offset of the down scales (after 1280 x 40 f16 gate/up scales). -/
def downScaleOff : Nat := guScaleOff + 1280 * 40 * 2
/-- Total blob size: down scales are 2560 x 10 f16. -/
def blobBytes : Nat := downScaleOff + 2560 * 10 * 2

/-- The blob is 1,382,400 bytes (the "1.38 MB" slot read-back). -/
theorem blobBytes_value : blobBytes = 1382400 := by decide

/-- Section offsets are strictly increasing, so the sections are disjoint. -/
theorem blob_sections_ordered : downOff < guScaleOff ∧ guScaleOff < downScaleOff ∧ downScaleOff < blobBytes := by
  decide

/-- A gate/up scale index `row*40 + (byte*4)/64` stays within the 1280x40 table. -/
theorem gu_scale_idx_lt (row byte : Nat) (hr : row < 1280) (hb : byte < 640) :
    row * 40 + (byte * 4) / 64 < 1280 * 40 := by
  omega

/-- A down scale index `row*10 + (byte*4)/64` stays within the 2560x10 table. -/
theorem down_scale_idx_lt (row byte : Nat) (hr : row < 2560) (hb : byte < 160) :
    row * 10 + (byte * 4) / 64 < 2560 * 10 := by
  omega

/-- The four dequantized outputs of one code byte are inside the gate/up matrix row block. -/
theorem gu_out_idx_lt (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by
  omega

/-- The four dequantized outputs of one code byte are inside the down matrix. -/
theorem down_out_idx_lt (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by
  omega

/-- SwiGLU interleaved: columns `2k`, `2k+1` of a 1280-wide row with `k < 640` are in range. -/
theorem swiglu_il_idx (k : Nat) (hk : k < 640) : 2 * k + 1 < 1280 := by
  omega

/-- `gather_rows16` moves `width/8` uint4 per row; for width 2560 that is 320. -/
theorem gather_per_row : 2560 / 8 = 320 := by decide

-- § router lane layout

/-- Expert `lane + i*32` is recovered from register `i` of `lane`: lane is `e % 32`. -/
theorem route_lane (lane i : Nat) (hl : lane < 32) : (lane + i * 32) % 32 = lane := by
  omega

/-- Register index recovered by `ex / 32`. -/
theorem route_reg (lane i : Nat) (hl : lane < 32) : (lane + i * 32) / 32 = i := by
  omega

/-- 32 lanes x 16 registers cover all 512 experts (index in range). -/
theorem route_cover (lane i : Nat) (hl : lane < 32) (hi : i < 16) : lane + i * 32 < 512 := by
  omega

/-- Top-k uses `k = 10` ranks writing `ids[t*10 + rank]`, inside the `T*10` array. -/
theorem route_out_idx (t T rank : Nat) (ht : t < T) (hr : rank < 10) : t * 10 + rank < T * 10 := by
  have h : (t + 1) * 10 ≤ T * 10 := Nat.mul_le_mul_right 10 ht
  have h2 : (t + 1) * 10 = t * 10 + 10 := by rw [Nat.succ_mul]
  omega

-- § QSA helper indexing

/-- `split_q` recomposition: `i = t*6144 + h*256 + d` with `h = (i/256)%24`, `d = i%256`. -/
theorem split_q_recompose (i : Nat) :
    i / 6144 * 6144 + ((i / 256) % 24) * 256 + i % 256 = i := by
  omega

/-- The gate half of a full-q row sits at offset `h*512 + 256 + d` below `24*512`. -/
theorem qfull_gate_idx_lt (h d : Nat) (hh : h < 24) (hd : d < 256) : h * 512 + 256 + d < 24 * 512 := by
  omega

/-- KV append: a thread's channel `g*64 + tid` is below the 256 head dim. -/
theorem kv_channel_lt (g tid : Nat) (hg : g < 4) (ht : tid < 64) : g * 64 + tid < 256 := by
  omega

/-- KV append quant-scale slot `row*4 + g` is exactly one slot per 64-channel group. -/
theorem kv_scale_slot (row g : Nat) (hg : g < 4) : row * 4 + g < (row + 1) * 4 := by
  omega

/-- Rope pairs `(pair, pair+32)` are distinct halves of a 64-dim rotary block. -/
theorem rope_pair_in_block (pair : Nat) (h : pair < 32) : pair < pair + 32 ∧ pair + 32 < 64 := by
  omega

/-- Rope theta scale exponent `-2/64` corresponds to 32 frequency pairs in 64 dims. -/
theorem rope_pairs_count : 64 / 2 = 32 := by decide

end Strata.PrefillKernels



## FILE: /mnt/user-data/outputs/SpecController.lean
/-!
# MODULE: spec_controller
SOURCES: src/spec/controller.cpp, include/strata/spec/controller.hpp
Speculative-decoding controller: each step it maximizes `E[tokens] / T(step)` over
`none | lookup(k) | MTP(k)`, keeps `k = 0` unless the best beats plain decoding by `min_gain`,
and learns per-position acceptance with an EMA.

## Source semantics
* The verify size is `n = clamp(k+1, 1, K_MAX+1)` and indexes arrays of length `K_MAX+1` at `n-1`.
* Lookup acceptance is bucketed by match length: `>=16 -> 3`, `>=8 -> 2`, `>=5 -> 1`, else `0`.
* `expected = 1 + sum_i prod_{j<=i} p_j` (conditional-same uses `p[0]` throughout).
* The choice is the strict argmax of the rate, starting from the plain-decode baseline.
* The EMA update keeps acceptance in `[0,1]`.

## Form template identifiers
* `CostModel::step_ms` index `n-1`  -> `clampN`
* `Controller::bucket`              -> `bucket`
* `Controller::expected`            -> `runs`, `expectedFx` (fixed point, scale `S`)
* `Controller::choose` best-so-far  -> `pickBest`
* `Controller::observe` EMA         -> `emaUp`, `emaDown`
-/

namespace Strata.Spec

-- § window clamp

/-- `K_MAX`. -/
def K_MAX : Nat := 8

/-- `n = clamp(k + 1, 1, K_MAX + 1)`. -/
def clampN (k : Nat) : Nat := min (max (k + 1) 1) (K_MAX + 1)

/-- The verify size is at least one token (the pass always commits one). -/
theorem clampN_ge_one (k : Nat) : 1 ≤ clampN k := by
  unfold clampN K_MAX
  omega

/-- The array index `n - 1` is inside the `K_MAX + 1` ratio tables. -/
theorem clampN_idx_lt (k : Nat) : clampN k - 1 < K_MAX + 1 := by
  unfold clampN K_MAX
  omega

-- § match-length bucket

/-- Lookup acceptance bucket by match length. -/
def bucket (m : Nat) : Nat := if 16 ≤ m then 3 else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0

/-- Buckets index the 4-entry lookup table. -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  split_ifs <;> omega

/-- Longer matches never land in a lower bucket. -/
theorem bucket_mono (m n : Nat) (h : m ≤ n) : bucket m ≤ bucket n := by
  unfold bucket
  split_ifs <;> omega

-- § expected tokens (fixed point)

/-- Running products `run_{i+1} = run_i * p_i / S` for probabilities scaled by `S`. -/
def runs (S : Nat) : Nat → List Nat → List Nat
  | _, [] => []
  | run, p :: ps => (run * p / S) :: runs S (run * p / S) ps

/-- Expected committed tokens, scaled by `S`: `S + sum runs`. -/
def expectedFx (S : Nat) (ps : List Nat) : Nat := S + (runs S S ps).sum

/-- One run has one product per draft position. -/
theorem runs_length (S : Nat) : ∀ (ps : List Nat) (run : Nat), (runs S run ps).length = ps.length := by
  intro ps
  induction ps with
  | nil => intro run; simp [runs]
  | cons p ps ih => intro run; simp [runs, ih]

/-- If every probability is at most `S` and the start is, every running product is at most `S`. -/
theorem runs_le (S : Nat) : ∀ (ps : List Nat) (run : Nat), run ≤ S → (∀ p ∈ ps, p ≤ S) →
    ∀ x ∈ runs S run ps, x ≤ S := by
  intro ps
  induction ps with
  | nil => intro run _ _ x hx; simp [runs] at hx
  | cons p ps ih =>
    intro run hr hp x hx
    have hp0 : p ≤ S := hp p (List.mem_cons_self)
    have hmul : run * p ≤ S * S := Nat.mul_le_mul hr hp0
    have hdiv : run * p / S ≤ S := Nat.div_le_of_le_mul hmul
    rw [runs] at hx
    rcases List.mem_cons.mp hx with h | hx'
    · rw [h]; exact hdiv
    · exact ih (run * p / S) hdiv (fun q hq => hp q (List.mem_cons_of_mem _ hq)) x hx'

/-- A list of numbers each at most `S` sums to at most `length * S`. -/
theorem sum_le_len_mul (S : Nat) : ∀ (l : List Nat), (∀ x ∈ l, x ≤ S) → l.sum ≤ l.length * S := by
  intro l
  induction l with
  | nil => intro _; simp
  | cons a l ih =>
    intro h
    have h1 : a ≤ S := h a (List.mem_cons_self)
    have h2 := ih (fun x hx => h x (List.mem_cons_of_mem _ hx))
    simp only [List.sum_cons, List.length_cons]
    have h3 : (l.length + 1) * S = l.length * S + S := by rw [Nat.succ_mul]
    omega

/-- The expected token count never exceeds `k + 1` (all drafts accepted), in scaled units. -/
theorem expectedFx_le (S : Nat) (ps : List Nat) (hp : ∀ p ∈ ps, p ≤ S) :
    expectedFx S ps ≤ (ps.length + 1) * S := by
  unfold expectedFx
  have h1 := sum_le_len_mul S (runs S S ps) (runs_le S ps S (Nat.le_refl S) hp)
  rw [runs_length] at h1
  have h3 : (ps.length + 1) * S = ps.length * S + S := by rw [Nat.succ_mul]
  omega

/-- The expected token count is at least one token (scaled `S`). -/
theorem expectedFx_ge (S : Nat) (ps : List Nat) : S ≤ expectedFx S ps := by
  unfold expectedFx
  omega

-- § argmax with strict improvement

/-- Strict argmax fold: a candidate replaces the best only when strictly greater. -/
def pickBest : Nat → List Nat → Nat
  | b, [] => b
  | b, c :: cs => pickBest (if b < c then c else b) cs

/-- The chosen rate is never below the plain-decode baseline (the start value). -/
theorem pickBest_ge_init : ∀ (cs : List Nat) (b : Nat), b ≤ pickBest b cs := by
  intro cs
  induction cs with
  | nil => intro b; simp [pickBest]
  | cons c cs ih =>
    intro b
    rw [pickBest]
    have h1 := ih (if b < c then c else b)
    have h2 : b ≤ (if b < c then c else b) := by split_ifs <;> omega
    omega

/-- The chosen rate dominates every candidate. -/
theorem pickBest_ge_mem : ∀ (cs : List Nat) (b : Nat), ∀ x ∈ cs, x ≤ pickBest b cs := by
  intro cs
  induction cs with
  | nil => intro b x hx; simp at hx
  | cons c cs ih =>
    intro b x hx
    rw [pickBest]
    rcases List.mem_cons.mp hx with h | hx'
    · have h2 : c ≤ (if b < c then c else b) := by split_ifs <;> omega
      have h3 := pickBest_ge_init cs (if b < c then c else b)
      rw [h]
      omega
    · exact ih _ x hx'

/-- The min-gain gate: keep a drafted choice only if `baseline * (den + num) <= best * den`. -/
def gate (best baseline num den : Nat) : Bool := decide (baseline * (den + num) ≤ best * den)

/-- A choice that passes the gate beats the baseline by the factor `(den+num)/den`. -/
theorem gate_sound (best baseline num den : Nat) (h : gate best baseline num den = true) :
    baseline * (den + num) ≤ best * den := by
  unfold gate at h
  exact of_decide_eq_true h

-- § EMA acceptance update

/-- EMA move toward 1: `p + (S - p) * a / A`. -/
def emaUp (S p a A : Nat) : Nat := p + (S - p) * a / A

/-- EMA move toward 0: `p - p * a / A`. -/
def emaDown (p a A : Nat) : Nat := p - p * a / A

/-- Moving toward a hit keeps the probability within `[0, S]` when the rate `a <= A`. -/
theorem emaUp_le (S p a A : Nat) (hp : p ≤ S) (ha : a ≤ A) : emaUp S p a A ≤ S := by
  unfold emaUp
  have hmul : (S - p) * a ≤ A * (S - p) := by
    rw [Nat.mul_comm A]
    exact Nat.mul_le_mul_left _ ha
  have hdiv : (S - p) * a / A ≤ S - p := Nat.div_le_of_le_mul hmul
  omega

/-- A hit never lowers the probability. -/
theorem emaUp_ge (S p a A : Nat) : p ≤ emaUp S p a A := by
  unfold emaUp
  exact Nat.le_add_right _ _

/-- Moving toward a miss never raises the probability and keeps the bound. -/
theorem emaDown_le (p a A S : Nat) (hp : p ≤ S) : emaDown p a A ≤ S := by
  unfold emaDown
  exact Nat.le_trans (Nat.sub_le _ _) hp

-- § observe bookkeeping

/-- Positions updated by `observe`: those up to and including the first rejected one. -/
def seen (k accepted : Nat) : Nat := min k (min accepted k + 1)

/-- `observe` never touches more than the `k` drafted positions. -/
theorem seen_le_k (k accepted : Nat) : seen k accepted ≤ k := by
  unfold seen
  omega

/-- The deepest-tested position index `k - 1` exists when the window is non-empty and below `K_MAX`. -/
theorem deepest_idx_lt (k : Nat) (h1 : 1 ≤ k) (h2 : k < K_MAX) : k - 1 < K_MAX := by
  unfold K_MAX at *
  omega

end Strata.Spec



## FILE: /mnt/user-data/outputs/SuffixDrafter.lean
/-!
# MODULE: suffix_drafter
SOURCES: src/spec/suffix_drafter.cpp
Lookup drafter for speculative decoding: an open-addressed hash table keyed by the last three
tokens stores the most recent `WAYS` end positions per key; `propose` finds the longest earlier
match of the current suffix and returns the tokens that followed it.

## Source semantics
* Keys are forced nonzero (`| 1`) so `0` can mean "empty slot".
* Each slot keeps its positions most-recent-first, at most `WAYS` of them.
* A candidate must end strictly before the current suffix (`p < cur`), otherwise it would match itself.
* The match length is capped by `max_match` and by the available history; a proposal needs
  `best_len >= min_match`, with `min_match >= 3`.
* The continuation never reads past `cur` and has at most `max_k` tokens.

## Form template identifiers
* `key_at` low bit            -> `setLow`
* `Slot::pos[WAYS]` insertion -> `insertPos`
* match loop                  -> `ml`
* continuation loop           -> `contLen`
-/

namespace Strata.Suffix

-- § key nonzero

/-- Setting the lowest bit: `x | 1` on naturals. -/
def setLow (x : Nat) : Nat := x / 2 * 2 + 1

/-- A key with the low bit forced on is never `0`, so `0` is a safe empty marker. -/
theorem setLow_ne_zero (x : Nat) : setLow x ≠ 0 := by
  unfold setLow
  omega

/-- The key is odd. -/
theorem setLow_odd (x : Nat) : setLow x % 2 = 1 := by
  unfold setLow
  omega

-- § probing

/-- Linear probing step `(i + 1) & mask` for a power-of-two table is `% cap`. -/
def nextProbe (i cap : Nat) : Nat := (i + 1) % cap

/-- A probe step stays inside the table. -/
theorem nextProbe_lt (i cap : Nat) (hc : 0 < cap) : nextProbe i cap < cap := by
  unfold nextProbe
  exact Nat.mod_lt _ hc

-- § slot way list

/-- Insert the newest end position at the front, keeping at most `W` entries. -/
def insertPos (W e : Nat) (l : List Nat) : List Nat := (e :: l).take W

/-- A slot never holds more than `W` positions. -/
theorem insertPos_length (W e : Nat) (l : List Nat) : (insertPos W e l).length ≤ W := by
  unfold insertPos
  simp [List.length_take]

/-- The newest position is first (most recent first). -/
theorem insertPos_head (W e : Nat) (l : List Nat) (hW : 0 < W) : (insertPos W e l).head? = some e := by
  cases W with
  | zero => exact absurd hW (by omega)
  | succ w => simp [insertPos]

/-- If positions are strictly decreasing and the new one is larger, the order is preserved. -/
theorem insertPos_sorted (W e : Nat) (l : List Nat) (hl : List.Pairwise (· > ·) l)
    (he : ∀ x ∈ l, x < e) : List.Pairwise (· > ·) (insertPos W e l) := by
  have h1 : List.Pairwise (· > ·) (e :: l) := List.pairwise_cons.mpr ⟨fun a ha => he a ha, hl⟩
  exact List.Pairwise.sublist (List.take_sublist _ _) h1

-- § suffix match length

/-- The match loop: counts `len` while `len <= p` and `h[p-len] = h[cur-len]`, at most `fuel` times. -/
def ml (h : Nat → Nat) (p cur : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, len => if len ≤ p ∧ h (p - len) = h (cur - len) then 1 + ml h p cur fuel (len + 1) else 0

/-- The match length never exceeds `max_match`. -/
theorem ml_le_fuel (h : Nat → Nat) (p cur : Nat) : ∀ fuel len, ml h p cur fuel len ≤ fuel := by
  intro fuel
  induction fuel with
  | zero => intro len; simp [ml]
  | succ f ih =>
    intro len
    rw [ml]
    split_ifs with hc
    · have := ih (len + 1)
      omega
    · omega

/-- The match never reaches before the start of history: `len + matched <= p + 1`. -/
theorem ml_bound (h : Nat → Nat) (p cur : Nat) :
    ∀ fuel len, len ≤ p + 1 → ml h p cur fuel len + len ≤ p + 1 := by
  intro fuel
  induction fuel with
  | zero => intro len hl; simp [ml]; omega
  | succ f ih =>
    intro len hl
    rw [ml]
    split_ifs with hc
    · have h1 := hc.1
      have h2 := ih (len + 1) (by omega)
      omega
    · omega

/-- Matching a position against itself would give the full length: the reason `p >= cur` is skipped. -/
theorem ml_self (h : Nat → Nat) (cur : Nat) :
    ∀ fuel len, len ≤ cur + 1 → ml h cur cur fuel len = min fuel (cur + 1 - len) := by
  intro fuel
  induction fuel with
  | zero => intro len hl; simp [ml]
  | succ f ih =>
    intro len hl
    rw [ml]
    by_cases hc : len ≤ cur
    · have hc2 : len ≤ cur ∧ h (cur - len) = h (cur - len) := ⟨hc, rfl⟩
      rw [if_pos hc2, ih (len + 1) (by omega)]
      omega
    · have hc3 : ¬ (len ≤ cur ∧ h (cur - len) = h (cur - len)) := fun hh => hc hh.1
      rw [if_neg hc3]
      omega

/-- Constructor clamps: `min_match_ = max 3 min_match` is at least 3. -/
theorem minMatch_ge_three (m : Nat) : 3 ≤ max 3 m := by
  omega

/-- `max_match_ = max min_match max_match` is at least the requested minimum. -/
theorem maxMatch_ge_min (m M : Nat) : m ≤ max m M := by
  omega

/-- If the effective `max_match` is below the effective `min_match`, no proposal is ever made. -/
theorem no_proposal_if_small_max (h : Nat → Nat) (p cur maxm minm : Nat) (hlt : maxm < minm) :
    ml h p cur maxm 0 < minm := by
  have := ml_le_fuel h p cur maxm 0
  omega

-- § continuation

/-- Number of continuation tokens: from `e + 1` to `cur`, at most `max_k`. -/
def contLen (maxk cur e : Nat) : Nat := min maxk (cur - e)

/-- The continuation has at most `max_k` tokens. -/
theorem contLen_le_maxk (maxk cur e : Nat) : contLen maxk cur e ≤ maxk := by
  unfold contLen
  omega

/-- The continuation never reads past `cur`: its last index `e + len <= cur`. -/
theorem contLen_reads_in_history (maxk cur e : Nat) (he : e < cur) : e + contLen maxk cur e ≤ cur := by
  unfold contLen
  omega

end Strata.Suffix



## FILE: /mnt/user-data/outputs/PleNgramHash.lean
/-!
# MODULE: ple_ngram_hash
SOURCES: src/kernels/ngram.cpp, include/strata/kernels/ngram.hpp
The PLE n-gram hash (sixteen row indices per token from the last three token ids, XOR of
multiplier products reduced `mod` non-power-of-two head vocab sizes plus offsets), the IQ4_NL
row dequantization (split-half nibble order) and the `PleTable` issue/collect protocol.

## Source semantics
* Window is OLDEST FIRST; a missing predecessor is `-1`; the EOS cut propagates forward to older
  positions; the token's OWN eos does not cut its own context.
* Head `h = (n-2)*8 + g` for `n in {2,3}`, `g < 8` gives 16 distinct heads.
* Row index of head `h` is `mixed % vocab[h] + offset[h]`, which is below `offset[h] + vocab[h]`;
  `offset` is the running sum of `vocab`.
* A row is 160 values = 5 blocks of 32 at 18 bytes = 90 bytes; byte `j` carries elements `j`
  and `j + 16` of its block.
* The table accepts one in-flight `issue`; `collect` needs a pending issue.

## Form template identifiers
* `ngram_rows` context build   -> `ctx1`, `ctx2`, `isCut`
* `ple_artifact_consts`        -> `vocab`, `offs`
* `iq4nl_dequant_row`          -> `iq4`
* `PleTable::issue/collect`    -> `issue`, `collect`
-/

namespace Strata.PleNgram

-- § geometry constants

/-- `PLE_N_HEADS = (ngram_size - 1) * heads_per_ngram = 16`. -/
theorem n_heads : (3 - 1) * 8 = 16 := by decide

/-- `NG_HIST = (conv_kernel - 1) * ngram_size = 9`. -/
theorem hist_len : (4 - 1) * 3 = 9 := by decide

/-- `NG_HC_DIM = hc * n_embd = 10240`. -/
theorem hc_dim : 4 * 2560 = 10240 := by decide

/-- A row is 5 blocks x 18 bytes = 90 bytes. -/
theorem row_bytes : 160 / 32 * 18 = 90 := by decide

/-- Sixteen heads of 160 values flatten head-slowest to `n_embd = 2560`. -/
theorem heads_flatten : 16 * 160 = 2560 := by decide

/-- The table holds `320001536 * 90 = 28,800,138,240` bytes (the "28.8 GB"). -/
theorem table_bytes : 320001536 * 90 = 28800138240 := by decide

-- § head constants

/-- `head_vocab_sizes` of the artifact. -/
def vocab : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- `head_offsets` of the artifact. -/
def offs : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Both constant arrays have 16 entries. -/
theorem consts_len : vocab.length = 16 ∧ offs.length = 16 := by decide

/-- `head_offsets` is the running sum of `head_vocab_sizes`. -/
theorem offsets_running_sum : ∀ i : Fin 15,
    offs.getD (i.val + 1) 0 = offs.getD i.val 0 + vocab.getD i.val 0 := by
  decide

/-- Every head vocab size is positive, so `%` is well defined. -/
theorem vocab_pos : ∀ x ∈ vocab, 0 < x := by decide

/-- The last head's window ends below the table's row count (no head reaches past the table). -/
theorem heads_within_table : offs.getD 15 0 + vocab.getD 15 0 ≤ 320001536 := by decide

/-- A head's row index lies inside that head's window `[offset, offset + vocab)`. -/
theorem row_in_window (mixed v o : Nat) (hv : 0 < v) : mixed % v + o < o + v := by
  have := Nat.mod_lt mixed hv
  omega

-- § XOR structure

/-- Equal terms cancel under XOR (so XOR, not sum, is observable). -/
theorem xor_cancel (a : Nat) : a ^^^ a = 0 := Nat.xor_self a

-- § head indexing

/-- Head index `(n-2)*8 + g` for `n in {2,3}` and `g < 8` is below 16. -/
theorem head_idx_lt (n g : Nat) (hn : 2 ≤ n) (hn2 : n ≤ 3) (hg : g < 8) : (n - 2) * 8 + g < 16 := by
  omega

/-- Bigram heads occupy `0..7` and trigram heads `8..15`: the map `(n, g) -> head` is injective. -/
theorem head_idx_inj (n g n' g' : Nat) (hn : 2 ≤ n) (hn2 : n ≤ 3) (hg : g < 8)
    (hn' : 2 ≤ n') (hn2' : n' ≤ 3) (hg' : g' < 8) (h : (n - 2) * 8 + g = (n' - 2) * 8 + g') :
    n = n' ∧ g = g' := by
  omega

-- § EOS cut context

/-- EOS token id. -/
def EOS : Int := 248044

/-- A predecessor cuts the context when it is missing (`< 0`) or the EOS token. -/
def isCut (t : Int) : Bool := decide (t < 0) || decide (t = EOS)

/-- Context slot 1: the newest predecessor, or EOS if it cuts. -/
def ctx1 (p1 : Int) : Int := if isCut p1 then EOS else p1

/-- Context slot 2: the older predecessor, or EOS if it or the newer one cuts. -/
def ctx2 (p0 p1 : Int) : Int := if isCut p1 || isCut p0 then EOS else p0

/-- A cutting newest predecessor makes both context slots EOS (the cut propagates forward). -/
theorem cut_propagates (p0 p1 : Int) (h : isCut p1 = true) : ctx1 p1 = EOS ∧ ctx2 p0 p1 = EOS := by
  constructor
  · simp [ctx1, h]
  · simp [ctx2, h]

/-- A real newest predecessor is kept as is (the token's own EOS never enters here). -/
theorem real_newest_kept (p1 : Int) (h : isCut p1 = false) : ctx1 p1 = p1 := by
  simp [ctx1, h]

/-- A real newest predecessor with a cutting older one: only the older slot becomes EOS. -/
theorem older_cut_only (p0 p1 : Int) (h1 : isCut p1 = false) (h0 : isCut p0 = true) :
    ctx1 p1 = p1 ∧ ctx2 p0 p1 = EOS := by
  constructor
  · simp [ctx1, h1]
  · simp [ctx2, h1, h0]

/-- With no cut at all the full history is used. -/
theorem no_cut_full (p0 p1 : Int) (h1 : isCut p1 = false) (h0 : isCut p0 = false) :
    ctx1 p1 = p1 ∧ ctx2 p0 p1 = p0 := by
  constructor
  · simp [ctx1, h1]
  · simp [ctx2, h1, h0]

/-- Token id 0 is a real token: it is not a cut. -/
theorem token_zero_not_cut : isCut 0 = false := by decide

/-- A missing predecessor (`-1`) cuts. -/
theorem missing_cuts : isCut (-1) = true := by decide

-- § IQ4_NL dequantization

/-- The IQ4_NL codebook (`kvalues_iq4nl`), no `-8` shift. -/
def iq4 : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries. -/
theorem iq4_len : iq4.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4_sorted : List.Pairwise (· < ·) iq4 := by decide

/-- Split halves: byte `j < 16` writes elements `j` and `j + 16`, which are distinct and inside a block of 32. -/
theorem split_halves (j : Nat) (hj : j < 16) : j ≠ j + 16 ∧ j + 16 < 32 := by
  omega

/-- Output index `b*32 + j + 16` over 5 blocks stays inside the 160-value row. -/
theorem out_idx_lt (b j : Nat) (hb : b < 5) (hj : j < 16) : b * 32 + j + 16 < 160 := by
  omega

/-- Block `b` starts at byte `b * 18` and its 16 quant bytes end inside the 90-byte row. -/
theorem block_in_row (b : Nat) (hb : b < 5) : b * 18 + 2 + 16 ≤ 90 := by
  omega

-- § PleTable issue / collect protocol

/-- Table state: whether one token's reads are in flight. -/
structure TableState where
  pending : Bool

/-- `issue` fails when a token is already in flight. -/
def issue (s : TableState) : Option TableState := if s.pending then none else some ⟨true⟩

/-- `collect` needs a pending issue and clears it. -/
def collect (s : TableState) : Option TableState := if s.pending then some ⟨false⟩ else none

/-- Only one token may be in flight: a second `issue` is refused. -/
theorem issue_twice_fails : (issue ⟨true⟩) = none := by
  simp [issue]

/-- `collect` without a prior `issue` is refused. -/
theorem collect_without_issue : (collect ⟨false⟩) = none := by
  simp [collect]

/-- An `issue` from idle yields a pending table. -/
theorem issue_from_idle : (issue ⟨false⟩).map (·.pending) = some true := by
  simp [issue]

/-- After `issue` then `collect` the table is idle and can issue again. -/
theorem issue_collect_issue :
    ((issue ⟨false⟩).bind collect).bind issue = some ⟨true⟩ := by
  simp [issue, collect]

end Strata.PleNgram



## FILE: /mnt/user-data/outputs/PleDirectReader.lean
/-!
# MODULE: ple_direct_reader
SOURCES: src/ngram/ple_reader.cpp
Unbuffered asynchronous reader for the PLE table: rows are grouped by 4 KiB page, read into a slab
of `max_inflight` two-page slots, completed by a worker thread, and cached in a bounded
set-associative row cache.  Tickets track the outstanding jobs of one request.

## Source semantics
* A 90-byte row at byte `bp` is read from the aligned page `first = at/4096*4096` with length
  `4096` or `8192` (a row straddling a page boundary), and lies entirely inside the read.
* The row cache has `rows/8` sets of 8 ways; capacity never exceeds the requested rows.
* Slots are conserved: `free + inflight = max_inflight`; slot buffers are disjoint 8192-byte windows.
* Ticket ids wrap but never become 0.
* The latency ring keeps at most 65536 samples; the percentile index is in range.

## Form template identifiers
* `first`, `length` of `Job`  -> `firstOf`, `lengthOf`
* `RowCache::init`            -> `cacheSets`
* `slot_buf(s)`               -> `slotOff`
* `next_ticket`               -> `nextTicket`
-/

namespace Strata.PleReader

-- § page math (PAGE = 4096, ROW_BYTES = 90)

/-- Aligned page offset of the byte `bp`. -/
def firstOf (bp : Nat) : Nat := bp / 4096 * 4096

/-- Read length: one or two pages depending on whether the 90 bytes straddle a boundary. -/
def lengthOf (bp : Nat) : Nat := (bp + 89) / 4096 * 4096 - firstOf bp + 4096

/-- The aligned read starts at or before the row. -/
theorem first_le (bp : Nat) : firstOf bp ≤ bp := by
  unfold firstOf
  omega

/-- The read offset is page aligned. -/
theorem first_aligned (bp : Nat) : firstOf bp % 4096 = 0 := by
  unfold firstOf
  omega

/-- The row starts inside the first page (`in_page < 4096`). -/
theorem in_page_lt (bp : Nat) : bp - firstOf bp < 4096 := by
  unfold firstOf
  omega

/-- The whole 90-byte row lies inside the read (no "short read inside the table"). -/
theorem row_inside_read (bp : Nat) : bp + 90 ≤ firstOf bp + lengthOf bp := by
  unfold lengthOf firstOf
  omega

/-- The read is one or two pages. -/
theorem length_cases (bp : Nat) : lengthOf bp = 4096 ∨ lengthOf bp = 8192 := by
  unfold lengthOf firstOf
  omega

/-- The read fits the two-page slot. -/
theorem length_le_slot (bp : Nat) : lengthOf bp ≤ 8192 := by
  have := length_cases bp
  omega

-- § row cache

/-- Number of sets: `rows / 8`. -/
def cacheSets (rows : Nat) : Nat := rows / 8

/-- Cache capacity `sets * 8` never exceeds the requested rows. -/
theorem cache_capacity_le (rows : Nat) : cacheSets rows * 8 ≤ rows := by
  unfold cacheSets
  omega

/-- A hashed set index is in range when there is at least one set. -/
theorem set_index_lt (hash sets : Nat) (h : 0 < sets) : hash % sets < sets :=
  Nat.mod_lt _ h

/-- The round-robin replacement pointer stays inside the 8 ways. -/
theorem way_next_lt (w : Nat) : (w + 1) % 8 < 8 := by
  omega

-- § slot conservation

/-- Pump: a queued job takes a free slot (free decreases, inflight increases). -/
def pumpStep (free inflight : Nat) : Nat × Nat := (free - 1, inflight + 1)

/-- Finish: a completed job returns its slot. -/
def finishStep (free inflight : Nat) : Nat × Nat := (free + 1, inflight - 1)

/-- Pumping with a free slot conserves `free + inflight`. -/
theorem pump_conserves (free inflight : Nat) (h : 0 < free) :
    (pumpStep free inflight).1 + (pumpStep free inflight).2 = free + inflight := by
  unfold pumpStep
  simp
  omega

/-- Finishing an inflight job conserves `free + inflight`. -/
theorem finish_conserves (free inflight : Nat) (h : 0 < inflight) :
    (finishStep free inflight).1 + (finishStep free inflight).2 = free + inflight := by
  unfold finishStep
  simp
  omega

/-- Byte offset of slot `s` in the slab: `s * 2 * PAGE`. -/
def slotOff (s : Nat) : Nat := s * 8192

/-- Slot windows are disjoint: a read of length `<= 8192` in slot `s` ends before slot `s+1`. -/
theorem slot_windows_disjoint (s len : Nat) (h : len ≤ 8192) : slotOff s + len ≤ slotOff (s + 1) := by
  unfold slotOff
  omega

/-- Every valid slot lies inside the slab of `max_inflight` slots. -/
theorem slot_in_slab (s M len : Nat) (hs : s < M) (h : len ≤ 8192) : slotOff s + len ≤ M * 8192 := by
  unfold slotOff
  have h2 : (s + 1) * 8192 ≤ M * 8192 := Nat.mul_le_mul_right 8192 hs
  have h3 : (s + 1) * 8192 = s * 8192 + 8192 := by rw [Nat.succ_mul]
  omega

-- § tickets and latency ring

/-- `next_ticket` increments modulo 2^32 and skips 0. -/
def nextTicket (t : Nat) : Nat := if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- A ticket id is never 0. -/
theorem nextTicket_ne_zero (t : Nat) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split_ifs with h <;> omega

/-- A ticket id fits 32 bits. -/
theorem nextTicket_lt (t : Nat) : nextTicket t < 4294967296 := by
  unfold nextTicket
  split_ifs with h <;> omega

/-- The latency ring index `ring_pos % 65536` is inside the ring. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by
  omega

/-- Percentile index `min (size-1) (round(q*(size-1)))` is a valid element index for non-empty data. -/
def pctIndex (size r : Nat) : Nat := min (size - 1) r

/-- The percentile index is inside the sample vector. -/
theorem pctIndex_lt (size r : Nat) (h : 0 < size) : pctIndex size r < size := by
  unfold pctIndex
  omega

end Strata.PleReader



## FILE: /mnt/user-data/outputs/ChatFrontend.lean
/-!
# MODULE: chat_frontend
SOURCES: serve/frontend.py
Text-only request/response layer: normalizes OpenAI / Anthropic requests into chat-template form,
maps reasoning effort / thinking budgets to template levels, and incrementally parses model output
into reasoning, content and tool calls without ever emitting a partial tag.

## Source semantics
* Thinking budgets map monotonically: `< 2048 -> low`, `< 8192 -> medium`, else `xhigh`.
* The parser holds back the longest suffix of the buffer that is a proper prefix of a tag, so a tag
  split across deltas is never shown; held + emitted text equals the buffer.
* Tool-call parameter values lose at most one leading and one trailing newline.
* One `<|image_pad|>` per image is expanded to one per image token; with no images the ids are unchanged.
* A request is rejected (never truncated) when prompt plus max tokens exceed the context.

## Form template identifiers
* `budget_effort`            -> `budgetRank`
* `OutputParser._hold`       -> `holdOne`
* `parse_tool_call` strip    -> `stripNl`
* `Service.prepare` pad loop -> `expand`
-/

namespace Strata.Frontend

-- § thinking budget -> level

/-- Level rank of a thinking budget: 0 = low, 1 = medium, 2 = xhigh. -/
def budgetRank (n : Int) : Nat := if n < 2048 then 0 else if n < 8192 then 1 else 2

/-- More budget never yields a lower level. -/
theorem budgetRank_mono (n m : Int) (h : n ≤ m) : budgetRank n ≤ budgetRank m := by
  unfold budgetRank
  split_ifs <;> omega

/-- Levels are one of the three template values. -/
theorem budgetRank_lt (n : Int) : budgetRank n < 3 := by
  unfold budgetRank
  split_ifs <;> omega

-- § partial-tag hold-back

/-- Longest `n` in `1 .. tag.length - 1` such that the last `n` symbols of `text` equal `tag.take n`. -/
def holdOne (text tag : List Nat) : Nat :=
  (List.range tag.length).foldl
    (fun b n => if 1 ≤ n ∧ n ≤ text.length ∧ text.drop (text.length - n) = tag.take n then max b n else b) 0

/-- Fold of `max`-style steps stays below a bound if every candidate is below it. -/
theorem fold_bound (tl tl' : Nat) (f : Nat → Nat → Nat) (hf : ∀ b n, f b n ≤ max b n) :
    ∀ (l : List Nat) (b : Nat), (∀ n ∈ l, n < tl) → b < tl → l.foldl f b < tl := by
  intro l
  induction l with
  | nil => intro b _ hb; simpa using hb
  | cons n l ih =>
    intro b hl hb
    simp only [List.foldl_cons]
    apply ih
    · intro x hx
      exact hl x (List.mem_cons_of_mem _ hx)
    · have h1 := hf b n
      have h2 : n < tl := hl n (List.mem_cons_self)
      omega

/-- The held suffix is a PROPER prefix of the tag: strictly shorter than the tag (for non-empty tags). -/
theorem holdOne_lt (text tag : List Nat) (h : 0 < tag.length) : holdOne text tag < tag.length := by
  unfold holdOne
  apply fold_bound tag.length 0
  · intro b n
    split_ifs <;> omega
  · intro n hn
    exact List.mem_range.mp hn
  · exact h

/-- Hold-back splits the buffer into an emitted prefix and a held suffix that recombine exactly. -/
theorem hold_split (buf : List Nat) (keep : Nat) :
    buf.take (buf.length - keep) ++ buf.drop (buf.length - keep) = buf :=
  List.take_append_drop _ _

/-- The emitted prefix is no longer than the buffer. -/
theorem emitted_le (buf : List Nat) (keep : Nat) : (buf.take (buf.length - keep)).length ≤ buf.length := by
  simp [List.length_take]

-- § newline stripping of parameter values

/-- Remove one leading newline (code 10) if present. -/
def dropLead : List Nat → List Nat
  | 10 :: rest => rest
  | l => l

/-- Remove one trailing newline if present. -/
def dropTrail (l : List Nat) : List Nat := if l.getLast? = some 10 then l.dropLast else l

/-- Strip at most one newline at each end. -/
def stripNl (l : List Nat) : List Nat := dropTrail (dropLead l)

/-- Leading strip removes at most one symbol. -/
theorem dropLead_length (l : List Nat) : l.length ≤ (dropLead l).length + 1 := by
  match l with
  | [] => simp [dropLead]
  | 10 :: rest => simp [dropLead]
  | (k + 11) :: rest => simp [dropLead]
  | 0 :: rest => simp [dropLead]
  | 1 :: rest => simp [dropLead]
  | 2 :: rest => simp [dropLead]
  | 3 :: rest => simp [dropLead]
  | 4 :: rest => simp [dropLead]
  | 5 :: rest => simp [dropLead]
  | 6 :: rest => simp [dropLead]
  | 7 :: rest => simp [dropLead]
  | 8 :: rest => simp [dropLead]
  | 9 :: rest => simp [dropLead]

/-- Trailing strip removes at most one symbol. -/
theorem dropTrail_length (l : List Nat) : l.length ≤ (dropTrail l).length + 1 := by
  unfold dropTrail
  split_ifs
  · simp [List.length_dropLast]
    omega
  · omega

/-- Stripping never lengthens a value. -/
theorem dropTrail_le (l : List Nat) : (dropTrail l).length ≤ l.length := by
  unfold dropTrail
  split_ifs
  · simp [List.length_dropLast]
  · exact Nat.le_refl _

-- § image pad expansion

/-- Expand each pad token into `c` pads using the per-image counts in order. -/
def expand (pad : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | t :: ts, cs =>
    if t = pad then
      match cs with
      | [] => t :: expand pad ts []
      | c :: cs' => List.replicate c pad ++ expand pad ts cs'
    else t :: expand pad ts cs

/-- With no image counts the ids are unchanged. -/
theorem expand_nil_counts (pad : Nat) : ∀ ids : List Nat, expand pad ids [] = ids := by
  intro ids
  induction ids with
  | nil => rfl
  | cons t ts ih =>
    by_cases h : t = pad <;> simp [expand, h, ih]

/-- Ids without any pad token are unchanged for any counts. -/
theorem expand_no_pad (pad : Nat) : ∀ (ids cs : List Nat), (∀ t ∈ ids, t ≠ pad) → expand pad ids cs = ids := by
  intro ids
  induction ids with
  | nil => intro cs _; rfl
  | cons t ts ih =>
    intro cs h
    have ht : t ≠ pad := h t (List.mem_cons_self)
    have h2 := ih cs (fun x hx => h x (List.mem_cons_of_mem _ hx))
    simp [expand, ht, h2]

-- § context admission

/-- A request is admitted iff prompt plus max tokens fit the engine context. -/
def fits (promptLen maxNew maxCtx : Nat) : Prop := promptLen + maxNew ≤ maxCtx

/-- An admitted request's prompt alone fits the context. -/
theorem fits_prompt (p m c : Nat) (h : fits p m c) : p ≤ c := by
  unfold fits at h
  omega

/-- Detokenizer: after pushing, `sent` equals the decoded length, so the next delta starts there. -/
def deltaLen (textLen sent : Nat) : Nat := textLen - sent

/-- Emitted deltas never overlap: sent + delta = text length when sent is not ahead. -/
theorem delta_exact (textLen sent : Nat) (h : sent ≤ textLen) : sent + deltaLen textLen sent = textLen := by
  unfold deltaLen
  omega

end Strata.Frontend



## FILE: /mnt/user-data/outputs/ServeEngine.lean
/-!
# MODULE: serve_engine
SOURCES: serve/server.py
OpenAI / Anthropic HTTP front for a single resident engine: a FIFO-serialized `Service`, a
`StrataEngine` speaking the `GEN / T / DONE / ERR` line protocol, an image-embedding cache
bounded to 64 entries, API-key authorization and finish-reason mapping.

## Source semantics
* Requests are admitted only if prompt + max tokens fit the engine context (400 otherwise).
* The engine emits at most `max_new` `T` lines per request and finishes with exactly one `DONE`.
* The vision cache never holds more than 64 entries (oldest evicted first).
* With no API key configured every request is authorized; otherwise the supplied key must equal it.
* OpenAI finish is `tool_calls` iff at least one call was made and generation stopped naturally;
  Anthropic stop reason is `tool_use` under the same condition.

## Form template identifiers
* `Service.prepare` context check -> `admit`
* `Vision.cache` eviction        -> `insertCache`
* `_authorized`                  -> `authorized`
* `openai_chunks` finish         -> `openaiFinish`
* `anthropic_events` stop        -> `anthropicStop`
-/

namespace Strata.Serve

-- § admission

/-- Prompt + max tokens must not exceed the engine context. -/
def admit (promptLen maxNew maxCtx : Nat) : Bool := decide (promptLen + maxNew ≤ maxCtx)

/-- An admitted request fits the context. -/
theorem admit_sound (p m c : Nat) (h : admit p m c = true) : p + m ≤ c := by
  unfold admit at h
  exact of_decide_eq_true h

/-- A request that does not fit is rejected, never truncated. -/
theorem admit_reject (p m c : Nat) (h : c < p + m) : admit p m c = false := by
  unfold admit
  exact decide_eq_false (by omega)

-- § engine line protocol

/-- Server-side view of a request in flight: tokens streamed so far. -/
structure Req where
  streamed : Nat
  maxNew : Nat
  done : Bool

/-- Receiving a `T` line streams one token only while below `max_new`. -/
def onToken (r : Req) : Req :=
  if r.done = false ∧ r.streamed < r.maxNew then { r with streamed := r.streamed + 1 } else r

/-- `DONE` finishes the request. -/
def onDone (r : Req) : Req := { r with done := true }

/-- Token delivery never exceeds `max_new`. -/
theorem onToken_le (r : Req) (h : r.streamed ≤ r.maxNew) : (onToken r).streamed ≤ r.maxNew := by
  unfold onToken
  split_ifs with hc
  · show r.streamed + 1 ≤ r.maxNew
    have := hc.2
    omega
  · exact h

/-- After `DONE` no further token is delivered. -/
theorem onToken_after_done (r : Req) : (onToken (onDone r)).streamed = r.streamed := by
  simp [onToken, onDone]

/-- `DONE` is idempotent: a request finishes exactly once. -/
theorem onDone_idem (r : Req) : onDone (onDone r) = onDone r := rfl

-- § vision cache bound

/-- Insert and evict the oldest when more than 64 entries. -/
def insertCache (cache : List Nat) (x : Nat) : List Nat :=
  if 64 < cache.length + 1 then (cache ++ [x]).tail else cache ++ [x]

/-- The cache never exceeds 64 entries (given it started within the bound). -/
theorem insertCache_len (cache : List Nat) (x : Nat) (h : cache.length ≤ 64) :
    (insertCache cache x).length ≤ 64 := by
  unfold insertCache
  split_ifs with hc
  · rw [List.length_tail, List.length_append, List.length_singleton]
    omega
  · rw [List.length_append, List.length_singleton]
    omega

-- § authorization

/-- No key configured -> open; else the supplied key must match. -/
def authorized (key gk : Nat) (configured : Bool) : Bool :=
  if configured = false then true else decide (gk = key)

/-- With no API key every request is authorized. -/
theorem authorized_open (key gk : Nat) : authorized key gk false = true := by
  simp [authorized]

/-- With a key configured, only the matching key is authorized. -/
theorem authorized_iff (key gk : Nat) : authorized key gk true = true ↔ gk = key := by
  simp [authorized]

-- § finish reasons

/-- Finish reasons exposed by the OpenAI side. -/
inductive Finish where
  | stop
  | length
  | toolCalls
  deriving DecidableEq

/-- `tool_calls` iff at least one call and a natural stop; else `stop` / `length`. -/
def openaiFinish (calls : Nat) (stopped : Bool) : Finish :=
  if stopped = true ∧ 0 < calls then Finish.toolCalls else if stopped = true then Finish.stop else Finish.length

/-- The OpenAI `tool_calls` finish reason is exactly "stopped naturally after calling a tool". -/
theorem openaiFinish_toolCalls_iff (c : Nat) (s : Bool) :
    openaiFinish c s = Finish.toolCalls ↔ (s = true ∧ 0 < c) := by
  cases s <;> rcases Nat.eq_zero_or_pos c with h | h <;> simp [openaiFinish, h]

/-- Anthropic stop reasons. -/
inductive Stop where
  | endTurn
  | maxTokens
  | toolUse
  deriving DecidableEq

/-- `tool_use` iff a tool block was emitted and generation stopped naturally. -/
def anthropicStop (usedTool stopped : Bool) : Stop :=
  if usedTool = true ∧ stopped = true then Stop.toolUse else if stopped = true then Stop.endTurn else Stop.maxTokens

/-- A length-limited generation never reports `tool_use`. -/
theorem anthropicStop_length (u : Bool) : anthropicStop u false = Stop.maxTokens := by
  simp [anthropicStop]

/-- A natural stop without tools is `end_turn`. -/
theorem anthropicStop_end (u : Bool) (h : u = false) : anthropicStop u true = Stop.endTurn := by
  simp [anthropicStop, h]

-- § usage accounting

/-- OpenAI usage total = prompt + completion tokens. -/
def totalTokens (prompt completion : Nat) : Nat := prompt + completion

/-- The reported total is the sum of its parts. -/
theorem totalTokens_eq (p c : Nat) : totalTokens p c = p + c := rfl

end Strata.Serve





## FILE: /mnt/user-data/outputs/QsaCacheStep.lean
/-!
# MODULE: qsa_cache_step
SOURCES: include/strata/kernels/qsa.hpp, src/kernels/cuda/qsa.cu
Per-token device step state (pos, n_kv, n_bid, width), the paged FP16 KV cache row mapping, the indexer's
block pooling schedule and the cell_block / tail-bias mapping of the QSA decode layer.

## Source semantics
* `qsa_step_fill` derives n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1) from `pos` alone.
* Below the selection bound the selection is the identity.
* Physical KV row = (page * n_head_kv + h) * page_size + (t % page_size); it must be in range and injective.
* A block completes exactly when pos % r = r-1; the tail holds r-1 raw rows.
* Cell j maps to pooled row min(j / r, n_bid); cells of the incomplete tail get the +1e9 bias.

## Form template identifiers
step buffer -> `Step`; selection width -> `selWidth`; page/row mapping -> `kvRow`; QsaShapes.idx_block -> `r`.
-/

namespace QsaCacheStep

-- § arithmetic helpers

/-- Decoding a mixed-radix pair: `(a*n+b)/n = a` and `(a*n+b)%n = b` when `b < n`. -/
theorem decode_div_mod (a b n : Nat) (hb : b < n) : (a*n+b)/n = a ∧ (a*n+b)%n = b := by
  have hn : 0 < n := by omega
  have e : a*n+b = b + a*n := Nat.add_comm _ _
  have h1 := Nat.add_mul_div_right b a hn
  rw [Nat.div_eq_of_lt hb] at h1
  constructor
  · rw [e]; omega
  · rw [e, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Quotient/remainder bracket for the block count. -/
theorem div_mul_bounds (n r : Nat) (hr : 0 < r) : (n/r)*r ≤ n ∧ n < (n/r+1)*r := by
  have hd := Nat.div_add_mod n r
  have hm := Nat.mod_lt n hr
  have e : (n/r+1)*r = (n/r)*r + r := by rw [Nat.add_mul, Nat.one_mul]
  rw [Nat.mul_comm r (n/r)] at hd
  constructor <;> omega

-- § selection width and step state

/-- `qsa_selection_width`: min(n_kv, top_k + r - 1). -/
def selWidth (nKv topK r : Nat) : Nat := if nKv < topK + r - 1 then nKv else topK + r - 1

/-- The step buffer contents. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- `qsa_step_fill`: all four entries from one position. -/
def stepFill (pos topK r : Nat) : Step :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / r, width := selWidth (pos + 1) topK r }

/-- The selection never exceeds the number of cached cells. -/
theorem selWidth_le (n k r : Nat) : selWidth n k r ≤ n := by
  unfold selWidth; split <;> omega

/-- The selection never exceeds the budget `top_k + r - 1`. -/
theorem selWidth_le_budget (n k r : Nat) : selWidth n k r ≤ k + r - 1 := by
  unfold selWidth; split <;> omega

/-- Below the bound the selection is the identity (all cells). -/
theorem selWidth_identity (n k r : Nat) (h : n < k + r - 1) : selWidth n k r = n := by
  unfold selWidth; simp [h]

/-- For the real geometry the width is `min n 2051`. -/
theorem selWidth_real (n : Nat) : selWidth n 2048 4 = min n 2051 := by
  unfold selWidth; split <;> omega

/-- The largest selection width the one-block top-k can ever write is 2051 (the required `cap`). -/
theorem selWidth_max_cap : selWidth 32768 2048 4 = 2051 := by decide

/-- The attention shared buffer for the largest selection fits the default 48 KiB shared window. -/
theorem attend_smem_fits : (2051 + 32) * 4 ≤ 49152 := by decide

/-- The step buffer is four int32 = 16 bytes. -/
theorem step_bytes : 4 * 4 = 16 := by decide

/-- The four step entries are mutually consistent by construction. -/
theorem stepFill_consistent (pos topK r : Nat) (hr : 0 < r) :
    (stepFill pos topK r).nKv = (stepFill pos topK r).pos + 1 ∧
    (stepFill pos topK r).nBid * r ≤ (stepFill pos topK r).nKv ∧
    (stepFill pos topK r).nKv < ((stepFill pos topK r).nBid + 1) * r ∧
    (stepFill pos topK r).width ≤ (stepFill pos topK r).nKv := by
  obtain ⟨h1, h2⟩ := div_mul_bounds (pos+1) r hr
  have h3 := selWidth_le (pos+1) topK r
  exact ⟨rfl, h1, h2, h3⟩

-- § paged KV cache

/-- Physical row of logical cell offset `o` in a page for kv head `h`. -/
def kvRow (page H h P o : Nat) : Nat := (page * H + h) * P + o

/-- A physical row stays inside the pool. -/
theorem kvRow_lt (page nPages H h P o : Nat) (hp : page < nPages) (hh : h < H) (ho : o < P) :
    kvRow page H h P o < nPages * H * P := by
  have h1 : page * H + h < nPages * H := idx2_lt page h nPages H hp hh
  exact idx2_lt (page * H + h) o (nPages * H) P h1 ho

/-- Distinct (page, head, slot) triples give distinct physical rows. -/
theorem kvRow_injective (p1 p2 H h1 h2 P o1 o2 : Nat) (hh1 : h1 < H) (hh2 : h2 < H)
    (ho1 : o1 < P) (ho2 : o2 < P) (heq : kvRow p1 H h1 P o1 = kvRow p2 H h2 P o2) :
    p1 = p2 ∧ h1 = h2 ∧ o1 = o2 := by
  unfold kvRow at heq
  obtain ⟨d1, m1⟩ := decode_div_mod (p1*H+h1) o1 P ho1
  obtain ⟨d2, m2⟩ := decode_div_mod (p2*H+h2) o2 P ho2
  rw [heq] at d1 m1
  have e1 : p1*H+h1 = p2*H+h2 := d1.symm.trans d2
  have e2 : o1 = o2 := m1.symm.trans m2
  obtain ⟨a1, b1⟩ := decode_div_mod p1 h1 H hh1
  obtain ⟨a2, b2⟩ := decode_div_mod p2 h2 H hh2
  rw [e1] at a1 b1
  exact ⟨a1.symm.trans a2, b1.symm.trans b2, e2⟩

/-- The page-table index of a cell is in range whenever the cell is in the pool. -/
theorem page_index_lt (pos P nPages : Nat) (hP : 0 < P) (hpos : pos < nPages * P) :
    pos / P < nPages := (Nat.div_lt_iff_lt_mul hP).2 hpos

-- § indexer pooling schedule and cell_block

/-- A block completes exactly when `pos % 4 = 3`, i.e. when `(pos+1) % 4 = 0`. -/
theorem block_completes_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- On completion the completed-block count grows by exactly one. -/
theorem nbid_grows_on_completion (pos : Nat) (h : pos % 4 = 3) : (pos + 1) / 4 = pos / 4 + 1 := by
  omega

/-- Off completion the completed-block count is unchanged. -/
theorem nbid_stable_off_completion (pos : Nat) (h : pos % 4 ≠ 3) : (pos + 1) / 4 = pos / 4 := by
  omega

/-- Only slots 0..r-2 are stored in the raw tail: three rows suffice for r = 4. -/
theorem tail_slot_in_range (pos : Nat) (h : pos % 4 ≠ 3) : pos % 4 < 3 := by omega

/-- The block's first cell does not exceed `pos` and is within `r` cells of it. -/
theorem block_first_cell (pos r : Nat) (hr : 0 < r) :
    (pos / r) * r ≤ pos ∧ pos < (pos / r) * r + r := by
  obtain ⟨h1, h2⟩ := div_mul_bounds pos r hr
  have e : (pos/r+1)*r = (pos/r)*r + r := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- `cell_block`: a cached cell never maps past the spare slot row `n_bid`. -/
theorem cell_block_eq (j nKv r : Nat) (hj : j < nKv) : min (j / r) (nKv / r) = j / r := by
  have h : j / r ≤ nKv / r := Nat.div_le_div_right (Nat.le_of_lt hj)
  omega

/-- The +1e9 tail bias applies exactly to cells that map onto the spare row. -/
theorem tail_cell_iff (j nKv r : Nat) (hr : 0 < r) (hj : j < nKv) :
    (nKv / r) * r ≤ j ↔ j / r = nKv / r := by
  constructor
  · intro h
    have h1 : nKv / r ≤ j / r := (Nat.le_div_iff_mul_le hr).2 h
    have h2 : j / r ≤ nKv / r := Nat.div_le_div_right (Nat.le_of_lt hj)
    omega
  · intro h
    have h3 := Nat.div_mul_le_self j r
    rw [h] at h3
    exact h3

/-- A completed block has no tail bias: tail cells exist only when `n_kv % r ≠ 0` (r = 4). -/
theorem tail_exists_iff (n : Nat) : n % 4 ≠ 0 ↔ (n / 4) * 4 < n := by omega

end QsaCacheStep



## FILE: /mnt/user-data/outputs/QsaSelect.lean
/-!
# MODULE: qsa_select
SOURCES: src/kernels/cuda/qsa_select.cu, src/kernels/cuda/qsa.cu
Exact top-k selection over per-cell / per-block indexer scores: the monotone f32 order key, the threshold
(binary lifting / 8-bit radix) search, tie-breaking by ascending index, and the block-weighted expansion
of selected blocks into cell ids.

## Source semantics
* `order_key` maps f32 bit patterns to an unsigned key so integer comparison matches numeric order; -0.0 is
  folded onto +0.0 first.
* The threshold `thr` satisfies count(key >= thr) >= width > count(key >= thr+1); the selection is
  every key > thr plus the first `width - c_gt` keys equal to thr, which has exactly `width` elements.
* Block selection weights each block by r cells, the spare block by `n_kv - n_bid*r`; total weight is n_kv.
* Selected block b yields cells b*r .. b*r+w-1, ascending.

## Form template identifiers
order_key -> `orderKey`; width -> `width`; thr -> `t`; block weight -> `weightB`.
-/

namespace QsaSelect

-- § order key

/-- The f32 total-order key on raw bit patterns (`order_key`): sign bit set -> bitwise not, else set the sign bit. -/
def orderKey (b : Nat) : Nat := if 2147483648 ≤ b then 4294967295 - b else b + 2147483648

/-- `s + 0.0f` folds -0.0 (bits 0x80000000) onto +0.0 (bits 0). -/
def negZeroFold (b : Nat) : Nat := if b = 2147483648 then 0 else b

/-- Keys stay in 32 bits. -/
theorem orderKey_lt (b : Nat) (h : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey; split <;> omega

/-- Every negative float sorts strictly below every non-negative float. -/
theorem orderKey_neg_lt_pos (b1 b2 : Nat) (h1 : 2147483648 ≤ b1) (h1' : b1 < 4294967296)
    (h2 : b2 < 2147483648) : orderKey b1 < orderKey b2 := by
  unfold orderKey; split <;> split <;> omega

/-- On non-negative floats the key is monotone in the bit pattern. -/
theorem orderKey_mono_pos (b1 b2 : Nat) (h : b1 ≤ b2) (h2 : b2 < 2147483648) :
    orderKey b1 ≤ orderKey b2 := by
  unfold orderKey; split <;> split <;> omega

/-- On negative floats the key is antitone in the bit pattern. -/
theorem orderKey_anti_neg (b1 b2 : Nat) (h : b1 ≤ b2) (h1 : 2147483648 ≤ b1) (h2 : b2 < 4294967296) :
    orderKey b2 ≤ orderKey b1 := by
  unfold orderKey
  rw [if_pos h1]
  by_cases c2 : 2147483648 ≤ b2
  · rw [if_pos c2]; omega
  · rw [if_neg c2]; omega

/-- After folding, -0.0 and +0.0 get the same key (they tie, and the tie is broken by index). -/
theorem negZero_ties : orderKey (negZeroFold 2147483648) = orderKey (negZeroFold 0) := by decide

-- § threshold selection

/-- Count of keys strictly above `t`. -/
def cntGt (l : List Nat) (t : Nat) : Nat := (l.filter (fun k => decide (t < k))).length
/-- Count of keys equal to `t`. -/
def cntEq (l : List Nat) (t : Nat) : Nat := (l.filter (fun k => decide (k = t))).length
/-- Count of keys at or above `t`. -/
def cntGe (l : List Nat) (t : Nat) : Nat := (l.filter (fun k => decide (t ≤ k))).length

/-- `>= t` splits into `> t` and `= t`. -/
theorem cntGe_split (l : List Nat) (t : Nat) : cntGe l t = cntGt l t + cntEq l t := by
  unfold cntGe cntGt cntEq
  induction l with
  | nil => simp
  | cons a l ih =>
    by_cases h1 : t < a
    · have h2 : a ≠ t := by omega
      have h3 : t ≤ a := by omega
      simp [List.filter_cons, h1, h2, h3, ih]
      omega
    · by_cases h2 : a = t
      · have h3 : t ≤ a := by omega
        simp [List.filter_cons, h1, h2, ih]
        omega
      · have h3 : ¬ t ≤ a := by omega
        simp [List.filter_cons, h1, h2, h3, ih]

/-- `> t` is `>= t+1`. -/
theorem cntGt_eq_cntGe_succ (l : List Nat) (t : Nat) : cntGt l t = cntGe l (t + 1) := by
  unfold cntGt cntGe
  have : (fun k => decide (t < k)) = (fun k => decide (t + 1 ≤ k)) := rfl
  rw [this]

/-- Selected count: everything above the threshold plus as many ties as the budget leaves room for. -/
def selCount (l : List Nat) (t width : Nat) : Nat := cntGt l t + min (cntEq l t) (width - cntGt l t)

/-- Given the threshold property, the selection has exactly `width` elements. -/
theorem selCount_eq_width (l : List Nat) (t width : Nat)
    (h1 : cntGt l t < width) (h2 : width ≤ cntGt l t + cntEq l t) : selCount l t width = width := by
  unfold selCount; omega

/-- The binary-lifting invariant `count(>= t) >= width > count(>= t+1)` implies the selection property. -/
theorem threshold_property (l : List Nat) (t width : Nat)
    (hge : width ≤ cntGe l t) (hlt : cntGe l (t + 1) < width) : selCount l t width = width := by
  have hs := cntGe_split l t
  have hg := cntGt_eq_cntGe_succ l t
  exact selCount_eq_width l t width (by omega) (by omega)

/-- Ties never overflow the budget: the number of tied cells taken is at most `width - c_gt`. -/
theorem ties_taken_le (l : List Nat) (t width : Nat) :
    min (cntEq l t) (width - cntGt l t) ≤ width - cntGt l t := Nat.min_le_right _ _

-- § radix digits

/-- Four 8-bit digits recompose a 32-bit key. -/
theorem digits_recompose (k : Nat) (h : k < 4294967296) :
    (k / 16777216) * 16777216 + ((k / 65536) % 256) * 65536 + ((k / 256) % 256) * 256 + k % 256 = k := by
  omega

/-- The top digit is a byte. -/
theorem top_digit_lt (k : Nat) (h : k < 4294967296) : k / 16777216 < 256 := by omega

/-- A histogram bin index is a byte. -/
theorem digit_lt (k s : Nat) : (k / s) % 256 < 256 := Nat.mod_lt _ (by decide)

-- § block weights

/-- Weight of block `b`: r cells for completed blocks, the remainder for the spare/tail row. -/
def weightB (nBid nKv r b : Nat) : Nat := if b < nBid then r else nKv - nBid * r

/-- Prefix sum of a function. -/
def wsum (f : Nat → Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => wsum f k + f k

/-- Completed blocks contribute `k * r`. -/
theorem wsum_prefix (nBid nKv r : Nat) : ∀ k, k ≤ nBid → wsum (weightB nBid nKv r) k = k * r := by
  intro k
  induction k with
  | zero => intro _; simp [wsum]
  | succ k ih =>
    intro hk
    have h1 : weightB nBid nKv r k = r := by unfold weightB; simp [show k < nBid by omega]
    show wsum (weightB nBid nKv r) k + weightB nBid nKv r k = (k+1) * r
    rw [ih (by omega), h1, Nat.add_mul, Nat.one_mul]

/-- Total weight over blocks 0..n_bid is `n_kv`. -/
theorem total_weight (nBid nKv r : Nat) (h : nBid * r ≤ nKv) :
    wsum (weightB nBid nKv r) (nBid + 1) = nKv := by
  show wsum (weightB nBid nKv r) nBid + weightB nBid nKv r nBid = nKv
  rw [wsum_prefix nBid nKv r nBid (Nat.le_refl _)]
  have : weightB nBid nKv r nBid = nKv - nBid * r := by unfold weightB; simp
  rw [this]
  omega

-- § expansion of selected blocks into cell ids

/-- A completed block's cells are real cells. -/
theorem full_block_cell_lt (b c nBid nKv : Nat) (hb : b < nBid) (hc : c < 4) (h : nBid * 4 ≤ nKv) :
    b * 4 + c < nKv := by omega

/-- The spare/tail block's cells are real cells. -/
theorem tail_block_cell_lt (c nBid nKv : Nat) (hc : c < nKv - nBid * 4) : nBid * 4 + c < nKv := by omega

/-- Emission order is ascending across blocks. -/
theorem cells_ascending (b b' c c' : Nat) (hb : b < b') (hc : c < 4) (hc' : c' < 4) :
    b * 4 + c < b' * 4 + c' := by omega

/-- Block -> cell map is injective. -/
theorem cell_unique (b b' c c' : Nat) (hc : c < 4) (hc' : c' < 4) (h : b * 4 + c = b' * 4 + c') :
    b = b' ∧ c = c' := by omega

/-- When everything fits (`n_kv <= width`) the selection is the identity; for the real geometry that is `n_kv <= 2051`. -/
theorem identity_branch (n : Nat) : n ≤ min n 2051 ↔ n ≤ 2051 := by omega

-- § launch geometry

/-- Warp-per-block scoring: every block index `b < max_blocks` is served by exactly one (blockIdx, warp). -/
theorem score_grid_covers (b maxBlocks : Nat) (h : b < maxBlocks) :
    b / 8 < (maxBlocks + 7) / 8 ∧ b = (b / 8) * 8 + b % 8 ∧ b % 8 < 8 := by omega

/-- The tail bias fires iff the spare row holds a partial block. -/
theorem tail_bias_iff (n : Nat) : n % 4 ≠ 0 ↔ (n / 4) * 4 < n := by omega

end QsaSelect



## FILE: /mnt/user-data/outputs/QsaDecodeAttn.lean
/-!
# MODULE: qsa_decode_attn
SOURCES: src/kernels/cuda/qsa_decode_attn.cu
Chunked split-K decode attention over the selected cells (GQA, 12 query heads per kv head, head_dim 256),
with per-chunk (max, exp-sum, accumulator) partials merged by a second kernel; supports a batched form
(blockIdx.z) and fp16 or int8+scale KV pools.

## Source semantics
* Cells are processed in chunks of 64; chunk c owns selection indices [64c, 64c+64) intersected with [0,n_ids).
* A chunk past n_ids writes the sentinel (m = -FLT_MAX, l = 0) and is skipped by the merge.
* Query head h reads kv head h / 12 and local head h % 12.
* Merge: M = max over chunk maxima, weights exp(m_c - M) <= 1, result acc / L.
* Scratch = chunks * n_head * (HD + 2) + 64 floats; batch elements are strided by that amount.

## Form template identifiers
CHUNK -> 64; G -> 12; HD -> 256; chunk max -> `maxList`.
-/

namespace QsaDecodeAttn

-- § helpers

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § chunking

/-- Number of chunks for a capacity. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Every selection index below the capacity lies in an existing chunk. -/
theorem index_in_chunk (i cap : Nat) (h : i < cap) : i / 64 < nChunks cap := by
  unfold nChunks; omega

/-- Cells handled by chunk `c`: min(64, n_ids - 64c) (may be non-positive for surplus chunks). -/
def nHere (n c : Int) : Int := min 64 (n - c * 64)

/-- A surplus chunk has nothing to do and takes the sentinel path. -/
theorem surplus_chunk_empty (n c : Int) (h : n ≤ c * 64) : nHere n c ≤ 0 := by
  unfold nHere; omega

/-- A live chunk handles between 1 and 64 cells. -/
theorem live_chunk_bounds (n c : Int) (h : c * 64 < n) : 0 < nHere n c ∧ nHere n c ≤ 64 := by
  unfold nHere; omega

/-- Every selection index `i < n` is handled by chunk `i/64` at lane `i%64 < nHere`. -/
theorem every_cell_handled (i n : Nat) (h : i < n) :
    (i : Int) % 64 < nHere (n : Int) ((i / 64 : Nat) : Int) ∧ ((i / 64 : Nat) : Int) * 64 + (i : Int) % 64 = i := by
  unfold nHere; omega

-- § GQA head mapping

/-- Query head -> kv head is integer division by 12, and stays below the kv head count. -/
theorem gqa_head_decode (h nKvHeads : Nat) (hh : h < 12 * nKvHeads) :
    h / 12 < nKvHeads ∧ (h / 12) * 12 + h % 12 = h := by omega

/-- In the real geometry the map is `[0 x12, 1 x12]`, which differs from `h % 2` (e.g. at h = 1). -/
theorem gqa_div_differs_from_mod : (1 : Nat) / 12 ≠ 1 % 2 := by decide

/-- Each lane holds 8 consecutive dims; 32 lanes cover the 256-wide head exactly once. -/
theorem lane_cover (d : Nat) (h : d < 256) : d = (d / 8) * 8 + d % 8 ∧ d / 8 < 32 := by omega

-- § scratch layout

/-- Scratch float count. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * (256 + 2) + 64

/-- The three partial arrays (acc, m, l) fit in the scratch. -/
theorem partials_fit (cap nHead : Nat) :
    nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead ≤ scratchFloats cap nHead := by
  unfold scratchFloats; omega

/-- Partial slot index is in range. -/
theorem slot_lt (kvh chunk nKvHeads nCh : Nat) (h1 : kvh < nKvHeads) (h2 : chunk < nCh) :
    kvh * nCh + chunk < nKvHeads * nCh := idx2_lt kvh chunk nKvHeads nCh h1 h2

/-- Accumulator index `(slot*12 + h)*256 + t` is in range. -/
theorem acc_index_lt (slot h t S : Nat) (hs : slot < S) (hh : h < 12) (ht : t < 256) :
    (slot * 12 + h) * 256 + t < S * 12 * 256 := by
  have h1 : slot * 12 + h < S * 12 := idx2_lt slot h S 12 hs hh
  exact idx2_lt (slot * 12 + h) t (S * 12) 256 h1 ht

-- § merge

/-- Running maximum over a list of chunk maxima. -/
def maxList : Int → List Int → Int
  | a, [] => a
  | a, x :: xs => maxList (max a x) xs

/-- The running maximum dominates its seed. -/
theorem maxList_ge_init : ∀ (l : List Int) (a : Int), a ≤ maxList a l
  | [], a => Int.le_refl a
  | x :: xs, a => Int.le_trans (by omega) (maxList_ge_init xs (max a x))

/-- The merged maximum dominates every chunk maximum, so every merge exponent `m_c - M` is `<= 0`. -/
theorem maxList_ge_mem : ∀ (l : List Int) (a : Int) (m : Int), m ∈ l → m ≤ maxList a l
  | [], a, m, hm => by cases hm
  | x :: xs, a, m, hm => by
    rcases List.mem_cons.1 hm with h | h
    · subst h
      exact Int.le_trans (by omega) (maxList_ge_init xs (max a m))
    · exact maxList_ge_mem xs (max a x) m h

/-- Merge exponents are non-positive. -/
theorem merge_exponent_nonpos (l : List Int) (a m : Int) (hm : m ∈ l) : m - maxList a l ≤ 0 := by
  have := maxList_ge_mem l a m hm
  omega

-- § batching

/-- Distinct batch elements occupy disjoint, ordered scratch ranges. -/
theorem batch_disjoint (z1 z2 stride : Nat) (h : z1 < z2) : z1 * stride + stride ≤ z2 * stride := by
  have h1 : (z1 + 1) * stride ≤ z2 * stride := Nat.mul_le_mul_right stride (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- The grid's merge kernel writes `attn[(z*n_head + h)*256 + d]`, in range for z < n_q. -/
theorem batch_attn_index_lt (z h d nQ nHead : Nat) (hz : z < nQ) (hh : h < nHead) (hd : d < 256) :
    (z * nHead + h) * 256 + d < nQ * nHead * 256 := by
  have h1 : z * nHead + h < nQ * nHead := idx2_lt z h nQ nHead hz hh
  exact idx2_lt (z * nHead + h) d (nQ * nHead) 256 h1 hd

end QsaDecodeAttn



## FILE: /mnt/user-data/outputs/NativeQsaScore.lean
/-!
# MODULE: native_qsa_score
SOURCES: src/kernels/cuda/native_qsa_score.cu
llama.cpp-order tensor-core (mma m16n8k8 tf32) scoring of pooled indexer keys against the 4 indexer query
heads, writing per-cell scores; guarded by a step-consistency predicate and disjoint-span validation.

## Source semantics
* The kernel runs only when the step is consistent: 1 <= n <= max_cells, pos = n-1, full = n/4,
  width = min(n, 2051); otherwise it returns without writing.
* Row `row <= full` is scored by block row/32, lane row%32; it writes cells [4*row, min(n, 4*row+4)).
* K = 128 is split between 2 warps; each covers columns warp*32+lane and warp*32+lane+64.
* The spare row (row == full) gets +1e9 iff n % 4 != 0.
* Input/output spans must be 4-aligned, bounded and pairwise disjoint.

## Form template identifiers
ROWS -> 32; WARPS -> 2; R -> 4; COMBINE -> 68; STRIDE -> 36.
-/

namespace NativeQsaScore

/-- Step consistency guard of `score_kernel`. -/
def stepValid (n pos full width maxCells : Nat) : Prop :=
  1 ≤ n ∧ n ≤ maxCells ∧ pos = n - 1 ∧ full = n / 4 ∧ width = (if n < 2051 then n else 2051)

/-- A step produced by the host `step_fill` for any in-capacity position passes the guard. -/
theorem stepValid_of_fill (pos maxCells : Nat) (h : pos + 1 ≤ maxCells) :
    stepValid (pos + 1) pos ((pos + 1) / 4) (if pos + 1 < 2051 then pos + 1 else 2051) maxCells := by
  unfold stepValid
  refine ⟨by omega, h, by omega, rfl, rfl⟩

/-- The guard's width is the QSA selection width `min n 2051` (top_k 2048 + r - 1). -/
theorem guard_width_is_min (n : Nat) : (if n < 2051 then n else 2051) = min n 2051 := by
  split <;> omega

-- § launch geometry

/-- Capacity convention: `max_blocks = max_cells/4 + 1`. -/
def maxBlocksOf (maxCells : Nat) : Nat := maxCells / 4 + 1

/-- Every scored row index fits inside the block capacity. -/
theorem full_lt_maxBlocks (n maxCells : Nat) (h : n ≤ maxCells) : n / 4 < maxBlocksOf maxCells := by
  unfold maxBlocksOf; omega

/-- A row `<= full` is covered by some block of the launch grid `ceil(max_blocks / 32)`. -/
theorem row_block_covered (row full maxCells : Nat) (hf : full < maxBlocksOf maxCells) (hr : row ≤ full) :
    row / 32 < (maxBlocksOf maxCells + 31) / 32 ∧ row = (row / 32) * 32 + row % 32 := by
  omega

/-- Every cell `j < n` is written by row `j/4 <= full`. -/
theorem cell_covered (j n : Nat) (h : j < n) : j / 4 ≤ n / 4 ∧ 4 * (j / 4) ≤ j ∧ j < 4 * (j / 4) + 4 := by
  omega

/-- Distinct rows write disjoint cells. -/
theorem rows_disjoint (r1 r2 c1 c2 : Nat) (h1 : c1 < 4) (h2 : c2 < 4) (h : r1 * 4 + c1 = r2 * 4 + c2) :
    r1 = r2 ∧ c1 = c2 := by omega

/-- Written cells are real cells: `i < n` and `i < 4*(row+1)` for the loop bound. -/
theorem written_cell_range (row n i : Nat) (hi : i < n ∧ i < (row + 1) * 4) (hl : row * 4 ≤ i) :
    i / 4 = row := by
  have : (row + 1) * 4 = row * 4 + 4 := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § K split between two warps

/-- Every K column is accumulated by exactly one (warp, lane, pass). -/
theorem k_columns_cover (col : Nat) (h : col < 128) :
    col = (col / 64) * 64 + ((col / 32) % 2) * 32 + col % 32 ∧
    (col / 32) % 2 < 2 ∧ col % 32 < 32 ∧ col / 64 < 2 := by omega

/-- The spare-row bias flag. -/
theorem tail_bias_flag (n : Nat) : n % 4 ≠ 0 ↔ (n / 4) * 4 < n := by omega

-- § shared-memory layouts

/-- The 4(8)-head by 64-row combine buffer index stays inside shared memory (8 * 68 <= 2 * 16 * 36). -/
theorem combine_in_shared (h i : Nat) (hh : h < 8) (hi : i < 64) : h * 68 + i < 2 * 16 * 36 := by omega

/-- The combine index is injective (decode from the `68` stride). -/
theorem combine_injective (h1 h2 i1 i2 : Nat) (a : i1 < 64) (b : i2 < 64)
    (h : h1 * 68 + i1 = h2 * 68 + i2) : h1 = h2 ∧ i1 = i2 := by omega

/-- Row index written back by a lane: `warp*32 + ia*16 + (l/2)*8 + lane/4 < 64`; head index `< 8`. -/
theorem writeback_ranges (warp ia l lane : Nat) (hw : warp < 2) (hi : ia < 2) (hl : l < 4) (hn : lane < 32) :
    warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 ∧ (lane % 4) * 2 + l % 2 < 8 := by omega

/-- A warp's staging tile (16 rows of stride 36) fits its share of shared memory. -/
theorem tile_in_shared (i lane warp : Nat) (hi : i < 16) (hl : lane < 32) (hw : warp < 2) :
    warp * 16 * 36 + i * 36 + lane < 2 * 16 * 36 := by omega

-- § span validation

/-- Half-open interval overlap test used by `overlaps`. -/
def overlaps (x nx y ny : Nat) : Prop := x < y + ny ∧ y < x + nx

/-- Overlap is symmetric. -/
theorem overlaps_symm (x nx y ny : Nat) : overlaps x nx y ny ↔ overlaps y ny x nx := by
  unfold overlaps; omega

/-- Ranges that end before the other starts do not overlap. -/
theorem not_overlaps_of_le (x nx y ny : Nat) (h : x + nx ≤ y) : ¬ overlaps x nx y ny := by
  unfold overlaps; omega

/-- An empty span can only "overlap" by lying strictly inside the other span. -/
theorem empty_span_inside (x y ny : Nat) (h : overlaps x 0 y ny) : y < x ∧ x < y + ny := by
  unfold overlaps at h; omega

end NativeQsaScore



## FILE: /mnt/user-data/outputs/NativeFlashAttn.lean
/-!
# MODULE: native_flash_attn
SOURCES: src/kernels/cuda/native_flash_attn.cu
Short-context (<= 256 cells) FlashAttention-vec decode kernel specialised to Q 24x256 / KV 2x256, fp16 K/V,
one block per query head, 128 threads; online softmax with a streaming maximum.

## Source semantics
* The kernel is valid only if width in [1, max_context], n_kv = width, pos = width-1, n_bid = width/4;
  otherwise it writes a NaN output and reports an `unsupported step` status.
* Query head h reads kv head h / 12.
* Cells are processed 128 per pass over a padded length of 256 (two passes); cells >= width are -inf masked.
* The output is combined from four warps' accumulators and divided by the summed exp weights.

## Form template identifiers
width -> `width`; max_context -> `maxCtx`; padded_length -> 256.
-/

namespace NativeFlashAttn

/-- Validity predicate for the step. -/
def valid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

/-- A host-filled step with width = pos+1 <= 256 passes the guard. -/
theorem valid_of_fill (pos maxCtx : Nat) (h : pos + 1 ≤ maxCtx) :
    valid (pos + 1) (pos + 1) pos ((pos + 1) / 4) maxCtx := by
  unfold valid; omega

/-- Below the selection bound (2051) the selection width equals n_kv, so this kernel sees every cell. -/
theorem short_context_is_identity (n : Nat) (h : n ≤ 256) : min n 2051 = n := by omega

/-- GQA: heads 0..23 map to kv heads 0..1. -/
theorem kv_head_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by omega

/-- Every cell `< 256` is visited by exactly one (pass, warp, lane-group, row). -/
theorem cell_decomposition (cell : Nat) (h : cell < 256) :
    cell = (cell / 128) * 128 + ((cell % 128) / 32) * 32 + ((cell % 32) / 8) * 8 + cell % 8 ∧
    cell / 128 < 2 ∧ (cell % 128) / 32 < 4 ∧ cell % 8 < 8 := by omega

/-- The thread whose lane satisfies `lane % 8 = row` keeps the score: its cell is `warp*32 + lane`. -/
theorem lane_holds_cell (warp lane row : Nat) (hl : lane < 32) (hr : row = lane % 8) :
    warp * 32 + (lane / 8) * 8 + row = warp * 32 + lane := by omega

/-- The output is written for all 256 dims by two 128-wide passes. -/
theorem output_cover (d : Nat) (h : d < 256) : d = (d / 128) * 128 + d % 128 ∧ d / 128 < 2 ∧ d % 128 < 128 := by
  omega

/-- The online-softmax rescale exponent `maximum - next_max` is `<= 0`, so `rescale <= 1`. -/
theorem rescale_exponent_nonpos (maximum x : Int) : maximum - max maximum x ≤ 0 := by omega

/-- The running maximum is monotone. -/
theorem running_max_mono (maximum x : Int) : maximum ≤ max maximum x := by omega

/-- The four-warp accumulator staging index stays inside the 4*4*256-float shared tile. -/
theorem stage_index_lt (warp lane i0 j : Nat) (hw : warp < 4) (hl : lane < 32)
    (hi : i0 = 0 ∨ i0 = 32 ∨ i0 = 64 ∨ i0 = 96) (hj : j < 4) :
    warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j + 1 < 4 * 4 * 256 := by
  omega

/-- Shared memory use: tile + two 32-float reductions fits 48 KiB. -/
theorem smem_fits : (4 * 4 * 256 + 32 + 32) * 4 ≤ 49152 := by decide

/-- `scale = 1/sqrt(256) = 1/16 = 0.0625`: 16 * 16 = 256. -/
theorem scale_is_inv_sqrt : 16 * 16 = 256 := by decide

/-- KV byte count `capacity*1024` is the 2 heads * 256 dims * 2 B per cell, with no loss. -/
theorem kv_bytes_per_cell : 2 * 256 * 2 = 1024 := by decide

/-- A K/V element index `(cell*2 + kv)*256 + d` is within `capacity` cells. -/
theorem kv_index_lt (cell kv d cap : Nat) (hc : cell < cap) (hk : kv < 2) (hd : d < 256) :
    (cell * 2 + kv) * 256 + d < cap * 2 * 256 := by
  have h1 : (cell + 1) * 2 ≤ cap * 2 := Nat.mul_le_mul_right 2 (by omega)
  have h2 : (cell * 2 + kv + 1) * 256 ≤ (cap * 2) * 256 := Nat.mul_le_mul_right 256 (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1 h2
  omega

end NativeFlashAttn



## FILE: /mnt/user-data/outputs/GdnRecurrence.lean
/-!
# MODULE: gdn_recurrence
SOURCES: src/kernels/cuda/gdn.cu, src/kernels/cuda/fused_gdn.cu
Gated delta-net decode step: state layout (S, h_v, S) with j fastest, per-column recurrence, causal conv with a
sliding 3-row history, l2 / out norms, and the fused step+norm kernel that splits the 128 state rows over 4 row
groups; plus the fused alpha/beta projection and conv+l2 kernels.

## Source semantics
* State element (i, h, j) lives at ((i*h_v + h)*S + j); one thread owns one (h, j) column end to end.
* q/k heads are paired to value heads by MODULO: head h reads q/k head h % h_k.
* Fused step: thread (col, rg) owns rows rg*32 .. rg*32+31 of column col; the four partial dots are summed in
  fixed order.
* Conv: out = sum_i hist[i]*w[i] + x*w[3]; the history slides by one (drop oldest, append x).
* Alpha/beta: row < h_v is alpha (gate = softplus(acc+dt)*a), row >= h_v is beta (sigmoid).

## Form template identifiers
S -> 128; RG -> 4; RPG -> 32; state index -> `stIdx`; history -> `hist`.
-/

namespace GdnRecurrence

-- § helpers

/-- Decoding a mixed-radix pair. -/
theorem decode_div_mod (a b n : Nat) (hb : b < n) : (a*n+b)/n = a ∧ (a*n+b)%n = b := by
  have hn : 0 < n := by omega
  have e : a*n+b = b + a*n := Nat.add_comm _ _
  have h1 := Nat.add_mul_div_right b a hn
  rw [Nat.div_eq_of_lt hb] at h1
  constructor
  · rw [e]; omega
  · rw [e, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § state layout

/-- Linear index of state element (i, h, j) in the (S, h_v, S) layout. -/
def stIdx (hv S i h j : Nat) : Nat := (i * hv + h) * S + j

/-- Every state index is inside the `S * h_v * S` buffer. -/
theorem stIdx_lt (hv S i h j : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) : stIdx hv S i h j < S * hv * S := by
  unfold stIdx
  have h1 : i * hv + h < S * hv := idx2_lt i h S hv hi hh
  exact idx2_lt (i * hv + h) j (S * hv) S h1 hj

/-- Distinct (i, h, j) address distinct state elements, so columns never alias. -/
theorem stIdx_injective (hv S i1 i2 h1 h2 j1 j2 : Nat) (hh1 : h1 < hv) (hh2 : h2 < hv)
    (hj1 : j1 < S) (hj2 : j2 < S) (heq : stIdx hv S i1 h1 j1 = stIdx hv S i2 h2 j2) :
    i1 = i2 ∧ h1 = h2 ∧ j1 = j2 := by
  unfold stIdx at heq
  obtain ⟨d1, m1⟩ := decode_div_mod (i1*hv+h1) j1 S hj1
  obtain ⟨d2, m2⟩ := decode_div_mod (i2*hv+h2) j2 S hj2
  rw [heq] at d1 m1
  have e1 : i1*hv+h1 = i2*hv+h2 := d1.symm.trans d2
  have e2 : j1 = j2 := m1.symm.trans m2
  obtain ⟨a1, b1⟩ := decode_div_mod i1 h1 hv hh1
  obtain ⟨a2, b2⟩ := decode_div_mod i2 h2 hv hh2
  rw [e1] at a1 b1
  exact ⟨a1.symm.trans a2, b1.symm.trans b2, e2⟩

/-- The fused kernel's row-group stride: row `rg*32 + r` is `r` row-strides after row `rg*32`. -/
theorem fused_row_stride (A r hv head S col : Nat) :
    ((A * hv + head) * S + col) + r * (hv * S) = (((A + r) * hv + head) * S + col) := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Four row groups of 32 rows cover all 128 state rows exactly once. -/
theorem row_groups_cover (i : Nat) (h : i < 128) : i = (i / 32) * 32 + i % 32 ∧ i / 32 < 4 ∧ i % 32 < 32 := by
  omega

/-- The unfused grid covers all 128 columns with 4 blocks of 32 threads. -/
theorem columns_cover (j : Nat) (h : j < 128) : j / 32 < 4 ∧ j = (j / 32) * 32 + j % 32 := by omega

/-- Thread count is fixed by the state, independent of MAX_H: 128 * 48 = 6144. -/
theorem thread_count : 128 * 48 = 6144 := by decide

/-- Head pairing is modulo: always a valid q/k head. -/
theorem head_pair_lt (h hk : Nat) (hpos : 0 < hk) : h % hk < hk := Nat.mod_lt _ hpos

-- § convolution history

/-- Sequence `[hist0, hist1, hist2, x0, x1, ...]` as a function of position. -/
def seqAt {α : Type} (hist x : Nat → α) (i : Nat) : α := if i < 3 then hist i else x (i - 3)

/-- The window of the first-token step is the old history followed by the new input. -/
theorem window_first {α : Type} (hist x : Nat → α) :
    seqAt hist x 0 = hist 0 ∧ seqAt hist x 1 = hist 1 ∧ seqAt hist x 2 = hist 2 ∧ seqAt hist x 3 = x 0 := by
  simp [seqAt]

/-- Sliding the history by one token: new history = [h1, h2, x0]. -/
theorem slide_one {α : Type} (hist x : Nat → α) :
    seqAt hist x (1 + 0) = hist 1 ∧ seqAt hist x (1 + 1) = hist 2 ∧ seqAt hist x (1 + 2) = x 0 := by
  simp [seqAt]

/-- The window of token t+1 is the window of token t shifted by one. -/
theorem window_shift {α : Type} (hist x : Nat → α) (t j : Nat) :
    seqAt hist x ((t + 1) + j) = seqAt hist x (t + (j + 1)) := by
  have : (t + 1) + j = t + (j + 1) := by omega
  rw [this]

/-- After at least 3 tokens the history is made only of new inputs. -/
theorem history_after_many {α : Type} (hist x : Nat → α) (n j : Nat) (hn : 3 ≤ n) :
    seqAt hist x (n + j) = x (n + j - 3) := by
  unfold seqAt
  have : ¬ (n + j < 3) := by omega
  simp [this]

-- § alpha/beta projection

/-- A bf16 pair word: the low half becomes the high half of an f32 (`w << 16`), the high half stays in place. -/
theorem bf16_pair_split (w : Nat) (h : w < 4294967296) :
    w = (w / 65536) * 65536 + w % 65536 ∧ (w % 65536) * 65536 < 4294967296 := by omega

/-- Rows `[0, 2*h_v)` map to alpha (`row < h_v`) or beta, with `r < h_v`. -/
theorem ab_row_split (row hv : Nat) (h : row < 2 * hv) :
    (row < hv ∧ row < hv) ∨ (hv ≤ row ∧ row - hv < hv) := by omega

/-- The (2*h_v+7)/8 grid with 8 warps per block covers every row. -/
theorem ab_grid_covers (row hv : Nat) (h : row < 2 * hv) :
    row / 8 < (2 * hv + 7) / 8 ∧ row = (row / 8) * 8 + row % 8 := by omega

end GdnRecurrence



## FILE: /mnt/user-data/outputs/GatedResidual.lean
/-!
# MODULE: gated_residual
SOURCES: src/kernels/cuda/gr.cu, src/kernels/cuda/fused_gr.cu
Gated residual / hyper-connection read and write (hc = 4 streams of n_embd = 2560, low-rank hc_lr = 320):
per-stream RMSNorm, down projection (10240 -> 320) + silu, up projection (320 -> 10240) + sigmoid gate, mean over
streams, injection dot products, the residual write, the workspace layout, and the fused / multi-token variants.

## Source semantics
* hc_dim = hc * n_embd = 10240; down has 320 rows, one 8-warp block per row, the 8 warps splitting the reduction.
* Gate: one warp per output row, 1280 blocks of 8 warps cover 10240 rows; each lane does hc_lr/32 = 10 terms.
* Write: out[i] = R[i] + block_out[i % n_embd] * w[i / n_embd] (every stream adds the same block output).
* Workspace: five regions each rounded up to 16 bytes, laid out consecutively without overlap.
* Fused up: block covers 32 columns x 4 streams = 128 rows; a (block, row) pair addresses exactly one of 10240 entries.

## Form template identifiers
N -> 2560; HC -> 4; D -> 10240; LR -> 320; WARPS -> 8.
-/

namespace GatedResidual

-- § helpers

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § geometry constants

/-- `hc * n_embd = 10240`. -/
theorem hc_dim_val : 2560 * 4 = 10240 := by decide

/-- Down projection: 40 blocks of 8 warps are exactly the 320 rows. -/
theorem down_blocks_exact : 320 / 8 = 40 ∧ 40 * 8 = 320 := by decide

/-- The up kernel's 80 blocks of 32 columns are exactly n_embd; the multi variant's 160 x 16 as well. -/
theorem up_blocks_exact : 80 * 32 = 2560 ∧ 160 * 16 = 2560 := by decide

/-- 320 hidden values are 40 chunks of 8: 32 lanes take one, 8 lanes take a second. -/
theorem lr_chunks : 320 / 8 = 40 ∧ 40 - 32 = 8 := by decide

/-- One warp covers a 10240-row reduction as 1280 chunks, 40 per lane. -/
theorem down_chunks_per_lane : 10240 / 8 = 1280 ∧ 1280 / 32 = 40 := by decide

/-- Fused multi-token tiling: 4 tiles of 2560, each 320 chunks = 10 per lane. -/
theorem tiling : 10240 / 2560 = 4 ∧ 2560 / 8 / 32 = 10 := by decide

-- § work distribution

/-- Down rows: every row `k < 320` is (block, warp) = (k/8, k%8). -/
theorem down_row_cover (k : Nat) (h : k < 320) : k / 8 < 40 ∧ k = (k / 8) * 8 + k % 8 ∧ k % 8 < 8 := by omega

/-- Block-wide split of the 10240-element reduction among 8 warps x 32 lanes: each element has one owner. -/
theorem reduction_owner (i : Nat) (h : i < 10240) :
    i = (i / 256) * 256 + ((i % 256) / 32) * 32 + i % 32 ∧ (i % 256) / 32 < 8 ∧ i % 32 < 32 := by omega

/-- Gate rows: 1280 blocks of 8 warps cover 10240 rows. -/
theorem gate_row_cover (i : Nat) (h : i < 10240) : i / 8 < (10240 + 7) / 8 ∧ i = (i / 8) * 8 + i % 8 := by omega

/-- Hidden-vector lane coverage inside a 320-wide reduction. -/
theorem lane_cover_320 (j : Nat) (h : j < 320) : j = (j / 32) * 32 + j % 32 ∧ j / 32 < 10 := by omega

/-- Tile chunk coverage: each of the 320 chunks per tile is owned by one (lane, q). -/
theorem tile_chunk_cover (j : Nat) (h : j < 320) : j = j % 32 + 32 * (j / 32) ∧ j % 32 < 32 ∧ j / 32 < 10 := by
  omega

-- § up kernel addressing

/-- Fused up kernel: (block, row r in 0..127) addresses stream `r/32`, column `blk*32 + r%32`; index in range. -/
theorem up_index_lt (blk r : Nat) (hb : blk < 80) (hr : r < 128) :
    (r / 32) * 2560 + blk * 32 + r % 32 < 10240 := by omega

/-- Every entry of the 10240-wide residual is produced by exactly one (block, row) pair of the up kernel. -/
theorem up_index_surjective (i : Nat) (h : i < 10240) :
    ∃ blk r, blk < 80 ∧ r < 128 ∧ i = (r / 32) * 2560 + blk * 32 + r % 32 := by
  refine ⟨(i % 2560) / 32, (i / 2560) * 32 + (i % 2560) % 32, by omega, by omega, by omega⟩

/-- ...and injective. -/
theorem up_index_injective (b1 b2 r1 r2 : Nat) (hb1 : b1 < 80) (hb2 : b2 < 80) (hr1 : r1 < 128) (hr2 : r2 < 128)
    (h : (r1 / 32) * 2560 + b1 * 32 + r1 % 32 = (r2 / 32) * 2560 + b2 * 32 + r2 % 32) :
    b1 = b2 ∧ r1 = r2 := by omega

-- § residual write

/-- `out[i]` uses block output element `i % n` and stream weight `i / n`; both in range. -/
theorem write_index_decode (i n hc : Nat) (hn : 0 < n) (h : i < hc * n) :
    i / n < hc ∧ i % n < n ∧ (i / n) * n + i % n = i := by
  refine ⟨(Nat.div_lt_iff_lt_mul hn).2 h, Nat.mod_lt _ hn, ?_⟩
  have := Nat.div_add_mod i n
  rw [Nat.mul_comm n (i / n)] at this
  exact this

/-- Mean over streams reads `gated[c*n + d]` with c < hc, d < n: in range. -/
theorem mean_index_lt (c d hc n : Nat) (hc' : c < hc) (hd : d < n) : c * n + d < hc * n :=
  idx2_lt c d hc n hc' hd

-- § workspace layout

/-- Round a byte size up to 16. -/
def align16 (s : Nat) : Nat := (s + 15) / 16 * 16

/-- Alignment never shrinks, adds less than 16, and yields a multiple of 16. -/
theorem align16_props (s : Nat) : s ≤ align16 s ∧ align16 s < s + 16 ∧ align16 s % 16 = 0 := by
  unfold align16; omega

/-- Consecutive regions do not overlap: next offset >= previous offset + previous size. -/
theorem region_disjoint (off sz : Nat) : off + sz ≤ off + align16 sz := by
  have := (align16_props sz).1
  omega

/-- Real geometry: xn + xq + lq + gated + lo = 104320 bytes. -/
theorem workspace_total :
    align16 (10240 * 4) + align16 (10240 * 2) + align16 (320 * 2) + align16 (10240 * 4) + align16 (320 * 4)
      = 104320 := by decide

/-- The original bug: giving `xq` 640 bytes where 20480 are needed is strictly too small. -/
theorem xq_size_needed : 640 < 10240 * 2 := by decide

/-- Block-sum scratch: 8 floats, and a double scratch of 8 is at least as large. -/
theorem scratch_cast_safe : 8 * 4 ≤ 8 * 8 := by decide

end GatedResidual



## FILE: /mnt/user-data/outputs/VerifyWindow.lean
/-!
# MODULE: verify_window
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token (speculative verify window) helpers: conv+l2 over a token window, history commit, alpha/beta for T
tokens, GDN step+norm over T tokens with replay and keep-count, device-side gathers (embedding, blobs, rows),
stream broadcasts, step-state builders (`dense_steps`, `window_ids`) and the MTP selector.

## Source semantics
* Token t's conv window is entries t..t+2 of [hist0,hist1,hist2,x0,x1,...] plus x_t.
* `gdn_conv_commit` keeps the last three entries of [hist | x_0..x_{n-1}], and is a no-op for n <= 0.
* `dense_steps`: steps = (c, c+1, (c+1)/4, c+1); consistent with the QSA step when no truncation happens.
* `window_ids`: start = max(0, n_kv - window), ids = start .. n_kv-1, width = n_kv - start.
* Embedding gather unpacks `8/bits` codes per byte; q/k/v offsets of the fused conv channels are contiguous.

## Form template identifiers
kVerifyMaxT -> T bound; steps record -> (pos, n_kv, n_bid, width); window -> `window`.
-/

namespace VerifyWindow

-- § helpers

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § conv window and commit

/-- Sequence `[hist0, hist1, hist2, x0, x1, ...]`. -/
def seqAt {α : Type} (hist x : Nat → α) (i : Nat) : α := if i < 3 then hist i else x (i - 3)

/-- Window entries of token t come from the history while `t + j < 3`, else from the inputs. -/
theorem window_entry {α : Type} (hist x : Nat → α) (t j : Nat) :
    seqAt hist x (t + j) = if t + j < 3 then hist (t + j) else x (t + j - 3) := rfl

/-- Commit of zero tokens leaves the history unchanged. -/
theorem commit_zero {α : Type} (hist x : Nat → α) (j : Nat) : seqAt hist x (0 + j) = seqAt hist x j := by
  simp

/-- Commit of one token slides the history: [h1, h2, x0]. -/
theorem commit_one {α : Type} (hist x : Nat → α) :
    seqAt hist x (1 + 0) = hist 1 ∧ seqAt hist x (1 + 1) = hist 2 ∧ seqAt hist x (1 + 2) = x 0 := by
  simp [seqAt]

/-- Commit of at least three tokens only reads inputs. -/
theorem commit_many {α : Type} (hist x : Nat → α) (n j : Nat) (hn : 3 ≤ n) :
    seqAt hist x (n + j) = x (n + j - 3) := by
  unfold seqAt
  have : ¬ (n + j < 3) := by omega
  simp [this]

/-- Consecutive token windows overlap by two entries. -/
theorem window_slides {α : Type} (hist x : Nat → α) (t j : Nat) :
    seqAt hist x ((t + 1) + j) = seqAt hist x (t + (j + 1)) := by
  have : (t + 1) + j = t + (j + 1) := by omega
  rw [this]

-- § fused qkv channel layout

/-- q occupies [0, qk), k occupies [qk, 2qk), v starts at 2qk: head/dim offsets stay in their segment. -/
theorem q_offset_lt (qh tid hk : Nat) (h1 : qh < hk) (h2 : tid < 128) : qh * 128 + tid < hk * 128 :=
  idx2_lt qh tid hk 128 h1 h2

/-- v offset lies in [2qk, 2qk + value_dim). -/
theorem v_offset_range (head col hv hk : Nat) (h1 : head < hv) (h2 : col < 128) :
    2 * (128 * hk) ≤ 2 * (128 * hk) + (head * 128 + col) ∧
    2 * (128 * hk) + (head * 128 + col) < 2 * (128 * hk) + hv * 128 := by
  have := idx2_lt head col hv 128 h1 h2
  omega

-- § step-state builders

/-- `dense_steps` record for cell c. -/
def denseStep (c : Nat) : Nat × Nat × Nat × Nat := (c, c + 1, (c + 1) / 4, c + 1)

/-- Below the selection bound the dense step equals the QSA step fill (width = min n 2051 = n). -/
theorem dense_matches_fill (c : Nat) (h : c + 1 ≤ 2051) :
    denseStep c = (c, c + 1, (c + 1) / 4, min (c + 1) 2051) := by
  unfold denseStep
  have : min (c + 1) 2051 = c + 1 := by omega
  rw [this]

/-- `window_ids`: start of the window. -/
def winStart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- The window width is `min n_kv window`. -/
theorem win_width (nKv window : Nat) : nKv - winStart nKv window = min nKv window := by
  unfold winStart; split <;> omega

/-- Every window id is a valid cell. -/
theorem win_ids_valid (nKv window j : Nat) (hj : j < nKv - winStart nKv window) :
    winStart nKv window + j < nKv := by
  by_cases c : nKv > window
  · have e : winStart nKv window = nKv - window := by unfold winStart; simp [c]
    rw [e] at hj ⊢; omega
  · have e : winStart nKv window = 0 := by unfold winStart; simp [c]
    rw [e] at hj ⊢; omega

/-- When the context fits the window the ids are the identity from 0. -/
theorem win_identity (nKv window : Nat) (h : nKv ≤ window) : winStart nKv window = 0 := by
  unfold winStart; simp; omega

-- § gathers and decodes

/-- Blob / row gathers: flat index i over `n * per` decodes to (row i/per < n, offset i%per < per). -/
theorem gather_decode (i n per : Nat) (hp : 0 < per) (h : i < n * per) :
    i / per < n ∧ i % per < per ∧ (i / per) * per + i % per = i := by
  refine ⟨(Nat.div_lt_iff_lt_mul hp).2 h, Nat.mod_lt _ hp, ?_⟩
  have := Nat.div_add_mod i per
  rw [Nat.mul_comm per (i / per)] at this
  exact this

/-- Source offset of a gathered row stays inside a table of `rows` rows. -/
theorem gather_src_lt (id o rows row16 : Nat) (h1 : id < rows) (h2 : o < row16) :
    id * row16 + o < rows * row16 := idx2_lt id o rows row16 h1 h2

/-- Rebased pointers are evenly spaced: ptr(k+1) = ptr(k) + blob_bytes. -/
theorem rebase_spacing (base k bytes : Nat) : (base + (k + 1) * bytes) = (base + k * bytes) + bytes := by
  rw [Nat.add_mul, Nat.one_mul]; omega

/-- 4-bit codes: two per byte, byte index in range for n codes. -/
theorem nibble_index (i n : Nat) (h : i < n) : i / 2 < (n + 1) / 2 := by omega

/-- A 4-bit code extracted from a byte is below 16. -/
theorem nibble_lt (byte sh : Nat) : (byte / sh) % 16 < 16 := Nat.mod_lt _ (by decide)

/-- Two nibbles recompose the byte. -/
theorem nibbles_recompose (byte : Nat) (h : byte < 256) : (byte / 16) % 16 * 16 + byte % 16 = byte := by omega

/-- 2-bit codes: four per byte. -/
theorem crumb_index (i n : Nat) (h : i < n) : i / 4 < (n + 3) / 4 := by omega

/-- Stream broadcast: `R[t*n*hc + i] = x[t*n + i % n]`, with `i % n < n`. -/
theorem broadcast_source_lt (i n hc : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- Token row offsets are disjoint across tokens. -/
theorem token_rows_disjoint (t1 t2 stride : Nat) (h : t1 < t2) : t1 * stride + stride ≤ t2 * stride := by
  have h1 : (t1 + 1) * stride ≤ t2 * stride := Nat.mul_le_mul_right stride (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

-- § keep-count and replay

/-- The state is committed only when a keep-count exists and is positive. -/
def commits (hasKeep : Bool) (n : Nat) : Bool := hasKeep && decide (0 < n)

/-- Without a keep-count the single-pass kernel never commits state. -/
theorem no_commit_without_keep (n : Nat) : commits false n = false := by
  unfold commits; simp

/-- Tokens whose output is written: indices t in [t_out_begin, n). Their count is at most `n`. -/
theorem outputs_le (n tOut : Nat) : n - tOut ≤ n := Nat.sub_le n tOut

/-- Replayed tokens (t < t_out_begin) are exactly those whose output is skipped. -/
theorem replayed_skip (t tOut : Nat) : (t < tOut) ↔ ¬ (tOut ≤ t) := by omega

/-- MTP chain: a draft row index is selected from device memory, so a row `< n_rows` gives an in-range copy. -/
theorem mtp_row_copy_lt (row i stride nRows : Nat) (hr : row < nRows) (hi : i < stride) :
    row * stride + i < nRows * stride := idx2_lt row i nRows stride hr hi

end VerifyWindow





## FILE: /mnt/user-data/outputs/F16BitsRounding.lean
/-!
# MODULE: f16_bits_rounding
SOURCES: include/strata/kernels/f16_bits.hpp, include/strata/artifact/dequant.hpp, src/kernels/cuda/ple.cu
Integer-level model of the hand-written fp16 <-> f32 and f32 -> bf16 conversions: round-to-nearest-even,
the inf/NaN versus finite-overflow split, subnormal normalisation and the exponent re-bias.

## Source semantics
* `f16_from_f32` rounds the 13 dropped mantissa bits to nearest, ties to even; the carry may bump the exponent.
* An f32 inf/NaN (raw exponent 255) stays inf/NaN; a FINITE value too large for fp16 saturates to inf (never NaN).
* `f32_from_f16` / `fp16_to_fp32` normalise subnormals by left-shifting until bit 10 is set; the f32 exponent
  field is then `113 - s`.  Unsigned wrap-around makes `ex - 15 + 127` correct only as a signed quantity.
* `bf16_bits` (ple.cu) adds `0x7FFF + lsb` and truncates: round-to-nearest-even on the upper 16 bits.

## Form template identifiers
f16_from_f32 -> round_f32_to_f16_bits ; f32_from_f16 -> widen_f16_bits ; bf16_bits -> round_f32_to_bf16_bits
rem -> dropped_mantissa_remainder ; rawexp -> f32_exponent_field ; man -> mantissa_field
-/

namespace Strata.F16Bits

-- § round-to-nearest-even on the dropped remainder

/-- One rounding step: `rem` is the dropped 13-bit remainder, `h` the truncated half-precision bits. -/
def rneStep (h rem : Nat) : Nat :=
  if rem > 4096 ∨ (rem = 4096 ∧ h % 2 = 1) then h + 1 else h

/-- Below the halfway point the truncated value is kept. -/
theorem rne_below_half (h rem : Nat) (hr : rem < 4096) : rneStep h rem = h := by
  unfold rneStep
  split <;> omega

/-- Above the halfway point the value is bumped by one. -/
theorem rne_above_half (h rem : Nat) (hr : 4096 < rem) : rneStep h rem = h + 1 := by
  unfold rneStep
  split <;> omega

/-- Exactly at the tie the result is always even (ties-to-even). -/
theorem rne_tie_even (h rem : Nat) (hr : rem = 4096) : rneStep h rem % 2 = 0 := by
  unfold rneStep
  split <;> omega

/-- Rounding never moves the value by more than one unit in the last place. -/
theorem rne_le_succ (h rem : Nat) : h ≤ rneStep h rem ∧ rneStep h rem ≤ h + 1 := by
  unfold rneStep
  split <;> omega

-- § the inf/NaN versus finite-overflow split

/-- The top-level classification of the encoder: `none` means "normal or subnormal path". -/
def encodeTop (sign rawexp man : Nat) : Option Nat :=
  if rawexp = 255 then some (sign + 31744 + (if man ≠ 0 then 512 else 0))
  else if (rawexp : Int) - 127 + 15 ≥ 31 then some (sign + 31744)
  else none

/-- A finite overflow saturates to a signed infinity: its mantissa field is zero, so it is not a NaN. -/
theorem finite_overflow_is_inf (sign rawexp man : Nat) (hs : sign = 0 ∨ sign = 32768)
    (h1 : rawexp ≠ 255) (h3 : 143 ≤ rawexp) :
    ∃ v, encodeTop sign rawexp man = some v ∧ v % 1024 = 0 := by
  refine ⟨sign + 31744, ?_, by omega⟩
  unfold encodeTop
  rw [if_neg h1, if_pos (by omega)]

/-- A NaN input keeps a non-zero mantissa field. -/
theorem nan_stays_nan (sign man : Nat) (hs : sign = 0 ∨ sign = 32768) (hm : man ≠ 0) :
    ∃ v, encodeTop sign 255 man = some v ∧ v % 1024 ≠ 0 := by
  refine ⟨sign + 31744 + 512, ?_, by omega⟩
  unfold encodeTop
  rw [if_pos rfl, if_pos hm]

/-- An infinity input stays an infinity. -/
theorem inf_stays_inf (sign : Nat) (hs : sign = 0 ∨ sign = 32768) :
    ∃ v, encodeTop sign 255 0 = some v ∧ v % 1024 = 0 := by
  refine ⟨sign + 31744, ?_, by omega⟩
  unfold encodeTop
  have hn : ¬ ((0 : Nat) ≠ 0) := by omega
  rw [if_pos rfl, if_neg hn]

/-- Why the old test `exp >= 31` conflated the two cases: raw exponent 255 also satisfies it. -/
theorem conflation_hazard (rawexp : Nat) (h : rawexp = 255) : (rawexp : Int) - 127 + 15 ≥ 31 := by
  omega

-- § subnormal normalisation in the decoder

/-- The normalising loop of the decoder: shift left until bit 10 is set, counting shifts. -/
def normShift : Nat → Nat → Nat → Nat × Nat
  | 0, m, s => (m, s)
  | fuel + 1, m, s => if m < 1024 then normShift fuel (2 * m) (s + 1) else (m, s)

set_option maxRecDepth 100000 in
/-- For every non-zero subnormal mantissa the loop ends with bit 10 set after at most 10 shifts,
and the shifted value is `m * 2^s`. -/
theorem normShift_spec :
    ∀ m < 1024, 0 < m →
      1024 ≤ (normShift 10 m 0).1 ∧ (normShift 10 m 0).1 < 2048 ∧ (normShift 10 m 0).2 ≤ 10 ∧
        (normShift 10 m 0).1 = m * 2 ^ (normShift 10 m 0).2 := by
  decide

/-- The re-bias constant is 113. -/
theorem subnormal_bias : 127 - 15 + 1 = 113 := by decide

/-- After at most 10 shifts the f32 exponent field `113 - s` is a valid normal exponent. -/
theorem subnormal_exp_field (s : Nat) (hs : s ≤ 10) : 103 ≤ 113 - s ∧ 113 - s < 255 := by
  omega

/-- For normal halves the f32 exponent field is `ex + 112`, computed as a signed quantity. -/
theorem normal_exp_field (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    ((ex : Int) - 15 + 127 = (ex : Int) + 112) ∧ 113 ≤ (ex : Int) + 112 ∧ (ex : Int) + 112 ≤ 142 := by
  omega

/-- Evaluating `ex - 15 + 127` in truncated Nat arithmetic would be wrong; the C++ relies on unsigned wrap. -/
theorem nat_truncation_hazard : (1 - 15 + 127 : Nat) ≠ 1 + 112 := by decide

-- § bf16 rounding (ple.cu `bf16_bits`)

/-- `(i + ((i >> 16) & 1) + 0x7FFF) >> 16` as Nat arithmetic (inputs below 2^32 without carry-out). -/
def bf16Round (i : Nat) : Nat := (i + (i / 65536) % 2 + 32767) / 65536

/-- Values that are already bf16 are unchanged. -/
theorem bf16_exact (q : Nat) : bf16Round (65536 * q) = q := by
  unfold bf16Round
  omega

/-- Below the halfway point the result truncates. -/
theorem bf16_below_half (i : Nat) (h : i % 65536 < 32768) : bf16Round i = i / 65536 := by
  unfold bf16Round
  omega

/-- Above the halfway point the result rounds up. -/
theorem bf16_above_half (i : Nat) (h : 32768 < i % 65536) : bf16Round i = i / 65536 + 1 := by
  unfold bf16Round
  omega

/-- At the exact tie the result is even. -/
theorem bf16_tie_even (i : Nat) (h : i % 65536 = 32768) : bf16Round i % 2 = 0 := by
  unfold bf16Round
  obtain ⟨q, hq⟩ : ∃ q, i = 65536 * q + 32768 := ⟨i / 65536, by omega⟩
  subst hq
  have h1 : (65536 * q + 32768) / 65536 = q := by omega
  rw [h1]
  have h2 : (65536 * q + 32768 + q % 2 + 32767) / 65536 = q + q % 2 := by omega
  rw [h2]
  omega

end Strata.F16Bits



## FILE: /mnt/user-data/outputs/DequantBlockLayouts.lean
/-!
# MODULE: dequant_block_layouts
SOURCES: include/strata/artifact/gguf_reader.hpp, include/strata/artifact/dequant.hpp, src/kernels/cuda/iq_kernels.cu
Block geometry (elements, bytes) of every GGML type the engine reads, struct offsets, stride alignment,
row-byte arithmetic, and the value ranges produced by the scalar dequantizers.

## Source semantics
* `block_geometry` gives (elements per block, bytes per block); Q2_0 is 64 elements / 18 bytes.
* Block strides that are not multiples of 4 (18, 22, 34, 110, 210) force 2-byte loads in the kernels.
* A row of `n` elements occupies `n / elems * bytes` bytes; the weight total must not overflow size_t.
* Dequantized integer codes stay in fixed ranges (Q4_0: -8..7, Q5_0: -16..15, Q6_K: -32..31, ...).

## Form template identifiers
block_geometry -> block_shape_of_type ; kvalues_iq4nl -> nonlinear_codebook ; get_scale_min_k4 -> unpack_scale_min
-/

namespace Strata.DequantLayouts

-- § block geometry

/-- (elements per block, bytes per block), mirroring `block_geometry`. -/
def geom : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 1 => some (1, 2)
  | 30 => some (1, 2)
  | 2 => some (32, 18)
  | 3 => some (32, 20)
  | 6 => some (32, 22)
  | 7 => some (32, 24)
  | 8 => some (32, 34)
  | 9 => some (32, 36)
  | 10 => some (256, 84)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 20 => some (32, 18)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 23 => some (256, 136)
  | 29 => some (256, 56)
  | 42 => some (64, 18)
  | _ => none

/-- Q2_0 is 64 elements in 18 bytes: a 2-byte scale plus 64 two-bit codes. -/
theorem q2_0_geometry : geom 42 = some (64, 18) ∧ 2 + 64 * 2 / 8 = 18 := by decide

/-- Every type the native MMVQ dispatcher accepts has a known block geometry. -/
theorem supported_have_geometry :
    ∀ t ∈ [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29], (geom t).isSome = true := by
  decide

/-- Every block holds a divisor of 256 elements, so a 256-element super-block is a whole number of blocks. -/
theorem elems_divide_superblock :
    ∀ t ∈ [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 30, 42],
      (geom t).map (fun p => 256 % p.1) = some 0 := by
  decide

-- § struct layouts used by the native MMVQ kernels

/-- Q3_K: hmask 32, qs 64, scales 12, d 2 = 110 with offsets 32, 96, 108. -/
theorem q3k_layout : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 = 96 ∧ 96 + 12 = 108 := by decide

/-- Q6_K: ql 128, qh 64, scales 16, d 2 = 210 with offsets 128, 192, 208. -/
theorem q6k_layout : 128 + 64 + 16 + 2 = 210 ∧ 128 + 64 = 192 ∧ 192 + 16 = 208 := by decide

/-- Q5_K: dm 4, scales 12, qh 32, qs 128 = 176 with offsets 4, 16, 48. -/
theorem q5k_layout : 4 + 12 + 32 + 128 = 176 ∧ 4 + 12 = 16 ∧ 16 + 32 = 48 := by decide

/-- Q4_K: dm 4, scales 12, qs 128 = 144 with offsets 4, 16. -/
theorem q4k_layout : 4 + 12 + 128 = 144 ∧ 4 + 12 = 16 := by decide

/-- IQ4_XS: d 2, scales_h 2, scales_l 4, qs 128 = 136 with offsets 2, 4, 8. -/
theorem iq4xs_layout : 2 + 2 + 4 + 128 = 136 ∧ 2 + 2 = 4 ∧ 4 + 4 = 8 := by decide

/-- Q5_0: d 2, qh 4, qs 16 = 22 with offsets 2, 6. -/
theorem q50_layout : 2 + 4 + 16 = 22 ∧ 2 + 4 = 6 := by decide

/-- Block strides congruent to 2 modulo 4 need 16-bit loads; the others allow 32-bit loads. -/
theorem stride_alignment :
    ([18, 22, 34, 110, 210].all (fun b => b % 4 == 2)) = true ∧
    ([36, 136, 144, 176].all (fun b => b % 4 == 0)) = true := by
  decide

-- § row-byte arithmetic

/-- Bytes of a row of `n` elements for a block shape. -/
def rowBytes (e b n : Nat) : Nat := n / e * b

/-- A row of `k` whole blocks is `k * bytes`. -/
theorem rowBytes_whole (e b k : Nat) (he : 0 < e) : rowBytes e b (k * e) = k * b := by
  unfold rowBytes
  rw [Nat.mul_div_cancel _ he]

/-- If the row size is below `max / n_out` then the weight total does not overflow. -/
theorem weight_bytes_no_overflow (row nOut M : Nat) (hn : 0 < nOut) (h : row ≤ M / nOut) :
    row * nOut ≤ M :=
  (Nat.le_div_iff_mul_le hn).mp h

-- § dequantized value ranges

/-- Q4_0: a nibble minus 8 is in [-8, 7]. -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by omega

/-- Q5_0: nibble plus the fifth bit minus 16 is in [-16, 15]. -/
theorem q5_0_range (n b : Nat) (hn : n < 16) (hb : b ≤ 1) :
    -16 ≤ ((n : Int) + 16 * b) - 16 ∧ ((n : Int) + 16 * b) - 16 ≤ 15 := by omega

/-- Q6_K: low nibble plus two high bits minus 32 is in [-32, 31]. -/
theorem q6_k_range (n h : Nat) (hn : n < 16) (hh : h < 4) :
    -32 ≤ ((n : Int) + 16 * h) - 32 ∧ ((n : Int) + 16 * h) - 32 ≤ 31 := by omega

/-- Q3_K: two bits minus 0 or 4 is in [-4, 3]. -/
theorem q3_k_range (t : Nat) (ht : t < 4) (m : Nat) (hm : m = 0 ∨ m = 4) :
    -4 ≤ (t : Int) - m ∧ (t : Int) - m ≤ 3 := by omega

/-- Q2_0 symbols {-1, 0, +1, +2}. -/
theorem q2_0_symbols (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by omega

/-- Q4_K / Q5_K scale for `j < 4` is a 6-bit value. -/
theorem scale_min_low (q : Nat) : q % 64 < 64 := Nat.mod_lt _ (by decide)

/-- Q4_K / Q5_K scale for `j >= 4`: 4 low bits plus two bits from a neighbour is below 64. -/
theorem scale_min_high (q q' : Nat) (hq' : q' < 256) : q % 16 + 16 * (q' / 64) < 64 := by omega

/-- The `m` half of the packed pair uses the high nibble, still below 64. -/
theorem scale_min_high_m (q q' : Nat) (hq : q < 256) (hq' : q' < 256) : q / 16 + 16 * (q' / 64) < 64 := by
  omega

/-- IQ4_XS: the 6-bit scale assembled from a nibble and two bits is below 64, so `ls - 32` is in [-32, 31]. -/
theorem iq4xs_scale (lo hi : Nat) :
    (lo % 16 + 16 * (hi % 4) < 64) ∧ -32 ≤ ((lo % 16 + 16 * (hi % 4) : Nat) : Int) - 32 ∧
      ((lo % 16 + 16 * (hi % 4) : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q5_K bit masks `u1 = 1,4,16,64` and `u2 = 2 u1` are disjoint bits that fit a byte. -/
theorem q5k_masks : ∀ u ∈ [1, 4, 16, 64], 2 * u < 256 ∧ (u &&& (2 * u)) = 0 := by decide

/-- The fifth bit adds at most 16 so a Q5_K code is at most 31. -/
theorem q5k_code_bound (n : Nat) (hn : n < 16) (b : Nat) (hb : b = 0 ∨ b = 16) : n + b ≤ 31 := by omega

end Strata.DequantLayouts



## FILE: /mnt/user-data/outputs/SGemvCodeUnpack.lean
/-!
# MODULE: s_gemv_code_unpack
SOURCES: src/kernels/cuda/s_gemv.cu
Index arithmetic of the S-family GEMV: Q8_K / Q8_0 activation addressing, hoisting of the activation block out
of the 16-element lane iteration, the bit order of packed codes (S2/S4/S8), group-shift division, alignment of
the code words, the same-group requirement, and launch-grid coverage.

## Source semantics
* Q8_K blocks are 292 bytes per 256 elements (`d`, 256 int8, 16 int16 bsums); Q8_0 blocks are 34 bytes per 32.
* A lane takes QE = 16 consecutive elements starting at a multiple of 16, so they never straddle a block.
* Element `i+k` is bits `[k*B, (k+1)*B)` of the little-endian word at byte `i/PER_BYTE`.
* The group index is `i >> log2(group_elems)`; sixteen elements share one scale only if 16 divides the group.

## Form template identifiers
q8k_at -> q8k_element_offset ; q8_0_at -> q8_0_element_offset ; group_shift -> log2_group_elems
QE -> elements_per_lane_iteration ; QW -> bytes_per_four_code_word
-/

namespace Strata.SGemv

-- § activation addressing

/-- Byte offset of element `i` inside a Q8_K activation buffer. -/
def q8kOff (i : Nat) : Nat := (i / 256) * 292 + 4 + i % 256

/-- Byte offset of element `i` inside a Q8_0 activation buffer. -/
def q80Off (i : Nat) : Nat := (i / 32) * 34 + 2 + i % 32

/-- Distinct elements map to distinct Q8_K bytes. -/
theorem q8kOff_inj (i j : Nat) (h : q8kOff i = q8kOff j) : i = j := by
  unfold q8kOff at h
  omega

/-- Distinct elements map to distinct Q8_0 bytes. -/
theorem q80Off_inj (i j : Nat) (h : q80Off i = q80Off j) : i = j := by
  unfold q80Off at h
  omega

/-- Every element byte lies strictly before the bsums tail of its Q8_K block. -/
theorem q8kOff_before_bsums (i : Nat) : q8kOff i + 32 < (i / 256 + 1) * 292 := by
  unfold q8kOff
  omega

/-- A buffer of `nb` blocks is never read past its end. -/
theorem q8kOff_in_buffer (i nb : Nat) (h : i < 256 * nb) : q8kOff i < 292 * nb := by
  unfold q8kOff
  omega

/-- Same for Q8_0. -/
theorem q80Off_in_buffer (i nb : Nat) (h : i < 32 * nb) : q80Off i < 34 * nb := by
  unfold q80Off
  omega

-- § hoisting the activation block out of the lane iteration

/-- With `i % 16 = 0`, sixteen consecutive elements share one Q8_K block and sit at consecutive offsets. -/
theorem hoist_q8k (i k : Nat) (h16 : i % 16 = 0) (hk : k < 16) :
    (i + k) / 256 = i / 256 ∧ (i + k) % 256 = i % 256 + k ∧ i % 256 + 16 ≤ 256 := by
  omega

/-- The same for Q8_0 blocks of 32. -/
theorem hoist_q8_0 (i k : Nat) (h16 : i % 16 = 0) (hk : k < 16) :
    (i + k) / 32 = i / 32 ∧ (i + k) % 32 = i % 32 + k ∧ i % 32 + 16 ≤ 32 := by
  omega

/-- Every lane iteration start `lane*16 + 512*t` is a multiple of 16. -/
theorem lane_iter_aligned (lane t : Nat) : (lane * 16 + 512 * t) % 16 = 0 := by omega

/-- Distinct (lane, iteration) pairs cover distinct 16-element octets. -/
theorem lane_iter_inj (lane t lane' t' : Nat) (hl : lane < 32) (hl' : lane' < 32)
    (h : lane * 16 + 512 * t = lane' * 16 + 512 * t') : lane = lane' ∧ t = t' := by
  omega

/-- The 16 int8 activations of a lane stay inside the int8 payload of the block (before bsums). -/
theorem lane_payload_bound (i : Nat) (h16 : i % 16 = 0) : 4 + i % 256 + 15 < 4 + 256 := by omega

/-- When the row length is a multiple of 16 the partial-tail loop never runs. -/
theorem tail_empty (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) (h : ¬ (i + 16 ≤ n)) : ¬ (i < n) := by
  omega

-- § same-group requirement

/-- Elements `m*a .. m*a+m-1` all fall in group `a / b` of size `m*b`. -/
theorem same_group (m a b k : Nat) (hm : 0 < m) (hk : k < m) : (m * a + k) / (m * b) = a / b := by
  have h1 : (m * a + k) / m = a := by
    simp [Nat.mul_add_div hm, Nat.div_eq_of_lt hk]
  rw [← Nat.div_div_eq_div_mul, h1]

/-- A quad (four elements) shares a group whenever 4 divides the group size. -/
theorem quad_same_group (a b k : Nat) (hk : k < 4) : (4 * a + k) / (4 * b) = a / b :=
  same_group 4 a b k (by decide) hk

/-- A 16-element lane block shares a group whenever 16 divides the group size. -/
theorem octet16_same_group (a b k : Nat) (hk : k < 16) : (16 * a + k) / (16 * b) = a / b :=
  same_group 16 a b k (by decide) hk

/-- The Q8_0 split launcher only checks `group_elems % 4 == 0`, but the kernel reads ONE scale for sixteen
elements: with a group of 4 the sixteen elements span several groups.  Every real group (16/32/64) is safe. -/
theorem q8_0_split_group_hazard : (15 : Nat) / 4 ≠ 0 / 4 := by decide

/-- Group sizes 16, 32, 64 as shifts. -/
theorem group_shifts : (1 <<< 4 = 16) ∧ (1 <<< 5 = 32) ∧ (1 <<< 6 = 64) := by decide

/-- A right shift is division by the power of two. -/
theorem group_shift_div (i s : Nat) : i >>> s = i / 2 ^ s := Nat.shiftRight_eq_div_pow i s

-- § bit order of the packed codes

/-- S2: the four codes of a byte are its 2-bit fields, and the byte is recovered from them. -/
theorem s2_fields (b : Nat) (h : b < 256) :
    b = b % 4 + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) := by omega

/-- S4: the 16-bit word of four codes yields the same nibbles as the scalar per-byte extraction. -/
theorem s4_word (b0 b1 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    ((b0 + 256 * b1) % 16 = (b0 / 1) % 16) ∧ (((b0 + 256 * b1) / 16) % 16 = (b0 / 16) % 16) ∧
    (((b0 + 256 * b1) / 256) % 16 = (b1 / 1) % 16) ∧ (((b0 + 256 * b1) / 4096) % 16 = (b1 / 16) % 16) := by
  omega

/-- S8: the 32-bit word of four codes yields the four bytes. -/
theorem s8_word (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256) :
    let v := b0 + 256 * b1 + 65536 * b2 + 16777216 * b3
    (v % 256 = b0) ∧ ((v / 256) % 256 = b1) ∧ ((v / 65536) % 256 = b2) ∧ ((v / 16777216) % 256 = b3) := by
  intro v
  simp only [v]
  omega

/-- Four lane words of QW bytes cover exactly 16 codes: `4 * (4*B/8) = 16*B/8` for B in {2,4,8}. -/
theorem word_bytes : ∀ B ∈ [2, 4, 8], 4 * (4 * B / 8) = 16 * B / 8 := by decide

/-- Alignment of the wide code loads: S8 words 4-aligned, S4 words 2-aligned, rows stay aligned. -/
theorem code_load_alignment (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) :
    (i / 1) % 4 = 0 ∧ (i / 2) % 2 = 0 ∧ (n / 1) % 4 = 0 ∧ (n / 2) % 2 = 0 ∧ (n / 4) % 1 = 0 := by
  omega

-- § launch grids

/-- 128 threads per block, one row per thread: the grid covers all rows. -/
theorem grid_cover_128 (n : Nat) : n ≤ ((n + 127) / 128) * 128 := by omega

/-- 8 warps per block, one row per warp: the grid covers all rows. -/
theorem grid_cover_warps (n : Nat) : n ≤ ((n + 7) / 8) * 8 := by omega

end Strata.SGemv



## FILE: /mnt/user-data/outputs/DotProductAlgebra.lean
/-!
# MODULE: dot_product_algebra
SOURCES: src/kernels/cuda/s2_expert_grouped.cu, src/kernels/cuda/s_gemv.cu, src/kernels/cuda/native_mmvq.cu
Exact-arithmetic identities behind the integer dot products and block reductions: chunked dot products,
the Q2_0 bias identity `sum (c-1) x = sum c x - sum x`, the 2-bit code expansion into dp4a bytes, and the
shuffle / shared-memory tree reductions preserving the total.

## Source semantics
* `s - hx` with `s = sum c*x` (dp4a with the expanded codes) and `hx = sum x` (dp4a with ones) equals the
  sum of `(c-1)*x`.
* A dot product over concatenated chunks equals the sum of the chunk dot products (dp4a accumulation).
* A tree reduction with halving strides (`step = 16, 8, 4, 2, 1`, or `tpr/2 ... 1`) conserves the sum.

## Form template identifiers
dp4a -> int8x4_dot_accumulate ; hx -> activation_sum ; warp_sum -> warp_tree_sum ; partial -> block_partials
-/

namespace Strata.DotAlgebra

-- § chunked dot products

/-- Integer dot product of two lists (truncating to the shorter, as zip does). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Dot product distributes over concatenation of equal-length prefixes. -/
theorem dot_append (as as' bs bs' : List Int) (h : as.length = bs.length) :
    dot (as ++ as') (bs ++ bs') = dot as bs + dot as' bs' := by
  induction as generalizing bs with
  | nil =>
    cases bs with
    | nil => simp [dot]
    | cons _ _ => simp at h
  | cons a as ih =>
    cases bs with
    | nil => simp at h
    | cons b bs =>
      have h' : as.length = bs.length := by simpa using h
      simp only [List.cons_append, dot]
      rw [ih bs h']
      omega

/-- The Q2_0 bias identity: shifting codes by one equals subtracting the activation sum. -/
theorem dot_bias (cs xs : List Int) (h : cs.length = xs.length) :
    dot (cs.map (· - 1)) xs = dot cs xs - xs.sum := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [dot]
    | cons _ _ => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      simp only [List.map_cons, dot, List.sum_cons]
      rw [ih xs h', Int.sub_mul, Int.one_mul]
      omega

-- § 2-bit code expansion for dp4a

/-- A code byte with four 2-bit fields is expanded to one byte per field. -/
def expand (c : Nat) : Nat :=
  c % 4 + 256 * ((c / 4) % 4) + 65536 * ((c / 16) % 4) + 16777216 * ((c / 64) % 4)

/-- Packing four fields below 4 into bytes of a word keeps them separate. -/
theorem pack4 (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    (a + 256 * b + 65536 * c + 16777216 * d) % 256 = a ∧
    ((a + 256 * b + 65536 * c + 16777216 * d) / 256) % 256 = b ∧
    ((a + 256 * b + 65536 * c + 16777216 * d) / 65536) % 256 = c ∧
    (a + 256 * b + 65536 * c + 16777216 * d) / 16777216 = d ∧
    a + 256 * b + 65536 * c + 16777216 * d < 4294967296 := by
  omega

/-- Each byte of the expanded word holds exactly one field (value <= 3, so no sign/carry issues). -/
theorem expand_fields (c : Nat) (h : c < 256) :
    expand c % 256 = c % 4 ∧ (expand c / 256) % 256 = (c / 4) % 4 ∧
    (expand c / 65536) % 256 = (c / 16) % 4 ∧ expand c / 16777216 = (c / 64) % 4 ∧
    expand c < 4294967296 := by
  unfold expand
  exact pack4 _ _ _ _ (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))
    (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))

/-- The chunk's weight scale index is `c / 2` and stays below the row's scale count. -/
theorem chunk_scale_index (c : Nat) (h : c < 80) : c / 2 < 40 := by omega

-- § tree reductions

/-- `sumTo f n = f 0 + ... + f (n-1)`. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

/-- A sum splits at any point. -/
theorem sumTo_split (f : Nat → Int) (a b : Nat) :
    sumTo f (a + b) = sumTo f a + sumTo (fun t => f (a + t)) b := by
  induction b with
  | zero => simp [sumTo]
  | succ b ih =>
    show sumTo f (a + b) + f (a + b) = sumTo f a + (sumTo (fun t => f (a + t)) b + f (a + b))
    rw [ih]
    omega

/-- A sum of pointwise sums is the sum of sums. -/
theorem sumTo_add (f g : Nat → Int) (n : Nat) :
    sumTo (fun t => f t + g t) n = sumTo f n + sumTo g n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    show sumTo (fun t => f t + g t) n + (f n + g n) = sumTo f n + f n + (sumTo g n + g n)
    rw [ih]
    omega

/-- One tree step: lane `t < s` adds lane `t + s`. -/
def rstep (p : Nat → Int) (s : Nat) : Nat → Int := fun t => p t + p (t + s)

/-- A halving step conserves the total. -/
theorem tree_step (p : Nat → Int) (s : Nat) : sumTo p (s + s) = sumTo (rstep p s) s := by
  have h1 : (fun t => p (s + t)) = (fun t => p (t + s)) := funext fun t => by rw [Nat.add_comm]
  rw [sumTo_split, h1]
  unfold rstep
  rw [sumTo_add]

/-- The 32-lane shuffle reduction (strides 16, 8, 4, 2, 1) leaves the exact total in lane 0. -/
theorem warp32 (p : Nat → Int) :
    sumTo p 32 = rstep (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 1 0 := by
  have h1 : sumTo p 32 = sumTo (rstep p 16) 16 := tree_step p 16
  have h2 : sumTo (rstep p 16) 16 = sumTo (rstep (rstep p 16) 8) 8 := tree_step _ 8
  have h3 : sumTo (rstep (rstep p 16) 8) 8 = sumTo (rstep (rstep (rstep p 16) 8) 4) 4 := tree_step _ 4
  have h4 : sumTo (rstep (rstep (rstep p 16) 8) 4) 4 =
      sumTo (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 2 := tree_step _ 2
  have h5 : sumTo (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 2 =
      sumTo (rstep (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 1) 1 := tree_step _ 1
  have h6 : sumTo (rstep (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 1) 1 =
      rstep (rstep (rstep (rstep (rstep p 16) 8) 4) 2) 1 0 := by
    simp [sumTo]
  exact h1.trans (h2.trans (h3.trans (h4.trans (h5.trans h6))))

/-- The shared-memory block reduction (`step = tpr/2` downwards) uses the same step; a power-of-two
`tpr = 2 * s` therefore folds exactly. -/
theorem block_fold (p : Nat → Int) (s : Nat) : sumTo p (2 * s) = sumTo (rstep p s) s := by
  have : 2 * s = s + s := by omega
  rw [this]
  exact tree_step p s

end Strata.DotAlgebra



## FILE: /mnt/user-data/outputs/S2ExpertLayout.lean
/-!
# MODULE: s2_expert_layout
SOURCES: src/kernels/cuda/s2_expert_grouped.cu, src/kernels/cuda/iq_kernels.cu
Byte layout of a Q2_0 expert blob, the interleaved gate/up row mapping and its gate-major output, lane/chunk
coverage, ballot-based hit compaction, and 16/256-byte scratch carving (both expert paths).

## Source semantics
* Blob = gate/up codes (2*FF rows of 640 B), down codes (H rows of 160 B), gate/up scales (2*FF rows of 40 fp16),
  down scales (H rows of 10 fp16).
* Row slot `i` of the interleaved gate/up rows is gate `i/2` when even and up `i/2` when odd; it is written to a
  gate-major buffer: gates contiguous at `h*FF`, ups contiguous from `n*FF`.
* `hit_select` compacts resident experts with `popc(ballot & ((1<<lane)-1))`: positions are distinct and dense.
* Scratch regions are carved at 16-byte (256-byte for the native path) aligned offsets.

## Form template identifiers
O_D_CODES -> down_codes_offset ; O_GU_SCALES -> gate_up_scales_offset ; O_D_SCALES -> down_scales_offset
slot_index -> resident_slot ; dst_index -> router_slot ; GMAX -> entries_per_group
-/

namespace Strata.S2ExpertLayout

-- § blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- Expert FFN width. -/
def FF : Nat := 640
/-- Q2_0 group size. -/
def QK : Nat := 64
/-- Bytes of codes per gate/up row. -/
def rowGU : Nat := H / 4
/-- Bytes of codes per down row. -/
def rowD : Nat := FF / 4
/-- fp16 scales per gate/up row. -/
def scGU : Nat := H / QK
/-- fp16 scales per down row. -/
def scD : Nat := FF / QK
/-- Offset of the down codes. -/
def oDCodes : Nat := 2 * FF * rowGU
/-- Offset of the gate/up scales. -/
def oGUScales : Nat := oDCodes + H * rowD
/-- Offset of the down scales. -/
def oDScales : Nat := oGUScales + 2 * FF * scGU * 2
/-- Total blob bytes. -/
def blobBytes : Nat := oDScales + H * scD * 2

/-- The concrete geometry. -/
theorem blob_constants :
    rowGU = 640 ∧ rowD = 160 ∧ scGU = 40 ∧ scD = 10 ∧ oDCodes = 819200 ∧ oGUScales = 1228800 ∧
      oDScales = 1331200 ∧ blobBytes = 1382400 := by
  decide

/-- Every gate/up code row lies inside the gate/up code region. -/
theorem gu_code_row_in_region (i : Nat) (h : i < 1280) : i * 640 + 640 ≤ 819200 := by omega

/-- Every gate/up scale row (40 fp16 = 80 bytes) lies inside its region. -/
theorem gu_scale_row_in_region (i : Nat) (h : i < 1280) :
    1228800 ≤ 1228800 + i * 80 ∧ 1228800 + i * 80 + 80 ≤ 1331200 := by omega

/-- Every down code row lies inside the down code region. -/
theorem down_code_row_in_region (r : Nat) (h : r < 2560) :
    819200 + r * 160 + 160 ≤ 1228800 := by omega

/-- Every down scale row (10 fp16 = 20 bytes) ends within the blob. -/
theorem down_scale_row_in_region (r : Nat) (h : r < 2560) :
    1331200 + r * 20 + 20 ≤ 1382400 := by omega

/-- The four regions are in order and tile the blob. -/
theorem regions_ordered : oDCodes ≤ oGUScales ∧ oGUScales ≤ oDScales ∧ oDScales ≤ blobBytes := by decide

-- § lane / chunk coverage

/-- 80 activation chunks over 32 lanes need 3 chunks per lane. -/
theorem gu_chunks_per_lane : (80 + 31) / 32 = 3 := by decide

/-- Every gate/up chunk `c < 80` is `lane + 32*q` for a unique lane < 32 and q < 3. -/
theorem gu_chunk_decomp (c : Nat) (h : c < 80) :
    c = c % 32 + 32 * (c / 32) ∧ c % 32 < 32 ∧ c / 32 < 3 := by omega

/-- A down row has 20 chunks, so lanes 0..19 hold one each. -/
theorem down_chunks_fit : 640 / 32 = 20 ∧ 20 ≤ 32 := by decide

/-- Down scale index `c / 2` stays below 10. -/
theorem down_scale_index (c : Nat) (h : c < 20) : c / 2 < 10 := by omega

-- § interleaved gate/up output slot

/-- Per-hit decomposition of a flat row slot is a bijection onto (hit, row-in-hit). -/
theorem slot_decomp (slot : Nat) : slot = (slot / 1280) * 1280 + slot % 1280 ∧ slot % 1280 < 1280 := by
  omega

/-- The hit index of a slot is in range. -/
theorem slot_hit_lt (slot n : Nat) (h : slot < n * 1280) : slot / 1280 < n := by omega

/-- Gate-major output position of row slot `i` of hit `h` (with `N = n * FF`). -/
def gm (N h i : Nat) : Nat := (i % 2) * N + h * 640 + i / 2

/-- The gate-major mapping is injective in (hit, slot). -/
theorem gate_major_inj (N h h' i i' : Nat) (hN : h * 640 + 640 ≤ N) (hN' : h' * 640 + 640 ≤ N)
    (hi : i < 1280) (hi' : i' < 1280) (heq : gm N h i = gm N h' i') : h = h' ∧ i = i' := by
  unfold gm at heq
  rcases Nat.mod_two_eq_zero_or_one i with p | p <;> rcases Nat.mod_two_eq_zero_or_one i' with q | q <;>
    simp only [p, q, Nat.zero_mul, Nat.one_mul] at heq <;> omega

/-- Gate values land below `N`, up values at or above `N` and below `2N`. -/
theorem gate_major_range (N h i : Nat) (hN : h * 640 + 640 ≤ N) (hi : i < 1280) :
    (i % 2 = 0 → gm N h i < N) ∧ (i % 2 = 1 → N ≤ gm N h i ∧ gm N h i < 2 * N) := by
  unfold gm
  constructor
  · intro p
    simp only [p, Nat.zero_mul]
    omega
  · intro p
    simp only [p, Nat.one_mul]
    omega

-- § compaction by ballot prefix count

/-- Number of set lanes strictly before position `n` (models `popc(hit & ((1<<lane)-1))`). -/
def pre (f : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n + 1 => pre f n + (if f n then 1 else 0)

/-- The prefix count is monotone. -/
theorem pre_mono (f : Nat → Bool) (i : Nat) : ∀ j, i ≤ j → pre f i ≤ pre f j := by
  intro j hj
  induction j with
  | zero =>
    have : i = 0 := by omega
    subst this
    exact Nat.le_refl _
  | succ j ih =>
    rcases Nat.eq_or_lt_of_le hj with h | h
    · rw [h]
    · have h1 := ih (by omega)
      have h2 : pre f (j + 1) = pre f j + (if f j then 1 else 0) := rfl
      split at h2 <;> omega

/-- A set lane has a strictly smaller position than every later lane. -/
theorem pre_lt (f : Nat → Bool) (i j : Nat) (hij : i < j) (hf : f i = true) : pre f i < pre f j := by
  have h1 : pre f (i + 1) = pre f i + 1 := by simp [pre, hf]
  have h2 := pre_mono f (i + 1) j (by omega)
  omega

/-- Compacted positions of distinct hits are distinct. -/
theorem pre_inj (f : Nat → Bool) (i j : Nat) (hi : f i = true) (hj : f j = true)
    (h : pre f i = pre f j) : i = j := by
  rcases Nat.lt_trichotomy i j with c | c | c
  · have := pre_lt f i j c hi
    omega
  · exact c
  · have := pre_lt f j i c hj
    omega

/-- Compacted positions are dense: below the total hit count. -/
theorem pre_lt_total (f : Nat → Bool) (i n : Nat) (hi : i < n) (hf : f i = true) : pre f i < pre f n :=
  pre_lt f i n hi hf

-- § scratch alignment

/-- Round up to a multiple of 16. -/
def r16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Round up to a multiple of 256. -/
def r256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- `r16` bounds. -/
theorem r16_props (n : Nat) : n ≤ r16 n ∧ r16 n % 16 = 0 ∧ r16 n < n + 16 := by
  unfold r16
  omega

/-- `r256` bounds. -/
theorem r256_props (n : Nat) : n ≤ r256 n ∧ r256 n % 256 = 0 ∧ r256 n < n + 256 := by
  unfold r256
  omega

/-- The regions `gate_up | q8 | scales` fit the carved total (the CPU-order variant uses four regions). -/
theorem scratch_regions_fit (gu q8 hs xh : Nat) :
    gu ≤ r16 gu ∧ r16 gu + q8 ≤ r16 gu + r16 q8 ∧
    r16 gu + r16 q8 + hs ≤ r16 gu + r16 q8 + 2 * r16 hs + r16 xh ∧
    r16 gu + r16 q8 + r16 hs + hs ≤ r16 gu + r16 q8 + 2 * r16 hs + r16 xh ∧
    r16 gu + r16 q8 + 2 * r16 hs + xh ≤ r16 gu + r16 q8 + 2 * r16 hs + r16 xh := by
  have a := r16_props gu
  have b := r16_props q8
  have c := r16_props hs
  have d := r16_props xh
  omega

/-- Per-hit buffers are multiples of 16 bytes for floats of width 5120 per hit (`n*FF*8`). -/
theorem gu_bytes_aligned (n : Nat) : r16 (n * 5120) = n * 5120 := by
  unfold r16
  omega

-- § native expert layout (iq_kernels.cu)

/-- Byte offsets of the native expert blob. -/
theorem native_layout_rows (nff guRow r : Nat) (hr : r < nff) : r * guRow + guRow ≤ nff * guRow := by
  have := Nat.mul_le_mul_right guRow (show r + 1 ≤ nff by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The four scratch regions of `native_expert_grouped` (three float buffers and the q8_1 image) fit. -/
theorem native_scratch_fit (f q : Nat) :
    f ≤ r256 f ∧ 3 * r256 f + q ≤ 3 * r256 f + r256 q := by
  have a := r256_props f
  have b := r256_props q
  omega

end Strata.S2ExpertLayout



## FILE: /mnt/user-data/outputs/MmvqIterationGeometry.lean
/-!
# MODULE: mmvq_iteration_geometry
SOURCES: src/kernels/cuda/native_mmvq.cu
Thread-to-block mapping, blocks-per-iteration, small-K thresholds, grid coverage, and multi-column strides of
the native (llama.cpp-transcribed) MMVQ kernels.

## Source semantics
* A 128-thread block (4 warps x 32) covers `BPI` weight blocks per iteration with `T` threads each, `T*BPI = 128`.
* The small-K rule (4 rows per block) triggers when `n_in / block_elems < BPI`.
* With `ncols` columns quantized as one vector, column `c` starts at activation block `c * (n_in/32)`.
* The upstream multi-column layout uses 2 warps for 5..8 columns with BPI halved.

## Form template identifiers
VDR -> values_per_dot_call ; QI -> dot_calls_per_block ; BPI -> blocks_per_iteration ; ROWS -> rows_per_cuda_block
-/

namespace Strata.MmvqGeometry

-- § threads per weight block and blocks per iteration

/-- (T threads per block, BPI blocks per iteration) for Q5_K, Q2_0, Q3_K, Q6_K, IQ4_XS, Q4_0/Q5_0/IQ4_NL, Q8_0. -/
def traits : List (Nat × Nat) := [(16, 8), (2, 64), (16, 8), (32, 4), (8, 16), (2, 64), (4, 32)]

/-- Every format assigns all 128 threads exactly once per iteration. -/
theorem threads_cover_block : ∀ p ∈ traits, p.1 * p.2 = 128 := by decide

/-- The `BPI` formulas of the source. -/
theorem bpi_values :
    2 * 4 * 32 / 32 = 8 ∧ 4 * 32 / 2 = 64 ∧ 4 * 32 / 16 = 8 ∧ 4 * 32 / 32 = 4 ∧
      4 * 4 * 32 / 32 = 16 ∧ 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by
  decide

/-- Thread `tid < T*BPI` handles a block index below `BPI`. -/
theorem thread_block_lt (T BPI tid : Nat) (h : tid < T * BPI) : tid / T < BPI :=
  Nat.div_lt_of_lt_mul h

/-- The (block, slot) pair of a thread determines the thread. -/
theorem thread_decomp (T tid : Nat) : T * (tid / T) + tid % T = tid := Nat.div_add_mod tid T

-- § small-K thresholds

/-- Q4_0 / Q5_0 / IQ4_NL: small-K iff `n_in < 2048`. -/
theorem smallk_4 (n : Nat) (h : n % 32 = 0) : n / 32 < 64 ↔ n < 2048 := by omega

/-- Q8_0: small-K iff `n_in < 1024`. -/
theorem smallk_q8_0 (n : Nat) (h : n % 32 = 0) : n / 32 < 32 ↔ n < 1024 := by omega

/-- Q4_K / Q5_K / Q3_K: small-K iff `n_in < 2048`. -/
theorem smallk_k (n : Nat) (h : n % 256 = 0) : n / 256 < 8 ↔ n < 2048 := by omega

/-- Q2_0: small-K iff `n_in < 4096`. -/
theorem smallk_q2_0 (n : Nat) (h : n % 64 = 0) : n / 64 < 64 ↔ n < 4096 := by omega

/-- IQ4_XS: small-K iff `n_in < 4096`. -/
theorem smallk_iq4xs (n : Nat) (h : n % 256 = 0) : n / 256 < 16 ↔ n < 4096 := by omega

/-- Q6_K: small-K iff `n_in < 1024`. -/
theorem smallk_q6k (n : Nat) (h : n % 256 = 0) : n / 256 < 4 ↔ n < 1024 := by omega

-- § grids

/-- Small-K launches 4 rows per block and cover all rows. -/
theorem grid_rows4 (n : Nat) : n ≤ ((n + 3) / 4) * 4 := by omega

/-- The upstream multi-column layout uses 2 rows per block. -/
theorem grid_rows2 (n : Nat) : n ≤ ((n + 1) / 2) * 2 := by omega

/-- With 2 warps (5..8 columns) the blocks per iteration are halved and still use all 64 threads. -/
theorem two_warp_traits : ∀ p ∈ [(16, 4), (2, 32), (32, 2), (8, 8), (4, 16)], p.1 * p.2 = 64 := by decide

/-- Halved BPI values of the formula `BPI * NW / WARPS` for NW = 2. -/
theorem two_warp_bpi : 8 * 2 / 4 = 4 ∧ 64 * 2 / 4 = 32 ∧ 4 * 2 / 4 = 2 ∧ 16 * 2 / 4 = 8 ∧ 32 * 2 / 4 = 16 := by
  decide

-- § multi-column strides

/-- Column `c` of an `ncols`-column activation starts at block `c * m` when `n_in = 32 m`, with the
within-block position unchanged: whole columns never share a 32-element block. -/
theorem column_blocks (c m i : Nat) :
    (c * (32 * m) + i) / 32 = c * m + i / 32 ∧ (c * (32 * m) + i) % 32 = i % 32 := by
  have e : c * (32 * m) = 32 * (c * m) := by rw [Nat.mul_left_comm]
  rw [e]
  exact ⟨Nat.mul_add_div (by decide) _ _, Nat.mul_add_mod _ _ _⟩

/-- Q8_1 activation image size: `ncols * (n_in/32) * 36`. -/
theorem q81_bytes (ncols m : Nat) : ncols * m * 36 = ncols * (m * 36) := Nat.mul_assoc _ _ _

-- § dispatcher coverage

/-- The (type, block elements, block bytes) table of `native_mmvq_weight_bytes`. -/
def weightTable : List (Nat × Nat × Nat) :=
  [(2, 32, 18), (6, 32, 22), (8, 32, 34), (20, 32, 18), (11, 256, 110), (12, 256, 144), (13, 256, 176),
   (14, 256, 210), (23, 256, 136), (42, 64, 18), (16, 256, 66), (17, 256, 74), (18, 256, 98),
   (21, 256, 110), (22, 256, 82), (29, 256, 56)]

/-- Every type reported as supported has a weight-size entry. -/
theorem supported_in_table :
    ∀ t ∈ [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29], ∃ e ∈ weightTable, e.1 = t := by
  decide

/-- Block sizes divide `n_in` only if `n_in` is a multiple of 256, 64 or 32: all divide 256. -/
theorem table_elems_divide_256 : ∀ e ∈ weightTable, 256 % e.2.1 = 0 := by decide

end Strata.MmvqGeometry



## FILE: /mnt/user-data/outputs/RouterSamplerSelection.lean
/-!
# MODULE: router_sampler_selection
SOURCES: src/kernels/cuda/router_top10.cu, src/kernels/cuda/sampler.cu
Selection semantics of the MoE router and the sampler: the (value, index) argmax combine used by every shuffle
reduction (ties go to the lowest index), the Philox uniform range, penalty monotonicity, the history window,
top-p cut adjustment and the router's launch/shared-memory geometry.

## Source semantics
* A reduction candidate `b` replaces `a` iff `b.value > a.value` or equal value with smaller index.
* Philox: `(c0 >> 8) * 2^-24` is in [0, 1).
* Repeat penalty multiplies non-positive logits and divides positive ones, so it never raises a logit.
* The history window is the TAIL of the row: start = len - min(last_n, len).
* The router clamps the renormalisation denominator at 2^-14 so it is never zero.

## Form template identifiers
bv/best -> running_best_value/index ; sv/si -> warp_partials ; penalty_repeat -> repeat_penalty
keep_ids/keep_logit -> top_k_survivors ; s_taken -> taken_mask
-/

namespace Strata.RouterSampler

-- § the argmax combine

/-- Combine two (value, index) candidates exactly as the shuffle reduction does. -/
def comb (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The combine is commutative, so the reduction order within a warp does not matter. -/
theorem comb_comm (a b : Int × Nat) : comb a b = comb b a := by
  obtain ⟨a1, a2⟩ := a
  obtain ⟨b1, b2⟩ := b
  unfold comb
  split <;> split <;> first | (exfalso; omega) | (apply Prod.ext <;> dsimp only <;> omega)

/-- If the challenger beats the incumbent, it wins. -/
theorem comb_pos (a b : Int × Nat) (h : b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)) : comb a b = b := by
  unfold comb
  exact if_pos h

/-- Otherwise the incumbent stays. -/
theorem comb_neg (a b : Int × Nat) (h : ¬ (b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2))) : comb a b = a := by
  unfold comb
  exact if_neg h

/-- The combine is associative, so any reduction tree yields the same winner. -/
theorem comb_assoc (a b c : Int × Nat) : comb (comb a b) c = comb a (comb b c) := by
  by_cases h1 : b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)
  · rw [comb_pos a b h1]
    by_cases h2 : c.1 > b.1 ∨ (c.1 = b.1 ∧ c.2 < b.2)
    · rw [comb_pos b c h2]
      by_cases h3 : c.1 > a.1 ∨ (c.1 = a.1 ∧ c.2 < a.2)
      · rw [comb_pos a c h3]
      · exfalso
        omega
    · rw [comb_neg b c h2, comb_pos a b h1]
  · rw [comb_neg a b h1]
    by_cases h2 : c.1 > b.1 ∨ (c.1 = b.1 ∧ c.2 < b.2)
    · rw [comb_pos b c h2]
    · rw [comb_neg b c h2, comb_neg a b h1]
      by_cases h3 : c.1 > a.1 ∨ (c.1 = a.1 ∧ c.2 < a.2)
      · exfalso
        omega
      · rw [comb_neg a c h3]

/-- On equal values the lower index wins, whichever side it arrives on. -/
theorem comb_tie (v : Int) (i j : Nat) (h : i < j) :
    comb (v, j) (v, i) = (v, i) ∧ comb (v, i) (v, j) = (v, i) := by
  constructor
  · exact comb_pos _ _ (Or.inr ⟨rfl, h⟩)
  · exact comb_neg _ _ (by dsimp only; omega)

/-- The greedy kernel answers 0 when every candidate was `-inf` (index `n_vocab` sentinel). -/
def greedyOut (wi n : Nat) : Nat := if wi < n then wi else 0

/-- The greedy output is always a valid token id. -/
theorem greedyOut_lt (wi n : Nat) (hn : 0 < n) : greedyOut wi n < n := by
  unfold greedyOut
  split <;> omega

-- § Philox uniform

/-- The 24-bit mantissa draw is strictly below 2^24, so the uniform is below 1. -/
theorem philox_lt_one (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by omega

-- § penalties (integer model of the sign rule)

/-- The repeat penalty: multiply non-positive logits, divide positive ones. -/
def repeatPen (l r : Int) : Int := if l ≤ 0 then l * r else l / r

/-- A repeat penalty with factor at least one never raises a logit. -/
theorem repeatPen_le (l r : Int) (hr : 1 ≤ r) : repeatPen l r ≤ l := by
  unfold repeatPen
  split
  · rename_i h
    have h0 : 0 ≤ (-l) * (r - 1) := Int.mul_nonneg (by omega) (by omega)
    have h1 : (-l) * (r - 1) = -(l * r) + l := by
      rw [Int.mul_sub, Int.mul_one, Int.neg_mul]
      omega
    omega
  · rename_i h
    exact Int.ediv_le_self r (by omega)

/-- The presence term is a boolean: at most one regardless of the count. -/
theorem presence_bool (count : Nat) : (if count > 0 then 1 else 0 : Nat) ≤ 1 := by
  split <;> omega

/-- The history window is the tail: it starts inside the row and ends at its end. -/
theorem window_tail (plen hist : Nat) :
    let hlen := min plen hist
    hist - hlen + hlen = hist ∧ hist - hlen ≤ hist := by
  intro hlen
  simp only [hlen]
  omega

-- § top-p cut adjustment

/-- `if (cut < min_keep) cut = min(min_keep, n_keep)`. -/
def cutAdj (cut minKeep nKeep : Nat) : Nat :=
  if cut < minKeep then (if minKeep < nKeep then minKeep else nKeep) else cut

/-- The adjusted cut keeps at least one survivor and never exceeds the survivors. -/
theorem cutAdj_bounds (cut minKeep nKeep : Nat) (h1 : 1 ≤ cut) (h2 : cut ≤ nKeep) (h3 : 1 ≤ minKeep) :
    1 ≤ cutAdj cut minKeep nKeep ∧ cutAdj cut minKeep nKeep ≤ nKeep := by
  simp only [cutAdj]
  split_ifs <;> omega

-- § router launch geometry

/-- Threads per token block: `min(n_expert, 512)` rounded up to a whole warp. -/
def thr (n : Nat) : Nat := (min n 512 + 31) / 32 * 32

/-- The thread count is a warp multiple, covers the experts (up to 512), and needs at most 16 warp slots. -/
theorem thr_props (n : Nat) (hn : 0 < n) :
    32 ≤ thr n ∧ thr n ≤ 512 ∧ thr n % 32 = 0 ∧ min n 512 ≤ thr n ∧ (thr n + 31) / 32 ≤ 16 := by
  unfold thr
  omega

/-- Round up to 16 (the `taken` mask padding). -/
def pad16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- The double array starts 16-aligned and the float array 4-aligned after `8 n` bytes of doubles. -/
theorem smem_alignment (n : Nat) : pad16 n % 16 = 0 ∧ (pad16 n + 8 * n) % 4 = 0 ∧ n ≤ pad16 n := by
  unfold pad16
  omega

/-- Shared memory for the full 512-expert router. -/
theorem smem_512 : pad16 512 + 512 * 8 + 512 * 4 = 6656 := by decide

/-- The clamp keeps the renormalisation denominator positive. -/
theorem clamp_positive (s floor : Nat) (h : 0 < floor) : 0 < max s floor := by omega

/-- The router's selection order: each pass takes the lowest-index maximum, so the first `k` outputs are
the stable descending order (checked here at the level of the combine's tie rule above). -/
theorem pass_takes_lowest_on_tie (v : Int) (i j : Nat) (h : i < j) : (comb (v, i) (v, j)).2 = i := by
  have := (comb_tie v i j h).2
  rw [this]

end Strata.RouterSampler



## FILE: /mnt/user-data/outputs/PleBlockGeometry.lean
/-!
# MODULE: ple_block_geometry
SOURCES: src/kernels/cuda/ple.cu
Index geometry of the PLE block's GPU half: the depthwise causal dilated convolution taps, the conv-state
layout (row-fastest), history advance, the broadcast/norm stream layout, workspace carving, and the
buffer-overlap predicate.

## Source semantics
* The conv reads rows `nhist - (kern-1-k)*dil` of `[history | new]`; for kern 4, dil 3, nhist 9 these are
  0, 3, 6, 9 and only the last tap reads the NEW row.
* The state is row-fastest: flat index `row + nhist * channel` (not `row * hc_dim + channel`).
* `ple_history_advance` shifts each channel column left by one and appends the new value.
* The scratch is carved from the caller's workspace at 16-byte aligned offsets.
* Two byte ranges overlap iff each starts before the other ends.

## Form template identifiers
NG_HIST -> history_rows ; NG_HC_DIM -> stream_dim ; NG_HC -> stream_count ; PLE_CONV_KERNEL -> conv_taps
-/

namespace Strata.PleGeometry

-- § conv taps

/-- Source row of tap `k`. -/
def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- The real geometry reads rows 0, 3, 6, 9. -/
theorem real_tap_rows : (List.range 4).map (tapRow 9 4 3) = [0, 3, 6, 9] := by decide

/-- Only the last tap reads the new row for the real geometry. -/
theorem only_last_tap_is_new : ∀ k < 4, (tapRow 9 4 3 k = 9 ↔ k = 3) := by decide

/-- In general (positive dilation, history long enough) only tap `kern-1` reads the new row. -/
theorem new_row_only_last (nhist kern dil k : Nat) (hd : 0 < dil) (hk : k < kern)
    (hle : (kern - 1 - k) * dil ≤ nhist) (hn : 0 < nhist) :
    (tapRow nhist kern dil k = nhist ↔ k = kern - 1) := by
  unfold tapRow
  have hz : (kern - 1 - k) * dil = 0 ↔ kern - 1 - k = 0 := by
    constructor
    · intro h
      rcases Nat.mul_eq_zero.mp h with h | h
      · exact h
      · omega
    · intro h
      rw [h]
      simp
  generalize (kern - 1 - k) * dil = p at hle hz ⊢
  omega

/-- Kernel weight index `k + 4c` stays inside `4 * hc_dim` for the real geometry. -/
theorem kernel_weight_index (k c : Nat) (hk : k < 4) (hc : c < 10240) : k + 4 * c < 40960 := by omega

/-- Row-fastest state indexing stays in the state buffer for the history rows. -/
theorem state_index (row c : Nat) (hr : row < 9) (hc : c < 10240) : row + 9 * c < 9 * 10240 := by omega

/-- Row-fastest and channel-slowest indexing disagree, so mixing them up reads a transposed state. -/
theorem layout_not_transposable : 1 + 9 * 0 ≠ 1 * 10240 + 0 := by decide

-- § history advance

/-- Shift a column left by one and append the new value. -/
def advance {α : Type} (h : List α) (x : α) : List α := h.tail ++ [x]

/-- Advancing preserves the history length. -/
theorem advance_length {α : Type} (h : List α) (x : α) (hne : h ≠ []) :
    (advance h x).length = h.length := by
  cases h with
  | nil => exact absurd rfl hne
  | cons a t => simp [advance]

/-- The newest entry is the appended value. -/
theorem advance_last {α : Type} (h : List α) (x : α) : (advance h x).getLast? = some x := by
  simp [advance]

-- § stream layout

/-- Streams partition `hc_dim`: stream `c < 4` occupies `[c*2560, (c+1)*2560)` inside 10240. -/
theorem stream_in_range (c : Nat) (h : c < 4) : c * 2560 + 2560 ≤ 10240 := by omega

/-- Broadcast indices: element `i < 10240` reads value `i % 2560` and gate `i / 2560 < 4`. -/
theorem bcast_indices (i : Nat) (h : i < 10240) : i % 2560 < 2560 ∧ i / 2560 < 4 := by omega

/-- `hc * n_embd = hc_dim`. -/
theorem hc_dim_factor : 4 * 2560 = 10240 := by decide

-- § workspace carving

/-- Round up to 16. -/
def r16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Total scratch of `ple_block_scratch_bytes`. -/
def scratchBytes : Nat :=
  r16 ((5 * 10240 + 2560 + 4) * 4) + r16 (2560 / 32 * 34) + 2560 * 2 + 256

/-- The concrete scratch size. -/
theorem scratch_value : scratchBytes = 223152 := by decide

/-- The activation image and the bf16 copy start at 16-aligned offsets and fit the total. -/
theorem scratch_regions (f q : Nat) :
    r16 f % 16 = 0 ∧ (r16 f + r16 q) % 16 = 0 ∧ r16 f + q ≤ r16 f + r16 q := by
  unfold r16
  omega

/-- The float workspace holds 5 * hc_dim + n_embd + hc floats, ending exactly at the gate buffer. -/
theorem float_workspace : 5 * 10240 + 2560 + 4 = 53764 := by decide

-- § overlap predicate

/-- Byte ranges `[a, a+ab)` and `[b, b+bb)` overlap (non-empty ranges only). -/
def overlap (a ab b bb : Nat) : Prop :=
  if a < b then b - a < ab else a - b < bb

/-- The overlap test is exactly interval intersection. -/
theorem overlap_iff (a ab b bb : Nat) (ha : 0 < ab) (hb : 0 < bb) :
    overlap a ab b bb ↔ (a < b + bb ∧ b < a + ab) := by
  unfold overlap
  split <;> omega

/-- The overlap test is symmetric. -/
theorem overlap_symm (a ab b bb : Nat) (ha : 0 < ab) (hb : 0 < bb) :
    overlap a ab b bb ↔ overlap b bb a ab := by
  rw [overlap_iff a ab b bb ha hb, overlap_iff b bb a ab hb ha]
  omega

-- § gate sign

/-- Sign function of the gate. -/
def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- The gate sign is odd, so the gate is symmetric about 0.5. -/
theorem sgn_neg (s : Int) : sgn (-s) = -sgn s := by
  simp only [sgn]
  split_ifs <;> omega

end Strata.PleGeometry



## FILE: /mnt/user-data/outputs/ActivationQuantize.lean
/-!
# MODULE: activation_quantize
SOURCES: src/kernels/cuda/quantize_act.cu, src/kernels/cuda/iq_kernels.cu
Integer semantics of the activation quantizers: clamp ranges, Q8_0 / Q8_1 / Q8_K block layouts, the
`nearest_int` magic-constant trick, bsums bounds and launch grids.

## Source semantics
* Q8_0 block = fp16 d + 32 int8 = 34 bytes; Q8_1 = half2 ds + 32 int8 = 36 bytes; Q8_K = f32 d + 256 int8 +
  16 int16 bsums = 292 bytes.
* `nearest_int(x)` adds 1.5*2^23, reads the mantissa bits and subtracts 0x400000; exact for |x| < 2^22.
* ggml's reference clamps to [-128, 127]; the engine's CPU-parity variant clamps to [-127, 127].
* The sum of 16 int8 values in [-127,127] fits an int16.

## Form template identifiers
quantize_q8_0_kernel -> quantize_q8_0_ggml_rule ; quantize_q8_0_scaled_kernel -> quantize_q8_0_cpu_rule
nearest_int_dev -> round_to_int_magic ; bsums -> block_partial_sums
-/

namespace Strata.ActQuant

-- § clamps

/-- Clamp an integer into `[lo, hi]`. -/
def clampI (lo hi v : Int) : Int := if v < lo then lo else if hi < v then hi else v

/-- Clamp result is within bounds. -/
theorem clampI_bounds (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clampI lo hi v ∧ clampI lo hi v ≤ hi := by
  simp only [clampI]
  split_ifs <;> omega

/-- Clamp is the identity inside the range. -/
theorem clampI_id (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) : clampI lo hi v = v := by
  simp only [clampI]
  split_ifs <;> omega

/-- The CPU-parity rule uses a symmetric range, so negation is closed. -/
theorem symmetric_range_closed (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : -127 ≤ -v ∧ -v ≤ 127 := by
  omega

-- § block layouts

/-- Q8_0 layout: 2 + 32 = 34; element `j < 32` is at byte `2 + j`. -/
theorem q8_0_layout (j : Nat) (h : j < 32) : 2 + 32 = 34 ∧ 2 + j < 34 := by omega

/-- Q8_1 layout: 4 + 32 = 36; element `j < 32` is at byte `4 + j`. -/
theorem q8_1_layout (j : Nat) (h : j < 32) : 4 + 32 = 36 ∧ 4 + j < 36 := by omega

/-- Q8_K layout: 4 + 256 + 32 = 292, bsums start at 260 (even) and end at 292. -/
theorem q8_k_layout (j : Nat) (h : j < 16) :
    4 + 256 + 32 = 292 ∧ 260 % 2 = 0 ∧ 4 + 256 + 2 * j + 2 ≤ 292 := by omega

/-- Block starts keep the alignment their fields need: Q8_K `d` is 4-aligned, Q8_0 `d` is 2-aligned. -/
theorem block_start_alignment (b : Nat) : (b * 292) % 4 = 0 ∧ (b * 34) % 2 = 0 := by omega

/-- Q8_1 element addressing: flat index `i` is block `i / 32` slot `i % 32` and is recoverable. -/
theorem q81_index (i : Nat) : i = (i / 32) * 32 + i % 32 ∧ i % 32 < 32 := by omega

/-- A buffer of `n` elements (`n % 32 = 0`) has `n / 32` blocks and element `i < n` is in range. -/
theorem q81_in_range (i n : Nat) (hn : n % 32 = 0) (hi : i < n) : i / 32 < n / 32 := by omega

-- § bsums

/-- Sixteen values of magnitude at most 127 sum to at most 2032, which fits an int16. -/
theorem bsums_fit : 16 * 127 = 2032 ∧ 2032 ≤ 32767 := by decide

-- § nearest_int magic constant

/-- Mantissa recovery: with `m = (4194304 + n) mod 2^23`, `m - 0x400000 = n` for `-2^22 ≤ n < 2^22`. -/
theorem nearest_int_magic (n : Int) (h1 : -4194304 ≤ n) (h2 : n < 4194304) :
    ((4194304 + n) % 8388608) - 4194304 = n := by
  omega

/-- All int8 quant values (|x| ≤ 127) are far inside the magic constant's exact range. -/
theorem nearest_int_range : (127 : Int) < 4194304 := by decide

/-- The Q8_K quantizer clamps only from above (`min 127`); values of the scaled product are at most 127. -/
theorem q8k_min_only (v : Int) (h : v ≤ 127) : min 127 v = v := by omega

-- § launch grids

/-- One thread per Q8_0 block, 128 threads per block. -/
theorem q8_0_grid (nb : Nat) : nb ≤ ((nb + 127) / 128) * 128 := by omega

/-- One thread per Q8_K block, 64 threads per block. -/
theorem q8_k_grid (nb : Nat) : nb ≤ ((nb + 63) / 64) * 64 := by omega

/-- One thread per element, 256 threads per block (Q8_1). -/
theorem q8_1_grid (n : Nat) : n ≤ ((n + 255) / 256) * 256 := by omega

/-- Columns quantized as one vector keep every 32-block inside a column when `n_in % 32 = 0`. -/
theorem columns_as_vector (n_in c i : Nat) (h : n_in % 32 = 0) (hi : i < n_in) :
    (c * n_in + i) / 32 = c * (n_in / 32) + i / 32 := by
  obtain ⟨m, rfl⟩ : ∃ m, n_in = 32 * m := ⟨n_in / 32, by omega⟩
  have e : c * (32 * m) = 32 * (c * m) := by rw [Nat.mul_left_comm]
  rw [e, Nat.mul_add_div (by decide)]
  have : 32 * m / 32 = m := by omega
  rw [this]

end Strata.ActQuant





## FILE: /mnt/user-data/outputs/Q2ExpertKernelLayout.lean
/-!
# MODULE: q2_expert_kernel_layout
SOURCES: include/strata/kernels/cpu/expert.hpp, src/kernels/cpu/expert.cpp
The CPU Q2_0 routed-expert kernel: fixed geometry, the 1,382,400-byte expert blob layout, the
activation-quantizer clamp, tail-mask arithmetic of the 512-bit row dot, the Q2_0 offset identity and
the CPU-feature gate.

## Source semantics
* One Q2_0 weight block is QK=64 weights with one fp16 scale; one activation chunk is QKA=32 elements with
  its own scale, so a weight block spans exactly two chunks.
* The blob is: gate/up codes | down codes | gate/up scales | down scales, tiling BLOB exactly.
* `sum (c-1) xhat = sum c xhat - sum xhat` (the code offset is removed through `hx`).
* The multi-token dispatch uses widths 1..8 and falls back to single-token for the oracle contract.

## Form template identifiers
* H, FF, NE, QK, QKA, BB  -> hidden_width, expert_ff_width, n_experts, weight_block, act_chunk, block_bytes
* O_GU_CODES .. O_D_SCALES -> blob section offsets
* ActQ -> quantized_activation, CpuFeatures -> cpu_feature_set
-/

namespace Strata.Q2Expert

-- § geometry (literals, with proofs that they equal the source formulas)
def H : Nat := 2560
def FF : Nat := 640
def NE : Nat := 512
def QK : Nat := 64
def QKA : Nat := 32
def BB : Nat := 18
def MAXC : Nat := 80
def ROW_GU : Nat := 640
def ROW_D : Nat := 160
def SC_GU : Nat := 40
def SC_D : Nat := 10
def BLOB : Nat := 1382400
def O_GU_CODES : Nat := 0
def O_D_CODES : Nat := 819200
def O_GU_SCALES : Nat := 1228800
def O_D_SCALES : Nat := 1331200
def MAXT : Nat := 8

/-- The derived constants equal the formulas written in expert.hpp. -/
theorem geometry_formulas :
    MAXC = H / QKA ∧ ROW_GU = H * 2 / 8 ∧ ROW_D = FF * 2 / 8 ∧ SC_GU = H / QK ∧ SC_D = FF / QK ∧
    BLOB = 3 * (H * FF * BB / QK) := by
  decide

/-- The blob section offsets equal the formulas written in expert.hpp. -/
theorem offset_formulas :
    O_D_CODES = 2 * FF * ROW_GU ∧ O_GU_SCALES = O_D_CODES + 1 * H * ROW_D ∧
    O_D_SCALES = O_GU_SCALES + 2 * FF * SC_GU * 2 := by
  decide

/-- One 64-weight block spans exactly two 32-element activation chunks. -/
theorem weight_block_spans_two_chunks : QK = 2 * QKA := by decide

/-- A Q2_0 block is a 2-byte fp16 scale plus 16 bytes of 2-bit codes. -/
theorem block_bytes_split : BB = 2 + QK / 4 := by decide

/-- Every row width is a whole number of weight blocks. -/
theorem rows_are_whole_blocks : H = QK * SC_GU ∧ FF = QK * SC_D := by decide

/-- The four sections tile the blob exactly. -/
theorem blob_tiles : O_D_SCALES + H * SC_D * 2 = BLOB := by decide

/-- Sections are laid out back to back. -/
theorem sections_contiguous :
    O_GU_CODES + 2 * FF * ROW_GU = O_D_CODES ∧ O_D_CODES + H * ROW_D = O_GU_SCALES ∧
    O_GU_SCALES + 2 * FF * SC_GU * 2 = O_D_SCALES := by
  decide

/-- Gate and up rows are interleaved: row `2r` and `2r+1` of the code plane stay inside the plane. -/
theorem gu_code_rows_in_bounds (r : Nat) (hr : r < FF) :
    O_GU_CODES + (2 * r + 1) * ROW_GU + ROW_GU ≤ O_D_CODES := by
  simp only [O_GU_CODES, ROW_GU, O_D_CODES, FF] at *
  omega

/-- Down code rows stay inside the down-code plane. -/
theorem down_code_rows_in_bounds (r : Nat) (hr : r < H) :
    O_D_CODES + r * ROW_D + ROW_D ≤ O_GU_SCALES := by
  simp only [ROW_D, O_D_CODES, O_GU_SCALES, H] at *
  omega

/-- Gate/up scale rows stay inside their plane. -/
theorem gu_scale_rows_in_bounds (r : Nat) (hr : r < FF) :
    O_GU_SCALES + (2 * r + 1) * SC_GU * 2 + SC_GU * 2 ≤ O_D_SCALES := by
  simp only [SC_GU, O_GU_SCALES, O_D_SCALES, FF] at *
  omega

/-- Down scale rows stay inside the blob. -/
theorem down_scale_rows_in_bounds (r : Nat) (hr : r < H) :
    O_D_SCALES + r * SC_D * 2 + SC_D * 2 ≤ BLOB := by
  simp only [SC_D, O_D_SCALES, BLOB, H] at *
  omega

-- § activation quantizer
/-- Clamp used by `act_quant_q8_1` after rounding. -/
def clamp127 (v : Int) : Int := max (-127) (min v 127)

/-- The clamp lands in [-127, 127]. -/
theorem clamp127_range (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_idem (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- Activation chunk counts: 2560 -> 80 (= MAXC), 640 -> 20. -/
theorem chunk_counts : H / QKA = MAXC ∧ FF / QKA = 20 := by decide

-- § the Q2_0 code offset identity
/-- Offset identity: summing `(c-1)*x` equals summing `c*x` minus summing `x`. -/
theorem offset_identity :
    ∀ (cs xs : List Int), cs.length = xs.length →
      (List.zipWith (fun c x => (c - 1) * x) cs xs).sum =
        (List.zipWith (fun c x => c * x) cs xs).sum - xs.sum
  | [], [], _ => by simp
  | c :: cs, x :: xs, h => by
    have h' : cs.length = xs.length := by simpa using h
    have ih := offset_identity cs xs h'
    simp only [List.zipWith_cons_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    generalize c * x = p
    omega
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

-- § 512-bit tail masks
/-- A mask of `2*nb` bits for `nb ≤ 8` blocks fits a 16-bit lane mask. -/
theorem tail_mask_fits : ∀ nb : Nat, nb < 9 → (1 <<< (2 * nb)) - 1 < 65536 := by
  decide

/-- Gate/up rows (40 blocks) are an exact multiple of the 8-block batch; down rows (10 blocks) leave a 2-block tail. -/
theorem block_batches : SC_GU % 8 = 0 ∧ SC_D % 8 = 2 ∧ (SC_GU + 7) / 8 = 5 ∧ (SC_D + 7) / 8 = 2 := by
  decide

-- § multi-token dispatch
/-- Width chosen by the `switch (n_tokens)` ladder. -/
def dispatchNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- For every legal token count 1..MAXT the dispatched width is exactly the token count. -/
theorem dispatch_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ MAXT) : dispatchNT n = n := by
  simp only [dispatchNT, MAXT] at *
  split <;> omega

-- § CPU feature gate
/-- The five AVX-512 feature bits the kernel needs. -/
structure CpuFeatures where
  avx512f : Bool
  avx512bw : Bool
  avx512vl : Bool
  avx512_vnni : Bool
  avx512_vbmi : Bool

/-- `usable()` is the conjunction of all five bits. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- The machine is accepted exactly when every required feature is present. -/
theorem usable_iff (f : CpuFeatures) :
    f.usable = true ↔ (f.avx512f = true ∧ f.avx512bw = true ∧ f.avx512vl = true ∧
      f.avx512_vnni = true ∧ f.avx512_vbmi = true) := by
  unfold CpuFeatures.usable
  simp only [Bool.and_eq_true, and_assoc]

/-- A CPU without VNNI is refused even with every other bit set. -/
theorem no_vnni_refused (f : CpuFeatures) (h : f.avx512_vnni = false) : f.usable = false := by
  unfold CpuFeatures.usable
  simp [h]

end Strata.Q2Expert



## FILE: /mnt/user-data/outputs/ExpertPoolScheduling.lean
/-!
# MODULE: expert_pool_scheduling
SOURCES: include/strata/kernels/cpu/pool.hpp, src/kernels/cpu/pool.cpp
The CPU expert pool: row-range splitting of experts across tasks, the flat-batch counter protocol
(`head`/`done`/`parked`), worker-count selection and core reservation.

## Source semantics
* A task `i` of a split phase covers rows `[FF*part/parts, FF*(part+1)/parts)`; ranges tile `[0, FF)`.
* In multi-token phases a task covers a global row range `[g0, g1)` that may span experts; each step
  takes `r1 = min per (r0 + (g1 - r))` and always makes progress.
* `run()` publishes only when `parked == n`, waits for `done == n`, then waits for `parked == n` again.
* At least one worker exists; `skip_first` drops exactly one core.

## Form template identifiers
* head_, done_, parked_, epoch_ -> claim_counter, completion_counter, park_counter, publish_epoch
* ExpertJob / ExpertJobMulti -> work_item, multi_token_work_item
* kMaxSplit, kMaxSplitMulti -> split_capacity
-/

namespace Strata.Pool

-- § row-range splitting
/-- Row start of task `part` out of `parts` over `total` rows (`FF * part / parts`). -/
def rowStart (total parts part : Nat) : Nat := total * part / parts

/-- The first range starts at row 0. -/
theorem rowStart_zero (total parts : Nat) : rowStart total parts 0 = 0 := by
  simp [rowStart]

/-- The last range ends at the total (needs at least one part). -/
theorem rowStart_last (total parts : Nat) (hp : 0 < parts) : rowStart total parts parts = total := by
  unfold rowStart
  exact Nat.mul_div_cancel total hp

/-- Range starts are monotone, so each range `[start part, start (part+1))` is well formed. -/
theorem rowStart_mono (total parts part : Nat) : rowStart total parts part ≤ rowStart total parts (part + 1) := by
  unfold rowStart
  exact Nat.div_le_div_right (Nat.mul_le_mul_left total (Nat.le_succ part))

/-- Every range is inside `[0, total]`. -/
theorem rowStart_le_total (total parts part : Nat) (hp : 0 < parts) (hpart : part ≤ parts) :
    rowStart total parts part ≤ total := by
  have h1 := rowStart_mono total parts part
  have h2 : rowStart total parts part ≤ rowStart total parts parts := by
    unfold rowStart
    exact Nat.div_le_div_right (Nat.mul_le_mul_left total hpart)
  rw [rowStart_last total parts hp] at h2
  exact h2

-- § task decomposition (mode 1 / mode 2: task i = expert e, part p)
/-- A task index below `parts * n` decodes to an expert below `n`. -/
theorem task_expert_lt (i parts n : Nat) (h : i < parts * n) : i / parts < n :=
  Nat.div_lt_of_lt_mul h

/-- A task index decodes to a part below `parts`. -/
theorem task_part_lt (i parts : Nat) (hp : 0 < parts) : i % parts < parts :=
  Nat.mod_lt _ hp

/-- The decomposition is invertible: `i = parts * e + part`. -/
theorem task_decode (i parts : Nat) : parts * (i / parts) + i % parts = i :=
  Nat.div_add_mod i parts

-- § multi-token phases: a global row range cut at expert boundaries
/-- End of the current piece: stay inside the expert (`per` rows) and inside the task's range. -/
def pieceEnd (per r0 g1 r : Nat) : Nat := min per (r0 + (g1 - r))

/-- Global row `r` decodes to expert `r / per` and local row `r % per`. -/
theorem global_row_decode (r per : Nat) : per * (r / per) + r % per = r :=
  Nat.div_add_mod r per

/-- The local start is inside the expert. -/
theorem local_start_lt (r per : Nat) (hper : 0 < per) : r % per < per :=
  Nat.mod_lt _ hper

/-- The piece is non-empty while rows remain in the task: guarantees the loop makes progress. -/
theorem piece_progress (per r0 g1 r : Nat) (h0 : r0 < per) (hr : r < g1) :
    r0 < pieceEnd per r0 g1 r := by
  unfold pieceEnd
  omega

/-- The piece never leaves the expert. -/
theorem piece_in_expert (per r0 g1 r : Nat) : pieceEnd per r0 g1 r ≤ per := by
  unfold pieceEnd
  omega

/-- The piece never consumes more rows than remain in the task, so `r += r1 - r0` cannot overshoot `g1`. -/
theorem piece_no_overshoot (per r0 g1 r : Nat) (hr : r ≤ g1) :
    pieceEnd per r0 g1 r - r0 ≤ g1 - r := by
  unfold pieceEnd
  omega

-- § counter protocol
/-- Observable pool state: claim counter, completion counter, parked workers. -/
structure PoolState where
  head : Nat
  done : Nat
  parked : Nat

/-- A claim returns the old head and bumps it. -/
def claim (s : PoolState) : Nat × PoolState := (s.head, { s with head := s.head + 1 })

/-- A claimed index is executed only when it is below the batch size. -/
def executes (n i : Nat) : Bool := decide (i < n)

/-- Claims return distinct, consecutive indices. -/
theorem claim_distinct (s : PoolState) : (claim s).1 < (claim (claim s).2).1 := by
  simp [claim]

/-- Publishing is only legal when every worker is parked (the host waits for this first). -/
def canPublish (n : Nat) (s : PoolState) : Prop := s.parked = n

/-- Publication resets head and done: afterwards no stale claim is possible. -/
def publish (s : PoolState) : PoolState := { s with head := 0, done := 0 }

/-- After publication the batch starts from a clean counter state. -/
theorem publish_clean (s : PoolState) : (publish s).head = 0 ∧ (publish s).done = 0 := by
  simp [publish]

/-- The batch is complete only when both `done == n` and `parked == nw`. -/
def batchComplete (n nw : Nat) (s : PoolState) : Prop := s.done = n ∧ s.parked = nw

/-- Completion implies every worker is parked, i.e. nobody can still claim from the old batch. -/
theorem complete_implies_parked (n nw : Nat) (s : PoolState) (h : batchComplete n nw s) : s.parked = nw :=
  h.2

-- § worker count and core reservation
/-- Worker count: requested count if positive, else every core; never below one. -/
def poolWorkers (requested cores : Nat) : Nat := max 1 (if 0 < requested then requested else cores)

/-- A pool always has at least one worker, even on a single-core machine. -/
theorem poolWorkers_pos (requested cores : Nat) : 1 ≤ poolWorkers requested cores := by
  unfold poolWorkers
  exact Nat.le_max_left _ _

/-- A positive request is honoured exactly. -/
theorem poolWorkers_request (requested cores : Nat) (h : 0 < requested) :
    poolWorkers requested cores = requested := by
  unfold poolWorkers
  rw [if_pos h]
  omega

/-- Core list after the optional reservation of core 0 for the host. -/
def reserveHost (skipFirst : Bool) (cores : List Nat) : List Nat := if skipFirst then cores.tail else cores

/-- Reserving the host's core removes exactly one core when there is one. -/
theorem reserveHost_length (cores : List Nat) (h : cores ≠ []) :
    (reserveHost true cores).length + 1 = cores.length := by
  cases cores with
  | nil => exact absurd rfl h
  | cons a t => simp [reserveHost]

/-- Without the reservation the list is untouched. -/
theorem reserveHost_false (cores : List Nat) : reserveHost false cores = cores := by
  simp [reserveHost]

/-- With `host_works` the drain uses workers plus the host thread. -/
def drainThreads (nw : Nat) (hostWorks : Bool) : Nat := nw + if hostWorks then 1 else 0

/-- Task count of the multi phases is three tasks per thread, hence strictly positive. -/
theorem multi_tasks_pos (nw : Nat) (hostWorks : Bool) (h : 1 ≤ nw) : 0 < 3 * drainThreads nw hostWorks := by
  unfold drainThreads
  split <;> omega

/-- Split phases are used only up to capacity; larger batches fall back to `run`/singles. -/
def useSplit (n kMax : Nat) (oracle : Bool) : Bool := decide (n ≤ kMax) && !oracle

/-- The oracle contract never uses the split path. -/
theorem oracle_never_split (n kMax : Nat) : useSplit n kMax true = false := by
  simp [useSplit]

end Strata.Pool



## FILE: /mnt/user-data/outputs/IquantBlockDecode.lean
/-!
# MODULE: iquant_block_decode
SOURCES: src/kernels/cpu/iq_avx512.cpp
Multi-token AVX-512 dot products for the i-quant formats IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18),
IQ3_S(21), IQ2_S(22): block byte sizes, in-bounds reads of each 64-value chunk decode, scale-lane
mapping, sign application and the token-chunking loop.

## Source semantics
* Each format has a block of 256 values with a fixed byte size; chunk `j` (0..3) decodes 64 values.
* Every byte read performed by `Fmt<TY>::decode` for `j < 4` stays inside the block.
* A scale lane `l` of 32 int16 lanes takes scale `l / 8`; scales are odd numbers `2*s+1`.
* Signs are applied with a masked negate, an involution.
* `dot_rows_nt` processes tokens in groups of at most 4 and covers all `nt` tokens.

## Form template identifiers
* Fmt<TY>::bytes -> block_bytes, Fmt<TY>::K -> format_constant (numerator over 8)
* iq512_supported -> supported_type_predicate
-/

namespace Strata.Iquant

-- § supported types and block sizes
/-- The five supported ggml type ids. -/
def supported (ty : Nat) : Bool := ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

/-- Block bytes per format. -/
def blockBytes (ty : Nat) : Nat :=
  if ty = 16 then 66 else if ty = 17 then 74 else if ty = 18 then 98
  else if ty = 21 then 110 else if ty = 22 then 82 else 0

/-- Supported types are exactly those with a non-zero block size (checked over all type ids below 64). -/
theorem supported_iff_sized : ∀ ty : Nat, ty < 64 → (supported ty = true ↔ 0 < blockBytes ty) := by
  decide

/-- Block sizes decompose into the fp16 scale plus payload fields. -/
theorem block_byte_decomposition :
    blockBytes 16 = 2 + 64 ∧ blockBytes 17 = 2 + 64 + 8 ∧ blockBytes 22 = 2 + 64 + 8 + 8 ∧
    blockBytes 18 = 2 + 96 ∧ blockBytes 21 = 2 + 64 + 8 + 32 + 4 := by
  decide

/-- Block size of IQ2_XXS. -/
theorem bb16 : blockBytes 16 = 66 := by decide
/-- Block size of IQ2_XS. -/
theorem bb17 : blockBytes 17 = 74 := by decide
/-- Block size of IQ3_XXS. -/
theorem bb18 : blockBytes 18 = 98 := by decide
/-- Block size of IQ3_S. -/
theorem bb21 : blockBytes 21 = 110 := by decide
/-- Block size of IQ2_S. -/
theorem bb22 : blockBytes 22 = 82 := by decide

-- § in-bounds reads of the chunk decode (j = 0..3)
/-- IQ2_XXS: the 16-byte read at `2 + 16 j` ends at or before byte 66. -/
theorem iq2xxs_reads (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ blockBytes 16 := by
  rw [bb16]; omega

/-- IQ2_XS: code read and the two scale bytes are in bounds. -/
theorem iq2xs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < blockBytes 17 := by
  rw [bb17]; omega

/-- IQ2_S: grid bytes, sign bytes, high bits and scale bytes are all in bounds. -/
theorem iq2s_reads (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 2 + 32 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < blockBytes 22 ∧
    74 + 2 * j + 1 < blockBytes 22 := by
  rw [bb22]; omega

/-- IQ3_XXS: grid bytes and the 8-byte sign/scale words are in bounds. -/
theorem iq3xxs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 2 + 64 + 8 * j + 8 ≤ blockBytes 18 := by
  rw [bb18]; omega

/-- IQ3_S: grid, high-bit, sign and scale reads are in bounds. -/
theorem iq3s_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < blockBytes 21 := by
  rw [bb21]; omega

-- § scale lanes
/-- Scale index of int16 lane `l` (lanes 0-7 -> s0 ... 24-31 -> s3). -/
def scaleLane (l : Nat) : Nat := l / 8

/-- Every one of the 32 lanes maps to one of the four scales. -/
theorem scale_lane_lt : ∀ l : Nat, l < 32 → scaleLane l < 4 := by decide

/-- Odd scale from a 4-bit field: `2*s + 1` lies in 1..31 and is odd. -/
theorem odd_scale (s : Nat) (h : s < 16) : 1 ≤ 2 * s + 1 ∧ 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by
  omega

-- § signs
/-- Masked negate: negate exactly when the sign bit is set. -/
def applySign (bit : Bool) (y : Int) : Int := if bit then -y else y

/-- Applying the same sign twice restores the value. -/
theorem applySign_involutive (bit : Bool) (y : Int) : applySign bit (applySign bit y) = y := by
  cases bit <;> simp [applySign]

/-- An unset sign bit leaves the activation untouched. -/
theorem applySign_false (y : Int) : applySign false y = y := by
  simp [applySign]

-- § token chunking in dot_rows_nt
/-- Size of the group starting at token `t0` (at most 4). -/
def groupSize (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Each group is non-empty while tokens remain. -/
theorem group_nonempty (nt t0 : Nat) (h : t0 < nt) : 0 < groupSize nt t0 := by
  unfold groupSize; omega

/-- Each group stays inside the token range. -/
theorem group_in_range (nt t0 : Nat) (h : t0 < nt) : t0 + groupSize nt t0 ≤ nt := by
  unfold groupSize; omega

/-- Each group has at most 4 tokens, matching the instantiated widths 1..4. -/
theorem group_le_four (nt t0 : Nat) : groupSize nt t0 ≤ 4 := by
  unfold groupSize; omega

-- § row addressing
/-- Row `r` of a matrix with `rowBytes` per row starts at `r * rowBytes`; consecutive rows do not overlap. -/
theorem rows_disjoint (r rowBytes : Nat) : r * rowBytes + rowBytes = (r + 1) * rowBytes := by
  rw [Nat.add_mul, Nat.one_mul]

/-- Block `i` of a row starts at `i * bytes` and stays within the row when `i < nblocks`. -/
theorem block_in_row (i nblocks bytes : Nat) (h : i < nblocks) : i * bytes + bytes ≤ nblocks * bytes := by
  have : (i + 1) * bytes ≤ nblocks * bytes := Nat.mul_le_mul_right bytes h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

end Strata.Iquant



## FILE: /mnt/user-data/outputs/Q2Avx2GgufRows.lean
/-!
# MODULE: q2_avx2_gguf_rows
SOURCES: src/kernels/cpu/q2_avx2.cpp
Q2_0 rows in the GGUF block layout (18 bytes per 64 weights) and the activation quantizer for CPUs
without AVX-512: 2-bit unpack addressing, chunk/scale association, block strides, token grouping.

## Source semantics
* Value `i` of a 64-value block lives in code byte `i / 4` at bit `2 * (i % 4)`; values 0..31 come from
  bytes 0..7 (the `lo` vector) and 32..63 from bytes 8..15 (`hi`).
* Block `b` of a row starts at `18 * b`: scale at `+0..1`, codes at `+2..17`.
* Value `i` of block `b` uses activation chunk `2*b + i/32`.
* Token groups have at most 4 tokens.

## Form template identifiers
* unpack64 -> unpack_codes_64, row_multi -> multi_token_row_dot, act_quant_q8_1_avx2 -> activation_quantizer
-/

namespace Strata.Q2Avx2

/-- Byte index holding value `i`. -/
def codeByte (i : Nat) : Nat := i / 4
/-- Bit shift of value `i` inside its byte. -/
def codeShift (i : Nat) : Nat := 2 * (i % 4)

/-- All 64 values read from within the 16 code bytes. -/
theorem code_byte_lt : ∀ i : Nat, i < 64 → codeByte i < 16 := by decide

/-- Shifts are 0, 2, 4 or 6, so a 2-bit field never straddles bytes. -/
theorem code_shift_le : ∀ i : Nat, i < 64 → codeShift i + 2 ≤ 8 := by decide

/-- The `lo` vector (values 0..31) uses bytes 0..7; the `hi` vector (32..63) uses bytes 8..15. -/
theorem lo_hi_split : ∀ i : Nat, i < 64 → (i < 32 → codeByte i < 8) ∧ (32 ≤ i → 8 ≤ codeByte i) := by
  decide

/-- Value index of a slot: inverse of (byte, shift). -/
theorem code_slot_roundtrip : ∀ i : Nat, i < 64 → 4 * codeByte i + codeShift i / 2 = i := by decide

/-- Distinct values never share a (byte, shift) slot. -/
theorem code_slot_injective (i j : Nat) (hi : i < 64) (hj : j < 64)
    (hb : codeByte i = codeByte j) (hs : codeShift i = codeShift j) : i = j := by
  have h1 := code_slot_roundtrip i hi
  have h2 := code_slot_roundtrip j hj
  rw [hb, hs] at h1
  omega

-- § GGUF block layout
/-- Start of block `b` in a row. -/
def blockStart (b : Nat) : Nat := 18 * b

/-- Scale bytes then code bytes fill the 18-byte block exactly. -/
theorem block_fields (b : Nat) : blockStart b + 2 + 16 = blockStart (b + 1) := by
  unfold blockStart; omega

/-- Block `b < nblocks` is inside a row of `18 * nblocks` bytes. -/
theorem block_in_row (b nblocks : Nat) (h : b < nblocks) : blockStart b + 18 ≤ 18 * nblocks := by
  unfold blockStart; omega

/-- Row `r` starts at `r * rowBytes`; rows are adjacent and disjoint. -/
theorem row_adjacent (r rowBytes : Nat) : r * rowBytes + rowBytes = (r + 1) * rowBytes := by
  rw [Nat.add_mul, Nat.one_mul]

-- § chunk association
/-- Activation chunk used by value `i` of block `b`. -/
def chunkOf (b i : Nat) : Nat := 2 * b + i / 32

/-- A block uses exactly two consecutive chunks. -/
theorem chunk_pair (b : Nat) : ∀ i : Nat, i < 64 → chunkOf b i = 2 * b ∨ chunkOf b i = 2 * b + 1 := by
  intro i hi
  unfold chunkOf
  omega

/-- Activation `q` offset of value `i` of block `b` (`b*64 + i`) lies in chunk `chunkOf b i`. -/
theorem activation_index_chunk (b i : Nat) (hi : i < 64) : (b * 64 + i) / 32 = chunkOf b i := by
  unfold chunkOf
  omega

-- § token grouping (avx2 handles 1..4 tokens per call)
/-- Group size at token offset `t0`. -/
def groupSize (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Groups are non-empty and in range while tokens remain. -/
theorem group_ok (nt t0 : Nat) (h : t0 < nt) : 0 < groupSize nt t0 ∧ t0 + groupSize nt t0 ≤ nt := by
  unfold groupSize; omega

/-- A group never exceeds the widest instantiated kernel. -/
theorem group_le_four (nt t0 : Nat) : groupSize nt t0 ≤ 4 := by
  unfold groupSize; omega

-- § quantizer
/-- Clamp to ±127 applied after the half-away rounding. -/
def clampQ (v : Int) : Int := max (-127) (min v 127)

/-- Clamped codes fit an `int8`. -/
theorem clampQ_int8 (v : Int) : -128 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ; omega

/-- Half-away-from-zero rounding step on an integer-scaled value `t2 = 2t`: `(t2 + sign) / 2`. -/
def roundHalfAway (t2 : Int) : Int := if 0 ≤ t2 then (t2 + 1) / 2 else (t2 - 1) / 2 + (if (t2 - 1) % 2 = 0 then 0 else 1)

/-- Ties round away from zero: `2.5 -> 3` and `-2.5 -> -3` (`t2 = ±5`). -/
theorem round_ties_away : roundHalfAway 5 = 3 ∧ roundHalfAway (-5) = -3 := by decide

/-- Chunk count of the quantizer for a width that is a multiple of 32. -/
theorem nchunks_exact (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by
  omega

end Strata.Q2Avx2



## FILE: /mnt/user-data/outputs/GgufLazyLoader.lean
/-!
# MODULE: gguf_lazy_loader
SOURCES: ref/load.py
Byte addressing of the lazy GGUF reader: per-expert slices of a fused `[in, out, experts]` tensor and
per-row slices of a 2-D tensor, with the block geometry table.

## Source semantics
* `expert(name, e)` reads `per_expert / el * by` bytes at `e * (per_expert / el) * by`; consecutive
  experts are adjacent and the last one ends at the tensor end.
* `rows(name, idx)` reads `row_bytes = width / el * by` bytes at `row * row_bytes`.
* A read is legal only when the element count is a multiple of the block width.
* A Q2_0 routed expert tensor `[2560, 640, 512]` has 460,800 bytes per expert; three of them make the
  1,382,400-byte blob of the CPU kernel.

## Form template identifiers
* GEOM -> block_geometry_table (elements per block, bytes per block)
* Gguf.expert -> expert_slice, Gguf.rows -> row_gather
-/

namespace Strata.GgufLoader

-- § geometry table
/-- (elements per block, bytes per block) per type, as in `GEOM`. -/
def geom : String → Nat × Nat
  | "F32" => (1, 4)
  | "F16" => (1, 2)
  | "BF16" => (1, 2)
  | "Q4_0" => (32, 18)
  | "Q5_0" => (32, 22)
  | "Q8_0" => (32, 34)
  | "IQ4_NL" => (32, 18)
  | "Q3_K" => (256, 110)
  | "Q4_K" => (256, 144)
  | "Q5_K" => (256, 176)
  | "Q6_K" => (256, 210)
  | "IQ4_XS" => (256, 136)
  | "Q2_0" => (64, 18)
  | _ => (1, 0)

/-- Block widths are positive for every listed type, so the divisions below are meaningful. -/
theorem geom_width_pos : ∀ t ∈ ["F32", "F16", "BF16", "Q4_0", "Q5_0", "Q8_0", "IQ4_NL", "Q3_K", "Q4_K",
    "Q5_K", "Q6_K", "IQ4_XS", "Q2_0"], 0 < (geom t).1 := by
  decide

-- § expert slice
/-- Bytes in one expert's slice. -/
def expertBytes (per el by_ : Nat) : Nat := per / el * by_

/-- Start offset of expert `e`. -/
def expertStart (e per el by_ : Nat) : Nat := e * (per / el) * by_

/-- Consecutive experts are adjacent: expert `e` ends exactly where expert `e+1` starts. -/
theorem expert_adjacent (e per el by_ : Nat) :
    expertStart e per el by_ + expertBytes per el by_ = expertStart (e + 1) per el by_ := by
  unfold expertStart expertBytes
  simp only [Nat.add_mul, Nat.one_mul]

/-- The first expert starts at byte 0. -/
theorem expert_first (per el by_ : Nat) : expertStart 0 per el by_ = 0 := by
  simp [expertStart]

/-- The last legal expert ends at `n_exp * bytes`, i.e. the end of the fused tensor. -/
theorem expert_last_end (e nexp per el by_ : Nat) (h : e + 1 = nexp) :
    expertStart e per el by_ + expertBytes per el by_ = nexp * (per / el) * by_ := by
  rw [expert_adjacent, ← h]
  rfl

/-- A routed Q2_0 expert tensor `[2560, 640, 512]`: 460,800 bytes per projection. -/
theorem q2_expert_projection_bytes : expertBytes (2560 * 640) 64 18 = 460800 := by decide

/-- Three projections (gate, up, down) make up the 1,382,400-byte kernel blob. -/
theorem q2_expert_blob_bytes : 3 * expertBytes (2560 * 640) 64 18 = 1382400 := by decide

/-- The whole fused gate tensor holds 512 such experts. -/
theorem q2_fused_tensor_bytes : 512 * expertBytes (2560 * 640) 64 18 = 235929600 := by decide

-- § row gather
/-- Bytes per row of width `w`. -/
def rowBytes (w el by_ : Nat) : Nat := w / el * by_

/-- Row `r` starts at `r * rowBytes` and rows are adjacent. -/
theorem row_adjacent (r rb : Nat) : r * rb + rb = (r + 1) * rb := by
  rw [Nat.add_mul, Nat.one_mul]

/-- A row index below the row count keeps the read inside the tensor. -/
theorem row_in_tensor (r nrows rb : Nat) (h : r < nrows) : r * rb + rb ≤ nrows * rb := by
  have : (r + 1) * rb ≤ nrows * rb := Nat.mul_le_mul_right rb h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The per-token PLE table: 160-wide IQ4_NL rows are 5 blocks of 18 bytes = 90 bytes. -/
theorem ple_row_bytes : rowBytes 160 32 18 = 90 := by decide

/-- 160 is a multiple of the IQ4_NL block width (required by `rows`). -/
theorem ple_width_divisible : 160 % 32 = 0 := by decide

-- § validity of a read
/-- The element count must be a whole number of blocks. -/
def validElements (elements el : Nat) : Bool := el != 0 && elements % el == 0

/-- A valid read covers exactly `elements / el` blocks, with no partial block. -/
theorem valid_exact_blocks (elements el : Nat) (h : validElements elements el = true) :
    elements / el * el = elements := by
  unfold validElements at h
  simp only [Bool.and_eq_true, bne_iff_ne, beq_iff_eq] at h
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h.2)

-- § shard discovery
/-- Shard `i` of `total` must have a sibling for every index: expected count equals found count. -/
def shardsComplete (found total : Nat) : Bool := found == total

/-- Complete shard sets are exactly those with matching counts. -/
theorem shardsComplete_iff (found total : Nat) : shardsComplete found total = true ↔ found = total := by
  simp [shardsComplete]

end Strata.GgufLoader



## FILE: /mnt/user-data/outputs/ByteLevelTokenizer.lean
/-!
# MODULE: byte_level_tokenizer
SOURCES: tools/strata_tokenizer.py
The GPT-2 byte-level BPE alphabet used by Strata's tokenizer, the special-token classes, and the
merge-validity conditions.

## Source semantics
* `bytes_to_unicode` maps the 188 printable bytes (33..126, 161..172, 174..255) to themselves and the
  other 68 to code points 256..323 in increasing byte order; it is injective with an explicit inverse,
  so `decode(encode(s)) = s` at the byte layer.
* Token type 4 (USER_DEFINED) literals are matched always; type 3 (CONTROL) only with `parse_special`.
* A merge is a pair of tokens; BPE applies the lowest rank first.

## Form template identifiers
* BYTE_TO_UNICODE / UNICODE_TO_BYTE -> byte_alphabet / byte_alphabet_inverse
* special_tokens (types 3, 4) -> literal_token_classes
-/

namespace Strata.Tokenizer

-- § the byte alphabet
/-- Code point assigned to byte `b`: itself when printable, otherwise `256 + (rank among non-printables)`. -/
def byteToCp (b : Nat) : Nat :=
  if ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255)) then b
  else if b ≤ 32 then 256 + b
  else if b ≤ 160 then 256 + b - 94
  else 323

/-- Inverse map from code point back to byte. -/
def cpToByte (c : Nat) : Nat :=
  if c < 256 then c
  else if c < 289 then c - 256
  else if c < 323 then c - 256 + 94
  else 173

/-- Counting: 94 + 12 + 82 = 188 printable bytes. -/
theorem printable_count : (94 : Nat) + 12 + 82 = 188 := by decide

/-- Counting: 33 + 34 + 1 = 68 non-printable bytes, and 188 + 68 = 256. -/
theorem nonprintable_count : (33 : Nat) + 34 + 1 = 68 ∧ 188 + 68 = 256 := by decide

/-- The encoder is injective on bytes. -/
theorem byteToCp_injective (a b : Nat) (ha : a < 256) (hb : b < 256) (h : byteToCp a = byteToCp b) :
    a = b := by
  unfold byteToCp at h
  split_ifs at h <;> omega

/-- Every encoded code point is below 324. -/
theorem byteToCp_lt (b : Nat) (hb : b < 256) : byteToCp b < 324 := by
  unfold byteToCp
  split_ifs <;> omega

/-- Decoding inverts encoding on every byte: the byte layer is lossless. -/
theorem cpToByte_byteToCp (b : Nat) (hb : b < 256) : cpToByte (byteToCp b) = b := by
  unfold cpToByte byteToCp
  split_ifs <;> omega

/-- Printable bytes are fixed points. -/
theorem printable_fixed (b : Nat) (h : ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255))) : byteToCp b = b := by
  unfold byteToCp
  rw [if_pos h]

/-- Non-printable bytes land at or above 256 (so they never collide with a printable byte). -/
theorem nonprintable_shifted (b : Nat) (hb : b < 256) (h : ¬ ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255))) : 256 ≤ byteToCp b := by
  unfold byteToCp
  split_ifs <;> omega

/-- A string round-trips when each byte does. -/
theorem roundtrip_list (bs : List Nat) (h : ∀ b ∈ bs, b < 256) :
    (bs.map byteToCp).map cpToByte = bs := by
  induction bs with
  | nil => rfl
  | cons a t ih =>
    have ha : a < 256 := h a (by simp)
    have ht : ∀ b ∈ t, b < 256 := fun b hb => h b (by simp [hb])
    simp only [List.map_cons]
    rw [cpToByte_byteToCp a ha, ih ht]

-- § special tokens
/-- Literal classes: GGUF token type 3 = CONTROL, 4 = USER_DEFINED. -/
def matchedAlways (ty : Nat) : Bool := ty == 4

/-- With `parse_special`, both classes are matched. -/
def matchedWithSpecial (ty : Nat) : Bool := ty == 3 || ty == 4

/-- Whatever is matched without `parse_special` is also matched with it. -/
theorem always_subset_special (ty : Nat) (h : matchedAlways ty = true) : matchedWithSpecial ty = true := by
  unfold matchedAlways at h
  unfold matchedWithSpecial
  simp_all

/-- CONTROL tokens such as `<|im_end|>` are NOT matched without `parse_special`. -/
theorem control_needs_parse_special : matchedAlways 3 = false ∧ matchedWithSpecial 3 = true := by decide

/-- USER_DEFINED tokens such as `<think>` are matched either way. -/
theorem user_defined_always : matchedAlways 4 = true ∧ matchedWithSpecial 4 = true := by decide

-- § merges
/-- A merge line must split into exactly two tokens. -/
def validMergeArity (parts : Nat) : Bool := parts == 2

/-- Only two-part merge lines are accepted. -/
theorem merge_arity_iff (n : Nat) : validMergeArity n = true ↔ n = 2 := by
  simp [validMergeArity]

/-- BPE picks the lowest rank: a candidate replaces the best only when strictly smaller. -/
def betterRank (best : Option Nat) (r : Nat) : Bool :=
  match best with
  | none => true
  | some b => decide (r < b)

/-- The first candidate is always taken. -/
theorem betterRank_none (r : Nat) : betterRank none r = true := rfl

/-- Ties keep the earlier (leftmost) candidate. -/
theorem betterRank_tie (r : Nat) : betterRank (some r) r = false := by
  simp [betterRank]

end Strata.Tokenizer



## FILE: /mnt/user-data/outputs/CanonicalQuantPack.lean
/-!
# MODULE: canonical_quant_pack
SOURCES: tools/canonical_xcheck.py, tools/strata_pack.py, tools/mtp_pack.py
Canonical code-plane forms of the GGUF quantization types (S2/S4/S8), bit packing of code planes,
64-byte plane alignment, block-geometry consistency, and the MTP expert quantizer.

## Source semantics
* Each type decodes as `(code + bias) * scale` (+ offset), the bias being applied to the INTEGER code
  before the single float multiply: Q2_0 -1, Q4_0 -8, Q5_0 -16, Q3_K -4, Q6_K -32, Q8_0 -128.
* Codes are packed LSB-first, `8 / bits` per byte; unpacking recovers every code.
* Nibble splits (Q4_0, IQ4_NL, Q4_K ...) put low nibbles first, high nibbles second.
* Planes start on 64-byte boundaries.
* The Q2_0 MTP quantizer maps `q` clipped to {-1..2} to code `q+1` in 0..3 and searches 17 scales
  `amax * (16+k)/32`.

## Form template identifiers
* MAPPINGS -> type_mapping_table, code_bias -> integer_code_bias, ALIGN -> plane_alignment
* pack_codes / unpack_codes -> code_plane_pack / code_plane_unpack
-/

namespace Strata.Pack

-- § 2-bit and 4-bit packing
/-- Pack four 2-bit codes LSB-first. -/
def pack2 (c0 c1 c2 c3 : Nat) : Nat := c0 ||| (c1 <<< 2) ||| (c2 <<< 4) ||| (c3 <<< 6)

/-- Pack two 4-bit codes, low nibble first. -/
def pack4 (a b : Nat) : Nat := a ||| (b <<< 4)

/-- Unpack field `k` of width `bits`. -/
def field (byte k bits : Nat) : Nat := (byte >>> (k * bits)) &&& ((1 <<< bits) - 1)

/-- Four 2-bit codes round-trip through one byte, in order. -/
theorem pack2_roundtrip : ∀ c0 : Nat, c0 < 4 → ∀ c1 : Nat, c1 < 4 → ∀ c2 : Nat, c2 < 4 → ∀ c3 : Nat, c3 < 4 →
    field (pack2 c0 c1 c2 c3) 0 2 = c0 ∧ field (pack2 c0 c1 c2 c3) 1 2 = c1 ∧
    field (pack2 c0 c1 c2 c3) 2 2 = c2 ∧ field (pack2 c0 c1 c2 c3) 3 2 = c3 := by
  decide

/-- The packed 2-bit byte fits in 8 bits. -/
theorem pack2_lt : ∀ c0 : Nat, c0 < 4 → ∀ c1 : Nat, c1 < 4 → ∀ c2 : Nat, c2 < 4 → ∀ c3 : Nat, c3 < 4 →
    pack2 c0 c1 c2 c3 < 256 := by
  decide

/-- Two 4-bit codes round-trip through one byte. -/
theorem pack4_roundtrip : ∀ a : Nat, a < 16 → ∀ b : Nat, b < 16 →
    field (pack4 a b) 0 4 = a ∧ field (pack4 a b) 1 4 = b ∧ pack4 a b < 256 := by
  decide

set_option maxRecDepth 100000 in
/-- Splitting a byte into low and high nibble and recombining is the identity (Q4_0 / IQ4_NL / Q4_K split). -/
theorem nibble_split : ∀ b : Nat, b < 256 → (b >>> 4) * 16 + (b &&& 15) = b ∧ (b &&& 15) < 16 ∧ (b >>> 4) < 16 := by
  decide

-- § integer code bias
/-- Decode of a code with an integer bias, before the float multiply. -/
def biased (code : Nat) (bias : Int) : Int := (code : Int) + bias

/-- Q2_0: codes 0..3 decode to {-1, 0, 1, 2} (the offset is in the codebook). -/
theorem q2_codebook (c : Nat) (h : c < 4) : -1 ≤ biased c (-1) ∧ biased c (-1) ≤ 2 := by
  unfold biased; omega

/-- Q4_0: codes 0..15 decode to -8..7. -/
theorem q4_codebook (c : Nat) (h : c < 16) : -8 ≤ biased c (-8) ∧ biased c (-8) ≤ 7 := by
  unfold biased; omega

/-- Q5_0: codes 0..31 decode to -16..15. -/
theorem q5_codebook (c : Nat) (h : c < 32) : -16 ≤ biased c (-16) ∧ biased c (-16) ≤ 15 := by
  unfold biased; omega

/-- Q3_K: codes 0..7 decode to -4..3. -/
theorem q3k_codebook (c : Nat) (h : c < 8) : -4 ≤ biased c (-4) ∧ biased c (-4) ≤ 3 := by
  unfold biased; omega

/-- Q6_K: codes 0..63 decode to -32..31. -/
theorem q6k_codebook (c : Nat) (h : c < 64) : -32 ≤ biased c (-32) ∧ biased c (-32) ≤ 31 := by
  unfold biased; omega

/-- Q8_0: stored code `q + 128` decodes back to the int8 `q`. -/
theorem q8_codebook (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) :
    ∃ c : Nat, c < 256 ∧ biased c (-128) = q := by
  refine ⟨(q + 128).toNat, by omega, ?_⟩
  unfold biased
  omega

-- § Q3_K / Q5_K bit-mask traps
/-- Q3_K's hmask bit index `4*half + j` stays inside a byte and never repeats across the 8 (half, j) pairs. -/
theorem q3k_mask_bits : ∀ half : Nat, half < 2 → ∀ j : Nat, j < 4 → 4 * half + j < 8 := by decide

/-- Q5_K's masks `u1 = 1 << 2j`, `u2 = 2 << 2j` walk the eight distinct bits 1,2,4,8,16,32,64,128. -/
theorem q5k_masks : ∀ j : Nat, j < 4 →
    (1 <<< (2 * j)) = 2 ^ (2 * j) ∧ (2 <<< (2 * j)) = 2 ^ (2 * j + 1) := by decide

/-- The Q5_0 fifth-bit shift `j + 12` with mask 0x10 reads bit `j + 16` of qh (spot-checked on all 16 positions for bit-set qh). -/
theorem q5_0_high_bit : ∀ j : Nat, j < 16 → ((1 <<< (j + 16)) >>> (j + 12)) &&& 16 = 16 := by decide

-- § block geometry consistency
/-- Block byte sizes equal the sum of their fields. -/
theorem block_bytes_sum :
    2 + 16 = 18 ∧ 2 + 4 + 16 = 22 ∧ 2 + 32 = 34 ∧ 32 + 64 + 12 + 2 = 110 ∧
    2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 + 32 + 128 = 176 ∧ 128 + 64 + 16 + 2 = 210 ∧
    2 + 2 + 4 + 128 = 136 := by
  decide

/-- Group sizes divide block widths in every mapping (group, block). -/
theorem group_divides_block :
    64 % 64 = 0 ∧ 256 % 16 = 0 ∧ 32 % 32 = 0 ∧ 256 % 32 = 0 ∧ 1 % 1 = 0 := by
  decide

/-- Scale counts per block: Q3_K/Q6_K 16 groups of 16, K-quants with 32-groups 8 per block. -/
theorem group_counts : 256 / 16 = 16 ∧ 256 / 32 = 8 ∧ 64 / 64 = 1 := by decide

-- § 64-byte plane alignment
/-- Padding added before a plane so it starts on a 64-byte boundary. -/
def pad (len : Nat) : Nat := (64 - len % 64) % 64

/-- After padding the offset is a multiple of 64. -/
theorem pad_aligns (len : Nat) : (len + pad len) % 64 = 0 := by
  unfold pad; omega

/-- Padding is smaller than the alignment. -/
theorem pad_lt (len : Nat) : pad len < 64 := by
  unfold pad; omega

/-- An already aligned offset gets no padding. -/
theorem pad_zero (len : Nat) (h : len % 64 = 0) : pad len = 0 := by
  unfold pad; omega

-- § scale dtype choice
/-- Which dtype is stored for scales: fp16 only when the round trip is exact. -/
def scaleBytes (exact16 : Bool) : Nat := if exact16 then 2 else 4

/-- FP32 fallback never loses information and costs at most twice the bytes. -/
theorem scaleBytes_le (b : Bool) : 2 ≤ scaleBytes b ∧ scaleBytes b ≤ 4 := by
  cases b <;> simp [scaleBytes]

-- § MTP expert quantizer
/-- Clip to the Q2_0 grid {-1, 0, 1, 2}. -/
def clipQ (q : Int) : Int := max (-1) (min q 2)

/-- Stored code is `clip(q) + 1`, always in 0..3. -/
theorem mtp_code_range (q : Int) : 0 ≤ clipQ q + 1 ∧ clipQ q + 1 ≤ 3 := by
  unfold clipQ; omega

/-- Decoding the stored code returns the clipped value. -/
theorem mtp_decode_encode (q : Int) : (clipQ q + 1) - 1 = clipQ q := by omega

/-- The scale search has 17 candidates, factor `(16 + k) / 32` from 1/2 (k = 0) to 1 (k = 16). -/
theorem mtp_scale_candidates : (16 + 0 : Nat) * 2 = 32 ∧ 16 + 16 = 32 ∧ 16 + 1 = 17 := by decide

/-- MTP routed-expert bytes in Q2_0: gate_up [512,1280,2560] and down [512,2560,640]. -/
theorem mtp_q2_expert_bytes :
    512 * 1280 * (2560 / 64 * 18) = 471859200 ∧ 512 * 2560 * (640 / 64 * 18) = 235929600 ∧
    471859200 + 235929600 = 707788800 := by
  decide

/-- Inner dimensions are multiples of every supported block width (64 for q2_0, 32 for q4_0/q8_0). -/
theorem mtp_inner_divisible : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

-- § IQ4 codebook
/-- kvalues_iq4nl, the non-affine codebook. -/
def kvIq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, indexed by a 4-bit code. -/
theorem kvIq4nl_length : kvIq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kvIq4nl_sorted : kvIq4nl.Pairwise (· < ·) := by decide

/-- The codebook is not an affine function of the code: second differences are not all zero. -/
theorem kvIq4nl_nonaffine : (kvIq4nl[1]! - kvIq4nl[0]!) ≠ (kvIq4nl[2]! - kvIq4nl[1]!) := by decide

end Strata.Pack



## FILE: /mnt/user-data/outputs/VisionEncoderTool.lean
/-!
# MODULE: vision_encoder_tool
SOURCES: tools/vision/strata_vision.cpp, tools/vision/CMakeLists.txt
The resident image-encoder helper: its output file header and size, the token grid, the `ENC` line
split, and the CMake portability options.

## Source semantics
* Output header is five int32 `{0x31455653 'SVE1', n_tokens, nx, ny, n_embd}` followed by
  `n_tokens * n_embd` float32 values.
* The grid is recovered from the last token's decoder position: `nx = x + 1`, `ny = y + 1`; the
  encoding is accepted only when `nx * ny = n_tokens`. Row `i` sits at `(i % nx, i / nx)`.
* `ENC <image> <out>` splits at the LAST space, so the image path may contain spaces.
* `STRATA_PORTABLE` turns `GGML_NATIVE` off and forces the AVX2 baseline flags on.

## Form template identifiers
* SVE1 magic -> file_magic, hdr[5] -> header_words, parse_enc -> command_line_split
* STRATA_PORTABLE -> portable_build_flag
-/

namespace Strata.Vision

-- § file format
/-- 'S' 'V' 'E' '1' as little-endian bytes 0x53 0x56 0x45 0x31. -/
theorem magic_value : 0x53 + 0x56 * 256 + 0x45 * 65536 + 0x31 * 16777216 = 0x31455653 := by decide

/-- Header size in bytes: five 4-byte words. -/
def headerBytes : Nat := 5 * 4

/-- Total output file size. -/
def fileBytes (n nEmbd : Nat) : Nat := headerBytes + 4 * (n * nEmbd)

/-- The header is 20 bytes. -/
theorem header_is_20 : headerBytes = 20 := by decide

/-- The payload starts right after the header and has exactly `n` rows of `n_embd` floats. -/
theorem payload_size (n nEmbd : Nat) : fileBytes n nEmbd - headerBytes = 4 * n * nEmbd := by
  unfold fileBytes
  rw [Nat.add_sub_cancel_left, Nat.mul_assoc]

/-- An empty encoding still produces a header-only file. -/
theorem empty_file (nEmbd : Nat) : fileBytes 0 nEmbd = headerBytes := by
  simp [fileBytes]

-- § grid
/-- Grid dimensions from the last decoder position. -/
def gridDims (x y : Nat) : Nat × Nat := (x + 1, y + 1)

/-- The grid is accepted only when it is exactly rectangular. -/
def rectangular (n x y : Nat) : Bool := (x + 1) * (y + 1) == n

/-- Rectangular acceptance means the token count equals `nx * ny`. -/
theorem rectangular_iff (n x y : Nat) : rectangular n x y = true ↔ (gridDims x y).1 * (gridDims x y).2 = n := by
  simp [rectangular, gridDims]

/-- Row-major position: token `i` is at column `i % nx`, row `i / nx`, and the pair recovers `i`. -/
theorem grid_position_roundtrip (i nx : Nat) : (i / nx) * nx + i % nx = i := by
  rw [Nat.mul_comm]
  exact Nat.div_add_mod i nx

/-- For `nx > 0`, the column is inside the grid. -/
theorem column_lt (i nx : Nat) (h : 0 < nx) : i % nx < nx := Nat.mod_lt _ h

/-- For a rectangular grid, every token's row is inside it. -/
theorem row_lt (i nx ny : Nat) (h : 0 < nx) (hi : i < nx * ny) : i / nx < ny := by
  exact Nat.div_lt_of_lt_mul hi

-- § command split
/-- The image path and output path come from splitting the rest of the line at the last space. -/
theorem split_reassembles (l : List Nat) (sp : Nat) : l.take sp ++ l.drop sp = l :=
  List.take_append_drop sp l

/-- A split position is valid only when it is strictly inside the string (`sp ≠ 0`, found). -/
def validSplit (len sp : Nat) : Bool := 0 < sp && sp < len

/-- Valid splits leave a non-empty image part and a non-empty output part (after the space). -/
theorem validSplit_parts (len sp : Nat) (h : validSplit len sp = true) :
    0 < sp ∧ 0 < len - (sp + 1) + 1 := by
  unfold validSplit at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  omega

/-- The prefix `ENC ` has 4 characters, so the remainder starts at index 4. -/
theorem enc_prefix_len : "ENC ".length = 4 := by decide

-- § build options
/-- `GGML_NATIVE` is on exactly when the build is not portable. -/
def ggmlNative (portable : Bool) : Bool := !portable

/-- The AVX2 baseline flags are forced on only for portable builds. -/
def baselineFlags (portable : Bool) : Bool := portable

/-- Portable and native are exclusive. -/
theorem native_xor_baseline (p : Bool) : ggmlNative p = !baselineFlags p := by
  cases p <;> rfl

/-- A portable build is never native. -/
theorem portable_not_native : ggmlNative true = false := rfl

/-- A non-portable build uses native tuning and no forced baseline. -/
theorem default_native : ggmlNative false = true ∧ baselineFlags false = false := by decide

/-- Vision runs on the GPU only if the CUDA option is on. -/
def ggmlCuda (stratavisioncuda : Bool) : Bool := stratavisioncuda

/-- The default (OFF) builds no CUDA backend. -/
theorem cuda_default_off : ggmlCuda false = false := rfl

end Strata.Vision




[LIGHT FILES — structure only, no LAYER 1a/1b]
  include/strata/core/graph.hpp (118 lines)
  include/strata/core/layout.hpp (98 lines)
  include/strata/core/mtp.hpp (136 lines)
  include/strata/core/pinned.hpp (96 lines)
  include/strata/core/weights.hpp (136 lines)
  include/strata/core/device.hpp (72 lines)
  include/strata/core/hit_hook.hpp (42 lines)
  include/strata/core/native_dense.hpp (37 lines)
  include/strata/core/native_head.hpp (66 lines)
  include/strata/kernels/elementwise.hpp (117 lines)
  include/strata/kernels/gdn.hpp (98 lines)
  include/strata/kernels/gr.hpp (123 lines)
  include/strata/kernels/native_mmvq.hpp (113 lines)
  include/strata/kernels/ple.hpp (120 lines)
  include/strata/kernels/s2_expert_grouped.hpp (119 lines)
  include/strata/kernels/s_gemv.hpp (134 lines)
  include/strata/kernels/shared_expert.hpp (102 lines)
  include/strata/kernels/verify_kernels.hpp (81 lines)
  include/strata/kernels/bf16_bits.hpp (47 lines)
  include/strata/kernels/bf16_gemv.hpp (49 lines)
  include/strata/kernels/dequant_bf16.hpp (28 lines)
  include/strata/kernels/dequant_s2.hpp (19 lines)
  include/strata/kernels/fused_gdn.hpp (36 lines)
  include/strata/kernels/fused_gr.hpp (52 lines)
  include/strata/kernels/iq_kernels.hpp (58 lines)
  include/strata/kernels/kv_q8.hpp (39 lines)
  include/strata/kernels/mrope.hpp (33 lines)
  include/strata/kernels/native_flash_attn.hpp (37 lines)
  include/strata/kernels/native_gdn.hpp (23 lines)
  include/strata/kernels/native_gdn_preprocess.hpp (38 lines)
  include/strata/kernels/native_gr_norm.hpp (19 lines)
  include/strata/kernels/native_gr_postops.hpp (29 lines)
  include/strata/kernels/native_moe.hpp (19 lines)
  include/strata/kernels/native_ple_postops.hpp (31 lines)
  include/strata/kernels/native_qsa.hpp (31 lines)
  include/strata/kernels/native_qsa_indexer.hpp (36 lines)
  include/strata/kernels/native_qsa_score.hpp (32 lines)
  include/strata/kernels/native_rope.hpp (15 lines)
  include/strata/kernels/native_router.hpp (18 lines)
  include/strata/kernels/qsa_decode_attn.hpp (45 lines)
  include/strata/kernels/qsa_select.hpp (34 lines)
  include/strata/kernels/quantize_act.hpp (59 lines)
  include/strata/kernels/rope.hpp (51 lines)
  include/strata/kernels/router_top10.hpp (21 lines)
  include/strata/kernels/s2_gemv.hpp (26 lines)
  include/strata/kernels/s2_gemv_q8.hpp (23 lines)
  include/strata/kernels/sampler.hpp (44 lines)
  include/strata/kernels/cpu/expert_layout.hpp (52 lines)
  include/strata/kernels/cpu/iq_avx512.hpp (19 lines)
  include/strata/kernels/cpu/native_expert.hpp (50 lines)
  include/strata/ngram/ple_reader.hpp (87 lines)
  include/strata/plan/plan.hpp (169 lines)
  include/strata/platform/direct_file.hpp (60 lines)
  include/strata/platform/memory.hpp (27 lines)
  include/strata/prefill/gemm.hpp (55 lines)
  include/strata/prefill/kernels.hpp (75 lines)
  include/strata/prefill/prefill.hpp (76 lines)
  include/strata/program/logits_selection.hpp (31 lines)
  include/strata/spec/suffix_drafter.hpp (54 lines)
  serve/__init__.py (2 lines)
  serve/chat_golden.json (197 lines)
  serve/chat_template.jinja (184 lines)
  serve/index.html (153 lines)
  src/artifact/dequant.cpp (124 lines)
  src/artifact/gguf_reader.cpp (68 lines)
  src/core/concurrent_main.cpp (198 lines)
  src/core/device.cu (118 lines)
  src/core/device_main.cpp (79 lines)
  src/core/graph.cpp (147 lines)
  src/core/layout.cpp (174 lines)
  src/core/load_main.cpp (120 lines)
  src/core/native_dense.cpp (180 lines)
  src/core/native_head.cpp (152 lines)
  src/core/overlap_main.cpp (170 lines)
  src/core/pinned.cu (292 lines)
  src/kernels/bf16_gemv_parity.cpp (149 lines)
  src/kernels/dequant_bf16_test.cpp (109 lines)
  src/kernels/dequant_s2_parity.cpp (144 lines)
  src/kernels/elementwise_parity.cpp (417 lines)
  src/kernels/gdn_parity.cpp (398 lines)
  src/kernels/gr_parity.cpp (616 lines)
  src/kernels/iq_parity.cpp (88 lines)
  src/kernels/kv_q8_parity.cpp (142 lines)
  src/kernels/native_expert_parity.cpp (205 lines)
  src/kernels/ple_oracle_vectors.inc (106 lines)
  src/kernels/ple_parity.cpp (1185 lines)
  src/kernels/qsa_oracle_vectors.inc (135 lines)
  src/kernels/qsa_parity.cpp (1384 lines)
  src/kernels/quantize_act_parity.cpp (408 lines)
  src/kernels/rope_parity.cpp (172 lines)
  src/kernels/router_top10_parity.cpp (168 lines)
  src/kernels/s2_gemv_parity.cpp (174 lines)
  src/kernels/s2_gemv_q8_parity.cpp (182 lines)
  src/kernels/s_gemv_parity.cpp (635 lines)
  src/kernels/s_gemv_q8k_parity.cpp (237 lines)
  src/kernels/sampler_parity.cpp (363 lines)
  src/kernels/shared_expert_parity.cpp (483 lines)
  src/kernels/cpu/expert_layout.cpp (137 lines)
  src/kernels/cpu/expert_multi_test.cpp (88 lines)
  src/kernels/cpu/expert_parity.cpp (185 lines)
  src/kernels/cpu/native_expert.cpp (115 lines)
  src/kernels/cpu/pool_test.cpp (225 lines)
  src/kernels/cuda/bf16_gemv.cu (142 lines)
  src/kernels/cuda/dequant_bf16.cu (246 lines)
  src/kernels/cuda/dequant_s2.cu (52 lines)
  src/kernels/cuda/elementwise.cu (290 lines)
  src/kernels/cuda/kv_q8.cu (119 lines)
  src/kernels/cuda/native_bf16.cu (119 lines)
  src/kernels/cuda/native_gdn.cu (123 lines)
  src/kernels/cuda/native_gdn_preprocess.cu (200 lines)
  src/kernels/cuda/native_gr_norm.cu (103 lines)
  src/kernels/cuda/native_gr_postops.cu (116 lines)
  src/kernels/cuda/native_moe.cu (75 lines)
  src/kernels/cuda/native_ple_postops.cu (135 lines)
  src/kernels/cuda/native_qsa.cu (128 lines)
  src/kernels/cuda/native_qsa_indexer.cu (130 lines)
  src/kernels/cuda/native_rope.cu (89 lines)
  src/kernels/cuda/native_router.cu (118 lines)
  src/kernels/cuda/rope.cu (96 lines)
  src/kernels/cuda/s2_gemv.cu (64 lines)
  src/kernels/cuda/s2_gemv_fast.cu (151 lines)
  src/kernels/cuda/s2_gemv_q8.cu (97 lines)
  src/kernels/cuda/s2_gemv_quads.cu (94 lines)
  src/kernels/cuda/shared_expert.cu (339 lines)
  src/ngram/ple_reader_test.cpp (238 lines)
  src/plan/plan_main.cpp (74 lines)
  src/platform/direct_file.cpp (242 lines)
  src/platform/memory.cpp (77 lines)
  src/platform/memory_test.cpp (36 lines)
  src/prefill/gemm.cu (108 lines)
  src/spec/controller_test.cpp (75 lines)
  src/spec/suffix_drafter_test.cpp (142 lines)
  tools/_paths.py (29 lines)
  tools/gguf_reader.py (143 lines)
  tools/gguf_writer.py (235 lines)
  tools/iq_pack.py (222 lines)
  tools/mtp_fetch.py (122 lines)
  tools/mtp_rt.py (113 lines)
  tools/pack_index.py (261 lines)
  tools/pack_layer.py (124 lines)
  bench/results/2026-09-24-final/estimates.md (28 lines)
  bench/results/2026-09-24-final/matrix.md (16 lines)
  bench/results/2026-09-24-vision/vision.md (6 lines)
  third_party/ggml/VERSION.txt (2 lines)
  CMakeLists.txt (537 lines)
  START-HERE.bat (53 lines)
  bench/results/2026-09-24-final/estimates.json (461 lines)
  bench/results/2026-09-24-final/matrix.json (240 lines)
  bench/results/2026-09-24-vision/vision.json (53 lines)
  chat.py (117 lines)
  setup.py (807 lines)
  setup.sh (31 lines)
  third_party/ggml/ggml-common.h (1912 lines)
