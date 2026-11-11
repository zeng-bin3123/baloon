/-!
# MODULE: prefill_kernel_indexing
SOURCES: src/prefill/kernels.cu
The index arithmetic and integer contracts of the batched prompt kernels: the four-stream
hyper-connection layout, the Strata 2-bit expert blob layout and its dequantization, the paged KV append
with optional int8 codes, the GDN causal-conv window, the router's warp arg-max reduction, and the
interleaved/split projection layouts.

## Source semantics
* `R` is `[T][HC=4][N=2560]`; flat index `i` decomposes as `t = i / D`, `c = (i % D) / N`, `d = i % N`
  with `D = N·HC = 10240`.
* Blob: gate/up codes `[1280][640 B]`, down codes `[2560][160 B]`, gate/up scales `[1280][40]` f16,
  down scales `[2560][10]` f16 — 1,382,400 bytes.  Each code byte holds four 2-bit codes; weight value
  is `((c >> 2k) & 3) - 1 ∈ {-1, 0, 1, 2}` times the block scale of its 64-wide group.
* KV append: row `(page·2 + kvh)·page_size + pos % page_size`; int8 codes are clamped to `[-127, 127]`.
* GDN conv: each channel keeps the last three inputs; after a chunk the history is its last three.
* Router: ten rounds of a warp arg-max whose pairwise step prefers the larger probability and, on a
  tie, the smaller expert id — an order-independent reduction.
* Interleaved SwiGLU: gate at `2k`, up at `2k+1`; QSA `Qf` rows hold 256 query values then 256 gate values
  per head.

## Form template identifiers
gr_* kernels' index math → grIndex · blob offsets → oDCodes / oGuSc / oDSc / blobBytes ·
`(c >> 2k) & 3` → code2 · kv_append row → kvRow · int8 clamp → clampQ · conv history → convStep ·
the shuffle step of route_kernel → pick · swiglu_il layout → gateIdx / upIdx
-/

namespace Strata.Prefill.Kernels

-- § hyper-connection layout

/-- Flat index of `R[t][c][d]`. -/
def grIndex (t c d : Nat) : Nat := t * 10240 + c * 2560 + d

/-- The kernels' decomposition of a flat index recovers the coordinates. -/
theorem grIndex_decomp (i : Nat) : grIndex (i / 10240) ((i % 10240) / 2560) (i % 2560) = i := by
  unfold grIndex
  omega

/-- Recovered coordinates are in range: stream below 4, element below 2560. -/
theorem grIndex_ranges (i : Nat) : (i % 10240) / 2560 < 4 ∧ i % 2560 < 2560 := by
  omega

/-- Distinct coordinates in range are distinct cells: `gr_write` writes every cell once. -/
theorem grIndex_injective {t t' c c' d d' : Nat} (hc : c < 4) (hc' : c' < 4) (hd : d < 2560)
    (hd' : d' < 2560) (h : grIndex t c d = grIndex t' c' d') : t = t' ∧ c = c' ∧ d = d' := by
  unfold grIndex at h
  omega

/-- `gr_mix` reads four streams of token `t` without leaving that token's row. -/
theorem grIndex_in_token {t c d : Nat} (hc : c < 4) (hd : d < 2560) :
    t * 10240 ≤ grIndex t c d ∧ grIndex t c d < (t + 1) * 10240 := by
  unfold grIndex
  omega

-- § the 2-bit expert blob

/-- Offset of the down codes. -/
def oDCodes : Nat := 1280 * 640
/-- Offset of the gate/up scales. -/
def oGuSc : Nat := oDCodes + 2560 * 160
/-- Offset of the down scales. -/
def oDSc : Nat := oGuSc + 1280 * 40 * 2
/-- Total blob size. -/
def blobBytes : Nat := oDSc + 2560 * 10 * 2

/-- The blob is 1,382,400 bytes — the 1.38 MB slot the cache verifies at startup. -/
theorem blobBytes_value : blobBytes = 1382400 := by decide

/-- The regions are laid out back to back in increasing order. -/
theorem blob_regions_ordered : oDCodes < oGuSc ∧ oGuSc < oDSc ∧ oDSc < blobBytes := by decide

/-- One gate/up row is 640 code bytes = 2560 weights; the down row is 160 bytes = 640 weights. -/
theorem blob_row_widths : 640 * 4 = 2560 ∧ 160 * 4 = 640 := by decide

/-- The 2-bit code `k` of a code byte. -/
def code2 (c k : Nat) : Nat := (c >>> (2 * k)) &&& 3

/-- Every code is in `0..3`, so the weight multiplier `code - 1` is in `{-1, 0, 1, 2}`. -/
theorem code2_lt (c k : Nat) : code2 c k < 4 := by
  unfold code2
  have h : (c >>> (2 * k)) &&& 3 = (c >>> (2 * k)) % 4 := Nat.and_two_pow_sub_one_eq_mod _ 2
  rw [h]
  omega

/-- The multiplier as an integer. -/
def weightMult (c k : Nat) : Int := (code2 c k : Int) - 1

/-- Range of the dequantized multiplier. -/
theorem weightMult_range (c k : Nat) : -1 ≤ weightMult c k ∧ weightMult c k ≤ 2 := by
  have h := code2_lt c k
  unfold weightMult
  omega

/-- Output index of gate/up weight `k` of code byte `byte` in row `row`. -/
def guOut (row byte k : Nat) : Nat := row * 2560 + byte * 4 + k

/-- Gate/up outputs stay in the `1280 × 2560` FP16 buffer and never collide. -/
theorem guOut_range_inj {row byte k row' byte' k' : Nat} (hr : row < 1280) (hb : byte < 640) (hk : k < 4)
    (hb' : byte' < 640) (hk' : k' < 4) :
    guOut row byte k < 1280 * 2560 ∧
      (guOut row byte k = guOut row' byte' k' → row = row' ∧ byte = byte' ∧ k = k') := by
  unfold guOut
  constructor
  · omega
  · intro h; omega

/-- Down outputs stay in the `2560 × 640` buffer and never collide. -/
theorem downOut_range_inj {row byte k row' byte' k' : Nat} (hr : row < 2560) (hb : byte < 160) (hk : k < 4)
    (hb' : byte' < 160) (hk' : k' < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 ∧
      (row * 640 + byte * 4 + k = row' * 640 + byte' * 4 + k' → row = row' ∧ byte = byte' ∧ k = k') := by
  constructor
  · omega
  · intro h; omega

/-- A gate/up byte's scale is that row's scale group `(4·byte)/64`, one of 40. -/
theorem gu_scale_group {byte : Nat} (hb : byte < 640) : byte * 4 / 64 < 40 := by omega

/-- A down byte's scale group is one of 10. -/
theorem down_scale_group {byte : Nat} (hb : byte < 160) : byte * 4 / 64 < 10 := by omega

-- § paged KV append

/-- KV row of position `pos` for kv head `kvh` on page `page`. -/
def kvRow (page kvh pos ps : Nat) : Nat := (page * 2 + kvh) * ps + pos % ps

/-- Helper: a row `a·ps + o` with `o < ps` determines `a` and `o`. -/
theorem row_split_unique {ps a a' o o' : Nat} (ho : o < ps) (ho' : o' < ps)
    (h : a * ps + o = a' * ps + o') : a = a' ∧ o = o' := by
  have hp : 0 < ps := by omega
  have hdiv : ∀ x y, y < ps → (x * ps + y) / ps = x := by
    intro x y hy
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hy, Nat.zero_add]
  have hmod : ∀ x y, y < ps → (x * ps + y) % ps = y := by
    intro x y hy
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hy]
  have h1 := hdiv a o ho
  have h2 := hdiv a' o' ho'
  have h3 := hmod a o ho
  have h4 := hmod a' o' ho'
  rw [h] at h1 h3
  exact ⟨h1.symm.trans h2, h3.symm.trans h4⟩

/-- Distinct (page, kv head, in-page offset) never share a KV row. -/
theorem kvRow_injective {ps page page' kvh kvh' pos pos' : Nat} (hps : 0 < ps) (hk : kvh < 2)
    (hk' : kvh' < 2) (h : kvRow page kvh pos ps = kvRow page' kvh' pos' ps) :
    page = page' ∧ kvh = kvh' ∧ pos % ps = pos' % ps := by
  unfold kvRow at h
  obtain ⟨ha, ho⟩ := row_split_unique (Nat.mod_lt pos hps) (Nat.mod_lt pos' hps) h
  omega

/-- The int8 code clamp. -/
def clampQ (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- Codes are symmetric int8: `-128` is never written. -/
theorem clampQ_range (q : Int) : -127 ≤ clampQ q ∧ clampQ q ≤ 127 := by
  unfold clampQ
  split
  · omega
  · split <;> omega

/-- In-range codes are unchanged. -/
theorem clampQ_id {q : Int} (h0 : -127 ≤ q) (h1 : q ≤ 127) : clampQ q = q := by
  unfold clampQ
  have ha : ¬ q < -127 := by omega
  have hb : ¬ q > 127 := by omega
  simp [ha, hb]

/-- One int8 scale per 64 values: a 256-wide head row has 4 scales. -/
theorem kv_scale_groups : 256 / 64 = 4 := by decide

-- § GDN causal conv

/-- The per-channel history step: `v0 = v1; v1 = v2; v2 = x`. -/
def convStep (h : Int × Int × Int) (x : Int) : Int × Int × Int := (h.2.1, h.2.2, x)

/-- After a chunk the history holds its last three inputs, whatever it held before. -/
theorem convStep_last_three (h : Int × Int × Int) (xs : List Int) (a b c : Int) :
    (xs ++ [a, b, c]).foldl convStep h = (a, b, c) := by
  simp [List.foldl_append, convStep]

/-- Splitting the prompt into chunks does not change the conv state. -/
theorem convStep_chunks (h : Int × Int × Int) (xs ys : List Int) :
    ys.foldl convStep (xs.foldl convStep h) = (xs ++ ys).foldl convStep h := by
  rw [List.foldl_append]

-- § router arg-max reduction

/-- The shuffle step: take the other lane's candidate if it is larger, or equal with a smaller id. -/
def pick (x y : Int × Nat) : Int × Nat := if y.1 > x.1 ∨ (y.1 = x.1 ∧ y.2 < x.2) then y else x

/-- The step is order-independent for distinct experts, so every lane agrees on the winner. -/
theorem pick_comm (a b : Int) (i j : Nat) (h : i ≠ j) : pick (a, i) (b, j) = pick (b, j) (a, i) := by
  unfold pick
  dsimp only
  by_cases h1 : b > a ∨ (b = a ∧ j < i)
  · by_cases h2 : a > b ∨ (a = b ∧ i < j)
    · exfalso; omega
    · simp only [h1, h2, if_true, if_false]
  · by_cases h2 : a > b ∨ (a = b ∧ i < j)
    · simp only [h1, h2, if_true, if_false]
    · exfalso; omega

/-- The winner's probability is at least both candidates'. -/
theorem pick_max (x y : Int × Nat) : x.1 ≤ (pick x y).1 ∧ y.1 ≤ (pick x y).1 := by
  unfold pick
  split
  · rename_i h; omega
  · rename_i h; omega

/-- The winner is one of the two candidates. -/
theorem pick_mem (x y : Int × Nat) : pick x y = x ∨ pick x y = y := by
  unfold pick
  split
  · exact Or.inr rfl
  · exact Or.inl rfl

-- § projection layouts

/-- Gate column of interleaved SwiGLU unit `k`. -/
def gateIdx (k : Nat) : Nat := 2 * k
/-- Up column of interleaved SwiGLU unit `k`. -/
def upIdx (k : Nat) : Nat := 2 * k + 1

/-- Gate and up columns of the 640 units tile the 1280-wide GEMM output without overlap. -/
theorem swiglu_layout {k k' : Nat} (hk : k < 640) :
    upIdx k < 1280 ∧ gateIdx k ≠ upIdx k' ∧ (gateIdx k = gateIdx k' → k = k') := by
  unfold gateIdx upIdx
  omega

/-- In `Qf`, head `h`'s query (`d < 256`) and gate (`256 + d`) never overlap and stay in the token's row. -/
theorem qf_split {t h h' d d' : Nat} (hh : h < 24) (hd : d < 256) (hd' : d' < 256) :
    t * 24 * 512 + h * 512 + d ≠ t * 24 * 512 + h' * 512 + 256 + d' ∧
      t * 24 * 512 + h * 512 + 256 + d < (t + 1) * 24 * 512 := by
  constructor <;> omega

/-- RoPE rotates pair `(p, p + 32)` of a 64-wide rotary span; the pairs partition the span. -/
theorem rope_pairs {p p' : Nat} (hp : p < 32) (hp' : p' < 32) :
    p + 32 < 64 ∧ p ≠ p' + 32 := by
  omega

end Strata.Prefill.Kernels
