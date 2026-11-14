/-!
# MODULE: prefill_kernel_indexing
SOURCES: src/prefill/kernels.cu
Index arithmetic for the batched prompt kernels.
-/

namespace Ref.Prefill.Kernels

-- § hyper-connection layout R[T][HC=4][N=2560]

/-- Linear index: element d of stream c of token t. -/
def grIndex (t c d : Nat) : Nat := d + 2560 * c + 10240 * t

/-- Flat index decomposes uniquely into (t, c, d). -/
theorem grIndex_decomp (i : Nat) :
    grIndex (i / 10240) ((i % 10240) / 2560) (i % 2560) = i := by
  simp only [grIndex]; omega

/-- Stream < 4 and element < 2560 from any flat index. -/
theorem grIndex_ranges (i : Nat) : (i % 10240) / 2560 < 4 ∧ i % 2560 < 2560 :=
  ⟨by omega, by omega⟩

/-- Distinct in-range coordinates yield distinct cells. -/
theorem grIndex_injective {t t' c c' d d' : Nat} (hc : c < 4) (hc' : c' < 4)
    (hd : d < 2560) (hd' : d' < 2560) (h : grIndex t c d = grIndex t' c' d') :
    t = t' ∧ c = c' ∧ d = d' := by
  simp only [grIndex] at h; omega

/-- gr_mix stays within token t's row. -/
theorem grIndex_in_token {t c d : Nat} (hc : c < 4) (hd : d < 2560) :
    t * 10240 ≤ grIndex t c d ∧ grIndex t c d < (t + 1) * 10240 := by
  simp only [grIndex]; omega

-- § 2-bit expert blob (1,382,400 bytes total)

/-- Start of down codes region. -/
def oDCodes : Nat := 1280 * 640
/-- Start of gate/up scale region. -/
def oGuSc : Nat := oDCodes + 2560 * 160
/-- Start of down scale region. -/
def oDSc : Nat := oGuSc + 1280 * 40 * 2
/-- Total blob size in bytes. -/
def blobBytes : Nat := oDSc + 51200

/-- Blob size is exactly 1,382,400 bytes. -/
theorem blobBytes_value : blobBytes = 1382400 := rfl

/-- Regions are non-overlapping and in order. -/
theorem blob_regions_ordered : oDCodes < oGuSc ∧ oGuSc < oDSc ∧ oDSc < blobBytes :=
  ⟨by decide, by decide, by decide⟩

/-- Row widths in elements. -/
theorem blob_row_widths : 640 * 4 = 2560 ∧ 160 * 4 = 640 := ⟨rfl, rfl⟩

/-- Extract 2-bit code k from byte c using bit operations. -/
def code2 (c k : Nat) : Nat := (c >>> (2 * k)) % 4

/-- code2 is always in {0,1,2,3}. -/
theorem code2_lt (c k : Nat) : code2 c k < 4 := by
  simp only [code2]; exact Nat.mod_lt _ (by decide)

/-- Weight multiplier = code - 1 ∈ {-1, 0, 1, 2}. -/
def weightMult (b p : Nat) : Int := (code2 b p : Int) - 1

/-- Multiplier lies in {-1, 0, 1, 2}. -/
theorem weightMult_range (c k : Nat) : -1 ≤ weightMult c k ∧ weightMult c k ≤ 2 := by
  have hlt := code2_lt c k
  simp only [weightMult]; omega

/-- Linear output index: element-major within a row. -/
def guOut (row byte k : Nat) : Nat := k + byte * 4 + row * 2560

/-- Gate/up outputs are in range and injective. -/
theorem guOut_range_inj {row byte k row' byte' k' : Nat}
    (hr : row < 1280) (hb : byte < 640) (hk : k < 4)
    (hb' : byte' < 640) (hk' : k' < 4) :
    guOut row byte k < 1280 * 2560 ∧
    (guOut row byte k = guOut row' byte' k' → row = row' ∧ byte = byte' ∧ k = k') := by
  simp only [guOut]
  exact ⟨by omega, fun h => by omega⟩

/-- Down outputs are in range and injective. -/
theorem downOut_range_inj {row byte k row' byte' k' : Nat}
    (hr : row < 2560) (hb : byte < 160) (hk : k < 4)
    (hb' : byte' < 160) (hk' : k' < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 ∧
    (row * 640 + byte * 4 + k = row' * 640 + byte' * 4 + k' →
     row = row' ∧ byte = byte' ∧ k = k') :=
  ⟨by omega, fun h => by omega⟩

/-- Gate/up scale group: one scale per 64 weights. -/
theorem gu_scale_group {byte : Nat} (hb : byte < 640) : byte * 4 / 64 < 40 :=
  Nat.div_lt_of_lt_mul (by omega)

/-- Down scale group: one scale per 64 weights. -/
theorem down_scale_group {byte : Nat} (hb : byte < 160) : byte * 4 / 64 < 10 :=
  Nat.div_lt_of_lt_mul (by omega)

-- § paged KV append

/-- KV row: position-major within (page, kv-head) block. -/
def kvRow (page kvh pos ps : Nat) : Nat := pos % ps + ps * (page * 2 + kvh)

/-- A row a*ps + o determines a and o uniquely. -/
theorem row_split_unique {ps a a' o o' : Nat} (ho : o < ps) (ho' : o' < ps)
    (h : a * ps + o = a' * ps + o') : a = a' ∧ o = o' := by
  have hp : 0 < ps := by omega
  have hdiv : ∀ x y, y < ps → (x * ps + y) / ps = x := fun x y hy => by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hy, Nat.zero_add]
  have hmod : ∀ x y, y < ps → (x * ps + y) % ps = y := fun x y hy => by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hy]
  have h1 := hdiv a o ho; have h2 := hdiv a' o' ho'
  have h3 := hmod a o ho; have h4 := hmod a' o' ho'
  rw [h] at h1 h3; exact ⟨h1.symm.trans h2, h3.symm.trans h4⟩

/-- Distinct (page, kvh, pos%ps) triples map to distinct KV rows. -/
theorem kvRow_injective {ps page page' kvh kvh' pos pos' : Nat}
    (hps : 0 < ps) (hk : kvh < 2) (hk' : kvh' < 2)
    (h : kvRow page kvh pos ps = kvRow page' kvh' pos' ps) :
    page = page' ∧ kvh = kvh' ∧ pos % ps = pos' % ps := by
  simp only [kvRow] at h
  obtain ⟨ha, ho⟩ := row_split_unique (Nat.mod_lt pos hps) (Nat.mod_lt pos' hps)
    (by linarith [h])
  omega

/-- Clamp to symmetric int8 range [-127, 127]. -/
def clampQ (q : Int) : Int :=
  q.max (-127) |>.min 127

/-- Clamped codes are symmetric int8. -/
theorem clampQ_range (q : Int) : -127 ≤ clampQ q ∧ clampQ q ≤ 127 := by
  simp only [clampQ, Int.max_def, Int.min_def]
  split <;> split <;> omega

/-- In-range values are unchanged by clamp. -/
theorem clampQ_id {q : Int} (h0 : -127 ≤ q) (h1 : q ≤ 127) : clampQ q = q := by
  simp only [clampQ, Int.max_def, Int.min_def]
  split <;> split <;> omega

/-- 256-wide head row has 4 scale groups. -/
theorem kv_scale_groups : 256 / 64 = 4 := rfl

-- § GDN causal conv (each channel keeps last 3 inputs)

/-- Shift the three-element conv history by one input. -/
def convStep (s : Int × Int × Int) (x : Int) : Int × Int × Int := (s.2.1, s.2.2, x)

/-- After any list ending in a,b,c the history is (a,b,c). -/
theorem convStep_last_three (s : Int × Int × Int) (xs : List Int) (a b c : Int) :
    (xs ++ [a, b, c]).foldl convStep s = (a, b, c) := by
  simp [List.foldl_append, convStep]

/-- Conv state is chunk-invariant. -/
theorem convStep_chunks (s : Int × Int × Int) (xs ys : List Int) :
    ys.foldl convStep (xs.foldl convStep s) = (xs ++ ys).foldl convStep s :=
  (List.foldl_append convStep s xs ys).symm

-- § router arg-max (larger value wins; ties to smaller expert id)

/-- Take the larger-valued candidate, breaking ties by smaller id. -/
def pick (p q : Int × Nat) : Int × Nat :=
  if q.1 > p.1 ∨ (q.1 = p.1 ∧ q.2 < p.2) then q else p

/-- pick is commutative for distinct experts. -/
theorem pick_comm (a b : Int) (i j : Nat) (h : i ≠ j) :
    pick (a, i) (b, j) = pick (b, j) (a, i) := by
  simp only [pick]
  by_cases h1 : b > a ∨ (b = a ∧ j < i)
  · by_cases h2 : a > b ∨ (a = b ∧ i < j)
    · exact absurd h (by omega)
    · simp [h1, h2]
  · by_cases h2 : a > b ∨ (a = b ∧ i < j)
    · simp [h1, h2]
    · exact absurd h (by omega)

/-- Winner probability is ≥ both candidates'. -/
theorem pick_max (x y : Int × Nat) : x.1 ≤ (pick x y).1 ∧ y.1 ≤ (pick x y).1 := by
  simp only [pick]; split <;> [simp; rename_i h; omega]

/-- Winner is one of the two candidates. -/
theorem pick_mem (x y : Int × Nat) : pick x y = x ∨ pick x y = y := by
  simp only [pick]; split <;> [right; left] <;> rfl

-- § projection layouts

/-- Gate column of SwiGLU unit k. -/
def gateIdx (k : Nat) : Nat := 2 * k
/-- Up column of SwiGLU unit k. -/
def upIdx (k : Nat) : Nat := 2 * k + 1

/-- Gate and up columns partition the 1280-wide output. -/
theorem swiglu_layout {k k' : Nat} (hk : k < 640) :
    upIdx k < 1280 ∧ gateIdx k ≠ upIdx k' ∧ (gateIdx k = gateIdx k' → k = k') := by
  simp only [gateIdx, upIdx]; omega

/-- Query and gate slots in Qf never overlap. -/
theorem qf_split {t h h' d d' : Nat} (hh : h < 24) (hd : d < 256) (hd' : d' < 256) :
    t * 24 * 512 + h * 512 + d ≠ t * 24 * 512 + h' * 512 + 256 + d' ∧
    t * 24 * 512 + h * 512 + 256 + d < (t + 1) * 24 * 512 :=
  ⟨by omega, by omega⟩

/-- RoPE rotates pair (p, p+32) within a 64-wide span. -/
theorem rope_pairs {p p' : Nat} (hp : p < 32) (hp' : p' < 32) :
    p + 32 < 64 ∧ p ≠ p' + 32 :=
  ⟨by omega, by omega⟩

end Ref.Prefill.Kernels
