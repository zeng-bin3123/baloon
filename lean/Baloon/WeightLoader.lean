-- Baloon.WeightLoader: weight-loader segment lists, fp16/plain scale conversions,
-- widen/narrow tiling, chunk-loop progress, arena placement and kind mapping.
-- Independent implementation proving the same mathematical facts as Ref.WeightLoader.

namespace Baloon.WeightLoader

def CHUNK : Nat := 8388608

-- § segment lists

def srcScaleBytes (scalesFp16 : Bool) (scales : Nat) : Nat :=
  if scalesFp16 then scales / 2 else scales

def sumSrc (codes scales offset : Nat) (fp16 : Bool) : Nat :=
  codes + srcScaleBytes fp16 scales + offset

def sumDst (codes scales offset : Nat) : Nat := codes + scales + offset

theorem sumDst_eq (c s o : Nat) : sumDst c s o = c + s + o := rfl
theorem sumSrc_plain (c s o : Nat) : sumSrc c s o false = sumDst c s o := by
  simp [sumSrc, sumDst, srcScaleBytes]
theorem sumSrc_fp16 (c s o : Nat) (he : s % 2 = 0) : sumSrc c s o true + s / 2 = sumDst c s o := by
  simp only [sumSrc, sumDst, srcScaleBytes, if_true]; omega
theorem fp16_half_double (s : Nat) (he : s % 2 = 0) : s / 2 * 2 = s := by omega

-- § conversions: widen, bf16, f16

theorem widen_out_bytes (n : Nat) (he : n % 2 = 0) : 4 * (n / 2) = 2 * n := by omega
theorem widen_tiles (done n : Nat) (he : n % 2 = 0) : done * 2 + 4 * (n / 2) = (done + n) * 2 := by omega
theorem widen_fits_stage (n : Nat) (h : n ≤ CHUNK) : 4 * (n / 2) ≤ 2 * CHUNK := by
  unfold CHUNK at *; omega
theorem narrow_tiles (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + 2 * (n / 4) = (done + n) / 2 := by omega
theorem promoted_saves (elems src dst : Nat) (h1 : src = elems * 4) (h2 : dst = elems * 2) :
    src - dst = elems * 2 := by omega

-- § chunk loop

def chunkLen (rem : Nat) : Nat := min CHUNK rem

theorem chunk_progress (rem : Nat) (h : 0 < rem) : 0 < chunkLen rem ∧ rem - chunkLen rem < rem := by
  unfold chunkLen CHUNK; omega
theorem chunk_le (rem : Nat) : chunkLen rem ≤ CHUNK := by unfold chunkLen; omega
theorem chunk_even_of_full (rem : Nat) (h : CHUNK ≤ rem) : chunkLen rem % 2 = 0 := by
  unfold chunkLen CHUNK at *; omega
theorem chunk_mult4 (rem : Nat) (h : rem % 4 = 0) : chunkLen rem % 4 = 0 := by
  unfold chunkLen CHUNK; omega

-- § arena placement and compaction

def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

def placeAll : Nat → List Nat → Nat
  | at_, [] => at_
  | at_, x :: xs => placeAll (at_ + alignUp256 x) xs

theorem placeAll_mod : ∀ (l : List Nat) (a : Nat), a % 256 = 0 → placeAll a l % 256 = 0
  | [], a, h => by simpa [placeAll] using h
  | x :: xs, a, h => by
    have h1 := alignUp256_mod x
    have := placeAll_mod xs (a + alignUp256 x) (by omega)
    simpa [placeAll] using this

theorem placeAll_ge : ∀ (l : List Nat) (a : Nat), a + l.sum ≤ placeAll a l
  | [], a => by simp [placeAll]
  | x :: xs, a => by
    have h1 := alignUp256_ge x
    have h2 := placeAll_ge xs (a + alignUp256 x)
    simp only [placeAll, List.sum_cons]; omega

theorem row_end_le_next (at_ x : Nat) : at_ + x ≤ at_ + alignUp256 x := by
  have := alignUp256_ge x; omega

-- § index kinds

def mapKind (k : Nat) : Nat := if k = 4 then 1 else if k = 5 then 3 else k

theorem mapKind_bf16 : mapKind 4 = 1 := by decide
theorem mapKind_f16 : mapKind 5 = 3 := by decide
theorem mapKind_other (k : Nat) (h4 : k ≠ 4) (h5 : k ≠ 5) : mapKind k = k := by
  simp [mapKind, h4, h5]

-- § conversion selection

inductive Conv | copy | bf16High | toF16 deriving Repr

def convFor (raw16 : Bool) (kind : Nat) : Conv :=
  if raw16 then Conv.copy else if kind = 1 then Conv.bf16High else if kind = 3 then Conv.toF16 else Conv.copy

theorem convFor_raw (k : Nat) : convFor true k = Conv.copy := by simp [convFor]
theorem convFor_bf16 : convFor false 1 = Conv.bf16High := by simp [convFor]
theorem convFor_f16 : convFor false 3 = Conv.toF16 := by simp [convFor]

-- § arena sufficiency

def arenaOk (arena pool : Nat) : Bool := decide (pool ≤ arena)

theorem arenaOk_sufficient (arena pool : Nat) (h : arenaOk arena pool = true) : pool ≤ arena := by
  unfold arenaOk at h; exact of_decide_eq_true h

theorem arenaOk_reject (arena pool : Nat) (h : arena < pool) : arenaOk arena pool = false := by
  unfold arenaOk; exact decide_eq_false (by omega)

end Baloon.WeightLoader
