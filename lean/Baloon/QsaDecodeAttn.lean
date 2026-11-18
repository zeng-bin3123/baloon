-- Baloon.QsaDecodeAttn: chunked split-K decode attention geometry for GQA (12 query heads per kv head, head_dim 256).

namespace Baloon.QsaDecodeAttn

-- § helpers

private theorem mulBound (a A B : Nat) (ha : a < A) : (a + 1) * B ≤ A * B :=
  Nat.mul_le_mul_right B (by omega)

theorem prodLt (a b A B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h := mulBound a A B ha; rw [Nat.add_mul, Nat.one_mul] at h; omega

-- § chunking (CHUNK = 64)

def chunkCount (cap : Nat) : Nat := (cap + 63) / 64

theorem cellInChunk (i cap : Nat) (h : i < cap) : i / 64 < chunkCount cap := by
  simp only [chunkCount]; omega

def cellsInChunk (n c : Int) : Int := min 64 (n - c * 64)

theorem surplusEmpty (n c : Int) (hc : n ≤ c * 64) : cellsInChunk n c ≤ 0 := by
  simp [cellsInChunk]; omega

theorem liveRange (n c : Int) (hc : c * 64 < n) : 0 < cellsInChunk n c ∧ cellsInChunk n c ≤ 64 := by
  simp [cellsInChunk]; omega

theorem cellCovered (i n : Nat) (h : i < n) :
    (i : Int) % 64 < cellsInChunk n ((i / 64 : Nat) : Int) ∧
    ((i / 64 : Nat) : Int) * 64 + (i : Int) % 64 = i := by
  simp [cellsInChunk]; omega

-- § GQA head mapping (G = 12 query heads per kv head)

theorem gqaKvHead (h nkv : Nat) (hh : h < 12 * nkv) : h / 12 < nkv ∧ (h / 12) * 12 + h % 12 = h := by
  constructor <;> omega

theorem divVsMod : (1 : Nat) / 12 ≠ 1 % 2 := by decide

theorem laneDims (d : Nat) (hd : d < 256) : d = (d / 8) * 8 + d % 8 ∧ d / 8 < 32 := by omega

-- § scratch layout (HD = 256 floats per head)

def scratchSize (cap nHead : Nat) : Nat := chunkCount cap * nHead * 258 + 64

theorem scratchFits (cap nHead : Nat) :
    chunkCount cap * nHead * 256 + chunkCount cap * nHead + chunkCount cap * nHead ≤ scratchSize cap nHead := by
  simp [scratchSize]; omega

theorem slotInBounds (kvh chunk nkv nch : Nat) (h1 : kvh < nkv) (h2 : chunk < nch) :
    kvh * nch + chunk < nkv * nch := prodLt kvh chunk nkv nch h1 h2

theorem accInBounds (slot g t S : Nat) (hs : slot < S) (hg : g < 12) (ht : t < 256) :
    (slot * 12 + g) * 256 + t < S * 12 * 256 :=
  prodLt _ _ _ _ (prodLt slot g S 12 hs hg) ht

-- § merge pass

def runMax : Int → List Int → Int
  | a, [] => a
  | a, x :: rest => runMax (max a x) rest

theorem runMax_ge_seed : ∀ (l : List Int) (a : Int), a ≤ runMax a l
  | [], _ => le_refl _
  | x :: rest, a => le_trans (by simp [max]; omega) (runMax_ge_seed rest (max a x))

theorem runMax_ge_elem : ∀ (l : List Int) (a x : Int), x ∈ l → x ≤ runMax a l
  | [], _, _, hm => by simp at hm
  | y :: rest, a, x, hm => by
    simp [List.mem_cons] at hm
    rcases hm with rfl | hrest
    · exact le_trans (by simp [max]; omega) (runMax_ge_seed rest (max a x))
    · exact runMax_ge_elem rest (max a y) x hrest

theorem mergeExpNonpos (l : List Int) (a m : Int) (hm : m ∈ l) : m - runMax a l ≤ 0 := by
  have h := runMax_ge_elem l a m hm; omega

-- § batch stride

theorem batchDisjoint (z1 z2 s : Nat) (hz : z1 < z2) : z1 * s + s ≤ z2 * s := by
  have h := mulBound z1 z2 s (by omega); rw [Nat.add_mul, Nat.one_mul] at h; exact h

theorem outputInRange (z h d nQ nHead : Nat) (hz : z < nQ) (hh : h < nHead) (hd : d < 256) :
    (z * nHead + h) * 256 + d < nQ * nHead * 256 :=
  prodLt _ _ _ _ (prodLt z h nQ nHead hz hh) hd

end Baloon.QsaDecodeAttn
