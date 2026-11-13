-- ChunkedAttnMerge.lean
-- Formal model of src/kernels/cuda/qsa_decode_attn.cu chunked attention with log-sum-exp merge.
namespace Baloon.DecodeAttn

def nChunks (cap : Nat) : Nat := (cap + 63) / 64

def nHere (nIds ch : Nat) : Nat := min 64 (nIds - ch * 64)

theorem chunk_of_cell_lt (cap nIds c : Nat) (hc : c < nIds) (hn : nIds ≤ cap) : c / 64 < nChunks cap := by
  unfold nChunks; omega

theorem cell_live_in_chunk (nIds c : Nat) (hc : c < nIds) : c % 64 < nHere nIds (c / 64) := by
  unfold nHere; omega

theorem chunk_empty_iff (nIds ch : Nat) : nHere nIds ch = 0 ↔ nIds ≤ ch * 64 := by
  unfold nHere; omega

theorem nHere_le (nIds ch : Nat) : nHere nIds ch ≤ 64 := Nat.min_le_left _ _

theorem gqa_decode (h nKv : Nat) (hh : h < 12 * nKv) : h / 12 < nKv ∧ h = h / 12 * 12 + h % 12 := by
  omega

theorem gqa_general (h g nKv : Nat) (hg : 0 < g) (hh : h < g * nKv) : h / g < nKv := by
  rw [Nat.mul_comm] at hh; exact (Nat.div_lt_iff_lt_mul hg).mpr hh

theorem gqa_mod_differs : (1 / 12 : Nat) ≠ 1 % 2 := by decide

def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * (256 + 2) + 64

theorem scratch_fits (cap nHead : Nat) :
    nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead < scratchFloats cap nHead := by
  unfold scratchFloats; rw [Nat.mul_add]; omega

theorem partSlot_lt (kvh nKv nC c hl : Nat) (hk : kvh < nKv) (hc : c < nC) (hl12 : hl < 12) :
    (kvh * nC + c) * 12 + hl < nKv * nC * 12 := by
  have h1 : (kvh + 1) * nC ≤ nKv * nC := Nat.mul_le_mul_right nC (Nat.succ_le_of_lt hk)
  rw [Nat.succ_mul] at h1
  have h2 : kvh * nC + c < nKv * nC := by omega
  have h3 : (kvh * nC + c + 1) * 12 ≤ nKv * nC * 12 := Nat.mul_le_mul_right 12 (Nat.succ_le_of_lt h2)
  rw [Nat.succ_mul] at h3; omega

def mergeMax (lo : Int) : List Int → Int
  | [] => lo
  | m :: ms => max m (mergeMax lo ms)

theorem mergeMax_ge (lo : Int) (ms : List Int) (m : Int) (h : m ∈ ms) : m - mergeMax lo ms ≤ 0 := by
  induction ms with
  | nil => simp at h
  | cons x xs ih =>
    simp only [mergeMax]
    rcases List.mem_cons.mp h with h | h
    · subst h; omega
    · have := ih h; omega

theorem mergeMax_all_sentinel (lo : Int) (ms : List Int) (h : ∀ m ∈ ms, m = lo) : mergeMax lo ms = lo := by
  induction ms with
  | nil => rfl
  | cons x xs ih =>
    simp only [mergeMax]
    have hx := h x (List.mem_cons_self ..)
    have hr := ih (fun m hm => h m (List.mem_cons_of_mem _ hm))
    omega

end Baloon.DecodeAttn
