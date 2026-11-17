-- Baloon.ExpertSource: independent implementation; proves the same invariants
-- as Ref.ExpertSource with different definitions and proof strategies.

namespace Baloon.ExpertSource

-- § blob addressing

/-- expert-major flat index: row = layer, col = expert, stride = nExpert. -/
def flatIdx (layer expert nExpert : Nat) : Nat := expert + layer * nExpert

theorem unchecked_flat_alias (E : Nat) : flatIdx 0 E E = flatIdx 1 0 E := by
  simp [flatIdx]

/-- Without the range check on expert, blob(0,E) aliases blob(1,0). -/
theorem alias_512 : flatIdx 0 512 512 = flatIdx 1 0 512 := by unfold flatIdx; ring

theorem flatIdx_recover (l e E : Nat) (hE : 0 < E) (he : e < E) :
    flatIdx l e E / E = l ∧ flatIdx l e E % E = e := by
  simp only [flatIdx]
  have heq : e + l * E = e + E * l := by ring
  exact ⟨by rw [heq, Nat.add_mul_div_left _ _ hE, Nat.div_eq_of_lt he, Nat.zero_add],
         by rw [heq, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he]⟩

theorem flatIdx_inj (l e l' e' E : Nat) (hE : 0 < E) (he : e < E) (he' : e' < E)
    (h : flatIdx l e E = flatIdx l' e' E) : l = l' ∧ e = e' := by
  have h1 := flatIdx_recover l e E hE he
  have h2 := flatIdx_recover l' e' E hE he'
  rw [h] at h1
  exact ⟨(h1.1.symm.trans h2.1), (h1.2.symm.trans h2.2)⟩

theorem flatIdx_lt (l e L E : Nat) (hl : l < L) (he : e < E) : flatIdx l e E < L * E := by
  simp only [flatIdx]
  have hLE : (l + 1) * E ≤ L * E := Nat.mul_le_mul_right E hl
  have hlin : (l + 1) * E = l * E + E := by ring
  omega

/-- 48 layers × 512 experts × 1,382,400 bytes/expert. -/
theorem experts_file_size : 48 * 512 * 1382400 = 33973862400 := by norm_num

/-- Total number of expert blobs in the artifact. -/
theorem blob_count : 48 * 512 = 24576 := by norm_num

-- § hit/miss job building

def nHit : List Bool → Nat
  | [] => 0
  | true  :: bs => 1 + nHit bs
  | false :: bs => nHit bs

def nMiss : List Bool → Nat
  | [] => 0
  | true  :: bs => nMiss bs
  | false :: bs => 1 + nMiss bs

theorem jobs_plus_hits : ∀ l : List Bool, nHit l + nMiss l = l.length
  | [] => rfl
  | true  :: bs => by simp only [nHit, nMiss, List.length_cons]; have := jobs_plus_hits bs; omega
  | false :: bs => by simp only [nHit, nMiss, List.length_cons]; have := jobs_plus_hits bs; omega

theorem nHit_le (l : List Bool) : nHit l ≤ l.length := by
  have := jobs_plus_hits l; omega

theorem no_hits_all_jobs : ∀ l : List Bool, nHit l = 0 → nMiss l = l.length := fun l h => by
  have := jobs_plus_hits l; omega

-- § PCIe share: m = (nmiss * pcie_num) >> 8

/-- PCIe share using Nat.div with explicit application. -/
def pcieShare (nmiss num : Nat) : Nat := Nat.div (nmiss * num) 256

theorem pcieShare_le (nmiss num : Nat) (h : num ≤ 256) : pcieShare nmiss num ≤ nmiss := by
  simp only [pcieShare]
  apply Nat.div_le_of_le_mul
  calc nmiss * num ≤ nmiss * 256 := Nat.mul_le_mul_left nmiss h
    _ = 256 * nmiss := Nat.mul_comm nmiss 256

/-- Ranks nmiss-m .. nmiss-1 number exactly m (subtraction cancel identity). -/
theorem pcie_ranks_count (nmiss m : Nat) (h : m ≤ nmiss) : nmiss - (nmiss - m) = m := by
  have := Nat.sub_add_cancel h; omega

theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  unfold pcieShare
  simp only [Nat.mul_zero]

-- § GGUF expert loading: blob layout [gate | up | down]

/-- (start, size) regions for one expert blob. -/
def blobRegions (upOff downOff blob : Nat) : List (Nat × Nat) :=
  [⟨0, upOff⟩, ⟨upOff, upOff⟩, ⟨downOff, blob - downOff⟩]

theorem blob_regions_tile (upOff blob : Nat) (h : 2 * upOff ≤ blob) :
    0 + upOff = upOff ∧ upOff + upOff = 2 * upOff ∧ 2 * upOff + (blob - 2 * upOff) = blob :=
  ⟨by omega, by omega, by omega⟩

/-- 16 experts per chunk never cover more than 16. -/
theorem chunk_experts_le (n per : Nat) (hn : n ≤ per * 16) : n / per ≤ 16 := by
  exact Nat.div_le_of_le_mul hn

-- § arena slack: arena = file + max_blob so any slot copy fits

/-- Arena size: one max-blob of slack beyond the file size. -/
def arenaSlack (want maxBlob : Nat) : Nat := maxBlob + want

theorem slot_copy_in_arena (off bb want mb : Nat) (h1 : off + bb ≤ want) :
    off + mb ≤ arenaSlack want mb := by
  unfold arenaSlack; omega

theorem reads_two (r : Nat) : r + 1 + 1 = r + 2 := by omega

end Baloon.ExpertSource
