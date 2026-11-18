-- Baloon.GdnRecurrence: gated delta-net decode step — state layout bijection, row groups, conv history, alpha/beta rows.

namespace Baloon.GdnRecurrence

-- § index helpers

theorem flatLt (a b A B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h; omega

/-- Mixed-radix: (a*n+b)/n = a and (a*n+b)%n = b when b < n. -/
theorem mixedRadix (a b n : Nat) (hb : b < n) : (a * n + b) / n = a ∧ (a * n + b) % n = b := by
  have hn : 0 < n := Nat.lt_of_le_of_lt (Nat.zero_le _) hb
  have key : a * n + b = b + a * n := Nat.add_comm _ _
  have step := Nat.add_mul_div_right b a hn
  rw [Nat.div_eq_of_lt hb] at step
  exact ⟨by rw [key]; linarith, by rw [key, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]⟩

-- § state layout (S = 128)

def stateIdx (hv S i h j : Nat) : Nat := (i * hv + h) * S + j

theorem stateInBuf (hv S i h j : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) :
    stateIdx hv S i h j < S * hv * S :=
  flatLt _ j (S * hv) S (flatLt i h S hv hi hh) hj

theorem stateInjective (hv S i1 i2 h1 h2 j1 j2 : Nat)
    (hh1 : h1 < hv) (hh2 : h2 < hv) (hj1 : j1 < S) (hj2 : j2 < S)
    (heq : stateIdx hv S i1 h1 j1 = stateIdx hv S i2 h2 j2) : i1 = i2 ∧ h1 = h2 ∧ j1 = j2 := by
  unfold stateIdx at heq
  obtain ⟨dv1, md1⟩ := mixedRadix (i1 * hv + h1) j1 S hj1
  obtain ⟨dv2, md2⟩ := mixedRadix (i2 * hv + h2) j2 S hj2
  have ej : j1 = j2 := by
    have heq2 : ((i1*hv+h1)*S+j1) % S = ((i2*hv+h2)*S+j2) % S := by rw [heq]
    rw [md1, md2] at heq2; exact heq2
  have eih : i1 * hv + h1 = i2 * hv + h2 := by
    have heq2 : ((i1*hv+h1)*S+j1) / S = ((i2*hv+h2)*S+j2) / S := by rw [heq]
    rw [dv1, dv2] at heq2; exact heq2
  obtain ⟨di1, mh1⟩ := mixedRadix i1 h1 hv hh1
  obtain ⟨di2, mh2⟩ := mixedRadix i2 h2 hv hh2
  have eh : h1 = h2 := by
    have : (i1*hv+h1) % hv = (i2*hv+h2) % hv := by rw [eih]
    rw [mh1, mh2] at this; exact this
  have ei : i1 = i2 := by
    have : (i1*hv+h1) / hv = (i2*hv+h2) / hv := by rw [eih]
    rw [di1, di2] at this; exact this
  exact ⟨ei, eh, ej⟩

theorem fuseStride (A r hv head S col : Nat) :
    ((A * hv + head) * S + col) + r * (hv * S) = (((A + r) * hv + head) * S + col) := by ring

theorem rowGroupTile (i : Nat) (hi : i < 128) : i = (i/32)*32 + i%32 ∧ i/32 < 4 ∧ i%32 < 32 := by omega

theorem colTile (j : Nat) (hj : j < 128) : j/32 < 4 ∧ j = (j/32)*32 + j%32 := by omega

theorem threadCount : 128 * 48 = (6144 : Nat) := by decide

theorem headPairValid (h hk : Nat) (hk_pos : 0 < hk) : h % hk < hk := Nat.mod_lt _ hk_pos

-- § convolution history

def histSeq {T : Type} (hist x : Nat → T) (pos : Nat) : T :=
  if pos < 3 then hist pos else x (pos - 3)

theorem firstWindowIsHist {T : Type} (hist x : Nat → T) :
    histSeq hist x 0 = hist 0 ∧ histSeq hist x 1 = hist 1 ∧
    histSeq hist x 2 = hist 2 ∧ histSeq hist x 3 = x 0 := by simp [histSeq]

theorem afterOneCommit {T : Type} (hist x : Nat → T) :
    histSeq hist x (1+0) = hist 1 ∧ histSeq hist x (1+1) = hist 2 ∧
    histSeq hist x (1+2) = x 0 := by simp [histSeq]

theorem windowOverlap {T : Type} (hist x : Nat → T) (t j : Nat) :
    histSeq hist x (t + (j + 1)) = histSeq hist x ((t + 1) + j) := by congr 1; omega

theorem deepHistIsInput {T : Type} (hist x : Nat → T) (n j : Nat) (hn : 3 ≤ n) :
    histSeq hist x (n + j) = x (n + j - 3) := by
  simp only [histSeq]
  have h : ¬(n + j < 3) := by omega
  simp [h]

-- § alpha/beta

theorem bf16PairSplit (w : Nat) (hw : w < 4294967296) :
    w = (w/65536)*65536 + w%65536 ∧ (w%65536)*65536 < 4294967296 := by omega

theorem abRowClass (row hv : Nat) (h : row < 2 * hv) :
    (row < hv ∧ row < hv) ∨ (hv ≤ row ∧ row - hv < hv) := by omega

theorem abGridCovers (row hv : Nat) (h : row < 2 * hv) :
    row/8 < (2*hv+7)/8 ∧ row = (row/8)*8 + row%8 := by omega

end Baloon.GdnRecurrence
