-- Baloon.VerifyWindow: token groups, ring values, plan layout, flag monotonicity, and kernel-side helpers.

namespace Baloon.VerifyWindow

-- § token groups

def grpBegin (T g : Nat) : Nat := if g = 0 then 0 else (T + 1) / 2
def grpEnd (T G g : Nat) : Nat := if g = 0 then (if G = 2 then (T+1)/2 else T) else T

theorem twoGroupsTile (T : Nat) :
    grpBegin T 0 = 0 ∧ grpEnd T 2 0 = grpBegin T 1 ∧ grpEnd T 2 1 = T := by
  simp [grpBegin, grpEnd]

theorem groupSizeSum (T : Nat) :
    (grpEnd T 2 0 - grpBegin T 0) + (grpEnd T 2 1 - grpBegin T 1) = T := by
  simp [grpBegin, grpEnd]; omega

theorem bothGroupsNonEmpty (T : Nat) (hT : 2 ≤ T) :
    grpBegin T 0 < grpEnd T 2 0 ∧ grpBegin T 1 < grpEnd T 2 1 := by
  simp [grpBegin, grpEnd]; omega

theorem oneGroupAll (T : Nat) : grpBegin T 0 = 0 ∧ grpEnd T 1 0 = T := by
  simp [grpBegin, grpEnd]

-- § ring values

def ringVal (l G g : Nat) : Nat := l * G + g + 1

theorem ringHostMatch (l G g : Nat) : l * G + g + 1 = ringVal l G g := rfl

theorem recoverLayer (l g : Nat) (hg : g < 2) : (l * 2 + g) / 2 = l := by omega

theorem recoverGroup (l g : Nat) (hg : g < 2) : (l * 2 + g) % 2 = g := by omega

theorem ringIncreasing (l g l' g' : Nat) (hg : g < 2) (hg' : g' < 2) (hl : l < l') :
    ringVal l 2 g < ringVal l' 2 g' := by simp [ringVal]; omega

-- § stream ordering indices

def streamPre (l g : Nat) : Nat := if l = 0 then g else 4 * l - 1 + 2 * g
def streamPost (l g : Nat) : Nat := 2 + 4 * l + 2 * g

theorem preBeforePost (l g : Nat) (hg : g < 2) : streamPre l g < streamPost l g := by
  unfold streamPre streamPost; split_ifs <;> omega

theorem postBeforeNextPre (l g : Nat) (hg : g < 2) : streamPost l g < streamPre (l + 1) g := by
  simp only [streamPost, streamPre, if_neg (Nat.succ_ne_zero l)]
  omega

theorem preAbeforeB (l : Nat) : streamPre l 0 < streamPre l 1 := by
  unfold streamPre; split_ifs <;> omega

theorem nextPreAbeforePostB (l : Nat) : streamPre (l + 1) 0 < streamPost l 1 := by
  unfold streamPre streamPost; split_ifs <;> omega

theorem postOrder (l : Nat) : streamPost l 0 < streamPost l 1 := by
  unfold streamPost; omega

-- § plan block layout

def planWords (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap
def planPtrOff (cap : Nat) : Nat := (planWords cap + 1) / 2 * 2
def planTotal (cap : Nat) : Nat := planPtrOff cap + 5 * cap + 2

theorem planAligned (cap : Nat) : planPtrOff cap % 2 = 0 := by
  simp only [planPtrOff]; omega

theorem planPad (cap : Nat) : planWords cap ≤ planPtrOff cap ∧ planPtrOff cap ≤ planWords cap + 1 := by
  simp [planPtrOff]; omega

theorem start2Fits (cap : Nat) : planPtrOff cap + 4 * cap + (cap + 1) ≤ planTotal cap := by
  simp [planTotal]; omega

-- § flag monotonicity

def raiseMax (cur v : Nat) : Nat := max cur v
theorem raiseNonDecr (cur v : Nat) : cur ≤ raiseMax cur v := le_max_left cur v
theorem raiseGeVal (cur v : Nat) : v ≤ raiseMax cur v := le_max_right cur v
theorem lateLow (cur v : Nat) (h : v ≤ cur) : raiseMax cur v = cur := max_eq_left h

-- § kernel-side helpers

def winSeq {A : Type} (hist x : Nat → A) (k : Nat) : A :=
  if k < 3 then hist k else x (k - 3)

theorem winSeqAt {A : Type} (hist x : Nat → A) (k : Nat) :
    winSeq hist x k = if k < 3 then hist k else x (k - 3) := rfl

theorem noopCommit {A : Type} (hist x : Nat → A) (j : Nat) :
    winSeq hist x (0 + j) = winSeq hist x j := by simp

theorem oneCommit {A : Type} (hist x : Nat → A) :
    winSeq hist x (1+0) = hist 1 ∧ winSeq hist x (1+1) = hist 2 ∧ winSeq hist x (1+2) = x 0 := by
  simp [winSeq]

theorem deepInput {A : Type} (hist x : Nat → A) (n j : Nat) (hn : 3 ≤ n) :
    winSeq hist x (n + j) = x (n + j - 3) := by
  simp only [winSeq]
  have h : ¬(n + j < 3) := by omega
  simp [h]

theorem slideWindow {A : Type} (hist x : Nat → A) (t j : Nat) :
    winSeq hist x ((t + 1) + j) = winSeq hist x (t + (j + 1)) := by
  congr 1; omega

def denseRec (c : Nat) : Nat × Nat × Nat × Nat := (c, c + 1, (c + 1) / 4, c + 1)

theorem denseMatchesFill (c : Nat) (h : c + 1 ≤ 2051) :
    denseRec c = (c, c + 1, (c + 1) / 4, min (c + 1) 2051) := by
  unfold denseRec
  rw [Nat.min_eq_left (by omega)]

def windowStart (nkv w : Nat) : Nat := if nkv > w then nkv - w else 0

theorem windowWidth (nkv w : Nat) : nkv - windowStart nkv w = min nkv w := by
  unfold windowStart; split_ifs with h <;> omega

theorem windowValid (nkv w j : Nat) (hj : j < nkv - windowStart nkv w) :
    windowStart nkv w + j < nkv := by
  unfold windowStart at *; split_ifs at * with h <;> omega

theorem windowFull (nkv w : Nat) (h : nkv ≤ w) : windowStart nkv w = 0 := by
  unfold windowStart; split_ifs <;> omega

private theorem auxLt (a b A B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h; omega

theorem gatherDecode (i n per : Nat) (hp : 0 < per) (h : i < n * per) :
    i / per < n ∧ i % per < per ∧ (i / per) * per + i % per = i := by
  refine ⟨(Nat.div_lt_iff_lt_mul hp).mpr h, Nat.mod_lt _ hp, ?_⟩
  have hdm := Nat.div_add_mod i per
  linarith [Nat.mul_comm per (i / per)]

theorem gatherSrcLt (id o rows row16 : Nat) (h1 : id < rows) (h2 : o < row16) :
    id * row16 + o < rows * row16 := auxLt id o rows row16 h1 h2

theorem rebaseStep (base k bytes : Nat) :
    base + (k + 1) * bytes = base + k * bytes + bytes := by ring

theorem broadcastSrcLt (i n hc : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

theorem rowsDisjoint (t1 t2 s : Nat) (ht : t1 < t2) : t1 * s + s ≤ t2 * s := by
  have h : (t1 + 1) * s ≤ t2 * s := Nat.mul_le_mul_right s (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h; exact h

def doCommit (hasKeep : Bool) (n : Nat) : Bool := hasKeep && decide (0 < n)

theorem noCommit (n : Nat) : doCommit false n = false := by simp [doCommit]

theorem outputsLe (n tOut : Nat) : n - tOut ≤ n := Nat.sub_le n tOut

theorem replayedSkip (t tOut : Nat) : (t < tOut) ↔ ¬(tOut ≤ t) := by omega

theorem mtp_row_copy_lt (row i stride nRows : Nat) (hr : row < nRows) (hi : i < stride) :
    row * stride + i < nRows * stride := auxLt row i nRows stride hr hi

end Baloon.VerifyWindow
