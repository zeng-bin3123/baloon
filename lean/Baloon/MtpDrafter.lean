-- Baloon.MtpDrafter: multi-token prediction drafter window/capacity, step-row
-- staging, draft-chain termination, chunked prefill, and tensor lookup.
-- Independent implementation proving the same mathematical facts as Ref.MtpDrafter.

namespace Baloon.MtpDrafter

-- § window and capacity

def effWindow (window maxCells : Nat) : Nat :=
  if 0 < window ∧ window < maxCells then window else 0

def attendLen (window maxCells : Nat) : Nat :=
  if effWindow window maxCells > 0 then effWindow window maxCells else maxCells

def capOf (window maxCells : Nat) : Nat := (attendLen window maxCells + 63) / 64 * 64

theorem effWindow_le (w m : Nat) : effWindow w m ≤ m := by
  unfold effWindow; split <;> omega
theorem attendLen_le (w m : Nat) : attendLen w m ≤ m := by
  unfold attendLen; have := effWindow_le w m; split <;> omega
theorem cap_covers (w m : Nat) : attendLen w m ≤ capOf w m := by unfold capOf; omega
theorem cap_mod (w m : Nat) : capOf w m % 64 = 0 := by unfold capOf; omega
theorem cap_lt (w m : Nat) : capOf w m < attendLen w m + 64 := by unfold capOf; omega

-- § step-row staging

def chainRow (T j : Nat) : Nat := T + j - 1
def fullRow (T : Nat) : Nat := 2 * T - 1

theorem chainRow_above_catchup (T j : Nat) (hj : 1 ≤ j) : T ≤ chainRow T j := by
  unfold chainRow; omega
theorem chainRow_below_full (T j : Nat) (hj : j ≤ T - 2) (hT : 3 ≤ T) :
    chainRow T j < fullRow T ∧ fullRow T < 2 * T := by
  unfold chainRow fullRow; omega
theorem chainRow_inj (T j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k) (h : chainRow T j = chainRow T k) :
    j = k := by unfold chainRow at h; omega
theorem step_block_mono (a b : Nat) (h : a ≤ b) : (a + 1) / 4 ≤ (b + 1) / 4 := by omega

-- § the draft chain

def chainFuel (maxT minP : Nat) (p : Nat → Nat) : Nat → Nat → Nat
  | 0, j => j
  | fuel + 1, j =>
    if j < maxT - 1 ∧ p (j - 1) ≥ minP then chainFuel maxT minP p fuel (j + 1) else j

theorem chain_ge (maxT minP : Nat) (p : Nat → Nat) :
    ∀ fuel j, j ≤ chainFuel maxT minP p fuel j
  | 0, j => by simp [chainFuel]
  | fuel + 1, j => by
    unfold chainFuel; split
    · have := chain_ge maxT minP p fuel (j + 1); omega
    · omega

theorem chain_le (maxT minP : Nat) (p : Nat → Nat) :
    ∀ fuel j, j ≤ maxT - 1 → chainFuel maxT minP p fuel j ≤ maxT - 1
  | 0, j, h => by simpa [chainFuel] using h
  | fuel + 1, j, h => by
    unfold chainFuel; split
    · rename_i hc; exact chain_le maxT minP p fuel (j + 1) (by omega)
    · exact h

/-- The chain never exceeds max_t (bounded by fuel). -/
theorem chainFuel_bounded (maxT minP : Nat) (p : Nat → Nat) (fuel j : Nat) (h : j < maxT) :
    chainFuel maxT minP p fuel j < maxT := by
  induction fuel generalizing j with
  | zero => simp [chainFuel]; exact h
  | succ n ih =>
    simp only [chainFuel]; split_ifs with hc
    · exact ih (j + 1) (by omega)
    · exact h

theorem chain_stops_on_low (maxT minP : Nat) (p : Nat → Nat) (fuel j : Nat) (h : p (j - 1) < minP) :
    chainFuel maxT minP p (fuel + 1) j = j := by
  unfold chainFuel; split
  · rename_i hc; omega
  · rfl

-- § argument validity of draft

def draftArgsOk (T maxT a : Nat) : Prop := 1 ≤ T ∧ T ≤ maxT ∧ a < T

theorem draftArgs_row (T maxT a : Nat) (h : draftArgsOk T maxT a) : a < maxT := by
  obtain ⟨_, h2, h3⟩ := h; omega

-- § chunked prefill

def prefillChunk (maxT n c : Nat) : Nat := min maxT (n - c)

theorem prefillChunk_pos (maxT n c : Nat) (hm : 0 < maxT) (hc : c < n) :
    0 < prefillChunk maxT n c := by unfold prefillChunk; omega
theorem prefillChunk_end (maxT n c : Nat) (hc : c ≤ n) : c + prefillChunk maxT n c ≤ n := by
  unfold prefillChunk; omega
theorem prefillChunk_le (maxT n c : Nat) : prefillChunk maxT n c ≤ maxT := by
  unfold prefillChunk; omega
theorem skipped_chunk_unneeded (cell0 c T firstNeeded : Int) (h : cell0 + c + T ≤ firstNeeded)
    (t : Int) (_ht0 : 0 ≤ t) (ht : t < T) : cell0 + c + t < firstNeeded := by omega

-- § tensor lookup by name and kind

structure Tensor where
  name : String; kind : String; off : Nat

def findTensor (ts : List Tensor) (name kind : String) : Option Tensor :=
  ts.find? (fun t => t.name == name && t.kind == kind)

theorem findTensor_sound (ts : List Tensor) (name kind : String) (t : Tensor)
    (h : findTensor ts name kind = some t) : t.name = name ∧ t.kind = kind := by
  have := List.find?_some h; simpa using this
theorem findTensor_nil (name kind : String) : findTensor [] name kind = none := rfl

end Baloon.MtpDrafter
