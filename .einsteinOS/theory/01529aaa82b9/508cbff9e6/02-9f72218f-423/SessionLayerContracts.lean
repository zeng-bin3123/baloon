/-!
# MODULE: session_layer_contracts
Numeric and protocol contracts of the session loop, the doorbell handoff, the speculative verify window and
the MTP drafter (layer.hpp, session.hpp, verify.hpp, mtp.hpp).

## Source semantics
* 48 layers = 12 QSA + 36 GDN.
* Doorbell: the device `doorbell_wait` passes iff the host's flag equals the ring value; the host writes the
  ring value after serving; rings are strictly increasing so a stale flag blocks.
* Verify window: `1 <= T <= max_t <= 8` (exec table has 9 entries); `commit(n_keep)` needs `1 <= n_keep <= T`;
  a window of T tokens carries `T - 1` drafts.
* R0.11 stage prefixes: stage `i` time is `prefix(i+1) - prefix(i)`, so the five stage times telescope to
  `prefix 5 - prefix 0`.
* Parts buffer: `k * n_embd` floats.

## Form template identifiers
* `Doorbell::h_seq`/`h_flag` -> `ring`/`flag`
* `Verifier::run`/`commit` -> `windowOk`/`commitOk`
* `session_replay_stage_prefixes` -> `stageTime`/`sumStages`
* `SessionLoopScratch::parts_bytes` -> `partsBytes`
-/

namespace Strata.SessionContracts

-- § layer split

/-- Number of QSA layers. -/
def nQsa : Nat := 12
/-- Number of GDN layers. -/
def nGdn : Nat := 36

/-- The two layer kinds cover all 48 layers. -/
theorem layers_total : nQsa + nGdn = 48 := by
  unfold nQsa nGdn; rfl

-- § doorbell protocol

/-- The device wait passes when the host's flag equals the ring value. -/
def devicePasses (flag ring : Nat) : Prop := flag = ring

/-- After serving ring `r` the host's flag is `r`. -/
def hostServe (_flag ring : Nat) : Nat := ring

/-- The next ring value. -/
def nextRing (r : Nat) : Nat := r + 1

/-- Once the host has served a ring, the device wait passes. -/
theorem served_passes (flag ring : Nat) : devicePasses (hostServe flag ring) ring := by
  unfold devicePasses hostServe; rfl

/-- A flag left over from the previous ring does not satisfy the next ring. -/
theorem stale_flag_blocks (r : Nat) : ¬ devicePasses r (nextRing r) := by
  unfold devicePasses nextRing; omega

/-- After `doorbell_reset`, the zeroed flag does not satisfy the first ring (value 1). -/
theorem reset_blocks_first : ¬ devicePasses 0 (nextRing 0) := by
  unfold devicePasses nextRing; omega

-- § verify window

/-- Window parameters accepted by `Verifier::init`/`run`. -/
def windowOk (T maxT : Nat) : Prop := 1 ≤ T ∧ T ≤ maxT ∧ maxT ≤ 8

/-- Valid windows index the 9-entry graph table in range. -/
theorem window_exec_index (T maxT : Nat) (h : windowOk T maxT) : T < 9 := by
  unfold windowOk at h; omega

/-- `commit(n_keep)`: keep a non-empty prefix of the window. -/
def commitOk (nKeep T : Nat) : Prop := 1 ≤ nKeep ∧ nKeep ≤ T

/-- Drafts carried by a window of `T` tokens. -/
def draftCount (T : Nat) : Nat := T - 1

/-- The window is the last accepted token plus the drafts. -/
theorem window_is_drafts_plus_one (T : Nat) (h : 1 ≤ T) : draftCount T + 1 = T := by
  unfold draftCount; omega

/-- Rejected rows are never committed: the kept prefix leaves `T - nKeep` rows to be overwritten. -/
theorem commit_rest (nKeep T : Nat) (h : commitOk nKeep T) : nKeep + (T - nKeep) = T := by
  unfold commitOk at h; omega

-- § stage prefixes

/-- Time of stage `i` from the prefix sweep `f`. -/
def stageTime (f : Nat → Int) (i : Nat) : Int := f (i + 1) - f i

/-- Sum of the five stage times. -/
def sumStages (f : Nat → Int) : Int :=
  stageTime f 0 + stageTime f 1 + stageTime f 2 + stageTime f 3 + stageTime f 4

/-- The five stage times telescope to the full-graph prefix minus the empty prefix. -/
theorem stages_telescope (f : Nat → Int) : sumStages f = f 5 - f 0 := by
  simp only [sumStages, stageTime, Nat.reduceAdd]
  omega

-- § parts buffer

/-- Bytes of the pool's `k x n_embd` float staging buffer. -/
def partsBytes (k nEmbd : Nat) : Nat := k * nEmbd * 4

/-- The parts buffer is a whole number of floats. -/
theorem partsBytes_mod (k nEmbd : Nat) : partsBytes k nEmbd % 4 = 0 := by
  unfold partsBytes; omega

end Strata.SessionContracts
