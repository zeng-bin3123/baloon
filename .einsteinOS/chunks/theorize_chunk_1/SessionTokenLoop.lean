import Mathlib

/-!
# MODULE: session_token_loop
One token through all 48 layers: the 12 QSA / 36 GDN split, the per-layer pre/pool/hits/post schedule, graph
launch counts, the PLE history layout, the residual-ladder dump layout, stage-prefix differencing, and the
idempotence of capture / free.

## Source semantics
* 48 layers = 12 QSA + 36 GDN; the per-layer loop launches `pre[l]` and `post[l]` (96 graphs per token).
* Within a layer the host order is `pre`, pool, hits, `post`; the pool result is combined with the SAME layer's
  router weights.
* `ple_hist[row + NG_HIST * channel]` is row-fastest and stays inside `NG_HIST * hc_dim`.
* `dump_layers` has `(n_layers + 1) * hc * n_embd` floats; slot `l + 1` is the residual after layer `l`.
  A residual snapshot is `hc * n_embd * 4 = 40960` bytes with `hc = 4`, `n_embd = 2560`.
* Stage times come from differencing prefix times `T(k)`; the five stage times telescope to `T(5) - T(0)`.
* `session_capture` is idempotent; `SessionLoopScratch::free` is safe to call twice.

## Form template identifiers
* `Phase` / `layerPhases` -> `LAYER_HOST_SCHEDULE`
* `pleIdx`                -> `PLE_HIST_INDEX`
* `dumpSlot`              -> `LADDER_SLOT`
* `stageTimes`            -> `PREFIX_DIFFERENCES`
-/

namespace StrataSession

-- § layer census and launch counts

/-- 12 QSA layers plus 36 GDN layers make the 48-layer stack. -/
theorem layer_census : 12 + 36 = 48 := by decide

/-- Two graphs (pre and post) per layer is 96 launches per token. -/
theorem launches_per_token : 2 * 48 = 96 := by decide

/-- `session_replay` (pre graphs only) launches half of what a full token needs. -/
theorem replay_omits_posts : 96 - 48 = 48 := by decide

-- § per-layer host schedule

/-- The four host-visible phases of one layer in the split-graph loop. -/
inductive Phase where
  | pre | pool | hits | post
  deriving DecidableEq, Repr

/-- Rank of a phase in the layer. -/
def rank : Phase → Nat
  | .pre => 0
  | .pool => 1
  | .hits => 2
  | .post => 3

/-- Phases in the order the loop issues them. -/
def layerPhases : List Phase := [.pre, .pool, .hits, .post]

/-- The pool runs after the router published its ids, and hits run after the misses are staged but before
`post[l]` reads the sum. -/
theorem layer_order : layerPhases.Pairwise (fun a b => rank a < rank b) := by decide

-- § PLE history layout

/-- Row-fastest index into the PLE history: `row + NG_HIST * channel`. -/
def pleIdx (ngHist row ch : Nat) : Nat := row + ngHist * ch

/-- The index stays inside `NG_HIST * hc_dim`. -/
theorem pleIdx_lt (ngHist hcDim row ch : Nat) (hr : row < ngHist) (hc : ch < hcDim) :
    pleIdx ngHist row ch < ngHist * hcDim := by
  unfold pleIdx
  have h1 : ngHist * (ch + 1) ≤ ngHist * hcDim := Nat.mul_le_mul_left _ hc
  have h2 : ngHist * (ch + 1) = ngHist * ch + ngHist := Nat.mul_succ ngHist ch
  omega

/-- Distinct `(row, channel)` pairs never collide. -/
theorem pleIdx_inj (ngHist r c r' c' : Nat) (hr : r < ngHist) (hr' : r' < ngHist)
    (h : pleIdx ngHist r c = pleIdx ngHist r' c') : r = r' ∧ c = c' := by
  have hn : 0 < ngHist := by omega
  unfold pleIdx at h
  have h1 : (r + ngHist * c) / ngHist = c := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt hr, Nat.zero_add]
  have h2 : (r' + ngHist * c') / ngHist = c' := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt hr', Nat.zero_add]
  have hc : c = c' := by rw [← h1, ← h2, h]
  subst hc
  omega

-- § residual-ladder dump

/-- Slot index of the residual after layer `l` (slot 0 is the entry residual). -/
def dumpSlot (l : Nat) : Nat := l + 1

/-- Every layer's slot lies inside the `n_layers + 1` slots of the dump. -/
theorem dumpSlot_lt (nl l : Nat) (h : l < nl) : dumpSlot l < nl + 1 := by
  unfold dumpSlot; omega

/-- A snapshot of `hc` streams of `n_embd` floats is 40 KiB-ish: 4 * 2560 * 4 = 40960 bytes. -/
theorem snapshot_bytes : 4 * 2560 * 4 = 40960 := by decide

-- § stage-prefix differencing

/-- Stage `i` time is the difference of consecutive prefix times. -/
def stageTime (T : Nat → Int) (i : Nat) : Int := T (i + 1) - T i

/-- The five stage times sum to the full-prefix time minus the empty prefix. -/
theorem stages_telescope (T : Nat → Int) :
    stageTime T 0 + stageTime T 1 + stageTime T 2 + stageTime T 3 + stageTime T 4 = T 5 - T 0 := by
  unfold stageTime
  show (T 1 - T 0) + (T 2 - T 1) + (T 3 - T 2) + (T 4 - T 3) + (T 5 - T 4) = T 5 - T 0
  omega

-- § idempotent capture and free

/-- Capture flag after a capture call. -/
def captureCall (_captured : Bool) : Bool := true

/-- Capturing twice is a no-op. -/
theorem capture_idempotent (c : Bool) : captureCall (captureCall c) = captureCall c := rfl

/-- Free resets to the empty state. -/
def freeScratch {α : Type} (_s : Option α) : Option α := none

/-- Freeing twice is safe. -/
theorem free_idempotent {α : Type} (s : Option α) : freeScratch (freeScratch s) = freeScratch s := rfl

end StrataSession
