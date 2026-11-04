/-!
# MODULE: session_graphs
SOURCES: src/core/session.cpp, include/strata/core/session.hpp
One token through all 48 layers: session arena sizing/carving, per-layer GDN/QSA state indexing, the
stage-prefix timing differences, and the dump-buffer layout of the host loop.

## Source semantics
* `session_bytes` = align256(sum of region sizes) + 4096; `session_init` carves each region at a 256-byte
  aligned cursor, so the carve must fit inside the reported size.
* GDN layers are numbered separately from QSA layers (`l % 4 == 3` is QSA): 36 GDN states, 12 QSA states.
* Stage times are differences of consecutive prefix times; the stage list telescopes to the last prefix.
* `session_loop` expects ring value `l + 1` from layer `l`; the dump holds `(n_layers + 1) * hc * n_embd` floats.

## Form template identifiers
- alignUp256 → align_up(n, SESSION_STATE_ALIGN) ; takeAll → session_init's `take` chain
- gdnIndex / qsaIndex → gdn_point_at / qsa_index ; stageDiffs → session_replay_stage_prefixes
-/

namespace Strata.SessionGraphs

-- § alignment to 256

/-- Round up to a multiple of 256 (`SESSION_STATE_ALIGN`). -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Rounding adds at most 255 bytes. -/
theorem alignUp256_le (n : Nat) : alignUp256 n ≤ n + 255 := by
  unfold alignUp256; omega

/-- The result is a multiple of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

-- § carving fits in the reported size

/-- `session_init`'s chain of `take` calls: advance the cursor by each region, re-aligned to 256. -/
def takeAll : Nat → List Nat → Nat
  | u, [] => u
  | u, x :: xs => takeAll (alignUp256 (u + x)) xs

/-- Each take wastes fewer than 256 bytes, so the end cursor is bounded by sum plus 255 per region. -/
theorem takeAll_le : ∀ (l : List Nat) (u : Nat), takeAll u l ≤ u + l.sum + 255 * l.length
  | [], u => by simp [takeAll]
  | x :: xs, u => by
    have h1 := alignUp256_le (u + x)
    have h2 := takeAll_le xs (alignUp256 (u + x))
    simp only [takeAll, List.sum_cons, List.length_cons]
    omega

/-- `session_bytes` formula. -/
def sessionBytes (parts : List Nat) : Nat := alignUp256 parts.sum + 4096

/-- With at most 16 regions (the session has 7) the carve always fits inside `session_bytes`. -/
theorem session_init_fits (parts : List Nat) (hlen : parts.length ≤ 16) :
    takeAll 0 parts ≤ sessionBytes parts := by
  have h1 := takeAll_le parts 0
  have h2 := alignUp256_ge parts.sum
  unfold sessionBytes
  omega

/-- The reported size is a multiple of 256 (4096 is a multiple of 256). -/
theorem sessionBytes_mod (parts : List Nat) : sessionBytes parts % 256 = 0 := by
  have := alignUp256_mod parts.sum
  unfold sessionBytes; omega

-- § per-layer state indices

/-- Whether layer `l` is QSA. -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- Index of GDN layer `l` among GDN layers: how many GDN layers precede it. -/
def gdnIndex (l : Nat) : Nat := ((List.range l).filter (fun i => !isQsa i)).length

/-- Index of QSA layer `l` among QSA layers. -/
def qsaIndex (l : Nat) : Nat := ((List.range l).filter isQsa).length

/-- Every GDN layer of the 48 gets a state slot below 36. -/
theorem gdnIndex_lt : ∀ l, l < 48 → l % 4 ≠ 3 → gdnIndex l < 36 := by
  decide

/-- Every QSA layer of the 48 gets a state slot below 12. -/
theorem qsaIndex_lt : ∀ l, l < 48 → l % 4 = 3 → qsaIndex l < 12 := by
  decide

/-- Consecutive GDN layers get consecutive slots, so no two layers share a recurrence. -/
theorem gdnIndex_succ : ∀ l, l < 47 → l % 4 ≠ 3 → gdnIndex (l + 1) = gdnIndex l + 1 := by
  decide

/-- Consecutive QSA layers get consecutive slots. -/
theorem qsaIndex_succ : ∀ l, l < 47 → l % 4 = 3 → qsaIndex (l + 1) = qsaIndex l + 1 := by
  decide

/-- Two GDN layers below 48 with equal slots are the same layer. -/
theorem gdnIndex_inj : ∀ a, a < 48 → ∀ b, b < 48 → a % 4 ≠ 3 → b % 4 ≠ 3 →
    gdnIndex a = gdnIndex b → a = b := by
  decide

-- § stage timing by prefix differences

/-- Stage times from prefix times t1..t5 (stage 0 is t1, stage i is t_{i+1} - t_i). -/
def stageDiffs (t1 t2 t3 t4 t5 : Nat) : List Nat := [t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4]

/-- For nondecreasing prefix times the stages telescope back to the last prefix. -/
theorem stageDiffs_sum (t1 t2 t3 t4 t5 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) (h3 : t3 ≤ t4)
    (h4 : t4 ≤ t5) : (stageDiffs t1 t2 t3 t4 t5).sum = t5 := by
  simp only [stageDiffs, List.sum_cons, List.sum_nil]
  omega

/-- The mixer time per layer is prefix 3 (stages 0..2): stage 0 + 1 + 2 equals t3. -/
theorem mixer_is_prefix3 (t1 t2 t3 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) :
    t1 + (t2 - t1) + (t3 - t2) = t3 := by
  omega

-- § ring numbering and dump layout

/-- The ring value `session_loop` waits for at layer `l`. -/
def wantRing (l : Nat) : Nat := l + 1

/-- Rings are strictly increasing in layer order. -/
theorem wantRing_strict (a b : Nat) (h : a < b) : wantRing a < wantRing b := by
  unfold wantRing; omega

/-- Slot `l+1` of the residual dump lies inside the `(L+1)`-slot buffer. -/
theorem dump_slot_in_bounds (l L n : Nat) (h : l < L) : (l + 1) * n + n ≤ (L + 1) * n := by
  have h2 : (l + 1) * n ≤ L * n := Nat.mul_le_mul_right n (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h2 ⊢
  omega

-- § per-token handoff bookkeeping

/-- Parts staging size `k * n_embd * 4` is positive for positive inputs, so the pinned scratch is non-empty. -/
theorem parts_bytes_pos (k n : Nat) (hk : 0 < k) (hn : 0 < n) : 0 < k * n * 4 := by
  have : 0 < k * n := Nat.mul_pos hk hn
  omega

/-- Token-graph flush interval check: a ring wait is only escalated after a `1024`-spin boundary. -/
def spinBoundary (spins : Nat) : Bool := decide ((spins + 1) % 1024 = 0)

/-- The boundary fires exactly every 1024th spin. -/
theorem spinBoundary_period (s : Nat) : spinBoundary (s + 1024) = spinBoundary s := by
  unfold spinBoundary
  simp only [decide_eq_decide]
  omega

end Strata.SessionGraphs
