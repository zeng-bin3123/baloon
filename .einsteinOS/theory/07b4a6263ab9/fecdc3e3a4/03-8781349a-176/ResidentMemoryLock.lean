/-!
# MODULE: resident_memory_lock
SOURCES: src/platform/memory.cpp
Keeping the expert arena resident without privileges: on Windows, raise the working-set minimum by
the region plus a 512 MiB margin and `VirtualLock` it in 1 GiB chunks; on POSIX, one `mlock`.

## Source semantics
* Null pointer or zero bytes: nothing locked, `ok = false`.
* Windows: `new_min = min + bytes + margin`, `new_max = max(max, new_min + margin)`; then chunks
  `[off, off + min(chunk, bytes - off))`; `locked_bytes` is the end of the last successful chunk;
  a failure stops the loop with `ok = (off > 0)`; all chunks succeeding gives `locked = bytes`.
* POSIX: `mlock` is all or nothing: `locked ∈ {0, bytes}`.

## Form template identifiers
`lock_resident` (Windows) → `lockChunks`; working-set sizing → `newMin`/`newMax`;
`lock_resident` (POSIX) → `mlockResult`; `LockResult` → `(ok, locked)`.
-/

namespace Strata.ResidentMemoryLock

-- § working-set sizing

/-- The 512 MiB headroom. -/
def margin : Nat := 512 * 2 ^ 20

/-- Raised minimum working set. -/
def newMin (minWs bytes : Nat) : Nat := minWs + bytes + margin

/-- Maximum working set: kept if already large enough, else `new_min + margin`. -/
def newMax (minWs maxWs bytes : Nat) : Nat :=
  if maxWs > newMin minWs bytes + margin then maxWs else newMin minWs bytes + margin

/-- The minimum grows by at least the locked region. -/
theorem newMin_covers (m b : Nat) : m + b ≤ newMin m b := by unfold newMin; omega

/-- The maximum always leaves `margin` above the new minimum. -/
theorem newMax_ge (m M b : Nat) : newMin m b + margin ≤ newMax m M b := by
  unfold newMax; split <;> omega

-- § chunked VirtualLock

/-- The 1 GiB lock chunk. -/
def chunk : Nat := 2 ^ 30

/-- Lock `[off, bytes)` in chunks of `c`; `ok i` says whether the chunk at offset `i` locks.
Returns `(ok, locked)`; `fuel` bounds the number of chunks. -/
def lockChunks (ok : Nat → Bool) (c bytes : Nat) : Nat → Nat → Nat → Bool × Nat
  | 0, _, locked => (true, locked)
  | fuel + 1, off, locked =>
    if off < bytes then
      if ok off then lockChunks ok c bytes fuel (off + c) (min bytes (off + c))
      else (decide (0 < off), locked)
    else (true, locked)

/-- When every chunk locks and there is fuel, the whole region is locked. -/
theorem lockChunks_all (c bytes : Nat) :
    ∀ fuel off locked, off ≤ bytes → bytes ≤ off + fuel * c → (off < bytes ∨ locked = bytes) →
      lockChunks (fun _ => true) c bytes fuel off locked = (true, bytes) := by
  intro fuel
  induction fuel with
  | zero =>
    intro off locked h1 h2 h3
    simp only [Nat.zero_mul, Nat.add_zero] at h2
    simp only [lockChunks]
    rcases h3 with h3 | h3
    · omega
    · rw [h3]
  | succ fuel ih =>
    intro off locked h1 h2 h3
    simp only [lockChunks]
    split
    · simp only [if_true]
      by_cases hb : off + c ≤ bytes
      · exact ih (off + c) _ hb (by rw [Nat.succ_mul] at h2; omega)
          (by by_cases hl : off + c < bytes
              · exact Or.inl hl
              · exact Or.inr (by simp [Nat.min_def]; omega))
      · have hmin : min bytes (off + c) = bytes := by simp [Nat.min_def]; omega
        rw [hmin]
        cases fuel with
        | zero => simp [lockChunks]
        | succ f =>
          simp only [lockChunks]
          have : ¬ (off + c < bytes) := by omega
          simp only [this, if_false]
    · rcases h3 with h3 | h3
      · omega
      · rw [h3]

/-- The locked count never exceeds the region. -/
theorem lockChunks_le (ok : Nat → Bool) (c bytes : Nat) :
    ∀ fuel off locked, locked ≤ bytes → (lockChunks ok c bytes fuel off locked).2 ≤ bytes := by
  intro fuel
  induction fuel with
  | zero => intro off locked h; simpa [lockChunks] using h
  | succ fuel ih =>
    intro off locked h
    simp only [lockChunks]
    split
    · split
      · exact ih _ _ (Nat.min_le_left _ _)
      · exact h
    · exact h

/-- A first-chunk failure reports `ok = false` and nothing locked. -/
theorem lockChunks_first_fail (ok : Nat → Bool) (c bytes fuel : Nat) (hb : 0 < bytes) (h0 : ok 0 = false) :
    lockChunks ok c bytes (fuel + 1) 0 0 = (false, 0) := by
  simp [lockChunks, hb, h0]

/-- The 32 chunks of a 32 GiB arena suffice to cover it. -/
theorem fuel_32GiB : 32 * 2 ^ 30 ≤ 0 + 32 * chunk := by decide

-- § POSIX mlock

/-- POSIX `lock_resident`: `(ok, locked)` from whether `mlock` succeeded. -/
def mlockResult (nonNull : Bool) (bytes : Nat) (mlockOk : Bool) : Bool × Nat :=
  if !nonNull || bytes == 0 then (false, 0)
  else if mlockOk then (true, bytes) else (false, 0)

/-- `mlock` is all or nothing. -/
theorem mlock_all_or_nothing (p : Bool) (b : Nat) (m : Bool) :
    (mlockResult p b m).2 = 0 ∨ (mlockResult p b m).2 = b := by
  unfold mlockResult; split
  · exact Or.inl rfl
  · split <;> simp

/-- `ok` is reported exactly when the whole non-empty region is locked. -/
theorem mlock_ok_iff (p : Bool) (b : Nat) (m : Bool) :
    (mlockResult p b m).1 = true ↔ p = true ∧ b ≠ 0 ∧ m = true := by
  unfold mlockResult
  cases p <;> cases m <;> by_cases hb : b = 0 <;> simp [hb]

end Strata.ResidentMemoryLock
