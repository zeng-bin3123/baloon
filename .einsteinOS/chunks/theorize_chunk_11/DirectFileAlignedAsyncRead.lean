/-!
# MODULE: direct_file_aligned_async_read
Model of `strata::platform::DirectFile`: an unbuffered, read-only file handle that accepts
aligned read requests tagged by the caller and reports completions through `wait`.
The POSIX backend completes each read inside `submit` and queues a completion FIFO;
the Windows backend splits 64-bit offsets and treats END_OF_FILE as a legal short read.

## Source semantics
* `submit` rejects any request whose offset, length or buffer address is not a multiple of
  the alignment, or whose length is zero; a rejected request changes no state.
* An accepted POSIX request enqueues exactly one completion carrying the caller's tag;
  a failed `pread` (negative result) yields `bytes = 0`, `ok = false`.
* `wait` returns at most `max` completions, in submission order, and removes them.
* `close` resets descriptor, size and pending completions.
* Windows: a completion is ok iff its NTSTATUS is 0 or STATUS_END_OF_FILE (0xC0000011);
  offsets are split into low/high 32-bit words without loss.

## Form template identifiers
* `DirectFile::submit`      → `submit`
* `DirectFile::wait`        → `wait`
* `DirectFile::close`       → `close`
* `DirectFile::Completion`  → `Completion`
* `alignment()`             → parameter `a`
* `Impl::done`              → `PosixState.done`
-/

namespace StrataPlatform.DirectFile

-- § Types

/-- A finished request: caller tag, bytes transferred, success flag. -/
structure Completion where
  tag : Nat
  bytes : Nat
  ok : Bool
  deriving DecidableEq

/-- POSIX backend state: descriptor (negative = closed), file size, FIFO of completions. -/
structure PosixState where
  fd : Int
  size : Nat
  done : List Completion

-- § Alignment check

/-- The request validity check performed at the top of `submit`. -/
def aligned (a off len buf : Nat) : Bool :=
  off % a == 0 && len % a == 0 && buf % a == 0 && len != 0

/-- Result of a `pread` as a completion. -/
def completionOf (tag : Nat) (got : Int) : Completion :=
  { tag := tag, bytes := if got < 0 then 0 else got.toNat, ok := decide (0 ≤ got) }

/-- POSIX `submit`: `none` is the unaligned-request error; otherwise one completion is queued. -/
def submit (a : Nat) (s : PosixState) (off len buf tag : Nat) (got : Int) : Option PosixState :=
  if aligned a off len buf then
    some { s with done := s.done ++ [completionOf tag got] }
  else none

/-- POSIX `wait`: pop at most `max` completions from the front of the queue. -/
def wait (s : PosixState) (max : Nat) : List Completion × PosixState :=
  (s.done.take max, { s with done := s.done.drop max })

/-- POSIX `close`. -/
def close (s : PosixState) : PosixState :=
  { s with fd := -1, size := 0, done := [] }

/-- `is_open`. -/
def isOpen (s : PosixState) : Bool := decide (0 ≤ s.fd)

-- § submit

/-- A zero-length request is always rejected. -/
theorem submit_zero_length (a : Nat) (s : PosixState) (off buf tag : Nat) (got : Int) :
    submit a s off 0 buf tag got = none := by
  simp [submit, aligned]

/-- An unaligned request is rejected and no state is produced. -/
theorem submit_unaligned (a : Nat) (s : PosixState) (off len buf tag : Nat) (got : Int)
    (h : aligned a off len buf = false) :
    submit a s off len buf tag got = none := by
  simp [submit, h]

/-- An aligned request is accepted and appends exactly its completion. -/
theorem submit_aligned (a : Nat) (s : PosixState) (off len buf tag : Nat) (got : Int)
    (h : aligned a off len buf = true) :
    submit a s off len buf tag got =
      some { s with done := s.done ++ [completionOf tag got] } := by
  simp [submit, h]

/-- Accepting a request grows the pending queue by exactly one. -/
theorem submit_queue_length (a : Nat) (s s' : PosixState) (off len buf tag : Nat) (got : Int)
    (h : submit a s off len buf tag got = some s') :
    s'.done.length = s.done.length + 1 := by
  unfold submit at h
  split at h
  · cases h
    simp
  · cases h

/-- Accepting a request never touches descriptor or size. -/
theorem submit_preserves_file (a : Nat) (s s' : PosixState) (off len buf tag : Nat) (got : Int)
    (h : submit a s off len buf tag got = some s') :
    s'.fd = s.fd ∧ s'.size = s.size := by
  unfold submit at h
  split at h
  · cases h
    exact ⟨rfl, rfl⟩
  · cases h

/-- A failed read is reported with zero bytes and `ok = false`. -/
theorem completion_failed (tag : Nat) (got : Int) (h : got < 0) :
    completionOf tag got = ⟨tag, 0, false⟩ := by
  have h' : ¬ (0 ≤ got) := by omega
  simp [completionOf, h, h']

/-- A successful read reports its byte count and `ok = true`. -/
theorem completion_success (tag : Nat) (got : Int) (h : 0 ≤ got) :
    completionOf tag got = ⟨tag, got.toNat, true⟩ := by
  have h' : ¬ (got < 0) := by omega
  simp [completionOf, h, h']

/-- Every completion keeps the caller's tag. -/
theorem completion_tag (tag : Nat) (got : Int) : (completionOf tag got).tag = tag := rfl

-- § wait

/-- `wait` never returns more than `max` completions. -/
theorem wait_length_le (s : PosixState) (max : Nat) :
    (wait s max).1.length ≤ max := by
  simp [wait]

/-- `wait` partitions the queue: returned prefix followed by the remaining queue. -/
theorem wait_partition (s : PosixState) (max : Nat) :
    (wait s max).1 ++ (wait s max).2.done = s.done := by
  simp [wait]

/-- `wait` on an empty queue returns nothing (the POSIX backend never blocks). -/
theorem wait_empty (s : PosixState) (max : Nat) (h : s.done = []) :
    (wait s max).1 = [] := by
  simp [wait, h]

/-- `wait` with `max = 0` returns nothing and leaves the queue unchanged. -/
theorem wait_zero (s : PosixState) :
    (wait s 0).1 = [] ∧ (wait s 0).2.done = s.done := by
  simp [wait]

/-- When the queue fits in `max`, `wait` drains it completely. -/
theorem wait_drains (s : PosixState) (max : Nat) (h : s.done.length ≤ max) :
    (wait s max).1 = s.done ∧ (wait s max).2.done = [] := by
  constructor
  · simp [wait, List.take_of_length_le h]
  · simp [wait, List.drop_of_length_le h]

/-- Core round trip: submit an aligned read on an empty queue, then `wait` returns exactly it. -/
theorem submit_then_wait (a : Nat) (s s' : PosixState) (off len buf tag : Nat) (got : Int)
    (hempty : s.done = []) (hal : aligned a off len buf = true)
    (hs : submit a s off len buf tag got = some s') (max : Nat) (hmax : 1 ≤ max) :
    (wait s' max).1 = [completionOf tag got] ∧ (wait s' max).2.done = [] := by
  rw [submit_aligned a s off len buf tag got hal] at hs
  cases hs
  have hl : ([completionOf tag got] : List Completion).length ≤ max := by
    simpa using hmax
  simp [wait, hempty, List.take_of_length_le hl, List.drop_of_length_le hl]

/-- FIFO: a completion queued earlier is delivered before one queued later. -/
theorem wait_fifo_head (s : PosixState) (c : Completion) (rest : List Completion)
    (h : s.done = c :: rest) (max : Nat) (hmax : 1 ≤ max) :
    ∃ tl, (wait s max).1 = c :: tl := by
  obtain ⟨m, rfl⟩ : ∃ m, max = m + 1 := ⟨max - 1, by omega⟩
  exact ⟨rest.take m, by simp [wait, h]⟩

-- § close

/-- After `close` the handle is not open. -/
theorem close_not_open (s : PosixState) : isOpen (close s) = false := by
  simp [isOpen, close]

/-- `close` discards pending completions and resets the size. -/
theorem close_resets (s : PosixState) :
    (close s).done = [] ∧ (close s).size = 0 := by
  simp [close]

/-- `close` is idempotent. -/
theorem close_idempotent (s : PosixState) : close (close s) = close s := by
  simp [close]

-- § Windows backend facts

/-- Completion status predicate: success or STATUS_END_OF_FILE. -/
def statusOk (st : Nat) : Bool := st == 0 || st == 0xC0000011

/-- STATUS_SUCCESS is accepted. -/
theorem statusOk_success : statusOk 0 = true := by decide

/-- STATUS_END_OF_FILE is accepted: a short read at the last page is legal. -/
theorem statusOk_eof : statusOk 0xC0000011 = true := by decide

/-- Any other status is an error. -/
theorem statusOk_other (st : Nat) (h0 : st ≠ 0) (h1 : st ≠ 0xC0000011) :
    statusOk st = false := by
  simp [statusOk, h0, h1]

/-- The low/high 32-bit split used for `OVERLAPPED.Offset/OffsetHigh` is lossless. -/
theorem offset_split (off : Nat) :
    off % 4294967296 + (off / 4294967296) * 4294967296 = off := by
  omega

/-- The low word fits in 32 bits. -/
theorem offset_low_lt (off : Nat) : off % 4294967296 < 4294967296 := by
  omega

/-- For offsets below 2^64 the high word also fits in 32 bits. -/
theorem offset_high_lt (off : Nat) (h : off < 18446744073709551616) :
    off / 4294967296 < 4294967296 := by
  omega

/-- Request-pool `take`: reuse the head of the free list when nonempty. -/
def takePool {α : Type} (fresh : α) : List α → α × List α
  | [] => (fresh, [])
  | r :: rs => (r, rs)

/-- Taking from a nonempty pool shrinks it by one. -/
theorem takePool_shrinks {α : Type} (fresh r : α) (rs : List α) :
    (takePool fresh (r :: rs)).2.length + 1 = (r :: rs).length := by
  simp [takePool]

/-- Taking from an empty pool allocates the fresh request. -/
theorem takePool_empty {α : Type} (fresh : α) : takePool fresh [] = (fresh, []) := rfl

end StrataPlatform.DirectFile
