/-!
# MODULE: prefill_chunk_orchestration
Host-side orchestration of batched prefill: bump allocation of the device buffers, the chunk loop,
the n-gram window carried across chunks, expert grouping by counting sort, and the staging/dequant rings.

## Source semantics
* `Alloc::take<T>(n)` reserves `((n*sizeof(T) + 256 + 255) & ~255)` bytes: at least 256 bytes of slack and a
  multiple of 256. A borrowed region fails when `used + bytes > cap`; count-only mode just sums.
* The chunk loop visits `c0 = 0, T, 2T, ...` with chunk length `min(T, n - c0)`.
* The n-gram window `prev = (older, newer)` advances `(p0, p1) -> (p1, tok)` per token, including when PLE is off.
* Experts are grouped with a counting sort: `off[e+1] = off[e] + cnt[e]`, `slot = fill[e]++`.
* Staging ring of 8 slots with lookahead 7; dequantised ring of 2 slots (`j % 2`).

## Form template identifiers
Alloc.take -> bump_take ; ms_since -> elapsed_ms ; STAGE -> stage_ring_size ; DQ -> dequant_ring_size ;
cnt/off -> expert_counts/expert_offsets ; ple_prev -> ngram_window ; Prefill::run -> chunk_loop
-/

namespace Strata.Prefill.Orchestration

-- § allocator

/-- Bytes reserved for a request of `b` raw bytes. -/
def takeBytes (b : Nat) : Nat := (b + 256 + 255) / 256 * 256

/-- A reservation is at least the request plus 256 bytes. -/
theorem takeBytes_ge (b : Nat) : b + 256 ≤ takeBytes b := by
  unfold takeBytes
  omega

/-- A reservation wastes less than 512 bytes. -/
theorem takeBytes_lt (b : Nat) : takeBytes b < b + 512 := by
  unfold takeBytes
  omega

/-- Every reservation is a multiple of 256 (all sub-buffers stay 256-aligned). -/
theorem takeBytes_aligned (b : Nat) : takeBytes b % 256 = 0 := by
  unfold takeBytes
  omega

/-- A bump pointer that starts aligned stays aligned. -/
theorem bump_aligned (used b : Nat) (h : used % 256 = 0) : (used + takeBytes b) % 256 = 0 := by
  have := takeBytes_aligned b
  omega

/-- A successful bump reservation stays within the region. -/
theorem bump_fits (used cap b : Nat) (h : used + takeBytes b ≤ cap) : used + takeBytes b ≤ cap := h

/-- Slack added by `bytes_needed` never reduces the size. -/
theorem bytes_needed_ge (used : Nat) : used ≤ used + 8388608 := by omega

/-- The 32 MiB element scratch and the 32 MiB workspace constants. -/
theorem gemm_constants : (32 : Nat) * 1048576 = 33554432 ∧ 8 * 1048576 = 8388608 := by decide

-- § chunk loop

/-- Start of chunk `i`. -/
def chunkStart (T i : Nat) : Nat := i * T

/-- Length of chunk `i` of an `n`-token run. -/
def chunkLen (T n i : Nat) : Nat := min T (n - i * T)

/-- A visited chunk is non-empty and ends inside the run. -/
theorem chunk_bounds (T n c0 : Nat) (hT : 0 < T) (h : c0 < n) :
    0 < min T (n - c0) ∧ c0 + min T (n - c0) ≤ n := by omega

/-- Consecutive full chunks abut. -/
theorem chunk_abut (T n i : Nat) (h : i * T + T ≤ n) :
    chunkStart T (i + 1) = chunkStart T i + chunkLen T n i := by
  unfold chunkStart chunkLen
  have e : (i + 1) * T = i * T + T := Nat.succ_mul i T
  omega

/-- The chunk containing the end of the run stops exactly at `n`. -/
theorem chunk_last_ends_at_n (T n i : Nat) (h1 : i * T < n) (h2 : n ≤ i * T + T) :
    chunkStart T i + chunkLen T n i = n := by
  unfold chunkStart chunkLen
  omega

-- § PLE batched rows

/-- One batched PLE request reads `T*16` rows into `T*16*160 = T*2560` floats. -/
theorem ple_batch_floats (T : Nat) : T * 16 * 160 = T * 2560 := by omega

/-- Row index of (token, head) inside the batch. -/
theorem ple_rows_index (t h T : Nat) (ht : t < T) (hh : h < 16) : t * 16 + h < T * 16 := by omega

-- § the n-gram window

/-- Advance the two-token window by one token. -/
def advance (p : Int × Int) (tok : Int) : Int × Int := (p.2, tok)

/-- The window holds the two newest tokens (`-1` when absent). -/
def prevWindow (l : List Int) : Int × Int := (l.reverse.getD 1 (-1), l.reverse.getD 0 (-1))

/-- Advancing over a prompt prefix gives exactly the window the token path reconstructs from the prompt. -/
theorem advance_fold (l : List Int) : l.foldl advance (-1, -1) = prevWindow l := by
  induction l using List.reverseRecOn with
  | nil => rfl
  | append_singleton l t ih =>
      rw [List.foldl_append, ih]
      simp [advance, prevWindow, List.reverse_append]

/-- Two tokens fully replace the window. -/
theorem advance_two (p : Int × Int) (a b : Int) : [a, b].foldl advance p = (a, b) := rfl

-- § expert grouping (counting sort)

/-- Exclusive prefix sums of the per-expert counts. -/
def offs (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offs cnt e + cnt e

/-- Offsets are monotone. -/
theorem offs_mono (cnt : Nat → Nat) : ∀ {e e' : Nat}, e ≤ e' → offs cnt e ≤ offs cnt e' := by
  intro e e' h
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => exact Nat.le_trans ih (by simp [offs])

/-- The k-th assignment of expert `e` lands inside expert `e`'s own range. -/
theorem slot_in_range (cnt : Nat → Nat) (e k : Nat) (hk : k < cnt e) :
    offs cnt e ≤ offs cnt e + k ∧ offs cnt e + k < offs cnt (e + 1) := by
  simp [offs]
  omega

/-- Ranges of different experts are disjoint. -/
theorem ranges_disjoint (cnt : Nat → Nat) (e e' : Nat) (h : e < e') : offs cnt (e + 1) ≤ offs cnt e' :=
  offs_mono cnt h

-- § rings

/-- Staging slot pointer step. -/
theorem stage_next_lt (s : Nat) : (s + 1) % 8 < 8 := by omega

/-- With lookahead 7, slots of staged blobs `i < k ≤ i + 7` are distinct: no in-flight slot is overwritten. -/
theorem stage_slots_distinct (i k : Nat) (h1 : i < k) (h2 : k - i < 8) : i % 8 ≠ k % 8 := by omega

/-- The dequantised ring alternates between its two slots. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by omega

end Strata.Prefill.Orchestration
