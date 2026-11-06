/-!
# MODULE: prefill_chunking
SOURCES: src/prefill/prefill.cpp
Chunked prompt prefill on the GPU: a device-buffer allocator (owned, borrowed region, or count-only), the chunk
loop over the prompt, the host-side counting sort that groups (token, expert) pairs, the expert staging ring and
dequantisation ring, and the two-token PLE history.

## Source semantics
* Every allocation takes `n*sz + 256` bytes rounded up to 256; a borrowed region fails the take when it would
  overflow; the count-only pass produces `bytes_needed` (plus 8 MiB slack).
* The prompt is processed in chunks of `min(T, n - c0)` tokens at `p0 = pos0 + c0`.
* Experts are grouped with prefix offsets `off[e+1] = off[e] + cnt[e]`; ids outside `[0, NE)` are an error.
* Non-resident experts stream through a ring of `STAGE = 8` buffers with lookahead `STAGE - 1`; dequantised
  experts alternate between `DQ = 2` buffers.
* The PLE history `prev` always holds the last two tokens seen.

## Form template identifiers
role mapping (source names -> abstract names):
  Alloc::take            -> region_take
  Prefill::bytes_needed  -> region_size
  Prefill::run chunk loop-> chunk_walk
  m.cnt / m.off          -> expert_histogram / expert_offsets
  STAGE / stage_next     -> upload_ring
  DQ                     -> dequant_ring
  prev[2]                -> token_history_pair
-/

namespace Strata.Prefill

-- § allocation accounting

/-- Bytes charged for `n` elements of `sz` bytes: `(n*sz + 256 + 255)` rounded down to a multiple of 256. -/
def allocBytes (n sz : Nat) : Nat := (n * sz + 256 + 255) / 256 * 256

/-- An allocation is at least its payload plus 256 bytes of slack. -/
theorem allocBytes_ge (n sz : Nat) : n * sz + 256 ≤ allocBytes n sz := by
  unfold allocBytes; omega

/-- Allocation sizes are multiples of 256. -/
theorem allocBytes_mod (n sz : Nat) : allocBytes n sz % 256 = 0 := by
  unfold allocBytes; omega

/-- The rounding adds less than 256 bytes beyond the slack. -/
theorem allocBytes_lt (n sz : Nat) : allocBytes n sz < n * sz + 512 := by
  unfold allocBytes; omega

/-- Borrowed-region take: `none` on overflow, otherwise the new `used`. -/
def takeAlloc (cap used n sz : Nat) : Option Nat :=
  if used + allocBytes n sz > cap then none else some (used + allocBytes n sz)

/-- A successful take stays inside the region and moves forward. -/
theorem takeAlloc_sound (cap used n sz u : Nat) (h : takeAlloc cap used n sz = some u) :
    u ≤ cap ∧ used + n * sz + 256 ≤ u := by
  unfold takeAlloc at h
  have := allocBytes_ge n sz
  split at h
  · cases h
  · injection h with h'
    subst h'
    omega

/-- A take succeeds exactly when it fits. -/
theorem takeAlloc_some_of_fits (cap used n sz : Nat) (h : used + allocBytes n sz ≤ cap) :
    takeAlloc cap used n sz = some (used + allocBytes n sz) := by
  unfold takeAlloc
  rw [if_neg (by omega)]

/-- Total bytes of a carve sequence. -/
def allocTotal (l : List (Nat × Nat)) : Nat := (l.map (fun p => allocBytes p.1 p.2)).sum

/-- The total of a concatenation is the sum of the totals. -/
theorem allocTotal_append (l1 l2 : List (Nat × Nat)) :
    allocTotal (l1 ++ l2) = allocTotal l1 + allocTotal l2 := by
  simp [allocTotal, List.map_append, List.sum_append]

/-- `bytes_needed` adds 8 MiB of alignment slack to the counted total. -/
def bytesNeeded (used : Nat) : Nat := used + 8 * 1048576

/-- The reported size always covers the counted total. -/
theorem bytesNeeded_ge (used : Nat) : used ≤ bytesNeeded used := by
  unfold bytesNeeded; omega

-- § chunk loop

/-- Chunk sizes of a prompt of `n` tokens with chunk capacity `T`, bounded by fuel. -/
def chunkList (T : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min T n :: chunkList T f (n - min T n)

/-- The chunks cover exactly the prompt. -/
theorem chunkList_sum (T : Nat) (hT : 1 ≤ T) : ∀ f n, n ≤ f → (chunkList T f n).sum = n := by
  intro f
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by omega
    subst h0
    simp [chunkList]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · subst h0
      simp [chunkList]
    · simp [chunkList, h0]
      have := ih (n - min T n) (by omega)
      omega

/-- Size of the chunk starting at offset `c0`. -/
def chunkAt (T n c0 : Nat) : Nat := min T (n - c0)

/-- A chunk inside the prompt is non-empty. -/
theorem chunkAt_pos (T n c0 : Nat) (hT : 1 ≤ T) (h : c0 < n) : 1 ≤ chunkAt T n c0 := by
  unfold chunkAt; omega

/-- A chunk never exceeds the buffer capacity. -/
theorem chunkAt_le (T n c0 : Nat) : chunkAt T n c0 ≤ T := by
  unfold chunkAt; omega

/-- A chunk never runs past the prompt. -/
theorem chunkAt_fits (T n c0 : Nat) (h : c0 < n) : c0 + chunkAt T n c0 ≤ n := by
  unfold chunkAt; omega

-- § counting sort of (token, expert) pairs

/-- `off[e]`: sum of the first `e` counts. -/
def offAt : List Nat → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | c :: cs, e + 1 => c + offAt cs e

/-- The last offset (`off[NE]`) is the total count of pairs. -/
theorem offAt_length (cnt : List Nat) : offAt cnt cnt.length = cnt.sum := by
  induction cnt with
  | nil => simp [offAt]
  | cons c cs ih => simp [offAt, ih]

/-- Offsets of the empty histogram are 0. -/
theorem offAt_nil (e : Nat) : offAt [] e = 0 := by
  cases e <;> rfl

/-- Offsets never decrease. -/
theorem offAt_mono (cnt : List Nat) : ∀ e, offAt cnt e ≤ offAt cnt (e + 1) := by
  induction cnt with
  | nil => intro e; simp [offAt_nil]
  | cons c cs ih =>
    intro e
    cases e with
    | zero => simp [offAt]
    | succ k =>
      simp only [offAt]
      have := ih k
      omega

/-- The first expert's region starts at 0. -/
theorem offAt_zero (cnt : List Nat) : offAt cnt 0 = 0 := by
  cases cnt <;> rfl

/-- The step from `off[e]` to `off[e+1]` adds exactly `cnt[e]`. -/
theorem offAt_succ_of_get (c : Nat) (cs : List Nat) (e : Nat) :
    offAt (c :: cs) (e + 1) = c + offAt cs e := rfl

/-- A routed id is valid exactly when it lies in `[0, NE)`. -/
def idOk (e : Int) (ne : Nat) : Bool := decide (0 ≤ e ∧ e < (ne : Int))

/-- Valid ids index a histogram of `NE` entries. -/
theorem idOk_index (e : Int) (ne : Nat) (h : idOk e ne = true) : e.toNat < ne := by
  have h' : 0 ≤ e ∧ e < (ne : Int) := of_decide_eq_true h
  omega

-- § staging and dequantisation rings

/-- Next ring slot. -/
def nextSlot (s : Nat) : Nat := (s + 1) % 8

/-- The next slot is a valid slot. -/
theorem nextSlot_lt (s : Nat) : nextSlot s < 8 := by
  unfold nextSlot; omega

/-- The experts staged at once (current plus lookahead of 7) occupy 8 distinct slots. -/
theorem stage_slots_distinct (a i1 i2 : Nat) (h1 : i1 < 8) (h2 : i2 < 8)
    (h : (a + i1) % 8 = (a + i2) % 8) : i1 = i2 := by
  omega

/-- Lookahead plus the current expert exactly fill the ring. -/
theorem lookahead_fills_ring : (8 - 1) + 1 = 8 := by decide

/-- Consecutive experts use different dequantisation buffers. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by
  omega

/-- Experts two apart reuse a dequantisation buffer (after the stream has synchronised on it). -/
theorem dq_reuse (j : Nat) : j % 2 = (j + 2) % 2 := by
  omega

-- § PLE history

/-- One token enters the two-token history. -/
def prevStep (p : Nat × Nat) (tok : Nat) : Nat × Nat := (p.2, tok)

/-- History after a token list. -/
def prevAdvance (p : Nat × Nat) (toks : List Nat) : Nat × Nat := toks.foldl prevStep p

/-- Appending a token pushes the previous newest into the older slot. -/
theorem prevAdvance_append (p : Nat × Nat) (l : List Nat) (x : Nat) :
    prevAdvance p (l ++ [x]) = ((prevAdvance p l).2, x) := by
  simp [prevAdvance, List.foldl_append, prevStep]

/-- After a run the newest entry is the last token. -/
theorem prevAdvance_snd (p : Nat × Nat) (l : List Nat) (x : Nat) :
    (prevAdvance p (l ++ [x])).2 = x := by
  simp [prevAdvance_append]

/-- Processing chunks one after another equals processing the concatenation. -/
theorem prevAdvance_concat (p : Nat × Nat) (l1 l2 : List Nat) :
    prevAdvance (prevAdvance p l1) l2 = prevAdvance p (l1 ++ l2) := by
  simp [prevAdvance, List.foldl_append]

-- § buffer-shape constants

/-- The streams times the width: `D = N * HC`. -/
theorem stream_width : 2560 * 4 = 10240 := by decide

/-- A gate/up row pair is `2 * 640 = 1280` wide. -/
theorem gate_up_width : 2 * 640 = 1280 := by decide

/-- The dequantised gate/up buffer holds 4 weights per code byte of the blob. -/
theorem dq_gu_elems : 1280 * 2560 = 1280 * 640 * 4 := by decide

/-- The dequantised down buffer holds 4 weights per code byte of the blob. -/
theorem dq_down_elems : 2560 * 640 = 2560 * 160 * 4 := by decide

/-- Geometry check of `init`: all five dimensions must match the artifact. -/
def geomOk (n hc lr ne k : Nat) : Bool :=
  decide (n = 2560 ∧ hc = 4 ∧ lr = 320 ∧ ne = 512 ∧ k = 10)

/-- A passing geometry makes each routed pair map into `T * 10` slots. -/
theorem geomOk_k (n hc lr ne k : Nat) (h : geomOk n hc lr ne k = true) : k = 10 := by
  have h' : n = 2560 ∧ hc = 4 ∧ lr = 320 ∧ ne = 512 ∧ k = 10 := of_decide_eq_true h
  exact h'.2.2.2.2

end Strata.Prefill
