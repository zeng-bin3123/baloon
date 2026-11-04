/-!
# MODULE: prefill_pipeline
Batched prompt processing: the prompt is cut into chunks, every layer runs on a chunk at once, and the MoE half
groups the (token, k) routing pairs by expert so each expert is dequantized once per chunk.

## Source semantics
* Chunks of at most `T` tokens cover the prompt exactly once.
* Expert grouping builds prefix-sum offsets from per-expert counts; the final offset is `T * K`.
* All device buffers are bump-allocated in 256-byte aligned pieces with 256 bytes of slack.
* The expert staging ring has 8 slots and looks ahead at most 7 experts, so a slot is never overwritten in flight.
* GDN state per layer is `S*HV*S` recurrence floats plus `C*(d_conv-1)` conv floats.

## Form template identifiers
* `m.T` -> `chunk`, `m.cnt`/`m.off` -> `cnt`/`offs`, `Alloc::take` -> `allocBytes`/`takeBump`
* `STAGE` -> `stageSlots`, `DQ` -> `dqSlots`
-/

namespace Strata.PrefillPipeline

-- § chunking

/-- Chunk sizes `min T remaining`, produced with fuel so the definition is structural. -/
def chunks (T : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, n => if n = 0 ∨ T = 0 then [] else min T n :: chunks T fuel (n - min T n)

/-- The chunks partition the prompt: their sizes add up to the number of tokens. -/
theorem chunks_sum (T : Nat) (hT : 0 < T) :
    ∀ (fuel n : Nat), n ≤ fuel → (chunks T fuel n).sum = n
  | 0, n, h => by
      have : n = 0 := by omega
      subst this
      simp [chunks]
  | fuel + 1, n, h => by
      by_cases hn : n = 0
      · subst hn
        simp [chunks]
      · have hc : ¬ (n = 0 ∨ T = 0) := by omega
        have ih := chunks_sum T hT fuel (n - min T n) (by omega)
        simp only [chunks, if_neg hc, List.sum_cons]
        omega

/-- Every chunk is non-empty and at most `T` tokens. -/
theorem chunks_bounded (T : Nat) (hT : 0 < T) :
    ∀ (fuel n c : Nat), c ∈ chunks T fuel n → 0 < c ∧ c ≤ T
  | 0, n, c, h => by simp [chunks] at h
  | fuel + 1, n, c, h => by
      by_cases hn : n = 0
      · subst hn
        simp [chunks] at h
      · have hc : ¬ (n = 0 ∨ T = 0) := by omega
        simp only [chunks, if_neg hc, List.mem_cons] at h
        rcases h with h | h
        · omega
        · exact chunks_bounded T hT fuel _ c h

-- § expert grouping

/-- Prefix-sum offsets starting at `a`, one entry per expert plus the final total. -/
def offs : Nat → List Nat → List Nat
  | a, [] => [a]
  | a, c :: cs => a :: offs (a + c) cs

/-- Offsets list has one entry more than the counts list. -/
theorem offs_length : ∀ (cs : List Nat) (a : Nat), (offs a cs).length = cs.length + 1
  | [], a => by simp [offs]
  | c :: cs, a => by simp [offs, offs_length cs]

/-- Final offset after consuming the counts. -/
def finalOff : Nat → List Nat → Nat
  | a, [] => a
  | a, c :: cs => finalOff (a + c) cs

/-- The final offset is the total number of routed pairs. -/
theorem finalOff_eq : ∀ (cs : List Nat) (a : Nat), finalOff a cs = a + cs.sum
  | [], a => by simp [finalOff]
  | c :: cs, a => by
      have ih := finalOff_eq cs (a + c)
      simp only [finalOff, List.sum_cons]
      omega

/-- The source token of pair `i` is `i / K` and is a valid chunk row (`K = 10`). -/
theorem pair_src_lt (T i : Nat) (h : i < T * 10) : i / 10 < T := by
  omega

-- § allocation

/-- `Alloc::take` rounding: payload plus 256 bytes of slack, up to a multiple of 256. -/
def allocBytes (x : Nat) : Nat := (x + 256 + 255) / 256 * 256

/-- Each allocation holds the payload and the slack. -/
theorem allocBytes_ge (x : Nat) : x + 256 ≤ allocBytes x := by
  unfold allocBytes
  omega

/-- Allocations are 256-byte aligned. -/
theorem allocBytes_aligned (x : Nat) : allocBytes x % 256 = 0 := by
  unfold allocBytes
  omega

/-- A bump allocation succeeds only inside the borrowed region. -/
def takeBump (used cap x : Nat) : Option Nat :=
  if used + allocBytes x ≤ cap then some (used + allocBytes x) else none

/-- A successful bump never exceeds the capacity. -/
theorem takeBump_le (used cap x u : Nat) (h : takeBump used cap x = some u) : u ≤ cap := by
  unfold takeBump at h
  split at h
  · injection h with h
    omega
  · simp at h

/-- Bump offsets stay aligned. -/
theorem takeBump_aligned (used cap x u : Nat) (ha : used % 256 = 0)
    (h : takeBump used cap x = some u) : u % 256 = 0 := by
  unfold takeBump at h
  split at h
  · injection h with h
    have := allocBytes_aligned x
    omega
  · simp at h

/-- Number of lent slots: ceiling of the need over the blob size covers the need. -/
theorem lent_slots_cover (need blob : Nat) (h : 0 < blob) :
    need ≤ (need + blob - 1) / blob * blob := by
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h2 := Nat.mod_lt (need + blob - 1) h
  generalize (need + blob - 1) / blob = q at *
  generalize (need + blob - 1) % blob = r at *
  rw [Nat.mul_comm] at h1
  omega

-- § staging ring

/-- Slots in the host-to-device expert staging ring. -/
def stageSlots : Nat := 8

/-- Slots in the dequantized-expert ring. -/
def dqSlots : Nat := 2

/-- The ring pointer stays in range. -/
theorem stage_next_lt (s : Nat) : (s + 1) % stageSlots < stageSlots := by
  unfold stageSlots
  omega

/-- The dequant ring index stays in range. -/
theorem dq_index_lt (j : Nat) : j % dqSlots < dqSlots := by
  unfold dqSlots
  omega

/-- The look-ahead window (`STAGE - 1` experts) never exceeds the ring size. -/
theorem lookahead_in_ring (staged j : Nat) (h : staged ≤ j + (stageSlots - 1)) :
    staged - j < stageSlots + 1 := by
  unfold stageSlots at *
  omega

-- § model geometry facts

/-- GDN state floats per layer: recurrence `128*48*128` plus conv `10240*3`. -/
def gdnFloats : Nat := 128 * 48 * 128 + 10240 * 3

/-- The value matches the planner's byte count (817152 floats = 3268608 bytes). -/
theorem gdnFloats_value : gdnFloats = 817152 ∧ gdnFloats * 4 = 3268608 := by
  unfold gdnFloats
  decide

/-- The channel layout of the GDN conv: q, k and v heads fill `C = 10240`. -/
theorem gdn_channels : 2 * 16 * 128 + 48 * 128 = 10240 := by decide

/-- The indexer's block index of any in-range position fits `max_cells / 4 + 2` blocks. -/
theorem block_index_lt (pos maxCells : Nat) (h : pos < maxCells) : pos / 4 < maxCells / 4 + 2 := by
  omega

end Strata.PrefillPipeline
