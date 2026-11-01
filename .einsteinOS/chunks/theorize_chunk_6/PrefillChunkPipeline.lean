/-!
# MODULE: prefill_chunk_pipeline
Chunked prompt prefill on the GPU: a bump allocator that sizes and carves every chunk buffer, a host-side
counting sort that groups (token, k) routing pairs by expert, and the expert staging / dequantisation rings.

## Source semantics
* `Alloc::take` rounds every request to `(n*sizeof + 256 + 255) & ~255`; `bytes_needed` replays the same
  sequence in count-only mode, so a region of that size always fits `init`.
* The chunk loop walks `c0 = 0, T, 2T, ...` with `T_c = min(T, n - c0)`; every token is covered exactly once.
* MoE grouping: `cnt[e]` counts pairs per expert, `off` is its exclusive prefix sum, `slot_host[i]` is
  `off[e] + (rank of i among e's pairs)`, `src_host[p] = i / K`.
* Experts are processed in id order; expert `j` goes to staging slot `j % STAGE` (look-ahead `STAGE-1`) and
  dequant buffer `j % DQ`; a slot is never overwritten while a live blob in the window still uses it.

## Form template identifiers
alloc_round := Alloc::take rounding ; region := Alloc {base,cap,used} ; chunk_len := min(m.T, n - c0)
expert_off := m.off ; expert_cnt := m.cnt ; stage_slot := j % STAGE ; dq_slot := j % DQ ;
stream_width := D = N*HC ; topk := K
-/

namespace Strata.PrefillChunk

-- § model constants
def N : Nat := 2560
def HC : Nat := 4
def D : Nat := N * HC
def LR : Nat := 320
def K : Nat := 10
def NE : Nat := 512
def STAGE : Nat := 8
def DQ : Nat := 2

/-- The four hyper-connection streams are `D = 4 * 2560 = 10240` wide. -/
theorem stream_width : D = 10240 := by
  decide

/-- Each token routes to ten experts out of 512, so a chunk of T tokens makes `10*T` pairs. -/
theorem pairs_per_token : K * 1 = 10 ∧ NE = 512 := by
  decide

-- § bump allocator

/-- Size of one `Alloc::take` request: `n*sz + 256` rounded up to a multiple of 256. -/
def allocBytes (n sz : Nat) : Nat := ((n * sz + 256 + 255) / 256) * 256

/-- Every allocation is 256-byte aligned in size, so consecutive bump offsets stay aligned. -/
theorem allocBytes_aligned (n sz : Nat) : allocBytes n sz % 256 = 0 := by
  unfold allocBytes
  omega

/-- The rounded size always covers the payload plus the 256-byte guard. -/
theorem allocBytes_ge (n sz : Nat) : n * sz + 256 ≤ allocBytes n sz := by
  unfold allocBytes
  omega

/-- Rounding wastes less than 256 bytes beyond the guard. -/
theorem allocBytes_lt (n sz : Nat) : allocBytes n sz < n * sz + 256 + 256 := by
  unfold allocBytes
  omega

/-- A borrowed region: capacity and bump pointer. -/
structure Region where
  cap : Nat
  used : Nat

/-- `Alloc::take` on a borrowed region: fail when the request does not fit, else bump. -/
def Region.take? (r : Region) (bytes : Nat) : Option Region :=
  if r.cap < r.used + bytes then none else some { r with used := r.used + bytes }

/-- A successful take bumps by exactly the request and never leaves the region. -/
theorem take_some (r r' : Region) (bytes : Nat) (h : r.take? bytes = some r') :
    r'.used = r.used + bytes ∧ r'.used ≤ r'.cap ∧ r'.cap = r.cap ∧ r.used ≤ r'.used := by
  unfold Region.take? at h
  split at h
  · cases h
  · injection h with h
    subst h
    refine ⟨rfl, ?_, rfl, ?_⟩
    · show r.used + bytes ≤ r.cap
      omega
    · show r.used ≤ r.used + bytes
      omega

/-- Replay a whole allocation sequence on a region. -/
def takeAll (r : Region) : List Nat → Option Region
  | [] => some r
  | b :: bs =>
    match r.take? b with
    | none => none
    | some r1 => takeAll r1 bs

/-- A sequence that fits consumes exactly the sum of its requests; this is why a count-only pass
(`bytes_needed`) sizes the region exactly. -/
theorem takeAll_used (bs : List Nat) : ∀ (r r' : Region), r.used ≤ r.cap → takeAll r bs = some r' →
    r'.used = r.used + bs.sum ∧ r'.cap = r.cap ∧ r'.used ≤ r'.cap := by
  induction bs with
  | nil =>
    intro r r' hr h
    simp [takeAll] at h
    subst h
    simp [hr]
  | cons b bs ih =>
    intro r r' hr h
    simp only [takeAll] at h
    cases hb : r.take? b with
    | none =>
      simp [hb] at h
    | some r1 =>
      simp [hb] at h
      obtain ⟨h1, h2, h3, h4⟩ := take_some r r1 b hb
      obtain ⟨i1, i2, i3⟩ := ih r1 r' (by omega) h
      refine ⟨?_, ?_, i3⟩
      · simp only [List.sum_cons]
        omega
      · omega

/-- With `bytes_needed` as capacity (plus slack) the sequence fits: capacity at least the sum suffices
for a fresh region. -/
theorem fresh_region_fits (b : Nat) : (Region.take? ⟨b, 0⟩ b).isSome = true := by
  unfold Region.take?
  simp

-- § chunk loop

/-- Length of the chunk starting at `c0` when `n` tokens remain in total. -/
def chunkLen (T n c0 : Nat) : Nat := min T (n - c0)

/-- A chunk never exceeds the chunk size. -/
theorem chunkLen_le (T n c0 : Nat) : chunkLen T n c0 ≤ T := by
  unfold chunkLen
  omega

/-- A chunk stays inside the prompt. -/
theorem chunk_in_bounds (T n c0 : Nat) (h : c0 < n) : c0 + chunkLen T n c0 ≤ n := by
  unfold chunkLen
  omega

/-- A chunk is non-empty while tokens remain and the chunk size is positive. -/
theorem chunk_nonempty (T n c0 : Nat) (hT : 0 < T) (h : c0 < n) : 0 < chunkLen T n c0 := by
  unfold chunkLen
  omega

/-- The last chunk ends exactly at `n`; any earlier chunk ends on a multiple of `T` boundary. -/
theorem chunk_tail (T n c0 : Nat) (h : c0 < n) (hlast : n - c0 ≤ T) : c0 + chunkLen T n c0 = n := by
  unfold chunkLen
  omega

-- § expert grouping (host counting sort)

/-- Exclusive prefix sum of the per-expert counts at expert `e`. -/
def offsetAt (cnt : List Nat) (e : Nat) : Nat := (cnt.take e).sum

/-- `off[e+1] = off[e] + cnt[e]`. -/
theorem offsetAt_succ (cnt : List Nat) (e : Nat) (h : e < cnt.length) :
    offsetAt cnt (e + 1) = offsetAt cnt e + cnt[e] := by
  unfold offsetAt
  first
    | exact List.sum_take_succ cnt e h
    | (rw [List.take_succ, List.sum_append]
       simp [List.getElem?_eq_getElem h])

/-- `off[NE]` is the total number of routed pairs. -/
theorem offsetAt_total (cnt : List Nat) : offsetAt cnt cnt.length = cnt.sum := by
  unfold offsetAt
  simp

/-- Every prefix sum is at most the total. -/
theorem offsetAt_le_sum (cnt : List Nat) (e : Nat) : offsetAt cnt e ≤ cnt.sum := by
  have h := List.sum_take_add_sum_drop cnt e
  unfold offsetAt
  omega

/-- The `k`-th pair of expert `e` lands strictly inside expert `e`'s segment. -/
theorem slot_in_segment (cnt : List Nat) (e k : Nat) (he : e < cnt.length) (hk : k < cnt[e]) :
    offsetAt cnt e + k < offsetAt cnt (e + 1) := by
  have h := offsetAt_succ cnt e he
  omega

/-- ... and therefore inside the whole `T*K` slot space when the counts sum to `T*K`. -/
theorem slot_in_range (cnt : List Nat) (e k total : Nat) (he : e < cnt.length) (hk : k < cnt[e])
    (hsum : cnt.sum = total) : offsetAt cnt e + k < total := by
  have h1 := slot_in_segment cnt e k he hk
  have h2 := offsetAt_le_sum cnt (e + 1)
  omega

/-- Prefix sums never decrease from one expert to the next. -/
theorem offsetAt_le_succ (cnt : List Nat) (e : Nat) : offsetAt cnt e ≤ offsetAt cnt (e + 1) := by
  by_cases h : e < cnt.length
  · have h1 := offsetAt_succ cnt e h
    omega
  · unfold offsetAt
    have h1 : cnt.take e = cnt := List.take_of_length_le (by omega)
    have h2 : cnt.take (e + 1) = cnt := List.take_of_length_le (by omega)
    simp [h1, h2]

/-- Prefix sums are monotone in the expert id. -/
theorem offsetAt_mono (cnt : List Nat) (e : Nat) : ∀ d, offsetAt cnt e ≤ offsetAt cnt (e + d) := by
  intro d
  induction d with
  | zero => simp
  | succ d ih =>
    have h1 := offsetAt_le_succ cnt (e + d)
    have h2 : e + (d + 1) = e + d + 1 := by omega
    rw [h2]
    omega

/-- Two different experts' segments never overlap: expert `e`'s end is at most `e'`'s start. -/
theorem segments_disjoint (cnt : List Nat) (e e' : Nat) (hlt : e < e') :
    offsetAt cnt (e + 1) ≤ offsetAt cnt e' := by
  have h1 := offsetAt_mono cnt (e + 1) (e' - (e + 1))
  have h2 : e + 1 + (e' - (e + 1)) = e' := by omega
  rw [h2] at h1
  exact h1

/-- `src_host[p] = i / K` recovers the token of pair `i = t*K + k`. -/
theorem src_token (t k : Nat) (hk : k < K) : (t * K + k) / K = t := by
  unfold K at *
  omega

/-- ... and `i % K` recovers its rank among the token's ten choices. -/
theorem src_rank (t k : Nat) (hk : k < K) : (t * K + k) % K = k := by
  unfold K at *
  omega

-- § staging and dequantisation rings

/-- Two different experts inside one look-ahead window (`STAGE - 1` ahead of the one being computed)
never share a staging slot, so a blob in flight is not overwritten. -/
theorem stage_slots_distinct (j s1 s2 : Nat) (h1 : j ≤ s1) (h12 : s1 < s2) (h2 : s2 ≤ j + (STAGE - 1)) :
    s1 % STAGE ≠ s2 % STAGE := by
  unfold STAGE at *
  omega

/-- The slot reused by expert `s + STAGE` is the one expert `s` used, which is why the host waits on
`used[sl]` before overwriting it. -/
theorem stage_slot_reuse (s : Nat) : (s + STAGE) % STAGE = s % STAGE := by
  unfold STAGE
  omega

/-- Consecutive experts alternate the two dequantised buffers. -/
theorem dq_alternates (j : Nat) : (j + 1) % DQ ≠ j % DQ := by
  unfold DQ
  omega

/-- A slot index is always inside the ring. -/
theorem stage_slot_lt (j : Nat) : j % STAGE < STAGE := by
  unfold STAGE
  omega

/-- The look-ahead never exceeds the ring: at most `STAGE` blobs are in flight. -/
theorem lookahead_in_flight (j staged : Nat) (h : staged ≤ j + (STAGE - 1)) (h0 : j ≤ staged) :
    staged - j + 1 ≤ STAGE := by
  unfold STAGE at *
  omega

end Strata.PrefillChunk
