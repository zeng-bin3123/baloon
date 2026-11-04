/-!
# MODULE: qsa_decode_attn_chunks
Split-K decode attention over the selected cells (`src/kernels/cuda/qsa_decode_attn.cu`):
cells are cut into chunks of 64, each (chunk, kv head) block writes a partial accumulator,
and a merge kernel combines them with log-sum-exp weights.

## Source semantics
* `n_chunks = ceil(cap / 64)`; chunk `c` covers cells `[64c, 64c + n_here)` with
  `n_here = min 64 (n_ids - 64c)`; chunks past the selection are empty and publish `m = -FLT_MAX`.
* The 24 query heads are grouped 12 per KV head: `h = kvh*12 + hl`.
* Scratch is `[acc: chunks*nHead*256][m: chunks*nHead][l: chunks*nHead]` plus 64 spare floats.

## Form template identifiers
* `nChunks`      <- `n_chunks`
* `nHere`        <- `n_here`
* `scratchFloats`<- `qsa_decode_attn_scratch_floats`
* `slotOf`       <- `kvh * n_chunks + chunk`
-/

namespace QsaDecodeAttnChunks

-- § chunking

/-- Number of 64-cell chunks for a capacity. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- The chunks cover the whole capacity. -/
theorem nChunks_cover (cap : Nat) : cap ≤ nChunks cap * 64 := by
  unfold nChunks
  omega

/-- ... and not a whole chunk more. -/
theorem nChunks_tight (cap : Nat) : nChunks cap * 64 < cap + 64 := by
  unfold nChunks
  omega

/-- Cells handled by chunk `c` for a selection of `n` ids. -/
def nHere (n c : Nat) : Nat := min 64 (n - 64 * c)

/-- Cells covered by the first `m` chunks. -/
def covered (n : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => covered n m + nHere n m

/-- The first `m` chunks cover exactly `min n (64 m)` cells. -/
theorem covered_eq (n m : Nat) : covered n m = min n (64 * m) := by
  induction m with
  | zero => simp [covered]
  | succ m ih =>
    simp only [covered, nHere, ih]
    omega

/-- All chunks together cover each selected cell exactly once. -/
theorem covered_all (n cap : Nat) (h : n ≤ cap) : covered n (nChunks cap) = n := by
  rw [covered_eq]
  have := nChunks_cover cap
  omega

/-- A chunk is empty exactly when it starts at or past the selection. -/
theorem nHere_zero_iff (n c : Nat) : nHere n c = 0 ↔ n ≤ 64 * c := by
  unfold nHere
  omega

/-- The number of non-empty chunks never exceeds the launched chunks. -/
theorem nonempty_chunks_le (n cap : Nat) (h : n ≤ cap) : (n + 63) / 64 ≤ nChunks cap := by
  unfold nChunks
  omega

-- § slots and heads

/-- Partial-result slot of a (kv head, chunk) block. -/
def slotOf (kvh nCh chunk : Nat) : Nat := kvh * nCh + chunk

/-- Slots stay inside `nKvHeads * nChunks`. -/
theorem slotOf_lt (kvh nCh chunk nKvHeads : Nat) (hk : kvh < nKvHeads) (hc : chunk < nCh) :
    slotOf kvh nCh chunk < nKvHeads * nCh := by
  unfold slotOf
  have h0 : (kvh + 1) * nCh ≤ nKvHeads * nCh := Nat.mul_le_mul_right _ hk
  have h0' : (kvh + 1) * nCh = kvh * nCh + nCh := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- A global query head splits into its KV head and local index. -/
theorem head_split (h : Nat) : h / 12 * 12 + h % 12 = h := by
  omega

/-- The local head index is below the group size. -/
theorem head_local_lt (h : Nat) : h % 12 < 12 := by
  omega

/-- The KV head of every query head is in range. -/
theorem kvh_lt (h nKvHeads : Nat) (hh : h < 12 * nKvHeads) : h / 12 < nKvHeads := by
  omega

-- § scratch layout

/-- Floats of scratch: acc rows of 256 plus m and l, plus 64 spare. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * 258 + 64

/-- The three regions `acc | m | l` fit inside the allocation without overlapping. -/
theorem scratch_layout (cap nHead : Nat) :
    nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead
      ≤ scratchFloats cap nHead := by
  unfold scratchFloats
  omega

end QsaDecodeAttnChunks
