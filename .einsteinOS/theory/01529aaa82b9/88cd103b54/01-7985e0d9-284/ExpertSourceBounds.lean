/-!
# MODULE: expert_source_bounds
The expert source adapter: blob lookup bounds, fixed artifact geometry, and the hit/miss job accounting that
splits one layer's routed experts between GPU (hits) and CPU pool (jobs).

## Source semantics
- `blob(layer, expert)` returns a pointer only when both axes are in range and the flat index is below `blobs`.
  The expert axis is checked separately because the flat index alone aliases the next layer.
- The artifact geometry is fixed: `experts.bin` holds exactly `48 * 512` blobs of 1,382,400 bytes.
- In the single-token dispatch, `njobs` indexes the job array and `i` the output; hits write zero rows and add
  no job, so `njobs = k - hits`.
- In a verify window the PCIe share is `(nmiss * pcie_num) >> 8`, never more than the missed experts.

## Form template identifiers
- `blobLookup`      <- `FileExpertSource::blob` / `ArenaExpertSource::blob` validity condition
- `hits`, `misses`  <- the `is_hit` / `host_res` split in `expert_pool_dispatch`
- `pcieShare`       <- `m = (nmiss * d.pcie_num) >> 8` in `expert_pool_dispatch_multi`
-/

namespace Strata.ExpertSource

/-- Blob index validity (layer unbounded above here: the flat index bound covers it). -/
def blobValid (ne blobs layer expert : Nat) : Prop :=
  expert < ne ∧ layer * ne + expert < blobs

/-- Number of hit entries in a per-index hit vector. -/
def hits : List Bool → Nat
  | [] => 0
  | b :: t => (if b then 1 else 0) + hits t

/-- Number of miss entries (the jobs handed to the CPU pool). -/
def misses : List Bool → Nat
  | [] => 0
  | b :: t => (if b then 0 else 1) + misses t

/-- PCIe share of the distinct missed experts. -/
def pcieShare (nmiss pcieNum : Nat) : Nat := (nmiss * pcieNum) / 256

/-- A valid blob request always has its expert id below the per-layer width. -/
theorem blobValid_expert_lt (ne blobs layer expert : Nat) (h : blobValid ne blobs layer expert) :
    expert < ne := h.1

/-- An id one past the end of a layer is refused even though its flat index may be in range. -/
theorem blobValid_rejects_overrun (ne blobs layer : Nat) : ¬ blobValid ne blobs layer ne := by
  intro h
  exact absurd h.1 (Nat.lt_irrefl ne)

/-- Flat index of `(0, ne)` collides with `(1, 0)`: the reason the axes are checked separately. -/
theorem flat_collision (ne : Nat) : 0 * ne + ne = 1 * ne + 0 := by
  simp

/-- Every request of a valid geometry stays inside the pack. -/
theorem blobValid_in_pack (nl ne layer expert : Nat) (hl : layer < nl) (he : expert < ne) :
    layer * ne + expert < nl * ne := by
  have h := Nat.mul_le_mul_right ne (Nat.succ_le_of_lt hl)
  simp only [Nat.succ_mul] at h
  omega

/-- Hits and misses partition the routed experts. -/
theorem hits_add_misses (l : List Bool) : hits l + misses l = l.length := by
  induction l with
  | nil => simp [hits, misses]
  | cons b t ih =>
    cases b <;> simp [hits, misses] <;> omega

/-- The job count never exceeds the routed count `k`. -/
theorem misses_le (l : List Bool) : misses l ≤ l.length := by
  have h := hits_add_misses l
  omega

/-- With no hits every routed expert becomes a job (the Phase-2 `h = 0` mode). -/
theorem misses_all_false (n : Nat) : misses (List.replicate n false) = n := by
  induction n with
  | zero => simp [misses]
  | succ n ih =>
    simp [List.replicate_succ, misses, ih] <;> omega

/-- The PCIe share never exceeds the number of missed experts when `pcie_num ≤ 256`. -/
theorem pcieShare_le (nmiss pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare nmiss pcieNum ≤ nmiss := by
  unfold pcieShare
  have h2 : nmiss * pcieNum ≤ nmiss * 256 := Nat.mul_le_mul_left nmiss h
  omega

/-- The last `m` ranks of the misses are the PCIe ones: exactly `m` ranks lie in `[nmiss - m, nmiss)`. -/
theorem pcie_tail_count (nmiss m : Nat) (h : m ≤ nmiss) : nmiss - (nmiss - m) = m := by
  omega

/-- The artifact holds exactly `48 * 512` blobs of 1,382,400 bytes: 33,973,862,400 bytes. -/
theorem pack_bytes : 48 * 512 * 1382400 = 33973862400 := by
  decide

/-- The single-token kernel is compiled for a 2560-wide activation; any other width is refused. -/
def widthOk (nEmbd : Nat) : Bool := decide (nEmbd = 2560)

/-- Width acceptance is exact. -/
theorem widthOk_iff (n : Nat) : widthOk n = true ↔ n = 2560 := by
  simp [widthOk]

end Strata.ExpertSource
