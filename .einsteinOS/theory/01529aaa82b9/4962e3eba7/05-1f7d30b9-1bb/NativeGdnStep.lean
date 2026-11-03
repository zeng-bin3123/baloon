/-!
# MODULE: native_gdn_step
Warp-per-column GDN recurrence step adapted from llama.cpp's gated delta net.  Each warp owns one
`(head, column)` pair and keeps its four state rows per lane in registers.  This file proves the
register-shard / column decomposition, the state-address bounds and the span-overlap predicate used by
the host validation.

## Source semantics
* Row `i = r*32 + lane` for `r < 4`, `lane < 32`: the four shards of a lane partition the 128 rows.
* Column `col = blockIdx.z * blockDim.y + threadIdx.y` with grid z = 32 and blockDim.y = 4.
* The host rejects unaligned spans and any overlap between the state/output writes and the inputs.

## Form template identifiers
- `step` -> `native_step`; `s_shard` -> `state_shard`; `warp_sum` -> `lane_reduce`
- `valid_span` -> `span_valid`; `overlap` -> `span_overlap`; `q_head` -> `key_head_of`
-/

namespace Strata.NativeGdn

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § register shard decomposition

/-- Every (shard, lane) pair addresses a row inside the 128 rows. -/
theorem shard_row_lt {r lane : Nat} (hr : r < 4) (hl : lane < 32) : r * 32 + lane < 128 := by omega

/-- Distinct (shard, lane) pairs address distinct rows. -/
theorem shard_row_inj {r l r' l' : Nat} (hl : l < 32) (hl' : l' < 32)
    (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by omega

/-- Every row is addressed by exactly one (shard, lane) pair. -/
theorem shard_row_surj {i : Nat} (hi : i < 128) :
    i / 32 < 4 ∧ i % 32 < 32 ∧ (i / 32) * 32 + i % 32 = i := by omega

-- § column decomposition

/-- The column index computed from `(blockIdx.z, threadIdx.y)` is below 128. -/
theorem column_lt {z y : Nat} (hz : z < 32) (hy : y < 4) : z * 4 + y < 128 := by omega

/-- Distinct `(blockIdx.z, threadIdx.y)` pairs address distinct columns. -/
theorem column_inj {z y z' y' : Nat} (hy : y < 4) (hy' : y' < 4)
    (h : z * 4 + y = z' * 4 + y') : z = z' ∧ y = y' := by omega

/-- Every column is owned by exactly one `(blockIdx.z, threadIdx.y)` pair. -/
theorem column_surj {c : Nat} (hc : c < 128) :
    c / 4 < 32 ∧ c % 4 < 4 ∧ (c / 4) * 4 + c % 4 = c := by omega

-- § address bounds

/-- State element `(i*h_v + head)*128 + col` lies inside the `128 x h_v x 128` buffer. -/
theorem state_index_lt {i head col h_v : Nat} (hi : i < 128) (hh : head < h_v) (hc : col < 128) :
    (i * h_v + head) * 128 + col < 128 * h_v * 128 :=
  lin2_lt (lin2_lt hi hh) hc

/-- Key/query element `q_head*128 + i` lies inside the `128 * h_k` buffer. -/
theorem qk_index_lt {q_head i h_k : Nat} (hq : q_head < h_k) (hi : i < 128) :
    q_head * 128 + i < h_k * 128 := lin2_lt hq hi

/-- Output/value element `head*128 + col` lies inside the `128 * h_v` buffer. -/
theorem out_index_lt {head col h_v : Nat} (hh : head < h_v) (hc : col < 128) :
    head * 128 + col < h_v * 128 := lin2_lt hh hc

/-- The modulo head pairing is a valid key/query head. -/
theorem key_head_lt {head h_k : Nat} (hk : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ hk

/-- Every value head is mapped onto a key head, and heads sharing one differ by multiples of `h_k`. -/
theorem key_head_period {head h_k : Nat} (hk : 0 < h_k) : (head + h_k) % h_k = head % h_k := by
  simp

-- § span overlap predicate (host validation)

/-- Half-open byte ranges `[ap, ap+an)` and `[bp, bp+bn)` are flagged as overlapping. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap predicate is symmetric. -/
theorem overlap_comm {ap an bp bn : Nat} : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap
  omega

/-- Ranges laid end to end are never flagged. -/
theorem not_overlap_of_adjacent {ap an bp bn : Nat} (h : ap + an ≤ bp) : ¬ overlap ap an bp bn := by
  unfold overlap
  omega

/-- A nonempty range always overlaps itself, so aliasing the state with the output is rejected. -/
theorem overlap_self {p n : Nat} (hn : 0 < n) : overlap p n p n := by
  unfold overlap
  omega

/-- The predicate also flags an empty range strictly inside a nonempty one (zero-byte spans are not exempt). -/
theorem overlap_zero_length_inside : overlap 0 10 5 0 := by
  unfold overlap
  omega

-- § byte sizes

/-- Per-call byte sizes: state `S*S*h_v` floats, output/value `S*h_v` floats, q/k `S*h_k` floats. -/
theorem state_bytes_eq (h_v : Nat) : 128 * 128 * h_v * 4 = 65536 * h_v := by omega

end Strata.NativeGdn
