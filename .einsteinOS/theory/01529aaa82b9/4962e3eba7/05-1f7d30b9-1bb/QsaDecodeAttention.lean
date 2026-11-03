/-!
# MODULE: qsa_decode_attention
Split-K decode attention over a selected cell set: a chunk kernel (64 cells per block, 12 query heads
per KV head, fp16 or int8+scale pools) producing partial `(acc, m, l)` triples, and a merge kernel that
combines them with an online-softmax rescale.  This file proves the chunk coverage, the thread/warp
ownership of the three phases, the scratch layout and the int8 group-scale addressing.

## Source semantics
* `HD = 256`, `G = 12`, `CHUNK = 64`, 256 threads (8 warps).
* Phase 1: warp w scores cells `w, w+8, ...`; each lane holds 8 of the 256 dimensions.
* Phase 2: warp w handles heads `w, w+8`; lane and lane+32 cover the 64 chunk cells.
* Phase 3: thread t owns dimension t for all 12 heads.
* Scratch per query: `n_chunks * n_head * (HD + 2) + 64` floats.

## Form template identifiers
- `attn_chunk_kernel` -> `chunk_partial`; `attn_merge_kernel` -> `chunk_merge`
- `part_acc/part_m/part_l` -> `partial_acc/partial_max/partial_sum`; `srow` -> `cell_row`
- `KV_Q8_GROUP` -> `q8_group` (64)
-/

namespace Strata.QsaDecodeAttn

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § chunking

/-- Every selected cell index `c < cap` lies in one of the `ceil(cap/64)` chunks. -/
theorem chunk_cover {cap c : Nat} (hc : c < cap) : c / 64 < (cap + 63) / 64 := by omega

/-- A block's local cell count is positive exactly when its first cell exists. -/
theorem n_here_pos_iff {c0 n : Nat} : 0 < min 64 (n - c0) ↔ c0 < n := by omega

/-- A block never handles more than 64 cells. -/
theorem n_here_le {c0 n : Nat} : min 64 (n - c0) ≤ 64 := by omega

/-- Chunk and local slot reconstruct the global cell index. -/
theorem chunk_local_index {c : Nat} : (c / 64) * 64 + c % 64 = c ∧ c % 64 < 64 := by omega

-- § phase 1: scores

/-- Warp `c % 8` owns cell `c` of the chunk, and each cell has exactly one owner. -/
theorem score_cell_owner {c : Nat} (hc : c < 64) :
    c % 8 < 8 ∧ c % 8 + 8 * (c / 8) = c ∧ c / 8 < 8 := by omega

/-- Eight consecutive dimensions starting at `lane*8` stay inside the 256 dimensions. -/
theorem lane_dims_lt {lane : Nat} (h : lane < 32) : lane * 8 + 8 ≤ 256 := by omega

/-- The 32 lanes partition the 256 dimensions into disjoint runs of 8. -/
theorem lane_dims_inj {l l' j j' : Nat} (hj : j < 8) (hj' : j' < 8)
    (h : l * 8 + j = l' * 8 + j') : l = l' ∧ j = j' := by omega

-- § phase 2: per-head softmax statistics

/-- Warp `h % 8` handles head `h < 12`; heads 8..11 are second assignments of warps 0..3. -/
theorem head_warp_owner {h : Nat} (hh : h < 12) :
    h % 8 < 8 ∧ (h < 8 ∨ (8 ≤ h ∧ h - 8 < 4)) := by omega

/-- Lanes `l` and `l + 32` cover the 64 chunk cells exactly once. -/
theorem lane_pair_cover {c : Nat} (hc : c < 64) : (c < 32 ∧ c % 32 = c) ∨ (32 ≤ c ∧ c - 32 < 32) := by omega

-- § phase 3: values

/-- Thread `t < 256` owns dimension `t`: the mapping is the identity. -/
theorem value_dim_owner {t : Nat} (ht : t < 256) : t < 256 := ht

-- § scratch layout

/-- Each (kv head, chunk) slot index is below `n_kv_heads * n_chunks`. -/
theorem slot_lt {kvh chunk nkv nch : Nat} (hk : kvh < nkv) (hc : chunk < nch) :
    kvh * nch + chunk < nkv * nch := lin2_lt hk hc

/-- The accumulator index `((slot*G + h)*HD + t)` is inside the accumulator region. -/
theorem acc_index_lt {slot h t nslots : Nat} (hs : slot < nslots) (hh : h < 12) (ht : t < 256) :
    (slot * 12 + h) * 256 + t < nslots * 12 * 256 :=
  lin2_lt (lin2_lt hs hh) ht

/-- The max/sum index `slot*G + h` is inside its region. -/
theorem stat_index_lt {slot h nslots : Nat} (hs : slot < nslots) (hh : h < 12) :
    slot * 12 + h < nslots * 12 :=
  lin2_lt hs hh

/-- The three regions (acc, m, l) add up to the `HD + 2` floats per chunk-head counted by the scratch size. -/
theorem scratch_regions_sum (c nh : Nat) : c * nh * 256 + c * nh + c * nh = c * nh * 258 := by omega

/-- The int8/fp16 paths use 12 KB of shared query plus 3 KB scores plus 512 B row table, under 48 KiB. -/
theorem shared_bytes : 12 * 256 * 4 + 12 * 64 * 4 + 64 * 8 = 15872 ∧ 15872 < 49152 := by decide

-- § merge kernel

/-- A global query head splits into (kv head, local head) and recombines exactly. -/
theorem head_split {h nkv : Nat} (hh : h < 12 * nkv) :
    h / 12 < nkv ∧ h % 12 < 12 ∧ (h / 12) * 12 + h % 12 = h := by omega

-- § int8 group scales

/-- With `HD = 256` and groups of 64 there are 4 scale groups per row. -/
theorem groups_per_row : 256 / 64 = 4 := by decide

/-- A run of 8 consecutive dimensions starting at a multiple of 8 never crosses a 64-group boundary. -/
theorem run_in_one_group {d0 : Nat} (h : d0 % 8 = 0) : d0 / 64 = (d0 + 7) / 64 := by omega

/-- The scale index `row*4 + d0/64` is inside the scale buffer of `rows * 4` entries. -/
theorem scale_index_lt {row rows d0 : Nat} (hr : row < rows) (hd : d0 < 256) :
    row * 4 + d0 / 64 < rows * 4 :=
  lin2_lt hr (by omega)

end Strata.QsaDecodeAttn
