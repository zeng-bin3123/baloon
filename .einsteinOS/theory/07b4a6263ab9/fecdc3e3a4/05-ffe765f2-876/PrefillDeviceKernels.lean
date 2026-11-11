/-!
# MODULE: prefill_device_kernels
SOURCES: src/prefill/kernels.cu
Index arithmetic and selection logic of the prefill CUDA kernels: launch-grid coverage (`blocks_for`),
hyper-connection index decoding (`gr_write`), the warp top-10 router's lane/slot decomposition and its
tie-breaking order, the Strata blob dequantizer's addressing (which must agree with the CPU blob layout),
and the paged KV append addressing.

## Source semantics
* `blocks_for(n, t) = ⌈n / t⌉` blocks of t threads cover all n elements.
* `gr_write`: i ∈ [0, T·D), D = N·HC = 10240; t = i / D, c = (i % D) / N < HC, d = i % N.
* Router: lane ℓ < 32 holds experts ℓ + 32·i (i < 16) of 512; expert ex is owned by lane ex & 31 at
  slot ex / 32.  The butterfly keeps (value, index) maximal for "higher value, then lower index", a
  strict total order on distinct experts, so all lanes agree on one winner per rank.  ids are written
  at t·10 + rank.
* Blob dequant: one thread per code byte; 1280·640 gate/up bytes then 2560·160 down bytes; scale index
  row·40 + byte·4/64 (gate/up) and row·10 + byte·4/64 (down); weight = (code − 1)·d.
* KV append: row = (page·2 + kvh)·page_size + pos % page_size; distinct (page, kv head, slot) give
  distinct rows; 8 z-blocks = 4 groups × {K, V}, group g covers dims 64g..64g+63 of 256.

## Form template identifiers
blocksFor → gridBlocks, better → routerOrder, kvRow → pagedKvRow, guScaleIdx/dScaleIdx →
dequantScaleIndex, O_D_CODES/O_GU_SC/O_D_SC → blobRegionOffsets
-/

namespace Strata.Prefill.Kernels

-- § grid coverage

/-- `blocks_for(n, t)`. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- The launched threads cover every element. -/
theorem blocksFor_covers (n t : Nat) (ht : 0 < t) : n ≤ t * blocksFor n t := by
  unfold blocksFor
  have h1 := Nat.div_add_mod (n + t - 1) t
  have h2 := Nat.mod_lt (n + t - 1) ht
  generalize (n + t - 1) / t = q at h1
  generalize (n + t - 1) % t = m at h1 h2
  omega

/-- …and do not launch a whole extra block. -/
theorem blocksFor_tight (n t : Nat) (ht : 0 < t) : t * blocksFor n t < n + t := by
  unfold blocksFor
  have h1 := Nat.div_add_mod (n + t - 1) t
  generalize (n + t - 1) / t = q at h1
  generalize (n + t - 1) % t = m at h1
  omega

/-- The router launches ⌈T/8⌉ blocks of 8 warps, one warp per token: every token is routed. -/
theorem route_grid_covers (T : Nat) : T ≤ 8 * ((T + 7) / 8) := by omega

-- § hyper-connection indexing

def N : Nat := 2560
def HC : Nat := 4
def D : Nat := N * HC

/-- `gr_write`: the stream index c of any flat index is one of the four streams. -/
theorem gr_write_stream (i : Nat) : (i % D) / N < HC := by
  show (i % 10240) / 2560 < 4; omega

/-- `gr_write` decodes i into (t, c, d) and that decomposition is exact. -/
theorem gr_write_decode (i : Nat) : (i / D) * D + ((i % D) / N) * N + i % N = i := by
  show (i / 10240) * 10240 + ((i % 10240) / 2560) * 2560 + i % 2560 = i; omega

-- § router

/-- Lane ℓ's slot i holds expert ℓ + 32·i; owner lane and slot are recovered by & 31 and / 32. -/
theorem router_slot (lane i : Nat) (hl : lane < 32) (hi : i < 16) :
    lane + i * 32 < 512 ∧ (lane + i * 32) % 32 = lane ∧ (lane + i * 32) / 32 = i := by omega

/-- Every one of the 512 experts is held by exactly one (lane, slot). -/
theorem router_owner (ex : Nat) (h : ex < 512) : ex % 32 < 32 ∧ ex / 32 < 16 ∧ ex % 32 + ex / 32 * 32 = ex := by
  omega

/-- The butterfly's preference: higher probability, then lower expert id. -/
def better (a b : Int × Nat) : Prop := a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

/-- On distinct experts the order is total: one of any two candidates wins. -/
theorem better_total (a b : Int × Nat) (h : a.2 ≠ b.2) : better a b ∨ better b a := by
  unfold better; omega

/-- …and asymmetric: two lanes never each believe the other's candidate is better. -/
theorem better_asymm (a b : Int × Nat) : ¬ (better a b ∧ better b a) := by
  unfold better; omega

/-- …and transitive, so the reduction's result does not depend on the shuffle order. -/
theorem better_trans (a b c : Int × Nat) (h1 : better a b) (h2 : better b c) : better a c := by
  unfold better at *; omega

/-- ids and weights of token t, rank r land in [0, T·10). -/
theorem route_out_index (t r T : Nat) (ht : t < T) (hr : r < 10) : t * 10 + r < T * 10 := by omega

-- § Strata blob dequant addressing

def O_D_CODES : Nat := 1280 * 640
def O_GU_SC : Nat := O_D_CODES + 2560 * 160
def O_D_SC : Nat := O_GU_SC + 1280 * 40 * 2

/-- The device offsets equal the CPU blob layout's (819200, 1228800, 1331200) and end at BLOB. -/
theorem offsets_match_cpu :
    O_D_CODES = 819200 ∧ O_GU_SC = 1228800 ∧ O_D_SC = 1331200 ∧ O_D_SC + 2560 * 10 * 2 = 1382400 := by
  decide

/-- One thread per code byte: the launch size. -/
theorem dequant_threads : 1280 * 640 + 2560 * 160 = 1228800 := by decide

/-- Gate/up thread i: row < 1280, scale index inside the 1280·40 table, outputs inside 1280·2560. -/
theorem gu_thread_bounds (i k : Nat) (hi : i < 1280 * 640) (hk : k < 4) :
    i / 640 < 1280 ∧ (i / 640) * 40 + (i % 640) * 4 / 64 < 1280 * 40 ∧
      (i / 640) * 2560 + (i % 640) * 4 + k < 1280 * 2560 := by omega

/-- Down thread j: row < 2560, scale index inside 2560·10, outputs inside 2560·640. -/
theorem d_thread_bounds (j k : Nat) (hj : j < 2560 * 160) (hk : k < 4) :
    j / 160 < 2560 ∧ (j / 160) * 10 + (j % 160) * 4 / 64 < 2560 * 10 ∧
      (j / 160) * 640 + (j % 160) * 4 + k < 2560 * 640 := by omega

/-- Each code byte's four weights share one 64-weight block scale. -/
theorem byte_in_one_block (byte k : Nat) (hk : k < 4) : (byte * 4 + k) / 64 = byte * 4 / 64 := by omega

/-- Dequantized weights are (code − 1)·d with code − 1 ∈ {−1, 0, 1, 2}. -/
theorem dequant_offset (c : Nat) : -1 ≤ ((c % 4 : Nat) : Int) - 1 ∧ ((c % 4 : Nat) : Int) - 1 ≤ 2 := by
  have : c % 4 < 4 := Nat.mod_lt _ (by decide)
  omega

-- § paged KV append

/-- The pool row of (page, kv head, slot). -/
def kvRow (page kvh slot ps : Nat) : Nat := (page * 2 + kvh) * ps + slot

/-- A row decodes back to (page·2 + kvh, slot). -/
theorem kvRow_decode (page kvh slot ps : Nat) (hs : slot < ps) :
    kvRow page kvh slot ps / ps = page * 2 + kvh ∧ kvRow page kvh slot ps % ps = slot := by
  have hps : 0 < ps := by omega
  unfold kvRow
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hps, Nat.div_eq_of_lt hs, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hs]

/-- Distinct (page, kv head, slot) never share a row: appends never overwrite each other. -/
theorem kvRow_injective (p1 h1 s1 p2 h2 s2 ps : Nat) (hh1 : h1 < 2) (hh2 : h2 < 2)
    (hs1 : s1 < ps) (hs2 : s2 < ps) (heq : kvRow p1 h1 s1 ps = kvRow p2 h2 s2 ps) :
    p1 = p2 ∧ h1 = h2 ∧ s1 = s2 := by
  have d1 := kvRow_decode p1 h1 s1 ps hs1
  have d2 := kvRow_decode p2 h2 s2 ps hs2
  rw [heq] at d1
  have hq : p1 * 2 + h1 = p2 * 2 + h2 := d1.1.symm.trans d2.1
  have hs : s1 = s2 := d1.2.symm.trans d2.2
  omega

/-- The 8 z-blocks split as group g = z/2 < 4 and K/V = z odd; dims 64g + tid stay below 256. -/
theorem kv_block_dims (z tid : Nat) (hz : z < 8) (ht : tid < 64) : z / 2 < 4 ∧ (z / 2) * 64 + tid < 256 := by
  omega

/-- The int8 scale of group g of row `row` is at row·4 + g, distinct for distinct (row, g). -/
theorem kv_scale_index (r1 g1 r2 g2 : Nat) (hg1 : g1 < 4) (hg2 : g2 < 4) (h : r1 * 4 + g1 = r2 * 4 + g2) :
    r1 = r2 ∧ g1 = g2 := by omega

end Strata.Prefill.Kernels
