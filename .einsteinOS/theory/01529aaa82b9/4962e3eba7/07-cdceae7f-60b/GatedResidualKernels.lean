/-!
# MODULE: gated_residual_kernels
Workspace layout, dispatch flags, and loop geometry of the gated residual / hyper-connection `gr_read`
and `gr_write` kernels (`gr.cu`).

## Source semantics
`gr_workspace_init` carves five regions (xn, xq, lq, gated, lo) from one base pointer with a single size
table; the regions must be consecutive, 16-byte aligned, and each at least as large as its payload. `gr_write`
computes `out[i] = R[i] + block_out[i % n] * w[i / n]`, so the stream/offset decomposition must be exact.
The native MMVF flag implies FP32 activations.

## Form template identifiers
- hc -> `hc`, n_embd -> `n`, hc_lr -> `lr`
- `align16` -> `(n+15) & ~15`
- `wsBytes` -> `GrWorkspace.bytes` / `gr_workspace_bytes`
-/

namespace Strata.GatedResidual

-- § alignment

def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never shrinks a size. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- The rounded size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Rounding adds fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- Rounding is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

-- § workspace table

def wsSizes (hc n lr : Nat) : List Nat := [hc * n * 4, hc * n * 2, lr * 2, hc * n * 4, lr * 4]
def wsAligned (hc n lr : Nat) : List Nat := (wsSizes hc n lr).map align16
def wsBytes (hc n lr : Nat) : Nat := (wsAligned hc n lr).sum
def wsOffset (hc n lr k : Nat) : Nat := ((wsAligned hc n lr).take k).sum

/-- The total is the sum of the five aligned regions. -/
theorem wsBytes_eq (hc n lr : Nat) :
    wsBytes hc n lr = align16 (hc * n * 4) + align16 (hc * n * 2) + align16 (lr * 2) +
      align16 (hc * n * 4) + align16 (lr * 4) := by
  simp [wsBytes, wsAligned, wsSizes, Nat.add_assoc]

/-- The workspace is at least as large as the sum of the raw payloads. -/
theorem wsBytes_ge (hc n lr : Nat) :
    hc * n * 4 + hc * n * 2 + lr * 2 + hc * n * 4 + lr * 4 ≤ wsBytes hc n lr := by
  rw [wsBytes_eq]
  have h1 := align16_ge (hc * n * 4)
  have h2 := align16_ge (hc * n * 2)
  have h3 := align16_ge (lr * 2)
  have h4 := align16_ge (lr * 4)
  omega

/-- For the real geometry the workspace is 104320 bytes. -/
theorem wsBytes_real : wsBytes 4 2560 320 = 104320 := by decide

/-- Region offsets for the real geometry: xn, xq, lq, gated, lo are consecutive. -/
theorem wsOffsets_real :
    wsOffset 4 2560 320 0 = 0 ∧ wsOffset 4 2560 320 1 = 40960 ∧ wsOffset 4 2560 320 2 = 61440 ∧
    wsOffset 4 2560 320 3 = 62080 ∧ wsOffset 4 2560 320 4 = 103040 ∧ wsOffset 4 2560 320 5 = 104320 := by
  decide

/-- Every real-geometry region offset is 16-byte aligned. -/
theorem wsOffsets_real_aligned :
    wsOffset 4 2560 320 1 % 16 = 0 ∧ wsOffset 4 2560 320 2 % 16 = 0 ∧
    wsOffset 4 2560 320 3 % 16 = 0 ∧ wsOffset 4 2560 320 4 % 16 = 0 := by
  decide

/-- The first version's xq sizing (hc_lr*2 bytes) was far too small for the real geometry. -/
theorem xq_old_sizing_too_small : 320 * 2 < 4 * 2560 * 2 := by decide

-- § gr_write decomposition

/-- Stream index of a flat element is below hc. -/
theorem write_stream_lt (n hc i : Nat) (h : i < hc * n) : i / n < hc := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  exact h

/-- A flat element index splits as `stream * n + offset`. -/
theorem write_decomp (n i : Nat) : (i / n) * n + i % n = i := by
  have h := Nat.div_add_mod i n
  rw [Nat.mul_comm] at h
  exact h

/-- The offset within a stream is below n_embd. -/
theorem write_offset_lt (n i : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt i hn

-- § grids

/-- A ceiling-divided grid covers every element. -/
theorem ceil_cover (n T : Nat) (hT : 0 < T) : n ≤ (n + T - 1) / T * T := by
  have h1 : (n + T - 1) / T * T + (n + T - 1) % T = n + T - 1 := by
    rw [Nat.mul_comm]; exact Nat.div_add_mod _ _
  have h2 := Nat.mod_lt (n + T - 1) hT
  omega

/-- The gate kernel launches exactly 1280 blocks at the real geometry. -/
theorem gate_blocks_real : (10240 + 8 - 1) / 8 = 1280 := by decide

/-- One-block-per-row down kernel: eight warps stride the reduction, covering each index once. -/
theorem down_stride_cover (i : Nat) (hi : i < 10240) :
    ∃ k w l, k < 40 ∧ w < 8 ∧ l < 32 ∧ i = k * 256 + w * 32 + l := by
  exact ⟨i / 256, i % 256 / 32, i % 32, by omega, by omega, by omega, by omega⟩

/-- Distinct (step, warp, lane) triples read distinct reduction indices. -/
theorem down_stride_inj (k w l k' w' l' : Nat) (hw : w < 8) (hw' : w' < 8) (hl : l < 32) (hl' : l' < 32)
    (h : k * 256 + w * 32 + l = k' * 256 + w' * 32 + l') : k = k' ∧ w = w' ∧ l = l' := by omega

/-- The inject kernel's warp index is valid exactly when the thread id is below 32*hc. -/
theorem inject_warp_valid (tid hc : Nat) : tid / 32 < hc ↔ tid < 32 * hc := by omega

/-- The block reduction needs 8 partials for 256 threads, and the double scratch can hold the float view. -/
theorem block_sum_scratch : (256 + 31) / 32 = 8 ∧ 8 * 4 ≤ 8 * 8 := by decide

-- § dispatch flags

def useFp32 (fp32 native : Bool) : Bool := fp32 || native

/-- The native MMVF path always selects FP32 activations. -/
theorem native_implies_fp32 (fp32 : Bool) : useFp32 fp32 true = true := by
  simp [useFp32]

/-- Without either flag the bf16-quantised activation path runs. -/
theorem default_is_bf16 : useFp32 false false = false := by
  simp [useFp32]

end Strata.GatedResidual
