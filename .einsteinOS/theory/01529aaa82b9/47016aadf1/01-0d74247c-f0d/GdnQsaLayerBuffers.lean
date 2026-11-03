/-!
# MODULE: gdn_qsa_layer_buffers
Buffer carving, plane layout, kernel dispatch and dump layout of `layer.cpp` (GDN layer, QSA mixer, MoE block).

## Source semantics
* Every buffer region is 16-byte aligned; `*_bytes` and `*_init` walk the same list, so they agree.
* A quantized tensor's planes (codes | scales | offsets) must add up exactly to the tensor size.
* Which GEMV runs is decided per tensor: native, else S2 (2-bit), else Q8_0 or Q8_K by `wants_q8k`.
* The per-layer dump stride equals the last slot's end.

## Form template identifiers
`align_up16` -> align16, `q8k_bytes` -> q8kBytes, `Planes` -> codes/scales/offset sizes, `dump_stride_floats` -> dumpStride.
-/

namespace Strata.Layer

-- § alignment and carving

/-- 16-byte alignment of every carved region. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Aligned size covers the request. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- Aligned size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Q8_K buffer size: `n/256` blocks of 292 bytes. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- For `n` a multiple of 256 the Q8_K size is exactly `n * 292 / 256`. -/
theorem q8kBytes_exact (n : Nat) (h : n % 256 = 0) : q8kBytes n * 256 = n * 292 := by
  unfold q8kBytes; omega

/-- End offset after carving `parts` from `acc` (the `total` loop of `*_buffers_bytes`). -/
def carveEnd (acc : Nat) (parts : List Nat) : Nat := parts.foldl (fun a p => a + align16 p) acc

/-- The carved total is the sum of the aligned parts. -/
theorem carveEnd_eq : ∀ (parts : List Nat) (acc : Nat), carveEnd acc parts = acc + (parts.map align16).sum
  | [], acc => by simp [carveEnd]
  | p :: ps, acc => by
      have := carveEnd_eq ps (acc + align16 p)
      simp only [carveEnd, List.foldl_cons, List.map_cons, List.sum_cons] at this ⊢
      omega

/-- Offset of the `i`th carved region. -/
def off16 : List Nat → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | p :: ps, i + 1 => align16 p + off16 ps i

/-- Every carved region starts 16-byte aligned. -/
theorem off16_aligned : ∀ (l : List Nat) (i : Nat), off16 l i % 16 = 0
  | _, 0 => by simp [off16]
  | [], _ + 1 => by simp [off16]
  | p :: ps, i + 1 => by
      have h := off16_aligned ps i
      have h2 := align16_mod p
      simp only [off16]
      omega

/-- Every region offset is within the carved total. -/
theorem off16_le_total : ∀ (l : List Nat) (i : Nat), off16 l i ≤ (l.map align16).sum
  | _, 0 => by simp [off16]
  | [], _ + 1 => by simp [off16]
  | p :: ps, i + 1 => by
      have h := off16_le_total ps i
      simp only [off16, List.map_cons, List.sum_cons]
      omega

-- § planes

/-- The three planes of a quantized tensor tile it exactly (the `plane_ptrs` check). -/
theorem planes_tile (codes scales offs bytes : Nat) (h : codes + scales + offs = bytes) :
    codes + scales ≤ bytes ∧ codes ≤ bytes := by omega

/-- With an offset plane, the offset pointer `codes + scales` plus its size ends at the tensor end. -/
theorem planes_offset_end (codes scales offs bytes : Nat) (h : codes + scales + offs = bytes) :
    (codes + scales) + offs = bytes := h

-- § kernel dispatch

/-- Which GEMV a quantized projection uses. -/
inductive Kind | native | s2 | q8_0 | q8k
  deriving DecidableEq

/-- The per-tensor dispatch of `gemv_quantized`. -/
def kernelOf (native : Bool) (bits : Nat) (wantsQ8k : Bool) : Kind :=
  if native then .native else if bits = 2 then .s2 else if wantsQ8k then .q8k else .q8_0

/-- A native tensor always takes the native path. -/
theorem kernelOf_native (bits : Nat) (w : Bool) : kernelOf true bits w = .native := by simp [kernelOf]

/-- A 2-bit tensor takes S2 (Q8_0 activation). -/
theorem kernelOf_s2 (w : Bool) : kernelOf false 2 w = .s2 := by simp [kernelOf]

/-- A legacy 4/8-bit form that does not want Q8_K takes the Q8_0 split kernel. -/
theorem kernelOf_q8_0 (bits : Nat) (h : bits ≠ 2) : kernelOf false bits false = .q8_0 := by
  simp [kernelOf, h]

/-- A K-quant takes Q8_K. -/
theorem kernelOf_q8k (bits : Nat) (h : bits ≠ 2) : kernelOf false bits true = .q8k := by
  simp [kernelOf, h]

-- § layer kinds

/-- Twelve of the 48 layers are QSA (`l % 4 = 3`). -/
theorem qsa_layer_count : (List.range 48).countP (fun l => decide (l % 4 = 3)) = 12 := by decide

/-- The other 36 are GDN. -/
theorem gdn_layer_count : (List.range 48).countP (fun l => decide (l % 4 ≠ 3)) = 36 := by decide

-- § q/gate split and RoPE bound

/-- The q/gate half-split reads row `h` of the `[nh x 2*hd]` buffer inside its bounds. -/
theorem qgate_row_in_bounds (nh hd h : Nat) (hh : h < nh) : h * (2 * hd) + hd ≤ nh * (2 * hd) := by
  have h1 := Nat.mul_le_mul_right (2 * hd) (show h + 1 ≤ nh by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- The RoPE table check: a position inside the table means the position itself is in range. -/
theorem rope_pos_in_table (posBase pos maxCells : Nat) (h : posBase + pos < maxCells) : pos < maxCells := by
  omega

-- § dump layout

/-- Floats per layer in the half-level dump. -/
def dumpStride (n hc nhhd nvk : Nat) : Nat := 2 * n + 2 * hc + nhhd + 5 * nvk + 8

/-- The last slot (`kcur`, at `base + 2*nvk`, length `nvk`) ends exactly at the stride. -/
theorem dump_last_slot_ends_at_stride (n hc nhhd nvk : Nat) :
    (2 * n + 2 * hc + nhhd + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc nhhd nvk := by
  unfold dumpStride; omega

/-- The gathered-`v` slots (`vs + nvk/2 + 4 + 4`) fit before the stride. -/
theorem dump_gather_slots_fit (n hc nhhd nvk : Nat) :
    (2 * n + 2 * hc + nhhd + 2 * nvk) + nvk / 2 + 4 + 4 ≤ dumpStride n hc nhhd nvk := by
  unfold dumpStride; omega

-- § compatibility gate

/-- The verify window requires every fast-path flag. -/
def verifyCompatible (bf16 gr gdn attn sel idx : Bool) : Bool := bf16 && gr && gdn && attn && sel && idx

/-- Compatibility implies the fused hyper-connection read is on. -/
theorem verifyCompatible_needs_gr (bf16 gr gdn attn sel idx : Bool)
    (h : verifyCompatible bf16 gr gdn attn sel idx = true) : gr = true := by
  unfold verifyCompatible at h
  simp at h
  tauto

end Strata.Layer
