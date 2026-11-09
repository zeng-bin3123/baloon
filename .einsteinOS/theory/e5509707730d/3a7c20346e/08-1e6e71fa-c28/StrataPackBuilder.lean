/-!
# MODULE: strata_pack_builder
SOURCES: tools/strata_pack.py
The pack builder and verifier: tensors are canonicalised into planes (codes, scales, offsets), each plane
starting on a 64-byte boundary, and `verify` decodes the pack from disk and compares it bit for bit.

## Source semantics
* `pack_codes` writes `8 / bits` codes per byte, LSB first; `unpack_codes` inverts it for in-range codes.
* `put` pads the running body to a 64-byte boundary before each plane. The ABSOLUTE offset is aligned when
  the segment base is aligned; segment bases advance by unpadded blob lengths, so alignment of later tensors
  holds only if each blob length is a multiple of 64 (a property the verifier does not check).
* The expert arena is addressed `(layer * 512 + e) * blob_bytes`; `verify_experts` samples every Nth expert.
* Scales are stored FP16 only when the FP16 round trip is exact, otherwise FP32.

## Form template identifiers
- pack_codes / unpack_codes      → `packLSB` / `unpackLSB`
- put() padding                  → `padTo64`
- (layer*NE + e) * blob          → `expertIndex`
- code_bits(form)                → `codeBits`
-/

namespace Strata.PackBuilder

-- § LSB-first code packing

/-- Pack codes LSB first, `b` bits per code. -/
def packLSB (b : Nat) : List Nat → Nat
  | [] => 0
  | c :: cs => c + 2 ^ b * packLSB b cs

/-- Unpack `n` codes of `b` bits, LSB first. -/
def unpackLSB (b : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | n + 1, v => (v % 2 ^ b) :: unpackLSB b n (v / 2 ^ b)

/-- Unpacking inverts packing for in-range codes, for every code width. -/
theorem unpack_pack (b : Nat) :
    ∀ cs : List Nat, (∀ c ∈ cs, c < 2 ^ b) → unpackLSB b cs.length (packLSB b cs) = cs := by
  intro cs
  induction cs with
  | nil => intro _; rfl
  | cons c cs ih =>
    intro h
    have hc : c < 2 ^ b := h c (by simp)
    have hr : ∀ x ∈ cs, x < 2 ^ b := fun x hx => h x (by simp [hx])
    have hpos : 0 < 2 ^ b := Nat.two_pow_pos b
    have h1 : (c + 2 ^ b * packLSB b cs) % 2 ^ b = c := by
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hc]
    have h2 : (c + 2 ^ b * packLSB b cs) / 2 ^ b = packLSB b cs := by
      rw [Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hc, Nat.zero_add]
    simp only [List.length_cons, unpackLSB, packLSB, h1, h2, ih hr]

/-- Canonical form widths. -/
def codeBits : String → Nat
  | "S2" => 2
  | "S4" => 4
  | "S8" => 8
  | _ => 0

/-- Every canonical width fills a byte with a whole number of codes. -/
theorem code_bits_fill :
    codeBits "S2" * (8 / codeBits "S2") = 8 ∧ codeBits "S4" * (8 / codeBits "S4") = 8 ∧
      codeBits "S8" * (8 / codeBits "S8") = 8 := by
  decide

-- § 64-byte plane alignment

/-- Padding `put` inserts: `(-len) % 64`. -/
def padTo64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- The padded body length is a multiple of 64. -/
theorem padTo64_aligned (n : Nat) : (n + padTo64 n) % 64 = 0 := by
  unfold padTo64
  omega

/-- Padding is smaller than one alignment unit. -/
theorem padTo64_lt (n : Nat) : padTo64 n < 64 := by
  unfold padTo64
  omega

/-- If the segment base is aligned, every plane's absolute offset is aligned. -/
theorem plane_offset_aligned (base len : Nat) (h : base % 64 = 0) :
    (base + (len + padTo64 len)) % 64 = 0 := by
  unfold padTo64
  omega

/-- If the base is NOT aligned and a blob length is not a multiple of 64, the next base can be unaligned:
(base 0, blob 10) leaves the next segment at 10. -/
theorem unaligned_blob_breaks_next_base : (0 + 10) % 64 ≠ 0 := by
  decide

-- § expert arena addressing

/-- Index of expert `e` of layer `l` in the arena (in blobs). -/
def expertIndex (l e : Nat) : Nat := l * 512 + e

/-- Distinct (layer, expert) pairs get distinct arena slots. -/
theorem expertIndex_injective (l1 l2 e1 e2 : Nat) (h1 : e1 < 512) (h2 : e2 < 512)
    (h : expertIndex l1 e1 = expertIndex l2 e2) : l1 = l2 ∧ e1 = e2 := by
  unfold expertIndex at h
  omega

/-- Sampling every 32nd expert checks 16 of the 512 per layer. -/
theorem sampled_experts : 512 / 32 = 16 := by
  decide

/-- The sampled expert indices `0, 32, ..., 480` are all in range. -/
theorem sampled_in_range (k : Nat) (h : k < 16) : k * 32 < 512 := by
  omega

-- § plane sizes

/-- Scale plane bytes for `n` scales at FP16 (2 bytes) or FP32 (4 bytes). -/
def scalePlaneBytes (n : Nat) (fp16 : Bool) : Nat := if fp16 then n * 2 else n * 4

/-- FP16 planes are exactly half the size of FP32 planes. -/
theorem fp16_half (n : Nat) : 2 * scalePlaneBytes n true = scalePlaneBytes n false := by
  unfold scalePlaneBytes
  simp <;> omega

/-- Codes plane bytes for `n` codes at `bits` bits, when `n` is a whole number of bytes. -/
theorem codes_plane_bytes (n per : Nat) (h : 0 < per) : (n * per) / per = n := by
  exact Nat.mul_div_cancel n h

end Strata.PackBuilder
