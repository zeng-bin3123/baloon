/-!
# MODULE: strata_pack_builder
Arithmetic of `tools/strata_pack.py` (build / verify / info): LSB-first code packing, the FP16-or-FP32
scale choice, 64-byte plane alignment and the expert arena addressing.

## Source semantics
* `pack_codes(codes, bits)` stores `8 // bits` codes per byte, LSB first; `unpack_codes` inverts it.
* A scale/offset plane is stored FP16 only if the round trip is exact, otherwise FP32; `verify`
  re-checks bit exactness so a wrong choice is a failed build, not a lossy pack.
* Every plane starts on a 64-byte boundary relative to its segment start (`put` pads the body);
  absolute alignment additionally needs the segment base to be 64-aligned.
* The expert arena is `layer*NE + e` blobs of 1,382,400 bytes: every blob start is 64-aligned.
* `rebuild_role` inverts the blob layout: block `blk` of down is (row = blk / 10, b_in = blk % 10),
  of gate/up is (row = blk / 40, b_in = blk % 40).

## Form template identifiers
* packLE / unpackLE  <- pack_codes / unpack_codes
* storeScale         <- choose_scale_dtype + decode via `<f2` / `<f4`
* padTo64            <- `pad = (-len(body)) % ALIGN`
-/

namespace Strata.PackBuilder

-- § LSB-first code packing

/-- Pack codes into one natural number, first code in the lowest `b` bits. -/
def packLE (b : Nat) : List Nat → Nat
  | [] => 0
  | c :: cs => c + 2 ^ b * packLE b cs

/-- Read `n` codes of `b` bits, lowest first. -/
def unpackLE (b : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | n + 1, x => x % 2 ^ b :: unpackLE b n (x / 2 ^ b)

/-- Unpacking inverts packing whenever every code fits `b` bits. -/
theorem unpackLE_packLE (b : Nat) :
    ∀ cs : List Nat, (∀ c ∈ cs, c < 2 ^ b) → unpackLE b cs.length (packLE b cs) = cs
  | [], _ => rfl
  | c :: cs, h => by
    have hc : c < 2 ^ b := h c (by simp)
    have hpos : 0 < 2 ^ b := Nat.two_pow_pos b
    have ih := unpackLE_packLE b cs (fun x hx => h x (List.mem_cons_of_mem c hx))
    have h1 : (c + 2 ^ b * packLE b cs) % 2 ^ b = c := by
      rw [Nat.add_mul_mod_self_left]
      exact Nat.mod_eq_of_lt hc
    have h2 : (c + 2 ^ b * packLE b cs) / 2 ^ b = packLE b cs := by
      rw [Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hc, Nat.zero_add]
    simp only [List.length_cons, unpackLE, packLE, h1, h2, ih]

/-- Two 4-bit codes fit one byte. -/
theorem pack_two_nibbles_lt (a b : Nat) (ha : a < 16) (hb : b < 16) : packLE 4 [a, b] < 256 := by
  simp [packLE]
  omega

/-- Four 2-bit codes fit one byte. -/
theorem pack_four_pairs_lt (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    packLE 2 [a, b, c, d] < 256 := by
  simp [packLE]
  omega

/-- Codes per byte for the three code widths. -/
theorem codes_per_byte : 8 / 2 = 4 ∧ 8 / 4 = 2 ∧ 8 / 8 = 1 := by decide

-- § Scale precision

/-- Store as the narrow type when the round trip is exact, else keep the wide value; return decoded value. -/
def storeScale {α β : Type} [DecidableEq α] (enc : α → β) (dec : β → α) (x : α) : α :=
  if dec (enc x) = x then dec (enc x) else x

/-- Whatever choice is made, decoding returns exactly the canonical scale. -/
theorem storeScale_lossless {α β : Type} [DecidableEq α] (enc : α → β) (dec : β → α) (x : α) :
    storeScale enc dec x = x := by
  unfold storeScale
  split
  · assumption
  · rfl

-- § Alignment

/-- Pad to the next 64-byte boundary, `(-n) % 64`. -/
def padTo64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- Inside a segment, a plane written after `len` body bytes starts 64-aligned. -/
theorem put_relative_aligned (len : Nat) : (len + padTo64 len) % 64 = 0 := by
  unfold padTo64
  omega

/-- With a 64-aligned segment base the plane offset is absolutely aligned. -/
theorem put_absolute_aligned (base len : Nat) (hb : base % 64 = 0) :
    (base + (len + padTo64 len)) % 64 = 0 := by
  unfold padTo64
  omega

/-- Every expert blob start in `experts.bin` is a multiple of 64. -/
theorem expert_blob_aligned (k : Nat) : (k * 1382400) % 64 = 0 := by
  omega

/-- The blob start for (layer, expert) is the flat index times the blob size. -/
theorem expert_flat_index (layer e : Nat) (he : e < 512) : (layer * 512 + e) < (layer + 1) * 512 := by
  omega

-- § Expert arena inversion

/-- Down role: block -> (row, in-row block) is a bijection onto 2560 x 10. -/
theorem down_block_split (blk : Nat) (h : blk < 25600) :
    blk / 10 < 2560 ∧ blk % 10 < 10 ∧ (blk / 10) * 10 + blk % 10 = blk := by
  omega

/-- Gate/up role: block -> (row, in-row block) is a bijection onto 640 x 40. -/
theorem gate_up_block_split (blk : Nat) (h : blk < 25600) :
    blk / 40 < 640 ∧ blk % 40 < 40 ∧ (blk / 40) * 40 + blk % 40 = blk := by
  omega

/-- Blocks per expert role: H*FF/64 = 25600 and bytes 25600*18 = 460800 = BLOB/3. -/
theorem role_bytes : 2560 * 640 / 64 = 25600 ∧ 25600 * 18 = 460800 ∧ 3 * 460800 = 1382400 := by
  decide

/-- With `--expert-every 32`, 16 experts per layer are checked, 48 role checks per layer. -/
theorem expert_every_32 : (512 + 31) / 32 = 16 ∧ 16 * 3 = 48 := by decide

end Strata.PackBuilder
