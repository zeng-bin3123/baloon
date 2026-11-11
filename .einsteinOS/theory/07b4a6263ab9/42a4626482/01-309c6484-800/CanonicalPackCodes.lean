/-!
# MODULE: canonical_pack_codes
SOURCES: tools/strata_pack.py, tools/canonical_xcheck.py
The canonical-form contract of the pack: one unsigned code per element, packed LSB-first with
`8 / bits` codes per byte (Q2_0's own convention, reused for every form), the MAPPINGS table that
assigns each ggml type its form, group size, code bias and offset plane, and the manifest-driven
decode `value = (code + code_bias) * scale [+ offset]`.

## Source semantics
* `pack_codes` puts code k of each group of `8/bits` at bit offset `k*bits`; `unpack_codes`
  inverts it. For bits = 2 and 4 the round trip is the identity both ways; bits = 8 is a copy.
* Q2_0 → S2 unpacking (`codes[:, t::4]`) and `pack_codes(bits=2)` use the same convention, so
  re-packing a canonical Q2_0 code plane reproduces the source code bytes.
* for every mapping, the largest code the source type can produce fits the form's width, and the
  group size divides the block size (so `elements / group_elems` scales exactly cover a tensor).
* exactly the types with a value-space minimum (Q4_K, Q5_K) carry an offset plane.
* `verify --expert-every N` checks experts 0, N, 2N, … of 512 per layer.

## Form template identifiers
pack_codes → `packByte`; unpack_codes → `unpackByte`; MAPPINGS → `mappings` / `Mapping`;
code_bits(form) → `formBits`; decode_entry's affine branch → `decodeAffine`; choose_scale_dtype →
`chooseScale`; verify_experts sampling → `sampledExperts`.
-/

namespace Strata.Canonical

-- § code packing

/-- Pack `per = 8/bits` codes into one byte, code `k` at bit offset `k*bits` (LSB-first). -/
def packByte (bits : Nat) (cs : List Nat) : Nat :=
  (cs.zipIdx.map fun (c, k) => (c &&& (2 ^ bits - 1)) <<< (k * bits)).foldl (· ||| ·) 0

/-- Unpack `per` codes from one byte. -/
def unpackByte (bits per b : Nat) : List Nat :=
  (List.range per).map fun k => (b >>> (k * bits)) &&& (2 ^ bits - 1)

/-- All lists of `n` codes below `m`. -/
def allCodes (m : Nat) : Nat → List (List Nat)
  | 0 => [[]]
  | n + 1 => (List.range m).flatMap fun c => (allCodes m n).map (c :: ·)

/-- 2-bit codes (S2): every 4-tuple of codes survives pack → unpack. -/
theorem s2_unpack_pack :
    (allCodes 4 4).all (fun cs => unpackByte 2 4 (packByte 2 cs) == cs) = true := by
  set_option maxRecDepth 100000 in decide

/-- 2-bit codes (S2): every byte survives unpack → pack, so a re-packed canonical Q2_0 code plane
is byte-identical to the source `qs`. -/
theorem s2_pack_unpack :
    (List.range 256).all (fun b => packByte 2 (unpackByte 2 4 b) == b) = true := by
  set_option maxRecDepth 100000 in decide

/-- 4-bit codes (S4): every pair of codes survives pack → unpack. -/
theorem s4_unpack_pack :
    (allCodes 16 2).all (fun cs => unpackByte 4 2 (packByte 4 cs) == cs) = true := by
  set_option maxRecDepth 100000 in decide

/-- 4-bit codes (S4): every byte survives unpack → pack. -/
theorem s4_pack_unpack :
    (List.range 256).all (fun b => packByte 4 (unpackByte 4 2 b) == b) = true := by
  set_option maxRecDepth 100000 in decide

/-- S2 placement: the code of element `j` lives in byte `j / 4` at bit `2 * (j % 4)` — the Q2_0
source layout and the `t::4` slice describe the same slot. -/
theorem s2_slot (b t : Nat) (ht : t < 4) : (4 * b + t) / 4 = b ∧ 2 * ((4 * b + t) % 4) = t * 2 := by
  omega

-- § mapping table

/-- One MAPPINGS row: (source type, block bytes, block elems, group elems, form bits, code bias,
has offset, non-affine codebook, largest code the source can produce). -/
structure Mapping where
  name : String
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat
  formBits : Nat
  codeBias : Int
  hasOffset : Bool
  iq4Codebook : Bool
  maxCode : Nat
deriving DecidableEq, Repr

/-- The quantized rows of MAPPINGS (the P16/P32 rows have no code plane). -/
def mappings : List Mapping :=
  [⟨"Q2_0", 18, 64, 64, 2, -1, false, false, 3⟩,
   ⟨"Q3_K", 110, 256, 16, 4, -4, false, false, 7⟩,
   ⟨"Q4_0", 18, 32, 32, 4, -8, false, false, 15⟩,
   ⟨"Q4_K", 144, 256, 32, 4, 0, true, false, 15⟩,
   ⟨"Q5_0", 22, 32, 32, 8, -16, false, false, 31⟩,
   ⟨"Q5_K", 176, 256, 32, 8, 0, true, false, 31⟩,
   ⟨"Q6_K", 210, 256, 16, 8, -32, false, false, 63⟩,
   ⟨"Q8_0", 34, 32, 32, 8, -128, false, false, 255⟩,
   ⟨"IQ4_NL", 18, 32, 32, 4, 0, false, true, 15⟩,
   ⟨"IQ4_XS", 136, 256, 32, 4, 0, false, true, 15⟩]

/-- Every source code fits the form's width. -/
theorem codes_fit_form : mappings.all (fun m => m.maxCode < 2 ^ m.formBits) = true := by decide

/-- Every group size divides its block size, so scale planes tile tensors exactly. -/
theorem group_divides_block : mappings.all (fun m => m.blockElems % m.groupElems == 0) = true := by
  decide

/-- Exactly Q4_K and Q5_K carry an offset plane (the three-part encoder output). -/
theorem offset_types : (mappings.filter (·.hasOffset)).map (·.name) = ["Q4_K", "Q5_K"] := by decide

/-- Biased types decode symmetric-ish integer ranges: for every affine row the decoded integer
`code + bias` over `0..maxCode` spans `bias .. maxCode + bias`, never above `maxCode`. -/
theorem affine_range :
    (mappings.filter (!·.iq4Codebook)).all
      (fun m => m.codeBias ≤ 0 && (m.maxCode : Int) + m.codeBias ≤ m.maxCode) = true := by decide

/-- The form bit width also packs whole codes into bytes: `8 % bits = 0`. -/
theorem form_bits_pack : mappings.all (fun m => 8 % m.formBits == 0) = true := by decide

-- § decode

/-- The affine branch of `decode_entry` on integers. -/
def decodeAffine (code : Nat) (bias scale offset : Int) : Int := ((code : Int) + bias) * scale + offset

/-- With bias −1 and no offset, the affine decode is exactly Q2_0's `(code − 1) * d`. -/
theorem decode_matches_q2 (c : Nat) (d : Int) : decodeAffine c (-1) d 0 = ((c : Int) - 1) * d := by
  unfold decodeAffine; simp [Int.sub_eq_add_neg]

/-- Scale-plane length: `elements / group` scales, exact when elements are whole blocks. -/
theorem scales_cover (nBlocks blockElems group : Nat) (_hg : 0 < group)
    (hd : blockElems % group = 0) :
    nBlocks * blockElems / group * group = nBlocks * blockElems := by
  have : (nBlocks * blockElems) % group = 0 := by
    rw [Nat.mul_mod, hd, Nat.mul_zero, Nat.zero_mod]
  have h := Nat.div_add_mod (nBlocks * blockElems) group
  rw [this, Nat.add_zero] at h
  rw [Nat.mul_comm]; exact h

/-- `choose_scale_dtype`: keep the 16-bit encoding only when it round-trips. -/
def chooseScale {α : Type} [DecidableEq α] (toHalf : α → α) (x : α) : α × Bool :=
  if toHalf x = x then (toHalf x, true) else (x, false)

/-- Whichever width is chosen, the stored scale equals the original: the choice is never lossy. -/
theorem chooseScale_exact {α : Type} [DecidableEq α] (toHalf : α → α) (x : α) :
    (chooseScale toHalf x).1 = x := by
  unfold chooseScale; split <;> simp_all

-- § expert sampling

/-- Experts visited by `range(0, 512, every)`. -/
def sampledExperts (every : Nat) : List Nat :=
  (List.range 512).filter fun e => e % every == 0

/-- The default `--expert-every 32` checks 16 experts per layer, starting at expert 0. -/
theorem default_sampling : (sampledExperts 32).length = 16 ∧ (sampledExperts 32).head? = some 0 := by
  set_option maxRecDepth 100000 in decide

end Strata.Canonical
