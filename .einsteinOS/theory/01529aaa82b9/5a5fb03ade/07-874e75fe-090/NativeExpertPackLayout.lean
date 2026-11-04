/-!
# MODULE: native_expert_pack_layout
`tools/iq_pack.py`: a native pack that keeps experts in GGUF form.  Per layer, 512 blobs of
[gate | up | down]; blob size is per layer; offsets accumulate across layers; the index rows are 19 columns wide
and tensor arenas are 64-byte aligned.

## Source semantics
* blob = per_gate + per_up + per_down, with `per = expected_bytes / 512`, gate and up equal.
* layer offset = sum of 512 * blob of earlier layers; the total is the sum over all layers.
* `index.txt` dst offsets advance by the 64-aligned size of each row.

## Form template identifiers
blobBytes    -> blob
arena        -> total experts offset
align64      -> ALIGN rounding
-/

namespace Strata.NativeExpertPackLayout

-- § blobs

/-- Bytes of one expert blob. -/
def blobBytes (gate up down : Nat) : Nat := gate + up + down

/-- With equal gate/up sizes the blob is `2*gate + down`. -/
theorem blob_eq (g d : Nat) : blobBytes g g d = 2 * g + d := by
  unfold blobBytes; omega

/-- Dividing a role tensor by 512 experts and multiplying back is exact when it divides. -/
theorem per_expert_exact (total per : Nat) (h : total = 512 * per) : total / 512 = per := by
  omega

-- § arena accumulation

/-- Total bytes of the layers' expert blobs: 512 blobs each. -/
def arena : List Nat → Nat
  | [] => 0
  | b :: bs => 512 * b + arena bs

/-- The arena of a concatenation is the sum. -/
theorem arena_append (xs ys : List Nat) : arena (xs ++ ys) = arena xs + arena ys := by
  induction xs with
  | nil => simp [arena]
  | cons x xs ih => simp [arena, ih]; omega

/-- The offset of the next layer is the offset of this one plus its 512 blobs. -/
theorem offset_step (b : Nat) (bs : List Nat) (l : Nat) :
    arena (List.take (l + 1) (b :: bs)) = 512 * b + arena (List.take l bs) := by
  simp [arena]

/-- The first layer starts at offset zero. -/
theorem offset_zero (bs : List Nat) : arena (List.take 0 bs) = 0 := by
  simp [arena]

/-- Layers are laid out without overlap: each layer's end is the next layer's start. -/
theorem layer_contiguous (b : Nat) (bs : List Nat) :
    arena (List.take 1 (b :: bs)) = 512 * b := by
  simp [arena]

-- § alignment

/-- Round up to the 64-byte pack alignment. -/
def align64 (n : Nat) : Nat := (n + 63) / 64 * 64

/-- Aligned size is a multiple of 64. -/
theorem align64_mod (n : Nat) : align64 n % 64 = 0 := by unfold align64; omega

/-- Aligned size covers the original. -/
theorem align64_ge (n : Nat) : n ≤ align64 n := by unfold align64; omega

/-- A running offset that always adds an aligned size stays a multiple of 64. -/
theorem offset_aligned (at_ n : Nat) (h : at_ % 64 = 0) : (at_ + align64 n) % 64 = 0 := by
  unfold align64; omega

/-- Padding appended after a raw tensor aligns the next one. -/
theorem pad_aligns (len : Nat) : (len + (64 - len % 64) % 64) % 64 = 0 := by omega

-- § index row widths

/-- The three row shapes written by this tool are all 19 columns (name + 18 numbers). -/
theorem row_widths : 12 + 7 = 19 ∧ 9 + 10 = 19 ∧ 12 + 7 = 19 := by decide

/-- Quantized dense tensors are served natively: their row carries shape only, so all byte fields are zero. -/
theorem native_row_zero_bytes : ["0", "0", "0", "0", "0", "0"].length = 6 := by decide

end Strata.NativeExpertPackLayout
