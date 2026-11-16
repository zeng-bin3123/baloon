-- Baloon.Vision: image-encoder tool file format (SVE1 magic, header, payload),
-- token grid geometry, ENC command-line split and REF_PORTABLE build options.
-- Independent implementation proving the same mathematical facts as Ref.Vision.

namespace Baloon.Vision

theorem magic_value : 0x53 + 0x56 * 256 + 0x45 * 65536 + 0x31 * 16777216 = 0x31455653 := by { decide }

def headerBytes : Nat := (5 * 4)
def fileBytes (n nEmbd : Nat) : Nat := (headerBytes + 4 * (n * nEmbd))

theorem header_is_20 : headerBytes = 20 := by { decide }

theorem payload_size (n nEmbd : Nat) : fileBytes n nEmbd - headerBytes = 4 * n * nEmbd := by
  unfold fileBytes; rw [Nat.add_sub_cancel_left, Nat.mul_assoc]

theorem empty_file (nEmbd : Nat) : fileBytes 0 nEmbd = headerBytes := by { simp [fileBytes] }
def gridDims (x y : Nat) : Nat × Nat := (((x + 1, y + 1)))
def rectangular (n x y : Nat) : Bool := ((x + 1) * (y + 1) == n)

theorem rectangular_iff (n x y : Nat) :
    rectangular n x y = true ↔ (gridDims x y).1 * (gridDims x y).2 = n := by { simp [rectangular, gridDims] }

theorem grid_position_roundtrip (i nx : Nat) : (i / nx) * nx + i % nx = i := by
  rw [Nat.mul_comm]; exact Nat.div_add_mod i nx

theorem column_lt (i nx : Nat) (h : 0 < nx) : i % nx < nx := by exact Nat.mod_lt _ h

theorem row_lt (i nx ny : Nat) (h : 0 < nx) (hi : i < nx * ny) : i / nx < ny :=
  Nat.div_lt_of_lt_mul hi

theorem split_reassembles (l : List Nat) (sp : Nat) : l.take sp ++ l.drop sp = l := by exact List.take_append_drop sp l

def validSplit (len sp : Nat) : Bool := (0 < sp && sp < len)

theorem validSplit_parts (len sp : Nat) (h : validSplit len sp = true) :
    0 < sp ∧ 0 < len - (sp + 1) + 1 := by
  unfold validSplit at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h; omega

theorem enc_prefix_len : "ENC ".length = 4 := by { decide }

def ggmlNative (portable : Bool) : Bool := (!portable)
def baselineFlags (portable : Bool) : Bool := (portable)
theorem native_xor_baseline (p : Bool) : ggmlNative p = !baselineFlags p := by { cases p <;> rfl }
theorem portable_not_native : ggmlNative true = false := by { rfl }
theorem default_native : ggmlNative false = true ∧ baselineFlags false = false := by { decide }

end Baloon.Vision
