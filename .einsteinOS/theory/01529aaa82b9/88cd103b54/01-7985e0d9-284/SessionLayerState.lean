/-!
# MODULE: session_layer_state
Per-session state carving and per-layer indexing for one token through all layers: the 256-byte session
arena, the GDN state slice owned by each non-QSA layer, and the doorbell sequence numbers the loop waits for.

## Source semantics
- The session arena is carved with 256-byte alignment and the total is rounded up to 256 (plus a 4096 pad).
- Each GDN layer owns its own slice of `gdn_state`; the slice index of a GDN layer is the number of GDN layers
  before it, which equals `l - l/4` for the interleave where QSA layers are `l % 4 == 3`.
- The loop's `expected` counter is pre-incremented, so layer `l` waits for sequence value `l + 1`.
- The per-token graph uses `want = l + 1` as well, and the flag it raises equals the value it waited for.

## Form template identifiers
- `alignUp256`, `sessionTake` <- `align_up(.., 256)` and the `take` lambda in `session_init`
- `qsaBefore`   <- count of QSA layers before `l` (the `gdn_index` loop complement)
- `gdnIndex`    <- `gdn_index` in `gdn_point_at` / `session_token`
- `ringWant`    <- `want = ++expected` in `session_loop`, `(uint32_t)(l + 1)` in `session_run_token`
-/

namespace Strata.Session

/-- Round up to a multiple of 256. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- One `take` of the session carve. -/
def sessionTake (used bytes : Nat) : Nat := alignUp256 (used + bytes)

/-- Number of QSA layers among layers `0..n-1` (QSA iff `l % 4 = 3`). -/
def qsaBefore : Nat → Nat
  | 0 => 0
  | n + 1 => qsaBefore n + (if n % 4 = 3 then 1 else 0)

/-- GDN slice index of layer `l` (number of non-QSA layers before it). -/
def gdnIndex (l : Nat) : Nat := l - qsaBefore l

/-- The doorbell sequence value layer `l` waits for. -/
def ringWant (l : Nat) : Nat := l + 1

/-- Floats of one GDN layer's recurrence plus conv history. -/
def gdnStateFloats (S hv dconv C : Nat) : Nat := S * hv * S + C * (dconv - 1)

/-- Rounding is monotone and covers the bytes. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Rounded sizes are multiples of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- A take covers the bytes it was given, so regions never overlap. -/
theorem sessionTake_covers (used bytes : Nat) : used + bytes ≤ sessionTake used bytes := by
  unfold sessionTake
  exact alignUp256_ge _

/-- Every region start stays 256-aligned. -/
theorem sessionTake_mod (used bytes : Nat) : sessionTake used bytes % 256 = 0 := by
  unfold sessionTake
  exact alignUp256_mod _

/-- Counting QSA layers: one in every four layers. -/
theorem qsaBefore_eq (n : Nat) : qsaBefore n = n / 4 := by
  induction n with
  | zero => simp [qsaBefore]
  | succ n ih =>
    simp only [qsaBefore]
    split_ifs <;> omega

/-- There are 12 QSA layers among the first 48. -/
theorem qsaBefore_48 : qsaBefore 48 = 12 := by
  rw [qsaBefore_eq]

/-- The slice index of a GDN layer is `l - l/4`. -/
theorem gdnIndex_eq (l : Nat) : gdnIndex l = l - l / 4 := by
  unfold gdnIndex
  rw [qsaBefore_eq]

/-- Consecutive non-QSA layers get consecutive slices (each layer owns its own state). -/
theorem gdnIndex_succ (l : Nat) (h : (l + 1) % 4 ≠ 3) (hl : l % 4 ≠ 3) :
    gdnIndex (l + 1) = gdnIndex l + 1 := by
  rw [gdnIndex_eq, gdnIndex_eq]
  omega

/-- Two distinct GDN layers never share a slice. -/
theorem gdnIndex_inj (l l' : Nat) (h : l % 4 ≠ 3) (h' : l' % 4 ≠ 3)
    (e : gdnIndex l = gdnIndex l') : l = l' := by
  rw [gdnIndex_eq, gdnIndex_eq] at e
  omega

/-- The 36 GDN layers occupy slice indices `0..35`. -/
theorem gdnIndex_lt_36 (l : Nat) (hl : l < 48) (h : l % 4 ≠ 3) : gdnIndex l < 36 := by
  rw [gdnIndex_eq]
  omega

/-- The doorbell value is strictly positive and strictly increasing across layers. -/
theorem ringWant_pos (l : Nat) : 0 < ringWant l := by
  unfold ringWant; omega

/-- Strict growth of the awaited sequence value. -/
theorem ringWant_lt (l : Nat) : ringWant l < ringWant (l + 1) := by
  unfold ringWant; omega

/-- State floats for the real geometry (128, 48 heads, conv 4, 10240 channels). -/
theorem gdnStateFloats_real : gdnStateFloats 128 48 4 10240 = 817152 := by
  decide

end Strata.Session
