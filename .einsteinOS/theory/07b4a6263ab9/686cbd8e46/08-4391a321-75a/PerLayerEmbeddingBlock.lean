/-!
# MODULE: per_layer_embedding_block
SOURCES: src/kernels/cuda/ple.cu
The GPU half of the per-layer-embedding (PLE / n-gram) block: key projection + grouped RMS norm,
query norm of the hidden state, a signed-sqrt sigmoid gate per stream, broadcast of the value
projection across streams, grouped norm, a depthwise causal dilated conv over a 9-row history,
and the residual `hidden + gated + conv`; plus the history advance and the caller-owned scratch
carving.

## Source semantics
* History is ROW-FASTEST: channel `c`, row `r` at `r + NG_HIST*c`; advancing drops row 0 and
  appends the normalized value as row `NG_HIST-1`.
* Conv tap `k` reads row `nhist - (kern-1-k)*dil`; at (kern 4, dil 3, hist 9) rows 0,3,6,9 and
  only row 9 (= nhist) is the NEW row taken from `norm`.
* Gate: `s = <key_c, query_c>/sqrt(n_embd)`, `mag = sqrt(max(|s|, 1e-6))`,
  `gate = sigmoid(sgn(s)*mag)` — symmetric about 0.5 because the sign is carried separately.
* `gated[c*n + d] = value[d] * gate[c]`; `result = hidden + gated + conv`.
* Scratch: five hc_dim floats + n_embd + hc floats, then the Q8 image (`n_embd/32 * 34` bytes),
  then the bf16 embedding copy, each region 16-byte aligned, plus 256 bytes slack. Five
  SEPARATE hc_dim buffers (aliasing `key` onto `normalized` returned the wrong key).
* bf16 rounding: `(i + ((i >> 16) & 1) + 0x7FFF) & 0xFFFF0000` (round to nearest even).
* `overlap(a, an, b, bn)` is false for null pointers or empty spans.

## Form template identifiers
history address ↦ `histAddr`, advance ↦ `advance`, tap row ↦ `tapRow`, sign ↦ `sgn`,
broadcast ↦ `bcast`, scratch layout ↦ `scratchBytes`/`actOff`/`embOff`, bf16 rounding ↦ `bf16Hi`.
-/

namespace Strata.PleBlock

-- § history

/-- Rows of history per channel. -/
def NG_HIST : Nat := 9

/-- Address of history row `r` of channel `c` (row-fastest). -/
def histAddr (r c : Nat) : Nat := r + NG_HIST * c

/-- Each channel's rows are contiguous and in range of the `NG_HIST * hc_dim` buffer. -/
theorem histAddr_lt {r c hcDim : Nat} (hr : r < NG_HIST) (hc : c < hcDim) :
    histAddr r c < NG_HIST * hcDim := by
  unfold histAddr NG_HIST at *
  omega

/-- Distinct (row, channel) pairs have distinct addresses. -/
theorem histAddr_unique {r c r' c' : Nat} (hr : r < NG_HIST) (hr' : r' < NG_HIST)
    (h : histAddr r c = histAddr r' c') : r = r' ∧ c = c' := by
  unfold histAddr NG_HIST at *
  omega

/-- History advance of one channel column. -/
def advance (col : List Int) (x : Int) : List Int := col.tail ++ [x]

/-- Advancing keeps a full column at `NG_HIST` rows and makes `x` the newest row. -/
theorem advance_spec (col : List Int) (x : Int) (h : col.length = 9) :
    (advance col x).length = 9 ∧ (advance col x).getLast? = some x := by
  cases col with
  | nil => simp at h
  | cons a t => simp_all [advance]

-- § dilated conv taps

/-- Row read by tap `k`. -/
def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- At the real geometry the taps read rows 0, 3, 6 and the new row 9. -/
theorem real_taps : (List.range 4).map (tapRow 9 4 3) = [0, 3, 6, 9] := by decide

/-- Only the last tap reads beyond the history (the new normalized row). -/
theorem only_last_tap_new : ∀ k < 4, (tapRow 9 4 3 k = 9 ↔ k = 3) := by decide

-- § gate

/-- Sign of `s` (−1, 0, 1). -/
def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- The sign is odd: `sgn(-s) = -sgn(s)`, so `gate(-s) = 1 - gate(s)` for sigmoid. -/
theorem sgn_neg (s : Int) : sgn (-s) = -sgn s := by
  unfold sgn
  by_cases h1 : s > 0
  · have h0 : ¬ (-s > 0) := by omega
    have h2 : -s < 0 := by omega
    simp only [h0, h2, h1, ↓reduceIte]
  · by_cases h3 : s < 0
    · have h0 : -s > 0 := by omega
      simp only [h0, h1, h3, ↓reduceIte, Int.neg_neg]
    · have : s = 0 := by omega
      subst this; decide

/-- The clamped magnitude argument `max(|s|, floor)` is never below the floor. -/
theorem clamp_floor (s : Int) (floor : Nat) : floor ≤ max s.natAbs floor := by omega

/-- The gate's signed argument depends on `s` only through `sgn s` and `|s|`. -/
theorem signed_arg_odd (m : Int → Int) (s : Int) :
    sgn (-s) * m ((-s).natAbs) = -(sgn s * m s.natAbs) := by
  rw [sgn_neg, Int.natAbs_neg, Int.neg_mul]

-- § broadcast / residual

/-- `gated[i] = value[i % n] * gate[i / n]`. -/
def bcast (value gate : Nat → Int) (n i : Nat) : Int := value (i % n) * gate (i / n)

/-- Element `(c, d)` is the value at `d` times stream `c`'s gate. -/
theorem bcast_at (value gate : Nat → Int) {n c d : Nat} (hd : d < n) :
    bcast value gate n (c * n + d) = value d * gate c := by
  unfold bcast
  have hn : 0 < n := by omega
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd,
    Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hd, Nat.zero_add]

/-- `add3` is associative-order independent in exact arithmetic. -/
theorem add3_comm (h g c : Int) : h + g + c = h + (g + c) := Int.add_assoc h g c

-- § scratch layout

/-- Round up to 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Float region bytes: `(5*hc_dim + n_embd + hc)*4`. -/
def floatBytes : Nat := (5 * 10240 + 2560 + 4) * 4
/-- Q8 activation image bytes: `n_embd/32 * 34`. -/
def q8Bytes : Nat := 2560 / 32 * 34
/-- BF16 embedding copy bytes. -/
def embBytes : Nat := 2560 * 2

/-- Offset of the Q8 image. -/
def actOff : Nat := align16 floatBytes
/-- Offset of the bf16 embedding copy. -/
def embOff : Nat := actOff + align16 q8Bytes
/-- `ple_block_scratch_bytes()`. -/
def scratchBytes : Nat := align16 floatBytes + align16 q8Bytes + embBytes + 256

/-- The three regions are ordered, aligned, and fit inside the reported scratch size. -/
theorem scratch_layout :
    floatBytes ≤ actOff ∧ actOff + q8Bytes ≤ embOff ∧ embOff + embBytes ≤ scratchBytes ∧
    actOff % 16 = 0 ∧ embOff % 16 = 0 := by decide

/-- The float sub-buffers `key, query, norm, gated, conv` (hc_dim each), `value` (n_embd) and
    `gate` (hc) are back to back: offsets in floats, all distinct and inside the region. -/
theorem float_buffers :
    let off := [0, 10240, 20480, 30720, 40960, 51200, 53760]
    let len := [10240, 10240, 10240, 10240, 10240, 2560, 4]
    (List.zipWith (· + ·) off len).dropLast = off.tail ∧
      off.getLast! + len.getLast! = 5 * 10240 + 2560 + 4 := by decide

-- § bf16 rounding

/-- Upper 16 bits after round-to-nearest-even (no 32-bit wrap; valid below the NaN range). -/
def bf16Hi (i : Nat) : Nat := (i + (i / 2 ^ 16) % 2 + 0x7FFF) / 2 ^ 16

/-- Values already representable in bf16 round to themselves. -/
theorem bf16_exact {i : Nat} (h : i % 2 ^ 16 = 0) : bf16Hi i * 2 ^ 16 = i := by
  unfold bf16Hi
  have hb : (i / 2 ^ 16) % 2 ≤ 1 := by omega
  omega

/-- Rounding moves a value by at most half a bf16 ulp (plus the tie bit). -/
theorem bf16_close (i : Nat) : bf16Hi i * 2 ^ 16 ≤ i + 2 ^ 15 ∧ i ≤ bf16Hi i * 2 ^ 16 + 2 ^ 15 := by
  unfold bf16Hi
  have hb : (i / 2 ^ 16) % 2 ≤ 1 := by omega
  omega

-- § overlap

/-- `overlap` as in ple.cu: null (0) or empty spans never overlap. -/
def overlap (a an b bn : Nat) : Bool :=
  if a = 0 || b = 0 || an = 0 || bn = 0 then false
  else if a < b then decide (b - a < an) else decide (a - b < bn)

/-- An empty span overlaps nothing. -/
theorem overlap_empty (a b bn : Nat) : overlap a 0 b bn = false := by simp [overlap]

/-- Adjacent non-null spans do not overlap. -/
theorem overlap_adjacent {a an bn : Nat} : overlap a an (a + an) bn = false := by
  unfold overlap
  by_cases han : an = 0
  · simp [han]
  · have hlt : a < a + an := by omega
    split
    · rfl
    · simp

end Strata.PleBlock
