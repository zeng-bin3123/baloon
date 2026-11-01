/-!
# MODULE: kv_q8_paged_cache
Paged int8 KV-cache append and gather kernels with one fp16 scale per 64-value group (`kv_append_q8_kernel`,
`kv_gather_q8_kernel`): addressing, grouping, clamping and the gather index inversion.

## Source semantics
* Append: one block = one 64-value group of one KV head of K (`z = 0`) or V (`z = 1`); the token position `pos` is read from
  the step buffer, its page is `table[pos / page_size]`, and the row is `(page * kv_heads + h) * page_size + pos % page_size`.
* Values are quantised against the STORED fp16 scale (`amax / 127` rounded to fp16), `q = round(x / scale)` clamped to
  `[-127, 127]`; scale 0 gives `q = 0`.
* Row element `row * head_dim + g * 64 + t`, scale slot `row * groups + g`, with `groups = head_dim / 64`.
* Gather: one thread handles 4 consecutive values (`char4`) of one (id, head); `i = (id * kv_heads + h) * per + q4`
  with `per = head_dim / 4`, so the destination index equals the thread index.

## Form template identifiers
* `KV_Q8_GROUP`     : 64
* `rowIndex`        : paged row of (page, head, in-page position)
* `clampQ`          : the `[-127, 127]` clamp
* `gatherDst`       : destination index of the gather
-/

namespace Strata.KvQ8

/-- Quantisation group width. -/
def KV_Q8_GROUP : Nat := 64

-- § row addressing

/-- A product bound for flat indices of the shape `(a*B + c)*D + e`. -/
theorem idx3_lt (A B D a c e : Nat) (ha : a < A) (hc : c < B) (he : e < D) :
    (a * B + c) * D + e < A * B * D := by
  have h1 : a * B + c < A * B := by
    have h : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B ha
    rw [Nat.add_mul, Nat.one_mul] at h
    omega
  have h2 : (a * B + c + 1) * D ≤ A * B * D := Nat.mul_le_mul_right D h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- Paged row of (page, head, in-page offset). -/
def rowIndex (kv_heads page_size page h off : Nat) : Nat := (page * kv_heads + h) * page_size + off

/-- A valid page, head and in-page offset give a row inside the pool of `pages * kv_heads * page_size` rows. -/
theorem rowIndex_lt (pages kv_heads page_size page h off : Nat) (hp : page < pages) (hh : h < kv_heads)
    (ho : off < page_size) : rowIndex kv_heads page_size page h off < pages * kv_heads * page_size := by
  unfold rowIndex
  exact idx3_lt pages kv_heads page_size page h off hp hh ho

/-- The in-page offset `pos % page_size` is always a valid offset. -/
theorem pos_offset_lt (pos page_size : Nat) (hp : 0 < page_size) : pos % page_size < page_size := Nat.mod_lt _ hp

/-- A position splits into its table slot and in-page offset. -/
theorem pos_split (pos page_size : Nat) (hp : 0 < page_size) :
    pos / page_size * page_size + pos % page_size = pos := by
  have := Nat.div_add_mod pos page_size
  rw [Nat.mul_comm] at this
  exact this

-- § groups and element addressing

/-- Element index of value `t` of group `g` in a row. -/
def elemIndex (head_dim row g t : Nat) : Nat := row * head_dim + g * 64 + t

/-- With `head_dim = groups * 64`, every element index stays inside its row. -/
theorem elem_in_row (groups row g t : Nat) (hg : g < groups) (ht : t < 64) :
    row * (groups * 64) + g * 64 + t < (row + 1) * (groups * 64) := by
  have h : (g + 1) * 64 ≤ groups * 64 := Nat.mul_le_mul_right 64 hg
  rw [Nat.add_mul, Nat.one_mul] at h
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Scale slot `row * groups + g` stays inside the `rows * groups` scale array. -/
theorem scale_slot_lt (rows groups row g : Nat) (hr : row < rows) (hg : g < groups) :
    row * groups + g < rows * groups := by
  have h : (row + 1) * groups ≤ rows * groups := Nat.mul_le_mul_right groups hr
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- A head dimension that is a multiple of 64 is also a multiple of 4 (needed by the `char4` gather). -/
theorem head_dim_div4 (head_dim : Nat) (h : head_dim % 64 = 0) : head_dim % 4 = 0 := by omega

/-- Four consecutive values `q4*4 .. q4*4+3` never straddle two 64-value groups. -/
theorem char4_in_one_group (q4 : Nat) : (q4 * 4) / 64 = (q4 * 4 + 3) / 64 := by omega

-- § clamp

/-- The C expression `q < -127 ? -127 : (q > 127 ? 127 : q)`. -/
def clampQ (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The clamp always lands in `[-127, 127]`. -/
theorem clampQ_bounds (q : Int) : -127 ≤ clampQ q ∧ clampQ q ≤ 127 := by
  unfold clampQ; split_ifs <;> omega

/-- The clamp is the identity on in-range values. -/
theorem clampQ_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clampQ q = q := by
  unfold clampQ; split_ifs <;> omega

/-- The clamp is `max (-127) (min 127 q)`. -/
theorem clampQ_eq_minmax (q : Int) : clampQ q = max (-127) (min 127 q) := by
  unfold clampQ; split_ifs <;> omega

-- § gather index inversion

/-- Destination index of the gather for (id, head, q4). -/
def gatherDst (kv_heads per id h q4 : Nat) : Nat := (id * kv_heads + h) * per + q4

/-- The thread index `i` decomposes as `id = i / (H*per)`, `rem = i % (H*per)`, `h = rem / per`, `q4 = rem % per`,
and the destination index equals `i`: the gather is a bijection onto a dense scratch. -/
theorem gather_dst_eq (H per i : Nat) :
    gatherDst H per (i / (H * per)) ((i % (H * per)) / per) ((i % (H * per)) % per) = i := by
  unfold gatherDst
  have h1 := Nat.div_add_mod i (H * per)
  have h2 := Nat.div_add_mod (i % (H * per)) per
  have h3 : (i / (H * per) * H + (i % (H * per)) / per) * per
      = H * per * (i / (H * per)) + per * ((i % (H * per)) / per) := by
    rw [Nat.add_mul]; ac_rfl
  omega

/-- Thread count of the gather: `n_ids * kv_heads * per`, with `(total + 255)/256` blocks. -/
theorem gather_blocks_cover (total i : Nat) (h : i < total) : i / 256 < (total + 255) / 256 := by omega

/-- Append grid has `kv_heads * groups * 2` blocks of 64 threads, one value per thread. -/
theorem append_threads (groups : Nat) : groups * 64 = groups * KV_Q8_GROUP := by unfold KV_Q8_GROUP; rfl

/-- The max reduction uses two warps of the 64-thread block. -/
theorem append_warps : 64 / 32 = 2 := by decide

end Strata.KvQ8
