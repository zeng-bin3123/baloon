/-!
# MODULE: shared_expert_dispatch
The shared expert and MoE combine (`shared_expert.cu`): scratch carving, the activation-contract dispatch
of each projection, the native/legacy path preconditions, the scalar gate reduction shape and the combine
kernel's bounds.

## Source semantics
* `h = silu(x @ gate.T) * (x @ up.T)`; `h = h @ down.T`; `g = sigmoid(x . gate_inp)` one scalar per token;
  result `h * g`, ADDED plain to the routed output.
* Scratch regions, in order: gate (n_ff f32), up (n_ff f32), Q8_0 image, Q8_K image, g (1 f32), each
  16-byte aligned.
* A projection's kernel is chosen by `(code_bits, act_kind)`: 2-bit forms use the Q8_0 S2 kernel, `act_kind = 1`
  the Q8_K split kernel, everything else the Q8_0 split kernel.  Q8_K needs `n_in % 256 = 0`.

## Form template identifiers
- `SForm.code_bits` → `bits`; `SForm.act_kind` → `act`; `n_ff` → `nFF`; `n_embd` → `nEmbd`
- `native_bf16` → `useNative`; `native->gate_data` etc. → `ng/nu/nd`
-/

namespace Strata.SharedExpert

-- § Scratch carving

/-- 16-byte alignment. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding up to 16 is above, within 15, and aligned. -/
theorem align16_spec (x : Nat) : x ≤ align16 x ∧ align16 x < x + 16 ∧ align16 x % 16 = 0 := by
  unfold align16; omega

/-- Region size of a plane of `n_ff` floats. -/
def planeBytes (nFF : Nat) : Nat := align16 (nFF * 4)
/-- Region size of the Q8_0 image (34 bytes per 32). -/
def q80Bytes (nFF : Nat) : Nat := align16 (nFF / 32 * 34)
/-- Region size of the Q8_K image (292 bytes per 256). -/
def q8kBytes (nFF : Nat) : Nat := align16 (nFF / 256 * 292)

/-- Total scratch bytes. -/
def scratchBytes (nFF : Nat) : Nat := planeBytes nFF * 2 + q80Bytes nFF + q8kBytes nFF + 32

/-- Offset of the scalar gate `g`. -/
def gOffset (nFF : Nat) : Nat := planeBytes nFF * 2 + q80Bytes nFF + q8kBytes nFF

/-- The scalar gate fits in the trailing 32 bytes. -/
theorem g_fits (nFF : Nat) : gOffset nFF + 4 ≤ scratchBytes nFF := by
  unfold gOffset scratchBytes; omega

/-- Every region start is 16-byte aligned. -/
theorem offsets_aligned (nFF : Nat) :
    planeBytes nFF % 16 = 0 ∧ (planeBytes nFF * 2) % 16 = 0 ∧
    (planeBytes nFF * 2 + q80Bytes nFF) % 16 = 0 ∧ gOffset nFF % 16 = 0 := by
  have h1 := align16_spec (nFF * 4)
  have h2 := align16_spec (nFF / 32 * 34)
  have h3 := align16_spec (nFF / 256 * 292)
  unfold gOffset planeBytes q80Bytes q8kBytes
  omega

/-- The regions are disjoint and ordered: gate | up | q8_0 | q8_k | g. -/
theorem regions_ordered (nFF : Nat) :
    planeBytes nFF ≤ planeBytes nFF * 2 ∧
    planeBytes nFF * 2 ≤ planeBytes nFF * 2 + q80Bytes nFF ∧
    planeBytes nFF * 2 + q80Bytes nFF ≤ gOffset nFF ∧ gOffset nFF < scratchBytes nFF := by
  unfold gOffset scratchBytes; omega

/-- Each plane holds `n_ff` floats. -/
theorem plane_holds (nFF : Nat) : nFF * 4 ≤ planeBytes nFF := by
  have := align16_spec (nFF * 4)
  unfold planeBytes; omega

/-- The Q8_0 image of `n_ff` (multiple of 32) elements fits its region. -/
theorem q80_holds (nFF : Nat) : nFF / 32 * 34 ≤ q80Bytes nFF := by
  have := align16_spec (nFF / 32 * 34)
  unfold q80Bytes; omega

/-- The Q8_K image fits its region. -/
theorem q8k_holds (nFF : Nat) : nFF / 256 * 292 ≤ q8kBytes nFF := by
  have := align16_spec (nFF / 256 * 292)
  unfold q8kBytes; omega

/-- For the 640-wide shared FFN: Q8_K image holds only two blocks and the tail is NOT covered. -/
theorem n640_q8k_truncated : 640 / 256 * 256 = 512 ∧ 512 < 640 := by decide

/-- `ffn_down_shexp`: `n_ff = 640` is a multiple of 32 but not 256, so Q8_K is structurally impossible. -/
theorem q8k_impossible_for_640 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- `n_embd = 2560` is a multiple of 256, so the gate/up projections may use Q8_K. -/
theorem q8k_ok_for_2560 : 2560 % 256 = 0 := by decide

-- § Kernel dispatch

/-- Kernels a projection can run. -/
inductive Kernel where
  | s2Q8        -- 2-bit codes, Q8_0 activation (`s2_gemv_q8`)
  | q8kSplit    -- `s_gemv_q8k_split`
  | q80Split    -- `s_gemv_q8_0_split`
  deriving DecidableEq, Repr

/-- Dispatch rule of the `gemv` lambda. -/
def dispatch (bits act : Nat) : Kernel :=
  if bits = 2 then Kernel.s2Q8 else if act = 1 then Kernel.q8kSplit else Kernel.q80Split

/-- Two-bit forms always take the S2 Q8_0 kernel, whatever `act_kind` says. -/
theorem dispatch_two_bit (act : Nat) : dispatch 2 act = Kernel.s2Q8 := by
  unfold dispatch; simp

/-- A wider form with `act_kind = 1` takes the Q8_K kernel. -/
theorem dispatch_q8k (bits : Nat) (h : bits ≠ 2) : dispatch bits 1 = Kernel.q8kSplit := by
  unfold dispatch; simp [h]

/-- A wider form with another act kind takes the Q8_0 split kernel. -/
theorem dispatch_q80 (bits act : Nat) (h : bits ≠ 2) (ha : act ≠ 1) : dispatch bits act = Kernel.q80Split := by
  unfold dispatch; simp [h, ha]

/-- The Q8_K kernel is never chosen for two-bit forms (its host launcher refuses them). -/
theorem q8k_never_two_bit (act : Nat) : dispatch 2 act ≠ Kernel.q8kSplit := by
  rw [dispatch_two_bit]; intro h; cases h

/-- Down projection: the Q8_K branch requires `n_ff % 256 = 0`; the guard is exactly that. -/
def downQ8kAllowed (act nFF : Nat) : Bool := act == 1 && nFF % 256 == 0

/-- With `act_kind = 1` and `n_ff = 640` the engine refuses (exits) rather than computing. -/
theorem down_refuses_640 : downQ8kAllowed 1 640 = false := by decide

/-- With `act_kind = 1` and a multiple of 256 it proceeds. -/
theorem down_accepts_2048 : downQ8kAllowed 1 2048 = true := by decide

-- § Native path preconditions

/-- `x_f32` is required when the native BF16 gate, native gate or native up projection is active. -/
def needsXf32 (useNative ng nu : Bool) : Bool := useNative || ng || nu

/-- Native down alone does not require `x_f32`. -/
theorem down_only_no_xf32 : needsXf32 false false false = false := by decide

/-- Any one of the three triggers the requirement. -/
theorem xf32_triggers (a b c : Bool) (h : a = true ∨ b = true ∨ c = true) : needsXf32 a b c = true := by
  unfold needsXf32
  rcases h with h | h | h <;> simp [h]

/-- `shared_expert_multi` token count is 1..8. -/
def multiOk (n : Nat) : Bool := 1 ≤ n && n ≤ 8

/-- Boundary cases of the multi-token guard. -/
theorem multi_bounds : multiOk 0 = false ∧ multiOk 1 = true ∧ multiOk 8 = true ∧ multiOk 9 = false := by decide

/-- The multi path reuses one q8_1 scratch for gate/up and then for down: needs `n_ff` ≥ ... both fit if the
larger of `n_embd`, `n_ff` columns fit; here `n_embd = 2560 ≥ n_ff = 640`. -/
theorem q81_scratch_reuse : 640 ≤ 2560 := by decide

/-- Gate rows scaled: element `(t, i)` at `t*n + i < n_tok * n`. -/
theorem scale_rows_index (t i n nTok : Nat) (ht : t < nTok) (hi : i < n) : t * n + i < nTok * n := by
  have : (t + 1) * n ≤ nTok * n := Nat.mul_le_mul_right n ht
  rw [Nat.add_mul] at this
  omega

-- § Scalar gate reduction shape

/-- A 256-thread block has 8 warps; the 8-slot scratch holds one partial each. -/
theorem warp_slots (nt : Nat) (h : nt ≤ 256) : (nt + 31) / 32 ≤ 8 := by omega

/-- Strided accumulation: element `i` belongs to thread `i % 256`, iteration `i / 256`. -/
theorem gate_stride (i : Nat) : i = i % 256 + 256 * (i / 256) := by omega

/-- 2560 elements over 256 threads: ten terms per thread (the shortened dependency chain). -/
theorem ten_terms : 2560 / 256 = 10 := by decide

/-- The dependency chain went from 2560 sequential terms to 10 plus a tree: strictly shorter. -/
theorem chain_shorter : 10 + 8 + 5 < 2560 := by decide

/-- Warp tree offsets 16, 8, 4, 2, 1 reach all 32 lanes: they sum to 31. -/
theorem tree_sum : 16 + 8 + 4 + 2 + 1 = 31 := by decide

-- § moe_combine

/-- The combine refuses `k > 64`. -/
def combineOk (k : Nat) : Bool := 0 < k && k ≤ 64

/-- Boundary cases. -/
theorem combine_bounds : combineOk 0 = false ∧ combineOk 10 = true ∧ combineOk 64 = true ∧ combineOk 65 = false := by
  decide

/-- Parts index `e*n_embd + j` stays below `k*n_embd`. -/
theorem parts_index (e j nEmbd k : Nat) (he : e < k) (hj : j < nEmbd) : e * nEmbd + j < k * nEmbd := by
  have : (e + 1) * nEmbd ≤ k * nEmbd := Nat.mul_le_mul_right nEmbd he
  rw [Nat.add_mul] at this
  omega

/-- 128-thread grid covers all `n_embd` outputs. -/
theorem combine_grid (n : Nat) : n ≤ 128 * ((n + 127) / 128) := by omega

end Strata.SharedExpert
