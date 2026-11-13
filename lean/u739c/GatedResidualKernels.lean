/-!
# MODULE: gated_residual_kernels
SOURCES: src/kernels/cuda/gr.cu
The gated residual / hyper-connection: `gr_read` (per-stream RMS norm, bf16 down projection
with SiLU, up projection with sigmoid gate, mean over streams, optional injection logits) and
`gr_write` (each residual stream adds the block output scaled by `2*sigmoid(inject/hc)`), plus
the caller-owned workspace carving `gr_workspace_init`.
-/

namespace Ref.GatedResidual

-- § 16-byte alignment

/-- `(x + 15) & ~15`, as arithmetic. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding up never shrinks a region. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Rounded sizes are multiples of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

/-- Rounding adds less than 16 bytes. -/
theorem align16_lt (x : Nat) : align16 x < x + 16 := by unfold align16; omega

-- § workspace carving

/-- Offset of region `k`: the sum of the rounded sizes of regions `0 .. k-1`. -/
def offAt (sz : Nat → Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => offAt sz k + align16 (sz k)

/-- Offsets are monotone. -/
theorem offAt_mono (sz : Nat → Nat) {k k' : Nat} (h : k ≤ k') : offAt sz k ≤ offAt sz k' := by
  induction k' with
  | zero => have : k = 0 := by omega
            subst this; exact Nat.le_refl _
  | succ k' ih =>
    rcases Nat.lt_or_eq_of_le h with hl | he
    · have := ih (by omega); simp only [offAt]; omega
    · subst he; exact Nat.le_refl _

/-- Region `k` ends at or before region `k'` starts whenever `k < k'`. -/
theorem regions_disjoint (sz : Nat → Nat) {k k' : Nat} (h : k < k') :
    offAt sz k + sz k ≤ offAt sz k' := by
  have h1 : offAt sz (k + 1) ≤ offAt sz k' := offAt_mono sz h
  have h2 : sz k ≤ align16 (sz k) := align16_ge _
  simp only [offAt] at h1
  omega

/-- Every region start is 16-byte aligned (given an aligned base). -/
theorem offAt_aligned (sz : Nat → Nat) (k : Nat) : offAt sz k % 16 = 0 := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [offAt]; have := align16_mod (sz k); omega

/-- Workspace sizes in table order. -/
def wsSize (hcDim hcLr : Nat) : Nat → Nat
  | 0 => hcDim * 4
  | 1 => hcDim * 2
  | 2 => hcLr * 2
  | 3 => hcDim * 4
  | 4 => hcLr * 4
  | _ => 0

/-- Total bytes reported by `gr_workspace_init`. -/
def totalBytes (hcDim hcLr : Nat) : Nat := offAt (wsSize hcDim hcLr) 5

/-- Every region fits inside the reported total. -/
theorem region_in_total (hcDim hcLr : Nat) {k : Nat} (hk : k < 5) :
    offAt (wsSize hcDim hcLr) k + wsSize hcDim hcLr k ≤ totalBytes hcDim hcLr :=
  regions_disjoint _ hk

/-- At the real geometry (hc = 4, n_embd = 2560, hc_lr = 320) the workspace is 104,320 bytes. -/
theorem real_total : totalBytes 10240 320 = 104320 := by decide

/-- The historical bug: giving `xq` the `lq` size (640 B) where it needs 20,480 B. -/
theorem old_bug_undersized : wsSize 10240 320 2 < wsSize 10240 320 1 := by decide

-- § strided visit of the down projection

/-- Element visited by warp `w`, lane `l` at step `m` with `nw` warps. -/
def strideIdx (nw w l m : Nat) : Nat := w * 32 + l + m * (nw * 32)

/-- The start position of every (warp, lane) is below the stride. -/
theorem stride_start_lt {nw w l : Nat} (hw : w < nw) (hl : l < 32) : w * 32 + l < nw * 32 := by
  have : (w + 1) * 32 ≤ nw * 32 := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at this; omega

/-- Every element is visited: `i` is reached from start `i % (nw*32)` at step `i / (nw*32)`. -/
theorem stride_cover (nw i : Nat) (hnw : 0 < nw) :
    ∃ w l m, w < nw ∧ l < 32 ∧ strideIdx nw w l m = i := by
  have hst : 0 < nw * 32 := by omega
  have hr := Nat.mod_lt i hst
  refine ⟨i % (nw * 32) / 32, i % (nw * 32) % 32, i / (nw * 32), ?_, ?_, ?_⟩
  · have : i % (nw * 32) / 32 < nw := by
      rw [Nat.div_lt_iff_lt_mul (by decide)]; exact hr
    exact this
  · omega
  · unfold strideIdx
    have e1 := Nat.mod_add_div i (nw * 32)
    have e2 : i % (nw * 32) / 32 * 32 + i % (nw * 32) % 32 = i % (nw * 32) := by omega
    rw [Nat.mul_comm (nw * 32)] at e1
    omega

-- § gate / mean / write

/-- The gate launch `ceil(hc_dim/8)` blocks of 8 warps covers every output row. -/
theorem gate_rows_cover {i hcDim : Nat} (hi : i < hcDim) :
    ∃ b wp, b < (hcDim + 7) / 8 ∧ wp < 8 ∧ b * 8 + wp = i :=
  ⟨i / 8, i % 8, by omega, by omega, by omega⟩

/-- Sum over the `hc` streams of column `d`. -/
def streamSum (g : Nat → Int) (n d : Nat) : Nat → Int
  | 0 => 0
  | c + 1 => streamSum g n d c + g (c * n + d)

/-- If all streams agree on column `d`, the stream sum is `hc` times that value. -/
theorem streamSum_const (g : Nat → Int) (n d v : Nat) (x : Int)
    (h : ∀ c, g (c * n + d) = x) : streamSum g n d v = v * x := by
  induction v with
  | zero => simp [streamSum]
  | succ v ih => simp only [streamSum, ih, h]; rw [Int.natCast_succ, Int.add_mul, Int.one_mul]

/-- `gr_write` on flat index `i`. -/
def grWrite (R bo w : Nat → Int) (n i : Nat) : Int := R i + bo (i % n) * w (i / n)

/-- Element `(c, d)` of stream `c` receives the block output at `d` with stream `c`'s weight. -/
theorem grWrite_at (R bo w : Nat → Int) {n c d : Nat} (hd : d < n) :
    grWrite R bo w n (c * n + d) = R (c * n + d) + bo d * w c := by
  unfold grWrite
  have h1 : (c * n + d) % n = d := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]
  have h2 : (c * n + d) / n = c := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hd, Nat.zero_add]
  rw [h1, h2]

/-- A zero block output leaves every stream unchanged. -/
theorem grWrite_zero (R w : Nat → Int) (n i : Nat) : grWrite R (fun _ => 0) w n i = R i := by
  simp [grWrite]

-- § path selection

/-- fp32 activations are used when requested or implied by the native path. -/
def useFp32 (fp32 native : Bool) : Bool := fp32 || native

/-- The native path always runs with fp32 activations. -/
theorem native_implies_fp32 (fp32 : Bool) : useFp32 fp32 true = true := by
  simp [useFp32]

/-- Native MMVF precondition: even reduction lengths. -/
def nativeOk (hcDim hcLr : Nat) : Bool := hcDim % 2 == 0 && hcLr % 2 == 0

/-- The real geometry satisfies the native precondition. -/
theorem real_native_ok : nativeOk 10240 320 = true := by decide

/-- Injection logits are produced iff the injection weight is present. -/
def injectWritten (wInject : Option Nat) : Bool := wInject.isSome

/-- The final mixer (no injection weight) writes no injection. -/
theorem final_mixer_no_inject : injectWritten none = false := rfl

end Ref.GatedResidual

