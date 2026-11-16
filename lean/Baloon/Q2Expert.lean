-- Baloon.Q2Expert: Q2_0 MoE expert blob geometry constants, section offsets,
-- row-in-bounds proofs and multi-token dispatch for the native GPU kernel.
-- Independent implementation proving the same mathematical facts as Ref.Q2Expert.

namespace Baloon.Q2Expert

def H : Nat := 2560;  def FF : Nat := 640;   def NE : Nat := 512
def QK : Nat := 64;   def QKA : Nat := 32;   def BB : Nat := 18
def MAXC : Nat := 80; def ROW_GU : Nat := 640; def ROW_D : Nat := 160
def SC_GU : Nat := 40; def SC_D : Nat := 10;  def BLOB : Nat := 1382400
def O_GU_CODES : Nat := 0;      def O_D_CODES : Nat := 819200
def O_GU_SCALES : Nat := 1228800; def O_D_SCALES : Nat := 1331200
def MAXT : Nat := 8

theorem geometry_formulas :
    MAXC = H / QKA ∧ ROW_GU = H * 2 / 8 ∧ ROW_D = FF * 2 / 8 ∧ SC_GU = H / QK ∧ SC_D = FF / QK ∧
    BLOB = 3 * (H * FF * BB / QK) := by decide

theorem offset_formulas :
    O_D_CODES = 2 * FF * ROW_GU ∧ O_GU_SCALES = O_D_CODES + 1 * H * ROW_D ∧
    O_D_SCALES = O_GU_SCALES + 2 * FF * SC_GU * 2 := by decide

theorem weight_block_spans_two_chunks : QK = 2 * QKA := by decide
theorem block_bytes_split : BB = 2 + QK / 4 := by decide
theorem rows_are_whole_blocks : H = QK * SC_GU ∧ FF = QK * SC_D := by decide
theorem blob_tiles : O_D_SCALES + H * SC_D * 2 = BLOB := by decide

theorem sections_contiguous :
    O_GU_CODES + 2 * FF * ROW_GU = O_D_CODES ∧ O_D_CODES + H * ROW_D = O_GU_SCALES ∧
    O_GU_SCALES + 2 * FF * SC_GU * 2 = O_D_SCALES := by decide

theorem gu_code_rows_in_bounds (r : Nat) (hr : r < FF) :
    O_GU_CODES + (2 * r + 1) * ROW_GU + ROW_GU ≤ O_D_CODES := by
  simp only [O_GU_CODES, ROW_GU, O_D_CODES, FF] at *; omega

theorem down_code_rows_in_bounds (r : Nat) (hr : r < H) :
    O_D_CODES + r * ROW_D + ROW_D ≤ O_GU_SCALES := by
  simp only [O_D_CODES, ROW_D, O_GU_SCALES, H] at *; omega

-- ── tail masks, block batches, multi-token dispatch, CPU feature gate ─────────

/-- A mask of 2*nb bits for nb <= 8 blocks fits a 16-bit lane mask. -/
theorem tail_mask_fits : ∀ nb : Nat, nb < 9 → (1 <<< (2 * nb)) - 1 < 65536 := by decide

/-- Gate/up rows (40 blocks) are an exact multiple of the 8-block batch. -/
theorem block_batches : SC_GU % 8 = 0 ∧ SC_D % 8 = 2 ∧ (SC_GU + 7) / 8 = 5 ∧ (SC_D + 7) / 8 = 2 := by
  decide

/-- Width chosen by the switch (n_tokens) ladder. -/
def dispatchNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- For every legal token count 1..MAXT the dispatched width is exactly the token count. -/
theorem dispatch_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ MAXT) : dispatchNT n = n := by
  simp only [dispatchNT, MAXT] at *; split <;> omega

structure CpuFeatures where
  avx512f : Bool; avx512bw : Bool; avx512vl : Bool; avx512_vnni : Bool; avx512_vbmi : Bool

def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

theorem usable_iff (f : CpuFeatures) :
    f.usable = true ↔ (f.avx512f = true ∧ f.avx512bw = true ∧ f.avx512vl = true ∧
      f.avx512_vnni = true ∧ f.avx512_vbmi = true) := by
  unfold CpuFeatures.usable; simp only [Bool.and_eq_true, and_assoc]

theorem no_vnni_refused (f : CpuFeatures) (h : f.avx512_vnni = false) : f.usable = false := by
  unfold CpuFeatures.usable; simp [h]

end Baloon.Q2Expert
