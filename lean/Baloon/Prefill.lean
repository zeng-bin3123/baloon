-- Baloon.Prefill: hyper-connection indexing, launch-grid coverage, GDN channel
-- layout and expert blob byte constants for the batched-prefill engine.
-- Independent implementation proving the same mathematical facts as Ref.PrefillKernels.

namespace Baloon.Prefill

theorem hc_index_recompose (i : Nat) :
    i / 10240 * 10240 + (i % 10240) / 2560 * 2560 + i % 2560 = i := by { omega }

theorem hc_stream_lt (i : Nat) : (i % 10240) / 2560 < 4 := by { omega }
theorem hc_dim_lt (i : Nat) : i % 2560 < 2560 := by { omega }

theorem hc_tok_lt (T i : Nat) (h : i < T * 10240) : i / 10240 < T := by { omega }

theorem broadcast_src_lt (T i : Nat) (h : i < T * 10240) :
    i / 10240 * 2560 + i % 2560 < T * 2560 := by { omega }

theorem write_inj_lt (i ld : Nat) (hld : 4 ≤ ld) :
    i / 10240 * ld + (i % 10240) / 2560 < (i / 10240 + 1) * ld := by
  have : (i % 10240) / 2560 < 4 := by { omega }
  have h2 : (i / 10240 + 1) * ld = i / 10240 * ld + ld := by rw [Nat.succ_mul]
  omega

def blocksFor (n t : Nat) : Nat := ((n + t - 1) / t)

theorem blocksFor_cover (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h := (Nat.div_lt_iff_lt_mul ht).mp (Nat.lt_succ_self ((n + t - 1) / t))
  rw [Nat.succ_mul] at h; omega

theorem gdn_conv_grid : 10240 / 128 * 128 = 10240 := by { decide }
theorem gdn_channels : (2 * 16 + 48) * 128 = 10240 := by { decide }
theorem gdn_rows_cover : 4 * (128 / 4) = 128 := by { decide }

theorem gdn_row_lt (rg r : Nat) (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by { omega }

def guCodes : Nat := (1280 * 640)
def downOff : Nat := (guCodes)
def guScaleOff : Nat := (downOff + 2560 * 160)
def downScaleOff : Nat := (guScaleOff + 1280 * 40 * 2)
def blobBytes : Nat := (downScaleOff + 2560 * 10 * 2)

theorem blob_is_1382400 : blobBytes = 1382400 := by decide

theorem gu_codes_in_blob (row : Nat) (h : row < 1280) : row * 640 + 640 ≤ guCodes := by
  unfold guCodes; omega

theorem down_codes_in_blob (row : Nat) (h : row < 2560) :
    downOff + row * 160 + 160 ≤ guScaleOff := by
  unfold guScaleOff downOff guCodes; omega

end Baloon.Prefill
