# Chunk 10 — Axle Verification Status

## Environment
- `axle.axiommath.ai` is not in the network egress allowlist
- DNS resolution fails from both sandbox and local machine
- Proofs are written with standard Lean 4 tactics and ready for verification

## Theorems and Expected Results

| Theorem | Tactic | Expected |
|---------|--------|----------|
| `scratch_alignment` | `omega` | ✓ (Nat division/multiplication inequality) |
| `down_output_row_bound` | `omega` | ✓ (linear arithmetic) |
| `scaled_clamp_symmetric` | `simp; split_ifs <;> omega` | ✓ (if-then-else branching) |
| `lut_code_range` | `omega` | ✓ (modular arithmetic bound) |
| `batch_grid_pos` | `Nat.mul_pos + omega` | ✓ (product positivity) |
| `q8_0_clamp_bounds` | `simp [max_def, min_def]; split_ifs <;> omega` | ✓ (clamp bounds) |

## Theorems marked proof_needed (require custom axioms)
- `warp_sum_butterfly` — floating-point butterfly reduction
- `chunk_softmax_invariant` — exp positivity
- `merge_logsumexp` — log-sum-exp rescaling identity
- `rope_neox_pair_invertible` — rotation matrix invertibility
- `softmax_sum_positive` — exp sum positivity
- `order_key_preserves_order` — IEEE 754 bit-level monotonicity
- `grouped_bitwise_per_entry` — shared-memory staging preserves bytes
- `hadd_order_matters` — floating-point non-associativity
