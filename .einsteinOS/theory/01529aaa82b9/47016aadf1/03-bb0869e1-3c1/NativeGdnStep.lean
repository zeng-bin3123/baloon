/-!
# MODULE: native_gdn_step
Addressing and validation arithmetic of the llama.cpp-derived native gated-delta-net step kernel:
each warp owns one (head, column) and keeps four rows (r*32+lane) in registers.

## Source semantics
Block (head, 1, z) with 32x4 threads; col = z*4 + ty; rows i = r*32 + lane for r in 0..3 cover 0..127.
Requires S=128, h_v % h_k = 0, h_v <= 65535, aligned and pairwise-disjoint spans.

## Form template identifiers
S -> 128; h_k/h_v -> key/value heads; lane -> threadIdx.x; col -> blockIdx.z*blockDim.y+threadIdx.y.
-/
namespace Strata.NativeGdn

/-- Row owned by (register r, lane). -/
def rowOf (r lane : Nat) : Nat := r * 32 + lane

/-- Rows stay inside the 128-row state. -/
theorem row_lt (r lane : Nat) (h1 : r < 4) (h2 : lane < 32) : rowOf r lane < 128 := by
  unfold rowOf; omega

/-- (register, lane) pairs map injectively onto rows. -/
theorem row_inj (r l r' l' : Nat) (h1 : r < 4) (h2 : l < 32) (h3 : r' < 4) (h4 : l' < 32)
    (e : rowOf r l = rowOf r' l') : r = r' ∧ l = l' := by
  unfold rowOf at e; omega

/-- Every row is owned by some (register, lane). -/
theorem row_cover (i : Nat) (h : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ rowOf r l = i :=
  ⟨i / 32, i % 32, by omega, by omega, by unfold rowOf; omega⟩

/-- Column from block z and thread y; z < 32, y < 4 gives col < 128. -/
theorem col_lt (z y : Nat) (h1 : z < 32) (h2 : y < 4) : z * 4 + y < 128 := by omega

/-- Columns partition: (z,y) pairs map injectively. -/
theorem col_inj (z y z' y' : Nat) (h2 : y < 4) (h4 : y' < 4) (e : z * 4 + y = z' * 4 + y') :
    z = z' ∧ y = y' := by omega

/-- Every column below 128 has a (z, y) pair with z < S/4 = 32. -/
theorem col_cover (c : Nat) (h : c < 128) : ∃ z y, z < 32 ∧ y < 4 ∧ z * 4 + y = c :=
  ⟨c / 4, c % 4, by omega, by omega, by omega⟩

/-- Each stored four-register shard holds 4 * 32 = 128 rows. -/
theorem shard_size : 4 * 32 = 128 := by decide

/-- Grid z extent S/4 times block y extent 4 equals S. -/
theorem grid_z : 128 / 4 * 4 = 128 := by decide

/-- Divisible head counts give an integral group factor. -/
theorem head_group (hk hv : Nat) (hp : 0 < hk) (h : hv % hk = 0) : hk * (hv / hk) = hv := by
  have := Nat.div_add_mod hv hk
  omega

/-- The modulo head pairing is within range. -/
theorem q_head_lt (head hk : Nat) (hp : 0 < hk) : head % hk < hk := Nat.mod_lt _ hp

/-- Interval overlap as tested by the host. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- Non-overlap yields separation of the two intervals. -/
theorem not_overlap_sep (a an b bn : Nat) (h : ¬ overlap a an b bn) : b + bn ≤ a ∨ a + an ≤ b := by
  unfold overlap at h; omega

/-- Separated intervals do not overlap. -/
theorem sep_not_overlap (a an b bn : Nat) (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap; omega

/-- A valid span (bytes <= UMAX - addr) cannot wrap around. -/
theorem valid_no_wrap (addr bytes umax : Nat) (h : bytes ≤ umax - addr) (h2 : addr ≤ umax) :
    addr + bytes ≤ umax := by omega

/-- Alignment of a float span: a 4-aligned address plus a multiple of 4 stays 4-aligned. -/
theorem aligned_add (a n : Nat) (h1 : a % 4 = 0) (h2 : n % 4 = 0) : (a + n) % 4 = 0 := by omega

end Strata.NativeGdn
