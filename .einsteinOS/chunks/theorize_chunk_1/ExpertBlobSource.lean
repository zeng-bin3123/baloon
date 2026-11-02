/-!
# MODULE: expert_blob_source
Index arithmetic of the file-backed and arena-backed expert sources: `(layer, expert) -> blob`.

## Source semantics
Blob index is `layer * n_expert + expert`; an expert id at or past `n_expert` must be refused separately because the
flat index would silently alias the next layer. The arena is one max-blob longer than the file so a whole-slot copy may
start at any expert. GGUF chunked loading stays inside the per-layer expert range.

## Form template identifiers
- FileExpertSource::blob / ArenaExpertSource::blob -> flat
- ArenaExpertSource::device_alias                  -> sliceOff
- load_experts_gguf                                -> gguf_expert_in_range
-/
namespace Strata.ExpertBlobSource

-- § flat index
def flat (ne l e : Nat) : Nat := l * ne + e

/-- In-range (layer, expert) gives an in-range flat index. -/
theorem flat_lt {ne nl l e : Nat} (hl : l < nl) (he : e < ne) : flat ne l e < nl * ne := by
  unfold flat
  have := Nat.mul_le_mul_right ne (show l + 1 ≤ nl by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- A smaller layer gives a smaller flat index when experts are in range. -/
theorem flat_lt_of_layer_lt {ne l l' e e' : Nat} (h : l < l') (he : e < ne) : flat ne l e < flat ne l' e' := by
  unfold flat
  have := Nat.mul_le_mul_right ne (show l + 1 ≤ l' by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- The flat index is injective on in-range experts. -/
theorem flat_inj {ne l l' e e' : Nat} (he : e < ne) (he' : e' < ne) (h : flat ne l e = flat ne l' e') :
    l = l' ∧ e = e' := by
  rcases Nat.lt_trichotomy l l' with hlt | heq | hgt
  · have := flat_lt_of_layer_lt (e' := e') hlt he
    omega
  · subst heq
    unfold flat at h
    omega
  · have := flat_lt_of_layer_lt (e' := e) hgt he'
    omega

/-- Why the separate `expert >= n_expert` check exists: the unchecked flat index aliases the next layer. -/
theorem flat_alias (ne : Nat) : flat ne 0 ne = flat ne 1 0 := by simp [flat]

-- § arena slices
def sliceOff (bb e : Nat) : Nat := e * bb

/-- An expert's blob lies inside its layer's registration slice. -/
theorem slice_blob_in_slice {bb ne e : Nat} (he : e < ne) : sliceOff bb e + bb ≤ ne * bb := by
  unfold sliceOff
  have := Nat.mul_le_mul_right bb (show e + 1 ≤ ne by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The arena is one max-blob longer than the data, so a max-blob copy from any blob start stays inside it. -/
theorem arena_slack {off bb mb total : Nat} (h : off + bb ≤ total) (hb : bb ≤ mb) : off + mb ≤ total + mb := by
  omega

-- § GGUF chunked loading
/-- With `per` bytes per expert, a chunk of `b` experts starting at expert `a` that fits in the tensor
yields only expert ids below `n_expert`. -/
theorem gguf_expert_in_range {per a b ne k : Nat} (hp : 0 < per) (h : per * a + per * b ≤ per * ne)
    (hk : k < b) : a + k < ne := by
  have h2 : per * (a + b) ≤ per * ne := by rw [Nat.mul_add]; exact h
  have h3 := Nat.le_of_mul_le_mul_left h2 hp
  omega

end Strata.ExpertBlobSource
