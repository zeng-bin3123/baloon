/-!
# MODULE: chat_request_normalizer
SOURCES: serve/frontend.py
Request-side half of the frontend: reasoning-effort normalisation (OpenAI / Anthropic spellings onto the
template's levels), thinking-budget bucketing, image-part extraction and role mapping.

## Source semantics
* Effort strings map onto {off, low, medium, xhigh}; `high` means `xhigh`; unknown values are a 400.
* `off` is expressed as `enable_thinking=false` and carries no `reasoning_effort`.
* `budget_tokens`: < 2048 low, < 8192 medium, else xhigh (monotone).
* Image parts keep their order; unknown parts are dropped; image sources are listed in prompt order.
* `developer` role becomes `system`.
* A `chat_template_kwargs` `enable_thinking=false` overrides any effort already chosen.

## Form template identifiers
- `effort_kwargs` -> `effort_normalize`; `budget_effort` -> `budget_bucket`; `_parts_of`/`images_of` -> `image_part_extract`;
  `openai_to_messages` -> `openai_request_adapter`; `anthropic_to_messages` -> `anthropic_request_adapter`.
-/

namespace Strata.ChatRequestNormalizer

-- § effort levels

/-- The template's thinking levels. `off` means thinking disabled. -/
inductive Thinking where
  | off
  | low
  | medium
  | xhigh
  deriving DecidableEq

/-- The EFFORT dictionary; `none` is an unknown spelling (a 400). -/
def effortOf : String → Option Thinking
  | "none" => some Thinking.off
  | "off" => some Thinking.off
  | "minimal" => some Thinking.off
  | "disabled" => some Thinking.off
  | "false" => some Thinking.off
  | "low" => some Thinking.low
  | "medium" => some Thinking.medium
  | "high" => some Thinking.xhigh
  | "xhigh" => some Thinking.xhigh
  | "max" => some Thinking.xhigh
  | "maximum" => some Thinking.xhigh
  | _ => none

/-- `high` is the template's `xhigh`. -/
theorem high_is_xhigh : effortOf "high" = some Thinking.xhigh := by
  rfl

/-- The five "off" spellings all disable thinking. -/
theorem off_spellings : ∀ s ∈ ["none", "off", "minimal", "disabled", "false"],
    effortOf s = some Thinking.off := by
  decide

/-- An unknown spelling is rejected rather than guessed. -/
theorem unknown_effort_rejected : effortOf "ultra" = none := by
  rfl

/-- Template kwargs: `enable_thinking` flag and `reasoning_effort` level. -/
structure Kw where
  enable : Option Bool := none
  effort : Option Thinking := none
  deriving DecidableEq

/-- A level as template kwargs. -/
def levelKw : Thinking → Kw
  | Thinking.off => { enable := some false, effort := none }
  | t => { enable := none, effort := some t }

/-- `off` never carries a reasoning effort (the two kwargs are mutually exclusive). -/
theorem off_has_no_effort : (levelKw Thinking.off).effort = none ∧ (levelKw Thinking.off).enable = some false := by
  decide

/-- A positive level never disables thinking. -/
theorem level_not_disabled (t : Thinking) (h : t ≠ Thinking.off) : (levelKw t).enable = none := by
  cases t <;> simp [levelKw] at h ⊢

-- § thinking budget

/-- Rank of a level for ordering. -/
def rank : Thinking → Nat
  | Thinking.off => 0
  | Thinking.low => 1
  | Thinking.medium => 2
  | Thinking.xhigh => 3

/-- Anthropic `budget_tokens` bucketed: under 2K low, under 8K medium, else xhigh. -/
def budgetEffort (n : Nat) : Thinking :=
  if n < 2048 then Thinking.low else if n < 8192 then Thinking.medium else Thinking.xhigh

/-- A bigger budget never gives a lower level. -/
theorem budget_monotone (n m : Nat) (h : n ≤ m) : rank (budgetEffort n) ≤ rank (budgetEffort m) := by
  unfold budgetEffort
  by_cases a : n < 2048
  · by_cases b : m < 2048
    · simp [a, b, rank]
    · by_cases c : m < 8192
      · simp [a, b, c, rank]
      · simp [a, b, c, rank]
  · have b : ¬ m < 2048 := by omega
    by_cases c : n < 8192
    · by_cases d : m < 8192
      · simp [a, b, c, d, rank]
      · simp [a, b, c, d, rank]
    · have d : ¬ m < 8192 := by omega
      simp [a, b, c, d, rank]

/-- A budget budget never produces `off`: a budget always leaves thinking enabled. -/
theorem budget_never_off (n : Nat) : budgetEffort n ≠ Thinking.off := by
  unfold budgetEffort
  by_cases a : n < 2048 <;> by_cases b : n < 8192 <;> simp [a, b]

/-- Bucket boundaries: 2047 is low, 2048 medium, 8191 medium, 8192 xhigh. -/
theorem budget_boundaries :
    budgetEffort 2047 = Thinking.low ∧ budgetEffort 2048 = Thinking.medium ∧
    budgetEffort 8191 = Thinking.medium ∧ budgetEffort 8192 = Thinking.xhigh := by
  decide

-- § chat_template_kwargs override

/-- `enable_thinking=false` from `chat_template_kwargs` replaces whatever was chosen. -/
def applyDisable (kw : Kw) (v : Bool) : Kw :=
  if v then kw else { enable := some false, effort := none }

/-- After a disable override thinking is off and no effort remains. -/
theorem disable_clears_effort (kw : Kw) : (applyDisable kw false).effort = none ∧ (applyDisable kw false).enable = some false := by
  simp [applyDisable]

/-- A truthy `enable_thinking` leaves the kwargs unchanged. -/
theorem enable_true_noop (kw : Kw) : applyDisable kw true = kw := by
  simp [applyDisable]

/-- A `reasoning_effort` key only applies when thinking was not explicitly disabled. -/
def applyEffort (kw new : Kw) : Kw :=
  if kw.enable = none then
    { enable := match new.enable with | some b => some b | none => kw.enable,
      effort := match new.effort with | some e => some e | none => kw.effort }
  else kw

/-- Once thinking is disabled a later effort key cannot re-enable it. -/
theorem disabled_is_sticky (kw new : Kw) (h : kw.enable = some false) : applyEffort kw new = kw := by
  simp [applyEffort, h]

-- § image parts

/-- A content part: text, an image with its source string, or something unrecognised. -/
inductive Part where
  | text (s : String)
  | image (src : String)
  | other

/-- Keep text and image items in order, drop anything else. -/
def toItems : List Part → List Part
  | [] => []
  | Part.other :: r => toItems r
  | p :: r => p :: toItems r

/-- Image sources in prompt order. -/
def imgs : List Part → List String
  | [] => []
  | Part.image s :: r => s :: imgs r
  | _ :: r => imgs r

/-- Dropping unrecognised parts never changes which images are sent, nor their order. -/
theorem imgs_toItems (ps : List Part) : imgs (toItems ps) = imgs ps := by
  induction ps with
  | nil => rfl
  | cons p r ih => cases p <;> simp [toItems, imgs, ih]

/-- The item list is never longer than the input. -/
theorem toItems_length_le (ps : List Part) : (toItems ps).length ≤ ps.length := by
  induction ps with
  | nil => simp [toItems]
  | cons p r ih => cases p <;> simp [toItems] <;> omega

/-- Normalising twice equals normalising once. -/
theorem toItems_idempotent (ps : List Part) : toItems (toItems ps) = toItems ps := by
  induction ps with
  | nil => rfl
  | cons p r ih => cases p <;> simp [toItems, ih]

-- § roles

/-- OpenAI `developer` is the template's `system`. -/
def roleMap (r : String) : String := if r = "developer" then "system" else r

/-- Role mapping is idempotent. -/
theorem roleMap_idempotent (r : String) : roleMap (roleMap r) = roleMap r := by
  by_cases h : r = "developer"
  · simp [roleMap, h]
  · simp [roleMap, h]

/-- A `developer` message becomes `system`. -/
theorem developer_becomes_system : roleMap "developer" = "system" := by
  simp [roleMap]

end Strata.ChatRequestNormalizer
