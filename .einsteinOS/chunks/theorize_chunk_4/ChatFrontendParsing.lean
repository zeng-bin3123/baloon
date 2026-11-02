/-!
# MODULE: chat_frontend_parsing
Text-only request/response layer between API clients and the engine: reasoning-effort normalisation,
Anthropic thinking-budget mapping, tool-call parameter value trimming, and the "hold back a partial tag"
rule of the incremental output parser.

## Source semantics
* Every client spelling of a reasoning effort maps onto {low, medium, xhigh} or "thinking disabled";
  an unknown spelling is rejected (HTTP 400), never a crash.
* `budget_tokens` maps monotonically: < 2048 → low, < 8192 → medium, otherwise xhigh.
* The streaming parser never emits a partial tag: it holds back the longest suffix of the buffer that is a
  proper prefix of the tag being looked for (so the held length is strictly smaller than the tag length).
* A tool-call parameter value loses at most one leading and one trailing newline.
* At end of generation an unterminated tool call is flushed as content, never as a tool_call event.

## Form template identifiers
* `effortKwargs` ↔ `effort_kwargs`; `budgetEffort` ↔ `budget_effort`
* `hold` ↔ `OutputParser._hold`; `trimValue` ↔ the newline stripping in `parse_tool_call`
* `finishKind` ↔ `OutputParser.finish` (state → event kind)
-/

namespace Strata.Frontend

-- § reasoning effort

/-- The template's three thinking levels. -/
inductive Level
  | low | medium | xhigh
  deriving DecidableEq, Repr

/-- Rank of a level, used to state monotonicity. -/
def Level.rank : Level → Nat
  | .low => 0
  | .medium => 1
  | .xhigh => 2

/-- The spellings of a reasoning effort a client may send. -/
inductive EffortKey
  | none_ | off | minimal | disabled | false_
  | low | medium | high | xhigh | max | maximum
  | unknown
  deriving DecidableEq, Repr

/-- Template kwargs produced by an effort: nothing, thinking disabled, or a level. -/
inductive Kw
  | empty
  | noThink
  | level (l : Level)
  deriving DecidableEq, Repr

/-- The `EFFORT` table: `some none` means "thinking off", `none` means unknown spelling. -/
def effortLevel : EffortKey → Option (Option Level)
  | .none_ => some none
  | .off => some none
  | .minimal => some none
  | .disabled => some none
  | .false_ => some none
  | .low => some (some .low)
  | .medium => some (some .medium)
  | .high => some (some .xhigh)
  | .xhigh => some (some .xhigh)
  | .max => some (some .xhigh)
  | .maximum => some (some .xhigh)
  | .unknown => none

/-- A reasoning effort → template kwargs, or `none` (a 400) for an unknown value. -/
def effortKwargs (k : EffortKey) : Option Kw :=
  match effortLevel k with
  | none => none
  | some none => some .noThink
  | some (some l) => some (.level l)

/-- "high" means xhigh, the template's default level. -/
theorem effortKwargs_high : effortKwargs .high = some (.level .xhigh) := rfl

/-- "max" and "maximum" are aliases of "high". -/
theorem effortKwargs_max_alias :
    effortKwargs .max = effortKwargs .high ∧ effortKwargs .maximum = effortKwargs .high := by
  exact ⟨rfl, rfl⟩

/-- Every "off" spelling disables thinking. -/
theorem effortKwargs_off_family :
    ∀ k ∈ [EffortKey.none_, .off, .minimal, .disabled, .false_], effortKwargs k = some .noThink := by
  decide

/-- An unknown spelling is rejected. -/
theorem effortKwargs_unknown : effortKwargs .unknown = none := rfl

/-- A key is accepted exactly when it is not the unknown spelling. -/
theorem effortKwargs_isSome_iff (k : EffortKey) :
    (effortKwargs k).isSome = true ↔ k ≠ .unknown := by
  cases k <;> simp [effortKwargs, effortLevel]

-- § thinking budget

/-- Anthropic `budget_tokens` → level: under 2K low, under 8K medium, else xhigh. -/
def budgetEffort (n : Nat) : Level :=
  if n < 2048 then .low else if n < 8192 then .medium else .xhigh

/-- Small budgets map to low. -/
theorem budgetEffort_low {n : Nat} (h : n < 2048) : budgetEffort n = .low := by
  simp [budgetEffort, h]

/-- Middle budgets map to medium. -/
theorem budgetEffort_medium {n : Nat} (h1 : 2048 ≤ n) (h2 : n < 8192) : budgetEffort n = .medium := by
  have h : ¬ n < 2048 := by omega
  simp [budgetEffort, h, h2]

/-- Large budgets map to xhigh. -/
theorem budgetEffort_xhigh {n : Nat} (h : 8192 ≤ n) : budgetEffort n = .xhigh := by
  have h1 : ¬ n < 2048 := by omega
  have h2 : ¬ n < 8192 := by omega
  simp [budgetEffort, h1, h2]

/-- A larger budget never selects a lower level. -/
theorem budgetEffort_mono {n m : Nat} (h : n ≤ m) :
    (budgetEffort n).rank ≤ (budgetEffort m).rank := by
  unfold budgetEffort
  split_ifs <;> simp only [Level.rank] <;> omega

-- § partial tag hold-back

/-- Longest `k ≤ n` such that the first `k` tag symbols are a suffix of the text (0 if none). -/
def holdGo {α : Type} [DecidableEq α] (text tag : List α) : Nat → Nat
  | 0 => 0
  | n + 1 => if (tag.take (n + 1)).isSuffixOf text then n + 1 else holdGo text tag n

/-- Length of the longest suffix of `text` that is a proper prefix of `tag`. -/
def hold {α : Type} [DecidableEq α] (text tag : List α) : Nat :=
  holdGo text tag (tag.length - 1)

/-- The search never exceeds its bound. -/
theorem holdGo_le {α : Type} [DecidableEq α] (text tag : List α) (n : Nat) :
    holdGo text tag n ≤ n := by
  induction n with
  | zero => simp [holdGo]
  | succ n ih =>
    unfold holdGo
    split <;> omega

/-- The held length is a proper prefix length: strictly shorter than a non-empty tag. -/
theorem hold_lt {α : Type} [DecidableEq α] (text tag : List α) (h : tag ≠ []) :
    hold text tag < tag.length := by
  cases tag with
  | nil => exact absurd rfl h
  | cons a t =>
    have := holdGo_le text (a :: t) ((a :: t).length - 1)
    simp only [hold, List.length_cons] at *
    omega

/-- An empty tag holds nothing back. -/
theorem hold_nil_tag {α : Type} [DecidableEq α] (text : List α) : hold text ([] : List α) = 0 := rfl

/-- Worked example: text ending in the first two symbols of the tag holds back two. -/
theorem hold_example : hold [9, 1, 2] [1, 2, 3, 4] = 2 := by decide

/-- Worked example: nothing matching holds nothing back. -/
theorem hold_example_none : hold [9, 8, 7] [1, 2, 3, 4] = 0 := by decide

-- § tool-call value trimming

/-- Drop one leading newline (code 10). -/
def stripLead (v : List Nat) : List Nat :=
  match v with
  | 10 :: r => r
  | _ => v

/-- Drop one trailing newline (code 10). -/
def stripTrail (v : List Nat) : List Nat :=
  match v.reverse with
  | 10 :: r => r.reverse
  | _ => v

/-- A parameter value is stripped of one leading and one trailing newline. -/
def trimValue (v : List Nat) : List Nat := stripTrail (stripLead v)

/-- A leading newline is removed. -/
theorem stripLead_newline (r : List Nat) : stripLead (10 :: r) = r := rfl

/-- Leading stripping never lengthens the value. -/
theorem stripLead_length_le (v : List Nat) : (stripLead v).length ≤ v.length := by
  unfold stripLead
  split <;> simp

/-- Trailing stripping never lengthens the value. -/
theorem stripTrail_length_le (v : List Nat) : (stripTrail v).length ≤ v.length := by
  unfold stripTrail
  split
  · rename_i r h
    have := congrArg List.length h
    simp at this ⊢
    omega
  · exact Nat.le_refl _

/-- Trimming never lengthens a value. -/
theorem trimValue_length_le (v : List Nat) : (trimValue v).length ≤ v.length :=
  Nat.le_trans (stripTrail_length_le _) (stripLead_length_le v)

/-- Trimming removes at most two characters. -/
theorem trimValue_length_ge (v : List Nat) : v.length ≤ (trimValue v).length + 2 := by
  have h1 : v.length ≤ (stripLead v).length + 1 := by
    unfold stripLead
    split <;> simp
  have h2 : (stripLead v).length ≤ (stripTrail (stripLead v)).length + 1 := by
    unfold stripTrail
    split
    · rename_i r h
      have := congrArg List.length h
      simp at this ⊢
      omega
    · omega
  unfold trimValue
  omega

/-- Worked example: `"\nabc\n"` becomes `"abc"`. -/
theorem trimValue_example : trimValue [10, 1, 2, 3, 10] = [1, 2, 3] := by decide

-- § end of generation flush

/-- Parser states. -/
inductive PState
  | reasoning | content | call
  deriving DecidableEq, Repr

/-- Event kinds. -/
inductive EvKind
  | reasoning | content | toolCall
  deriving DecidableEq, Repr

/-- Kind of the flush event at end of generation: an unterminated call is returned as content. -/
def finishKind : PState → EvKind
  | .reasoning => .reasoning
  | .content => .content
  | .call => .content

/-- The final flush never fabricates a tool call. -/
theorem finishKind_ne_toolCall (s : PState) : finishKind s ≠ .toolCall := by
  cases s <;> simp [finishKind]

end Strata.Frontend
