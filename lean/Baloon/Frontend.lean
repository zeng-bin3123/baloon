-- Baloon.Frontend: thinking-budget rank, partial-tag hold-back and newline
-- stripping invariants for the streaming text frontend.
-- Independent implementation proving the same mathematical facts as Ref.Frontend.

namespace Baloon.Frontend

def budgetRank (n : Int) : Nat := (if n < 2048 then 0 else if n < 8192 then 1 else 2)

theorem budgetRank_mono (n m : Int) (h : n ≤ m) : budgetRank n ≤ budgetRank m := by
  unfold budgetRank; split_ifs <;> omega

theorem budgetRank_lt (n : Int) : budgetRank n < 3 := by
  unfold budgetRank; split_ifs <;> omega

def holdOne (text tag : List Nat) : Nat :=
  (List.range tag.length).foldl
    (fun b n => if 1 ≤ n ∧ n ≤ text.length ∧ text.drop (text.length - n) = tag.take n
                then max b n else b) (0)

theorem fold_bound (tl : Nat) (f : Nat → Nat → Nat) (hf : ∀ b n, f b n ≤ max b n) :
    ∀ (l : List Nat) (b : Nat), (∀ n ∈ l, n < tl) → b < tl → l.foldl f b < tl := by
  intro l
  induction l with
  | nil => intro b _ hb; simpa using hb
  | cons n l ih =>
    intro b hl hb
    simp only [List.foldl_cons]
    apply ih
    · intro x hx; exact hl x (List.mem_cons_of_mem _ hx)
    · have h1 := hf b n
      have h2 : n < tl := hl n List.mem_cons_self
      omega

theorem holdOne_lt (text tag : List Nat) (h : 0 < tag.length) : holdOne text tag < tag.length := by
  unfold holdOne
  apply fold_bound tag.length (fun b n => if 1 ≤ n ∧ n ≤ text.length ∧
      text.drop (text.length - n) = tag.take n then max b n else b)
  · intro b n; split_ifs <;> omega
  · intro n hn; exact List.mem_range.mp hn
  · exact h

theorem hold_split (buf : List Nat) (keep : Nat) :
    buf.take (buf.length - keep) ++ buf.drop (buf.length - keep) = buf := by exact List.take_append_drop _ _

theorem emitted_le (buf : List Nat) (keep : Nat) :
    (buf.take (buf.length - keep)).length ≤ buf.length := by { simp [List.length_take] }

def dropLead : List Nat → List Nat
  | 10 :: rest => (rest)
  | l => (l)

def dropTrail (l : List Nat) : List Nat :=
  if (l.getLast? = some 10) then l.dropLast else l

theorem dropLead_length (l : List Nat) : (dropLead l).length ≤ l.length := by
  cases l with
  | nil => simp [dropLead]
  | cons h t => cases Nat.decEq h 10 with
    | isTrue he => simp [dropLead, he]
    | isFalse hne => simp [dropLead, show h ≠ 10 from hne]

end Baloon.Frontend
