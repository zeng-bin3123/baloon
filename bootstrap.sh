#!/bin/bash
# Task: "项目在 zeng-bin3123/baloon，分支是 main"
# The BI-V100 machine's /home/claude/baloon held only ENV.md + README.md (no tools/, no .git main history), so
# tools/preflight.sh --pull could not run there: the fix lived in the files that were missing. This script needs
# nothing from the checkout, so it can be fetched on its own and run against a directory in ANY state:
#   missing dir                      -> git clone -b main
#   git clone of baloon on main      -> fetch + fast-forward (refuses if uncommitted changes)
#   git clone of baloon on another   -> the kernel's build tree ("dev=owner/name, branch=<branch>": the BI-V100
#     branch (e.g. einsteinOS/dev-*)    machine's /home/claude/baloon is einsteinOS/dev-*, src/ilu_*.cu + tests/k*,
#                                       no tools/). It is NOT touched: main goes into a linked worktree
#                                       <dir>-main (git worktree, shares objects), tools run from there.
#   anything else (plain files, other origin, broken .git)
#                                    -> moved aside to <dir>.pre-bootstrap-<UTC time>, fresh clone, and every file
#                                       of the old dir that differs from / is absent in main is listed (nothing deleted)
# Then runs tools/preflight.sh from the new checkout. Exit: 0 ready, 1 git/checkout problem, 2 toolchain (preflight).
# Usage on the machine (one line, no prior checkout needed):
#   curl -fsSL https://raw.githubusercontent.com/zeng-bin3123/baloon/main/bootstrap.sh | bash -s -- /home/claude/baloon
# GH_TOKEN (optional) is used only in the clone URL for this run and is stripped from the saved origin.
set -u
REPO="zeng-bin3123/baloon"; BRANCH="main"
DIR="$(realpath -m "${1:-$PWD/baloon}")"
PUB="https://github.com/$REPO.git"
URL="$PUB"; [ -n "${GH_TOKEN:-}" ] && URL="https://x-access-token:$GH_TOKEN@github.com/$REPO.git"

# Every run is also appended to <dir>.bootstrap.log (beside the trees, so neither gets dirty): a mirrored call that
# comes back "exit=0" with empty output can be read back with tools/status.sh or `tail <dir>.bootstrap.log`.
LOG="$DIR.bootstrap.log"
mkdir -p "$(dirname "$DIR")" && exec > >(tee -a "$LOG") 2>&1
echo "=== bootstrap $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) dir=$DIR"
say() { echo "[bootstrap] $*"; }
clone() {
  git clone -q -b "$BRANCH" "$URL" "$DIR" || { say "FAIL: clone of $REPO@$BRANCH into $DIR"; exit 1; }
  git -C "$DIR" remote set-url origin "$PUB"
}
is_baloon() {
  git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1
  [ "$(git -C "$1" rev-parse --show-toplevel 2>/dev/null)" = "$1" ] || return 1
  case "$(git -C "$1" remote get-url origin 2>/dev/null)" in *zeng-bin3123/baloon*) return 0;; esac
  return 1
}

if [ ! -e "$DIR" ] || { [ -d "$DIR" ] && [ -z "$(ls -A "$DIR")" ]; }; then
  rmdir "$DIR" 2>/dev/null; say "no checkout at $DIR -> cloning"; clone
elif is_baloon "$DIR" && [ "$(git -C "$DIR" symbolic-ref -q --short HEAD 2>/dev/null || echo DETACHED)" != "$BRANCH" ]; then
  CUR="$(git -C "$DIR" symbolic-ref -q --short HEAD 2>/dev/null || echo "detached $(git -C "$DIR" rev-parse --short HEAD)")"
  WT="$DIR-$BRANCH"
  say "$DIR is a baloon clone on '$CUR' (the build tree) -> left as is; $BRANCH goes to worktree $WT"
  git -C "$DIR" fetch -q "$URL" "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" || { say "FAIL: fetch"; exit 1; }
  git -C "$DIR" worktree prune
  if [ -e "$WT/.git" ] && [ "$(cd "$WT" && realpath "$(git rev-parse --git-common-dir 2>/dev/null)")" = "$(cd "$DIR" && realpath "$(git rev-parse --git-common-dir)")" ]; then
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || { git -C "$WT" status --short --untracked-files=no; say "FAIL: uncommitted changes in $WT"; exit 1; }
    git -C "$WT" merge -q --ff-only "origin/$BRANCH" || { say "FAIL: $WT $BRANCH diverged from origin/$BRANCH"; exit 1; }
  elif [ -e "$WT" ]; then
    say "FAIL: $WT exists and is not a worktree of $DIR; move it away and rerun"; exit 1
  else
    if git -C "$DIR" show-ref -q --verify "refs/heads/$BRANCH"; then
      git -C "$DIR" worktree add -q "$WT" "$BRANCH" || exit 1
      git -C "$WT" merge -q --ff-only "origin/$BRANCH" || { say "FAIL: local $BRANCH diverged from origin/$BRANCH"; exit 1; }
    else
      git -C "$DIR" worktree add -q -b "$BRANCH" "$WT" "origin/$BRANCH" || exit 1
    fi
  fi
  say "build tree $DIR still on '$CUR' ($(git -C "$DIR" log --oneline -1 | cut -c1-60))"
  say "run tools as: python3 $WT/tools/feature_issues.py --next ; python3 $WT/tools/ref_check.py"
  DIR="$WT"
elif is_baloon "$DIR"; then
  say "$DIR is a baloon clone -> updating to origin/$BRANCH"
  if [ -n "$(git -C "$DIR" status --porcelain --untracked-files=no)" ]; then
    git -C "$DIR" status --short --untracked-files=no
    say "FAIL: uncommitted changes above; commit/stash them, then rerun"; exit 1
  fi
  git -C "$DIR" fetch -q "$URL" "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" || { say "FAIL: fetch"; exit 1; }
  git -C "$DIR" checkout -q "$BRANCH" 2>/dev/null || git -C "$DIR" checkout -q -b "$BRANCH" "origin/$BRANCH" || exit 1
  git -C "$DIR" merge -q --ff-only "origin/$BRANCH" || { say "FAIL: local $BRANCH diverged from origin/$BRANCH"; exit 1; }
else
  BAK="$DIR.pre-bootstrap-$(date -u +%Y%m%dT%H%M%SZ)"
  say "$DIR is not a baloon clone ($(git -C "$DIR" remote get-url origin 2>/dev/null || echo 'no git origin')) -> moving to $BAK"
  mv "$DIR" "$BAK" || { say "FAIL: cannot move $DIR"; exit 1; }
  clone
  n_same=0; n_diff=0
  while IFS= read -r -d '' f; do
    rel="${f#$BAK/}"
    if [ -f "$DIR/$rel" ] && cmp -s "$f" "$DIR/$rel"; then n_same=$((n_same+1))
    else n_diff=$((n_diff+1)); say "kept only in backup: $rel$([ -f "$DIR/$rel" ] && echo ' (differs from main)')"; fi
  done < <(find "$BAK" -path "$BAK/.git" -prune -o -type f -print0)
  say "old dir: $n_same files identical to main, $n_diff only in $BAK"
fi

say "HEAD $(git -C "$DIR" log --oneline -1 | cut -c1-90)"
say "files: $(git -C "$DIR" ls-files | wc -l) tracked, tools/: $(ls "$DIR/tools" 2>/dev/null | tr '\n' ' ')"
[ -x "$DIR/tools/preflight.sh" ] || { say "FAIL: tools/preflight.sh missing after bootstrap"; exit 1; }
exec "$DIR/tools/preflight.sh"
