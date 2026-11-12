#!/bin/bash
# Task: "你可以看看这台机器上执行的工具调用" - the machine's mirrored calls came back "[pid 15 exit=0]" and
# "[pid 16 exit=0]" with NO output, which measures nothing. This prints, read-only and always, one "key: value" line
# per fact, so a single call on the machine returns its real state:
#   build tree <dir> (branch, HEAD, dirty), main worktree <dir>-main (HEAD, behind origin/main, tools present),
#   toolchain (ENV.md clang++, ixsmi GPU count), last bootstrap run from <dir>.bootstrap.log.
# Needs nothing from a checkout: curl -fsSL https://raw.githubusercontent.com/zeng-bin3123/baloon/main/tools/status.sh | bash -s -- /home/claude/baloon
# Exit: 0 READY = main tree has tools, is not behind, and ENV.md's clang++ runs; 1 NOT_READY (reasons listed).
# Never writes anything.
set -u
DIR="$(realpath -m "${1:-/home/claude/baloon}")"
say() { printf '%-16s %s\n' "$1:" "$2"; }
gitinfo() {  # dir -> "branch HEAD subject [dirty N]" or why not
  [ -e "$1" ] || { echo "missing"; return; }
  git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "not a git tree ($(ls -A "$1" | wc -l) entries)"; return; }
  local b d; b="$(git -C "$1" symbolic-ref -q --short HEAD || echo detached)"
  d="$(git -C "$1" status --porcelain --untracked-files=no | wc -l)"
  echo "$b $(git -C "$1" log -1 --format='%h %ad %s' --date=short | cut -c1-70)$([ "$d" -gt 0 ] && echo " [dirty $d]")"
}
say time "$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
say build_tree "$DIR: $(gitinfo "$DIR")"
say origin "$(git -C "$DIR" remote get-url origin 2>/dev/null || echo none)"
MAIN=""
if [ "$(git -C "$DIR" symbolic-ref -q --short HEAD 2>/dev/null)" = main ]; then MAIN="$DIR"
elif [ -e "$DIR-main/.git" ]; then MAIN="$DIR-main"; fi
why=""
if [ -n "$MAIN" ]; then
  say main_tree "$MAIN: $(gitinfo "$MAIN")"
  behind="$(git -C "$MAIN" rev-list --count HEAD..origin/main 2>/dev/null || echo '?')"
  say behind_main "$behind (as of last fetch)"
  tools="$(cd "$MAIN/tools" 2>/dev/null && ls *.py *.sh 2>/dev/null | tr '\n' ' ')"
  say tools "${tools:-none}"
  [ -f "$MAIN/tools/feature_issues.py" ] || why="$why tools-missing"
  [ "$behind" = 0 ] || why="$why main-behind"
else
  say main_tree "none: run bootstrap.sh $DIR"; why="$why no-main-tree"
fi
ENVMD="${MAIN:-$DIR}/ENV.md"
CXX="$(grep -m1 -E '^CXX +=' "$ENVMD" 2>/dev/null | sed 's/^CXX *= *//')"
if [ -n "$CXX" ] && [ -x "$CXX" ]; then say clang "$CXX: $("$CXX" --version 2>&1 | head -1)"
else say clang "${CXX:-ENV.md not found}: not executable"; why="$why no-clang"; fi
if command -v ixsmi >/dev/null; then say gpus "$(ixsmi -L 2>/dev/null | grep -c GPU) ($(ixsmi -L 2>/dev/null | head -1 | cut -c1-60))"
else say gpus "ixsmi not in PATH"; fi
if [ -f "$DIR.bootstrap.log" ]; then
  say last_bootstrap "$(grep '^=== bootstrap' "$DIR.bootstrap.log" | tail -1 | cut -c5-)"
  tail -n 4 "$DIR.bootstrap.log" | sed 's/^/                 | /'
else say last_bootstrap "no $DIR.bootstrap.log"; fi
if [ -z "$why" ]; then say status READY; exit 0; fi
say status "NOT_READY:$why"; exit 1
