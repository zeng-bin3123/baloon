#!/bin/bash
# Task: "项目在 zeng-bin3123/baloon，分支是 main" + "把编译环境（clang）、硬件那些硬信息和shell跑的命令md先从 baloon main 拿取"
# Before any build on the BI-V100 machine, prove the checkout IS baloon main and the environment matches ENV.md:
#   1. git: this is a clone of zeng-bin3123/baloon, on main, not behind origin/main  (--pull fast-forwards it)
#   2. files: every tests/*.cu, tests/*.cpp named by build_*.sh / run_all_tests.sh exists in this checkout
#   3. toolchain: the clang++ named in ENV.md exists and runs; ixsmi present; GPU count
# Exit 0 = ready, 1 = checkout stale or files missing (fix: --pull), 2 = toolchain missing.
# Usage: tools/preflight.sh [--pull]       (run_all_tests.sh calls it first)
cd "$(dirname "$0")/.." || exit 2
rc=0
echo "### preflight: git"
url=$(git remote get-url origin 2>/dev/null)
case "$url" in *zeng-bin3123/baloon*) echo "origin: $url";; *) echo "FAIL: not a clone of zeng-bin3123/baloon (origin='$url')"; rc=1;; esac
br=$(git rev-parse --abbrev-ref HEAD 2>/dev/null); [ "$br" = main ] || { echo "WARN: on branch '$br', not main"; }
if git fetch -q origin main 2>/dev/null; then
  behind=$(git rev-list --count HEAD..origin/main); ahead=$(git rev-list --count origin/main..HEAD)
  echo "HEAD $(git log --oneline -1 | cut -c1-80) | behind origin/main: $behind, ahead: $ahead"
  if [ "$behind" -gt 0 ]; then
    if [ "$1" = "--pull" ]; then git merge -q --ff-only origin/main && echo "fast-forwarded to $(git log --oneline -1 | cut -c1-60)" || { echo "FAIL: cannot fast-forward (local changes?)"; rc=1; }
    else echo "FAIL: checkout is $behind commits behind origin/main - run tools/preflight.sh --pull"; rc=1; fi
  fi
else
  echo "WARN: cannot fetch origin (offline?) - staleness not checked"
fi
echo "### preflight: files named by build scripts"
missing=0
for s in build_*.sh; do
  for f in $(grep -oE '(tests|src)/[A-Za-z0-9_./-]+\.(cu|cpp|h)' "$s" | sort -u); do
    [ -f "$f" ] || { echo "MISSING $f (needed by $s)"; missing=$((missing+1)); }
  done
done
for s in $(grep -oE 'build_[A-Za-z0-9_]+\.sh' run_all_tests.sh | sort -u); do [ -f "$s" ] || { echo "MISSING $s (listed in run_all_tests.sh)"; missing=$((missing+1)); }; done
[ $missing -eq 0 ] && echo "all files present" || { echo "FAIL: $missing missing - checkout is not baloon main (tools/preflight.sh --pull)"; rc=1; }
echo "### preflight: toolchain from ENV.md"
source tools/ix_env.sh || exit 2
if [ -x "$IX_CXX" ]; then "$IX_CXX" --version 2>&1 | head -1
else echo "FAIL: $IX_CXX (from ENV.md) not found"; [ $rc -eq 0 ] && rc=2; fi
if command -v ixsmi >/dev/null; then echo "GPUs: $(ixsmi -L 2>/dev/null | grep -c GPU)"; else echo "WARN: ixsmi not in PATH"; fi
echo "### preflight: exit $rc"
exit $rc
