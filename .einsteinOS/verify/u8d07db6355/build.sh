#!/usr/bin/env bash
# Compile the whole project: every src/baloon/*.cpp plus every tests/test_*.cpp (one binary each).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/corex/bin:$PATH"
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
  if command -v clang++ >/dev/null 2>&1; then CXX=clang++; else CXX=g++; fi
fi
OUT="$ROOT/build/tests"
rm -rf "$OUT"; mkdir -p "$OUT"
FLAGS=(-std=c++17 -O1 -Wall -Wextra -Werror -I include)
SRCS=()
if compgen -G "src/baloon/*.cpp" >/dev/null; then SRCS=(src/baloon/*.cpp); fi
shopt -s nullglob
TESTS=(tests/test_*.cpp)
[ ${#TESTS[@]} -gt 0 ] || { echo "no tests found" >&2; exit 1; }
for t in "${TESTS[@]}"; do
  name="$(basename "$t" .cpp)"
  echo "[build] $name"
  "$CXX" "${FLAGS[@]}" "$t" "${SRCS[@]}" -o "$OUT/$name"
done
echo "build OK"
