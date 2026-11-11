#!/bin/bash
# Task: "把编译环境（clang）、硬件那些硬信息和shell跑的命令md先从zeng-bin3123/baloon main 拿取"
# Single source of truth: the "## Compiler Toolchain" block of ENV.md. Every build_*.sh sources this file
# instead of hard-coding clang++ path / arch / flags, so editing ENV.md changes every build.
# Exports: IX_CXX IX_ARCH IX_CUDAPATH IX_CF (compile) IX_LF (link) IX_SOF (shared lib) and ix_build SRC OUT [extra flags].
# IX_DRYRUN=1 prints the commands instead of running them (works without corex).
_ix_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ix_md="${IX_ENV_MD:-$_ix_root/ENV.md}"
[ -f "$_ix_md" ] || { echo "ix_env: $_ix_md not found" >&2; return 2 2>/dev/null || exit 2; }
# first ``` block after "## Compiler Toolchain"; join "\" continuations; strip "# comments"
_ix_block="$(awk '/^## Compiler Toolchain/{f=1;next} f&&/^```/{if(i){exit}i=1;next} f&&i' "$_ix_md" \
  | sed 's/[[:space:]]#.*$//' | sed -e ':a' -e '/\\$/N; s/\\\n[[:space:]]*/ /; ta')"
_ix_get() { printf '%s\n' "$_ix_block" | sed -n "s/^$1[[:space:]]*=[[:space:]]*//p" | head -1 | sed 's/[[:space:]]*$//'; }
IX_CXX="${IX_CXX:-$(_ix_get CXX)}"
IX_ARCH="$(_ix_get ARCH)"
IX_CUDAPATH="$(_ix_get CUDAPATH)"
IX_CF="$(_ix_get CFLAGS)"
IX_LF="$(_ix_get LDFLAGS)"
IX_SOF="$(_ix_get SOFLAGS)"
for _v in IX_CXX IX_ARCH IX_CUDAPATH IX_CF IX_LF; do
  [ -n "${!_v}" ] || { echo "ix_env: $_v not found in $_ix_md" >&2; return 2 2>/dev/null || exit 2; }
done
export IX_CXX IX_ARCH IX_CUDAPATH IX_CF IX_LF IX_SOF
# ix_build SRC OUT [extra flags...]  -> $IX_CXX $IX_CF -std=c++17 -D__ILUVATAR__ -Isrc -Itests [extra] SRC -o OUT $IX_LF
ix_build() {
  local src="$1" out="$2"; shift 2
  local cmd=("$IX_CXX" $IX_CF -std=c++17 -D__ILUVATAR__ -I"$_ix_root/src" -I"$_ix_root/tests" "$@" "$src" -o "$out" $IX_LF)
  if [ -n "$IX_DRYRUN" ]; then echo "${cmd[*]}"; return 0; fi
  [ -x "$IX_CXX" ] || { echo "corex toolchain not found at $IX_CXX (from $_ix_md)"; return 2; }
  "${cmd[@]}"
}
if [ "${BASH_SOURCE[0]}" = "$0" ]; then  # run directly: show what ENV.md resolves to
  for _v in IX_CXX IX_ARCH IX_CUDAPATH IX_CF IX_LF IX_SOF; do echo "$_v=${!_v}"; done
fi
