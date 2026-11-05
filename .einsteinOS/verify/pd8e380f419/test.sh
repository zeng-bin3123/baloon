#!/bin/bash
# Unit pd8e380f419: port table (6 names defined in the file named) + platform invariants + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pd8e_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/pd8e380f419.tsv <<'PY' || rc=1
import re, sys
want = {"Completion","DirectFile","now_us","LockResult","lock_resident","unlock_resident"}
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add(f[1])
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(f[3]) + r'(?![A-Za-z0-9_])', t):
        print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 6 or seen != want: print("port table wrong: %d rows, names %s" % (n, sorted(seen))); bad += 1
sys.exit(1 if bad else 0)
PY
# the test runs in its own scratch dir (it creates temp files in the cwd)
bt() { (cd "$1" && $HOSTCXX $HF -Iinclude -Itests/pk_unit3 tests/k23_platform/test_platform.cpp -o "$2"); }
runt() { local d="$1" b="$2"; mkdir -p "$d"; (cd "$d" && "$b"); }
echo "== unit test =="
bt "$ROOT" "$OUT/t23" || { echo "BUILD FAILED (unit test)"; exit 1; }
runt "$OUT/run" "$OUT/t23" | tee "$OUT/t23.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 23 TESTS PASSED" "$OUT/t23.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! bt "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if runt "$M/run" "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
F=include/baloon/direct_file.hpp
R=include/baloon/resident_memory.hpp
mutate noaccept   $F 's|if (fd_ < 0 \|\| !acceptable(offset, (uint64_t)(uintptr_t)buf, length)) return false;|if (fd_ < 0) return false;|'
mutate wrongoff   $F 's|(off_t)offset);\n||; 0,/(off_t)offset);/s//(off_t)(offset + 1));/'
mutate lifo       $F 's|for (uint64_t i = 0; i < n; ++i) out\[i\] = q_\[i\];|for (uint64_t i = 0; i < n; ++i) out[i] = q_[q_.size() - 1 - i];|'
mutate nodrain    $F 's|q_.erase(q_.begin(), q_.begin() + (std::ptrdiff_t)n);||'
mutate sizefield  $F 's|fd_ = fd; size_ = (uint64_t)st.st_size;|fd_ = fd; size_ = (uint64_t)st.st_size + 1;|'
mutate clockms    $F 's|duration_cast<std::chrono::microseconds>|duration_cast<std::chrono::milliseconds>|'
mutate clockwall  $F 's|steady_clock::now|system_clock::now|;s|std::chrono::system_clock::now().time_since_epoch()).count();|std::chrono::system_clock::now().time_since_epoch()).count() * 0 + (int64_t)(::rand() % 1000);|'
mutate noclose    $F 's|fd_ = -1; size_ = 0; direct_ = false; q_.clear();|fd_ = -1; direct_ = false;|'
mutate nullok     $R 's|if (p == nullptr) return false;|if (p == nullptr) return true;|'
mutate nowrap     $R 's|return n <= ~a;|return true;|'
mutate zerofail   $R 's|if (n == 0) return true;|if (n == 0) return false;|'
mutate bytes0     $R 's|r.ok = true; r.bytes = n;|r.ok = true; r.bytes = 0;|'
mutate errlost    $R 's|if (::mlock(p, (size_t)n) != 0) { r.err = errno; return r; }|if (::mlock(p, (size_t)n) != 0) { return r; }|'
mutate nolock     $R 's|if (::mlock(p, (size_t)n) != 0) { r.err = errno; return r; }||'
mutate unlockbad  $R 's|if (::munlock(p, (size_t)n) != 0) { r.err = errno; return r; }|if (::munlock(p, (size_t)n) == 0) { r.err = EPERM; return r; }|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
