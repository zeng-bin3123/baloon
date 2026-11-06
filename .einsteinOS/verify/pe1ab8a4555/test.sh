#!/bin/bash
# Unit pe1ab8a4555: port table + device runtime tests (arena/poison, pinned loads, replay graphs, layout, dense/embed/head,
# gemv) on the real GPUs + the four CLIs end to end + mutation self-tests. Exit 0 only if everything passes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pe1ab_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -O2"
rc=0

echo "== port table =="
python3 - .einsteinOS/port/pe1ab8a4555.tsv <<'PY' || rc=1
import re, sys
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\r\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add((f[0], f[1]))
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    name = f[3]; esc = re.escape(name)
    if "::" in name: pat = r'(^|[^\w:])' + esc + r'\s*\('
    else: pat = (r'(struct|class|enum)\s+' + esc + r'\b|using\s+' + esc + r'\s*=|#\s*define\s+' + esc + r'\b|'
                 r'\b' + esc + r'\s*(\(|=|;)')
    if not re.search(pat, t, re.M): print("port: %s not defined in %s" % (name, f[2])); bad += 1
if n != 59 or len(seen) != 59: print("port table has %d rows (%d distinct), expected 59" % (n, len(seen))); bad += 1
sys.exit(1 if bad else 0)
PY

# build_rt <root> <objdir> [only-file]: compiles the runtime + unit test + CLIs of <root> into <objdir>. With only-file, every
# object already in <objdir> is kept except the one(s) compiled from that source (a header rebuilds everything).
build_rt() {
  local R="$1" O="$2" ONLY="${3:-}" f o fl
  mkdir -p "$O"
  local flags="-I$R/include $CUFLAGS"   # the tree under test must win over the original include dir
  local srcs=(src/rt/device.cu src/rt/pinned.cu src/rt/kernels.cu src/rt/graph.cpp src/rt/layout_check.cpp src/rt/dense_bank.cpp src/rt/embed_head.cpp
              src/rt/device_main.cpp src/rt/concurrent_main.cpp src/rt/overlap_main.cpp src/rt/load_main.cpp tests/k29_rt/test_rt.cpp src/core/weights.cpp)
  case "$ONLY" in include/*) ONLY="" ;; esac
  for f in "${srcs[@]}"; do
    o="$O/$(echo "$f" | sed 's#/#__#g').o"
    if [ -n "$ONLY" ] && [ "$f" != "$ONLY" ] && [ -f "$o" ]; then continue; fi
    "$CLANG" $flags -c "$R/$f" -o "$o" || return 1
    case "$f" in src/rt/device_main.cpp|tests/k29_rt/test_rt.cpp)   # library flavour of device_main (human) for the unit test
      if [ "$f" = src/rt/device_main.cpp ]; then "$CLANG" $flags -DRT_DEVICE_MAIN_DISABLED -c "$R/$f" -o "$O/device_main_lib.o" || return 1
      else "$CLANG" $flags -DRT_DEVICE_MAIN_DISABLED -c "$R/$f" -o "$o" || return 1; fi ;;
    esac
  done
  local pre="$O/src__rt__"
  local RTOBJ="${pre}device.cu.o ${pre}pinned.cu.o ${pre}kernels.cu.o ${pre}graph.cpp.o ${pre}layout_check.cpp.o ${pre}dense_bank.cpp.o ${pre}embed_head.cpp.o"
  local W="$O/src__core__weights.cpp.o"
  local L="-L$COREX/lib64 -lcudart -lpthread"
  "$CLANG" "$O/tests__k29_rt__test_rt.cpp.o" "$O/device_main_lib.o" $W $RTOBJ -o "$O/t29" $L || return 1
  "$CLANG" "${pre}device_main.cpp.o" $RTOBJ -o "$O/rt_device" $L || return 1
  "$CLANG" "${pre}concurrent_main.cpp.o" $RTOBJ -o "$O/rt_concurrent" $L || return 1
  "$CLANG" "${pre}overlap_main.cpp.o" $RTOBJ -o "$O/rt_overlap" $L || return 1
  "$CLANG" "${pre}load_main.cpp.o" $W $RTOBJ -o "$O/rt_load" $L || return 1
  return 0
}

echo "== build =="
build_rt "$ROOT" "$OUT/base" || { echo "BUILD FAILED"; exit 1; }

echo "== unit test (opens the GPU) =="
"$OUT/base/t29" | tee "$OUT/t29.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 29 TESTS PASSED" "$OUT/t29.log" || { echo "suite did not report PASSED"; rc=1; }

echo "== CLIs end to end =="
B="$OUT/base"
"$B/rt_device" --selftest > "$OUT/dev.log" 2>&1; [ $? -eq 0 ] || { echo "rt_device --selftest failed"; cat "$OUT/dev.log"; rc=1; }
cat "$OUT/dev.log"
grep -q "^device 0: .*warp" "$OUT/dev.log" || { echo "rt_device printed no device 0 line"; rc=1; }
grep -q "selftest: ok" "$OUT/dev.log" || { echo "rt_device selftest line missing"; rc=1; }
"$B/rt_device" --bogus >/dev/null 2>&1; [ $? -eq 2 ] || { echo "rt_device: bad flag must exit 2"; rc=1; }
"$B/rt_device" --require 9999 >/dev/null 2>&1; [ $? -eq 1 ] || { echo "rt_device: unmet --require must exit 1"; rc=1; }
"$B/rt_concurrent" > "$OUT/conc.log" 2>&1; [ $? -eq 0 ] || { echo "rt_concurrent failed"; rc=1; }
cat "$OUT/conc.log"; grep -q "CONCURRENT OK" "$OUT/conc.log" || { echo "rt_concurrent did not report OK"; rc=1; }
"$B/rt_concurrent" --devices 0 >/dev/null 2>&1; [ $? -eq 1 ] || { echo "rt_concurrent: zero devices must exit 1"; rc=1; }
"$B/rt_concurrent" --nope >/dev/null 2>&1; [ $? -eq 2 ] || { echo "rt_concurrent: bad flag must exit 2"; rc=1; }
"$B/rt_overlap" > "$OUT/ovl.log" 2>&1; [ $? -eq 0 ] || { echo "rt_overlap failed"; rc=1; }
cat "$OUT/ovl.log"; grep -q "OVERLAP OK" "$OUT/ovl.log" || { echo "rt_overlap did not report OK"; rc=1; }
"$B/rt_overlap" --chunks 1 >/dev/null 2>&1; [ $? -eq 1 ] || { echo "rt_overlap: one chunk must exit 1"; rc=1; }
"$B/rt_load" --selftest > "$OUT/load.log" 2>&1; [ $? -eq 0 ] || { echo "rt_load --selftest failed"; rc=1; }
cat "$OUT/load.log"; grep -q "LOAD OK" "$OUT/load.log" || { echo "rt_load did not report OK"; rc=1; }
"$B/rt_load" >/dev/null 2>&1; [ $? -eq 2 ] || { echo "rt_load: no args must exit 2"; rc=1; }
"$B/rt_load" /nonexistent/dir >/dev/null 2>&1; [ $? -eq 1 ] || { echo "rt_load: missing dir must exit 1"; rc=1; }
# a model whose layout is wrong must be rejected: shrink one norm vector in a generated selftest model
LD="$OUT/badmodel"; mkdir -p "$LD"
python3 - "$LD" <<'PY' || rc=1
import sys, os
d = sys.argv[1]
# 2 layers, n_embd 4: layer 0 GDN, layer 1 GDN; missing ssm_out on layer 1
shapes = ["blk.0.attn_norm 4 0", "blk.0.ffn_norm 4 0", "blk.0.ssm_in 8 4", "blk.0.ssm_out 4 8",
          "blk.1.attn_norm 4 0", "blk.1.ffn_norm 4 0", "blk.1.ssm_in 8 4"]
idx = []; off = 0
for s in shapes:
    n, r, c = s.split(); sz = int(r) * (int(c) or 1) * 2
    idx.append("%s 4 %d %d %d 0 0 0 0" % (n, off, sz, sz)); off += sz
open(os.path.join(d, "shapes.txt"), "w").write("\n".join(shapes) + "\n")
open(os.path.join(d, "index.txt"), "w").write("\n".join(idx) + "\n")
open(os.path.join(d, "weights.bin"), "wb").write(bytes(off))
PY
"$B/rt_load" "$LD" --geometry 2,4,1,2,1 >"$OUT/badload.log" 2>&1; [ $? -eq 1 ] || { echo "rt_load: broken layout must exit 1"; rc=1; }
grep -q "layout check failed: missing blk.1.ssm_out" "$OUT/badload.log" || { echo "rt_load: wrong layout message"; cat "$OUT/badload.log"; rc=1; }
# the same directory with the missing tensor added must load
python3 - "$LD" <<'PY' || rc=1
import os, sys
d = sys.argv[1]
open(os.path.join(d, "shapes.txt"), "a").write("blk.1.ssm_out 4 8\n")
open(os.path.join(d, "index.txt"), "a").write("blk.1.ssm_out 4 %d 64 64 0 0 0 0\n" % os.path.getsize(os.path.join(d, "weights.bin")))
open(os.path.join(d, "weights.bin"), "ab").write(bytes(64))
PY
"$B/rt_load" "$LD" --geometry 2,4,1,2,1 >"$OUT/goodload.log" 2>&1; [ $? -eq 0 ] || { echo "rt_load: repaired layout must load"; cat "$OUT/goodload.log"; rc=1; }
grep -q "LOAD OK" "$OUT/goodload.log" || { echo "rt_load: repaired layout did not report OK"; rc=1; }

echo "== mutation self-tests =="
mutate() {   # mutate name file sed-expression
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  cp -r "$OUT/base" "$M/obj"
  if ! (build_rt "$M" "$M/obj" "$file" >/dev/null 2>&1); then echo "mutant $name failed to build"; rc=1; return; fi
  if timeout 600 "$M/obj/t29" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
  rm -rf "$M"
}
P=src/rt/pinned.cu; D=src/rt/device.cu; G=src/rt/graph.cpp; Y=src/rt/layout_check.cpp; N=src/rt/dense_bank.cpp; E=src/rt/embed_head.cpp; K=src/rt/kernels.cu
mutate fnvprime   $P 's|h \*= kFnvPrime64;|h *= 3;|'
mutate bounds     $P 's|(i < r ? 1 : 0)|0|'
mutate chksum     $P 's|st.checksum = fnv1a64(d, m, st.checksum);|st.checksum = fnv1a64(d, m);|'
mutate rangedst   $P 's|dst += r.second;|dst += 0;|'
mutate moved      $P 's|a.bytes() / chunk \* chunk;|a.bytes();|'
mutate pageslice  $P 's|if (slices > pages) slices = pages;||'
mutate arenaalign $D 's|(used_ + kArenaAlign - 1) / kArenaAlign \* kArenaAlign|used_|'
mutate arenawrap  $D 's|bytes > cap_ - at|bytes + at > cap_|'
mutate poisonk    $D 's|p\[i\] = v;|p[i] = (uint8_t)(v + 1);|'
mutate poisonrst  $D 's|if (poison \&\& base_) {|if (false) {|'
mutate ordinal    $D 's|if (ordinal < 0 \|\| ordinal >= n)|if (ordinal >= n)|'
mutate grorder    $G 's|for (auto\& op : ops_) {|for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) { auto\& op = *it;|'
mutate grstop     $G 's|return graph_fail("op failed"); }|graph_fail("op failed"); }|'
mutate grreplace  $G 's|g_\[name\] = std::move(g);|g_.emplace(name, std::move(g));|'
mutate grwait     $G 's|if (!ran_ \|\| failed_ \|\| !e1_) return -1.0;|if (!e1_) return -1.0;|'
mutate grready    $G 's|if (!ready_) return graph_fail("launch before end()");||'
mutate layvec     $Y 's|if (it->second.cols != 0) return layout_fail(why, n + " must be a vector");||'
mutate layrows    $Y 's|if (w.rows \&\& s.rows != w.rows)|if (false)|'
mutate layforeign $Y 's|for (const char\* s : foreign_gdn) if (idx.count(v.name(s)))|for (const char* s : foreign_gdn) if (false)|'
mutate layrange   $Y 's|if (layer < 0 \|\| layer >= g.n_layers)|if (layer < 0)|'
mutate layperiod  include/baloon/rt_layout.hpp 's|l % qsa_period == qsa_period - 1|l % qsa_period == 0|'
mutate eligible   $N 's|r.engine_kind == 1 \&\&|(r.engine_kind == 1 \|\| r.engine_kind == 3) \&\&|'
mutate shapechk   $N 's|rows \* cols \* 2 != r.dst_bytes|false|'
mutate embbound   $E 's|token >= vocab_|token > vocab_|'
mutate headrun    $E 's|bf16_gemv(static_cast<const uint16_t\*>(dev_.get()), dx, vocab_, n_, dy)|bf16_gemv(static_cast<const uint16_t*>(dev_.get()), dx, n_, vocab_, dy)|'
mutate gatherzero $K 's|valid ? table\[(int64_t)t \* n + c\] : 0.f|valid ? table[(int64_t)t * n + c] : 1.f|'
mutate gemvred    $K 's|part\[threadIdx.x\] += part\[threadIdx.x + s\];|part[threadIdx.x] -= part[threadIdx.x + s];|'
mutate human      src/rt/device_main.cpp 's|v /= 1024.0;|v /= 1000.0;|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED"
exit $rc
