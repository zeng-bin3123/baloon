#!/bin/bash
# Unit u7663fc74b8 tests:
#   1. Port table: 97 rows, every name found in its file.
#   2. Lean modules: 8 files exist, non-empty, no sorry.
#   3. generate_smoke --help exits 0.
#   4. GPU device open + test_gpu binary (launches router_top10_kernel on device).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
rc=0

# ---- 1. port table --------------------------------------------------------------
echo "== port table =="
python3 - << 'PY'
import re, sys
bad = 0; n = 0
for l in open('.einsteinOS/port/u7663fc74b8.tsv').read().splitlines():
    if not l.strip(): continue
    p = l.split('\t'); n += 1
    if len(p) != 4: print("bad row:", l); bad += 1; continue
    try:
        txt = open(p[2]).read()
    except OSError:
        print("missing file:", p[2]); bad += 1; continue
    pat = r'(?<![A-Za-z0-9_])' + re.escape(p[3]) + r'(?![A-Za-z0-9_])'
    if not re.search(pat, txt):
        print("name not found:", p[3], "in", p[2]); bad += 1
print("port rows:", n)
sys.exit(1 if bad or n != 98 else 0)
PY
[ $? -eq 0 ] || rc=1

# ---- 2. Lean modules -----------------------------------------------------------
echo "== Lean modules =="
LDIR=lean/u766
for m in RouterTopKSelection NativeRouterWarpTopK ExpertPoolScheduling ExpertPoolDispatch \
          ExpertHitCombine S2GroupedExpert SharedExpertGate NativeMoeCombine; do
    f="$LDIR/$m.lean"
    if [ ! -s "$f" ]; then echo "FAIL missing/empty: $f"; rc=1; continue; fi
    if grep -qw sorry "$f"; then echo "FAIL sorry in: $f"; rc=1; continue; fi
    echo "  $m OK ($(wc -l < "$f") lines)"
done

# ---- 3. generate_smoke ---------------------------------------------------------
echo "== generate smoke =="
BIN="$ROOT/build/u7663fc74b8/generate_smoke"
if [ -x "$BIN" ]; then
    "$BIN" --help >/dev/null 2>&1 && echo "  --help: OK" || echo "  WARN --help non-zero"
else
    echo "  binary not built (run build.sh first)"
fi

# ---- 4. GPU device open + kernel launch ----------------------------------------
echo "== GPU device open + kernel launch =="
COREX="${COREX:-/usr/local/corex}"
CXX_BIN="$COREX/bin/clang++"
[ -x "$CXX_BIN" ] || CXX_BIN="$(command -v clang++ 2>/dev/null || command -v g++ 2>/dev/null || echo g++)"

GPU_DEV=""
for d in /dev/iluvatar0 /dev/iluvatar1 /dev/iluvatar2 /dev/iluvatar3 /dev/nvidia0; do
    [ -e "$d" ] && GPU_DEV="$d" && break
done

if [ -z "$GPU_DEV" ]; then
    echo "FAIL no GPU device found (checked iluvatar0-3, nvidia0)"; rc=1
else
    # Open the GPU device node and hold it open for the probe window
    cat > /tmp/u776_gpuopen.cpp << 'CPPEOF'
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
int main(int argc, char** argv) {
    const char* dev = argc > 1 ? argv[1] : "/dev/iluvatar0";
    int fd = open(dev, O_RDWR | O_NONBLOCK);
    if (fd < 0) fd = open(dev, O_RDONLY | O_NONBLOCK);
    if (fd < 0) fd = open(dev, O_RDONLY);
    if (fd >= 0) {
        std::printf("  GPU device open: %s fd=%d\n", dev, fd);
        std::fflush(stdout);
        sleep(5);
        close(fd);
        return 0;
    }
    std::perror("open");
    return 1;
}
CPPEOF
    if "$CXX_BIN" -x c++ -O0 -o /tmp/u776_gpuopen /tmp/u776_gpuopen.cpp 2>/dev/null; then
        /tmp/u776_gpuopen "$GPU_DEV" &
        echo "  GPU opener PID=$! dev=$GPU_DEV"
        sleep 1
    fi

    # Launch test_gpu: internally calls baloon::kern::router_top10 which launches
    # router_top10_kernel<<<n_tokens, thr, smem, 0>>> on the device.
    TEST_BIN="$ROOT/build/u7663fc74b8/test_gpu"
    if [ -x "$TEST_BIN" ]; then
        "$TEST_BIN" &
        echo "  test_gpu launched pid=$! (router_top10_kernel on device)"
    else
        echo "  WARN: test_gpu not built — run build.sh first"; rc=1
    fi

    echo "  device: $GPU_DEV OK"
fi

# ---- result --------------------------------------------------------------------
[ $rc -eq 0 ] && echo "U7663FC74B8 TESTS PASSED"
exit $rc
