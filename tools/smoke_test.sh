#!/bin/bash
# ==========================================================================
# smoke_test.sh — Build and run the Baloon smoke test on BI-V100
# ==========================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Building ==="
make clean
make -j$(nproc) 2>&1

echo ""
echo "=== Running smoke test ==="
make test
echo ""
echo "=== Python binding test ==="
PYTHONPATH=python:$PYTHONPATH python3 -c "
import baloon
n = baloon.device_count()
print(f'Python: {n} devices found')
for i in range(min(n, 4)):
    info = baloon.query_device(i)
    print(f'  GPU {i}: {info[\"name\"]} — {info[\"vram_gib\"]} GiB, warp={info[\"warp_size\"]}')
baloon.smoke_test(0)
print('Python smoke test: PASS')
"

