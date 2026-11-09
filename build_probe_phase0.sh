#!/bin/bash
# Phase 0: capability probes for BI-V100 (ivcore10). Run on the machine that has /usr/local/corex.
# Each probe is compiled separately; COMPILE_FAIL means the intrinsic/header does not build on ivcore10.
# Usage: ./build_probe_phase0.sh   -> writes phase0_results_<time>.log (paste it back)
CXX=/usr/local/corex/bin/clang++
[ -x "$CXX" ] || { echo "corex toolchain not found at $CXX"; exit 2; }
CF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -I/usr/local/corex/include -O2"
LF="-L/usr/local/corex/lib64 -lcudart -lpthread"
LOG=phase0_results_$(date +%Y%m%d_%H%M%S).log
{
  echo "### git: $(git log --oneline -1)"; echo "### host: $(hostname) $(date -u)"
  $CXX --version 2>&1 | head -2
  echo "### ixsmi"; ixsmi 2>&1 | head -30
  echo "### topo"; ixsmi topo -m 2>&1 | head -20
  for P in DEVPROPS SHFL64 BALLOT64 DP4A MATCH_ANY MMA THREADFENCE_SYSTEM NANOSLEEP CUDA_GRAPH HBM_BW; do
    echo; echo "=== $P"
    BIN=/tmp/p0_$P
    if ! $CXX $CF -DPROBE_$P tests/probe_phase0.cu -o $BIN $LF 2> /tmp/p0_$P.err; then
      echo "PROBE $P COMPILE_FAIL"; head -5 /tmp/p0_$P.err; continue
    fi
    CUDA_VISIBLE_DEVICES=0 timeout 60 $BIN 2>&1 | tail -8; echo "exit=${PIPESTATUS[0]}"
  done
  echo; echo "=== PEER (existing probe)"; [ -x ./build_peer_probe.sh ] && ./build_peer_probe.sh 2>&1 | tail -30
} 2>&1 | tee "$LOG"
echo; echo "Log written to $LOG - paste its contents back."
