#!/bin/bash
# Runs every Phase 1 test on the machine that has /usr/local/corex and writes ONE log to paste back.
# Usage: git pull origin main && ./run_all_tests.sh   (uses GPU 0 only, builds are single-threaded)
LOG=phase1_results_$(date +%Y%m%d_%H%M%S).log
{
  echo "### git: $(git log --oneline -1)"
  echo "### host: $(hostname)  $(date -u)"
  ls -l /usr/local/corex/bin/clang++ 2>&1
  /usr/local/corex/bin/clang++ --version 2>&1 | head -2
  ixsmi 2>&1 | head -20
  for s in build_probe.sh build_shared_expert_test.sh build_moe_combine_test.sh build_bf16_gemv_test.sh build_gdn_test.sh build_router_test.sh build_router_debug.sh; do
    echo; echo "================ $s"; [ -x "./$s" ] || { echo "MISSING $s (git pull?)"; continue; }
    ./$s 2>&1 | tail -60; echo "exit=${PIPESTATUS[0]}"
  done
} 2>&1 | tee "$LOG"
echo; echo "Log written to $LOG - paste its contents back."
