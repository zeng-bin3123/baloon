#!/bin/bash
# Build + run the peer-store probe on BI-V100 (needs >=2 cards). Run on the machine that has /usr/local/corex.
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/probe_peer_a2a.cu /tmp/probe_peer_a2a
CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 120 /tmp/probe_peer_a2a
