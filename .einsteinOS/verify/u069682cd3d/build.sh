#!/bin/bash
# Builds the unit-4 host test (header-only modules) and syntax-checks every unit-4 header as a translation unit.
set -e
cd "$(dirname "$0")/../../.."
CXX=${CXX:-g++}; command -v "$CXX" >/dev/null || CXX=/usr/local/corex/bin/clang++
mkdir -p build
for h in ix_u4_common ix_qsa_decode ix_qsa_select ix_quantize_act ix_router_top10 ix_verify_kernels ix_sampler ix_s2_expert ix_sgemv; do
  echo "#include \"../src/$h.h\"" > build/hdr_$h.cpp
  "$CXX" -std=c++17 -O1 -Wall -Werror -fsyntax-only build/hdr_$h.cpp
done
"$CXX" -std=c++17 -O1 -Wall -o build/test_unit4 tests/test_unit4.cpp
echo "build ok"
