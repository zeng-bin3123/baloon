#!/bin/bash
# build_ix_kernels.sh — Build the BI-V100 CUDA kernel library
#
# Usage: ./build_ix_kernels.sh [clean|test|bench]
#
# Prerequisites:
#   - Iluvatar CoreX SDK at /usr/local/corex
#   - clang++ (clang-16) compiler

set -euo pipefail

CUDAPATH="${CUDAPATH:-/usr/local/corex}"
CXX="${CXX:-${CUDAPATH}/bin/clang++}"
ARCH="${ARCH:-ivcore10}"

CFLAGS="-x cuda --cuda-gpu-arch=${ARCH} --cuda-path=${CUDAPATH} \
        -std=c++17 -O3 \
        -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ -D__ILUVATAR_DIAG__ \
        -cl-single-precision-constant -fPIC -mllvm --bonus-inst-threshold=0 \
        -I${CUDAPATH}/include -Iinclude"

LDFLAGS="-L${CUDAPATH}/lib64 -lcudart"

SRCDIR="src/strata/kernels/cuda"
OUTDIR="build"
LIBNAME="libix_kernels.so"

SOURCES=(
    "${SRCDIR}/ix_infccl_allreduce.cu"
    "${SRCDIR}/ix_native_mmvq.cu"
    "${SRCDIR}/ix_native_router.cu"
    "${SRCDIR}/ix_moe_elementwise.cu"
    "${SRCDIR}/ix_bf16_gemv.cu"
    "${SRCDIR}/ix_gdn.cu"
)

case "${1:-build}" in
    clean)
        echo "Cleaning..."
        rm -rf "${OUTDIR}" "${LIBNAME}" tests/test_unit7
        exit 0
        ;;
    test)
        echo "Building tests..."
        mkdir -p "${OUTDIR}"
        ${CXX} ${CFLAGS} tests/test_unit7.cu \
            -o tests/test_unit7 \
            -L. -lix_kernels ${LDFLAGS} \
            -Wl,-rpath,'$ORIGIN/..' -Wl,-rpath,${CUDAPATH}/lib64
        echo "Running tests..."
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
            LD_LIBRARY_PATH="$(pwd):${CUDAPATH}/lib64" \
            tests/test_unit7
        exit 0
        ;;
    bench)
        echo "Running benchmarks..."
        CUDA_VISIBLE_DEVICES=0 \
            LD_LIBRARY_PATH="$(pwd):${CUDAPATH}/lib64" \
            tests/test_unit7
        exit 0
        ;;
esac

echo "Building ${LIBNAME} for ${ARCH}..."
echo "  CXX: ${CXX}"
echo "  CUDA: ${CUDAPATH}"

mkdir -p "${OUTDIR}"

# Compile each .cu to .o
OBJECTS=()
for src in "${SOURCES[@]}"; do
    obj="${OUTDIR}/$(basename "${src}" .cu).o"
    echo "  CC  ${src} -> ${obj}"
    ${CXX} ${CFLAGS} -c "${src}" -o "${obj}"
    OBJECTS+=("${obj}")
done

# Link shared library
echo "  LD  ${LIBNAME}"
${CXX} ${CFLAGS} -shared -o "${LIBNAME}" "${OBJECTS[@]}" ${LDFLAGS}

echo ""
echo "Built: ${LIBNAME} ($(stat -c%s ${LIBNAME} 2>/dev/null || stat -f%z ${LIBNAME}) bytes)"
echo ""
echo "Next steps:"
echo "  ./build_ix_kernels.sh test     # run GPU tests"
echo "  ./build_ix_kernels.sh bench    # run benchmarks"
echo ""
echo "Python usage:"
echo "  import sys; sys.path.insert(0, 'src/strata/kernels/cuda')"
echo "  import ix_kernels"
echo "  ix_kernels.load('./libix_kernels.so')"
