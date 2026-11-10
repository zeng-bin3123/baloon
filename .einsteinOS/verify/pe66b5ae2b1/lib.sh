# Shared by build.sh and test.sh of unit pe66b5ae2b1: compiles the device runtime (src/rt), its four CLIs and this unit's test.
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -O2"
LIBS="-L$COREX/lib64 -lcudart -lpthread"
RT_SRCS=(src/rt/device.cu src/rt/pinned.cu src/rt/kernels.cu src/rt/graph.cpp src/rt/layout_check.cpp src/rt/dense_bank.cpp src/rt/embed_head.cpp
         src/rt/device_main.cpp src/rt/concurrent_main.cpp src/rt/overlap_main.cpp src/rt/load_main.cpp src/core/weights.cpp
         tests/k62_rt_dev/test_rt_dev.cu)

# build_rt <root> <objdir> [only-file]: objects + executables t62, rt_device, rt_concurrent, rt_overlap, rt_load in <objdir>.
# With only-file, objects already in <objdir> are kept except the one compiled from that source.
build_rt() {
  local R="$1" O="$2" ONLY="${3:-}" f o
  mkdir -p "$O"
  local flags="-I$R/include $CUFLAGS"          # the tree under test wins over the original include dir
  for f in "${RT_SRCS[@]}"; do
    o="$O/$(echo "$f" | sed 's#/#__#g').o"
    if [ -n "$ONLY" ] && [ "$f" != "$ONLY" ] && [ -f "$o" ]; then continue; fi
    echo "[cuda] $f"
    "$CLANG" $flags -c "$R/$f" -o "$o" || return 1
    if [ "$f" = src/rt/device_main.cpp ]; then   # library flavour (human() only) for the unit test
      "$CLANG" $flags -DRT_DEVICE_MAIN_DISABLED -c "$R/$f" -o "$O/device_main_lib.o" || return 1
    fi
  done
  local p="$O/src__rt__"
  local RTOBJ="${p}device.cu.o ${p}pinned.cu.o ${p}kernels.cu.o ${p}graph.cpp.o ${p}layout_check.cpp.o ${p}dense_bank.cpp.o ${p}embed_head.cpp.o"
  local W="$O/src__core__weights.cpp.o"
  "$CLANG" "$O/tests__k62_rt_dev__test_rt_dev.cu.o" "$O/device_main_lib.o" "${p}device.cu.o" -o "$O/t62" $LIBS || return 1
  "$CLANG" "${p}device_main.cpp.o" $RTOBJ -o "$O/rt_device" $LIBS || return 1
  "$CLANG" "${p}concurrent_main.cpp.o" $RTOBJ -o "$O/rt_concurrent" $LIBS || return 1
  "$CLANG" "${p}overlap_main.cpp.o" $RTOBJ -o "$O/rt_overlap" $LIBS || return 1
  "$CLANG" "${p}load_main.cpp.o" $W $RTOBJ -o "$O/rt_load" $LIBS || return 1
}
