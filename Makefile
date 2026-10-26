# ===========================================================================
# Baloon Makefile — Iluvatar BI-V100 (ivcore10) build
# ===========================================================================
# Usage:
#   make              — build device_info executable + libbaloon_core.so
#   make test         — build and run smoke test
#   make clean        — remove build artifacts
#
# Override on command line:
#   make CXX=/path/to/clang++ ARCH=ivcore11
# ===========================================================================

# --- Toolchain detection ---
COREX_CXX    := /usr/local/corex/bin/clang++
COREX_PATH   := /usr/local/corex
ARCH         ?= ivcore10

# Detect CoreX — use -e (exists) + readlink to handle symlinks
HAS_COREX := $(shell if [ -e $(COREX_CXX) ]; then echo 1; else echo 0; fi)

# --- Directories ---
SRCDIR     = src
INCDIR     = include
BUILDDIR   = build

ifeq ($(HAS_COREX),1)
  # ========== GPU build (Iluvatar BI-V100) ==========
  CXX       := $(COREX_CXX)
  CUDAPATH  := $(COREX_PATH)
  CUFLAGS   := -x cuda --cuda-gpu-arch=$(ARCH) --cuda-path=$(CUDAPATH) \
               -std=c++17 -O2 \
               -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ \
               -I$(INCDIR) -I$(CUDAPATH)/include
  LDFLAGS   := -L$(CUDAPATH)/lib64 -lcudart -lpthread
  SOFLAGS   := $(CUFLAGS) -O3 -fPIC -shared \
               -cl-single-precision-constant \
               -mllvm --bonus-inst-threshold=0
  BUILD_MODE := gpu
  DEVICE_SRC := $(SRCDIR)/core/device_info.cu
else
  # ========== Host-only stub (no GPU, no CUDA) ==========
  CXX       := g++
  CUFLAGS   := -std=c++17 -O2 -DBALOON_PLATFORM_CPU=1 -I$(INCDIR)
  LDFLAGS   := -lpthread
  SOFLAGS   := $(CUFLAGS) -O3 -fPIC -shared
  BUILD_MODE := host-stub
  DEVICE_SRC := $(SRCDIR)/core/device_info_stub.cpp
endif

# --- Targets ---
DEVICE_INFO_EXE := $(BUILDDIR)/device_info
CORE_SO         := $(BUILDDIR)/libbaloon_core.so

.PHONY: all test clean dirs info

all: dirs info $(DEVICE_INFO_EXE) $(CORE_SO)

dirs:
	@mkdir -p $(BUILDDIR)

info:
	@echo "Build mode : $(BUILD_MODE)"
	@echo "CXX        : $(CXX)"
	@echo "HAS_COREX  : $(HAS_COREX)"

$(DEVICE_INFO_EXE): $(DEVICE_SRC) $(INCDIR)/baloon/config.h | dirs
	$(CXX) $(CUFLAGS) $< -o $@ $(LDFLAGS)

$(CORE_SO): $(DEVICE_SRC) $(INCDIR)/baloon/config.h | dirs
	$(CXX) $(SOFLAGS) -DBALOON_LIB_ONLY $< -o $@ $(LDFLAGS)

ifeq ($(HAS_COREX),1)
test: all
	@echo "--- Running device_info smoke test (GPU) ---"
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $(DEVICE_INFO_EXE)
	@echo "--- DONE ---"
else
test: all
	@echo "--- Running host stub test (no GPU) ---"
	$(DEVICE_INFO_EXE)
	@echo "--- DONE ---"
endif

clean:
	rm -rf $(BUILDDIR)
