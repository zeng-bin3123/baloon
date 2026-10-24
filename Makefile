# strata — Makefile for Iluvatar BI-V100 (ivcore10)
CUDAPATH  := /usr/local/corex
CXX       := $(CUDAPATH)/bin/clang++
ARCH      := ivcore10

CXXFLAGS  := -x cuda --cuda-gpu-arch=$(ARCH) --cuda-path=$(CUDAPATH) \
             -std=c++17 -O2 \
             -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ \
             -I$(CUDAPATH)/include -Iinclude \
             -fPIC
LDFLAGS   := -L$(CUDAPATH)/lib64 -lcudart -lpthread

# .so compile flags (optimized)
SOFLAGS   := -x cuda --cuda-gpu-arch=$(ARCH) --cuda-path=$(CUDAPATH) \
             -std=c++17 -O3 \
             -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ -D__ILUVATAR_DIAG__ \
             -cl-single-precision-constant -fPIC -mllvm --bonus-inst-threshold=0 \
             -I$(CUDAPATH)/include -Iinclude \
             -L$(CUDAPATH)/lib64 -lcudart

# ─── Core library sources ──────────────────────────────────────────
CORE_SRCS := src/strata/core/graph.cpp \
             src/strata/core/pinned.cpp \
             src/strata/core/weights.cpp \
             src/strata/core/verify.cpp \
             src/strata/core/mtp.cpp \
             src/strata/core/expert_cache.cpp \
             src/strata/core/expert_source.cpp \
             src/strata/core/layer.cpp \
             src/strata/core/session.cpp

# ─── Host-side kernel dispatchers ──────────────────────────────────
KERN_SRCS := src/strata/kernels/elementwise.cpp \
             src/strata/kernels/elementwise_ops.cpp \
             src/strata/kernels/gdn.cpp \
             src/strata/kernels/gr.cpp \
             src/strata/kernels/ngram.cpp \
             src/strata/kernels/ple.cpp \
             src/strata/kernels/qsa.cpp \
             src/strata/kernels/s_gemv.cpp \
             src/strata/kernels/shared_expert.cpp

# ─── Spec (suffix drafter) ────────────────────────────────────────
SPEC_SRCS := src/strata/spec/suffix_drafter.cpp

# ─── CUDA kernel sources (.cu) ────────────────────────────────────
CUDA_SRCS := src/strata/kernels/cuda/ix_infccl_allreduce.cu \
             src/strata/kernels/cuda/ix_native_mmvq.cu \
             src/strata/kernels/cuda/ix_native_router.cu \
             src/strata/kernels/cuda/ix_moe_elementwise.cu \
             src/strata/kernels/cuda/ix_bf16_gemv.cu \
             src/strata/kernels/cuda/ix_gdn.cu \
             src/strata/kernels/cuda/ix_f16_conversions.cu \
             src/strata/kernels/cuda/ix_activation_quantizer.cu \
             src/strata/kernels/cuda/ix_elementwise.cu \
             src/strata/kernels/cuda/ix_bf16_gemv_ext.cu \
             src/strata/kernels/cuda/ix_s_family_gemv.cu \
             src/strata/kernels/cuda/ix_s2_gemv_q8.cu \
             src/strata/kernels/cuda/ix_s2_gemv_quads.cu \
             src/strata/kernels/cuda/ix_gguf_mmvq.cu \
             src/strata/kernels/cuda/ix_fused_gdn.cu \
             src/strata/kernels/cuda/ix_fused_gr.cu \
             src/strata/kernels/cuda/ix_quantize_act.cu \
             src/strata/kernels/cuda/ix_qsa_decode_attn.cu \
             src/strata/kernels/cuda/ix_iq_kernels.cu \
             src/strata/kernels/cuda/ix_rope.cu \
             src/strata/kernels/cuda/ix_kv_q8.cu \
             src/strata/kernels/cuda/ix_router_top10.cu \
             src/strata/kernels/cuda/ix_qsa_select.cu \
             src/strata/kernels/cuda/ix_sampler.cu

ALL_SRCS  := $(CORE_SRCS) $(KERN_SRCS) $(SPEC_SRCS)
OBJS      := $(ALL_SRCS:.cpp=.o)
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)

# ─── Output libraries ─────────────────────────────────────────────
CORE_LIB   := libstrata_core.so
KERNEL_LIB := libix_kernels.so

# ─── Build targets ─────────────────────────────────────────────────
.PHONY: all clean test lib kernels test1 test2 test2g test2s test3 test4 test5 test6 test7 test8 test9

all: lib kernels

lib: $(CORE_LIB)
kernels: $(KERNEL_LIB)

# Core library (host-side, compiled with -x cuda so needs --cuda-path for linking)
$(CORE_LIB): $(OBJS)
	$(CXX) --cuda-path=$(CUDAPATH) -shared -o $@ $^ $(LDFLAGS)

# CUDA kernel library (GPU-side, optimized)
# NOTE: link step must NOT use -x cuda (SOFLAGS) — that makes clang
# try to compile .o files as CUDA source. Use plain linker flags.
$(KERNEL_LIB): $(CUDA_OBJS)
	$(CXX) --cuda-path=$(CUDAPATH) -shared -fPIC -o $@ $^ -L$(CUDAPATH)/lib64 -lcudart

# Compile rules
%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA kernel compile (use SOFLAGS for optimization)
src/strata/kernels/cuda/%.o: src/strata/kernels/cuda/%.cu
	@mkdir -p $(dir $@)
	$(CXX) $(SOFLAGS) -c $< -o $@

# ─── Tests ─────────────────────────────────────────────────────────
define TEST_RULE
$(1): tests/$(1).cpp $$(CORE_LIB)
	@mkdir -p tests
	$$(CXX) $$(CXXFLAGS) $$< -o tests/$$@ -L. -lstrata_core $$(LDFLAGS) \
		-Wl,-rpath,'$$$$ORIGIN/..' -Wl,-rpath,$$(CUDAPATH)/lib64
endef

$(eval $(call TEST_RULE,test_unit1))
$(eval $(call TEST_RULE,test_unit2))
$(eval $(call TEST_RULE,test_unit3))
$(eval $(call TEST_RULE,test_unit4))
$(eval $(call TEST_RULE,test_unit5))
$(eval $(call TEST_RULE,test_unit6))

# Unit 7: CUDA kernel integration test
tests/test_unit7: tests/test_unit7.cu $(CORE_LIB) $(KERNEL_LIB)
	@mkdir -p tests
	$(CXX) $(SOFLAGS) $< -o $@ -L. -lstrata_core -lix_kernels $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/..' -Wl,-rpath,$(CUDAPATH)/lib64

test1: tests/test_unit1
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit1

test2: tests/test_unit2
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit2

# Unit 2 GEMV: CUDA kernel integration test
tests/test_unit2_gemv: tests/test_unit2_gemv.cu $(CORE_LIB) $(KERNEL_LIB)
	@mkdir -p tests
	$(CXX) $(SOFLAGS) $< -o $@ -L. -lstrata_core -lix_kernels $(LDFLAGS) -Wl,-rpath,'$$ORIGIN/..' -Wl,-rpath,$(CUDAPATH)/lib64

test2g: tests/test_unit2_gemv
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit2_gemv

# Unit 2 session: graph_registry + session_orchestrator (CUDA test)
tests/test_unit2_session: tests/test_unit2_session.cu $(CORE_LIB)
	@mkdir -p tests
	$(CXX) $(SOFLAGS) $< -o $@ -L. -lstrata_core $(LDFLAGS) 		-Wl,-rpath,'$$ORIGIN/..' -Wl,-rpath,$(CUDAPATH)/lib64

test2s: tests/test_unit2_session
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit2_session

test3: tests/test_unit3
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit3

test4: tests/test_unit4
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit4

test5: tests/test_unit5
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit5

test6: tests/test_unit6
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit6

test7: tests/test_unit7
	CUDA_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit7

# Unit 8: native CUDA ops integration test
tests/test_unit8: tests/test_unit8.cu $(CORE_LIB) $(KERNEL_LIB)
	@mkdir -p tests
	$(CXX) $(SOFLAGS) $< -o $@ -L. -lstrata_core -lix_kernels $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/..' -Wl,-rpath,$(CUDAPATH)/lib64

test8: tests/test_unit8
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit8

$(eval $(call TEST_RULE,test_unit9))

test9: tests/test_unit9
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit9

test: test1 test2 test3 test4 test5 test6 test7 test8 test9

clean:
	rm -f $(OBJS) $(CUDA_OBJS) $(CORE_LIB) $(KERNEL_LIB)
	rm -f tests/test_unit1 tests/test_unit2 tests/test_unit3
	rm -f tests/test_unit4 tests/test_unit5 tests/test_unit6 tests/test_unit7 tests/test_unit8

# Unit 4: Fused GDN, fused GR, quantize_act, QSA decode attn, IQ kernels
tests/test_unit4_cuda: tests/test_unit4_cuda.cu $(CORE_LIB) $(KERNEL_LIB)
	@mkdir -p tests
	$(CXX) $(SOFLAGS) $< -o $@ -L. -lstrata_core -lix_kernels $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/..' -Wl,-rpath,$(CUDAPATH)/lib64

test4g: tests/test_unit4_cuda
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./tests/test_unit4_cuda
