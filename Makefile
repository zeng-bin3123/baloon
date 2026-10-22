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

# Sources
SRCS := src/strata/core/graph.cpp \
        src/strata/core/pinned.cpp \
        src/strata/core/weights.cpp \
        src/strata/core/verify.cpp \
        src/strata/core/mtp.cpp \
        src/strata/core/expert_cache.cpp \
        src/strata/core/expert_source.cpp \
        src/strata/core/layer.cpp \
        src/strata/core/session.cpp

OBJS := $(SRCS:.cpp=.o)

# Shared library
LIBNAME := libstrata_core.so

# Test binaries
TEST1_SRC := tests/test_unit1.cpp
TEST1_BIN := tests/test_unit1
TEST2_SRC := tests/test_unit2.cpp
TEST2_BIN := tests/test_unit2
TEST3_SRC := tests/test_unit3.cpp
TEST3_BIN := tests/test_unit3

.PHONY: all clean test test1 test2 test3 lib

all: lib

lib: $(LIBNAME)

$(LIBNAME): $(OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TEST1_BIN): $(TEST1_SRC) $(LIBNAME)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $< -o $@ -L. -lstrata_core $(LDFLAGS) 		-Wl,-rpath,'$$ORIGIN/..' -Wl,-rpath,$(CUDAPATH)/lib64

$(TEST2_BIN): $(TEST2_SRC) $(LIBNAME)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $< -o $@ -L. -lstrata_core $(LDFLAGS) 		-Wl,-rpath,'$$ORIGIN/..' -Wl,-rpath,$(CUDAPATH)/lib64

$(TEST3_BIN): $(TEST3_SRC) $(LIBNAME)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $< -o $@ -L. -lstrata_core $(LDFLAGS) 		-Wl,-rpath,'$$ORIGIN/..' -Wl,-rpath,$(CUDAPATH)/lib64

test1: $(TEST1_BIN)
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./$(TEST1_BIN)

test2: $(TEST2_BIN)
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./$(TEST2_BIN)

test3: $(TEST3_BIN)
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(CURDIR):$(CUDAPATH)/lib64 ./$(TEST3_BIN)

test: test1 test2 test3

clean:
	rm -f $(OBJS) $(LIBNAME) $(TEST1_BIN) $(TEST2_BIN) $(TEST3_BIN)
