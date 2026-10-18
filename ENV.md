# Baloon — Build Environment & Hardware Reference

Extracted from zeng-bin3123/ikun (infccl-v0.1). Source code excluded — this is toolchain, hardware, and shell commands only.

## Hardware: Iluvatar BI-V100

- GPU: 4× Iluvatar BI-V100, 32GB HBM2 each
- Architecture: ivcore10
- IX-ML: 3.2.3 | Driver: 3.2.1 | CUDA compat: 10.2
- SM count: check via `ixsmi` or torch probe
- Interconnect: PCIe (no NVLink)

## Compiler Toolchain

```
CXX      = /usr/local/corex/bin/clang++
ARCH     = ivcore10
CUDAPATH = /usr/local/corex
CFLAGS   = -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -I/usr/local/corex/include -O2
LDFLAGS  = -L/usr/local/corex/lib64 -lcudart -lpthread

# For .so shared libraries:
SOFLAGS  = -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -std=c++17 -O3 \
           -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ -D__ILUVATAR_DIAG__ \
           -cl-single-precision-constant -fPIC -mllvm --bonus-inst-threshold=0 -shared \
           -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart
```

Alternate `--cuda-path` if versioned SDK: `/usr/local/corex-3.2.3`

### nvcc vs clang

`/usr/local/corex/bin/nvcc` is a 194-byte wrapper script, not real nvcc. The actual compiler is `/usr/local/corex/bin/clang-16`. Always use `clang++` directly with `-x cuda`.

### Known limitations (ivcore10)

- `__threadfence_system()` — NOT supported, use `__threadfence()` (device-level) or host barriers
- CUDA IPC (`cudaIpcGetMemHandle`) — NOT supported on BI-V100, use `cudaMemcpyPeer` instead
- Max CUDA compat: 10.2 — no cooperative_groups advanced features
- warpSize: 64 (not 32 like NVIDIA)

## Available Libraries

```
/usr/local/corex/lib64/
  libcublas.so, libcublasLt.so    — BLAS
  libcuda.so, libcudart.so        — CUDA runtime
  libcudnn.so                     — cuDNN
  libcutlass.so                   — CUTLASS
  libcufft.so, libcusolver.so     — math libs
  libixattn.so                    — ixformer attention kernel
  libcuinfer.so                   — inference
  libibverbs.so                   — RDMA verbs
```

## Python Environment

```
Python: 3.10
torch:  /usr/local/corex/lib64/python3/dist-packages/torch/
vllm:   /usr/local/corex/lib64/python3/dist-packages/vllm/
```

### ixformer SDK

Key functions available via `import ixformer`:
- `flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)`
- `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)`
- `fused_add_rms_norm(input, residual, weight, eps=1e-05)`
- `silu_and_mul(input, output=None)`
- `rms_norm(input, weight, output=None, eps=1e-06)`
- `rotary_embedding(...)` 
- `scaled_dot_product_attention(query, key, value, ...)`

No topk/moe/expert/gate ops in ixformer — must implement ourselves.

## Probe Scripts

### Toolchain probe
```bash
#!/bin/bash
echo "=== nvcc identity ==="
head -c 20 /usr/local/corex/bin/nvcc | xxd | head -1
stat -c "nvcc size: %s bytes" /usr/local/corex/bin/nvcc
stat -c "clang-16 size: %s bytes" /usr/local/corex/bin/clang-16

echo "=== llc CPU targets ==="
/usr/local/corex/bin/llc -march=bi -mattr=help 2>&1 | grep -E '^\s+(generic|ivcore|sm_bi)' | head -10

echo "=== CUDA version ==="
grep -E 'CUDART_VERSION|CUDA_VERSION' /usr/local/corex/include/cuda_runtime_api.h /usr/local/corex/include/cuda.h 2>/dev/null | grep -v defgroup

echo "=== FP16 intrinsics count ==="
grep -c 'hfma\|h2exp\|__hmul\|__hadd\|__hfma2\|__half2half2' /usr/local/corex/include/cuda_fp16.h 2>/dev/null

echo "=== NCCL version ==="
grep -E 'NCCL_MAJOR|NCCL_MINOR|NCCL_PATCH' /usr/local/corex/include/nccl.h 2>/dev/null | head -3
```

### Hardware probe (compile & run)
```bash
CXX=/usr/local/corex/bin/clang++
FLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex-3.2.3 -L/usr/local/corex/lib64 -lcudart -I/usr/local/corex/include"
$CXX $FLAGS probe_hw.cu -o /tmp/probe_hw && CUDA_VISIBLE_DEVICES=0,1,2,3 /tmp/probe_hw
```

### Quick GPU check
```bash
ixsmi                              # GPU status (like nvidia-smi)
ixsmi -q                           # detailed query
ixsmi topo -m                      # topology matrix
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

## Build Example (Makefile)

```makefile
CXX      := /usr/local/corex/bin/clang++
ARCH     := ivcore10
CUDAPATH := /usr/local/corex
CFLAGS   := -x cuda --cuda-gpu-arch=$(ARCH) --cuda-path=$(CUDAPATH) -I$(CUDAPATH)/include -O2
LDFLAGS  := -L$(CUDAPATH)/lib64 -lcudart -lpthread

all: my_test

my_test: src/my_kernel.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test:
	CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 120 ./my_test

clean:
	rm -f my_test
```


## Benchmark (tok/s measurement)

```python
import requests, time
url = "http://localhost:80/v1/chat/completions"
payload = {
    "model": "llm",
    "messages": [{"role": "user", "content": "请详细解释量子计算的基本原理"}],
    "max_tokens": 2048, "temperature": 0.7, "stream": False
}
# warmup 3 runs, then 5 timed runs
for i in range(5):
    start = time.perf_counter()
    r = requests.post(url, json=payload, timeout=300)
    elapsed = time.perf_counter() - start
    tps = r.json()["usage"]["completion_tokens"] / elapsed
    print(f"Run {i+1}: {tps:.1f} tok/s")
```
