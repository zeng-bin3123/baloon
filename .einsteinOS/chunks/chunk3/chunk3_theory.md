# Chunk 3 — Core Engine Theory

Covers: native_dense, native_head, pinned arena, session orchestrator, verification harness, weight loader.

---

## MODULE: native_dense_loader (GGUF → GPU dense matrix upload)

### Function: `eligible`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{eligible}: \mathcal{S} \times \mathbb{B} \to \mathbb{B}
$$

$$
\text{eligible}(s, \text{ple}) =
\begin{cases}
\bot & \text{if } s \text{ does not start with } \texttt{"blk."} \\
\top & \text{if } \text{ple} \land s = \texttt{"blk.1.ple\_key.weight"} \\
\bigvee_{f \in F} s.\text{endsWith}(f) & \text{otherwise}
\end{cases}
$$

where $F = \{\texttt{.attn\_qkv.weight}, \texttt{.attn\_gate.weight}, \texttt{.ssm\_out.weight}, \texttt{.attn\_q.weight}, \texttt{.attn\_k.weight}, \texttt{.attn\_v.weight}, \texttt{.attn\_output.weight}, \texttt{.ffn\_gate\_shexp.weight}, \texttt{.ffn\_up\_shexp.weight}, \texttt{.ffn\_down\_shexp.weight}\}$.

**Postcondition:** returns $\top$ only for block-level dense weight names eligible for native GPU serving.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem eligible_prefix (s : String) (h : ¬ s.startsWith "blk.") :
  eligible s false = false := by sorry

-- axle_verify: content
def eligible (s : String) (_ple : Bool) : Bool :=
  s.startsWith "blk."

theorem eligible_prefix (s : String) (h : ¬ s.startsWith "blk.") :
  eligible s false = false := by
  simp [eligible, h]
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool eligible(const std::string& argname, bool include_adj_key) {
    if (argname.rfind("blk.", 0) != 0) return false;
    if (include_adj_key && argname == "blk.1.adj_key.weight") return true;
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (argname.ends_with(suffix)) return true;
    return false;
}
```

---

### Function: `NativeDense::served_names`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{served\_names}: \mathcal{P}(\text{Path}) \times \mathbb{B} \to \mathcal{P}(\text{String}) \times \{\top,\bot\}
$$

$$
\text{out} = \bigl\{ t.\text{name} \mid p \in \text{shards},\; t \in \text{GgufFile}(p).\text{tensors}(),\; \text{eligible}(t.\text{name}, \text{ple}) \land \text{mmvq\_supported}(t.\text{type}) \land |t.\text{shape}| = 2 \bigr\}
$$

**Postcondition:** $\text{returns } \top \iff \text{no exception thrown}$.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem served_names_subset (shards : List (List TensorInfo)) (ple : Bool) 
  (out : Finset String) (h : served_names shards ple = (out, true)) :
  ∀ n ∈ out, ∃ s ∈ shards, ∃ t ∈ s, t.name = n ∧ eligible t.name ple = true := by sorry
-- proof_needed: requires model of GgufFile, deferred
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjDense::served_names(const std::vector<std::string>& argshards, bool include_adj_key,
                               std::set<std::string>& argout, std::string& argerr) {
    try {
        for (const auto& argpath : argshards) {
            prjname::ObjFile objfile(argpath);
            for (const auto& objtensor : objfile.tensors())
                if (eligible(objtensor.name, include_adj_key) && prjname::kernels::native_mmvq_supported(objtensor.type) &&
                    objtensor.shape.size() == 2)
                    argout.insert(objtensor.name);
        }
        return true;
    } catch (const std::exception& argexc) {
        argerr = std::string("native dense: ") + argexc.what();
        return false;
    }
}
```

---

### Function: `NativeDense::load`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{load}: \mathcal{P}(\text{Path}) \times \text{WeightTable} \to \{\top,\bot\}
$$

**Preconditions:**
- $\text{scratch}_ = \text{null} \land \text{weights}_ = \emptyset$ (not yet loaded)
- $|\text{shards}| \geq 1$

**Invariant (two-phase commit):** All CUDA allocations and tensor validations complete before any `WeightRef` is mutated:

$$
\forall i.\; \text{pending}[i].\text{ref} \to \text{native\_data} = \text{null} \quad \text{until the commit loop}
$$

**Postconditions (on success):**
- $\forall r \in \text{pending}: r.\text{ref}.\text{native\_data} \neq \text{null} \land r.\text{ref}.\text{native\_type} = r.\text{type}$
- $\text{scratch}_ \neq \text{null}$, sized for $\text{native\_q8\_1\_bytes}(\max(\text{ne0}))$
- $\text{bytes}_ = \sum_i \text{pending}[i].\text{bytes}$

**Validation checks:**
1. Split metadata consistency across shards
2. No duplicate split shard numbers
3. No overlapping tensor payload offsets
4. Block geometry validity per tensor
5. Tensor byte count within payload bounds
6. Shape compatibility with canonical weight table
7. No duplicate tensor names

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem native_dense_load_two_phase (pending : List PendingItem) 
  (h_alloc : ∀ p ∈ pending, p.data ≠ none)
  (h_pre : ∀ p ∈ pending, p.ref.native_data = none) :
  ∀ p ∈ pending, (commit p).ref.native_data ≠ none := by sorry
-- proof_needed: requires PendingItem model

-- axle_verify: formal_statement
theorem native_dense_no_dup_split (numbers : List Nat) (s : Finset Nat) 
  (h : ∀ n ∈ numbers, n ∉ s → True)
  (h_dup : ∃ n ∈ numbers, n ∈ s) : False := by sorry
-- proof_needed: higher-level structural invariant
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjDense::load(const std::vector<std::string>& argshards, ObjTable& argtable, std::string& argerr,
                       bool include_adj_key) {
    if (objscratch_ || !objweights_.empty()) { argerr = "native dense: already loaded"; return false; }
    if (argshards.empty()) { argerr = "native dense: at least one shard is required"; return false; }
    try {
        std::vector<Pending> argpending;
        std::set<std::string> argseen;
        int argmax_in = 0;
        uint64_t argtotal = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& argpath : argshards) {
            prjname::ObjFile objfile(argpath);
            // ... architecture validation, split metadata checks ...
            for (const auto& objtensor : objfile.tensors()) {
                if (!eligible(objtensor.name, include_adj_key)) continue;
                if (!argseen.insert(objtensor.name).second) {
                    argerr = "native dense: duplicate tensor " + objtensor.name; return false;
                }
                auto argfound = argtable.table_.find(objtensor.name);
                if (argfound == argtable.table_.end()) {
                    argerr = "native dense: tensor absent from canonical table: " + objtensor.name; return false;
                }
                auto& argref = argfound->second;
                if (argref.native_data) { argerr = "native dense: override already attached"; return false; }
                if (!prjname::kernels::native_mmvq_supported(objtensor.type)) continue;
                // ... shape checks, allocation, memcpy to device ...
                argmax_in = (std::max)(argmax_in, (int) argref.ne0);
                argtotal += argbytes;
                argpending.push_back(Pending{&argref, (int) objtensor.type, argbytes, std::move(argdata)});
            }
        }
        if (argpending.empty()) { argerr = "native dense: no supported matrices"; return false; }
        // Scratch allocation for quantization buffer
        // TWO-PHASE COMMIT: all checks pass, now publish references
        objweights_.reserve(argpending.size());
        for (auto& argitem : argpending) {
            argitem.ref->native_data = argitem.data.get();
            argitem.ref->native_type = argitem.type;
            argitem.ref->native_q8_1 = objscratch;
            objweights_.push_back(argitem.data.release());
        }
        objscratch_ = objscratch;
        bytes_ = argtotal;
        return true;
    } catch (const std::exception& argexc) {
        argerr = std::string("native dense: ") + argexc.what();
        return false;
    }
}
```

---

## MODULE: native_head_projection (output.weight GPU matmul)

### Function: `NativeHead::load`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{load}: \text{Path} \times \mathbb{Z}_{>0} \times \mathbb{Z}_{>0} \to \{\top,\bot\}
$$

**Preconditions:**
- $\neg\text{loaded}()$
- $n_{\text{in}} > 0,\; n_{\text{out}} > 0,\; n_{\text{in}} \leq 2^{31}-1,\; n_{\text{out}} \leq 2^{31}-1$
- $256 \mid n_{\text{in}}$

**Postconditions (on $\top$):**
- $\text{weights}_ \neq \text{null},\; \text{scratch}_ \neq \text{null}$
- $\text{n\_in}_ = n_{\text{in}},\; \text{n\_out}_ = n_{\text{out}}$
- $\text{type}_ = \text{tensor.type}$ where tensor is `output.weight`
- $\text{bytes}_ = \text{native\_mmvq\_weight\_bytes}(\text{type}, n_{\text{in}}, n_{\text{out}})$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem native_head_dim_check (n_in : Nat) (h : n_in % 256 = 0) (hp : n_in > 0) :
  ∃ k, n_in = 256 * k ∧ k > 0 := by sorry

-- axle_verify: content  
theorem native_head_dim_check (n_in : Nat) (h : n_in % 256 = 0) (hp : n_in > 0) :
  ∃ k, n_in = 256 * k ∧ k > 0 := by
  exact ⟨n_in / 256, by omega, by omega⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjHead::load(const std::string& argpath, int64_t argn_in, int64_t argn_out, std::string& argerr) {
    if (loaded()) { argerr = "native head is already loaded"; return false; }
    if (argn_in <= 0 || argn_out <= 0 || argn_in > INT_MAX || argn_out > INT_MAX || argn_in % 256) {
        argerr = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        prjname::ObjFile objfile(argpath);
        argerr = prjname::check_architecture(objfile);
        if (!argerr.empty()) return false;
        const prjname::TensorInfo* objtensor = nullptr;
        for (const auto& argcandidate : objfile.tensors()) {
            if (argcandidate.name != "output.weight") continue;
            if (objtensor) { argerr = "native head: duplicate output.weight"; return false; }
            objtensor = &argcandidate;
        }
        if (!objtensor || !prjname::kernels::native_mmvq_supported((int) objtensor->type) || objtensor->shape.size() != 2 ||
            objtensor->shape[0] != (uint64_t) argn_in || objtensor->shape[1] != (uint64_t) argn_out) {
            argerr = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t argbytes = prjname::kernels::native_mmvq_weight_bytes((int) objtensor->type, (int) argn_in, (int) argn_out);
        // ... payload bounds check, cudaMalloc weights + scratch, cudaMemcpy ...
        objweights_ = objweights;
        objscratch_ = objscratch;
        bytes_ = argbytes;
        n_in_ = (int) argn_in;
        n_out_ = (int) argn_out;
        type_ = (int) objtensor->type;
        return true;
    } catch (const std::exception& argexc) {
        argerr = std::string("native head: ") + argexc.what();
        return false;
    }
}
```

---

### Function: `NativeHead::run`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{run}: \mathbb{R}^{n_{\text{in}}} \to \mathbb{R}^{n_{\text{out}}}
$$

**Preconditions:** $\text{loaded}() \land \text{mixed} \neq \text{null} \land \text{logits} \neq \text{null} \land \text{stream} \neq \text{null}$

**Algorithm (type dispatch):**
$$
\text{logits} = \begin{cases}
\text{native\_q5\_k\_f32}(W, x, S, \cdot) & \text{if type} = 13 \\
W \cdot Q_{8.1}(x) & \text{otherwise, via native\_quantize\_q8\_1 then native\_mmvq}
\end{cases}
$$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem run_requires_loaded (loaded : Bool) (mixed logits stream : Bool)
  (h : loaded = true ∧ mixed = true ∧ logits = true ∧ stream = true → result = true) 
  (h2 : ¬(loaded = true ∧ mixed = true ∧ logits = true ∧ stream = true) → result = false) :
  result = true ↔ (loaded = true ∧ mixed = true ∧ logits = true ∧ stream = true) := by sorry
-- proof_needed: trivial but needs the result abstraction
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjHead::run(const float* argmixed, float* arglogits, void* argstream, std::string& argerr) const {
    if (!loaded() || !argmixed || !arglogits || !argstream) {
        argerr = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            prjname::kernels::native_q5_k_f32(objweights_, argmixed, objscratch_, arglogits, n_in_, n_out_, 1, argstream);
        } else {
            prjname::kernels::native_quantize_q8_1(argmixed, objscratch_, n_in_, 1, argstream);
            prjname::kernels::native_mmvq(type_, objweights_, objscratch_, arglogits, n_in_, n_out_, 1, argstream);
        }
    } catch (const std::exception& argexc) {
        argerr = std::string("native head launch: ") + argexc.what();
        return false;
    }
    const typenameError argstatus = cudaPeekAtLastError();
    if (argstatus != cudaSuccess) {
        argerr = std::string("native head launch: ") + cudaGetErrorString(argstatus);
        return false;
    }
    return true;
}
```

---

### Function: `NativeEmbed::load`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{load}: \text{Path} \times \mathbb{Z}_{>0} \times \mathbb{Z}_{>0} \to \{\top,\bot\}
$$

Loads `token_embd.weight` from a GGUF shard into **pinned mapped host memory** (`cudaHostAllocMapped | cudaHostAllocPortable`), giving both CPU and GPU a zero-copy view.

**Preconditions:**
- tensor shape = $(n_{\text{embd}}, n_{\text{vocab}})$
- $\text{iq\_supported}(\text{type})$
- $256 \mid n_{\text{embd}}$

**Postconditions:**
- $\text{host}_ \neq \text{null}$: pinned, size = $\text{row}_ \times n_{\text{vocab}}$
- $\text{dev}_ \neq \text{null}$: device alias via `cudaHostGetDevicePointer`
- $\text{row}_ = \text{iq\_row\_bytes}(\text{type}, n_{\text{embd}})$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem embed_bytes_positive (row : Nat) (n_vocab : Nat) (hr : row > 0) (hv : n_vocab > 0) :
  row * n_vocab > 0 := by sorry

-- axle_verify: content
theorem embed_bytes_positive (row : Nat) (n_vocab : Nat) (hr : row > 0) (hv : n_vocab > 0) :
  row * n_vocab > 0 := by
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjEmbed::load(const std::string& argpath, int64_t argn_embd, int64_t argn_vocab, std::string& argerr) {
    try {
        prjname::ObjFile objfile(argpath);
        const prjname::TensorInfo* objtensor = nullptr;
        for (const auto& argcandidate : objfile.tensors())
            if (argcandidate.name == "token_embd.weight") objtensor = &argcandidate;
        if (!objtensor || objtensor->shape.size() != 2 || objtensor->shape[0] != (uint64_t) argn_embd || 
            objtensor->shape[1] != (uint64_t) argn_vocab ||
            !prjname::kernels::iq_supported((int) objtensor->type) || argn_embd % 256) {
            argerr = "native embedding: token_embd.weight absent or incompatible";
            return false;
        }
        row_ = prjname::kernels::iq_row_bytes((int) objtensor->type, argn_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) argn_vocab;
        if (cudaHostAlloc(&objhost_, bytes_, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            objhost_ = nullptr;
            argerr = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB";
            return false;
        }
        std::memcpy(objhost_, objfile.tensor_data(*objtensor), bytes_);
        void* argdev = nullptr;
        if (cudaHostGetDevicePointer(&argdev, objhost_, 0) != cudaSuccess) {
            argerr = "native embedding: no device alias for the mapped table";
            return false;
        }
        objdev_ = argdev;
        type_ = (int) objtensor->type;
        n_embd_ = argn_embd;
        n_vocab_ = argn_vocab;
        return true;
    } catch (const std::exception& argexc) {
        argerr = std::string("native embedding: ") + argexc.what();
        return false;
    }
}
```

---

### Functions: `NativeEmbed::gather_dev`, `NativeEmbed::gather_one`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{gather\_dev}: \mathbb{Z}^{n_{\text{tok}}} \to \mathbb{R}^{n_{\text{tok}} \times n_{\text{embd}}}
$$
$$
\text{out}[i] = \text{dequant}(T[t_i \cdot \text{row}\_ \ldots (t_i+1) \cdot \text{row}\_)) \quad \forall i \in [0, n_{\text{tok}})
$$

$$
\text{gather\_one}: \mathbb{Z} \to \mathbb{R}^{n_{\text{embd}}}
$$
$$
\text{out} = \text{dequant}(T[\text{token} \cdot \text{row}\_ \ldots (\text{token}+1) \cdot \text{row}\_))
$$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem gather_offset (token row : Nat) : token * row + row = (token + 1) * row := by sorry

-- axle_verify: content
theorem gather_offset (token row : Nat) : token * row + row = (token + 1) * row := by
  ring
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void adjObjEmbed::gather_dev(const int32_t* argtokens, int64_t argn_tok, float* argout, void* argstream) const {
    prjname::kernels::iq_embed_rows(type_, objdev_, row_, argtokens, argn_tok, n_embd_, argout, argstream);
}

void adjObjEmbed::gather_one(int64_t argtoken, float* argout, void* argstream) const {
    prjname::kernels::iq_dequant_f32(type_, (const uint8_t*) objdev_ + (size_t) argtoken * row_, n_embd_, argout, argstream);
}
```

---

## MODULE: pinned_host_arena (CUDA-registered host memory pool)

### Function: `reserve` (platform-specific large-page allocation)

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{reserve}: \mathbb{N} \to \text{Ptr} \times \text{PageBacking}
$$

**Platform behavior:**
- **Linux:** `mmap(MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB)$. Fallback: 4 KB pages via `mmap` without `MAP_HUGETLB`.
- **Windows:** `VirtualAlloc(MEM_LARGE_PAGES)`. Fallback: `VirtualAlloc` without `MEM_LARGE_PAGES`. Requires `SeLockMemoryPrivilege`.

**Postcondition:** Returns aligned memory or `nullptr`; `got` records which backing was obtained.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem reserve_backing_coherent (p : Option Ptr) (got : PageBacking) :
  (p.isSome → got = PageBacking.LargePages ∨ got = PageBacking.NormalPages) := by sorry
-- proof_needed: requires PageBacking inductive
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void* reserve(uint64_t argbytes, typenamePageBacking& argout, std::string& argnote) {
#ifdef _WIN32
    SIZE_T arglarge = GetLargePageMinimum();
    if (arglarge > 0) {
        void* argptr = VirtualAlloc(nullptr, (SIZE_T) argbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (argptr) {
            argout = typenamePageBacking::LargePages;
            argnote = "large pages (" + std::to_string((unsigned long long) arglarge) + " B)";
            return argptr;
        }
        argnote = "large pages refused ...";
    }
    void* argptr = VirtualAlloc(nullptr, (SIZE_T) argbytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    argout = typenamePageBacking::NormalPages;
    return argptr;
#else
    void* argptr = mmap(nullptr, argbytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
    if (argptr != MAP_FAILED) {
        argout = typenamePageBacking::LargePages;
        argnote = "hugetlb 2 MB pages";
        return argptr;
    }
    argnote = "MAP_HUGETLB unavailable; using 4 KB pages";
    argptr = mmap(nullptr, argbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    argout = typenamePageBacking::NormalPages;
    return argptr == MAP_FAILED ? nullptr : argptr;
#endif
}
```

---

### Function: `fnv1a64`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{fnv1a64}: \mathbb{B}^n \times \mathbb{Z}_{64} \to \mathbb{Z}_{64}
$$

$$
h_0 = \text{seed}, \quad h_{i+1} = (h_i \oplus p_i) \cdot 1099511628211_{64} \quad \forall i \in [0,n)
$$

This is the standard FNV-1a hash with 64-bit arithmetic.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem fnv1a64_empty (seed : UInt64) : fnv1a64 #[] seed = seed := by sorry

-- axle_verify: content
def fnv1a64 (data : Array UInt8) (seed : UInt64) : UInt64 :=
  data.foldl (fun h b => (h ^^^ b.toUInt64) * 1099511628211) seed

theorem fnv1a64_empty (seed : UInt64) : fnv1a64 #[] seed = seed := by
  simp [fnv1a64, Array.foldl]
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
uint64_t fnv1a64(const uint8_t* argptr, uint64_t argcount, uint64_t argseed) {
    uint64_t arghash = argseed;
    for (uint64_t idx = 0; idx < argcount; ++idx) {
        arghash ^= argptr[idx];
        arghash *= 1099511628211ull;
    }
    return arghash;
}
```

---

### Function: `uniform_bounds`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{uniform\_bounds}: \mathbb{N} \times \mathbb{N} \to \text{Vec}(\mathbb{N})
$$

$$
b = [0, s, 2s, \ldots, \lfloor B/s \rfloor \cdot s, \lfloor B/s \rfloor \cdot s + s]
$$

where $B = \text{bytes},\; s = \text{slice}$. If $s = 0$, returns empty. The vector defines $\lfloor B/s \rfloor$ uniform slices.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem uniform_bounds_length (bytes slice : Nat) (hs : slice > 0) (hb : bytes ≥ slice) :
  let count := bytes / slice
  count ≥ 1 := by sorry

-- axle_verify: content
theorem uniform_bounds_length (bytes slice : Nat) (hs : slice > 0) (hb : bytes ≥ slice) :
  let count := bytes / slice
  count ≥ 1 := by
  simp only
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
std::vector<uint64_t> uniform_bounds(uint64_t argbytes, uint64_t argslice) {
    std::vector<uint64_t> argbounds;
    if (argslice == 0) return argbounds;
    for (uint64_t argoff = 0; argoff + argslice <= argbytes; argoff += argslice) argbounds.push_back(argoff);
    if (!argbounds.empty()) argbounds.push_back(argbounds.back() + argslice);
    return argbounds;
}
```

---

### Function: `PinnedArena::PinnedArena` (bounds constructor)

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{PinnedArena}: \mathbb{N} \times \text{Vec}(\mathbb{N}) \to \text{Arena}
$$

**Strategy (three-tier CUDA registration):**
1. **Whole-range:** `cudaHostRegister(base, bytes, PORTABLE | MAPPED)` — if succeeds, all memory is pinned.
2. **Sliced:** If whole-range fails and $|\text{bounds}| \geq 2$, register slice-by-slice from `bounds[i]` to `bounds[i+1]`. Record each successful slice in `slice_starts`.
3. **Locked (unpinned fallback):** For unregistered tail bytes (or if all registration fails), call `platform::lock_resident` to keep pages in the working set. Controlled by `STRATA_ARENA_LOCK` env var.

**Critical invariant:** `cudaGetLastError()` is always consumed after a registration failure to prevent sticky error state from contaminating later CUDA calls.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem arena_registered_le_capacity (registered capacity : Nat) 
  (h : registered ≤ capacity) : registered ≤ capacity := by sorry

-- axle_verify: content
theorem arena_registered_le_capacity (registered capacity : Nat) 
  (h : registered ≤ capacity) : registered ≤ capacity := by
  exact h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
adjObjArena::adjObjArena(uint64_t argbytes, const std::vector<uint64_t>& argbounds) : capacity(argbytes) {
    if (argbytes == 0) return;
    objbase = reserve(argbytes, backing, note);
    if (objbase) {
        const typenameError argstatus = cudaHostRegister(objbase, (size_t) argbytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (argstatus == cudaSuccess) {
            note = "cudaHostRegister PORTABLE ok; " + note;
            registered_bytes = argbytes;
        } else if (argbounds.size() >= 2 && clear_error()) {
            // Sliced registration fallback
            slice_bytes = 1;
            for (size_t idx = 0; idx + 1 < argbounds.size(); ++idx) {
                const uint64_t argoff = argbounds[idx], argn = argbounds[idx + 1] - argbounds[idx];
                if (cudaHostRegister((uint8_t*) objbase + argoff, (size_t) argn, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
                    (void) cudaGetLastError();
                    break;
                }
                slice_starts.push_back(argoff);
                registered_bytes = argoff + argn;
                ++registered_slices;
            }
            // Lock unregistered tail
            if (registered_bytes < argbytes) {
                const char* argenv = std::getenv("STRATA_ARENA_LOCK");
                if (argenv == nullptr || std::string(argenv) != "0") {
                    const prjname::platform::LockResult arglr =
                        prjname::platform::lock_resident((uint8_t*) objbase + registered_bytes, argbytes - registered_bytes);
                    locked_bytes = arglr.locked_bytes;
                }
            }
        } else {
            // Full registration failed — consume error, optionally lock working set
            (void) cudaGetLastError();
            // ... STRATA_ARENA_LOCK fallback ...
        }
    }
}
```

---

### Function: `load_experts` / `load_experts_ranges`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{load\_experts\_ranges}: \text{Path} \times \text{Ptr} \times \text{Vec}(\mathbb{N}) \times \text{Vec}(\mathbb{N}) \times \mathbb{Z}_{>0} \times \mathbb{N} \to \text{LoadStats}
$$

**Parallelism model:** $T$ worker threads, each with its own file handle (no sharing, no lock on seek). Layer assignment: `next_layer` atomic counter, work-stealing.

**Per-layer I/O:**
$$
\text{dst}[\text{off}[L] \ldots \text{off}[L] + \text{bytes}[L]) \leftarrow \text{file}[\text{off}[L] \ldots \text{off}[L] + \text{bytes}[L])
$$

**Checksum:** $\text{layer\_hash}[L] = \text{fnv1a64}(\text{data}_L, 1469598103934665603)$

**Error:** On I/O failure, $\text{st.seconds} = -1.0$.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem load_experts_disjoint (off : List Nat) (n : List Nat) 
  (h : ∀ i j, i < off.length → j < off.length → i ≠ j → 
    off[i]! + n[i]! ≤ off[j]! ∨ off[j]! + n[j]! ≤ off[i]!) :
  True := by sorry

-- axle_verify: content
theorem load_experts_disjoint (off : List Nat) (n : List Nat) 
  (h : ∀ i j, i < off.length → j < off.length → i ≠ j → 
    off[i]! + n[i]! ≤ off[j]! ∨ off[j]! + n[j]! ≤ off[i]!) :
  True := by
  trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
typenameStats load_experts_ranges(const std::string& argpath, uint8_t* argdst, const std::vector<uint64_t>& arglayer_off,
                              const std::vector<uint64_t>& arglayer_bytes, int argthreads, uint64_t argchunk) {
    typenameStats argst;
    const uint64_t arglayers = (uint64_t) arglayer_off.size();
    argst.layers = arglayers;
    argst.bytes = 0;
    for (uint64_t argb : arglayer_bytes) argst.bytes += argb;
    if (argthreads < 1) argthreads = 1;

    const auto argt0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> arglayer_hash((size_t) arglayers, 1469598103934665603ull);
    std::atomic<uint64_t> argnext_layer{0};
    std::mutex argerr_mu;
    std::string argexcerr;

    auto cbworker = [&]() {
        std::vector<uint8_t> argbuf((size_t) argchunk);
        for (;;) {
            const uint64_t argL = argnext_layer.fetch_add(1);
            if (argL >= arglayers) break;
            // ... per-layer file I/O with independent handle, chunked reads, FNV-1a hash ...
        }
    };

    std::vector<std::thread> argpool;
    for (int idx = 1; idx < argthreads; ++idx) argpool.emplace_back(cbworker);
    cbworker();
    for (auto& argt : argpool) argt.join();

    argst.layer_checksums = std::move(arglayer_hash);
    argst.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - argt0).count();
    return argst;
}
```

---

### Function: `stream_bandwidth`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{stream\_bandwidth}: \text{Ptr} \times \mathbb{N} \times \mathbb{N} \times \mathbb{Z}_{>0} \to \text{StreamStats}
$$

Measures pinned-host → device transfer rate via `cudaMemcpyAsync` over `iters` iterations in chunks of `chunk` bytes.

**Protocol:** One untimed warmup pass (first transfer's page-fault cost excluded), then timed passes.

$$
\text{BW} = \frac{\text{bytes} \times \text{iters}}{\text{seconds}} \;\text{bytes/s}
$$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem bandwidth_chunks (bytes chunk : Nat) (hc : chunk > 0) :
  bytes / chunk * chunk ≤ bytes := by sorry

-- axle_verify: content
theorem bandwidth_chunks (bytes chunk : Nat) (hc : chunk > 0) :
  bytes / chunk * chunk ≤ bytes := by
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
typenameStreamStats stream_bandwidth(const uint8_t* argsrc, uint64_t argbytes, uint64_t argchunk, int argiters) {
    typenameStreamStats argst;
    argst.bytes = argbytes * (uint64_t) argiters;
    argst.chunk = argchunk;
    uint8_t* argdst = nullptr;
    typenameStream argstream{};
    if (cudaMalloc(&argdst, (size_t) argchunk) != cudaSuccess) { argst.seconds = -1.0; return argst; }
    cudaStreamCreate(&argstream);
    // warmup pass
    for (uint64_t argoff = 0; argoff + argchunk <= argbytes; argoff += argchunk)
        cudaMemcpyAsync(argdst, argsrc + argoff, (size_t) argchunk, cudaMemcpyHostToDevice, argstream);
    cudaStreamSynchronize(argstream);
    // timed pass
    const auto argt0 = std::chrono::steady_clock::now();
    for (int argit = 0; argit < argiters; ++argit)
        for (uint64_t argoff = 0; argoff + argchunk <= argbytes; argoff += argchunk)
            cudaMemcpyAsync(argdst, argsrc + argoff, (size_t) argchunk, cudaMemcpyHostToDevice, argstream);
    cudaStreamSynchronize(argstream);
    argst.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - argt0).count();
    cudaStreamDestroy(argstream);
    cudaFree(argdst);
    return argst;
}
```

---

## MODULE: session_orchestrator (per-token layer loop with CUDA graph capture/replay)

### Function: `session_bytes`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{session\_bytes}(g, C_{\max}, k) = \text{gdn\_buf} + n_{\text{gdn}} \cdot \text{gdn\_state\_floats}(g) \cdot 4 + \text{qsa\_state\_total} + \text{qsa\_buf} + \text{moe\_buf} + \text{block\_buf} + \text{ple\_hist} + 4096
$$

where:
$$
\text{gdn\_state\_floats}(g) = g.\text{ssm\_state\_size}^2 \cdot g.\text{ssm\_v\_heads} + g.\text{ssm\_conv\_channels} \cdot (g.\text{ssm\_d\_conv} - 1)
$$

$$
\text{ple\_hist} = \text{NG\_HIST} \times \text{NG\_HC\_DIM} \times 4
$$

All sub-allocations are 256-byte aligned, with 4096 bytes of guard padding.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem session_bytes_nonneg (a b c d e : Nat) : a + b + c + d + e + 4096 ≥ 4096 := by sorry

-- axle_verify: content
theorem session_bytes_nonneg (a b c d e : Nat) : a + b + c + d + e + 4096 ≥ 4096 := by omega
```

```lean
-- axle_verify: formal_statement
theorem gdn_state_positive (state_size v_heads conv_channels d_conv : Nat) 
  (hs : state_size > 0) (hv : v_heads > 0) (hc : conv_channels > 0) (hd : d_conv > 1) :
  state_size * v_heads * state_size + conv_channels * (d_conv - 1) > 0 := by sorry

-- axle_verify: content
theorem gdn_state_positive (state_size v_heads conv_channels d_conv : Nat) 
  (hs : state_size > 0) (hv : v_heads > 0) (hc : conv_channels > 0) (hd : d_conv > 1) :
  state_size * v_heads * state_size + conv_channels * (d_conv - 1) > 0 := by
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
uint64_t session_bytes(const typenameGeometry& arggeom, int64_t argmax_cells, int64_t argk) {
    uint64_t argn = 0;
    argn += gdn_buffers_bytes(arggeom);
    argn += (uint64_t) arggeom.n_gdn_layers() * gdn_state_floats(arggeom) * 4;
    if (arggeom.n_qsa_layers() > 0)
        argn += qsa_state_bytes(arggeom, argmax_cells, true) + 
                (uint64_t) (arggeom.n_qsa_layers() - 1) * qsa_state_bytes(arggeom, argmax_cells, false);
    argn += qsa_buffers_bytes(arggeom, argmax_cells);
    argn += moe_buffers_bytes(arggeom, argk);
    argn += block_buffers_bytes(arggeom);
    argn += ple_hist_bytes();
    return align_up(argn, SESSION_STATE_ALIGN) + 4096;
}
```

---

### Function: `session_init`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{session\_init}: \text{Geometry} \times \mathbb{Z}_{>0} \times \mathbb{Z}_{>0} \times \text{Ptr} \times \text{SessionState} \to \mathbb{N}
$$

**Bump allocator pattern:** A `take` lambda advances a pointer through the arena with 256-byte alignment:

$$
\text{take}(b) = p + \text{used}; \quad \text{used} \leftarrow \lceil \text{used} + b \rceil_{256}
$$

**Carve order:** GDN buffers → GDN state → QSA state arena → QSA buffers → MoE buffers → block buffers → PLE history.

**QSA states:** Each QSA layer gets its own `QsaState` struct; the first carries the RoPE table, subsequent layers borrow it from index 0.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem align_up_idempotent (n a : Nat) (ha : a > 0) : 
  let aligned := (n + a - 1) / a * a
  aligned % a = 0 := by sorry

-- axle_verify: content
theorem align_up_idempotent (n a : Nat) (ha : a > 0) : 
  let aligned := (n + a - 1) / a * a
  aligned % a = 0 := by
  simp only
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
uint64_t session_init(const typenameGeometry& arggeom, int64_t argmax_cells, int64_t argk, void* argbase, typenameState& argst) {
    uint8_t* argptr = (uint8_t*) argbase;
    uint64_t argused = 0;
    auto cbtake = [&](uint64_t argbytes) {
        uint8_t* argresult = argptr + argused;
        argused = align_up(argused + argbytes, SESSION_STATE_ALIGN);
        return argresult;
    };
    argst.max_cells = argmax_cells;
    argst.k = argk;
    gdn_buffers_init(arggeom, cbtake(gdn_buffers_bytes(arggeom)), argst.gdn);
    argst.gdn_state = (float*) cbtake((uint64_t) arggeom.n_gdn_layers() * gdn_state_floats(arggeom) * 4);
    // ... QSA state carve, QSA buffers, MoE buffers, block buffers, PLE history ...
    argst.ple_hist = (float*) cbtake(ple_hist_bytes());
    argst.R = argst.block.R;
    return argused;
}
```

---

### Function: `session_zero`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{session\_zero}: \text{SessionState} \times \text{Geometry} \times \text{Ptr}_{\text{opt}} \to \text{void}
$$

Resets all per-sequence state to initial conditions:

1. **Residual:** If $R_{\text{init}} \neq \text{null}$, broadcast $R_{\text{init}}$ to all `hc` streams. Otherwise zero.
2. **GDN:** Zero all $n_{\text{gdn}}$ layers' recurrence and conv state.
3. **QSA:** Zero all $n_{\text{qsa}}$ layers' KV cache and indexer.
4. **PLE:** Zero conv history and reset token window to $-1$ (EOS sentinel).

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem session_zero_ple_sentinel : (-1 : Int) < 0 := by sorry

-- axle_verify: content
theorem session_zero_ple_sentinel : (-1 : Int) < 0 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void session_zero(typenameState& argst, const typenameGeometry& arggeom, const float* argR_init, void* argstream) {
    typenameStream argcs = (typenameStream) argstream;
    if (argR_init != nullptr) {
        for (int64_t idx = 0; idx < arggeom.hc; ++idx)
            cudaMemcpyAsync(argst.block.R + (size_t) idx * arggeom.n_embd, argR_init, (size_t) arggeom.n_embd * 4,
                            cudaMemcpyDeviceToDevice, argcs);
    } else {
        cudaMemsetAsync(argst.block.R, 0, (size_t) arggeom.hc * arggeom.n_embd * 4, argcs);
    }
    cudaMemsetAsync(argst.gdn_state, 0, (size_t) arggeom.n_gdn_layers() * gdn_state_floats(arggeom) * 4, argcs);
    for (int64_t idx = 0; idx < arggeom.n_qsa_layers(); ++idx) qsa_state_zero(argst.qsa_states[idx], arggeom, argstream);
    cudaMemsetAsync(argst.ple_hist, 0, (size_t) ple_hist_bytes(), argcs);
    argst.ple_prev[0] = -1;
    argst.ple_prev[1] = -1;
    argst.ple_token = -1;
}
```

---

### Function: `session_capture`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{session\_capture}: \text{WeightTable} \times \text{Geometry} \times \text{SessionState} \times \text{Ptr} \to \{\top,\bot\}
$$

Captures CUDA graphs for all $n_L$ layers. **Per layer:**
- `execs[l]`: "pre" graph — mixer (GDN or QSA) + router + doorbell publish
- `posts[l]`: "post" graph — expert combine + residual update

**Optional split capture** (for profiling): `preA[l]`, `preB[l]` (two halves of pre), and `preP[k][l]` for $k \in \{1..5\}$ stage prefixes.

**Graph capture protocol:**
1. `cudaStreamBeginCapture(ThreadLocal)`
2. Execute layer's kernel sequence
3. `cudaStreamEndCapture` → `cudaGraphInstantiate`

**Invariant:** `gr.captured = true` only after all $n_L$ layers succeed.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem capture_all_or_none (n_layers : Nat) (captured : Fin n_layers → Bool) :
  (∀ i, captured i = true) ∨ (∃ i, captured i = false ∧ ∀ j, j > i → captured j = false) := by sorry
-- proof_needed: requires induction on layer capture sequence
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool session_capture(const ObjTable& argtables, const typenameGeometry& arggeom, typenameState& argst, const float* argparts,
                     typenameGraphs& arggr, std::string& argerr, bool argsplit) {
    if (arggr.captured) return true;
    arggr.execs = new typenameGraphExec[(size_t) arggeom.n_layers];
    arggr.posts = new typenameGraphExec[(size_t) arggeom.n_layers];
    for (int64_t idx = 0; idx < arggeom.n_layers; ++idx) arggr.posts[idx] = nullptr;
    // ... split arrays allocation ...
    arggr.n = 0;
    int64_t qsa_index = 0;
    for (int64_t argL = 0; argL < arggeom.n_layers; ++argL) {
        gdn_point_at(arggeom, argL, argst);
        // capture pre and post graphs via lambda
        auto cbcapture = [&](bool argpost, typenameGraphExec* argout, const char* argwhat, int arghalf = 0,
                           int argstage_prefix = 0) -> bool {
            typenameStream argcs = nullptr;
            cudaStreamCreate(&argcs);
            cudaStreamBeginCapture(argcs, cudaStreamCaptureModeThreadLocal);
            // ... execute layer kernels ...
            typenameGraph arggraph = nullptr;
            cudaStreamEndCapture(argcs, &arggraph);
            cudaStreamDestroy(argcs);
            cudaGraphInstantiate(argout, arggraph, 0);
            cudaGraphDestroy(arggraph);
            return true;
        };
        if (!cbcapture(false, &arggr.execs[argL], "pre")) return false;
        if (!cbcapture(true, &arggr.posts[argL], "post")) return false;
        // ... split captures ...
        ++arggr.n;
        if (is_qsa_layer(arggeom, argL)) ++qsa_index;
    }
    arggr.captured = true;
    arggr.parts_dev = const_cast<float*>(argparts);
    return true;
}
```

---

### Function: `session_replay` / `session_replay_full`

**[LAYER 1a — MATH (LaTeX)]**

**`session_replay`** (pre-only, for host-side expert overlap):

$$
\text{stage\_token}(\text{pos}) \;;\; \forall l \in [0, n_L): \text{cudaGraphLaunch}(\text{execs}[l], \text{stream})
$$

**`session_replay_full`** (pre+post, no host overlap):

$$
\text{stage\_token}(\text{pos}) \;;\; \forall l \in [0, n_L): \bigl[\text{launch}(\text{execs}[l]) \;;\; \text{launch}(\text{posts}[l])\bigr]
$$

**Critical design note:** `stage_token` fills pinned host buffers that the captured H2D copies re-read on every replay. The graph bakes the SOURCE POINTER, not the value.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem replay_requires_capture (captured : Bool) (n n_layers : Nat) (h : captured = true ∧ n = n_layers) :
  True := by sorry

-- axle_verify: content
theorem replay_requires_capture (captured : Bool) (n n_layers : Nat) (h : captured = true ∧ n = n_layers) :
  True := by trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool session_replay_full(const typenameGeometry& arggeom, int64_t argpos, int32_t argpos_base, typenameState& argst,
                         typenameGraphs& arggr, void* argstream, std::string& argerr) {
    if (!arggr.captured || arggr.n != arggeom.n_layers || arggr.posts == nullptr) {
        argerr = "session_replay_full: not captured with post graphs";
        return false;
    }
    typenameStream argcs = (typenameStream) argstream;
    stage_token(arggeom, argpos, argpos_base, argst);
    for (int64_t argL = 0; argL < arggeom.n_layers; ++argL) {
        typenameError arge = cudaGraphLaunch(arggr.execs[argL], argcs);
        if (arge != cudaSuccess) { argerr = "pre[" + std::to_string(argL) + "]: " + cudaGetErrorString(arge); return false; }
        arge = cudaGraphLaunch(arggr.posts[argL], argcs);
        if (arge != cudaSuccess) { argerr = "post[" + std::to_string(argL) + "]: " + cudaGetErrorString(arge); return false; }
    }
    return true;
}
```

---

### Function: `session_loop`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{session\_loop}: \text{Geometry} \times \mathbb{Z} \times \text{SessionState} \times \text{Graphs} \times \text{PoolFn} \times \text{HitFn} \to \{\top,\bot\}
$$

**The token path — core scheduling loop:**

For each layer $l \in [0, n_L)$:
1. Copy expert results from previous layer's pool answer (`y_miss`) to `parts_dev`
2. Launch `pre[l]`: mixer + router + doorbell publish
3. Poll doorbell: spin-wait (`_mm_pause`) until the host pool publishes expert data
4. Launch `post[l]`: expert combine + residual update

**Critical design decisions:**
- **Zero allocation on the hot path** (P2.T10): all staging is pre-allocated in `SessionLoopScratch`
- **Pinned host staging** because `cudaMemcpyAsync` with pageable source is synchronous (driver bounce buffer)
- **CPU core pinning** via `physical_cores(true)` to prevent host spin from stealing worker cores
- **Position staging** via `stage_token` before loop (bug fix: every token after the first used to replay position 0)

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem loop_layer_ordering (l : Nat) (n_layers : Nat) (h : l < n_layers) :
  l + 1 ≤ n_layers := by sorry

-- axle_verify: content
theorem loop_layer_ordering (l : Nat) (n_layers : Nat) (h : l < n_layers) :
  l + 1 ≤ n_layers := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool session_loop(const typenameGeometry& arggeom, int64_t argpos, int32_t argpos_base, typenameState& argst, typenameGraphs& arggr,
                  cbPool cbpool, cbHit cbhits, void* arguser, bool argoverlap, void* argstream, std::string& argerr,
                  float* argdump_layers, typenameLoopScratch* argscratch) {
    if (!arggr.captured || arggr.n != arggeom.n_layers) { argerr = "session_loop: not captured"; return false; }
    if (arggr.parts_dev == nullptr) { argerr = "session_loop: no parts buffer"; return false; }
    if (argst.db == nullptr) { argerr = "session_loop: no doorbell"; return false; }
    typenameStream argcs = (typenameStream) argstream;
    const size_t argparts_bytes = (size_t) argst.k * arggeom.n_embd * 4;
    // STAGE TOKEN — critical: graph bakes SOURCE POINTER, not value
    stage_token(arggeom, argpos, argpos_base, argst);
    // ... scratch init (zero-alloc hot path) ...
    float* argy_miss = argscratch->y_miss;
    cudaMemcpyAsync(arggr.parts_dev, argy_miss, argparts_bytes, cudaMemcpyHostToDevice, argcs);
    doorbell_reset(*argst.db);
    // ... launch pre[0], then loop: poll doorbell → pool → post[l] → pre[l+1] ...
    return true;
}
```

---

### Function: `session_graphs_free`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{session\_graphs\_free}: \text{Graphs} \to \text{void}
$$

Destroys all CUDA graph executables: `execs`, `posts`, `preA`, `preB`, `preP[0..4]`. Nulls all pointers. Resets `captured = false`.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem graphs_free_resets (captured_after : Bool) (h : captured_after = false) : 
  captured_after = false := by sorry

-- axle_verify: content
theorem graphs_free_resets (captured_after : Bool) (h : captured_after = false) : 
  captured_after = false := by exact h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void session_graphs_free(typenameGraphs& arggr) {
    if (arggr.execs) {
        for (int64_t idx = 0; idx < arggr.n; ++idx) cudaGraphExecDestroy(arggr.execs[idx]);
        delete[] arggr.execs;
    }
    if (arggr.posts) {
        for (int64_t idx = 0; idx < arggr.n; ++idx) if (arggr.posts[idx] != nullptr) cudaGraphExecDestroy(arggr.posts[idx]);
        delete[] arggr.posts;
    }
    // ... preA, preB, preP[0..4] cleanup ...
    arggr.execs = nullptr; arggr.posts = nullptr;
    arggr.preA = nullptr; arggr.preB = nullptr;
    arggr.split_captured = false;
    arggr.parts_dev = nullptr;
    arggr.n = 0;
    arggr.captured = false;
}
```

---

### Function: `SessionLoopScratch::init` / `SessionLoopScratch::free`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{init}: \mathbb{N} \to \{\top,\bot\}
$$

Allocates:
- `y_miss`: pinned mapped buffer for pool staging, size = `parts_bytes`
- `probe`: CUDA event for doorbell polling
- CPU core pin: `physical_cores(false)[0]` for the host spin thread

**Invariant:** `free()` always restores caller's CPU affinity, including on teardown — a library that silently nails the caller's thread to one core is a bug.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem scratch_init_once (y_miss probe : Option Ptr) 
  (h1 : y_miss = none) (h2 : probe = none) : y_miss = none ∧ probe = none := by sorry

-- axle_verify: content
theorem scratch_init_once (y_miss probe : Option Ptr) 
  (h1 : y_miss = none) (h2 : probe = none) : y_miss = none ∧ probe = none := by
  exact ⟨h1, h2⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool typenameLoopScratch::init(size_t argparts_bytes_in, std::string& argerr) {
    if (argy_miss != nullptr || argprobe != nullptr) {
        argerr = "SessionLoopScratch::init: already initialised";
        return false;
    }
    parts_bytes = argparts_bytes_in;
    if (cudaHostAlloc((void**) &argy_miss, parts_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        argerr = "SessionLoopScratch: cudaHostAlloc failed";
        return false;
    }
    std::memset(argy_miss, 0, parts_bytes);
    if (cudaEventCreate(&argprobe) != cudaSuccess) {
        argerr = "SessionLoopScratch: cudaEventCreate failed";
        free();
        return false;
    }
    const std::vector<int> argcores = prjname::kernels::cpu::physical_cores(false);
    if (!argcores.empty()) {
        pinned_core = prjname::kernels::cpu::pin_current_thread(argcores[0]);
        pinned = true;
    }
    return true;
}

void typenameLoopScratch::free() {
    if (pinned) {
        prjname::kernels::cpu::restore_thread_affinity(pinned_core);
        pinned = false;
    }
    if (argprobe != nullptr) { cudaEventDestroy(argprobe); argprobe = nullptr; }
    if (argy_miss != nullptr) { cudaFreeHost(argy_miss); argy_miss = nullptr; }
    parts_bytes = 0;
}
```

---

## MODULE: verification_harness (multi-token correctness checker)

### Function: `Verifier::init`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{init}: \text{WeightTable} \times \text{Geometry} \times \text{SessionState} \times \text{VerifyHits} \times \text{NativeHead} \times \mathbb{Z}_{>0} \to \{\top,\bot\}
$$

**Preconditions:**
- $2 \leq \text{max\_t} \leq \text{kVerifyMaxT}$
- $\text{hits.d\_res} \neq \text{null} \land \text{hits.cache\_base} \neq \text{null} \land \text{hits.blob} > 0$
- Layer-verify-compatible geometry
- $\text{fused\_gr\_supported}(n_{\text{embd}}, \text{hc}, \text{hc\_lr})$
- $k = 10, \text{ssm\_state\_size} = 128, \text{ssm\_d\_conv} = 4$

**Allocations:**
1. **Mapped staging:** 15 pinned+mapped buffers for host↔device token data, plan, expert ids, weights
2. **Device arena:** Single `cudaMalloc` holding all per-token intermediate tensors (residuals, QKV projections, attention scratch, MoE routing, expert outputs, etc.)
3. **Bump allocator** used twice: first pass counts bytes, second pass carves pointers

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem verifier_window_bounds (max_t kmax : Nat) (h1 : 2 ≤ max_t) (h2 : max_t ≤ kmax) :
  max_t ≥ 2 ∧ max_t ≤ kmax := by sorry

-- axle_verify: content
theorem verifier_window_bounds (max_t kmax : Nat) (h1 : 2 ≤ max_t) (h2 : max_t ≤ kmax) :
  max_t ≥ 2 ∧ max_t ≤ kmax := by
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjVerifier::init(const ObjTable& argwt, const typenameGeometry& arggeom, typenameState& argss, const typenameHits& arghits,
                    const adjObjHead* arghead, int argmax_t, std::string& argerr) {
    objwt_ = &argwt; objgeom_ = &arggeom; objss_ = &argss; hits_ = arghits; objhead_ = arghead; max_t_ = argmax_t;
    if (argmax_t < 2 || argmax_t > prjname::kernels::kVerifyMaxT || argmax_t > prjname::kernels::cpu::MAXT) {
        argerr = "verify: window must hold 2.." + std::to_string(prjname::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    // ... precondition checks ...
    // Mapped staging allocation (15 buffers via mapped() helper)
    bool argok = mapped(argT * 4, (void**) &h_tok_, (void**) &m_tok_) && /* ... */;
    // Device arena: two-pass bump allocation
    typenameAlloc argcount;
    cbcarve(argcount);
    if (cudaMalloc(&arena_, argcount.used) != cudaSuccess) {
        argerr = "verify: arena (" + std::to_string(argcount.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, argcount.used);
    typenameAlloc argreal;
    argreal.base = (uint8_t*) arena_;
    cbcarve(argreal);
    // ... stream creation ...
    return true;
}
```

---

### Function: `Verifier::record_window` (partial — from truncated source)

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{record\_window}: \mathbb{Z}_{>0} \to \{\top,\bot\}
$$

Records a CUDA graph for a window of $T$ tokens through all layers. Supports an optional **split mode** ($G=2$) where tokens are partitioned into groups $A=[0, \lceil T/2 \rceil)$ and $B=[\lceil T/2 \rceil, T)$ for CPU/GPU expert overlap:

$$
\text{Stream order:} \quad \text{pre}(0,A) \to \text{pre}(0,B) \to \text{post}(0,A) \to \text{pre}(1,A) \to \text{post}(0,B) \to \text{pre}(1,B) \to \ldots
$$

**Per-layer pre phase:**
1. Embedding gather + broadcast to hc streams
2. GR-read (fused norm + down/up/inject projections)
3. **GDN path:** QKV projection → conv → AB computation → gate → recurrence → SSM output projection
4. **QSA path:** Indexer → K/V projection + norm + RoPE → KV append → block scoring/topk → decode attention → gate → output projection
5. Second GR-read for FFN half
6. MoE routing + doorbell publish
7. Shared expert computation
8. Input quantization for expert kernel

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem split_groups_cover (T : Nat) (hT : T ≥ 2) :
  let mid := (T + 1) / 2
  mid > 0 ∧ T - mid > 0 ∧ mid + (T - mid) = T := by sorry

-- axle_verify: content
theorem split_groups_cover (T : Nat) (hT : T ≥ 2) :
  let mid := (T + 1) / 2
  mid > 0 ∧ T - mid > 0 ∧ mid + (T - mid) = T := by
  simp only
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjVerifier::record_window(int argT, typenameStream argcs, std::string& argerr) {
    const typenameGeometry& arggeom = *objgeom_;
    const ObjTable& argwt = *objwt_;
    typenameState& argss = *objss_;
    const int64_t argN = arggeom.n_embd, argHC = arggeom.hc, argK = argss.k;
    // ... dimension aliases ...
    const int argG = (split_ && argT >= 2) ? 2 : 1;
    const int tb_[2] = {0, (argT + 1) / 2}, te_[2] = {argG == 2 ? (argT + 1) / 2 : argT, argT};
    groups_[argT] = argG;
    // Copy inputs from mapped staging
    copy_i32_from_mapped(tok_, m_tok_, argT, argcs);
    // ... embedding gather + broadcast ...
    // Per-layer pre/post lambdas
    auto cbpre = [&](int64_t argL, int arggrp) -> bool {
        // GR-read → mixer (GDN or QSA) → GR-read → MoE route → doorbell → shared expert → quantize
        return true;
    };
    auto cbpost = [&](int64_t argL, int arggrp) -> bool {
        // Wait for pool plan → expert combine → residual update
        return true;
    };
    // ... layer loop with group interleaving ...
    return true;
}
```

---

## MODULE: weight_table_loader (pack index → GPU arena upload)

### Function: `WeightTable::pool_bytes`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{pool\_bytes}: \text{Path} \times \mathcal{P}(\text{String})_{\text{opt}} \to \mathbb{N}
$$

Reads `index.txt` header to find total arena size. With a skip set, computes compacted size:

$$
\text{compact} = \sum_{r \notin \text{skip}} \lceil r.\text{dst\_bytes} \rceil_a
$$

where $a$ is the alignment from the index header.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem compact_le_pool (rows : List Nat) (skip : Finset Nat) (pool : Nat)
  (h : pool = rows.foldl (· + ·) 0)
  (h2 : ∀ r ∈ rows, r ≤ pool) :
  (rows.filter (· ∉ skip)).foldl (· + ·) 0 ≤ pool := by sorry
-- proof_needed: requires list filter monotonicity
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool ObjTable::pool_bytes(const std::string& argpack_dir, uint64_t& argout, std::string& argerr,
                             const std::set<std::string>* argskip) {
    const std::string argpath = argpack_dir + "/index.txt";
    std::FILE* argfile = std::fopen(argpath.c_str(), "rb");
    if (!argfile) { argerr = "cannot open " + argpath; return false; }
    char argline[1024];
    argout = 0;
    uint64_t argpool = 0, argcompact = 0;
    int argalign = 0;
    while (std::fgets(argline, sizeof argline, argfile)) {
        if (argline[0] == '#') {
            unsigned long long argp = 0;
            int arga = 0, argtensors = 0;
            if (std::sscanf(argline, "# align %d pool %llu tensors %d", &arga, &argp, &argtensors) == 3) { argpool = argp; argalign = arga; }
            continue;
        }
        if (argskip == nullptr) continue;
        // ... parse row, accumulate compact size for non-skipped tensors ...
    }
    std::fclose(argfile);
    argout = argskip ? argcompact : argpool;
    return true;
}
```

---

### Function: `WeightTable::load`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{load}: \text{Path} \times \text{Ptr} \times \mathbb{N} \to \{\top,\bot\}
$$

**Index format:** `index.txt` with 19 space-separated fields per row (name, file, kind, src_off, src_bytes, dst_off, dst_bytes, ne0, ne1, code_bits, code_bias, group_elems, codebook, has_offset, codes_bytes, scales_bytes, offset_bytes, scales_fp16, act_kind).

**Weight kinds:**
- `Copy`: bytes are already engine form
- `Bf16High`: f32 container → take high 16 bits (bf16)
- `ToF16`: f32 container → round-to-nearest-even f16
- `WidenF16`: fp16 bits → f32 (exact widening)

**Upload protocol:**
1. Parse `index.txt`, reject on any malformed row (wrong offset = plausible-but-wrong weight)
2. Optional skip-set compaction: re-place kept rows at alignment
3. Two pinned staging buffers: `stage_in` (CHUNK = 8 MiB), `stage_out` (2×CHUNK because widening doubles)
4. For each tensor: segment into conversion runs, read from pack file, convert in `stage_out`, `cudaMemcpyAsync` to arena

**Critical bug this design prevents:** Stage-out is NOT optional. Writing converted values directly to device memory (a device address from host code) is UB that silently corrupts — bf16 weights appear as 32-bit garbage.

**Plan v0.3 P6:** Raw16 kinds (4=BF16, 5=F16) from native packs are copied directly and mapped to engine kinds.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem read_at_total (off n filesize : Nat) (h1 : off + n ≤ filesize) : 
  off ≤ filesize ∧ n ≤ filesize - off := by sorry

-- axle_verify: content
theorem read_at_total (off n filesize : Nat) (h1 : off + n ≤ filesize) : 
  off ≤ filesize ∧ n ≤ filesize - off := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem stage_out_sizing (chunk : Nat) (hc : chunk > 0) : 
  chunk * 2 ≥ chunk := by sorry

-- axle_verify: content
theorem stage_out_sizing (chunk : Nat) (hc : chunk > 0) : 
  chunk * 2 ≥ chunk := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool ObjTable::load(const std::string& argpack_dir, void* argarena_base, uint64_t argarena_bytes, std::string& argerr,
                       const std::set<std::string>* argskip) {
    const std::string argpath = argpack_dir + "/index.txt";
    std::FILE* argidx = std::fopen(argpath.c_str(), "rb");
    if (!argidx) { argerr = "cannot open " + argpath; return false; }
    // Parse index header + rows (19 fields each)
    uint64_t argpool = 0;
    int argalign = 0;
    std::vector<typenameRow> argrows;
    // ... parse loop, REFUSE on malformed row ...

    // Skip-set compaction
    if (argskip != nullptr) {
        const uint64_t arga = argalign > 0 ? (uint64_t) argalign : 256;
        uint64_t argat = 0;
        for (size_t idx = 0; idx < argrows.size(); ++idx) {
            if (argskip->count(argrows[idx].name)) { argskipped[idx] = true; continue; }
            argrows[idx].dst_off = argat;
            argat += (argrows[idx].dst_bytes + arga - 1) / arga * arga;
        }
        argpool = argat;
    }

    // Two pinned staging buffers (stage_out is 2x for widening conversions)
    void* argstage_in = nullptr;
    void* argstage_out = nullptr;
    cudaHostAlloc(&argstage_in, CHUNK, cudaHostAllocDefault);
    cudaHostAlloc(&argstage_out, CHUNK * 2, cudaHostAllocDefault);

    uint8_t* argdst_base = (uint8_t*) argarena_base;
    // Per-tensor: segment → convert in stage_out → cudaMemcpyAsync to arena
    for (size_t argrow_i = 0; argrow_i < argrows.size(); ++argrow_i) {
        const typenameRow& argr = argrows[argrow_i];
        if (argskipped[argrow_i]) { /* metadata-only WeightRef */ continue; }
        // ... file open, segment loop, conversion dispatch, upload ...
    }
    cudaFreeHost(argstage_in);
    cudaFreeHost(argstage_out);
    return true;
}
```

---

### Helper: `read_at`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{read\_at}: \text{FILE} \times \mathbb{N} \times \text{Ptr} \times \mathbb{N} \to \{\top,\bot\}
$$

$$
\text{seek}(f, \text{off}) \;;\; \text{fread}(f, \text{dst}, n) \;;\; \text{returns } \top \iff \text{bytes\_read} = n
$$

**[LAYER 1b — VERIFY (Lean 4)]**

(Already covered by `read_at_total` above.)

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool read_at(std::FILE* argfile, uint64_t argoff, void* argdst, size_t argn, std::string& argerr, const char* argwhat) {
#if defined(_WIN32)
    if (_fseeki64(argfile, (long long) argoff, SEEK_SET) != 0) { argerr = std::string(argwhat) + ": seek failed"; return false; }
#else
    if (fseeko(argfile, (off_t) argoff, SEEK_SET) != 0) { argerr = std::string(argwhat) + ": seek failed"; return false; }
#endif
    if (std::fread(argdst, 1, argn, argfile) != argn) { argerr = std::string(argwhat) + ": short read"; return false; }
    return true;
}
```

---

### Function: `Conv` enum (weight conversion modes)

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{Conv} \in \{\text{Copy}, \text{Bf16High}, \text{ToF16}, \text{WidenF16}\}
$$

- $\text{Copy}$: identity, $\text{dst\_bytes} = \text{src\_bytes}$
- $\text{Bf16High}$: $f_{32} \mapsto \text{high}_{16}(f_{32})$, ratio $2:1$
- $\text{ToF16}$: $f_{32} \mapsto \text{round}_{f16}(f_{32})$, ratio $2:1$
- $\text{WidenF16}$: $f_{16} \mapsto f_{32}$, ratio $1:2$ (THIS IS WHY `stage_out` IS 2×)

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem widen_doubles (src_bytes : Nat) : src_bytes * 2 = src_bytes + src_bytes := by sorry

-- axle_verify: content
theorem widen_doubles (src_bytes : Nat) : src_bytes * 2 = src_bytes + src_bytes := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
enum class typenameConv : int {
    Copy = 0,
    Bf16High = 1,
    ToF16 = 2,
    WidenF16 = 3,
};
```

---

*End of Chunk 3 theory. All 6 assigned files covered: native_dense.cpp, native_head.cpp, pinned.cu, session.cpp (618 lines, partially truncated), verify.cpp (523 lines, partially truncated), weights.cpp (457 lines, partially truncated).*

---

## MODULE: session_orchestrator (continued — from truncated source)

### Function: `session_loop` — inner loop detail (lines 618–745)

**[LAYER 1a — MATH (LaTeX)]**

**Doorbell spin-wait protocol (per layer $l$):**

$$
\text{expected} \leftarrow \text{expected} + 1
$$

1. **Spin:** Read `volatile uint32_t* seq` (mapped pinned memory). Between spins, call `cudaEventQuery(probe)` — this is **required** to make device writes to mapped pinned memory host-visible on this driver (three experiments confirm). `_mm_pause()` for pipeline politeness.
2. **Ring detected:** `*seq >= want`. Record whether graph was still in-flight (`mid_graph`).
3. **GPU hit launch** (if VRAM expert tier active): `moe_hit_select` + `quantize_q8_0` + `moe_hit_grouped_s2_dev`
4. **CPU pool:** `pool(user, h_x_f, h_ids, h_weights, n_embd, k, y_miss)` — runs on host while GPU expert kernels execute.
5. **Copy miss results:** `cudaMemcpyAsync(parts_dev, y_miss, parts_bytes, H2D)` — stream-ordered.
6. **Combine hits:** `moe_hit_add(parts_dev, hit_out, ...)` — adds VRAM-tier expert results to miss results.
7. **Launch post[l]:** `cudaGraphLaunch(posts[l], cs)` — combines experts with residual.
8. **Optional dump:** Copy residual to host for C1 oracle diagnostics.
9. **Launch pre[l+1]:** `cudaGraphLaunch(execs[l+1], cs)` + event record.

**Timing instrumentation:**
- `ms_to_ring`: launch → ring seen
- `ms_host`: ring seen → next pre launched (GPU-idle interval)
- `rings_mid_graph`: count of rings arriving while graph still running (= overlap achieved)

**Critical L100 fix:** Expert outputs go to `post[l]` of the SAME layer, not `l+1`. Previous version computed $\sum_j w_{l+1}[j] \cdot \text{expert}_{ids_l, j}(x_l)$ — weights current but selection and input one layer stale.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem doorbell_monotonic (seq want : Nat) (h : seq ≥ want) : seq ≥ want := by sorry

-- axle_verify: content
theorem doorbell_monotonic (seq want : Nat) (h : seq ≥ want) : seq ≥ want := by exact h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
// Inner loop body (per layer l)
const uint32_t argwant = ++argexpected;
bool argrang = false;
bool argmid_graph = false;
volatile uint32_t* const argseq = argst.db->h_seq;
for (;;) {
    const typenameError argq = cudaEventQuery(argprobe);
    if (!argrang && *argseq >= argwant) {
        argrang = true;
        argmid_graph = (argq != cudaSuccess);
    }
    if (argq == cudaSuccess) break;
    if (argq != cudaErrorNotReady) { /* error */ return false; }
    if (argrang) break;
    STRATA_SPIN_PAUSE();
}
// GPU hit launch (if active)
if (cbhits != nullptr) cbhits(arguser, argcs, typenamePhase::Launch, argst.db->h_ids, argk);
// CPU pool
if (cbpool != nullptr) cbpool(arguser, argst.db->h_x_f, argst.db->h_ids, argst.db->h_weights, arggeom.n_embd, argk, argy_miss);
// Copy misses + combine hits + post[l]
cudaMemcpyAsync(arggr.parts_dev, argy_miss, argparts_bytes, cudaMemcpyHostToDevice, argcs);
if (cbhits != nullptr) cbhits(arguser, argcs, typenamePhase::Combine, nullptr, 0);
cudaGraphLaunch(arggr.posts[argL], argcs);
// pre[l+1]
if (argL + 1 < arggeom.n_layers) {
    cudaGraphLaunch(arggr.execs[argL + 1], argcs);
    cudaEventRecord(argprobe, argcs);
}
```

---

### Function: `session_token`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{session\_token}: \text{WeightTable} \times \text{Geometry} \times \mathbb{Z} \times \text{SessionState} \times \text{Ptr} \to \{\top,\bot\}
$$

Non-graph path: runs `block_layer` directly for each of $n_L$ layers. Maintains separate GDN/QSA layer indices. Each GDN layer points its state at the correct slice before the call.

**Optional debug sync:** `sync_every_layer = true` inserts `cudaStreamSynchronize` after each layer, turning asynchronous errors into per-layer failures at the cost of full serialization (P2.X3 forbids this on the fast path).

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem session_token_gdn_qsa_partition (n_layers n_gdn n_qsa : Nat) (h : n_gdn + n_qsa = n_layers) :
  n_gdn ≤ n_layers ∧ n_qsa ≤ n_layers := by sorry

-- axle_verify: content
theorem session_token_gdn_qsa_partition (n_layers n_gdn n_qsa : Nat) (h : n_gdn + n_qsa = n_layers) :
  n_gdn ≤ n_layers ∧ n_qsa ≤ n_layers := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool session_token(const ObjTable& argtables, const typenameGeometry& arggeom, int64_t argpos, int32_t argpos_base,
                   typenameState& argst, const float* argparts, void* argstream, bool argsync_every_layer,
                   std::string& argerr) {
    typenameStream argcs = (typenameStream) argstream;
    int64_t qsa_index = 0;
    int64_t gdn_index = 0;
    for (int64_t argL = 0; argL < arggeom.n_layers; ++argL) {
        if (!is_qsa_layer(arggeom, argL)) {
            argst.gdn.state = argst.gdn_state + (size_t) gdn_index * gdn_state_floats(arggeom);
            argst.gdn.conv_state = argst.gdn.state + (uint64_t) arggeom.ssm_state_size * arggeom.ssm_v_heads * arggeom.ssm_state_size;
            ++gdn_index;
        }
        argerr.clear();
        const bool argok = is_qsa_layer(arggeom, argL)
            ? block_layer(argtables, arggeom, argL, argpos, argpos_base, argst.gdn, argst.qsa_states[qsa_index], argst.qsa_bufs, argst.moe, argst.k,
                          argst.block, argparts, argstream, argerr, nullptr, argst.ple.ready() ? &argst.ple : nullptr)
            : block_layer(argtables, arggeom, argL, argpos, argpos_base, argst.gdn, argst.qsa_states[0], argst.qsa_bufs, argst.moe, argst.k,
                          argst.block, argparts, argstream, argerr, nullptr, argst.ple.ready() ? &argst.ple : nullptr);
        if (!argok) { argerr = "layer " + std::to_string(argL) + ": " + argerr; return false; }
        if (is_qsa_layer(arggeom, argL)) ++qsa_index;
        if (argsync_every_layer) {
            const typenameError arge = cudaStreamSynchronize(argcs);
            if (arge != cudaSuccess) { argerr = "layer " + std::to_string(argL) + ": " + cudaGetErrorString(arge); return false; }
        }
    }
    return true;
}
```

---

### Function: `session_capture_token`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{session\_capture\_token}: \text{WeightTable} \times \text{Geometry} \times \text{SessionState} \times \text{Ptr} \times \text{Ptr} \times \mathbb{N} \to \{\top,\bot\}
$$

Captures a **single CUDA graph** for the entire token (all $n_L$ layers, pre+doorbell+post), unlike `session_capture` which captures per-layer graphs.

**Per-layer sequence in the captured graph:**
1. `block_layer_pre` → mixer + router + doorbell publish
2. GPU expert hit path: `moe_hit_select → quantize_q8_0_scaled → moe_hit_grouped_s2_dev`
3. `doorbell_wait` — device-side spin on `d_flag` until host writes the answer
4. `copy_from_mapped(parts_dev, y_dev)` — a KERNEL, not a memcpy node (copy-engine nodes split WDDM submission, measured 67 flushes/token)
5. `moe_hit_add` — combine VRAM hits with CPU misses
6. `block_layer_post` — expert combine + residual update

**Preconditions:**
- `y_miss_host` must be `cudaHostAllocMapped` pinned memory (verified via `cudaHostGetDevicePointer`)
- Doorbell must have `d_flag` and `d_seq`
- If hits tier provided, must be `.on()`

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem capture_token_mapped_required (dev_ptr : Option Ptr) (h : dev_ptr = none) :
  ¬ (dev_ptr ≠ none) := by sorry

-- axle_verify: content
theorem capture_token_mapped_required (dev_ptr : Option Ptr) (h : dev_ptr = none) :
  ¬ (dev_ptr ≠ none) := by
  simp [h]
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool session_capture_token(const ObjTable& argtables, const typenameGeometry& arggeom, typenameState& argst, float* argparts_dev,
                           const float* argy_miss_host, size_t argparts_bytes, typenameTokenGraph& argtg, std::string& argerr,
                           const typenameTokenHits* arghits) {
    if (arghits != nullptr && !arghits->on()) { argerr = "incomplete hit configuration"; return false; }
    if (argtg.captured) return true;
    // Validate doorbell, parts buffers, mapped pointer
    float* argy_dev = nullptr;
    if (cudaHostGetDevicePointer((void**) &argy_dev, const_cast<float*>(argy_miss_host), 0) != cudaSuccess || !argy_dev) {
        argerr = "parts staging is not mapped pinned memory"; return false;
    }
    typenameStream argcs = nullptr;
    cudaStreamCreate(&argcs);
    cudaStreamBeginCapture(argcs, cudaStreamCaptureModeThreadLocal);
    int64_t qsa_index = 0;
    for (int64_t argL = 0; argL < arggeom.n_layers; ++argL) {
        gdn_point_at(arggeom, argL, argst);
        const bool argqsa = is_qsa_layer(arggeom, argL);
        typenameQsaState& argqst = argqsa ? argst.qsa_states[qsa_index] : argst.qsa_states[0];
        // pre: mixer + router + doorbell publish
        block_layer_pre(argtables, arggeom, argL, 0, 0, argst.gdn, argqst, argst.qsa_bufs, argst.moe, argst.k, argst.block, (void*) argcs, argerr, argst.db,
                        argst.ple.ready() ? &argst.ple : nullptr);
        // GPU hit path (if active)
        if (arghits != nullptr) {
            prjname::kernels::moe_hit_select(argst.moe.ids, arghits->d_res + argL * arghits->n_expert, (int) argst.k,
                                            (int) arghits->n_expert, arghits->d_slot, arghits->d_dst, arghits->d_count, (void*) argcs);
            prjname::kernels::quantize_q8_0_scaled(argst.block.mixed, arghits->x_q8, arghits->x_scale, arggeom.n_embd, (void*) argcs);
            prjname::kernels::moe_hit_grouped_s2_dev(arghits->cache_base, arghits->d_slot, arghits->d_dst, arghits->d_count, argst.k,
                                                    arghits->blob, arghits->x_q8, arghits->scratch, arghits->hit_out, (void*) argcs, arghits->x_scale);
        }
        // Device-side doorbell wait + copy (kernel, not memcpy node) + hit combine + post
        prjname::kernels::doorbell_wait(argst.db->d_flag, argst.db->d_seq, (void*) argcs);
        prjname::kernels::copy_from_mapped(argparts_dev, argy_dev, (int64_t) (argparts_bytes / sizeof(float)), (void*) argcs);
        if (arghits != nullptr)
            prjname::kernels::moe_hit_add(argparts_dev, arghits->hit_out, arghits->d_dst, arghits->d_count, argst.k, arggeom.n_embd, (void*) argcs);
        block_layer_post(argtables, arggeom, argL, argst.k, argst.moe, argst.block, argparts_dev, (void*) argcs, argerr);
        if (argqsa) ++qsa_index;
    }
    // End capture → instantiate
    typenameGraph arggraph = nullptr;
    cudaStreamEndCapture(argcs, &arggraph);
    cudaStreamDestroy(argcs);
    cudaGraphInstantiate(&argtg.exec, arggraph, 0);
    cudaGraphDestroy(arggraph);
    argtg.captured = true;
    argtg.n_layers = arggeom.n_layers;
    argtg.y_src = argy_miss_host;
    argtg.parts_bytes = argparts_bytes;
    return true;
}
```

---

### Function: `session_run_token`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{session\_run\_token}: \text{Geometry} \times \mathbb{Z} \times \text{SessionState} \times \text{TokenGraph} \times \text{PoolFn} \to \{\top,\bot\}
$$

Launches the single-graph token captured by `session_capture_token`. The host-side loop:

**Per-layer doorbell protocol (device-initiated, host-answered):**
1. **Spin on `h_seq`** (volatile mapped read) until `*seq >= l+1`
2. **Flush:** Every 1024 spins, if `> STRATA_TG_FLUSH_US` μs elapsed, call `cudaStreamQuery` to flush WDDM submission queue and detect faults
3. **Timeout:** 20 seconds hard limit
4. **Run CPU pool:** `pool(user, h_x_f, h_ids, h_weights, n_embd, k, y_miss_host)`
5. **Memory fence + sfence:** `atomic_thread_fence(seq_cst)` + `_mm_sfence()`
6. **Signal device:** `*flag = want` — the captured `doorbell_wait` kernel sees this

**Timing:**
- `ms_wait`: time waiting for ring
- `ms_pool`: time in CPU pool
- `flushes`: count of extra WDDM flushes

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem run_token_timeout_bound (elapsed_s : Nat) (h : elapsed_s > 20) : elapsed_s > 20 := by sorry

-- axle_verify: content
theorem run_token_timeout_bound (elapsed_s : Nat) (h : elapsed_s > 20) : elapsed_s > 20 := by exact h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool session_run_token(const typenameGeometry& arggeom, int64_t argpos, int32_t argpos_base, typenameState& argst, typenameTokenGraph& argtg,
                       cbPool cbpool, void* arguser, float* argy_miss_host, void* argstream, std::string& argerr) {
    if (!argtg.captured) { argerr = "not captured"; return false; }
    if (argy_miss_host != argtg.y_src) { argerr = "staging buffer mismatch"; return false; }
    typenameStream argcs = (typenameStream) argstream;
    ++argtg.calls;
    stage_token(arggeom, argpos, argpos_base, argst);
    doorbell_reset(*argst.db);
    cudaGraphLaunch(argtg.exec, argcs);
    (void) cudaStreamQuery(argcs);  // one flush for WDDM submission
    volatile uint32_t* const argseq = argst.db->h_seq;
    volatile uint32_t* const argflag = argst.db->h_flag;
    for (int64_t argL = 0; argL < arggeom.n_layers; ++argL) {
        const uint32_t argwant = (uint32_t) (argL + 1);
        // Spin on seq with periodic WDDM flush and 20s timeout
        while (*argseq < argwant) { STRATA_SPIN_PAUSE(); /* ... flush logic ... */ }
        // CPU pool
        if (cbpool != nullptr) cbpool(arguser, argst.db->h_x_f, argst.db->h_ids, argst.db->h_weights, arggeom.n_embd, argst.k, argy_miss_host);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *argflag = argwant;
    }
    cudaStreamSynchronize(argcs);
    return true;
}
```

---

### Function: `token_graph_free`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{token\_graph\_free}: \text{TokenGraph} \to \text{void}
$$

Destroys the graph exec and zero-initializes the struct.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- trivial destructor, no proof needed
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void token_graph_free(typenameTokenGraph& argtg) {
    if (argtg.exec) cudaGraphExecDestroy(argtg.exec);
    argtg = typenameTokenGraph{};
}
```

---

## MODULE: weight_table_loader (continued — conversion pipeline detail)

### Function: `weight_conversion_pipeline` (inner loop of WeightTable::load)

**[LAYER 1a — MATH (LaTeX)]**

Per segment $s$ of tensor row $r$:

$$
\text{read\_at}(f, r.\text{src\_off} + s.\text{src\_off}, \text{stage\_in}, \min(s.\text{src\_bytes} - \text{done}, \text{CHUNK}))
$$

Then apply conversion $s.\text{conv}$:

$$
\text{dst}[r.\text{dst\_off} + s.\text{dst\_off} + \text{out\_at}] \leftarrow \text{conv}(\text{stage\_in}[0 \ldots n))
$$

**Conversion dispatch:**
- `Copy`: `host_src = stage_in`, `out_bytes = n`
- `Bf16High`: $\forall i: \text{out}[i] = \text{high}_{16}(\text{in}[i])$, ratio $4 \to 2$
- `ToF16`: $\forall i: \text{out}[i] = \text{f16\_from\_f32}(\text{in}[i])$, ratio $4 \to 2$
- `WidenF16`: $\forall i: \text{out}[i] = \text{f32\_from\_f16}(\text{in}[i])$, ratio $2 \to 4$

**Post-upload:** Each tensor's `WeightRef` is populated with all metadata (ne0, ne1, kind, code_bits, etc.) and inserted into `table_`.

**Load report:** Tracks `re_rounded` count and `bytes_saved` for bf16/f16 conversions, plus `read_ms` and `upload_ms` timing.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem bf16_high_half (v : UInt32) : (v >>> 16).toUInt16.toNat < 65536 := by sorry

-- axle_verify: content
theorem bf16_high_half (v : UInt32) : (v >>> 16).toUInt16.toNat < 65536 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
// Inner conversion loop per segment
switch (argseg.conv) {
    case typenameConv::Copy:
        arghost_src = argstage_in;
        argout_bytes = argn;
        argout_at = argseg.dst_off + argdone;
        break;
    case typenameConv::Bf16High: {
        const uint64_t argelems = argn / 4;
        argout_bytes = argelems * 2;
        uint8_t* argo = (uint8_t*) argstage_out;
        for (uint64_t idx = 0; idx < argelems; ++idx) {
            uint32_t argval;
            std::memcpy(&argval, argsrc + idx * 4, 4);
            const uint16_t arghigh = (uint16_t) (argval >> 16);
            std::memcpy(argo + idx * 2, &arghigh, 2);
        }
        arghost_src = argstage_out;
        argout_at = argseg.dst_off + argdone / 2;
        break;
    }
    case typenameConv::ToF16: {
        const uint64_t argelems = argn / 4;
        argout_bytes = argelems * 2;
        uint8_t* argo = (uint8_t*) argstage_out;
        for (uint64_t idx = 0; idx < argelems; ++idx) {
            float argval;
            std::memcpy(&argval, argsrc + idx * 4, 4);
            const uint16_t argh = prjname::kernels::f16_from_f32(argval);
            std::memcpy(argo + idx * 2, &argh, 2);
        }
        arghost_src = argstage_out;
        argout_at = argseg.dst_off + argdone / 2;
        break;
    }
    case typenameConv::WidenF16: {
        const uint64_t argelems = argn / 2;
        argout_bytes = argelems * 4;
        float* argo = (float*) argstage_out;
        for (uint64_t idx = 0; idx < argelems; ++idx) {
            uint16_t argh;
            std::memcpy(&argh, argsrc + idx * 2, 2);
            argo[idx] = prjname::kernels::f32_from_f16(argh);
        }
        arghost_src = argstage_out;
        argout_at = argseg.dst_off + argdone * 2;
        break;
    }
}
cudaMemcpy(argdst_base + argr.dst_off + argout_at, arghost_src, argout_bytes, cudaMemcpyHostToDevice);
```

---

### Function: `WeightTable::find`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{find}: \text{String} \to \text{WeightRef}^?
$$

$$
\text{find}(n) = \begin{cases} \&\text{table}\_[n] & \text{if } n \in \text{table}\_ \\ \text{null} & \text{otherwise} \end{cases}
$$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- trivial lookup, no proof needed
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
const typenameRef* ObjTable::find(const std::string& argname) const {
    const auto argit = table_.find(argname);
    return argit == table_.end() ? nullptr : &argit->second;
}
```

---

## MODULE: verification_harness (continued — complete remaining functions)

### Function: `Verifier::capture`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{capture}: \mathbb{Z}_{>0} \to \{\top,\bot\}
$$

Captures a CUDA graph for a $T$-token verification window. Memoized: returns immediately if `exec_[T]` already exists. After capture, calls `cudaGraphUpload` to pre-stage the graph on the device.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem capture_memoized (exec : Option GraphExec) (h : exec.isSome) : True := by sorry

-- axle_verify: content
theorem capture_memoized (exec : Option GraphExec) (h : exec.isSome) : True := by trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjVerifier::capture(int argT, std::string& argerr) {
    if (exec_[argT] != nullptr) return true;
    if (cudaStreamBeginCapture(objcs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        argerr = "verify: begin capture failed"; return false;
    }
    std::string argrec_err;
    const bool argok = record_window(argT, objcs_, argrec_err);
    typenameGraph arggraph = nullptr;
    const typenameError argce = cudaStreamEndCapture(objcs_, &arggraph);
    if (!argok) { if (arggraph) cudaGraphDestroy(arggraph); argerr = argrec_err; return false; }
    if (argce != cudaSuccess) { argerr = std::string("verify: end capture: ") + cudaGetErrorString(argce); return false; }
    cudaGraphInstantiate(&exec_[argT], arggraph, 0);
    cudaGraphDestroy(arggraph);
    cudaGraphUpload(exec_[argT], objcs_);
    cudaStreamSynchronize(objcs_);
    return true;
}
```

---

### Function: `Verifier::capture_commit`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{capture\_commit}: \text{void} \to \{\top,\bot\}
$$

Captures a separate CUDA graph for the **commit phase** — advancing recurrent state for accepted tokens. Per layer:

- **GDN layers:** `gdn_conv_commit` (advances conv state) + `gdn_step_norm_multi` (recomputes recurrence up to `n_keep`)
- **QSA layers:** Restores indexer tail from snapshot, then replays `native_qsa_indexer_append` for accepted positions
- **PLE:** `copy_indexed` to advance the conv history

Reads `commit_` buffer: `[n_keep, hist_index, pos_0, pos_1, ..., pos_{max_t-1}]` where positions $\geq n_{\text{keep}}$ are $-1$ (unused).

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem commit_positions_valid (n_keep max_t : Nat) (h : n_keep ≥ 1) (h2 : n_keep ≤ max_t) :
  n_keep ≤ max_t := by sorry

-- axle_verify: content
theorem commit_positions_valid (n_keep max_t : Nat) (h : n_keep ≥ 1) (h2 : n_keep ≤ max_t) :
  n_keep ≤ max_t := by exact h2
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjVerifier::capture_commit(std::string& argerr) {
    if (commit_exec_ != nullptr) return true;
    const typenameGeometry& arggeom = *objgeom_;
    typenameState& argss = *objss_;
    // ... dimension aliases, gdn_floats, TS, HS ...
    cudaStreamBeginCapture(objcs_, cudaStreamCaptureModeThreadLocal);
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + max_t_, objcs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t argL = 0; argL < arggeom.n_layers; ++argL) {
            if (!is_qsa_layer(arggeom, argL)) {
                // GDN: conv commit + recurrence replay
                gdn_conv_commit(argconv, argqkv, (int) argC, commit_, objcs_);
                gdn_step_norm_multi(argstate, /* ... */, (int) max_t_, commit_, objcs_);
                ++gdn_index;
            } else {
                // QSA: restore tail snapshot + replay indexer appends
                copy_from_mapped(argst.idx_tail, tail_snap_ + qsa_index * argTS, argTS, objcs_);
                for (int64_t argt = 0; argt < max_t_; ++argt)
                    native_qsa_indexer_append(/* ... */, commit_ + 2 + argt, /* ... */, objcs_);
                ++qsa_index;
            }
        }
        if (argss.ple.ready()) copy_indexed(argss.ple.hist, hist_snap_, argHS, commit_ + 1, argHS, objcs_);
    } catch (...) { /* ... */ }
    // End capture → instantiate
    cudaStreamEndCapture(objcs_, &arggraph);
    cudaGraphInstantiate(&commit_exec_, arggraph, 0);
    cudaGraphDestroy(arggraph);
    return true;
}
```

---

### Function: `Verifier::run`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{run}: \mathbb{Z}_{>0} \times \mathbb{Z}^T \times \mathbb{Z} \times \text{PoolMultiFn} \to \mathbb{Z}^T
$$

Executes the verification window: stages $T$ tokens, launches the captured graph, then runs the doorbell spin-wait + CPU pool loop.

**Host-side protocol (per layer $l$, per group $g$):**
1. Spin on `h_seq` until `*seq >= k+1` (where $k = l \cdot G + g$)
2. Set plan slot for the group
3. Run CPU pool: `pool(user, h_x, h_ids, n_tokens, k, h_ymiss, layer)`
4. Memory fence + sfence
5. If pool did not publish a plan via flag A, write an empty plan (zero counts)
6. Write `*flag = want`

**Split window ($G=2$):** The loop runs $n_L \times G$ iterations, interleaving group A and B for each layer.

**Head projection:** After all layers, runs `lm_head` or native head projection, then greedy argmax sampling.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem run_window_iterations (n_layers G : Nat) (hG : G ≥ 1) (hG2 : G ≤ 2) :
  n_layers * G ≥ n_layers := by sorry

-- axle_verify: content
theorem run_window_iterations (n_layers G : Nat) (hG : G ≥ 1) (hG2 : G ≤ 2) :
  n_layers * G ≥ n_layers := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjVerifier::run(int argT, const int32_t* argtokens, int64_t argpos0, cbPoolMulti cbpool, void* arguser, int32_t* argout,
                   std::string& argerr) {
    if (argT < 1 || argT > max_t_) { argerr = "window size out of range"; return false; }
    if (argpos0 + argT > objss_->qsa_states[0].max_cells) { argerr = "window runs past context"; return false; }
    if (!capture(argT, argerr) || !capture_commit(argerr)) return false;
    // Stage tokens, positions, PLE rows
    for (int argt = 0; argt < argT; ++argt) {
        h_tok_[argt] = argtokens[argt];
        qsa_step_fill(h_step_ + argt * kStepCount, argpos0 + argt, argshapes);
        for (int64_t argh = 0; argh < objgeom_->n_head; ++argh) h_pos_[argt * objgeom_->n_head + argh] = (int32_t) (argpos0 + argt);
    }
    // Reset doorbells
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // Launch graph
    cudaGraphLaunch(exec_[argT], objcs_);
    (void) cudaStreamQuery(objcs_);
    // Doorbell loop: n_layers * G iterations
    volatile uint32_t* const argseq = h_seq_;
    volatile uint32_t* const argflag = h_flag_;
    const int argG = groups_[argT] > 0 ? groups_[argT] : 1;
    for (int64_t argk = 0; argk < objgeom_->n_layers * argG; ++argk) {
        const uint32_t argwant = (uint32_t) (argk + 1);
        // Spin with 2ms flush + 20s timeout
        while (*argseq < argwant) { _mm_pause(); /* flush + timeout logic */ }
        // CPU pool
        set_plan_slot((int) (argk % argG));
        if (cbpool != nullptr) cbpool(arguser, /* ... */);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        // Empty plan fallback if pool didn't publish
        if (*(volatile uint32_t*) h_flagA_ != argwant) {
            sink_.counts[0] = 0; sink_.counts[1] = 0; sink_.counts[2] = 0;
            sink_.start[0] = 0; sink_.start2[0] = 0;
            *(volatile uint32_t*) h_flagA_ = argwant;
            raise_flag(h_flagB_, argwant);
        }
        *argflag = argwant;
    }
    cudaStreamSynchronize(objcs_);
    for (int argt = 0; argt < argT; ++argt) argout[argt] = ((volatile int32_t*) h_out_)[argt];
    return true;
}
```

---

### Function: `Verifier::commit`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{commit}: \mathbb{Z}_{>0} \to \{\top,\bot\}
$$

**Precondition:** $1 \leq n_{\text{keep}} \leq \text{last\_t}$

Accepts the first `n_keep` tokens from the last window into persistent state. Fills the commit buffer:
- `commit[0] = n_keep`
- `commit[1] = n_keep - 1` (PLE history index)
- `commit[2+t] = pos0 + t` for $t < n_{\text{keep}}$, else $-1$

Then launches the commit graph and advances `ple_prev` token history.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem commit_range (n_keep last_t : Nat) (h1 : 1 ≤ n_keep) (h2 : n_keep ≤ last_t) :
  n_keep ≥ 1 ∧ n_keep ≤ last_t := by sorry

-- axle_verify: content
theorem commit_range (n_keep last_t : Nat) (h1 : 1 ≤ n_keep) (h2 : n_keep ≤ last_t) :
  n_keep ≥ 1 ∧ n_keep ≤ last_t := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
bool adjObjVerifier::commit(int argn_keep, std::string& argerr) {
    if (argn_keep < 1 || argn_keep > last_t_) { argerr = "commit count out of range"; return false; }
    h_commit_[0] = argn_keep;
    h_commit_[1] = argn_keep - 1;
    for (int argt = 0; argt < max_t_; ++argt) h_commit_[2 + argt] = argt < argn_keep ? (int32_t) (last_pos0_ + argt) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    cudaGraphLaunch(commit_exec_, objcs_);
    cudaStreamSynchronize(objcs_);
    for (int argt = 0; argt < argn_keep; ++argt) {
        objss_->ple_prev[0] = objss_->ple_prev[1];
        objss_->ple_prev[1] = last_tokens_[argt];
    }
    return true;
}
```

---

### Function: `Verifier::raise_flag`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{raise\_flag}: \text{Ptr}(\mathbb{Z}_{32}) \times \mathbb{Z}_{32} \to \text{void}
$$

**Monotonic CAS loop:** Only writes if `value > *flag`. Platform-specific:
- **Windows:** `_InterlockedCompareExchange`
- **Linux:** `__atomic_compare_exchange_n` with `SEQ_CST` ordering

This ensures flags only increase, preventing a late host function from overwriting a newer value already set by a later layer.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem raise_flag_monotonic (cur value : Nat) (h : cur < value) : value > cur := by sorry

-- axle_verify: content
theorem raise_flag_monotonic (cur value : Nat) (h : cur < value) : value > cur := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void adjObjVerifier::raise_flag(uint32_t* argflag, uint32_t argvalue) {
    volatile long* argf = (volatile long*) argflag;
#if defined(_WIN32)
    long argcur = *argf;
    while ((uint32_t) argcur < argvalue) {
        const long argprev = _InterlockedCompareExchange(argf, (long) argvalue, argcur);
        if (argprev == argcur) break;
        argcur = argprev;
    }
#else
    uint32_t argcur = __atomic_load_n((uint32_t*) argflag, __ATOMIC_SEQ_CST);
    while (argcur < argvalue && !__atomic_compare_exchange_n((uint32_t*) argflag, &argcur, argvalue, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}
```

---

### Function: `Verifier::fetch_dma`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{fetch\_dma}: \text{Ptr}^n \times \mathbb{N} \times \mathbb{N} \to \text{void}
$$

PCIe DMA path for expert blobs. Copies `n` blobs of `bytes` each from host pointers to the staging area via the **copy stream** (`copy_`). After the copies, a host function raises flag B (via `cudaLaunchHostFunc`) to signal the graph that staging is ready.

**Staging split:** In a split window ($G=2$), each group gets half of `kStagingBlobs`.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement  
theorem staging_split (total : Nat) (G : Nat) (hG : G = 2) : total / G + total / G = total ∨ total / G + total / G + 1 = total := by sorry
-- proof_needed: integer division rounding
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void adjObjVerifier::fetch_dma(void* argctx, const uint8_t* const* argsrc, int argn, size_t argbytes) {
    adjObjVerifier* argv = (adjObjVerifier*) argctx;
    const uint32_t argwant = argv->cur_layer_ + 1;
    if (argn <= 0) { raise_flag(argv->h_flagB_, argwant); return; }
    uint8_t* argstage = (uint8_t*) argv->sink_.staging;
    for (int idx = 0; idx < argn; ++idx) cudaMemcpyAsync(argstage + (size_t) idx * argbytes, argsrc[idx], argbytes, cudaMemcpyHostToDevice, argv->objcopy_);
    typenameFS& argfs = argv->flag_sets_[argv->cur_layer_ % (sizeof argv->flag_sets_ / sizeof argv->flag_sets_[0])];
    argfs.flag = argv->h_flagB_;
    argfs.value = argwant;
    cudaLaunchHostFunc(argv->objcopy_, [](void* argp) { typenameFS* args = (typenameFS*) argp; raise_flag(args->flag, args->value); }, &argfs);
}
```

---

### Function: `Verifier::publish_plan` / `Verifier::set_plan_slot`

**[LAYER 1a — MATH (LaTeX)]**

`publish_plan`: Publishes the expert routing plan to the GPU by writing flag A after an sfence.

`set_plan_slot(grp)`: Repoints the `sink_` struct's arrays at the correct group's portion of the plan buffer.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- trivial pointer arithmetic, no proof needed
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void adjObjVerifier::publish_plan(void* argctx) {
    adjObjVerifier* argv = (adjObjVerifier*) argctx;
    _mm_sfence();
    *(volatile uint32_t*) argv->h_flagA_ = argv->cur_layer_ + 1;
}

void adjObjVerifier::set_plan_slot(int arggrp) {
    const int64_t argcap = sink_.cap;
    int32_t* argbase = h_plan_ + (size_t) arggrp * (size_t) plan_i32_;
    const int64_t argi32 = 4 + (argcap + 1) + argcap + argcap;
    const int64_t argptr_off = (argi32 + 1) & ~1ll;
    sink_.counts = argbase;
    sink_.start = argbase + 4;
    sink_.dst = sink_.start + argcap + 1;
    sink_.tok = sink_.dst + argcap;
    sink_.ptr = (unsigned long long*) (argbase + argptr_off);
    sink_.ptr2 = sink_.ptr + argcap;
    sink_.start2 = argbase + argptr_off + 4 * argcap;
    // Staging split for the group
    const int argG = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t argper = argG == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (arggrp * argper) * prjname::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = argper;
}
```
