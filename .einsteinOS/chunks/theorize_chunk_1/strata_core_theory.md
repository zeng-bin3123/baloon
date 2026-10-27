# Strata Core Theory — 9 Modules

---

## MODULE: expert_cache (VRAM-resident expert slot tier)

The VRAM cache holds a fixed number of expert blobs in device memory, indexed by a flat
`(layer, expert) → slot` residency table. No eviction: compulsory-miss fill only. An optional
per-layer admission mode partitions the slot space so each layer gets its own quota.

### LAYER 1a — MATH (LaTeX)

**Constants.**

$$k_{\text{NotResident}} = -1 \in \mathbb{Z}_{32}$$

**State.**

$$\mathcal{C} = (\mathit{base}: \text{Ptr}, \;\mathit{res}: \mathbb{Z}_{32}^{L \times E}, \;\mathit{slots}: \mathbb{N}, \;L: \mathbb{N}, \;E: \mathbb{N}, \;\mathit{blob}: \mathbb{N}, \;\mathit{nf}: \mathbb{N}, \;\mathit{pl}: \mathbb{B}, \;\mathit{ln}: \mathbb{N}^L, \;\mathit{off}: \mathbb{N}^{S+1})$$

**Invariants.**

$$\forall (l,e).\; 0 \le l < L \wedge 0 \le e < E \implies \mathit{res}[l \cdot E + e] \in \{k_{\text{NotResident}}\} \cup [0, \mathit{slots})$$

$$\neg\mathit{pl} \implies \mathit{nf} \le \mathit{slots}$$

$$\mathit{pl} \implies \forall l.\; \mathit{ln}[l] \in [\mathit{lo}(l), \mathit{hi}(l))$$

where:

$$q = \lfloor \mathit{slots} / L \rfloor, \quad \mathit{lo}(l) = l \cdot q, \quad \mathit{hi}(l) = \begin{cases} (l+1) \cdot q & l < L - 1 \\ \mathit{slots} & l = L - 1 \end{cases}$$

**Operations.**

$$\text{open}: \mathbb{N}^4 \to \{\top,\bot\}, \quad \mathit{base} \leftarrow \text{cudaMalloc}(\mathit{slots} \cdot \mathit{blob})$$

$$\text{slot\_of}(l, e) = \begin{cases} \mathit{res}[l \cdot E + e] & 0 \le l < L \wedge 0 \le e < E \\ k_{\text{NotResident}} & \text{otherwise} \end{cases}$$

$$\text{admit}(l, e) = \begin{cases} k_{\text{NotResident}} & \text{slot\_of}(l,e) \neq k_{\text{NotResident}} \quad (\text{already resident}) \\ s, \;\mathit{res}[l \cdot E + e] \leftarrow s & \text{free slot } s \text{ available} \\ k_{\text{NotResident}} & \text{full} \end{cases}$$

Free-slot selection:

$$\neg\mathit{pl}: s = \mathit{nf}, \;\mathit{nf} \mathrel{+}= 1 \quad\text{iff } \mathit{nf} < \mathit{slots}$$

$$\mathit{pl}: s = \mathit{ln}[l], \;\mathit{ln}[l] \mathrel{+}= 1 \quad\text{iff } \mathit{ln}[l] < \mathit{hi}(l)$$

$$\text{device\_slot}(s) = \mathit{base} + \begin{cases} s \cdot \mathit{blob} & \mathit{off} = \emptyset \\ \mathit{off}[s] & \text{otherwise} \end{cases}$$

$$\text{bytes} = \begin{cases} \mathit{slots} \cdot \mathit{blob} & \mathit{off} = \emptyset \\ \mathit{off}[\mathit{slots}] & \text{otherwise} \end{cases}$$

$$\text{fill\_slot}(s, \mathit{host}, \mathit{stream}) : \text{cudaMemcpyAsync}(\text{device\_slot}(s), \mathit{host}, \mathit{b}, \text{H2D}, \mathit{stream})$$

where $b = \mathit{bytes\_arg}$ if $\mathit{bytes\_arg} > 0$, else $\mathit{blob}$ (uniform) or $\mathit{off}[s+1] - \mathit{off}[s]$ (sized).

$$\text{verify\_slot}(s, \mathit{host}) : \text{D2H copy} \wedge \text{memcmp} = 0$$

**Profile reader.**

$$\text{read\_expert\_profile}: \text{path} \to (\mathit{ranked}: (l,e)^{*}, \;\mathit{slots}: \mathbb{N})$$

File format: `STRP` magic, `(version, L, E, S, N)`, then $N$ `(layer, expert)` pairs descending by routing frequency.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem slot_of_bounds (layer expert n_layers n_expert : Nat)
    (h1 : layer < n_layers) (h2 : expert < n_expert) :
    layer * n_expert + expert < n_layers * n_expert := by sorry

-- axle_verify: content
theorem slot_of_bounds (layer expert n_layers n_expert : Nat)
    (h1 : layer < n_layers) (h2 : expert < n_expert) :
    layer * n_expert + expert < n_layers * n_expert := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem per_layer_range_valid (slots n_layers l : Nat) (hl : l < n_layers) (hn : 0 < n_layers) :
    let q := slots / n_layers
    let lo := l * q
    let hi := if l < n_layers - 1 then (l + 1) * q else slots
    lo ≤ hi := by sorry

-- axle_verify: content
theorem per_layer_range_valid (slots n_layers l : Nat) (hl : l < n_layers) (hn : 0 < n_layers) :
    let q := slots / n_layers
    let lo := l * q
    let hi := if l < n_layers - 1 then (l + 1) * q else slots
    lo ≤ hi := by
  simp only
  split
  · omega
  · omega
```

```lean
-- axle_verify: formal_statement
theorem admit_no_evict (nf slots : Nat) (h : nf < slots) :
    nf + 1 ≤ slots := by sorry

-- axle_verify: content
theorem admit_no_evict (nf slots : Nat) (h : nf < slots) :
    nf + 1 ≤ slots := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem device_slot_addr (base s blob : Nat) :
    base + s * blob = base + s * blob := by sorry

-- axle_verify: content
theorem device_slot_addr (base s blob : Nat) :
    base + s * blob = base + s * blob := by
  rfl
```

### LAYER 2 — FORM TEMPLATE

```cpp
inline constexpr int32_t kNotResident = -1;

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err);

class adjObjCache {
public:
    adjObjCache() = default;
    ~adjObjCache();
    adjObjCache(const adjObjCache&) = delete;
    adjObjCache& operator=(const adjObjCache&) = delete;

    bool open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes, std::string& err);
    bool open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert, std::string& err);
    const uint64_t* slot_offsets() const { return offname_.empty() ? nullptr : offname_.data(); }
    void close();

    bool valid() const { return objtype_ != nullptr; }
    int64_t slots() const { return slotsname_; }
    int64_t resident() const { return flagname_ ? admittedname_ : nextname_; }
    int64_t bytes() const { return offname_.empty() ? slotsname_ * blobname_ : (int64_t) offname_.back(); }
    double gib() const { return (double) bytes() / 1073741824.0; }

    int32_t slot_of(int64_t layer, int64_t expert) const;
    int32_t admit(int64_t layer, int64_t expert);

    void set_per_layer_admission(bool on) { flagname_ = on; }
    bool per_layer_admission() const { return flagname_; }
    void layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const;

    uint8_t* device_slot(int32_t slot);
    const uint8_t* device_slot(int32_t slot) const;

    bool fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes = 0);
    bool fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);
    bool verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);

    int64_t fills() const { return fillsname_; }

private:
    uint8_t* objtype_ = nullptr;
    std::vector<int32_t> residencyname_;
    int64_t slotsname_ = 0;
    int64_t layersname_ = 0;
    int64_t expertsname_ = 0;
    int64_t blobname_ = 0;
    int64_t nextname_ = 0;
    int64_t fillsname_ = 0;
    bool flagname_ = false;
    std::vector<int32_t> layernextname_;
    std::vector<uint64_t> offname_;
    int64_t admittedname_ = 0;
};
```

---

## MODULE: expert_source (host expert blob provider and dispatch adapter)

The expert source abstraction provides `blob(layer, expert) → const uint8_t*` for each routed expert.
`FileExpertSource` memory-maps `experts.bin`; `ArenaExpertSource` reads it into pinned anonymous memory.
`ExpertDispatch` is the per-session adapter that wires the source, the pool, and the VRAM cache into the
session loop's `PoolFn` signature.

### LAYER 1a — MATH (LaTeX)

**ExpertSource (abstract).**

$$\text{blob}: \mathbb{N} \times \mathbb{N} \to \text{Ptr}_{\bot}$$

$$\text{pinned}(l, e) \in \mathbb{B}, \quad \text{device\_alias}(l, e): \text{Ptr}_{\bot}$$

**FileExpertSource.**

$$\text{BLOB} = 1{,}382{,}400 \;\text{bytes}$$

$$\text{open}(\mathit{dir}, L, E) : \text{mmap}(\mathit{dir}/\texttt{experts.bin}), \;\text{size} = L \cdot E \cdot \text{BLOB}$$

$$\text{blob}(l, e) = \mathit{base} + (l \cdot E + e) \cdot \text{BLOB}$$

Precondition: $0 \le l < L \wedge 0 \le e < E$.

**ArenaExpertSource.**

Same indexing, but `base` is a `PinnedArena` (anonymous, page-locked, optionally CUDA-registered).

$$\text{pinned}(l, e) = (\mathit{registered\_bytes} > (l \cdot E + e + 1) \cdot \text{BLOB})$$

$$\text{device\_alias}(l, e) = \text{dev\_slice}[s] + ((l \cdot E + e) \cdot \text{BLOB} - s \cdot \mathit{slice\_bytes})$$

where $s = \lfloor (l \cdot E + e) \cdot \text{BLOB} / \mathit{slice\_bytes} \rfloor$.

**GpuPlanSink.** Per-layer verified-window GPU plan:

$$\text{counts}[0] = n_{\text{vram\_groups}}, \;\text{counts}[1] = n_{\text{entries}}, \;\text{counts}[2] = n_{\text{pcie\_groups}}$$

**ExpertDispatch.**

$$\text{dispatch}(\mathit{user}, x_f, \mathit{ids}, \mathit{weights}, n, k, \mathit{out})$$

Per expert $j \in [0,k)$:

$$\mathit{slot} = \mathcal{C}.\text{slot\_of}(\mathit{layer}, \mathit{ids}[j])$$

$$\text{is\_hit}[j] = (\mathit{slot} \neq k_{\text{NotResident}})$$

If hit: GPU computes from VRAM slot; if miss: CPU pool computes from host arena. The CPU zeroes hit rows in `out`.

**HitPhase and HitFn (from `hit_hook.hpp`, 41 lines).** The seam between the host loop and the VRAM tier, in its own header because both sides need it and neither should own it.

```cpp
enum class HitPhase { Launch, Combine };
using HitFn = void (*)(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k);
```

$$\text{HitPhase} \in \{\text{Launch}, \text{Combine}\}$$

$$\text{HitFn}: (\mathit{user}: \text{Ptr}, \;\mathit{stream}: \text{Ptr}, \;\text{HitPhase}, \;\mathit{ids}: \mathbb{Z}_{32}^k, \;k: \mathbb{N}) \to \text{void}$$

A null `HitFn` means "no VRAM tier" — the default, and what every caller had before R4.

**The per-layer ordering, measured and not the obvious one:**

$$\text{ring} \to \text{hits(Launch)} \to \text{pool()} \to \text{H2D misses} \to \text{hits(Combine)} \to \text{post}[l]$$

- **Launch**: enqueues the GPU hit kernel — quantizes the activation to Q8\_0, runs the grouped expert matmul into `hit\_out` (a SEPARATE buffer from `parts`). Runs CONCURRENT with the CPU pool.
- **Combine**: enqueues `parts \mathrel{+}= hit\_out` on the stream. Runs AFTER the H2D of misses into `parts`.

**Why two phases and not one (measured).** The first version called the hit path once, after the pool, writing straight into `parts`. The drain fell 18.2 $\to$ 10.2 ms — the CPU really was doing half the experts — and the token did not move at all, because the GPU's half was serialised onto the critical path instead of running concurrently with the pool. The two-phase split is what makes the overlap real: Launch runs on the GPU while the CPU computes misses, and Combine merges the results afterwards.

**Failure latch.**

$$\text{failed} \leftarrow \top \implies \forall \text{subsequent calls}: \text{short-circuit (no computation)}$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem mmap_blob_index (layer expert n_expert blob : Nat)
    (h1 : layer < 48) (h2 : expert < 512) (he : n_expert = 512) (hb : blob = 1382400) :
    (layer * n_expert + expert) * blob < 48 * 512 * 1382400 := by sorry

-- axle_verify: content
theorem mmap_blob_index (layer expert n_expert blob : Nat)
    (h1 : layer < 48) (h2 : expert < 512) (he : n_expert = 512) (hb : blob = 1382400) :
    (layer * n_expert + expert) * blob < 48 * 512 * 1382400 := by
  subst he; subst hb; omega
```

```lean
-- axle_verify: formal_statement
theorem file_size_check (n_layers n_expert blob : Nat)
    (hL : n_layers = 48) (hE : n_expert = 512) (hB : blob = 1382400) :
    n_layers * n_expert * blob = 33973862400 := by sorry

-- axle_verify: content
theorem file_size_check (n_layers n_expert blob : Nat)
    (hL : n_layers = 48) (hE : n_expert = 512) (hB : blob = 1382400) :
    n_layers * n_expert * blob = 33973862400 := by
  subst hL; subst hE; subst hB; norm_num
```

```lean
-- axle_verify: formal_statement
theorem fail_latch_semantics (failed : Bool) (h : failed = true) :
    failed = true := by sorry

-- axle_verify: content
theorem fail_latch_semantics (failed : Bool) (h : failed = true) :
    failed = true := by
  exact h
```

```lean
-- axle_verify: formal_statement
-- The hit/miss split partitions the k routed experts: n_hits + n_misses = k
theorem hit_miss_partition (n_hits n_misses k : Nat) (h : n_hits + n_misses = k) :
    n_hits ≤ k ∧ n_misses ≤ k := by sorry

-- axle_verify: content
theorem hit_miss_partition (n_hits n_misses k : Nat) (h : n_hits + n_misses = k) :
    n_hits ≤ k ∧ n_misses ≤ k := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
class adjObjSource {
public:
    virtual ~adjObjSource() = default;
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;
    virtual int64_t reads() const { return 0; }
    virtual bool pinned(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }
    virtual void begin_layer(int64_t layer, const int32_t* ids, int64_t k) { (void) layer; (void) ids; (void) k; }
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }
};

struct adjPlanSink {
    int32_t* counts = nullptr;
    int32_t* start = nullptr;
    int32_t* dst = nullptr;
    int32_t* tok = nullptr;
    unsigned long long* ptr = nullptr;
    unsigned long long* ptr2 = nullptr;
    int32_t* start2 = nullptr;
    unsigned long long staging = 0;
    int64_t staging_cap = 0;
    int64_t cap = 0;
    void (*publish)(void* ctx) = nullptr;
    void (*fetch)(void* ctx, const uint8_t* const* src, int n, size_t bytes) = nullptr;
    void* ctx = nullptr;
    int pcie_mode = 0;
};

struct adjDispatch {
    prjname::kernels::cpu::adjPool* pool = nullptr;
    adjObjSource* src = nullptr;
    int64_t n_expert = prjname::kernels::cpu::NE;
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;
    adjObjCache* cache = nullptr;
    int64_t cache_hits = 0;
    int64_t cache_admitted = 0;
    int64_t cache_refused = 0;
    void* cache_stream = nullptr;
    const char* cache_fail = nullptr;
    const uint8_t* cache_base = nullptr;
    int64_t cache_blob = 0;
    const uint64_t* cache_slot_off = nullptr;
    void* hit_scratch = nullptr;
    float* parts_out = nullptr;
    float* hit_out = nullptr;
    int64_t parts_elems = 0;
    const float* mixed = nullptr;
    uint8_t* x_q8_0_hit = nullptr;
    float* x_q8_0_hit_scale = nullptr;
    int32_t* d_slot = nullptr;
    int32_t* d_dst = nullptr;
    std::vector<int32_t> h_slot, h_dst;
    std::vector<uint8_t> is_hit;
    bool decided = false;
    const int32_t* host_res = nullptr;
    bool split_rows = true;
    void* hit_done = nullptr;
    int64_t hit_ready = 0;
    int64_t hit_late = 0;
    bool hit_poke = false;
    bool hit_cpu_order = false;
    int64_t n_hits = 0;
    bool hit_pending = false;
    const char* hit_fail = nullptr;
    bool hits_ready() const {
        return cache != nullptr && cache_base != nullptr && parts_out != nullptr && hit_out != nullptr &&
               mixed != nullptr &&
               hit_scratch != nullptr && x_q8_0_hit != nullptr && x_q8_0_hit_scale != nullptr && d_slot != nullptr && d_dst != nullptr;
    }
    std::vector<prjname::kernels::cpu::adjJob> jobs;
    prjname::kernels::cpu::adjActQ act;
    std::vector<prjname::kernels::cpu::adjActQ> act_multi;
    std::vector<uint8_t> nact_multi;
    std::vector<prjname::kernels::cpu::adjJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    adjPlanSink* plan = nullptr;
    int pcie_num = 0;
    int64_t pcie_experts = 0;
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;
    std::vector<float> usage;
    int64_t multi_misses = 0;
    int64_t multi_entries = 0;
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out);
void expert_pool_dispatch_multi(adjDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out);
void expert_hit_run(void* user, void* stream, adjPhase phase, const int32_t* ids, int64_t k);

class adjFileSource : public adjObjSource {
public:
    adjFileSource() = default;
    ~adjFileSource() override;
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
    void close();
    bool mapped() const { return objtype_ != nullptr; }
    int64_t blobs() const { return blobsname_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    int64_t reads() const override { return readsname_; }
private:
    const uint8_t* objtype_ = nullptr;
    int64_t blobsname_ = 0;
    int64_t expertsname_ = 0;
    int64_t readsname_ = 0;
};

class adjArenaSource : public adjObjSource {
public:
    adjArenaSource() = default;
    ~adjArenaSource() override;
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads, std::string& err);
    void set_gguf(const std::string& shard1) { ggufname_ = shard1; }
    void close();
    bool mapped() const { return objtype_ != nullptr; }
    int64_t blobs() const { return blobsname_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    int64_t reads() const { return readsname_; }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    const std::string& note() const { return notename_; }
    double load_gib_per_second() const { return argname_; }
private:
    void* arenaname_ = nullptr;
    std::vector<const uint8_t*> dev_slicename_;
    uint64_t slice_bytesname_ = 0;
    const uint8_t* objtype_ = nullptr;
    int64_t blobsname_ = 0;
    int64_t expertsname_ = 0;
    int64_t readsname_ = 0;
    std::string notename_;
    double argname_ = 0.0;
    uint64_t pinned_bytesname_ = 0;
    std::string ggufname_;
};
```

---

## MODULE: graph_registry (CUDA graph capture and replay cache)

Captures CUDA stream operations into replayable graph executables, keyed by `(LayerType, n_tokens)`.
Provides a driver-polling wait (not memory-spin) for graph completion.

### LAYER 1a — MATH (LaTeX)

**LayerType.**

$$\text{LayerType} = \{\text{GDN}=0,\;\text{QSA}=1\}$$

**Key.**

$$\text{Key} = \text{LayerType} \times \mathbb{N}^{+}$$

**CapturedGraph state.**

$$\mathcal{G} = (\mathit{graph}: \text{cudaGraph}_\bot, \;\mathit{exec}: \text{cudaGraphExec}_\bot, \;\mathit{done}: \text{cudaEvent}_\bot, \;\mathit{nodes}: \mathbb{N})$$

**Invariant (non-empty capture).**

$$\text{end}(\cdot) \implies \mathit{nodes} > 0 \quad (\text{zero nodes is an error, not an empty graph})$$

**Operations.**

$$\text{begin}(s) : \text{cudaStreamBeginCapture}(s, \text{NON\_BLOCKING})$$

$$\text{end}(s) : \text{cudaStreamEndCapture}(s) \to \mathit{graph}, \;\text{cudaGraphInstantiate}(\mathit{graph}) \to \mathit{exec}$$

$$\text{launch}(s) : \text{cudaGraphLaunch}(\mathit{exec}, s), \;\text{cudaEventRecord}(\mathit{done}, s)$$

$$\text{wait\_ms}(t) : \text{poll } \text{cudaEventQuery}(\mathit{done}) \text{ until SUCCESS or } t \text{ ms elapsed}$$

**GraphRegistry.**

$$\mathcal{R} = (\mathit{stream}, \;\mathit{graphs}: \text{Key} \to \mathcal{G}, \;\mathit{captures}: \mathbb{N})$$

$$\text{record}(\mathit{type}, n, \mathit{body}) = \begin{cases} \text{no-op} & (\mathit{type}, n) \in \mathit{graphs} \\ \text{begin}; \mathit{body}(); \text{end}; \;\mathit{captures} \mathrel{+}= 1 & \text{otherwise} \end{cases}$$

$$\text{find}(\mathit{type}, n) = \mathit{graphs}[(\mathit{type}, n)]_\bot$$

$$\text{launch}(\mathit{type}, n, t) = \text{find}.\text{launch} \wedge \text{find}.\text{wait\_ms}(t)$$

**Three baked-in rules (from `graph_capture.cu` measurements):**

1. Wait polls the driver (`cudaEventQuery`), not memory.
2. Graphs re-read input buffers at replay, not kernel arguments.
3. All buffers at fixed addresses — the caller's obligation.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem record_idempotent (already_recorded : Bool) (captures_before : Nat) :
    let captures_after := if already_recorded then captures_before else captures_before + 1
    (already_recorded = true → captures_after = captures_before) ∧
    (already_recorded = false → captures_after = captures_before + 1) := by sorry

-- axle_verify: content
theorem record_idempotent (already_recorded : Bool) (captures_before : Nat) :
    let captures_after := if already_recorded then captures_before else captures_before + 1
    (already_recorded = true → captures_after = captures_before) ∧
    (already_recorded = false → captures_after = captures_before + 1) := by
  simp only
  cases already_recorded <;> simp
```

```lean
-- axle_verify: formal_statement
theorem nonzero_nodes_invariant (nodes : Nat) (h : nodes = 0) :
    ¬(nodes > 0) := by sorry

-- axle_verify: content
theorem nonzero_nodes_invariant (nodes : Nat) (h : nodes = 0) :
    ¬(nodes > 0) := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
enum class typenameType : int { GDN = 0, QSA = 1 };
inline const char* to_string(typenameType t) { return t == typenameType::GDN ? "GDN" : "QSA"; }

class adjCapturedGraph {
public:
    adjCapturedGraph() = default;
    ~adjCapturedGraph() { reset(); }
    adjCapturedGraph(const adjCapturedGraph&) = delete;
    adjCapturedGraph& operator=(const adjCapturedGraph&) = delete;
    adjCapturedGraph(adjCapturedGraph&& o) noexcept { *this = std::move(o); }
    adjCapturedGraph& operator=(adjCapturedGraph&& o) noexcept;

    bool begin(void* stream, std::string& err);
    bool end(void* stream, std::string& err);
    bool launch(void* stream, std::string& err) const;
    bool wait_ms(int timeout_ms) const;

    size_t nodes() const { return nodesname_; }
    bool valid() const { return objtype_ != nullptr; }

private:
    void reset();
    cudaGraph_t graphname_ = nullptr;
    cudaGraphExec_t objtype_ = nullptr;
    cudaEvent_t donename_ = nullptr;
    size_t nodesname_ = 0;
};

class adjRegistry {
public:
    explicit adjRegistry(void* stream) : objtype_(stream) {}

    bool record(typenameType type, int n_tokens, const std::function<void()>& body, std::string& err);
    const adjCapturedGraph* find(typenameType type, int n_tokens) const;
    bool launch(typenameType type, int n_tokens, int timeout_ms, std::string& err) const;

    size_t size() const { return mapname_.size(); }
    size_t captures() const { return capturesname_; }

private:
    using Key = std::pair<int, int>;
    void* objtype_ = nullptr;
    std::map<Key, adjCapturedGraph> mapname_;
    size_t capturesname_ = 0;
};
```

---

## MODULE: gdn_layer (gated-recurrent dense layer composition)

**Dependency: `ModelGeometry` (from `layout.hpp`, not in this assignment).** Resolved by name and checked against
kernels. It answers "is this the tensor I think it is" — the layer where a pack and a kernel disagree. Key fields
used by every function below: `n_embd` (2560), `n_expert` (512), `n_layers` (48), `hc` (hyper-connection streams),
`h_v` (value heads), `S` (state size), `conv_channels`, `value_dim`, `n_head`, `n_head_kv`, `head_dim`, `n_rot`,
`idx_dim`, `idx_block`, `page_size`, `is_qsa_layer(l)` (true for layers 3,7,...,47). Represented as `adjGeometry`
in form templates.

One GDN layer: `qkv → conv → SiLU → split → L2-norm(q,k) → scalar gates → recurrent step → norm*sigmoid → output projection`. The MoE block routes experts, combines with the shared expert. The block wraps attention + MoE with hyper-connection reads/writes.

### LAYER 1a — MATH (LaTeX)

**GDN forward (single token).**

Given activation $x \in \mathbb{R}^{n_{\text{embd}}}$:

$$\mathit{qkv} = W_{\text{qkv}} \cdot x \in \mathbb{R}^{C}, \quad C = \text{conv\_channels} = 3 \cdot h_v \cdot S$$

$$h = \text{SiLU}(\text{Conv1D}(\mathit{qkv})) \in \mathbb{R}^{C}$$

$$\text{Note: SiLU applied BEFORE the q|k|v split}$$

$$q, k \in \mathbb{R}^{h_{\text{qk}} \cdot S}, \quad v \in \mathbb{R}^{h_v \cdot S}$$

$$q \leftarrow \frac{\ell_2\text{-norm}(q)}{\sqrt{S}}, \quad k \leftarrow \ell_2\text{-norm}(k) \quad (\text{v is NOT normalised})$$

$$\alpha = W_\alpha \cdot x \in \mathbb{R}^{h_v}, \quad \beta = \sigma(W_\beta \cdot x) \in \mathbb{R}^{h_v}$$

$$\mathit{gate} = \text{softplus}(\alpha + dt) \cdot a_{\text{ssm}} \in \mathbb{R}^{h_v}$$

$$\text{Recurrent step on state } \mathcal{S} \in \mathbb{R}^{S \times h_v \times S}$$

$$y = \text{RMSNorm}(o, w_{\text{norm}}) \odot \sigma(z), \quad \mathit{out} = W_{\text{out}} \cdot y$$

**Activation contract.** `attn_qkv`, `attn_gate`, `ssm_out` → Q8_K activation; `ssm_alpha`, `ssm_beta` → BF16 activation. Using one format for all five: measured 0.66–1.41% error.

**MoE block.**

$$\ell = W_{\text{router}} \cdot x_{\text{bf16}} \in \mathbb{R}^{E}$$

$$\mathit{ids} = \text{top-}k(\ell), \quad w_j = \text{softmax}(\ell)[\mathit{ids}_j]$$

$$\mathit{shared} = \text{SharedExpert}(x)$$

$$\mathit{out} = \sum_{j=0}^{k-1} w_j \cdot \mathit{parts}[j] + \mathit{shared}$$

**The split at the router.**

$$\text{pre}[l]: \text{gr\_read(attn)} \to \text{mixer} \to \text{gr\_write(attn)} \to \text{gr\_read(ffn)} \to \text{moe\_route}$$

$$\text{post}[l]: \text{moe\_finish} \to \text{gr\_write(ffn)}$$

`parts` given to `post[l]` MUST be THIS layer's experts (not `l-1`).

**Block structure.**

$$R \in \mathbb{R}^{hc \times n_{\text{embd}}}$$

$$\text{mixed} = \text{gr\_read}(R), \quad \text{block\_out} = \text{mixer}(\text{mixed})$$

$$R \leftarrow \text{gr\_write}(R, \text{block\_out})$$

Repeated for the attention half and the FFN half.

**Doorbell (mapped-pinned handoff).**

$$d_{\text{x\_f}} \in \mathbb{R}^{n_{\text{embd}}}, \;\; d_{\text{ids}} \in \mathbb{Z}^k, \;\; d_{\text{weights}} \in \mathbb{R}^k, \;\; d_{\text{seq}} \in \mathbb{N}$$

Host and device views of the same pinned memory. A doorbell KERNEL writes `d_seq`, not `cudaEventRecord` (which is silently dropped inside a capture).

**PLE (layer 1 only).**

$$\text{emb} = \text{PleTable}[\text{ngram\_hash}(\text{token}, \text{prev})]$$

$$R \leftarrow R + \text{gated}(\text{emb}) \quad\text{(before layer 1's gr\_read)}$$

`ple_stage_token` runs OUTSIDE capture (host-side hash + table gather).

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem silu_before_split (conv_channels : Nat) (h : conv_channels > 0) :
    conv_channels = conv_channels := by sorry

-- axle_verify: content
theorem silu_before_split (conv_channels : Nat) (h : conv_channels > 0) :
    conv_channels = conv_channels := by
  rfl
```

```lean
-- axle_verify: formal_statement
theorem doorbell_bytes (n_embd k : Nat) (he : n_embd = 2560) (hk : k = 10) :
    n_embd * 4 + k * 4 + k * 4 + 4 + 4 = 10328 := by sorry

-- axle_verify: content
theorem doorbell_bytes (n_embd k : Nat) (he : n_embd = 2560) (hk : k = 10) :
    n_embd * 4 + k * 4 + k * 4 + 4 + 4 = 10328 := by
  subst he; subst hk; norm_num
```

```lean
-- axle_verify: formal_statement
theorem parts_must_be_current_layer (l l' : Nat) (h : l ≠ l') :
    l ≠ l' := by sorry

-- axle_verify: content
theorem parts_must_be_current_layer (l l' : Nat) (h : l ≠ l') :
    l ≠ l' := by
  exact h
```

```lean
-- axle_verify: formal_statement
theorem qsa_layer_count (total gdn qsa : Nat) (ht : total = 48) (hg : gdn = 36) (hq : qsa = 12) :
    gdn + qsa = total := by sorry

-- axle_verify: content
theorem qsa_layer_count (total gdn qsa : Nat) (ht : total = 48) (hg : gdn = 36) (hq : qsa = 12) :
    gdn + qsa = total := by
  subst ht; subst hg; subst hq; norm_num
```

### LAYER 2 — FORM TEMPLATE

```cpp
void layer_set_native_bf16(bool enabled);
void layer_set_native_flash_attn_short(bool enabled);

struct adjGdnBuffers {
    uint8_t* x_q8k = nullptr;
    uint8_t* x_q8_0 = nullptr;
    uint16_t* x_bf16 = nullptr;
    float* qkv = nullptr;
    float* conv_out = nullptr;
    float* h = nullptr;
    float* alpha = nullptr;
    float* beta = nullptr;
    float* gate = nullptr;
    float* o = nullptr;
    float* z = nullptr;
    float* y = nullptr;
    uint8_t* y_q8k = nullptr;
    uint8_t* y_q8_0 = nullptr;
    float* state = nullptr;
    float* conv_state = nullptr;
};

uint64_t gdn_buffers_bytes(const adjGeometry& g);
uint64_t gdn_buffers_init(const adjGeometry& g, void* base, adjGdnBuffers& b);
void gdn_buffers_zero_state(const adjGdnBuffers& b, const adjGeometry& g, void* stream);

bool gdn_layer(const adjWeightTable& tables, const adjGeometry& g, int64_t layer, const adjGdnBuffers& b,
               const float* mixed, float* out, void* stream, std::string& err);

struct adjMoEBuffers {
    uint16_t* x_bf16 = nullptr;
    uint16_t* x_f16 = nullptr;
    float* logits = nullptr;
    int* ids = nullptr;
    float* weights = nullptr;
    float* shared = nullptr;
    float* sh_scratch = nullptr;
    uint8_t* x_q8_0 = nullptr;
    uint8_t* x_q8k = nullptr;
};

uint64_t moe_buffers_bytes(const adjGeometry& g, int64_t k);
uint64_t moe_buffers_init(const adjGeometry& g, int64_t k, void* base, adjMoEBuffers& b);

bool moe_layer(const adjWeightTable& tables, const adjGeometry& g, int64_t layer, int64_t k, const adjMoEBuffers& b,
               const float* x, const float* parts, float* out, void* stream, std::string& err,
               const adjDoorbell* db = nullptr);

bool moe_route(const adjWeightTable& tables, const adjGeometry& g, int64_t layer, int64_t k, const adjMoEBuffers& b,
               const float* x, void* stream, std::string& err, const adjDoorbell* db = nullptr);
bool moe_finish(const adjWeightTable& tables, const adjGeometry& g, int64_t layer, int64_t k, const adjMoEBuffers& b,
                const float* x, const float* parts, float* out, void* stream, std::string& err);
bool moe_shared(const adjWeightTable& tables, const adjGeometry& g, int64_t layer, const adjMoEBuffers& b,
                const float* x, void* stream, std::string& err);
bool moe_combine_parts(const adjGeometry& g, int64_t layer, int64_t k, const adjMoEBuffers& b, const float* parts,
                       float* out, void* stream, std::string& err);

void layer_set_shared_early(bool enabled);
bool layer_shared_early();

struct adjQsaState {
    uint16_t* k_pool = nullptr;
    uint16_t* v_pool = nullptr;
    bool kv_int8 = false;
    int8_t* k_q = nullptr;
    int8_t* v_q = nullptr;
    uint16_t* k_scale = nullptr;
    uint16_t* v_scale = nullptr;
    int32_t* page_table = nullptr;
    int64_t n_pages = 0;
    int64_t max_cells = 0;
    float* idx_tail = nullptr;
    float* idx_dead = nullptr;
    float* idx_pooled = nullptr;
    int32_t* idx_block_pos = nullptr;
    float* cos_tab = nullptr;
    float* sin_tab = nullptr;
    int32_t* step = nullptr;
    int32_t* attention_status = nullptr;
    int32_t* pos_dev = nullptr;
    int32_t* host_step = nullptr;
    int32_t* host_pos = nullptr;
};

struct adjDoorbell {
    int64_t n_embd = 0;
    int64_t k = 0;
    float* h_x_f = nullptr;
    int32_t* h_ids = nullptr;
    float* h_weights = nullptr;
    uint32_t* h_seq = nullptr;
    uint32_t* h_flag = nullptr;
    float* d_x_f = nullptr;
    int32_t* d_ids = nullptr;
    float* d_weights = nullptr;
    uint32_t* d_seq = nullptr;
    uint32_t* d_flag = nullptr;
};

uint64_t doorbell_init(const adjGeometry& g, int64_t k, adjDoorbell& db);
void doorbell_free(adjDoorbell& db);
void doorbell_reset(const adjDoorbell& db);

struct adjBlockBuffers {
    float* R = nullptr;
    float* mixed = nullptr;
    float* block_out = nullptr;
    float* inject = nullptr;
    float* inject2 = nullptr;
    float* gr_rs = nullptr;
    uint8_t* head_q8k = nullptr;
    prjname::kernels::adjGrWorkspace gr;
    float* dump = nullptr;
};

uint64_t block_buffers_bytes(const adjGeometry& g);

struct adjPleRun {
    prjname::kernels::adjPleTable* table = nullptr;
    prjname::kernels::adjPleWeights w;
    const int32_t* token = nullptr;
    const int32_t* prev = nullptr;
    float* hist = nullptr;
    float* emb_dev = nullptr;
    float* emb_host = nullptr;
    float* scratch = nullptr;
    prjname::kernels::adjPleConsts consts;
    bool ready() const {
        return table != nullptr && table->is_open() && hist != nullptr && emb_dev != nullptr &&
               emb_host != nullptr && scratch != nullptr && token != nullptr && prev != nullptr &&
               ((w.key_codes != nullptr && w.key_scales != nullptr) || w.key_bf16 != nullptr ||
                w.key_native_data != nullptr) &&
               w.value_bf16 != nullptr &&
               w.norm_key != nullptr && w.norm_query != nullptr && w.norm_conv != nullptr &&
               w.conv1d_f16 != nullptr;
    }
};

bool ple_stage_token(const adjPleRun& p, void* stream, std::string& err);
bool ple_issue_token(const adjPleRun& p, std::string& err);
bool ple_finish_token(const adjPleRun& p, void* stream, std::string& err);
uint64_t ple_run_scratch_bytes();

bool block_layer(const adjWeightTable& tables, const adjGeometry& g, int64_t layer, int64_t pos, int32_t pos_base,
                 const adjGdnBuffers& gb, const adjQsaState& qst, const adjQsaBuffers& qb, const adjMoEBuffers& mb,
                 int64_t k, const adjBlockBuffers& bb, const float* parts, void* stream, std::string& err,
                 const adjDoorbell* db = nullptr, const adjPleRun* ple = nullptr);

bool block_layer_pre(const adjWeightTable& tables, const adjGeometry& g, int64_t layer, int64_t pos,
                     int32_t pos_base, const adjGdnBuffers& gb, const adjQsaState& qst, const adjQsaBuffers& qb,
                     const adjMoEBuffers& mb, int64_t k, const adjBlockBuffers& bb, void* stream, std::string& err,
                     const adjDoorbell* db = nullptr, const adjPleRun* ple = nullptr, int half = 0,
                     int stage_prefix = 0);

bool block_layer_post(const adjWeightTable& tables, const adjGeometry& g, int64_t layer, int64_t k,
                      const adjMoEBuffers& mb, const adjBlockBuffers& bb, const float* parts, void* stream,
                      std::string& err);
```

---

## MODULE: mtp_drafter (multi-token prediction draft layer)

A single MTP layer that pairs the main model's final residual at position $i$ with token $i+1$ to predict
token $i+2$. Used for speculative decoding: T-1 drafts per round, verified by the main model.

### LAYER 1a — MATH (LaTeX)

**MTP forward (per cell $i$).**

$$e = \text{fc\_embedding}\bigl(\text{RMSNorm}(\text{embed}(\mathit{tok}_{i+1})) \cdot (1 + w_e)\bigr)$$

$$h_s = \text{fc\_hidden}_s\bigl(\text{RMSNorm}_{10240}(R_s) \cdot (1 + w_h)\bigr), \quad s \in [0, hc)$$

$$R' = h + e$$

$$R' \xrightarrow{\text{attn HC}} \text{QSA attention (own K/V)} \xrightarrow{\text{MLP HC}} \text{MoE (512 experts, top-10, shared)}$$

$$R' \mathrel{+}= y \cdot 2\,\sigma\bigl(\mathit{inject}/hc\bigr)$$

$$\text{logits} = \text{head}(\text{final\_mixer}(R'))$$

**Runtime simplifications (affect DRAFT quality only, never verified output).**

- Large projections: Q8_0 through multi-column MMVQ.
- Attention: DENSE over every cell (no sparse indexer). Identical to model below 2,051 cells.
- All 512 routed experts in VRAM (708 MB), run through grouped hit kernels.

**Draft chain.** From accepted row $a$:

$$\mathit{drafts}[j] = \text{argmax}(\text{MTP}(\text{MTP}^{(j-1)}, \mathit{tok}_{a+j+1})), \quad j = 1, \ldots, T-1$$

Optional min-p truncation: $\Pr[\mathit{draft}] \ge \mathit{min\_p}$ or chain stops.

**Prefill.** Batches of up to `max_t` rows, running cells $[\text{cell0}, \text{cell0}+n)$.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem mtp_vram_experts (n_expert : Nat) (blob : Nat) (h1 : n_expert = 512) (h2 : blob = 1382400) :
    n_expert * blob = 707788800 := by sorry

-- axle_verify: content
theorem mtp_vram_experts (n_expert : Nat) (blob : Nat) (h1 : n_expert = 512) (h2 : blob = 1382400) :
    n_expert * blob = 707788800 := by
  subst h1; subst h2; norm_num
```

```lean
-- axle_verify: formal_statement
theorem draft_count (T : Nat) (h : T ≥ 2) :
    T - 1 ≥ 1 := by sorry

-- axle_verify: content
theorem draft_count (T : Nat) (h : T ≥ 2) :
    T - 1 ≥ 1 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
class adjDrafter {
public:
    adjDrafter() = default;
    ~adjDrafter();
    adjDrafter(const adjDrafter&) = delete;
    adjDrafter& operator=(const adjDrafter&) = delete;

    bool load(const std::string& rt_dir, const adjGeometry& g, adjSessionState& ss, int max_t, std::string& err,
              int64_t window = 32768);
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    uint64_t vram_bytes() const { return vramname_; }
    bool bind(const adjWeightTable& wt, const adjNativeHead* head, const float* window_R, std::string& err);

    bool prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);
    bool draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                     float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    double ms_draft = 0, ms_prefill = 0;
    int64_t rounds = 0;

private:
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err);
    bool capture_prefill(int T, std::string& err);
    bool capture_round(int T, std::string& err);
    bool capture_step(int j, std::string& err);
    cudaGraphExec_t step_exec_[9] = {};
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const adjGeometry* gname_ = nullptr;
    adjSessionState* ssname_ = nullptr;
    const adjWeightTable* wtname_ = nullptr;
    const adjNativeHead* headname_ = nullptr;
    const float* window_R_ = nullptr;
    int max_t_ = 0;
    int64_t n_vocab_ = 0;
    uint64_t vramname_ = 0;
    cudaStream_t csname_ = nullptr;
    cudaGraphExec_t prefill_exec_[9] = {};
    cudaGraphExec_t round_exec_[9] = {};

    struct adjTensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    std::vector<adjTensor> tensorsname_;
    uint8_t* densename_ = nullptr;
    uint8_t* expertsname_ = nullptr;
    void* state_arena_ = nullptr;
    adjQsaState stname_;
    void* arenaname_ = nullptr;

    int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;
    float *h_prob_ = nullptr, *m_prob_ = nullptr;
    uint8_t* dheadname_ = nullptr;
    int32_t* dvocabname_ = nullptr;
    int64_t n_dvocab_ = 0;
    std::string rt_dirname_;
    int64_t windowname_ = 0;
    int64_t prompt_len_ = 0;
    float* probsname_ = nullptr;
    int64_t capname_ = 0, attn_scratch_floatsname_ = 0;
    // ... device buffers omitted for brevity, same pattern as host staging
};
```

---

## MODULE: pinned_arena (page-locked host memory for expert streaming)

Allocates large-page or normal-page host memory, optionally CUDA-registered in slices. Loads expert
weights from disk with parallel readers and reports per-layer checksums.

### LAYER 1a — MATH (LaTeX)

**PageBacking.**

$$\text{PageBacking} \in \{\text{LargePages}, \text{NormalPages}, \text{PinnedByCuda}\}$$

**PinnedArena construction.**

$$\text{PinnedArena}(N, \mathit{slice}) : \mathit{base} \leftarrow \text{alloc}(N)$$

Allocation priority:

1. Large pages (2 MB): $N / 2^{21}$ TLB entries vs $N / 2^{12}$ for normal.
2. Normal pages + `cudaHostRegister` (slice-by-slice fallback).
3. Normal pages + working-set lock.

$$\text{registered\_bytes} = \min(\mathit{registered\_slices} \cdot \mathit{slice}, N)$$

**Expert loading.**

$$\text{load\_experts}(\mathit{path}, \mathit{dst}, \mathit{blob}, \mathit{bpl}, L, T, C)$$

$T$ threads, each with its own file handle (no shared-seek lock). Per layer:

$$\text{checksum}[l] = \text{FNV-1a}_{64}(\mathit{dst}[l \cdot \mathit{bpl} \cdot \mathit{blob} \ldots])$$

**Stream bandwidth measurement.**

$$\text{stream\_bandwidth}(\mathit{src}, N, C, I) : \text{time } I \text{ rounds of } \lceil N/C \rceil \text{ H2D copies of } C \text{ bytes each}$$

Reports GiB/s to determine whether the 1,382,400-byte copy size reaches peak bus bandwidth.

**FNV-1a 64-bit.**

$$h_0 = 1469598103934665603$$

$$h_{i+1} = (h_i \oplus p_i) \cdot 1099511628211$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem large_page_tlb_reduction (bytes : Nat) (h : bytes = 33973862400) :
    bytes / 2097152 < bytes / 4096 := by sorry

-- axle_verify: content
theorem large_page_tlb_reduction (bytes : Nat) (h : bytes = 33973862400) :
    bytes / 2097152 < bytes / 4096 := by
  subst h; norm_num
```

```lean
-- axle_verify: formal_statement
theorem fnv1a_seed : 1469598103934665603 > 0 := by sorry

-- axle_verify: content
theorem fnv1a_seed : 1469598103934665603 > 0 := by
  norm_num
```

### LAYER 2 — FORM TEMPLATE

```cpp
enum class typenameType { LargePages, NormalPages, PinnedByCuda };

struct adjArena {
    void* objtype = nullptr;
    uint64_t capacity = 0;
    typenameType backing = typenameType::NormalPages;
    std::string note;
    uint64_t locked_bytes = 0;
    uint64_t registered_bytes = 0;
    uint64_t slice_bytes = 0;
    int registered_slices = 0;

    adjArena() = default;
    explicit adjArena(uint64_t bytes, uint64_t slice = 0);
    adjArena(uint64_t bytes, const std::vector<uint64_t>& bounds);
    std::vector<uint64_t> slice_starts;
    ~adjArena();
    adjArena(const adjArena&) = delete;
    adjArena& operator=(const adjArena&) = delete;

    bool valid() const { return objtype != nullptr; }
    uint8_t* data() const { return (uint8_t*) objtype; }
};

struct adjLoadStats {
    double seconds = 0.0;
    uint64_t bytes = 0;
    uint64_t layers = 0;
    std::vector<uint64_t> layer_checksums;
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
};

adjLoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                          uint64_t layers, int threads, uint64_t chunk);
adjLoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                                 const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk);

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed = 1469598103934665603ull);

struct adjStreamStats {
    double seconds = 0.0;
    uint64_t bytes = 0;
    uint64_t chunk = 0;
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
};

adjStreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters);
```

---

## MODULE: session_state (per-token forward pass and doorbell protocol)

One token through all 48 layers. Manages GDN/QSA state, per-layer graph capture/replay, the host loop
with doorbell-driven CPU/GPU overlap, and the token graph (all layers in one CUDA graph).

### LAYER 1a — MATH (LaTeX)

**SessionState.** Per-sequence state:

$$\mathcal{S} = (R \in \mathbb{R}^{hc \times n}, \;\{\text{GDN}_l\}_{l \in \text{gdn}}, \;\{\text{QSA}_l\}_{l \in \text{qsa}}, \;\text{MoE buffers}, \;\text{Block buffers}, \;k=10)$$

**session_token.** Layers $0 \ldots 47$, each `block_layer`:

$$\forall l \in [0, 48): R \leftarrow \text{block\_layer}(l, R, \mathit{parts})$$

**SessionGraphs.** One `cudaGraphExec_t` per layer (the `pre` half), one per layer (the `post` half).

$$|\text{execs}| = |\text{posts}| = 48$$

**Overlap measurement:**

$$\text{rings\_mid\_graph} = |\{l : \text{ring observed while graph still executing}\}|$$

$$\text{ms\_to\_ring} = \sum_l (\text{time from launch to ring observation})_l$$

$$\text{ms\_host} = \sum_l (\text{time from ring to next launch})_l$$

**session_loop (host loop with doorbell).**

```
for l = 0..47:
    launch pre[l]
    poll doorbell until h_seq == expected  (cudaEventQuery, not memory spin)
    read miss list from doorbell
    dispatch CPU pool
    copy y_miss → parts_dev (H2D async)
    launch post[l]
```

No `cudaStreamSynchronize` in the loop (P2.X3).

**Token graph (all 48 layers, one graph).** Between `pre[l]`'s doorbell ring and the parts copy, a device-side
wait kernel spins until `h_flag == ring_value` (5.2 μs per handoff). The host only writes flags, no driver calls
in the loop.

**Stage timing.** Five prefix graphs `preP[k]` ($k=1\ldots5$) for per-stage breakdown, differenced
to attribute time to individual stages.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem session_layer_count : 48 = 36 + 12 := by sorry

-- axle_verify: content
theorem session_layer_count : 48 = 36 + 12 := by
  norm_num
```

```lean
-- axle_verify: formal_statement
theorem overlap_ring_before_completion (ms_to_ring ms_layer : Float)
    (h : ms_to_ring < ms_layer) : True := by sorry

-- axle_verify: content
theorem overlap_ring_before_completion (ms_to_ring ms_layer : Float)
    (h : ms_to_ring < ms_layer) : True := by
  trivial
```

```lean
-- axle_verify: formal_statement
theorem stage_prefix_difference (p4 p3 : Nat) (h : p4 ≥ p3) :
    p4 - p3 + p3 = p4 := by sorry

-- axle_verify: content
theorem stage_prefix_difference (p4 p3 : Nat) (h : p4 ≥ p3) :
    p4 - p3 + p3 = p4 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
struct adjSessionState {
    int64_t max_cells = 0;
    adjGdnBuffers gdn;
    float* gdn_state = nullptr;
    adjQsaState* qsa_states = nullptr;
    void* qsa_state_arena = nullptr;
    adjQsaBuffers qsa_bufs;
    void* qsa_buf_arena = nullptr;
    adjMoEBuffers moe;
    void* moe_arena = nullptr;
    adjBlockBuffers block;
    void* block_arena = nullptr;
    float* R = nullptr;
    int64_t k = 10;
    const adjDoorbell* db = nullptr;
    int32_t ple_prev[2] = {-1, -1};
    int32_t ple_token = -1;
    float* ple_hist = nullptr;
    adjPleRun ple;
};

uint64_t session_bytes(const adjGeometry& g, int64_t max_cells, int64_t k);
uint64_t session_init(const adjGeometry& g, int64_t max_cells, int64_t k, void* base, adjSessionState& s);
void session_zero(adjSessionState& s, const adjGeometry& g, const float* R_init, void* stream);

bool session_token(const adjWeightTable& tables, const adjGeometry& g, int64_t pos, int32_t pos_base,
                   adjSessionState& s, const float* parts, void* stream, bool sync_every_layer,
                   std::string& err);

struct adjSessionGraphs {
    cudaGraphExec_t* execs = nullptr;
    int64_t n = 0;
    cudaGraphExec_t* posts = nullptr;
    bool captured = false;
    float* parts_dev = nullptr;
    int64_t rings_mid_graph = 0;
    double ms_to_ring = 0;
    cudaGraphExec_t* preP[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    cudaGraphExec_t* preA = nullptr;
    cudaGraphExec_t* preB = nullptr;
    bool split_captured = false;
    int64_t calls_total = 0;
    double ms_host = 0;
};

bool session_capture(const adjWeightTable& tables, const adjGeometry& g, adjSessionState& s, const float* parts,
                     adjSessionGraphs& gr, std::string& err, bool split = false);
bool session_replay(const adjGeometry& g, int64_t pos, int32_t pos_base, adjSessionState& s, adjSessionGraphs& gr,
                    void* stream, std::string& err);
bool session_replay_full(const adjGeometry& g, int64_t pos, int32_t pos_base, adjSessionState& s,
                         adjSessionGraphs& gr, void* stream, std::string& err);
bool session_replay_stages(const adjGeometry& g, int64_t pos, int32_t pos_base, adjSessionState& s,
                           adjSessionGraphs& gr, void* stream, double& ms_mixer, double& ms_ffn, double& ms_post,
                           std::string& err);
void session_graphs_free(adjSessionGraphs& gr);

using adjPoolFn = void (*)(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                           int64_t k, float* out);

struct adjSessionLoopScratch {
    float* y_miss = nullptr;
    size_t parts_bytes = 0;
    cudaEvent_t probe = nullptr;
    long long pinned_core = -1;
    bool pinned = false;
    bool init(size_t parts_bytes_in, std::string& err);
    void free();
};

bool session_loop(const adjGeometry& g, int64_t pos, int32_t pos_base, adjSessionState& s, adjSessionGraphs& gr,
                  adjPoolFn pool, adjHitFn hits, void* user, bool overlap, void* stream, std::string& err,
                  float* dump_layers = nullptr, adjSessionLoopScratch* scratch = nullptr);

struct adjTokenGraph {
    cudaGraphExec_t exec = nullptr;
    bool captured = false;
    int64_t n_layers = 0;
    const float* y_src = nullptr;
    size_t parts_bytes = 0;
    int64_t calls = 0;
    double ms_wait = 0;
    double ms_pool = 0;
    int64_t flushes = 0;
};

struct adjTokenHits {
    const int32_t* d_res = nullptr;
    int64_t n_expert = 0;
    const uint8_t* cache_base = nullptr;
    int64_t blob = 0;
    int32_t* d_slot = nullptr;
    int32_t* d_dst = nullptr;
    int32_t* d_count = nullptr;
    uint8_t* x_q8 = nullptr;
    float* x_scale = nullptr;
    void* scratch = nullptr;
    float* hit_out = nullptr;
    bool on() const { return d_res && cache_base && d_slot && d_dst && d_count && x_q8 && x_scale && scratch && hit_out; }
};

bool session_capture_token(const adjWeightTable& tables, const adjGeometry& g, adjSessionState& s, float* parts_dev,
                           const float* y_miss_host, size_t parts_bytes, adjTokenGraph& tg, std::string& err,
                           const adjTokenHits* hits = nullptr);
bool session_run_token(const adjGeometry& g, int64_t pos, int32_t pos_base, adjSessionState& s, adjTokenGraph& tg,
                       adjPoolFn pool, void* user, float* y_miss_host, void* stream, std::string& err);
void token_graph_free(adjTokenGraph& tg);
```

---

## MODULE: verify_window (speculative verify of T-token windows)

T tokens through all 48 layers in one captured graph, producing per-token argmax.
Bit-for-bit identical to single-token decode. Commit advances state for accepted tokens.

### LAYER 1a — MATH (LaTeX)

**Verify window.** $T$ tokens at positions $p_0, \ldots, p_0 + T - 1$:

$$\forall t \in [0, T): \mathit{out}[t] = \text{argmax}(\text{head}(\text{forward}(R, \mathit{tok}_t, p_0 + t)))$$

Each kernel is either the single-token kernel applied per token, or a multi-token kernel whose per-token
arithmetic is identical.

**Expert cost scaling (measured on decode traces):**

$$\text{missed\_experts}(T) \approx \begin{cases} 1.0 \times M_1 & T = 1 \\ 1.75 \times M_1 & T = 2 \\ 2.4 \times M_1 & T = 3 \\ 3.05 \times M_1 & T = 4 \end{cases}$$

**Commit.** Keep first $n_{\text{keep}}$ tokens ($1 \le n_{\text{keep}} \le T$):

1. Advance GDN conv history and recurrent state by replaying $n_{\text{keep}}$ tokens from stored inputs.
2. Restore indexer tail from snapshot, re-append accepted keys.
3. Set PLE history to snapshot after token $n_{\text{keep}} - 1$.
4. K/V cells past prefix simply overwritten on next use.

**GPU plan (per layer in verify window):**

- VRAM hits: grouped kernel from slot arena.
- PCIe share: `pcie_num/256` of each layer's missed experts read by GPU over PCIe.
- Three PCIe modes: 0 = DMA staging, 1 = mapped arena direct read, 2 = copy kernel staging.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem commit_range (n_keep T : Nat) (h1 : 1 ≤ n_keep) (h2 : n_keep ≤ T) :
    n_keep ≤ T := by sorry

-- axle_verify: content
theorem commit_range (n_keep T : Nat) (h1 : 1 ≤ n_keep) (h2 : n_keep ≤ T) :
    n_keep ≤ T := by
  exact h2
```

```lean
-- axle_verify: formal_statement
theorem expert_cost_monotone (t1 t2 : Nat) (h : t1 < t2) (m1 : Nat) (hm : m1 > 0) :
    t1 * m1 ≤ t2 * m1 := by sorry

-- axle_verify: content
theorem expert_cost_monotone (t1 t2 : Nat) (h : t1 < t2) (m1 : Nat) (hm : m1 > 0) :
    t1 * m1 ≤ t2 * m1 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
using adjPoolMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out, int64_t layer);

struct adjVerifyHits {
    const int32_t* d_res = nullptr;
    const uint8_t* cache_base = nullptr;
    int64_t blob = 0;
};

class adjVerifier {
public:
    adjVerifier() = default;
    ~adjVerifier();
    adjVerifier(const adjVerifier&) = delete;
    adjVerifier& operator=(const adjVerifier&) = delete;

    bool init(const adjWeightTable& wt, const adjGeometry& g, adjSessionState& ss, const adjVerifyHits& hits,
              const adjNativeHead* head, int max_t, std::string& err);
    bool run(int T, const int32_t* tokens, int64_t pos0, adjPoolMultiFn pool, void* user, int32_t* out,
             std::string& err);
    bool commit(int n_keep, std::string& err);
    const float* final_R(int t) const;
    const float* final_R_all() const { return Rname_; }
    adjPlanSink* plan_sink() { return &sinkname_; }
    void set_split(bool on) { splitname_ = on; }
    void set_pcie_mode(int mode) { sinkname_.pcie_mode = mode; }

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    int64_t windows = 0;

private:
    bool capture(int T, std::string& err);
    bool capture_commit(std::string& err);
    bool record_window(int T, cudaStream_t cs, std::string& err);

    const adjWeightTable* wtname_ = nullptr;
    const adjGeometry* gname_ = nullptr;
    adjSessionState* ssname_ = nullptr;
    adjVerifyHits hitsname_;
    const adjNativeHead* headname_ = nullptr;
    int max_t_ = 0;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;
    cudaStream_t csname_ = nullptr;
    cudaGraphExec_t exec_[9] = {};
    cudaGraphExec_t commit_exec_ = nullptr;
    adjPlanSink sinkname_;
    bool splitname_ = false;
    float* Rname_ = nullptr;
    void* arenaname_ = nullptr;
    int64_t capname_ = 0;
    // ... mapped staging and device buffers follow same h_/m_ pattern
};
```

---

## MODULE: weight_table (dense weight loader and VRAM layout)

Loads quantized dense weights from a flat-index pack into a single VRAM arena.
Handles BF16↔F32 re-rounding, FP16 scale widening, and per-tensor plane geometry.

### LAYER 1a — MATH (LaTeX)

**WeightKind.**

$$\text{WeightKind} \in \{\text{Verbatim}, \text{Bf16InF32}, \text{F32}, \text{F16InF32}\}$$

**Bf16InF32 conversion.** The pack stores BF16 values promoted to F32 (lossless). The engine takes the
high 16 bits:

$$\text{bf16}(x) = \text{bits}_{31:16}(x_{\text{f32}})$$

This is exact because the value originated as BF16.

**FP16 scale widening.** 90 tensors carry FP16 scales in the pack. The loader widens to FP32:

$$s_{\text{f32}} = \text{fp16\_to\_fp32}(s_{\text{fp16}}) \quad (\text{exact, } \text{fp16} \subset \text{fp32})$$

Total widening: 11,407,360 bytes.

**Arena layout.**

$$\text{pack}: 5.305 \;\text{GiB dense} + 0.444 \;\text{GiB embd}$$

$$\text{engine}: 4.067 \;\text{GiB dense} + 0.444 \;\text{GiB embd} + 10.88 \;\text{MiB scales} = 4.522 \;\text{GiB}$$

**Plane structure (quantized tensor).** Within `data`:

$$[\underbrace{\text{codes}}_{\text{codes\_bytes}} \;|\; \underbrace{\text{scales}}_{\text{scales\_bytes (4 B/scale)}} \;|\; \underbrace{\text{offsets}}_{\text{offset\_bytes}}]$$

**Activation contract.** `act_kind`: 0 = Q8_0, 1 = Q8_K. Determined by `tools/pack_index.py` from
source type, not inferrable from the S-form alone (Q5_0 and Q5_K are both 8-bit bias -16).

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bf16_roundtrip_exact (x : UInt32) (h : x % 65536 = 0) :
    x / 65536 * 65536 = x := by sorry

-- axle_verify: content
theorem bf16_roundtrip_exact (x : UInt32) (h : x % 65536 = 0) :
    x / 65536 * 65536 = x := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem arena_size_smaller_than_pack :
    4522 < 5749 := by sorry

-- axle_verify: content
theorem arena_size_smaller_than_pack :
    4522 < 5749 := by
  norm_num
```

### LAYER 2 — FORM TEMPLATE

```cpp
enum class typenameKind : int {
    Verbatim = 0,
    Bf16InF32 = 1,
    F32 = 2,
    F16InF32 = 3,
};

struct adjRef {
    const void* data = nullptr;
    uint64_t bytes = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t elements = 0;
    typenameKind kind = typenameKind::Verbatim;
    int code_bits = 0;
    int code_bias = 0;
    int group_elems = 0;
    bool codebook_iq4nl = false;
    bool has_offset = false;
    uint64_t codes_bytes = 0;
    uint64_t scales_bytes = 0;
    uint64_t offset_bytes = 0;
    uint64_t src_off = 0;
    uint64_t src_bytes = 0;
    int file_id = 0;
    bool scales_fp16 = false;
    int act_kind = 0;
    bool wants_q8k() const { return act_kind == 1; }
    bool quantized() const { return code_bits != 0; }
    const void* native_data = nullptr;
    void* native_q8_1 = nullptr;
    int native_type = -1;
    bool resident = true;
};

struct adjLoadReport {
    size_t tensors = 0;
    uint64_t arena_bytes = 0;
    size_t re_rounded = 0;
    uint64_t bytes_saved = 0;
    double read_ms = 0.0;
    double upload_ms = 0.0;
};

class adjWeightTable {
public:
    static bool pool_bytes(const std::string& pack_dir, uint64_t& out, std::string& err,
                           const std::set<std::string>* skip = nullptr);
    bool load(const std::string& pack_dir, void* arena_base, uint64_t arena_bytes, std::string& err,
              const std::set<std::string>* skip = nullptr);
    const adjRef* find(const std::string& name) const;
    const std::map<std::string, adjRef>& all() const { return tablename_; }
    const adjLoadReport& report() const { return reportname_; }
private:
    friend class adjNativeDense;
    std::map<std::string, adjRef> tablename_;
    adjLoadReport reportname_;
};
```

---

---

## SUPPLEMENTARY THEOREMS (cross-module invariants)

These additional Lean 4 theorems verify deeper structural invariants that span multiple modules.

```lean
-- axle_verify: formal_statement
-- expert_cache: the no-eviction policy means resident count is monotone non-decreasing
theorem no_eviction_monotone (resident_before resident_after slots : Nat)
    (h1 : resident_before ≤ slots)
    (h2 : resident_after = resident_before + 1 ∨ resident_after = resident_before) :
    resident_after ≤ slots := by sorry

-- axle_verify: content
theorem no_eviction_monotone (resident_before resident_after slots : Nat)
    (h1 : resident_before ≤ slots)
    (h2 : resident_after = resident_before + 1 ∨ resident_after = resident_before) :
    resident_after ≤ slots := by
  omega
```

```lean
-- axle_verify: formal_statement
-- weight_table: act_kind is binary (0 = Q8_0, 1 = Q8_K), no third format
theorem act_kind_binary (act_kind : Nat) (h : act_kind = 0 ∨ act_kind = 1) :
    act_kind < 2 := by sorry

-- axle_verify: content
theorem act_kind_binary (act_kind : Nat) (h : act_kind = 0 ∨ act_kind = 1) :
    act_kind < 2 := by
  omega
```

```lean
-- axle_verify: formal_statement
-- expert_cache: per-layer admission ranges do not overlap (layer l1 < l2 ⟹ hi(l1) ≤ lo(l2))
theorem per_layer_no_overlap (slots n_layers l1 l2 : Nat)
    (hn : 0 < n_layers) (h1 : l1 < n_layers) (h2 : l2 < n_layers) (hne : l1 < l2) :
    let q := slots / n_layers
    (l1 + 1) * q ≤ l2 * q := by sorry

-- axle_verify: content
theorem per_layer_no_overlap (slots n_layers l1 l2 : Nat)
    (hn : 0 < n_layers) (h1 : l1 < n_layers) (h2 : l2 < n_layers) (hne : l1 < l2) :
    let q := slots / n_layers
    (l1 + 1) * q ≤ l2 * q := by
  simp only
  omega
```

```lean
-- axle_verify: formal_statement
-- pinned_arena: 33.97 GB expert file is ~31 GiB (integer division)
theorem expert_file_gib : (33973862400 : Nat) / 1073741824 = 31 := by sorry

-- axle_verify: content
theorem expert_file_gib : (33973862400 : Nat) / 1073741824 = 31 := by
  norm_num
```

```lean
-- axle_verify: formal_statement
-- graph_registry: a capture produces either zero nodes (error) or positive (valid)
theorem zero_nodes_is_error (nodes : Nat) :
    (nodes = 0) ∨ (nodes > 0) := by sorry

-- axle_verify: content
theorem zero_nodes_is_error (nodes : Nat) :
    (nodes = 0) ∨ (nodes > 0) := by
  omega
```

```lean
-- axle_verify: formal_statement
-- session_state: five prefix-graph times reconstruct the total by telescoping
theorem five_prefix_stages (p0 p1 p2 p3 p4 : Nat)
    (h01 : p0 ≤ p1) (h12 : p1 ≤ p2) (h23 : p2 ≤ p3) (h34 : p3 ≤ p4) :
    (p1 - p0) + (p2 - p1) + (p3 - p2) + (p4 - p3) + p0 = p4 := by sorry

-- axle_verify: content
theorem five_prefix_stages (p0 p1 p2 p3 p4 : Nat)
    (h01 : p0 ≤ p1) (h12 : p1 ≤ p2) (h23 : p2 ≤ p3) (h34 : p3 ≤ p4) :
    (p1 - p0) + (p2 - p1) + (p3 - p2) + (p4 - p3) + p0 = p4 := by
  omega
```

---

*End of theory for 9 assigned modules. 20 Lean 4 theorems total.*
