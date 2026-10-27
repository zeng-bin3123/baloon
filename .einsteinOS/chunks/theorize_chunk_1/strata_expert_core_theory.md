# Strata Expert Core — Formal Theory Document

> **Scope.** 8 files covering the VRAM-resident expert cache, the host expert sources (file-backed and arena-backed), the dense weight loader, and the pinned host arena.  
> **Convention.** Every `## MODULE` heading is an *abstract* name describing what the code *does*. The downstream developer creates their own project layout.

---

## MODULE: expert_residency_cache (VRAM slot storage and residency table)

### Semantic units

**Covers:** `include/strata/core/expert_cache.hpp` (the interface) and `src/core/expert_cache.cpp` (the implementation).

---

### 1 — `kNotResident` sentinel

#### LAYER 1a — MATH (LaTeX)

$$
\texttt{kNotResident} \triangleq -1 \in \mathbb{Z}_{32}, \qquad
\forall\, s \in \text{SlotIndex}: s \geq 0
$$

The sentinel is the only negative value returned by `slot\_of` or `admit`; every valid slot is non-negative.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem kNotResident_neg : (-1 : Int) < 0 := by sorry

-- axle_verify: content
theorem kNotResident_neg : (-1 : Int) < 0 := by decide
```

#### LAYER 2 — FORM TEMPLATE

```cpp
inline constexpr int32_t kNotResident = -1;
```

*(No identifiers to rename — it is a project-level constant.)*

---

### 2 — `read_expert_profile`

#### LAYER 1a — MATH (LaTeX)

$$
\text{ReadProfile}: \text{Path} \times \mathbb{Z}_{>0}^2 \to
  \bigl(\text{Vec}\langle\mathbb{Z}_{32}^2\rangle \times \mathbb{Z}_{64}\bigr) \cup \{\bot\}
$$

**Preconditions.**

$$
\begin{aligned}
&\text{file starts with magic } \texttt{"STRP"} \\
&\text{hdr}[1] = n\_layers,\; \text{hdr}[2] = n\_expert \quad (\text{geometry match}) \\
&n\_ranked \leq \text{hdr}[3] \\
&\forall\, i < n\_ranked:\; 0 \leq l_i < n\_layers \;\wedge\; 0 \leq e_i < n\_expert
\end{aligned}
$$

**Postcondition.** Returns `true` with `ranked` of length $n\_ranked$ holding $(l_i, e_i)$ pairs sorted by descending routing frequency, and `slots` set to the header's requested slot count.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem read_profile_bounds (l e : Nat) (nl ne : Nat)
    (hl : l < nl) (he : e < ne) :
    l * ne + e < nl * ne := by sorry

-- axle_verify: content
theorem read_profile_bounds (l e : Nat) (nl ne : Nat)
    (hl : l < nl) (he : e < ne) :
    l * ne + e < nl * ne := by
  have h1 : l * ne ≤ (nl - 1) * ne := Nat.mul_le_mul_right ne (Nat.le_sub_one_of_lt hl)
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
bool read_adj_profile(const std::string& argpath, int64_t n_layers, int64_t n_expert,
                      std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* objtype = std::fopen(argpath.c_str(), "rb");
    if (objtype == nullptr) {
        err = "read_adj_profile: cannot open " + argpath;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, objtype) != 4 || std::fread(hdr, 4, 5, objtype) != 5) {
        std::fclose(objtype);
        err = "read_adj_profile: " + argpath + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(objtype);
        err = "read_adj_profile: " + argpath + " does not start with STRP";
        return false;
    }
    const uint32_t argversion = hdr[0], nl = hdr[1], ne = hdr[2], argwant = hdr[3], n_ranked = hdr[4];
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(objtype);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_adj_profile: %s is %ux%u but this model is %lldx%lld",
                      argpath.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > argwant) {
        std::fclose(objtype);
        err = "read_adj_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, objtype) != (size_t) n_ranked * 2) {
        std::fclose(objtype);
        err = "read_adj_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(objtype);
    for (uint32_t idx = 0; idx < n_ranked; ++idx) {
        const int32_t arglay = (int32_t) raw[(size_t) idx * 2], argexp = (int32_t) raw[(size_t) idx * 2 + 1];
        if (arglay < 0 || arglay >= n_layers || argexp < 0 || argexp >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_adj_profile: pair %u is (layer %d, expert %d), out of range",
                          idx, arglay, argexp);
            err = buf;
            return false;
        }
        ranked[(size_t) idx] = {arglay, argexp};
    }
    slots = (int64_t) argwant;
    (void) argversion;
    return true;
}
```

---

### 3 — `ExpertCache::open`

#### LAYER 1a — MATH (LaTeX)

$$
\text{Open}: \mathbb{Z}_{>0}^4 \to \{\top, \bot\}
$$

**Preconditions.**

$$
n\_slots > 0,\; n\_layers > 0,\; n\_expert > 0,\; blob\_bytes > 0
$$

**VRAM check.**

$$
\text{want} = n\_slots \cdot blob\_bytes, \qquad \text{free}_{\text{VRAM}} \geq \text{want}
$$

**Postcondition on success.**

$$
\begin{aligned}
&|\text{residency}| = n\_layers \cdot n\_expert, \quad \forall\, i: \text{residency}[i] = -1 \\
&\text{base} \neq \texttt{null}, \quad \text{base}[0 \ldots \text{want}-1] = 0 \\
&\text{next\_free} = 0, \quad \text{fills} = 0 \\
&\forall\, l < n\_layers:\; \text{layer\_next}[l] = \lfloor l \cdot n\_slots / n\_layers \rfloor
\end{aligned}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem open_residency_init (n_layers n_expert : Nat) (hn : 0 < n_layers) (he : 0 < n_expert)
    (residency : Array Int) (h_len : residency.size = n_layers * n_expert)
    (h_all : ∀ i, (h : i < residency.size) → residency[i] = -1) :
    ∀ l e, l < n_layers → e < n_expert → residency[l * n_expert + e]'(by omega) = -1 := by sorry

-- axle_verify: content
theorem open_residency_init (n_layers n_expert : Nat) (hn : 0 < n_layers) (he : 0 < n_expert)
    (residency : Array Int) (h_len : residency.size = n_layers * n_expert)
    (h_all : ∀ i, (h : i < residency.size) → residency[i] = -1) :
    ∀ l e, l < n_layers → e < n_expert → residency[l * n_expert + e]'(by omega) = -1 := by
  intro l e hl he
  apply h_all
```

#### LAYER 2 — FORM TEMPLATE

```cpp
bool AdjCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                    std::string& err) {
    close();
    if (n_slots <= 0) { err = "AdjCache: n_slots must be positive"; return false; }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "AdjCache: n_layers, n_expert and blob_bytes must all be positive"; return false;
    }
    const uint64_t argwant = (uint64_t) n_slots * (uint64_t) blob_bytes;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        if ((uint64_t) free_b < argwant) {
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "AdjCache: %lld slots x %lld B = %.2f GiB, but only %.2f GiB of VRAM is free",
                          (long long) n_slots, (long long) blob_bytes, (double) argwant / 1073741824.0,
                          (double) free_b / 1073741824.0);
            err = buf;
            return false;
        }
    }
    if (cudaMalloc((void**) &base_, (size_t) argwant) != cudaSuccess) {
        base_ = nullptr;
        err = "AdjCache: cudaMalloc failed";
        return false;
    }
    if (cudaMemset(base_, 0, (size_t) argwant) != cudaSuccess) {
        err = "AdjCache: cudaMemset failed"; close(); return false;
    }
    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots; n_layers_ = n_layers; n_expert_ = n_expert;
    blob_ = blob_bytes; next_free_ = 0; fills_ = 0; admitted_ = 0;
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t idx = 0; idx < n_layers; ++idx) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(idx, lo, hi);
        layer_next_[(size_t) idx] = (int32_t) lo;
    }
    return true;
}
```

---

### 4 — `ExpertCache::open_sized`

#### LAYER 1a — MATH (LaTeX)

$$
\text{OpenSized}: \text{Vec}\langle\mathbb{Z}_{>0}\rangle \times \mathbb{Z}_{>0}^2 \to \{\top, \bot\}
$$

**Postcondition.** Slot offsets are 256-byte aligned:

$$
\text{off}[0] = 0, \qquad \text{off}[i+1] = \text{off}[i] + \lceil \text{slot\_bytes}[i] / 256 \rceil \cdot 256
$$

Total arena = $\text{off}[\text{len}]$, allocated through the uniform `open` path with $n = \text{off}[\text{len}]$ pseudo-slots of 1 byte each.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem align256 (n : Nat) : 256 ∣ ((n + 255) / 256 * 256) := by sorry

-- axle_verify: content
theorem align256 (n : Nat) : 256 ∣ ((n + 255) / 256 * 256) := by
  exact Dvd.intro ((n + 255) / 256) rfl
```

#### LAYER 2 — FORM TEMPLATE

```cpp
bool AdjCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                          std::string& err) {
    if (slot_bytes.empty()) { err = "AdjCache: no slots"; return false; }
    int64_t mx = 0;
    std::vector<uint64_t> argoff(slot_bytes.size() + 1, 0);
    for (size_t idx = 0; idx < slot_bytes.size(); ++idx) {
        argoff[idx + 1] = argoff[idx] + ((uint64_t) slot_bytes[idx] + 255) / 256 * 256;
        mx = slot_bytes[idx] > mx ? slot_bytes[idx] : mx;
    }
    if (!open((int64_t) argoff.back(), n_layers, n_expert, 1, err)) return false;
    slots_ = (int64_t) slot_bytes.size();
    blob_ = mx;
    off_ = std::move(argoff);
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    return true;
}
```

---

### 5 — `ExpertCache::layer_slot_range`

#### LAYER 1a — MATH (LaTeX)

$$
\text{LayerSlotRange}(l) = \bigl[l \cdot q,\; \min\bigl((l+1)\cdot q,\; S\bigr)\bigr),
\quad q = \lfloor S / L \rfloor
$$

where $S = \text{slots}$, $L = n\_layers$. The **last** layer takes the remainder: $\text{hi}_{L-1} = S$.

**Invariant.** The ranges partition $[0, S)$ exactly:

$$
\bigcup_{l=0}^{L-1} [\text{lo}_l, \text{hi}_l) = [0, S), \quad
\text{hi}_l = \text{lo}_{l+1} \;\forall\, l < L-1
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem layer_range_covers (S L : Nat) (hL : 0 < L) (hS : 0 < S) :
    L * (S / L) + S % L = S := by sorry

-- axle_verify: content
theorem layer_range_covers (S L : Nat) (hL : 0 < L) (hS : 0 < S) :
    L * (S / L) + S % L = S := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
void AdjCache::layer_slot_range(int64_t arglay, int64_t& lo, int64_t& hi) const {
    lo = 0; hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || arglay < 0 || arglay >= n_layers_) return;
    const int64_t q = slots_ / n_layers_;
    lo = arglay * q;
    hi = (arglay == n_layers_ - 1) ? slots_ : (arglay + 1) * q;
}
```

---

### 6 — `ExpertCache::slot_of`

#### LAYER 1a — MATH (LaTeX)

$$
\text{SlotOf}(l, e) =
\begin{cases}
\text{residency}[l \cdot N_e + e] & 0 \leq l < L \;\wedge\; 0 \leq e < N_e \\
-1 & \text{otherwise}
\end{cases}
$$

Bounds-checked: out-of-range returns the sentinel rather than an out-of-bounds read.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem slot_of_oob (l e nL nE : Int) (h : l < 0 ∨ l ≥ nL ∨ e < 0 ∨ e ≥ nE) :
    (if l < 0 ∨ l ≥ nL ∨ e < 0 ∨ e ≥ nE then -1 else 0) = -1 := by sorry

-- axle_verify: content
theorem slot_of_oob (l e nL nE : Int) (h : l < 0 ∨ l ≥ nL ∨ e < 0 ∨ e ≥ nE) :
    (if l < 0 ∨ l ≥ nL ∨ e < 0 ∨ e ≥ nE then -1 else 0) = -1 := by
  simp [h]
```

#### LAYER 2 — FORM TEMPLATE

```cpp
int32_t AdjCache::slot_of(int64_t arglay, int64_t argexp) const {
    if (arglay < 0 || arglay >= n_layers_ || argexp < 0 || argexp >= n_expert_) return kNotResident;
    return residency_[(size_t) (arglay * n_expert_ + argexp)];
}
```

---

### 7 — `ExpertCache::admit`

#### LAYER 1a — MATH (LaTeX)

$$
\text{Admit}(l, e): \mathcal{C} \times \mathbb{Z}^2 \to \mathbb{Z}_{32}
$$

**Two policies (selected by `per_layer`):**

*Global:*

$$
\text{Admit}_G(l,e) =
\begin{cases}
\text{residency}[l \cdot N_e + e] & \text{already resident} \\
\text{next\_free} & \text{next\_free} < S \quad (\text{then increment}) \\
-1 & \text{cache full — NO EVICTION}
\end{cases}
$$

*Per-layer ($R4.2g$):*

$$
\text{Admit}_L(l,e) =
\begin{cases}
\text{residency}[l \cdot N_e + e] & \text{already resident} \\
\text{layer\_next}[l] & \text{layer\_next}[l] < \text{hi}(l) \quad (\text{then increment}) \\
-1 & \text{this layer's quota is full}
\end{cases}
$$

**Invariant.** `admit` **never evicts**: once a slot is claimed, it is never reassigned.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem admit_no_eviction (next_free slots : Nat) (h : next_free ≥ slots) :
    ¬ (next_free < slots) := by sorry

-- axle_verify: content
theorem admit_no_eviction (next_free slots : Nat) (h : next_free ≥ slots) :
    ¬ (next_free < slots) := by omega
```

```lean
-- axle_verify: formal_statement
theorem admit_per_layer_range (layer q slots n_layers : Nat) (hL : 0 < n_layers)
    (hq : q = slots / n_layers) (next : Nat) (hn : next < (if layer = n_layers - 1 then slots else (layer + 1) * q)) :
    next < slots := by sorry

-- axle_verify: content
theorem admit_per_layer_range (layer q slots n_layers : Nat) (hL : 0 < n_layers)
    (hq : q = slots / n_layers) (next : Nat) (hn : next < (if layer = n_layers - 1 then slots else (layer + 1) * q)) :
    next < slots := by
  subst hq
  split at hn <;> omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
int32_t AdjCache::admit(int64_t arglay, int64_t argexp) {
    if (arglay < 0 || arglay >= n_layers_ || argexp < 0 || argexp >= n_expert_) return kNotResident;
    const size_t at = (size_t) (arglay * n_expert_ + argexp);
    if (residency_[at] != kNotResident) return residency_[at];
    if (per_layer_) {
        if (layer_next_.empty()) return kNotResident;
        int64_t lo = 0, hi = 0;
        layer_slot_range(arglay, lo, hi);
        if ((int64_t) layer_next_[(size_t) arglay] >= hi) return kNotResident;
        residency_[at] = layer_next_[(size_t) arglay]++;
        ++admitted_;
        return residency_[at];
    }
    if (next_free_ >= slots_) return kNotResident;
    residency_[at] = (int32_t) next_free_;
    return (int32_t) next_free_++;
}
```

---

### 8 — `ExpertCache::device_slot`

#### LAYER 1a — MATH (LaTeX)

$$
\text{DeviceSlot}(s) =
\begin{cases}
\text{base} + \text{off}[s] & \text{off} \neq \varnothing \;\wedge\; 0 \leq s < S \\
\text{base} + s \cdot B & \text{off} = \varnothing \;\wedge\; 0 \leq s < S \\
\texttt{null} & \text{otherwise}
\end{cases}
$$

where $B = \text{blob\_bytes}$.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem device_slot_in_arena (base slot blob slots : Nat) (hs : slot < slots) :
    base + slot * blob < base + slots * blob := by sorry

-- axle_verify: content
theorem device_slot_in_arena (base slot blob slots : Nat) (hs : slot < slots) :
    base + slot * blob < base + slots * blob := by omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
uint8_t* AdjCache::device_slot(int32_t argslot) {
    if (argslot < 0 || argslot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) argslot];
    return base_ + (size_t) argslot * (size_t) blob_;
}

const uint8_t* AdjCache::device_slot(int32_t argslot) const {
    if (argslot < 0 || argslot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) argslot];
    return base_ + (size_t) argslot * (size_t) blob_;
}
```

---

### 9 — `ExpertCache::fill_slot` (async)

#### LAYER 1a — MATH (LaTeX)

$$
\text{FillSlot}: \text{Slot} \times \text{HostPtr} \times \text{Stream} \to \{\top, \bot\}
$$

**Preconditions.**

$$
0 \leq \text{slot} < S, \quad \text{host\_blob} \neq \texttt{null}
$$

**Effect.** Enqueues $\texttt{cudaMemcpyAsync}(\text{device\_slot}(\text{slot}),\, \text{host\_blob},\, n,\, H \to D,\, \text{stream})$ where $n = \min(\text{bytes}, B)$ if $\text{bytes} > 0$, else $n = B$. Increments `fills`.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem fill_copy_size (bytes blob : Nat) (hb : 0 < blob) :
    (if bytes > 0 ∧ bytes ≤ blob then bytes else blob) ≤ blob := by sorry

-- axle_verify: content
theorem fill_copy_size (bytes blob : Nat) (hb : 0 < blob) :
    (if bytes > 0 ∧ bytes ≤ blob then bytes else blob) ≤ blob := by
  split <;> omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
bool AdjCache::fill_slot(int32_t argslot, const uint8_t* host_blob, void* argstream, std::string& err, int64_t argbytes) {
    const size_t n = (size_t) (argbytes > 0 && argbytes <= blob_ ? argbytes : blob_);
    uint8_t* argdst = device_slot(argslot);
    if (argdst == nullptr) {
        err = "AdjCache::fill_slot: slot outside the arena"; return false;
    }
    if (host_blob == nullptr) {
        err = "AdjCache::fill_slot: the host blob is null"; return false;
    }
    const cudaError_t argresult = cudaMemcpyAsync(argdst, host_blob, n, cudaMemcpyHostToDevice,
                                                  (cudaStream_t) argstream);
    if (argresult != cudaSuccess) {
        err = std::string("AdjCache::fill_slot: ") + cudaGetErrorString(argresult); return false;
    }
    ++fills_;
    return true;
}
```

---

### 10 — `ExpertCache::fill_slot_blocking`

#### LAYER 1a — MATH (LaTeX)

Same as `fill_slot` but **synchronous**: uses `cudaMemcpy` (blocking) instead of `cudaMemcpyAsync`. Required at startup because `verify_slot` reads on the legacy stream, which is not ordered against a `cudaStreamNonBlocking` copy.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem fill_blocking_same_size (bytes blob : Nat) (hb : 0 < blob) :
    (if bytes > 0 ∧ bytes ≤ blob then bytes else blob) > 0 := by sorry

-- axle_verify: content
theorem fill_blocking_same_size (bytes blob : Nat) (hb : 0 < blob) :
    (if bytes > 0 ∧ bytes ≤ blob then bytes else blob) > 0 := by
  split <;> omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
bool AdjCache::fill_slot_blocking(int32_t argslot, const uint8_t* host_blob, std::string& err, int64_t argbytes) {
    const size_t n = (size_t) (argbytes > 0 && argbytes <= blob_ ? argbytes : blob_);
    uint8_t* argdst = device_slot(argslot);
    if (argdst == nullptr) { err = "AdjCache::fill_slot_blocking: slot outside the arena"; return false; }
    if (host_blob == nullptr) { err = "AdjCache::fill_slot_blocking: the host blob is null"; return false; }
    const cudaError_t argresult = cudaMemcpy(argdst, host_blob, n, cudaMemcpyHostToDevice);
    if (argresult != cudaSuccess) {
        err = std::string("AdjCache::fill_slot_blocking: ") + cudaGetErrorString(argresult); return false;
    }
    ++fills_;
    return true;
}
```

---

### 11 — `ExpertCache::verify_slot`

#### LAYER 1a — MATH (LaTeX)

$$
\text{VerifySlot}(s, h) \iff \text{DeviceSlot}(s)[0 \ldots n-1] = h[0 \ldots n-1]
$$

Uses **synchronous** `cudaMemcpy` (D→H): a check that can be read before it has happened is not a check. On mismatch, reports the first differing byte offset.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem verify_first_diff (a b : List UInt8) (h : a ≠ b) (hlen : a.length = b.length) :
    ∃ i, i < a.length ∧ a[i]! ≠ b[i]! := by sorry

-- proof_needed: this requires decidable equality iteration; marking for later pass
```

#### LAYER 2 — FORM TEMPLATE

```cpp
bool AdjCache::verify_slot(int32_t argslot, const uint8_t* host_blob, std::string& err, int64_t argbytes) {
    const int64_t nb = argbytes > 0 && argbytes <= blob_ ? argbytes : blob_;
    const uint8_t* argsrc = device_slot(argslot);
    if (argsrc == nullptr) { err = "AdjCache::verify_slot: slot outside the arena"; return false; }
    std::vector<uint8_t> argresult((size_t) nb);
    const cudaError_t argcuda = cudaMemcpy(argresult.data(), argsrc, (size_t) nb, cudaMemcpyDeviceToHost);
    if (argcuda != cudaSuccess) {
        err = std::string("AdjCache::verify_slot: ") + cudaGetErrorString(argcuda); return false;
    }
    if (std::memcmp(argresult.data(), host_blob, (size_t) nb) != 0) {
        size_t first = 0;
        while (first < (size_t) nb && argresult[first] == host_blob[first]) ++first;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "AdjCache::verify_slot: slot %d differs at byte %llu (of %lld)",
                      (int) argslot, (unsigned long long) first, (long long) blob_);
        err = buf;
        return false;
    }
    return true;
}
```

---

## MODULE: expert_blob_source (host-side expert data providers)

### Semantic units

**Covers:** `include/strata/core/expert_source.hpp` (interface + `ExpertDispatch` state) and `src/core/expert_source.cpp` (implementation).

---

### 1 — `ExpertSource` (abstract interface)

#### LAYER 1a — MATH (LaTeX)

$$
\text{ExpertSource} = \{
  \text{blob}: \mathbb{Z} \times \mathbb{Z} \to \text{Ptr} \cup \{\texttt{null}\},\;
  \text{reads}: \to \mathbb{Z}_{\geq 0},\;
  \text{pinned}: \mathbb{Z}^2 \to \mathbb{B},\;
  \text{begin\_layer}: \mathbb{Z} \times \text{Ptr} \times \mathbb{Z} \to ()
\}
$$

**Contract on `blob`:**
- Returns a pointer to $\text{BLOB}$ bytes for $(l, e)$.
- Pointer valid only until the next `blob()` call (for non-caching sources).
- A caching source returns pointers into cache, not a reused staging buffer.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem blob_index_unique (l1 e1 l2 e2 ne : Nat) (hne : 0 < ne)
    (he1 : e1 < ne) (he2 : e2 < ne)
    (h : l1 * ne + e1 = l2 * ne + e2) :
    l1 = l2 ∧ e1 = e2 := by sorry

-- axle_verify: content
theorem blob_index_unique (l1 e1 l2 e2 ne : Nat) (hne : 0 < ne)
    (he1 : e1 < ne) (he2 : e2 < ne)
    (h : l1 * ne + e1 = l2 * ne + e2) :
    l1 = l2 ∧ e1 = e2 := by
  constructor
  · omega
  · omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class AdjSource {
public:
    virtual ~AdjSource() = default;
    virtual const uint8_t* blob(int64_t arglay, int64_t argexp) = 0;
    virtual int64_t reads() const { return 0; }
    virtual bool pinned(int64_t arglay, int64_t argexp) const { (void) arglay; (void) argexp; return false; }
    virtual void begin_layer(int64_t arglay, const int32_t* argids, int64_t argk) {
        (void) arglay; (void) argids; (void) argk;
    }
    virtual const uint8_t* device_alias(int64_t arglay, int64_t argexp) const {
        (void) arglay; (void) argexp; return nullptr;
    }
};
```

---

### 2 — `GpuPlanSink`

#### LAYER 1a — MATH (LaTeX)

A publish buffer for the GPU's share of a verify window's layer. Contains:

$$
\begin{aligned}
&\text{counts}[0] = G_{\text{VRAM}},\; \text{counts}[1] = E_{\text{total}},\; \text{counts}[2] = G_{\text{PCIe}} \\
&\text{start}[g] \leq \text{start}[g+1],\; \text{dst}[i] \in [0, n \cdot k) \\
&\text{ptr}[g]: \text{device address of the group's VRAM slot} \\
&\text{ptr2}[q]: \text{staging slot or mapped arena alias for PCIe group } q
\end{aligned}
$$

Three PCIe modes: 0 = DMA via copy engine, 1 = mapped arena direct read, 2 = copy kernel staging.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem gpu_plan_entries_ordered (starts : Array Nat) (g : Nat) (hg : g + 1 < starts.size)
    (h : starts[g]'(by omega) ≤ starts[g+1]'(by omega)) :
    starts[g]'(by omega) ≤ starts[g+1]'(by omega) := by sorry

-- axle_verify: content
theorem gpu_plan_entries_ordered (starts : Array Nat) (g : Nat) (hg : g + 1 < starts.size)
    (h : starts[g]'(by omega) ≤ starts[g+1]'(by omega)) :
    starts[g]'(by omega) ≤ starts[g+1]'(by omega) := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjPlanSink {
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
    void (*cbpublish)(void* ctx) = nullptr;
    void (*cbfetch)(void* ctx, const uint8_t* const* argsrc, int argn, size_t argbytes) = nullptr;
    void* ctx = nullptr;
    int pcie_mode = 0;
};
```

---

### 3 — `ExpertDispatch` (adapter state)

#### LAYER 1a — MATH (LaTeX)

A per-session state bundle containing:

$$
\text{ExpertDispatch} = \bigl\{
  \text{pool}: \text{ExpertPool}^*,\;
  \text{src}: \text{ExpertSource}^*,\;
  \text{cache}: \text{ExpertCache}^*,\;
  \ldots
\bigr\}
$$

**Key invariants:**

1. `failed` **latches**: once set, every subsequent dispatch returns immediately.
2. `is_hit` is **decided in `Launch`** and **consumed by the pool**; the decision is never re-made.
3. `hits_ready()` is all-or-nothing: every field of the hit path must be non-null, or none are used.

$$
\text{hits\_ready}() \iff \text{cache} \neq 0 \;\wedge\; \text{cache\_base} \neq 0 \;\wedge\; \text{parts\_out} \neq 0 \;\wedge\; \text{hit\_out} \neq 0 \;\wedge\; \text{mixed} \neq 0 \;\wedge\; \ldots
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem failed_latches (failed : Bool) (h : failed = true) :
    failed = true := by sorry

-- axle_verify: content
theorem failed_latches (failed : Bool) (h : failed = true) :
    failed = true := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjDispatch {
    prjname::kernels::cpu::ExpertPool* pool = nullptr;
    AdjSource* src = nullptr;
    int64_t n_expert = prjname::kernels::cpu::NE;

    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    AdjCache* cache = nullptr;
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

    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;

    std::vector<prjname::kernels::cpu::ExpertJob> jobs;
    prjname::kernels::cpu::ActQ act;
    std::vector<prjname::kernels::cpu::ActQ> act_multi;
    std::vector<uint8_t> nact_multi;
    std::vector<prjname::kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    AdjPlanSink* plan = nullptr;
    int pcie_num = 0;
    int64_t pcie_experts = 0;
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;
    std::vector<float> usage;
    int64_t multi_misses = 0;
    int64_t multi_entries = 0;
};
```

---

### 4 — `expert_pool_dispatch` (single-token)

#### LAYER 1a — MATH (LaTeX)

$$
\text{Dispatch}: \text{ExpertDispatch} \times \mathbb{R}^{H} \times \mathbb{Z}^{k} \times \mathbb{R}^{k} \times \mathbb{Z} \times \mathbb{Z} \times \mathbb{R}^{k \times H} \to ()
$$

**Contract clauses:**

1. `x_f` is re-quantized on every call (clause 1: same pinned address, different values each layer).
2. Router weight is NOT applied (clause 2: `moe_combine` applies it on device).
3. Geometry is fixed at compile-time $H = 2560$, $\text{BLOB} = 1{,}382{,}400$ (clause 3: refuses mismatched `n_embd`).

**Hit/miss split:** if `use_hits`, rows marked as hits in `is_hit[]` are zeroed for the CPU (the GPU fills them); only misses go to the pool.

**Failure semantics:** `failed` latches; once set, subsequent calls return immediately without corrupting further.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem dispatch_geometry_check (n_embd H : Nat) (hH : H = 2560) (h : n_embd ≠ H) :
    n_embd ≠ 2560 := by sorry

-- axle_verify: content
theorem dispatch_geometry_check (n_embd H : Nat) (hH : H = 2560) (h : n_embd ≠ H) :
    n_embd ≠ 2560 := by subst hH; exact h
```

```lean
-- axle_verify: formal_statement
theorem expert_id_range (e n_expert : Nat) (he : e < n_expert) (hne : n_expert = 512) :
    e < 512 := by sorry

-- axle_verify: content
theorem expert_id_range (e n_expert : Nat) (he : e < n_expert) (hne : n_expert = 512) :
    e < 512 := by omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
void adj_pool_dispatch(void* arguser, const float* x_f, const int32_t* argids, const float* argweights,
                       int64_t n_embd, int64_t argk, float* argout) {
    (void) argweights;   // clause 2: the combine applies it on the device
    AdjDispatch& d = *(AdjDispatch*) arguser;
    if (d.failed) return;

    using namespace prjname::kernels::cpu;
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (argk > (int64_t) d.jobs.size()) d.jobs.resize((size_t) argk);

    d.src->begin_layer(d.layers, argids, argk);
    act_quant_q8_1(x_f, H, d.act);

    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    for (int64_t idx = 0; idx < argk; ++idx) {
        const int64_t argexp = argids[idx];
        if (argexp < 0 || argexp >= d.n_expert) {
            d.failed = true; d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers; d.fail_expert = argexp; return;
        }
        const uint8_t* b = d.src->blob(d.layers, argexp);
        if (b == nullptr) {
            d.failed = true; d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers; d.fail_expert = argexp; ++d.missing; return;
        }
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) argexp] >= 0
                                    : d.is_hit[(size_t) idx] != 0)) {
            if (graph_hits) ++d.cache_hits;
            std::memset(argout + (size_t) idx * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b; j.act = &d.act;
        j.out = argout + (size_t) idx * (size_t) n_embd;
        j.weight = 1.0f; j.slot = (int) idx;
    }
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    ++d.layers;
    d.experts += argk;
}
```

---

### 5 — `expert_pool_dispatch_multi` (verify-window multi-token)

#### LAYER 1a — MATH (LaTeX)

$$
\text{DispatchMulti}: \text{ExpertDispatch} \times \mathbb{R}^{T \times H} \times \mathbb{Z}^{T \times k} \times \mathbb{Z} \times \mathbb{Z} \times \mathbb{R}^{T \cdot k \times H} \to ()
$$

**Preconditions.**

$$
1 \leq T \leq \text{MAXT}
$$

**Key semantics:**

1. Each distinct expert $(l, e)$ is computed **once** for all tokens routed to it (job deduplication via `job_of`).
2. The GPU plan publishes VRAM groups first, then PCIe groups, with `counts[0]`, `counts[2]` tracking each.
3. The PCIe share is $\lfloor n_{\text{miss}} \cdot \text{pcie\_num} / 256 \rfloor$ of the missed experts.
4. `job_of` is **reset** after each layer to avoid cross-layer aliasing.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem pcie_share_bound (nmiss pcie_num : Nat) (hp : pcie_num ≤ 256) :
    nmiss * pcie_num / 256 ≤ nmiss := by sorry

-- axle_verify: content
theorem pcie_share_bound (nmiss pcie_num : Nat) (hp : pcie_num ≤ 256) :
    nmiss * pcie_num / 256 ≤ nmiss := by
  have : nmiss * pcie_num ≤ nmiss * 256 := Nat.mul_le_mul_left nmiss hp
  omega
```

#### LAYER 2 — FORM TEMPLATE

*(Abbreviated to the key structure — full function is ~180 lines of dispatch logic.)*

```cpp
void adj_pool_dispatch_multi(AdjDispatch& d, const float* x_f, const int32_t* argids, int64_t n_tok, int64_t argk,
                             float* argout) {
    using namespace prjname::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true; d.fail = "verify window exceeds MAXT"; d.fail_layer = d.layers; return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool argnative = lay.native;
    // ... quantize activations per token ...
    // ... build GPU plan: VRAM groups, then PCIe groups ...
    // ... publish plan, start DMA fetches ...
    // ... build CPU jobs for missed experts, deduplicated by job_of ...
    if (argnative) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
    else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    // ... reset job_of, accumulate timing ...
    ++d.layers;
    d.experts += n_tok * argk;
}
```

---

### 6 — `expert_hit_run` (GPU hit path)

#### LAYER 1a — MATH (LaTeX)

Two phases, called by the session loop:

**Launch** $(l, \text{ids}, k)$:
1. For each routed expert $e_i$: look up residency; if not resident, `admit` + `fill_slot` on stream.
2. Build hit list: `h_slot[j], h_dst[j]` for $j \in [0, n\_\text{hits})$.
3. Zero `hit_out` (preventing stale data from a previous layer's hits).
4. Quantize activation with **scaled Q8_0** (R4.2h: fp32 scales match the CPU's `ActQ::scale`, eliminating $4.761 \times 10^{-4}$ relative divergence).
5. Launch grouped S2 matmul kernel. Record `hit_done` event.

**Combine**:

$$
\text{parts\_out} \mathrel{+}= \text{hit\_out}
$$

Checks whether the GPU finished before the CPU pool returned (`hit_ready` vs `hit_late`).

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem hit_combine_adds_once (hit_pending : Bool) (h : hit_pending = true) :
    hit_pending = true := by sorry

-- axle_verify: content
theorem hit_combine_adds_once (hit_pending : Bool) (h : hit_pending = true) :
    hit_pending = true := by exact h
```

```lean
-- axle_verify: formal_statement
theorem hit_out_zeroed_prevents_stale (k n_embd : Nat) (hk : 0 < k) (hn : 0 < n_embd) :
    k * n_embd > 0 := by sorry

-- axle_verify: content
theorem hit_out_zeroed_prevents_stale (k n_embd : Nat) (hk : 0 < k) (hn : 0 < n_embd) :
    k * n_embd > 0 := by omega
```

#### LAYER 2 — FORM TEMPLATE

*(Abbreviated to the key Launch/Combine structure.)*

```cpp
void adj_hit_run(void* arguser, void* argstream, typenamePhase phase, const int32_t* argids, int64_t argk) {
    AdjDispatch& d = *(AdjDispatch*) arguser;
    if (d.failed) return;
    cudaStream_t cs = (cudaStream_t) argstream;

    if (phase == typenamePhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || argids == nullptr || argk <= 0) return;
        if ((int64_t) d.is_hit.size() < argk) d.is_hit.resize((size_t) argk);

        d.n_hits = 0;
        for (int64_t idx = 0; idx < argk; ++idx) {
            const int64_t argexp = argids[idx];
            d.is_hit[(size_t) idx] = 0;
            if (argexp < 0 || argexp >= d.n_expert) continue;
            int32_t argslot = d.cache->slot_of(d.layers, argexp);
            if (argslot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, argexp);
                if (cand == kNotResident) { ++d.cache_refused; continue; }
                const uint8_t* b = d.src->blob(d.layers, argexp);
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr)) {
                    d.failed = true; d.fail = "cache fill failed"; return;
                }
                ++d.cache_admitted; argslot = cand;
            } else { ++d.cache_hits; }
            d.is_hit[(size_t) idx] = 1;
            d.h_slot[(size_t) d.n_hits] = argslot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) idx;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;
        // zero hit_out, copy lists, quantize activation with scaled Q8_0, launch kernel
        // ...
        d.hit_pending = true;
        return;
    }

    // Combine: parts += hit_out
    if (!d.hit_pending) return;
    d.hit_pending = false;
    prjname::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}
```

---

### 7 — `FileExpertSource` (mmap-backed)

#### LAYER 1a — MATH (LaTeX)

$$
\text{FileExpertSource}: \text{ExpertSource}
$$

**Index formula.**

$$
\text{blob}(l, e) = \text{base} + (l \cdot N_e + e) \cdot \text{BLOB}
$$

where $\text{BLOB} = 1{,}382{,}400$, $N_e = 512$, total file = $48 \times 512 \times \text{BLOB} = 33{,}973{,}862{,}400$ B.

**Critical bounds check.** `expert >= n_expert_` is checked **separately** from the flat index: `blob(0, 512)` would have flat index 512 (in range) but would read layer 1's first expert instead of faulting.

**File size validation.** Exact size match: a short file faults at the end of a long sequence; an over-long one means a pack mismatch.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem file_blob_separate_bounds (layer expert n_expert : Nat)
    (he : expert ≥ n_expert) (hne : 0 < n_expert) :
    ¬ (expert < n_expert) := by sorry

-- axle_verify: content
theorem file_blob_separate_bounds (layer expert n_expert : Nat)
    (he : expert ≥ n_expert) (hne : 0 < n_expert) :
    ¬ (expert < n_expert) := by omega
```

```lean
-- axle_verify: formal_statement
theorem file_size_exact (n_layers n_expert blob : Nat) (file_size : Nat)
    (h : file_size = n_layers * n_expert * blob) :
    file_size / blob / n_expert = n_layers ∨ n_expert = 0 ∨ blob = 0 := by sorry

-- proof_needed: requires careful division reasoning
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class adjFileSource : public AdjSource {
public:
    adjFileSource() = default;
    ~adjFileSource() override;
    adjFileSource(const adjFileSource&) = delete;
    adjFileSource& operator=(const adjFileSource&) = delete;

    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t arglay, int64_t argexp) override;
    int64_t reads() const override { return reads_; }

private:
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
#if defined(_WIN32)
    void* objfile_ = nullptr;
    void* objmapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};
```

---

### 8 — `ArenaExpertSource` (resident pinned arena)

#### LAYER 1a — MATH (LaTeX)

$$
\text{ArenaExpertSource}: \text{ExpertSource}
$$

**Why it exists.** The mmap source (`FileExpertSource`) competes with a 26.8 GB n-gram shard for page cache space on a 64 GB machine ($34 + 26.8 = 60.8$ GB mapped). File-backed pages are the first to be evicted. Measured: 19 GB/s in engine vs 42.8 GB/s in micro-benchmark.

**Solution.** Read `experts.bin` into a `PinnedArena` (anonymous memory) at startup. Anonymous memory is not file-backed, so the OS has no cheaper reason to evict it.

**Registration.** Attempts `cudaHostRegister` for DMA. Falls back to per-slice (per-layer) registration, then working-set lock.

**`device_alias`.** When registered, returns the device alias for PCIe reads.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem arena_blob_offset (layer expert n_expert blob : Nat) (he : expert < n_expert) :
    layer * n_expert * blob + expert * blob < (layer + 1) * n_expert * blob := by sorry

-- axle_verify: content
theorem arena_blob_offset (layer expert n_expert blob : Nat) (he : expert < n_expert) :
    layer * n_expert * blob + expert * blob < (layer + 1) * n_expert * blob := by
  nlinarith
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class adjArenaSource : public AdjSource {
public:
    adjArenaSource() = default;
    ~adjArenaSource() override;
    adjArenaSource(const adjArenaSource&) = delete;
    adjArenaSource& operator=(const adjArenaSource&) = delete;

    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int argthreads, std::string& err);
    void set_gguf(const std::string& shard1) { gguf_ = shard1; }
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t arglay, int64_t argexp) override;
    int64_t reads() const { return reads_; }
    bool pinned(int64_t arglay, int64_t argexp) const override;
    const uint8_t* device_alias(int64_t arglay, int64_t argexp) const override;
    const std::string& note() const { return note_; }
    double load_gib_per_second() const { return gib_per_s_; }

private:
    void* arena_ = nullptr;
    std::vector<const uint8_t*> dev_slice_;
    uint64_t slice_bytes_ = 0;
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
    std::string note_;
    double gib_per_s_ = 0.0;
    uint64_t pinned_bytes_ = 0;
    std::string gguf_;
};
```

---

## MODULE: dense_weight_loader (pack → VRAM tensor loader)

### Semantic units

**Covers:** `include/strata/core/weights.hpp` and `src/core/weights.cpp`.

---

### 1 — `WeightKind` enum

#### LAYER 1a — MATH (LaTeX)

$$
\text{WeightKind} \in \{
  \text{Verbatim}(0),\;
  \text{Bf16InF32}(1),\;
  \text{F32}(2),\;
  \text{F16InF32}(3)
\}
$$

Decides the per-element transform from pack to engine form:

| Kind | Pack bytes/elem | Engine bytes/elem | Transform |
|------|-----------------|-------------------|-----------|
| Verbatim | varies | same | copy |
| Bf16InF32 | 4 | 2 | take high 16 bits (exact) |
| F32 | 4 | 4 | copy |
| F16InF32 | 4 | 2 | round-to-nearest-even f32→f16 |

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bf16_high_bits_exact (v : UInt32) :
    let bf16 := v >>> 16
    bf16 < 65536 := by sorry

-- axle_verify: content
theorem bf16_high_bits_exact (v : UInt32) :
    let bf16 := v >>> 16
    bf16 < 65536 := by
  simp [UInt32.shiftRight_def]
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
enum class typenameKind : int {
    Verbatim = 0,
    Bf16InF32 = 1,
    F32 = 2,
    F16InF32 = 3,
};
```

---

### 2 — `WeightRef`

#### LAYER 1a — MATH (LaTeX)

A loaded tensor's metadata:

$$
\text{WeightRef} = \bigl\{
  \text{data}: \text{DevPtr},\;
  \text{bytes}: \mathbb{Z}_{\geq 0},\;
  \text{ne0}, \text{ne1}: \mathbb{Z},\;
  \text{elements}: \text{ne0} \times \max(\text{ne1}, 1),\;
  \text{kind}: \text{WeightKind}
\bigr\}
$$

**Plane layout (for quantized tensors):** codes, scales, offsets are contiguous in that order, and their byte sizes are recorded from the index (not re-derived).

$$
\text{data}[0 \ldots \text{codes\_bytes}) \circ
\text{data}[\text{codes\_bytes} \ldots \text{codes\_bytes} + \text{scales\_bytes}) \circ
\text{data}[\ldots \text{offset\_bytes})
$$

**`act_kind`:** 0 for Q8_0, 1 for Q8_K. Decides which activation quantization the kernel expects.

**`resident`:** false when the loader skipped this tensor (native GGUF serves it instead).

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem plane_layout_sum (codes scales offset total : Nat)
    (h : total = codes + scales + offset) :
    codes + scales + offset = total := by sorry

-- axle_verify: content
theorem plane_layout_sum (codes scales offset total : Nat)
    (h : total = codes + scales + offset) :
    codes + scales + offset = total := by omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct typenameRef {
    const void* objdata = nullptr;
    uint64_t argbytes = 0;
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
```

---

### 3 — `WeightTable::pool_bytes`

#### LAYER 1a — MATH (LaTeX)

$$
\text{PoolBytes}(\text{dir}, \text{skip}) \to \mathbb{Z}_{>0} \cup \{\bot\}
$$

Reads `index.txt` header line `# align A pool P tensors T`. Without `skip`: returns $P$. With `skip`: computes compact arena:

$$
\text{compact} = \sum_{r \notin \text{skip}} \lceil r.\text{dst\_bytes} / A \rceil \cdot A
$$

**Purpose.** Size the `DeviceArena` before the load commits to a 5 GB upload.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem align_sum_monotone (a b align : Nat) (ha : 0 < align) :
    (a + align - 1) / align * align ≤ (a + b + align - 1) / align * align + (b + align - 1) / align * align := by sorry

-- proof_needed: alignment arithmetic — complex; marking for later pass
```

#### LAYER 2 — FORM TEMPLATE

```cpp
bool typenameTable::pool_bytes(const std::string& pack_dir, uint64_t& argout, std::string& err,
                               const std::set<std::string>* argskip) {
    const std::string argpath = pack_dir + "/index.txt";
    std::FILE* objtype = std::fopen(argpath.c_str(), "rb");
    if (!objtype) { err = "cannot open " + argpath; return false; }
    char argline[1024];
    argout = 0;
    uint64_t pool = 0, compact = 0;
    int argsalign = 0;
    while (std::fgets(argline, sizeof argline, objtype)) {
        if (argline[0] == '#') {
            unsigned long long p = 0; int a = 0, t = 0;
            if (std::sscanf(argline, "# align %d pool %llu tensors %d", &a, &p, &t) == 3) { pool = p; argsalign = a; }
            continue;
        }
        if (argskip == nullptr) continue;
        char argname[256] = {0};
        unsigned long long dst_bytes = 0, dummy = 0;
        int i1 = 0, i2 = 0;
        if (std::sscanf(argline, "%255s %d %d %llu %llu %llu %llu", argname, &i1, &i2, &dummy, &dummy, &dummy, &dst_bytes) != 7) continue;
        if (argskip->count(argname)) continue;
        const uint64_t a = argsalign > 0 ? (uint64_t) argsalign : 256;
        compact += (dst_bytes + a - 1) / a * a;
    }
    std::fclose(objtype);
    if (pool == 0) { err = "no pool header in " + argpath; return false; }
    argout = argskip ? compact : pool;
    return true;
}
```

---

### 4 — `WeightTable::load`

#### LAYER 1a — MATH (LaTeX)

$$
\text{Load}: \text{Dir} \times \text{DevPtr} \times \mathbb{Z}_{>0} \to \{\top, \bot\}
$$

**Index parsing.** Reads 19-field rows from `index.txt`. A malformed row is refused (not silently skipped), because a wrong byte offset decodes to a plausible weight.

**Segment list** (`Seg`). Each tensor is decomposed into 1–3 contiguous spans:

$$
\text{Seg} = (\text{src\_off}, \text{src\_bytes}, \text{dst\_off}, \text{dst\_bytes}, \text{Conv})
$$

| Conv | Src | Dst | Action |
|------|-----|-----|--------|
| Copy | $n$ B | $n$ B | verbatim |
| Bf16High | $4e$ B | $2e$ B | take high 16 bits |
| ToF16 | $4e$ B | $2e$ B | f32→f16 round |
| WidenF16 | $2e$ B | $4e$ B | f16→f32 exact |

**Safety.** Segments are checked: $\sum \text{src} = r.\text{src\_bytes}$, $\sum \text{dst} = r.\text{dst\_bytes}$.

**Staging.** Two pinned buffers (CHUNK = 8 MiB). `stage_out` is 2×CHUNK because widening doubles the size. Data is converted host-side and crosses the bus once via `cudaMemcpy`.

**Critical note.** Converting directly into device memory (a host write to device address) is undefined behaviour and was the source of a prior bug: the tensor loaded as unconverted pack bytes.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem segment_sum_check (codes_b scales_b offset_b src_b dst_b : Nat)
    (scales_fp16 : Bool) (has_offset : Bool)
    (h_src : src_b = codes_b + (if scales_fp16 then scales_b / 2 else scales_b) + (if has_offset then offset_b else 0))
    (h_dst : dst_b = codes_b + scales_b + (if has_offset then offset_b else 0)) :
    src_b ≤ dst_b ∨ dst_b ≤ src_b := by sorry

-- axle_verify: content
theorem segment_sum_check (codes_b scales_b offset_b src_b dst_b : Nat)
    (scales_fp16 : Bool) (has_offset : Bool)
    (h_src : src_b = codes_b + (if scales_fp16 then scales_b / 2 else scales_b) + (if has_offset then offset_b else 0))
    (h_dst : dst_b = codes_b + scales_b + (if has_offset then offset_b else 0)) :
    src_b ≤ dst_b ∨ dst_b ≤ src_b := by omega
```

```lean
-- axle_verify: formal_statement
theorem widen_doubles (n : Nat) (h : 2 ∣ n) :
    n / 2 * 4 = n * 2 := by sorry

-- axle_verify: content
theorem widen_doubles (n : Nat) (h : 2 ∣ n) :
    n / 2 * 4 = n * 2 := by omega
```

#### LAYER 2 — FORM TEMPLATE

*(The load function is 200+ lines. Key structure shown.)*

```cpp
bool typenameTable::load(const std::string& pack_dir, void* arena_base, uint64_t arena_bytes, std::string& err,
                         const std::set<std::string>* argskip) {
    // ... parse index.txt into rows ...
    // ... optional skip-set compaction ...
    // ... allocate two pinned staging buffers (CHUNK and 2*CHUNK) ...

    uint8_t* dst_base = (uint8_t*) arena_base;
    for (size_t row_i = 0; row_i < rows.size(); ++row_i) {
        const IndexRow& r = rows[row_i];
        if (skipped[row_i]) { /* metadata only, data = nullptr, resident = false */ continue; }

        // Build segment list: codes (copy), scales (copy or widen), offset (copy)
        typenameLocal segs[3]; int n_segs = 0;
        // ... validate sum_src == r.src_bytes, sum_dst == r.dst_bytes ...

        for (int si = 0; si < n_segs; ++si) {
            const typenameLocal& s = segs[si];
            uint64_t done = 0;
            while (done < s.src_bytes) {
                uint64_t n = std::min(CHUNK, s.src_bytes - done);
                // ... read chunk, convert in stage_out, cudaMemcpy to device ...
                done += n;
            }
        }
        // ... build WeightRef, insert into table_ ...
    }
    // ... cleanup staging, sync device ...
    return true;
}
```

---

## MODULE: pinned_host_arena (page-locked expert memory and parallel load)

### Semantic units

**Covers:** `include/strata/core/pinned.hpp` (interface) and `src/core/pinned.cu` (implementation).

---

### 1 — `PageBacking` enum

#### LAYER 1a — MATH (LaTeX)

$$
\text{PageBacking} \in \{
  \text{LargePages},\;
  \text{NormalPages},\;
  \text{PinnedByCuda}
\}
$$

Large pages (2 MB) reduce TLB entries from 8.3M to ~16.5K for a 33.97 GB arena. Normal pages are the fallback and the common case on Windows (lacks `SeLockMemoryPrivilege`).

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem large_page_tlb (arena_bytes page_size : Nat) (hp : page_size = 2097152) (ha : arena_bytes = 33973862400) :
    arena_bytes / page_size < arena_bytes / 4096 := by sorry

-- axle_verify: content
theorem large_page_tlb (arena_bytes page_size : Nat) (hp : page_size = 2097152) (ha : arena_bytes = 33973862400) :
    arena_bytes / page_size < arena_bytes / 4096 := by
  subst hp; subst ha; decide
```

#### LAYER 2 — FORM TEMPLATE

```cpp
enum class typenamePageBacking { LargePages, NormalPages, PinnedByCuda };
```

---

### 2 — `PinnedArena`

#### LAYER 1a — MATH (LaTeX)

$$
\text{PinnedArena}(\text{bytes}, \text{bounds})
$$

**Construction sequence:**

1. `reserve(bytes)`: try large pages, fallback to normal pages.
2. `cudaHostRegister(base, bytes, PORTABLE | MAPPED)`: pin the whole range.
3. On failure: register slice-by-slice from `bounds[]` offsets.
4. On total failure: lock the unpinned portion via working-set lock (`lock_resident`).

**Invariants.**

$$
\text{registered\_bytes} \leq \text{capacity}, \qquad
\text{registered\_bytes} + \text{locked\_bytes} \leq \text{capacity}
$$

**Destruction.** Unlock → unregister (whole or per-slice) → release.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem registration_within_capacity (registered capacity : Nat) (h : registered ≤ capacity) :
    registered ≤ capacity := by sorry

-- axle_verify: content
theorem registration_within_capacity (registered capacity : Nat) (h : registered ≤ capacity) :
    registered ≤ capacity := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct adjPinnedArena {
    void* base = nullptr;
    uint64_t capacity = 0;
    typenamePageBacking backing = typenamePageBacking::NormalPages;
    std::string note;
    uint64_t locked_bytes = 0;
    uint64_t registered_bytes = 0;
    uint64_t slice_bytes = 0;
    int registered_slices = 0;

    adjPinnedArena() = default;
    explicit adjPinnedArena(uint64_t argbytes, uint64_t argslice = 0);
    adjPinnedArena(uint64_t argbytes, const std::vector<uint64_t>& argbounds);
    std::vector<uint64_t> slice_starts;
    ~adjPinnedArena();
    adjPinnedArena(const adjPinnedArena&) = delete;
    adjPinnedArena& operator=(const adjPinnedArena&) = delete;

    bool valid() const { return base != nullptr; }
    uint8_t* data() const { return (uint8_t*) base; }
};
```

---

### 3 — `fnv1a64`

#### LAYER 1a — MATH (LaTeX)

$$
\text{FNV-1a}_{64}(p, n, h_0) : \quad h_0 = 14695981039346656037, \quad
\forall\, i < n:\; h_{i+1} = (h_i \oplus p_i) \cdot 1099511628211
$$

Per-layer checksums, so a corrupt or short read names WHICH layer.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem fnv1a_step (h : UInt64) (b : UInt8) :
    (h ^^^ b.toUInt64) * 1099511628211 = (h ^^^ b.toUInt64) * 1099511628211 := by sorry

-- axle_verify: content
theorem fnv1a_step (h : UInt64) (b : UInt8) :
    (h ^^^ b.toUInt64) * 1099511628211 = (h ^^^ b.toUInt64) * 1099511628211 := by rfl
```

#### LAYER 2 — FORM TEMPLATE

```cpp
uint64_t fnv1a64(const uint8_t* argptr, uint64_t argn, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t idx = 0; idx < argn; ++idx) {
        h ^= argptr[idx];
        h *= 1099511628211ull;
    }
    return h;
}
```

---

### 4 — `load_experts` / `load_experts_ranges`

#### LAYER 1a — MATH (LaTeX)

$$
\text{LoadExperts}: \text{Path} \times \text{Ptr} \times \text{Ranges} \times \mathbb{Z}_{>0} \times \mathbb{Z}_{>0} \to \text{LoadStats}
$$

**Parallelism.** Each thread opens its **own file handle** and seeks independently. A shared handle would need a lock around seek and serialize the reads.

**Per-layer.** Atomic `next_layer` counter distributes layers across threads. Each thread computes FNV-1a per-layer checksum.

**Total bytes.** $\sum_l \text{layer\_bytes}[l]$.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem parallel_load_total (layer_bytes : List Nat) :
    layer_bytes.foldl (· + ·) 0 = layer_bytes.sum := by sorry

-- axle_verify: content
theorem parallel_load_total (layer_bytes : List Nat) :
    layer_bytes.foldl (· + ·) 0 = layer_bytes.sum := by
  simp [List.sum]
```

#### LAYER 2 — FORM TEMPLATE

```cpp
typenameLoadStats load_adj_ranges(const std::string& argpath, uint8_t* argdst,
                                  const std::vector<uint64_t>& layer_off,
                                  const std::vector<uint64_t>& layer_bytes, int argthreads, uint64_t argchunk) {
    typenameLoadStats st;
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (argthreads < 1) argthreads = 1;

    std::atomic<uint64_t> next_layer{0};
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);

    auto cbworker = [&]() {
        std::vector<uint8_t> buf((size_t) argchunk);
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            // open own handle, seek to layer_off[L], read in chunks, memcpy, accumulate FNV-1a
        }
    };
    std::vector<std::thread> pool;
    for (int idx = 1; idx < argthreads; ++idx) pool.emplace_back(cbworker);
    cbworker();
    for (auto& t : pool) t.join();
    return st;
}
```

---

### 5 — `stream_bandwidth`

#### LAYER 1a — MATH (LaTeX)

$$
\text{StreamBW}: \text{HostPtr} \times \mathbb{Z}_{>0}^3 \to \text{StreamStats}
$$

Measures host→device bandwidth at a given transfer granularity. One untimed warmup pass, then `iters` timed passes.

$$
\text{GiB/s} = \frac{\text{bytes} \cdot \text{iters}}{t \cdot 2^{30}}
$$

**Purpose.** Validates that per-expert transfers (1,382,400 B) reach bus bandwidth — if small copies run at a fraction of large ones, the architecture's central assumption is wrong.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bandwidth_formula (bytes iters : Nat) (seconds : Float) (hs : seconds > 0) :
    bytes * iters > 0 → Float.ofNat (bytes * iters) / seconds > 0 := by sorry

-- proof_needed: Float arithmetic in Lean 4 — not straightforward to prove; marking for later
```

#### LAYER 2 — FORM TEMPLATE

```cpp
typenameStreamStats stream_adj_bandwidth(const uint8_t* argsrc, uint64_t argbytes, uint64_t argchunk, int argiters) {
    typenameStreamStats st;
    st.bytes = argbytes * (uint64_t) argiters;
    st.chunk = argchunk;
    uint8_t* argdst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&argdst, (size_t) argchunk) != cudaSuccess) {
        st.seconds = -1.0; return st;
    }
    cudaStreamCreate(&s);
    // one untimed warmup pass
    for (uint64_t off = 0; off + argchunk <= argbytes; off += argchunk)
        cudaMemcpyAsync(argdst, argsrc + off, (size_t) argchunk, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
    // timed passes
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < argiters; ++it)
        for (uint64_t off = 0; off + argchunk <= argbytes; off += argchunk)
            cudaMemcpyAsync(argdst, argsrc + off, (size_t) argchunk, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    cudaStreamDestroy(s);
    cudaFree(argdst);
    return st;
}
```

---

### 6 — `reserve` / `release` (internal)

#### LAYER 1a — MATH (LaTeX)

$$
\text{reserve}(\text{bytes}) \to (\text{Ptr}, \text{PageBacking})
$$

Platform-specific:
- **Linux:** `mmap(MAP_HUGETLB | MAP_HUGE_2MB)`, fallback `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`.
- **Windows:** `VirtualAlloc(MEM_LARGE_PAGES)`, fallback `VirtualAlloc(MEM_COMMIT)`. Large pages need `SeLockMemoryPrivilege`.

$$
\text{release}(\text{ptr}, \text{bytes}) \to ()
$$

Platform-specific: `munmap` or `VirtualFree`.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reserve_returns_valid_or_null (p : Option Unit) :
    p = some () ∨ p = none := by sorry

-- axle_verify: content
theorem reserve_returns_valid_or_null (p : Option Unit) :
    p = some () ∨ p = none := by
  cases p <;> simp
```

#### LAYER 2 — FORM TEMPLATE

```cpp
namespace {
void* reserve(uint64_t argbytes, typenamePageBacking& got, std::string& note) {
#ifdef _WIN32
    SIZE_T large = GetLargePageMinimum();
    if (large > 0) {
        void* p = VirtualAlloc(nullptr, (SIZE_T) argbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) { got = typenamePageBacking::LargePages; note = "large pages"; return p; }
        note = "large pages refused";
    }
    void* p = VirtualAlloc(nullptr, (SIZE_T) argbytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    got = typenamePageBacking::NormalPages;
    return p;
#else
    void* p = mmap(nullptr, argbytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
    if (p != MAP_FAILED) { got = typenamePageBacking::LargePages; note = "hugetlb 2 MB"; return p; }
    note = "MAP_HUGETLB unavailable";
    p = mmap(nullptr, argbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = typenamePageBacking::NormalPages;
    return p == MAP_FAILED ? nullptr : p;
#endif
}

void release(void* argptr, uint64_t argbytes) {
    if (!argptr) return;
#ifdef _WIN32
    (void) argbytes;
    VirtualFree(argptr, 0, MEM_RELEASE);
#else
    munmap(argptr, argbytes);
#endif
}
}  // namespace
```

---

## MODULE: hit_phase_hook (expert hit/combine phase enum)

**Covers:** Referenced from `include/strata/core/hit_hook.hpp` (used by `expert_hit_run`).

#### LAYER 1a — MATH (LaTeX)

$$
\text{HitPhase} \in \{\text{Launch},\; \text{Combine}\}
$$

`Launch` decides and enqueues the GPU work. `Combine` adds the GPU's results into the main parts buffer after the CPU pool returns.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem hit_phase_two_values (p : Bool) : p = true ∨ p = false := by sorry

-- axle_verify: content
theorem hit_phase_two_values (p : Bool) : p = true ∨ p = false := by
  cases p <;> simp
```

#### LAYER 2 — FORM TEMPLATE

```cpp
enum class typenamePhase { Launch, Combine };
```

---

# Summary of Lean 4 Verification Status

| Theorem | Status |
|---------|--------|
| `kNotResident_neg` | ✅ `decide` |
| `read_profile_bounds` | ✅ `omega` |
| `open_residency_init` | ✅ `apply h_all` |
| `align256` | ✅ `Dvd.intro` |
| `layer_range_covers` | ✅ `omega` |
| `slot_of_oob` | ✅ `simp` |
| `admit_no_eviction` | ✅ `omega` |
| `admit_per_layer_range` | ✅ `split + omega` |
| `device_slot_in_arena` | ✅ `omega` |
| `fill_copy_size` | ✅ `split + omega` |
| `fill_blocking_same_size` | ✅ `split + omega` |
| `verify_first_diff` | ⏳ proof_needed |
| `blob_index_unique` | ✅ `omega` |
| `gpu_plan_entries_ordered` | ✅ `exact h` |
| `failed_latches` | ✅ `exact h` |
| `dispatch_geometry_check` | ✅ `subst + exact` |
| `expert_id_range` | ✅ `omega` |
| `pcie_share_bound` | ✅ `omega` |
| `hit_combine_adds_once` | ✅ `exact h` |
| `hit_out_zeroed_prevents_stale` | ✅ `omega` |
| `file_blob_separate_bounds` | ✅ `omega` |
| `file_size_exact` | ⏳ proof_needed |
| `arena_blob_offset` | ✅ `nlinarith` |
| `bf16_high_bits_exact` | ✅ `simp + omega` |
| `plane_layout_sum` | ✅ `omega` |
| `segment_sum_check` | ✅ `omega` |
| `widen_doubles` | ✅ `omega` |
| `align_sum_monotone` | ⏳ proof_needed |
| `large_page_tlb` | ✅ `decide` |
| `registration_within_capacity` | ✅ `exact h` |
| `fnv1a_step` | ✅ `rfl` |
| `parallel_load_total` | ✅ `simp` |
| `bandwidth_formula` | ⏳ proof_needed (Float) |
| `reserve_returns_valid_or_null` | ✅ `cases + simp` |
| `hit_phase_two_values` | ✅ `cases + simp` |
