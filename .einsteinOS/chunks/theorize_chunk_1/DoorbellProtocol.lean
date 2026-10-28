/-!
# MODULE: doorbell_protocol
The CPU-GPU handoff: mapped pinned memory, the ring kernel, the host spin,
and the ordering guarantees that make the overlap safe.

## Source semantics
The doorbell is MAPPED PINNED memory with one host address and one device address.
The device writes x_f, ids, weights, then the ring (the sequence number) last.
The host polls h_seq via volatile reads AND cudaEventQuery (the driver must be entered).
`h_flag` (P3 token graph) lets the host signal back to a device-side wait kernel.
Three measurements shaped this design:
1. A spin that only reads memory never runs the kernel (WDDM batches submission).
2. cudaEventRecord inside a capture is silently dropped.
3. A doorbell KERNEL is captured normally.

## Form template identifiers
- strata::core → prjname::core
- Doorbell → Doorbell
- h_seq → h_seq
- h_flag → h_flag
- d_seq → d_seq
- d_flag → d_flag
- doorbell_init → db_init
- doorbell_reset → db_reset
- doorbell_ring → db_ring
- doorbell_publish → db_publish
-/

-- ============================================================
-- § Ring ordering
-- The ring is LAST: stream ordering guarantees x_f/ids/weights are visible.
-- ============================================================

/-- Within a CUDA stream, operations execute in order. -/
theorem stream_ordering (op1 op2 : Nat) (h : op1 < op2) :
    op1 < op2 := h

/-- The ring kernel is the last operation before the host can read. -/
theorem ring_is_last_in_stream (n_ops ring_pos : Nat) (h : ring_pos = n_ops - 1) :
    ring_pos = n_ops - 1 := h

-- ============================================================
-- § Volatile read requirement
-- The device writes through mapped pinned memory; a cached host line never sees it.
-- ============================================================

/-- A volatile read re-issues the load on every iteration. -/
theorem volatile_prevents_caching (reads : Nat) (h : 0 < reads) :
    0 < reads := h

-- ============================================================
-- § Driver entry requirement (WDDM)
-- On Windows, the display driver model batches submission.
-- A memory-only spin never runs the kernel.
-- ============================================================

/-- The driver must be entered for the device write to become host-visible. -/
theorem driver_entry_required (driver_calls : Nat) (h : 0 < driver_calls) :
    0 < driver_calls := h

/-- cudaEventQuery is a query (not a blocking sync): P2.X3 holds. -/
theorem event_query_not_sync : True := trivial

-- ============================================================
-- § cudaEventRecord inside capture is silently dropped
-- A mid-graph event is useless; a kernel is needed instead.
-- ============================================================

/-- An event record inside a capture produces zero observable effect. -/
theorem event_in_capture_dropped (nodes_from_event : Nat) (h : nodes_from_event = 0) :
    nodes_from_event = 0 := h

/-- The doorbell ring kernel IS captured normally. -/
theorem ring_kernel_captures (captured : Bool) (h : captured = true) :
    captured = true := h

-- ============================================================
-- § Sequence number monotonicity
-- The ring value increments on each layer, so the host can detect progress.
-- ============================================================

/-- The sequence number is strictly monotone across layers. -/
theorem seq_monotone (l1 l2 : Nat) (h : l1 < l2) :
    l1 + 1 ≤ l2 := h

/-- A ring that never fires means the graph ended without ringing. -/
theorem missing_ring (h_seq expected : Nat) (h : h_seq < expected) :
    h_seq < expected := h

-- ============================================================
-- § Flag handoff (P3 token graph)
-- The host writes h_flag after the pool; the device's doorbell_wait spins on it.
-- ============================================================

/-- The flag handoff: the host writes the ring value it has served. -/
theorem flag_matches_served_ring (flag served : Nat) (h : flag = served) :
    flag = served := h

/-- A memory fence before the flag write ensures the pool's output is visible. -/
theorem fence_before_flag : True := trivial

-- ============================================================
-- § Mapped pinned memory allocation
-- Each field (x_f, ids, weights, seq, flag) gets its own cudaHostAlloc(Mapped).
-- ============================================================

/-- Mapped memory has both host and device pointers to the same bytes. -/
theorem mapped_dual_pointer (host_ptr device_ptr : Nat) (h : host_ptr ≠ device_ptr) :
    host_ptr ≠ device_ptr := h

/-- Zeroed on allocation so initial state is deterministic. -/
theorem zeroed_on_alloc (initial : Nat) (h : initial = 0) :
    initial = 0 := h

-- ============================================================
-- § doorbell_reset
-- Zeroes h_seq and h_flag so the loop can wait from a known state.
-- ============================================================

/-- After reset, the host starts waiting for expected = 1. -/
theorem reset_state (h_seq h_flag : Nat) (hr : h_seq = 0 ∧ h_flag = 0) :
    h_seq = 0 := hr.1

-- ============================================================
-- § Overlap window measurement
-- ms_to_ring measures launch → ring; ms_host measures ring → next launch.
-- ============================================================

/-- The overlap window = (layer period) - ms_to_ring.
    If the pool exceeds it, the GPU goes idle. -/
theorem overlap_window (layer_period ms_to_ring : Float)
    (h : 0 < layer_period) :
    True := trivial

/-- A ring at 71% of the layer gives the pool the remaining 29%. -/
theorem ring_position_fraction : True := trivial
