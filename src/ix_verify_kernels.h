// VerifyKernels index/semantics helpers (multi-token verify window).
#pragma once
#include "ix_u4_common.h"
#define IX_VERIFY_MAX_T 16
IX4_HD int verify_conv_src(int t, int j) { return t + j; }          // index into [hist(3) | x0..]
IX4_HD int verify_commit_src(int n, int j) { return n + j; }        // keeps last three of [hist | x0..x_{n-1}]
IX4_HD int verify_wstart(int n_kv, int window) { return n_kv > window ? n_kv - window : 0; }
IX4_HD int verify_wwidth(int n_kv, int window) { return n_kv - verify_wstart(n_kv, window); }
struct VerifyDense { int n_kv, n_bid, width; };
IX4_HD VerifyDense verify_dense_step(int pos) { VerifyDense s; s.n_kv = pos + 1; s.n_bid = (pos + 1) / 4; s.width = pos + 1; return s; }
IX4_HD int verify_bcast_src(int i, int n) { return i % n; }
IX4_HD int verify_bcast_stream(int i, int n) { return i / n; }
IX4_HD long verify_rebase(long base, long bytes, int k) { return base + (long)k * bytes; }
IX4_HD int verify_ab_grid(int hv) { return (2 * hv + 7) / 8; }
