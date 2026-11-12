#!/usr/bin/env python3
"""Task: "产品经理会把一个个feature列出来表格，工程师一个一个实现。但我们现在的issues都只是猜测的指标。"

The PM table for baloon, generated from measurements only. One row per Strata CUDA source that the engine
actually compiles, across all three paths (the v17 table saw the decode path alone):

  decode   src/kernels/cuda/*.cu      per-token kernels (the 49 files of the earlier table)
  prompt   src/prefill/*.cu,*.cuh     prefill / prompt-phase kernels
  engine   src/split/src/*.cu,*.cuh   multi-GPU engine kernels

Every number is read from git or from the source text, and every row carries the evidence SHAs; no number is
an estimate. The blocker columns are the ENV.md "Known limitations (ivcore10)" turned into greppable facts,
so a row says what an engineer must do to port it, not how fast it might be:

  wave64      STRATA_HIP_GFX906 present -> Strata already keeps a 64-lane layout; port it verbatim
  w32 sync    __shfl*_sync / __ballot_sync / __any|__all_sync / __activemask  (wave64 semantics review each)
  lane32      0xffffffff full-warp masks and & 31 / >> 5 / % 32 / / 32 lane arithmetic (hard-coded 32 lanes)
  warpSize    uses of warpSize (portable as written; counted so a row is not scored for them)
  tc          wmma:: / mma.sync / ldmatrix        (no tensor core on BI-V100 -> needs a scalar path)
  dynsmem     cudaFuncAttributeMaxDynamicSharedMemorySize opt-in (ivcore10 smem limit must be re-checked)
  tfs         __threadfence_system (NOT supported on ivcore10 -> __threadfence() or a host barrier)
  ipc         cudaIpc* (NOT supported on BI-V100 -> cudaMemcpyPeer)
  fp64        uses of double (BI-V100 fp64 rate; reductions kept in double are a cost, not a blocker)

A row's kind is measured too, because cost alone ranks the wrong work first: "kernel" has at least one
__global__ and is not a parity/test/bench/harness file by name; "harness" is a parity/test/bench/harness
file (Strata's own correctness programs — read them, do not port them); "host" has no __global__ at all
(host or template-only source). The backlog lists kernel rows only.

port cost = 3*(tc>0) + 2*(tfs>0) + 2*(ipc>0) + w32sync + lane32, minus 4 when a wave64 path already exists,
floored at 0 -- a count of the places a human must touch, stated so the ordering can be argued with.
The backlog section lists the unported rows cheapest-first, which is the order engineers take them.

Usage: python3 tools/feature_table.py [--strata ../Strata] [--fetch] [-o STRATA_FEATURE_TABLE_GEN.md]
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_check  # noqa: E402
import strata_repo  # noqa: E402

ROOT = ref_check.ROOT
GROUPS = [("decode", "src/kernels/cuda", (".cu",)),
          ("prompt", "src/prefill", (".cu", ".cuh")),
          ("engine", "src/split/src", (".cu", ".cuh"))]
PATS = dict(
    w32sync=re.compile(r"__shfl(?:_xor|_down|_up)?_sync|__ballot_sync|__activemask|__any_sync|__all_sync"),
    lane32=re.compile(r"0xffffffffu?\b|&\s*31\b|>>\s*5\b|%\s*32\b|/\s*32\b"),
    warpsize=re.compile(r"\bwarpSize\b"),
    tc=re.compile(r"\bwmma::|\bmma\.sync|\bldmatrix\b|\bmma_sync\b"),
    dynsmem=re.compile(r"MaxDynamicSharedMemorySize"),
    tfs=re.compile(r"__threadfence_system"),
    ipc=re.compile(r"\bcudaIpc\w*"),
    fp64=re.compile(r"\bdouble\b"),
    kernels=re.compile(r"__global__"))
WAVE64 = "STRATA_HIP_GFX906"
HARNESS = re.compile(r"(parity|_test|test_|bench|harness)", re.I)


def cost(r):
    c = 3 * (r["tc"] > 0) + 2 * (r["tfs"] > 0) + 2 * (r["ipc"] > 0) + r["w32sync"] + r["lane32"]
    return max(0, c - 4 if r["wave64"] else c)


def main():
    strata = strata_repo.ensure()
    out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else os.path.join(ROOT, "STRATA_FEATURE_TABLE_GEN.md")
    _, head = ref_check.git(strata, "log", "-1", "--format=%h %ad", "--date=short")
    _, total = ref_check.git(strata, "rev-list", "--count", "HEAD")

    ports = {}
    for f in sorted(os.listdir(os.path.join(ROOT, "src"))):
        if f.startswith("ix_"):
            for r in ref_check.check(strata, os.path.join(ROOT, "src", f)):
                ports.setdefault(r["strata"], []).append(f"{r['port'].split('/')[-1]} {r['status']}")

    rows = []
    for group, d, exts in GROUPS:
        base = os.path.join(strata, d)
        for f in sorted(os.listdir(base)):
            if not f.endswith(exts):
                continue
            rel = f"{d}/{f}"
            txt = open(os.path.join(base, f), errors="replace").read()
            _, log = ref_check.git(strata, "log", "--format=%h %ad", "--date=short", "--", rel)
            log = log.strip().split("\n") if log.strip() else []
            r = dict(group=group, file=f, rel=rel, commits=len(log), last=log[0] if log else "-",
                     first=log[-1] if log else "-", loc=txt.count("\n"), wave64=WAVE64 in txt,
                     port="; ".join(ports.get(rel, [])) or "-")
            for k, p in PATS.items():
                r[k] = len(p.findall(txt))
            r["kind"] = "harness" if HARNESS.search(f) else ("kernel" if r["kernels"] else "host")
            r["cost"] = cost(r)
            rows.append(r)
    rows.sort(key=lambda r: (r["port"] == "-", r["kind"] != "kernel", r["cost"], -r["commits"], r["rel"]))

    hdr = ("| # | path | group | kind | commits | first | last | loc | `__global__` | wave64 | w32 sync | lane32 | "
           "warpSize | tc | dynsmem | tfs | ipc | fp64 | port cost | baloon port |")
    L = ["# Strata kernel feature table (generated by tools/feature_table.py — do not edit by hand)", "",
         f"Strata HEAD {head.strip()}, {total.strip()} commits. Every value is measured from git or the source text "
         "at that HEAD; the SHAs are the evidence. Columns and the cost formula are documented in the script's "
         "docstring. Re-run to refresh.", "",
         "Covered: " + ", ".join(f"`{d}/` ({g})" for g, d, _ in GROUPS) + ".", "",
         hdr, "|" + "---|" * 20]
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | `{r['rel']}` | {r['group']} | {r['kind']} | {r['commits']} | {r['first']} | {r['last']} | {r['loc']} | "
                 f"{r['kernels']} | {'yes' if r['wave64'] else 'no'} | {r['w32sync']} | {r['lane32']} | "
                 f"{r['warpsize']} | {r['tc']} | {r['dynsmem']} | {r['tfs']} | {r['ipc']} | {r['fp64']} | "
                 f"{r['cost']} | {r['port']} |")

    todo = [r for r in rows if r["port"] == "-" and r["kind"] == "kernel"]
    L += ["", "## Backlog — unported rows, cheapest first", "",
          "Kernel rows only (harness and host rows carry no kernel to port). Ordered by measured port cost, then by "
          "upstream churn (commits), which is how much Strata itself invests in the file. One row = one task.", "",
          "| rank | path | cost | wave64 | w32 sync | lane32 | tc | tfs | commits | last touched upstream |",
          "|" + "---|" * 10]
    for i, r in enumerate(todo[:25], 1):
        L.append(f"| {i} | `{r['rel']}` | {r['cost']} | {'yes' if r['wave64'] else 'no'} | {r['w32sync']} | "
                 f"{r['lane32']} | {r['tc']} | {r['tfs']} | {r['commits']} | {r['last']} |")

    byg = {g: [r for r in rows if r["group"] == g] for g, _, _ in GROUPS}
    L += ["", "## Totals", "",
          "| group | files | kernel | harness | host | ported | wave64 path | needs tensor core | `__threadfence_system` | portable kernels (cost 0) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g, _, _ in GROUPS:
        v = byg[g]
        L.append(f"| {g} | {len(v)} | {sum(r['kind'] == 'kernel' for r in v)} | {sum(r['kind'] == 'harness' for r in v)} | "
                 f"{sum(r['kind'] == 'host' for r in v)} | {sum(r['port'] != '-' for r in v)} | {sum(r['wave64'] for r in v)} | "
                 f"{sum(r['tc'] > 0 for r in v)} | {sum(r['tfs'] > 0 for r in v)} | "
                 f"{sum(r['cost'] == 0 and r['port'] == '-' and r['kind'] == 'kernel' for r in v)} |")
    L += ["", f"{len(rows)} files ({sum(r['kind'] == 'kernel' for r in rows)} carry kernels), "
          f"{sum(r['port'] != '-' for r in rows)} with a baloon port, {len(todo)} kernel files unported, "
          f"{sum(r['cost'] == 0 for r in todo)} of them with no measured blocker."]
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:10]))
    print(f"... wrote {out} ({len(rows)} rows, {len(todo)} unported)")


if __name__ == "__main__":
    main()
