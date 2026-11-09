#!/usr/bin/env python3
"""Self-contained: fetch Niko1221/Strata commits since a date, dedupe against STRATA_COMMITS_FULL.txt,
fetch per-commit file lists, print Phase-1-relevant ones. Needs env GH_TOKEN. Run from repo root.
Usage: GH_TOKEN=... python3 tools/strata_delta.py 2026-11-02"""
import json, os, re, sys, urllib.request
since = sys.argv[1] if len(sys.argv) > 1 else "2026-11-02"
tok = os.environ["GH_TOKEN"]
def api(path):
    r = urllib.request.Request("https://api.github.com" + path, headers={"Authorization": "Bearer " + tok, "Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(r, timeout=60))
log = "STRATA_COMMITS_FULL.txt"
old = {l.split("|")[0] for l in open(log)} if os.path.exists(log) else set()
new, p = [], 1
while True:
    page = api(f"/repos/Niko1221/Strata/commits?since={since}T00:00:00Z&per_page=100&page={p}")
    if not page: break
    for c in page:
        if c["sha"] not in old and all(c["sha"] != n[0] for n in new):
            who = (c.get("author") or {}).get("login") or c["commit"]["author"]["name"]
            new.append((c["sha"], c["commit"]["author"]["date"][:10], who, c["commit"]["message"].split("\n")[0].replace("|", "/")))
    p += 1
print("new commits:", len(new))
pat = re.compile(r"(native_mmvq|native_bf16|fused_gr|gdn|shared_expert|moe_combine|router|topk|ix_compat|ix_q81|expert_source|src/kernels/cuda|src/prefill|src/split)", re.I)
for sha, d, who, msg in new:
    fl = [f["filename"] for f in api(f"/repos/Niko1221/Strata/commits/{sha}").get("files", []) if pat.search(f["filename"])]
    if fl: print(sha[:8], d, msg[:90], "|", ", ".join(fl[:3]))
if "--write" in sys.argv and new:
    prev = open(log).read() if os.path.exists(log) else ""
    open(log, "w").write("".join("|".join(n) + "\n" for n in new) + prev)
