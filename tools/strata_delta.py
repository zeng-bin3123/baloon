#!/usr/bin/env python3
"""Collect ALL Niko1221/Strata commits, then scan them slowly (resumable), under the GitHub rate limit.

Task sentence -> code:
  "收集github Strata 仓库的所有commit(先在你的code exec环境遍历, 然后慢慢扫。)"
      step 1 `list`: traverse every commit on the branch (or --since DATE) -> STRATA_COMMITS_FULL.txt
      step 2 `scan`: fetch per-commit file lists a few at a time, resumable via tools/strata_scan_state.json
  "api限流就用这个 ...token"
      token is read from env GH_TOKEN (never written to the repo); every request checks
      X-RateLimit-Remaining / Retry-After and sleeps instead of failing on 403/429.

Usage (from repo root):
  GH_TOKEN=... python3 tools/strata_delta.py list [--since 2026-11-09] [--write]
  GH_TOKEN=... python3 tools/strata_delta.py scan [--batch 200]
  python3 tools/strata_delta.py report            # offline: Phase-1-relevant commits from scan state
"""
import json, os, re, sys, time, urllib.error, urllib.request

REPO = "Niko1221/Strata"
LOG = "STRATA_COMMITS_FULL.txt"
STATE = "tools/strata_scan_state.json"
PAT = re.compile(r"(native_mmvq|native_bf16|fused_gr|gdn|shared_expert|moe_combine|router|topk|ix_compat|ix_q81|"
                 r"expert_source|src/kernels/cuda|src/prefill|src/split)", re.I)
MIN_REMAINING = 50  # keep a reserve so other tools sharing the token still work


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def api(path, tries=8):
    tok = os.environ.get("GH_TOKEN")
    h = {"Accept": "application/vnd.github+json", "User-Agent": "baloon-strata-delta"}
    if tok:
        h["Authorization"] = "Bearer " + tok
    for i in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request("https://api.github.com" + path, headers=h), timeout=60)
            rem, reset = r.headers.get("X-RateLimit-Remaining"), r.headers.get("X-RateLimit-Reset")
            body = json.load(r)
            if rem is not None and int(rem) < MIN_REMAINING and reset:
                wait = max(0, int(reset) - int(time.time())) + 5
                print(f"[ratelimit] remaining={rem}, sleeping {wait}s until reset", file=sys.stderr)
                time.sleep(wait)
            return body
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                ra, reset = e.headers.get("Retry-After"), e.headers.get("X-RateLimit-Reset")
                wait = int(ra) if ra else (max(0, int(reset) - int(time.time())) + 5 if reset else 60 * (i + 1))
                print(f"[ratelimit] HTTP {e.code}, sleeping {wait}s (try {i + 1}/{tries})", file=sys.stderr)
                time.sleep(wait)
            elif e.code >= 500:
                time.sleep(5 * (i + 1))
            else:
                raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (i + 1))
    raise SystemExit(f"giving up on {path}")


def read_log():
    if not os.path.exists(LOG):
        return []
    return [l.rstrip("\n").split("|", 3) for l in open(LOG) if l.strip()]


def cmd_list():
    """Step 1: traverse all commits (newest first) until we reach ones already in the log."""
    since = arg("--since")
    old = {r[0] for r in read_log()}
    new, seen, p = [], set(), 1
    q = f"&since={since}T00:00:00Z" if since else ""
    while True:
        page = api(f"/repos/{REPO}/commits?sha={arg('--branch', 'main')}&per_page=100&page={p}{q}")
        if not page:
            break
        for c in page:
            if c["sha"] in old or c["sha"] in seen:
                continue
            seen.add(c["sha"])
            who = (c.get("author") or {}).get("login") or c["commit"]["author"]["name"]
            msg = c["commit"]["message"].split("\n")[0].replace("|", "/")
            new.append((c["sha"], c["commit"]["author"]["date"][:10], who, msg))
        # incremental run: stop at the first page that is entirely in the log; --full walks to the root
        if "--full" not in sys.argv and all(c["sha"] in old for c in page):
            break
        p += 1
    print(f"new commits: {len(new)} (log had {len(old)})")
    for sha, d, who, msg in new[:40]:
        print(sha[:8], d, who, msg[:100])
    if "--write" in sys.argv and new:
        prev = open(LOG).read() if os.path.exists(LOG) else ""
        open(LOG, "w").write("".join("|".join(n) + "\n" for n in new) + prev)
        print(f"wrote {LOG}: {len(new) + len(old)} commits")


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"files": {}}


def cmd_scan():
    """Step 2: slowly fetch file lists for commits not yet scanned; safe to stop and rerun."""
    st, batch = load_state(), int(arg("--batch", "200"))
    todo = [r for r in read_log() if r[0] not in st["files"]]
    print(f"to scan: {len(todo)}, this run: {min(batch, len(todo))}")
    for n, (sha, *_rest) in enumerate(todo[:batch], 1):
        st["files"][sha] = [f["filename"] for f in api(f"/repos/{REPO}/commits/{sha}").get("files", [])]
        if n % 20 == 0:  # checkpoint so an interrupted scan loses at most 20 requests
            json.dump(st, open(STATE, "w"), separators=(",", ":"))
    json.dump(st, open(STATE, "w"), separators=(",", ":"))
    print(f"scanned {len(st['files'])}/{len(read_log())}")
    cmd_report(st)


def cmd_report(st=None):
    st = st or load_state()
    for sha, d, who, msg in read_log():
        fl = [f for f in st["files"].get(sha, []) if PAT.search(f)]
        if fl:
            print(sha[:8], d, msg[:90], "|", ", ".join(fl[:3]))


if __name__ == "__main__":
    {"list": cmd_list, "scan": cmd_scan, "report": cmd_report}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
