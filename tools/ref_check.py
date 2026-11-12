#!/usr/bin/env python3
"""Task: "我们要开发的是 ilu 级别的底层代码，所以直接参考 Niko1221/Strata"
Every src/ix_* port names its Strata source in its header ("Strata <path> (HEAD <sha>, L<a>-<b>)").
This checks those references against a local Strata clone:
  BROKEN  sha unknown, or path missing at that sha            -> exit 1
  UNPINNED  header names a Strata path but no sha: every upstream commit to it is listed (warning only)
  STALE   Strata changed the file after the port sha; if the header gives line ranges, only diff hunks that
          touch those lines count (the ported region itself changed)
  OK      unchanged upstream since the port
Usage: python3 tools/ref_check.py [--strata ../Strata] [--json]
"""
import json, os, re, subprocess, sys

def _require_checkout(*siblings):
    """Task: "项目在 zeng-bin3123/baloon，分支是 main". Run outside a baloon clone (the BI-V100 machine had a bare
    /home/claude/baloon with copied files, no .git, no sibling tools/*.py) this script used to die on an import
    traceback; now it names what is missing and the one line that fixes it. Exit 4 = not a baloon checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    missing = [f"tools/{m}.py" for m in siblings if not os.path.exists(os.path.join(here, m + ".py"))]
    if not os.path.isdir(os.path.join(root, ".git")):
        missing.append(".git (not a git clone)")
    if missing:
        sys.stderr.write(f"{root} is not a zeng-bin3123/baloon main checkout; missing: {', '.join(missing)}\n"
                         "fix (keeps old files in <dir>.pre-bootstrap-<time>, then runs preflight):\n"
                         "  curl -fsSL https://raw.githubusercontent.com/zeng-bin3123/baloon/main/bootstrap.sh"
                         f" | bash -s -- {root}\n")
        sys.exit(4)


_require_checkout('strata_repo')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_repo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_RE = re.compile(r"Strata ((?:src|include|sycl)/[\w/.\-]+\.(?:cu|cuh|cpp|hpp|h))")
SHA_RE = re.compile(r"(?:HEAD |Strata )([0-9a-f]{7,40})\b")
LINES_RE = re.compile(r"\bL(\d+)(?:-(\d+))?\b")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? ", re.M)


def git(strata, *a):
    r = subprocess.run(["git", "-C", strata, *a], capture_output=True, text=True)
    return r.returncode, r.stdout


def refs(path, head_lines=12):
    """(strata_path, sha, [(lo,hi)...]) for each Strata file named in the port's header."""
    head = "".join(open(path, errors="replace").readlines()[:head_lines])
    paths = PATH_RE.findall(head)
    m = SHA_RE.search(head)
    sha = m.group(1) if m else None
    out = []
    for p in dict.fromkeys(paths):
        # line ranges written on the same header line as this path (or the line right after it)
        seg = head[head.index(p):].split("\n")
        ranges = [(int(a), int(b or a)) for a, b in LINES_RE.findall("\n".join(seg[:2]))]
        out.append((p, sha, ranges))
    return out


def check(strata, port):
    rows = []
    for p, sha, ranges in refs(port):
        row = {"port": os.path.relpath(port, ROOT), "strata": p, "sha": sha, "lines": ranges}
        if not sha:
            _, log = git(strata, "log", "--format=%h|%ad|%s", "--date=short", "-5", "--", p)
            rows.append({**row, "status": "UNPINNED" if os.path.exists(os.path.join(strata, p)) else "BROKEN",
                         "commits": [l.split("|", 2) for l in log.splitlines() if l],
                         "why": "no Strata sha in header; latest upstream commits to the file listed"}); continue
        if git(strata, "cat-file", "-e", f"{sha}^{{commit}}")[0]:
            rows.append({**row, "status": "BROKEN", "why": f"sha {sha} not in Strata"}); continue
        if git(strata, "cat-file", "-e", f"{sha}:{p}")[0]:
            rows.append({**row, "status": "BROKEN", "why": f"{p} missing at {sha}"}); continue
        _, log = git(strata, "log", "--format=%h|%ad|%s", "--date=short", f"{sha}..HEAD", "--", p)
        commits = [l.split("|", 2) for l in log.splitlines() if l]
        hit = commits
        ranges = [(lo, hi) for lo, hi in ranges if hi > lo]  # a bare "L146" is a start anchor, cannot bound a region
        if commits and ranges:
            _, diff = git(strata, "diff", "-U0", sha, "HEAD", "--", p)
            touched = [(int(a), int(a) + max(int(n or 1), 1) - 1) for a, n in HUNK_RE.findall(diff)]
            if not any(lo <= hb and ha <= hi for lo, hi in ranges for ha, hb in touched):
                hit = []  # file changed, ported lines did not
        row.update(commits=commits, status="STALE" if hit else "OK",
                   why=(f"{len(commits)} upstream commits" + ("" if hit or not commits else ", none in ported lines")))
        rows.append(row)
    if not rows:  # commit-only reference, e.g. "Strata 7d1016f3, #3948"
        head = "".join(open(port, errors="replace").readlines()[:12])
        m = re.search(r"Strata ([0-9a-f]{7,40})\b", head)
        if m:
            bad = git(strata, "cat-file", "-e", f"{m.group(1)}^{{commit}}")[0]
            _, subj = git(strata, "log", "-1", "--format=%s", m.group(1))
            rows.append({"port": os.path.relpath(port, ROOT), "strata": "(commit)", "sha": m.group(1), "lines": [],
                         "status": "BROKEN" if bad else "OK", "why": "commit not in Strata" if bad else subj.strip()[:80]})
    return rows


def main():
    strata = strata_repo.ensure()  # finds or clones the full Strata history (exit 3 if impossible)
    ports = sorted(os.path.join(ROOT, "src", f) for f in os.listdir(os.path.join(ROOT, "src")) if f.startswith("ix_"))
    rows = [r for p in ports for r in check(strata, p)]
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1)); return 1 if any(r["status"] == "BROKEN" for r in rows) else 0
    _, head = git(strata, "log", "-1", "--format=%h %ad", "--date=short")
    print(f"Strata HEAD {head.strip()}")
    for r in rows:
        lines = ",".join(f"L{a}-{b}" for a, b in r["lines"]) or "-"
        print(f"{r['status']:6} {r['port']:28} <- {r['strata']} @{r['sha']} {lines}  ({r['why']})")
        for c in (r.get("commits") or [])[:5]:
            print(f"         {c[0]} {c[1]} {c[2][:90]}")
    return 1 if any(r["status"] == "BROKEN" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
