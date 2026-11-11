#!/usr/bin/env python3
"""Task: "收集github Strata 仓库的所有commit (先在你的code exec环境遍历, 然后慢慢扫。)"
The local tools (ref_check, feature_table) read Strata's full git history, so the machine running them must have a
full clone. ensure() finds it or makes it, so no tool depends on someone having cloned ../Strata by hand:
  1. --strata PATH, else env STRATA_DIR, else <baloon>/../Strata
  2. missing -> git clone --no-single-branch https://github.com/Niko1221/Strata.git (full history, all commits)
  3. shallow -> git fetch --unshallow (a shallow clone would undercount commits)
  4. --fetch (or STRATA_FETCH=1) -> git fetch + fast-forward main, so HEAD is current
Exit code 3 = environment problem (no clone and clone failed), distinct from ref_check's 1 = broken reference.
Usage: python3 tools/strata_repo.py [--strata PATH] [--fetch]   -> prints path, HEAD, commit count
"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("STRATA_URL", "https://github.com/Niko1221/Strata.git")
ENV_ERROR = 3


def _git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)


def path(argv=None):
    argv = sys.argv if argv is None else argv
    if "--strata" in argv:
        return os.path.abspath(argv[argv.index("--strata") + 1])
    return os.path.abspath(os.environ.get("STRATA_DIR") or os.path.join(ROOT, "..", "Strata"))


def ensure(argv=None):
    argv = sys.argv if argv is None else argv
    p = path(argv)
    if _git("rev-parse", "HEAD", cwd=p).returncode if os.path.isdir(p) else True:
        if os.path.exists(p) and os.listdir(p):
            sys.stderr.write(f"strata_repo: {p} exists but is not a git clone\n"); sys.exit(ENV_ERROR)
        sys.stderr.write(f"strata_repo: no clone at {p}, cloning {URL} (full history)...\n")
        r = _git("clone", "-q", "--no-single-branch", URL, p)
        if r.returncode:
            sys.stderr.write(f"strata_repo: clone failed: {r.stderr.strip()}\n"); sys.exit(ENV_ERROR)
    if _git("rev-parse", "--is-shallow-repository", cwd=p).stdout.strip() == "true":
        sys.stderr.write("strata_repo: shallow clone, fetching full history...\n")
        _git("fetch", "-q", "--unshallow", cwd=p)
    if "--fetch" in argv or os.environ.get("STRATA_FETCH") == "1":
        _git("fetch", "-q", "origin", cwd=p)
        r = _git("merge", "-q", "--ff-only", "origin/main", cwd=p)
        if r.returncode:
            sys.stderr.write(f"strata_repo: could not fast-forward main: {r.stderr.strip()}\n")
    return p


if __name__ == "__main__":
    p = ensure()
    head = _git("log", "-1", "--format=%h %ad", "--date=short", cwd=p).stdout.strip()
    n = _git("rev-list", "--count", "HEAD", cwd=p).stdout.strip()
    print(f"{p}\nHEAD {head}, {n} commits")
