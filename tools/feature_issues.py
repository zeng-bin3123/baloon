#!/usr/bin/env python3
"""Task: "产品经理会把一个个feature列出来表格，工程师一个一个实现。但我们现在的issues都只是猜测的指标。"

feature_table.py made the PM table; this turns the table into the issue list, so the issues stop being guessed
metrics and become the table's rows, which engineers take one at a time.

  1. one issue per kernel row of the table (tools/feature_table.measure, same Strata HEAD), titled
     "feature: <strata path>", with the row's measured columns and evidence SHAs in the body and a hidden marker
     <!-- baloon-feature: <path> --> so a re-run updates the same issue instead of opening a second one.
  2. a row is DONE only when all three are measured true, otherwise the issue stays open:
       port    some src/ix_*.cu names the Strata file in its header and ref_check says OK at Strata HEAD
       test    some tests/*.cu or *.cpp includes that port
       runner  a build_*.sh compiles that test and run_all_tests.sh runs that build script
     STALE / BROKEN ports reopen the row ("upstream moved, re-port").
  3. "one by one": open rows are numbered in the table's backlog order (cost, then upstream churn); --next prints
     the single next row with what has to exist for it to close.
  4. every issue that carries no feature marker is audited instead of trusted: its title's numbers are listed and
     checked against what the body itself proves - a referenced SHA must exist in baloon or Strata, and an
     "N verified, M unverified" line with M > 0 means the pass count was never measured. Such issues are reported
     as UNMEASURED. Pull requests count: the pipeline files its "dev:"/"theory:" reports as PRs (with --apply: label "unmeasured" + one comment with the reasons; nothing is closed).

Default is a dry run that writes FEATURE_ISSUES.md. --apply writes to GitHub with the token from env GH_TOKEN
(never from argv or a file). Repo from --repo, default zeng-bin3123/baloon.

Usage: python3 tools/feature_issues.py [--next] [--apply] [--repo owner/name] [--offline] [--strata DIR]
Exit: 0 ok, 2 GitHub unreachable or token missing for --apply, 3 no Strata clone (strata_repo).
"""
import json, os, re, subprocess, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_table  # noqa: E402
import ref_check  # noqa: E402
import strata_repo  # noqa: E402

ROOT = ref_check.ROOT
MARK = re.compile(r"<!-- baloon-feature: (\S+) -->")
SHA = re.compile(r"\b[0-9a-f]{7,40}\b")
NUM = re.compile(r"\d+(?:/\d+)?")
UNVER = re.compile(r"(\d+) verified, (\d+) failed, (\d+) unverified")
LABEL = "unmeasured"


# ---------- measured row state ----------

def port_evidence(rows):
    """{strata rel: dict(ports=[...], status, tests=[...], builds=[...], runner=[...])} measured from this checkout."""
    src = os.path.join(ROOT, "src")
    tdir = os.path.join(ROOT, "tests")
    tests = {f: open(os.path.join(tdir, f), errors="replace").read()
             for f in sorted(os.listdir(tdir)) if f.endswith((".cu", ".cpp"))}
    builds = {f: open(os.path.join(ROOT, f), errors="replace").read()
              for f in sorted(os.listdir(ROOT)) if f.startswith("build_") and f.endswith(".sh")}
    runner = open(os.path.join(ROOT, "run_all_tests.sh"), errors="replace").read()
    ev = {}
    for r in rows:
        if r["port"] == "-":
            continue
        e = ev.setdefault(r["rel"], dict(ports=[], statuses=[], tests=[], builds=[], runner=[]))
        for item in r["port"].split("; "):
            name, status = item.rsplit(" ", 1)
            e["ports"].append(name); e["statuses"].append(status)
            t = [f for f, txt in tests.items() if name in txt]
            b = [f for f, txt in builds.items() if any(x in txt for x in t)]
            e["tests"] += [x for x in t if x not in e["tests"]]
            e["builds"] += [x for x in b if x not in e["builds"]]
            e["runner"] += [x for x in b if x in runner and x not in e["runner"]]
        assert all(os.path.exists(os.path.join(src, p)) for p in e["ports"])
    return ev


def state(r, e):
    if not e:
        return "OPEN", "not ported: no src/ix_*.cu names this file"
    bad = [f"{p} {s}" for p, s in zip(e["ports"], e["statuses"]) if s != "OK"]
    if bad:
        return "OPEN", "upstream moved, re-port: " + ", ".join(bad)
    if not e["tests"]:
        return "OPEN", "ported, but no test includes the port"
    if not e["runner"]:
        return "OPEN", "ported and tested, but run_all_tests.sh never builds the test"
    return "DONE", "port OK at Strata HEAD, test " + ", ".join(e["tests"]) + " via " + ", ".join(e["runner"])


def feature_body(r, st, why, e, rank, head):
    cols = ("group", "commits", "loc", "kernels", "wave64", "w32sync", "lane32", "warpsize", "tc", "dynsmem",
            "tfs", "ipc", "fp64", "cost")
    L = [f"<!-- baloon-feature: {r['rel']} -->",
         f"Strata `{r['rel']}` measured at Strata HEAD {head} by tools/feature_table.py (no estimate in this issue).", "",
         f"**State: {st}** — {why}" + (f" — backlog rank {rank}" if rank else ""), "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
         "| " + " | ".join(("yes" if r[c] else "no") if c == "wave64" else str(r[c]) for c in cols) + " |", "",
         f"Evidence: first {r['first']}, last {r['last']} (git log -- {r['rel']}).", "",
         "Closes when all three hold (re-run `python3 tools/feature_issues.py --apply`):",
         f"- [{'x' if e and all(s == 'OK' for s in e['statuses']) else ' '}] port: src/ix_*.cu header names "
         f"`{r['rel']}` + a Strata sha, ref_check OK" + (f" ({', '.join(e['ports'])})" if e else ""),
         f"- [{'x' if e and e['tests'] else ' '}] test: tests/*.cu includes the port" + (f" ({', '.join(e['tests'])})" if e and e['tests'] else ""),
         f"- [{'x' if e and e['runner'] else ' '}] runner: a build_*.sh run by run_all_tests.sh compiles that test"
         + (f" ({', '.join(e['runner'])})" if e and e['runner'] else "")]
    if r["tc"]:
        L.append(f"- note: {r['tc']} tensor-core sites — BI-V100 has none, a scalar path is required")
    if r["tfs"]:
        L.append(f"- note: {r['tfs']} __threadfence_system — not supported on ivcore10 (ENV.md)")
    if r["ipc"]:
        L.append(f"- note: {r['ipc']} cudaIpc* — not supported on BI-V100, use cudaMemcpyPeer (ENV.md)")
    return "\n".join(L) + "\n"


def plan(strata):
    rows, head, total = feature_table.measure(strata)
    head = head.strip()
    kernels = [r for r in rows if r["kind"] == "kernel"]
    ev = port_evidence(kernels)
    feats, rank = [], 0
    for r in kernels:  # measure() returns table order: ported first, then the backlog order
        e = ev.get(r["rel"])
        st, why = state(r, e)
        rk = None
        if st == "OPEN":
            rank += 1; rk = rank
        feats.append(dict(rel=r["rel"], state=st, why=why, rank=rk, row=r,
                          title=f"feature: {r['rel']}", body=feature_body(r, st, why, e, rk, head)))
    feats.sort(key=lambda f: (f["rank"] is None, f["rank"] or 0))
    return feats, head, total.strip()


# ---------- audit of issues that are not table rows ----------

def sha_exists(sha, strata):
    for repo in (ROOT, strata):
        if subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0:
            return True
    return False


def claims(title):
    """The metric part of a title: after the last em dash when there is one ("theory: chunk_1 — 10 files")."""
    return NUM.findall(title.rsplit("—", 1)[-1])


def audit(issue, strata, pr=None):
    """Reasons the issue's numbers are not measured; [] means its body (or, for a PR, GitHub's own count) backs them.
    pr is the list of /pulls/N/files for a pull request: GitHub's file list is a measured count, so "N files" is
    checked against it, not counting the pipeline's own conv.json manifest (which every theory PR adds)."""
    title, body = issue["title"], issue.get("body") or ""
    if not claims(title):
        return []
    m = re.search(r"(\d+) files?\b", title.rsplit("—", 1)[-1])
    if pr is not None and m and set(claims(title)) == {m.group(1)}:
        n = sum(not f["filename"].endswith("/conv.json") for f in pr)
        return [] if n == int(m.group(1)) else [f"title says {m.group(1)} files, the PR adds {n} besides conv.json"]
    why = []
    shas = [s for s in dict.fromkeys(SHA.findall(body)) if not s.isdigit()]
    real = [s for s in shas if sha_exists(s, strata)]
    if not real:
        why.append(f"title claims {', '.join(claims(title))} but the body names no commit that exists in baloon or Strata"
                   + (f" (checked {', '.join(shas[:5])})" if shas else ""))
    if re.search(r"Reference: \S+ @\s*$", body, re.M):
        why.append("reference line has no commit after '@'")
    m = UNVER.search(body)
    if m and int(m.group(3)) > 0:
        why.append(f"{m.group(3)} of {sum(map(int, m.groups()))} theorems unverified, so a pass count is not measured")
    return why


# ---------- GitHub ----------

def gh(method, path, data=None, token=None):
    req = urllib.request.Request("https://api.github.com" + path, method=method,
                                 data=json.dumps(data).encode() if data is not None else None)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", "token " + token)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                left = resp.headers.get("X-RateLimit-Remaining")
                if left is not None and int(left) < 5:
                    time.sleep(max(1, int(resp.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time()))
                return json.loads(resp.read() or "null")
        except urllib.error.HTTPError as err:
            if err.code in (403, 429) and attempt < 4:
                time.sleep(int(err.headers.get("Retry-After") or 2 ** (attempt + 2))); continue
            raise


def issues(repo, token):
    out, page = [], 1
    while True:
        b = gh("GET", f"/repos/{repo}/issues?state=all&per_page=100&page={page}", token=token)
        out += b  # the pipeline files its "dev:"/"theory:" reports as PRs; they are issues here too
        if len(b) < 100:
            return out
        page += 1


def apply(repo, token, feats, audited, existing):
    by_mark = {}
    for i in existing:
        m = MARK.search(i.get("body") or "")
        if m:
            by_mark[m.group(1)] = i
    n = dict(created=0, updated=0, closed=0, reopened=0, labelled=0)
    for f in feats:
        want = "closed" if f["state"] == "DONE" else "open"
        cur = by_mark.get(f["rel"])
        if cur is None:
            if want == "closed":
                continue  # a done row with no issue needs none
            gh("POST", f"/repos/{repo}/issues", dict(title=f["title"], body=f["body"]), token); n["created"] += 1
            continue
        patch = {}
        if (cur.get("body") or "") != f["body"]:
            patch["body"] = f["body"]
        if cur["state"] != want:
            patch["state"] = want; n["closed" if want == "closed" else "reopened"] += 1
        if patch:
            gh("PATCH", f"/repos/{repo}/issues/{cur['number']}", patch, token); n["updated"] += 1
    for i, why in audited:
        if any(l["name"] == LABEL for l in i["labels"]):
            continue
        gh("POST", f"/repos/{repo}/issues/{i['number']}/labels", dict(labels=[LABEL]), token)
        gh("POST", f"/repos/{repo}/issues/{i['number']}/comments",
           dict(body="tools/feature_issues.py: the numbers in this title are not measured —\n"
                     + "\n".join("- " + w for w in why)
                     + "\n\nThe measured work list is the `feature:` issues (one per STRATA_FEATURE_TABLE_GEN.md "
                       "kernel row). Not closed; remove the label once the body names the commit/log that proves it."),
           token)
        n["labelled"] += 1
    return n


# ---------- main ----------

def report(feats, head, total, audited, trusted, repo, gh_note):
    done = [f for f in feats if f["state"] == "DONE"]
    todo = [f for f in feats if f["state"] == "OPEN"]
    L = ["# Feature issues (generated by tools/feature_issues.py — do not edit by hand)", "",
         f"Source: tools/feature_table.measure at Strata HEAD {head}, {total} commits. Repo {repo}. {gh_note}", "",
         f"{len(feats)} kernel rows -> {len(todo)} open feature issues, {len(done)} done.", "",
         "## Open, in the order engineers take them", "",
         "| rank | issue title | cost | why open |", "|---|---|---|---|"]
    L += [f"| {f['rank']} | {f['title']} | {f['row']['cost']} | {f['why']} |" for f in todo]
    L += ["", "## Done", "", "| issue title | evidence |", "|---|---|"]
    L += [f"| {f['title']} | {f['why']} |" for f in done] or ["| (none) | |"]
    L += ["", "## Issues that are not table rows", "", "| # | title | verdict |", "|---|---|---|"]
    L += [f"| {i['number']}{' (PR)' if 'pull_request' in i else ''} | {i['title']} | UNMEASURED: {'; '.join(w)} |" for i, w in audited]
    L += [f"| {i['number']}{' (PR)' if 'pull_request' in i else ''} | {i['title']} | measured (body or GitHub's own count backs the title) |" for i in trusted]
    if not audited and not trusted:
        L.append("| - | (issues not fetched) | |")
    return "\n".join(L) + "\n"


def main(argv=sys.argv[1:]):
    repo = argv[argv.index("--repo") + 1] if "--repo" in argv else "zeng-bin3123/baloon"
    strata = strata_repo.ensure(argv)
    feats, head, total = plan(strata)
    if "--next" in argv:
        nxt = next((f for f in feats if f["state"] == "OPEN"), None)
        print("nothing open" if nxt is None else f"next: rank {nxt['rank']} {nxt['rel']} (cost {nxt['row']['cost']})\n{nxt['body']}")
        return 0
    token = os.environ.get("GH_TOKEN")
    if "--apply" in argv and ("--offline" in argv or not token):
        # without the current issue list a write would open duplicates; without a token it cannot write at all
        print("--apply needs GH_TOKEN in the environment and cannot be combined with --offline"); return 2
    existing, audited, trusted, note = [], [], [], "Issues not fetched (--offline)."
    if "--offline" not in argv:
        try:
            existing = issues(repo, token)
            for i in existing:
                if MARK.search(i.get("body") or ""):
                    continue
                pr = (gh("GET", f"/repos/{repo}/pulls/{i['number']}/files?per_page=100", token=token)
                      if "pull_request" in i else None)
                why = audit(i, strata, pr)
                (audited.append((i, why)) if why else trusted.append(i))
            note = f"{len(existing)} issues/PRs fetched."
        except (urllib.error.URLError, OSError) as err:
            if "--apply" in argv:
                print(f"GitHub unreachable: {err}"); return 2
            existing, audited, trusted, note = [], [], [], f"Issues not fetched: {err}."
    out = os.path.join(ROOT, "FEATURE_ISSUES.md")
    open(out, "w").write(report(feats, head, total, audited, trusted, repo, note))
    print(f"wrote {out}: {sum(f['state'] == 'OPEN' for f in feats)} open / {len(feats)} rows, "
          f"{len(audited)} existing issues UNMEASURED")
    if "--apply" in argv:
        print("applied:", apply(repo, token, feats, audited, existing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
