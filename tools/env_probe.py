#!/usr/bin/env python3
"""Task: "把编译环境（clang）、硬件那些硬信息和shell跑的命令md先从zeng-bin3123/baloon main 拿取"

tools/ix_env.sh already takes the compiler (clang) from ENV.md. This takes the other two parts of the sentence:
  "shell跑的命令md"  every ```bash block under "## Probe Scripts" in ENV.md is extracted and run VERBATIM
                     (in a temp dir; any *.cu it names is copied in from tests/, e.g. tests/probe_hw.cu),
                     so ENV.md stays the only place the commands are written.
  "硬件那些硬信息"     every hard claim in ENV.md's "## Hardware" section and "warpSize" limit is parsed out of
                     ENV.md and compared with what the machine reports (probe_hw key=value + ixsmi text):
                     GPU count, GPU name, memory per GPU, arch, IX-ML / driver / CUDA compat versions, warpSize
                     (property AND device-side builtin), PCIe peer access, clang++ presence.
Writes ENV_MEASURED.md (generated; claim | ENV.md | measured | status, then each block's raw output).
Exit: 0 every claim OK, 1 some claim DIFF (ENV.md is wrong for this machine), 2 nothing measurable (no toolchain).
Usage (repo root):  python3 tools/env_probe.py           # run on the BI-V100 machine
                    python3 tools/env_probe.py --list    # only show what was parsed from ENV.md (runs nothing)
"""
import os, re, shutil, socket, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.environ.get("IX_ENV_MD", os.path.join(ROOT, "ENV.md"))
OUT = os.path.join(ROOT, "ENV_MEASURED.md")


def section(text, title):
    m = re.search(rf"^## {re.escape(title)}.*?$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def probe_blocks(text):
    """[(### title, bash source)] for every ```bash block in '## Probe Scripts', in file order."""
    out, title = [], "?"
    for m in re.finditer(r"^### (.+?)$|^```bash\n(.*?)^```", section(text, "Probe Scripts"), re.M | re.S):
        if m.group(1):
            title = m.group(1).strip()
        else:
            out.append((title, m.group(2)))
    return out


def claims(text):
    hw, c = section(text, "Hardware"), {}
    m = re.search(r"GPU:\s*(\d+)\s*[×x]\s*(.+?),\s*([\d.]+)\s*GB", hw)
    if m:
        c["gpu_count"], c["gpu_name"], c["mem_gb"] = int(m.group(1)), m.group(2).strip(), float(m.group(3))
        c["gpu_model"] = c["gpu_name"].split()[-1]  # "Iluvatar BI-V100" -> "BI-V100"
    for key, pat in (("arch", r"Architecture:\s*(\S+)"), ("ixml", r"IX-ML:\s*([\d.]+)"),
                     ("driver", r"Driver:\s*([\d.]+)"), ("cuda", r"CUDA compat:\s*([\d.]+)"),
                     ("interconnect", r"Interconnect:\s*(.+)")):
        m = re.search(pat, hw)
        if m:
            c[key] = m.group(1).strip()
    m = re.search(r"warpSize:\s*(\d+)", text)
    if m:
        c["warp"] = int(m.group(1))
    return c


def toolchain():
    r = subprocess.run(["bash", "-c", f'source "{ROOT}/tools/ix_env.sh" && echo "$IX_CXX" && echo "$IX_ARCH"'],
                       capture_output=True, text=True)
    lines = r.stdout.split("\n")
    return (lines[0], lines[1]) if r.returncode == 0 and len(lines) > 1 else (None, None)


def sh(cmd, cwd, timeout=300):
    t = time.time()
    try:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr), time.time() - t
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s\n{e.stdout or ''}", time.time() - t


def kv(text):
    return dict(m.groups() for m in re.finditer(r"^([a-z]\w*(?:[.-]\w+)*)=(.*)$", text, re.M))


def main():
    text = open(MD).read()
    blocks, cl = probe_blocks(text), claims(text)
    cxx, arch = toolchain()
    if "--list" in sys.argv:
        print(f"ENV.md: {MD}\nclaims: {cl}\ntoolchain (via ix_env.sh): CXX={cxx} ARCH={arch}")
        for t, b in blocks:
            refs = sorted(set(re.findall(r"[\w.-]+\.cu\b", b)))
            print(f"--- block '{t}': {len(b.splitlines())} lines, needs {refs or 'no .cu'}")
        return 0

    work = tempfile.mkdtemp(prefix="env_probe_")
    runs = []
    for t, b in blocks:  # 1. ENV.md's own shell commands, verbatim
        notes = []
        for ref in sorted(set(re.findall(r"[\w.-]+\.cu\b", b))):
            src = os.path.join(ROOT, "tests", ref)
            if os.path.exists(src):
                shutil.copy(src, work); notes.append(f"{ref} <- tests/{ref}")
            else:
                notes.append(f"{ref} NOT IN REPO")
        rc, out, dt = sh(b, work)
        runs.append((f"ENV.md: {t}", b, rc, out, dt, notes))
    # 2. same probe through the ENV.md toolchain block (ix_build), so the facts are measured even if the
    #    probe block's own --cuda-path (corex-3.2.3) does not exist on this machine
    bcmd = f'source "{ROOT}/tools/ix_env.sh" && ix_build "{ROOT}/tests/probe_hw.cu" "{work}/probe_hw_ix" && "{work}/probe_hw_ix"'
    rc, out, dt = sh(bcmd, work)
    runs.append(("ix_env.sh toolchain: probe_hw", bcmd, rc, out, dt, []))
    rc, out, dt = sh("ixsmi -q 2>&1; echo; ixsmi topo -m 2>&1", work, 60)
    runs.append(("ixsmi -q + topo", "ixsmi -q; ixsmi topo -m", rc, out, dt, []))
    shutil.rmtree(work, ignore_errors=True)

    facts = {}
    for name, _, rc, out, *_ in runs:
        if "devices=" in out and not facts:
            facts = kv(out); facts["_src"] = name
    alltext = "\n".join(r[3] for r in runs)
    n = int(facts.get("devices", -1))
    devs = range(max(n, 0))
    rows = []

    def row(claim, want, got, ok):
        rows.append((claim, str(want), str(got), "UNMEASURED" if ok is None else ("OK" if ok else "DIFF")))

    have = bool(facts)
    row("GPU count", cl.get("gpu_count"), n if have else "-", (n == cl.get("gpu_count")) if have else None)
    names = sorted({facts.get(f"d{d}.name", "?") for d in devs})
    row("GPU model", cl.get("gpu_model"), ", ".join(names) or "-",
        all(cl.get("gpu_model", "?") in x for x in names) if have and names else None)
    mems = [float(facts.get(f"d{d}.mem_gib", 0)) for d in devs]
    row("memory per GPU (GB, ±10%)", cl.get("mem_gb"), ", ".join(f"{m:.1f}" for m in mems) or "-",
        all(abs(m - cl["mem_gb"]) <= 0.1 * cl["mem_gb"] for m in mems) if have and mems and "mem_gb" in cl else None)
    w = sorted({facts.get(f"d{d}.{k}") for d in devs for k in ("warp", "warp_dev")})
    row("warpSize (prop + device builtin)", cl.get("warp"), ", ".join(map(str, w)) or "-",
        w == [str(cl.get("warp"))] if have else None)
    rt = int(facts.get("runtime", 0))
    rts = f"{rt // 1000}.{rt % 1000 // 10}" if rt else "-"
    row("CUDA compat (cudaRuntimeGetVersion)", cl.get("cuda"), rts, (rts == cl.get("cuda")) if rt else None)
    ixsmi_out = runs[-1][3] if runs[-1][2] == 0 else ""
    for key, label in (("driver", "Driver version (in ixsmi -q)"), ("ixml", "IX-ML version (in ixsmi -q)")):
        hit = [l.strip() for l in ixsmi_out.splitlines() if cl.get(key, "#") in l]
        ver = [l.strip() for l in ixsmi_out.splitlines() if re.search(r"version", l, re.I)][:3]
        row(label, cl.get(key), (hit or ver or ["-"])[0][:70], (bool(hit)) if ixsmi_out else None)
    row("arch (ix_env.sh ARCH; probe built with it)", cl.get("arch"), arch or "-",
        (arch == cl.get("arch") and "devices=" in runs[-2][3]) if arch and have else None)
    peers = {k: v for k, v in facts.items() if k.startswith("p")}
    row("interconnect: peer access pairs", cl.get("interconnect"),
        f"{sum(v == '1' for v in peers.values())}/{len(peers)} pairs can peer" if peers else "-",
        True if peers else None)
    cx_ok = bool(cxx) and os.access(cxx, os.X_OK)
    cv = sh(f'"{cxx}" --version 2>&1 | head -1', ROOT, 30)[1].strip() if cx_ok else "not found"
    row("clang++ (ENV.md toolchain)", cxx, cv, cx_ok)

    st = [r[3] for r in rows]
    ex = 2 if not have else (1 if "DIFF" in st else 0)
    md = [f"# ENV.md measured on this machine (generated by tools/env_probe.py — do not edit by hand)\n",
          f"host `{socket.gethostname()}`, {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}, "
          f"baloon `{sh('git log --oneline -1', ROOT, 10)[1].strip()[:60]}`, facts from: {facts.get('_src', 'none')}. "
          f"Exit {ex}: {['all claims OK', 'ENV.md differs from this machine', 'nothing measurable'][ex]}.\n",
          "| claim | ENV.md | measured | status |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {c.replace('|', '/')} | {s} |" for a, b, c, s in rows]
    for name, cmd, rc, out, dt, notes in runs:
        md += [f"\n## {name} — exit {rc}, {dt:.1f}s" + (f" ({'; '.join(notes)})" if notes else ""),
               "```bash", cmd.rstrip(), "```", "```", out.rstrip()[-6000:] or "(no output)", "```"]
    open(OUT, "w").write("\n".join(md) + "\n")
    for a, b, c, s in rows:
        print(f"{s:10} {a:42} ENV.md={b:18} measured={c[:60]}")
    print(f"block exits: {[(r[0], r[2]) for r in runs]}\nwrote {OUT}, exit {ex}")
    return ex


if __name__ == "__main__":
    sys.exit(main())
