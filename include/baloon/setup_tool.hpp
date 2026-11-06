// setup_tool.hpp - the machine-probing and build-planning logic of the installer. Everything here is a pure function of
// text/predicates handed in (cpuinfo, meminfo, ixsmi csv, PATH, an exists() predicate), so the decisions are testable
// without touching the machine; only `run` (and `download`/`pip_install` through it) can execute anything, and a Runner
// in dry mode only records the command lines.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baloon/vision_build_config.hpp"

namespace baloon {
namespace setup {

inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) --b;
    return s.substr(a, b - a);
}
inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> v; std::string cur;
    for (char c : s) { if (c == sep) { v.push_back(cur); cur.clear(); } else cur += c; }
    v.push_back(cur);
    return v;
}

// ---- console text (pure formatting; the caller prints) -----------------------------------------------------------
inline std::string say(const std::string& m) { return "  " + m; }
inline std::string ok(const std::string& m) { return "[ ok ] " + m; }
inline std::string warn(const std::string& m) { return "[warn] " + m; }
inline std::string fail(const std::string& m) { return "[FAIL] " + m; }
inline std::string step(int n, int total, const std::string& title) {
    return "[" + std::to_string(n) + "/" + std::to_string(total) + "] " + title;
}
inline std::string mark(bool good) { return good ? "OK" : "--"; }
inline std::string start(const std::string& name) { return "== " + name + " setup =="; }
inline std::string done(const std::string& run_cmd) { return "Setup complete. Start with: " + run_cmd; }
// Captured command output: trailing blanks/newlines removed, nothing else touched.
inline std::string out(const std::string& s) {
    size_t b = s.size();
    while (b > 0 && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) --b;
    return s.substr(0, b);
}
// Yes/no answer: empty -> default; y/yes -> true; n/no -> false (case-insensitive); anything else -> default.
inline bool ask(const std::string& answer, bool dflt) {
    std::string a = trim(answer);
    for (char& c : a) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    if (a == "y" || a == "yes") return true;
    if (a == "n" || a == "no") return false;
    return dflt;
}

// ---- memory ------------------------------------------------------------------------------------------------------
inline double ram_gb(uint64_t bytes) { return (double)bytes / (double)(1ull << 30); }
inline double free_gb(uint64_t bytes) { return (double)bytes / (double)(1ull << 30); }
// Value of a "/proc/meminfo" key in bytes (the file reports kB = 1024 bytes); 0 when missing/malformed.
inline uint64_t meminfo_bytes(const std::string& text, const std::string& key) {
    for (const std::string& line : split(text, '\n')) {
        if (line.compare(0, key.size() + 1, key + ":") != 0) continue;
        std::istringstream is(line.substr(key.size() + 1));
        unsigned long long v = 0; std::string unit;
        if (!(is >> v)) return 0;
        is >> unit;
        return unit == "kB" ? v * 1024ull : v;
    }
    return 0;
}

// ---- cpu ---------------------------------------------------------------------------------------------------------
struct CpuInfo { std::string model; bool avx2 = false, fma = false, f16c = false, avx512_full = false; int logical = 0; };

inline bool has_flag(const std::set<std::string>& flags, const char* f) { return flags.count(f) != 0; }
// The complete AVX-512 set the engine kernels need: F, BW, DQ, VL, VBMI and VNNI (plus F16C and FMA).
inline bool _cpuid_avx512_full(const std::set<std::string>& flags) {
    static const char* need[] = {"avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512_vbmi", "avx512_vnni", "f16c", "fma"};
    for (const char* f : need) if (!has_flag(flags, f)) return false;
    return true;
}
// Parses /proc/cpuinfo text: model name, logical processor count, flags of the first processor.
inline CpuInfo cpu_info(const std::string& cpuinfo) {
    CpuInfo c; std::set<std::string> flags; bool flags_done = false;
    for (const std::string& line : split(cpuinfo, '\n')) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon)), val = trim(line.substr(colon + 1));
        if (key == "processor") ++c.logical;
        else if (key == "model name" && c.model.empty()) c.model = val;
        else if (key == "flags" && !flags_done) {
            for (const std::string& f : split(val, ' ')) if (!f.empty()) flags.insert(f);
            flags_done = true;
        }
    }
    c.avx2 = has_flag(flags, "avx2"); c.fma = has_flag(flags, "fma"); c.f16c = has_flag(flags, "f16c");
    c.avx512_full = _cpuid_avx512_full(flags);
    return c;
}

// ---- gpu ---------------------------------------------------------------------------------------------------------
struct GpuInfo { std::string name; uint64_t mib = 0; std::string driver; };
// Major version of "3.2.1" -> 3; no leading digits -> -1.
inline int driver_major(const std::string& v) {
    const std::string s = trim(v);
    size_t i = 0; long m = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9' && m < 100000000) m = m * 10 + (s[i++] - '0');
    return i == 0 ? -1 : (int)m;
}
// Parses `ixsmi --query-gpu=name,memory.total,driver_version --format=csv` text ("32768 MiB" cells); header and
// malformed rows are skipped.
inline std::vector<GpuInfo> gpu_info(const std::string& csv) {
    std::vector<GpuInfo> g;
    for (const std::string& line : split(csv, '\n')) {
        const std::vector<std::string> c = split(line, ',');
        if (c.size() != 3) continue;
        GpuInfo x; x.name = trim(c[0]); x.driver = trim(c[2]);
        std::istringstream is(trim(c[1])); unsigned long long v = 0; std::string unit;
        if (!(is >> v >> unit) || unit != "MiB" || x.name.empty()) continue;
        x.mib = v; g.push_back(x);
    }
    return g;
}

// ---- tool discovery ----------------------------------------------------------------------------------------------
using Exists = std::function<bool(const std::string&)>;
// First `dir/name` of the PATH-like list (':'-separated) that exists; empty when none.
inline std::string find_tool(const std::string& name, const std::string& path, const Exists& exists) {
    for (const std::string& d : split(path, ':')) {
        if (d.empty()) continue;
        const std::string p = d + "/" + name;
        if (exists(p)) return p;
    }
    return "";
}
// nvcc: the toolchain root's bin first (corex), then PATH.
inline std::string find_nvcc(const std::string& root, const std::string& path, const Exists& exists) {
    if (!root.empty() && exists(root + "/bin/nvcc")) return root + "/bin/nvcc";
    return find_tool("nvcc", path, exists);
}
inline std::vector<std::string> cuda_lib_dirs(const std::string& root) { return {root + "/lib64", root + "/lib"}; }
// vcvars64.bat under the given Visual Studio roots (newest edition first); empty when none exists.
inline std::string find_vcvars(const std::vector<std::string>& roots, const Exists& exists) {
    static const char* editions[] = {"Enterprise", "Professional", "Community", "BuildTools"};
    for (const std::string& r : roots)
        for (const char* e : editions) {
            const std::string p = r + "/" + e + "/VC/Auxiliary/Build/vcvars64.bat";
            if (exists(p)) return p;
        }
    return "";
}

// ---- engine configurations ---------------------------------------------------------------------------------------
// Which built configurations exist, from directory names like "engine-avx512": the known tags in fixed order.
inline std::vector<std::string> installed_configs(const std::vector<std::string>& dirs) {
    static const char* tags[] = {"portable", "avx2", "avx512", "cuda"};
    std::vector<std::string> r;
    for (const char* t : tags)
        for (const std::string& d : dirs)
            if (d == std::string("engine-") + t) { r.push_back(t); break; }
    return r;
}
// Best configuration for the machine: any GPU -> cuda; full AVX-512 -> avx512; AVX2+FMA+F16C -> avx2; else portable.
inline std::string choose_config(const CpuInfo& c, const std::vector<GpuInfo>& g) {
    if (!g.empty()) return "cuda";
    if (c.avx512_full) return "avx512";
    if (c.avx2 && c.fma && c.f16c) return "avx2";
    return "portable";
}

// ---- command execution -------------------------------------------------------------------------------------------
struct Runner {
    bool dry = true;
    std::vector<std::string> log;
};
// Records the command; executes it only when not dry. Returns the exit status (0 in dry mode).
inline int run(Runner& r, const std::string& cmd) {
    r.log.push_back(cmd);
    return r.dry ? 0 : std::system(cmd.c_str());
}
inline std::string shq(const std::string& s) {
    std::string q = "'";
    for (char c : s) { if (c == '\'') q += "'\\''"; else q += c; }
    return q + "'";
}
inline int download(Runner& r, const std::string& url, const std::string& dest) {
    return run(r, "curl -fL --retry 3 -o " + shq(dest) + " " + shq(url));
}
inline int pip_install(Runner& r, const std::vector<std::string>& pkgs) {
    if (pkgs.empty()) return 0;
    std::string c = "python3 -m pip install --break-system-packages";
    for (const std::string& p : pkgs) c += " " + shq(p);
    return run(r, c);
}
inline int install_build_tools(Runner& r) {
    int rc = run(r, "apt-get update");
    return rc != 0 ? rc : run(r, "apt-get install -y build-essential cmake git");
}
inline int get_llama_cpp(Runner& r, const std::string& dir, const std::string& tag) {
    int rc = run(r, "git clone --depth 1 --branch " + shq(tag) + " https://github.com/ggml-org/llama.cpp " + shq(dir));
    return rc;
}
// Prebuilt engine archive for a configuration, unpacked into dest.
inline int get_prebuilt(Runner& r, const std::string& config, const std::string& base_url, const std::string& dest) {
    const std::string arc = dest + "/engine-" + config + ".tar.gz";
    int rc = download(r, base_url + "/engine-" + config + ".tar.gz", arc);
    return rc != 0 ? rc : run(r, "tar -xzf " + shq(arc) + " -C " + shq(dest));
}
// CMake -D flags of a configuration: the portable switch forces the generic ISA, native does not; static libraries,
// no tools; CUDA on for the cuda configuration.
inline std::vector<std::string> cmake_flags(const std::string& config) {
    vision_cfg::Options o; o.portable = config == "portable";
    o.cuda = config == "cuda";
    const vision_cfg::Settings s = vision_cfg::configure(o);
    auto onoff = [](bool b) { return b ? "ON" : "OFF"; };
    std::vector<std::string> f;
    f.push_back(std::string("-DGGML_NATIVE=") + onoff(s.native));
    if (s.openmp_forced) f.push_back(std::string("-DGGML_OPENMP=") + onoff(s.openmp));
    f.push_back(std::string("-DGGML_CUDA=") + onoff(s.ggml_cuda));
    f.push_back(std::string("-DBUILD_SHARED_LIBS=") + onoff(s.shared_libs));
    f.push_back(std::string("-DGGML_BACKEND_DL=") + onoff(s.backend_dl));
    for (const char* k : vision_cfg::ALWAYS_OFF)
        if (std::string(k) != "BUILD_SHARED_LIBS" && std::string(k) != "GGML_BACKEND_DL") f.push_back(std::string("-D") + k + "=OFF");
    for (const char* k : vision_cfg::ALWAYS_ON) f.push_back(std::string("-D") + k + "=ON");
    return f;
}
inline int cmake_build(Runner& r, const std::string& src, const std::string& bld, const std::string& config) {
    std::string c = "cmake -S " + shq(src) + " -B " + shq(bld) + " -DCMAKE_BUILD_TYPE=Release";
    for (const std::string& f : cmake_flags(config)) c += " " + f;
    int rc = run(r, c);
    return rc != 0 ? rc : run(r, "cmake --build " + shq(bld) + " --config Release -j");
}
inline int build_engine(Runner& r, const std::string& root, const std::string& config) {
    const std::string src = root + "/llama.cpp", bld = root + "/build-" + config;
    int rc = get_llama_cpp(r, src, "master");
    return rc != 0 ? rc : cmake_build(r, src, bld, config);
}

// Launcher script text: environment first, then the engine invocation with all script arguments forwarded.
inline std::string write_run_script(const std::string& root, const std::string& config, const std::string& model) {
    std::string s = "#!/bin/sh\nset -e\nROOT=" + shq(root) + "\n";
    if (config == "cuda") {
        std::string ld;
        for (const std::string& d : cuda_lib_dirs("/usr/local/corex")) ld += d + ":";
        s += "export LD_LIBRARY_PATH=" + ld + "${LD_LIBRARY_PATH:-}\n";
    }
    s += "exec \"$ROOT/build-" + config + "/bin/engine\" --model " + shq(model) + " \"$@\"\n";
    return s;
}

}  // namespace setup
}  // namespace baloon
