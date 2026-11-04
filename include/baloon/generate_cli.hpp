// generate_cli.hpp - the `generate` command line driver: option parsing and validation, the teacher-forced decode loop
// (one logits row per position 0..np-2+max_new, generation starting at the last prompt position), the speculative verify
// loop (accept the longest draft prefix equal to the verifier outputs, commit accepted+1 tokens), a multi-request worker
// pool, the `--serve` line protocol and a deterministic synthetic backend used for self-tests and the CLI smoke run.
// The numeric contracts (fit, rows, batched split, auto expert slots, PCIe clamp, admission) come from
// generate_driver_accounting.hpp; this file adds the control flow around them.
#pragma once
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "baloon/generate_driver_accounting.hpp"

namespace baloon {
namespace gencli {

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------------------------------------------------
// integer lists
// ---------------------------------------------------------------------------------------------------------------------
inline bool parse_i64(const std::string& s, int64_t& v) {
    if (s.empty()) return false;
    const char* b = s.data();
    const char* e = b + s.size();
    int64_t x = 0;
    auto r = std::from_chars(b, e, x, 10);
    if (r.ec != std::errc() || r.ptr != e) return false;
    v = x;
    return true;
}

// "1,2,-3" -> {1,2,-3}. The empty string is the empty list; empty items, spaces, '+' signs and overflow are errors
// (the output is cleared on error).
inline bool parse_i64_list(const std::string& s, std::vector<int64_t>& out) {
    out.clear();
    if (s.empty()) return true;
    size_t i = 0;
    for (;;) {
        const size_t j = s.find(',', i);
        const std::string tok = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
        int64_t v = 0;
        if (!parse_i64(tok, v)) { out.clear(); return false; }
        out.push_back(v);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return true;
}

inline std::string join_i64(const std::vector<int64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += ','; s += std::to_string(v[i]); }
    return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------------------------------------------------
struct Options {
    std::vector<int64_t> prompt;     // --prompt-ids
    int64_t max_new = 32;            // --max-new
    int64_t max_context = 4096;      // --max-context
    int64_t prefill_until = 0;       // --prefill-until (0 = whole prompt but the last position)
    int64_t spec = 0;                // --spec S (0 = off, else verify window size >= 2)
    bool expert_cache_auto = false;  // --expert-cache auto
    int64_t expert_cache = 0;        // --expert-cache N
    int64_t vram_free = -1;          // --vram-free BYTES
    int64_t vram_reserve = 0;        // --vram-reserve BYTES
    int64_t blob = -1;               // --blob BYTES (largest expert blob)
    int64_t profile_len = -1;        // --profile-len N (-1 = no profile)
    double pcie = 0.5;               // --pcie FRACTION
    bool serve = false;              // --serve
    bool trace = false;              // --trace
    int64_t vocab = 257;             // --vocab (synthetic backend)
    int64_t workers = 1;             // --workers
    std::vector<int64_t> gpus;       // --gpus
    bool help = false;               // --help
};

inline std::string usage(const char* prog) {
    std::string p = prog && *prog ? prog : "generate";
    std::string s;
    s += "usage: " + p + " [options]\n";
    s += "  --prompt-ids a,b,c     prompt token ids (required unless --serve)\n";
    s += "  --max-new N            tokens to generate (default 32, must be > 0)\n";
    s += "  --max-context N        context size (default 4096); needs prompt + max-new <= N\n";
    s += "  --prefill-until N      batched prefill positions (0 = all but the last prompt position)\n";
    s += "  --spec S               speculative verify window (0 = off, else >= 2)\n";
    s += "  --expert-cache auto|N  VRAM expert tier: auto-sized from --vram-free/--vram-reserve/--blob, or N slots\n";
    s += "  --vram-free BYTES  --vram-reserve BYTES  --blob BYTES  --profile-len N\n";
    s += "  --pcie F               PCIe share in [0,1] (sent as numerator/256)\n";
    s += "  --trace                print per-layer routing trace index\n";
    s += "  --serve                request loop on stdin: 'GEN <max_new> <ids>' | 'QUIT'\n";
    s += "  --vocab V              synthetic backend vocabulary (default 257)\n";
    s += "  --workers N            worker threads for the request pool (default 1)\n";
    s += "  --gpus a,b             device list\n";
    s += "  -h, --help             this text\n";
    return s;
}

// Parses argv[1..]. On failure returns false and sets `err`; validation of values is done by validate().
inline bool parse_args(int argc, const char* const* argv, Options& o, std::string& err) {
    err.clear();
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](std::string& out) -> bool {
            if (i + 1 >= argc) { err = "missing value for " + a; return false; }
            out = argv[++i];
            return true;
        };
        std::string v;
        auto num = [&](int64_t& dst) -> bool {
            if (!val(v)) return false;
            if (!parse_i64(v, dst)) { err = "bad integer for " + a + ": " + v; return false; }
            return true;
        };
        if (a == "-h" || a == "--help") o.help = true;
        else if (a == "--serve") o.serve = true;
        else if (a == "--trace") o.trace = true;
        else if (a == "--prompt-ids") {
            if (!val(v)) return false;
            if (!parse_i64_list(v, o.prompt)) { err = "bad id list for --prompt-ids: " + v; return false; }
        } else if (a == "--gpus") {
            if (!val(v)) return false;
            if (!parse_i64_list(v, o.gpus)) { err = "bad list for --gpus: " + v; return false; }
        } else if (a == "--max-new") { if (!num(o.max_new)) return false; }
        else if (a == "--max-context") { if (!num(o.max_context)) return false; }
        else if (a == "--prefill-until") { if (!num(o.prefill_until)) return false; }
        else if (a == "--spec") { if (!num(o.spec)) return false; }
        else if (a == "--vram-free") { if (!num(o.vram_free)) return false; }
        else if (a == "--vram-reserve") { if (!num(o.vram_reserve)) return false; }
        else if (a == "--blob") { if (!num(o.blob)) return false; }
        else if (a == "--profile-len") { if (!num(o.profile_len)) return false; }
        else if (a == "--vocab") { if (!num(o.vocab)) return false; }
        else if (a == "--workers") { if (!num(o.workers)) return false; }
        else if (a == "--expert-cache") {
            if (!val(v)) return false;
            if (v == "auto") { o.expert_cache_auto = true; }
            else if (parse_i64(v, o.expert_cache)) { o.expert_cache_auto = false; }
            else { err = "bad value for --expert-cache: " + v; return false; }
        } else if (a == "--pcie") {
            if (!val(v)) return false;
            char* end = nullptr;
            const double d = std::strtod(v.c_str(), &end);
            if (v.empty() || end != v.c_str() + v.size()) { err = "bad number for --pcie: " + v; return false; }
            o.pcie = d;
        } else { err = "unknown option: " + a; return false; }
    }
    return true;
}

// Returns "" if the options describe a runnable configuration, else the first problem found.
inline std::string validate(const Options& o) {
    if (o.max_new <= 0) return "--max-new must be > 0";
    if (o.max_context <= 0) return "--max-context must be > 0";
    if (o.max_new > o.max_context) return "--max-new must not exceed --max-context";
    if (!o.serve) {
        if (o.prompt.empty()) return "--prompt-ids is required (or use --serve)";
        if (!generate::size_check((uint64_t)o.prompt.size(), (uint64_t)o.max_new, (uint64_t)o.max_context))
            return "prompt does not fit: need prompt <= max-context - max-new";
    }
    if (o.prefill_until < 0) return "--prefill-until must be >= 0";
    if (o.spec < 0 || o.spec == 1) return "--spec must be 0 or >= 2";
    if (o.vocab < 2) return "--vocab must be >= 2";
    if (o.workers < 1) return "--workers must be >= 1";
    if (o.expert_cache < 0) return "--expert-cache must be >= 0";
    for (int64_t t : o.prompt) if (t < 0) return "prompt ids must be >= 0";
    return "";
}

// Expert slots implied by the options: explicit N, or auto = (free - reserve) / blob truncated to the profile length.
// Returns -1 when auto sizing lacks --vram-free or --blob.
inline int64_t resolve_slots(const Options& o) {
    if (!o.expert_cache_auto) return o.expert_cache;
    if (o.vram_free < 0 || o.blob <= 0) return -1;
    uint64_t s = generate::auto_slots((uint64_t)o.vram_free, (uint64_t)std::max<int64_t>(0, o.vram_reserve), (uint64_t)o.blob);
    if (o.profile_len >= 0) s = generate::profile_want((uint64_t)o.profile_len, s);
    return (int64_t)s;
}

// ---------------------------------------------------------------------------------------------------------------------
// argmax
// ---------------------------------------------------------------------------------------------------------------------
// Index of the largest value, lowest index on ties, NaN entries never win. -1 for n == 0; 0 if every entry is NaN.
inline int64_t argmax(const float* x, int64_t n) {
    if (n <= 0 || !x) return -1;
    int64_t best = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (x[i] != x[i]) continue;
        if (best < 0 || x[i] > x[best]) best = i;
    }
    return best < 0 ? 0 : best;
}
inline int64_t argmax(const std::vector<float>& x) { return argmax(x.data(), (int64_t)x.size()); }

// ---------------------------------------------------------------------------------------------------------------------
// the decode loop
// ---------------------------------------------------------------------------------------------------------------------
// The model backend. `step` runs one position (input token at position pos) and writes its logits row. The others are
// optional: `prefill` runs the first n prompt positions as one batch and writes n rows; `draft` proposes up to n tokens
// following `history`; `verify` runs a window of cells starting at `pos` and returns the argmax output of every cell.
struct Drive {
    std::function<void(int64_t tok, int64_t pos, std::vector<float>& logits)> step;
    std::function<void(const std::vector<int64_t>& toks, int64_t n, std::vector<std::vector<float>>& rows)> prefill;
    std::function<std::vector<int64_t>(const std::vector<int64_t>& history, int64_t n)> draft;
    std::function<std::vector<int64_t>(const std::vector<int64_t>& window, int64_t pos)> verify;
};

struct DriveResult {
    bool ok = false;
    std::string error;
    std::vector<int64_t> tokens;       // generated tokens, exactly max_new when ok
    std::vector<int64_t> row_argmax;   // argmax of every dumped logits row, in position order
    std::vector<int64_t> accepted_hist;// accepted draft count of every speculative round
    int64_t rounds = 0;
    int64_t n_batched = 0;             // prompt positions eligible for the batched path
    int64_t positions = 0;             // positions processed (token path + prefill + committed spec cells)
    double seconds = 0;
};

// Length of the longest prefix of ds equal to os (stops at the shorter list).
inline int64_t accept_prefix(const std::vector<int64_t>& ds, const std::vector<int64_t>& os) {
    size_t i = 0;
    while (i < ds.size() && i < os.size() && ds[i] == os[i]) ++i;
    return (int64_t)i;
}

inline DriveResult drive_pool(const Drive& d, const std::vector<int64_t>& prompt, const Options& o, bool serve_admit = false) {
    DriveResult r;
    const auto t0 = Clock::now();
    auto fail = [&](const std::string& m) { r.ok = false; r.error = m; r.seconds = std::chrono::duration<double>(Clock::now() - t0).count(); return r; };
    const int64_t np = (int64_t)prompt.size();
    if (np < 1) return fail("empty prompt");
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context) return fail("bad max_new/max_context");
    if (!generate::size_check((uint64_t)np, (uint64_t)o.max_new, (uint64_t)o.max_context)) return fail("prompt does not fit context");
    if (serve_admit && !generate::serve_accepts((uint64_t)np, (uint64_t)o.max_new, (uint64_t)o.max_context)) return fail("not admitted: need prompt + max_new + 8 <= max_context");
    if (!d.step) return fail("no step function");

    const int64_t dump = (int64_t)generate::dump_positions((uint64_t)np, (uint64_t)o.max_new);  // np - 1 + max_new
    const int64_t last = dump - 1;                                                              // last position processed
    const int64_t nb = (int64_t)generate::n_batched((uint64_t)np, (uint64_t)o.prefill_until);
    r.n_batched = nb;
    const bool use_spec = o.spec >= 2 && d.verify && d.draft;

    std::vector<float> logits;
    int64_t pos = 0;
    if (nb > 0 && d.prefill) {
        std::vector<std::vector<float>> rows;
        d.prefill(prompt, nb, rows);
        if ((int64_t)rows.size() != nb) return fail("prefill returned a wrong number of rows");
        for (const auto& row : rows) r.row_argmax.push_back(argmax(row));
        pos = nb;
    }
    int64_t cur = prompt[(size_t)pos];
    bool to_spec = false;
    for (; pos <= last; ++pos) {
        if (use_spec && pos == np - 1) { to_spec = true; break; }
        d.step(cur, pos, logits);
        const int64_t nx = argmax(logits);
        if (nx < 0) return fail("step produced no logits");
        r.row_argmax.push_back(nx);
        if (pos >= np - 1) r.tokens.push_back(nx);
        cur = pos + 1 < np ? prompt[(size_t)pos + 1] : nx;  // teacher forcing inside the prompt, else the sampled token
    }
    if (to_spec) {
        std::vector<int64_t> history = prompt;
        while ((int64_t)r.tokens.size() < o.max_new) {
            const int64_t remaining = o.max_new - (int64_t)r.tokens.size();
            const int64_t T = std::min<int64_t>(o.spec, remaining);
            std::vector<int64_t> window{cur};
            std::vector<int64_t> drafts;
            if (T > 1) {
                drafts = d.draft(history, T - 1);
                if ((int64_t)drafts.size() > T - 1) drafts.resize((size_t)(T - 1));
                window.insert(window.end(), drafts.begin(), drafts.end());
            }
            const std::vector<int64_t> outs = d.verify(window, pos);
            if (outs.size() != window.size()) return fail("verify returned a wrong number of outputs");
            const int64_t a = accept_prefix(drafts, outs);  // a <= drafts.size() = T' - 1
            for (int64_t i = 0; i <= a; ++i) { r.tokens.push_back(outs[(size_t)i]); history.push_back(outs[(size_t)i]); }
            r.accepted_hist.push_back(a);
            ++r.rounds;
            pos += a + 1;
            cur = outs[(size_t)a];
        }
    }
    r.positions = pos;
    r.ok = (int64_t)r.tokens.size() == o.max_new;
    if (!r.ok) r.error = "internal: produced token count differs from max_new";
    r.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return r;
}

// Runs every prompt as its own request over `workers` threads; each worker builds its own backend via make(worker).
// Results come back in request order whatever the scheduling.
inline std::vector<DriveResult> drive_pool_multi(const std::function<Drive(int)>& make,
                                                 const std::vector<std::vector<int64_t>>& prompts, const Options& o,
                                                 int64_t workers, bool serve_admit = false) {
    std::vector<DriveResult> out(prompts.size());
    if (prompts.empty()) return out;
    int64_t nw = std::max<int64_t>(1, std::min<int64_t>(workers, (int64_t)prompts.size()));
    std::atomic<size_t> next{0};
    auto work = [&](int w) {
        bool have = false;
        Drive d;
        std::string build_err;
        try { d = make(w); have = true; } catch (const std::exception& e) { build_err = e.what(); } catch (...) { build_err = "backend construction failed"; }
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= prompts.size()) break;
            if (!have) { out[i].ok = false; out[i].error = build_err; continue; }
            try { out[i] = drive_pool(d, prompts[i], o, serve_admit); }
            catch (const std::exception& e) { out[i] = DriveResult(); out[i].error = e.what(); }
            catch (...) { out[i] = DriveResult(); out[i].error = "unknown failure"; }
        }
    };
    if (nw == 1) { work(0); return out; }
    std::vector<std::thread> th;
    for (int64_t w = 0; w < nw; ++w) th.emplace_back(work, (int)w);
    for (auto& t : th) t.join();
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// synthetic backend: a deterministic first-order model, next(tok) = (31 tok + 7) mod V, with a draft source that is wrong
// at a fixed cadence so the speculative loop sees full, partial and zero acceptance.
// ---------------------------------------------------------------------------------------------------------------------
inline int64_t synth_next(int64_t tok, int64_t vocab) {
    const int64_t t = ((tok % vocab) + vocab) % vocab;
    return (t * 31 + 7) % vocab;
}

inline Drive synthetic_drive(int64_t vocab, int64_t wrong_every = 5) {
    Drive d;
    d.step = [vocab](int64_t tok, int64_t, std::vector<float>& lg) {
        lg.assign((size_t)vocab, 0.0f);
        lg[(size_t)synth_next(tok, vocab)] = 1.0f;
    };
    d.prefill = [vocab](const std::vector<int64_t>& toks, int64_t n, std::vector<std::vector<float>>& rows) {
        rows.assign((size_t)n, std::vector<float>((size_t)vocab, 0.0f));
        for (int64_t i = 0; i < n; ++i) rows[(size_t)i][(size_t)synth_next(toks[(size_t)i], vocab)] = 1.0f;
    };
    d.verify = [vocab](const std::vector<int64_t>& w, int64_t) {
        std::vector<int64_t> o;
        for (int64_t t : w) o.push_back(synth_next(t, vocab));
        return o;
    };
    d.draft = [vocab, wrong_every](const std::vector<int64_t>& hist, int64_t n) {
        std::vector<int64_t> ds;
        int64_t t = hist.empty() ? 0 : hist.back();
        for (int64_t j = 0; j < n; ++j) {
            int64_t g = synth_next(t, vocab);
            if (wrong_every > 0 && (((int64_t)hist.size() + j) % wrong_every) == wrong_every - 1) g = (g + 1) % vocab;
            ds.push_back(g);
            t = g;  // later drafts continue from this draft; a wrong draft makes the rest wrong too
        }
        return ds;
    };
    return d;
}

// ---------------------------------------------------------------------------------------------------------------------
// --serve: line protocol. "GEN <max_new> <id,id,...>" -> "OK <n> <id,id,...>" or "ERR <reason>"; "QUIT" stops.
// Returns the number of requests answered (OK or ERR).
// ---------------------------------------------------------------------------------------------------------------------
inline int64_t serve_loop(std::istream& in, std::ostream& out, const Options& o, const std::function<Drive(int)>& make) {
    int64_t answered = 0;
    Drive d = make(0);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line == "QUIT") break;
        std::istringstream ss(line);
        std::string cmd, mn, ids;
        ss >> cmd >> mn;
        std::getline(ss >> std::ws, ids);
        ++answered;
        int64_t max_new = 0;
        std::vector<int64_t> prompt;
        if (cmd != "GEN") { out << "ERR unknown command\n"; continue; }
        if (!parse_i64(mn, max_new)) { out << "ERR bad max_new\n"; continue; }
        if (!parse_i64_list(ids, prompt) || prompt.empty()) { out << "ERR bad prompt\n"; continue; }
        bool neg = false;
        for (int64_t t : prompt) if (t < 0) neg = true;
        if (neg) { out << "ERR bad prompt\n"; continue; }
        Options r = o;
        r.max_new = max_new;
        const DriveResult res = drive_pool(d, prompt, r, /*serve_admit=*/true);
        if (!res.ok) { out << "ERR " << res.error << "\n"; continue; }
        out << "OK " << res.tokens.size() << " " << join_i64(res.tokens) << "\n";
    }
    out.flush();
    return answered;
}

}  // namespace gencli
}  // namespace baloon
