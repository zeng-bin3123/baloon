// Unit 34 invariants: the worker-thread DirectFile and the resident-memory calls of src/platform.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "baloon/platform_io.hpp"

using namespace baloon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static std::string make_file(uint64_t bytes) {
    char tmpl[] = "./k34_XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd < 0) { std::printf("mkstemp failed\n"); std::exit(2); }
    std::vector<unsigned char> d(bytes);
    for (uint64_t i = 0; i < bytes; ++i) d[i] = (unsigned char)((i * 131 + (i >> 12) * 7 + 3) & 0xFF);
    if (::write(fd, d.data(), d.size()) != (ssize_t)d.size()) std::exit(2);
    ::close(fd);
    return tmpl;
}
static unsigned char pat(uint64_t i) { return (unsigned char)((i * 131 + (i >> 12) * 7 + 3) & 0xFF); }

static long vmlck_kb() {   // locked memory of this process, from the kernel (-1 when unavailable)
    FILE* f = std::fopen("/proc/self/status", "r"); if (!f) return -1;
    char line[256]; long v = -1;
    while (std::fgets(line, sizeof line, f)) if (std::sscanf(line, "VmLck: %ld kB", &v) == 1) break;
    std::fclose(f); return v;
}

int main() {
    // clock: monotonic, microsecond scale
    { const int64_t a = platform::now_us(); ::usleep(20000); const int64_t b = platform::now_us();
      CHECK(b >= a); CHECK(b - a >= 15000); CHECK(b - a < 2000000); }

    // aligned allocation
    { void* p = platform::DirectFile::alloc_aligned(1);
      CHECK(p != nullptr); CHECK(((uintptr_t)p % 4096) == 0); std::memset(p, 1, 4096); platform::DirectFile::free_aligned(p);
      CHECK(platform::DirectFile::alloc_aligned(0) == nullptr);
      CHECK(platform::DirectFile::alloc_aligned(UINT64_MAX) == nullptr);
      platform::DirectFile::free_aligned(nullptr); }

    const uint64_t kSize = 4096 * 5 + 100;  // not a multiple of the alignment: the tail read is short
    const std::string path = make_file(kSize);
    {
        platform::DirectFile f;
        CHECK(!f.is_open()); CHECK(f.size() == 0);
        CHECK(!f.open("./no_such_file_k34")); CHECK(!f.is_open());
        CHECK(f.open(path.c_str())); CHECK(f.is_open()); CHECK(f.size() == kSize);

        unsigned char* buf = (unsigned char*)platform::DirectFile::alloc_aligned(4096 * 8);
        // rejection: misaligned offset / length / buffer, zero length -> false, nothing queued
        CHECK(!f.submit(1, 1, buf, 4096)); CHECK(!f.submit(2, 0, buf, 100)); CHECK(!f.submit(3, 0, buf + 8, 4096));
        CHECK(!f.submit(4, 0, buf, 0)); CHECK(f.in_flight() == 0);
        platform::Completion c[8];
        CHECK(f.wait(c, 8, 20) == 0);                                   // timeout, nothing ready

        // ordered completions with correct data
        CHECK(f.submit(10, 0, buf, 4096)); CHECK(f.submit(11, 4096, buf + 4096, 4096)); CHECK(f.submit(12, 8192, buf + 8192, 4096));
        uint64_t got = 0; platform::Completion all[3];
        while (got < 3) { const uint64_t n = f.wait(all + got, 3 - got, 2000); if (n == 0) break; got += n; }
        CHECK(got == 3);
        for (int i = 0; i < 3 && got == 3; ++i) { CHECK(all[i].tag == (uint64_t)(10 + i)); CHECK(all[i].ok); CHECK(all[i].bytes == 4096); }
        for (uint64_t i = 0; i < 3 * 4096; ++i) if (buf[i] != pat(i)) { CHECK(!"data mismatch"); break; }
        CHECK(f.in_flight() == 0);

        // max bound: wait never returns more than max
        for (uint64_t t = 0; t < 4; ++t) CHECK(f.submit(20 + t, t * 4096, buf, 4096));
        ::usleep(100000);
        uint64_t total = 0; int rounds = 0;
        while (total < 4 && rounds++ < 10) { const uint64_t n = f.wait(c, 2, 1000); CHECK(n <= 2); total += n; }
        CHECK(total == 4);

        // short read at EOF: ok with fewer bytes; read past EOF: ok with 0 bytes
        CHECK(f.submit(30, 4096 * 5, buf, 4096)); CHECK(f.submit(31, 4096 * 7, buf, 4096));
        uint64_t k = 0; platform::Completion e[2];
        while (k < 2) { const uint64_t n = f.wait(e + k, 2 - k, 2000); if (!n) break; k += n; }
        CHECK(k == 2);
        if (k == 2) { CHECK(e[0].tag == 30 && e[0].ok && e[0].bytes == 100); CHECK(e[1].tag == 31 && e[1].ok && e[1].bytes == 0); }
        for (uint64_t i = 0; i < 100; ++i) if (buf[i] != pat(4096 * 5 + i)) { CHECK(!"tail data mismatch"); break; }

        // wake: a blocked wait returns promptly with 0 completions
        { uint64_t r = 99; const int64_t t0 = platform::now_us();
          std::thread th([&] { platform::Completion x[1]; r = f.wait(x, 1, -1); });
          ::usleep(100000); f.wake(); th.join();
          CHECK(r == 0); CHECK(platform::now_us() - t0 < 5000000); }
        // wake is sticky: the next wait returns at once, and is consumed afterwards
        { f.wake(); const int64_t t0 = platform::now_us(); CHECK(f.wait(c, 1, -1) == 0); CHECK(platform::now_us() - t0 < 1000000);
          const int64_t t1 = platform::now_us(); CHECK(f.wait(c, 1, 100) == 0); CHECK(platform::now_us() - t1 >= 80000); }
        // wait with max 0 / null out does nothing
        CHECK(f.wait(c, 0, 10) == 0); CHECK(f.wait(nullptr, 4, 10) == 0);

        // close: back to closed state, submit rejected, size 0; reopen works
        CHECK(f.submit(40, 0, buf, 4096));
        f.close(); CHECK(!f.is_open()); CHECK(f.size() == 0); CHECK(!f.submit(41, 0, buf, 4096)); CHECK(f.in_flight() == 0);
        CHECK(f.open(path.c_str()) && f.size() == kSize);
        CHECK(f.submit(50, 0, buf, 4096));
        CHECK(f.wait(c, 1, 2000) == 1 && c[0].tag == 50 && c[0].ok);
        platform::DirectFile::free_aligned(buf);
    }
    // destructor with work in flight must not hang or crash
    { platform::DirectFile f; CHECK(f.open(path.c_str()));
      void* b = platform::DirectFile::alloc_aligned(4096); for (int i = 0; i < 8; ++i) f.submit(i, 0, b, 4096);
      f.close(); platform::DirectFile::free_aligned(b); }
    ::unlink(path.c_str());

    // resident memory
    {
        char* p = (char*)platform::DirectFile::alloc_aligned(4096 * 4);
        resident::LockResult r = platform::lock_resident(p, 4096 * 4);
        if (r.ok) { const long locked = vmlck_kb();
          if (locked >= 0) CHECK(locked >= 16);
          CHECK(r.err == 0); CHECK(r.bytes == 4096 * 4); resident::LockResult u = platform::unlock_resident(p, 4096 * 4); CHECK(u.ok && u.bytes == 4096 * 4);
          const long after = vmlck_kb(); if (after >= 0 && locked >= 0) CHECK(after <= locked - 16); }
        else { CHECK(r.err != 0); CHECK(r.bytes == 0); }   // RLIMIT_MEMLOCK may forbid it; the failure must be reported, not hidden
        resident::LockResult z = platform::lock_resident(nullptr, 0); CHECK(z.ok && z.err == 0 && z.bytes == 0);
        z = platform::unlock_resident(nullptr, 0); CHECK(z.ok && z.bytes == 0);
        resident::LockResult nl = platform::lock_resident(nullptr, 4096); CHECK(!nl.ok && nl.err == EINVAL && nl.bytes == 0);
        nl = platform::unlock_resident(nullptr, 4096); CHECK(!nl.ok && nl.err == EINVAL);
        resident::LockResult wr = platform::lock_resident(p, UINT64_MAX); CHECK(!wr.ok && wr.err == EINVAL && wr.bytes == 0);
        wr = platform::unlock_resident(p, UINT64_MAX); CHECK(!wr.ok && wr.err == EINVAL);
        platform::DirectFile::free_aligned(p);
    }
    if (g_fail) { std::printf("%d CHECK(S) FAILED\n", g_fail); return 1; }
    std::printf("UNIT 34 TESTS PASSED\n");
    return 0;
}
