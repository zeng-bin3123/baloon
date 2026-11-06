// Invariants of include/baloon/direct_file.hpp and include/baloon/resident_memory.hpp.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include "baloon/direct_file.hpp"
#include "baloon/resident_memory.hpp"
#include "check.hpp"
using namespace baloon;
using dfio::DirectFile; using dfio::Completion;

static uint8_t pat(uint64_t i) { return (uint8_t)((i * 131 + (i >> 8) * 7 + 3) & 0xFF); }

static std::string make_file(uint64_t n) {
    char tmpl[] = "./k23_XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd < 0) { std::fprintf(stderr, "mkstemp failed\n"); std::exit(2); }
    std::vector<uint8_t> v(n); for (uint64_t i = 0; i < n; ++i) v[i] = pat(i);
    if (::write(fd, v.data(), n) != (ssize_t)n) { std::fprintf(stderr, "write failed\n"); std::exit(2); }
    ::close(fd);
    return tmpl;
}
static void* aligned(uint64_t n) { void* p = nullptr; if (posix_memalign(&p, 4096, n)) std::exit(2); std::memset(p, 0xEE, n); return p; }

static void test_clock() {
    int64_t prev = dfio::now_us();
    for (int i = 0; i < 1000; ++i) { int64_t t = dfio::now_us(); CHECK(t >= prev); prev = t; }
    int64_t a = dfio::now_us(); ::usleep(20000); int64_t b = dfio::now_us();
    CHECK(b - a >= 19000);          // a 20 ms sleep is measured as at least ~20 ms
    CHECK(b - a < 5000000);         // and not seconds
}

static void test_direct_file() {
    const uint64_t N = 3 * 4096 + 1000;     // not a multiple of the alignment: last block is short
    const std::string path = make_file(N);
    DirectFile f;
    CHECK(!f.is_open() && f.size() == 0 && !f.direct());
    CHECK(!f.open("./k23_does_not_exist") && !f.is_open());
    uint8_t* buf = (uint8_t*)aligned(4 * 4096);
    Completion c[8];
    CHECK(!f.submit(1, 0, buf, 4096));      // closed file rejects
    CHECK(f.open(path.c_str()) && f.is_open() && f.size() == N);
    // rejections queue nothing
    CHECK(!f.submit(1, 1, buf, 4096));
    CHECK(!f.submit(2, 0, buf + 8, 4096));
    CHECK(!f.submit(3, 0, buf, 100));
    CHECK(!f.submit(4, 0, buf, 0));
    CHECK(!f.submit(5, 4096 + 512, buf, 4096));
    CHECK(f.pending() == 0 && f.wait(c, 8) == 0);
    // accepted reads: content and bytes
    CHECK(f.submit(10, 4096, buf, 4096));
    CHECK(f.submit(11, 0, buf + 4096, 4096));
    CHECK(f.pending() == 2);
    CHECK(f.wait(c, 1) == 1 && c[0].tag == 10 && c[0].bytes == 4096 && c[0].ok);     // bound respected, oldest first
    CHECK(f.pending() == 1);
    CHECK(f.wait(c, 8) == 1 && c[0].tag == 11 && c[0].bytes == 4096 && c[0].ok);
    bool same = true;
    for (uint64_t i = 0; i < 4096; ++i) { if (buf[i] != pat(4096 + i)) same = false; if (buf[4096 + i] != pat(i)) same = false; }
    CHECK(same);
    // short read at end of file is ok with the remaining bytes; fully past EOF is ok with 0
    CHECK(f.submit(12, 3 * 4096, buf, 2 * 4096));
    CHECK(f.wait(c, 8) == 1 && c[0].tag == 12 && c[0].ok && c[0].bytes == 1000);
    same = true; for (uint64_t i = 0; i < 1000; ++i) if (buf[i] != pat(3 * 4096 + i)) same = false;
    CHECK(same);
    CHECK(f.submit(13, 8 * 4096, buf, 4096));
    CHECK(f.wait(c, 8) == 1 && c[0].tag == 13 && c[0].ok && c[0].bytes == 0);
    CHECK(f.wait(c, 0) == 0);
    // move keeps the state; close resets
    DirectFile g(std::move(f));
    CHECK(!f.is_open() && g.is_open() && g.size() == N);
    CHECK(g.submit(14, 0, buf, 4096) && g.pending() == 1);
    g.close();
    CHECK(!g.is_open() && g.pending() == 0 && g.size() == 0);
    std::free(buf);
    ::unlink(path.c_str());
}

static void test_lock() {
    using namespace resident;
    // rejected ranges
    LockResult r = lock_resident(nullptr, 4096);
    CHECK(!r.ok && r.err == EINVAL && r.bytes == 0);
    r = unlock_resident(nullptr, 4096);
    CHECK(!r.ok && r.err == EINVAL && r.bytes == 0);
    int x = 0;
    r = lock_resident(&x, ~0ull);                         // wraps the address space
    CHECK(!r.ok && r.err == EINVAL);
    // empty ranges succeed trivially, even with a null pointer
    r = lock_resident(nullptr, 0);
    CHECK(r.ok && r.err == 0 && r.bytes == 0);
    r = unlock_resident(nullptr, 0);
    CHECK(r.ok && r.err == 0 && r.bytes == 0);
    // a real range: result is self-consistent; when locked the pages are resident
    const uint64_t n = 4 * 4096;
    uint8_t* p = (uint8_t*)aligned(n);
    r = lock_resident(p, n);
    CHECK(r.ok == (r.err == 0));
    CHECK(r.ok ? r.bytes == n : (r.bytes == 0 && (r.err == ENOMEM || r.err == EPERM || r.err == EAGAIN)));
    if (r.ok) {
        unsigned char vec[4];
        CHECK(::mincore(p, n, vec) == 0);
        for (int i = 0; i < 4; ++i) CHECK(vec[i] & 1);
        LockResult u = unlock_resident(p, n);
        CHECK(u.ok && u.err == 0 && u.bytes == n);
    } else std::printf("note: mlock refused (errno %d); residency check skipped\n", r.err);
    std::free(p);
    // an unmapped range is refused by the kernel with ENOMEM; the failure is reported, never swallowed
    void* m = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(m != MAP_FAILED);
    if (m != MAP_FAILED) {
        ::munmap(m, 4096);
        r = lock_resident(m, 4096);
        CHECK(!r.ok && r.err == ENOMEM && r.bytes == 0);
        r = unlock_resident(m, 4096);
        CHECK(!r.ok && r.err == ENOMEM && r.bytes == 0);
    }
}

int main() {
    test_clock();
    test_direct_file();
    test_lock();
    std::printf("%ld checks, %ld failed\n", th::checks(), th::fails());
    if (th::fails()) return 1;
    std::printf("UNIT 23 TESTS PASSED\n");
    return 0;
}
