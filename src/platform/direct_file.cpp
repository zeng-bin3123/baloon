// direct_file.cpp - worker-thread implementation of baloon::platform::DirectFile (see include/baloon/platform_io.hpp).
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

#include "baloon/platform_io.hpp"

namespace baloon {
namespace platform {

int64_t now_us() {
    return (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

namespace {
struct Req { uint64_t tag; uint64_t offset; void* buf; uint64_t length; };

#ifdef _WIN32
using Handle = HANDLE;
inline Handle bad_handle() { return INVALID_HANDLE_VALUE; }
inline bool valid(Handle h) { return h != INVALID_HANDLE_VALUE; }
#else
using Handle = int;
inline Handle bad_handle() { return -1; }
inline bool valid(Handle h) { return h >= 0; }
#endif

// Reads up to `length` bytes at `offset`; returns bytes read or -1 on error.
int64_t os_read(Handle h, void* buf, uint64_t length, uint64_t offset) {
#ifdef _WIN32
    OVERLAPPED ov; std::memset(&ov, 0, sizeof ov);
    ov.Offset = dfio::low_word(offset); ov.OffsetHigh = dfio::high_word(offset);
    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)length, &got, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        return -1;
    }
    return (int64_t)got;
#else
    return (int64_t)::pread(h, buf, (size_t)length, (off_t)offset);
#endif
}
}  // namespace

struct DirectFile::Impl {
    Handle h = bad_handle();
    uint64_t size = 0;
    bool direct = false;
    bool open = false;

    mutable std::mutex m;
    std::condition_variable cv_req, cv_done;
    std::deque<Req> reqs;
    std::deque<Completion> done;
    uint64_t in_flight = 0;       // queued + running + undelivered
    bool stop = false;
    bool woken = false;
    std::thread worker;

    void run() {
        for (;;) {
            Req r;
            {
                std::unique_lock<std::mutex> lk(m);
                cv_req.wait(lk, [&] { return stop || !reqs.empty(); });
                if (stop) return;
                r = reqs.front(); reqs.pop_front();
            }
            const int64_t got = os_read(h, r.buf, r.length, r.offset);
            {
                std::lock_guard<std::mutex> lk(m);
                done.push_back({r.tag, (uint64_t)(got < 0 ? 0 : got), got >= 0});
            }
            cv_done.notify_all();
        }
    }
};

DirectFile::DirectFile() : p_(new Impl) {}
DirectFile::~DirectFile() { close(); }

bool DirectFile::open(const char* path) {
    close();
    if (path == nullptr) return false;
    Impl& s = *p_;
    Handle h = bad_handle();
    bool direct = false;
#ifdef _WIN32
    h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    if (valid(h)) direct = true;
    else h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (!valid(h)) return false;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz)) { CloseHandle(h); return false; }
    s.size = (uint64_t)sz.QuadPart;
#else
#ifdef O_DIRECT
    h = ::open(path, O_RDONLY | O_DIRECT);
    if (valid(h)) direct = true;
#endif
    if (!valid(h)) h = ::open(path, O_RDONLY);
    if (!valid(h)) return false;
    struct stat st;
    if (::fstat(h, &st) != 0) { ::close(h); return false; }
    s.size = (uint64_t)st.st_size;
#endif
    s.h = h; s.direct = direct; s.open = true;
    {
        std::lock_guard<std::mutex> lk(s.m);
        s.stop = false; s.woken = false; s.in_flight = 0;
        s.reqs.clear(); s.done.clear();
    }
    s.worker = std::thread([&s] { s.run(); });
    return true;
}

void DirectFile::close() {
    Impl& s = *p_;
    if (!s.open) return;
    {
        std::lock_guard<std::mutex> lk(s.m);
        s.stop = true;
    }
    s.cv_req.notify_all();
    s.cv_done.notify_all();
    if (s.worker.joinable()) s.worker.join();
#ifdef _WIN32
    CloseHandle(s.h);
#else
    ::close(s.h);
#endif
    s.h = bad_handle(); s.size = 0; s.direct = false; s.open = false;
    std::lock_guard<std::mutex> lk(s.m);
    s.reqs.clear(); s.done.clear(); s.in_flight = 0; s.woken = false;
}

bool DirectFile::is_open() const { return p_->open; }
uint64_t DirectFile::size() const { return p_->size; }
bool DirectFile::direct() const { return p_->direct; }

bool DirectFile::submit(uint64_t tag, uint64_t offset, void* buf, uint64_t length) {
    Impl& s = *p_;
    if (!s.open || !dfio::acceptable(offset, (uint64_t)(uintptr_t)buf, length)) return false;
    {
        std::lock_guard<std::mutex> lk(s.m);
        s.reqs.push_back({tag, offset, buf, length});
        ++s.in_flight;
    }
    s.cv_req.notify_one();
    return true;
}

uint64_t DirectFile::wait(Completion* out, uint64_t max, int timeout_ms) {
    Impl& s = *p_;
    if (out == nullptr || max == 0) return 0;
    std::unique_lock<std::mutex> lk(s.m);
    auto ready = [&] { return !s.done.empty() || s.woken || s.stop; };
    if (timeout_ms < 0) s.cv_done.wait(lk, ready);
    else s.cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    s.woken = false;
    const uint64_t n = dfio::drain_count(s.done.size(), max);
    for (uint64_t i = 0; i < n; ++i) { out[i] = s.done.front(); s.done.pop_front(); }
    s.in_flight -= n;
    return n;
}

void DirectFile::wake() {
    Impl& s = *p_;
    {
        std::lock_guard<std::mutex> lk(s.m);
        s.woken = true;
    }
    s.cv_done.notify_all();
}

uint64_t DirectFile::in_flight() const {
    std::lock_guard<std::mutex> lk(p_->m);
    return p_->in_flight;
}

void* DirectFile::alloc_aligned(uint64_t length) {
    if (length == 0) return nullptr;
    const uint64_t a = dfio::kAlign;
    if (length > UINT64_MAX - (a - 1)) return nullptr;
    const uint64_t n = (length + a - 1) / a * a;
    if (n > (uint64_t)SIZE_MAX) return nullptr;
#ifdef _WIN32
    return _aligned_malloc((size_t)n, (size_t)a);
#else
    void* p = nullptr;
    if (::posix_memalign(&p, (size_t)a, (size_t)n) != 0) return nullptr;
    return p;
#endif
}

void DirectFile::free_aligned(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

}  // namespace platform
}  // namespace baloon
