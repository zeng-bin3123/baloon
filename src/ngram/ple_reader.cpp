// ple_reader.cpp - asynchronous page-grouped reader of the PLE table (see include/baloon/ple_reader.hpp).
#include "baloon/ple_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace baloon {
namespace pler {

static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t ReaderStats::percentile(double q) const {
    if (latency_ns.empty()) return 0;
    std::vector<uint64_t> c(latency_ns);
    std::sort(c.begin(), c.end());
    const double qq = q < 0.0 ? 0.0 : (q > 1.0 ? 1.0 : q);
    const uint64_t r = (uint64_t)std::llround(qq * (double)(c.size() - 1));
    return c[pct_index(c.size(), r)];
}

PleReader::PleReader() : ring_(LATENCY_RING, 0) {}
PleReader::~PleReader() { close(); }

bool PleReader::is_open() const { std::lock_guard<std::mutex> g(mu_); return open_; }

bool PleReader::open(const char* path, uint64_t table_off, uint64_t n_rows, uint64_t cache_rows, uint32_t max_inflight) {
    close();
    if (!path || max_inflight == 0) return false;
    int fd = ::open(path, O_RDONLY | O_DIRECT);  // unbuffered when the file system allows it
    if (fd < 0) fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat sb;
    if (::fstat(fd, &sb) != 0 || (uint64_t)sb.st_size < table_off || n_rows > ((uint64_t)sb.st_size - table_off) / ROW_BYTES) {
        ::close(fd);  // the whole table must lie inside the file
        return false;
    }
    void* slab = nullptr;
    if (posix_memalign(&slab, PAGE, (size_t)max_inflight * SLOT_BYTES) != 0) { ::close(fd); return false; }
    {
        std::lock_guard<std::mutex> g(mu_);
        fd_ = fd; table_off_ = table_off; n_rows_ = n_rows; max_inflight_ = max_inflight;
        slab_ = (uint8_t*)slab;
        free_.clear();
        for (uint32_t s = max_inflight; s-- > 0;) free_.push_back(s);
        queue_.clear(); inflight_.clear(); tickets_.clear();
        cache_ = RowCache(cache_rows);
        std::fill(ring_.begin(), ring_.end(), 0); ring_pos_ = 0;
        st_ = ReaderStats();
        st_.free_slots = max_inflight; st_.max_inflight = max_inflight;
        stop_ = false; open_ = true;
    }
    th_ = std::thread([this] { worker(); });
    return true;
}

void PleReader::close() {
    {
        std::lock_guard<std::mutex> g(mu_);
        if (!open_) return;
        stop_ = true;
    }
    cv_work_.notify_all();
    if (th_.joinable()) th_.join();
    std::lock_guard<std::mutex> g(mu_);
    for (auto& kv : tickets_) { kv.second.pending = 0; kv.second.ok = false; }  // release waiting collectors
    queue_.clear(); inflight_.clear();
    open_ = false;
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    std::free(slab_); slab_ = nullptr;
    cv_done_.notify_all();
}

void PleReader::pump_locked() {
    while (!queue_.empty() && !free_.empty()) {
        const uint32_t s = free_.back(); free_.pop_back();
        inflight_.push_back({std::move(queue_.front()), s});
        queue_.pop_front();
        --st_.free_slots; ++st_.inflight;  // free + inflight stays max_inflight
    }
    if (!inflight_.empty()) cv_work_.notify_one();
}

uint32_t PleReader::issue(const uint64_t* rows, size_t n, uint8_t* out) {
    if ((n && (!rows || !out))) return 0;
    std::lock_guard<std::mutex> g(mu_);
    if (!open_) return 0;
    for (size_t i = 0; i < n; ++i) if (rows[i] >= n_rows_) return 0;
    uint32_t t = next_ticket(next_ticket_);
    while (tickets_.count(t)) t = next_ticket(t);
    next_ticket_ = t;
    std::map<uint64_t, Job> jobs;  // page-aligned first byte -> deduplicated job
    for (size_t i = 0; i < n; ++i) {
        ++st_.rows_requested;
        uint8_t* dst = out + i * ROW_BYTES;
        if (cache_.lookup(rows[i], dst)) { ++st_.cache_hits; continue; }
        const uint64_t at = table_off_ + rows[i] * ROW_BYTES;
        const uint64_t f = first_of(at), l = length_of(at);
        Job& j = jobs[f];
        if (j.uses.empty()) { j.first = f; j.length = l; } else j.length = std::max(j.length, l);
        j.uses.push_back({rows[i], (uint32_t)(at - f), t, dst});
    }
    TicketState ts;
    ts.pending = jobs.size();
    tickets_[t] = ts;
    const uint64_t t0 = now_ns();
    for (auto& kv : jobs) {
        kv.second.t_issue_ns = t0;
        queue_.push_back(std::move(kv.second));
        ++st_.jobs_issued;
    }
    pump_locked();
    return t;
}

bool PleReader::collect(uint32_t ticket) {
    std::unique_lock<std::mutex> lk(mu_);
    auto it = tickets_.find(ticket);
    if (it == tickets_.end()) return false;
    cv_done_.wait(lk, [&] { auto i2 = tickets_.find(ticket); return i2 == tickets_.end() || i2->second.pending == 0; });
    it = tickets_.find(ticket);
    if (it == tickets_.end()) return false;
    const bool ok = it->second.ok;
    tickets_.erase(it);
    return ok;
}

void PleReader::worker() {
    for (;;) {
        Inflight cur;
        uint32_t delay;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [&] { return stop_ || !inflight_.empty(); });
            if (stop_) return;
            cur = std::move(inflight_.front());
            inflight_.pop_front();
            delay = delay_us_;
        }
        if (delay) std::this_thread::sleep_for(std::chrono::microseconds(delay));
        uint8_t* buf = slab_ + slot_off(cur.slot);  // disjoint 8192-byte window of this slot
        ssize_t got = ::pread(fd_, buf, (size_t)cur.job.length, (off_t)cur.job.first);
        if (got < 0 && errno == EINVAL) {  // file system refused the unbuffered read: continue buffered
            const int fl = ::fcntl(fd_, F_GETFL);
            if (fl >= 0 && (fl & O_DIRECT)) {
                ::fcntl(fd_, F_SETFL, fl & ~O_DIRECT);
                got = ::pread(fd_, buf, (size_t)cur.job.length, (off_t)cur.job.first);
            }
        }
        bool ok = got >= 0;
        if (ok) for (const Use& u : cur.job.uses) if ((uint64_t)got < (uint64_t)u.in_page + ROW_BYTES) ok = false;  // short read
        if (ok) for (const Use& u : cur.job.uses) std::memcpy(u.dst, buf + u.in_page, ROW_BYTES);
        const uint64_t lat = now_ns() - cur.job.t_issue_ns;
        std::lock_guard<std::mutex> g(mu_);
        if (!open_ || fd_ < 0) continue;
        if (ok) for (const Use& u : cur.job.uses) cache_.insert(u.row, u.dst);
        ring_[ring_index(ring_pos_)] = lat; ++ring_pos_;
        free_.push_back(cur.slot);
        ++st_.free_slots; --st_.inflight;
        ++st_.jobs_done;
        if (ok) st_.bytes_read += (uint64_t)got; else ++st_.errors;
        if (!cur.job.uses.empty()) {
            auto it = tickets_.find(cur.job.uses[0].ticket);
            if (it != tickets_.end()) { if (it->second.pending) --it->second.pending; if (!ok) it->second.ok = false; }
        }
        pump_locked();
        cv_done_.notify_all();
    }
}

uint64_t PleReader::cache_capacity() const { std::lock_guard<std::mutex> g(mu_); return cache_.capacity(); }
uint64_t PleReader::cache_size() const { std::lock_guard<std::mutex> g(mu_); return cache_.resident(); }

ReaderStats PleReader::stats() const {
    std::lock_guard<std::mutex> g(mu_);
    ReaderStats s = st_;
    const uint64_t n = ring_pos_ < LATENCY_RING ? ring_pos_ : LATENCY_RING;
    s.latency_ns.assign(ring_.begin(), ring_.begin() + (std::ptrdiff_t)n);
    return s;
}

void PleReader::reset_stats() {
    std::lock_guard<std::mutex> g(mu_);
    const uint64_t fr = st_.free_slots, inf = st_.inflight, mx = st_.max_inflight;
    st_ = ReaderStats();
    st_.free_slots = fr; st_.inflight = inf; st_.max_inflight = mx;
    ring_pos_ = 0;
}

void PleReader::set_injected_delay_us(uint32_t us) { std::lock_guard<std::mutex> g(mu_); delay_us_ = us; }
void PleReader::debug_set_next_ticket(uint32_t t) { std::lock_guard<std::mutex> g(mu_); next_ticket_ = t; }

}  // namespace pler
}  // namespace baloon
