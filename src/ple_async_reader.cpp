// ple_async_reader.cpp - see include/baloon/ple_async_reader.hpp.
#include "baloon/ple_async_reader.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace baloon {
namespace pleasync {

// ---- row cache -----------------------------------------------------------------------------------------------------
void RowCache::init(uint64_t rows) {
    sets = rows / WAYS;
    tag.assign(sets * WAYS, EMPTY);
    data.assign(sets * WAYS * kRowBytes, 0);
    next_way.assign(sets, 0);
}
uint64_t RowCache::set_of(uint64_t row) const {
    row ^= row >> 33; row *= 0xff51afd7ed558ccdull; row ^= row >> 33;
    return sets ? row % sets : 0;
}
const uint8_t* RowCache::find(uint64_t row) const {
    if (!sets) return nullptr;
    const uint64_t s = set_of(row);
    for (int w = 0; w < WAYS; ++w)
        if (tag[s * WAYS + w] == row) return &data[(s * WAYS + w) * kRowBytes];
    return nullptr;
}
void RowCache::put(uint64_t row, const uint8_t* bytes) {
    if (!sets) return;
    const uint64_t s = set_of(row);
    for (int w = 0; w < WAYS; ++w)
        if (tag[s * WAYS + w] == row) { std::memcpy(&data[(s * WAYS + w) * kRowBytes], bytes, kRowBytes); return; }
    const int w = next_way[s];
    tag[s * WAYS + w] = row;
    std::memcpy(&data[(s * WAYS + w) * kRowBytes], bytes, kRowBytes);
    next_way[s] = (uint8_t)((w + 1) % WAYS);
}
uint64_t RowCache::resident() const {
    uint64_t n = 0;
    for (uint64_t t : tag) n += (t != EMPTY);
    return n;
}

uint64_t ReaderStats::percentile(double q) const {
    if (latency_ns.empty()) return 0;
    std::vector<uint64_t> c(latency_ns);
    std::sort(c.begin(), c.end());
    q = std::min(1.0, std::max(0.0, q));
    const uint64_t idx = std::min<uint64_t>(c.size() - 1, (uint64_t)std::llround(q * (double)(c.size() - 1)));
    return c[idx];
}

// ---- reader --------------------------------------------------------------------------------------------------------
PleReader::PleReader() {}

PleReader::~PleReader() { close(); }

bool PleReader::is_open() const { return fd_ >= 0; }

uint64_t PleReader::cache_capacity() const { return cache_.capacity(); }

void PleReader::set_injected_delay_us(uint64_t us) { delay_us_ = us; }

bool PleReader::open(const std::string& path, uint64_t table_off, uint64_t n_rows, uint64_t cache_rows,
                     int max_inflight, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (fd_ >= 0) return fail("already open");
    if (max_inflight <= 0) return fail("max_inflight must be positive");
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    bool direct = fd >= 0;
    if (fd < 0) fd = ::open(path.c_str(), O_RDONLY);  // filesystems without O_DIRECT
    if (fd < 0) return fail("cannot open " + path + ": " + std::strerror(errno));
    struct stat sb;
    if (fstat(fd, &sb) != 0) { ::close(fd); return fail("fstat failed"); }
    const uint64_t fsz = (uint64_t)sb.st_size;
    if (table_off + n_rows * kRowBytes > fsz) { ::close(fd); return fail("table extends past the end of the file"); }
    void* slab = nullptr;
    if (posix_memalign(&slab, kPageBytes, (size_t)max_inflight * kSlotBytes) != 0) { ::close(fd); return fail("slab allocation failed"); }
    fd_ = fd; direct_ = direct; table_off_ = table_off; n_rows_ = n_rows; file_size_ = fsz;
    max_inflight_ = (uint64_t)max_inflight; free_slots_ = max_inflight_; inflight_ = 0;
    slab_ = (uint8_t*)slab;
    free_list_.clear();
    for (uint64_t s = 0; s < max_inflight_; ++s) free_list_.push_back(s);
    cache_.init(cache_rows);
    ring_.assign(LATENCY_RING, 0); ring_pos_ = 0; st_ = ReaderStats{};
    last_ticket_ = 0; stop_ = false;
    th_ = std::thread(&PleReader::worker, this);
    return true;
}

void PleReader::close() {
    if (fd_ < 0) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_work_.notify_all();
    if (th_.joinable()) th_.join();
    ::close(fd_);
    fd_ = -1;
    std::free(slab_);
    slab_ = nullptr;
    queue_.clear(); tickets_.clear();
}

uint32_t PleReader::issue(const uint64_t* rows, uint64_t n, uint8_t* out) {
    if (fd_ < 0 || (!out && n)) return 0;
    for (uint64_t i = 0; i < n; ++i) if (rows[i] >= n_rows_) return 0;
    std::unique_lock<std::mutex> lk(mu_);
    const uint32_t t = last_ticket_ = next_ticket(last_ticket_);
    TicketState& ts = tickets_[t];
    ts.out = out;
    std::map<uint64_t, QueuedJob> jobs;  // first -> job, deduplicated per page within this ticket
    for (uint64_t i = 0; i < n; ++i) {
        ++st_.requests;
        if (const uint8_t* c = cache_.find(rows[i])) {
            ++st_.hits;
            std::memcpy(out + i * kRowBytes, c, kRowBytes);
            continue;
        }
        ++st_.misses;
        const uint64_t at = table_off_ + rows[i] * kRowBytes;
        const uint64_t f = first_of(at), l = length_of(at);
        QueuedJob& q = jobs[f];
        q.job.first = f;
        q.job.length = std::max(q.job.length, l);
        q.uses.push_back(Use{t, i, rows[i], at - f});
    }
    ts.pending = jobs.size();
    for (auto& j : jobs) queue_.push_back(std::move(j.second));
    lk.unlock();
    cv_work_.notify_one();
    return t;
}

bool PleReader::read_job(const Job& j, uint8_t* slot, std::string* err) {
    size_t got = 0;
    const size_t want = (size_t)j.length;
    while (got < want) {
        const ssize_t r = ::pread(fd_, slot + got, want - got, (off_t)(j.first + got));
        if (r < 0) { if (errno == EINTR) continue; if (err) *err = std::strerror(errno); return false; }
        if (r == 0) break;  // end of file: a legal short read when the page is the last of the file
        got += (size_t)r;
    }
    if (j.first + got < std::min<uint64_t>(j.first + want, file_size_)) { if (err) *err = "short read inside the table"; return false; }
    return true;
}

void PleReader::worker() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
        cv_work_.wait(lk, [&] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return;
        // pump: every queued job that finds a free slot goes in flight
        std::vector<std::pair<QueuedJob, uint64_t>> batch;
        while (!queue_.empty() && free_slots_ > 0) {
            const uint64_t s = free_list_.back(); free_list_.pop_back();
            --free_slots_; ++inflight_;
            batch.push_back({std::move(queue_.front()), s});
            queue_.pop_front();
        }
        lk.unlock();
        std::vector<char> ok(batch.size(), 0);
        std::vector<std::string> errs(batch.size());
        std::vector<uint64_t> lat(batch.size(), 0);
        for (size_t b = 0; b < batch.size(); ++b) {
            const auto t0 = std::chrono::steady_clock::now();
            if (delay_us_) std::this_thread::sleep_for(std::chrono::microseconds(delay_us_));
            ok[b] = read_job(batch[b].first.job, slab_ + batch[b].second * kSlotBytes, &errs[b]);
            lat[b] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        }
        lk.lock();
        for (size_t b = 0; b < batch.size(); ++b) {
            const Job& j = batch[b].first.job;
            const uint8_t* slot = slab_ + batch[b].second * kSlotBytes;
            uint32_t owner = 0;
            for (const Use& u : batch[b].first.uses) {
                owner = u.ticket;
                auto tk = tickets_.find(u.ticket);
                if (tk == tickets_.end()) continue;
                if (ok[b]) {
                    std::memcpy(tk->second.out + u.dst * kRowBytes, slot + u.in_page, kRowBytes);
                    cache_.put(u.row, slot + u.in_page);
                } else { tk->second.failed = true; tk->second.error = errs[b]; }
            }
            auto tk = tickets_.find(owner);
            if (tk != tickets_.end() && tk->second.pending) --tk->second.pending;
            ++st_.jobs; st_.bytes_read += j.length;
            ring_[ring_pos_ % LATENCY_RING] = lat[b]; ++ring_pos_;
            free_list_.push_back(batch[b].second);
            ++free_slots_; --inflight_;
        }
        cv_done_.notify_all();
    }
}

bool PleReader::collect(uint32_t ticket, std::string* err) {
    std::unique_lock<std::mutex> lk(mu_);
    auto it = tickets_.find(ticket);
    if (it == tickets_.end()) { if (err) *err = "unknown ticket"; return false; }
    cv_done_.wait(lk, [&] { auto i2 = tickets_.find(ticket); return i2 == tickets_.end() || i2->second.pending == 0; });
    it = tickets_.find(ticket);
    if (it == tickets_.end()) return false;
    const bool ok = !it->second.failed;
    if (!ok && err) *err = it->second.error;
    tickets_.erase(it);
    return ok;
}

uint64_t PleReader::cache_size() const { std::lock_guard<std::mutex> lk(mu_); return cache_.resident(); }

ReaderStats PleReader::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    ReaderStats s = st_;
    const uint64_t n = std::min<uint64_t>(ring_pos_, LATENCY_RING);
    s.latency_ns.assign(ring_.begin(), ring_.begin() + (std::ptrdiff_t)n);
    return s;
}

void PleReader::reset_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    st_ = ReaderStats{};
    ring_pos_ = 0;
}

}  // namespace pleasync
}  // namespace baloon
