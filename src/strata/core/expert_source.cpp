// strata/core/expert_source.cpp — Expert blob sources + dispatch functions
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/pinned.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata {

// Expert blob size constant (architecture-specific)
static constexpr int64_t BLOB_BYTES = 1382400;

// ─── FileExpertSource ────────────────────────────────────────────────

FileExpertSource::~FileExpertSource() {
    close();
}

bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers,
                            int64_t n_expert, std::string& err) {
    close();

    std::string path = pack_dir + "/experts.bin";
    int64_t expected_size = n_layers * n_expert * BLOB_BYTES;

#if defined(_WIN32)
    file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
        err = "cannot open " + path;
        file_ = nullptr;
        return false;
    }

    LARGE_INTEGER fsize;
    GetFileSizeEx(file_, &fsize);
    if (fsize.QuadPart != expected_size) {
        err = "file size mismatch: expected " + std::to_string(expected_size) +
              " got " + std::to_string(fsize.QuadPart);
        CloseHandle(file_);
        file_ = nullptr;
        return false;
    }

    mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_) {
        err = "CreateFileMapping failed";
        CloseHandle(file_);
        file_ = nullptr;
        return false;
    }

    base_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
    if (!base_) {
        err = "MapViewOfFile failed";
        CloseHandle(mapping_);
        CloseHandle(file_);
        mapping_ = nullptr;
        file_ = nullptr;
        return false;
    }
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "cannot open " + path + ": " + strerror(errno);
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        err = "fstat failed: " + std::string(strerror(errno));
        ::close(fd);
        return false;
    }

    if (st.st_size != expected_size) {
        err = "file size mismatch: expected " + std::to_string(expected_size) +
              " got " + std::to_string(st.st_size);
        ::close(fd);
        return false;
    }

    void* p = mmap(nullptr, static_cast<size_t>(expected_size),
                   PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        err = "mmap failed: " + std::string(strerror(errno));
        ::close(fd);
        return false;
    }

    base_ = static_cast<const uint8_t*>(p);
    fd_ = fd;
#endif

    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    read_count_ = 0;

    return true;
}

void FileExpertSource::close() {
#if defined(_WIN32)
    if (base_) { UnmapViewOfFile(base_); base_ = nullptr; }
    if (mapping_) { CloseHandle(mapping_); mapping_ = nullptr; }
    if (file_) { CloseHandle(file_); file_ = nullptr; }
#else
    if (base_ && blobs_ > 0) {
        munmap(const_cast<uint8_t*>(base_),
               static_cast<size_t>(blobs_ * BLOB_BYTES));
        base_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    blobs_ = 0;
    n_expert_ = 0;
    read_count_ = 0;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (!base_ || layer < 0 || expert < 0 || expert >= n_expert_) {
        return nullptr;
    }
    int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++read_count_;
    return base_ + idx * BLOB_BYTES;
}

// ─── ArenaExpertSource ───────────────────────────────────────────────

ArenaExpertSource::~ArenaExpertSource() {
    close();
}

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers,
                             int64_t n_expert, int threads, std::string& err) {
    close();

    int64_t total_blobs = n_layers * n_expert;
    uint64_t total_bytes = static_cast<uint64_t>(total_blobs) * BLOB_BYTES;

    // Allocate PinnedArena
    auto* arena = new (std::nothrow) PinnedArena(total_bytes);
    if (!arena || !arena->valid()) {
        err = "ArenaExpertSource: arena allocation failed for " +
              std::to_string(total_bytes) + " bytes";
        delete arena;
        return false;
    }

    arena_ = arena;
    base_ = arena->data();

    // Load from disk using parallel file loading
    std::string path = pack_dir + "/experts.bin";
    LoadStats ls = load_experts(path, const_cast<uint8_t*>(base_),
                                BLOB_BYTES, static_cast<uint64_t>(n_expert),
                                static_cast<uint64_t>(n_layers),
                                threads > 0 ? threads : 4,
                                4 * 1024 * 1024);  // 4 MB chunks

    load_speed_ = ls.gib_per_second();
    note_ = arena->note + " (" + std::to_string(ls.gib_per_second()) + " GiB/s)";

    blobs_ = total_blobs;
    n_expert_ = n_expert;
    read_count_ = 0;
    pinned_bytes_ = arena->registered_bytes;

    // Build device alias table for pinned slices
    if (arena->registered_bytes > 0) {
        slice_bytes_ = arena->slice_size > 0 ? arena->slice_size : total_bytes;
        size_t n_slices = (total_bytes + slice_bytes_ - 1) / slice_bytes_;
        dev_slice_.resize(n_slices, nullptr);

        for (size_t i = 0; i < n_slices; ++i) {
            uint64_t off = i * slice_bytes_;
            if (off >= arena->registered_bytes) break;

            void* d_ptr = nullptr;
            cudaError_t rc = cudaHostGetDevicePointer(&d_ptr,
                                                       const_cast<uint8_t*>(base_) + off,
                                                       0);
            if (rc == cudaSuccess && d_ptr) {
                dev_slice_[i] = static_cast<const uint8_t*>(d_ptr);
            }
        }
    }

    return true;
}

void ArenaExpertSource::close() {
    if (arena_) {
        delete static_cast<PinnedArena*>(arena_);
        arena_ = nullptr;
    }
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
    read_count_ = 0;
    dev_slice_.clear();
    slice_bytes_ = 0;
    pinned_bytes_ = 0;
    note_.clear();
    load_speed_ = 0.0;
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (!base_ || layer < 0 || expert < 0 || expert >= n_expert_) {
        return nullptr;
    }
    int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++read_count_;
    return base_ + idx * BLOB_BYTES;
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    (void)layer; (void)expert;
    return pinned_bytes_ > 0;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (dev_slice_.empty() || !base_ || layer < 0 || expert < 0 || expert >= n_expert_) {
        return nullptr;
    }
    int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;

    uint64_t byte_off = static_cast<uint64_t>(idx) * BLOB_BYTES;
    if (slice_bytes_ == 0) return nullptr;

    size_t slice_idx = static_cast<size_t>(byte_off / slice_bytes_);
    if (slice_idx >= dev_slice_.size() || !dev_slice_[slice_idx]) return nullptr;

    uint64_t within_slice = byte_off - slice_idx * slice_bytes_;
    return dev_slice_[slice_idx] + within_slice;
}

// ─── Dispatch functions ──────────────────────────────────────────────

void expert_pool_dispatch(void* user, const float* x_f,
                          const int32_t* ids, const float* weights,
                          int64_t n_embd, int64_t k, float* out) {
    (void)weights;  // weight applied by moe_combine, not here

    auto* d = static_cast<ExpertDispatch*>(user);
    if (!d || !d->src) return;

    for (int64_t j = 0; j < k; ++j) {
        int32_t eid = ids[j];
        const uint8_t* blob = d->src->blob(d->layers, eid);

        if (blob) {
            // Expert evaluation would go here (calls into kernels::cpu).
            // For now, the dispatch just records the blob and job.
            // Actual evaluation is done by the ExpertPool in later units.
            ++d->experts;
        } else {
            // Missing expert: output zeros
            std::memset(out + j * n_embd, 0, static_cast<size_t>(n_embd) * sizeof(float));
            ++d->missing;
        }
    }

    ++d->layers;
}

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f,
                                const int32_t* ids, int64_t n_tok,
                                int64_t k, float* out) {
    if (!d.src) return;

    // Zero output buffer
    std::memset(out, 0, static_cast<size_t>(n_tok * k) * sizeof(float) *
                        (d.n_expert > 0 ? 1 : 0));

    // Build deduplicated job list
    d.jobs_multi.clear();
    d.job_of.assign(static_cast<size_t>(n_tok * k), -1);

    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            int64_t flat = t * k + j;
            int32_t eid = ids[flat];

            // Check if GPU handles this one (cache hit)
            if (d.host_res && d.decided) {
                size_t hit_idx = static_cast<size_t>(flat);
                if (hit_idx < d.is_hit.size() && d.is_hit[hit_idx]) {
                    continue;  // GPU computes this expert
                }
            }

            const uint8_t* blob = d.src->blob(d.layers, eid);
            if (!blob) {
                ++d.missing;
                continue;
            }

            ++d.experts;
            // Record job (actual compute in later units)
        }
    }

    ++d.layers;
    d.multi_entries += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase,
                    const int32_t* ids, int64_t k) {
    auto* d = static_cast<ExpertDispatch*>(user);
    if (!d || !d->hits_ready()) return;

    // No-op when no hits
    if (d->n_hits == 0) return;

    // Actual GPU expert evaluation is in later units (CUDA kernels).
    // This function sets up the dispatch plan and launches GPU work
    // for cache-hit experts only.
    (void)stream;
    (void)phase;
    (void)ids;
    (void)k;
}

}  // namespace strata

