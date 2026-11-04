// core_verify.hpp - window verifier (host model): record a draft window, publish per-(layer, group) expert plans in two
// plan slots, raise the monotone progress flag, DMA-fetch experts into bounded staging, accept the longest agreeing prefix
// and commit it. Ring value of step k = layer*G + group is k+1; groups split the T window rows at ceil(T/2).
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace baloon {
namespace core {

class Verifier {
public:
    struct Config {
        uint64_t max_t = 8, n_layers = 48, groups = 1;       // groups: 1 or 2
        uint64_t plan_cap = 64;                              // router entries per plan slot
        uint64_t staging_cap = 64;                           // expert fetches per window
        uint64_t row_floats = 0;                             // floats of one residual row (hc * n_embd)
    };
    struct Step { uint64_t layer, group, ring, row_begin, row_end; };

    Verifier() = default;
    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;
    ~Verifier();

    bool init(const Config& c);
    bool record_window(const int32_t* tokens, uint64_t T);       // tokens[0] = last committed, then T-1 drafts
    bool capture();                                              // build the (layer, group) schedule
    bool capture_commit();                                       // arm the commit graph
    uint64_t run(const std::function<void(const Step&)>& body);  // execute the schedule, raising the flag per step
    void set_plan_slot(uint64_t group);                          // select the plan slot written by publish_plan
    bool publish_plan(uint64_t layer, uint64_t group, const std::vector<int32_t>& words);
    void raise_flag(uint32_t v);                                 // never lowers the flag
    uint32_t flag() const { return flag_.load(std::memory_order_acquire); }
    bool fetch_dma(const uint8_t* src, uint64_t bytes);          // staging copy; false once the fetch budget is spent
    uint64_t accept_len(const std::vector<int32_t>& greedy) const;      // a: longest prefix with window[a+1] == greedy[a]
    bool commit(uint64_t a, int64_t pos0, std::vector<int32_t>& entries);   // commit vector for a accepted drafts
    float* final_R(uint64_t a);                                  // residual row of the last accepted position

    const std::vector<Step>& schedule() const { return schedule_; }
    const std::vector<int32_t>& plan_slot_words(uint64_t group) const { return plan_[group & 1]; }
    uint64_t T() const { return T_; }
    uint64_t fetches() const { return fetches_; }
    uint64_t staged_bytes() const { return staged_; }
    bool commit_armed() const { return commit_armed_; }

private:
    Config c_;
    bool inited_ = false, commit_armed_ = false;
    uint64_t T_ = 0, slot_ = 0, fetches_ = 0, staged_ = 0;
    std::vector<int32_t> window_;
    std::vector<Step> schedule_;
    std::vector<int32_t> plan_[2];
    std::vector<uint8_t> staging_;
    std::vector<float> R_;
    std::atomic<uint32_t> flag_{0};
};

}  // namespace core
}  // namespace baloon
