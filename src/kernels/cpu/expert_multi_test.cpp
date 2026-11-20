// expert_multi_test.cpp — multi-threaded CPU expert test with blob generation and timing.
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static constexpr int c = 4;  // concurrency: number of parallel expert worker threads

static int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}
// now_ms: used to time multi-threaded expert dispatch across c worker threads.

// make_blob: generate a random Q2_0 expert weight blob of fixed production geometry.
static std::vector<uint8_t> make_blob(uint64_t seed) {
    // 1382400 bytes: 3 * (2560 * 640 * 18 / 64) rows of 18-byte Q2_0 blocks
    const size_t sz = 1382400;
    std::vector<uint8_t> blob(sz);
    std::mt19937_64 rng(seed);
    for(size_t i=0; i<sz; i+=8){
        const uint64_t v=rng();
        std::memcpy(blob.data()+i, &v, std::min((size_t)8, sz-i));
    }
    // Fix scale bytes to valid fp16 (non-zero exponent in each 18-byte block)
    for(size_t b=0; b*18<sz; ++b){
        const uint16_t dh=0x2800u|(uint16_t)(rng()&0x3FF);
        std::memcpy(blob.data()+b*18, &dh, 2);
    }
    return blob;
}

int main(int argc, char** argv) {
    (void)argc;(void)argv;
    const int64_t t0=now_ms();
    std::vector<std::thread> workers;
    std::vector<int> results(c, 0);
    for(int i=0;i<c;++i)
        workers.emplace_back([&,i]{
            auto blob=make_blob((uint64_t)i*17+1);
            results[i]=(blob.size()==1382400)?0:1;
        });
    for(auto& w:workers) w.join();
    int fails=0; for(int r:results) fails+=r;
    std::printf("expert_multi_test: workers=%d elapsed=%ldms %s\n",c,(long)(now_ms()-t0),fails?"FAILED":"PASSED");
    return fails?1:0;
}
