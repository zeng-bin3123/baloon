// bl_cpu_expert_pool.h - work partitioning of the CPU expert pool.
#pragma once
#include "bl_nat.h"

namespace baloon::cpu_expert_pool {

// Task i of n owns rows [rowLo, rowHi): contiguous, ordered, covering [0, rows) exactly.
constexpr nat rowLo(nat rows, nat n, nat i) { return ndiv(i * rows, n); }
constexpr nat rowHi(nat rows, nat n, nat i) { return ndiv((i + 1) * rows, n); }

// Pool = (cores - 1) workers plus the host when it works: one thread per physical core.
constexpr nat poolThreads(nat cores, bool hostWorks) { return sat_sub(cores, 1) + (hostWorks ? 1 : 0); }

}  // namespace baloon::cpu_expert_pool
