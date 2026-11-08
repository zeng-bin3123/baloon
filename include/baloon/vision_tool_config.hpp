// vision_tool_config.hpp - build switches of the vision helper: GPU on/off and portable mode. Portable mode targets
// machines without a matching CPU or runtime library: native tuning and OpenMP off, a fixed AVX2 baseline on.
#pragma once
#include <string>
#include <vector>

namespace baloon {
namespace vision_cfg {

struct Config { bool portable = false; bool gpu = false; };

inline bool native_tuning(const Config& c) { return !c.portable; }
inline bool openmp_forced_off(const Config& c) { return c.portable; }
inline bool gpu_backend(const Config& c) { return c.gpu; }
// Instruction-set baseline forced on in portable mode only.
inline bool forces_avx2_baseline(const Config& c) { return c.portable; }

// Ordered (name, value) switches handed to the build of the encoder library.
struct Switch { std::string name; bool value; };
inline std::vector<Switch> switches(const Config& c) {
    std::vector<Switch> v;
    v.push_back({"GGML_CUDA", gpu_backend(c)});
    v.push_back({"GGML_NATIVE", native_tuning(c)});
    if (openmp_forced_off(c)) v.push_back({"GGML_OPENMP", false});
    if (forces_avx2_baseline(c))
        for (const char* n : {"GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C", "GGML_BMI2"}) v.push_back({n, true});
    return v;
}

}  // namespace vision_cfg
}  // namespace baloon
