cmake_minimum_required(VERSION 3.18)
project(baloon LANGUAGES CXX)

# ===========================================================================
# Baloon CMake — Iluvatar BI-V100 (ivcore10)
#
# Because CoreX's "nvcc" is a wrapper, we drive clang++ directly with
# -x cuda flags rather than enabling CMake's CUDA language support.
# ===========================================================================

# --- Configurable paths ---
set(COREX_PATH "/usr/local/corex" CACHE PATH "CoreX SDK root")
set(GPU_ARCH   "ivcore10"          CACHE STRING "GPU architecture target")

# --- Compiler ---
set(CMAKE_CXX_COMPILER "${COREX_PATH}/bin/clang++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Common flags ---
set(CUDA_COMPILE_FLAGS
    "-x cuda"
    "--cuda-gpu-arch=${GPU_ARCH}"
    "--cuda-path=${COREX_PATH}"
    "-D__ILUVATAR__"
    "-D__ILUVATAR_WORKAROUND__"
)
string(REPLACE ";" " " CUDA_FLAGS_STR "${CUDA_COMPILE_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_FLAGS_STR}")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${COREX_PATH}/include
)

link_directories(${COREX_PATH}/lib64)

# --- Executable: device_info smoke test ---
add_executable(device_info src/core/device_info.cu)
target_link_libraries(device_info cudart pthread)

# --- Shared library: libbaloon_core.so ---
add_library(baloon_core SHARED src/core/device_info.cu)
target_compile_definitions(baloon_core PRIVATE BALOON_LIB_ONLY)
target_compile_options(baloon_core PRIVATE -fPIC -O3)
target_link_libraries(baloon_core cudart)

# --- Install ---
install(TARGETS device_info RUNTIME DESTINATION bin)
install(TARGETS baloon_core LIBRARY DESTINATION lib)
install(DIRECTORY include/baloon DESTINATION include)

