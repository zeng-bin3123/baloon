// s2_gemv_tree.cu - block-per-row Q2 GEMV with a constant-memory code table and a halving-tree reduction.
// Thread t of a row walks quads q = t, t + tpr, ...; quad q is code byte q, its four values share the scale q >> 4.
// The per-thread partials go to shared memory and are folded with step = tpr/2, ..., 1 (partial[t] += partial[t + step]
// for t < step), so the sum is a fixed balanced tree and does not depend on the launch's scheduling.
#include <cuda_runtime.h>

#include <cstdint>

#include "baloon/f16_bits.hpp"
#include "baloon/s2_gemv_tree.hpp"

namespace baloon {
namespace kern {

namespace {

#define C4(b) {(int8_t)((((b) >> 0) & 3) - 1), (int8_t)((((b) >> 2) & 3) - 1), (int8_t)((((b) >> 4) & 3) - 1), (int8_t)((((b) >> 6) & 3) - 1)}
// byte -> four biased code values, ((b >> 2k) & 3) - 1; statically initialised constant memory
__constant__ int8_t c_codes[256][4] = {
    C4(0),
    C4(1),
    C4(2),
    C4(3),
    C4(4),
    C4(5),
    C4(6),
    C4(7),
    C4(8),
    C4(9),
    C4(10),
    C4(11),
    C4(12),
    C4(13),
    C4(14),
    C4(15),
    C4(16),
    C4(17),
    C4(18),
    C4(19),
    C4(20),
    C4(21),
    C4(22),
    C4(23),
    C4(24),
    C4(25),
    C4(26),
    C4(27),
    C4(28),
    C4(29),
    C4(30),
    C4(31),
    C4(32),
    C4(33),
    C4(34),
    C4(35),
    C4(36),
    C4(37),
    C4(38),
    C4(39),
    C4(40),
    C4(41),
    C4(42),
    C4(43),
    C4(44),
    C4(45),
    C4(46),
    C4(47),
    C4(48),
    C4(49),
    C4(50),
    C4(51),
    C4(52),
    C4(53),
    C4(54),
    C4(55),
    C4(56),
    C4(57),
    C4(58),
    C4(59),
    C4(60),
    C4(61),
    C4(62),
    C4(63),
    C4(64),
    C4(65),
    C4(66),
    C4(67),
    C4(68),
    C4(69),
    C4(70),
    C4(71),
    C4(72),
    C4(73),
    C4(74),
    C4(75),
    C4(76),
    C4(77),
    C4(78),
    C4(79),
    C4(80),
    C4(81),
    C4(82),
    C4(83),
    C4(84),
    C4(85),
    C4(86),
    C4(87),
    C4(88),
    C4(89),
    C4(90),
    C4(91),
    C4(92),
    C4(93),
    C4(94),
    C4(95),
    C4(96),
    C4(97),
    C4(98),
    C4(99),
    C4(100),
    C4(101),
    C4(102),
    C4(103),
    C4(104),
    C4(105),
    C4(106),
    C4(107),
    C4(108),
    C4(109),
    C4(110),
    C4(111),
    C4(112),
    C4(113),
    C4(114),
    C4(115),
    C4(116),
    C4(117),
    C4(118),
    C4(119),
    C4(120),
    C4(121),
    C4(122),
    C4(123),
    C4(124),
    C4(125),
    C4(126),
    C4(127),
    C4(128),
    C4(129),
    C4(130),
    C4(131),
    C4(132),
    C4(133),
    C4(134),
    C4(135),
    C4(136),
    C4(137),
    C4(138),
    C4(139),
    C4(140),
    C4(141),
    C4(142),
    C4(143),
    C4(144),
    C4(145),
    C4(146),
    C4(147),
    C4(148),
    C4(149),
    C4(150),
    C4(151),
    C4(152),
    C4(153),
    C4(154),
    C4(155),
    C4(156),
    C4(157),
    C4(158),
    C4(159),
    C4(160),
    C4(161),
    C4(162),
    C4(163),
    C4(164),
    C4(165),
    C4(166),
    C4(167),
    C4(168),
    C4(169),
    C4(170),
    C4(171),
    C4(172),
    C4(173),
    C4(174),
    C4(175),
    C4(176),
    C4(177),
    C4(178),
    C4(179),
    C4(180),
    C4(181),
    C4(182),
    C4(183),
    C4(184),
    C4(185),
    C4(186),
    C4(187),
    C4(188),
    C4(189),
    C4(190),
    C4(191),
    C4(192),
    C4(193),
    C4(194),
    C4(195),
    C4(196),
    C4(197),
    C4(198),
    C4(199),
    C4(200),
    C4(201),
    C4(202),
    C4(203),
    C4(204),
    C4(205),
    C4(206),
    C4(207),
    C4(208),
    C4(209),
    C4(210),
    C4(211),
    C4(212),
    C4(213),
    C4(214),
    C4(215),
    C4(216),
    C4(217),
    C4(218),
    C4(219),
    C4(220),
    C4(221),
    C4(222),
    C4(223),
    C4(224),
    C4(225),
    C4(226),
    C4(227),
    C4(228),
    C4(229),
    C4(230),
    C4(231),
    C4(232),
    C4(233),
    C4(234),
    C4(235),
    C4(236),
    C4(237),
    C4(238),
    C4(239),
    C4(240),
    C4(241),
    C4(242),
    C4(243),
    C4(244),
    C4(245),
    C4(246),
    C4(247),
    C4(248),
    C4(249),
    C4(250),
    C4(251),
    C4(252),
    C4(253),
    C4(254),
    C4(255)
};
#undef C4

template <bool STAGE>
__global__ void s2_gemv_tree_kernel(const uint8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                    const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out, int tpr) {
    extern __shared__ __align__(16) unsigned char smem[];
    uint16_t* xs = reinterpret_cast<uint16_t*>(smem);
    float* partial = reinterpret_cast<float*>(smem + (STAGE ? kS2TreeStageBytes : 0));
    const int row = blockIdx.x;
    if (row >= n_out) return;   // uniform over the block
    const int t = threadIdx.x;
    if (STAGE) {
        for (int i = t; i < n_in; i += tpr) xs[i] = f16bits::f16_from_f32(x[i]);
        __syncthreads();
    }
    const int nq = n_in / 4;
    const uint8_t* crow = codes + (size_t)row * nq;
    const uint16_t* srow = scales + (size_t)row * (n_in / 64);
    float acc = 0.f;
    for (int q = t; q < nq; q += tpr) {
        const float s = f16bits::f32_from_f16(srow[q >> 4]);
        const int8_t* c = c_codes[crow[q]];
        float x0, x1, x2, x3;
        if (STAGE) {
            x0 = f16bits::f32_from_f16(xs[4 * q]);     x1 = f16bits::f32_from_f16(xs[4 * q + 1]);
            x2 = f16bits::f32_from_f16(xs[4 * q + 2]); x3 = f16bits::f32_from_f16(xs[4 * q + 3]);
        } else {
            x0 = x[4 * q]; x1 = x[4 * q + 1]; x2 = x[4 * q + 2]; x3 = x[4 * q + 3];
        }
        acc += s * (((float)c[0] * x0 + (float)c[1] * x1) + ((float)c[2] * x2 + (float)c[3] * x3));
    }
    partial[t] = acc;
    __syncthreads();
    for (int step = tpr >> 1; step > 0; step >>= 1) {
        if (t < step) partial[t] += partial[t + step];
        __syncthreads();
    }
    if (t == 0) y[row] = partial[0];
}

}  // namespace

cudaError_t s2_gemv_tree(const uint8_t* codes, const uint16_t* scales, const float* x, float* y, int n_in, int n_out,
                         int threads_per_row, bool stage, cudaStream_t stream) {
    if (!codes || !scales || !x || !y || n_in <= 0 || n_out <= 0 || n_in % 64 != 0) return cudaErrorInvalidValue;
    if (!s2_tree_threads_ok(threads_per_row)) return cudaErrorInvalidValue;
    if (stage && n_in > kS2TreeMaxStageHalves) return cudaErrorInvalidValue;
    const size_t smem = s2_tree_shared_bytes(stage, threads_per_row);
    if (stage) s2_gemv_tree_kernel<true><<<n_out, threads_per_row, smem, stream>>>(codes, scales, x, y, n_in, n_out, threads_per_row);
    else s2_gemv_tree_kernel<false><<<n_out, threads_per_row, smem, stream>>>(codes, scales, x, y, n_in, n_out, threads_per_row);
    return cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon
