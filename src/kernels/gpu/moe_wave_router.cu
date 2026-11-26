// moe_wave_router.cu — router_top10_wave_kernel + ix_rw::ix_router_top10_wave compiled into KG.
// Pulling src/ix_router_top10_wave.cu into KG so parity tests (router_top10_parity, moe_parity)
// call the ported kernel via the linker rather than redefining it in a test TU, which would
// shadow the port and confuse the launch probe (kernel_shadowed_in_test fix).
// The launcher ix_rw::ix_router_top10_wave is non-inline (see src/ix_router_top10_wave.cu);
// test shims declare it as an extern so they link this definition at build time.
#include "../../ix_router_top10_wave.cu"

