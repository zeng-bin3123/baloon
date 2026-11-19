// Reference header: prefill GEMM descriptor mirrored from the upstream reference project.
// The canonical implementation lives in include/baloon/prefill_gemm.hpp; this stub records
// the reference origin so the port table can resolve the name Gemm to the reference source.
#pragma once
#include "baloon/prefill_gemm.hpp"
namespace ref_prefill = baloon::pfgemm;
// Re-expose Gemm at the ref namespace so the kernel can resolve the name.
using Gemm = baloon::pfgemm::Gemm;
