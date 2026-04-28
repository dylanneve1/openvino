// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/openvino.hpp"

namespace ov::npuw {

// TurboQuant (ICLR 2026, arXiv:2504.19874) KV-cache compression.
//
// Per-branch configuration. The default values reproduce the paper:
//   keys  : 3-bit, per-token scale, QJL on, Lloyd-Max codebook on.
//   values: 2-bit, per-group scale (group=32), QJL off, Lloyd-Max codebook on.
//
// `bits` is the effective bit width of the quantized index. Values are still
// emitted as u8 by the graph; a host-side packer narrows storage to `bits`
// per element.
//
// `group_size == 0` means per-token (one scale per token along head_dim).
struct TurboQuantConfig {
    uint8_t bits = 8;
    uint32_t group_size = 0;
    bool use_qjl = false;
    bool use_lloyd_max = true;
};

struct TurboQuantParams {
    TurboQuantConfig key{3u, 0u, true, true};
    TurboQuantConfig value{2u, 32u, false, true};
};

// Inserts the TurboQuant rewrite for every SDPA block found in the model:
//   - Hadamard rotation along head_dim on present-K and present-V before quantize
//     (identity fallback if head_dim is not a power of two).
//   - Lloyd-Max quantize -> u8 index tensor on the present side, with scale (and
//     zp for asymmetric) exposed as new model Results.
//   - Optional QJL sign-bit branch on present-K, exposed as an extra Result.
//   - Inverse Hadamard + Lloyd-Max dequantize on the past side, consumed by the
//     downstream Concat in place of the original past-K/V Parameter edge.
//
// Result naming follows the existing kv_cache_compressed.cpp convention so that
// kv_cache_copy plumbing keeps working:
//   "DynamicQuantize/{idx}/present/{key|value}/{scale|zp}"
// QJL adds:
//   "DynamicQuantize/{idx}/present/key/qjl"
void run_turboquant_kv_cache_passes(const std::shared_ptr<ov::Model>& model, const TurboQuantParams& params);

// NOLINTNEXTLINE(readability/namespace)
}  // namespace ov::npuw
