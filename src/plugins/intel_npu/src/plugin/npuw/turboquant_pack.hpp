// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace ov::npuw::turboquant {

// ---------------------------------------------------------------------------
// Host-side bit packers/unpackers for TurboQuant KV-cache storage.
//
// The TurboQuant graph emits u8 index tensors whose values fit in `bits` bits
// (e.g. [0, 7] for 3-bit keys, [0, 3] for 2-bit values). These helpers narrow
// the storage to `bits` bits per element on the way into the user's cache
// pool, and widen them back to u8 on the way out — matching the paper's 4.6×
// compression claim.
//
// The packed layout is little-endian LSB-first: element i at bit offset
// i * bits within the packed byte stream. `pack_bits` is always ≤ 8.
//
// QJL sign bits (values in {-1, +1} stored in i8) pack to 1 bit per element
// via pack_signs; 0 maps to -1 when unpacking.
//
// All helpers operate on logically flat buffers — total element count is
// `shape_size(u8_tensor.get_shape())`. Tensor shape is preserved on round
// trip.
// ---------------------------------------------------------------------------

// Returns the number of bytes needed to densely pack `num_elements` values at
// `bits` bits each (rounded up to the nearest byte).
inline std::size_t packed_size_bytes(std::size_t num_elements, std::uint8_t bits) {
    return (num_elements * bits + 7u) / 8u;
}

// Pack src (ov::element::u8, values in [0, 2^bits - 1]) into a dense bit
// buffer. `dst.size()` must equal packed_size_bytes(N, bits). Values
// exceeding the bit range are silently masked (caller's responsibility).
void pack_bits(const ov::Tensor& src_u8, std::uint8_t bits, std::uint8_t* dst, std::size_t dst_bytes);

// Unpack a packed bit buffer back into an ov::element::u8 tensor. The caller
// supplies the destination tensor with the correct shape and element count.
void unpack_bits(const std::uint8_t* src, std::size_t src_bytes, std::uint8_t bits, ov::Tensor& dst_u8);

// Convenience: pack and return a std::vector<uint8_t>.
std::vector<std::uint8_t> pack_bits(const ov::Tensor& src_u8, std::uint8_t bits);

// Sign packing for QJL: maps ±1 (i8) to {0, 1} bits. dst_bytes must be
// ceil(N / 8).
void pack_signs(const ov::Tensor& src_i8_pm1, std::uint8_t* dst, std::size_t dst_bytes);
void unpack_signs(const std::uint8_t* src, std::size_t src_bytes, ov::Tensor& dst_i8_pm1);
std::vector<std::uint8_t> pack_signs(const ov::Tensor& src_i8_pm1);

}  // namespace ov::npuw::turboquant
