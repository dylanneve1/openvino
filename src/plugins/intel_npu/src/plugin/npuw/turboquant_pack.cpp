// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "turboquant_pack.hpp"

#include <cstring>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"

namespace ov::npuw::turboquant {

namespace {

std::size_t num_elements(const ov::Tensor& t) {
    const auto& shape = t.get_shape();
    std::size_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    return n;
}

}  // namespace

void pack_bits(const ov::Tensor& src_u8, std::uint8_t bits, std::uint8_t* dst, std::size_t dst_bytes) {
    OPENVINO_ASSERT(src_u8.get_element_type() == ov::element::u8, "pack_bits: source must be u8");
    OPENVINO_ASSERT(bits >= 1 && bits <= 8, "pack_bits: bits must be in [1, 8]");

    const std::size_t n = num_elements(src_u8);
    const std::size_t need = packed_size_bytes(n, bits);
    OPENVINO_ASSERT(dst_bytes >= need, "pack_bits: dst_bytes too small (need ", need, ", got ", dst_bytes, ")");

    std::memset(dst, 0, need);
    if (bits == 8) {
        std::memcpy(dst, src_u8.data<const std::uint8_t>(), n);
        return;
    }

    const std::uint8_t mask = static_cast<std::uint8_t>((1u << bits) - 1u);
    const std::uint8_t* src = src_u8.data<const std::uint8_t>();
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t bit_pos = static_cast<std::uint32_t>(i) * bits;
        const std::size_t byte_idx = bit_pos >> 3u;
        const std::uint32_t bit_off = bit_pos & 7u;
        const std::uint8_t v = static_cast<std::uint8_t>(src[i] & mask);
        dst[byte_idx] |= static_cast<std::uint8_t>(v << bit_off);
        // Crosses byte boundary if bit_off + bits > 8.
        if (bit_off + bits > 8) {
            dst[byte_idx + 1] |= static_cast<std::uint8_t>(v >> (8 - bit_off));
        }
    }
}

std::vector<std::uint8_t> pack_bits(const ov::Tensor& src_u8, std::uint8_t bits) {
    const std::size_t n = num_elements(src_u8);
    std::vector<std::uint8_t> out(packed_size_bytes(n, bits));
    pack_bits(src_u8, bits, out.data(), out.size());
    return out;
}

void unpack_bits(const std::uint8_t* src, std::size_t src_bytes, std::uint8_t bits, ov::Tensor& dst_u8) {
    OPENVINO_ASSERT(dst_u8.get_element_type() == ov::element::u8, "unpack_bits: dst must be u8");
    OPENVINO_ASSERT(bits >= 1 && bits <= 8, "unpack_bits: bits must be in [1, 8]");

    const std::size_t n = num_elements(dst_u8);
    const std::size_t need = packed_size_bytes(n, bits);
    OPENVINO_ASSERT(src_bytes >= need, "unpack_bits: src_bytes too small (need ", need, ", got ", src_bytes, ")");

    std::uint8_t* dst = dst_u8.data<std::uint8_t>();
    if (bits == 8) {
        std::memcpy(dst, src, n);
        return;
    }

    const std::uint8_t mask = static_cast<std::uint8_t>((1u << bits) - 1u);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t bit_pos = static_cast<std::uint32_t>(i) * bits;
        const std::size_t byte_idx = bit_pos >> 3u;
        const std::uint32_t bit_off = bit_pos & 7u;
        std::uint16_t lo = src[byte_idx];
        std::uint16_t hi = (bit_off + bits > 8) ? src[byte_idx + 1] : 0;
        std::uint16_t window = static_cast<std::uint16_t>(lo | (hi << 8));
        dst[i] = static_cast<std::uint8_t>((window >> bit_off) & mask);
    }
}

void pack_signs(const ov::Tensor& src_i8_pm1, std::uint8_t* dst, std::size_t dst_bytes) {
    OPENVINO_ASSERT(src_i8_pm1.get_element_type() == ov::element::i8, "pack_signs: source must be i8");

    const std::size_t n = num_elements(src_i8_pm1);
    const std::size_t need = (n + 7u) / 8u;
    OPENVINO_ASSERT(dst_bytes >= need, "pack_signs: dst_bytes too small (need ", need, ", got ", dst_bytes, ")");

    std::memset(dst, 0, need);
    const std::int8_t* src = src_i8_pm1.data<const std::int8_t>();
    for (std::size_t i = 0; i < n; ++i) {
        // Map +1 -> 1, any other value (typically -1 or 0) -> 0.
        if (src[i] > 0) {
            dst[i >> 3u] |= static_cast<std::uint8_t>(1u << (i & 7u));
        }
    }
}

std::vector<std::uint8_t> pack_signs(const ov::Tensor& src_i8_pm1) {
    const std::size_t n = num_elements(src_i8_pm1);
    std::vector<std::uint8_t> out((n + 7u) / 8u);
    pack_signs(src_i8_pm1, out.data(), out.size());
    return out;
}

void unpack_signs(const std::uint8_t* src, std::size_t src_bytes, ov::Tensor& dst_i8_pm1) {
    OPENVINO_ASSERT(dst_i8_pm1.get_element_type() == ov::element::i8, "unpack_signs: dst must be i8");

    const std::size_t n = num_elements(dst_i8_pm1);
    const std::size_t need = (n + 7u) / 8u;
    OPENVINO_ASSERT(src_bytes >= need, "unpack_signs: src_bytes too small (need ", need, ", got ", src_bytes, ")");

    std::int8_t* dst = dst_i8_pm1.data<std::int8_t>();
    for (std::size_t i = 0; i < n; ++i) {
        const bool bit = (src[i >> 3u] >> (i & 7u)) & 1u;
        dst[i] = bit ? std::int8_t(1) : std::int8_t(-1);
    }
}

}  // namespace ov::npuw::turboquant
