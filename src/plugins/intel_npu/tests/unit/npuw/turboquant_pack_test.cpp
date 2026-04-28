// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Roundtrip tests for the TurboQuant host-side bit packer/unpacker.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "turboquant_pack.hpp"

namespace {

using namespace ov;

// Fill a u8 tensor with uniformly-random values in [0, max_val].
void fill_random_u8(ov::Tensor& t, std::uint8_t max_val, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> dist(0u, max_val);
    auto* p = t.data<std::uint8_t>();
    for (std::size_t i = 0; i < t.get_size(); ++i) {
        p[i] = static_cast<std::uint8_t>(dist(rng));
    }
}

// Fill an i8 tensor with ±1 values.
void fill_random_signs(ov::Tensor& t, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto* p = t.data<std::int8_t>();
    for (std::size_t i = 0; i < t.get_size(); ++i) {
        p[i] = (rng() & 1u) ? std::int8_t(1) : std::int8_t(-1);
    }
}

}  // namespace

class TurboQuantPackRoundtripTest
    : public ::testing::TestWithParam<std::tuple<std::uint8_t /*bits*/, std::size_t /*n*/>> {};

TEST_P(TurboQuantPackRoundtripTest, BitExactRoundtrip) {
    const auto [bits, n] = GetParam();
    const std::uint8_t max_val = static_cast<std::uint8_t>((1u << bits) - 1u);

    ov::Tensor src(ov::element::u8, ov::Shape{n});
    fill_random_u8(src, max_val, /*seed=*/bits * 100u + n);

    const auto packed = ov::npuw::turboquant::pack_bits(src, bits);
    EXPECT_EQ(packed.size(), ov::npuw::turboquant::packed_size_bytes(n, bits));

    ov::Tensor dst(ov::element::u8, ov::Shape{n});
    std::fill_n(dst.data<std::uint8_t>(), n, 0xFFu);  // poison
    ov::npuw::turboquant::unpack_bits(packed.data(), packed.size(), bits, dst);

    const auto* s = src.data<const std::uint8_t>();
    const auto* d = dst.data<const std::uint8_t>();
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(d[i], s[i]) << "mismatch at i=" << i << " bits=" << int(bits) << " n=" << n;
    }
}

TEST_P(TurboQuantPackRoundtripTest, CompressionRatioMatchesBitWidth) {
    const auto [bits, n] = GetParam();
    ov::Tensor src(ov::element::u8, ov::Shape{n});
    fill_random_u8(src, static_cast<std::uint8_t>((1u << bits) - 1u), /*seed=*/bits + n);

    const auto packed = ov::npuw::turboquant::pack_bits(src, bits);
    // Packed size matches ceil(n*bits / 8).
    EXPECT_EQ(packed.size(), (n * bits + 7u) / 8u);
    // It should never exceed the u8 source size.
    EXPECT_LE(packed.size(), n);
}

INSTANTIATE_TEST_SUITE_P(TurboQuant,
                         TurboQuantPackRoundtripTest,
                         ::testing::Values(std::make_tuple(std::uint8_t{1}, std::size_t{64}),
                                           std::make_tuple(std::uint8_t{2}, std::size_t{64}),
                                           std::make_tuple(std::uint8_t{2}, std::size_t{133}),
                                           std::make_tuple(std::uint8_t{3}, std::size_t{64}),
                                           std::make_tuple(std::uint8_t{3}, std::size_t{137}),
                                           std::make_tuple(std::uint8_t{4}, std::size_t{128}),
                                           std::make_tuple(std::uint8_t{5}, std::size_t{77}),
                                           std::make_tuple(std::uint8_t{8}, std::size_t{128})));

TEST(TurboQuantPackSignRoundtrip, BitExact) {
    constexpr std::size_t n = 257;  // not a multiple of 8
    ov::Tensor src(ov::element::i8, ov::Shape{n});
    fill_random_signs(src, /*seed=*/12345);

    const auto packed = ov::npuw::turboquant::pack_signs(src);
    EXPECT_EQ(packed.size(), (n + 7u) / 8u);

    ov::Tensor dst(ov::element::i8, ov::Shape{n});
    std::fill_n(dst.data<std::int8_t>(), n, std::int8_t(0));  // poison
    ov::npuw::turboquant::unpack_signs(packed.data(), packed.size(), dst);

    const auto* s = src.data<const std::int8_t>();
    const auto* d = dst.data<const std::int8_t>();
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(d[i], s[i]) << "sign mismatch at i=" << i;
    }
}

TEST(TurboQuantPackSizeHelpers, PackedSizeBytesMatchesCeil) {
    using ov::npuw::turboquant::packed_size_bytes;
    // Spot checks — values taken directly from the paper / integer math.
    EXPECT_EQ(packed_size_bytes(8, 3), 3u);    // 8*3=24 bits = 3 bytes
    EXPECT_EQ(packed_size_bytes(9, 3), 4u);    // 9*3=27 bits -> 4 bytes
    EXPECT_EQ(packed_size_bytes(16, 2), 4u);   // 16*2=32 bits = 4 bytes
    EXPECT_EQ(packed_size_bytes(17, 2), 5u);   // 17*2=34 bits -> 5 bytes
    EXPECT_EQ(packed_size_bytes(64, 8), 64u);  // 8-bit is a no-op
    EXPECT_EQ(packed_size_bytes(0, 3), 0u);
}
