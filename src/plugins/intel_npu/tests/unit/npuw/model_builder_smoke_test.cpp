// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Smoke coverage for test_engine model builders that no functional/unit test
// exercises yet.  These only check that the builders produce valid ov::Models
// with the expected interface — keeping the builders compiled, linked and
// structurally sane until real NPUW tests adopt them.

#include <gtest/gtest.h>

#include "model_builder.hpp"
#include "openvino/op/ops.hpp"

namespace {

using namespace ov::test::npuw;

template <typename OpType>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<OpType>(op)) {
            ++count;
        }
    }
    return count;
}

WhisperConfig make_whisper_test_config() {
    WhisperConfig cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.intermediate_size = 128;
    cfg.vocab_size = 128;
    cfg.num_mel_bins = 16;
    cfg.max_source_positions = 64;
    cfg.max_target_positions = 32;
    return cfg;
}

LLMConfig make_llm_test_config() {
    LLMConfig cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.intermediate_size = 128;
    cfg.vocab_size = 128;
    return cfg;
}

TEST(ModelBuilderSmoke, WhisperEncoder) {
    ModelBuilder mb;
    auto model = mb.build_whisper_encoder(make_whisper_test_config());
    ASSERT_NE(model, nullptr);

    EXPECT_NO_THROW(model->input("input_features"));
    EXPECT_NO_THROW(model->output("last_hidden_state"));
    // Conv1D preprocessing: two Convolution+Gelu stages in front of the layers
    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 2u);
    EXPECT_GE(count_ops<ov::op::v7::Gelu>(model), 2u);
}

TEST(ModelBuilderSmoke, WhisperEncoderF16KeepsF32Boundaries) {
    auto cfg = make_whisper_test_config();
    cfg.precision = ov::element::f16;

    ModelBuilder mb;
    auto model = mb.build_whisper_encoder(cfg);
    ASSERT_NE(model, nullptr);

    // WhisperPipeline talks f32 regardless of internal compute precision
    EXPECT_EQ(model->input("input_features").get_element_type(), ov::element::f32);
    EXPECT_EQ(model->output("last_hidden_state").get_element_type(), ov::element::f32);
}

TEST(ModelBuilderSmoke, WhisperEncoderDecoderSeqLenAgree) {
    auto cfg = make_whisper_test_config();

    ModelBuilder mb;
    auto encoder = mb.build_whisper_encoder(cfg);
    auto decoder = mb.build_whisper_decoder(cfg);
    ASSERT_NE(encoder, nullptr);
    ASSERT_NE(decoder, nullptr);

    const auto enc_out = encoder->output("last_hidden_state").get_partial_shape();
    const auto dec_in = decoder->input("encoder_hidden_states").get_partial_shape();
    ASSERT_EQ(enc_out.rank().get_length(), 3);
    ASSERT_EQ(dec_in.rank().get_length(), 3);
    EXPECT_EQ(enc_out[1], dec_in[1]) << "encoder seq len must match decoder's encoder_hidden_states";
    EXPECT_EQ(enc_out[2], dec_in[2]) << "encoder hidden size must match decoder's encoder_hidden_states";
}

TEST(ModelBuilderSmoke, LLMWithGELUFFN) {
    auto cfg = make_llm_test_config();
    cfg.ffn = GELU(cfg.hidden_size, cfg.intermediate_size, cfg.precision, cfg.weight);

    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(count_ops<ov::op::v7::Gelu>(model), cfg.num_layers);
}

TEST(ModelBuilderSmoke, LLMWithInterleavedRoPE) {
    auto cfg = make_llm_test_config();
    auto position_ids = make_position_ids_2d();
    cfg.position_ids = position_ids;
    cfg.rope = InterleavedRoPE(cfg.head_dim, cfg.precision, position_ids);

    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ASSERT_NE(model, nullptr);
    EXPECT_NO_THROW(model->input("position_ids"));
}

TEST(ModelBuilderSmoke, LLMWithMRoPEPositionIds3D) {
    auto cfg = make_llm_test_config();
    cfg.position_ids = make_position_ids_3d();

    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ASSERT_NE(model, nullptr);

    const auto pos_shape = model->input("position_ids").get_partial_shape();
    ASSERT_EQ(pos_shape.rank().get_length(), 3);
    EXPECT_EQ(pos_shape[0], 3) << "m-rope position_ids carry 3 rotary sections";
}

}  // namespace
