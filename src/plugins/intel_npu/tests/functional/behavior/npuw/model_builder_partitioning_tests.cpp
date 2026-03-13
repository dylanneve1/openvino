// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_tests.hpp"
#include "test_engine/comparators/nrmse.hpp"
#include "openvino/util/file_util.hpp"

using namespace testing;
using namespace ov::npuw::tests;
using namespace ov::intel_npu::npuw;
using namespace ov::test::npuw;

// ============================================================
// Parameterized test fixture: build model from ModelConfig,
// compile via TEMPLATE plugin through NPUW with online
// partitioning (REP vs NONE pipeline), infer, compare outputs.
// ============================================================

struct ModelConfigTestParam {
    std::string name;
    ModelConfig config;
};

class ModelBuilderPartitioningTests : public ::testing::TestWithParam<ModelConfigTestParam> {
public:
    void SetUp() override {
        auto plugin_path =
            ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                               std::string(ov::test::utils::TEMPLATE_LIB)
                                               + OV_BUILD_POSTFIX);
        if (!ov::util::file_exists(plugin_path)) {
            GTEST_SKIP() << "TEMPLATE plugin not found, skipping";
        }
        core.register_plugin(plugin_path, ov::test::utils::DEVICE_TEMPLATE);
    }

protected:
    ov::Core core;
    ModelBuilder model_builder;
};

TEST_P(ModelBuilderPartitioningTests, RepAndNonePipelinesMatch) {
    auto param = GetParam();
    auto model = model_builder.build_model(param.config);

    ov::AnyMap rep_props{::ov::intel_npu::use_npuw(true),
                         devices("TEMPLATE"),
                         partitioning::online::pipeline("REP")};
    ov::AnyMap none_props{::ov::intel_npu::use_npuw(true),
                          devices("TEMPLATE"),
                          partitioning::online::pipeline("NONE")};

    auto rep_compiled = core.compile_model(model, "NPU", rep_props);
    auto none_compiled = core.compile_model(model, "NPU", none_props);

    auto rep_req = rep_compiled.create_infer_request();
    auto none_req = none_compiled.create_infer_request();

    set_random_inputs<int32_t>(rep_req);
    set_random_inputs<int32_t>(none_req);

    rep_req.infer();
    none_req.infer();

    metrics::NRMSE nrmse(0.01);
    for (const auto& output : rep_req.get_compiled_model().outputs()) {
        EXPECT_TRUE(nrmse(rep_req.get_tensor(output), none_req.get_tensor(output)))
            << "Output mismatch for: " << output.get_any_name();
    }
}

// ============================================================
// Helper to build configs concisely
// ============================================================

static ModelConfig base_llm(size_t hs, size_t heads, size_t layers, size_t inter, size_t vocab = 1000) {
    ModelConfig c;
    c.hidden_size = hs;
    c.num_heads = heads;
    c.head_dim = hs / heads;
    c.intermediate_size = inter;
    c.num_layers = layers;
    c.vocab_size = vocab;
    return c;
}

static ModelConfig base_whisper_enc(size_t hs, size_t heads, size_t layers, size_t inter) {
    ModelConfig c = base_llm(hs, heads, layers, inter);
    c.use_conv_features = true;
    c.use_kv_cache = false;
    c.num_mel_bins = 80;
    c.max_source_positions = 128;
    return c;
}

static ModelConfig base_whisper_dec(size_t hs, size_t heads, size_t layers, size_t inter) {
    ModelConfig c = base_llm(hs, heads, layers, inter);
    c.use_cross_attention = true;
    c.encoder_layers = layers;
    c.decoder_layers = layers;
    c.max_target_positions = 64;
    return c;
}

static ModelConfig base_bert(size_t hs, size_t heads, size_t layers, size_t inter) {
    ModelConfig c = base_llm(hs, heads, layers, inter);
    c.use_token_type_embedding = true;
    c.use_kv_cache = false;
    c.pre_norm = false;
    c.max_position_embeddings = 128;
    c.type_vocab_size = 2;
    return c;
}

// ============================================================
// 100 test configurations
// ============================================================

// clang-format off
static std::vector<ModelConfigTestParam> generate_test_configs() {
    std::vector<ModelConfigTestParam> params;

    // --- LLM: basic dimension sweeps (1-20) ---
    {
        auto c = base_llm(64, 4, 2, 256);
        params.push_back({"LLM_tiny_2L", c});
    }
    {
        auto c = base_llm(64, 4, 4, 256);
        params.push_back({"LLM_tiny_4L", c});
    }
    {
        auto c = base_llm(64, 4, 8, 256);
        params.push_back({"LLM_tiny_8L", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        params.push_back({"LLM_small_2L", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        params.push_back({"LLM_small_4L", c});
    }
    {
        auto c = base_llm(128, 8, 6, 512);
        params.push_back({"LLM_small_6L", c});
    }
    {
        auto c = base_llm(256, 8, 2, 1024);
        params.push_back({"LLM_medium_2L", c});
    }
    {
        auto c = base_llm(256, 16, 4, 1024);
        params.push_back({"LLM_medium_16h_4L", c});
    }
    {
        auto c = base_llm(128, 4, 3, 512);
        params.push_back({"LLM_odd_layers_3L", c});
    }
    {
        auto c = base_llm(128, 4, 5, 512);
        params.push_back({"LLM_odd_layers_5L", c});
    }
    {
        auto c = base_llm(128, 4, 7, 512);
        params.push_back({"LLM_odd_layers_7L", c});
    }
    {
        auto c = base_llm(128, 4, 1, 512);
        params.push_back({"LLM_single_layer", c});
    }
    {
        auto c = base_llm(64, 2, 2, 128);
        params.push_back({"LLM_2heads", c});
    }
    {
        auto c = base_llm(256, 4, 2, 512);
        params.push_back({"LLM_large_head_dim", c});
    }
    {
        auto c = base_llm(64, 4, 2, 64);
        params.push_back({"LLM_small_intermediate", c});
    }
    {
        auto c = base_llm(64, 4, 2, 1024);
        params.push_back({"LLM_large_intermediate", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256, 500);
        params.push_back({"LLM_small_vocab", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256, 5000);
        params.push_back({"LLM_large_vocab", c});
    }
    {
        auto c = base_llm(128, 8, 10, 512);
        params.push_back({"LLM_10L_default", c});
    }
    {
        auto c = base_llm(64, 4, 12, 256);
        params.push_back({"LLM_12L", c});
    }

    // --- LLM: GQA variants (21-30) ---
    {
        auto c = base_llm(128, 8, 2, 512);
        c.num_kv_heads = 4;
        params.push_back({"LLM_GQA_8q_4kv", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.num_kv_heads = 2;
        params.push_back({"LLM_GQA_8q_2kv", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.num_kv_heads = 1;
        params.push_back({"LLM_GQA_8q_1kv_MQA", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.num_kv_heads = 4;
        params.push_back({"LLM_GQA_4L", c});
    }
    {
        auto c = base_llm(256, 16, 2, 1024);
        c.num_kv_heads = 4;
        params.push_back({"LLM_GQA_16q_4kv", c});
    }
    {
        auto c = base_llm(256, 16, 2, 1024);
        c.num_kv_heads = 8;
        params.push_back({"LLM_GQA_16q_8kv", c});
    }
    {
        auto c = base_llm(256, 16, 2, 1024);
        c.num_kv_heads = 2;
        params.push_back({"LLM_GQA_16q_2kv", c});
    }
    {
        auto c = base_llm(256, 16, 2, 1024);
        c.num_kv_heads = 1;
        params.push_back({"LLM_GQA_16q_1kv_MQA", c});
    }
    {
        auto c = base_llm(64, 4, 6, 256);
        c.num_kv_heads = 2;
        params.push_back({"LLM_GQA_tiny_6L", c});
    }
    {
        auto c = base_llm(128, 8, 8, 512);
        c.num_kv_heads = 2;
        params.push_back({"LLM_GQA_8L", c});
    }

    // --- LLM: precision and weight variants (31-45) ---
    {
        auto c = base_llm(64, 4, 2, 256);
        c.weight = FP16Weight{};
        params.push_back({"LLM_FP16_weight", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.weight = INT8Weight{};
        params.push_back({"LLM_INT8_weight", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.weight = INT4Weight{};
        params.push_back({"LLM_INT4_weight", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = INT4GroupWeight{128};
        params.push_back({"LLM_INT4_group128", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = INT4GroupWeight{64};
        params.push_back({"LLM_INT4_group64", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.weight = CompressedWeight(ov::element::i8, 0, DCOffPattern::SYMM_NO_ZP);
        params.push_back({"LLM_SYMM_NO_ZP_i8", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.weight = CompressedWeight(ov::element::i4, 0, DCOffPattern::SYMM_NO_ZP_F32);
        params.push_back({"LLM_SYMM_NO_ZP_F32_i4", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = CompressedWeight(ov::element::u4, 0, DCOffPattern::SYMM_ZP);
        params.push_back({"LLM_SYMM_ZP_u4", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::GPTQ);
        params.push_back({"LLM_GPTQ_u4_g128", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::ASYMM_ZP);
        params.push_back({"LLM_ASYMM_ZP_u4_g128", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.weight = INT8Weight{};
        c.num_kv_heads = 4;
        params.push_back({"LLM_INT8_GQA_4L", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.weight = INT4GroupWeight{128};
        c.num_kv_heads = 2;
        params.push_back({"LLM_INT4g128_GQA_4L", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = FP16Weight{};
        c.num_kv_heads = 4;
        params.push_back({"LLM_FP16_GQA", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        params.push_back({"LLM_f16_precision", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        c.num_kv_heads = 4;
        params.push_back({"LLM_f16_GQA", c});
    }

    // --- LLM: feature toggles (46-65) ---
    {
        auto c = base_llm(64, 4, 2, 256);
        c.use_kv_cache = false;
        params.push_back({"LLM_no_kv_cache", c});
    }
    {
        auto c = base_llm(64, 4, 4, 256);
        c.use_kv_cache = false;
        params.push_back({"LLM_no_kv_cache_4L", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.use_inputs_embeds = true;
        params.push_back({"LLM_inputs_embeds", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.use_inputs_embeds = true;
        c.num_kv_heads = 4;
        params.push_back({"LLM_inputs_embeds_GQA", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.pre_norm = false;
        params.push_back({"LLM_post_norm", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.pre_norm = false;
        params.push_back({"LLM_post_norm_4L", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.lm_head_weight = WeightFn{};  // no LM head
        params.push_back({"LLM_no_lm_head", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.norm = RMSNorm(64);
        params.push_back({"LLM_RMSNorm", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.norm = RMSNorm(128);
        params.push_back({"LLM_RMSNorm_4L", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.norm = RMSNorm(128);
        c.num_kv_heads = 4;
        params.push_back({"LLM_RMSNorm_GQA", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.norm = LayerNorm(64);
        params.push_back({"LLM_explicit_LayerNorm", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.ffn = GELU(64, 256, ov::element::f32, FP32Weight{});
        params.push_back({"LLM_GELU_FFN", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{});
        params.push_back({"LLM_GELU_FFN_4L", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{});
        c.norm = RMSNorm(128);
        params.push_back({"LLM_GELU_RMSNorm", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.internal_position_ids = true;
        params.push_back({"LLM_internal_pos_ids", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.use_kv_cache = false;
        c.pre_norm = false;
        params.push_back({"LLM_no_kv_post_norm", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.use_inputs_embeds = true;
        c.use_kv_cache = false;
        params.push_back({"LLM_embeds_no_kv", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = INT8Weight{};
        c.use_inputs_embeds = true;
        params.push_back({"LLM_INT8_embeds", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = FP16Weight{};
        c.pre_norm = false;
        c.norm = RMSNorm(128);
        params.push_back({"LLM_FP16_post_RMS", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.weight = INT4GroupWeight{128};
        c.norm = RMSNorm(128);
        c.num_kv_heads = 2;
        c.use_inputs_embeds = true;
        params.push_back({"LLM_INT4g_RMS_GQA_embeds", c});
    }

    // --- LLM: hidden_size hotfix regression (66-75) ---
    // Verify deferred norm/ffn defaults work when hidden_size is changed after construction
    {
        ModelConfig c;
        c.hidden_size = 128;
        c.num_heads = 8;
        c.head_dim = 16;
        c.intermediate_size = 512;
        c.num_layers = 2;
        params.push_back({"LLM_hs128_deferred_defaults", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 256;
        c.num_heads = 8;
        c.head_dim = 32;
        c.intermediate_size = 1024;
        c.num_layers = 2;
        params.push_back({"LLM_hs256_deferred_defaults", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 32;
        c.num_heads = 2;
        c.head_dim = 16;
        c.intermediate_size = 128;
        c.num_layers = 3;
        params.push_back({"LLM_hs32_deferred_defaults", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 96;
        c.num_heads = 6;
        c.head_dim = 16;
        c.intermediate_size = 384;
        c.num_layers = 2;
        params.push_back({"LLM_hs96_deferred_defaults", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 192;
        c.num_heads = 12;
        c.head_dim = 16;
        c.intermediate_size = 768;
        c.num_layers = 2;
        params.push_back({"LLM_hs192_deferred_defaults", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 128;
        c.num_heads = 4;
        c.head_dim = 32;
        c.intermediate_size = 256;
        c.num_layers = 4;
        c.weight = FP16Weight{};
        params.push_back({"LLM_hs128_FP16_deferred", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 64;
        c.num_heads = 4;
        c.head_dim = 16;
        c.intermediate_size = 128;
        c.num_layers = 2;
        c.weight = INT8Weight{};
        params.push_back({"LLM_hs64_INT8_deferred", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 128;
        c.num_heads = 8;
        c.head_dim = 16;
        c.intermediate_size = 512;
        c.num_layers = 2;
        c.num_kv_heads = 4;
        params.push_back({"LLM_hs128_GQA_deferred", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 256;
        c.num_heads = 16;
        c.head_dim = 16;
        c.intermediate_size = 512;
        c.num_layers = 3;
        c.pre_norm = false;
        params.push_back({"LLM_hs256_post_norm_deferred", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 128;
        c.num_heads = 8;
        c.head_dim = 16;
        c.intermediate_size = 512;
        c.num_layers = 2;
        c.use_inputs_embeds = true;
        c.num_kv_heads = 2;
        params.push_back({"LLM_hs128_embeds_GQA_deferred", c});
    }

    // --- Whisper encoder (76-85) ---
    {
        auto c = base_whisper_enc(64, 4, 2, 256);
        params.push_back({"Whisper_enc_tiny", c});
    }
    {
        auto c = base_whisper_enc(128, 8, 4, 512);
        params.push_back({"Whisper_enc_small_4L", c});
    }
    {
        auto c = base_whisper_enc(64, 4, 1, 256);
        params.push_back({"Whisper_enc_1L", c});
    }
    {
        auto c = base_whisper_enc(128, 8, 2, 512);
        c.weight = FP16Weight{};
        params.push_back({"Whisper_enc_FP16", c});
    }
    {
        auto c = base_whisper_enc(64, 4, 2, 256);
        c.weight = INT8Weight{};
        params.push_back({"Whisper_enc_INT8", c});
    }
    {
        auto c = base_whisper_enc(64, 4, 3, 256);
        params.push_back({"Whisper_enc_3L", c});
    }
    {
        auto c = base_whisper_enc(128, 4, 2, 512);
        params.push_back({"Whisper_enc_4h_large_hd", c});
    }
    {
        auto c = base_whisper_enc(256, 8, 2, 1024);
        params.push_back({"Whisper_enc_medium", c});
    }
    {
        auto c = base_whisper_enc(64, 4, 6, 256);
        params.push_back({"Whisper_enc_6L", c});
    }
    {
        auto c = base_whisper_enc(64, 2, 2, 128);
        params.push_back({"Whisper_enc_2heads", c});
    }

    // --- Whisper decoder (86-90) ---
    {
        auto c = base_whisper_dec(64, 4, 2, 256);
        params.push_back({"Whisper_dec_tiny", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        params.push_back({"Whisper_dec_small", c});
    }
    {
        auto c = base_whisper_dec(64, 4, 1, 256);
        params.push_back({"Whisper_dec_1L", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 4, 512);
        c.weight = FP16Weight{};
        params.push_back({"Whisper_dec_FP16_4L", c});
    }
    {
        auto c = base_whisper_dec(64, 4, 3, 256);
        params.push_back({"Whisper_dec_3L", c});
    }

    // --- BERT / embedding encoder (91-100) ---
    {
        auto c = base_bert(64, 4, 2, 256);
        params.push_back({"BERT_tiny", c});
    }
    {
        auto c = base_bert(128, 8, 4, 512);
        params.push_back({"BERT_small_4L", c});
    }
    {
        auto c = base_bert(64, 4, 1, 256);
        params.push_back({"BERT_1L", c});
    }
    {
        auto c = base_bert(256, 8, 2, 1024);
        params.push_back({"BERT_medium", c});
    }
    {
        auto c = base_bert(128, 8, 6, 512);
        params.push_back({"BERT_6L", c});
    }
    {
        auto c = base_bert(64, 4, 2, 256);
        c.weight = FP16Weight{};
        params.push_back({"BERT_FP16", c});
    }
    {
        auto c = base_bert(64, 4, 2, 256);
        c.weight = INT8Weight{};
        params.push_back({"BERT_INT8", c});
    }
    {
        auto c = base_bert(128, 4, 2, 512);
        params.push_back({"BERT_4h_large_hd", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.max_position_embeddings = 256;
        params.push_back({"BERT_pos256", c});
    }
    {
        auto c = base_bert(64, 2, 3, 128);
        c.type_vocab_size = 4;
        params.push_back({"BERT_2h_tvs4", c});
    }

    return params;
}
// clang-format on

INSTANTIATE_TEST_SUITE_P(
    ModelBuilderPartitioning,
    ModelBuilderPartitioningTests,
    ::testing::ValuesIn(generate_test_configs()),
    [](const ::testing::TestParamInfo<ModelConfigTestParam>& info) { return info.param.name; });
