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
// Additional fixture: Folding correctness
// Compiles with fold=true vs fold=false (both NONE pipeline),
// then compares outputs to verify loop folding is numerically correct.
// ============================================================

class ModelBuilderFoldingTests : public ::testing::TestWithParam<ModelConfigTestParam> {
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

TEST_P(ModelBuilderFoldingTests, FoldedMatchesUnfolded) {
    auto param = GetParam();
    auto model = model_builder.build_model(param.config);

    ov::AnyMap folded_props{::ov::intel_npu::use_npuw(true),
                            devices("TEMPLATE"),
                            partitioning::online::pipeline("NONE"),
                            partitioning::fold(true)};
    ov::AnyMap unfolded_props{::ov::intel_npu::use_npuw(true),
                              devices("TEMPLATE"),
                              partitioning::online::pipeline("NONE")};

    auto folded_compiled = core.compile_model(model, "NPU", folded_props);
    auto unfolded_compiled = core.compile_model(model, "NPU", unfolded_props);

    auto folded_req = folded_compiled.create_infer_request();
    auto unfolded_req = unfolded_compiled.create_infer_request();

    set_random_inputs<int32_t>(folded_req);
    set_random_inputs<int32_t>(unfolded_req);

    folded_req.infer();
    unfolded_req.infer();

    metrics::NRMSE nrmse(0.01);
    for (const auto& output : folded_req.get_compiled_model().outputs()) {
        EXPECT_TRUE(nrmse(folded_req.get_tensor(output), unfolded_req.get_tensor(output)))
            << "Folding mismatch for: " << output.get_any_name();
    }
}

// ============================================================
// Additional fixture: Build, compile, and infer sanity check.
// Validates that every model config builds without crashing,
// compiles, creates an infer request, and produces non-empty output.
// ============================================================

class ModelBuilderSanityTests : public ::testing::TestWithParam<ModelConfigTestParam> {
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

TEST_P(ModelBuilderSanityTests, ModelBuildsCompilesAndInfers) {
    auto param = GetParam();

    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = model_builder.build_model(param.config))
        << "build_model() threw for config: " << param.name;
    ASSERT_NE(model, nullptr);

    ov::AnyMap props{::ov::intel_npu::use_npuw(true),
                     devices("TEMPLATE"),
                     partitioning::online::pipeline("REP")};

    ov::CompiledModel compiled;
    ASSERT_NO_THROW(compiled = core.compile_model(model, "NPU", props))
        << "compile_model() threw for config: " << param.name;

    ov::InferRequest req;
    ASSERT_NO_THROW(req = compiled.create_infer_request())
        << "create_infer_request() threw for config: " << param.name;

    set_random_inputs<int32_t>(req);

    ASSERT_NO_THROW(req.infer())
        << "infer() threw for config: " << param.name;

    for (const auto& output : compiled.outputs()) {
        auto tensor = req.get_tensor(output);
        EXPECT_GT(tensor.get_size(), 0u)
            << "Empty output tensor for: " << output.get_any_name();
    }
}

// ============================================================
// Additional fixture: Fold + funcall_for_all (production path).
// Validates the fold + funcall code path used in real NPUW
// deployment produces outputs matching the simple NONE pipeline.
// ============================================================

class ModelBuilderFoldFuncallTests : public ::testing::TestWithParam<ModelConfigTestParam> {
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

TEST_P(ModelBuilderFoldFuncallTests, FoldFuncallMatchesNone) {
    auto param = GetParam();
    auto model = model_builder.build_model(param.config);

    ov::AnyMap fold_funcall_props{::ov::intel_npu::use_npuw(true),
                                  devices("TEMPLATE"),
                                  partitioning::online::pipeline("NONE"),
                                  partitioning::fold(true),
                                  partitioning::funcall_for_all(true)};
    ov::AnyMap none_props{::ov::intel_npu::use_npuw(true),
                          devices("TEMPLATE"),
                          partitioning::online::pipeline("NONE")};

    auto fold_compiled = core.compile_model(model, "NPU", fold_funcall_props);
    auto none_compiled = core.compile_model(model, "NPU", none_props);

    auto fold_req = fold_compiled.create_infer_request();
    auto none_req = none_compiled.create_infer_request();

    set_random_inputs<int32_t>(fold_req);
    set_random_inputs<int32_t>(none_req);

    fold_req.infer();
    none_req.infer();

    metrics::NRMSE nrmse(0.01);
    for (const auto& output : fold_req.get_compiled_model().outputs()) {
        EXPECT_TRUE(nrmse(fold_req.get_tensor(output), none_req.get_tensor(output)))
            << "Fold+funcall mismatch for: " << output.get_any_name();
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
// 300 test configurations
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

    // --- Edge case dimensions (101-110) ---
    {
        auto c = base_llm(2, 1, 1, 4, 2);
        params.push_back({"EdgeCase_MinimalDimensions", c});
    }
    {
        auto c = base_llm(256, 1, 1, 512, 3);
        params.push_back({"EdgeCase_SingleHeadLargeHeadDim", c});
    }
    {
        auto c = base_llm(18, 3, 1, 36, 5);
        params.push_back({"EdgeCase_OddHeadsNonPow2HeadDim", c});
    }
    {
        auto c = base_llm(4, 2, 1, 8, 1);
        params.push_back({"EdgeCase_VocabSizeOne", c});
    }
    {
        auto c = base_llm(64, 32, 1, 128, 10);
        params.push_back({"EdgeCase_MinHeadDimManyHeads", c});
    }
    {
        auto c = base_llm(16, 4, 2, 8, 50);
        params.push_back({"EdgeCase_IntermediateSmallerThanHidden", c});
    }
    {
        auto c = base_llm(32, 4, 1, 32, 100);
        params.push_back({"EdgeCase_HiddenEqualsIntermediate", c});
    }
    {
        auto c = base_llm(4, 1, 20, 8, 10);
        params.push_back({"EdgeCase_ManyLayersMinimalSize", c});
    }
    {
        auto c = base_llm(42, 7, 2, 84, 20);
        c.num_kv_heads = 1;
        params.push_back({"EdgeCase_GQASingleKVHead", c});
    }
    {
        auto c = base_llm(50, 5, 3, 100, 20);
        c.num_kv_heads = 1;
        params.push_back({"EdgeCase_NonAlignedHeadDim", c});
    }

    // --- Weight format stress tests (111-120) ---
    {
        auto c = base_llm(256, 4, 2, 512);
        c.precision = ov::element::f16;
        c.weight = INT4GroupWeight{64};
        c.lm_head_weight = FP16Weight{};
        params.push_back({"WtStress_F16PrecINT4Group64", c});
    }
    {
        auto c = base_llm(128, 4, 1, 256, 500);
        c.num_kv_heads = 2;
        c.weight = CompressedWeight(ov::element::u4, 64, DCOffPattern::SYMM_NO_ZP);
        params.push_back({"WtStress_GQA_SYMM_NO_ZP_U4", c});
    }
    {
        auto c = base_llm(256, 8, 3, 512);
        c.num_kv_heads = 2;
        c.precision = ov::element::f16;
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::GPTQ);
        c.lm_head_weight = FP16Weight{};
        params.push_back({"WtStress_F16_GQA_GPTQ", c});
    }
    {
        auto c = base_llm(128, 4, 2, 256);
        c.weight = CompressedWeight(ov::element::u4, 64, DCOffPattern::ASYMM_ZP);
        c.norm = LayerNorm(128);
        params.push_back({"WtStress_ASYMM_ZP_LayerNorm", c});
    }
    {
        auto c = base_llm(256, 4, 1, 512);
        c.weight = CompressedWeight(ov::element::i4, 0, DCOffPattern::SYMM_NO_ZP_F32);
        c.lm_head_weight = INT8Weight{};
        params.push_back({"WtStress_SYMM_NO_ZP_F32_MixedLM", c});
    }
    {
        auto c = base_whisper_enc(128, 4, 2, 256);
        c.weight = CompressedWeight(ov::element::u4, 64, DCOffPattern::SYMM_ZP);
        params.push_back({"WtStress_WhisperEnc_SYMM_ZP", c});
    }
    {
        auto c = base_whisper_dec(128, 4, 2, 256);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::GPTQ);
        params.push_back({"WtStress_WhisperDec_GPTQ", c});
    }
    {
        auto c = base_bert(128, 4, 2, 256);
        c.weight = CompressedWeight(ov::element::i4, 0, DCOffPattern::SYMM_NO_ZP);
        params.push_back({"WtStress_BERT_SYMM_NO_ZP", c});
    }
    {
        auto c = base_llm(256, 8, 4, 512);
        c.num_kv_heads = 2;
        c.norm = RMSNorm(256);
        c.weight = CompressedWeight(ov::element::u4, 64, DCOffPattern::SYMM_ZP);
        params.push_back({"WtStress_GQA_RMS_SYMM_ZP", c});
    }
    {
        auto c = base_llm(128, 4, 6, 256);
        c.num_kv_heads = 1;
        c.precision = ov::element::f16;
        c.norm = RMSNorm(128);
        c.weight = CompressedWeight(ov::element::u4, 64, DCOffPattern::ASYMM_ZP);
        c.lm_head_weight = INT4GroupWeight{64};
        params.push_back({"WtStress_Deep_ASYMM_INT4LMHead", c});
    }

    // --- Deferred defaults regression (121-130) ---
    {
        ModelConfig c;
        c.hidden_size = 384;
        c.num_heads = 12;
        c.head_dim = 32;
        c.intermediate_size = 1536;
        c.num_layers = 3;
        c.use_inputs_embeds = true;
        params.push_back({"Deferred_HS384_InputsEmbeds", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 160;
        c.num_heads = 10;
        c.head_dim = 16;
        c.num_kv_heads = 2;
        c.intermediate_size = 640;
        c.num_layers = 2;
        c.precision = ov::element::f16;
        c.weight = INT8Weight{};
        params.push_back({"Deferred_HS160_GQA_FP16_INT8", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 48;
        c.num_heads = 6;
        c.head_dim = 8;
        c.num_kv_heads = 1;
        c.intermediate_size = 192;
        c.num_layers = 2;
        c.pre_norm = false;
        params.push_back({"Deferred_HS48_MQA_PostNorm", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 512;
        c.num_heads = 16;
        c.head_dim = 32;
        c.intermediate_size = 2048;
        c.num_layers = 4;
        c.weight = INT8Weight{};
        params.push_back({"Deferred_HS512_INT8", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 96;
        c.num_heads = 12;
        c.head_dim = 8;
        c.num_kv_heads = 4;
        c.intermediate_size = 384;
        c.num_layers = 2;
        params.push_back({"Deferred_HS96_GQA_KV4", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 256;
        c.num_heads = 8;
        c.head_dim = 32;
        c.intermediate_size = 1024;
        c.num_layers = 3;
        c.precision = ov::element::f16;
        params.push_back({"Deferred_HS256_FP16", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 32;
        c.num_heads = 4;
        c.head_dim = 8;
        c.intermediate_size = 128;
        c.num_layers = 2;
        params.push_back({"Deferred_HS32_SmallerThanDefault", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 192;
        c.num_heads = 6;
        c.head_dim = 32;
        c.intermediate_size = 768;
        c.num_layers = 2;
        c.pre_norm = false;
        c.weight = FP16Weight{};
        params.push_back({"Deferred_HS192_PostNorm_FP16", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 1024;
        c.num_heads = 16;
        c.head_dim = 64;
        c.num_kv_heads = 4;
        c.intermediate_size = 4096;
        c.num_layers = 2;
        c.weight = INT8Weight{};
        c.use_inputs_embeds = true;
        c.pre_norm = false;
        params.push_back({"Deferred_HS1024_AllOpts", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 320;
        c.num_heads = 10;
        c.head_dim = 32;
        c.intermediate_size = 1280;
        c.num_layers = 3;
        c.weight = FP16Weight{};
        c.num_kv_heads = 5;
        params.push_back({"Deferred_HS320_GQA5", c});
    }

    // --- RoPE and position ID variants (131-140) ---
    {
        auto c = base_llm(256, 4, 1, 512);
        c.internal_position_ids = true;
        params.push_back({"RoPE_InternalPosIds_Basic", c});
    }
    {
        auto c = base_llm(512, 8, 2, 1024);
        c.internal_position_ids = true;
        c.num_kv_heads = 2;
        params.push_back({"RoPE_InternalPosIds_GQA", c});
    }
    {
        auto c = base_llm(384, 6, 1, 768);
        c.internal_position_ids = true;
        c.use_kv_cache = false;
        params.push_back({"RoPE_InternalPosIds_NoKVCache", c});
    }
    {
        auto c = base_llm(256, 4, 3, 512);
        c.internal_position_ids = true;
        c.use_inputs_embeds = true;
        params.push_back({"RoPE_InternalPosIds_Embeds", c});
    }
    {
        auto c = base_llm(512, 4, 1, 1024);
        c.internal_position_ids = true;
        params.push_back({"RoPE_InternalPosIds_HeadDim128", c});
    }
    {
        auto c = base_llm(256, 8, 1, 512);
        c.internal_position_ids = true;
        c.num_kv_heads = 4;
        params.push_back({"RoPE_InternalPosIds_HeadDim32_GQA", c});
    }
    {
        auto c = base_llm(384, 6, 2, 768);
        c.use_kv_cache = false;
        params.push_back({"RoPE_Default_NoKV_HeadDim64", c});
    }
    {
        auto c = base_llm(512, 8, 2, 1024);
        c.internal_position_ids = true;
        c.weight = FP16Weight{};
        c.precision = ov::element::f16;
        params.push_back({"RoPE_InternalPosIds_FP16", c});
    }
    {
        auto c = base_llm(256, 4, 1, 512);
        c.internal_position_ids = true;
        c.num_kv_heads = 1;
        c.use_inputs_embeds = true;
        params.push_back({"RoPE_InternalPosIds_MQA_Embeds", c});
    }
    {
        auto c = base_llm(768, 12, 2, 1536);
        c.internal_position_ids = true;
        c.num_kv_heads = 4;
        c.norm = RMSNorm(768);
        params.push_back({"RoPE_InternalPosIds_GQA3_RMS", c});
    }

    // --- KV cache edge cases (141-150) ---
    {
        auto c = base_llm(512, 16, 4, 1024);
        c.num_kv_heads = 1;
        c.norm = RMSNorm(512);
        params.push_back({"KVCache_MQA_16q_1kv", c});
    }
    {
        auto c = base_llm(256, 8, 16, 512);
        c.use_kv_cache = false;
        c.ffn = GELU(256, 512, ov::element::f32, FP32Weight{});
        params.push_back({"KVCache_Off_16Layers_GELU", c});
    }
    {
        auto c = base_llm(384, 12, 3, 768);
        c.num_kv_heads = 4;
        c.use_inputs_embeds = true;
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        params.push_back({"KVCache_Embeds_GQA_FP16", c});
    }
    {
        auto c = base_llm(512, 16, 4, 1024);
        c.num_kv_heads = 4;
        c.weight = INT4GroupWeight{64};
        params.push_back({"KVCache_INT4Group_GQA", c});
    }
    {
        auto c = base_llm(256, 8, 5, 512);
        c.num_kv_heads = 2;
        c.pre_norm = false;
        c.ffn = GELU(256, 512, ov::element::f32, FP32Weight{});
        params.push_back({"KVCache_PostNorm_GELU", c});
    }
    {
        auto c = base_llm(128, 4, 32, 256);
        c.num_kv_heads = 2;
        c.norm = RMSNorm(128);
        params.push_back({"KVCache_32Layers_GQA", c});
    }
    {
        auto c = base_llm(512, 16, 6, 1024);
        c.num_kv_heads = 2;
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        c.norm = RMSNorm(512);
        params.push_back({"KVCache_GQA_8to1_FP16_RMS", c});
    }
    {
        auto c = base_llm(384, 12, 4, 768);
        c.num_kv_heads = 3;
        c.internal_position_ids = true;
        c.weight = INT8Weight{};
        params.push_back({"KVCache_INT8_InternalPos_GQA", c});
    }
    {
        auto c = base_llm(256, 8, 6, 512);
        c.num_kv_heads = 1;
        c.use_kv_cache = false;
        c.norm = RMSNorm(256);
        params.push_back({"KVCache_Off_MQA", c});
    }
    {
        auto c = base_llm(512, 16, 8, 1024);
        c.num_kv_heads = 1;
        c.use_inputs_embeds = true;
        c.lm_head_weight = WeightFn{};
        c.weight = INT4GroupWeight{64};
        params.push_back({"KVCache_MQA_NoLMHead_Embeds_INT4g", c});
    }

    // --- Whisper stress tests (151-160) ---
    {
        auto c = base_whisper_enc(128, 4, 2, 512);
        c.num_mel_bins = 40;
        params.push_back({"WhisperStress_Enc_40Mels", c});
    }
    {
        auto c = base_whisper_enc(256, 8, 6, 1024);
        params.push_back({"WhisperStress_Enc_6L_Medium", c});
    }
    {
        auto c = base_whisper_enc(64, 4, 1, 256);
        c.weight = INT8Weight{};
        c.max_source_positions = 64;
        params.push_back({"WhisperStress_Enc_INT8_ShortSrc", c});
    }
    {
        auto c = base_whisper_enc(128, 4, 4, 512);
        c.weight = FP16Weight{};
        c.num_mel_bins = 128;
        params.push_back({"WhisperStress_Enc_FP16_128Mels", c});
    }
    {
        auto c = base_whisper_dec(128, 4, 2, 512);
        c.encoder_layers = 4;
        c.decoder_layers = 2;
        params.push_back({"WhisperStress_Dec_AsymLayers_4e2d", c});
    }
    {
        auto c = base_whisper_dec(64, 4, 1, 256);
        c.max_target_positions = 16;
        params.push_back({"WhisperStress_Dec_ShortTarget", c});
    }
    {
        auto c = base_whisper_dec(256, 8, 4, 1024);
        c.weight = FP16Weight{};
        params.push_back({"WhisperStress_Dec_FP16_4L", c});
    }
    {
        auto c = base_whisper_enc(64, 2, 8, 128);
        params.push_back({"WhisperStress_Enc_Deep_2h", c});
    }
    {
        auto c = base_whisper_dec(128, 4, 3, 512);
        c.encoder_layers = 6;
        c.decoder_layers = 3;
        c.weight = INT8Weight{};
        params.push_back({"WhisperStress_Dec_INT8_AsymLayers", c});
    }
    {
        auto c = base_whisper_enc(384, 12, 2, 1536);
        c.max_source_positions = 256;
        params.push_back({"WhisperStress_Enc_Large_LongSrc", c});
    }

    // --- BERT stress tests (161-170) ---
    {
        auto c = base_bert(64, 2, 1, 256);
        c.type_vocab_size = 64;
        c.max_position_embeddings = 8;
        c.ffn = GELU(64, 256, ov::element::f32, FP32Weight{});
        params.push_back({"BertStress_TinyLargeTypeVocab", c});
    }
    {
        auto c = base_bert(128, 4, 2, 512);
        c.max_position_embeddings = 8192;
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        params.push_back({"BertStress_LargePosEmb_FP16", c});
    }
    {
        auto c = base_bert(96, 1, 12, 384);
        c.weight = INT8Weight{};
        c.ffn = GELU(96, 384, ov::element::f32, INT8Weight{});
        params.push_back({"BertStress_SingleHeadDeep_INT8", c});
    }
    {
        auto c = base_bert(192, 6, 4, 768);
        c.type_vocab_size = 16;
        c.weight = INT4Weight{};
        params.push_back({"BertStress_INT4_TypeVocab16", c});
    }
    {
        auto c = base_bert(256, 8, 3, 1024);
        c.max_position_embeddings = 1;
        c.weight = FP16Weight{};
        params.push_back({"BertStress_SinglePosEmb", c});
    }
    {
        auto c = base_bert(384, 12, 8, 1536);
        c.type_vocab_size = 128;
        c.weight = FP16Weight{};
        params.push_back({"BertStress_HugeTypeVocab_Deep", c});
    }
    {
        auto c = base_bert(2, 1, 1, 4);
        c.type_vocab_size = 1;
        c.max_position_embeddings = 2;
        c.vocab_size = 16;
        params.push_back({"BertStress_MinimalDimensions", c});
    }
    {
        auto c = base_bert(768, 32, 6, 3072);
        c.weight = INT8Weight{};
        c.lm_head_weight = WeightFn{};
        c.ffn = GELU(768, 3072, ov::element::f32, INT8Weight{});
        params.push_back({"BertStress_Large_INT8_NoLMHead", c});
    }
    {
        auto c = base_bert(320, 10, 5, 1280);
        c.type_vocab_size = 32;
        c.max_position_embeddings = 4096;
        c.weight = INT4Weight{};
        params.push_back({"BertStress_OddHidden_LargePos", c});
    }
    {
        auto c = base_bert(128, 4, 2, 512);
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{}, FP32Weight{});
        params.push_back({"BertStress_GELU_WithBias", c});
    }

    // --- Cross-feature interactions (171-180) ---
    {
        auto c = base_llm(512, 8, 4, 1024);
        c.num_kv_heads = 2;
        c.use_inputs_embeds = true;
        c.pre_norm = false;
        c.norm = RMSNorm(512);
        params.push_back({"CrossFeat_GQA_Embeds_PostNorm_RMS", c});
    }
    {
        auto c = base_llm(256, 8, 12, 512);
        c.num_kv_heads = 4;
        c.weight = INT4Weight{};
        c.ffn = GELU(256, 512, ov::element::f32, INT4Weight{});
        c.norm = RMSNorm(256);
        params.push_back({"CrossFeat_INT4_GQA_GELU_12L", c});
    }
    {
        auto c = base_llm(128, 4, 3, 256);
        c.precision = ov::element::f16;
        c.weight = INT8Weight{};
        c.internal_position_ids = true;
        c.lm_head_weight = WeightFn{};
        c.norm = RMSNorm(128);
        params.push_back({"CrossFeat_F16_INT8_IntPos_NoLMHead", c});
    }
    {
        auto c = base_llm(256, 4, 2, 512);
        c.pre_norm = false;
        c.use_kv_cache = false;
        c.weight = FP16Weight{};
        c.ffn = GELU(256, 512, ov::element::f32, FP16Weight{});
        params.push_back({"CrossFeat_PostNorm_GELU_NoKV_FP16", c});
    }
    {
        auto c = base_llm(512, 8, 4, 1024);
        c.num_kv_heads = 4;
        c.use_inputs_embeds = true;
        c.internal_position_ids = true;
        c.precision = ov::element::f16;
        c.weight = INT4Weight{};
        c.norm = RMSNorm(512);
        params.push_back({"CrossFeat_AllFeatures_F16_INT4", c});
    }
    {
        auto c = base_llm(256, 8, 3, 512);
        c.num_kv_heads = 2;
        c.use_inputs_embeds = true;
        c.norm = RMSNorm(256);
        c.weight = INT4GroupWeight{64};
        c.ffn = GELU(256, 512, ov::element::f32, INT4GroupWeight{64});
        params.push_back({"CrossFeat_RMS_GELU_GQA_INT4g_Embeds", c});
    }
    {
        auto c = base_llm(128, 4, 5, 384);
        c.pre_norm = false;
        c.use_kv_cache = false;
        c.internal_position_ids = true;
        c.weight = INT8Weight{};
        params.push_back({"CrossFeat_PostNorm_INT8_NoKV_IntPos", c});
    }
    {
        auto c = base_llm(384, 12, 2, 768);
        c.precision = ov::element::f16;
        c.num_kv_heads = 3;
        c.use_inputs_embeds = true;
        c.lm_head_weight = WeightFn{};
        c.ffn = GELU(384, 768, ov::element::f16, FP32Weight{});
        params.push_back({"CrossFeat_F16_GQA_GELU_Embeds_NoLM", c});
    }
    {
        auto c = base_llm(512, 16, 8, 1024);
        c.num_kv_heads = 4;
        c.pre_norm = false;
        c.use_kv_cache = false;
        c.internal_position_ids = true;
        c.weight = INT4GroupWeight{128};
        c.norm = RMSNorm(512);
        params.push_back({"CrossFeat_INT4g_PostNorm_NoKV_IntPos_Deep", c});
    }
    {
        auto c = base_llm(256, 8, 4, 512);
        c.num_kv_heads = 2;
        c.precision = ov::element::f16;
        c.use_inputs_embeds = true;
        c.internal_position_ids = true;
        c.weight = INT4Weight{};
        c.norm = RMSNorm(256);
        c.ffn = GELU(256, 512, ov::element::f16, INT4Weight{});
        params.push_back({"CrossFeat_AllFlags_INT4_GELU_RMS", c});
    }

    // --- Norm and FFN variant combinations (181-190) ---
    {
        auto c = base_llm(256, 4, 2, 512);
        c.norm = RMSNorm(256);
        c.ffn = GELU(256, 512, ov::element::f32, FP32Weight{});
        params.push_back({"NormFFN_RMSNorm_GELU", c});
    }
    {
        auto c = base_llm(512, 8, 2, 1024);
        c.norm = LayerNorm(512);
        c.ffn = SwiGLU(512, 1024, ov::element::f32, FP32Weight{});
        params.push_back({"NormFFN_LayerNorm_SwiGLU", c});
    }
    {
        auto c = base_llm(256, 4, 2, 512);
        c.norm = RMSNorm(256);
        c.ffn = SwiGLU(256, 512, ov::element::f16, FP16Weight{});
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        params.push_back({"NormFFN_RMS_SwiGLU_FP16", c});
    }
    {
        auto c = base_llm(128, 4, 2, 256);
        c.norm = RMSNorm(128);
        c.ffn = GELU(128, 256, ov::element::f32, FP32Weight{}, FP32Weight{});
        params.push_back({"NormFFN_RMS_GELU_WithBias", c});
    }
    {
        auto c = base_llm(512, 8, 2, 1024);
        c.norm = LayerNorm(512, ov::element::f32, 1e-6f);
        c.ffn = GELU(512, 1024, ov::element::f32, FP32Weight{});
        params.push_back({"NormFFN_LayerNorm_CustomEps_GELU", c});
    }
    {
        auto c = base_llm(256, 4, 2, 512);
        c.norm = RMSNorm(256, ov::element::f16);
        c.ffn = SwiGLU(256, 512, ov::element::f16, FP16Weight{});
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        params.push_back({"NormFFN_RMS_f16_SwiGLU_f16", c});
    }
    {
        auto c = base_llm(256, 4, 2, 512);
        c.norm = RMSNorm(256);
        c.ffn = GELU(256, 512, ov::element::f32, INT8Weight{});
        params.push_back({"NormFFN_RMS_GELU_INT8", c});
    }
    {
        auto c = base_llm(64, 2, 2, 512);
        c.norm = LayerNorm(64);
        c.ffn = SwiGLU(64, 512, ov::element::f32, FP32Weight{});
        params.push_back({"NormFFN_LayerNorm_SwiGLU_8xInter", c});
    }
    {
        auto c = base_llm(256, 4, 2, 512);
        c.norm = RMSNorm(256);
        c.ffn = GELU(256, 512, ov::element::f32, FP32Weight{}, FP32Weight{});
        c.pre_norm = false;
        params.push_back({"NormFFN_RMS_GELU_Bias_PostNorm", c});
    }
    {
        auto c = base_llm(512, 8, 3, 1024);
        c.num_kv_heads = 2;
        c.norm = LayerNorm(512);
        c.ffn = SwiGLU(512, 1024, ov::element::f16, FP16Weight{});
        params.push_back({"NormFFN_LayerNorm_SwiGLU_GQA", c});
    }

    // --- Many-layer deep models (191-200) ---
    {
        auto c = base_llm(64, 4, 16, 256);
        params.push_back({"Deep_LLM_16L", c});
    }
    {
        auto c = base_llm(32, 2, 20, 128);
        params.push_back({"Deep_LLM_20L_tiny", c});
    }
    {
        auto c = base_llm(64, 4, 24, 256);
        c.num_kv_heads = 2;
        params.push_back({"Deep_LLM_24L_GQA", c});
    }
    {
        auto c = base_llm(32, 2, 32, 128);
        c.weight = INT4Weight{};
        params.push_back({"Deep_LLM_32L_INT4", c});
    }
    {
        auto c = base_llm(64, 4, 13, 256);
        params.push_back({"Deep_LLM_13L_prime", c});
    }
    {
        auto c = base_llm(64, 4, 17, 256);
        c.norm = RMSNorm(64);
        params.push_back({"Deep_LLM_17L_prime_RMS", c});
    }
    {
        auto c = base_llm(64, 4, 19, 256);
        c.use_kv_cache = false;
        params.push_back({"Deep_LLM_19L_NoKV", c});
    }
    {
        auto c = base_llm(64, 4, 23, 256);
        c.num_kv_heads = 1;
        c.weight = FP16Weight{};
        params.push_back({"Deep_LLM_23L_MQA_FP16", c});
    }
    {
        auto c = base_bert(64, 4, 16, 256);
        params.push_back({"Deep_BERT_16L", c});
    }
    {
        auto c = base_whisper_enc(64, 4, 12, 256);
        params.push_back({"Deep_WhisperEnc_12L", c});
    }

    // --- Attention bias tests (201-215) ---
    {
        auto c = base_llm(64, 4, 2, 256);
        c.attn_bias = FP32Weight{};
        params.push_back({"AttnBias_FP32_Basic", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.attn_bias = FP16Weight{};
        params.push_back({"AttnBias_FP16", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.attn_bias = INT8Weight{};
        params.push_back({"AttnBias_INT8", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.attn_bias = FP32Weight{};
        c.num_kv_heads = 4;
        params.push_back({"AttnBias_GQA", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.attn_bias = FP32Weight{};
        c.num_kv_heads = 1;
        params.push_back({"AttnBias_MQA", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.attn_bias = FP32Weight{};
        c.pre_norm = false;
        params.push_back({"AttnBias_PostNorm", c});
    }
    {
        auto c = base_llm(64, 4, 3, 256);
        c.attn_bias = FP32Weight{};
        c.use_kv_cache = false;
        params.push_back({"AttnBias_NoKVCache", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.attn_bias = FP32Weight{};
        c.weight = INT4Weight{};
        params.push_back({"AttnBias_INT4_Body", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.attn_bias = FP16Weight{};
        c.norm = RMSNorm(128);
        params.push_back({"AttnBias_FP16_RMSNorm", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.attn_bias = FP32Weight{};
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{});
        params.push_back({"AttnBias_GELU_FFN", c});
    }
    {
        auto c = base_llm(256, 8, 2, 512);
        c.attn_bias = FP32Weight{};
        c.use_inputs_embeds = true;
        params.push_back({"AttnBias_InputsEmbeds", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.attn_bias = FP32Weight{};
        c.internal_position_ids = true;
        params.push_back({"AttnBias_InternalPosIds", c});
    }
    {
        auto c = base_llm(64, 4, 8, 256);
        c.attn_bias = FP32Weight{};
        c.norm = RMSNorm(64);
        c.num_kv_heads = 2;
        params.push_back({"AttnBias_Deep_RMS_GQA", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.attn_bias = FP32Weight{};
        params.push_back({"AttnBias_BERT", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.attn_bias = FP32Weight{};
        params.push_back({"AttnBias_WhisperDec", c});
    }

    // --- QK normalization tests (216-230) ---
    {
        auto c = base_llm(64, 4, 2, 256);
        c.qk_norm = RMSNorm(16);  // head_dim = 64/4 = 16
        params.push_back({"QKNorm_RMS_Basic", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.qk_norm = LayerNorm(16);  // head_dim = 128/8 = 16
        params.push_back({"QKNorm_LayerNorm_Basic", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.qk_norm = RMSNorm(16);
        c.num_kv_heads = 4;
        params.push_back({"QKNorm_RMS_GQA", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.qk_norm = RMSNorm(16);
        c.num_kv_heads = 1;
        params.push_back({"QKNorm_RMS_MQA", c});
    }
    {
        auto c = base_llm(256, 8, 2, 512);
        c.qk_norm = RMSNorm(32);  // head_dim = 256/8 = 32
        params.push_back({"QKNorm_RMS_HeadDim32", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.qk_norm = RMSNorm(16);
        c.attn_bias = FP32Weight{};
        params.push_back({"QKNorm_AttnBias_Combined", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.qk_norm = RMSNorm(16);
        c.attn_bias = FP32Weight{};
        c.num_kv_heads = 4;
        c.weight = INT4Weight{};
        params.push_back({"QKNorm_AttnBias_GQA_INT4", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.qk_norm = LayerNorm(16);
        c.norm = RMSNorm(128);
        params.push_back({"QKNorm_LayerNorm_BodyRMS", c});
    }
    {
        auto c = base_llm(64, 4, 6, 256);
        c.qk_norm = RMSNorm(16);
        c.ffn = GELU(64, 256, ov::element::f32, FP32Weight{});
        params.push_back({"QKNorm_RMS_GELU_6L", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.qk_norm = RMSNorm(16);
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        params.push_back({"QKNorm_RMS_FP16", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.qk_norm = RMSNorm(16);
        c.pre_norm = false;
        params.push_back({"QKNorm_PostNorm", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.qk_norm = RMSNorm(16);
        c.use_kv_cache = false;
        params.push_back({"QKNorm_NoKVCache", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.qk_norm = RMSNorm(16);
        c.internal_position_ids = true;
        params.push_back({"QKNorm_InternalPosIds", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.qk_norm = RMSNorm(16);
        params.push_back({"QKNorm_WhisperDec", c});
    }
    {
        auto c = base_llm(128, 8, 10, 512);
        c.qk_norm = RMSNorm(16);
        c.attn_bias = FP32Weight{};
        c.num_kv_heads = 2;
        c.norm = RMSNorm(128);
        c.internal_position_ids = true;
        params.push_back({"QKNorm_AttnBias_AllOpts_Deep", c});
    }

    // --- Whisper advanced features (231-245) ---
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.num_kv_heads = 4;
        params.push_back({"WhisperAdv_Dec_GQA", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.num_kv_heads = 1;
        params.push_back({"WhisperAdv_Dec_MQA", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{});
        params.push_back({"WhisperAdv_Dec_GELU", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.norm = RMSNorm(128);
        params.push_back({"WhisperAdv_Dec_RMSNorm", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.attn_bias = FP32Weight{};
        params.push_back({"WhisperAdv_Dec_AttnBias", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.qk_norm = RMSNorm(16);
        params.push_back({"WhisperAdv_Dec_QKNorm", c});
    }
    {
        auto c = base_whisper_enc(128, 8, 2, 512);
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{});
        params.push_back({"WhisperAdv_Enc_GELU", c});
    }
    {
        auto c = base_whisper_enc(128, 8, 2, 512);
        c.norm = RMSNorm(128);
        params.push_back({"WhisperAdv_Enc_RMSNorm", c});
    }
    {
        auto c = base_whisper_enc(64, 4, 12, 256);
        c.weight = INT4Weight{};
        params.push_back({"WhisperAdv_Enc_Deep12L_INT4", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.encoder_layers = 6;
        c.decoder_layers = 2;
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{});
        params.push_back({"WhisperAdv_Dec_AsymLayers_GELU", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.lm_head_weight = WeightFn{};
        params.push_back({"WhisperAdv_Dec_NoLMHead", c});
    }
    {
        auto c = base_whisper_enc(128, 8, 2, 512);
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(16);
        params.push_back({"WhisperAdv_Enc_AttnBias_QKNorm", c});
    }
    {
        auto c = base_whisper_dec(256, 8, 2, 1024);
        c.precision = ov::element::f16;
        c.num_kv_heads = 4;
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::SYMM_ZP);
        params.push_back({"WhisperAdv_Dec_FP16_GQA_SYMM_ZP", c});
    }
    {
        auto c = base_whisper_enc(128, 8, 2, 512);
        c.pre_norm = false;
        params.push_back({"WhisperAdv_Enc_PostNorm", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.num_kv_heads = 4;
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(16);
        c.norm = RMSNorm(128);
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{});
        params.push_back({"WhisperAdv_Dec_AllFeatures", c});
    }

    // --- BERT advanced (246-258) ---
    {
        auto c = base_bert(256, 8, 2, 1024);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::GPTQ);
        params.push_back({"BertAdv_GPTQ", c});
    }
    {
        auto c = base_bert(256, 8, 2, 1024);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::ASYMM_ZP);
        params.push_back({"BertAdv_ASYMM_ZP", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.attn_bias = FP32Weight{};
        params.push_back({"BertAdv_AttnBias", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.qk_norm = RMSNorm(16);
        params.push_back({"BertAdv_QKNorm_RMS", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.norm = RMSNorm(128);
        params.push_back({"BertAdv_RMSNorm", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.ffn = SwiGLU(128, 512, ov::element::f32, FP32Weight{});
        params.push_back({"BertAdv_SwiGLU", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.pre_norm = true;
        params.push_back({"BertAdv_PreNorm", c});
    }
    {
        auto c = base_bert(128, 8, 8, 512);
        c.attn_bias = FP32Weight{};
        c.qk_norm = LayerNorm(16);
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{}, FP32Weight{});
        params.push_back({"BertAdv_AttnBias_QKNorm_GELU_Bias_Deep", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.norm = RMSNorm(128);
        c.ffn = SwiGLU(128, 512, ov::element::f32, INT8Weight{});
        c.weight = INT8Weight{};
        params.push_back({"BertAdv_RMS_SwiGLU_INT8", c});
    }
    {
        auto c = base_bert(256, 8, 4, 1024);
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(32);
        c.norm = RMSNorm(256);
        c.weight = INT4Weight{};
        c.type_vocab_size = 8;
        c.max_position_embeddings = 512;
        params.push_back({"BertAdv_AllFeatures_INT4", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.use_inputs_embeds = true;
        c.weight = FP16Weight{};
        params.push_back({"BertAdv_InputsEmbeds_FP16", c});
    }
    {
        auto c = base_bert(128, 4, 2, 512);
        c.pre_norm = true;
        c.norm = RMSNorm(128);
        c.qk_norm = RMSNorm(32);
        c.attn_bias = FP32Weight{};
        c.ffn = SwiGLU(128, 512, ov::element::f32, INT8Weight{});
        c.weight = INT8Weight{};
        c.max_position_embeddings = 256;
        params.push_back({"BertAdv_PreNorm_RMS_AllOpts", c});
    }
    {
        auto c = base_bert(64, 4, 2, 256);
        c.lm_head_weight = WeightFn{};
        c.ffn = GELU(64, 256, ov::element::f32, FP32Weight{}, FP32Weight{});
        params.push_back({"BertAdv_NoLMHead_GELU_Bias", c});
    }

    // --- Mixed LM head weight types (259-270) ---
    {
        auto c = base_llm(64, 4, 2, 256);
        c.weight = INT4Weight{};
        c.lm_head_weight = FP32Weight{};
        params.push_back({"MixedLM_INT4body_FP32head", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = INT8Weight{};
        c.lm_head_weight = FP16Weight{};
        params.push_back({"MixedLM_INT8body_FP16head", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::GPTQ);
        c.lm_head_weight = INT8Weight{};
        params.push_back({"MixedLM_GPTQbody_INT8head", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = FP16Weight{};
        c.lm_head_weight = INT4Weight{};
        params.push_back({"MixedLM_FP16body_INT4head", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.weight = CompressedWeight(ov::element::i4, 0, DCOffPattern::SYMM_NO_ZP);
        c.lm_head_weight = FP16Weight{};
        params.push_back({"MixedLM_SYMM_body_FP16head", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = INT4GroupWeight{64};
        c.lm_head_weight = FP32Weight{};
        params.push_back({"MixedLM_INT4g64body_FP32head", c});
    }
    {
        auto c = base_llm(256, 8, 2, 1024);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::ASYMM_ZP);
        c.lm_head_weight = FP16Weight{};
        c.precision = ov::element::f16;
        c.num_kv_heads = 4;
        params.push_back({"MixedLM_ASYMM_FP16head_GQA", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.weight = INT4Weight{};
        c.lm_head_weight = INT8Weight{};
        c.num_kv_heads = 2;
        c.norm = RMSNorm(128);
        params.push_back({"MixedLM_INT4_INT8head_GQA_RMS", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.weight = CompressedWeight(ov::element::i8, 0, DCOffPattern::SYMM_NO_ZP);
        c.lm_head_weight = FP32Weight{};
        c.ffn = GELU(128, 512, ov::element::f32, CompressedWeight(ov::element::i8, 0, DCOffPattern::SYMM_NO_ZP));
        params.push_back({"MixedLM_SYMM_i8_FP32head_GELU", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.weight = INT4Weight{};
        c.lm_head_weight = FP32Weight{};
        params.push_back({"MixedLM_WhisperDec_INT4_FP32head", c});
    }
    {
        auto c = base_bert(128, 8, 2, 512);
        c.weight = INT8Weight{};
        c.lm_head_weight = FP16Weight{};
        params.push_back({"MixedLM_BERT_INT8_FP16head", c});
    }
    {
        auto c = base_llm(256, 8, 2, 512);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::SYMM_ZP);
        c.lm_head_weight = FP32Weight{};
        c.attn_bias = FP32Weight{};
        params.push_back({"MixedLM_SYMM_ZP_FP32head_AttnBias", c});
    }

    // --- Kitchen sink / extreme stress tests (271-285) ---
    {
        auto c = base_llm(128, 8, 8, 512);
        c.num_kv_heads = 2;
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(16);
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{}, FP32Weight{});
        c.norm = RMSNorm(128);
        c.internal_position_ids = true;
        c.use_inputs_embeds = true;
        c.weight = INT4Weight{};
        c.precision = ov::element::f16;
        params.push_back({"KitchenSink_LLM_AllFeatures", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.num_kv_heads = 1;
        c.attn_bias = FP16Weight{};
        c.qk_norm = LayerNorm(16);
        c.pre_norm = false;
        c.lm_head_weight = WeightFn{};
        c.weight = INT8Weight{};
        params.push_back({"KitchenSink_MQA_PostNorm_NoLM", c});
    }
    {
        auto c = base_llm(64, 4, 32, 256);
        c.num_kv_heads = 2;
        c.norm = RMSNorm(64);
        c.weight = INT4Weight{};
        c.ffn = GELU(64, 256, ov::element::f32, INT4Weight{});
        c.use_inputs_embeds = true;
        params.push_back({"KitchenSink_32L_GQA_INT4_GELU", c});
    }
    {
        auto c = base_llm(256, 8, 4, 1024);
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        c.ffn = GELU(256, 1024, ov::element::f16, FP16Weight{}, FP16Weight{});
        c.norm = LayerNorm(256, ov::element::f16, 1e-6f);
        c.internal_position_ids = true;
        c.num_kv_heads = 4;
        params.push_back({"KitchenSink_F16_GELU_Bias_CustomEps_GQA", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.pre_norm = false;
        c.use_kv_cache = false;
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(16);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::ASYMM_ZP);
        c.ffn = GELU(128, 512, ov::element::f32, CompressedWeight(ov::element::u4, 128, DCOffPattern::ASYMM_ZP));
        params.push_back({"KitchenSink_PostNorm_NoKV_ASYMM_GELU", c});
    }
    {
        auto c = base_whisper_enc(128, 8, 4, 512);
        c.weight = INT8Weight{};
        c.norm = RMSNorm(128);
        c.ffn = GELU(128, 512, ov::element::f32, INT8Weight{});
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(16);
        c.num_mel_bins = 128;
        params.push_back({"KitchenSink_WhisperEnc_INT8_AllOpts", c});
    }
    {
        auto c = base_whisper_dec(128, 8, 2, 512);
        c.num_kv_heads = 4;
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(16);
        c.weight = CompressedWeight(ov::element::u4, 128, DCOffPattern::GPTQ);
        c.ffn = GELU(128, 512, ov::element::f32, CompressedWeight(ov::element::u4, 128, DCOffPattern::GPTQ));
        c.norm = RMSNorm(128);
        c.encoder_layers = 4;
        c.decoder_layers = 2;
        params.push_back({"KitchenSink_WhisperDec_GPTQ_AllOpts", c});
    }
    {
        auto c = base_bert(128, 8, 6, 512);
        c.norm = RMSNorm(128);
        c.ffn = SwiGLU(128, 512, ov::element::f32, INT4Weight{});
        c.attn_bias = FP32Weight{};
        c.qk_norm = LayerNorm(16);
        c.weight = INT4Weight{};
        c.type_vocab_size = 16;
        params.push_back({"KitchenSink_BERT_INT4_AllOpts", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.num_kv_heads = 1;
        c.attn_bias = FP16Weight{};
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{}, FP32Weight{});
        c.precision = ov::element::f16;
        c.weight = INT4GroupWeight{128};
        c.pre_norm = false;
        c.lm_head_weight = WeightFn{};
        params.push_back({"KitchenSink_MQA_GELU_Bias_INT4g_NoLM", c});
    }
    {
        auto c = base_llm(128, 8, 4, 512);
        c.use_inputs_embeds = true;
        c.internal_position_ids = true;
        c.num_kv_heads = 2;
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(16);
        c.norm = RMSNorm(128);
        c.ffn = SwiGLU(128, 512, ov::element::f32, INT8Weight{});
        c.weight = CompressedWeight(ov::element::i4, 0, DCOffPattern::SYMM_NO_ZP_F32);
        params.push_back({"KitchenSink_AllBools_SYMM_F32_SwiGLU", c});
    }
    {
        auto c = base_bert(256, 8, 6, 1024);
        c.pre_norm = true;
        c.qk_norm = LayerNorm(32);
        c.attn_bias = FP32Weight{};
        c.ffn = SwiGLU(256, 1024, ov::element::f32, INT8Weight{});
        c.weight = INT8Weight{};
        c.max_position_embeddings = 2048;
        params.push_back({"KitchenSink_BERT_PreNorm_SwiGLU_INT8", c});
    }
    {
        auto c = base_whisper_dec(128, 4, 2, 512);
        c.num_kv_heads = 1;
        c.pre_norm = false;
        c.ffn = GELU(128, 512, ov::element::f32, FP32Weight{}, FP32Weight{});
        c.attn_bias = FP16Weight{};
        c.qk_norm = LayerNorm(32);
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        params.push_back({"KitchenSink_WhisperDec_MQA_GELU_Bias", c});
    }
    {
        auto c = base_llm(4, 2, 1, 8, 4);
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(2);
        c.norm = RMSNorm(4);
        c.ffn = GELU(4, 8, ov::element::f32, FP32Weight{}, FP32Weight{});
        c.num_kv_heads = 1;
        c.use_inputs_embeds = true;
        params.push_back({"KitchenSink_Minimal_AllFeatures", c});
    }
    {
        auto c = base_llm(256, 16, 2, 1024);
        c.num_kv_heads = 4;
        c.attn_bias = FP32Weight{};
        c.qk_norm = RMSNorm(16);
        c.weight = INT4GroupWeight{128};
        c.norm = RMSNorm(256);
        c.internal_position_ids = true;
        c.use_inputs_embeds = true;
        c.ffn = SwiGLU(256, 1024, ov::element::f32, INT4GroupWeight{128});
        params.push_back({"KitchenSink_Large_INT4g_AllOpts", c});
    }
    {
        auto c = base_llm(512, 16, 4, 2048);
        c.num_kv_heads = 4;
        c.precision = ov::element::f16;
        c.weight = INT4GroupWeight{64};
        c.lm_head_weight = FP16Weight{};
        c.attn_bias = FP16Weight{};
        c.qk_norm = RMSNorm(32);
        c.norm = RMSNorm(512, ov::element::f16);
        c.ffn = GELU(512, 2048, ov::element::f16, INT4GroupWeight{64});
        c.internal_position_ids = true;
        params.push_back({"KitchenSink_XL_F16_INT4g_AllOpts", c});
    }

    // --- Deferred defaults v2: all model types (286-295) ---
    {
        ModelConfig c;
        c.hidden_size = 256;
        c.num_heads = 8;
        c.head_dim = 32;
        c.intermediate_size = 1024;
        c.num_layers = 2;
        c.precision = ov::element::f16;
        params.push_back({"DeferredV2_PrecisionOnly_F16", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 128;
        c.num_heads = 4;
        c.head_dim = 32;
        c.intermediate_size = 512;
        c.num_layers = 2;
        c.weight = INT4GroupWeight{128};
        params.push_back({"DeferredV2_INT4Group_Weight", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 2048;
        c.num_heads = 16;
        c.head_dim = 128;
        c.intermediate_size = 8192;
        c.num_layers = 1;
        c.vocab_size = 100;
        params.push_back({"DeferredV2_VeryLargeHS_2048", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 256;
        c.num_heads = 8;
        c.head_dim = 32;
        c.intermediate_size = 1024;
        c.num_layers = 2;
        c.use_conv_features = true;
        c.use_kv_cache = false;
        c.num_mel_bins = 80;
        c.max_source_positions = 128;
        params.push_back({"DeferredV2_WhisperEnc_HS256", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 256;
        c.num_heads = 8;
        c.head_dim = 32;
        c.intermediate_size = 1024;
        c.num_layers = 2;
        c.use_cross_attention = true;
        c.encoder_layers = 2;
        c.decoder_layers = 2;
        c.max_target_positions = 64;
        params.push_back({"DeferredV2_WhisperDec_HS256", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 256;
        c.num_heads = 8;
        c.head_dim = 32;
        c.intermediate_size = 1024;
        c.num_layers = 2;
        c.use_token_type_embedding = true;
        c.use_kv_cache = false;
        c.pre_norm = false;
        c.max_position_embeddings = 128;
        c.type_vocab_size = 2;
        params.push_back({"DeferredV2_BERT_HS256", c});
    }
    {
        ModelConfig c;
        c.intermediate_size = 512;
        c.num_layers = 2;
        params.push_back({"DeferredV2_DefaultHS_ChangedInter", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 128;
        c.num_heads = 8;
        c.head_dim = 16;
        c.intermediate_size = 512;
        c.num_layers = 3;
        c.num_kv_heads = 2;
        c.use_inputs_embeds = true;
        c.internal_position_ids = true;
        c.weight = INT8Weight{};
        params.push_back({"DeferredV2_AllDimsChanged_INT8", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 384;
        c.num_heads = 6;
        c.head_dim = 64;
        c.intermediate_size = 1536;
        c.num_layers = 2;
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        c.num_kv_heads = 3;
        c.attn_bias = FP32Weight{};
        params.push_back({"DeferredV2_F16_FP16wt_GQA_AttnBias", c});
    }
    {
        ModelConfig c;
        c.hidden_size = 160;
        c.num_heads = 10;
        c.head_dim = 16;
        c.intermediate_size = 640;
        c.num_layers = 2;
        c.pre_norm = false;
        c.use_kv_cache = false;
        params.push_back({"DeferredV2_PostNorm_NoKV", c});
    }

    // --- Precision mixing and epsilon edge cases (296-300) ---
    {
        auto c = base_llm(64, 4, 2, 256);
        c.norm = LayerNorm(64, ov::element::f16);
        c.ffn = SwiGLU(64, 256, ov::element::f16, FP16Weight{});
        c.precision = ov::element::f32;
        params.push_back({"PrecMix_F32body_F16norm_F16ffn", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.norm = RMSNorm(128, ov::element::f16);
        c.precision = ov::element::f32;
        c.weight = INT8Weight{};
        params.push_back({"PrecMix_F32body_F16RMS_INT8", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.norm = LayerNorm(64, ov::element::f32, 1e-12f);
        params.push_back({"PrecMix_TinyEps", c});
    }
    {
        auto c = base_llm(64, 4, 2, 256);
        c.norm = LayerNorm(64, ov::element::f32, 1e-1f);
        params.push_back({"PrecMix_LargeEps", c});
    }
    {
        auto c = base_llm(128, 8, 2, 512);
        c.precision = ov::element::f16;
        c.weight = FP16Weight{};
        c.norm = RMSNorm(128, ov::element::f16, 1e-6f);
        c.ffn = GELU(128, 512, ov::element::f16, FP16Weight{}, FP16Weight{});
        c.attn_bias = FP16Weight{};
        c.qk_norm = RMSNorm(16, ov::element::f16);
        params.push_back({"PrecMix_AllF16_WithBias", c});
    }

    return params;
}
// clang-format on

INSTANTIATE_TEST_SUITE_P(
    ModelBuilderPartitioning,
    ModelBuilderPartitioningTests,
    ::testing::ValuesIn(generate_test_configs()),
    [](const ::testing::TestParamInfo<ModelConfigTestParam>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    ModelBuilderFolding,
    ModelBuilderFoldingTests,
    ::testing::ValuesIn(generate_test_configs()),
    [](const ::testing::TestParamInfo<ModelConfigTestParam>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    ModelBuilderSanity,
    ModelBuilderSanityTests,
    ::testing::ValuesIn(generate_test_configs()),
    [](const ::testing::TestParamInfo<ModelConfigTestParam>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    ModelBuilderFoldFuncall,
    ModelBuilderFoldFuncallTests,
    ::testing::ValuesIn(generate_test_configs()),
    [](const ::testing::TestParamInfo<ModelConfigTestParam>& info) { return info.param.name; });
