// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for run_turboquant_kv_cache_passes().
//
// Each test builds a synthetic SDPA model (matching the structure used by the
// existing kv_cache_compression_pass_test.cpp), retypes past/present KV
// tensors to u8 via PPP (the production path does this inside
// cvt_kvcache_to_low_precision), then runs the TurboQuant pass and verifies
// the structural outcome.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <regex>
#include <unordered_map>

#include "llm_pass_test_fixture.hpp"
#include "npuw_transformations/turboquant_kv_cache.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/openvino.hpp"

namespace {

using namespace ov;

// ----------------------------------------------------------------------------
// Minimal synthetic SDPA model. Mirrors build_sdpa_model() in
// kv_cache_compression_pass_test.cpp so find_all_sdpa_pattern_nodes() picks
// it up.
// ----------------------------------------------------------------------------
std::shared_ptr<Model> build_sdpa_model(size_t num_sdpa, size_t head_dim = 64) {
    const Shape past_shape = {1, 4, 8, head_dim};
    const Shape new_token_shape = {1, 4, 1, head_dim};
    const Shape mask_shape = {1, 1, 1, 9};

    ParameterVector params;
    ResultVector results;

    for (size_t n = 0; n < num_sdpa; ++n) {
        const std::string idx = std::to_string(n);

        auto make_param = [&](const std::string& name, const Shape& shape) {
            auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        auto past_key = make_param("past_key_values." + idx + ".key", past_shape);
        auto past_val = make_param("past_key_values." + idx + ".value", past_shape);
        auto query = make_param("query." + idx, new_token_shape);
        auto new_key = make_param("new_key." + idx, new_token_shape);
        auto new_val = make_param("new_value." + idx, new_token_shape);
        auto mask = make_param("mask." + idx, mask_shape);

        auto concat_key = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
        auto concat_val = std::make_shared<op::v0::Concat>(OutputVector{past_val, new_val}, 2);

        auto qk = std::make_shared<op::v0::MatMul>(query, concat_key, false, true);
        auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
        auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
        auto attn_out = std::make_shared<op::v0::MatMul>(softmax->output(0), concat_val->output(0));

        auto make_result = [&](Output<Node> out, const std::string& name) {
            auto r = std::make_shared<op::v0::Result>(out);
            r->set_friendly_name(name);
            r->output(0).get_tensor().set_names({name});
            results.push_back(r);
        };

        make_result(concat_key->output(0), "present." + idx + ".key");
        make_result(concat_val->output(0), "present." + idx + ".value");
        make_result(attn_out->output(0), "attn_out." + idx);
    }

    auto model = std::make_shared<Model>(results, params, "sdpa_test_" + std::to_string(num_sdpa));
    model->validate_nodes_and_infer_types();
    return model;
}

// Retype past_key_values.* and present.* to u8 via PPP, matching what
// cvt_kvcache_to_low_precision does in production for TurboQuant.
std::shared_ptr<Model> apply_u8_kv_io(const std::shared_ptr<Model>& model) {
    ov::preprocess::PrePostProcessor ppp(model);
    const std::regex past_key_re(R"(past_key_values\.\d+\.key)");
    const std::regex past_val_re(R"(past_key_values\.\d+\.value)");
    const std::regex present_key_re(R"(present\.\d+\.key)");
    const std::regex present_val_re(R"(present\.\d+\.value)");

    for (const auto& tensor : model->inputs()) {
        const auto& name = tensor.get_any_name();
        if (std::regex_match(name, past_key_re) || std::regex_match(name, past_val_re)) {
            ppp.input(name).tensor().set_element_type(element::u8);
        }
    }
    for (const auto& tensor : model->outputs()) {
        const auto& name = tensor.get_any_name();
        if (std::regex_match(name, present_key_re) || std::regex_match(name, present_val_re)) {
            ppp.output(name).tensor().set_element_type(element::u8);
        }
    }
    return ppp.build();
}

// ----------------------------------------------------------------------------
// Test parameter struct.
// ----------------------------------------------------------------------------
struct TQParams {
    std::string name;
    size_t num_sdpa;
    size_t head_dim;
    uint8_t key_bits;
    uint8_t value_bits;
    bool use_qjl;
};

std::ostream& operator<<(std::ostream& os, const TQParams& p) {
    return os << p.name;
}

class TurboQuantKVCachePassTest : public ::testing::TestWithParam<TQParams> {
protected:
    ov::npuw::TurboQuantParams make_params(const TQParams& p) const {
        ov::npuw::TurboQuantParams tq;
        tq.key.bits = p.key_bits;
        tq.key.use_qjl = p.use_qjl;
        tq.value.bits = p.value_bits;
        tq.value.use_qjl = false;  // paper applies QJL only to keys
        return tq;
    }
};

// ----------------------------------------------------------------------------
// Helpers for assertions.
// ----------------------------------------------------------------------------
template <typename T>
size_t count_nodes(const std::shared_ptr<Model>& model) {
    size_t n = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<T>(op)) {
            ++n;
        }
    }
    return n;
}

bool has_result_with_name(const std::shared_ptr<Model>& model, const std::string& name) {
    for (const auto& r : model->get_results()) {
        if (r->get_friendly_name() == name) {
            return true;
        }
    }
    return false;
}

}  // namespace

// ============================================================================
// Tests
// ============================================================================

// Pass must run cleanly and leave a model that passes type inference.
TEST_P(TurboQuantKVCachePassTest, ModelValidAfterPass) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa, p.head_dim);
    model = apply_u8_kv_io(model);

    ASSERT_NO_THROW(ov::npuw::run_turboquant_kv_cache_passes(model, make_params(p)));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

// Hadamard MatMul constants must be inserted on K and V paths (provided
// head_dim is a power of two — every parametrized case picks one).
TEST_P(TurboQuantKVCachePassTest, HadamardMatMulsInserted) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa, p.head_dim);
    model = apply_u8_kv_io(model);
    const size_t matmul_before = count_nodes<op::v0::MatMul>(model);

    ASSERT_NO_THROW(ov::npuw::run_turboquant_kv_cache_passes(model, make_params(p)));
    const size_t matmul_after = count_nodes<op::v0::MatMul>(model);

    // Each SDPA branch (key + value) gets:
    //   - 1 Hadamard MatMul on the past-dequant inverse-rotation
    //   - 1 Hadamard MatMul on the present-side rotation
    // Plus QJL adds 1 more MatMul per key branch when use_qjl is true.
    const size_t expected_added =
        p.num_sdpa * (2 /*K&V*/ * 2 /*present+past*/ + (p.use_qjl ? 1 : 0));
    EXPECT_EQ(matmul_after, matmul_before + expected_added)
        << "Expected exactly " << expected_added << " new MatMul nodes, got "
        << (matmul_after - matmul_before);
}

// Lloyd-Max Gather (codebook lookup) must appear once per K/V branch on the
// past-dequant side.
TEST_P(TurboQuantKVCachePassTest, LloydMaxGatherInserted) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa, p.head_dim);
    model = apply_u8_kv_io(model);

    ASSERT_NO_THROW(ov::npuw::run_turboquant_kv_cache_passes(model, make_params(p)));

    const size_t gathers = count_nodes<op::v8::Gather>(model);
    EXPECT_EQ(gathers, p.num_sdpa * 2u)
        << "Expected " << (p.num_sdpa * 2u) << " Gather nodes (one per K/V branch past dequant), got "
        << gathers;
}

// Each SDPA block must add scale Result + scale Parameter for both K and V.
TEST_P(TurboQuantKVCachePassTest, ScaleResultsAndParamsAdded) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa, p.head_dim);
    model = apply_u8_kv_io(model);

    const size_t params_before = model->get_parameters().size();
    const size_t results_before = model->get_results().size();

    ASSERT_NO_THROW(ov::npuw::run_turboquant_kv_cache_passes(model, make_params(p)));

    // Per SDPA: 2 new params (key/value past scale), 2 + (1 if use_qjl) new
    // results (key/value present scale, plus optional QJL).
    const size_t expected_new_params = p.num_sdpa * 2u;
    const size_t expected_new_results = p.num_sdpa * (2u + (p.use_qjl ? 1u : 0u));

    EXPECT_EQ(model->get_parameters().size(), params_before + expected_new_params);
    EXPECT_EQ(model->get_results().size(), results_before + expected_new_results);

    for (size_t n = 0; n < p.num_sdpa; ++n) {
        const std::string idx = std::to_string(n);
        EXPECT_TRUE(
            has_result_with_name(model, "DynamicQuantize/" + idx + "/present/key/scale"));
        EXPECT_TRUE(
            has_result_with_name(model, "DynamicQuantize/" + idx + "/present/value/scale"));
        if (p.use_qjl) {
            EXPECT_TRUE(
                has_result_with_name(model, "DynamicQuantize/" + idx + "/present/key/qjl"));
        }
    }
}

// QJL Sign nodes appear only when use_qjl is true.
TEST_P(TurboQuantKVCachePassTest, QJLSignBranchPresenceMatchesConfig) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa, p.head_dim);
    model = apply_u8_kv_io(model);

    ASSERT_NO_THROW(ov::npuw::run_turboquant_kv_cache_passes(model, make_params(p)));

    const size_t signs = count_nodes<op::v0::Sign>(model);
    if (p.use_qjl) {
        EXPECT_EQ(signs, p.num_sdpa) << "Expected one Sign per SDPA block when QJL is on";
    } else {
        EXPECT_EQ(signs, 0u) << "QJL is off; no Sign nodes should be present";
    }
}

// ============================================================================
// Hadamard self-check (algorithmic): H * H^T should equal n * I within f32
// rounding error. We can't directly call the static helper from the test, so
// we replicate the math.
// ============================================================================
namespace {
std::vector<float> sylvester_hadamard(size_t n) {
    std::vector<float> H(n * n, 0.0f);
    H[0] = 1.0f;
    for (size_t step = 1; step < n; step *= 2) {
        for (size_t i = 0; i < step; ++i) {
            for (size_t j = 0; j < step; ++j) {
                const float v = H[i * n + j];
                H[i * n + (j + step)] = v;
                H[(i + step) * n + j] = v;
                H[(i + step) * n + (j + step)] = -v;
            }
        }
    }
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(n));
    for (auto& x : H) x *= inv_sqrt_n;
    return H;
}
}  // namespace

TEST(TurboQuantHadamardSelfCheck, OrthogonalitySylvester64) {
    constexpr size_t n = 64;
    const auto H = sylvester_hadamard(n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float dot = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                dot += H[i * n + k] * H[j * n + k];
            }
            const float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(dot, expected, 1e-5f) << "H·H^T at (" << i << "," << j << ") off";
        }
    }
}

// ============================================================================
// Non-power-of-two head_dim: pass must still complete without throwing,
// falling back to identity rotation.
// ============================================================================
TEST(TurboQuantNonPowerOfTwo, FallsBackGracefully) {
    constexpr size_t head_dim = 80;  // not a power of two
    auto model = build_sdpa_model(1, head_dim);
    model = apply_u8_kv_io(model);

    ov::npuw::TurboQuantParams tq;
    tq.key.bits = 3;
    tq.value.bits = 2;
    ASSERT_NO_THROW(ov::npuw::run_turboquant_kv_cache_passes(model, tq));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

// ============================================================================
// Instantiation
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    TurboQuant,
    TurboQuantKVCachePassTest,
    ::testing::Values(
        TQParams{"single_sdpa_hd64_3_2_qjl", 1u, 64u, 3u, 2u, true},
        TQParams{"single_sdpa_hd128_3_2_qjl", 1u, 128u, 3u, 2u, true},
        TQParams{"multi_sdpa_hd64_3_2_qjl", 3u, 64u, 3u, 2u, true},
        TQParams{"single_sdpa_hd64_3_2_no_qjl", 1u, 64u, 3u, 2u, false},
        TQParams{"single_sdpa_hd64_4_4_no_qjl", 1u, 64u, 4u, 4u, false}),
    [](const ::testing::TestParamInfo<TQParams>& info) {
        return info.param.name;
    });

// ============================================================================
// End-to-end test: build a real LLM via ModelBuilder, compile with
// NPUW_LLM_TURBOQUANT=YES, and confirm the resulting generate sub-model has
// Hadamard MatMuls + Lloyd-Max Gather + scale Result outputs on the K/V paths.
// ============================================================================
namespace {

// Walk a model's results and return the count whose name contains needle.
size_t count_outputs_with_name(const std::shared_ptr<ov::Model>& model, std::string_view needle) {
    size_t n = 0;
    for (const auto& out : model->outputs()) {
        for (const auto& name : out.get_names()) {
            if (name.find(needle) != std::string::npos) {
                ++n;
                break;
            }
        }
    }
    return n;
}

// Walk a model and check whether any Constant feeding a MatMul has the given
// shape — the marker that a Hadamard MatMul has been inserted.
bool model_has_square_matmul_constant(const std::shared_ptr<ov::Model>& model, size_t n) {
    for (const auto& op : model->get_ordered_ops()) {
        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op);
        if (!matmul) continue;
        for (size_t i = 0; i < matmul->get_input_size(); ++i) {
            auto src = matmul->input_value(i).get_node_shared_ptr();
            auto cst = ov::as_type_ptr<ov::op::v0::Constant>(src);
            if (!cst) continue;
            const auto shape = cst->get_shape();
            if (shape.size() == 2 && shape[0] == n && shape[1] == n) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

class TurboQuantRealLLMTest : public ov::test::npuw::LLMPassTestFixture {};

TEST_F(TurboQuantRealLLMTest, GenerateSubModelHasTurboQuantRewrite) {
    using ov::test::npuw::RecordingFactory;
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ov::AnyMap props = {
        {"NPUW_LLM_TURBOQUANT", "YES"},
        {"NPUW_LLM_TURBOQUANT_KEY_BITS", "3"},
        {"NPUW_LLM_TURBOQUANT_VALUE_BITS", "2"},
    };
    ASSERT_NO_THROW(compiled = create_compiled_model(props, recorder));
    ASSERT_NE(compiled, nullptr);

    // The generate sub-model is the one whose name contains "_kv" (kv-cache
    // present); see require_sub_model_containing usage in
    // convert_kvcache_to_precision_test.cpp.
    const auto& generate = require_sub_model_containing(recorder, "_kv");
    ASSERT_NE(generate.model, nullptr);

    // Test config sets head_dim = 16 — that's a power of two, so Hadamard
    // should be inserted.
    constexpr size_t expected_head_dim = 16;
    EXPECT_TRUE(model_has_square_matmul_constant(generate.model, expected_head_dim))
        << "Expected at least one MatMul with a [" << expected_head_dim << "x" << expected_head_dim
        << "] Hadamard constant in the rewritten generate sub-model";

    // Lloyd-Max Gather should be present on K/V branches. With 2 layers in
    // make_test_model_config and one Gather per K and V branch on the past
    // dequant side, expect at least 2 Gather ops total.
    EXPECT_GE(count_ops<ov::op::v8::Gather>(generate.model), 2u);

    // Per K/V branch, expect a present scale output. With 2 layers x 2
    // branches = 4 scale outputs minimum, and (key has QJL) 2 QJL outputs.
    EXPECT_GE(count_outputs_with_name(generate.model, "/present/key/scale"), 2u);
    EXPECT_GE(count_outputs_with_name(generate.model, "/present/value/scale"), 2u);
    EXPECT_GE(count_outputs_with_name(generate.model, "/present/key/qjl"), 2u);
}

TEST_F(TurboQuantRealLLMTest, FlagOffMeansNoTurboQuantRewrite) {
    using ov::test::npuw::RecordingFactory;
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    // Default: TurboQuant is off.
    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");
    ASSERT_NE(generate.model, nullptr);

    EXPECT_EQ(count_outputs_with_name(generate.model, "/present/key/qjl"), 0u);
    // Hadamard 16x16 constants should not be present.
    EXPECT_FALSE(model_has_square_matmul_constant(generate.model, 16));
}
