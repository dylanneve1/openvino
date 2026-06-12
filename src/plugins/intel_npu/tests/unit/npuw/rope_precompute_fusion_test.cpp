// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RoPE precompute (NPUW RopeCache) vs OpenVINO RoPEFusion interplay tests.
//
// The NPUW RopeCache pass replaces the sin/cos precompute subgraph with a
// LUT + Gather chain. These tests follow the recipe from the PR #35240 review:
//  1. Create a rope model (llama2-like sin/cos precompute + rotate-half).
//  2. Run the aggregated ov::pass::RoPEFusion -- the internal RoPE op must
//     appear, proving the synthetic graph is a real rope pattern.
//  3. Run the NPUW RopeCache pass on a cloned model -- no Cos/Sin nodes may
//     survive, the LUT gather must take their place.
//  4. Run the dedicated rope fusion passes (GPTNEOX + CosSinPreprocess
//     expectations) on the cached model -- the internal RoPE op must still
//     appear, i.e. the RopeCache output stays fusable.
//
// Note: NPUW RopeCache currently matches the llama2 sin/cos precompute and
// the Phi LongRope variants (see partitioning/patterns/pre_compute.hpp).
// The OV RoPEFusion family is much wider (GPTNEOX, Flux, GPTJ, ChatGLM(HF),
// Qwen, GPTOSS, LtxVideo, IO-slicing/preprocess variants) -- extending the
// cache coverage and these tests to those patterns is tracked follow-up work.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "partitioning/patterns/pre_compute.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

namespace {

constexpr int64_t kBatch = 1;
constexpr int64_t kHeads = 1;
constexpr int64_t kSeqLen = 16;
constexpr int64_t kRotaryNdims = 128;
constexpr int64_t kHalfNdims = kRotaryNdims / 2;
constexpr uint32_t kMaxPromptLen = 2048;

// Llama2-like rope graph: inv_freq broadcast driven by ShapeOf(x), position
// ids matmul, Cos/Sin tables and the rotate-half application formula. The
// precompute part matches NPUW's RopePatternLLama2, the application part
// matches ov::pass::RoPEFusionGPTNEOX.
std::shared_ptr<ov::Model> build_llama2_like_rope_graph() {
    using namespace ov;

    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{kBatch, kHeads, kSeqLen, kRotaryNdims});
    x->set_friendly_name("x");
    auto position_ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{kBatch, kHeads, kSeqLen});
    position_ids->set_friendly_name("position_ids");

    // inv_freq constant: [batch, heads, half_ndims, 1]
    auto inv_freq_data = std::vector<float>(kHalfNdims, 1.0f);
    auto inv_freq = op::v0::Constant::create(element::f32, Shape{kBatch, kHeads, kHalfNdims, 1}, inv_freq_data);

    // ShapeOf -> Gather -> Concat -> Broadcast chain (RopePatternLLama2 entry)
    auto shape_of = std::make_shared<op::v3::ShapeOf>(x, element::i64);
    auto gather_idx = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
    auto gather_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto gather = std::make_shared<op::v8::Gather>(shape_of, gather_idx, gather_axis);

    auto shape_half = op::v0::Constant::create(element::i64, Shape{1}, {kHalfNdims});
    auto shape_seq = op::v0::Constant::create(element::i64, Shape{1}, {kSeqLen});
    auto concat_1 = std::make_shared<op::v0::Concat>(OutputVector{gather, shape_half, shape_seq}, 0);

    auto broadcast = std::make_shared<op::v3::Broadcast>(inv_freq, concat_1);

    // position_ids -> Unsqueeze -> Convert(f32) -> MatMul(inv_freq broadcast)
    auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto pos_unsqueeze = std::make_shared<op::v0::Unsqueeze>(position_ids, unsqueeze_axis);
    auto pos_convert = std::make_shared<op::v0::Convert>(pos_unsqueeze, element::f32);

    auto matmul = std::make_shared<op::v0::MatMul>(broadcast, pos_convert, false, false);
    auto transpose_axes = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});
    auto transpose = std::make_shared<op::v1::Transpose>(matmul, transpose_axes);

    auto concat_2 = std::make_shared<op::v0::Concat>(OutputVector{transpose, transpose}, -1);
    auto cos = std::make_shared<op::v0::Cos>(concat_2);
    auto sin = std::make_shared<op::v0::Sin>(concat_2);

    // rotate_half(x) = concat(-x2, x1)
    auto split_lengths = op::v0::Constant::create(element::i64, Shape{2}, {kHalfNdims, kHalfNdims});
    auto split_axis = op::v0::Constant::create(element::i64, Shape{}, {3});
    auto split = std::make_shared<op::v1::VariadicSplit>(x, split_axis, split_lengths);

    auto minus_one = op::v0::Constant::create(element::f32, Shape{1}, {-1.0f});
    auto neg_half = std::make_shared<op::v1::Multiply>(split->output(1), minus_one);
    auto rotate_half = std::make_shared<op::v0::Concat>(OutputVector{neg_half, split->output(0)}, -1);

    // y = x * cos + rotate_half(x) * sin
    auto mul_cos = std::make_shared<op::v1::Multiply>(x, cos);
    auto mul_sin = std::make_shared<op::v1::Multiply>(rotate_half, sin);
    auto rope_formula = std::make_shared<op::v1::Add>(mul_cos, mul_sin);

    auto result = std::make_shared<op::v0::Result>(rope_formula);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{x, position_ids});
}

size_t count_nodes_of_type(const std::shared_ptr<ov::Model>& model, const ov::DiscreteTypeInfo& type_info) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (node->get_type_info() == type_info) {
            count++;
        }
    }
    return count;
}

size_t count_internal_rope_nodes(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (std::dynamic_pointer_cast<ov::op::internal::RoPE>(node)) {
            count++;
        }
    }
    return count;
}

size_t count_sin_cos_nodes(const std::shared_ptr<ov::Model>& model) {
    return count_nodes_of_type(model, ov::op::v0::Sin::get_type_info_static()) +
           count_nodes_of_type(model, ov::op::v0::Cos::get_type_info_static());
}

void run_rope_cache(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::npuw::patterns::pre_compute::RopeCache>(kMaxPromptLen, "");
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

// Step 2 of the recipe: the synthetic graph must be recognized by the
// aggregated OV RoPEFusion pass on its own.
TEST(NPUW_RopePrecomputeAndFusion, AggregatedRoPEFusionCreatesInternalRope) {
    auto model = build_llama2_like_rope_graph();
    EXPECT_EQ(count_internal_rope_nodes(model), 0);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::RoPEFusion>();
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    EXPECT_GE(count_internal_rope_nodes(model), 1) << "Aggregated RoPEFusion must recognize the llama2-like rope graph";
}

// Step 3 of the recipe: RopeCache must take the sin/cos precompute out and
// gather from the precomputed LUT instead.
TEST(NPUW_RopePrecomputeAndFusion, RopeCacheRemovesSinCosNodes) {
    auto model = build_llama2_like_rope_graph();
    EXPECT_GT(count_sin_cos_nodes(model), 0);

    run_rope_cache(model);

    EXPECT_EQ(count_sin_cos_nodes(model), 0) << "RopeCache must eliminate the sin/cos precompute subgraph";
    EXPECT_GE(count_nodes_of_type(model, ov::op::v8::Gather::get_type_info_static()), 2)
        << "RopeCache must gather both cos and sin values from the precomputed LUT";
    EXPECT_EQ(count_internal_rope_nodes(model), 0) << "RopeCache itself must not create RoPE ops";
}

// Step 4 of the recipe: the RopeCache output must still be fusable by the
// dedicated rope fusion passes (rotate-half matcher + cos/sin preprocess
// expectations).
TEST(NPUW_RopePrecomputeAndFusion, RopeCachedGraphStillFusesToInternalRope) {
    auto model = build_llama2_like_rope_graph();
    run_rope_cache(model);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::RoPEFusionGPTNEOX>(4);
    // NOTE: the LUT chain produced by RopeCache (f16 table -> Gather over
    // i64 position ids -> Convert -> Squeeze) is not absorbed into the RoPE
    // op by RoPEFusionCosSinPreprocess yet -- its patterns expect f32 tables
    // sliced by i32 positions. Registered here to pin the expected pipeline;
    // extending either side to close that gap is follow-up work.
    manager.register_pass<ov::pass::RoPEFusionCosSinPreprocess>();
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    EXPECT_GE(count_internal_rope_nodes(model), 1)
        << "Rope application formula must stay fusable after RopeCache rewrote the precompute";
}

// The combined pipeline from the review sketch: RopeCache first, then the
// aggregated RoPEFusion on the same model.
TEST(NPUW_RopePrecomputeAndFusion, Llama2LikePatternCreatesInternalRope) {
    auto model = build_llama2_like_rope_graph();

    EXPECT_EQ(count_internal_rope_nodes(model), 0);

    ov::pass::Manager manager;
    manager.register_pass<ov::npuw::patterns::pre_compute::RopeCache>(kMaxPromptLen, "");
    manager.register_pass<ov::pass::RoPEFusion>();

    manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_sin_cos_nodes(model), 0);
    EXPECT_GE(count_internal_rope_nodes(model), 1);
}

}  // namespace
