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
// Qwen, GPTOSS, LtxVideo, IO-slicing/preprocess variants); the family suite
// at the bottom covers a representative graph per family, verifying it
// fuses and that RopeCache passes over it safely (no-op). Teaching RopeCache
// to precompute those families' tables is follow-up work.

#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "partitioning/patterns/pre_compute.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"
#include "transformations/utils/gen_pattern.hpp"

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

// ============================================================================
// Per-family RoPE pattern coverage
//
// One representative graph per OV RoPEFusion family (model parts ported from
// src/common/transformations/tests/common_optimizations/
// fuse_rotary_positional_embeddings.cpp). For every family we pin down two
// NPUW-relevant facts:
//  - the graph fuses to the internal RoPE op via the aggregated RoPEFusion,
//  - NPUW RopeCache runs safely on it: these families carry their cos/sin
//    as precomputed constant tables (no runtime Sin/Cos to cache), so the
//    pass must leave the graph untouched and still fusable.
// The auxiliary matchers (CosSinPreprocess, IOSlicing, Preprocess,
// ShareCosSin) are exercised through the table-gather variants below.
// ============================================================================

using ov::gen_pattern::makeConst;
using ov::gen_pattern::makeOP;

ov::OutputVector make_cos_sin_lut(size_t max_position_embeddings, size_t rotary_ndims) {
    std::vector<float> lut_sin(max_position_embeddings * rotary_ndims, 0.0f);
    std::vector<float> lut_cos(max_position_embeddings * rotary_ndims, 0.0f);

    for (size_t i = 0, k = 0; i < rotary_ndims; i += 2, k++) {
        auto xita_i = 1.0 / std::pow(10000.0, static_cast<double>(i) / rotary_ndims);
        float* psin = lut_sin.data();
        float* pcos = lut_cos.data();
        for (size_t m = 0; m < max_position_embeddings; m++, psin += rotary_ndims, pcos += rotary_ndims) {
            pcos[k] = pcos[k + rotary_ndims / 2] = static_cast<float>(std::cos(xita_i * m));
            psin[k] = psin[k + rotary_ndims / 2] = static_cast<float>(std::sin(xita_i * m));
        }
    }
    auto Cos = makeConst(ov::element::f32, ov::Shape({1, 1, max_position_embeddings, rotary_ndims}), lut_cos);
    auto Sin = makeConst(ov::element::f32, ov::Shape({1, 1, max_position_embeddings, rotary_ndims}), lut_sin);
    return {Cos, Sin};
}

// llama2 application formula with constant cos/sin tables gathered by
// position ids (exercises the CosSinPreprocess llama path).
std::shared_ptr<ov::Model> build_llama2_rope_with_table_gather() {
    const size_t batch = 2;
    const size_t seq_length = 16;
    const size_t max_position_embeddings = 2048;
    const size_t ndims = 128;

    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_length, 32, ndims});
    auto seq_len = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
    auto gather_id = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1, seq_length});

    auto gather_from_sin_cos = [&](const ov::Output<ov::Node>& const_tab) {
        auto ScatterUpdate = makeOP<ov::opset3::ScatterUpdate>({{0, 0, 0}, {2}, seq_len, {0}});
        auto slice_Slice = makeOP<ov::opset1::StridedSlice>({const_tab, {0, 0, 0}, ScatterUpdate, {1, 1, 1}},
                                                            {{"begin_mask", {1, 1, 0}},
                                                             {"end_mask", {1, 1, 0}},
                                                             {"new_axis_mask", {}},
                                                             {"shrink_axis_mask", {}},
                                                             {"ellipsis_mask", {}}});
        auto squeeze =
            makeOP<ov::opset1::Reshape>({slice_Slice, {-1, static_cast<int>(ndims)}}, {{"special_zero", false}});
        auto index_Gather = makeOP<ov::opset8::Gather>({squeeze, gather_id, {0}}, {{"batch_dims", 0}});
        return makeOP<ov::opset1::Reshape>({index_Gather, {1, 1, -1, static_cast<int>(ndims)}},
                                           {{"special_zero", false}});
    };

    auto cos_sin_cache = make_cos_sin_lut(max_position_embeddings, ndims);
    ov::OutputVector cos_sin(2);
    cos_sin[0] = gather_from_sin_cos(cos_sin_cache[0]);
    cos_sin[1] = gather_from_sin_cos(cos_sin_cache[1]);

    auto transpose_Transpose = makeOP<ov::opset1::Transpose>({input, {0, 2, 1, 3}});
    auto mul_Multiply = makeOP<ov::opset1::Multiply>({transpose_Transpose, cos_sin[0]}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice_459 =
        makeOP<ov::opset1::StridedSlice>({transpose_Transpose, {0, 0, 0, 64}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto minus_one = makeConst(ov::element::f32, ov::Shape({1, 1, 1, 1}), {-1.0f});
    auto neg_Multiply = makeOP<ov::opset1::Multiply>({slice_Slice_459, minus_one}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice =
        makeOP<ov::opset1::StridedSlice>({transpose_Transpose, {0, 0, 0, 0}, {0, 0, 0, 64}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat = makeOP<ov::opset1::Concat>({neg_Multiply, slice_Slice}, {{"axis", -1}});
    auto mul_Multiply_463 = makeOP<ov::opset1::Multiply>({cat_Concat, cos_sin[1]}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<ov::opset1::Add>({mul_Multiply, mul_Multiply_463}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::OutputVector{add_Add}, ov::ParameterVector{input, seq_len, gather_id});
}

// GPTNEOX application formula with table slicing + GatherElements preprocess
// (exercises the CosSinPreprocess gptneox path).
std::shared_ptr<ov::Model> build_gptneox_rope() {
    const int seq_length = 16;
    const int max_position_embeddings = 2048;
    const int ndims = 80;
    const int rotary_ndims = 20;
    const size_t batch_s = 2;
    const size_t seq_length_s = 16;
    const size_t num_heads_s = 32;
    const size_t ndims_s = 80;
    const size_t rotary_ndims_s = 20;
    (void)seq_length;

    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                         ov::Shape{batch_s, seq_length_s, num_heads_s, ndims_s * 3});
    auto gather_idx =
        std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1, 1, seq_length_s, rotary_ndims_s});
    auto batch_limit = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});

    auto cos_sin_lut = make_cos_sin_lut(max_position_embeddings, rotary_ndims);
    auto slice_cos = makeOP<ov::opset1::StridedSlice>({cos_sin_lut[0], {0}, batch_limit, {1}},
                                                      {{"begin_mask", {0}},
                                                       {"end_mask", {0}},
                                                       {"new_axis_mask", {}},
                                                       {"shrink_axis_mask", {}},
                                                       {"ellipsis_mask", {}}});
    auto cos_tab = makeOP<ov::opset6::GatherElements>({slice_cos, gather_idx}, {{"axis", 2}});
    auto slice_sin = makeOP<ov::opset1::StridedSlice>({cos_sin_lut[1], {0}, batch_limit, {1}},
                                                      {{"begin_mask", {0}},
                                                       {"end_mask", {0}},
                                                       {"new_axis_mask", {}},
                                                       {"shrink_axis_mask", {}},
                                                       {"ellipsis_mask", {}}});
    auto sin_tab = makeOP<ov::opset6::GatherElements>({slice_sin, gather_idx}, {{"axis", 2}});

    auto slice_input = makeOP<ov::opset1::StridedSlice>({input, {0, 0, 0, 0}, {0, 0, 0, ndims}, {1, 1, 1, 1}},
                                                        {{"begin_mask", {1, 1, 1, 0}},
                                                         {"end_mask", {1, 1, 1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
    auto permute_Transpose = makeOP<ov::opset1::Transpose>({slice_input, {0, 2, 1, 3}});
    auto slice_rotary =
        makeOP<ov::opset1::StridedSlice>({permute_Transpose, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto mul_Multiply = makeOP<ov::opset1::Multiply>({slice_rotary, cos_tab}, {{"auto_broadcast", "numpy"}});
    auto slice_hi =
        makeOP<ov::opset1::StridedSlice>({slice_rotary, {0, 0, 0, rotary_ndims / 2}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto minus_one = makeConst(ov::element::f32, ov::Shape({1, 1, 1, 1}), {-1.0f});
    auto neg_Multiply = makeOP<ov::opset1::Multiply>({slice_hi, minus_one}, {{"auto_broadcast", "numpy"}});
    auto slice_lo =
        makeOP<ov::opset1::StridedSlice>({slice_rotary, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims / 2}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat = makeOP<ov::opset1::Concat>({neg_Multiply, slice_lo}, {{"axis", -1}});
    auto mul_sin = makeOP<ov::opset1::Multiply>({cat_Concat, sin_tab}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<ov::opset1::Add>({mul_Multiply, mul_sin}, {{"auto_broadcast", "numpy"}});
    auto slice_rest =
        makeOP<ov::opset1::StridedSlice>({permute_Transpose, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat_458 = makeOP<ov::opset1::Concat>({add_Add, slice_rest}, {{"axis", -1}});

    return std::make_shared<ov::Model>(ov::OutputVector{cat_Concat_458},
                                       ov::ParameterVector{input, gather_idx, batch_limit});
}

// GPTJ interleaved rope (repeat-interleaved sin/cos, even/odd stacking).
std::shared_ptr<ov::Model> build_gptj_rope() {
    const int batch = 2;
    const int seq_len = 7;
    const int num_heads = 16;
    const int ndims = 256;
    const int rotary_ndims = 64;

    std::vector<int32_t> rpi_idx(rotary_ndims);
    for (int i = 0, index = 0; i < rotary_ndims; i += 2, index++) {
        rpi_idx[i] = index;
        rpi_idx[i + 1] = index;
    }
    auto repeat_interleave_index = makeConst(ov::element::i32, ov::Shape({static_cast<size_t>(rotary_ndims)}), rpi_idx);

    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                         ov::Shape{static_cast<size_t>(batch),
                                                                   static_cast<size_t>(seq_len),
                                                                   static_cast<size_t>(num_heads),
                                                                   static_cast<size_t>(ndims)});
    auto gather_sin_cos = std::make_shared<ov::opset1::Parameter>(
        ov::element::f32,
        ov::Shape{1, static_cast<size_t>(seq_len), static_cast<size_t>(rotary_ndims)});

    auto split = makeOP<ov::opset1::VariadicSplit>({gather_sin_cos, {-1}, {rotary_ndims / 2, -1}});
    auto sin_tab =
        makeOP<ov::opset1::Reshape>({split->output(0), {1, -1, 1, rotary_ndims / 2}}, {{"special_zero", false}});
    auto cos_tab =
        makeOP<ov::opset1::Reshape>({split->output(1), {1, -1, 1, rotary_ndims / 2}}, {{"special_zero", false}});

    auto slice_Slice_576 =
        makeOP<ov::opset1::StridedSlice>({input, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto repeat_interleave_Cos =
        makeOP<ov::opset8::Gather>({cos_tab, repeat_interleave_index, {3}}, {{"batch_dims", 0}});
    auto mul_Multiply_757 =
        makeOP<ov::opset1::Multiply>({slice_Slice_576, repeat_interleave_Cos}, {{"auto_broadcast", "numpy"}});

    auto slice_Slice_787 =
        makeOP<ov::opset1::StridedSlice>({slice_Slice_576, {0, 0, 0, 1}, {0, 0, 0, INT_MAX}, {1, 1, 1, 2}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto minus_one = makeConst(ov::element::f32, ov::Shape({1, 1, 1, 1}), {-1.0f});
    auto neg_Multiply_790 = makeOP<ov::opset1::Multiply>({slice_Slice_787, minus_one}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_61918 = makeOP<ov::opset1::Unsqueeze>({neg_Multiply_790, {-1}});
    auto slice_Slice_781 =
        makeOP<ov::opset1::StridedSlice>({slice_Slice_576, {0, 0, 0, 0}, {0, 0, 0, INT_MAX}, {1, 1, 1, 2}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto Unsqueeze_61919 = makeOP<ov::opset1::Unsqueeze>({slice_Slice_781, {-1}});
    auto stack_795 = makeOP<ov::opset1::Concat>({Unsqueeze_61918, Unsqueeze_61919}, {{"axis", -1}});
    auto ShapeOf_165368 = makeOP<ov::op::TypeRelaxed<ov::opset1::ShapeOf>>(
        {stack_795},
        {{"type_relax", true}, {"input_data_types", {}}, {"output_data_types", {ov::element::i32}}});
    auto flatten_Slice_811 = makeOP<ov::opset1::StridedSlice>({ShapeOf_165368, {0}, {3}, {1}},
                                                              {{"begin_mask", {0}},
                                                               {"end_mask", {0}},
                                                               {"new_axis_mask", {}},
                                                               {"shrink_axis_mask", {}},
                                                               {"ellipsis_mask", {}}});
    auto flatten_Concat_814 = makeOP<ov::opset1::Concat>({flatten_Slice_811, {-1}}, {{"axis", 0}});
    auto flatten_Reshape_815 = makeOP<ov::opset1::Reshape>({stack_795, flatten_Concat_814}, {{"special_zero", true}});
    auto repeat_interleave_Sin =
        makeOP<ov::opset8::Gather>({sin_tab, repeat_interleave_index, {3}}, {{"batch_dims", 0}});
    auto mul_Multiply_816 =
        makeOP<ov::opset1::Multiply>({flatten_Reshape_815, repeat_interleave_Sin}, {{"auto_broadcast", "numpy"}});
    auto add_Add_819 = makeOP<ov::opset1::Add>({mul_Multiply_757, mul_Multiply_816}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice_582 =
        makeOP<ov::opset1::StridedSlice>({input, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat_826 = makeOP<ov::opset1::Concat>({add_Add_819, slice_Slice_582}, {{"axis", -1}});
    auto permute_Transpose_828 = makeOP<ov::opset1::Transpose>({cat_Concat_826, {0, 2, 1, 3}});
    return std::make_shared<ov::Model>(ov::OutputVector{permute_Transpose_828},
                                       ov::ParameterVector{input, gather_sin_cos});
}

// ChatGLM rope (interleaved even/odd pairs with a gathered cos/sin cache).
std::shared_ptr<ov::Model> build_chatglm_rope() {
    const int batch = 2;
    const int seq_len = 7;
    const int num_heads = 32;
    const int ndims = 128;
    const int rotary_ndims = 64;
    const int max_pos_length = 2048;

    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, seq_len, 4608});
    auto cos_sin_cache = std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                                 ov::PartialShape{max_pos_length, rotary_ndims / 2, 2});
    auto position_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{batch, seq_len});

    auto cache_Gather = makeOP<ov::opset8::Gather>({cos_sin_cache, position_ids, 0}, {{"batch_dims", 0}});

    auto ListUnpack_321 = makeOP<ov::opset1::VariadicSplit>({input, -1, {4096, 256, 256}});
    auto view_Reshape =
        makeOP<ov::opset1::Reshape>({ListUnpack_321->output(0), {0, 0, num_heads, ndims}}, {{"special_zero", true}});

    auto permute_Transpose = makeOP<ov::opset1::Transpose>({view_Reshape, {0, 2, 1, 3}}, {});

    auto slice_Slice_357 =
        makeOP<ov::opset1::StridedSlice>({permute_Transpose, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});

    auto aten_view_Reshape_1 =
        makeOP<ov::opset1::Reshape>({ListUnpack_321->output(1), {0, 0, 2, ndims}}, {{"special_zero", true}});
    auto aten_transpose_1 = makeOP<ov::opset8::Transpose>({aten_view_Reshape_1, {0, 2, 1, 3}});
    auto shape_of_105249 = makeOP<ov::opset8::ShapeOf>({aten_transpose_1}, {{"output_type", "i32"}});
    auto gather_105252 = makeOP<ov::opset8::Gather>({shape_of_105249, {2}, {0}}, {{"batch_dims", 0}});
    auto scatter_update_63441 = makeOP<ov::opset8::ScatterUpdate>({{0, 0}, {1}, gather_105252, {0}});
    auto slice_Slice_369 = makeOP<ov::opset1::StridedSlice>({cache_Gather, {0, 0}, scatter_update_63441, {1, 1}},
                                                            {{"begin_mask", {1, 0}},
                                                             {"end_mask", {1, 0}},
                                                             {"new_axis_mask", {}},
                                                             {"shrink_axis_mask", {}},
                                                             {"ellipsis_mask", {}}});
    auto list_construct_concat_1 =
        makeOP<ov::opset1::Concat>({{-1}, {1}, gather_105252, {rotary_ndims / 2}, {2}}, {{"axis", 0}});

    auto reshape_Reshape_373 =
        makeOP<ov::opset1::Reshape>({slice_Slice_357, {0, 32, 0, 32, 2}}, {{"special_zero", true}});
    auto x_even = makeOP<ov::opset8::Gather>({reshape_Reshape_373, 0, -1}, {{"batch_dims", 0}});
    auto x_odd = makeOP<ov::opset8::Gather>({reshape_Reshape_373, 1, -1}, {{"batch_dims", 0}});
    auto view_Reshape_380 =
        makeOP<ov::opset1::Reshape>({slice_Slice_369, list_construct_concat_1}, {{"special_zero", false}});
    auto cos_tab = makeOP<ov::opset8::Gather>({view_Reshape_380, 0, -1}, {{"batch_dims", 0}});
    auto sin_tab = makeOP<ov::opset8::Gather>({view_Reshape_380, 1, -1}, {{"batch_dims", 0}});

    auto x_odd_sin = makeOP<ov::opset1::Multiply>({x_odd, sin_tab}, {{"auto_broadcast", "numpy"}});
    auto x_even_cos = makeOP<ov::opset1::Multiply>({x_even, cos_tab}, {{"auto_broadcast", "numpy"}});
    auto neg_x_odd_sin = makeOP<ov::opset1::Multiply>({x_odd_sin, -1.000000f}, {{"auto_broadcast", "numpy"}});
    auto sub_Subtract_389 = makeOP<ov::opset1::Add>({x_even_cos, neg_x_odd_sin}, {{"auto_broadcast", "numpy"}});

    auto x_odd_cos = makeOP<ov::opset1::Multiply>({x_odd, cos_tab}, {{"auto_broadcast", "numpy"}});
    auto x_even_sin = makeOP<ov::opset1::Multiply>({x_even, sin_tab}, {{"auto_broadcast", "numpy"}});
    auto add_Add_396 = makeOP<ov::opset1::Add>({x_odd_cos, x_even_sin}, {{"auto_broadcast", "numpy"}});

    auto Unsqueeze_62716 = makeOP<ov::opset1::Unsqueeze>({sub_Subtract_389, -1}, {});
    auto Unsqueeze_62717 = makeOP<ov::opset1::Unsqueeze>({add_Add_396, -1}, {});

    auto stack_401 = makeOP<ov::opset1::Concat>({Unsqueeze_62716, Unsqueeze_62717}, {{"axis", -1}});
    auto flatten_Reshape_421 =
        makeOP<ov::opset1::Reshape>({stack_401, {0, num_heads, 0, rotary_ndims}}, {{"special_zero", true}});
    auto slice_Slice_363 =
        makeOP<ov::opset1::StridedSlice>({permute_Transpose, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat_425 = makeOP<ov::opset1::Concat>({flatten_Reshape_421, slice_Slice_363}, {{"axis", -1}});
    return std::make_shared<ov::Model>(ov::OutputVector{cat_Concat_425},
                                       ov::ParameterVector{input, cos_sin_cache, position_ids});
}

// ChatGLM-HF rope (repeat-interleaved tables, reshape-based unsqueezes).
std::shared_ptr<ov::Model> build_chatglm_hf_rope() {
    const int seq_len = 7;
    const int num_heads = 32;
    const int ndims = 128;
    const int rotary_ndims = 64;

    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{seq_len, 1, 4096});
    auto cos =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{seq_len, 1, 1, rotary_ndims / 2});
    auto sin =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{seq_len, 1, 1, rotary_ndims / 2});

    auto transpose = makeOP<ov::opset1::Reshape>({input, {-1, num_heads, 1, ndims}}, {{"special_zero", false}});
    auto slice_1 = makeOP<ov::opset1::StridedSlice>({transpose, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                                    {{"begin_mask", {1, 1, 1, 0}},
                                                     {"end_mask", {1, 1, 1, 0}},
                                                     {"new_axis_mask", {}},
                                                     {"shrink_axis_mask", {}},
                                                     {"ellipsis_mask", {}}});

    std::vector<int32_t> rpi_idx(rotary_ndims);
    for (int i = 0, index = 0; i < rotary_ndims; i += 2, index++) {
        rpi_idx[i] = index;
        rpi_idx[i + 1] = index;
    }
    auto repeat_interleave_index = makeConst(ov::element::i32, ov::Shape({static_cast<size_t>(rotary_ndims)}), rpi_idx);
    auto repeat_interleave_cos = makeOP<ov::opset8::Gather>({cos, repeat_interleave_index, -1}, {{"batch_dims", 0}});
    auto repeat_interleave_sin = makeOP<ov::opset8::Gather>({cos, repeat_interleave_index, -1}, {{"batch_dims", 0}});

    auto multiply = makeOP<ov::opset1::Multiply>({slice_1, repeat_interleave_cos}, {{"auto_broadcast", "numpy"}});
    auto slice_2 = makeOP<ov::opset1::StridedSlice>({slice_1, {0, 0, 0, 1}, {0, 0, 0, INT_MAX}, {1, 1, 1, 2}},
                                                    {{"begin_mask", {1, 1, 1, 0}},
                                                     {"end_mask", {1, 1, 1, 0}},
                                                     {"new_axis_mask", {}},
                                                     {"shrink_axis_mask", {}},
                                                     {"ellipsis_mask", {}}});
    auto neg = makeOP<ov::opset1::Multiply>({slice_2, -1.000000f}, {{"auto_broadcast", "numpy"}});
    auto unsqueeze_1 =
        makeOP<ov::opset1::Reshape>({neg, {-1, num_heads, 1, rotary_ndims / 2, 1}}, {{"special_zero", false}});
    auto slice_3 = makeOP<ov::opset1::StridedSlice>({slice_1, {0, 0, 0, 0}, {0, 0, 0, INT_MAX}, {1, 1, 1, 2}},
                                                    {{"begin_mask", {1, 1, 1, 0}},
                                                     {"end_mask", {1, 1, 1, 0}},
                                                     {"new_axis_mask", {}},
                                                     {"shrink_axis_mask", {}},
                                                     {"ellipsis_mask", {}}});
    auto unsqueeze_2 =
        makeOP<ov::opset1::Reshape>({slice_3, {-1, num_heads, 1, rotary_ndims / 2, 1}}, {{"special_zero", false}});
    auto stack = makeOP<ov::opset1::Concat>({unsqueeze_1, unsqueeze_2}, {{"axis", -1}});
    auto flatten = makeOP<ov::opset1::Reshape>({stack, {0, num_heads, 0, rotary_ndims}}, {{"special_zero", true}});
    auto multiply_1 = makeOP<ov::opset1::Multiply>({flatten, repeat_interleave_sin}, {{"auto_broadcast", "numpy"}});
    auto add = makeOP<ov::opset1::Add>({multiply, multiply_1}, {{"auto_broadcast", "numpy"}});

    auto slice_5 =
        makeOP<ov::opset1::StridedSlice>({transpose, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto concat = makeOP<ov::opset1::Concat>({add, slice_5}, {{"axis", -1}});
    return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input, cos, sin});
}

// Qwen rope (rotary table gathered by position ids, split-based rotate-half).
std::shared_ptr<ov::Model> build_qwen_rope() {
    constexpr int head_cnt = 32;
    constexpr int head_size = 128;

    auto position_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    auto qkv =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 3 * head_cnt * head_size});

    auto qkv_proj = makeOP<ov::opset1::VariadicSplit>({qkv, 2, {head_cnt * head_size, head_cnt * head_size, -1}});
    auto view =
        makeOP<ov::opset1::Reshape>({qkv_proj->output(0), {0, 0, head_cnt, head_size}}, {{"special_zero", true}});

    auto slice = makeOP<ov::opset8::Slice>({view, {0}, {128}, {1}, {3}});

    auto rotary_emp = makeConst(ov::element::f32, ov::Shape({1, 4096, 1, 128}), {1});
    auto pos_i32 = makeOP<ov::opset1::Convert>({position_ids}, {{"destination_type", "i32"}});
    auto pos_reshaped = makeOP<ov::opset1::Reshape>({pos_i32, {-1, 1}}, {{"special_zero", false}});
    auto gathered = makeOP<ov::opset8::Gather>({rotary_emp, pos_reshaped, 1}, {{"batch_dims", 0}});
    auto gathered_reshape = makeOP<ov::opset1::Reshape>({gathered, {-1, 1, 1, 128}}, {{"special_zero", false}});

    auto mul = makeOP<ov::opset1::Multiply>({slice, gathered_reshape}, {{"auto_broadcast", "numpy"}});

    auto reshaped = makeOP<ov::opset1::Reshape>({slice, {0, 0, 32, 2, 64}}, {{"special_zero", true}});
    auto split = makeOP<ov::opset1::Split>({reshaped, -2}, {{"num_splits", 2}});
    auto neg = makeOP<ov::opset1::Multiply>({split->output(1), -1.0f}, {{"auto_broadcast", "numpy"}});
    auto squeeze0 = makeOP<ov::opset1::Reshape>({neg, {-1, 1, 32, 64}}, {{"special_zero", false}});
    auto squeeze1 = makeOP<ov::opset1::Reshape>({split->output(0), {-1, 1, 32, 64}}, {{"special_zero", false}});
    auto cat = makeOP<ov::opset1::Concat>({squeeze0, squeeze1}, {{"axis", -1}});

    auto rotary_emp2 = makeConst(ov::element::f32, ov::Shape({1, 4096, 1, 128}), {1});
    auto gathered2 = makeOP<ov::opset8::Gather>({rotary_emp2, pos_reshaped, 1}, {{"batch_dims", 0}});
    auto gathered2_reshape = makeOP<ov::opset1::Reshape>({gathered2, {-1, 1, 1, 128}}, {{"special_zero", false}});
    auto mul2 = makeOP<ov::opset1::Multiply>({cat, gathered2_reshape}, {{"auto_broadcast", "numpy"}});

    auto add = makeOP<ov::opset1::Add>({mul, mul2}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{position_ids, qkv});
}

// GPT-OSS rope (half-split formula without rotate-half concat on the input).
std::shared_ptr<ov::Model> build_gptoss_rope() {
    const int batch = 2;
    const int num_head = 32;
    const int seq_len = 16;
    const int ndims = 128;
    const int half_ndims = ndims / 2;

    auto input =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, num_head, seq_len, ndims});
    auto t_cos =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, 1, seq_len, half_ndims});
    auto t_sin =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, 1, seq_len, half_ndims});

    auto split_lengths = makeConst(ov::element::i64, ov::Shape({2}), std::vector<int64_t>{half_ndims, half_ndims});
    auto axis_const = makeConst(ov::element::i64, ov::Shape({}), std::vector<int64_t>{-1});
    auto vsplit = makeOP<ov::opset1::VariadicSplit>({input, axis_const, split_lengths});

    auto first_half_mul_cos = makeOP<ov::opset1::Multiply>({vsplit->output(0), t_cos}, {{"auto_broadcast", "numpy"}});
    auto second_half_mul_sin = makeOP<ov::opset1::Multiply>({vsplit->output(1), t_sin}, {{"auto_broadcast", "numpy"}});
    auto neg = makeOP<ov::opset1::Multiply>({second_half_mul_sin, -1.0f}, {{"auto_broadcast", "numpy"}});
    auto first_ = makeOP<ov::opset1::Add>({first_half_mul_cos, neg}, {{"auto_broadcast", "numpy"}});

    auto second_half_mul_cos = makeOP<ov::opset1::Multiply>({vsplit->output(1), t_cos}, {{"auto_broadcast", "numpy"}});
    auto first_half_mul_sin = makeOP<ov::opset1::Multiply>({vsplit->output(0), t_sin}, {{"auto_broadcast", "numpy"}});
    auto second_ = makeOP<ov::opset1::Add>({second_half_mul_cos, first_half_mul_sin}, {{"auto_broadcast", "numpy"}});

    auto result = makeOP<ov::opset1::Concat>({first_, second_}, {{"axis", -1}});

    return std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input, t_cos, t_sin});
}

// Flux vision rope (pairwise reshape/split rotate, multiplied tables).
std::shared_ptr<ov::Model> build_flux_rope() {
    const int batch = 2;
    const int num_heads = 32;
    const int ndims = 128;

    auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, num_heads, -1, ndims});
    auto t_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
    auto t_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});

    auto x1_shape = makeConst(ov::element::i64, ov::Shape({5}), std::vector<int64_t>{0, num_heads, 0, -1, 2});
    auto x1 = std::make_shared<ov::op::v1::Reshape>(x, x1_shape, true);

    auto split_axis = makeConst(ov::element::i64, ov::Shape(), std::vector<int64_t>{-1});
    auto split = std::make_shared<ov::op::v1::Split>(x1, split_axis, 2);

    auto minus_one = makeConst(ov::element::f32, ov::Shape({}), {-1.0f});
    auto x1_1_neg = std::make_shared<ov::op::v1::Multiply>(split->output(1), minus_one);

    auto x2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_1_neg->output(0), split->output(0)}, -1);

    auto x3_shape = makeConst(ov::element::i64, ov::Shape({4}), std::vector<int64_t>{0, num_heads, 0, ndims});
    auto x3 = std::make_shared<ov::op::v1::Reshape>(x2, x3_shape, true);

    auto y1 = std::make_shared<ov::op::v1::Multiply>(x, t_cos);
    auto y2 = std::make_shared<ov::op::v1::Multiply>(x3, t_sin);
    auto y = std::make_shared<ov::op::v1::Add>(y1, y2);

    return std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x, t_cos, t_sin});
}

// LTX-Video spatial-temporal rope (interleaved complex format).
std::shared_ptr<ov::Model> build_ltxvideo_rope() {
    const int batch = 1;
    const int seq_len = 2520;
    const int rotary_ndims = 2048;
    const int half_rotary_ndims = rotary_ndims / 2;

    auto input =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, seq_len, rotary_ndims});
    auto cos_freqs =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, seq_len, rotary_ndims});
    auto sin_freqs =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, seq_len, rotary_ndims});

    auto reshape_shape =
        makeConst(ov::element::i64, ov::Shape({4}), std::vector<int64_t>{batch, seq_len, half_rotary_ndims, 2});
    auto x_reshape = makeOP<ov::opset1::Reshape>({input, reshape_shape}, {{"special_zero", false}});

    auto split = makeOP<ov::opset1::Split>({x_reshape, -1}, {{"num_splits", 2}});

    auto neg_imag_mul = makeOP<ov::opset1::Multiply>({split->output(1), -1.0f}, {{"auto_broadcast", "numpy"}});

    auto squeeze_imag = makeOP<ov::opset1::Squeeze>({neg_imag_mul, -1});
    auto neg_imag_unsqueeze = makeOP<ov::opset1::Unsqueeze>({squeeze_imag, -1});

    auto x_rotated_concat = makeOP<ov::opset1::Concat>({neg_imag_unsqueeze, split->output(0)}, {{"axis", -1}});

    auto reshape_back_shape =
        makeConst(ov::element::i64, ov::Shape({3}), std::vector<int64_t>{batch, seq_len, rotary_ndims});
    auto x_rotated = makeOP<ov::opset1::Reshape>({x_rotated_concat, reshape_back_shape}, {{"special_zero", false}});

    auto real_mul_cos = makeOP<ov::opset1::Multiply>({input, cos_freqs}, {{"auto_broadcast", "numpy"}});
    auto imag_mul_sin = makeOP<ov::opset1::Multiply>({x_rotated, sin_freqs}, {{"auto_broadcast", "numpy"}});
    auto result = makeOP<ov::opset1::Add>({real_mul_cos, imag_mul_sin}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input, cos_freqs, sin_freqs});
}

struct RopeFamilyCase {
    std::string name;
    bool support_2d_rope;
    std::function<std::shared_ptr<ov::Model>()> build;
};

class RopeFamilyTest : public ::testing::TestWithParam<RopeFamilyCase> {};

TEST_P(RopeFamilyTest, FusesToInternalRope) {
    const auto& family = GetParam();
    auto model = family.build();
    EXPECT_EQ(count_internal_rope_nodes(model), 0);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::RoPEFusion>(family.support_2d_rope);
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    EXPECT_GE(count_internal_rope_nodes(model), 1)
        << "RoPEFusion must recognize the " << family.name << " rope pattern";
}

TEST_P(RopeFamilyTest, RopeCacheIsNoOpAndKeepsGraphFusable) {
    const auto& family = GetParam();
    auto model = family.build();

    // These families carry cos/sin as precomputed constant tables -- there is
    // no runtime Sin/Cos subgraph for RopeCache to fold into a LUT. The pass
    // must walk past them without touching the graph...
    EXPECT_EQ(count_sin_cos_nodes(model), 0);
    const size_t ops_before = model->get_ops().size();
    run_rope_cache(model);
    EXPECT_EQ(model->get_ops().size(), ops_before) << "RopeCache must not modify the " << family.name << " rope graph";

    // ...and the graph must remain fusable afterwards.
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::RoPEFusion>(family.support_2d_rope);
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    EXPECT_GE(count_internal_rope_nodes(model), 1)
        << "RoPEFusion must still recognize the " << family.name << " rope pattern after RopeCache";
}

INSTANTIATE_TEST_SUITE_P(
    NPUW_RopeFamilies,
    RopeFamilyTest,
    ::testing::Values(RopeFamilyCase{"Llama2TableGather", false, build_llama2_rope_with_table_gather},
                      RopeFamilyCase{"GPTNEOX", false, build_gptneox_rope},
                      RopeFamilyCase{"GPTJ", false, build_gptj_rope},
                      RopeFamilyCase{"ChatGLM", true, build_chatglm_rope},
                      RopeFamilyCase{"ChatGLMHF", true, build_chatglm_hf_rope},
                      RopeFamilyCase{"Qwen", false, build_qwen_rope},
                      RopeFamilyCase{"GPTOSS", false, build_gptoss_rope},
                      RopeFamilyCase{"Flux", true, build_flux_rope},
                      RopeFamilyCase{"LtxVideo", false, build_ltxvideo_rope}),
    [](const ::testing::TestParamInfo<RopeFamilyCase>& info) {
        return info.param.name;
    });

}  // namespace
