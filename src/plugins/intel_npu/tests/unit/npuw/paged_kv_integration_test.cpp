// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Integration test that USES the two paged-KV building blocks together for the
// first time on a real inference:
//
//   * SplitKVCacheIntoBlocks  (graph side) — rewrites each contiguous past-KV
//     Parameter [1, H, past, D] into past/block_size block Parameters
//     [1, H, block_size, D] re-concatenated in order.
//   * KVCacheBlockManager     (memory side) — a pool of [1, H, block_size, D]
//     block tensors that back those block Parameters.
//
// The test compiles a small stateless LLM on CPU two ways — with a single
// contiguous KV input, and with the KV split into manager-backed blocks — feeds
// the SAME key/value data through both, and asserts the logits are identical.
// This proves the two halves compose into a working paged-KV inference (the
// foundation a future continuous-batching path would build on), without any
// scheduler or pipeline wiring.

#include <gtest/gtest.h>

#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "kv_cache_block_manager.hpp"
#include "llm_test_helpers.hpp"
#include "npuw_transformations/split_kvcache_into_blocks.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

using namespace ov::npuw;

namespace {

constexpr std::size_t kSeq = 4;        // tokens processed this step
constexpr std::size_t kPast = 4;       // past-KV length (divisible by block_size)
constexpr std::size_t kBlock = 2;      // KV block size  -> kPast/kBlock = 2 blocks
constexpr std::size_t kKVTotal = kSeq + kPast;

ov::Core& core() {
    static ov::Core c;
    return c;
}

// Build a static, stateless, batch-1 LLM with past-KV length kPast.
std::shared_ptr<ov::Model> build_static_stateless_llm() {
    auto model = ov::test::npuw::build_llm_test_model();
    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    std::map<std::string, ov::PartialShape> shapes;
    for (const auto& in : model->inputs()) {
        const auto& n = in.get_any_name();
        const auto& ps = in.get_partial_shape();
        if (n.find("input_ids") != std::string::npos || n.find("token_type_ids") != std::string::npos) {
            shapes[n] = ov::PartialShape{1, kSeq};
        } else if (n.find("attention_mask") != std::string::npos) {
            shapes[n] = ov::PartialShape{1, kKVTotal};
        } else if (n.find("position_ids") != std::string::npos) {
            shapes[n] = ov::PartialShape{1, kSeq};
        } else {  // past_key_values.* : [1, H, past, D]
            auto s = ps;
            s[0] = 1;
            s[2] = kPast;
            shapes[n] = s;
        }
    }
    model->reshape(shapes);
    model->validate_nodes_and_infer_types();
    return model;
}

bool is_past_kv(const std::string& name) {
    return name.find("past_key_values") != std::string::npos;
}

// Fill every non-KV input with deterministic data; returns the chosen values so the
// two runs are fed identically.
void fill_common_inputs(ov::InferRequest& req, const std::shared_ptr<const ov::Model>& model) {
    for (const auto& in : model->inputs()) {
        const auto& n = in.get_any_name();
        if (is_past_kv(n)) {
            continue;
        }
        auto t = req.get_tensor(in);
        if (t.get_element_type() == ov::element::i64) {
            auto* d = t.data<int64_t>();
            std::iota(d, d + t.get_size(), 1);  // 1,2,3,...
        } else {
            auto* d = t.data<float>();
            for (size_t i = 0; i < t.get_size(); ++i)
                d[i] = 0.01f * static_cast<float>(i % 7);
        }
    }
}

}  // namespace

class PagedKVIntegrationTest : public ::testing::Test {};

TEST_F(PagedKVIntegrationTest, BlockedKVMatchesContiguousKV) {
    // ---- Baseline: contiguous KV ------------------------------------------------
    auto baseline = build_static_stateless_llm();

    // Capture the past-KV params and a fixed payload to feed both runs.
    std::map<std::string, ov::Shape> kv_shapes;
    for (const auto& in : baseline->inputs()) {
        if (is_past_kv(in.get_any_name())) {
            kv_shapes[in.get_any_name()] = in.get_shape();
        }
    }
    ASSERT_FALSE(kv_shapes.empty()) << "model has no past_key_values inputs";

    // Deterministic KV payload per param (contiguous [1,H,kPast,D]).
    std::map<std::string, std::vector<float>> kv_data;
    for (const auto& [name, shp] : kv_shapes) {
        size_t n = ov::shape_size(shp);
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i)
            v[i] = 0.001f * static_cast<float>((i * 7 + name.size()) % 53);
        kv_data[name] = std::move(v);
    }

    auto compiled_ref = core().compile_model(baseline, "CPU");
    auto req_ref = compiled_ref.create_infer_request();
    fill_common_inputs(req_ref, baseline);
    for (const auto& in : baseline->inputs()) {
        const auto& n = in.get_any_name();
        if (!is_past_kv(n))
            continue;
        auto t = req_ref.get_tensor(in);
        std::copy(kv_data[n].begin(), kv_data[n].end(), t.data<float>());
    }
    req_ref.infer();
    auto logits_ref = req_ref.get_tensor("logits");
    std::vector<float> ref(logits_ref.data<float>(), logits_ref.data<float>() + logits_ref.get_size());

    // ---- Paged: split KV into blocks, back them with KVCacheBlockManager --------
    auto paged = build_static_stateless_llm();
    const bool changed = pass::SplitKVCacheIntoBlocks(kBlock, /*v_transposed=*/false).run_on_model(paged);
    ASSERT_TRUE(changed) << "SplitKVCacheIntoBlocks did not transform the model";
    paged->validate_nodes_and_infer_types();

    // The split must have produced block params named <kv>_block_<i>.
    std::size_t block_param_count = 0;
    for (const auto& in : paged->inputs()) {
        if (in.get_any_name().find("_block_") != std::string::npos)
            ++block_param_count;
    }
    // kPast/kBlock == 2 blocks per original KV param.
    ASSERT_EQ(block_param_count, kv_shapes.size() * (kPast / kBlock));

    auto compiled_paged = core().compile_model(paged, "CPU");
    auto req_paged = compiled_paged.create_infer_request();
    fill_common_inputs(req_paged, paged);

    // One manager per original KV param; block shape [1, H, kBlock, D].
    const std::size_t blocks_per_param = kPast / kBlock;
    for (const auto& [name, shp] : kv_shapes) {
        ov::Shape block_shape = shp;
        block_shape[2] = kBlock;
        KVCacheBlockManager mgr(static_cast<uint32_t>(kBlock),
                                static_cast<uint32_t>(blocks_per_param),
                                block_shape,
                                ov::element::f32,
                                "CPU",
                                nullptr);

        const std::size_t per_block_elems = ov::shape_size(block_shape);
        for (std::size_t b = 0; b < blocks_per_param; ++b) {
            auto block_id = mgr.allocate_block();
            ASSERT_TRUE(block_id.has_value());
            auto block_tensor = mgr.get_block_tensor(block_id.value());
            mgr.update_block_tokens(block_id.value(), static_cast<uint32_t>(kBlock));

            // Copy this block's slice of the contiguous KV into the manager block,
            // then bind it to the matching model block-parameter. KV layout is
            // [1, H, kPast, D] split along the sequence (dim 2): block b covers
            // seq rows [b*kBlock, (b+1)*kBlock).
            const std::size_t H = shp[1], D = shp[3];
            const auto& src = kv_data[name];
            auto* dst = block_tensor->data<float>();
            for (std::size_t h = 0; h < H; ++h) {
                for (std::size_t s = 0; s < kBlock; ++s) {
                    const std::size_t src_seq = b * kBlock + s;
                    const std::size_t src_off = (h * kPast + src_seq) * D;
                    const std::size_t dst_off = (h * kBlock + s) * D;
                    std::copy_n(src.begin() + src_off, D, dst + dst_off);
                }
            }
            (void)per_block_elems;

            const std::string block_name = name + "_block_" + std::to_string(b);
            req_paged.set_tensor(block_name, ov::make_tensor(block_tensor));
        }
    }

    req_paged.infer();
    auto logits_paged = req_paged.get_tensor("logits");
    std::vector<float> paged_out(logits_paged.data<float>(),
                                 logits_paged.data<float>() + logits_paged.get_size());

    // ---- Compare ----------------------------------------------------------------
    ASSERT_EQ(ref.size(), paged_out.size());
    float max_abs = 0.f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_abs = std::max(max_abs, std::abs(ref[i] - paged_out[i]));
    EXPECT_LT(max_abs, 1e-4f) << "blocked-KV logits diverge from contiguous-KV (max_abs_diff=" << max_abs << ")";
}
