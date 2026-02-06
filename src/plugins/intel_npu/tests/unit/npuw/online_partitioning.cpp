// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>
#include <regex>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "model_builder/model_builder.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/openvino.hpp"
#include "partitioning/online/compiler.hpp"
#include "partitioning/online/group.hpp"
#include "partitioning/online/snapshot.hpp"

using namespace ov::test::npuw;

namespace {

::intel_npu::Config createConfigWithKeepBlockSize(std::size_t size) {
    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    auto cfg = ::intel_npu::Config(opt_desc);
    ::intel_npu::registerNPUWOptions(*opt_desc);
    std::map<std::string, std::string> cfg_map = {{"NPUW_ONLINE_KEEP_BLOCK_SIZE", std::to_string(size)}};
    cfg.update(cfg_map);
    return cfg;
}

bool isEqualEns(ov::npuw::Ensemble& ens1, ov::npuw::Ensemble& ens2);
bool isEqualEns(ov::npuw::Ensemble& ens1, ov::npuw::Ensemble& ens2) {
    if (ens1.groups.size() != ens2.groups.size()) {
        return false;
    }

    for (auto& g : ens1.groups) {
        std::sort(g.input_layers.begin(), g.input_layers.end());
        std::sort(g.output_layers.begin(), g.output_layers.end());
        std::sort(g.all_layers.begin(), g.all_layers.end());
    }

    for (auto& g : ens2.groups) {
        std::sort(g.input_layers.begin(), g.input_layers.end());
        std::sort(g.output_layers.begin(), g.output_layers.end());
        std::sort(g.all_layers.begin(), g.all_layers.end());
    }

    std::sort(ens1.groups.begin(), ens1.groups.end(), [](const ov::npuw::Group& g1, const ov::npuw::Group& g2) {
        return g1.all_layers.front() < g2.all_layers.front();
    });

    std::sort(ens2.groups.begin(), ens2.groups.end(), [](const ov::npuw::Group& g1, const ov::npuw::Group& g2) {
        return g1.all_layers.front() < g2.all_layers.front();
    });

    for (size_t i = 0; i < ens1.groups.size(); ++i) {
        const auto& g1 = ens1.groups.at(i);
        const auto& g2 = ens2.groups.at(i);

        if (g1.avoid_list != g2.avoid_list || g1.input_layers != g2.input_layers ||
            g1.output_layers != g2.output_layers || g1.all_layers != g2.all_layers) {
            return false;
        }

        // Can't compare them directly since they are random, but dont't affect the structure
        if ((g1.repeated_id.empty() && !g2.repeated_id.empty()) ||
            (!g1.repeated_id.empty() && g2.repeated_id.empty())) {
            return false;
        }
    }

    if (ens1.repeated.size() != ens2.repeated.size()) {
        return false;
    }

    auto get_sorted_rep = [](const std::map<std::string, ov::npuw::RepeatedBlock>& rep) {
        std::vector<std::vector<std::set<std::string>>> sorted_rep;

        std::transform(rep.begin(), rep.end(), std::back_inserter(sorted_rep), [](const auto& v) {
            return v.second.matches;
        });

        for (auto& g : sorted_rep) {
            std::sort(g.begin(), g.end(), [](const auto& a, const auto& b) {
                return *a.begin() < *b.begin();
            });
        }

        std::sort(sorted_rep.begin(), sorted_rep.end(), [](const auto& a, const auto& b) {
            return *a.front().begin() < *b.front().begin();
        });

        return sorted_rep;
    };

    if (get_sorted_rep(ens1.repeated) != get_sorted_rep(ens2.repeated)) {
        return false;
    }

    return true;
}

class IsRegularResultCaseParametrized : public ::testing::TestWithParam<std::tuple<std::vector<std::size_t>, bool>> {};
class IsRegularParameterCaseParametrized : public ::testing::TestWithParam<std::tuple<std::vector<std::size_t>, bool>> {
};

};  // namespace

TEST(OnlinePartitioningTest, Partitioning_IsTheSame_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto cfg = createConfigWithKeepBlockSize(9);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    for (size_t i = 0; i < 100; ++i) {
        auto ens_again = ov::npuw::online::buildPartitioning(model, cfg);
        EXPECT_TRUE(isEqualEns(ens, ens_again));
    }
}

TEST(OnlinePartitioningTest, Partitioning_IsTheSame_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto cfg = createConfigWithKeepBlockSize(9);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    for (size_t i = 0; i < 100; ++i) {
        auto ens_again = ov::npuw::online::buildPartitioning(model, cfg);
        EXPECT_TRUE(isEqualEns(ens, ens_again));
    }
}

TEST(OnlinePartitioningTest, Partitioning_SingleGroup_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->singleGroup();
    EXPECT_EQ(snap->graphSize(), 1);
}

TEST(OnlinePartitioningTest, Partitioning_SingleGroup_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->singleGroup();
    EXPECT_EQ(snap->graphSize(), 1);
}

TEST(OnlinePartitioningTest, Partitioning_buildGraph_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();
    auto g = snap->getGraph();
    for (const auto& nh : g->sorted()) {
        ov::npuw::online::Group::GPtr group = g->meta(nh).get<ov::npuw::online::Group::GPtr>();
        EXPECT_EQ(group->size(), 1);
    }
    EXPECT_EQ(snap->getNodeToGroupMap()->size(), snap->graphSize());
}

TEST(OnlinePartitioningTest, Partitioning_buildGraph_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();
    auto g = snap->getGraph();
    for (const auto& nh : g->sorted()) {
        ov::npuw::online::Group::GPtr group = g->meta(nh).get<ov::npuw::online::Group::GPtr>();
        EXPECT_EQ(group->size(), 1);
    }
    EXPECT_EQ(snap->getNodeToGroupMap()->size(), snap->graphSize());
}

TEST(OnlinePartitioningTest, Partitioning_earlyAvoids_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    ov::npuw::online::PassContext ctx;
    ctx.avoids = {{ov::npuw::online::PatternType::OP, "Gather", "mydevice"},
                  {ov::npuw::online::PatternType::OP, "MatMul", "mydevice"}};
    snap->setCtx(ctx);
    snap->buildGraph();
    snap->earlyAvoids();
    auto g = snap->getGraph();
    size_t count = 0;
    for (const auto& nh : g->sorted()) {
        ov::npuw::online::Group::GPtr group = g->meta(nh).get<ov::npuw::online::Group::GPtr>();
        EXPECT_EQ(group->size(), 1);
        if (group->avoidedTargets().size() == 1 && *(group->avoidedTargets().begin()) == "mydevice") {
            ++count;
        }
    }
    EXPECT_EQ(count, 2);
}

TEST(OnlinePartitioningTest, Partitioning_earlyAvoids_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    ov::npuw::online::PassContext ctx;
    ctx.avoids = {{ov::npuw::online::PatternType::OP, "Gather", "mydevice"},
                  {ov::npuw::online::PatternType::OP, "MatMul", "mydevice"}};
    snap->setCtx(ctx);
    snap->buildGraph();
    snap->earlyAvoids();
    auto g = snap->getGraph();
    size_t count = 0;
    for (const auto& nh : g->sorted()) {
        ov::npuw::online::Group::GPtr group = g->meta(nh).get<ov::npuw::online::Group::GPtr>();
        EXPECT_EQ(group->size(), 1);
        if (group->avoidedTargets().size() == 1 && *(group->avoidedTargets().begin()) == "mydevice") {
            ++count;
        }
    }
    EXPECT_EQ(count, 20);
}

TEST(OnlinePartitioningTest, Partitioning_collectLHF_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {10, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->collectLHF();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_collectLHF_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {82, 82};
    size_t iter = 0;

    snap->repeat([&] {
        snap->collectLHF();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseRemnants_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {10, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseRemnants();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseRemnants_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {75, 38, 19, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseRemnants();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseRemnantsExtended_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {10, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseRemnantsExtended_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {10, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseInputs_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {15, 14, 14};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseInputs();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseInputs_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {148, 138, 138};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseInputs();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_Just_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes_lhf = {10, 10};
    size_t iter_lhf = 0;

    std::vector<std::size_t> sizes_fr = {10, 10};
    size_t iter_fr = 0;

    snap->repeat([&] {
        snap->collectLHF();
        EXPECT_LT(iter_lhf, sizes_lhf.size());
        EXPECT_EQ(snap->graphSize(), sizes_lhf[iter_lhf++]);
    });
    snap->repeat([&] {
        snap->fuseRemnants();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_Just_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes_lhf = {82, 82};
    size_t iter_lhf = 0;

    std::vector<std::size_t> sizes_fr = {41, 21, 11, 10, 10};
    size_t iter_fr = 0;

    snap->repeat([&] {
        snap->collectLHF();
        EXPECT_LT(iter_lhf, sizes_lhf.size());
        EXPECT_EQ(snap->graphSize(), sizes_lhf[iter_lhf++]);
    });
    snap->repeat([&] {
        snap->fuseRemnants();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_RepeatedBlocks_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes_fr = {10, 10};
    size_t iter_fr = 0;

    snap->earlyAvoids();
    snap->earlyRegroup();
    snap->repeatedBlocks();
    EXPECT_EQ(snap->graphSize(), 17);

    auto matches = snap->getMatches();
    EXPECT_EQ(matches.size(), 0);

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_RepeatedBlocks_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes_fr = {12, 12};
    size_t iter_fr = 0;

    snap->earlyAvoids();
    snap->earlyRegroup();
    snap->repeatedBlocks();
    EXPECT_EQ(snap->graphSize(), 18);

    auto matches = snap->getMatches();
    EXPECT_EQ(matches.size(), 1);

    for (const auto& m : matches) {
        EXPECT_EQ(m.second.size(), 17);
        for (const auto& layers : m.second) {
            EXPECT_EQ(layers.size(), 10);
        }
    }

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_Compute_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);

    std::vector<std::size_t> sizes_fr = {10, 10};
    size_t iter_fr = 0;

    ov::npuw::online::PassContext ctx;
    ctx.isolates = {{ov::npuw::online::PatternType::OP, "Transpose", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "ScatterUpdate", "test_compute"}};
    ctx.nofolds = {"test_compute"};
    snap->setCtx(ctx);

    snap->buildGraph();
    snap->earlyAvoids();
    snap->earlyRegroup();
    snap->repeatedBlocks();
    EXPECT_EQ(snap->graphSize(), 17);

    auto matches = snap->getMatches();
    EXPECT_EQ(matches.size(), 0);

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_Compute_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);

    std::vector<std::size_t> sizes_fr = {10, 10};
    size_t iter_fr = 0;

    ov::npuw::online::PassContext ctx;
    ctx.isolates = {{ov::npuw::online::PatternType::OP, "Gather", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "ScatterUpdate", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "ShapeOf", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "Divide", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "Floor", "test_compute"}};
    ctx.nofolds = {"test_compute"};
    snap->setCtx(ctx);

    snap->buildGraph();
    snap->earlyAvoids();
    snap->earlyRegroup();
    snap->repeatedBlocks();
    EXPECT_EQ(snap->graphSize(), 38);

    // FIXME: create a config in which there will be repeated blocks
    auto matches = snap->getMatches();
    EXPECT_EQ(matches.size(), 0);

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Avoids_Pipeline_None) {
    std::shared_ptr<ov::op::v0::Parameter> input =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    input->set_friendly_name("input");

    auto n1 = std::make_shared<ov::op::v1::Add>(input, input);
    n1->set_friendly_name("n1");
    auto n2 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{1});
    n2->set_friendly_name("n2");
    auto n3 = std::make_shared<ov::op::v1::Divide>(n1, n2, true);
    n3->set_friendly_name("n3");
    auto n4 = std::make_shared<ov::op::v0::Sin>(n1);
    n4->set_friendly_name("n4");
    auto n5 = std::make_shared<ov::op::v0::Cos>(n1);
    n5->set_friendly_name("n5");
    auto n6 = std::make_shared<ov::op::v0::Sin>(n3);
    n6->set_friendly_name("n6");
    auto n7 = std::make_shared<ov::op::v0::Cos>(n3);
    n7->set_friendly_name("n7");
    auto n8 = std::make_shared<ov::op::v0::Concat>(std::vector<std::shared_ptr<ov::Node>>{n1, n4, n5, n6, n7}, -1);
    n8->set_friendly_name("n8");

    auto result = std::make_shared<ov::op::v0::Result>(n8);
    result->set_friendly_name("res");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    auto cfg = ::intel_npu::Config(opt_desc);
    ::intel_npu::registerNPUWOptions(*opt_desc);
    std::map<std::string, std::string> cfg_map = {{"NPUW_ONLINE_AVOID", "Op:Sin/NPU,Op:Cos/NPU"},
                                                  {"NPUW_ONLINE_PIPELINE", "NONE"}};
    cfg.update(cfg_map);

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_EQ(ens.groups.size(), 3);
}

TEST(OnlinePartitioningTest, IsRegularIOCaseWhenNoRB) {
    bool expected_result = false;

    ModelBuilder mb;
    std::vector<std::shared_ptr<ov::Model>> models = {mb.get_model_with_one_op(),
                                                      mb.get_model_without_repeated_blocks()};

    for (auto model : models) {
        auto cfg = createConfigWithKeepBlockSize(9);
        auto ens = ov::npuw::online::buildPartitioning(model, cfg);

        EXPECT_EQ(ens.repeated.size(), 0);  // sanity check that we don't have repeated blocks
        EXPECT_EQ(ens.irregular_io, expected_result);
    }
}

TEST(OnlinePartitioningTest, IsRegularResultCaseMultipleOutputs) {
    ModelBuilder mb;
    std::vector<std::pair<std::shared_ptr<ov::Model>, bool>> model_expected = {
        {mb.get_model_with_multi_output_repeating_blocks(10, /*irregular_io=*/true), /*irregular_io=*/true},
        {mb.get_model_with_multi_output_repeating_blocks(10, /*irregular_io=*/false),
         /*irregular_io=*/false}};

    for (auto [model, expected_result] : model_expected) {
        auto cfg = createConfigWithKeepBlockSize(3);
        auto ens = ov::npuw::online::buildPartitioning(model, cfg);

        EXPECT_EQ(ens.repeated.size(), 1);  // sanity check that we have repeated blocks
        EXPECT_EQ(ens.irregular_io, expected_result);
    }
}

TEST_P(IsRegularResultCaseParametrized, CheckForDifferentResultConfigs) {
    auto [block_indices, expected_result] = GetParam();

    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks_and_results(10, block_indices);

    auto cfg = createConfigWithKeepBlockSize(9);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_EQ(ens.repeated.size(), 1);  // sanity check that we have repeated blocks
    EXPECT_EQ(ens.irregular_io, expected_result);
}

INSTANTIATE_TEST_SUITE_P(OnlinePartitioningTest,
                         IsRegularResultCaseParametrized,
                         ::testing::Values(
                             // All blocks have an ov::Result consumer
                             std::make_tuple(std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                             /*irregular_io=*/false),
                             // Some blocks have an ov::Result consumer
                             std::make_tuple(std::vector<std::size_t>{2, 5, 8}, /*irregular_io=*/true),
                             // Only last block has an additional ov::Result consumer
                             std::make_tuple(std::vector<std::size_t>{9}, /*irregular_io=*/true),
                             // No blocks have an additional ov::Result consumers
                             std::make_tuple(std::vector<std::size_t>{}, /*irregular_io=*/false)));

TEST_P(IsRegularParameterCaseParametrized, CheckForDifferentParameterConfigs) {
    auto [block_indices, expected_result] = GetParam();

    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks_and_parameters(10, block_indices);
    auto cfg = createConfigWithKeepBlockSize(4);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_EQ(ens.repeated.size(), 1);  // sanity check that we have repeated blocks
    EXPECT_EQ(ens.irregular_io, expected_result);
}

INSTANTIATE_TEST_SUITE_P(OnlinePartitioningTest,
                         IsRegularParameterCaseParametrized,
                         ::testing::Values(
                             // All blocks have an ov::Parameter producer
                             std::make_tuple(std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                             /*irregular_io=*/false),
                             // Some blocks have an ov::Parameter producer
                             std::make_tuple(std::vector<std::size_t>{1, 2, 3}, /*irregular_io=*/true),
                             // Only one block has an additional ov::Parameter producer
                             std::make_tuple(std::vector<std::size_t>{5}, /*irregular_io=*/true),
                             // No blocks have an additional ov::Parameter producers
                             std::make_tuple(std::vector<std::size_t>{}, /*irregular_io=*/false)));

// ============== LLM Model Builder Tests ==============

TEST(LLMModelBuilderTest, BasicLLMModel_HasCorrectStructure) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 2;
    config.hidden_size = 64;
    config.num_heads = 4;
    config.head_dim = 16;
    config.vocab_size = 100;

    auto model = mb.build_llm(config);

    ASSERT_NE(model, nullptr);

    auto params = model->get_parameters();
    bool has_input_ids = false;
    bool has_attention_mask = false;
    bool has_position_ids = false;

    for (const auto& param : params) {
        if (param->get_friendly_name() == "input_ids") has_input_ids = true;
        if (param->get_friendly_name() == "attention_mask") has_attention_mask = true;
        if (param->get_friendly_name() == "position_ids") has_position_ids = true;
    }

    EXPECT_TRUE(has_input_ids) << "Missing input_ids parameter";
    EXPECT_TRUE(has_attention_mask) << "Missing attention_mask parameter";
    EXPECT_TRUE(has_position_ids) << "Missing position_ids parameter";

    // KV cache is stateful (ReadValue/Assign with Variables), not parametric
    auto variables = model->get_variables();
    size_t kv_var_count = 0;
    for (const auto& var : variables) {
        if (var->get_info().variable_id.find("past_key_values") != std::string::npos) {
            kv_var_count++;
        }
    }
    EXPECT_EQ(kv_var_count, config.num_layers * 2) << "Expected " << config.num_layers * 2 << " KV cache variables";

    // Sinks (Assign nodes) for KV cache state updates
    auto sinks = model->get_sinks();
    EXPECT_EQ(sinks.size(), config.num_layers * 2) << "Expected K and V Assign sinks per layer";

    auto results = model->get_results();
    EXPECT_GE(results.size(), 1) << "Expected at least logits result";
}

TEST(LLMModelBuilderTest, BasicLLMModel_ContainsSDPA) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 1;

    auto model = mb.build_llm(config);

    // Native SDPA v13 op used for attention
    size_t sdpa_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("ScaledDotProductAttention")) {
            sdpa_count++;
        }
    }
    EXPECT_EQ(sdpa_count, config.num_layers) << "Expected one SDPA per layer";
}

TEST(LLMModelBuilderTest, BasicLLMModel_HasKVCacheConcat) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 2;

    auto model = mb.build_llm(config);

    size_t concat_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("Concat")) {
            concat_count++;
        }
    }
    EXPECT_GE(concat_count, config.num_layers * 2) << "Expected K and V concat per layer";
}

TEST(LLMModelBuilderTest, Partitioning_LLMModel) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 4;

    auto model = mb.build_llm(config);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->get_friendly_name(), "llm_test_model");

    size_t sdpa_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("ScaledDotProductAttention")) {
            sdpa_count++;
        }
    }
    EXPECT_EQ(sdpa_count, config.num_layers) << "Expected one SDPA per layer";
}

TEST(LLMModelBuilderTest, LLMModel_WithoutPositionIds) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 1;
    config.use_position_ids = false;

    auto model = mb.build_llm(config);
    ASSERT_NE(model, nullptr);

    auto params = model->get_parameters();
    bool has_position_ids = false;
    for (const auto& param : params) {
        if (param->get_friendly_name() == "position_ids") has_position_ids = true;
    }
    EXPECT_FALSE(has_position_ids) << "Should not have position_ids when disabled";

    // Should still have SDPA (just no RoPE)
    size_t sdpa_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("ScaledDotProductAttention")) {
            sdpa_count++;
        }
    }
    EXPECT_EQ(sdpa_count, config.num_layers);
}

TEST(LLMModelBuilderTest, LLMModel_WithRoPE_HasGatherOps) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 1;
    config.use_position_ids = true;
    config.rope_type = RoPEType::HALF_ROTATION;

    auto model = mb.build_llm(config);
    ASSERT_NE(model, nullptr);

    // RoPE uses Gather to index cos/sin tables by position_ids
    // 2 Gathers per RoPE (cos + sin), applied to Q and K = 4 Gathers per layer
    // Plus 1 Gather for the embedding
    size_t gather_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("Gather")) {
            gather_count++;
        }
    }
    EXPECT_GE(gather_count, 4 * config.num_layers + 1) << "Expected Gather ops for RoPE cos/sin + embedding";
}

TEST(LLMModelBuilderTest, LLMModel_CompressedWeights_FP16) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 1;
    config.weight_format = WeightFormat::FP16;
    config.use_position_ids = false;

    auto model = mb.build_llm(config);
    ASSERT_NE(model, nullptr);

    // FP16 weights produce Convert ops
    size_t convert_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("Convert")) {
            convert_count++;
        }
    }
    EXPECT_GT(convert_count, 0) << "Expected Convert ops for FP16 weight decompression";
}

TEST(LLMModelBuilderTest, LLMModel_CompressedWeights_INT8) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 1;
    config.weight_format = WeightFormat::INT8;
    config.use_position_ids = false;

    auto model = mb.build_llm(config);
    ASSERT_NE(model, nullptr);

    // INT8 weights produce Convert + Multiply(scale) ops
    size_t convert_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("Convert")) {
            convert_count++;
        }
    }
    EXPECT_GT(convert_count, 0) << "Expected Convert ops for INT8 weight decompression";
}

TEST(LLMModelBuilderTest, LLMModel_ConfigurableLMHead) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 1;
    config.weight_format = WeightFormat::FP32;
    config.lm_head_format = WeightFormat::FP16;
    config.use_position_ids = false;

    auto model = mb.build_llm(config);
    ASSERT_NE(model, nullptr);

    // The LM head should have FP16 weights with Convert,
    // while other layers use FP32.
    // The attention mask also has a Convert (i64 -> f32), so expect 2 total.
    size_t convert_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("Convert")) {
            convert_count++;
        }
    }
    EXPECT_EQ(convert_count, 2) << "Expected Convert for LM head FP16 weight + attention mask";
}

// ============== Modular Building Block Tests ==============

TEST(ModelBuilderBlocksTest, MakeLinear_CreatesCorrectStructure) {
    ModelBuilder mb;
    mb.clear();

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 64});
    mb.track(input);

    auto output = mb.make_linear(input->output(0), 128, "test_linear", ov::element::f32, false);

    auto shape = output.get_partial_shape();
    EXPECT_EQ(shape.rank().get_length(), 3);
    EXPECT_EQ(shape[2].get_length(), 128);
}

TEST(ModelBuilderBlocksTest, MakeLinear_CompressedWeights) {
    ModelBuilder mb;
    mb.clear();

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 64});
    mb.track(input);

    // FP16 compressed weight
    auto output = mb.make_linear(input->output(0), 128, "test_linear_fp16",
                                  ov::element::f32, false, WeightFormat::FP16);

    auto shape = output.get_partial_shape();
    EXPECT_EQ(shape[2].get_length(), 128);

    // Verify the MatMul input has a Convert in its chain (from FP16 weight)
    auto matmul_node = output.get_node_shared_ptr();
    EXPECT_EQ(matmul_node->get_type_name(), std::string("MatMul"));
    auto weight_input = matmul_node->input(1).get_source_output().get_node_shared_ptr();
    EXPECT_EQ(weight_input->get_type_name(), std::string("Convert"));
}

TEST(ModelBuilderBlocksTest, MakeNorm_CreatesLayerNormalization) {
    ModelBuilder mb;
    mb.clear();

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 64});
    mb.track(input);

    auto output = mb.make_norm(input->output(0), 64, "test_ln", NormType::LAYER_NORM);

    auto shape = output.get_partial_shape();
    EXPECT_EQ(shape[2].get_length(), 64);
}

TEST(ModelBuilderBlocksTest, MakeNorm_CreatesRMSNormalization) {
    ModelBuilder mb;
    mb.clear();

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 64});
    mb.track(input);

    auto output = mb.make_norm(input->output(0), 64, "test_rms", NormType::RMS_NORM);

    auto shape = output.get_partial_shape();
    EXPECT_EQ(shape[2].get_length(), 64);
}

TEST(ModelBuilderBlocksTest, MakeFFN_CreatesSwiGLU) {
    ModelBuilder mb;
    mb.clear();

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 64});
    mb.track(input);

    auto output = mb.make_ffn(input->output(0), 64, 256, "test_ffn", FFNType::SWIGLU);

    auto shape = output.get_partial_shape();
    EXPECT_EQ(shape[2].get_length(), 64);
}

TEST(ModelBuilderBlocksTest, MakeFFN_CreatesGELU) {
    ModelBuilder mb;
    mb.clear();

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 64});
    mb.track(input);

    auto output = mb.make_ffn(input->output(0), 64, 256, "test_ffn_gelu", FFNType::GELU);

    auto shape = output.get_partial_shape();
    EXPECT_EQ(shape[2].get_length(), 64);
}

TEST(ModelBuilderBlocksTest, MakeKVCacheConcat_CreatesNPUWPattern) {
    ModelBuilder mb;
    mb.clear();

    auto current_kv = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1, 16});
    auto batch_src = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 1});
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    mb.track(current_kv);
    mb.track(batch_src);
    mb.track(beam_idx);

    auto result = mb.make_kv_cache_concat(current_kv->output(0), batch_src->output(0),
                                           beam_idx->output(0), 4, 16, "past_key_values.0.key");

    ASSERT_NE(result.assign, nullptr);
    EXPECT_EQ(result.assign->get_type_name(), std::string("Assign"));

    auto concat_node = result.concatenated.get_node_shared_ptr();
    EXPECT_EQ(concat_node->get_type_name(), std::string("Concat"));
}

TEST(ModelBuilderBlocksTest, MakeRoPE_HalfRotation) {
    ModelBuilder mb;
    mb.clear();

    // Input: [batch, seq, heads, head_dim]
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 4, 16});
    auto pos_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 8});
    mb.track(input);
    mb.track(pos_ids);

    auto output = mb.make_rope(input->output(0), pos_ids->output(0), 16, 128,
                                "test_rope", RoPEType::HALF_ROTATION);

    // Output should have same shape as input
    auto shape = output.get_partial_shape();
    EXPECT_EQ(shape.rank().get_length(), 4);
    EXPECT_EQ(shape[3].get_length(), 16);
}

TEST(ModelBuilderBlocksTest, MakeRoPE_Interleaved) {
    ModelBuilder mb;
    mb.clear();

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 4, 16});
    auto pos_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 8});
    mb.track(input);
    mb.track(pos_ids);

    auto output = mb.make_rope(input->output(0), pos_ids->output(0), 16, 128,
                                "test_rope_interleaved", RoPEType::INTERLEAVED);

    auto shape = output.get_partial_shape();
    EXPECT_EQ(shape.rank().get_length(), 4);
    EXPECT_EQ(shape[3].get_length(), 16);
}

TEST(ModelBuilderBlocksTest, MakeDecoderLayer_CreatesCompleteBlock) {
    ModelBuilder mb;
    mb.clear();

    LLMConfig config;
    config.hidden_size = 64;
    config.num_heads = 4;
    config.head_dim = 16;
    config.intermediate_size = 256;
    config.use_kv_cache = true;
    config.use_position_ids = false;

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 64});
    auto batch_src = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 1, -1});
    mb.track(input);
    mb.track(batch_src);
    mb.track(beam_idx);
    mb.track(attn_mask);

    ov::Output<ov::Node> empty_pos_ids;
    auto result = mb.make_decoder_layer(input->output(0), config, 0, empty_pos_ids,
                                         batch_src->output(0), beam_idx->output(0),
                                         attn_mask->output(0));

    EXPECT_EQ(result.sinks.size(), 2) << "Expected K and V cache Assign sinks";

    auto shape = result.output.get_partial_shape();
    EXPECT_EQ(shape[2].get_length(), 64);
}

TEST(ModelBuilderBlocksTest, LLMModel_WithoutKVCache) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 1;
    config.use_kv_cache = false;

    auto model = mb.build_llm(config);

    auto variables = model->get_variables();
    EXPECT_EQ(variables.size(), 0) << "Should have no KV cache variables when disabled";

    auto sinks = model->get_sinks();
    EXPECT_EQ(sinks.size(), 0) << "Should have no Assign sinks when KV cache disabled";
}

TEST(ModelBuilderBlocksTest, LLMModel_GQA_DifferentKVHeads) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 1;
    config.num_heads = 8;
    config.num_kv_heads = 2;
    config.head_dim = 16;
    config.hidden_size = 128;

    auto model = mb.build_llm(config);
    ASSERT_NE(model, nullptr);

    // KV cache is stateful — check variables have correct kv_heads dimension
    auto variables = model->get_variables();
    size_t kv_var_count = 0;
    for (const auto& var : variables) {
        if (var->get_info().variable_id.find("past_key_values") != std::string::npos) {
            auto shape = var->get_info().data_shape;
            EXPECT_EQ(shape[1].get_length(), config.num_kv_heads);
            kv_var_count++;
        }
    }
    EXPECT_EQ(kv_var_count, config.num_layers * 2) << "Expected K and V cache variables per layer";

    size_t sdpa_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("ScaledDotProductAttention")) {
            sdpa_count++;
        }
    }
    EXPECT_EQ(sdpa_count, config.num_layers) << "Expected one SDPA per layer";
}

// ============== Builder Pattern Tests ==============

TEST(ModelBuilderPatternTest, BuilderPattern_ManualModel) {
    ModelBuilder mb;
    mb.clear();

    auto input = mb.parameter(ov::element::f32, ov::PartialShape{-1, -1, 64}, "input");
    auto normed = mb.make_norm(input->output(0), 64, "norm", NormType::RMS_NORM);
    auto ffn_out = mb.make_ffn(normed, 64, 256, "ffn", FFNType::SWIGLU);
    mb.result(ffn_out, "output");

    auto model = mb.build("test_builder_model");

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->get_friendly_name(), "test_builder_model");
    EXPECT_EQ(model->get_parameters().size(), 1);
    EXPECT_EQ(model->get_results().size(), 1);
    EXPECT_EQ(model->get_parameters()[0]->get_friendly_name(), "input");
    EXPECT_EQ(model->get_results()[0]->get_friendly_name(), "output");
}

// ============== KV Cache Pattern Tests ==============

TEST(KVCachePatternTest, LLMModel_KVCacheNaming_MatchesNPUWPattern) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 2;
    config.use_kv_cache = true;

    auto model = mb.build_llm(config);
    ASSERT_NE(model, nullptr);

    // KV cache is stateful — check variable naming follows StatefulToStateless convention:
    //   "past_key_values.N.keypresent.N.key" (input name + output name concatenated)
    auto variables = model->get_variables();
    size_t key_count = 0;
    size_t value_count = 0;

    std::regex key_pattern(R"(past_key_values\.\d+\.keypresent\.\d+\.key)");
    std::regex value_pattern(R"(past_key_values\.\d+\.valuepresent\.\d+\.value)");

    for (const auto& var : variables) {
        const std::string name = var->get_info().variable_id;
        if (std::regex_match(name, key_pattern)) {
            key_count++;
        }
        if (std::regex_match(name, value_pattern)) {
            value_count++;
        }
    }

    EXPECT_EQ(key_count, config.num_layers) << "Expected one key cache variable per layer";
    EXPECT_EQ(value_count, config.num_layers) << "Expected one value cache variable per layer";
}

TEST(KVCachePatternTest, LLMModel_ConcatPattern_ForKVCache) {
    ModelBuilder mb;
    LLMConfig config;
    config.num_layers = 1;
    config.use_kv_cache = true;

    auto model = mb.build_llm(config);
    ASSERT_NE(model, nullptr);

    // Stateful KV cache pattern: ReadValue -> Gather(beam_idx) -> Concat
    // Check for Concat nodes whose input chain contains a Gather fed by ReadValue
    size_t kv_concat_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("Concat")) {
            auto input0 = op->input(0).get_source_output().get_node_shared_ptr();
            // input0 should be Gather (beam search reordering)
            if (input0->get_type_name() == std::string("Gather")) {
                auto gather_input = input0->input(0).get_source_output().get_node_shared_ptr();
                if (gather_input->get_type_name() == std::string("ReadValue")) {
                    kv_concat_count++;
                }
            }
        }
    }
    EXPECT_EQ(kv_concat_count, config.num_layers * 2) << "Expected ReadValue->Gather->Concat pattern for K and V";
}
