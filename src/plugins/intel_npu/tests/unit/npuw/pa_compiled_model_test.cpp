// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "pa_compiled_model.hpp"

namespace {

using ov::npuw::PACompiledModel;

std::string str(const ov::AnyMap& config, const char* key) {
    return config.at(key).as<std::string>();
}

TEST(PAVariantConfig, PartitionRecipe) {
    const ov::AnyMap properties = {{"NPUW_PA", "YES"}, {"NPUW_PA_DEVICE", "CPU"}, {"PERF_COUNT", "YES"}};
    const auto config = PACompiledModel::make_variant_config(properties, "CPU", "bank0");

    EXPECT_EQ(str(config, "NPUW_DEVICES"), "NPU,CPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_PIPELINE"), "REP");
    EXPECT_EQ(str(config, "NPUW_ONLINE_ISOLATE"), "Op:PagedAttentionExtension/pa");
    EXPECT_EQ(str(config, "NPUW_ONLINE_AVOID"), "Op:PagedAttentionExtension/NPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_KEEP_BLOCKS_TAGGED"), "pa");
    EXPECT_EQ(str(config, "NPUW_FOLD"), "YES");
    EXPECT_EQ(str(config, "NPUW_WEIGHTS_BANK"), "bank0");

    // The PA front-end keys must not reach the inner model, everything else
    // the caller passed is forwarded.
    EXPECT_EQ(config.count("NPUW_PA"), 0u);
    EXPECT_EQ(config.count("NPUW_PA_DEVICE"), 0u);
    EXPECT_EQ(str(config, "PERF_COUNT"), "YES");
}

TEST(PAVariantConfig, StripsCallerPartitioningDecisions) {
    const ov::AnyMap properties = {{"NPUW_PLAN", "plan.xml"}, {"NPUW_SUBMODEL_DEVICE", "0:CPU"}};
    const auto config = PACompiledModel::make_variant_config(properties, "GPU", "bank1");

    EXPECT_EQ(config.count("NPUW_PLAN"), 0u);
    EXPECT_EQ(config.count("NPUW_SUBMODEL_DEVICE"), 0u);
    EXPECT_EQ(str(config, "NPUW_DEVICES"), "NPU,GPU");
}

TEST(PAVariantConfig, AppendsToCallerCsvListsWithoutDuplicates) {
    const ov::AnyMap properties = {{"NPUW_ONLINE_AVOID", "Op:Select/NPU"},
                                   {"NPUW_ONLINE_ISOLATE", "Op:PagedAttentionExtension/pa"}};
    const auto config = PACompiledModel::make_variant_config(properties, "CPU", "bank2");

    EXPECT_EQ(str(config, "NPUW_ONLINE_AVOID"), "Op:Select/NPU,Op:PagedAttentionExtension/NPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_ISOLATE"), "Op:PagedAttentionExtension/pa");
}

TEST(PAVariantConfig, SharesOneBankAcrossVariants) {
    const ov::AnyMap properties = {};
    const auto a = PACompiledModel::make_variant_config(properties, "CPU", "shared");
    const auto b = PACompiledModel::make_variant_config(properties, "CPU", "shared");
    EXPECT_EQ(str(a, "NPUW_WEIGHTS_BANK"), str(b, "NPUW_WEIGHTS_BANK"));
}

}  // anonymous namespace
