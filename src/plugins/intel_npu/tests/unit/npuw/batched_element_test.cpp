// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for the batched v1 element (ov::npuw::batched). The element wraps an
// inner compiled model that only supports batch size 1 and unrolls a batched
// [N, ...] inference into N independent [1, ...] inner inferences, stacking the
// per-row outputs back into [N, ...].
//
// The tests use a recording mock inner compiled model (no device/plugin needed),
// mirroring the failsafe element tests. The mock applies a deterministic per-row
// transform so the test can assert that:
//   * each row is sliced and fed to the inner request correctly,
//   * the inner is invoked once per row with its state reset between rows,
//   * outputs are stacked into the right [N, ...] slots,
//   * a batched result equals scoring each row on its own.

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "v1/elements/batched.hpp"

namespace {

constexpr std::size_t K = 4;       // feature width
constexpr float ROW_BIAS = 100.0f;  // deterministic per-element transform: out = in + ROW_BIAS

// Shared recorder so tests can observe what the inner request saw.
struct Recorder {
    int infer_count = 0;
    int reset_count = 0;
    std::vector<std::vector<float>> seen_inputs;  // one entry per inner infer()
};

std::shared_ptr<ov::Model> make_passthrough_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, K});
    param->set_friendly_name("data");
    param->get_output_tensor(0).set_names({"data"});
    auto result = std::make_shared<ov::op::v0::Result>(param);
    result->get_output_tensor(0).set_names({"out"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "passthrough");
}

class MockState : public ov::IVariableState {
public:
    explicit MockState(std::shared_ptr<Recorder> rec) : ov::IVariableState("mock_state"), m_rec(std::move(rec)) {}
    void reset() override {
        m_rec->reset_count++;
    }
    void set_state(const ov::SoPtr<ov::ITensor>&) override {}
    ov::SoPtr<ov::ITensor> get_state() const override {
        return {};
    }

private:
    std::shared_ptr<Recorder> m_rec;
};

// Inner sync request: out[r][k] = in[r][k] + ROW_BIAS, on a [1, K] row.
class MockInnerSync : public ov::ISyncInferRequest {
public:
    MockInnerSync(std::shared_ptr<const ov::ICompiledModel> compiled_model, std::shared_ptr<Recorder> rec)
        : ov::ISyncInferRequest(std::move(compiled_model)),
          m_rec(std::move(rec)) {}

    void infer() override {
        const auto in = get_tensor(get_inputs()[0]);
        const auto shape = in->get_shape();
        const std::size_t n = in->get_size();
        const float* in_data = static_cast<const float*>(in->data());

        m_rec->infer_count++;
        m_rec->seen_inputs.emplace_back(in_data, in_data + n);

        auto out = ov::get_tensor_impl(ov::Tensor(ov::element::f32, shape));
        float* out_data = static_cast<float*>(out->data());
        for (std::size_t i = 0; i < n; ++i) {
            out_data[i] = in_data[i] + ROW_BIAS;
        }
        set_tensor(get_outputs()[0], out);
    }

    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {ov::SoPtr<ov::IVariableState>(std::make_shared<MockState>(m_rec))};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

private:
    std::shared_ptr<Recorder> m_rec;
};

class MockInnerCompiled : public ov::ICompiledModel {
public:
    MockInnerCompiled(const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      std::shared_ptr<Recorder> rec)
        : ov::ICompiledModel(model, plugin),
          m_rec(std::move(rec)) {}

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return std::make_shared<MockInnerSync>(shared_from_this(), m_rec);
    }
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                        get_task_executor(),
                                                        get_callback_executor());
    }
    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return nullptr;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }

private:
    std::shared_ptr<Recorder> m_rec;
};

// Minimal IPlugin so ICompiledModel can be constructed in a unit test.
class NullPlugin : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        return {};
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        return {};
    }
};

class NPUWBatchedElementTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
        m_model = make_passthrough_model();
        m_recorder = std::make_shared<Recorder>();
    }

    ov::SoPtr<ov::ICompiledModel> make_inner() {
        return {std::make_shared<MockInnerCompiled>(m_model, m_plugin, m_recorder), {}};
    }

    static ov::Tensor make_input(const std::vector<std::vector<float>>& rows) {
        ov::Tensor t(ov::element::f32, ov::Shape{rows.size(), K});
        float* data = t.data<float>();
        for (std::size_t r = 0; r < rows.size(); ++r) {
            std::copy(rows[r].begin(), rows[r].end(), data + r * K);
        }
        return t;
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<Recorder> m_recorder;
};

TEST_F(NPUWBatchedElementTest, DisabledReturnsInnerUnwrapped) {
    auto inner = make_inner();
    auto wrapped = ov::npuw::batched::CompiledModel::create(m_model, m_plugin, inner, /*enabled=*/false);
    EXPECT_EQ(wrapped._ptr, inner._ptr);
}

TEST_F(NPUWBatchedElementTest, EnabledWrapsInner) {
    auto inner = make_inner();
    auto wrapped = ov::npuw::batched::CompiledModel::create(m_model, m_plugin, inner, /*enabled=*/true);
    EXPECT_NE(wrapped._ptr, inner._ptr);
    EXPECT_NE(std::dynamic_pointer_cast<ov::npuw::batched::CompiledModel>(wrapped._ptr), nullptr);
}

TEST_F(NPUWBatchedElementTest, BatchedMatchesPerRowAndStacks) {
    auto wrapped =
        ov::npuw::batched::CompiledModel::create(m_model, m_plugin, make_inner(), /*enabled=*/true);
    auto request = wrapped->create_infer_request();

    const std::vector<std::vector<float>> rows = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {10.0f, 20.0f, 30.0f, 40.0f},
        {-1.0f, -2.0f, -3.0f, -4.0f},
    };
    const auto input = make_input(rows);
    request->set_tensor(wrapped->inputs()[0], ov::get_tensor_impl(input));

    request->infer();

    const auto output = request->get_tensor(wrapped->outputs()[0]);
    ASSERT_EQ(output->get_shape(), (ov::Shape{rows.size(), K}));

    const float* out = static_cast<const float*>(output->data());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        for (std::size_t k = 0; k < K; ++k) {
            EXPECT_FLOAT_EQ(out[r * K + k], rows[r][k] + ROW_BIAS) << "row " << r << " col " << k;
        }
    }

    // One inner infer per row, state reset before each row, correct slices seen.
    EXPECT_EQ(m_recorder->infer_count, static_cast<int>(rows.size()));
    EXPECT_EQ(m_recorder->reset_count, static_cast<int>(rows.size()));
    ASSERT_EQ(m_recorder->seen_inputs.size(), rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        EXPECT_EQ(m_recorder->seen_inputs[r], rows[r]) << "inner saw wrong slice for row " << r;
    }

    // Rows must be distinct (proves no row clobbered another).
    EXPECT_NE(m_recorder->seen_inputs[0], m_recorder->seen_inputs[1]);
}

TEST_F(NPUWBatchedElementTest, SingleRowIsTransparent) {
    auto wrapped =
        ov::npuw::batched::CompiledModel::create(m_model, m_plugin, make_inner(), /*enabled=*/true);
    auto request = wrapped->create_infer_request();

    const std::vector<std::vector<float>> rows = {{5.0f, 6.0f, 7.0f, 8.0f}};
    const auto input = make_input(rows);
    request->set_tensor(wrapped->inputs()[0], ov::get_tensor_impl(input));

    request->infer();

    const auto output = request->get_tensor(wrapped->outputs()[0]);
    ASSERT_EQ(output->get_shape(), (ov::Shape{1, K}));
    const float* out = static_cast<const float*>(output->data());
    for (std::size_t k = 0; k < K; ++k) {
        EXPECT_FLOAT_EQ(out[k], rows[0][k] + ROW_BIAS);
    }
    EXPECT_EQ(m_recorder->infer_count, 1);
}

}  // namespace
