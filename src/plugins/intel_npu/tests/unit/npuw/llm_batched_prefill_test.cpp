// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// End-to-end tests for batched (batch > 1) prefill scoring through the NPUW LLM
// pipeline. The static models are compiled with a batch size of 1, so a batched
// input is unrolled and scored row-by-row inside LLMInferRequest. The logits of
// a batched infer must match the logits of scoring each row separately.
//
// The LLM sub-models (prefill / generate) are compiled and executed on CPU via a
// test factory, so the whole LLMInferRequest data path runs for real.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "llm_compiled_model.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"

namespace {

using ov::test::npuw::NullPlugin;

ov::Core& test_core() {
    static ov::Core core;
    return core;
}

// Sync infer request backed by a real CPU infer request. Tensors are owned by
// this request (the ISyncInferRequest storage) and re-bound to the CPU request
// by name on every infer, so tensor replacements done by the LLM pipeline
// (e.g. KV-cache binding) are picked up.
class CpuSubInferRequest final : public ov::ISyncInferRequest {
public:
    CpuSubInferRequest(std::shared_ptr<const ov::ICompiledModel> compiled_model, const ov::CompiledModel& cpu_compiled)
        : ov::ISyncInferRequest(std::move(compiled_model)),
          m_cpu_request(const_cast<ov::CompiledModel&>(cpu_compiled).create_infer_request()) {
        for (const auto& input : get_inputs()) {
            auto tensor = ov::Tensor(input.get_element_type(), input.get_shape());
            ov::ISyncInferRequest::set_tensor(input, ov::get_tensor_impl(tensor));
        }
        for (const auto& output : get_outputs()) {
            auto tensor = ov::Tensor(output.get_element_type(), output.get_shape());
            ov::ISyncInferRequest::set_tensor(output, ov::get_tensor_impl(tensor));
        }
    }

    void infer() override {
        for (const auto& input : get_inputs()) {
            m_cpu_request.set_tensor(input.get_any_name(), ov::make_tensor(get_tensor(input)));
        }
        for (const auto& output : get_outputs()) {
            m_cpu_request.set_tensor(output.get_any_name(), ov::make_tensor(get_tensor(output)));
        }
        m_cpu_request.infer();
    }

    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

private:
    ov::InferRequest m_cpu_request;
};

// Sub-compiled-model that compiles the transformed static model on CPU.
class CpuSubCompiledModel final : public ov::npuw::ICompiledModel_v0 {
public:
    CpuSubCompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::npuw::ICompiledModel_v0(model, plugin),
          m_model(model),
          m_cpu_compiled(test_core().compile_model(model, "CPU")) {}

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return std::make_shared<CpuSubInferRequest>(shared_from_this(), m_cpu_compiled);
    }

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(), nullptr, nullptr);
    }

    std::shared_ptr<ov::npuw::IBaseInferRequest> create_base_infer_request() const override {
        // Only consumed by the chunked-prefill path, which these tests disable.
        return nullptr;
    }

    std::shared_ptr<ov::IAsyncInferRequest> wrap_async_infer_request(
        std::shared_ptr<ov::npuw::IBaseInferRequest>) const override {
        return create_infer_request();
    }

    std::string submodel_device(std::size_t) const override {
        return "CPU";
    }
    std::size_t num_submodels() const override {
        return 1;
    }

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return std::string{};
    }
    std::shared_ptr<ov::npuw::weights::Bank> get_weights_bank() const override {
        return {};
    }
    void set_weights_bank(std::shared_ptr<ov::npuw::weights::Bank>) override {}
    void finalize_weights_bank() override {}
    void reconstruct_closure() override {}
    void serialize(std::ostream&, const ov::npuw::s11n::CompiledContext&) const override {}

private:
    std::shared_ptr<ov::Model> m_model;
    ov::CompiledModel m_cpu_compiled;
};

class NPUWBatchedPrefillTest : public ::testing::Test {
protected:
    static constexpr int64_t kVocabSize = 256;

    std::shared_ptr<ov::npuw::LLMCompiledModel> compile_llm() {
        auto model = ov::test::npuw::build_llm_test_model();
        auto plugin = std::make_shared<NullPlugin>();
        const ov::AnyMap props = {{"NPUW_LLM", "YES"},
                                  {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                                  {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                                  {"NPUW_LLM_SHARED_HEAD", "NO"},
                                  {"NPUW_LLM_PREFILL_HINT", "STATIC"}};
        const auto factory = [](const std::shared_ptr<ov::Model>& submodel,
                                const std::shared_ptr<const ov::IPlugin>& subplugin,
                                const ov::AnyMap&) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            return std::make_shared<CpuSubCompiledModel>(submodel, subplugin);
        };
        return std::make_shared<ov::npuw::LLMCompiledModel>(model, plugin, props, factory);
    }

    static ov::Output<const ov::Node> port_by_name(const std::vector<ov::Output<const ov::Node>>& ports,
                                                   const std::string& name) {
        for (const auto& port : ports) {
            if (port.get_names().count(name) > 0) {
                return port;
            }
        }
        OPENVINO_THROW("Port not found: ", name);
    }

    // Run one infer for the given rows of token ids; returns a copy of the logits.
    static ov::Tensor run_prefill(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                  const std::vector<std::vector<int64_t>>& rows) {
        const size_t batch = rows.size();
        const size_t seq_len = rows.front().size();

        ov::Tensor input_ids(ov::element::i64, {batch, seq_len});
        ov::Tensor attention_mask(ov::element::i64, {batch, seq_len});
        ov::Tensor position_ids(ov::element::i64, {batch, seq_len});
        for (size_t b = 0; b < batch; ++b) {
            std::copy(rows[b].begin(), rows[b].end(), input_ids.data<int64_t>() + b * seq_len);
            std::fill_n(attention_mask.data<int64_t>() + b * seq_len, seq_len, 1);
            std::iota(position_ids.data<int64_t>() + b * seq_len, position_ids.data<int64_t>() + (b + 1) * seq_len, 0);
        }

        const auto& inputs = request->get_inputs();
        request->set_tensor(port_by_name(inputs, "input_ids"), ov::get_tensor_impl(input_ids));
        request->set_tensor(port_by_name(inputs, "attention_mask"), ov::get_tensor_impl(attention_mask));
        request->set_tensor(port_by_name(inputs, "position_ids"), ov::get_tensor_impl(position_ids));

        request->infer();

        auto logits = request->get_tensor(port_by_name(request->get_outputs(), "logits"));
        ov::Tensor copy(logits->get_element_type(), logits->get_shape());
        ov::make_tensor(logits).copy_to(copy);
        return copy;
    }

    static void expect_rows_equal(const ov::Tensor& batched, size_t row, const ov::Tensor& single) {
        ASSERT_EQ(single.get_shape()[0], 1u);
        const size_t row_elems = single.get_size();
        ASSERT_EQ(batched.get_size(), batched.get_shape()[0] * row_elems);

        const float* batched_data = batched.data<float>() + row * row_elems;
        const float* single_data = single.data<float>();
        for (size_t i = 0; i < row_elems; ++i) {
            ASSERT_FLOAT_EQ(batched_data[i], single_data[i]) << "row " << row << ", element " << i;
        }
    }
};

TEST_F(NPUWBatchedPrefillTest, BatchedPrefillMatchesPerRowScoring) {
    auto compiled = compile_llm();
    auto request = compiled->create_infer_request();
    ASSERT_NE(request, nullptr);

    const std::vector<std::vector<int64_t>> rows = {
        {11, 23, 35, 47, 59, 61},
        {7, 110, 42, 250, 3, 91},
        {200, 1, 199, 2, 198, 3},
    };

    const auto batched_logits = run_prefill(request, rows);
    ASSERT_EQ(batched_logits.get_shape()[0], rows.size());

    for (size_t row = 0; row < rows.size(); ++row) {
        const auto single_logits = run_prefill(request, {rows[row]});
        expect_rows_equal(batched_logits, row, single_logits);
    }
}

TEST_F(NPUWBatchedPrefillTest, BatchedGenerationIsRejected) {
    auto compiled = compile_llm();
    auto request = compiled->create_infer_request();
    ASSERT_NE(request, nullptr);

    // A batched single-token (generation-shaped) input must be rejected.
    const size_t batch = 2;
    ov::Tensor input_ids(ov::element::i64, {batch, 1});
    ov::Tensor attention_mask(ov::element::i64, {batch, 1});
    ov::Tensor position_ids(ov::element::i64, {batch, 1});
    std::fill_n(input_ids.data<int64_t>(), batch, 5);
    std::fill_n(attention_mask.data<int64_t>(), batch, 1);
    std::fill_n(position_ids.data<int64_t>(), batch, 7);

    const auto& inputs = request->get_inputs();
    request->set_tensor(port_by_name(inputs, "input_ids"), ov::get_tensor_impl(input_ids));
    request->set_tensor(port_by_name(inputs, "attention_mask"), ov::get_tensor_impl(attention_mask));
    request->set_tensor(port_by_name(inputs, "position_ids"), ov::get_tensor_impl(position_ids));

    EXPECT_THROW(request->infer(), ov::Exception);
}

}  // namespace
