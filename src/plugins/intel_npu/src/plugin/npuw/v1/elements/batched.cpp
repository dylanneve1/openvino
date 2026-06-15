// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batched.hpp"

#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

// Zero-copy view of one batch row (axis 0) of a tensor: [N, ...] -> [1, ...].
ov::SoPtr<ov::ITensor> row_slice(const ov::SoPtr<ov::ITensor>& tensor, std::size_t row) {
    ov::Shape start_shape(tensor->get_shape().size(), 0u);
    start_shape[0] = row;
    ov::Shape end_shape = tensor->get_shape();
    end_shape[0] = row + 1;
    return ov::get_tensor_impl(ov::Tensor(ov::make_tensor(tensor), start_shape, end_shape));
}

bool has_batch_dim(const ov::SoPtr<ov::ITensor>& tensor) {
    return !tensor->get_shape().empty();
}

}  // namespace

ov::SoPtr<ov::ICompiledModel> ov::npuw::batched::CompiledModel::create(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    ov::SoPtr<ov::ICompiledModel> inner_compiled,
    bool enabled) {
    OPENVINO_ASSERT(inner_compiled._ptr != nullptr, "Batched compiled model requires an inner compiled model");

    // No-op wrapper: hand back the inner model unchanged for the zero-overhead path.
    if (!enabled) {
        return inner_compiled;
    }

    auto compiled_model = std::make_shared<CompiledModel>(model, plugin, std::move(inner_compiled));
    return {compiled_model, {}};
}

ov::npuw::batched::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                const std::shared_ptr<const ov::IPlugin>& plugin,
                                                ov::SoPtr<ov::ICompiledModel> inner_compiled)
    : ov::ICompiledModel(model, plugin),
      m_inner(std::move(inner_compiled)) {
    OPENVINO_ASSERT(m_inner._ptr != nullptr, "Batched compiled model requires an inner compiled model");
}

void ov::npuw::batched::CompiledModel::export_model(std::ostream& model) const {
    m_inner->export_model(model);
}

std::shared_ptr<const ov::Model> ov::npuw::batched::CompiledModel::get_runtime_model() const {
    return m_inner->get_runtime_model();
}

void ov::npuw::batched::CompiledModel::set_property(const ov::AnyMap& properties) {
    m_inner->set_property(properties);
}

ov::Any ov::npuw::batched::CompiledModel::get_property(const std::string& name) const {
    return m_inner->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::batched::CompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const CompiledModel>(shared_from_this());
    return std::make_shared<InferRequest>(std::move(self));
}

std::shared_ptr<ov::IAsyncInferRequest> ov::npuw::batched::CompiledModel::create_infer_request() const {
    return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                    get_task_executor(),
                                                    get_callback_executor());
}

ov::npuw::batched::InferRequest::InferRequest(std::shared_ptr<const CompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled(std::move(compiled_model)) {}

void ov::npuw::batched::InferRequest::ensure_inner_request_locked() const {
    if (m_inner_request == nullptr) {
        m_inner_request = m_compiled->m_inner->create_infer_request();
        OPENVINO_ASSERT(m_inner_request != nullptr, "Batched element: inner compiled model returned a null request");
    }
}

void ov::npuw::batched::InferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();

    const auto& wrapper_inputs = m_compiled->inputs();
    const auto& wrapper_outputs = m_compiled->outputs();
    const auto& inner_inputs = m_compiled->m_inner->inputs();
    const auto& inner_outputs = m_compiled->m_inner->outputs();

    OPENVINO_ASSERT(wrapper_inputs.size() == inner_inputs.size() && wrapper_outputs.size() == inner_outputs.size(),
                    "Batched element: inner model I/O does not match the wrapped model");

    // Batch size is taken from the first input that carries a batch dimension.
    std::size_t batch = 1;
    for (const auto& port : wrapper_inputs) {
        const auto tensor = get_tensor(port);
        if (has_batch_dim(tensor)) {
            batch = tensor->get_shape()[0];
            break;
        }
    }

    // Aggregated [N, ...] outputs, allocated lazily once the per-row output shape is known.
    std::vector<ov::SoPtr<ov::ITensor>> aggregated_outputs(wrapper_outputs.size());

    for (std::size_t row = 0; row < batch; ++row) {
        // Rows are independent prompts: clear the inner request's variable state
        // (KV-cache) so row i never sees row i-1.  Harmless for stateless inners.
        for (auto& state : m_inner_request->query_state()) {
            state->reset();
        }

        // Bind the row's slice of every batched input; pass non-batched (shared)
        // inputs through unchanged.
        for (std::size_t i = 0; i < wrapper_inputs.size(); ++i) {
            const auto full = get_tensor(wrapper_inputs[i]);
            const bool sliceable = batch > 1 && has_batch_dim(full) && full->get_shape()[0] == batch;
            m_inner_request->set_tensor(inner_inputs[i], sliceable ? row_slice(full, row) : full);
        }

        m_inner_request->infer();

        // Stack each per-row output into row i of the aggregated [N, ...] tensor.
        for (std::size_t i = 0; i < wrapper_outputs.size(); ++i) {
            const auto inner_out = m_inner_request->get_tensor(inner_outputs[i]);
            if (!aggregated_outputs[i]) {
                ov::Shape out_shape = inner_out->get_shape();
                if (!out_shape.empty()) {
                    out_shape[0] = batch;
                }
                aggregated_outputs[i] = ov::get_tensor_impl(ov::Tensor(inner_out->get_element_type(), out_shape));
            }
            if (batch > 1 && has_batch_dim(inner_out)) {
                const auto slot = row_slice(aggregated_outputs[i], row);
                inner_out->copy_to(slot._ptr);
            } else {
                inner_out->copy_to(aggregated_outputs[i]._ptr);
            }
        }
    }

    // Publish the aggregated outputs as this request's public output tensors.
    for (std::size_t i = 0; i < wrapper_outputs.size(); ++i) {
        set_tensor(wrapper_outputs[i], aggregated_outputs[i]);
    }
}

void ov::npuw::batched::InferRequest::check_tensors() const {}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::batched::InferRequest::query_state() const {
    // The batched element resets inner state between rows and exposes no
    // cross-call state of its own, so it presents an empty state list.
    return {};
}

std::vector<ov::ProfilingInfo> ov::npuw::batched::InferRequest::get_profiling_info() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    return m_inner_request->get_profiling_info();
}
