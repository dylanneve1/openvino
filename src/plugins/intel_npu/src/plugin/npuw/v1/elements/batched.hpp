// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "../../compiled_model.hpp"
#include "../../perf.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov::npuw::batched {

// True when the compiled model is tagged for batched single-shot scoring
// (NPUW_TEXT_RERANK). The tag lives in the model's own config and survives blob
// serialization, so compile and import decide identically.
bool requested(const std::shared_ptr<ov::npuw::ICompiledModel>& model);

// A transparent decorator that adds batched (batch > 1) execution on top of an
// inner compiled model that only supports batch 1 (as NPUW's LLM pipeline does).
// It exposes the inner model's ports and forwards everything but inference: a
// batched [N, ...] infer is unrolled into N independent batch-1 inferences on
// the inner request, with the variable state (KV-cache) reset between rows and
// the per-row outputs stacked into [N, ...] output tensors.
//
// Valid only for single-shot scoring whose rows are independent (text rerank);
// NOT valid for autoregressive generation, where state must persist across
// calls. Applied via create() at the NPUW entry points - compile and blob
// import - as the element is runtime-only and not part of the blob.
class CompiledModel final : public ov::npuw::ICompiledModel {
public:
    // Returns the inner model wrapped when it is tagged for batched scoring
    // (see requested()), or as-is otherwise.
    static std::shared_ptr<ov::npuw::ICompiledModel> create(const std::shared_ptr<ov::npuw::ICompiledModel>& inner,
                                                            const std::shared_ptr<const ov::IPlugin>& plugin);

    CompiledModel(const std::shared_ptr<ov::npuw::ICompiledModel>& inner,
                  const std::shared_ptr<const ov::IPlugin>& plugin);

    // The wrapper adds no I/O of its own -- it exposes the inner model's ports.
    const std::vector<ov::Output<const ov::Node>>& inputs() const override;
    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    void release_memory() override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    std::shared_ptr<ov::npuw::ICompiledModel> m_inner;
};

// Sync infer request that unrolls a batched inference over the batch-1 inner
// request. The batch size is the largest leading dimension across the inputs;
// an input with a leading dim of 1 is shared across rows. Per row it resets the
// inner state, binds the row's [1, ...] input views, infers, and copies the
// outputs into row i of the [N, ...] output tensors.
class InferRequest final : public ov::ISyncInferRequest {
public:
    InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                 std::shared_ptr<ov::IAsyncInferRequest> inner_request);

    void infer() override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    // The input tensors snapshotted for one infer() call and the batch size
    // derived from them.
    struct BatchedInputs {
        std::vector<ov::SoPtr<ov::ITensor>> tensors;  // parallel to get_inputs()
        std::size_t batch = 1;
    };

    // Snapshot the inputs and derive the batch size. Every input must either
    // carry the batch ([N, ...]) or be shared across rows ([1, ...]); anything
    // else throws.
    BatchedInputs extract_batch() const;

    // Make the public output tensors [batch, ...], reusing caller-bound tensors
    // that already fit. The ports are dynamic, so this can only run once the
    // first row has been scored and the inner output shapes are known.
    void ensure_batched_outputs(std::size_t batch);

    std::shared_ptr<ov::IAsyncInferRequest> m_inner;
    mutable std::mutex m_mutex;

    // Per-phase timings of the unroll.
    using MS = ov::npuw::perf::metric<ov::npuw::perf::MSec>;
    ov::npuw::perf::Profile<MS> m_profile;
};

}  // namespace ov::npuw::batched
