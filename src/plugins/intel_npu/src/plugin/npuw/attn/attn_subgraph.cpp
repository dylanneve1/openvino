// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "attn_subgraph.hpp"

#include <array>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "../attention.hpp"
#include "../compiled_model.hpp"
#include "../host_flash_attention.hpp"
#include "../infer_request_utils.hpp"
#include "../just_sync_infer_request.hpp"
#include "../kv_cache_block_manager.hpp"
#include "../logging.hpp"
#include "../paged_attn_runtime.hpp"
#include "../partitioning/partitioning.hpp"
#include "../partitioning/patterns/paged_attn.hpp"
#include "../partitioning/patterns/sdpa.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "../pyramid_attention.hpp"
#include "../serialization.hpp"

namespace ov {
namespace npuw {
namespace attn {
namespace {

constexpr uint32_t ATTN_KV_DIM = 3;

struct BehaviorIO {
    std::vector<ov::SoPtr<ov::ITensor>> inputs;
    std::vector<ov::SoPtr<ov::ITensor>> outputs;
};

struct PyramidRequestSet {
    std::vector<ov::SoPtr<ov::IAsyncInferRequest>> infer_requests;
    std::vector<ov::SoPtr<ov::IAsyncInferRequest>> pipeline_requests;
    std::vector<ov::SoPtr<ov::ITensor>> anchors;
};

struct HFARequestSet {
    enum TileIdx : std::size_t {
        REGULAR_TILE = 0,
        FINAL_TILE = 1,
        COUNT = 2,
    };

    std::array<ov::SoPtr<ov::IAsyncInferRequest>, COUNT> infer_requests{};
    std::array<ov::SoPtr<ov::IAsyncInferRequest>, COUNT> pipeline_requests{};
};

// Per-subgraph runtime state for the PagedAttention NPU lowering. One
// instance lives in RuntimeState alongside the HFA equivalents. Allocated
// lazily on the first prepare_paged() call.
struct PagedRuntimeState {
    ov::SoPtr<ov::IAsyncInferRequest> tile_request;
    ov::SoPtr<ov::IAsyncInferRequest> final_tile_request;
    // Persistent (acc, max, d) state threaded across tile-loop iterations.
    ov::SoPtr<ov::ITensor> state_acc;
    ov::SoPtr<ov::ITensor> state_max;
    ov::SoPtr<ov::ITensor> state_d;
    // Mask tile buffer, refilled in place per iteration.
    ov::SoPtr<ov::ITensor> mask_tile_buffer;
    bool initialized = false;
};

struct RuntimeState {
    std::unordered_map<std::size_t, BehaviorIO> call_io;
    runtime::attention::Selector::Ptr attention_selector;
    runtime::pyramid_attention::Selector::Ptr pyramid_selector;
    runtime::host_flash_attention::Selector::Ptr hfa_selector;
    ov::SoPtr<ov::ITensor> cached_attention_mask;
    std::optional<runtime::host_flash_attention::HFARuntimeContext> hfa_runtime_ctx;
    PyramidRequestSet pyramid_requests;
    HFARequestSet hfa_requests;
    PagedRuntimeState pa_state;
    ov::SoPtr<ov::IAsyncInferRequest> base_request;
    ov::SoPtr<ov::IAsyncInferRequest> base_pipeline_request;
};

ov::npuw::JustInferRequest& get_request(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto* request = dynamic_cast<ov::npuw::JustInferRequest*>(&ctx.infer_request);
    OPENVINO_ASSERT(request != nullptr, "Expected JustInferRequest for attention runtime behavior");
    return *request;
}

BehaviorKind get_behavior_kind(const ov::npuw::v1::subgraphs::Context& ctx) {
    if (const auto* kind = ctx.get_if<BehaviorKind>()) {
        return *kind;
    }
    OPENVINO_THROW("Attention behavior context is missing BehaviorKind");
}

template <typename T>
T* get_compiled_state(ov::npuw::v1::subgraphs::Context& context) {
    auto* state = context.get_if<std::shared_ptr<T>>();
    return state == nullptr ? nullptr : state->get();
}

template <typename T>
const T* get_compiled_state(const ov::npuw::v1::subgraphs::Context& context) {
    auto* state = context.get_if<std::shared_ptr<T>>();
    return state == nullptr ? nullptr : state->get();
}

const ov::npuw::v1::subgraphs::CompiledPipeline& get_subgraph_pipeline(ov::npuw::v1::subgraphs::InferContext& ctx,
                                                                       std::size_t real_idx) {
    return get_request(ctx).subgraph_pipeline(real_idx);
}

const ov::SoPtr<ov::ICompiledModel>& get_compiled_submodel(ov::npuw::v1::subgraphs::InferContext& ctx,
                                                           std::size_t real_idx) {
    return get_request(ctx).compiled_submodel(real_idx);
}

std::size_t get_param_base(ov::npuw::v1::subgraphs::InferContext& ctx, std::size_t real_idx) {
    return get_request(ctx).subgraph_param_base(real_idx);
}

RuntimeState& get_runtime_state(ov::npuw::v1::subgraphs::InferContext& ctx) {
    OPENVINO_ASSERT(ctx.runtime_state != nullptr, "Expected runtime state storage for attention behavior");
    auto* state = ctx.runtime_state->get_if<RuntimeState>();
    if (state == nullptr) {
        state = &ctx.runtime_state->emplace<RuntimeState>();
    }
    return *state;
}

BehaviorIO& get_behavior_io(RuntimeState& state,
                            std::size_t subgraph_idx,
                            std::size_t num_inputs,
                            std::size_t num_outputs) {
    auto& io = state.call_io[subgraph_idx];
    if (io.inputs.size() != num_inputs) {
        io.inputs.resize(num_inputs);
    }
    if (io.outputs.size() != num_outputs) {
        io.outputs.resize(num_outputs);
    }
    return io;
}

void ensure_base_requests(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    auto& request = get_request(ctx);
    if (!state.base_request) {
        state.base_request = request.get_subrequest(ctx.real_subgraph_idx);
    }
    if (request.is_subrequest_pipelined(ctx.real_subgraph_idx) && !state.base_pipeline_request) {
        state.base_pipeline_request = request.get_pipeline_subrequest(ctx.real_subgraph_idx);
    }
}

void ensure_dynamic_selector(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (state.attention_selector) {
        return;
    }
    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* dynamic = ov::npuw::attn::get_compiled_dynamic(pipeline.context);
    OPENVINO_ASSERT(dynamic != nullptr, "Missing compiled dynamic attention state");

    auto& request = get_request(ctx);
    if (!ctx.compiled_model.attention_dynamic_enabled()) {
        state.attention_selector = std::make_shared<runtime::attention::All>();
        return;
    }

    state.attention_selector = runtime::attention::PositionIDs::find(*dynamic, request);
    if (!state.attention_selector) {
        LOG_WARN("Dynamic capability is enabled, but no run-time features were found.");
        state.attention_selector = std::make_shared<runtime::attention::All>();
    }
}

void ensure_pyramid_selector(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (state.pyramid_selector) {
        return;
    }
    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* pyramid = ov::npuw::attn::get_compiled_pyramid(pipeline.context);
    OPENVINO_ASSERT(pyramid != nullptr, "Missing compiled pyramid attention state");

    auto& request = get_request(ctx);
    const auto pyramid_count = pyramid->_compiled_models.size();
    if (!ctx.compiled_model.attention_dynamic_enabled()) {
        state.pyramid_selector.reset(new runtime::pyramid_attention::All(pyramid_count));
        return;
    }

    state.pyramid_selector = runtime::pyramid_attention::PositionIDs::find(*pyramid, request);
    if (!state.pyramid_selector) {
        LOG_WARN("Pyramid dynamic capability is enabled, but no run-time features were found.");
        state.pyramid_selector.reset(new runtime::pyramid_attention::All(pyramid_count));
    }
}

void ensure_hfa_selector(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (state.hfa_selector) {
        return;
    }
    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* hfa = ov::npuw::attn::get_compiled_hfa(pipeline.context);
    OPENVINO_ASSERT(hfa != nullptr, "Missing compiled HFA state");

    auto& request = get_request(ctx);
    const size_t query_size = hfa->_sdpa_attention_info._query_size;
    state.hfa_selector = runtime::host_flash_attention::PositionIDs::find(query_size, request);
    if (!state.hfa_selector) {
        OPENVINO_THROW("HFA dynamic capability is enabled, but no run-time features were found.");
    }
}

void ensure_pyramid_requests(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (!state.pyramid_requests.infer_requests.empty()) {
        return;
    }

    ensure_base_requests(ctx, state);

    const auto& compiled_model = get_compiled_submodel(ctx, ctx.real_subgraph_idx);
    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* pyramid = ov::npuw::attn::get_compiled_pyramid(pipeline.context);
    OPENVINO_ASSERT(pyramid != nullptr, "Missing compiled pyramid attention state");

    auto& request = get_request(ctx);
    const auto& pyramid_models = pyramid->_compiled_models;
    const size_t num_pyramid_models = pyramid_models.size();
    const bool is_piped = request.is_subrequest_pipelined(ctx.real_subgraph_idx);

    state.pyramid_requests.infer_requests.resize(num_pyramid_models);
    if (is_piped) {
        state.pyramid_requests.pipeline_requests.resize(num_pyramid_models);
    }

    const auto& main_inputs = compiled_model->inputs();
    const bool block_mode = pyramid->is_block_mode();

    // Block KV: alias all block slots on the main request to the same buffer (block_0).
    // bind_block_ports() in prologue() will set the real per-block tensors at inference time.
    if (block_mode) {
        auto alias_block_slots = [&](ov::SoPtr<ov::IAsyncInferRequest>& req, size_t num_blocks, auto get_global_idx) {
            if (num_blocks == 0) {
                return;
            }
            auto block0 = req->get_tensor(main_inputs[get_global_idx(0)]);
            for (size_t i = 1; i < num_blocks; ++i) {
                req->set_tensor(main_inputs[get_global_idx(i)], block0);
            }
        };
        const size_t nk = pyramid->num_key_blocks_global();
        alias_block_slots(state.base_request, nk, [&](size_t i) {
            return pyramid->key_block_global_at(i);
        });
        alias_block_slots(state.base_request, nk, [&](size_t i) {
            return pyramid->val_block_global_at(i);
        });
        if (is_piped) {
            alias_block_slots(state.base_pipeline_request, nk, [&](size_t i) {
                return pyramid->key_block_global_at(i);
            });
            alias_block_slots(state.base_pipeline_request, nk, [&](size_t i) {
                return pyramid->val_block_global_at(i);
            });
        }
    }

    // Build a name→main-port map once for sharing non-block inputs into pyramid variants.
    std::unordered_map<std::string, ov::Output<const ov::Node>> main_name_to_port;
    main_name_to_port.reserve(main_inputs.size());
    for (const auto& inp : main_inputs) {
        if (!inp.get_names().empty()) {
            main_name_to_port[inp.get_any_name()] = inp;
        }
    }

    for (size_t model_idx = 0; model_idx + 1 < num_pyramid_models; ++model_idx) {
        state.pyramid_requests.infer_requests[model_idx] = pyramid_models[model_idx]->create_infer_request();
        if (is_piped) {
            state.pyramid_requests.pipeline_requests[model_idx] = pyramid_models[model_idx]->create_infer_request();
        }

        auto share_inputs_for_req = [&](ov::SoPtr<ov::IAsyncInferRequest>& req,
                                        ov::SoPtr<ov::IAsyncInferRequest>& main_req) {
            const auto& variant_inputs = pyramid_models[model_idx]->inputs();
            const auto& key_block_ports = pyramid->key_block_port_set_at(model_idx);
            const auto& val_block_ports = pyramid->val_block_port_set_at(model_idx);
            ov::SoPtr<ov::ITensor> key_block0, val_block0;
            if (block_mode) {
                key_block0 = main_req->get_tensor(main_inputs[pyramid->key_block_global_at(0)]);
                val_block0 = main_req->get_tensor(main_inputs[pyramid->val_block_global_at(0)]);
            }

            for (size_t j = 0; j < variant_inputs.size(); ++j) {
                if (block_mode && key_block_ports.count(j)) {
                    req->set_tensor(variant_inputs[j], key_block0);
                } else if (block_mode && val_block_ports.count(j)) {
                    req->set_tensor(variant_inputs[j], val_block0);
                } else {
                    if (variant_inputs[j].get_names().empty()) {
                        continue;
                    }
                    const auto it = main_name_to_port.find(variant_inputs[j].get_any_name());
                    if (it == main_name_to_port.end()) {
                        continue;
                    }
                    auto main_tensor = main_req->get_tensor(it->second);
                    auto pyr_tensor = req->get_tensor(variant_inputs[j]);
                    auto shared = ov::get_tensor_impl(
                        ov::Tensor(pyr_tensor->get_element_type(), pyr_tensor->get_shape(), main_tensor->data()));
                    req->set_tensor(variant_inputs[j], shared);
                }
            }
        };

        share_inputs_for_req(state.pyramid_requests.infer_requests[model_idx], state.base_request);
        if (is_piped) {
            share_inputs_for_req(state.pyramid_requests.pipeline_requests[model_idx], state.base_pipeline_request);
        }
    }

    if (num_pyramid_models > 0) {
        const size_t last_model_idx = num_pyramid_models - 1;
        state.pyramid_requests.infer_requests[last_model_idx] = state.base_request;
        if (is_piped) {
            state.pyramid_requests.pipeline_requests[last_model_idx] = state.base_pipeline_request;
        }

        const size_t num_inputs = compiled_model->inputs().size();
        for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            const auto main_input = compiled_model->inputs()[input_idx];
            state.pyramid_requests.anchors.push_back(state.base_request->get_tensor(main_input));
            if (is_piped) {
                state.pyramid_requests.anchors.push_back(state.base_pipeline_request->get_tensor(main_input));
            }
        }
    }
}

void ensure_hfa_requests(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]) {
        return;
    }

    ensure_base_requests(ctx, state);

    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    const auto* hfa = ov::npuw::attn::get_compiled_hfa(pipeline.context);
    OPENVINO_ASSERT(hfa != nullptr, "Missing compiled HFA state");

    auto& request = get_request(ctx);
    const bool is_piped = request.is_subrequest_pipelined(ctx.real_subgraph_idx);

    state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE] = hfa->_compiled_tile_model->create_infer_request();
    state.hfa_requests.infer_requests[HFARequestSet::FINAL_TILE] = state.base_request;
    if (is_piped) {
        state.hfa_requests.pipeline_requests[HFARequestSet::REGULAR_TILE] =
            hfa->_compiled_tile_model->create_infer_request();
        state.hfa_requests.pipeline_requests[HFARequestSet::FINAL_TILE] = state.base_pipeline_request;
    }

    const size_t num_inputs = hfa->_compiled_tile_model->inputs().size();
    for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
        const auto tile_input = hfa->_compiled_tile_model->inputs()[input_idx];
        const auto final_tile_input = hfa->_compiled_final_tile_model->inputs()[input_idx];

        auto main_tensor = state.base_request->get_tensor(final_tile_input);
        state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]->set_tensor(tile_input, main_tensor);

        if (is_piped) {
            auto pipeline_tensor = state.base_pipeline_request->get_tensor(final_tile_input);
            state.hfa_requests.pipeline_requests[HFARequestSet::REGULAR_TILE]->set_tensor(tile_input, pipeline_tensor);
        }
    }

    if (!state.hfa_runtime_ctx.has_value()) {
        state.hfa_runtime_ctx.emplace();
    }

    state.hfa_runtime_ctx->initialize_mask_cache(
        *hfa,
        request.subgraph_device(ctx.real_subgraph_idx),
        [&request](const ov::element::Type& dtype, const ov::Shape& shape, const std::string& device) {
            return request.allocate_mem(dtype, shape, device);
        });

    const auto& tile_in = hfa->_sdpa_attention_info._tile_input_indices;
    auto state_acc = state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]->get_tensor(
        hfa->_compiled_tile_model->inputs()[tile_in.acc]);
    auto state_max = state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]->get_tensor(
        hfa->_compiled_tile_model->inputs()[tile_in.max]);
    auto state_sum = state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE]->get_tensor(
        hfa->_compiled_tile_model->inputs()[tile_in.d]);

    runtime::host_flash_attention::HFARuntimeContext::initialize_state_tensors(state_acc, state_max, state_sum);
    runtime::host_flash_attention::HFARuntimeContext::StateBuffers initial_buffers{state_acc, state_max, state_sum};
    state.hfa_runtime_ctx->initialize_state_buffers(
        initial_buffers,
        *hfa,
        request.subgraph_device(ctx.real_subgraph_idx),
        [&request](const ov::element::Type& dtype, const ov::Shape& shape, const std::string& device) {
            return request.allocate_mem(dtype, shape, device);
        });
}

void prepare_dynamic(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto& state = get_runtime_state(ctx);
    ensure_dynamic_selector(ctx, state);
    state.attention_selector->prepare(get_request(ctx).history_size());
    state.cached_attention_mask = {};
}

void prepare_pyramid(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto& request = get_request(ctx);
    auto& state = get_runtime_state(ctx);
    ensure_pyramid_selector(ctx, state);
    ensure_pyramid_requests(ctx, state);
    state.pyramid_selector->prepare(request.history_size());
    state.cached_attention_mask = {};

    const auto pyramid_id = state.pyramid_selector->pyramid_id();
    request.set_active_subrequest(ctx.real_subgraph_idx, state.pyramid_requests.infer_requests.at(pyramid_id));
    if (request.is_subrequest_pipelined(ctx.real_subgraph_idx)) {
        request.set_pipeline_subrequest(ctx.real_subgraph_idx, state.pyramid_requests.pipeline_requests.at(pyramid_id));
    }
}

void prepare_hfa(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto& request = get_request(ctx);
    auto& state = get_runtime_state(ctx);
    ensure_hfa_selector(ctx, state);
    ensure_hfa_requests(ctx, state);
    state.hfa_selector->prepare(request.history_size());
    state.cached_attention_mask = {};
    if (state.hfa_runtime_ctx) {
        state.hfa_runtime_ctx->clear_mask_cache();
    }
    request.set_active_subrequest(ctx.real_subgraph_idx, state.base_request);
    if (request.is_subrequest_pipelined(ctx.real_subgraph_idx)) {
        request.set_pipeline_subrequest(ctx.real_subgraph_idx, state.base_pipeline_request);
    }
}

// Fill the persistent (acc, max, d) state tensors for a fresh tile-loop
// pass. acc=0, d=0, max=most-negative-finite. Mirrors the math in
// runtime::host_flash_attention::HFARuntimeContext::initialize_state_tensors
// but operates on the PA state buffers, which carry the same dtype/shape
// contract via OnlineSoftmaxTileInputId::PAST_*.
void initialize_pa_state(PagedRuntimeState& pa_state) {
    if (pa_state.state_acc) {
        ov::npuw::util::fill_tensor_bytes(pa_state.state_acc, 0u);
    }
    if (pa_state.state_d) {
        ov::npuw::util::fill_tensor_bytes(pa_state.state_d, 0u);
    }
    if (!pa_state.state_max) {
        return;
    }
    const auto et = pa_state.state_max->get_element_type();
    const auto n = pa_state.state_max->get_size();
    if (et == ov::element::f32) {
        auto* p = pa_state.state_max->data<float>();
        const float v = std::numeric_limits<float>::lowest();
        for (std::size_t i = 0; i < n; ++i) p[i] = v;
    } else if (et == ov::element::f16) {
        // -65504 (most-negative finite f16), bit pattern 0xFBFF
        auto* p = reinterpret_cast<uint16_t*>(pa_state.state_max->data());
        for (std::size_t i = 0; i < n; ++i) p[i] = 0xFBFF;
    } else {
        OPENVINO_THROW("PA tile loop: unsupported PAST_MAX element type ", et.get_type_name());
    }
}

// Materialize the mask tile [batch, 1, query_size, block_size] for one
// iteration of the cached-block loop. Slot k in logical block block_idx
// is valid (mask=0) iff its absolute key position (block_idx*block_size+k)
// is < past_lens (cache fill). Otherwise -inf. No per-query variation —
// every query in this step is, by construction, newer than any cached
// position, so the mask is purely a validity gate.
void fill_pa_mask_for_cached_block(const ov::SoPtr<ov::ITensor>& mask_tile,
                                   uint32_t past_lens,
                                   uint32_t block_idx) {
    const auto shape = mask_tile->get_shape();
    NPUW_ASSERT(shape.size() == 4);
    const auto query_size = shape[2];
    const auto block_size = shape[3];
    const uint64_t k_abs_start = static_cast<uint64_t>(block_idx) * static_cast<uint64_t>(block_size);
    const auto et = mask_tile->get_element_type();
    if (et == ov::element::f32) {
        auto* data = mask_tile->data<float>();
        const float neg = std::numeric_limits<float>::lowest();
        for (std::size_t q = 0; q < query_size; ++q) {
            for (std::size_t k = 0; k < block_size; ++k) {
                data[q * block_size + k] = ((k_abs_start + k) < past_lens) ? 0.0f : neg;
            }
        }
    } else if (et == ov::element::f16) {
        auto* data = reinterpret_cast<uint16_t*>(mask_tile->data());
        for (std::size_t q = 0; q < query_size; ++q) {
            for (std::size_t k = 0; k < block_size; ++k) {
                data[q * block_size + k] = ((k_abs_start + k) < past_lens) ? uint16_t{0x0000} : uint16_t{0xFBFF};
            }
        }
    } else {
        OPENVINO_THROW("PA tile loop: unsupported MASK_TILE element type ", et.get_type_name());
    }
}

// Materialize the mask tile for the FINAL iteration that attends over the
// new tokens' K/V. Strictly causal within the new-token window:
//   mask[q, k] = 0     iff k <= q && k < new_tokens
//   mask[q, k] = -inf  otherwise
void fill_pa_mask_for_present(const ov::SoPtr<ov::ITensor>& mask_tile, uint32_t new_tokens) {
    const auto shape = mask_tile->get_shape();
    NPUW_ASSERT(shape.size() == 4);
    const auto query_size = shape[2];
    const auto block_size = shape[3];
    const auto et = mask_tile->get_element_type();
    if (et == ov::element::f32) {
        auto* data = mask_tile->data<float>();
        const float neg = std::numeric_limits<float>::lowest();
        for (std::size_t q = 0; q < query_size; ++q) {
            for (std::size_t k = 0; k < block_size; ++k) {
                const bool ok = (k <= q) && (k < new_tokens);
                data[q * block_size + k] = ok ? 0.0f : neg;
            }
        }
    } else if (et == ov::element::f16) {
        auto* data = reinterpret_cast<uint16_t*>(mask_tile->data());
        for (std::size_t q = 0; q < query_size; ++q) {
            for (std::size_t k = 0; k < block_size; ++k) {
                const bool ok = (k <= q) && (k < new_tokens);
                data[q * block_size + k] = ok ? uint16_t{0x0000} : uint16_t{0xFBFF};
            }
        }
    } else {
        OPENVINO_THROW("PA tile loop: unsupported MASK_TILE element type ", et.get_type_name());
    }
}

void ensure_paged_requests(ov::npuw::v1::subgraphs::InferContext& ctx, RuntimeState& state) {
    if (state.pa_state.initialized) {
        return;
    }

    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
    auto* pa = ov::npuw::attn::get_compiled_pa(pipeline.context);
    OPENVINO_ASSERT(pa != nullptr,
                    "ensure_paged_requests: missing compiled PA state in subgraph context");

    // The partition_stage hook constructs compiled::PagedAttention without
    // tile-model handles (the partitioner has no back-reference to
    // LLMCompiledModel where setup_paged_runtime() stored the compiled
    // tile sub-models). Real handle resolution is left as a follow-up to
    // be done together with the NPUW runtime team — see open question #6
    // in PAGED_ATTN_LOWERING_DESIGN.md ("compiled-handle propagation").
    OPENVINO_ASSERT(pa->is_valid(),
                    "ensure_paged_requests: PagedAttention compiled tile handles are NULL. "
                    "The partitioner -> attn_subgraph compiled-handle bridge is not yet wired; "
                    "set NPUW_ATTN_PAGED_NPU_LOWERING=NO (default) to use the CPU PA fallback.");

    state.pa_state.tile_request = pa->_compiled_tile_model->create_infer_request();
    state.pa_state.final_tile_request = pa->_compiled_final_tile_model->create_infer_request();

    const auto port_of = [&](OnlineSoftmaxTileInputId id) {
        return state.pa_state.tile_request->get_compiled_model()->inputs().at(static_cast<std::size_t>(id));
    };
    const auto acc_port = port_of(OnlineSoftmaxTileInputId::PAST_ACC);
    const auto max_port = port_of(OnlineSoftmaxTileInputId::PAST_MAX);
    const auto d_port = port_of(OnlineSoftmaxTileInputId::PAST_D);
    const auto mask_port = port_of(OnlineSoftmaxTileInputId::MASK_TILE);

    state.pa_state.state_acc =
        ov::SoPtr<ov::ITensor>(ov::make_tensor(acc_port.get_element_type(), acc_port.get_shape()), nullptr);
    state.pa_state.state_max =
        ov::SoPtr<ov::ITensor>(ov::make_tensor(max_port.get_element_type(), max_port.get_shape()), nullptr);
    state.pa_state.state_d =
        ov::SoPtr<ov::ITensor>(ov::make_tensor(d_port.get_element_type(), d_port.get_shape()), nullptr);
    state.pa_state.mask_tile_buffer =
        ov::SoPtr<ov::ITensor>(ov::make_tensor(mask_port.get_element_type(), mask_port.get_shape()), nullptr);

    state.pa_state.initialized = true;
}

void prepare_paged(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto& request = get_request(ctx);
    auto& state = get_runtime_state(ctx);
    ensure_base_requests(ctx, state);
    ensure_paged_requests(ctx, state);
    state.cached_attention_mask = {};
    request.set_active_subrequest(ctx.real_subgraph_idx, state.base_request);
    if (request.is_subrequest_pipelined(ctx.real_subgraph_idx)) {
        request.set_pipeline_subrequest(ctx.real_subgraph_idx, state.base_pipeline_request);
    }
}

void extract_and_copy_tile(const ov::SoPtr<ov::ITensor>& source_tensor,
                           const ov::SoPtr<ov::ITensor>& dest_tensor,
                           uint32_t sequence_dim,
                           int64_t sequence_offset,
                           int64_t sequence_length,
                           const std::string& tensor_name) {
    if (!dest_tensor->is_continuous()) {
        OPENVINO_THROW("HFA tile extraction error: destination tensor for '",
                       tensor_name,
                       "' is not continuous - cannot perform direct copy");
    }

    auto source_view = ov::npuw::util::view(source_tensor, sequence_dim, sequence_offset, sequence_length);
    const auto dest_type = dest_tensor->get_element_type();
    const auto source_type = source_tensor->get_element_type();

    if (dest_type == source_type) {
        ov::npuw::util::copy_tensor_by_dim(source_view, dest_tensor, sequence_dim, sequence_dim);
        return;
    }

    LOG_WARN("Performing type conversion for " << tensor_name << " tile: " << source_type << " -> " << dest_type);
    auto intermediate_tensor = ov::Tensor(source_type, source_view->get_shape());
    ov::npuw::util::copy_tensor_by_dim(source_view,
                                       ov::get_tensor_impl(intermediate_tensor),
                                       sequence_dim,
                                       sequence_dim);

    const size_t total_elements = intermediate_tensor.get_size();
    if (dest_type == ov::element::f32 && source_type == ov::element::f16) {
        auto src_data = intermediate_tensor.data<ov::float16>();
        auto dst_data = dest_tensor->data<float>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
        return;
    }
    if (dest_type == ov::element::f16 && source_type == ov::element::f32) {
        auto src_data = intermediate_tensor.data<float>();
        auto dst_data = dest_tensor->data<ov::float16>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<ov::float16>(src_data[i]);
        }
        return;
    }
    OPENVINO_THROW("Unsupported type conversion for ", tensor_name, " tile: ", source_type, " -> ", dest_type);
}

bool can_reuse_tensor_zero_copy(const ov::SoPtr<ov::ITensor>& source_tensor,
                                const ov::SoPtr<ov::ITensor>& dest_tensor,
                                uint32_t sequence_dim,
                                int64_t sequence_offset,
                                int64_t tile_length) {
    const auto source_shape = source_tensor->get_shape();
    const int64_t source_full_length = static_cast<int64_t>(source_shape[sequence_dim]);
    return (sequence_offset == 0 && tile_length == source_full_length &&
            dest_tensor->get_element_type() == source_tensor->get_element_type());
}

ov::npuw::v1::subgraphs::RuntimeBehaviorFactory make_runtime_factory() {
    return [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
        class AttnBehavior final : public ov::npuw::v1::subgraphs::ISubgraphBehavior {
        public:
            explicit AttnBehavior(BehaviorKind kind) : m_kind(kind) {}

            void prepare(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                    prepare_dynamic(ctx);
                    return;
                case BehaviorKind::Pyramid:
                    prepare_pyramid(ctx);
                    return;
                case BehaviorKind::HFA:
                    prepare_hfa(ctx);
                    return;
                case BehaviorKind::Paged:
                    prepare_paged(ctx);
                    return;
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

            bool bind_function_input(ov::npuw::v1::subgraphs::InferContext& ctx,
                                     std::size_t input_idx,
                                     const ov::SoPtr<ov::ITensor>& tensor) override {
                auto& state = get_runtime_state(ctx);
                const auto& compiled_model = get_compiled_submodel(ctx, ctx.real_subgraph_idx);
                const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                    if (const auto* dynamic = ov::npuw::attn::get_compiled_dynamic(pipeline.context)) {
                        // Block-split KV cache (SplitKVCacheIntoBlocks) is incompatible with this path
                        // because it replaces the single past-KV parameter with N block parameters and
                        // leaves dynamic->params empty.  Block mode is only supported with Pyramid/HFA.
                        NPUW_ASSERT(!dynamic->params.empty() &&
                                    "Dynamic attention does not support block-based KV cache. "
                                    "Use Pyramid or HFA attention with NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE.");
                        auto& io =
                            get_behavior_io(state, ctx.subgraph_idx, get_param_base(ctx, ctx.real_subgraph_idx), 0u);
                        ensure_dynamic_selector(ctx, state);
                        auto kv_param_it =
                            std::find_if(dynamic->params.begin(), dynamic->params.end(), [&](const auto& p) {
                                return p.idx == input_idx;
                            });
                        const bool is_kv_param = (kv_param_it != dynamic->params.end());
                        const bool is_mask = (input_idx == dynamic->mask_idx);
                        const auto& iport = compiled_model->inputs()[input_idx];
                        if (is_mask) {
                            // Mask requires context-dependent construction — defer to prologue()
                            io.inputs.at(input_idx) = tensor;
                        } else if (is_kv_param) {
                            const auto& param = *kv_param_it;
                            const auto pos_id = state.attention_selector->length();
                            if (pos_id == -1) {
                                // Fallback: dynamic range not identified — bind full tensor
                                ctx.target_request->set_tensor(iport, tensor);
                            } else {
                                const auto past_len = state.attention_selector->past_length();
                                const auto& view = ov::npuw::util::view(tensor, param.dim, 0, past_len);
                                const auto& shape = view->get_shape();
                                const bool do_copy = get_request(ctx).subgraph_needs_copy(ctx.subgraph_idx) &&
                                                     !get_request(ctx).attention_no_copy();
                                if (do_copy && ov::shape_size(shape) > 0) {
                                    const auto& dst = ctx.target_request->get_tensor(iport);
                                    dst->set_shape(shape);
                                    view->copy_to(dst._ptr);
                                } else if (do_copy && ov::shape_size(shape) == 0) {
                                    ctx.target_request->get_tensor(iport)->set_shape(shape);
                                } else {
                                    ctx.target_request->set_tensor(iport, view);
                                }
                            }
                        } else {
                            // Non-KV, non-mask: bind directly
                            ctx.target_request->set_tensor(iport, tensor);
                        }
                        return true;
                    }
                    return false;
                case BehaviorKind::Pyramid:
                    if (const auto* pyramid = ov::npuw::attn::get_compiled_pyramid(pipeline.context)) {
                        auto& io =
                            get_behavior_io(state, ctx.subgraph_idx, get_param_base(ctx, ctx.real_subgraph_idx), 0u);
                        ensure_pyramid_selector(ctx, state);
                        const auto pyramid_id = state.pyramid_selector->pyramid_id();
                        const std::size_t info_mask_idx = pyramid->mask_idx_at(pyramid_id);
                        const bool is_mask = (input_idx == info_mask_idx);
                        const auto& iport = compiled_model->inputs()[input_idx];
                        if (pyramid->is_block_mode()) {
                            // Block KV mode: bind block tensors directly to variant ports;
                            // mask and other inputs fall through to the non-KV handling below.
                            using namespace ov::npuw::runtime;
                            NPUW_ASSERT(state.pyramid_selector->this_case() ==
                                            pyramid_attention::Selector::Case::PREFILL &&
                                        "Pyramid block KV cache is only supported in PREFILL. "
                                        "For GENERATE use NPUW_LLM_GENERATE_PYRAMID=YES.");
                            auto try_bind_block = [&](const std::unordered_map<size_t, size_t>& port_map) -> bool {
                                auto it = port_map.find(input_idx);
                                if (it == port_map.end()) {
                                    return false;  // Not a block KV input.
                                }
                                // It is a block KV input. Check if this variant has a port for it.
                                // std::numeric_limits<size_t>::max() means no port (e.g. model[0]).
                                if (it->second == std::numeric_limits<size_t>::max()) {
                                    return true;  // Consume silently — no set_tensor needed.
                                }
                                const auto& block_iport = pyramid->_compiled_models[pyramid_id]->inputs()[it->second];
                                ctx.target_request->set_tensor(block_iport, tensor);
                                return true;
                            };
                            if (try_bind_block(pyramid->key_block_port_map_at(pyramid_id)) ||
                                try_bind_block(pyramid->val_block_port_map_at(pyramid_id))) {
                                return true;  // Block KV input handled.
                            }
                            // Not a block KV input: mask is deferred to prologue(), others bind directly.
                            if (is_mask) {
                                io.inputs.at(input_idx) = tensor;
                            } else {
                                ctx.target_request->set_tensor(iport, tensor);
                            }
                        } else {
                            // Contiguous KV mode: look up whether input_idx is a KV param
                            // and retrieve its sequence dimension via the virtual interface.
                            const auto dim_opt = pyramid->kv_param_dim(pyramid_id, input_idx);
                            const bool is_kv_param = dim_opt.has_value();
                            if (is_mask) {
                                // Mask requires context-dependent construction — defer to prologue()
                                io.inputs.at(input_idx) = tensor;
                            } else if (is_kv_param) {
                                const auto dim = dim_opt.value();
                                using namespace ov::npuw::runtime;
                                // iport comes from the main compiled model (full KV shape).
                                // For set_tensor we must use the port from the pyramid model's compiled
                                // model so the zero backend shape check passes. For the last pyramid model
                                // _compiled_models[pyramid_id] == the main compiled model, so pyramid_iport
                                // == iport and there is no behavioural difference.
                                // get_tensor calls below intentionally keep iport: they look up by index /
                                // name without strict shape validation.
                                const auto& pyramid_iport = pyramid->_compiled_models[pyramid_id]->inputs()[input_idx];
                                if (state.pyramid_selector->length() == -1) {
                                    // Fallback: dynamic range not identified — bind directly
                                    ctx.target_request->set_tensor(pyramid_iport, tensor);
                                } else {
                                    // Pyramid dynamic range identified
                                    const auto past_len = state.pyramid_selector->past_length();
                                    const auto this_case = state.pyramid_selector->this_case();

                                    // Strided I/O is available for non-last pyramid models compiled with
                                    // enable_strides_for. The last model reuses the main compiled subgraph (compiled
                                    // without enable_strides_for), so it always falls back to the copy path.
                                    const bool use_tensor_view =
                                        pyramid->_can_use_tensor_view && (pyramid_id < pyramid->num_models() - 1);

                                    const auto& input_shape = tensor->get_shape();
                                    if (this_case == pyramid_attention::Selector::Case::PREFILL) {
                                        if (static_cast<int64_t>(input_shape[dim]) == past_len) {
                                            ctx.target_request->set_tensor(pyramid_iport, tensor);
                                        } else {
                                            const auto& view = ov::npuw::util::view(tensor, dim, 0, past_len);
                                            const auto& shape = view->get_shape();
                                            if (ov::shape_size(shape) == 0) {
                                                ctx.target_request->get_tensor(iport)->set_shape(shape);
                                            } else if (use_tensor_view) {
                                                const auto model_past_len =
                                                    static_cast<int64_t>(pyramid->get_context_length(pyramid_id)) -
                                                    static_cast<int64_t>(pyramid->query_size_at(pyramid_id));
                                                LOG_DEBUG("Use tensor view: past_len=" << past_len << " model_past_len="
                                                                                       << model_past_len);
                                                ctx.target_request->set_tensor(
                                                    pyramid_iport,
                                                    ov::npuw::util::view(tensor, dim, 0, model_past_len));
                                            } else {
                                                const auto& dst = ctx.target_request->get_tensor(iport);
                                                ov::npuw::copy_tensor_by_dim(view,
                                                                                   dst,
                                                                                   static_cast<uint32_t>(dim),
                                                                                   static_cast<uint32_t>(dim));
                                            }
                                        }
                                    } else {
                                        NPUW_ASSERT(this_case == pyramid_attention::Selector::Case::GENERATE);
                                        NPUW_ASSERT(static_cast<int64_t>(input_shape[dim]) != past_len);
                                        const auto& dst = ctx.target_request->get_tensor(iport);
                                        if (dst->get_shape() == input_shape) {
                                            ctx.target_request->set_tensor(pyramid_iport, tensor);
                                        } else if (use_tensor_view) {
                                            const auto model_past_len =
                                                static_cast<int64_t>(pyramid->get_context_length(pyramid_id)) -
                                                static_cast<int64_t>(pyramid->query_size_at(pyramid_id));
                                            LOG_DEBUG("Use tensor view: past_len=" << past_len << " model_past_len="
                                                                                   << model_past_len);
                                            ctx.target_request->set_tensor(
                                                pyramid_iport,
                                                ov::npuw::util::view(tensor, dim, 0, model_past_len));
                                        } else {
                                            const auto& view = ov::npuw::util::view(tensor, dim, 0, past_len);
                                            const auto& dst_slice = ov::npuw::util::view(dst, dim, 0, past_len);
                                            ov::npuw::util::copy_tensor_by_dim(view,
                                                                               dst_slice,
                                                                               static_cast<uint32_t>(dim),
                                                                               static_cast<uint32_t>(dim));
                                        }
                                    }
                                }
                            } else {
                                // Non-KV, non-mask: bind directly
                                ctx.target_request->set_tensor(iport, tensor);
                            }
                        }
                        return true;
                    }
                    return false;
                case BehaviorKind::HFA:
                    if (const auto* hfa = ov::npuw::attn::get_compiled_hfa(pipeline.context)) {
                        auto& io = get_behavior_io(state,
                                                   ctx.subgraph_idx,
                                                   get_param_base(ctx, ctx.real_subgraph_idx),
                                                   hfa->_compiled_final_tile_model->outputs().size());
                        // HFA stores all inputs including block KV tensors
                        // (past_key_blocks / past_value_blocks vectors handled transparently
                        //  since every input_idx is stored and looked up by index in run())
                        io.inputs.at(input_idx) = tensor;
                        return true;
                    }
                    return false;
                case BehaviorKind::Paged:
                    if (const auto* pa = ov::npuw::attn::get_compiled_pa(pipeline.context)) {
                        // Mirrors HFA: cache every PA input by index. run() looks
                        // them up via _pa_param_index_map[PAInputId::*].
                        auto& io = get_behavior_io(state,
                                                   ctx.subgraph_idx,
                                                   get_param_base(ctx, ctx.real_subgraph_idx),
                                                   pa->_compiled_final_tile_model->outputs().size());
                        io.inputs.at(input_idx) = tensor;
                        return true;
                    }
                    return false;
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

            bool bind_function_output(ov::npuw::v1::subgraphs::InferContext& ctx,
                                      std::size_t output_idx,
                                      const ov::SoPtr<ov::ITensor>& tensor) override {
                (void)ctx;
                (void)output_idx;
                (void)tensor;
                return false;
            }

            void prologue(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                if (ctx.opaque_prologue) {
                    ctx.opaque_prologue();
                }
                auto& state = get_runtime_state(ctx);
                const auto& compiled_model = get_compiled_submodel(ctx, ctx.real_subgraph_idx);
                const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
                auto& io = get_behavior_io(state, ctx.subgraph_idx, get_param_base(ctx, ctx.real_subgraph_idx), 0u);
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                    if (const auto* dynamic = ov::npuw::attn::get_compiled_dynamic(pipeline.context)) {
                        auto mask_iport = compiled_model->inputs()[dynamic->mask_idx];
                        const auto& graph_mask = io.inputs.at(dynamic->mask_idx);
                        const auto this_case = state.attention_selector->this_case();
                        auto pos_id = state.attention_selector->length();

                        if (pos_id == -1) {
                            ctx.target_request->set_tensor(mask_iport, graph_mask);
                            return;
                        }

                        const auto past_len = state.attention_selector->past_length();
                        const auto present_len = dynamic->query_size;
                        auto set_or_copy = [&](const ov::SoPtr<ov::ITensor>& view) {
                            if (!get_request(ctx).subgraph_needs_copy(ctx.subgraph_idx)) {
                                ctx.target_request->set_tensor(mask_iport, view);
                            } else {
                                const auto& dst = ctx.target_request->get_tensor(mask_iport);
                                dst->set_shape(view->get_shape());
                                view->copy_to(dst._ptr);
                            }
                        };

                        using namespace ov::npuw::runtime;
                        if (this_case == attention::Selector::Case::GENERATE) {
                            set_or_copy(ov::npuw::util::view(ov::get_tensor_impl(dynamic->attend_all),
                                                             ATTN_KV_DIM,
                                                             0,
                                                             past_len + 1));
                            return;
                        }
                        if (this_case == attention::Selector::Case::PREFILL) {
                            if (state.cached_attention_mask) {
                                ctx.target_request->set_tensor(mask_iport, state.cached_attention_mask);
                                return;
                            }

                            auto full_mask_shape = graph_mask->get_shape();
                            auto actual_mask_shape = full_mask_shape;
                            actual_mask_shape[ATTN_KV_DIM] = present_len + past_len;

                            const auto& dst = ctx.target_request->get_tensor(mask_iport);
                            dst->set_shape(actual_mask_shape);

                            const auto& present_dst_view =
                                ov::npuw::util::view(dst, ATTN_KV_DIM, past_len, present_len);
                            const auto& present_src_view =
                                ov::npuw::util::view(graph_mask,
                                                     ATTN_KV_DIM,
                                                     full_mask_shape[ATTN_KV_DIM] - present_len,
                                                     present_len);
                            present_src_view->copy_to(present_dst_view._ptr);

                            if (past_len > 0) {
                                const auto& past_dst_view = ov::npuw::util::view(dst, ATTN_KV_DIM, 0, past_len);
                                const auto& past_src_view = ov::npuw::util::view(graph_mask, ATTN_KV_DIM, 0, past_len);
                                past_src_view->copy_to(past_dst_view._ptr);
                            }
                            state.cached_attention_mask = dst;
                            return;
                        }
                    }
                    return;
                case BehaviorKind::Pyramid:
                    if (const auto* pyramid = ov::npuw::attn::get_compiled_pyramid(pipeline.context)) {
                        const auto pyramid_id = state.pyramid_selector->pyramid_id();
                        const std::size_t dyn_mask_idx = pyramid->mask_idx_at(pyramid_id);
                        const std::size_t dyn_query_size = pyramid->query_size_at(pyramid_id);
                        auto mask_iport = pyramid->_compiled_models[pyramid_id]->inputs()[dyn_mask_idx];
                        const auto& graph_mask = io.inputs.at(dyn_mask_idx);
                        const auto this_case = state.pyramid_selector->this_case();
                        const auto present_len = dyn_query_size;
                        const auto& dst = ctx.target_request->get_tensor(mask_iport);

                        auto copy_mask_segment = [&](std::size_t dst_offset,
                                                     std::size_t src_offset,
                                                     std::size_t length) {
                            if (length == 0) {
                                return;
                            }
                            const auto& dst_view = ov::npuw::util::view(dst, ATTN_KV_DIM, dst_offset, length);
                            const auto& src_view = ov::npuw::util::view(graph_mask, ATTN_KV_DIM, src_offset, length);
                            ov::npuw::copy_tensor_by_dim(src_view, dst_view, ATTN_KV_DIM, ATTN_KV_DIM);
                        };

                        if (state.pyramid_selector->length() == -1) {
                            ctx.target_request->set_tensor(mask_iport, graph_mask);
                            return;
                        }

                        const auto past_len = state.pyramid_selector->past_length();
                        if (state.cached_attention_mask) {
                            ctx.target_request->set_tensor(mask_iport, state.cached_attention_mask);
                            return;
                        }

                        const auto full_mask_shape = graph_mask->get_shape();
                        using namespace ov::npuw::runtime;
                        if (this_case == pyramid_attention::Selector::Case::GENERATE) {
                            const auto dst_shape = dst->get_shape();
                            if (dst_shape == full_mask_shape) {
                                ctx.target_request->set_tensor(mask_iport, graph_mask);
                                state.cached_attention_mask = graph_mask;
                                return;
                            }

                            const std::size_t dst_present_offset = dst_shape[ATTN_KV_DIM] - present_len;
                            copy_mask_segment(dst_present_offset,
                                              full_mask_shape[ATTN_KV_DIM] - present_len,
                                              present_len);
                            copy_mask_segment(0, 0, dst_present_offset);
                            state.cached_attention_mask = dst;
                            return;
                        }
                        if (this_case == pyramid_attention::Selector::Case::PREFILL) {
                            copy_mask_segment(past_len, full_mask_shape[ATTN_KV_DIM] - present_len, present_len);
                            copy_mask_segment(0, 0, past_len);
                            state.cached_attention_mask = dst;
                            return;
                        }
                        OPENVINO_ASSERT(false, "Unsupported pyramid attention case");
                    }
                    return;
                case BehaviorKind::HFA:
                    return;
                case BehaviorKind::Paged:
                    // PA prologue is a no-op: the per-iteration mask is materialized
                    // inside run() (no caller-supplied mask to slice from), and
                    // initialize_pa_state() runs there too because state buffers must
                    // be reset on every call.
                    return;
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

            void run(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                case BehaviorKind::Pyramid:
                    ctx.legacy_infer();
                    return;
                case BehaviorKind::HFA:
                    if (const auto* hfa_desc = ov::npuw::attn::get_compiled_hfa(
                            get_subgraph_pipeline(ctx, ctx.real_subgraph_idx).context)) {
                        auto& state = get_runtime_state(ctx);
                        auto& io = get_behavior_io(state,
                                                   ctx.subgraph_idx,
                                                   get_param_base(ctx, ctx.real_subgraph_idx),
                                                   hfa_desc->_compiled_final_tile_model->outputs().size());

                        OPENVINO_ASSERT(hfa_desc->is_valid(), "HFA configuration must be valid");
                        const int64_t tile_size = hfa_desc->_tile_size;
                        const int64_t total_kv_length = state.hfa_selector->context_length();
                        const int64_t num_tiles = total_kv_length / tile_size;
                        OPENVINO_ASSERT(total_kv_length % tile_size == 0,
                                        "HFA total KV length must be multiple of tile size for now");

                        const auto& hfa_inputs = io.inputs;
                        const auto& sdpa_info = hfa_desc->_sdpa_attention_info;
                        const auto& sdpa_in = sdpa_info._sdpa_indices;

                        // Collect all KV block tensors (works for single-block and multi-block cases)
                        NPUW_ASSERT(!sdpa_in.past_key_blocks.empty() && !sdpa_in.past_value_blocks.empty() &&
                                    "SDPA indices must have at least one past_key/value block");
                        NPUW_ASSERT(sdpa_in.past_key_blocks.size() == sdpa_in.past_value_blocks.size() &&
                                    "Number of past key blocks must match number of past value blocks");
                        std::vector<ov::SoPtr<ov::ITensor>> past_key_blocks;
                        std::vector<ov::SoPtr<ov::ITensor>> past_value_blocks;
                        for (size_t i = 0; i < sdpa_in.past_key_blocks.size(); ++i) {
                            past_key_blocks.push_back(hfa_inputs.at(sdpa_in.past_key_blocks[i]));
                            past_value_blocks.push_back(hfa_inputs.at(sdpa_in.past_value_blocks[i]));
                        }
                        auto query_tensor = hfa_inputs.at(sdpa_in.query);
                        auto present_key_tensor = hfa_inputs.at(sdpa_in.present_key);
                        auto attention_mask_tensor = hfa_inputs.at(sdpa_in.attention_mask);
                        auto present_value_tensor = hfa_inputs.at(sdpa_in.present_value);
                        auto& regular_tile_request = state.hfa_requests.infer_requests[HFARequestSet::REGULAR_TILE];
                        auto& final_tile_request = state.hfa_requests.infer_requests[HFARequestSet::FINAL_TILE];
                        auto attention_output_tensor =
                            final_tile_request->get_tensor(hfa_desc->_compiled_final_tile_model->outputs()[0]);
                        const auto& tile_in = sdpa_info._tile_input_indices;
                        const auto& tile_out = sdpa_info._tile_output_indices;

                        ov::SoPtr<ov::ITensor> state_acc, state_max, state_sum;
                        if (state.hfa_runtime_ctx && state.hfa_runtime_ctx->has_state_buffers()) {
                            const auto& current_buffer = state.hfa_runtime_ctx->get_current_state_buffers();
                            state_acc = current_buffer.acc;
                            state_max = current_buffer.max;
                            state_sum = current_buffer.sum;
                            regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.acc],
                                                             state_acc);
                            regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.max],
                                                             state_max);
                            regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.d],
                                                             state_sum);
                        } else {
                            state_acc =
                                regular_tile_request->get_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.acc]);
                            state_max =
                                regular_tile_request->get_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.max]);
                            state_sum =
                                regular_tile_request->get_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.d]);
                            runtime::host_flash_attention::HFARuntimeContext::initialize_state_tensors(state_acc,
                                                                                                       state_max,
                                                                                                       state_sum);
                        }

                        regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->inputs()[tile_in.q],
                                                         query_tensor);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->inputs()[tile_in.q],
                                                       query_tensor);
                        regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->outputs()[tile_out.acc],
                                                         state_acc);
                        regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->outputs()[tile_out.max],
                                                         state_max);
                        regular_tile_request->set_tensor(hfa_desc->_compiled_tile_model->outputs()[tile_out.d],
                                                         state_sum);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->inputs()[tile_in.acc],
                                                       state_acc);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->inputs()[tile_in.max],
                                                       state_max);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->inputs()[tile_in.d],
                                                       state_sum);
                        final_tile_request->set_tensor(hfa_desc->_compiled_final_tile_model->outputs()[0],
                                                       attention_output_tensor);

                        const uint32_t K_SEQ_DIM = static_cast<uint32_t>(sdpa_info._k_seq_dim);
                        const uint32_t V_SEQ_DIM = static_cast<uint32_t>(sdpa_info._v_seq_dim);
                        constexpr uint32_t MASK_KV_SEQ_DIM = 3;
                        size_t next_available_mask_buffer_idx = 0;

                        auto process_tile = [&](auto& request,
                                                auto& model,
                                                const ov::SoPtr<ov::ITensor>& k_source,
                                                const ov::SoPtr<ov::ITensor>& v_source,
                                                int64_t kv_offset,
                                                int64_t mask_offset,
                                                int64_t tile_length,
                                                bool async = false) {
                            auto k_tile_buffer = request->get_tensor(model->inputs()[tile_in.k]);
                            auto v_tile_buffer = request->get_tensor(model->inputs()[tile_in.v]);
                            auto mask_tile_buffer = request->get_tensor(model->inputs()[tile_in.mask]);

                            if (can_reuse_tensor_zero_copy(k_source,
                                                           k_tile_buffer,
                                                           K_SEQ_DIM,
                                                           kv_offset,
                                                           tile_length)) {
                                request->set_tensor(model->inputs()[tile_in.k], k_source);
                            } else if (hfa_desc->_can_use_tensor_view) {
                                request->set_tensor(model->inputs()[tile_in.k],
                                                    ov::npuw::util::view(k_source, K_SEQ_DIM, kv_offset, tile_length));
                            } else {
                                extract_and_copy_tile(k_source, k_tile_buffer, K_SEQ_DIM, kv_offset, tile_length, "K");
                            }

                            if (can_reuse_tensor_zero_copy(v_source,
                                                           v_tile_buffer,
                                                           V_SEQ_DIM,
                                                           kv_offset,
                                                           tile_length)) {
                                request->set_tensor(model->inputs()[tile_in.v], v_source);
                            } else if (hfa_desc->_can_use_tensor_view) {
                                request->set_tensor(model->inputs()[tile_in.v],
                                                    ov::npuw::util::view(v_source, V_SEQ_DIM, kv_offset, tile_length));
                            } else {
                                extract_and_copy_tile(v_source, v_tile_buffer, V_SEQ_DIM, kv_offset, tile_length, "V");
                            }

                            if (attention_mask_tensor) {
                                if (can_reuse_tensor_zero_copy(attention_mask_tensor,
                                                               mask_tile_buffer,
                                                               MASK_KV_SEQ_DIM,
                                                               mask_offset,
                                                               tile_length)) {
                                    request->set_tensor(model->inputs()[tile_in.mask], attention_mask_tensor);
                                } else if (state.hfa_runtime_ctx.has_value()) {
                                    auto cached_tile =
                                        state.hfa_runtime_ctx->find_cached_mask_tile(attention_mask_tensor,
                                                                                     mask_offset,
                                                                                     tile_length);
                                    if (cached_tile) {
                                        request->set_tensor(model->inputs()[tile_in.mask], cached_tile);
                                    } else {
                                        ov::SoPtr<ov::ITensor> cached_mask_tile =
                                            state.hfa_runtime_ctx->get_mask_tile_buffer(next_available_mask_buffer_idx);
                                        extract_and_copy_tile(attention_mask_tensor,
                                                              cached_mask_tile,
                                                              MASK_KV_SEQ_DIM,
                                                              mask_offset,
                                                              tile_length,
                                                              "Mask");
                                        state.hfa_runtime_ctx->cache_mask_tile(attention_mask_tensor,
                                                                               mask_offset,
                                                                               tile_length,
                                                                               cached_mask_tile);
                                        request->set_tensor(model->inputs()[tile_in.mask], cached_mask_tile);
                                        next_available_mask_buffer_idx++;
                                    }
                                } else {
                                    extract_and_copy_tile(attention_mask_tensor,
                                                          mask_tile_buffer,
                                                          MASK_KV_SEQ_DIM,
                                                          mask_offset,
                                                          tile_length,
                                                          "Mask");
                                }
                            }

                            if (async) {
                                request->start_async();
                                if (state.hfa_runtime_ctx && state.hfa_runtime_ctx->has_state_buffers()) {
                                    state.hfa_runtime_ctx->prepare_next_state_buffers();
                                }
                                request->wait();
                            } else {
                                request->infer();
                            }
                        };

                        int64_t mask_tile_offset = 0;
                        int64_t past_kv_tiles = num_tiles - 1;  // tiles driven from past blocks

                        // Iterate through KV blocks; each block contributes block_size/tile_size tiles.
                        for (size_t block_idx = 0; block_idx < past_key_blocks.size() && past_kv_tiles > 0;
                             ++block_idx) {
                            const auto& k_block = past_key_blocks[block_idx];
                            const auto& v_block = past_value_blocks[block_idx];
                            const int64_t block_size = static_cast<int64_t>(k_block->get_shape()[K_SEQ_DIM]);
                            NPUW_ASSERT(block_size % tile_size == 0 &&
                                        "HFA block size must be a multiple of tile size");
                            const int64_t tiles_in_block = block_size / tile_size;

                            for (int64_t t = 0; t < tiles_in_block && past_kv_tiles > 0; ++t) {
                                process_tile(regular_tile_request,
                                             hfa_desc->_compiled_tile_model,
                                             k_block,
                                             v_block,
                                             t * tile_size,
                                             mask_tile_offset,
                                             tile_size);
                                mask_tile_offset += tile_size;
                                past_kv_tiles--;
                            }
                        }
                        NPUW_ASSERT(past_kv_tiles == 0 &&
                                    "HFA: All past KV blocks should contain exactly (num_tiles - 1) tiles");

                        if (num_tiles > 0) {
                            const size_t present_seq_length = present_key_tensor->get_shape()[K_SEQ_DIM];
                            const int64_t final_tile_length = static_cast<int64_t>(present_seq_length);
                            OPENVINO_ASSERT(
                                final_tile_length == tile_size,
                                "Final tile must process entire present KV sequence in a single inference. "
                                "This is guaranteed during compilation (tile_size = query_size = present_seq_length).");
                            const int64_t mask_total_length = attention_mask_tensor->get_shape()[MASK_KV_SEQ_DIM];
                            const int64_t final_mask_offset = mask_total_length - final_tile_length;
                            process_tile(final_tile_request,
                                         hfa_desc->_compiled_final_tile_model,
                                         present_key_tensor,
                                         present_value_tensor,
                                         0,
                                         final_mask_offset,
                                         final_tile_length,
                                         true);
                        }

                        if (state.hfa_runtime_ctx && state.hfa_runtime_ctx->has_state_buffers()) {
                            state.hfa_runtime_ctx->switch_buffers();
                        }
                        return;
                    }
                    return;
                case BehaviorKind::Paged: {
                    // PagedAttention NPU lowering. One tile loop per call:
                    //   walk active blocks (block_indices[0..num_active_blocks)),
                    //   bind each layer-pool block as K/V tile,
                    //   thread (acc, max, d) state across iterations,
                    //   final tile attends over the present (new tokens) K/V,
                    //   write the new K/V into the tail block of the pool so
                    //   the next call sees them as cached.
                    //
                    // Uses the per-layer pool selected by pa_desc->_layer_index
                    // (one PA op per attention layer). The layer managers + the
                    // compiled tile sub-models are sourced from the
                    // compiled::PagedAttention scaffold populated by the
                    // partition_stage hook from the LLMCompiledModel scope.
                    auto& state = get_runtime_state(ctx);
                    const auto& pipeline = get_subgraph_pipeline(ctx, ctx.real_subgraph_idx);
                    auto* pa_desc = ov::npuw::attn::get_compiled_pa(pipeline.context);
                    if (pa_desc == nullptr || !pa_desc->is_valid() ||
                        pa_desc->_layer_key_managers.empty() || pa_desc->_layer_value_managers.empty()) {
                        OPENVINO_THROW("PagedAttention dispatch: compiled state incomplete (compiled tile "
                                       "models or layer managers missing). The partitioner -> dispatch "
                                       "compiled-handle bridge is not yet wired.");
                    }
                    if (pa_desc->_layer_index >= pa_desc->_layer_key_managers.size()) {
                        OPENVINO_THROW("PagedAttention dispatch: layer ",
                                       pa_desc->_layer_index,
                                       " out of range (have ",
                                       pa_desc->_layer_key_managers.size(),
                                       " pools)");
                    }
                    const auto& layer_key = pa_desc->_layer_key_managers[pa_desc->_layer_index];
                    const auto& layer_value = pa_desc->_layer_value_managers[pa_desc->_layer_index];
                    OPENVINO_ASSERT(layer_key && layer_value,
                                    "PagedAttention dispatch: layer ",
                                    pa_desc->_layer_index,
                                    " has no K/V pool");

                    auto& io = get_behavior_io(state,
                                               ctx.subgraph_idx,
                                               get_param_base(ctx, ctx.real_subgraph_idx),
                                               pa_desc->_compiled_final_tile_model->outputs().size());
                    const auto& base = pa_desc->_pa_param_index_map;
                    auto pa_input = [&](PAInputId id) -> ov::SoPtr<ov::ITensor> {
                        const auto it = base.find(id);
                        if (it == base.end()) {
                            return {};
                        }
                        return io.inputs.at(it->second);
                    };

                    auto block_indices_tensor = pa_input(PAInputId::BLOCK_INDICES);
                    auto past_lens_tensor = pa_input(PAInputId::PAST_LENS);
                    auto subseq_begins_tensor = pa_input(PAInputId::SUBSEQUENCE_BEGINS);
                    auto block_indices_begins_tensor = pa_input(PAInputId::BLOCK_INDICES_BEGINS);
                    auto query_tensor = pa_input(PAInputId::QUERY);
                    auto present_key_tensor = pa_input(PAInputId::KEY);
                    auto present_value_tensor = pa_input(PAInputId::VALUE);

                    OPENVINO_ASSERT(block_indices_tensor && past_lens_tensor && subseq_begins_tensor &&
                                        block_indices_begins_tensor && query_tensor && present_key_tensor &&
                                        present_value_tensor,
                                    "PagedAttention dispatch: required input tensor missing");

                    // Single-sequence path: one row of past_lens / subseq_begins /
                    // block_indices_begins is enough. Multi-sequence batching
                    // (ContinuousBatchingPipeline) is out of scope for v1.
                    const auto* past_lens_data = past_lens_tensor->data<int32_t>();
                    const auto* block_indices_begins_data = block_indices_begins_tensor->data<int32_t>();
                    const auto* subseq_begins_data = subseq_begins_tensor->data<int32_t>();
                    const auto* block_indices_data = block_indices_tensor->data<int32_t>();
                    const uint32_t past_lens = static_cast<uint32_t>(past_lens_data[0]);
                    const uint32_t bib_start = static_cast<uint32_t>(block_indices_begins_data[0]);
                    const uint32_t bib_end = static_cast<uint32_t>(block_indices_begins_data[1]);
                    const uint32_t num_active_blocks = bib_end - bib_start;
                    const uint32_t new_tokens =
                        static_cast<uint32_t>(subseq_begins_data[1] - subseq_begins_data[0]);

                    initialize_pa_state(state.pa_state);

                    auto& tile_req = state.pa_state.tile_request;
                    auto& final_req = state.pa_state.final_tile_request;
                    const auto& tile_inputs = tile_req->get_compiled_model()->inputs();
                    const auto& tile_outputs = tile_req->get_compiled_model()->outputs();
                    const auto& final_inputs = final_req->get_compiled_model()->inputs();
                    const auto& final_outputs = final_req->get_compiled_model()->outputs();
                    auto bind_tile_in = [&](OnlineSoftmaxTileInputId id, const ov::SoPtr<ov::ITensor>& t) {
                        tile_req->set_tensor(tile_inputs.at(static_cast<std::size_t>(id)), t);
                    };
                    auto bind_tile_out = [&](OnlineSoftmaxTileOutputId id, const ov::SoPtr<ov::ITensor>& t) {
                        tile_req->set_tensor(tile_outputs.at(static_cast<std::size_t>(id)), t);
                    };
                    auto bind_final_in = [&](OnlineSoftmaxTileInputId id, const ov::SoPtr<ov::ITensor>& t) {
                        final_req->set_tensor(final_inputs.at(static_cast<std::size_t>(id)), t);
                    };

                    bind_tile_in(OnlineSoftmaxTileInputId::Q, query_tensor);
                    bind_tile_in(OnlineSoftmaxTileInputId::MASK_TILE, state.pa_state.mask_tile_buffer);
                    bind_tile_in(OnlineSoftmaxTileInputId::PAST_ACC, state.pa_state.state_acc);
                    bind_tile_in(OnlineSoftmaxTileInputId::PAST_MAX, state.pa_state.state_max);
                    bind_tile_in(OnlineSoftmaxTileInputId::PAST_D, state.pa_state.state_d);
                    bind_tile_out(OnlineSoftmaxTileOutputId::ACC, state.pa_state.state_acc);
                    bind_tile_out(OnlineSoftmaxTileOutputId::MAXX, state.pa_state.state_max);
                    bind_tile_out(OnlineSoftmaxTileOutputId::D, state.pa_state.state_d);

                    // The cached-block tile loop. The "tail" block (the one being
                    // filled by the new tokens) is excluded — it is consumed by
                    // the final tile via the present K/V tensors instead. So we
                    // iterate up to (num_active_blocks - 1).
                    const uint32_t cached_block_count = num_active_blocks > 0 ? num_active_blocks - 1 : 0;
                    for (uint32_t i = 0; i < cached_block_count; ++i) {
                        fill_pa_mask_for_cached_block(state.pa_state.mask_tile_buffer, past_lens, i);
                        const auto phys_id = static_cast<uint32_t>(block_indices_data[bib_start + i]);
                        auto k_block = layer_key->get_block_tensor(phys_id);
                        auto v_block = layer_value->get_block_tensor(phys_id);
                        bind_tile_in(OnlineSoftmaxTileInputId::K_TILE, k_block);
                        bind_tile_in(OnlineSoftmaxTileInputId::V_TILE, v_block);
                        tile_req->infer();
                    }

                    // Final tile: present K/V from this step. Mask switches to
                    // causal-within-new-tokens. Output is the layer's attention
                    // result, written into io.outputs[0] for downstream consumers.
                    fill_pa_mask_for_present(state.pa_state.mask_tile_buffer, new_tokens);
                    bind_final_in(OnlineSoftmaxTileInputId::Q, query_tensor);
                    bind_final_in(OnlineSoftmaxTileInputId::MASK_TILE, state.pa_state.mask_tile_buffer);
                    bind_final_in(OnlineSoftmaxTileInputId::PAST_ACC, state.pa_state.state_acc);
                    bind_final_in(OnlineSoftmaxTileInputId::PAST_MAX, state.pa_state.state_max);
                    bind_final_in(OnlineSoftmaxTileInputId::PAST_D, state.pa_state.state_d);
                    bind_final_in(OnlineSoftmaxTileInputId::K_TILE, present_key_tensor);
                    bind_final_in(OnlineSoftmaxTileInputId::V_TILE, present_value_tensor);
                    if (!io.outputs.empty() && io.outputs.at(0)) {
                        final_req->set_tensor(final_outputs.at(0), io.outputs.at(0));
                    }
                    final_req->infer();

                    // Write the new K/V into the pool blocks so the next call
                    // sees them as cached. SDPAToPagedAttention's PA op does this
                    // implicitly inside its CPU executor; we have to do it
                    // explicitly because the lowering replaces the executor.
                    //
                    // Generate steps land within a single block (new_tokens=1)
                    // but a prefill call delivers seq_len tokens at once and may
                    // span multiple blocks. Walk the contiguous block range
                    // [bib_start, bib_end) starting from the block that holds
                    // (past_lens % block_size) and copy block_size-sized chunks
                    // until all new_tokens are placed.
                    if (num_active_blocks > 0 && new_tokens > 0) {
                        const uint32_t block_size = static_cast<uint32_t>(pa_desc->_block_size);
                        uint32_t remaining = new_tokens;
                        uint32_t src_offset = 0;
                        const uint32_t first_block_idx = past_lens / block_size;
                        uint32_t cur_block = first_block_idx;
                        uint32_t cur_offset = past_lens % block_size;
                        while (remaining > 0) {
                            OPENVINO_ASSERT(cur_block < num_active_blocks,
                                            "PagedAttention dispatch: block-table too short to absorb "
                                            "new K/V at layer ",
                                            pa_desc->_layer_index,
                                            " (need=",
                                            new_tokens,
                                            " active_blocks=",
                                            num_active_blocks,
                                            " block_size=",
                                            block_size,
                                            "). Block-table growth must precede this call.");
                            const uint32_t fits = std::min(block_size - cur_offset, remaining);
                            const auto phys_id =
                                static_cast<uint32_t>(block_indices_data[bib_start + cur_block]);
                            auto blk_k = layer_key->get_block_tensor(phys_id);
                            auto blk_v = layer_value->get_block_tensor(phys_id);
                            // Sequence dim differs between K and V (axis 2 for K,
                            // axis 3 for V — see online_softmax_tile.cpp:create_tile_params).
                            ov::npuw::copy_tensor_by_dim(
                                ov::npuw::util::view(present_key_tensor, /*seq_dim=*/2, src_offset, fits),
                                ov::npuw::util::view(blk_k, /*seq_dim=*/2, cur_offset, fits),
                                /*src_dim=*/2,
                                /*dst_dim=*/2);
                            ov::npuw::copy_tensor_by_dim(
                                ov::npuw::util::view(present_value_tensor, /*seq_dim=*/3, src_offset, fits),
                                ov::npuw::util::view(blk_v, /*seq_dim=*/3, cur_offset, fits),
                                /*src_dim=*/3,
                                /*dst_dim=*/3);
                            layer_key->update_block_tokens(phys_id, cur_offset + fits);
                            layer_value->update_block_tokens(phys_id, cur_offset + fits);
                            src_offset += fits;
                            remaining -= fits;
                            ++cur_block;
                            cur_offset = 0;
                        }
                    }
                    return;
                }
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

        private:
            BehaviorKind m_kind;
        };

        return std::make_unique<AttnBehavior>(get_behavior_kind(ctx));
    };
}

}  // namespace

void put_compiled_dynamic(v1::subgraphs::Context& context, CompiledDynamicState state) {
    context.put<CompiledDynamicState>(std::move(state));
}

void put_compiled_pyramid(v1::subgraphs::Context& context, CompiledPyramidState state) {
    context.put<CompiledPyramidState>(std::move(state));
}

void put_compiled_hfa(v1::subgraphs::Context& context, CompiledHFAState state) {
    context.put<CompiledHFAState>(std::move(state));
}

ov::npuw::compiled::Attention* get_compiled_dynamic(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::Attention>(context);
}

const ov::npuw::compiled::Attention* get_compiled_dynamic(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::Attention>(context);
}

ov::npuw::compiled::PyramidAttention* get_compiled_pyramid(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::PyramidAttention>(context);
}

const ov::npuw::compiled::PyramidAttention* get_compiled_pyramid(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::PyramidAttention>(context);
}

ov::npuw::compiled::HostFlashAttention* get_compiled_hfa(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::HostFlashAttention>(context);
}

const ov::npuw::compiled::HostFlashAttention* get_compiled_hfa(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::HostFlashAttention>(context);
}

void put_compiled_pa(v1::subgraphs::Context& context, CompiledPagedState state) {
    context.put<CompiledPagedState>(std::move(state));
}

ov::npuw::compiled::PagedAttention* get_compiled_pa(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::PagedAttention>(context);
}

const ov::npuw::compiled::PagedAttention* get_compiled_pa(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::PagedAttention>(context);
}

bool has_compiled_state(const v1::subgraphs::CompiledPipeline& pipeline) {
    return get_compiled_dynamic(pipeline.context) != nullptr || get_compiled_pyramid(pipeline.context) != nullptr ||
           get_compiled_hfa(pipeline.context) != nullptr;
}

void serialize_compiled_state(v1::subgraphs::Context& context,
                              ov::npuw::s11n::Stream& stream,
                              const ov::npuw::s11n::SubmodelDeserializeCtx* submodel_ctx) {
    std::optional<ov::npuw::compiled::Attention> dynamic;
    if (const auto* state = get_compiled_dynamic(context)) {
        dynamic = *state;
    }
    stream & dynamic;
    if (stream.input() && dynamic.has_value()) {
        put_compiled_dynamic(context, std::make_shared<ov::npuw::compiled::Attention>(dynamic.value()));
    }

    bool has_pyramid = (get_compiled_pyramid(context) != nullptr);
    stream & has_pyramid;
    if (has_pyramid) {
        if (stream.output()) {
            uint8_t tag = get_compiled_pyramid(context)->is_block_mode() ? 1u : 0u;
            stream & tag;
            ov::npuw::orc::serialize(stream, *get_compiled_pyramid(context));
        } else {
            uint8_t tag;
            stream & tag;
            auto pyramid_ptr = ov::npuw::orc::make_pyramid_from_stream(stream, tag);
            put_compiled_pyramid(context, std::move(pyramid_ptr));
        }
    }

    auto* mutable_pyramid = get_compiled_pyramid(context);
    if (mutable_pyramid != nullptr) {
        size_t num_models = 0;
        if (stream.output()) {
            num_models = mutable_pyramid->_compiled_models.size();
        }
        stream & num_models;

        if (stream.output()) {
            for (size_t i = 0; i < num_models - 1; ++i) {
                std::stringstream ss;
                mutable_pyramid->_compiled_models[i]->export_model(ss);
                std::string model_str = ss.str();
                stream & model_str;
            }
        } else if (num_models > 0) {
            mutable_pyramid->_compiled_models.resize(num_models);
            NPUW_ASSERT(submodel_ctx != nullptr);
            for (size_t i = 0; i < num_models - 1; ++i) {
                std::string model_str;
                stream & model_str;
                std::stringstream ss(model_str);
                mutable_pyramid->_compiled_models[i] =
                    submodel_ctx->plugin->get_core()->import_model(ss,
                                                                   submodel_ctx->device,
                                                                   submodel_ctx->import_config);
            }
            if (submodel_ctx->compiled_model) {
                mutable_pyramid->_compiled_models[num_models - 1] = submodel_ctx->compiled_model;
                LOG_DEBUG("Reused compiled_model for the last pyramid attention model");
            }
        }
    }

    std::optional<ov::npuw::compiled::HostFlashAttention> hfa;
    if (const auto* state = get_compiled_hfa(context)) {
        hfa = *state;
    }
    stream & hfa;
    if (stream.input() && hfa.has_value()) {
        put_compiled_hfa(context, std::make_shared<ov::npuw::compiled::HostFlashAttention>(hfa.value()));
    }

    auto* mutable_hfa = get_compiled_hfa(context);
    if (mutable_hfa != nullptr) {
        bool has_compiled_model = false;
        if (stream.output()) {
            has_compiled_model = mutable_hfa->_compiled_tile_model != nullptr;
        }
        stream & has_compiled_model;
        if (has_compiled_model) {
            if (stream.output()) {
                std::stringstream ss;
                mutable_hfa->_compiled_tile_model->export_model(ss);
                std::string model_str = ss.str();
                stream & model_str;
            } else {
                NPUW_ASSERT(submodel_ctx != nullptr);
                std::string model_str;
                stream & model_str;
                std::stringstream ss(model_str);
                mutable_hfa->_compiled_tile_model =
                    submodel_ctx->plugin->get_core()->import_model(ss,
                                                                   submodel_ctx->device,
                                                                   submodel_ctx->import_config);
                LOG_DEBUG("Imported compiled tile model for host flash attention");
            }
        }
        if (stream.input()) {
            NPUW_ASSERT(submodel_ctx != nullptr);
            mutable_hfa->_compiled_final_tile_model = submodel_ctx->compiled_model;
            LOG_DEBUG("Set compiled final tile model reference for host flash attention");
        }
    }
}

void attach_runtime_behavior(ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                             ov::npuw::v1::subgraphs::Context& compiled_context,
                             BehaviorKind kind) {
    compiled_pipeline.registration.group = ov::npuw::patterns::attn::SDPA::group_name();
    compiled_pipeline.registration.name = ov::npuw::patterns::attn::SDPA::pattern_name();
    compiled_context.put<BehaviorKind>(kind);
    ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
    spec.registration = compiled_pipeline.registration;
    spec.context = compiled_context;
    spec.factory = make_runtime_factory();
    spec.handles_function_prologue = true;
    compiled_pipeline.runtime_behavior = std::move(spec);
}

std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> register_patterns(
    ov::npuw::v1::subgraphs::PatternRegistry& registry) {
    std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> registrations;
    registrations.reserve(3);

    // Matcher-only registrations so the registry handles SDPA/SDPADecomposed isolation
    // (replaces the legacy HNDL_ATTN fallback in snapshot.cpp)
    registrations.emplace_back(registry.on<ov::npuw::patterns::attn::SDPA>().scoped());
    registrations.emplace_back(registry.on<ov::npuw::patterns::attn::SDPADecomposed>().scoped());
    // Paged Attention matcher: tags ov::op::PagedAttentionExtension with the
    // shared "attn" isolation tag so the online partitioner pulls the paged-
    // attention subgraph into its own group. Routed to NPUW_ATTN_DEVICE
    // (default CPU) via the per-submodel device override.
    registrations.emplace_back(registry.on<ov::npuw::patterns::attn::PagedAttn>().scoped());

    // Behavior registration: fires for any function tagged "attn" (i.e. from any SDPA-family
    // isolation pattern). The at_partition callback checks whether dynamic
    // attention was detected (f._attention set by Partitioner::attention()) and, if so,
    // stashes the compile-time descriptor in the pipeline context for later stages.
    ov::npuw::v1::subgraphs::PatternRegistration attn_behavior;
    attn_behavior.tag = ov::npuw::patterns::attn::SDPA::isolation_tag();
    attn_behavior.partition_stage = [](ov::npuw::Function& f, ov::npuw::v1::subgraphs::Context& ctx) {
        if (f._attention.has_value()) {
            put_compiled_dynamic(ctx, std::make_shared<ov::npuw::compiled::Attention>(f._attention.value(), f._model));
            ctx.put<BehaviorKind>(BehaviorKind::Dynamic);
            return;
        }
        if (f._pyramid_attention.has_value()) {
            put_compiled_pyramid(ctx, ov::npuw::compiled::PyramidAttention::make(f._pyramid_attention.value()));
            ctx.put<BehaviorKind>(BehaviorKind::Pyramid);
            return;
        }
        if (f._host_flash_attention.has_value()) {
            put_compiled_hfa(ctx,
                             std::make_shared<ov::npuw::compiled::HostFlashAttention>(f._host_flash_attention.value()));
            ctx.put<BehaviorKind>(BehaviorKind::HFA);
            return;
        }
        if (f._paged_attention.has_value() && f._paged_attention->is_valid()) {
            // PagedAttention NPU lowering. Both the compiled tile sub-models
            // and the per-layer KV pool managers are sourced from the
            // PagedAttentionPartitionerScope thread-local set up by the LLM
            // compile path before invoking the partitioner. The partitioner
            // has no back-reference to LLMCompiledModel, so this thread-local
            // is the handoff. Without a context, we don't push BehaviorKind::Paged
            // and the legacy CPU PA auto-route handles execution.
            const auto& fpa = f._paged_attention.value();
            const auto* pa_ctx = ov::npuw::function::current_pa_partitioner_context();
            if (pa_ctx == nullptr || !pa_ctx->compiled_tile_model || !pa_ctx->compiled_final_tile_model) {
                LOG_WARN("PagedAttention NPU lowering: no partitioner context for compiled handles; "
                         "falling back to CPU PA auto-route for layer "
                         << fpa._layer_index);
            } else {
                put_compiled_pa(ctx,
                                std::make_shared<ov::npuw::compiled::PagedAttention>(fpa,
                                                                                     pa_ctx->compiled_tile_model,
                                                                                     pa_ctx->compiled_final_tile_model,
                                                                                     pa_ctx->key_managers,
                                                                                     pa_ctx->value_managers));
                ctx.put<BehaviorKind>(BehaviorKind::Paged);
                LOG_INFO("PagedAttention NPU lowering: pushed BehaviorKind::Paged for layer "
                         << fpa._layer_index);
            }
        }
    };
    attn_behavior.compile_stage = [](ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                                     ov::npuw::v1::subgraphs::Context& compiled_context) {
        const auto* kind = compiled_context.get_if<BehaviorKind>();
        if (kind == nullptr) {
            return;
        }
        // For PA: at partition_stage we pushed BehaviorKind::Paged but the
        // compiled::PagedAttention handle in the context still points at
        // the un-narrowed compiled-model handles set by setup_paged_runtime
        // — those are valid across the lifetime of the LLMCompiledModel.
        // Nothing extra to do at compile_stage beyond attach_runtime_behavior.
        attach_runtime_behavior(compiled_pipeline, compiled_context, *kind);
    };
    registrations.emplace_back(registry.add(std::move(attn_behavior)));

    return registrations;
}

}  // namespace attn
}  // namespace npuw
}  // namespace ov
