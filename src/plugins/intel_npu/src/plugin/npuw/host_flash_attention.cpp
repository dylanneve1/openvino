// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "host_flash_attention.hpp"

#include "intel_npu/ops/flash_attention_tile.hpp"
#include "logging.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace function {

namespace opp = ov::pass::pattern;


// ============================================================================
// Helper function: Extract actual Parameter by skipping Convert nodes
// ============================================================================
static std::shared_ptr<ov::Node> skip_convert_nodes(const std::shared_ptr<ov::Node>& node) {
    auto current = node;
    while (current && ov::is_type<ov::op::v0::Convert>(current.get())) {
        if (current->get_input_size() > 0) {
            current = current->get_input_node_shared_ptr(0);
        } else {
            break;
        }
    }
    return current;
}

// ============================================================================
// Helper function: Build SDPA parameter index mapping
// ============================================================================
static void build_sdpa_param_mapping(HostFlashAttention& hfa,
                                     const std::shared_ptr<ov::Model>& model,
                                     const ov::npuw::util::SDPAPatternNodes& pattern_nodes) {
    LOG_INFO("Building SDPA input parameter index mapping...");

    // Helper lambda to safely extract parameter from node (skipping Convert ops)
    auto extract_param = [&](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::op::v0::Parameter> {
        return ov::as_type_ptr<ov::op::v0::Parameter>(skip_convert_nodes(node));
    };

    // Extract Q (query) parameter - input 0 of MatMul1
    if (auto q_param = extract_param(pattern_nodes.matmul1_node->get_input_node_shared_ptr(0))) {
        hfa._query_param_idx = model->get_parameter_index(q_param);
    }

    // Extract past KV parameters from a Concat node: all inputs except the last are treated as
    // past (one entry in non-block mode, multiple entries in block mode); the last input is
    // the present key/value. Key and value follow identical logic.
    auto extract_kv_params = [&](const std::shared_ptr<ov::Node>& concat_node,
                                 std::vector<std::size_t>& block_indices,
                                 std::size_t& present_idx_out,
                                 const char* kv_name) {
        if (!concat_node)
            return;
        const size_t n = concat_node->get_input_size();
        block_indices.clear();
        block_indices.reserve(n - 1);
        for (size_t i = 0; i < n - 1; ++i) {
            if (auto param = extract_param(concat_node->get_input_node_shared_ptr(i))) {
                const std::size_t idx = model->get_parameter_index(param);
                block_indices.push_back(idx);
                LOG_DEBUG("  Found " << kv_name << " block[" << i << "] at parameter index " << idx);
            } else {
                LOG_WARN("Could not extract parameter from " << kv_name << " Concat input[" << i << "]");
            }
        }
        if (auto param = extract_param(concat_node->get_input_node_shared_ptr(n - 1))) {
            present_idx_out = model->get_parameter_index(param);
            LOG_DEBUG("  Found " << kv_name << "_present at parameter index " << present_idx_out);
        }
    };

    extract_kv_params(pattern_nodes.past_key_concat_node,
                      hfa._past_key_block_indices,
                      hfa._present_key_param_idx,
                      "past_key");
    extract_kv_params(pattern_nodes.past_value_concat_node,
                      hfa._past_value_block_indices,
                      hfa._present_value_param_idx,
                      "past_value");

    // Extract mask parameter - input 1 of add_node
    if (auto add_param = extract_param(pattern_nodes.add_node->get_input_node_shared_ptr(1))) {
        hfa._attention_mask_param_idx = model->get_parameter_index(add_param);
    }

    LOG_INFO("Built SDPA input mapping: query="
             << hfa._query_param_idx << ", present_key=" << hfa._present_key_param_idx
             << ", present_value=" << hfa._present_value_param_idx << ", mask=" << hfa._attention_mask_param_idx);
    LOG_INFO("  Past key blocks: " << hfa._past_key_block_indices.size());
    LOG_INFO("  Past value blocks: " << hfa._past_value_block_indices.size());

    // Print KV cache blocks
    LOG_DEBUG("Past key blocks (" << hfa._past_key_block_indices.size() << "):");
    for (size_t i = 0; i < hfa._past_key_block_indices.size(); ++i) {
        LOG_DEBUG("  block[" << i << "] -> parameter[" << hfa._past_key_block_indices[i] << "]");
    }

    LOG_DEBUG("Past value blocks (" << hfa._past_value_block_indices.size() << "):");
    for (size_t i = 0; i < hfa._past_value_block_indices.size(); ++i) {
        LOG_DEBUG("  block[" << i << "] -> parameter[" << hfa._past_value_block_indices[i] << "]");
    }

    LOG_DEBUG("=============================================");
}

// ============================================================================
// Helper function: Build tile model parameter index mapping
// ============================================================================
static void build_tile_param_mapping(HostFlashAttention& hfa, const std::shared_ptr<ov::Model>& tile_model) {
    LOG_INFO("Building HFA Tile Model input index mapping...");

    // Parse tile model inputs by their tensor names
    // Expected input order: [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    const auto& tile_inputs = tile_model->inputs();
    for (std::size_t i = 0; i < tile_inputs.size(); ++i) {
        const auto& tensor_names = tile_inputs[i].get_names();
        if (tensor_names.empty()) {
            LOG_WARN("Tile model input[" << i << "] has no tensor name");
            continue;
        }

        const std::string& name = *tensor_names.begin();

        // Map tensor name to enum ID
        if (name == online_softmax_tile_input_id_to_string(OnlineSoftmaxTileInputId::PAST_ACC)) {
            hfa._tile_param_index_map[OnlineSoftmaxTileInputId::PAST_ACC] = i;
        } else if (name == online_softmax_tile_input_id_to_string(OnlineSoftmaxTileInputId::PAST_MAX)) {
            hfa._tile_param_index_map[OnlineSoftmaxTileInputId::PAST_MAX] = i;
        } else if (name == online_softmax_tile_input_id_to_string(OnlineSoftmaxTileInputId::PAST_D)) {
            hfa._tile_param_index_map[OnlineSoftmaxTileInputId::PAST_D] = i;
        } else if (name == online_softmax_tile_input_id_to_string(OnlineSoftmaxTileInputId::K_TILE)) {
            hfa._tile_param_index_map[OnlineSoftmaxTileInputId::K_TILE] = i;
        } else if (name == online_softmax_tile_input_id_to_string(OnlineSoftmaxTileInputId::V_TILE)) {
            hfa._tile_param_index_map[OnlineSoftmaxTileInputId::V_TILE] = i;
        } else if (name == online_softmax_tile_input_id_to_string(OnlineSoftmaxTileInputId::Q)) {
            hfa._tile_param_index_map[OnlineSoftmaxTileInputId::Q] = i;
        } else if (name == online_softmax_tile_input_id_to_string(OnlineSoftmaxTileInputId::MASK_TILE)) {
            hfa._tile_param_index_map[OnlineSoftmaxTileInputId::MASK_TILE] = i;
        } else {
            LOG_WARN("Unknown tile model input name: " << name);
        }
    }

    // Print the tile input mapping
    LOG_DEBUG("");
    LOG_DEBUG("========== HFA Tile Model Input Mapping ==========");
    LOG_DEBUG("Total entries: " << hfa._tile_param_index_map.size());

    for (const auto& [input_id, input_idx] : hfa._tile_param_index_map) {
        LOG_DEBUG("  " << online_softmax_tile_input_id_to_string(input_id) << " -> input[" << input_idx << "]");
    }
    LOG_DEBUG("==================================================");
}

// ============================================================================
// Helper function: Build tile model output index mapping
// ============================================================================
static void build_tile_output_mapping(HostFlashAttention& hfa, const std::shared_ptr<ov::Model>& tile_model) {
    LOG_INFO("Building HFA Tile Model output index mapping...");

    // Parse tile model outputs by their tensor names
    // Expected output order: [acc, maxx, d]
    const auto& tile_outputs = tile_model->outputs();
    for (std::size_t i = 0; i < tile_outputs.size(); ++i) {
        const auto& tensor_names = tile_outputs[i].get_names();
        if (tensor_names.empty()) {
            LOG_WARN("Tile model output[" << i << "] has no tensor name");
            continue;
        }

        const std::string& name = *tensor_names.begin();

        // Map tensor name to enum ID
        if (name == "acc") {
            hfa._tile_output_index_map[OnlineSoftmaxTileOutputId::ACC] = i;
        } else if (name == "maxx") {
            hfa._tile_output_index_map[OnlineSoftmaxTileOutputId::MAXX] = i;
        } else if (name == "d") {
            hfa._tile_output_index_map[OnlineSoftmaxTileOutputId::D] = i;
        } else {
            LOG_WARN("Unknown tile model output name: " << name);
        }
    }

    // Print the tile output mapping
    LOG_DEBUG("");
    LOG_DEBUG("========== HFA Tile Model Output Mapping ==========");
    LOG_DEBUG("Total entries: " << hfa._tile_output_index_map.size());

    for (const auto& [output_id, output_idx] : hfa._tile_output_index_map) {
        LOG_DEBUG("  " << online_softmax_tile_output_id_to_string(output_id) << " -> output[" << output_idx << "]");
    }
    LOG_DEBUG("==================================================");
}

// ============================================================================
// Helper function: Extract sequence dimension from Concat node
// ============================================================================
static std::optional<std::size_t> extract_sequence_dim_from_concat(const std::shared_ptr<ov::Node>& concat_node,
                                                                   const std::string& tensor_name) {
    if (!concat_node) {
        LOG_WARN("Failed to extract " << tensor_name << " concat node");
        return std::nullopt;
    }

    auto concat_op = std::dynamic_pointer_cast<ov::op::v0::Concat>(concat_node);
    if (!concat_op) {
        LOG_WARN("Failed to cast " << tensor_name << "_concat to Concat op");
        return std::nullopt;
    }

    const auto& concat_out_shape = concat_op->get_output_partial_shape(0);
    return ov::util::try_normalize_axis(concat_op->get_axis(), concat_out_shape.rank(), *concat_op);
}

std::optional<HostFlashAttention> HostFlashAttention::from(const std::shared_ptr<ov::Model>& model,
                                                           bool fused_flash_attention) {
    LOG_INFO("Attempting to create HostFlashAttention"
             << (fused_flash_attention ? " with fused flash attention node" : ""));
    LOG_BLOCK();

    // ========================================================================
    // Step 1: Validate SDPA pattern and extract key nodes
    // ========================================================================
    auto pattern_nodes = ov::npuw::util::find_sdpa_pattern_nodes(model);
    if (!pattern_nodes.is_valid()) {
        LOG_WARN("Failed to re-find SDPA pattern nodes");
        return std::nullopt;
    }

    auto q_input = pattern_nodes.matmul1_node->get_input_node_shared_ptr(0);
    auto k_concat = pattern_nodes.past_key_concat_node;

    // Skip Convert nodes to get to the actual Parameter/input
    q_input = skip_convert_nodes(q_input);

    if (!q_input || !k_concat) {
        LOG_WARN("Failed to extract Q input or K concat from pattern");
        return std::nullopt;
    }

    // ========================================================================
    // Step 2: Extract shape and data type information
    // ========================================================================
    auto q_shape = q_input->get_output_partial_shape(0);
    if (q_shape.is_dynamic()) {
        LOG_WARN("Dynamic shapes not yet supported for HFA");
        return std::nullopt;
    }

    auto q_shape_static = q_shape.to_shape();
    auto dtype = q_input->get_output_element_type(0);

    // Validate Q shape and extract query_size (seq_len dimension)
    if (q_shape_static.size() != 4) {
        LOG_WARN("Q shape must be 4D, got " << q_shape_static.size() << "D shape");
        return std::nullopt;
    }
    std::size_t query_size = q_shape_static[2];  // seq_len at index 2
    LOG_DEBUG("Extracted query_size (seq_len) from Q shape: " << query_size);

    auto mask_param = ov::npuw::util::find_mask_parameter(pattern_nodes.add_node);
    if (!mask_param) {
        LOG_WARN("Could not find mask parameter in model");
        return std::nullopt;
    }
    auto mask_dtype = mask_param->get_output_element_type(0);

    auto output_dtype = ov::element::f16;  // Default fallback
    if (model->outputs().size() > 0) {
        output_dtype = model->output(0).get_element_type();
        LOG_DEBUG("Original SDPA output data type: " << output_dtype);
    } else {
        LOG_WARN("No outputs found in model, using default output dtype: " << output_dtype);
    }

    // ========================================================================
    // Step 3: Extract K/V sequence dimensions from Concat nodes
    // ========================================================================
    auto k_seq_dim_opt = extract_sequence_dim_from_concat(pattern_nodes.past_key_concat_node, "K");
    if (!k_seq_dim_opt) {
        return std::nullopt;
    }
    std::size_t k_seq_dim = k_seq_dim_opt.value();

    auto v_seq_dim_opt = extract_sequence_dim_from_concat(pattern_nodes.past_value_concat_node, "V");
    if (!v_seq_dim_opt) {
        return std::nullopt;
    }
    std::size_t v_seq_dim = v_seq_dim_opt.value();

    // ========================================================================
    // Step 4: Extract KV heads configuration and context size
    // ========================================================================
    size_t kv_num_heads = 0;
    size_t context_size = 0;
    if (!k_concat->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }

    auto k_full_shape = k_concat->get_output_partial_shape(0).to_shape();
    // K shape after concat: [batch, kv_num_heads, kv_cache_size, head_dim]
    if (k_full_shape.size() != 4) {
        return std::nullopt;
    }

    kv_num_heads = k_full_shape[1];          // Extract kv_num_heads from K shape
    context_size = k_full_shape[k_seq_dim];  // Extract context size from sequence dimension

    if (kv_num_heads == 0) {
        LOG_WARN("Failed to determine KV num_heads");
        return std::nullopt;
    }

    if (context_size == 0) {
        LOG_WARN("Failed to determine context_size");
        return std::nullopt;
    }

    // ========================================================================
    // Step 5: Create tile models using query_size as tile_size
    // ========================================================================
    LOG_INFO("Creating HFA tile models with tile_size=" << query_size);
    auto tile_model = create_online_softmax_tile_model(q_shape_static,
                                            dtype,
                                            mask_dtype,
                                            query_size,
                                            kv_num_heads,
                                            false,
                                            fused_flash_attention);
    if (!tile_model) {
        LOG_WARN("Failed to create HFA tile model");
        return std::nullopt;
    }

    auto final_tile_model = create_online_softmax_tile_model(q_shape_static,
                                                  dtype,
                                                  mask_dtype,
                                                  query_size,
                                                  kv_num_heads,
                                                  true,
                                                  fused_flash_attention,
                                                  output_dtype);
    if (!final_tile_model) {
        LOG_WARN("Failed to create HFA final tile model");
        return std::nullopt;
    }

    // ========================================================================
    // Step 6: Create HostFlashAttention structure and set configuration
    // ========================================================================
    HostFlashAttention hfa;
    hfa._tile_model = tile_model;
    hfa._final_tile_model = final_tile_model;
    hfa._query_size = query_size;
    hfa._context_size = context_size;
    hfa._tile_size = query_size;
    hfa._k_seq_dim = k_seq_dim;
    hfa._v_seq_dim = v_seq_dim;

    // ========================================================================
    // Step 7: Build SDPA parameter index mapping
    // ========================================================================
    build_sdpa_param_mapping(hfa, model, pattern_nodes);

    // ========================================================================
    // Step 8: Build tile model parameter index mapping
    // ========================================================================
    build_tile_param_mapping(hfa, tile_model);

    // ========================================================================
    // Step 9: Build tile model output index mapping
    // ========================================================================
    build_tile_output_mapping(hfa, tile_model);

    LOG_INFO("Successfully created HostFlashAttention with query_size="
             << query_size << ", context_size=" << context_size << ", tile_size=" << query_size);

    return hfa;
}

}  // namespace function

namespace compiled {

// Constructor implementation - extracts metadata
HostFlashAttention::HostFlashAttention(const function::HostFlashAttention& func_hfa) {
    LOG_INFO("Constructing compiled::HostFlashAttention");
    LOG_BLOCK();

    // Extract tile configuration from function HFA
    _tile_size = func_hfa._tile_size;

    // Store the tile models for later compilation
    _tile_model_to_compile = func_hfa._tile_model;
    _final_tile_model_to_compile = func_hfa._final_tile_model;

    // Copy query size, context size, and K/V sequence dimensions from function HFA
    _sdpa_attention_info._query_size = func_hfa._query_size;
    _sdpa_attention_info._context_size = func_hfa._context_size;
    _sdpa_attention_info._k_seq_dim = func_hfa._k_seq_dim;
    _sdpa_attention_info._v_seq_dim = func_hfa._v_seq_dim;

    // Pre-cache all indices from function HFA maps
    LOG_INFO("Pre-caching SDPA and tile indices...");

    // Pre-cache SDPA parameter indices (direct field access — no map lookup)
    _sdpa_attention_info._sdpa_indices.query = func_hfa._query_param_idx;

    // Copy all KV cache block indices
    _sdpa_attention_info._sdpa_indices.past_key_blocks = func_hfa._past_key_block_indices;
    _sdpa_attention_info._sdpa_indices.past_value_blocks = func_hfa._past_value_block_indices;

    _sdpa_attention_info._sdpa_indices.present_key = func_hfa._present_key_param_idx;
    _sdpa_attention_info._sdpa_indices.present_value = func_hfa._present_value_param_idx;
    _sdpa_attention_info._sdpa_indices.attention_mask = func_hfa._attention_mask_param_idx;

    // Pre-cache tile input indices
    auto get_tile_input_idx = [&](OnlineSoftmaxTileInputId input_id) -> std::size_t {
        auto it = func_hfa._tile_param_index_map.find(input_id);
        if (it == func_hfa._tile_param_index_map.end()) {
            OPENVINO_THROW("HFA: Tile input mapping not found for input ID: ", static_cast<uint8_t>(input_id));
        }
        return it->second;
    };

    auto get_tile_output_idx = [&](OnlineSoftmaxTileOutputId output_id) -> std::size_t {
        auto it = func_hfa._tile_output_index_map.find(output_id);
        if (it == func_hfa._tile_output_index_map.end()) {
            OPENVINO_THROW("HFA: Tile output mapping not found for output ID: ", static_cast<uint8_t>(output_id));
        }
        return it->second;
    };

    // Cache all tile input indices
    _sdpa_attention_info._tile_input_indices.q = get_tile_input_idx(OnlineSoftmaxTileInputId::Q);
    _sdpa_attention_info._tile_input_indices.k = get_tile_input_idx(OnlineSoftmaxTileInputId::K_TILE);
    _sdpa_attention_info._tile_input_indices.v = get_tile_input_idx(OnlineSoftmaxTileInputId::V_TILE);
    _sdpa_attention_info._tile_input_indices.mask = get_tile_input_idx(OnlineSoftmaxTileInputId::MASK_TILE);
    _sdpa_attention_info._tile_input_indices.acc = get_tile_input_idx(OnlineSoftmaxTileInputId::PAST_ACC);
    _sdpa_attention_info._tile_input_indices.max = get_tile_input_idx(OnlineSoftmaxTileInputId::PAST_MAX);
    _sdpa_attention_info._tile_input_indices.d = get_tile_input_idx(OnlineSoftmaxTileInputId::PAST_D);

    // Cache all tile output indices
    _sdpa_attention_info._tile_output_indices.acc = get_tile_output_idx(OnlineSoftmaxTileOutputId::ACC);
    _sdpa_attention_info._tile_output_indices.max = get_tile_output_idx(OnlineSoftmaxTileOutputId::MAXX);
    _sdpa_attention_info._tile_output_indices.d = get_tile_output_idx(OnlineSoftmaxTileOutputId::D);

    LOG_INFO("Pre-cached SDPA indices: [query="
             << _sdpa_attention_info._sdpa_indices.query
             << ", present_key=" << _sdpa_attention_info._sdpa_indices.present_key
             << ", present_value=" << _sdpa_attention_info._sdpa_indices.present_value
             << ", attention_mask=" << _sdpa_attention_info._sdpa_indices.attention_mask << "]");
    LOG_INFO("  Past key blocks: " << _sdpa_attention_info._sdpa_indices.past_key_blocks.size());
    LOG_INFO("  Past value blocks: " << _sdpa_attention_info._sdpa_indices.past_value_blocks.size());
    LOG_INFO("Attention configuration: query_size="
             << _sdpa_attention_info._query_size << ", context_size=" << _sdpa_attention_info._context_size
             << ", k_seq_dim=" << _sdpa_attention_info._k_seq_dim << ", v_seq_dim=" << _sdpa_attention_info._v_seq_dim);
    LOG_INFO("Pre-cached tile indices: inputs[q=" << _sdpa_attention_info._tile_input_indices.q
                                                  << ", k=" << _sdpa_attention_info._tile_input_indices.k
                                                  << ", v=" << _sdpa_attention_info._tile_input_indices.v
                                                  << ", mask=" << _sdpa_attention_info._tile_input_indices.mask
                                                  << ", acc=" << _sdpa_attention_info._tile_input_indices.acc
                                                  << ", max=" << _sdpa_attention_info._tile_input_indices.max
                                                  << ", d=" << _sdpa_attention_info._tile_input_indices.d
                                                  << "], outputs[acc=" << _sdpa_attention_info._tile_output_indices.acc
                                                  << ", max=" << _sdpa_attention_info._tile_output_indices.max
                                                  << ", d=" << _sdpa_attention_info._tile_output_indices.d << "]");

    // Note: _compiled_tile_model and _compiled_final_tile_model will be set later by
    // compile_host_flash_attention_model()
}
}  // namespace compiled

namespace runtime {
namespace host_flash_attention {

// PositionIDs constructor
PositionIDs::PositionIDs(std::size_t param_idx, std::size_t query_size, const ov::ISyncInferRequest& rq)
    : _position_ids_idx(param_idx),
      _query_size(query_size),
      _rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    _case = _query_size == 1 ? Case::GENERATE : Case::PREFILL;
}

Selector::Ptr PositionIDs::find(std::size_t query_size, const ov::ISyncInferRequest& rq) {
    auto is_position_ids = [](const ov::Output<const ov::Node>& p) {
        const auto& shape = p.get_shape();
        // FIXME: 2D/3D position IDs are not supported here YET
        return p.get_node()->get_friendly_name() == "position_ids" &&
               (shape.size() == 1 || (shape.size() == 2 && shape[0] == 1));
    };

    const auto& inputs = rq.get_inputs();
    auto pos_ids_iter = std::find_if(inputs.begin(), inputs.end(), is_position_ids);
    if (pos_ids_iter != inputs.end()) {
        const auto param_idx = std::distance(inputs.begin(), pos_ids_iter);
        return Selector::Ptr{new PositionIDs(param_idx, query_size, rq)};
    }
    return Selector::Ptr{};
}

void PositionIDs::prepare(int64_t past_len) {
    const auto& iport = _rq.get().get_compiled_model()->inputs()[_position_ids_idx];
    const auto in_tensor = _rq.get().get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();

    // Same logic as regular attention PositionIDs
    auto* pos_data_ptr = in_tensor->data<int64_t>();
    for (int64_t idx = static_cast<int64_t>(in_dims.back()) - 1; idx >= 0; idx--) {
        if (pos_data_ptr[idx] > 0) {
            // Initialize fields
            _current_length = pos_data_ptr[idx];
            switch (_case) {
            case Case::GENERATE:
                // decode case, we have pos_id-1 past elements to take from kvcache
                _past_length = _current_length;
                break;
            case Case::PREFILL:
                // chunked prefill case. calculate the past_length in full chunks
                // FIXME: We know too much about chunking here
                _past_length = ((past_len + _query_size - 1) / _query_size) * _query_size;
                break;
            default:
                NPUW_ASSERT(false && "Reached the unreachable code");
            }
            return;
        }
    }
    LOG_WARN("Dynamic selector - no data found in the feature?");
    _current_length = -1;
}

int64_t PositionIDs::context_length() const {
    return _query_size + _past_length;
}

// ============================================================================
// HFARuntimeContext Implementation
// ============================================================================

void HFARuntimeContext::reset() {
    m_mask_tile_cache.clear();
    m_mask_tile_buffers.clear();
    m_state_buffers.reset();
    m_current_buffer_idx = 0;
}

ov::SoPtr<ov::ITensor> HFARuntimeContext::find_cached_mask_tile(const ov::SoPtr<ov::ITensor>& mask_tensor,
                                                                int64_t mask_offset,
                                                                int64_t tile_length) const {
    HFATileMaskKey cache_key{mask_tensor, mask_offset, tile_length};
    auto it = m_mask_tile_cache.find(cache_key);
    if (it != m_mask_tile_cache.end()) {
        return it->second;
    }
    return {};
}

ov::SoPtr<ov::ITensor> HFARuntimeContext::get_mask_tile_buffer(size_t index) const {
    if (index >= m_mask_tile_buffers.size()) {
        throw std::out_of_range("HFA: mask tile buffer index " + std::to_string(index) + " out of range [0, " +
                                std::to_string(m_mask_tile_buffers.size()) + ")");
    }
    return m_mask_tile_buffers[index];
}

void HFARuntimeContext::cache_mask_tile(const ov::SoPtr<ov::ITensor>& mask_tensor,
                                        int64_t mask_offset,
                                        int64_t tile_length,
                                        const ov::SoPtr<ov::ITensor>& cached_tile) {
    HFATileMaskKey cache_key{mask_tensor, mask_offset, tile_length};
    m_mask_tile_cache[cache_key] = cached_tile;
}

void HFARuntimeContext::clear_mask_cache() {
    m_mask_tile_cache.clear();
}

void HFARuntimeContext::initialize_state_tensors(ov::SoPtr<ov::ITensor>& acc,
                                                 ov::SoPtr<ov::ITensor>& max,
                                                 ov::SoPtr<ov::ITensor>& sum) {
    const auto type = acc->get_element_type();
    if (type == ov::element::f16) {
        std::memset(acc->data<ov::float16>(), 0, acc->get_byte_size());
        std::fill_n(max->data<ov::float16>(), max->get_size(), std::numeric_limits<ov::float16>::lowest());
        std::memset(sum->data<ov::float16>(), 0, sum->get_byte_size());
    } else if (type == ov::element::f32) {
        std::memset(acc->data<float>(), 0, acc->get_byte_size());
        std::fill_n(max->data<float>(), max->get_size(), std::numeric_limits<float>::lowest());
        std::memset(sum->data<float>(), 0, sum->get_byte_size());
    } else {
        throw std::runtime_error("HFA: Unsupported state tensor type");
    }
}

void HFARuntimeContext::prepare_next_state_buffers() {
    if (!m_state_buffers.has_value()) {
        return;
    }
    size_t next_idx = 1 - m_current_buffer_idx;
    auto& next_buffer = (*m_state_buffers)[next_idx];
    initialize_state_tensors(next_buffer.acc, next_buffer.max, next_buffer.sum);
}

void HFARuntimeContext::switch_buffers() {
    if (m_state_buffers.has_value()) {
        m_current_buffer_idx = 1 - m_current_buffer_idx;
    }
}

}  // namespace host_flash_attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
