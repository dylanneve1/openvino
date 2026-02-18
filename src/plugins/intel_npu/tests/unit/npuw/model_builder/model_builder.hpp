// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

// ============================================================================
// Result types (namespace-level)
// ============================================================================

struct KVCacheResult {
    ov::Output<ov::Node> concatenated;
    ov::Output<ov::Node> beam_gather;  // Past KV after beam reorder (before concat)
    std::shared_ptr<ov::Node> assign;  // Sink node for stateful model
};

/// Read-side state from a KV cache Variable (before concat/assign)
struct KVCacheReadState {
    std::shared_ptr<ov::op::util::Variable> variable;
    ov::Output<ov::Node> beam_gather;  // ReadValue -> Gather(beam_idx)
};

// ============================================================================
// Functor type aliases
// ============================================================================

/// Weight functor: construction captures format-specific params;
/// call takes (name, shape, compute_precision) -> output
using WeightFn = std::function<ov::Output<ov::Node>(const std::string&, const ov::Shape&, ov::element::Type)>;

/// Norm functor: construction captures hidden_size, precision, eps;
/// call takes (input, name) -> output
using NormFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;

/// FFN functor: construction captures hidden_size, intermediate_size, precision, weight_fn;
/// call takes (input, name) -> output
using FFNFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;

/// RoPE functor: construction captures head_dim, precision, and position_ids;
/// call takes (input, name) -> output.  Position IDs are baked in at construction.
using RoPEFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;

/// Layer function: takes (hidden_state, prefix, layer_idx), returns hidden state output.
using LayerFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&, size_t)>;

// ============================================================================
// Weight functors
// ============================================================================

/// Float weights (f32 or f16 storage) with Convert to compute precision when types differ
struct FloatWeight {
    ov::element::Type storage_type;

    FloatWeight(ov::element::Type st = ov::element::f32) : storage_type(st) {}

    ov::Output<ov::Node> operator()(const std::string& name,
                                    const ov::Shape& shape,
                                    ov::element::Type compute_precision) const;
};

/// Convenience aliases
using FP32Weight = FloatWeight;
struct FP16Weight : FloatWeight {
    FP16Weight() : FloatWeight(ov::element::f16) {}
};

/// Compressed integer weights -> Convert(f16) -> Multiply(f16 scale)
/// Matches DCOFF SymmNoZP pattern. Parameterized by storage element type (i8, i4, u4, etc.)
/// When group_size > 0, builds a group quantization pattern:
///   Constant(storage) -> Convert(f16) -> Reshape[rows, cols/gs, gs] -> Multiply(scale[rows, cols/gs, 1]) -> Reshape[rows, cols]
struct CompressedWeight {
    ov::element::Type storage_type;
    size_t group_size;  ///< 0 = per-channel scale, >0 = per-group scale

    explicit CompressedWeight(ov::element::Type st, size_t gs = 0) : storage_type(st), group_size(gs) {}

    ov::Output<ov::Node> operator()(const std::string& name,
                                    const ov::Shape& shape,
                                    ov::element::Type compute_precision) const;
};

/// i8 compressed weights (per-channel)
struct INT8Weight : CompressedWeight {
    INT8Weight() : CompressedWeight(ov::element::i8) {}
};

/// i4 compressed weights (per-channel)
struct INT4Weight : CompressedWeight {
    INT4Weight() : CompressedWeight(ov::element::i4) {}
};

/// i4 group-quantized weights
struct INT4GroupWeight : CompressedWeight {
    explicit INT4GroupWeight(size_t gs = 128) : CompressedWeight(ov::element::i4, gs) {}
};

// ============================================================================
// Norm functors
// ============================================================================

/// Layer normalization (MVN + scale + bias)
struct LayerNorm {
    size_t hidden_size;
    ov::element::Type precision;
    float eps;

    LayerNorm(size_t hs, ov::element::Type prec = ov::element::f32, float e = 1e-5f)
        : hidden_size(hs),
          precision(prec),
          eps(e) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

/// RMS normalization (scale only, no mean subtraction)
struct RMSNorm {
    size_t hidden_size;
    ov::element::Type precision;
    float eps;

    RMSNorm(size_t hs, ov::element::Type prec = ov::element::f32, float e = 1e-5f)
        : hidden_size(hs),
          precision(prec),
          eps(e) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

// ============================================================================
// RoPE functors
// ============================================================================

/// Half-rotation RoPE: split into first/second half, negate+swap, apply cos/sin.
/// Uses frequency-based Sin/Cos nodes. Position IDs are baked in at construction;
/// cos/sin are computed once and shared across all layers (like real models).
/// operator() only applies the rotation.
struct HalfRotationRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;  // pre-built from constructor

    HalfRotationRoPE(size_t head_dim,
                     ov::element::Type precision,
                     const ov::Output<ov::Node>& position_ids);

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input,
                                    const std::string& name) const;
};

/// Interleaved RoPE: interleave odd/even elements.
/// Uses frequency-based Sin/Cos nodes. Position IDs are baked in at construction;
/// cos/sin are computed once and shared across all layers (like real models).
/// operator() only applies the rotation.
struct InterleavedRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;  // pre-built from constructor

    InterleavedRoPE(size_t head_dim,
                    ov::element::Type precision,
                    const ov::Output<ov::Node>& position_ids);

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input,
                                    const std::string& name) const;
};

// ============================================================================
// Position IDs helpers
// ============================================================================

/// Create 2D position_ids Parameter [batch, seq].
/// The returned output IS the Parameter node (auto-tracked by build_llm's graph traversal).
ov::Output<ov::Node> make_position_ids_2d();

/// Create 3D position_ids Parameter [3, batch, seq] for m-rope (Qwen2.5-VL).
/// Returns a 2D [batch, seq] slice (section 0) for downstream RoPE consumption.
/// The 3D Parameter is auto-tracked by build_llm's graph traversal.
ov::Output<ov::Node> make_position_ids_3d();

// ============================================================================
// FFN functors
// ============================================================================

/// SwiGLU FFN: gate_proj * silu(up_proj), then down_proj
struct SwiGLU {
    size_t hidden_size;
    size_t intermediate_size;
    ov::element::Type precision;
    WeightFn weight_fn;

    SwiGLU(size_t hs, size_t is, ov::element::Type prec, WeightFn wf)
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

/// GELU FFN: up_proj -> GELU -> down_proj
struct GELUFn {
    size_t hidden_size;
    size_t intermediate_size;
    ov::element::Type precision;
    WeightFn weight_fn;
    WeightFn bias_fn;

    GELUFn(size_t hs, size_t is, ov::element::Type prec, WeightFn wf, WeightFn bf = {})
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)),
          bias_fn(std::move(bf)) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

// ============================================================================
// Free building-block functions
// ============================================================================

/// Linear projection (MatMul with weight + optional bias).
/// When bias_fn is provided, bias is created through it (matching weight compression pattern).
ov::Output<ov::Node> make_linear(const ov::Output<ov::Node>& input,
                                 size_t in_features,
                                 size_t out_features,
                                 const std::string& name,
                                 ov::element::Type precision = ov::element::f32,
                                 const WeightFn& weight_fn = FP32Weight{},
                                 const WeightFn& bias_fn = {});

/// Reshape for multi-head: [batch, seq, hidden] -> [batch, seq, heads, head_dim]
ov::Output<ov::Node> make_multihead_reshape(const ov::Output<ov::Node>& input,
                                            size_t num_heads,
                                            size_t head_dim,
                                            const std::string& name);

/// Transpose for attention: [batch, seq, heads, dim] <-> [batch, heads, seq, dim]
ov::Output<ov::Node> make_attention_transpose(const ov::Output<ov::Node>& input, const std::string& name);

/// Repeat KV heads for GQA: [batch, kv_heads, seq, dim] -> [batch, num_heads, seq, dim]
ov::Output<ov::Node> make_repeat_kv(const ov::Output<ov::Node>& kv,
                                    size_t num_heads,
                                    size_t num_kv_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    const ov::Output<ov::Node>& shared_broadcast_shape = {});

/// Create KV cache read state: Variable + init(zeros) + ReadValue + Gather(beam_idx)
KVCacheReadState make_kv_cache_read(const ov::Output<ov::Node>& batch_source,
                                    const ov::Output<ov::Node>& beam_idx,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision = ov::element::f32);

/// Create stateful KV cache pattern (ReadValue -> Gather(beam_idx) -> Concat -> Assign)
KVCacheResult make_kv_cache_concat(const ov::Output<ov::Node>& current_kv,
                                   const ov::Output<ov::Node>& batch_source,
                                   const ov::Output<ov::Node>& beam_idx,
                                   size_t num_heads,
                                   size_t head_dim,
                                   const std::string& name,
                                   ov::element::Type precision = ov::element::f32);

/// Compute scaled dot-product attention using native SDPA v13 op
/// When head_dim_for_scale > 0, creates a 5-input SDPA with explicit scale constant
/// (required for embedding model ReConstructEmbeddingModel pattern matching)
ov::Output<ov::Node> make_sdpa(const ov::Output<ov::Node>& q,
                               const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               const std::string& name,
                               const ov::Output<ov::Node>& attention_mask = ov::Output<ov::Node>(),
                               size_t head_dim_for_scale = 0);

/// Attention output: transpose back + reshape [batch, seq, hidden] + O projection
ov::Output<ov::Node> make_attention_output(const ov::Output<ov::Node>& sdpa_output,
                                           size_t hidden_size,
                                           const std::string& name,
                                           ov::element::Type precision,
                                           const WeightFn& weight_fn,
                                           const WeightFn& bias_fn = {});

/// Token embedding lookup
ov::Output<ov::Node> make_embedding(const ov::Output<ov::Node>& input_ids,
                                    size_t vocab_size,
                                    size_t hidden_size,
                                    const std::string& name,
                                    ov::element::Type precision = ov::element::f32);

/// LM head (linear projection to vocabulary)
ov::Output<ov::Node> make_lm_head(const ov::Output<ov::Node>& hidden_states,
                                  size_t hidden_size,
                                  size_t vocab_size,
                                  const std::string& name,
                                  ov::element::Type precision = ov::element::f32,
                                  const WeightFn& weight_fn = FP32Weight{});

/// 1D convolution: [batch, in_channels, length] -> [batch, out_channels, out_length]
/// Uses ov::op::v1::Convolution with bias Add.
ov::Output<ov::Node> make_conv1d(const ov::Output<ov::Node>& input,
                                 size_t in_channels,
                                 size_t out_channels,
                                 size_t kernel_size,
                                 size_t stride,
                                 size_t padding,
                                 const std::string& name,
                                 ov::element::Type precision = ov::element::f32);

/// Encoder KV cache (store-only, no concat): ReadValue(init) -> {SDPA, Assign}
/// For cross-attention where encoder KV is computed once and reused across decode steps.
/// No Gather(beam_idx) — encoder KV is identical across beams.
KVCacheResult make_encoder_kv_cache(const ov::Output<ov::Node>& encoder_kv,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision = ov::element::f32);

/// Run N transformer layers.
/// layer_fn is called for each layer with (hidden_state, prefix, layer_idx).
/// Returns final hidden state. Sinks are tracked via Attention::on_sink callback.
ov::Output<ov::Node> run_transformer_layers(
    const ov::Output<ov::Node>& initial,
    size_t num_layers,
    const std::string& prefix_base,
    const LayerFn& layer_fn);

// ============================================================================
// Attention functor
// ============================================================================

/// Unified attention mechanism: Q/K/V proj -> reshape -> optional QK-norm ->
/// transpose -> optional RoPE -> optional KV cache -> GQA repeat -> SDPA -> O proj.
/// Handles self-attention, cross-attention, and all cache modes.
struct Attention {
    // Dimensions
    size_t hidden_size, num_heads, num_kv_heads, head_dim;
    ov::element::Type precision;
    WeightFn weight_fn;

    // Per-projection options
    WeightFn bias_fn;  // Optional bias functor for Q/K/V/O projections (empty = no bias)
    NormFn qk_norm;    // Optional QK-norm applied to Q and K after reshape, before RoPE

    // RoPE (empty = no RoPE).  Position IDs are baked into the functor at construction.
    RoPEFn rope_fn;

    // KV cache config (architecture-level, not per-call)
    enum class CacheMode { None, ConcatBeam, StoreOnly };
    std::string cache_infix = ".";  // "." for LLM, ".decoder." / ".encoder." for whisper

    // Whisper layer-0: reuse pre-built key cache Variable to avoid duplicate variable_ids
    std::shared_ptr<ov::op::util::Variable> prebuilt_k_variable;
    ov::Output<ov::Node> prebuilt_k_beam_gather;

    // Mask
    ov::Output<ov::Node> sdpa_mask;

    // Shared GQA broadcast shape (embedding models): when set, always do Unsqueeze→Broadcast→Reshape
    // even for MHA (n_rep=1), using this shared Concat node so all SDPA nodes reference the same shape.
    ov::Output<ov::Node> shared_broadcast_shape;

    // Sink callback: called for each Assign node created by KV cache ops.
    // Set by ModelBuilder::setup_sink_tracking() to push into m_sinks.
    std::function<void(std::shared_ptr<ov::Node>)> on_sink;

    // Projection naming (defaults = LLM-style)
    std::string q_proj_name = "self_attn.q_proj";
    std::string k_proj_name = "self_attn.k_proj";
    std::string v_proj_name = "self_attn.v_proj";
    std::string o_proj_name = "self_attn.o_proj";

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input,
                                    const std::string& prefix,
                                    size_t layer_idx = 0,
                                    CacheMode cache_mode = CacheMode::None,
                                    const ov::Output<ov::Node>& kv_source = {},
                                    const ov::Output<ov::Node>& batch_source = {},
                                    const ov::Output<ov::Node>& beam_idx = {}) const;
};

// ============================================================================
// Template decoder layer composition
// ============================================================================

/// Whisper decoder layer: 3 sublayers (self-attn, cross-attn, FFN) with norm + residual.
/// Sinks are tracked via Attention::on_sink callback (not returned).
template <typename Norm, typename SelfAttn, typename CrossAttn, typename FFN>
ov::Output<ov::Node> make_whisper_decoder_layer(const ov::Output<ov::Node>& input,
                                                const Norm& norm,
                                                const SelfAttn& self_attn,
                                                const CrossAttn& cross_attn,
                                                const FFN& ffn,
                                                const std::string& prefix) {
    // Self-attention: norm -> self_attn -> residual
    auto normed1 = norm(input, prefix + "self_attn_layer_norm");
    auto self_attn_out = self_attn(normed1, prefix);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, self_attn_out);
    residual1->set_friendly_name(prefix + "self_attn_residual");

    // Cross-attention: norm -> cross_attn -> residual
    auto normed2 = norm(residual1->output(0), prefix + "encoder_attn_layer_norm");
    auto cross_attn_out = cross_attn(normed2, prefix);
    auto residual2 = std::make_shared<ov::opset11::Add>(residual1, cross_attn_out);
    residual2->set_friendly_name(prefix + "cross_attn_residual");

    // FFN: norm -> ffn -> residual
    auto normed3 = norm(residual2->output(0), prefix + "final_layer_norm");
    auto ffn_out = ffn(normed3, prefix + "fc");
    auto residual3 = std::make_shared<ov::opset11::Add>(residual2, ffn_out);
    residual3->set_friendly_name(prefix + "ffn_residual");

    return residual3->output(0);
}

/// Compose norm + attention + FFN + residual connections into a decoder layer.
/// Extra args are forwarded to the attention functor.
/// Sinks are tracked via Attention::on_sink callback (not returned).
template <typename Norm, typename Attn, typename FFN, typename... Args>
ov::Output<ov::Node> make_decoder_layer(const ov::Output<ov::Node>& input,
                                        const Norm& norm,
                                        const Attn& attention,
                                        const FFN& ffn,
                                        const std::string& prefix,
                                        Args&&... args) {
    // Pre-attention norm + attention + residual
    auto normed1 = norm(input, prefix + "input_layernorm");
    auto attn_out = attention(normed1, prefix, std::forward<Args>(args)...);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, attn_out);
    residual1->set_friendly_name(prefix + "attn_residual");

    // Post-attention norm + FFN + residual
    auto normed2 = norm(residual1->output(0), prefix + "post_attention_layernorm");
    auto ffn_out = ffn(normed2, prefix + "mlp");
    auto residual2 = std::make_shared<ov::opset11::Add>(residual1, ffn_out);
    residual2->set_friendly_name(prefix + "ffn_residual");

    return residual2->output(0);
}

/// BERT-style post-norm layer: attn -> Add(residual) -> Norm -> FFN -> Add(residual) -> Norm.
/// Norm is applied AFTER the residual connection (post-norm), unlike make_decoder_layer (pre-norm).
/// Sinks are tracked via Attention::on_sink callback (not returned).
template <typename Norm, typename Attn, typename FFN, typename... Args>
ov::Output<ov::Node> make_post_norm_layer(const ov::Output<ov::Node>& input,
                                          const Norm& norm,
                                          const Attn& attention,
                                          const FFN& ffn,
                                          const std::string& prefix,
                                          Args&&... args) {
    // Attention + residual + norm
    auto attn_out = attention(input, prefix, std::forward<Args>(args)...);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, attn_out);
    residual1->set_friendly_name(prefix + "attn_residual");
    auto normed1 = norm(residual1->output(0), prefix + "attention.output.LayerNorm");

    // FFN + residual + norm
    auto ffn_out = ffn(normed1, prefix + "output");
    auto residual2 = std::make_shared<ov::opset11::Add>(normed1, ffn_out);
    residual2->set_friendly_name(prefix + "ffn_residual");
    auto normed2 = norm(residual2->output(0), prefix + "output.LayerNorm");

    return normed2;
}

// ============================================================================
// ModelType enum
// ============================================================================

/// Model architecture type — drives build_model() behavior
enum class ModelType {
    LLM,               ///< Causal LM decoder (Llama, Qwen, Phi, etc.)
    VLM,               ///< Vision-language model (future; same decoder, different input pipeline)
    WhisperEncoder,    ///< Whisper encoder (conv frontend + self-attention layers)
    WhisperDecoder,    ///< Whisper decoder (self-attention + cross-attention layers)
    EmbeddingDecoder,  ///< Decoder-based embedding model (LLM without lm_head)
    EmbeddingEncoder,  ///< Encoder-based embedding model (BERT-style, post-norm)
};

// ============================================================================
// ModelConfig — unified configuration
// ============================================================================

/// Unified configuration for all model types.
///
/// Replaces LLMConfig, WhisperConfig, BERTConfig with a single struct.
/// The model_type field drives which build path is taken.
/// Fields irrelevant to a given model_type are simply ignored.
///
/// Declaration order matters for the default constructor: weight MUST be
/// declared before norm and ffn so that the member-initializer list can
/// reference it.
struct ModelConfig {
    // ----- Architecture selector -----
    ModelType model_type = ModelType::LLM;

    // ----- Core transformer dimensions -----
    size_t hidden_size = 64;       ///< d_model for Whisper
    size_t num_heads = 4;          ///< encoder_attention_heads / decoder_attention_heads for Whisper
    size_t head_dim = 16;
    size_t num_kv_heads = 0;       ///< 0 = MHA (same as num_heads)
    size_t intermediate_size = 256; ///< ffn_dim for Whisper
    size_t vocab_size = 1000;
    size_t num_layers = 2;

    // ----- Behavioral flags -----
    bool use_kv_cache = true;
    bool use_inputs_embeds = false;     ///< true = inputs_embeds parameter, false = input_ids + embedding
    bool use_lm_head = true;            ///< false = output last_hidden_state (embedding models)
    bool internal_position_ids = false;  ///< true = arange from input shape (no position_ids parameter)

    // ----- Precision -----
    ov::element::Type precision = ov::element::f32;

    // ----- Weight functors (declared BEFORE norm/ffn — C++ init order) -----
    WeightFn weight = FP32Weight{};
    WeightFn lm_head_weight;  ///< Empty = falls back to config.weight in build

    // ----- Functor fields — defaults derived from scalar fields -----
    NormFn norm;
    FFNFn ffn;
    RoPEFn rope;  ///< 2-arg: (input, name). Empty = auto HalfRotationRoPE. Set identity lambda to disable.

    /// Position IDs for RoPE. Empty = auto-creates [batch, seq] Parameter + HalfRotationRoPE.
    /// Provide a custom node (e.g. make_position_ids_3d(), Range subgraph) for non-standard cases.
    ov::Output<ov::Node> position_ids;

    NormFn qk_norm;  ///< Optional QK-norm forwarded to Attention

    // ----- Whisper-specific fields -----
    size_t encoder_layers = 0;       ///< 0 = use num_layers for encoder
    size_t decoder_layers = 0;       ///< 0 = use num_layers for decoder
    size_t num_mel_bins = 80;
    size_t max_source_positions = 1500;
    size_t max_target_positions = 448;

    // ----- BERT/Encoder-specific fields -----
    size_t max_position_embeddings = 512;
    size_t type_vocab_size = 2;

    // ----- Default constructor: derives norm/ffn from scalar fields -----
    ModelConfig() : norm(LayerNorm(hidden_size, precision)),
                    ffn(SwiGLU(hidden_size, intermediate_size, precision, weight)) {}

    // ----- Accessors -----

    size_t get_kv_heads() const {
        return num_kv_heads == 0 ? num_heads : num_kv_heads;
    }

    size_t get_encoder_layers() const {
        return encoder_layers == 0 ? num_layers : encoder_layers;
    }

    size_t get_decoder_layers() const {
        return decoder_layers == 0 ? num_layers : decoder_layers;
    }

    // ----- Factory methods: create Attention structs from config -----

    /// Self-attention: used by LLM, VLM, EmbeddingDecoder, WhisperEncoder, WhisperDecoder,
    /// and EmbeddingEncoder.  model_type determines naming, bias, cache_infix, etc.
    Attention make_self_attention() const;

    /// Cross-attention: used by WhisperDecoder.
    /// Returns Attention with encoder_attn.* projection names and ".encoder." cache_infix.
    Attention make_cross_attention() const;
};

// ============================================================================
// Deprecated config aliases — backward compatibility
// ============================================================================

/// \deprecated Use ModelConfig with model_type = LLM instead.
using LLMConfig = ModelConfig;

/// Whisper configuration — wraps a pair of ModelConfigs for encoder and decoder.
/// \deprecated Use ModelConfig with model_type = WhisperEncoder / WhisperDecoder instead.
struct WhisperConfig {
    size_t d_model = 384;
    size_t encoder_layers = 4;
    size_t decoder_layers = 4;
    size_t encoder_attention_heads = 6;
    size_t decoder_attention_heads = 6;
    size_t encoder_ffn_dim = 1536;
    size_t decoder_ffn_dim = 1536;
    size_t vocab_size = 51865;
    size_t num_mel_bins = 80;
    size_t max_source_positions = 1500;
    size_t max_target_positions = 448;
    ov::element::Type precision = ov::element::f32;
    WeightFn weight = FP32Weight{};

    size_t head_dim() const {
        return d_model / decoder_attention_heads;
    }

    /// Convert to unified ModelConfig for encoder
    ModelConfig to_encoder_config() const;

    /// Convert to unified ModelConfig for decoder
    ModelConfig to_decoder_config() const;

    /// \deprecated Use ModelConfig::make_self_attention() on to_encoder_config()
    Attention make_encoder_attention() const;

    /// \deprecated Use ModelConfig::make_self_attention() on to_decoder_config()
    Attention make_decoder_self_attention() const;

    /// \deprecated Use ModelConfig::make_cross_attention() on to_decoder_config()
    Attention make_decoder_cross_attention() const;
};

/// \deprecated Use ModelConfig with model_type = EmbeddingEncoder instead.
struct BERTConfig {
    size_t hidden_size = 768;
    size_t num_heads = 12;
    size_t head_dim = 64;
    size_t intermediate_size = 3072;
    size_t vocab_size = 30522;
    size_t num_layers = 12;
    size_t max_position_embeddings = 512;
    size_t type_vocab_size = 2;
    ov::element::Type precision = ov::element::f32;
    WeightFn weight = FP32Weight{};
    NormFn norm;
    FFNFn ffn;

    /// Default constructor: derives norm/ffn from scalar fields
    BERTConfig() : norm(LayerNorm(hidden_size, precision)),
                   ffn(GELUFn(hidden_size, intermediate_size, precision, weight, FloatWeight(precision))) {}

    /// Convert to unified ModelConfig
    ModelConfig to_config() const;

    /// \deprecated Use ModelConfig::make_self_attention() on to_config()
    Attention make_attention() const;
};

// ============================================================================
// ModelBuilder
// ============================================================================

/// Modular model builder for NPUW testing
///
/// Provides composable building blocks for constructing transformer models.
/// Variant behavior (norm, FFN, RoPE, weights) is controlled via functors
/// stored in ModelConfig, enabling extension without API changes.
///
/// Example with functor blocks:
///   ModelBuilder b;
///   auto input = b.parameter(ov::element::f32, {-1, -1, 64}, "input");
///   RMSNorm norm(64);
///   auto normed = norm(input->output(0), "norm");
///   b.result(normed, "output");
///   auto model = b.build("synthetic_model");
///
/// Unified entry point:
///   ModelConfig cfg;
///   cfg.model_type = ModelType::LLM;
///   cfg.hidden_size = 256;
///   auto model = ModelBuilder().build_model(cfg);
///
/// Or use the deprecated convenience methods:
///   auto model = b.build_llm(config);
class ModelBuilder {
public:
    ModelBuilder() = default;

    // ===== Simple test models (backward compatibility) =====
    std::shared_ptr<ov::Model> get_model_with_one_op();
    std::shared_ptr<ov::Model> get_model_without_repeated_blocks();
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks(std::size_t repetitions);
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks();
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_and_results(
        std::size_t repetitions,
        const std::vector<std::size_t>& block_indices);
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_and_parameters(
        std::size_t repetitions,
        const std::vector<std::size_t>& block_indices);
    std::shared_ptr<ov::Model> get_model_with_multi_output_repeating_blocks(std::size_t repetitions,
                                                                            bool last_block_has_direct_result);

    // ===== Builder interface =====

    /// Create and track a model parameter
    std::shared_ptr<ov::op::v0::Parameter> parameter(ov::element::Type type,
                                                     const ov::PartialShape& shape,
                                                     const std::string& name);

    /// Create and track a model result
    std::shared_ptr<ov::op::v0::Result> result(const ov::Output<ov::Node>& output, const std::string& name);

    /// Build model from all tracked parameters and results
    std::shared_ptr<ov::Model> build(const std::string& name = "");

    // ===== Unified build entry point =====

    /// Build a complete model from a unified ModelConfig.
    /// model_type drives the architecture: LLM, WhisperEncoder, WhisperDecoder,
    /// EmbeddingDecoder, EmbeddingEncoder, VLM.
    std::shared_ptr<ov::Model> build_model(const ModelConfig& config);

    // ===== Deprecated type-specific builders (backward compatibility) =====

    /// \deprecated Use build_model() with model_type = LLM
    std::shared_ptr<ov::Model> build_llm(const LLMConfig& config = LLMConfig{});

    /// \deprecated Use build_model() with model_type = WhisperEncoder
    std::shared_ptr<ov::Model> build_whisper_encoder(const WhisperConfig& config);

    /// \deprecated Use build_model() with model_type = WhisperDecoder
    std::shared_ptr<ov::Model> build_whisper_decoder(const WhisperConfig& config);

    /// \deprecated Use build_model() with model_type = EmbeddingEncoder
    std::shared_ptr<ov::Model> build_bert_encoder(const BERTConfig& config);

    // ===== State management =====

    void clear();

private:
    /// Set up position IDs: handles internal/custom/auto-create modes.
    /// May auto-create HalfRotationRoPE on config.rope.
    ov::Output<ov::Node> setup_position_ids(ModelConfig& config, const ov::Output<ov::Node>& seq_source);

    /// Set Attention::on_sink to push into m_sinks
    void setup_sink_tracking(Attention& attn);

    /// Create Result + build ov::Model with auto-discovered Parameters
    std::shared_ptr<ov::Model> make_model(const ov::Output<ov::Node>& output,
                                           const std::string& result_name,
                                           const std::string& model_name);

    // ===== Per-type build implementations (called by build_model) =====
    std::shared_ptr<ov::Model> build_llm_impl(const ModelConfig& config);
    std::shared_ptr<ov::Model> build_whisper_encoder_impl(const ModelConfig& config);
    std::shared_ptr<ov::Model> build_whisper_decoder_impl(const ModelConfig& config);
    std::shared_ptr<ov::Model> build_embedding_encoder_impl(const ModelConfig& config);

    std::shared_ptr<ov::Node> get_block(const std::shared_ptr<ov::Node>& input);
    void set_name(const std::shared_ptr<ov::Node>& node);

    std::vector<std::shared_ptr<ov::Node>> m_nodes;
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> m_parameters;
    std::vector<std::shared_ptr<ov::op::v0::Result>> m_results;
    ov::SinkVector m_sinks;
    size_t m_name_idx = 0;
};

}  // namespace npuw
}  // namespace test
}  // namespace ov
