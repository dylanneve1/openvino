// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "compiled_model.hpp"
#include "npuw_transformations/kv_axes_position.hpp"
#include "paged_attn_runtime.hpp"

namespace ov {
namespace npuw {

class KVCacheBlockManager;
class LLMInferRequest;
class WhisperInferRequest;
class BlockKVCacheExtension;
struct PrefixCacheRestorationContext;
class LLMCompiledModel : public ov::npuw::ICompiledModel {
    using GetPropertiesMap =
        std::map<std::string, std::tuple<ov::PropertyMutability, std::function<ov::Any(const ::intel_npu::Config&)>>>;

public:
    static constexpr const char* output_embeds = "npuw_output_embed";

    static constexpr uint32_t whisper_batch_dim = 0u;
    static constexpr uint32_t whisper_seq_len_dim = 2u;
    static constexpr uint32_t whisper_max_prompt_size = 4u;
    static constexpr uint32_t whisper_kvcache_size = 448u;

    struct KVCacheDesc {
        uint32_t max_prompt_size = 0u;
        uint32_t total_size = 0u;
        uint32_t num_stored_tokens = 0u;
        uint32_t dim = 0u;
        uint32_t max_generation_token_len = 0u;
        bool v_tensors_transposed_pre = false;  // prefill
        bool v_tensors_transposed_gen = false;  // generate
    };

    // Factory type for creating sub-compiled-models (prefill / generate / lm_head).
    // The default builds an ov::npuw::CompiledModel via ICompiledModel::create().
    // Tests may inject a custom factory to inspect the transformed IR or stub the
    // compilation stage entirely.
    using CompiledModelFactory =
        std::function<std::shared_ptr<ov::npuw::ICompiledModel_v0>(const std::shared_ptr<ov::Model>&,
                                                                   const std::shared_ptr<const ov::IPlugin>&,
                                                                   const ov::AnyMap&)>;

    static std::shared_ptr<ov::npuw::ICompiledModel_v0> make_compiled_model(
        const std::shared_ptr<ov::Model>& model,
        const std::shared_ptr<const ov::IPlugin>& plugin,
        const ov::AnyMap& properties);

    LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     const ov::AnyMap& properties,
                     CompiledModelFactory factory = make_compiled_model);
    LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     const bool serialized);
    LLMCompiledModel() = delete;

    void export_model(std::ostream& model) const override;
    static std::shared_ptr<LLMCompiledModel> import_model(std::istream& stream,
                                                          const std::shared_ptr<const ov::IPlugin>& plugin,
                                                          const ov::AnyMap& properties);

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    friend class LLMInferBaseRequest;
    friend class LLMInferRequest;
    friend class PagedLLMInferRequest;
    friend class WhisperInferRequest;
    friend class EmbeddingInferRequest;
    friend class BlockKVCacheExtension;

    std::shared_ptr<ov::ISyncInferRequest> create_llm_infer_request();
    std::shared_ptr<ov::ISyncInferRequest> create_whisper_infer_request();
    std::shared_ptr<ov::ISyncInferRequest> create_embedding_infer_request();
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    void implement_properties();

    void serialize(std::ostream& stream, const ov::npuw::s11n::CompiledContext& ctx) const;
    static std::shared_ptr<LLMCompiledModel> deserialize(std::istream& stream,
                                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                                         const ov::AnyMap& properties,
                                                         const ov::npuw::s11n::CompiledContext& ctx);

    std::string m_name;
    std::shared_ptr<::intel_npu::OptionsDesc> m_options_desc;
    ::intel_npu::Config m_cfg;
    GetPropertiesMap m_prop_to_opt;
    ov::AnyMap m_non_llm_props;

    // Cache bf16 constants for weightless deserialization
    ov::npuw::s11n::BF16Cache m_bf16_consts;

    KVCacheDesc m_kvcache_desc;
    uint64_t m_prefill_chunk_size = 0;
    bool m_use_chunk_prefill = false;
    std::shared_ptr<ov::npuw::ICompiledModel_v0> m_kvcache_compiled;
    std::shared_ptr<ov::npuw::ICompiledModel_v0> m_prefill_compiled;
    // This model is optional, so can be null.
    std::shared_ptr<ov::npuw::ICompiledModel_v0> m_lm_head_compiled;

    CompiledModelFactory m_compiled_model_factory;

    // Multiple generate models with different static KV cache shapes (1K, 2K, 4K, 8K stepping)
    std::vector<std::shared_ptr<ov::npuw::ICompiledModel_v0>> m_generate_compiled_variants;
    std::vector<uint32_t> m_kvcache_sizes;  // Corresponding KV cache sizes for each variant

    // Support LoRA
    uint32_t m_max_lora_rank = 32;

    // Support prefix caching
    bool m_enable_prefix_caching = false;
    uint64_t m_prefix_caching_block_size = 0;
    uint64_t m_prefix_caching_max_num_blocks = 0;

    // Friend declarations for PrefixCachingHelper to access protected members
    friend class PrefixCachingHelper;

    bool m_is_whisper = false;
    uint64_t m_eos_token_id = 0;
    size_t m_decomposed_sdpa_size = 0;

    bool m_is_embedding = false;

    // Create generate model variants with different sizes
    std::vector<std::shared_ptr<ov::Model>> create_generate_model_variants(
        const std::shared_ptr<ov::Model>& generate_model,
        const KVAxesPosition& axes,
        const uint32_t whisper_lhs_seq_size);

    // Compile multiple generate model variants
    void compile_generate_model_variants(const std::vector<std::shared_ptr<ov::Model>>& generate_model_variants,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const ov::AnyMap& generate_config);

    bool m_is_eagle = false;

    // Paged Attention path.
    // m_pa_mode is true when either PREFILL or GENERATE attention hint is PAGED,
    // or when SDPAToPagedAttention has already been applied upstream by GenAI's
    // ContinuousBatchingPipeline before core.compile_model().
    // m_pa_npu_lowering mirrors NPUW_ATTN_PAGED_NPU_LOWERING — affects how
    // PagedLLMInferRequest pads its prefill metadata.
    bool m_pa_mode = false;
    bool m_pa_npu_lowering = false;
    uint32_t m_pa_block_size = 32;
    uint32_t m_pa_num_blocks = 1024;
    uint32_t m_pa_max_seqs = 16;

    // Per-layer K/V block pools for the NPU-resident PagedAttention lowering
    // (see PAGED_ATTN_LOWERING_DESIGN.md). One entry per transformer layer,
    // populated by setup_paged_block_managers() when m_pa_mode is true.
    // Shared with PagedLLMInferRequest via the existing friend declaration.
    struct PagedLayerManagers {
        std::shared_ptr<KVCacheBlockManager> key;
        std::shared_ptr<KVCacheBlockManager> value;
    };
    std::vector<PagedLayerManagers> m_pa_layer_managers;

    // Discovers key_cache.N / value_cache.N Parameters on the given model
    // (left there by SDPAToPagedAttention + BakePagedAttentionStaticShapes)
    // and constructs one KVCacheBlockManager pair per layer. Safe to call
    // multiple times — repopulates m_pa_layer_managers. No-op when
    // m_pa_mode is false.
    void setup_paged_block_managers(const std::shared_ptr<ov::Model>& shape_source_model,
                                    const std::shared_ptr<const ov::IPlugin>& plugin,
                                    const std::string& device);

    // PA NPU-resident lowering. Populated when m_pa_mode is true and the
    // tile sub-models build + compile successfully. Consumed by
    // PagedLLMInferRequest's tile-loop orchestrator (Step 5).
    // When m_pa_runtime is set but compiled handles are null, the lowering
    // bailed during compilation and the caller should fall back to the
    // existing CPU PA execution path.
    std::optional<function::PagedAttention> m_pa_runtime;
    std::optional<function::PagedAttention> m_pa_runtime_generate;
    // PA tile sub-models compile via ov::Core directly (not via the NPUW
    // factory) so the full NPUW pipeline doesn't recurse into a leaf model.
    // Prefill and generate variants differ only in q_size — both are needed
    // because the partitioner-scope handoff stamps compiled::PagedAttention
    // for each Function with whichever scope was active at compile time.
    ov::SoPtr<ov::ICompiledModel> m_pa_tile_compiled_ext;
    ov::SoPtr<ov::ICompiledModel> m_pa_final_tile_compiled_ext;
    ov::SoPtr<ov::ICompiledModel> m_pa_tile_compiled_ext_generate;
    ov::SoPtr<ov::ICompiledModel> m_pa_final_tile_compiled_ext_generate;

    // Builds the function::PagedAttention runtime extension from
    // `shape_source_model` (post-Bake prefill or generate model) and
    // compiles both tile sub-models on `plugin` with `tile_compile_config`.
    // No-op when m_pa_mode is false. On any failure (no PA op detected,
    // tile-model construction failure, or compile error) the partial state
    // is rolled back and the existing CPU PA fallback remains in effect.
    enum class PagedRuntimeKind { Prefill, Generate };
    void setup_paged_runtime(PagedRuntimeKind kind,
                             const std::shared_ptr<ov::Model>& shape_source_model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const ov::AnyMap& tile_compile_config);
};

}  // namespace npuw
}  // namespace ov
