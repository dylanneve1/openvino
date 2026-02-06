// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Utility to generate synthetic LLM models for NPUW testing
// Outputs OpenVINO IR models compatible with llm_bench and LLMPipeline
//
// Usage: npuw_model_generator_demo --type llm [options]

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

#include "model_builder.hpp"
#include "openvino/pass/serialize.hpp"

namespace fs = std::filesystem;

using namespace ov::test::npuw;

static void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " --type <type> [options]\n\n";
    std::cout << "Generate synthetic OpenVINO IR models for NPUW testing.\n\n";
    std::cout << "Required:\n";
    std::cout << "  --type <type>           Model type: llm\n\n";
    std::cout << "Output:\n";
    std::cout << "  -o, --output <dir>      Output directory (default: npuw_test_models)\n";
    std::cout << "  -n, --name <name>       Model subdirectory name (default: model)\n\n";
    std::cout << "LLM options (--type llm):\n";
    std::cout << "  --num-layers <N>        Number of decoder layers (default: 2)\n";
    std::cout << "  --hidden-size <N>       Hidden dimension (default: 256)\n";
    std::cout << "  --num-heads <N>         Number of attention heads (default: 8)\n";
    std::cout << "  --num-kv-heads <N>      Number of KV heads, 0=MHA (default: 0)\n";
    std::cout << "  --head-dim <N>          Head dimension (default: 32)\n";
    std::cout << "  --intermediate-size <N> FFN intermediate size (default: 1024)\n";
    std::cout << "  --vocab-size <N>        Vocabulary size (default: 32000)\n";
    std::cout << "  --context-len <N>       Max context length (default: 128)\n";
    std::cout << "  --ffn-type <type>       FFN type: swiglu, gelu (default: swiglu)\n";
    std::cout << "  --norm-type <type>      Normalization: rms, layer (default: rms)\n";
    std::cout << "  --rope-type <type>      RoPE type: half, interleaved, none (default: half)\n";
    std::cout << "  --no-kv-cache           Disable KV cache (stateless model)\n\n";
    std::cout << "General:\n";
    std::cout << "  -h, --help              Show this help\n";
}

static void write_config_json(const fs::path& dir, const LLMConfig& config,
                              const TokenizerConfig& tok_config) {
    // config.json (HuggingFace model config)
    {
        std::ofstream ofs(dir / "config.json");
        ofs << "{\n";
        ofs << "  \"model_type\": \"llama\",\n";
        ofs << "  \"hidden_size\": " << config.hidden_size << ",\n";
        ofs << "  \"num_attention_heads\": " << config.num_heads << ",\n";
        ofs << "  \"num_key_value_heads\": " << config.get_kv_heads() << ",\n";
        ofs << "  \"num_hidden_layers\": " << config.num_layers << ",\n";
        ofs << "  \"intermediate_size\": " << config.intermediate_size << ",\n";
        ofs << "  \"vocab_size\": " << config.vocab_size << ",\n";
        ofs << "  \"max_position_embeddings\": " << config.context_len << ",\n";
        ofs << "  \"bos_token_id\": " << tok_config.bos_token_id << ",\n";
        ofs << "  \"eos_token_id\": " << tok_config.eos_token_id << ",\n";
        ofs << "  \"pad_token_id\": " << tok_config.pad_token_id << "\n";
        ofs << "}\n";
    }

    // generation_config.json (required by GenAI LLMPipeline)
    {
        std::ofstream ofs(dir / "generation_config.json");
        ofs << "{\n";
        ofs << "  \"bos_token_id\": " << tok_config.bos_token_id << ",\n";
        ofs << "  \"eos_token_id\": " << tok_config.eos_token_id << ",\n";
        ofs << "  \"pad_token_id\": " << tok_config.pad_token_id << ",\n";
        ofs << "  \"max_length\": 4096\n";
        ofs << "}\n";
    }
}

static void serialize_model(const std::shared_ptr<ov::Model>& model, const fs::path& dir,
                            const LLMConfig& config, const TokenizerConfig& tok_config,
                            const TokenizerResult& tok_result) {
    fs::create_directories(dir);

    // Serialize LLM model
    auto xml_path = dir / "openvino_model.xml";
    auto bin_path = dir / "openvino_model.bin";
    ov::pass::Serialize serializer(xml_path.string(), bin_path.string());
    serializer.run_on_model(model);
    write_config_json(dir, config, tok_config);

    // Serialize tokenizer/detokenizer if available
    if (tok_result.tokenizer) {
        ov::pass::Serialize tok_ser((dir / "openvino_tokenizer.xml").string(),
                                    (dir / "openvino_tokenizer.bin").string());
        tok_ser.run_on_model(tok_result.tokenizer);
    }
    if (tok_result.detokenizer) {
        ov::pass::Serialize detok_ser((dir / "openvino_detokenizer.xml").string(),
                                      (dir / "openvino_detokenizer.bin").string());
        detok_ser.run_on_model(tok_result.detokenizer);
    }

    // Write tokenizer config files
    ModelBuilder::write_tokenizer_configs(dir, tok_config);

    std::cout << "Model saved to: " << dir.string() << "/\n";
}

static bool parse_size_t(const char* str, size_t& out) {
    try {
        out = std::stoull(str);
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char* argv[]) {
    fs::path output_dir = "npuw_test_models";
    std::string model_name = "model";
    std::string model_type;

    // LLM config defaults (Llama-like)
    LLMConfig config;
    config.num_layers = 2;
    config.hidden_size = 256;
    config.num_heads = 8;
    config.head_dim = 32;
    config.num_kv_heads = 0;
    config.intermediate_size = 1024;
    config.vocab_size = 32000;
    config.context_len = 128;
    config.ffn_type = FFNType::SWIGLU;
    config.norm_type = NormType::RMS_NORM;
    config.rope_type = RoPEType::HALF_ROTATION;
    config.use_kv_cache = true;
    config.use_position_ids = true;
    config.weight_format = WeightFormat::FP32;
    config.lm_head_format = WeightFormat::FP32;
    config.precision = ov::element::f32;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--type" && i + 1 < argc) {
            model_type = argv[++i];
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if ((arg == "--name" || arg == "-n") && i + 1 < argc) {
            model_name = argv[++i];
        } else if (arg == "--num-layers" && i + 1 < argc) {
            if (!parse_size_t(argv[++i], config.num_layers)) {
                std::cerr << "Error: invalid value for --num-layers\n";
                return 1;
            }
        } else if (arg == "--hidden-size" && i + 1 < argc) {
            if (!parse_size_t(argv[++i], config.hidden_size)) {
                std::cerr << "Error: invalid value for --hidden-size\n";
                return 1;
            }
        } else if (arg == "--num-heads" && i + 1 < argc) {
            if (!parse_size_t(argv[++i], config.num_heads)) {
                std::cerr << "Error: invalid value for --num-heads\n";
                return 1;
            }
        } else if (arg == "--num-kv-heads" && i + 1 < argc) {
            if (!parse_size_t(argv[++i], config.num_kv_heads)) {
                std::cerr << "Error: invalid value for --num-kv-heads\n";
                return 1;
            }
        } else if (arg == "--head-dim" && i + 1 < argc) {
            if (!parse_size_t(argv[++i], config.head_dim)) {
                std::cerr << "Error: invalid value for --head-dim\n";
                return 1;
            }
        } else if (arg == "--intermediate-size" && i + 1 < argc) {
            if (!parse_size_t(argv[++i], config.intermediate_size)) {
                std::cerr << "Error: invalid value for --intermediate-size\n";
                return 1;
            }
        } else if (arg == "--vocab-size" && i + 1 < argc) {
            if (!parse_size_t(argv[++i], config.vocab_size)) {
                std::cerr << "Error: invalid value for --vocab-size\n";
                return 1;
            }
        } else if (arg == "--context-len" && i + 1 < argc) {
            if (!parse_size_t(argv[++i], config.context_len)) {
                std::cerr << "Error: invalid value for --context-len\n";
                return 1;
            }
        } else if (arg == "--ffn-type" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "swiglu") {
                config.ffn_type = FFNType::SWIGLU;
            } else if (val == "gelu") {
                config.ffn_type = FFNType::GELU;
            } else {
                std::cerr << "Error: unknown --ffn-type '" << val << "' (expected: swiglu, gelu)\n";
                return 1;
            }
        } else if (arg == "--norm-type" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "rms") {
                config.norm_type = NormType::RMS_NORM;
            } else if (val == "layer") {
                config.norm_type = NormType::LAYER_NORM;
            } else {
                std::cerr << "Error: unknown --norm-type '" << val << "' (expected: rms, layer)\n";
                return 1;
            }
        } else if (arg == "--rope-type" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "half") {
                config.rope_type = RoPEType::HALF_ROTATION;
                config.use_position_ids = true;
            } else if (val == "interleaved") {
                config.rope_type = RoPEType::INTERLEAVED;
                config.use_position_ids = true;
            } else if (val == "none") {
                config.use_position_ids = false;
            } else {
                std::cerr << "Error: unknown --rope-type '" << val << "' (expected: half, interleaved, none)\n";
                return 1;
            }
        } else if (arg == "--no-kv-cache") {
            config.use_kv_cache = false;
        } else {
            std::cerr << "Error: unknown option '" << arg << "'\n";
            std::cerr << "Run with --help for usage information.\n";
            return 1;
        }
    }

    // Validate required arguments
    if (model_type.empty()) {
        std::cerr << "Error: --type is required\n";
        std::cerr << "Run with --help for usage information.\n";
        return 1;
    }

    if (model_type != "llm") {
        std::cerr << "Error: unsupported model type '" << model_type << "'\n";
        std::cerr << "Currently supported types: llm\n";
        return 1;
    }

    // Print configuration summary
    auto ffn_str = config.ffn_type == FFNType::SWIGLU ? "swiglu" : "gelu";
    auto norm_str = config.norm_type == NormType::RMS_NORM ? "rms" : "layer";
    auto rope_str = !config.use_position_ids ? "none"
                    : config.rope_type == RoPEType::HALF_ROTATION ? "half" : "interleaved";

    std::cout << "Generating LLM model:\n";
    std::cout << "  layers:            " << config.num_layers << "\n";
    std::cout << "  hidden_size:       " << config.hidden_size << "\n";
    std::cout << "  num_heads:         " << config.num_heads << "\n";
    std::cout << "  num_kv_heads:      " << config.get_kv_heads()
              << (config.num_kv_heads == 0 ? " (MHA)" : " (GQA)") << "\n";
    std::cout << "  head_dim:          " << config.head_dim << "\n";
    std::cout << "  intermediate_size: " << config.intermediate_size << "\n";
    std::cout << "  vocab_size:        " << config.vocab_size << "\n";
    std::cout << "  context_len:       " << config.context_len << "\n";
    std::cout << "  ffn:               " << ffn_str << "\n";
    std::cout << "  norm:              " << norm_str << "\n";
    std::cout << "  rope:              " << rope_str << "\n";
    std::cout << "  kv_cache:          " << (config.use_kv_cache ? "yes" : "no") << "\n";
    std::cout << "\n";

    ModelBuilder mb;

    // Build tokenizer
    TokenizerConfig tok_config;
    tok_config.vocab_size = config.vocab_size;
    auto tok_result = mb.build_tokenizer(tok_config);
    if (tok_result.tokenizer) {
        std::cout << "Tokenizer generated (libopenvino_tokenizers found)\n";
    } else {
        std::cerr << "Warning: libopenvino_tokenizers not found, skipping tokenizer generation\n";
    }

    // Build and serialize model
    auto model = mb.build_llm(config);
    serialize_model(model, output_dir / model_name, config, tok_config, tok_result);

    std::cout << "Done.\n";
    return 0;
}
