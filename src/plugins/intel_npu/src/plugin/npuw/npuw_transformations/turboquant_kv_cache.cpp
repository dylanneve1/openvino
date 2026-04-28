// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "turboquant_kv_cache.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/subtract.hpp"

namespace {

// ---------------------------------------------------------------------------
// Constants matching ov::npuw::run_kv_cache_dynamic_quantization_passes() so
// downstream code that filters by these substrings keeps working.
// ---------------------------------------------------------------------------
constexpr auto g_key_cache_name = "key";
constexpr auto g_value_cache_name = "value";
constexpr auto g_past_name = "past_key_values";
constexpr auto g_present_name = "present";

std::string cache_name(bool is_key) {
    return is_key ? g_key_cache_name : g_value_cache_name;
}

bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// ---------------------------------------------------------------------------
// Hadamard (Sylvester construction).
//
// H_n = recursive [[H_{n/2}, H_{n/2}], [H_{n/2}, -H_{n/2}]], scaled by
// 1/sqrt(n) so H * H^T = I. Hadamard is symmetric (H == H^T), so the same
// constant is its own inverse and can be reused on quant and dequant sides.
//
// Requires n to be a power of two; caller falls back to identity otherwise.
// ---------------------------------------------------------------------------
std::shared_ptr<ov::op::v0::Constant> make_hadamard_constant(size_t n, ov::element::Type element_type) {
    OPENVINO_ASSERT(is_power_of_two(n), "Hadamard size must be a power of two, got ", n);

    std::vector<float> H(n * n, 0.0f);
    H[0] = 1.0f;
    for (size_t step = 1; step < n; step *= 2) {
        for (size_t i = 0; i < step; ++i) {
            for (size_t j = 0; j < step; ++j) {
                const float v = H[i * n + j];
                H[i * n + (j + step)] = v;
                H[(i + step) * n + j] = v;
                H[(i + step) * n + (j + step)] = -v;
            }
        }
    }
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(n));
    for (auto& x : H) {
        x *= inv_sqrt_n;
    }

    if (element_type == ov::element::f32) {
        return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{n, n}, H);
    }
    if (element_type == ov::element::f16) {
        std::vector<ov::float16> H16(H.size());
        for (size_t i = 0; i < H.size(); ++i) {
            H16[i] = ov::float16(H[i]);
        }
        return ov::op::v0::Constant::create(ov::element::f16, ov::Shape{n, n}, H16);
    }
    OPENVINO_THROW("Unsupported element type for Hadamard constant: ", element_type);
}

// ---------------------------------------------------------------------------
// Lloyd-Max codebooks for unit-variance Gaussian input.
//
// After Hadamard rotation along head_dim, by the central limit theorem the
// per-coordinate distribution is approximately Gaussian. We use the
// well-known Lloyd-Max optimal levels for unit-variance Gaussian (Max 1960).
// The TurboQuant paper claims a Beta distribution after rotation; using
// Gaussian levels here is a reasonable demo approximation that removes any
// data-dependent calibration step.
// ---------------------------------------------------------------------------
std::vector<float> lloyd_max_levels_gaussian(uint8_t bits) {
    switch (bits) {
        case 1:
            return {-0.7979f, 0.7979f};
        case 2:
            return {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
        case 3:
            return {-2.1519f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1519f};
        case 4:
            return {-2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
                    0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f};
        default: {
            std::vector<float> levels;
            const size_t k = 1u << bits;
            for (size_t i = 0; i < k; ++i) {
                const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(k);
                levels.push_back(-3.0f + 6.0f * t);
            }
            return levels;
        }
    }
}

std::shared_ptr<ov::op::v0::Constant> make_codebook_constant(uint8_t bits, ov::element::Type element_type) {
    auto levels = lloyd_max_levels_gaussian(bits);
    if (element_type == ov::element::f32) {
        return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{levels.size()}, levels);
    }
    if (element_type == ov::element::f16) {
        std::vector<ov::float16> levels16(levels.size());
        for (size_t i = 0; i < levels.size(); ++i) {
            levels16[i] = ov::float16(levels[i]);
        }
        return ov::op::v0::Constant::create(ov::element::f16, ov::Shape{levels.size()}, levels16);
    }
    OPENVINO_THROW("Unsupported element type for codebook: ", element_type);
}

// ---------------------------------------------------------------------------
// QJL random {+1,-1} projection. Deterministic — keyed on a fixed seed so
// the rewrite is reproducible across compiles.
// ---------------------------------------------------------------------------
std::shared_ptr<ov::op::v0::Constant> make_qjl_projection_constant(size_t head_dim,
                                                                   size_t qjl_dim,
                                                                   ov::element::Type element_type,
                                                                   uint64_t seed = 0xA17F00DCAFEULL) {
    std::mt19937_64 rng(seed);
    std::vector<float> R(head_dim * qjl_dim, 0.0f);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(qjl_dim));
    for (auto& v : R) {
        v = (rng() & 1u) ? inv_sqrt : -inv_sqrt;
    }
    if (element_type == ov::element::f32) {
        return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{head_dim, qjl_dim}, R);
    }
    if (element_type == ov::element::f16) {
        std::vector<ov::float16> R16(R.size());
        for (size_t i = 0; i < R.size(); ++i) {
            R16[i] = ov::float16(R[i]);
        }
        return ov::op::v0::Constant::create(ov::element::f16, ov::Shape{head_dim, qjl_dim}, R16);
    }
    OPENVINO_THROW("Unsupported element type for QJL constant: ", element_type);
}

// Apply Hadamard rotation along the last axis of `x`. Returns x unchanged
// (with a warning) if head_dim is dynamic or not a power of two.
ov::Output<ov::Node> apply_hadamard(const ov::Output<ov::Node>& x, const std::string& name_prefix) {
    const auto& shape = x.get_partial_shape();
    if (shape.rank().is_dynamic() || shape.rank().get_length() < 1) {
        LOG_WARN("TurboQuant: cannot apply Hadamard, dynamic rank");
        return x;
    }
    const auto last_dim = shape[shape.rank().get_length() - 1];
    if (last_dim.is_dynamic()) {
        LOG_WARN("TurboQuant: cannot apply Hadamard, dynamic head_dim");
        return x;
    }
    const size_t head_dim = static_cast<size_t>(last_dim.get_length());
    if (!is_power_of_two(head_dim)) {
        LOG_WARN("TurboQuant: head_dim=" << head_dim << " is not a power of two; skipping rotation");
        return x;
    }

    auto H = make_hadamard_constant(head_dim, x.get_element_type());
    H->set_friendly_name(name_prefix + "/Hadamard_const");
    auto matmul = std::make_shared<ov::op::v0::MatMul>(x, H, /*transpose_a=*/false, /*transpose_b=*/false);
    matmul->set_friendly_name(name_prefix + "/Hadamard");
    return matmul->output(0);
}

// ---------------------------------------------------------------------------
// Quantize a (rotated) f32/f16 tensor via cascaded thresholds against the
// scaled codebook. Returns idx_u8 and scale outputs.
//
// scale = ReduceMax(|x|, last_axis, keepdims=true) / max(|level|)
// thr_i = scale * (level[i] + level[i+1]) / 2
// idx   = sum_i Convert<u8>(Greater(x, thr_i))     -> values in [0, 2^bits-1]
//
// Per-token only for now (group_size is reserved for future per-group support
// but currently always reduces along the last axis).
// ---------------------------------------------------------------------------
struct QuantizeOutputs {
    ov::Output<ov::Node> idx_u8;
    ov::Output<ov::Node> scale;
};

QuantizeOutputs apply_lloyd_max_quantize(const ov::Output<ov::Node>& x,
                                         uint8_t bits,
                                         uint32_t /*group_size*/,
                                         const std::string& name_prefix) {
    auto x_f32 = (x.get_element_type() == ov::element::f32)
                     ? x
                     : std::make_shared<ov::op::v0::Convert>(x, ov::element::f32)->output(0);

    auto levels = lloyd_max_levels_gaussian(bits);
    OPENVINO_ASSERT(levels.size() == (1u << bits), "Lloyd-Max codebook size mismatch");

    const float max_abs_level = std::abs(levels.back());

    const auto& shape = x_f32.get_partial_shape();
    const int64_t last_axis = shape.rank().get_length() - 1;

    auto abs_x = std::make_shared<ov::op::v0::Abs>(x_f32);
    abs_x->set_friendly_name(name_prefix + "/abs");

    auto reduce_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {last_axis});
    auto max_abs = std::make_shared<ov::op::v1::ReduceMax>(abs_x, reduce_axis, /*keep_dims=*/true);
    max_abs->set_friendly_name(name_prefix + "/reduce_max");

    auto inv_max_level = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f / max_abs_level});
    auto scale = std::make_shared<ov::op::v1::Multiply>(max_abs, inv_max_level);
    scale->set_friendly_name(name_prefix + "/scale");

    std::shared_ptr<ov::Node> idx_accumulator = nullptr;
    for (size_t i = 0; i + 1 < levels.size(); ++i) {
        const float thr = 0.5f * (levels[i] + levels[i + 1]);
        auto thr_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {thr});
        thr_const->set_friendly_name(name_prefix + "/thr_" + std::to_string(i));
        auto scaled_thr = std::make_shared<ov::op::v1::Multiply>(scale, thr_const);
        auto greater = std::make_shared<ov::op::v1::Greater>(x_f32, scaled_thr);
        greater->set_friendly_name(name_prefix + "/gt_" + std::to_string(i));
        auto greater_u8 = std::make_shared<ov::op::v0::Convert>(greater, ov::element::u8);
        if (i == 0) {
            idx_accumulator = greater_u8;
        } else {
            idx_accumulator = std::make_shared<ov::op::v1::Add>(idx_accumulator, greater_u8);
        }
    }
    OPENVINO_ASSERT(idx_accumulator != nullptr, "Codebook must have at least 2 levels");
    idx_accumulator->set_friendly_name(name_prefix + "/idx_u8");

    return {idx_accumulator->output(0), scale->output(0)};
}

// Dequant: levels = codebook[idx_u8]; out = levels * scale; cast to out_type.
ov::Output<ov::Node> apply_lloyd_max_dequantize(const ov::Output<ov::Node>& idx_u8,
                                                const ov::Output<ov::Node>& scale,
                                                uint8_t bits,
                                                ov::element::Type out_type,
                                                const std::string& name_prefix) {
    auto codebook = make_codebook_constant(bits, ov::element::f32);
    codebook->set_friendly_name(name_prefix + "/codebook");

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gathered = std::make_shared<ov::op::v8::Gather>(codebook, idx_u8, axis);
    gathered->set_friendly_name(name_prefix + "/gather");

    auto scale_f32 = (scale.get_element_type() == ov::element::f32)
                         ? scale
                         : std::make_shared<ov::op::v0::Convert>(scale, ov::element::f32)->output(0);

    auto multiplied = std::make_shared<ov::op::v1::Multiply>(gathered, scale_f32);
    multiplied->set_friendly_name(name_prefix + "/dequant");

    if (out_type == ov::element::f32) {
        return multiplied->output(0);
    }
    auto converted = std::make_shared<ov::op::v0::Convert>(multiplied, out_type);
    converted->set_friendly_name(name_prefix + "/dequant_cast");
    return converted->output(0);
}

// QJL: emit Sign(x_rot @ R) as i8 ±1.
ov::Output<ov::Node> apply_qjl(const ov::Output<ov::Node>& x_rot, size_t head_dim, const std::string& name_prefix) {
    auto R = make_qjl_projection_constant(head_dim, head_dim, x_rot.get_element_type());
    R->set_friendly_name(name_prefix + "/qjl_R");
    auto proj = std::make_shared<ov::op::v0::MatMul>(x_rot, R, false, false);
    proj->set_friendly_name(name_prefix + "/qjl_matmul");
    auto sign_f = std::make_shared<ov::op::v0::Sign>(proj);
    sign_f->set_friendly_name(name_prefix + "/qjl_sign");
    auto sign_i8 = std::make_shared<ov::op::v0::Convert>(sign_f, ov::element::i8);
    sign_i8->set_friendly_name(name_prefix + "/qjl_sign_i8");
    return sign_i8->output(0);
}

// Compute "scale shape" for the per-token case: kv_shape with the last axis
// (head_dim) collapsed to 1. This matches our quantize subgraph which reduces
// along the last axis of the rotated tensor.
ov::PartialShape compute_per_token_scale_shape(const ov::PartialShape& kv_shape) {
    ov::PartialShape s = kv_shape;
    if (s.rank().is_static() && s.rank().get_length() > 0) {
        s[s.rank().get_length() - 1] = 1;
    }
    return s;
}

}  // anonymous namespace

namespace ov::npuw {

void run_turboquant_kv_cache_passes(const std::shared_ptr<ov::Model>& model, const TurboQuantParams& params) {
    auto sdpa_blocks = ov::npuw::util::find_all_sdpa_pattern_nodes(model);
    LOG_INFO("TurboQuant: " << sdpa_blocks.size() << " SDPA pattern(s); "
                            << "key.bits=" << static_cast<int>(params.key.bits)
                            << " value.bits=" << static_cast<int>(params.value.bits)
                            << " key.use_qjl=" << params.key.use_qjl);

    if (sdpa_blocks.empty()) {
        LOG_WARN("TurboQuant: no SDPA patterns; running present-side rewrite only");
    }

    auto add_result = [&model](const ov::Output<ov::Node>& src, const std::string& name) {
        auto res = std::make_shared<ov::op::v0::Result>(src);
        res->set_friendly_name(name);
        res->output(0).get_tensor().set_names({name});
        model->add_results({res});
        return res;
    };

    auto add_param = [&model](ov::element::Type element_type,
                              const ov::PartialShape& shape,
                              const std::string& name) {
        auto p = std::make_shared<ov::op::v0::Parameter>(element_type, shape);
        p->set_friendly_name(name);
        p->output(0).get_tensor().set_names({name});
        model->add_parameters({p});
        return p;
    };

    // ───────────────────────────────────────────────────────────────────────
    // PAST SIDE: replace the past_key_values.X.{key,value} Parameter (now
    // u8 thanks to PPP) with: Lloyd-Max dequant + scale Param -> inverse
    // Hadamard. Wired into the K/V concat that was identified by the SDPA
    // pattern matcher.
    // ───────────────────────────────────────────────────────────────────────
    size_t pattern_index = 0;
    for (const auto& block : sdpa_blocks) {
        if (!block.past_key_concat_node || !block.past_value_concat_node) {
            LOG_WARN("TurboQuant: SDPA block #" << pattern_index << " missing K or V concat, skipping past side");
            ++pattern_index;
            continue;
        }

        const std::string idx_str = std::to_string(pattern_index);

        for (bool is_key : {true, false}) {
            const auto& cfg = is_key ? params.key : params.value;
            const auto concat = is_key ? block.past_key_concat_node : block.past_value_concat_node;
            const std::string kv = cache_name(is_key);
            const std::string past_prefix = "DynamicQuantize/" + idx_str + "/" + g_past_name + "/" + kv;
            const std::string past_block_name = "TurboQuant/" + idx_str + "/" + kv + "/past";

            // Locate the past-K/V Parameter input feeding this concat.
            std::shared_ptr<ov::op::v0::Parameter> past_param;
            size_t past_input_idx = 0;
            for (size_t i = 0; i < concat->get_input_size(); ++i) {
                auto src_node = concat->input_value(i).get_node_shared_ptr();
                if (auto p = ov::as_type_ptr<ov::op::v0::Parameter>(src_node)) {
                    const auto pname = p->get_friendly_name();
                    if ((is_key ? ov::npuw::util::isPastKeyValuesKey(pname)
                                : ov::npuw::util::isPastKeyValuesValue(pname))
                            .has_value()) {
                        past_param = p;
                        past_input_idx = i;
                        break;
                    }
                }
            }
            if (!past_param) {
                LOG_WARN("TurboQuant: could not find past " << kv << " Parameter on concat "
                                                            << concat->get_friendly_name() << ", skipping");
                continue;
            }

            // Add a fresh scale Parameter; its shape is the past-kv shape with
            // the head_dim axis collapsed to 1 (per-token scale).
            auto scale_param = add_param(ov::element::f32,
                                         compute_per_token_scale_shape(past_param->get_partial_shape()),
                                         past_prefix + "/scale");

            // Determine the dtype the concat expects: the OTHER input (the
            // "new K/V" coming from the LLM's projection) tells us what dtype
            // the rotated/dequantized output has to land in.
            ov::element::Type kv_dtype = ov::element::f16;
            for (size_t i = 0; i < concat->get_input_size(); ++i) {
                if (i == past_input_idx) continue;
                kv_dtype = concat->input_value(i).get_element_type();
                break;
            }

            // Build the dequant + inverse-rotate chain.
            auto dequantized = apply_lloyd_max_dequantize(past_param->output(0),
                                                          scale_param->output(0),
                                                          cfg.bits,
                                                          kv_dtype,
                                                          past_block_name);
            auto unrotated = apply_hadamard(dequantized, past_block_name + "/inv");

            // Rewire the concat input.
            concat->input(past_input_idx).replace_source_output(unrotated);
        }

        ++pattern_index;
    }

    model->validate_nodes_and_infer_types();

    // ───────────────────────────────────────────────────────────────────────
    // PRESENT SIDE: walk the model results, find present.X.{key,value},
    // bypass any PPP-inserted Convert, then insert Hadamard -> Lloyd-Max
    // quantize -> u8 idx and rewire the Result to the idx output. Also
    // expose scale (and optionally QJL) as new Results.
    // ───────────────────────────────────────────────────────────────────────
    auto original_results = model->get_results();
    for (const auto& result : original_results) {
        const auto result_name = result->get_friendly_name();
        auto is_key_opt = ov::npuw::util::isPresentKeyValuesKey(result_name);
        auto is_value_opt = ov::npuw::util::isPresentKeyValuesValue(result_name);

        if (!is_key_opt && !is_value_opt) {
            continue;
        }
        if (is_key_opt && is_value_opt) {
            LOG_WARN("TurboQuant: ambiguous result name " << result_name << ", skipping");
            continue;
        }

        const bool is_key = is_key_opt.has_value();
        const auto& cfg = is_key ? params.key : params.value;
        const std::string idx_str = std::to_string(is_key ? is_key_opt.value() : is_value_opt.value());
        const std::string kv = cache_name(is_key);
        const std::string present_prefix_dq = "DynamicQuantize/" + idx_str + "/" + g_present_name + "/" + kv;
        const std::string present_block_name = "TurboQuant/" + idx_str + "/" + kv + "/present";

        auto kv_in_value = result->input_value(0);
        auto kv_in_node = kv_in_value.get_node_shared_ptr();

        // Bypass PPP-inserted Convert (the existing kv_cache_compressed pass
        // does the same — PPP retypes the present output to u8, inserting a
        // Convert in front of the original Result).
        if (ov::is_type<ov::op::v0::Convert>(kv_in_node) && kv_in_node->get_users().size() == 1) {
            kv_in_value = kv_in_node->input_value(0);
        }

        const auto kv_shape = kv_in_value.get_partial_shape();
        ov::Dimension head_dim_dim = (kv_shape.rank().is_static() && kv_shape.rank().get_length() > 0)
                                         ? kv_shape[kv_shape.rank().get_length() - 1]
                                         : ov::Dimension::dynamic();

        // Hadamard rotation on the present-side concat output.
        auto rotated = apply_hadamard(kv_in_value, present_block_name);

        // Lloyd-Max quantize.
        auto qout = apply_lloyd_max_quantize(rotated, cfg.bits, cfg.group_size, present_block_name);

        // Rewire the existing present.X.key/value Result to the idx_u8.
        result->input(0).replace_source_output(qout.idx_u8);

        // Expose scale.
        add_result(qout.scale, present_prefix_dq + "/scale");

        // Optional QJL branch (key only by paper convention; honor cfg.use_qjl).
        if (cfg.use_qjl && head_dim_dim.is_static()) {
            const size_t hd = static_cast<size_t>(head_dim_dim.get_length());
            if (is_power_of_two(hd)) {
                auto qjl = apply_qjl(rotated, hd, present_block_name);
                add_result(qjl, present_prefix_dq + "/qjl");
            } else {
                LOG_WARN("TurboQuant: head_dim=" << hd << " not power of two; skipping QJL on " << kv);
            }
        }
    }

    model->validate_nodes_and_infer_types();
    LOG_INFO("TurboQuant: rewrite complete");
}

}  // namespace ov::npuw
