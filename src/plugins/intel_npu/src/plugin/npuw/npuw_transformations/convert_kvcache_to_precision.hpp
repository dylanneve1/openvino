// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

ov::element::Type optimize_kv_cache_storage(const std::shared_ptr<ov::Model>& model);

// Optional TurboQuant override for the KV-cache compression path. When
// `enabled` is true, the integer-storage branch routes through
// run_turboquant_kv_cache_passes() instead of run_kv_cache_dynamic_quantization_passes().
struct TurboQuantSettings {
    bool enabled = false;
    uint32_t key_bits = 3;
    uint32_t value_bits = 2;
};

class ConvertKVCacheToPrecision : public ov::pass::ModelPass {
    ov::element::Type m_lp_type;
    TurboQuantSettings m_turboquant;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ConvertKVCacheToPrecision");
    explicit ConvertKVCacheToPrecision(const ov::element::Type lptype);
    ConvertKVCacheToPrecision(const ov::element::Type lptype, const TurboQuantSettings& turboquant);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
