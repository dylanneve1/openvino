# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Demo utility to showcase the NPUW test model generator
# Generates LLM-like models and prints their structure

# Only build if tests are enabled (we need commonTestUtils)
if(NOT ENABLE_TESTS)
    return()
endif()

set(TARGET_NAME "npuw_model_generator_demo")

add_executable(${TARGET_NAME}
    demo_model_generator.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/tests/unit/npuw/model_builder/model_builder.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/tests/unit/npuw/model_builder
)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::commonTestUtils
    ${CMAKE_DL_LIBS}
)

set_target_properties(${TARGET_NAME} PROPERTIES
    FOLDER "tools"
)

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION tools
    COMPONENT tools
    EXCLUDE_FROM_ALL
)
