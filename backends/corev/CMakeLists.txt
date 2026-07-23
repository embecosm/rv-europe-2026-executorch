# CMake config for the CORE-V backend

# Copyright (C) 2026 Embecosm Limited
# Contributor Jeremy Bennett <jeremy.bennett@embecosm.com>
# Contributor Shane Slattery <shane.slattery@embecosm.com>

# This file is part of the Embecosm ExecuTorch tutorial for RISC-V.

# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)
project(corev_backend)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  cmake_path(SET EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  cmake_path(ABSOLUTE_PATH EXECUTORCH_ROOT NORMALIZE)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

set(_corev_backend_dir backends/corev/runtime)
list(TRANSFORM _corev_backend_dir PREPEND "${EXECUTORCH_ROOT}/")

set(_corev_backend_srcs ${_corev_backend_dir}/CoreVBackend.cpp)

add_library(corev_delegate STATIC ${_corev_backend_srcs})
target_include_directories(corev_delegate PUBLIC
    ${_corev_backend_dir}
    ${_common_include_directories}
)

if(EXECUTORCH_BUILD_COREV_RUNNER)
  include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

  # Baseline libraries that corev_runner will link against.
  set(_corev_runner_libs
    executorch
    program_schema
    common_schema
    extension_runner_util
    "-Wl,--whole-archive"
    corev_delegate
    "-Wl,--no-whole-archive"
  )
  target_include_directories(executorch INTERFACE ${_common_include_directories})

  set(_corev_runner_srcs
    ${EXECUTORCH_ROOT}/backends/corev/runtime/corev_pal.cpp
    ${EXECUTORCH_ROOT}/examples/corev/executor_runner/corev_executor_runner.cpp
    ${EXECUTORCH_ROOT}/runtime/executor/program.cpp
    ${EXECUTORCH_ROOT}/extension/evalue_util/print_evalue.cpp
  )

  # NOTE: CORE-V Runner cannot support optimized_native_cpu_ops_lib (aka
  #       EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
  #       while threadpool and std::mutex aren't supported. Just use the
  #       generic kernel implementations for now.
  list(APPEND _corev_runner_libs portable_ops_lib)

  # Register quantized ops if enabled
  if(EXECUTORCH_BUILD_KERNELS_QUANTIZED)
    list(APPEND _corev_runner_libs quantized_ops_lib)
  endif()

  add_executable(corev_runner ${_corev_runner_srcs})
  if(EXECUTORCH_ENABLE_LOGGING)
    target_compile_definitions(corev_runner PUBLIC ET_LOG_ENABLED=1)
  else()
    target_compile_definitions(corev_runner PUBLIC ET_LOG_ENABLED=0)
  endif()
  target_include_directories(corev_runner PUBLIC
    ${CMAKE_BINARY_DIR}
    ${_common_include_directories}
  )

  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      target_link_options(corev_runner PRIVATE "LINKER:--gc-sections")
  endif()
  target_link_libraries(corev_runner PRIVATE ${_corev_runner_libs})
  target_compile_options(corev_runner PUBLIC ${_common_compile_options})
endif()

