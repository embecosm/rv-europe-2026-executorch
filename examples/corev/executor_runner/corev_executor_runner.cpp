/*
 * ExecuTorch runtime for CORE-V main program
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (C) 2026 Embecosm Limited
 * Contributor Jeremy Bennett <jeremy.bennett@embecosm.com>
 * Contributor Shane Slattery <shane.slattery@embecosm.com>
 *
 * This file is part of the Embecosm ExecuTorch tutorial for RISC-V.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/**
 * @file
 *
 * This tool can run ExecuTorch model files that only use operators that
 * are covered by the portable kernels, with possible delegate to the
 * test_backend_compiler_lib.
 *
 * It sets all input tensor data to ones, and assumes that the outputs are
 * all fp32 tensors.
 */

#include <cstdint>
#include <memory>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#ifndef RUNNER_MODEL_PATH
  #error "RUNNER_MODEL_PATH definition has not been specified!"
#endif

alignas(alignof(std::max_align_t)) const uint8_t model_pte[] = {
  #include RUNNER_MODEL_PATH
};

// These are pretty small pools, but we are a 4MB system!
static uint8_t method_allocator_pool[1 * 1024U * 1024U]; // 1 MB
static uint8_t temp_allocator_pool[1024U * 256U]; // 256KB

constexpr uint32_t FLAGS_num_executions = 1;

using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::aten::Tensor;

#include <stdio.h>

int main() {
  executorch::runtime::runtime_init();

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  constexpr size_t program_data_len = sizeof(model_pte);
  ET_LOG(Info, "Size of model is %zu.", sizeof(model_pte));
  BufferDataLoader loader = BufferDataLoader(&model_pte, program_data_len);

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(&loader);
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_pte);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_pte);

  // Use the first method in the program.
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  // MethodMeta describes the memory requirements of the method.
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      (uint32_t)method_meta.error());

  // The runtime does not use malloc/new; it allocates all memory using the
  // MemoryManger provided by the client. Clients are responsible for allocating
  // the memory ahead of time, or providing MemoryAllocator subclasses that can
  // do it dynamically.
  //

  // The method allocator is used to allocate all dynamic C++ metadata/objects
  // used to represent the loaded method. This allocator is only used during
  // loading a method of the program, which will return an error if there was
  // not enough memory.
  //
  // The amount of memory required depends on the loaded method and the runtime
  // code itself. The amount of memory here is usually determined by running the
  // method and seeing how much memory is actually used, though it's possible to
  // subclass MemoryAllocator so that it calls malloc() under the hood (see
  // MallocMemoryAllocator).
  //
  // In this example we use a statically allocated memory pool.
  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

  // Temporary memory required by kernels
  MemoryAllocator temp_allocator{
      MemoryAllocator(sizeof(temp_allocator_pool), temp_allocator_pool)};

  // The memory-planned buffers will back the mutable tensors used by the
  // method. The sizes of these buffers were determined ahead of time during the
  // memory-planning passes.
  //
  // Each buffer typically corresponds to a different hardware memory bank. Most
  // mobile environments will only have a single buffer. Some embedded
  // environments may have more than one for, e.g., slow/large DRAM and
  // fast/small SRAM, or for memory associated with particular cores.
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
  std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    // .get() will always succeed because id < num_memory_planned_buffers.
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  // Assemble all of the allocators into the MemoryManager that the Executor
  // will use.
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  // Load the method from the program, using the provided allocators. Running
  // the method can mutate the memory-planned buffers, so the method should only
  // be used by a single thread at at time, but it can be reused.
  //
  Result<Method> method = program->load_method(
      method_name, &memory_manager, nullptr);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)method.error());
  ET_LOG(Info, "Method loaded.");

  // Set up the inputs and run the model.
  for (uint32_t i = 0; i < FLAGS_num_executions; i++) {
    ET_LOG(Debug, "Preparing inputs.");
    // Allocate input tensors and set all of their elements to desired
    // values. The `inputs` variable owns the allocated memory and must live
    // past the last call to `execute()`.
    //
    // NOTE: we have to re-prepare input tensors on every execution
    // because inputs whose space gets reused by memory planning (if
    // any such inputs exist) will not be preserved for the next
    // execution.
    auto isz = method_meta->input_tensor_meta(0)->sizes()[0];
    char *arr1 = new char[isz];
    char *arr2 = new char[isz];
    for (size_t i = 0; i < isz; i++) {
      arr1[i] = i % 128;
      arr2[i] = i % 128;
    }
    auto inputs = executorch::extension::prepare_input_tensors(
        *method, {}, {{arr1, isz}, {arr2, isz}});
    ET_CHECK_MSG(
        inputs.ok(),
        "Could not prepare inputs: 0x%" PRIx32,
        (uint32_t)inputs.error());
    ET_LOG(Debug, "Inputs prepared.");

    Error status = method->execute();
    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        (uint32_t)status);
  }
  ET_LOG(
      Info,
      "Model executed successfully %" PRIu32 " time(s).",
      FLAGS_num_executions
  );
  // Horrible kludge, because ET logging is too big!
  printf("Model executed successfully %" PRIu32 " time(s).\n",
	 FLAGS_num_executions);
  // Print the outputs.  Change the #if 0 to enable this.
#if 0
  std::vector<EValue> outputs(method->outputs_size());
  ET_LOG(Info, "%zu outputs: ", outputs.size());
  Error status = method->get_outputs(outputs.data(), outputs.size());
  Tensor& ot = outputs.data()[0].toTensor();
  const uint8_t * res = static_cast<const uint8_t *>(ot.const_data_ptr());
  for(int i = 0; i < ot.numel(); i++)
    printf("res[%d] = %d\n", i, res[i]);
  ET_CHECK(status == Error::Ok);
#endif

  return 0;
}
