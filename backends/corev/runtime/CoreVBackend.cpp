/*
 * ExecuTorch backend for CORE-V: definition
 *
 * Copyright (C) 2026 Embecosm Limited
 * Contributor Jeremy Bennett <jeremy.bennett@embecosm.com>
 * Contributor Shane Slattery <shane.slattery@embecosm.com>
 *
 * This file is part of the Embecosm ExecuTorch tutorial for RISC-V.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <executorch/runtime/core/error.h>
#include "CoreVBackend.h"

using namespace executorch::runtime;

namespace executorch::backends::corev {

CoreVPayload::CoreVPayload(executorch::runtime::FreeableBuffer* buffer)
 : Data((uint8_t*)buffer->data()) {}

Result<DelegateHandle*> CoreVBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  MemoryAllocator* allocator = context.get_runtime_allocator();
  CoreVPayload* payload = allocator->allocateInstance<CoreVPayload>();
  if (payload == nullptr)
    return Error::MemoryAllocationFailed;

  new (payload) CoreVPayload(processed);
  return payload;
}

Error CoreVBackend::execute(
    ET_UNUSED BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  CoreVPayload* payload = static_cast<CoreVPayload*>(handle);
  if (payload->operatorId == 1)
    return doAdd(args);
  else
    return Error::Ok;
}

Error CoreVBackend::doAdd(
    Span<EValue*> args) const {
  printf ("Doing an 8-bit Tensor Add\n");
  return Error::Ok;
}

void CoreVBackend::destroy(DelegateHandle* handle) const {
  return;
}

bool CoreVBackend::is_available() const {
  return true;
}

} // namespace executorch::backends::corev

namespace {
auto cls = executorch::backends::corev::CoreVBackend();
executorch::runtime::Backend backend{"CoreVBackend", &cls};
volatile static auto success_with_compiler =
    executorch::runtime::register_backend(backend);
} // namespace
