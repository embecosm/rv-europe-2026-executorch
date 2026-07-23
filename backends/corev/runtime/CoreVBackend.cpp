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
 : Data((uint8_t*)buffer->data()) {
  // First 4 bytes of the data are the Operator.
  operatorId = * ((int *) Data);
}

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

Error CoreVBackend::doAdd(Span<EValue*> args) const {
  const uint8_t *arg0 =
    static_cast<const uint8_t *>(args[0]->toTensor().const_data_ptr());
  const uint8_t *arg1 =
    static_cast<const uint8_t *>(args[1]->toTensor().const_data_ptr());
  uint8_t *arg2 =
    static_cast<uint8_t *>(args[2]->toTensor().mutable_data_ptr());
  const uint8_t *arg0_end = arg0 + args[0]->toTensor().numel ();

  // This is a demonstration.  It assume the pointers are all 32-bit aligned
  // and size is a multiple of 4.

  // The inline assembly block uses two temporary registers, wire these to
  // actual earlyclobber variables, so GCC does not use these for the input
  // pointers (arg0 and arg1).
  unsigned tmpr0, tmpr1;		// Type doesn't matter

  // Assembly code uses the PULP post-increment and SIMD instructions for
  // efficiency.
  while (arg0 < arg0_end)
    __asm__ ("cv.lw  %[tmpr0],(%[arg0]),4\n"		// Input 1 ptr
	     "cv.lw  %[tmpr1],(%[arg1]),4\n"		// Input 2 ptr
	     "cv.add.b %[tmpr0], %[tmpr0], %[tmpr1]\n"	// Do 4 adds
	     "cv.sw  %[tmpr0],(%[arg2]),4\n"		// Output ptr
	     : [arg0] "+r"(arg0), [arg1] "+r"(arg1), [arg2] "+r"(arg2),
	       [tmpr0] "=&r"(tmpr0), [tmpr1] "=&r"(tmpr1)
	     : : "memory");

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
