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

Result<DelegateHandle*> CoreVBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  return nullptr;
}

Error CoreVBackend::execute(
    ET_UNUSED BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
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
static auto success_with_compiler =
    executorch::runtime::register_backend(backend);
} // namespace
