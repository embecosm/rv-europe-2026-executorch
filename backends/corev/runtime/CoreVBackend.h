/*
 * ExecuTorch backend for CORE-V: declaration
 *
 * Copyright (C) 2026 Embecosm Limited
 * Contributor Jeremy Bennett <jeremy.bennett@embecosm.com>
 * Contributor Shane Slattery <shane.slattery@embecosm.com>
 *
 * This file is part of the Embecosm ExecuTorch tutorial for RISC-V.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

using namespace executorch::runtime;

namespace executorch::backends::corev {

struct CoreVPayload {
  CoreVPayload(executorch::runtime::FreeableBuffer* buffer);
  uint32_t operatorId;
  uint8_t *Data;
};

class CoreVBackend final : public BackendInterface {
public:
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec>
          compile_specs) const override;

  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override;

  Error doAdd(
       Span<EValue*> args) const;

  void destroy(DelegateHandle* handle) const override;

  bool is_available() const override;
};

} // namespace executorch::backends::corev
