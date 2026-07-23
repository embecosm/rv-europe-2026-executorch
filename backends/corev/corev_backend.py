# CORE-V backend main program

# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (C) 2026 Embecosm Limited
# Contributor Jeremy Bennett <jeremy.bennett@embecosm.com>
# Contributor Shane Slattery <shane.slattery@embecosm.com>

# This file is part of the Embecosm ExecuTorch tutorial for RISC-V.

# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import struct
from typing import final, List, Optional
from executorch.exir.dialects._ops import ops as exir_ops
from torch import uint8 as torch_uint8

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

def generate_corev_compile_spec(
    system_config: Optional[str] = None,
    operators_to_skip: Optional[List[str]] = None,
    extra_flags: Optional[str] = None,
) -> List[CompileSpec]:
    compile_spec: List[CompileSpec] = []
    compiler_flags = []
    output_format = None

    if system_config is not None:
        compile_spec.append(CompileSpec("system_config",
                                        system_config.encode()))

    # TODO: Likely broken, a List needs to be encoded properly.
    if operators_to_skip is not None:
        compile_spec.append(CompileSpec("operators_to_skip",
                                        operators_to_skip.encode()))

    if extra_flags is not None:
        compile_spec.append(CompileSpec("extra_flags", extra_flags.encode()))

    return compile_spec


@final
class CoreVBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        logging.info("CoreVBackend::preprocess")
        binary = bytes()

        for node in edge_program.graph.nodes:
            if node.op == "call_function":
                logging.debug(f"Operator to be processed: {node.target}")
                if node.target == exir_ops.edge.aten.add.Tensor:
                    # We have an add, but can only delegate if all the
                    # arguments are torch.uint8
                    can_delegate = True
                    for a in node.args:
                        if a.meta['val'].dtype != torch_uint8:
                            can_delegate = False
                    if can_delegate:
                        logging.info("CoreVBackend: 8-bit add delegated")
                    # Tag with a little endian integer (since RISC-V is
                    # little-endian)
                    binary += struct.pack("<I", 1)

        return PreprocessResult(processed_bytes=binary)
