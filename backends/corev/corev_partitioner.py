# CORE-V partitioner

# Copyright (C) 2026 Embecosm Limited
# Contributor Jeremy Bennett <jeremy.bennett@embecosm.com>
# Contributor Shane Slattery <shane.slattery@embecosm.com>

# This file is part of the Embecosm ExecuTorch tutorial for RISC-V.

# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from typing import final, List

import torch
from executorch.backends.corev.corev_backend import CoreVBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

from torch.fx.passes.operator_support import OperatorSupportBase

# The operators we currently support
SupportedOperatorsList = [
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.convolution.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
]

class CoreVSupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in SupportedOperatorsList

@final
class CoreVPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(CoreVBackend.__name__,
                                              compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logging.info("CoreVPartitioner::partition")
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            CoreVSupportedOperators(),
            allows_single_node_partition=True,
        )

        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

        tag_constant_data(exported_program)
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags
        )
