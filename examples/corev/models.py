# Simple example models

# Copyright (C) 2026 Embecosm Limited
# Contributor Jeremy Bennett <jeremy.bennett@embecosm.com>
# Contributor Shane Slattery <shane.slattery@embecosm.com>

# This file is part of the Embecosm ExecuTorch tutorial for RISC-V.

# SPDX-License-Identifier: GPL-3.0-or-later

import torch

# Simple example models
class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y

    can_delegate = True
    example_input = (
        torch.rand(5, dtype=torch.float32),
        torch.rand(5, dtype=torch.float32),
    )
    calibration_data = (
        torch.randn(1, 5),
        torch.randn(1, 5),
    )

COREV_MODELS = {
    "add": AddModule,
}
