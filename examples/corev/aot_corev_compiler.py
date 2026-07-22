# Example script for exporting example models to flatbuffer (.pte)

# Copyright (C) 2026 Embecosm Limited
# Contributor Jeremy Bennett <jeremy.bennett@embecosm.com>
# Contributor Shane Slattery <shane.slattery@embecosm.com>

# This file is part of the Embecosm ExecuTorch tutorial for RISC-V.

# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import logging
import os
import torch

from tabulate import tabulate
from typing import Any, List, Tuple

from examples.devtools.scripts.export_bundled_program import save_bundled_program

from executorch.backends.corev.corev_partitioner import CoreVPartitioner
from executorch.backends.corev.corev_backend import generate_corev_compile_spec
from executorch.backends.corev.corev_quantizer import CoreVQuantizer

from executorch.devtools.backend_debug import get_delegation_info
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite

from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.utils import format_delegated_graph
from executorch.extension.export_util.utils import save_pte_program

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.utils.data import DataLoader

from .models import COREV_MODELS
from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

def get_model_and_inputs_from_name(
    model_name: str, model_input: str | None
) -> Tuple[torch.nn.Module, Any]:
    """Given the name of an example pytorch model, return it and example inputs.

    Raises RuntimeError if there is no example model corresponding to the given name.
    """
    example_inputs = None
    if model_input is not None:
        logging.info(f"Load model input from {model_input}")
        if model_input.endswith(".pt"):
            example_inputs = torch.load(model_input, weights_only=False)
        else:
            raise RuntimeError(
                f"Model input data '{model_input}' is not a valid name. Use --model_input <FILE>.pt e.g. saved with torch.save()"
            )

    # Case 1: Model is defined in this file
    if model_name in COREV_MODELS.keys():
        logging.info(f"Internal model {model_name}")
        model = COREV_MODELS[model_name]()
        if example_inputs is None:
            example_inputs = COREV_MODELS[model_name].example_input

    # Case 2: Model is defined in examples/models/
    elif model_name in MODEL_NAME_TO_MODEL.keys():
        logging.warning(
            "Using a model from examples/models not all of these are currently supported"
        )
        logging.info(
            f"Load {model_name} -> {MODEL_NAME_TO_MODEL[model_name]} from examples/models"
        )

        model, tmp_example_inputs, _, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL[model_name]
        )
        if example_inputs is None:
            example_inputs = tmp_example_inputs

    else:
        raise RuntimeError(
            f"Model '{model_name}' is not a valid name. Use --help for a list of available models."
        )
    logging.debug(f"Loaded model: {model}")
    logging.debug(f"Loaded input: {example_inputs}")
    return model, example_inputs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        default="add",
        help=f"Builtin model name or a model from examples/models. Valid names: {set(list(COREV_MODELS.keys())+list(MODEL_NAME_TO_MODEL.keys()))}",
    )
    parser.add_argument(
        "--model_input",
        required=False,
        default=None,
        help="Provide model input .pt file, or python variable name",
    )
    parser.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing a CoreV delegated model",
    )
    parser.add_argument(
        "--bundleio",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing BundleIO bpte file with input/output test/ref data.",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        required=False,
        default=False,
        help="Produce a quantized model",
    )
    parser.add_argument(
        "-s",
        "--so_library",
        required=False,
        default=None,
        help="Provide path to so library. E.g., cmake-out/examples/portable/custom_ops/libcustom_ops_aot_lib.so",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set the logging level to debug."
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=False,
        help="Filename (if .pte or .bpte is used) or a folder for outputs, if not specified the default is to place files in cwd.",
    )
    parser.add_argument(
        "--system_config",
        required=False,
        default="chimera",
        help="Configuartion for CoreV Delegation",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, force=True)

    if args.quantize and not args.so_library:
        logging.warning(
            "Quantization enabled without supplying path to libcustom_ops_aot_lib using -s flag."
            + "This is required for running quantized models with unquantized input."
        )

    # if we have custom ops, register them before processing the model
    if args.so_library is not None:
        logging.info(f"Loading custom ops from {args.so_library}")
        torch.ops.load_library(args.so_library)

    if (
        args.model_name in COREV_MODELS.keys()
        and args.delegate is True
        and COREV_MODELS[args.model_name].can_delegate is False
    ):
        raise RuntimeError(f"Model {args.model_name} cannot be delegated.")

    return args

def quantize(
    model: torch.nn.Module,
    model_name: str,
    example_inputs: Tuple[torch.Tensor],
) -> torch.nn.Module:
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    logging.info("Quantizing Model...")
    logging.debug(f"Original model: {model}")

    m = prepare_pt2e(model, CoreVQuantizer())

    dataset = get_calibration_data(model_name, example_inputs)

    # The dataset could be a tuple of tensors or a DataLoader
    # These two cases need to be accounted for
    if isinstance(dataset, DataLoader):
        for sample, _ in dataset:
            m(sample)
    else:
        m(*dataset)

    m = convert_pt2e(m)
    logging.debug(f"Quantized model: {m}")
    return m

def get_calibration_data(
    model_name: str,
    example_inputs: Tuple[torch.Tensor],
):
    # If the model is in the calibration_data dictionary, get the data from there
    # This is used for the simple model examples provided
    if model_name in COREV_MODELS:
        if hasattr(COREV_MODELS[model_name], "calibration_data"):
            return COREV_MODELS[model_name].calibration_data

    # As a last resort, fallback to the scripts previous behavior and return the example inputs
    return example_inputs

def dump_delegation_info(edge):
    graph_module = edge.exported_program().graph_module
    delegation_info = get_delegation_info(graph_module)
    df = delegation_info.get_operator_delegation_dataframe()
    table = tabulate(df, headers="keys", tablefmt="fancy_grid")
    delegation_info_string = f"Delegation info:\n{delegation_info.get_summary()}\nDelegation table:\n{table}\n"
    logging.info(delegation_info_string)

    delegated_graph_str = format_delegated_graph(graph_module)
    logging.info(delegated_graph_str)

def save_bpte_program(exec_prog, original_model: torch.nn.Module, output_name: str):
    # Construct MethodTestSuite for Each Method

    # Generate Test Suites
    method_names = [
        method.name for method in exec_prog.executorch_program.execution_plan
    ]

    program_inputs = {m_name: [example_inputs] for m_name in method_names}

    method_test_suites: List[MethodTestSuite] = []
    for m_name in method_names:
        method_inputs = program_inputs[m_name]

        # To create a bundled program, we first create every test cases from input. We leverage eager model
        # to generate expected output for each test input, and use MethodTestCase to hold the information of
        # each test case. We gather all MethodTestCase for same method into one MethodTestSuite, and generate
        # bundled program by all MethodTestSuites.
        method_test_cases: List[MethodTestCase] = []

        method_index = 0
        for method_input in method_inputs:
            output_ref = original_model(*method_input)

            logging.debug(f"input_{method_index}: {method_input}")
            logging.debug(f"output_ref_{method_index}: {output_ref}")

            method_test_cases.append(
                MethodTestCase(
                    inputs=method_input,
                    expected_outputs=output_ref,
                )
            )

            method_index = method_index + 1

        method_test_suites.append(
            MethodTestSuite(
                method_name=m_name,
                test_cases=method_test_cases,
            )
        )

    # Generate BundledProgram
    save_bundled_program(exec_prog, method_test_suites, output_name)

if __name__ == "__main__":
    args = get_args()
    output_name = f"{os.path.basename(os.path.splitext(args.model_name)[0])}"

    # Pick model from one of the supported lists
    original_model, example_inputs = get_model_and_inputs_from_name(
        args.model_name, args.model_input
    )
    model = original_model.eval()

    # export_for_training under the assumption we quantize, the exported form also works
    # in to_edge if we don't quantize
    exported_program = torch.export.export(model, example_inputs, strict=True)
    model_fp32 = exported_program.module()
    model_uint8 = None # If we are quantizing

    compile_spec:List[CompileSpec] = generate_corev_compile_spec(
        system_config=args.system_config
    )

    if args.quantize:
        output_name += "_quantized"
        model_uint8 = quantize(
            model_fp32,
            args.model_name,
            example_inputs,
        )
        # Wrap quantized model back into an exported_program
        exported_program = torch.export.export_for_training(model_uint8, example_inputs)

    if args.delegate:
        output_name += "_delegated"
        edge_program_manager = to_edge_transform_and_lower(
            exported_program,
            partitioner=[CoreVPartitioner(compile_spec)],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )
    else:
        output_name += "_non_delegated"
        edge_program_manager = to_edge_transform_and_lower(
            exported_program,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )

    dump_delegation_info(edge_program_manager)

    executorch_program = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    if args.bundleio:
        output_name = f"{output_name}.bpte"
    else:
        output_name = f"{output_name}.pte"

    if args.output is not None:
        if args.output.endswith(".pte") or args.output.endswith(".bpte"):
            # --output is a pte or bundle pte filename use it as output name
            if args.bundleio and not args.output.endswith(".bpte"):
                raise RuntimeError(
                    f"--output expects a .bpte file when using --bundleio. .pte provided: {args.output}"
                )
            if not args.bundleio and not args.output.endswith(".pte"):
                raise RuntimeError(
                    f"--output expects a .pte file when NOT using --bundleio. .bpte provided: {args.output}"
                )
            output_name = args.output
        else:
            # --output is a folder
            output_name = os.path.join(args.output, output_name)

    if args.bundleio:
        save_bpte_program(executorch_program, original_model, output_name)
        print(f"Bundle PTE file saved as {output_name}")
    else:
        save_pte_program(executorch_program, output_name)
        print(f"PTE file saved as {output_name}")
