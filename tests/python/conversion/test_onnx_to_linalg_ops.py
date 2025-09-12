import pytest
import numpy as np
import os
import sys
import difflib
import textwrap
from mlir.dialects import func
from mlir.ir import (
    Context,
    InsertionPoint,
    Location,
    Module,
)
from mlir.passmanager import PassManager

import onnx2mlir.support as support
from onnx2mlir.dialect import onnx, register_onnx_dialect
from onnx2mlir.passes import register_onnx_to_linag_pass


def test_onnx_ConstantOp_lower():
    """
    Test ONNX ConstantOp lower.
    """
    EXPECTED_OUTPUT = textwrap.dedent(
        """
        module {
          func.func @main() -> tensor<2x2xi8> {
            %cst = arith.constant dense<[[-128, 8], [-9, 127]]> : tensor<2x2xi8>
            return %cst : tensor<2x2xi8>
          }
        }
        """
    )

    def create_mlir_module():
        with Context() as ctx, Location.unknown() as unk:
            register_onnx_dialect(ctx)
            module = Module.create(unk)
            np_array = np.array([[-128, 8], [-9, 127]], dtype=np.int8)
            tensor = support.mlir_dense_from_numpy(np_array)
            func_op = func.FuncOp("main", ([], [tensor.type]))
            with InsertionPoint(func_op.add_entry_block()):
                const = onnx.ConstantOp(tensor.type, value=tensor)
                func.ReturnOp([const.result])
            module.body.append(func_op)
            module.operation.verify()
        return module

    mlir_module = create_mlir_module()

    register_onnx_to_linag_pass()

    with Context() as ctx, Location.unknown() as unk:
        pm = PassManager()
        pm.add("lower-onnx-to-linalg")
        pm.run(mlir_module.operation)
        mlir_module.operation.verify()

    actual_output = str(mlir_module)

    expected_lines = EXPECTED_OUTPUT.strip().splitlines()
    actual_lines = actual_output.strip().splitlines()

    diff = difflib.unified_diff(
        expected_lines,
        actual_lines,
        fromfile="expected.mlir",
        tofile="actual.mlir",
        lineterm="",
    )
    diff_output = "\n".join(list(diff))

    assert not diff_output, f"MLIR output mismatch detected:\n{diff_output}"
