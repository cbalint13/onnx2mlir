import pytest
import numpy as np
import os
import sys
import difflib
import textwrap
from mlir.dialects import func
from mlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    RankedTensorType,
    DenseElementsAttr,
    F32Type,
)
from onnx2mlir.dialect import onnx, register_onnx_dialect


def test_onnx_mlir_generation():
    """
    Test ONNX dialect ops MLIR generation.
    """

    EXPECTED_OUTPUT = textwrap.dedent(
    """
    module {
      func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %1 = "onnx.Constant"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
        %2 = "onnx.Add"(%1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %2 : tensor<2x2xf32>
      }
    }
    """
    )

    def create_mlir_module():
        """
        Creates and returns an MLIR module with ONNX operations.
        """
        with Context() as ctx, Location.unknown() as unk:
            register_onnx_dialect(ctx)
            module = Module.create(unk)
            f32 = F32Type.get(ctx)
            tensor_type = RankedTensorType.get(shape=[2, 2], element_type=f32, loc=unk)
            func_op = func.FuncOp(
                "main", ([tensor_type, tensor_type], [tensor_type]), loc=unk
            )
            with InsertionPoint(func_op.add_entry_block()):
                arg0, arg1 = func_op.arguments
                mul_op = onnx.MulOp(tensor_type, arg0, arg1)
                tensor_attr = DenseElementsAttr.get(
                    np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                    type=tensor_type,
                    context=ctx,
                )
                const = onnx.ConstantOp(tensor_type, value=tensor_attr)
                add_op = onnx.AddOp(tensor_type, const, mul_op)
                func.ReturnOp([add_op.result])
            module.body.append(func_op)
            module.operation.verify()
            return module

    mlir_module = create_mlir_module()
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
