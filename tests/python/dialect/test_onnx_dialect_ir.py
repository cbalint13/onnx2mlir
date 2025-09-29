###############################################################################
#
#  ONNX2MLIR (ONNX dialect mappings for composable optimizations)
#
#  Authors:
#   Cristian Balint <cristian dot balint at gmail dot com>
#
#  Copyright (c) 2021,2025
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################
# pylint: disable=line-too-long,invalid-name

"""
\file tests/python/dialect/test_onnx_dialect_ir.py
\brief Tests for Onnx dialect IR
"""

import difflib
import textwrap
import numpy as np
from mlir.dialects import func
from mlir.ir import (
    Context,
    InsertionPoint,
    Location,
    Module,
)

from onnx2mlir import support
from onnx2mlir.dialect import onnx, register_onnx_dialect


def test_onnx_mlir_parse_ir():
    """
    Test ONNX dialect IR parser.
    """
    ONNX_MLIR_SAMPLE = textwrap.dedent(
        """
        module {
          func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0 = onnx.Mul(
                    A = %arg0 : tensor<2x2xf32>
                    B = %arg1 : tensor<2x2xf32>
                    attributes {}
                ) : tensor<2x2xf32>
            %1 = onnx.Constant(
                    attributes {value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>}
                ) : tensor<2x2xf32>
            %2 = onnx.Add(
                    A = %1 : tensor<2x2xf32>
                    B = %0 : tensor<2x2xf32>
                    attributes {}
                ) : tensor<2x2xf32>
            return %2 : tensor<2x2xf32>
          }
        }
        """
    )

    def onnx_module_parse():
        with Context() as ctx:
            register_onnx_dialect(ctx)
            module = Module.parse(ONNX_MLIR_SAMPLE)
        return module

    mlir_module = onnx_module_parse()
    actual_output = str(mlir_module)

    expected_lines = ONNX_MLIR_SAMPLE.strip().splitlines()
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


def test_onnx_mlir_generation():
    """
    Test ONNX dialect ops MLIR generation.
    """
    EXPECTED_OUTPUT = textwrap.dedent(
        """
        module {
          func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0 = onnx.Mul(
                    A = %arg0 : tensor<2x2xf32>
                    B = %arg1 : tensor<2x2xf32>
                    attributes {}
                ) : tensor<2x2xf32>
            %1 = onnx.Constant(
                    attributes {value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>}
                ) : tensor<2x2xf32>
            %2 = onnx.Add(
                    A = %1 : tensor<2x2xf32>
                    B = %0 : tensor<2x2xf32>
                    attributes {}
                ) : tensor<2x2xf32>
            return %2 : tensor<2x2xf32>
          }
        }
        """
    )

    def create_mlir_module():
        with Context() as ctx, Location.unknown() as unk:
            register_onnx_dialect(ctx)
            module = Module.create(unk)
            np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            tensor = support.mlir_dense_from_numpy(np_array)
            func_op = func.FuncOp(
                "main", ([tensor.type, tensor.type], [tensor.type]), loc=unk
            )
            with InsertionPoint(func_op.add_entry_block()):
                arg0, arg1 = func_op.arguments
                mul_op = onnx.MulOp(tensor.type, arg0, arg1)
                const = onnx.ConstantOp(tensor.type, value=tensor)
                add_op = onnx.AddOp(tensor.type, const, mul_op)
                func.ReturnOp([add_op])
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
