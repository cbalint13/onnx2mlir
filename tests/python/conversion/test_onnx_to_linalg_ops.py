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
\file tests/python/conversion/test_onnx_to_linalg_ops.py
\brief Tests for Onnx to Linalg operator lowering
"""

import pytest
import numpy as np

from onnx import TensorProto
from onnx.defs import get_all_schemas_with_history
from onnx.helper import (
    make_model,
    make_node,
    make_tensor_value_info,
    make_graph,
    make_tensor,
    make_opsetid,
)
from onnx.checker import check_model

from mlir.ir import (
    Context,
    Location,
)

from onnx2mlir.importer import import_from_onnx
from onnx2mlir.pipeline import llvm_lower_pipeline, runner


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION",
    [
        schema.since_version
        for schema in get_all_schemas_with_history()
        if "Constant" == schema.name
    ],
)
def test_onnx_ConstantOp_lower(ONNX_OPSET_VERSION):
    """
    Test ONNX ConstantOp lower.
    """

    def create_onnx_model(np_array):
        constant_value = np_array
        output_tensor_info = make_tensor_value_info(
            "output_tensor", TensorProto.FLOAT, [2, 2]
        )
        constant_node = make_node(
            "Constant",
            inputs=[],
            outputs=["output_tensor"],
            value=make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=[2, 2],
                vals=constant_value.flatten().tolist(),
            ),
        )
        graph = make_graph(
            [constant_node],
            "constant_graph",
            # no inputs
            [],
            [output_tensor_info],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    onnx_model = create_onnx_model(np_array)

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(np_array)
        outputs = runner(llvm_module, "main", [], [output])

        np.testing.assert_allclose(outputs[0], np_array, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION",
    [
        schema.since_version
        for schema in get_all_schemas_with_history()
        if "Cast" == schema.name
    ],
)
def test_onnx_Cast_lower(ONNX_OPSET_VERSION):
    """
    Test ONNX CastOp lower.
    """

    def create_onnx_model(np_array):
        input_tensor = make_tensor_value_info(
            "input_tensor", TensorProto.FLOAT, np_array.shape
        )
        output_tensor = make_tensor_value_info(
            "output_tensor", TensorProto.INT32, np_array.shape
        )
        cast_node = make_node(
            "Cast",
            ["input_tensor"],
            ["output_tensor"],
            to=TensorProto.INT32 if ONNX_OPSET_VERSION > 1 else "INT32",
        )
        graph = make_graph(
            nodes=[cast_node],
            name="cast_graph",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    onnx_model = create_onnx_model(np_array)

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(np_array).astype(np.int32)
        outputs = runner(llvm_module, "main", [output], [np_array])

        np.testing.assert_allclose(outputs[0], np_array.astype(np.int32), atol=1e-3)
