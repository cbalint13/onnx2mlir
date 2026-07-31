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
# pylint: disable=line-too-long,invalid-name,too-many-lines,too-many-locals

"""
\file tests/python/conversion/test_onnx_to_linalg_ops.py
\brief Tests for Onnx to Linalg operator lowering
"""

import random
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
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

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
def test_onnx_Constant_lower(ONNX_OPSET_VERSION):
    """
    Test ONNX Constant lowering.
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
    Test ONNX Cast lowering.
    """

    def create_onnx_model(np_array):
        input_tensor = make_tensor_value_info(
            "input", TensorProto.FLOAT, np_array.shape
        )
        output_tensor = make_tensor_value_info(
            "output", TensorProto.INT32, np_array.shape
        )
        cast_node = make_node(
            "Cast",
            ["input"],
            ["output"],
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

    np_array = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    onnx_model = create_onnx_model(np_array)

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(np_array).astype(np.int32)
        outputs = runner(llvm_module, "main", [np_array], [output])

        np.testing.assert_allclose(outputs[0], np_array.astype(np.int32), atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION",
    [
        (schema.name, schema.since_version)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Add", "Sub", "Mul", "Div", "Pow"]
    ],
)
def test_onnx_arith_binary_lower(ONNX_OP_NAME, ONNX_OPSET_VERSION):
    """
    Test ONNX arith binary operators lowering.
    """

    def create_onnx_model(inp_array0, inp_array1):
        input_tensor_0 = make_tensor_value_info(
            "input0", TensorProto.FLOAT, inp_array0.shape
        )
        input_tensor_1 = make_tensor_value_info(
            "input1", TensorProto.FLOAT, inp_array1.shape
        )
        output_tensor = make_tensor_value_info(
            "output", TensorProto.FLOAT, (inp_array0 + inp_array1).shape
        )
        arith_node = make_node(
            ONNX_OP_NAME,
            # binary arg
            ["input0", "input1"],
            ["output"],
        )
        graph = make_graph(
            nodes=[arith_node],
            name="arith_graph",
            inputs=[input_tensor_0, input_tensor_1],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    inp_array0 = np.random.rand(1, 3, 1).astype(np.float32)
    inp_array1 = np.random.rand(4, 1, 5).astype(np.float32)

    onnx_model = create_onnx_model(inp_array0, inp_array1)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input0": inp_array0, "input1": inp_array1})[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp_array0, inp_array1], [res_array])
        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION",
    [
        (schema.name, schema.since_version)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Sin", "Cos", "Elu"]
    ],
)
def test_onnx_unary_lower(ONNX_OP_NAME, ONNX_OPSET_VERSION):
    """
    Test ONNX arith unary operators lowering.
    """

    def create_onnx_model(np_array):
        input_tensor = make_tensor_value_info(
            "input", TensorProto.FLOAT, np_array.shape
        )
        output_tensor = make_tensor_value_info(
            "output", TensorProto.FLOAT, np_array.shape
        )
        cast_node = make_node(
            ONNX_OP_NAME,
            ["input"],
            ["output"],
        )
        graph = make_graph(
            nodes=[cast_node],
            name="arith_graph",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    np_array = np.random.rand(2, 2).astype(np.float32)
    onnx_model = create_onnx_model(np_array)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input": np_array})[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(np_array)
        outputs = runner(llvm_module, "main", [np_array], [output])

        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION",
    [
        (schema.name, schema.since_version)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Hardmax", "Softmax", "LogSoftmax"]
    ],
)
def test_onnx_softmax_lower(ONNX_OP_NAME, ONNX_OPSET_VERSION):
    """
    Test ONNX softmax family of operators lowering.
    """

    def create_onnx_model(np_array):
        input_tensor = make_tensor_value_info(
            "input", TensorProto.FLOAT, np_array.shape
        )
        output_tensor = make_tensor_value_info(
            "output", TensorProto.FLOAT, np_array.shape
        )
        cast_node = make_node(
            ONNX_OP_NAME,
            # i/o
            ["input"],
            ["output"],
            axis=1,
        )
        graph = make_graph(
            nodes=[cast_node],
            name="softmax_graph",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    np_array = np.random.rand(8, 8).astype(np.float32)
    onnx_model = create_onnx_model(np_array)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input": np_array})[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(np_array)
        outputs = runner(llvm_module, "main", [np_array], [output])

        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION",
    [
        schema.since_version
        for schema in get_all_schemas_with_history()
        if "Transpose" == schema.name
    ],
)
def test_onnx_transpose_lower(ONNX_OPSET_VERSION):
    """
    Test ONNX Transpose operator lowering.
    """

    def create_onnx_model(np_array):

        perm = random.sample(range(np_array.ndim), np_array.ndim)
        np_arrayT = np_array.transpose(perm)

        input_tensor = make_tensor_value_info(
            "input", TensorProto.FLOAT, np_array.shape
        )
        output_tensor = make_tensor_value_info(
            "output", TensorProto.FLOAT, np_arrayT.shape
        )
        cast_node = make_node(
            "Transpose",
            # i/o
            ["input"],
            ["output"],
            perm=perm,
        )
        graph = make_graph(
            nodes=[cast_node],
            name="transpose_graph",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    np_array = np.random.rand(1, 3, 8, 5).astype(np.float32)
    onnx_model = create_onnx_model(np_array)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input": np_array})[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(np_array)
        outputs = runner(llvm_module, "main", [np_array], [output])

        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION",
    [
        (schema.name, schema.since_version)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Greather", "GreatherOrEqual", "Less", "LessOrEqual"]
    ],
)
def test_onnx_compare_binary_lower(ONNX_OP_NAME, ONNX_OPSET_VERSION):
    """
    Test ONNX comparison binary operators lowering.
    """

    def create_onnx_model(inp_array0, inp_array1):
        input_tensor_0 = make_tensor_value_info(
            "input0", TensorProto.FLOAT, inp_array0.shape
        )
        input_tensor_1 = make_tensor_value_info(
            "input1", TensorProto.FLOAT, inp_array1.shape
        )
        output_tensor = make_tensor_value_info(
            "output", TensorProto.BOOL, (inp_array0 + inp_array1).shape
        )
        arith_node = make_node(
            ONNX_OP_NAME,
            # binary arg
            ["input0", "input1"],
            ["output"],
        )
        graph = make_graph(
            nodes=[arith_node],
            name="compare_graph",
            inputs=[input_tensor_0, input_tensor_1],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    inp_array0 = np.random.rand(1, 3, 1).astype(np.float32)
    inp_array1 = np.random.rand(4, 1, 5).astype(np.float32)

    onnx_model = create_onnx_model(inp_array0, inp_array1)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input0": inp_array0, "input1": inp_array1})[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp_array0, inp_array1], [res_array])
        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION",
    [
        schema.since_version
        for schema in get_all_schemas_with_history()
        # V1 legacy is not available in onnx evaluator
        if "Gemm" == schema.name and schema.since_version != 1
    ],
)
def test_onnx_gemm_lower(ONNX_OPSET_VERSION):
    """
    Test ONNX Gemm operator lowering.
    """

    def create_onnx_model(inp_arr0, inp_arr1, inp_bias):
        m, k = inp_arr0.shape
        _, n = inp_arr1.shape

        input_tensor_0 = make_tensor_value_info("input0", TensorProto.FLOAT, [m, k])
        input_tensor_1 = make_tensor_value_info("input1", TensorProto.FLOAT, [k, n])
        input_tensor_2 = make_tensor_value_info("bias0", TensorProto.FLOAT, [m, n])
        output_tensor = make_tensor_value_info("output", TensorProto.FLOAT, [m, n])

        bias_init = make_tensor(
            "bias0", TensorProto.FLOAT, [m, n], inp_bias.flatten().tolist()
        )

        arith_node = make_node(
            "Gemm",
            ["input0", "input1", "bias0"],
            ["output"],
        )
        graph = make_graph(
            nodes=[arith_node],
            name="gemm_graph",
            inputs=[input_tensor_0, input_tensor_1, input_tensor_2],
            outputs=[output_tensor],
            initializer=[bias_init],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    inp_arr0 = np.random.rand(16, 32).astype(np.float32)
    inp_arr1 = np.random.rand(32, 16).astype(np.float32)
    inp_bias = np.random.rand(16, 16).astype(np.float32)

    onnx_model = create_onnx_model(inp_arr0, inp_arr1, inp_bias)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(
        None, {"input0": inp_arr0, "input1": inp_arr1, "bias0": inp_bias}
    )[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp_arr0, inp_arr1, inp_bias], [res_arr])
        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype, shapes",
    [
        (opset, dtype, shapes)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Where" == schema.name
        ]
        for dtype in [TensorProto.FLOAT, TensorProto.INT32]
        for shapes in [
            ((4, 4), (4, 4), (4, 4)),  # Standard
            ((1,), (4, 4), (4, 4)),  # Broadcast Condition
            ((4, 1), (1, 4), (4, 4)),  # Multi-directional broadcast
        ]
    ],
)
# pylint: disable=too-many-locals
def test_onnx_where_lower(ONNX_OPSET_VERSION, dtype, shapes):
    """
    Test ONNX Where operator lowering.
    """
    cond_shape, x_shape, y_shape = shapes

    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.int32
    res_shape = np.broadcast(
        np.empty(cond_shape), np.empty(x_shape), np.empty(y_shape)
    ).shape

    def create_onnx_model():
        input_cond = make_tensor_value_info("condition", TensorProto.BOOL, cond_shape)
        input_x = make_tensor_value_info("X", dtype, x_shape)
        input_y = make_tensor_value_info("Y", dtype, y_shape)
        output_tensor = make_tensor_value_info("output", dtype, res_shape)

        where_node = make_node(
            "Where",
            ["condition", "X", "Y"],
            ["output"],
        )

        graph = make_graph(
            nodes=[where_node],
            name=f"where_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_cond, input_x, input_y],
            outputs=[output_tensor],
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    cond_arr = np.random.choice([True, False], size=cond_shape)
    x_arr = (np.random.rand(*x_shape) * 10).astype(np_dtype)
    y_arr = (np.random.rand(*y_shape) * 10).astype(np_dtype)

    onnx_model = create_onnx_model()

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"condition": cond_arr, "X": x_arr, "Y": y_arr})[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [cond_arr, x_arr, y_arr], [res_arr])

        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype, shape, axes",
    [
        (opset, dtype, shape, axes)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Unsqueeze" == schema.name
        ]
        for dtype in [TensorProto.FLOAT, TensorProto.INT32]
        for shape, axes in [
            ((3, 4), [0]),  # Leading dim: (1, 3, 4)
            ((3, 4), [1]),  # Middle dim:  (3, 1, 4)
            ((3, 4), [2]),  # Trailing dim: (3, 4, 1)
            ((3, 4), [0, 3]),  # Multiple: (1, 3, 4, 1)
            ((2, 3, 2), [1, 2]),  # Consecutive: (2, 1, 1, 3, 2)
        ]
    ],
)
# pylint: disable=too-many-locals
def test_onnx_unsqueeze_lower(ONNX_OPSET_VERSION, dtype, shape, axes):
    """
    Test ONNX Unsqueeze operator lowering.
    """
    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.int32

    res_shape = list(shape)
    for axis in sorted(axes):
        res_shape.insert(axis, 1)

    def create_onnx_model():
        input_tensor = make_tensor_value_info("data", dtype, shape)
        output_tensor = make_tensor_value_info("output", dtype, res_shape)

        if ONNX_OPSET_VERSION < 13:
            unsqueeze_node = make_node("Unsqueeze", ["data"], ["output"], axes=axes)
            inputs = [input_tensor]
            initializers = []
        else:
            axes_tensor = make_tensor("axes", TensorProto.INT64, [len(axes)], axes)
            unsqueeze_node = make_node(
                "Unsqueeze",
                ["data", "axes"],
                ["output"],
            )
            inputs = [input_tensor]
            initializers = [axes_tensor]

        graph = make_graph(
            nodes=[unsqueeze_node],
            name="unsqueeze_test",
            inputs=inputs,
            outputs=[output_tensor],
            initializer=initializers,
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    data_arr = (np.random.rand(*shape) * 10).astype(np_dtype)
    onnx_model = create_onnx_model()

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"data": data_arr})[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [data_arr], [res_arr])

        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-5)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype, shape, axes",
    [
        (opset, dtype, shape, axes)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Squeeze" == schema.name
        ]
        for dtype in [TensorProto.FLOAT, TensorProto.INT32]
        for shape, axes in [
            ((1, 3, 4), [0]),  # Squeeze leading: (3, 4)
            ((3, 1, 4), [1]),  # Squeeze middle:  (3, 4)
            ((3, 4, 1), [2]),  # Squeeze trailing: (3, 4)
            ((1, 3, 4, 1), [0, 3]),  # Squeeze multiple: (3, 4)
            ((1, 1, 3, 4), [0, 1]),  # Squeeze consecutive: (3, 4)
            ((1, 3, 1, 4), None),  # Squeeze all unit dims (Opset dependent)
        ]
    ],
)
# pylint: disable=too-many-locals
def test_onnx_squeeze_lower(ONNX_OPSET_VERSION, dtype, shape, axes):
    """
    Test ONNX Squeeze operator lowering.
    """
    if axes is None:
        res_shape = [d for d in shape if d != 1]
    else:
        res_shape = []
        for i, d in enumerate(shape):
            if i not in axes:
                res_shape.append(d)

    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.int32

    def create_onnx_model():
        input_tensor = make_tensor_value_info("data", dtype, shape)
        output_tensor = make_tensor_value_info("output", dtype, res_shape)

        inputs = ["data"]
        initializers = []

        # Handle Opset < 13 (axes as attribute) vs Opset >= 13 (axes as optional input)
        if ONNX_OPSET_VERSION < 13:
            kwargs = {}
            if axes is not None:
                kwargs["axes"] = axes
            squeeze_node = make_node("Squeeze", inputs, ["output"], **kwargs)
        else:
            if axes is not None:
                axes_tensor = make_tensor("axes", TensorProto.INT64, [len(axes)], axes)
                initializers.append(axes_tensor)
                inputs.append("axes")

            squeeze_node = make_node("Squeeze", inputs, ["output"])

        graph = make_graph(
            nodes=[squeeze_node],
            name="squeeze_test",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=initializers,
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    data_arr = (np.random.rand(*shape) * 10).astype(np_dtype)
    onnx_model = create_onnx_model()

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"data": data_arr})[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [data_arr], [res_arr])

        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-5)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype, input_shape, kernel, strides, pads",
    [
        (opset, dtype, shape, kernel, stride, pad)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "MaxPool" == schema.name
        ]
        for dtype in [TensorProto.FLOAT, TensorProto.INT8]
        for shape, kernel, stride, pad in [
            ((1, 3, 32, 32), [2, 2], [2, 2], [0, 0, 0, 0]),  # Standard NCHW
            ((1, 1, 10, 10), [3, 2], [1, 2], [0, 0, 0, 0]),  # Non-square
            ((1, 1, 5, 5), [3, 3], [1, 1], [1, 1, 1, 1]),  # With padding
        ]
    ],
)
# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def test_onnx_maxpool_lower(
    ONNX_OPSET_VERSION, dtype, input_shape, kernel, strides, pads
):
    """
    Test ONNX MaxPool operator lowering.
    """

    if ONNX_OPSET_VERSION < 12 and dtype == TensorProto.INT8:
        pytest.skip(f"MaxPool V{ONNX_OPSET_VERSION} only supports Float")

    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.int8

    h_in, w_in = input_shape[2], input_shape[3]
    h_out = (h_in + pads[0] + pads[2] - kernel[0]) // strides[0] + 1
    w_out = (w_in + pads[1] + pads[3] - kernel[1]) // strides[1] + 1
    output_shape = (input_shape[0], input_shape[1], h_out, w_out)

    def create_onnx_model():
        input_x = make_tensor_value_info("X", dtype, input_shape)
        output_y = make_tensor_value_info("Y", dtype, output_shape)

        maxpool_node = make_node(
            "MaxPool",
            ["X"],
            ["Y"],
            kernel_shape=kernel,
            strides=strides,
            pads=pads,
        )

        graph = make_graph(
            nodes=[maxpool_node],
            name=f"maxpool_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_x],
            outputs=[output_y],
        )

        model = make_model(graph, opset_imports=[make_opsetid("", ONNX_OPSET_VERSION)])
        check_model(model)
        return model

    x_arr = np.random.randint(-100, 100, size=input_shape).astype(np_dtype)

    onnx_model = create_onnx_model()

    if dtype == TensorProto.INT8:
        float_model = create_onnx_model()
        float_model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
        float_model.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
        ref = ReferenceEvaluator(float_model)
        onnx_result = ref.run(None, {"X": x_arr.astype(np.float32)})[0].astype(np.int8)
    else:
        ref = ReferenceEvaluator(onnx_model)
        onnx_result = ref.run(None, {"X": x_arr})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros(output_shape, dtype=np_dtype)
        outputs = runner(llvm_module, "main", [x_arr], [res_arr])

        np.testing.assert_allclose(outputs[0], onnx_result, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype, input_shape, weight_shape, strides, pads, has_bias",
    [
        (opset, dtype, in_shape, w_shape, stride, pad, bias)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Conv" == schema.name
        ]
        for dtype in [TensorProto.FLOAT, TensorProto.FLOAT16]
        for in_shape, w_shape, stride, pad, bias in [
            ((1, 3, 32, 32), (8, 3, 3, 3), [1, 1], [0, 0, 0, 0], False),
            ((1, 3, 32, 32), (16, 3, 3, 3), [2, 2], [1, 1, 1, 1], True),
            ((2, 1, 10, 10), (1, 1, 5, 5), [1, 1], [2, 2, 2, 2], False),
        ]
    ],
)
# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def test_onnx_conv_lower(
    ONNX_OPSET_VERSION, dtype, input_shape, weight_shape, strides, pads, has_bias
):
    """
    Test ONNX Conv operator lowering.
    """

    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.float16

    n, _, h_in, w_in = input_shape
    f, _, kh, kw = weight_shape
    h_out = (h_in + pads[0] + pads[2] - kh) // strides[0] + 1
    w_out = (w_in + pads[1] + pads[3] - kw) // strides[1] + 1
    output_shape = (n, f, h_out, w_out)

    x_arr = np.random.randn(*input_shape).astype(np_dtype)
    w_arr = np.random.randn(*weight_shape).astype(np_dtype)
    b_arr = np.random.randn(f).astype(np_dtype) if has_bias else None

    def create_onnx_model():
        input_x = make_tensor_value_info("X", dtype, input_shape)
        output_y = make_tensor_value_info("Y", dtype, output_shape)

        weight_init = make_tensor("W", dtype, weight_shape, w_arr.flatten().tolist())

        initializers = [weight_init]
        input_names = ["X", "W"]

        if has_bias:
            bias_init = make_tensor("B", dtype, (f,), b_arr.flatten().tolist())
            initializers.append(bias_init)
            input_names.append("B")

        conv_node = make_node(
            "Conv",
            input_names,
            ["Y"],
            kernel_shape=[kh, kw],
            strides=strides,
            pads=pads,
            group=1,
        )

        graph = make_graph(
            nodes=[conv_node],
            name=f"conv_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_x],
            outputs=[output_y],
            initializer=initializers,
        )

        model = make_model(graph, opset_imports=[make_opsetid("", ONNX_OPSET_VERSION)])
        check_model(model)
        return model

    onnx_model = create_onnx_model()

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"X": x_arr})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros(output_shape, dtype=np_dtype)

        outputs = runner(llvm_module, "main", [x_arr], [res_arr])

        atol = 1e-1 if dtype == TensorProto.FLOAT16 else 1e-5
        rtol = 1e-1 if dtype == TensorProto.FLOAT16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype, input_shape, axis",
    [
        (ONNX_OPSET_VERSION, dtype, shape, ax)
        for ONNX_OPSET_VERSION in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Flatten" == schema.name
        ]
        for dtype in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE,
            TensorProto.INT32,
        ]
        for shape, ax in [
            ((2, 3, 4, 5), 1),  # Default case
            ((2, 3, 4, 5), 0),  # Axis at start
            ((2, 3, 4, 5), 2),  # Axis in middle
            ((2, 3, 4, 5), 4),  # Axis at end
            ((2, 3, 4, 5), -1),  # Negative axis
            ((5,), 1),  # Rank 1 input
            ((1, 10), 1),  # Rank 2 input
        ]
    ],
)
# pylint: disable=too-many-locals
def test_onnx_flatten_lower(ONNX_OPSET_VERSION, dtype, input_shape, axis):
    """
    Test ONNX Flatten operator lowering.
    """

    if ONNX_OPSET_VERSION == 1 and dtype == TensorProto.INT32:
        pytest.skip(f"Flatten V{ONNX_OPSET_VERSION} only supports Float")

    np_dtype = None
    if dtype == TensorProto.FLOAT:
        np_dtype = np.float32
    elif dtype == TensorProto.FLOAT16:
        np_dtype = np.float16
    elif dtype == TensorProto.DOUBLE:
        np_dtype = np.float64
    elif dtype == TensorProto.INT32:
        np_dtype = np.int32
    else:
        pytest.skip(f"DataType {np_dtype} not implemented in test")

    x_arr = np.random.randn(*input_shape).astype(np_dtype)

    rank = len(input_shape)
    norm_axis = axis if axis >= 0 else axis + rank

    dim0 = int(np.prod(input_shape[:norm_axis]))
    dim1 = int(np.prod(input_shape[norm_axis:]))
    output_shape = (dim0, dim1)

    def create_onnx_model():
        input_x = make_tensor_value_info("X", dtype, input_shape)
        output_y = make_tensor_value_info("Y", dtype, output_shape)

        flatten_node = make_node(
            "Flatten",
            ["X"],
            ["Y"],
            axis=axis,
        )

        graph = make_graph(
            nodes=[flatten_node],
            name=f"flatten_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_x],
            outputs=[output_y],
        )

        model = make_model(graph, opset_imports=[make_opsetid("", ONNX_OPSET_VERSION)])
        check_model(model)
        return model

    onnx_model = create_onnx_model()

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"X": x_arr})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros(output_shape, dtype=np_dtype)
        outputs = runner(llvm_module, "main", [x_arr], [res_arr])

        np.testing.assert_allclose(onnx_result, outputs[0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape0, shape1",
    [
        (schema.name, schema.since_version, dtype_proto, shape0, shape1)
        for schema in get_all_schemas_with_history()
        if schema.name in ["BitwiseAnd", "BitwiseOr", "BitwiseXor"]
        for dtype_proto in [
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.UINT32,
            TensorProto.UINT64,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.INT64,
        ]
        for shape0, shape1 in [
            ((2, 3, 4), (2, 3, 4)),  # Non-broadcast
            ((1, 3, 1), (4, 1, 5)),  # Broadcast (3D expansion)
            ((2, 3, 4), (1, 3, 4)),  # Broadcast (outer dimension)
        ]
    ],
)
def test_onnx_bitwise_binary_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape0, shape1
):
    """
    Test ONNX Bitwise binary operators lowering.
    """
    np_dtype = None
    if dtype_proto == TensorProto.UINT8:
        np_dtype = np.uint8
    elif dtype_proto == TensorProto.UINT16:
        np_dtype = np.uint16
    elif dtype_proto == TensorProto.UINT32:
        np_dtype = np.uint32
    elif dtype_proto == TensorProto.UINT64:
        np_dtype = np.uint64
    elif dtype_proto == TensorProto.INT8:
        np_dtype = np.int8
    elif dtype_proto == TensorProto.INT16:
        np_dtype = np.int16
    elif dtype_proto == TensorProto.INT32:
        np_dtype = np.int32
    elif dtype_proto == TensorProto.INT64:
        np_dtype = np.int64
    else:
        pytest.skip(f"DataType {dtype_proto} not implemented in test")

    low, high = (0, 200) if np.issubdtype(np_dtype, np.unsignedinteger) else (-100, 100)
    inp0 = np.random.randint(low, high, size=shape0).astype(np_dtype)
    inp1 = np.random.randint(low, high, size=shape1).astype(np_dtype)

    input_tensor_0 = make_tensor_value_info("input0", dtype_proto, inp0.shape)
    input_tensor_1 = make_tensor_value_info("input1", dtype_proto, inp1.shape)

    out_shape = np.broadcast_shapes(inp0.shape, inp1.shape)
    output_tensor = make_tensor_value_info("output", dtype_proto, out_shape)

    logic_node = make_node(
        ONNX_OP_NAME,
        ["input0", "input1"],
        ["output"],
    )
    graph = make_graph(
        nodes=[logic_node],
        name="logic_graph",
        inputs=[input_tensor_0, input_tensor_1],
        outputs=[output_tensor],
        initializer=[],
    )
    opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
    onnx_model = make_model(graph, opset_imports=opset_imports)
    check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input0": inp0, "input1": inp1})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp0, inp1], [res_array])
        np.testing.assert_array_equal(outputs[0], onnx_result)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape",
    [
        (schema.name, schema.since_version, dtype_proto, shape)
        for schema in get_all_schemas_with_history()
        if schema.name in ["BitwiseNot"]
        for dtype_proto in [
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.UINT32,
            TensorProto.UINT64,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.INT64,
        ]
        for shape in [
            (5,),
            (2, 3),
            (2, 3, 4),
        ]
    ],
)
def test_onnx_bitwise_unary_lower(ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape):
    """
    Test ONNX Bitwise unary operators lowering.
    """
    np_dtype = None
    if dtype_proto == TensorProto.UINT8:
        np_dtype = np.uint8
    elif dtype_proto == TensorProto.UINT16:
        np_dtype = np.uint16
    elif dtype_proto == TensorProto.UINT32:
        np_dtype = np.uint32
    elif dtype_proto == TensorProto.UINT64:
        np_dtype = np.uint64
    elif dtype_proto == TensorProto.INT8:
        np_dtype = np.int8
    elif dtype_proto == TensorProto.INT16:
        np_dtype = np.int16
    elif dtype_proto == TensorProto.INT32:
        np_dtype = np.int32
    elif dtype_proto == TensorProto.INT64:
        np_dtype = np.int64
    else:
        pytest.skip(f"DataType {dtype_proto} not implemented in test")

    low, high = (0, 200) if np.issubdtype(np_dtype, np.unsignedinteger) else (-100, 100)
    inp0 = np.random.randint(low, high, size=shape).astype(np_dtype)

    input_tensor_0 = make_tensor_value_info("input0", dtype_proto, inp0.shape)
    output_tensor = make_tensor_value_info("output", dtype_proto, inp0.shape)

    logic_node = make_node(
        ONNX_OP_NAME,
        ["input0"],
        ["output"],
    )
    graph = make_graph(
        nodes=[logic_node],
        name="logic_unary_graph",
        inputs=[input_tensor_0],
        outputs=[output_tensor],
        initializer=[],
    )
    opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
    onnx_model = make_model(graph, opset_imports=opset_imports)
    check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input0": inp0})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp0], [res_array])
        np.testing.assert_array_equal(outputs[0], onnx_result)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape0, shape1",
    [
        (schema.name, schema.since_version, dtype_proto, shape0, shape1)
        for schema in get_all_schemas_with_history()
        if schema.name in ["And", "Or", "Xor"]
        for dtype_proto in [TensorProto.BOOL]
        for shape0, shape1 in [
            ((2, 3, 4), (2, 3, 4)),  # Non-broadcast case
            ((1, 3, 1), (4, 1, 5)),  # Broadcast 3D expansion case
            ((2, 3, 4), (1, 3, 4)),  # Broadcast outer dimension case
        ]
    ],
)
def test_onnx_boolean_binary_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape0, shape1
):
    """
    Test ONNX Boolean binary operators lowering.
    """
    np_dtype = None
    if dtype_proto == TensorProto.BOOL:
        np_dtype = np.bool_
    else:
        pytest.skip(f"DataType {dtype_proto} not implemented in test")

    inp0 = np.random.choice([True, False], size=shape0).astype(np_dtype)
    inp1 = np.random.choice([True, False], size=shape1).astype(np_dtype)

    input_tensor_0 = make_tensor_value_info("input0", dtype_proto, inp0.shape)
    input_tensor_1 = make_tensor_value_info("input1", dtype_proto, inp1.shape)

    out_shape = np.broadcast_shapes(inp0.shape, inp1.shape)
    output_tensor = make_tensor_value_info("output", dtype_proto, out_shape)

    logic_node = make_node(
        ONNX_OP_NAME,
        ["input0", "input1"],
        ["output"],
    )
    graph = make_graph(
        nodes=[logic_node],
        name="logic_binary_graph",
        inputs=[input_tensor_0, input_tensor_1],
        outputs=[output_tensor],
        initializer=[],
    )
    opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
    onnx_model = make_model(graph, opset_imports=opset_imports)
    check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input0": inp0, "input1": inp1})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp0, inp1], [res_array])

        np.testing.assert_array_equal(outputs[0], onnx_result)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape",
    [
        (schema.name, schema.since_version, dtype_proto, shape)
        for schema in get_all_schemas_with_history()
        if schema.name in ["GlobalAveragePool"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.FLOAT16,
        ]
        for shape in [
            (2, 3, 10),  # 3D (1D spatial) #
            (2, 3, 4, 5),  # 4D (2D spatial NCHW) #
            (2, 3, 2, 4, 4),  # 5D (3D spatial NCDHW)
        ]
    ],
)
def test_onnx_global_average_pooling_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape
):
    """
    Test ONNX GlobalAveragePooling lowering.
    """
    np_dtype = None
    if dtype_proto == TensorProto.FLOAT:
        np_dtype = np.float32
    elif dtype_proto == TensorProto.DOUBLE:
        np_dtype = np.float64
    elif dtype_proto == TensorProto.FLOAT16:
        np_dtype = np.float16
    else:
        pytest.skip(f"DataType {dtype_proto} not implemented in test")

    # Generate random test inputs (negative & positive values)
    inp0 = np.random.uniform(-5.0, 5.0, size=shape).astype(np_dtype)

    # Global pooling output shape: (N, C, 1, 1, ...)
    out_shape = list(shape[:2]) + [1] * (len(shape) - 2)

    input_tensor = make_tensor_value_info("input0", dtype_proto, inp0.shape)
    output_tensor = make_tensor_value_info("output", dtype_proto, out_shape)

    pool_node = make_node(ONNX_OP_NAME, ["input0"], ["output"])
    graph = make_graph(
        nodes=[pool_node],
        name="global_average_pooling_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[],
    )
    opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
    onnx_model = make_model(graph, opset_imports=opset_imports)
    check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input0": inp0})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp0], [res_array])

        # Assert numeric precision within standard tolerances
        atol = 1e-2 if np_dtype == np.float16 else 1e-5
        rtol = 1e-2 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape, p_val",
    [
        (schema.name, schema.since_version, dtype_proto, shape, p_val)
        for schema in get_all_schemas_with_history()
        if schema.name in ["GlobalLpPool"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.FLOAT16,
        ]
        for shape in [
            (2, 3, 10),  # 3D (1D spatial)
            (2, 3, 4, 5),  # 4D (2D spatial NCHW)
            (2, 3, 2, 4, 4),  # 5D (3D spatial NCDHW)
        ]
        for p_val in ([1, 2, 3])
    ],
)
def test_onnx_global_lp_pooling_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape, p_val
):
    """
    Test ONNX GlobalLpPooling lowering.
    """

    # ReferenceEvaluator
    class GlobalLpPool(OpRun):
        """
        Global Lp Pooling reduces spatial dimensions (H, W, ...) using p-norm
        Ins: (N, C, D1, D2, ..., Dn)
        Out: (N, C, 1, 1, ..., 1)
        """

        def _run(self, x, p=2):  # pylint: disable=arguments-differ
            axes = tuple(range(2, len(x.shape)))
            if p == 1:
                res = np.sum(np.abs(x), axis=axes, keepdims=True)
            elif p == 2:
                res = np.sqrt(np.sum(np.square(x), axis=axes, keepdims=True))
            else:
                res = np.power(
                    np.sum(np.power(np.abs(x), p), axis=axes, keepdims=True), 1.0 / p
                )
            return (res,)

    np_dtype = None
    if dtype_proto == TensorProto.FLOAT:
        np_dtype = np.float32
    elif dtype_proto == TensorProto.DOUBLE:
        np_dtype = np.float64
    elif dtype_proto == TensorProto.FLOAT16:
        np_dtype = np.float16
    else:
        pytest.skip(f"DataType {dtype_proto} not implemented in test")

    # Generate random test inputs (negative & positive values)
    inp0 = np.random.uniform(-5.0, 5.0, size=shape).astype(np_dtype)

    # Global pooling output shape: (N, C, 1, 1, ...)
    out_shape = list(shape[:2]) + [1] * (len(shape) - 2)

    input_tensor = make_tensor_value_info("input0", dtype_proto, inp0.shape)
    output_tensor = make_tensor_value_info("output", dtype_proto, out_shape)

    kwargs = {}
    if p_val is not None:
        if ONNX_OPSET_VERSION == 1:
            if dtype_proto == TensorProto.FLOAT16:
                pytest.skip(f"GlobalLpPool V{ONNX_OPSET_VERSION} float16 overflow")
            else:
                p_val = np_dtype(p_val)

        kwargs["p"] = p_val

    pool_node = make_node(ONNX_OP_NAME, ["input0"], ["output"], **kwargs)
    graph = make_graph(
        nodes=[pool_node],
        name="global_lp_pooling_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[],
    )
    opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
    onnx_model = make_model(graph, opset_imports=opset_imports)
    check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model, new_ops=[GlobalLpPool])
    onnx_result = ref.run(None, {"input0": inp0})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp0], [res_array])

        # Assert numeric precision within standard tolerances
        atol = 1e-2 if np_dtype == np.float16 else 1e-5
        rtol = 1e-2 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape",
    [
        (schema.name, schema.since_version, dtype_proto, shape)
        for schema in get_all_schemas_with_history()
        if schema.name in ["GlobalMaxPool"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.FLOAT16,
        ]
        for shape in [
            (2, 3, 10),  # 3D (1D spatial) #
            (2, 3, 4, 5),  # 4D (2D spatial NCHW) #
            (2, 3, 2, 4, 4),  # 5D (3D spatial NCDHW)
        ]
    ],
)
def test_onnx_global_max_pooling_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape
):
    """
    Test ONNX GlobalMaxPooling lowering.
    """

    # ReferenceEvaluator
    class GlobalMaxPool(OpRun):
        """
        Global Max Pooling reduces spatial dimensions (D1, D2, ..., Dn).
        Ins: (N, C, D1, D2, ..., Dn)
        Out: (N, C, 1, 1, ..., 1)
        """

        def _run(self, x):  # pylint: disable=arguments-differ
            spatial_axes = tuple(range(2, len(x.shape)))
            res = np.max(x, axis=spatial_axes, keepdims=True)
            return (res,)

    np_dtype = None
    if dtype_proto == TensorProto.FLOAT:
        np_dtype = np.float32
    elif dtype_proto == TensorProto.DOUBLE:
        np_dtype = np.float64
    elif dtype_proto == TensorProto.FLOAT16:
        np_dtype = np.float16
    else:
        pytest.skip(f"DataType {dtype_proto} not implemented in test")

    # Generate random test inputs (negative & positive values)
    inp0 = np.random.uniform(-5.0, 5.0, size=shape).astype(np_dtype)

    # Global pooling output shape: (N, C, 1, 1, ...)
    out_shape = list(shape[:2]) + [1] * (len(shape) - 2)

    input_tensor = make_tensor_value_info("input0", dtype_proto, inp0.shape)
    output_tensor = make_tensor_value_info("output", dtype_proto, out_shape)

    pool_node = make_node(ONNX_OP_NAME, ["input0"], ["output"])
    graph = make_graph(
        nodes=[pool_node],
        name="global_max_pooling_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[],
    )
    opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
    onnx_model = make_model(graph, opset_imports=opset_imports)
    check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model, new_ops=[GlobalMaxPool])
    onnx_result = ref.run(None, {"input0": inp0})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp0], [res_array])

        # Assert numeric precision within standard tolerances
        atol = 1e-2 if np_dtype == np.float16 else 1e-5
        rtol = 1e-2 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype, shape, axis, split_sizes",
    [
        (opset, dtype, shape, axis, split_sizes)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Split" == schema.name
        ]
        for dtype in [TensorProto.FLOAT, TensorProto.INT32]
        # Opset 1 only supports Floating-point types according to spec
        if not (opset == 1 and dtype != TensorProto.FLOAT)
        for shape, axis, split_sizes in [
            ((6, 4), 0, [2, 4]),  # Unequal split along leading axis
            ((6, 4), 0, None),  # Equal split along leading axis (3, 3)
            ((4, 6), 1, [1, 2, 3]),  # Multi-part split along inner axis
            ((4, 6), 1, None),  # Equal split along inner axis (3, 3)
            ((4, 8), -1, [3, 5]),  # Negative axis indexing
        ]
    ],
)
# pylint: disable=too-many-statements
def test_onnx_split_lower(ONNX_OPSET_VERSION, dtype, shape, axis, split_sizes):
    """
    Test ONNX Split lowering.
    """

    # ReferenceEvaluator
    class Split(OpRun):
        """
        Global Lp Pooling reduces spatial dimensions (H, W, ...) using p-norm
        Ins: (N, C, D1, D2, ..., Dn)
        Out: (N, C, 1, 1, ..., 1)
        """

        # pylint: disable=arguments-differ,unused-argument
        def _run(self, x, split=None, axis=0, num_outputs=None):
            onnx_results = []
            curr_idx = 0
            for s_size in effective_splits:
                slc = [slice(None)] * len(x.shape)
                slc[norm_axis] = slice(curr_idx, curr_idx + s_size)
                onnx_results.append(x[tuple(slc)])
                curr_idx += s_size
            return tuple(onnx_results)

    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.int32

    # Normalize axis and calculate target split dimension sizes
    norm_axis = axis if axis >= 0 else axis + len(shape)
    axis_dim_size = shape[norm_axis]

    if split_sizes is None:
        num_outputs = 2
        effective_splits = [axis_dim_size // num_outputs] * num_outputs
    else:
        num_outputs = len(split_sizes)
        effective_splits = split_sizes

    # Calculate output shapes
    output_shapes = []
    for split_size in effective_splits:
        res_shape = list(shape)
        res_shape[norm_axis] = split_size
        output_shapes.append(res_shape)

    def create_onnx_model():
        input_tensor = make_tensor_value_info("input", dtype, shape)
        output_names = [f"output_{i}" for i in range(num_outputs)]
        output_tensors = [
            make_tensor_value_info(name, dtype, out_shape)
            for name, out_shape in zip(output_names, output_shapes)
        ]

        inputs = [input_tensor]
        initializers = []
        kwargs = {"axis": axis}

        if ONNX_OPSET_VERSION < 13:
            # Opsets 1 - 11: split attribute
            if split_sizes is not None:
                kwargs["split"] = split_sizes
        else:
            # Opsets 13+: split is passed as 1D INT64 input tensor
            if split_sizes is not None:
                split_tensor = make_tensor(
                    "split",
                    TensorProto.INT64,
                    [len(split_sizes)],
                    split_sizes,
                )
                inputs.append(split_tensor)
                initializers.append(split_tensor)
            elif ONNX_OPSET_VERSION >= 18:
                kwargs["num_outputs"] = num_outputs

        split_node = make_node(
            "Split",
            [inp.name if hasattr(inp, "name") else inp for inp in inputs],
            output_names,
            **kwargs,
        )

        graph = make_graph(
            nodes=[split_node],
            name="split_test",
            inputs=[input_tensor],
            outputs=output_tensors,
            initializer=initializers,
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    data_arr = (np.random.rand(*shape) * 10).astype(np_dtype)
    onnx_model = create_onnx_model()

    new_ops = [Split] if ONNX_OPSET_VERSION == 1 else []
    ref = ReferenceEvaluator(onnx_model, new_ops=new_ops)
    onnx_results = ref.run(None, {"input": data_arr})

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arrs = [np.zeros(out_shape, dtype=np_dtype) for out_shape in output_shapes]
        outputs = runner(llvm_module, "main", [data_arr], res_arrs)

        for res, onnx_res in zip(outputs, onnx_results):
            np.testing.assert_allclose(res, onnx_res, atol=1e-5)
