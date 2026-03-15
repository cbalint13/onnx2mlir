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
