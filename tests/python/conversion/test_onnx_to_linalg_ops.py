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
from typing import Tuple

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
    tensor_dtype_to_np_dtype,
)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

from mlir.ir import Context, Location, MLIRError

from onnx2mlir.importer import import_from_onnx
from onnx2mlir.pipeline import llvm_lower_pipeline, runner


def _generate_dtype_random(
    dtype: np.dtype,
    shape: Tuple[int, ...],
    max_val=1e3,
    min_val=None,
) -> np.ndarray:
    """
    Generates a numpy array of a given shape and data type within [min_val, max_val].

    Parameters:
        dtype (np.dtype): Target dtype (e.g., np.float64, np.complex64).
        shape (Tuple[int, ...]): Desired shape of the output array.
        max_val (float/int): Upper bound for values.
        min_val (float/int): Lower bound for values, default is -max_val

    Returns:
        np.ndarray: Array of specified shape and dtype filled with random numbers.
    """
    rng = np.random.default_rng(42)

    if min_val is None:
        min_val = -max_val

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = max(int(min_val), info.min)
        high = min(int(max_val), info.max)
        if low >= high:
            low = info.min
            high = min(int(max_val), info.max) if max_val > info.min else info.max
        return rng.integers(low, high, size=shape, dtype=dtype, endpoint=True)

    if np.issubdtype(dtype, np.floating):
        return rng.uniform(min_val, max_val, size=shape).astype(dtype)

    if np.issubdtype(dtype, np.complexfloating):
        if dtype == np.complex64:
            float_dtype = np.float32
        else:
            float_dtype = np.float64
        real_part = rng.uniform(min_val, max_val, size=shape).astype(float_dtype)
        imag_part = rng.uniform(min_val, max_val, size=shape).astype(float_dtype)
        return (real_part + 1j * imag_part).astype(dtype)

    if dtype == np.dtype(bool):
        return rng.choice([True, False], size=shape)

    raise ValueError(f"Unsupported or unrecognized numpy dtype: {dtype}")


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, shape",
    [
        (schema.since_version, dtype_proto, shape)
        for schema in get_all_schemas_with_history()
        if "Constant" == schema.name
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.UINT32,
            TensorProto.UINT64,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.BOOL,
        ]
        for shape in [
            (2, 3, 4),
        ]
    ],
)
def test_onnx_constant_lower(ONNX_OPSET_VERSION, dtype_proto, shape):
    """
    Test ONNX Constant lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model(np_array, dtype_proto):
        constant_value = np_array
        output_tensor_info = make_tensor_value_info(
            "output_tensor", dtype_proto, np_array.shape
        )
        constant_node = make_node(
            "Constant",
            inputs=[],
            outputs=["output_tensor"],
            value=make_tensor(
                name="constant_opset_{ONNX_OPSET_VERSION}",
                data_type=dtype_proto,
                dims=np_array.shape,
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

    np_array = _generate_dtype_random(np_dtype, shape=shape, max_val=127)
    onnx_model = create_onnx_model(np_array, dtype_proto)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "must be", "but got"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Constant V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(np_array, dtype=np_dtype)
        outputs = runner(llvm_module, "main", [], [output])

        np.testing.assert_allclose(outputs[0], np_array, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, output_shape, fill_value",
    [
        (schema.since_version, dtype_proto, output_shape, fill_value)
        for schema in get_all_schemas_with_history()
        if schema.name == "ConstantOfShape"
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.UINT32,
            TensorProto.UINT64,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.BOOL,
        ]
        for output_shape in [
            (2, 3, 4),  # 3D Tensor
            (1, 5),  # 2D Matrix
            (4,),  # 1D Vector
        ]
        for fill_value in [
            None,  # No value attribute
            3.5,  # Positive numeric fill
            -2,  # Negative numeric fill
        ]
    ],
)
# pylint: disable=too-many-branches,too-many-statements
def test_onnx_constantofshape_lower(
    ONNX_OPSET_VERSION, dtype_proto, output_shape, fill_value
):
    """
    Test ONNX ConstantOfShape operator lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    if fill_value is not None:
        if fill_value < 0 and np.issubdtype(np_dtype, np.unsignedinteger):
            pytest.skip("Negative fill value is invalid for unsigned integer")
        typed_fill = np.array(fill_value, dtype=np_dtype).item()
    else:
        typed_fill = None

    def create_onnx_model(shape_array, dtype_proto, typed_fill):
        input_tensor = make_tensor_value_info(
            "input", TensorProto.INT64, shape_array.shape
        )
        output_tensor = make_tensor_value_info(
            "output",
            dtype_proto if typed_fill is not None else TensorProto.FLOAT,
            [int(x) for x in shape_array],
        )

        node_kwargs = {}
        if typed_fill is not None:
            value_tensor = make_tensor(
                name="value",
                data_type=dtype_proto,
                dims=[1],
                vals=[typed_fill],
            )
            node_kwargs["value"] = value_tensor

        cos_node = make_node(
            "ConstantOfShape",
            inputs=["input"],
            outputs=["output"],
            **node_kwargs,
        )
        graph = make_graph(
            nodes=[cos_node],
            name=f"constantofshape_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    shape_input = np.array(output_shape, dtype=np.int64)
    onnx_model = create_onnx_model(shape_input, dtype_proto, typed_fill)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input": shape_input})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"ConstantOfShape V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(
            onnx_result, dtype=np_dtype if fill_value is not None else np.float32
        )
        outputs = runner(llvm_module, "main", [shape_input], [res_array])

        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, src_dtype_proto, tgt_dtype_proto, shape",
    [
        (
            schema.since_version,
            src_dtype_proto,
            tgt_dtype_proto,
            shape,
        )
        for schema in get_all_schemas_with_history()
        if schema.name == "Cast"
        for src_dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE,
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.UINT32,
            TensorProto.UINT64,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.BOOL,
        ]
        for tgt_dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE,
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.UINT32,
            TensorProto.UINT64,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.BOOL,
        ]
        for shape in [
            (1, 33, 22),
        ]
    ],
)
def test_onnx_cast_lower(ONNX_OPSET_VERSION, src_dtype_proto, tgt_dtype_proto, shape):
    """
    Test ONNX Cast lowering.
    """
    src_np_dtype = tensor_dtype_to_np_dtype(src_dtype_proto)
    tgt_np_dtype = tensor_dtype_to_np_dtype(tgt_dtype_proto)

    np_array = _generate_dtype_random(src_np_dtype, shape, max_val=127)

    def create_onnx_model(np_array, src_dtype_proto, tgt_dtype_proto):

        input_tensor = make_tensor_value_info("input", src_dtype_proto, np_array.shape)
        output_tensor = make_tensor_value_info(
            "output", tgt_dtype_proto, np_array.shape
        )

        node_kwargs = {}
        if ONNX_OPSET_VERSION == 1:
            node_kwargs["to"] = str(TensorProto.DataType.Name(tgt_dtype_proto))
        else:
            node_kwargs["to"] = int(tgt_dtype_proto)
        if ONNX_OPSET_VERSION >= 19:
            node_kwargs["saturate"] = 1
        if ONNX_OPSET_VERSION >= 24:
            node_kwargs["round_mode"] = "up"

        cast_node = make_node(
            "Cast", inputs=["input"], outputs=["output"], **node_kwargs
        )

        graph = make_graph(
            nodes=[cast_node],
            name="cast_graph",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(np_array, src_dtype_proto, tgt_dtype_proto)
    check_model(onnx_model)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Cast V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(tgt_dtype_proto)}"
                    f" -> {TensorProto.DataType.Name(tgt_dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        if ONNX_OPSET_VERSION == 1:
            onnx_result = np_array.astype(tgt_np_dtype)
        else:
            ref_inputs = {"input": np_array}
            ref = ReferenceEvaluator(onnx_model)
            onnx_result = ref.run(None, ref_inputs)[0]

        output_buffer = np.zeros_like(onnx_result, dtype=tgt_np_dtype)
        outputs = runner(llvm_module, "main", [np_array], [output_buffer])

        if np.issubdtype(tgt_np_dtype, np.integer):
            np.testing.assert_array_equal(outputs[0], onnx_result)
        else:
            np.testing.assert_allclose(outputs[0], onnx_result, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shapes, fmod",
    [
        (schema.name, schema.since_version, dtype_proto, shapes, fmod)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Add", "Div", "Mod", "Mul", "Pow", "Sub"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.UINT32,
            TensorProto.UINT64,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.INT64,
        ]
        for shapes in [
            [(1, 3, 3), (1, 3, 3)],  # Non-broadcasting
            [(1, 3, 1), (4, 1, 5)],  # Broadcasting
        ]
        for fmod in ([0, 1] if schema.name == "Mod" else [None])
    ],
)
# pylint: disable=too-many-branches,too-many-statements
def test_onnx_arith_binary_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shapes, fmod
):
    """
    Test ONNX arith binary operators lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model(inp_array0, inp_array1, dtype_proto):
        input_tensor_0 = make_tensor_value_info("input0", dtype_proto, inp_array0.shape)
        input_tensor_1 = make_tensor_value_info("input1", dtype_proto, inp_array1.shape)
        output_tensor = make_tensor_value_info(
            "output", dtype_proto, (inp_array0 + inp_array1).shape
        )
        kwargs = {"fmod": fmod} if ONNX_OP_NAME == "Mod" else {}
        arith_node = make_node(
            ONNX_OP_NAME,
            ["input0", "input1"],
            ["output"],
            **kwargs,
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

    inp_array0 = _generate_dtype_random(np_dtype, shape=shapes[0], max_val=10)
    inp_array1 = _generate_dtype_random(np_dtype, shape=shapes[1], max_val=10)

    if ONNX_OP_NAME in ["Div", "Mod"]:
        inp_array1 += inp_array1 >= 0
    if ONNX_OP_NAME == "Pow":
        inp_array1 = np.abs(inp_array1)

    onnx_model = create_onnx_model(inp_array0, inp_array1, dtype_proto)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"{ONNX_OP_NAME} V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        ref = ReferenceEvaluator(onnx_model)
        onnx_result = ref.run(None, {"input0": inp_array0, "input1": inp_array1})[0]

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp_array0, inp_array1], [res_array])

        atol = 1e-2 if np_dtype == np.float16 else 1e-5
        rtol = 1e-2 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, ONNX_OPSET_DOMAIN, dtype_proto",
    [
        (schema.name, schema.since_version, schema.domain, dtype_proto)
        for schema in get_all_schemas_with_history()
        if schema.name
        in [
            "Abs",
            "Acos",
            "Acosh",
            "Asin",
            "Asinh",
            "Atan",
            "Atanh",
            "Bernoulli",
            "Binarizer",
            "Ceil",
            "Celu",
            "Cos",
            "Cosh",
            "Elu",
            "Erf",
            "Exp",
            "Floor",
            "Gelu",
            "HardSigmoid",
            "HardSwish",
            "Identity",
            "IsInf",
            "IsNaN",
            "LeakyRelu",
            "Log",
            "Mish",
            "Neg",
            "Not",
            "Reciprocal",
            "Relu",
            "Round",
            "Selu",
            "Sigmoid",
            "Sign",
            "Sin",
            "Sinh",
            "Softplus",
            "Softsign",
            "Sqrt",
            "Swish",
            "Tan",
            "Tanh",
        ]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.BOOL,
        ]
    ],
)
def test_onnx_arith_unary_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, ONNX_OPSET_DOMAIN, dtype_proto
):
    """
    Test ONNX arith unary operators lowering.
    """

    if ONNX_OP_NAME == "Bernoulli":
        pytest.skip(f"{ONNX_OP_NAME} statistical behaviour cannot be tested as unary.")

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    is_bool_output_op = ONNX_OP_NAME in ["IsInf", "IsNaN"]
    out_dtype_proto = TensorProto.BOOL if is_bool_output_op else dtype_proto
    out_np_dtype = np.bool_ if is_bool_output_op else np_dtype

    def create_onnx_model(np_array, inp_dtype_proto, out_dtype_proto):
        inp_tensor = make_tensor_value_info("input", inp_dtype_proto, np_array.shape)
        out_tensor = make_tensor_value_info("output", out_dtype_proto, np_array.shape)
        arith_node = make_node(
            ONNX_OP_NAME,
            inputs=["input"],
            outputs=["output"],
            domain=ONNX_OPSET_DOMAIN,
        )
        graph = make_graph(
            nodes=[arith_node],
            name="arith_graph",
            inputs=[inp_tensor],
            outputs=[out_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid(ONNX_OPSET_DOMAIN, ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    np_array = _generate_dtype_random(np_dtype, shape=(2, 2), max_val=10)
    onnx_model = create_onnx_model(np_array, dtype_proto, out_dtype_proto)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"{ONNX_OP_NAME} V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(np_array, dtype=out_np_dtype)
        outputs = runner(llvm_module, "main", [np_array], [output])

        ref = ReferenceEvaluator(onnx_model)
        onnx_result = ref.run(None, {"input": np_array})[0]

        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto",
    [
        (schema.name, schema.since_version, dtype_proto)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Hardmax", "Softmax", "LogSoftmax"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE,
        ]
    ],
)
def test_onnx_softmax_lower(ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto):
    """
    Test ONNX softmax family of operators lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model(np_array, dtype_proto):
        input_tensor = make_tensor_value_info("input", dtype_proto, np_array.shape)
        output_tensor = make_tensor_value_info("output", dtype_proto, np_array.shape)
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

    np_array = _generate_dtype_random(np_dtype, shape=(8, 8), max_val=10)
    onnx_model = create_onnx_model(np_array, dtype_proto)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input": np_array})[0]

    with Context() as ctx, Location.unknown():

        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(np_array)
        outputs = runner(llvm_module, "main", [np_array], [output])

        atol = 1e-1 if np_dtype == np.float16 else 1e-5
        rtol = 1e-1 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto",
    [
        (schema.since_version, dtype_proto)
        for schema in get_all_schemas_with_history()
        if "Transpose" == schema.name
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.INT8,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.UINT32,
            TensorProto.UINT64,
            TensorProto.BOOL,
        ]
    ],
)
def test_onnx_transpose_lower(ONNX_OPSET_VERSION, dtype_proto):
    """
    Test ONNX Transpose operator lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model(np_array, dtype_proto):

        perm = random.sample(range(np_array.ndim), np_array.ndim)
        np_arrayT = np_array.transpose(perm)

        input_tensor = make_tensor_value_info("input", dtype_proto, np_array.shape)
        output_tensor = make_tensor_value_info("output", dtype_proto, np_arrayT.shape)
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

    np_array = _generate_dtype_random(np_dtype, shape=(1, 3, 8, 5), max_val=10)

    onnx_model = create_onnx_model(np_array, dtype_proto)

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
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shapes",
    [
        (schema.name, schema.since_version, dtype_proto, shapes)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Greather", "GreaterOrEqual", "Less", "LessOrEqual"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.INT8,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.UINT8,
            TensorProto.UINT32,
            TensorProto.UINT64,
            TensorProto.BOOL,
        ]
        for shapes in [
            [(1, 3, 3), (1, 3, 3)],  # Non-broadcasting
            [(1, 3, 1), (4, 1, 5)],  # Broadcasting
        ]
    ],
)
def test_onnx_compare_binary_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shapes
):
    """
    Test ONNX comparison binary operators lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model(inp_array0, inp_array1, dtype_proto):
        input_tensor_0 = make_tensor_value_info("input0", dtype_proto, inp_array0.shape)
        input_tensor_1 = make_tensor_value_info("input1", dtype_proto, inp_array1.shape)
        output_tensor = make_tensor_value_info(
            "output", TensorProto.BOOL, (inp_array0 + inp_array1).shape
        )
        arith_node = make_node(
            ONNX_OP_NAME,
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

    inp_array0 = _generate_dtype_random(np_dtype, shape=shapes[0], max_val=10)
    inp_array1 = _generate_dtype_random(np_dtype, shape=shapes[1], max_val=10)

    onnx_model = create_onnx_model(inp_array0, inp_array1, dtype_proto)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"{ONNX_OP_NAME} V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        ref = ReferenceEvaluator(onnx_model)
        onnx_result = ref.run(None, {"input0": inp_array0, "input1": inp_array1})[0]

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [inp_array0, inp_array1], [res_array])
        np.testing.assert_array_equal(outputs[0], onnx_result)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto",
    [
        (schema.since_version, dtype_proto)
        for schema in get_all_schemas_with_history()
        if schema.name == "Gemm"
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.UINT32,
            TensorProto.UINT64,
        ]
    ],
)
@pytest.mark.parametrize(
    "shape_cfg, has_bias, alpha, beta, transA, transB",
    [
        # Standard shape with bias, default alpha=1.0f, beta=1.0f
        ((16, 32, 16), True, 1.0, 1.0, 0, 0),
        # Standard shape without bias (bias absence), default alpha=1.0f, beta=1.0f
        ((16, 32, 16), False, 1.0, 1.0, 0, 0),
        # Custom shape with bias, non-default alpha and beta
        ((8, 16, 24), True, 0.5, 2.0, 0, 0),
        # Custom shape without bias, alpha scaling
        ((32, 16, 32), False, 2.0, 1.0, 0, 0),
        # Tall/skinny matrix shape, beta=0.0 with bias, transA=1
        ((1, 64, 32), True, 1.0, 0.0, 1, 0),
        # Non-power-of-2 dimensions with bias, transB=1
        ((35, 17, 23), True, 1.5, 0.5, 0, 1),
        # Small odd dimensions without bias, both transA=1 and transB=1
        ((7, 13, 11), False, 1.0, 1.0, 1, 1),
    ],
)
# pylint: disable=too-many-statements,too-many-arguments,too-many-positional-arguments
def test_onnx_gemm_lower(
    ONNX_OPSET_VERSION, dtype_proto, shape_cfg, has_bias, alpha, beta, transA, transB
):
    """
    Test ONNX Gemm operator lowering.
    """

    class Gemm(OpRun):
        """
        ONNX Gemm operator.
        Computes:
            Y = alpha * A' * B' + beta * C
        where:
            A' = A.T if transA != 0 else A
            B' = B.T if transB != 0 else B
        """

        # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        def _run(self, A, B, C=None, alpha=1.0, beta=1.0, transA=0, transB=0):
            a_mat = A.T if transA != 0 else A
            b_mat = B.T if transB != 0 else B
            prod = np.matmul(a_mat, b_mat)
            if alpha != 1.0:
                prod = prod * alpha
            if C is not None and beta != 0.0:
                bias = C * beta if beta != 1.0 else C
                res = prod + bias
            else:
                res = prod
            if res.dtype != A.dtype:
                res = res.astype(A.dtype)

            return (res,)

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    # pylint: disable=unused-variable
    def create_onnx_model(inp_arr0, inp_arr1, inp_bias, dtype_proto):
        m = inp_arr0.shape[0] if transA == 0 else inp_arr0.shape[1]
        k = inp_arr0.shape[1] if transA == 0 else inp_arr0.shape[0]
        n = inp_arr1.shape[1] if transB == 0 else inp_arr1.shape[0]

        input_tensor_0 = make_tensor_value_info(
            "input0", dtype_proto, list(inp_arr0.shape)
        )
        input_tensor_1 = make_tensor_value_info(
            "input1", dtype_proto, list(inp_arr1.shape)
        )
        output_tensor = make_tensor_value_info("output", dtype_proto, [m, n])

        inputs = ["input0", "input1"]
        graph_inputs = [input_tensor_0, input_tensor_1]
        initializers = []

        if inp_bias is not None:
            input_tensor_2 = make_tensor_value_info(
                "bias0", dtype_proto, list(inp_bias.shape)
            )
            bias_init = make_tensor(
                "bias0", dtype_proto, list(inp_bias.shape), inp_bias.flatten().tolist()
            )
            inputs.append("bias0")
            graph_inputs.append(input_tensor_2)
            initializers.append(bias_init)

        arith_node = make_node(
            "Gemm",
            inputs,
            ["output"],
            alpha=float(alpha),
            beta=float(beta),
            transA=int(transA),
            transB=int(transB),
        )
        graph = make_graph(
            nodes=[arith_node],
            name="gemm_graph",
            inputs=graph_inputs,
            outputs=[output_tensor],
            initializer=initializers,
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    m, k, n = shape_cfg
    shape_a = (k, m) if transA else (m, k)
    shape_b = (n, k) if transB else (k, n)

    inp_arr0 = _generate_dtype_random(np_dtype, shape=shape_a, max_val=10)
    inp_arr1 = _generate_dtype_random(np_dtype, shape=shape_b, max_val=10)
    inp_bias = (
        _generate_dtype_random(np_dtype, shape=(m, n), max_val=5) if has_bias else None
    )

    if ONNX_OPSET_VERSION < 11 and not has_bias:
        pytest.skip(f"Gemm V{ONNX_OPSET_VERSION} requires bias input C (min=3 inputs)")

    if np.issubdtype(np_dtype, np.integer) and (alpha % 1 != 0 or beta % 1 != 0):
        pytest.skip(
            f"Gemm with integer dtype {TensorProto.DataType.Name(dtype_proto)} "
            f"does not support non-integer alpha/beta scaling ({alpha}, {beta})"
        )

    onnx_model = create_onnx_model(inp_arr0, inp_arr1, inp_bias, dtype_proto)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op result", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Constant V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Gemm V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        new_ops = [Gemm]
        ref = ReferenceEvaluator(onnx_model, new_ops=new_ops)

        inputs_dict = {"input0": inp_arr0, "input1": inp_arr1}
        runner_inputs = [inp_arr0, inp_arr1]
        if inp_bias is not None:
            inputs_dict["bias0"] = inp_bias
            runner_inputs.append(inp_bias)

        onnx_result = ref.run(None, inputs_dict)[0]

        res_arr = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", runner_inputs, [res_arr])

        if np.issubdtype(np_dtype, np.integer):
            atol = 0
            rtol = 0
        elif np_dtype == np.float16:
            atol = 1e-1
            rtol = 1e-1
        else:
            atol = 1e-3
            rtol = 1e-3

        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, shapes",
    [
        (opset, dtype_proto, shapes)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Where" == schema.name
        ]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.INT32,
            TensorProto.INT8,
            TensorProto.UINT8,
            TensorProto.BOOL,
        ]
        for shapes in [
            ((4, 4), (4, 4), (4, 4)),  # Standard
            ((1,), (4, 4), (4, 4)),  # Broadcast Condition
            ((4, 1), (1, 4), (4, 4)),  # Multi-directional broadcast
        ]
    ],
)
# pylint: disable=too-many-locals
def test_onnx_where_lower(ONNX_OPSET_VERSION, dtype_proto, shapes):
    """
    Test ONNX Where operator lowering.
    """
    cond_shape, x_shape, y_shape = shapes
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)
    res_shape = np.broadcast(
        np.empty(cond_shape), np.empty(x_shape), np.empty(y_shape)
    ).shape

    def create_onnx_model():
        input_cond = make_tensor_value_info("condition", TensorProto.BOOL, cond_shape)
        input_x = make_tensor_value_info("X", dtype_proto, x_shape)
        input_y = make_tensor_value_info("Y", dtype_proto, y_shape)
        output_tensor = make_tensor_value_info("output", dtype_proto, res_shape)

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

    cond_arr = _generate_dtype_random(np.bool, shape=cond_shape)
    x_arr = _generate_dtype_random(np_dtype, shape=x_shape, max_val=127)
    y_arr = _generate_dtype_random(np_dtype, shape=y_shape, max_val=127)

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
    "ONNX_OPSET_VERSION, dtype_proto, shape, axes",
    [
        (opset, dtype_proto, shape, axes)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Unsqueeze" == schema.name
        ]
        for dtype_proto in [TensorProto.FLOAT, TensorProto.INT32]
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
def test_onnx_unsqueeze_lower(ONNX_OPSET_VERSION, dtype_proto, shape, axes):
    """
    Test ONNX Unsqueeze operator lowering.
    """
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    res_shape = list(shape)
    for axis in sorted(axes):
        res_shape.insert(axis, 1)

    def create_onnx_model():
        input_tensor = make_tensor_value_info("data", dtype_proto, shape)
        output_tensor = make_tensor_value_info("output", dtype_proto, res_shape)

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
            name="unsqueeze_opset_{ONNX_OPSET_VERSION}",
            inputs=inputs,
            outputs=[output_tensor],
            initializer=initializers,
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    data_arr = _generate_dtype_random(np_dtype, shape=shape, max_val=127)

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
    "ONNX_OPSET_VERSION, dtype_proto, shape, axes",
    [
        (opset, dtype_proto, shape, axes)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Squeeze" == schema.name
        ]
        for dtype_proto in [TensorProto.FLOAT, TensorProto.INT32]
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
def test_onnx_squeeze_lower(ONNX_OPSET_VERSION, dtype_proto, shape, axes):
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

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model():
        input_tensor = make_tensor_value_info("data", dtype_proto, shape)
        output_tensor = make_tensor_value_info("output", dtype_proto, res_shape)

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
            name="squeeze_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=initializers,
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    data_arr = _generate_dtype_random(np_dtype, shape=shape, max_val=127)

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
    "ONNX_OPSET_VERSION, dtype_proto, input_shape, kernel, strides, pads",
    [
        (opset, dtype_proto, shape, kernel, stride, pad)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "MaxPool" == schema.name
        ]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.INT8,
            TensorProto.UINT8,
        ]
        for shape, kernel, stride, pad in [
            ((1, 3, 32, 32), [2, 2], [2, 2], [0, 0, 0, 0]),  # Standard NCHW
            ((1, 1, 10, 10), [3, 2], [1, 2], [0, 0, 0, 0]),  # Non-square
            ((1, 1, 5, 5), [3, 3], [1, 1], [1, 1, 1, 1]),  # With padding
        ]
    ],
)
# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def test_onnx_maxpool_lower(
    ONNX_OPSET_VERSION, dtype_proto, input_shape, kernel, strides, pads
):
    """
    Test ONNX MaxPool operator lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    h_in, w_in = input_shape[2], input_shape[3]
    h_out = (h_in + pads[0] + pads[2] - kernel[0]) // strides[0] + 1
    w_out = (w_in + pads[1] + pads[3] - kernel[1]) // strides[1] + 1
    output_shape = (input_shape[0], input_shape[1], h_out, w_out)

    def create_onnx_model():
        input_x = make_tensor_value_info("X", dtype_proto, input_shape)
        output_y = make_tensor_value_info("Y", dtype_proto, output_shape)

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

    x_arr = _generate_dtype_random(np_dtype, shape=input_shape, max_val=127)

    onnx_model = create_onnx_model()

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"MaxPool V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros(output_shape, dtype=np_dtype)
        outputs = runner(llvm_module, "main", [x_arr], [res_arr])

        if dtype_proto == TensorProto.INT8:
            float_model = create_onnx_model()
            float_model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
            float_model.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
            ref = ReferenceEvaluator(float_model)
            onnx_result = ref.run(None, {"X": x_arr.astype(np.float32)})[0].astype(
                np.int8
            )
        else:
            ref = ReferenceEvaluator(onnx_model)
            onnx_result = ref.run(None, {"X": x_arr})[0]

        atol = 1e-2 if np_dtype == np.float16 else 1e-5
        rtol = 1e-2 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype, input_shape, weight_shape, strides, pads, group, has_bias",
    [
        (opset, dtype, in_shape, w_shape, stride, pad, group, bias)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Conv" == schema.name
        ]
        for dtype in [TensorProto.FLOAT, TensorProto.FLOAT16]
        for in_shape, w_shape, stride, pad, group, bias in [
            ((1, 3, 32, 32), (8, 3, 3, 3), [1, 1], [0, 0, 0, 0], 1, False),
            ((1, 3, 32, 32), (16, 3, 3, 3), [2, 2], [1, 1, 1, 1], 1, True),
            ((2, 1, 10, 10), (1, 1, 5, 5), [1, 1], [2, 2, 2, 2], 1, False),
            # Grouped Convolution (group = 2)
            ((1, 4, 16, 16), (8, 2, 3, 3), [1, 1], [1, 1, 1, 1], 2, False),
            # Grouped Convolution with stride = 2 (group = 4)
            ((2, 8, 20, 20), (16, 2, 3, 3), [2, 2], [1, 1, 1, 1], 4, True),
            # Depthwise Convolution (group = 256)
            ((1, 256, 20, 20), (256, 1, 3, 3), [1, 1], [1, 1, 1, 1], 256, True),
        ]
    ],
)
# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def test_onnx_conv_lower(
    ONNX_OPSET_VERSION, dtype, input_shape, weight_shape, strides, pads, group, has_bias
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

    x_arr = _generate_dtype_random(np_dtype, shape=input_shape, max_val=5)
    w_arr = _generate_dtype_random(np_dtype, shape=weight_shape, max_val=5)
    b_arr = _generate_dtype_random(np_dtype, shape=(f), max_val=5) if has_bias else None

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
            group=group,
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
    "ONNX_OPSET_VERSION, dtype_proto, input_shape, axis",
    [
        (ONNX_OPSET_VERSION, dtype_proto, shape, ax)
        for ONNX_OPSET_VERSION in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Flatten" == schema.name
        ]
        for dtype_proto in [
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
def test_onnx_flatten_lower(ONNX_OPSET_VERSION, dtype_proto, input_shape, axis):
    """
    Test ONNX Flatten operator lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    x_arr = _generate_dtype_random(np_dtype, shape=input_shape, max_val=127)

    rank = len(input_shape)
    norm_axis = axis if axis >= 0 else axis + rank

    dim0 = int(np.prod(input_shape[:norm_axis]))
    dim1 = int(np.prod(input_shape[norm_axis:]))
    output_shape = (dim0, dim1)

    def create_onnx_model():
        input_x = make_tensor_value_info("X", dtype_proto, input_shape)
        output_y = make_tensor_value_info("Y", dtype_proto, output_shape)

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
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Flatten V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

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
            TensorProto.BOOL,
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

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    inp0 = _generate_dtype_random(np_dtype, shape=shape0, max_val=127)
    inp1 = _generate_dtype_random(np_dtype, shape=shape1, max_val=127)

    def create_onnx_model(inp0, inp1, dtype_proto):

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
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(inp0, inp1, dtype_proto)
    check_model(onnx_model)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"{ONNX_OP_NAME} V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        ref = ReferenceEvaluator(onnx_model)
        onnx_result = ref.run(None, {"input0": inp0, "input1": inp1})[0]

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
            TensorProto.BOOL,
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
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    inp0 = _generate_dtype_random(np_dtype, shape=shape, max_val=127)

    def create_onnx_model(inp0, dtype_proto):

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
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(inp0, dtype_proto)
    check_model(onnx_model)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"{ONNX_OP_NAME} V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        ref = ReferenceEvaluator(onnx_model)
        onnx_result = ref.run(None, {"input0": inp0})[0]

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
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    inp0 = _generate_dtype_random(np_dtype, shape=shape0, max_val=127)
    inp1 = _generate_dtype_random(np_dtype, shape=shape1, max_val=127)

    def create_onnx_model(inp0, inp1, dtype_proto):

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
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(inp0, inp1, dtype_proto)
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
    "ONNX_OPSET_VERSION, dtype_proto, shape",
    [
        (schema.since_version, dtype_proto, shape)
        for schema in get_all_schemas_with_history()
        if schema.name in ["GlobalAveragePool"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE,
        ]
        for shape in [
            (2, 3, 10),  # 3D (1D spatial) #
            (2, 3, 4, 5),  # 4D (2D spatial NCHW) #
            (2, 3, 2, 4, 4),  # 5D (3D spatial NCDHW)
        ]
    ],
)
def test_onnx_globalaveragepool_lower(ONNX_OPSET_VERSION, dtype_proto, shape):
    """
    Test ONNX GlobalAveragePooling lowering.
    """
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    # Generate random test inputs (negative & positive values)
    inp0 = _generate_dtype_random(np_dtype, shape=shape, max_val=127)

    def create_onnx_model(inp0, dtype_proto):
        # Global pooling output shape: (N, C, 1, 1, ...)
        out_shape = list(shape[:2]) + [1] * (len(shape) - 2)

        input_tensor = make_tensor_value_info("input0", dtype_proto, inp0.shape)
        output_tensor = make_tensor_value_info("output", dtype_proto, out_shape)

        pool_node = make_node("GlobalAveragePool", ["input0"], ["output"])
        graph = make_graph(
            nodes=[pool_node],
            name=f"globalaveragepool_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(inp0, dtype_proto)
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
    "ONNX_OPSET_VERSION, dtype_proto, shape, p_val",
    [
        (schema.since_version, dtype_proto, shape, p_val)
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
def test_onnx_globallppool_lower(ONNX_OPSET_VERSION, dtype_proto, shape, p_val):
    """
    Test ONNX GlobalLpPool lowering.
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

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    inp0 = _generate_dtype_random(np_dtype, shape=shape, max_val=127)

    def create_onnx_model(inp0, dtype_proto, p_val):
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
                    p_val = np_dtype.type(p_val)

            kwargs["p"] = p_val

        pool_node = make_node("GlobalLpPool", ["input0"], ["output"], **kwargs)
        graph = make_graph(
            nodes=[pool_node],
            name=f"globallppool_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(inp0, dtype_proto, p_val)
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
    "ONNX_OPSET_VERSION, dtype_proto, shape",
    [
        (schema.since_version, dtype_proto, shape)
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
def test_onnx_globalmaxpool_lower(ONNX_OPSET_VERSION, dtype_proto, shape):
    """
    Test ONNX GlobalMaxPool lowering.
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

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    # Generate random test inputs (negative & positive values)
    inp0 = _generate_dtype_random(np_dtype, shape=shape, max_val=127)

    def create_onnx_model(inp0, dtype_proto):
        # Global pooling output shape: (N, C, 1, 1, ...)
        out_shape = list(shape[:2]) + [1] * (len(shape) - 2)
        input_tensor = make_tensor_value_info("input0", dtype_proto, inp0.shape)
        output_tensor = make_tensor_value_info("output", dtype_proto, out_shape)

        pool_node = make_node("GlobalMaxPool", ["input0"], ["output"])
        graph = make_graph(
            nodes=[pool_node],
            name=f"globalmaxpool_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(inp0, dtype_proto)
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
    "ONNX_OPSET_VERSION, dtype_proto, shape, axis, split_sizes",
    [
        (opset, dtype_proto, shape, axis, split_sizes)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Split" == schema.name
        ]
        for dtype_proto in [TensorProto.FLOAT, TensorProto.INT32]
        # Opset 1 only supports Floating-point types
        if not (opset == 1 and dtype_proto != TensorProto.FLOAT)
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
def test_onnx_split_lower(ONNX_OPSET_VERSION, dtype_proto, shape, axis, split_sizes):
    """
    Test ONNX Split lowering.
    """

    # ReferenceEvaluator
    class Split(OpRun):
        """
        SplitOp reference implementation.
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

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

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
        input_tensor = make_tensor_value_info("input", dtype_proto, shape)
        output_names = [f"output_{i}" for i in range(num_outputs)]
        output_tensors = [
            make_tensor_value_info(name, dtype_proto, out_shape)
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
            name=f"split_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_tensor],
            outputs=output_tensors,
            initializer=initializers,
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    data_arr = _generate_dtype_random(np_dtype, shape=shape, max_val=127)
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


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, input_shapes, axis",
    [
        (opset, dtype_proto, shapes, axis)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Concat" == schema.name
        ]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.INT32,
            TensorProto.UINT32,
            TensorProto.INT8,
            TensorProto.UINT8,
            TensorProto.BOOL,
        ]
        for shapes, axis in [
            ([(2, 3), (2, 3)], 0),  # Concat along dim 0
            ([(3, 4), (3, 4)], 1),  # Concat along dim 1
            ([(2, 2, 2), (2, 2, 2)], 2),  # Concat along dim 2
            ([(2, 3), (2, 3)], -1),  # Negative axis
        ]
    ],
)
def test_onnx_concat_lower(ONNX_OPSET_VERSION, dtype_proto, input_shapes, axis):
    """
    Test ONNX Concat operator lowering.
    """
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    num_inputs = len(input_shapes)
    input_names = [f"input_{i}" for i in range(num_inputs)]

    # Calculate output shape based on input shapes and axis
    rank = len(input_shapes[0])
    norm_axis = axis if axis >= 0 else axis + rank

    output_shape = list(input_shapes[0])
    concat_dim_sum = sum(shape[norm_axis] for shape in input_shapes)
    output_shape[norm_axis] = concat_dim_sum

    def create_onnx_model():
        input_tensors = [
            make_tensor_value_info(name, dtype_proto, shape)
            for name, shape in zip(input_names, input_shapes)
        ]
        output_tensor = make_tensor_value_info("output", dtype_proto, output_shape)

        kwargs = {"axis": axis}

        concat_node = make_node(
            "Concat",
            input_names,
            ["output"],
            **kwargs,
        )

        graph = make_graph(
            nodes=[concat_node],
            name=f"concat_opset_{ONNX_OPSET_VERSION}",
            inputs=input_tensors,
            outputs=[output_tensor],
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    data_arrs = [
        _generate_dtype_random(np_dtype, shape=shape, max_val=127)
        for shape in input_shapes
    ]

    onnx_model = create_onnx_model()

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Concat V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        ref = ReferenceEvaluator(onnx_model)
        onnx_results = ref.run(None, dict(zip(input_names, data_arrs)))

        outputs = runner(
            llvm_module, "main", data_arrs, [np.zeros(output_shape, dtype=np_dtype)]
        )

        np.testing.assert_allclose(outputs[0], onnx_results[0], atol=1e-5)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, input_shape, target_shape",
    [
        (opset, dtype_proto, in_shape, out_shape)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Reshape" == schema.name
        ]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.INT32,
            TensorProto.UINT8,
            TensorProto.BOOL,
        ]
        for in_shape, out_shape in [
            ((2, 3), (3, 2)),  # Transpose-like reshape
            ((2, 4), (8,)),  # Flatten
            ((1, 2, 2), (4,)),  # 3D to 1D
            ((2, 2, 1), (1, 4)),  # 3D to 2D
        ]
    ],
)
def test_onnx_reshape_lower(ONNX_OPSET_VERSION, dtype_proto, input_shape, target_shape):
    """
    Test ONNX Reshape operator lowering.
    """

    # ReferenceEvaluator
    class Reshape(OpRun):
        """
        ReshapeOp reference implementation.
        """

        @staticmethod
        def _impl(
            data: np.ndarray, shape: np.ndarray, allowzero: int = 0
        ) -> np.ndarray:
            new_shape = np.copy(shape)
            if allowzero == 0:
                zeros_index = np.where(shape == 0)
                new_shape[zeros_index] = np.array(data.shape)[zeros_index]
            return np.reshape(data, new_shape)

        # pylint: disable=arguments-differ,unused-argument
        def _run(self, data, shape=None, allowzero=0):
            target = (
                shape if shape is not None else getattr(self, "shape", target_shape)
            )
            return (self._impl(data, np.asarray(target), 0),)

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)
    shape_arr = np.array(target_shape, dtype=np.int64)

    def create_onnx_model():
        input_tensor = make_tensor_value_info("input", dtype_proto, input_shape)
        output_tensor = make_tensor_value_info("output", dtype_proto, target_shape)

        inputs = [input_tensor]
        kwargs = {}

        if ONNX_OPSET_VERSION < 5:
            # Opsets 1 - 4: shape is an attribute
            kwargs["shape"] = target_shape
        else:
            # Opsets 5+: shape is an input operand
            shape_info = make_tensor_value_info(
                "shape", TensorProto.INT64, [len(target_shape)]
            )
            inputs.append(shape_info)

        reshape_node = make_node(
            "Reshape",
            [inp.name if hasattr(inp, "name") else inp for inp in inputs],
            ["output"],
            **kwargs,
        )

        graph = make_graph(
            nodes=[reshape_node],
            name=f"reshape_opset_{ONNX_OPSET_VERSION}",
            inputs=inputs,
            outputs=[output_tensor],
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    data_arr = _generate_dtype_random(np_dtype, shape=input_shape, max_val=127)
    onnx_model = create_onnx_model()

    new_ops = [Reshape] if ONNX_OPSET_VERSION == 1 else []
    ref = ReferenceEvaluator(onnx_model, new_ops=new_ops)
    feed_dict = {"input": data_arr}
    runner_inputs = [data_arr]

    if ONNX_OPSET_VERSION >= 5:
        feed_dict["shape"] = shape_arr
        runner_inputs.append(shape_arr)

    onnx_results = ref.run(None, feed_dict)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Reshape V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros(target_shape, dtype=np_dtype)
        outputs = runner(llvm_module, "main", runner_inputs, [res_arr])

        np.testing.assert_allclose(outputs[0], onnx_results[0], atol=1e-5)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape_a, shape_b",
    [
        (schema.name, schema.since_version, dtype_proto, shape_a, shape_b)
        for schema in get_all_schemas_with_history()
        if schema.name in ["MatMul", "MatMulInteger"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.UINT32,
            TensorProto.INT64,
            TensorProto.INT8,
        ]
        for shape_a, shape_b in [
            ((2, 3), (3, 4)),  # 2D Standard GEMM
            ((1, 4), (4, 2)),  # 2D Vector-Matrix
            ((3, 2, 4), (3, 4, 2)),  # 3D Batch MatMul
            ((1, 2, 3), (4, 3, 2)),  # 3D Broadcast MatMul
            ((2, 1, 3, 4), (2, 5, 4, 2)),  # 4D N-D Broadcast MatMul
            ((4,), (4, 3)),  # 1D x 2D MatMul
            ((3, 4), (4,)),  # 2D x 1D MatMul
            ((5,), (5,)),  # 1D x 1D Dot Product
        ]
    ],
)
# pylint: disable=too-many-branches,too-many-statements
def test_onnx_matmul_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape_a, shape_b
):
    """
    Test ONNX MatMul operator lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    data_a = _generate_dtype_random(np_dtype, shape=shape_a, max_val=5)
    data_b = _generate_dtype_random(np_dtype, shape=shape_b, max_val=5)

    expected_out = np.matmul(data_a, data_b)
    target_shape = expected_out.shape

    # enforce minimum rank 1 results
    if len(expected_out.shape) == 0:
        target_shape = (1,)

    def create_onnx_model(dtype_proto):
        input_a = make_tensor_value_info("A", dtype_proto, shape_a)
        input_b = make_tensor_value_info("B", dtype_proto, shape_b)

        if ONNX_OP_NAME == "MatMulInteger":
            dtype_proto = TensorProto.INT32

        output_y = make_tensor_value_info("Y", dtype_proto, target_shape)

        matmul_node = make_node(
            ONNX_OP_NAME,
            ["A", "B"],
            ["Y"],
        )

        graph = make_graph(
            nodes=[matmul_node],
            name="matmul_graph_test",
            inputs=[input_a, input_b],
            outputs=[output_y],
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    onnx_model = create_onnx_model(dtype_proto)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"{ONNX_OP_NAME} V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros(
            target_shape,
            dtype=np.int32 if ONNX_OP_NAME == "MatMulInteger" else np_dtype,
        )
        outputs = runner(llvm_module, "main", [data_a, data_b], [res_arr])

        ref = ReferenceEvaluator(onnx_model)
        onnx_results = ref.run(None, {"A": data_a, "B": data_b})

        atol = 1e-2 if np_dtype == np.float16 else 1e-5
        rtol = 1e-2 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_results[0], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, mode, coord_trans_mode, nearest_mode, resize_type, input_shape, scale_or_size",
    [
        (opset, dtype_proto, mode, coord_trans, nearest, r_type, in_shape, param)
        for opset in [
            schema.since_version
            for schema in get_all_schemas_with_history()
            if "Resize" == schema.name
        ]
        if not (opset in [10, 11])  # buggy refeval
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.UINT8,
        ]
        for mode in ["nearest"]
        for coord_trans in [
            "half_pixel",
            "asymmetric",
            "align_corners",
            # "tf_half_pixel_for_nn", # missing refeval
            "pytorch_half_pixel",
        ]
        for nearest in [
            "round_prefer_floor",
            "round_prefer_ceil",
            "floor",
            "ceil",
        ]
        for r_type, in_shape, param in [
            ("scales", (1, 1, 4, 4), (1.0, 1.0, 2.0, 2.0)),  # Upsampling via scales
            ("scales", (2, 6), (1.0, 0.5)),  # Downsampling via scales
            ("sizes", (1, 1, 4, 4), (1, 1, 8, 8)),  # Upsampling via sizes
            ("sizes", (2, 6), (2, 3)),  # Downsampling via sizes
        ]
        if not (opset == 10 and r_type == "sizes")  # Opset 10 only supports scales
        if not (
            opset == 10 and coord_trans != "half_pixel" and coord_trans != "asymmetric"
        )
    ],
)
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-statements
def test_onnx_resize_lower(
    ONNX_OPSET_VERSION,
    dtype_proto,
    mode,
    coord_trans_mode,
    nearest_mode,
    resize_type,
    input_shape,
    scale_or_size,
):
    """
    Test ONNX Resize operator lowering to Linalg dialect across all opsets and modes.
    """
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    if resize_type == "scales":
        scales_arr = np.array(scale_or_size, dtype=np.float32)
        target_shape = tuple(
            int(np.round(input_shape[i] * scales_arr[i]))
            for i in range(len(input_shape))
        )
        sizes_arr = np.array(target_shape, dtype=np.int64)
    else:
        sizes_arr = np.array(scale_or_size, dtype=np.int64)
        target_shape = tuple(int(x) for x in sizes_arr)
        scales_arr = np.array(
            [target_shape[i] / input_shape[i] for i in range(len(input_shape))],
            dtype=np.float32,
        )

    def create_onnx_model():
        input_tensor = make_tensor_value_info("X", dtype_proto, input_shape)
        output_tensor = make_tensor_value_info("output", dtype_proto, target_shape)

        inputs = [input_tensor]
        node_inputs = ["X"]
        kwargs = {
            "mode": mode,
        }

        if ONNX_OPSET_VERSION >= 11:
            kwargs["coordinate_transformation_mode"] = coord_trans_mode
            kwargs["nearest_mode"] = nearest_mode

        if ONNX_OPSET_VERSION == 10:
            scales_info = make_tensor_value_info(
                "scales", TensorProto.FLOAT, [len(input_shape)]
            )
            inputs.append(scales_info)
            node_inputs.append("scales")
        else:
            roi_info = make_tensor_value_info("roi", TensorProto.FLOAT, [0])
            inputs.append(roi_info)

            if resize_type == "scales":
                scales_info = make_tensor_value_info(
                    "scales", TensorProto.FLOAT, [len(input_shape)]
                )
                sizes_info = make_tensor_value_info("sizes", TensorProto.INT64, [0])
                inputs.extend([scales_info, sizes_info])
                node_inputs.extend(["roi", "scales", ""])
            else:
                scales_info = make_tensor_value_info("scales", TensorProto.FLOAT, [0])
                sizes_info = make_tensor_value_info(
                    "sizes", TensorProto.INT64, [len(input_shape)]
                )
                inputs.extend([scales_info, sizes_info])
                node_inputs.extend(["roi", "", "sizes"])

        resize_node = make_node(
            "Resize",
            node_inputs,
            ["output"],
            **kwargs,
        )

        graph = make_graph(
            nodes=[resize_node],
            name=f"resize_opset_{ONNX_OPSET_VERSION}",
            inputs=inputs,
            outputs=[output_tensor],
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    data_arr = _generate_dtype_random(np_dtype, shape=input_shape, max_val=127)

    onnx_model = create_onnx_model()

    feed_dict = {"X": data_arr}
    runner_inputs = [data_arr]

    if ONNX_OPSET_VERSION == 10:
        feed_dict["scales"] = scales_arr
        runner_inputs.append(scales_arr)
    else:
        empty_roi = np.array([], dtype=np.float32)
        feed_dict["roi"] = empty_roi
        runner_inputs.append(empty_roi)

        if resize_type == "scales":
            feed_dict["scales"] = scales_arr
            feed_dict["sizes"] = np.array([], dtype=np.int64)
            runner_inputs.extend([scales_arr, np.array([], dtype=np.int64)])
        else:
            feed_dict["scales"] = np.array([], dtype=np.float32)
            feed_dict["sizes"] = sizes_arr
            runner_inputs.extend([np.array([], dtype=np.float32), sizes_arr])

    ref = ReferenceEvaluator(onnx_model)

    onnx_results = ref.run(None, feed_dict)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_arr = np.zeros(target_shape, dtype=np_dtype)
        outputs = runner(llvm_module, "main", runner_inputs, [res_arr])

        np.testing.assert_allclose(outputs[0], onnx_results[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, shape, start_end",
    [
        (schema.since_version, dtype_proto, shape, start_end)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Shape"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.FLOAT16,
        ]
        for shape in [
            (10,),  # 1D
            (2, 3),  # 2D
            (2, 3, 10),  # 3D
            (2, 3, 4, 5),  # 4D
            (2, 3, 2, 4, 4),  # 5D
        ]
        for start_end in (
            # Test different start/end slicing configurations
            [(None, None), (0, None), (1, 3), (-2, None), (0, -1)]
            if schema.since_version >= 15
            else [(None, None)]  # Opset < 15 does not support start/end
        )
    ],
)
def test_onnx_shape_lower(ONNX_OPSET_VERSION, dtype_proto, shape, start_end):
    """
    Test ONNX Shape operation lowering.
    """
    start_attr, end_attr = start_end

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    inp0 = _generate_dtype_random(np_dtype, shape=shape, max_val=127)

    def create_onnx_model(inp0, dtype_proto):

        node_kwargs = {}
        if start_attr is not None:
            node_kwargs["start"] = start_attr
        if end_attr is not None:
            node_kwargs["end"] = end_attr

        # Compute expected output shape and values
        rank = len(shape)
        s = start_attr if start_attr is not None else 0
        if s < 0:
            s += rank
        s = max(0, min(s, rank))

        e = end_attr if end_attr is not None else rank
        if e < 0:
            e += rank
        e = max(0, min(e, rank))

        expected_output_dim = max(0, e - s)
        out_shape = (expected_output_dim,)

        input_tensor = make_tensor_value_info("input0", dtype_proto, inp0.shape)
        output_tensor = make_tensor_value_info("output", TensorProto.INT64, out_shape)

        shape_node = make_node(
            "Shape",
            inputs=["input0"],
            outputs=["output"],
            **node_kwargs,
        )

        graph = make_graph(
            nodes=[shape_node],
            name="shape_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(inp0, dtype_proto)
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

        # ONNX Shape results demand exact array equality
        np.testing.assert_array_equal(outputs[0], onnx_result)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, indices_dtype_proto, data_shape, indices_shape, axis",
    [
        (
            schema.since_version,
            dtype_proto,
            indices_dtype_proto,
            data_shape,
            indices_shape,
            axis,
        )
        for schema in get_all_schemas_with_history()
        if schema.name in ["Gather"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.FLOAT16,
        ]
        for indices_dtype_proto in [
            TensorProto.INT32,
            TensorProto.INT64,
        ]
        for data_shape, indices_shape, axis in [
            ((10,), (2,), 0),  # 1D data, 1D indices, axis 0
            ((4, 5), (3,), 0),  # 2D data, 1D indices, outer axis
            ((4, 5), (2, 3), 1),  # 2D data, 2D indices, inner axis
            ((3, 4, 5), (2, 2), 1),  # 3D data, 2D indices, middle axis
            ((3, 4, 5), (2,), -1),  # 3D data, 1D indices, negative axis
            ((3, 4, 5), (1, 2), -2),  # 3D data, 2D indices, negative inner axis
        ]
    ],
)
# pylint: disable=too-many-arguments,too-many-positional-arguments
def test_onnx_gather_lower(
    ONNX_OPSET_VERSION,
    dtype_proto,
    indices_dtype_proto,
    data_shape,
    indices_shape,
    axis,
):
    """
    Test ONNX Gather operation lowering.
    """
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)
    np_idx_dtype = tensor_dtype_to_np_dtype(indices_dtype_proto)

    data_input = _generate_dtype_random(np_dtype, shape=data_shape, max_val=127)

    # Normalize axis position
    norm_axis = axis if axis >= 0 else axis + len(data_shape)
    axis_dim = data_shape[norm_axis]
    # Pick valid indices in range [-axis_dim, axis_dim - 1] to test negative wrapping
    indices_input = np.random.randint(-axis_dim, axis_dim, size=indices_shape).astype(
        np_idx_dtype
    )

    def create_onnx_model(data_input, indices_input, dtype_proto):

        # Output shape rule
        expected_out_shape = (
            data_shape[:norm_axis] + indices_shape + data_shape[norm_axis + 1 :]
        )

        input_data_info = make_tensor_value_info("data", dtype_proto, data_input.shape)
        input_indices_info = make_tensor_value_info(
            "indices", indices_dtype_proto, indices_input.shape
        )
        output_tensor_info = make_tensor_value_info(
            "output", dtype_proto, expected_out_shape
        )

        gather_node = make_node(
            "Gather",
            inputs=["data", "indices"],
            outputs=["output"],
            axis=axis,
        )

        graph = make_graph(
            nodes=[gather_node],
            name=f"gather_opset_{ONNX_OPSET_VERSION}",
            inputs=[input_data_info, input_indices_info],
            outputs=[output_tensor_info],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(data_input, indices_input, dtype_proto)
    check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"data": data_input, "indices": indices_input})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [data_input, indices_input], [res_array])

        if np.issubdtype(np_dtype, np.integer):
            np.testing.assert_array_equal(outputs[0], onnx_result)
        else:
            atol = 1e-2 if np_dtype == np.float16 else 1e-5
            rtol = 1e-2 if np_dtype == np.float16 else 1e-5
            np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, shape, slice_config",
    [
        (schema.since_version, dtype_proto, shape, slice_config)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Slice"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.FLOAT16,
        ]
        for shape, slice_config in [
            # YOLOv11 head slicing pattern
            ((1, 4, 8400), {"starts": [0], "ends": [4], "axes": [1], "steps": [1]}),
            # 1D basic slicing
            ((10,), {"starts": [2], "ends": [7], "axes": [0], "steps": [1]}),
            # 2D multi-axis slicing
            (
                (6, 8),
                {"starts": [1, 2], "ends": [5, 7], "axes": [0, 1], "steps": [1, 1]},
            ),
            # Strided slicing
            ((12,), {"starts": [1], "ends": [10], "axes": [0], "steps": [2]}),
            # Negative index slicing
            ((10,), {"starts": [-8], "ends": [-2], "axes": [0], "steps": [1]}),
            # Partial axis slicing on 3D tensor
            ((4, 5, 6), {"starts": [1], "ends": [4], "axes": [1], "steps": [1]}),
        ]
    ],
)
# pylint: disable=too-many-branches,too-many-statements
def test_onnx_slice_lower(ONNX_OPSET_VERSION, dtype_proto, shape, slice_config):
    """
    Test ONNX Slice operation lowering.
    """
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    data_input = _generate_dtype_random(np_dtype, shape=shape, max_val=127)

    def create_onnx_model(data_input, shape, dtype_proto):
        starts = slice_config["starts"]
        ends = slice_config["ends"]
        axes = slice_config.get("axes", None)
        steps = slice_config.get("steps", None)

        # Opset < 10 does not support non-unit steps (defaults to 1)
        if ONNX_OPSET_VERSION < 10:
            steps = None

        # Compute expected output shape
        out_shape = list(shape)
        eff_axes = axes if axes is not None else list(range(len(starts)))
        eff_steps = steps if steps is not None else [1] * len(starts)

        for s_val, e_val, ax, st_val in zip(starts, ends, eff_axes, eff_steps):
            if ax < 0:
                ax += len(shape)
            dim_len = shape[ax]
            if st_val > 0:
                s_norm = s_val + dim_len if s_val < 0 else s_val
                s_norm = max(0, min(s_norm, dim_len))
                e_norm = e_val + dim_len if e_val < 0 else e_val
                e_norm = max(0, min(e_norm, dim_len))
                out_shape[ax] = max(0, (e_norm - s_norm + st_val - 1) // st_val)
            else:
                s_norm = s_val + dim_len if s_val < 0 else s_val
                s_norm = max(-1, min(s_norm, dim_len - 1))
                e_norm = e_val + dim_len if e_val < 0 else e_val
                e_norm = max(-1, min(e_norm, dim_len - 1))
                abs_st = -st_val
                out_shape[ax] = max(0, (s_norm - e_norm + abs_st - 1) // abs_st)

        input_data_info = make_tensor_value_info("data", dtype_proto, data_input.shape)
        output_tensor_info = make_tensor_value_info(
            "output", dtype_proto, tuple(out_shape)
        )

        initializers = []
        inputs = [input_data_info]

        if ONNX_OPSET_VERSION < 10:
            node_kwargs = {"starts": starts, "ends": ends}
            if axes is not None:
                node_kwargs["axes"] = axes
            slice_node = make_node(
                "Slice", inputs=["data"], outputs=["output"], **node_kwargs
            )
        else:
            node_inputs = ["data", "starts", "ends"]

            starts_tensor = make_tensor(
                "starts",
                TensorProto.INT64,
                [len(starts)],
                np.array(starts, dtype=np.int64),
            )
            ends_tensor = make_tensor(
                "ends", TensorProto.INT64, [len(ends)], np.array(ends, dtype=np.int64)
            )
            initializers.extend([starts_tensor, ends_tensor])

            if axes is not None:
                node_inputs.append("axes")
                axes_tensor = make_tensor(
                    "axes",
                    TensorProto.INT64,
                    [len(axes)],
                    np.array(axes, dtype=np.int64),
                )
                initializers.append(axes_tensor)
            elif steps is not None:
                node_inputs.append("")

            if steps is not None:
                node_inputs.append("steps")
                steps_tensor = make_tensor(
                    "steps",
                    TensorProto.INT64,
                    [len(steps)],
                    np.array(steps, dtype=np.int64),
                )
                initializers.append(steps_tensor)

            slice_node = make_node("Slice", inputs=node_inputs, outputs=["output"])

        graph = make_graph(
            nodes=[slice_node],
            name=f"slice_opset_{ONNX_OPSET_VERSION}",
            inputs=inputs,
            outputs=[output_tensor_info],
            initializer=initializers,
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(data_input, shape, dtype_proto)
    check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"data": data_input})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx)
        mlir_module.operation.verify()

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [data_input], [res_array])

        if np.issubdtype(np_dtype, np.integer):
            np.testing.assert_array_equal(outputs[0], onnx_result)
        else:
            atol = 1e-2 if np_dtype == np.float16 else 1e-5
            rtol = 1e-2 if np_dtype == np.float16 else 1e-5
            np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, shape, bounds",
    [
        (schema.since_version, dtype_proto, shape, bounds)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Clip"]
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.INT8,
            TensorProto.UINT8,
        ]
        for shape in [
            (10,),
            (4, 5),
            (2, 3, 4),
            (2, 2, 3, 4),
        ]
        for bounds in [
            (-2.0, 3.0),
            (-1.0, None),
            (None, 4.0),
            (None, None),
        ]
    ],
)
# pylint: disable=too-many-branches,too-many-statements
def test_onnx_clip_lower(ONNX_OPSET_VERSION, dtype_proto, shape, bounds):
    """
    Test ONNX Clip operator lowering.
    """

    class Clip(OpRun):
        """
        Reference implementation for ONNX Clip operator.
        """

        # pylint: disable=redefined-builtin,arguments-differ
        def _run(self, x, min=None, max=None):
            res = x.copy()
            if min is not None:
                min_val = (
                    np.array(min, dtype=x.dtype)
                    if not isinstance(min, np.ndarray)
                    else min.astype(x.dtype)
                )
                res = np.maximum(res, min_val)
            if max is not None:
                max_val = (
                    np.array(max, dtype=x.dtype)
                    if not isinstance(max, np.ndarray)
                    else max.astype(x.dtype)
                )
                res = np.minimum(res, max_val)
            return (res,)

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    min_val, max_val = bounds
    data_input = _generate_dtype_random(np_dtype, shape=shape, max_val=127)

    def create_onnx_model(data_input, dtype_proto):

        input_data_info = make_tensor_value_info("input", dtype_proto, data_input.shape)
        output_tensor_info = make_tensor_value_info(
            "output", dtype_proto, data_input.shape
        )

        initializers = []
        inputs = [input_data_info]
        node_inputs = ["input"]
        node_kwargs = {}

        if ONNX_OPSET_VERSION < 11:
            # Opset 1 & 6 use attributes for min and max
            if min_val is not None:
                node_kwargs["min"] = float(min_val)
            if max_val is not None:
                node_kwargs["max"] = float(max_val)
            clip_node = make_node(
                "Clip", inputs=["input"], outputs=["output"], **node_kwargs
            )
        else:
            if min_val is not None:
                node_inputs.append("min")
                if np.issubdtype(np_dtype, np.integer):
                    iinfo = np.iinfo(np_dtype)
                    c_min = int(np.clip(min_val, iinfo.min, iinfo.max))
                    min_array = np.array(c_min, dtype=np_dtype)
                else:
                    min_array = np.array(min_val, dtype=np_dtype)
                min_tensor = make_tensor("min", dtype_proto, [], [min_array.item()])
                initializers.append(min_tensor)
            else:
                node_inputs.append("")

            if max_val is not None:
                if len(node_inputs) == 2 and node_inputs[1] == "":
                    pass  # Empty placeholder for min
                node_inputs.append("max")
                if np.issubdtype(np_dtype, np.integer):
                    iinfo = np.iinfo(np_dtype)
                    c_max = int(np.clip(max_val, iinfo.min, iinfo.max))
                    max_array = np.array(c_max, dtype=np_dtype)
                else:
                    max_array = np.array(max_val, dtype=np_dtype)
                max_tensor = make_tensor("max", dtype_proto, [], [max_array.item()])
                initializers.append(max_tensor)

            # Trim trailing empty optional operand inputs
            while node_inputs and node_inputs[-1] == "":
                node_inputs.pop()

            clip_node = make_node("Clip", inputs=node_inputs, outputs=["output"])

        graph = make_graph(
            nodes=[clip_node],
            name=f"clip_opset_V{ONNX_OPSET_VERSION}",
            inputs=inputs,
            outputs=[output_tensor_info],
            initializer=initializers,
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(data_input, dtype_proto)
    check_model(onnx_model)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "must be", "but got"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Clip V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        ref_inputs = {"input": data_input}

        try:
            ref = ReferenceEvaluator(onnx_model)
            onnx_result = ref.run(None, ref_inputs)[0]
        except (RuntimeError, NotImplementedError, ValueError):
            ref = ReferenceEvaluator(onnx_model, new_ops=[Clip])
            onnx_result = ref.run(None, ref_inputs)[0]

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [data_input], [res_array])

        if np.issubdtype(np_dtype, np.integer):
            np.testing.assert_array_equal(outputs[0], onnx_result)
        else:
            atol = 1e-2 if np_dtype == np.float16 else 1e-5
            rtol = 1e-2 if np_dtype == np.float16 else 1e-5
            np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, pads_val, mode",
    [
        (schema.since_version, dtype_proto, pads_val, mode)
        for schema in get_all_schemas_with_history()
        if "Pad" == schema.name
        for dtype_proto in [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.UINT8,
            TensorProto.UINT16,
            TensorProto.UINT32,
            TensorProto.UINT32,
        ]
        for mode in ["constant", "edge", "reflect", "wrap"]
        for pads_val in [
            [0, 0, 1, 1, 0, 0, 1, 1],  # Spatial 4D padding
            [0, 0, 0, 0, 0, 0, 0, 0],  # Zero padding
            [0, 0, 2, 2, 0, 0, 2, 2],  # Larger spatial padding
        ]
    ],
)
def test_onnx_pad_lower(ONNX_OPSET_VERSION, dtype_proto, pads_val, mode):
    """
    Test ONNX Pad operator lowering.
    """
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model(np_array, dtype_proto, pads_val):
        input_tensor = make_tensor_value_info("input", dtype_proto, np_array.shape)

        out_shape = list(np_array.shape)
        ndim = np_array.ndim
        for i in range(ndim):
            out_shape[i] += pads_val[i] + pads_val[i + ndim]

        output_tensor = make_tensor_value_info("output", dtype_proto, out_shape)

        if ONNX_OPSET_VERSION >= 11:
            pads_tensor = make_tensor(
                name="pads",
                data_type=TensorProto.INT64,
                dims=[len(pads_val)],
                vals=pads_val,
            )
            pad_inputs = ["input", "pads"]
            node_kwargs = {}
            initializers = [pads_tensor]
        else:
            pad_inputs = ["input"]
            initializers = []
            # Opset 1 uses 'paddings', Opsets 2-10 use 'pads'
            attr_name = "paddings" if ONNX_OPSET_VERSION == 1 else "pads"
            node_kwargs = {attr_name: pads_val, "value": 0.0}

        pad_node = make_node(
            "Pad",
            inputs=pad_inputs,
            outputs=["output"],
            mode=mode,
            **node_kwargs,
        )

        graph = make_graph(
            nodes=[pad_node],
            name="pad_graph",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=initializers,
        )

        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    np_array = _generate_dtype_random(np_dtype, shape=(1, 1, 2, 2), max_val=127)

    onnx_model = create_onnx_model(np_array, dtype_proto, pads_val)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {"input": np_array})[0]

    # pylint: disable=broad-exception-caught
    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except Exception as e:
            error_keywords = ["error", "must be", "but got"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Pad V{ONNX_OPSET_VERSION} does not support "
                    f"{TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [np_array], [output])

        np.testing.assert_allclose(outputs[0], onnx_result, atol=1e-3)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape",
    [
        (schema.name, schema.since_version, dtype_proto, shape)
        for schema in get_all_schemas_with_history()
        if schema.name == "BatchNormalization"
        for dtype_proto in [
            TensorProto.FLOAT16,
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
        ]
        for shape in [
            (2, 3, 4),
            (2, 3, 4, 5),
        ]
    ],
)
# pylint: disable=too-many-statements
def test_onnx_batchnormalization_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape
):
    """
    Test ONNX normalization operators lowering.
    """

    class BatchNormalization(OpRun):
        """
        ONNX BatchNormalization operator.
        Computes: Y = (X - mean) / sqrt(var + epsilon) * scale + B
        """

        # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        def _run(self, x, scale, b, mean, var, epsilon=1e-05, **kwargs):
            broadcast_shape = [1] * len(x.shape)
            broadcast_shape[1] = scale.shape[0]

            orig_dtype = x.dtype
            x_f32 = x.astype(np.float32)
            scale_f32 = scale.reshape(broadcast_shape).astype(np.float32)
            b_f32 = b.reshape(broadcast_shape).astype(np.float32)
            mean_f32 = mean.reshape(broadcast_shape).astype(np.float32)
            var_f32 = var.reshape(broadcast_shape).astype(np.float32)

            res = scale_f32 * (x_f32 - mean_f32) / np.sqrt(var_f32 + epsilon) + b_f32
            return (res.astype(orig_dtype),)

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)
    c_dim = shape[1]

    inp_x = _generate_dtype_random(np_dtype, shape=shape, max_val=10)
    inp_scale = _generate_dtype_random(np_dtype, shape=(c_dim,), max_val=5)
    inp_b = _generate_dtype_random(np_dtype, shape=(c_dim,), max_val=5)
    inp_mean = _generate_dtype_random(np_dtype, shape=(c_dim,), max_val=5)
    inp_var = np.abs(
        _generate_dtype_random(np_dtype, shape=(c_dim,), max_val=5)
    ) + np.array(0.1, dtype=np_dtype)

    x_info = make_tensor_value_info("X", dtype_proto, shape)
    scale_info = make_tensor_value_info("scale", dtype_proto, (c_dim,))
    b_info = make_tensor_value_info("B", dtype_proto, (c_dim,))
    mean_info = make_tensor_value_info("mean", dtype_proto, (c_dim,))
    var_info = make_tensor_value_info("var", dtype_proto, (c_dim,))
    y_info = make_tensor_value_info("Y", dtype_proto, shape)

    node_inputs = ["X", "scale", "B", "mean", "var"]
    node_outputs = ["Y"]

    epsilon = 1e-05
    node_kwargs = {"epsilon": epsilon}

    if ONNX_OPSET_VERSION == 1:
        node_kwargs["consumed_inputs"] = [0, 0, 0, 0, 0]

    if ONNX_OPSET_VERSION in [1, 6]:
        node_kwargs["is_test"] = 1
    elif ONNX_OPSET_VERSION >= 14:
        node_kwargs["training_mode"] = 0

    bn_node = make_node(
        ONNX_OP_NAME,
        node_inputs,
        node_outputs,
        **node_kwargs,
    )

    graph = make_graph(
        nodes=[bn_node],
        name="batch_norm_graph",
        inputs=[x_info, scale_info, b_info, mean_info, var_info],
        outputs=[y_info],
        initializer=[],
    )
    opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
    onnx_model = make_model(graph, opset_imports=opset_imports)

    check_model(onnx_model)

    feed_dict = {
        "X": inp_x,
        "scale": inp_scale,
        "B": inp_b,
        "mean": inp_mean,
        "var": inp_var,
    }

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"{ONNX_OP_NAME} V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        new_ops = [BatchNormalization] if ONNX_OPSET_VERSION in [1, 7, 9] else []
        ref = ReferenceEvaluator(onnx_model, new_ops=new_ops)
        onnx_result = ref.run(None, feed_dict)[0]

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(
            llvm_module,
            "main",
            [inp_x, inp_scale, inp_b, inp_mean, inp_var],
            [res_array],
        )

        rtol = 1e-2 if dtype_proto == TensorProto.FLOAT16 else 1e-5
        atol = 1e-2 if dtype_proto == TensorProto.FLOAT16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape, axis",
    [
        (schema.name, schema.since_version, dtype_proto, shape, axis)
        for schema in get_all_schemas_with_history()
        if schema.name == "LayerNormalization"
        for dtype_proto in [
            TensorProto.FLOAT16,
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
        ]
        for shape in [
            (2, 3, 4),
            (2, 3, 4, 5),
        ]
        for axis in [-1, -2, 1]
    ],
)
# pylint: disable=too-many-statements
def test_onnx_layernormalization_lower(
    ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape, axis
):
    """
    Test ONNX LayerNormalization operator lowering.
    """

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    norm_axis = axis if axis >= 0 else axis + len(shape)
    param_shape = shape[norm_axis:]

    inp_x = _generate_dtype_random(np_dtype, shape=shape, max_val=10)
    inp_scale = _generate_dtype_random(np_dtype, shape=param_shape, max_val=5)
    inp_b = _generate_dtype_random(np_dtype, shape=param_shape, max_val=5)

    x_info = make_tensor_value_info("X", dtype_proto, shape)
    scale_info = make_tensor_value_info("scale", dtype_proto, param_shape)
    b_info = make_tensor_value_info("B", dtype_proto, param_shape)
    y_info = make_tensor_value_info("Y", dtype_proto, shape)

    node_inputs = ["X", "scale", "B"]
    node_outputs = ["Y"]

    epsilon = 1e-05
    node_kwargs = {"axis": axis, "epsilon": epsilon}

    ln_node = make_node(
        ONNX_OP_NAME,
        node_inputs,
        node_outputs,
        **node_kwargs,
    )

    graph = make_graph(
        nodes=[ln_node],
        name="layer_norm_graph",
        inputs=[x_info, scale_info, b_info],
        outputs=[y_info],
        initializer=[],
    )
    opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
    onnx_model = make_model(graph, opset_imports=opset_imports)

    check_model(onnx_model)

    feed_dict = {
        "X": inp_x,
        "scale": inp_scale,
        "B": inp_b,
    }

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "op operand", "must be"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"{ONNX_OP_NAME} V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        ref = ReferenceEvaluator(onnx_model)
        onnx_result = ref.run(None, feed_dict)[0]

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(
            llvm_module,
            "main",
            [inp_x, inp_scale, inp_b],
            [res_array],
        )

        rtol = 1e-2 if dtype_proto == TensorProto.FLOAT16 else 1e-5
        atol = 1e-2 if dtype_proto == TensorProto.FLOAT16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)
