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
\file tests/python/dialect/test_onnx_dialect_ir.py
\brief Tests for Onnx dialect IR
"""

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

from mlir.ir import Context, Location, MLIRError
from mlir.passmanager import PassManager

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
    "ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape",
    [
        (schema.name, schema.since_version, dtype_proto, shape)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Add", "Sub", "Mul", "Div", "Mod"]
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
        for shape in [
            (2, 3, 4),
        ]
    ],
)
def test_onnx_arith_binary_fold(ONNX_OP_NAME, ONNX_OPSET_VERSION, dtype_proto, shape):
    """
    Test ONNX arithmetic binary folding.
    """
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model(val0_np, val1_np, op_name, dtype_proto, opset_ver):
        c0_tensor = make_tensor(
            name="constant0_val",
            data_type=dtype_proto,
            dims=val0_np.shape,
            vals=val0_np.flatten().tolist(),
        )
        c1_tensor = make_tensor(
            name="constant1_val",
            data_type=dtype_proto,
            dims=val1_np.shape,
            vals=val1_np.flatten().tolist(),
        )
        const0_node = make_node(
            "Constant",
            inputs=[],
            outputs=["c0_out"],
            value=c0_tensor,
            name="Constant_Node_0",
        )
        const1_node = make_node(
            "Constant",
            inputs=[],
            outputs=["c1_out"],
            value=c1_tensor,
            name="Constant_Node_1",
        )
        arith_node = make_node(
            op_name,
            inputs=["c0_out", "c1_out"],
            outputs=["output_tensor"],
            name=f"Binary_{op_name}_Node",
        )
        output_info = make_tensor_value_info(
            "output_tensor", dtype_proto, val0_np.shape
        )
        graph = make_graph(
            nodes=[const0_node, const1_node, arith_node],
            name="constant_binary_fold_graph",
            inputs=[],
            outputs=[output_info],
        )
        opset_imports = [make_opsetid("", opset_ver)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    inp_array0 = _generate_dtype_random(np_dtype, shape=shape, max_val=10)
    inp_array1 = _generate_dtype_random(np_dtype, shape=shape, max_val=10)

    if ONNX_OP_NAME in ["Div", "Mod"]:
        inp_array1 += inp_array1 >= 0

    onnx_model = create_onnx_model(
        inp_array0, inp_array1, ONNX_OP_NAME, dtype_proto, ONNX_OPSET_VERSION
    )

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "must be", "but got"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Graph with V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        pm = PassManager()
        pm.add("canonicalize")
        pm.run(mlir_module.operation)

        flat_ops = [
            op.name
            for func_op in mlir_module.body.operations
            for op in func_op.entry_block.operations
            if op.name.startswith("onnx.")
        ]
        assert all(op_name.startswith("onnx.Constant") for op_name in flat_ops)

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [], [res_array])

        atol = 1e-2 if np_dtype == np.float16 else 1e-5
        rtol = 1e-2 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "ONNX_OPSET_VERSION, dtype_proto, shape",
    [
        (schema.since_version, dtype_proto, shape)
        for schema in get_all_schemas_with_history()
        if schema.name in ["Transpose"]
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
        for shape in [
            (1, 3, 8, 5),
        ]
    ],
)
def test_onnx_transpose_fold(ONNX_OPSET_VERSION, dtype_proto, shape):
    """
    Test ONNX Transpose folding.
    """
    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model(val_np, dtype_proto, opset_ver):
        cst_tensor = make_tensor(
            name="constant_val",
            data_type=dtype_proto,
            dims=val_np.shape,
            vals=val_np.flatten().tolist(),
        )
        const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["cst_out"],
            value=cst_tensor,
            name="Constant_Node",
        )
        trans_node = make_node(
            "Transpose",
            inputs=["cst_out"],
            outputs=["output_tensor"],
            name="Transpose_Node",
        )
        output_info = make_tensor_value_info(
            "output_tensor", dtype_proto, val_np.T.shape
        )
        graph = make_graph(
            nodes=[const_node, trans_node],
            name="constant_transpose_graph",
            inputs=[],
            outputs=[output_info],
        )
        opset_imports = [make_opsetid("", opset_ver)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    np_array = _generate_dtype_random(np_dtype, shape=shape, max_val=10)

    onnx_model = create_onnx_model(np_array, dtype_proto, ONNX_OPSET_VERSION)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "must be", "but got"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Graph with V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        pm = PassManager()
        pm.add("canonicalize")
        pm.run(mlir_module.operation)

        flat_ops = [
            op.name
            for func_op in mlir_module.body.operations
            for op in func_op.entry_block.operations
            if op.name.startswith("onnx.")
        ]
        assert all(op_name.startswith("onnx.Constant") for op_name in flat_ops)

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [], [res_array])

        atol = 1e-2 if np_dtype == np.float16 else 1e-5
        rtol = 1e-2 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


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
            [(None, None), (0, None), (1, 3), (-2, None), (0, -1)]
            if schema.since_version >= 15
            else [(None, None)]
        )
    ],
)
# pylint: disable=too-many-statements
def test_onnx_shape_fold(ONNX_OPSET_VERSION, dtype_proto, shape, start_end):
    """
    Test ONNX Shape folding.
    """
    start_attr, end_attr = start_end

    np_dtype = tensor_dtype_to_np_dtype(dtype_proto)

    def create_onnx_model(val_np, dtype_proto, opset_ver):
        cst_tensor = make_tensor(
            name="constant_val",
            data_type=dtype_proto,
            dims=val_np.shape,
            vals=val_np.flatten().tolist(),
        )
        const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["cst_out"],
            value=cst_tensor,
            name="Constant_Node",
        )

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

        shape_node = make_node(
            "Shape",
            inputs=["cst_out"],
            outputs=["output_tensor"],
            **node_kwargs,
        )
        output_info = make_tensor_value_info(
            "output_tensor", TensorProto.INT64, out_shape
        )
        graph = make_graph(
            nodes=[const_node, shape_node],
            name="constant_shape_graph",
            inputs=[],
            outputs=[output_info],
        )
        opset_imports = [make_opsetid("", opset_ver)]
        model = make_model(graph, opset_imports=opset_imports)
        check_model(model)
        return model

    np_array = _generate_dtype_random(np_dtype, shape=shape, max_val=10)

    onnx_model = create_onnx_model(np_array, dtype_proto, ONNX_OPSET_VERSION)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "must be", "but got"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Graph with V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                )
            else:
                raise

        pm = PassManager()
        pm.add("canonicalize")
        pm.run(mlir_module.operation)

        flat_ops = [
            op.name
            for func_op in mlir_module.body.operations
            for op in func_op.entry_block.operations
            if op.name.startswith("onnx.")
        ]
        assert all(op_name.startswith("onnx.Constant") for op_name in flat_ops)

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        res_array = np.zeros_like(onnx_result)
        outputs = runner(llvm_module, "main", [], [res_array])

        atol = 1e-2 if np_dtype == np.float16 else 1e-5
        rtol = 1e-2 if np_dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], onnx_result, rtol=rtol, atol=atol)


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
def test_onnx_cast_fold(ONNX_OPSET_VERSION, src_dtype_proto, tgt_dtype_proto, shape):
    """
    Test ONNX Cast folding.
    """
    src_np_dtype = tensor_dtype_to_np_dtype(src_dtype_proto)
    tgt_np_dtype = tensor_dtype_to_np_dtype(tgt_dtype_proto)

    val_np = _generate_dtype_random(src_np_dtype, shape, max_val=127)
    if np.issubdtype(src_np_dtype, np.floating) and np.issubdtype(
        tgt_np_dtype, np.integer
    ):
        val_np = np.trunc(np.abs(val_np))

    def create_onnx_model(val_np, src_dtype_proto, tgt_dtype_proto):
        cst_tensor = make_tensor(
            name="constant_val",
            data_type=src_dtype_proto,
            dims=val_np.shape,
            vals=val_np.flatten().tolist(),
        )
        const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["cst_out"],
            value=cst_tensor,
            name="Constant_Node",
        )

        node_kwargs = {}
        if ONNX_OPSET_VERSION == 1:
            node_kwargs["to"] = str(TensorProto.DataType.Name(tgt_dtype_proto))
        else:
            node_kwargs["to"] = int(tgt_dtype_proto)

        cast_node = make_node(
            "Cast",
            inputs=["cst_out"],
            outputs=["output"],
            **node_kwargs,
        )
        output_tensor = make_tensor_value_info("output", tgt_dtype_proto, val_np.shape)
        graph = make_graph(
            nodes=[const_node, cast_node],
            name="cast_graph",
            inputs=[],
            outputs=[output_tensor],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(val_np, src_dtype_proto, tgt_dtype_proto)
    check_model(onnx_model)

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "must be", "but got"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Graph V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(tgt_dtype_proto)}"
                    f" -> {TensorProto.DataType.Name(tgt_dtype_proto)}"
                )
            else:
                raise

        pm = PassManager()
        pm.add("canonicalize")
        pm.run(mlir_module.operation)

        flat_ops = [
            op.name
            for func_op in mlir_module.body.operations
            for op in func_op.entry_block.operations
            if op.name.startswith("onnx.")
        ]
        assert all(op_name.startswith("onnx.Constant") for op_name in flat_ops)

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        if ONNX_OPSET_VERSION == 1:
            onnx_result = val_np.astype(tgt_np_dtype)
        else:
            ref = ReferenceEvaluator(onnx_model)
            onnx_result = ref.run(None, {})[0]

        output_buffer = np.zeros_like(onnx_result, dtype=tgt_np_dtype)
        outputs = runner(llvm_module, "main", [], [output_buffer])

        if np.issubdtype(tgt_np_dtype, np.integer):
            np.testing.assert_array_equal(outputs[0], onnx_result)
        else:
            np.testing.assert_allclose(outputs[0], onnx_result, rtol=1e-3, atol=1e-3)


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
def test_onnx_gather_fold(
    ONNX_OPSET_VERSION,
    dtype_proto,
    indices_dtype_proto,
    data_shape,
    indices_shape,
    axis,
):
    """
    Test ONNX Gather folding.
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
        cst_data_tensor = make_tensor(
            name="constant_data",
            data_type=dtype_proto,
            dims=data_input.shape,
            vals=data_input.flatten().tolist(),
        )
        const_data_node = make_node(
            "Constant",
            inputs=[],
            outputs=["const_dat_out"],
            value=cst_data_tensor,
            name="Constant_Data_Node",
        )
        cst_indices_tensor = make_tensor(
            name="constant_indices",
            data_type=indices_dtype_proto,
            dims=indices_input.shape,
            vals=indices_input.flatten().tolist(),
        )
        const_indices_node = make_node(
            "Constant",
            inputs=[],
            outputs=["const_ind_out"],
            value=cst_indices_tensor,
            name="Constant_Indices_Node",
        )

        # Output shape rule
        expected_out_shape = (
            data_shape[:norm_axis] + indices_shape + data_shape[norm_axis + 1 :]
        )

        output_tensor_info = make_tensor_value_info(
            "output", dtype_proto, expected_out_shape
        )
        gather_node = make_node(
            "Gather",
            inputs=["const_dat_out", "const_ind_out"],
            outputs=["output"],
            axis=axis,
        )
        graph = make_graph(
            nodes=[const_data_node, const_indices_node, gather_node],
            name=f"gather_opset_{ONNX_OPSET_VERSION}",
            inputs=[],
            outputs=[output_tensor_info],
            initializer=[],
        )
        opset_imports = [make_opsetid("", ONNX_OPSET_VERSION)]
        return make_model(graph, opset_imports=opset_imports)

    onnx_model = create_onnx_model(data_input, indices_input, dtype_proto)
    check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    onnx_result = ref.run(None, {})[0]

    with Context() as ctx, Location.unknown():
        mlir_module = import_from_onnx(onnx_model, ctx, verify=False)
        try:
            mlir_module.operation.verify()
        except MLIRError as e:
            error_keywords = ["error", "must be", "but got"]
            if all(kw in str(e) for kw in error_keywords):
                pytest.skip(
                    f"Graph V{ONNX_OPSET_VERSION} does not support"
                    f" {TensorProto.DataType.Name(dtype_proto)}"
                    f" and {TensorProto.DataType.Name(indices_dtype_proto)}"
                )
            else:
                raise

        pm = PassManager()
        pm.add("canonicalize")
        pm.run(mlir_module.operation)

        flat_ops = [
            op.name
            for func_op in mlir_module.body.operations
            for op in func_op.entry_block.operations
            if op.name.startswith("onnx.")
        ]
        assert all(op_name.startswith("onnx.Constant") for op_name in flat_ops)

        llvm_module = llvm_lower_pipeline(mlir_module)
        llvm_module.operation.verify()

        output_buffer = np.zeros_like(onnx_result, dtype=np_dtype)
        outputs = runner(llvm_module, "main", [], [output_buffer])

        if np.issubdtype(np_dtype, np.integer):
            np.testing.assert_array_equal(outputs[0], onnx_result)
        else:
            np.testing.assert_allclose(outputs[0], onnx_result, rtol=1e-3, atol=1e-3)
