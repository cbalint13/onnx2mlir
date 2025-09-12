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

##
## \file python/onnx2mlir/support/numpy_to_mlir.py
## \brief NumPy to MLIR helpers
##

import numpy as np
from mlir.ir import (
    DenseElementsAttr,
    RankedTensorType,
    IntegerType,
    FloatType,
    F16Type,
    F32Type,
    F64Type,
)


def mlir_type_from_numpy(np_dtype):
    """Get MLIR element type from NumPy element type

    Parameters
    ----------
    type:
      The NumPy data type

    Returns:
    --------
    elem_dtype:
      The equivalent MLIR data type
    """
    if np.issubdtype(np_dtype, np.integer):
        elem_bits = np_dtype.itemsize * 8
        if np.issubdtype(np_dtype, np.unsignedinteger):
            elem_type = IntegerType.get_unsigned(elem_bits)
        else:
            elem_type = IntegerType.get_signless(elem_bits)
    elif "float16" == np_dtype:
        elem_type = F16Type()
    elif "float32" == np_dtype:
        elem_type = F32Type.get()
    elif "float64" == np_dtype:
        elem_type = F64Type.get()
    else:
        raise TypeError(f"Unsupported numpy datatype: {np_dtype}")

    return elem_type


def mlir_rankedtensor_type_from_numpy(np_array):
    """Get MLIR ranked tensor type from NumPy array

    Parameters
    ----------
    np_array:
      The NumPy array

    Returns:
    --------
    tensor_type:
      The equivalent MLIR tensor type
    """
    elem_type = mlir_type_from_numpy(np_array.dtype)
    tensor_type = RankedTensorType.get(shape=np_array.shape, element_type=elem_type)

    return tensor_type


def mlir_dense_from_numpy(np_array, tensor_type=None):
    """Get MLIR dense elements attribute from NumPy array

    Parameters
    ----------
    np_array:
      The NumPy array

    Returns:
    --------
    tensor_attr:
      The equivalent MLIR dense tensor attribute
    """
    tensor_type = mlir_rankedtensor_type_from_numpy(np_array)
    tensor_attr = DenseElementsAttr.get(
        np_array,
        type=tensor_type,
    )

    return tensor_attr
