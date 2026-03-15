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

"""
\file python/onnx2mlir/pipeline/runner.py
\brief Executor pipeline functions
"""

import ctypes
import warnings
from mlir.dialects import llvm
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy


def runner(module, func_entry, inputs, outputs):
    """Execute a lowered LLVM module

    Parameters
    ----------
    module:
      High level MLIR IR module.

    func_entry:
      The name of entry function.

    inputs:
      A list with numpy arrays as inputs

    outputs:
      A list with numpy arrays as outputs

    Returns:
    --------
    outputs:
      The resulting list with output numpy arrays.
    """

    assert isinstance(inputs, (list, tuple))
    assert isinstance(outputs, (list, tuple))

    if not any(isinstance(op, llvm.LLVMFuncOp) for op in module.body.operations):
        raise TypeError("Module has no LLVM functions")

    try:
        # pylint: disable=import-outside-toplevel
        from onnx2mlir.version import LLVM_LIBRARY_PATH

        shared_libs = [
            f"{LLVM_LIBRARY_PATH}/libmlir_runner_utils.so",
            f"{LLVM_LIBRARY_PATH}/libmlir_c_runner_utils.so",
        ]
        engine = ExecutionEngine(module, opt_level=3, shared_libs=shared_libs)
    except:  # pylint: disable=bare-except
        warnings.warn("MLIR executor running without utility libraries", RuntimeWarning)
        engine = ExecutionEngine(module, opt_level=3)

    engine.initialize()

    inp_descs = [get_ranked_memref_descriptor(inp) for inp in inputs]
    out_descs = [get_ranked_memref_descriptor(out) for out in outputs]
    inp_cargs = [ctypes.pointer(ctypes.pointer(inp)) for inp in inp_descs]
    out_cargs = [ctypes.pointer(ctypes.pointer(out)) for out in out_descs]

    all_cargs = out_cargs + inp_cargs
    engine.invoke(func_entry, *all_cargs)

    outputs = [ranked_memref_to_numpy(carg[idx]) for idx, carg in enumerate(out_cargs)]

    return outputs
