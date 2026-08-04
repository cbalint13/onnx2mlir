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
\file python/onnx2mlir/pipeline/builder.py
\brief MLIR conversion builder pipelines
"""

from mlir.dialects import func, llvm
from mlir.ir import Type, StringAttr
from mlir.passmanager import PassManager

from onnx2mlir.passes import register_onnx_to_linalg_pass


def get_module_signatures(module):
    """Get function signatures from a lowered module

    Parameters
    ----------
    module:
      Any level MLIR IR module with signatures.

    Returns:
    --------
    mlir_func_type:
      A dictionary with MLIR types of functions, None if not found.
    """
    mlir_func_types = {}
    for op in module.body.operations:
        if isinstance(op, llvm.LLVMFuncOp):
            if "func.signature" in op.attributes:
                func_sign_attr = op.attributes["func.signature"]
                mlir_func_types[op.sym_name] = Type.parse(func_sign_attr.value)

    return mlir_func_types


def set_module_signatures(module):
    """Set functions signature visible after lowering

    Parameters
    ----------
    module:
      High level MLIR IR module.

    Returns:
    --------
    mlir_func_type:
      The MLIR type from the signature, None if not found.
    """

    for op in module.body.operations:
        if isinstance(op, func.FuncOp):
            func_params_str = str(op.type)
            op.attributes["func.signature"] = StringAttr.get(func_params_str)

    return module


def llvm_lower_pipeline(module, signatures=False, affine_loops=True):
    """Convert high level IR to final LLVM module

    Parameters
    ----------
    module:
      High level MLIR IR module.

    signatures:
      Add function info signatures visible after lowering.

    affine_loops:
      Lower to affine loops instead of scf loops.

    Returns:
    --------
    module:
      The lowered LLVM module
    """

    if signatures:
        module = set_module_signatures(module)

    register_onnx_to_linalg_pass()

    pm = PassManager()

    # 1. High level conversion
    pm.add("lower-onnx-to-linalg")
    # pm.add("generate-runtime-verification")
    pm.add("linalg-generalize-named-ops")

    # 2. Optimization
    pm.add("linalg-fuse-elementwise-ops")
    pm.add("inline")

    # 3. Bufferization & sparsification
    pm.add("sparse-assembler")
    pm.add("sparsification-and-bufferization")
    pm.add("func.func(buffer-hoisting)")
    pm.add("func.func(buffer-loop-hoisting)")
    pm.add("func.func(promote-buffers-to-stack)")

    # 4. Loop lowering
    if affine_loops:
        pm.add("convert-linalg-to-affine-loops")
        pm.add("lower-affine")
    else:
        pm.add("convert-linalg-to-loops")

    # 5. Loop optimization
    pm.add("loop-invariant-code-motion")
    pm.add("loop-invariant-subset-hoisting")

    # 6. General optimization
    pm.add("canonicalize")
    pm.add("cse")

    # 7. Memory lowering
    pm.add("arith-expand")
    pm.add("expand-realloc")
    pm.add("expand-strided-metadata")

    # 8. Low level conversions
    pm.add("func.func(llvm-request-c-wrappers)")
    pm.add("buffer-deallocation-pipeline")
    pm.add("sparse-storage-specifier-to-llvm")
    pm.add("symbol-dce")

    # 9. Final conversions
    pm.add("convert-shape-to-std")
    pm.add("convert-math-to-funcs")
    pm.add("convert-scf-to-cf")
    pm.add("convert-cf-to-llvm")
    pm.add("convert-math-to-llvm")
    pm.add("convert-arith-to-llvm")
    pm.add("convert-complex-to-llvm")
    pm.add("convert-vector-to-llvm")
    pm.add("convert-func-to-llvm")
    pm.add("convert-bufferization-to-memref")
    pm.add("finalize-memref-to-llvm")
    pm.add("reconcile-unrealized-casts")

    pm.run(module.operation)

    return module
