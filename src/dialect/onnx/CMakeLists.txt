# Onnx tablegen ops, types, attrs
set(LLVM_TARGET_DEFINITIONS Onnx.td)
mlir_tablegen(Onnx.h.inc -gen-op-decls -attrdefs-dialect=onnx)
mlir_tablegen(Onnx.cpp.inc -gen-op-defs -attrdefs-dialect=onnx)
mlir_tablegen(OnnxDialect.h.inc -gen-dialect-decls -dialect=onnx)
mlir_tablegen(OnnxDialect.cpp.inc -gen-dialect-defs -dialect=onnx)
mlir_tablegen(OnnxTypes.h.inc -gen-typedef-decls -typedefs-dialect=onnx)
mlir_tablegen(OnnxTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=onnx)
mlir_tablegen(OnnxAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=onnx)
mlir_tablegen(OnnxAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=onnx)
add_public_tablegen_target(MLIROnnxIncGen)

# Onnx tablegen interfaces
set(LLVM_TARGET_DEFINITIONS OnnxInterfaces.td)
mlir_tablegen(OnnxInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OnnxInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIROnnxInterfacesIncGen)

# Onnx tablegen docs
add_mlir_doc(Onnx OnnxDialect onnx/ -gen-dialect-doc)
add_mlir_doc(Onnx OnnxOps onnx/ -gen-op-doc)
add_mlir_doc(Onnx OnnxOpsInterfaces onnx/ -gen-op-interface-docs)
add_mlir_doc(Onnx OnnxTypes onnx/ -gen-typedef-doc)
add_mlir_doc(Onnx OnnxTypesInterfaces onnx/ -gen-type-interface-docs)
add_mlir_doc(Onnx OnnxAttrs onnx/ -gen-attrdef-doc)
add_mlir_doc(Onnx OnnxAttrsInterfaces onnx/ -gen-attr-interface-docs)

# Onnx dialect python gen
if(BUILD_PYTHON_BINDINGS)
  set(LLVM_TARGET_DEFINITIONS Onnx.td)
  mlir_tablegen(Onnx.py.inc -gen-python-op-bindings --bind-dialect=onnx)
  add_public_tablegen_target(MLIROnnxPythonIncGen)
  set(PY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/python/onnx2mlir/dialect")
  set(PY_TBLGEN_FILE "${CMAKE_CURRENT_BINARY_DIR}/Onnx.py.inc")
  add_custom_command(
    OUTPUT "${OUTPUT_PY_PATH}/__init__.py"
    DEPENDS MLIROnnxPythonIncGen
    COMMAND ${CMAKE_COMMAND} -E copy ${PY_TBLGEN_FILE} onnx.py
    COMMAND
      echo "from . import onnx" > __init__.py
    COMMAND
      echo "from ._onnx2mlirDialectsOnnx import register_onnx_dialect" >> __init__.py
    COMMAND
      echo "__all__ = ['onnx', 'register_onnx_dialect']" >> __init__.py
    COMMAND
      sed -i 's|from ._ods_common|from mlir.dialects._ods_common|g' onnx.py
    WORKING_DIRECTORY ${PY_OUTPUT_PATH}
    COMMENT "Update Onnx dialect python bindings"
  )
  add_custom_target(generate_onnx_py_bindings
    DEPENDS "${OUTPUT_PY_PATH}/__init__.py"
    COMMENT "Finish Onnx dialect python bindings"
  )
endif()

# Onnx dialect ops gen
set(ONNX_MLIR_OPS_GEN "${CMAKE_SOURCE_DIR}/tools/gen_onnx_dia.py")
set(ONNX_MLIR_OPS_TD "${CMAKE_CURRENT_SOURCE_DIR}/OnnxOps.td")
set(ONNX_MLIR_OPS_INC "${CMAKE_CURRENT_SOURCE_DIR}/OnnxOps.td.inc")
add_custom_command(
  OUTPUT ${ONNX_MLIR_OPS_INC}
  COMMAND ${Python3_EXECUTABLE} ${ONNX_MLIR_OPS_GEN}
  ARGS ${ONNX_MLIR_OPS_INC}
  COMMAND ${CMAKE_COMMAND}
  ARGS -E touch ${ONNX_MLIR_OPS_TD}
  DEPENDS ${ONNX_MLIR_OPS_GEN}
  COMMENT "Generating ONNX MLIR Ops using gen_onnx_dia.py"
)
add_custom_target(generate_onnx_ops
  DEPENDS ${ONNX_MLIR_OPS_INC}
  COMMENT "MLIR ONNX Ops generator"
)
add_dependencies(MLIROnnxIncGen generate_onnx_ops)

# Onnx dialect library
add_mlir_dialect_library(onnx2mlir_onnx_dialect
    ir/onnx_attrs.cpp
    ir/onnx_dialect.cpp
    ir/onnx_interfaces.cpp
    ir/onnx_ops.cpp
    ir/onnx_types.cpp
    support/onnx_shapes.cpp
  DEPENDS
    MLIROnnxIncGen
    MLIROnnxInterfacesIncGen
  LINK_LIBS PUBLIC
    MLIR
)
if(BUILD_PYTHON_BINDINGS)
  add_dependencies(onnx2mlir_onnx_dialect generate_onnx_py_bindings)
endif()
