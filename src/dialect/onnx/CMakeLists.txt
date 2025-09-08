# Onnx dialect tablegen
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
add_dependencies(mlir-headers MLIROnnxIncGen)

set(LLVM_TARGET_DEFINITIONS OnnxInterface.td)
mlir_tablegen(OnnxInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(OnnxInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIROnnxInterfaceIncGen)
add_dependencies(mlir-headers MLIROnnxInterfaceIncGen)

if(BUILD_PYTHON_BINDINGS)
  set(LLVM_TARGET_DEFINITIONS Onnx.td)
  mlir_tablegen(Onnx.py.inc -gen-python-op-bindings --bind-dialect=onnx)
  add_public_tablegen_target(MLIROnnxPythonIncGen)

  # change to global python mlir path
  set(OUTPUT_PY_PATH "${CMAKE_BINARY_DIR}/python/onnx2mlir/dialect")
  set(TBLGEN_PY_FILE "${CMAKE_CURRENT_BINARY_DIR}/Onnx.py.inc")
  add_custom_command(
    OUTPUT "${OUTPUT_PY_PATH}/onnx.py"
    DEPENDS MLIROnnxPythonIncGen
    COMMAND ${CMAKE_COMMAND} -E copy ${TBLGEN_PY_FILE} "${OUTPUT_PY_PATH}/onnx.py"
    COMMAND ${CMAKE_COMMAND} -E chdir "${OUTPUT_PY_PATH}"
      echo "from . import onnx" > __init__.py
    COMMAND ${CMAKE_COMMAND} -E chdir "${OUTPUT_PY_PATH}"
      echo "from ._onnx2mlirDialectsOnnx import register_onnx_dialect" >> __init__.py
    COMMAND ${CMAKE_COMMAND} -E chdir "${OUTPUT_PY_PATH}"
      echo "__all__ = ['onnx', 'register_onnx_dialect']" >> __init__.py
    COMMAND ${CMAKE_COMMAND} -E chdir "${OUTPUT_PY_PATH}"
      sed -i 's|from ._ods_common|from mlir.dialects._ods_common|g' onnx.py
    COMMENT "Fixing Onnx.py.inc bindings"
  )
  add_custom_target(generate_onnx_py_bindings ALL
    DEPENDS "${OUTPUT_PY_PATH}/onnx.py"
    COMMENT "Finalize onnx.py bindings"
  )

  add_dependencies(mlir-headers MLIROnnxPythonIncGen generate_onnx_py_bindings)
endif()

add_mlir_doc(OnnxDialect OnnxDialect onnx/ -gen-dialect-doc)
add_mlir_doc(OnnxOps OnnxOps onnx/ -gen-op-doc)
add_mlir_doc(OnnxOpsInter OnnxOpsInter onnx/ -gen-op-interface-doc)
add_mlir_doc(OnnxTypes OnnxTypes onnx/ -gen-typedef-doc)
add_mlir_doc(OnnxTypesInter OnnxTypesInter onnx/ -gen-type-interface-doc)
add_mlir_doc(OnnxAttrs OnnxAttrs onnx/ -gen-attrdef-doc)
add_mlir_doc(OnnxAttrsInter OnnxAttrsInter onnx/ -gen-attr-interface-doc)

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

add_custom_target(generate_onnx_ops ALL
  DEPENDS ${ONNX_MLIR_OPS_INC}
  COMMENT "MLIR ONNX Ops generator"
)

add_dependencies(MLIROnnxIncGen generate_onnx_ops)

# Onnx dialect library
add_mlir_dialect_library(onnx2mlir_onnx_dialect
  OBJECT
    ir/onnx_attrs.cpp
    ir/onnx_dialect.cpp
    ir/onnx_interface.cpp
    ir/onnx_ops.cpp
    ir/onnx_types.cpp
    support/onnx_shapes.cpp
  DEPENDS
    MLIROnnxIncGen
  LINK_LIBS PUBLIC
    MLIR
)
