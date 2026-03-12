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
unset(TABLEGEN_OUTPUT)

# Onnx tablegen interfaces
set(LLVM_TARGET_DEFINITIONS OnnxInterfaces.td)
mlir_tablegen(OnnxInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OnnxInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIROnnxInterfacesIncGen)
unset(TABLEGEN_OUTPUT)

# Onnx tablegen python bindings
set(LLVM_TARGET_DEFINITIONS Onnx.td)
mlir_tablegen(Onnx.py.inc -gen-python-op-bindings --bind-dialect=onnx)
add_public_tablegen_target(MLIROnnxPythonIncGen)
unset(TABLEGEN_OUTPUT)

# Onnx tablegen docs
add_mlir_doc(Onnx OnnxDialect onnx/ -gen-dialect-doc)
add_mlir_doc(Onnx OnnxOps onnx/ -gen-op-doc)
add_mlir_doc(Onnx OnnxOpsInterfaces onnx/ -gen-op-interface-docs)
add_mlir_doc(Onnx OnnxTypes onnx/ -gen-typedef-doc)
add_mlir_doc(Onnx OnnxTypesInterfaces onnx/ -gen-type-interface-docs)
add_mlir_doc(Onnx OnnxAttrs onnx/ -gen-attrdef-doc)
add_mlir_doc(Onnx OnnxAttrsInterfaces onnx/ -gen-attr-interface-docs)

# Onnx dialect ops gen
set(ONNX_MLIR_OPS_GEN "${CMAKE_SOURCE_DIR}/tools/gen_onnx_dia.py")
set(ONNX_MLIR_OPS_TD "${CMAKE_CURRENT_SOURCE_DIR}/OnnxOps.td")
set(ONNX_MLIR_OPS_INC "${CMAKE_CURRENT_SOURCE_DIR}/OnnxOps.td.inc")
set(ONNX_MLIR_OPS_CPP "${CMAKE_CURRENT_SOURCE_DIR}/ir/onnx_ops.cpp")
add_custom_command(
  OUTPUT ${ONNX_MLIR_OPS_INC}
  COMMAND ${Python3_EXECUTABLE} ${ONNX_MLIR_OPS_GEN}
  ARGS ${ONNX_MLIR_OPS_INC}
  COMMAND ${CMAKE_COMMAND}
  ARGS -E touch ${ONNX_MLIR_OPS_TD}
  COMMAND ${CMAKE_COMMAND}
  ARGS -E touch ${ONNX_MLIR_OPS_CPP}
  DEPENDS ${ONNX_MLIR_OPS_GEN}
  COMMENT "Generating ONNX MLIR Ops using gen_onnx_dia.py"
)
add_custom_target(generate_onnx_ops
  DEPENDS ${ONNX_MLIR_OPS_INC}
)
add_dependencies(MLIROnnxIncGen generate_onnx_ops)

# Onnx dialect library
add_mlir_dialect_library(onnx2mlir_onnx_dialect
    ir/onnx_attrs.cpp
    ir/onnx_dialect.cpp
    ir/onnx_interfaces.cpp
    ir/onnx_ops.cpp
    ir/onnx_types.cpp
    ir/onnx_printers.cpp
    support/onnx_shapes.cpp
  DEPENDS
    MLIROnnxIncGen
    MLIROnnxInterfacesIncGen
  LINK_LIBS PUBLIC
    MLIR
)

if(BUILD_PYTHON_BINDINGS)
  # Onnx dialect python bindings
  set(PY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/python/onnx2mlir/dialect")
  add_mlir_python_extension(onnx2mlir_py_onnx_dialect _onnx2mlirDialectsOnnx
    nanobind-static
    _PRIVATE_SUPPORT_LIB
    SOURCES
      python/onnx_dialect_bindings.cpp
    OUTPUT_DIRECTORY
      ${PY_OUTPUT_PATH}
    LINK_LIBS
      onnx2mlir_onnx_dialect
      MLIRPythonCAPI
  )
  set_target_properties(onnx2mlir_py_onnx_dialect PROPERTIES PREFIX "")
  add_dependencies(onnx2mlir_py_onnx_dialect MLIROnnxPythonIncGen)
  set(PY_TBLGEN_FILE "${CMAKE_CURRENT_BINARY_DIR}/Onnx.py.inc")
  add_custom_command(TARGET onnx2mlir_py_onnx_dialect
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PY_TBLGEN_FILE} onnx.py
    COMMAND
      echo "from . import onnx" > __init__.py
    COMMAND
      echo "from ._onnx2mlirDialectsOnnx import *" >> __init__.py
    COMMAND
      echo "__all__ = ['onnx', 'register_onnx_dialect']" >> __init__.py
    COMMAND
      sed -i 's|from ._ods_common|from mlir.dialects._ods_common|g' onnx.py
    WORKING_DIRECTORY ${PY_OUTPUT_PATH}
  )
endif()
