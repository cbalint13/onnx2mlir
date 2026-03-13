# Onnx tablegen passes gen
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name conversion)
add_public_tablegen_target(MLIRConversionPassesIncGen)
unset(TABLEGEN_OUTPUT)

# Onnx docs
add_mlir_doc(Passes ConversionPasses conversion/ -gen-pass-doc)

add_subdirectory(passes)

if(BUILD_PYTHON_BINDINGS)
  # Onnx passes python bindings
  set(PY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/python/onnx2mlir/passes")
  add_mlir_python_extension(onnx2mlir_py_onnx_passes _onnx2mlirOnnxPasses
    nanobind-static
    _PRIVATE_SUPPORT_LIB
    SOURCES
      python/onnx_passes_bindings.cpp
    OUTPUT_DIRECTORY
      ${PY_OUTPUT_PATH}
    LINK_LIBS
      onnx2mlir_onnx_dialect
      onnx2mlir_onnx_linalg_pass
      MLIRPythonCAPI
  )
  set_target_properties(onnx2mlir_py_onnx_passes PROPERTIES PREFIX "")
  add_custom_command(TARGET onnx2mlir_py_onnx_passes
    POST_BUILD
    COMMAND
      echo "from ._onnx2mlirOnnxPasses import *" > __init__.py
    COMMAND
      echo "__all__ = ['register_onnx_to_linag_pass']" >> __init__.py
    WORKING_DIRECTORY ${PY_OUTPUT_PATH}
  )
endif()
