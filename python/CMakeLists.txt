add_mlir_python_extension(onnx2mlir_py_onnx_passes _onnx2mlir_onnx_passes
  SOURCES
    src/conversion/passes/onnx_pass_bindings.cpp
  LINK_LIBS
    onnx2mlir_onnx_linalg_pass
  OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/passes
)

add_mlir_python_extension(onnx2mlir_py_onnx_dialect _onnx2mlir_onnx_dialect
  SOURCES
    src/dialect/onnx_dialect_bindings.cpp
  LINK_LIBS
    onnx2mlir_onnx_dialect
  OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/dialect
)
