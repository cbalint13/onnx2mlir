add_mlir_python_extension(onnx2mlir_py_onnx_dialect _onnx2mlirDialectsOnnx
  SOURCES
    src/dialect/onnx_dialect_bindings.cpp
  OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/onnx2mlir/dialect
  LINK_LIBS
    onnx2mlir_onnx_dialect
    MLIRPythonCAPI
)

add_mlir_python_extension(onnx2mlir_py_onnx_passes _onnx2mlirOnnxPasses
  SOURCES
    src/conversion/passes/onnx_passes_bindings.cpp
  OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/onnx2mlir/passes
  LINK_LIBS
    onnx2mlir_onnx_dialect
    onnx2mlir_onnx_linalg_pass
    MLIRPythonCAPI
)

