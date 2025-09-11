add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

add_library(onnx2mlir_converters
  onnx.cpp
)
target_link_libraries(onnx2mlir_converters
  PRIVATE
    onnx2mlir_onnx_linalg_pass
  PUBLIC
    MLIR
    LLVM
)
