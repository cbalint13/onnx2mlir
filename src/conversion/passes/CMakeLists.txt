add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

add_library(onnx_to_linalg_ops
  OBJECT
    onnx_to_linalg/constant.cpp
)

add_mlir_conversion_library(onnx2mlir_onnx_linalg_pass
  OBJECT
    onnx_to_linalg.cpp
  DEPENDS
    onnx2mlir_onnx_dialect
    MLIRConversionPassIncGen
  LINK_LIBS
    onnx_to_linalg_ops
    PUBLIC MLIR
)
