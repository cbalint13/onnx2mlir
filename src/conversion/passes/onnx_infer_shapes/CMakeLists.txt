add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

add_mlir_conversion_library(onnx2mlir_onnx_infershapes_pass
    onnx_infer_shapes.cpp
  DEPENDS
    onnx2mlir_onnx_dialect
    MLIRConversionPassesIncGen
  LINK_LIBS PUBLIC
    onnx2mlir_onnx_dialect
    MLIR
)

