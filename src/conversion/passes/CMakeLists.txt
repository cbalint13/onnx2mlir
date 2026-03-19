add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

add_mlir_conversion_library(onnx2mlir_onnx_linalg_pass
    conversion.cpp
    onnx_to_linalg.cpp
    onnx_to_linalg/arith_binary.cpp
    onnx_to_linalg/arith_unary.cpp
    onnx_to_linalg/cast.cpp
    onnx_to_linalg/compare_binary.cpp
    onnx_to_linalg/constant.cpp
    onnx_to_linalg/conv.cpp
    onnx_to_linalg/gemm.cpp
    onnx_to_linalg/maxpool.cpp
    onnx_to_linalg/squeeze.cpp
    onnx_to_linalg/transpose.cpp
    onnx_to_linalg/unsqueeze.cpp
    onnx_to_linalg/where.cpp
  DEPENDS
    onnx2mlir_onnx_dialect
    MLIRConversionPassesIncGen
  LINK_LIBS PUBLIC
    onnx2mlir_onnx_dialect
    MLIR
)
