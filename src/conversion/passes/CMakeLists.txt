add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

add_mlir_conversion_library(onnx2mlir_onnx_linalg_pass
    conversion.cpp
    onnx_to_linalg.cpp
    onnx_to_linalg/cast.cpp
    onnx_to_linalg/binaries.cpp
    onnx_to_linalg/clip.cpp
    onnx_to_linalg/compare.cpp
    onnx_to_linalg/concat.cpp
    onnx_to_linalg/constant.cpp
    onnx_to_linalg/conv.cpp
    onnx_to_linalg/flatten.cpp
    onnx_to_linalg/gather.cpp
    onnx_to_linalg/gemm.cpp
    onnx_to_linalg/globalpool.cpp
    onnx_to_linalg/matmul.cpp
    onnx_to_linalg/maxpool.cpp
    onnx_to_linalg/pad.cpp
    onnx_to_linalg/reshape.cpp
    onnx_to_linalg/resize.cpp
    onnx_to_linalg/shape.cpp
    onnx_to_linalg/slice.cpp
    onnx_to_linalg/softmax.cpp
    onnx_to_linalg/split.cpp
    onnx_to_linalg/squeeze.cpp
    onnx_to_linalg/transpose.cpp
    onnx_to_linalg/unaries.cpp
    onnx_to_linalg/unsqueeze.cpp
    onnx_to_linalg/where.cpp
  DEPENDS
    onnx2mlir_onnx_dialect
    MLIRConversionPassesIncGen
  LINK_LIBS PUBLIC
    onnx2mlir_onnx_dialect
    MLIR
)
