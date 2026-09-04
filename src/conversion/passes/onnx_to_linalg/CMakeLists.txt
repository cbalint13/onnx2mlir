add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

add_mlir_conversion_library(onnx2mlir_onnx_linalg_pass
    onnx_to_linalg.cpp
    operators/cast.cpp
    operators/binaries.cpp
    operators/clip.cpp
    operators/compare.cpp
    operators/concat.cpp
    operators/constant.cpp
    operators/conv.cpp
    operators/flatten.cpp
    operators/gather.cpp
    operators/gemm.cpp
    operators/globalpool.cpp
    operators/matmul.cpp
    operators/maxpool.cpp
    operators/norm.cpp
    operators/pad.cpp
    operators/reshape.cpp
    operators/resize.cpp
    operators/shape.cpp
    operators/slice.cpp
    operators/softmax.cpp
    operators/split.cpp
    operators/squeeze.cpp
    operators/transpose.cpp
    operators/unaries.cpp
    operators/unsqueeze.cpp
    operators/where.cpp
  DEPENDS
    onnx2mlir_onnx_dialect
    MLIRConversionPassesIncGen
  LINK_LIBS PUBLIC
    onnx2mlir_onnx_dialect
    MLIR
)
