
add_mlir_dialect_library(onnx2mlir_onnx_dialect
  OBJECT
    onnx_attrs.cpp
    onnx_dialect.cpp
    onnx_interface.cpp
    onnx_ops.cpp
    onnx_types.cpp
  DEPENDS
    MLIROnnxIncGen
  LINK_LIBS PUBLIC
    MLIR
)
