set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(MLIRConversionPassIncGen)
add_mlir_doc(Passes ConversionPasses ./ -gen-pass-doc)


# Onnx dialect library
add_mlir_conversion_library(onnx2mlir_onnx_linalg_pass
  OBJECT
    passes/onnx_to_linalg.cpp
  DEPENDS
    onnx2mlir_onnx_dialect
    MLIRConversionPassIncGen
  LINK_LIBS PUBLIC
    MLIR
)
