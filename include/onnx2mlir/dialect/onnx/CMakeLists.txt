add_mlir_dialect(Onnx onnx)
add_mlir_doc(OnnxDialect OnnxDialect onnx/ -gen-dialect-doc)
add_mlir_doc(OnnxOps OnnxOps onnx/ -gen-op-doc)
add_mlir_doc(OnnxTypes OnnxTypes onnx/ -gen-type-doc)

set(ONNX_MLIR_OPS_GEN "${CMAKE_SOURCE_DIR}/tools/gen_onnx_dia.py")
set(ONNX_MLIR_OPS_INC "${CMAKE_CURRENT_SOURCE_DIR}/OnnxOps.td.inc")

add_custom_command(
  OUTPUT ${ONNX_MLIR_OPS_INC}
  COMMAND ${Python3_EXECUTABLE} ${ONNX_MLIR_OPS_GEN}
  ARGS ${ONNX_MLIR_OPS_INC}
  DEPENDS ${ONNX_MLIR_OPS_GEN}
  COMMENT "Generating ONNX MLIR Ops using gen_onnx_dia.py"
)

add_custom_target(generate_onnx_ops ALL
    DEPENDS ${ONNX_MLIR_OPS_INC}
    COMMENT "MLIR ONNX Ops generator"
)

add_dependencies(MLIROnnxIncGen generate_onnx_ops)

#set(LLVM_TARGET_DEFINITIONS Passes.td)
#mlir_tablegen(Passes.h.inc -gen-pass-decls -name RelayOpt)
#add_public_tablegen_target(mlir_relay_pass_inc_gen)
#add_mlir_doc(Passes RelayPasses ./ -gen-pass-doc)
