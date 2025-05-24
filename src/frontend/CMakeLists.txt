set(SOURCE_IMPORTER_FRONTENDS
  importers/onnx.cpp
)
set(SOURCE_CONVERTER_FRONTENDS
  converters/onnx.cpp
)
add_library(frontends OBJECT
  ${SOURCE_IMPORTER_FRONTENDS}
  ${SOURCE_CONVERTER_FRONTENDS}
)
target_link_libraries(frontends
  PRIVATE
    onnx2mlir_onnx_dialect
    onnx2mlir_onnx_linalg_pass
  PUBLIC
    ONNX::onnx
)
