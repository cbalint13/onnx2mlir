add_subdirectory(dialect)
add_subdirectory(conversion)
add_subdirectory(frontend)

add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

add_executable(onnx2mlir main.cpp)
target_link_libraries(onnx2mlir
  PRIVATE
    onnx2mlir_importers
    onnx2mlir_converters
)
