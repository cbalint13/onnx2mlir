add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

# Onnx dialect python gen
if(BUILD_PYTHON_BINDINGS)
  set(PY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/python/onnx2mlir/passes")
  add_custom_command(
    OUTPUT "${OUTPUT_PY_PATH}/__init__.py"
    COMMAND
      echo "from ._onnx2mlirOnnxPasses import *" > __init__.py
    WORKING_DIRECTORY ${PY_OUTPUT_PATH}
    COMMENT "Update ConversionPasses python bindings"
  )
  add_custom_target(generate_onnx_passes_py_bindings
    DEPENDS "${OUTPUT_PY_PATH}/__init__.py"
    COMMENT "Finish ConversionPasses python bindings"
  )
endif()

add_mlir_conversion_library(onnx2mlir_onnx_linalg_pass
    onnx_to_linalg.cpp
    onnx_to_linalg/constant.cpp
  DEPENDS
    onnx2mlir_onnx_dialect
    MLIRConversionPassesIncGen
  LINK_LIBS PUBLIC
    onnx2mlir_onnx_dialect
    MLIR
)
if(BUILD_PYTHON_BINDINGS)
  add_dependencies(onnx2mlir_onnx_linalg_pass generate_onnx_passes_py_bindings)
endif()
