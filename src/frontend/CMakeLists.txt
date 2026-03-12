add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

add_subdirectory(importers)
add_subdirectory(converters)

if(BUILD_PYTHON_BINDINGS)
  # Onnx passes python bindings
  add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})
  set(PY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/python/onnx2mlir/importer")
  add_mlir_python_extension(onnx2mlir_py_importers _onnx2mlirImporters
    nanobind-static
    _PRIVATE_SUPPORT_LIB
    SOURCES
      python/onnx_importer_bindings.cpp
    OUTPUT_DIRECTORY
      ${PY_OUTPUT_PATH}
    LINK_LIBS
      onnx2mlir_importers
      MLIRPythonCAPI
  )
  set_target_properties(onnx2mlir_py_importers PROPERTIES PREFIX "")
  add_custom_command(TARGET onnx2mlir_py_importers
    POST_BUILD
    COMMAND
      echo "from ._onnx2mlirImporters import *" > __init__.py
    COMMAND
      echo "__all__ = ['import_from_onnx']" >> __init__.py
    WORKING_DIRECTORY ${PY_OUTPUT_PATH}
  )
endif()
