cmake_minimum_required(VERSION 3.15)
project(onnx2mlir LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_PYTHON_BINDINGS "Build with python bindings." ON)

if(BUILD_PYTHON_BINDINGS)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python3 COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)
endif()

find_package(ONNX REQUIRED 1.17)
if(ONNX_FOUND)
  message(STATUS "Found ONNX version: ${ONNX_VERSION}")
  message(STATUS "Found ONNX includes: ${ONNX_INCLUDE_DIRS}")
endif()
# onnx variant
find_path(ONNX_PROTOBUF_INCLUDE_DIR NAMES onnx/onnx.pb.h)
find_path(ONNX_PROTOBUF_ML_INCLUDE_DIR NAMES onnx/onnx-ml.pb.h)
if(ONNX_PROTOBUF_INCLUDE_DIR)
    message(STATUS "Standard ONNX_ML=0 protobuf headers found.")
    set(ONNX_USE_ML 0)
elseif(ONNX_PROTOBUF_ML_INCLUDE_DIR)
    message(STATUS "Full ONNX_ML=1 protobuf headers found.")
    set(ONNX_USE_ML 1)
else()
    message(FATAL "ONNX protobuf header not found.")
endif()

# hardcode for tests
set(LLVM_CONFIG "llvm-config-22" CACHE STRING "Build with LLVM set to specific llvm-config path")

# use llvm config
if (DEFINED LLVM_CONFIG)
  message(STATUS "Using ${LLVM_CONFIG}")
  separate_arguments(LLVM_CONFIG)
  execute_process(COMMAND ${LLVM_CONFIG} --libdir
    RESULT_VARIABLE __llvm_exit_code
    OUTPUT_VARIABLE __llvm_cmakedir_space
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT "${__llvm_exit_code}" STREQUAL "0")
    message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} --cmakedir")
  else()
    list(APPEND CMAKE_PREFIX_PATH "${__llvm_cmakedir_space}/cmake")
  endif()
else()
  message(FATAL_ERROR "Please use -DLLVM_CONFIG=<path-to-llvm-config> for a valid llvm setup.")
endif()

find_package(MLIR REQUIRED CONFIG)
if(MLIR_FOUND)
  message(STATUS "Found LLVM/MLIR version: ${LLVM_VERSION}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Werror -Wno-comment -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-function")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src)

# python bindings
if(BUILD_PYTHON_BINDINGS)
  include(AddMLIRPython)
  add_subdirectory(python)
endif()
