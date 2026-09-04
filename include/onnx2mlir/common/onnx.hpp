/******************************************************************************
 *
 * ONNX2MLIR (ONNX dialect mappings for composable optimizations)
 *
 * Authors:
 *     Cristian Balint <cristian dot balint at gmail dot com>
 *
 * Copyright (c) 2021,2025
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 *****************************************************************************/

/*!
 * \file include/onnx2mlir/common/onnx.hpp
 * \brief ONNX MLIR common routines
 */

#ifndef INCLUDE_ONNX2MLIR_COMMON_ONNX_HPP_
#define INCLUDE_ONNX2MLIR_COMMON_ONNX_HPP_

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <onnx/defs/parser.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <string>

#include "onnx2mlir/dialect/onnx/OnnxTypes.hpp"
#include "onnx2mlir/support/support.hpp"

/*
 *  Onnx to Mlir
 */

static inline mlir::Type OnnxToMlir_dType(const int32_t data_type_int,
                                          mlir::MLIRContext *ctx) {
  switch (data_type_int) {
  case onnx::TensorProto_DataType_FLOAT:
    return mlir::Float32Type::get(ctx);
#if ONNX2MLIR_ONNX_VERSION >= 121
  case onnx::TensorProto_DataType_INT2:
    return mlir::IntegerType::get(ctx, 2, mlir::IntegerType::Signed);
#endif
  case onnx::TensorProto_DataType_INT4:
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Signed);
  case onnx::TensorProto_DataType_INT8:
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
  case onnx::TensorProto_DataType_INT16:
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Signed);
  case onnx::TensorProto_DataType_INT32:
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
  case onnx::TensorProto_DataType_INT64:
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
  case onnx::TensorProto_DataType_BOOL:
    return mlir::IntegerType::get(ctx, 1);
#if ONNX2MLIR_ONNX_VERSION >= 121
  case onnx::TensorProto_DataType_UINT2:
    return mlir::IntegerType::get(ctx, 2, mlir::IntegerType::Unsigned);
#endif
  case onnx::TensorProto_DataType_UINT4:
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Unsigned);
  case onnx::TensorProto_DataType_UINT8:
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
  case onnx::TensorProto_DataType_UINT16:
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
  case onnx::TensorProto_DataType_UINT32:
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
  case onnx::TensorProto_DataType_UINT64:
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
  case onnx::TensorProto_DataType_STRING:
    return onnx2mlir::dialect::onnx::OnnxStringType::get(ctx);
  case onnx::TensorProto_DataType_FLOAT16:
    return mlir::Float16Type::get(ctx);
  case onnx::TensorProto_DataType_DOUBLE:
    return mlir::Float64Type::get(ctx);
  case onnx::TensorProto_DataType_BFLOAT16:
    return mlir::BFloat16Type::get(ctx);
  case onnx::TensorProto_DataType_FLOAT8E4M3FN:
    return mlir::Float8E4M3FNType::get(ctx);
  case onnx::TensorProto_DataType_FLOAT8E4M3FNUZ:
    return mlir::Float8E4M3FNUZType::get(ctx);
  case onnx::TensorProto_DataType_FLOAT8E5M2:
    return mlir::Float8E5M2Type::get(ctx);
  case onnx::TensorProto_DataType_FLOAT8E5M2FNUZ:
    return mlir::Float8E5M2FNUZType::get(ctx);
#if ONNX2MLIR_ONNX_VERSION >= 120
#if ONNX2MLIR_MLIR_VERSION >= 220
  case onnx::TensorProto_DataType_FLOAT8E8M0:
    return mlir::Float8E8M0FNUType::get(ctx);
#endif
#endif
#if ONNX2MLIR_ONNX_VERSION >= 123
#if ONNX2MLIR_MLIR_VERSION >= 220
  case onnx::TensorProto_DataType_FLOAT6E2M3:
    return mlir::Float6E2M3FNType::get(ctx);
  case onnx::TensorProto_DataType_FLOAT6E3M2:
    return mlir::Float6E3M2FNType::get(ctx);
#endif
#endif
  case onnx::TensorProto_DataType_FLOAT4E2M1:
    return mlir::Float4E2M1FNType::get(ctx);
  case onnx::TensorProto_DataType_COMPLEX64:
    return mlir::ComplexType::get(mlir::Float32Type::get(ctx));
  case onnx::TensorProto_DataType_COMPLEX128:
    return mlir::ComplexType::get(mlir::Float64Type::get(ctx));
  case onnx::TensorProto_DataType_UNDEFINED:
    return mlir::NoneType::get(ctx);

  default:
    onnx2mlir::error() << "Unknown ONNX data type integer value: "
                       << data_type_int << "\n";
    exit(-1);
  }

  return nullptr;
}

static inline mlir::Type OnnxToMlir_dType(const std::string data_type_str,
                                          mlir::MLIRContext *ctx) {
  std::string lcase_str = std::string(data_type_str);
  std::transform(lcase_str.begin(), lcase_str.end(), lcase_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  auto data_type_int = onnx::PrimitiveTypeNameMap::Lookup(lcase_str);
  if (data_type_int == onnx::TensorProto_DataType_UNDEFINED) {
    onnx2mlir::error() << "Unsupported ONNX data type string: '"
                       << data_type_str << "'\n";
    exit(-1);
  }

  return OnnxToMlir_dType(data_type_int, ctx);
}

/*
 *  Mlir to Onnx
 */

static inline int32_t MlirToOnnx_dType(mlir::Type type) {
  if (type.isF32())
    return onnx::TensorProto_DataType_FLOAT;
  if (type.isF64())
    return onnx::TensorProto_DataType_DOUBLE;
  if (type.isF16())
    return onnx::TensorProto_DataType_FLOAT16;
  if (type.isBF16())
    return onnx::TensorProto_DataType_BFLOAT16;
  if (llvm::isa<mlir::Float8E4M3FNType>(type))
    return onnx::TensorProto_DataType_FLOAT8E4M3FN;
  if (llvm::isa<mlir::Float8E4M3FNUZType>(type))
    return onnx::TensorProto_DataType_FLOAT8E4M3FNUZ;
  if (llvm::isa<mlir::Float8E5M2Type>(type))
    return onnx::TensorProto_DataType_FLOAT8E5M2;
  if (llvm::isa<mlir::Float8E5M2FNUZType>(type))
    return onnx::TensorProto_DataType_FLOAT8E5M2FNUZ;
#if ONNX2MLIR_ONNX_VERSION >= 120
#if ONNX2MLIR_MLIR_VERSION >= 220
  if (llvm::isa<mlir::Float8E8M0FNUType>(type))
    return onnx::TensorProto_DataType_FLOAT8E8M0;
#endif
#endif
#if ONNX2MLIR_ONNX_VERSION >= 123
#if ONNX2MLIR_MLIR_VERSION >= 220
  if (llvm::isa<mlir::Float6E2M3FNType>(type))
    return onnx::TensorProto_DataType_FLOAT6E2M3;
  if (llvm::isa<mlir::Float6E3M2FNType>(type))
    return onnx::TensorProto_DataType_FLOAT6E3M2;
#endif
#endif
  if (llvm::isa<mlir::Float4E2M1FNType>(type))
    return onnx::TensorProto_DataType_FLOAT4E2M1;
  if (auto complexType = llvm::dyn_cast<mlir::ComplexType>(type)) {
    auto elementType = complexType.getElementType();
    if (elementType.isF32())
      return onnx::TensorProto_DataType_COMPLEX64;
    if (elementType.isF64())
      return onnx::TensorProto_DataType_COMPLEX128;
  }
  if (type.isInteger(1))
    return onnx::TensorProto_DataType_BOOL;
#if ONNX2MLIR_ONNX_VERSION >= 121
  if (type.isInteger(2))
    return type.isSignedInteger() ? onnx::TensorProto_DataType_INT2
                                  : onnx::TensorProto_DataType_UINT2;
#endif
  if (type.isInteger(4))
    return type.isSignedInteger() ? onnx::TensorProto_DataType_INT4
                                  : onnx::TensorProto_DataType_UINT4;
  if (type.isInteger(8))
    return type.isSignedInteger() ? onnx::TensorProto_DataType_INT8
                                  : onnx::TensorProto_DataType_UINT8;
  if (type.isInteger(16))
    return type.isSignedInteger() ? onnx::TensorProto_DataType_INT16
                                  : onnx::TensorProto_DataType_UINT16;
  if (type.isInteger(32))
    return type.isSignedInteger() ? onnx::TensorProto_DataType_INT32
                                  : onnx::TensorProto_DataType_UINT32;
  if (type.isInteger(64))
    return type.isSignedInteger() ? onnx::TensorProto_DataType_INT64
                                  : onnx::TensorProto_DataType_UINT64;

  onnx2mlir::error() << "Unknown Mlir to Onnx data type conversion: " << type
                     << "\n";
  exit(-1);
}

static inline void MlirToOnnx_dType(mlir::Type mlirType,
                                    onnx::TypeProto &onnxType) {
  auto tensorType = llvm::dyn_cast<mlir::TensorType>(mlirType);
  if (!tensorType)
    return;

  auto *tensorProto = onnxType.mutable_tensor_type();
  tensorProto->set_elem_type(MlirToOnnx_dType(tensorType.getElementType()));

  if (tensorType.hasRank()) {
    auto *shapeProto = tensorProto->mutable_shape();
    for (int64_t dim : tensorType.getShape()) {
      auto *dimProto = shapeProto->add_dim();
      if (dim >= 0) {
        dimProto->set_dim_value(dim);
      }
    }
  }
}

#endif // INCLUDE_ONNX2MLIR_COMMON_ONNX_HPP_
