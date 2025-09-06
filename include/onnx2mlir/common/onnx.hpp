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

#include "onnx/onnx_pb.h"

static inline mlir::Type OnnxToMlir_dType(const int32_t data_type_int,
                                          mlir::MLIRContext *ctx) {
  switch (data_type_int) {
  case onnx::TensorProto_DataType_FLOAT:
    return mlir::Float32Type::get(ctx);
  case onnx::TensorProto_DataType_INT4:
    return mlir::IntegerType::get(ctx, 4);
  case onnx::TensorProto_DataType_INT8:
    return mlir::IntegerType::get(ctx, 8);
  case onnx::TensorProto_DataType_INT16:
    return mlir::IntegerType::get(ctx, 16);
  case onnx::TensorProto_DataType_INT32:
    return mlir::IntegerType::get(ctx, 32);
  case onnx::TensorProto_DataType_INT64:
    return mlir::IntegerType::get(ctx, 64);
  case onnx::TensorProto_DataType_BOOL:
    return mlir::IntegerType::get(ctx, 1);
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
    return mlir::NoneType::get(ctx);
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
  case onnx::TensorProto_DataType_FLOAT8E8M0:
    return mlir::Float8E8M0FNUType::get(ctx);
  case onnx::TensorProto_DataType_FLOAT4E2M1:
    return mlir::Float4E2M1FNType::get(ctx);
  case onnx::TensorProto_DataType_COMPLEX64:
    return mlir::ComplexType::get(mlir::Float32Type::get(ctx));
  case onnx::TensorProto_DataType_COMPLEX128:
    return mlir::ComplexType::get(mlir::Float64Type::get(ctx));
  case onnx::TensorProto_DataType_UNDEFINED:
    return mlir::NoneType::get(ctx);

  default:
    llvm::errs() << "ERROR: Unknown ONNX data type integer value: "
                 << data_type_int << "\n";
    exit(-1);
  }

  return nullptr;
}

#endif // INCLUDE_ONNX2MLIR_COMMON_ONNX_HPP_
