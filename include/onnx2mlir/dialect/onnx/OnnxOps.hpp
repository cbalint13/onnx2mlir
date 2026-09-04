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
 * \file include/onnx2mlir/dialect/onnx/OnnxOps.hpp
 * \brief Onnx dialect operations declaration
 */

#ifndef INCLUDE_ONNX2MLIR_DIALECT_ONNX_ONNXOPS_HPP_
#define INCLUDE_ONNX2MLIR_DIALECT_ONNX_ONNXOPS_HPP_

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace onnx2mlir::dialect::onnx {

/*
 * Shape inference helper
 */

void inferONNXOpShape(mlir::Operation *op, llvm::StringRef onnxOpName,
                      int opsetVersion);

/*
 * Fold operation helpers
 */

mlir::OpFoldResult foldONNXOp(mlir::Operation *op,
                              llvm::ArrayRef<mlir::Attribute> operands);

mlir::OpFoldResult foldONNXOp(mlir::Operation *op, mlir::DictionaryAttr attrs);

template <typename Adaptor>
inline auto foldONNXOp(mlir::Operation *op, Adaptor adaptor)
    -> decltype(adaptor.getAttributes(), mlir::OpFoldResult()) {
  return foldONNXOp(op, adaptor.getAttributes());
}

} // namespace onnx2mlir::dialect::onnx

#define GET_OP_CLASSES
#include "dialect/onnx/Onnx.h.inc"
#undef GET_OP_CLASSES

#endif // INCLUDE_ONNX2MLIR_DIALECT_ONNX_ONNXOPS_HPP_
