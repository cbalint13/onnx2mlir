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
 * \file src/dialect/onnx/support/onnx_materialize.cpp
 * \brief Onnx dialect materialization implementation
 */

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect::onnx {

mlir::Operation *OnnxDialect::materializeConstant(mlir::OpBuilder &builder,
                                                  mlir::Attribute value,
                                                  mlir::Type type,
                                                  mlir::Location loc) {
  llvm::StringRef valueKey;
  if (llvm::isa<mlir::DenseElementsAttr>(value))
    valueKey = "value";
  else if (llvm::isa<mlir::SparseElementsAttr>(value))
    valueKey = "sparse_value";
  else
    return nullptr;

  mlir::NamedAttribute attrs[] = {
      builder.getNamedAttr(valueKey, value),
      builder.getNamedAttr("onnx.origin",
                           builder.getStringAttr("materializer"))};

  return ConstantOp::create(builder, loc, type, mlir::ValueRange{}, attrs);
}

} // namespace onnx2mlir::dialect::onnx
