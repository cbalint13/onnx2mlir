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
 * \file src/conversion/passes/onnx_to_linalg/constant.cpp
 * \brief ONNX ConstantOp to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/PatternMatch.h>

#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_ConstantOp(mlir::Operation *op,
                                            mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  // Cannot handle NoneType return
  if (mlir::isa<mlir::NoneType>(op->getResult(0).getType())) {
    return mlir::emitError(loc,
                           "onnx.Constant with 'NoneType' is not supported");
  }

  mlir::Attribute valueAttr = op->getAttr("value");
  auto elemValueAttr = mlir::dyn_cast_or_null<mlir::ElementsAttr>(valueAttr);

  // Cannot handle empty tensor
  if (!elemValueAttr) {
    return mlir::emitError(
        loc, "onnx.Constant without a valid tensor 'value' attribute");
  }

  // Create the new arithmetic constant op
  rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
      op, elemValueAttr.getType(), elemValueAttr);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
