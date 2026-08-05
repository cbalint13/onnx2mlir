/******************************************************************************
 *
 * ONNX2MLIR (ONNX dialect mappings for composable optimizations)
 *
 * Authors:
 * Cristian Balint <cristian dot balint at gmail dot com>
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
 * \file src/conversion/passes/onnx_to_linalg/concat.cpp
 * \brief ONNX Concat operation to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_ConcatOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInputs = op->getOperands();

  // checks
  if (op->getNumOperands() == 0)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " must have at least one input";

  auto inpDatType =
      mlir::dyn_cast<mlir::RankedTensorType>(opInputs[0].getType());

  auto inputRank = inpDatType.getRank();

  /*
   * Attributes
   */

  // axis
  auto axisAttr = op->getAttr("axis");
  if (!axisAttr)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " missing 'axis' attribute";
  auto axisInt = mlir::dyn_cast_or_null<mlir::IntegerAttr>(axisAttr);
  if (!axisInt)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " invalid 'axis' attribute type";
  auto attr_axis = axisInt.getInt();
  if (attr_axis < -inputRank || attr_axis >= inputRank)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " invalid axis: " << attr_axis;

  if (attr_axis < 0) {
    attr_axis = inputRank + attr_axis;
  }

  /*
   * Linalg ops staging
   */

  llvm::SmallVector<mlir::Value> remappedOperands;
  if (mlir::failed(convRewriter.getRemappedValues(opInputs, remappedOperands)))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " failed to remap operands";

  auto concatOp = mlir::tensor::ConcatOp::create(rewriter, loc, attr_axis,
                                                 remappedOperands);

  concatOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, concatOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
