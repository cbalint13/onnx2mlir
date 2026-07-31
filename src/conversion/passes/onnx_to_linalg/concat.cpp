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
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_ConcatOp(mlir::Operation *op,
                                          mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  // Validate operands
  if (op->getNumOperands() == 0) {
    return mlir::emitError(loc, opName + " must have at least one input");
  }

  // Get 'axis' attribute (default is 0 if absent)
  int64_t axisValue = 0;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis")) {
    axisValue = axisAttr.getInt();
  }

  // Normalize axis if needed
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!inputType) {
    return mlir::emitError(loc, opName + " input must be ranked tensor");
  }

  int64_t rank = inputType.getRank();
  if (axisValue < 0) {
    axisValue += rank;
  }

  auto concatOp = mlir::tensor::ConcatOp::create(rewriter, loc, axisValue, op->getOperands());

  // Tag for transform dialect
  concatOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, concatOp.getResult());
  return mlir::success();
}

} // namespace onnx2mlir::dialect
