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
 * \file src/conversion/passes/onnx_to_linalg/flatten.cpp
 * \brief ONNX FlattenOp to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_FlattenOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();
  auto *context = rewriter.getContext();

  mlir::Value inp = op->getOperand(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());

  if (!inpType || !resType) {
    return mlir::emitError(loc, opName + " requires ranked tensor types");
  }

  // Get axis attribute
  int64_t axis = 1;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis")) {
    axis = axisAttr.getInt();
  }
  int64_t rank = inpType.getRank();
  if (axis < 0)
    axis += rank;
  axis = std::clamp<int64_t>(axis, 0, rank);

  // Flatten to 2D output
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      2, mlir::utils::IteratorType::parallel);

  // Output map is identity: (d0, d1) -> (d0, d1)
  auto outMap = rewriter.getMultiDimIdentityMap(2);

  // Input map: (d0, d1) -> (i_0, i_1, ..., i_{rank-1})
  auto inpShape = inpType.getShape();
  auto d0 = rewriter.getAffineDimExpr(0);
  auto d1 = rewriter.getAffineDimExpr(1);

  llvm::SmallVector<mlir::AffineExpr> inpExprs(rank);

  // Delinearize d0 into input dims [0, axis)
  mlir::AffineExpr current0 = d0;
  for (int i = axis - 1; i >= 0; --i) {
    if (i == 0) {
      inpExprs[i] = current0;
    } else {
      inpExprs[i] = current0 % inpShape[i];
      current0 = current0.floorDiv(inpShape[i]);
    }
  }

  // Delinearize d1 into input dims [axis, rank)
  mlir::AffineExpr current1 = d1;
  for (int i = rank - 1; i >= axis; --i) {
    if (i == axis) {
      inpExprs[i] = current1;
    } else {
      inpExprs[i] = current1 % inpShape[i];
      current1 = current1.floorDiv(inpShape[i]);
    }
  }

  auto inpMap = mlir::AffineMap::get(2, 0, inpExprs, context);

  // Create the GenericOp
  auto outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), resType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, resType, mlir::ValueRange{inp},
      mlir::ValueRange{outBuff.getResult()},
      llvm::ArrayRef<mlir::AffineMap>{inpMap, outMap}, iterators,
      [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
        mlir::linalg::YieldOp::create(nest, l, args[0]);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp.getResult(0));

  return mlir::success();
}

} // namespace onnx2mlir::dialect
