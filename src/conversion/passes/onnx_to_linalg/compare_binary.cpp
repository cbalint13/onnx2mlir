/******************************************************************+************
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
 * \file src/conversion/passes/onnx_to_linalg/compare_binary.cpp
 * \brief ONNX Comparison binary operations to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_CompBinaryOps(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) {
  auto opName = op->getName().getStringRef();

  mlir::Value lhs = op->getOperand(0);
  mlir::Value rhs = op->getOperand(1);
  mlir::Value res = op->getResult(0);

  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if ((!lhsType) || (!rhsType)) {
    return rewriter.notifyMatchFailure(
        op, opName + " operands must be ranked tensor type");
  }

  if (lhsType.getElementType() != rhsType.getElementType()) {
    return rewriter.notifyMatchFailure(
        op, opName + " operands element type are different");
  }

  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, opName + " result must be a ranked tensor type");
  }

  if (auto intType =
          mlir::dyn_cast<mlir::IntegerType>(resType.getElementType())) {
    if (intType.getWidth() != 1) {
      return rewriter.notifyMatchFailure(
          op, opName + " result must have boolean (i1) element type");
    }
  }

  // Infer broadcasted shape output
  auto outBrdType = getBroadcastShape(lhsType, rhsType);

  if (!outBrdType) {
    return rewriter.notifyMatchFailure(
        op, opName + " operands are not broadcastable");
  }

  if (resType.getShape() != outBrdType.getShape()) {
    return rewriter.notifyMatchFailure(
        op, opName + " result not match operands broadcast shape");
  }

  // Create an empty tensor for the output
  mlir::Value outBuff = rewriter.create<mlir::tensor::EmptyOp>(
      op->getLoc(), resType.getShape(), resType.getElementType());

  // Create indexing maps for the generic op
  llvm::SmallVector<mlir::AffineMap, 4> idxMaps;
  mlir::AffineMap lhsMap, rhsMap, resMap;

  // Create identity map for the result
  resMap = rewriter.getMultiDimIdentityMap(resType.getRank());

  mlir::Builder builder(op->getContext());
  mlir::AffineExpr zero = builder.getAffineConstantExpr(0);

  // Create the map for the LHS
  llvm::SmallVector<mlir::AffineExpr, 4> lhsExprs;
  for (unsigned i = 0; i < resType.getRank(); ++i) {
    int64_t lhsDimIndex = lhsType.getRank() - (resType.getRank() - i);
    if (lhsDimIndex >= 0) {
      if (lhsType.getDimSize(lhsDimIndex) == 1)
        lhsExprs.push_back(zero);
      else
        lhsExprs.push_back(builder.getAffineDimExpr(i));
    }
  }
  lhsMap = mlir::AffineMap::get(resType.getRank(), 0, lhsExprs,
                                builder.getContext());

  // Create the map for the RHS
  llvm::SmallVector<mlir::AffineExpr, 4> rhsExprs;
  for (unsigned i = 0; i < resType.getRank(); ++i) {
    int64_t rhsDimIndex = rhsType.getRank() - (resType.getRank() - i);
    if (rhsDimIndex >= 0) {
      if (rhsType.getDimSize(rhsDimIndex) == 1)
        rhsExprs.push_back(zero);
      else
        rhsExprs.push_back(builder.getAffineDimExpr(i));
    }
  }
  rhsMap = mlir::AffineMap::get(resType.getRank(), 0, rhsExprs,
                                builder.getContext());

  idxMaps.push_back(lhsMap);
  idxMaps.push_back(rhsMap);
  idxMaps.push_back(resMap);

  // Create the comparison op with linalg.generic
  auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
      op->getLoc(), resType, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{outBuff}, idxMaps,
      llvm::SmallVector<mlir::utils::IteratorType>(
          resType.getRank(), mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &nest, mlir::Location loc, mlir::ValueRange vals) {
        mlir::Value outOp;
        if (mlir::isa<mlir::FloatType>(lhsType.getElementType())) {
          if (opNameBeginsWith(opName, "Equal"))
            outOp = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OEQ, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Greater"))
            outOp = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGT, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "GreaterOrEqual"))
            outOp = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Less"))
            outOp = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "LessOrEqual"))
            outOp = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLE, vals[0], vals[1]);
        } else {
          if (opNameBeginsWith(opName, "Equal"))
            outOp = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Greater"))
            outOp = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::sgt, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "GreaterOrEqual"))
            outOp = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::sge, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Less"))
            outOp = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::slt, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "LessOrEqual"))
            outOp = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::sle, vals[0], vals[1]);
        }
        nest.create<mlir::linalg::YieldOp>(loc, outOp);
      });

  // Tag for transform optimization
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
