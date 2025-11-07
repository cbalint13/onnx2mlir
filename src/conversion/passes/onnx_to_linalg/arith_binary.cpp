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
 * \file src/conversion/passes/onnx_to_linalg/arith_binary.cpp
 * \brief ONNX Arith binary operations to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_ArithBinaryOps(mlir::Operation *op,
                            mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value lhs = op->getOperand(0);
  mlir::Value rhs = op->getOperand(1);
  mlir::Value res = op->getResult(0);

  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if ((!lhsType) || (!rhsType)) {
    return mlir::emitError(loc,
                           opName + " operands must be ranked tensor type");
  }

  if (lhsType.getElementType() != rhsType.getElementType()) {
    return mlir::emitError(loc,
                           opName + " operands element type are different");
  }

  if (!resType) {
    return mlir::emitError(loc,
                           opName + " result must be a ranked tensor type");
  }

  if (opNameBeginsWith(opName, "Pow") &&
      mlir::isa<mlir::IntegerType>(resType.getElementType())) {
    return mlir::emitError(loc, opName + " not supported with integer types");
  }

  // Infer broadcasted shape output
  auto outBrdType = getBroadcastShape(lhsType, rhsType);

  if (!outBrdType) {
    return mlir::emitError(loc, opName + " operands are not broadcastable");
  }

  if ((outBrdType) && (resType != outBrdType)) {
    return mlir::emitError(loc,
                           opName + " result not match operands broadcast");
  }

  // Create an empty tensor for the output
  mlir::Value outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), resType.getElementType());

  // Create indexing maps for the elementwise op
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

  auto idxMapsAttr = rewriter.getAffineMapArrayAttr(idxMaps);

  mlir::linalg::ElementwiseKind kindEnum;
  if (opNameBeginsWith(opName, "Add")) {
    kindEnum = mlir::linalg::ElementwiseKind::add;
  } else if (opNameBeginsWith(opName, "Sub")) {
    kindEnum = mlir::linalg::ElementwiseKind::sub;
  } else if (opNameBeginsWith(opName, "Mul")) {
    kindEnum = mlir::linalg::ElementwiseKind::mul;
  } else if (opNameBeginsWith(opName, "Div")) {
    kindEnum = mlir::linalg::ElementwiseKind::div;
  } else if (opNameBeginsWith(opName, "Pow")) {
    if (mlir::isa<mlir::FloatType>(resType.getElementType()))
      kindEnum = mlir::linalg::ElementwiseKind::powf;
    else
      return mlir::emitError(loc,
                             opName + " supports only float element types");
  } else {
    return mlir::emitError(
        loc, opName + " is unsupported for linalg.elementwise operation");
  }

  auto kindAttr =
      mlir::linalg::ElementwiseKindAttr::get(op->getContext(), kindEnum);

  auto elmwiseOp = mlir::linalg::ElementwiseOp::create(
      rewriter, loc, mlir::ValueRange{lhs, rhs}, mlir::ValueRange{outBuff},
      kindAttr, idxMapsAttr);

  // Tag for transform optimization
  elmwiseOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, elmwiseOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
