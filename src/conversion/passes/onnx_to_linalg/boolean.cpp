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
 * \file src/conversion/passes/onnx_to_linalg/boolean.cpp
 * \brief ONNX boolean operations to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_BooleanBinaryOps(mlir::Operation *op,
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
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " operands must be ranked tensor type");
  }

  if (lhsType.getElementType() != rhsType.getElementType()) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " operands element type are different");
  }

  if (!resType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " result must be a ranked tensor type");
  }

  auto elemType = mlir::dyn_cast<mlir::IntegerType>(resType.getElementType());
  if (!elemType || elemType.getWidth() != 1) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " requires boolean element type");
  }

  auto outBrdType = getBroadcastShape(lhsType, rhsType);

  if (!outBrdType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " operands are not broadcastable");
  }

  if ((outBrdType) && (resType != outBrdType)) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " result not match operands broadcast");
  }

  // Create an empty tensor for the output buffer
  mlir::Value outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), resType.getElementType());

  llvm::SmallVector<mlir::AffineMap, 3> idxMaps;
  mlir::AffineMap lhsMap, rhsMap, resMap;

  // Create identity map for the result tensor
  resMap = rewriter.getMultiDimIdentityMap(resType.getRank());

  mlir::Builder builder(op->getContext());
  mlir::AffineExpr zero = builder.getAffineConstantExpr(0);

  // Create broadcast mapping for the LHS tensor
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

  // Create broadcast mapping for the RHS tensor
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

  // Loop iterator types for generic operation (all dims are parallel)
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
      resType.getRank(), mlir::utils::IteratorType::parallel);

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, mlir::TypeRange{resType}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{outBuff}, idxMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value scalarLhs = args[0];
        mlir::Value scalarRhs = args[1];
        mlir::Value logicRes;

        if (opNameBeginsWith(opName, "And")) {
          logicRes = mlir::arith::AndIOp::create(nestedBuilder, nestedLoc,
                                                 scalarLhs, scalarRhs);
        } else if (opNameBeginsWith(opName, "Or")) {
          logicRes = mlir::arith::OrIOp::create(nestedBuilder, nestedLoc,
                                                scalarLhs, scalarRhs);
        } else if (opNameBeginsWith(opName, "Xor")) {
          logicRes = mlir::arith::XOrIOp::create(nestedBuilder, nestedLoc,
                                                 scalarLhs, scalarRhs);
        }

        mlir::linalg::YieldOp::create(nestedBuilder, nestedLoc, logicRes);
      });

  // Tag operation for downstream pass transform optimizations
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp.getResults());

  return mlir::success();
}

} // namespace onnx2mlir::dialect
