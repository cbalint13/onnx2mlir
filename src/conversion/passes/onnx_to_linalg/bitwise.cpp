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
 * \file src/conversion/passes/onnx_to_linalg/bitwise.cpp
 * \brief ONNX bitwise operations to Linalg lowering
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
OnnxToLinalg_BitwiseBinaryOps(mlir::Operation *op,
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

  auto elemType = mlir::dyn_cast<mlir::IntegerType>(resType.getElementType());
  if (!elemType) {
    return mlir::emitError(loc, opName + " requires integer element types");
  }

  auto outBrdType = getBroadcastShape(lhsType, rhsType);

  if (!outBrdType) {
    return mlir::emitError(loc, opName + " operands are not broadcastable");
  }

  if ((outBrdType) && (resType != outBrdType)) {
    return mlir::emitError(loc,
                           opName + " result not match operands broadcast");
  }

  // Create an empty tensor for the output buffer
  mlir::Value outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), resType.getElementType());

  // Create indexing maps for the elementwise broadcast operation
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

  // MLIR arith bitwise ops strictly require signless integers
  auto signlessIntType =
      mlir::IntegerType::get(op->getContext(), elemType.getWidth());

  // Lower bitwise binary operations using linalg.generic with arith dialect ops
  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, mlir::TypeRange{resType}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{outBuff}, idxMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value scalarLhs = args[0];
        mlir::Value scalarRhs = args[1];

        // Convert signed (si*)/(ui*) integer types to signless (i*)
        if (!elemType.isSignless()) {
          scalarLhs = mlir::UnrealizedConversionCastOp::create(
                          nestedBuilder, nestedLoc, signlessIntType, scalarLhs)
                          .getResult(0);
          scalarRhs = mlir::UnrealizedConversionCastOp::create(
                          nestedBuilder, nestedLoc, signlessIntType, scalarRhs)
                          .getResult(0);
        }

        mlir::Value bitwiseRes;

        if (opNameBeginsWith(opName, "BitwiseAnd")) {
          bitwiseRes = mlir::arith::AndIOp::create(nestedBuilder, nestedLoc,
                                                   scalarLhs, scalarRhs);
        } else if (opNameBeginsWith(opName, "BitwiseOr")) {
          bitwiseRes = mlir::arith::OrIOp::create(nestedBuilder, nestedLoc,
                                                  scalarLhs, scalarRhs);
        } else if (opNameBeginsWith(opName, "BitwiseXor")) {
          bitwiseRes = mlir::arith::XOrIOp::create(nestedBuilder, nestedLoc,
                                                   scalarLhs, scalarRhs);
        }

        // Convert signless results back before yielding
        if (!elemType.isSignless()) {
          bitwiseRes = mlir::UnrealizedConversionCastOp::create(
                           nestedBuilder, nestedLoc, elemType, bitwiseRes)
                           .getResult(0);
        }

        mlir::linalg::YieldOp::create(nestedBuilder, nestedLoc, bitwiseRes);
      });

  // Tag operation for downstream pass transform optimizations
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp.getResults());

  return mlir::success();
}

mlir::LogicalResult
OnnxToLinalg_BitwiseUnaryOps(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value input = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inputType) {
    return mlir::emitError(loc,
                           opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return mlir::emitError(loc,
                           opName + " result must be a ranked tensor type");
  }

  auto elemType = mlir::dyn_cast<mlir::IntegerType>(resType.getElementType());
  if (!elemType) {
    return mlir::emitError(loc, opName + " requires integer element types");
  }

  if (inputType != resType) {
    return mlir::emitError(loc, opName + " operand and result types mismatch");
  }

  // Create an empty tensor for the output buffer
  mlir::Value outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), resType.getElementType());

  // Indexing map for unary identity mapping
  mlir::AffineMap idMap = rewriter.getMultiDimIdentityMap(resType.getRank());
  llvm::SmallVector<mlir::AffineMap, 2> idxMaps = {idMap, idMap};

  // Loop iterator types for generic operation (all dims are parallel)
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
      resType.getRank(), mlir::utils::IteratorType::parallel);

  // MLIR arith bitwise ops strictly require signless integers
  auto signlessIntType =
      mlir::IntegerType::get(op->getContext(), elemType.getWidth());

  // Lower bitwise unary operations using linalg.generic with arith dialect ops
  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, mlir::TypeRange{resType}, mlir::ValueRange{input},
      mlir::ValueRange{outBuff}, idxMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value scalarInput = args[0];

        // Convert signed (si*)/(ui*) integer types to signless (i*)
        if (!elemType.isSignless()) {
          scalarInput = mlir::UnrealizedConversionCastOp::create(
                            nestedBuilder, nestedLoc, signlessIntType, scalarInput)
                            .getResult(0);
        }

        // Create an all-ones constant mask
        mlir::Value allOnesConst = mlir::arith::ConstantOp::create(
            nestedBuilder, nestedLoc, signlessIntType,
            nestedBuilder.getIntegerAttr(signlessIntType, -1));

        // Bitwise NOT is equivalent to XOR with all ones: ~x == x ^ (-1)
        mlir::Value bitwiseRes = mlir::arith::XOrIOp::create(
            nestedBuilder, nestedLoc, scalarInput, allOnesConst);

        // Convert signless results back before yielding
        if (!elemType.isSignless()) {
          bitwiseRes = mlir::UnrealizedConversionCastOp::create(
                           nestedBuilder, nestedLoc, elemType, bitwiseRes)
                           .getResult(0);
        }

        mlir::linalg::YieldOp::create(nestedBuilder, nestedLoc, bitwiseRes);
      });

  // Tag operation for downstream pass transform optimizations
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp.getResults());

  return mlir::success();
}

} // namespace onnx2mlir::dialect
