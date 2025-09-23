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
 * \file src/conversion/passes/onnx_to_linalg/cast.cpp
 * \brief ONNX CastOp to Linalg lowering
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

mlir::Value createArithCastOp(mlir::OpBuilder *builder,
                              const mlir::Location &loc,
                              const mlir::Value &inpElem,
                              const mlir::Type &tgtElemType) {

  mlir::Type inpElemType = inpElem.getType();

  // Same elements
  if (inpElemType == tgtElemType) {
    return inpElem;
  }

  // Float -> Float
  if (inpElemType.isFloat() && tgtElemType.isFloat()) {
    unsigned inpWidth = inpElemType.getIntOrFloatBitWidth();
    unsigned outWidth = tgtElemType.getIntOrFloatBitWidth();
    if (inpWidth < outWidth) {
      return builder->create<mlir::arith::ExtFOp>(loc, tgtElemType, inpElem);
    } else {
      return builder->create<mlir::arith::TruncFOp>(loc, tgtElemType, inpElem);
    }
    // Integer -> Integer
  } else if (inpElemType.isInteger() && tgtElemType.isInteger()) {
    unsigned inpWidth = inpElemType.getIntOrFloatBitWidth();
    unsigned outWidth = tgtElemType.getIntOrFloatBitWidth();
    // extend
    if (inpWidth < outWidth) {
      if (inpElemType.isSignedInteger()) {
        return builder->create<mlir::arith::ExtSIOp>(loc, tgtElemType, inpElem);
      } else {
        return builder->create<mlir::arith::ExtUIOp>(loc, tgtElemType, inpElem);
      }
      // truncate
    } else if (inpWidth > outWidth) {
      return builder->create<mlir::arith::TruncIOp>(loc, tgtElemType, inpElem);
    } else {
      // reinterpret (same bitwidth, different signedness)
      return builder->create<mlir::arith::BitcastOp>(loc, tgtElemType, inpElem);
    }
    // Floating -> Integer
  } else if (inpElemType.isFloat() && tgtElemType.isInteger()) {
    if (tgtElemType.isSignedInteger()) {
      return builder->create<mlir::arith::FPToSIOp>(loc, tgtElemType, inpElem);
    } else {
      return builder->create<mlir::arith::FPToUIOp>(loc, tgtElemType, inpElem);
    }
    // Integer -> Floating
  } else if (inpElemType.isInteger() && tgtElemType.isFloat()) {
    if (inpElemType.isSignedInteger()) {
      return builder->create<mlir::arith::SIToFPOp>(loc, tgtElemType, inpElem);
    } else {
      return builder->create<mlir::arith::UIToFPOp>(loc, tgtElemType, inpElem);
    }
  }

  return nullptr;
}

mlir::LogicalResult OnnxToLinalg_CastOp(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter) {

  // Input tensor type
  mlir::Value inputTensor = op->getOperand(0);
  mlir::ShapedType inputShapedType =
      mlir::dyn_cast_or_null<mlir::ShapedType>(inputTensor.getType());
  if (!inputShapedType) {
    return rewriter.notifyMatchFailure(op,
                                       "onnx.Cast input is not a shaped type");
  }

  mlir::Attribute toAttr = op->getAttr("to");
  if (!toAttr) {
    return rewriter.notifyMatchFailure(op, "onnx.Cast is missing 'to' attribute");
  }

  mlir::Type tgtElemType = {};
  // Lookup 'to' MLIR element type
  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(toAttr)) {
    tgtElemType = OnnxToMlir_dType(intAttr.getInt(), rewriter.getContext());
  } else if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(toAttr)) {
    tgtElemType = OnnxToMlir_dType(strAttr.getValue().str(), rewriter.getContext());
  } else {
    return rewriter.notifyMatchFailure(op, "onnx.Cast has invalid 'to' attribute type");
  }

  if (!tgtElemType || mlir::dyn_cast_or_null<mlir::NoneType>(tgtElemType)) {
    return rewriter.notifyMatchFailure(
        op, "onnx.Cast unsupported `to` attribute value");
  }

  mlir::Type resultTensorType = inputShapedType.clone(tgtElemType);
  mlir::Location loc = op->getLoc();

  // Input and Output are identical
  if (inputTensor.getType() == resultTensorType) {
    rewriter.replaceOp(op, inputTensor);
    return mlir::success();
  }

  // Input is a scalar
  if (inputShapedType.getRank() == 0) {
    mlir::Value castResult =
        createArithCastOp(&rewriter, loc, inputTensor, tgtElemType);
    if (!castResult) {
      return rewriter.notifyMatchFailure(
          op, "onnx.Cast unsupported scalar conversion");
    }
    rewriter.replaceOp(op, castResult);
    return mlir::success();
  }

  // 1. Create an empty tensor for the output
  mlir::Value outputBuffer = rewriter.create<mlir::tensor::EmptyOp>(
      loc, inputShapedType.getShape(), tgtElemType);

  // 2. Create the linalg.generic operation
  mlir::SmallVector<mlir::utils::IteratorType> iterators;
  for (int i = 0; i < inputShapedType.getRank(); ++i) {
    iterators.push_back(mlir::utils::IteratorType::parallel);
  }

  mlir::SmallVector<mlir::AffineMap> indexingMaps;
  indexingMaps.push_back(
      rewriter.getMultiDimIdentityMap(inputShapedType.getRank()));
  indexingMaps.push_back(
      rewriter.getMultiDimIdentityMap(inputShapedType.getRank()));

  bool bodyBuildFailed = false;
  mlir::linalg::GenericOp genericOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, resultTensorType, mlir::ValueRange{inputTensor},
      mlir::ValueRange{outputBuffer}, indexingMaps, iterators,
      [&](mlir::OpBuilder nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value inpElem = args[0];
        mlir::Value castResult =
            createArithCastOp(&nestedBuilder, nestedLoc, inpElem, tgtElemType);
        if (!castResult) {
          bodyBuildFailed = true;
          return;
        }
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, castResult);
      });

  if (bodyBuildFailed) {
    if (genericOp) {
      genericOp.erase();
    }
    return rewriter.notifyMatchFailure(op,
                                       "onnx.Cast unsupported element type "
                                       "conversion within linalg.generic body");
  }

  // Tag for transform optimization
  genericOp->setAttr("transform.target_tag",
                     rewriter.getStringAttr("onnx.Cast"));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
