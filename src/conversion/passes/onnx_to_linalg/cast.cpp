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
      return mlir::arith::ExtFOp::create(*builder, loc, tgtElemType, inpElem);
    } else {
      return mlir::arith::TruncFOp::create(*builder, loc, tgtElemType, inpElem);
    }
    // Integer -> Integer
  } else if (inpElemType.isInteger() && tgtElemType.isInteger()) {
    unsigned inpWidth = inpElemType.getIntOrFloatBitWidth();
    unsigned outWidth = tgtElemType.getIntOrFloatBitWidth();
    // extend
    if (inpWidth < outWidth) {
      if (inpElemType.isSignedInteger()) {
        return mlir::arith::ExtSIOp::create(*builder, loc, tgtElemType,
                                            inpElem);
      } else {
        return mlir::arith::ExtUIOp::create(*builder, loc, tgtElemType,
                                            inpElem);
      }
      // truncate
    } else if (inpWidth > outWidth) {
      return mlir::arith::TruncIOp::create(*builder, loc, tgtElemType, inpElem);
    } else {
      // reinterpret (same bitwidth, different signedness)
      return mlir::arith::BitcastOp::create(*builder, loc, tgtElemType,
                                            inpElem);
    }
    // Floating -> Integer
  } else if (inpElemType.isFloat() && tgtElemType.isInteger()) {
    if (tgtElemType.isSignedInteger()) {
      return mlir::arith::FPToSIOp::create(*builder, loc, tgtElemType, inpElem);
    } else {
      return mlir::arith::FPToUIOp::create(*builder, loc, tgtElemType, inpElem);
    }
    // Integer -> Floating
  } else if (inpElemType.isInteger() && tgtElemType.isFloat()) {
    if (inpElemType.isSignedInteger()) {
      return mlir::arith::SIToFPOp::create(*builder, loc, tgtElemType, inpElem);
    } else {
      auto signlessIntType = mlir::IntegerType::get(
          builder->getContext(), inpElemType.getIntOrFloatBitWidth());
      auto signlessVal = mlir::UnrealizedConversionCastOp::create(
          *builder, loc, signlessIntType, inpElem);
      return mlir::arith::UIToFPOp::create(*builder, loc, tgtElemType,
                                           signlessVal.getResult(0));
    }
  }

  return nullptr;
}

mlir::LogicalResult OnnxToLinalg_CastOp(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter) {
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return rewriter.notifyMatchFailure(op,
                                       opName + " input is not a tensor type");
  }

  auto toAttr = op->getAttr("to");
  if (!toAttr) {
    return rewriter.notifyMatchFailure(op,
                                       opName + " is missing 'to' attribute");
  }

  mlir::Type tgtElemType = {};
  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(toAttr)) {
    tgtElemType = OnnxToMlir_dType(intAttr.getInt(), rewriter.getContext());
  } else if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(toAttr)) {
    tgtElemType =
        OnnxToMlir_dType(strAttr.getValue().str(), rewriter.getContext());
  } else {
    return rewriter.notifyMatchFailure(
        op, opName + " has invalid 'to' attribute type");
  }

  if (!tgtElemType || mlir::dyn_cast_or_null<mlir::NoneType>(tgtElemType)) {
    return rewriter.notifyMatchFailure(
        op, opName + " unsupported `to` attribute value");
  }

  // Set output type using 'to' attribute
  auto outType = inpType.clone(tgtElemType);

  if (outType != resType) {
    return rewriter.notifyMatchFailure(
        op, opName + " 'to' data type not match the result type");
  }

  mlir::Location loc = op->getLoc();

  // Input and Output are identical
  if (inpType == outType) {
    rewriter.replaceOp(op, inp);
    return mlir::success();
  }

  // Input is a scalar
  if (inpType.getRank() == 0) {
    auto castResult = createArithCastOp(&rewriter, loc, inp, tgtElemType);
    if (!castResult) {
      return rewriter.notifyMatchFailure(
          op, opName + " unsupported scalar conversion");
    }
    rewriter.replaceOp(op, castResult);
    return mlir::success();
  }

  // 1. Create an empty tensor for the output
  mlir::Value outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, inpType.getShape(), tgtElemType);

  // 2. Create the linalg.generic operation
  mlir::SmallVector<mlir::utils::IteratorType> iterators;
  for (int i = 0; i < inpType.getRank(); ++i) {
    iterators.push_back(mlir::utils::IteratorType::parallel);
  }

  mlir::SmallVector<mlir::AffineMap> idxMaps;
  idxMaps.push_back(rewriter.getMultiDimIdentityMap(inpType.getRank()));
  idxMaps.push_back(rewriter.getMultiDimIdentityMap(inpType.getRank()));

  bool bodyBuildFailed = false;
  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, outType, mlir::ValueRange{inp}, mlir::ValueRange{outBuff},
      idxMaps, iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value outOp = createArithCastOp(&nest, loc, args[0], tgtElemType);
        if (!outOp) {
          bodyBuildFailed = true;
          return;
        }
        mlir::linalg::YieldOp::create(nest, loc, outOp);
      });

  if (bodyBuildFailed) {
    if (genericOp)
      genericOp.erase();
    return rewriter.notifyMatchFailure(
        op, opName + " unsupported element type within linalg.generic body");
  }

  // Tag for transform optimization
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
