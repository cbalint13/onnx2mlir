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
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

static mlir::Value createArithCastOp(mlir::OpBuilder *builder,
                                     const mlir::Location &loc,
                                     const mlir::Value &inp,
                                     const mlir::Type &srcElemType,
                                     const mlir::Type &tgtElemType) {
  unsigned srcWidth = srcElemType.getIntOrFloatBitWidth();
  unsigned tgtWidth = tgtElemType.getIntOrFloatBitWidth();

  if (srcElemType == tgtElemType)
    return inp;

  if (tgtElemType.isInteger(1)) {
    // Floating -> Bool
    if (srcElemType.isFloat()) {
      auto zero = mlir::arith::ConstantOp::create(
          *builder, loc, builder->getFloatAttr(srcElemType, 0.0));
      return mlir::arith::CmpFOp::create(
          *builder, loc, mlir::arith::CmpFPredicate::UNE, inp, zero);
      // Bool -> Floating
    } else if (srcElemType.isInteger()) {
      mlir::Type nosgnSrcType = builder->getIntegerType(srcWidth);
      auto zero = mlir::arith::ConstantOp::create(
          *builder, loc, builder->getIntegerAttr(nosgnSrcType, 0));
      return mlir::arith::CmpIOp::create(
          *builder, loc, mlir::arith::CmpIPredicate::ne, inp, zero);
    }
  }

  // Float -> Float
  if (srcElemType.isFloat() && tgtElemType.isFloat()) {
    if (srcWidth < tgtWidth) {
      return mlir::arith::ExtFOp::create(*builder, loc, tgtElemType, inp);
    } else if (srcWidth > tgtWidth) {
      return mlir::arith::TruncFOp::create(*builder, loc, tgtElemType, inp);
    }
    // Integer -> Integer
  } else if (srcElemType.isInteger() && tgtElemType.isInteger()) {
    mlir::Type nosgnTgtType = builder->getIntegerType(tgtWidth);
    if (srcWidth < tgtWidth) {
      if (srcElemType.isSignedInteger()) {
        return mlir::arith::ExtSIOp::create(*builder, loc, nosgnTgtType, inp);
      } else {
        return mlir::arith::ExtUIOp::create(*builder, loc, nosgnTgtType, inp);
      }
    } else if (srcWidth > tgtWidth) {
      return mlir::arith::TruncIOp::create(*builder, loc, nosgnTgtType, inp);
    } else {
      // Same bitwidth, different signedness
      return mlir::arith::BitcastOp::create(*builder, loc, nosgnTgtType, inp);
    }
    // Floating -> Integer
  } else if (srcElemType.isFloat() && tgtElemType.isInteger()) {
    mlir::Type nosgnTgtType = builder->getIntegerType(tgtWidth);
    if (tgtElemType.isSignedInteger()) {
      return mlir::arith::FPToSIOp::create(*builder, loc, nosgnTgtType, inp);
    } else {
      return mlir::arith::FPToUIOp::create(*builder, loc, nosgnTgtType, inp);
    }
    // Integer -> Floating
  } else if (srcElemType.isInteger() && tgtElemType.isFloat()) {
    if (srcElemType.isSignedInteger()) {
      return mlir::arith::SIToFPOp::create(*builder, loc, tgtElemType, inp);
    } else {
      return mlir::arith::UIToFPOp::create(*builder, loc, tgtElemType, inp);
    }
  }

  return nullptr;
}

mlir::LogicalResult
OnnxToLinalg_CastOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter) {
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  mlir::Value inp = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(
      typeConverter->convertType(res.getType()));

  auto srcType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  // Input and Output are identical
  if (srcType == outType) {
    rewriter.replaceOp(op, inp);
    return mlir::success();
  }

  if (!inpType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " input is not a tensor type");
  }

  auto toAttr = op->getAttr("to");
  if (!toAttr) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + "  is missing 'to' attribute");
  }

  mlir::Type srcElemType = srcType.getElementType();
  mlir::Type tgtElemType = {};
  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(toAttr)) {
    tgtElemType = OnnxToMlir_dType(intAttr.getInt(), rewriter.getContext());
  } else if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(toAttr)) {
    tgtElemType =
        OnnxToMlir_dType(strAttr.getValue().str(), rewriter.getContext());
  } else {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " has invalid 'to' attribute type");
  }

  if (!tgtElemType || mlir::dyn_cast_or_null<mlir::NoneType>(tgtElemType)) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " unsupported `to` attribute value");
  }

  // Set output type using 'to' attribute
  auto tgtType = inpType.clone(tgtElemType);

  if (tgtType != outType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName +
                               " 'to' data type not match the result type");
  }

  mlir::Location loc = op->getLoc();

  // Input is a scalar
  if (inpType.getRank() == 0) {
    auto castResult =
        createArithCastOp(&rewriter, loc, inp, srcElemType, tgtElemType);
    if (!castResult) {
      return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                             opName + " unsupported scalar conversion");
    }
    rewriter.replaceOp(op, castResult);
    return mlir::success();
  }

  // 1. Create an empty tensor for the output
  mlir::Value outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, inpType.getShape(), resType.getElementType());

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
      rewriter, loc, resType, mlir::ValueRange{inp}, mlir::ValueRange{outBuff},
      idxMaps, iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value outOp =
            createArithCastOp(&nest, loc, args[0], srcElemType, tgtElemType);
        if (!outOp) {
          bodyBuildFailed = true;
          return;
        }
        mlir::linalg::YieldOp::create(nest, loc, outOp);
      });

  if (bodyBuildFailed) {
    if (genericOp)
      genericOp.erase();
    return mlir::emitError(
        Onnx2Mlir_SrcLoc(rewriter),
        opName + " unsupported element type within linalg.generic body");
  }

  // Tag for transform optimization
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
