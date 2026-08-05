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
 * \file src/conversion/passes/onnx_to_linalg/constant.cpp
 * \brief ONNX ConstantOp, ConstantOfShapeOp to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_ConstantOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                        const mlir::TypeConverter *typeConverter) {
  auto opName = op->getName().getStringRef();

  // Get legit result type
  auto resType = typeConverter->convertType(op->getResult(0));

  // Cannot handle NoneType return
  if (mlir::isa<mlir::NoneType>(resType)) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " with 'NoneType' is not supported");
  }
  // Get the 'value' attribute
  mlir::Attribute valueAttr = op->getAttr("value");
  auto typedAttr = mlir::dyn_cast_or_null<mlir::TypedAttr>(valueAttr);

  // Cannot handle empty tensor
  if (!typedAttr) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName +
                               " without a valid tensor 'value' attribute");
  }

  bool isChanged = false;
  // Match value type to result type
  if (resType != typedAttr.getType()) {
    isChanged = true;
    typedAttr = changeAttrType(valueAttr, resType);
  }

  // Create the new arithmetic constant op
  auto constOp = rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
      op, resType, typedAttr);

  if (isChanged) {
    // preserve metadata
    constOp->setAttr("onnx_value", valueAttr);
  }

  return mlir::success();
}

mlir::LogicalResult
OnnxToLinalg_ConstantOfShapeOp(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter,
                               const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto resType = typeConverter->convertType(op->getResult(0));

  // Cannot handle NoneType return
  if (mlir::isa<mlir::NoneType>(resType)) {
    return mlir::emitError(loc, opName + " with 'NoneType' is not supported");
  }

  auto rankedResType = mlir::dyn_cast<mlir::RankedTensorType>(resType);
  if (!rankedResType) {
    return mlir::emitError(loc, opName + " result must be a tensor type");
  }

  mlir::TypedAttr typedScalarAttr;
  mlir::Attribute valueAttr = op->getAttr("value");

  if (valueAttr) {
    if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr)) {
      if (denseAttr.isSplat()) {
        typedScalarAttr = mlir::dyn_cast<mlir::TypedAttr>(
            denseAttr.getSplatValue<mlir::Attribute>());
      } else if (denseAttr.getNumElements() > 0) {
        typedScalarAttr = mlir::dyn_cast<mlir::TypedAttr>(
            *denseAttr.value_begin<mlir::Attribute>());
      }
    } else if (auto tAttr = mlir::dyn_cast<mlir::TypedAttr>(valueAttr)) {
      typedScalarAttr = tAttr;
    }
  }

  bool isUnsignedInteger = false;
  if (auto intAttr =
          mlir::dyn_cast_or_null<mlir::IntegerAttr>(typedScalarAttr)) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(intAttr.getType())) {
      isUnsignedInteger = intType.isUnsigned();
      if (!intType.isSignless()) {
        auto signlessType = rewriter.getIntegerType(intType.getWidth());
        typedScalarAttr =
            rewriter.getIntegerAttr(signlessType, intAttr.getValue());
      }
    }
  }

  // if no 'value' default to f32
  if (!typedScalarAttr) {
    typedScalarAttr = rewriter.getFloatAttr(rewriter.getF32Type(), 0.0);
  }

  // Fill element type using 'value' attribute
  mlir::Type fillElemType = typedScalarAttr.getType();

  mlir::Value shapeInput = op->getOperand(0);
  mlir::SmallVector<mlir::Value, 4> dynamicSizes;

  for (int64_t i = 0; i < rankedResType.getRank(); ++i) {
    if (rankedResType.isDynamicDim(i)) {
      auto idxVal = mlir::arith::ConstantOp::create(rewriter, loc,
                                                    rewriter.getIndexAttr(i));
      auto dimVal = mlir::tensor::ExtractOp::create(rewriter, loc, shapeInput,
                                                    mlir::ValueRange{idxVal});
      mlir::Value dimIndex = dimVal;
      if (dimVal.getType() != rewriter.getIndexType()) {
        dimIndex = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), dimVal);
      }
      dynamicSizes.push_back(dimIndex);
    }
  }

  auto fillTensorType =
      mlir::RankedTensorType::get(rankedResType.getShape(), fillElemType);

  auto initTensor = mlir::tensor::EmptyOp::create(
      rewriter, loc, fillTensorType.getShape(), fillElemType, dynamicSizes);

  auto scalarValue =
      mlir::arith::ConstantOp::create(rewriter, loc, typedScalarAttr);

  auto fillOp =
      mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{scalarValue},
                                   mlir::ValueRange{initTensor});

  rewriter.replaceOp(op, fillOp.getResults());
  return mlir::success();
}

} // namespace onnx2mlir::dialect
