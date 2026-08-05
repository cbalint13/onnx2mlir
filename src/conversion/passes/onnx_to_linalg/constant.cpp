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
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opResult = op->getResult(0);
  auto opOutput = convRewriter.getRemappedValue(opResult);

  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());
  auto dstDatType = mlir::dyn_cast<mlir::RankedTensorType>(opResult.getType());

  // checks
  if (mlir::isa<mlir::NoneType>(outDatType))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " with 'NoneType' is not supported";

  /*
   * Attributes
   */

  // value
  auto valueAttr = op->getAttr("value");
  auto typedAttr = mlir::dyn_cast_or_null<mlir::TypedAttr>(valueAttr);
  if (!typedAttr)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " without a valid tensor 'value' attribute";
  if (typedAttr.getType() != dstDatType)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " 'value' attribute type does not match result type";

  bool isChanged = false;
  // convert value type to result type
  if (outDatType != typedAttr.getType()) {
    isChanged = true;
    typedAttr = changeAttrType(valueAttr, outDatType);
  }

  /*
   *  Linalg ops staging
   */

  auto constOp =
      mlir::arith::ConstantOp::create(rewriter, loc, outDatType, typedAttr);

  if (isChanged)
    constOp->setAttr("onnx_value", valueAttr);

  rewriter.replaceOp(op, constOp);

  return mlir::success();
}

mlir::LogicalResult
OnnxToLinalg_ConstantOfShapeOp(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter,
                               const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  // checks
  if (mlir::isa<mlir::NoneType>(outDatType))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " with 'NoneType' is not supported";

  /*
   * Attributes
   */

  // value
  mlir::Attribute valueAttr = op->getAttr("value");
  mlir::TypedAttr typScalarAttr;
  if (auto denseAttr =
          mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(valueAttr)) {
    if (!denseAttr.empty())
      typScalarAttr = *denseAttr.value_begin<mlir::TypedAttr>();
  } else {
    typScalarAttr = mlir::dyn_cast_or_null<mlir::TypedAttr>(valueAttr);
  }
  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(typScalarAttr)) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(intAttr.getType());
        intType && !intType.isSignless()) {
      typScalarAttr = rewriter.getIntegerAttr(
          rewriter.getIntegerType(intType.getWidth()), intAttr.getValue());
    }
  }
  if (!typScalarAttr)
    typScalarAttr = rewriter.getFloatAttr(rewriter.getF32Type(), 0.0);

  /*
   *  Linalg ops staging
   */

  mlir::SmallVector<mlir::Value, 4> dynamicSizes;

  for (int64_t i = 0; i < outDatType.getRank(); ++i) {
    if (outDatType.isDynamicDim(i)) {
      auto idxVal = mlir::arith::ConstantOp::create(rewriter, loc,
                                                    rewriter.getIndexAttr(i));
      auto dimVal = mlir::tensor::ExtractOp::create(rewriter, loc, opInput,
                                                    mlir::ValueRange{idxVal});
      mlir::Value dimIndex = dimVal;
      if (dimVal.getType() != rewriter.getIndexType()) {
        dimIndex = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), dimVal);
      }
      dynamicSizes.push_back(dimIndex);
    }
  }

  auto fillTensorType = mlir::RankedTensorType::get(outDatType.getShape(),
                                                    typScalarAttr.getType());
  auto initTensor =
      mlir::tensor::EmptyOp::create(rewriter, loc, fillTensorType.getShape(),
                                    typScalarAttr.getType(), dynamicSizes);
  auto scalarValue =
      mlir::arith::ConstantOp::create(rewriter, loc, typScalarAttr);
  auto fillOp =
      mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{scalarValue},
                                   mlir::ValueRange{initTensor});

  fillOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, fillOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
