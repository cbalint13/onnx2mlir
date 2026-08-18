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
 * \file src/conversion/passes/onnx_to_linalg/clip.cpp
 * \brief ONNX Clip operation to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <cfloat>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

static mlir::TypedAttr getAttrScalar(mlir::PatternRewriter &rewriter,
                                     mlir::Attribute valAttr,
                                     mlir::Type inpElmType) {
  if (!valAttr)
    return nullptr;

  double floatVal = 0.0;
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(valAttr)) {
    floatVal = floatAttr.getValueAsDouble();
  } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(valAttr)) {
    floatVal = static_cast<double>(intAttr.getInt());
  } else {
    return nullptr;
  }

  mlir::TypedAttr attr;
  if (inpElmType.isFloat()) {
    attr = rewriter.getFloatAttr(inpElmType, floatVal);
  } else if (inpElmType.isInteger()) {
    attr = rewriter.getIntegerAttr(inpElmType, static_cast<int64_t>(floatVal));
  } else {
    return nullptr;
  }

  return attr;
}

mlir::LogicalResult
OnnxToLinalg_ClipOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));
  auto opInpMin = (op->getNumOperands() > 1 &&
                   !mlir::isa<mlir::NoneType>(op->getOperand(1).getType()))
                      ? op->getOperand(1)
                      : nullptr;
  auto opInpMax = (op->getNumOperands() > 2 &&
                   !mlir::isa<mlir::NoneType>(op->getOperand(2).getType()))
                      ? op->getOperand(2)
                      : nullptr;

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto srcDatType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());

  auto minDatType =
      opInpMin ? mlir::dyn_cast<mlir::RankedTensorType>(opInpMin.getType())
               : nullptr;
  auto maxDatType =
      opInpMax ? mlir::dyn_cast<mlir::RankedTensorType>(opInpMax.getType())
               : nullptr;

  // linalg type conversion
  if (opInpMin)
    opInpMin = convRewriter.getRemappedValue(op->getOperand(1));
  if (opInpMax)
    opInpMax = convRewriter.getRemappedValue(op->getOperand(2));

  // onnx original data types
  auto srcElmType = srcDatType.getElementType();
  // linalg converted data types (signless)
  auto inpElmType = inpDatType.getElementType();

  // value checks
  if (mlir::dyn_cast<mlir::RankedTensorType>(inpDatType).getShape() !=
      mlir::dyn_cast<mlir::RankedTensorType>(outDatType).getShape())
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " input and output shapes are different";
  if (opInpMin && (!minDatType || minDatType.getRank() != 0))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " operand 'min' should be a tensor of rank zero";
  if (opInpMax && (!maxDatType || maxDatType.getRank() != 0))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " operand 'min' should be a tensor of rank zero";
  if (minDatType && minDatType.getElementType() != srcElmType)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " operand 'min' and input element types are different";
  if (maxDatType && maxDatType.getElementType() != srcElmType)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " operand 'max' and input element types are different";

  /*
   * Attributes
   */

  // min
  auto minAttr = op->getAttr("min");
  auto minScalar = getAttrScalar(rewriter, minAttr, inpElmType);

  // max
  auto maxAttr = op->getAttr("max");
  auto maxScalar = getAttrScalar(rewriter, maxAttr, inpElmType);

  // no clipping identity
  if (!minAttr && !maxAttr && !opInpMin && !opInpMax) {
    rewriter.replaceOp(op, opInput);
    return mlir::success();
  }

  /*
   *  Affine mappings
   */

  auto inpIdentityMap = rewriter.getMultiDimIdentityMap(inpDatType.getRank());
  auto outIdentityMap = rewriter.getMultiDimIdentityMap(outDatType.getRank());

  mlir::SmallVector<mlir::AffineMap, 2> indexingMaps;
  indexingMaps = {inpIdentityMap, outIdentityMap};

  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
      inpDatType.getRank(), mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  auto outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, loc, outDatType.getShape(), outDatType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInput},
      /*output_values=*/mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      /*builder_callback=*/
      [&](/*op_builder*/ mlir::OpBuilder nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::Value scalar;
        mlir::Value val = args[0];

        if (opInpMin || minScalar) {
          if (opInpMin)
            scalar = mlir::tensor::ExtractOp::create(nest, nloc, opInpMin,
                                                     mlir::ValueRange{});
          else
            scalar = mlir::arith::ConstantOp::create(rewriter, loc, minScalar);
          if (srcElmType.isFloat())
            val = mlir::arith::MaximumFOp::create(nest, nloc, val, scalar);
          else if (srcElmType.isUnsignedInteger())
            val = mlir::arith::MaxUIOp::create(nest, nloc, val, scalar);
          else
            val = mlir::arith::MaxSIOp::create(nest, nloc, val, scalar);
        }

        if (opInpMax || maxScalar) {
          if (opInpMax)
            scalar = mlir::tensor::ExtractOp::create(nest, nloc, opInpMax,
                                                     mlir::ValueRange{});
          else
            scalar = mlir::arith::ConstantOp::create(rewriter, loc, maxScalar);
          if (srcElmType.isFloat())
            val = mlir::arith::MinimumFOp::create(nest, nloc, val, scalar);
          else if (srcElmType.isUnsignedInteger())
            val = mlir::arith::MinUIOp::create(nest, nloc, val, scalar);
          else
            val = mlir::arith::MinSIOp::create(nest, nloc, val, scalar);
        }

        mlir::linalg::YieldOp::create(nest, nloc, val);
      });

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
