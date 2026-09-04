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
 * \file src/conversion/passes/onnx_to_linalg/pad.cpp
 * \brief ONNX Pad operation to Linalg lowering
 */

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_PadOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                   const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opInpPads = (op->getNumOperands() > 1 &&
                    !mlir::isa<mlir::NoneType>(op->getOperand(1).getType()))
                       ? convRewriter.getRemappedValue(op->getOperand(1))
                       : nullptr;
  auto opInpConst = (op->getNumOperands() > 2 &&
                     !mlir::isa<mlir::NoneType>(op->getOperand(2).getType()))
                        ? convRewriter.getRemappedValue(op->getOperand(2))
                        : nullptr;
  auto opInpAxes = (op->getNumOperands() > 3 &&
                    !mlir::isa<mlir::NoneType>(op->getOperand(3).getType()))
                       ? convRewriter.getRemappedValue(op->getOperand(3))
                       : nullptr;
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto inpElmType = inpDatType.getElementType();

  int64_t inputRank = inpDatType.getRank();

  /*
   * Attributes
   */

  // mode
  llvm::StringRef attr_mode = "constant";
  if (auto modeAttr = op->getAttrOfType<mlir::StringAttr>("mode"))
    attr_mode = modeAttr.getValue();

  // paddings / pads
  mlir::ArrayAttr attr_pads;
  if (op->hasAttr("paddings"))
    attr_pads = op->getAttrOfType<mlir::ArrayAttr>("paddings");
  if (op->hasAttr("pads"))
    attr_pads = op->getAttrOfType<mlir::ArrayAttr>("pads");

  // value
  int64_t attr_const_int = 0;
  double attr_const_float = 0.0;
  if (auto valAttr = op->getAttrOfType<mlir::FloatAttr>("value"))
    attr_const_float = valAttr.getValueAsDouble();
  else if (auto intAttr = op->getAttrOfType<mlir::IntegerAttr>("value"))
    attr_const_int = intAttr.getInt();

  /*
   *  Affine mappings
   */

  auto outputMap = rewriter.getMultiDimIdentityMap(inputRank);
  llvm::SmallVector<mlir::AffineMap, 1> indexingMaps = {outputMap};

  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      inputRank, mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  auto zeroIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

  llvm::SmallVector<mlir::Value> lowPadVals(inputRank, zeroIdx);
  llvm::SmallVector<mlir::Value> highPadVals(inputRank, zeroIdx);

  auto extractPadValue = [&](mlir::Value tensorVal,
                             int64_t idx) -> mlir::Value {
    auto cIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, idx);
    auto extractedVal = rewriter.createOrFold<mlir::tensor::ExtractOp>(
        loc, tensorVal, mlir::ValueRange{cIdx});
    if (mlir::isa<mlir::IndexType>(extractedVal.getType()))
      return extractedVal;
    return mlir::arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), extractedVal);
  };

  // retrieve constant value
  auto getConstantIntValueFromIR =
      [](mlir::Value val) -> std::optional<int64_t> {
    // infer via constant
    if (auto constVal = mlir::getConstantIntValue(val))
      return constVal;
    // infer via cast
    if (auto castOp = val.getDefiningOp<mlir::arith::IndexCastOp>()) {
      return mlir::getConstantIntValue(castOp.getIn());
    }
    // infer in pure IR
    return std::nullopt;
  };

  if (opInpPads) {
    // use operands
    auto padDatType =
        mlir::dyn_cast<mlir::RankedTensorType>(opInpPads.getType());
    int64_t totalPads =
        padDatType ? padDatType.getNumElements() : (2 * inputRank);
    int64_t numAxes =
        opInpAxes ? mlir::cast<mlir::RankedTensorType>(opInpAxes.getType())
                        .getNumElements()
                  : 0;

    if (numAxes > 0) {
      for (int64_t i = 0; i < numAxes; ++i) {
        auto axisVal = extractPadValue(opInpAxes, i);
        auto lowVal = extractPadValue(opInpPads, i);
        auto highVal = extractPadValue(opInpPads, i + numAxes);
        if (auto axisConst = getConstantIntValueFromIR(axisVal)) {
          int64_t ax = *axisConst;
          if (ax < 0)
            ax += inputRank;
          if (ax >= 0 && ax < inputRank) {
            lowPadVals[ax] = lowVal;
            highPadVals[ax] = highVal;
          }
        }
      }
    } else {
      int64_t half = totalPads / 2;
      for (int64_t i = 0; i < inputRank && i < half; ++i) {
        lowPadVals[i] = extractPadValue(opInpPads, i);
        highPadVals[i] = extractPadValue(opInpPads, i + half);
      }
    }
  } else {
    // use attributes
    if (attr_pads) {
      int64_t totalPads = attr_pads.size();
      int64_t half = totalPads / 2;
      for (int64_t i = 0; i < inputRank && i < half; ++i) {
        int64_t lowPad = mlir::cast<mlir::IntegerAttr>(attr_pads[i]).getInt();
        int64_t highPad =
            mlir::cast<mlir::IntegerAttr>(attr_pads[i + half]).getInt();
        lowPadVals[i] =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, lowPad);
        highPadVals[i] =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, highPad);
      }
    }
  }

  // compute out shape and dynamic dims from IR
  llvm::SmallVector<mlir::Value> outputDynDims;
  llvm::SmallVector<int64_t> outShape(inputRank);

  if (outDatType && outDatType.hasStaticShape()) {
    for (int64_t i = 0; i < inputRank; ++i) {
      outShape[i] = outDatType.getDimSize(i);
    }
  } else {
    for (int64_t i = 0; i < inputRank; ++i) {
      int64_t staticDim = inpDatType.getDimSize(i);
      auto lowConst = getConstantIntValueFromIR(lowPadVals[i]);
      auto highConst = getConstantIntValueFromIR(highPadVals[i]);

      if (!mlir::ShapedType::isDynamic(staticDim) && lowConst && highConst) {
        outShape[i] = staticDim + *lowConst + *highConst;
      } else {
        outShape[i] = mlir::ShapedType::kDynamic;
        auto dimVal =
            mlir::ShapedType::isDynamic(staticDim)
                ? mlir::tensor::DimOp::create(rewriter, loc, opInput, i)
                      .getResult()
                : mlir::arith::ConstantIndexOp::create(rewriter, loc, staticDim)
                      .getResult();
        auto addLow =
            mlir::arith::AddIOp::create(rewriter, loc, dimVal, lowPadVals[i]);
        auto addHigh =
            mlir::arith::AddIOp::create(rewriter, loc, addLow, highPadVals[i]);
        outputDynDims.push_back(addHigh);
      }
    }
  }

  // pad values
  mlir::Value padVal;
  if (opInpConst) {
    padVal = opInpConst;
  } else {
    if (inpElmType.isFloat()) {
      padVal = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(inpElmType, attr_const_float));
    } else if (inpElmType.isInteger()) {
      padVal = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(inpElmType, attr_const_int));
    } else {
      padVal = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(inpElmType));
    }
  }

  if (attr_mode == "constant") {
    llvm::SmallVector<mlir::OpFoldResult> lowPadsOfr;
    llvm::SmallVector<mlir::OpFoldResult> highPadsOfr;
    for (int64_t i = 0; i < inputRank; ++i) {
      if (auto constInt = getConstantIntValueFromIR(lowPadVals[i]))
        lowPadsOfr.push_back(rewriter.getIndexAttr(*constInt));
      else
        lowPadsOfr.push_back(lowPadVals[i]);

      if (auto constInt = getConstantIntValueFromIR(highPadVals[i]))
        highPadsOfr.push_back(rewriter.getIndexAttr(*constInt));
      else
        highPadsOfr.push_back(highPadVals[i]);
    }

    auto padOp = mlir::tensor::PadOp::create(rewriter, loc, outDatType, opInput,
                                             lowPadsOfr, highPadsOfr, padVal);
    rewriter.replaceOp(op, padOp);

    return mlir::success();
  }

  auto outBuffer = mlir::tensor::EmptyOp::create(rewriter, loc, outShape,
                                                 inpElmType, outputDynDims);

  auto genericOpBuilder = [&](/*op_builder*/ mlir::OpBuilder &nest,
                              /*src_location*/ mlir::Location nloc,
                              /*value_args*/ mlir::ValueRange args) {
    llvm::SmallVector<mlir::Value> mappedIndices;

    for (int64_t i = 0; i < inputRank; ++i) {
      auto outIdx = mlir::linalg::IndexOp::create(nest, nloc, i);

      mlir::Value dimSize;
      int64_t staticDim = inpDatType.getDimSize(i);
      if (mlir::ShapedType::isDynamic(staticDim)) {
        dimSize = mlir::tensor::DimOp::create(nest, nloc, opInput, i);
      } else {
        dimSize = mlir::arith::ConstantIndexOp::create(nest, nloc, staticDim);
      }

      auto lowPadVal = lowPadVals[i];
      auto origIdx = mlir::arith::SubIOp::create(nest, nloc, outIdx, lowPadVal);

      if (attr_mode == "edge") {
        auto zero = mlir::arith::ConstantIndexOp::create(nest, nloc, 0);
        auto one = mlir::arith::ConstantIndexOp::create(nest, nloc, 1);
        auto maxIdx = mlir::arith::SubIOp::create(nest, nloc, dimSize, one);
        auto geZero = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sge, origIdx, zero);
        auto clampedLow =
            mlir::arith::SelectOp::create(nest, nloc, geZero, origIdx, zero);
        auto leMax = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sle, clampedLow, maxIdx);
        auto clamped = mlir::arith::SelectOp::create(nest, nloc, leMax,
                                                     clampedLow, maxIdx);

        mappedIndices.push_back(clamped);

      } else if (attr_mode == "reflect") {
        auto zero = mlir::arith::ConstantIndexOp::create(nest, nloc, 0);
        auto one = mlir::arith::ConstantIndexOp::create(nest, nloc, 1);
        auto two = mlir::arith::ConstantIndexOp::create(nest, nloc, 2);

        auto cmpEQ1 = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::eq, dimSize, one);
        auto twoDim = mlir::arith::MulIOp::create(nest, nloc, dimSize, two);
        auto period = mlir::arith::SubIOp::create(nest, nloc, twoDim, two);

        auto rem = mlir::arith::RemSIOp::create(nest, nloc, origIdx, period);
        auto remLtZero = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::slt, rem, zero);
        auto remPlusP = mlir::arith::AddIOp::create(nest, nloc, rem, period);
        auto xMod =
            mlir::arith::SelectOp::create(nest, nloc, remLtZero, remPlusP, rem);

        auto geN = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sge, xMod, dimSize);
        auto pSubX = mlir::arith::SubIOp::create(nest, nloc, period, xMod);
        auto reflected =
            mlir::arith::SelectOp::create(nest, nloc, geN, pSubX, xMod);

        auto finalIdx =
            mlir::arith::SelectOp::create(nest, nloc, cmpEQ1, zero, reflected);

        mappedIndices.push_back(finalIdx);

      } else if (attr_mode == "wrap") {
        auto zero = mlir::arith::ConstantIndexOp::create(nest, nloc, 0);
        auto rem = mlir::arith::RemSIOp::create(nest, nloc, origIdx, dimSize);
        auto remLtZero = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::slt, rem, zero);
        auto remPlusN = mlir::arith::AddIOp::create(nest, nloc, rem, dimSize);
        auto wrappedIdx =
            mlir::arith::SelectOp::create(nest, nloc, remLtZero, remPlusN, rem);
        mappedIndices.push_back(wrappedIdx);

      } else {
        mappedIndices.push_back(origIdx);
      }
    }

    auto extracted =
        mlir::tensor::ExtractOp::create(nest, nloc, opInput, mappedIndices);

    mlir::linalg::YieldOp::create(nest, nloc, extracted.getResult());
  };

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iterator_types*/ iteratorTypes,
      /*builder_callback*/ genericOpBuilder);

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
