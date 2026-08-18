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
 * \file src/conversion/passes/onnx_to_linalg/resize.cpp
 * \brief ONNX Resize operation to Linalg lowering
 */

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

static inline mlir::Value getOptionalOperand(mlir::Operation *op,
                                             unsigned idx) {
  auto operand = op->getOperand(idx);
  if (mlir::isa<mlir::NoneType>(operand.getType()))
    return nullptr;

  return operand;
}

static mlir::Value
computeSourceCoordinate(mlir::OpBuilder &nest, mlir::Location loc,
                        mlir::Value outIdx, mlir::Value inDim,
                        mlir::Value outDim, mlir::Value scale,
                        llvm::StringRef coordTransMode) {
  mlir::FloatType f32Type = nest.getF32Type();

  // convert index -> i64 -> f32
  auto castIndexToF32 = [&](mlir::Value indexVal) -> mlir::Value {
    auto i64Val = mlir::arith::IndexCastOp::create(nest, loc, nest.getI64Type(),
                                                   indexVal);
    return mlir::arith::SIToFPOp::create(nest, loc, f32Type, i64Val);
  };

  auto outIdxF32 = castIndexToF32(outIdx);
  auto inDimF32 = castIndexToF32(inDim);
  auto outDimF32 = castIndexToF32(outDim);

  if (coordTransMode == "asymmetric") {
    return mlir::arith::DivFOp::create(nest, loc, outIdxF32, scale);
  }

  if (coordTransMode == "align_corners") {
    auto one = mlir::arith::ConstantFloatOp::create(nest, loc, f32Type,
                                                    llvm::APFloat(1.0f));
    auto inMinusOne = mlir::arith::SubFOp::create(nest, loc, inDimF32, one);
    auto outMinusOne = mlir::arith::SubFOp::create(nest, loc, outDimF32, one);

    auto zeroFloat = mlir::arith::ConstantFloatOp::create(nest, loc, f32Type,
                                                          llvm::APFloat(0.0f));
    auto isOne = mlir::arith::CmpFOp::create(
        nest, loc, mlir::arith::CmpFPredicate::OEQ, outMinusOne, zeroFloat);

    auto scaled = mlir::arith::MulFOp::create(nest, loc, outIdxF32, inMinusOne);
    auto divResult =
        mlir::arith::DivFOp::create(nest, loc, scaled, outMinusOne);

    return mlir::arith::SelectOp::create(nest, loc, isOne, zeroFloat,
                                         divResult);
  }

  if (coordTransMode == "tf_half_pixel_for_nn") {
    auto half = mlir::arith::ConstantFloatOp::create(nest, loc, f32Type,
                                                     llvm::APFloat(0.5f));
    auto idxPlusHalf = mlir::arith::AddFOp::create(nest, loc, outIdxF32, half);

    return mlir::arith::DivFOp::create(nest, loc, idxPlusHalf, scale);
  }

  if (coordTransMode == "pytorch_half_pixel") {
    auto half = mlir::arith::ConstantFloatOp::create(nest, loc, f32Type,
                                                     llvm::APFloat(0.5f));
    auto zero = mlir::arith::ConstantFloatOp::create(nest, loc, f32Type,
                                                     llvm::APFloat(0.0f));
    auto one = mlir::arith::ConstantFloatOp::create(nest, loc, f32Type,
                                                    llvm::APFloat(1.0f));

    auto isOutDimGtOne = mlir::arith::CmpFOp::create(
        nest, loc, mlir::arith::CmpFPredicate::OGT, outDimF32, one);

    auto idxPlusHalf = mlir::arith::AddFOp::create(nest, loc, outIdxF32, half);
    auto divResult = mlir::arith::DivFOp::create(nest, loc, idxPlusHalf, scale);
    auto subHalf = mlir::arith::SubFOp::create(nest, loc, divResult, half);

    return mlir::arith::SelectOp::create(nest, loc, isOutDimGtOne, subHalf,
                                         zero);
  }

  // default: "half_pixel"
  auto half = mlir::arith::ConstantFloatOp::create(nest, loc, f32Type,
                                                   llvm::APFloat(0.5f));
  auto idxPlusHalf = mlir::arith::AddFOp::create(nest, loc, outIdxF32, half);
  auto divResult = mlir::arith::DivFOp::create(nest, loc, idxPlusHalf, scale);

  return mlir::arith::SubFOp::create(nest, loc, divResult, half);
}

static mlir::Value computeNearestIndex(mlir::OpBuilder &nest,
                                       mlir::Location loc, mlir::Value coord,
                                       mlir::Value inDimIndex,
                                       llvm::StringRef nearestMode) {
  mlir::Value rounded;

  if (nearestMode == "floor") {
    rounded = mlir::math::FloorOp::create(nest, loc, coord);
  } else if (nearestMode == "ceil") {
    rounded = mlir::math::CeilOp::create(nest, loc, coord);
  } else if (nearestMode == "round_prefer_ceil") {
    auto half = mlir::arith::ConstantFloatOp::create(
        nest, loc, nest.getF32Type(), llvm::APFloat(0.5f));
    auto shifted = mlir::arith::AddFOp::create(nest, loc, coord, half);
    rounded = mlir::math::FloorOp::create(nest, loc, shifted);
  } else {
    // default: "round_prefer_floor" (ceil(coord - 0.5))
    auto half = mlir::arith::ConstantFloatOp::create(
        nest, loc, nest.getF32Type(), llvm::APFloat(0.5f));
    auto shifted = mlir::arith::SubFOp::create(nest, loc, coord, half);
    rounded = mlir::math::CeilOp::create(nest, loc, shifted);
  }

  auto idxI64 =
      mlir::arith::FPToSIOp::create(nest, loc, nest.getI64Type(), rounded);
  auto inDimI64 = mlir::arith::IndexCastOp::create(nest, loc, nest.getI64Type(),
                                                   inDimIndex);

  // clamp index to [0, inDim - 1]
  auto zeroI64 = mlir::arith::ConstantIntOp::create(nest, loc, 0, 64);
  auto oneI64 = mlir::arith::ConstantIntOp::create(nest, loc, 1, 64);
  auto maxI64 = mlir::arith::SubIOp::create(nest, loc, inDimI64, oneI64);

  auto clampLow = mlir::arith::MaxSIOp::create(nest, loc, idxI64, zeroI64);
  auto clampHigh = mlir::arith::MinSIOp::create(nest, loc, clampLow, maxI64);

  return mlir::arith::IndexCastOp::create(nest, loc, nest.getIndexType(),
                                          clampHigh);
}

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_ResizeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));

  mlir::Value opInputScales;
  mlir::Value opInputSizes;
  if (opName.contains("V10")) {
    opInputScales = getOptionalOperand(op, 1);
  } else {
    opInputScales = getOptionalOperand(op, 2);
    opInputSizes = getOptionalOperand(op, 3);
  }

  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  int64_t inputRank = inpDatType.getRank();

  /*
   * Attributes
   */

  // mode
  llvm::StringRef attr_mode = "nearest";
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("mode"))
    attr_mode = attr.getValue();

  // coordinate_transformation_mode
  llvm::StringRef attr_coord_trans_mode = "half_pixel";
  if (auto attr =
          op->getAttrOfType<mlir::StringAttr>("coordinate_transformation_mode"))
    attr_coord_trans_mode = attr.getValue();

  // nearest_mode
  llvm::StringRef attr_nearest_mode = "round_prefer_floor";
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("nearest_mode"))
    attr_nearest_mode = attr.getValue();

  /*
   * Affine mappings
   */

  mlir::AffineMap outputMap = rewriter.getMultiDimIdentityMap(inputRank);
  llvm::SmallVector<mlir::AffineMap, 2> indexingMaps = {outputMap};

  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      inputRank, mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  llvm::SmallVector<mlir::Value> outputDynDims;
  llvm::SmallVector<mlir::Value> outDims(inputRank);
  llvm::SmallVector<mlir::Value> inDims(inputRank);
  llvm::SmallVector<mlir::Value> axisScales(inputRank);

  for (int64_t i = 0; i < inputRank; ++i) {
    inDims[i] = mlir::tensor::DimOp::create(rewriter, loc, opInput, i);

    if (outDatType.isDynamicDim(i)) {
      mlir::Value outDimIdx;
      if (opInputSizes) {
        auto idxVal = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
        auto extractedSize = mlir::tensor::ExtractOp::create(
            rewriter, loc, opInputSizes, mlir::ValueRange{idxVal});
        outDimIdx = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), extractedSize);
      } else if (outDatType.hasStaticShape()) {
        outDimIdx = mlir::arith::ConstantIndexOp::create(
            rewriter, loc, outDatType.getDimSize(i));
      } else {
        outDimIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      }
      outDims[i] = outDimIdx;
      outputDynDims.push_back(outDimIdx);
    } else {
      outDims[i] = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, outDatType.getDimSize(i));
    }

    if (opInputScales) {
      auto idxVal = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
      auto extractedScale = mlir::tensor::ExtractOp::create(
          rewriter, loc, opInputScales, mlir::ValueRange{idxVal});
      if (mlir::isa<mlir::Float64Type>(extractedScale.getType())) {
        axisScales[i] = mlir::arith::TruncFOp::create(
            rewriter, loc, rewriter.getF32Type(), extractedScale);
      } else {
        axisScales[i] = extractedScale;
      }
    } else {
      auto inI64 = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), inDims[i]);
      auto inF32 = mlir::arith::SIToFPOp::create(rewriter, loc,
                                                 rewriter.getF32Type(), inI64);
      auto outI64 = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), outDims[i]);
      auto outF32 = mlir::arith::SIToFPOp::create(
          rewriter, loc, rewriter.getF32Type(), outI64);
      axisScales[i] = mlir::arith::DivFOp::create(rewriter, loc, outF32, inF32);
    }
  }

  auto outBuffer =
      mlir::tensor::EmptyOp::create(rewriter, loc, outDatType.getShape(),
                                    outDatType.getElementType(), outputDynDims);

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps, /*inter_types*/ iteratorTypes,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        llvm::SmallVector<mlir::Value> inpIndices;

        for (int64_t d = 0; d < inputRank; ++d) {
          auto outIdx = mlir::linalg::IndexOp::create(nest, nloc, d);
          auto srcCoord =
              computeSourceCoordinate(nest, nloc, outIdx, inDims[d], outDims[d],
                                      axisScales[d], attr_coord_trans_mode);

          if (attr_mode == "nearest") {
            auto nearestIdx = computeNearestIndex(nest, nloc, srcCoord,
                                                  inDims[d], attr_nearest_mode);
            inpIndices.push_back(nearestIdx);
          } else {
            auto nearestIdx =
                computeNearestIndex(nest, nloc, srcCoord, inDims[d], "floor");
            inpIndices.push_back(nearestIdx);
          }
        }

        auto sampledVal =
            mlir::tensor::ExtractOp::create(nest, nloc, opInput, inpIndices);
        mlir::linalg::YieldOp::create(nest, nloc, sampledVal.getResult());
      });

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
