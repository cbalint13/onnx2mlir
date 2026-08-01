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

/// Helper to safely retrieve an optional operand
static inline mlir::Value getOptionalOperand(mlir::Operation *op,
                                             unsigned idx) {
  mlir::Value operand = op->getOperand(idx);
  if (mlir::isa<mlir::NoneType>(operand.getType()))
    return nullptr;

  return operand;
}

/// Helper to check if a tensor operand is present and non-empty.
static bool isNonEmptyTensor(mlir::Value val) {
  if (!val)
    return false;
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!tensorType || tensorType.getRank() == 0)
    return false;
  if (tensorType.hasStaticShape() && tensorType.getNumElements() == 0) {
    return false;
  }
  return true;
}

/// Computes the transformed input floating-point index for a given axis.
static mlir::Value
computeSourceCoordinate(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::Value outIdx, mlir::Value inDim,
                        mlir::Value outDim, mlir::Value scale,
                        llvm::StringRef coordTransMode) {
  mlir::FloatType f32Type = b.getF32Type();

  // Helper lambda to safely convert index -> i64 -> f32
  auto castIndexToF32 = [&](mlir::Value indexVal) -> mlir::Value {
    mlir::Value i64Val =
        mlir::arith::IndexCastOp::create(b, loc, b.getI64Type(), indexVal);
    return mlir::arith::SIToFPOp::create(b, loc, f32Type, i64Val);
  };

  mlir::Value outIdxF32 = castIndexToF32(outIdx);
  mlir::Value inDimF32 = castIndexToF32(inDim);
  mlir::Value outDimF32 = castIndexToF32(outDim);

  if (coordTransMode == "asymmetric") {
    return mlir::arith::DivFOp::create(b, loc, outIdxF32, scale);
  }

  if (coordTransMode == "align_corners") {
    mlir::Value one = mlir::arith::ConstantFloatOp::create(b, loc, f32Type,
                                                           llvm::APFloat(1.0f));
    mlir::Value inMinusOne = mlir::arith::SubFOp::create(b, loc, inDimF32, one);
    mlir::Value outMinusOne =
        mlir::arith::SubFOp::create(b, loc, outDimF32, one);

    mlir::Value zeroFloat = mlir::arith::ConstantFloatOp::create(
        b, loc, f32Type, llvm::APFloat(0.0f));
    mlir::Value isOne = mlir::arith::CmpFOp::create(
        b, loc, mlir::arith::CmpFPredicate::OEQ, outMinusOne, zeroFloat);

    mlir::Value scaled =
        mlir::arith::MulFOp::create(b, loc, outIdxF32, inMinusOne);
    mlir::Value divResult =
        mlir::arith::DivFOp::create(b, loc, scaled, outMinusOne);

    return mlir::arith::SelectOp::create(b, loc, isOne, zeroFloat, divResult);
  }

  if (coordTransMode == "tf_half_pixel_for_nn") {
    mlir::Value half = mlir::arith::ConstantFloatOp::create(
        b, loc, f32Type, llvm::APFloat(0.5f));
    mlir::Value idxPlusHalf =
        mlir::arith::AddFOp::create(b, loc, outIdxF32, half);
    return mlir::arith::DivFOp::create(b, loc, idxPlusHalf, scale);
  }

  if (coordTransMode == "pytorch_half_pixel") {
    mlir::Value half = mlir::arith::ConstantFloatOp::create(
        b, loc, f32Type, llvm::APFloat(0.5f));
    mlir::Value zero = mlir::arith::ConstantFloatOp::create(
        b, loc, f32Type, llvm::APFloat(0.0f));
    mlir::Value one = mlir::arith::ConstantFloatOp::create(b, loc, f32Type,
                                                           llvm::APFloat(1.0f));

    mlir::Value isOutDimGtOne = mlir::arith::CmpFOp::create(
        b, loc, mlir::arith::CmpFPredicate::OGT, outDimF32, one);

    mlir::Value idxPlusHalf =
        mlir::arith::AddFOp::create(b, loc, outIdxF32, half);
    mlir::Value divResult =
        mlir::arith::DivFOp::create(b, loc, idxPlusHalf, scale);
    mlir::Value subHalf = mlir::arith::SubFOp::create(b, loc, divResult, half);

    return mlir::arith::SelectOp::create(b, loc, isOutDimGtOne, subHalf, zero);
  }

  // Default: "half_pixel"
  mlir::Value half = mlir::arith::ConstantFloatOp::create(b, loc, f32Type,
                                                          llvm::APFloat(0.5f));
  mlir::Value idxPlusHalf =
      mlir::arith::AddFOp::create(b, loc, outIdxF32, half);
  mlir::Value divResult =
      mlir::arith::DivFOp::create(b, loc, idxPlusHalf, scale);
  return mlir::arith::SubFOp::create(b, loc, divResult, half);
}

static mlir::Value computeNearestIndex(mlir::OpBuilder &b, mlir::Location loc,
                                       mlir::Value coord,
                                       mlir::Value inDimIndex,
                                       llvm::StringRef nearestMode) {
  mlir::Value rounded;

  if (nearestMode == "floor") {
    rounded = mlir::math::FloorOp::create(b, loc, coord);
  } else if (nearestMode == "ceil") {
    rounded = mlir::math::CeilOp::create(b, loc, coord);
  } else if (nearestMode == "round_prefer_ceil") {
    mlir::Value half = mlir::arith::ConstantFloatOp::create(
        b, loc, b.getF32Type(), llvm::APFloat(0.5f));
    mlir::Value shifted = mlir::arith::AddFOp::create(b, loc, coord, half);
    rounded = mlir::math::FloorOp::create(b, loc, shifted);
  } else {
    // Default: "round_prefer_floor" (ceil(coord - 0.5))
    mlir::Value half = mlir::arith::ConstantFloatOp::create(
        b, loc, b.getF32Type(), llvm::APFloat(0.5f));
    mlir::Value shifted = mlir::arith::SubFOp::create(b, loc, coord, half);
    rounded = mlir::math::CeilOp::create(b, loc, shifted);
  }

  mlir::Value idxI64 =
      mlir::arith::FPToSIOp::create(b, loc, b.getI64Type(), rounded);
  mlir::Value inDimI64 =
      mlir::arith::IndexCastOp::create(b, loc, b.getI64Type(), inDimIndex);

  // Clamp index to [0, inDim - 1]
  mlir::Value zeroI64 = mlir::arith::ConstantIntOp::create(b, loc, 0, 64);
  mlir::Value oneI64 = mlir::arith::ConstantIntOp::create(b, loc, 1, 64);
  mlir::Value maxI64 = mlir::arith::SubIOp::create(b, loc, inDimI64, oneI64);

  mlir::Value clampedLow =
      mlir::arith::MaxSIOp::create(b, loc, idxI64, zeroI64);
  mlir::Value clampedHigh =
      mlir::arith::MinSIOp::create(b, loc, clampedLow, maxI64);

  return mlir::arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                          clampedHigh);
}

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_ResizeOp(mlir::Operation *op,
                                          mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value data = op->getOperand(0);
  mlir::Value result = op->getResult(0);
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());

  if (!resultType || !dataType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName +
                               " inputs and outputs must be ranked tensors");
  }

  int64_t rank = dataType.getRank();

  llvm::StringRef mode = "nearest";
  llvm::StringRef coordTransMode = "half_pixel";
  llvm::StringRef nearestMode = "round_prefer_floor";

  if (auto attr = op->getAttrOfType<mlir::StringAttr>("mode")) {
    mode = attr.getValue();
  }
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(
          "coordinate_transformation_mode")) {
    coordTransMode = attr.getValue();
  }
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("nearest_mode")) {
    nearestMode = attr.getValue();
  }

  mlir::Value scalesOperand;
  mlir::Value sizesOperand;

  if (opName.contains("V10")) {
    scalesOperand = getOptionalOperand(op, 1);
  } else {
    // Resize V11, V13, V18, V19
    scalesOperand = getOptionalOperand(op, 2);
    sizesOperand = getOptionalOperand(op, 3);
  }

  bool hasScales = isNonEmptyTensor(scalesOperand);
  bool hasSizes = isNonEmptyTensor(sizesOperand);

  llvm::SmallVector<mlir::Value> outputDynDims;
  llvm::SmallVector<mlir::Value> outDims(rank);
  llvm::SmallVector<mlir::Value> inDims(rank);
  llvm::SmallVector<mlir::Value> axisScales(rank);

  for (int64_t i = 0; i < rank; ++i) {
    inDims[i] = mlir::tensor::DimOp::create(rewriter, loc, data, i);

    if (resultType.isDynamicDim(i)) {
      mlir::Value outDimIdx;
      if (hasSizes) {
        mlir::Value idxVal =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
        mlir::Value extractedSize = mlir::tensor::ExtractOp::create(
            rewriter, loc, sizesOperand, mlir::ValueRange{idxVal});
        outDimIdx = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), extractedSize);
      } else if (resultType.hasStaticShape()) {
        outDimIdx = mlir::arith::ConstantIndexOp::create(
            rewriter, loc, resultType.getDimSize(i));
      } else {
        outDimIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      }
      outDims[i] = outDimIdx;
      outputDynDims.push_back(outDimIdx);
    } else {
      outDims[i] = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, resultType.getDimSize(i));
    }

    // Extract dynamic scales if provided or calculate from dimension ratio
    if (hasScales) {
      mlir::Value idxVal =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
      mlir::Value extractedScale = mlir::tensor::ExtractOp::create(
          rewriter, loc, scalesOperand, mlir::ValueRange{idxVal});
      if (mlir::isa<mlir::Float64Type>(extractedScale.getType())) {
        axisScales[i] = mlir::arith::TruncFOp::create(
            rewriter, loc, rewriter.getF32Type(), extractedScale);
      } else {
        axisScales[i] = extractedScale;
      }
    } else {
      mlir::Value inI64 = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), inDims[i]);
      mlir::Value inF32 = mlir::arith::SIToFPOp::create(
          rewriter, loc, rewriter.getF32Type(), inI64);
      mlir::Value outI64 = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), outDims[i]);
      mlir::Value outF32 = mlir::arith::SIToFPOp::create(
          rewriter, loc, rewriter.getF32Type(), outI64);
      axisScales[i] = mlir::arith::DivFOp::create(rewriter, loc, outF32, inF32);
    }
  }

  mlir::Value initTensor =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), outputDynDims);

  mlir::AffineMap outputMap = rewriter.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap, 2> indexingMaps = {outputMap};
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      rank, mlir::utils::IteratorType::parallel);

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/mlir::TypeRange{resultType},
      /*inputs=*/mlir::ValueRange{},
      /*outputs=*/mlir::ValueRange{initTensor}, indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::ValueRange args) {
        llvm::SmallVector<mlir::Value> inputIndices;

        for (int64_t d = 0; d < rank; ++d) {
          mlir::Value outIdx = mlir::linalg::IndexOp::create(b, nestedLoc, d);
          mlir::Value srcCoord = computeSourceCoordinate(
              b, nestedLoc, outIdx, inDims[d], outDims[d], axisScales[d],
              coordTransMode);

          if (mode == "nearest") {
            mlir::Value nearestIdx = computeNearestIndex(
                b, nestedLoc, srcCoord, inDims[d], nearestMode);
            inputIndices.push_back(nearestIdx);
          } else {
            // Fallback indexing for nearest modes
            mlir::Value nearestIdx =
                computeNearestIndex(b, nestedLoc, srcCoord, inDims[d], "floor");
            inputIndices.push_back(nearestIdx);
          }
        }

        mlir::Value sampledVal =
            mlir::tensor::ExtractOp::create(b, nestedLoc, data, inputIndices);
        mlir::linalg::YieldOp::create(b, nestedLoc, sampledVal);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp.getResult(0));
  return mlir::success();
}

} // namespace onnx2mlir::dialect
