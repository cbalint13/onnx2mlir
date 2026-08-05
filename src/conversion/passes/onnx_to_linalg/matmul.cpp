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
 * \file src/conversion/passes/onnx_to_linalg/matmul.cpp
 * \brief ONNX MatMul and MatMulInteger operation to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>

#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_MatMulOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  mlir::Value valA = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value valB = convRewriter.getRemappedValue(op->getOperand(1));
  mlir::Value res = op->getResult(0);

  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(valA.getType());
  auto bType = mlir::dyn_cast<mlir::RankedTensorType>(valB.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(
      typeConverter->convertType(res.getType()));

  auto orgResType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!aType || !bType || !resType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " inputs and result must be tensor types");
  }

  mlir::Type resElmType = resType.getElementType();
  mlir::Type orgElmType = orgResType.getElementType();
  bool isFloat = mlir::isa<mlir::FloatType>(orgElmType);

  // Check for optional zero-point operands (used by MatMulInteger)
  mlir::Value valAZp;
  mlir::Value valBZp;
  bool hasAZp = false;
  bool hasBZp = false;

  if (op->getNumOperands() > 2) {
    valAZp = convRewriter.getRemappedValue(op->getOperand(2));
    if (valAZp && !mlir::isa<mlir::NoneType>(valAZp.getType())) {
      hasAZp = true;
    }
  }

  if (op->getNumOperands() > 3) {
    valBZp = convRewriter.getRemappedValue(op->getOperand(3));
    if (valBZp && !mlir::isa<mlir::NoneType>(valBZp.getType())) {
      hasBZp = true;
    }
  }

  // Promote rank 0 tensor to minimum rank 1
  if (resType.getRank() == 0) {
    resType = mlir::RankedTensorType::get({1}, resElmType);
  }

  bool aIs1D = (aType.getRank() == 1);
  bool bIs1D = (bType.getRank() == 1);

  if (aIs1D) {
    llvm::SmallVector<mlir::ReassociationIndices, 1> reassoc = {{0, 1}};
    auto tgtType = mlir::RankedTensorType::get({1, aType.getDimSize(0)},
                                               aType.getElementType());
    valA = mlir::tensor::ExpandShapeOp::create(rewriter, loc, tgtType, valA,
                                               reassoc);
    aType = mlir::cast<mlir::RankedTensorType>(valA.getType());
  }

  if (bIs1D) {
    llvm::SmallVector<mlir::ReassociationIndices, 1> reassoc = {{0, 1}};
    auto tgtType = mlir::RankedTensorType::get({bType.getDimSize(0), 1},
                                               bType.getElementType());
    valB = mlir::tensor::ExpandShapeOp::create(rewriter, loc, tgtType, valB,
                                               reassoc);
    bType = mlir::cast<mlir::RankedTensorType>(valB.getType());
  }

  int64_t rA = aType.getRank();
  int64_t rB = bType.getRank();
  int64_t rOut = std::max(rA, rB);
  int64_t numBatchDims = rOut - 2;

  llvm::SmallVector<mlir::Value, 4> dynamicSizes;
  int64_t outDimIdx = 0;

  for (int64_t i = 0; i < numBatchDims; ++i) {
    if (resType.isDynamicDim(outDimIdx)) {
      int64_t aBatchIdx = i - (numBatchDims - (rA - 2));
      int64_t bBatchIdx = i - (numBatchDims - (rB - 2));
      if (aBatchIdx >= 0 && aType.isDynamicDim(aBatchIdx)) {
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, valA, aBatchIdx));
      } else if (bBatchIdx >= 0 && bType.isDynamicDim(bBatchIdx)) {
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, valB, bBatchIdx));
      }
    }
    outDimIdx++;
  }

  if (!aIs1D && outDimIdx < resType.getRank()) {
    if (resType.isDynamicDim(outDimIdx)) {
      dynamicSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, valA, rA - 2));
    }
    outDimIdx++;
  }

  if (!bIs1D && outDimIdx < resType.getRank()) {
    if (resType.isDynamicDim(outDimIdx)) {
      dynamicSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, valB, rB - 1));
    }
    outDimIdx++;
  }

  auto outTBuff =
      mlir::tensor::EmptyOp::create(rewriter, loc, resType, dynamicSizes);

  // Initialize output buffer with zero
  auto zeroAttr = rewriter.getZeroAttr(resType.getElementType());
  mlir::Value constZero = mlir::arith::ConstantOp::create(
      rewriter, loc, resType.getElementType(), zeroAttr);

  mlir::Value outBuff =
      mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{constZero},
                                   mlir::ValueRange{outTBuff.getResult()})
          ->getResult(0);

  mlir::SmallVector<mlir::utils::IteratorType> gemmIters;
  for (int64_t i = 0; i < numBatchDims + 2; ++i) {
    gemmIters.push_back(mlir::utils::IteratorType::parallel);
  }
  gemmIters.push_back(mlir::utils::IteratorType::reduction);

  mlir::AffineExpr mExpr = rewriter.getAffineDimExpr(numBatchDims);
  mlir::AffineExpr nExpr = rewriter.getAffineDimExpr(numBatchDims + 1);
  mlir::AffineExpr kExpr = rewriter.getAffineDimExpr(numBatchDims + 2);

  llvm::SmallVector<mlir::AffineExpr, 4> exprsA;
  for (int64_t i = 0; i < rA - 2; ++i) {
    int64_t outBatchIdx = numBatchDims - (rA - 2) + i;
    if (aType.getDimSize(i) == 1) {
      exprsA.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      exprsA.push_back(rewriter.getAffineDimExpr(outBatchIdx));
    }
  }
  exprsA.push_back(mExpr);
  exprsA.push_back(kExpr);
  mlir::AffineMap mapA =
      mlir::AffineMap::get(numBatchDims + 3, 0, exprsA, op->getContext());

  llvm::SmallVector<mlir::AffineExpr, 4> exprsB;
  for (int64_t j = 0; j < rB - 2; ++j) {
    int64_t outBatchIdx = numBatchDims - (rB - 2) + j;
    if (bType.getDimSize(j) == 1) {
      exprsB.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      exprsB.push_back(rewriter.getAffineDimExpr(outBatchIdx));
    }
  }
  exprsB.push_back(kExpr);
  exprsB.push_back(nExpr);
  mlir::AffineMap mapB =
      mlir::AffineMap::get(numBatchDims + 3, 0, exprsB, op->getContext());

  llvm::SmallVector<mlir::AffineExpr, 4> exprsOut;
  for (int64_t d = 0; d < numBatchDims; ++d) {
    exprsOut.push_back(rewriter.getAffineDimExpr(d));
  }
  if (!aIs1D) {
    exprsOut.push_back(mExpr);
  }
  if (!bIs1D) {
    exprsOut.push_back(nExpr);
  }
  while (exprsOut.size() < static_cast<size_t>(resType.getRank())) {
    exprsOut.push_back(rewriter.getAffineConstantExpr(0));
  }
  if (exprsOut.size() > static_cast<size_t>(resType.getRank())) {
    exprsOut.resize(resType.getRank());
  }
  mlir::AffineMap mapOut =
      mlir::AffineMap::get(numBatchDims + 3, 0, exprsOut, op->getContext());

  mlir::SmallVector<mlir::Value, 4> inputOperands = {valA, valB};
  mlir::SmallVector<mlir::AffineMap, 4> gemmMaps = {mapA, mapB};

  if (hasAZp) {
    inputOperands.push_back(valAZp);
    auto zpType = mlir::dyn_cast<mlir::RankedTensorType>(valAZp.getType());
    if (zpType && zpType.getRank() == 0) {
      gemmMaps.push_back(
          mlir::AffineMap::get(numBatchDims + 3, 0, {}, op->getContext()));
    } else {
      gemmMaps.push_back(mapA);
    }
  }

  if (hasBZp) {
    inputOperands.push_back(valBZp);
    auto zpType = mlir::dyn_cast<mlir::RankedTensorType>(valBZp.getType());
    if (zpType && zpType.getRank() == 0) {
      gemmMaps.push_back(
          mlir::AffineMap::get(numBatchDims + 3, 0, {}, op->getContext()));
    } else {
      gemmMaps.push_back(mapB);
    }
  }

  gemmMaps.push_back(mapOut);

  auto matmulLinalgOp = mlir::linalg::GenericOp::create(
      rewriter, loc, resType, inputOperands, mlir::ValueRange{outBuff},
      gemmMaps, gemmIters,
      [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
        size_t argIdx = 0;
        mlir::Value aVal = args[argIdx++];
        mlir::Value bVal = args[argIdx++];
        mlir::Value aZpVal = hasAZp ? args[argIdx++] : nullptr;
        mlir::Value bZpVal = hasBZp ? args[argIdx++] : nullptr;
        mlir::Value yVal = args[argIdx++];

        mlir::Value product;
        if (isFloat) {
          product = mlir::arith::MulFOp::create(nest, l, aVal, bVal);
          yVal = mlir::arith::AddFOp::create(nest, l, yVal, product);
        } else {
          // Extend i8 -> i32
          auto extendType = [&](mlir::Value val) -> mlir::Value {
            if (val.getType() == resElmType)
              return val;
            if (auto intType =
                    mlir::dyn_cast<mlir::IntegerType>(val.getType())) {
              if (intType.isUnsigned()) {
                return mlir::arith::ExtUIOp::create(nest, l, resElmType, val);
              } else {
                return mlir::arith::ExtSIOp::create(nest, l, resElmType, val);
              }
            }
            return val;
          };

          mlir::Value aValExt = extendType(aVal);
          mlir::Value bValExt = extendType(bVal);

          // Zero-point subtraction if zero points are present
          if (hasAZp && aZpVal) {
            mlir::Value aZpExt = extendType(aZpVal);
            aValExt = mlir::arith::SubIOp::create(nest, l, aValExt, aZpExt);
          }
          if (hasBZp && bZpVal) {
            mlir::Value bZpExt = extendType(bZpVal);
            bValExt = mlir::arith::SubIOp::create(nest, l, bValExt, bZpExt);
          }

          product = mlir::arith::MulIOp::create(nest, l, aValExt, bValExt);
          yVal = mlir::arith::AddIOp::create(nest, l, yVal, product);
        }
        mlir::linalg::YieldOp::create(nest, l, yVal);
      });

  // Tag operation for downstream pass transform optimizations
  matmulLinalgOp->setAttr("transform.target_tag",
                          rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, matmulLinalgOp);
  return mlir::success();
}

} // namespace onnx2mlir::dialect
