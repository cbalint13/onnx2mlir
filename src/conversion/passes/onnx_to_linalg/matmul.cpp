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
 * \brief ONNX MatMul operation to Linalg lowering
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

mlir::LogicalResult OnnxToLinalg_MatMulOp(mlir::Operation *op,
                                          mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value valA = op->getOperand(0);
  mlir::Value valB = op->getOperand(1);

  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(valA.getType());
  auto bType = mlir::dyn_cast<mlir::RankedTensorType>(valB.getType());
  auto origResType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());

  if (!aType || !bType || !origResType) {
    return mlir::emitError(
        Onnx2Mlir_SrcLoc(rewriter),
        opName + " operands and result must be ranked tensor types");
  }

  mlir::Type elmType = origResType.getElementType();
  bool isFloat = mlir::isa<mlir::FloatType>(elmType);

  // Determine compute type (signless for integers)
  mlir::Type cptType = elmType;
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elmType)) {
    cptType = mlir::IntegerType::get(op->getContext(), intType.getWidth());
  }

  // Promote rank 0 tensor to minimum rank 1
  auto resType = origResType;
  if (resType.getRank() == 0) {
    resType = mlir::RankedTensorType::get({1}, elmType);
  }

  bool aIs1D = (aType.getRank() == 1);
  bool bIs1D = (bType.getRank() == 1);

  if (aIs1D) {
    llvm::SmallVector<mlir::ReassociationIndices, 1> reassoc = {{0, 1}};
    auto tgtType =
        mlir::RankedTensorType::get({1, aType.getDimSize(0)}, elmType);
    valA = mlir::tensor::ExpandShapeOp::create(rewriter, loc, tgtType, valA,
                                               reassoc);
    aType = mlir::cast<mlir::RankedTensorType>(valA.getType());
  }

  if (bIs1D) {
    llvm::SmallVector<mlir::ReassociationIndices, 1> reassoc = {{0, 1}};
    auto tgtType =
        mlir::RankedTensorType::get({bType.getDimSize(0), 1}, elmType);
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

  // Initialize out buffer, match elmType (signless)
  auto zeroAttr = rewriter.getZeroAttr(cptType);
  mlir::Value constZero =
      mlir::arith::ConstantOp::create(rewriter, loc, cptType, zeroAttr);

  if (cptType != elmType) {
    constZero = mlir::UnrealizedConversionCastOp::create(rewriter, loc, elmType,
                                                         constZero)
                    .getResult(0);
  }

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

  mlir::SmallVector<mlir::AffineMap> gemmMaps = {mapA, mapB, mapOut};

  auto matmulLinalgOp = mlir::linalg::GenericOp::create(
      rewriter, loc, resType, mlir::ValueRange{valA, valB},
      mlir::ValueRange{outBuff}, gemmMaps, gemmIters,
      [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
        mlir::Value aVal = args[0];
        mlir::Value bVal = args[1];
        mlir::Value yVal = args[2];

        // Convert to compute type for arithmetic
        if (cptType != elmType) {
          aVal =
              mlir::UnrealizedConversionCastOp::create(nest, l, cptType, aVal)
                  .getResult(0);
          bVal =
              mlir::UnrealizedConversionCastOp::create(nest, l, cptType, bVal)
                  .getResult(0);
          yVal =
              mlir::UnrealizedConversionCastOp::create(nest, l, cptType, yVal)
                  .getResult(0);
        }

        mlir::Value product;
        if (isFloat) {
          product = mlir::arith::MulFOp::create(nest, l, aVal, bVal);
          yVal = mlir::arith::AddFOp::create(nest, l, yVal, product);
        } else {
          product = mlir::arith::MulIOp::create(nest, l, aVal, bVal);
          yVal = mlir::arith::AddIOp::create(nest, l, yVal, product);
        }

        // Convert back to original element type
        if (cptType != elmType) {
          yVal =
              mlir::UnrealizedConversionCastOp::create(nest, l, elmType, yVal)
                  .getResult(0);
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
