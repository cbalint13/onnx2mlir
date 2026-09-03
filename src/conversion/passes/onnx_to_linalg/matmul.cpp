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

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInputA = convRewriter.getRemappedValue(op->getOperand(0));
  auto opInputB = convRewriter.getRemappedValue(op->getOperand(1));
  auto opInpAZp = (op->getNumOperands() > 2 &&
                   !mlir::isa<mlir::NoneType>(op->getOperand(2).getType()))
                      ? convRewriter.getRemappedValue(op->getOperand(2))
                      : nullptr;
  auto opInpBZp = (op->getNumOperands() > 3 &&
                   !mlir::isa<mlir::NoneType>(op->getOperand(3).getType()))
                      ? convRewriter.getRemappedValue(op->getOperand(3))
                      : nullptr;
  auto opResult = op->getResult(0);
  auto opOutput = convRewriter.getRemappedValue(opResult);

  auto inADatType = mlir::dyn_cast<mlir::RankedTensorType>(opInputA.getType());
  auto inBDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInputB.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());
  auto orgDatType = mlir::dyn_cast<mlir::RankedTensorType>(opResult.getType());

  auto outElmType = outDatType.getElementType();
  auto orgElmType = orgDatType.getElementType();

  bool aIs1D = (inADatType.getRank() == 1);
  bool bIs1D = (inBDatType.getRank() == 1);

  /*
   * Rank expansion
   */

  // rank 0 -> rank 1 (output)
  if (outDatType.getRank() == 0)
    outDatType = mlir::RankedTensorType::get({1}, outElmType);

  // rank 1 -> rank 2 (inputs)
  if (inADatType.getRank() == 1) {
    llvm::SmallVector<mlir::ReassociationIndices, 1> reassoc = {{0, 1}};
    auto tgtType = mlir::RankedTensorType::get({1, inADatType.getDimSize(0)},
                                               inADatType.getElementType());
    opInputA = mlir::tensor::ExpandShapeOp::create(rewriter, loc, tgtType,
                                                   opInputA, reassoc);
    inADatType = mlir::cast<mlir::RankedTensorType>(opInputA.getType());
  }
  if (inBDatType.getRank() == 1) {
    llvm::SmallVector<mlir::ReassociationIndices, 1> reassoc = {{0, 1}};
    auto tgtType = mlir::RankedTensorType::get({inBDatType.getDimSize(0), 1},
                                               inBDatType.getElementType());
    opInputB = mlir::tensor::ExpandShapeOp::create(rewriter, loc, tgtType,
                                                   opInputB, reassoc);
    inBDatType = mlir::cast<mlir::RankedTensorType>(opInputB.getType());
  }

  int64_t inputARank = inADatType.getRank();
  int64_t inputBRank = inBDatType.getRank();
  int64_t outputRank = std::max(inputARank, inputBRank);
  int64_t nBatchDims = outputRank - 2;

  int64_t outDimIdx = 0;
  llvm::SmallVector<mlir::Value, 4> dynSizes;

  for (int64_t i = 0; i < nBatchDims; ++i) {
    if (outDatType.isDynamicDim(outDimIdx)) {
      int64_t aBatchIdx = i - (nBatchDims - (inputARank - 2));
      int64_t bBatchIdx = i - (nBatchDims - (inputBRank - 2));
      if (aBatchIdx >= 0 && inADatType.isDynamicDim(aBatchIdx)) {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, opInputA, aBatchIdx));
      } else if (bBatchIdx >= 0 && inBDatType.isDynamicDim(bBatchIdx)) {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, opInputB, bBatchIdx));
      }
    }
    outDimIdx++;
  }

  if (!aIs1D && outDimIdx < outDatType.getRank()) {
    if (outDatType.isDynamicDim(outDimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, opInputA, inputARank - 2));
    outDimIdx++;
  }

  if (!bIs1D && outDimIdx < outDatType.getRank()) {
    if (outDatType.isDynamicDim(outDimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, opInputB, inputBRank - 1));
    outDimIdx++;
  }

  mlir::SmallVector<mlir::Value, 4> inputOperands = {opInputA, opInputB};

  /*
   * Affine mappings
   */

  auto mExpr = rewriter.getAffineDimExpr(nBatchDims);
  auto nExpr = rewriter.getAffineDimExpr(nBatchDims + 1);
  auto kExpr = rewriter.getAffineDimExpr(nBatchDims + 2);

  // mapA
  llvm::SmallVector<mlir::AffineExpr, 4> exprsA;
  for (int64_t i = 0; i < inputARank - 2; ++i) {
    int64_t outBatchIdx = nBatchDims - (inputARank - 2) + i;
    if (inADatType.getDimSize(i) == 1) {
      exprsA.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      exprsA.push_back(rewriter.getAffineDimExpr(outBatchIdx));
    }
  }
  exprsA.push_back(mExpr);
  exprsA.push_back(kExpr);
  auto mapA = mlir::AffineMap::get(nBatchDims + 3, 0, exprsA, op->getContext());

  // mapB
  llvm::SmallVector<mlir::AffineExpr, 4> exprsB;
  for (int64_t j = 0; j < inputBRank - 2; ++j) {
    int64_t outBatchIdx = nBatchDims - (inputBRank - 2) + j;
    if (inBDatType.getDimSize(j) == 1) {
      exprsB.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      exprsB.push_back(rewriter.getAffineDimExpr(outBatchIdx));
    }
  }
  exprsB.push_back(kExpr);
  exprsB.push_back(nExpr);
  auto mapB = mlir::AffineMap::get(nBatchDims + 3, 0, exprsB, op->getContext());

  // indexing maps
  mlir::SmallVector<mlir::AffineMap, 4> indexingGemmMaps = {mapA, mapB};

  // mapA & opAZp
  if (opInpAZp) {
    inputOperands.push_back(opInpAZp);
    auto zpType = mlir::dyn_cast<mlir::RankedTensorType>(opInpAZp.getType());
    if (zpType && zpType.getRank() == 0) {
      indexingGemmMaps.push_back(
          mlir::AffineMap::get(nBatchDims + 3, 0, {}, op->getContext()));
    } else {
      indexingGemmMaps.push_back(mapA);
    }
  }

  // mapB & opBZp
  if (opInpBZp) {
    inputOperands.push_back(opInpBZp);
    auto zpType = mlir::dyn_cast<mlir::RankedTensorType>(opInpBZp.getType());
    if (zpType && zpType.getRank() == 0) {
      indexingGemmMaps.push_back(
          mlir::AffineMap::get(nBatchDims + 3, 0, {}, op->getContext()));
    } else {
      indexingGemmMaps.push_back(mapB);
    }
  }

  // mapOut
  llvm::SmallVector<mlir::AffineExpr, 4> exprsOut;
  for (int64_t d = 0; d < nBatchDims; ++d)
    exprsOut.push_back(rewriter.getAffineDimExpr(d));
  if (!aIs1D)
    exprsOut.push_back(mExpr);
  if (!bIs1D)
    exprsOut.push_back(nExpr);
  while (exprsOut.size() < static_cast<size_t>(outDatType.getRank())) {
    exprsOut.push_back(rewriter.getAffineConstantExpr(0));
  }
  if (exprsOut.size() > static_cast<size_t>(outDatType.getRank()))
    exprsOut.resize(outDatType.getRank());
  auto mapOut =
      mlir::AffineMap::get(nBatchDims + 3, 0, exprsOut, op->getContext());
  indexingGemmMaps.push_back(mapOut);

  // iterators
  mlir::SmallVector<mlir::utils::IteratorType> interatorGemmTypes;
  for (int64_t i = 0; i < nBatchDims + 2; ++i)
    interatorGemmTypes.push_back(mlir::utils::IteratorType::parallel);
  interatorGemmTypes.push_back(mlir::utils::IteratorType::reduction);

  /*
   *  Linalg ops staging
   */

  auto out = mlir::tensor::EmptyOp::create(rewriter, loc, outDatType, dynSizes);
  auto zero = rewriter.getZeroAttr(outDatType.getElementType());
  auto constZero = mlir::arith::ConstantOp::create(
      rewriter, loc, outDatType.getElementType(), zero);
  auto outBuffer = mlir::linalg::FillOp::create(
      rewriter, loc, mlir::ValueRange{constZero}, mlir::ValueRange{out});

  auto matmulOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ inputOperands,
      /*output_values*/ mlir::ValueRange{outBuffer.getResults()},
      /*affine_maps*/ indexingGemmMaps,
      /*iter_types*/ interatorGemmTypes,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location l,
          /*value_args*/ mlir::ValueRange args) {
        size_t argIdx = 0;
        auto aVal = args[argIdx++];
        auto bVal = args[argIdx++];
        auto aZpVal = opInpAZp ? args[argIdx++] : nullptr;
        auto bZpVal = opInpBZp ? args[argIdx++] : nullptr;
        auto yVal = args[argIdx++];

        mlir::Value product;
        if (orgElmType.isFloat()) {
          product = mlir::arith::MulFOp::create(nest, l, aVal, bVal);
          yVal = mlir::arith::AddFOp::create(nest, l, yVal, product);
        } else {
          // extend integer bitwidth
          auto extendType = [&](mlir::Value val) -> mlir::Value {
            if (val.getType() == outElmType)
              return val;
            if (auto iType = mlir::dyn_cast<mlir::IntegerType>(val.getType())) {
              if (iType.isUnsigned())
                return mlir::arith::ExtUIOp::create(nest, l, outElmType, val);
              else
                return mlir::arith::ExtSIOp::create(nest, l, outElmType, val);
            }
            return val;
          };

          aVal = extendType(aVal);
          bVal = extendType(bVal);

          if (opInpAZp && aZpVal) {
            aZpVal = extendType(aZpVal);
            aVal = mlir::arith::SubIOp::create(nest, l, aVal, aZpVal);
          }
          if (opInpBZp && bZpVal) {
            bZpVal = extendType(bZpVal);
            bVal = mlir::arith::SubIOp::create(nest, l, bVal, bZpVal);
          }

          product = mlir::arith::MulIOp::create(nest, l, aVal, bVal);
          yVal = mlir::arith::AddIOp::create(nest, l, yVal, product);
        }
        mlir::linalg::YieldOp::create(nest, l, yVal);
      });

  rewriter.replaceOp(op, matmulOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
