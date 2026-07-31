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
 * \file src/conversion/passes/onnx_to_linalg/globalpool.cpp
 * \brief ONNX Global pooling operations to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_GlobalAveragePoolOp(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value input = op->getOperand(0);
  mlir::Value result = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());

  if (!inpType || !resType) {
    return mlir::emitError(
        Onnx2Mlir_SrcLoc(rewriter),
        opName + " operand and result must be ranked tensor type");
  }

  auto elemType = inpType.getElementType();
  auto floatType = mlir::dyn_cast<mlir::FloatType>(elemType);
  if (!floatType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " requires float element type");
  }

  int64_t rank = inpType.getRank();
  if (rank < 3) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " input tensor rank must be at least 3");
  }

  int64_t numSpatialElements = 1;
  for (int64_t i = 2; i < rank; ++i) {
    int64_t dimSize = inpType.getDimSize(i);
    if (dimSize == mlir::ShapedType::kDynamic) {
      return mlir::emitError(
          Onnx2Mlir_SrcLoc(rewriter),
          opName + " dynamic spatial dimensions are not supported");
    }
    numSpatialElements *= dimSize;
  }

  mlir::Value initConst;
  // Fill accumulator buffer with zero for sum / Lp reduction
  initConst = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(elemType, 0.0));

  mlir::Value sumEmpty = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), elemType);

  mlir::Value sumBuffer =
      mlir::linalg::FillOp::create(rewriter, loc, initConst, sumEmpty)
          .getResult(0);

  // Iterators: N, C are parallel; spatial dims (H, W, ...) are reduction
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes;
  iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // N
  iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // C
  for (int64_t i = 2; i < rank; ++i) {
    iteratorTypes.push_back(mlir::utils::IteratorType::reduction);
  }

  // Input Map: (d0, d1, d2, ..., d_{rank-1}) -> (d0, d1, d2, ..., d_{rank-1})
  mlir::AffineMap inputMap = rewriter.getMultiDimIdentityMap(rank);

  // Output Map: (d0, d1, d2, ..., d_{rank-1}) -> (d0, d1, 0, 0, ..., 0)
  mlir::Builder builder(op->getContext());
  mlir::AffineExpr zeroExpr = builder.getAffineConstantExpr(0);
  llvm::SmallVector<mlir::AffineExpr, 4> outputExprs;
  outputExprs.push_back(builder.getAffineDimExpr(0)); // N
  outputExprs.push_back(builder.getAffineDimExpr(1)); // C
  for (int64_t i = 2; i < rank; ++i) {
    outputExprs.push_back(zeroExpr);
  }
  mlir::AffineMap outputMap =
      mlir::AffineMap::get(rank, 0, outputExprs, op->getContext());

  llvm::SmallVector<mlir::AffineMap, 2> indexingMaps = {inputMap, outputMap};

  auto reductionGenericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, mlir::TypeRange{resType}, mlir::ValueRange{input},
      mlir::ValueRange{sumBuffer}, indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value scalarInput = args[0];
        mlir::Value scalarAcc = args[1];
        mlir::Value updatedAcc;

        // Average pooling accumulation: updatedAcc = scalarInput + scalarAcc
        updatedAcc = mlir::arith::AddFOp::create(nestedBuilder, nestedLoc,
                                                 scalarInput, scalarAcc);

        mlir::linalg::YieldOp::create(nestedBuilder, nestedLoc, updatedAcc);
      });

  reductionGenericOp->setAttr(
      "transform.target_tag",
      rewriter.getStringAttr(opName.str() + "_reduction"));

  mlir::Value postEmpty = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), elemType);

  mlir::AffineMap idMap = rewriter.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap, 2> postIndexingMaps = {idMap, idMap};
  llvm::SmallVector<mlir::utils::IteratorType, 4> postIteratorTypes(
      rank, mlir::utils::IteratorType::parallel);

  auto postGenericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, mlir::TypeRange{resType},
      mlir::ValueRange{reductionGenericOp.getResult(0)},
      mlir::ValueRange{postEmpty}, postIndexingMaps, postIteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value scalarSum = args[0];
        mlir::Value finalVal;

        // Division by spatial volume count: avg = sum / numSpatialElements
        mlir::Value countConst = mlir::arith::ConstantOp::create(
            nestedBuilder, nestedLoc,
            nestedBuilder.getFloatAttr(
                elemType, static_cast<double>(numSpatialElements)));
        finalVal = mlir::arith::DivFOp::create(nestedBuilder, nestedLoc,
                                               scalarSum, countConst);
        mlir::linalg::YieldOp::create(nestedBuilder, nestedLoc, finalVal);
      });

  postGenericOp->setAttr("transform.target_tag",
                         rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, postGenericOp.getResults());

  return mlir::success();
}

mlir::LogicalResult
OnnxToLinalg_GlobalLpPoolOp(mlir::Operation *op,
                            mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value input = op->getOperand(0);
  mlir::Value result = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());

  if (!inpType || !resType) {
    return mlir::emitError(
        Onnx2Mlir_SrcLoc(rewriter),
        opName + " operand and result must be ranked tensor type");
  }

  auto elemType = inpType.getElementType();
  auto floatType = mlir::dyn_cast<mlir::FloatType>(elemType);
  if (!floatType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " requires float element type");
  }

  int64_t rank = inpType.getRank();
  if (rank < 3) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " input tensor rank must be at least 3");
  }

  int64_t numSpatialElements = 1;
  for (int64_t i = 2; i < rank; ++i) {
    int64_t dimSize = inpType.getDimSize(i);
    if (dimSize == mlir::ShapedType::kDynamic) {
      return mlir::emitError(
          Onnx2Mlir_SrcLoc(rewriter),
          opName + " dynamic spatial dimensions are not supported");
    }
    numSpatialElements *= dimSize;
  }

  double pVal = 2.0;
  if (auto pAttr = op->getAttr("p")) {
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(pAttr)) {
      pVal = floatAttr.getValueAsDouble();
    } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(pAttr)) {
      pVal = static_cast<double>(intAttr.getInt());
    }
  }

  mlir::Value initConst;
  // Fill accumulator buffer with zero for sum / Lp reduction
  initConst = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(elemType, 0.0));

  mlir::Value sumEmpty = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), elemType);

  mlir::Value sumBuffer =
      mlir::linalg::FillOp::create(rewriter, loc, initConst, sumEmpty)
          .getResult(0);

  // Iterators: N, C are parallel; spatial dims (H, W, ...) are reduction
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes;
  iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // N
  iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // C
  for (int64_t i = 2; i < rank; ++i) {
    iteratorTypes.push_back(mlir::utils::IteratorType::reduction);
  }

  // Input Map: (d0, d1, d2, ..., d_{rank-1}) -> (d0, d1, d2, ..., d_{rank-1})
  mlir::AffineMap inputMap = rewriter.getMultiDimIdentityMap(rank);

  // Output Map: (d0, d1, d2, ..., d_{rank-1}) -> (d0, d1, 0, 0, ..., 0)
  mlir::Builder builder(op->getContext());
  mlir::AffineExpr zeroExpr = builder.getAffineConstantExpr(0);
  llvm::SmallVector<mlir::AffineExpr, 4> outputExprs;
  outputExprs.push_back(builder.getAffineDimExpr(0)); // N
  outputExprs.push_back(builder.getAffineDimExpr(1)); // C
  for (int64_t i = 2; i < rank; ++i) {
    outputExprs.push_back(zeroExpr);
  }
  mlir::AffineMap outputMap =
      mlir::AffineMap::get(rank, 0, outputExprs, op->getContext());

  llvm::SmallVector<mlir::AffineMap, 2> indexingMaps = {inputMap, outputMap};

  auto reductionGenericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, mlir::TypeRange{resType}, mlir::ValueRange{input},
      mlir::ValueRange{sumBuffer}, indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value scalarInput = args[0];
        mlir::Value scalarAcc = args[1];
        mlir::Value updatedAcc;

        // Lp pooling accumulation: updatedAcc = scalarAcc + |scalarInput|^p
        mlir::Value absInput =
            mlir::math::AbsFOp::create(nestedBuilder, nestedLoc, scalarInput);
        mlir::Value poweredInput;
        if (pVal == 1.0) {
          poweredInput = absInput;
        } else if (pVal == 2.0) {
          poweredInput = mlir::arith::MulFOp::create(nestedBuilder, nestedLoc,
                                                     absInput, absInput);
        } else {
          mlir::Value pConst = mlir::arith::ConstantOp::create(
              nestedBuilder, nestedLoc,
              nestedBuilder.getFloatAttr(elemType, pVal));
          poweredInput = mlir::math::PowFOp::create(nestedBuilder, nestedLoc,
                                                    absInput, pConst);
        }
        updatedAcc = mlir::arith::AddFOp::create(nestedBuilder, nestedLoc,
                                                 poweredInput, scalarAcc);

        mlir::linalg::YieldOp::create(nestedBuilder, nestedLoc, updatedAcc);
      });

  reductionGenericOp->setAttr(
      "transform.target_tag",
      rewriter.getStringAttr(opName.str() + "_reduction"));

  mlir::Value postEmpty = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), elemType);

  mlir::AffineMap idMap = rewriter.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap, 2> postIndexingMaps = {idMap, idMap};
  llvm::SmallVector<mlir::utils::IteratorType, 4> postIteratorTypes(
      rank, mlir::utils::IteratorType::parallel);

  auto postGenericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, mlir::TypeRange{resType},
      mlir::ValueRange{reductionGenericOp.getResult(0)},
      mlir::ValueRange{postEmpty}, postIndexingMaps, postIteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value scalarSum = args[0];
        mlir::Value finalVal;

        // Root extraction: (sum)^(1/p)
        if (pVal == 1.0) {
          finalVal = scalarSum;
        } else if (pVal == 2.0) {
          finalVal =
              mlir::math::SqrtOp::create(nestedBuilder, nestedLoc, scalarSum);
        } else {
          mlir::Value invPConst = mlir::arith::ConstantOp::create(
              nestedBuilder, nestedLoc,
              nestedBuilder.getFloatAttr(elemType, 1.0 / pVal));
          finalVal = mlir::math::PowFOp::create(nestedBuilder, nestedLoc,
                                                scalarSum, invPConst);
        }

        mlir::linalg::YieldOp::create(nestedBuilder, nestedLoc, finalVal);
      });

  postGenericOp->setAttr("transform.target_tag",
                         rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, postGenericOp.getResults());

  return mlir::success();
}

mlir::LogicalResult
OnnxToLinalg_GlobalMaxPoolOp(mlir::Operation *op,
                             mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value input = op->getOperand(0);
  mlir::Value result = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());

  if (!inpType || !resType) {
    return mlir::emitError(
        Onnx2Mlir_SrcLoc(rewriter),
        opName + " operand and result must be ranked tensor type");
  }

  auto elemType = inpType.getElementType();
  auto floatType = mlir::dyn_cast<mlir::FloatType>(elemType);
  if (!floatType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " requires float element type");
  }

  int64_t rank = inpType.getRank();
  if (rank < 3) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " input tensor rank must be at least 3");
  }

  auto negInf =
      llvm::APFloat::getInf(floatType.getFloatSemantics(), /*Negative=*/true);
  mlir::Value initConst = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(elemType, negInf));

  mlir::Value emptyOut = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), elemType);

  mlir::Value initBuf =
      mlir::linalg::FillOp::create(rewriter, loc, initConst, emptyOut)
          .getResult(0);

  // Iterators: N, C are parallel; spatial dims (H, W, ...) are reduction
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes;
  iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // N
  iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // C
  for (int64_t i = 2; i < rank; ++i) {
    iteratorTypes.push_back(mlir::utils::IteratorType::reduction);
  }

  // Input Map: identity
  mlir::AffineMap inputMap = rewriter.getMultiDimIdentityMap(rank);

  // Output Map: N, C, 0, 0, ...
  mlir::Builder builder(op->getContext());
  mlir::AffineExpr zeroExpr = builder.getAffineConstantExpr(0);
  llvm::SmallVector<mlir::AffineExpr, 4> outputExprs;
  outputExprs.push_back(builder.getAffineDimExpr(0)); // N
  outputExprs.push_back(builder.getAffineDimExpr(1)); // C
  for (int64_t i = 2; i < rank; ++i) {
    outputExprs.push_back(zeroExpr);
  }
  mlir::AffineMap outputMap =
      mlir::AffineMap::get(rank, 0, outputExprs, op->getContext());

  llvm::SmallVector<mlir::AffineMap, 2> indexingMaps = {inputMap, outputMap};

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, mlir::TypeRange{resType}, mlir::ValueRange{input},
      mlir::ValueRange{initBuf}, indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value scalarInput = args[0];
        mlir::Value scalarAcc = args[1];

        // Max pooling accumulation
        auto maxOp = mlir::arith::MaximumFOp::create(nestedBuilder, nestedLoc,
                                                     scalarInput, scalarAcc);
        mlir::linalg::YieldOp::create(nestedBuilder, nestedLoc,
                                      maxOp->getResult(0));
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp.getResults());

  return mlir::success();
}

} // namespace onnx2mlir::dialect
