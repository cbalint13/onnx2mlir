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
 * \file src/conversion/passes/onnx_to_linalg/norm.cpp
 * \brief ONNX BatchNormalization operation to Linalg lowering
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

#include <algorithm>

#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_BatchNormalizationOp(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter,
                                  const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInputX = convRewriter.getRemappedValue(op->getOperand(0));
  auto opInputScale = convRewriter.getRemappedValue(op->getOperand(1));
  auto opInputB = convRewriter.getRemappedValue(op->getOperand(2));
  auto opInputMean = convRewriter.getRemappedValue(op->getOperand(3));
  auto opInputVar = convRewriter.getRemappedValue(op->getOperand(4));
  auto opResult = op->getResult(0);
  auto opOutput = convRewriter.getRemappedValue(opResult);

  auto inXDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInputX.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto outElmType = outDatType.getElementType();

  int64_t inputXRank = inXDatType.getRank();

  /*
   * Attributes
   */

  double epsilonVal = 1e-05;
  if (auto epsAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon"))
    epsilonVal = epsAttr.getValueAsDouble();

  /*
   * Affine mappings
   */

  llvm::SmallVector<mlir::AffineExpr, 4> exprsX;
  for (int64_t i = 0; i < inputXRank; ++i) {
    exprsX.push_back(rewriter.getAffineDimExpr(i));
  }
  auto mapX = mlir::AffineMap::get(inputXRank, 0, exprsX, op->getContext());

  // Per-channel parameter mapping (dimension 1 in ONNX NCHW/NCDHW layout)
  llvm::SmallVector<mlir::AffineExpr, 1> exprs1D;
  exprs1D.push_back(rewriter.getAffineDimExpr(1));
  auto map1D = mlir::AffineMap::get(inputXRank, 0, exprs1D, op->getContext());

  mlir::SmallVector<mlir::AffineMap, 6> indexingMaps = {
      mapX,  // X
      map1D, // scale
      map1D, // B
      map1D, // mean
      map1D, // var
      mapX   // Y
  };

  mlir::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      inputXRank, mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  // dynamic sizes
  llvm::SmallVector<mlir::Value, 4> dynSizes;
  for (int64_t i = 0; i < inputXRank; ++i) {
    if (inXDatType.isDynamicDim(i)) {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, opInputX, i));
    }
  }

  auto out = mlir::tensor::EmptyOp::create(rewriter, loc, outDatType, dynSizes);
  auto zero = rewriter.getZeroAttr(outElmType);
  auto constZero =
      mlir::arith::ConstantOp::create(rewriter, loc, outElmType, zero);
  auto outBuffer = mlir::linalg::FillOp::create(
      rewriter, loc, mlir::ValueRange{constZero}, mlir::ValueRange{out});

  mlir::SmallVector<mlir::Value, 5> inputOperands = {
      opInputX, opInputScale, opInputB, opInputMean, opInputVar};

  auto batchNormOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ inputOperands,
      /*output_values*/ mlir::ValueRange{outBuffer.getResults()},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        auto xVal = args[0];
        auto scaleVal = args[1];
        auto bVal = args[2];
        auto meanVal = args[3];
        auto varVal = args[4];

        auto e = nest.getFloatAttr(outElmType, epsilonVal);
        auto eps = mlir::arith::ConstantOp::create(nest, nloc, outElmType, e);
        // Y = (X - mean) / sqrt(var + epsilon) * scale + B
        auto xSubMean = mlir::arith::SubFOp::create(nest, nloc, xVal, meanVal);
        auto addEps = mlir::arith::AddFOp::create(nest, nloc, varVal, eps);
        auto stdDev = mlir::math::SqrtOp::create(nest, nloc, addEps);
        auto norm = mlir::arith::DivFOp::create(nest, nloc, xSubMean, stdDev);
        auto scaled = mlir::arith::MulFOp::create(nest, nloc, norm, scaleVal);
        auto yVal = mlir::arith::AddFOp::create(nest, nloc, scaled, bVal);

        mlir::linalg::YieldOp::create(nest, nloc, yVal.getResult());
      });

  llvm::SmallVector<mlir::Value, 5> replacements;
  replacements.push_back(batchNormOp.getResult(0));

  for (size_t i = 1; i < op->getNumResults(); ++i) {
    auto res = op->getResult(i);
    if (mlir::isa<mlir::NoneType>(res.getType())) {
      replacements.push_back(nullptr);
    } else if (i == 1 || i == 3) {
      // mean / running_mean / saved_mean
      replacements.push_back(opInputMean);
    } else if (i == 2 || i == 4) {
      // var / running_var / saved_var
      replacements.push_back(opInputVar);
    } else {
      replacements.push_back(nullptr);
    }
  }

  rewriter.replaceOp(op, replacements);

  return mlir::success();
}

mlir::LogicalResult
OnnxToLinalg_LayerNormalizationOp(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter,
                                  const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInputX = convRewriter.getRemappedValue(op->getOperand(0));
  auto opInpScale = convRewriter.getRemappedValue(op->getOperand(1));
  auto opInputB = (op->getNumOperands() > 2 &&
                   !mlir::isa<mlir::NoneType>(op->getOperand(2).getType()))
                      ? convRewriter.getRemappedValue(op->getOperand(2))
                      : nullptr;
  auto opResult = op->getResult(0);
  auto opOutput = convRewriter.getRemappedValue(opResult);

  auto inXDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInputX.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());
  auto sclDatType =
      mlir::dyn_cast<mlir::RankedTensorType>(opInpScale.getType());

  auto outElmType = outDatType.getElementType();

  int64_t inputXRank = inXDatType.getRank();
  int64_t scaleRank = sclDatType.getRank();

  /*
   * Attributes
   */

  // epsilon
  double epsilonVal = 1e-05;
  if (auto epsAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon"))
    epsilonVal = epsAttr.getValueAsDouble();
  // axis
  int64_t axisVal = -1;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axisVal = axisAttr.getInt();
  if (axisVal < 0)
    axisVal += inputXRank;

  /*
   * Affine mappings
   */

  llvm::SmallVector<mlir::AffineExpr, 4> exprsX;
  for (int64_t i = 0; i < inputXRank; ++i)
    exprsX.push_back(rewriter.getAffineDimExpr(i));
  auto mapX = mlir::AffineMap::get(inputXRank, 0, exprsX, op->getContext());

  llvm::SmallVector<mlir::AffineExpr, 4> exprsMeanVar;
  for (int64_t i = 0; i < inputXRank; ++i) {
    if (i < axisVal)
      exprsMeanVar.push_back(rewriter.getAffineDimExpr(i));
    else
      exprsMeanVar.push_back(rewriter.getAffineConstantExpr(0));
  }
  auto mapMeanVar =
      mlir::AffineMap::get(inputXRank, 0, exprsMeanVar, op->getContext());

  llvm::SmallVector<mlir::AffineExpr, 4> exprsScale;
  if (scaleRank == inputXRank - axisVal) {
    for (int64_t i = axisVal; i < inputXRank; ++i)
      exprsScale.push_back(rewriter.getAffineDimExpr(i));
  } else if (scaleRank == inputXRank) {
    for (int64_t i = 0; i < inputXRank; ++i) {
      if (sclDatType.getDimSize(i) == 1)
        exprsScale.push_back(rewriter.getAffineConstantExpr(0));
      else
        exprsScale.push_back(rewriter.getAffineDimExpr(i));
    }
  } else {
    int64_t diff = inputXRank - scaleRank;
    for (int64_t i = 0; i < scaleRank; ++i)
      exprsScale.push_back(rewriter.getAffineDimExpr(diff + i));
  }
  auto mapScale =
      mlir::AffineMap::get(inputXRank, 0, exprsScale, op->getContext());

  mlir::SmallVector<mlir::AffineMap, 6> indexingMaps = {mapX, mapMeanVar,
                                                        mapMeanVar, mapScale};
  if (opInputB)
    indexingMaps.push_back(mapScale);
  indexingMaps.push_back(mapX);

  mlir::SmallVector<mlir::utils::IteratorType> parallelIterators(
      inputXRank, mlir::utils::IteratorType::parallel);

  mlir::SmallVector<mlir::utils::IteratorType> reduceIterators;
  for (int64_t i = 0; i < inputXRank; ++i) {
    if (i < axisVal)
      reduceIterators.push_back(mlir::utils::IteratorType::parallel);
    else
      reduceIterators.push_back(mlir::utils::IteratorType::reduction);
  }

  /*
   *  Linalg ops staging
   */

  llvm::SmallVector<int64_t, 4> meanVarShape;
  llvm::SmallVector<mlir::Value, 4> meanVarDynSizes;
  for (int64_t i = 0; i < inputXRank; ++i) {
    if (i < axisVal) {
      meanVarShape.push_back(inXDatType.getDimSize(i));
      if (inXDatType.isDynamicDim(i))
        meanVarDynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, opInputX, i));
    } else {
      meanVarShape.push_back(1);
    }
  }
  auto meanVarDatType = mlir::RankedTensorType::get(meanVarShape, outElmType);

  auto zero = rewriter.getZeroAttr(outElmType);
  auto constZero =
      mlir::arith::ConstantOp::create(rewriter, loc, outElmType, zero);

  mlir::Value normElemCount = nullptr;
  int64_t staticCount = 1;
  bool isDynamicCount = false;
  for (int64_t i = axisVal; i < inputXRank; ++i) {
    if (inXDatType.isDynamicDim(i))
      isDynamicCount = true;
    else
      staticCount *= inXDatType.getDimSize(i);
  }

  if (!isDynamicCount) {
    auto cstAttr =
        rewriter.getFloatAttr(outElmType, static_cast<double>(staticCount));
    normElemCount =
        mlir::arith::ConstantOp::create(rewriter, loc, outElmType, cstAttr);
  } else {
    mlir::Value accCnt = nullptr;
    for (int64_t i = axisVal; i < inputXRank; ++i) {
      mlir::Value dimVal;
      if (inXDatType.isDynamicDim(i)) {
        dimVal = mlir::tensor::DimOp::create(rewriter, loc, opInputX, i);
      } else {
        auto cstDim = rewriter.getIndexAttr(inXDatType.getDimSize(i));
        dimVal = mlir::arith::ConstantOp::create(
            rewriter, loc, rewriter.getIndexType(), cstDim);
      }
      if (!accCnt)
        accCnt = dimVal;
      else
        accCnt = mlir::arith::MulIOp::create(rewriter, loc, accCnt, dimVal);
    }
    auto i64Count = mlir::arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI64Type(), accCnt);
    normElemCount =
        mlir::arith::SIToFPOp::create(rewriter, loc, outElmType, i64Count);
  }

  auto meanInit = mlir::tensor::EmptyOp::create(rewriter, loc, meanVarDatType,
                                                meanVarDynSizes);
  auto meanSumBuffer = mlir::linalg::FillOp::create(
      rewriter, loc, mlir::ValueRange{constZero}, mlir::ValueRange{meanInit});

  auto meanSumOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{meanVarDatType},
      /*input_values*/ mlir::ValueRange{opInputX},
      /*output_values*/ mlir::ValueRange{meanSumBuffer.getResults()},
      /*affine_maps*/ mlir::SmallVector<mlir::AffineMap, 2>{mapX, mapMeanVar},
      /*iter_types*/ reduceIterators,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        auto xVal = args[0];
        auto accVal = args[1];
        auto sumVal = mlir::arith::AddFOp::create(nest, nloc, xVal, accVal);
        mlir::linalg::YieldOp::create(nest, nloc, sumVal.getResult());
      });

  auto meanBufferAlloc = mlir::tensor::EmptyOp::create(
      rewriter, loc, meanVarDatType, meanVarDynSizes);
  auto meanBufferFill =
      mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{constZero},
                                   mlir::ValueRange{meanBufferAlloc});

  auto meanOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{meanVarDatType},
      /*input_values*/ mlir::ValueRange{meanSumOp.getResult(0), normElemCount},
      /*output_values*/ mlir::ValueRange{meanBufferFill.getResults()},
      /*affine_maps*/
      mlir::SmallVector<mlir::AffineMap, 3>{
          mapX, mlir::AffineMap::get(inputXRank, 0, op->getContext()), mapX},
      /*iter_types*/ parallelIterators,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        auto sumVal = args[0];
        auto countVal = args[1];
        auto avgVal = mlir::arith::DivFOp::create(nest, nloc, sumVal, countVal);
        mlir::linalg::YieldOp::create(nest, nloc, avgVal.getResult());
      });
  auto meanValTensor = meanOp.getResult(0);

  auto varInit = mlir::tensor::EmptyOp::create(rewriter, loc, meanVarDatType,
                                               meanVarDynSizes);
  auto varSumBuffer = mlir::linalg::FillOp::create(
      rewriter, loc, mlir::ValueRange{constZero}, mlir::ValueRange{varInit});

  auto varSumOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{meanVarDatType},
      /*input_values*/ mlir::ValueRange{opInputX, meanValTensor},
      /*output_values*/ mlir::ValueRange{varSumBuffer.getResults()},
      /*affine_maps*/
      mlir::SmallVector<mlir::AffineMap, 3>{mapX, mapMeanVar, mapMeanVar},
      /*iter_types*/ reduceIterators,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*result_types*/ mlir::ValueRange args) {
        auto xVal = args[0];
        auto meanVal = args[1];
        auto accVal = args[2];
        auto diff = mlir::arith::SubFOp::create(nest, nloc, xVal, meanVal);
        auto sqr = mlir::arith::MulFOp::create(nest, nloc, diff, diff);
        auto sumVal = mlir::arith::AddFOp::create(nest, nloc, sqr, accVal);
        mlir::linalg::YieldOp::create(nest, nloc, sumVal.getResult());
      });

  auto varBufferAlloc = mlir::tensor::EmptyOp::create(
      rewriter, loc, meanVarDatType, meanVarDynSizes);
  auto varBufferFill =
      mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{constZero},
                                   mlir::ValueRange{varBufferAlloc});

  auto varOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{meanVarDatType},
      /*input_values*/ mlir::ValueRange{varSumOp.getResult(0), normElemCount},
      /*output_values*/ mlir::ValueRange{varBufferFill.getResults()},
      /*affine_maps*/
      mlir::SmallVector<mlir::AffineMap, 3>{
          mapX, mlir::AffineMap::get(inputXRank, 0, op->getContext()), mapX},
      /*iter_types*/ parallelIterators,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        auto sumVal = args[0];
        auto countVal = args[1];
        auto avgVal = mlir::arith::DivFOp::create(nest, nloc, sumVal, countVal);
        mlir::linalg::YieldOp::create(nest, nloc, avgVal.getResult());
      });
  auto varValTensor = varOp.getResult(0);

  llvm::SmallVector<mlir::Value, 4> dynSizes;
  for (int64_t i = 0; i < inputXRank; ++i) {
    if (inXDatType.isDynamicDim(i)) {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, opInputX, i));
    }
  }

  auto out = mlir::tensor::EmptyOp::create(rewriter, loc, outDatType, dynSizes);
  auto outBuffer = mlir::linalg::FillOp::create(
      rewriter, loc, mlir::ValueRange{constZero}, mlir::ValueRange{out});

  mlir::SmallVector<mlir::Value, 5> inputOperands = {opInputX, meanValTensor,
                                                     varValTensor, opInpScale};
  if (opInputB)
    inputOperands.push_back(opInputB);

  auto layerNormOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ inputOperands,
      /*output_values*/ mlir::ValueRange{outBuffer.getResults()},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ parallelIterators,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        auto xVal = args[0];
        auto meanVal = args[1];
        auto varVal = args[2];
        auto scaleVal = args[3];
        auto bVal = opInputB ? args[4] : nullptr;

        auto e = nest.getFloatAttr(outElmType, epsilonVal);
        auto eps = mlir::arith::ConstantOp::create(nest, nloc, outElmType, e);
        // Y = (X - mean) / sqrt(var + epsilon) * scale + B
        auto xSubMean = mlir::arith::SubFOp::create(nest, nloc, xVal, meanVal);
        auto addEps = mlir::arith::AddFOp::create(nest, nloc, varVal, eps);
        auto stdDev = mlir::math::SqrtOp::create(nest, nloc, addEps);
        auto norm = mlir::arith::DivFOp::create(nest, nloc, xSubMean, stdDev);
        auto scaled = mlir::arith::MulFOp::create(nest, nloc, norm, scaleVal);
        auto yVal = scaled.getResult();

        if (bVal)
          yVal = mlir::arith::AddFOp::create(nest, nloc, scaled, bVal);

        mlir::linalg::YieldOp::create(nest, nloc, yVal);
      });

  llvm::SmallVector<mlir::Value, 3> replacements;
  replacements.push_back(layerNormOp.getResult(0));

  for (size_t i = 1; i < op->getNumResults(); ++i) {
    auto res = op->getResult(i);
    if (mlir::isa<mlir::NoneType>(res.getType())) {
      replacements.push_back(nullptr);
    } else if (i == 1) {
      // mean
      replacements.push_back(meanValTensor);
    } else if (i == 2) {
      // invstddev / variance
      replacements.push_back(varValTensor);
    } else {
      replacements.push_back(nullptr);
    }
  }

  rewriter.replaceOp(op, replacements);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
