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
  auto opName = op->getName().getStringRef();

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
  auto orgDatType = mlir::dyn_cast<mlir::RankedTensorType>(opResult.getType());

  auto outElmType = outDatType.getElementType();
  auto orgElmType = orgDatType.getElementType();

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

} // namespace onnx2mlir::dialect
