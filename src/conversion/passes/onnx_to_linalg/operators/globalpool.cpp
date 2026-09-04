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

#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_GlobalPoolOps(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                           const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto inpElmType = inpDatType.getElementType();

  int64_t inputRank = inpDatType.getRank();

  // checks
  if (inputRank < 3)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " input tensor rank must be at least 3";
  if (!inpElmType.isFloat())
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " requires float element type";

  int64_t numSpatialElements = 1;
  for (int64_t i = 2; i < inputRank; ++i) {
    int64_t dimSize = inpDatType.getDimSize(i);
    if (dimSize == mlir::ShapedType::kDynamic)
      return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
             << opName << " dynamic spatial dimensions are not supported";
    numSpatialElements *= dimSize;
  }

  /*
   *  Attributes
   */

  // p
  double attr_p = 2.0;
  if (auto pAttr = op->getAttr("p")) {
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(pAttr))
      attr_p = floatAttr.getValueAsDouble();
    else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(pAttr))
      attr_p = static_cast<double>(intAttr.getInt());
  }

  /*
   *  Affine mappings
   */

  auto zeroExpr = rewriter.getAffineConstantExpr(0);
  llvm::SmallVector<mlir::AffineExpr, 4> outExprs;
  outExprs.push_back(rewriter.getAffineDimExpr(0)); // N
  outExprs.push_back(rewriter.getAffineDimExpr(1)); // C
  for (int64_t i = 2; i < inputRank; ++i)
    outExprs.push_back(zeroExpr);

  auto inpMap = rewriter.getMultiDimIdentityMap(inputRank);
  auto outMap = mlir::AffineMap::get(inputRank, 0, outExprs, op->getContext());

  llvm::SmallVector<mlir::AffineMap, 2> indexingMaps = {inpMap, outMap};

  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes;
  iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // N
  iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // C
  for (int64_t i = 2; i < inputRank; ++i)
    iteratorTypes.push_back(mlir::utils::IteratorType::reduction);

  // post
  auto postMap = rewriter.getMultiDimIdentityMap(inputRank);
  llvm::SmallVector<mlir::AffineMap, 2> postIndexingMaps = {postMap, postMap};

  llvm::SmallVector<mlir::utils::IteratorType, 4> postIteratorTypes(
      inputRank, mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  mlir::linalg::GenericOp globalpoolOp;

  auto empty = mlir::tensor::EmptyOp::create(
      rewriter, loc, outDatType.getShape(), inpDatType.getElementType());
  auto zero = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(inpElmType, 0.0));
  auto sumBuffer = mlir::linalg::FillOp::create(
      rewriter, loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});

  auto reductionGenericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInput},
      /*output_values*/ mlir::ValueRange{sumBuffer.getResult(0)},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::Value accum;
        if (opNameBeginsWith(opName, "GlobalAveragePool")) {
          accum = mlir::arith::AddFOp::create(nest, nloc, args[0], args[1]);
        } else if (opNameBeginsWith(opName, "GlobalMaxPool")) {
          accum = mlir::arith::MaximumFOp::create(nest, nloc, args[0], args[1]);
        } else if (opNameBeginsWith(opName, "GlobalLpPool")) {
          mlir::Value pwrIn;
          auto absInput = mlir::math::AbsFOp::create(nest, nloc, args[0]);
          if (attr_p == 1.0f) {
            pwrIn = absInput;
          } else if (attr_p == 2.0f) {
            pwrIn = mlir::arith::MulFOp::create(nest, nloc, absInput, absInput);
          } else {
            auto pAttr = nest.getFloatAttr(inpElmType, attr_p);
            auto pConst = mlir::arith::ConstantOp::create(nest, nloc, pAttr);
            pwrIn = mlir::math::PowFOp::create(nest, nloc, absInput, pConst);
          }
          accum = mlir::arith::AddFOp::create(nest, nloc, pwrIn, args[1]);
        }
        mlir::linalg::YieldOp::create(nest, nloc, accum);
      });

  globalpoolOp = reductionGenericOp;

  if (opNameBeginsWith(opName, {"GlobalAveragePool", "GlobalLpPool"})) {
    auto postBuffer = mlir::tensor::EmptyOp::create(
        rewriter, loc, outDatType.getShape(), inpDatType.getElementType());

    auto postGenericOp = mlir::linalg::GenericOp::create(
        /*op_builder*/ rewriter, /*src_location*/ loc,
        /*result_types*/ mlir::TypeRange{outDatType},
        /*input_values*/ mlir::ValueRange{reductionGenericOp.getResult(0)},
        /*output_values*/ mlir::ValueRange{postBuffer},
        /*affine_maps*/ postIndexingMaps,
        /*iter_types*/ postIteratorTypes,
        [&](/*op_builder*/ mlir::OpBuilder &nest,
            /*src_location*/ mlir::Location nloc,
            /*value_args*/ mlir::ValueRange args) {
          mlir::Value val;
          if (opNameBeginsWith(opName, "GlobalAveragePool")) {
            auto count = nest.getFloatAttr(
                inpElmType, static_cast<double>(numSpatialElements));
            auto cntConst = mlir::arith::ConstantOp::create(nest, nloc, count);
            val = mlir::arith::DivFOp::create(nest, nloc, args[0], cntConst);
          } else if (opNameBeginsWith(opName, "GlobalLpPool")) {
            if (attr_p == 1.0f) {
              val = args[0];
            } else if (attr_p == 2.0f) {
              val = mlir::math::SqrtOp::create(nest, nloc, args[0]);
            } else {
              auto pAttr = nest.getFloatAttr(inpElmType, 1.0 / attr_p);
              auto invP = mlir::arith::ConstantOp::create(nest, nloc, pAttr);
              val = mlir::math::PowFOp::create(nest, nloc, args[0], invP);
            }
          }
          mlir::linalg::YieldOp::create(nest, nloc, val);
        });

    globalpoolOp = postGenericOp;
  }

  rewriter.replaceOp(op, globalpoolOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
