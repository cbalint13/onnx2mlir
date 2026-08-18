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
 * \file src/conversion/passes/onnx_to_linalg/gather.cpp
 * \brief ONNX Gather operation to Linalg lowering pass
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_GatherOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opIndices = convRewriter.getRemappedValue(op->getOperand(1));
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto indDatType = mlir::dyn_cast<mlir::RankedTensorType>(opIndices.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  int64_t inputRank = inpDatType.getRank();
  int64_t indicesRank = indDatType.getRank();
  int64_t outputRank = indicesRank + inputRank - 1;

  // checks
  if (inputRank < 1)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " input operand rank must be >= 1";

  /*
   * Attributes
   */

  // axis
  int64_t attr_axis = 0;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    attr_axis = axisAttr.getInt();
  if (attr_axis < -inputRank || attr_axis >= inputRank)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " invalid axis: " << attr_axis;
  if (attr_axis < 0)
    attr_axis += inputRank;
  attr_axis = std::clamp<int64_t>(attr_axis, 0, inputRank);

  /*
   *  Affine mappings
   */

  llvm::SmallVector<mlir::AffineExpr> indicesExprs;
  for (int64_t i : llvm::seq(attr_axis, attr_axis + indicesRank))
    indicesExprs.push_back(rewriter.getAffineDimExpr(i));

  auto indicesMap =
      mlir::AffineMap::get(outputRank, 0, indicesExprs, rewriter.getContext());
  auto outputMap = rewriter.getMultiDimIdentityMap(outputRank);

  llvm::SmallVector<mlir::AffineMap> indexingMaps = {indicesMap, outputMap};

  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      outputRank, mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  auto axisConst =
      mlir::arith::ConstantIndexOp::create(rewriter, loc, attr_axis);
  auto axisDimVal =
      mlir::tensor::DimOp::create(rewriter, loc, opInput, axisConst);

  // output buffer dynamic dimensions
  llvm::SmallVector<mlir::Value> dynamicSizes;
  for (int64_t i = 0; i < outputRank; ++i) {
    if (outDatType.isDynamicDim(i)) {
      if (i < attr_axis) {
        auto dimIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, opInput, dimIdx));
      } else if (i < attr_axis + indicesRank) {
        auto dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, i - attr_axis);
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, opIndices, dimIdx));
      } else {
        int64_t inpDim = i - indicesRank + 1;
        auto dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, inpDim);
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, opInput, dimIdx));
      }
    }
  }

  auto outBuffer =
      mlir::tensor::EmptyOp::create(rewriter, loc, outDatType, dynamicSizes);

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_type=*/mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opIndices},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      /*builder_callback*/
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::Value indVal = args[0];

        auto indIndex = mlir::arith::IndexCastOp::create(
            nest, nloc, nest.getIndexType(), indVal);
        auto zeroIndex = mlir::arith::ConstantIndexOp::create(nest, nloc, 0);
        auto isNeg = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::slt, indIndex, zeroIndex);
        auto posIndex =
            mlir::arith::AddIOp::create(nest, nloc, indIndex, axisDimVal);
        auto realIndex = mlir::arith::SelectOp::create(nest, nloc, isNeg,
                                                       posIndex, indIndex);
        // n-dim extract coordinate vector
        llvm::SmallVector<mlir::Value> inpCoords;
        inpCoords.reserve(inputRank);

        for (int64_t m = 0; m < inputRank; ++m) {
          if (m < attr_axis) {
            inpCoords.push_back(mlir::linalg::IndexOp::create(
                nest, nloc, static_cast<uint64_t>(m)));
          } else if (m == attr_axis) {
            inpCoords.push_back(realIndex);
          } else {
            uint64_t loopDim = static_cast<uint64_t>(m + indicesRank - 1);
            inpCoords.push_back(
                mlir::linalg::IndexOp::create(nest, nloc, loopDim));
          }
        }
        auto extracted =
            mlir::tensor::ExtractOp::create(nest, nloc, opInput, inpCoords);

        mlir::linalg::YieldOp::create(nest, nloc, extracted.getResult());
      });

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
