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

  mlir::Value inp = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value ind = convRewriter.getRemappedValue(op->getOperand(1));

  // Retrieve input type descriptors
  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto indType = mlir::dyn_cast<mlir::RankedTensorType>(ind.getType());
  if (!inpType || !indType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " requires ranked tensor inputs");
  }

  int64_t inpRank = inpType.getRank();
  int64_t indRank = indType.getRank();

  if (inpRank < 1) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " input operand rank must be >= 1");
  }

  // Retrieve 'axis' attribute (default = 0)
  int64_t axis = 0;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis")) {
    axis = axisAttr.getInt();
  }
  // Normalize negative axis index
  if (axis < 0) {
    axis += inpRank;
  }
  if (axis < 0 || axis >= inpRank) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " axis attribute out of valid bounds");
  }

  // Calculate output rank: q + (r - 1)
  int64_t outRank = indRank + inpRank - 1;

  // Retrieve expected result type
  mlir::Value res = op->getResult(0);
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(
      typeConverter->convertType(res.getType()));
  if (!resType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " result must be a ranked tensor");
  }

  // Query runtime dimension size of gathered axis
  mlir::Value axisConst =
      mlir::arith::ConstantIndexOp::create(rewriter, loc, axis);
  mlir::Value axisDimVal =
      mlir::tensor::DimOp::create(rewriter, loc, inp, axisConst);

  // Collect dynamic dimensions for empty init output tensor
  llvm::SmallVector<mlir::Value> dynamicSizes;
  for (int64_t i = 0; i < outRank; ++i) {
    if (resType.isDynamicDim(i)) {
      if (i < axis) {
        mlir::Value dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, inp, dimIdx));
      } else if (i < axis + indRank) {
        mlir::Value dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, i - axis);
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, ind, dimIdx));
      } else {
        int64_t inpDim = i - indRank + 1;
        mlir::Value dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, inpDim);
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, inp, dimIdx));
      }
    }
  }

  // Create output initial destination buffer
  auto outBuff =
      mlir::tensor::EmptyOp::create(rewriter, loc, resType, dynamicSizes);

  // Define indexing map for indices input operand
  llvm::SmallVector<mlir::AffineExpr> indExprs;
  indExprs.reserve(indRank);
  for (int64_t i = 0; i < indRank; ++i) {
    indExprs.push_back(rewriter.getAffineDimExpr(axis + i));
  }
  auto indMap =
      mlir::AffineMap::get(outRank, 0, indExprs, rewriter.getContext());

  // Define identity map for output result tensor
  auto outMap = rewriter.getMultiDimIdentityMap(outRank);
  llvm::SmallVector<mlir::AffineMap> indexingMaps = {indMap, outMap};

  // Define parallel iteration domain
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      outRank, mlir::utils::IteratorType::parallel);

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/mlir::TypeRange{resType},
      /*inputs=*/mlir::ValueRange{ind},
      /*outputs=*/mlir::ValueRange{outBuff},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location nstLoc,
          mlir::ValueRange blockArgs) {
        mlir::Value indVal = blockArgs[0];

        // Cast index integer to MLIR Index type
        mlir::Value indIndex = mlir::arith::IndexCastOp::create(
            b, nstLoc, b.getIndexType(), indVal);

        // Normalize negative lookup indices (if index < 0: index += axis_dim)
        mlir::Value zeroIndex =
            mlir::arith::ConstantIndexOp::create(b, nstLoc, 0);
        mlir::Value isNeg = mlir::arith::CmpIOp::create(
            b, nstLoc, mlir::arith::CmpIPredicate::slt, indIndex, zeroIndex);

        mlir::Value posIndex =
            mlir::arith::AddIOp::create(b, nstLoc, indIndex, axisDimVal);
        mlir::Value realIndex =
            mlir::arith::SelectOp::create(b, nstLoc, isNeg, posIndex, indIndex);

        // Assemble N-dimensional extract coordinate vector for input tensor
        llvm::SmallVector<mlir::Value> inpCoords;
        inpCoords.reserve(inpRank);

        for (int64_t m = 0; m < inpRank; ++m) {
          if (m < axis) {
            inpCoords.push_back(mlir::linalg::IndexOp::create(
                b, nstLoc, static_cast<uint64_t>(m)));
          } else if (m == axis) {
            inpCoords.push_back(realIndex);
          } else {
            uint64_t loopDim = static_cast<uint64_t>(m + indRank - 1);
            inpCoords.push_back(
                mlir::linalg::IndexOp::create(b, nstLoc, loopDim));
          }
        }

        // Dynamically extract scalar value from input tensor
        mlir::Value extracted =
            mlir::tensor::ExtractOp::create(b, nstLoc, inp, inpCoords);

        // Yield scalar to output element position
        mlir::linalg::YieldOp::create(b, nstLoc, extracted);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));
  rewriter.replaceOp(op, genericOp.getResult(0));

  return mlir::success();
}

} // namespace onnx2mlir::dialect
