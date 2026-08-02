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
 * \brief ONNX Gather operation to Linalg dynamic indexing lowering pass
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

mlir::LogicalResult OnnxToLinalg_GatherOp(mlir::Operation *op,
                                          mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value data = op->getOperand(0);
  mlir::Value indices = op->getOperand(1);

  // Retrieve input type descriptors
  auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  auto indicesType = mlir::dyn_cast<mlir::RankedTensorType>(indices.getType());
  if (!dataType || !indicesType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " requires ranked tensor inputs");
  }

  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();

  if (dataRank < 1) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " data operand rank must be >= 1");
  }

  // Retrieve 'axis' attribute (default = 0)
  int64_t axis = 0;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis")) {
    axis = axisAttr.getInt();
  }
  // Normalize negative axis index
  if (axis < 0) {
    axis += dataRank;
  }
  if (axis < 0 || axis >= dataRank) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " axis attribute out of valid bounds");
  }

  // Calculate output rank: q + (r - 1)
  int64_t outRank = indicesRank + dataRank - 1;

  // Retrieve expected result type
  mlir::Value result = op->getResult(0);
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  if (!resultType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " result must be a ranked tensor");
  }

  // Query runtime dimension size of gathered axis
  mlir::Value axisConst =
      mlir::arith::ConstantIndexOp::create(rewriter, loc, axis);
  mlir::Value axisDimVal =
      mlir::tensor::DimOp::create(rewriter, loc, data, axisConst);

  // Collect dynamic dimensions for empty init output tensor
  llvm::SmallVector<mlir::Value> dynamicSizes;
  for (int64_t i = 0; i < outRank; ++i) {
    if (resultType.isDynamicDim(i)) {
      if (i < axis) {
        mlir::Value dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, data, dimIdx));
      } else if (i < axis + indicesRank) {
        mlir::Value dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, i - axis);
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, indices, dimIdx));
      } else {
        int64_t dataDim = i - indicesRank + 1;
        mlir::Value dimIdx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, dataDim);
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, data, dimIdx));
      }
    }
  }

  // Create output initial destination buffer
  auto initTensor =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType, dynamicSizes);

  // Define indexing map for indices input operand
  llvm::SmallVector<mlir::AffineExpr> indicesExprs;
  indicesExprs.reserve(indicesRank);
  for (int64_t i = 0; i < indicesRank; ++i) {
    indicesExprs.push_back(rewriter.getAffineDimExpr(axis + i));
  }
  auto indicesMap =
      mlir::AffineMap::get(outRank, 0, indicesExprs, rewriter.getContext());

  // Define identity map for output result tensor
  auto outMap = rewriter.getMultiDimIdentityMap(outRank);
  llvm::SmallVector<mlir::AffineMap> indexingMaps = {indicesMap, outMap};

  // Define parallel iteration domain
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      outRank, mlir::utils::IteratorType::parallel);

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/mlir::TypeRange{resultType},
      /*inputs=*/mlir::ValueRange{indices},
      /*outputs=*/mlir::ValueRange{initTensor},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location nestedLoc,
          mlir::ValueRange blockArgs) {
        mlir::Value indicesVal = blockArgs[0];

        // Cast index integer to MLIR Index type
        mlir::Value indicesIndex = mlir::arith::IndexCastOp::create(
            b, nestedLoc, b.getIndexType(), indicesVal);

        // Normalize negative lookup indices (if index < 0: index += axis_dim)
        mlir::Value zeroIndex =
            mlir::arith::ConstantIndexOp::create(b, nestedLoc, 0);
        mlir::Value isNeg = mlir::arith::CmpIOp::create(
            b, nestedLoc, mlir::arith::CmpIPredicate::slt, indicesIndex,
            zeroIndex);

        mlir::Value posIndex =
            mlir::arith::AddIOp::create(b, nestedLoc, indicesIndex, axisDimVal);
        mlir::Value realIndex = mlir::arith::SelectOp::create(
            b, nestedLoc, isNeg, posIndex, indicesIndex);

        // Assemble N-dimensional extract coordinate vector for data tensor
        llvm::SmallVector<mlir::Value> dataCoords;
        dataCoords.reserve(dataRank);

        for (int64_t m = 0; m < dataRank; ++m) {
          if (m < axis) {
            dataCoords.push_back(mlir::linalg::IndexOp::create(
                b, nestedLoc, static_cast<uint64_t>(m)));
          } else if (m == axis) {
            dataCoords.push_back(realIndex);
          } else {
            uint64_t loopDim = static_cast<uint64_t>(m + indicesRank - 1);
            dataCoords.push_back(
                mlir::linalg::IndexOp::create(b, nestedLoc, loopDim));
          }
        }

        // Dynamically extract scalar value from input data tensor
        mlir::Value extracted =
            mlir::tensor::ExtractOp::create(b, nestedLoc, data, dataCoords);

        // Yield scalar to output element position
        mlir::linalg::YieldOp::create(b, nestedLoc, extracted);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));
  rewriter.replaceOp(op, genericOp.getResult(0));

  return mlir::success();
}

} // namespace onnx2mlir::dialect
