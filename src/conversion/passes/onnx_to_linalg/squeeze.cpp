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
 * \file src/conversion/passes/onnx_to_linalg/squeeze.cpp
 * \brief ONNX Squeeze operation to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_SqueezeOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value data = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!dataType || !resType) {
    return mlir::emitError(loc, opName + " operand must be ranked tensor type");
  }

  int64_t dataRank = dataType.getRank();
  int64_t resRank = resType.getRank();

  if (resRank > dataRank) {
    return mlir::emitError(
        loc, opName + " result rank cannot be greater than input rank");
  }

  // Create output buffer
  mlir::Value outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), resType.getElementType());

  // Indexing maps for the generic op
  mlir::Builder builder(op->getContext());
  llvm::SmallVector<mlir::AffineExpr, 4> exprs;

  // Dimensions that are squeezed (size 1) are skipped in the output map.
  int64_t currResDim = 0;
  for (int64_t i = 0; i < dataRank; ++i) {
    int64_t dimSize = dataType.getDimSize(i);
    // Assume this dimension is preserved.
    if (currResDim < resRank &&
        (dimSize == resType.getDimSize(currResDim) ||
         dimSize == mlir::ShapedType::kDynamic ||
         resType.getDimSize(currResDim) == mlir::ShapedType::kDynamic)) {
      // Map the input dimension 'i' to the output dimension 'currResDim'
      exprs.push_back(builder.getAffineDimExpr(i));
      currResDim++;
    }
  }

  // Map exactly the number of result dimensions
  if (currResDim != resRank) {
    return mlir::emitError(
        loc, opName + " failed to map input to output shape correctly");
  }

  mlir::SmallVector<mlir::AffineMap, 2> idxMaps;
  // Input map: (d0, d1, ..., d_dataRank-1) -> (indices of preserved dims)
  // Since we are iterating over the input rank in the generic op,
  // the result map is identity on the output.
  idxMaps.push_back(rewriter.getMultiDimIdentityMap(dataRank));
  idxMaps.push_back(
      mlir::AffineMap::get(dataRank, 0, exprs, builder.getContext()));

  // Create linalg.generic
  mlir::SmallVector<mlir::utils::IteratorType> iterators(
      dataRank, mlir::utils::IteratorType::parallel);

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, resType, mlir::ValueRange{data}, // Input
      mlir::ValueRange{outBuff},                      // Output init
      idxMaps, iterators,
      [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
        mlir::linalg::YieldOp::create(nest, l, args[0]);
      });

  // Set transform tag for downstream optimization
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);
  return mlir::success();
}

} // namespace onnx2mlir::dialect
