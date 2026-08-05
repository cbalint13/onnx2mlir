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
 * \file src/conversion/passes/onnx_to_linalg/shape.cpp
 * \brief ONNX Shape operation to Linalg lowering
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
OnnxToLinalg_ShapeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                     const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  mlir::Value inp = convRewriter.getRemappedValue(op->getOperand(0));

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  if (!inpType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " requires ranked tensor input");
  }

  int64_t rank = inpType.getRank();

  // Handle optional 'start' attribute (default = 0)
  int64_t start = 0;
  if (auto startAttr = op->getAttrOfType<mlir::IntegerAttr>("start")) {
    start = startAttr.getInt();
  }
  // Normalize negative index relative to rank
  if (start < 0) {
    start += rank;
  }
  // Clamp start into valid range [0, rank]
  start = std::max<int64_t>(0, std::min<int64_t>(start, rank));

  // Handle optional 'end' attribute (default = rank)
  int64_t end = rank;
  if (auto endAttr = op->getAttrOfType<mlir::IntegerAttr>("end")) {
    end = endAttr.getInt();
    if (end < 0) {
      end += rank;
    }
    // Clamp end into valid range [0, rank]
    end = std::max<int64_t>(0, std::min<int64_t>(end, rank));
  }

  // Collect dimensions in range [start, end)
  llvm::SmallVector<mlir::Value> dimVals;
  if (start < end) {
    dimVals.reserve(end - start);
    for (int64_t i = start; i < end; ++i) {
      // Create index constant for dimension position
      auto indexVal = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);

      // Query dynamic dimension size from input tensor
      auto dimVal = mlir::tensor::DimOp::create(rewriter, loc, inp, indexVal);

      // Convert index type to i64 tensor element type
      auto dimI64 = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), dimVal);

      dimVals.push_back(dimI64);
    }
  }

  // Create target 1D int64 ranked tensor type
  auto resType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(dimVals.size())}, rewriter.getI64Type());

  // Construct tensor from individual element dimension values
  auto shapeOp =
      mlir::tensor::FromElementsOp::create(rewriter, loc, resType, dimVals);

  // Attach transform dialect target tag
  shapeOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, shapeOp.getResult());
  return mlir::success();
}

} // namespace onnx2mlir::dialect
