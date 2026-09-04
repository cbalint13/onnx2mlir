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

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());

  int64_t inputRank = inpDatType.getRank();

  /*
   * Attributes
   */

  // start
  int64_t attr_start = 0;
  if (auto startAttr = op->getAttrOfType<mlir::IntegerAttr>("start"))
    attr_start = startAttr.getInt();
  if (attr_start < 0)
    attr_start += inputRank;
  attr_start = std::clamp<int64_t>(attr_start, 0, inputRank);

  // end
  int64_t attr_end = inputRank;
  if (auto endAttr = op->getAttrOfType<mlir::IntegerAttr>("end")) {
    attr_end = endAttr.getInt();
    if (attr_end < 0)
      attr_end += inputRank;
    attr_end = std::clamp<int64_t>(attr_end, 0, inputRank);
  }

  /*
   *  Linalg ops staging
   */

  // collect dims in range [start, end)
  llvm::SmallVector<mlir::Value> dimVals;
  if (attr_start < attr_end) {
    dimVals.reserve(attr_end - attr_start);
    for (int64_t i = attr_start; i < attr_end; ++i) {
      mlir::Value out;
      out = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
      out = mlir::tensor::DimOp::create(rewriter, loc, opInput, out);
      out = mlir::arith::IndexCastOp::create(rewriter, loc,
                                             rewriter.getI64Type(), out);
      dimVals.push_back(out);
    }
  }

  auto tgtDatType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(dimVals.size())}, rewriter.getI64Type());

  auto shapeOp =
      mlir::tensor::FromElementsOp::create(rewriter, loc, tgtDatType, dimVals);

  rewriter.replaceOp(op, shapeOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
