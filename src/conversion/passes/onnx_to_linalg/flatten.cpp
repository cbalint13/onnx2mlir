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
 * \file src/conversion/passes/onnx_to_linalg/flatten.cpp
 * \brief ONNX FlattenOp to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_FlattenOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                       const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto inputRank = inpDatType.getRank();

  /*
   * Attributes
   */

  // axis
  int64_t attr_axis = 1;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    attr_axis = axisAttr.getInt();
  if (attr_axis < 0)
    attr_axis += inputRank;
  attr_axis = std::clamp<int64_t>(attr_axis, 0, inputRank);

  /*
   *  Affine mappings
   */

  auto inpShape = inpDatType.getShape();
  llvm::SmallVector<mlir::AffineExpr> inpExprs(inputRank);

  // delinearize a dimension expression over index range [start, end)
  auto delinearize = [&](mlir::AffineExpr expr, int start, int end) {
    for (int i = end - 1; i >= start; --i) {
      inpExprs[i] = (i == start) ? expr : expr % inpShape[i];
      expr = expr.floorDiv(inpShape[i]);
    }
  };

  delinearize(rewriter.getAffineDimExpr(0), 0, attr_axis);
  delinearize(rewriter.getAffineDimExpr(1), attr_axis, inputRank);

  auto inpMap = mlir::AffineMap::get(2, 0, inpExprs, rewriter.getContext());
  auto outMap = rewriter.getMultiDimIdentityMap(2);

  llvm::SmallVector<mlir::AffineMap, 2> indexingMaps = {inpMap, outMap};

  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      2, mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  auto outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, loc, outDatType.getShape(), outDatType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_type*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInput},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      /*builder_callback*/
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::linalg::YieldOp::create(nest, nloc, args[0]);
      });

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
