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
 * \file src/conversion/passes/onnx_to_linalg/where.cpp
 * \brief ONNX Where operation to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_WhereOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                     const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  if (op->getNumOperands() != 3)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " is expecting 3 operands";

  auto opInputCond = convRewriter.getRemappedValue(op->getOperand(0));
  auto opInputX = convRewriter.getRemappedValue(op->getOperand(1));
  auto opInputY = convRewriter.getRemappedValue(op->getOperand(2));
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto cndDatType =
      mlir::dyn_cast<mlir::RankedTensorType>(opInputCond.getType());
  auto inXDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInputX.getType());
  auto inYDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInputY.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  int64_t outputRank = outDatType.getRank();

  /*
   *  Affine mappings
   */
  auto getBroadcastMap = [&](mlir::RankedTensorType operType) {
    int64_t operRank = operType.getRank();
    int64_t rankDiff = outputRank - operRank;

    llvm::SmallVector<mlir::AffineExpr, 4> exprs;
    exprs.reserve(operRank);

    for (auto [i, dim] : llvm::enumerate(operType.getShape()))
      exprs.push_back(dim == 1 ? rewriter.getAffineConstantExpr(0)
                               : rewriter.getAffineDimExpr(rankDiff + i));

    return mlir::AffineMap::get(outputRank, 0, exprs, rewriter.getContext());
  };

  auto cndBroadcastMap = getBroadcastMap(cndDatType);
  auto inXBroadcastMap = getBroadcastMap(inXDatType);
  auto inYBroadcastMap = getBroadcastMap(inYDatType);
  auto outputIdentityMap = rewriter.getMultiDimIdentityMap(outputRank);

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps = {
      cndBroadcastMap, inXBroadcastMap, inYBroadcastMap, outputIdentityMap};

  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      outputRank, mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  mlir::Value outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, loc, outDatType.getShape(), outDatType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_type*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInputCond, opInputX, opInputY},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::Value selected = mlir::arith::SelectOp::create(
            nest, nloc, args[0], args[1], args[2]);
        mlir::linalg::YieldOp::create(nest, nloc, selected);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
