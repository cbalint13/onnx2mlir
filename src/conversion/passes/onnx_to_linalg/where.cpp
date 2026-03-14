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
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_WhereOp(mlir::Operation *op,
                                         mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  if (op->getNumOperands() != 3) {
    return mlir::emitError(loc, opName + " expected 3 operands");
  }

  mlir::Value cond = op->getOperand(0);
  mlir::Value x = op->getOperand(1);
  mlir::Value y = op->getOperand(2);
  mlir::Value res = op->getResult(0);

  auto condType = mlir::dyn_cast<mlir::RankedTensorType>(cond.getType());
  auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
  auto yType = mlir::dyn_cast<mlir::RankedTensorType>(y.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!condType || !xType || !yType || !resType) {
    return mlir::emitError(loc, opName + " operand must be ranked tensor type");
  }

  int64_t resRank = resType.getRank();

  // Create output buffer
  mlir::Value outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), resType.getElementType());

  // Define indexing maps for broadcasting
  mlir::Builder builder(op->getContext());
  mlir::AffineExpr zero = builder.getAffineConstantExpr(0);

  auto getIndexingMap = [&](mlir::RankedTensorType type) {
    llvm::SmallVector<mlir::AffineExpr, 4> exprs;
    int64_t rank = type.getRank();
    for (unsigned i = 0; i < resRank; ++i) {
      int64_t dimIdx = rank - (resRank - i);
      if (dimIdx >= 0) {
        if (type.getDimSize(dimIdx) == 1)
          exprs.push_back(zero);
        else
          exprs.push_back(builder.getAffineDimExpr(i));
      }
    }
    return mlir::AffineMap::get(resRank, 0, exprs, builder.getContext());
  };

  llvm::SmallVector<mlir::AffineMap, 4> idxMaps;
  idxMaps.push_back(getIndexingMap(condType));
  idxMaps.push_back(getIndexingMap(xType));
  idxMaps.push_back(getIndexingMap(yType));
  idxMaps.push_back(rewriter.getMultiDimIdentityMap(resRank));

  // Create linalg.generic
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      resRank, mlir::utils::IteratorType::parallel);

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, resType, mlir::ValueRange{cond, x, y}, // Inputs
      mlir::ValueRange{outBuff},                            // Output init
      idxMaps, iterators,
      [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
        // args[0]: cond (i1), args[1]: x, args[2]: y
        mlir::Value selected =
            mlir::arith::SelectOp::create(nest, l, args[0], args[1], args[2]);
        mlir::linalg::YieldOp::create(nest, l, selected);
      });

  // Set transform tag for downstream optimization
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);
  return mlir::success();
}

} // namespace onnx2mlir::dialect
