/******************************************************************+************
 *
 * ONNX2MLIR (ONNX dialect mappings for composable optimizations)
 *
 * Authors:
 *     Cristian Balint <cristian dot balint at gmail dot com>
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
 * \file src/conversion/passes/onnx_to_linalg/transpose.cpp
 * \brief ONNX TransposeOp to Linalg lowering
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

mlir::LogicalResult OnnxToLinalg_TransposeOp(mlir::Operation *op,
                                             mlir::PatternRewriter &rewriter) {
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return rewriter.notifyMatchFailure(
        op, opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, opName + " result must be a ranked tensor type");
  }

  if (!inpType.hasStaticShape() || !resType.hasStaticShape()) {
    return rewriter.notifyMatchFailure(
        op, opName + " input and result must be static shaped");
  }

  mlir::Location loc = op->getLoc();

  auto rank = inpType.getRank();

  auto permAttr = op->getAttr("perm");
  mlir::SmallVector<int64_t> perms, out_shape;
  if (auto arrayPermAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(permAttr)) {
    for (auto attr : arrayPermAttr) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        perms.push_back(intAttr.getInt());
        out_shape.push_back(inpType.getShape()[intAttr.getInt()]);
      } else {
        return rewriter.notifyMatchFailure(
            op, opName + " 'perm' array contains non-integer values");
      }
    }
    if (static_cast<int64_t>(perms.size()) != rank) {
      return rewriter.notifyMatchFailure(
          op, opName + " 'perm' array size mismatch with input rank");
    }
  } else {
    // 'perm' attribute is not present
    // default to reversing dimensions
    for (int64_t i = 0; i < rank; ++i) {
      perms.push_back(rank - 1 - i);
    }
  }
  auto permsAttr = rewriter.getDenseI64ArrayAttr(perms);

  mlir::Value outBuff = rewriter.create<mlir::tensor::EmptyOp>(
      loc, out_shape, inpType.getElementType());

  auto TransOp =
      rewriter.create<mlir::linalg::TransposeOp>(loc, inp, outBuff, permsAttr);
  mlir::Value result = TransOp.getResult().front();

  rewriter.replaceOp(op, result);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
