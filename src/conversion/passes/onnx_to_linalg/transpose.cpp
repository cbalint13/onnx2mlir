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

  mlir::Location loc = op->getLoc();

  auto rank = inpType.getRank();

  auto permAttr = op->getAttr("perm");
  mlir::SmallVector<int64_t> perms, out_shape;
  mlir::SmallVector<mlir::Value> dynamic_dims;
  if (auto arrayPermAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(permAttr)) {
    for (auto attr : arrayPermAttr) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        int64_t dim_index = intAttr.getInt();
        perms.push_back(dim_index);

        int64_t dim_size = inpType.getShape()[dim_index];
        if (mlir::ShapedType::isDynamic(dim_size)) {
          mlir::Value dim_value =
              rewriter.create<mlir::tensor::DimOp>(loc, inp, dim_index);
          dynamic_dims.push_back(dim_value);
          out_shape.push_back(mlir::ShapedType::kDynamic);
        } else {
          out_shape.push_back(dim_size);
        }
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
      int64_t dim_index = rank - 1 - i;
      perms.push_back(dim_index);
      int64_t dim_size = inpType.getShape()[dim_index];

      if (mlir::ShapedType::isDynamic(dim_size)) {
        mlir::Value dim_value =
            rewriter.create<mlir::tensor::DimOp>(loc, inp, dim_index);
        dynamic_dims.push_back(dim_value);
        out_shape.push_back(mlir::ShapedType::kDynamic);
      } else {
        out_shape.push_back(dim_size);
      }
    }
  }
  auto permsAttr = rewriter.getDenseI64ArrayAttr(perms);

  auto outType =
      mlir::RankedTensorType::get(out_shape, inpType.getElementType());
  auto outBuff =
      rewriter.create<mlir::tensor::EmptyOp>(loc, outType, dynamic_dims);

  auto transOp =
      rewriter.create<mlir::linalg::TransposeOp>(loc, inp, outBuff, permsAttr);

  // Tag for transform optimization
  transOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, transOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
