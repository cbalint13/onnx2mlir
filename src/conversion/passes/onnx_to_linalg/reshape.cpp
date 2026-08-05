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
 * \file src/conversion/passes/onnx_to_linalg/reshape.cpp
 * \brief ONNX Reshape operation to Linalg lowering
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
OnnxToLinalg_ReshapeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                       const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  mlir::Value inp = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value res = convRewriter.getRemappedValue(op->getResult(0));
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  mlir::Value shapeOperand;

  if (op->getNumOperands() > 1) {
    shapeOperand = convRewriter.getRemappedValue(op->getOperand(1));
  } else {
    // Opset 1-4: shape is an INTS attribute
    llvm::SmallVector<int64_t> shapeValues;

    if (auto denseI64Attr =
            op->getAttrOfType<mlir::DenseI64ArrayAttr>("shape")) {
      shapeValues = llvm::to_vector(denseI64Attr.asArrayRef());
    } else if (auto arrayAttr = op->getAttrOfType<mlir::ArrayAttr>("shape")) {
      for (auto attr : arrayAttr) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
          shapeValues.push_back(intAttr.getInt());
        }
      }
    } else if (auto denseIntAttr =
                   op->getAttrOfType<mlir::DenseIntElementsAttr>("shape")) {
      for (auto val : denseIntAttr.getValues<mlir::APInt>()) {
        shapeValues.push_back(val.getSExtValue());
      }
    }

    if (shapeValues.empty()) {
      return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                             opName + " missing or invalid shape attribute");
    }

    auto shapedType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(shapeValues.size())}, rewriter.getI64Type());

    shapeOperand = mlir::arith::ConstantOp::create(
        rewriter, loc,
        mlir::DenseElementsAttr::get(shapedType,
                                     llvm::ArrayRef<int64_t>(shapeValues)));
  }

  auto reshapeOp = mlir::tensor::ReshapeOp::create(rewriter, loc, resType, inp,
                                                   shapeOperand);

  // Tag for transform dialect
  reshapeOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, reshapeOp.getResult());
  return mlir::success();
}

} // namespace onnx2mlir::dialect
