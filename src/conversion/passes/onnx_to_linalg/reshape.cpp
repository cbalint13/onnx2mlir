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

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));
  auto opInputShp = (op->getNumOperands() > 1 &&
                     !mlir::isa<mlir::NoneType>(op->getOperand(1).getType()))
                        ? convRewriter.getRemappedValue(op->getOperand(1))
                        : nullptr;

  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  /*
   * Attributes
   */

  // shape
  llvm::SmallVector<int64_t> attr_shape;
  if (auto a = op->getAttrOfType<mlir::DenseI64ArrayAttr>("shape")) {
    attr_shape = llvm::to_vector(a.asArrayRef());
  } else if (auto a = op->getAttrOfType<mlir::ArrayAttr>("shape")) {
    for (auto i : a.getAsRange<mlir::IntegerAttr>())
      attr_shape.push_back(i.getInt());
  } else if (auto a = op->getAttrOfType<mlir::DenseIntElementsAttr>("shape")) {
    attr_shape = llvm::to_vector(a.getValues<int64_t>());
  }
  if (!opInputShp && attr_shape.empty())
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " missing or invalid shape attribute";

  /*
   *  Linalg ops staging
   */

  if (!opInputShp) {
    auto shapedType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(attr_shape.size())}, rewriter.getI64Type());
    auto shapeBuffer = mlir::DenseElementsAttr::get(
        shapedType, llvm::ArrayRef<int64_t>(attr_shape));
    opInputShp = mlir::arith::ConstantOp::create(rewriter, loc, shapeBuffer);
  }

  auto reshapeOp = mlir::tensor::ReshapeOp::create(rewriter, loc, outDatType,
                                                   opInput, opInputShp);

  reshapeOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, reshapeOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
