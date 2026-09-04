/******************************************************************************
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
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_TransposeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                         const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

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

  // perm
  mlir::SmallVector<int64_t> attr_perms;
  if (auto permAttr = op->getAttrOfType<mlir::ArrayAttr>("perm")) {
    for (auto intAttr : permAttr.getAsRange<mlir::IntegerAttr>()) {
      if (!intAttr)
        return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
               << opName << " 'perm' array contains non-integer values";
      attr_perms.push_back(intAttr.getInt());
    }
    if (static_cast<int64_t>(attr_perms.size()) != inputRank)
      return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
             << opName << " 'perm' array size mismatch with input rank";
  } else {
    for (int64_t i = inputRank - 1; i >= 0; --i)
      attr_perms.push_back(i);
  }

  /*
   *  Linalg ops staging
   */

  mlir::SmallVector<int64_t> outShape;
  mlir::SmallVector<mlir::Value> dynDims;
  outShape.reserve(inputRank);

  for (int64_t dim_idx : attr_perms) {
    int64_t dim_size = inpDatType.getShape()[dim_idx];
    outShape.push_back(dim_size);

    if (mlir::ShapedType::isDynamic(dim_size))
      dynDims.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, opInput, dim_idx));
  }

  auto outBuffer =
      mlir::tensor::EmptyOp::create(rewriter, loc, outDatType, dynDims);

  auto transOp = mlir::linalg::TransposeOp::create(
      rewriter, loc, opInput, outBuffer,
      rewriter.getDenseI64ArrayAttr(attr_perms));

  rewriter.replaceOp(op, transOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
