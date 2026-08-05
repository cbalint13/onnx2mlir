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
 * \file src/conversion/passes/onnx_to_linalg/squeeze.cpp
 * \brief ONNX Squeeze operation to Linalg lowering
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
OnnxToLinalg_SqueezeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
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

  int64_t inputRank = inpDatType.getRank();
  int64_t outputRank = outDatType.getRank();

  // checks
  if (outputRank > inputRank)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " result rank cannot be greater than input rank";

  /*
   *  Affine mappings
   */

  int64_t currResDim = 0;
  llvm::SmallVector<mlir::AffineExpr, 4> exprs;
  for (int64_t i = 0; i < inputRank && currResDim < outputRank; ++i) {
    int64_t inDimSize = inpDatType.getDimSize(i);
    int64_t outDimSize = outDatType.getDimSize(currResDim);
    if (inDimSize == outDimSize || mlir::ShapedType::isDynamic(inDimSize) ||
        mlir::ShapedType::isDynamic(outDimSize)) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
      currResDim++;
    }
  }

  if (currResDim != outputRank)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " failed to map input to output shape correctly";

  auto inpMap = rewriter.getMultiDimIdentityMap(inputRank);
  auto outMap = mlir::AffineMap::get(inputRank, 0, exprs, op->getContext());

  mlir::SmallVector<mlir::AffineMap, 2> indexingMaps = {inpMap, outMap};

  mlir::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      inputRank, mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  auto outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, loc, outDatType.getShape(), outDatType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInput},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::linalg::YieldOp::create(nest, nloc, args[0]);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
