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
 * \file src/conversion/passes/onnx_to_linalg/compare_binary.cpp
 * \brief ONNX Comparison binary operations to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_CompBinaryOps(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                           const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  mlir::Value opInput0 = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value opInput1 = convRewriter.getRemappedValue(op->getOperand(1));
  mlir::Value opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto lhsDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput0.getType());
  auto rhsDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput1.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto bcastShpType = getBroadcastShape(lhsDatType, rhsDatType);

  int64_t outputRank = outDatType.getRank();

  // value checks
  if (!outDatType.getElementType().isInteger(1))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " result must have i1 (bool) element type";
  if (!bcastShpType)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " operands are not broadcastable";
  if ((bcastShpType) && (outDatType.getShape() != bcastShpType.getShape()))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " result not match operands broadcast shape";

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

  auto lhsBroadcastMap = getBroadcastMap(lhsDatType);
  auto rhsBroadcastMap = getBroadcastMap(rhsDatType);
  auto outputIdentityMap = rewriter.getMultiDimIdentityMap(outputRank);

  llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {
      lhsBroadcastMap, rhsBroadcastMap, outputIdentityMap};

  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
      outDatType.getRank(), mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  mlir::Value outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, op->getLoc(), outDatType.getShape(),
      outDatType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_type*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInput0, opInput1},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::Value out;
        mlir::Value lhs = args[0];
        mlir::Value rhs = args[1];
        if (lhsDatType.getElementType().isFloat()) {
          if (opNameBeginsWith(opName, "Equal"))
            out = mlir::arith::CmpFOp::create(
                nest, nloc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
          if (opNameBeginsWith(opName, "Greater"))
            out = mlir::arith::CmpFOp::create(
                nest, nloc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
          if (opNameBeginsWith(opName, "GreaterOrEqual"))
            out = mlir::arith::CmpFOp::create(
                nest, nloc, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
          if (opNameBeginsWith(opName, "Less"))
            out = mlir::arith::CmpFOp::create(
                nest, nloc, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
          if (opNameBeginsWith(opName, "LessOrEqual"))
            out = mlir::arith::CmpFOp::create(
                nest, nloc, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
        } else {
          if (opNameBeginsWith(opName, "Equal"))
            out = mlir::arith::CmpIOp::create(
                nest, nloc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
          if (opNameBeginsWith(opName, "Greater"))
            out = mlir::arith::CmpIOp::create(
                nest, nloc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
          if (opNameBeginsWith(opName, "GreaterOrEqual"))
            out = mlir::arith::CmpIOp::create(
                nest, nloc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
          if (opNameBeginsWith(opName, "Less"))
            out = mlir::arith::CmpIOp::create(
                nest, nloc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
          if (opNameBeginsWith(opName, "LessOrEqual"))
            out = mlir::arith::CmpIOp::create(
                nest, nloc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
        }
        mlir::linalg::YieldOp::create(nest, nloc, out);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
