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
 * \file src/conversion/passes/onnx_to_linalg/binaries.cpp
 * \brief ONNX Binary operations to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_BinaryOps(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                       const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  mlir::Value opInput0 = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value opInput1 = convRewriter.getRemappedValue(op->getOperand(1));
  mlir::Value opOutput = op->getResult(0);

  auto lhsDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput0.getType());
  auto rhsDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput1.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(
      typeConverter->convertType(opOutput.getType()));

  auto dstDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());
  auto bcastShpType = getBroadcastShape(lhsDatType, rhsDatType);

  int64_t outRank = outDatType.getRank();

  // value checks
  if ((!lhsDatType) || (!rhsDatType))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " operands must be tensor type";
  if (lhsDatType.getElementType() != rhsDatType.getElementType())
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " operands element type are different";
  if (!outDatType)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " result must be a tensor type";
  if (!bcastShpType)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " operands are not broadcastable";
  if ((bcastShpType) && (outDatType != bcastShpType))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " result not match operands broadcast";
  /*
   * Attributes
   */

  // fmod
  int64_t attr_fmod = 0;
  if (auto fmodAttr = op->getAttrOfType<mlir::IntegerAttr>("fmod"))
    attr_fmod = fmodAttr.getInt();

  /*
   *  Affine mappings
   */

  auto getBroadcastMap = [&](mlir::RankedTensorType operType) {
    int64_t operRank = operType.getRank();
    int64_t rankDiff = outRank - operRank;

    llvm::SmallVector<mlir::AffineExpr, 4> exprs;
    exprs.reserve(operRank);

    for (int64_t i = 0; i < operRank; ++i) {
      if (operType.getDimSize(i) == 1)
        exprs.push_back(rewriter.getAffineConstantExpr(0));
      else
        exprs.push_back(rewriter.getAffineDimExpr(rankDiff + i));
    }
    return mlir::AffineMap::get(outRank, 0, exprs, rewriter.getContext());
  };

  auto lhsBroadcastMap = getBroadcastMap(lhsDatType);
  auto rhsBroadcastMap = getBroadcastMap(rhsDatType);
  auto outputIdentityMap = rewriter.getMultiDimIdentityMap(outRank);

  llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {
      lhsBroadcastMap, rhsBroadcastMap, outputIdentityMap};

  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
      outDatType.getRank(), mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  auto genericOpBuilder = [&](/*op_builder*/ mlir::OpBuilder &nest,
                              /*src_location*/ mlir::Location nloc,
                              /*value_args*/ mlir::ValueRange args) {
    mlir::Value out;
    mlir::Value lhs = args[0];
    mlir::Value rhs = args[1];

    // onnx original result element type
    mlir::Type dstElmType = dstDatType.getElementType();
    // linalg converted result element type
    mlir::Type outElmType = outDatType.getElementType();

    bool isInteger = outElmType.isInteger();

    if (opNameBeginsWith(opName, "Add")) {
      if (isInteger)
        out = mlir::arith::AddIOp::create(nest, nloc, lhs, rhs);
      else
        out = mlir::arith::AddFOp::create(nest, nloc, lhs, rhs);
    } else if (opNameBeginsWith(opName, {"BitwiseAnd", "And"})) {
      out = mlir::arith::AndIOp::create(nest, nloc, lhs, rhs);
    } else if (opNameBeginsWith(opName, {"BitwiseOr", "Or"})) {
      out = mlir::arith::OrIOp::create(nest, nloc, lhs, rhs);
    } else if (opNameBeginsWith(opName, {"BitwiseXor", "Xor"})) {
      out = mlir::arith::XOrIOp::create(nest, nloc, lhs, rhs);
    } else if (opNameBeginsWith(opName, "Div")) {
      if (isInteger && dstElmType.isUnsignedInteger())
        out = mlir::arith::DivUIOp::create(nest, nloc, lhs, rhs);
      else if (isInteger)
        out = mlir::arith::DivSIOp::create(nest, nloc, lhs, rhs);
      else
        out = mlir::arith::DivFOp::create(nest, nloc, lhs, rhs);
    } else if (opNameBeginsWith(opName, "Mod")) {
      if (isInteger && dstElmType.isUnsignedInteger()) {
        out = mlir::arith::RemUIOp::create(nest, nloc, lhs, rhs);
      } else if (isInteger && attr_fmod == 1) {
        out = mlir::arith::RemSIOp::create(nest, nloc, lhs, rhs);
      } else if (isInteger && attr_fmod == 0) {
        // floor modulo ((a % b) + b) % b
        auto rem = mlir::arith::RemSIOp::create(nest, nloc, lhs, rhs);
        auto add = mlir::arith::AddIOp::create(nest, nloc, rem, rhs);
        out = mlir::arith::RemSIOp::create(nest, nloc, add, rhs);
      } else {
        out = mlir::arith::RemFOp::create(nest, nloc, lhs, rhs);
      }
    } else if (opNameBeginsWith(opName, "Mul")) {
      if (isInteger)
        out = mlir::arith::MulIOp::create(nest, nloc, lhs, rhs);
      else
        out = mlir::arith::MulFOp::create(nest, nloc, lhs, rhs);
    } else if (opNameBeginsWith(opName, "Pow")) {
      if (isInteger)
        out = mlir::math::IPowIOp::create(nest, nloc, lhs, rhs);
      else
        out = mlir::math::PowFOp::create(nest, nloc, lhs, rhs);
    } else if (opNameBeginsWith(opName, "Sub")) {
      if (isInteger)
        out = mlir::arith::SubIOp::create(nest, nloc, lhs, rhs);
      else
        out = mlir::arith::SubFOp::create(nest, nloc, lhs, rhs);
    }
    mlir::linalg::YieldOp::create(nest, nloc, out);
  };

  mlir::Value outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, loc, outDatType.getShape(), outDatType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter,
      /*src_location*/ loc,
      /*result_type*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInput0, opInput1},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      /*builder_callback*/ genericOpBuilder);

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp.getResults());

  return mlir::success();
}

} // namespace onnx2mlir::dialect
