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
 * \file src/conversion/passes/onnx_to_linalg/cast.cpp
 * \brief ONNX CastOp to Linalg lowering
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
OnnxToLinalg_CastOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto *ctx = rewriter.getContext();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  mlir::Value opInput = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto srcDatType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto dstDatType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // onnx original data types
  mlir::Type srcElmType = srcDatType.getElementType();
  // linalg converted data types (signless)
  mlir::Type inpElmType = inpDatType.getElementType();
  mlir::Type outElmType = outDatType.getElementType();

  // identity
  if (srcDatType == dstDatType) {
    rewriter.replaceOp(op, opInput);
    return mlir::success();
  }

  // value checks
  if (mlir::dyn_cast<mlir::RankedTensorType>(inpDatType).getShape() !=
      mlir::dyn_cast<mlir::RankedTensorType>(outDatType).getShape())
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " input and output shapes are different";

  /*
   * Attributes
   */

  auto toAttr = op->getAttr("to");
  if (!toAttr)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << "  missing 'to' attribute";

  mlir::Type tgtElmType = {};
  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(toAttr)) {
    tgtElmType = OnnxToMlir_dType(intAttr.getInt(), ctx);
  } else if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(toAttr)) {
    tgtElmType = OnnxToMlir_dType(strAttr.getValue().str(), ctx);
  } else {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " invalid 'to' attribute type";
  }
  if (!tgtElmType || mlir::dyn_cast_or_null<mlir::NoneType>(tgtElmType))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " unsupported `to` attribute value";

  // target 'to' data type
  auto tgtDatType = inpDatType.clone(tgtElmType);

  if (tgtDatType != dstDatType)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " 'to' data type not match the result type";

  /*
   *  Affine mappings
   */

  auto inpIdentityMap = rewriter.getMultiDimIdentityMap(inpDatType.getRank());
  auto outIdentityMap = rewriter.getMultiDimIdentityMap(outDatType.getRank());

  mlir::SmallVector<mlir::AffineMap, 2> indexingMaps;
  indexingMaps = {inpIdentityMap, outIdentityMap};

  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
      inpDatType.getRank(), mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  mlir::Value outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, loc, inpDatType.getShape(), outDatType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_type*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInput},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*inter_types*/ iteratorTypes,
      [&](/*op_builder*/ mlir::OpBuilder nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::Value out;
        mlir::Value inp = args[0];

        auto srcWidth = srcElmType.getIntOrFloatBitWidth();
        auto tgtWidth = tgtElmType.getIntOrFloatBitWidth();

        if (srcElmType == tgtElmType) {
          out = inp;
        } else if (tgtElmType.isInteger(1)) {
          // float -> bool
          if (srcElmType.isFloat()) {
            auto zero = mlir::arith::ConstantOp::create(
                nest, nloc, nest.getFloatAttr(srcElmType, 0.0));
            out = mlir::arith::CmpFOp::create(
                nest, nloc, mlir::arith::CmpFPredicate::UNE, inp, zero);
            // int -> bool
          } else if (srcElmType.isInteger()) {
            auto zero = mlir::arith::ConstantOp::create(
                nest, nloc, nest.getIntegerAttr(inpElmType, 0));
            out = mlir::arith::CmpIOp::create(
                nest, nloc, mlir::arith::CmpIPredicate::ne, inp, zero);
          }
        } else if (srcElmType.isFloat() && tgtElmType.isFloat()) {
          // float -> float
          if (srcWidth < tgtWidth) {
            out = mlir::arith::ExtFOp::create(nest, nloc, tgtElmType, inp);
          } else if (srcWidth > tgtWidth) {
            out = mlir::arith::TruncFOp::create(nest, nloc, tgtElmType, inp);
          }
        } else if (srcElmType.isInteger() && tgtElmType.isInteger()) {
          // int -> int
          if (srcWidth < tgtWidth) {
            if (srcElmType.isSignedInteger()) {
              out = mlir::arith::ExtSIOp::create(nest, nloc, outElmType, inp);
            } else {
              out = mlir::arith::ExtUIOp::create(nest, nloc, outElmType, inp);
            }
          } else if (srcWidth > tgtWidth) {
            out = mlir::arith::TruncIOp::create(nest, nloc, outElmType, inp);
          } else {
            // same bitwidth, different signedness
            out = mlir::arith::BitcastOp::create(nest, nloc, outElmType, inp);
          }
        } else if (srcElmType.isFloat() && tgtElmType.isInteger()) {
          // float -> int (non-bool)
          mlir::Type outElmType = nest.getIntegerType(tgtWidth);
          if (tgtElmType.isSignedInteger()) {
            out = mlir::arith::FPToSIOp::create(nest, nloc, outElmType, inp);
          } else {
            out = mlir::arith::FPToUIOp::create(nest, nloc, outElmType, inp);
          }
        } else if (srcElmType.isInteger() && tgtElmType.isFloat()) {
          // int -> float
          if (srcElmType.isSignedInteger()) {
            out = mlir::arith::SIToFPOp::create(nest, nloc, tgtElmType, inp);
          } else {
            out = mlir::arith::UIToFPOp::create(nest, nloc, tgtElmType, inp);
          }
        }

        mlir::linalg::YieldOp::create(nest, nloc, out);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
