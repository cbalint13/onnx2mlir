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
 * \file src/conversion/passes/onnx_to_linalg/unaries.cpp
 * \brief ONNX Unary operations to Linalg lowering
 */

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

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
OnnxToLinalg_UnaryOps(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  mlir::Value opInput = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  // value checks
  if (mlir::dyn_cast<mlir::RankedTensorType>(inpDatType).getShape() !=
      mlir::dyn_cast<mlir::RankedTensorType>(outDatType).getShape())
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " input and output shapes are different";

  /*
   * Attributes
   */

  // alpha
  double attr_alpha = 1.0;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
    attr_alpha = attr.getValueAsDouble();
  // detect_positive
  int64_t attr_detect_pos = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("detect_positive"))
    attr_detect_pos = attr.getInt();
  // detect_negative
  int64_t attr_detect_neg = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("detect_negative"))
    attr_detect_neg = attr.getInt();

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

  auto genericOpBuilder = [&](/*op_builder*/ mlir::OpBuilder &nest,
                              /*src_location*/ mlir::Location nloc,
                              /*value_args*/ mlir::ValueRange args) {
    mlir::Value out;
    mlir::Value inp = args[0];

    // linalg converted source element type
    mlir::Type inpElmType = inpDatType.getElementType();

    if (opNameBeginsWith(opName, "Abs")) {
      if (inpElmType.isFloat())
        out = mlir::math::AbsFOp::create(nest, nloc, inp);
      else
        out = mlir::math::AbsIOp::create(nest, nloc, inp);
    }
    if (opNameBeginsWith(opName, "Acos"))
      out = mlir::math::AcosOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Acosh"))
      out = mlir::math::AcoshOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Asin"))
      out = mlir::math::AsinOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Asinh"))
      out = mlir::math::AsinhOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Atan"))
      out = mlir::math::AtanOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Atanh"))
      out = mlir::math::AtanhOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Ceil"))
      out = mlir::math::CeilOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Cos"))
      out = mlir::math::CosOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Cosh"))
      out = mlir::math::CoshOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Elu")) {
      if (inpElmType.isFloat()) {
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, attr_alpha));
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto cnd = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OGE, inp, c0);
        auto exp = mlir::math::ExpOp::create(nest, nloc, inp);
        auto sub = mlir::arith::SubFOp::create(nest, nloc, exp, c1);
        auto neg = mlir::arith::MulFOp::create(nest, nloc, cA, sub);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, inp, neg);
      } else {
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc,
            nest.getIntegerAttr(inpElmType, static_cast<int>(attr_alpha)));
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto cnd = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sge, inp, c0);
        auto exp = mlir::math::ExpOp::create(nest, nloc, inp);
        auto sub = mlir::arith::SubIOp::create(nest, nloc, exp, c1);
        auto neg = mlir::arith::MulIOp::create(nest, nloc, cA, sub);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, inp, neg);
      }
    }
    if (opNameBeginsWith(opName, "Erf"))
      out = mlir::math::ErfOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Exp"))
      out = mlir::math::ExpOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Floor"))
      out = mlir::math::FloorOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "HardSwish")) {
      if (inpElmType.isFloat()) {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.0));
        auto c3 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 3.0));
        auto c6 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 6.0));
        auto xPlus3 = mlir::arith::AddFOp::create(nest, nloc, inp, c3);
        auto condPos = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OGT, xPlus3, c0);
        auto max0 =
            mlir::arith::SelectOp::create(nest, nloc, condPos, xPlus3, c0);
        auto condLimit = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OLT, max0, c6);
        auto relu6_arg =
            mlir::arith::SelectOp::create(nest, nloc, condLimit, max0, c6);
        auto numerator =
            mlir::arith::MulFOp::create(nest, nloc, inp, relu6_arg);
        out = mlir::arith::DivFOp::create(nest, nloc, numerator, c6);
      } else {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto c3 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 3));
        auto c6 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 6));
        auto xPlus3 = mlir::arith::AddIOp::create(nest, nloc, inp, c3);
        auto condPos = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sgt, xPlus3, c0);
        auto max0 =
            mlir::arith::SelectOp::create(nest, nloc, condPos, xPlus3, c0);
        auto condLimit = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::slt, max0, c6);
        auto relu6_arg =
            mlir::arith::SelectOp::create(nest, nloc, condLimit, max0, c6);
        auto numerator =
            mlir::arith::MulIOp::create(nest, nloc, inp, relu6_arg);
        out = mlir::arith::DivSIOp::create(nest, nloc, numerator, c6);
      }
    }
    if (opNameBeginsWith(opName, "Identity"))
      out = inp;
    if (opNameBeginsWith(opName, "IsInf")) {
      auto fltType = mlir::cast<mlir::FloatType>(inpDatType.getElementType());
      const auto &semantics = fltType.getFloatSemantics();
      mlir::Value isPosInf = nullptr;
      mlir::Value isNegInf = nullptr;
      if (attr_detect_pos) {
        auto posInfVal = llvm::APFloat::getInf(semantics, /*isNegative=*/false);
        auto posInfCst = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(fltType, posInfVal));
        isPosInf = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OEQ, inp, posInfCst);
      }
      if (attr_detect_neg) {
        auto negInfVal = llvm::APFloat::getInf(semantics, /*isNegative=*/true);
        auto negInfCst = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(fltType, negInfVal));
        isNegInf = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OEQ, inp, negInfCst);
      }
      if (attr_detect_pos && attr_detect_neg) {
        out = mlir::arith::OrIOp::create(nest, nloc, isPosInf, isNegInf);
      } else if (attr_detect_pos) {
        out = isPosInf;
      } else if (attr_detect_neg) {
        out = isNegInf;
      } else {
        out = mlir::arith::ConstantOp::create(nest, nloc,
                                              nest.getBoolAttr(false));
      }
    }
    if (opNameBeginsWith(opName, "IsNaN"))
      out = mlir::arith::CmpFOp::create(
          nest, nloc, mlir::arith::CmpFPredicate::UNO, inp, inp);
    if (opNameBeginsWith(opName, "Log"))
      out = mlir::math::LogOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Neg")) {
      if (inpElmType.isFloat()) {
        out = mlir::arith::NegFOp::create(nest, nloc, inp);
      } else if (inpElmType.isInteger()) {
        mlir::Value c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        out = mlir::arith::SubIOp::create(nest, nloc, c0, inp);
      }
    }
    if (opNameBeginsWith(opName, {"Not", "BitwiseNot"})) {
      if (inpElmType.isInteger()) {
        int bitW = mlir::cast<mlir::IntegerType>(inpElmType).getWidth();
        auto ones = llvm::APInt::getAllOnes(bitW);
        auto allOnes = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, ones));
        out = mlir::arith::XOrIOp::create(nest, nloc, inp, allOnes);
      }
    }
    if (opNameBeginsWith(opName, "Reciprocal")) {
      if (inpElmType.isFloat()) {
        mlir::Value c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        out = mlir::arith::DivFOp::create(nest, nloc, c1, inp);
      }
    }
    if (opNameBeginsWith(opName, "Relu")) {
      if (inpElmType.isFloat()) {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.0));
        auto cnd = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OGE, inp, c0);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, inp, c0);
      } else if (inpElmType.isInteger()) {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto cnd = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sge, inp, c0);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, inp, c0);
      }
    }
    if (opNameBeginsWith(opName, "Round"))
      out = mlir::math::RoundOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Sigmoid")) {
      if (inpElmType.isFloat()) {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto negX = mlir::arith::NegFOp::create(nest, nloc, inp);
        auto expNegX = mlir::math::ExpOp::create(nest, nloc, negX);
        auto denom = mlir::arith::AddFOp::create(nest, nloc, c1, expNegX);
        out = mlir::arith::DivFOp::create(nest, nloc, c1, denom);
      } else {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto negX = mlir::arith::SubIOp::create(nest, nloc, c0, inp);
        auto expNegX = mlir::math::ExpOp::create(nest, nloc, negX);
        auto denom = mlir::arith::AddIOp::create(nest, nloc, c1, expNegX);
        out = mlir::arith::DivSIOp::create(nest, nloc, c1, denom);
      }
    }
    if (opNameBeginsWith(opName, "Sign")) {
      mlir::Value c0, cPos1, cNeg1, cndPos, cndNeg;
      if (inpElmType.isFloat()) {
        c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.0));
        cPos1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        cNeg1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, -1.0));
        cndPos = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OGT, inp, c0);
        cndNeg = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OLT, inp, c0);
      } else if (inpElmType.isInteger()) {
        c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        cPos1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        cNeg1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, -1));
        cndPos = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sgt, inp, c0);
        cndNeg = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::slt, inp, c0);
      }
      auto resIfNeg =
          mlir::arith::SelectOp::create(nest, nloc, cndNeg, cNeg1, c0);
      out = mlir::arith::SelectOp::create(nest, nloc, cndPos, cPos1, resIfNeg);
    }
    if (opNameBeginsWith(opName, "Sin"))
      out = mlir::math::SinOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Sinh"))
      out = mlir::math::SinhOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Softplus")) {
      if (inpElmType.isFloat()) {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto expX = mlir::math::ExpOp::create(nest, nloc, inp);
        auto logArg = mlir::arith::AddFOp::create(nest, nloc, c1, expX);
        out = mlir::math::LogOp::create(nest, nloc, logArg);
      } else if (inpElmType.isInteger()) {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto expX = mlir::math::ExpOp::create(nest, nloc, inp);
        auto logArg = mlir::arith::AddIOp::create(nest, nloc, c1, expX);
        out = mlir::math::LogOp::create(nest, nloc, logArg);
      }
    }
    if (opNameBeginsWith(opName, "Softsign")) {
      if (inpElmType.isFloat()) {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto absX = mlir::math::AbsFOp::create(nest, nloc, inp);
        auto denom = mlir::arith::AddFOp::create(nest, nloc, c1, absX);
        out = mlir::arith::DivFOp::create(nest, nloc, inp, denom);
      } else if (inpElmType.isInteger()) {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto absX = mlir::math::AbsIOp::create(nest, nloc, inp);
        auto denom = mlir::arith::AddIOp::create(nest, nloc, c1, absX);
        out = mlir::arith::DivSIOp::create(nest, nloc, inp, denom);
      }
    }
    if (opNameBeginsWith(opName, "Sqrt"))
      out = mlir::math::SqrtOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Tan"))
      out = mlir::math::TanOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Tanh"))
      out = mlir::math::TanhOp::create(nest, nloc, inp);

    mlir::linalg::YieldOp::create(nest, nloc, out);
  };

  mlir::Value outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, loc, outDatType.getShape(), outDatType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter,
      /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInput},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      /*builder_callback*/ genericOpBuilder);

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
