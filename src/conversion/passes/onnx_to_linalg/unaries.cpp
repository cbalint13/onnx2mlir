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

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

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
  if (opNameBeginsWith(opName, "Selu"))
    attr_alpha = 1.6732632423543772;
  if (opNameBeginsWith(opName, "HardSigmoid"))
    attr_alpha = 0.2;
  if (opNameBeginsWith(opName, "LeakyRelu"))
    attr_alpha = 0.01;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
    attr_alpha = attr.getValueAsDouble();
  // beta
  double attr_beta = 1.0;
  if (opNameBeginsWith(opName, "HardSigmoid"))
    attr_beta = 0.5;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("beta"))
    attr_beta = attr.getValueAsDouble();
  // approximate
  llvm::StringRef attr_approximate = "none";
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("approximate"))
    attr_approximate = attr.getValue();
  // detect_positive
  int64_t attr_detect_pos = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("detect_positive"))
    attr_detect_pos = attr.getInt();
  // detect_negative
  int64_t attr_detect_neg = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("detect_negative"))
    attr_detect_neg = attr.getInt();
  // gamma
  double attr_gamma = 1.0507009873554805;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("gamma"))
    attr_gamma = attr.getValueAsDouble();
  // seed
  float attr_seed = 0.0f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("seed"))
    attr_seed = attr.getValueAsDouble();
  // threshold
  double attr_threshold = 0.0;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("threshold"))
    attr_threshold = attr.getValueAsDouble();

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
    if (opNameBeginsWith(opName, "Bernoulli")) {
      auto i32Type = nest.getI32Type();
      auto f32Type = nest.getF32Type();
      mlir::Type outElmType = outDatType.getElementType();
      // unique per-element hash seed from spatial iteration indices
      int64_t seedInt = static_cast<int32_t>(attr_seed * 10000.0f);
      mlir::Value elementHash = mlir::arith::ConstantOp::create(
          nest, nloc, nest.getI32IntegerAttr(seedInt));
      int64_t rank = inpDatType.getRank();
      int32_t strideMultiplier = 31;
      int32_t currentMultiplier = 1;
      for (int64_t i = 0; i < rank; ++i) {
        auto idx = mlir::linalg::IndexOp::create(nest, nloc, i);
        auto idxI32 =
            mlir::arith::IndexCastOp::create(nest, nloc, i32Type, idx);
        auto multCst = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getI32IntegerAttr(currentMultiplier));
        auto scaledIdx =
            mlir::arith::MulIOp::create(nest, nloc, idxI32, multCst);
        elementHash =
            mlir::arith::AddIOp::create(nest, nloc, elementHash, scaledIdx);
        currentMultiplier *= strideMultiplier;
      }
      // advance state = (hash * 1664525 + 1013904223)
      auto cA = mlir::arith::ConstantOp::create(
          nest, nloc, nest.getI32IntegerAttr(1664525));
      auto cC = mlir::arith::ConstantOp::create(
          nest, nloc, nest.getI32IntegerAttr(1013904223));
      mlir::Value state =
          mlir::arith::MulIOp::create(nest, nloc, elementHash, cA);
      state = mlir::arith::AddIOp::create(nest, nloc, state, cC);
      // normalize integer state to uniform float u in range [0.0, 1.0)
      auto mask = mlir::arith::ConstantOp::create(
          nest, nloc, nest.getI32IntegerAttr(0x7FFFFFFF));
      auto positiveState = mlir::arith::AndIOp::create(nest, nloc, state, mask);
      auto floatState =
          mlir::arith::SIToFPOp::create(nest, nloc, f32Type, positiveState);
      auto cMax = mlir::arith::ConstantOp::create(
          nest, nloc, nest.getFloatAttr(f32Type, 2147483648.0f));
      auto rndUni = mlir::arith::DivFOp::create(nest, nloc, floatState, cMax);
      mlir::Value prob = inp;
      if (!inpElmType.isF32())
        prob = mlir::arith::ExtFOp::create(nest, nloc, f32Type, inp);
      // condition: u < p
      auto cond = mlir::arith::CmpFOp::create(
          nest, nloc, mlir::arith::CmpFPredicate::OLT, rndUni, prob);
      mlir::Value val1, val0;
      if (outElmType.isFloat()) {
        val1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(outElmType, 1.0));
        val0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(outElmType, 0.0));
      } else {
        val1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(outElmType, 1));
        val0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(outElmType, 0));
      }
      out = mlir::arith::SelectOp::create(nest, nloc, cond, val1, val0);
    }
    if (opNameBeginsWith(opName, "Binarizer")) {
      if (inpElmType.isFloat()) {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto cThresh = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, attr_threshold));
        auto cnd = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OGT, inp, cThresh);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, c1, c0);
      } else if (inpElmType.isInteger()) {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto cThresh = mlir::arith::ConstantOp::create(
            nest, nloc,
            nest.getIntegerAttr(inpElmType,
                                static_cast<int64_t>(attr_threshold)));
        auto cnd = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sgt, inp, cThresh);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, c1, c0);
      }
    }
    if (opNameBeginsWith(opName, "Ceil"))
      out = mlir::math::CeilOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Celu")) {
      if (inpElmType.isFloat()) {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, attr_alpha));
        auto cnd = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OGE, inp, c0);
        auto xDivAlpha = mlir::arith::DivFOp::create(nest, nloc, inp, cA);
        auto expVal = mlir::math::ExpOp::create(nest, nloc, xDivAlpha);
        auto sub1 = mlir::arith::SubFOp::create(nest, nloc, expVal, c1);
        auto negB = mlir::arith::MulFOp::create(nest, nloc, cA, sub1);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, inp, negB);
      } else {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc,
            nest.getIntegerAttr(inpElmType, static_cast<int64_t>(attr_alpha)));
        auto cnd = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sge, inp, c0);
        auto xDivAlpha = mlir::arith::DivSIOp::create(nest, nloc, inp, cA);
        auto expVal = mlir::math::ExpOp::create(nest, nloc, xDivAlpha);
        auto sub1 = mlir::arith::SubIOp::create(nest, nloc, expVal, c1);
        auto negB = mlir::arith::MulIOp::create(nest, nloc, cA, sub1);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, inp, negB);
      }
    }
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
    if (opNameBeginsWith(opName, "Gelu")) {
      if (inpElmType.isFloat()) {
        auto c0_5 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.5));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        if (attr_approximate == "tanh") {
          auto cSqrt2OverPi = mlir::arith::ConstantOp::create(
              nest, nloc, nest.getFloatAttr(inpElmType, 0.7978845608028654));
          auto cCoeff = mlir::arith::ConstantOp::create(
              nest, nloc, nest.getFloatAttr(inpElmType, 0.044715));
          auto x2 = mlir::arith::MulFOp::create(nest, nloc, inp, inp);
          auto x3 = mlir::arith::MulFOp::create(nest, nloc, x2, inp);
          auto coeffX3 = mlir::arith::MulFOp::create(nest, nloc, cCoeff, x3);
          auto poly = mlir::arith::AddFOp::create(nest, nloc, inp, coeffX3);
          auto tanhArg =
              mlir::arith::MulFOp::create(nest, nloc, cSqrt2OverPi, poly);
          auto tanhVal = mlir::math::TanhOp::create(nest, nloc, tanhArg);
          auto onePlusTanh =
              mlir::arith::AddFOp::create(nest, nloc, c1, tanhVal);
          auto xMulTanh =
              mlir::arith::MulFOp::create(nest, nloc, inp, onePlusTanh);
          out = mlir::arith::MulFOp::create(nest, nloc, c0_5, xMulTanh);
        } else {
          auto cInvSqrt2 = mlir::arith::ConstantOp::create(
              nest, nloc, nest.getFloatAttr(inpElmType, 0.7071067811865475));
          auto xScaled =
              mlir::arith::MulFOp::create(nest, nloc, inp, cInvSqrt2);
          auto erfVal = mlir::math::ErfOp::create(nest, nloc, xScaled);
          auto onePlusErf = mlir::arith::AddFOp::create(nest, nloc, c1, erfVal);
          auto xMulErf =
              mlir::arith::MulFOp::create(nest, nloc, inp, onePlusErf);
          out = mlir::arith::MulFOp::create(nest, nloc, c0_5, xMulErf);
        }
      }
    }
    if (opNameBeginsWith(opName, "HardSigmoid")) {
      if (inpElmType.isFloat()) {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, attr_alpha));
        auto cB = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, attr_beta));
        auto alphaX = mlir::arith::MulFOp::create(nest, nloc, cA, inp);
        auto linear = mlir::arith::AddFOp::create(nest, nloc, alphaX, cB);
        auto condGt0 = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OGT, linear, c0);
        auto max0 =
            mlir::arith::SelectOp::create(nest, nloc, condGt0, linear, c0);
        auto condLt1 = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OLT, max0, c1);
        out = mlir::arith::SelectOp::create(nest, nloc, condLt1, max0, c1);
      } else {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc,
            nest.getIntegerAttr(inpElmType, static_cast<int64_t>(attr_alpha)));
        auto cB = mlir::arith::ConstantOp::create(
            nest, nloc,
            nest.getIntegerAttr(inpElmType, static_cast<int64_t>(attr_beta)));
        auto alphaX = mlir::arith::MulIOp::create(nest, nloc, cA, inp);
        auto linear = mlir::arith::AddIOp::create(nest, nloc, alphaX, cB);
        auto condGt0 = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sgt, linear, c0);
        auto max0 =
            mlir::arith::SelectOp::create(nest, nloc, condGt0, linear, c0);
        auto condLt1 = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::slt, max0, c1);
        out = mlir::arith::SelectOp::create(nest, nloc, condLt1, max0, c1);
      }
    }
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
    if (opNameBeginsWith(opName, "LeakyRelu")) {
      if (inpElmType.isFloat()) {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.0));
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, attr_alpha));
        auto cnd = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OGE, inp, c0);
        auto negB = mlir::arith::MulFOp::create(nest, nloc, cA, inp);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, inp, negB);
      } else {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc,
            nest.getIntegerAttr(inpElmType, static_cast<int64_t>(attr_alpha)));
        auto cnd = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sge, inp, c0);
        auto negB = mlir::arith::MulIOp::create(nest, nloc, cA, inp);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, inp, negB);
      }
    }
    if (opNameBeginsWith(opName, "Log"))
      out = mlir::math::LogOp::create(nest, nloc, inp);
    if (opNameBeginsWith(opName, "Mish")) {
      if (inpElmType.isFloat()) {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto expX = mlir::math::ExpOp::create(nest, nloc, inp);
        auto logArg = mlir::arith::AddFOp::create(nest, nloc, c1, expX);
        auto softplus = mlir::math::LogOp::create(nest, nloc, logArg);
        auto tanhSoftplus = mlir::math::TanhOp::create(nest, nloc, softplus);
        out = mlir::arith::MulFOp::create(nest, nloc, inp, tanhSoftplus);
      } else if (inpElmType.isInteger()) {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto expX = mlir::math::ExpOp::create(nest, nloc, inp);
        auto logArg = mlir::arith::AddIOp::create(nest, nloc, c1, expX);
        auto softplus = mlir::math::LogOp::create(nest, nloc, logArg);
        auto tanhSoftplus = mlir::math::TanhOp::create(nest, nloc, softplus);
        out = mlir::arith::MulIOp::create(nest, nloc, inp, tanhSoftplus);
      }
    }
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
    if (opNameBeginsWith(opName, "Selu")) {
      if (inpElmType.isFloat()) {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 0.0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, attr_alpha));
        auto cG = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, attr_gamma));
        auto cnd = mlir::arith::CmpFOp::create(
            nest, nloc, mlir::arith::CmpFPredicate::OGE, inp, c0);
        auto posB = mlir::arith::MulFOp::create(nest, nloc, cG, inp);
        auto expX = mlir::math::ExpOp::create(nest, nloc, inp);
        auto sub = mlir::arith::SubFOp::create(nest, nloc, expX, c1);
        auto mulA = mlir::arith::MulFOp::create(nest, nloc, cA, sub);
        auto negB = mlir::arith::MulFOp::create(nest, nloc, cG, mulA);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, posB, negB);
      } else {
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto cA = mlir::arith::ConstantOp::create(
            nest, nloc,
            nest.getIntegerAttr(inpElmType, static_cast<int64_t>(attr_alpha)));
        auto cG = mlir::arith::ConstantOp::create(
            nest, nloc,
            nest.getIntegerAttr(inpElmType, static_cast<int64_t>(attr_gamma)));
        auto cnd = mlir::arith::CmpIOp::create(
            nest, nloc, mlir::arith::CmpIPredicate::sge, inp, c0);
        auto posB = mlir::arith::MulIOp::create(nest, nloc, cG, inp);
        auto expX = mlir::math::ExpOp::create(nest, nloc, inp);
        auto sub = mlir::arith::SubIOp::create(nest, nloc, expX, c1);
        auto mulAlpha = mlir::arith::MulIOp::create(nest, nloc, cA, sub);
        auto negB = mlir::arith::MulIOp::create(nest, nloc, cG, mulAlpha);
        out = mlir::arith::SelectOp::create(nest, nloc, cnd, posB, negB);
      }
    }
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
    if (opNameBeginsWith(opName, "Swish")) {
      if (inpElmType.isFloat()) {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, 1.0));
        auto cB = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getFloatAttr(inpElmType, attr_beta));
        auto betaX = mlir::arith::MulFOp::create(nest, nloc, cB, inp);
        auto negBetaX = mlir::arith::NegFOp::create(nest, nloc, betaX);
        auto expNegBetaX = mlir::math::ExpOp::create(nest, nloc, negBetaX);
        auto denom = mlir::arith::AddFOp::create(nest, nloc, c1, expNegBetaX);
        auto sigmoidVal = mlir::arith::DivFOp::create(nest, nloc, c1, denom);
        out = mlir::arith::MulFOp::create(nest, nloc, inp, sigmoidVal);
      } else if (inpElmType.isInteger()) {
        auto c1 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 1));
        auto c0 = mlir::arith::ConstantOp::create(
            nest, nloc, nest.getIntegerAttr(inpElmType, 0));
        auto cB = mlir::arith::ConstantOp::create(
            nest, nloc,
            nest.getIntegerAttr(inpElmType, static_cast<int64_t>(attr_beta)));
        auto betaX = mlir::arith::MulIOp::create(nest, nloc, cB, inp);
        auto negBetaX = mlir::arith::SubIOp::create(nest, nloc, c0, betaX);
        auto expNegBetaX = mlir::math::ExpOp::create(nest, nloc, negBetaX);
        auto denom = mlir::arith::AddIOp::create(nest, nloc, c1, expNegBetaX);
        auto sigmoidVal = mlir::arith::DivSIOp::create(nest, nloc, c1, denom);
        out = mlir::arith::MulIOp::create(nest, nloc, inp, sigmoidVal);
      }
    }
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

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
