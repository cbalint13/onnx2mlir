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
 * \file src/conversion/passes/onnx_to_linalg/arith_unary.cpp
 * \brief ONNX Arith unary operations to Linalg lowering
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

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_ArithUnaryOps(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return mlir::emitError(loc, opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return mlir::emitError(loc,
                           opName + " result must be a ranked tensor type");
  }

  // 1. Create an empty tensor for the output
  mlir::Value outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, inpType.getShape(), inpType.getElementType());

  // 2. Create the linalg.generic operation
  mlir::SmallVector<mlir::utils::IteratorType> iterators;
  for (int i = 0; i < inpType.getRank(); ++i) {
    iterators.push_back(mlir::utils::IteratorType::parallel);
  }

  mlir::SmallVector<mlir::AffineMap> idxMaps;
  idxMaps.push_back(rewriter.getMultiDimIdentityMap(inpType.getRank()));
  idxMaps.push_back(rewriter.getMultiDimIdentityMap(inpType.getRank()));

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, inpType, mlir::ValueRange{inp}, mlir::ValueRange{outBuff},
      idxMaps, iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value outOp;
        if (opNameBeginsWith(opName, "Abs")) {
          if (mlir::isa<mlir::FloatType>(inpType.getElementType()))
            outOp = mlir::math::AbsFOp::create(nest, loc, args[0]);
          else
            outOp = mlir::math::AbsIOp::create(nest, loc, args[0]);
        }
        if (opNameBeginsWith(opName, "Acos"))
          outOp = mlir::math::AcosOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Acosh"))
          outOp = mlir::math::AcoshOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Asin"))
          outOp = mlir::math::AsinOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Asinh"))
          outOp = mlir::math::AsinhOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Atan"))
          outOp = mlir::math::AtanOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Atanh"))
          outOp = mlir::math::AtanhOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Ceil"))
          outOp = mlir::math::CeilOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Cos"))
          outOp = mlir::math::CosOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Cosh"))
          outOp = mlir::math::CoshOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Elu")) {
          double alpha = 1.0;
          auto alphaAttr = op->getAttr("alpha");
          if (alphaAttr) {
            if (auto floatAttr =
                    mlir::dyn_cast_or_null<mlir::FloatAttr>(alphaAttr)) {
              alpha = floatAttr.getValueAsDouble();
            }
          }
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto cA = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, alpha));
            auto c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 0.0));
            auto c1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 1.0));
            auto cnd = mlir::arith::CmpFOp::create(
                nest, loc, mlir::arith::CmpFPredicate::OGE, args[0], c0);
            auto exp = mlir::math::ExpOp::create(nest, loc, args[0]);
            auto sub = mlir::arith::SubFOp::create(nest, loc, exp, c1);
            auto neg = mlir::arith::MulFOp::create(nest, loc, cA, sub);
            outOp = mlir::arith::SelectOp::create(nest, loc, cnd, args[0], neg);
          } else {
            auto cA = mlir::arith::ConstantOp::create(
                nest, loc,
                nest.getIntegerAttr(elmType, static_cast<int>(alpha)));
            auto c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 0));
            auto c1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 1));
            auto cnd = mlir::arith::CmpIOp::create(
                nest, loc, mlir::arith::CmpIPredicate::sge, args[0], c0);
            auto exp = mlir::math::ExpOp::create(nest, loc, args[0]);
            auto sub = mlir::arith::SubIOp::create(nest, loc, exp, c1);
            auto neg = mlir::arith::MulIOp::create(nest, loc, cA, sub);
            outOp = mlir::arith::SelectOp::create(nest, loc, cnd, args[0], neg);
          }
        }
        if (opNameBeginsWith(opName, "Erf"))
          outOp = mlir::math::ErfOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Exp"))
          outOp = mlir::math::ExpOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Floor"))
          outOp = mlir::math::FloorOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "HardSwish")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 0.0));
            auto c3 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 3.0));
            auto c6 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 6.0));
            auto xPlus3 = mlir::arith::AddFOp::create(nest, loc, args[0], c3);
            auto condPos = mlir::arith::CmpFOp::create(
                nest, loc, mlir::arith::CmpFPredicate::OGT, xPlus3, c0);
            auto max0 =
                mlir::arith::SelectOp::create(nest, loc, condPos, xPlus3, c0);
            auto condLimit = mlir::arith::CmpFOp::create(
                nest, loc, mlir::arith::CmpFPredicate::OLT, max0, c6);
            auto relu6_arg =
                mlir::arith::SelectOp::create(nest, loc, condLimit, max0, c6);
            auto numerator =
                mlir::arith::MulFOp::create(nest, loc, args[0], relu6_arg);
            outOp = mlir::arith::DivFOp::create(nest, loc, numerator, c6);
          } else {
            auto c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 0));
            auto c3 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 3));
            auto c6 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 6));
            auto xPlus3 = mlir::arith::AddIOp::create(nest, loc, args[0], c3);
            auto condPos = mlir::arith::CmpIOp::create(
                nest, loc, mlir::arith::CmpIPredicate::sgt, xPlus3, c0);
            auto max0 =
                mlir::arith::SelectOp::create(nest, loc, condPos, xPlus3, c0);
            auto condLimit = mlir::arith::CmpIOp::create(
                nest, loc, mlir::arith::CmpIPredicate::slt, max0, c6);
            auto relu6_arg =
                mlir::arith::SelectOp::create(nest, loc, condLimit, max0, c6);
            auto numerator =
                mlir::arith::MulIOp::create(nest, loc, args[0], relu6_arg);
            outOp = mlir::arith::DivSIOp::create(nest, loc, numerator, c6);
          }
        }
        if (opNameBeginsWith(opName, "Identity"))
          outOp = args[0];
        if (opNameBeginsWith(opName, "IsInf"))
          outOp = mlir::math::IsInfOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "IsNaN"))
          outOp = mlir::math::IsNaNOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Log"))
          outOp = mlir::math::LogOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Neg")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            outOp = mlir::arith::NegFOp::create(nest, loc, args[0]);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            mlir::Value c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 0));
            outOp = mlir::arith::SubIOp::create(nest, loc, c0, args[0]);
          }
        }
        // TODO(cbalint13): Not is integer only
        if (opNameBeginsWith(opName, "Not")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::IntegerType>(elmType)) {
            int bitW = mlir::cast<mlir::IntegerType>(elmType).getWidth();
            auto ones = llvm::APInt::getAllOnes(bitW);
            auto allOnes = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, ones));
            outOp = mlir::arith::XOrIOp::create(nest, loc, args[0], allOnes);
          }
        }
        // TODO(cbalint13): Reciprocal is float only
        if (opNameBeginsWith(opName, "Reciprocal")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            mlir::Value c1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 1.0));
            outOp = mlir::arith::DivFOp::create(nest, loc, c1, args[0]);
          }
        }
        if (opNameBeginsWith(opName, "Relu")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 0.0));
            auto cnd = mlir::arith::CmpFOp::create(
                nest, loc, mlir::arith::CmpFPredicate::OGE, args[0], c0);
            outOp = mlir::arith::SelectOp::create(nest, loc, cnd, args[0], c0);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            auto c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 0));
            auto cnd = mlir::arith::CmpIOp::create(
                nest, loc, mlir::arith::CmpIPredicate::sge, args[0], c0);
            outOp = mlir::arith::SelectOp::create(nest, loc, cnd, args[0], c0);
          }
        }
        if (opNameBeginsWith(opName, "Round"))
          outOp = mlir::math::RoundOp::create(nest, loc, args[0]);

        if (opNameBeginsWith(opName, "Sigmoid")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 1.0));
            auto negX = mlir::arith::NegFOp::create(nest, loc, args[0]);
            auto expNegX = mlir::math::ExpOp::create(nest, loc, negX);
            auto denom = mlir::arith::AddFOp::create(nest, loc, c1, expNegX);
            outOp = mlir::arith::DivFOp::create(nest, loc, c1, denom);
          } else {
            auto c1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 1));
            auto c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 0));
            auto negX = mlir::arith::SubIOp::create(nest, loc, c0, args[0]);
            auto expNegX = mlir::math::ExpOp::create(nest, loc, negX);
            auto denom = mlir::arith::AddIOp::create(nest, loc, c1, expNegX);
            outOp = mlir::arith::DivSIOp::create(nest, loc, c1, denom);
          }
        }
        if (opNameBeginsWith(opName, "Sign")) {
          mlir::Type elmType = inpType.getElementType();
          mlir::Value c0, cPos1, cNeg1, cndPos, cndNeg;
          if (mlir::isa<mlir::FloatType>(elmType)) {
            c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 0.0));
            cPos1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 1.0));
            cNeg1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, -1.0));
            cndPos = mlir::arith::CmpFOp::create(
                nest, loc, mlir::arith::CmpFPredicate::OGT, args[0], c0);
            cndNeg = mlir::arith::CmpFOp::create(
                nest, loc, mlir::arith::CmpFPredicate::OLT, args[0], c0);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            c0 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 0));
            cPos1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 1));
            cNeg1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, -1));
            cndPos = mlir::arith::CmpIOp::create(
                nest, loc, mlir::arith::CmpIPredicate::sgt, args[0], c0);
            cndNeg = mlir::arith::CmpIOp::create(
                nest, loc, mlir::arith::CmpIPredicate::slt, args[0], c0);
          }
          auto resIfPos = mlir::arith::SelectOp::create(
              nest, loc, cndPos, cPos1, /*else_value=*/nullptr);
          auto resIfNonPos =
              mlir::arith::SelectOp::create(nest, loc, cndNeg, cNeg1, c0);
          outOp = mlir::arith::SelectOp::create(nest, loc, cndPos, cPos1,
                                                resIfNonPos);
        }
        if (opNameBeginsWith(opName, "Sin"))
          outOp = mlir::math::SinOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Sinh"))
          outOp = mlir::math::SinhOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Softplus")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 1.0));
            auto expX = mlir::math::ExpOp::create(nest, loc, args[0]);
            auto logArg = mlir::arith::AddFOp::create(nest, loc, c1, expX);
            outOp = mlir::math::LogOp::create(nest, loc, logArg);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            auto c1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 1));
            auto expX = mlir::math::ExpOp::create(nest, loc, args[0]);
            auto logArg = mlir::arith::AddIOp::create(nest, loc, c1, expX);
            outOp = mlir::math::LogOp::create(nest, loc, logArg);
          }
        }
        if (opNameBeginsWith(opName, "Softsign")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getFloatAttr(elmType, 1.0));
            auto absX = mlir::math::AbsFOp::create(nest, loc, args[0]);
            auto denom = mlir::arith::AddFOp::create(nest, loc, c1, absX);
            outOp = mlir::arith::DivFOp::create(nest, loc, args[0], denom);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            auto c1 = mlir::arith::ConstantOp::create(
                nest, loc, nest.getIntegerAttr(elmType, 1));
            auto absX = mlir::math::AbsIOp::create(nest, loc, args[0]);
            auto denom = mlir::arith::AddIOp::create(nest, loc, c1, absX);
            outOp = mlir::arith::DivSIOp::create(nest, loc, args[0], denom);
          }
        }
        if (opNameBeginsWith(opName, "Sqrt"))
          outOp = mlir::math::SqrtOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Tan"))
          outOp = mlir::math::TanOp::create(nest, loc, args[0]);
        if (opNameBeginsWith(opName, "Tanh"))
          outOp = mlir::math::TanhOp::create(nest, loc, args[0]);

        mlir::linalg::YieldOp::create(nest, loc, outOp);
      });

  // Tag for transform optimization
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

mlir::LogicalResult OnnxToLinalg_SoftmaxOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter) {
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return mlir::emitError(loc, opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return mlir::emitError(loc,
                           opName + " result must be a ranked tensor type");
  }

  auto inpElmType = inpType.getElementType();
  if (!mlir::isa<mlir::FloatType>(inpElmType)) {
    return mlir::emitError(loc, opName + " requires float element type");
  }

  auto axisAttr = op->getAttr("axis");
  if (!axisAttr) {
    return mlir::emitError(loc, opName + " is missing 'axis' attribute");
  }

  auto axisInt = mlir::dyn_cast_or_null<mlir::IntegerAttr>(axisAttr);
  if (!axisInt) {
    return mlir::emitError(loc, opName + " has invalid 'axis' attribute type");
  }

  auto axis = axisInt.getInt();
  auto rank = inpType.getRank();

  if (axis < -rank || axis >= rank) {
    return mlir::emitError(loc, opName + " invalid axis");
  }

  if (axis < 0) {
    axis = rank + axis;
  }

  // parallel iterators once
  mlir::SmallVector<mlir::utils::IteratorType> parallel_iterators(
      rank, mlir::utils::IteratorType::parallel);

  // map and shape definitions
  mlir::SmallVector<int64_t> reduce_shape;
  mlir::SmallVector<mlir::AffineExpr> reduce_outputMapExprs;
  for (int i = 0; i < rank; ++i) {
    if (i != axis) {
      reduce_shape.push_back(inpType.getShape()[i]);
      reduce_outputMapExprs.push_back(rewriter.getAffineDimExpr(i));
    }
  }
  auto reduceType = mlir::RankedTensorType::get(reduce_shape, inpElmType);

  // affine maps for broadcasting
  mlir::AffineMap reduce_broadcast_map =
      mlir::AffineMap::get(rank, 0, reduce_outputMapExprs, ctx);

  // attribute for reduction dims [axis]
  mlir::SmallVector<int64_t> dims = {axis};
  auto dimsAttr = rewriter.getDenseI64ArrayAttr(dims);

  // 1. Max Reduction (find row-wise max for stability)
  auto maxTBuff =
      mlir::tensor::EmptyOp::create(rewriter, loc, reduce_shape, inpElmType);
  auto fltType = mlir::cast<mlir::FloatType>(inpElmType);
  mlir::Value negInf = mlir::arith::ConstantOp::create(
      rewriter, loc,
      rewriter.getFloatAttr(fltType,
                            llvm::APFloat::getInf(fltType.getFloatSemantics(),
                                                  /*Negative=*/true)));
  auto maxBuff =
      mlir::linalg::FillOp::create(rewriter, loc, negInf, maxTBuff.getResult())
          .getResult(0);

  auto maxOp = mlir::linalg::ReduceOp::create(
      rewriter, loc, mlir::ValueRange{inp}, mlir::ValueRange{maxBuff}, dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            mlir::arith::MaximumFOp::create(rewriter, loc, args[0], args[1]);
        mlir::linalg::YieldOp::create(rewriter, loc, result);
      });
  mlir::Value maxVal = maxOp.getResult(0);

  // 2. Subtraction (x - max(x)) and Exponentiation (e^(x - max(x)))
  mlir::SmallVector<mlir::AffineMap> exp_maps;
  exp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  exp_maps.push_back(reduce_broadcast_map);
  exp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto expTBuff = mlir::tensor::EmptyOp::create(rewriter, loc,
                                                inpType.getShape(), inpElmType);

  auto expOp = mlir::linalg::GenericOp::create(
      rewriter, loc, inpType, mlir::ValueRange{inp, maxVal},
      mlir::ValueRange{expTBuff.getResult()}, exp_maps, parallel_iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value diff =
            mlir::arith::SubFOp::create(nest, loc, args[0], args[1]);
        mlir::Value expX = mlir::math::ExpOp::create(nest, loc, diff);
        mlir::linalg::YieldOp::create(nest, loc, expX);
      });
  mlir::Value expVal = expOp.getResult(0);

  // 3. Sum Reduction (row-wise sum of exp values)
  auto sumTBuff =
      mlir::tensor::EmptyOp::create(rewriter, loc, reduce_shape, inpElmType);
  mlir::Value zero = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(inpElmType, 0.0));
  mlir::Value sumBuff =
      mlir::linalg::FillOp::create(rewriter, loc, zero, sumTBuff.getResult())
          .getResult(0);

  auto sumOp = mlir::linalg::ReduceOp::create(
      rewriter, loc, mlir::ValueRange{expVal}, mlir::ValueRange{sumBuff},
      dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            mlir::arith::AddFOp::create(nest, loc, args[0], args[1]);
        mlir::linalg::YieldOp::create(nest, loc, result);
      });
  mlir::Value sumVal = sumOp.getResult(0);

  // 4. Final Normalization (element-wise div)
  auto outBuff = mlir::tensor::EmptyOp::create(rewriter, loc,
                                               inpType.getShape(), inpElmType);

  mlir::SmallVector<mlir::AffineMap> norm_maps;
  norm_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  norm_maps.push_back(reduce_broadcast_map);
  norm_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto idxMapsAttr = rewriter.getAffineMapArrayAttr(norm_maps);
  auto kindAttr = mlir::linalg::ElementwiseKindAttr::get(
      op->getContext(), mlir::linalg::ElementwiseKind::div);

  auto normOp = mlir::linalg::ElementwiseOp::create(
      rewriter, loc, mlir::ValueRange{expVal, sumVal},
      mlir::ValueRange{outBuff.getResult()}, kindAttr, idxMapsAttr);

  rewriter.replaceOp(op, normOp.getResult(0));

  return mlir::success();
}

mlir::LogicalResult OnnxToLinalg_LogSoftmaxOp(mlir::Operation *op,
                                              mlir::PatternRewriter &rewriter) {
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return mlir::emitError(loc, opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return mlir::emitError(loc,
                           opName + " result must be a ranked tensor type");
  }

  auto inpElmType = inpType.getElementType();
  if (!mlir::isa<mlir::FloatType>(inpElmType)) {
    return mlir::emitError(loc, opName + " requires float element type");
  }

  auto axisAttr = op->getAttr("axis");
  if (!axisAttr) {
    return mlir::emitError(loc, opName + " is missing 'axis' attribute");
  }

  auto axisInt = mlir::dyn_cast_or_null<mlir::IntegerAttr>(axisAttr);
  if (!axisInt) {
    return mlir::emitError(loc, opName + " has invalid 'axis' attribute type");
  }

  auto axis = axisInt.getInt();
  auto rank = inpType.getRank();

  if (axis < -rank || axis >= rank) {
    return mlir::emitError(loc, opName + " invalid axis");
  }

  if (axis < 0) {
    axis = rank + axis;
  }

  // parallel iterators
  mlir::SmallVector<mlir::utils::IteratorType> parallel_iterators(
      rank, mlir::utils::IteratorType::parallel);

  // map and shape Definitions
  mlir::SmallVector<int64_t> reduce_shape;
  mlir::SmallVector<mlir::AffineExpr> reduce_outputMapExprs;
  for (int i = 0; i < rank; ++i) {
    if (i != axis) {
      reduce_shape.push_back(inpType.getShape()[i]);
      reduce_outputMapExprs.push_back(rewriter.getAffineDimExpr(i));
    }
  }
  auto reduceType = mlir::RankedTensorType::get(reduce_shape, inpElmType);

  // affine map for broadcasting
  mlir::AffineMap reduce_broadcast_map =
      mlir::AffineMap::get(rank, 0, reduce_outputMapExprs, ctx);

  // attribute for reduction dims [axis]
  mlir::SmallVector<int64_t> dims = {axis};
  auto dimsAttr = rewriter.getDenseI64ArrayAttr(dims);

  // 1. Max Reduction (find row-wise max for stability)
  auto maxTBuff =
      mlir::tensor::EmptyOp::create(rewriter, loc, reduce_shape, inpElmType);
  auto fltType = mlir::cast<mlir::FloatType>(inpElmType);
  mlir::Value negInf = mlir::arith::ConstantOp::create(
      rewriter, loc,
      rewriter.getFloatAttr(fltType,
                            llvm::APFloat::getInf(fltType.getFloatSemantics(),
                                                  /*Negative=*/true)));
  auto maxBuff =
      mlir::linalg::FillOp::create(rewriter, loc, negInf, maxTBuff.getResult())
          .getResult(0);

  auto maxOp = mlir::linalg::ReduceOp::create(
      rewriter, loc, mlir::ValueRange{inp}, mlir::ValueRange{maxBuff}, dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            mlir::arith::MaximumFOp::create(nest, loc, args[0], args[1]);
        mlir::linalg::YieldOp::create(nest, loc, result);
      });
  mlir::Value maxVal = maxOp.getResult(0);

  // 2. Subtraction (x - max(x)) and Exponentiation (e^(x - max(x)))
  mlir::SmallVector<mlir::AffineMap> exp_maps;
  exp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  exp_maps.push_back(reduce_broadcast_map);
  exp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto expTBuff = mlir::tensor::EmptyOp::create(rewriter, loc,
                                                inpType.getShape(), inpElmType);

  auto expOp = mlir::linalg::GenericOp::create(
      rewriter, loc, inpType, mlir::ValueRange{inp, maxVal},
      mlir::ValueRange{expTBuff.getResult()}, exp_maps, parallel_iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value diff =
            mlir::arith::SubFOp::create(nest, loc, args[0], args[1]);
        mlir::Value expX = mlir::math::ExpOp::create(nest, loc, diff);
        mlir::linalg::YieldOp::create(nest, loc, expX);
      });
  mlir::Value expVal = expOp.getResult(0);

  // 3. Sum Reduction (row-wise sum of exp values)
  auto sumTBuff =
      mlir::tensor::EmptyOp::create(rewriter, loc, reduce_shape, inpElmType);
  mlir::Value zero = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(inpElmType, 0.0));
  mlir::Value sumBuff =
      mlir::linalg::FillOp::create(rewriter, loc, zero, sumTBuff.getResult())
          .getResult(0);

  auto sumOp = mlir::linalg::ReduceOp::create(
      rewriter, loc, mlir::ValueRange{expVal}, mlir::ValueRange{sumBuff},
      dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            mlir::arith::AddFOp::create(nest, loc, args[0], args[1]);
        mlir::linalg::YieldOp::create(nest, loc, result);
      });
  mlir::Value sumVal = sumOp.getResult(0);

  // 4. Fused (x - max(x)) - log(sum(e^(x - max(x))))
  auto logSoftmaxTBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, inpType.getShape(), inpElmType);

  mlir::SmallVector<mlir::AffineMap> final_maps;
  final_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  final_maps.push_back(reduce_broadcast_map);
  final_maps.push_back(reduce_broadcast_map);
  final_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto logSoftmaxOp = mlir::linalg::GenericOp::create(
      rewriter, loc, inpType, mlir::ValueRange{inp, maxVal, sumVal},
      mlir::ValueRange{logSoftmaxTBuff.getResult()}, final_maps,
      parallel_iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        auto diffMx = mlir::arith::SubFOp::create(nest, loc, args[0], args[1]);
        auto logSum = mlir::math::LogOp::create(nest, loc, args[2]);
        mlir::Value result =
            mlir::arith::SubFOp::create(nest, loc, diffMx, logSum);
        mlir::linalg::YieldOp::create(nest, loc, result);
      });

  rewriter.replaceOp(op, logSoftmaxOp.getResult(0));

  return mlir::success();
}

mlir::LogicalResult OnnxToLinalg_HardmaxOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter) {
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return mlir::emitError(loc, opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return mlir::emitError(loc,
                           opName + " result must be a ranked tensor type");
  }

  auto inpElmType = inpType.getElementType();
  if (!mlir::isa<mlir::FloatType>(inpElmType)) {
    return mlir::emitError(loc, opName + " requires float element type");
  }

  auto axisAttr = op->getAttr("axis");
  if (!axisAttr) {
    return mlir::emitError(loc, opName + " is missing 'axis' attribute");
  }

  auto axisInt = mlir::dyn_cast_or_null<mlir::IntegerAttr>(axisAttr);
  if (!axisInt) {
    return mlir::emitError(loc, opName + " has invalid 'axis' attribute type");
  }

  auto axis = axisInt.getInt();
  auto rank = inpType.getRank();

  if (axis < -rank || axis >= rank) {
    return mlir::emitError(loc, opName + " invalid axis");
  }

  if (axis < 0) {
    axis = rank + axis;
  }

  // map and shape definitions
  mlir::SmallVector<int64_t> reduce_shape;
  mlir::SmallVector<mlir::AffineExpr> reduce_outputMapExprs;
  for (int i = 0; i < rank; ++i) {
    if (i != axis) {
      reduce_shape.push_back(inpType.getShape()[i]);
      reduce_outputMapExprs.push_back(rewriter.getAffineDimExpr(i));
    }
  }

  // affine maps for broadcasting
  auto reduceType = mlir::RankedTensorType::get(reduce_shape, inpElmType);
  mlir::AffineMap reduce_broadcast_map =
      mlir::AffineMap::get(rank, 0, reduce_outputMapExprs, ctx);

  // attribute for reduction dims [axis]
  mlir::SmallVector<int64_t> dims = {axis};
  auto dimsAttr = rewriter.getDenseI64ArrayAttr(dims);

  // 1. Max reduction (find max value along the axis)
  auto maxTBuff =
      mlir::tensor::EmptyOp::create(rewriter, loc, reduce_shape, inpElmType);
  auto fltType = mlir::cast<mlir::FloatType>(inpElmType);
  mlir::Value negInf = mlir::arith::ConstantOp::create(
      rewriter, loc,
      rewriter.getFloatAttr(fltType,
                            llvm::APFloat::getInf(fltType.getFloatSemantics(),
                                                  /*Negative=*/true)));
  auto maxBuff =
      mlir::linalg::FillOp::create(rewriter, loc, negInf, maxTBuff.getResult())
          .getResult(0);

  auto maxOp = mlir::linalg::ReduceOp::create(
      rewriter, loc, mlir::ValueRange{inp}, mlir::ValueRange{maxBuff}, dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            mlir::arith::MaximumFOp::create(nest, loc, args[0], args[1]);
        mlir::linalg::YieldOp::create(nest, loc, result);
      });
  mlir::Value maxVal = maxOp.getResult(0);

  // 2. Comparison and selection (Input == MaxVal) ? 1.0 : 0.0
  mlir::Value one = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(inpElmType, 1.0));
  mlir::Value zero = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(inpElmType, 0.0));

  mlir::SmallVector<mlir::AffineMap> cmp_maps;
  cmp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  cmp_maps.push_back(reduce_broadcast_map);
  cmp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto outTBuff = mlir::tensor::EmptyOp::create(rewriter, loc,
                                                inpType.getShape(), inpElmType);

  mlir::SmallVector<mlir::utils::IteratorType> parallel_iterators(
      rank, mlir::utils::IteratorType::parallel);

  auto hardmaxOp = mlir::linalg::GenericOp::create(
      rewriter, loc, resType, mlir::ValueRange{inp, maxVal},
      mlir::ValueRange{outTBuff.getResult()}, cmp_maps, parallel_iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value condition = mlir::arith::CmpFOp::create(
            nest, loc, mlir::arith::CmpFPredicate::OEQ, args[0], args[1]);
        mlir::Value result =
            mlir::arith::SelectOp::create(nest, loc, condition, one, zero);

        mlir::linalg::YieldOp::create(nest, loc, result);
      });

  rewriter.replaceOp(op, hardmaxOp.getResult(0));

  return mlir::success();
}

} // namespace onnx2mlir::dialect
