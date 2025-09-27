/******************************************************************+************
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
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return rewriter.notifyMatchFailure(
        op, opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, opName + " result must be a ranked tensor type");
  }

  mlir::Location loc = op->getLoc();

  // 1. Create an empty tensor for the output
  mlir::Value outBuff = rewriter.create<mlir::tensor::EmptyOp>(
      loc, inpType.getShape(), inpType.getElementType());

  // 2. Create the linalg.generic operation
  mlir::SmallVector<mlir::utils::IteratorType> iterators;
  for (int i = 0; i < inpType.getRank(); ++i) {
    iterators.push_back(mlir::utils::IteratorType::parallel);
  }

  mlir::SmallVector<mlir::AffineMap> idxMaps;
  idxMaps.push_back(rewriter.getMultiDimIdentityMap(inpType.getRank()));
  idxMaps.push_back(rewriter.getMultiDimIdentityMap(inpType.getRank()));

  auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, inpType, mlir::ValueRange{inp}, mlir::ValueRange{outBuff}, idxMaps,
      iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value outOp;
        if (opNameBeginsWith(opName, "Abs")) {
          if (mlir::isa<mlir::FloatType>(inpType.getElementType()))
            outOp = nest.create<mlir::math::AbsFOp>(loc, args[0]);
          else
            outOp = nest.create<mlir::math::AbsIOp>(loc, args[0]);
        }
        if (opNameBeginsWith(opName, "Acos"))
          outOp = nest.create<mlir::math::AcosOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Acosh"))
          outOp = nest.create<mlir::math::AcoshOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Asin"))
          outOp = nest.create<mlir::math::AsinOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Asinh"))
          outOp = nest.create<mlir::math::AsinhOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Atan"))
          outOp = nest.create<mlir::math::AtanOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Atanh"))
          outOp = nest.create<mlir::math::AtanhOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Ceil"))
          outOp = nest.create<mlir::math::CeilOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Cos"))
          outOp = nest.create<mlir::math::CosOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Cosh"))
          outOp = nest.create<mlir::math::CoshOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Elu")) {
          double alpha = 1.0;
          auto alphaAttr = op->getAttr("alpha");
          if (alphaAttr) {
            if(auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(alphaAttr)) {
              alpha = floatAttr.getValueAsDouble();
            }
          }
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto cA = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, alpha));
            auto c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 0.0));
            auto c1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 1.0));
            auto cnd = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, args[0], c0);
            auto exp = nest.create<mlir::math::ExpOp>(loc, args[0]);
            auto sub = nest.create<mlir::arith::SubFOp>(loc, exp, c1);
            auto neg = nest.create<mlir::arith::MulFOp>(loc, cA, sub);
            outOp = nest.create<mlir::arith::SelectOp>(loc, cnd, args[0], neg);
          } else {
            auto cA = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, int(alpha)));
            auto c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 0));
            auto c1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 1));
            auto cnd = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::sge, args[0], c0);
            auto exp = nest.create<mlir::math::ExpOp>(loc, args[0]);
            auto sub = nest.create<mlir::arith::SubIOp>(loc, exp, c1);
            auto neg = nest.create<mlir::arith::MulIOp>(loc, cA, sub);
            outOp = nest.create<mlir::arith::SelectOp>(loc, cnd, args[0], neg);
          }
        }
        if (opNameBeginsWith(opName, "Erf"))
          outOp = nest.create<mlir::math::ErfOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Exp"))
          outOp = nest.create<mlir::math::ExpOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Floor"))
          outOp = nest.create<mlir::math::FloorOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "HardSwish")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 0.0));
            auto c3 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 3.0));
            auto c6 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 6.0));
            auto xPlus3 = nest.create<mlir::arith::AddFOp>(loc, args[0], c3);
            auto condPos = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGT, xPlus3, c0);
            auto max0 =
                nest.create<mlir::arith::SelectOp>(loc, condPos, xPlus3, c0);
            auto condLimit = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, max0, c6);
            auto relu6_arg =
                nest.create<mlir::arith::SelectOp>(loc, condLimit, max0, c6);
            auto numerator =
                nest.create<mlir::arith::MulFOp>(loc, args[0], relu6_arg);
            outOp = nest.create<mlir::arith::DivFOp>(loc, numerator, c6);
          } else {
            auto c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 0));
            auto c3 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 3));
            auto c6 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 6));
            auto xPlus3 = nest.create<mlir::arith::AddIOp>(loc, args[0], c3);
            auto condPos = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::sgt, xPlus3, c0);
            auto max0 =
                nest.create<mlir::arith::SelectOp>(loc, condPos, xPlus3, c0);
            auto condLimit = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::slt, max0, c6);
            auto relu6_arg =
                nest.create<mlir::arith::SelectOp>(loc, condLimit, max0, c6);
            auto numerator =
                nest.create<mlir::arith::MulIOp>(loc, args[0], relu6_arg);
            outOp = nest.create<mlir::arith::DivSIOp>(loc, numerator, c6);
          }
        }
        if (opNameBeginsWith(opName, "Identity"))
          outOp = args[0];
        if (opNameBeginsWith(opName, "IsInf"))
          outOp = nest.create<mlir::math::IsInfOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "IsNaN"))
          outOp = nest.create<mlir::math::IsNaNOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Log"))
          outOp = nest.create<mlir::math::LogOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Neg")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            outOp = nest.create<mlir::arith::NegFOp>(loc, args[0]);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            mlir::Value c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 0));
            outOp = nest.create<mlir::arith::SubIOp>(loc, c0, args[0]);
          }
        }
        // TODO(cbalint13): Not is integer only
        if (opNameBeginsWith(opName, "Not")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::IntegerType>(elmType)) {
            int bitW = mlir::cast<mlir::IntegerType>(elmType).getWidth();
            auto ones = llvm::APInt::getAllOnes(bitW);
            auto allOnes = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, ones));
            outOp = nest.create<mlir::arith::XOrIOp>(loc, args[0], allOnes);
          }
        }
        // TODO(cbalint13): Reciprocal is float only
        if (opNameBeginsWith(opName, "Reciprocal")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            mlir::Value c1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 1.0));
            outOp = nest.create<mlir::arith::DivFOp>(loc, c1, args[0]);
          }
        }
        if (opNameBeginsWith(opName, "Relu")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 0.0));
            auto cnd = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, args[0], c0);
            outOp = nest.create<mlir::arith::SelectOp>(loc, cnd, args[0], c0);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            auto c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 0));
            auto cnd = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::sge, args[0], c0);
            outOp = nest.create<mlir::arith::SelectOp>(loc, cnd, args[0], c0);
          }
        }
        if (opNameBeginsWith(opName, "Round"))
          outOp = nest.create<mlir::math::RoundOp>(loc, args[0]);

        if (opNameBeginsWith(opName, "Sigmoid")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 1.0));
            auto negX = nest.create<mlir::arith::NegFOp>(loc, args[0]);
            auto expNegX = nest.create<mlir::math::ExpOp>(loc, negX);
            auto denom = nest.create<mlir::arith::AddFOp>(loc, c1, expNegX);
            outOp = nest.create<mlir::arith::DivFOp>(loc, c1, denom);
          } else {
            auto c1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 1));
            auto c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 0));
            auto negX = nest.create<mlir::arith::SubIOp>(loc, c0, args[0]);
            auto expNegX = nest.create<mlir::math::ExpOp>(loc, negX);
            auto denom = nest.create<mlir::arith::AddIOp>(loc, c1, expNegX);
            outOp = nest.create<mlir::arith::DivSIOp>(loc, c1, denom);
          }
        }
        if (opNameBeginsWith(opName, "Sign")) {
          mlir::Type elmType = inpType.getElementType();
          mlir::Value c0, cPos1, cNeg1, cndPos, cndNeg;
          if (mlir::isa<mlir::FloatType>(elmType)) {
            c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 0.0));
            cPos1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 1.0));
            cNeg1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, -1.0));
            cndPos = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGT, args[0], c0);
            cndNeg = nest.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, args[0], c0);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            c0 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 0));
            cPos1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 1));
            cNeg1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, -1));
            cndPos = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::sgt, args[0], c0);
            cndNeg = nest.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::slt, args[0], c0);
          }
          auto resIfPos = nest.create<mlir::arith::SelectOp>(
              loc, cndPos, cPos1, /*else_value=*/nullptr);
          auto resIfNonPos =
              nest.create<mlir::arith::SelectOp>(loc, cndNeg, cNeg1, c0);
          outOp = nest.create<mlir::arith::SelectOp>(loc, cndPos, cPos1,
                                                     resIfNonPos);
        }
        if (opNameBeginsWith(opName, "Sin"))
          outOp = nest.create<mlir::math::SinOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Sinh"))
          outOp = nest.create<mlir::math::SinhOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Softplus")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 1.0));
            auto expX = nest.create<mlir::math::ExpOp>(loc, args[0]);
            auto logArg = nest.create<mlir::arith::AddFOp>(loc, c1, expX);
            outOp = nest.create<mlir::math::LogOp>(loc, logArg);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            auto c1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 1));
            auto expX = nest.create<mlir::math::ExpOp>(loc, args[0]);
            auto logArg = nest.create<mlir::arith::AddIOp>(loc, c1, expX);
            outOp = nest.create<mlir::math::LogOp>(loc, logArg);
          }
        }
        if (opNameBeginsWith(opName, "Softsign")) {
          mlir::Type elmType = inpType.getElementType();
          if (mlir::isa<mlir::FloatType>(elmType)) {
            auto c1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getFloatAttr(elmType, 1.0));
            auto absX = nest.create<mlir::math::AbsFOp>(loc, args[0]);
            auto denom = nest.create<mlir::arith::AddFOp>(loc, c1, absX);
            outOp = nest.create<mlir::arith::DivFOp>(loc, args[0], denom);
          } else if (mlir::isa<mlir::IntegerType>(elmType)) {
            auto c1 = nest.create<mlir::arith::ConstantOp>(
                loc, nest.getIntegerAttr(elmType, 1));
            auto absX = nest.create<mlir::math::AbsIOp>(loc, args[0]);
            auto denom = nest.create<mlir::arith::AddIOp>(loc, c1, absX);
            outOp = nest.create<mlir::arith::DivSIOp>(loc, args[0], denom);
          }
        }
        if (opNameBeginsWith(opName, "Sqrt"))
          outOp = nest.create<mlir::math::SqrtOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Tan"))
          outOp = nest.create<mlir::math::TanOp>(loc, args[0]);
        if (opNameBeginsWith(opName, "Tanh"))
          outOp = nest.create<mlir::math::TanhOp>(loc, args[0]);

        nest.create<mlir::linalg::YieldOp>(loc, outOp);
      });

  // Tag for transform optimization
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

mlir::LogicalResult OnnxToLinalg_SoftmaxOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter) {
  auto ctx = rewriter.getContext();
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return rewriter.notifyMatchFailure(
        op, opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, opName + " result must be a ranked tensor type");
  }

  auto inpElmType = inpType.getElementType();
  if (!mlir::isa<mlir::FloatType>(inpElmType)) {
    return rewriter.notifyMatchFailure(op,
                                       opName + " requires float element type");
  }

  mlir::Location loc = op->getLoc();

  auto axisAttr = op->getAttr("axis");
  if (!axisAttr) {
    return rewriter.notifyMatchFailure(op,
                                       opName + " is missing 'axis' attribute");
  }

  auto axisInt = mlir::dyn_cast_or_null<mlir::IntegerAttr>(axisAttr);
  if (!axisInt) {
    return rewriter.notifyMatchFailure(
        op, opName + " has invalid 'axis' attribute type");
  }

  auto axis = axisInt.getInt();
  auto rank = inpType.getRank();

  if (axis < 0 || axis >= rank) {
    return rewriter.notifyMatchFailure(op, opName + " invalid axis");
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
      rewriter.create<mlir::tensor::EmptyOp>(loc, reduce_shape, inpElmType);
  auto fltType = mlir::cast<mlir::FloatType>(inpElmType);
  mlir::Value negInf = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(
               fltType, llvm::APFloat::getInf(fltType.getFloatSemantics(),
                                              /*Negative=*/true)));
  auto maxBuff =
      rewriter.create<mlir::linalg::FillOp>(loc, negInf, maxTBuff.getResult())
          .getResult(0);

  auto maxOp = rewriter.create<mlir::linalg::ReduceOp>(
      loc, mlir::ValueRange{inp}, mlir::ValueRange{maxBuff}, dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            nest.create<mlir::arith::MaximumFOp>(loc, args[0], args[1]);
        nest.create<mlir::linalg::YieldOp>(loc, result);
      });
  mlir::Value maxVal = maxOp.getResult(0);

  // 2. Subtraction (x - max(x)) and Exponentiation (e^(x - max(x)))
  mlir::SmallVector<mlir::AffineMap> exp_maps;
  exp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  exp_maps.push_back(reduce_broadcast_map);
  exp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto expTBuff = rewriter.create<mlir::tensor::EmptyOp>(
      loc, inpType.getShape(), inpElmType);

  auto expOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, inpType, mlir::ValueRange{inp, maxVal},
      mlir::ValueRange{expTBuff.getResult()}, exp_maps, parallel_iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value diff =
            nest.create<mlir::arith::SubFOp>(loc, args[0], args[1]);
        mlir::Value expX = nest.create<mlir::math::ExpOp>(loc, diff);
        nest.create<mlir::linalg::YieldOp>(loc, expX);
      });
  mlir::Value expVal = expOp.getResult(0);

  // 3. Sum Reduction (row-wise sum of exp values)
  auto sumTBuff =
      rewriter.create<mlir::tensor::EmptyOp>(loc, reduce_shape, inpElmType);
  mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(inpElmType, 0.0));
  mlir::Value sumBuff =
      rewriter.create<mlir::linalg::FillOp>(loc, zero, sumTBuff.getResult())
          .getResult(0);

  auto sumOp = rewriter.create<mlir::linalg::ReduceOp>(
      loc, mlir::ValueRange{expVal}, mlir::ValueRange{sumBuff}, dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            nest.create<mlir::arith::AddFOp>(loc, args[0], args[1]);
        nest.create<mlir::linalg::YieldOp>(loc, result);
      });
  mlir::Value sumVal = sumOp.getResult(0);

  // 4. Final Normalization (element-wise div)
  auto outBuff = rewriter.create<mlir::tensor::EmptyOp>(loc, inpType.getShape(),
                                                        inpElmType);

  mlir::SmallVector<mlir::AffineMap> norm_maps;
  norm_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  norm_maps.push_back(reduce_broadcast_map);
  norm_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto idxMapsAttr = rewriter.getAffineMapArrayAttr(norm_maps);
  auto kindAttr = mlir::linalg::ElementwiseKindAttr::get(
      op->getContext(), mlir::linalg::ElementwiseKind::div);

  auto normOp = rewriter.create<mlir::linalg::ElementwiseOp>(
      loc, mlir::ValueRange{expVal, sumVal},
      mlir::ValueRange{outBuff.getResult()}, kindAttr, idxMapsAttr);

  rewriter.replaceOp(op, normOp.getResult(0));

  return mlir::success();
}

mlir::LogicalResult OnnxToLinalg_LogSoftmaxOp(mlir::Operation *op,
                                              mlir::PatternRewriter &rewriter) {
  auto ctx = rewriter.getContext();
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return rewriter.notifyMatchFailure(
        op, opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, opName + " result must be a ranked tensor type");
  }

  auto inpElmType = inpType.getElementType();
  if (!mlir::isa<mlir::FloatType>(inpElmType)) {
    return rewriter.notifyMatchFailure(op,
                                       opName + " requires float element type");
  }

  mlir::Location loc = op->getLoc();

  auto axisAttr = op->getAttr("axis");
  if (!axisAttr) {
    return rewriter.notifyMatchFailure(op,
                                       opName + " is missing 'axis' attribute");
  }

  auto axisInt = mlir::dyn_cast_or_null<mlir::IntegerAttr>(axisAttr);
  if (!axisInt) {
    return rewriter.notifyMatchFailure(
        op, opName + " has invalid 'axis' attribute type");
  }

  auto axis = axisInt.getInt();
  auto rank = inpType.getRank();

  if (axis < 0 || axis >= rank) {
    return rewriter.notifyMatchFailure(op, opName + " invalid axis");
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
      rewriter.create<mlir::tensor::EmptyOp>(loc, reduce_shape, inpElmType);
  auto fltType = mlir::cast<mlir::FloatType>(inpElmType);
  mlir::Value negInf = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(
               fltType, llvm::APFloat::getInf(fltType.getFloatSemantics(),
                                              /*Negative=*/true)));
  auto maxBuff =
      rewriter.create<mlir::linalg::FillOp>(loc, negInf, maxTBuff.getResult())
          .getResult(0);

  auto maxOp = rewriter.create<mlir::linalg::ReduceOp>(
      loc, mlir::ValueRange{inp}, mlir::ValueRange{maxBuff}, dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            nest.create<mlir::arith::MaximumFOp>(loc, args[0], args[1]);
        nest.create<mlir::linalg::YieldOp>(loc, result);
      });
  mlir::Value maxVal = maxOp.getResult(0);

  // 2. Subtraction (x - max(x)) and Exponentiation (e^(x - max(x)))
  mlir::SmallVector<mlir::AffineMap> exp_maps;
  exp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  exp_maps.push_back(reduce_broadcast_map);
  exp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto expTBuff = rewriter.create<mlir::tensor::EmptyOp>(
      loc, inpType.getShape(), inpElmType);

  auto expOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, inpType, mlir::ValueRange{inp, maxVal},
      mlir::ValueRange{expTBuff.getResult()}, exp_maps, parallel_iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value diff =
            nest.create<mlir::arith::SubFOp>(loc, args[0], args[1]);
        mlir::Value expX = nest.create<mlir::math::ExpOp>(loc, diff);
        nest.create<mlir::linalg::YieldOp>(loc, expX);
      });
  mlir::Value expVal = expOp.getResult(0);

  // 3. Sum Reduction (row-wise sum of exp values)
  auto sumTBuff =
      rewriter.create<mlir::tensor::EmptyOp>(loc, reduce_shape, inpElmType);
  mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(inpElmType, 0.0));
  mlir::Value sumBuff =
      rewriter.create<mlir::linalg::FillOp>(loc, zero, sumTBuff.getResult())
          .getResult(0);

  auto sumOp = rewriter.create<mlir::linalg::ReduceOp>(
      loc, mlir::ValueRange{expVal}, mlir::ValueRange{sumBuff}, dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            nest.create<mlir::arith::AddFOp>(loc, args[0], args[1]);
        nest.create<mlir::linalg::YieldOp>(loc, result);
      });
  mlir::Value sumVal = sumOp.getResult(0);

  // 4. Fused (x - max(x)) - log(sum(e^(x - max(x))))
  auto logSoftmaxTBuff = rewriter.create<mlir::tensor::EmptyOp>(
      loc, inpType.getShape(), inpElmType);

  mlir::SmallVector<mlir::AffineMap> final_maps;
  final_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  final_maps.push_back(reduce_broadcast_map);
  final_maps.push_back(reduce_broadcast_map);
  final_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto logSoftmaxOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, inpType, mlir::ValueRange{inp, maxVal, sumVal},
      mlir::ValueRange{logSoftmaxTBuff.getResult()}, final_maps,
      parallel_iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        auto diffMx = nest.create<mlir::arith::SubFOp>(loc, args[0], args[1]);
        auto logSum = nest.create<mlir::math::LogOp>(loc, args[2]);
        mlir::Value result =
            nest.create<mlir::arith::SubFOp>(loc, diffMx, logSum);
        nest.create<mlir::linalg::YieldOp>(loc, result);
      });

  rewriter.replaceOp(op, logSoftmaxOp.getResult(0));

  return mlir::success();
}

mlir::LogicalResult OnnxToLinalg_HardmaxOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter) {
  auto ctx = rewriter.getContext();
  auto opName = op->getName().getStringRef();

  mlir::Value inp = op->getOperand(0);
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType) {
    return rewriter.notifyMatchFailure(
        op, opName + " operand must be ranked tensor type");
  }

  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, opName + " result must be a ranked tensor type");
  }

  auto inpElmType = inpType.getElementType();
  if (!mlir::isa<mlir::FloatType>(inpElmType)) {
    return rewriter.notifyMatchFailure(op,
                                       opName + " requires float element type");
  }

  mlir::Location loc = op->getLoc();

  auto axisAttr = op->getAttr("axis");
  if (!axisAttr) {
    return rewriter.notifyMatchFailure(op,
                                       opName + " is missing 'axis' attribute");
  }

  auto axisInt = mlir::dyn_cast_or_null<mlir::IntegerAttr>(axisAttr);
  if (!axisInt) {
    return rewriter.notifyMatchFailure(
        op, opName + " has invalid 'axis' attribute type");
  }

  auto axis = axisInt.getInt();
  auto rank = inpType.getRank();

  if (axis < 0 || axis >= rank) {
    return rewriter.notifyMatchFailure(op, opName + " invalid axis");
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
      rewriter.create<mlir::tensor::EmptyOp>(loc, reduce_shape, inpElmType);
  auto fltType = mlir::cast<mlir::FloatType>(inpElmType);
  mlir::Value negInf = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(
               fltType, llvm::APFloat::getInf(fltType.getFloatSemantics(),
                                              /*Negative=*/true)));
  auto maxBuff =
      rewriter.create<mlir::linalg::FillOp>(loc, negInf, maxTBuff.getResult())
          .getResult(0);

  auto maxOp = rewriter.create<mlir::linalg::ReduceOp>(
      loc, mlir::ValueRange{inp}, mlir::ValueRange{maxBuff}, dimsAttr,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result =
            nest.create<mlir::arith::MaximumFOp>(loc, args[0], args[1]);
        nest.create<mlir::linalg::YieldOp>(loc, result);
      });
  mlir::Value maxVal = maxOp.getResult(0);

  // 2. Comparison and selection (Input == MaxVal) ? 1.0 : 0.0
  mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(inpElmType, 1.0));
  mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(inpElmType, 0.0));

  mlir::SmallVector<mlir::AffineMap> cmp_maps;
  cmp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));
  cmp_maps.push_back(reduce_broadcast_map);
  cmp_maps.push_back(rewriter.getMultiDimIdentityMap(rank));

  auto outTBuff = rewriter.create<mlir::tensor::EmptyOp>(
      loc, inpType.getShape(), inpElmType);

  mlir::SmallVector<mlir::utils::IteratorType> parallel_iterators(
      rank, mlir::utils::IteratorType::parallel);

  auto hardmaxOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, resType, mlir::ValueRange{inp, maxVal},
      mlir::ValueRange{outTBuff.getResult()}, cmp_maps, parallel_iterators,
      [&](mlir::OpBuilder nest, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value condition = nest.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, args[0], args[1]);
        mlir::Value result =
            nest.create<mlir::arith::SelectOp>(loc, condition, one, zero);

        nest.create<mlir::linalg::YieldOp>(loc, result);
      });

  rewriter.replaceOp(op, hardmaxOp.getResult(0));

  return mlir::success();
}

} // namespace onnx2mlir::dialect
