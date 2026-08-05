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
 * \file src/conversion/passes/onnx_to_linalg/clip.cpp
 * \brief ONNX Clip operation to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <cfloat>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

// Utility function to extract a scalar Value for min/max bounds from either
// an SSA tensor operand (Opsets 11+) or an operation attribute (Opsets 1 & 6).
static mlir::Value getScalarBound(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value boundOperand,
                                  mlir::Attribute boundAttr,
                                  mlir::Type targetElemType,
                                  mlir::Type origElemType) {
  mlir::Value scalarVal = nullptr;

  // 1. Unwrap cast operations on the bound operand tensor
  while (boundOperand && boundOperand.getDefiningOp()) {
    auto *defOp = boundOperand.getDefiningOp();
    if (auto castOp = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(defOp)) {
      if (castOp.getInputs().empty())
        break;
      boundOperand = castOp.getInputs()[0];
    } else if (auto castOp = mlir::dyn_cast<mlir::tensor::CastOp>(defOp)) {
      boundOperand = castOp.getSource();
    } else if (defOp->getName().getStringRef().contains("Cast")) {
      if (defOp->getNumOperands() > 0)
        boundOperand = defOp->getOperand(0);
      else
        break;
    } else {
      break;
    }
  }

  // 2. Extract constant scalar bound directly from constant defining op
  // attribute if present
  if (boundOperand && boundOperand.getDefiningOp()) {
    auto *defOp = boundOperand.getDefiningOp();
    mlir::Attribute valAttr = defOp->getAttr("value");
    if (!valAttr)
      valAttr = boundAttr;

    if (valAttr) {
      if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valAttr)) {
        if (denseAttr.isSplat() || denseAttr.getNumElements() == 1) {
          if (mlir::isa<mlir::FloatType>(targetElemType)) {
            double doubleVal = 0.0;
            if (mlir::isa<mlir::FloatType>(denseAttr.getElementType())) {
              doubleVal =
                  denseAttr.getSplatValue<mlir::APFloat>().convertToDouble();
            } else if (mlir::isa<mlir::IntegerType>(
                           denseAttr.getElementType())) {
              doubleVal = static_cast<double>(
                  denseAttr.getSplatValue<mlir::APInt>().getSExtValue());
            }
            scalarVal = mlir::arith::ConstantOp::create(
                rewriter, loc, targetElemType,
                rewriter.getFloatAttr(targetElemType, doubleVal));
          } else if (auto intType =
                         mlir::dyn_cast<mlir::IntegerType>(targetElemType)) {
            int64_t intVal = 0;
            if (mlir::isa<mlir::IntegerType>(denseAttr.getElementType())) {
              auto apInt = denseAttr.getSplatValue<mlir::APInt>();
              auto origInt = mlir::dyn_cast<mlir::IntegerType>(origElemType);
              bool isUnsigned = origInt && origInt.isUnsigned();
              intVal = isUnsigned ? apInt.getZExtValue() : apInt.getSExtValue();
            } else if (mlir::isa<mlir::FloatType>(denseAttr.getElementType())) {
              intVal = static_cast<int64_t>(
                  denseAttr.getSplatValue<mlir::APFloat>().convertToDouble());
            }
            scalarVal = mlir::arith::ConstantOp::create(
                rewriter, loc, targetElemType,
                rewriter.getIntegerAttr(targetElemType, intVal));
          }
        }
      }
    }
  }

  // 3. Extract scalar from dynamic/runtime tensor operand if present and not
  // NoneType
  if (!scalarVal && boundOperand &&
      !mlir::isa<mlir::NoneType>(boundOperand.getType())) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(boundOperand.getType());
    if (tensorType) {
      int64_t rank = tensorType.getRank();
      llvm::SmallVector<mlir::Value> zeroIndices;
      zeroIndices.reserve(rank);
      for (int64_t i = 0; i < rank; ++i) {
        zeroIndices.push_back(
            mlir::arith::ConstantIndexOp::create(rewriter, loc, 0));
      }
      scalarVal = mlir::tensor::ExtractOp::create(rewriter, loc, boundOperand,
                                                  zeroIndices);
    }
  }

  // 4. Fall back to operation attribute for Opset 1 / Opset 6
  if (!scalarVal && boundAttr) {
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(boundAttr)) {
      if (mlir::isa<mlir::FloatType>(targetElemType)) {
        scalarVal = mlir::arith::ConstantOp::create(
            rewriter, loc, targetElemType,
            rewriter.getFloatAttr(targetElemType,
                                  floatAttr.getValueAsDouble()));
      } else if (auto intType =
                     mlir::dyn_cast<mlir::IntegerType>(targetElemType)) {
        int64_t intVal = static_cast<int64_t>(floatAttr.getValueAsDouble());
        scalarVal = mlir::arith::ConstantOp::create(
            rewriter, loc, targetElemType,
            rewriter.getIntegerAttr(targetElemType, intVal));
      }
    } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(boundAttr)) {
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(targetElemType)) {
        scalarVal = mlir::arith::ConstantOp::create(
            rewriter, loc, targetElemType,
            rewriter.getIntegerAttr(targetElemType, intAttr.getInt()));
      } else if (mlir::isa<mlir::FloatType>(targetElemType)) {
        scalarVal = mlir::arith::ConstantOp::create(
            rewriter, loc, targetElemType,
            rewriter.getFloatAttr(targetElemType,
                                  static_cast<double>(intAttr.getInt())));
      }
    }
  }

  // 3. Cast scalar type to match target input element type if needed
  if (scalarVal && scalarVal.getType() != targetElemType) {
    auto srcType = scalarVal.getType();
    if (mlir::isa<mlir::FloatType>(srcType) &&
        mlir::isa<mlir::FloatType>(targetElemType)) {
      auto srcFloat = mlir::cast<mlir::FloatType>(srcType);
      auto dstFloat = mlir::cast<mlir::FloatType>(targetElemType);
      if (srcFloat.getWidth() < dstFloat.getWidth()) {
        scalarVal = mlir::arith::ExtFOp::create(rewriter, loc, targetElemType,
                                                scalarVal);
      } else if (srcFloat.getWidth() > dstFloat.getWidth()) {
        scalarVal = mlir::arith::TruncFOp::create(rewriter, loc, targetElemType,
                                                  scalarVal);
      }
    } else if (mlir::isa<mlir::IntegerType>(srcType) &&
               mlir::isa<mlir::IntegerType>(targetElemType)) {
      auto srcInt = mlir::cast<mlir::IntegerType>(srcType);
      auto dstInt = mlir::cast<mlir::IntegerType>(targetElemType);

      auto dstType = rewriter.getIntegerType(dstInt.getWidth());

      if (srcInt.getWidth() < dstInt.getWidth()) {
        bool isUnsigned = srcInt.isUnsigned();
        if (auto origInt = mlir::dyn_cast<mlir::IntegerType>(origElemType)) {
          if (origInt.isUnsigned())
            isUnsigned = true;
        }
        if (isUnsigned) {
          scalarVal =
              mlir::arith::ExtUIOp::create(rewriter, loc, dstType, scalarVal);
        } else {
          scalarVal =
              mlir::arith::ExtSIOp::create(rewriter, loc, dstType, scalarVal);
        }
      } else if (srcInt.getWidth() > dstInt.getWidth()) {
        scalarVal =
            mlir::arith::TruncIOp::create(rewriter, loc, dstType, scalarVal);
      }
    }
  }

  return scalarVal;
}

mlir::LogicalResult
OnnxToLinalg_ClipOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  mlir::Value inp = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value res = op->getResult(0);

  auto inpType = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(
      typeConverter->convertType(res.getType()));

  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if (!inpType || !resType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " requires ranked tensor input and result");
  }

  int64_t rank = inpType.getRank();
  mlir::Type origElemType = outType.getElementType();
  mlir::Type elemType = typeConverter->convertType(origElemType);

  // Extract min and max operands if provided (Opset 11+)
  mlir::Value minOperand =
      (op->getNumOperands() > 1) ? op->getOperand(1) : nullptr;
  mlir::Value maxOperand =
      (op->getNumOperands() > 2) ? op->getOperand(2) : nullptr;

  // Retrieve min and max attributes if present (Opset 1 and Opset 6)
  mlir::Attribute minAttr = op->getAttr("min");
  mlir::Attribute maxAttr = op->getAttr("max");

  // Retrieve scalar values for bounds converted to signless elemType
  mlir::Value minScalar = getScalarBound(rewriter, loc, minOperand, minAttr,
                                         elemType, origElemType);
  mlir::Value maxScalar = getScalarBound(rewriter, loc, maxOperand, maxAttr,
                                         elemType, origElemType);

  // no clipping required
  if (!minScalar && !maxScalar) {
    rewriter.replaceOp(op, op->getOperand(0));
    return mlir::success();
  }

  // Query dynamic dimensions for creating output empty tensor destination
  llvm::SmallVector<mlir::Value> dynamicSizes;
  for (int64_t i = 0; i < rank; ++i) {
    if (resType.isDynamicDim(i)) {
      mlir::Value dimIdx =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
      dynamicSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, inp, dimIdx));
    }
  }

  auto outBuff =
      mlir::tensor::EmptyOp::create(rewriter, loc, resType, dynamicSizes);

  auto identityMap = rewriter.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap> indexingMaps = {identityMap, identityMap};

  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      rank, mlir::utils::IteratorType::parallel);

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc,
      /*resTypes=*/mlir::TypeRange{resType},
      /*inputs=*/mlir::ValueRange{inp},
      /*outputs=*/mlir::ValueRange{outBuff},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location nestedLoc,
          mlir::ValueRange blockArgs) {
        mlir::Value val = blockArgs[0];

        // Apply lower bound clipping (val = max(val, minScalar))
        if (minScalar) {
          if (mlir::isa<mlir::FloatType>(elemType)) {
            mlir::Value isLess = mlir::arith::CmpFOp::create(
                b, nestedLoc, mlir::arith::CmpFPredicate::OLT, val, minScalar);
            val = mlir::arith::SelectOp::create(b, nestedLoc, isLess, minScalar,
                                                val);
          } else if (mlir::isa<mlir::IntegerType>(elemType)) {
            auto origInt = mlir::dyn_cast<mlir::IntegerType>(origElemType);
            auto pred = (origInt && origInt.isUnsigned())
                            ? mlir::arith::CmpIPredicate::ult
                            : mlir::arith::CmpIPredicate::slt;
            mlir::Value isLess =
                mlir::arith::CmpIOp::create(b, nestedLoc, pred, val, minScalar);
            val = mlir::arith::SelectOp::create(b, nestedLoc, isLess, minScalar,
                                                val);
          }
        }

        // Apply upper bound clipping (val = min(val, maxScalar))
        if (maxScalar) {
          if (mlir::isa<mlir::FloatType>(elemType)) {
            mlir::Value isGreater = mlir::arith::CmpFOp::create(
                b, nestedLoc, mlir::arith::CmpFPredicate::OGT, val, maxScalar);
            val = mlir::arith::SelectOp::create(b, nestedLoc, isGreater,
                                                maxScalar, val);
          } else if (mlir::isa<mlir::IntegerType>(elemType)) {
            auto origInt = mlir::dyn_cast<mlir::IntegerType>(origElemType);
            auto pred = (origInt && origInt.isUnsigned())
                            ? mlir::arith::CmpIPredicate::ugt
                            : mlir::arith::CmpIPredicate::sgt;
            mlir::Value isGreater =
                mlir::arith::CmpIOp::create(b, nestedLoc, pred, val, maxScalar);
            val = mlir::arith::SelectOp::create(b, nestedLoc, isGreater,
                                                maxScalar, val);
          }
        }

        mlir::linalg::YieldOp::create(b, nestedLoc, val);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));
  mlir::Value output = genericOp.getResult(0);

  rewriter.replaceOp(op, output);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
