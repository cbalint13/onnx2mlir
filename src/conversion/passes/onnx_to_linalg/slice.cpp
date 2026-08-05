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
 * \file src/conversion/passes/onnx_to_linalg/slice.cpp
 * \brief ONNX Slice operation to Linalg lowering pass
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

static bool extractIntArrayFromValue(mlir::Value val,
                                     llvm::SmallVectorImpl<int64_t> &res) {
  if (!val || mlir::isa<mlir::NoneType>(val.getType()))
    return false;

  // Unwrap common conversion / cast operations
  while (val && val.getDefiningOp()) {
    auto *defOp = val.getDefiningOp();
    if (auto castOp = mlir::dyn_cast<mlir::arith::IndexCastOp>(defOp)) {
      val = castOp.getIn();
    } else if (auto castOp = mlir::dyn_cast<mlir::tensor::CastOp>(defOp)) {
      val = castOp.getSource();
    } else if (defOp->getName().getStringRef() == "onnx.Cast") {
      val = defOp->getOperand(0);
    } else {
      break;
    }
  }

  if (!val)
    return false;

  auto processAttr = [&](mlir::Attribute attr) -> bool {
    if (!attr)
      return false;

    if (auto denseInt = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
      for (auto v : denseInt.getValues<mlir::APInt>()) {
        res.push_back(v.getSExtValue());
      }
      return !res.empty();
    }
    if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
      if (denseAttr.getElementType().isIntOrIndex()) {
        for (auto v : denseAttr.getValues<mlir::APInt>()) {
          res.push_back(v.getSExtValue());
        }
        return !res.empty();
      }
    }
    if (auto denseI64 = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
      res = llvm::to_vector(denseI64.asArrayRef());
      return !res.empty();
    }
    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      for (auto a : arrayAttr) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(a)) {
          res.push_back(intAttr.getInt());
        }
      }
      return !res.empty();
    }
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
      res.push_back(intAttr.getInt());
      return true;
    }
    return false;
  };

  if (auto constOp = val.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (processAttr(constOp.getValue()))
      return true;
  }

  if (auto *op = val.getDefiningOp()) {
    if (auto attr = op->getAttr("value")) {
      if (processAttr(attr))
        return true;
    }

    if (auto fromElem = mlir::dyn_cast<mlir::tensor::FromElementsOp>(op)) {
      llvm::SmallVector<int64_t> elemValues;
      bool allConstants = true;
      for (auto elem : fromElem.getElements()) {
        llvm::SmallVector<int64_t> singleRes;
        if (extractIntArrayFromValue(elem, singleRes) && !singleRes.empty()) {
          elemValues.push_back(singleRes[0]);
        } else {
          allConstants = false;
          break;
        }
      }
      if (allConstants && !elemValues.empty()) {
        res = std::move(elemValues);
        return true;
      }
    }
  }

  return false;
}

static bool extractIntArrayFromAttr(mlir::Operation *op,
                                    llvm::StringRef attrName,
                                    llvm::SmallVectorImpl<int64_t> &res) {
  if (auto denseI64 = op->getAttrOfType<mlir::DenseI64ArrayAttr>(attrName)) {
    res = llvm::to_vector(denseI64.asArrayRef());
    return true;
  }
  if (auto arrayAttr = op->getAttrOfType<mlir::ArrayAttr>(attrName)) {
    for (auto attr : arrayAttr) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        res.push_back(intAttr.getInt());
      }
    }
    return !res.empty();
  }
  if (auto denseInt = op->getAttrOfType<mlir::DenseIntElementsAttr>(attrName)) {
    for (auto v : denseInt.getValues<mlir::APInt>()) {
      res.push_back(v.getSExtValue());
    }
    return !res.empty();
  }
  if (auto denseAttr = op->getAttrOfType<mlir::DenseElementsAttr>(attrName)) {
    if (denseAttr.getElementType().isIntOrIndex()) {
      for (auto v : denseAttr.getValues<mlir::APInt>()) {
        res.push_back(v.getSExtValue());
      }
      return !res.empty();
    }
  }
  return false;
}

mlir::LogicalResult
OnnxToLinalg_SliceOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                     const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  mlir::Value inp = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value res = convRewriter.getRemappedValue(op->getResult(0));

  auto inpType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(inp.getType());
  if (!inpType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " requires ranked tensor input");
  }

  int64_t rank = inpType.getRank();
  if (rank < 1) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " input operand rank must be >= 1");
  }

  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());
  if (!resType) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " result must be a ranked tensor");
  }

  llvm::SmallVector<int64_t> rawStarts;
  llvm::SmallVector<int64_t> rawEnds;
  llvm::SmallVector<int64_t> rawAxes;
  llvm::SmallVector<int64_t> rawSteps;

  mlir::Value startsVal = nullptr;
  mlir::Value endsVal = nullptr;
  mlir::Value axesVal = nullptr;
  mlir::Value stepsVal = nullptr;

  if (op->getNumOperands() > 1) {
    startsVal = op->getOperand(1);
    extractIntArrayFromValue(startsVal, rawStarts);
  }
  if (op->getNumOperands() > 2) {
    endsVal = op->getOperand(2);
    extractIntArrayFromValue(endsVal, rawEnds);
  }
  if (op->getNumOperands() > 3) {
    axesVal = op->getOperand(3);
    extractIntArrayFromValue(axesVal, rawAxes);
  }
  if (op->getNumOperands() > 4) {
    stepsVal = op->getOperand(4);
    extractIntArrayFromValue(stepsVal, rawSteps);
  }

  // Fallback to attributes for Slice v1
  if (rawStarts.empty())
    extractIntArrayFromAttr(op, "starts", rawStarts);
  if (rawEnds.empty())
    extractIntArrayFromAttr(op, "ends", rawEnds);
  if (rawAxes.empty())
    extractIntArrayFromAttr(op, "axes", rawAxes);
  if (rawSteps.empty())
    extractIntArrayFromAttr(op, "steps", rawSteps);

  llvm::SmallVector<mlir::Value> startVals(rank, nullptr);
  llvm::SmallVector<mlir::Value> stepVals(rank, nullptr);
  llvm::SmallVector<mlir::Value> outDimVals(rank, nullptr);

  mlir::Value zeroIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
  mlir::Value oneIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);

  bool isStatic = !rawStarts.empty() && !rawEnds.empty();

  if (isStatic) {
    if (rawAxes.empty()) {
      for (size_t i = 0; i < rawStarts.size(); ++i) {
        rawAxes.push_back(static_cast<int64_t>(i));
      }
    }
    if (rawSteps.empty()) {
      rawSteps.assign(rawStarts.size(), 1);
    }

    llvm::SmallVector<int64_t> staticStarts(rank, 0);
    llvm::SmallVector<int64_t> staticSteps(rank, 1);
    llvm::SmallVector<int64_t> staticEnds(rank, -1);
    llvm::SmallVector<bool> axisSliced(rank, false);

    for (size_t k = 0; k < rawAxes.size(); ++k) {
      int64_t axis = rawAxes[k];
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank) {
        return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                               opName + " axis out of bounds");
      }

      axisSliced[axis] = true;
      staticStarts[axis] = rawStarts[k];
      staticEnds[axis] = rawEnds[k];
      staticSteps[axis] = (k < rawSteps.size()) ? rawSteps[k] : 1;
    }

    for (int64_t i = 0; i < rank; ++i) {
      mlir::Value dimVal = mlir::tensor::DimOp::create(
          rewriter, loc, inp,
          mlir::arith::ConstantIndexOp::create(rewriter, loc, i));

      if (!axisSliced[i]) {
        startVals[i] = zeroIdx;
        stepVals[i] = oneIdx;
        outDimVals[i] = dimVal;
        continue;
      }

      int64_t st = staticSteps[i];
      int64_t s = staticStarts[i];
      int64_t e = staticEnds[i];
      int64_t staticDim = inpType.getDimSize(i);

      if (staticDim >= 0) {
        // Fully static dimension calculations
        if (st > 0) {
          if (s < 0)
            s += staticDim;
          s = std::max<int64_t>(0, std::min<int64_t>(s, staticDim));
          if (e < 0)
            e += staticDim;
          e = std::max<int64_t>(0, std::min<int64_t>(e, staticDim));
          int64_t outLen = (e > s) ? (e - s + st - 1) / st : 0;

          startVals[i] = mlir::arith::ConstantIndexOp::create(rewriter, loc, s);
          stepVals[i] = mlir::arith::ConstantIndexOp::create(rewriter, loc, st);
          outDimVals[i] =
              mlir::arith::ConstantIndexOp::create(rewriter, loc, outLen);
        } else if (st < 0) {
          if (s < 0)
            s += staticDim;
          s = std::max<int64_t>(-1, std::min<int64_t>(s, staticDim - 1));
          if (e < 0)
            e += staticDim;
          e = std::max<int64_t>(-1, std::min<int64_t>(e, staticDim - 1));
          int64_t absSt = -st;
          int64_t outLen = (s > e) ? (s - e + absSt - 1) / absSt : 0;

          startVals[i] = mlir::arith::ConstantIndexOp::create(rewriter, loc, s);
          stepVals[i] = mlir::arith::ConstantIndexOp::create(rewriter, loc, st);
          outDimVals[i] =
              mlir::arith::ConstantIndexOp::create(rewriter, loc, outLen);
        }
      } else {
        // Dynamic dimension calculations with static start/end/step
        mlir::Value sVal =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, s);
        mlir::Value eVal =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, e);
        mlir::Value stVal =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, st);

        if (s < 0) {
          sVal = mlir::arith::AddIOp::create(rewriter, loc, dimVal, sVal);
        }
        if (e < 0) {
          eVal = mlir::arith::AddIOp::create(rewriter, loc, dimVal, eVal);
        }

        mlir::Value normS = mlir::arith::MaxSIOp::create(
            rewriter, loc, zeroIdx,
            mlir::arith::MinSIOp::create(rewriter, loc, sVal, dimVal));
        mlir::Value normE = mlir::arith::MaxSIOp::create(
            rewriter, loc, zeroIdx,
            mlir::arith::MinSIOp::create(rewriter, loc, eVal, dimVal));

        mlir::Value len =
            mlir::arith::SubIOp::create(rewriter, loc, normE, normS);
        len = mlir::arith::MaxSIOp::create(rewriter, loc, zeroIdx, len);

        mlir::Value num = mlir::arith::AddIOp::create(
            rewriter, loc, len,
            mlir::arith::ConstantIndexOp::create(rewriter, loc, st - 1));
        mlir::Value outLen =
            mlir::arith::DivUIOp::create(rewriter, loc, num, stVal);

        startVals[i] = normS;
        stepVals[i] = stVal;
        outDimVals[i] = outLen;
      }
    }
  } else {
    if (!startsVal || !endsVal) {
      return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                             opName + " requires starts and ends operands");
    }

    // Dynamic slicing using SSA values
    for (int64_t i = 0; i < rank; ++i) {
      mlir::Value dimVal = mlir::tensor::DimOp::create(
          rewriter, loc, inp,
          mlir::arith::ConstantIndexOp::create(rewriter, loc, i));

      mlir::Value idxI = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);

      // Extract raw values from 1D tensors
      mlir::Value rawS = mlir::tensor::ExtractOp::create(
          rewriter, loc, startsVal, mlir::ValueRange{idxI});
      mlir::Value rawE = mlir::tensor::ExtractOp::create(
          rewriter, loc, endsVal, mlir::ValueRange{idxI});

      mlir::Value sIdx = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), rawS);
      mlir::Value eIdx = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), rawE);

      mlir::Value stIdx = oneIdx;
      if (stepsVal && !mlir::isa<mlir::NoneType>(stepsVal.getType())) {
        mlir::Value rawSt = mlir::tensor::ExtractOp::create(
            rewriter, loc, stepsVal, mlir::ValueRange{idxI});
        stIdx = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), rawSt);
      }

      // Handle negative start/end
      mlir::Value isNegS = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, sIdx, zeroIdx);
      mlir::Value posS =
          mlir::arith::AddIOp::create(rewriter, loc, sIdx, dimVal);
      mlir::Value normS =
          mlir::arith::SelectOp::create(rewriter, loc, isNegS, posS, sIdx);
      normS = mlir::arith::MaxSIOp::create(
          rewriter, loc, zeroIdx,
          mlir::arith::MinSIOp::create(rewriter, loc, normS, dimVal));

      mlir::Value isNegE = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, eIdx, zeroIdx);
      mlir::Value posE =
          mlir::arith::AddIOp::create(rewriter, loc, eIdx, dimVal);
      mlir::Value normE =
          mlir::arith::SelectOp::create(rewriter, loc, isNegE, posE, eIdx);
      normE = mlir::arith::MaxSIOp::create(
          rewriter, loc, zeroIdx,
          mlir::arith::MinSIOp::create(rewriter, loc, normE, dimVal));

      mlir::Value len =
          mlir::arith::SubIOp::create(rewriter, loc, normE, normS);
      len = mlir::arith::MaxSIOp::create(rewriter, loc, zeroIdx, len);

      mlir::Value num = mlir::arith::AddIOp::create(
          rewriter, loc, len,
          mlir::arith::SubIOp::create(rewriter, loc, stIdx, oneIdx));
      mlir::Value outLen =
          mlir::arith::DivUIOp::create(rewriter, loc, num, stIdx);

      startVals[i] = normS;
      stepVals[i] = stIdx;
      outDimVals[i] = outLen;
    }
  }

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i = 0; i < rank; ++i) {
    if (resType.isDynamicDim(i)) {
      dynSizes.push_back(outDimVals[i]);
    }
  }

  auto outBuff =
      mlir::tensor::EmptyOp::create(rewriter, loc, resType, dynSizes);

  auto outMap = rewriter.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap> indexingMaps = {outMap};

  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      rank, mlir::utils::IteratorType::parallel);

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/mlir::TypeRange{resType},
      /*inputs=*/mlir::ValueRange{},
      /*outputs=*/mlir::ValueRange{outBuff},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location nstLoc,
          mlir::ValueRange blockArgs) {
        llvm::SmallVector<mlir::Value> inpCoords;
        inpCoords.reserve(rank);

        for (int64_t d = 0; d < rank; ++d) {
          mlir::Value loopIdx = mlir::linalg::IndexOp::create(
              b, nstLoc, static_cast<uint64_t>(d));

          mlir::Value mulVal =
              mlir::arith::MulIOp::create(b, nstLoc, loopIdx, stepVals[d]);
          mlir::Value coord =
              mlir::arith::AddIOp::create(b, nstLoc, startVals[d], mulVal);
          inpCoords.push_back(coord);
        }

        mlir::Value extracted =
            mlir::tensor::ExtractOp::create(b, nstLoc, inp, inpCoords);
        mlir::linalg::YieldOp::create(b, nstLoc, extracted);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));
  rewriter.replaceOp(op, genericOp.getResult(0));

  return mlir::success();
}

} // namespace onnx2mlir::dialect
