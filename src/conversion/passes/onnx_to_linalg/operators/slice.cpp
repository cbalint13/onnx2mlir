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
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_SliceOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                     const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opInpStarts = (op->getNumOperands() > 1 &&
                      !mlir::isa<mlir::NoneType>(op->getOperand(1).getType()))
                         ? convRewriter.getRemappedValue(op->getOperand(1))
                         : nullptr;
  auto opInpEnds = (op->getNumOperands() > 2 &&
                    !mlir::isa<mlir::NoneType>(op->getOperand(2).getType()))
                       ? convRewriter.getRemappedValue(op->getOperand(2))
                       : nullptr;
  auto opInpAxes = (op->getNumOperands() > 3 &&
                    !mlir::isa<mlir::NoneType>(op->getOperand(3).getType()))
                       ? convRewriter.getRemappedValue(op->getOperand(3))
                       : nullptr;
  auto opInpSteps = (op->getNumOperands() > 4 &&
                     !mlir::isa<mlir::NoneType>(op->getOperand(4).getType()))
                        ? convRewriter.getRemappedValue(op->getOperand(4))
                        : nullptr;
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  int64_t inputRank = inpDatType.getRank();

  /*
   * Attributes
   */

  auto getI64Array = [&](llvm::StringRef name, llvm::ArrayRef<int64_t> def) {
    llvm::SmallVector<int64_t> vals;
    if (auto attr = op->getAttrOfType<mlir::ArrayAttr>(name)) {
      for (auto a : attr.getAsRange<mlir::IntegerAttr>())
        vals.push_back(a.getInt());
    } else {
      vals.assign(def.begin(), def.end());
    }
    return vals;
  };

  // starts
  auto attr_starts = getI64Array("starts", {});

  // ends
  auto attr_ends = getI64Array("ends", {});

  // axes
  auto attr_axes = getI64Array("axes", {});

  // steps
  auto attr_steps = getI64Array("steps", {});

  /*
   * Affine mappings
   */

  auto outMap = rewriter.getMultiDimIdentityMap(inputRank);
  llvm::SmallVector<mlir::AffineMap> indexingMaps = {outMap};

  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      inputRank, mlir::utils::IteratorType::parallel);

  /*
   *  Linalg ops staging
   */

  auto i64Type = rewriter.getI64Type();
  if (!attr_starts.empty()) {
    auto tType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(attr_starts.size())}, i64Type);
    opInpStarts = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::DenseIntElementsAttr::get(tType, attr_starts));
  }
  if (!attr_ends.empty()) {
    auto tType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(attr_ends.size())}, i64Type);
    opInpEnds = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::DenseIntElementsAttr::get(tType, attr_ends));
  }
  if (!attr_axes.empty()) {
    auto tType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(attr_axes.size())}, i64Type);
    opInpAxes = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::DenseIntElementsAttr::get(tType, attr_axes));
  }
  if (!attr_steps.empty()) {
    auto tType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(attr_steps.size())}, i64Type);
    opInpSteps = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::DenseIntElementsAttr::get(tType, attr_steps));
  }

  if (!opInpStarts || !opInpEnds)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " requires starts and ends operands or attributes";

  int64_t numSpecs = inputRank;
  if (auto startsType =
          mlir::dyn_cast<mlir::RankedTensorType>(opInpStarts.getType())) {
    if (startsType.hasRank() && startsType.getRank() == 1 &&
        startsType.getDimSize(0) > 0)
      numSpecs = startsType.getDimSize(0);
  }

  auto zeroIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto oneIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto minusOneIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, -1);
  auto rankIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, inputRank);

  llvm::SmallVector<mlir::Value> startVals(inputRank, nullptr);
  llvm::SmallVector<mlir::Value> stepVals(inputRank, nullptr);
  llvm::SmallVector<mlir::Value> outDimVals(inputRank, nullptr);

  // default slice parameters
  for (int64_t i = 0; i < inputRank; ++i) {
    auto cstIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
    auto dimVal = mlir::tensor::DimOp::create(rewriter, loc, opInput, cstIdx);

    startVals[i] = zeroIdx;
    stepVals[i] = oneIdx;
    outDimVals[i] = dimVal;
  }

  // dynamic starts, ends, steps, and axes
  for (int64_t k = 0; k < numSpecs; ++k) {
    auto idxK = mlir::arith::ConstantIndexOp::create(rewriter, loc, k);

    // start and end values
    auto rawS = mlir::tensor::ExtractOp::create(rewriter, loc, opInpStarts,
                                                mlir::ValueRange{idxK});
    auto rawE = mlir::tensor::ExtractOp::create(rewriter, loc, opInpEnds,
                                                mlir::ValueRange{idxK});

    auto sIdx = mlir::arith::IndexCastOp::create(rewriter, loc,
                                                 rewriter.getIndexType(), rawS);
    auto eIdx = mlir::arith::IndexCastOp::create(rewriter, loc,
                                                 rewriter.getIndexType(), rawE);

    // step value or default to 1
    mlir::Value stIdx = oneIdx;
    if (opInpSteps && !mlir::isa<mlir::NoneType>(opInpSteps.getType())) {
      auto rawSt = mlir::tensor::ExtractOp::create(rewriter, loc, opInpSteps,
                                                   mlir::ValueRange{idxK});
      stIdx = mlir::arith::IndexCastOp::create(rewriter, loc,
                                               rewriter.getIndexType(), rawSt);
    }

    // axis value or default to index k
    mlir::Value rawAx = nullptr;
    if (opInpAxes && !mlir::isa<mlir::NoneType>(opInpAxes.getType())) {
      auto extractedAx = mlir::tensor::ExtractOp::create(
          rewriter, loc, opInpAxes, mlir::ValueRange{idxK});
      rawAx = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), extractedAx);
    } else {
      rawAx = idxK;
    }

    auto isNegAx = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::slt, rawAx, zeroIdx);
    auto posAx = mlir::arith::AddIOp::create(rewriter, loc, rawAx, rankIdx);
    auto normAx =
        mlir::arith::SelectOp::create(rewriter, loc, isNegAx, posAx, rawAx);

    for (int64_t d = 0; d < inputRank; ++d) {
      auto dIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, d);
      auto isMatch = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::eq, normAx, dIdx);

      auto dimVal = mlir::tensor::DimOp::create(rewriter, loc, opInput, dIdx);

      auto isNegSt = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, stIdx, zeroIdx);

      // positive step (st > 0)
      auto isNegS = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, sIdx, zeroIdx);
      auto posS = mlir::arith::AddIOp::create(rewriter, loc, sIdx, dimVal);
      auto ns =
          mlir::arith::SelectOp::create(rewriter, loc, isNegS, posS, sIdx);
      auto minS_pos = mlir::arith::MinSIOp::create(rewriter, loc, ns, dimVal);
      auto normS_pos =
          mlir::arith::MaxSIOp::create(rewriter, loc, zeroIdx, minS_pos);

      auto isNegE = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, eIdx, zeroIdx);
      auto posE = mlir::arith::AddIOp::create(rewriter, loc, eIdx, dimVal);
      auto ne =
          mlir::arith::SelectOp::create(rewriter, loc, isNegE, posE, eIdx);
      auto minE_pos = mlir::arith::MinSIOp::create(rewriter, loc, ne, dimVal);
      auto normE_pos =
          mlir::arith::MaxSIOp::create(rewriter, loc, zeroIdx, minE_pos);

      auto le_pos =
          mlir::arith::SubIOp::create(rewriter, loc, normE_pos, normS_pos);
      auto len_pos =
          mlir::arith::MaxSIOp::create(rewriter, loc, zeroIdx, le_pos);

      auto stMinusOne =
          mlir::arith::SubIOp::create(rewriter, loc, stIdx, oneIdx);
      auto num_pos =
          mlir::arith::AddIOp::create(rewriter, loc, len_pos, stMinusOne);
      auto outLen_pos =
          mlir::arith::DivUIOp::create(rewriter, loc, num_pos, stIdx);

      // negative step (st < 0)
      auto dimMinusOne =
          mlir::arith::SubIOp::create(rewriter, loc, dimVal, oneIdx);
      auto minS_neg =
          mlir::arith::MinSIOp::create(rewriter, loc, ns, dimMinusOne);
      auto normS_neg =
          mlir::arith::MaxSIOp::create(rewriter, loc, minusOneIdx, minS_neg);

      auto minE_neg =
          mlir::arith::MinSIOp::create(rewriter, loc, ne, dimMinusOne);
      auto normE_neg =
          mlir::arith::MaxSIOp::create(rewriter, loc, minusOneIdx, minE_neg);

      auto le_neg =
          mlir::arith::SubIOp::create(rewriter, loc, normS_neg, normE_neg);
      auto len_neg =
          mlir::arith::MaxSIOp::create(rewriter, loc, zeroIdx, le_neg);

      auto absSt = mlir::arith::SubIOp::create(rewriter, loc, zeroIdx, stIdx);
      auto absStMinusOne =
          mlir::arith::SubIOp::create(rewriter, loc, absSt, oneIdx);
      auto num_neg =
          mlir::arith::AddIOp::create(rewriter, loc, len_neg, absStMinusOne);
      auto outLen_neg =
          mlir::arith::DivUIOp::create(rewriter, loc, num_neg, absSt);

      // slice bounds according to step direction
      auto normS = mlir::arith::SelectOp::create(rewriter, loc, isNegSt,
                                                 normS_neg, normS_pos);
      auto outLen = mlir::arith::SelectOp::create(rewriter, loc, isNegSt,
                                                  outLen_neg, outLen_pos);

      // slice parameters for matched axis
      startVals[d] = mlir::arith::SelectOp::create(rewriter, loc, isMatch,
                                                   normS, startVals[d]);
      stepVals[d] = mlir::arith::SelectOp::create(rewriter, loc, isMatch, stIdx,
                                                  stepVals[d]);
      outDimVals[d] = mlir::arith::SelectOp::create(rewriter, loc, isMatch,
                                                    outLen, outDimVals[d]);
    }
  }

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i = 0; i < inputRank; ++i) {
    if (outDatType.isDynamicDim(i))
      dynSizes.push_back(outDimVals[i]);
  }

  auto outBuffer =
      mlir::tensor::EmptyOp::create(rewriter, loc, outDatType, dynSizes);

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_mapes*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      /*callback_body*/
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        llvm::SmallVector<mlir::Value> inpCoords;
        inpCoords.reserve(inputRank);

        for (int64_t d = 0; d < inputRank; ++d) {
          auto loopIdx = mlir::linalg::IndexOp::create(
              nest, nloc, static_cast<uint64_t>(d));
          auto mulVal =
              mlir::arith::MulIOp::create(nest, nloc, loopIdx, stepVals[d]);
          auto coord =
              mlir::arith::AddIOp::create(nest, nloc, startVals[d], mulVal);
          inpCoords.push_back(coord);
        }

        auto extracted =
            mlir::tensor::ExtractOp::create(nest, nloc, opInput, inpCoords);
        mlir::linalg::YieldOp::create(nest, nloc, extracted.getResult());
      });

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
