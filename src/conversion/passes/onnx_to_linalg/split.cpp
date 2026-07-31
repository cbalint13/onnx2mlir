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
 * \file src/conversion/passes/onnx_to_linalg/split.cpp
 * \brief ONNX Split operation to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_SplitOp(mlir::Operation *op,
                                         mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  // Extract input operand
  mlir::Value input = op->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());

  if (!inputType) {
    return mlir::emitError(
        loc, opName + " input operand must be ranked tensor type");
  }

  int64_t inputRank = inputType.getRank();
  if (inputRank == 0) {
    return mlir::emitError(loc, opName + " input rank must be greater than 0");
  }

  // Get 'axis' attribute (defaults to 0 if absent)
  int64_t axisValue = 0;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis")) {
    axisValue = axisAttr.getInt();
  }

  // Handle negative axis index normalization
  if (axisValue < 0) {
    axisValue += inputRank;
  }

  if (axisValue < 0 || axisValue >= inputRank) {
    return mlir::emitError(loc, opName + " axis attribute is out of range");
  }

  unsigned numResults = op->getNumResults();
  if (numResults == 0) {
    return mlir::emitError(
        loc, opName + " operation must produce at least 1 result");
  }

  llvm::SmallVector<int64_t, 4> splitSizes;
  if (auto splitAttr = op->getAttrOfType<mlir::ArrayAttr>("split")) {
    for (auto attr : splitAttr) {
      splitSizes.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    }
  } else if (auto inputSplitAttr =
                 op->getAttrOfType<mlir::ArrayAttr>("input_split")) {
    for (auto attr : inputSplitAttr) {
      splitSizes.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    }
  } else if (op->getNumOperands() > 1) {
    mlir::Value splitOperand = op->getOperand(1);
    if (splitOperand && !mlir::isa<mlir::NoneType>(splitOperand.getType())) {
      if (auto constOp = splitOperand.getDefiningOp()) {
        if (auto denseAttr =
                constOp->getAttrOfType<mlir::DenseElementsAttr>("value")) {
          for (auto val : denseAttr.getValues<mlir::APInt>()) {
            splitSizes.push_back(val.getSExtValue());
          }
        }
      }
    }
  }

  // Populate split sizes for equal splitting if operand/attr was omitted
  if (splitSizes.empty()) {
    int64_t axisDim = inputType.getDimSize(axisValue);
    if (axisDim != mlir::ShapedType::kDynamic && numResults > 0) {
      int64_t equalSize = axisDim / numResults;
      for (unsigned i = 0; i < numResults; ++i) {
        splitSizes.push_back(equalSize);
      }
    }
  }

  llvm::SmallVector<mlir::Value, 4> newResults;
  int64_t currentOffset = 0;

  for (unsigned i = 0; i < numResults; ++i) {
    mlir::Value res = op->getResult(i);
    auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

    if (!resType) {
      return mlir::emitError(loc,
                             opName + " result must be ranked tensor type");
    }

    if (resType.getRank() != inputRank) {
      return mlir::emitError(
          loc, opName + " result rank must match input tensor rank");
    }

    // Determine target dimension size along the split axis
    int64_t dimSize = resType.getDimSize(axisValue);
    if (dimSize == mlir::ShapedType::kDynamic && i < splitSizes.size()) {
      dimSize = splitSizes[i];
    }

    // Construct slice parameters (offsets, sizes, strides)
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;

    for (int64_t d = 0; d < inputRank; ++d) {
      strides.push_back(rewriter.getIndexAttr(1));
      if (d == axisValue) {
        offsets.push_back(rewriter.getIndexAttr(currentOffset));
        if (dimSize != mlir::ShapedType::kDynamic) {
          sizes.push_back(rewriter.getIndexAttr(dimSize));
        } else {
          sizes.push_back(
              mlir::tensor::DimOp::create(rewriter, loc, input, d).getResult());
        }
      } else {
        offsets.push_back(rewriter.getIndexAttr(0));
        if (resType.isDynamicDim(d)) {
          sizes.push_back(
              mlir::tensor::DimOp::create(rewriter, loc, input, d).getResult());
        } else {
          sizes.push_back(rewriter.getIndexAttr(resType.getDimSize(d)));
        }
      }
    }

    mlir::Value slice = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, resType, input, offsets, sizes, strides);

    // Tag the slice for the transform dialect
    slice.getDefiningOp()->setAttr("transform.target_tag",
                                   rewriter.getStringAttr(opName));

    newResults.push_back(slice);

    // Advance offset along split axis
    if (dimSize != mlir::ShapedType::kDynamic) {
      currentOffset += dimSize;
    }
  }

  rewriter.replaceOp(op, newResults);
  return mlir::success();
}

} // namespace onnx2mlir::dialect
