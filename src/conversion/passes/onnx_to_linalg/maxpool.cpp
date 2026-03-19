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
 * \file src/conversion/passes/onnx_to_linalg/maxpool.cpp
 * \brief ONNX MaxPool operation to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_MaxPoolOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  mlir::Value input = op->getOperand(0);
  mlir::Value result = op->getResult(0);

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());

  if (!inputType || !resType) {
    return mlir::emitError(loc, opName + " operand must be ranked tensor type");
  }

  auto elementType = inputType.getElementType();
  int64_t rank = inputType.getRank();
  int64_t spatialRank = rank - 2;

  if (spatialRank != 2) {
    return mlir::emitError(
        loc, opName + " only 2D (NCHW) spatial pooling is supported");
  }

  // Extract attributes
  auto getI64Array = [&](llvm::StringRef attrName, int64_t defaultValue) {
    llvm::SmallVector<int64_t> values;
    if (auto attr = op->getAttrOfType<mlir::ArrayAttr>(attrName)) {
      for (auto val : attr.getAsRange<mlir::IntegerAttr>())
        values.push_back(val.getInt());
    } else {
      values.assign(spatialRank, defaultValue);
    }
    return values;
  };

  auto kernelShape = getI64Array("kernel_shape", 1);
  auto strides = getI64Array("strides", 1);
  auto dilations = getI64Array("dilations", 1);
  auto pads = getI64Array("pads", 0);

  // Initialize constant padding value
  mlir::Value initValue;
  if (mlir::isa<mlir::FloatType>(elementType)) {
    auto minFloat = llvm::APFloat::getLargest(
        mlir::cast<mlir::FloatType>(elementType).getFloatSemantics(), true);
    initValue = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elementType, minFloat));
  } else {
    auto minInt =
        llvm::APInt::getSignedMinValue(elementType.getIntOrFloatBitWidth());
    initValue = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(elementType, minInt));
  }

  // Handle Padding
  mlir::Value paddedInput = input;
  bool hasPadding = llvm::any_of(pads, [](int64_t p) { return p != 0; });
  if (hasPadding) {
    llvm::SmallVector<mlir::OpFoldResult> lowPads;
    llvm::SmallVector<mlir::OpFoldResult> highPads;
    // N, C dimensions have no padding
    lowPads.push_back(rewriter.getIndexAttr(0));
    lowPads.push_back(rewriter.getIndexAttr(0));
    highPads.push_back(rewriter.getIndexAttr(0));
    highPads.push_back(rewriter.getIndexAttr(0));

    // ONNX pads are [x1_begin, x2_begin... x1_end, x2_end...]
    for (int i = 0; i < spatialRank; ++i) {
      lowPads.push_back(rewriter.getIndexAttr(pads[i]));
      highPads.push_back(rewriter.getIndexAttr(pads[i + spatialRank]));
    }

    auto padOp = mlir::tensor::PadOp::create(
        rewriter, loc, /*resultType=*/nullptr, input, lowPads, highPads,
        /*nofold=*/false);

    mlir::Region &region = padOp.getRegion();
    mlir::Block *block = rewriter.createBlock(&region);
    for (int64_t i = 0; i < rank; ++i)
      block->addArgument(rewriter.getIndexType(), loc);

    rewriter.setInsertionPointToStart(block);
    mlir::tensor::YieldOp::create(rewriter, loc, initValue);
    rewriter.setInsertionPointAfter(padOp);

    paddedInput = padOp.getResult();
  }

  // Output buffer
  auto emptyTensor = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), elementType);
  auto fillOp = mlir::linalg::FillOp::create(rewriter, loc, initValue,
                                             emptyTensor.getResult());
  mlir::Value outBuff = fillOp.getResult(0);

  // Mapping to Linalg Generic
  // Iterators: [N, C, OH, OW, KH, KW]
  mlir::SmallVector<mlir::utils::IteratorType> iterators(
      rank, mlir::utils::IteratorType::parallel); // N, C, OH, OW
  for (int i = 0; i < spatialRank; ++i) {
    iterators.push_back(mlir::utils::IteratorType::reduction); // KH, KW
  }

  auto context = rewriter.getContext();
  // We need 6 dimensions for NCHW + KH, KW
  mlir::AffineExpr dN, dC, dOH, dOW, dKH, dKW;
  mlir::bindDims(context, dN, dC, dOH, dOW, dKH, dKW);

  // Input Map: [n, c, oh * stride_h + kh * dilation_h, ow * stride_w + kw *
  // dilation_w]
  auto inputMap =
      mlir::AffineMap::get(6, 0,
                           {dN, dC, dOH * strides[0] + dKH * dilations[0],
                            dOW * strides[1] + dKW * dilations[1]},
                           context);

  // Kernel Map (dummy for reduction shape): [kh, kw]
  auto kernelMap = mlir::AffineMap::get(6, 0, {dKH, dKW}, context);

  // Output Map: [n, c, oh, ow]
  auto outputMap = mlir::AffineMap::get(6, 0, {dN, dC, dOH, dOW}, context);

  // We need a tensor that represents the kernel shape for the reduction loops
  auto kernelTensor = mlir::tensor::EmptyOp::create(
      rewriter, loc, llvm::ArrayRef<int64_t>(kernelShape), elementType);

  mlir::SmallVector<mlir::AffineMap> indexingMaps = {inputMap, kernelMap,
                                                     outputMap};

  auto genericOp = mlir::linalg::GenericOp::create(
      rewriter, loc, resType,
      mlir::ValueRange{paddedInput, kernelTensor.getResult()},
      mlir::ValueRange{outBuff}, indexingMaps, iterators,
      [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
        mlir::Value inputVal = args[0];
        mlir::Value outVal = args[2];
        mlir::Value maxVal;
        if (mlir::isa<mlir::FloatType>(elementType)) {
          maxVal = mlir::arith::MaximumFOp::create(nest, l, inputVal, outVal);
        } else {
          maxVal = mlir::arith::MaxSIOp::create(nest, l, inputVal, outVal);
        }
        mlir::linalg::YieldOp::create(nest, l, maxVal);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp->getResults());
  return mlir::success();
}

} // namespace onnx2mlir::dialect
