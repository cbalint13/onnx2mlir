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
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_SplitOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                     const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opInputSplit = (op->getNumOperands() > 1 &&
                       !mlir::isa<mlir::NoneType>(op->getOperand(1).getType()))
                          ? convRewriter.getRemappedValue(op->getOperand(1))
                          : nullptr;

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());

  int64_t inputRank = inpDatType.getRank();

  unsigned numOutputs = op->getNumResults();

  if (numOutputs == 0)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " operation must produce at least 1 result";

  /*
   * Attributes
   */

  // axis
  int64_t attr_axis = 0;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    attr_axis = axisAttr.getInt();
  if (attr_axis < -inputRank || attr_axis >= inputRank)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " invalid axis: " << attr_axis;
  if (attr_axis < 0)
    attr_axis += inputRank;
  attr_axis = std::clamp<int64_t>(attr_axis, 0, inputRank);

  // split
  llvm::SmallVector<int64_t, 4> attr_split;
  auto arrayAttr = op->getAttrOfType<mlir::ArrayAttr>("split");
  if (!arrayAttr)
    arrayAttr = op->getAttrOfType<mlir::ArrayAttr>("input_split");
  if (arrayAttr) {
    for (auto intAttr : arrayAttr.getAsRange<mlir::IntegerAttr>())
      attr_split.push_back(intAttr.getInt());
  }
  if (attr_split.empty() && !opInputSplit) {
    int64_t axisDim = inpDatType.getDimSize(attr_axis);
    if (axisDim != mlir::ShapedType::kDynamic) {
      int64_t equalSize = axisDim / numOutputs;
      attr_split.assign(numOutputs, equalSize);
    }
  }

  /*
   *  Linalg ops staging
   */

  mlir::Value currOfsetVal;
  int64_t currentOffsetStatic = 0;
  if (attr_split.empty())
    currOfsetVal = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

  llvm::SmallVector<mlir::Value, 4> newOutputs;

  for (unsigned i = 0; i < numOutputs; ++i) {
    auto opOut = convRewriter.getRemappedValue(op->getResult(i));
    auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOut.getType());

    if (outDatType.getRank() != inputRank)
      return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
             << opName << " result rank must match input tensor rank";

    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;

    for (int64_t d = 0; d < inputRank; ++d) {
      strides.push_back(rewriter.getIndexAttr(1));

      if (d == attr_axis) {
        if (!attr_split.empty()) {
          // use split attribute
          offsets.push_back(rewriter.getIndexAttr(currentOffsetStatic));
          sizes.push_back(rewriter.getIndexAttr(attr_split[i]));
          currentOffsetStatic += attr_split[i];
        } else {
          // use split operand
          offsets.push_back(currOfsetVal);

          mlir::Value sliceSizeVal;
          if (opInputSplit) {
            auto idxConst =
                mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
            auto extractedElem = mlir::tensor::ExtractOp::create(
                rewriter, loc, opInputSplit, mlir::ValueRange{idxConst});
            sliceSizeVal = mlir::arith::IndexCastOp::create(
                rewriter, loc, rewriter.getIndexType(), extractedElem);
          } else {
            // equal split (default)
            auto axisDimVal =
                mlir::tensor::DimOp::create(rewriter, loc, opInput, attr_axis);
            auto numOutputsConst =
                mlir::arith::ConstantIndexOp::create(rewriter, loc, numOutputs);
            sliceSizeVal = mlir::arith::DivUIOp::create(
                rewriter, loc, axisDimVal, numOutputsConst);
          }

          if (outDatType.isDynamicDim(attr_axis))
            sizes.push_back(sliceSizeVal);
          else
            sizes.push_back(
                rewriter.getIndexAttr(outDatType.getDimSize(attr_axis)));

          currOfsetVal = mlir::arith::AddIOp::create(
              rewriter, loc, currOfsetVal, sliceSizeVal);
        }
      } else {
        offsets.push_back(rewriter.getIndexAttr(0));
        if (outDatType.isDynamicDim(d)) {
          auto dimOp = mlir::tensor::DimOp::create(rewriter, loc, opInput, d);
          sizes.push_back(dimOp.getResult());
        } else {
          sizes.push_back(rewriter.getIndexAttr(outDatType.getDimSize(d)));
        }
      }
    }

    auto sliceOp = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, outDatType, opInput, offsets, sizes, strides);

    sliceOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

    newOutputs.push_back(sliceOp);
  }

  rewriter.replaceOp(op, newOutputs);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
