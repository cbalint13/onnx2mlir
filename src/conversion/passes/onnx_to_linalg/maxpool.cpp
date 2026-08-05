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

#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_MaxPoolOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
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
  auto orgDatType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());

  auto inpElmType = inpDatType.getElementType();
  auto orgElmType = orgDatType.getElementType();

  int64_t inputRank = inpDatType.getRank();
  int64_t spatialRank = inputRank - 2;

  // check
  if (spatialRank != 2)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " only 2D (NCHW) spatial pooling is supported";

  /*
   * Attributes
   */

  auto getI64Array = [&](llvm::StringRef name, int64_t def) {
    llvm::SmallVector<int64_t> vals;
    if (auto attr = op->getAttrOfType<mlir::ArrayAttr>(name)) {
      for (auto val : attr.getAsRange<mlir::IntegerAttr>())
        vals.push_back(val.getInt());
    } else {
      vals.assign(spatialRank, def);
    }
    return vals;
  };

  // kernel_shape
  auto attr_kernel_shape = getI64Array("kernel_shape", 1);

  // strides
  auto attr_strides = getI64Array("strides", 1);

  // dilation
  auto attr_dilations = getI64Array("dilations", 1);

  // pads
  auto attr_pads = getI64Array("pads", 0);

  /*
   *  Affine mappings
   */

  // 6 dims: NCHW + KH, KW
  mlir::AffineExpr dN, dC, dOH, dOW, dKH, dKW;
  mlir::bindDims(op->getContext(), dN, dC, dOH, dOW, dKH, dKW);

  // input:
  // [N, C, OH * stride_h + KH * dilation_h, OW * stride_w + KW * dilation_w]
  auto inputMap = mlir::AffineMap::get(
      /*dimCount=*/6, /*symbolCount=*/0,
      {dN, dC, dOH * attr_strides[0] + dKH * attr_dilations[0],
       dOW * attr_strides[1] + dKW * attr_dilations[1]},
      op->getContext());

  // kernel: [KH, kW]
  auto kernelMap = mlir::AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0,
                                        {dKH, dKW}, op->getContext());

  // output: [N, C, OH, OW]
  auto outputMap = mlir::AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0,
                                        {dN, dC, dOH, dOW}, op->getContext());

  mlir::SmallVector<mlir::AffineMap, 3> indexingMaps = {inputMap, kernelMap,
                                                        outputMap};

  // iterators: [N, C, OH, OW, KH, KW]
  mlir::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      inputRank, mlir::utils::IteratorType::parallel); // N, C, OH, OW
  for (int i = 0; i < spatialRank; ++i)
    iteratorTypes.push_back(mlir::utils::IteratorType::reduction); // KH, KW

  /*
   *  Linalg ops staging
   */

  // padding
  mlir::Value initValue;
  if (mlir::isa<mlir::FloatType>(inpElmType)) {
    auto minFloat = llvm::APFloat::getLargest(
        mlir::cast<mlir::FloatType>(inpElmType).getFloatSemantics(), true);
    initValue = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(inpElmType, minFloat));
  } else {
    auto origInt = mlir::dyn_cast_or_null<mlir::IntegerType>(orgElmType);
    if (origInt && origInt.isUnsigned()) {
      initValue = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(inpElmType, 0));
    } else {
      auto minInt =
          llvm::APInt::getSignedMinValue(inpElmType.getIntOrFloatBitWidth());
      initValue = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(inpElmType, minInt));
    }
  }
  mlir::Value paddedInput = opInput;
  bool hasPadding = llvm::any_of(attr_pads, [](int64_t p) { return p != 0; });
  if (hasPadding) {
    llvm::SmallVector<mlir::OpFoldResult> lowPads;
    llvm::SmallVector<mlir::OpFoldResult> highPads;
    // no padding: N, C
    lowPads.push_back(rewriter.getIndexAttr(0));
    lowPads.push_back(rewriter.getIndexAttr(0));
    highPads.push_back(rewriter.getIndexAttr(0));
    highPads.push_back(rewriter.getIndexAttr(0));

    // pads: [x1_begin, x2_begin... x1_end, x2_end...]
    for (int i = 0; i < spatialRank; ++i) {
      lowPads.push_back(rewriter.getIndexAttr(attr_pads[i]));
      highPads.push_back(rewriter.getIndexAttr(attr_pads[i + spatialRank]));
    }

    auto padOp = mlir::tensor::PadOp::create(
        rewriter, loc, /*resultType=*/nullptr, opInput, lowPads, highPads,
        /*nofold=*/false);

    mlir::Region &region = padOp.getRegion();
    mlir::Block *block = rewriter.createBlock(&region);
    for (int64_t i = 0; i < inputRank; ++i)
      block->addArgument(rewriter.getIndexType(), loc);

    rewriter.setInsertionPointToStart(block);

    mlir::tensor::YieldOp::create(rewriter, loc, initValue);
    rewriter.setInsertionPointAfter(padOp);

    paddedInput = padOp.getResult();
  }

  auto emptyTensor = mlir::tensor::EmptyOp::create(
      rewriter, loc, outDatType.getShape(), outDatType.getElementType());
  auto fillOp = mlir::linalg::FillOp::create(rewriter, loc, initValue,
                                             emptyTensor.getResult());
  auto outBuffer = fillOp.getResult(0);

  auto kernelTensor = mlir::tensor::EmptyOp::create(
      rewriter, loc, llvm::ArrayRef<int64_t>(attr_kernel_shape),
      inpDatType.getElementType());

  auto genericOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_types*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{paddedInput, kernelTensor.getResult()},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingMaps,
      /*iter_types*/ iteratorTypes,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::Value inpVal = args[0];
        mlir::Value outVal = args[2];
        mlir::Value maxVal;
        if (orgElmType.isFloat()) {
          maxVal = mlir::arith::MaximumFOp::create(nest, nloc, inpVal, outVal);
        } else {
          auto origInt = mlir::dyn_cast_or_null<mlir::IntegerType>(orgElmType);
          if (origInt && origInt.isUnsigned()) {
            maxVal = mlir::arith::MaxUIOp::create(nest, nloc, inpVal, outVal);
          } else {
            maxVal = mlir::arith::MaxSIOp::create(nest, nloc, inpVal, outVal);
          }
        }
        mlir::linalg::YieldOp::create(nest, nloc, maxVal);
      });

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
