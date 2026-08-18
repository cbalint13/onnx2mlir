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
 * \file src/conversion/passes/onnx_to_linalg/conv.cpp
 * \brief ONNX Conv operation to Linalg lowering
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
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_ConvOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto *ctx = op->getContext();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInput = convRewriter.getRemappedValue(op->getOperand(0));
  auto opWeight = convRewriter.getRemappedValue(op->getOperand(1));
  auto opBias = op->getNumOperands() > 2
                    ? convRewriter.getRemappedValue(op->getOperand(2))
                    : nullptr;
  auto opResult = op->getResult(0);
  auto opOutput = convRewriter.getRemappedValue(opResult);

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto wgtDatType = mlir::dyn_cast<mlir::RankedTensorType>(opWeight.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto inpElmType = inpDatType.getElementType();
  auto wgtElmType = wgtDatType.getElementType();
  auto outElmType = outDatType.getElementType();

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

  // group
  int64_t attr_group = 1;
  if (auto groupAttr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    attr_group = groupAttr.getInt();
  if (attr_group <= 0)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " group attribute must be positive";

  // kernel_shape
  auto attr_kernel_shape = getI64Array("kernel_shape", {});
  if (attr_kernel_shape.empty()) {
    for (int64_t i = 2; i < wgtDatType.getRank(); ++i)
      attr_kernel_shape.push_back(wgtDatType.getDimSize(i));
  }
  if (attr_kernel_shape[0] != wgtDatType.getDimSize(2) ||
      attr_kernel_shape[1] != wgtDatType.getDimSize(3))
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " 'kernel_shape' not match weight kernel shape";

  // strides
  auto attr_strides = getI64Array("strides", {1, 1});

  // dilations
  auto attr_dilations = getI64Array("dilations", {1, 1});

  // pads
  auto padsAttr = op->getAttrOfType<mlir::ArrayAttr>("pads");

  int64_t inChannels = inpDatType.getDimSize(1);
  int64_t outChannels = outDatType.getDimSize(1);
  int64_t weightOutChannels = wgtDatType.getDimSize(0);
  int64_t weightCPerGroup = wgtDatType.getDimSize(1);

  if (!mlir::ShapedType::isDynamic(inChannels) &&
      inChannels % attr_group != 0) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " input channels must be divisible by group";
  }

  if (!mlir::ShapedType::isDynamic(outChannels) &&
      outChannels % attr_group != 0) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " output channels must be divisible by group";
  }

  // weight shape: [M, C / group, KH, KW]
  int64_t cPerGroup = weightCPerGroup;
  int64_t fPerGroup = weightOutChannels / attr_group;

  /*
   *  Linalg ops staging
   */

  // padding
  auto paddedInput = opInput;
  if (padsAttr) {
    llvm::SmallVector<int64_t> p;
    for (auto a : padsAttr.getValue())
      p.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());

    auto isPositive = [](int64_t val) { return val > 0; };
    if (llvm::any_of(p, isPositive)) {
      llvm::SmallVector<mlir::OpFoldResult> low = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(p[0]), rewriter.getIndexAttr(p[1])};
      llvm::SmallVector<mlir::OpFoldResult> high = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(p[2]), rewriter.getIndexAttr(p[3])};
      llvm::SmallVector<int64_t> pShape = {
          inpDatType.getDimSize(0), inpDatType.getDimSize(1),
          inpDatType.getDimSize(2) + p[0] + p[2],
          inpDatType.getDimSize(3) + p[1] + p[3]};

      auto pType = mlir::RankedTensorType::get(pShape, inpElmType);
      auto padVal = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(inpElmType));
      auto padOp = mlir::tensor::PadOp::create(rewriter, loc, pType, opInput,
                                               low, high, padVal,
                                               /*nofold=*/false);
      paddedInput = padOp.getResult();
    }
  }

  // output buffer
  mlir::Value zero = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(outElmType));
  mlir::Value outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, loc, outDatType.getShape(), outElmType);
  auto fillBuffer = mlir::linalg::FillOp::create(
      rewriter, loc, mlir::TypeRange{outDatType}, zero, outBuffer);

  mlir::Value convOut;

  if (attr_group == 1) {
    // ---------------------------------------------------------------------
    // Group == 1: standard 4D convolution (7D iteration space)
    // Loops: n, f, oh, ow (parallel), c, kh, kw (reduction)
    // ---------------------------------------------------------------------
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes = {
        mlir::utils::IteratorType::parallel,  // n (batch)
        mlir::utils::IteratorType::parallel,  // f (output channel)
        mlir::utils::IteratorType::parallel,  // oh (output height)
        mlir::utils::IteratorType::parallel,  // ow (output width)
        mlir::utils::IteratorType::reduction, // c (input channel)
        mlir::utils::IteratorType::reduction, // kh (kernel height)
        mlir::utils::IteratorType::reduction  // kw (kernel width)
    };

    auto dN = rewriter.getAffineDimExpr(0);
    auto dF = rewriter.getAffineDimExpr(1);
    auto dOH = rewriter.getAffineDimExpr(2);
    auto dOW = rewriter.getAffineDimExpr(3);
    auto dC = rewriter.getAffineDimExpr(4);
    auto dKH = rewriter.getAffineDimExpr(5);
    auto dKW = rewriter.getAffineDimExpr(6);

    // input map: [n, c, oh*sh + kh*dh, ow*sw + kw*dw]
    auto inputMap = mlir::AffineMap::get(
        /*dimCount=*/7, /*symbolCount=*/0,
        /*results*/
        {dN, dC, dOH * attr_strides[0] + dKH * attr_dilations[0],
         dOW * attr_strides[1] + dKW * attr_dilations[1]},
        ctx);
    // weight map: [f, c, kh, kw]
    auto weightMap = mlir::AffineMap::get(/*dimCount=*/7, /*symbolCount=*/0,
                                          /*results*/ {dF, dC, dKH, dKW}, ctx);
    // output map: [n, f, oh, ow]
    auto outputMap = mlir::AffineMap::get(/*dimCount=*/7, /*symbolCount=*/0,
                                          /*results*/ {dN, dF, dOH, dOW}, ctx);

    llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {inputMap, weightMap,
                                                          outputMap};

    auto convRes = mlir::linalg::GenericOp::create(
        /*op_builder*/ rewriter, /*src_location*/ loc,
        /*result_types*/ mlir::TypeRange{outDatType},
        /*input_values*/ mlir::ValueRange{paddedInput, opWeight},
        /*output_values*/ fillBuffer.getResult(0),
        /**affine_maps*/ indexingMaps,
        /*iter_types*/ iteratorTypes,
        /*builder_callback*/
        [&](/*op_builder*/ mlir::OpBuilder &nest,
            /*src_location*/ mlir::Location nloc,
            /*value_args*/ mlir::ValueRange args) {
          auto mul = mlir::arith::MulFOp::create(nest, nloc, args[0], args[1]);
          auto add = mlir::arith::AddFOp::create(nest, nloc, mul, args[2]);
          mlir::linalg::YieldOp::create(nest, nloc, add.getResult());
        });

    convOut = convRes.getResult(0);

  } else {
    // ---------------------------------------------------------------------
    // Group > 1: grouped convolution using 5D reshaped tensors
    //   input [N, C, H, W]       -> [N, G, C/G, H, W]
    //   weight [M, C/G, KH, KW]  -> [G, M/G, C/G, KH, KW]
    //   output [N, M, OH, OW]    -> [N, G, M/G, OH, OW]
    // Loops: g (group), n, f_g, oh, ow (parallel), c_g, kh, kw (reduction)
    // ---------------------------------------------------------------------
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes = {
        mlir::utils::IteratorType::parallel,  // g (group)
        mlir::utils::IteratorType::parallel,  // n (batch)
        mlir::utils::IteratorType::parallel,  // f_g (output channel per group)
        mlir::utils::IteratorType::parallel,  // oh (output height)
        mlir::utils::IteratorType::parallel,  // ow (output width)
        mlir::utils::IteratorType::reduction, // c_g (input channel per group)
        mlir::utils::IteratorType::reduction, // kh (kernel height)
        mlir::utils::IteratorType::reduction  // kw (kernel width)
    };

    auto pInputType = mlir::cast<mlir::RankedTensorType>(paddedInput.getType());

    auto inp5DType = mlir::RankedTensorType::get(
        {pInputType.getDimSize(0), attr_group, cPerGroup,
         pInputType.getDimSize(2), pInputType.getDimSize(3)},
        inpElmType);
    auto wgt5DType = mlir::RankedTensorType::get(
        {attr_group, fPerGroup, cPerGroup, attr_kernel_shape[0],
         attr_kernel_shape[1]},
        wgtElmType);
    auto out5DType = mlir::RankedTensorType::get(
        {outDatType.getDimSize(0), attr_group, fPerGroup,
         outDatType.getDimSize(2), outDatType.getDimSize(3)},
        outElmType);

    auto dG = rewriter.getAffineDimExpr(0);
    auto dN = rewriter.getAffineDimExpr(1);
    auto dFg = rewriter.getAffineDimExpr(2);
    auto dOH = rewriter.getAffineDimExpr(3);
    auto dOW = rewriter.getAffineDimExpr(4);
    auto dCg = rewriter.getAffineDimExpr(5);
    auto dKH = rewriter.getAffineDimExpr(6);
    auto dKW = rewriter.getAffineDimExpr(7);

    // input map (5D): [n, g, c_g, oh*sh + kh*dh, ow*sw + kw*dw]
    auto inputMap = mlir::AffineMap::get(
        /*dimCount=*/8, /*symbolCount=*/0,
        /*results*/
        {dN, dG, dCg, dOH * attr_strides[0] + dKH * attr_dilations[0],
         dOW * attr_strides[1] + dKW * attr_dilations[1]},
        /*context*/ ctx);
    // weight map (5D): [g, f_g, c_g, kh, kw]
    auto weightMap = mlir::AffineMap::get(/*dimCount=*/8, /*symbolCount=*/0,
                                          /*results*/ {dG, dFg, dCg, dKH, dKW},
                                          /*context*/ ctx);
    // output map (5D): [n, g, f_g, oh, ow]
    auto outputMap = mlir::AffineMap::get(/*dimCount=*/8, /*symbolCount=*/0,
                                          /*results*/ {dN, dG, dFg, dOH, dOW},
                                          /*context*/ ctx);

    llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {inputMap, weightMap,
                                                          outputMap};

    llvm::SmallVector<mlir::ReassociationIndices, 4> inputReassoc = {
        {0}, {1, 2}, {3}, {4}};
    llvm::SmallVector<mlir::ReassociationIndices, 4> weightReassoc = {
        {0, 1}, {2}, {3}, {4}};
    llvm::SmallVector<mlir::ReassociationIndices, 4> outputReassoc = {
        {0}, {1, 2}, {3}, {4}};

    auto inp5D = mlir::tensor::ExpandShapeOp::create(rewriter, loc, inp5DType,
                                                     paddedInput, inputReassoc);
    auto wgt5D = mlir::tensor::ExpandShapeOp::create(rewriter, loc, wgt5DType,
                                                     opWeight, weightReassoc);
    auto initBuff5D = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, out5DType, fillBuffer.getResult(0), outputReassoc);

    auto convRes5D = mlir::linalg::GenericOp::create(
        /*op_builder*/ rewriter, /*src_location*/ loc,
        /*result_type*/ mlir::TypeRange{out5DType},
        /*input_values*/ mlir::ValueRange{inp5D, wgt5D},
        /*output_values*/ mlir::ValueRange{initBuff5D},
        /*affine_maps*/ indexingMaps,
        /*iter_types*/ iteratorTypes,
        /*builder_callback*/
        [&](/*op_builder*/ mlir::OpBuilder &nest,
            /*src_location*/ mlir::Location nloc,
            /*value_args*/ mlir::ValueRange args) {
          auto mul = mlir::arith::MulFOp::create(nest, nloc, args[0], args[1]);
          auto add = mlir::arith::AddFOp::create(nest, nloc, mul, args[2]);
          mlir::linalg::YieldOp::create(nest, nloc, add.getResult());
        });

    // collapse 5D output back to 4D [N, M, OH, OW]
    auto convRes = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, outDatType, convRes5D.getResult(0), outputReassoc);

    convOut = convRes.getResult();
  }

  // bias
  if (opBias && !mlir::isa<mlir::NoneType>(opBias.getType())) {
    llvm::SmallVector<mlir::utils::IteratorType> biasIters(
        4, mlir::utils::IteratorType::parallel);

    // mapped to output [N, F, OH, OW] via dim 1
    auto bMap = mlir::AffineMap::get(4, 0, {rewriter.getAffineDimExpr(1)}, ctx);
    auto rMap = rewriter.getMultiDimIdentityMap(4);

    llvm::SmallVector<mlir::AffineMap, 2> indexingMaps = {bMap, rMap};

    auto convRes = mlir::linalg::GenericOp::create(
        /*op_builder*/ rewriter, /*src_location*/ loc,
        /*result_type*/ mlir::TypeRange{outDatType},
        /*input_values*/ mlir::ValueRange{opBias},
        /*output_values*/ mlir::ValueRange{convOut},
        /*affine_maps*/ indexingMaps,
        /*iter_types*/ biasIters,
        /*builder_callback*/
        [&](/*op_builder*/ mlir::OpBuilder &nest,
            /*src_location*/ mlir::Location nloc,
            /*value_args*/ mlir::ValueRange args) {
          auto addB = mlir::arith::AddFOp::create(nest, nloc, args[0], args[1]);
          mlir::linalg::YieldOp::create(nest, nloc, addB.getResult());
        });

    convOut = convRes.getResult(0);
  }

  rewriter.replaceOp(op, convOut);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
