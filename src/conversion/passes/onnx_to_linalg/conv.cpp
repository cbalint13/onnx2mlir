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
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_ConvOp(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto *ctx = op->getContext();
  auto opName = op->getName().getStringRef();

  // Get operands
  mlir::Value input = op->getOperand(0);
  mlir::Value weight = op->getOperand(1);
  mlir::Value bias = op->getNumOperands() > 2 ? op->getOperand(2) : nullptr;

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(weight.getType());
  auto resType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());

  if (!inputType || !weightType || !resType)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " operand must be ranked tensor");

  int64_t group = 1;
  if (auto groupAttr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    group = groupAttr.getInt();

  if (group <= 0)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName + " group attribute must be positive");

  int64_t inChannels = inputType.getDimSize(1);
  int64_t outChannels = resType.getDimSize(1);
  int64_t weightOutChannels = weightType.getDimSize(0);
  int64_t weightCPerGroup = weightType.getDimSize(1);

  if (!mlir::ShapedType::isDynamic(inChannels) && inChannels % group != 0) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName +
                               " input channels must be divisible by group");
  }

  if (!mlir::ShapedType::isDynamic(outChannels) && outChannels % group != 0) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter),
                           opName +
                               " output channels must be divisible by group");
  }

  // ONNX weight shape: [M, C / group, KH, KW]
  int64_t cPerGroup = weightCPerGroup;
  int64_t fPerGroup = weightOutChannels / group;

  // Extract Attributes
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

  auto strides = getI64Array("strides", {1, 1});
  auto dilations = getI64Array("dilations", {1, 1});
  auto padsAttr = op->getAttrOfType<mlir::ArrayAttr>("pads");

  // Handle padding
  mlir::Value paddedInput = input;
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
          inputType.getDimSize(0), inputType.getDimSize(1),
          inputType.getDimSize(2) + p[0] + p[2],
          inputType.getDimSize(3) + p[1] + p[3]};
      auto pType =
          mlir::RankedTensorType::get(pShape, inputType.getElementType());

      // Create the padding value constant
      mlir::Value padVal = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(inputType.getElementType()));

      auto padOp = mlir::tensor::PadOp::create(rewriter, loc, pType, input, low,
                                               high, padVal,
                                               /*nofold=*/false);
      paddedInput = padOp.getResult();
    }
  }

  // Create zero init output buffer
  auto outBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), resType.getElementType());
  mlir::Value zero = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(resType.getElementType()));
  auto fill = mlir::linalg::FillOp::create(
      rewriter, loc, mlir::TypeRange{resType}, zero, outBuff.getResult());
  mlir::Value initBuff = fill.getResult(0);

  mlir::Value convRes;

  if (group == 1) {
    // ---------------------------------------------------------------------
    // Group == 1: Standard 4D convolution (7D iteration space)
    // Loops: n, f, oh, ow (parallel), c, kh, kw (reduction)
    // ---------------------------------------------------------------------
    llvm::SmallVector<mlir::utils::IteratorType> iters = {
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

    // Input Map: [n, c, oh*sh + kh*dh, ow*sw + kw*dw]
    auto inMap =
        mlir::AffineMap::get(7, 0,
                             {dN, dC, dOH * strides[0] + dKH * dilations[0],
                              dOW * strides[1] + dKW * dilations[1]},
                             ctx);

    // Weight Map: [f, c, kh, kw]
    auto wMap = mlir::AffineMap::get(7, 0, {dF, dC, dKH, dKW}, ctx);

    // Output Map: [n, f, oh, ow]
    auto outMap = mlir::AffineMap::get(7, 0, {dN, dF, dOH, dOW}, ctx);

    convRes = mlir::linalg::GenericOp::create(
                  rewriter, loc, resType, mlir::ValueRange{paddedInput, weight},
                  initBuff,
                  llvm::ArrayRef<mlir::AffineMap>{inMap, wMap, outMap}, iters,
                  [&](mlir::OpBuilder &nest, mlir::Location l,
                      mlir::ValueRange args) {
                    mlir::Value mul =
                        mlir::arith::MulFOp::create(nest, l, args[0], args[1]);
                    mlir::Value add =
                        mlir::arith::AddFOp::create(nest, l, mul, args[2]);
                    mlir::linalg::YieldOp::create(nest, l, add);
                  })
                  .getResult(0);
  } else {
    // ---------------------------------------------------------------------
    // Group > 1: Grouped Convolution using 5D Reshaped Tensors
    // Input [N, C, H, W]       -> [N, G, C/G, H, W]
    // Weight [M, C/G, KH, KW]  -> [G, M/G, C/G, KH, KW]
    // Output [N, M, OH, OW]    -> [N, G, M/G, OH, OW]
    // ---------------------------------------------------------------------
    auto pInputType = mlir::cast<mlir::RankedTensorType>(paddedInput.getType());

    auto in5DType = mlir::RankedTensorType::get(
        {pInputType.getDimSize(0), group, cPerGroup, pInputType.getDimSize(2),
         pInputType.getDimSize(3)},
        pInputType.getElementType());
    auto w5DType = mlir::RankedTensorType::get({group, fPerGroup, cPerGroup,
                                                weightType.getDimSize(2),
                                                weightType.getDimSize(3)},
                                               weightType.getElementType());
    auto out5DType = mlir::RankedTensorType::get(
        {resType.getDimSize(0), group, fPerGroup, resType.getDimSize(2),
         resType.getDimSize(3)},
        resType.getElementType());

    llvm::SmallVector<mlir::ReassociationIndices, 4> inReassoc = {
        {0}, {1, 2}, {3}, {4}};
    llvm::SmallVector<mlir::ReassociationIndices, 4> wReassoc = {
        {0, 1}, {2}, {3}, {4}};
    llvm::SmallVector<mlir::ReassociationIndices, 4> outReassoc = {
        {0}, {1, 2}, {3}, {4}};

    auto in5D = mlir::tensor::ExpandShapeOp::create(rewriter, loc, in5DType,
                                                    paddedInput, inReassoc);
    auto w5D = mlir::tensor::ExpandShapeOp::create(rewriter, loc, w5DType,
                                                   weight, wReassoc);
    auto initBuff5D = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, out5DType, initBuff, outReassoc);

    // Loop structure (8D): g (group), n, f_g (parallel), oh, ow (parallel),
    // c_g, kh, kw (reduction)
    llvm::SmallVector<mlir::utils::IteratorType> iters = {
        mlir::utils::IteratorType::parallel,  // g (group)
        mlir::utils::IteratorType::parallel,  // n (batch)
        mlir::utils::IteratorType::parallel,  // f_g (output channel per group)
        mlir::utils::IteratorType::parallel,  // oh (output height)
        mlir::utils::IteratorType::parallel,  // ow (output width)
        mlir::utils::IteratorType::reduction, // c_g (input channel per group)
        mlir::utils::IteratorType::reduction, // kh (kernel height)
        mlir::utils::IteratorType::reduction  // kw (kernel width)
    };

    auto dG = rewriter.getAffineDimExpr(0);
    auto dN = rewriter.getAffineDimExpr(1);
    auto dFg = rewriter.getAffineDimExpr(2);
    auto dOH = rewriter.getAffineDimExpr(3);
    auto dOW = rewriter.getAffineDimExpr(4);
    auto dCg = rewriter.getAffineDimExpr(5);
    auto dKH = rewriter.getAffineDimExpr(6);
    auto dKW = rewriter.getAffineDimExpr(7);

    // Input Map (5D): [n, g, c_g, oh*sh + kh*dh, ow*sw + kw*dw]
    auto inMap = mlir::AffineMap::get(8, 0,
                                      {dN, dG, dCg,
                                       dOH * strides[0] + dKH * dilations[0],
                                       dOW * strides[1] + dKW * dilations[1]},
                                      ctx);

    // Weight Map (5D): [g, f_g, c_g, kh, kw]
    auto wMap = mlir::AffineMap::get(8, 0, {dG, dFg, dCg, dKH, dKW}, ctx);

    // Output Map (5D): [n, g, f_g, oh, ow]
    auto outMap = mlir::AffineMap::get(8, 0, {dN, dG, dFg, dOH, dOW}, ctx);

    mlir::Value convRes5D =
        mlir::linalg::GenericOp::create(
            rewriter, loc, out5DType,
            mlir::ValueRange{in5D.getResult(), w5D.getResult()},
            initBuff5D.getResult(),
            llvm::ArrayRef<mlir::AffineMap>{inMap, wMap, outMap}, iters,
            [&](mlir::OpBuilder &nest, mlir::Location l,
                mlir::ValueRange args) {
              mlir::Value mul =
                  mlir::arith::MulFOp::create(nest, l, args[0], args[1]);
              mlir::Value add =
                  mlir::arith::AddFOp::create(nest, l, mul, args[2]);
              mlir::linalg::YieldOp::create(nest, l, add);
            })
            .getResult(0);

    // Collapse 5D output back to 4D [N, M, OH, OW]
    convRes = mlir::tensor::CollapseShapeOp::create(rewriter, loc, resType,
                                                    convRes5D, outReassoc)
                  .getResult();
  }

  // Handle Bias as a separate parallel GenericOp
  mlir::Value finalResult = convRes;
  if (bias && !mlir::isa<mlir::NoneType>(bias.getType())) {
    llvm::SmallVector<mlir::utils::IteratorType> biasIters(
        4, mlir::utils::IteratorType::parallel);
    // Bias [F] mapped to Output [N, F, OH, OW] via dim 1
    auto bMap = mlir::AffineMap::get(4, 0, {rewriter.getAffineDimExpr(1)}, ctx);
    auto rMap = rewriter.getMultiDimIdentityMap(4);

    finalResult = mlir::linalg::GenericOp::create(
                      rewriter, loc, resType, mlir::ValueRange{bias}, convRes,
                      llvm::ArrayRef<mlir::AffineMap>{bMap, rMap}, biasIters,
                      [&](mlir::OpBuilder &nest, mlir::Location l,
                          mlir::ValueRange args) {
                        mlir::Value addB = mlir::arith::AddFOp::create(
                            nest, l, args[0], args[1]);
                        mlir::linalg::YieldOp::create(nest, l, addB);
                      })
                      .getResult(0);
  }

  // Set transform tag for downstream optimization
  auto *finalOp = finalResult.getDefiningOp();
  if (finalOp)
    finalOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, finalResult);
  return mlir::success();
}

} // namespace onnx2mlir::dialect
