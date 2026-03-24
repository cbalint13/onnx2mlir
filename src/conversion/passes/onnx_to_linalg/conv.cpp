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
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

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
    return mlir::emitError(loc, opName + " operand must be ranked tensor");

  auto groupAttr = op->getAttrOfType<mlir::IntegerAttr>("group");
  if (groupAttr && groupAttr.getInt() != 1)
    return mlir::emitError(loc, opName + " currently only supports group=1");

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

  // Define genericOp structure
  // Loops: n, f, oh, ow (parallel), c, kh, kw (reduction)
  llvm::SmallVector<mlir::utils::IteratorType> iters = {
      mlir::utils::IteratorType::parallel,  // n
      mlir::utils::IteratorType::parallel,  // f
      mlir::utils::IteratorType::parallel,  // oh
      mlir::utils::IteratorType::parallel,  // ow
      mlir::utils::IteratorType::reduction, // c
      mlir::utils::IteratorType::reduction, // kh
      mlir::utils::IteratorType::reduction  // kw
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

  mlir::Value convRes =
      mlir::linalg::GenericOp::create(
          rewriter, loc, resType, mlir::ValueRange{paddedInput, weight},
          initBuff, llvm::ArrayRef<mlir::AffineMap>{inMap, wMap, outMap}, iters,
          [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
            mlir::Value mul =
                mlir::arith::MulFOp::create(nest, l, args[0], args[1]);
            mlir::Value add =
                mlir::arith::AddFOp::create(nest, l, mul, args[2]);
            mlir::linalg::YieldOp::create(nest, l, add);
          })
          .getResult(0);

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
