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
 * \file src/conversion/passes/onnx_to_linalg/gemm.cpp
 * \brief ONNX Gemm operation to Linalg lowering
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
OnnxToLinalg_GemmOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  /*
   * I/O Values
   */

  auto opInputA = convRewriter.getRemappedValue(op->getOperand(0));
  auto opInputB = convRewriter.getRemappedValue(op->getOperand(1));
  auto opInputC = (op->getNumOperands() > 2 &&
                   !mlir::isa<mlir::NoneType>(op->getOperand(2).getType()))
                      ? convRewriter.getRemappedValue(op->getOperand(2))
                      : nullptr;
  auto opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto outElmType = outDatType.getElementType();

  int64_t outputRank = outDatType.getRank();

  /*
   * Attributes
   */

  // alpha
  float attr_alpha = 1.0f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
    attr_alpha = attr.getValueAsDouble();

  // beta
  float attr_beta = 1.0f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("beta"))
    attr_beta = attr.getValueAsDouble();

  // transA
  int64_t attr_transA = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("transA"))
    attr_transA = attr.getInt();

  // transB
  int64_t attr_transB = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("transB"))
    attr_transB = attr.getInt();

  /*
   *  Affine mappings
   */

  mlir::AffineExpr m, n, k;
  mlir::bindDims(op->getContext(), m, n, k);
  auto mapA = attr_transA
                  ? mlir::AffineMap::get(3, 0, {k, m}, op->getContext())
                  : mlir::AffineMap::get(3, 0, {m, k}, op->getContext());
  auto mapB = attr_transB
                  ? mlir::AffineMap::get(3, 0, {n, k}, op->getContext())
                  : mlir::AffineMap::get(3, 0, {k, n}, op->getContext());
  auto mapY = mlir::AffineMap::get(3, 0, {m, n}, op->getContext());

  mlir::SmallVector<mlir::AffineMap, 3> indexingGemmMaps = {mapA, mapB, mapY};

  // indices: m (row), n (col), k (reduction)
  mlir::SmallVector<mlir::utils::IteratorType> iteratorGemmTypes = {
      mlir::utils::IteratorType::parallel, // m
      mlir::utils::IteratorType::parallel, // n
      mlir::utils::IteratorType::reduction // k
  };

  /*
   *  Linalg ops staging
   */

  mlir::Value outBuffer;

  if (opInputC) {
    auto cType = mlir::dyn_cast<mlir::RankedTensorType>(opInputC.getType());
    // use bias
    if ((cType.getShape() == outDatType.getShape()) && (attr_beta == 1.0f)) {
      outBuffer = opInputC;
    } else {
      // use bias and beta
      mlir::SmallVector<mlir::AffineExpr> cExprs;
      int64_t rankOffset = outputRank - cType.getRank();
      for (auto [i, dim] : llvm::enumerate(cType.getShape()))
        cExprs.push_back(dim == 1 ? rewriter.getAffineConstantExpr(0)
                                  : rewriter.getAffineDimExpr(rankOffset + i));
      auto mapC = mlir::AffineMap::get(outputRank, 0, cExprs, op->getContext());
      auto mapO = rewriter.getMultiDimIdentityMap(outputRank);

      mlir::SmallVector<mlir::AffineMap, 2> indexingCMaps = {mapC, mapO};

      mlir::SmallVector<mlir::utils::IteratorType> iteratorCTypes(
          outputRank, mlir::utils::IteratorType::parallel);

      auto out = mlir::tensor::EmptyOp::create(
          rewriter, loc, outDatType.getShape(), outElmType);
      auto broadcastOp = mlir::linalg::GenericOp::create(
          /*op_builder*/ rewriter, /*src_location*/ loc,
          /*result_types*/ mlir::TypeRange{outDatType},
          /*input_values*/ mlir::ValueRange{opInputC},
          /*output_values*/ mlir::ValueRange{out},
          /*affine_maps*/ indexingCMaps,
          /*iter_types*/ iteratorCTypes,
          [&](/*op_builder*/ mlir::OpBuilder &nest,
              /*src_location*/ mlir::Location nloc,
              /*value_args*/ mlir::ValueRange args) {
            mlir::Value val = args[0];
            if (attr_beta != 1.0f) {
              if (outElmType.isFloat()) {
                auto beta = nest.getFloatAttr(outElmType, attr_beta);
                auto bConst = mlir::arith::ConstantOp::create(nest, nloc, beta);
                val = mlir::arith::MulFOp::create(nest, nloc, val, bConst);
              } else {
                auto beta = nest.getIntegerAttr(
                    outElmType, static_cast<int64_t>(attr_beta));
                auto bConst = mlir::arith::ConstantOp::create(nest, nloc, beta);
                val = mlir::arith::MulIOp::create(nest, nloc, val, bConst);
              }
            }
            mlir::linalg::YieldOp::create(nest, nloc, val);
          });
      outBuffer = broadcastOp->getResult(0);
    }

  } else {
    // use zeros as bias
    auto out = mlir::tensor::EmptyOp::create(
        rewriter, loc, outDatType.getShape(), outDatType.getElementType());
    auto zero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(outElmType));
    auto fill = mlir::linalg::FillOp::create(
        rewriter, loc, mlir::ValueRange{zero}, mlir::ValueRange{out});
    outBuffer = fill->getResult(0);
  }

  auto gemmOp = mlir::linalg::GenericOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*result_type*/ mlir::TypeRange{outDatType},
      /*input_values*/ mlir::ValueRange{opInputA, opInputB},
      /*output_values*/ mlir::ValueRange{outBuffer},
      /*affine_maps*/ indexingGemmMaps,
      /*iter_types*/ iteratorGemmTypes,
      [&](/*op_builder*/ mlir::OpBuilder &nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::Value val;
        mlir::Value aVal = args[0];
        mlir::Value bVal = args[1];
        mlir::Value yVal = args[2];
        if (outElmType.isFloat()) {
          val = mlir::arith::MulFOp::create(nest, nloc, aVal, bVal);
          if (attr_alpha != 1.0f) {
            auto alpha = nest.getFloatAttr(outElmType, attr_alpha);
            auto aConst = mlir::arith::ConstantOp::create(nest, nloc, alpha);
            val = mlir::arith::MulFOp::create(nest, nloc, val, aConst);
          }
          val = mlir::arith::AddFOp::create(nest, nloc, yVal, val);
        } else {
          val = mlir::arith::MulIOp::create(nest, nloc, aVal, bVal);
          if (attr_alpha != 1.0f) {
            auto alpha = nest.getIntegerAttr(outElmType,
                                             static_cast<int64_t>(attr_alpha));
            auto aConst = mlir::arith::ConstantOp::create(nest, nloc, alpha);
            val = mlir::arith::MulIOp::create(nest, nloc, val, aConst);
          }
          val = mlir::arith::AddIOp::create(nest, nloc, yVal, val);
        }
        mlir::linalg::YieldOp::create(nest, nloc, val);
      });

  gemmOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, gemmOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
