/******************************************************************+************
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

#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_GemmOp(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  // Parse Operands
  mlir::Value A = op->getOperand(0);
  mlir::Value B = op->getOperand(1);
  mlir::Value C = op->getNumOperands() > 2 ? op->getOperand(2) : mlir::Value();

  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(A.getType());
  auto bType = mlir::dyn_cast<mlir::RankedTensorType>(B.getType());
  auto resType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());

  if (!aType || !bType || !resType) {
    return mlir::emitError(loc, opName + " operand must be ranked tensor type");
  }

  mlir::Type elementType = resType.getElementType();
  bool isFloat = mlir::isa<mlir::FloatType>(elementType);

  // Parse Attributes
  float alpha = 1.0f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
    alpha = attr.getValueAsDouble();

  float beta = 1.0f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("beta"))
    beta = attr.getValueAsDouble();

  int64_t transA = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("transA"))
    transA = attr.getInt();

  int64_t transB = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("transB"))
    transB = attr.getInt();

  // Initialize Output Buffer (handles Beta * C)
  mlir::Value outBuff;
  bool hasC = C && !mlir::isa<mlir::NoneType>(C.getType());

  if (hasC) {
    auto cType = mlir::dyn_cast<mlir::RankedTensorType>(C.getType());
    bool sameShape = (cType.getShape() == resType.getShape());
    bool betaIsOne = std::abs(beta - 1.0f) < 1e-6;
    // Use bias term
    if (sameShape && betaIsOne) {
      outBuff = C;
    } else {
      // Use bias and beta terms
      auto outTBuff = mlir::tensor::EmptyOp::create(
          rewriter, loc, resType.getShape(), elementType);

      // Define indexing for C (Broadcasting logic)
      mlir::SmallVector<mlir::AffineMap> cMaps;
      mlir::SmallVector<mlir::AffineExpr> cExprs;
      for (unsigned i = 0; i < cType.getRank(); ++i) {
        int64_t resDimIdx = resType.getRank() - cType.getRank() + i;
        if (cType.getShape()[i] == 1)
          cExprs.push_back(rewriter.getAffineConstantExpr(0));
        else
          cExprs.push_back(rewriter.getAffineDimExpr(resDimIdx));
      }
      cMaps.push_back(
          mlir::AffineMap::get(resType.getRank(), 0, cExprs, op->getContext()));
      cMaps.push_back(rewriter.getMultiDimIdentityMap(resType.getRank()));

      mlir::SmallVector<mlir::utils::IteratorType> cIters(
          resType.getRank(), mlir::utils::IteratorType::parallel);

      auto broadcastOp = mlir::linalg::GenericOp::create(
          rewriter, loc, resType, mlir::ValueRange{C},
          mlir::ValueRange{outTBuff}, cMaps, cIters,
          [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
            mlir::Value val = args[0];
            if (!betaIsOne) {
              if (isFloat) {
                mlir::Value bConst = mlir::arith::ConstantOp::create(
                    nest, l, nest.getFloatAttr(elementType, beta));
                val = mlir::arith::MulFOp::create(nest, l, val, bConst);
              } else {
                mlir::Value bConst = mlir::arith::ConstantOp::create(
                    nest, l,
                    nest.getIntegerAttr(elementType,
                                        static_cast<int64_t>(beta)));
                val = mlir::arith::MulIOp::create(nest, l, val, bConst);
              }
            }
            mlir::linalg::YieldOp::create(nest, l, val);
          });
      outBuff = broadcastOp->getResult(0);
    }
  } else {
    // Use zeros as bias term
    auto outTBuff = mlir::tensor::EmptyOp::create(
        rewriter, loc, resType.getShape(), elementType);
    auto zeroAttr = rewriter.getZeroAttr(elementType);
    mlir::Value constantZero =
        mlir::arith::ConstantOp::create(rewriter, loc, zeroAttr);
    outBuff = mlir::linalg::FillOp::create(
                  rewriter, loc, mlir::ValueRange{constantZero},
                  mlir::ValueRange{outTBuff.getResult()})
                  ->getResult(0);
  }

  // Matrix multiplication via Linalg generic
  // Indices: m (row), n (col), k (reduction)
  mlir::AffineExpr m, n, k;
  mlir::bindDims(op->getContext(), m, n, k);
  // Maps for A and B based on transposition
  mlir::AffineMap mapA =
      transA ? mlir::AffineMap::get(3, 0, {k, m}, op->getContext())
             : mlir::AffineMap::get(3, 0, {m, k}, op->getContext());
  mlir::AffineMap mapB =
      transB ? mlir::AffineMap::get(3, 0, {n, k}, op->getContext())
             : mlir::AffineMap::get(3, 0, {k, n}, op->getContext());
  mlir::AffineMap mapY = mlir::AffineMap::get(3, 0, {m, n}, op->getContext());

  mlir::SmallVector<mlir::AffineMap> gemmMaps = {mapA, mapB, mapY};
  mlir::SmallVector<mlir::utils::IteratorType> gemmIters = {
      mlir::utils::IteratorType::parallel, // m
      mlir::utils::IteratorType::parallel, // n
      mlir::utils::IteratorType::reduction // k
  };

  auto gemmOp = mlir::linalg::GenericOp::create(
      rewriter, loc, resType, mlir::ValueRange{A, B}, mlir::ValueRange{outBuff},
      gemmMaps, gemmIters,
      [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
        mlir::Value aVal = args[0];
        mlir::Value bVal = args[1];
        mlir::Value yVal = args[2];

        mlir::Value product;
        if (isFloat) {
          product = mlir::arith::MulFOp::create(nest, l, aVal, bVal);
          if (std::abs(alpha - 1.0f) > 1e-6) {
            mlir::Value aConst = mlir::arith::ConstantOp::create(
                nest, l, nest.getFloatAttr(elementType, alpha));
            product = mlir::arith::MulFOp::create(nest, l, product, aConst);
          }
        } else {
          product = mlir::arith::MulIOp::create(nest, l, aVal, bVal);
          if (std::abs(alpha - 1.0f) > 1e-6) {
            mlir::Value aConst = mlir::arith::ConstantOp::create(
                nest, l,
                nest.getIntegerAttr(elementType, static_cast<int64_t>(alpha)));
            product = mlir::arith::MulIOp::create(nest, l, product, aConst);
          }
        }

        mlir::Value res;
        if (isFloat) {
          res = mlir::arith::AddFOp::create(nest, l, yVal, product);
        } else {
          res = mlir::arith::AddIOp::create(nest, l, yVal, product);
        }
        mlir::linalg::YieldOp::create(nest, l, res);
      });

  gemmOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, gemmOp->getResult(0));
  return mlir::success();
}

} // namespace onnx2mlir::dialect
