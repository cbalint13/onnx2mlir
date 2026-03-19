/******************************************************************************
 *
 * ONNX2MLIR (ONNX dialect mappings for composable optimizations)
 *
 * Authors:
 *     Cristian Balint <cristian dot balint at gmail dot com>
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
 * \file src/conversion/onnx_to_linalg.hpp
 * \brief MLIR to Linalg operators conversion
 */

#ifndef SRC_CONVERSION_PASSES_ONNX_TO_LINALG_HPP_
#define SRC_CONVERSION_PASSES_ONNX_TO_LINALG_HPP_

#include <mlir/IR/PatternMatch.h>

namespace onnx2mlir::dialect {

/*
 *  Onnx to Linalg Operator conversions
 *
 */

// onnx.{Add, Sub, Mul, Div, Pow}
mlir::LogicalResult
OnnxToLinalg_ArithBinaryOps(mlir::Operation *op,
                            mlir::PatternRewriter &rewriter);

// onnx.{Abs, Acos, Acosh, Asin, Asinh, Atan, Atanh, Ceil, Cos, Cosh, Elu,
//       Erf, Exp, Floor, HardSwish, Identity, IsInf, IsNaN, Log, Neg, Not
//       Reciprocal, Relu, Round, Sigmoid, Sign, Sin, Sinh, Softplus, Softsign,
//       Sqrt, Tan, Tanh}
mlir::LogicalResult OnnxToLinalg_ArithUnaryOps(mlir::Operation *op,
                                               mlir::PatternRewriter &rewriter);

// onnx.Cast
mlir::LogicalResult OnnxToLinalg_CastOp(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter);

// onnx.{Equal, Greater, GreatherOrEqual, Less, LessOrEqual}
mlir::LogicalResult OnnxToLinalg_CompBinaryOps(mlir::Operation *op,
                                               mlir::PatternRewriter &rewriter);

// onnx.Constant
mlir::LogicalResult OnnxToLinalg_ConstantOp(mlir::Operation *op,
                                            mlir::PatternRewriter &rewriter);

// onnx.Conv
mlir::LogicalResult OnnxToLinalg_ConvOp(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter);

// onnx.Gemm
mlir::LogicalResult OnnxToLinalg_GemmOp(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter);

// onnx.Hardmax
mlir::LogicalResult OnnxToLinalg_HardmaxOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter);

// onnx.LogSoftmax
mlir::LogicalResult OnnxToLinalg_LogSoftmaxOp(mlir::Operation *op,
                                              mlir::PatternRewriter &rewriter);

// onnx.Maxpool
mlir::LogicalResult OnnxToLinalg_MaxPoolOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter);

// onnx.Softmax
mlir::LogicalResult OnnxToLinalg_SoftmaxOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter);

// onnx.Squeeze
mlir::LogicalResult OnnxToLinalg_SqueezeOp(mlir::Operation *op,
                                           mlir::PatternRewriter &rewriter);

// onnx.Transpose
mlir::LogicalResult OnnxToLinalg_TransposeOp(mlir::Operation *op,
                                             mlir::PatternRewriter &rewriter);

// onnx.Unsqueeze
mlir::LogicalResult OnnxToLinalg_UnsqueezeOp(mlir::Operation *op,
                                             mlir::PatternRewriter &rewriter);

// onnx.Where
mlir::LogicalResult OnnxToLinalg_WhereOp(mlir::Operation *op,
                                         mlir::PatternRewriter &rewriter);

/*
 * Helpers
 *
 */

mlir::Value createArithCastOp(mlir::OpBuilder *builder,
                              const mlir::Location &loc,
                              const mlir::Value &inputElement,
                              const mlir::Type &targetElementType);

} // namespace onnx2mlir::dialect

#endif // SRC_CONVERSION_PASSES_ONNX_TO_LINALG_HPP_
