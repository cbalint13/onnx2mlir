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

mlir::LogicalResult
OnnxToLinalg_BatchNormalizationOp(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter,
                                  const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_BinaryOps(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                       const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_CastOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_ClipOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_CompBinaryOps(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                           const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_ConcatOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_ConstantOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                        const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_ConstantOfShapeOp(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter,
                               const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_ConvOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_FlattenOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                       const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_GatherOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_GemmOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                    const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_GlobalPoolOps(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                           const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_LayerNormalizationOp(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter,
                                  const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_MatMulOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_MaxPoolOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                       const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_PadOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                   const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_ReshapeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                       const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_ResizeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_ShapeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                     const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_SliceOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                     const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_SoftmaxOps(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                        const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_SplitOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                     const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_SqueezeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                       const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_TransposeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                         const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_UnaryOps(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                      const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_UnsqueezeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                         const mlir::TypeConverter *typeConverter);

mlir::LogicalResult
OnnxToLinalg_WhereOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                     const mlir::TypeConverter *typeConverter);

} // namespace onnx2mlir::dialect

#endif // SRC_CONVERSION_PASSES_ONNX_TO_LINALG_HPP_
