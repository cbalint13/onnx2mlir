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
 * \file include/onnx2mlir/conversion/onnx_passes.hpp
 * \brief MLIR conversion passes
 */

#ifndef INCLUDE_ONNX2MLIR_CONVERSION_ONNX_PASSES_HPP_
#define INCLUDE_ONNX2MLIR_CONVERSION_ONNX_PASSES_HPP_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>
#include <string>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_LOWERONNXTOLINALGPASS
#include "conversion/Passes.h.inc"
} // namespace mlir

namespace onnx2mlir::dialect {

/*
 *  Onnx to Linalg
 *
 */
void registerLowerONNXToLINALGPass();
std::unique_ptr<mlir::Pass> createLowerONNXToLINALGPass();

/*
 * Common utilities
 *
 */

bool opNameBeginsWith(const llvm::StringRef &OpName, const std::string &match);

bool opNameBeginsWith(const llvm::StringRef &opName,
                      const std::vector<std::string> &matches);

mlir::RankedTensorType getBroadcastShape(mlir::RankedTensorType lhsType,
                                         mlir::RankedTensorType rhsType);

mlir::Type getSignlessType(mlir::Type type);

mlir::TypedAttr changeAttrType(mlir::Attribute attr, mlir::Type dstType);

mlir::Value createArithConstant(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Type type, mlir::Attribute attr);

} // namespace onnx2mlir::dialect

#endif // INCLUDE_ONNX2MLIR_CONVERSION_ONNX_PASSES_HPP_
