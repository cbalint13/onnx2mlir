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
 * \file src/dialect/onnx/ir/onnx_ops.cpp
 * \brief Onnx dialect operations implementation
 */

#include "onnx2mlir/dialect/onnx/OnnxAttrs.hpp"
#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
#include "onnx2mlir/dialect/onnx/OnnxInterface.hpp"
#include "onnx2mlir/dialect/onnx/OnnxOps.hpp"
#include "onnx2mlir/dialect/onnx/OnnxTypes.hpp"

#define GET_OP_CLASSES
#include "onnx2mlir/dialect/onnx/Onnx.cpp.inc"

namespace onnx2mlir::dialect::onnx {

void OnnxDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "onnx2mlir/dialect/onnx/Onnx.cpp.inc"
      >();
}

} // namespace onnx2mlir::dialect::onnx
