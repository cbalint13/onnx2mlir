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
 * \file python/src/dialect/onnx_dialect_bindings.cpp
 * \brief Python bindings for the ONNX dialect
 */

#include <mlir-c/IR.h>
#include <mlir/Bindings/Python/NanobindAdaptors.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/IR/Dialect.h>

#include "onnx2mlir/dialect/onnx/Onnx.hpp"

NB_MODULE(_onnx2mlirDialectsOnnx, m) {
  m.doc() = "Python bindings for the ONNX2MLIR dialect";

  m.def(
      "register_onnx_dialect",
      [](MlirContext context) {
        mlir::MLIRContext *cppContext = unwrap(context);
        cppContext->loadDialect<onnx2mlir::dialect::onnx::OnnxDialect>();
      },
      nanobind::arg("context"),
      "Registers the ONNX dialect in the given MLIR context.");
}
