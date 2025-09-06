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
 * \brief Onnx passes bindings to python
 */

#include <mlir/Bindings/Python/PybindAdaptors.h>
#include <mlir/IR/Dialect.h>

#include "onnx2mlir/dialect/onnx/Onnx.hpp"

PYBIND11_MODULE(_onnx2mlir_onnx_dialect, m) {
  m.doc() = "ONNX dialect python bindings";

  // Register the core dialect with the module.
  m.def("register_dialect",
        [](mlir::MLIRContext *context) { registerOnnxDialect(context); });
}
