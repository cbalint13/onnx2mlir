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
 * \file src/dialect/onnx/OnnxInterfaces.td
 * \brief Onnx dialect interfaces definition
 */

#ifndef ONNX2MLIR_DIALECT_ONNX_ONNXINTERFACES_TD_
#define ONNX2MLIR_DIALECT_ONNX_ONNXINTERFACES_TD_

include "mlir/IR/OpBase.td"

def OPCountInfo : OpInterface<"OPCountInfo"> {
  let description = [{
    An interface to query custom count information for an operation type.
  }];

  let cppNamespace = "::onnx2mlir::dialect::onnx";

  let methods = [
    InterfaceMethod<"Get defined operand count", "int", "getDefinedOperandCount", (ins)>,
    InterfaceMethod<"Get defined result count", "int", "getDefinedResultCount", (ins)>
  ];
}

def ShapeInferenceOpInterface : OpInterface<"ShapeInferenceOpInterface"> {
  let description = [{
    Interface to implement shape inference for ONNX operations.
  }];

  let cppNamespace = "::onnx2mlir::dialect::onnx";

  let methods = [
    InterfaceMethod<
      "Infer dynamic shapes and update result types",
      "void", "inferShapes", (ins)
    >
  ];
}

#endif // ONNX2MLIR_DIALECT_ONNX_ONNXINTERFACES_TD_
