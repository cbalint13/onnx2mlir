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
 * \file src/conversion/Passes.td
 * \brief Dialect conversion passes
 */

#ifndef ONNX2MLIR_CONVERSION_PASSES_TD_
#define ONNX2MLIR_CONVERSION_PASSES_TD_

include "mlir/Pass/PassBase.td"

def LowerONNXToLINALGPass : Pass<"lower-onnx-to-linalg", "ModuleOp"> {
  let summary = "Lower ONNX operations to LINALG dialect operations";
  let description = [{
    This pass converts ONNX operations to their LINALG dialect equivalent.
  }];

  let constructor = "createLowerONNXToLINALGPass()";
}

#endif // ONNX2MLIR_CONVERSION_PASSES_TD_
