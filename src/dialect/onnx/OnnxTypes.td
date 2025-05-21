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
 * \file src/dialect/onnx/OnnxTypes.td
 * \brief Onnx dialect types definition
 */

#ifndef ONNX2MLIR_FRONTEND_ONNXTYPES_TD_
#define ONNX2MLIR_FRONTEND_ONNXTYPES_TD_

include "mlir/IR/PatternBase.td"

class Onnx_Type<string name, string typeMnemonic,
                string baseCppClass = "::mlir::Type">
    : TypeDef<Onnx_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}

include "OnnxTypes.td.inc"

def StringType : Type<CPred<"mlir::isa<OnnxStringType>($_self)">, "string type">;

def IsSeqTypePred : CPred<"mlir::isa<SeqType>($_self)">;

class SeqOf<list<Type> allowedTypes> :
  ContainerType<AnyTypeOf<allowedTypes>, IsSeqTypePred,
                "mlir::cast<SeqType>($_self).getElementType()", "SeqType">;

def IsOptTypePred : CPred<"mlir::isa<OptType>($_self)">;

class OptOf<Type type> :
  ContainerType<type, IsOptTypePred,
                "mlir::cast<OptType>($_self).getElementType()", "OptType">;

def ONNXConstantOpFromDenseAttr: NativeCodeCall<
  "onnx_mlir::OnnxBuilder($_builder, $_loc).constant($0)">;

#endif // ONNX2MLIR_FRONTEND_ONNXTYPES_TD_
