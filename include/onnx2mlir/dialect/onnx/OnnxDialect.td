/******************************************************************************
 *
 * Copyright (c) 2021,2025
 *
 *     Cristian Balint <cristian dot balint at gmail dot com>
 *
 * ONNX2MLIR (ONNX dialect mappings for composable optimizations)
 *
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
 * \file OnnxDialect.td
 *
 */

#ifndef ONNX2MLIR_FRONTEND_ONNXDIALECT_TD_
#define ONNX2MLIR_FRONTEND_ONNXDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Onnx_Dialect : Dialect {
  let name = "onnx";
  let summary = "A dialect for ONNX representation";
  let cppNamespace = "::onnx2mlir::dialect::onnx";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
  let extraClassDeclaration = [{
    private:
      void registerOps();
      void registerTypes();
    public:
      ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                     ::mlir::Type type) const override;
      void printAttribute(::mlir::Attribute attr,
                          ::mlir::DialectAsmPrinter &printer) const override;
    }];
}

#endif // ONNX2MLIR_FRONTEND_ONNXDIALECT_TD_
