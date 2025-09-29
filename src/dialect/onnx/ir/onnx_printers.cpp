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
 * \file src/dialect/onnx/ir/onnx_printers.cpp
 * \brief Onnx dialect assembly printer implementation
 */

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/DialectImplementation.h>

namespace onnx2mlir::dialect::onnx {

void printOnnxDictAsmPrinter(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                             mlir::DictionaryAttr dict,
                             mlir::DenseSet<mlir::StringRef> orderedAttrs = {},
                             const bool masked = false) {
  printer << "{";
  bool isFirst = true;
  for (const auto &attrName : orderedAttrs) {
    auto attrValue = dict.get(attrName);
    if (attrValue) {
      if (!isFirst) {
        printer << ", ";
      }
      printer << attrName;
      printer << " = ";
      printer.printAttribute(attrValue);
      isFirst = false;
    }
  }
  if (!masked) {
    for (const auto &namedAttr : dict) {
      if (orderedAttrs.count(namedAttr.getName()) == 0) {
        if (!isFirst) {
          printer << ", ";
        }
        printer << namedAttr.getName().str() << " = ";
        printer.printAttribute(namedAttr.getValue());
        isFirst = false;
      }
    }
  }
  printer << "}";
}

mlir::ParseResult
parseOnnxDictAsmPrinter(mlir::OpAsmParser &parser,
                        mlir::NamedAttrList &attributes,
                        mlir::DenseSet<mlir::StringRef> orderedAttrs = {},
                        const bool masked = false) {
  return parser.parseOptionalAttrDict(attributes);
}

} // namespace onnx2mlir::dialect::onnx
