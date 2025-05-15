#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/DialectImplementation.h>
#include <llvm/ADT/TypeSwitch.h>

#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
#include "onnx2mlir/dialect/onnx/OnnxAttrs.hpp"

#define GET_ATTRDEF_CLASSES
#include "onnx2mlir/dialect/onnx/OnnxAttrs.cpp.inc"

namespace onnx2mlir::dialect::onnx {

/// Parse an attribute registered to this dialect.
mlir::Attribute OnnxDialect::parseAttribute(
    mlir::DialectAsmParser &parser, mlir::Type type) const {

  printf("\n\nPARSEEER\n\n");
  exit(-1);

  // generatedAttributeParser is generated in ONNXAttributes.cpp.inc
  mlir::StringRef attrTag;
  if (mlir::Attribute attr;
      generatedAttributeParser(parser, &attrTag, type, attr).has_value())
    return attr;
/*
  if (attrTag == DisposableElementsAttr::getMnemonic()) {
    auto shapedTy = mlir::cast<mlir::ShapedType>(type);
    if (auto membuf = mlir::DisposableElementsAttr::parse(parser, shapedTy))
      return OnnxElementsAttrBuilder(type.getContext())
          .fromMemoryBuffer(shapedTy, std::move(membuf));
    else
      return {};
  }
*/
  parser.emitError(parser.getCurrentLocation())
      << "unknown attribute `" << attrTag << "` in dialect `ONNX`";
  return {};
}

/// Print an attribute registered to this dialect.
void OnnxDialect::printAttribute(
    mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const {

  printf("\n\nPARINTED\n\n");
  exit(-1);

  // generatedAttributePrinter is generated in ONNXAttributes.cpp.inc
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
//  if (auto elements = mlir::dyn_cast<DisposableElementsAttr>(attr))
//    elements.printWithoutType(printer);

}


mlir::Attribute OnnxTensorEncodingAttr::parse(mlir::AsmParser &parser, mlir::Type type) {

  printf("\n\nPARSEEER\n\n");
  exit(-1);

  if (failed(parser.parseLess()))
    return {};
  // Parse the data as a dictionary.
  mlir::DictionaryAttr dict;
  if (llvm::failed(parser.parseAttribute(dict)))
    return {};
  if (llvm::failed(parser.parseGreater()))
    return {};

  OnnxTensorEncodingAttr::DataLayout dataLayout =
      OnnxTensorEncodingAttr::DataLayout::STANDARD;
  int64_t xFactor = 0;
  int64_t yFactor = 0;

  // Process the data from the parsed dictionary value into struct-like data.
  for (const mlir::NamedAttribute &attr : dict) {
    if (attr.getName() == "dataLayout") {
      mlir::StringAttr layoutAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue());
      if (!layoutAttr) {
        parser.emitError(
            parser.getNameLoc(), "expected a string value for data layout");
        return {};
      }
//      if (!onnx_mlir::convertStringToONNXCustomTensorDataLayout(
//              layoutAttr, dataLayout, xFactor, yFactor)) {
//        parser.emitError(
//            parser.getNameLoc(), "unexpected data layout attribute value: ")
//            << layoutAttr.getValue();
//        return {};
//      }
    } else { // Attribute different than "dataLayout".
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().str();
      return {};
    }
  }
  // Construct struct-like storage for attribute.
  return parser.getChecked<OnnxTensorEncodingAttr>(
      parser.getContext(), dataLayout, xFactor, yFactor);

  return {};

}

void OnnxTensorEncodingAttr::print(mlir::AsmPrinter &printer) const {
  // Print the struct-like storage in dictionary fashion.
  printer << "<{dataLayout = ";
//  mlir::StringRef layoutStr = onnx_mlir::convertOnnxTensorDataLayoutToString(
//      getDataLayout(), getXFactor(), getYFactor());
//  printer << "\"" << layoutStr.str() << "\"";
  printer << "}>";
}


} // namespace onnx2mlir::dialect::onnx
