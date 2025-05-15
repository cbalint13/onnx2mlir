#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>
#include <llvm/ADT/TypeSwitch.h>

#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
#include "onnx2mlir/dialect/onnx/OnnxTypes.hpp"

#define GET_TYPEDEF_CLASSES
#include "onnx2mlir/dialect/onnx/OnnxTypes.cpp.inc"

namespace onnx2mlir::dialect::onnx {

void OnnxDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "onnx2mlir/dialect/onnx/OnnxTypes.cpp.inc"
      >();
}

mlir::Type SeqType::parse(mlir::AsmParser &parser) {
  Type elementType;
  if (parser.parseLess() || parser.parseType(elementType) ||
      parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation())
        << "failed to parse !onnx.Seq type";
    return Type();
  }

  return get(elementType, mlir::ShapedType::kDynamic);
}

void SeqType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getElementType() << ">";
}

} // namespace onnx2mlir::dialect::onnx
