
// String
def Onnx_StringType : Onnx_Type<"OnnxString", "String"> {
  let summary = "ONNX StringType";
  let description = [{
    An array of characters.
  }];
}

// Seq
def Onnx_SeqType : Onnx_Type<"Seq", "Seq"> {
  let summary = "ONNX SeqType";
  let description = [{
    An list of tensors which may have different shape.
  }];

  let parameters = (ins "::mlir::Type":$ElementType, "int64_t":$Length);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType,
                                        "int64_t":$length), [{
      return Base::get(elementType.getContext(), elementType, length);
    }]>
  ];
}

// Opt
def Onnx_OptType : Onnx_Type<"Opt", "Opt"> {
  let summary = "ONNX OptType";
  let description = [{
    An optional tensor or sequence.
  }];

  let parameters = (ins "::mlir::Type":$ElementType);

  let assemblyFormat = "`<` $ElementType `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
}
