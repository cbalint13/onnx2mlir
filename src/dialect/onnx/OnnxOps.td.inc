/********************************************************
 *   ONNX version [1.19.0]                              *
 *   Generated at [2025-05-20 16:19:11.132233]          *
 ********************************************************/

/// Adagrad [v1]
def Onnx_AdagradOp : Onnx_Op<"Adagrad", [Pure, OPCountInfo]> {
  let summary = "ONNX Adagrad operation [v1]";
  let description = [{
      Compute one iteration of ADAGRAD, a stochastic gradient based optimization
  algorithm.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       TensorOf<[I64]>:$T,
                       Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>:$inputs,
                       DefaultValuedAttr<F32Attr, "0">:$decay_factor,
                       DefaultValuedAttr<F32Attr, "1e-06">:$epsilon,
                       DefaultValuedAttr<F32Attr, "0">:$norm_coefficient);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Momentum [v1]
def Onnx_MomentumOp : Onnx_Op<"Momentum", [Pure, OPCountInfo]> {
  let summary = "ONNX Momentum operation [v1]";
  let description = [{
      Compute one iteration of stochastic gradient update with momentum.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       TensorOf<[I64]>:$T,
                       Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>:$inputs,
                       F32Attr:$alpha,
                       F32Attr:$beta,
                       StrAttr:$mode,
                       F32Attr:$norm_coefficient);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RotaryEmbedding [v23]
def Onnx_RotaryEmbeddingOp : Onnx_Op<"RotaryEmbedding", [Pure, OPCountInfo]> {
  let summary = "ONNX RotaryEmbedding operation [v23]";
  let description = [{
      RotaryEmbedding is the implementation of rotary positional embeddings (RoPE) based on the paper https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$X,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$cos_cache,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$sin_cache,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$position_ids,
                       DefaultValuedAttr<I64Attr, "0">:$interleaved,
                       OptionalAttr<I64Attr>:$num_heads,
                       DefaultValuedAttr<I64Attr, "0">:$rotary_embedding_dim);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// StringSplit [v20]
def Onnx_StringSplitOp : Onnx_Op<"StringSplit", [Pure, OPCountInfo]> {
  let summary = "ONNX StringSplit operation [v20]";
  let description = [{
      StringSplit splits a string tensor's elements into substrings based on a delimiter attribute and a maxsplit attribute.
  }];

  let arguments = (ins TensorOf<[StringType]>:$X,
                       OptionalAttr<StrAttr>:$delimiter,
                       OptionalAttr<I64Attr>:$maxsplit);
  let results = (outs TensorOf<[StringType]>:$Y,
                      TensorOf<[I64]>:$Z);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// Adam [v1]
def Onnx_AdamOp : Onnx_Op<"Adam", [Pure, OPCountInfo]> {
  let summary = "ONNX Adam operation [v1]";
  let description = [{
      Compute one iteration of Adam, a stochastic gradient based optimization
  algorithm.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       TensorOf<[I64]>:$T,
                       Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>:$inputs,
                       DefaultValuedAttr<F32Attr, "0.9">:$alpha,
                       DefaultValuedAttr<F32Attr, "0.999">:$beta,
                       DefaultValuedAttr<F32Attr, "1e-06">:$epsilon,
                       DefaultValuedAttr<F32Attr, "0">:$norm_coefficient,
                       DefaultValuedAttr<F32Attr, "0">:$norm_coefficient_post);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gelu [v20]
def Onnx_GeluOp : Onnx_Op<"Gelu", [Pure, OPCountInfo]> {
  let summary = "ONNX Gelu operation [v20]";
  let description = [{
      Gelu takes one input data (Tensor<T>) and produces one
  output data (Tensor<T>) where the gaussian error linear units function,
  $y = 0.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "none">:$approximate);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// AffineGrid [v20]
def Onnx_AffineGridOp : Onnx_Op<"AffineGrid", [Pure, OPCountInfo]> {
  let summary = "ONNX AffineGrid operation [v20]";
  let description = [{
      Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta
  (https://pytorch.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$theta,
                       TensorOf<[I64]>:$size,
                       DefaultValuedAttr<I64Attr, "0">:$align_corners);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$grid);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RMSNormalization [v23]
def Onnx_RMSNormalizationOp : Onnx_Op<"RMSNormalization", [Pure, OPCountInfo]> {
  let summary = "ONNX RMSNormalization operation [v23]";
  let description = [{
      This is RMS normalization defined in ONNX as function as described in the paper https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$scale,
                       DefaultValuedAttr<I64Attr, "-1">:$axis,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       DefaultValuedAttr<I64Attr, "1">:$stash_type);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// BitwiseOr [v18]
def Onnx_BitwiseOrOp : Onnx_Op<"BitwiseOr", [Pure, OPCountInfo]> {
  let summary = "ONNX BitwiseOr operation [v18]";
  let description = [{
      Returns the tensor resulting from performing the bitwise `or` operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// BitwiseAnd [v18]
def Onnx_BitwiseAndOp : Onnx_Op<"BitwiseAnd", [Pure, OPCountInfo]> {
  let summary = "ONNX BitwiseAnd operation [v18]";
  let description = [{
      Returns the tensor resulting from performing the bitwise `and` operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mish [v18]
def Onnx_Mish_V18Op : Onnx_Op<"Mish_V18", [Pure, OPCountInfo]> {
  let summary = "ONNX Mish_V18 operation [v18]";
  let description = [{
      Mish: A Self Regularized Non-Monotonic Neural Activation Function.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mish [v22]
def Onnx_MishOp : Onnx_Op<"Mish", [Pure, OPCountInfo]> {
  let summary = "ONNX Mish operation [v22]";
  let description = [{
      Mish: A Self Regularized Non-Monotonic Neural Activation Function.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MelWeightMatrix [v17]
def Onnx_MelWeightMatrixOp : Onnx_Op<"MelWeightMatrix", [Pure, OPCountInfo]> {
  let summary = "ONNX MelWeightMatrix operation [v17]";
  let description = [{
      Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$num_mel_bins,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$dft_length,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$sample_rate,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$lower_edge_hertz,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$upper_edge_hertz,
                       DefaultValuedAttr<I64Attr, "1">:$output_datatype);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// BlackmanWindow [v17]
def Onnx_BlackmanWindowOp : Onnx_Op<"BlackmanWindow", [Pure, OPCountInfo]> {
  let summary = "ONNX BlackmanWindow operation [v17]";
  let description = [{
      Generates a Blackman window as described in the paper https://ieeexplore.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$size,
                       DefaultValuedAttr<I64Attr, "1">:$output_datatype,
                       DefaultValuedAttr<I64Attr, "1">:$periodic);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gradient [v1]
def Onnx_GradientOp : Onnx_Op<"Gradient", [Pure, OPCountInfo]> {
  let summary = "ONNX Gradient operation [v1]";
  let description = [{
      Gradient operator computes the partial derivatives of a specific tensor w.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$Inputs,
                       StrArrayAttr:$xs,
                       StrAttr:$y,
                       OptionalAttr<StrArrayAttr>:$zs);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$Outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// HammingWindow [v17]
def Onnx_HammingWindowOp : Onnx_Op<"HammingWindow", [Pure, OPCountInfo]> {
  let summary = "ONNX HammingWindow operation [v17]";
  let description = [{
      Generates a Hamming window as described in the paper https://ieeexplore.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$size,
                       DefaultValuedAttr<I64Attr, "1">:$output_datatype,
                       DefaultValuedAttr<I64Attr, "1">:$periodic);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// HannWindow [v17]
def Onnx_HannWindowOp : Onnx_Op<"HannWindow", [Pure, OPCountInfo]> {
  let summary = "ONNX HannWindow operation [v17]";
  let description = [{
      Generates a Hann window as described in the paper https://ieeexplore.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$size,
                       DefaultValuedAttr<I64Attr, "1">:$output_datatype,
                       DefaultValuedAttr<I64Attr, "1">:$periodic);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SequenceMap [v17]
def Onnx_SequenceMapOp : Onnx_Op<"SequenceMap", [Pure, OPCountInfo]> {
  let summary = "ONNX SequenceMap operation [v17]";
  let description = [{
      Applies a sub-graph to each sample in the input sequence(s).
  }];

  let arguments = (ins AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$input_sequence,
                       Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>>:$additional_inputs);
  let results = (outs Variadic<AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>>:$out_sequence);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LayerNormalization [v17]
def Onnx_LayerNormalizationOp : Onnx_Op<"LayerNormalization", [Pure, OPCountInfo]> {
  let summary = "ONNX LayerNormalization operation [v17]";
  let description = [{
      This is layer normalization defined in ONNX as function.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$B,
                       DefaultValuedAttr<I64Attr, "-1">:$axis,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       DefaultValuedAttr<I64Attr, "1">:$stash_type);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y,
                      AnyTypeOf<[TensorOf<[F32]>, TensorOf<[BF16]>, NoneType]>:$Mean,
                      AnyTypeOf<[TensorOf<[F32]>, TensorOf<[BF16]>, NoneType]>:$InvStdDev);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 3;
    }
  }];
}

/// GridSample [v16]
def Onnx_GridSample_V16Op : Onnx_Op<"GridSample_V16", [Pure, OPCountInfo]> {
  let summary = "ONNX GridSample_V16 operation [v16]";
  let description = [{
      Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from `grid`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$grid,
                       DefaultValuedAttr<I64Attr, "0">:$align_corners,
                       DefaultValuedStrAttr<StrAttr, "bilinear">:$mode,
                       DefaultValuedStrAttr<StrAttr, "zeros">:$padding_mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GridSample [v20]
def Onnx_GridSample_V20Op : Onnx_Op<"GridSample_V20", [Pure, OPCountInfo]> {
  let summary = "ONNX GridSample_V20 operation [v20]";
  let description = [{
      Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$grid,
                       DefaultValuedAttr<I64Attr, "0">:$align_corners,
                       DefaultValuedStrAttr<StrAttr, "linear">:$mode,
                       DefaultValuedStrAttr<StrAttr, "zeros">:$padding_mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GridSample [v22]
def Onnx_GridSampleOp : Onnx_Op<"GridSample", [Pure, OPCountInfo]> {
  let summary = "ONNX GridSample operation [v22]";
  let description = [{
      Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$grid,
                       DefaultValuedAttr<I64Attr, "0">:$align_corners,
                       DefaultValuedStrAttr<StrAttr, "linear">:$mode,
                       DefaultValuedStrAttr<StrAttr, "zeros">:$padding_mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// CastLike [v15]
def Onnx_CastLike_V15Op : Onnx_Op<"CastLike_V15", [Pure, OPCountInfo]> {
  let summary = "ONNX CastLike_V15 operation [v15]";
  let description = [{
      The operator casts the elements of a given input tensor (the first input) to
  the same data type as the elements of the second input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>:$input,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>:$target_type);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// CastLike [v19]
def Onnx_CastLike_V19Op : Onnx_Op<"CastLike_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX CastLike_V19 operation [v19]";
  let description = [{
      The operator casts the elements of a given input tensor (the first input) to
  the same data type as the elements of the second input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$input,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$target_type,
                       DefaultValuedAttr<I64Attr, "1">:$saturate);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// CastLike [v21]
def Onnx_CastLike_V21Op : Onnx_Op<"CastLike_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX CastLike_V21 operation [v21]";
  let description = [{
      The operator casts the elements of a given input tensor (the first input) to
  the same data type as the elements of the second input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$input,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$target_type,
                       DefaultValuedAttr<I64Attr, "1">:$saturate);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// CastLike [v23]
def Onnx_CastLikeOp : Onnx_Op<"CastLike", [Pure, OPCountInfo]> {
  let summary = "ONNX CastLike operation [v23]";
  let description = [{
      The operator casts the elements of a given input tensor (the first input) to
  the same data type as the elements of the second input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$input,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$target_type,
                       DefaultValuedAttr<I64Attr, "1">:$saturate);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// OptionalHasElement [v15]
def Onnx_OptionalHasElement_V15Op : Onnx_Op<"OptionalHasElement_V15", [Pure, OPCountInfo]> {
  let summary = "ONNX OptionalHasElement_V15 operation [v15]";
  let description = [{
      Returns true if the optional-type input contains an element.
  }];

  let arguments = (ins AnyTypeOf<[OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$input);
  let results = (outs TensorOf<[I1]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// OptionalHasElement [v18]
def Onnx_OptionalHasElementOp : Onnx_Op<"OptionalHasElement", [Pure, OPCountInfo]> {
  let summary = "ONNX OptionalHasElement operation [v18]";
  let description = [{
      Returns true if (1) the input is an optional-type and contains an element,
  or, (2) the input is a tensor or sequence type.
  }];

  let arguments = (ins AnyTypeOf<[OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, NoneType]>:$input);
  let results = (outs TensorOf<[I1]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Optional [v15]
def Onnx_OptionalOp : Onnx_Op<"Optional", [Pure, OPCountInfo]> {
  let summary = "ONNX Optional operation [v15]";
  let description = [{
      Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
  or a non-empty value containing the input element.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, NoneType]>:$input,
                       OptionalAttr<TypeAttr>:$type);
  let results = (outs AnyTypeOf<[OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SoftmaxCrossEntropyLoss [v12]
def Onnx_SoftmaxCrossEntropyLoss_V12Op : Onnx_Op<"SoftmaxCrossEntropyLoss_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX SoftmaxCrossEntropyLoss_V12 operation [v12]";
  let description = [{
      Loss function that measures the softmax cross entropy
  between 'scores' and 'labels'.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scores,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$labels,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$weights,
                       OptionalAttr<I64Attr>:$ignore_index,
                       DefaultValuedStrAttr<StrAttr, "mean">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$log_prob);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// SoftmaxCrossEntropyLoss [v13]
def Onnx_SoftmaxCrossEntropyLossOp : Onnx_Op<"SoftmaxCrossEntropyLoss", [Pure, OPCountInfo]> {
  let summary = "ONNX SoftmaxCrossEntropyLoss operation [v13]";
  let description = [{
      Loss function that measures the softmax cross entropy
  between 'scores' and 'labels'.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$scores,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$labels,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$weights,
                       OptionalAttr<I64Attr>:$ignore_index,
                       DefaultValuedStrAttr<StrAttr, "mean">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$log_prob);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// GreaterOrEqual [v12]
def Onnx_GreaterOrEqual_V12Op : Onnx_Op<"GreaterOrEqual_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX GreaterOrEqual_V12 operation [v12]";
  let description = [{
      Returns the tensor resulted from performing the `greater_equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GreaterOrEqual [v16]
def Onnx_GreaterOrEqualOp : Onnx_Op<"GreaterOrEqual", [Pure, OPCountInfo]> {
  let summary = "ONNX GreaterOrEqual operation [v16]";
  let description = [{
      Returns the tensor resulted from performing the `greater_equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LessOrEqual [v12]
def Onnx_LessOrEqual_V12Op : Onnx_Op<"LessOrEqual_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX LessOrEqual_V12 operation [v12]";
  let description = [{
      Returns the tensor resulted from performing the `less_equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LessOrEqual [v16]
def Onnx_LessOrEqualOp : Onnx_Op<"LessOrEqual", [Pure, OPCountInfo]> {
  let summary = "ONNX LessOrEqual operation [v16]";
  let description = [{
      Returns the tensor resulted from performing the `less_equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Celu [v12]
def Onnx_CeluOp : Onnx_Op<"Celu", [Pure, OPCountInfo]> {
  let summary = "ONNX Celu operation [v12]";
  let description = [{
      Continuously Differentiable Exponential Linear Units:
  Perform the linear unit element-wise on the input tensor X
  using formula:
  ```
  max(0,x) + min(0,alpha*(exp(x/alpha)-1))
  ```.
  }];

  let arguments = (ins TensorOf<[F32]>:$X,
                       DefaultValuedAttr<F32Attr, "1">:$alpha);
  let results = (outs TensorOf<[F32]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SequenceLength [v11]
def Onnx_SequenceLengthOp : Onnx_Op<"SequenceLength", [Pure, OPCountInfo]> {
  let summary = "ONNX SequenceLength operation [v11]";
  let description = [{
      Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
  }];

  let arguments = (ins AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$input_sequence);
  let results = (outs TensorOf<[I64]>:$length);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// OptionalGetElement [v15]
def Onnx_OptionalGetElement_V15Op : Onnx_Op<"OptionalGetElement_V15", [Pure, OPCountInfo]> {
  let summary = "ONNX OptionalGetElement_V15 operation [v15]";
  let description = [{
      Outputs the element in the optional-type input.
  }];

  let arguments = (ins AnyTypeOf<[OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// OptionalGetElement [v18]
def Onnx_OptionalGetElementOp : Onnx_Op<"OptionalGetElement", [Pure, OPCountInfo]> {
  let summary = "ONNX OptionalGetElement operation [v18]";
  let description = [{
      If the input is a tensor or sequence type, it returns the input.
  }];

  let arguments = (ins AnyTypeOf<[OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SequenceAt [v11]
def Onnx_SequenceAtOp : Onnx_Op<"SequenceAt", [Pure, OPCountInfo]> {
  let summary = "ONNX SequenceAt operation [v11]";
  let description = [{
      Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
  }];

  let arguments = (ins AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$input_sequence,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$position);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$tensor);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SequenceConstruct [v11]
def Onnx_SequenceConstructOp : Onnx_Op<"SequenceConstruct", [Pure, OPCountInfo]> {
  let summary = "ONNX SequenceConstruct operation [v11]";
  let description = [{
      Construct a tensor sequence containing 'inputs' tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$inputs);
  let results = (outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$output_sequence);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GatherND [v11]
def Onnx_GatherND_V11Op : Onnx_Op<"GatherND_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX GatherND_V11 operation [v11]";
  let description = [{
      Given `data` tensor of rank `r` >= 1, and `indices` tensor of rank `q` >= 1, this operator gathers
  slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$indices);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GatherND [v12]
def Onnx_GatherND_V12Op : Onnx_Op<"GatherND_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX GatherND_V12 operation [v12]";
  let description = [{
      Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
  slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$indices,
                       DefaultValuedAttr<I64Attr, "0">:$batch_dims);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GatherND [v13]
def Onnx_GatherNDOp : Onnx_Op<"GatherND", [Pure, OPCountInfo]> {
  let summary = "ONNX GatherND operation [v13]";
  let description = [{
      Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
  slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$indices,
                       DefaultValuedAttr<I64Attr, "0">:$batch_dims);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ScatterND [v11]
def Onnx_ScatterND_V11Op : Onnx_Op<"ScatterND_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ScatterND_V11 operation [v11]";
  let description = [{
      ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
  and `updates` tensor of rank q + r - indices.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ScatterND [v13]
def Onnx_ScatterND_V13Op : Onnx_Op<"ScatterND_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ScatterND_V13 operation [v13]";
  let description = [{
      ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
  and `updates` tensor of rank q + r - indices.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ScatterND [v16]
def Onnx_ScatterND_V16Op : Onnx_Op<"ScatterND_V16", [Pure, OPCountInfo]> {
  let summary = "ONNX ScatterND_V16 operation [v16]";
  let description = [{
      ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
  and `updates` tensor of rank q + r - indices.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates,
                       DefaultValuedStrAttr<StrAttr, "none">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ScatterND [v18]
def Onnx_ScatterNDOp : Onnx_Op<"ScatterND", [Pure, OPCountInfo]> {
  let summary = "ONNX ScatterND operation [v18]";
  let description = [{
      ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
  and `updates` tensor of rank q + r - indices.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates,
                       DefaultValuedStrAttr<StrAttr, "none">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GatherElements [v11]
def Onnx_GatherElements_V11Op : Onnx_Op<"GatherElements_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX GatherElements_V11 operation [v11]";
  let description = [{
      GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
  and an optional attribute `axis` that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GatherElements [v13]
def Onnx_GatherElementsOp : Onnx_Op<"GatherElements", [Pure, OPCountInfo]> {
  let summary = "ONNX GatherElements operation [v13]";
  let description = [{
      GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
  and an optional attribute `axis` that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// CumSum [v11]
def Onnx_CumSum_V11Op : Onnx_Op<"CumSum_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX CumSum_V11 operation [v11]";
  let description = [{
      Performs cumulative sum of the input elements along the given axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$x,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$exclusive,
                       DefaultValuedAttr<I64Attr, "0">:$reverse);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// CumSum [v14]
def Onnx_CumSumOp : Onnx_Op<"CumSum", [Pure, OPCountInfo]> {
  let summary = "ONNX CumSum operation [v14]";
  let description = [{
      Performs cumulative sum of the input elements along the given axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$x,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$exclusive,
                       DefaultValuedAttr<I64Attr, "0">:$reverse);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RoiAlign [v10]
def Onnx_RoiAlign_V10Op : Onnx_Op<"RoiAlign_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX RoiAlign_V10 operation [v10]";
  let description = [{
      Region of Interest (RoI) align operation described in the
  [Mask R-CNN paper](https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$rois,
                       TensorOf<[I64]>:$batch_indices,
                       DefaultValuedStrAttr<StrAttr, "avg">:$mode,
                       DefaultValuedAttr<I64Attr, "1">:$output_height,
                       DefaultValuedAttr<I64Attr, "1">:$output_width,
                       DefaultValuedAttr<I64Attr, "0">:$sampling_ratio,
                       DefaultValuedAttr<F32Attr, "1">:$spatial_scale);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RoiAlign [v16]
def Onnx_RoiAlign_V16Op : Onnx_Op<"RoiAlign_V16", [Pure, OPCountInfo]> {
  let summary = "ONNX RoiAlign_V16 operation [v16]";
  let description = [{
      Region of Interest (RoI) align operation described in the
  [Mask R-CNN paper](https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$rois,
                       TensorOf<[I64]>:$batch_indices,
                       DefaultValuedStrAttr<StrAttr, "half_pixel">:$coordinate_transformation_mode,
                       DefaultValuedStrAttr<StrAttr, "avg">:$mode,
                       DefaultValuedAttr<I64Attr, "1">:$output_height,
                       DefaultValuedAttr<I64Attr, "1">:$output_width,
                       DefaultValuedAttr<I64Attr, "0">:$sampling_ratio,
                       DefaultValuedAttr<F32Attr, "1">:$spatial_scale);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RoiAlign [v22]
def Onnx_RoiAlignOp : Onnx_Op<"RoiAlign", [Pure, OPCountInfo]> {
  let summary = "ONNX RoiAlign operation [v22]";
  let description = [{
      Region of Interest (RoI) align operation described in the
  [Mask R-CNN paper](https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$rois,
                       TensorOf<[I64]>:$batch_indices,
                       DefaultValuedStrAttr<StrAttr, "half_pixel">:$coordinate_transformation_mode,
                       DefaultValuedStrAttr<StrAttr, "avg">:$mode,
                       DefaultValuedAttr<I64Attr, "1">:$output_height,
                       DefaultValuedAttr<I64Attr, "1">:$output_width,
                       DefaultValuedAttr<I64Attr, "0">:$sampling_ratio,
                       DefaultValuedAttr<F32Attr, "1">:$spatial_scale);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// NonMaxSuppression [v10]
def Onnx_NonMaxSuppression_V10Op : Onnx_Op<"NonMaxSuppression_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX NonMaxSuppression_V10 operation [v10]";
  let description = [{
      Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
  }];

  let arguments = (ins TensorOf<[F32]>:$boxes,
                       TensorOf<[F32]>:$scores,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$max_output_boxes_per_class,
                       AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$iou_threshold,
                       AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$score_threshold,
                       DefaultValuedAttr<I64Attr, "0">:$center_point_box);
  let results = (outs TensorOf<[I64]>:$selected_indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// NonMaxSuppression [v11]
def Onnx_NonMaxSuppressionOp : Onnx_Op<"NonMaxSuppression", [Pure, OPCountInfo]> {
  let summary = "ONNX NonMaxSuppression operation [v11]";
  let description = [{
      Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
  }];

  let arguments = (ins TensorOf<[F32]>:$boxes,
                       TensorOf<[F32]>:$scores,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$max_output_boxes_per_class,
                       AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$iou_threshold,
                       AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$score_threshold,
                       DefaultValuedAttr<I64Attr, "0">:$center_point_box);
  let results = (outs TensorOf<[I64]>:$selected_indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DequantizeLinear [v10]
def Onnx_DequantizeLinear_V10Op : Onnx_Op<"DequantizeLinear_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX DequantizeLinear_V10 operation [v10]";
  let description = [{
      The linear dequantization operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>]>:$x,
                       TensorOf<[F32]>:$x_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>, NoneType]>:$x_zero_point);
  let results = (outs TensorOf<[F32]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DequantizeLinear [v13]
def Onnx_DequantizeLinear_V13Op : Onnx_Op<"DequantizeLinear_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX DequantizeLinear_V13 operation [v13]";
  let description = [{
      The linear dequantization operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>]>:$x,
                       TensorOf<[F32]>:$x_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>, NoneType]>:$x_zero_point,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs TensorOf<[F32]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DequantizeLinear [v19]
def Onnx_DequantizeLinear_V19Op : Onnx_Op<"DequantizeLinear_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX DequantizeLinear_V19 operation [v19]";
  let description = [{
      The linear dequantization operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$x,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$x_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$x_zero_point,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DequantizeLinear [v21]
def Onnx_DequantizeLinear_V21Op : Onnx_Op<"DequantizeLinear_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX DequantizeLinear_V21 operation [v21]";
  let description = [{
      The linear dequantization operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$x,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$x_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, NoneType]>:$x_zero_point,
                       DefaultValuedAttr<I64Attr, "1">:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$block_size);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DequantizeLinear [v23]
def Onnx_DequantizeLinearOp : Onnx_Op<"DequantizeLinear", [Pure, OPCountInfo]> {
  let summary = "ONNX DequantizeLinear operation [v23]";
  let description = [{
      The linear dequantization operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$x,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$x_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, NoneType]>:$x_zero_point,
                       DefaultValuedAttr<I64Attr, "1">:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$block_size,
                       DefaultValuedAttr<I64Attr, "0">:$output_dtype);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DFT [v17]
def Onnx_DFT_V17Op : Onnx_Op<"DFT_V17", [Pure, OPCountInfo]> {
  let summary = "ONNX DFT_V17 operation [v17]";
  let description = [{
      Computes the discrete Fourier transform of input.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$dft_length,
                       DefaultValuedAttr<I64Attr, "1">:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$inverse,
                       DefaultValuedAttr<I64Attr, "0">:$onesided);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DFT [v20]
def Onnx_DFTOp : Onnx_Op<"DFT", [Pure, OPCountInfo]> {
  let summary = "ONNX DFT operation [v20]";
  let description = [{
      Computes the discrete Fourier Transform (DFT) of the input.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$dft_length,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$inverse,
                       DefaultValuedAttr<I64Attr, "0">:$onesided);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// QuantizeLinear [v10]
def Onnx_QuantizeLinear_V10Op : Onnx_Op<"QuantizeLinear_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX QuantizeLinear_V10 operation [v10]";
  let description = [{
      The linear per-tensor/layer quantization operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>]>:$x,
                       TensorOf<[F32]>:$y_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$y_zero_point);
  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// QuantizeLinear [v13]
def Onnx_QuantizeLinear_V13Op : Onnx_Op<"QuantizeLinear_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX QuantizeLinear_V13 operation [v13]";
  let description = [{
      The linear quantization operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>]>:$x,
                       TensorOf<[F32]>:$y_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$y_zero_point,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// QuantizeLinear [v19]
def Onnx_QuantizeLinear_V19Op : Onnx_Op<"QuantizeLinear_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX QuantizeLinear_V19 operation [v19]";
  let description = [{
      The linear quantization operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$x,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$y_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$y_zero_point,
                       DefaultValuedAttr<I64Attr, "1">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$saturate);
  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// QuantizeLinear [v21]
def Onnx_QuantizeLinear_V21Op : Onnx_Op<"QuantizeLinear_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX QuantizeLinear_V21 operation [v21]";
  let description = [{
      The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
  low-precision/quantized tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$x,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$y_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, NoneType]>:$y_zero_point,
                       DefaultValuedAttr<I64Attr, "1">:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$block_size,
                       DefaultValuedAttr<I64Attr, "0">:$output_dtype,
                       DefaultValuedAttr<I64Attr, "1">:$saturate);
  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// QuantizeLinear [v23]
def Onnx_QuantizeLinearOp : Onnx_Op<"QuantizeLinear", [Pure, OPCountInfo]> {
  let summary = "ONNX QuantizeLinear operation [v23]";
  let description = [{
      The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
  low-precision/quantized tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$x,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$y_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, NoneType]>:$y_zero_point,
                       DefaultValuedAttr<I64Attr, "1">:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$block_size,
                       DefaultValuedAttr<I64Attr, "0">:$output_dtype,
                       DefaultValuedAttr<I64Attr, "0">:$precision,
                       DefaultValuedAttr<I64Attr, "1">:$saturate);
  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[UI16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// QLinearConv [v10]
def Onnx_QLinearConvOp : Onnx_Op<"QLinearConv", [Pure, OPCountInfo]> {
  let summary = "ONNX QLinearConv operation [v10]";
  let description = [{
      The convolution operator consumes a quantized input tensor, its scale and zero point,
  a quantized filter, its scale and zero point, and output's scale and zero point,
  and computes the quantized output.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$x,
                       TensorOf<[F32]>:$x_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$x_zero_point,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$w,
                       TensorOf<[F32]>:$w_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$w_zero_point,
                       TensorOf<[F32]>:$y_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y_zero_point,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$B,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 9;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ConvInteger [v10]
def Onnx_ConvIntegerOp : Onnx_Op<"ConvInteger", [Pure, OPCountInfo]> {
  let summary = "ONNX ConvInteger operation [v10]";
  let description = [{
      The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
  and computes the output.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$x,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$w,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$x_zero_point,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$w_zero_point,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs TensorOf<[I32]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Det [v11]
def Onnx_Det_V11Op : Onnx_Op<"Det_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Det_V11 operation [v11]";
  let description = [{
      Det calculates determinant of a square matrix or batches of square matrices.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Det [v22]
def Onnx_DetOp : Onnx_Op<"Det", [Pure, OPCountInfo]> {
  let summary = "ONNX Det operation [v22]";
  let description = [{
      Det calculates determinant of a square matrix or batches of square matrices.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// QLinearMatMul [v10]
def Onnx_QLinearMatMul_V10Op : Onnx_Op<"QLinearMatMul_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX QLinearMatMul_V10 operation [v10]";
  let description = [{
      Matrix product that behaves like [numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$a,
                       TensorOf<[F32]>:$a_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$a_zero_point,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$b,
                       TensorOf<[F32]>:$b_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$b_zero_point,
                       TensorOf<[F32]>:$y_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y_zero_point);
  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 8;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// QLinearMatMul [v21]
def Onnx_QLinearMatMulOp : Onnx_Op<"QLinearMatMul", [Pure, OPCountInfo]> {
  let summary = "ONNX QLinearMatMul operation [v21]";
  let description = [{
      Matrix product that behaves like [numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$a,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$a_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$a_zero_point,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$b,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$b_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$b_zero_point,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y_scale,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$y_zero_point);
  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 8;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MatMulInteger [v10]
def Onnx_MatMulIntegerOp : Onnx_Op<"MatMulInteger", [Pure, OPCountInfo]> {
  let summary = "ONNX MatMulInteger operation [v10]";
  let description = [{
      Matrix product that behaves like [numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$A,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$B,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$a_zero_point,
                       AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$b_zero_point);
  let results = (outs TensorOf<[I32]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RandomNormal [v1]
def Onnx_RandomNormal_V1Op : Onnx_Op<"RandomNormal_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX RandomNormal_V1 operation [v1]";
  let description = [{
      Generate a tensor with random values drawn from a normal distribution.
  }];

  let arguments = (ins DefaultValuedAttr<I64Attr, "1">:$dtype,
                       DefaultValuedAttr<F32Attr, "0">:$mean,
                       DefaultValuedAttr<F32Attr, "1">:$scale,
                       OptionalAttr<F32Attr>:$seed,
                       I64ArrayAttr:$shape);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RandomNormal [v22]
def Onnx_RandomNormalOp : Onnx_Op<"RandomNormal", [Pure, OPCountInfo]> {
  let summary = "ONNX RandomNormal operation [v22]";
  let description = [{
      Generate a tensor with random values drawn from a normal distribution.
  }];

  let arguments = (ins DefaultValuedAttr<I64Attr, "1">:$dtype,
                       DefaultValuedAttr<F32Attr, "0">:$mean,
                       DefaultValuedAttr<F32Attr, "1">:$scale,
                       OptionalAttr<F32Attr>:$seed,
                       I64ArrayAttr:$shape);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Attention [v23]
def Onnx_AttentionOp : Onnx_Op<"Attention", [Pure, OPCountInfo]> {
  let summary = "ONNX Attention operation [v23]";
  let description = [{
      Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Q,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$K,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$V,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, NoneType]>:$attn_mask,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$past_key,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$past_value,
                       DefaultValuedAttr<I64Attr, "0">:$is_causal,
                       OptionalAttr<I64Attr>:$kv_num_heads,
                       OptionalAttr<I64Attr>:$q_num_heads,
                       DefaultValuedAttr<I64Attr, "0">:$qk_matmul_output_mode,
                       OptionalAttr<F32Attr>:$scale,
                       DefaultValuedAttr<F32Attr, "0">:$softcap,
                       OptionalAttr<I64Attr>:$softmax_precision);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y,
                      AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$present_key,
                      AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$present_value,
                      AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$qk_matmul_output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 4;
    }
  }];
}

/// IsInf [v10]
def Onnx_IsInf_V10Op : Onnx_Op<"IsInf_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX IsInf_V10 operation [v10]";
  let description = [{
      Map infinity to true and other values to false.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<I64Attr, "1">:$detect_negative,
                       DefaultValuedAttr<I64Attr, "1">:$detect_positive);
  let results = (outs TensorOf<[I1]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// IsInf [v20]
def Onnx_IsInfOp : Onnx_Op<"IsInf", [Pure, OPCountInfo]> {
  let summary = "ONNX IsInf operation [v20]";
  let description = [{
      Map infinity to true and other values to false.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$X,
                       DefaultValuedAttr<I64Attr, "1">:$detect_negative,
                       DefaultValuedAttr<I64Attr, "1">:$detect_positive);
  let results = (outs TensorOf<[I1]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Or [v1]
def Onnx_Or_V1Op : Onnx_Op<"Or_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Or_V1 operation [v1]";
  let description = [{
      Returns the tensor resulted from performing the `or` logical operation
  elementwise on the input tensors `A` and `B`.
  }];

  let arguments = (ins TensorOf<[I1]>:$A,
                       TensorOf<[I1]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Or [v7]
def Onnx_OrOp : Onnx_Op<"Or", [Pure, OPCountInfo]> {
  let summary = "ONNX Or operation [v7]";
  let description = [{
      Returns the tensor resulted from performing the `or` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins TensorOf<[I1]>:$A,
                       TensorOf<[I1]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ConcatFromSequence [v11]
def Onnx_ConcatFromSequenceOp : Onnx_Op<"ConcatFromSequence", [Pure, OPCountInfo]> {
  let summary = "ONNX ConcatFromSequence operation [v11]";
  let description = [{
      Concatenate a sequence of tensors into a single tensor.
  }];

  let arguments = (ins AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$input_sequence,
                       I64Attr:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$new_axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$concat_result);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Constant [v1]
def Onnx_Constant_V1Op : Onnx_Op<"Constant_V1", [Pure, OPCountInfo, ConstantLike]> {
  let summary = "ONNX Constant_V1 operation [v1]";
  let description = [{
      A constant tensor.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Constant [v9]
def Onnx_Constant_V9Op : Onnx_Op<"Constant_V9", [Pure, OPCountInfo, ConstantLike]> {
  let summary = "ONNX Constant_V9 operation [v9]";
  let description = [{
      A constant tensor.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, NoneType]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Constant [v11]
def Onnx_Constant_V11Op : Onnx_Op<"Constant_V11", [Pure, OPCountInfo, ConstantLike]> {
  let summary = "ONNX Constant_V11 operation [v11]";
  let description = [{
      A constant tensor.
  }];

  let arguments = (ins OptionalAttr<AnyAttr>:$sparse_value,
                       OptionalAttr<AnyAttr>:$value);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, NoneType]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Constant [v12]
def Onnx_Constant_V12Op : Onnx_Op<"Constant_V12", [Pure, OPCountInfo, ConstantLike]> {
  let summary = "ONNX Constant_V12 operation [v12]";
  let description = [{
      This operator produces a constant tensor.
  }];

  let arguments = (ins OptionalAttr<AnyAttr>:$sparse_value,
                       OptionalAttr<AnyAttr>:$value,
                       OptionalAttr<F32Attr>:$value_float,
                       OptionalAttr<F32ArrayAttr>:$value_floats,
                       OptionalAttr<I64Attr>:$value_int,
                       OptionalAttr<I64ArrayAttr>:$value_ints,
                       OptionalAttr<StrAttr>:$value_string,
                       OptionalAttr<StrArrayAttr>:$value_strings);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, NoneType]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Constant [v13]
def Onnx_Constant_V13Op : Onnx_Op<"Constant_V13", [Pure, OPCountInfo, ConstantLike]> {
  let summary = "ONNX Constant_V13 operation [v13]";
  let description = [{
      This operator produces a constant tensor.
  }];

  let arguments = (ins OptionalAttr<AnyAttr>:$sparse_value,
                       OptionalAttr<AnyAttr>:$value,
                       OptionalAttr<F32Attr>:$value_float,
                       OptionalAttr<F32ArrayAttr>:$value_floats,
                       OptionalAttr<I64Attr>:$value_int,
                       OptionalAttr<I64ArrayAttr>:$value_ints,
                       OptionalAttr<StrAttr>:$value_string,
                       OptionalAttr<StrArrayAttr>:$value_strings);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, NoneType]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Constant [v19]
def Onnx_Constant_V19Op : Onnx_Op<"Constant_V19", [Pure, OPCountInfo, ConstantLike]> {
  let summary = "ONNX Constant_V19 operation [v19]";
  let description = [{
      This operator produces a constant tensor.
  }];

  let arguments = (ins OptionalAttr<AnyAttr>:$sparse_value,
                       OptionalAttr<AnyAttr>:$value,
                       OptionalAttr<F32Attr>:$value_float,
                       OptionalAttr<F32ArrayAttr>:$value_floats,
                       OptionalAttr<I64Attr>:$value_int,
                       OptionalAttr<I64ArrayAttr>:$value_ints,
                       OptionalAttr<StrAttr>:$value_string,
                       OptionalAttr<StrArrayAttr>:$value_strings);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Constant [v21]
def Onnx_Constant_V21Op : Onnx_Op<"Constant_V21", [Pure, OPCountInfo, ConstantLike]> {
  let summary = "ONNX Constant_V21 operation [v21]";
  let description = [{
      This operator produces a constant tensor.
  }];

  let arguments = (ins OptionalAttr<AnyAttr>:$sparse_value,
                       OptionalAttr<AnyAttr>:$value,
                       OptionalAttr<F32Attr>:$value_float,
                       OptionalAttr<F32ArrayAttr>:$value_floats,
                       OptionalAttr<I64Attr>:$value_int,
                       OptionalAttr<I64ArrayAttr>:$value_ints,
                       OptionalAttr<StrAttr>:$value_string,
                       OptionalAttr<StrArrayAttr>:$value_strings);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, NoneType]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Constant [v23]
def Onnx_ConstantOp : Onnx_Op<"Constant", [Pure, OPCountInfo, ConstantLike]> {
  let summary = "ONNX Constant operation [v23]";
  let description = [{
      This operator produces a constant tensor.
  }];

  let arguments = (ins OptionalAttr<AnyAttr>:$sparse_value,
                       OptionalAttr<AnyAttr>:$value,
                       OptionalAttr<F32Attr>:$value_float,
                       OptionalAttr<F32ArrayAttr>:$value_floats,
                       OptionalAttr<I64Attr>:$value_int,
                       OptionalAttr<I64ArrayAttr>:$value_ints,
                       OptionalAttr<StrAttr>:$value_string,
                       OptionalAttr<StrArrayAttr>:$value_strings);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, NoneType]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mul [v1]
def Onnx_Mul_V1Op : Onnx_Op<"Mul_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Mul_V1 operation [v1]";
  let description = [{
      Performs element-wise binary multiplication (with limited broadcast support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mul [v6]
def Onnx_Mul_V6Op : Onnx_Op<"Mul_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Mul_V6 operation [v6]";
  let description = [{
      Performs element-wise binary multiplication (with limited broadcast support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mul [v7]
def Onnx_Mul_V7Op : Onnx_Op<"Mul_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Mul_V7 operation [v7]";
  let description = [{
      Performs element-wise binary multiplication (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mul [v13]
def Onnx_Mul_V13Op : Onnx_Op<"Mul_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Mul_V13 operation [v13]";
  let description = [{
      Performs element-wise binary multiplication (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mul [v14]
def Onnx_MulOp : Onnx_Op<"Mul", [Pure, OPCountInfo]> {
  let summary = "ONNX Mul operation [v14]";
  let description = [{
      Performs element-wise binary multiplication (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Max [v1]
def Onnx_Max_V1Op : Onnx_Op<"Max_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Max_V1 operation [v1]";
  let description = [{
      Element-wise max of each of the input tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$max);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Max [v6]
def Onnx_Max_V6Op : Onnx_Op<"Max_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Max_V6 operation [v6]";
  let description = [{
      Element-wise max of each of the input tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$max);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Max [v8]
def Onnx_Max_V8Op : Onnx_Op<"Max_V8", [Pure, OPCountInfo]> {
  let summary = "ONNX Max_V8 operation [v8]";
  let description = [{
      Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$max);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Max [v12]
def Onnx_Max_V12Op : Onnx_Op<"Max_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX Max_V12 operation [v12]";
  let description = [{
      Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$max);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Max [v13]
def Onnx_MaxOp : Onnx_Op<"Max", [Pure, OPCountInfo]> {
  let summary = "ONNX Max operation [v13]";
  let description = [{
      Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$max);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GRU [v1]
def Onnx_GRU_V1Op : Onnx_Op<"GRU_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX GRU_V1 operation [v1]";
  let description = [{
      Computes an one-layer GRU.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       OptionalAttr<StrArrayAttr>:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "foward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$output_sequence);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y_h);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// GRU [v3]
def Onnx_GRU_V3Op : Onnx_Op<"GRU_V3", [Pure, OPCountInfo]> {
  let summary = "ONNX GRU_V3 operation [v3]";
  let description = [{
      Computes an one-layer GRU.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       OptionalAttr<StrArrayAttr>:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$linear_before_reset,
                       DefaultValuedAttr<I64Attr, "0">:$output_sequence);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// GRU [v7]
def Onnx_GRU_V7Op : Onnx_Op<"GRU_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX GRU_V7 operation [v7]";
  let description = [{
      Computes an one-layer GRU.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       OptionalAttr<StrArrayAttr>:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$linear_before_reset);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// GRU [v14]
def Onnx_GRU_V14Op : Onnx_Op<"GRU_V14", [Pure, OPCountInfo]> {
  let summary = "ONNX GRU_V14 operation [v14]";
  let description = [{
      Computes an one-layer GRU.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       OptionalAttr<StrArrayAttr>:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$layout,
                       DefaultValuedAttr<I64Attr, "0">:$linear_before_reset);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// GRU [v22]
def Onnx_GRUOp : Onnx_Op<"GRU", [Pure, OPCountInfo]> {
  let summary = "ONNX GRU operation [v22]";
  let description = [{
      Computes an one-layer GRU.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       OptionalAttr<StrArrayAttr>:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$layout,
                       DefaultValuedAttr<I64Attr, "0">:$linear_before_reset);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// GroupNormalization [v18]
def Onnx_GroupNormalization_V18Op : Onnx_Op<"GroupNormalization_V18", [Pure, OPCountInfo]> {
  let summary = "ONNX GroupNormalization_V18 operation [v18]";
  let description = [{
      A GroupNormalization function.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$bias,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       I64Attr:$num_groups);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GroupNormalization [v21]
def Onnx_GroupNormalizationOp : Onnx_Op<"GroupNormalization", [Pure, OPCountInfo]> {
  let summary = "ONNX GroupNormalization operation [v21]";
  let description = [{
      A GroupNormalization function.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scale,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$bias,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       I64Attr:$num_groups,
                       DefaultValuedAttr<I64Attr, "1">:$stash_type);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mod [v10]
def Onnx_Mod_V10Op : Onnx_Op<"Mod_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX Mod_V10 operation [v10]";
  let description = [{
      Performs element-wise binary modulus (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       DefaultValuedAttr<I64Attr, "0">:$fmod);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mod [v13]
def Onnx_ModOp : Onnx_Op<"Mod", [Pure, OPCountInfo]> {
  let summary = "ONNX Mod operation [v13]";
  let description = [{
      Performs element-wise binary modulus (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B,
                       DefaultValuedAttr<I64Attr, "0">:$fmod);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Log [v1]
def Onnx_Log_V1Op : Onnx_Op<"Log_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Log_V1 operation [v1]";
  let description = [{
      Calculates the natural log of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Log [v6]
def Onnx_Log_V6Op : Onnx_Op<"Log_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Log_V6 operation [v6]";
  let description = [{
      Calculates the natural log of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Log [v13]
def Onnx_LogOp : Onnx_Op<"Log", [Pure, OPCountInfo]> {
  let summary = "ONNX Log operation [v13]";
  let description = [{
      Calculates the natural log of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ArgMax [v1]
def Onnx_ArgMax_V1Op : Onnx_Op<"ArgMax_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ArgMax_V1 operation [v1]";
  let description = [{
      Computes the indices of the max elements of the input tensor's element along the
  provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs TensorOf<[I64]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ArgMax [v11]
def Onnx_ArgMax_V11Op : Onnx_Op<"ArgMax_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ArgMax_V11 operation [v11]";
  let description = [{
      Computes the indices of the max elements of the input tensor's element along the
  provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs TensorOf<[I64]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ArgMax [v12]
def Onnx_ArgMax_V12Op : Onnx_Op<"ArgMax_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX ArgMax_V12 operation [v12]";
  let description = [{
      Computes the indices of the max elements of the input tensor's element along the
  provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$select_last_index);
  let results = (outs TensorOf<[I64]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ArgMax [v13]
def Onnx_ArgMaxOp : Onnx_Op<"ArgMax", [Pure, OPCountInfo]> {
  let summary = "ONNX ArgMax operation [v13]";
  let description = [{
      Computes the indices of the max elements of the input tensor's element along the
  provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$select_last_index);
  let results = (outs TensorOf<[I64]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMax [v1]
def Onnx_ReduceMax_V1Op : Onnx_Op<"ReduceMax_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMax_V1 operation [v1]";
  let description = [{
      Computes the max of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMax [v11]
def Onnx_ReduceMax_V11Op : Onnx_Op<"ReduceMax_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMax_V11 operation [v11]";
  let description = [{
      Computes the max of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMax [v12]
def Onnx_ReduceMax_V12Op : Onnx_Op<"ReduceMax_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMax_V12 operation [v12]";
  let description = [{
      Computes the max of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMax [v13]
def Onnx_ReduceMax_V13Op : Onnx_Op<"ReduceMax_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMax_V13 operation [v13]";
  let description = [{
      Computes the max of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMax [v18]
def Onnx_ReduceMax_V18Op : Onnx_Op<"ReduceMax_V18", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMax_V18 operation [v18]";
  let description = [{
      Computes the max of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMax [v20]
def Onnx_ReduceMaxOp : Onnx_Op<"ReduceMax", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMax operation [v20]";
  let description = [{
      Computes the max of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Split [v1]
def Onnx_Split_V1Op : Onnx_Op<"Split_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Split_V1 operation [v1]";
  let description = [{
      Split a tensor into a list of tensors, along the specified
  'axis'.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$split,
                       OptionalAttr<I64Attr>:$axis,
                       OptionalAttr<I64ArrayAttr>:$input_split);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Split [v2]
def Onnx_Split_V2Op : Onnx_Op<"Split_V2", [Pure, OPCountInfo]> {
  let summary = "ONNX Split_V2 operation [v2]";
  let description = [{
      Split a tensor into a list of tensors, along the specified
  'axis'.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       OptionalAttr<I64ArrayAttr>:$split);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Split [v11]
def Onnx_Split_V11Op : Onnx_Op<"Split_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Split_V11 operation [v11]";
  let description = [{
      Split a tensor into a list of tensors, along the specified
  'axis'.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       OptionalAttr<I64ArrayAttr>:$split);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Split [v13]
def Onnx_Split_V13Op : Onnx_Op<"Split_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Split_V13 operation [v13]";
  let description = [{
      Split a tensor into a list of tensors, along the specified
  'axis'.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$split,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Split [v18]
def Onnx_SplitOp : Onnx_Op<"Split", [Pure, OPCountInfo]> {
  let summary = "ONNX Split operation [v18]";
  let description = [{
      Split a tensor into a list of tensors, along the specified 'axis'.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$split,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       OptionalAttr<I64Attr>:$num_outputs);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// If [v1]
def Onnx_If_V1Op : Onnx_Op<"If_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX If_V1 operation [v1]";
  let description = [{
      If conditional.
  }];

  let arguments = (ins TensorOf<[I1]>:$cond);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// If [v11]
def Onnx_If_V11Op : Onnx_Op<"If_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX If_V11 operation [v11]";
  let description = [{
      If conditional.
  }];

  let arguments = (ins TensorOf<[I1]>:$cond);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// If [v13]
def Onnx_If_V13Op : Onnx_Op<"If_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX If_V13 operation [v13]";
  let description = [{
      If conditional.
  }];

  let arguments = (ins TensorOf<[I1]>:$cond);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// If [v16]
def Onnx_If_V16Op : Onnx_Op<"If_V16", [Pure, OPCountInfo]> {
  let summary = "ONNX If_V16 operation [v16]";
  let description = [{
      If conditional.
  }];

  let arguments = (ins TensorOf<[I1]>:$cond);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// If [v19]
def Onnx_If_V19Op : Onnx_Op<"If_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX If_V19 operation [v19]";
  let description = [{
      If conditional.
  }];

  let arguments = (ins TensorOf<[I1]>:$cond);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[F8E4M3FN]>]>, SeqOf<[TensorOf<[F8E4M3FNUZ]>]>, SeqOf<[TensorOf<[F8E5M2]>]>, SeqOf<[TensorOf<[F8E5M2FNUZ]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, OptOf<TensorOf<[F8E4M3FN]>>, OptOf<TensorOf<[F8E4M3FNUZ]>>, OptOf<TensorOf<[F8E5M2]>>, OptOf<TensorOf<[F8E5M2FNUZ]>>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// If [v21]
def Onnx_If_V21Op : Onnx_Op<"If_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX If_V21 operation [v21]";
  let description = [{
      If conditional.
  }];

  let arguments = (ins TensorOf<[I1]>:$cond);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[F8E4M3FN]>]>, SeqOf<[TensorOf<[F8E4M3FNUZ]>]>, SeqOf<[TensorOf<[F8E5M2]>]>, SeqOf<[TensorOf<[F8E5M2FNUZ]>]>, SeqOf<[TensorOf<[UI<4>]>]>, SeqOf<[TensorOf<[I<4>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, OptOf<TensorOf<[F8E4M3FN]>>, OptOf<TensorOf<[F8E4M3FNUZ]>>, OptOf<TensorOf<[F8E5M2]>>, OptOf<TensorOf<[F8E5M2FNUZ]>>, OptOf<TensorOf<[UI<4>]>>, OptOf<TensorOf<[I<4>]>>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// If [v23]
def Onnx_IfOp : Onnx_Op<"If", [Pure, OPCountInfo]> {
  let summary = "ONNX If operation [v23]";
  let description = [{
      If conditional.
  }];

  let arguments = (ins TensorOf<[I1]>:$cond);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[F8E4M3FN]>]>, SeqOf<[TensorOf<[F8E4M3FNUZ]>]>, SeqOf<[TensorOf<[F8E5M2]>]>, SeqOf<[TensorOf<[F8E5M2FNUZ]>]>, SeqOf<[TensorOf<[UI<4>]>]>, SeqOf<[TensorOf<[I<4>]>]>, SeqOf<[TensorOf<[F4E2M1FN]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, OptOf<TensorOf<[F8E4M3FN]>>, OptOf<TensorOf<[F8E4M3FNUZ]>>, OptOf<TensorOf<[F8E5M2]>>, OptOf<TensorOf<[F8E5M2FNUZ]>>, OptOf<TensorOf<[UI<4>]>>, OptOf<TensorOf<[I<4>]>>, OptOf<TensorOf<[F4E2M1FN]>>]>>:$outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMin [v1]
def Onnx_ReduceMin_V1Op : Onnx_Op<"ReduceMin_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMin_V1 operation [v1]";
  let description = [{
      Computes the min of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMin [v11]
def Onnx_ReduceMin_V11Op : Onnx_Op<"ReduceMin_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMin_V11 operation [v11]";
  let description = [{
      Computes the min of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMin [v12]
def Onnx_ReduceMin_V12Op : Onnx_Op<"ReduceMin_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMin_V12 operation [v12]";
  let description = [{
      Computes the min of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMin [v13]
def Onnx_ReduceMin_V13Op : Onnx_Op<"ReduceMin_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMin_V13 operation [v13]";
  let description = [{
      Computes the min of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMin [v18]
def Onnx_ReduceMin_V18Op : Onnx_Op<"ReduceMin_V18", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMin_V18 operation [v18]";
  let description = [{
      Computes the min of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMin [v20]
def Onnx_ReduceMinOp : Onnx_Op<"ReduceMin", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMin operation [v20]";
  let description = [{
      Computes the min of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DeformConv [v19]
def Onnx_DeformConv_V19Op : Onnx_Op<"DeformConv_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX DeformConv_V19 operation [v19]";
  let description = [{
      Performs deformable convolution as described in https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$offset,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$mask,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       DefaultValuedAttr<I64Attr, "1">:$offset_group,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DeformConv [v22]
def Onnx_DeformConvOp : Onnx_Op<"DeformConv", [Pure, OPCountInfo]> {
  let summary = "ONNX DeformConv operation [v22]";
  let description = [{
      Performs deformable convolution as described in https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$offset,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$mask,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       DefaultValuedAttr<I64Attr, "1">:$offset_group,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MaxPool [v1]
def Onnx_MaxPool_V1Op : Onnx_Op<"MaxPool_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxPool_V1 operation [v1]";
  let description = [{
      MaxPool consumes an input tensor X and applies max pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MaxPool [v8]
def Onnx_MaxPool_V8Op : Onnx_Op<"MaxPool_V8", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxPool_V8 operation [v8]";
  let description = [{
      MaxPool consumes an input tensor X and applies max pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       DefaultValuedAttr<I64Attr, "0">:$storage_order,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y,
                      AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$Indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// MaxPool [v10]
def Onnx_MaxPool_V10Op : Onnx_Op<"MaxPool_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxPool_V10 operation [v10]";
  let description = [{
      MaxPool consumes an input tensor X and applies max pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       DefaultValuedAttr<I64Attr, "0">:$storage_order,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y,
                      AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$Indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// MaxPool [v11]
def Onnx_MaxPool_V11Op : Onnx_Op<"MaxPool_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxPool_V11 operation [v11]";
  let description = [{
      MaxPool consumes an input tensor X and applies max pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       DefaultValuedAttr<I64Attr, "0">:$storage_order,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y,
                      AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$Indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// MaxPool [v12]
def Onnx_MaxPool_V12Op : Onnx_Op<"MaxPool_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxPool_V12 operation [v12]";
  let description = [{
      MaxPool consumes an input tensor X and applies max pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[UI8]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       DefaultValuedAttr<I64Attr, "0">:$storage_order,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[UI8]>]>:$Y,
                      AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$Indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// MaxPool [v22]
def Onnx_MaxPoolOp : Onnx_Op<"MaxPool", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxPool operation [v22]";
  let description = [{
      MaxPool consumes an input tensor X and applies max pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[UI8]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       DefaultValuedAttr<I64Attr, "0">:$storage_order,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[UI8]>]>:$Y,
                      AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$Indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// Sign [v9]
def Onnx_Sign_V9Op : Onnx_Op<"Sign_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Sign_V9 operation [v9]";
  let description = [{
      Calculate the sign of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sign [v13]
def Onnx_SignOp : Onnx_Op<"Sign", [Pure, OPCountInfo]> {
  let summary = "ONNX Sign operation [v13]";
  let description = [{
      Calculate the sign of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Min [v1]
def Onnx_Min_V1Op : Onnx_Op<"Min_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Min_V1 operation [v1]";
  let description = [{
      Element-wise min of each of the input tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$min);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Min [v6]
def Onnx_Min_V6Op : Onnx_Op<"Min_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Min_V6 operation [v6]";
  let description = [{
      Element-wise min of each of the input tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$min);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Min [v8]
def Onnx_Min_V8Op : Onnx_Op<"Min_V8", [Pure, OPCountInfo]> {
  let summary = "ONNX Min_V8 operation [v8]";
  let description = [{
      Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$min);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Min [v12]
def Onnx_Min_V12Op : Onnx_Op<"Min_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX Min_V12 operation [v12]";
  let description = [{
      Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$min);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Min [v13]
def Onnx_MinOp : Onnx_Op<"Min", [Pure, OPCountInfo]> {
  let summary = "ONNX Min operation [v13]";
  let description = [{
      Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$min);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Range [v11]
def Onnx_RangeOp : Onnx_Op<"Range", [Pure, OPCountInfo]> {
  let summary = "ONNX Range operation [v11]";
  let description = [{
      Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
  up to `limit` (exclusive).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$start,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$limit,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$delta);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cast [v1]
def Onnx_Cast_V1Op : Onnx_Op<"Cast_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Cast_V1 operation [v1]";
  let description = [{
      The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>:$input,
                       StrAttr:$to);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cast [v6]
def Onnx_Cast_V6Op : Onnx_Op<"Cast_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Cast_V6 operation [v6]";
  let description = [{
      The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>:$input,
                       I64Attr:$to);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cast [v9]
def Onnx_Cast_V9Op : Onnx_Op<"Cast_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Cast_V9 operation [v9]";
  let description = [{
      The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>]>:$input,
                       I64Attr:$to);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cast [v13]
def Onnx_Cast_V13Op : Onnx_Op<"Cast_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Cast_V13 operation [v13]";
  let description = [{
      The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>:$input,
                       I64Attr:$to);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cast [v19]
def Onnx_Cast_V19Op : Onnx_Op<"Cast_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX Cast_V19 operation [v19]";
  let description = [{
      The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$saturate,
                       I64Attr:$to);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cast [v21]
def Onnx_Cast_V21Op : Onnx_Op<"Cast_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Cast_V21 operation [v21]";
  let description = [{
      The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$saturate,
                       I64Attr:$to);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[StringType]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cast [v23]
def Onnx_CastOp : Onnx_Op<"Cast", [Pure, OPCountInfo]> {
  let summary = "ONNX Cast operation [v23]";
  let description = [{
      The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$saturate,
                       I64Attr:$to);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// PRelu [v1]
def Onnx_PRelu_V1Op : Onnx_Op<"PRelu_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX PRelu_V1 operation [v1]";
  let description = [{
      PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$slope,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// PRelu [v6]
def Onnx_PRelu_V6Op : Onnx_Op<"PRelu_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX PRelu_V6 operation [v6]";
  let description = [{
      PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$slope);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// PRelu [v7]
def Onnx_PRelu_V7Op : Onnx_Op<"PRelu_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX PRelu_V7 operation [v7]";
  let description = [{
      PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$slope);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// PRelu [v9]
def Onnx_PRelu_V9Op : Onnx_Op<"PRelu_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX PRelu_V9 operation [v9]";
  let description = [{
      PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$slope);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// PRelu [v16]
def Onnx_PReluOp : Onnx_Op<"PRelu", [Pure, OPCountInfo]> {
  let summary = "ONNX PRelu operation [v16]";
  let description = [{
      PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$slope);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RandomUniform [v1]
def Onnx_RandomUniform_V1Op : Onnx_Op<"RandomUniform_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX RandomUniform_V1 operation [v1]";
  let description = [{
      Generate a tensor with random values drawn from a uniform distribution.
  }];

  let arguments = (ins DefaultValuedAttr<I64Attr, "1">:$dtype,
                       DefaultValuedAttr<F32Attr, "1">:$high,
                       DefaultValuedAttr<F32Attr, "0">:$low,
                       OptionalAttr<F32Attr>:$seed,
                       I64ArrayAttr:$shape);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RandomUniform [v22]
def Onnx_RandomUniformOp : Onnx_Op<"RandomUniform", [Pure, OPCountInfo]> {
  let summary = "ONNX RandomUniform operation [v22]";
  let description = [{
      Generate a tensor with random values drawn from a uniform distribution.
  }];

  let arguments = (ins DefaultValuedAttr<I64Attr, "1">:$dtype,
                       DefaultValuedAttr<F32Attr, "1">:$high,
                       DefaultValuedAttr<F32Attr, "0">:$low,
                       OptionalAttr<F32Attr>:$seed,
                       I64ArrayAttr:$shape);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pad [v1]
def Onnx_Pad_V1Op : Onnx_Op<"Pad_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Pad_V1 operation [v1]";
  let description = [{
      Given `data` tensor, paddings, mode, and value.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedStrAttr<StrAttr, "constant">:$mode,
                       I64ArrayAttr:$paddings,
                       DefaultValuedAttr<F32Attr, "0">:$value);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pad [v2]
def Onnx_Pad_V2Op : Onnx_Op<"Pad_V2", [Pure, OPCountInfo]> {
  let summary = "ONNX Pad_V2 operation [v2]";
  let description = [{
      Given `data` tensor, pads, mode, and value.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedStrAttr<StrAttr, "constant">:$mode,
                       I64ArrayAttr:$pads,
                       DefaultValuedAttr<F32Attr, "0">:$value);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pad [v11]
def Onnx_Pad_V11Op : Onnx_Op<"Pad_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Pad_V11 operation [v11]";
  let description = [{
      Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
  a padded tensor (`output`) is generated.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       TensorOf<[I64]>:$pads,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$constant_value,
                       DefaultValuedStrAttr<StrAttr, "constant">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pad [v13]
def Onnx_Pad_V13Op : Onnx_Op<"Pad_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Pad_V13 operation [v13]";
  let description = [{
      Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
  a padded tensor (`output`) is generated.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$pads,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, NoneType]>:$constant_value,
                       DefaultValuedStrAttr<StrAttr, "constant">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pad [v18]
def Onnx_Pad_V18Op : Onnx_Op<"Pad_V18", [Pure, OPCountInfo]> {
  let summary = "ONNX Pad_V18 operation [v18]";
  let description = [{
      Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
  a padded tensor (`output`) is generated.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$pads,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, NoneType]>:$constant_value,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedStrAttr<StrAttr, "constant">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pad [v19]
def Onnx_Pad_V19Op : Onnx_Op<"Pad_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX Pad_V19 operation [v19]";
  let description = [{
      Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
  a padded tensor (`output`) is generated.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$pads,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, NoneType]>:$constant_value,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedStrAttr<StrAttr, "constant">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pad [v21]
def Onnx_Pad_V21Op : Onnx_Op<"Pad_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Pad_V21 operation [v21]";
  let description = [{
      Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
  a padded tensor (`output`) is generated.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$data,
                       TensorOf<[I64]>:$pads,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, NoneType]>:$constant_value,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedStrAttr<StrAttr, "constant">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pad [v23]
def Onnx_PadOp : Onnx_Op<"Pad", [Pure, OPCountInfo]> {
  let summary = "ONNX Pad operation [v23]";
  let description = [{
      Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
  a padded tensor (`output`) is generated.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$data,
                       TensorOf<[I64]>:$pads,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, NoneType]>:$constant_value,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedStrAttr<StrAttr, "constant">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// NonZero [v9]
def Onnx_NonZero_V9Op : Onnx_Op<"NonZero_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX NonZero_V9 operation [v9]";
  let description = [{
      Returns the indices of the elements that are non-zero
  (in row-major order - by dimension).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X);
  let results = (outs TensorOf<[I64]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// NonZero [v13]
def Onnx_NonZeroOp : Onnx_Op<"NonZero", [Pure, OPCountInfo]> {
  let summary = "ONNX NonZero operation [v13]";
  let description = [{
      Returns the indices of the elements that are non-zero
  (in row-major order - by dimension).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X);
  let results = (outs TensorOf<[I64]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Ceil [v1]
def Onnx_Ceil_V1Op : Onnx_Op<"Ceil_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Ceil_V1 operation [v1]";
  let description = [{
      Ceil takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the ceil is, y = ceil(x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Ceil [v6]
def Onnx_Ceil_V6Op : Onnx_Op<"Ceil_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Ceil_V6 operation [v6]";
  let description = [{
      Ceil takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the ceil is, y = ceil(x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Ceil [v13]
def Onnx_CeilOp : Onnx_Op<"Ceil", [Pure, OPCountInfo]> {
  let summary = "ONNX Ceil operation [v13]";
  let description = [{
      Ceil takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the ceil is, y = ceil(x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Tan [v7]
def Onnx_Tan_V7Op : Onnx_Op<"Tan_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Tan_V7 operation [v7]";
  let description = [{
      Calculates the tangent of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Tan [v22]
def Onnx_TanOp : Onnx_Op<"Tan", [Pure, OPCountInfo]> {
  let summary = "ONNX Tan operation [v22]";
  let description = [{
      Calculates the tangent of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Not [v1]
def Onnx_NotOp : Onnx_Op<"Not", [Pure, OPCountInfo]> {
  let summary = "ONNX Not operation [v1]";
  let description = [{
      Returns the negation of the input tensor element-wise.
  }];

  let arguments = (ins TensorOf<[I1]>:$X);
  let results = (outs TensorOf<[I1]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Clip [v1]
def Onnx_Clip_V1Op : Onnx_Op<"Clip_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Clip_V1 operation [v1]";
  let description = [{
      Clip operator limits the given input within an interval.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs,
                       OptionalAttr<F32Attr>:$max,
                       OptionalAttr<F32Attr>:$min);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Clip [v6]
def Onnx_Clip_V6Op : Onnx_Op<"Clip_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Clip_V6 operation [v6]";
  let description = [{
      Clip operator limits the given input within an interval.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<F32Attr, "3.40282e+38">:$max,
                       DefaultValuedAttr<F32Attr, "-3.40282e+38">:$min);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Clip [v11]
def Onnx_Clip_V11Op : Onnx_Op<"Clip_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Clip_V11 operation [v11]";
  let description = [{
      Clip operator limits the given input within an interval.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$min,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$max);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Clip [v12]
def Onnx_Clip_V12Op : Onnx_Op<"Clip_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX Clip_V12 operation [v12]";
  let description = [{
      Clip operator limits the given input within an interval.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$min,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$max);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Clip [v13]
def Onnx_ClipOp : Onnx_Op<"Clip", [Pure, OPCountInfo]> {
  let summary = "ONNX Clip operation [v13]";
  let description = [{
      Clip operator limits the given input within an interval.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$min,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$max);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceL2 [v1]
def Onnx_ReduceL2_V1Op : Onnx_Op<"ReduceL2_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceL2_V1 operation [v1]";
  let description = [{
      Computes the L2 norm of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceL2 [v11]
def Onnx_ReduceL2_V11Op : Onnx_Op<"ReduceL2_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceL2_V11 operation [v11]";
  let description = [{
      Computes the L2 norm of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceL2 [v13]
def Onnx_ReduceL2_V13Op : Onnx_Op<"ReduceL2_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceL2_V13 operation [v13]";
  let description = [{
      Computes the L2 norm of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceL2 [v18]
def Onnx_ReduceL2Op : Onnx_Op<"ReduceL2", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceL2 operation [v18]";
  let description = [{
      Computes the L2 norm of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Neg [v1]
def Onnx_Neg_V1Op : Onnx_Op<"Neg_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Neg_V1 operation [v1]";
  let description = [{
      Neg takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where each element flipped sign, y = -x, is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Neg [v6]
def Onnx_Neg_V6Op : Onnx_Op<"Neg_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Neg_V6 operation [v6]";
  let description = [{
      Neg takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where each element flipped sign, y = -x, is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Neg [v13]
def Onnx_NegOp : Onnx_Op<"Neg", [Pure, OPCountInfo]> {
  let summary = "ONNX Neg operation [v13]";
  let description = [{
      Neg takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where each element flipped sign, y = -x, is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// BitwiseXor [v18]
def Onnx_BitwiseXorOp : Onnx_Op<"BitwiseXor", [Pure, OPCountInfo]> {
  let summary = "ONNX BitwiseXor operation [v18]";
  let description = [{
      Returns the tensor resulting from performing the bitwise `xor` operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Conv [v1]
def Onnx_Conv_V1Op : Onnx_Op<"Conv_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Conv_V1 operation [v1]";
  let description = [{
      The convolution operator consumes an input tensor and a filter, and
  computes the output.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Conv [v11]
def Onnx_Conv_V11Op : Onnx_Op<"Conv_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Conv_V11 operation [v11]";
  let description = [{
      The convolution operator consumes an input tensor and a filter, and
  computes the output.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Conv [v22]
def Onnx_ConvOp : Onnx_Op<"Conv", [Pure, OPCountInfo]> {
  let summary = "ONNX Conv operation [v22]";
  let description = [{
      The convolution operator consumes an input tensor and a filter, and
  computes the output.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// StringConcat [v20]
def Onnx_StringConcatOp : Onnx_Op<"StringConcat", [Pure, OPCountInfo]> {
  let summary = "ONNX StringConcat operation [v20]";
  let description = [{
      StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support).
  }];

  let arguments = (ins TensorOf<[StringType]>:$X,
                       TensorOf<[StringType]>:$Y);
  let results = (outs TensorOf<[StringType]>:$Z);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Abs [v1]
def Onnx_Abs_V1Op : Onnx_Op<"Abs_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Abs_V1 operation [v1]";
  let description = [{
      Absolute takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the absolute is, y = abs(x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Abs [v6]
def Onnx_Abs_V6Op : Onnx_Op<"Abs_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Abs_V6 operation [v6]";
  let description = [{
      Absolute takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the absolute is, y = abs(x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Abs [v13]
def Onnx_AbsOp : Onnx_Op<"Abs", [Pure, OPCountInfo]> {
  let summary = "ONNX Abs operation [v13]";
  let description = [{
      Absolute takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where absolute value, y = abs(x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Softplus [v1]
def Onnx_Softplus_V1Op : Onnx_Op<"Softplus_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Softplus_V1 operation [v1]";
  let description = [{
      Softplus takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Softplus [v22]
def Onnx_SoftplusOp : Onnx_Op<"Softplus", [Pure, OPCountInfo]> {
  let summary = "ONNX Softplus operation [v22]";
  let description = [{
      Softplus takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MaxUnpool [v9]
def Onnx_MaxUnpool_V9Op : Onnx_Op<"MaxUnpool_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxUnpool_V9 operation [v9]";
  let description = [{
      MaxUnpool essentially computes the partial inverse of the MaxPool op.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       TensorOf<[I64]>:$I,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$output_shape,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MaxUnpool [v11]
def Onnx_MaxUnpool_V11Op : Onnx_Op<"MaxUnpool_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxUnpool_V11 operation [v11]";
  let description = [{
      MaxUnpool essentially computes the partial inverse of the MaxPool op.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       TensorOf<[I64]>:$I,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$output_shape,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MaxUnpool [v22]
def Onnx_MaxUnpoolOp : Onnx_Op<"MaxUnpool", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxUnpool operation [v22]";
  let description = [{
      MaxUnpool essentially computes the partial inverse of the MaxPool op.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       TensorOf<[I64]>:$I,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$output_shape,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// TfIdfVectorizer [v9]
def Onnx_TfIdfVectorizerOp : Onnx_Op<"TfIdfVectorizer", [Pure, OPCountInfo]> {
  let summary = "ONNX TfIdfVectorizer operation [v9]";
  let description = [{
      This transform extracts n-grams from the input sequence and save them as a vector.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[StringType]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$X,
                       I64Attr:$max_gram_length,
                       I64Attr:$max_skip_count,
                       I64Attr:$min_gram_length,
                       StrAttr:$mode,
                       I64ArrayAttr:$ngram_counts,
                       I64ArrayAttr:$ngram_indexes,
                       OptionalAttr<I64ArrayAttr>:$pool_int64s,
                       OptionalAttr<StrArrayAttr>:$pool_strings,
                       OptionalAttr<F32ArrayAttr>:$weights);
  let results = (outs TensorOf<[F32]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ConvTranspose [v1]
def Onnx_ConvTranspose_V1Op : Onnx_Op<"ConvTranspose_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ConvTranspose_V1 operation [v1]";
  let description = [{
      The convolution transpose operator consumes an input tensor and a filter,
  and computes the output.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$output_padding,
                       OptionalAttr<I64ArrayAttr>:$output_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ConvTranspose [v11]
def Onnx_ConvTranspose_V11Op : Onnx_Op<"ConvTranspose_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ConvTranspose_V11 operation [v11]";
  let description = [{
      The convolution transpose operator consumes an input tensor and a filter,
  and computes the output.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$output_padding,
                       OptionalAttr<I64ArrayAttr>:$output_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ConvTranspose [v22]
def Onnx_ConvTransposeOp : Onnx_Op<"ConvTranspose", [Pure, OPCountInfo]> {
  let summary = "ONNX ConvTranspose operation [v22]";
  let description = [{
      The convolution transpose operator consumes an input tensor and a filter,
  and computes the output.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64Attr, "1">:$group,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$output_padding,
                       OptionalAttr<I64ArrayAttr>:$output_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SequenceEmpty [v11]
def Onnx_SequenceEmptyOp : Onnx_Op<"SequenceEmpty", [Pure, OPCountInfo]> {
  let summary = "ONNX SequenceEmpty operation [v11]";
  let description = [{
      Construct an empty tensor sequence, with given data type.
  }];

  let arguments = (ins OptionalAttr<I64Attr>:$dtype);
  let results = (outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 0;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// And [v1]
def Onnx_And_V1Op : Onnx_Op<"And_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX And_V1 operation [v1]";
  let description = [{
      Returns the tensor resulted from performing the `and` logical operation
  elementwise on the input tensors `A` and `B`.
  }];

  let arguments = (ins TensorOf<[I1]>:$A,
                       TensorOf<[I1]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// And [v7]
def Onnx_AndOp : Onnx_Op<"And", [Pure, OPCountInfo]> {
  let summary = "ONNX And operation [v7]";
  let description = [{
      Returns the tensor resulted from performing the `and` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins TensorOf<[I1]>:$A,
                       TensorOf<[I1]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Flatten [v1]
def Onnx_Flatten_V1Op : Onnx_Op<"Flatten_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Flatten_V1 operation [v1]";
  let description = [{
      Flattens the input tensor into a 2D matrix.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Flatten [v9]
def Onnx_Flatten_V9Op : Onnx_Op<"Flatten_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Flatten_V9 operation [v9]";
  let description = [{
      Flattens the input tensor into a 2D matrix.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Flatten [v11]
def Onnx_Flatten_V11Op : Onnx_Op<"Flatten_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Flatten_V11 operation [v11]";
  let description = [{
      Flattens the input tensor into a 2D matrix.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Flatten [v13]
def Onnx_Flatten_V13Op : Onnx_Op<"Flatten_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Flatten_V13 operation [v13]";
  let description = [{
      Flattens the input tensor into a 2D matrix.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Flatten [v21]
def Onnx_Flatten_V21Op : Onnx_Op<"Flatten_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Flatten_V21 operation [v21]";
  let description = [{
      Flattens the input tensor into a 2D matrix.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Flatten [v23]
def Onnx_FlattenOp : Onnx_Op<"Flatten", [Pure, OPCountInfo]> {
  let summary = "ONNX Flatten operation [v23]";
  let description = [{
      Flattens the input tensor into a 2D matrix.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceLogSum [v1]
def Onnx_ReduceLogSum_V1Op : Onnx_Op<"ReduceLogSum_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceLogSum_V1 operation [v1]";
  let description = [{
      Computes the log sum of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceLogSum [v11]
def Onnx_ReduceLogSum_V11Op : Onnx_Op<"ReduceLogSum_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceLogSum_V11 operation [v11]";
  let description = [{
      Computes the log sum of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceLogSum [v13]
def Onnx_ReduceLogSum_V13Op : Onnx_Op<"ReduceLogSum_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceLogSum_V13 operation [v13]";
  let description = [{
      Computes the log sum of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceLogSum [v18]
def Onnx_ReduceLogSumOp : Onnx_Op<"ReduceLogSum", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceLogSum operation [v18]";
  let description = [{
      Computes the log sum of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Einsum [v12]
def Onnx_EinsumOp : Onnx_Op<"Einsum", [Pure, OPCountInfo]> {
  let summary = "ONNX Einsum operation [v12]";
  let description = [{
      An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation
  ```
  output[output-term] = reduce-sum( input1[term1] * input2[term2] )
  ```
  where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
  that do not occur in the output-term.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$Inputs,
                       StrAttr:$equation);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceLogSumExp [v1]
def Onnx_ReduceLogSumExp_V1Op : Onnx_Op<"ReduceLogSumExp_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceLogSumExp_V1 operation [v1]";
  let description = [{
      Computes the log sum exponent of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceLogSumExp [v11]
def Onnx_ReduceLogSumExp_V11Op : Onnx_Op<"ReduceLogSumExp_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceLogSumExp_V11 operation [v11]";
  let description = [{
      Computes the log sum exponent of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceLogSumExp [v13]
def Onnx_ReduceLogSumExp_V13Op : Onnx_Op<"ReduceLogSumExp_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceLogSumExp_V13 operation [v13]";
  let description = [{
      Computes the log sum exponent of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceLogSumExp [v18]
def Onnx_ReduceLogSumExpOp : Onnx_Op<"ReduceLogSumExp", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceLogSumExp operation [v18]";
  let description = [{
      Computes the log sum exponent of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sub [v1]
def Onnx_Sub_V1Op : Onnx_Op<"Sub_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Sub_V1 operation [v1]";
  let description = [{
      Performs element-wise binary subtraction (with limited broadcast support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sub [v6]
def Onnx_Sub_V6Op : Onnx_Op<"Sub_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Sub_V6 operation [v6]";
  let description = [{
      Performs element-wise binary subtraction (with limited broadcast support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sub [v7]
def Onnx_Sub_V7Op : Onnx_Op<"Sub_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Sub_V7 operation [v7]";
  let description = [{
      Performs element-wise binary subtraction (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sub [v13]
def Onnx_Sub_V13Op : Onnx_Op<"Sub_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Sub_V13 operation [v13]";
  let description = [{
      Performs element-wise binary subtraction (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sub [v14]
def Onnx_SubOp : Onnx_Op<"Sub", [Pure, OPCountInfo]> {
  let summary = "ONNX Sub operation [v14]";
  let description = [{
      Performs element-wise binary subtraction (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Floor [v1]
def Onnx_Floor_V1Op : Onnx_Op<"Floor_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Floor_V1 operation [v1]";
  let description = [{
      Floor takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the floor is, y = floor(x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Floor [v6]
def Onnx_Floor_V6Op : Onnx_Op<"Floor_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Floor_V6 operation [v6]";
  let description = [{
      Floor takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the floor is, y = floor(x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Floor [v13]
def Onnx_FloorOp : Onnx_Op<"Floor", [Pure, OPCountInfo]> {
  let summary = "ONNX Floor operation [v13]";
  let description = [{
      Floor takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the floor is, y = floor(x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RandomUniformLike [v1]
def Onnx_RandomUniformLike_V1Op : Onnx_Op<"RandomUniformLike_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX RandomUniformLike_V1 operation [v1]";
  let description = [{
      Generate a tensor with random values drawn from a uniform distribution.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       OptionalAttr<I64Attr>:$dtype,
                       DefaultValuedAttr<F32Attr, "1">:$high,
                       DefaultValuedAttr<F32Attr, "0">:$low,
                       OptionalAttr<F32Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RandomUniformLike [v22]
def Onnx_RandomUniformLikeOp : Onnx_Op<"RandomUniformLike", [Pure, OPCountInfo]> {
  let summary = "ONNX RandomUniformLike operation [v22]";
  let description = [{
      Generate a tensor with random values drawn from a uniform distribution.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       OptionalAttr<I64Attr>:$dtype,
                       DefaultValuedAttr<F32Attr, "1">:$high,
                       DefaultValuedAttr<F32Attr, "0">:$low,
                       OptionalAttr<F32Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MaxRoiPool [v1]
def Onnx_MaxRoiPool_V1Op : Onnx_Op<"MaxRoiPool_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxRoiPool_V1 operation [v1]";
  let description = [{
      ROI max pool consumes an input tensor X and region of interests (RoIs) to
  apply max pooling across each RoI, to produce output 4-D tensor of shape
  (num_rois, channels, pooled_shape[0], pooled_shape[1]).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$rois,
                       I64ArrayAttr:$pooled_shape,
                       DefaultValuedAttr<F32Attr, "1">:$spatial_scale);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MaxRoiPool [v22]
def Onnx_MaxRoiPoolOp : Onnx_Op<"MaxRoiPool", [Pure, OPCountInfo]> {
  let summary = "ONNX MaxRoiPool operation [v22]";
  let description = [{
      ROI max pool consumes an input tensor X and region of interests (RoIs) to
  apply max pooling across each RoI, to produce output 4-D tensor of shape
  (num_rois, channels, pooled_shape[0], pooled_shape[1]).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$rois,
                       I64ArrayAttr:$pooled_shape,
                       DefaultValuedAttr<F32Attr, "1">:$spatial_scale);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Concat [v1]
def Onnx_Concat_V1Op : Onnx_Op<"Concat_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Concat_V1 operation [v1]";
  let description = [{
      Concatenate a list of tensors into a single tensor.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$inputs,
                       OptionalAttr<I64Attr>:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$concat_result);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Concat [v4]
def Onnx_Concat_V4Op : Onnx_Op<"Concat_V4", [Pure, OPCountInfo]> {
  let summary = "ONNX Concat_V4 operation [v4]";
  let description = [{
      Concatenate a list of tensors into a single tensor.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$inputs,
                       I64Attr:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$concat_result);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Concat [v11]
def Onnx_Concat_V11Op : Onnx_Op<"Concat_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Concat_V11 operation [v11]";
  let description = [{
      Concatenate a list of tensors into a single tensor.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$inputs,
                       I64Attr:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$concat_result);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Concat [v13]
def Onnx_ConcatOp : Onnx_Op<"Concat", [Pure, OPCountInfo]> {
  let summary = "ONNX Concat operation [v13]";
  let description = [{
      Concatenate a list of tensors into a single tensor.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$inputs,
                       I64Attr:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$concat_result);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sigmoid [v1]
def Onnx_Sigmoid_V1Op : Onnx_Op<"Sigmoid_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Sigmoid_V1 operation [v1]";
  let description = [{
      Sigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
  tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sigmoid [v6]
def Onnx_Sigmoid_V6Op : Onnx_Op<"Sigmoid_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Sigmoid_V6 operation [v6]";
  let description = [{
      Sigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
  tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sigmoid [v13]
def Onnx_SigmoidOp : Onnx_Op<"Sigmoid", [Pure, OPCountInfo]> {
  let summary = "ONNX Sigmoid operation [v13]";
  let description = [{
      Sigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
  tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SequenceInsert [v11]
def Onnx_SequenceInsertOp : Onnx_Op<"SequenceInsert", [Pure, OPCountInfo]> {
  let summary = "ONNX SequenceInsert operation [v11]";
  let description = [{
      Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
  }];

  let arguments = (ins AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$input_sequence,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$tensor,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$position);
  let results = (outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$output_sequence);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Less [v1]
def Onnx_Less_V1Op : Onnx_Op<"Less_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Less_V1 operation [v1]";
  let description = [{
      Returns the tensor resulted from performing the `less` logical operation
  elementwise on the input tensors `A` and `B`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Less [v7]
def Onnx_Less_V7Op : Onnx_Op<"Less_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Less_V7 operation [v7]";
  let description = [{
      Returns the tensor resulted from performing the `less` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Less [v9]
def Onnx_Less_V9Op : Onnx_Op<"Less_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Less_V9 operation [v9]";
  let description = [{
      Returns the tensor resulted from performing the `less` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Less [v13]
def Onnx_LessOp : Onnx_Op<"Less", [Pure, OPCountInfo]> {
  let summary = "ONNX Less operation [v13]";
  let description = [{
      Returns the tensor resulted from performing the `less` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Softmax [v1]
def Onnx_Softmax_V1Op : Onnx_Op<"Softmax_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Softmax_V1 operation [v1]";
  let description = [{
      The operator computes the softmax (normalized exponential) values for each layer in the batch
  of the given input.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Softmax [v11]
def Onnx_Softmax_V11Op : Onnx_Op<"Softmax_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Softmax_V11 operation [v11]";
  let description = [{
      The operator computes the softmax (normalized exponential) values for each layer in the batch
  of the given input.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Softmax [v13]
def Onnx_SoftmaxOp : Onnx_Op<"Softmax", [Pure, OPCountInfo]> {
  let summary = "ONNX Softmax operation [v13]";
  let description = [{
      The operator computes the normalized exponential values for the given input:
  Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1) 
  The "axis" attribute indicates the dimension along which Softmax
  will be performed.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input,
                       DefaultValuedAttr<I64Attr, "-1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Add [v1]
def Onnx_Add_V1Op : Onnx_Op<"Add_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Add_V1 operation [v1]";
  let description = [{
      Performs element-wise binary addition (with limited broadcast support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Add [v6]
def Onnx_Add_V6Op : Onnx_Op<"Add_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Add_V6 operation [v6]";
  let description = [{
      Performs element-wise binary addition (with limited broadcast support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Add [v7]
def Onnx_Add_V7Op : Onnx_Op<"Add_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Add_V7 operation [v7]";
  let description = [{
      Performs element-wise binary addition (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Add [v13]
def Onnx_Add_V13Op : Onnx_Op<"Add_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Add_V13 operation [v13]";
  let description = [{
      Performs element-wise binary addition (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Add [v14]
def Onnx_AddOp : Onnx_Op<"Add", [Pure, OPCountInfo]> {
  let summary = "ONNX Add operation [v14]";
  let description = [{
      Performs element-wise binary addition (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DynamicQuantizeLinear [v11]
def Onnx_DynamicQuantizeLinearOp : Onnx_Op<"DynamicQuantizeLinear", [Pure, OPCountInfo]> {
  let summary = "ONNX DynamicQuantizeLinear operation [v11]";
  let description = [{
      A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
  }];

  let arguments = (ins TensorOf<[F32]>:$x);
  let results = (outs TensorOf<[UI8]>:$y,
                      TensorOf<[F32]>:$y_scale,
                      TensorOf<[UI8]>:$y_zero_point);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 3;
    }
  }];
}

/// InstanceNormalization [v1]
def Onnx_InstanceNormalization_V1Op : Onnx_Op<"InstanceNormalization_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX InstanceNormalization_V1 operation [v1]";
  let description = [{
      Carries out instance normalization as described in the paper
  https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// InstanceNormalization [v6]
def Onnx_InstanceNormalization_V6Op : Onnx_Op<"InstanceNormalization_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX InstanceNormalization_V6 operation [v6]";
  let description = [{
      Carries out instance normalization as described in the paper
  https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// InstanceNormalization [v22]
def Onnx_InstanceNormalizationOp : Onnx_Op<"InstanceNormalization", [Pure, OPCountInfo]> {
  let summary = "ONNX InstanceNormalization operation [v22]";
  let description = [{
      Carries out instance normalization as described in the paper
  https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scale,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Loop [v1]
def Onnx_Loop_V1Op : Onnx_Op<"Loop_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Loop_V1 operation [v1]";
  let description = [{
      Generic Looping construct.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$M,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$cond,
                       Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$v_initial);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$v_final_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Loop [v11]
def Onnx_Loop_V11Op : Onnx_Op<"Loop_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Loop_V11 operation [v11]";
  let description = [{
      Generic Looping construct.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$M,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$cond,
                       Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$v_initial);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$v_final_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Loop [v13]
def Onnx_Loop_V13Op : Onnx_Op<"Loop_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Loop_V13 operation [v13]";
  let description = [{
      Generic Looping construct.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$M,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$cond,
                       Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>>:$v_initial);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>>:$v_final_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Loop [v16]
def Onnx_Loop_V16Op : Onnx_Op<"Loop_V16", [Pure, OPCountInfo]> {
  let summary = "ONNX Loop_V16 operation [v16]";
  let description = [{
      Generic Looping construct.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$M,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$cond,
                       Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>>:$v_initial);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>>:$v_final_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Loop [v19]
def Onnx_Loop_V19Op : Onnx_Op<"Loop_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX Loop_V19 operation [v19]";
  let description = [{
      Generic Looping construct.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$M,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$cond,
                       Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[F8E4M3FN]>]>, SeqOf<[TensorOf<[F8E4M3FNUZ]>]>, SeqOf<[TensorOf<[F8E5M2]>]>, SeqOf<[TensorOf<[F8E5M2FNUZ]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, OptOf<TensorOf<[F8E4M3FN]>>, OptOf<TensorOf<[F8E4M3FNUZ]>>, OptOf<TensorOf<[F8E5M2]>>, OptOf<TensorOf<[F8E5M2FNUZ]>>]>>:$v_initial);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[F8E4M3FN]>]>, SeqOf<[TensorOf<[F8E4M3FNUZ]>]>, SeqOf<[TensorOf<[F8E5M2]>]>, SeqOf<[TensorOf<[F8E5M2FNUZ]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, OptOf<TensorOf<[F8E4M3FN]>>, OptOf<TensorOf<[F8E4M3FNUZ]>>, OptOf<TensorOf<[F8E5M2]>>, OptOf<TensorOf<[F8E5M2FNUZ]>>]>>:$v_final_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Loop [v21]
def Onnx_Loop_V21Op : Onnx_Op<"Loop_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Loop_V21 operation [v21]";
  let description = [{
      Generic Looping construct.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$M,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$cond,
                       Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[F8E4M3FN]>]>, SeqOf<[TensorOf<[F8E4M3FNUZ]>]>, SeqOf<[TensorOf<[F8E5M2]>]>, SeqOf<[TensorOf<[F8E5M2FNUZ]>]>, SeqOf<[TensorOf<[UI<4>]>]>, SeqOf<[TensorOf<[I<4>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, OptOf<TensorOf<[F8E4M3FN]>>, OptOf<TensorOf<[F8E4M3FNUZ]>>, OptOf<TensorOf<[F8E5M2]>>, OptOf<TensorOf<[F8E5M2FNUZ]>>, OptOf<TensorOf<[UI<4>]>>, OptOf<TensorOf<[I<4>]>>]>>:$v_initial);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[F8E4M3FN]>]>, SeqOf<[TensorOf<[F8E4M3FNUZ]>]>, SeqOf<[TensorOf<[F8E5M2]>]>, SeqOf<[TensorOf<[F8E5M2FNUZ]>]>, SeqOf<[TensorOf<[UI<4>]>]>, SeqOf<[TensorOf<[I<4>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, OptOf<TensorOf<[F8E4M3FN]>>, OptOf<TensorOf<[F8E4M3FNUZ]>>, OptOf<TensorOf<[F8E5M2]>>, OptOf<TensorOf<[F8E5M2FNUZ]>>, OptOf<TensorOf<[UI<4>]>>, OptOf<TensorOf<[I<4>]>>]>>:$v_final_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Loop [v23]
def Onnx_LoopOp : Onnx_Op<"Loop", [Pure, OPCountInfo]> {
  let summary = "ONNX Loop operation [v23]";
  let description = [{
      Generic Looping construct.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$M,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$cond,
                       Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[F8E4M3FN]>]>, SeqOf<[TensorOf<[F8E4M3FNUZ]>]>, SeqOf<[TensorOf<[F8E5M2]>]>, SeqOf<[TensorOf<[F8E5M2FNUZ]>]>, SeqOf<[TensorOf<[UI<4>]>]>, SeqOf<[TensorOf<[I<4>]>]>, SeqOf<[TensorOf<[F4E2M1FN]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, OptOf<TensorOf<[F8E4M3FN]>>, OptOf<TensorOf<[F8E4M3FNUZ]>>, OptOf<TensorOf<[F8E5M2]>>, OptOf<TensorOf<[F8E5M2FNUZ]>>, OptOf<TensorOf<[UI<4>]>>, OptOf<TensorOf<[I<4>]>>, OptOf<TensorOf<[F4E2M1FN]>>]>>:$v_initial);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[BF16]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, SeqOf<[TensorOf<[F8E4M3FN]>]>, SeqOf<[TensorOf<[F8E4M3FNUZ]>]>, SeqOf<[TensorOf<[F8E5M2]>]>, SeqOf<[TensorOf<[F8E5M2FNUZ]>]>, SeqOf<[TensorOf<[UI<4>]>]>, SeqOf<[TensorOf<[I<4>]>]>, SeqOf<[TensorOf<[F4E2M1FN]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[BF16]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[BF16]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>, OptOf<TensorOf<[F8E4M3FN]>>, OptOf<TensorOf<[F8E4M3FNUZ]>>, OptOf<TensorOf<[F8E5M2]>>, OptOf<TensorOf<[F8E5M2FNUZ]>>, OptOf<TensorOf<[UI<4>]>>, OptOf<TensorOf<[I<4>]>>, OptOf<TensorOf<[F4E2M1FN]>>]>>:$v_final_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LpPool [v1]
def Onnx_LpPool_V1Op : Onnx_Op<"LpPool_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX LpPool_V1 operation [v1]";
  let description = [{
      LpPool consumes an input tensor X and applies Lp pooling across the
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       OptionalAttr<I64ArrayAttr>:$kernel_shape,
                       DefaultValuedAttr<F32Attr, "2">:$p,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LpPool [v2]
def Onnx_LpPool_V2Op : Onnx_Op<"LpPool_V2", [Pure, OPCountInfo]> {
  let summary = "ONNX LpPool_V2 operation [v2]";
  let description = [{
      LpPool consumes an input tensor X and applies Lp pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       I64ArrayAttr:$kernel_shape,
                       DefaultValuedAttr<I64Attr, "2">:$p,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LpPool [v11]
def Onnx_LpPool_V11Op : Onnx_Op<"LpPool_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX LpPool_V11 operation [v11]";
  let description = [{
      LpPool consumes an input tensor X and applies Lp pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       I64ArrayAttr:$kernel_shape,
                       DefaultValuedAttr<I64Attr, "2">:$p,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LpPool [v18]
def Onnx_LpPool_V18Op : Onnx_Op<"LpPool_V18", [Pure, OPCountInfo]> {
  let summary = "ONNX LpPool_V18 operation [v18]";
  let description = [{
      LpPool consumes an input tensor X and applies Lp pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       I64ArrayAttr:$kernel_shape,
                       DefaultValuedAttr<I64Attr, "2">:$p,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LpPool [v22]
def Onnx_LpPoolOp : Onnx_Op<"LpPool", [Pure, OPCountInfo]> {
  let summary = "ONNX LpPool operation [v22]";
  let description = [{
      LpPool consumes an input tensor X and applies Lp pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       I64ArrayAttr:$kernel_shape,
                       DefaultValuedAttr<I64Attr, "2">:$p,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ArgMin [v1]
def Onnx_ArgMin_V1Op : Onnx_Op<"ArgMin_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ArgMin_V1 operation [v1]";
  let description = [{
      Computes the indices of the min elements of the input tensor's element along the
  provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs TensorOf<[I64]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ArgMin [v11]
def Onnx_ArgMin_V11Op : Onnx_Op<"ArgMin_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ArgMin_V11 operation [v11]";
  let description = [{
      Computes the indices of the min elements of the input tensor's element along the
  provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs TensorOf<[I64]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ArgMin [v12]
def Onnx_ArgMin_V12Op : Onnx_Op<"ArgMin_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX ArgMin_V12 operation [v12]";
  let description = [{
      Computes the indices of the min elements of the input tensor's element along the
  provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$select_last_index);
  let results = (outs TensorOf<[I64]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ArgMin [v13]
def Onnx_ArgMinOp : Onnx_Op<"ArgMin", [Pure, OPCountInfo]> {
  let summary = "ONNX ArgMin operation [v13]";
  let description = [{
      Computes the indices of the min elements of the input tensor's element along the
  provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$select_last_index);
  let results = (outs TensorOf<[I64]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Equal [v1]
def Onnx_Equal_V1Op : Onnx_Op<"Equal_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Equal_V1 operation [v1]";
  let description = [{
      Returns the tensor resulted from performing the `equal` logical operation
  elementwise on the input tensors `A` and `B`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I1]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
                       AnyTypeOf<[TensorOf<[I1]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Equal [v7]
def Onnx_Equal_V7Op : Onnx_Op<"Equal_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Equal_V7 operation [v7]";
  let description = [{
      Returns the tensor resulted from performing the `equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I1]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
                       AnyTypeOf<[TensorOf<[I1]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Equal [v11]
def Onnx_Equal_V11Op : Onnx_Op<"Equal_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Equal_V11 operation [v11]";
  let description = [{
      Returns the tensor resulted from performing the `equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Equal [v13]
def Onnx_Equal_V13Op : Onnx_Op<"Equal_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Equal_V13 operation [v13]";
  let description = [{
      Returns the tensor resulted from performing the `equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Equal [v19]
def Onnx_EqualOp : Onnx_Op<"Equal", [Pure, OPCountInfo]> {
  let summary = "ONNX Equal operation [v19]";
  let description = [{
      Returns the tensor resulted from performing the `equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[StringType]>]>:$A,
                       AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[StringType]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SplitToSequence [v11]
def Onnx_SplitToSequenceOp : Onnx_Op<"SplitToSequence", [Pure, OPCountInfo]> {
  let summary = "ONNX SplitToSequence operation [v11]";
  let description = [{
      Split a tensor into a sequence of tensors, along the specified 'axis'.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$split,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$output_sequence);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Round [v11]
def Onnx_Round_V11Op : Onnx_Op<"Round_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Round_V11 operation [v11]";
  let description = [{
      Round takes one input Tensor and rounds the values, element-wise, meaning
  it finds the nearest integer for each value.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Round [v22]
def Onnx_RoundOp : Onnx_Op<"Round", [Pure, OPCountInfo]> {
  let summary = "ONNX Round operation [v22]";
  let description = [{
      Round takes one input Tensor and rounds the values, element-wise, meaning
  it finds the nearest integer for each value.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Unique [v11]
def Onnx_UniqueOp : Onnx_Op<"Unique", [Pure, OPCountInfo]> {
  let summary = "ONNX Unique operation [v11]";
  let description = [{
      Find the unique elements of a tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$sorted);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y,
                      AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$indices,
                      AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$inverse_indices,
                      AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$counts);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 4;
    }
  }];
}

/// BitShift [v11]
def Onnx_BitShiftOp : Onnx_Op<"BitShift", [Pure, OPCountInfo]> {
  let summary = "ONNX BitShift operation [v11]";
  let description = [{
      Bitwise shift operator performs element-wise operation.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>]>:$X,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>]>:$Y,
                       StrAttr:$direction);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>]>:$Z);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// AveragePool [v1]
def Onnx_AveragePool_V1Op : Onnx_Op<"AveragePool_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX AveragePool_V1 operation [v1]";
  let description = [{
      AveragePool consumes an input tensor X and applies average pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// AveragePool [v7]
def Onnx_AveragePool_V7Op : Onnx_Op<"AveragePool_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX AveragePool_V7 operation [v7]";
  let description = [{
      AveragePool consumes an input tensor X and applies average pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$count_include_pad,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// AveragePool [v10]
def Onnx_AveragePool_V10Op : Onnx_Op<"AveragePool_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX AveragePool_V10 operation [v10]";
  let description = [{
      AveragePool consumes an input tensor X and applies average pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       DefaultValuedAttr<I64Attr, "0">:$count_include_pad,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// AveragePool [v11]
def Onnx_AveragePool_V11Op : Onnx_Op<"AveragePool_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX AveragePool_V11 operation [v11]";
  let description = [{
      AveragePool consumes an input tensor X and applies average pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       DefaultValuedAttr<I64Attr, "0">:$count_include_pad,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// AveragePool [v19]
def Onnx_AveragePool_V19Op : Onnx_Op<"AveragePool_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX AveragePool_V19 operation [v19]";
  let description = [{
      AveragePool consumes an input tensor X and applies average pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       DefaultValuedAttr<I64Attr, "0">:$count_include_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// AveragePool [v22]
def Onnx_AveragePoolOp : Onnx_Op<"AveragePool", [Pure, OPCountInfo]> {
  let summary = "ONNX AveragePool operation [v22]";
  let description = [{
      AveragePool consumes an input tensor X and applies average pooling across
  the tensor according to kernel sizes, stride sizes, and pad lengths.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
                       DefaultValuedAttr<I64Attr, "0">:$count_include_pad,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       I64ArrayAttr:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Slice [v1]
def Onnx_Slice_V1Op : Onnx_Op<"Slice_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Slice_V1 operation [v1]";
  let description = [{
      Produces a slice of the input tensor along multiple axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       I64ArrayAttr:$ends,
                       I64ArrayAttr:$starts);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Slice [v10]
def Onnx_Slice_V10Op : Onnx_Op<"Slice_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX Slice_V10 operation [v10]";
  let description = [{
      Produces a slice of the input tensor along multiple axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$starts,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$ends,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$steps);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Slice [v11]
def Onnx_Slice_V11Op : Onnx_Op<"Slice_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Slice_V11 operation [v11]";
  let description = [{
      Produces a slice of the input tensor along multiple axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$starts,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$ends,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$steps);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Slice [v13]
def Onnx_SliceOp : Onnx_Op<"Slice", [Pure, OPCountInfo]> {
  let summary = "ONNX Slice operation [v13]";
  let description = [{
      Produces a slice of the input tensor along multiple axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$starts,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$ends,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$steps);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Dropout [v1]
def Onnx_Dropout_V1Op : Onnx_Op<"Dropout_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Dropout_V1 operation [v1]";
  let description = [{
      Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
  output (Tensor<float>) and mask (Tensor<bool>).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs,
                       DefaultValuedAttr<I64Attr, "0">:$is_test,
                       DefaultValuedAttr<F32Attr, "0.5">:$ratio);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$mask);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// Dropout [v6]
def Onnx_Dropout_V6Op : Onnx_Op<"Dropout_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Dropout_V6 operation [v6]";
  let description = [{
      Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
  output (Tensor<float>) and mask (Tensor<bool>).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedAttr<I64Attr, "0">:$is_test,
                       DefaultValuedAttr<F32Attr, "0.5">:$ratio);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$mask);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// Dropout [v7]
def Onnx_Dropout_V7Op : Onnx_Op<"Dropout_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Dropout_V7 operation [v7]";
  let description = [{
      Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
  output (Tensor<float>) and mask (Tensor<bool>).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedAttr<F32Attr, "0.5">:$ratio);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$mask);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// Dropout [v10]
def Onnx_Dropout_V10Op : Onnx_Op<"Dropout_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX Dropout_V10 operation [v10]";
  let description = [{
      Dropout takes one input floating tensor and produces two tensor outputs,
  output (floating tensor) and mask (`Tensor<bool>`).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       DefaultValuedAttr<F32Attr, "0.5">:$ratio);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output,
                      AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$mask);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// Dropout [v12]
def Onnx_Dropout_V12Op : Onnx_Op<"Dropout_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX Dropout_V12 operation [v12]";
  let description = [{
      Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$ratio,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$training_mode,
                       OptionalAttr<I64Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output,
                      AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$mask);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// Dropout [v13]
def Onnx_Dropout_V13Op : Onnx_Op<"Dropout_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Dropout_V13 operation [v13]";
  let description = [{
      Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$ratio,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$training_mode,
                       OptionalAttr<I64Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output,
                      AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$mask);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// Dropout [v22]
def Onnx_DropoutOp : Onnx_Op<"Dropout", [Pure, OPCountInfo]> {
  let summary = "ONNX Dropout operation [v22]";
  let description = [{
      Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$data,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$ratio,
                       AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$training_mode,
                       OptionalAttr<I64Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output,
                      AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$mask);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// Exp [v1]
def Onnx_Exp_V1Op : Onnx_Op<"Exp_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Exp_V1 operation [v1]";
  let description = [{
      Calculates the exponential of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Exp [v6]
def Onnx_Exp_V6Op : Onnx_Op<"Exp_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Exp_V6 operation [v6]";
  let description = [{
      Calculates the exponential of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Exp [v13]
def Onnx_ExpOp : Onnx_Op<"Exp", [Pure, OPCountInfo]> {
  let summary = "ONNX Exp operation [v13]";
  let description = [{
      Calculates the exponential of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// EyeLike [v9]
def Onnx_EyeLike_V9Op : Onnx_Op<"EyeLike_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX EyeLike_V9 operation [v9]";
  let description = [{
      Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>:$input,
                       OptionalAttr<I64Attr>:$dtype,
                       DefaultValuedAttr<I64Attr, "0">:$k);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// EyeLike [v22]
def Onnx_EyeLikeOp : Onnx_Op<"EyeLike", [Pure, OPCountInfo]> {
  let summary = "ONNX EyeLike operation [v22]";
  let description = [{
      Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
                       OptionalAttr<I64Attr>:$dtype,
                       DefaultValuedAttr<I64Attr, "0">:$k);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LSTM [v1]
def Onnx_LSTM_V1Op : Onnx_Op<"LSTM_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX LSTM_V1 operation [v1]";
  let description = [{
      Computes an one-layer LSTM.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_c,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$P,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       OptionalAttr<StrArrayAttr>:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$input_forget,
                       DefaultValuedAttr<I64Attr, "0">:$output_sequence);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_c);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 8;
    }
    int getDefinedResultCount() {
      return 3;
    }
  }];
}

/// LSTM [v7]
def Onnx_LSTM_V7Op : Onnx_Op<"LSTM_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX LSTM_V7 operation [v7]";
  let description = [{
      Computes an one-layer LSTM.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_c,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$P,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       OptionalAttr<StrArrayAttr>:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$input_forget);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_c);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 8;
    }
    int getDefinedResultCount() {
      return 3;
    }
  }];
}

/// LSTM [v14]
def Onnx_LSTM_V14Op : Onnx_Op<"LSTM_V14", [Pure, OPCountInfo]> {
  let summary = "ONNX LSTM_V14 operation [v14]";
  let description = [{
      Computes an one-layer LSTM.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_c,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$P,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       OptionalAttr<StrArrayAttr>:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$input_forget,
                       DefaultValuedAttr<I64Attr, "0">:$layout);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_c);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 8;
    }
    int getDefinedResultCount() {
      return 3;
    }
  }];
}

/// LSTM [v22]
def Onnx_LSTMOp : Onnx_Op<"LSTM", [Pure, OPCountInfo]> {
  let summary = "ONNX LSTM operation [v22]";
  let description = [{
      Computes an one-layer LSTM.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_c,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$P,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       OptionalAttr<StrArrayAttr>:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$input_forget,
                       DefaultValuedAttr<I64Attr, "0">:$layout);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h,
                      AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_c);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 8;
    }
    int getDefinedResultCount() {
      return 3;
    }
  }];
}

/// MatMul [v1]
def Onnx_MatMul_V1Op : Onnx_Op<"MatMul_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX MatMul_V1 operation [v1]";
  let description = [{
      Matrix product that behaves like [numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MatMul [v9]
def Onnx_MatMul_V9Op : Onnx_Op<"MatMul_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX MatMul_V9 operation [v9]";
  let description = [{
      Matrix product that behaves like [numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MatMul [v13]
def Onnx_MatMulOp : Onnx_Op<"MatMul", [Pure, OPCountInfo]> {
  let summary = "ONNX MatMul operation [v13]";
  let description = [{
      Matrix product that behaves like [numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LeakyRelu [v1]
def Onnx_LeakyRelu_V1Op : Onnx_Op<"LeakyRelu_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX LeakyRelu_V1 operation [v1]";
  let description = [{
      LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "0.01">:$alpha,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LeakyRelu [v6]
def Onnx_LeakyRelu_V6Op : Onnx_Op<"LeakyRelu_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX LeakyRelu_V6 operation [v6]";
  let description = [{
      LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "0.01">:$alpha);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LeakyRelu [v16]
def Onnx_LeakyReluOp : Onnx_Op<"LeakyRelu", [Pure, OPCountInfo]> {
  let summary = "ONNX LeakyRelu operation [v16]";
  let description = [{
      LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "0.01">:$alpha);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMean [v1]
def Onnx_ReduceMean_V1Op : Onnx_Op<"ReduceMean_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMean_V1 operation [v1]";
  let description = [{
      Computes the mean of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMean [v11]
def Onnx_ReduceMean_V11Op : Onnx_Op<"ReduceMean_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMean_V11 operation [v11]";
  let description = [{
      Computes the mean of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMean [v13]
def Onnx_ReduceMean_V13Op : Onnx_Op<"ReduceMean_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMean_V13 operation [v13]";
  let description = [{
      Computes the mean of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceMean [v18]
def Onnx_ReduceMeanOp : Onnx_Op<"ReduceMean", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceMean operation [v18]";
  let description = [{
      Computes the mean of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReverseSequence [v10]
def Onnx_ReverseSequenceOp : Onnx_Op<"ReverseSequence", [Pure, OPCountInfo]> {
  let summary = "ONNX ReverseSequence operation [v10]";
  let description = [{
      Reverse batch of sequences having different lengths specified by `sequence_lens`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       TensorOf<[I64]>:$sequence_lens,
                       DefaultValuedAttr<I64Attr, "1">:$batch_axis,
                       DefaultValuedAttr<I64Attr, "0">:$time_axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// BatchNormalization [v1]
def Onnx_BatchNormalization_V1Op : Onnx_Op<"BatchNormalization_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX BatchNormalization_V1 operation [v1]";
  let description = [{
      Carries out batch normalization as described in the paper
  https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$mean,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$var,
                       I64ArrayAttr:$consumed_inputs,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       DefaultValuedAttr<I64Attr, "0">:$is_test,
                       DefaultValuedAttr<F32Attr, "0.9">:$momentum,
                       DefaultValuedAttr<I64Attr, "1">:$spatial);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output_var,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$saved_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$saved_var);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 5;
    }
  }];
}

/// BatchNormalization [v6]
def Onnx_BatchNormalization_V6Op : Onnx_Op<"BatchNormalization_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX BatchNormalization_V6 operation [v6]";
  let description = [{
      Carries out batch normalization as described in the paper
  https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$mean,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$var,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       DefaultValuedAttr<I64Attr, "0">:$is_test,
                       DefaultValuedAttr<F32Attr, "0.9">:$momentum,
                       DefaultValuedAttr<I64Attr, "1">:$spatial);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output_var,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$saved_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$saved_var);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 5;
    }
  }];
}

/// BatchNormalization [v7]
def Onnx_BatchNormalization_V7Op : Onnx_Op<"BatchNormalization_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX BatchNormalization_V7 operation [v7]";
  let description = [{
      Carries out batch normalization as described in the paper
  https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$mean,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$var,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       DefaultValuedAttr<F32Attr, "0.9">:$momentum,
                       DefaultValuedAttr<I64Attr, "1">:$spatial);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output_var,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$saved_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$saved_var);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 5;
    }
  }];
}

/// BatchNormalization [v9]
def Onnx_BatchNormalization_V9Op : Onnx_Op<"BatchNormalization_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX BatchNormalization_V9 operation [v9]";
  let description = [{
      Carries out batch normalization as described in the paper
  https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$mean,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$var,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       DefaultValuedAttr<F32Attr, "0.9">:$momentum);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output_var,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$saved_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$saved_var);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 5;
    }
  }];
}

/// BatchNormalization [v14]
def Onnx_BatchNormalization_V14Op : Onnx_Op<"BatchNormalization_V14", [Pure, OPCountInfo]> {
  let summary = "ONNX BatchNormalization_V14 operation [v14]";
  let description = [{
      Carries out batch normalization as described in the paper
  https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input_mean,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input_var,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       DefaultValuedAttr<F32Attr, "0.9">:$momentum,
                       DefaultValuedAttr<I64Attr, "0">:$training_mode);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$running_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$running_var);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 3;
    }
  }];
}

/// BatchNormalization [v15]
def Onnx_BatchNormalizationOp : Onnx_Op<"BatchNormalization", [Pure, OPCountInfo]> {
  let summary = "ONNX BatchNormalization operation [v15]";
  let description = [{
      Carries out batch normalization as described in the paper
  https://arxiv.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$scale,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input_mean,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input_var,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       DefaultValuedAttr<F32Attr, "0.9">:$momentum,
                       DefaultValuedAttr<I64Attr, "0">:$training_mode);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$running_mean,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$running_var);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 5;
    }
    int getDefinedResultCount() {
      return 3;
    }
  }];
}

/// LpNormalization [v1]
def Onnx_LpNormalization_V1Op : Onnx_Op<"LpNormalization_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX LpNormalization_V1 operation [v1]";
  let description = [{
      Given a matrix, apply Lp-normalization along the provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "-1">:$axis,
                       DefaultValuedAttr<I64Attr, "2">:$p);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LpNormalization [v22]
def Onnx_LpNormalizationOp : Onnx_Op<"LpNormalization", [Pure, OPCountInfo]> {
  let summary = "ONNX LpNormalization operation [v22]";
  let description = [{
      Given a matrix, apply Lp-normalization along the provided axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "-1">:$axis,
                       DefaultValuedAttr<I64Attr, "2">:$p);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gemm [v1]
def Onnx_Gemm_V1Op : Onnx_Op<"Gemm_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Gemm_V1 operation [v1]";
  let description = [{
      General Matrix multiplication:
  https://en.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C,
                       DefaultValuedAttr<F32Attr, "1">:$alpha,
                       DefaultValuedAttr<F32Attr, "1">:$beta,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast,
                       DefaultValuedAttr<I64Attr, "0">:$transA,
                       DefaultValuedAttr<I64Attr, "0">:$transB);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gemm [v6]
def Onnx_Gemm_V6Op : Onnx_Op<"Gemm_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Gemm_V6 operation [v6]";
  let description = [{
      General Matrix multiplication:
  https://en.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C,
                       DefaultValuedAttr<F32Attr, "1">:$alpha,
                       DefaultValuedAttr<F32Attr, "1">:$beta,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast,
                       DefaultValuedAttr<I64Attr, "0">:$transA,
                       DefaultValuedAttr<I64Attr, "0">:$transB);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gemm [v7]
def Onnx_Gemm_V7Op : Onnx_Op<"Gemm_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Gemm_V7 operation [v7]";
  let description = [{
      General Matrix multiplication:
  https://en.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C,
                       DefaultValuedAttr<F32Attr, "1">:$alpha,
                       DefaultValuedAttr<F32Attr, "1">:$beta,
                       DefaultValuedAttr<I64Attr, "0">:$transA,
                       DefaultValuedAttr<I64Attr, "0">:$transB);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gemm [v9]
def Onnx_Gemm_V9Op : Onnx_Op<"Gemm_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Gemm_V9 operation [v9]";
  let description = [{
      General Matrix multiplication:
  https://en.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$C,
                       DefaultValuedAttr<F32Attr, "1">:$alpha,
                       DefaultValuedAttr<F32Attr, "1">:$beta,
                       DefaultValuedAttr<I64Attr, "0">:$transA,
                       DefaultValuedAttr<I64Attr, "0">:$transB);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gemm [v11]
def Onnx_Gemm_V11Op : Onnx_Op<"Gemm_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Gemm_V11 operation [v11]";
  let description = [{
      General Matrix multiplication:
  https://en.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$C,
                       DefaultValuedAttr<F32Attr, "1">:$alpha,
                       DefaultValuedAttr<F32Attr, "1">:$beta,
                       DefaultValuedAttr<I64Attr, "0">:$transA,
                       DefaultValuedAttr<I64Attr, "0">:$transB);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gemm [v13]
def Onnx_GemmOp : Onnx_Op<"Gemm", [Pure, OPCountInfo]> {
  let summary = "ONNX Gemm operation [v13]";
  let description = [{
      General Matrix multiplication:
  https://en.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$B,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, NoneType]>:$C,
                       DefaultValuedAttr<F32Attr, "1">:$alpha,
                       DefaultValuedAttr<F32Attr, "1">:$beta,
                       DefaultValuedAttr<I64Attr, "0">:$transA,
                       DefaultValuedAttr<I64Attr, "0">:$transB);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// OneHot [v9]
def Onnx_OneHot_V9Op : Onnx_Op<"OneHot_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX OneHot_V9 operation [v9]";
  let description = [{
      Produces a one-hot tensor based on inputs.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$depth,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$values,
                       DefaultValuedAttr<I64Attr, "-1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// OneHot [v11]
def Onnx_OneHotOp : Onnx_Op<"OneHot", [Pure, OPCountInfo]> {
  let summary = "ONNX OneHot operation [v11]";
  let description = [{
      Produces a one-hot tensor based on inputs.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$depth,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$values,
                       DefaultValuedAttr<I64Attr, "-1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Bernoulli [v15]
def Onnx_Bernoulli_V15Op : Onnx_Op<"Bernoulli_V15", [Pure, OPCountInfo]> {
  let summary = "ONNX Bernoulli_V15 operation [v15]";
  let description = [{
      Draws binary random numbers (0 or 1) from a Bernoulli distribution.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       OptionalAttr<I64Attr>:$dtype,
                       OptionalAttr<F32Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[I1]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Bernoulli [v22]
def Onnx_BernoulliOp : Onnx_Op<"Bernoulli", [Pure, OPCountInfo]> {
  let summary = "ONNX Bernoulli operation [v22]";
  let description = [{
      Draws binary random numbers (0 or 1) from a Bernoulli distribution.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       OptionalAttr<I64Attr>:$dtype,
                       OptionalAttr<F32Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// NegativeLogLikelihoodLoss [v12]
def Onnx_NegativeLogLikelihoodLoss_V12Op : Onnx_Op<"NegativeLogLikelihoodLoss_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX NegativeLogLikelihoodLoss_V12 operation [v12]";
  let description = [{
      A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$target,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$weight,
                       OptionalAttr<I64Attr>:$ignore_index,
                       DefaultValuedStrAttr<StrAttr, "mean">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$loss);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// NegativeLogLikelihoodLoss [v13]
def Onnx_NegativeLogLikelihoodLoss_V13Op : Onnx_Op<"NegativeLogLikelihoodLoss_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX NegativeLogLikelihoodLoss_V13 operation [v13]";
  let description = [{
      A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$target,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$weight,
                       OptionalAttr<I64Attr>:$ignore_index,
                       DefaultValuedStrAttr<StrAttr, "mean">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$loss);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// NegativeLogLikelihoodLoss [v22]
def Onnx_NegativeLogLikelihoodLossOp : Onnx_Op<"NegativeLogLikelihoodLoss", [Pure, OPCountInfo]> {
  let summary = "ONNX NegativeLogLikelihoodLoss operation [v22]";
  let description = [{
      A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$target,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$weight,
                       OptionalAttr<I64Attr>:$ignore_index,
                       DefaultValuedStrAttr<StrAttr, "mean">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$loss);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Greater [v1]
def Onnx_Greater_V1Op : Onnx_Op<"Greater_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Greater_V1 operation [v1]";
  let description = [{
      Returns the tensor resulted from performing the `greater` logical operation
  elementwise on the input tensors `A` and `B`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Greater [v7]
def Onnx_Greater_V7Op : Onnx_Op<"Greater_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Greater_V7 operation [v7]";
  let description = [{
      Returns the tensor resulted from performing the `greater` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Greater [v9]
def Onnx_Greater_V9Op : Onnx_Op<"Greater_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Greater_V9 operation [v9]";
  let description = [{
      Returns the tensor resulted from performing the `greater` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Greater [v13]
def Onnx_GreaterOp : Onnx_Op<"Greater", [Pure, OPCountInfo]> {
  let summary = "ONNX Greater operation [v13]";
  let description = [{
      Returns the tensor resulted from performing the `greater` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RNN [v1]
def Onnx_RNN_V1Op : Onnx_Op<"RNN_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX RNN_V1 operation [v1]";
  let description = [{
      Computes an one-layer simple RNN.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       DefaultValuedAttr<StrArrayAttr, "{\"Tanh\",\"Tanh\"}">:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$output_sequence);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// RNN [v7]
def Onnx_RNN_V7Op : Onnx_Op<"RNN_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX RNN_V7 operation [v7]";
  let description = [{
      Computes an one-layer simple RNN.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       DefaultValuedAttr<StrArrayAttr, "{\"Tanh\",\"Tanh\"}">:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// RNN [v14]
def Onnx_RNN_V14Op : Onnx_Op<"RNN_V14", [Pure, OPCountInfo]> {
  let summary = "ONNX RNN_V14 operation [v14]";
  let description = [{
      Computes an one-layer simple RNN.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       DefaultValuedAttr<StrArrayAttr, "{\"Tanh\",\"Tanh\"}">:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$layout);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// RNN [v22]
def Onnx_RNNOp : Onnx_Op<"RNN", [Pure, OPCountInfo]> {
  let summary = "ONNX RNN operation [v22]";
  let description = [{
      Computes an one-layer simple RNN.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
                       AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
                       AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h,
                       OptionalAttr<F32ArrayAttr>:$activation_alpha,
                       OptionalAttr<F32ArrayAttr>:$activation_beta,
                       DefaultValuedAttr<StrArrayAttr, "{\"Tanh\",\"Tanh\"}">:$activations,
                       OptionalAttr<F32Attr>:$clip,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       OptionalAttr<I64Attr>:$hidden_size,
                       DefaultValuedAttr<I64Attr, "0">:$layout);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
                      AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y_h);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 6;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// GlobalLpPool [v1]
def Onnx_GlobalLpPool_V1Op : Onnx_Op<"GlobalLpPool_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX GlobalLpPool_V1 operation [v1]";
  let description = [{
      GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
  the values in the same channel.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "2">:$p);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GlobalLpPool [v2]
def Onnx_GlobalLpPool_V2Op : Onnx_Op<"GlobalLpPool_V2", [Pure, OPCountInfo]> {
  let summary = "ONNX GlobalLpPool_V2 operation [v2]";
  let description = [{
      GlobalLpPool consumes an input tensor X and applies lp pool pooling across
  the values in the same channel.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<I64Attr, "2">:$p);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GlobalLpPool [v22]
def Onnx_GlobalLpPoolOp : Onnx_Op<"GlobalLpPool", [Pure, OPCountInfo]> {
  let summary = "ONNX GlobalLpPool operation [v22]";
  let description = [{
      GlobalLpPool consumes an input tensor X and applies lp pool pooling across
  the values in the same channel.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<I64Attr, "2">:$p);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gather [v1]
def Onnx_Gather_V1Op : Onnx_Op<"Gather_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Gather_V1 operation [v1]";
  let description = [{
      Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
  entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
  them in an output tensor of rank q + (r - 1).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gather [v11]
def Onnx_Gather_V11Op : Onnx_Op<"Gather_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Gather_V11 operation [v11]";
  let description = [{
      Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
  entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
  them in an output tensor of rank q + (r - 1).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Gather [v13]
def Onnx_GatherOp : Onnx_Op<"Gather", [Pure, OPCountInfo]> {
  let summary = "ONNX Gather operation [v13]";
  let description = [{
      Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
  entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
  them in an output tensor of rank q + (r - 1).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// HardSwish [v14]
def Onnx_HardSwish_V14Op : Onnx_Op<"HardSwish_V14", [Pure, OPCountInfo]> {
  let summary = "ONNX HardSwish_V14 operation [v14]";
  let description = [{
      HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
  the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
  where alpha = 1/6 and beta = 0.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// HardSwish [v22]
def Onnx_HardSwishOp : Onnx_Op<"HardSwish", [Pure, OPCountInfo]> {
  let summary = "ONNX HardSwish operation [v22]";
  let description = [{
      HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
  the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
  where alpha = 1/6 and beta = 0.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mean [v1]
def Onnx_Mean_V1Op : Onnx_Op<"Mean_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Mean_V1 operation [v1]";
  let description = [{
      Element-wise mean of each of the input tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$mean);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mean [v6]
def Onnx_Mean_V6Op : Onnx_Op<"Mean_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Mean_V6 operation [v6]";
  let description = [{
      Element-wise mean of each of the input tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$mean);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mean [v8]
def Onnx_Mean_V8Op : Onnx_Op<"Mean_V8", [Pure, OPCountInfo]> {
  let summary = "ONNX Mean_V8 operation [v8]";
  let description = [{
      Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$mean);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Mean [v13]
def Onnx_MeanOp : Onnx_Op<"Mean", [Pure, OPCountInfo]> {
  let summary = "ONNX Mean operation [v13]";
  let description = [{
      Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$mean);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// IsNaN [v9]
def Onnx_IsNaN_V9Op : Onnx_Op<"IsNaN_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX IsNaN_V9 operation [v9]";
  let description = [{
      Returns which elements of the input are NaN.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs TensorOf<[I1]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// IsNaN [v13]
def Onnx_IsNaN_V13Op : Onnx_Op<"IsNaN_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX IsNaN_V13 operation [v13]";
  let description = [{
      Returns which elements of the input are NaN.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs TensorOf<[I1]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// IsNaN [v20]
def Onnx_IsNaNOp : Onnx_Op<"IsNaN", [Pure, OPCountInfo]> {
  let summary = "ONNX IsNaN operation [v20]";
  let description = [{
      Returns which elements of the input are NaN.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$X);
  let results = (outs TensorOf<[I1]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RegexFullMatch [v20]
def Onnx_RegexFullMatchOp : Onnx_Op<"RegexFullMatch", [Pure, OPCountInfo]> {
  let summary = "ONNX RegexFullMatch operation [v20]";
  let description = [{
      RegexFullMatch performs a full regex match on each element of the input tensor.
  }];

  let arguments = (ins TensorOf<[StringType]>:$X,
                       OptionalAttr<StrAttr>:$pattern);
  let results = (outs TensorOf<[I1]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Asin [v7]
def Onnx_Asin_V7Op : Onnx_Op<"Asin_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Asin_V7 operation [v7]";
  let description = [{
      Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Asin [v22]
def Onnx_AsinOp : Onnx_Op<"Asin", [Pure, OPCountInfo]> {
  let summary = "ONNX Asin operation [v22]";
  let description = [{
      Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DepthToSpace [v1]
def Onnx_DepthToSpace_V1Op : Onnx_Op<"DepthToSpace_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX DepthToSpace_V1 operation [v1]";
  let description = [{
      DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       I64Attr:$blocksize);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DepthToSpace [v11]
def Onnx_DepthToSpace_V11Op : Onnx_Op<"DepthToSpace_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX DepthToSpace_V11 operation [v11]";
  let description = [{
      DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       I64Attr:$blocksize,
                       DefaultValuedStrAttr<StrAttr, "DCR">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// DepthToSpace [v13]
def Onnx_DepthToSpaceOp : Onnx_Op<"DepthToSpace", [Pure, OPCountInfo]> {
  let summary = "ONNX DepthToSpace operation [v13]";
  let description = [{
      DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       I64Attr:$blocksize,
                       DefaultValuedStrAttr<StrAttr, "DCR">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Div [v1]
def Onnx_Div_V1Op : Onnx_Op<"Div_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Div_V1 operation [v1]";
  let description = [{
      Performs element-wise binary division (with limited broadcast support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Div [v6]
def Onnx_Div_V6Op : Onnx_Op<"Div_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Div_V6 operation [v6]";
  let description = [{
      Performs element-wise binary division (with limited broadcast support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Div [v7]
def Onnx_Div_V7Op : Onnx_Op<"Div_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Div_V7 operation [v7]";
  let description = [{
      Performs element-wise binary division (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Div [v13]
def Onnx_Div_V13Op : Onnx_Op<"Div_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Div_V13 operation [v13]";
  let description = [{
      Performs element-wise binary division (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Div [v14]
def Onnx_DivOp : Onnx_Op<"Div", [Pure, OPCountInfo]> {
  let summary = "ONNX Div operation [v14]";
  let description = [{
      Performs element-wise binary division (with Numpy-style broadcasting support).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Softsign [v1]
def Onnx_Softsign_V1Op : Onnx_Op<"Softsign_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Softsign_V1 operation [v1]";
  let description = [{
      Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Softsign [v22]
def Onnx_SoftsignOp : Onnx_Op<"Softsign", [Pure, OPCountInfo]> {
  let summary = "ONNX Softsign operation [v22]";
  let description = [{
      Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GlobalMaxPool [v1]
def Onnx_GlobalMaxPool_V1Op : Onnx_Op<"GlobalMaxPool_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX GlobalMaxPool_V1 operation [v1]";
  let description = [{
      GlobalMaxPool consumes an input tensor X and applies max pooling across
  the values in the same channel.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GlobalMaxPool [v22]
def Onnx_GlobalMaxPoolOp : Onnx_Op<"GlobalMaxPool", [Pure, OPCountInfo]> {
  let summary = "ONNX GlobalMaxPool operation [v22]";
  let description = [{
      GlobalMaxPool consumes an input tensor X and applies max pooling across
  the values in the same channel.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reciprocal [v1]
def Onnx_Reciprocal_V1Op : Onnx_Op<"Reciprocal_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Reciprocal_V1 operation [v1]";
  let description = [{
      Reciprocal takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the reciprocal is, y = 1/x, is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reciprocal [v6]
def Onnx_Reciprocal_V6Op : Onnx_Op<"Reciprocal_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Reciprocal_V6 operation [v6]";
  let description = [{
      Reciprocal takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the reciprocal is, y = 1/x, is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reciprocal [v13]
def Onnx_ReciprocalOp : Onnx_Op<"Reciprocal", [Pure, OPCountInfo]> {
  let summary = "ONNX Reciprocal operation [v13]";
  let description = [{
      Reciprocal takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the reciprocal is, y = 1/x, is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MeanVarianceNormalization [v9]
def Onnx_MeanVarianceNormalization_V9Op : Onnx_Op<"MeanVarianceNormalization_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX MeanVarianceNormalization_V9 operation [v9]";
  let description = [{
      A MeanVarianceNormalization Function: Perform mean variance normalization
  on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<I64ArrayAttr, "{0,2,3}">:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// MeanVarianceNormalization [v13]
def Onnx_MeanVarianceNormalizationOp : Onnx_Op<"MeanVarianceNormalization", [Pure, OPCountInfo]> {
  let summary = "ONNX MeanVarianceNormalization operation [v13]";
  let description = [{
      A MeanVarianceNormalization Function: Perform mean variance normalization
  on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       DefaultValuedAttr<I64ArrayAttr, "{0,2,3}">:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceL1 [v1]
def Onnx_ReduceL1_V1Op : Onnx_Op<"ReduceL1_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceL1_V1 operation [v1]";
  let description = [{
      Computes the L1 norm of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceL1 [v11]
def Onnx_ReduceL1_V11Op : Onnx_Op<"ReduceL1_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceL1_V11 operation [v11]";
  let description = [{
      Computes the L1 norm of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceL1 [v13]
def Onnx_ReduceL1_V13Op : Onnx_Op<"ReduceL1_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceL1_V13 operation [v13]";
  let description = [{
      Computes the L1 norm of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceL1 [v18]
def Onnx_ReduceL1Op : Onnx_Op<"ReduceL1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceL1 operation [v18]";
  let description = [{
      Computes the L1 norm of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Relu [v1]
def Onnx_Relu_V1Op : Onnx_Op<"Relu_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Relu_V1 operation [v1]";
  let description = [{
      Relu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Relu [v6]
def Onnx_Relu_V6Op : Onnx_Op<"Relu_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Relu_V6 operation [v6]";
  let description = [{
      Relu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Relu [v13]
def Onnx_Relu_V13Op : Onnx_Op<"Relu_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Relu_V13 operation [v13]";
  let description = [{
      Relu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Relu [v14]
def Onnx_ReluOp : Onnx_Op<"Relu", [Pure, OPCountInfo]> {
  let summary = "ONNX Relu operation [v14]";
  let description = [{
      Relu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
  the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ScatterElements [v11]
def Onnx_ScatterElements_V11Op : Onnx_Op<"ScatterElements_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ScatterElements_V11 operation [v11]";
  let description = [{
      ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
  rank r >= 1 and an optional attribute axis that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ScatterElements [v13]
def Onnx_ScatterElements_V13Op : Onnx_Op<"ScatterElements_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ScatterElements_V13 operation [v13]";
  let description = [{
      ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
  rank r >= 1 and an optional attribute axis that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ScatterElements [v16]
def Onnx_ScatterElements_V16Op : Onnx_Op<"ScatterElements_V16", [Pure, OPCountInfo]> {
  let summary = "ONNX ScatterElements_V16 operation [v16]";
  let description = [{
      ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
  rank r >= 1 and an optional attribute axis that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedStrAttr<StrAttr, "none">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ScatterElements [v18]
def Onnx_ScatterElementsOp : Onnx_Op<"ScatterElements", [Pure, OPCountInfo]> {
  let summary = "ONNX ScatterElements operation [v18]";
  let description = [{
      ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
  rank r >= 1 and an optional attribute axis that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates,
                       DefaultValuedAttr<I64Attr, "0">:$axis,
                       DefaultValuedStrAttr<StrAttr, "none">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceSum [v1]
def Onnx_ReduceSum_V1Op : Onnx_Op<"ReduceSum_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceSum_V1 operation [v1]";
  let description = [{
      Computes the sum of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceSum [v11]
def Onnx_ReduceSum_V11Op : Onnx_Op<"ReduceSum_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceSum_V11 operation [v11]";
  let description = [{
      Computes the sum of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceSum [v13]
def Onnx_ReduceSumOp : Onnx_Op<"ReduceSum", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceSum operation [v13]";
  let description = [{
      Computes the sum of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Elu [v1]
def Onnx_Elu_V1Op : Onnx_Op<"Elu_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Elu_V1 operation [v1]";
  let description = [{
      Elu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "1">:$alpha,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Elu [v6]
def Onnx_Elu_V6Op : Onnx_Op<"Elu_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Elu_V6 operation [v6]";
  let description = [{
      Elu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "1">:$alpha);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Elu [v22]
def Onnx_EluOp : Onnx_Op<"Elu", [Pure, OPCountInfo]> {
  let summary = "ONNX Elu operation [v22]";
  let description = [{
      Elu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "1">:$alpha);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reshape [v1]
def Onnx_Reshape_V1Op : Onnx_Op<"Reshape_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Reshape_V1 operation [v1]";
  let description = [{
      Reshape the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs,
                       OptionalAttr<I64ArrayAttr>:$shape);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reshaped);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reshape [v5]
def Onnx_Reshape_V5Op : Onnx_Op<"Reshape_V5", [Pure, OPCountInfo]> {
  let summary = "ONNX Reshape_V5 operation [v5]";
  let description = [{
      Reshape the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$shape);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$reshaped);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reshape [v13]
def Onnx_Reshape_V13Op : Onnx_Op<"Reshape_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Reshape_V13 operation [v13]";
  let description = [{
      Reshape the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$shape);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$reshaped);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reshape [v14]
def Onnx_Reshape_V14Op : Onnx_Op<"Reshape_V14", [Pure, OPCountInfo]> {
  let summary = "ONNX Reshape_V14 operation [v14]";
  let description = [{
      Reshape the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$shape,
                       DefaultValuedAttr<I64Attr, "0">:$allowzero);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$reshaped);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reshape [v19]
def Onnx_Reshape_V19Op : Onnx_Op<"Reshape_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX Reshape_V19 operation [v19]";
  let description = [{
      Reshape the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$data,
                       TensorOf<[I64]>:$shape,
                       DefaultValuedAttr<I64Attr, "0">:$allowzero);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$reshaped);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reshape [v21]
def Onnx_Reshape_V21Op : Onnx_Op<"Reshape_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Reshape_V21 operation [v21]";
  let description = [{
      Reshape the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$data,
                       TensorOf<[I64]>:$shape,
                       DefaultValuedAttr<I64Attr, "0">:$allowzero);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$reshaped);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Reshape [v23]
def Onnx_ReshapeOp : Onnx_Op<"Reshape", [Pure, OPCountInfo]> {
  let summary = "ONNX Reshape operation [v23]";
  let description = [{
      Reshape the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$data,
                       TensorOf<[I64]>:$shape,
                       DefaultValuedAttr<I64Attr, "0">:$allowzero);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$reshaped);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ImageDecoder [v20]
def Onnx_ImageDecoderOp : Onnx_Op<"ImageDecoder", [Pure, OPCountInfo]> {
  let summary = "ONNX ImageDecoder operation [v20]";
  let description = [{
      Loads and decodes and image from a file.
  }];

  let arguments = (ins TensorOf<[UI8]>:$encoded_stream,
                       DefaultValuedStrAttr<StrAttr, "RGB">:$pixel_format);
  let results = (outs TensorOf<[UI8]>:$image);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Shape [v1]
def Onnx_Shape_V1Op : Onnx_Op<"Shape_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Shape_V1 operation [v1]";
  let description = [{
      Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data);
  let results = (outs TensorOf<[I64]>:$shape);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Shape [v13]
def Onnx_Shape_V13Op : Onnx_Op<"Shape_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Shape_V13 operation [v13]";
  let description = [{
      Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data);
  let results = (outs TensorOf<[I64]>:$shape);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Shape [v15]
def Onnx_Shape_V15Op : Onnx_Op<"Shape_V15", [Pure, OPCountInfo]> {
  let summary = "ONNX Shape_V15 operation [v15]";
  let description = [{
      Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       OptionalAttr<I64Attr>:$end,
                       DefaultValuedAttr<I64Attr, "0">:$start);
  let results = (outs TensorOf<[I64]>:$shape);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Shape [v19]
def Onnx_Shape_V19Op : Onnx_Op<"Shape_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX Shape_V19 operation [v19]";
  let description = [{
      Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$data,
                       OptionalAttr<I64Attr>:$end,
                       DefaultValuedAttr<I64Attr, "0">:$start);
  let results = (outs TensorOf<[I64]>:$shape);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Shape [v21]
def Onnx_Shape_V21Op : Onnx_Op<"Shape_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Shape_V21 operation [v21]";
  let description = [{
      Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$data,
                       OptionalAttr<I64Attr>:$end,
                       DefaultValuedAttr<I64Attr, "0">:$start);
  let results = (outs TensorOf<[I64]>:$shape);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Shape [v23]
def Onnx_ShapeOp : Onnx_Op<"Shape", [Pure, OPCountInfo]> {
  let summary = "ONNX Shape operation [v23]";
  let description = [{
      Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$data,
                       OptionalAttr<I64Attr>:$end,
                       DefaultValuedAttr<I64Attr, "0">:$start);
  let results = (outs TensorOf<[I64]>:$shape);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Selu [v1]
def Onnx_Selu_V1Op : Onnx_Op<"Selu_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Selu_V1 operation [v1]";
  let description = [{
      Selu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the scaled exponential linear unit function,
  `y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
  is applied to the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "1.6732">:$alpha,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs,
                       DefaultValuedAttr<F32Attr, "1.0507">:$gamma);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Selu [v6]
def Onnx_Selu_V6Op : Onnx_Op<"Selu_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Selu_V6 operation [v6]";
  let description = [{
      Selu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the scaled exponential linear unit function,
  `y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
  is applied to the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "1.67326">:$alpha,
                       DefaultValuedAttr<F32Attr, "1.0507">:$gamma);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Selu [v22]
def Onnx_SeluOp : Onnx_Op<"Selu", [Pure, OPCountInfo]> {
  let summary = "ONNX Selu operation [v22]";
  let description = [{
      Selu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the scaled exponential linear unit function,
  `y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
  is applied to the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "1.67326">:$alpha,
                       DefaultValuedAttr<F32Attr, "1.0507">:$gamma);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ConstantOfShape [v9]
def Onnx_ConstantOfShape_V9Op : Onnx_Op<"ConstantOfShape_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX ConstantOfShape_V9 operation [v9]";
  let description = [{
      Generate a tensor with given value and shape.
  }];

  let arguments = (ins TensorOf<[I64]>:$input,
                       OptionalAttr<AnyAttr>:$value);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ConstantOfShape [v20]
def Onnx_ConstantOfShape_V20Op : Onnx_Op<"ConstantOfShape_V20", [Pure, OPCountInfo]> {
  let summary = "ONNX ConstantOfShape_V20 operation [v20]";
  let description = [{
      Generate a tensor with given value and shape.
  }];

  let arguments = (ins TensorOf<[I64]>:$input,
                       OptionalAttr<AnyAttr>:$value);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ConstantOfShape [v21]
def Onnx_ConstantOfShape_V21Op : Onnx_Op<"ConstantOfShape_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX ConstantOfShape_V21 operation [v21]";
  let description = [{
      Generate a tensor with given value and shape.
  }];

  let arguments = (ins TensorOf<[I64]>:$input,
                       OptionalAttr<AnyAttr>:$value);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ConstantOfShape [v23]
def Onnx_ConstantOfShapeOp : Onnx_Op<"ConstantOfShape", [Pure, OPCountInfo]> {
  let summary = "ONNX ConstantOfShape operation [v23]";
  let description = [{
      Generate a tensor with given value and shape.
  }];

  let arguments = (ins TensorOf<[I64]>:$input,
                       OptionalAttr<AnyAttr>:$value);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[F4E2M1FN]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GlobalAveragePool [v1]
def Onnx_GlobalAveragePool_V1Op : Onnx_Op<"GlobalAveragePool_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX GlobalAveragePool_V1 operation [v1]";
  let description = [{
      GlobalAveragePool consumes an input tensor X and applies average pooling across
  the values in the same channel.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// GlobalAveragePool [v22]
def Onnx_GlobalAveragePoolOp : Onnx_Op<"GlobalAveragePool", [Pure, OPCountInfo]> {
  let summary = "ONNX GlobalAveragePool operation [v22]";
  let description = [{
      GlobalAveragePool consumes an input tensor X and applies average pooling across
  the values in the same channel.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Size [v1]
def Onnx_Size_V1Op : Onnx_Op<"Size_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Size_V1 operation [v1]";
  let description = [{
      Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data);
  let results = (outs TensorOf<[I64]>:$size);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Size [v13]
def Onnx_Size_V13Op : Onnx_Op<"Size_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Size_V13 operation [v13]";
  let description = [{
      Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data);
  let results = (outs TensorOf<[I64]>:$size);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Size [v19]
def Onnx_Size_V19Op : Onnx_Op<"Size_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX Size_V19 operation [v19]";
  let description = [{
      Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$data);
  let results = (outs TensorOf<[I64]>:$size);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Size [v21]
def Onnx_Size_V21Op : Onnx_Op<"Size_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Size_V21 operation [v21]";
  let description = [{
      Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$data);
  let results = (outs TensorOf<[I64]>:$size);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Size [v23]
def Onnx_SizeOp : Onnx_Op<"Size", [Pure, OPCountInfo]> {
  let summary = "ONNX Size operation [v23]";
  let description = [{
      Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$data);
  let results = (outs TensorOf<[I64]>:$size);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// HardSigmoid [v1]
def Onnx_HardSigmoid_V1Op : Onnx_Op<"HardSigmoid_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX HardSigmoid_V1 operation [v1]";
  let description = [{
      HardSigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
  is applied to the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "0.2">:$alpha,
                       DefaultValuedAttr<F32Attr, "0.5">:$beta,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// HardSigmoid [v6]
def Onnx_HardSigmoid_V6Op : Onnx_Op<"HardSigmoid_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX HardSigmoid_V6 operation [v6]";
  let description = [{
      HardSigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
  is applied to the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "0.2">:$alpha,
                       DefaultValuedAttr<F32Attr, "0.5">:$beta);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// HardSigmoid [v22]
def Onnx_HardSigmoidOp : Onnx_Op<"HardSigmoid", [Pure, OPCountInfo]> {
  let summary = "ONNX HardSigmoid operation [v22]";
  let description = [{
      HardSigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
  is applied to the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "0.2">:$alpha,
                       DefaultValuedAttr<F32Attr, "0.5">:$beta);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LogSoftmax [v1]
def Onnx_LogSoftmax_V1Op : Onnx_Op<"LogSoftmax_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX LogSoftmax_V1 operation [v1]";
  let description = [{
      The operator computes the logsoftmax (log of softmax) values for each layer in the batch
  of the given input.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LogSoftmax [v11]
def Onnx_LogSoftmax_V11Op : Onnx_Op<"LogSoftmax_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX LogSoftmax_V11 operation [v11]";
  let description = [{
      The operator computes the logsoftmax (log of softmax) values for each layer in the batch
  of the given input.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LogSoftmax [v13]
def Onnx_LogSoftmaxOp : Onnx_Op<"LogSoftmax", [Pure, OPCountInfo]> {
  let summary = "ONNX LogSoftmax operation [v13]";
  let description = [{
      The operator computes the log of softmax values for the given input:
  LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
  The "axis" attribute indicates the dimension along which LogSoftmax
  will be performed.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input,
                       DefaultValuedAttr<I64Attr, "-1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SpaceToDepth [v1]
def Onnx_SpaceToDepth_V1Op : Onnx_Op<"SpaceToDepth_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX SpaceToDepth_V1 operation [v1]";
  let description = [{
      SpaceToDepth rearranges blocks of spatial data into depth.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       I64Attr:$blocksize);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SpaceToDepth [v13]
def Onnx_SpaceToDepthOp : Onnx_Op<"SpaceToDepth", [Pure, OPCountInfo]> {
  let summary = "ONNX SpaceToDepth operation [v13]";
  let description = [{
      SpaceToDepth rearranges blocks of spatial data into depth.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       I64Attr:$blocksize);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// BitwiseNot [v18]
def Onnx_BitwiseNotOp : Onnx_Op<"BitwiseNot", [Pure, OPCountInfo]> {
  let summary = "ONNX BitwiseNot operation [v18]";
  let description = [{
      Returns the bitwise not of the input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceSumSquare [v1]
def Onnx_ReduceSumSquare_V1Op : Onnx_Op<"ReduceSumSquare_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceSumSquare_V1 operation [v1]";
  let description = [{
      Computes the sum square of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceSumSquare [v11]
def Onnx_ReduceSumSquare_V11Op : Onnx_Op<"ReduceSumSquare_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceSumSquare_V11 operation [v11]";
  let description = [{
      Computes the sum square of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceSumSquare [v13]
def Onnx_ReduceSumSquare_V13Op : Onnx_Op<"ReduceSumSquare_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceSumSquare_V13 operation [v13]";
  let description = [{
      Computes the sum square of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceSumSquare [v18]
def Onnx_ReduceSumSquareOp : Onnx_Op<"ReduceSumSquare", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceSumSquare operation [v18]";
  let description = [{
      Computes the sum square of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sqrt [v1]
def Onnx_Sqrt_V1Op : Onnx_Op<"Sqrt_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Sqrt_V1 operation [v1]";
  let description = [{
      Square root takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the square root is, y = x^0.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sqrt [v6]
def Onnx_Sqrt_V6Op : Onnx_Op<"Sqrt_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Sqrt_V6 operation [v6]";
  let description = [{
      Square root takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the square root is, y = x^0.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sqrt [v13]
def Onnx_SqrtOp : Onnx_Op<"Sqrt", [Pure, OPCountInfo]> {
  let summary = "ONNX Sqrt operation [v13]";
  let description = [{
      Square root takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the square root is, y = x^0.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Col2Im [v18]
def Onnx_Col2ImOp : Onnx_Op<"Col2Im", [Pure, OPCountInfo]> {
  let summary = "ONNX Col2Im operation [v18]";
  let description = [{
      The operator rearranges column blocks back into a multidimensional image
  Col2Im behaves similarly to PyTorch's fold https://pytorch.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       TensorOf<[I64]>:$image_shape,
                       TensorOf<[I64]>:$block_shape,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RandomNormalLike [v1]
def Onnx_RandomNormalLike_V1Op : Onnx_Op<"RandomNormalLike_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX RandomNormalLike_V1 operation [v1]";
  let description = [{
      Generate a tensor with random values drawn from a normal distribution.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       OptionalAttr<I64Attr>:$dtype,
                       DefaultValuedAttr<F32Attr, "0">:$mean,
                       DefaultValuedAttr<F32Attr, "1">:$scale,
                       OptionalAttr<F32Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// RandomNormalLike [v22]
def Onnx_RandomNormalLikeOp : Onnx_Op<"RandomNormalLike", [Pure, OPCountInfo]> {
  let summary = "ONNX RandomNormalLike operation [v22]";
  let description = [{
      Generate a tensor with random values drawn from a normal distribution.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       OptionalAttr<I64Attr>:$dtype,
                       DefaultValuedAttr<F32Attr, "0">:$mean,
                       DefaultValuedAttr<F32Attr, "1">:$scale,
                       OptionalAttr<F32Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Multinomial [v7]
def Onnx_Multinomial_V7Op : Onnx_Op<"Multinomial_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Multinomial_V7 operation [v7]";
  let description = [{
      Generate a tensor of samples from a multinomial distribution according to the probabilities
  of each of the possible outcomes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "6">:$dtype,
                       DefaultValuedAttr<I64Attr, "1">:$sample_size,
                       OptionalAttr<F32Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Multinomial [v22]
def Onnx_MultinomialOp : Onnx_Op<"Multinomial", [Pure, OPCountInfo]> {
  let summary = "ONNX Multinomial operation [v22]";
  let description = [{
      Generate a tensor of samples from a multinomial distribution according to the probabilities
  of each of the possible outcomes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "6">:$dtype,
                       DefaultValuedAttr<I64Attr, "1">:$sample_size,
                       OptionalAttr<F32Attr>:$seed);
  let results = (outs AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Identity [v1]
def Onnx_Identity_V1Op : Onnx_Op<"Identity_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Identity_V1 operation [v1]";
  let description = [{
      Identity operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Identity [v13]
def Onnx_Identity_V13Op : Onnx_Op<"Identity_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Identity_V13 operation [v13]";
  let description = [{
      Identity operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Identity [v14]
def Onnx_Identity_V14Op : Onnx_Op<"Identity_V14", [Pure, OPCountInfo]> {
  let summary = "ONNX Identity_V14 operation [v14]";
  let description = [{
      Identity operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Identity [v16]
def Onnx_Identity_V16Op : Onnx_Op<"Identity_V16", [Pure, OPCountInfo]> {
  let summary = "ONNX Identity_V16 operation [v16]";
  let description = [{
      Identity operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Identity [v19]
def Onnx_Identity_V19Op : Onnx_Op<"Identity_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX Identity_V19 operation [v19]";
  let description = [{
      Identity operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Identity [v21]
def Onnx_Identity_V21Op : Onnx_Op<"Identity_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Identity_V21 operation [v21]";
  let description = [{
      Identity operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Identity [v23]
def Onnx_IdentityOp : Onnx_Op<"Identity", [Pure, OPCountInfo]> {
  let summary = "ONNX Identity operation [v23]";
  let description = [{
      Identity operator.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>, OptOf<SeqOf<[TensorOf<[UI8]>]>>, OptOf<SeqOf<[TensorOf<[UI16]>]>>, OptOf<SeqOf<[TensorOf<[UI32]>]>>, OptOf<SeqOf<[TensorOf<[UI64]>]>>, OptOf<SeqOf<[TensorOf<[I8]>]>>, OptOf<SeqOf<[TensorOf<[I16]>]>>, OptOf<SeqOf<[TensorOf<[I32]>]>>, OptOf<SeqOf<[TensorOf<[I64]>]>>, OptOf<SeqOf<[TensorOf<[F16]>]>>, OptOf<SeqOf<[TensorOf<[F32]>]>>, OptOf<SeqOf<[TensorOf<[F64]>]>>, OptOf<SeqOf<[TensorOf<[StringType]>]>>, OptOf<SeqOf<[TensorOf<[I1]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F32>]>]>>, OptOf<SeqOf<[TensorOf<[Complex<F64>]>]>>, OptOf<TensorOf<[UI8]>>, OptOf<TensorOf<[UI16]>>, OptOf<TensorOf<[UI32]>>, OptOf<TensorOf<[UI64]>>, OptOf<TensorOf<[I8]>>, OptOf<TensorOf<[I16]>>, OptOf<TensorOf<[I32]>>, OptOf<TensorOf<[I64]>>, OptOf<TensorOf<[F16]>>, OptOf<TensorOf<[F32]>>, OptOf<TensorOf<[F64]>>, OptOf<TensorOf<[StringType]>>, OptOf<TensorOf<[I1]>>, OptOf<TensorOf<[Complex<F32>]>>, OptOf<TensorOf<[Complex<F64>]>>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Expand [v8]
def Onnx_Expand_V8Op : Onnx_Op<"Expand_V8", [Pure, OPCountInfo]> {
  let summary = "ONNX Expand_V8 operation [v8]";
  let description = [{
      Broadcast the input tensor following the given shape and the broadcast rule.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       TensorOf<[I64]>:$shape);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Expand [v13]
def Onnx_ExpandOp : Onnx_Op<"Expand", [Pure, OPCountInfo]> {
  let summary = "ONNX Expand operation [v13]";
  let description = [{
      Broadcast the input tensor following the given shape and the broadcast rule.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       TensorOf<[I64]>:$shape);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Squeeze [v1]
def Onnx_Squeeze_V1Op : Onnx_Op<"Squeeze_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Squeeze_V1 operation [v1]";
  let description = [{
      Remove single-dimensional entries from the shape of a tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$squeezed);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Squeeze [v11]
def Onnx_Squeeze_V11Op : Onnx_Op<"Squeeze_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Squeeze_V11 operation [v11]";
  let description = [{
      Remove single-dimensional entries from the shape of a tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$squeezed);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Squeeze [v13]
def Onnx_Squeeze_V13Op : Onnx_Op<"Squeeze_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Squeeze_V13 operation [v13]";
  let description = [{
      Remove single-dimensional entries from the shape of a tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$squeezed);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Squeeze [v21]
def Onnx_Squeeze_V21Op : Onnx_Op<"Squeeze_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Squeeze_V21 operation [v21]";
  let description = [{
      Remove single-dimensional entries from the shape of a tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$squeezed);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Squeeze [v23]
def Onnx_SqueezeOp : Onnx_Op<"Squeeze", [Pure, OPCountInfo]> {
  let summary = "ONNX Squeeze operation [v23]";
  let description = [{
      Remove single-dimensional entries from the shape of a tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$squeezed);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// STFT [v17]
def Onnx_STFTOp : Onnx_Op<"STFT", [Pure, OPCountInfo]> {
  let summary = "ONNX STFT operation [v17]";
  let description = [{
      Computes the Short-time Fourier Transform of the signal.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$signal,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$frame_step,
                       AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$window,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$frame_length,
                       DefaultValuedAttr<I64Attr, "1">:$onesided);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sum [v1]
def Onnx_Sum_V1Op : Onnx_Op<"Sum_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Sum_V1 operation [v1]";
  let description = [{
      Element-wise sum of each of the input tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$sum);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sum [v6]
def Onnx_Sum_V6Op : Onnx_Op<"Sum_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Sum_V6 operation [v6]";
  let description = [{
      Element-wise sum of each of the input tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$sum);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sum [v8]
def Onnx_Sum_V8Op : Onnx_Op<"Sum_V8", [Pure, OPCountInfo]> {
  let summary = "ONNX Sum_V8 operation [v8]";
  let description = [{
      Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$sum);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sum [v13]
def Onnx_SumOp : Onnx_Op<"Sum", [Pure, OPCountInfo]> {
  let summary = "ONNX Sum operation [v13]";
  let description = [{
      Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>:$data_0);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$sum);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Upsample [v1]
def Onnx_Upsample_V1Op : Onnx_Op<"Upsample_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Upsample_V1 operation [v1]";
  let description = [{
      Upsample the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I1]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       F32Attr:$height_scale,
                       DefaultValuedStrAttr<StrAttr, "nearest">:$mode,
                       F32Attr:$width_scale);
  let results = (outs AnyTypeOf<[TensorOf<[I1]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Upsample [v7]
def Onnx_Upsample_V7Op : Onnx_Op<"Upsample_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Upsample_V7 operation [v7]";
  let description = [{
      Upsample the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "nearest">:$mode,
                       F32ArrayAttr:$scales);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Upsample [v9]
def Onnx_Upsample_V9Op : Onnx_Op<"Upsample_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Upsample_V9 operation [v9]";
  let description = [{
      Upsample the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       TensorOf<[F32]>:$scales,
                       DefaultValuedStrAttr<StrAttr, "nearest">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Upsample [v10]
def Onnx_UpsampleOp : Onnx_Op<"Upsample", [Pure, OPCountInfo]> {
  let summary = "ONNX Upsample operation [v10]";
  let description = [{
      Upsample the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       TensorOf<[F32]>:$scales,
                       DefaultValuedStrAttr<StrAttr, "nearest">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Tanh [v1]
def Onnx_Tanh_V1Op : Onnx_Op<"Tanh_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Tanh_V1 operation [v1]";
  let description = [{
      Calculates the hyperbolic tangent of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       OptionalAttr<I64ArrayAttr>:$consumed_inputs);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Tanh [v6]
def Onnx_Tanh_V6Op : Onnx_Op<"Tanh_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Tanh_V6 operation [v6]";
  let description = [{
      Calculates the hyperbolic tangent of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Tanh [v13]
def Onnx_TanhOp : Onnx_Op<"Tanh", [Pure, OPCountInfo]> {
  let summary = "ONNX Tanh operation [v13]";
  let description = [{
      Calculates the hyperbolic tangent of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Tile [v1]
def Onnx_Tile_V1Op : Onnx_Op<"Tile_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Tile_V1 operation [v1]";
  let description = [{
      Repeat the elements of a tensor along an axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$tiles,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Tile [v6]
def Onnx_Tile_V6Op : Onnx_Op<"Tile_V6", [Pure, OPCountInfo]> {
  let summary = "ONNX Tile_V6 operation [v6]";
  let description = [{
      Constructs a tensor by tiling a given tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       TensorOf<[I64]>:$repeats);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Tile [v13]
def Onnx_TileOp : Onnx_Op<"Tile", [Pure, OPCountInfo]> {
  let summary = "ONNX Tile operation [v13]";
  let description = [{
      Constructs a tensor by tiling a given tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       TensorOf<[I64]>:$repeats);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// TopK [v1]
def Onnx_TopK_V1Op : Onnx_Op<"TopK_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX TopK_V1 operation [v1]";
  let description = [{
      Retrieve the top-K elements along a specified axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<I64Attr, "-1">:$axis,
                       I64Attr:$k);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Values,
                      TensorOf<[I64]>:$Indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// TopK [v10]
def Onnx_TopK_V10Op : Onnx_Op<"TopK_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX TopK_V10 operation [v10]";
  let description = [{
      Retrieve the top-K elements along a specified axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       TensorOf<[I64]>:$K,
                       DefaultValuedAttr<I64Attr, "-1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Values,
                      TensorOf<[I64]>:$Indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// TopK [v11]
def Onnx_TopKOp : Onnx_Op<"TopK", [Pure, OPCountInfo]> {
  let summary = "ONNX TopK operation [v11]";
  let description = [{
      Retrieve the top-K largest or smallest elements along a specified axis.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       TensorOf<[I64]>:$K,
                       DefaultValuedAttr<I64Attr, "-1">:$axis,
                       DefaultValuedAttr<I64Attr, "1">:$largest,
                       DefaultValuedAttr<I64Attr, "1">:$sorted);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Values,
                      TensorOf<[I64]>:$Indices);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 2;
    }
  }];
}

/// LRN [v1]
def Onnx_LRN_V1Op : Onnx_Op<"LRN_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX LRN_V1 operation [v1]";
  let description = [{
      Local Response Normalization proposed in the [AlexNet paper](https://papers.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "0.0001">:$alpha,
                       DefaultValuedAttr<F32Attr, "0.75">:$beta,
                       DefaultValuedAttr<F32Attr, "1">:$bias,
                       I64Attr:$size);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// LRN [v13]
def Onnx_LRNOp : Onnx_Op<"LRN", [Pure, OPCountInfo]> {
  let summary = "ONNX LRN operation [v13]";
  let description = [{
      Local Response Normalization proposed in the [AlexNet paper](https://papers.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       DefaultValuedAttr<F32Attr, "0.0001">:$alpha,
                       DefaultValuedAttr<F32Attr, "0.75">:$beta,
                       DefaultValuedAttr<F32Attr, "1">:$bias,
                       I64Attr:$size);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Unsqueeze [v1]
def Onnx_Unsqueeze_V1Op : Onnx_Op<"Unsqueeze_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Unsqueeze_V1 operation [v1]";
  let description = [{
      Insert single-dimensional entries to the shape of a tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       I64ArrayAttr:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$expanded);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Unsqueeze [v11]
def Onnx_Unsqueeze_V11Op : Onnx_Op<"Unsqueeze_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Unsqueeze_V11 operation [v11]";
  let description = [{
      Insert single-dimensional entries to the shape of an input tensor (`data`).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       I64ArrayAttr:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$expanded);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Unsqueeze [v13]
def Onnx_Unsqueeze_V13Op : Onnx_Op<"Unsqueeze_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Unsqueeze_V13 operation [v13]";
  let description = [{
      Insert single-dimensional entries to the shape of an input tensor (`data`).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       TensorOf<[I64]>:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$expanded);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Unsqueeze [v21]
def Onnx_Unsqueeze_V21Op : Onnx_Op<"Unsqueeze_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Unsqueeze_V21 operation [v21]";
  let description = [{
      Insert single-dimensional entries to the shape of an input tensor (`data`).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$data,
                       TensorOf<[I64]>:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$expanded);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Unsqueeze [v23]
def Onnx_UnsqueezeOp : Onnx_Op<"Unsqueeze", [Pure, OPCountInfo]> {
  let summary = "ONNX Unsqueeze operation [v23]";
  let description = [{
      Insert single-dimensional entries to the shape of an input tensor (`data`).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$data,
                       TensorOf<[I64]>:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$expanded);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ThresholdedRelu [v10]
def Onnx_ThresholdedRelu_V10Op : Onnx_Op<"ThresholdedRelu_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX ThresholdedRelu_V10 operation [v10]";
  let description = [{
      ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
  is applied to the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "1">:$alpha);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ThresholdedRelu [v22]
def Onnx_ThresholdedReluOp : Onnx_Op<"ThresholdedRelu", [Pure, OPCountInfo]> {
  let summary = "ONNX ThresholdedRelu operation [v22]";
  let description = [{
      ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
  is applied to the tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       DefaultValuedAttr<F32Attr, "1">:$alpha);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// SequenceErase [v11]
def Onnx_SequenceEraseOp : Onnx_Op<"SequenceErase", [Pure, OPCountInfo]> {
  let summary = "ONNX SequenceErase operation [v11]";
  let description = [{
      Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
  }];

  let arguments = (ins AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$input_sequence,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$position);
  let results = (outs AnyTypeOf<[SeqOf<[TensorOf<[UI8]>]>, SeqOf<[TensorOf<[UI16]>]>, SeqOf<[TensorOf<[UI32]>]>, SeqOf<[TensorOf<[UI64]>]>, SeqOf<[TensorOf<[I8]>]>, SeqOf<[TensorOf<[I16]>]>, SeqOf<[TensorOf<[I32]>]>, SeqOf<[TensorOf<[I64]>]>, SeqOf<[TensorOf<[F16]>]>, SeqOf<[TensorOf<[F32]>]>, SeqOf<[TensorOf<[F64]>]>, SeqOf<[TensorOf<[StringType]>]>, SeqOf<[TensorOf<[I1]>]>, SeqOf<[TensorOf<[Complex<F32>]>]>, SeqOf<[TensorOf<[Complex<F64>]>]>]>:$output_sequence);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Xor [v1]
def Onnx_Xor_V1Op : Onnx_Op<"Xor_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Xor_V1 operation [v1]";
  let description = [{
      Returns the tensor resulted from performing the `xor` logical operation
  elementwise on the input tensors `A` and `B`.
  }];

  let arguments = (ins TensorOf<[I1]>:$A,
                       TensorOf<[I1]>:$B,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Xor [v7]
def Onnx_XorOp : Onnx_Op<"Xor", [Pure, OPCountInfo]> {
  let summary = "ONNX Xor operation [v7]";
  let description = [{
      Returns the tensor resulted from performing the `xor` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  }];

  let arguments = (ins TensorOf<[I1]>:$A,
                       TensorOf<[I1]>:$B);
  let results = (outs TensorOf<[I1]>:$C);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Acos [v7]
def Onnx_Acos_V7Op : Onnx_Op<"Acos_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Acos_V7 operation [v7]";
  let description = [{
      Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Acos [v22]
def Onnx_AcosOp : Onnx_Op<"Acos", [Pure, OPCountInfo]> {
  let summary = "ONNX Acos operation [v22]";
  let description = [{
      Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pow [v1]
def Onnx_Pow_V1Op : Onnx_Op<"Pow_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Pow_V1 operation [v1]";
  let description = [{
      Pow takes input data (Tensor<T>) and exponent Tensor, and
  produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
  is applied to the data tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y,
                       OptionalAttr<I64Attr>:$axis,
                       DefaultValuedAttr<I64Attr, "0">:$broadcast);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Z);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pow [v7]
def Onnx_Pow_V7Op : Onnx_Op<"Pow_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Pow_V7 operation [v7]";
  let description = [{
      Pow takes input data (Tensor<T>) and exponent Tensor, and
  produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
  is applied to the data tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Z);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pow [v12]
def Onnx_Pow_V12Op : Onnx_Op<"Pow_V12", [Pure, OPCountInfo]> {
  let summary = "ONNX Pow_V12 operation [v12]";
  let description = [{
      Pow takes input data (Tensor<T>) and exponent Tensor, and
  produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
  is applied to the data tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let results = (outs AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Z);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pow [v13]
def Onnx_Pow_V13Op : Onnx_Op<"Pow_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Pow_V13 operation [v13]";
  let description = [{
      Pow takes input data (Tensor<T>) and exponent Tensor, and
  produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
  is applied to the data tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let results = (outs AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Z);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Pow [v15]
def Onnx_PowOp : Onnx_Op<"Pow", [Pure, OPCountInfo]> {
  let summary = "ONNX Pow operation [v15]";
  let description = [{
      Pow takes input data (Tensor<T>) and exponent Tensor, and
  produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
  is applied to the data tensor elementwise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y);
  let results = (outs AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Z);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Atan [v7]
def Onnx_Atan_V7Op : Onnx_Op<"Atan_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Atan_V7 operation [v7]";
  let description = [{
      Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Atan [v22]
def Onnx_AtanOp : Onnx_Op<"Atan", [Pure, OPCountInfo]> {
  let summary = "ONNX Atan operation [v22]";
  let description = [{
      Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cos [v7]
def Onnx_Cos_V7Op : Onnx_Op<"Cos_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Cos_V7 operation [v7]";
  let description = [{
      Calculates the cosine of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cos [v22]
def Onnx_CosOp : Onnx_Op<"Cos", [Pure, OPCountInfo]> {
  let summary = "ONNX Cos operation [v22]";
  let description = [{
      Calculates the cosine of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sin [v7]
def Onnx_Sin_V7Op : Onnx_Op<"Sin_V7", [Pure, OPCountInfo]> {
  let summary = "ONNX Sin_V7 operation [v7]";
  let description = [{
      Calculates the sine of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sin [v22]
def Onnx_SinOp : Onnx_Op<"Sin", [Pure, OPCountInfo]> {
  let summary = "ONNX Sin operation [v22]";
  let description = [{
      Calculates the sine of the given input tensor, element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Transpose [v1]
def Onnx_Transpose_V1Op : Onnx_Op<"Transpose_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Transpose_V1 operation [v1]";
  let description = [{
      Transpose the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$perm);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$transposed);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Transpose [v13]
def Onnx_Transpose_V13Op : Onnx_Op<"Transpose_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Transpose_V13 operation [v13]";
  let description = [{
      Transpose the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$perm);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$transposed);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Transpose [v21]
def Onnx_Transpose_V21Op : Onnx_Op<"Transpose_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Transpose_V21 operation [v21]";
  let description = [{
      Transpose the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$perm);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>:$transposed);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Transpose [v23]
def Onnx_TransposeOp : Onnx_Op<"Transpose", [Pure, OPCountInfo]> {
  let summary = "ONNX Transpose operation [v23]";
  let description = [{
      Transpose the input tensor similar to numpy.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$perm);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>:$transposed);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceProd [v1]
def Onnx_ReduceProd_V1Op : Onnx_Op<"ReduceProd_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceProd_V1 operation [v1]";
  let description = [{
      Computes the product of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceProd [v11]
def Onnx_ReduceProd_V11Op : Onnx_Op<"ReduceProd_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceProd_V11 operation [v11]";
  let description = [{
      Computes the product of the input tensor's element along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceProd [v13]
def Onnx_ReduceProd_V13Op : Onnx_Op<"ReduceProd_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceProd_V13 operation [v13]";
  let description = [{
      Computes the product of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// ReduceProd [v18]
def Onnx_ReduceProdOp : Onnx_Op<"ReduceProd", [Pure, OPCountInfo]> {
  let summary = "ONNX ReduceProd operation [v18]";
  let description = [{
      Computes the product of the input tensor's elements along the provided axes.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
                       DefaultValuedAttr<I64Attr, "1">:$keepdims,
                       DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Scan [v8]
def Onnx_Scan_V8Op : Onnx_Op<"Scan_V8", [Pure, OPCountInfo]> {
  let summary = "ONNX Scan_V8 operation [v8]";
  let description = [{
      Scan can be used to iterate over one or more scan_input tensors,
  constructing zero or more scan_output tensors.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$sequence_lens,
                       Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$initial_state_and_scan_inputs,
                       OptionalAttr<I64ArrayAttr>:$directions,
                       I64Attr:$num_scan_inputs);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$final_state_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Scan [v9]
def Onnx_Scan_V9Op : Onnx_Op<"Scan_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Scan_V9 operation [v9]";
  let description = [{
      Scan can be used to iterate over one or more scan_input tensors,
  constructing zero or more scan_output tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$initial_state_and_scan_inputs,
                       I64Attr:$num_scan_inputs,
                       OptionalAttr<I64ArrayAttr>:$scan_input_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_input_directions,
                       OptionalAttr<I64ArrayAttr>:$scan_output_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_output_directions);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$final_state_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Scan [v11]
def Onnx_Scan_V11Op : Onnx_Op<"Scan_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Scan_V11 operation [v11]";
  let description = [{
      Scan can be used to iterate over one or more scan_input tensors,
  constructing zero or more scan_output tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$initial_state_and_scan_inputs,
                       I64Attr:$num_scan_inputs,
                       OptionalAttr<I64ArrayAttr>:$scan_input_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_input_directions,
                       OptionalAttr<I64ArrayAttr>:$scan_output_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_output_directions);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$final_state_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Scan [v16]
def Onnx_Scan_V16Op : Onnx_Op<"Scan_V16", [Pure, OPCountInfo]> {
  let summary = "ONNX Scan_V16 operation [v16]";
  let description = [{
      Scan can be used to iterate over one or more scan_input tensors,
  constructing zero or more scan_output tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$initial_state_and_scan_inputs,
                       I64Attr:$num_scan_inputs,
                       OptionalAttr<I64ArrayAttr>:$scan_input_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_input_directions,
                       OptionalAttr<I64ArrayAttr>:$scan_output_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_output_directions);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$final_state_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Scan [v19]
def Onnx_Scan_V19Op : Onnx_Op<"Scan_V19", [Pure, OPCountInfo]> {
  let summary = "ONNX Scan_V19 operation [v19]";
  let description = [{
      Scan can be used to iterate over one or more scan_input tensors,
  constructing zero or more scan_output tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>>:$initial_state_and_scan_inputs,
                       I64Attr:$num_scan_inputs,
                       OptionalAttr<I64ArrayAttr>:$scan_input_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_input_directions,
                       OptionalAttr<I64ArrayAttr>:$scan_output_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_output_directions);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>>:$final_state_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Scan [v21]
def Onnx_Scan_V21Op : Onnx_Op<"Scan_V21", [Pure, OPCountInfo]> {
  let summary = "ONNX Scan_V21 operation [v21]";
  let description = [{
      Scan can be used to iterate over one or more scan_input tensors,
  constructing zero or more scan_output tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>>:$initial_state_and_scan_inputs,
                       I64Attr:$num_scan_inputs,
                       OptionalAttr<I64ArrayAttr>:$scan_input_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_input_directions,
                       OptionalAttr<I64ArrayAttr>:$scan_output_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_output_directions);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>]>>:$final_state_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Scan [v23]
def Onnx_ScanOp : Onnx_Op<"Scan", [Pure, OPCountInfo]> {
  let summary = "ONNX Scan operation [v23]";
  let description = [{
      Scan can be used to iterate over one or more scan_input tensors,
  constructing zero or more scan_output tensors.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>>:$initial_state_and_scan_inputs,
                       I64Attr:$num_scan_inputs,
                       OptionalAttr<I64ArrayAttr>:$scan_input_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_input_directions,
                       OptionalAttr<I64ArrayAttr>:$scan_output_axes,
                       OptionalAttr<I64ArrayAttr>:$scan_output_directions);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>>:$final_state_and_scan_outputs);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return -1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Compress [v9]
def Onnx_Compress_V9Op : Onnx_Op<"Compress_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Compress_V9 operation [v9]";
  let description = [{
      Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       TensorOf<[I1]>:$condition,
                       OptionalAttr<I64Attr>:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Compress [v11]
def Onnx_CompressOp : Onnx_Op<"Compress", [Pure, OPCountInfo]> {
  let summary = "ONNX Compress operation [v11]";
  let description = [{
      Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       TensorOf<[I1]>:$condition,
                       OptionalAttr<I64Attr>:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Scatter [v9]
def Onnx_Scatter_V9Op : Onnx_Op<"Scatter_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Scatter_V9 operation [v9]";
  let description = [{
      Given `data`, `updates` and `indices` input tensors of rank r >= 1, write the values provided by `updates`
  into the first input, `data`, along `axis` dimension of `data` (by default outer-most one as axis=0) at corresponding `indices`.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Scatter [v11]
def Onnx_ScatterOp : Onnx_Op<"Scatter", [Pure, OPCountInfo]> {
  let summary = "ONNX Scatter operation [v11]";
  let description = [{
      This operator is deprecated.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$updates,
                       DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// CenterCropPad [v18]
def Onnx_CenterCropPadOp : Onnx_Op<"CenterCropPad", [Pure, OPCountInfo]> {
  let summary = "ONNX CenterCropPad operation [v18]";
  let description = [{
      Center crop or pad an input to given dimensions.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input_data,
                       AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$shape,
                       OptionalAttr<I64ArrayAttr>:$axes);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output_data);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sinh [v9]
def Onnx_Sinh_V9Op : Onnx_Op<"Sinh_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Sinh_V9 operation [v9]";
  let description = [{
      Calculates the hyperbolic sine of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Sinh [v22]
def Onnx_SinhOp : Onnx_Op<"Sinh", [Pure, OPCountInfo]> {
  let summary = "ONNX Sinh operation [v22]";
  let description = [{
      Calculates the hyperbolic sine of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Asinh [v9]
def Onnx_Asinh_V9Op : Onnx_Op<"Asinh_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Asinh_V9 operation [v9]";
  let description = [{
      Calculates the hyperbolic arcsine of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Asinh [v22]
def Onnx_AsinhOp : Onnx_Op<"Asinh", [Pure, OPCountInfo]> {
  let summary = "ONNX Asinh operation [v22]";
  let description = [{
      Calculates the hyperbolic arcsine of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Trilu [v14]
def Onnx_TriluOp : Onnx_Op<"Trilu", [Pure, OPCountInfo]> {
  let summary = "ONNX Trilu operation [v14]";
  let description = [{
      Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$input,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$k,
                       DefaultValuedAttr<I64Attr, "1">:$upper);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Acosh [v9]
def Onnx_Acosh_V9Op : Onnx_Op<"Acosh_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Acosh_V9 operation [v9]";
  let description = [{
      Calculates the hyperbolic arccosine of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Acosh [v22]
def Onnx_AcoshOp : Onnx_Op<"Acosh", [Pure, OPCountInfo]> {
  let summary = "ONNX Acosh operation [v22]";
  let description = [{
      Calculates the hyperbolic arccosine of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cosh [v9]
def Onnx_Cosh_V9Op : Onnx_Op<"Cosh_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Cosh_V9 operation [v9]";
  let description = [{
      Calculates the hyperbolic cosine of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Cosh [v22]
def Onnx_CoshOp : Onnx_Op<"Cosh", [Pure, OPCountInfo]> {
  let summary = "ONNX Cosh operation [v22]";
  let description = [{
      Calculates the hyperbolic cosine of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// StringNormalizer [v10]
def Onnx_StringNormalizerOp : Onnx_Op<"StringNormalizer", [Pure, OPCountInfo]> {
  let summary = "ONNX StringNormalizer operation [v10]";
  let description = [{
      StringNormalization performs string operations for basic cleaning.
  }];

  let arguments = (ins TensorOf<[StringType]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NONE">:$case_change_action,
                       DefaultValuedAttr<I64Attr, "0">:$is_case_sensitive,
                       OptionalAttr<StrAttr>:$locale,
                       OptionalAttr<StrArrayAttr>:$stopwords);
  let results = (outs TensorOf<[StringType]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Atanh [v9]
def Onnx_Atanh_V9Op : Onnx_Op<"Atanh_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Atanh_V9 operation [v9]";
  let description = [{
      Calculates the hyperbolic arctangent of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Atanh [v22]
def Onnx_AtanhOp : Onnx_Op<"Atanh", [Pure, OPCountInfo]> {
  let summary = "ONNX Atanh operation [v22]";
  let description = [{
      Calculates the hyperbolic arctangent of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Shrink [v9]
def Onnx_ShrinkOp : Onnx_Op<"Shrink", [Pure, OPCountInfo]> {
  let summary = "ONNX Shrink operation [v9]";
  let description = [{
      Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
  having same datatype and shape with input.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<F32Attr, "0">:$bias,
                       DefaultValuedAttr<F32Attr, "0.5">:$lambd);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Hardmax [v1]
def Onnx_Hardmax_V1Op : Onnx_Op<"Hardmax_V1", [Pure, OPCountInfo]> {
  let summary = "ONNX Hardmax_V1 operation [v1]";
  let description = [{
      The operator computes the hardmax (1 for the first maximum value, and 0 for all others) values for each layer in the batch
  of the given input.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Hardmax [v11]
def Onnx_Hardmax_V11Op : Onnx_Op<"Hardmax_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Hardmax_V11 operation [v11]";
  let description = [{
      The operator computes the hardmax (1 for the first maximum value, and 0 for all others) values for each layer in the batch
  of the given input.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
                       DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Hardmax [v13]
def Onnx_HardmaxOp : Onnx_Op<"Hardmax", [Pure, OPCountInfo]> {
  let summary = "ONNX Hardmax operation [v13]";
  let description = [{
      The operator computes the hardmax values for the given input:
  Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise
  The "axis" attribute indicates the dimension along which Hardmax
  will be performed.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input,
                       DefaultValuedAttr<I64Attr, "-1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Erf [v9]
def Onnx_Erf_V9Op : Onnx_Op<"Erf_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Erf_V9 operation [v9]";
  let description = [{
      Computes the error function of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Erf [v13]
def Onnx_ErfOp : Onnx_Op<"Erf", [Pure, OPCountInfo]> {
  let summary = "ONNX Erf operation [v13]";
  let description = [{
      Computes the error function of the given input tensor element-wise.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 1;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Where [v9]
def Onnx_Where_V9Op : Onnx_Op<"Where_V9", [Pure, OPCountInfo]> {
  let summary = "ONNX Where_V9 operation [v9]";
  let description = [{
      Return elements, either from X or Y, depending on condition.
  }];

  let arguments = (ins TensorOf<[I1]>:$condition,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Where [v16]
def Onnx_WhereOp : Onnx_Op<"Where", [Pure, OPCountInfo]> {
  let summary = "ONNX Where operation [v16]";
  let description = [{
      Return elements, either from X or Y, depending on condition.
  }];

  let arguments = (ins TensorOf<[I1]>:$condition,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$output);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 3;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Resize [v10]
def Onnx_Resize_V10Op : Onnx_Op<"Resize_V10", [Pure, OPCountInfo]> {
  let summary = "ONNX Resize_V10 operation [v10]";
  let description = [{
      Resize the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       TensorOf<[F32]>:$scales,
                       DefaultValuedStrAttr<StrAttr, "nearest">:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 2;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Resize [v11]
def Onnx_Resize_V11Op : Onnx_Op<"Resize_V11", [Pure, OPCountInfo]> {
  let summary = "ONNX Resize_V11 operation [v11]";
  let description = [{
      Resize the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$roi,
                       TensorOf<[F32]>:$scales,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$sizes,
                       DefaultValuedStrAttr<StrAttr, "half_pixel">:$coordinate_transformation_mode,
                       DefaultValuedAttr<F32Attr, "-0.75">:$cubic_coeff_a,
                       DefaultValuedAttr<I64Attr, "0">:$exclude_outside,
                       DefaultValuedAttr<F32Attr, "0">:$extrapolation_value,
                       DefaultValuedStrAttr<StrAttr, "nearest">:$mode,
                       DefaultValuedStrAttr<StrAttr, "round_prefer_floor">:$nearest_mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Resize [v13]
def Onnx_Resize_V13Op : Onnx_Op<"Resize_V13", [Pure, OPCountInfo]> {
  let summary = "ONNX Resize_V13 operation [v13]";
  let description = [{
      Resize the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$roi,
                       AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$scales,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$sizes,
                       DefaultValuedStrAttr<StrAttr, "half_pixel">:$coordinate_transformation_mode,
                       DefaultValuedAttr<F32Attr, "-0.75">:$cubic_coeff_a,
                       DefaultValuedAttr<I64Attr, "0">:$exclude_outside,
                       DefaultValuedAttr<F32Attr, "0">:$extrapolation_value,
                       DefaultValuedStrAttr<StrAttr, "nearest">:$mode,
                       DefaultValuedStrAttr<StrAttr, "round_prefer_floor">:$nearest_mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Resize [v18]
def Onnx_Resize_V18Op : Onnx_Op<"Resize_V18", [Pure, OPCountInfo]> {
  let summary = "ONNX Resize_V18 operation [v18]";
  let description = [{
      Resize the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$roi,
                       AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$scales,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$sizes,
                       DefaultValuedAttr<I64Attr, "0">:$antialias,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedStrAttr<StrAttr, "half_pixel">:$coordinate_transformation_mode,
                       DefaultValuedAttr<F32Attr, "-0.75">:$cubic_coeff_a,
                       DefaultValuedAttr<I64Attr, "0">:$exclude_outside,
                       DefaultValuedAttr<F32Attr, "0">:$extrapolation_value,
                       DefaultValuedStrAttr<StrAttr, "stretch">:$keep_aspect_ratio_policy,
                       DefaultValuedStrAttr<StrAttr, "nearest">:$mode,
                       DefaultValuedStrAttr<StrAttr, "round_prefer_floor">:$nearest_mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}

/// Resize [v19]
def Onnx_ResizeOp : Onnx_Op<"Resize", [Pure, OPCountInfo]> {
  let summary = "ONNX Resize operation [v19]";
  let description = [{
      Resize the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$X,
                       AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$roi,
                       AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$scales,
                       AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$sizes,
                       DefaultValuedAttr<I64Attr, "0">:$antialias,
                       OptionalAttr<I64ArrayAttr>:$axes,
                       DefaultValuedStrAttr<StrAttr, "half_pixel">:$coordinate_transformation_mode,
                       DefaultValuedAttr<F32Attr, "-0.75">:$cubic_coeff_a,
                       DefaultValuedAttr<I64Attr, "0">:$exclude_outside,
                       DefaultValuedAttr<F32Attr, "0">:$extrapolation_value,
                       DefaultValuedStrAttr<StrAttr, "stretch">:$keep_aspect_ratio_policy,
                       DefaultValuedStrAttr<StrAttr, "nearest">:$mode,
                       DefaultValuedStrAttr<StrAttr, "round_prefer_floor">:$nearest_mode);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$Y);
  let extraClassDeclaration = [{
    int getDefinedOperandCount() {
      return 4;
    }
    int getDefinedResultCount() {
      return 1;
    }
  }];
}
