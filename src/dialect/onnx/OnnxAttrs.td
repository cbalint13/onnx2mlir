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
 * \file src/dialect/onnx/OnnxAttr.td
 * \brief Onnx dialect attributes definition
 */

#ifndef ONNX2MLIR_FRONTEND_ONNXATTR_TD_
#define ONNX2MLIR_FRONTEND_ONNXATTR_TD_

include "mlir/IR/AttrTypeBase.td"

class BaseLayoutAttr<Dialect dialect, string name, list<Trait> traits = []>
    : AttrDef<dialect, name, traits> {
  let mnemonic = "layout";
}

class Onnx_LayoutAttr<string name, list<Trait> traits = []>
    : BaseLayoutAttr<Onnx_Dialect, name, traits>;

def LayoutAttr : Attr<
  CPred<"mlir::isa<::mlir::Attribute>($_self) ">,
        //"&& mlir::cast<Attribute>($_self).getMnemonic().equal(\"layout\")">,
  "layout attribute"
>;

def OnnxTensorEncodingAttr : Onnx_LayoutAttr<"OnnxTensorEncoding"> {
  let hasCustomAssemblyFormat = 1;
  let description = [{
    An attribute to encode information on properties of ONNX tensors. A tensor with
    this encoding attribute indicates that it is a tensor whose type is
    the original type of the pre-transformed custom-layout data.

    `dataLayout` indicates the data layout of the pre-transformed data.

    `x` indicates the tiling factor for the first tiled dimension. Typical values
    are multiples of the SIMD vector length of the target machine.

    `y` indicates the tiling factor for the first tiled dimension. Typical values
    are multiples of the SIMD vector length of the target machine. For layout
    with only one tiling dimension, `y` should be zero.

    Example:

    ```mlir
    #ONNX_NCHWxC= #onnx.encoding<{
      dataLayout = "NCHWxC",
      xFactor = 4,
      yFactor = 0
    }>

    ... tensor<8x8x128x128xf32, #ONNX_NCHWxC> ...
    ```
  }];

  // Data in ONNX tensor encoding.
  let parameters = (ins
    // A original data layout of the pre-transformed custom-layout data.
    "OnnxTensorEncodingAttr::DataLayout":$dataLayout,
    "int64_t":$xFactor, // typical is vector length (VL).
    "int64_t":$yFactor  // typical is VL when used, 0 otherwise.
  );

  let extraClassDeclaration = [{
    enum class DataLayout {
      STANDARD, // Standard ONNX layout as defined by the specs.
      NCHWxC,   // Layout for conv image, with C=Cin tiled by x.
      KCNMxCyK  // Layout for conv weights, with C=Cin & K=Cout tiled by x & y.
    };
  }];

  //let cppNamespace = "::mlir";
}

#endif // ONNX2MLIR_FRONTEND_ONNXATTR_TD_
