
# ONNX2MLIR

ONNX2MLIR dialect mappings for composable optimizations

![ONNX2MLIR](docs/logo/onnx2mlir-logo.png)

ONNX to MLIR is a graph converter for MLIR Linalg/Affine dialects with composable Transform optimizations.

---------------------------------------------------------------------------------------------------------

#### ONNX graph
```textproto
<
   ir_version: 11,
   opset_import: ["" : 23],
   producer_name: "onnx-example"
>
graph subtract_graph (
  %input_a[FLOAT, 2x3]
  %input_b[FLOAT, 2x1]
) {
  %output_c = Sub(%input_a, %input_b)
  return %output_c
}
```

#### MLIR onnx dialect
```mlir
module {
  func.func @main(
              %arg0: tensor<2x3xf32> {onnx.name = "input_a"},
              %arg1: tensor<2x1xf32> {onnx.name = "input_b"})
              -> (tensor<2x3xf32> {onnx.name = "output_c"}) {
    %0 = onnx.Sub(
            A = %arg0 : tensor<2x3xf32>
            B = %arg1 : tensor<2x1xf32>
            attributes {onnx.node.name = "/subtract0"}
        ) : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
```

#### MLIR linalg + affine + transform
```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @main(
              %arg0: tensor<2x3xf32> {onnx.name = "input_a"},
              %arg1: tensor<2x1xf32> {onnx.name = "input_b"})
              -> (tensor<2x3xf32> {onnx.name = "output_c"}) {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = linalg.elementwise kind=#linalg.elementwise_kind<sub>
            indexing_maps = [#map, #map1, #map]
            {transform.target_tag = "onnx.Sub"}
            ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x1xf32>)
            outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
```

---------------------------------------------------------------------------------------------------------

## Status

#### Overview
  * API design with booth: **python** (first) and **c++**
  * ONNX2MLIR ships reusable library for any ML/MLC project
  * ONNX2MLIR ships standalone tools as graph optimizers
  
#### The ONNX dialect
  * MLIR Onnx dialect is **self generated** out of onnx internal schema
  * MLIR Onnx dialect covers all ops including their **backward versions**
  * MLIR Onnx dialect is fully **assembled**, **parametrized** and **tracked**
  * MLIR Onnx dialect ops are **test covered** assureing no future regressions

#### The MLIR Linalg lowering backend
  * MLIR Onnx dialect gets lowered to **Linalg** (**TOSA** is considered WiP)
  * MLIR Linalg uses **Transform** dialect **tags** for further optimizations
  * MLIR lowered Ops are **covered by tests** against original onnx evaluators

#### The LLVM executor runtime
  * The executor runtime is **python first** and can lower via LLVM
  * The executor runtime use **reconfigurable passes** on lowering pipeline
  * The executor runtime can emmit reusable DSO for **any target machine**

---------------------------------------------------------------------------------------------------------

## WiP
  * Coverage of all ONNX ops lowering with on-line auto-generated status table
  * Add Transform schedules as templates, use knobs to control loops and layouts
  * Standardize transform templates to be fully shareable, reusable and configurable
  * Add schedule/knob tunner module possibly use xgboost (see halide/tvm)

## Motto

Keep it simple, it works the best !
