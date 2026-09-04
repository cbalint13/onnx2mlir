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
 * \file src/dialect/onnx/support/onnx_shapes.cpp
 * \brief Onnx dialect shapes inference implementation
 */

#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>
#include <onnx/onnx_pb.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

static void populateTensorProtoFromDenseAttr(mlir::DenseElementsAttr denseAttr,
                                             onnx::TensorProto &tensorProto) {
  auto shapedType = denseAttr.getType();
  int32_t dtype = MlirToOnnx_dType(shapedType.getElementType());
  tensorProto.set_data_type(dtype);

  for (int64_t dim : shapedType.getShape())
    tensorProto.add_dims(dim);

  auto rawData = denseAttr.getRawData();
  tensorProto.set_raw_data(rawData.data(), rawData.size());

  if (dtype == onnx::TensorProto_DataType_INT64) {
    for (auto val : denseAttr.getValues<mlir::APInt>())
      tensorProto.add_int64_data(val.getSExtValue());
  } else if (dtype == onnx::TensorProto_DataType_INT32) {
    for (auto val : denseAttr.getValues<mlir::APInt>())
      tensorProto.add_int32_data(val.getSExtValue());
  } else if (dtype == onnx::TensorProto_DataType_UINT64) {
    for (auto val : denseAttr.getValues<mlir::APInt>())
      tensorProto.add_uint64_data(val.getZExtValue());
  } else if (dtype == onnx::TensorProto_DataType_FLOAT) {
    for (auto val : denseAttr.getValues<mlir::APFloat>())
      tensorProto.add_float_data(val.convertToFloat());
  } else if (dtype == onnx::TensorProto_DataType_DOUBLE) {
    for (auto val : denseAttr.getValues<mlir::APFloat>())
      tensorProto.add_double_data(val.convertToDouble());
  } else if (dtype == onnx::TensorProto_DataType_STRING) {
    for (auto val : denseAttr.getValues<llvm::StringRef>())
      tensorProto.add_string_data(val.str());
  }
}

class MLIRInferenceContext : public onnx::InferenceContext {
public:
  explicit MLIRInferenceContext(mlir::Operation *op) : op_(op) {
    // input types
    inputs_.resize(op_->getNumOperands());
    for (size_t i = 0; i < op_->getNumOperands(); ++i) {
      mlir::Value inputVal = op_->getOperand(i);
      MlirToOnnx_dType(inputVal.getType(), inputs_[i]);

      mlir::Attribute cstAttr;
      if (mlir::matchPattern(inputVal, mlir::m_Constant(&cstAttr))) {
        if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(cstAttr)) {
          inputTensors_[i] = onnx::TensorProto();
          populateTensorProtoFromDenseAttr(denseAttr, inputTensors_[i]);
        }
      }
    }

    // output types
    outputs_.resize(op_->getNumResults());

    // attributes
    for (auto attr : op_->getAttrs()) {
      std::string name = attr.getName().str();
      onnx::AttributeProto attrProto;
      attrProto.set_name(name);

      auto attrValue = attr.getValue();
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attrValue)) {
        attrProto.set_type(onnx::AttributeProto_AttributeType_INT);
        attrProto.set_i(intAttr.getInt());
        attributes_[name] = attrProto;
      } else if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attrValue)) {
        attrProto.set_type(onnx::AttributeProto_AttributeType_FLOAT);
        attrProto.set_f(floatAttr.getValueAsDouble());
        attributes_[name] = attrProto;
      } else if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attrValue)) {
        attrProto.set_type(onnx::AttributeProto_AttributeType_STRING);
        attrProto.set_s(strAttr.getValue().str());
        attributes_[name] = attrProto;
      } else if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attrValue)) {
        if (!arrayAttr.empty()) {
          if (llvm::isa<mlir::IntegerAttr>(arrayAttr[0])) {
            attrProto.set_type(onnx::AttributeProto_AttributeType_INTS);
            for (auto elt : arrayAttr) {
              if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(elt))
                attrProto.add_ints(ia.getInt());
            }
            attributes_[name] = attrProto;
          } else if (llvm::isa<mlir::FloatAttr>(arrayAttr[0])) {
            attrProto.set_type(onnx::AttributeProto_AttributeType_FLOATS);
            for (auto elt : arrayAttr) {
              if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(elt))
                attrProto.add_floats(fa.getValueAsDouble());
            }
            attributes_[name] = attrProto;
          } else if (llvm::isa<mlir::StringAttr>(arrayAttr[0])) {
            attrProto.set_type(onnx::AttributeProto_AttributeType_STRINGS);
            for (auto elt : arrayAttr) {
              if (auto sa = mlir::dyn_cast<mlir::StringAttr>(elt))
                attrProto.add_strings(sa.getValue().str());
            }
            attributes_[name] = attrProto;
          }
        }
      }
    }
  }

  const onnx::AttributeProto *
  getAttribute(const std::string &name) const override {
    auto it = attributes_.find(name);
    if (it != attributes_.end())
      return &it->second;
    return nullptr;
  }

  size_t getNumInputs() const override { return inputs_.size(); }

  const onnx::TypeProto *getInputType(size_t index) const override {
    if (index < inputs_.size())
      return &inputs_[index];
    return nullptr;
  }

  const onnx::TensorProto *getInputData(size_t index) const override {
    auto it = inputTensors_.find(index);
    if (it != inputTensors_.end())
      return &it->second;
    return nullptr;
  }

  size_t getNumOutputs() const override { return outputs_.size(); }

  onnx::TypeProto *getOutputType(size_t index) override {
    if (index < outputs_.size())
      return &outputs_[index];
    return nullptr;
  }

  onnx::GraphInferencer *
  getGraphAttributeInferencer(const std::string &attribute_name) override {
    (void)attribute_name;
    return nullptr;
  }

  const onnx::SparseTensorProto *
  getInputSparseData(size_t index) const override {
    (void)index;
    return nullptr;
  }

  const onnx::TensorShapeProto *getSymbolicInput(size_t index) const override {
    (void)index;
    return nullptr;
  }

  void applyToMLIR() {
    mlir::MLIRContext *context = op_->getContext();
    for (size_t i = 0; i < op_->getNumResults(); ++i) {
      const auto &onnxType = outputs_[i];
      if (!onnxType.has_tensor_type())
        continue;

      const auto &tensorTypeProto = onnxType.tensor_type();
      auto elemType = OnnxToMlir_dType(tensorTypeProto.elem_type(), context);

      if (tensorTypeProto.has_shape()) {
        std::vector<int64_t> dims;
        const auto &shapeProto = tensorTypeProto.shape();
        for (int j = 0; j < shapeProto.dim_size(); ++j) {
          const auto &dim = shapeProto.dim(j);
          if (dim.has_dim_value())
            dims.push_back(dim.dim_value());
          else
            dims.push_back(mlir::ShapedType::kDynamic);
        }
        op_->getResult(i).setType(mlir::RankedTensorType::get(dims, elemType));
      } else {
        op_->getResult(i).setType(mlir::UnrankedTensorType::get(elemType));
      }
    }
  }

private:
  mlir::Operation *op_;
  std::vector<onnx::TypeProto> inputs_;
  std::vector<onnx::TypeProto> outputs_;
  std::unordered_map<size_t, onnx::TensorProto> inputTensors_;
  std::unordered_map<std::string, onnx::AttributeProto> attributes_;
};

namespace onnx2mlir::dialect::onnx {

void inferONNXOpShape(mlir::Operation *op, llvm::StringRef onnxOpName,
                      int opsetVersion) {
  const auto *schema =
      ::onnx::OpSchemaRegistry::Schema(onnxOpName.str(), opsetVersion);
  if (schema && schema->has_type_and_shape_inference_function()) {
    try {
      MLIRInferenceContext ctx(op);
      schema->GetTypeAndShapeInferenceFunction()(ctx);
      ctx.applyToMLIR();
    } catch (const std::exception &e) {
      onnx2mlir::error() << "ONNX shape inference failed for op '" << onnxOpName
                         << "': " << e.what() << "\n";
    }
  }
}

} // namespace onnx2mlir::dialect::onnx
