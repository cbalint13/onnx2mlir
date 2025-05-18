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
 * \file src/frontend/importers/onnx.cpp
 *
 */

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>

#include <onnx/common/constants.h>
#include <onnx/common/version.h>
#include <onnx/defs/operator_sets.h>
#include <onnx/defs/schema.h>
#include <onnx/shape_inference/implementation.h>
#include <onnx/version_converter/convert.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
#include "onnx2mlir/dialect/onnx/OnnxInterface.hpp"
#include "onnx2mlir/dialect/onnx/OnnxOps.hpp"
#include "onnx2mlir/frontend/onnx.hpp"

static mlir::Type OnnxToMlir_dType(const int32_t data_type_int,
                                   mlir::MLIRContext *ctx) {
  auto onnx_type = static_cast<onnx::TensorProto::DataType>(data_type_int);

  switch (onnx_type) {
  case onnx::TensorProto::FLOAT:
    return mlir::Float32Type::get(ctx);
  case onnx::TensorProto::INT4:
    return mlir::IntegerType::get(ctx, 4);
  case onnx::TensorProto::INT8:
    return mlir::IntegerType::get(ctx, 8);
  case onnx::TensorProto::INT16:
    return mlir::IntegerType::get(ctx, 16);
  case onnx::TensorProto::INT32:
    return mlir::IntegerType::get(ctx, 32);
  case onnx::TensorProto::INT64:
    return mlir::IntegerType::get(ctx, 64);
  case onnx::TensorProto::BOOL:
    return mlir::IntegerType::get(ctx, 1);
  case onnx::TensorProto::UINT4:
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::UINT8:
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::UINT16:
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::UINT32:
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::UINT64:
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::STRING:
    return mlir::NoneType::get(ctx);
  case onnx::TensorProto::FLOAT16:
    return mlir::Float16Type::get(ctx);
  case onnx::TensorProto::DOUBLE:
    return mlir::Float64Type::get(ctx);
  case onnx::TensorProto::BFLOAT16:
    return mlir::BFloat16Type::get(ctx);
  case onnx::TensorProto::FLOAT8E4M3FN:
    return mlir::Float8E4M3FNType::get(ctx);
  case onnx::TensorProto::FLOAT8E4M3FNUZ:
    return mlir::Float8E4M3FNUZType::get(ctx);
  case onnx::TensorProto::FLOAT8E5M2:
    return mlir::Float8E5M2Type::get(ctx);
  case onnx::TensorProto::FLOAT8E5M2FNUZ:
    return mlir::Float8E5M2FNUZType::get(ctx);
  case onnx::TensorProto::FLOAT4E2M1:
    return mlir::Float4E2M1FNType::get(ctx);
  case onnx::TensorProto::COMPLEX64:
    return mlir::ComplexType::get(mlir::Float32Type::get(ctx));
  case onnx::TensorProto::COMPLEX128:
    return mlir::ComplexType::get(mlir::Float64Type::get(ctx));
  case onnx::TensorProto::UNDEFINED:
    return mlir::NoneType::get(ctx);
  default:
    llvm::errs() << "ERROR: Unknown ONNX data type integer value: "
                 << data_type_int << "\n";
    exit(-1);
  }
}

template <typename shp_T, typename typ_T>
static mlir::DenseElementsAttr
getMlirTensor(const std::string &data, shp_T shape, typ_T dType,
              const mlir::Attribute &eAttr = {}) {
  auto dims = llvm::ArrayRef(shape.data(), shape.size());
  auto shapedType = mlir::RankedTensorType::get(dims, dType, eAttr);
  auto denseAttrs = mlir::DenseElementsAttr::getFromRawBuffer(
      shapedType, llvm::ArrayRef(data.data(), data.size()));
  return denseAttrs;
}

template <typename shp_T, typename typ_T, typename Container>
static mlir::DenseElementsAttr
getMlirTensor(const Container &data, shp_T shape, typ_T dType,
              const mlir::Attribute &eAttr = {}) {
  using dat_T = typename Container::value_type;
  auto dims = llvm::ArrayRef(shape.data(), shape.size());
  auto shapedType = mlir::RankedTensorType::get(dims, dType, eAttr);
  auto denseAttrs = mlir::DenseElementsAttr::get(
      shapedType, llvm::ArrayRef<dat_T>(data.data(), data.size()));
  return denseAttrs;
}

template <typename Container>
static mlir::ArrayAttr getMlirArray(mlir::MLIRContext *ctx,
                                    const Container &data) {
  using dat_T = typename Container::value_type;
  llvm::SmallVector<mlir::Attribute> attrVec;
  for (const dat_T &value : data) {
    if constexpr (std::is_same_v<dat_T, int64_t>)
      attrVec.push_back(
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), value));
    else if constexpr (std::is_same_v<dat_T, float>)
      attrVec.push_back(
          mlir::FloatAttr::get(mlir::Float32Type::get(ctx), value));
    else if constexpr (std::is_same_v<dat_T, std::string>)
      attrVec.push_back(mlir::StringAttr::get(ctx, value));
    else {
      llvm::errs() << "ERROR: unimplemented array type requested.\n";
      exit(-1);
    }
  }
  return mlir::ArrayAttr::get(ctx, attrVec);
}

static mlir::ElementsAttr OnnxToMlir_Tensor(const onnx::TensorProto &tensor,
                                            mlir::MLIRContext *ctx,
                                            const mlir::Attribute &eAttr = {}) {

  auto dType = OnnxToMlir_dType(tensor.data_type(), ctx);

  if (tensor.has_raw_data()) {
    switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT16:
    case onnx::TensorProto::FLOAT:
    case onnx::TensorProto::DOUBLE:
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::INT4:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::INT64:
    case onnx::TensorProto::UINT4:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT16:
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
      return getMlirTensor(tensor.raw_data(), tensor.dims(), dType, eAttr);
    default:
      llvm::errs() << "ERROR: Raw data read not supported for this type.\n";
      exit(-1);
    }
  } else {
    switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT:
      return getMlirTensor(tensor.float_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::DOUBLE:
      return getMlirTensor(tensor.double_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::INT4:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::UINT4:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT16:
      return getMlirTensor(tensor.int32_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::INT64:
      return getMlirTensor(tensor.int64_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
      return getMlirTensor(tensor.uint64_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::STRING:
    // TODO STRING
    case onnx::TensorProto::UNDEFINED:
    default:
      llvm::errs() << "ERROR: Data read not supported for this type.\n";
      exit(-1);
    }
  }
  return nullptr;
}

static mlir::Type OnnxToMlir_Type(const onnx::ValueInfoProto &value_proto,
                                  mlir::MLIRContext *ctx) {
  const auto &type_proto = value_proto.type();

  // value data type
  switch (type_proto.value_case()) {
  case onnx::TypeProto::kTensorType:
  case onnx::TypeProto::kSparseTensorType: {
    std::vector<int64_t> dataShape;
    const auto &tensor_type = type_proto.tensor_type();
    auto elemType = OnnxToMlir_dType(tensor_type.elem_type(), ctx);

    if (tensor_type.has_shape()) {
      for (int i = 0; i < tensor_type.shape().dim_size(); ++i) {
        const auto &dim = tensor_type.shape().dim(i);
        if (dim.has_dim_value()) {
          dataShape.push_back(dim.dim_value());
        } else if (dim.has_dim_param()) {
          dataShape.push_back(mlir::ShapedType::kDynamic);
        } else {
          llvm::errs() << "ERROR: Tensor has invalid dimension.\n";
          exit(-1);
        }
      }
    } else {
      llvm::errs() << "ERROR: Tensor has no shape.\n";
      exit(-1);
    }
    return mlir::RankedTensorType::get(dataShape, elemType);
  }
  case onnx::TypeProto::kSequenceType:
  case onnx::TypeProto::kMapType:
  case onnx::TypeProto::kOptionalType:
  case onnx::TypeProto::VALUE_NOT_SET:
  default:
    llvm::errs() << "ERROR: TypeProto is unsupported.\n";
    exit(-1);
  }
}

static std::optional<mlir::NamedAttribute>
OnnxToMlir_Attr(const onnx::AttributeProto &attribute, mlir::MLIRContext *ctx,
                const mlir::Attribute &eAttr = {}) {

  mlir::OpBuilder builder(ctx);

  switch (attribute.type()) {
  case onnx::AttributeProto::FLOAT:
    return mlir::NamedAttribute(
        attribute.name(),
        mlir::FloatAttr::get(mlir::Float32Type::get(ctx), attribute.f()));
  case onnx::AttributeProto::INT:
    return mlir::NamedAttribute(
        attribute.name(),
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), attribute.i()));
  case onnx::AttributeProto::STRING:
    return mlir::NamedAttribute(attribute.name(),
                                mlir::StringAttr::get(ctx, attribute.s()));
  case onnx::AttributeProto::TENSOR:
    return mlir::NamedAttribute(attribute.name(),
                                OnnxToMlir_Tensor(attribute.t(), ctx, eAttr));
  case onnx::AttributeProto::GRAPH:
    llvm::errs() << "ERROR: Parsing GRAPH attribute type is not implemented.\n";
    exit(-1);
  case onnx::AttributeProto::FLOATS:
    return mlir::NamedAttribute(attribute.name(),
                                getMlirArray(ctx, attribute.floats()));
  case onnx::AttributeProto::INTS:
    return mlir::NamedAttribute(attribute.name(),
                                getMlirArray(ctx, attribute.ints()));
  case onnx::AttributeProto::STRINGS:
    return mlir::NamedAttribute(attribute.name(),
                                getMlirArray(ctx, attribute.strings()));
  case onnx::AttributeProto::TENSORS: {
    llvm::SmallVector<mlir::Attribute> attrVec;
    for (int i = 0; i < attribute.tensors_size(); ++i)
      attrVec.push_back(OnnxToMlir_Tensor(attribute.tensors(i), ctx, eAttr));
    return mlir::NamedAttribute(attribute.name(),
                                mlir::ArrayAttr::get(ctx, attrVec));
  }
  case onnx::AttributeProto::GRAPHS:
    llvm::errs()
        << "ERROR: Parsing GRAPHS attribute type is not implemented.\n";
    exit(-1);
  case onnx::AttributeProto::SPARSE_TENSOR:
    // TODO
    break;
  case onnx::AttributeProto::SPARSE_TENSORS:
    // TODO
    for (int i = 0; i < attribute.sparse_tensors_size(); ++i) {
    }
    break;
  case onnx::AttributeProto::TYPE_PROTO:
    llvm::errs()
        << "ERROR: Parsing TYPE_PROTO attribute type is not implemented.\n";
    exit(-1);
  case onnx::AttributeProto::TYPE_PROTOS:
    llvm::errs()
        << "ERROR: Parsing TYPE_PROTOS attribute type is not implemented.\n";
    exit(-1);
  case onnx::AttributeProto::UNDEFINED:
    llvm::errs()
        << "ERROR: Parsing UNDEFINED attribute type is not implemented.\n";
    exit(-1);
  default:
    llvm::errs() << "ERROR: Parsing unknown attribute type.\n";
    exit(-1);
  }

  return std::nullopt;
}

static int getOnnxOpNumIO(const std::string &opName,
                          const bool dir_out = false) {

  int nIO;
  mlir::MLIRContext ctx;
  mlir::OpBuilder builder(&ctx);
  ctx.loadDialect<onnx2mlir::dialect::onnx::OnnxDialect>();

  mlir::OperationState state(builder.getUnknownLoc(), "onnx." + opName);
  auto op = builder.create(state);
  if (dir_out)
    nIO = mlir::cast<onnx2mlir::dialect::onnx::OPCountInfo>(op)
              .getDefinedResultCount();
  else
    nIO = mlir::cast<onnx2mlir::dialect::onnx::OPCountInfo>(op)
              .getDefinedOperandCount();
  op->erase();

  return nIO;
}

static bool checkOnnxOpExists(mlir::MLIRContext *ctx,
                              const std::string &opName) {
  if (!mlir::OperationName("onnx." + opName, ctx).isRegistered())
    return false;
  return true;
}

static mlir::Operation *
createOnnxOp(mlir::OpBuilder &builder, const std::string &opName,
             const std::vector<mlir::Type> &types = {},
             const std::vector<mlir::Value> &values = {},
             const std::vector<mlir::NamedAttribute> &attrs = {}) {
  // setup operation
  mlir::OperationState state(builder.getUnknownLoc(), "onnx." + opName);

  for (auto type : types)
    state.addTypes(type);
  for (auto value : values)
    state.addOperands(value);
  for (auto attr : attrs)
    state.addAttributes(attr);

  mlir::Operation *op = builder.create(state);

  return op;
}

namespace onnx2mlir::frontend {

/*
 *  ONNXImporter class
 */

ONNXImporter::ONNXImporter() {
  // context setup
  mlirCtx->loadDialect<mlir::func::FuncDialect,
                       onnx2mlir::dialect::onnx::OnnxDialect>();
  mlirCtx->disableMultithreading();
  // initialize the module
  mlir::OpBuilder builder(mlirCtx.get());
  module = mlir::ModuleOp::create(builder.getUnknownLoc());
}

void ONNXImporter::parse_graph_nodes(const onnx::GraphProto &graph_proto) {

  // i/o map storage
  std::map<std::string, std::shared_ptr<onnx::ValueInfoProto>> vis;

  // map inferred value infos
  for (const auto &vi : graph_proto.value_info()) {
    auto vi_ptr = std::make_shared<onnx::ValueInfoProto>(vi);
    vis.insert({vi.name(), vi_ptr});
  }

  /*
   * Main function
   */

  // get main func
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  // add func body
  auto block = func.addEntryBlock();

  // args storage
  std::map<std::string, std::shared_ptr<mlir::Value>> func_inputs;
  std::map<std::string, std::shared_ptr<mlir::Type>> func_outputs;

  // map func inputs argument
  for (unsigned int i = 0; i < func.getNumArguments(); i++) {
    mlir::Value arg = block->getArgument(i);
    auto attr = func.getArgAttrOfType<mlir::StringAttr>(i, "onnx.name");
    if (attr.size()) {
      auto arg_ptr = std::make_shared<mlir::Value>(arg);
      func_inputs.insert({attr.getValue().str(), arg_ptr});
    }
  }

  // map func results argument
  for (unsigned int i = 0; i < func.getNumResults(); i++) {
    mlir::Type res = func.getFunctionType().getResult(i);
    auto attr = func.getResultAttrOfType<mlir::StringAttr>(i, "onnx.name");
    if (attr.size()) {
      auto res_ptr = std::make_shared<mlir::Type>(res);
      func_outputs.insert({attr.getValue().str(), res_ptr});
    }
  }

  /*
   * Build the MLIR graph
   */

  mlir::OpBuilder builder(mlirCtx.get());

  // map mlir ops i/o
  std::map<std::string, std::shared_ptr<mlir::Operation *>> ops_by_name;
  std::multimap<std::string, std::shared_ptr<mlir::Operation *>> ops_by_inputs;
  std::multimap<std::string, std::shared_ptr<mlir::Operation *>> ops_by_outputs;

  // add constant nodes
  for (const auto &node : graph_proto.node()) {

    if (!checkOnnxOpExists(mlirCtx.get(), node.op_type())) {
      llvm::errs() << "ERROR: operation [" << node.op_type()
                   << "] not registered.\n";
      exit(-1);
    }

    if (node.op_type() == "Constant") {
      // attributes
      std::vector<mlir::NamedAttribute> attrs;
      auto attr = OnnxToMlir_Attr(node.attribute()[0], mlirCtx.get());
      attrs.push_back(*attr);
      // origin
      auto nameVal = builder.getStringAttr(node.name());
      mlir::NamedAttribute label("onnx.node.name", nameVal);
      attrs.push_back(label);
      // result type
      auto types = std::vector<mlir::Type>(
          {mlir::dyn_cast<mlir::DenseElementsAttr>(attr->getValue())
               .getType()});

      auto op = createOnnxOp(builder, "Constant", types, {}, attrs);

      block->push_back(op);
      // store output
      auto res_ptr = std::make_shared<mlir::Operation *>(op);
      ops_by_outputs.insert({node.output()[0], res_ptr});
    }
  }

  // add the initializers
  for (const auto &initializer : graph_proto.initializer()) {
    // attributes
    std::vector<mlir::NamedAttribute> attrs;
    auto value = OnnxToMlir_Tensor(initializer, mlirCtx.get());
    mlir::NamedAttribute attr("value", value);
    attrs.push_back(attr);
    // origin
    auto nameVal = builder.getStringAttr(initializer.name());
    mlir::NamedAttribute labels("onnx.init.name", nameVal);
    attrs.push_back(labels);
    // result type
    auto types = std::vector<mlir::Type>({value.getType()});

    auto op = createOnnxOp(builder, "Constant", types, {}, attrs);

    block->push_back(op);
    // map to i/o
    auto res_ptr = std::make_shared<mlir::Operation *>(op);
    ops_by_outputs.insert({initializer.name(), res_ptr});
  }

  // add NoneType constant
  mlir::Operation *notype;
  {
    std::vector<mlir::NamedAttribute> attrs;
    // origin
    auto nameVal = builder.getStringAttr("NoneType");
    mlir::NamedAttribute labels("mlir.value", nameVal);
    attrs.push_back(labels);
    // result type
    mlir::Type noneType = mlir::NoneType::get(mlirCtx.get());
    auto types = std::vector<mlir::Type>({noneType});

    notype = createOnnxOp(builder, "Constant", types, {}, attrs);
    block->push_back(notype);
  }

  // add nodes
  for (const auto &node : graph_proto.node()) {
    if (node.op_type() != "Constant") {
      // attributes
      std::vector<mlir::NamedAttribute> attrs;
      for (const auto &attribute : node.attribute()) {
        auto attr = OnnxToMlir_Attr(attribute, mlirCtx.get());
        attrs.push_back(*attr);
      }
      // origin
      auto nameVal = builder.getStringAttr(node.name());
      mlir::NamedAttribute labels("onnx.node.name", nameVal);
      attrs.push_back(labels);
      // result types
      std::vector<mlir::Type> types;
      const int nResults = getOnnxOpNumIO(node.op_type(), /*outputs*/ true);
      for (int i = 0; i < nResults; ++i) {
        // unmapped outputs
        if (i >= node.output().size()) {
          types.push_back(mlir::NoneType::get(mlirCtx.get()));
          continue;
        }

        // node output
        const auto out = node.output()[i];

        // lookup in vis table
        auto it = vis.find(out);
        if (it != vis.end()) {
          auto type = OnnxToMlir_Type(*(it->second), mlirCtx.get());
          types.push_back(type);
        } else {
          // lookup neighbour outputs
          auto oitr = ops_by_inputs.find(out);
          if (oitr != ops_by_inputs.end()) {
            auto oadj = *oitr->second;
            oadj->print(llvm::errs());
            // TODO which result ?!
            types.push_back(oadj->getResults()[0].getType());
          }
          // lookup main func outputs
          auto fitr = func_outputs.find(out);
          if (fitr != func_outputs.end()) {
            auto fadj = *fitr->second;
            fadj.print(llvm::errs());
            types.push_back(fadj);
          }
        }
      }

      // add operands (temporary)
      std::vector<mlir::Value> operands;
      int nOperands = getOnnxOpNumIO(node.op_type(), /*inputs*/ false);
      // variadic case
      if (nOperands < 0)
        nOperands = node.input().size();
      for (int i = 0; i < nOperands; ++i) {
        operands.push_back(notype->getResult(0));
      }

      auto op = createOnnxOp(builder, node.op_type(), types, operands, attrs);
      block->push_back(op);

      // map to i/o
      auto op_ptr = std::make_shared<mlir::Operation *>(op);
      for (const auto &inp : node.input())
        ops_by_inputs.insert({inp, op_ptr});
      for (const auto &out : node.output())
        ops_by_outputs.insert({out, op_ptr});
      ops_by_name.insert({node.name(), op_ptr});
    }
  }

  // set nodes inputs
  for (const auto &node : graph_proto.node()) {

    // skip on no inputs
    if (node.input().size() == 0)
      continue;

    auto node_op = *ops_by_name[node.name()];

    // set all operands inputs
    for (int idx = 0; idx < node.input().size(); ++idx) {
      // unused
      if (node.input()[idx].size() == 0)
        continue;
      // map node input to main func input
      auto f_it = func_inputs.find(node.input()[idx]);
      if (f_it != func_inputs.end()) {
        mlir::Value arg = *(f_it->second);
        node_op->setOperand(idx, arg);
      }
      // map node input to upper node output
      auto o_it = ops_by_outputs.find(node.input()[idx]);
      if (o_it != ops_by_outputs.end()) {
        auto op = *(o_it->second);
        mlir::Value arg = op->getOpResult(0);
        node_op->setOperand(idx, arg);
        continue;
      }
    }
  }

  // map func outputs
  mlir::SmallVector<mlir::Value> ret_values;
  for (const auto &out : func_outputs) {
    auto o_it = ops_by_outputs.find(out.first);
    if (o_it != ops_by_outputs.end()) {
      auto op = *(o_it->second);
      mlir::Value val = op->getOpResult(0);
      ret_values.push_back(val);
    }
  }
  auto ret =
      builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), ret_values);
  block->push_back(ret);

  // remove NoType Constant if unused
  if (notype->getResults()[0].use_empty())
    notype->erase();
}

void ONNXImporter::parse_graph_io(const onnx::GraphProto &graph_proto) {

  std::vector<mlir::Type> inputs;

  // main inputs
  for (const auto &input : graph_proto.input()) {
    if (input.has_type()) {
      inputs.push_back(OnnxToMlir_Type(input, mlirCtx.get()));
    } else {
      llvm::errs() << "ERROR: Type Not Specified.\n";
      exit(-1);
    }
  }

  std::vector<mlir::Type> outputs;

  // main outputs
  for (const auto &output : graph_proto.output()) {
    if (output.has_type()) {
      outputs.push_back(OnnxToMlir_Type(output, mlirCtx.get()));
    } else {
      llvm::errs() << "ERROR: Type Not Specified.\n";
      exit(-1);
    }
  }

  // mlir graph main function
  mlir::OpBuilder builder(mlirCtx.get());
  auto funcType = mlir::FunctionType::get(mlirCtx.get(), inputs, outputs);
  auto func = mlir::func::FuncOp::create(builder.getUnknownLoc(), "main",
                                         funcType, /*attr*/ {});

  // main function args attribute
  for (int i = 0; i < graph_proto.input().size(); i++) {
    auto attr = builder.getStringAttr(graph_proto.input()[i].name());
    func.setArgAttr(i, "onnx.name", attr);
  }

  // main function results attribute
  for (int i = 0; i < graph_proto.output().size(); i++) {
    auto attr = builder.getStringAttr(graph_proto.output()[i].name());
    func.setResultAttr(i, "onnx.name", attr);
  }

  module->push_back(func);
}

void ONNXImporter::import(const std::string &filepath) {

  llvm::outs() << "ONNX engine version: " << onnx::LAST_RELEASE_VERSION << "\n";
  llvm::outs() << "ONNX engine IR version: " << onnx::IR_VERSION << "\n";

  int last_op_version = -1;
  auto all_schemas = onnx::OpSchemaRegistry::get_all_schemas();
  for (const auto &schema : all_schemas) {
    if (last_op_version < schema.SinceVersion())
      last_op_version = schema.SinceVersion();
  }

  llvm::outs() << "ONNX engine last opset version: " << last_op_version << "\n";

  std::ifstream model_file(filepath, std::ios::binary);

  if (!model_file.is_open()) {
    llvm::errs() << "Error opening file: " << filepath << "\n";
    exit(-1);
  }

  onnx::ModelProto model_import;
  /// parse onnx binary
  if (!model_import.ParseFromIstream(&model_file)) {
    llvm::errs() << "ERROR: ONNX model file parsing error.\n";
    exit(-1);
  }

  int OpSetVer = -1;
  /// see https://github.com/onnx/onnx/blob/main/onnx/docs/Versioning.md
  for (auto it = model_import.opset_import().begin();
       it != model_import.opset_import().end(); ++it) {
    if (it->domain() == "" || it->domain() == "ai.onnx") {
      OpSetVer = it->version();
      break;
    }
  }
  llvm::outs() << "\n";
  llvm::outs() << "Model IR version: " << model_import.ir_version() << "\n";
  llvm::outs() << "Model OPSET conversion: " << OpSetVer << " -> "
               << last_op_version << "\n";
  llvm::outs() << "\n";

  /// convert model
  auto model_proto =
      onnx::version_conversion::ConvertVersion(model_import, last_op_version);
  /// infer shapes
  onnx::shape_inference::InferShapes(model_proto);

  const onnx::GraphProto &graph_proto = model_proto.graph();
  llvm::outs() << "Graph Name: " << graph_proto.name() << "\n";

  /*
   * MLIR ONNX
   */

  // construct function args
  parse_graph_io(graph_proto);

  // pupulate body operators
  parse_graph_nodes(graph_proto);

  // verify module
  if (llvm::failed(mlir::verify(*module))) {
    llvm::errs() << "MLIR module verification failed.\n";
    // exit(-1);
  }

  // DEBUG
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  llvm::outs().enable_colors(true);
  module->print(llvm::outs(), flags);
  llvm::outs().enable_colors(false);
}

} // end namespace onnx2mlir::frontend
