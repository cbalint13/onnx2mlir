
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>

#include "llvm/Support/Casting.h"

#include "onnx/common/constants.h"
#include <onnx/common/version.h>
#include <onnx/defs/schema.h>
#include <onnx/defs/operator_sets.h>
#include <onnx/shape_inference/implementation.h>
#include <onnx/version_converter/convert.h>

#include <any>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
#include "onnx2mlir/dialect/onnx/OnnxInterface.hpp"
#include "onnx2mlir/dialect/onnx/OnnxOps.hpp"
#include "onnx2mlir/frontend/onnx.hpp"

static std::string
onnx_attrtype_tostr(const onnx::AttributeProto_AttributeType attr_type) {
  switch (attr_type) {
  case onnx::AttributeProto::FLOAT:
    return "FLOAT";
  case onnx::AttributeProto::INT:
    return "INT";
  case onnx::AttributeProto::STRING:
    return "STRING";
  case onnx::AttributeProto::TENSOR:
    return "TENSOR";
  case onnx::AttributeProto::UNDEFINED:
    return "UNDEFINED";
  case onnx::AttributeProto::SPARSE_TENSOR:
    return "SPARSE-TENSOR";
  case onnx::AttributeProto::TYPE_PROTO:
    return "TYPE_PROTO";
  case onnx::AttributeProto::FLOATS:
    return "FLOATS";
  case onnx::AttributeProto::INTS:
    return "INTS";
  case onnx::AttributeProto::STRINGS:
    return "STRING";
  case onnx::AttributeProto::TENSORS:
    return "TENSORS";
  case onnx::AttributeProto::GRAPHS:
    return "GRAPHS";
  case onnx::AttributeProto::SPARSE_TENSORS:
    return "SPARSE_TENSORS";
  case onnx::AttributeProto::TYPE_PROTOS:
    return "TYPE_PROTOS";
  case onnx::AttributeProto::GRAPH:
    return "GRAPHS";
  default:
    return "UNKNOWN";
  }
}

static std::string onnx_datatype_tostr(const int32_t data_type_int) {
  auto data_type = static_cast<onnx::TensorProto::DataType>(data_type_int);

  switch (data_type) {
  case onnx::TensorProto::FLOAT:
    return "FLOAT";
  case onnx::TensorProto::UINT8:
    return "UINT8";
  case onnx::TensorProto::INT8:
    return "INT8";
  case onnx::TensorProto::UINT16:
    return "UINT16";
  case onnx::TensorProto::INT16:
    return "INT16";
  case onnx::TensorProto::INT32:
    return "INT32";
  case onnx::TensorProto::INT64:
    return "INT64";
  case onnx::TensorProto::STRING:
    return "STRING";
  case onnx::TensorProto::BOOL:
    return "BOOL";
  case onnx::TensorProto::FLOAT16:
    return "FLOAT16";
  case onnx::TensorProto::DOUBLE:
    return "DOUBLE";
  case onnx::TensorProto::UINT32:
    return "UINT32";
  case onnx::TensorProto::UINT64:
    return "UINT64";
  case onnx::TensorProto::COMPLEX64:
    return "COMPLEX64";
  case onnx::TensorProto::COMPLEX128:
    return "COMPLEX128";
  case onnx::TensorProto::BFLOAT16:
    return "BFLOAT16";
  case onnx::TensorProto::FLOAT8E4M3FN:
    return "FLOAT8E4M3FN";
  case onnx::TensorProto::FLOAT8E4M3FNUZ:
    return "FLOAT8E4M3FNUZ";
  case onnx::TensorProto::FLOAT8E5M2:
    return "FLOAT8E5M2";
  case onnx::TensorProto::FLOAT8E5M2FNUZ:
    return "FLOAT8E5M2FNUZ";
  case onnx::TensorProto::UINT4:
    return "UINT4";
  case onnx::TensorProto::INT4:
    return "INT4";
  case onnx::TensorProto::UNDEFINED:
    return "UNDEFINED";
  default:
    return "UNKNOWN";
  }
}

static mlir::Type onnx_datatype_to_mlir_type(const int32_t data_type_int,
                                             mlir::MLIRContext *context) {
  auto onnx_type = static_cast<onnx::TensorProto::DataType>(data_type_int);

  switch (onnx_type) {
  case onnx::TensorProto::FLOAT:
    return mlir::Float32Type::get(context);
  case onnx::TensorProto::UINT8:
    return mlir::IntegerType::get(context, 8, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::INT8:
    return mlir::IntegerType::get(context, 8);
  case onnx::TensorProto::UINT16:
    return mlir::IntegerType::get(context, 16, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::INT16:
    return mlir::IntegerType::get(context, 16);
  case onnx::TensorProto::INT32:
    return mlir::IntegerType::get(context, 32);
  case onnx::TensorProto::INT64:
    return mlir::IntegerType::get(context, 64);
  case onnx::TensorProto::STRING:
    return mlir::NoneType::get(context);
  case onnx::TensorProto::BOOL:
    return mlir::IntegerType::get(context, 1);
  case onnx::TensorProto::FLOAT16:
    return mlir::Float16Type::get(context);
  case onnx::TensorProto::DOUBLE:
    return mlir::Float64Type::get(context);
  case onnx::TensorProto::UINT32:
    return mlir::IntegerType::get(context, 32, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::UINT64:
    return mlir::IntegerType::get(context, 64, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::COMPLEX64:
    return mlir::ComplexType::get(mlir::Float32Type::get(context));
  case onnx::TensorProto::COMPLEX128:
    return mlir::ComplexType::get(mlir::Float64Type::get(context));
  case onnx::TensorProto::BFLOAT16:
    return mlir::BFloat16Type::get(context);
  case onnx::TensorProto::FLOAT8E4M3FN:
    return mlir::Float8E4M3FNType::get(context);
  case onnx::TensorProto::FLOAT8E4M3FNUZ:
    return mlir::Float8E4M3FNUZType::get(context);
  case onnx::TensorProto::FLOAT8E5M2:
    return mlir::Float8E5M2Type::get(context);
  case onnx::TensorProto::FLOAT8E5M2FNUZ:
    return mlir::Float8E5M2FNUZType::get(context);
  case onnx::TensorProto::UINT4:
    return mlir::IntegerType::get(context, 4, mlir::IntegerType::Unsigned);
  case onnx::TensorProto::INT4:
    return mlir::IntegerType::get(context, 4);
  case onnx::TensorProto::FLOAT4E2M1:
    return mlir::Float4E2M1FNType::get(context);
  case onnx::TensorProto::UNDEFINED:
    return mlir::NoneType::get(context);
  default:
    llvm::errs() << "ERROR: Unknown ONNX data type integer value: "
                 << data_type_int << "\n";
    exit(-1);
  }
}

static std::string
onnx_typecase_tostr(const onnx::TypeProto::ValueCase value_case) {
  switch (value_case) {
  case onnx::TypeProto::kTensorType:
    return "TensorType";
  case onnx::TypeProto::kSequenceType:
    return "SequenceType";
  case onnx::TypeProto::kMapType:
    return "MapType";
  case onnx::TypeProto::kOptionalType:
    return "OptionalType";
  case onnx::TypeProto::kSparseTensorType:
    return "SparseTensorType";
  case onnx::TypeProto::VALUE_NOT_SET:
    return "VALUE_NOT_SET";
  default:
    return "UNKNOWN_TYPE_CASE";
  }
}
template <typename shp_T, typename typ_T>
static mlir::DenseElementsAttr
get_mlir_tensor(const std::string &data, shp_T shape, typ_T dType,
                const mlir::Attribute &eAttr = {}) {
  auto dims = llvm::ArrayRef(shape.data(), shape.size());
  auto shapedType = mlir::RankedTensorType::get(dims, dType, eAttr);
  auto denseAttrs = mlir::DenseElementsAttr::getFromRawBuffer(
      shapedType, llvm::ArrayRef(data.data(), data.size()));
  return denseAttrs;
}

template <typename shp_T, typename dat_T, typename typ_T>
static mlir::DenseElementsAttr
get_mlir_tensor(const google::protobuf::RepeatedField<dat_T> &data, shp_T shape,
                typ_T dType, const mlir::Attribute &eAttr = {}) {
  auto dims = llvm::ArrayRef(shape.data(), shape.size());
  auto shapedType = mlir::RankedTensorType::get(dims, dType, eAttr);
  auto denseAttrs = mlir::DenseElementsAttr::get(
      shapedType, llvm::ArrayRef<dat_T>(data.data(), data.size()));
  return denseAttrs;
}

template <typename shp_T, typename dat_T, typename typ_T>
static mlir::DenseElementsAttr
get_mlir_tensor(const std::vector<dat_T> &data, shp_T shape, typ_T dType,
                const mlir::Attribute &eAttr = {}) {
  auto dims = llvm::ArrayRef(shape.data(), shape.size());
  auto shapedType = mlir::RankedTensorType::get(dims, dType, eAttr);
  auto denseAttrs = mlir::DenseElementsAttr::get(
      shapedType, llvm::ArrayRef<dat_T>(data.data(), data.size()));
  return denseAttrs;
}

static mlir::ElementsAttr
onnx_tensorproto_to_mlir(const onnx::TensorProto &tensor,
                         mlir::MLIRContext *context,
                         const mlir::Attribute &eAttr = {}) {
  std::cout << "      Tensor Name: " << tensor.name() << std::endl;

  auto dType = onnx_datatype_to_mlir_type(tensor.data_type(), context);

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
      return get_mlir_tensor(tensor.raw_data(), tensor.dims(), dType, eAttr);
    default:
      std::cout << "ERROR: Raw data read not supported for "
                << onnx_datatype_tostr(tensor.data_type()) << std::endl;
      exit(-1);
    }
  } else {
    switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT:
      return get_mlir_tensor(tensor.float_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::DOUBLE:
      return get_mlir_tensor(tensor.double_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::INT4:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::UINT4:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT16:
      return get_mlir_tensor(tensor.int32_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::INT64:
      return get_mlir_tensor(tensor.int64_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
      return get_mlir_tensor(tensor.uint64_data(), tensor.dims(), dType, eAttr);
    case onnx::TensorProto::STRING: {
      const auto &data = tensor.string_data();
      for (int i = 0; i < data.size(); ++i) {
        std::cout << '"' << data[i] << '"';
        if (i < data.size()) {
          std::cout << ", ";
        }
      }
    } break;
    case onnx::TensorProto::UNDEFINED:
    default:
      std::cout << "ERROR: Data read not supported for "
                << onnx_datatype_tostr(tensor.data_type()) << std::endl;
      exit(-1);
    }
  }

  llvm::outs() << "      Type: " << onnx_datatype_tostr(tensor.data_type())
               << "\n";

  return nullptr;
}

static mlir::Type
onnx_valueinfo_to_mlir_type(const onnx::ValueInfoProto &value_proto,
                            mlir::MLIRContext *context) {
  const auto &type_proto = value_proto.type();

  // value data type
  switch (type_proto.value_case()) {
  case onnx::TypeProto::kTensorType: {
    std::vector<int64_t> dataShape;
    const auto &tensor_type = type_proto.tensor_type();
    auto elemType =
        onnx_datatype_to_mlir_type(tensor_type.elem_type(), context);

    std::cout << "<" << onnx_datatype_tostr(tensor_type.elem_type()) << ">"
              << std::endl;

    if (tensor_type.has_shape()) {
      std::cout << "  Shape: [";

      for (int i = 0; i < tensor_type.shape().dim_size(); ++i) {
        const auto &dim = tensor_type.shape().dim(i);
        if (dim.has_dim_value()) {
          std::cout << dim.dim_value();
          dataShape.push_back(dim.dim_value());
        } else if (dim.has_dim_param()) {
          std::cout << dim.dim_param();
          dataShape.push_back(mlir::ShapedType::kDynamic);
        } else {
          std::cout << "?";
          std::cout << "ERROR: Tensor has unknown dimension." << std::endl;
          exit(-1);
        }
        if (i < tensor_type.shape().dim_size() - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]";
    } else {
      std::cout << "  Shape: [] (Scalar or unknown)";
      std::cout << "ERROR: Tensor has no shape." << std::endl;
      exit(-1);
    }
    return mlir::RankedTensorType::get(dataShape, elemType);
  }
  case onnx::TypeProto::kSparseTensorType:
  case onnx::TypeProto::kSequenceType:
  case onnx::TypeProto::kMapType:
  case onnx::TypeProto::kOptionalType:
  case onnx::TypeProto::VALUE_NOT_SET:
  default:
    std::cout << "ERROR: TypeProto is unsupported." << std::endl;
    exit(-1);
  }
}

static mlir::Attribute get_node_attribute(const onnx::AttributeProto &attribute,
                                          mlir::MLIRContext *context,
                                          const mlir::Attribute &eAttr = {}) {
  std::cout << "    Name: " << attribute.name() << std::endl;
  std::cout << "    Type: " << onnx_attrtype_tostr(attribute.type())
            << std::endl;

  mlir::OpBuilder builder(context);

  switch (attribute.type()) {
  case onnx::AttributeProto::FLOAT:
    std::cout << "    Value: \"" << attribute.f() << "\"" << std::endl;
    return mlir::FloatAttr::get(mlir::Float32Type::get(context), attribute.f());
  case onnx::AttributeProto::INT:
    std::cout << "    Value: \"" << attribute.i() << "\"" << std::endl;
    return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                  attribute.i());
  case onnx::AttributeProto::STRING:
    std::cout << "    Value: \"" << attribute.s() << "\"" << std::endl;
    return mlir::StringAttr::get(context, attribute.s());
  case onnx::AttributeProto::TENSOR:
    std::cout << "    Value (Tensor):" << std::endl;
    return onnx_tensorproto_to_mlir(attribute.t(), context, eAttr);
  case onnx::AttributeProto::GRAPH:
    std::cout
        << "    Value (Graph): (Graph details not printed in this function)"
        << std::endl;
    std::cout << "ERROR: Parsing of this type is not implemented." << std::endl;
    exit(-1);
  case onnx::AttributeProto::FLOATS:
    return get_mlir_tensor(attribute.floats(),
                           llvm::ArrayRef<long int>({attribute.floats_size()}),
                           mlir::Float32Type::get(context), eAttr);
  case onnx::AttributeProto::INTS:
    return get_mlir_tensor(attribute.ints(),
                           llvm::ArrayRef<long int>({attribute.ints_size()}),
                           mlir::IntegerType::get(context, 64), eAttr);
  case onnx::AttributeProto::STRINGS:
    std::cout << "    Value (Strings): [";
    for (int i = 0; i < attribute.strings_size(); ++i) {
      std::cout << "\"" << attribute.strings(i) << "\"";
      if (i < attribute.strings_size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    break;
  case onnx::AttributeProto::TENSORS:
    std::cout << "    Value (Tensors):" << std::endl;
    for (int i = 0; i < attribute.tensors_size(); ++i) {
      std::cout << "      Tensor " << i << ":" << std::endl;
      onnx_tensorproto_to_mlir(attribute.tensors(i), context, eAttr);
    }
    break;
  case onnx::AttributeProto::GRAPHS:
    std::cout
        << "    Value (Graphs): (Graph details not printed in this function)"
        << std::endl;
    std::cout << "ERROR: Parsing of this type is not implemented." << std::endl;
    exit(-1);
  case onnx::AttributeProto::SPARSE_TENSOR:
    std::cout << "    Value (Sparse Tensor):" << std::endl;
    // TODO
    break;
  case onnx::AttributeProto::SPARSE_TENSORS:
    std::cout << "    Value (Sparse Tensors):" << std::endl;
    for (int i = 0; i < attribute.sparse_tensors_size(); ++i) {
      std::cout << "      Sparse Tensor " << i << ":" << std::endl;
      // TODO
    }
    break;
  case onnx::AttributeProto::TYPE_PROTO:
    std::cout << "    Value (Type Proto): (Type Proto details not printed in "
                 "this function)"
              << std::endl;
    std::cout << "ERROR: Parsing of this type is not implemented." << std::endl;
    exit(-1);
  case onnx::AttributeProto::TYPE_PROTOS:
    std::cout << "    Value (Type Protos): (Type Proto details not printed in "
                 "this function)"
              << std::endl;
    std::cout << "ERROR: Parsing of this type is not implemented." << std::endl;
    exit(-1);
  case onnx::AttributeProto::UNDEFINED:
  default:
    std::cout << "    Value: (Unsupported or Undefined Attribute Type)"
              << std::endl;
    std::cout << "ERROR: Parsing of this type is not implemented." << std::endl;
    exit(-1);
  }

  return nullptr;
}

static void print_graph_node(const onnx::NodeProto &node,
                             mlir::MLIRContext *context) {
  std::cout << std::endl;
  std::cout << "------------------[node begin]------------------" << std::endl;
  std::cout << "Op_Type: \x1B[31m" << node.op_type() << "\033[0m\t\t"
            << std::endl;
  std::cout << "Node_Name: " << node.name() << std::endl;

  std::cout << "Inputs: #" << node.input().size() << std::endl;

  // Op inputs
  for (const auto &input_name : node.input()) {
    std::cout << "    [" << input_name << "]" << std::endl;
  }

  std::cout << std::endl;

  std::cout << "Outputs: #" << node.output().size() << std::endl;

  // Op outputs
  for (const auto &output_name : node.output()) {
    std::cout << "    [" << output_name << "]" << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Attributes: #" << node.attribute().size() << std::endl;

  // Op attributes
  for (const auto &attribute : node.attribute()) {
    auto attr = get_node_attribute(attribute, context);
    // DEBUG
    mlir::OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(16);
    llvm::outs() << "      Data: ";
    mlir::AsmState state(context, flags);
    attr.print(llvm::outs(), state);
    llvm::outs() << "\n";
    llvm::outs().flush();
  }

  std::cout << "------------------[node end]--------------------" << std::endl;
}

static void print_graph_data(const onnx::TensorProto &initializer,
                             mlir::MLIRContext *context) {
  std::cout << std::endl;
  std::cout << "------------------[data begin]------------------" << std::endl;
  std::cout << "Name: \x1B[31m" << initializer.name() << "\033[0m\t\t"
            << std::endl;
  std::cout << "    DataType: " << onnx_datatype_tostr(initializer.data_type())
            << std::endl;

  onnx_tensorproto_to_mlir(initializer, context);

  std::cout << "------------------[data end]--------------------" << std::endl;
  std::cout << std::endl;
}

static mlir::Operation *
createOnnxOp(mlir::OpBuilder &builder, const std::string &opName,
             const std::vector<mlir::Type> &types = {},
             const std::vector<mlir::Value> &values = {},
             const std::vector<mlir::NamedAttribute> &attrs = {}) {

  // setup operation
  mlir::StringAttr opNameStr = builder.getStringAttr("onnx." + opName);
  mlir::OperationState state(builder.getUnknownLoc(), opNameStr);

//  auto ops = mlir::cast<onnx2mlir::dialect::onnx::OperandCountInfo>(state);


  for (auto type : types)
    state.addTypes(type);
  for (auto value : values)
    state.addOperands(value);
  for (auto attr : attrs)
    state.addAttributes(attr);

  mlir::Operation *op = builder.create(state);

  auto opw = mlir::cast<onnx2mlir::dialect::onnx::OperandCountInfo>(op);
  printf("OPDBG [%s] num_in[%i]\n", opName.c_str(), opw.getDefinedOperandCount());

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
  std::cout << std::endl;

  // i/o map storage
  std::map<std::string, std::shared_ptr<onnx::ValueInfoProto>> vis;
//  std::multimap<std::string, std::shared_ptr<onnx::NodeProto>> node_inputs;
//  std::multimap<std::string, std::shared_ptr<onnx::NodeProto>> node_outputs;
/*
  for (const auto &node : graph_proto.node()) {
    // DEBUG
    // print_graph_node(node, mlirCtx.get());
    auto node_ptr = std::make_shared<onnx::NodeProto>(node);
    // map the inputs
    for (const auto &input_name : node.input()) {
      if (input_name.size()) {
        node_inputs.insert({input_name, node_ptr});
      }
    }
    // map the outputs
    for (const auto &output_name : node.output()) {
      if (output_name.size()) {
        node_outputs.insert({output_name, node_ptr});
      }
    }
  }
*/
  // map value infos
  for (const auto &vi : graph_proto.value_info()) {
    auto vi_ptr = std::make_shared<onnx::ValueInfoProto>(vi);

    printf("\n\nVIS [%s]\n", vi.name().c_str());
    auto type = onnx_valueinfo_to_mlir_type(vi, mlirCtx.get());
    type.print(llvm::outs());

    vis.insert({vi.name(), vi_ptr});
  }

  // initializers map storage
  std::map<std::string, std::shared_ptr<onnx::TensorProto>> node_inits;

  // map the initializers
  for (const auto &initializer : graph_proto.initializer()) {
    // DEBUG
    // print_graph_data(initializer, mlirCtx.get());
    auto data_ptr = std::make_shared<onnx::TensorProto>(initializer);
    node_inits.insert({initializer.name(), data_ptr});
  }

  // get main func
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  // add func body
  auto block = func.addEntryBlock();

  // args storage
  std::map<std::string, std::shared_ptr<mlir::Value>> func_inputs;
  // std::map<std::string, mlir::Value> func_inputs;
  std::map<std::string, std::shared_ptr<mlir::Type>> func_outputs;

  // map func inputs argument
  for (unsigned int i = 0; i < func.getNumArguments(); i++) {
    mlir::Value arg = block->getArgument(i);
    auto attr = func.getArgAttrOfType<mlir::StringAttr>(i, "onnx.name");
    if (attr.size()) {
      auto arg_ptr = std::make_shared<mlir::Value>(arg);
      func_inputs.insert({attr.getValue().str(), arg_ptr});
      // func_inputs.insert({attr.getValue().str(), arg});
    }
  }
  // map func results argument
  for (unsigned int i = 0; i < func.getNumResults(); i++) {
     mlir::Type res = func.getFunctionType().getResult(i);
//    printf("\n\nFUNC res[%i][%lu]\n\n", func.getNumResults(), res.size());
//    func.print(llvm::outs());
//    exit(-1);
    auto attr = func.getResultAttrOfType<mlir::StringAttr>(i, "onnx.name");
    if (attr.size()) {
      auto res_ptr = std::make_shared<mlir::Type>(res);
      func_outputs.insert({attr.getValue().str(), res_ptr});
    }
  }
//  exit(-1);

  /*
   * Build the MLIR graph
   */

  mlir::OpBuilder builder(mlirCtx.get());

  // map ops i/o
  std::map<std::string, std::shared_ptr<mlir::Operation *>> ops_by_name;
  std::multimap<std::string, std::shared_ptr<mlir::Operation *>> ops_by_inputs;
  std::multimap<std::string, std::shared_ptr<mlir::Operation *>> ops_by_outputs;

  // add constant nodes
  for (const auto &node : graph_proto.node()) {
    if (node.op_type() == "Constant") {
      // DEBUG
      // print_graph_node(node, mlirCtx.get());

      // attributes
      std::vector<mlir::NamedAttribute> attrs;
      auto value = get_node_attribute(node.attribute()[0], mlirCtx.get());
      mlir::NamedAttribute attr(node.attribute()[0].name(), value);
      attrs.push_back(attr);
      // origin
      auto nameVal = builder.getStringAttr(node.name());
      mlir::NamedAttribute label("onnx.node.name", nameVal);
      attrs.push_back(label);
      // result type
      auto types = std::vector<mlir::Type>(
          {mlir::dyn_cast<mlir::DenseElementsAttr>(value).getType()});

      auto op = createOnnxOp(builder, "Constant", types, {}, attrs);

      block->push_back(op);
      // store output
      auto res_ptr = std::make_shared<mlir::Operation *>(op);
      ops_by_outputs.insert({node.output()[0], res_ptr});
    }
  }
  // add the initializers
  for (const auto &initializer : graph_proto.initializer()) {
    // DEBUG
    // parse_graph_data(initializer, mlirCtx.get());

    // attributes
    std::vector<mlir::NamedAttribute> attrs;
    auto value = onnx_tensorproto_to_mlir(initializer, mlirCtx.get());
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

  // add nodes
  for (const auto &node : graph_proto.node()) {
    if (node.op_type() != "Constant") {
      // DEBUG
      // print_graph_node(node, mlirCtx.get());

      // attributes
      std::vector<mlir::NamedAttribute> attrs;
      for (const auto &attribute : node.attribute()) {
        auto value = get_node_attribute(attribute, mlirCtx.get());
        mlir::NamedAttribute attr(attribute.name(), value);
        attrs.push_back(attr);
      }
      // origin
      auto nameVal = builder.getStringAttr(node.name());
      mlir::NamedAttribute labels("onnx.node.name", nameVal);
      attrs.push_back(labels);

      // result types
      std::vector<mlir::Type> types;
      for (const auto &out : node.output()) {
        auto it = vis.find(out);
        if (it != vis.end()) {
          auto type = onnx_valueinfo_to_mlir_type(*(it->second), mlirCtx.get());
          types.push_back(type);
        } else {
          std::cout << "\nWARN: [" << node.name() << "] not in value_info() catalog" << std::endl;
          // TODO lookup adjacency if empty !
          printf("DBG [%s]->[%s]->[%s]\n", node.input()[0].c_str(), node.op_type().c_str(), node.output()[0].c_str());

          // lookup ops outputs
          auto oitr = ops_by_inputs.find(out);
          if (oitr != ops_by_inputs.end()) {
            auto oadj = *oitr->second;
            oadj->print(llvm::outs());
            // TODO which result ?!
            types.push_back(oadj->getResults()[0].getType());
          }

          // lookup func outputs
          auto fitr = func_outputs.find(out);
          if (fitr != func_outputs.end()) {
            auto fadj = *fitr->second;
            fadj.print(llvm::outs());
            types.push_back(fadj);
          }
        }
      }

//      mlir::OperationName opName("onnx.Slice", mlirCtx.get());
//      auto registeredInfo = mlir::RegisteredOperationName::lookup(opName.getTypeID(), mlirCtx.get());
//      auto abstractOp = registeredInfo->getAbstractOperation();
/*
      const std::string domain = "";
      const onnx::OpSchemaRegistry* registry = onnx::OpSchemaRegistry::Instance();
      int latest_version = onnx::OpSchemaRegistry::GetLatestOpSetVersion(domain);
      const onnx::OpSchema* schema = registry->GetSchema(node.op_type(), latest_version, domain);
      int num_formal_inputs = schema->inputs().size();
*/

      // operands (temporary)
      std::vector<mlir::Value> operands;
      for (int i = 0; i < node.input().size(); ++i) {
        auto arg = *func_inputs.begin()->second;
        operands.push_back(arg);
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

    if (node.op_type() == "Slice") {
      printf("\n\nDBG [%s][%i]\n\n", node.name().c_str(), node.input().size());
      op->print(llvm::outs());
      exit(-1);
    }

    }

  }

  // set nodes inputs
  for (const auto &node : graph_proto.node()) {

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
        print_graph_node(node, mlirCtx.get());
        mlir::Value arg = *(f_it->second);

        //arg.print(llvm::outs());

        node_op->setOperand(idx, arg);
      }

      // map node input to upper node output
      auto o_it = ops_by_outputs.find(node.input()[idx]);
      if (o_it != ops_by_outputs.end()) {
        print_graph_node(node, mlirCtx.get());
        auto op = *(o_it->second);
        mlir::Value arg = op->getOpResult(0);

        //arg.print(llvm::outs());

        node_op->setOperand(idx, arg);
        continue;
      }
    }

    // erase unused inputs
    for (int idx = 0; idx < node.input().size(); ++idx) {
      if (node.input()[idx].size() == 0) {
        node_op->eraseOperand(idx);
      }
    }
  }

  // map func outputs
  for (const auto &out : func_outputs) {
    auto o_it = ops_by_outputs.find(out.first);
    mlir::SmallVector<mlir::Value> values;

    if (o_it != ops_by_outputs.end()) {
      auto op = *(o_it->second);
      //printf("OUTPUT out[%s] [%i]\n", out.first.c_str(), op->getNumResults());
      mlir::Value val = op->getOpResult(0);
      values.push_back(val);
    }
    auto ret = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), values);
    block->push_back(ret);
  }

}

void ONNXImporter::parse_graph_io(const onnx::GraphProto &graph_proto) {

  std::vector<mlir::Type> inputs;

  // main inputs
  std::cout << "Graph Inputs:" << std::endl;
  for (const auto &input : graph_proto.input()) {
    std::cout << "  Name: " << input.name() << std::endl;
    if (input.has_type()) {
      std::cout << "  Type: " << onnx_typecase_tostr(input.type().value_case());
      inputs.push_back(onnx_valueinfo_to_mlir_type(input, mlirCtx.get()));
    } else {
      std::cout << "ERROR: Type Not Specified.";
      exit(-1);
    }
    std::cout << std::endl;
  }

  std::vector<mlir::Type> outputs;

  // main outputs
  std::cout << "Graph Outputs:" << std::endl;
  for (const auto &output : graph_proto.output()) {
    std::cout << "  Name: " << output.name() << std::endl;
    if (output.has_type()) {
      std::cout << "  Type: "
                << onnx_typecase_tostr(output.type().value_case());
      outputs.push_back(onnx_valueinfo_to_mlir_type(output, mlirCtx.get()));
    } else {
      std::cout << "ERROR: Type Not Specified.";
      exit(-1);
    }
    std::cout << std::endl;
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

  std::cout << "ONNX engine version: " << onnx::LAST_RELEASE_VERSION
            << std::endl;
  std::cout << "ONNX engine IR version: " << onnx::IR_VERSION << std::endl;

  int last_op_version = -1;
  auto all_schemas = onnx::OpSchemaRegistry::get_all_schemas();
  for (const auto &schema : all_schemas) {
    if (last_op_version < schema.SinceVersion())
      last_op_version = schema.SinceVersion();
  }

  std::cout << "ONNX engine last opset version: " << last_op_version
            << std::endl;

  std::ifstream model_file(filepath, std::ios::binary);

  if (!model_file.is_open()) {
    std::cerr << "Error opening file: " << filepath << std::endl;
    exit(-1);
  }

  onnx::ModelProto model_import;
  /// parse onnx binary
  if (!model_import.ParseFromIstream(&model_file)) {
    std::cerr << "Error parsing the ONNX model file." << std::endl;
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
  std::cout << std::endl;
  std::cout << "Model IR version: " << model_import.ir_version() << std::endl;
  std::cout << "Model OPSET conversion: " << OpSetVer << " -> "
            << last_op_version << std::endl;
  std::cout << std::endl;

  /// convert model
  auto model_proto =
      onnx::version_conversion::ConvertVersion(model_import, last_op_version);
  /// infer shapes
  onnx::shape_inference::InferShapes(model_proto);

  const onnx::GraphProto &graph_proto = model_proto.graph();
  std::cout << "Graph Name: " << graph_proto.name() << std::endl;

  /*
   * MLIR ONNX
   */

  // construct function args
  parse_graph_io(graph_proto);

  // pupulate body operators
  parse_graph_nodes(graph_proto);

  // verify module
  if (llvm::failed(mlir::verify(*module))) {
    llvm::errs() << "Module verification failed!\n";
    exit(-1);
  }

  // DEBUG
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  llvm::outs().enable_colors(true);
  module->print(llvm::outs(), flags);
  llvm::outs().enable_colors(false);
}

} // end namespace onnx2mlir::frontend
