
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinTypes.h>

#include <onnx/common/version.h>
#include <onnx/shape_inference/implementation.h>
#include <onnx/version_converter/convert.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
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

template <typename T> static void print_vector(const std::vector<T> &vec) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  const size_t max_elements_to_print = 10;
  for (size_t i = 0; i < vec.size(); ++i) {
    buffer.clear();
    os << vec[i];
    os.flush();
    if (i > 0)
      std::cout << ", ";
    if (i >= max_elements_to_print) {
      std::cout << "...";
      break;
    }
    std::cout << os.str();
  }
}

template <typename shp_T, typename typ_T>
static mlir::DenseElementsAttr get_mlir_tensor(const std::string &data,
                                               shp_T shape, typ_T dType) {
  auto dims = llvm::ArrayRef(shape.data(), shape.size());
  auto shapedType = mlir::RankedTensorType::get(dims, dType);
  auto denseAttrs = mlir::DenseElementsAttr::getFromRawBuffer(
      shapedType, llvm::ArrayRef(data.data(), data.size()));
  return denseAttrs;
}

template <typename shp_T, typename dat_T, typename typ_T>
static mlir::DenseElementsAttr
get_mlir_tensor(const google::protobuf::RepeatedField<dat_T> &data, shp_T shape,
                typ_T dType) {
  auto dims = llvm::ArrayRef(shape.data(), shape.size());
  auto shapedType = mlir::RankedTensorType::get(dims, dType);
  auto denseAttrs = mlir::DenseElementsAttr::get(
      shapedType, llvm::ArrayRef<dat_T>(data.data(), data.size()));
  return denseAttrs;
}

static void onnx_tensorproto_to_mlir(const onnx::TensorProto &tensor,
                                     mlir::MLIRContext *context) {
  std::cout << "      Tensor Name: " << tensor.name() << std::endl;

  mlir::DenseElementsAttr denseAttrs;
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
      denseAttrs = get_mlir_tensor(tensor.raw_data(), tensor.dims(), dType);
      break;
    default:
      std::cout << "ERROR: Raw data read not supported for "
                << onnx_datatype_tostr(tensor.data_type()) << std::endl;
      exit(-1);
    }
  } else {
    switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT:
      denseAttrs = get_mlir_tensor(tensor.float_data(), tensor.dims(), dType);
      break;
    case onnx::TensorProto::DOUBLE:
      denseAttrs = get_mlir_tensor(tensor.double_data(), tensor.dims(), dType);
      break;
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::INT4:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::UINT4:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT16:
      denseAttrs = get_mlir_tensor(tensor.int32_data(), tensor.dims(), dType);
      break;
    case onnx::TensorProto::INT64:
      denseAttrs = get_mlir_tensor(tensor.int64_data(), tensor.dims(), dType);
      break;
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
      denseAttrs = get_mlir_tensor(tensor.uint64_data(), tensor.dims(), dType);
      break;
    case onnx::TensorProto::STRING: {
      const auto &data = tensor.string_data();
      for (int i = 0; i < data.size(); ++i) {
        std::cout << '"' << data[i] << '"';
        if (i < data.size()) {
          std::cout << ", ";
        }
      }
    }
    case onnx::TensorProto::UNDEFINED:
    default:
      std::cout << "ERROR: Data read not supported for "
                << onnx_datatype_tostr(tensor.data_type()) << std::endl;
      exit(-1);
    }
  }

  llvm::outs() << "      Type: " << onnx_datatype_tostr(tensor.data_type())
               << "\n";

  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  llvm::outs() << "      Data: [";
  mlir::AsmState state(context, flags);
  denseAttrs.print(llvm::outs(), state);
  llvm::outs() << "\n";
}

static mlir::Type
onnx_valuetype_to_mlir_type(const onnx::ValueInfoProto &value_proto,
                            mlir::MLIRContext *context) {
  const auto &type_proto = value_proto.type();

  // builder
  mlir::OpBuilder builder(context);

  // attribute
  mlir::StringAttr nameKey = builder.getStringAttr("onnx.name");
  mlir::StringAttr nameVal = builder.getStringAttr(value_proto.name());
  mlir::NamedAttribute namedAttr(nameKey, nameVal);
  mlir::Attribute encodedAttr = builder.getDictionaryAttr({namedAttr});

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
          std::cout << "ERROR: Tensor has no dimension value." << std::endl;
          exit(-1);
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
    return mlir::RankedTensorType::get(dataShape, elemType, encodedAttr);
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

void parse_node_attributes(const onnx::AttributeProto &attribute,
                           mlir::MLIRContext *context) {
  std::cout << "    Name: " << attribute.name() << std::endl;
  std::cout << "    Type: " << onnx_attrtype_tostr(attribute.type())
            << std::endl;

  switch (attribute.type()) {
  case onnx::AttributeProto::FLOAT:
    std::cout << "    Value: " << attribute.f() << std::endl;
    break;
  case onnx::AttributeProto::INT:
    std::cout << "    Value: " << attribute.i() << std::endl;
    break;
  case onnx::AttributeProto::STRING:
    std::cout << "    Value: \"" << attribute.s() << "\"" << std::endl;
    break;
  case onnx::AttributeProto::TENSOR:
    std::cout << "    Value (Tensor):" << std::endl;
    onnx_tensorproto_to_mlir(attribute.t(), context);
    break;
  case onnx::AttributeProto::GRAPH:
    std::cout
        << "    Value (Graph): (Graph details not printed in this function)"
        << std::endl;
    std::cout << "ERROR: Parsing of this type is not implemented." << std::endl;
    exit(-1);
  case onnx::AttributeProto::FLOATS:
    std::cout << "    Value (Floats): [";
    for (int i = 0; i < attribute.floats_size(); ++i) {
      std::cout << attribute.floats(i);
      if (i < attribute.floats_size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    break;
  case onnx::AttributeProto::INTS:
    std::cout << "    Value (Ints): [";
    for (int i = 0; i < attribute.ints_size(); ++i) {
      std::cout << attribute.ints(i);
      if (i < attribute.ints_size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    break;
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
      onnx_tensorproto_to_mlir(attribute.tensors(i), context);
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
}

static void parse_graph_node(const onnx::NodeProto &node,
                             mlir::MLIRContext *context) {
  std::cout << std::endl;
  std::cout << "------------------[node begin]------------------" << std::endl;
  std::cout << "Op_Type: \x1B[31m" << node.op_type() << "\033[0m\t\t"
            << std::endl;
  std::cout << "Node_Name: " << node.name() << std::endl;

  std::cout << "Inputs: #" << node.input().size() << std::endl;
  for (const auto &input_name : node.input()) {
    std::cout << "    [" << input_name << "]" << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Outputs: #" << node.output().size() << std::endl;
  for (const auto &output_name : node.output()) {
    std::cout << "    [" << output_name << "]" << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Attributes: #" << node.attribute().size() << std::endl;
  for (const auto &attribute : node.attribute()) {
    parse_node_attributes(attribute, context);
  }
  std::cout << "------------------[node end]--------------------" << std::endl;
}

/*
// Sort graph into lexicographically smallest topological ordering.
// Returns true if sorted succesfully and false otherwise.
bool SortGraph(onnx::GraphProto *graph) {
  int nNodes = graph->node().size();
  // Map of edges / node-outputs to their parent ops
  std::map<std::string, int> origIndex;
  int index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &output : node.output()) {
      origIndex[output] = index;
    }
    index++;
  }
  assert(index == nNodes);

  // graph inputs and initializers should not be counted as dependencies.
  std::set<std::string> graphInputsAndInitializers;
  for (const auto &initializer : graph->initializer()) {
    const auto &initializerName = initializer.name();
    graphInputsAndInitializers.insert(initializerName);
  }
  for (const auto &input : graph->input()) {
    graphInputsAndInitializers.insert(input.name());
  }
  // Empty input names should be ignored.
  graphInputsAndInitializers.insert("");

  // Users tracks idx of the ops which consumes a given ops outputs.
  std::vector<std::vector<int>> users(nNodes);
  index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &input : node.input()) {
      // Input edges to node are graph inputs or initializers.
      if (graphInputsAndInitializers.count(input))
        continue;
      // Check if input edges to node aren't graph inputs or initializers and
      // don't have a parent op, in which case its not possible to
      // topologically sort the graph.
      if (!origIndex.count(input)) {
        return false;
      }
      // Add current node as a user of the op that produces input.
      users[origIndex[input]].push_back(index);
    }
    index++;
  }

  // inDegrees stores the number of inputs to a given node not counting inputs
  // which are graph inputs or initializers.
  std::vector<int> inDegrees(nNodes, 0);
  index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &input : node.input()) {
      if (!graphInputsAndInitializers.count(input)) {
        inDegrees[index]++;
      }
    }
    index++;
  }
  assert(index == nNodes);

  // Create a set and inserting all nodes with indegree 0.
  std::multiset<int> nodeList;
  for (int i = 0; i < nNodes; i++) {
    if (inDegrees[i] == 0) {
      nodeList.insert(i);
    }
  }

  // The number of visited nodes.
  int nVisited = 0;
  // The final topological order.
  std::vector<int> topOrder;

  // Now we follow Kahn's algorithm for topological sorting
  while (!nodeList.empty()) {
    // Extract node with minimum number from multiset
    // and add it to topological order.
    int u = *nodeList.begin();
    nodeList.erase(nodeList.begin());
    topOrder.push_back(u);

    // Iterate through all its users
    // and decreament inDegrees by 1.
    for (auto v : users[u]) {
      // If inDegree becomes zero, add it to queue.
      if (--inDegrees[v] == 0) {
        nodeList.insert(v);
      }
    }
    nVisited++;
  }
  // No possible topological order.
  if (nVisited != nNodes) {
    return false;
  }

  // Generate SwapElements to reach desired order.
  std::vector<int> curOrder(nNodes);
  for (int i = 0; i < nNodes; i++)
    curOrder[i] = i;
  for (int resIndex = 0; resIndex < nNodes; resIndex++) {
    if (topOrder[resIndex] == curOrder[resIndex])
      continue;
    for (int search = resIndex + 1; search < nNodes; search++) {
      if (topOrder[resIndex] == curOrder[search]) {
        graph->mutable_node()->SwapElements(resIndex, search);
        std::swap(curOrder[search], curOrder[resIndex]);
        break;
      }
    }
  }
  return true;
}*/

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
  for (const auto &node : graph_proto.node()) {
    parse_graph_node(node, mlirCtx.get());
  }
  printf("\n\n");
  for (const auto &initializer : graph_proto.initializer()) {
    printf("ENTRY [%s] size:[%i]\n", initializer.name().c_str(),
           initializer.dims().size());
    printf("DIM");
    /// https://github.com/onnx/onnx/blob/main/docs/IR.md
    for (const auto &d : initializer.dims()) {
      printf(" %lu", d);
    }
    printf("\n");

    printf("\n");
  }
}

void ONNXImporter::parse_graph_io(const onnx::GraphProto &graph_proto) {

  std::vector<mlir::Type> inputs;

  std::cout << "Graph Inputs:" << std::endl;
  for (const auto &input : graph_proto.input()) {
    std::cout << "  Name: " << input.name() << std::endl;
    if (input.has_type()) {
      std::cout << "  Type: " << onnx_typecase_tostr(input.type().value_case());
      inputs.push_back(onnx_valuetype_to_mlir_type(input, mlirCtx.get()));
    } else {
      std::cout << "ERROR: Type Not Specified.";
      exit(-1);
    }
    std::cout << std::endl;
  }

  std::vector<mlir::Type> outputs;

  std::cout << "Graph Outputs:" << std::endl;
  for (const auto &output : graph_proto.output()) {
    std::cout << "  Name: " << output.name() << std::endl;
    if (output.has_type()) {
      std::cout << "  Type: "
                << onnx_typecase_tostr(output.type().value_case());
      outputs.push_back(onnx_valuetype_to_mlir_type(output, mlirCtx.get()));
    } else {
      std::cout << "ERROR: Type Not Specified.";
      exit(-1);
    }
    std::cout << std::endl;
  }

  mlir::OpBuilder builder(mlirCtx.get());
  auto funcType = mlir::FunctionType::get(mlirCtx.get(), inputs, outputs);
  auto func = mlir::func::FuncOp::create(builder.getUnknownLoc(), "main",
                                         funcType, /*attr*/ {});
  module->push_back(func);
}

void ONNXImporter::import(const std::string &filepath) {
  std::cout << "ONNX engine version: " << onnx::LAST_RELEASE_VERSION
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
  std::cout << "Model OPSET version: " << OpSetVer << std::endl;
  std::cout << std::endl;

  /// convert model
  auto model_proto =
      model_import; // onnx::version_conversion::ConvertVersion(model_import,
                    //  23);
  /// infer shapes
  onnx::shape_inference::InferShapes(model_proto);

  const onnx::GraphProto &graph_proto = model_proto.graph();
  std::cout << "Graph Name: " << graph_proto.name() << std::endl;

  /*
   * MLIR ONNX
   */

  // contrust func ins & outs
  parse_graph_io(graph_proto);

  // pupulate body operators
  parse_graph_nodes(graph_proto);

  // auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  // mlir::Block *block = func.addEntryBlock();

  // DEBUG
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  llvm::outs().enable_colors(true);
  module->print(llvm::outs(), flags);
  llvm::outs().enable_colors(false);
}

} // end namespace onnx2mlir::frontend
