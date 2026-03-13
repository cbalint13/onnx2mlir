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
 * \brief ONNX format file importer
 */

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SparseTensor/IR/SparseTensor.h>
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
#include <memory>
#include <string>
#include <vector>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
#include "onnx2mlir/dialect/onnx/OnnxInterfaces.hpp"
#include "onnx2mlir/dialect/onnx/OnnxOps.hpp"
#include "onnx2mlir/frontend/onnx.hpp"

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
    if constexpr (std::is_same_v<dat_T, int64_t>) {
      attrVec.push_back(
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), value));
    } else if constexpr (std::is_same_v<dat_T, float>) {
      attrVec.push_back(
          mlir::FloatAttr::get(mlir::Float32Type::get(ctx), value));
    } else if constexpr (std::is_same_v<dat_T, std::string>) {
      attrVec.push_back(mlir::StringAttr::get(ctx, value));
    } else {
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
    case onnx::TensorProto::DOUBLE:
    case onnx::TensorProto::FLOAT:
    case onnx::TensorProto::FLOAT16:
    case onnx::TensorProto::BFLOAT16:
    case onnx::TensorProto::FLOAT8E4M3FN:
    case onnx::TensorProto::FLOAT8E4M3FNUZ:
    case onnx::TensorProto::FLOAT8E5M2:
    case onnx::TensorProto::FLOAT8E5M2FNUZ:
#if ONNX2MLIR_ONNX_VERSION >= 120
    case onnx::TensorProto::FLOAT8E8M0:
#endif
    case onnx::TensorProto::FLOAT4E2M1:
    case onnx::TensorProto::BOOL:
#if ONNX2MLIR_ONNX_VERSION >= 121
    case onnx::TensorProto::INT2:
#endif
    case onnx::TensorProto::INT4:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::INT64:
#if ONNX2MLIR_ONNX_VERSION >= 121
    case onnx::TensorProto::UINT2:
#endif
    case onnx::TensorProto::UINT4:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT16:
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
    case onnx::TensorProto::COMPLEX64:
    case onnx::TensorProto::COMPLEX128:
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
#if ONNX2MLIR_ONNX_VERSION >= 121
    case onnx::TensorProto::INT2:
#endif
    case onnx::TensorProto::INT4:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
#if ONNX2MLIR_ONNX_VERSION >= 121
    case onnx::TensorProto::UINT2:
#endif
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
    // TODO(cbalint13): STRING
    case onnx::TensorProto::UNDEFINED:
    default:
      llvm::errs() << "ERROR: Data read not supported for this type.\n";
      exit(-1);
    }
  }

  return nullptr;
}

static mlir::sparse_tensor::SparseTensorEncodingAttr
GetCOOEncoding(const size_t &rank, mlir::MLIRContext *ctx) {
  // COO type
  mlir::SmallVector<mlir::sparse_tensor::LevelType> dimTypes;
  for (size_t i = 0; i < rank; ++i) {
    auto lvl = mlir::sparse_tensor::LevelType(
        mlir::sparse_tensor::LevelFormat::Compressed);
    dimTypes.push_back(lvl);
  }
  // identity affine map
  // (d0, d1, ...) -> (d0, d1, ...)
  mlir::SmallVector<mlir::AffineExpr> idResults;
  for (size_t i = 0; i < rank; ++i) {
    idResults.push_back(mlir::getAffineDimExpr(i, ctx));
  }
  auto dimToLvlOrderingMap = mlir::AffineMap::get(rank, 0, idResults, ctx);
  // lvl-to-dim map (inverse of dimToLvl)
  // if dimToLvl is identity, lvlToDim is also identity.
  auto lvlToDimOrderingMap = dimToLvlOrderingMap;

  return mlir::sparse_tensor::SparseTensorEncodingAttr::get(
      ctx, dimTypes, dimToLvlOrderingMap, lvlToDimOrderingMap,
      0, // point bitWidth (0 = auto)
      0, // coord bitWidth (0 = auto)
      {}, {});
}

static mlir::ElementsAttr
OnnxToMlir_SparseTensor(const onnx::SparseTensorProto &tensor,
                        mlir::MLIRContext *ctx,
                        const mlir::Attribute &eAttr = {}) {
  auto valAttr = OnnxToMlir_Tensor(tensor.values(), ctx);
  auto indAttr = OnnxToMlir_Tensor(tensor.indices(), ctx);
  auto valType = mlir::cast<mlir::RankedTensorType>(valAttr.getType());
  auto indType = mlir::cast<mlir::RankedTensorType>(indAttr.getType());

  int64_t nnz = valType.getShape()[0];
  int64_t rank = indType.getShape()[1];

  if (!valType || valType.getRank() != 1) {
    llvm::errs() << "ERROR: sparse values must be a 1D tensor.\n";
    exit(-1);
  }

  if (!indType || indType.getRank() != 2) {
    llvm::errs() << "ERROR: sparse indices must be a 2D tensor.\n";
    exit(-1);
  }

  if (!valType || valType.getRank() != 1) {
    llvm::errs() << "ERROR: sparse values must be a 1D tensor.\n";
    exit(-1);
  }

  if (nnz != indType.getShape()[0]) {
    llvm::errs() << "ERROR: Number of sparse values (" << nnz
                 << ") does not match number of sparse index rows ("
                 << indType.getShape()[0] << ").\n";
    exit(-1);
  }

  // dense shape inferrence
  std::vector<int64_t> denseShape(rank, 0);
  auto indexValues = indAttr.getValues<int64_t>();
  // max index for each dim
  for (int64_t i = 0; i < nnz; ++i) {
    for (int64_t j = 0; j < rank; ++j) {
      int64_t currentIndex = indexValues[i * rank + j] + 1;
      if (currentIndex >= denseShape[j]) {
        denseShape[j] = currentIndex;
      }
    }
  }

  auto encCOO = GetCOOEncoding(rank, ctx);

  auto sparseTensorType =
      mlir::RankedTensorType::get(denseShape, valType.getElementType(), encCOO);

  auto sparseAttr = mlir::SparseElementsAttr::get(
      sparseTensorType, mlir::cast<mlir::DenseElementsAttr>(valAttr),
      mlir::cast<mlir::DenseElementsAttr>(indAttr));

  return sparseAttr;
}

template <typename dat_T>
static std::vector<int64_t> OnnxToMlir_Shape(const dat_T &tensor_type) {
  // extract shape
  std::vector<int64_t> dataShape;
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

  return dataShape;
}

static mlir::Type OnnxToMlir_Type(const onnx::ValueInfoProto &value_proto,
                                  mlir::MLIRContext *ctx) {
  const auto &type_proto = value_proto.type();
  // triage value data type
  switch (type_proto.value_case()) {
  case onnx::TypeProto::kTensorType: {
    const auto &tensor_type = type_proto.tensor_type();
    const auto dShape = OnnxToMlir_Shape(type_proto.tensor_type());
    auto dType = OnnxToMlir_dType(tensor_type.elem_type(), ctx);
    return mlir::RankedTensorType::get(dShape, dType);
  }
  case onnx::TypeProto::kSparseTensorType: {
    const auto &tensor_type = type_proto.sparse_tensor_type();
    const auto dShape = OnnxToMlir_Shape(tensor_type);
    auto dType = OnnxToMlir_dType(tensor_type.elem_type(), ctx);
    auto encCOO = GetCOOEncoding(/*rank*/ dShape.size(), ctx);
    return mlir::RankedTensorType::get(dShape, dType, encCOO);
  }
  case onnx::TypeProto::kSequenceType:
  case onnx::TypeProto::kMapType:
  case onnx::TypeProto::kOptionalType:
  case onnx::TypeProto::VALUE_NOT_SET:
  default:
    llvm::errs() << "ERROR: TypeProto is unsupported.\n";
    exit(-1);
  }

  return nullptr;
}

static std::optional<mlir::NamedAttribute>
OnnxToMlir_Attr(const onnx::AttributeProto &attribute, mlir::MLIRContext *ctx,
                const mlir::Attribute &eAttr = {}) {
  mlir::OpBuilder builder(ctx);
  // triage attribute type
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
  case onnx::AttributeProto::SPARSE_TENSOR:
    return mlir::NamedAttribute(
        attribute.name(),
        OnnxToMlir_SparseTensor(attribute.sparse_tensor(), ctx, eAttr));
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
  case onnx::AttributeProto::SPARSE_TENSORS:
    llvm::errs()
        << "ERROR: Parsing SPARSE_TENSORS attribute type is not implemented.\n";
    exit(-1);
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
createOnnxOp(mlir::OpBuilder *builder, const std::string &opName,
             const std::vector<mlir::Type> &types = {},
             const std::vector<mlir::Value> &values = {},
             const std::vector<mlir::NamedAttribute> &attrs = {}) {
  // setup operation
  mlir::OperationState state(builder->getUnknownLoc(), "onnx." + opName);

  for (auto type : types)
    state.addTypes(type);
  for (auto value : values)
    state.addOperands(value);
  for (auto attr : attrs)
    state.addAttributes(attr);

  mlir::Operation *op = builder->create(state);

  return op;
}

namespace onnx2mlir::frontend {

/*
 *  ONNXImporter class
 */

ONNXImporter::ONNXImporter(const std::map<std::string, std::string> &options)
    : FrontendImporter(options) {}

const std::string ONNXImporter::get_versioned_name(const std::string &OpName) {
  // const 1,3,9,23
  // opset_ver = 11
  int maxversion = -1;
  int subversion = -1;
  for (const int &ver : ops_versions[OpName]) {
    if (maxversion < ver)
      maxversion = ver;
    if (ver <= model_opset_version)
      subversion = ver;
  }
  if (maxversion > model_opset_version)
    return OpName + "_V" + std::to_string(subversion);
  return OpName;
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
  auto func = module.lookupSymbol<mlir::func::FuncOp>("main");
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

  mlir::OpBuilder builder(mlirCtx);

  // map mlir ops i/o
  std::map<std::string, std::shared_ptr<mlir::Operation *>> ops_by_name;
  std::multimap<std::string, std::shared_ptr<mlir::Operation *>> ops_by_inputs;
  std::multimap<std::string, std::shared_ptr<mlir::Value>> res_by_outputs;

  // Step 1, add constant nodes
  for (const auto &node : graph_proto.node()) {
    // check operator
    const auto opFullName = get_versioned_name(node.op_type());
    if (!checkOnnxOpExists(mlirCtx, opFullName)) {
      llvm::errs() << "ERROR: operation [" << opFullName
                   << "] not registered.\n";
      exit(-1);
    }

    if (node.op_type() == "Constant") {
      // attributes
      std::vector<mlir::NamedAttribute> attrs;
      auto attr = OnnxToMlir_Attr(node.attribute()[0], mlirCtx);
      attrs.push_back(*attr);
      // origin
      auto nameVal = builder.getStringAttr(node.name());
      mlir::NamedAttribute label("onnx.node.name", nameVal);
      attrs.push_back(label);
      // result type
      auto types = std::vector<mlir::Type>(
          {mlir::dyn_cast<mlir::ElementsAttr>(attr->getValue()).getType()});

      const auto cstFullName = get_versioned_name("Constant");
      auto op = createOnnxOp(&builder, cstFullName, types, {}, attrs);

      block->push_back(op);
      // store output
      auto res_ptr = std::make_shared<mlir::Value>(op->getResult(0));
      res_by_outputs.insert({node.output()[0], res_ptr});
    }
  }

  // Step 2, add the initializers
  for (const auto &initializer : graph_proto.initializer()) {
    // attributes
    std::vector<mlir::NamedAttribute> attrs;
    auto value = OnnxToMlir_Tensor(initializer, mlirCtx);
    mlir::NamedAttribute attr("value", value);
    attrs.push_back(attr);
    // origin
    auto nameVal = builder.getStringAttr(initializer.name());
    mlir::NamedAttribute labels("onnx.init.name", nameVal);
    attrs.push_back(labels);
    // result type
    auto types = std::vector<mlir::Type>({value.getType()});

    const auto cstFullName = get_versioned_name("Constant");
    auto op = createOnnxOp(&builder, cstFullName, types, {}, attrs);

    block->push_back(op);
    // map to i/o
    auto res_ptr = std::make_shared<mlir::Value>(op->getResult(0));
    res_by_outputs.insert({initializer.name(), res_ptr});
  }

  // Step 3, add a NoneType constant
  mlir::Operation *notype;
  {
    std::vector<mlir::NamedAttribute> attrs;
    // origin
    auto nameVal = builder.getStringAttr("NoneType");
    mlir::NamedAttribute labels("mlir.value", nameVal);
    attrs.push_back(labels);
    // result type
    mlir::Type noneType = mlir::NoneType::get(mlirCtx);
    auto types = std::vector<mlir::Type>({noneType});

    const auto cstFullName = get_versioned_name("Constant");
    notype = createOnnxOp(&builder, cstFullName, types, {}, attrs);
    block->push_back(notype);
  }

  // Step 4, add all nodes
  for (const auto &node : graph_proto.node()) {
    const auto opFullName = get_versioned_name(node.op_type());
    if (node.op_type() != "Constant") {
      // Pass 1, set attributes
      std::vector<mlir::NamedAttribute> attrs;
      for (const auto &attribute : node.attribute()) {
        auto attr = OnnxToMlir_Attr(attribute, mlirCtx);
        attrs.push_back(*attr);
      }
      // attributes origin
      auto nameVal = builder.getStringAttr(node.name());
      mlir::NamedAttribute labels("onnx.node.name", nameVal);
      attrs.push_back(labels);
      // Pass 2, set result types
      std::vector<mlir::Type> types;
      int nResults = getOnnxOpNumIO(opFullName, /*outputs*/ true);
      // variadic case
      if (nResults < 0)
        nResults = node.output().size();
      for (int i = 0; i < nResults; ++i) {
        // unmapped outputs
        if (i >= node.output().size()) {
          types.push_back(mlir::NoneType::get(mlirCtx));
          continue;
        }
        // node output to set
        const auto out = node.output()[i];
        // lookup in vis table
        auto it = vis.find(out);
        if (it != vis.end()) {
          auto type = OnnxToMlir_Type(*(it->second), mlirCtx);
          types.push_back(type);
        } else {
          // lookup neighbour outputs
          auto oitr = ops_by_inputs.find(out);
          if (oitr != ops_by_inputs.end()) {
            auto oadj = *oitr->second;
            // oadj->print(llvm::errs());
            // TODO(cbalint13): which result ?!
            types.push_back(oadj->getResults()[0].getType());
          }
          // lookup main func outputs
          auto fitr = func_outputs.find(out);
          if (fitr != func_outputs.end()) {
            auto fadj = *fitr->second;
            // fadj.print(llvm::errs());
            types.push_back(fadj);
          }
        }
      }
      // Pass 3, set operands (temporary)
      std::vector<mlir::Value> operands;
      int nOperands = getOnnxOpNumIO(opFullName, /*inputs*/ false);
      // variadic case
      if (nOperands < 0)
        nOperands = node.input().size();
      for (int i = 0; i < nOperands; ++i) {
        operands.push_back(notype->getResult(0));
      }
      // Pass 4, create the operation
      auto op = createOnnxOp(&builder, opFullName, types, operands, attrs);
      block->push_back(op);
      // Pass 5, map the operation by i/o
      auto op_ptr = std::make_shared<mlir::Operation *>(op);
      for (const auto &inp : node.input())
        ops_by_inputs.insert({inp, op_ptr});
      for (int idx = 0; idx < node.output().size(); ++idx) {
        auto res_ptr = std::make_shared<mlir::Value>(op->getResult(idx));
        res_by_outputs.insert({node.output()[idx], res_ptr});
      }
      ops_by_name.insert({node.name(), op_ptr});
    }
  }

  // Step 5, set operands (final)
  for (const auto &node : graph_proto.node()) {
    // skip on no inputs
    if (node.input().size() == 0)
      continue;
    // set all operands by inputs
    auto node_op = *ops_by_name[node.name()];
    for (int idx = 0; idx < node.input().size(); ++idx) {
      // unused one
      if (node.input()[idx].size() == 0)
        continue;
      // map node input to main func input
      auto f_it = func_inputs.find(node.input()[idx]);
      if (f_it != func_inputs.end()) {
        mlir::Value arg = *(f_it->second);
        node_op->setOperand(idx, arg);
      }
      // map node input to upper node output
      auto r_it = res_by_outputs.find(node.input()[idx]);
      if (r_it != res_by_outputs.end()) {
        auto res = *(r_it->second);
        node_op->setOperand(idx, res);
        continue;
      }
    }
  }

  // Step 6, set main func results
  mlir::SmallVector<mlir::Value> ret_vals;
  for (const auto &out : func_outputs) {
    auto r_it = res_by_outputs.find(out.first);
    if (r_it != res_by_outputs.end()) {
      auto res = *(r_it->second);
      ret_vals.push_back(res);
    }
  }
  auto ret =
      mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), ret_vals);
  block->push_back(ret);

  // remove unused NoType constant
  if (notype->getResults()[0].use_empty())
    notype->erase();
}

void ONNXImporter::parse_graph_io(const onnx::GraphProto &graph_proto) {
  // main inputs
  std::vector<mlir::Type> inputs;
  for (const auto &input : graph_proto.input()) {
    if (input.has_type()) {
      inputs.push_back(OnnxToMlir_Type(input, mlirCtx));
    } else {
      llvm::errs() << "ERROR: Type Not Specified.\n";
      exit(-1);
    }
  }

  // main outputs
  std::vector<mlir::Type> outputs;
  for (const auto &output : graph_proto.output()) {
    if (output.has_type()) {
      outputs.push_back(OnnxToMlir_Type(output, mlirCtx));
    } else {
      llvm::errs() << "ERROR: Type Not Specified.\n";
      exit(-1);
    }
  }

  // mlir graph main function
  mlir::OpBuilder builder(mlirCtx);
  auto funcType = mlir::FunctionType::get(mlirCtx, inputs, outputs);
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

  module.push_back(func);
}

void ONNXImporter::import(const std::string &file_or_string,
                          mlir::MLIRContext *ctx) {
  bool verbose = false;
  if (opt_args.count("--verbose") > 0)
    verbose = true;

  // context setup
  mlirCtx = ctx;
  mlirCtx->loadDialect<mlir::affine::AffineDialect,
                       mlir::complex::ComplexDialect, mlir::func::FuncDialect,
                       mlir::sparse_tensor::SparseTensorDialect,
                       onnx2mlir::dialect::onnx::OnnxDialect>();
  // initialize the module
  mlir::OpBuilder builder(mlirCtx);
  module = mlir::ModuleOp::create(builder.getUnknownLoc());

  if (verbose) {
    llvm::outs() << "ONNX engine version: " << onnx::LAST_RELEASE_VERSION;
    llvm::outs() << "\n";
    llvm::outs() << "ONNX engine IR version: " << onnx::IR_VERSION;
    llvm::outs() << "\n";
  }

  // get ops versioning
  engine_opset_version = -1;
  auto all_schemas = onnx::OpSchemaRegistry::get_all_schemas_with_history();
  for (const auto &schema : all_schemas) {
    if (engine_opset_version < schema.SinceVersion())
      engine_opset_version = schema.SinceVersion();
    ops_versions[schema.Name()].push_back(schema.SinceVersion());
  }

  if (verbose) {
    llvm::outs() << "ONNX engine opset version: " << engine_opset_version;
    llvm::outs() << "\n\n";
  }

  // parse onnx source
  onnx::ModelProto model_import;
  if (opt_args.count("--import-serialized") > 0) {
    if (!model_import.ParseFromString(file_or_string)) {
      llvm::errs() << "ERROR: ONNX model string parsing error." << "\n";
      exit(-1);
    }
    if (verbose)
      llvm::outs() << "Model data: " << file_or_string.length()
                   << " bytes <serialized>\n";
  } else {
    if (verbose)
      llvm::outs() << "Model path: " << file_or_string << "\n";
    std::ifstream model_file(file_or_string, std::ios::binary);
    if (!model_file.is_open()) {
      llvm::errs() << "Error opening file: " << file_or_string << "\n";
      exit(-1);
    }
    if (!model_import.ParseFromIstream(&model_file)) {
      llvm::errs() << "ERROR: ONNX model file parsing error.\n";
      exit(-1);
    }
  }

  model_opset_version = -1;
  // see https://github.com/onnx/onnx/blob/main/onnx/docs/Versioning.md
  for (auto it = model_import.opset_import().begin();
       it != model_import.opset_import().end(); ++it) {
    if (it->domain() == "" || it->domain() == "ai.onnx") {
      model_opset_version = it->version();
      break;
    }
  }

  if (verbose)
    llvm::outs() << "Model IR version: " << model_import.ir_version() << "\n";

  // convert model
  onnx::ModelProto model_proto;
  if (opt_args.count("--onnx-convert-ops") > 0) {
    int convert_version = engine_opset_version;
    if (opt_args["--onnx-convert-ops"].size() > 0)
      convert_version = std::stoi(opt_args["--onnx-convert-ops"]);
    if (convert_version <= model_opset_version) {
      llvm::errs() << "ERROR: Model cannot be downgraded.\n";
      exit(-1);
    }
    if (verbose)
      llvm::outs() << "Model OPSet conversion: " << model_opset_version
                   << " -> " << convert_version << "\n";
    try {
      model_proto = onnx::version_conversion::ConvertVersion(model_import,
                                                             convert_version);
      model_opset_version = convert_version;
    } catch (const std::exception &e) {
    }
  } else {
    model_proto = model_import;
  }
  if (verbose)
    llvm::outs() << "Model OPSet version: " << model_opset_version << "\n";

  // infer shapes
  onnx::ShapeInferenceOptions iopts(
      /*check_type=*/false, /*error_mode=*/false, /*data_prop=*/true);
  onnx::shape_inference::InferShapes(
      model_proto, onnx::OpSchemaRegistry::Instance(), iopts, {});

  const onnx::GraphProto &graph_proto = model_proto.graph();

  if (verbose) {
    llvm::outs() << "\n";
    llvm::outs() << "Graph Name: " << graph_proto.name() << "\n";
  }

  /*
   * MLIR ONNX
   */

  // construct function args
  parse_graph_io(graph_proto);

  // pupulate body operators
  parse_graph_nodes(graph_proto);

  // verify module
  if (llvm::failed(mlir::verify(module))) {
    llvm::errs() << "MLIR module verification failed.\n";
    exit(-1);
  }
}

} // end namespace onnx2mlir::frontend
