//#include <onnx/common/version.h>
//#include <onnx/onnx_pb.h>
//#include <onnx/shape_inference/implementation.h>
//#include <onnx/version_converter/convert.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/IR/Verifier.h>

#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
#include "onnx2mlir/dialect/onnx/OnnxInterface.hpp"
#include "onnx2mlir/dialect/onnx/OnnxOps.hpp"

#include "onnx2mlir/frontend/onnx.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "onnx2mlir/frontend/onnx.hpp"

// mlir::Value ImportTensor(const onnx::TensorProto &tensor) {
//   mlir::ElementsAttr mlirAttr =
//     onnxTensorProtoToElmAttr(&context_, options_.externalDataDir, tensor);
//  Use the tensor name as Location.
//  auto loc =
//    mlir::NameLoc::get(builder_.getStringAttr("Initializer_" +
//    tensor.name()));
//  Value initializer = createConstantValue(mlirAttr, loc);
//  num_of_parameters_ += mlirAttr.getShapedType().getNumElements();
//  return initializer;
//}


void mlir_tosa_sample() {

  /// MLIR
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::func::FuncDialect, mlir::tosa::TosaDialect>();
  mlirCtx.disableMultithreading();


  /*
    /// Configure pass manager
    mlir::PassManager pm(&mlirCtx, mlir::ModuleOp::getOperationName(),
                         mlir::PassManager::Nesting::Implicit);
  */
  // create builder
  mlir::OpBuilder builder(&mlirCtx);

  /// create module
  mlir::ModuleOp mod = mlir::ModuleOp::create(builder.getUnknownLoc());

  /// create function
  auto funcType = mlir::FunctionType::get(
      &mlirCtx,
      {mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()),
       mlir::RankedTensorType::get({1,1,1}, builder.getF32Type())},
      {mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()),
       mlir::RankedTensorType::get({1,1,1}, builder.getF32Type())});
      //mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()));
  //auto funcType = mlir::FunctionType::get( &mlirCtx,{},{});
  mlir::func::FuncOp func = mlir::func::FuncOp::create(builder.getUnknownLoc(), "main_func", funcType, /*attr*/ {});

//  builder.setInsertionPointToStart(&func.getBody().back());
//  func.getBody().addArgument(funcType, builder.getUnknownLoc());
  //func.setType(funcType);
  /// add func to module
  mod.push_back(func);
  


  /*
  module attributes {module.state = "TOSA_F32", top.weight_file = "add_top_weight.npz"}
  { func.func @main(%arg0: tensor<1x2x2x72xf32>, %arg1: tensor<1x2x2x72xf32>) -> tensor<1x2x72x2xf32>
    { %0 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
      %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x2x2x72xf32>, tensor<4xi32>) -> tensor<1x2x72x2xf32>
      %2 = "tosa.transpose"(%arg1, %0) : (tensor<1x2x2x72xf32>, tensor<4xi32>) -> tensor<1x2x72x2xf32>
      %3 = "tosa.add"(%1, %2) : (tensor<1x2x72x2xf32>, tensor<1x2x72x2xf32>) -> tensor<1x2x72x2xf32>
      return %3 : tensor<1x2x72x2xf32>
    }
  } */



  mlir::Block *block = func.addEntryBlock();
  mlir::Value arg1 = block->getArgument(0);
  mlir::Value arg2 = block->getArgument(1);

//  auto sum = builder.create<mlir::tosa::AddOp>(builder.getUnknownLoc(), mlir::RankedTensorType::get({1}, builder.getF32Type()), arg1, arg2);
//  block->push_back(sum);
//  auto ret = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), sum.getOutput());
//  block->push_back(ret);


  auto mul = builder.create<mlir::tosa::MatMulOp>(builder.getUnknownLoc(), mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()), arg1, arg2);
  block->push_back(mul);


  llvm::ArrayRef<float> b0 = { 33.3f };
  auto inputType = mlir::RankedTensorType::get({1,1,1}, builder.getF32Type());
  mlir::ElementsAttr inputAttr = mlir::DenseElementsAttr::get(inputType, b0);
  auto bias = builder.create<mlir::tosa::ConstOp>(builder.getUnknownLoc(), inputType, inputAttr);
  block->push_back(bias);

  auto sum = builder.create<mlir::tosa::AddOp>(builder.getUnknownLoc(), mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()), mul, bias);
  block->push_back(sum);

  auto ret = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::SmallVector<mlir::Value>({static_cast<mlir::Value>(sum), static_cast<mlir::Value>(mul)}));
  block->push_back(ret);


  //auto ret = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), static_cast<mlir::Value>(mul));




    //build(mlir::OpBuilder&, mlir::OperationState&, mlir::Type, mlir::Value, mlir::Value, mlir::Value, mlir::DenseI64ArrayAttr, mlir::DenseI64ArrayAttr, mlir::DenseI64ArrayAttr)
//  Value tempConv2D = tosa::CreateOpAndInfer<mlir::tosa::Conv2DOp>(rewriter, op->getLoc(), newConvOutputType, newSliceInput, newSliceWeight, newSliceBias, pads, strides, dilations);

/*

    mlir::RankedTensorType inputType = mlir::RankedTensorType::get({1, 28, 28, 3}, builder.getF32Type());
    mlir::RankedTensorType filterType = mlir::RankedTensorType::get({3, 3, 3, 16}, builder.getF32Type());
    mlir::RankedTensorType biasType = mlir::RankedTensorType::get({16}, builder.getF32Type());

    mlir::ElementsAttr attr = {};
    mlir::Value inputValue = builder.create<mlir::tosa::ConstOp>(builder.getUnknownLoc(), inputType, attr);
    mlir::Value filterValue = builder.create<mlir::tosa::ConstOp>(builder.getUnknownLoc(), filterType, attr);
    mlir::Value biasValue = builder.create<mlir::tosa::ConstOp>(builder.getUnknownLoc(), biasType, attr);

    mlir::DenseI64ArrayAttr strides = builder.getDenseI64ArrayAttr({1, 1});
    mlir::DenseI64ArrayAttr dilations = builder.getDenseI64ArrayAttr({1, 1});
    mlir::DenseI64ArrayAttr padding = builder.getDenseI64ArrayAttr({0, 0, 0, 0});

    //mlir::Value out = builder.create<mlir::tosa::Conv2DOp>(builder.getUnknownLoc(), builder.getF32Type(), inputValue, filterValue, biasValue, padding, strides, dilations);

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
*/
  // graph_proto.GetTypeName().

  std::cout << std::endl;
  std::cout << "============== MLIR IR ============ " << std::endl;

  if (llvm::failed(mlir::verify(mod))) {
    llvm::errs() << "Module verification failed!\n";
    mod.dump();
    exit(-1);
  }

  // Print the MLIR
   // Customize printing flags
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  mod.print(llvm::outs(), flags);
}

void mlir_onnx_sample() {

  /// MLIR
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::func::FuncDialect, mlir::tosa::TosaDialect, onnx2mlir::dialect::onnx::OnnxDialect>();
  mlirCtx.disableMultithreading();

  /*
    /// Configure pass manager
    mlir::PassManager pm(&mlirCtx, mlir::ModuleOp::getOperationName(),
                         mlir::PassManager::Nesting::Implicit);
  */
  // create builder
  mlir::OpBuilder builder(&mlirCtx);

  /// create module
  mlir::ModuleOp mod = mlir::ModuleOp::create(builder.getUnknownLoc());

/*
func.func @main_graph(%arg0: tensor<1x1x129x124xf32> {onnx.name = "input_3"}) -> (tensor<1x8xf32> {onnx.name = "Identity"}) {
func.func @main_func(%arg0: tensor<1x1x1xf32, {onnx.name = "input"}>, %arg1: tensor<1x1x1xf32>) -> (tensor<1x1x1xf32>, tensor<1x1x1xf32>)
module {
  func.func @main_func(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1x1xf32>) -> (tensor<1x1x1xf32>, tensor<1x1x1xf32>) {
    %0 = "onnx.Constant"() <{value = dense<3.330000e+01> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %1 = "onnx.Constant"() <{value = dense<3.330000e+01> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %2 = "onnx.Add"(%0, %1) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    return %2, %2 : tensor<1x1x1xf32>, tensor<1x1x1xf32>
  }
}
*/

  // attribute
  mlir::StringAttr nameKey = builder.getStringAttr("onnx.name");
  mlir::StringAttr nameVal = builder.getStringAttr("atensor");
  mlir::NamedAttribute namedAttr(nameKey, nameVal);
  mlir::Attribute encodedAttr = builder.getDictionaryAttr(namedAttr);

  /// create function
  auto funcType = mlir::FunctionType::get(
      &mlirCtx, // context
      {mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()), // input
       mlir::RankedTensorType::get({1,1,1}, builder.getF32Type(), encodedAttr)},
      {mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()), // output
       mlir::RankedTensorType::get({1,1,1}, builder.getF32Type())});
      //mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()));
  //auto funcType = mlir::FunctionType::get( &mlirCtx,{},{});
  mlir::func::FuncOp func = mlir::func::FuncOp::create(builder.getUnknownLoc(), "main_func", funcType, namedAttr);

  mlir::StringAttr anattr = builder.getStringAttr("input");
  func.setArgAttr(0, "onnx.name", anattr);

  /// add func to module
  mod.push_back(func);



  mlir::Block *block = func.addEntryBlock();
  mlir::Value arg1 = block->getArgument(0);
  mlir::Value arg2 = block->getArgument(1);


//  auto mul = builder.create<mlir::tosa::MatMulOp>(builder.getUnknownLoc(), mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()), arg1, arg2);
//  block->push_back(mul);


  llvm::ArrayRef<float> b0 = { 33.3f };
  auto inputType = mlir::RankedTensorType::get({1,1,1}, builder.getF32Type());
  mlir::ElementsAttr inputAttr = mlir::DenseElementsAttr::get(inputType, b0);

/*
  mlir::Operation* c0 = builder.create<onnx2mlir::dialect::onnx::ConstantOp>(builder.getUnknownLoc(), inputType, inputAttr);
  block->push_back(c0);

  auto c1 = builder.create<onnx2mlir::dialect::onnx::ConstantOp>(builder.getUnknownLoc(), inputType, inputAttr);
  block->push_back(c1);

//  auto sum = builder.create<onnx2mlir::dialect::onnx::AddOp>(builder.getUnknownLoc(), mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()), c0, c1);
  auto sum = builder.create<onnx2mlir::dialect::onnx::AddOp>(builder.getUnknownLoc(), mlir::RankedTensorType::get({1,1,1}, builder.getF32Type()), arg1, c1);
  block->push_back(sum);

  auto ret = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::SmallVector<mlir::Value>({static_cast<mlir::Value>(sum), static_cast<mlir::Value>(sum)}));
  block->push_back(ret);
*/

//  auto ret = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), static_cast<mlir::Value>(sum));
//  block->push_back(ret);


  std::cout << std::endl;
  std::cout << "============== MLIR IR ============ " << std::endl;

  if (llvm::failed(mlir::verify(mod))) {
    llvm::errs() << "Module verification failed!\n";
    mod.dump();
    exit(-1);
  }

  // Print the MLIR
   // Customize printing flags
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  mod.print(llvm::outs(), flags);

  // TESTS
  printf("NUMARGAS [%i]\n", func.getNumArguments());
  mlir::ArrayAttr argAttrs = func.getArgAttrsAttr();
  argAttrs.print(llvm::outs());

  auto xxx = func.getAllArgAttrs();
  xxx.print(llvm::outs());
}

int main(int argc, char **argv) {
  /// command-line params
  bool printUsage = false;
  std::string ONNXFilename = "";

  /// command-line parser
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      if (!strcmp(argv[i], "--help")) {
        printUsage = true;
        i++;
        continue;
      }
      std::cout << "Unknown option: " << argv[i] << std::endl;
      printUsage = true;
      break;
    } else if (argv[i][0] != '-') {
      if (!ONNXFilename.size()) {
        ONNXFilename = argv[i];
        continue;
      }
    }
  }
  if (!ONNXFilename.size()) {
    std::cout << "ERROR: missing onnx_file" << std::endl;
  }
  /// check required options
  if (!ONNXFilename.size() || printUsage) {
    std::cout << std::endl;
    std::cout << "Usage: onnx2mlir [--help]\n"
              << "             onnx_file\n\n";
    exit(-1);
  }

  auto ONNXLoader = new onnx2mlir::Importer<onnx2mlir::frontend::ONNXImporter>();
  auto ONNXConverter = new onnx2mlir::Converter<onnx2mlir::frontend::ONNXConverter>();
  ONNXLoader->importModule(ONNXFilename);
  ONNXConverter->convertModule(ONNXLoader->getMLIRModule());

  //mlir_tosa_sample();
  //mlir_onnx_sample();

  return 0;
}
