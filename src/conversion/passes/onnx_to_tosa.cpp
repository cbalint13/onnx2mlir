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
 * \file src/conversion/onnx_to_tosa.cpp
 * \brief Onnx to Linalg dialect conversion
 */

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TOSA.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <utility>

#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

/*
 * onnx::Conv -> tosa::Conv2D
 */
/*
struct ONNXConvToTOSAConvPattern
    : public mlir::OpConversionPattern<onnx::ConvOp> {
  // Use a constructor that takes the MLIRContext.
  // This is the recommended way to pass context to patterns.
  explicit ONNXConvToTOSAConvPattern(mlir::MLIRContext *context)
      : mlir::OpConversionPattern<onnx::ConvOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(onnx::ConvOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // get input, weight and optional bias
    mlir::Value input = adaptor.getX();
    mlir::Value weight = adaptor.getW();
    mlir::Value bias = adaptor.getB();

    // get attributes
    auto dilations = op.getDilations();
    auto strides = op.getStrides();
    auto pads = op.getPads();
    mlir::IntegerAttr group = op.getGroupAttr();

    llvm::SmallVector<int64_t> dilationVals;
    if (dilations.has_value()) {
      for (mlir::Attribute attr : dilations.value().getValue())
        dilationVals.push_back(
            mlir::dyn_cast<mlir::IntegerAttr>(attr).getInt());
    } else {
      dilationVals = {1, 1};
    }

    llvm::SmallVector<int64_t> strideVals;
    if (strides.has_value()) {
      for (mlir::Attribute attr : strides.value().getValue())
        strideVals.push_back(mlir::dyn_cast<mlir::IntegerAttr>(attr).getInt());
    } else {
      strideVals = {1, 1};
    }

    llvm::SmallVector<int64_t> tosaPads;
    if (pads.has_value()) {
      auto padVals = pads.value().getValue();
      // pad_h_start (top)
      tosaPads.push_back(
          mlir::dyn_cast<mlir::IntegerAttr>(padVals[0]).getInt());
      // pad_h_end (bottom)
      tosaPads.push_back(
          mlir::dyn_cast<mlir::IntegerAttr>(padVals[2]).getInt());
      // pad_w_start (left)
      tosaPads.push_back(
          mlir::dyn_cast<mlir::IntegerAttr>(padVals[1]).getInt());
      // pad_w_end (right)
      tosaPads.push_back(
          mlir::dyn_cast<mlir::IntegerAttr>(padVals[3]).getInt());
    } else {
      tosaPads = {0, 0, 0, 0};
    }

    // ONNX 'group' attribute can be 1, but tosa.conv2d doesn't take 'group'.
    // TOSA's `conv2d` fundamentally operates on groups.
    // If `group` is not 1, this conversion is not directly possible
    // without splitting the convolution into multiple `tosa.conv2d` ops
    // or using a different TOSA op. For simplicity, we'll enforce group == 1.
    if (group && group.getInt() != 1) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group value for tosa.conv2d");
    }

    // get output type
    mlir::Type outputType = op.getResult().getType();

    // extra Tosa Conv2d handles

    if (!bias) {
      // bias cannot be optional in TOSA
      mlir::RankedTensorType biasType = mlir::RankedTensorType::get(
          {mlir::cast<mlir::ShapedType>(outputType)
               .getShape()[1]}, // Channels dimension
          mlir::cast<mlir::ShapedType>(outputType).getElementType());

      mlir::Attribute biasValue = mlir::DenseElementsAttr::get(
          biasType, rewriter.getZeroAttr(biasType.getElementType()));
      bias = rewriter.create<mlir::tosa::ConstOp>(
          op.getLoc(), biasType, mlir::cast<mlir::ElementsAttr>(biasValue));
    }

    // input_zp, weight_zp
    mlir::Type zpType = rewriter.getI32Type();
    mlir::Attribute zeroScalarAttr = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get({}, zpType), rewriter.getI32IntegerAttr(0));

    mlir::Value inputZpOperand = rewriter.create<mlir::tosa::ConstOp>(
        op.getLoc(), mlir::RankedTensorType::get({}, zpType),
        mlir::cast<mlir::ElementsAttr>(zeroScalarAttr));

    mlir::Value weightZpOperand = rewriter.create<mlir::tosa::ConstOp>(
        op.getLoc(), mlir::RankedTensorType::get({}, zpType),
        mlir::cast<mlir::ElementsAttr>(zeroScalarAttr));

    // acctype
    mlir::Type accType =
        mlir::cast<mlir::ShapedType>(outputType).getElementType();
    mlir::TypeAttr accTypeAttr = mlir::TypeAttr::get(accType);
    // localboundattr
    mlir::BoolAttr localBoundAttr = rewriter.getBoolAttr(false);

    mlir::tosa::Conv2DOp convOp = rewriter.create<mlir::tosa::Conv2DOp>(
        op.getLoc(), outputType, input, weight, bias, inputZpOperand,
        weightZpOperand,
        mlir::DenseI64ArrayAttr::get(rewriter.getContext(), tosaPads),
        mlir::DenseI64ArrayAttr::get(rewriter.getContext(), strideVals),
        mlir::DenseI64ArrayAttr::get(rewriter.getContext(), dilationVals),
        accTypeAttr, localBoundAttr);

    // replace onnx op
    rewriter.replaceOp(op, convOp.getResult());

    return mlir::success();
  }
};*/

static inline bool opNameBeginsWith(const llvm::StringRef &OpName,
                                    const std::string &match) {
  return std::regex_match(OpName.str(), std::regex("^onnx." + match + ".*"));
}

struct ONNXToTOSALowering : public mlir::RewritePattern {
  explicit ONNXToTOSALowering(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(),
                             /*PatternBenefit=*/true, ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // triage by onnx operation names
    llvm::StringRef opName = op->getName().getStringRef();
    /*
     * onnx::Constant
     */
    if (opNameBeginsWith(opName, "Constant")) {
      // onnx::Constant
      mlir::Attribute valueAttr = op->getAttr("value");
      auto elemValueAttr =
          mlir::dyn_cast_or_null<mlir::ElementsAttr>(valueAttr);

      // tosa cannot handle NoneType return
      if (mlir::isa<mlir::NoneType>(op->getResult(0).getType())) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Constant with 'NoneType' is not supported");
      }

      if (!elemValueAttr) {
        // tosa cannot handle empty tensor
        return rewriter.notifyMatchFailure(
            op, "onnx.Constant without a valid tensor 'value' attribute");
      }

      // replace with tosa::Const
      rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(
          op, elemValueAttr.getType(), elemValueAttr);

      return mlir::success();

      /*
       * onnx::Unsqueeze
       */
    } else if (opNameBeginsWith(opName, "Unsqueeze")) {
      mlir::Value inputData = op->getOperand(0);
      mlir::Value axesTensor = op->getOperand(1);
      mlir::Location loc = op->getLoc();

      op->print(llvm::outs());
      // %37 = "onnx.Unsqueeze"(%arg0, %1) {onnx.node.name = "/Unsqueeze"} :
      // (tensor<1616x2880x3xui8>, tensor<1xi64>) -> tensor<1x1616x2880x3xui8>
      llvm::outs() << "\n" << inputData << "\n";
      // <block argument> of type 'tensor<1616x2880x3xui8>' at index: 0
      llvm::outs() << "\n" << axesTensor << "\n";
      // %1 = "onnx.Constant"() <{value = dense<0> : tensor<1xi64>}>
      // {onnx.node.name = "/Constant"} : () -> tensor<1xi64>

      mlir::ShapedType inputShapeType =
          mlir::dyn_cast<mlir::ShapedType>(inputData.getType());
      if (!inputShapeType || !inputShapeType.hasRank()) {
        return rewriter.notifyMatchFailure(
            op, "Unsqueeze input must have a ranked shaped type.");
      }

      mlir::DenseIntElementsAttr axesAttr;
      if (mlir::Operation *definingOp = axesTensor.getDefiningOp()) {
        if (definingOp->getName().getStringRef() == "onnx.Constant") {
          mlir::Attribute attr = definingOp->getAttr("value");
          axesAttr = mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(attr);
        }
      }

      if (!axesAttr) {
        return rewriter.notifyMatchFailure(
            op, "Unsqueeze 'axes' must be a constant integer tensor from "
                "onnx.Constant for TOSA conversion.");
      }

      llvm::ArrayRef<int64_t> inputDims = inputShapeType.getShape();
      int64_t inputRank = inputShapeType.getRank();

      llvm::SmallVector<int64_t> axesValues;
      for (auto val : axesAttr.getValues<int64_t>()) {
        axesValues.push_back(val);
      }
      std::sort(axesValues.begin(), axesValues.end());
      axesValues.erase(std::unique(axesValues.begin(), axesValues.end()),
                       axesValues.end());

      int64_t outputRank = inputRank + axesValues.size();
      llvm::SmallVector<int64_t> newShapeVector(outputRank, -1);

      for (int64_t axis : axesValues) {
        if (axis < 0) {
          axis += outputRank;
        }
        if (axis < 0 || axis >= outputRank) {
          return rewriter.notifyMatchFailure(
              op, "Unsqueeze 'axes' value out of bounds for output rank.");
        }
        newShapeVector[axis] = 1;
      }

      int64_t currentInputDimIdx = 0;
      for (int64_t i = 0; i < outputRank; ++i) {
        if (newShapeVector[i] == -1) {
          if (currentInputDimIdx >= inputRank) {
            return rewriter.notifyMatchFailure(
                op, "Unsqueeze logic error: not enough input dimensions for "
                    "new shape.");
          }
          newShapeVector[i] = inputDims[currentInputDimIdx++];
        }
      }

      mlir::Type newResultType = mlir::RankedTensorType::get(
          newShapeVector, inputShapeType.getElementType());

      mlir::Attribute shapeAttr = rewriter.getDenseI64ArrayAttr(newShapeVector);
      mlir::NamedAttribute shapeNamedAttr =
          rewriter.getNamedAttr("shape", shapeAttr);

      rewriter.replaceOpWithNewOp<mlir::tosa::ReshapeOp>(
          op, mlir::TypeRange(newResultType),
          mlir::ValueRange(std::vector<mlir::Value>({inputData, axesTensor})),
          mlir::ArrayRef<mlir::NamedAttribute>(shapeNamedAttr));
      return mlir::success();
    }

    return mlir::success();
  }
};

// ONNX dialect to TOSA dialect pass
struct LowerONNXToTOSAPass
    : public mlir::PassWrapper<LowerONNXToTOSAPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerONNXToTOSAPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tosa::LinalgDialect>();
    registry.insert<onnx::OnnxDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    // enlist all operations by name
    std::set<std::string> onnx_op_names;
    for (mlir::RegisteredOperationName opName :
         ctx->getRegisteredOperationsByDialect("onnx")) {
      onnx_op_names.insert(opName.getStringRef().str());
    }
    mlir::ConversionTarget target(*ctx);

    // legal dialects
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::tosa::LinalgDialect>();
    // illegal dialects
    target.addIllegalDialect<onnx::OnnxDialect>();

    // legal operations
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    // allow onnx NoneType (postpone rewrite)
    for (const auto &opName : onnx_op_names) {
      if (opNameBeginsWith(opName, "Constant")) {
        target.addDynamicallyLegalOp(
            mlir::OperationName(opName, ctx), [](mlir::Operation *op) {
              return mlir::isa<mlir::NoneType>(op->getResult(0).getType());
            });
      }
    }

    // create a set of patterns.
    mlir::RewritePatternSet patterns(ctx);

    // add ONNX to TOSA lowering pattern
    patterns.add<ONNXToTOSALowering>(ctx);

    // apply the partial conversion pattern
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
      exit(-1);
    }
  }
};

std::unique_ptr<mlir::Pass> createLowerONNXToTOSAPass() {
  return std::make_unique<onnx2mlir::dialect::LowerONNXToTOSAPass>();
}

} // namespace onnx2mlir::dialect
