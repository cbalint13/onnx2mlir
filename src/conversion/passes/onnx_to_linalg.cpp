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
 * \file src/conversion/onnx_to_linalg.cpp
 * \brief Onnx to Linalg dialect conversion
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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

static inline bool opNameBeginsWith(const llvm::StringRef &OpName,
                                    const std::string &match) {
  return std::regex_match(OpName.str(), std::regex("^onnx." + match + ".*"));
}

struct ONNXToLINALGLowering : public mlir::RewritePattern {
  explicit ONNXToLINALGLowering(mlir::MLIRContext *ctx)
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
      // cannot handle NoneType return
      if (mlir::isa<mlir::NoneType>(op->getResult(0).getType())) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Constant with 'NoneType' is not supported");
      }

      mlir::Attribute valueAttr = op->getAttr("value");
      auto elemValueAttr =
          mlir::dyn_cast_or_null<mlir::ElementsAttr>(valueAttr);

      // cannot handle empty tensor
      if (!elemValueAttr) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Constant without a valid tensor 'value' attribute");
      }

      // replace with arith::ConstantOp
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
          op, elemValueAttr.getType(), elemValueAttr);

      return mlir::success();

      /*
       * onnx::Unsqueeze
       */
    } else if (opNameBeginsWith(opName, "Unsqueeze")) {
      // 1. Access the 'data' input operand (operand 0).
      mlir::Value inputData = op->getOperand(0);
      mlir::RankedTensorType inputType =
          mlir::dyn_cast<mlir::RankedTensorType>(inputData.getType());
      if (!inputType) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Unsqueeze input must be a ranked tensor");
      }

      // 2. Access the output result type (result 0).
      if (op->getNumResults() != 1) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Unsqueeze expected a single result");
      }

      mlir::RankedTensorType outputType =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
      if (!outputType) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Unsqueeze output must be a ranked tensor");
      }

      // 3. Access the 'axes' input operand (operand 1).
      mlir::Value axesValue = op->getOperand(1);
      mlir::Operation *definingOp = axesValue.getDefiningOp();
      if (!definingOp) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Unsqueeze axes operand must be an op for static axes");
      }
      mlir::Attribute axesAttr = definingOp->getAttr("value");
      mlir::DenseIntElementsAttr axesDenseAttr =
          mlir::dyn_cast<mlir::DenseIntElementsAttr>(axesAttr);
      if (!axesDenseAttr) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Unsqueeze axes value is not DenseIntElementsAttr");
      }
      llvm::SmallVector<int64_t> axes;
      for (auto val : axesDenseAttr.getValues<mlir::APInt>()) {
        axes.push_back(val.getSExtValue());
      }
      for (auto &axis : axes) {
        if (axis < 0) {
          axis += outputType.getRank();
        }
      }
      std::sort(axes.begin(), axes.end());

      // 4. Construct the output shape components (static / dynamic)
      llvm::SmallVector<int64_t> staticOutputShape;
      llvm::SmallVector<mlir::Value> dynamicOutputShapeValues;
      llvm::SmallVector<mlir::Value> outputShapeSSAValues;

      int64_t currentInputDimIdx = 0;

      for (int64_t i = 0; i < outputType.getRank(); ++i) {
        if (std::binary_search(axes.begin(), axes.end(), i)) {
          // an unsqueezed dimension (new dim of size 1)
          staticOutputShape.push_back(1);
          // create an arith.constant for '1'
          outputShapeSSAValues.push_back(
              rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 1));
        } else {
          // output dimension corresponds to an original input dimension.
          if (currentInputDimIdx >= inputType.getRank()) {
            return rewriter.notifyMatchFailure(
                op, "onnx.Unsqueeze input-output dimension mismatch during "
                    "reassociation construction");
          }
          if (inputType.isDynamicDim(currentInputDimIdx)) {
            // dynamic dimensions
            mlir::Value dimValue = rewriter.create<mlir::tensor::DimOp>(
                op->getLoc(), inputData,
                rewriter.create<mlir::arith::ConstantIndexOp>(
                    op->getLoc(), currentInputDimIdx));
            dynamicOutputShapeValues.push_back(dimValue);
            staticOutputShape.push_back(mlir::ShapedType::kDynamic);
            outputShapeSSAValues.push_back(dimValue);
          } else {
            // static dimensions
            staticOutputShape.push_back(
                inputType.getDimSize(currentInputDimIdx));
            // create arith.constant for shape operand
            outputShapeSSAValues.push_back(
                rewriter.create<mlir::arith::ConstantIndexOp>(
                    op->getLoc(), inputType.getDimSize(currentInputDimIdx)));
          }
          currentInputDimIdx++;
        }
      }

      // check that all input dimensions have been mapped.
      if (currentInputDimIdx != inputType.getRank()) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Unsqueeze input-output dimension mapping incomplete");
      }

      // 5. Create the tensor.from_elements op to get the shape as an SSA value.
      // The type of this shape tensor will be tensor<Rank x index>.
      mlir::RankedTensorType shapeTensorType = mlir::RankedTensorType::get(
          {static_cast<int64_t>(outputType.getRank())},
          rewriter.getIndexType());

      mlir::Value outputShapeValue =
          rewriter.create<mlir::tensor::FromElementsOp>(
              op->getLoc(), shapeTensorType, outputShapeSSAValues);

      // 6. Create the tensor.reshape operation.
      mlir::Value reshapedTensor = rewriter.create<mlir::tensor::ReshapeOp>(
          op->getLoc(), outputType, inputData, outputShapeValue);

      // 7. Replace the original operation.
      rewriter.replaceOp(op, reshapedTensor);

      return mlir::success();

    } else if (opNameBeginsWith(opName, "Transpose")) {
      // 1. Get input and output types
      mlir::Value input = op->getOperand(0);
      mlir::RankedTensorType inputType =
          mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
      if (!inputType) {
        return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");
      }
      int64_t rank = inputType.getRank();

      mlir::RankedTensorType outputType =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
      if (!outputType) {
        return rewriter.notifyMatchFailure(op,
                                           "output must be a ranked tensor");
      }

      // 2. Get the 'perm' attribute from the operation
      mlir::Attribute permAttr = op->getAttr("perm");
      llvm::SmallVector<int64_t> permutations;
      auto arrayPermAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(permAttr);
      if (arrayPermAttr) {
        for (mlir::Attribute attr : arrayPermAttr) {
          if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
            permutations.push_back(intAttr.getInt());
          } else {
            return rewriter.notifyMatchFailure(
                op, "perm array contains non-integer attributes");
          }
        }
        if ((int64_t)permutations.size() != rank) {
          return rewriter.notifyMatchFailure(
              op, "perm attribute size mismatch with input rank");
        }
      } else {
        // if 'perm' attribute is not present
        // default to reversing dimensions
        for (int64_t i = 0; i < rank; ++i) {
          permutations.push_back(rank - 1 - i);
        }
      }

      // 3. Create linalg.transpose's 'permutation' attribute
      mlir::DenseI64ArrayAttr permutationsAttr =
          rewriter.getDenseI64ArrayAttr(permutations);

      // --- DEBUG prints ---
      llvm::errs() << "Lowering onnx.TransposeOp to linalg.transposeOp:\n";
      llvm::errs() << "  Input Type: " << inputType << "\n";
      llvm::errs() << "  Output Type: " << outputType << "\n";
      llvm::errs() << "  Permutations: [";
      llvm::interleave(permutations, llvm::errs(), ", ");
      llvm::errs() << "]\n";
      // --- END DEBUG ---

      // 4. Create a tensor.empty operation for the 'init' operand.
      llvm::SmallVector<mlir::Value> dynamicDims;
      for (int64_t i = 0; i < outputType.getRank(); ++i) {
        if (outputType.isDynamicDim(i)) {
          mlir::Value dimValue = rewriter.create<mlir::tensor::DimOp>(
              op->getLoc(), input,
              rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(),
                                                            permutations[i]));
          dynamicDims.push_back(dimValue);
        }
      }

      mlir::Value emptyTensor = rewriter.create<mlir::tensor::EmptyOp>(
          op->getLoc(), outputType, dynamicDims);

      // 5. Create linalg.transpose op
      auto linalgTransposeOp =
          rewriter.create<mlir::linalg::TransposeOp>(
              op->getLoc(), input, emptyTensor, permutationsAttr);
      mlir::Value transposedTensor = linalgTransposeOp.getResult().front();

      // 6. Replace the original ONNX op with the new linalg.transpose op
      rewriter.replaceOp(op, transposedTensor);

      llvm::errs()
          << "Successfully lowered onnx.TransposeOp to linalg.transposeOp.\n";
      return mlir::success();
    }

    return mlir::success();
  }
};

// ONNX dialect to TOSA dialect pass
struct LowerONNXToLINALGPass
    : public mlir::PassWrapper<LowerONNXToLINALGPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerONNXToLINALGPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
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
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    // illegal dialects
    target.addIllegalDialect<onnx::OnnxDialect>();

    // illegal operations (must convert)
    // target.addIllegalOp<onnx::ConstantOp>();
    // target.addIllegalOp<onnx::ConvOp>();
    // target.addIllegalOp(mlir::OperationName("my_dialect.custom_op", ctx));

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

    // add Onnx ConvOp to LINALG ConvOp pattern
    //    patterns.add<ONNXConstantToTOSAConstPattern>(ctx);
    //    patterns.add<ONNXConvToTOSAConvPattern>(ctx);
    patterns.add<ONNXToLINALGLowering>(ctx);

    // apply the partial conversion pattern
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
      exit(-1);
    }
  }
};

std::unique_ptr<mlir::Pass> createLowerONNXToLINALGPass() {
  return std::make_unique<onnx2mlir::dialect::LowerONNXToLINALGPass>();
}

} // namespace onnx2mlir::dialect
