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
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

#include "onnx_to_linalg.hpp" // NOLINT

namespace onnx2mlir::dialect {

struct ONNXToLINALGLowering : public mlir::ConversionPattern {
  explicit ONNXToLINALGLowering(mlir::TypeConverter &typeConverter,
                                mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(typeConverter,
                                mlir::Pattern::MatchAnyOpTypeTag(),
                                /*PatternBenefit=*/true, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // triage by onnx operation names
    llvm::StringRef opName = op->getName().getStringRef();

    if (opNameBeginsWith(opName, {"Add", "Sub", "Mul", "Div", "Pow"})) {
      return OnnxToLinalg_ArithBinaryOps(op, rewriter);
    } else if (opNameBeginsWith( // clang-format off
        opName,
        {"Abs",      "Acos",       "Acosh",     "Asin",
         "Asinh",    "Atan",       "Atanh",     "Ceil",
         "Cos",      "Cosh",       "Elu",       "Erf",
         "Exp",      "Floor",      "HardSwish", "Identity",
         "IsInf",    "IsNaN",      "Log",       "Neg",
         "Not",      "Reciprocal", "Relu",      "Round",
         "Sign",     "Sigmoid",    "Sin",       "Sinh",
         "Softplus", "Softsign",   "Sqrt",      "Tan",
         "Tanh"
        })) { // clang-format on
      return OnnxToLinalg_ArithUnaryOps(op, rewriter);
    } else if (opNameBeginsWith(opName, "Cast")) {
      return OnnxToLinalg_CastOp(op, rewriter);
    } else if (opNameBeginsWith( // clang-format off
        opName, {
        "Equal", "Greater", "GreatherOrEqual", "Less", "LessOrEqual",
        })) { // clang-format on
      return OnnxToLinalg_CompBinaryOps(op, rewriter);
    } else if (opNameBeginsWith(opName, "Constant")) {
      return OnnxToLinalg_ConstantOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "Conv")) {
      return OnnxToLinalg_ConvOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "Gemm")) {
      return OnnxToLinalg_GemmOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "Hardmax")) {
      return OnnxToLinalg_HardmaxOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "LogSoftmax")) {
      return OnnxToLinalg_LogSoftmaxOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "MaxPool")) {
      return OnnxToLinalg_MaxPoolOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "Softmax")) {
      return OnnxToLinalg_SoftmaxOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "Squeeze")) {
      return OnnxToLinalg_SqueezeOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "Transpose")) {
      return OnnxToLinalg_TransposeOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "Unsqueeze")) {
      return OnnxToLinalg_UnsqueezeOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "Where")) {
      return OnnxToLinalg_WhereOp(op, rewriter);
    }

    return mlir::success();
  }
};

// ONNX dialect to LINALG dialect pass
struct LowerONNXToLINALGPass
    : public ::mlir::impl::LowerONNXToLINALGPassBase<LowerONNXToLINALGPass> {

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
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    // illegal dialects
    target.addIllegalDialect<onnx::OnnxDialect>();

    // illegal operations (must convert)
    // target.addIllegalOp<onnx::ConstantOp>();
    // target.addIllegalOp<onnx::AbsOp>();

    // legal operations
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    // allow onnx NoneType (postpone rewrite)
    target.addDynamicallyLegalDialect<onnx::OnnxDialect>(
        [](mlir::Operation *op) {
          if (opNameBeginsWith(op->getName().getStringRef(), "Constant")) {
            return mlir::isa<mlir::NoneType>(op->getResult(0).getType());
          }
          return false;
        });

    /*
     * Type conversions
     *
     */

    mlir::TypeConverter typeConverter;

    // Default type
    typeConverter.addConversion([](mlir::Type type) { return type; });

    // Values <- source
    typeConverter.addSourceMaterialization(
        [&](mlir::OpBuilder builder, mlir::Type resType,
            mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1) {
            return nullptr;
          }

          mlir::Value inp = inputs[0];
          mlir::Type inpType = inp.getType();
          auto inpSType = mlir::dyn_cast<mlir::ShapedType>(inpType);
          auto resSType = mlir::dyn_cast<mlir::ShapedType>(resType);

          // dynamic source and static target
          if (inpSType && resSType && !inpSType.hasStaticShape() &&
              resSType.hasStaticShape()) {
            // same rank & element type
            if ((inpSType.getRank() == resSType.getRank()) &&
                (inpSType.getElementType() == resSType.getElementType())) {
              return mlir::tensor::CastOp::create(builder, loc, resType, inp);
            }
          }

          return nullptr;
        });

    // Values -> target
    typeConverter.addTargetMaterialization(
        [&](mlir::OpBuilder builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> mlir::Value { return nullptr; });

    /*
     * Rewriter patterns
     *
     */

    // create a set of patterns.
    mlir::RewritePatternSet patterns(ctx);

    // add Onnx ConvOp to LINALG ConvOp pattern
    patterns.add<ONNXToLINALGLowering>(typeConverter, ctx);

    // apply the partial conversion pattern
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // clean up NoneType Constant ops if they are unused
    module.walk<mlir::WalkOrder::PostOrder>([](mlir::Operation *op) {
      if (opNameBeginsWith(op->getName().getStringRef(), "Constant") &&
          mlir::isa<mlir::NoneType>(op->getResult(0).getType())) {
        if (op->use_empty()) {
          op->erase();
        }
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createLowerONNXToLINALGPass() {
  return std::make_unique<onnx2mlir::dialect::LowerONNXToLINALGPass>();
}

void registerLowerONNXToLINALGPass() {
  mlir::PassRegistration<onnx2mlir::dialect::LowerONNXToLINALGPass>();
}

} // namespace onnx2mlir::dialect
