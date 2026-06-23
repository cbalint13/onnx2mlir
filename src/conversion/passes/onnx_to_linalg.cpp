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
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

#include "onnx_to_linalg.hpp" // NOLINT

namespace onnx2mlir::dialect {

using LoweringFunc = std::function<mlir::LogicalResult(
    mlir::Operation *, mlir::PatternRewriter &, const mlir::TypeConverter *)>;

template <typename Func>
static void registerOps(const std::vector<std::string> &opNames,
                        std::unordered_map<std::string, LoweringFunc> &map,
                        Func func) {
  for (const auto &opname : opNames) {
    map[opname] = [func](mlir::Operation *op, mlir::PatternRewriter &rewriter,
                         const mlir::TypeConverter *typconv) {
      if constexpr (std::is_invocable_v<Func, mlir::Operation *,
                                        mlir::PatternRewriter &,
                                        const mlir::TypeConverter *>) {
        return func(op, rewriter, typconv);
      } else {
        return func(op, rewriter);
      }
    };
  }
}

static const std::unordered_map<std::string, LoweringFunc> &getLoweringMap() {
  static const auto loweringMap = [] {
    std::unordered_map<std::string, LoweringFunc> map;
    registerOps( // clang-format off
                {"Add",      "Sub",        "Mul",       "Div",
                 "Pow"}, map,
        OnnxToLinalg_ArithBinaryOps); // clang-format on
    registerOps( // clang-format off
                {"Abs",      "Acos",       "Acosh",     "Asin",
                 "Asinh",    "Atan",       "Atanh",     "Ceil",
                 "Cos",      "Cosh",       "Elu",       "Erf",
                 "Exp",      "Floor",      "HardSwish", "Identity",
                 "IsInf",    "IsNaN",      "Log",       "Neg",
                 "Not",      "Reciprocal", "Relu",      "Round",
                 "Sign",     "Sigmoid",    "Sin",       "Sinh",
                 "Softplus", "Softsign",   "Sqrt",      "Tan",
                 "Tanh"}, map,
        OnnxToLinalg_ArithUnaryOps);  // clang-format on
    registerOps({"Cast"}, map, OnnxToLinalg_CastOp);
    registerOps( // clang-format off
                {"Equal",    "Greater",    "GreatherOrEqual",
                 "Less",     "LessOrEqual"}, map,
        OnnxToLinalg_CompBinaryOps); // clang-format on
    registerOps({"Constant"}, map, OnnxToLinalg_ConstantOp);
    registerOps({"Conv"}, map, OnnxToLinalg_ConvOp);
    registerOps({"Flatten"}, map, OnnxToLinalg_FlattenOp);
    registerOps({"Gemm"}, map, OnnxToLinalg_GemmOp);
    registerOps({"Hardmax"}, map, OnnxToLinalg_HardmaxOp);
    registerOps({"LogSoftmax"}, map, OnnxToLinalg_LogSoftmaxOp);
    registerOps({"MaxPool"}, map, OnnxToLinalg_MaxPoolOp);
    registerOps({"Softmax"}, map, OnnxToLinalg_SoftmaxOp);
    registerOps({"Squeeze"}, map, OnnxToLinalg_SqueezeOp);
    registerOps({"Transpose"}, map, OnnxToLinalg_TransposeOp);
    registerOps({"Unsqueeze"}, map, OnnxToLinalg_UnsqueezeOp);
    registerOps({"Where"}, map, OnnxToLinalg_WhereOp);
    return map;
  }();

  return loweringMap;
}

struct ONNXToLINALGLowering : public mlir::ConversionPattern {
  explicit ONNXToLINALGLowering(mlir::TypeConverter &typeConverter,
                                mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(typeConverter,
                                mlir::Pattern::MatchAnyOpTypeTag(),
                                /*PatternBenefit=*/true, ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const llvm::StringRef opName = op->getName().getStringRef();
    // lookup op register catalog
    const auto &map = getLoweringMap();
    for (const auto &[name, func] : map) {
      if (opNameBeginsWith(opName, name)) {
        // lower the named op
        return func(op, rewriter, typeConverter);
      }
    }

    return mlir::failure();
  }
};

// ONNX dialect to LINALG dialect pass
struct LowerONNXToLINALGPass
    : public ::mlir::impl::LowerONNXToLINALGPassBase<LowerONNXToLINALGPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerONNXToLINALGPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<onnx::OnnxDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();
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

    // legal operations
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();

    // allow onnx.Constant -> NoneType (postpone rewrite)
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

    // default mappings (Type -> Type)
    typeConverter.addConversion([](mlir::Type type) -> mlir::Type {
      if (!type)
        return nullptr;
      // scalar integer type convert (ui/si -> signless)
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        if (!intType.isSignless()) {
          return mlir::IntegerType::get(type.getContext(), intType.getWidth());
        }
      }
      // shaped integer type convert (ui/si -> signless)
      if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type)) {
        mlir::Type elementType = shapedType.getElementType();
        mlir::Type signlessElt = getSignlessType(elementType);
        if (signlessElt != elementType) {
          return shapedType.clone(signlessElt);
        }
      }
      // default
      return type;
    });

    // mark type conversion as unrealized (New -> Old)
    typeConverter.addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resType,
            mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          return mlir::UnrealizedConversionCastOp::create(builder, loc, resType,
                                                          inputs)
              .getResult(0);
        });

    // mark type conversion as unrealized (Old -> New)
    typeConverter.addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resType,
            mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          return mlir::UnrealizedConversionCastOp::create(builder, loc, resType,
                                                          inputs)
              .getResult(0);
        });

    /*
     * Rewriter patterns
     *
     */

    // create a set of patterns.
    mlir::RewritePatternSet patterns(ctx);

    // add type converter
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

std::vector<std::string> registerLowerONNXToLINALGPass() {
  mlir::PassRegistration<onnx2mlir::dialect::LowerONNXToLINALGPass>();
  std::vector<std::string> supportedOps;
  const auto &map = getLoweringMap();
  for (const auto &[name, func] : map)
    supportedOps.push_back(name);
  return supportedOps;
}

} // namespace onnx2mlir::dialect
