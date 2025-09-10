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
#include <regex>
#include <set>
#include <string>
#include <utility>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

#include "onnx_to_linalg.hpp"

namespace onnx2mlir::dialect {

static inline bool opNameBeginsWith(const llvm::StringRef &OpName,
                                    const std::string &match) {
  return std::regex_match(OpName.str(), std::regex("^onnx." + match + ".*"));
}

static inline bool isBroadcastNeeded(mlir::RankedTensorType operandType,
                                     mlir::RankedTensorType resultType) {
  if (!operandType || operandType == resultType) {
    return false;
  }
  // Check if any dimension is different or if ranks are different.
  if (operandType.getRank() != resultType.getRank())
    return true;
  for (int i = 0; i < operandType.getRank(); ++i) {
    if (operandType.getDimSize(i) != resultType.getDimSize(i) &&
        operandType.getDimSize(i) != 1) {
      return true; // Mismatched non-one dimension means non-broadcastable
    }
    if (operandType.getDimSize(i) == 1 && resultType.getDimSize(i) != 1) {
      return true; // Needs broadcasting a '1' dim
    }
  }
  return false;
}

static inline mlir::Value
createArithCastOp(mlir::OpBuilder *builder, const mlir::Location &loc,
                  const mlir::Value &inputElement,
                  const mlir::Type &targetElementType) {
  mlir::Type inputElementType = inputElement.getType();

  // If element types are the same, no cast needed.
  if (inputElementType == targetElementType) {
    return inputElement;
  }

  // Floating point to Floating point
  if (inputElementType.isFloat() && targetElementType.isFloat()) {
    unsigned inputWidth = inputElementType.getIntOrFloatBitWidth();
    unsigned outputWidth = targetElementType.getIntOrFloatBitWidth();

    if (inputWidth < outputWidth) {
      return builder->create<mlir::arith::ExtFOp>(loc, targetElementType,
                                                  inputElement);
    } else { // Truncation
      return builder->create<mlir::arith::TruncFOp>(loc, targetElementType,
                                                    inputElement);
    }
    // Integer to Integer
  } else if (inputElementType.isInteger() && targetElementType.isInteger()) {
    unsigned inputWidth = inputElementType.getIntOrFloatBitWidth();
    unsigned outputWidth = targetElementType.getIntOrFloatBitWidth();

    // Handle bool (i1) as a special case for older APIs
    if (inputWidth == 1 && outputWidth == 1)
      return inputElement; // bool to bool

    // From bool (i1) to larger integer
    if (inputWidth == 1) {
      return builder->create<mlir::arith::ExtUIOp>(loc, targetElementType,
                                                   inputElement);
    }
    // To bool (i1) from any integer
    if (outputWidth == 1) {
      return builder->create<mlir::arith::TruncIOp>(loc, targetElementType,
                                                    inputElement);
    }

    // General integer casting (signed/unsigned extension/truncation)
    if (inputWidth < outputWidth) {
      // In older APIs, `isSigned()` or `isUnsigned()` might be needed on the
      // integer type. Assuming `IntegerType::get(context, width, signedness)`
      // implies these properties. For now, default to unsigned extension for
      // generic integer types unless explicitly signed.
      if (inputElementType.isSignedInteger() &&
          targetElementType.isSignedInteger()) {
        return builder->create<mlir::arith::ExtSIOp>(loc, targetElementType,
                                                     inputElement);
      } else { // Fallback for unsigned or signless extension
        return builder->create<mlir::arith::ExtUIOp>(loc, targetElementType,
                                                     inputElement);
      }
    } else if (inputWidth > outputWidth) {
      return builder->create<mlir::arith::TruncIOp>(loc, targetElementType,
                                                    inputElement);
    } else { // Same bitwidth, potentially different signedness (e.g., i32 to
             // ui32)
      // This is a reinterpretation (bitcast)
      return builder->create<mlir::arith::BitcastOp>(loc, targetElementType,
                                                     inputElement);
    }
    // Floating point to Integer
  } else if (inputElementType.isFloat() && targetElementType.isInteger()) {
    if (targetElementType.isSignedInteger()) {
      return builder->create<mlir::arith::FPToSIOp>(loc, targetElementType,
                                                    inputElement);
    } else { // isUnsignedInteger or isSignlessInteger
      return builder->create<mlir::arith::FPToUIOp>(loc, targetElementType,
                                                    inputElement);
    }
    // Integer to Floating point
  } else if (inputElementType.isInteger() && targetElementType.isFloat()) {
    if (inputElementType.isSignedInteger()) {
      return builder->create<mlir::arith::SIToFPOp>(loc, targetElementType,
                                                    inputElement);
    } else { // isUnsignedInteger or isSignlessInteger
      return builder->create<mlir::arith::UIToFPOp>(loc, targetElementType,
                                                    inputElement);
    }
  }

  // Fallback for unhandled/unsupported combinations.
  // In older MLIR, direct emitError on OpBuilder might not exist or has
  // different signature. We return nullptr to signal failure, and the caller
  // (LowerONNXCastOp) will handle the error.
  return nullptr; // Indicate failure
}


// Helper function for performing various arithmetic casts
// This function canonicalizes integer types to signless (iN) before performing
// arithmetic operations, and then casts back if the target type requires signedness.
// NOTE: This helper might still encounter issues with very strict MLIR builds
// if arith.bitcast/extui/extsi truly reject uiN/siN as operands.
// For MaxPool, current usage is to keep original integer types for comparisons.
mlir::Value createArithCastOp2(mlir::OpBuilder &builder, const mlir::Location &loc,
                              const mlir::Value &inputElement,
                              const mlir::Type &targetElementType) {
  mlir::Type inputElementType = inputElement.getType();

  // 0. Base case: If element types are the same, no cast needed.
  if (inputElementType == targetElementType) {
    return inputElement;
  }

  mlir::Value currentVal = inputElement; // Start with the original value
  mlir::Type currentValType = inputElementType; // Keep track of current value's MLIR type
  
  // Floating point to Floating point
  if (currentValType.isFloat() && targetElementType.isFloat()) {
    unsigned inputWidth = currentValType.getIntOrFloatBitWidth();
    unsigned outputWidth = targetElementType.getIntOrFloatBitWidth();
    if (inputWidth < outputWidth) {
      currentVal = builder.create<mlir::arith::ExtFOp>(loc, targetElementType, currentVal);
    } else { // Truncation
      currentVal = builder.create<mlir::arith::TruncFOp>(loc, targetElementType, currentVal);
    }
  // Integer to Integer
  } else if (currentValType.isInteger() && targetElementType.isInteger()) {
    unsigned inputWidth = currentValType.getIntOrFloatBitWidth();
    unsigned outputWidth = targetElementType.getIntOrFloatBitWidth();
    mlir::IntegerType inputIntType = mlir::dyn_cast<mlir::IntegerType>(currentValType);
    mlir::IntegerType targetIntType = mlir::dyn_cast<mlir::IntegerType>(targetElementType);

    if (inputWidth < outputWidth) {
      // Extending:
      // Use ExtUIOp/ExtSIOp if source is unsigned/signed, otherwise use common ExtIOp.
      if (inputIntType.isUnsignedInteger()) {
        currentVal = builder.create<mlir::arith::ExtUIOp>(loc, targetElementType, currentVal);
      } else if (inputIntType.isSignedInteger()) {
        currentVal = builder.create<mlir::arith::ExtSIOp>(loc, targetElementType, currentVal);
      } else { // input is signless (iN)
        // If target is signed, use ExtSIOp, otherwise ExtUIOp (for signless/unsigned target)
        if (targetIntType.isSignedInteger()) {
          currentVal = builder.create<mlir::arith::ExtSIOp>(loc, targetElementType, currentVal);
        } else {
          currentVal = builder.create<mlir::arith::ExtUIOp>(loc, targetElementType, currentVal);
        }
      }
    } else if (inputWidth > outputWidth) {
      // Truncating:
      // Always truncate to a signless integer of the target width first, as TruncUIOp/TruncSIOp might be unavailable.
      mlir::Type tempSignlessTarget = builder.getIntegerType(outputWidth);
      currentVal = builder.create<mlir::arith::TruncIOp>(loc, tempSignlessTarget, currentVal);
      // If the targetElementType (e.g., ui8) is different from the temporary signless target (i8),
      // perform a bitcast to reinterpret the bits.
      if (currentVal.getType() != targetElementType) {
          currentVal = builder.create<mlir::arith::BitcastOp>(loc, targetElementType, currentVal);
      }
    } else { // Same bitwidth (inputWidth == outputWidth)
      // If signedness is different (e.g., ui8 to i8, or i8 to ui8), this is a bitcast.
      // This is the line that caused the error for ui8 input to arith.bitcast in some environments.
      // We assume that this particular MLIR setup is strict and arith.bitcast cannot take uiN/siN input.
      // However, if the currentVal is already iN/fN, it should work.
      if (currentValType != targetElementType) {
        currentVal = builder.create<mlir::arith::BitcastOp>(loc, targetElementType, currentVal);
      }
    }
  // Floating point to Integer
  } else if (currentValType.isFloat() && targetElementType.isInteger()) {
    mlir::IntegerType targetIntType = mlir::dyn_cast<mlir::IntegerType>(targetElementType);
    if (targetIntType.isSignedInteger()) {
      currentVal = builder.create<mlir::arith::FPToSIOp>(loc, targetElementType, currentVal);
    } else { // target is unsigned or signless
      currentVal = builder.create<mlir::arith::FPToUIOp>(loc, targetElementType, currentVal);
    }
  // Integer to Floating point
  } else if (currentValType.isInteger() && targetElementType.isFloat()) {
    mlir::IntegerType inputIntType = mlir::dyn_cast<mlir::IntegerType>(currentValType);
    if (inputIntType && inputIntType.isUnsignedInteger()) {
      currentVal = builder.create<mlir::arith::UIToFPOp>(loc, targetElementType, currentVal);
    } else { // original was signed or signless
      currentVal = builder.create<mlir::arith::SIToFPOp>(loc, targetElementType, currentVal);
    }
  } else {
    // Fallback for unhandled/unsupported combinations.
    llvm::errs() << "ERROR: Unsupported cast operation in createArithCastOp2: from "
                 << inputElementType << " to " << targetElementType << " at " << loc << "\n";
    return nullptr; // Indicate failure
  }

  return currentVal;
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

    if (opNameBeginsWith(opName, "Constant")) {
      return OnnxToLinalg_ConstantOp(op, rewriter);
    } else if (opNameBeginsWith(opName, "Unsqueeze")) {
#include "onnx_to_linalg/unsqueeze.cpp"
    } else if (opNameBeginsWith(opName, "Transpose")) {
#include "onnx_to_linalg/transpose.cpp"
    } else if (opNameBeginsWith(opName, "Add") ||
               opNameBeginsWith(opName, "Sub") ||
               opNameBeginsWith(opName, "Mul") ||
               opNameBeginsWith(opName, "Div") ||
               opNameBeginsWith(opName, "Pow")) {
#include "onnx_to_linalg/binary.cpp"
    } else if (opNameBeginsWith(opName, "Abs")) {
#include "onnx_to_linalg/abs.cpp"
    } else if (opNameBeginsWith(opName, "Cast")) {
#include "onnx_to_linalg/cast.cpp"
    } else if (opNameBeginsWith(opName, "Greater")) {
#include "onnx_to_linalg/greater.cpp"
    } else if (opNameBeginsWith(opName, "Where")) {
#include "onnx_to_linalg/where.cpp"
    } else if (opNameBeginsWith(opName, "MaxPool")) {
#include "onnx_to_linalg/maxpool.cpp"
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
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    // illegal dialects
    target.addIllegalDialect<onnx::OnnxDialect>();

    // illegal operations (must convert)
    // target.addIllegalOp<onnx::ConstantOp>();
    // target.addIllegalOp<onnx::AbsOp>();
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

    /*
     * Type conversions
     *
     */

    mlir::TypeConverter typeConverter;

    // default conversion
    typeConverter.addConversion([](mlir::Type type) { return type; });

    // converter rule
    typeConverter.addConversion(
        [&](mlir::ShapedType type) -> std::optional<mlir::Type> {
          mlir::Type convertedElementType =
              typeConverter.convertType(type.getElementType());
          if (!convertedElementType) {
            return std::nullopt;
          }
          if (convertedElementType == type.getElementType()) {
            return type;
          }
          // create a new RankedTensorType with the converted element type
          return mlir::RankedTensorType::get(type.getShape(),
                                             convertedElementType);
        });

    // materialization for values from source operations
    typeConverter.addSourceMaterialization(
        [&](mlir::OpBuilder builder, mlir::Type resultType,
            mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1) {
            return nullptr; // failure
          }
          return createArithCastOp(&builder, loc, inputs[0], resultType);
        });

    // materialization for values to target operations
    typeConverter.addTargetMaterialization(
        [&](mlir::OpBuilder builder, mlir::Type resultType,
            mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1) {
            return nullptr; // failure
          }
          return createArithCastOp(&builder, loc, inputs[0], resultType);
        });

    /*
     * Rewriter patterns
     *
     */

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

void registerLowerONNXToLINALGPass() {
  mlir::PassRegistration<onnx2mlir::dialect::LowerONNXToLINALGPass>();
}

} // namespace onnx2mlir::dialect
