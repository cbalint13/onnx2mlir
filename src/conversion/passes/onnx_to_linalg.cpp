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

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/dialect/onnx/Onnx.hpp"

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

      mlir::Location fusedLoc = rewriter.getFusedLoc(
          {op->getLoc()}, rewriter.getStringAttr(op->getName().getStringRef()));

      auto constValue = rewriter.create<mlir::arith::ConstantOp>(
          fusedLoc, elemValueAttr.getType(), elemValueAttr);
      rewriter.replaceOp(op, constValue);

      return mlir::success();

      /*
       * onnx::Unsqueeze
       */
    } else if (opNameBeginsWith(opName, "Unsqueeze")) {
      // 1. Access the 'data' input operand (operand 0).
      mlir::Value inputData = op->getOperand(0);
      auto inputType =
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

      auto outputType =
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

      mlir::Location fusedLoc = rewriter.getFusedLoc(
          {op->getLoc()}, rewriter.getStringAttr(op->getName().getStringRef()));

      for (int64_t i = 0; i < outputType.getRank(); ++i) {
        if (std::binary_search(axes.begin(), axes.end(), i)) {
          // an unsqueezed dimension (new dim of size 1)
          staticOutputShape.push_back(1);
          // create an arith.constant for '1'
          outputShapeSSAValues.push_back(
              rewriter.create<mlir::arith::ConstantIndexOp>(fusedLoc, 1));
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
                fusedLoc, inputData,
                rewriter.create<mlir::arith::ConstantIndexOp>(
                    fusedLoc, currentInputDimIdx));
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
                    fusedLoc, inputType.getDimSize(currentInputDimIdx)));
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
              fusedLoc, shapeTensorType, outputShapeSSAValues);

      // 6. Create the tensor.reshape operation.
      mlir::Value reshapedTensor = rewriter.create<mlir::tensor::ReshapeOp>(
          fusedLoc, outputType, inputData, outputShapeValue);

      // 7. Replace the original operation.
      rewriter.replaceOp(op, reshapedTensor);

      return mlir::success();

    } else if (opNameBeginsWith(opName, "Transpose")) {
      // 1. Get input and output types
      mlir::Value input = op->getOperand(0);
      auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
      if (!inputType) {
        return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");
      }
      int64_t rank = inputType.getRank();

      auto outputType =
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
        if (static_cast<int64_t>(permutations.size()) != rank) {
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

      mlir::Location fusedLoc = rewriter.getFusedLoc(
          {op->getLoc()}, rewriter.getStringAttr(op->getName().getStringRef()));

      // 4. Create a tensor.empty operation for the 'init' operand.
      llvm::SmallVector<mlir::Value> dynamicDims;
      for (int64_t i = 0; i < outputType.getRank(); ++i) {
        if (outputType.isDynamicDim(i)) {
          mlir::Value dimValue = rewriter.create<mlir::tensor::DimOp>(
              fusedLoc, input,
              rewriter.create<mlir::arith::ConstantIndexOp>(fusedLoc,
                                                            permutations[i]));
          dynamicDims.push_back(dimValue);
        }
      }

      mlir::Value emptyTensor = rewriter.create<mlir::tensor::EmptyOp>(
          fusedLoc, outputType, dynamicDims);

      // 5. Create linalg.transpose op
      auto linalgTransposeOp = rewriter.create<mlir::linalg::TransposeOp>(
          fusedLoc, input, emptyTensor, permutationsAttr);
      mlir::Value transposedTensor = linalgTransposeOp.getResult().front();

      // 6. Replace the original ONNX op with the new linalg.transpose op
      rewriter.replaceOp(op, transposedTensor);

      llvm::errs()
          << "Successfully lowered onnx.TransposeOp to linalg.transposeOp.\n";
      return mlir::success();

    } else if (opNameBeginsWith(opName, "Add") ||
               opNameBeginsWith(opName, "Sub")) {
      mlir::Value lhs = op->getOperand(0);
      mlir::Value rhs = op->getOperand(1);

      auto resultType =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());

      if (!resultType) {
        return rewriter.notifyMatchFailure(
            op, "onnx.Sub result must be a ranked tensor type");
      }

      mlir::Location fusedLoc = rewriter.getFusedLoc(
          {op->getLoc()}, rewriter.getStringAttr(op->getName().getStringRef()));

      // Helper lambda to apply appropriate broadcasting
      auto applyBroadcasting = [&](mlir::Value value) -> mlir::Value {
        mlir::RankedTensorType valueType =
            mlir::dyn_cast<mlir::RankedTensorType>(value.getType());

        llvm::errs() << "Applying broadcasting for valueType: " << valueType
                     << ", resultType: " << resultType << "\n";

        if (!isBroadcastNeeded(valueType, resultType)) {
          llvm::errs()
              << "  No broadcasting needed. Returning original value.\n";
          return value;
        }

        int valueRank = valueType.getRank();
        int resultRank = resultType.getRank();
        llvm::errs() << "  valueRank: " << valueRank
                     << ", resultRank: " << resultRank << "\n";

        // 1. Prepare the output buffer for the broadcasted value
        llvm::SmallVector<mlir::Value> broadcastOutputDynamicDims;
        for (int64_t i = 0; i < resultType.getRank(); ++i) {
          if (resultType.isDynamicDim(i)) {
            broadcastOutputDynamicDims.push_back(
                rewriter.create<mlir::tensor::DimOp>(
                    fusedLoc, value,
                    rewriter.create<mlir::arith::ConstantIndexOp>(fusedLoc,
                                                                  i)));
          }
        }
        mlir::Value initBuffer = rewriter.create<mlir::tensor::EmptyOp>(
            fusedLoc, resultType, broadcastOutputDynamicDims);

        // 2. Decide between linalg.broadcast and linalg.generic
        if (valueRank < resultRank) {
          llvm::errs() << "  Entering linalg.broadcast candidate path "
                          "(valueRank < resultRank).\n";

          // Linalg.broadcast's 'dimensions' attribute typically refers to the
          // mapping of input dimensions to output dimensions. For right-aligned
          // rank extension, input_dim_i maps to output_dim_(resultRank -
          // valueRank + i).
          llvm::SmallVector<int64_t> broadcastMappingDims;
          for (int i = 0; i < valueRank; ++i) {
            broadcastMappingDims.push_back(resultRank - valueRank + i);
          }

          bool compatibleForLinalgBroadcast = true;

          int numNewLeadingDims = resultRank - valueRank;
          if (numNewLeadingDims > 0) {
            // If there are new leading dimensions being introduced
            // Condition 1: Catches cases like tensor<3xf32> broadcast to
            // tensor<2x3xf32> where the output's leading dim (2) is not 1 AND
            // the input's first dim (3) is not 1 AND they don't match.
            if (resultType.getDimSize(0) != 1 &&
                valueType.getDimSize(0) != mlir::ShapedType::kDynamic &&
                valueType.getDimSize(0) != 1 &&
                valueType.getDimSize(0) != resultType.getDimSize(0)) {
              compatibleForLinalgBroadcast = false;
              llvm::errs()
                  << "    Leading dim incompatibility (non-1, non-matching "
                     "input/output dim 0) detected for linalg.broadcast.\n";
            }

            // Condition 2: Catches cases like tensor<1xf32> broadcast to
            // tensor<2x3xf32> where the input's first dim is 1, but the
            // output's first dim is not 1. linalg.broadcast struggles with
            // replicating a 1-sized leading dimension to a non-1 output leading
            // dimension.
            if (compatibleForLinalgBroadcast &&  // Only check if not already
                                                 // marked incompatible
                valueType.getDimSize(0) == 1 &&  // Input's first dim is 1
                resultType.getDimSize(0) != 1) { // Output's first dim is not 1
              compatibleForLinalgBroadcast = false;
              llvm::errs()
                  << "    Leading dim incompatibility (input dim 0 is 1, "
                     "output dim 0 is non-1) detected for linalg.broadcast.\n";
            }
          }

          // Ensure all *mapped* trailing dimensions align correctly or are 1 in
          // input for linalg.broadcast This loop checks the dimensions that
          // *are* explicitly mapped by broadcastMappingDims.
          if (compatibleForLinalgBroadcast) {
            // Only proceed if the leading dim checks passed
            for (int i = 0; i < valueRank; ++i) {
              if (valueType.getDimSize(i) !=
                      resultType.getDimSize(broadcastMappingDims[i]) &&
                  valueType.getDimSize(i) != 1) {
                compatibleForLinalgBroadcast = false;
                llvm::errs() << "    Mapped trailing dim incompatibility "
                                "detected for linalg.broadcast.\n";
                break;
              }
            }
          }

          if (compatibleForLinalgBroadcast) {
            llvm::errs() << "    Proceeding with linalg.broadcast.\n";
            mlir::linalg::BroadcastOp broadcastOp =
                rewriter.create<mlir::linalg::BroadcastOp>(
                    fusedLoc,
                    value,                 // The input tensor to broadcast
                    initBuffer,            // The output buffer ('init' value)
                    broadcastMappingDims); // 'dimensions' attribute (mapping)
            return broadcastOp.getOperation()->getResult(0);
          } else {
            llvm::errs()
                << "    Skipping linalg.broadcast due to incompatibility. "
                   "Falling back to linalg.generic.\n";
          }
        }

        // Fallback to linalg.generic for in-rank broadcasting or complex
        // rank-extension that linalg.broadcast cannot handle.

        llvm::errs() << "  Proceeding with linalg.generic.\n";

        llvm::SmallVector<mlir::AffineMap> indexingMaps;
        mlir::MLIRContext *ctx = rewriter.getContext();

        // Build the affine map for the input tensor.
        // The domain of this map has `resultRank` dimensions (d0, d1, ...).
        // The range of this map must have `valueRank` expressions, as that's
        // the rank of the input tensor.

        llvm::SmallVector<mlir::AffineExpr> valueMapExprs(valueRank);

        // Special case: Vector to Matrix broadcasting, e.g., [M] -> [M, N]
        // In ONNX, this typically means `input[i]` maps to `output[i, j]` for
        // all `j`. The affine map needed is `(d0, d1, ...) -> (d0, ...)` where
        // dimensions of output corresponding to 1s in the input (after padding
        // with 1s) are mapped to constant 0. For [M] to [M, N], it implies [M,
        // 1] to [M, N]. input_dim_0 (size M) maps to output_dim_0 (d0).
        // input_dim_1 (size 1) maps to constant 0 for output_dim_1 (d1).
        // But `valueMapExprs` only has `valueRank` elements.
        // So for `tensor<2xf32>` to `tensor<2x3xf32>`, the map is
        // `affine_map<(d0, d1) -> (d0)>`.
        if (valueRank == 1 && resultRank > 0 &&
            valueType.getDimSize(0) == resultType.getDimSize(0)) {
          // Check if the first dimension of the input matches the first
          // dimension of the result. This covers [M] -> [M, N] or [M] -> [M, N,
          // K] etc.
          valueMapExprs[0] = rewriter.getAffineDimExpr(
              0); // Map the input's first dim to the output's first dim
          llvm::errs() << "    Special case: Vector to higher rank tensor "
                          "(e.g., [M] to [M,N]). Mapped to d0.\n";
        } else {
          // General ONNX broadcasting (right-aligned, or scalar broadcast to
          // any rank)
          int inputDimCursor =
              valueRank - 1; // Current input dimension index (from right)

          // Iterate through the output dimensions (iteration space) from right
          // to left.
          for (int outIterDim = resultRank - 1; outIterDim >= 0; --outIterDim) {
            if (inputDimCursor >= 0) {
              int64_t inputSize = valueType.getDimSize(inputDimCursor);
              int64_t outputSize = resultType.getDimSize(outIterDim);

              if (inputSize == outputSize) {
                // Perfect match: input dimension maps directly to this output
                // iteration dimension.
                valueMapExprs[inputDimCursor] =
                    rewriter.getAffineDimExpr(outIterDim);
                inputDimCursor--; // Consume this input dimension
                llvm::errs() << "    General: Direct match for inputDim "
                             << inputDimCursor + 1 << " to outIterDim "
                             << outIterDim << "\n";
              } else if (inputSize == 1) {
                // Input dimension is 1, so it broadcasts by taking element 0.
                // It maps to a constant 0.
                valueMapExprs[inputDimCursor] =
                    rewriter.getAffineConstantExpr(0);
                inputDimCursor--; // Consume this input dimension
                llvm::errs()
                    << "    General: Input dim 1 (broadcast) for inputDim "
                    << inputDimCursor + 1 << " to outIterDim " << outIterDim
                    << "\n";
              } else if (outputSize == 1) {
                // Output dimension is 1. Input dimension maps to this output
                // iteration dimension. This implies squeezing.
                valueMapExprs[inputDimCursor] =
                    rewriter.getAffineDimExpr(outIterDim);
                inputDimCursor--; // Consume this input dimension
                llvm::errs()
                    << "    General: Output dim 1 (squeeze) for inputDim "
                    << inputDimCursor + 1 << " to outIterDim " << outIterDim
                    << "\n";
              } else {
                // Mismatched non-one dimensions. This is an invalid broadcast.
                // This should have been caught by `isBroadcastNeeded`.
                // If we reach here, map to 0 as a robust fallback, but
                // indicates likely incorrectness.
                valueMapExprs[inputDimCursor] =
                    rewriter.getAffineConstantExpr(0);
                inputDimCursor--;
                llvm::errs()
                    << "    General: Mismatched non-one dims for inputDim "
                    << inputDimCursor + 1 << " to outIterDim " << outIterDim
                    << ". Falling back to const 0.\n";
              }
            } else {
              // No more input dimensions to match from the right.
              // The remaining `outIterDim` on the left are new leading
              // dimensions of the output. These don't directly correspond to
              // any input dimensions in the affine map range.
              break; // All input dimensions have been considered.
            }
          }

          // After iterating through all output dimensions, any remaining
          // unmapped input dimensions (i.e., `inputDimCursor >= 0`) are leading
          // input dimensions that were not matched. These should generally map
          // to a constant 0 (e.g., scalar input to higher rank, or if valueRank
          // > resultRank).
          while (inputDimCursor >= 0) {
            valueMapExprs[inputDimCursor] = rewriter.getAffineConstantExpr(0);
            inputDimCursor--;
            llvm::errs() << "    General: Unmapped leading input dim "
                         << inputDimCursor + 1 << ". Mapped to const 0.\n";
          }
        }

        // Sanity check: Ensure all `valueMapExprs` elements are set.
        // This loop covers cases where `valueRank > resultRank` or other
        // unhandled scenarios.
        for (int i = 0; i < valueRank; ++i) {
          if (!valueMapExprs[i]) {
            valueMapExprs[i] = rewriter.getAffineConstantExpr(0); // Fallback
            llvm::errs() << "    Sanity check: Unset input dim " << i
                         << ". Mapped to const 0.\n";
          }
        }

        indexingMaps.push_back(
            mlir::AffineMap::get(resultRank, 0, valueMapExprs, ctx));

        // Output map (identity map for the result tensor)
        indexingMaps.push_back(
            mlir::AffineMap::getMultiDimIdentityMap(resultRank, ctx));

        // Iterator types (all parallel for element-wise ops)
        llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
            resultRank, mlir::utils::IteratorType::parallel);

        // Use InsertionGuard for automatic restoration of insertion point
        mlir::OpBuilder::InsertionGuard guard(rewriter);

        // Create the linalg.generic op using the build method that takes a
        // region-building lambda
        mlir::linalg::GenericOp genericOp =
            rewriter.create<mlir::linalg::GenericOp>(
                fusedLoc,
                /*resultTypes=*/resultType, // TypeRange for a single result is
                                            // just the type
                /*inputs=*/mlir::ValueRange{value},
                /*outputs=*/mlir::ValueRange{initBuffer},
                /*indexingMaps=*/indexingMaps,   // Pass SmallVector directly
                /*iteratorTypes=*/iteratorTypes, // Pass SmallVector directly
                // Region builder lambda for the linalg.generic body
                [&](mlir::OpBuilder &b, mlir::Location loc,
                    mlir::ValueRange regionArgs) {
                  // regionArgs[0] is the input element from 'value'
                  // regionArgs[1] is the output element from 'initBuffer' (for
                  // in-place operations) For a broadcast, we just yield the
                  // first input element, which corresponds to the value after
                  // its indexing map has been applied.
                  b.create<mlir::linalg::YieldOp>(loc, regionArgs[0]);
                });

        return genericOp.getOperation()->getResult(0);
      };

      // Apply broadcasting to LHS and RHS if their shapes are not already the
      // result shape.
      lhs = applyBroadcasting(lhs);
      rhs = applyBroadcasting(rhs);

      // Collect dynamic dimension values for the final output buffer, using the
      // final result shape.
      llvm::SmallVector<mlir::Value> dynamicDims;
      for (int64_t i = 0; i < resultType.getRank(); ++i) {
        if (resultType.isDynamicDim(i)) {
          mlir::Value dimValue = rewriter.create<mlir::tensor::DimOp>(
              fusedLoc,
              lhs, // Use lhs (now potentially broadcasted to the result shape)
              rewriter.create<mlir::arith::ConstantIndexOp>(fusedLoc, i));
          dynamicDims.push_back(dimValue);
        }
      }

      // create the empty final output buffer with the final, broadcasted shape
      mlir::Value buff = rewriter.create<mlir::tensor::EmptyOp>(
          fusedLoc, resultType, dynamicDims);

      // create the linalg Op with the (potentially broadcasted) inputs
      mlir::Value finalResult;
      if (opNameBeginsWith(opName, "Sub"))
        finalResult = rewriter
                          .create<mlir::linalg::SubOp>(
                              fusedLoc, mlir::ValueRange({lhs, rhs}),
                              mlir::ValueRange({buff}))
                          .getResult(0);
      else
        finalResult = rewriter
                          .create<mlir::linalg::AddOp>(
                              fusedLoc, mlir::ValueRange({lhs, rhs}),
                              mlir::ValueRange({buff}))
                          .getResult(0);

      // replace onnx.{Add,Sub}Op -> linalg.{add,sub}Op
      rewriter.replaceOp(op, finalResult);

      return mlir::success();

    } else if (opNameBeginsWith(opName, "Abs")) {
      llvm::errs() << "DEBUG: LowerONNXAbsOp pattern invoked for operation: "
                   << op->getName() << "\n";
      llvm::errs() << "DEBUG: Full op dump: " << *op << "\n";

      // Check if resultType or inputType extraction fails (though less likely)
      mlir::Value inputX = op->getOperand(0);
      auto resultType =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
      auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(inputX.getType());

      if (!resultType || !inputType) {
        llvm::errs() << "DEBUG: LowerONNXAbsOp: Input or result type not "
                        "ranked tensor. Returning failure.\n";
        return rewriter.notifyMatchFailure(
            op, "input/output must be ranked tensor types");
      }

      mlir::Location fusedLoc = rewriter.getFusedLoc(
          {op->getLoc()}, rewriter.getStringAttr(op->getName().getStringRef()));

      llvm::SmallVector<mlir::Value> dynamicDims;
      for (int64_t i = 0; i < resultType.getRank(); ++i) {
        if (resultType.isDynamicDim(i)) {
          mlir::Value dimValue = rewriter.create<mlir::tensor::DimOp>(
              fusedLoc, inputX,
              rewriter.create<mlir::arith::ConstantIndexOp>(fusedLoc, i));
          dynamicDims.push_back(dimValue);
        }
      }
      mlir::Value outputBuffer = rewriter.create<mlir::tensor::EmptyOp>(
          fusedLoc, resultType, dynamicDims);

      llvm::errs() << "DEBUG: LowerONNXAbsOp: Created tensor.empty: "
                   << outputBuffer << "\n";
      //   %empty1 = tensor.empty() : tensor<1x1800x256xf32>
      //   %exp1 = linalg.abs ins(%0 : tensor<1x1800x256xf32>) outs(%empty1 :
      //   tensor<1x1800x256xf32>) -> tensor<1x1800x256xf32>

      auto linalgAbsOp = rewriter.create<mlir::linalg::AbsOp>(
          fusedLoc, mlir::ValueRange({inputX}), mlir::ValueRange({inputX}));

      llvm::errs() << "DEBUG: LowerONNXAbsOp: Created linalg.absOp: "
                   << linalgAbsOp << "\n";

      // BEFORE REPLACEMENT: Verify uses
      llvm::errs()
          << "DEBUG: LowerONNXAbsOp: Before replaceOp. Original op has "
          << std::distance(op->result_begin(), op->result_end())
          << " results.\n";
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        if (!op->getResult(i).use_empty()) {
          llvm::errs() << "DEBUG: LowerONNXAbsOp: Result " << i
                       << " of original op has "
                       << std::distance(op->getResult(i).use_begin(),
                                        op->getResult(i).use_end())
                       << " uses.\n";
        } else {
          llvm::errs() << "DEBUG: LowerONNXAbsOp: Result " << i
                       << " of original op has no uses.\n";
        }
      }

      rewriter.replaceOp(op, linalgAbsOp.getOperation()->getResult(0));
      // rewriter.eraseOp(op);

      //  llvm::errs() << "DEBUG: LowerONNXAbsOp: Called replaceOp and eraseOp.
      //  Now returning success.\n";

      llvm::errs() << "Successfully lowered onnx.AbsOp to linalg.absOp.\n";

      return mlir::success();

    } else if (opNameBeginsWith(opName, "Cast")) {
      llvm::errs() << "DEBUG: LowerONNXCastOp pattern invoked for operation: "
                   << op->getName() << "\n";
      llvm::errs() << "DEBUG: Full op dump: ";
      op->dump();

      mlir::Value inputTensor = op->getOperand(0);

      auto toAttr = op->getAttrOfType<mlir::IntegerAttr>("to");
      if (!toAttr) {
        return rewriter.notifyMatchFailure(op, "missing 'to' attribute");
      }
      // Re-introduced: toValue
      int64_t toValue = toAttr.getInt();

      mlir::Location loc = op->getLoc();
      // Re-introduced: context
      mlir::MLIRContext *context = rewriter.getContext();

      // Reverted: targetElementType now derived from 'to' attribute
      mlir::Type targetElementType = OnnxToMlir_dType(toValue, context);
      // Re-introduced check for unsupported 'to' attribute value
      if (!targetElementType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported 'to' attribute value for ONNX Cast");
      }

      // Ensure input is a ShapedType before attempting to get its shape
      if (!mlir::isa<mlir::ShapedType>(inputTensor.getType())) {
        return rewriter.notifyMatchFailure(
            op, "input tensor to ONNX Cast is not a shaped type");
      }
      mlir::ShapedType inputShapedType =
          mlir::cast<mlir::ShapedType>(inputTensor.getType());

      // Reverted: resultTensorType now derived from input shape and
      // targetElementType
      mlir::Type resultTensorType = mlir::RankedTensorType::get(
          inputShapedType.getShape(), targetElementType);

      // Updated debug messages
      llvm::errs() << "DEBUG: LowerONNXCastOp: Derived targetElementType (from "
                      "'to' attribute): "
                   << targetElementType << "\n";
      llvm::errs() << "DEBUG: LowerONNXCastOp: Derived resultTensorType (from "
                      "input shape + target element type): "
                   << resultTensorType << "\n";

      // 1. Create an empty tensor for the output (using inputShapedType for
      // shape)
      mlir::Value outputBuffer = rewriter.create<mlir::tensor::EmptyOp>(
          loc, inputShapedType.getShape(), targetElementType);
      llvm::errs() << "DEBUG: LowerONNXCastOp: Created tensor.empty: ";
      outputBuffer.getDefiningOp()->dump();

      mlir::Value linalgCastOpResult;
      bool bodyBuildFailed = false;

      // Get the rank once
      int rank = mlir::cast<mlir::ShapedType>(inputTensor.getType()).getRank();

      llvm::SmallVector<mlir::utils::IteratorType> iterators;
      for (int i = 0; i < rank; ++i) {
        iterators.push_back(mlir::utils::IteratorType::parallel);
      }

      mlir::SmallVector<mlir::AffineMap> indexingMaps;
      indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
      indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

      mlir::linalg::GenericOp genericOp =
          rewriter.create<mlir::linalg::GenericOp>(
              loc, resultTensorType, mlir::ValueRange{inputTensor},
              mlir::ValueRange{outputBuffer}, indexingMaps, iterators,
              [&](mlir::OpBuilder nestedBuilder, mlir::Location nestedLoc,
                  mlir::ValueRange args) {
                mlir::Value inputElement = args[0];

                mlir::Value castResult = createArithCastOp(
                    &nestedBuilder, nestedLoc, inputElement, targetElementType);
                if (!castResult) {
                  bodyBuildFailed = true;
                  return;
                }
                nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc,
                                                            castResult);
              });

      if (bodyBuildFailed) {
        if (genericOp) {
          genericOp.erase();
        }
        return rewriter.notifyMatchFailure(
            op, "unsupported element type conversion for ONNX Cast within "
                "linalg.generic body");
      }

      linalgCastOpResult = genericOp.getResult(0);

      llvm::errs() << "DEBUG: LowerONNXCastOp: Created linalg.genericOp: ";
      genericOp.dump();

      rewriter.replaceOp(op, linalgCastOpResult);
      llvm::errs() << "DEBUG: LowerONNXCastOp: Called replaceOp. Now returning "
                      "success.\n";

      llvm::errs() << "Successfully lowered onnx.CastOp to linalg.genericOp.\n";

      return mlir::success();

    } else if (opNameBeginsWith(opName, "Greater")) {
      llvm::errs()
          << "DEBUG: LowerONNXGreaterOp pattern invoked for operation: "
          << op->getName() << "\n";
      llvm::errs() << "DEBUG: Full op dump: ";
      op->dump();

      mlir::Value lhs = op->getOperand(0);
      mlir::Value rhs = op->getOperand(1);
      mlir::Location loc = op->getLoc();
      mlir::MLIRContext *context = rewriter.getContext();

      // 1. Get input shaped types and their element types
      if (!mlir::isa<mlir::ShapedType>(lhs.getType()) ||
          !mlir::isa<mlir::ShapedType>(rhs.getType())) {
        return rewriter.notifyMatchFailure(
            op, "inputs to ONNX Greater must be shaped types");
      }
      mlir::ShapedType lhsShapedType =
          mlir::cast<mlir::ShapedType>(lhs.getType());
      mlir::ShapedType rhsShapedType =
          mlir::cast<mlir::ShapedType>(rhs.getType());

      mlir::Type lhsElementType = lhsShapedType.getElementType();
      mlir::Type rhsElementType = rhsShapedType.getElementType();

      int64_t lhsRank = lhsShapedType.getRank();
      int64_t rhsRank = rhsShapedType.getRank();

      // Determine the output rank for the linalg.generic op
      // This will be the maximum rank of the input tensors
      int64_t outputRank = std::max(lhsRank, rhsRank);

      // Determine the result shape. For scalars, the non-scalar input defines
      // the shape. For compatible shapes, the larger dimensions define the
      // shape. Since ONNX has already performed shape inference,
      // op->getResult(0).getType() is reliable.
      mlir::ShapedType originalResultShapedType =
          mlir::cast<mlir::ShapedType>(op->getResult(0).getType());
      mlir::ArrayRef<int64_t> resultShape = originalResultShapedType.getShape();

      // Output element type for comparison ops is i1 (boolean)
      mlir::Type outputElementType = mlir::IntegerType::get(context, 1);
      mlir::Type resultTensorType =
          mlir::RankedTensorType::get(resultShape, outputElementType);

      // 2. Create an empty tensor for the output
      mlir::Value outputBuffer = rewriter.create<mlir::tensor::EmptyOp>(
          loc, resultShape, outputElementType);
      llvm::errs() << "DEBUG: LowerONNXGreaterOp: Created tensor.empty: ";
      outputBuffer.getDefiningOp()->dump();

      mlir::Value linalgGreaterOpResult;

      llvm::SmallVector<mlir::utils::IteratorType> iterators;
      for (int i = 0; i < outputRank; ++i) { // Iterate based on outputRank
        iterators.push_back(mlir::utils::IteratorType::parallel);
      }

      mlir::SmallVector<mlir::AffineMap> indexingMaps;
      // Map for LHS
      if (lhsShapedType.getRank() == 0) { // If LHS is a scalar
        // indexingMaps.push_back(rewriter.getMultiDimEmptyAffineMap(outputRank));
        indexingMaps.push_back(mlir::AffineMap::get(
            outputRank, /*numSymbols=*/0, {}, rewriter.getContext()));
      } else { // If LHS is a tensor, use identity map of outputRank
        indexingMaps.push_back(rewriter.getMultiDimIdentityMap(
            lhsRank)); // Use actual rank of LHS for its map if it's not scalar
      }

      // Map for RHS
      if (rhsShapedType.getRank() == 0) { // If RHS is a scalar
        // indexingMaps.push_back(rewriter.getMultiDimEmptyAffineMap(outputRank));
        indexingMaps.push_back(mlir::AffineMap::get(
            outputRank, /*numSymbols=*/0, {}, rewriter.getContext()));
      } else { // If RHS is a tensor, use identity map of outputRank
        indexingMaps.push_back(rewriter.getMultiDimIdentityMap(
            rhsRank)); // Use actual rank of RHS for its map if it's not scalar
      }

      // Map for Output
      indexingMaps.push_back(rewriter.getMultiDimIdentityMap(outputRank));

      mlir::linalg::GenericOp genericOp =
          rewriter.create<mlir::linalg::GenericOp>(
              loc, resultTensorType,
              mlir::ValueRange{lhs, rhs}, // Two input operands
              mlir::ValueRange{
                  outputBuffer}, // One output operand (init tensor)
              indexingMaps, iterators,
              [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
                  mlir::ValueRange args) {
                mlir::Value lhsElement = args[0];
                mlir::Value rhsElement = args[1];
                mlir::Value cmpResult;

                if (lhsElementType.isSignedInteger() &&
                    rhsElementType.isSignedInteger()) {
                  cmpResult = nestedBuilder.create<mlir::arith::CmpIOp>(
                      nestedLoc, mlir::arith::CmpIPredicate::sgt, lhsElement,
                      rhsElement);
                } else if (lhsElementType.isUnsignedInteger() &&
                           rhsElementType.isUnsignedInteger()) {
                  cmpResult = nestedBuilder.create<mlir::arith::CmpIOp>(
                      nestedLoc, mlir::arith::CmpIPredicate::ugt, lhsElement,
                      rhsElement);
                } else if (lhsElementType.isFloat() &&
                           rhsElementType.isFloat()) {
                  cmpResult = nestedBuilder.create<mlir::arith::CmpFOp>(
                      nestedLoc, mlir::arith::CmpFPredicate::OGT, lhsElement,
                      rhsElement);
                } else {
                  // This case should ideally be handled by ONNX type validation
                  // or pre-processing if mixed types are not allowed by ONNX
                  // Greater or require implicit casts.
                  op->emitOpError(
                      "unsupported element type combination for ONNX Greater: ")
                      << lhsElementType << " vs " << rhsElementType;
                  return; // Abort body building if types are unsupported
                }
                nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc,
                                                            cmpResult);
              });

      linalgGreaterOpResult = genericOp.getResult(0);

      llvm::errs() << "DEBUG: LowerONNXGreaterOp: Created linalg.genericOp: ";
      genericOp.dump();

      rewriter.replaceOp(op, linalgGreaterOpResult);
      llvm::errs() << "DEBUG: LowerONNXGreaterOp: Called replaceOp. Now "
                      "returning success.\n";

      llvm::errs()
          << "Successfully lowered onnx.GreaterOp to linalg.genericOp.\n";
      return mlir::success();

    } else if (opNameBeginsWith(opName, "Where")) {
      llvm::errs() << "DEBUG: LowerONNXWhereOp pattern invoked for operation: "
                   << op->getName() << "\n";
      llvm::errs() << "DEBUG: Full op dump: ";
      op->dump();

      mlir::Value condition = op->getOperand(0);
      mlir::Value x = op->getOperand(1);
      mlir::Value y = op->getOperand(2);
      mlir::Location loc = op->getLoc();
      // mlir::MLIRContext *context = rewriter.getContext(); // REMOVED: Unused
      // variable

      if (!mlir::isa<mlir::ShapedType>(condition.getType()) ||
          !mlir::isa<mlir::ShapedType>(x.getType()) ||
          !mlir::isa<mlir::ShapedType>(y.getType())) {
        return rewriter.notifyMatchFailure(
            op, "inputs to ONNX Where must be shaped types");
      }

      mlir::ShapedType conditionShapedType =
          mlir::cast<mlir::ShapedType>(condition.getType());
      mlir::ShapedType xShapedType = mlir::cast<mlir::ShapedType>(x.getType());
      mlir::ShapedType yShapedType = mlir::cast<mlir::ShapedType>(y.getType());

      // ONNX Where expects condition to be i1 (boolean)
      if (!conditionShapedType.getElementType().isInteger(1)) {
        return rewriter.notifyMatchFailure(
            op, "condition input to ONNX Where must have i1 element type");
      }

      // X and Y must have the same element type for arith.select
      if (xShapedType.getElementType() != yShapedType.getElementType()) {
        return rewriter.notifyMatchFailure(
            op, "X and Y inputs to ONNX Where must have the same element type");
      }

      // The output element type is the same as X (and Y)
      mlir::Type outputElementType = xShapedType.getElementType();

      // The result type of the linalg.generic should match the ONNX op's result
      // type
      mlir::ShapedType originalResultShapedType =
          mlir::cast<mlir::ShapedType>(op->getResult(0).getType());
      mlir::ArrayRef<int64_t> resultShape = originalResultShapedType.getShape();
      int64_t outputRank = originalResultShapedType.getRank();

      mlir::Type resultTensorType =
          mlir::RankedTensorType::get(resultShape, outputElementType);

      // Create an empty tensor for the output, initialized with the determined
      // result shape and element type
      mlir::Value outputBuffer = rewriter.create<mlir::tensor::EmptyOp>(
          loc, resultShape, outputElementType);
      llvm::errs() << "DEBUG: LowerONNXWhereOp: Created tensor.empty: ";
      outputBuffer.getDefiningOp()->dump();

      // Prepare iterators for the linalg.generic op (all parallel for
      // element-wise operation)
      llvm::SmallVector<mlir::utils::IteratorType> iterators;
      for (int i = 0; i < outputRank; ++i) {
        iterators.push_back(mlir::utils::IteratorType::parallel);
      }

      // Prepare indexing maps for inputs and output to handle broadcasting
      mlir::SmallVector<mlir::AffineMap> indexingMaps;
      // Map for Condition (scalar or identity based on rank)
      if (conditionShapedType.getRank() == 0) {
        indexingMaps.push_back(mlir::AffineMap::get(
            outputRank, /*numSymbols=*/0, {}, rewriter.getContext()));
      } else {
        indexingMaps.push_back(rewriter.getMultiDimIdentityMap(
            conditionShapedType.getRank())); // Use actual rank of condition
      }

      // Map for X (scalar or identity based on rank)
      if (xShapedType.getRank() == 0) {
        indexingMaps.push_back(mlir::AffineMap::get(
            outputRank, /*numSymbols=*/0, {}, rewriter.getContext()));
      } else {
        indexingMaps.push_back(rewriter.getMultiDimIdentityMap(
            xShapedType.getRank())); // Use actual rank of X
      }

      // Map for Y (scalar or identity based on rank)
      if (yShapedType.getRank() == 0) {
        // indexingMaps.push_back(rewriter.getEmptyAffineMap());
        // indexingMaps.push_back(rewriter.getMultiDimEmptyAffineMap(outputRank));
        indexingMaps.push_back(mlir::AffineMap::get(
            outputRank, /*numSymbols=*/0, {}, rewriter.getContext()));
      } else {
        // indexingMaps.push_back(rewriter.getMultiDimIdentityMap(outputRank));
        indexingMaps.push_back(rewriter.getMultiDimIdentityMap(
            yShapedType.getRank())); // Use actual rank of Y
      }

      // Map for Output (always identity)
      indexingMaps.push_back(rewriter.getMultiDimIdentityMap(outputRank));

      // Create the linalg.generic operation
      mlir::linalg::GenericOp genericOp =
          rewriter.create<mlir::linalg::GenericOp>(
              loc, resultTensorType,
              mlir::ValueRange{condition, x, y}, // Three input operands
              mlir::ValueRange{
                  outputBuffer}, // One output operand (init tensor)
              indexingMaps, iterators,
              [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
                  mlir::ValueRange args) {
                mlir::Value condElement = args[0];
                mlir::Value xElement = args[1];
                mlir::Value yElement = args[2];

                // Use arith.select: result = cond ? x : y
                mlir::Value selectResult =
                    nestedBuilder.create<mlir::arith::SelectOp>(
                        nestedLoc, condElement, xElement, yElement);

                nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc,
                                                            selectResult);
              });

      mlir::Value linalgWhereOpResult = genericOp.getResult(0);

      llvm::errs() << "DEBUG: LowerONNXWhereOp: Created linalg.genericOp: ";
      genericOp.dump();

      // Replace the original onnx.Where op with the new linalg.generic op
      rewriter.replaceOp(op, linalgWhereOpResult);
      llvm::errs() << "DEBUG: LowerONNXWhereOp: Called replaceOp. Now "
                      "returning success.\n";

      llvm::errs()
          << "Successfully lowered onnx.WhereOp to linalg.genericOp.\n";

      return mlir::success();
    } else if (opNameBeginsWith(opName, "MaxPool")) {
#include "maxpool.cpp"
    }

    return mlir::success();
  }
};

// ONNX dialect to LINALG dialect pass
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

} // namespace onnx2mlir::dialect
