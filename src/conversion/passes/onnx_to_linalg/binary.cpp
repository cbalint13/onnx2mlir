
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
      if (opNameBeginsWith(opName, "Add"))
        finalResult = rewriter
                          .create<mlir::linalg::AddOp>(
                              fusedLoc, mlir::ValueRange({lhs, rhs}),
                              mlir::ValueRange({buff}))
                          .getResult(0);
      else if (opNameBeginsWith(opName, "Sub"))
        finalResult = rewriter
                          .create<mlir::linalg::SubOp>(
                              fusedLoc, mlir::ValueRange({lhs, rhs}),
                              mlir::ValueRange({buff}))
                          .getResult(0);
      else if (opNameBeginsWith(opName, "Mul"))
        finalResult = rewriter
                          .create<mlir::linalg::MulOp>(
                              fusedLoc, mlir::ValueRange({lhs, rhs}),
                              mlir::ValueRange({buff}))
                          .getResult(0);
      else if (opNameBeginsWith(opName, "Div"))
        finalResult = rewriter
                          .create<mlir::linalg::DivOp>(
                              fusedLoc, mlir::ValueRange({lhs, rhs}),
                              mlir::ValueRange({buff}))
                          .getResult(0);
      else if (opNameBeginsWith(opName, "Pow"))
        finalResult = rewriter
                          .create<mlir::linalg::PowFOp>(
                              fusedLoc, mlir::ValueRange({lhs, rhs}),
                              mlir::ValueRange({buff}))
                          .getResult(0);
      else {
        llvm::errs() << "ERROR: Unknown binary op [" << opName << "]\n";
        exit(-1);
      }


      // replace onnx.{Add,Sub}Op -> linalg.{add,sub}Op
      rewriter.replaceOp(op, finalResult);

      return mlir::success();
