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
