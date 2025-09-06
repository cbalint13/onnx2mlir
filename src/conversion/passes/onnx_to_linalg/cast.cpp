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
