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
