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
      return rewriter.notifyMatchFailure(op, "output must be a ranked tensor");
    }

    // 2. Get the 'perm' attribute from the operation
    mlir::Attribute permAttr = op->getAttr("perm");

    llvm::SmallVector<int64_t> permutations;

    // FIX: Correctly cast to ArrayAttr and iterate its elements
    auto arrayPermAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(permAttr);

    if (arrayPermAttr) {
      for (mlir::Attribute attr : arrayPermAttr) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
          permutations.push_back(intAttr.getInt());
        } else {
          return rewriter.notifyMatchFailure(op, "perm array contains non-integer attributes");
        }
      }

      if ((int64_t)permutations.size() != rank) {
          return rewriter.notifyMatchFailure(op, "perm attribute size mismatch with input rank");
      }
    } else {
      // If 'perm' attribute is not present, default to reversing dimensions.
      for (int64_t i = 0; i < rank; ++i) {
        permutations.push_back(rank - 1 - i);
      }
    }

    // 3. Create the DenseI64ArrayAttr for linalg.transpose's 'permutation' attribute
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
                rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), permutations[i])
            );
            dynamicDims.push_back(dimValue);
        }
    }
    
    mlir::Value emptyTensor = rewriter.create<mlir::tensor::EmptyOp>(
        op->getLoc(), outputType, dynamicDims
    );

    // 5. Create linalg.transpose op
    mlir::linalg::TransposeOp linalgTransposeOp = rewriter.create<mlir::linalg::TransposeOp>(
        op->getLoc(),
        input,              // $input
        emptyTensor,        // $init (destination)
        permutationsAttr    // $permutation attribute
    );
    mlir::Value transposedTensor = linalgTransposeOp.getResult().front();

    // 6. Replace the original ONNX op with the new linalg.transpose op
    rewriter.replaceOp(op, transposedTensor);

    llvm::errs() << "Successfully lowered onnx.TransposeOp to linalg.transposeOp.\n";
    return mlir::success();
