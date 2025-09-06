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
