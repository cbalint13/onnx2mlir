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
