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
