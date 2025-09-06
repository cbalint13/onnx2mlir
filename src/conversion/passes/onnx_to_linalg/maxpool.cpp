        llvm::errs() << "DEBUG: LowerONNXMaxPoolOp pattern invoked for operation: " << op->getName() << "\n";
        llvm::errs() << "DEBUG: Full op dump: "; op->dump();

        auto maxPoolOp = mlir::dyn_cast<onnx::MaxPoolOp>(op);
        if (!maxPoolOp) {
            return rewriter.notifyMatchFailure(op, "failed to cast to onnx.MaxPoolOp");
        }

        mlir::Value inputX = maxPoolOp.getX();
        mlir::Location loc = op->getLoc();

        mlir::ShapedType inputShapedType = mlir::dyn_cast_or_null<mlir::ShapedType>(inputX.getType());
        if (!inputShapedType || !inputShapedType.hasRank() || inputShapedType.getRank() != 4) {
            return rewriter.notifyMatchFailure(op, "input to ONNX MaxPool must be a 4D shaped type");
        }
        // Assuming NCHW format
        int64_t h_in_dim = inputShapedType.getDimSize(2);
        int64_t w_in_dim = inputShapedType.getDimSize(3);

        // --- Attribute Extraction and Validation ---
        auto autoPadAttr = maxPoolOp.getAutoPadAttr();
        if (autoPadAttr && autoPadAttr.str() != "NOTSET") {
            return rewriter.notifyMatchFailure(op, "only auto_pad='NOTSET' is supported for ONNX MaxPool lowering");
        }

        auto ceilModeAttr = maxPoolOp.getCeilModeAttr();
        if (ceilModeAttr && ceilModeAttr.getInt() != 0) {
            return rewriter.notifyMatchFailure(op, "only ceil_mode=0 is supported for ONNX MaxPool lowering");
        }

        auto dilationsAttr = maxPoolOp.getDilationsAttr();
        if (dilationsAttr) {
            for (mlir::Attribute attr : dilationsAttr.getValue()) {
                if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                    if (intAttr.getInt() != 1) {
                        return rewriter.notifyMatchFailure(op, "dilations other than 1 are not supported for ONNX MaxPool lowering");
                    }
                } else {
                    return rewriter.notifyMatchFailure(op, "dilations attribute contains non-integer values");
                }
            }
        }

        auto storageOrderAttr = maxPoolOp.getStorageOrderAttr();
        if (storageOrderAttr && storageOrderAttr.getInt() != 0) { // 0 for NCHW
             return rewriter.notifyMatchFailure(op, "only storage_order=0 (NCHW) is supported for ONNX MaxPool lowering");
        }

        mlir::ArrayAttr kernelShapeAttr = maxPoolOp.getKernelShapeAttr();
        if (!kernelShapeAttr || kernelShapeAttr.empty()) {
            return rewriter.notifyMatchFailure(op, "onnx.MaxPool missing 'kernel_shape' attribute");
        }
        llvm::SmallVector<int64_t> kernelShape;
        for (mlir::Attribute attr : kernelShapeAttr.getValue()) {
            if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                kernelShape.push_back(intAttr.getInt());
            } else {
                return rewriter.notifyMatchFailure(op, "kernel_shape attribute contains non-integer values");
            }
        }
        if (kernelShape.size() != 2) { // Assuming 2D pooling (H, W)
             return rewriter.notifyMatchFailure(op, "only 2D kernel_shape supported (H, W) for ONNX MaxPool lowering");
        }

        llvm::SmallVector<int64_t> stridesVec;
        auto stridesAttr = maxPoolOp.getStridesAttr();
        if (stridesAttr) {
            for (mlir::Attribute attr : stridesAttr.getValue()) {
                if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                    stridesVec.push_back(intAttr.getInt());
                } else {
                    return rewriter.notifyMatchFailure(op, "strides attribute contains non-integer values");
                }
            }
        } else {
            stridesVec.push_back(1); // Default stride H
            stridesVec.push_back(1); // Default stride W
        }
        if (stridesVec.size() != 2) { // Assuming 2D pooling (H, W)
             return rewriter.notifyMatchFailure(op, "only 2D strides supported (H, W) for ONNX MaxPool lowering");
        }
        int64_t sh_val = stridesVec[0];
        int64_t sw_val = stridesVec[1];


        llvm::SmallVector<int64_t> padsVec;
        auto padsAttr = maxPoolOp.getPadsAttr();
        if (padsAttr) {
            for (mlir::Attribute attr : padsAttr.getValue()) {
                if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                    padsVec.push_back(intAttr.getInt());
                } else {
                    return rewriter.notifyMatchFailure(op, "pads attribute contains non-integer values");
                }
            }
        } else {
            padsVec.push_back(0); // Default pad_h_start
            padsVec.push_back(0); // Default pad_w_start
            padsVec.push_back(0); // Default pad_h_end
            padsVec.push_back(0); // Default pad_w_end
        }
        if (padsVec.size() != 4) { // Assuming 2D pooling (top, left, bottom, right)
             return rewriter.notifyMatchFailure(op, "only 4-element pads (H_start, W_start, H_end, W_end) supported for ONNX MaxPool lowering");
        }
        int64_t ph_start = padsVec[0];
        int64_t pw_start = padsVec[1];


        mlir::ShapedType outputShapedTypeY = mlir::cast<mlir::ShapedType>(op->getResult(0).getType());
        mlir::Type originalElemType = inputShapedType.getElementType();

        // Add explicit checks for value validity before linalg.GenericOp creation
        if (!inputX || mlir::isa<mlir::NoneType>(inputX.getType())) {
            llvm::errs() << "ERROR: inputX is invalid or NoneType before linalg.GenericOp: ";
            if (inputX) inputX.print(llvm::errs());
            llvm::errs() << "\n";
            return rewriter.notifyMatchFailure(op, "inputX is invalid for linalg.GenericOp");
        }
        
        // 1. Determine the element type for sentinel and initial values
        bool originalInputWasUnsigned = false;
        if (mlir::IntegerType intElemType = mlir::dyn_cast<mlir::IntegerType>(originalElemType)) {
            if (intElemType.isUnsignedInteger()) {
                originalInputWasUnsigned = true;
            }
        }

        // 2. Create the scalar sentinel value
        mlir::Value sentinel_val_scalar;
        if (originalElemType.isF32()) {
            mlir::Attribute negInfAttr = mlir::FloatAttr::get(originalElemType, -std::numeric_limits<float>::infinity());
            sentinel_val_scalar = rewriter.create<mlir::arith::ConstantOp>(loc, originalElemType, mlir::cast<mlir::TypedAttr>(negInfAttr));
        } else if (mlir::IntegerType intElemType = mlir::dyn_cast<mlir::IntegerType>(originalElemType)) {
            mlir::Attribute initialAttr;
            if (originalInputWasUnsigned) {
                initialAttr = rewriter.getIntegerAttr(originalElemType, 0); // For unsigned, 0 is typically the smallest
            } else {
                initialAttr = rewriter.getIntegerAttr(originalElemType, llvm::APInt::getSignedMinValue(intElemType.getWidth())); // For signed, min signed value
            }
            if (!mlir::isa<mlir::TypedAttr>(initialAttr)) {
                op->emitOpError("failed to create TypedAttr for integer sentinel value");
                return mlir::failure();
            }
            sentinel_val_scalar = rewriter.create<mlir::arith::ConstantOp>(loc, originalElemType, mlir::cast<mlir::TypedAttr>(initialAttr));
        } else {
            op->emitOpError("unsupported element type for MaxPool sentinel value (not float or integer)");
            return mlir::failure();
        }

        // 3. Create the output tensor (empty)
        mlir::Value outputTensorEmpty = rewriter.create<mlir::tensor::EmptyOp>(
            loc, outputShapedTypeY.getShape(), originalElemType);

        // 4. Fill the output tensor with the sentinel value (this becomes the initial accumulator)
        mlir::Value initialAccumulator = rewriter.create<mlir::linalg::FillOp>(
            loc, sentinel_val_scalar, outputTensorEmpty).getResult(0);

        if (!initialAccumulator || mlir::isa<mlir::NoneType>(initialAccumulator.getType())) {
            llvm::errs() << "ERROR: initialAccumulator is invalid or NoneType after linalg.FillOp: ";
            if (initialAccumulator) initialAccumulator.print(llvm::errs());
            llvm::errs() << "\n";
            return rewriter.notifyMatchFailure(op, "initialAccumulator is invalid for linalg.GenericOp");
        }


        // --- LINALG.GENERIC CREATION ---

        // 1. Define iteration space and iteration types
        llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes;
        iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // N - d0
        iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // C - d1
        iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // OH - d2
        iteratorTypes.push_back(mlir::utils::IteratorType::parallel); // OW - d3
        iteratorTypes.push_back(mlir::utils::IteratorType::reduction);// KH - d4
        iteratorTypes.push_back(mlir::utils::IteratorType::reduction);// KW - d5
        unsigned numDims = iteratorTypes.size(); // 6 dimensions

        // 2. Define indexing maps
        mlir::MLIRContext *context = rewriter.getContext();
        
        // Affine expressions for dimensions (d0..d5 for n, c, oh, ow, kh, kw)
        mlir::AffineExpr d0 = rewriter.getAffineDimExpr(0); // n
        mlir::AffineExpr d1 = rewriter.getAffineDimExpr(1); // c
        mlir::AffineExpr d2 = rewriter.getAffineDimExpr(2); // oh
        mlir::AffineExpr d3 = rewriter.getAffineDimExpr(3); // ow
        mlir::AffineExpr d4 = rewriter.getAffineDimExpr(4); // kh
        mlir::AffineExpr d5 = rewriter.getAffineDimExpr(5); // kw

        // Input map for inputX: (n, c, oh*sh + kh - ph_start, ow*sw + kw - pw_start)
        mlir::AffineExpr h_in_expr = (d2 * rewriter.getAffineConstantExpr(sh_val)) + d4 - rewriter.getAffineConstantExpr(ph_start);
        mlir::AffineExpr w_in_expr = (d3 * rewriter.getAffineConstantExpr(sw_val)) + d5 - rewriter.getAffineConstantExpr(pw_start);
        mlir::AffineMap inputMap = mlir::AffineMap::get(numDims, 0, {
            d0, d1, h_in_expr, w_in_expr
        }, context);

        // Output map: (n, c, oh, ow) - This map describes how the output tensor is indexed by the loops.
        // It's also effectively the accumulator map (as the output is the accumulator).
        mlir::AffineMap outputMap = mlir::AffineMap::get(numDims, 0, {
            d0, d1, d2, d3
        }, context);


        // When an operand is both an input and an output, it's typically treated as *one* input
        // and *one* output for the purposes of indexing maps.
        // If initialAccumulator is ONLY in outputs, then blockArgs should be 2.
        // Inputs: {inputX} (1 input)
        // Outputs: {initialAccumulator} (1 output)
        // Total maps = 1 + 1 = 2
        llvm::SmallVector<mlir::AffineMap> indexingMaps = {inputMap, outputMap};


        // DEBUG PRINTS FOR LINALG.GENERIC CONSTRUCTION PARAMETERS
        llvm::errs() << "DEBUG: ----- Parameters for linalg.GenericOp construction -----\n";
        llvm::errs() << "DEBUG: resultTensorTypes: ";
        for (mlir::Type t : mlir::TypeRange{outputShapedTypeY}) t.print(llvm::errs());
        llvm::errs() << "\n";

        llvm::errs() << "DEBUG: inputs (inputX): "; // Only inputX now
        inputX.print(llvm::errs());
        llvm::errs() << "\n";

        llvm::errs() << "DEBUG: outputs (initialAccumulator): ";
        initialAccumulator.print(llvm::errs());
        llvm::errs() << "\n";

        llvm::errs() << "DEBUG: iteratorTypes size: " << iteratorTypes.size() << "\n";
        for (unsigned i = 0; i < iteratorTypes.size(); ++i) {
            llvm::errs() << "DEBUG: iteratorTypes[" << i << "]: ";
            if (iteratorTypes[i] == mlir::utils::IteratorType::parallel) llvm::errs() << "parallel\n";
            else if (iteratorTypes[i] == mlir::utils::IteratorType::reduction) llvm::errs() << "reduction\n";
            else llvm::errs() << "UNKNOWN_ITERATOR_TYPE\n"; // Should not happen
        }

        llvm::errs() << "DEBUG: indexingMaps size: " << indexingMaps.size() << "\n";
        for (unsigned i = 0; i < indexingMaps.size(); ++i) {
            llvm::errs() << "DEBUG: indexingMaps[" << i << "]: " << indexingMaps[i] << "\n";
        }
        llvm::errs() << "DEBUG: -----------------------------------------------------\n";


        // 3. Create the linalg.generic operation, with the lambda defined inline
        mlir::linalg::GenericOp genericOp = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            /*resultTensorTypes=*/mlir::TypeRange{outputShapedTypeY}, // The final output tensor type
            /*inputs=*/mlir::ValueRange{inputX}, // Only inputX as input
            /*outputs=*/mlir::ValueRange{initialAccumulator}, // The accumulator being updated (also used for initial value)
            indexingMaps,
            iteratorTypes,
            // Lambda for the linalg.generic body
            [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::ValueRange blockArgs) {
                // If inputs is {inputX} and outputs is {initialAccumulator},
                // blockArgs[0]: input_element (from inputX)
                // blockArgs[1]: output_accumulator (from initialAccumulator, current max value)
                llvm::errs() << "DEBUG: blockArgs size inside linalg.generic lambda: " << blockArgs.size() << "\n";

                mlir::Value inputElemVal = blockArgs[0]; // element from the input tensor (X)
                mlir::Value outputAccumVal = blockArgs[1]; // current max value for the output cell (accumulator)

                // Access parallel and reduction indices using linalg.index ops
                mlir::Value nArg = nestedBuilder.create<mlir::linalg::IndexOp>(nestedLoc, 0); // N is at position 0 in iteratorTypes
                mlir::Value cArg = nestedBuilder.create<mlir::linalg::IndexOp>(nestedLoc, 1); // C is at position 1
                mlir::Value ohArg = nestedBuilder.create<mlir::linalg::IndexOp>(nestedLoc, 2); // OH is at position 2
                mlir::Value owArg = nestedBuilder.create<mlir::linalg::IndexOp>(nestedLoc, 3); // OW is at position 3
                mlir::Value khArg = nestedBuilder.create<mlir::linalg::IndexOp>(nestedLoc, 4); // KH is at position 4 (a reduction dim)
                mlir::Value kwArg = nestedBuilder.create<mlir::linalg::IndexOp>(nestedLoc, 5); // KW is at position 5 (a reduction dim)
                
                mlir::Type indexType = nestedBuilder.getIndexType();
                
                // Determine if original element type was unsigned, for CmpIOp predicate
                bool originalInputWasUnsigned = false;
                if (mlir::IntegerType intElemType = mlir::dyn_cast<mlir::IntegerType>(inputElemVal.getType())) {
                    if (intElemType.isUnsignedInteger()) {
                        originalInputWasUnsigned = true;
                    }
                }

                // Constants and values needed for calculations within the region
                mlir::Value sh_val_idx = nestedBuilder.create<mlir::arith::ConstantOp>(nestedLoc, indexType, nestedBuilder.getIndexAttr(sh_val));
                mlir::Value sw_val_idx = nestedBuilder.create<mlir::arith::ConstantOp>(nestedLoc, indexType, nestedBuilder.getIndexAttr(sw_val));
                mlir::Value ph_start_idx = nestedBuilder.create<mlir::arith::ConstantOp>(nestedLoc, indexType, nestedBuilder.getIndexAttr(ph_start));
                mlir::Value pw_start_idx = nestedBuilder.create<mlir::arith::ConstantOp>(nestedLoc, indexType, nestedBuilder.getIndexAttr(pw_start));
                mlir::Value h_in_dim_idx = nestedBuilder.create<mlir::arith::ConstantOp>(nestedLoc, indexType, nestedBuilder.getIndexAttr(h_in_dim));
                mlir::Value w_in_dim_idx = nestedBuilder.create<mlir::arith::ConstantOp>(nestedLoc, indexType, nestedBuilder.getIndexAttr(w_in_dim));
                mlir::Value zero_const_idx = nestedBuilder.create<mlir::arith::ConstantOp>(nestedLoc, indexType, nestedBuilder.getIndexAttr(0));

                // Calculate h_in and w_in (input coordinates)
                mlir::Value h_in_pos = nestedBuilder.create<mlir::arith::SubIOp>(nestedLoc, nestedBuilder.create<mlir::arith::AddIOp>(nestedLoc, nestedBuilder.create<mlir::arith::MulIOp>(nestedLoc, ohArg, sh_val_idx), khArg), ph_start_idx);
                mlir::Value w_in_pos = nestedBuilder.create<mlir::arith::SubIOp>(nestedLoc, nestedBuilder.create<mlir::arith::AddIOp>(nestedLoc, nestedBuilder.create<mlir::arith::MulIOp>(nestedLoc, owArg, sw_val_idx), kwArg), pw_start_idx);

                // Bounds checking: h_in_pos >= 0 && h_in_pos < h_in_dim
                mlir::Value h_in_ge_zero = nestedBuilder.create<mlir::arith::CmpIOp>(nestedLoc, mlir::arith::CmpIPredicate::sge, h_in_pos, zero_const_idx);
                mlir::Value h_in_lt_h_in_dim = nestedBuilder.create<mlir::arith::CmpIOp>(nestedLoc, mlir::arith::CmpIPredicate::slt, h_in_pos, h_in_dim_idx);
                mlir::Value h_in_valid = nestedBuilder.create<mlir::arith::AndIOp>(nestedLoc, h_in_ge_zero, h_in_lt_h_in_dim);

                // Bounds checking: w_in_pos >= 0 && w_in_pos < w_in_dim
                mlir::Value w_in_ge_zero = nestedBuilder.create<mlir::arith::CmpIOp>(nestedLoc, mlir::arith::CmpIPredicate::sge, w_in_pos, zero_const_idx);
                mlir::Value w_in_lt_w_in_dim = nestedBuilder.create<mlir::arith::CmpIOp>(nestedLoc, mlir::arith::CmpIPredicate::slt, w_in_pos, w_in_dim_idx);
                mlir::Value w_in_valid = nestedBuilder.create<mlir::arith::AndIOp>(nestedLoc, h_in_valid, w_in_lt_w_in_dim);
                
                // Combine validity checks
                mlir::Value pixel_valid = nestedBuilder.create<mlir::arith::AndIOp>(nestedLoc, h_in_valid, w_in_valid);

                // Perform the max reduction:
                // If pixel_valid is true, result is max(outputAccumVal, inputElemVal).
                // If pixel_valid is false, result is outputAccumVal (don't update).
                mlir::Value value_if_valid;
                if (inputElemVal.getType().isF32()) {
                    mlir::Value isGreater = nestedBuilder.create<mlir::arith::CmpFOp>(
                        nestedLoc, mlir::arith::CmpFPredicate::OGT, outputAccumVal, inputElemVal);
                    value_if_valid = nestedBuilder.create<mlir::arith::SelectOp>(
                        nestedLoc, isGreater, outputAccumVal, inputElemVal);
                } else if (mlir::IntegerType::classof(inputElemVal.getType())) {
                    mlir::arith::CmpIPredicate predicate = originalInputWasUnsigned ?
                                                           mlir::arith::CmpIPredicate::ugt :
                                                           mlir::arith::CmpIPredicate::sgt;
                    mlir::Value isGreater = nestedBuilder.create<mlir::arith::CmpIOp>(
                        nestedLoc, predicate, outputAccumVal, inputElemVal);
                    value_if_valid = nestedBuilder.create<mlir::arith::SelectOp>(
                        nestedLoc, isGreater, outputAccumVal, inputElemVal);
                } else {
                    op->emitOpError("unsupported element type for max operation in linalg.generic body");
                    return;
                }

                // Final selection: if pixel is valid, use the calculated max; otherwise, keep the current accumulator value.
                mlir::Value final_output_value = nestedBuilder.create<mlir::arith::SelectOp>(
                    nestedLoc, pixel_valid, value_if_valid, outputAccumVal);

                // Yield the result.
                nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, final_output_value);
            } // End of lambda
        );

        // Restore insertion point after the generic op
        rewriter.setInsertionPointAfter(genericOp); 
        mlir::Value linalgMaxPoolResultY = genericOp.getResult(0);

        mlir::Value finalResultY = linalgMaxPoolResultY;

        // **CRITICAL CODE FOR REPLACEMENT:**
        // 1. Replace all uses of the first result (Y) with the finalResultY
        op->getResult(0).replaceAllUsesWith(finalResultY);

        // 2. Handle the second result (Indices) which is `NoneType`.
        mlir::Type indicesResultType = op->getResult(1).getType();
        if (mlir::isa<mlir::NoneType>(indicesResultType)) {
            if (!op->getResult(1).use_empty()) {
                llvm::errs() << "ERROR: onnx.MaxPool Indices result (NoneType) is unexpectedly used! Cannot lower without generating indices.\n";
                op->getResult(1).dump(); // Dump its uses for debugging
                return rewriter.notifyMatchFailure(op, "onnx.MaxPool Indices result is used, but lowering does not generate it.");
            }
        } else if (!op->getResult(1).use_empty()) {
             return rewriter.notifyMatchFailure(op, "onnx.MaxPool Indices result is used and not NoneType, but lowering does not generate it.");
        }

        // 3. Erase the original onnx.MaxPool op.
        rewriter.eraseOp(op);
        
        llvm::errs() << "DEBUG: LowerONNXMaxPoolOp: Called replaceAllUsesWith and eraseOp. Now returning success.\n";

        llvm::errs() << "Successfully lowered onnx.MaxPoolOp to linalg.genericOp.\n";
        return mlir::success();