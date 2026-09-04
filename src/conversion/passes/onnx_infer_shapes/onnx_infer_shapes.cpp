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
 * \file src/conversion/onnx_infer_shapes.cpp
 * \brief Onnx operators shape inference pass
 */

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>

#include <memory>

#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/dialect/onnx/OnnxInterfaces.hpp"

namespace onnx2mlir::dialect {

struct InferONNXShapesPass
    : public ::mlir::impl::InferONNXShapesPassBase<InferONNXShapesPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferONNXShapesPass)

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    bool changed = true;

   /*
    * Inspect operators
    *
    */

    while (changed) {
      changed = false;

      module->walk([&](mlir::Operation *op) {
        bool needsShapeInference =
            llvm::any_of(op->getResultTypes(), [](mlir::Type t) {
              if (llvm::isa<mlir::UnrankedTensorType>(t))
                return true;
              if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(t))
                return !ranked.hasStaticShape();
              return false;
            });

        if (!needsShapeInference)
          return;

        auto shapeInfer = mlir::dyn_cast<onnx::ShapeInferenceOpInterface>(op);
        if (!shapeInfer)
          return;

        llvm::SmallVector<mlir::Type, 4> oldTypes(op->getResultTypes());

        shapeInfer.inferShapes();

        for (auto [oldType, newType] :
             llvm::zip(oldTypes, op->getResultTypes())) {
          if (oldType != newType) {
            changed = true;
            break;
          }
        }
      });
    }

    /*
     * Inspect result types
     *
     */

    module->walk([](mlir::func::FuncOp funcOp) {
      funcOp.walk([&](mlir::func::ReturnOp returnOp) {
        auto currentFuncType = funcOp.getFunctionType();
        llvm::SmallVector<mlir::Type, 4> returnTypes(
            returnOp.getOperandTypes());
        if (currentFuncType.getResults() !=
            llvm::ArrayRef<mlir::Type>(returnTypes)) {
          auto newFuncType = mlir::FunctionType::get(
              funcOp.getContext(), currentFuncType.getInputs(), returnTypes);
          funcOp.setFunctionType(newFuncType);
        }
      });
    });
  }
};

std::unique_ptr<mlir::Pass> createInferONNXShapesPass() {
  return std::make_unique<onnx2mlir::dialect::InferONNXShapesPass>();
}

void registerInferONNXShapesPass() {
  mlir::PassRegistration<onnx2mlir::dialect::InferONNXShapesPass>();
}

} // namespace onnx2mlir::dialect
