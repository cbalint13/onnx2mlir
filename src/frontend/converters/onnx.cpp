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
 * \file src/frontend/converters/onnx.cpp
 * \brief Onnx converter implementation
 */

#include <llvm/Support/SourceMgr.h>

#include <mlir/Pass/PassManager.h>

#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/frontend/onnx.hpp"

namespace onnx2mlir::frontend {

void ONNXConverter::convert(mlir::ModuleOp *module) {
  // context
  auto ctx = module->getContext();

  // diagnostics handler
  llvm::SourceMgr srcMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(srcMgr, ctx);

  // DEBUG
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  llvm::outs().enable_colors(true);
  module->print(llvm::outs(), flags);
  llvm::outs().enable_colors(false);

  // create pass manager
  mlir::PassManager pm(ctx);

  // add Onnx to Linalg
  pm.addPass(::onnx2mlir::dialect::createLowerONNXToLINALGPass());

  llvm::outs() << "\n";
  llvm::outs() << "Run passes: ONNX to LINALG\n";

  // run all passes
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "ERROR: pass pipeline failed.\n";
    exit(-1);
  }
}

} // end namespace onnx2mlir::frontend
