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
#include <mlir/Transforms/Passes.h>

#include <map>
#include <string>

#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/frontend/onnx.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::frontend {

/*
 *  ONNXConverter class
 */

ONNXConverter::ONNXConverter(const std::map<std::string, std::string> &options)
    : FrontendConverter(options) {}

void ONNXConverter::convert(mlir::ModuleOp *module) {
  bool verbose = false;
  bool canonicalize = false;
  bool infershapes = false;
  bool onnx2linalg = false;

  if (opt_args.count("--verbose") > 0)
    verbose = true;
  if (opt_args.count("--canonicalize") > 0)
    canonicalize = true;
  if (opt_args.count("--infershapes") > 0)
    infershapes = true;
  if (opt_args.count("--onnx2linalg") > 0)
    onnx2linalg = true;

  // context
  auto *ctx = module->getContext();

  // diagnostics handler
  llvm::SourceMgr srcMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(srcMgr, ctx);

  // create pass manager
  mlir::PassManager pm(ctx);

  if (verbose)
    llvm::outs() << "\n";

  if (canonicalize) {
    pm.addPass(mlir::createCanonicalizerPass());
    if (verbose)
      llvm::outs() << "Run pass: ONNX canonicalizer\n";
  }

  if (infershapes) {
    pm.addPass(::onnx2mlir::dialect::createInferONNXShapesPass());
    if (verbose)
      llvm::outs() << "Run pass: ONNX shape inference\n";
  }

  if (onnx2linalg) {
    pm.addPass(::onnx2mlir::dialect::createLowerONNXToLINALGPass());
    if (verbose)
      llvm::outs() << "Run pass: ONNX to Linalg lowering\n";
  }

  // run all passes
  if (mlir::failed(pm.run(*module))) {
    onnx2mlir::error() << "Pass pipeline failed.\n";
    exit(-1);
  }
}

} // end namespace onnx2mlir::frontend
