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
 * \file src/onnx2mlir.cpp
 * \brief Onnx to Mlir compiler tool
 */

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/MLIRContext.h>

#include <iostream>
#include <map>
#include <regex>
#include <string>

#include "onnx2mlir/frontend/onnx.hpp"

template <typename ModuleType>
static bool saveModuleToFile(const ModuleType &module,
                             const std::string &filename) {
  std::error_code ec;
  llvm::raw_fd_ostream outputStream(filename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "ERROR: Failed to open file '" << filename
                 << "' for writing: " << ec.message() << "\n";
    return false;
  }
  // export to file
  mlir::OpPrintingFlags flags;
  flags.printLargeElementsAttrWithHex();
  module->print(outputStream, flags);
  outputStream.flush();

  return true;
}

template <typename ModuleType>
static void printModule(const ModuleType &module, bool elide = true) {
  mlir::OpPrintingFlags flags;
  if (elide)
    flags.elideLargeElementsAttrs(16);
  // flags.printLargeElementsAttrWithHex();
  // flags.enableDebugInfo();
  llvm::outs() << "\n";
  llvm::outs().enable_colors(true);
  module->print(llvm::outs(), flags);
  llvm::outs().enable_colors(false);
}

static void printUsage() {
  llvm::outs() << "\n";
  llvm::outs()
      << "Usage: onnx2mlir <input_onnx_file>\n"
      << "            [--export-mlir <filename>]\n"
      << "            [--canonicalize]\n"
      << "            [--infershapes]\n"
      << "            [--onnx2linalg]\n"
      << "            [--onnx-convert-ops <int : (optional | default is "
      << "max supported)>]\n"
      << "            [--no-elide]\n"
      << "            [--verbose]\n"
      << "            [--help]\n"
      << "\n";
}

int main(int argc, char **argv) {
  // command-line params
  bool elide = true;
  bool verbose = false;
  std::string ONNXFilename = "";
  std::string exportMLIRFilename = "";
  std::map<std::string, std::string> options;

  // command-line parser
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      const auto &arg = std::string(argv[i]);
      if (arg == "--help") {
        printUsage();
        exit(0);
      } else if (arg == "--onnx-convert-ops") {
        options[argv[i]] = "";
        if ((i + 1) < argc) {
          bool isDigitsOnly =
              std::regex_match(argv[i + 1], std::regex(R"(\d+)"));
          if (isDigitsOnly) {
            options[argv[i]] = argv[i + 1];
            i++;
          }
        }
        continue;
      } else if (arg == "--verbose") {
        options[argv[i]] = "";
        verbose = true;
      } else if (arg == "--no-elide") {
        elide = false;
      } else if (arg == "--export-mlir") {
        if ((i + 1) < argc && argv[i + 1][0] != '-') {
          exportMLIRFilename = argv[++i];
        } else {
          llvm::errs() << "ERROR: --export-mlir requires a target filename.\n";
          printUsage();
          exit(-1);
        }
      } else if ((arg == "--canonicalize") || (arg == "--infershapes") ||
                 (arg == "--onnx2linalg")) {
        options[argv[i]] = "";
      } else {
        llvm::errs() << "ERROR: Unknown argument `" << arg << "`" << "\n";
        printUsage();
        exit(-1);
      }
    } else {
      if (!ONNXFilename.size()) {
        ONNXFilename = argv[i];
        continue;
      }
    }
  }

  // check input file
  if (!ONNXFilename.size()) {
    llvm::errs() << "ERROR: missing onnx_file\n";
    printUsage();
    exit(-1);
  }

  auto ONNXLoader =
      onnx2mlir::Importer<onnx2mlir::frontend::ONNXImporter>(options);
  auto ONNXConverter =
      onnx2mlir::Converter<onnx2mlir::frontend::ONNXConverter>(options);

  mlir::MLIRContext ctx;
  ONNXLoader.importModule(ONNXFilename, &ctx);

  auto module = ONNXLoader.getMLIRModule();
  ONNXConverter.convertModule(module);
  module = ONNXLoader.getMLIRModule();

  if (verbose)
    printModule(module, elide);

  // export MLIR IR
  if (!exportMLIRFilename.empty()) {
    if (!saveModuleToFile(module, exportMLIRFilename)) {
      llvm::errs() << "ERROR: Saving MLIR IR\n";
      exit(-1);
    }
    llvm::outs() << "Saved MLIR IR to: " << exportMLIRFilename << "\n";
  }

  llvm::outs() << "\n";
  llvm::outs() << "Program finished successfully.\n";

  return 0;
}
