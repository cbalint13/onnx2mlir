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
 * \file src/main.cpp
 * \brief Main compiler program
 */

#include <iostream>
#include <map>
#include <regex>
#include <string>

#include "onnx2mlir/frontend/onnx.hpp"

static void printUsage() {
  std::cout << std::endl;
  std::cout << "Usage: onnx2mlir [--help]\n"
            << "             [--onnx-convert-ops <int : (optional | default is "
               "max supported)>]\n"
            << "       input_file\n\n";
}

int main(int argc, char **argv) {
  /// command-line params
  std::string ONNXFilename = "";
  std::map<std::string, std::string> options;

  /// command-line parser
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
      } else {
        std::cout << "Unknown argument `" << arg << "`" << std::endl;
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
    std::cout << "ERROR: missing onnx_file" << std::endl;
    printUsage();
    exit(-1);
  }

  auto ONNXLoader = new onnx2mlir::Importer<onnx2mlir::frontend::ONNXImporter>(options);
  auto ONNXConverter = new onnx2mlir::Converter<onnx2mlir::frontend::ONNXConverter>();

  ONNXLoader->importModule(ONNXFilename);
  ONNXConverter->convertModule(ONNXLoader->getMLIRModule());

  auto module = ONNXLoader->getMLIRModule();
  // DEBUG
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  // flags.printLargeElementsAttrWithHex();
  flags.enableDebugInfo();
  llvm::outs().enable_colors(true);
  module->print(llvm::outs(), flags);
  llvm::outs().enable_colors(false);

  return 0;
}
