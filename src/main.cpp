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

#include "onnx2mlir/frontend/onnx.hpp"

int main(int argc, char **argv) {

  /// command-line params
  bool printUsage = false;
  std::string ONNXFilename = "";

  /// command-line parser
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      if (!strcmp(argv[i], "--help")) {
        printUsage = true;
        i++;
        continue;
      }
      std::cout << "Unknown option: " << argv[i] << std::endl;
      printUsage = true;
      break;
    } else if (argv[i][0] != '-') {
      if (!ONNXFilename.size()) {
        ONNXFilename = argv[i];
        continue;
      }
    }
  }
  if (!ONNXFilename.size()) {
    std::cout << "ERROR: missing onnx_file" << std::endl;
  }
  /// check required options
  if (!ONNXFilename.size() || printUsage) {
    std::cout << std::endl;
    std::cout << "Usage: onnx2mlir [--help]\n"
              << "             onnx_file\n\n";
    exit(-1);
  }

  auto ONNXLoader = new onnx2mlir::Importer<onnx2mlir::frontend::ONNXImporter>();
  auto ONNXConverter = new onnx2mlir::Converter<onnx2mlir::frontend::ONNXConverter>();
  ONNXLoader->importModule(ONNXFilename);
  ONNXConverter->convertModule(ONNXLoader->getMLIRModule());

  return 0;
}
