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
 * \file python/src/dialect/onnx_dialect_bindings.cpp
 * \brief Python bindings for the ONNX dialect
 */

#include <mlir/Bindings/Python/PybindAdaptors.h>
#include <mlir/CAPI/Registration.h>

#include <map>
#include <string>

#include "onnx2mlir/frontend/onnx.hpp"

PYBIND11_MODULE(_onnx2mlirImporters, m) {
  m.doc() = "Python bindings for the ONNX2MLIR importers";

  m.def(
      "import_from_onnx",
      [](const std::string &ONNXFilename, int onnxConvertOps) -> MlirModule {
        std::map<std::string, std::string> options;
        if (onnxConvertOps >= 0)
          options["--onnx-convert-ops"] = std::to_string(onnxConvertOps);
        auto ONNXLoader =
            new onnx2mlir::Importer<onnx2mlir::frontend::ONNXImporter>(options);
        ONNXLoader->importModule(ONNXFilename);
        auto moduleOp = ONNXLoader->getMLIRModule();
        return wrap(moduleOp);
      },
      py::arg("onnxfilename"), py::arg("onnx_convert_ops") = -1,
      "Import from an ONNX file path");

  m.def(
      "import_from_onnx",
      [](py::object onnx_model_proto, int onnxConvertOps) -> MlirModule {
        std::map<std::string, std::string> options;
        if (onnxConvertOps >= 0)
          options["--onnx-convert-ops"] = std::to_string(onnxConvertOps);
        options["--import-serialized"] = "";
        py::bytes serialized_bytes =
            onnx_model_proto.attr("SerializeToString")();
        std::string ONNXSerialString = serialized_bytes;
        auto ONNXLoader =
            new onnx2mlir::Importer<onnx2mlir::frontend::ONNXImporter>(options);
        ONNXLoader->importModule(ONNXSerialString);
        auto moduleOp = ONNXLoader->getMLIRModule();
        return wrap(moduleOp);
      },
      py::arg("onnxfilename"), py::arg("onnx_convert_ops") = -1,
      "Import from an ONNX ModelProto object");
}
