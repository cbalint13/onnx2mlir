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

#include <mlir/Bindings/Python/NanobindAdaptors.h>
#include <mlir/CAPI/Registration.h>

#include <map>
#include <string>

#include "onnx2mlir/frontend/onnx.hpp"

NB_MODULE(_onnx2mlirImporters, m) {
  m.doc() = "Python bindings for the ONNX2MLIR importers";

  m.def(
      "import_from_onnx",
      [](const std::string &ONNXFilename, MlirContext context,
         int onnxConvertOps) -> MlirModule {
        std::map<std::string, std::string> options;
        if (onnxConvertOps >= 0)
          options["--onnx-convert-ops"] = std::to_string(onnxConvertOps);
        auto ONNXLoader =
            onnx2mlir::Importer<onnx2mlir::frontend::ONNXImporter>(options);
        mlir::MLIRContext *mlirCtx = unwrap(context);
        ONNXLoader.importModule(ONNXFilename, mlirCtx);
        auto moduleOp = ONNXLoader.getMLIRModule();
        return wrap(moduleOp);
      },
      nanobind::rv_policy::take_ownership, nanobind::arg("onnxfilename"),
      nanobind::arg("context"), nanobind::arg("onnx_convert_ops") = -1,
      "Import from an ONNX file path");

  m.def(
      "import_from_onnx",
      [](nanobind::object onnx_model_proto, MlirContext context,
         int onnxConvertOps) -> MlirModule {
        std::map<std::string, std::string> options;
        if (onnxConvertOps >= 0)
          options["--onnx-convert-ops"] = std::to_string(onnxConvertOps);
        options["--import-serialized"] = "";
        nanobind::bytes serialized_bytes(
            onnx_model_proto.attr("SerializeToString")());
        std::string ONNXSerialString(serialized_bytes.c_str(),
                                     serialized_bytes.size());
        auto ONNXLoader =
            onnx2mlir::Importer<onnx2mlir::frontend::ONNXImporter>(options);
        mlir::MLIRContext *mlirCtx = unwrap(context);
        ONNXLoader.importModule(ONNXSerialString, mlirCtx);
        auto moduleOp = ONNXLoader.getMLIRModule();
        return wrap(moduleOp);
      },
      nanobind::rv_policy::take_ownership, nanobind::arg("onnxfilename"),
      nanobind::arg("context"), nanobind::arg("onnx_convert_ops") = -1,
      "Import from an ONNX ModelProto object");
}
