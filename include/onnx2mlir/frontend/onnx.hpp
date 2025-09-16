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
 * \file include/onnx2mlir/frontend/onnx.hpp
 * \brief Onnx frontend declarations
 */

#ifndef INCLUDE_ONNX2MLIR_FRONTEND_ONNX_HPP_
#define INCLUDE_ONNX2MLIR_FRONTEND_ONNX_HPP_

#include <onnx/onnx_pb.h>

#include <map>
#include <string>
#include <vector>

#include "onnx2mlir/frontend/frontend.hpp"

namespace onnx2mlir {
namespace frontend {

/*
 * ONNX importer to ONNX MLIR dialect
 */
class ONNXImporter : public FrontendImporter {
public:
  explicit ONNXImporter(const std::map<std::string, std::string> &options);

protected:
  void import(const std::string &file_or_string) override;

private:
  // parse the graph ins & outs
  void parse_graph_io(const onnx::GraphProto &graph_proto);
  // parse the graph nodes
  void parse_graph_nodes(const onnx::GraphProto &graph_proto);
  // get versioned op name
  const std::string get_versioned_name(const std::string &OpName);
  // imported opset
  int model_opset_version;
  // onnx opset version
  int engine_opset_version;
  // onnx ops versioning catalog
  std::map<std::string, std::vector<int>> ops_versions;
};

/*
 * ONNX converter to MLIR TOSA dialect
 */
class ONNXConverter : public FrontendConverter {
protected:
  void convert(mlir::ModuleOp *module) override;
};

} // namespace frontend
} // namespace onnx2mlir

#endif // INCLUDE_ONNX2MLIR_FRONTEND_ONNX_HPP_
