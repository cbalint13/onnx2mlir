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
 * \file include/onnx2mlir/frontend/frontend.hpp
 * \brief Generic frontend declarations
 */

#ifndef ONNX2MLIR_FRONTEND_IMPORTER_HPP_
#define ONNX2MLIR_FRONTEND_IMPORTER_HPP_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <string>

namespace onnx2mlir {

/*
 * Importer backend caller
 */
template <typename ImporterBackend> class Importer : public ImporterBackend {
public:
  // input file parser
  void importModule(const std::string &filepath) {
    ImporterBackend::import(filepath);
  }
  // get the MLIR context
  mlir::MLIRContext getMLIRCtx() {
    return ImporterBackend::get_mlir_ctx();
  }
  // get the MLIR module
  mlir::ModuleOp getMLIRModule() {
    return ImporterBackend::get_mlir_module();
  }
};

/*
 * Converter backend caller
 */
template <typename ConverterBackend> class Converter : public ConverterBackend {
public:
  // input file parser
  void convertModule(mlir::ModuleOp module) {
    ConverterBackend::convert(module);
  }
};

namespace frontend {

// importer interface
class FrontendImporter {
public:
  FrontendImporter()
  : mlirCtx(std::make_unique<mlir::MLIRContext>()) {};

  virtual ~FrontendImporter() = default;

protected:
  virtual void import(const std::string &filepath) = 0;

  mlir::MLIRContext* get_mlir_ctx() {
    return mlirCtx.get();
  };
  mlir::ModuleOp get_mlir_module() {
    return module.get();
  };

  // MLIR context
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  // MLIR module
  mlir::OwningOpRef<mlir::ModuleOp> module{nullptr};
};

// converter interface
class FrontendConverter {
public:
  virtual ~FrontendConverter() = default;

protected:
  virtual void convert(mlir::ModuleOp &module) = 0;
};

} // namespace frontend
} // namespace onnx2mlir

#endif // ONNX2MLIR_FRONTEND_IMPORTER_HPP_
