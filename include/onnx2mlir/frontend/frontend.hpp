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

#ifndef INCLUDE_ONNX2MLIR_FRONTEND_FRONTEND_HPP_
#define INCLUDE_ONNX2MLIR_FRONTEND_FRONTEND_HPP_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace onnx2mlir {

/*
 * Importer backend caller
 */
template <typename ImporterBackend> class Importer : public ImporterBackend {
public:
  // forward constructor
  template <typename... Args>
  explicit Importer(Args &&...args)
      : ImporterBackend(std::forward<Args>(args)...) {}
  // input file parser
  void importModule(const std::string &filepath, mlir::MLIRContext *ctx) {
    ImporterBackend::import(filepath, ctx);
  }
  // get the MLIR context
  mlir::MLIRContext getMLIRCtx() { return ImporterBackend::get_mlir_ctx(); }
  // get the MLIR module
  mlir::ModuleOp getMLIRModule() { return ImporterBackend::get_mlir_module(); }
};

/*
 * Converter backend caller
 */
template <typename ConverterBackend> class Converter : public ConverterBackend {
public:
  // forward constructor
  template <typename... Args>
  explicit Converter(Args &&...args)
      : ConverterBackend(std::forward<Args>(args)...) {}
  // module converter
  void convertModule(mlir::ModuleOp module) {
    ConverterBackend::convert(&module);
  }
};

namespace frontend {

// importer interface
class FrontendImporter {
public:
  explicit FrontendImporter(const std::map<std::string, std::string> &options)
      : opt_args(options) {}

  virtual ~FrontendImporter() = default;

protected:
  virtual void import(const std::string &filepath, mlir::MLIRContext *ctx) = 0;

  mlir::MLIRContext *get_mlir_ctx() { return mlirCtx; }
  mlir::ModuleOp get_mlir_module() { return module; }

  // driver options argument
  std::map<std::string, std::string> opt_args;
  // MLIR context
  mlir::MLIRContext *mlirCtx{nullptr};
  // MLIR module
  mlir::ModuleOp module;
};

// converter interface
class FrontendConverter {
public:
  explicit FrontendConverter(const std::map<std::string, std::string> &options)
      : opt_args(options) {}

  virtual ~FrontendConverter() = default;

protected:
  virtual void convert(mlir::ModuleOp *module) = 0;

  // converter options argument
  std::map<std::string, std::string> opt_args;
};

} // namespace frontend
} // namespace onnx2mlir

#endif // INCLUDE_ONNX2MLIR_FRONTEND_FRONTEND_HPP_
