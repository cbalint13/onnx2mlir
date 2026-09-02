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
 * \file include/onnx2mlir/support/support.hpp
 * \brief ONNX2MLIR support utilities
 */

#ifndef INCLUDE_ONNX2MLIR_SUPPORT_SUPPORT_HPP_
#define INCLUDE_ONNX2MLIR_SUPPORT_SUPPORT_HPP_

#include <llvm/Support/WithColor.h>
#include <mlir/IR/Builders.h>

#include <regex>
#include <source_location>
#include <string>

inline mlir::Location Onnx2Mlir_SrcLoc(
    mlir::OpBuilder &builder,
    const std::source_location loc = std::source_location::current()) {
  return mlir::FileLineColLoc::get(builder.getStringAttr(loc.file_name()),
                                   loc.line(), loc.column());
}

namespace onnx2mlir {

// logging helpers
inline llvm::raw_ostream &
error(const std::source_location loc = std::source_location::current()) {
  llvm::errs() << loc.file_name() << ":" << loc.line() << ":" << loc.column()
               << " ";
  return llvm::WithColor::error(llvm::errs());
}

// op name helpers
static inline bool opNameBeginsWith(const llvm::StringRef &opName,
                                    llvm::StringRef match) {
  auto rExpr = std::regex("^onnx." + match.str() + "(_.*)?$");
  return std::regex_match(opName.str(), rExpr);
}

static inline bool opNameBeginsWith(const llvm::StringRef &opName,
                                    llvm::ArrayRef<llvm::StringRef> matches) {
  for (const auto &match : matches) {
    auto rExpr = std::regex("^onnx." + match.str() + "(_.*)?$");
    if (opNameBeginsWith(opName, match))
      return true;
  }
  return false;
}

} // namespace onnx2mlir

#endif // INCLUDE_ONNX2MLIR_SUPPORT_SUPPORT_HPP_
