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
 * \file src/conversion/conversion.cpp
 * \brief Common conversion helper functions
 */

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

namespace onnx2mlir::dialect {

bool opNameBeginsWith(const llvm::StringRef &OpName, const std::string &match) {
  return std::regex_match(OpName.str(),
                          std::regex("^onnx." + match + "(_.*)?$"));
}

bool opNameBeginsWith(const llvm::StringRef &opName,
                      const std::vector<std::string> &matches) {
  for (const auto &match : matches) {
    if (std::regex_match(opName.str(),
                         std::regex("^onnx." + match + "(_.*)?$"))) {
      return true;
    }
  }
  return false;
}

mlir::RankedTensorType getBroadcastShape(mlir::RankedTensorType lhsType,
                                         mlir::RankedTensorType rhsType) {
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();

  unsigned lhsRank = lhsType.getRank();
  unsigned rhsRank = rhsType.getRank();
  unsigned maxRank = std::max(lhsRank, rhsRank);

  llvm::SmallVector<int64_t, 4> resultShape;
  resultShape.resize(maxRank);

  for (unsigned i = 1; i <= maxRank; ++i) {
    int64_t lhsDim = (lhsRank >= i) ? lhsShape[lhsRank - i] : 1;
    int64_t rhsDim = (rhsRank >= i) ? rhsShape[rhsRank - i] : 1;
    int64_t resultDim;

    // dim are equal,
    // or one of them is 1
    // or kDynamic.
    if (lhsDim == rhsDim) {
      resultDim = lhsDim;
    } else if (lhsDim == 1) {
      resultDim = rhsDim;
    } else if (rhsDim == 1) {
      resultDim = lhsDim;
    } else if (lhsDim == mlir::ShapedType::kDynamic ||
               rhsDim == mlir::ShapedType::kDynamic) {
      resultDim = mlir::ShapedType::kDynamic;
    } else {
      return {};
    }

    resultShape[maxRank - i] = resultDim;
  }

  return mlir::RankedTensorType::get(resultShape, lhsType.getElementType());
}

/**
 * Helper to ensure a type is signless for the arith dialect.
 * Converts unsigned/signed to signless recursively for tensors.
 */
mlir::Type getSignlessType(mlir::Type type) {
  if (!type)
    return nullptr;

  // handle scalars
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (!intType.isSignless()) {
      return mlir::IntegerType::get(type.getContext(), intType.getWidth());
    }
    return type;
  }

  // handle shaped types
  if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type)) {
    mlir::Type elementType = shapedType.getElementType();
    mlir::Type signlessElt = getSignlessType(elementType);

    // recreate on change
    if (signlessElt != elementType) {
      return shapedType.clone(signlessElt);
    }
  }

  return type;
}

/**
 * Public helper to change attribute signedness
 * from a potentially signed/unsigned attribute.
 */
mlir::TypedAttr changeAttrType(mlir::Attribute attr, mlir::Type toType) {
  if (!attr || !toType)
    return nullptr;

  if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    // element types mismatch, bitcast the data
    auto targetEltType = mlir::cast<mlir::ShapedType>(toType).getElementType();
    if (denseAttr.getElementType() != targetEltType) {
      return mlir::cast<mlir::TypedAttr>(denseAttr.bitcast(targetEltType));
    }
    return denseAttr;
  }

  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    // scalar types mismatch, re-wrap the value
    if (intAttr.getType() != toType) {
      return mlir::IntegerAttr::get(toType, intAttr.getValue());
    }
    return intAttr;
  }

  // default
  return mlir::dyn_cast<mlir::TypedAttr>(attr);
}

} // namespace onnx2mlir::dialect
