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
 * \file src/dialect/onnx/support/onnx_fold.cpp
 * \brief Onnx operator folder implementation
 */

#include <llvm/Support/Casting.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect::onnx {

static mlir::DenseElementsAttr getConstantAttr(mlir::Value val) {
  mlir::Attribute attr;
  if (mlir::matchPattern(val, mlir::m_Constant(&attr)))
    return llvm::dyn_cast_if_present<mlir::DenseElementsAttr>(attr);
  return nullptr;
}

mlir::OpFoldResult foldONNXOp(mlir::Operation *op,
                              llvm::ArrayRef<mlir::Attribute>) {
  llvm::StringRef opName = op->getName().getStringRef();

  if (opNameBeginsWith(opName, "Constant")) {
    if (auto valAttr = op->getAttrOfType<mlir::Attribute>("value"))
      return valAttr;
    return nullptr;
  }

  // todo ConstantOfShape, Shape, Gather,

  if (opNameBeginsWith(opName, {"Add", "Sub", "Mul", "Div", "Mod"})) {
    auto lhsAttr = getConstantAttr(op->getOperand(0));
    auto rhsAttr = getConstantAttr(op->getOperand(1));

    if (!lhsAttr || !rhsAttr)
      return nullptr;

    auto lhsType = llvm::cast<mlir::ShapedType>(lhsAttr.getType());
    auto rhsType = llvm::cast<mlir::ShapedType>(rhsAttr.getType());

    // broadcast not supported
    if (lhsType.getShape() != rhsType.getShape())
      return nullptr;

    auto elemType = lhsType.getElementType();
    size_t numElements = lhsType.getNumElements();

    if (elemType.isFloat()) {
      auto lhsValues = lhsAttr.getValues<llvm::APFloat>();
      auto rhsValues = rhsAttr.getValues<llvm::APFloat>();

      llvm::SmallVector<llvm::APFloat, 4> resultValues;
      resultValues.reserve(numElements);

      auto computeFloat = [=](llvm::APFloat l, llvm::APFloat r) {
        llvm::APFloat val = l;
        if (opNameBeginsWith(opName, "Add")) {
          val.add(r, llvm::APFloat::rmNearestTiesToEven);
        } else if (opNameBeginsWith(opName, "Sub")) {
          val.subtract(r, llvm::APFloat::rmNearestTiesToEven);
        } else if (opNameBeginsWith(opName, "Mul")) {
          val.multiply(r, llvm::APFloat::rmNearestTiesToEven);
        } else if (opNameBeginsWith(opName, "Div")) {
          if (!r.isZero())
            val.divide(r, llvm::APFloat::rmNearestTiesToEven);
        } else if (opNameBeginsWith(opName, "Mod")) {
          double lDouble = l.convertToDouble();
          double rDouble = r.convertToDouble();
          if (rDouble != 0.0) {
            double modRes = std::fmod(lDouble, rDouble);
            val = llvm::APFloat(modRes);
          }
        }
        return val;
      };

      if (lhsAttr.isSplat() && !rhsAttr.isSplat()) {
        const auto lhsVal = lhsValues[0];
        for (const auto &rhsVal : rhsValues)
          resultValues.push_back(computeFloat(lhsVal, rhsVal));
      } else if (!lhsAttr.isSplat() && rhsAttr.isSplat()) {
        const auto rhsVal = rhsValues[0];
        for (const auto &lhsVal : lhsValues)
          resultValues.push_back(computeFloat(lhsVal, rhsVal));
      } else {
        for (size_t i = 0; i < numElements; ++i)
          resultValues.push_back(computeFloat(lhsValues[i], rhsValues[i]));
      }
      return mlir::DenseElementsAttr::get(lhsType, resultValues);
    }

    if (elemType.isInteger()) {
      auto lhsValues = lhsAttr.getValues<llvm::APInt>();
      auto rhsValues = rhsAttr.getValues<llvm::APInt>();

      unsigned bitWidth = elemType.getIntOrFloatBitWidth();
      llvm::SmallVector<llvm::APInt, 4> resultValues;
      resultValues.reserve(numElements);

      auto computeInt = [=](llvm::APInt l, llvm::APInt r) {
        if (opNameBeginsWith(opName, "Add"))
          return l + r;
        if (opNameBeginsWith(opName, "Sub"))
          return l - r;
        if (opNameBeginsWith(opName, "Mul"))
          return l * r;
        if (opNameBeginsWith(opName, "Div"))
          return (!r.isZero()) ? l.sdiv(r) : llvm::APInt(bitWidth, 0);
        if (opNameBeginsWith(opName, "Mod"))
          return (!r.isZero()) ? l.srem(r) : llvm::APInt(bitWidth, 0);
        return llvm::APInt(bitWidth, 0);
      };

      if (lhsAttr.isSplat() && !rhsAttr.isSplat()) {
        const auto lhsVal = lhsValues[0];
        for (const auto &rhsVal : rhsValues)
          resultValues.push_back(computeInt(lhsVal, rhsVal));
      } else if (!lhsAttr.isSplat() && rhsAttr.isSplat()) {
        const auto rhsVal = rhsValues[0];
        for (const auto &lhsVal : lhsValues)
          resultValues.push_back(computeInt(lhsVal, rhsVal));
      } else {
        for (size_t i = 0; i < numElements; ++i)
          resultValues.push_back(computeInt(lhsValues[i], rhsValues[i]));
      }
      return mlir::DenseElementsAttr::get(lhsType, resultValues);
    }
  }
  return nullptr;
}

mlir::OpFoldResult foldONNXOp(mlir::Operation *op, mlir::DictionaryAttr attrs) {
  llvm::SmallVector<mlir::Attribute, 4> attrVec;
  for (auto namedAttr : attrs)
    attrVec.push_back(namedAttr.getValue());
  return foldONNXOp(op, llvm::ArrayRef<mlir::Attribute>(attrVec));
}

} // namespace onnx2mlir::dialect::onnx
