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

#include <llvm/ADT/APSInt.h>
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
  auto opName = op->getName().getStringRef();

  if (opNameBeginsWith(opName, "Constant")) {
    if (auto valAttr = op->getAttrOfType<mlir::Attribute>("value"))
      return valAttr;
    return nullptr;
  }

  if (opNameBeginsWith(opName, "Cast")) {
    auto inputAttr = getConstantAttr(op->getOperand(0));
    if (!inputAttr)
      return nullptr;

    auto inputType = llvm::dyn_cast<mlir::ShapedType>(inputAttr.getType());
    auto resType = llvm::dyn_cast<mlir::ShapedType>(op->getResult(0).getType());

    if (!inputType || !resType || !inputType.hasStaticShape() ||
        !resType.hasStaticShape())
      return nullptr;

    if (inputType == resType)
      return inputAttr;

    auto srcElemType = inputType.getElementType();
    auto dstElemType = resType.getElementType();

    // round_mode
    llvm::APFloat::roundingMode attr_rm = llvm::APFloat::rmTowardZero;
    if (auto roundModeAttr =
            op->getAttrOfType<mlir::StringAttr>("round_mode")) {
      llvm::StringRef mode = roundModeAttr.getValue();
      if (mode == "up" || mode == "ceil")
        attr_rm = llvm::APFloat::rmTowardPositive;
      else if (mode == "down" || mode == "floor")
        attr_rm = llvm::APFloat::rmTowardNegative;
      else if (mode == "towards_zero")
        attr_rm = llvm::APFloat::rmTowardZero;
      else if (mode == "half_to_even")
        attr_rm = llvm::APFloat::rmNearestTiesToEven;
    }
    // saturate
    bool saturate = true;
    if (auto satAttr = op->getAttrOfType<mlir::IntegerAttr>("saturate"))
      saturate = (satAttr.getInt() != 0);

    auto mapElements = [&](auto dummySrc, auto convertVal) {
      using SrcValT = decltype(dummySrc);

      if (inputAttr.isSplat()) {
        auto val = convertVal(inputAttr.getSplatValue<SrcValT>());
        return mlir::DenseElementsAttr::get(resType, val);
      }

      using DstValT =
          std::decay_t<decltype(convertVal(std::declval<const SrcValT &>()))>;
      llvm::SmallVector<DstValT, 4> resultValues;
      resultValues.reserve(inputType.getNumElements());
      for (const auto &val : inputAttr.getValues<SrcValT>())
        resultValues.push_back(convertVal(val));

      return mlir::DenseElementsAttr::get(resType, resultValues);
    };

    // float -> float
    if (srcElemType.isFloat() && dstElemType.isFloat()) {
      auto floatType = mlir::cast<mlir::FloatType>(dstElemType);
      const auto &dstSemantics = floatType.getFloatSemantics();
      return mapElements(llvm::APFloat(0.0), [&](const llvm::APFloat &val) {
        llvm::APFloat res = val;
        bool losesInfo;
        res.convert(dstSemantics, attr_rm, &losesInfo);
        return res;
      });
    }

    // int -> int
    if (srcElemType.isInteger() && dstElemType.isInteger()) {
      unsigned srcBitWidth = srcElemType.getIntOrFloatBitWidth();
      unsigned dstBitWidth = dstElemType.getIntOrFloatBitWidth();
      bool srcIsUnsigned = srcElemType.isUnsignedInteger() || srcBitWidth == 1;
      bool dstIsUnsigned = dstElemType.isUnsignedInteger() || dstBitWidth == 1;
      return mapElements(llvm::APInt(), [&](const llvm::APInt &val) {
        if (dstBitWidth == 1)
          return llvm::APInt(1, val.isZero() ? 0 : 1);
        if (srcBitWidth == 1)
          return llvm::APInt(dstBitWidth, val.getZExtValue(), !dstIsUnsigned);
        if (srcBitWidth == dstBitWidth)
          return val;
        return srcIsUnsigned ? val.zextOrTrunc(dstBitWidth)
                             : val.sextOrTrunc(dstBitWidth);
      });
    }

    // float -> int
    if (srcElemType.isFloat() && dstElemType.isInteger()) {
      unsigned dstBitWidth = dstElemType.getIntOrFloatBitWidth();
      bool dstIsUnsigned = dstElemType.isUnsignedInteger() || dstBitWidth == 1;
      return mapElements(llvm::APFloat(0.0), [&](const llvm::APFloat &val) {
        if (dstBitWidth == 1)
          return llvm::APInt(1, (!val.isZero() && !val.isNaN()) ? 1 : 0);
        llvm::APSInt minInt =
            dstIsUnsigned ? llvm::APSInt::getMinValue(dstBitWidth, true)
                          : llvm::APSInt::getMinValue(dstBitWidth, false);
        llvm::APSInt maxInt =
            dstIsUnsigned ? llvm::APSInt::getMaxValue(dstBitWidth, true)
                          : llvm::APSInt::getMaxValue(dstBitWidth, false);
        if (val.isNaN())
          return static_cast<llvm::APInt>(
              llvm::APSInt(dstBitWidth, dstIsUnsigned));
        llvm::APSInt intVal(dstBitWidth, dstIsUnsigned);
        bool isExact;
        auto status = val.convertToInteger(intVal, attr_rm, &isExact);
        if (saturate && (status & (llvm::APFloat::opOverflow |
                                   llvm::APFloat::opInvalidOp))) {
          if (val.isNegative())
            return static_cast<llvm::APInt>(minInt);
          else
            return static_cast<llvm::APInt>(maxInt);
        }
        return static_cast<llvm::APInt>(intVal);
      });
    }

    // int -> float
    if (srcElemType.isInteger() && dstElemType.isFloat()) {
      auto floatType = mlir::cast<mlir::FloatType>(dstElemType);
      const auto &dstSemantics = floatType.getFloatSemantics();
      unsigned srcBitWidth = srcElemType.getIntOrFloatBitWidth();
      bool srcIsUnsigned = srcElemType.isUnsignedInteger() || srcBitWidth == 1;
      return mapElements(llvm::APInt(), [&](const llvm::APInt &val) {
        llvm::APFloat fltVal(dstSemantics);
        fltVal.convertFromAPInt(val, !srcIsUnsigned, attr_rm);
        return fltVal;
      });
    }
  }

  if (opNameBeginsWith(opName, "Shape")) {
    auto operandType =
        llvm::dyn_cast<mlir::ShapedType>(op->getOperand(0).getType());

    if (!operandType || !operandType.hasStaticShape())
      return nullptr;

    auto resType = llvm::dyn_cast<mlir::ShapedType>(op->getResult(0).getType());
    if (!resType)
      return nullptr;

    int64_t inputRank = operandType.getRank();

    // start
    int64_t attr_start = 0;
    if (auto startAttr = op->getAttrOfType<mlir::IntegerAttr>("start"))
      attr_start = startAttr.getInt();
    if (attr_start < 0)
      attr_start += inputRank;
    attr_start = std::clamp<int64_t>(attr_start, 0, inputRank);
    // end
    int64_t attr_end = inputRank;
    if (auto endAttr = op->getAttrOfType<mlir::IntegerAttr>("end")) {
      attr_end = endAttr.getInt();
      if (attr_end < 0)
        attr_end += inputRank;
      attr_end = std::clamp<int64_t>(attr_end, 0, inputRank);
    }

    auto shape = operandType.getShape();
    auto elemType = resType.getElementType();
    unsigned bitWidth = elemType.getIntOrFloatBitWidth();

    llvm::SmallVector<llvm::APInt, 4> resultValues;
    resultValues.reserve(attr_end - attr_start);

    for (int64_t i = attr_start; i < attr_end; ++i)
      resultValues.push_back(llvm::APInt(bitWidth, shape[i]));

    return mlir::DenseElementsAttr::get(resType, resultValues);
  }

  if (opNameBeginsWith(opName, "Gather")) {
    auto dataAttr = getConstantAttr(op->getOperand(0));
    auto indicesAttr = getConstantAttr(op->getOperand(1));

    if (!dataAttr || !indicesAttr)
      return nullptr;

    auto dataType = llvm::dyn_cast<mlir::ShapedType>(dataAttr.getType());
    auto indicesType = llvm::dyn_cast<mlir::ShapedType>(indicesAttr.getType());

    if (!dataType || !dataType.hasStaticShape() || !indicesType ||
        !indicesType.hasStaticShape())
      return nullptr;

    int64_t inputRank = dataType.getRank();
    if (inputRank < 1)
      return nullptr;

    // axis
    int64_t attr_axis = 0;
    if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
      attr_axis = axisAttr.getInt();
    if (attr_axis < -inputRank || attr_axis >= inputRank)
      return nullptr;
    if (attr_axis < 0)
      attr_axis += inputRank;
    attr_axis = std::clamp<int64_t>(attr_axis, 0, inputRank);

    if (attr_axis < 0 || attr_axis >= inputRank)
      return nullptr;

    auto dataShape = dataType.getShape();
    auto indicesShape = indicesType.getShape();
    int64_t indicesRank = indicesType.getRank();

    llvm::SmallVector<int64_t, 4> resultShape;
    resultShape.reserve(indicesRank + inputRank - 1);
    for (int64_t i = 0; i < attr_axis; ++i)
      resultShape.push_back(dataShape[i]);
    for (int64_t i = 0; i < indicesRank; ++i)
      resultShape.push_back(indicesShape[i]);
    for (int64_t i = attr_axis + 1; i < inputRank; ++i)
      resultShape.push_back(dataShape[i]);

    auto resType =
        mlir::RankedTensorType::get(resultShape, dataType.getElementType());

    if (dataAttr.isSplat())
      return mlir::DenseElementsAttr::get(
          resType, dataAttr.getSplatValue<mlir::Attribute>());

    int64_t outerSize = 1;
    for (int64_t i = 0; i < attr_axis; ++i)
      outerSize *= dataShape[i];

    int64_t axisSize = dataShape[attr_axis];

    int64_t innerSize = 1;
    for (int64_t i = attr_axis + 1; i < inputRank; ++i)
      innerSize *= dataShape[i];

    int64_t indicesNumElements = indicesType.getNumElements();

    llvm::SmallVector<int64_t, 4> parsedIndices;
    parsedIndices.reserve(indicesNumElements);
    if (indicesAttr.isSplat()) {
      int64_t k = indicesAttr.getSplatValue<llvm::APInt>().getSExtValue();
      if (k < 0)
        k += axisSize;
      if (k < 0 || k >= axisSize)
        return nullptr;
      parsedIndices.assign(indicesNumElements, k);
    } else {
      for (const auto &idxVal : indicesAttr.getValues<llvm::APInt>()) {
        int64_t k = idxVal.getSExtValue();
        if (k < 0)
          k += axisSize;
        if (k < 0 || k >= axisSize)
          return nullptr;
        parsedIndices.push_back(k);
      }
    }

    auto elemType = dataType.getElementType();
    size_t numResultElements = resType.getNumElements();

    if (elemType.isFloat() || elemType.isInteger()) {
      auto dataValues = dataAttr.getValues<mlir::Attribute>();
      llvm::SmallVector<mlir::Attribute, 4> resultValues;
      resultValues.reserve(numResultElements);

      for (int64_t outer_idx = 0; outer_idx < outerSize; ++outer_idx) {
        for (int64_t idx_elem = 0; idx_elem < indicesNumElements; ++idx_elem) {
          int64_t k = parsedIndices[idx_elem];
          int64_t base = (outer_idx * axisSize + k) * innerSize;
          for (int64_t inner_idx = 0; inner_idx < innerSize; ++inner_idx)
            resultValues.push_back(dataValues[base + inner_idx]);
        }
      }
      return mlir::DenseElementsAttr::get(resType, resultValues);
    }
  }

  if (opNameBeginsWith(opName, "Transpose")) {
    auto dataAttr = getConstantAttr(op->getOperand(0));
    if (!dataAttr)
      return nullptr;

    auto dataType = llvm::dyn_cast<mlir::ShapedType>(dataAttr.getType());
    if (!dataType || !dataType.hasStaticShape())
      return nullptr;

    int64_t r = dataType.getRank();
    if (r <= 0)
      return nullptr;

    // perms
    llvm::SmallVector<int64_t, 4> attr_perms;
    if (auto permAttr = op->getAttrOfType<mlir::ArrayAttr>("perm")) {
      for (auto intAttr : permAttr.getAsRange<mlir::IntegerAttr>()) {
        if (!intAttr)
          return nullptr;
        attr_perms.push_back(intAttr.getInt());
      }
      if (static_cast<int64_t>(attr_perms.size()) != r)
        return nullptr;
    } else {
      attr_perms.reserve(r);
      for (int64_t i = r - 1; i >= 0; --i)
        attr_perms.push_back(i);
    }

    llvm::SmallVector<bool, 4> seen(r, false);
    for (int64_t p : attr_perms) {
      if (p < 0 || p >= r || seen[p])
        return nullptr;
      seen[p] = true;
    }

    auto dataShape = dataType.getShape();
    llvm::SmallVector<int64_t, 4> resultShape(r);
    for (int64_t i = 0; i < r; ++i)
      resultShape[i] = dataShape[attr_perms[i]];

    auto resType =
        mlir::RankedTensorType::get(resultShape, dataType.getElementType());

    if (dataAttr.isSplat())
      return mlir::DenseElementsAttr::get(
          resType, dataAttr.getSplatValue<mlir::Attribute>());

    llvm::SmallVector<int64_t, 4> inputStrides(r, 1);
    for (int64_t i = r - 2; i >= 0; --i)
      inputStrides[i] = inputStrides[i + 1] * dataShape[i + 1];

    auto elemType = dataType.getElementType();
    size_t numResultElements = resType.getNumElements();

    if (elemType.isFloat() || elemType.isInteger()) {
      auto dataValues = dataAttr.getValues<mlir::Attribute>();
      llvm::SmallVector<mlir::Attribute, 4> resultValues;
      resultValues.reserve(numResultElements);

      llvm::SmallVector<int64_t, 4> outMultiIndex(r, 0);
      for (size_t outIdx = 0; outIdx < numResultElements; ++outIdx) {
        int64_t inIdx = 0;
        for (int64_t p = 0; p < r; ++p) {
          inIdx += outMultiIndex[p] * inputStrides[attr_perms[p]];
        }
        resultValues.push_back(dataValues[inIdx]);

        for (int64_t p = r - 1; p >= 0; --p) {
          outMultiIndex[p]++;
          if (outMultiIndex[p] < resultShape[p])
            break;
          outMultiIndex[p] = 0;
        }
      }
      return mlir::DenseElementsAttr::get(resType, resultValues);
    }

    return nullptr;
  }

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

    int64_t attr_fmod = 0;
    if (opNameBeginsWith(opName, "Mod")) {
      if (auto fmodAttr = op->getAttrOfType<mlir::IntegerAttr>("fmod"))
        attr_fmod = fmodAttr.getInt();
    }

    auto processBinaryOp = [&](auto dummy, auto computeFn) {
      using ValueT = decltype(dummy);
      auto lhsValues = lhsAttr.getValues<ValueT>();
      auto rhsValues = rhsAttr.getValues<ValueT>();

      llvm::SmallVector<ValueT, 4> resultValues;
      resultValues.reserve(numElements);

      if (lhsAttr.isSplat() && !rhsAttr.isSplat()) {
        const auto lhsVal = lhsValues[0];
        for (const auto &rhsVal : rhsValues)
          resultValues.push_back(computeFn(lhsVal, rhsVal));
      } else if (!lhsAttr.isSplat() && rhsAttr.isSplat()) {
        const auto rhsVal = rhsValues[0];
        for (const auto &lhsVal : lhsValues)
          resultValues.push_back(computeFn(lhsVal, rhsVal));
      } else {
        for (size_t i = 0; i < numElements; ++i)
          resultValues.push_back(computeFn(lhsValues[i], rhsValues[i]));
      }
      return mlir::DenseElementsAttr::get(lhsType, resultValues);
    };

    if (elemType.isFloat()) {
      auto computeFloatOp = [&](llvm::APFloat l,
                                llvm::APFloat r) -> llvm::APFloat {
        llvm::APFloat val = l;
        if (opNameBeginsWith(opName, "Add"))
          val.add(r, llvm::APFloat::rmNearestTiesToEven);
        if (opNameBeginsWith(opName, "Sub"))
          val.subtract(r, llvm::APFloat::rmNearestTiesToEven);
        if (opNameBeginsWith(opName, "Mul"))
          val.multiply(r, llvm::APFloat::rmNearestTiesToEven);
        if (opNameBeginsWith(opName, "Div")) {
          if (!r.isZero())
            val.divide(r, llvm::APFloat::rmNearestTiesToEven);
        }
        if (opNameBeginsWith(opName, "Mod")) {
          if (!r.isZero()) {
            double lDouble = l.convertToDouble();
            double rDouble = r.convertToDouble();
            double resDouble = std::fmod(lDouble, rDouble);
            bool losesInfo = false;
            val = llvm::APFloat(resDouble);
            val.convert(l.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                        &losesInfo);
          }
        }
        return val;
      };
      return processBinaryOp(llvm::APFloat(0.0), computeFloatOp);
    }

    if (elemType.isInteger()) {
      unsigned bitWidth = elemType.getIntOrFloatBitWidth();
      auto computeIntOp = [&](llvm::APInt l, llvm::APInt r) -> llvm::APInt {
        if (opNameBeginsWith(opName, "Add"))
          return l + r;
        if (opNameBeginsWith(opName, "Sub"))
          return l - r;
        if (opNameBeginsWith(opName, "Mul"))
          return l * r;
        if (opNameBeginsWith(opName, "Div"))
          return (!r.isZero()) ? l.sdiv(r) : llvm::APInt(bitWidth, 0);
        if (opNameBeginsWith(opName, "Mod")) {
          if (r.isZero())
            return llvm::APInt(bitWidth, 0);
          if (elemType.isUnsignedInteger())
            return l.urem(r);
          if (attr_fmod == 1)
            return l.srem(r);

          auto rem = l.srem(r);
          return (rem + r).srem(r);
        }
        return llvm::APInt(bitWidth, 0);
      };
      return processBinaryOp(llvm::APInt(), computeIntOp);
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
