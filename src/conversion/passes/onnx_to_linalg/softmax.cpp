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
 * \file src/conversion/passes/onnx_to_linalg/softmax.cpp
 * \brief ONNX Hardmax, LogSoftmax, Softmax operations to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"
#include "onnx2mlir/support/support.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_SoftmaxOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                       const mlir::TypeConverter *typeConverter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  auto &convRewriter = mlir::cast<mlir::ConversionPatternRewriter>(rewriter);

  mlir::Value opInput = convRewriter.getRemappedValue(op->getOperand(0));
  mlir::Value opOutput = convRewriter.getRemappedValue(op->getResult(0));

  auto inpDatType = mlir::dyn_cast<mlir::RankedTensorType>(opInput.getType());
  auto outDatType = mlir::dyn_cast<mlir::RankedTensorType>(opOutput.getType());

  auto inpElmType = inpDatType.getElementType();

  auto inputRank = inpDatType.getRank();

  /*
   * Attributes
   */

  // axis
  auto axisAttr = op->getAttr("axis");
  if (!axisAttr)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " missing 'axis' attribute";
  auto axisInt = mlir::dyn_cast_or_null<mlir::IntegerAttr>(axisAttr);
  if (!axisInt)
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " invalid 'axis' attribute type";
  auto attr_axis = axisInt.getInt();
  if (attr_axis < -inputRank || attr_axis >= inputRank) {
    return mlir::emitError(Onnx2Mlir_SrcLoc(rewriter))
           << opName << " invalid axis: " << attr_axis;
  }

  if (attr_axis < 0) {
    attr_axis = inputRank + attr_axis;
  }

  /*
   * Affine mappings
   */

  llvm::SmallVector<int64_t> reduceShp(inpDatType.getShape());
  reduceShp.erase(reduceShp.begin() + attr_axis);

  auto identityMap = rewriter.getMultiDimIdentityMap(inputRank);
  auto reduceBroadcastMap = identityMap.dropResult(attr_axis);

  mlir::SmallVector<mlir::AffineMap, 3> indexingMaps = {
      identityMap, reduceBroadcastMap, identityMap};

  mlir::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      inputRank, mlir::utils::IteratorType::parallel);

  /*
   * Linalg ops staging
   */

  auto fltType = mlir::cast<mlir::FloatType>(inpElmType);
  auto nInf = llvm::APFloat::getInf(fltType.getFloatSemantics(), /*neg=*/true);
  // -Inf
  mlir::Value negInf = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(fltType, nInf));
  // 0.0f
  mlir::Value zero = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(inpElmType, 0.0));
  // 1.0f
  mlir::Value one;

  mlir::Value expBuffer, sumBuffer;
  mlir::linalg::FillOp sumFillBuffer;

  mlir::Value maxBuffer =
      mlir::tensor::EmptyOp::create(rewriter, loc, reduceShp, inpElmType);
  auto maxFillBuffer =
      mlir::linalg::FillOp::create(rewriter, loc, negInf, maxBuffer);

  mlir::Value outBuffer = mlir::tensor::EmptyOp::create(
      rewriter, loc, inpDatType.getShape(), inpElmType);

  auto maxOp = mlir::linalg::ReduceOp::create(
      /*op_builder*/ rewriter, /*src_location*/ loc,
      /*input_values*/ mlir::ValueRange{opInput},
      /*output_values*/ mlir::ValueRange{maxFillBuffer.getResult(0)},
      /*reduct_dims*/ rewriter.getDenseI64ArrayAttr({attr_axis}),
      /*builder_callback*/
      [&](/*op_builder*/ mlir::OpBuilder nest,
          /*src_location*/ mlir::Location nloc,
          /*value_args*/ mlir::ValueRange args) {
        mlir::Value out;
        out = mlir::arith::MaximumFOp::create(nest, nloc, args[0], args[1]);
        mlir::linalg::YieldOp::create(nest, nloc, out);
      });

  mlir::linalg::GenericOp expOp;
  mlir::linalg::ReduceOp sumOp;

  if (opNameBeginsWith(opName, {"Softmax", "LogSoftmax"})) {
    expBuffer = mlir::tensor::EmptyOp::create(
        rewriter, loc, inpDatType.getShape(), inpElmType);

    sumBuffer =
        mlir::tensor::EmptyOp::create(rewriter, loc, reduceShp, inpElmType);
    sumFillBuffer =
        mlir::linalg::FillOp::create(rewriter, loc, zero, sumBuffer);

    expOp = mlir::linalg::GenericOp::create(
        /*op_builder*/ rewriter, /*src_location*/ loc,
        /*result_type*/ mlir::TypeRange{inpDatType},
        /*input_values*/ mlir::ValueRange{opInput, maxOp.getResult(0)},
        /*output_values*/ mlir::ValueRange{expBuffer},
        /*affine_maps*/ indexingMaps, /*iter_types*/ iteratorTypes,
        /*builder_callback*/
        [&](/*op_builder*/ mlir::OpBuilder nest,
            /*src_location*/ mlir::Location nloc,
            /*value_args*/ mlir::ValueRange args) {
          mlir::Value out;
          out = mlir::arith::SubFOp::create(nest, nloc, args[0], args[1]);
          out = mlir::math::ExpOp::create(nest, nloc, out);
          mlir::linalg::YieldOp::create(nest, nloc, out);
        });

    sumOp = mlir::linalg::ReduceOp::create(
        /*op_builder*/ rewriter, /*src_location*/ loc,
        /*input_values*/ mlir::ValueRange{expOp.getResult(0)},
        /*output_values*/ mlir::ValueRange{sumFillBuffer.getResult(0)},
        /*reduct_dims*/ rewriter.getDenseI64ArrayAttr({attr_axis}),
        /*builder_callback*/
        [&](/*op_builder*/ mlir::OpBuilder nest,
            /*src_location*/ mlir::Location nloc,
            /*value_args*/ mlir::ValueRange args) {
          mlir::Value out;
          out = mlir::arith::AddFOp::create(nest, nloc, args[0], args[1]);
          mlir::linalg::YieldOp::create(nest, nloc, out);
        });
  }

  mlir::linalg::GenericOp genericOp;

  if (opNameBeginsWith(opName, "Softmax")) {
    genericOp = mlir::linalg::GenericOp::create(
        /*op_builder*/ rewriter, /*src_location*/ loc,
        /*result_type*/ mlir::TypeRange{outDatType},
        /*input_values*/
        mlir::ValueRange{expOp.getResult(0), sumOp.getResult(0)},
        /*output_values*/ mlir::ValueRange{outBuffer},
        /*affine_maps*/ indexingMaps,
        /*iter_types*/ iteratorTypes,
        /*builder_callback*/
        [&](/*op_builder*/ mlir::OpBuilder nest,
            /*src_location*/ mlir::Location nloc,
            /*value_args*/ mlir::ValueRange args) {
          mlir::Value out;
          out = mlir::arith::DivFOp::create(nest, nloc, args[0], args[1]);
          mlir::linalg::YieldOp::create(nest, nloc, out);
        });
  } else if (opNameBeginsWith(opName, "LogSoftmax")) {
    genericOp = mlir::linalg::GenericOp::create(
        /*op_builder*/ rewriter, /*src_location*/ loc,
        /*result_type*/ mlir::TypeRange{outDatType},
        /*input_values*/
        mlir::ValueRange{expOp.getResult(0), sumOp.getResult(0)},
        /*output_values*/ mlir::ValueRange{outBuffer},
        /*affine_maps*/ indexingMaps,
        /*iter_types*/ iteratorTypes,
        [&](/*op_builder*/ mlir::OpBuilder nest,
            /*src_location*/ mlir::Location nloc,
            /*builder_callback*/ mlir::ValueRange args) {
          mlir::Value out;
          auto logExp = mlir::math::LogOp::create(nest, nloc, args[0]);
          auto logSum = mlir::math::LogOp::create(nest, nloc, args[1]);
          out = mlir::arith::SubFOp::create(nest, nloc, logExp, logSum);
          mlir::linalg::YieldOp::create(nest, nloc, out);
        });
  } else if (opNameBeginsWith(opName, "Hardmax")) {
    one = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(inpElmType, 1.0));

    genericOp = mlir::linalg::GenericOp::create(
        /*op_builder*/ rewriter, /*src_location*/ loc,
        /*result_type*/ mlir::TypeRange{outDatType},
        /*input_values*/ mlir::ValueRange{opInput, maxOp.getResult(0)},
        /*output_values*/ mlir::ValueRange{outBuffer},
        /*affine_maps*/ indexingMaps,
        /*iter_types*/ iteratorTypes,
        /*builder_callback*/
        [&](/*op_builder*/ mlir::OpBuilder nest,
            /*src_location*/ mlir::Location nloc,
            /*value_args*/ mlir::ValueRange args) {
          mlir::Value out;
          out = mlir::arith::CmpFOp::create(
              nest, nloc, mlir::arith::CmpFPredicate::OEQ, args[0], args[1]);
          out = mlir::arith::SelectOp::create(nest, nloc, out, one, zero);
          mlir::linalg::YieldOp::create(nest, nloc, out);
        });
  }

  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
