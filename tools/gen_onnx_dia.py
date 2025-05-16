#!/usr/bin/env python3

###############################################################################
#
#  ONNX2MLIR (ONNX dialect mappings for composable optimizations)
#
#  Authors:
#   Cristian Balint <cristian dot balint at gmail dot com>
#
#  Copyright (c) 2021,2025
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

##
## \file tools/gen_onnx_dia.py
##


import re
import sys
import onnx
import argparse
from collections import defaultdict, OrderedDict
from datetime import datetime
from onnx import defs


def get_mlir_types_from_str(type_strs, schema_constraints, option = None):
  onnx_to_mlir_types = {
    "(": "<[",
    ")": "]>",
    "tensor": "TensorOf",
    "seq": "SeqOf",
    "map": "TupleOf",
    "bool": "I1",
    "uint4": "UI<4>",
    "uint8": "UI8",
    "uint16": "UI16",
    "uint32": "UI32",
    "uint64": "UI64",
    "int4": "I<4>",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "double": "F64",
    "float": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "float8e4m3fn": "F8E4M3FN",
    "float8e4m3fnuz": "F8E4M3FNUZ",
    "float8e5m2": "F8E5M2",
    "float8e5m2fnuz": "F8E5M2FNUZ",
    "float4e2m1": "F4E2M1FN",
    "complex64": "Complex<F32>",
    "complex128": "Complex<F64>",
    "string": "StringType"
  }

  attr_constraints = []
  # pick up direct constraint
  attr_constraints.append(type_strs.type_str)
  # pick up matched constraints
  for attr in schema_constraints:
    if (attr.type_param_str == type_strs.type_str):
      attr_constraints = attr.allowed_type_strs
      break;

  mlir_types = ""
  # build mlir attributes
  for idx, attr in enumerate(attr_constraints):

    if attr.find("optional") == 0:
      attr = attr.replace("optional(", "OptOf<", 1)
      attr = attr[:-1] + ">"

    pattern = '|'.join(re.escape(key) for key in sorted(onnx_to_mlir_types.keys(), key=len, reverse=True))
    mlir_type_str = re.sub(pattern, lambda m: onnx_to_mlir_types[m.group(0)], attr)
    mlir_types += f"{mlir_type_str}" + (", " if idx+1 < len(attr_constraints) else "")

  if option == onnx.defs.OpSchema.FormalParameterOption.Optional:
    mlir_types += ', NoneType'

  if ", " in mlir_types:
    mlir_types = f"AnyTypeOf<[{mlir_types}]>"

  if option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
    mlir_types = f"Variadic<{mlir_types}>"

  return mlir_types

def get_mlir_attrs_from_str(attr):

  onnx_to_mlir_attrs = {
    "INT": "I64Attr",
    "INTS": "I64ArrayAttr",
    "FLOAT": "F32Attr",
    "FLOATS": "F32ArrayAttr",
    "STRING": "StrAttr",
    "STRINGS": "StrArrayAttr",
    "TENSOR": "",
    "TENSORS": "",
    "SPARSE_TENSOR": "",
    "SPARSE_TENSORS": "",
    "GRAPH": None,
    "TYPE_PROTO": "TypeAttr",
    "TYPE_PROTOS": "",
    "UNDEFINED": "AnyAttr"
  }

  mlir_attr = onnx_to_mlir_attrs[attr.type.name]

  if mlir_attr is None:
    return None
  if not mlir_attr:
    mlir_attr = "AnyAttr"

  default_value = onnx.helper.get_attribute_value(attr.default_value)
  if default_value:
    if isinstance(default_value, bytes):
      default_value = default_value.decode('utf-8')
    attr_type = "StrAttr" if (mlir_attr == "StrAttr") else "Attr"
    mlir_attr = f'DefaultValued{attr_type}<{mlir_attr}, "{default_value}">'

  if not attr.required:
    mlir_attr = f"OptionalAttr<{mlir_attr}>"

  return mlir_attr

def main():

  parser = argparse.ArgumentParser(
    description="MLIR ONNX ops generator."
  )

  parser.add_argument(
    "output_mlir_ops_inc",
    help="Path to the MLIR ops file to be generated."
  )

  parser.add_argument(
    "-debug",
    action="store_true",
    help="Enable debug mode"
  )

  args = parser.parse_args()

  inc = open(args.output_mlir_ops_inc, "w")
  inc.write("/********************************************************\n")
  inc.write(" *   ONNX version [%s]%s*\n" % (onnx.__version__, " " * 30))
  inc.write(" *   Generated at [%s]%s*\n" % (datetime.now(), " " * 10))
  inc.write(" ********************************************************/\n")

  ops_versions = {}
  # map Ops versions
  for schema in defs.get_all_schemas_with_history():
    if schema.name not in ops_versions:
      ops_versions[schema.name] = []
    ops_versions[schema.name].append(int(schema.since_version))

  # iterate Ops
  for schema in defs.get_all_schemas_with_history():

    # use latest version Op
    if schema.since_version != max(ops_versions[schema.name]):
      continue

    opinterfaces = "Pure, OperandCountInfo"
    if schema.name == "Constant":
      opinterfaces += ", ConstantLike"

    # definition
    inc.write(f'\n')
    inc.write(f'/// {schema.name} [v{schema.since_version}]\n')
    inc.write(f'def Onnx_{schema.name}Op : Onnx_Op<"{schema.name}", [{opinterfaces}]> {{\n')
    inc.write(f'  let summary = "ONNX {schema.name} operation";\n')
    inc.write(f'  let description = [{{\n')
    doctxt = "\n".join([("  " + line.lstrip()) if line else "" for line in schema.doc.splitlines() if line])
    doctxt = doctxt.split('.')[0]
    if '.' not in doctxt[-3:] != '.': doctxt += '.'
    inc.write(f'    {doctxt}\n')
    inc.write(f'  }}];\n')
    inc.write(f'\n')

    # arguments
    inp_types_str = "  let arguments = (ins "
    if len(schema.inputs):
      for inp in schema.inputs:
        mlir_types = get_mlir_types_from_str(inp, schema.type_constraints, inp.option)
        inp_types_str += f'{mlir_types}:${inp.name},\n' + (' '*23)

    # attributes
    if len(schema.attributes):
      for idx, attr in enumerate(sorted(schema.attributes)):
        mlir_attr = get_mlir_attrs_from_str(schema.attributes[attr])
        if mlir_attr is not None:
          inp_types_str += f'{mlir_attr}:${attr},\n' + (' '*23)

    # trim last comma
    if len(inp_types_str):
      inp_types_str = inp_types_str[:inp_types_str.rfind(',')]
    inc.write(f'{inp_types_str});\n')

    # results
    out_types_str = "  let results = (outs "
    for idx, out in enumerate(schema.outputs):
      opt = None if (schema.name != "Constant") else onnx.defs.OpSchema.FormalParameterOption.Optional
      mlir_types = get_mlir_types_from_str(out, schema.type_constraints, opt)
      out_types_str += f'{mlir_types}:${out.name},%s' \
        % (('\n'+' '*21 if idx+1 != len(schema.outputs) else ''))
    out_types_str = out_types_str[:-1] if out_types_str[-1] == ',' else out_types_str
    inc.write(f'{out_types_str});\n')

    inc.write(f'  let extraClassDeclaration = [{{\n')
    inc.write(f'    int getDefinedOperandCount() {{\n')
    inc.write(f'      return %i;\n' % (-1 if "Variadic" in inp_types_str else len(schema.inputs)))
    inc.write(f'    }}\n')
    inc.write(f'  }}];\n')

    inc.write(f'}}\n')

    if args.debug:
      print("\n====================[%s]=======================\n" % schema.name)
      print(dir(schema.outputs))
      print("DBG [%s][%s][%s] [%s]" % (schema.domain, schema.support_level, schema.name, schema.since_version))
      print("   ATTR [%s]" % (schema.attributes))
      print("   DOC [%s]" % (schema.doc))
      for idx, inp in enumerate(schema.inputs):
        print("   IN#%i [%s]" % (idx, inp))
      for idx, out in enumerate(schema.outputs):
        print("   OUT#%i [%s]" % (idx, out))
      print("   FUNCBODY [%s]" % (schema.function_body))
      print("   TYPECONSTRAINTS [%s]" % (schema.type_constraints))

  inc.close()

if __name__ == "__main__":
  main()
