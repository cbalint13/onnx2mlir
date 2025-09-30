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
# pylint: disable=line-too-long,too-many-locals,too-many-branches,too-many-statements
# pylint: disable=consider-using-with,consider-using-f-string,f-string-without-interpolation

"""
\file tools/gen_onnx_dia.py
\brief A tool for generating ONNX operator tablegen definitions
"""

import re
import argparse
from datetime import datetime
import onnx
from onnx import defs


def get_mlir_types_from_str(type_strs, schema_constraints, option=None):
    """Get MLIR types from ONNX types

    Parameters
    ----------
    type_strs:
      The Onnx types string

    schema_constraints:
      Te Onnx schema constrains

    option:
      The formal parameter option

    Returns:
    --------
    mlir_types:
      The MLIR types string
    """
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
        "float8e8m0": "F8E8M0FNU",
        "float4e2m1": "F4E2M1FN",
        "complex64": "Complex<F32>",
        "complex128": "Complex<F64>",
        "string": "StringType",
    }

    attr_constraints = []
    # pick up direct constraint
    attr_constraints.append(type_strs.type_str)
    # pick up matched constraints
    for attr in schema_constraints:
        if attr.type_param_str == type_strs.type_str:
            attr_constraints = attr.allowed_type_strs
            break

    mlir_types = ""
    # build mlir attributes
    for idx, attr in enumerate(attr_constraints):

        if attr.find("optional") == 0:
            attr = attr.replace("optional(", "OptOf<", 1)
            attr = attr[:-1] + ">"

        pattern = "|".join(
            re.escape(key)
            for key in sorted(onnx_to_mlir_types.keys(), key=len, reverse=True)
        )
        mlir_types += re.sub(pattern, lambda m: onnx_to_mlir_types[m.group(0)], attr)
        mlir_types += ", " if (idx + 1) < len(attr_constraints) else ""

    # Optional absence to NoneType
    if option in ("NoneType", defs.OpSchema.FormalParameterOption.Optional):
        mlir_types += ", NoneType"

    if ", " in mlir_types:
        mlir_types = f"AnyTypeOf<[{mlir_types}]>"

    if option == defs.OpSchema.FormalParameterOption.Optional:
        mlir_types = f"Optional<{mlir_types}>"

    if option == defs.OpSchema.FormalParameterOption.Variadic:
        mlir_types = f"Variadic<{mlir_types}>"

    return mlir_types


def get_mlir_attrs_from_str(attr):
    """Get MLIR attributes from ONNX attributes

    Parameters
    ----------
    attr:
      The Onnx attributes string

    Returns:
    --------
    mlir_attr:
      The MLIR attributes string
    """

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
        "UNDEFINED": "AnyAttr",
    }

    mlir_attr = onnx_to_mlir_attrs[attr.type.name]

    if mlir_attr is None:
        return None
    if not mlir_attr:
        mlir_attr = "AnyAttr"

    default_value = onnx.helper.get_attribute_value(attr.default_value)
    if default_value is not None:
        if isinstance(default_value, float):
            default_value = "%g" % default_value
        if isinstance(default_value, bytes):
            default_value = default_value.decode("utf-8")
        if isinstance(default_value, list):
            if all(isinstance(item, int) for item in default_value):
                decoded_strings = [str(item) for item in default_value]
            else:
                decoded_strings = [
                    '\\"' + item.decode("utf-8") + '\\"' for item in default_value
                ]
            default_value = "{" + ",".join(decoded_strings) + "}"
        attr_type = "StrAttr" if (mlir_attr == "StrAttr") else "Attr"
        mlir_attr = f'DefaultValued{attr_type}<{mlir_attr}, "{default_value}">'

    if (not attr.required) and (default_value is None):
        mlir_attr = f"OptionalAttr<{mlir_attr}>"

    return mlir_attr


def main():
    """ONNX dialect tablegen file generator"""

    parser = argparse.ArgumentParser(description="MLIR ONNX ops generator.")
    parser.add_argument("output_mlir_ops_inc", help="MLIR Ops file output")
    parser.add_argument("-debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    inc = open(args.output_mlir_ops_inc, "w", encoding="utf-8")

    ##
    ## Header
    ##

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

    ##
    ## Operators
    ##

    for schema in defs.get_all_schemas_with_history():

        opname = schema.name
        # older Op versioning
        if schema.since_version != max(ops_versions[schema.name]):
            opname += f"_V{schema.since_version}"

        opinterfaces = "Pure, OPCountInfo"
        if schema.name == "Constant":
            opinterfaces += ", ConstantLike"

        if any(
            inp.option == defs.OpSchema.FormalParameterOption.Optional
            for inp in schema.inputs
        ):
            opinterfaces += ", SameVariadicOperandSize"

        if any(
            out.option == defs.OpSchema.FormalParameterOption.Optional
            for out in schema.outputs
        ):
            opinterfaces += ", SameVariadicResultSize"

        ##
        ## Definition
        ##

        inc.write(f"\n")
        inc.write(f"// {schema.name} [v{schema.since_version}]\n")
        inc.write(f'def Onnx_{opname}Op : Onnx_Op<"{opname}", [{opinterfaces}]> {{\n')
        inc.write(f'  let summary = "ONNX {opname} [v{schema.since_version}]";\n')
        inc.write(f"  let description = [{{\n")

        # shorted doc text
        doctxt = "\n".join(
            [
                ("  " + line.lstrip()) if line else ""
                for line in schema.doc.splitlines()
                if line
            ]
        )
        doctxt = doctxt.split(".", maxsplit=1)[0]
        if "." not in doctxt[-3:]:
            doctxt += "."

        inc.write(f"    {doctxt}\n")
        inc.write(f"  }}];\n")
        inc.write(f"\n")

        ##
        ## Arguments
        ##

        inp_args = []
        inp_args_str = "  let arguments = (ins "
        prefill = " " * len(inp_args_str)
        for inp in schema.inputs:
            mlir_types = get_mlir_types_from_str(
                inp, schema.type_constraints, inp.option
            )
            inp_name = inp.name
            if any(inp_name == item[0] for item in inp_args):
                inp_name = "input_" + inp_name
            inp_args_str += f"{mlir_types}:${inp_name},\n" + prefill
            if "Optional" in mlir_types:
                inp_args.append((inp_name, "optional"))
            elif "Variadic" in mlir_types:
                inp_args.append((inp_name, "variadic"))
            else:
                inp_args.append((inp_name, "operand"))

        ##
        ## Attributes
        ##

        inp_attrs = []
        for attr in sorted(schema.attributes):
            mlir_attr = get_mlir_attrs_from_str(schema.attributes[attr])
            if mlir_attr is not None:
                inp_attr = attr
                if any(inp_attr == item[0] for item in inp_args):
                    inp_attr = "input_" + inp_attr
                inp_args_str += f"{mlir_attr}:${inp_attr},\n" + prefill
                if "Default" in mlir_attr:
                    inp_attrs.append((inp_attr, "default"))
                elif "Optional" in mlir_attr:
                    inp_attrs.append((inp_attr, "optional"))
                elif "Variadic" in mlir_attr:
                    inp_attrs.append((inp_attr, "variadic"))
                else:
                    inp_attrs.append((inp_attr, "attribute"))
        if inp_args_str:
            # trim last comma
            inp_args_str = inp_args_str[: inp_args_str.rfind(",")]

        inc.write(f"{inp_args_str});\n")

        ##
        ## Results
        ##

        out_results = []
        out_results_str = "  let results = (outs "
        prefill = " " * len(out_results_str)
        for out in schema.outputs:
            out_option = (
                out.option
                # allow NoneType for Constant
                if (schema.name != "Constant")
                else "NoneType"
            )
            mlir_types = get_mlir_types_from_str(
                out, schema.type_constraints, out_option
            )
            out_name = out.name.replace(".", "")
            if any(out_name == item[0] for item in inp_args):
                out_name = "output_" + out_name
            out_results_str += f"{mlir_types}:${out_name},\n" + prefill
            if "Optional" in mlir_types:
                out_results.append((out_name, "optional"))
            elif "Variadic" in mlir_types:
                out_results.append((out_name, "variadic"))
            else:
                out_results.append((out_name, "result"))
        if out_results_str:
            # trim last comma
            out_results_str = out_results_str[: out_results_str.rfind(",")]

        inc.write(f"{out_results_str});\n")

        ##
        ## Assembly format
        ##

        indent_spaces = "`\\n`" + (" ` `" * 8)
        out_assembly_str = "  let assemblyFormat = [{\n"
        prefill = " " * (len(out_assembly_str) // 2)
        # operands
        indent_spaces = "`\\n`" + (" ` `" * 8)
        out_assembly_str += prefill + f"`(` {indent_spaces}\n"
        for oper, otype in inp_args:
            if otype in ("operand", "variadic"):
                out_assembly_str += " " * 4 + prefill
                out_assembly_str += f"`{oper}` `=` ${oper} `:` type(${oper})"
                out_assembly_str += f" {indent_spaces}\n"
            if otype == "optional":
                out_assembly_str += " " * 4 + prefill
                out_assembly_str += (
                    f"(`{oper}` `=` ${oper}^ `:` type(${oper}) {indent_spaces})?"
                )
        # attributes
        attr_list = '"{' + ",".join(f'\\"{attr}\\"' for attr, atype in inp_attrs) + '}"'
        out_assembly_str += prefill
        out_assembly_str += (
            f"`attributes` custom<OnnxDictAsmPrinter>(attr-dict, {attr_list})\n"
        )
        out_assembly_str += prefill + f"{indent_spaces[0:20]}\n"
        out_assembly_str += prefill + "`)`"
        # results
        out_assembly_str += " `:` type(results)"
        inc.write(f"%s\n" % out_assembly_str)

        inc.write(f"  }}];\n")

        ##
        ## Class appendix
        ##

        def_operands = -1 if "Variadic" in inp_args_str else len(schema.inputs)
        inc.write(f"  let extraClassDeclaration = [{{\n")
        inc.write(f"    int getDefinedOperandCount() {{\n")
        inc.write(f"      return %i;\n" % def_operands)
        inc.write(f"    }}\n")
        def_results = -1 if "Variadic" in out_results_str else len(schema.outputs)
        inc.write(f"    int getDefinedResultCount() {{\n")
        inc.write(f"      return %i;\n" % def_results)
        inc.write(f"    }}\n")
        inc.write(f"  }}];\n")

        inc.write(f"}}\n")

        ##
        ## Debug
        ##

        if args.debug:
            print("\n====================[%s]=======================\n" % schema.name)
            print(
                "OP domain:[%s] level:[%s] name:[%s] ver:[%s]"
                % (
                    schema.domain,
                    schema.support_level,
                    schema.name,
                    schema.since_version,
                )
            )
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
