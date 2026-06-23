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
# pylint: disable=invalid-name
# pylint: disable=line-too-long,too-many-locals,too-many-branches,too-many-statements
# pylint: disable=consider-using-with,consider-using-f-string,f-string-without-interpolation

"""
\file tools/gen_onnx_cov.py
\brief A tool for generating MLIR ONNX operator coverage
"""

import argparse
from datetime import datetime
import onnx
from onnx import defs

from onnx2mlir.passes import register_onnx_to_linag_pass


def main():
    """Generate MLIR ONNX operators coverage"""

    parser = argparse.ArgumentParser(description="MLIR ONNX ops coversge generator.")
    parser.add_argument("output_doc_ops_cov", help="ONNX htm+svg coverage output")
    parser.add_argument("-debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    htm = open(args.output_doc_ops_cov + ".htm", "w", encoding="utf-8")
    svg = open(args.output_doc_ops_cov + ".svg", "w", encoding="utf-8")

    ops_versions = {}
    # map Ops versions
    for schema in defs.get_all_schemas_with_history():
        if schema.name not in ops_versions:
            ops_versions[schema.name] = []
        ops_versions[schema.name].append(int(schema.since_version))

    ##
    ## Operators
    ##

    onnx_ops = {}
    onnx_lowerable = register_onnx_to_linag_pass()

    for schema in defs.get_all_schemas_with_history():

        opname = schema.name
        if opname not in onnx_ops:
            onnx_ops[opname] = [max(ops_versions[schema.name])]

        # older Op versioning
        if schema.since_version != max(ops_versions[schema.name]):
            onnx_ops[opname].append(schema.since_version)

    ###
    ### HTML Table
    ###

    OPS_PER_ROW = 6
    HTM_BOX_HEIGHT = "50px"
    SVG_BOX_WIDTH = 135
    SVG_BOX_HEIGHT = 50
    SVG_SPACING = 1

    htm.write("<html><body style='font-family: sans-serif; padding: 20px;'>\n")

    htm.write("<div style='text-align: center; margin-bottom: 20px;'><font size=2>\n")
    htm.write("<h2>ONNX Operator Coverage</h2>\n")
    htm.write("<p><strong>ONNX version:</strong> %s\n" % onnx.__version__)
    htm.write(
        "<br><strong>Generated at:</strong> %s</p>\n"
        % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    htm.write("</font></div>\n")

    htm.write(
        "<table border=2px align=center style='border-collapse: collapse; text-align: center; width: 65%; table-layout: fixed;'>\n"
    )
    htm.write("<tr>\n")

    sorted_ops = sorted(onnx_ops.keys())
    total_rows = (len(sorted_ops) + OPS_PER_ROW - 1) // OPS_PER_ROW
    svg_width = OPS_PER_ROW * (SVG_BOX_WIDTH + SVG_SPACING) + SVG_SPACING
    svg_height = total_rows * (SVG_BOX_HEIGHT + SVG_SPACING) + SVG_SPACING
    svg.write(
        "<svg width='%i' height='%i' xmlns='http://www.w3.org/2000/svg'>\n"
        % (svg_width, svg_height)
    )
    svg.write(
        "<style>.op-text { font-family: sans-serif; font-size: 10px; } .ver-text { font-size: 7px; }</style>\n"
    )

    col_width = 100 // OPS_PER_ROW

    for i, op in enumerate(sorted_ops):
        if i > 0 and i % OPS_PER_ROW == 0:
            htm.write("</tr>\n<tr>")

        col = i % OPS_PER_ROW
        row = i // OPS_PER_ROW
        x = SVG_SPACING + col * (SVG_BOX_WIDTH + SVG_SPACING)
        y = SVG_SPACING + row * (SVG_BOX_HEIGHT + SVG_SPACING)

        bg_color = "#ccffcc" if op in onnx_lowerable else "#ffcccc"

        htm.write(
            "  <td style='border: 2px solid gray; width: %i%%; height: %s; vertical-align: middle; word-wrap: break-word; background-color: %s'>\n"
            % (col_width, HTM_BOX_HEIGHT, bg_color)
        )
        htm.write("  <font size=2><b>%s</b></font><br>" % op)

        svg.write(
            "  <rect x='%i' y='%i' width='%i' height='%i' fill='%s' stroke='black' stroke-width='2'/>\n"
            % (x, y, SVG_BOX_WIDTH, SVG_BOX_HEIGHT, bg_color)
        )
        svg.write(
            "  <text x='%i' y='%i' text-anchor='middle' font-weight='bold' class='op-text'>%s</text>\n"
            % (x + SVG_BOX_WIDTH / 2, y + 25, op)
        )

        vers = onnx_ops[op]
        vers_htm = "<font size=1 color='gray'>[</font>"
        vers_svg = "["
        for idx, ver in enumerate(vers):
            color = "blue" if idx == 0 else "gray"
            vers_htm += "<font size=1 color='%s'>%s</font>" % (color, ver)
            vers_svg += "<tspan fill='%s'>%s </tspan>" % (color, ver)
            if idx < len(vers) - 1:
                vers_htm += ",&nbsp"
                vers_svg += ", "
        vers_htm += "<font size=1 color='gray'>]</font>"
        vers_svg += "]"

        htm.write("  <sub>%s</sub>\n" % vers_htm)
        htm.write("  </td>\n")

        svg.write(
            "  <text x='%i' y='%i' text-anchor='middle' class='op-text ver-text'>%s</text>\n"
            % (x + SVG_BOX_WIDTH / 2, y + 40, vers_svg)
        )

    remaining = (OPS_PER_ROW - (len(sorted_ops) % OPS_PER_ROW)) % OPS_PER_ROW
    for _ in range(remaining):
        htm.write(
            "  <td style='width: %i%%; height: %s;'></td>\n"
            % (col_width, HTM_BOX_HEIGHT)
        )

    htm.write("</tr>\n</table>\n</body></html>\n")
    svg.write("</svg>\n")


if __name__ == "__main__":
    main()
