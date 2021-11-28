import os
import re
import sys
import json
import sop_parser

# Parse sop
sop_filepath = sys.argv[1]
sop_file = open(sop_filepath, "r")
sop_tree = sop_parser.sop_grammar.parse(sop_file.read())
sop_file.close()
sop_content = sop_parser.SopVisitor().visit(sop_tree)

# Fix array indentation
edges_str = str(sop_content["edges"])
edges_str = re.sub(" ", "", edges_str)
edges_str = re.sub("\[\[", "[|", edges_str)
edges_str = re.sub("\]\]", "|]", edges_str)
edges_str = re.sub("\],\[", ",\n             |", edges_str)
dzn_content_str = "n = {};\ndistances = {};".format(sop_content["nodes"], edges_str)

# Write dzn
dzn_filepath = os.path.splitext(sop_filepath)[0] + ".dzn"
dzn_file = open(dzn_filepath, "w")
dzn_file.write(dzn_content_str)
dzn_file.close()
