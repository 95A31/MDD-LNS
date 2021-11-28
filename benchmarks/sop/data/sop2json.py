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
json_content_str = json.dumps(sop_content, sort_keys=False, indent=4)
json_content_str = re.sub("\s{8,}", "", json_content_str)
#json_content_str = re.sub("\[\[", "[\n        [", json_content_str)
#json_content_str = re.sub("\],\[", "],\n        [", json_content_str)

# Write json
json_filepath = os.path.splitext(sop_filepath)[0] + ".json"
json_file = open(json_filepath, "w")
json_file.write(json_content_str)
json_file.close()
