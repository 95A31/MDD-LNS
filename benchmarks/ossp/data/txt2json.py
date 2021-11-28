import os
import re
import sys
import json
import txt_parser

# Parse txt
txt_filepath = sys.argv[1]
txt_file = open(txt_filepath, "r")
txt_tree = txt_parser.txt_grammar.parse(txt_file.read())
txt_file.close()
txt_content = txt_parser.TxtVisitor().visit(txt_tree)

# Fix array indentation
json_content_str = json.dumps(txt_content, sort_keys=False, indent=4)
json_content_str = re.sub("\s{8,}", "", json_content_str)
json_content_str = re.sub("\[\[", "[\n        [", json_content_str)
json_content_str = re.sub("\],\[", "],\n        [", json_content_str)

# Write json
json_filepath = os.path.splitext(txt_filepath)[0] + ".json"
json_file = open(json_filepath, "w")
json_file.write(json_content_str)
json_file.close()
