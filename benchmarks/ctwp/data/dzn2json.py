import os
import re
import sys
import json
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# Dzn grammar
dzn_grammar = Grammar("""
    dzn = ws* k ws* b ws* atomic ws* disjunctive ws* soft ws* direct ws*
    k = "k = " int ";"
    b = "b = " int ";"
    atomic = "AtomicConstraints =  [" matrix_int "];"
    disjunctive = "DisjunctiveConstraints =  [" matrix_int "];"
    soft = "SoftAtomicConstraints =  [" matrix_int "];"
    direct = "DirectSuccessors =  [" matrix_int "];"
    matrix_int = ~"[0-9|,\\s]*"
    int = ~"[0-9]+"
    ws = ~"\\s"  
    """)

# Dzn tree visitor
class DznVisitor(NodeVisitor):
    def visit_dzn(self, node, visited_children):
        k = visited_children[1]
        b = visited_children[3]
        atomics = visited_children[5]
        disjunctive = visited_children[7]
        soft = visited_children[9]
        direct = visited_children[11]
        output = {}
        output.update({"k": k})
        output.update({"b": b})
        output.update({"AtomicConstraints": atomics})
        output.update({"DisjunctiveConstraints": disjunctive})
        output.update({"SoftAtomicConstraints": soft})
        output.update({"DirectSuccessors": direct})
        return output

    def visit_k(self, node, visited_children):
        return visited_children[1]
    
    def visit_b(self, node, visited_children):
        return visited_children[1]
    
    def visit_atomic(self, node, visited_children):
        return visited_children[1]
    
    def visit_disjunctive(self, node, visited_children):
        return visited_children[1]
    
    def visit_soft(self, node, visited_children):
        return visited_children[1]
    
    def visit_direct(self, node, visited_children):
        m = visited_children[1]
        if m:
            m = m[0]
        return m
    
    def visit_matrix_int(self, node, visited_children):
        m = node.text.strip("| ")
        m = re.sub("\s+", " ", m)
        if m:
            m = m.split("|")
            m = [[int(i) for i in r.split(",")] for r in m]
        else:
            m = []
        return m
    
    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return None
    
# Parse dzn
dzn_filepath = sys.argv[1]
dzn_file = open(dzn_filepath, "r")
dzn_tree = dzn_grammar.parse(dzn_file.read())
dzn_file.close()

# Initialize json content
json_content = DznVisitor().visit(dzn_tree)
json_content_str = json.dumps(json_content, sort_keys=False, indent=4)

# Fix array indentation
json_content_str = re.sub("\s{4,}\]", "]", json_content_str)
json_content_str = re.sub("\s{8,}", "", json_content_str)
json_content_str = re.sub("\[\[", "[\n        [", json_content_str)
json_content_str = re.sub("\],\[", "],\n        [", json_content_str)
json_content_str = re.sub("\]\],\n", "]\n    ],\n", json_content_str)

# Write json
json_filepath = os.path.splitext(dzn_filepath)[0] + ".json"
json_file = open(json_filepath, "w")
json_file.write(json_content_str)
json_file.close()
