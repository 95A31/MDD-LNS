from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# Sop grammar
sop_grammar = Grammar("""
    sop = (info nl)* nodes ws+ edges ws* "EOF" ws*
    info = ~"[A-Z]" ~"."+
    edges = (ws* int ws*)*
    nodes = " "? int    
    int = ~"[-+]?" ~"[0-9]+"
    nl = ~"\\n"
    ws = ~"\\s"  
    """)

# Sop tree visitor
class SopVisitor(NodeVisitor):
    def visit_sop(self, node, visited_children):
        n = visited_children[1]
        e = visited_children[3]
        output = {}
        output.update({"nodes": n})
        output.update({"edges": e})
        return output

    def visit_edges(self, node, visited_children):
        t = node.text.strip()
        t = t.split("\n")
        t = [[int(i) for i in r.split()] for r in t]
        return t

    def visit_nodes(self, node, visited_children):
        return int(node.text)

    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return None
    
