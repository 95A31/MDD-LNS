from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# Txt grammar
txt_grammar = Grammar("""
    txt = comment nl int ws+ int ws+ tasks
    comment = "//" ~"."+
    tasks = (ws* int ws*)*    
    int = ~"[0-9]+"
    nl = ~"\\n"
    ws = ~"\\s"  
    """)

# Txt tree visitor
class TxtVisitor(NodeVisitor):
    def visit_txt(self, node, visited_children):
        j = visited_children[2]
        m = visited_children[4]
        t = visited_children[6]
        output = {}
        output.update({"jobs": j})
        output.update({"machines": m})
        output.update({"tasks": t})
        return output

    def visit_tasks(self, node, visited_children):
        t = node.text.strip()
        t = t.split("\n")
        t = [[int(i) for i in r.split()] for r in t]
        return t
    
    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return None
    
