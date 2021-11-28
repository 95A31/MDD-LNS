from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# MiniZinc output grammar
mzn_grammar = Grammar("""
    output = result+
    result = instance run solutions statistics?
    instance = "%%%% Instance: " name nl
    name = ~"\\S"+
    run = "%%%% Run: " int " / " int nl
    solutions = (some_solutions / no_solutions)
    some_solutions = solution+ ("==========" nl)?
    no_solutions = ("=====UNKNOWN=====" nl)?
    solution = value elements time "----------" nl
    value = "Value = " int nl
    elements = "Solution = [" int_list "]" nl
    time = "% time elapsed: " float " s" nl 
    int_list = ~"[0-9, ]"+    
    statistics = time? (statistic+)?
    statistic = ("%%%mzn-stat: " / "%%%mzn-stat-end" /  "%% copies: ") ~"\\S"* nl
    int = ~"[-+]?" ~"[0-9]+"
    float = int "." int
    nl = ~"\\n"
    ws = ~"\\s"  
    """)

# MiniZinc output tree visitor
class MznVisitor(NodeVisitor):

    def __init__ (self, timeout):
        self.instance = ""
        self.output = {}
        self.timeout = timeout

    def visit_output(self, node, visited_children):
        return self.output

    def visit_name(self, node, visited_children):
        self.instance = node.text
        if self.instance not in self.output.keys():
            self.output[self.instance] = []

    def visit_solution(self, node, visited_children):
        (cost, time) = (visited_children[0], visited_children[2])
        if time <= self.timeout:
            self.output[self.instance][-1] = (cost, time)

    def visit_run(self, node, visited_children):
        print("Parsing output {} of instance {}".format(visited_children[1], self.instance))
        self.output[self.instance].append(None)

    def visit_value(self, node, visited_children):
        return visited_children[1]

    def visit_time(self, node, visited_children):
        return visited_children[1]

    def visit_float(self, node, visited_children):
        return float(node.text)

    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return None