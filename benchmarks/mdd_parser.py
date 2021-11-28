from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# MDD output grammar
mdd_grammar = Grammar("""
    output = result+
    result = instance run (solution / info)+
    instance = "%%%% Instance: " name nl
    name = ~"\\S"+
    run = "%%%% Run: " int " / " int nl
    solution = "[SOLUTION] " source  " | " time " | " cost ~"."+ nl
    source = "Source: " ("CPU" / "GPU")
    time = "Time: " int ":" int ":" int "." int
    cost = "Cost: " int 
    info = "[INFO] "  ~"."+ nl
    int = ~"[-+]?" ~"[0-9]+"
    float = int "." int
    nl = ~"\\n"
    ws = ~"\\s"  
    """)

# MDD output tree visitor
class MddVisitor(NodeVisitor):

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
        (cost, time) = (visited_children[5], visited_children[3])
        if time <= self.timeout:
            self.output[self.instance][-1] = (cost, time)

    def visit_run(self, node, visited_children):
        print("Parsing output {} of instance {}".format(visited_children[1], self.instance))
        self.output[self.instance].append(None)

    def visit_cost(self, node, visited_children):
        return visited_children[1]

    def visit_time(self, node, visited_children):
        time = visited_children[1] * 60 * 60 + visited_children[3] * 60 + visited_children[5] + visited_children[7] / 1000;
        return time;

    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return None
