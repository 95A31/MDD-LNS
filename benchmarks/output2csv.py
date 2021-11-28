import os
import json
import sys
import mzn_parser
import mdd_parser


def getAvgBestSolutions(solutions, timeout):
    solutions = [s for s in solutions if s is not None]
    if solutions:
        best_solution_avg_cost = 0
        best_solution_avg_time = 0
        for cost, time in solutions:
            if time < timeout:
                best_solution_avg_cost = best_solution_avg_cost + cost
                best_solution_avg_time = best_solution_avg_time + time
        best_solution_avg_cost = round(best_solution_avg_cost / len(solutions))
        best_solution_avg_time = round(best_solution_avg_time / len(solutions))
        return best_solution_avg_cost, best_solution_avg_time
    else:
        return None

def getTxtFileInfo(txt_filename):
    return txt_filename[0:-4].split("_")


def checkArgs(argv):
    if len(argv) < 4:
        print("[ERROR] Usage: python output2cvs.py <Best knows solution file> <Timeout> <Output file>")
        exit()
    best_know_solutions_filepath = argv[1]
    timeout = int(argv[2])
    output_filepath = argv[3]
    output_format = getTxtFileInfo(output_filepath)[1]
    if output_format != "mdd" and output_format != "mzn":
        print("[ERROR] Unsupported output format: {}".format(output_format))
        exit()
    return best_know_solutions_filepath, timeout, output_format, output_filepath


# Main
best_know_solutions_filepath, timeout, output_format, output_filepath = checkArgs(sys.argv)

# Parse best know solutions
best_know_solutions_file = open(best_know_solutions_filepath, "r")
best_know_solutions = json.load(best_know_solutions_file)
best_know_solutions_file.close()

# Parse output
output_file = open(output_filepath, "r")
if output_format == "mdd":
    output_tree = mdd_parser.mdd_grammar.parse(output_file.read())
    benchmarks = mdd_parser.MddVisitor(timeout).visit(output_tree)
elif output_format == "mzn":
    output_tree = mzn_parser.mzn_grammar.parse(output_file.read())
    benchmarks = mzn_parser.MznVisitor(timeout).visit(output_tree)
output_file.close()

# Compare benchmarks and best know solutions
csv_content_str = "Instance,Cost,Time,Gap\n"
for instance in best_know_solutions.keys():
    print("Comparing value of {}".format(instance))
    best_know_solution = best_know_solutions[instance][1]
    best_solution_avg = None
    if instance in benchmarks.keys():
        s = getAvgBestSolutions(benchmarks[instance], timeout)
        if s:
            cost_avg, time_avg = s
            csv_content_str = csv_content_str + "{},{},{},{:.2f}\n".format(instance, cost_avg, time_avg, cost_avg / best_know_solution)
        else:
            csv_content_str = csv_content_str + "{},,,\n".format(instance)
    else:
        csv_content_str = csv_content_str + "{},,,\n".format(instance)

# Write csv
csv_filepath = os.path.splitext(output_filepath)[0] + "_" + str(timeout) + ".csv"
csv_file = open(csv_filepath, "w")
csv_file.write(csv_content_str)
csv_file.close()
