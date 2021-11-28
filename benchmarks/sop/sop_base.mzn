% Sequential Ordering Problem
% ===
% This problem is an asymmetric traveling salesman problem with additional precedence constraints. 
% Given a set of n nodes and distances for each pair of nodes, find a Hamiltonian path from node 1
% to node n of minimal length which takes given precedence constraints into account.

include "globals.mzn";

% Problem
int: n;
array[int,int] of int: distances;

% Variables
array[1..n] of var 1..n: path;
array[1..n] of var 1..n: path_indices;

% Path constraint
constraint all_different(path);
constraint inverse(path, path_indices);

% Precedence constraints
constraint forall (i,j in 1..n)(if distances[i,j] < 0 then path_indices[i] > path_indices[j] endif);

% Objcetive
var int: total_distance = sum(i in 1..n-1)(distances[path[i],path[i+1]]);

% Output
output ["Value = \(total_distance)\nSolution = \(path)"];