
# MDD-LNS
This repository contains the code and the benchmarks used in the paper *Solutions of Sequencing Problems using MDDs and GPUs*.

# Description
The resolution of combinatorial optimization problems, especially in the area concerned with the sequencing of task, is an important challenge.
This domain covers a wide breadth of applications often expressed as scheduling or routing problems. 
Such problems can often be optimally solved using Constraint Programming techniques, but if a good solution is needed in a short amount of time, Constraint Programming techniques may not be feasible.

We integrate tree techniques within a solver, focused on a time-bounded search for high-quality solutions:
- **Multi-valued Decision Diagrams** whose transitional semantic offers several benefits in terms of modeling and efficiency.
- **Large Neighborhood Search** as an effective local search strategy. 
- **GPGPU** to enhance efficiency in the exploration of the search space.

The solver was evaluated on several classes of benchmarks, with positive outcomes in terms of time and solution quality compared to state-of-the-art constraint-based solvers.

## Requirements
- An NVIDIA GPU with [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) >= 5.2:
- CMake >= 3.18
- GCC >= 9
- CUDA >= 11

## Compilation
Execute the following commands for a release version:
```sh
mkdir "build"
./autoCMake.sh -r ./build
cd ./build
make
```
Use `-d` insted of `-r` for a debug version. 

## Usage
An example of usage is the following:
```sh
./build/mdd-lns -t 60 --wc 1000 --mc 16 --wg 5 --mg 10000 --eq 0.7 --rs 10 ./benchmarks/sop/data/json/kro124p.2.json
[INFO] Input file: ./sop/data/json/kro124p.2.json
[INFO] Timeout: 60
[INFO] CPU: Width 1000 | MDDs 16
[INFO] GPU: Width 5 | MDDs 10000
[INFO] LNS: = 0.700 | Random seed 10
[INFO] Initializing data structures | Time: 00:00:00.000
[INFO] Initializing random engines | Time: 00:00:00.240
[INFO] Start initial search | Time: 00:00:00.242
[SOLUTION] Source: CPU | Time: 00:00:00.842 | Cost: 50154 | Solution: [0,92,27,...,40,38,100]
[SOLUTION] Source: CPU | Time: 00:00:02.642 | Cost: 50039 | Solution: [0,5,62,...,40,38,100]
[SOLUTION] Source: CPU | Time: 00:00:03.542 | Cost: 49073 | Solution: [0,73,71,...,95,38,100]
[INFO] Switch to LNS | Time: 00:00:10.744
[SOLUTION] Source: CPU | Time: 00:00:11.345 | Cost: 49002 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:11.645 | Cost: 47963 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:12.245 | Cost: 47927 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:12.845 | Cost: 47523 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:13.445 | Cost: 47501 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:14.046 | Cost: 47225 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:14.946 | Cost: 46968 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: CPU | Time: 00:00:15.547 | Cost: 46836 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:16.148 | Cost: 46736 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: CPU | Time: 00:00:17.048 | Cost: 46720 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:17.348 | Cost: 46662 | Solution: [0,73,71,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:18.549 | Cost: 46659 | Solution: [0,62,73,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:19.449 | Cost: 46610 | Solution: [0,5,62,...,95,38,100]
[SOLUTION] Source: GPU | Time: 00:00:20.349 | Cost: 46577 | Solution: [0,62,73,...,95,38,100]
```
To know all the avaiable options use `./build/mdd-lns --help`
