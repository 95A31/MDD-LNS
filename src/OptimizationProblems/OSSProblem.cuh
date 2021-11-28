#pragma once

#include <fstream>
#include <Containers/Array.cuh>
#include <Containers/Pair.cuh>
#include <Containers/Triple.cuh>
#include <External/Nlohmann/json.hpp>
#include <Utils/Algorithms.cuh>
#include <DynamicProgramming/Context.h>
#include <OptimizationProblems/Problem.cuh>

namespace OptimizationProblems
{
    class OSSProblem : public Problem
    {
        // Members
        public:
        OptimizationProblems::ValueType const jobs;
        OptimizationProblems::ValueType const machines;
        Array<u16> tasks;

        // Functions
        public:
        OSSProblem(u32 jobs, u32 machines, Memory::MallocType mallocType);
    };

    template<>
    OptimizationProblems::OSSProblem* parseInstance<OSSProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OptimizationProblems::OSSProblem::OSSProblem(u32 jobs, u32 machines, Memory::MallocType mallocType) :
    Problem(jobs * machines, mallocType),
    jobs(jobs),
    machines(machines),
    tasks(jobs * machines, mallocType)
{}

template<>
OptimizationProblems::OSSProblem* OptimizationProblems::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse json
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    u32 const problemSize = sizeof(OptimizationProblems::OSSProblem);
    OptimizationProblems::OSSProblem* const problem = reinterpret_cast<OptimizationProblems::OSSProblem*>(Memory::safeMalloc(problemSize, mallocType));
    OptimizationProblems::ValueType const jobs = problemJson["jobs"];
    OptimizationProblems::ValueType const machines = problemJson["machines"];
    new (problem) OptimizationProblems::OSSProblem(jobs, machines, mallocType);

    // Init variables
    Variable const variable(0, jobs * machines - 1);
    for (OptimizationProblems::ValueType variableIdx = 0; variableIdx < jobs * machines; variableIdx += 1)
    {
        problem->add(&variable);
    }

    // Init tasks
    for (OptimizationProblems::ValueType job = 0; job < jobs; job += 1)
    {
        for (OptimizationProblems::ValueType machine = 0; machine < machines; machine += 1)
        {
            u16 const task = (job * machines) + machine;
            *problem->tasks[task] = problemJson["tasks"][job][machine]; // Duration
        }
    }
    return problem;
}