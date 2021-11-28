#pragma once

#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>
#include <External/Nlohmann/json.hpp>
#include <DynamicProgramming/Context.h>
#include <OptimizationProblems/Problem.cuh>

namespace OptimizationProblems
{
    class SOProblem: public Problem
    {
        // Members
        public:
        Array<DynamicProgramming::CostType> distances;

        // Functions
        public:
        SOProblem(u32 variablesCount, Memory::MallocType mallocType);
        __host__ __device__ inline DynamicProgramming::CostType getDistance(ValueType const from, ValueType const to) const;
    };

    template<>
    OptimizationProblems::SOProblem* parseInstance<OptimizationProblems::SOProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OptimizationProblems::SOProblem::SOProblem(unsigned int variablesCount, Memory::MallocType mallocType) :
    Problem(variablesCount, mallocType),
    distances(variablesCount * variablesCount, mallocType)
{}

__host__ __device__
DynamicProgramming::CostType OptimizationProblems::SOProblem::getDistance(ValueType const from, ValueType const to) const
{
    return *distances[(from * variables.getCapacity()) + to];
}

template<>
OptimizationProblems::SOProblem* OptimizationProblems::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse instance
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    u32 const problemSize = sizeof(OptimizationProblems::SOProblem);
    OptimizationProblems::SOProblem* const problem = reinterpret_cast<OptimizationProblems::SOProblem*>(Memory::safeMalloc(problemSize, mallocType));
    u32 const nodes = problemJson["nodes"];
    new (problem) OptimizationProblems::SOProblem(nodes, mallocType);

    // Init variables
    Variable variable(0,nodes - 1);
    for (u32 variableIdx = 0; variableIdx < nodes; variableIdx += 1)
    {
        problem->add(&variable);
    }

    // Init distances
    for (u32 from = 0; from < nodes; from += 1)
    {
        for (u32 to = 0; to < nodes; to += 1)
        {
            *problem->distances[(from * nodes) + to] = problemJson["edges"][from][to];
        }
    }

    return problem;
}