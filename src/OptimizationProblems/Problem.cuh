#pragma once

#include <Utils/Algorithms.cuh>
#include <Containers/Vector.cuh>
#include <OptimizationProblems/Variable.cuh>

namespace OptimizationProblems
{
    class Problem
    {
        // Members
        public:
        ValueType maxValue;
        u32 maxBranchingFactor;
        Vector<Variable> variables;

        // Functions
        public:
        Problem(u32 variablesCount, Memory::MallocType mallocType);
        void add(Variable const * variable);
    };

    template<typename ProblemType>
    ProblemType* parseInstance(char const * problemFilename, Memory::MallocType mallocType);
}

OptimizationProblems::Problem::Problem(u32 variablesCount, Memory::MallocType mallocType) :
    maxValue(0),
    maxBranchingFactor(0),
    variables(variablesCount, mallocType)
{}

void OptimizationProblems::Problem::add(Variable const * variable)
{
    variables.pushBack(variable);
    maxValue = Algorithms::max(maxValue, variable->maxValue);
    maxBranchingFactor = Algorithms::max(maxBranchingFactor, static_cast<u32>(maxValue + 1));
}



