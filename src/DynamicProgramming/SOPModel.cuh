#pragma once

#include <DecisionDiagram/StateMetadata.cuh>
#include <OptimizationProblems/SOProblem.cuh>
#include <DynamicProgramming/SOPState.cuh>

namespace DynamicProgramming
{
    __host__ __device__ inline DynamicProgramming::CostType calcCost(OptimizationProblems::SOProblem const * problem, SOPState const * currentState, OptimizationProblems::ValueType const value);
    void makeRoot(OptimizationProblems::SOProblem const * problem, SOPState* root);
    __host__ __device__ inline void makeState(OptimizationProblems::SOProblem const * problem, SOPState const * currentState, OptimizationProblems::ValueType value, DynamicProgramming::CostType cost, SOPState* nextState);
}

__host__ __device__
DynamicProgramming::CostType DynamicProgramming::calcCost(OptimizationProblems::SOProblem const * problem, SOPState const * currentState, OptimizationProblems::ValueType const value)
{
    if(not currentState->selectedValues.isEmpty())
    {
        OptimizationProblems::ValueType const from = *currentState->selectedValues.back();
        OptimizationProblems::ValueType const to = value;
        return currentState->cost + problem->getDistance(from, to);
    }
    else
    {
        return 0;
    }
}

void DynamicProgramming::makeRoot(OptimizationProblems::SOProblem const* problem, SOPState* root)
{
    //Initialize cost
    root->cost = 0;

    // Initialize precedences
    thrust::fill(thrust::seq, root->precedencesCount.begin(), root->precedencesCount.end(), 0);
    for (OptimizationProblems::ValueType from = 0; from <= problem->maxValue; from += 1)
    {
        for (OptimizationProblems::ValueType to = 0; to <= problem->maxValue; to += 1)
        {
            DynamicProgramming::CostType const distance = problem->getDistance(from, to);
            if (distance < 0)
            {
                *root->precedencesCount[from] += 1;
            }
        }
    }

    //Initialize admissible values
    root->admissibleValuesMap.insert(0);
}

__host__ __device__
void DynamicProgramming::makeState(OptimizationProblems::SOProblem const * problem, SOPState const * currentState, OptimizationProblems::ValueType value, DynamicProgramming::CostType cost, SOPState* nextState)
{
    // Generic
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->admissibleValuesMap.erase(value);
    nextState->selectValue(value);

    // Update admissible values
    OptimizationProblems::ValueType to = value;
    for (OptimizationProblems::ValueType from = 0; from <= problem->maxValue; from += 1)
    {
        if (*nextState->precedencesCount[from] > 0)
        {
            if (problem->getDistance(from, to) < 0)
            {
                *nextState->precedencesCount[from] -= 1;
                if (*nextState->precedencesCount[from] == 0)
                {
                    nextState->admissibleValuesMap.insert(from);
                }
            }
        }
    }
}
