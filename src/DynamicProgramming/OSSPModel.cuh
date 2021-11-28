#pragma once

#include <thrust/functional.h>
#include <Utils/Algorithms.cuh>
#include <DecisionDiagram/StateMetadata.cuh>
#include <DynamicProgramming/OSSPState.cuh>
#include <OptimizationProblems/OSSProblem.cuh>

namespace DynamicProgramming
{
    __host__ __device__ inline DynamicProgramming::CostType calcCost(OptimizationProblems::OSSProblem const * problem, OSSPState const * currentState, OptimizationProblems::ValueType const value);
    void makeRoot(OptimizationProblems::OSSProblem const * problem, OSSPState* root);
    __host__ __device__ inline void makeState(OptimizationProblems::OSSProblem const * problem, OSSPState const * currentState, OptimizationProblems::ValueType value, DynamicProgramming::CostType cost, OSSPState* nextState);
}

__host__ __device__
DynamicProgramming::CostType DynamicProgramming::calcCost(OptimizationProblems::OSSProblem const * problem, OSSPState const * currentState, OptimizationProblems::ValueType const value)
{
    // Retrive task, job, machine, duration
    OptimizationProblems::ValueType const task = value;
    OptimizationProblems::ValueType const job = task / problem->machines;
    OptimizationProblems::ValueType const machine = task % problem->machines;
    u16 const duration = *problem->tasks[task];
    u16 const task_start = Algorithms::max(*currentState->machines_makespan[machine], *currentState->jobs_makespan[job]);
    return Algorithms::max(currentState->cost, static_cast<i32>(task_start + duration));
}


void DynamicProgramming::makeRoot(OptimizationProblems::OSSProblem const* problem, OSSPState* root)
{
    //Initialize cost
    root->cost = 0;

    // Initialize jobs
    for (OptimizationProblems::ValueType job = 0; job < problem->jobs; job += 1)
    {
        *root->jobs_makespan[job] = 0;
    }

    // Initialize machines
    for (OptimizationProblems::ValueType machine = 0; machine < problem->machines; machine += 1)
    {
        *root->machines_makespan[machine] = 0;
        *root->machines_progress[machine] = 0;
    }

    //Initialize tasks
    u32 const tasks_count = problem->tasks.getCapacity();
    for(u32 task = 0; task < tasks_count; task += 1)
    {
        *root->tasks_start[task] = 0;
        root->admissibleValuesMap.insert(task);
    }
}

__host__ __device__
void DynamicProgramming::makeState(OptimizationProblems::OSSProblem const * problem, OSSPState const * currentState, OptimizationProblems::ValueType value, DynamicProgramming::CostType cost, OSSPState* nextState)
{
    // Generic
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->admissibleValuesMap.erase(value);
    nextState->selectValue(value);

    // Retrive task, job, machine, duration
    OptimizationProblems::ValueType const task = value;
    OptimizationProblems::ValueType const job = task / problem->machines;
    OptimizationProblems::ValueType const machine = task % problem->machines;
    u16 const duration = *problem->tasks[task];

    // Calculate task start
    u16 const task_start = Algorithms::max(*nextState->machines_makespan[machine], *nextState->jobs_makespan[job]);
    *nextState->tasks_start[task] = task_start;

    // Update makespans
    u16 task_makespan = task_start + duration;
    *nextState->machines_makespan[machine] = task_makespan;
    *nextState->jobs_makespan[job] = task_makespan;
}